"""Insider open-market trades (SEC Form 4) from the SEC's quarterly "Insider Transactions Data Sets".

One zip per quarter (2006+) instead of one request per filing. Kept: Form 4, transaction codes
P (open-market purchase) and S (open-market sale), for universe tickers. Point-in-time key is the
filing date (the data sets have no acceptance time) -> features use the next trading day.

Why: clustered open-market buying by officers/directors is one of the longest-documented free signals;
sales are noisier (10b5-1 plans, diversification) but kept for a net measure.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from ..config import store_dir
from ..db import connect, upsert
from ..utils import http_get, log
from .sec import _hdr, _lim

SCHEMA = """create table if not exists insider_trades(
  accession varchar, code varchar, ticker varchar, filing_date date, trans_date date, owner_cik varchar,
  officer boolean, director boolean, shares double, value double, primary key(accession, code, owner_cik))"""

URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"


def _quarters(start_year: int) -> list[str]:
    now = pd.Timestamp.now()
    out = []
    for y in range(start_year, now.year + 1):
        for q in range(1, 5):
            if pd.Timestamp(year=y, month=3 * q, day=1) <= now:
                out.append(f"{y}q{q}")
    return out


def _read(z: zipfile.ZipFile, name: str, cols: list[str]) -> pd.DataFrame:
    with z.open(name) as f:
        return pd.read_csv(f, sep="\t", dtype=str, usecols=lambda c: c in cols, on_bad_lines="skip",
                           quoting=3, encoding="utf-8", encoding_errors="replace")


def _parse_quarter(blob: bytes, universe: set[str]) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(blob))
    sub = _read(z, "SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERTRADINGSYMBOL"])
    sub = sub[sub["DOCUMENT_TYPE"].isin(["4", "4/A"])]
    sub["ticker"] = sub["ISSUERTRADINGSYMBOL"].fillna("").str.upper().str.strip().str.replace("-", ".", regex=False)
    sub = sub[sub["ticker"].isin(universe)]
    if sub.empty:
        return pd.DataFrame()
    tr = _read(z, "NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                                         "TRANS_PRICEPERSHARE"])
    tr = tr[tr["TRANS_CODE"].isin(["P", "S"]) & tr["ACCESSION_NUMBER"].isin(set(sub["ACCESSION_NUMBER"]))]
    if tr.empty:
        return pd.DataFrame()
    tr["shares"] = pd.to_numeric(tr["TRANS_SHARES"], errors="coerce")
    tr["value"] = tr["shares"] * pd.to_numeric(tr["TRANS_PRICEPERSHARE"], errors="coerce")
    tr["trans_date"] = pd.to_datetime(tr["TRANS_DATE"], format="%d-%b-%Y", errors="coerce")
    agg = tr.groupby(["ACCESSION_NUMBER", "TRANS_CODE"]).agg(shares=("shares", "sum"), value=("value", "sum"),
                                                             trans_date=("trans_date", "min")).reset_index()
    own = _read(z, "REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"])
    own = own[own["ACCESSION_NUMBER"].isin(set(agg["ACCESSION_NUMBER"]))].drop_duplicates("ACCESSION_NUMBER")
    rel = own["RPTOWNER_RELATIONSHIP"].fillna("")
    own = own.assign(officer=rel.str.contains("Officer", case=False), director=rel.str.contains("Director", case=False))
    df = agg.merge(sub[["ACCESSION_NUMBER", "FILING_DATE", "ticker"]], on="ACCESSION_NUMBER") \
            .merge(own[["ACCESSION_NUMBER", "RPTOWNERCIK", "officer", "director"]], on="ACCESSION_NUMBER", how="left")
    return pd.DataFrame({
        "accession": df["ACCESSION_NUMBER"], "code": df["TRANS_CODE"], "ticker": df["ticker"],
        "filing_date": pd.to_datetime(df["FILING_DATE"], format="%d-%b-%Y", errors="coerce").dt.date,
        "trans_date": df["trans_date"].dt.date, "owner_cik": df["RPTOWNERCIK"].fillna(""),
        "officer": df["officer"].fillna(False).astype(bool), "director": df["director"].fillna(False).astype(bool),
        "shares": df["shares"], "value": df["value"]}).dropna(subset=["filing_date"])


def collect_insider(start_year: int = 2014) -> int:
    cache = Path(store_dir("raw")) / "insider"
    cache.mkdir(parents=True, exist_ok=True)
    with connect() as con:
        con.execute(SCHEMA)
        universe = {r[0] for r in con.execute("select distinct ticker from universe").fetchall()}
    total = 0
    for q in _quarters(start_year):
        f = cache / f"{q}_form345.zip"
        if not f.exists():
            try:
                r = http_get(URL.format(q=q), headers=_hdr(), limiter=_lim, timeout=300)
            except Exception as e:  # noqa: BLE001 - the newest quarter is published ~a month after it ends
                log.info("insider %s not available: %s", q, str(e)[:80])
                continue
            f.write_bytes(r.content)
        try:
            df = _parse_quarter(f.read_bytes(), universe)
        except zipfile.BadZipFile:
            f.unlink(missing_ok=True)
            continue
        if len(df):
            df = df.drop_duplicates(["accession", "code", "owner_cik"])
            with connect() as con:
                upsert(con, "insider_trades", df)
            total += len(df)
        log.info("insider %s: %d trades (total %d)", q, len(df), total)
    return total

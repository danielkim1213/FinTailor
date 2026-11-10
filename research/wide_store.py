"""All-US-stocks research store (store_wide), kept apart from the production store.

Who: every issuer in the SEC Form 3/4/5 data sets with at least 20 filings, including companies that later disappeared,
so the list is not limited to today's survivors. Prices: Alpaca SIP daily bars (2016+, carries delisted symbols).
A symbol can be reused by a later company, so bars are kept only around the dates that issuer was actually filing.
Everything is resumable.   python research/wide_store.py prep prices liquid fundamentals
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIDE_CFG = ROOT / "research" / "config_wide.toml"
WIDE_CFG.write_text((ROOT / "config.toml").read_text(encoding="utf-8").replace("min_dollar_volume = 20000000", "min_dollar_volume = 3000000"), encoding="utf-8")
os.environ.update(FINTAILOR_CONFIG=str(WIDE_CFG), FINTAILOR_STORE="store_wide", ALPACA_FEED="sip")
sys.path.insert(0, str(ROOT))

import glob  # noqa: E402
import zipfile  # noqa: E402

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402

from fintailor.config import store_dir  # noqa: E402
from fintailor.data.prices import COLS, _alpaca_download  # noqa: E402
from fintailor.data.sec import collect_fundamentals  # noqa: E402
from fintailor.db import connect, upsert  # noqa: E402
from fintailor.utils import log  # noqa: E402

MAIN = ROOT / "store" / "fintailor.duckdb"
WIDE = ROOT / "store_wide"
MIN_FILINGS = 20
START = "2016-01-01"
ACTIVE_AFTER = pd.Timestamp("2025-10-01")
MIN_DVOL = 3_000_000
BATCH = 300


def issuers() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(str(ROOT / "store/raw/insider/*_form345.zip"))):
        with zipfile.ZipFile(path) as z, z.open("SUBMISSION.tsv") as f:
            frames.append(pd.read_csv(f, sep="\t", usecols=["FILING_DATE", "ISSUERCIK", "ISSUERNAME", "ISSUERTRADINGSYMBOL"], dtype=str))
    s = pd.concat(frames, ignore_index=True).dropna(subset=["ISSUERTRADINGSYMBOL", "ISSUERCIK"])
    s["ticker"] = s["ISSUERTRADINGSYMBOL"].str.upper().str.strip().str.replace("-", ".", regex=False)
    s["filed"] = pd.to_datetime(s["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
    s = s[s["ticker"].str.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?") & s["filed"].notna()]
    g = s.groupby(["ticker", "ISSUERCIK"]).agg(first=("filed", "min"), last=("filed", "max"), n=("filed", "size"),
                                               name=("ISSUERNAME", "last")).reset_index()
    g = g[g["n"] >= MIN_FILINGS].sort_values("n", ascending=False).drop_duplicates("ticker")  # reused symbol: the issuer with most filings
    g["active"] = g["last"] >= ACTIVE_AFTER
    g["cik"] = g["ISSUERCIK"].astype(int)
    return g[["ticker", "cik", "name", "first", "last", "n", "active"]].reset_index(drop=True)


def prep() -> None:
    g = issuers()
    WIDE.mkdir(exist_ok=True)
    g.to_parquet(WIDE / "issuers.parquet", index=False)
    with connect() as con:
        con.execute(f"attach '{MAIN}' as main_db (read_only)")
        etf = con.execute("select * from main_db.universe where is_etf or sector = 'benchmark'").df()
        for table in ("macro", "world_daily", "events", "sector_views"):
            con.execute(f"delete from {table}")
            con.execute(f"insert into {table} select * from main_db.{table}")
        con.execute("delete from prices_daily where source <> 'alpaca'")
        names = "','".join(etf["ticker"])
        con.execute(f"insert into prices_daily select * from main_db.prices_daily where ticker in ('{names}')")
        con.execute("detach main_db")
        u = pd.DataFrame({"ticker": g["ticker"], "name": g["name"], "sector": "unknown", "start_date": g["first"].dt.date,
                          "end_date": [None if a else (d + pd.Timedelta(days=400)).date() for a, d in zip(g["active"], g["last"])],
                          "is_etf": False, "source": "form345"})
        u = u[~u["ticker"].isin(set(etf["ticker"]))]
        all_rows = pd.concat([etf[u.columns], u], ignore_index=True)  # universe has no primary key, so replace it whole
        con.execute("delete from universe")
        con.execute("insert into universe by name select * from all_rows")
    log.info("prep: %d issuers (%d active), %d ETFs/benchmark", len(g), int(g["active"].sum()), len(etf))


def prices() -> None:
    g = pd.read_parquet(WIDE / "issuers.parquet").set_index("ticker")
    done_file = WIDE / "alpaca_done.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    todo = [t for t in g.index if t not in done]
    log.info("prices: %d to fetch", len(todo))
    for i in range(0, len(todo), BATCH):
        names = todo[i:i + BATCH]
        try:
            df = _alpaca_download(names, START)
        except Exception as e:  # noqa: BLE001
            log.warning("batch %d failed: %s", i, e)
            continue
        if len(df):
            df["d"] = pd.to_datetime(df["date"])
            lo, hi = df["ticker"].map(g["first"]) - pd.Timedelta(days=45), df["ticker"].map(g["last"]) + pd.Timedelta(days=400)
            df = df[(df["d"] >= lo) & ((df["d"] <= hi) | df["ticker"].map(g["active"]))][COLS]
            with connect() as con:
                upsert(con, "prices_daily", df)
        with open(done_file, "a") as f:
            f.write(" ".join(names) + " ")
        log.info("prices: %d/%d tickers, %d bars in last batch", i + len(names), len(todo), len(df))


def liquid() -> None:
    """Drop issuers that never traded enough to be investable, which keeps the panel a workable size."""
    with connect() as con:
        before = con.execute("select count(distinct ticker) from prices_daily where source='alpaca'").fetchone()[0]
        con.execute(f"""create or replace temp table keep_t as
            select ticker from (select ticker, avg(raw_close * volume) over (partition by ticker order by date rows 19 preceding) dv, raw_close
                                from prices_daily where source='alpaca')
            where dv >= {MIN_DVOL} and raw_close >= 5 group by 1""")
        con.execute("delete from prices_daily where source='alpaca' and ticker not in (select ticker from keep_t)")
        con.execute("delete from universe where source='form345' and ticker not in (select ticker from keep_t)")
        after = con.execute("select count(distinct ticker) from prices_daily where source='alpaca'").fetchone()[0]
        n, lo, hi = con.execute("select count(*), min(date), max(date) from prices_daily where source='alpaca'").fetchone()
    log.info("liquid: %d -> %d tickers; %s bars, %s..%s", before, after, f"{n:,}", lo, hi)


def fundamentals() -> None:
    g = pd.read_parquet(WIDE / "issuers.parquet")
    done_file = WIDE / "fund_done.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    with connect(read_only=True) as con:
        names = [r[0] for r in con.execute("select distinct ticker from universe where source='form345'").fetchall()]
    todo = [t for t in names if t not in done]
    ciks = dict(zip(g["ticker"], g["cik"]))
    log.info("fundamentals: %d to fetch", len(todo))
    for i in range(0, len(todo), 200):
        chunk = todo[i:i + 200]
        try:
            collect_fundamentals(chunk, ciks=ciks)
        except Exception as e:  # noqa: BLE001
            log.warning("fundamentals chunk %d failed: %s", i, e)
            continue
        with open(done_file, "a") as f:
            f.write(" ".join(chunk) + " ")
        log.info("fundamentals: %d/%d", i + len(chunk), len(todo))


if __name__ == "__main__":
    for step in sys.argv[1:]:
        log.info("=== %s", step)
        {"prep": prep, "prices": prices, "liquid": liquid, "fundamentals": fundamentals}[step]()

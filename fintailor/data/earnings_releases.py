"""Earnings press releases from SEC 8-K filings (Item 2.02, exhibit EX-99.1): a long, point-in-time
"news" history (2015+) with exact acceptance timestamps.

The full release is mostly tables, non-GAAP reconciliations and legal boilerplate. The numbers are
already in the earnings table, so only the qualitative core is kept for the LLM: the opening summary,
management quotes and the outlook / guidance section (~1k tokens instead of ~10k).

    from fintailor.data.earnings_releases import collect_releases
    collect_releases(since="2015-01-01")          # resumable; skips accessions already stored
"""
from __future__ import annotations

import re

import pandas as pd

from ..db import connect, upsert
from ..utils import http_get, log
from .sec import _hdr, _lim, _tickers, cik_map

SCHEMA = """create table if not exists earnings_releases(
  accession varchar primary key, ticker varchar, cik varchar, filing_date date, accepted_at timestamp,
  doc_chars integer, excerpt varchar)"""

DROP_HEADINGS = re.compile(
    r"^(forward[- ]looking statements?|safe harbor|cautionary (statement|note)|about [A-Z][\w&.,' -]{1,60}$|"
    r"conference call|webcast|non-gaap (financial )?measures?|use of non-gaap|reconciliation|"
    r"investor (relations )?contacts?|media contacts?|contacts?:?$|source:)", re.I)
OUTLOOK = re.compile(r"\b(outlook|guidance|expects?|expected|anticipates?|forecast|full[- ]year|fiscal (year )?20\d\d|"
                     r"next quarter|raises?|raised|lowers?|lowered|reaffirm|reiterat|updat\w* (its|the) )", re.I)
QUOTE = re.compile(r"[\"“].{40,}[\"”]|\b(said|commented|stated)\b", re.I)


def _filings_202(cik: int, since: str) -> pd.DataFrame:
    hdr = _hdr()
    js = http_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", headers=hdr, limiter=_lim).json()
    pages = [js.get("filings", {}).get("recent", {})]
    recent_min = min(pages[0].get("filingDate", ["9999"]) or ["9999"])
    if recent_min > since:  # older filings live in extra pages
        for f in js.get("filings", {}).get("files", []):
            if f.get("filingTo", "9999") >= since:
                pages.append(http_get(f"https://data.sec.gov/submissions/{f['name']}", headers=hdr, limiter=_lim).json())
    cols = ("accessionNumber", "form", "filingDate", "acceptanceDateTime", "items")
    df = pd.concat([pd.DataFrame({k: p.get(k, []) for k in cols}) for p in pages], ignore_index=True)
    df = df[df["form"].isin(["8-K"]) & df["items"].fillna("").str.contains(r"\b2\.02\b") & (df["filingDate"] >= since)]
    return df.drop_duplicates("accessionNumber")


def _release_text(cik: int, accession: str) -> str:
    """Full submission .txt -> the EX-99.1 (else first EX-99*) document as plain text."""
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{accession}.txt"
    raw = http_get(url, headers=_hdr(), limiter=_lim, timeout=60).text
    docs = re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", raw, flags=re.S | re.I)
    best = None
    for d in docs:
        m = re.search(r"<TYPE>\s*([^\s<]+)", d, flags=re.I)
        typ = (m.group(1) if m else "").upper()
        if typ.startswith("EX-99"):
            if typ in ("EX-99.1", "EX-99.01", "EX-99"):
                best = d
                break
            best = best or d
    if best is None:
        return ""
    body = re.search(r"<TEXT>(.*)</TEXT>", best, flags=re.S | re.I)
    return html_to_text(body.group(1) if body else best)


def html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup
    if "<" not in html[:2000]:
        return html
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["table", "script", "style"]):  # financial tables: numbers are in the earnings table already
        t.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def excerpt(text: str, max_chars: int = 3000) -> str:
    """Opening summary + management quotes + outlook/guidance paragraphs, boilerplate removed."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 30]
    keep, dropping = [], False
    for p in paras:
        first_line = p.split("\n", 1)[0].strip()
        if DROP_HEADINGS.match(first_line) or (len(first_line) < 60 and DROP_HEADINGS.match(first_line.rstrip(":"))):
            dropping = True
            continue
        if dropping and len(first_line) < 60 and first_line.isupper():  # a new section heading ends the drop
            dropping = False
        if dropping:
            continue
        keep.append(re.sub(r"\s+", " ", p))
    head = keep[:4]
    rest = keep[4:]
    outlook = [p for p in rest if OUTLOOK.search(p)]
    quotes = [p for p in rest if QUOTE.search(p) and p not in outlook][:2]
    out, n = [], 0
    for p in head + outlook + quotes:
        if n + len(p) > max_chars:
            p = p[: max(0, max_chars - n)]
        if p:
            out.append(p)
            n += len(p)
        if n >= max_chars:
            break
    return "\n\n".join(out)


def collect_releases(tickers: list[str] | None = None, since: str = "2015-01-01", max_filings: int | None = None) -> int:
    with connect() as con:
        con.execute(SCHEMA)
        done = {r[0] for r in con.execute("select accession from earnings_releases").fetchall()}
        tickers = tickers or _tickers(con)
    cmap = cik_map()
    rows, n_new = [], 0

    def flush():
        nonlocal rows
        if rows:
            with connect() as con:
                upsert(con, "earnings_releases", pd.DataFrame(rows))
            rows = []

    for i, t in enumerate(tickers):
        cik = cmap.get(t)
        if not cik:
            continue
        try:
            fl = _filings_202(cik, since)
        except Exception as e:  # noqa: BLE001
            log.warning("8-K list %s: %s", t, e)
            continue
        for r in fl.itertuples():
            if r.accessionNumber in done:
                continue
            try:
                txt = _release_text(cik, r.accessionNumber)
            except Exception as e:  # noqa: BLE001
                log.warning("8-K %s %s: %s", t, r.accessionNumber, e)
                continue
            acc = pd.to_datetime(r.acceptanceDateTime, errors="coerce", utc=True)
            rows.append({"accession": r.accessionNumber, "ticker": t, "cik": str(cik),
                         "filing_date": pd.Timestamp(r.filingDate).date(),
                         "accepted_at": acc.tz_convert("UTC").tz_localize(None) if pd.notna(acc) else None,
                         "doc_chars": len(txt), "excerpt": excerpt(txt) if txt else ""})
            done.add(r.accessionNumber)
            n_new += 1
            if len(rows) >= 200:
                flush()
            if max_filings and n_new >= max_filings:
                flush()
                return n_new
        if i % 25 == 0:
            log.info("earnings releases: %d/%d tickers, %d new filings", i, len(tickers), n_new)
    flush()
    log.info("earnings releases: %d new filings", n_new)
    return n_new

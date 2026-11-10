"""Short interest (FINRA consolidated, twice a month, 2017-12+) via the free FINRA API.

Point in time: positions are as of the settlement date but published ~7 business days later; features
use them only from settlement + 10 business days (conservative).
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from ..db import connect, upsert
from ..utils import log

SCHEMA = """create table if not exists short_interest(
  ticker varchar, settlement_date date, short_qty double, avg_daily_volume double, days_to_cover double,
  primary key(ticker, settlement_date))"""
API = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
PUBLISH_LAG_BDAYS = 10


def _fetch(symbol: str) -> list[dict]:
    rows, off = [], 0
    while True:
        body = {"limit": 1000, "offset": off, "compareFilters": [
            {"compareType": "equal", "fieldName": "symbolCode", "fieldValue": symbol}],
            "fields": ["settlementDate", "currentShortPositionQuantity", "averageDailyVolumeQuantity",
                       "daysToCoverQuantity"]}
        for attempt in range(4):
            r = requests.post(API, json=body, headers={"Accept": "application/json"}, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            break
        if not r.ok:
            raise RuntimeError(f"FINRA {symbol}: HTTP {r.status_code} {r.text[:120]}")
        j = r.json() if r.text.strip() else []
        rows += j
        if len(j) < 1000:
            return rows
        off += 1000


def collect_short_interest(tickers: list[str] | None = None) -> int:
    with connect() as con:
        con.execute(SCHEMA)
        if tickers is None:
            tickers = [r[0] for r in con.execute("select distinct ticker from universe where not is_etf").fetchall()]
    frames, failed = [], 0
    for i, t in enumerate(tickers):
        try:
            rows = _fetch(t)  # FINRA uses dots for share classes (BRK.B), like our tickers
        except Exception as e:  # noqa: BLE001
            failed += 1
            log.debug("short interest %s: %s", t, e)
            continue
        if rows:
            d = pd.DataFrame(rows)
            frames.append(pd.DataFrame({
                "ticker": t, "settlement_date": pd.to_datetime(d["settlementDate"]).dt.date,
                "short_qty": pd.to_numeric(d.get("currentShortPositionQuantity"), errors="coerce"),
                "avg_daily_volume": pd.to_numeric(d.get("averageDailyVolumeQuantity"), errors="coerce"),
                "days_to_cover": pd.to_numeric(d.get("daysToCoverQuantity"), errors="coerce")}))
        if i % 100 == 0:
            log.info("short interest: %d/%d tickers (%d failed)", i, len(tickers), failed)
        time.sleep(0.2)
    if not frames:
        return 0
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["ticker", "settlement_date"])
    with connect() as con:
        n = upsert(con, "short_interest", df)
    log.info("short interest: %d rows, %d tickers, %d failed", n, df["ticker"].nunique(), failed)
    return n

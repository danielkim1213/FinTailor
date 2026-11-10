"""Earnings dates + EPS surprise. Finnhub calendar (ongoing) and yfinance (history backfill)."""
from __future__ import annotations

import pandas as pd

from ..db import connect, upsert
from ..utils import ET, log, to_yahoo
from .news import _finnhub


def collect_finnhub_calendar(days_back: int = 30, days_fwd: int = 60) -> int:
    today = pd.Timestamp.today().normalize()
    try:
        js = _finnhub("calendar/earnings", **{"from": str((today - pd.Timedelta(days=days_back)).date()),
                                              "to": str((today + pd.Timedelta(days=days_fwd)).date())})
    except Exception as e:  # noqa: BLE001
        log.warning("finnhub earnings: %s", e)
        return 0
    cal = pd.DataFrame(js.get("earningsCalendar", []))
    if cal.empty:
        return 0
    d = pd.to_datetime(cal["date"])
    # bmo -> 08:00 ET, amc -> 16:30 ET, unknown -> 16:30 (conservative)
    hrs = cal.get("hour", pd.Series("", index=cal.index)).map({"bmo": 8, "amc": 16.5, "dmh": 12}).fillna(16.5)
    ann = (d + pd.to_timedelta(hrs, unit="h")).dt.tz_localize(ET).dt.tz_convert("UTC").dt.tz_localize(None)
    df = pd.DataFrame({"ticker": cal["symbol"].str.replace("-", "."), "date": d.dt.date, "announced_at": ann,
                       "eps_actual": pd.to_numeric(cal.get("epsActual"), errors="coerce"),
                       "eps_estimate": pd.to_numeric(cal.get("epsEstimate"), errors="coerce"), "source": "finnhub"})
    with connect() as con:
        return upsert(con, "earnings", df)


def collect_yf_history(tickers: list[str] | None = None, limit: int = 60) -> int:
    import yfinance as yf

    with connect(read_only=True) as con:
        if tickers is None:
            tickers = con.execute("select distinct ticker from universe where not is_etf and end_date is null").df()["ticker"].tolist()
    frames = []
    for i, t in enumerate(tickers):
        try:
            ed = yf.Ticker(to_yahoo(t)).get_earnings_dates(limit=limit)
        except Exception as e:  # noqa: BLE001
            log.debug("yf earnings %s: %s", t, e)
            continue
        if ed is None or ed.empty:
            continue
        ts = pd.DatetimeIndex(ed.index)
        ts = ts.tz_convert("UTC").tz_localize(None) if ts.tz is not None else ts
        frames.append(pd.DataFrame({
            "ticker": t, "date": ts.tz_localize("UTC").tz_convert(ET).date, "announced_at": ts,
            "eps_actual": pd.to_numeric(ed.get("Reported EPS"), errors="coerce").values,
            "eps_estimate": pd.to_numeric(ed.get("EPS Estimate"), errors="coerce").values, "source": "yfinance"}))
        if i % 50 == 0:
            log.info("yf earnings: %d/%d", i, len(tickers))
    if not frames:
        return 0
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["ticker", "date"])
    with connect() as con:
        n = upsert(con, "earnings", df)
    log.info("earnings history: %d rows", n)
    return n


def next_earnings_yf(tickers: list[str]) -> dict[str, pd.Timestamp | None]:
    """Next scheduled earnings date for a few candidate tickers (used by the signal filter)."""
    import yfinance as yf

    out = {}
    for t in tickers:
        try:
            cal = yf.Ticker(to_yahoo(t)).calendar
            dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
            out[t] = pd.Timestamp(dates[0]) if dates else None
        except Exception:  # noqa: BLE001
            out[t] = None
    return out

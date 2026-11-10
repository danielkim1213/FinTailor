"""Daily bars. Columns stored:
  close     = split-adjusted close
  adj_close = split + dividend adjusted close (returns are computed from this)
  raw_close = unadjusted close (for price filters and P/E style ratios)
  open/high/low are split-adjusted like `close`; vwap if the source provides it.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..config import env, get_config
from ..db import connect, replace_where, upsert
from ..utils import RateLimiter, from_yahoo, http_get, log, to_yahoo

COLS = ["ticker", "date", "open", "high", "low", "close", "adj_close", "raw_close", "volume", "vwap", "source"]


# ----------------------------------------------------------------------------- yfinance
def _yf_download(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    ymap = {to_yahoo(t): t for t in tickers}
    raw = yf.download(list(ymap), start=start, end=end, auto_adjust=False, actions=True, group_by="column",
                      threads=True, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=COLS)
    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product([raw.columns, [list(ymap)[0]]])
    long = raw.stack(level=1).reset_index()
    long.columns = [str(c) for c in long.columns]
    long = long.rename(columns={"Date": "date", "Ticker": "ticker", "level_1": "ticker", "Open": "open", "High": "high",
                                "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
                                "Stock Splits": "splits"})
    long = long.dropna(subset=["close"])
    long["ticker"] = long["ticker"].map(lambda t: ymap.get(t, from_yahoo(t)))
    long = long.sort_values(["ticker", "date"])
    if "splits" in long:
        s = long["splits"].fillna(0).replace(0, 1.0).astype(float)
        # raw_close[t] = close[t] * prod(split ratios strictly after t)
        rev_cum = s.iloc[::-1].groupby(long["ticker"].iloc[::-1]).cumprod().iloc[::-1]
        long["raw_close"] = long["close"] * (rev_cum / s)
    else:
        long["raw_close"] = long["close"]
    long["vwap"] = np.nan
    long["source"] = "yfinance"
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None).dt.date
    return long[COLS]


# ----------------------------------------------------------------------------- Alpaca
def _alpaca_bars(tickers: list[str], start: str, end: str | None, adjustment: str) -> pd.DataFrame:
    key, sec = env("ALPACA_API_KEY"), env("ALPACA_SECRET_KEY")
    if not key or not sec:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set")
    hdr = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
    lim = RateLimiter(180)
    out = []
    for i in range(0, len(tickers), 100):
        params = {"symbols": ",".join(tickers[i:i + 100]), "timeframe": "1Day", "start": start, "limit": 10000,
                  "adjustment": adjustment, "feed": env("ALPACA_FEED", "iex")}
        if end:
            params["end"] = end
        while True:
            r = http_get("https://data.alpaca.markets/v2/stocks/bars", params=params, headers=hdr, limiter=lim).json()
            for sym, bars in (r.get("bars") or {}).items():
                for b in bars:
                    out.append((sym, b["t"][:10], b["o"], b["h"], b["l"], b["c"], b["v"], b.get("vw")))
            if not r.get("next_page_token"):
                break
            params["page_token"] = r["next_page_token"]
    return pd.DataFrame(out, columns=["ticker", "date", "open", "high", "low", "close", "volume", "vwap"])


def _alpaca_download(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    split = _alpaca_bars(tickers, start, end, "split")
    allad = _alpaca_bars(tickers, start, end, "all")[["ticker", "date", "close"]].rename(columns={"close": "adj_close"})
    raw = _alpaca_bars(tickers, start, end, "raw")[["ticker", "date", "close"]].rename(columns={"close": "raw_close"})
    df = split.merge(allad, on=["ticker", "date"], how="left").merge(raw, on=["ticker", "date"], how="left")
    df["source"] = "alpaca"
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[COLS]


# ----------------------------------------------------------------------------- Tiingo
def _tiingo_download(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    tok = env("TIINGO_API_KEY")
    if not tok:
        raise RuntimeError("TIINGO_API_KEY not set")
    lim = RateLimiter(50)
    frames = []
    for t in tickers:
        try:
            r = http_get(f"https://api.tiingo.com/tiingo/daily/{t.replace('.', '-')}/prices",
                         params={"startDate": start, "endDate": end, "token": tok}, limiter=lim).json()
        except Exception as e:  # noqa: BLE001
            log.warning("tiingo %s: %s", t, e)
            continue
        if not r:
            continue
        d = pd.DataFrame(r)
        f = d["adjClose"] / d["close"]
        # Tiingo: close is raw; split-only adjustment = raw / cumulative future split factor
        split_f = d.get("splitFactor", pd.Series(1.0, index=d.index)).replace(0, 1.0)
        rev = split_f.iloc[::-1].cumprod().iloc[::-1] / split_f
        frames.append(pd.DataFrame({
            "ticker": t, "date": pd.to_datetime(d["date"]).dt.tz_localize(None).dt.date,
            "open": d["open"] / rev, "high": d["high"] / rev, "low": d["low"] / rev, "close": d["close"] / rev,
            "adj_close": d["adjClose"], "raw_close": d["close"], "volume": d["volume"] * rev, "vwap": np.nan,
            "source": "tiingo"}))
        del f
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)


SOURCES = {"yfinance": _yf_download, "alpaca": _alpaca_download, "tiingo": _tiingo_download}


def backfill_tiingo_paced(tickers: list[str] | None = None, start: str | None = None, per_hour: int = 45) -> dict:
    """Fill tickers that have no prices (mostly delisted names yfinance dropped) from Tiingo's free tier:
    50 requests/hour, 1,000/day, 500 unique symbols/month. Paced under the hourly cap, saves each ticker
    as soon as it arrives, and remembers empty answers, so it can be stopped and re-run at any time."""
    import json as _json

    from ..db import get_state, set_state
    tok = env("TIINGO_API_KEY")
    if not tok:
        raise RuntimeError("TIINGO_API_KEY not set")
    cfg = get_config()
    start = start or str(pd.Timestamp(cfg["universe"]["start_date"]) - pd.Timedelta(days=400))[:10]
    with connect(read_only=True) as con:
        if tickers is None:
            tickers = [r[0] for r in con.execute(
                "select distinct ticker from universe u where not exists "
                "(select 1 from prices_daily p where p.ticker = u.ticker) order by 1").fetchall()]
        empty = set(_json.loads(get_state(con, "tiingo_empty") or "[]"))
    todo = [t for t in tickers if t not in empty]
    gap = 3600.0 / per_hour
    got, nodata, failed = 0, 0, []
    log.info("tiingo paced: %d tickers to try (%d known empty), ~%.0f min", len(todo), len(empty), len(todo) * gap / 60)
    for i, t in enumerate(todo):
        t0 = time.monotonic()
        while True:
            try:
                r = _session_get_tiingo(t, start, tok)
            except Exception as e:  # noqa: BLE001
                log.warning("tiingo %s: %s", t, e)
                failed.append(t)
                r = None
                break
            if r is not None and r.status_code == 429:
                log.info("tiingo hourly cap hit at %s; sleeping 15 min", t)
                time.sleep(900)
                continue
            break
        rows = []
        if r is not None and r.ok:
            js = r.json()
            rows = js if isinstance(js, list) else []
        def save(write):  # another process may hold the DB (panel build, nightly): wait and retry
            for attempt in range(10):
                try:
                    with connect() as con:
                        return write(con)
                except Exception as e:  # noqa: BLE001
                    log.warning("tiingo %s: DB busy (%s), retry in 30s", t, str(e)[:80])
                    time.sleep(30)
            failed.append(t)

        if rows:
            df = _tiingo_frame(t, rows)

            def write(con, df=df):
                d = _validity_filter(df, con)
                if len(d):
                    replace_where(con, "prices_daily", d, "ticker = ?", [t])
                    return True
                empty.add(t)
                set_state(con, "tiingo_empty", _json.dumps(sorted(empty)))
                return False
            ok = save(write)
            got += ok is True
            nodata += ok is False
        elif r is not None and r.ok:
            nodata += 1
            empty.add(t)
            save(lambda con: set_state(con, "tiingo_empty", _json.dumps(sorted(empty))))
        if i % 10 == 0:
            log.info("tiingo paced: %d/%d (saved %d, empty %d, failed %d)", i + 1, len(todo), got, nodata, len(failed))
        time.sleep(max(0.0, gap - (time.monotonic() - t0)))
    out = {"saved": got, "empty_or_invalid": nodata, "failed": failed}
    log.info("tiingo paced done: %s", out)
    return out


def _session_get_tiingo(t: str, start: str, tok: str):
    import requests
    return requests.get(f"https://api.tiingo.com/tiingo/daily/{t.replace('.', '-')}/prices",
                        params={"startDate": start, "token": tok}, timeout=60)


def _tiingo_frame(t: str, rows: list[dict]) -> pd.DataFrame:
    d = pd.DataFrame(rows)
    split_f = d.get("splitFactor", pd.Series(1.0, index=d.index)).replace(0, 1.0)
    rev = split_f.iloc[::-1].cumprod().iloc[::-1] / split_f
    return pd.DataFrame({
        "ticker": t, "date": pd.to_datetime(d["date"]).dt.tz_localize(None).dt.date,
        "open": d["open"] / rev, "high": d["high"] / rev, "low": d["low"] / rev, "close": d["close"] / rev,
        "adj_close": d["adjClose"], "raw_close": d["close"], "volume": d["volume"] * rev, "vwap": np.nan,
        "source": "tiingo"})


def download(tickers: list[str], start: str, end: str | None = None, source: str | None = None) -> pd.DataFrame:
    source = source or get_config()["prices"]["source"]
    fn = SOURCES[source]
    chunk = get_config()["prices"]["chunk_size"]
    frames = []
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        try:
            frames.append(fn(part, start, end))
        except Exception as e:  # noqa: BLE001
            log.warning("price download failed for chunk %d (%s): %s", i // chunk, source, e)
        log.info("prices: %d/%d tickers", min(i + chunk, len(tickers)), len(tickers))
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)


def _validity_filter(df: pd.DataFrame, con) -> pd.DataFrame:
    """Drop rows for removed members whose price history starts after their membership ended
    (= ticker was recycled by another company)."""
    u = con.execute("select ticker, min(start_date) s, max(coalesce(end_date, date '2999-01-01')) e "
                    "from universe group by ticker").df()
    first = df.groupby("ticker")["date"].min().rename("first").reset_index()
    chk = first.merge(u, on="ticker", how="left")
    bad = chk[pd.to_datetime(chk["first"]) > pd.to_datetime(chk["e"])]["ticker"]
    if len(bad):
        log.info("dropping %d recycled/invalid tickers: %s", len(bad), ", ".join(bad[:15]))
    return df[~df["ticker"].isin(bad)]


def backfill(tickers: list[str] | None = None, start: str | None = None, source: str | None = None) -> int:
    cfg = get_config()
    start = start or str(pd.Timestamp(cfg["universe"]["start_date"]) - pd.Timedelta(days=400))[:10]
    with connect() as con:
        if tickers is None:
            from .universe import all_tickers
            tickers = all_tickers(con)
    df = download(tickers, start, source=source)
    with connect() as con:
        df = _validity_filter(df, con)
        got = sorted(df["ticker"].unique())
        if got:
            con.register("_tk", pd.DataFrame({"t": got}))
            n = replace_where(con, "prices_daily", df, "ticker in (select t from _tk)")
            con.unregister("_tk")
        else:
            n = 0
    missing = sorted(set(tickers) - set(df["ticker"]))
    log.info("backfill: %d rows, %d tickers, %d missing (delisted tickers are often unavailable on free sources)",
             n, len(got), len(missing))
    download_market_series(start)
    return n


def update_recent(days: int = 10, source: str | None = None) -> int:
    """Incremental update. Tickers whose stored adjustment no longer matches (split/dividend)
    are re-downloaded in full so adj_close stays internally consistent."""
    with connect() as con:
        last = con.execute("select max(date) from prices_daily").fetchone()[0]
        if last is None:
            raise RuntimeError("no prices yet; run backfill first")
        start = str(pd.Timestamp(last) - pd.Timedelta(days=days))[:10]
        live = con.execute("select distinct ticker from universe where end_date is null "
                           "or end_date >= current_date - interval 45 day").df()["ticker"].tolist()
        old = con.execute("select ticker, date, close, adj_close from prices_daily where date >= ?", [start]).df()
    new = download(live, start, source=source)
    if new.empty:
        return 0
    old["date"] = pd.to_datetime(old["date"]).dt.date
    new["date"] = pd.to_datetime(new["date"]).dt.date
    chk = new.merge(old, on=["ticker", "date"], suffixes=("", "_old"))
    drift = (np.abs(chk["adj_close"] / chk["adj_close_old"] - 1) > 1e-4) | (np.abs(chk["close"] / chk["close_old"] - 1) > 1e-4)
    redo = sorted(chk.loc[drift, "ticker"].unique())
    with connect() as con:
        n = upsert(con, "prices_daily", new[~new["ticker"].isin(redo)])
    if redo:
        log.info("re-adjusting %d tickers after splits/dividends", len(redo))
        n += backfill(redo, source=source)
    download_market_series(start)
    return n


def download_market_series(start: str) -> int:
    """VIX etc. from yfinance into the macro table (known at the close -> lag 0)."""
    tick = get_config()["prices"].get("market_tickers", [])
    if not tick:
        return 0
    try:
        import yfinance as yf
        raw = yf.download(tick, start=start, auto_adjust=False, progress=False, group_by="column")
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].set_axis(tick, axis=1)
    except Exception as e:  # noqa: BLE001
        log.warning("market series download failed: %s", e)
        return 0
    rows = []
    for t in close.columns:
        s = close[t].dropna()
        sid = "YF_" + t.replace("^", "").replace("=", "").replace("-", "").replace(".", "")
        rows.append(pd.DataFrame({"series_id": sid, "date": pd.to_datetime(s.index).tz_localize(None).date,
                                  "value": s.values, "available_date": pd.to_datetime(s.index).tz_localize(None).date}))
    df = pd.concat(rows, ignore_index=True)
    with connect() as con:
        return upsert(con, "macro", df)

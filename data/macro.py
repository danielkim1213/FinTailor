"""Macro / world-risk series and event calendar.

Point-in-time: each observation gets an `available_date` = date + lag (config [fred.series]).
Without a FRED key the public fredgraph CSV endpoint is used (no key needed).
"""
from __future__ import annotations

import io
import re

import pandas as pd

from ..config import env, get_config
from ..db import connect, upsert
from ..utils import http_get, log, third_fridays


def _bday_offset(d: pd.Series, n: int) -> pd.Series:
    return (pd.to_datetime(d) + pd.offsets.BDay(n)) if n < 20 else (pd.to_datetime(d) + pd.Timedelta(days=n))


def fetch_fred(series_id: str, start: str = "2010-01-01") -> pd.DataFrame:
    key = env("FRED_API_KEY")
    if key:
        r = http_get("https://api.stlouisfed.org/fred/series/observations",
                     params={"series_id": series_id, "api_key": key, "file_type": "json",
                             "observation_start": start}).json()
        df = pd.DataFrame(r.get("observations", []))[["date", "value"]]
    else:
        r = http_get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series_id}, timeout=15, retries=0)
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] >= start].dropna()


def collect_fred(start: str | None = None) -> int:
    start = start or str(pd.Timestamp(get_config()["universe"]["start_date"]) - pd.Timedelta(days=500))[:10]
    frames = []
    for sid, lag in get_config()["fred"]["series"].items():
        try:
            df = fetch_fred(sid, start)
        except Exception as e:  # noqa: BLE001
            log.warning("FRED %s: %s", sid, e)
            if not env("FRED_API_KEY") and not frames:
                log.warning("public FRED endpoint unreachable; set FRED_API_KEY (yfinance market series are the fallback)")
                break
            continue
        df["series_id"] = sid
        df["available_date"] = _bday_offset(df["date"], int(lag)).dt.date
        df["date"] = df["date"].dt.date
        frames.append(df)
    if not frames:
        return 0
    df = pd.concat(frames, ignore_index=True)
    with connect() as con:
        n = upsert(con, "macro", df)
    log.info("FRED: %d observations", n)
    return n


def collect_epu_gpr() -> int:
    rows = []
    try:
        r = http_get("https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv", timeout=60)
        d = pd.read_csv(io.StringIO(r.text))
        d = d.dropna(subset=["year", "month", "day"])
        dt = pd.to_datetime(dict(year=d.year.astype(int), month=d.month.astype(int), day=d.day.astype(int)))
        rows.append(pd.DataFrame({"series_id": "EPU_DAILY", "date": dt.dt.date,
                                  "value": pd.to_numeric(d["daily_policy_index"], errors="coerce"),
                                  "available_date": (dt + pd.offsets.BDay(1)).dt.date}))
    except Exception as e:  # noqa: BLE001
        log.warning("EPU: %s", e)
    try:
        r = http_get("https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls", timeout=60)
        g = pd.read_excel(io.BytesIO(r.content))
        dcol = next(c for c in g.columns if str(c).lower() in ("date", "day"))
        dt = pd.to_datetime(g[dcol], errors="coerce")
        rows.append(pd.DataFrame({"series_id": "GPR_DAILY", "date": dt.dt.date,
                                  "value": pd.to_numeric(g["GPRD"], errors="coerce"),
                                  "available_date": (dt + pd.Timedelta(days=7)).dt.date}))  # updated weekly
    except Exception as e:  # noqa: BLE001
        log.warning("GPR: %s", e)
    if not rows:
        return 0
    df = pd.concat(rows).dropna()
    with connect() as con:
        return upsert(con, "macro", df)


# ----------------------------------------------------------------------------- events
def scrape_fomc() -> pd.DataFrame:
    """FOMC decision days from the Fed calendar page (covers ~last 5 years + scheduled)."""
    html = http_get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm").text
    out = []
    months = "January|February|March|April|May|June|July|August|September|October|November|December"
    parts = re.split(r"(\d{4}) FOMC Meetings", html)
    for i in range(1, len(parts) - 1, 2):
        year, body = int(parts[i]), parts[i + 1]
        for m, days in re.findall(rf'fomc-meeting__month[^>]*>\s*<strong>({months})(?:/[A-Za-z]+)?</strong>.*?'
                                  r'fomc-meeting__date[^>]*>([\d\-]+)', body, flags=re.S):
            last_day = int(days.split("-")[-1])
            try:
                out.append(pd.Timestamp(f"{year}-{m}-{last_day}"))
            except ValueError:
                continue
    return pd.DataFrame({"date": [d.date() for d in out], "event": "FOMC"}).drop_duplicates()


def fred_release_dates(release_id: int, name: str) -> pd.DataFrame:
    key = env("FRED_API_KEY")
    if not key:
        return pd.DataFrame(columns=["date", "event"])
    r = http_get("https://api.stlouisfed.org/fred/release/dates",
                 params={"release_id": release_id, "api_key": key, "file_type": "json",
                         "include_release_dates_with_no_data": "true", "limit": 10000}).json()
    d = pd.to_datetime(pd.DataFrame(r.get("release_dates", []))["date"])
    return pd.DataFrame({"date": d.dt.date, "event": name})


def collect_events() -> int:
    cfg = get_config()
    frames = [pd.DataFrame({"date": [d.date() for d in third_fridays(cfg["universe"]["start_date"], "2028-12-31")],
                            "event": "OPEX"})]
    for fn, args in ((scrape_fomc, ()), (fred_release_dates, (10, "CPI")), (fred_release_dates, (50, "NFP"))):
        try:
            frames.append(fn(*args))
        except Exception as e:  # noqa: BLE001
            log.warning("events %s%s: %s", fn.__name__, args, e)
    manual = cfg["paths"].get("events_csv")
    if manual:
        try:
            frames.append(pd.read_csv(manual, parse_dates=["date"]).assign(date=lambda x: x.date.dt.date))
        except Exception as e:  # noqa: BLE001
            log.warning("manual events: %s", e)
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    with connect() as con:
        n = upsert(con, "events", df)
    log.info("events: %d (%s)", n, df.event.value_counts().to_dict())
    return n

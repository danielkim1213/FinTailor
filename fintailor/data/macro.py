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
        # The sheet has both `DAY` (int 19850101) and `date`. Parsing the int column as a timestamp reads it as nanoseconds
        # since 1970 and collapsed the whole series into one 1970-01-01 row, so `date` first, then DAY with its format.
        if "date" in g.columns:
            dt = pd.to_datetime(g["date"], errors="coerce")
        else:
            dt = pd.to_datetime(g["DAY"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        ok = dt.notna() & (dt >= "1985-01-01")
        rows.append(pd.DataFrame({"series_id": "GPR_DAILY", "date": dt[ok].dt.date,
                                  "value": pd.to_numeric(g.loc[ok, "GPRD"], errors="coerce"),
                                  "available_date": (dt[ok] + pd.Timedelta(days=7)).dt.date}))  # updated weekly
    except Exception as e:  # noqa: BLE001
        log.warning("GPR: %s", e)
    if not rows:
        return 0
    df = pd.concat(rows).dropna()
    with connect() as con:
        con.execute("delete from macro where series_id = 'GPR_DAILY' and date < date '1985-01-01'")  # rows from the old bug
        return upsert(con, "macro", df)


# ----------------------------------------------------------------------------- events
def _fomc_decision_day(month_text: str, day_text: str, year: int, note: str = "") -> pd.Timestamp | None:
    """'January', '28-29' -> Jan 29; 'April/May', '30-1' -> May 1. Unscheduled meetings were not known in advance, so they
    are not events a point-in-time 'days to the next FOMC' may use."""
    text = f"{month_text} {day_text} {note}".lower()
    if "unscheduled" in text or "notation" in text:  # emergency meetings and notation votes were not on the calendar
        return None
    months = [m for m in re.findall(r"[A-Za-z]+", month_text) if m[:3].lower() in
              ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")]
    days = re.findall(r"\d+", day_text)
    if not months or not days:
        return None
    try:
        return pd.Timestamp(f"{year}-{months[-1]}-{days[-1]}")
    except ValueError:
        return None


def scrape_fomc() -> pd.DataFrame:
    """Scheduled FOMC decision days from the Fed calendar page (the current and the next ~year, plus a few past years)."""
    html = http_get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm").text
    out = []
    parts = re.split(r"(\d{4}) FOMC Meetings", html)
    for i in range(1, len(parts) - 1, 2):
        year, body = int(parts[i]), parts[i + 1]
        for mo, da in re.findall(r'fomc-meeting__month[^>]*>(.*?)</div>\s*<div class="fomc-meeting__date[^>]*>(.*?)</div>',
                                 body, flags=re.S):
            d = _fomc_decision_day(re.sub(r"<[^>]+>", "", mo), re.sub(r"<[^>]+>", "", da), year)
            if d is not None:
                out.append(d)
    return pd.DataFrame({"date": [d.date() for d in out], "event": "FOMC"}).drop_duplicates()


def scrape_fomc_history(years: list[int]) -> pd.DataFrame:
    """Scheduled meetings from the per-year history pages (fomchistorical<year>.htm), whose headings read
    'January 28-29 Meeting - 2014' or 'April/May 30-1 Meeting - 2019'."""
    out = []
    for y in years:
        try:
            html = http_get(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm").text
        except Exception as e:  # noqa: BLE001
            log.warning("FOMC history %d: %s", y, e)
            continue
        for h in re.findall(r"<h5[^>]*>(.*?)</h5>", html, flags=re.S):
            text = re.sub(r"<[^>]+>", "", h)
            m = re.match(r"\s*([A-Za-z/]+)\s+([\d\-]+)(.*)", text)  # the rest holds 'Meeting - 2014' or '(unscheduled) - 2014'
            if m:
                d = _fomc_decision_day(m.group(1), m.group(2), y, m.group(3))
                if d is not None:
                    out.append(d)
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
    # The calendar page only goes back a few years; the model needs the schedule over the whole price history, otherwise
    # `days_to_fomc` is blank for most of the training window and only appears in recent years.
    fomc = pd.concat([f for f in frames if len(f) and (f["event"] == "FOMC").any()], ignore_index=True) if frames else pd.DataFrame()
    first_year = int(pd.Timestamp(cfg["universe"]["start_date"]).year) - 1
    have = set(pd.to_datetime(fomc["date"]).dt.year) if len(fomc) else set()
    missing = [y for y in range(first_year, pd.Timestamp.today().year) if y not in have]
    if missing:
        frames.append(scrape_fomc_history(missing))
    manual = cfg["paths"].get("events_csv")
    if manual:
        try:
            frames.append(pd.read_csv(manual, parse_dates=["date"]).assign(date=lambda x: x.date.dt.date))
        except Exception as e:  # noqa: BLE001
            log.warning("manual events: %s", e)
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    with connect() as con:
        # FOMC rows are replaced year by year: a date that stops being scheduled (or was a notation vote) must not linger
        years = sorted(set(pd.to_datetime(df.loc[df["event"] == "FOMC", "date"]).dt.year))
        if years:
            con.execute("delete from events where event = 'FOMC' and extract(year from date) in (" + ",".join(map(str, years)) + ")")
        n = upsert(con, "events", df)
    log.info("events: %d (%s)", n, df.event.value_counts().to_dict())
    return n

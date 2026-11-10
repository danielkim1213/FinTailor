"""GDELT DOC 2.0 API: daily article volume and tone for world-risk themes.

The free DOC API only covers a rolling ~3 months, so history is short: we archive daily from
now on (world_daily table). The model sees NaN before the archive starts.
"""
from __future__ import annotations

import pandas as pd

from ..config import get_config
from ..db import connect, upsert
from ..utils import RateLimiter, log

API = "https://api.gdeltproject.org/api/v2/doc/doc"
_lim = RateLimiter(4)  # GDELT asks for <= 1 request / 5s; stay well below


class Throttled(Exception):
    pass


def _timeline(query: str, mode: str, timespan: str) -> pd.DataFrame:
    import requests
    _lim.wait()
    r = requests.get(API, params={"query": query + " sourcelang:english", "mode": mode, "format": "json",
                                  "timespan": timespan, "timelinesmooth": 0},
                     headers={"User-Agent": "FinTailor/0.1"}, timeout=60)
    if r.status_code == 429:
        raise Throttled()
    r.raise_for_status()
    js = r.json()
    data = (js.get("timeline") or [{}])[0].get("data", [])
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"].str[:8], format="%Y%m%d")
    return df.groupby("date", as_index=False)["value"].mean()


def collect_gdelt(timespan: str = "3months") -> int:
    frames = []
    throttled = 0
    for name, q in get_config()["gdelt"]["themes"].items():
        for mode, metric in (("timelinevolraw", "vol"), ("timelinetone", "tone")):
            if throttled >= 2:
                break
            try:
                df = _timeline(q, mode, timespan)
                throttled = 0
            except Throttled:
                throttled += 1
                log.warning("gdelt %s/%s: rate limited%s", name, mode, " - giving up until next run" if throttled >= 2 else "")
                continue
            except Exception as e:  # noqa: BLE001
                log.warning("gdelt %s/%s: %s", name, mode, e)
                continue
            if len(df):
                frames.append(df.assign(source="gdelt", metric=f"{name}_{metric}", date=df["date"].dt.date))
    if not frames:
        return 0
    with connect() as con:
        n = upsert(con, "world_daily", pd.concat(frames, ignore_index=True))
    log.info("gdelt: %d rows", n)
    return n

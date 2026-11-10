"""Shared helpers: logging, HTTP with retry/rate limit, time/ticker conversions."""
from __future__ import annotations

import hashlib
import logging
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("fintailor")
ET = ZoneInfo("America/New_York")
UA = "FinTailor/0.1 (research prototype)"


def setup_logging(level: str = "INFO") -> None:
    if log.handlers:
        return
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(level)


class RateLimiter:
    """Simple sliding-window limiter (calls per minute), thread safe."""

    def __init__(self, per_minute: float):
        self.interval = 60.0 / per_minute
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
                now = time.monotonic()
            self._next = now + self.interval


_session = requests.Session()


def http_get(url: str, params: dict | None = None, headers: dict | None = None, timeout: float = 30,
             retries: int = 3, limiter: RateLimiter | None = None) -> requests.Response:
    hdrs = {"User-Agent": UA}
    if headers:
        hdrs.update(headers)
    last: Exception | None = None
    for attempt in range(retries + 1):
        if limiter:
            limiter.wait()
        try:
            r = _session.get(url, params=params, headers=hdrs, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r
        except requests.RequestException as e:  # noqa: PERF203
            last = e
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status and 400 <= status < 500 and status != 429:
                raise
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"GET failed after retries: {url}: {last}")


def utcnow() -> datetime:
    """Naive UTC timestamp (all stored timestamps are naive UTC)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def stable_id(*parts: object) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def to_yahoo(t: str) -> str:
    return t.replace(".", "-")


def from_yahoo(t: str) -> str:
    return t if t.startswith("^") else t.replace("-", ".")


def close_times_utc(trading_dates: pd.DatetimeIndex, cutoff_et: str = "16:00") -> np.ndarray:
    """Cutoff timestamps (naive UTC) for each trading date."""
    hh, mm = map(int, cutoff_et.split(":"))
    local = (pd.DatetimeIndex(trading_dates).normalize() + pd.Timedelta(hours=hh, minutes=mm)).tz_localize(ET)
    return local.tz_convert("UTC").tz_localize(None).values


def assign_trading_date(ts_utc: pd.Series, trading_dates: pd.DatetimeIndex, cutoff_et: str = "16:00") -> pd.Series:
    """Point-in-time mapping: an item received at ts belongs to the first trading date whose
    cutoff is >= ts. Items after the last known cutoff get NaT (they belong to a future date)."""
    trading_dates = pd.DatetimeIndex(trading_dates)
    ct = close_times_utc(trading_dates, cutoff_et)
    ts = pd.to_datetime(ts_utc).values
    idx = np.searchsorted(ct, ts, side="left")
    out = np.full(len(ts), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = (idx < len(trading_dates)) & ~pd.isna(ts)
    out[ok] = trading_dates.values[idx[ok]]
    return pd.Series(out, index=ts_utc.index)


def third_fridays(start: str, end: str) -> pd.DatetimeIndex:
    months = pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS")
    out = []
    for m in months:
        first_fri = m + pd.Timedelta(days=(4 - m.weekday()) % 7)
        out.append(first_fri + pd.Timedelta(days=14))
    return pd.DatetimeIndex(out)

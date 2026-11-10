"""Download the Alpaca (Benzinga) news archive month by month, newest month first. Resumable: finished months are skipped.

Stops at 2025-10-31 on purpose: everything later belongs to the locked holdout and is fetched only when that test is approved.

Usage: python research/alpaca_news.py <output dir> [first month YYYY-MM] [last month YYYY-MM] [file with comma-separated symbols]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://data.alpaca.markets/v1beta1/news"
HOLDOUT_START = pd.Timestamp("2025-11-01")
PAGE = 50
MIN_GAP = 60 / 195  # stay under the 200 requests per minute of the free plan


def headers() -> dict:
    vals = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, v = s.split("=", 1)
            vals[k.strip()] = v.strip().strip('"').strip("'")
    return {"APCA-API-KEY-ID": vals["ALPACA_API_KEY"], "APCA-API-SECRET-KEY": vals["ALPACA_SECRET_KEY"], "User-Agent": "fintailor-research"}


class Pacer:
    def __init__(self):
        self.last = 0.0
        self.calls = 0
        self.limit = None

    def wait(self) -> None:
        pause = self.last + MIN_GAP - time.time()
        if pause > 0:
            time.sleep(pause)
        self.last = time.time()
        self.calls += 1

    def after(self, r: requests.Response) -> None:
        self.limit = r.headers.get("X-RateLimit-Limit", self.limit)
        left, reset = r.headers.get("X-RateLimit-Remaining"), r.headers.get("X-RateLimit-Reset")
        if left is not None and reset is not None and int(left) <= 2:
            time.sleep(max(0.0, int(reset) - time.time()) + 1.0)


def fetch_page(session: requests.Session, hdr: dict, params: dict, pacer: Pacer) -> dict:
    for attempt in range(10):
        pacer.wait()
        try:
            r = session.get(URL, params=params, headers=hdr, timeout=60)
        except requests.RequestException as e:
            print(f"  network error ({type(e).__name__}), retry {attempt + 1}", flush=True)
            time.sleep(min(120, 3 * 2 ** attempt))
            continue
        if r.status_code == 429:
            reset = r.headers.get("X-RateLimit-Reset")
            time.sleep(max(1.0, int(reset) - time.time() + 1.0) if reset else 60.0)
            continue
        if r.status_code >= 500:
            print(f"  server error {r.status_code}, retry {attempt + 1}", flush=True)
            time.sleep(min(120, 3 * 2 ** attempt))
            continue
        if r.status_code != 200:
            raise RuntimeError(f"news request failed with status {r.status_code}")
        pacer.after(r)
        return r.json()
    raise RuntimeError("news request failed after repeated retries")


def fetch_month(month: pd.Period, session: requests.Session, hdr: dict, pacer: Pacer, symbols: str | None = None) -> pd.DataFrame:
    start = month.start_time
    end = min((month + 1).start_time, HOLDOUT_START)
    params = {"start": f"{start:%Y-%m-%dT%H:%M:%SZ}", "end": f"{end:%Y-%m-%dT%H:%M:%SZ}", "limit": PAGE, "sort": "asc", "include_content": "false"}
    if symbols:
        params["symbols"] = symbols
    rows, pages = [], 0
    while True:
        j = fetch_page(session, hdr, params, pacer)
        pages += 1
        for it in j.get("news", []):
            rows.append({"id": it.get("id"), "created_at": it.get("created_at"), "updated_at": it.get("updated_at"), "headline": it.get("headline"),
                         "summary": it.get("summary"), "author": it.get("author"), "source": it.get("source"), "url": it.get("url"),
                         "symbols": ",".join(it.get("symbols") or [])})
        token = j.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
    df = pd.DataFrame(rows, columns=["id", "created_at", "updated_at", "headline", "summary", "author", "source", "url", "symbols"])
    for c in ("created_at", "updated_at"):
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce").dt.tz_localize(None)
    df.attrs["pages"] = pages
    return df.drop_duplicates("id")


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    first = pd.Period(sys.argv[2] if len(sys.argv) > 2 else "2015-01", "M")
    last = pd.Period(sys.argv[3] if len(sys.argv) > 3 else "2025-10", "M")
    if last.end_time >= HOLDOUT_START:
        raise SystemExit("refusing to fetch into the locked holdout")
    symbols = Path(sys.argv[4]).read_text(encoding="utf-8").strip() if len(sys.argv) > 4 else None
    hdr, pacer, t0 = headers(), Pacer(), time.time()
    with requests.Session() as session:
        for month in pd.period_range(first, last, freq="M")[::-1]:
            path = out / f"{month}.parquet"
            if path.exists():
                continue
            m0, c0 = time.time(), pacer.calls
            df = fetch_month(month, session, hdr, pacer, symbols)
            tmp = path.with_suffix(".tmp")
            df.to_parquet(tmp, index=False, compression="zstd")
            tmp.replace(path)
            print(f"{month}: {len(df):,} articles, {pacer.calls - c0} calls, {time.time() - m0:.0f}s, plan limit {pacer.limit}/min, total {(time.time() - t0) / 60:.1f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()

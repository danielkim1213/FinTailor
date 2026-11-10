"""Re-download SEC fundamentals for a store, several processes at once, saving as it goes.

Usage: python research/refresh_fundamentals.py main|wide   (resumable; delete the store's fund_done.txt to start over)
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WHICH = sys.argv[1] if len(sys.argv) > 1 else "main"
if WHICH == "wide":
    os.environ.update(FINTAILOR_CONFIG=str(ROOT / "research" / "config_wide.toml"), FINTAILOR_STORE="store_wide")
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from fintailor.data import sec  # noqa: E402
from fintailor.db import connect, upsert  # noqa: E402
from fintailor.utils import RateLimiter  # noqa: E402

WORKERS, CHUNK = 4, 40


def init() -> None:
    sec._lim = RateLimiter(450 / WORKERS)  # SEC allows 10 requests/second in total


def fetch(names: list[str], ciks: dict[str, int]) -> tuple[list[str], pd.DataFrame]:
    return names, sec.fetch_fundamentals(names, ciks)


def main() -> None:
    store = ROOT / ("store_wide" if WHICH == "wide" else "store")
    if WHICH == "wide":
        g = pd.read_parquet(store / "issuers.parquet")
        ciks = dict(zip(g["ticker"], g["cik"]))
        where = "source = 'form345'"
    else:
        ciks = {**sec.cik_map(), **sec.historical_cik_map()}  # a name's filings while in the index beat today's ticker list
        where = "not is_etf"
    done_file = store / "fund_done.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    with connect(read_only=True) as con:
        names = sorted(r[0] for r in con.execute(f"select distinct ticker from universe where {where}").fetchall())
    todo = [t for t in names if t not in done and t in ciks]
    print(f"{len(todo)} to fetch of {len(names)}", flush=True)
    chunks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]
    finished = 0
    with ProcessPoolExecutor(WORKERS, initializer=init) as ex:
        futures = [ex.submit(fetch, c, ciks) for c in chunks]
        for fut in as_completed(futures):
            try:
                got, df = fut.result()
            except Exception as e:  # noqa: BLE001
                print("chunk failed:", repr(e)[:200], flush=True)
                continue
            with connect() as con:
                placeholders = ",".join(f"'{t}'" for t in got)
                con.execute(f"delete from fundamentals where ticker in ({placeholders})")
                if len(df):
                    upsert(con, "fundamentals", df)
            with open(done_file, "a") as f:
                f.write(" ".join(got) + " ")
            finished += len(got)
            if finished % 400 == 0:
                print(f"{finished}/{len(todo)}", flush=True)
    print("done", finished, flush=True)


if __name__ == "__main__":
    main()

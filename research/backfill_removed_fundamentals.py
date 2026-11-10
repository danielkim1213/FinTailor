"""Fill the fundamentals gap for names that left the index, using CIKs recovered from SEC Form 3/4/5 data sets."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fintailor.data.sec import collect_fundamentals, historical_cik_map  # noqa: E402
from fintailor.db import connect  # noqa: E402

with connect(read_only=True) as con:
    all_t = [r[0] for r in con.execute("select distinct ticker from universe where not is_etf").fetchall()]
    have = {r[0] for r in con.execute("select distinct ticker from fundamentals").fetchall()}
missing = sorted(set(all_t) - have)
cmap = historical_cik_map()
found = [t for t in missing if t in cmap]
print(f"missing {len(missing)}, CIK recovered for {len(found)}; still unknown: {[t for t in missing if t not in cmap][:60]}", flush=True)
print("rows written:", collect_fundamentals(found, ciks=cmap), flush=True)
with connect(read_only=True) as con:
    now = {r[0] for r in con.execute("select distinct ticker from fundamentals").fetchall()}
print(f"gained {len(now - have)} tickers; no XBRL revenue data for: {sorted(set(found) - now)}", flush=True)

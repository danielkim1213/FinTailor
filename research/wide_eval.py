"""Does reaching beyond the S&P 500 help? All-US universe, realistic trading costs by liquidity, daily simulator.

Usage: python research/wide_eval.py [ml_score_parquet]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(FINTAILOR_CONFIG=str(ROOT / "research" / "config_wide.toml"), FINTAILOR_STORE="store_wide")
sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fintailor.strategy.signals import factor_score, sector_score, zscore  # noqa: E402
from fintailor.strategy.simulator import load_market, run_book, tilt_weights  # noqa: E402
from fintailor.strategy.stats import block_bootstrap, newey_west_t, sharpe  # noqa: E402

pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 100, "display.float_format", lambda v: f"{v:.2f}")
START, END, DEV_END = pd.Timestamp("2019-01-01"), pd.Timestamp("2025-10-31"), pd.Timestamp("2023-12-31")
COLS = ["date", "ticker", "mcap", "is_etf_flag", "dvol20", "sector", "fund_known", "raw_close", "mom_12_1", "gp_assets_nf", "asset_growth_nf", "book_to_price_nf", "earn_yield_nf"]


def tiered_cost(dv: pd.Series) -> pd.Series:
    """Round-trip cost by typical dollar volume."""
    return pd.cut(dv, [0, 5e6, 20e6, 100e6, np.inf], labels=[0.008, 0.004, 0.002, 0.001]).astype(float).fillna(0.008)


def main() -> None:
    p = pd.read_parquet(ROOT / "store_wide/features/panel.parquet", columns=COLS)
    p = p[(~p["is_etf_flag"].astype(bool)) & (p["date"] >= START) & (p["date"] <= END) & p["fund_known"] & (p["mcap"] > 0)].reset_index(drop=True)
    print(f"rows {len(p):,}, names per day {p.groupby('date').size().median():.0f}, tickers {p['ticker'].nunique():,}", flush=True)
    for _ in range(40):
        try:
            con = duckdb.connect(str(ROOT / "store_wide/fintailor.duckdb"), read_only=True)
            break
        except duckdb.Error:
            time.sleep(3)
    m = load_market(con, "2018-09-01", str(END.date()))
    p["z_factor"] = factor_score(p)
    p["z_sector"] = sector_score(p, con)
    con.close()
    p["z_fs"] = zscore(p["z_factor"] + p["z_sector"], p["date"])
    if len(sys.argv) > 1:
        ml = pd.read_parquet(sys.argv[1])
        p = p.merge(ml, on=["date", "ticker"], how="left")
        p["z_ml"] = zscore(p["ml"], p["date"])
        p["z_all"] = zscore(p["z_factor"] + p["z_sector"] + p["z_ml"], p["date"])
    cost = tiered_cost(p.groupby("ticker")["dvol20"].median())
    dates = np.sort(p["date"].unique())
    groups = {d: g for d, g in p.groupby("date")}

    def tilt_book(col, lam, offset):
        return {pd.Timestamp(d): tilt_weights(pd.DataFrame({"ticker": groups[d]["ticker"].values, "mcap": groups[d]["mcap"].values, "z": groups[d][col].values if col else 0.0}), lam) for d in dates[offset::5]}

    def top_book(col, n, offset, floor=0.0):
        out = {}
        for d in dates[offset::5]:
            g = groups[d]
            g = g[g["mcap"] >= floor]
            pick = g.nlargest(n, col)
            out[pd.Timestamp(d)] = pd.Series(1.0 / len(pick), index=pick["ticker"].values)
        return out

    def avg(builder):
        runs = [run_book(m, builder(i), cost) for i in range(5)]
        return pd.DataFrame({"ret": pd.concat([r["ret"] for r in runs], axis=1).mean(axis=1), "turnover": pd.concat([r["turnover"] for r in runs], axis=1).mean(axis=1)})

    control = avg(lambda i: tilt_book(None, 0.0, i))
    x = (control["ret"] - m.spy.reindex(control.index)).dropna()
    print(f"control (cap-weighted all-US) vs SPY: {x.mean() * 25200:+.2f}%/yr, tracking {x.std() * np.sqrt(252) * 100:.2f}%", flush=True)

    def report(name, r):
        a = (r["ret"] - control["ret"]).dropna()
        v = (r["ret"] - m.spy.reindex(r.index)).dropna()
        bb = block_bootstrap(a)
        s = {"vs_spy": v.mean() * 25200, "active": a.mean() * 25200, "te": a.std() * np.sqrt(252) * 100, "ir": sharpe(a), "t": newey_west_t(a), "p_pos": bb["p_positive"] * 100,
             "dev": a[a.index <= DEV_END].mean() * 25200, "val": a[a.index > DEV_END].mean() * 25200, "spy_dev": v[v.index <= DEV_END].mean() * 25200, "spy_val": v[v.index > DEV_END].mean() * 25200, "turn": r["turnover"].mean() * 252}
        print(name, {k: round(float(val), 2) for k, val in s.items()}, flush=True)
        return s

    rows = {}
    cols = {"factors": "z_factor", "sector": "z_sector", "fact+sector": "z_fs"}
    if "z_all" in p:
        cols.update({"ml": "z_ml", "all3": "z_all"})
    for name, col in cols.items():
        rows[f"{name} tilt .5"] = report(f"{name} tilt .5", avg(lambda i: tilt_book(col, 0.5, i)))
        rows[f"{name} top100 eq"] = report(f"{name} top100 eq", avg(lambda i: top_book(col, 100, i)))
        rows[f"{name} top100 eq, cap>2B"] = report(f"{name} top100 eq, cap>2B", avg(lambda i: top_book(col, 100, i, 2e9)))
        rows[f"{name} top100 eq, cap>10B"] = report(f"{name} top100 eq, cap>10B", avg(lambda i: top_book(col, 100, i, 10e9)))
    print("\n", pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main()

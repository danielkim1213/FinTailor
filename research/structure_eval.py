"""Same forecasts, different ways of turning them into positions. S&P 500 universe, daily simulator, 10bp costs.

additive : cap weight + kappa * z / N, floored at zero. Every stock's active bet has the same size, however big the company is.
top50    : the 50 best-ranked stocks at equal weight, keeping a holding while it stays in the top 150.
Forecasts come from the engine's out-of-sample cache, so nothing here trains on the test period.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fintailor.db import connect  # noqa: E402
from fintailor.strategy.engine import DEV_END, _load_panel, _scores  # noqa: E402
from fintailor.strategy.simulator import load_market, run_book, tilt_weights  # noqa: E402
from fintailor.strategy.stats import block_bootstrap, newey_west_t, sharpe  # noqa: E402

pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 100, "display.float_format", lambda v: f"{v:.2f}")
START, END = pd.Timestamp("2019-01-01"), pd.Timestamp("2025-10-31")


def additive(g: pd.DataFrame, col: str, kappa: float) -> pd.Series:
    base = (g["mcap"] / g["mcap"].sum()).to_numpy()
    w = np.clip(base + kappa * g[col].clip(-2.5, 2.5).to_numpy() / len(g), 0, None)
    return pd.Series(w / w.sum(), index=g["ticker"].to_numpy())


def top_with_buffer(g: pd.DataFrame, col: str, held: set, k: int = 50, buffer: int = 3) -> pd.Series:
    ranked = g.sort_values(col, ascending=False)["ticker"].to_numpy()
    keep = [t for t in ranked[: k * buffer] if t in held][:k]
    for t in ranked:
        if len(keep) >= k:
            break
        if t not in keep:
            keep.append(t)
    return pd.Series(1.0 / len(keep), index=keep)


def main() -> None:
    p, _ = _load_panel()
    ml = pd.read_parquet(ROOT / "store/strategy/ml_oos.parquet")["ml"]
    with connect(read_only=True) as con:
        s = _scores(p, con, ml)[(p["date"] >= START) & (p["date"] <= END)].dropna(subset=["z_ml"]).reset_index(drop=True)
        m = load_market(con, "2018-09-01", str(END.date()))
    s["z_fs"] = (s["z_factor"] + s["z_sector"]) / np.sqrt(2)
    dates = np.sort(s["date"].unique())
    groups = {d: g for d, g in s.groupby("date")}

    def avg(builder):
        runs = [run_book(m, builder(i), 0.001) for i in range(5)]
        return pd.DataFrame({"ret": pd.concat([r["ret"] for r in runs], axis=1).mean(axis=1), "turnover": pd.concat([r["turnover"] for r in runs], axis=1).mean(axis=1)})

    def add_book(col, kappa, offset):
        return {pd.Timestamp(d): additive(groups[d], col, kappa) for d in dates[offset::5]}

    def top_book(col, offset):
        out, held = {}, set()
        for d in dates[offset::5]:
            out[pd.Timestamp(d)] = w = top_with_buffer(groups[d], col, held)
            held = set(w.index)
        return out

    def cap_book(offset):
        return {pd.Timestamp(d): tilt_weights(pd.DataFrame({"ticker": groups[d]["ticker"].values, "mcap": groups[d]["mcap"].values, "z": 0.0}), 0.0) for d in dates[offset::5]}

    def equal_book(offset):
        return {pd.Timestamp(d): pd.Series(1.0 / len(groups[d]), index=groups[d]["ticker"].values) for d in dates[offset::5]}

    ctrl = avg(cap_book)
    eq = avg(equal_book)
    x = (eq["ret"] - m.spy.reindex(eq.index)).dropna()
    print(f"equal-weight index members vs SPY: {x.mean() * 25200:+.2f}%/yr")
    rows, series = {}, {}

    def report(name, r, base):
        a = (r["ret"] - base).dropna()
        v = (r["ret"] - m.spy.reindex(r.index)).dropna()
        bb = block_bootstrap(a)
        rows[name] = {"vs_spy": v.mean() * 25200, "spy_dev": v[v.index <= DEV_END].mean() * 25200, "spy_val": v[v.index > DEV_END].mean() * 25200, "active": a.mean() * 25200,
                      "te": a.std() * np.sqrt(252) * 100, "ir": sharpe(a), "t": newey_west_t(a), "p_pos": bb["p_positive"] * 100, "dev": a[a.index <= DEV_END].mean() * 25200, "val": a[a.index > DEV_END].mean() * 25200, "turn": r["turnover"].mean() * 252}
        series[name] = a
        print(name, {k: round(float(val), 2) for k, val in rows[name].items()}, flush=True)

    for col, label in (("z_ml", "ml"), ("z_fs", "fact+sector"), ("z_factor", "factors"), ("z_sector", "sector"), ("z", "all3")):
        for kappa in (1.0, 2.0, 4.0):
            report(f"{label} additive k={kappa}", avg(lambda i: add_book(col, kappa, i)), ctrl["ret"])
        report(f"{label} top50 vs equal-weight control", avg(lambda i: top_book(col, i)), eq["ret"])
    print("\n", pd.DataFrame(rows).T.to_string())
    d = pd.DataFrame(series)
    print("\nby year, active %:\n", d.groupby(d.index.year).mean().mul(25200).T.round(1).to_string())


if __name__ == "__main__":
    main()

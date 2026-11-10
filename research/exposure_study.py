"""How much of the market to hold, tested on 100 years of daily US market returns (Ken French).

Rules are taken from published practice and judged across a whole parameter grid, so a single lucky setting cannot win.
Signals use data through day t and apply to day t+1. Costs: 5bp on each change in exposure, borrowing at T-bill + 0.5%/yr.
Data after 2025-10-31 is excluded (locked holdout).

Usage: python -I research/exposure_study.py <french_directory>
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 200, "display.float_format", lambda v: f"{v:.2f}")
END = pd.Timestamp("2025-10-31")
COST = 0.0005
SPREAD = 0.005 / 252
BAND = 0.10


def load(folder: Path) -> pd.DataFrame:
    with zipfile.ZipFile(folder / "F-F_Research_Data_Factors_daily_CSV.zip") as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    rows = []
    for line in io.StringIO(text):
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 5 and p[0].isdigit() and len(p[0]) == 8:
            rows.append((pd.Timestamp(p[0]), float(p[1]), float(p[4])))
    d = pd.DataFrame(rows, columns=["date", "mkt_rf", "rf"]).set_index("date") / 100
    d = d[d.index <= END]
    d["mkt"] = d["mkt_rf"] + d["rf"]
    return d


def exposure(d: pd.DataFrame, ma=200, vol_win=20, target=None, lev=1.0, below=0.0) -> pd.Series:
    """Target weight known at each day's close. target=None: no volatility scaling."""
    idx = (1 + d["mkt"]).cumprod()
    up = idx > idx.rolling(ma, min_periods=ma).mean() if ma else pd.Series(True, index=d.index)
    w = pd.Series(lev, index=d.index, dtype=float)
    if target:
        sig = d["mkt"].rolling(vol_win, min_periods=vol_win).std() * np.sqrt(252)
        w = (target / sig).clip(upper=lev)
    w = w.where(up, below * w.clip(upper=1.0))
    return w.fillna(1.0)


def run(d: pd.DataFrame, w: pd.Series) -> pd.Series:
    """Daily net return. Exposure moves only when the target drifts more than BAND from the position."""
    tv, held, out = w.values, 1.0, np.zeros(len(w))
    r, rf = d["mkt"].values, d["rf"].values
    for i in range(len(w)):
        out[i] = held * r[i] + (1 - held) * rf[i] - max(held - 1, 0) * SPREAD
        if abs(tv[i] - held) > BAND:
            out[i] -= abs(tv[i] - held) * COST
            held = tv[i]
    return pd.Series(out, index=w.index)


def stats(r: pd.Series, rf: pd.Series) -> dict:
    n = len(r) / 252
    cum = (1 + r).cumprod()
    return {"cagr": 100 * (cum.iloc[-1] ** (1 / n) - 1), "vol": 100 * r.std() * np.sqrt(252),
            "sharpe": (r - rf).mean() / r.std() * np.sqrt(252), "maxdd": 100 * (cum / cum.cummax() - 1).min(),
            "worst_yr": 100 * (r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)).min()}


def rolling_beat(r: pd.Series, b: pd.Series, years: int) -> float:
    w = 252 * years
    a = np.log1p(r).rolling(w).sum()
    c = np.log1p(b).rolling(w).sum()
    return float((a - c).dropna().gt(0).mean() * 100)


def bootstrap_excess(r: pd.Series, b: pd.Series, n=2000, block=252, seed=7) -> tuple[float, float, float]:
    """Moving-block bootstrap of the annualised log-return difference; blocks of one year keep crises intact."""
    x = (np.log1p(r) - np.log1p(b)).values
    rng, T = np.random.default_rng(seed), len(x)
    means = np.empty(n)
    k = int(np.ceil(T / block))
    for i in range(n):
        s = rng.integers(0, T - block + 1, k)
        means[i] = np.concatenate([x[a:a + block] for a in s])[:T].mean() * 252 * 100
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float((means > 0).mean() * 100)


d = load(Path(sys.argv[1]))
print(f"market data {d.index[0].date()}..{d.index[-1].date()}, {len(d):,} days")
PERIODS = {"1926-2025": ("1926", "2025-10"), "1926-1962": ("1926", "1962"), "1963-1999": ("1963", "1999"),
           "2000-2025": ("2000", "2025-10"), "2010-2025": ("2010", "2025-10"), "2019-2025": ("2019", "2025-10")}
RULES = {
    "buy & hold": {"ma": 0, "lev": 1.0},
    "trend 200d (1x/cash)": {"ma": 200, "lev": 1.0},
    "vol target 15% (max 1x)": {"ma": 0, "target": 0.15, "lev": 1.0},
    "vol target 15% (max 1.5x)": {"ma": 0, "target": 0.15, "lev": 1.5},
    "vol target 18% (max 1.5x)": {"ma": 0, "target": 0.18, "lev": 1.5},
    "trend + 1.5x": {"ma": 200, "lev": 1.5},
    "trend + 2x": {"ma": 200, "lev": 2.0},
    "trend + vol15 (max 1.5x)": {"ma": 200, "target": 0.15, "lev": 1.5},
    "trend + vol18 (max 1.5x)": {"ma": 200, "target": 0.18, "lev": 1.5},
    "trend + vol20 (max 2x)": {"ma": 200, "target": 0.20, "lev": 2.0},
    "half below trend, vol18 (max 1.5x)": {"ma": 200, "target": 0.18, "lev": 1.5, "below": 0.5},
}
res = {k: run(d, exposure(d, **v)) for k, v in RULES.items()}
bh = res["buy & hold"]
for label, (a, b) in PERIODS.items():
    rows = {k: stats(v[a:b], d["rf"][a:b]) for k, v in res.items()}
    print(f"\n=== {label}\n", pd.DataFrame(rows).T.round(2).to_string())

print("\n=== beating buy & hold, whole sample")
rows = {}
for k, v in res.items():
    if k == "buy & hold":
        continue
    lo, hi, p = bootstrap_excess(v, bh)
    rows[k] = {"excess_log_%/yr": (np.log1p(v) - np.log1p(bh)).mean() * 252 * 100, "ci_lo": lo, "ci_hi": hi, "P(excess>0)%": p,
               "beat_5y_%": rolling_beat(v, bh, 5), "beat_10y_%": rolling_beat(v, bh, 10), "beat_20y_%": rolling_beat(v, bh, 20)}
print(pd.DataFrame(rows).T.round(2).to_string())

print("\n=== robustness: all 300 parameter combinations (trend + vol target)")
grid = []
for ma in (100, 150, 200, 250, 300):
    for vw in (10, 20, 40, 60):
        for t in (0.12, 0.15, 0.18, 0.20, 0.25):
            for lev in (1.25, 1.5, 2.0):
                r = run(d, exposure(d, ma=ma, vol_win=vw, target=t, lev=lev))
                s = stats(r, d["rf"])
                r2 = r["2000":]
                grid.append({"ma": ma, "vw": vw, "t": t, "lev": lev, **s, "ex": s["cagr"] - stats(bh, d["rf"])["cagr"],
                             "ex_2000": stats(r2, d["rf"]["2000":])["cagr"] - stats(bh["2000":], d["rf"]["2000":])["cagr"]})
g = pd.DataFrame(grid)
print(f"share of settings beating buy & hold on CAGR: all years {g.ex.gt(0).mean():.0%}, since 2000 {g.ex_2000.gt(0).mean():.0%}; "
      f"on Sharpe {g.sharpe.gt(stats(bh, d['rf'])['sharpe']).mean():.0%}")
print(g.groupby("lev")[["cagr", "vol", "sharpe", "maxdd", "ex", "ex_2000"]].mean().round(2).to_string())
print(g.groupby("ma")[["cagr", "sharpe", "maxdd", "ex", "ex_2000"]].mean().round(2).to_string())
print(g.groupby("t")[["cagr", "vol", "sharpe", "maxdd", "ex", "ex_2000"]].mean().round(2).to_string())
print(g.groupby("vw")[["cagr", "sharpe", "maxdd", "ex", "ex_2000"]].mean().round(2).to_string())

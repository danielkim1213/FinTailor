"""Stress test of volatility targeting on 100 years of daily data: full parameter grid, financing cost, execution delay,
and whether the gain is more than simply holding more market on average. Data after 2025-10-31 excluded.

Usage: python -I research/exposure_robust.py <french_directory>
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
BAND = 0.10
PERIODS = {"1926-62": ("1926", "1962"), "1963-99": ("1963", "1999"), "2000-09": ("2000", "2009"), "2010-19": ("2010", "2019"), "2020-25": ("2020", "2025-10")}


def load(folder: Path) -> pd.DataFrame:
    with zipfile.ZipFile(folder / "F-F_Research_Data_Factors_daily_CSV.zip") as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    rows = [(pd.Timestamp(p[0]), float(p[1]), float(p[4])) for p in (([x.strip() for x in line.split(",")]) for line in io.StringIO(text))
            if len(p) >= 5 and p[0].isdigit() and len(p[0]) == 8]
    d = pd.DataFrame(rows, columns=["date", "mkt_rf", "rf"]).set_index("date") / 100
    d = d[d.index <= END]
    d["mkt"] = d["mkt_rf"] + d["rf"]
    return d


def target_weight(d, target, lev, win, estimator="sma"):
    if estimator == "ewma":
        sig = np.sqrt(d["mkt"].pow(2).ewm(halflife=win / 2, min_periods=win).mean() * 252)
    else:
        sig = d["mkt"].rolling(win, min_periods=win).std() * np.sqrt(252)
    return (target / sig).clip(upper=lev).fillna(1.0)


def run(d, w, spread=0.005, cost=0.0005, delay=0):
    tv = w.shift(delay).fillna(1.0).values
    held, out, r, rf = 1.0, np.zeros(len(w)), d["mkt"].values, d["rf"].values
    sp = spread / 252
    pos = np.zeros(len(w))
    for i in range(len(w)):
        pos[i] = held
        out[i] = held * r[i] + (1 - held) * rf[i] - max(held - 1, 0) * sp
        if abs(tv[i] - held) > BAND:
            out[i] -= abs(tv[i] - held) * cost
            held = tv[i]
    return pd.Series(out, index=w.index), pd.Series(pos, index=w.index)


def cagr(r):
    return 100 * ((1 + r).prod() ** (252 / len(r)) - 1)


def sharpe(r, rf):
    return (r - rf).mean() / r.std() * np.sqrt(252)


def mdd(r):
    c = (1 + r).cumprod()
    return 100 * (c / c.cummax() - 1).min()


d = load(Path(sys.argv[1]))
bh = d["mkt"]
print("=== full grid, no trend filter: excess CAGR over buy & hold (points) and Sharpe gain, by period")
rows = []
for t in (0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25):
    for lev in (1.0, 1.25, 1.5, 2.0):
        for win in (10, 20, 40, 60):
            r, pos = run(d, target_weight(d, t, lev, win))
            x = {"t": t, "lev": lev, "win": win, "cagr": cagr(r), "vol": r.std() * np.sqrt(252) * 100, "avg_pos": pos.mean(),
                 "sharpe": sharpe(r, d["rf"]), "mdd": mdd(r), "ex": cagr(r) - cagr(bh)}
            for k, (a, b) in PERIODS.items():
                x[f"ex_{k}"], x[f"sh_{k}"] = cagr(r[a:b]) - cagr(bh[a:b]), sharpe(r[a:b], d["rf"][a:b]) - sharpe(bh[a:b], d["rf"][a:b])
            rows.append(x)
g = pd.DataFrame(rows)
print(f"{len(g)} settings. buy & hold: CAGR {cagr(bh):.2f}, vol {bh.std() * np.sqrt(252) * 100:.1f}, Sharpe {sharpe(bh, d['rf']):.2f}, maxDD {mdd(bh):.0f}")
ex_cols = [f"ex_{k}" for k in PERIODS]
sh_cols = [f"sh_{k}" for k in PERIODS]
print("share of settings with higher Sharpe than buy & hold, by period:", {k[3:]: f"{(g[k] > 0).mean():.0%}" for k in sh_cols})
print("share of settings with higher CAGR, by period:             ", {k[3:]: f"{(g[k] > 0).mean():.0%}" for k in ex_cols})
print("\nby target vol (all windows, cap 1.5x):")
print(g[g.lev == 1.5].groupby("t")[["cagr", "vol", "avg_pos", "sharpe", "mdd", "ex"] + ex_cols + sh_cols].mean().round(2).to_string())
print("\nby leverage cap (target 18%):")
print(g[g.t == 0.18].groupby("lev")[["cagr", "vol", "avg_pos", "sharpe", "mdd", "ex"] + ex_cols + sh_cols].mean().round(2).to_string())
print("\nby window (target 18%, cap 1.5x):")
print(g[(g.t == 0.18) & (g.lev == 1.5)].groupby("win")[["cagr", "vol", "avg_pos", "sharpe", "mdd", "ex"] + ex_cols + sh_cols].mean().round(2).to_string())

base = target_weight(d, 0.18, 1.5, 20)
print("\n=== reference setting (target 18%, cap 1.5x, 20 days): sensitivity")
rows = {}
for label, kw in {"base (RF+0.5%, 5bp, same-close)": {}, "borrow RF+1.5%": {"spread": 0.015}, "borrow RF+2.5%": {"spread": 0.025}, "cost 20bp": {"cost": 0.002},
                  "one day later": {"delay": 1}, "two days later": {"delay": 2}}.items():
    r, pos = run(d, base, **kw)
    rows[label] = {"cagr": cagr(r), "sharpe": sharpe(r, d["rf"]), "mdd": mdd(r), "ex": cagr(r) - cagr(bh), **{k: cagr(r[a:b]) - cagr(bh[a:b]) for k, (a, b) in PERIODS.items()}}
print(pd.DataFrame(rows).T.round(2).to_string())

print("\n=== same average market exposure? alpha over a constant-exposure mix (daily regression)")
r, pos = run(d, base)
mx = d["mkt_rf"].values
y = (r - d["rf"]).values
for label, m in {"whole sample": np.ones(len(y), bool), "1926-1962": d.index.year <= 1962, "1963-1999": (d.index.year >= 1963) & (d.index.year <= 1999),
                 "2000-2025": d.index.year >= 2000, "2010-2025": d.index.year >= 2010}.items():
    X = np.column_stack([np.ones(m.sum()), mx[m]])
    b = np.linalg.lstsq(X, y[m], rcond=None)[0]
    e = y[m] - X @ b
    T, L, u = m.sum(), 20, X * e[:, None]  # Newey-West (20 lags)
    S = u.T @ u / T
    for l in range(1, L + 1):
        G = u[l:].T @ u[:-l] / T
        S += (1 - l / (L + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / T)
    se = np.sqrt(np.diag(Q @ S @ Q / T))
    print(f"{label:13s} beta {b[1]:.2f}  alpha {b[0] * 252 * 100:+.2f}%/yr  t {b[0] / se[0]:+.2f}  avg exposure {pos[m].mean():.2f}")

print("\n=== year-by-year, 2015..2025-10 (buy & hold vs reference)")
yr = pd.DataFrame({"bh": bh.groupby(bh.index.year).apply(lambda x: (1 + x).prod() - 1) * 100, "vt": r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1) * 100})
yr["diff"] = yr.vt - yr.bh
print(yr.loc[2015:].round(1).T.to_string())
print(f"years ahead of buy & hold: since 1927 {(yr.loc[1927:]['diff'] > 0).mean():.0%}; since 2000 {(yr.loc[2000:]['diff'] > 0).mean():.0%}")

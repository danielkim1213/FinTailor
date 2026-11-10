"""Two more century-long questions. (1) Does holding the strongest industries beat the market? (2) What does plain extra market
exposure actually buy, and at what risk? Value-weighted Ken French data, costs included, data after 2025-10 excluded.

Usage: python -I research/sector_study.py <industry_directory> <factors_directory>
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 240, "display.max_columns", 40, "display.float_format", lambda v: f"{v:.2f}")
END = pd.Period("2025-10", "M")
COST = 0.001
PERIODS = {"1927-62": ("1927-01", "1962-12"), "1963-99": ("1963-01", "1999-12"), "2000-09": ("2000-01", "2009-12"), "2010-19": ("2010-01", "2019-12"), "2020-25": ("2020-01", "2025-10")}


def monthly_block(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    head, rows, on = None, [], False
    for line in lines:
        p = [x.strip() for x in line.split(",")]
        if p[0].isdigit() and len(p[0]) == 6:
            on = True
            rows.append(p)
        elif on:
            break
        elif len(p) > 3 and p[0] == "":
            head = p[1:]
    d = pd.DataFrame([r[1:] for r in rows], index=pd.PeriodIndex([f"{r[0][:4]}-{r[0][4:]}" for r in rows], freq="M"), columns=head).astype(float)
    return d.where(d > -99) / 100


def nw_t(x, L=12):
    v = pd.Series(x).dropna().values
    e, T = v - v.mean(), len(v)
    s = e @ e / T
    for k in range(1, L + 1):
        s += 2 * (1 - k / (L + 1)) * (e[k:] @ e[:-k]) / T
    return v.mean() / np.sqrt(s / T)


def strategy(R: pd.DataFrame, look: int, skip: int, top: int) -> pd.Series:
    """Hold the top industries by trailing return. The signal for month t uses returns up to t-1-skip."""
    mom = np.log1p(R).shift(1 + skip).rolling(look, min_periods=look).sum()
    pick = mom.rank(axis=1, ascending=False, method="first").le(top) & mom.notna()
    w = pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0)
    gross = (w * R).sum(axis=1).where(pick.sum(axis=1) > 0)
    turn = (w.fillna(0) - w.fillna(0).shift(1)).abs().sum(axis=1) / 2
    return gross - turn * COST * 2


inds, fac = Path(sys.argv[1]), Path(sys.argv[2])
ff = monthly_block(fac / "F-F_Research_Data_Factors_CSV.zip")
mkt = (ff["Mkt-RF"] + ff["RF"])
mkt = mkt[mkt.index <= END]
for name, top_range in (("10_Industry_Portfolios_CSV.zip", (2, 3, 4, 5)), ("30_Industry_Portfolios_CSV.zip", (3, 5, 8, 10))):
    R = monthly_block(inds / name)
    R = R[R.index <= END]
    print(f"\n=========== {name}, {R.index[0]}..{R.index[-1]}, {R.shape[1]} industries")
    rows, grid = {}, []
    for look in (3, 6, 9, 12):
        for skip in (0, 1):
            for top in top_range:
                ex = (strategy(R, look, skip, top) - mkt).dropna()
                g = {"look": look, "skip": skip, "top": top, "ex": ex.mean() * 1200, "t": nw_t(ex), "te": ex.std() * np.sqrt(12) * 100}
                for k, (a, b) in PERIODS.items():
                    g[k] = ex[a:b].mean() * 1200
                grid.append(g)
    g = pd.DataFrame(grid)
    print(f"{len(g)} settings. share with positive excess, by period:", {k: f"{(g[k] > 0).mean():.0%}" for k in PERIODS}, "| all years", f"{(g.ex > 0).mean():.0%}")
    print(g.groupby(["look", "skip"])[["ex", "t", "te"] + list(PERIODS)].mean().round(2).to_string())
    print("by number held:\n", g.groupby("top")[["ex", "t", "te"] + list(PERIODS)].mean().round(2).to_string())
    mid = top_range[1]
    ex = (strategy(R, 12, 1, mid) - mkt).dropna()
    roll = np.log1p(strategy(R, 12, 1, mid)).sub(np.log1p(mkt)).dropna().rolling(120).sum().dropna()
    print(f"reference 12-1, top {mid}: excess {ex.mean() * 1200:+.2f}%/yr, t {nw_t(ex):+.2f}, TE {ex.std() * np.sqrt(12) * 100:.1f}%, IR {ex.mean() / ex.std() * np.sqrt(12):.2f}; "
          f"beat market over 10y: {(roll > 0).mean():.0%}; t since 2000: {nw_t(ex['2000-01':]):+.2f}; since 2010: {nw_t(ex['2010-01':]):+.2f}")

# ---- constant extra exposure
with zipfile.ZipFile(fac / "F-F_Research_Data_Factors_daily_CSV.zip") as z:
    text = z.read(z.namelist()[0]).decode("latin-1")
rows = [(pd.Timestamp(p[0]), float(p[1]), float(p[4])) for p in ([x.strip() for x in line.split(",")] for line in io.StringIO(text)) if len(p) >= 5 and p[0].isdigit() and len(p[0]) == 8]
d = pd.DataFrame(rows, columns=["date", "mkt_rf", "rf"]).set_index("date") / 100
d = d[d.index <= "2025-10-31"]
d["mkt"] = d["mkt_rf"] + d["rf"]
print("\n=========== constant exposure, rebalanced daily; borrowing at T-bill + 1%/yr (roughly a leveraged-ETF blend)")
out = {}
for L in (1.0, 1.1, 1.25, 1.5, 2.0):
    r = L * d["mkt"] - (L - 1) * (d["rf"] + 0.01 / 252)
    c = (1 + r).cumprod()
    res = {"cagr": 100 * (c.iloc[-1] ** (252 / len(r)) - 1), "vol": r.std() * np.sqrt(252) * 100, "sharpe": (r - d["rf"]).mean() / r.std() * np.sqrt(252), "maxdd": 100 * (c / c.cummax() - 1).min()}
    y = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    res["worst_yr"] = y.min()
    lb = np.log1p(r) - np.log1p(d["mkt"])
    for yrs in (5, 10, 20):
        res[f"beat_{yrs}y"] = float(lb.rolling(252 * yrs).sum().dropna().gt(0).mean() * 100)
    for k, (a, b) in {"2000-25": ("2000", "2025-10"), "2010-25": ("2010", "2025-10")}.items():
        res[k] = 100 * ((1 + r[a:b]).prod() ** (252 / len(r[a:b])) - 1)
    out[L] = res
print(pd.DataFrame(out).T.round(2).to_string())

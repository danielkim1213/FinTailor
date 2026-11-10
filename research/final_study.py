"""The candidate program on a century of daily data, with every parameter fixed before looking.

  sector sleeve : half the book in the top 3 of 10 industries by 12-month return, skipping the latest month, reset monthly
  core          : the other half in the whole market
  exposure      : scale the book to a 17% volatility target (the market's own long-run volatility) from the last 20 days,
                  capped at 1.5x, acted on one day late, and only when the target moves more than 10% from the position.

Costs: 10bp on sector turnover, 5bp on exposure changes, borrowing at T-bill + 1%/yr. Data after 2025-10-31 excluded.
Usage: python -I research/final_study.py <industry_daily_directory> <factors_directory>
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 240, "display.max_columns", 40, "display.float_format", lambda v: f"{v:.2f}")
END = pd.Timestamp("2025-10-31")
BAND, COST_EXP, COST_SEC = 0.10, 0.0005, 0.001
PERIODS = {"1927-62": ("1927", "1962"), "1963-99": ("1963", "1999"), "2000-09": ("2000", "2009"), "2010-19": ("2010", "2019"), "2020-25": ("2020", "2025-10")}


def read_daily(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    head, rows, on = None, [], False
    for line in lines:
        p = [x.strip() for x in line.split(",")]
        if p[0].isdigit() and len(p[0]) == 8:
            on = True
            rows.append(p)
        elif on:
            break
        elif len(p) > 3 and p[0] == "":
            head = p[1:]
    d = pd.DataFrame([r[1:] for r in rows], index=pd.to_datetime([r[0] for r in rows]), columns=head).astype(float)
    return d.where(d > -99) / 100


def sector_sleeve(R: pd.DataFrame, top: int = 3, look: int = 12, skip: int = 1) -> pd.Series:
    """Monthly reset. The choice for a month uses returns ending `skip` month(s) before that month starts."""
    months = R.index.to_period("M")
    mret = (1 + R.fillna(0)).groupby(months).prod() - 1
    mom = np.log1p(mret).shift(1 + skip).rolling(look, min_periods=look).sum()
    pick = mom.rank(axis=1, ascending=False, method="first").le(top) & mom.notna()
    w = pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0)
    turn = (w.fillna(0) - w.fillna(0).shift(1)).abs().sum(axis=1) / 2
    daily_w = w.reindex(months).set_axis(R.index)
    gross = (daily_w * R).sum(axis=1).where(daily_w.notna().any(axis=1))
    fee = turn.reindex(months).set_axis(R.index).where(pd.Series(months, index=R.index).ne(pd.Series(months, index=R.index).shift(1)), 0.0).fillna(0.0)
    return gross - fee * COST_SEC * 2


def apply_exposure(r: pd.Series, rf: pd.Series, target=0.17, cap=1.5, win=20, delay=1, spread=0.01) -> tuple[pd.Series, pd.Series]:
    sig = r.rolling(win, min_periods=win).std() * np.sqrt(252)
    tv = (target / sig).clip(upper=cap).shift(delay).fillna(1.0).to_numpy()
    held, out, pos = 1.0, np.zeros(len(r)), np.zeros(len(r))
    x, f, sp = r.to_numpy(), rf.reindex(r.index).to_numpy(), spread / 252
    for i in range(len(r)):
        pos[i] = held
        out[i] = held * x[i] + (1 - held) * f[i] - max(held - 1, 0) * sp
        if abs(tv[i] - held) > BAND:
            out[i] -= abs(tv[i] - held) * COST_EXP
            held = tv[i]
    return pd.Series(out, index=r.index), pd.Series(pos, index=r.index)


def stat(r, rf):
    n = len(r) / 252
    c = (1 + r).cumprod()
    return {"cagr": 100 * (c.iloc[-1] ** (1 / n) - 1), "vol": 100 * r.std() * np.sqrt(252), "sharpe": (r - rf.reindex(r.index)).mean() / r.std() * np.sqrt(252), "maxdd": 100 * (c / c.cummax() - 1).min()}


def boot(x, block=252, n=4000):
    rng, v, T = np.random.default_rng(3), x.to_numpy(), len(x)
    k = int(np.ceil(T / block))
    s = rng.integers(0, T - block + 1, (n, k))
    cum = np.concatenate([[0.0], np.cumsum(v)])
    m = (cum[s + block] - cum[s]).sum(axis=1) / (k * block) * 25200
    return np.percentile(m, 2.5), np.percentile(m, 97.5), (m > 0).mean() * 100


ind, fac = Path(sys.argv[1]), Path(sys.argv[2])
R = read_daily(ind / "10_Industry_Portfolios_daily_CSV.zip")
with zipfile.ZipFile(fac / "F-F_Research_Data_Factors_daily_CSV.zip") as z:
    text = z.read(z.namelist()[0]).decode("latin-1")
rows = [(pd.Timestamp(p[0]), float(p[1]), float(p[4])) for p in ([x.strip() for x in line.split(",")] for line in io.StringIO(text)) if len(p) >= 5 and p[0].isdigit() and len(p[0]) == 8]
d = pd.DataFrame(rows, columns=["date", "mkt_rf", "rf"]).set_index("date") / 100
d["mkt"] = d["mkt_rf"] + d["rf"]
idx = d.index[(d.index <= END)].intersection(R.index)
d, R = d.loc[idx], R.loc[idx]
sec = sector_sleeve(R).dropna()
idx = sec.index
d, sec = d.loc[idx], sec
mkt, rf = d["mkt"], d["rf"]
print(f"{idx[0].date()}..{idx[-1].date()}; market vol through 2018: {mkt[:'2018'].std() * np.sqrt(252) * 100:.1f}%")

cand = {"market": mkt, "sector sleeve alone (top 3 of 10)": sec, "half market, half sector": 0.5 * mkt + 0.5 * sec}
cand["vol target on market"], _ = apply_exposure(mkt, rf)
cand["THE PROGRAM (half sector + vol target)"], pos = apply_exposure(0.5 * mkt + 0.5 * sec, rf)
print(f"average exposure of the program: {pos.mean():.2f}; share of days above 1x: {(pos > 1).mean():.0%}; below 0.75x: {(pos < 0.75).mean():.0%}")
for name, (a, b) in {"1927-2025": ("1927", "2025-10"), **PERIODS}.items():
    print(f"\n=== {name}\n", pd.DataFrame({k: stat(v[a:b], rf) for k, v in cand.items()}).T.round(2).to_string())

print("\n=== against the market, whole period")
out = {}
for k, v in cand.items():
    if k == "market":
        continue
    e = np.log1p(v) - np.log1p(mkt)
    lo, hi, p = boot(e)
    out[k] = {"excess_log": e.mean() * 25200, "lo": lo, "hi": hi, "P>0": p, **{f"beat_{y}y": e.rolling(252 * y).sum().dropna().gt(0).mean() * 100 for y in (3, 5, 10, 20)}}
print(pd.DataFrame(out).T.round(2).to_string())
y = cand["THE PROGRAM (half sector + vol target)"]
yr = pd.DataFrame({"market": mkt.groupby(mkt.index.year).apply(lambda x: (1 + x).prod() - 1) * 100, "program": y.groupby(y.index.year).apply(lambda x: (1 + x).prod() - 1) * 100})
yr["diff"] = yr.program - yr.market
print("\nyears ahead:", {k: f"{(yr.loc[a:b]['diff'] > 0).mean():.0%}" for k, (a, b) in [("all", (1927, 2025)), ("since 2000", (2000, 2025)), ("since 2010", (2010, 2025))]})
print(yr.loc[2015:].round(1).T.to_string())

print("\n=== is it just more market exposure? regression on the market with Newey-West t (daily)")
for label, m in {"whole": np.ones(len(y), bool), "1963-99": (y.index.year >= 1963) & (y.index.year <= 1999), "2000-25": y.index.year >= 2000, "2010-25": y.index.year >= 2010}.items():
    X = np.column_stack([np.ones(m.sum()), d["mkt_rf"].to_numpy()[m]])
    z = (y - rf).to_numpy()[m]
    b = np.linalg.lstsq(X, z, rcond=None)[0]
    e = z - X @ b
    T, u = m.sum(), X * e[:, None]
    S = u.T @ u / T
    for l in range(1, 21):
        G = u[l:].T @ u[:-l] / T
        S += (1 - l / 21) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / T)
    print(f"{label:9s} beta {b[1]:.2f}  alpha {b[0] * 25200:+.2f}%/yr  t {b[0] / np.sqrt((Q @ S @ Q / T)[0, 0]):+.2f}")

print("\n=== sensitivity of the program's excess CAGR over the market (points)")
base = stat(mkt, rf)["cagr"]
rows = {}
for label, kw, f in [("reference", {}, 0.5), ("borrow T-bill+2%", {"spread": 0.02}, 0.5), ("act 2 days late", {"delay": 2}, 0.5), ("act same day", {"delay": 0}, 0.5), ("target 15%", {"target": 0.15}, 0.5), ("target 20%", {"target": 0.20}, 0.5),
                     ("cap 1.25x", {"cap": 1.25}, 0.5), ("cap 2x", {"cap": 2.0}, 0.5), ("window 10", {"win": 10}, 0.5), ("window 60", {"win": 60}, 0.5), ("sector 25%", {}, 0.25), ("sector 100%", {}, 1.0), ("no exposure step (cap 1, target 99%)", {"target": 9.9, "cap": 1.0}, 0.5)]:
    r, _ = apply_exposure((1 - f) * mkt + f * sec, rf, **kw)
    rows[label] = {"ex_all": stat(r, rf)["cagr"] - base, "sharpe": stat(r, rf)["sharpe"], "maxdd": stat(r, rf)["maxdd"], **{k: stat(r[a:b], rf)["cagr"] - stat(mkt[a:b], rf)["cagr"] for k, (a, b) in PERIODS.items()}}
print(pd.DataFrame(rows).T.round(2).to_string())

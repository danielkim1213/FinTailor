"""Pre-registered variants of the allocation program, on a century of daily data and on real ETFs.

The reference program (fintailor/strategy/allocation.py): half market, half the top 3 industries by 12-1 momentum reset monthly,
scaled to 17% volatility from the last 20 days, capped at 1.5x, acted on one day late, 10-point band.

Variants were written down before any of them was run. Each changes one thing:
  ewma        volatility from an exponentially weighted estimator (half-life 10 days) instead of a 20-day sample standard deviation
  trend_cap   borrowing allowed only while the market is above its 200-day average; below it the cap is 1.0 (never below the
              vol target, never out of the market because of the trend)
  tsmom_cap   same gate, using the market's 12-month return > 0 instead of the 200-day average
  floor_50    exposure never below 0.5 (the vol target alone can go near 0.3 in a crash and then miss the recovery)
  futures     borrowing at T-bill + 0.4%/yr (index futures / cheap margin) instead of + 1%
  lev_only    no vol target: 1.25x above the 200-day average, 1.0 below (is the vol target worth anything on top of a trend gate?)
  combo       ewma + trend_cap
  combo_fut   ewma + trend_cap + futures financing
A grid over the trend window (100..300 days) shows whether the gate depends on one lucky length.

Judging rule (fixed in advance): a variant replaces the reference only if it has a higher Sharpe AND a shallower or equal maximum
drawdown on the century data, AND its excess over the market is positive in at least as many of the five eras, AND it is not worse
on the real ETF data. Costs: 10bp on sector turnover, 5bp on exposure changes. Data after 2025-10-31 excluded (locked holdout).

Usage: python -I research/alloc_v2.py <industry_daily_dir> <factors_dir> <etf_dir>
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_rows", 200, "display.float_format", lambda v: f"{v:.2f}")
END = pd.Timestamp("2025-10-31")
COST_EXP, COST_SEC, BAND = 0.0005, 0.001, 0.10
ERAS = {"1927-62": ("1927", "1962"), "1963-99": ("1963", "1999"), "2000-09": ("2000", "2009"), "2010-19": ("2010", "2019"), "2020-25": ("2020", "2025-10")}
ETF_ERAS = {"1999-09": ("1999", "2009"), "2010-19": ("2010", "2019"), "2020-25": ("2020", "2025-10")}
SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]

REF = dict(vol="sma", vol_days=20, hl=10.0, target=0.17, cap=1.5, trend_ma=0, tsmom=0, floor=0.0, spread=0.01, delay=1, const_lev=None)
VARIANTS = {
    "reference": {},
    "ewma": {"vol": "ewma"},
    "trend_cap": {"trend_ma": 200},
    "tsmom_cap": {"tsmom": 252},
    "floor_50": {"floor": 0.5},
    "futures": {"spread": 0.004},
    "lev_only": {"const_lev": 1.25, "trend_ma": 200},
    "combo": {"vol": "ewma", "trend_ma": 200},
    "combo_fut": {"vol": "ewma", "trend_ma": 200, "spread": 0.004},
    "combo_floor": {"vol": "ewma", "trend_ma": 200, "floor": 0.5},
}


# ----------------------------------------------------------------------------------------------------------------- data
def read_french_daily(path: Path) -> pd.DataFrame:
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


def read_factors_daily(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    rows = [(pd.Timestamp(p[0]), float(p[1]), float(p[4])) for p in ([x.strip() for x in line.split(",")] for line in io.StringIO(text))
            if len(p) >= 5 and p[0].isdigit() and len(p[0]) == 8]
    d = pd.DataFrame(rows, columns=["date", "mkt_rf", "rf"]).set_index("date") / 100
    d["mkt"] = d["mkt_rf"] + d["rf"]
    return d


# ----------------------------------------------------------------------------------------------------------------- engine
def sector_sleeve(R: pd.DataFrame, top: int = 3, look: int = 12, skip: int = 1) -> pd.Series:
    """Daily net return of the equal-weight top-`top` sectors by 12-1 momentum, reset at each month start (as in final_study)."""
    months = R.index.to_period("M")
    mret = (1 + R.fillna(0)).groupby(months).prod() - 1
    mret = mret.where(R.notna().groupby(months).sum() > 0)
    mom = np.log1p(mret).shift(1 + skip).rolling(look, min_periods=look).sum()
    pick = mom.rank(axis=1, ascending=False, method="first").le(top) & mom.notna()
    w = pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0)
    turn = (w.fillna(0) - w.fillna(0).shift(1)).abs().sum(axis=1) / 2
    daily_w = w.reindex(months).set_axis(R.index)
    gross = (daily_w * R.fillna(0)).sum(axis=1).where(daily_w.notna().any(axis=1))
    first = pd.Series(months, index=R.index).ne(pd.Series(months, index=R.index).shift(1))
    fee = turn.reindex(months).set_axis(R.index).where(first, 0.0).fillna(0.0)
    return gross - fee * COST_SEC * 2


def target_exposure(book: pd.Series, core: pd.Series, s: dict) -> pd.Series:
    """Exposure wanted at each close, from information through that close."""
    if s["vol"] == "ewma":
        sig = np.sqrt(book.pow(2).ewm(halflife=s["hl"], min_periods=s["vol_days"]).mean() * 252)
    else:
        sig = book.rolling(s["vol_days"], min_periods=s["vol_days"]).std() * np.sqrt(252)
    cap = pd.Series(s["cap"], index=book.index)
    gate = None
    if s["trend_ma"]:  # same convention as allocation.leverage_cap: no gate before the average exists
        level = (1 + core).cumprod()
        ma = level.rolling(s["trend_ma"], min_periods=s["trend_ma"]).mean()
        gate = (level > ma) | ma.isna()
    if s["tsmom"]:
        tot = (1 + core).rolling(s["tsmom"], min_periods=s["tsmom"]).apply(np.prod, raw=True)
        gate = (tot > 1.0) | tot.isna()
    if gate is not None:
        cap = cap.where(gate, 1.0)
    if s["const_lev"] is not None:  # no vol target: fixed leverage, gated by the trend
        return cap.where(cap <= 1.0, s["const_lev"])
    tv = (s["target"] / sig).clip(upper=cap)
    if s["floor"]:
        tv = tv.clip(lower=s["floor"])
    return tv


def run(book: pd.Series, core: pd.Series, rf: pd.Series, s: dict) -> tuple[pd.Series, pd.Series]:
    """Daily net return and position. A target computed at close t-delay is implemented after close t (effective t+1)."""
    tv = target_exposure(book, core, s).shift(s["delay"]).fillna(1.0).to_numpy()
    x, f, sp = book.to_numpy(), rf.reindex(book.index).ffill().fillna(0.0).to_numpy(), s["spread"] / 252
    held, out, pos = 1.0, np.zeros(len(book)), np.zeros(len(book))
    for i in range(len(book)):
        pos[i] = held
        out[i] = held * x[i] + (1 - held) * f[i] - max(held - 1, 0) * sp
        if abs(tv[i] - held) > BAND:
            out[i] -= abs(tv[i] - held) * COST_EXP
            held = tv[i]
    return pd.Series(out, index=book.index), pd.Series(pos, index=book.index)


def stat(r: pd.Series, rf: pd.Series) -> dict:
    n = len(r) / 252
    c = (1 + r).cumprod()
    rfx = rf.reindex(r.index).ffill().fillna(0.0)
    return {"cagr": 100 * (c.iloc[-1] ** (1 / n) - 1), "vol": 100 * r.std() * np.sqrt(252), "sharpe": (r - rfx).mean() / r.std() * np.sqrt(252),
            "maxdd": 100 * (c / c.cummax() - 1).min()}


def boot(x: pd.Series, block: int = 252, n: int = 4000) -> tuple[float, float, float]:
    rng, v, T = np.random.default_rng(3), x.to_numpy(), len(x)
    k = int(np.ceil(T / block))
    s = rng.integers(0, T - block + 1, (n, k))
    cum = np.concatenate([[0.0], np.cumsum(v)])
    m = (cum[s + block] - cum[s]).sum(axis=1) / (k * block) * 25200
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m > 0).mean() * 100)


def table(core: pd.Series, sleeve: pd.Series, rf: pd.Series, eras: dict, variants: dict, label: str) -> pd.DataFrame:
    book = (0.5 * core + 0.5 * sleeve).dropna()
    core, rf = core.reindex(book.index), rf
    base = stat(core, rf)
    rows = {"market / SPY": {**base, "avg_exp": 1.0, "ex_log": 0.0, "lo": 0.0, "hi": 0.0, "P>0": np.nan,
                             **{k: 0.0 for k in eras}, "eras+": np.nan, "yrs_ahead%": np.nan}}
    for name, ch in variants.items():
        s = {**REF, **ch}
        r, pos = run(book, core, rf, s)
        e = np.log1p(r) - np.log1p(core)
        lo, hi, p = boot(e)
        row = {**stat(r, rf), "avg_exp": pos.mean(), "ex_log": e.mean() * 25200, "lo": lo, "hi": hi, "P>0": p}
        per = {k: stat(r[a:b], rf)["cagr"] - stat(core[a:b], rf)["cagr"] for k, (a, b) in eras.items() if len(r[a:b]) > 252}
        row.update(per)
        row["eras+"] = sum(v > 0 for v in per.values())
        yr = pd.DataFrame({"m": core.groupby(core.index.year).apply(lambda x: (1 + x).prod() - 1), "p": r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)})
        row["yrs_ahead%"] = 100 * (yr["p"] > yr["m"]).mean()
        rows[name] = row
    out = pd.DataFrame(rows).T
    print(f"\n=== {label}: {book.index[0].date()}..{book.index[-1].date()}  (excess columns = CAGR points over the market in that era)")
    print(out.round(2).to_string())
    ledger = Path(__file__).resolve().parents[1] / "store/research/trials.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a", encoding="utf-8") as f:  # every variant counts as one more idea tried
        for name, row in out.iterrows():
            if name != "market / SPY":
                f.write(json.dumps({"at": pd.Timestamp.now().isoformat(timespec="seconds"), "name": f"alloc_v2:{name}", "data": label,
                                    **{k: (None if pd.isna(v) else float(v)) for k, v in row.items()}}) + "\n")
    return out


def trend_grid(core: pd.Series, sleeve: pd.Series, rf: pd.Series, label: str) -> None:
    book = (0.5 * core + 0.5 * sleeve).dropna()
    core = core.reindex(book.index)
    rows = {}
    for ma in (100, 150, 200, 250, 300):
        for vol in ("sma", "ewma"):
            r, pos = run(book, core, rf, {**REF, "vol": vol, "trend_ma": ma})
            rows[f"ma{ma} {vol}"] = {**stat(r, rf), "avg_exp": pos.mean(), "ex_log": (np.log1p(r) - np.log1p(core)).mean() * 25200}
    for lb in (126, 189, 252):
        r, pos = run(book, core, rf, {**REF, "tsmom": lb})
        rows[f"tsmom{lb}"] = {**stat(r, rf), "avg_exp": pos.mean(), "ex_log": (np.log1p(r) - np.log1p(core)).mean() * 25200}
    print(f"\n--- {label}: gate robustness (all settings, nothing picked)")
    print(pd.DataFrame(rows).T.round(2).to_string())


def main() -> None:
    ind, fac, etf = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    # century of data
    R = read_french_daily(ind / "10_Industry_Portfolios_daily_CSV.zip")
    d = read_factors_daily(fac / "F-F_Research_Data_Factors_daily_CSV.zip")
    idx = d.index[d.index <= END].intersection(R.index)
    d, R = d.loc[idx], R.loc[idx]
    sleeve = sector_sleeve(R)
    table(d["mkt"], sleeve, d["rf"], ERAS, VARIANTS, "KEN FRENCH, market + 10 industries")
    trend_grid(d["mkt"], sleeve, d["rf"], "KEN FRENCH")
    # real ETFs
    close = pd.read_parquet(etf / "etf_prices.parquet")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close[close.index <= END].dropna(subset=["SPY"])
    r = close.pct_change(fill_method=None)
    sl = sector_sleeve(r[[c for c in SECTOR_ETFS if c in r]])
    rf = d["rf"]
    table(r["SPY"].dropna(), sl, rf, ETF_ERAS, VARIANTS, "REAL ETFs, SPY + SPDR sectors (fund fees inside prices)")
    trend_grid(r["SPY"].dropna(), sl, rf, "REAL ETFs")


if __name__ == "__main__":
    main()

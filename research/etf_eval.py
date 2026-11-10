"""The allocation program on real, investable ETF prices since 1999 (total return, with fund fees already inside the prices).

Settings are the ones fixed in fintailor/strategy/allocation.py. Prices after 2025-10-31 are excluded (locked holdout).
Usage: python -I research/etf_eval.py <folder_for_prices> <factors_directory> [download]
"""
from __future__ import annotations

import io
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fintailor.strategy.allocation import SECTOR_ETFS, Settings, book_returns, scale  # noqa: E402

pd.set_option("display.width", 240, "display.max_columns", 40, "display.float_format", lambda v: f"{v:.2f}")
END = pd.Timestamp("2025-10-31")
PERIODS = {"1999-2009": ("1999", "2009"), "2010-2019": ("2010", "2019"), "2020-2025": ("2020", "2025-10"), "2019-2025": ("2019", "2025-10")}

folder, fac = Path(sys.argv[1]), Path(sys.argv[2])
path = folder / "etf_prices.parquet"
if len(sys.argv) > 3 and sys.argv[3] == "download":
    import yfinance as yf
    folder.mkdir(parents=True, exist_ok=True)
    d = yf.download(["SPY", *SECTOR_ETFS], start="1998-12-22", end="2026-11-07", auto_adjust=True, progress=False, group_by="column")["Close"]
    d.to_parquet(path)
close = pd.read_parquet(path)
close.index = pd.to_datetime(close.index).tz_localize(None)
close = close[close.index <= END].dropna(subset=["SPY"])
print("prices", close.index[0].date(), close.index[-1].date(), close.notna().sum().to_dict())

with zipfile.ZipFile(fac / "F-F_Research_Data_Factors_daily_CSV.zip") as z:
    text = z.read(z.namelist()[0]).decode("latin-1")
rows = [(pd.Timestamp(p[0]), float(p[4])) for p in ([x.strip() for x in line.split(",")] for line in io.StringIO(text)) if len(p) >= 5 and p[0].isdigit() and len(p[0]) == 8]
rf = pd.DataFrame(rows, columns=["date", "rf"]).set_index("date")["rf"] / 100


def stat(r):
    c = (1 + r).cumprod()
    rfx = rf.reindex(r.index).ffill().fillna(0.0)
    return {"cagr": 100 * (c.iloc[-1] ** (252 / len(r)) - 1), "vol": 100 * r.std() * np.sqrt(252), "sharpe": (r - rfx).mean() / r.std() * np.sqrt(252), "maxdd": 100 * (c / c.cummax() - 1).min(), "worst_yr": 100 * r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1).min()}


def boot(x, block=252, n=4000):
    rng, v, T = np.random.default_rng(3), x.to_numpy(), len(x)
    k = int(np.ceil(T / block))
    s = rng.integers(0, T - block + 1, (n, k))
    cum = np.concatenate([[0.0], np.cumsum(v)])
    m = (cum[s + block] - cum[s]).sum(axis=1) / (k * block) * 25200
    return np.percentile(m, 2.5), np.percentile(m, 97.5), (m > 0).mean() * 100


def program(s: Settings):
    b = book_returns(close, s).dropna(subset=["book"])
    r = scale(b["book"], rf, s, b["core"])
    return r, b


s0 = Settings()
base, parts = program(s0)
start = parts["sleeve"].first_valid_index()
idx = parts.loc[start:].index
spy = close["SPY"].pct_change().reindex(idx).fillna(0.0)
curves = {"SPY": spy, "half SPY, half sector": parts["book"].reindex(idx), "sector sleeve alone": parts["sleeve"].reindex(idx), "THE PROGRAM": base["ret"].reindex(idx)}
print("first investable month:", start.date(), "| average exposure", round(base["exposure"].reindex(idx).mean(), 2))
for name, (a, b) in {"whole": (str(idx[0].year), "2025-10"), **PERIODS}.items():
    print(f"\n=== {name}\n", pd.DataFrame({k: stat(v[a:b]) for k, v in curves.items()}).T.round(2).to_string())
e = np.log1p(curves["THE PROGRAM"]) - np.log1p(spy)
lo, hi, p = boot(e)
print(f"\nprogram minus SPY: {e.mean() * 25200:+.2f}%/yr log, 95% interval [{lo:+.2f}, {hi:+.2f}], P(>0) {p:.1f}%")
print("beat SPY over rolling windows:", {f"{y}y": f"{e.rolling(252 * y).sum().dropna().gt(0).mean():.0%}" for y in (1, 3, 5, 10)})
yr = pd.DataFrame({"SPY": spy.groupby(spy.index.year).apply(lambda x: (1 + x).prod() - 1) * 100, "program": curves["THE PROGRAM"].groupby(curves["THE PROGRAM"].index.year).apply(lambda x: (1 + x).prod() - 1) * 100})
yr["diff"] = yr.program - yr.SPY
print("\n", yr.round(1).T.to_string())

print("\n=== settings, all against SPY")
table = {}
for label, ch in [("reference", {}), ("no trend gate (pre-2026-11-10 program)", {"trend_ma": 0}), ("no borrowing (max 1x)", {"max_exposure": 1.0}), ("max 1.25x", {"max_exposure": 1.25}), ("max 2x", {"max_exposure": 2.0}), ("target 15%", {"vol_target": 0.15}), ("target 20%", {"vol_target": 0.20}), ("sector 25%", {"sector_share": 0.25}), ("sector 100%", {"sector_share": 1.0}), ("top 2", {"top": 2}), ("top 4", {"top": 4}), ("6-month momentum", {"look_months": 6}), ("9-month momentum", {"look_months": 9}), ("no skip month", {"skip_months": 0}), ("borrow +2%", {"borrow_spread": 0.02}), ("borrow +0.4% (futures)", {"borrow_spread": 0.004}), ("vol 60 days", {"vol_days": 60}), ("2 days later", {"act_days_late": 2}), ("trend 100 days", {"trend_ma": 100}), ("trend 300 days", {"trend_ma": 300})]:
    r, _ = program(replace(s0, **ch))
    r = r["ret"].reindex(idx).dropna()
    z = stat(r)
    table[label] = {"ex_cagr": z["cagr"] - stat(spy.reindex(r.index))["cagr"], "vol": z["vol"], "sharpe": z["sharpe"], "maxdd": z["maxdd"],
                    **{k: stat(r[a:b])["cagr"] - stat(spy[a:b])["cagr"] for k, (a, b) in PERIODS.items()}}
print(pd.DataFrame(table).T.round(2).to_string())

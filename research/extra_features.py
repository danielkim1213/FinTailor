"""Extra point-in-time stock features for the research bench (x_ prefix). Every value uses data up to that day's close.

Run:  python research/extra_features.py   ->  store/research/extra_features.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fintailor.config import get_config  # noqa: E402
from fintailor.utils import assign_trading_date  # noqa: E402


def build() -> pd.DataFrame:
    con = duckdb.connect(str(ROOT / "store/fintailor.duckdb"), read_only=True)
    px = con.execute("select ticker, date, adj_close, close, volume from prices_daily").df()
    earn = con.execute("select ticker, date, announced_at, eps_actual, eps_estimate from earnings where eps_actual is not null").df()
    con.close()
    px["date"] = pd.to_datetime(px["date"])
    C = px.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").sort_index()
    V = px.pivot_table(index="date", columns="ticker", values="volume", aggfunc="last").reindex_like(C)
    R = px.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").reindex_like(C)
    r = C.pct_change(fill_method=None)
    spy = r["SPY"]
    F: dict[str, pd.DataFrame] = {}

    # momentum at other lookbacks, and with the market removed
    F["x_mom_6_1"] = C.shift(21) / C.shift(126) - 1
    F["x_mom_3_1"] = C.shift(21) / C.shift(63) - 1
    F["x_mom_36_12"] = C.shift(252) / C.shift(756) - 1
    mv = spy.rolling(252, min_periods=126).var()
    beta = r.mul(spy, axis=0).rolling(252, min_periods=126).mean().sub(
        r.rolling(252, min_periods=126).mean().mul(spy.rolling(252, min_periods=126).mean(), axis=0)).div(mv, axis=0)
    res = r.sub(beta.shift(1).mul(spy, axis=0))
    sd = res.rolling(252, min_periods=126).std()
    F["x_resid_mom_12_1"] = res.shift(21).rolling(231, min_periods=126).sum() / sd
    F["x_resid_mom_6_1"] = res.shift(21).rolling(105, min_periods=63).sum() / sd
    F["x_resid_rev_5"] = -res.rolling(5, min_periods=4).sum() / sd
    F["x_resid_rev_20"] = -res.rolling(20, min_periods=15).sum() / sd
    F["x_mom_consistency"] = (r.shift(21) > 0).rolling(231, min_periods=126).mean()

    # risk shape and lottery-like behaviour
    F["x_vol_252"] = r.rolling(252, min_periods=126).std()
    F["x_skew_60"] = r.rolling(60, min_periods=40).skew()
    F["x_resid_skew_60"] = res.rolling(60, min_periods=40).skew()
    F["x_max5_20"] = -r.rolling(20, min_periods=15).apply(lambda v: np.sort(v)[-5:].mean(), raw=True)
    down = r.where(spy < 0)
    F["x_down_beta"] = down.rolling(252, min_periods=60).cov(spy.where(spy < 0)) / spy.where(spy < 0).rolling(252, min_periods=60).var()

    # liquidity and attention
    dv = (R * V).where(lambda d: d > 0)
    F["x_amihud_20"] = np.log1p(1e9 * (r.abs() / dv).rolling(20, min_periods=15).mean())
    F["x_vol_trend"] = np.log(V.rolling(5, min_periods=4).mean() / V.rolling(60, min_periods=40).mean())
    F["x_dvol_chg_60"] = np.log(dv.rolling(20, min_periods=15).mean() / dv.rolling(120, min_periods=80).mean())

    # price anchors
    F["x_dist_52w_low"] = C / C.rolling(252, min_periods=126).min() - 1
    F["x_dist_3y_high"] = C / C.rolling(756, min_periods=252).max() - 1

    # the same calendar month in past years (5-year average of that month's return)
    me = C.resample("ME").last().pct_change(fill_method=None)
    # A month-end row is what days of the following month see, so shifting by 12k-1 lands on that next month k years ago.
    seas = pd.concat([me.shift(12 * k - 1) for k in range(1, 6)]).groupby(level=0).mean()
    F["x_season_month"] = seas.reindex(C.index, method="ffill")

    # earnings drift: surprise scaled by the stock's own surprise history, and the price reaction to the last report
    earn["date"] = pd.to_datetime(earn["date"])
    earn["announced_at"] = pd.to_datetime(earn["announced_at"]).fillna(earn["date"] + pd.Timedelta(hours=21))
    earn["tdate"] = assign_trading_date(earn["announced_at"], C.index, get_config()["news"]["cutoff_et"])
    earn = earn.dropna(subset=["tdate", "eps_estimate"]).sort_values(["ticker", "tdate"])
    earn["raw"] = earn["eps_actual"] - earn["eps_estimate"]
    g = earn.groupby("ticker")["raw"]
    earn["sue"] = (earn["raw"] / g.transform(lambda s: s.shift(1).rolling(8, min_periods=3).std()).replace(0, np.nan)).clip(-10, 10)
    earn["beat_streak"] = g.transform(lambda s: (s > 0).astype(float).rolling(4, min_periods=2).sum())
    excess = r.sub(spy, axis=0)
    earn["react"] = [excess.at[d, t] if t in excess.columns else np.nan for t, d in zip(earn["ticker"], earn["tdate"])]
    for col, name in (("sue", "x_sue"), ("beat_streak", "x_beat_streak"), ("react", "x_earn_react")):
        w = earn.pivot_table(index="tdate", columns="ticker", values=col, aggfunc="last")
        F[name] = w.reindex(C.index).reindex(columns=C.columns).ffill(limit=63)

    stacked = [w.reindex(columns=C.columns).stack(future_stack=True).rename(k).astype("float32") for k, w in F.items()]
    return pd.concat(stacked, axis=1).dropna(how="all").rename_axis(["date", "ticker"]).reset_index()


if __name__ == "__main__":
    out = ROOT / "store/research"
    out.mkdir(parents=True, exist_ok=True)
    df = build()
    df = df[df["date"] < pd.Timestamp("2025-11-01")]
    df.to_parquet(out / "extra_features.parquet", index=False)
    print(df.shape, [c for c in df.columns if c.startswith("x_")])

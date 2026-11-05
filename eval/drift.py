"""Distribution drift (PSI). Cross-sectional ranks are uniform by construction, so drift is
measured on market-level features, missingness, and the prediction distribution."""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) < 50 or len(cur) < 5:
        return np.nan
    qs = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    qs[0], qs[-1] = -np.inf, np.inf
    r = np.histogram(ref, qs)[0] / len(ref) + 1e-4
    c = np.histogram(cur, qs)[0] / len(cur) + 1e-4
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(panel: pd.DataFrame, run_date: pd.Timestamp, ref_days: int = 504, cur_days: int = 20) -> dict:
    meta = panel.attrs["meta"]
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    pos = dates.searchsorted(run_date, side="right")
    cur_d = dates[max(0, pos - cur_days):pos]
    ref_d = dates[max(0, pos - cur_days - ref_days):max(0, pos - cur_days)]
    by_date = panel.groupby("date")[meta["market"]].first() if meta["market"] else pd.DataFrame()
    out = {"market_psi": {}, "missing_jump": {}}
    for f in meta["market"]:
        v = psi(by_date.loc[by_date.index.isin(ref_d), f].to_numpy(float), by_date.loc[by_date.index.isin(cur_d), f].to_numpy(float))
        if np.isfinite(v):
            out["market_psi"][f] = round(v, 3)
    cur = panel[panel["date"].isin(cur_d)]
    ref = panel[panel["date"].isin(ref_d)]
    for f in meta["features"]:
        a, b = ref[f].isna().mean(), cur[f].isna().mean()
        if abs(b - a) > 0.2:
            out["missing_jump"][f] = (round(float(a), 2), round(float(b), 2))
    out["alerts"] = [f for f, v in out["market_psi"].items() if v > 0.25] + list(out["missing_jump"])
    return out

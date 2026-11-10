"""Evaluation metrics on out-of-sample predictions.

Input frame columns: date, ticker, horizon, score, [prob], y (net excess), g (gross excess), [gap], [vix_bucket].
Strategy = each day buy the top-K by score, hold h days, measured vs SPY (excess).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rank_ic(df: pd.DataFrame, score: str = "score", target: str = "g", min_names: int = 6) -> pd.Series:
    """Daily Spearman correlation between score and target (Pearson on within-day ranks), one value per date.
    Vectorised: the per-day Python callback this replaced took most of a backtest's evaluation time."""
    d = df[[score, target, "date"]].dropna()
    if d.empty:
        return pd.Series(dtype=float)
    dates = d["date"].to_numpy()
    r = d.groupby("date")[[score, target]].rank()
    g = r.groupby(dates)
    x, v = r[score] - g[score].transform("mean"), r[target] - g[target].transform("mean")
    num = (x * v).groupby(dates).sum()
    den = np.sqrt((x ** 2).groupby(dates).sum() * (v ** 2).groupby(dates).sum())
    n = x.groupby(dates).size()
    ic = (num / den.replace(0, np.nan))[n >= min_names].dropna()
    ic.index = pd.DatetimeIndex(ic.index)
    return ic


def top_k(df: pd.DataFrame, k: int, score: str = "score") -> pd.DataFrame:
    d = df.dropna(subset=[score, "y"])
    return d[d.groupby("date")[score].rank(ascending=False, method="first") <= k]


def _sharpe(cohort: pd.Series, h: int) -> float:
    if len(cohort) < 10 or cohort.std() == 0:
        return np.nan
    return float(cohort.mean() / cohort.std() * np.sqrt(252 / h))


def _max_dd(cohort: pd.Series, h: int) -> float:
    """Average max drawdown over the h non-overlapping sub-schedules (offset 0..h-1)."""
    dds = []
    vals = cohort.values
    for off in range(min(h, len(vals))):
        eq = np.cumprod(1 + vals[off::h])
        if len(eq) < 2:
            continue
        dds.append(float((eq / np.maximum.accumulate(eq) - 1).min()))
    return float(np.mean(dds)) if dds else np.nan


def calibration_table(df: pd.DataFrame, bins=(0, .45, .5, .55, .6, .65, 1.0)) -> pd.DataFrame:
    d = df.dropna(subset=["prob", "y"])
    if d.empty:
        return pd.DataFrame()
    b = pd.cut(d["prob"], bins)
    t = d.groupby(b, observed=True).agg(n=("y", "size"), mean_prob=("prob", "mean"),
                                        hit=("y", lambda s: (s > 0).mean()))
    t["gap_pp"] = (t["hit"] - t["mean_prob"]) * 100
    return t.reset_index().rename(columns={"prob": "bin"}).assign(bin=lambda x: x["bin"].astype(str))


def evaluate(df: pd.DataFrame, h: int, k: int = 10, cost: float = 0.001, score: str = "score",
             by_regime: bool = True) -> dict:
    """Core report for one horizon."""
    d = df[df["y"].notna()]
    if d.empty:
        return {"n_days": 0}
    ic = rank_ic(d, score)
    tk = top_k(d, k, score)
    cohort = tk.groupby("date")["y"].mean()
    wins, losses = tk["y"][tk["y"] > 0], tk["y"][tk["y"] <= 0]
    t_stat = cohort.mean() / (cohort.std() / np.sqrt(max(len(cohort) / h, 1))) if len(cohort) > 2 else np.nan
    out = {
        "horizon": h, "n_days": int(d["date"].nunique()), "n_trades": int(len(tk)),
        "net_per_trade": float(tk["y"].mean()),
        "net_per_trade_2x_cost": float(tk["y"].mean() - cost),
        "hit_rate": float((tk["y"] > 0).mean()),
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "t_stat_nonoverlap": float(t_stat),
        "rank_ic": float(ic.mean()) if len(ic) else np.nan,
        "rank_ic_ir": float(ic.mean() / ic.std() * np.sqrt(252 / h)) if len(ic) > 2 and ic.std() > 0 else np.nan,
        "ic_hit": float((ic > 0).mean()) if len(ic) else np.nan,
        "sharpe": _sharpe(cohort, h),
        "max_drawdown": _max_dd(cohort, h),
        "universe_mean_net": float(d["y"].mean()),
    }
    if "gap" in tk and tk["gap"].notna().any():
        lt = tk[tk["y"] < 0]
        tot = lt["y"].sum()
        out["gap_loss_share"] = float((np.expm1(lt["gap"]).clip(upper=0).sum() / tot)) if tot < 0 else 0.0
    if "prob" in d and d["prob"].notna().any():
        cal = calibration_table(d)
        out["calib_max_gap_pp"] = float(cal.loc[cal["n"] > 200, "gap_pp"].abs().max()) if len(cal) else np.nan
        out["brier"] = float(((d["prob"] - (d["y"] > 0)) ** 2).mean())
    if by_regime:
        years = pd.to_datetime(cohort.index).year
        out["by_year"] = {int(y): float(v) for y, v in cohort.groupby(years).mean().items()}
        out["ic_by_year"] = {int(y): float(v) for y, v in ic.groupby(pd.to_datetime(ic.index).year).mean().items()}
        if "vix_bucket" in d:
            vb = d.groupby("date")["vix_bucket"].first().reindex(cohort.index)
            out["by_vix_bucket"] = {str(int(b)): float(v) for b, v in cohort.groupby(vb).mean().items() if pd.notna(b)}
    return out


def summary_table(results: dict[str, dict]) -> pd.DataFrame:
    keys = ["n_days", "n_trades", "net_per_trade", "net_per_trade_2x_cost", "hit_rate", "avg_win", "avg_loss",
            "rank_ic", "rank_ic_ir", "sharpe", "max_drawdown", "gap_loss_share", "calib_max_gap_pp"]
    rows = {name: {k: r.get(k) for k in keys} for name, r in results.items()}
    return pd.DataFrame(rows).T


def gate_check(r: dict, k: int) -> dict:
    """Plan section 10 launch bars (per horizon). True/False per criterion."""
    return {
        "net_per_trade>0": (r.get("net_per_trade") or -1) > 0 and (r.get("t_stat_nonoverlap") or 0) > 2,
        "hit_rate>=52%": (r.get("hit_rate") or 0) >= 0.52,
        "avg_win>=|avg_loss|": (r.get("avg_win") or 0) >= abs(r.get("avg_loss") or 1),
        "rank_ic>=0.02": (r.get("rank_ic") or 0) >= 0.02,
        "sharpe>=1": (r.get("sharpe") or 0) >= 1,
        "2x_cost>0": (r.get("net_per_trade_2x_cost") or -1) > 0,
        "calibration<=5pp": (r.get("calib_max_gap_pp") if r.get("calib_max_gap_pp") is not None else 99) <= 5,
        "years_positive>=70%": np.mean([v > 0 for v in r.get("by_year", {}).values()] or [0]) >= 0.7,
    }

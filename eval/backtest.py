"""Walk-forward backtest (same purging as production) + baselines."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ..config import cost, get_config, horizons, store_dir
from ..models.lgbm import fit_horizon, ranking_score, select_features
from ..models.registry import config_overrides, train_window
from ..utils import log
from .metrics import evaluate, gate_check, summary_table
from .portfolio import portfolio_report, print_report

BASELINES = {
    "reversal_1w": lambda p: -p["ret_5"],
    "momentum_1m": lambda p: p["ret_20"],
    "momentum_12_1": lambda p: p["mom_12_1"],
}


def _frame(panel: pd.DataFrame, h: int, score: pd.Series, prob: pd.Series | None = None) -> pd.DataFrame:
    d = pd.DataFrame({"date": panel["date"].values, "ticker": panel["ticker"].values, "horizon": h,
                      "score": np.asarray(score), "y": panel[f"y_{h}"].values, "g": panel[f"g_{h}"].values,
                      "gap": panel[f"gap_{h}"].values})
    if "vix_bucket" in panel:
        d["vix_bucket"] = panel["vix_bucket"].values
    if prob is not None:
        d["prob"] = np.asarray(prob)
    return d


def run_baselines(panel: pd.DataFrame, start_test: str | None = None, hs: list[int] | None = None) -> dict:
    cfg = get_config()
    start_test = pd.Timestamp(start_test or cfg["backtest"]["start_test"])
    p = panel[panel["date"] >= start_test]
    k = cfg["backtest"]["top_k"]
    res = {}
    for h in hs or horizons():
        for name, fn in BASELINES.items():
            res[f"{name}@{h}d"] = evaluate(_frame(p, h, fn(p)), h, k, cost())
        res[f"spy_hold@{h}d"] = {"horizon": h, "net_per_trade": 0.0, "hit_rate": np.nan, "rank_ic": 0.0,
                                 "note": "benchmark: excess vs SPY is 0 by definition"}
    return res


def walk_forward(panel: pd.DataFrame, hs: list[int] | None = None, config_name: str = "base",
                 start_test: str | None = None, retrain_every: int | None = None, stride: int | None = None,
                 end_test: str | None = None) -> pd.DataFrame:
    """Out-of-sample predictions: retrain every `retrain_every` trading days on a rolling window,
    purged by horizon + embargo. Returns the concatenated OOS frame."""
    cfg = get_config()
    hs = hs or horizons()
    retrain_every = retrain_every or cfg["backtest"]["retrain_every_days"]
    ov = dict(config_overrides(config_name))
    if stride:
        ov["train_date_stride"] = stride
    feats = select_features(panel.attrs["meta"]["features"], ov)
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    s = pd.Timestamp(start_test or cfg["backtest"]["start_test"])
    first = max(dates.searchsorted(s), cfg["model"]["min_train_days"] + max(hs) + 10)
    last = len(dates) if end_test is None else dates.searchsorted(pd.Timestamp(end_test), side="right")
    blocks = list(range(first, last, retrain_every))
    out = []
    t0 = time.time()
    for bi, b in enumerate(blocks):
        d0, d1 = dates[b], dates[min(b + retrain_every, last) - 1]
        test = panel[(panel["date"] >= d0) & (panel["date"] <= d1)]
        for h in hs:
            tr = train_window(panel, d0, h)
            if tr[f"y_{h}"].notna().sum() < 1000:
                continue
            m = fit_horizon(tr, feats, h, ov)
            pr = m.predict(test)
            out.append(_frame(test, h, ranking_score(pr, ov.get("rank_by")), pr["prob"]).assign(pred_excess=pr["pred_excess"].values))
        log.info("walk-forward [%s] block %d/%d %s..%s (%.0fs)", config_name, bi + 1, len(blocks), d0.date(),
                 d1.date(), time.time() - t0)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def run_backtest(panel: pd.DataFrame, fast: bool = False, config_name: str = "base", hs: list[int] | None = None,
                 start_test: str | None = None, save: bool = True, final: bool = False) -> dict:
    cfg = get_config()
    k = cfg["backtest"]["top_k"]
    kw = {}
    if fast:
        f = cfg["backtest"]["fast"]
        hs = hs or f["horizons"]
        kw = dict(retrain_every=f["retrain_every_days"], stride=f["train_date_stride"])
    hs = hs or horizons()
    # Locked holdout: dates from holdout_start on are only evaluated with --final (once, at the end).
    # Tuning on them would turn the last honest test into one more in-sample period.
    hold = cfg["backtest"].get("holdout_start")
    end_test = None
    if hold and not final:
        end_test = str((pd.Timestamp(hold) - pd.Timedelta(days=1)).date())
        log.info("backtest ends %s (locked holdout from %s; use --final to include it)", end_test, hold)
    oos = walk_forward(panel, hs, config_name, start_test, end_test=end_test, **kw)
    res = {}
    for h in hs:
        d = oos[oos["horizon"] == h]
        if d.empty:
            continue
        r = evaluate(d, h, k, cost())
        r["gate"] = gate_check(r, k)
        res[f"model[{config_name}]@{h}d"] = r
    if not oos.empty:
        res.update(run_baselines(panel, str(oos["date"].min().date()), hs))
        if "g_5" in panel:
            rets = pd.DataFrame({"date": panel["date"].values, "ticker": panel["ticker"].values,
                                 "r": panel["g_5"].values, "vol": panel["atr_pct_raw"].values,
                                 "sector": panel["sector"].astype(str).values,
                                 "mcap": panel["mcap"].values if "mcap" in panel else np.nan,
                                 "dvol": panel["dvol20"].values})
            rets = rets[rets["date"].isin(oos["date"].unique())]
            res["_portfolio"] = portfolio_report(oos, rets, hs, cfg, cost())
    if save:
        d = store_dir("backtests")
        tag = f"{pd.Timestamp.now():%Y%m%d_%H%M%S}_{config_name}{'_fast' if fast else ''}"
        oos.to_parquet(d / f"{tag}_oos.parquet", index=False)
        (d / f"{tag}.json").write_text(json.dumps(res, indent=1, default=str))
        res["_files"] = {"oos": str(d / f"{tag}_oos.parquet"), "report": str(d / f"{tag}.json")}
    print(summary_table({k_: v for k_, v in res.items() if not k_.startswith("_")}).to_string(float_format=lambda x: f"{x:.4f}"))
    if res.get("_portfolio"):
        print_report(res["_portfolio"])
    return res

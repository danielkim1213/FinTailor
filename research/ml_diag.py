"""Why does the production learner keep only a handful of trees, and does a better-regularised fit rank stocks better?

Development window only (test blocks 2019-01 .. 2023-12, 126 trading days each); the validation window (2024-) and the locked
holdout are not touched. Every configuration is scored by the same two numbers per horizon: mean daily rank correlation with
the sector/size/beta-neutral outcome (IC) and its t-statistic across days. Trees kept (best_iteration) is printed to show
whether early stopping is doing anything sensible.

Usage: python research/ml_diag.py [config names...]   (default: all)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fintailor.models.lgbm import _neutralize, fit_horizon, ranking_score, select_features  # noqa: E402
from fintailor.models.registry import train_window  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", lambda v: f"{v:.4f}")
HOLDOUT, DEV_START, DEV_END = pd.Timestamp("2025-11-01"), pd.Timestamp("2019-01-01"), pd.Timestamp("2023-12-31")
HS = (5, 10)
EVERY = 126
BASE = {"target": "rank", "neutralize": ["sector", "size", "beta"], "drop_prefixes": ["ins_", "si_", "news_", "sector_news", "llm_sector"]}
CONFIGS = {
    "prod": {},                                                                 # the strategy's learner as it is
    "prod_s3": {"train_date_stride": 3},                                        # thinned dates (3x faster), same learner
    "reg_only": {"rank_by": "reg"},                                             # drop the un-neutralised classifier half
    "slow": {"learning_rate": 0.02, "num_leaves": 15, "min_data_in_leaf": 2000, "lambda_l2": 50.0, "feature_fraction": 0.5,
             "early_stopping_rounds": 150, "num_boost_round": 1500, "train_date_stride": 3, "rank_by": "reg"},
    "slow_fixed300": {"learning_rate": 0.02, "num_leaves": 15, "min_data_in_leaf": 2000, "lambda_l2": 50.0, "feature_fraction": 0.5,
                      "early_stop": False, "num_boost_round": 300, "train_date_stride": 3, "rank_by": "reg"},
    "slow_fixed800": {"learning_rate": 0.02, "num_leaves": 15, "min_data_in_leaf": 2000, "lambda_l2": 50.0, "feature_fraction": 0.5,
                      "early_stop": False, "num_boost_round": 800, "train_date_stride": 3, "rank_by": "reg"},
    "shallow_fixed500": {"learning_rate": 0.02, "num_leaves": 7, "max_depth": 3, "min_data_in_leaf": 3000, "lambda_l2": 100.0,
                         "feature_fraction": 0.5, "early_stop": False, "num_boost_round": 500, "train_date_stride": 3, "rank_by": "reg"},
    "slow_fixed300_s5": {"learning_rate": 0.02, "num_leaves": 15, "min_data_in_leaf": 2000, "lambda_l2": 50.0, "feature_fraction": 0.5,
                         "early_stop": False, "num_boost_round": 300, "train_date_stride": 3, "rank_by": "reg", "n_seeds": 5},
}


def daily_ic(dates: np.ndarray, score: np.ndarray, y: np.ndarray) -> pd.Series:
    d = pd.DataFrame({"date": dates, "s": score, "y": y}).dropna()
    r = d.groupby("date")[["s", "y"]].rank()
    r["date"] = d["date"].to_numpy()
    g = r.groupby("date")
    x, v = r["s"] - g["s"].transform("mean"), r["y"] - g["y"].transform("mean")
    num = (x * v).groupby(r["date"]).sum()
    den = np.sqrt((x ** 2).groupby(r["date"]).sum() * (v ** 2).groupby(r["date"]).sum())
    return (num / den.replace(0, np.nan)).dropna()


def main(names: list[str]) -> None:
    meta = json.loads((ROOT / "store/features/panel_meta.json").read_text())
    p = pd.read_parquet(ROOT / "store/features/panel.parquet")
    p = p[(p["date"] < HOLDOUT) & (~p["is_etf_flag"].astype(bool))].reset_index(drop=True)
    p["sector"] = p["sector"].astype(str)
    p.attrs["meta"] = meta
    dates = pd.DatetimeIndex(np.sort(p["date"].unique()))
    neutral = {h: _neutralize(p, f"y_{h}", BASE["neutralize"]) for h in HS}
    first = dates.searchsorted(DEV_START)
    last = dates.searchsorted(DEV_END, side="right")
    blocks = list(range(first, last, EVERY))
    print(f"panel {len(p):,} rows; {len(blocks)} dev blocks of {EVERY} days; horizons {HS}", flush=True)
    rows = {}
    for name in names or list(CONFIGS):
        ov = {**BASE, **CONFIGS[name]}
        feats = select_features(meta["features"], ov)
        t0 = time.time()
        res = {}
        for h in HS:
            ics, iters = [], []
            for b in blocks:
                d0, d1 = dates[b], dates[min(b + EVERY, last) - 1]
                test = p[(p["date"] >= d0) & (p["date"] <= d1)]
                m = fit_horizon(train_window(p, d0, h), feats, h, ov)
                sc = ranking_score(m.predict(test), ov.get("rank_by")).to_numpy()
                ics.append(daily_ic(test["date"].to_numpy(), sc, neutral[h][test.index.to_numpy()]))
                iters.append((m.info["best_iter_reg"], m.info["best_iter_clf"]))
            ic = pd.concat(ics)
            res[f"ic{h}"] = ic.mean()
            res[f"t{h}"] = ic.mean() / ic.std() * np.sqrt(len(ic))
            res[f"ic{h}_by_year"] = {int(y): round(float(v), 4) for y, v in ic.groupby(pd.DatetimeIndex(ic.index).year).mean().items()}
            res[f"trees{h}"] = f"reg {np.median([a for a, _ in iters]):.0f} (max {max(a for a, _ in iters)}), clf {np.median([c for _, c in iters]):.0f}"
        res["seconds"] = round(time.time() - t0)
        res["n_features"] = len(feats)
        rows[name] = res
        print(f"{name:18s} IC5 {res['ic5']:+.4f} (t {res['t5']:+.1f})  IC10 {res['ic10']:+.4f} (t {res['t10']:+.1f})  trees5 [{res['trees5']}]  trees10 [{res['trees10']}]  {res['seconds']}s", flush=True)
        print(f"{'':18s} IC5 by year {res['ic5_by_year']}", flush=True)
    out = ROOT / "store/research/ml_diag.json"
    out.write_text(json.dumps(rows, indent=1, default=str))
    print("\n", pd.DataFrame({k: {c: v[c] for c in ("ic5", "t5", "ic10", "t10", "seconds")} for k, v in rows.items()}).T.to_string())


if __name__ == "__main__":
    main(sys.argv[1:])

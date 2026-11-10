"""Out-of-sample ML scores on the all-US universe (5 seeds, every third training date to keep it tractable).

Usage: python research/wide_ml.py <output.parquet>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(FINTAILOR_CONFIG=str(ROOT / "research" / "config_wide.toml"), FINTAILOR_STORE="store_wide")
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from fintailor.strategy import signals  # noqa: E402

signals.ML_SETTINGS.update(n_seeds=5, train_date_stride=3)
meta = json.loads((ROOT / "store_wide/features/panel_meta.json").read_text())
feats = signals.ml_features(meta)
cols = sorted({*feats, "date", "ticker", "is_etf_flag", "fund_known", "sector", "log_dvol", "beta_60", "y_5", "y_10"} & set(pd.read_parquet(ROOT / "store_wide/features/panel.parquet", columns=["date"]).columns) | {*feats, "date", "ticker", "is_etf_flag", "fund_known", "sector", "log_dvol", "beta_60", "y_5", "y_10"})
p = pd.read_parquet(ROOT / "store_wide/features/panel.parquet", columns=cols)
p = p[(~p["is_etf_flag"].astype(bool)) & p["fund_known"] & (p["date"] < pd.Timestamp("2025-11-01"))].reset_index(drop=True)
print(f"{len(p):,} rows, {len(feats)} features", flush=True)
ml = signals.ml_walk_forward(p, feats, "2019-01-01", "2025-10-31", log=lambda s: print(s, flush=True))
out = p.loc[ml.index, ["date", "ticker"]].assign(ml=ml.to_numpy())
out.to_parquet(sys.argv[1], index=False)
print("saved", len(out), flush=True)

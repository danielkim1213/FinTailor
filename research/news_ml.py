"""Out-of-sample ML forecasts with or without news, under two fixed ways of weighting past data.

recent : the project's default. Only the last 5 years, and each year counts half as much as the next one (about 2.7 years of effective data).
long   : everything since 2015, and the weight halves every 5 years (up to about 8.7 years of effective data).

Both windows were fixed before any news result was seen. Nothing from the locked holdout (2025-11-01 on) is loaded, and labels whose
window would reach into it are blanked.

Usage: python research/news_ml.py <plain|news> <recent|long> <output.parquet>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODE, WEIGHTING, OUT = sys.argv[1:4]
if MODE == "news":
    os.environ["FINTAILOR_STORE"] = "store_news"
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fintailor.config import get_config, store_dir  # noqa: E402
from fintailor.strategy import signals  # noqa: E402

HOLDOUT = pd.Timestamp("2025-11-01")
WEIGHTS = {"recent": (1260, 252), "long": (100_000, 1260)}
STRUCTURED = ["ana_net_21", "ana_net_63", "gui_net_63"]

cfg = get_config()["model"]
cfg["train_window_days"], cfg["recency_half_life_days"] = WEIGHTS[WEIGHTING]

d = store_dir("features")
meta = json.loads((d / "panel_meta.json").read_text())
feats = signals.ml_features(meta)
news = [f for f in meta["features"] if f.startswith("news_")] if MODE == "news" else []
cols = sorted({*feats, *news, "date", "ticker", "is_etf_flag", "sector", "log_dvol", "beta_60", "y_5", "y_10"})
p = pd.read_parquet(d / "panel.parquet", columns=cols)
p = p[(~p["is_etf_flag"].astype(bool)) & (p["date"] < HOLDOUT)].reset_index(drop=True)
if MODE == "news":
    extra = pd.read_parquet(d / "news_extra.parquet", columns=["date", "ticker", *STRUCTURED])
    p = p.merge(extra, on=["date", "ticker"], how="left", validate="one_to_one")
    feats = [*feats, *news, *STRUCTURED]
dates = np.sort(p["date"].unique())
for h in (5, 10):
    p.loc[p["date"] > dates[len(dates) - 1 - h], f"y_{h}"] = np.nan
print(f"{MODE}/{WEIGHTING}: {len(p):,} rows, {len(feats)} features, last date {pd.Timestamp(dates[-1]).date()}", flush=True)
ml = signals.ml_walk_forward(p, feats, "2019-01-01", "2025-10-31", log=lambda s: print(s, flush=True))
out = p.loc[ml.index, ["date", "ticker"]].assign(ml=ml.to_numpy())
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(OUT, index=False)
print("saved", len(out), flush=True)

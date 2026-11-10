"""The three forecast sources, each reported as a daily z-score across stocks.

factors  : momentum, profitability, low investment and value. Signs and ingredients come from about 60 years of large-stock
           evidence, not from this data, so nothing is fitted.
sector   : 12-month sector return, skipping the latest month (about 100 years of evidence). Nothing is fitted.
ml       : averaged LightGBM models for 5- and 10-day excess returns. Settings follow research/ml_diag.py (2026-11-10): the
           production defaults stopped after a median of 2 trees, so the learner here uses a slower rate, bigger leaves and more
           patience, and ranks on the regression model alone (the classifier was fitted on the un-neutralised outcome, so
           blending it in undid half of the sector/size/beta neutralisation). Stock-level IC after neutralisation stays below
           0.01 on the development window, so this source has no validated edge; it is kept for the experimental strategy only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import sector_etfs
from ..models.lgbm import fit_horizon, ranking_score, select_features
from ..models.registry import train_window

FACTOR_INPUTS = {"mom": ("mom_12_1", 1), "prof": ("gp_assets_nf", 1), "inv": ("asset_growth_nf", -1), "book": ("book_to_price_nf", 1), "earn": ("earn_yield_nf", 1)}
ML_HORIZONS = (5, 10)
ML_SETTINGS = {"target": "rank", "neutralize": ["sector", "size", "beta"], "drop_prefixes": ["ins_", "si_", "news_", "sector_news", "llm_sector"],
               "rank_by": "reg", "learning_rate": 0.02, "num_leaves": 15, "min_data_in_leaf": 2000, "lambda_l2": 50.0, "feature_fraction": 0.5,
               "early_stopping_rounds": 150, "num_boost_round": 1500, "train_date_stride": 3, "n_seeds": 5, "seed_base": 3017}


def zscore(s: pd.Series, dates) -> pd.Series:
    g = s.groupby(np.asarray(dates))
    return ((s - g.transform("mean")) / (g.transform("std") + 1e-12)).fillna(0.0)


def factor_score(p: pd.DataFrame) -> pd.Series:
    x = pd.DataFrame({k: p[c] * sign for k, (c, sign) in FACTOR_INPUTS.items()})
    x["value"] = x[["book", "earn"]].mean(axis=1)
    x = x[["mom", "prof", "inv", "value"]]
    return zscore(x.mean(axis=1).where(x.notna().sum(axis=1) >= 2), p["date"])


def sector_score(p: pd.DataFrame, con) -> pd.Series:
    etfs = sector_etfs()
    names = "','".join(etfs)
    px = con.execute(f"select ticker, date, adj_close from prices_daily where ticker in ('{names}')").df()
    px["date"] = pd.to_datetime(px["date"])
    C = px.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").sort_index()
    mom = C.shift(21) / C.shift(252) - 1
    z = mom.sub(mom.mean(axis=1), axis=0).div(mom.std(axis=1) + 1e-12, axis=0).rename(columns=etfs)
    long = z.stack().rename("sec").rename_axis(["date", "sector"]).reset_index()
    merged = p[["date", "sector"]].assign(sector=p["sector"].astype(str)).merge(long, on=["date", "sector"], how="left")
    return merged["sec"].set_axis(p.index).fillna(0.0)


def ml_features(meta: dict) -> list[str]:
    return select_features(meta["features"], ML_SETTINGS)


def ml_fit_predict(panel: pd.DataFrame, feats: list[str], asof: pd.Timestamp, test: pd.DataFrame) -> pd.Series:
    """Models trained only on labels that had finished before `asof`, scoring the rows in `test`."""
    parts = []
    for h in ML_HORIZONS:
        model = fit_horizon(train_window(panel, asof, h), feats, h, ML_SETTINGS)
        s = ranking_score(model.predict(test), ML_SETTINGS.get("rank_by"))
        parts.append(zscore(pd.Series(s.to_numpy(), index=test.index), test["date"]))
    return pd.concat(parts, axis=1).mean(axis=1)


def ml_walk_forward(panel: pd.DataFrame, feats: list[str], start: str, end: str | None = None, every: int = 63, log=None) -> pd.Series:
    """Out-of-sample ML score for every row from `start`, retraining every `every` trading days."""
    dates = pd.DatetimeIndex(np.sort(panel["date"].unique()))
    first = dates.searchsorted(pd.Timestamp(start))
    last = len(dates) if end is None else dates.searchsorted(pd.Timestamp(end), side="right")
    out = []
    for i, b in enumerate(range(first, last, every)):
        lo, hi = dates[b], dates[min(b + every, last) - 1]
        test = panel[(panel["date"] >= lo) & (panel["date"] <= hi)]
        out.append(ml_fit_predict(panel, feats, lo, test))
        if log:
            log(f"ml walk-forward: block {i + 1}, through {hi.date()}")
    return pd.concat(out).sort_index()

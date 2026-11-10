"""Research bench: fast walk-forward for trying model ideas against one scoreboard.

Usage:  python research/lab.py ridge lgbm_small blend:ridge+lgbm_small

Every run is appended to store/research/trials.jsonl, so the number of ideas tried is on record when judging the best
one. The locked holdout (2025-11-01 on) is removed at load, including labels whose window would reach into it.
Ideas are ranked on the development window; the validation window is only for confirming finalists.
"""
from __future__ import annotations

import json
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fintailor.eval.portfolio import ensemble_scores, simulate  # noqa: E402
from fintailor.models.lgbm import _neutralize, _rank_gauss  # noqa: E402

HOLDOUT = pd.Timestamp("2025-11-01")
DEV_END = pd.Timestamp("2023-12-31")
OUT = ROOT / "store" / "research"
HS = (5, 10, 30)
EMBARGO = 5
WINDOW = 1260
EVERY = 63
FIRST_TEST = "2019-01-01"
PER = 252 / 5
NEUTRAL = ["sector", "size", "beta"]
UNUSED = ("news_", "ins_", "si_", "rel_", "is_etf", "sector_news", "llm_sector")


@dataclass
class Bench:
    p: pd.DataFrame
    pos: np.ndarray
    dates: pd.DatetimeIndex
    stock: list[str]
    market: list[str]
    rets: pd.DataFrame
    targets: dict[int, np.ndarray]


def load(extra: Path | None = OUT / "extra_features.parquet", hs=HS) -> Bench:
    meta = json.loads((ROOT / "store/features/panel_meta.json").read_text())
    p = pd.read_parquet(ROOT / "store/features/panel.parquet")
    p = p[(p["date"] < HOLDOUT) & (~p["is_etf_flag"].astype(bool))].reset_index(drop=True)
    p["sector"] = p["sector"].astype(str)
    stock = [f for f in meta["ranked"] + meta["raw"] if not f.startswith(UNUSED)]
    if extra is not None and extra.exists():
        x = pd.read_parquet(extra)
        new = [c for c in x.columns if c not in ("date", "ticker")]
        p = p.merge(x, on=["date", "ticker"], how="left")
        stock += new
    dates = pd.DatetimeIndex(np.sort(p["date"].unique()))
    pos = dates.searchsorted(p["date"].values)
    for h in hs:  # outcome windows that would end inside the holdout are unusable
        late = pos + h >= len(dates)
        for c in (f"y_{h}", f"g_{h}"):
            p.loc[late, c] = np.nan
    # Every stock feature becomes a daily percentile in -0.5..0.5. A gap takes a fixed pseudo-random percentile for that
    # ticker and feature: a constant would mark the stocks whose data is thin, which are the ones that later left the index.
    codes, uniq = pd.factorize(p["ticker"])
    for f in stock:
        fixed = np.array([zlib.crc32(f"{t}|{f}".encode()) / 2**32 - 0.5 for t in uniq], dtype=np.float32)[codes]
        rank = (p.groupby("date")[f].rank(pct=True) - 0.5).to_numpy(np.float32)
        p[f] = np.where(np.isnan(rank), fixed, rank)
    targets = {}
    for h in hs:
        y = pd.Series(_neutralize(p, f"y_{h}", NEUTRAL), index=p.index)
        targets[h] = _rank_gauss(y, p["date"])
    rets = p.rename(columns={"g_5": "r", "atr_pct_raw": "vol"})[["date", "ticker", "r", "vol", "sector"]]
    return Bench(p, pos, dates, stock, meta["market"], rets, targets)


# ------------------------------------------------------------------------------ models
Fit = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def ridge(lam: float = 1.0, half_life: float | None = None) -> Fit:
    """Closed-form ridge on daily-percentile features. No intercept: targets are centred within each day."""
    def fit(Xa, ya, pa, Xb):
        ok = np.isfinite(ya)
        X, y = Xa[ok].astype(np.float64), ya[ok]
        if half_life:
            w = 0.5 ** ((pa[ok].max() - pa[ok]) / half_life)
            sw = np.sqrt(w)[:, None]
            X, y = X * sw, y * sw[:, 0]
        A = X.T @ X + lam * len(X) * np.eye(X.shape[1]) * 0.01
        return Xb @ np.linalg.solve(A, X.T @ y)
    return fit


def gbm(rounds: int = 200, lr: float = 0.02, leaves: int = 15, min_leaf: int = 2000, stride: int = 3, ff: float = 0.5,
        bag: float = 0.7, seeds: int = 1) -> Fit:
    """Fixed-size LightGBM (no early stopping): low learning rate, large leaves, thinned dates."""
    def fit(Xa, ya, pa, Xb):
        ok = np.isfinite(ya) & (pa % stride == 0)
        out = np.zeros(len(Xb))
        for s in range(seeds):
            ds = lgb.Dataset(Xa[ok], ya[ok], free_raw_data=True)
            params = dict(objective="regression", learning_rate=lr, num_leaves=leaves, min_data_in_leaf=min_leaf,
                          feature_fraction=ff, bagging_fraction=bag, bagging_freq=1, lambda_l2=10.0, verbosity=-1,
                          seed=17 + s, num_threads=8, force_col_wise=True)
            out += lgb.train(params, ds, num_boost_round=rounds).predict(Xb, num_threads=8)
        return out / seeds
    return fit


# ------------------------------------------------------------------------------ walk-forward + scoreboard
def walk(b: Bench, feats: list[str], model: Fit, hs=HS, name: str = "?") -> pd.DataFrame:
    X = b.p[feats].to_numpy(np.float32)
    first = max(b.dates.searchsorted(pd.Timestamp(FIRST_TEST)), 504 + max(hs) + 10)
    out, t0 = [], time.time()
    for s in range(first, len(b.dates), EVERY):
        te = np.flatnonzero((b.pos >= s) & (b.pos < s + EVERY))
        for h in hs:
            hi = s - h - EMBARGO - 1
            tr = np.flatnonzero((b.pos <= hi) & (b.pos > hi - WINDOW))
            out.append(pd.DataFrame({"date": b.p["date"].values[te], "ticker": b.p["ticker"].values[te], "horizon": h,
                                     "score": model(X[tr], b.targets[h][tr], b.pos[tr], X[te]),
                                     "g": b.p[f"g_{h}"].values[te]}))
    print(f"[{name}] fitted in {time.time() - t0:.0f}s", flush=True)
    return pd.concat(out, ignore_index=True)


def walk_pipe(b: Bench, feats: list[str], overrides: dict, hs=(5, 10), name: str = "?") -> pd.DataFrame:
    """Walk-forward using the production learner (early-stopped LightGBM pair, blended ranking)."""
    from fintailor.models.lgbm import fit_horizon, ranking_score
    ov = {"target": "rank", "neutralize": NEUTRAL, **overrides}
    first = max(b.dates.searchsorted(pd.Timestamp(FIRST_TEST)), 504 + max(hs) + 10)
    out, t0 = [], time.time()
    for s in range(first, len(b.dates), EVERY):
        test = b.p[(b.pos >= s) & (b.pos < s + EVERY)]
        for h in hs:
            hi = s - h - EMBARGO - 1
            m = fit_horizon(b.p[(b.pos <= hi) & (b.pos > hi - WINDOW)], feats, h, ov)
            out.append(pd.DataFrame({"date": test["date"].values, "ticker": test["ticker"].values, "horizon": h,
                                     "score": ranking_score(m.predict(test), ov.get("rank_by")).values, "g": test[f"g_{h}"].values}))
    print(f"[{name}] fitted in {time.time() - t0:.0f}s", flush=True)
    return pd.concat(out, ignore_index=True)


def _ic(o: pd.DataFrame) -> float:
    from fintailor.eval.metrics import rank_ic
    return float(rank_ic(o, "score", "g").mean())


def scoreboard(b: Bench, oos: pd.DataFrame, name: str, note: dict | None = None, hs=HS) -> dict:
    sc = ensemble_scores(oos, list(hs))
    runs = [simulate(sc, b.rets, k=50, buffer=3.0, offset=i, weighting="equal", sector_cap=None) for i in range(5)]
    dev_lim, res = pd.Timestamp(DEV_END), {"name": name}
    for label, keep in (("dev", lambda d: d <= dev_lim), ("val", lambda d: d > dev_lim)):
        nets = [r[keep(pd.to_datetime(r["date"]))]["ret"].mean() * PER * 100 for r in runs]
        res[f"net_{label}"], res[f"min_{label}"], res[f"max_{label}"] = np.mean(nets), min(nets), max(nets)
    for h in hs:
        for label, keep in (("dev", oos["date"] <= dev_lim), ("val", oos["date"] > dev_lim)):
            res[f"ic{h}_{label}"] = _ic(oos[(oos["horizon"] == h) & keep])
    allr = pd.concat(runs)
    res["turnover"] = float(allr["turnover"].mean() * PER)
    res["gross_all"] = float(allr["gross"].mean() * PER * 100)
    res["net_all_offsets"] = [round(float(r["ret"].mean() * PER * 100), 2) for r in runs]
    res.update(note or {})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "trials.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": pd.Timestamp.now().isoformat(timespec="seconds"), **res}, default=float) + "\n")
    ics = " ".join(f"IC{h} {res[f'ic{h}_dev']:+.4f}" for h in hs)
    print(f"{name:26s} DEV net {res['net_dev']:+6.2f} [{res['min_dev']:+6.2f},{res['max_dev']:+6.2f}] {ics} | turn {res['turnover']:4.1f}", flush=True)
    return res


def blend(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Average of per-date, per-horizon z-scores."""
    keys = ["date", "ticker", "horizon"]
    parts = []
    for f in frames:
        g = f.groupby(["date", "horizon"])["score"]
        parts.append(f.set_index(keys)["score"].sub(g.transform("mean").values).div(g.transform("std").values + 1e-12))
    z = pd.concat(parts, axis=1).mean(axis=1).rename("score").reset_index()
    return z.merge(frames[0][keys + ["g"]], on=keys)


# ------------------------------------------------------------------------------ experiments
def experiments(b: Bench) -> dict:
    orig = [f for f in b.stock if not f.startswith("x_")]
    market = b.market
    return {
        "ridge": (orig, ridge(1.0)),
        "ridge_all": (b.stock, ridge(1.0)),
        "ridge_recent": (orig, ridge(1.0, half_life=252)),
        "lgbm_stock": (orig, gbm()),
        "lgbm_market": (orig + market, gbm()),
        "lgbm_all": (b.stock + market, gbm()),
        "lgbm_all_seeds": (b.stock + market, gbm(seeds=3)),
        **{f"p_{k}": (orig + v + market, {}) for k, v in GROUPS.items()},
        "p_all": (orig + [f for v in GROUPS.values() for f in v] + market, {}),
        "p_seeds5": (orig + market, {"n_seeds": 5}),
        "p_seeds5_b": (orig + market, {"n_seeds": 5, "seed_base": 1017}),
        "p_seeds5_c": (orig + market, {"n_seeds": 5, "seed_base": 2017}),
        "p_seeds15": (orig + market, {"n_seeds": 15, "seed_base": 3017}),
        "p_all_s15": (orig + [f for v in GROUPS.values() for f in v] + market, {"n_seeds": 15, "seed_base": 3017}),
        "p_all_s5": (orig + [f for v in GROUPS.values() for f in v] + market, {"n_seeds": 5}),
    }


GROUPS = {
    "base": [],
    "mom": ["x_mom_6_1", "x_mom_3_1", "x_mom_36_12", "x_mom_consistency"],
    "resid": ["x_resid_mom_12_1", "x_resid_mom_6_1", "x_resid_rev_5", "x_resid_rev_20"],
    "risk": ["x_vol_252", "x_skew_60", "x_resid_skew_60", "x_max5_20", "x_down_beta"],
    "liq": ["x_amihud_20", "x_vol_trend", "x_dvol_chg_60"],
    "anchor": ["x_dist_52w_low", "x_dist_3y_high"],
    "season": ["x_season_month"],
    "earn": ["x_sue", "x_beat_streak", "x_earn_react"],
}


def main(names: list[str]) -> None:
    (OUT / "oos").mkdir(parents=True, exist_ok=True)
    b = load()
    print(f"panel {len(b.p):,} rows, {len(b.stock)} stock + {len(b.market)} market features", flush=True)
    ex = experiments(b)
    for n in names:
        if n.startswith("blend:"):
            parts = n[6:].split("+")
            o = blend([pd.read_parquet(OUT / "oos" / f"{x}.parquet") for x in parts])
            scoreboard(b, o, n, {"parts": parts})
            continue
        feats, model = ex[n]
        pipe = isinstance(model, dict)
        o = walk_pipe(b, feats, model, name=n) if pipe else walk(b, feats, model, name=n)
        o.to_parquet(OUT / "oos" / f"{n}.parquet", index=False)
        scoreboard(b, o, n, {"n_features": len(feats)}, hs=(5, 10) if pipe else HS)


if __name__ == "__main__":
    main(sys.argv[1:])

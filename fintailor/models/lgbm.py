"""LightGBM per-horizon models: regressor (net excess return) + classifier (beat) + isotonic
calibration. SHAP-style drivers come from LightGBM's built-in pred_contrib (fast, exact TreeSHAP)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from ..config import get_config


def model_params(overrides: dict | None = None) -> dict:
    cfg = get_config()["model"]
    p = dict(cfg["lgbm"])
    for k, v in (overrides or {}).items():
        if k in p or k in ("min_sum_hessian_in_leaf", "max_depth", "extra_trees", "lambda_l1"):
            p[k] = v
    return p


def select_features(all_feats: list[str], overrides: dict | None = None) -> list[str]:
    o = overrides or {}
    drop = set(o.get("drop_features", []))
    include = set(o.get("include_prefixes", []))  # opt in to a globally excluded group
    excluded = [x for x in get_config()["model"].get("exclude_prefixes", []) if x not in include]
    pref = tuple(o.get("drop_prefixes", [])) + tuple(excluded)
    return [f for f in all_feats if f not in drop and not (pref and f.startswith(pref))]


def _lgb_params(p: dict, objective: str, seed: int = 17) -> dict:
    cfg = get_config()["model"]
    out = {k: v for k, v in p.items() if k not in ("num_boost_round", "early_stopping_rounds")}
    out.update(objective=objective, verbosity=-1, num_threads=cfg.get("num_threads", 0), seed=seed,
               deterministic=False, force_col_wise=True)
    return out


def recency_weights(dates: pd.Series, half_life_days: float) -> np.ndarray:
    d = pd.to_datetime(dates)
    u = np.sort(d.unique())
    pos = np.searchsorted(u, d.values)
    age = (len(u) - 1) - pos
    return np.power(0.5, age / max(half_life_days, 1)).astype(np.float32)


class AvgBooster:
    """Average of several LightGBM boosters (same data, different seeds); quacks like a Booster for predict."""

    def __init__(self, boosters: list[lgb.Booster]):
        self.boosters = boosters
        self.best_iteration = boosters[0].best_iteration

    def predict(self, X, **kw):
        return np.mean([b.predict(X, **kw) for b in self.boosters], axis=0)


def _save_booster(b, d: Path, stem: str) -> None:
    for i, x in enumerate(b.boosters if isinstance(b, AvgBooster) else [b]):
        x.save_model(str(d / (f"{stem}.txt" if i == 0 else f"{stem}_s{i}.txt")))


def _load_booster(d: Path, stem: str, n: int):
    bs = [lgb.Booster(model_file=str(d / (f"{stem}.txt" if i == 0 else f"{stem}_s{i}.txt"))) for i in range(n)]
    return bs[0] if n == 1 else AvgBooster(bs)


@dataclass
class HorizonModel:
    horizon: int
    features: list[str]
    reg: lgb.Booster
    clf: lgb.Booster
    iso_x: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    iso_y: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    exp_x: np.ndarray = field(default_factory=lambda: np.array([]))
    exp_y: np.ndarray = field(default_factory=lambda: np.array([]))
    info: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ predict
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = _matrix(df, self.features)
        reg = self.reg.predict(X, num_threads=0)
        raw = self.clf.predict(X, num_threads=0)
        if self.info.get("calib_mode") == "rank":  # calibrate on the day's cross-sectional rank
            cr, rr = _xs_rank(raw, df), _xs_rank(reg, df)
        else:
            cr, rr = raw, reg
        prob = np.interp(cr, self.iso_x, self.iso_y)
        exp = np.interp(rr, self.exp_x, self.exp_y) if len(self.exp_x) > 1 else reg
        return pd.DataFrame({"pred_excess": exp, "pred_raw": reg, "prob_raw": raw, "prob": prob}, index=df.index)

    def drivers(self, df: pd.DataFrame, top_n: int = 5) -> list[list[tuple[str, float]]]:
        X = _matrix(df, self.features)
        contrib = self.clf.predict(X, pred_contrib=True)[:, :-1]
        out = []
        for row in contrib:
            idx = np.argsort(-np.abs(row))[:top_n]
            out.append([(self.features[i], float(row[i])) for i in idx])
        return out

    # ------------------------------------------------------------------ io
    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        _save_booster(self.reg, d, f"reg_{self.horizon}")
        _save_booster(self.clf, d, f"clf_{self.horizon}")
        (d / f"cal_{self.horizon}.json").write_text(json.dumps({
            "features": self.features, "iso_x": self.iso_x.tolist(), "iso_y": self.iso_y.tolist(),
            "exp_x": self.exp_x.tolist(), "exp_y": self.exp_y.tolist(), "info": self.info}))

    @classmethod
    def load(cls, d: Path, h: int) -> "HorizonModel":
        c = json.loads((d / f"cal_{h}.json").read_text())
        n = int(c.get("info", {}).get("n_seeds", 1))
        return cls(h, c["features"], _load_booster(d, f"reg_{h}", n),
                   _load_booster(d, f"clf_{h}", n), np.array(c["iso_x"]), np.array(c["iso_y"]),
                   np.array(c["exp_x"]), np.array(c["exp_y"]), c.get("info", {}))


def _xs_rank(v: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Percentile rank within each date (0..1)."""
    return pd.Series(v, index=df.index).groupby(df["date"].values).rank(pct=True).to_numpy()


def _matrix(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    cols = [df[f].to_numpy(np.float32, na_value=np.nan) if f in df else np.full(len(df), np.nan, np.float32) for f in feats]
    return np.column_stack(cols) if cols else np.empty((len(df), 0), np.float32)


def fit_horizon(train: pd.DataFrame, features: list[str], h: int, overrides: dict | None = None,
                embargo: int | None = None) -> HorizonModel:
    """train: rows with label y_h. The last `calib_days` dates (after a purge gap) are held out for
    early stopping + isotonic calibration of probability and expected return."""
    mcfg = get_config()["model"]
    lcfg = get_config()["labels"]
    p = model_params(overrides)
    y_col = f"y_{h}"
    d = train[train[y_col].notna()]
    stride = int((overrides or {}).get("train_date_stride", mcfg.get("train_date_stride", 1)))
    udates = np.sort(d["date"].unique())
    if stride > 1:
        keep = set(udates[::stride])
        d = d[d["date"].isin(keep)]
        udates = np.sort(d["date"].unique())
    gap = embargo if embargo is not None else h + lcfg["embargo_extra_days"]
    n_cal = min(mcfg["calib_days"], max(len(udates) // 5, 1))
    cal_start = udates[-n_cal]
    # strided dates: round the purge UP so the gap never shrinks below `gap` trading days (floor let 10-day labels of the last
    # fit dates overlap the first calibration dates when stride=3)
    fit_end = udates[max(0, len(udates) - n_cal - max(-(-gap // stride), 1) - 1)]
    fit = d[d["date"] <= fit_end]
    cal = d[d["date"] >= cal_start]

    ov = overrides or {}
    target_mode = ov.get("target", mcfg.get("target", "raw"))
    refit = bool(ov.get("refit_full", mcfg.get("refit_full", False)))
    n_seeds = int(ov.get("n_seeds", mcfg.get("n_seeds", 1)))
    q = mcfg.get("target_clip_quantile", 0.01)
    y = fit[y_col].to_numpy(np.float64)
    lo, hi = np.nanquantile(y, [q, 1 - q])

    neutral = list(ov.get("neutralize", mcfg.get("neutralize", [])))

    def reg_target(frame: pd.DataFrame) -> np.ndarray:
        yv = frame[y_col]
        if neutral:  # remove what sector / size / beta explain that day (don't learn "megacaps went up")
            yv = pd.Series(_neutralize(frame, y_col, neutral), index=frame.index)
        if target_mode == "rank":  # per-date normal scores: pure cross-sectional ranking, no fat tails
            return _rank_gauss(yv, frame["date"])
        return np.clip(yv.to_numpy(np.float64), lo, hi)

    half = mcfg["recency_half_life_days"] / stride
    w = recency_weights(fit["date"], half)
    Xf, Xc = _matrix(fit, features), _matrix(cal, features)
    yc = cal[y_col].to_numpy(np.float64)
    nbr, esr = p.get("num_boost_round", 400), p.get("early_stopping_rounds", 50)

    early_stop = bool(ov.get("early_stop", mcfg.get("early_stop", True)))
    clf_neutral = bool(ov.get("clf_neutral", False)) and bool(neutral)
    seed_base = int(ov.get("seed_base", 17))

    def clf_target(frame: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
        return ((_neutralize(frame, y_col, neutral) if clf_neutral else raw) > 0).astype(float)

    def train_one(obj, target, target_cal):
        ds = lgb.Dataset(Xf, target, weight=w, feature_name=features, free_raw_data=True)
        if not early_stop:
            return lgb.train(_lgb_params(p, obj, seed_base), ds, num_boost_round=nbr)
        dv = lgb.Dataset(Xc, target_cal, reference=ds)
        return lgb.train(_lgb_params(p, obj, seed_base), ds, num_boost_round=nbr, valid_sets=[dv],
                         callbacks=[lgb.early_stopping(esr, verbose=False)])

    reg = train_one("regression", reg_target(fit), reg_target(cal))
    clf = train_one("binary", clf_target(fit, y), clf_target(cal, yc))

    mode = mcfg.get("calib_mode", "rank")
    raw_c, reg_c = clf.predict(Xc), reg.predict(Xc)
    if mode == "rank":
        raw_c, reg_c = _xs_rank(raw_c, cal), _xs_rank(reg_c, cal)
    px, py = binned_isotonic(raw_c, (yc > 0).astype(float), mcfg.get("calib_bins", 20))
    ex, ey = binned_isotonic(reg_c, yc, mcfg.get("calib_bins", 20))
    info = {"n_fit": len(fit), "n_cal": len(cal), "fit_end": str(pd.Timestamp(fit_end).date()),
            "train_end": str(pd.Timestamp(udates[-1]).date()), "train_start": str(pd.Timestamp(udates[0]).date()),
            "best_iter_reg": reg.best_iteration, "best_iter_clf": clf.best_iteration, "calib_mode": mode,
            "target": target_mode, "refit_full": False, "n_seeds": n_seeds}
    if refit and mode == "rank":
        # The holdout above only picks the number of rounds and the calibration map. Without a refit the model
        # would never see its most recent ~calib_days of data (a year-old model; also hides any recent feature
        # such as news). Rank calibration maps the day's cross-sectional rank -> outcome, so it transfers to the
        # refit booster. Raw-score calibration would not, hence rank mode only.
        Xa = _matrix(d, features)
        wa = recency_weights(d["date"], half)
        ya = d[y_col].to_numpy(np.float64)

        def refit_one(obj, target, iters):
            bs = [lgb.train(_lgb_params(p, obj, seed_base + i), lgb.Dataset(Xa, target, weight=wa, feature_name=features),
                            num_boost_round=max(int(round(iters * 1.1)), 10)) for i in range(n_seeds)]
            return bs[0] if n_seeds == 1 else AvgBooster(bs)

        reg = refit_one("regression", reg_target(d), reg.best_iteration or nbr)
        clf = refit_one("binary", clf_target(d, ya), clf.best_iteration or nbr)
        info.update(refit_full=True, refit_end=str(pd.Timestamp(udates[-1]).date()))
    elif n_seeds > 1:  # same rounds and data as the early-stopped model, other seeds
        def more(obj, target, first):
            extra = [lgb.train(_lgb_params(p, obj, seed_base + i), lgb.Dataset(Xf, target, weight=w, feature_name=features),
                               num_boost_round=first.best_iteration or nbr) for i in range(1, n_seeds)]
            return AvgBooster([first] + extra)

        reg = more("regression", reg_target(fit), reg)
        clf = more("binary", clf_target(fit, y), clf)
    return HorizonModel(h, features, reg, clf, px, py, ex, ey, info)


def _neutralize(frame: pd.DataFrame, y_col: str, what: list[str]) -> np.ndarray:
    """Per-date OLS residual of the target on sector dummies and/or size (log_dvol) and beta (beta_60)."""
    y = frame[y_col].to_numpy(np.float64)
    out = np.full(len(frame), np.nan)
    cols = [c for c, key in (("log_dvol", "size"), ("beta_60", "beta")) if key in what and c in frame]
    sec = frame["sector"].astype(str).to_numpy() if "sector" in what and "sector" in frame else None
    dates = frame["date"].to_numpy()
    order = np.argsort(dates, kind="stable")
    sd = dates[order]
    bounds = np.flatnonzero(np.r_[True, sd[1:] != sd[:-1], True])
    num = frame[cols].to_numpy(np.float64) if cols else np.zeros((len(frame), 0))
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        yi = y[idx]
        ok = np.isfinite(yi)
        if ok.sum() < 20:
            continue
        parts = [np.ones((len(idx), 1))]
        if cols:
            xi = num[idx]
            fill = np.nanmean(np.where(np.isfinite(xi), xi, np.nan), axis=0)
            parts.append(np.where(np.isfinite(xi), xi, np.nan_to_num(fill)))
        if sec is not None:
            levels, inv = np.unique(sec[idx], return_inverse=True)
            if len(levels) > 1:
                parts.append(np.eye(len(levels))[inv][:, 1:])
        X = np.nan_to_num(np.hstack(parts))
        coef, *_ = np.linalg.lstsq(X[ok], yi[ok], rcond=None)
        out[idx[ok]] = yi[ok] - X[ok] @ coef
    return out


def _rank_gauss(y: pd.Series, dates: pd.Series) -> np.ndarray:
    """Per-date rank -> standard normal score (NaN stays NaN)."""
    from scipy.special import ndtri
    g = pd.Series(y.to_numpy(np.float64), index=y.index).groupby(dates.values)
    r, n = g.rank(method="average"), g.transform("count")
    return ndtri(((r - 0.5) / n).clip(1e-4, 1 - 1e-4)).to_numpy(np.float64)


def binned_isotonic(score: np.ndarray, y: np.ndarray, bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Calibration map score -> E[y]: quantile bins (each ~1/bins of the calibration rows), monotone
    via weighted isotonic over bin means. Avoids raw isotonic's overfit extremes (thin tails ->
    0.99 probabilities). Outside the range np.interp clamps to the end bins."""
    ok = np.isfinite(score) & np.isfinite(y)
    s, t = score[ok], y[ok]
    if len(s) < bins * 20:
        bins = max(2, len(s) // 20)
    edges = np.unique(np.quantile(s, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)
    cnt = np.bincount(idx, minlength=len(edges) - 1)
    keep = cnt > 0
    xm = (np.bincount(idx, weights=s, minlength=len(cnt))[keep] / cnt[keep])
    ym = (np.bincount(idx, weights=t, minlength=len(cnt))[keep] / cnt[keep])
    fit = IsotonicRegression().fit(xm, ym, sample_weight=cnt[keep]).predict(xm)
    return xm, fit


def ranking_score(pred: pd.DataFrame, mode: str | None = None) -> pd.Series:
    mode = mode or get_config()["model"].get("rank_by", "blend")
    if mode == "reg":
        return pred["pred_raw"]
    if mode == "clf":
        return pred["prob_raw"]
    # blend: average of z-scores (per call; caller passes one cross-section or groups)
    z = lambda s: (s - s.mean()) / (s.std() + 1e-12)  # noqa: E731
    return 0.5 * z(pred["pred_raw"]) + 0.5 * z(pred["prob_raw"])

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
    pref = tuple(o.get("drop_prefixes", []))
    return [f for f in all_feats if f not in drop and not (pref and f.startswith(pref))]


def _lgb_params(p: dict, objective: str) -> dict:
    cfg = get_config()["model"]
    out = {k: v for k, v in p.items() if k not in ("num_boost_round", "early_stopping_rounds")}
    out.update(objective=objective, verbosity=-1, num_threads=cfg.get("num_threads", 0), seed=17,
               deterministic=False, force_col_wise=True)
    return out


def recency_weights(dates: pd.Series, half_life_days: float) -> np.ndarray:
    d = pd.to_datetime(dates)
    u = np.sort(d.unique())
    pos = np.searchsorted(u, d.values)
    age = (len(u) - 1) - pos
    return np.power(0.5, age / max(half_life_days, 1)).astype(np.float32)


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
        self.reg.save_model(str(d / f"reg_{self.horizon}.txt"))
        self.clf.save_model(str(d / f"clf_{self.horizon}.txt"))
        (d / f"cal_{self.horizon}.json").write_text(json.dumps({
            "features": self.features, "iso_x": self.iso_x.tolist(), "iso_y": self.iso_y.tolist(),
            "exp_x": self.exp_x.tolist(), "exp_y": self.exp_y.tolist(), "info": self.info}))

    @classmethod
    def load(cls, d: Path, h: int) -> "HorizonModel":
        c = json.loads((d / f"cal_{h}.json").read_text())
        return cls(h, c["features"], lgb.Booster(model_file=str(d / f"reg_{h}.txt")),
                   lgb.Booster(model_file=str(d / f"clf_{h}.txt")), np.array(c["iso_x"]), np.array(c["iso_y"]),
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
    fit_end = udates[max(0, len(udates) - n_cal - max(gap // stride, 1) - 1)]
    fit = d[d["date"] <= fit_end]
    cal = d[d["date"] >= cal_start]

    y = fit[y_col].to_numpy(np.float64)
    q = mcfg.get("target_clip_quantile", 0.01)
    lo, hi = np.nanquantile(y, [q, 1 - q])
    w = recency_weights(fit["date"], mcfg["recency_half_life_days"] / stride)
    Xf, Xc = _matrix(fit, features), _matrix(cal, features)
    yc = cal[y_col].to_numpy(np.float64)
    nbr, esr = p.get("num_boost_round", 400), p.get("early_stopping_rounds", 50)

    def train_one(obj, target, target_cal):
        ds = lgb.Dataset(Xf, target, weight=w, feature_name=features, free_raw_data=True)
        dv = lgb.Dataset(Xc, target_cal, reference=ds)
        return lgb.train(_lgb_params(p, obj), ds, num_boost_round=nbr, valid_sets=[dv],
                         callbacks=[lgb.early_stopping(esr, verbose=False)])

    reg = train_one("regression", np.clip(y, lo, hi), np.clip(yc, lo, hi))
    clf = train_one("binary", (y > 0).astype(float), (yc > 0).astype(float))

    mode = mcfg.get("calib_mode", "rank")
    raw_c, reg_c = clf.predict(Xc), reg.predict(Xc)
    if mode == "rank":
        raw_c, reg_c = _xs_rank(raw_c, cal), _xs_rank(reg_c, cal)
    px, py = binned_isotonic(raw_c, (yc > 0).astype(float), mcfg.get("calib_bins", 20))
    ex, ey = binned_isotonic(reg_c, yc, mcfg.get("calib_bins", 20))
    info = {"n_fit": len(fit), "n_cal": len(cal), "fit_end": str(pd.Timestamp(fit_end).date()),
            "train_end": str(pd.Timestamp(udates[-1]).date()), "train_start": str(pd.Timestamp(udates[0]).date()),
            "best_iter_reg": reg.best_iteration, "best_iter_clf": clf.best_iteration, "calib_mode": mode}
    return HorizonModel(h, features, reg, clf, px, py, ex, ey, info)


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


def ranking_score(pred: pd.DataFrame) -> pd.Series:
    mode = get_config()["model"].get("rank_by", "blend")
    if mode == "reg":
        return pred["pred_raw"]
    if mode == "clf":
        return pred["prob_raw"]
    # blend: average of z-scores (per call; caller passes one cross-section or groups)
    z = lambda s: (s - s.mean()) / (s.std() + 1e-12)  # noqa: E731
    return 0.5 * z(pred["pred_raw"]) + 0.5 * z(pred["prob_raw"])

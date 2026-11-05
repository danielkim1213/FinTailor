"""Model bundles (all horizons) + versioned registry with champion pointer and rollback."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import get_config, horizons, store_dir
from ..utils import log, stable_id, utcnow
from .lgbm import HorizonModel, fit_horizon, select_features


@dataclass
class ModelBundle:
    version: str
    config_name: str
    models: dict[int, HorizonModel]
    meta: dict = field(default_factory=dict)

    def predict(self, df: pd.DataFrame, with_drivers: bool = False, top_n: int = 5) -> pd.DataFrame:
        from .lgbm import ranking_score
        out = []
        for h, m in self.models.items():
            p = m.predict(df)
            p["score"] = ranking_score(p)
            p["horizon"] = h
            p["date"] = df["date"].values
            p["ticker"] = df["ticker"].values
            if with_drivers:
                p["drivers"] = [json.dumps(x) for x in m.drivers(df, top_n)]
            out.append(p)
        return pd.concat(out, ignore_index=True)

    def save(self) -> Path:
        d = store_dir("models", self.version)
        for m in self.models.values():
            m.save(d)
        (d / "meta.json").write_text(json.dumps({**self.meta, "version": self.version, "config_name": self.config_name,
                                                 "horizons": list(self.models)}, indent=1, default=str))
        return d

    @classmethod
    def load(cls, version: str) -> "ModelBundle":
        d = store_dir("models", version)
        meta = json.loads((d / "meta.json").read_text())
        return cls(version, meta.get("config_name", "?"), {h: HorizonModel.load(d, h) for h in meta["horizons"]}, meta)


def config_overrides(name: str) -> dict:
    if name in ("champion", "base"):
        return {}
    for c in get_config().get("challengers", []):
        if c["name"] == name:
            return {k: v for k, v in c.items() if k != "name"}
    raise KeyError(f"unknown model config {name}")


def train_window(panel: pd.DataFrame, end_date: pd.Timestamp, h: int, window_days: int | None = None) -> pd.DataFrame:
    """Rows usable for training a model that will be used from `end_date` on (purged by h + embargo)."""
    cfg = get_config()
    window_days = window_days or cfg["model"]["train_window_days"]
    gap = h + cfg["labels"]["embargo_extra_days"]
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    pos = dates.searchsorted(end_date, side="left") - gap - 1
    if pos < 0:
        return panel.iloc[0:0]
    start = dates[max(0, pos - window_days + 1)]
    return panel[(panel["date"] >= start) & (panel["date"] <= dates[pos])]


def train_bundle(panel: pd.DataFrame, end_date: pd.Timestamp, config_name: str = "base",
                 hs: list[int] | None = None, features: list[str] | None = None) -> ModelBundle:
    hs = hs or horizons()
    ov = config_overrides(config_name)
    feats = select_features(features or panel.attrs["meta"]["features"], ov)
    models = {}
    for h in hs:
        tr = train_window(panel, end_date, h)
        if tr[f"y_{h}"].notna().sum() < 1000:
            raise ValueError(f"not enough training rows for h={h}")
        models[h] = fit_horizon(tr, feats, h, ov)
    ver = f"v{pd.Timestamp(end_date):%Y%m%d}-{config_name}-{stable_id(utcnow(), config_name)[:6]}"
    return ModelBundle(ver, config_name, models, {"trained_for": str(pd.Timestamp(end_date).date()),
                                                  "created_at": str(utcnow()), "features": feats,
                                                  "train": {h: m.info for h, m in models.items()}})


# ----------------------------------------------------------------------------- registry
def _reg_file() -> Path:
    return store_dir("models") / "registry.json"


def registry() -> dict:
    f = _reg_file()
    return json.loads(f.read_text()) if f.exists() else {"champion": None, "history": [], "versions": []}


def _write(reg: dict) -> None:
    _reg_file().write_text(json.dumps(reg, indent=1, default=str))


def register(bundle: ModelBundle, metrics: dict | None = None, promote: bool = False, reason: str = "") -> None:
    bundle.save()
    reg = registry()
    reg["versions"] = [v for v in reg["versions"] if v["version"] != bundle.version]
    reg["versions"].append({"version": bundle.version, "config_name": bundle.config_name,
                            "created_at": str(utcnow()), "metrics": metrics or {}})
    if promote:
        reg["history"].append({"from": reg["champion"], "to": bundle.version, "at": str(utcnow()), "reason": reason})
        reg["champion"] = bundle.version
    _prune(reg)
    _write(reg)
    log.info("registered %s%s", bundle.version, " (champion)" if promote else "")


def _prune(reg: dict) -> None:
    keep = get_config()["gate"]["keep_versions"]
    vers = reg["versions"]
    if len(vers) <= keep:
        return
    protected = {reg["champion"]} | {h["from"] for h in reg["history"][-3:] if h.get("from")}
    drop = [v for v in vers[:-keep] if v["version"] not in protected]
    for v in drop:
        shutil.rmtree(store_dir("models", v["version"]), ignore_errors=True)
    names = {v["version"] for v in drop}
    reg["versions"] = [v for v in vers if v["version"] not in names]


def champion() -> ModelBundle | None:
    v = registry()["champion"]
    return ModelBundle.load(v) if v else None


def rollback(to_version: str | None = None) -> str:
    reg = registry()
    if to_version is None:
        prev = [h["from"] for h in reversed(reg["history"]) if h.get("from") and h["from"] != reg["champion"]]
        if not prev:
            raise RuntimeError("no previous champion to roll back to")
        to_version = prev[0]
    if not store_dir("models", to_version).joinpath("meta.json").exists():
        raise RuntimeError(f"version {to_version} not on disk")
    reg["history"].append({"from": reg["champion"], "to": to_version, "at": str(utcnow()), "reason": "rollback"})
    reg["champion"] = to_version
    _write(reg)
    return to_version

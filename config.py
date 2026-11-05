"""Configuration loading: config.toml + .env (secrets)."""
from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@lru_cache(maxsize=4)
def _load(path: str, store_override: str | None) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    store = Path(store_override or cfg["paths"]["store"])
    if not store.is_absolute():
        store = ROOT / store
    cfg["paths"]["store"] = str(store)
    return cfg


def get_config() -> dict:
    path = os.environ.get("FINTAILOR_CONFIG", str(ROOT / "config.toml"))
    return _load(path, os.environ.get("FINTAILOR_STORE"))


def reset_config() -> None:
    _load.cache_clear()


def store_dir(*parts: str) -> Path:
    p = Path(get_config()["paths"]["store"]).joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def horizons() -> list[int]:
    return list(get_config()["labels"]["horizons"])


def cost() -> float:
    return get_config()["labels"]["cost_bps"] / 1e4


def sector_etfs() -> dict[str, str]:
    return dict(get_config()["universe"]["sector_etfs"])

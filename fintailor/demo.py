"""Synthetic end-to-end demo in its own store: data -> panel -> backtest -> train -> simulate live
record -> nightly loop (no collection) -> signals. Runs offline in about a minute."""
from __future__ import annotations

import os

from .config import ROOT, reset_config


def run_demo(n_stocks: int = 120) -> dict:
    if not os.environ.get("FINTAILOR_STORE"):
        os.environ["FINTAILOR_STORE"] = str(ROOT / "store_demo")
    os.environ["FINTAILOR_OFFLINE"] = "1"
    reset_config()
    import shutil

    from .config import get_config
    shutil.rmtree(get_config()["paths"]["store"], ignore_errors=True)

    from .data.synthetic import load_synthetic
    from .eval.backtest import run_backtest
    from .features.panel import build_panel
    from .llm.news_scorer import score_pending
    from .loop.nightly import run_nightly, simulate
    from .signals.profiles import SAMPLE_CUSTOMERS, save_customer

    out = {"store": get_config()["paths"]["store"], "loaded": load_synthetic(n_stocks=n_stocks)}
    for c in SAMPLE_CUSTOMERS:
        save_customer(c)
    out["lexicon"] = score_pending(force_lexicon=True)
    panel = build_panel()
    bt = run_backtest(panel, fast=True)
    out["backtest"] = {k: {m: v.get(m) for m in ("net_per_trade", "hit_rate", "rank_ic", "sharpe")}
                       for k, v in bt.items() if not k.startswith("_")}
    out["simulate"] = simulate(days=80, retrain_every=40)
    rep = run_nightly(collect=False, challengers=True, asof=str(panel["date"].max().date()))
    out["nightly"] = {k: rep.get(k) for k in ("run_date", "steps", "signals", "kill_switch")}
    return out

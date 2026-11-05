"""Command line: python -m fintailor <command> [options]   (see README for the workflow)"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _p(obj) -> None:
    print(json.dumps(obj, indent=1, default=str, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="fintailor")
    ap.add_argument("--store", help="override store directory (e.g. a scratch store for experiments)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create DB schema + sample customers")
    d = sub.add_parser("demo", help="synthetic end-to-end run in a separate store (no network)")
    d.add_argument("--stocks", type=int, default=120)

    sub.add_parser("universe", help="refresh S&P 500 membership history + ETFs")
    b = sub.add_parser("backfill", help="download price history for the universe")
    b.add_argument("--source", choices=["yfinance", "alpaca", "tiingo"])
    b.add_argument("--start")
    b.add_argument("--tickers", nargs="*")
    sub.add_parser("update-prices", help="incremental price update")
    c = sub.add_parser("collect", help="collect one data family")
    c.add_argument("what", choices=["news", "news-backfill", "rss", "fred", "epu-gpr", "events", "gdelt", "earnings",
                                    "earnings-history", "filings", "fundamentals", "all"])
    sub.add_parser("score-news", help="structure pending news (LLM or lexicon)").add_argument("--lexicon", action="store_true")
    sub.add_parser("features", help="build the point-in-time panel")
    bl = sub.add_parser("baseline", help="baseline strategies (targets to beat)")
    bl.add_argument("--start")
    bt = sub.add_parser("backtest", help="walk-forward backtest")
    bt.add_argument("--fast", action="store_true")
    bt.add_argument("--config", default="base")
    bt.add_argument("--horizons", type=int, nargs="*")
    bt.add_argument("--start")
    tr = sub.add_parser("train", help="train a model bundle on the latest data and make it champion")
    tr.add_argument("--config", default="base")
    tr.add_argument("--no-promote", action="store_true")
    sub.add_parser("predict", help="predict the latest date with the champion")
    sub.add_parser("signals", help="generate client signals for the latest date")
    n = sub.add_parser("nightly", help="full nightly loop")
    n.add_argument("--no-collect", action="store_true")
    n.add_argument("--no-retrain", action="store_true")
    n.add_argument("--no-challengers", action="store_true")
    n.add_argument("--no-signals", action="store_true")
    n.add_argument("--asof")
    s = sub.add_parser("simulate", help="replay the last N days as a paper live record")
    s.add_argument("--days", type=int, default=120)
    s.add_argument("--retrain-every", type=int, default=21)
    sub.add_parser("review", help="post-mortem of matured predictions")
    rb = sub.add_parser("rollback", help="restore previous champion")
    rb.add_argument("--to")
    rs = sub.add_parser("resume", help="clear the kill switch after human review")
    rs.add_argument("--note", default="")
    sub.add_parser("status", help="system status")
    sub.add_parser("dashboard", help="launch Streamlit dashboard")
    sub.add_parser("api", help="launch FastAPI server")
    a = ap.parse_args(argv)

    if a.store:
        os.environ["FINTAILOR_STORE"] = a.store
    from .config import reset_config
    reset_config()
    from .utils import setup_logging
    setup_logging("DEBUG" if a.verbose else "INFO")

    if a.cmd == "init":
        from .db import connect
        from .signals.profiles import SAMPLE_CUSTOMERS, save_customer
        with connect():
            pass
        for cst in SAMPLE_CUSTOMERS:
            save_customer(cst)
        print("initialized; sample customers:", [c.customer_id for c in SAMPLE_CUSTOMERS])
    elif a.cmd == "demo":
        from .demo import run_demo
        _p(run_demo(a.stocks))
    elif a.cmd == "universe":
        from .data.universe import refresh_universe
        refresh_universe()
    elif a.cmd == "backfill":
        from .data.prices import backfill
        backfill(a.tickers, a.start, a.source)
    elif a.cmd == "update-prices":
        from .data.prices import update_recent
        print(update_recent())
    elif a.cmd == "collect":
        from .data import earnings, gdelt, macro, news, sec
        fns = {"news": news.collect_news, "news-backfill": lambda: news.collect_news(backfill=True),
               "rss": news.collect_rss, "fred": macro.collect_fred, "epu-gpr": macro.collect_epu_gpr,
               "events": macro.collect_events, "gdelt": gdelt.collect_gdelt,
               "earnings": earnings.collect_finnhub_calendar, "earnings-history": earnings.collect_yf_history,
               "filings": sec.collect_filings, "fundamentals": sec.collect_fundamentals}
        if a.what == "all":
            from .loop.nightly import collect_all
            _p(collect_all())
        else:
            print(fns[a.what]())
    elif a.cmd == "score-news":
        from .llm.news_scorer import score_pending
        _p(score_pending(force_lexicon=a.lexicon))
    elif a.cmd == "features":
        from .features.panel import build_panel
        build_panel()
    elif a.cmd == "baseline":
        from .eval.backtest import run_baselines
        from .eval.metrics import summary_table
        from .features.panel import load_panel
        r = run_baselines(load_panel(), a.start)
        print(summary_table(r).to_string(float_format=lambda x: f"{x:.4f}"))
    elif a.cmd == "backtest":
        from .eval.backtest import run_backtest
        from .features.panel import load_panel
        r = run_backtest(load_panel(), fast=a.fast, config_name=a.config, hs=a.horizons, start_test=a.start)
        for k, v in r.items():
            if isinstance(v, dict) and "gate" in v:
                print(k, "gate:", v["gate"])
    elif a.cmd == "train":
        from .features.panel import load_panel
        from .models.registry import register, train_bundle
        p = load_panel()
        bnd = train_bundle(p, p["date"].max(), a.config)
        register(bnd, promote=not a.no_promote, reason="manual train")
        _p(bnd.meta["train"])
    elif a.cmd == "predict":
        from .features.panel import load_panel
        from .loop.steps import predict_and_log
        p = load_panel()
        out = predict_and_log(p, p["date"].max())
        print(out.sort_values(["horizon", "score"], ascending=[True, False]).groupby("horizon").head(10)
              [["ticker", "horizon", "prob", "pred_excess", "score"]].to_string())
    elif a.cmd == "signals":
        from .features.panel import load_panel
        from .signals.engine import generate_signals
        p = load_panel()
        d = p["date"].max()
        sig = generate_signals(d, p[p["date"] == d])
        for r in sig.itertuples():
            print(f"[{r.customer_id}] {r.status} {r.ticker or ''}\n  {r.explanation}\n")
    elif a.cmd == "nightly":
        from .loop.nightly import run_nightly
        rep = run_nightly(collect=not a.no_collect, retrain=not a.no_retrain, challengers=not a.no_challengers,
                          make_signals=not a.no_signals, asof=a.asof)
        _p({k: rep.get(k) for k in ("run_date", "steps", "live", "kill_switch", "signals")})
    elif a.cmd == "simulate":
        from .loop.nightly import simulate
        _p(simulate(a.days, a.retrain_every))
    elif a.cmd == "review":
        from .llm.reviewer import review_matured
        _p(review_matured())
    elif a.cmd == "rollback":
        from .models.registry import rollback
        print("champion ->", rollback(a.to))
    elif a.cmd == "resume":
        from .loop.steps import resume
        resume(a.note)
        print("kill switch cleared")
    elif a.cmd == "status":
        from .status import status
        _p(status())
    elif a.cmd == "dashboard":
        import subprocess
        from .config import ROOT
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "fintailor" / "app" / "dashboard.py")])
    elif a.cmd == "api":
        import uvicorn
        uvicorn.run("fintailor.app.api:app", host="127.0.0.1", port=8000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

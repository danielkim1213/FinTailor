"""Nightly loop: collect -> check -> score news -> features -> grade matured predictions ->
evaluate/kill switch -> post-mortem -> retrain + gate -> predict -> signals -> report.

Every step is logged in run_log with timing; failures in non-critical steps don't stop the run.
Run after the US close (e.g. 17:30 ET). Idempotent: re-running the same night overwrites.
"""
from __future__ import annotations

import json
import time
import traceback

import numpy as np
import pandas as pd

from ..config import get_config, store_dir
from ..db import connect, get_state, set_state, upsert
from ..utils import log, new_id, utcnow
from . import steps


class Run:
    def __init__(self):
        self.id = new_id("run_")
        self.report: dict = {"run_id": self.id, "started_at": str(utcnow()), "steps": {}}

    def step(self, name: str, fn, *a, critical: bool = False, **kw):
        t0, start = time.time(), utcnow()
        status, detail, res = "ok", {}, None
        try:
            res = fn(*a, **kw)
            detail = res if isinstance(res, dict) else {"result": str(res)[:500]}
        except Exception as e:  # noqa: BLE001
            status, detail = "error", {"error": str(e), "trace": traceback.format_exc()[-2000:]}
            log.error("step %s failed: %s", name, e)
        self.report["steps"][name] = {"status": status, "seconds": round(time.time() - t0, 1),
                                      **({"error": detail.get("error")} if status == "error" else {})}
        try:
            with connect() as con:
                upsert(con, "run_log", pd.DataFrame([{"run_id": self.id, "step": name, "status": status,
                                                      "detail": json.dumps(_jsonable(detail), default=str)[:20000],
                                                      "started_at": start, "finished_at": utcnow()}]), mode="append")
        except Exception as e:  # noqa: BLE001
            log.warning("run_log write failed: %s", e)
        if status == "error" and critical:
            raise RuntimeError(f"critical step {name} failed: {detail.get('error')}")
        return res


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items() if not isinstance(v, pd.DataFrame)}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def _due(name: str, days: int) -> bool:
    with connect() as con:
        last = get_state(con, f"last_{name}")
        if last and (utcnow() - pd.Timestamp(last)).days < days:
            return False
        set_state(con, f"last_{name}", str(utcnow()))
    return True


def collect_all() -> dict:
    from ..data import earnings, gdelt, macro, news, prices, sec
    out = {}
    for name, fn, every in (("prices", prices.update_recent, 0),("news", news.collect_news, 0), ("gdelt", gdelt.collect_gdelt, 0),
                            ("fred", macro.collect_fred, 0), ("epu_gpr", macro.collect_epu_gpr, 3),
                            ("events", macro.collect_events, 7), ("earnings_cal", earnings.collect_finnhub_calendar, 0),
                            ("filings", sec.collect_filings, 1), ("fundamentals", sec.collect_fundamentals, 7)):
        if every and not _due(name, every):
            continue
        try:
            out[name] = fn()
        except Exception as e:  # noqa: BLE001
            out[name] = f"error: {e}"
            log.warning("collect %s: %s", name, e)
    return out


def run_nightly(collect: bool = True, retrain: bool = True, challengers: bool = True, make_signals: bool = True,
                review: bool = True, asof: str | None = None) -> dict:
    from ..data.health import check_data
    from ..features.panel import build_panel
    from ..llm.news_scorer import score_pending, sector_view
    from ..llm.reviewer import review_matured
    from ..models.registry import champion, register, train_bundle
    from ..signals.engine import generate_signals

    run = Run()
    if collect:
        run.step("collect", collect_all)
    with connect(read_only=True) as con:
        health = check_data(con, pd.Timestamp(asof) if asof else None)
    run.report["health"] = health
    run.step("news_score", score_pending)
    if health.get("last_date"):
        run.step("sector_view", sector_view, pd.Timestamp(asof or health["last_date"]))
    panel = run.step("features", build_panel, critical=True)
    run_date = pd.Timestamp(asof) if asof else panel["date"].max()
    run.report["run_date"] = str(run_date.date())
    run.step("outcomes", steps.score_outcomes, panel)
    live = run.step("live_eval", steps.live_metrics) or {}
    run.report["live"] = {k: v for k, v in live.items() if not isinstance(v, dict)}
    run.report["drift"] = run.step("drift", steps.drift_report, panel, run_date)
    run.report["kill_switch"] = run.step("kill_switch", steps.kill_switch_check, health, live)
    if review:
        run.step("review", review_matured)
    if retrain:
        if champion() is None:
            run.step("bootstrap_train", lambda: register(train_bundle(panel, run_date, "base"), promote=True,
                                                         reason="bootstrap") or {"bootstrap": True}, critical=True)
        else:
            run.report["gate"] = run.step("retrain_gate", steps.retrain_and_gate, panel, run_date, challengers)
    run.step("predict", steps.predict_and_log, panel, run_date, critical=True)
    halted = (run.report.get("kill_switch") or {}).get("halted", False)
    if make_signals and not halted:
        sig = run.step("signals", generate_signals, run_date, panel[panel["date"] == run_date])
        if sig is not None and len(sig):
            run.report["signals"] = sig["status"].value_counts().to_dict()
    elif halted:
        run.report["signals"] = "suppressed: kill switch active"
    run.report["finished_at"] = str(utcnow())
    write_report(run.report)
    return run.report


def write_report(rep: dict) -> None:
    d = store_dir("reports")
    day = rep.get("run_date", str(utcnow().date()))
    (d / f"{day}.json").write_text(json.dumps(_jsonable(rep), indent=1, default=str))
    lines = [f"# FinTailor nightly {day}", "", f"run `{rep['run_id']}`", ""]
    lines += [f"- **{k}**: {v['status']} ({v['seconds']}s){' — ' + v['error'] if v.get('error') else ''}"
              for k, v in rep["steps"].items()]
    lv = rep.get("live", {})
    if lv.get("n_trades"):
        lines += ["", f"live (last {get_config()['killswitch']['lookback_days']}d): net/trade {lv.get('net_per_trade', 0):.4f}, "
                      f"hit {lv.get('hit_rate', 0):.3f}, IC {lv.get('rank_ic', 0):.4f}"]
    lines += ["", f"kill switch: {rep.get('kill_switch')}", f"gate: {json.dumps(_jsonable(rep.get('gate')), default=str)[:800]}",
              f"signals: {rep.get('signals')}", f"drift alerts: {(rep.get('drift') or {}).get('alerts')}"]
    (d / f"{day}.md").write_text("\n".join(lines), encoding="utf-8")
    if get_config()["tracking"].get("mlflow"):
        try:
            import mlflow
            with mlflow.start_run(run_name=f"nightly_{day}"):
                for k, v in (rep.get("live") or {}).items():
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        mlflow.log_metric(f"live_{k}", v)
                mlflow.log_dict(_jsonable(rep), "report.json")
        except Exception as e:  # noqa: BLE001
            log.warning("mlflow logging failed: %s", e)


def simulate(days: int = 120, retrain_every: int = 21, make_signals: bool = True) -> dict:
    """Replay the last `days` trading days as if the loop had been running live (paper record).
    Uses the same purging; retrains every `retrain_every` days. Fast way to test the loop and
    populate the dashboard."""
    from ..features.panel import build_panel, load_panel
    from ..models.registry import register, train_bundle
    from ..signals.engine import generate_signals

    try:
        panel = load_panel()
    except FileNotFoundError:
        panel = build_panel()
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    sim = dates[-days:]
    bundle = None
    for i, d in enumerate(sim):
        if bundle is None or i % retrain_every == 0:
            bundle = train_bundle(panel, d, "base")
            register(bundle, promote=True, reason=f"simulate {d.date()}")
        steps.predict_and_log(panel, d, bundle)
        if make_signals:
            generate_signals(d, panel[panel["date"] == d])
        if i % 20 == 0:
            log.info("simulate %d/%d %s", i + 1, len(sim), d.date())
    steps.score_outcomes(panel)
    live = steps.live_metrics()
    return {"days": len(sim), "live": {k: v for k, v in live.items() if not isinstance(v, dict)}}

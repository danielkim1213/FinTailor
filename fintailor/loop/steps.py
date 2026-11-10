"""Building blocks of the nightly loop: predict+log, score outcomes, live metrics, kill switch,
retrain + challenger gate."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import cost, get_config, horizons
from ..db import connect, get_state, replace_where, set_state, upsert
from ..eval.drift import drift_report
from ..eval.metrics import evaluate
from ..models.lgbm import fit_horizon, ranking_score, select_features
from ..models.registry import (ModelBundle, champion, config_overrides, register, registry, train_bundle,
                               train_window)
from ..utils import log, utcnow


# ----------------------------------------------------------------------------- 1) log predictions
def predict_and_log(panel: pd.DataFrame, run_date: pd.Timestamp, bundle: ModelBundle | None = None,
                    official: bool = True) -> pd.DataFrame:
    bundle = bundle or champion()
    if bundle is None:
        raise RuntimeError("no champion model; run `fintailor train` first")
    rows = panel[panel["date"] == run_date]
    if rows.empty:
        raise RuntimeError(f"no panel rows for {run_date.date()}")
    pr = bundle.predict(rows, with_drivers=True)
    pr["rank_pct"] = pr.groupby("horizon")["score"].rank(pct=True)
    out = pd.DataFrame({"run_date": run_date.date(), "ticker": pr["ticker"], "horizon": pr["horizon"].astype(int),
                        "model_version": bundle.version, "pred_excess": pr["pred_excess"], "prob": pr["prob"],
                        "score": pr["score"], "rank_pct": pr["rank_pct"], "official": official,
                        "drivers": pr["drivers"], "created_at": utcnow()})
    with connect() as con:
        if official:  # one official set per run_date
            con.execute("update predictions set official = false where run_date = ? and model_version <> ?",
                        [run_date.date(), bundle.version])
        upsert(con, "predictions", out)
    return out


# ----------------------------------------------------------------------------- 2) score matured
def score_outcomes(panel: pd.DataFrame) -> int:
    """Attach realized labels to predictions whose horizon has passed."""
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    with connect(read_only=True) as con:
        pending = con.execute("""select distinct p.run_date, p.ticker, p.horizon from predictions p
            where not exists (select 1 from outcomes o where o.run_date = p.run_date and o.ticker = p.ticker
                              and o.horizon = p.horizon)""").df()
    if pending.empty:
        return 0
    pending["run_date"] = pd.to_datetime(pending["run_date"])
    rows = []
    lab = panel[["date", "ticker"] + [c for c in panel.columns if c[:2] in ("y_", "g_")]].set_index(["date", "ticker"])
    for h, g in pending.groupby("horizon"):
        pos = dates.searchsorted(g["run_date"])
        mat_pos = pos + h
        ok = mat_pos < len(dates)
        g = g[ok]
        if g.empty:
            continue
        key = pd.MultiIndex.from_arrays([g["run_date"], g["ticker"]])
        y = lab[f"y_{h}"].reindex(key).values
        gg = lab[f"g_{h}"].reindex(key).values
        spy = panel.attrs.get("spy_fwd", {}).get(h)
        rows.append(pd.DataFrame({"run_date": g["run_date"].dt.date.values, "ticker": g["ticker"].values,
                                  "horizon": h, "realized_net": y, "realized_gross": gg,
                                  "spy_ret": np.nan if spy is None else spy.reindex(g["run_date"]).values,
                                  "matured_date": dates[mat_pos[ok]].date}))
    if not rows:
        return 0
    df = pd.concat(rows, ignore_index=True).dropna(subset=["realized_net"])
    if df["spy_ret"].isna().all():
        df["spy_ret"] = _spy_window(df)
    with connect() as con:
        n = upsert(con, "outcomes", df)
    log.info("scored %d matured predictions", n)
    return n


def _spy_window(df: pd.DataFrame) -> np.ndarray:
    bench = get_config()["universe"]["benchmark"]
    with connect(read_only=True) as con:
        s = con.execute("select date, open, close, adj_close from prices_daily where ticker = ? order by date", [bench]).df()
    s["date"] = pd.to_datetime(s["date"])
    s = s.set_index("date")
    o = s["open"] * s["adj_close"] / s["close"]
    c = s["adj_close"]
    idx = s.index
    pairs = df[["run_date", "matured_date"]].drop_duplicates().copy()
    vals = []
    for rd, md in zip(pd.to_datetime(pairs["run_date"]), pd.to_datetime(pairs["matured_date"])):
        i = idx.searchsorted(rd) + 1
        vals.append(c.get(md, np.nan) / o.iloc[i] - 1 if i < len(idx) else np.nan)
    pairs["spy"] = vals
    return df[["run_date", "matured_date"]].merge(pairs, how="left")["spy"].values


# ----------------------------------------------------------------------------- 3) live metrics
def live_metrics(lookback: int | None = None, h: int | None = None) -> dict:
    kc = get_config()["killswitch"]
    lookback = lookback or kc["lookback_days"]
    h = h or kc["primary_horizon"]
    with connect(read_only=True) as con:
        df = con.execute("""select p.run_date date, p.ticker, p.score, p.prob, o.realized_net y, o.realized_gross g
            from predictions p join outcomes o using (run_date, ticker, horizon)
            where p.official and p.horizon = ? and p.run_date >= (
              select min(d) from (select distinct run_date d from outcomes where horizon = ? order by d desc limit ?))""",
                         [h, h, lookback]).df()
    if df.empty:
        return {"n_days": 0, "n_trades": 0}
    df["date"] = pd.to_datetime(df["date"])
    return evaluate(df, h, get_config()["backtest"]["top_k"], cost(), by_regime=False)


def kill_switch_check(health: dict, live: dict) -> dict:
    kc = get_config()["killswitch"]
    with connect() as con:
        st = get_state(con, "kill_switch", {"halted": False})
        reasons = []
        if not health.get("ok", True):
            reasons += [f"data: {x}" for x in health.get("issues", [])]
        if live.get("n_trades", 0) >= kc["min_samples"]:
            if live["net_per_trade"] < kc["min_net_per_trade"]:
                reasons.append(f"live net/trade {live['net_per_trade']:.4f} < {kc['min_net_per_trade']}")
            if live["hit_rate"] < kc["min_hit_rate"]:
                reasons.append(f"live hit rate {live['hit_rate']:.3f} < {kc['min_hit_rate']}")
        if reasons and not st.get("halted"):
            st = {"halted": True, "reasons": reasons, "since": str(utcnow())}
            set_state(con, "kill_switch", st)
            log.error("KILL SWITCH TRIPPED: %s", reasons)
        elif reasons:
            st["reasons"] = sorted(set(st.get("reasons", []) + reasons))
            set_state(con, "kill_switch", st)
    return st


def resume(note: str = "") -> None:
    with connect() as con:
        set_state(con, "kill_switch", {"halted": False, "resumed_at": str(utcnow()), "note": note})


# ----------------------------------------------------------------------------- 4) retrain + gate
def _holdout_metric(panel: pd.DataFrame, run_date: pd.Timestamp, config_name: str, h: int, holdout: int) -> dict:
    """Train config at cutoff (run_date - holdout - purge) and evaluate on the last `holdout` matured days."""
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    end_pos = dates.searchsorted(run_date) - h            # last date whose h-label is known
    ho_dates = dates[max(0, end_pos - holdout + 1): end_pos + 1]
    test = panel[panel["date"].isin(ho_dates)]
    ov = config_overrides(config_name)
    feats = select_features(panel.attrs["meta"]["features"], ov)
    tr = train_window(panel, ho_dates[0], h)
    m = fit_horizon(tr, feats, h, ov)
    pr = m.predict(test)
    d = pd.DataFrame({"date": test["date"].values, "ticker": test["ticker"].values, "score": ranking_score(pr, ov.get("rank_by")).values,
                      "prob": pr["prob"].values, "y": test[f"y_{h}"].values, "g": test[f"g_{h}"].values})
    return evaluate(d, h, get_config()["backtest"]["top_k"], cost(), by_regime=False)


def retrain_and_gate(panel: pd.DataFrame, run_date: pd.Timestamp, challengers: bool = True) -> dict:
    g = get_config()["gate"]
    reg = registry()
    champ = champion()
    champ_cfg = champ.config_name if champ else "base"
    metric = g["metric"]
    result = {"champion_before": reg.get("champion"), "compared": {}}
    names = [champ_cfg] + ([c["name"] for c in get_config().get("challengers", []) if c["name"] != champ_cfg]
                           if challengers else [])
    if champ_cfg != "base" and "base" not in names and challengers:
        names.append("base")
    scores = {}
    for name in names:
        vals = []
        for h in g["gate_horizons"]:
            try:
                r = _holdout_metric(panel, run_date, name, h, g["holdout_days"])
                vals.append(r.get(metric, np.nan))
                result["compared"][f"{name}@{h}"] = {k: r.get(k) for k in ("rank_ic", "net_per_trade", "hit_rate", "sharpe")}
            except Exception as e:  # noqa: BLE001
                log.warning("gate eval %s h=%d failed: %s", name, h, e)
                vals.append(np.nan)
        scores[name] = float(np.nanmean(vals)) if np.isfinite(vals).any() else -np.inf
    best = max(scores, key=scores.get)
    winner = champ_cfg
    if best != champ_cfg and scores[best] >= scores.get(champ_cfg, -np.inf) + g["min_improvement"]:
        winner = best
    result.update(scores=scores, winner_config=winner)

    # nightly refit of the winning config on the latest data (weights change daily)
    new = train_bundle(panel, run_date, winner)
    rows = panel[panel["date"] == run_date]
    promote, reason = True, f"nightly refit ({winner})"
    if champ is not None and len(rows):
        a = new.predict(rows).query(f"horizon == {g['gate_horizons'][0]}")["score"]
        b = champ.predict(rows).query(f"horizon == {g['gate_horizons'][0]}")["score"]
        corr = float(pd.Series(a.values).corr(pd.Series(b.values), method="spearman"))
        result["pred_corr_vs_prev"] = corr
        bad = not np.isfinite(a).all() or (winner == champ_cfg and corr < g["min_pred_corr"])
        if bad:
            promote, reason = False, f"sanity failed (corr {corr:.2f})"
    if winner != champ_cfg:
        reason = f"challenger {winner} beat {champ_cfg} on {metric}: {scores[winner]:.4f} vs {scores.get(champ_cfg, float('nan')):.4f}"
    register(new, {"gate": result}, promote=promote, reason=reason)
    result.update(new_version=new.version, promoted=promote, reason=reason)
    return result


__all__ = ["predict_and_log", "score_outcomes", "live_metrics", "kill_switch_check", "resume", "retrain_and_gate",
           "drift_report", "horizons", "json", "replace_where"]

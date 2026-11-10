"""Strategy engine.

Book: every index member at its market-cap weight, leaned toward stocks with better combined forecasts. Starting from cap
weights means the book carries no structural handicap against SPY; whatever it earns beyond its untilted control is the
forecasts' doing. The combined forecast is an equal-weight average of three sources chosen in advance, so nothing here depends
on which of them happened to do best in the test period.

    strategy-validate   out-of-sample test up to the locked holdout, with statistics that account for the number of ideas tried
    strategy-today      target weights for the next trading day
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import get_config, store_dir
from ..db import connect
from ..utils import log
from .signals import factor_score, ml_features, ml_fit_predict, ml_walk_forward, sector_score, zscore
from .simulator import load_market, run_book, tilt_weights
from .stats import block_bootstrap, deflated_sharpe, newey_west_t, sharpe

STRENGTH = 0.5           # tilt per unit of score; a one-sigma stock is held at about 1.65x its cap weight
MAX_MULT = 3.0           # no stock above 3x its cap weight
REBALANCE_DAYS = 5
COST = 0.001             # round trip on traded weight
TEST_START = "2019-01-01"
DEV_END = pd.Timestamp("2023-12-31")
IDEAS_TRIED = 90         # configurations examined over the project's life, used to deflate the final statistic
IDEA_SPREAD = 0.30       # spread of annualised information ratios across those ideas (ledger and backtest summaries)


def _load_panel() -> tuple[pd.DataFrame, dict]:
    d = store_dir("features")
    meta = json.loads((d / "panel_meta.json").read_text())
    p = pd.read_parquet(d / "panel.parquet")
    return p[~p["is_etf_flag"].astype(bool)].reset_index(drop=True), meta


def _scores(p: pd.DataFrame, con, ml: pd.Series | None) -> pd.DataFrame:
    out = p[["date", "ticker", "mcap"]].copy()
    out["z_factor"] = factor_score(p)
    out["z_sector"] = sector_score(p, con)
    out["z_ml"] = ml.reindex(p.index) if ml is not None else np.nan
    has_ml = out["z_ml"].notna()
    total = out["z_factor"] + out["z_sector"] + out["z_ml"].fillna(0.0)
    out["z"] = zscore(total.where(has_ml | (ml is None)), out["date"])
    return out


def _book(s: pd.DataFrame, column: str | None, offset: int, strength: float = STRENGTH) -> dict:
    targets = {}
    dates = np.sort(s["date"].unique())[offset::REBALANCE_DAYS]
    for d, g in s[s["date"].isin(dates)].groupby("date"):
        f = pd.DataFrame({"ticker": g["ticker"].to_numpy(), "mcap": g["mcap"].to_numpy(float), "z": g[column].to_numpy(float) if column else 0.0})
        targets[pd.Timestamp(d)] = tilt_weights(f, strength, MAX_MULT)
    return targets


def _mean_run(m, s: pd.DataFrame, column: str | None) -> pd.DataFrame:
    runs = [run_book(m, _book(s, column, i), COST) for i in range(REBALANCE_DAYS)]
    return pd.DataFrame({"ret": pd.concat([r["ret"] for r in runs], axis=1).mean(axis=1),
                         "turnover": pd.concat([r["turnover"] for r in runs], axis=1).mean(axis=1)})


def _report(r: pd.DataFrame, ctrl: pd.Series, spy: pd.Series) -> dict:
    a = (r["ret"] - ctrl).dropna()
    x = (r["ret"] - spy.reindex(r.index)).dropna()
    bb = block_bootstrap(a)
    return {"vs_spy_%": x.mean() * 25200, "active_%": a.mean() * 25200, "tracking_%": a.std() * np.sqrt(252) * 100, "ir": sharpe(a), "t": newey_west_t(a),
            "ci95_lo_%": bb["lo"] * 100, "ci95_hi_%": bb["hi"] * 100, "p_positive_%": bb["p_positive"] * 100,
            "dev_%": a[a.index <= DEV_END].mean() * 25200, "val_%": a[a.index > DEV_END].mean() * 25200, "turnover_x": r["turnover"].mean() * 252,
            "by_year": {int(y): round(float(v * 25200), 1) for y, v in a.groupby(a.index.year).mean().items()}}


def validate(final: bool = False, fresh_ml: bool = False) -> dict:
    """Test on data the forecasts did not train on. final=True includes the locked holdout, which may be done only once."""
    cfg = get_config()
    hold = pd.Timestamp(cfg["backtest"]["holdout_start"])
    used = store_dir("strategy") / "holdout_used.json"
    if final and used.exists():
        raise RuntimeError(f"The holdout was already used ({used.read_text()}). Re-testing on it would no longer be an independent test.")
    end = None if final else hold - pd.Timedelta(days=1)
    p, meta = _load_panel()
    cache = store_dir("strategy") / ("ml_oos_final.parquet" if final else "ml_oos.parquet")
    if cache.exists() and not fresh_ml:
        ml = pd.read_parquet(cache)["ml"]
    else:
        ml = ml_walk_forward(p, ml_features(meta), TEST_START, None if end is None else str(end.date()), log=log.info)
        ml.to_frame("ml").to_parquet(cache)
    keep = p["date"] >= pd.Timestamp(TEST_START)
    if end is not None:
        keep &= p["date"] <= end
    with connect(read_only=True) as con:
        s = _scores(p, con, ml)[keep].dropna(subset=["z_ml"]).reset_index(drop=True)
        m = load_market(con, "2018-09-01", None if end is None else str(end.date()))
    control = _mean_run(m, s, None)
    x = (control["ret"] - m.spy.reindex(control.index)).dropna()
    results = {"_control_vs_spy_%": x.mean() * 25200, "_control_tracking_%": x.std() * np.sqrt(252) * 100}
    for name, col in (("factors only", "z_factor"), ("sector only", "z_sector"), ("ml only", "z_ml"), ("COMBINED (the strategy)", "z")):
        results[name] = _report(_mean_run(m, s, col), control["ret"], m.spy)
    combined = (_mean_run(m, s, "z")["ret"] - control["ret"]).dropna()
    results["deflated_probability_%"] = 100 * deflated_sharpe(combined, IDEAS_TRIED, IDEA_SPREAD / np.sqrt(252))
    results["days"] = len(combined)
    if final:
        used.write_text(json.dumps({"at": pd.Timestamp.now().isoformat(timespec="seconds")}))
    (store_dir("strategy") / ("validation_final.json" if final else "validation.json")).write_text(json.dumps(results, indent=1, default=float))
    return results


def today() -> dict:
    """Target weights for the next trading day, using only what is known at the latest close."""
    p, meta = _load_panel()
    last = pd.Timestamp(p["date"].max())
    now = p[p["date"] == last]
    log.info("training forecast models on data known at %s", last.date())
    ml = ml_fit_predict(p, ml_features(meta), last, now)
    with connect(read_only=True) as con:
        s = _scores(now, con, ml)
    s = s.dropna(subset=["z"])
    s["weight"] = s["ticker"].map(tilt_weights(s[["ticker", "mcap", "z"]], STRENGTH, MAX_MULT))
    s["cap_weight"] = s["mcap"] / s["mcap"].sum()
    s["active"] = s["weight"] - s["cap_weight"]
    s["rank_pct"] = s["z"].rank(pct=True)
    s = s.merge(now[["ticker", "sector"]].assign(sector=now["sector"].astype(str)), on="ticker")
    out: Path = store_dir("strategy") / f"target_{last.date()}.csv"
    cols = ["ticker", "sector", "weight", "cap_weight", "active", "rank_pct", "z", "z_factor", "z_sector", "z_ml"]
    s.sort_values("weight", ascending=False)[cols].to_csv(out, index=False, float_format="%.5f")
    return {"as_of": str(last.date()), "file": str(out), "names": len(s), "one_way_active_share": float(s["active"].abs().sum() / 2), "table": s[cols]}


def print_validation(res: dict) -> None:
    print(f"\ncontrol (no forecasts) vs SPY: {res['_control_vs_spy_%']:+.2f}%/yr, tracking {res['_control_tracking_%']:.2f}%")
    keys = ["active_%", "tracking_%", "ir", "t", "ci95_lo_%", "ci95_hi_%", "p_positive_%", "dev_%", "val_%", "turnover_x", "vs_spy_%"]
    rows = {k: {c: v[c] for c in keys} for k, v in res.items() if isinstance(v, dict)}
    print(pd.DataFrame(rows).T.round(2).to_string())
    print("\nactive return by year (%):")
    print(pd.DataFrame({k: v["by_year"] for k, v in res.items() if isinstance(v, dict)}).T.to_string())
    print(f"\nProbability the combined edge is real after allowing for {IDEAS_TRIED} ideas tried: {res['deflated_probability_%']:.1f}%")


def print_today(res: dict) -> None:
    t = res["table"]
    print(f"\nTarget book for the session after {res['as_of']}: {res['names']} stocks; {res['one_way_active_share'] * 100:.1f}% differs from cap weights. Saved to {res['file']}")
    print("\nlargest overweights:\n", t.nlargest(12, "active")[["ticker", "sector", "weight", "cap_weight", "rank_pct"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nlargest underweights:\n", t.nsmallest(12, "active")[["ticker", "sector", "weight", "cap_weight", "rank_pct"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    sec = t.groupby("sector")[["weight", "cap_weight"]].sum()
    sec["tilt"] = sec["weight"] - sec["cap_weight"]
    print("\nsector weights:\n", sec.sort_values("tilt", ascending=False).to_string(float_format=lambda v: f"{v:.3f}"))

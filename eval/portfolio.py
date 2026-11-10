"""Portfolio-level backtest of model scores: what a real book would have earned vs SPY.

Per-trade averages (metrics.evaluate) hide the things that decide real returns: how many names, how they
are weighted, how much is traded, and how sensitive the result is to the rebalance day. This simulates a
weekly-rebalanced long-only book on top of out-of-sample scores:

- P&L uses the 5-day forward excess return vs SPY (g_5: next open -> close 5 days later), so any score
  horizon (or an ensemble of horizons) is judged on the same weekly holding period.
- buffer rule: a held name stays while it ranks inside top k*buffer (cuts turnover), new names fill to k.
- optional sector cap and inverse-volatility weights (capped at 2x equal weight).
- cost: round-trip `cost` charged on traded weight (half per side).
- every rebalance offset (0..rebalance-1) is run separately; results report the mean and the range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ensemble_scores(oos: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Average of per-date z-scores across horizons (rows present for all requested horizons)."""
    parts = []
    for h in horizons:
        d = oos[oos["horizon"] == h][["date", "ticker", "score"]].copy()
        if d.empty:
            continue
        g = d.groupby("date")["score"]
        d["z"] = (d["score"] - g.transform("mean")) / (g.transform("std") + 1e-12)
        parts.append(d.set_index(["date", "ticker"])["z"].rename(h))
    if not parts:
        return pd.DataFrame()
    z = pd.concat(parts, axis=1).dropna()
    return z.mean(axis=1).rename("score").reset_index()


def simulate_tilt(scores: pd.DataFrame, rets: pd.DataFrame, strength: float = 0.5, rebalance: int = 5,
                  offset: int = 0, cost: float = 0.001, max_mult: float = 3.0) -> pd.DataFrame:
    """Enhanced index: hold every name at its market-cap weight times exp(strength * score z), capped at
    max_mult x cap weight, renormalised. r is already excess vs SPY, so sum(w * r) is the excess; the
    cap-weighted universe itself tracks SPY closely, so the result is mostly the active tilt."""
    dates = np.sort(scores["date"].unique())[offset::rebalance]
    s = scores[scores["date"].isin(dates)].merge(rets, on=["date", "ticker"], how="inner")
    held: dict[str, float] = {}
    rows = []
    for d, x in s.groupby("date", sort=True):
        x = x.dropna(subset=["score"])
        size = x["mcap"].where(x["mcap"] > 0)
        if size.notna().sum() < len(x) * 0.5:  # not enough market caps yet -> dollar volume proxy
            size = x["dvol"]
        x = x.assign(size=size.fillna(size.median()))
        if x.empty or x["size"].sum() <= 0:
            continue
        z = (x["score"] - x["score"].mean()) / (x["score"].std() + 1e-12)
        base = x["size"] / x["size"].sum()
        w = base * np.exp(strength * z.clip(-3, 3))
        w = np.minimum(w / w.sum(), base * max_mult)
        w = w / w.sum()
        new = dict(zip(x["ticker"], w))
        traded = sum(abs(new.get(t, 0.0) - held.get(t, 0.0)) for t in set(new) | set(held))
        bench = float((base * x["r"].fillna(0.0)).sum())  # cap-weighted universe vs SPY (tracking noise)
        gross = float((w * x["r"].fillna(0.0)).sum())
        rows.append({"date": d, "gross": gross, "turnover": traded / 2, "ret": gross - traded * cost / 2,
                     "n": len(x), "active": gross - bench})
        held = new
    return pd.DataFrame(rows)


def simulate(scores: pd.DataFrame, rets: pd.DataFrame, k: int = 50, buffer: float = 3.0, rebalance: int = 5,
             offset: int = 0, cost: float = 0.001, weighting: str = "inv_vol", sector_cap: float | None = 0.3,
             smooth: int = 1, score_tilt: float = 0.5, score_max_mult: float = 3.0,
             trace: list | None = None) -> pd.DataFrame:
    """scores: date, ticker, score. rets: date, ticker, r (5d fwd excess), vol, sector.
    Returns one row per rebalance date: ret (net), gross, turnover."""
    s = scores.copy()
    if smooth > 1:  # average of the last `smooth` daily scores per ticker (steadier ranks, less turnover)
        s = s.sort_values("date")
        s["score"] = s.groupby("ticker")["score"].transform(lambda x: x.rolling(smooth, min_periods=1).mean())
    dates = np.sort(s["date"].unique())[offset::rebalance]
    s = s[s["date"].isin(dates)].merge(rets, on=["date", "ticker"], how="inner")
    held: dict[str, float] = {}
    rows = []
    for d, x in s.groupby("date", sort=True):
        x = x.dropna(subset=["score"]).sort_values("score", ascending=False).reset_index(drop=True)
        if x.empty:
            continue
        x["rk"] = np.arange(len(x))
        inbuf = set(x.loc[x["rk"] < int(k * buffer), "ticker"])
        port: list[str] = [t for t in x["ticker"] if t in held and t in inbuf][:k]
        sec_n: dict[str, int] = {}
        sec_of = dict(zip(x["ticker"], x["sector"].astype(str)))
        for t in port:
            sec_n[sec_of[t]] = sec_n.get(sec_of[t], 0) + 1
        cap = int(np.ceil(sector_cap * k)) if sector_cap else k
        for t in x["ticker"]:
            if len(port) >= k:
                break
            if t in port:
                continue
            sc = sec_of[t]
            if sec_n.get(sc, 0) >= cap and sc not in ("unknown", "nan"):
                continue
            port.append(t)
            sec_n[sc] = sec_n.get(sc, 0) + 1
        sel = x.set_index("ticker").loc[port]
        if weighting == "inv_vol" and sel["vol"].notna().any():
            iv = 1.0 / sel["vol"].clip(lower=sel["vol"].quantile(0.05) if sel["vol"].notna().sum() > 5 else 1e-4)
            iv = iv.fillna(iv.median())
            w = (iv / iv.sum()).clip(upper=2.0 / len(sel))
            w = w / w.sum()
        elif weighting == "score":  # lean toward the book's highest scores, capped at score_max_mult x equal weight
            z = (sel["score"] - sel["score"].mean()) / (sel["score"].std() + 1e-12)
            w = np.exp(score_tilt * z.clip(-3, 3))
            w = (w / w.sum()).clip(upper=score_max_mult / len(sel))
            w = w / w.sum()
        else:
            w = pd.Series(1.0 / len(sel), index=sel.index)
        new = w.to_dict()
        traded = sum(abs(new.get(t, 0.0) - held.get(t, 0.0)) for t in set(new) | set(held))
        gross = float((w * sel["r"].fillna(0.0)).sum())
        rows.append({"date": d, "gross": gross, "turnover": traded / 2, "ret": gross - traded * cost / 2,
                     "n": len(sel)})
        if trace is not None:  # per-position record for attribution
            trace.append(pd.DataFrame({"date": d, "ticker": sel.index, "w": w.values, "r": sel["r"].values, "score": sel["score"].values}))
        held = new
    return pd.DataFrame(rows)


def summarize(runs: list[pd.DataFrame], rebalance: int = 5) -> dict:
    per_year = 252 / rebalance
    stats = []
    for r in runs:
        if r.empty:
            continue
        x = r["ret"]
        cum = x.cumsum()
        stats.append({"ann": x.mean() * per_year, "ir": x.mean() / (x.std() + 1e-12) * np.sqrt(per_year),
                      "maxdd": float((cum - cum.cummax()).min()), "worst": float(x.min()),
                      "turnover_ann": r["turnover"].mean() * per_year, "hit": float((x > 0).mean())})
    if not stats:
        return {}
    st = pd.DataFrame(stats)
    allr = pd.concat(runs)
    by_year = allr.assign(y=pd.to_datetime(allr["date"]).dt.year).groupby("y")["ret"].mean() * per_year
    return {"ann_excess": float(st["ann"].mean()), "ann_min": float(st["ann"].min()), "ann_max": float(st["ann"].max()),
            "ir": float(st["ir"].mean()), "maxdd": float(st["maxdd"].mean()), "worst_period": float(st["worst"].min()),
            "turnover_ann": float(st["turnover_ann"].mean()), "hit": float(st["hit"].mean()),
            "by_year": {int(k): round(float(v), 4) for k, v in by_year.items()}, "n_periods": int(len(runs[0]))}


def portfolio_report(oos: pd.DataFrame, rets: pd.DataFrame, horizons: list[int], cfg: dict, cost: float) -> dict:
    """Portfolio stats for each horizon's score and the cross-horizon ensemble."""
    pc = cfg.get("portfolio", {})
    kw = dict(k=pc.get("k", 50), buffer=pc.get("buffer", 3.0), rebalance=pc.get("rebalance", 5), cost=cost,
              weighting=pc.get("weighting", "inv_vol"), sector_cap=pc.get("sector_cap", 0.3),
              smooth=pc.get("smooth", 1))
    sources = {f"h{h}": oos[oos["horizon"] == h][["date", "ticker", "score"]] for h in horizons}
    ens_h = [h for h in pc.get("ensemble_horizons", [5, 10, 30]) if h in horizons]
    if len(ens_h) > 1:
        sources["ens(" + "+".join(map(str, ens_h)) + ")"] = ensemble_scores(oos, ens_h)
    out = {}
    for name, sc in sources.items():
        if sc is None or sc.empty:
            continue
        runs = [simulate(sc, rets, offset=o, **kw) for o in range(kw["rebalance"])]
        out[name] = summarize(runs, kw["rebalance"])
        if pc.get("tilt_strength") and {"mcap", "dvol"} <= set(rets.columns):
            truns = [simulate_tilt(sc, rets, strength=pc["tilt_strength"], rebalance=kw["rebalance"], offset=o,
                                   cost=cost, max_mult=pc.get("tilt_max_mult", 3.0)) for o in range(kw["rebalance"])]
            out[name + " tilt"] = summarize(truns, kw["rebalance"])
    out["_settings"] = {**kw, "tilt_strength": pc.get("tilt_strength")}
    return out


def print_report(rep: dict) -> None:
    rows = {k: {c: v.get(c) for c in ("ann_excess", "ann_min", "ann_max", "ir", "maxdd", "worst_period",
                                       "turnover_ann", "hit")}
            for k, v in rep.items() if not k.startswith("_") and v}
    if not rows:
        return
    print("\nportfolio (weekly, vs SPY, net of cost; mean over rebalance offsets, min..max range):")
    print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.3f}"))
    for k, v in rep.items():
        if not k.startswith("_") and v:
            print(f"  {k} by year: {v['by_year']}")

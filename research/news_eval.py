"""Does news add anything? The plan below was written before any return on this news archive was examined.

Data     : S&P 500 members, 2019-01 .. 2025-10. Development 2019-2023, validation 2024-2025-10. The locked holdout is never loaded.
Scoring  : daily rank correlation with the following 5/10/30-day return after removing sector, size and beta; and the tradable test:
           market-cap weights tilted by the signal (strength 0.5, weekly, five start days averaged, 0.10% cost) minus the untilted control.

Hypotheses, each with its expected sign:
  H1  analyst actions read from headlines (ana_net_63) point the right way: positive. Rests on decades of published evidence.
  H2  news tone (news_sent_20d) predicts returns: either sign. Earlier 2025-26 checks pointed negative; that period is not used here.
  H3  adding the news variables improves the walk-forward ML forecasts, tested under both ways of weighting history.
Every other signal in the tables is explanatory only.

Verdict rule, applied to the tilted book's active return over the whole test period:
  ADOPT      same sign in development and validation, right sign, and Newey-West |t| >= 3 (the hurdle for a new factor after many ideas tried)
  PROMISING  same sign in both windows, right sign, |t| >= 2: paper-trade only, not in the program
  REJECT     anything else
For H3 the same rule applies to the difference between the book with news and the book without it, and the correlation must also improve in both windows.
A verdict of ADOPT still has to pass the one-time locked-holdout test before it can enter the program.

Usage: python research/news_eval.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["FINTAILOR_STORE"] = "store_news"
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fintailor.config import store_dir  # noqa: E402
from fintailor.db import connect  # noqa: E402
from fintailor.models.lgbm import _neutralize  # noqa: E402
from fintailor.strategy.engine import IDEA_SPREAD, IDEAS_TRIED, _mean_run  # noqa: E402
from fintailor.strategy.signals import factor_score, sector_score, zscore  # noqa: E402
from fintailor.strategy.simulator import load_market  # noqa: E402
from fintailor.strategy.stats import block_bootstrap, deflated_sharpe, newey_west_t  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200, "display.float_format", lambda v: f"{v:.2f}")

HOLDOUT, START, DEV_END, LAST5 = pd.Timestamp("2025-11-01"), pd.Timestamp("2019-01-01"), pd.Timestamp("2023-12-31"), pd.Timestamp("2020-11-01")
HORIZONS = (5, 10, 30)
FACTOR_COLS = ["mom_12_1", "gp_assets_nf", "asset_growth_nf", "book_to_price_nf", "earn_yield_nf"]
NEWS_COLS = ["news_cnt_5d", "news_sent_5d", "news_sent_20d", "news_sent_chg", "news_attn_5d", "news_rating_sent_5d", "news_earn_sent_5d"]
STRUCTURED = ["ana_net_21", "ana_net_63", "gui_net_63"]
SIGNALS = ["ana_net_63", "news_sent_20d", "ana_net_21", "gui_net_63", "news_sent_5d", "news_sent_chg", "news_attn_5d", "news_rating_sent_5d", "news_earn_sent_5d"]
SIGN = {"ana_net_63": 1, "news_sent_20d": 0}  # 1 = must be positive, 0 = either sign
OUT = ROOT / "store/research/news"
RUNS = {"plain/recent": "plain_recent", "plain/long": "plain_long", "news/recent": "news_recent", "news/long": "news_long"}


def windows(idx: pd.DatetimeIndex) -> dict:
    return {"all": np.ones(len(idx), bool), "dev": idx <= DEV_END, "val": idx > DEV_END, "last5y": idx >= LAST5}


def load() -> pd.DataFrame:
    cols = ["date", "ticker", "is_etf_flag", "mcap", "sector", "log_dvol", "beta_60", *[f"y_{h}" for h in HORIZONS], *FACTOR_COLS, *NEWS_COLS]
    d = store_dir("features")
    p = pd.read_parquet(d / "panel.parquet", columns=cols)
    p = p[(~p["is_etf_flag"].astype(bool)) & (p["date"] < HOLDOUT)].reset_index(drop=True)
    p = p.merge(pd.read_parquet(d / "news_extra.parquet"), on=["date", "ticker"], how="left", validate="one_to_one")
    dates = np.sort(p["date"].unique())
    for h in HORIZONS:
        p.loc[p["date"] > dates[len(dates) - 1 - h], f"y_{h}"] = np.nan
    return p


def audit(p: pd.DataFrame) -> None:
    """Coverage only. If news is thinner for names that later left the index, news variables could leak survivorship."""
    last = p.groupby("ticker")["date"].transform("max")
    gone = last < p["date"].max()
    a = pd.DataFrame({"year": p["date"].dt.year, "gone": gone, "has_news_5d": p["news_cnt_5d"].fillna(0) > 0, "articles_5d": np.expm1(p["news_cnt_5d"].fillna(0)), "analyst_63": p["ana_net_63"].fillna(0) != 0})
    t = a.groupby(["year", "gone"])[["has_news_5d", "articles_5d", "analyst_63"]].mean().unstack("gone")
    t.columns = [f"{c}_{'left index' if g else 'still in'}" for c, g in t.columns]
    print("\nCOVERAGE (share of member-days)\n", t.round(2).to_string(), flush=True)


def daily_ic(p: pd.DataFrame, signal: str, y: np.ndarray) -> pd.Series:
    d = pd.DataFrame({"date": p["date"].to_numpy(), "s": p[signal].to_numpy(float), "y": y}).dropna()
    r = d.groupby("date")[["s", "y"]].rank()
    r["date"] = d["date"].to_numpy()
    r["n"] = r.groupby("date")["s"].transform("size")
    r = r[r["n"] >= 30]
    g = r.groupby("date")
    sx, sy = g["s"].transform("mean"), g["y"].transform("mean")
    x, v = r["s"] - sx, r["y"] - sy
    num = (x * v).groupby(r["date"]).sum()
    den = np.sqrt((x ** 2).groupby(r["date"]).sum() * (v ** 2).groupby(r["date"]).sum())
    return (num / den.replace(0, np.nan)).dropna()


def ic_table(p: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    rows = {}
    neutral = {h: _neutralize(p, f"y_{h}", ["sector", "size", "beta"]) for h in HORIZONS}
    for s in names:
        for h in HORIZONS:
            ic = daily_ic(p, s, neutral[h])
            row = {}
            for w, keep in windows(pd.DatetimeIndex(ic.index)).items():
                row[f"ic_{w}"] = ic[keep].mean() * 100
                row[f"t_{w}"] = newey_west_t(ic[keep], lags=max(2 * h, 10))
            rows[(s, h)] = row
    return pd.DataFrame(rows).T


def book_row(r: pd.DataFrame, ctrl: pd.Series, spy: pd.Series) -> dict:
    a = (r["ret"] - ctrl).dropna()
    idx = pd.DatetimeIndex(a.index)
    row = {"vs_spy": (r["ret"] - spy.reindex(r.index)).dropna().mean() * 25200, "active": a.mean() * 25200, "te": a.std() * np.sqrt(252) * 100, "t": newey_west_t(a)}
    for w, keep in windows(idx).items():
        if w != "all":
            row[w] = a[keep].mean() * 25200
    row["p_pos"] = block_bootstrap(a)["p_positive"] * 100
    row["defl_p"] = 100 * deflated_sharpe(a, IDEAS_TRIED + 10, IDEA_SPREAD / np.sqrt(252))
    row["turn"] = r["turnover"].mean() * 252
    return row


def verdict(row: dict, sign: int) -> str:
    dev, val, t = row["dev"], row["val"], row["t"]
    same = np.sign(dev) == np.sign(val) and np.sign(dev) == np.sign(row["active"])
    right = sign == 0 or np.sign(row["active"]) == sign
    if same and right and abs(t) >= 3:
        return "ADOPT"
    if same and right and abs(t) >= 2:
        return "PROMISING"
    return "REJECT"


def load_ml(p: pd.DataFrame, file: str) -> pd.Series:
    o = pd.read_parquet(OUT / f"{file}.parquet")
    return p[["date", "ticker"]].merge(o, on=["date", "ticker"], how="left")["ml"].set_axis(p.index)


def main() -> None:
    p = load()
    audit(p)
    test = p[p["date"] >= START].reset_index(drop=True)
    print("\nRANK CORRELATION x100 WITH LATER RETURN, sector/size/beta removed\n", ic_table(test, SIGNALS).round(2).to_string(), flush=True)

    with connect(read_only=True) as con:
        m = load_market(con, "2018-09-01", "2025-10-31")
        sec = sector_score(test, con)
    fac = factor_score(test)
    s = test[["date", "ticker", "mcap"]].copy()
    d = test["date"]
    s["z_fs"] = zscore(fac + sec, d)
    columns = {"factors+sector": "z_fs"}
    for name in SIGNALS:
        s[f"z_{name}"] = zscore(test[name], d)
        columns[name] = f"z_{name}"
    s["z_fs_ana"] = zscore(zscore(fac, d) + zscore(sec, d) + s["z_ana_net_63"], d)
    columns["factors+sector+analyst"] = "z_fs_ana"
    control = _mean_run(m, s, None)
    x = (control["ret"] - m.spy.reindex(control.index)).dropna()
    print(f"\ncontrol vs SPY: {x.mean() * 25200:+.2f}%/yr, tracking {x.std() * np.sqrt(252) * 100:.2f}%", flush=True)
    rows, runs = {}, {}
    for name, col in columns.items():
        runs[name] = _mean_run(m, s, col)
        rows[name] = book_row(runs[name], control["ret"], m.spy)
        print(f"{name}: {rows[name]}", flush=True)

    ml_rows, ml_ic = {}, {}
    y5, y10 = _neutralize(test, "y_5", ["sector", "size", "beta"]), _neutralize(test, "y_10", ["sector", "size", "beta"])
    for label, file in RUNS.items():
        if not (OUT / f"{file}.parquet").exists():
            print("missing", file, flush=True)
            continue
        ml = load_ml(test, file)
        keep = ml.notna()
        col = "z_ml_" + file
        s[col] = zscore(ml, d)
        run = _mean_run(m, s[keep], col)
        runs[label] = run
        ml_rows[label] = book_row(run, control["ret"].reindex(run.index), m.spy)
        test[col] = ml
        for h, y in ((5, y5), (10, y10)):
            ic = daily_ic(test, col, y)
            for w, k in windows(pd.DatetimeIndex(ic.index)).items():
                ml_ic.setdefault(label, {})[f"ic{h}_{w}"] = ic[k].mean() * 100
    print("\nML BOOKS\n", pd.DataFrame(ml_rows).T.round(2).to_string(), "\n\nML RANK CORRELATION x100\n", pd.DataFrame(ml_ic).T.round(2).to_string(), flush=True)

    print("\nSIGNAL BOOKS\n", pd.DataFrame(rows).T.round(2).to_string(), flush=True)
    print("\nVERDICTS")
    ledger = []
    for name, hyp in (("ana_net_63", "H1"), ("news_sent_20d", "H2")):
        v = verdict(rows[name], SIGN[name])
        print(f"  {hyp} {name}: {v}  active {rows[name]['active']:+.2f}%/yr, dev {rows[name]['dev']:+.2f}, val {rows[name]['val']:+.2f}, t {rows[name]['t']:+.2f}, luck-adjusted probability {rows[name]['defl_p']:.1f}%")
        ledger.append({"name": f"news:{name}", "hypothesis": hyp, "verdict": v, **rows[name]})
    for weighting in ("recent", "long"):
        a, b = f"news/{weighting}", f"plain/{weighting}"
        if a not in runs or b not in runs:
            continue
        diff = (runs[a]["ret"] - runs[b]["ret"]).dropna()
        idx = pd.DatetimeIndex(diff.index)
        w = windows(idx)
        row = {"active": diff.mean() * 25200, "dev": diff[w["dev"]].mean() * 25200, "val": diff[w["val"]].mean() * 25200, "last5y": diff[w["last5y"]].mean() * 25200, "t": newey_west_t(diff)}
        ic_up = all(ml_ic[a][f"ic{h}_{win}"] > ml_ic[b][f"ic{h}_{win}"] for h in (5, 10) for win in ("dev", "val"))
        v = verdict(row, 1)
        v = v if ic_up or v == "REJECT" else "REJECT"
        print(f"  H3 news minus no news ({weighting}): {v}  {row['active']:+.2f}%/yr, dev {row['dev']:+.2f}, val {row['val']:+.2f}, last 5y {row['last5y']:+.2f}, t {row['t']:+.2f}, correlation better in both windows at 5 and 10 days: {ic_up}")
        ledger.append({"name": f"news:ml_diff_{weighting}", "hypothesis": "H3", "verdict": v, "ic_up": ic_up, **row})
    for weighting, a, b in (("no news", "plain/long", "plain/recent"), ("with news", "news/long", "news/recent")):
        if a in runs and b in runs:
            diff = (runs[a]["ret"] - runs[b]["ret"]).dropna()
            w = windows(pd.DatetimeIndex(diff.index))
            print(f"  long history minus recent weighting, {weighting}: {diff.mean() * 25200:+.2f}%/yr, dev {diff[w['dev']].mean() * 25200:+.2f}, val {diff[w['val']].mean() * 25200:+.2f}, last 5y {diff[w['last5y']].mean() * 25200:+.2f}, t {newey_west_t(diff):+.2f}")
    if os.environ.get("NEWS_EVAL_DRY"):  # code check on partial data: output is discarded and nothing is recorded
        return
    with open(ROOT / "store/research/trials.jsonl", "a", encoding="utf-8") as f:
        for r in ledger:
            f.write(json.dumps({"at": pd.Timestamp.now().isoformat(timespec="seconds"), **r}, default=float) + "\n")


if __name__ == "__main__":
    main()

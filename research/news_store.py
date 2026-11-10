"""Research store with the 2015-2025 Alpaca (Benzinga) news archive, kept apart from production (store_news).

Copies the production database, replaces only its two news tables, and rebuilds the panel with the production feature code, so
every non-news column stays comparable. Rules below were fixed before any return was looked at:

  - article time = created_at; anything edited more than a day after publication is dropped (later edits can carry later facts)
  - articles from the automated "Benzinga Insights" desk are dropped
  - only articles tagged with at most 3 symbols count (longer lists are market round-ups, not news about one company)
  - nothing dated 2025-11-01 or later is loaded (locked holdout)

Headline-parsed features (news_extra.parquet), one row per panel row:
  ana_net_21 / ana_net_63  (upgrades + price-target raises - downgrades - price-target cuts) / (their sum + 3), over 21 / 63 trading days
  gui_net_63               (guidance raises - cuts) / (their sum + 1), over 63 trading days

Usage: python research/news_store.py
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["FINTAILOR_STORE"] = "store_news"
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fintailor.config import get_config, store_dir  # noqa: E402
from fintailor.db import connect  # noqa: E402
from fintailor.features.panel import build_panel  # noqa: E402
from fintailor.llm.news_scorer import lexicon_score  # noqa: E402
from fintailor.utils import assign_trading_date  # noqa: E402

RAW = ROOT / "store/raw/alpaca_news"
ALIASES = ROOT / "store/research/news_aliases.parquet"
MAIN = ROOT / "store/fintailor.duckdb"
TARGET = ROOT / "store_news"
HOLDOUT = pd.Timestamp("2025-11-01")
FIRST = pd.Timestamp("2015-01-01")  # the archive holds only a few stray items before this
MAX_TAGS = 3
AUTOMATED = {"Benzinga Insights"}

UP = r"(?:raises?|raised|raising|boosts?|boosted|lifts?|lifted|increases?|increased|hikes?|hiked|bumps?|bumped)"
DOWN = r"(?:lowers?|lowered|lowering|cuts?|cutting|reduces?|reduced|trims?|trimmed|slashes|slashed|decreases?|decreased)"
TARGET_WORDS = r"(?:price target|price objective|pt)"
GUIDE_WORDS = r"(?:guidance|outlook|forecast)"
GAP = r"\b[^;]{0,40}?\b"  # periods allowed: "Raises FY15 Adj. EPS Guidance", "Maintains Buy on U.S. Bancorp"


def load_articles() -> pd.DataFrame:
    files = sorted(RAW.glob("*.parquet"))
    a = pd.concat([pd.read_parquet(f, columns=["id", "created_at", "updated_at", "headline", "summary", "author", "symbols"]) for f in files], ignore_index=True)
    a = a.drop_duplicates("id")
    print(f"{len(files)} months, {len(a):,} articles", flush=True)
    a = a[(a["created_at"] >= FIRST) & (a["created_at"] < HOLDOUT)]
    late = (a["updated_at"] - a["created_at"]) > pd.Timedelta(days=1)
    auto = a["author"].isin(AUTOMATED)
    print(f"dropped: edited more than a day later {late.mean():.2%}, automated desk {auto.mean():.2%}", flush=True)
    return a[~late & ~auto].reset_index(drop=True)


def link_tickers(a: pd.DataFrame) -> pd.DataFrame:
    tags = a["symbols"].fillna("").map(lambda s: [x for x in s.split(",") if x])
    n = tags.map(len)
    print(f"articles with 1-{MAX_TAGS} tags: {n.between(1, MAX_TAGS).mean():.1%}", flush=True)
    ex = a.assign(symbol=tags)[n.between(1, MAX_TAGS)].explode("symbol")
    ex["quarter"] = ex["created_at"].dt.to_period("Q").astype(str)
    m = ex.merge(pd.read_parquet(ALIASES), on=["symbol", "quarter"], how="inner").drop_duplicates(["id", "ticker"])
    print(f"article-ticker rows {len(m):,}, panel tickers reached {m['ticker'].nunique()}", flush=True)
    return m.drop(columns=["symbol", "quarter", "symbols", "updated_at", "author"])


def score(m: pd.DataFrame) -> pd.DataFrame:
    u = m.drop_duplicates("id")[["id", "headline", "summary"]]
    res = [lexicon_score(h, s if isinstance(s, str) else "", "X") for h, s in zip(u["headline"], u["summary"])]
    s = pd.DataFrame({"id": u["id"].to_numpy(), "sentiment": [r.sentiment for r in res], "severity": [float(r.severity) for r in res],
                      "novelty": [r.novelty for r in res], "event_type": [r.event_type for r in res]})
    print("lexicon: share neutral", round(float((s["sentiment"] == 0).mean()), 3), "| event types:", s["event_type"].value_counts(normalize=True).head(6).round(3).to_dict(), flush=True)
    return m.merge(s, on="id", validate="many_to_one")


def fill_database(m: pd.DataFrame) -> None:
    TARGET.mkdir(exist_ok=True)
    shutil.copy2(MAIN, TARGET / "fintailor.duckdb")
    key = m["id"].astype(str) + ":" + m["ticker"]
    news = pd.DataFrame({"id": key, "ticker": m["ticker"], "category": "company", "headline": m["headline"], "summary": "", "source": "benzinga", "url": "",
                         "published_at": m["created_at"], "received_at": m["created_at"], "backfilled": True})
    scores = pd.DataFrame({"news_id": key, "scorer": "lexicon", "payload": '{"confidence":0.3}', "sentiment": m["sentiment"], "severity": m["severity"],
                           "novelty": m["novelty"], "event_type": m["event_type"], "created_at": pd.Timestamp("2026-11-09")})
    with connect() as con:
        con.execute("delete from news_scores")
        con.execute("delete from news")
        con.register("n_", news)
        con.register("s_", scores)
        con.execute("insert into news select * from n_")
        con.execute("insert into news_scores select * from s_")
        print("news rows in research store:", con.execute("select count(*), min(received_at), max(received_at) from news").fetchone(), flush=True)


def net_ratio(up: pd.DataFrame, down: pd.DataFrame, days: int, prior: float) -> pd.DataFrame:
    u, d = up.rolling(days, min_periods=1).sum(), down.rolling(days, min_periods=1).sum()
    return (u - d) / (u + d + prior)


def headline_features(m: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(np.sort(panel["date"].unique()))
    tickers = sorted(panel["ticker"].unique())
    t = m["headline"].fillna("").str.lower()
    # "upgrades X to Buy" or "upgraded"; a bare "upgrade" is usually a product ("to upgrade patient monitoring")
    upgrade, downgrade = t.str.contains(r"\bupgrades\b[^;]{0,80}?\bto\b|\bupgraded\b", regex=True), t.str.contains(r"\bdowngrades\b[^;]{0,80}?\bto\b|\bdowngraded\b", regex=True)
    pt_up, pt_down = t.str.contains(rf"\b{UP}{GAP}{TARGET_WORDS}\b", regex=True), t.str.contains(rf"\b{DOWN}{GAP}{TARGET_WORDS}\b", regex=True)
    a = np.where(upgrade & ~downgrade, 1, np.where(downgrade & ~upgrade, -1, np.where(pt_up & ~pt_down, 1, np.where(pt_down & ~pt_up, -1, 0))))
    g_up = t.str.contains(rf"\b{UP}{GAP}{GUIDE_WORDS}\b", regex=True)
    # no bare "lower" here: "lower demand weighs on results, guidance..." is not a guidance cut
    g_down = t.str.contains(rf"\b(?:lowers|lowered|lowering|cut|cuts|cutting|reduces|reduced|trims|trimmed|slashes|slashed|decreases|decreased|withdraws?|withdrawn|suspends?|suspended){GAP}{GUIDE_WORDS}\b", regex=True)
    g = np.where(g_up & ~g_down, 1, np.where(g_down & ~g_up, -1, 0))
    tdate = assign_trading_date(m["created_at"], dates, get_config()["news"]["cutoff_et"])
    e = pd.DataFrame({"tdate": tdate.to_numpy(), "ticker": m["ticker"].to_numpy(), "a_up": (a > 0).astype(float), "a_down": (a < 0).astype(float), "g_up": (g > 0).astype(float), "g_down": (g < 0).astype(float)}).dropna(subset=["tdate"])
    e = e[e["ticker"].isin(tickers)]
    print("share of article-ticker rows that are: analyst up", round(float(e["a_up"].mean()), 3), "analyst down", round(float(e["a_down"].mean()), 3), "guidance up", round(float(e["g_up"].mean()), 4), "guidance down", round(float(e["g_down"].mean()), 4), flush=True)
    by_year = e.groupby(pd.to_datetime(e["tdate"]).dt.year)[["a_up", "a_down", "g_up", "g_down"]].sum().astype(int)
    print("headline events per year:\n", by_year.to_string(), flush=True)
    wide = {c: e.pivot_table(index="tdate", columns="ticker", values=c, aggfunc="sum").reindex(index=dates, columns=tickers).fillna(0.0) for c in ("a_up", "a_down", "g_up", "g_down")}
    frames = {"ana_net_21": net_ratio(wide["a_up"], wide["a_down"], 21, 3.0), "ana_net_63": net_ratio(wide["a_up"], wide["a_down"], 63, 3.0), "gui_net_63": net_ratio(wide["g_up"], wide["g_down"], 63, 1.0)}
    long = pd.DataFrame({k: v.to_numpy().ravel() for k, v in frames.items()})
    long["date"] = np.repeat(dates.to_numpy(), len(tickers))
    long["ticker"] = np.tile(np.array(tickers, dtype=object), len(dates))
    return panel[["date", "ticker"]].merge(long, on=["date", "ticker"], how="left", validate="one_to_one")


def compare_with_production(p: pd.DataFrame) -> None:
    cols = ["date", "ticker", "mom_12_1", "gp_assets_nf", "mcap", "y_5", "y_10", "log_dvol"]
    a = pd.read_parquet(ROOT / "store/features/panel.parquet", columns=cols)
    b = p[cols]
    j = a.merge(b, on=["date", "ticker"], suffixes=("_a", "_b"))
    print(f"rows: production {len(a):,}, research {len(b):,}, shared {len(j):,}", flush=True)
    for c in cols[2:]:
        x, y = j[f"{c}_a"].to_numpy(float), j[f"{c}_b"].to_numpy(float)
        both = np.isfinite(x) & np.isfinite(y)
        print(f"  {c}: differing values {int((np.abs(x[both] - y[both]) > 1e-9).sum())}, one-sided blanks {int((np.isfinite(x) != np.isfinite(y)).sum())}", flush=True)


def main() -> None:
    m = score(link_tickers(load_articles()))
    fill_database(m)
    panel = build_panel()
    compare_with_production(panel)
    extra = headline_features(m, panel)
    extra.to_parquet(store_dir("features") / "news_extra.parquet", index=False)
    print("saved headline features", len(extra), flush=True)


if __name__ == "__main__":
    main()

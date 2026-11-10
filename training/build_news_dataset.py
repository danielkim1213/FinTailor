"""FinTailor-News dataset (plan 4.6).

1) headlines from the store (Finnhub / RSS / SEC 8-K titles)    2) 1d/5d excess return after RECEIPT
3) teacher LLM writes the structure WITHOUT seeing outcomes       4) dedupe + drop "already moved" articles
5) split: received >= 2026-04-01 -> test only (after Qwen3.5 release; no memorization)

Output: chat JSONL (system/user/assistant) with the same prompts used at inference.

    python training/build_news_dataset.py --out training/data --max 50000            # teacher = TEACHER_* in .env
    python training/build_news_dataset.py --out training/data --teacher lexicon      # offline dry run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fintailor.config import get_config  # noqa: E402
from fintailor.db import connect, trading_dates  # noqa: E402
from fintailor.llm.client import LLMClient  # noqa: E402
from fintailor.llm.news_scorer import lexicon_score  # noqa: E402
from fintailor.llm.prompts import NEWS_SYSTEM, news_user  # noqa: E402
from fintailor.llm.schemas import NewsBatch  # noqa: E402
from fintailor.utils import assign_trading_date, log, setup_logging  # noqa: E402

TEST_START = pd.Timestamp("2026-04-01")


def load_news(con, max_train: int, max_test: int, seed: int = 0) -> pd.DataFrame:
    """Random sample from each side of TEST_START (oldest-first would leave the test period empty).
    Oversampled 2x because dedupe / stale / no-outcome filters drop a share."""
    q = """select id, ticker, headline, summary, source, received_at, backfilled from news
           where headline is not null and (received_at >= ?) = ? order by hash(id || ?) limit ?"""
    tr = con.execute(q, [TEST_START, False, str(seed), int(max_train * 2)]).df()
    te = con.execute(q, [TEST_START, True, str(seed), int(max_test * 2)]).df()
    return pd.concat([tr, te], ignore_index=True)


def outcome_labels(con, news: pd.DataFrame) -> pd.DataFrame:
    """Excess return vs SPY from the first open AFTER receipt to the close 1/5 days later, plus the
    same-day move before the article (for stale filtering)."""
    td = trading_dates(con)
    bench = get_config()["universe"]["benchmark"]
    news = news.copy()
    news["tdate"] = assign_trading_date(news["received_at"], td, "16:00")
    tick = [t for t in news["ticker"].dropna().unique()] + [bench]
    con.register("_t", pd.DataFrame({"t": tick}))
    px = con.execute("select ticker, date, open, close, adj_close from prices_daily where ticker in (select t from _t)").df()
    con.unregister("_t")
    px["date"] = pd.to_datetime(px["date"])
    px["aopen"] = px["open"] * px["adj_close"] / px["close"]
    O = px.pivot(index="date", columns="ticker", values="aopen").reindex(td)
    C = px.pivot(index="date", columns="ticker", values="adj_close").reindex(td)
    rv = C.pct_change(fill_method=None).rolling(20).std()
    pos = pd.Series(np.arange(len(td)), index=td)
    out = {k: np.full(len(news), np.nan) for k in ("ex_1d", "ex_5d", "pre_move", "rv")}
    for i, (t, d) in enumerate(zip(news["ticker"], news["tdate"])):
        if pd.isna(d):
            continue
        p = pos.get(d)
        if p is None or p + 5 >= len(td):
            continue
        cols = [t] if isinstance(t, str) and t in C else []
        for h, key in ((1, "ex_1d"), (5, "ex_5d")):
            spy = C[bench].iloc[p + h] / O[bench].iloc[p + 1] - 1
            out[key][i] = (C[t].iloc[p + h] / O[t].iloc[p + 1] - 1 - spy) if cols else spy
        if cols:
            out["pre_move"][i] = C[t].iloc[p] / C[t].iloc[p - 1] - 1 if p > 0 else np.nan
            out["rv"][i] = rv[t].iloc[p]
    for k, v in out.items():
        news[k] = v
    return news


def dedupe(df: pd.DataFrame, sim: float = 0.9) -> pd.DataFrame:
    from sklearn.feature_extraction.text import TfidfVectorizer
    df = df.copy()
    df["norm"] = df["headline"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    df = df.drop_duplicates(["ticker", "norm"])
    keep = np.ones(len(df), bool)
    df = df.sort_values("received_at").reset_index(drop=True)
    for _, g in df.groupby(df["ticker"].fillna("-")):
        if len(g) < 2:
            continue
        X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(g["norm"])
        S = (X @ X.T).tocoo()
        times = g["received_at"].values
        idx = g.index.values
        for i, j, v in zip(S.row, S.col, S.data):
            if i < j and v >= sim and keep[idx[i]] and abs(times[j] - times[i]) <= np.timedelta64(3, "D"):
                keep[idx[j]] = False
    return df[keep].drop(columns="norm")


def teacher_label(df: pd.DataFrame, teacher: str, batch: int = 8, workers: int = 8) -> tuple[list[str], list[bool]]:
    """Returns (label json per row, whether the label came from the teacher). Failed batches fall back to
    the lexicon and are flagged so they can be dropped instead of training on weak labels."""
    from concurrent.futures import ThreadPoolExecutor

    client = LLMClient(prefix="TEACHER") if teacher == "llm" else None
    if client is not None and not client.available:
        raise SystemExit("TEACHER_BASE_URL / TEACHER_MODEL not set (or use --teacher lexicon for a dry run)")
    recs = df.to_dict("records")
    chunks = [recs[i:i + batch] for i in range(0, len(recs), batch)]

    def one(chunk):
        res = None
        if client:
            try:
                res = client.chat_json([{"role": "system", "content": NEWS_SYSTEM},
                                        {"role": "user", "content": news_user(chunk)}], NewsBatch,
                                       max_tokens=250 * len(chunk))
            except Exception as e:  # noqa: BLE001 - one bad batch must not kill a multi-hour run
                log.warning("teacher batch failed: %s", e)
        if res and len(res.items) == len(chunk):
            return [it.model_dump_json() for it in res.items], True
        return [lexicon_score(c["headline"], c.get("summary") or "", c.get("ticker")).model_dump_json()
                for c in chunk], False

    out: list[str] = []
    ok: list[bool] = []
    with ThreadPoolExecutor(max_workers=workers if client else 1) as ex:
        for k, (labels, good) in enumerate(ex.map(one, chunks)):  # map keeps input order
            out += labels
            ok += [good] * len(labels)
            if k % 50 == 0:
                log.info("teacher %d/%d batches, %d fallback rows so far", k, len(chunks), ok.count(False))
    return out, ok


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="training/data")
    ap.add_argument("--max", type=int, default=50000, help="train examples (received before TEST_START)")
    ap.add_argument("--test-max", type=int, default=2000, help="test examples (received on/after TEST_START)")
    ap.add_argument("--workers", type=int, default=8, help="parallel teacher requests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher", choices=["llm", "lexicon"], default="llm")
    ap.add_argument("--stale-z", type=float, default=2.5, help="drop if same-day move before receipt > z * daily vol")
    ap.add_argument("--direction-threshold", type=float, default=0.005)
    a = ap.parse_args()
    with connect(read_only=True) as con:
        news = load_news(con, a.max, a.test_max, a.seed)
        news = outcome_labels(con, news)
    n0 = len(news)
    news = news[news["ex_5d"].notna()]  # no 5d outcome yet (too recent / no prices): label would be a fake 0
    n1 = len(news)
    news = dedupe(news)
    stale = (news["pre_move"].abs() > a.stale_z * news["rv"]) & news["ticker"].notna()
    news = news[~stale]
    is_test = news["received_at"] >= TEST_START
    news = pd.concat([news[~is_test].head(a.max), news[is_test].head(a.test_max)]).reset_index(drop=True)
    log.info("news: %d sampled -> %d with 5d outcome -> %d after dedupe/stale filter (%d train / %d test)",
             n0, n1, len(news), (news["received_at"] < TEST_START).sum(), (news["received_at"] >= TEST_START).sum())
    news["target"], news["teacher_ok"] = teacher_label(news, a.teacher, workers=a.workers)
    if a.teacher == "llm":
        log.info("dropping %d rows whose teacher batch failed (lexicon fallback)", (~news["teacher_ok"]).sum())
        news = news[news["teacher_ok"]]
    outd = Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    n_tr = n_te = 0
    with open(outd / "news_train.jsonl", "w", encoding="utf-8") as ftr, open(outd / "news_test.jsonl", "w", encoding="utf-8") as fte:
        for r in news.to_dict("records"):
            tgt = json.loads(r["target"])
            # outcome-derived direction (plan step 2): sign of 5d excess when it clears the threshold
            ex = r.get("ex_5d")
            tgt["direction_5d"] = 0 if pd.isna(ex) or abs(ex) < a.direction_threshold else int(np.sign(ex))
            rec = {"messages": [{"role": "system", "content": NEWS_SYSTEM},
                                {"role": "user", "content": news_user([r])},
                                {"role": "assistant", "content": json.dumps({"items": [tgt]}, ensure_ascii=False)}],
                   "meta": {"id": r["id"], "received_at": str(r["received_at"]), "ex_1d": r.get("ex_1d"),
                            "ex_5d": ex, "backfilled": bool(r.get("backfilled"))}}
            line = json.dumps(rec, ensure_ascii=False, default=lambda x: None if pd.isna(x) else str(x)) + "\n"
            if pd.Timestamp(r["received_at"]) >= TEST_START:
                fte.write(line)
                n_te += 1
            else:
                ftr.write(line)
                n_tr += 1
    log.info("wrote %d train / %d test (test = received >= %s)", n_tr, n_te, TEST_START.date())


if __name__ == "__main__":
    main()


__all__ = ["re"]

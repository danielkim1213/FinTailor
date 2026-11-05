"""Nightly post-mortem of matured predictions.

LLM "why" is a hypothesis: every claim is checked against data (SPY move, sector ETF move,
earnings inside the window). Misreads go to the fine-tuning queue; everything goes to RAG.
Without an LLM a rule-based reviewer produces the same record shape.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import get_config, sector_etfs
from ..db import connect, upsert
from ..rag.store import RagStore
from ..utils import log, stable_id, utcnow
from .client import LLMClient
from .prompts import REVIEW_SYSTEM
from .schemas import Claim, Review


def _window_returns(con, tickers: list[str], d0, d1) -> dict[str, float]:
    if not tickers:
        return {}
    con.register("_t", pd.DataFrame({"t": tickers}))
    df = con.execute("""select ticker, date, adj_close from prices_daily
                        where ticker in (select t from _t) and date >= ? and date <= ? order by date""", [d0, d1]).df()
    con.unregister("_t")
    out = {}
    for t, g in df.groupby("ticker"):
        if len(g) >= 2:
            out[t] = float(g["adj_close"].iloc[-1] / g["adj_close"].iloc[0] - 1)
    return out


def verify(claims: list[Claim], facts: dict) -> tuple[bool, list[dict]]:
    checks = []
    for c in claims:
        ok = None
        if c.type == "market_move" and c.direction != "none":
            ok = (facts["spy_ret"] < 0) == (c.direction == "down")
        elif c.type == "sector_move" and c.direction != "none" and facts.get("sector_ret") is not None:
            ok = (facts["sector_ret"] - facts["spy_ret"] < 0) == (c.direction == "down")
        elif c.type == "earnings":
            ok = facts["earnings_in_window"]
        checks.append({"claim": c.model_dump(), "verified": ok})
    decided = [x["verified"] for x in checks if x["verified"] is not None]
    return (all(decided) if decided else False), checks


def rule_review(facts: dict) -> Review:
    hit = (facts["realized_net"] > 0) == (facts["pred_excess"] > 0)
    claims = [Claim(type="market_move", direction="up" if facts["spy_ret"] > 0 else "down")]
    if facts["earnings_in_window"]:
        claims.append(Claim(type="earnings", detail="earnings inside holding window"))
    if hit:
        cat = "correct"
    elif abs(facts["realized_gross"]) < 1.5 * facts.get("vol_h", 0.03):
        cat = "noise"
    else:
        cat = "market_opposite"
    return Review(category=cat, summary=f"rule-based: realized {facts['realized_net']:+.3f} vs predicted "
                                        f"{facts['pred_excess']:+.3f}; SPY {facts['spy_ret']:+.3f}", claims=claims)


def review_matured(run_dates: list | None = None, max_n: int | None = None) -> dict:
    cfg = get_config()
    max_n = max_n or cfg["review"]["max_reviews_per_night"]
    client = LLMClient()
    rag = RagStore()
    etf_by_sector = {v: k for k, v in sector_etfs().items()}
    with connect(read_only=True) as con:
        cand = con.execute(f"""
            select p.run_date, p.ticker, p.horizon, p.pred_excess, p.prob, p.drivers, o.realized_net, o.realized_gross,
                   o.spy_ret, o.matured_date
            from predictions p join outcomes o using (run_date, ticker, horizon)
            where p.official and not exists (select 1 from reviews r where r.run_date = p.run_date and
                  r.ticker = p.ticker and r.horizon = p.horizon)
            order by o.matured_date desc, p.score desc limit {int(max_n)}""").df()
        if cand.empty:
            return {"reviewed": 0}
        sectors = con.execute("select ticker, sector from universe").df().drop_duplicates("ticker").set_index("ticker")["sector"]
        rows, queue = [], []
        for r in cand.itertuples():
            d0, d1 = pd.Timestamp(r.run_date), pd.Timestamp(r.matured_date)
            etf = etf_by_sector.get(sectors.get(r.ticker, ""))
            wr = _window_returns(con, [etf] if etf else [], d0, d1)
            earn = con.execute("select count(*) from earnings where ticker = ? and date > ? and date <= ?",
                               [r.ticker, d0.date(), d1.date()]).fetchone()[0] > 0
            news = con.execute("""select n.id, n.headline, s.sentiment, s.event_type from news n
                                  left join news_scores s on s.news_id = n.id
                                  where n.ticker = ? and n.received_at between ? and ? limit 10""",
                               [r.ticker, d0 - pd.Timedelta(days=5), d1]).df()
            facts = {"ticker": r.ticker, "horizon": int(r.horizon), "run_date": str(d0.date()),
                     "pred_excess": float(r.pred_excess), "prob": float(r.prob), "realized_net": float(r.realized_net),
                     "realized_gross": float(r.realized_gross), "spy_ret": float(r.spy_ret),
                     "sector_ret": wr.get(etf), "earnings_in_window": bool(earn),
                     "drivers": json.loads(r.drivers) if r.drivers else [], "vol_h": 0.02 * np.sqrt(r.horizon)}
            rev = None
            if client.available:
                ctx = rag.search(f"{r.ticker} {' '.join(news['headline'].head(3))}", k=2)
                rev = client.chat_json([{"role": "system", "content": REVIEW_SYSTEM},
                                        {"role": "user", "content": json.dumps({**facts, "news": news.to_dict("records"),
                                                                                "similar_cases": [h["text"][:300] for h in ctx]},
                                                                               default=str)}], Review)
            rev = rev or rule_review(facts)
            verified, checks = verify(rev.claims, facts)
            rid = stable_id(r.run_date, r.ticker, r.horizon)
            payload = {"facts": facts, "review": rev.model_dump(), "checks": checks,
                       "source": "llm" if client.available else "rule"}
            rows.append({"review_id": rid, "run_date": d0.date(), "ticker": r.ticker, "horizon": int(r.horizon),
                         "category": rev.category, "verified": verified, "payload": json.dumps(payload, default=str),
                         "created_at": utcnow()})
            if rev.category == "misread":
                for nid in rev.misread_news_ids:
                    hl = news.loc[news["id"] == nid, "headline"]
                    queue.append({"news_id": nid, "headline": hl.iloc[0] if len(hl) else "", "reason": rev.summary,
                                  "payload": json.dumps(facts, default=str), "created_at": utcnow()})
    with connect() as con:
        upsert(con, "reviews", pd.DataFrame(rows))
        if queue:
            upsert(con, "finetune_queue", pd.DataFrame(queue), mode="append")
    rag.add_many([{"kind": "review", "text": f"{x['ticker']} {x['run_date']} h={x['horizon']} [{x['category']}] "
                                             f"{json.loads(x['payload'])['review']['summary']}",
                   "meta": {"review_id": x["review_id"], "verified": x["verified"]}} for x in rows])
    cats = pd.Series([x["category"] for x in rows]).value_counts().to_dict()
    log.info("reviews: %d (%s), misread queued: %d", len(rows), cats, len(queue))
    return {"reviewed": len(rows), "categories": cats, "queued": len(queue)}


def feature_ideas(limit: int = 200) -> pd.Series:
    """Recurring feature ideas proposed by reviews (candidates to implement and gate-test)."""
    with connect(read_only=True) as con:
        df = con.execute(f"select payload from reviews order by created_at desc limit {limit}").df()
    ideas = [json.loads(p)["review"].get("feature_idea", "").strip().lower() for p in df["payload"]]
    return pd.Series([i for i in ideas if i]).value_counts()

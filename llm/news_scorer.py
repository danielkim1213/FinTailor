"""Turn news into structured scores. Uses the LLM when configured; otherwise a fast lexicon
scorer so the full pipeline runs before the fine-tuned model exists."""
from __future__ import annotations

import json
import re

import pandas as pd

from ..config import get_config
from ..db import connect, upsert
from ..utils import log, utcnow
from .client import LLMClient
from .prompts import NEWS_SYSTEM, SECTOR_SYSTEM, news_user
from .schemas import SECTORS, NewsBatch, NewsScore, SectorView

POS = set("""beat beats surge surges soar soars jump jumps rally rallies record upgrade upgraded raises raised
raise outperform strong growth profit profits gain gains approval approved wins win boost boosts expands expansion
buyback dividend higher tops exceeds exceed optimistic bullish rebound recovers recovery accelerate partnership
breakthrough""".split())
NEG = set("""miss misses plunge plunges slump slumps fall falls drop drops tumble tumbles downgrade downgraded cut
cuts lower lowers weak loss losses lawsuit probe investigation recall recalls layoffs layoff bankruptcy default
warns warning fraud halt halts decline declines bearish delay delays fine fined sanction sanctions strike strikes
crash slowdown shortfall resign resigns""".split())
EVENTS = [
    ("earnings", r"\b(earnings|eps|quarterly results|q[1-4] results|revenue)\b"),
    ("guidance", r"\b(guidance|outlook|forecast)\b"),
    ("m_and_a", r"\b(acquire|acquisition|merger|takeover|buyout|to buy)\b"),
    ("analyst_rating", r"\b(upgrade|downgrade|price target|initiates coverage)\b"),
    ("legal_regulatory", r"\b(lawsuit|sec|ftc|doj|antitrust|probe|regulator|fine)\b"),
    ("central_bank", r"\b(fed|fomc|rate cut|rate hike|powell|ecb|central bank)\b"),
    ("macro_data", r"\b(cpi|inflation|jobs report|payrolls|gdp|unemployment|pmi)\b"),
    ("trade_policy", r"\b(tariff|tariffs|trade war|export controls?)\b"),
    ("military_conflict", r"\b(war|missile|airstrike|military|invasion|troops|drone strike)\b"),
    ("sanctions", r"\b(sanction|sanctions)\b"),
    ("commodity_supply", r"\b(opec|oil supply|pipeline|refinery|crude output)\b"),
    ("management", r"\b(ceo|cfo|resigns|appointed|steps down)\b"),
    ("capital_return", r"\b(buyback|dividend|repurchase)\b"),
]
SECTOR_RULES = [
    (r"\b(oil|crude|opec|gasoline)\b", {"energy": "+"}),
    (r"\b(war|missile|military|defense)\b", {"industrials": "+", "energy": "+"}),
    (r"\b(rate cut)\b", {"real_estate": "+", "utilities": "+", "technology": "+"}),
    (r"\b(rate hike|yields? surge)\b", {"real_estate": "-", "utilities": "-", "financials": "+"}),
    (r"\b(tariff|tariffs|trade war)\b", {"industrials": "-", "consumer_discretionary": "-", "materials": "-"}),
    (r"\b(chip|semiconductor|ai)\b", {"technology": "+"}),
    (r"\b(drug|fda|biotech)\b", {"health_care": "+"}),
]


def lexicon_score(headline: str, summary: str = "", ticker: str | None = None) -> NewsScore:
    text = f"{headline} {summary[:300]}".lower()
    words = re.findall(r"[a-z]+", text)
    p, n = sum(w in POS for w in words), sum(w in NEG for w in words)
    sent = 0.0 if p + n == 0 else (p - n) / (p + n)
    et = next((name for name, rx in EVENTS if re.search(rx, text)), "other")
    impact: dict[str, str] = {}
    if not ticker:
        for rx, imp in SECTOR_RULES:
            if re.search(rx, text):
                impact.update(imp)
    sev = min(5, 1 + (p + n) // 2 + (et in ("military_conflict", "central_bank", "m_and_a", "sanctions")))
    return NewsScore(tickers=[ticker] if ticker else [], event_type=et, severity=sev, sentiment=round(sent, 3),
                     sector_impact=impact, confidence=0.3)


def _store(rows: list[tuple[str, NewsScore]], scorer: str) -> int:
    if not rows:
        return 0
    df = pd.DataFrame([{"news_id": nid, "scorer": scorer, "payload": s.model_dump_json(), "sentiment": s.sentiment,
                        "severity": float(s.severity), "novelty": s.novelty, "event_type": s.event_type,
                        "created_at": utcnow()} for nid, s in rows])
    with connect() as con:
        return upsert(con, "news_scores", df)


def score_pending(limit: int = 5000, force_lexicon: bool = False) -> dict:
    client = LLMClient()
    use_llm = client.available and not force_lexicon
    scorer = "llm" if use_llm else "lexicon"
    with connect(read_only=True) as con:
        todo = con.execute(f"""select id, ticker, headline, summary from news n
            where not exists (select 1 from news_scores s where s.news_id = n.id and s.scorer = ?)
            order by received_at desc limit {int(limit)}""", [scorer]).df()
    if todo.empty:
        return {"scored": 0, "scorer": scorer}
    out: list[tuple[str, NewsScore]] = []
    if not use_llm:
        out = [(r.id, lexicon_score(r.headline, r.summary or "", r.ticker)) for r in todo.itertuples()]
    else:
        bs = get_config()["llm"]["news_batch"]
        recs = todo.to_dict("records")
        for i in range(0, len(recs), bs):
            chunk = recs[i:i + bs]
            res = client.chat_json([{"role": "system", "content": NEWS_SYSTEM},
                                    {"role": "user", "content": news_user(chunk)}], NewsBatch, max_tokens=250 * len(chunk))
            if res is None or len(res.items) != len(chunk):
                res_items = [lexicon_score(c["headline"], c.get("summary") or "", c.get("ticker")) for c in chunk]
                log.debug("LLM batch fallback to lexicon")
            else:
                res_items = res.items
            out += [(c["id"], s) for c, s in zip(chunk, res_items)]
            if i % (bs * 25) == 0:
                log.info("news scoring %d/%d", i, len(recs))
                _store(out, scorer)
                out = []
    n = _store(out, scorer)
    return {"scored": len(todo), "scorer": scorer, "stored_last": n}


def sector_view(date: pd.Timestamp, use_rag: bool = True) -> dict[str, float]:
    """Nightly scenario judgment per sector -> one feature (llm_sector_view). Falls back to an
    aggregate of the day's sector_impact scores when no LLM is configured."""
    from ..utils import assign_trading_date
    from ..db import trading_dates

    with connect(read_only=True) as con:
        td = trading_dates(con)
        lo = date - pd.Timedelta(days=4)
        df = con.execute("""select n.id, n.headline, n.received_at, s.payload, s.event_type, s.severity, s.sentiment
            from news n join news_scores s on s.news_id = n.id
            where n.received_at >= ? and n.received_at <= ? + interval 1 day""", [lo, date]).df()
    if df.empty:
        return {}
    df["tdate"] = assign_trading_date(df["received_at"], td, get_config()["news"]["cutoff_et"])
    df = df[df["tdate"] == date]
    agg = {s: 0.0 for s in SECTORS}
    for p, sev in zip(df["payload"], df["severity"].fillna(2)):
        try:
            for k, v in (json.loads(p).get("sector_impact") or {}).items():
                if k in agg:
                    agg[k] += {"+": 1, "-": -1}.get(v, 0) * sev / 5
        except (ValueError, TypeError):
            continue
    fallback = {k: max(-2.0, min(2.0, v)) for k, v in agg.items()}
    client = LLMClient()
    source, scores, rationale = "aggregate", fallback, ""
    if client.available and len(df):
        top = df.sort_values("severity", ascending=False).head(40)
        digest = "\n".join(f"- [{r.event_type}, sev {r.severity:.0f}, sent {r.sentiment:+.2f}] {r.headline}" for r in top.itertuples())
        ctx = ""
        if use_rag:
            from ..rag.store import RagStore
            hits = RagStore().search(digest[:2000], k=3)
            ctx = "\n".join(f"* {h['text'][:300]}" for h in hits)
        res = client.chat_json([{"role": "system", "content": SECTOR_SYSTEM},
                                {"role": "user", "content": f"Date: {date.date()}\nDigest:\n{digest}\n\nSimilar past cases:\n{ctx or '-'}"
                                                            f"\n\nAggregate sector impact: {json.dumps(fallback)}"}], SectorView)
        if res:
            source, scores, rationale = "llm", res.scores, res.rationale
    rows = pd.DataFrame([{"date": date.date(), "sector": k, "score": float(v), "source": source, "rationale": rationale}
                         for k, v in scores.items()])
    with connect() as con:
        upsert(con, "sector_views", rows)
    return scores

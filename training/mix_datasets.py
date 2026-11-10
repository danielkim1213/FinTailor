"""Mix own data with a little public data into one chat JSONL (default 7000 news / 2000 explain /
1000 general finance Q&A).
Run in Colab (needs `datasets`). CHECK LICENSES before use - some sets are research-only.

    python training/mix_datasets.py --data training/data --out training/data/train_mix.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fintailor.llm.prompts import NEWS_SYSTEM  # noqa: E402

SENT_MAP = {"positive": 0.6, "negative": -0.6, "neutral": 0.0, "moderately positive": 0.3, "moderately negative": -0.3,
            "strong positive": 0.9, "strong negative": -0.9, "mildly positive": 0.3, "mildly negative": -0.3}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def public_sentiment(n: int, seed: int) -> list[dict]:
    """FinGPT sentiment -> our NewsScore format (sentiment only; other fields default)."""
    from datasets import load_dataset
    ds = load_dataset("FinGPT/fingpt-sentiment-train", split="train").shuffle(seed=seed).select(range(n))
    out = []
    for r in ds:
        lab = str(r.get("output", "")).strip().lower()
        if lab not in SENT_MAP:
            continue
        tgt = {"items": [{"tickers": [], "event_type": "other", "region": "US", "severity": 2, "novelty": "new",
                          "sentiment": SENT_MAP[lab], "sector_impact": {}, "expected_horizon": "days",
                          "confidence": 0.5, "direction_5d": 0}]}
        out.append({"messages": [{"role": "system", "content": NEWS_SYSTEM},
                                 {"role": "user", "content": "Headlines:\n1. [-] " + str(r["input"])[:400]},
                                 {"role": "assistant", "content": json.dumps(tgt)}], "meta": {"src": "fingpt-sentiment"}})
    return out


def public_instruct(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("Josephgflowers/Finance-Instruct-500k", split="train").shuffle(seed=seed).select(range(n))
    out = []
    for r in ds:
        u, a = r.get("user") or r.get("instruction"), r.get("assistant") or r.get("output")
        if not u or not a or len(a) > 4000:
            continue
        msgs = ([{"role": "system", "content": r["system"]}] if r.get("system") else []) + \
               [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        out.append({"messages": msgs, "meta": {"src": "finance-instruct"}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training/data")
    ap.add_argument("--out", default="training/data/train_mix.jsonl")
    # Explicit counts so the mix survives finetune's --max-examples cap (it used to be ~95% news).
    ap.add_argument("--n-news", type=int, default=7000)
    ap.add_argument("--n-explain", type=int, default=2000)
    ap.add_argument("--public-instruct", type=int, default=1000, help="general finance Q&A (keeps base skills)")
    # FinGPT sentiment is OFF by default: its converted targets hard-code event_type="other", severity=2,
    # direction_5d=0 for every row, which teaches the model to stop classifying events.
    ap.add_argument("--public-sent", type=int, default=0)
    ap.add_argument("--no-public", action="store_true")
    ap.add_argument("--seed", type=int, default=3407)
    a = ap.parse_args()
    d = Path(a.data)
    rng = random.Random(a.seed)
    news, expl = read_jsonl(d / "news_train.jsonl"), read_jsonl(d / "explain_train.jsonl")
    news = rng.sample(news, min(a.n_news, len(news)))
    expl = rng.sample(expl, min(a.n_explain, len(expl)))
    pub = []
    if not a.no_public:
        if a.public_sent:
            pub += public_sentiment(a.public_sent, a.seed)
        if a.public_instruct:
            pub += public_instruct(a.public_instruct, a.seed)
    allr = news + expl + pub
    random.Random(a.seed).shuffle(allr)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in allr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"mix: news {len(news)}, explain {len(expl)}, public {len(pub)} -> {a.out}")


if __name__ == "__main__":
    main()

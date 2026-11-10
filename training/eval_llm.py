"""LLM evaluation + Gate 1 check (plan section 10). Compares two OpenAI-compatible endpoints,
e.g. base vs fine-tuned (llama.cpp `llama-server -m model.gguf --port 8080`, or vLLM).

    python training/eval_llm.py --news training/data/news_test.jsonl --explain training/data/explain_train.jsonl \
        --base http://localhost:8081/v1:qwen-base --tuned http://localhost:8080/v1:fintailor --n 300

Metrics: JSON validity, event-type accuracy (vs teacher), sentiment F1 (financial_phrasebank if
`datasets` is installed, else teacher labels), direction_5d accuracy vs realized, invented numbers
and banned words in explanations, latency p95.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fintailor.llm import guardrails  # noqa: E402
from fintailor.llm.client import LLMClient, extract_json  # noqa: E402
from fintailor.llm.prompts import NEWS_SYSTEM  # noqa: E402
from fintailor.llm.schemas import NewsBatch  # noqa: E402

GATE1 = {"json_valid": 0.99, "invented_numbers_rate_max": 0.01, "event_acc": 0.85, "sent_f1_gain": 0.05,
         "latency_p95_max": 15.0}


def _client(spec: str) -> LLMClient:
    url, _, model = spec.rpartition(":")
    if not url.startswith("http"):
        url, model = spec, "default"
    return LLMClient(base_url=url, model=model)


def _f1(y, p, labels=(-1, 0, 1)) -> float:
    y, p = np.asarray(y), np.asarray(p)
    f = []
    for c in labels:
        tp, fp, fn = np.sum((p == c) & (y == c)), np.sum((p == c) & (y != c)), np.sum((p != c) & (y == c))
        f.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def _bucket(s: float) -> int:
    return 1 if s > 0.2 else -1 if s < -0.2 else 0


def eval_news(client: LLMClient, recs: list[dict]) -> dict:
    ok, ev_hit, sp, st, dp, dr, lat = 0, 0, [], [], [], [], []
    for r in recs:
        msgs = r["messages"][:2]
        teacher = json.loads(r["messages"][2]["content"])["items"][0]
        t0 = time.time()
        try:
            out = NewsBatch.model_validate(extract_json(client.chat(msgs, max_tokens=300, json_mode=True))).items[0]
            ok += 1
        except Exception:  # noqa: BLE001
            lat.append(time.time() - t0)
            continue
        lat.append(time.time() - t0)
        ev_hit += out.event_type == teacher["event_type"]
        sp.append(_bucket(out.sentiment))
        st.append(_bucket(teacher["sentiment"]))
        ex = (r.get("meta") or {}).get("ex_5d")
        if ex not in (None, "None") and abs(float(ex)) > 0.005:
            dp.append(out.direction_5d)
            dr.append(int(np.sign(float(ex))))
    n = len(recs)
    return {"n": n, "json_valid": ok / n, "event_acc": ev_hit / max(ok, 1), "sent_f1_vs_teacher": _f1(st, sp),
            "direction_acc_vs_realized": float(np.mean(np.array(dp) == np.array(dr))) if dp else None,
            "latency_p95": float(np.percentile(lat, 95)) if lat else None}


def eval_phrasebank(client: LLMClient, n: int) -> float | None:
    try:
        from datasets import load_dataset
        ds = load_dataset("takala/financial_phrasebank", "sentences_allagree", split="train", trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        print("phrasebank unavailable:", e)
        return None
    ds = ds.shuffle(seed=1).select(range(min(n, len(ds))))
    mp = {0: -1, 1: 0, 2: 1}
    y, p = [], []
    for r in ds:
        msgs = [{"role": "system", "content": NEWS_SYSTEM}, {"role": "user", "content": "Headlines:\n1. [-] " + r["sentence"]}]
        try:
            out = NewsBatch.model_validate(extract_json(client.chat(msgs, max_tokens=300, json_mode=True))).items[0]
            p.append(_bucket(out.sentiment))
        except Exception:  # noqa: BLE001
            p.append(0)
        y.append(mp[r["label"]])
    return _f1(y, p)


def eval_explain(client: LLMClient, recs: list[dict]) -> dict:
    inv, ban, n = 0, 0, 0
    for r in recs:
        msgs = r["messages"][:2]
        payload = json.loads(msgs[1]["content"].split("\n", 1)[1])
        lang = (r.get("meta") or {}).get("lang", "en")
        try:
            text = client.chat(msgs, max_tokens=700)
        except Exception:  # noqa: BLE001
            continue
        res = guardrails.check(text, payload, lang)
        inv += bool(res["invented_numbers"])
        ban += bool(res["banned"])
        n += 1
    return {"n": n, "invented_numbers_rate": inv / max(n, 1), "banned_rate": ban / max(n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", required=True)
    ap.add_argument("--explain")
    ap.add_argument("--base", help="URL:model of the base model")
    ap.add_argument("--tuned", required=True, help="URL:model of the fine-tuned model")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    news = [json.loads(x) for x in Path(a.news).read_text(encoding="utf-8").splitlines()][: a.n]
    expl = [json.loads(x) for x in Path(a.explain).read_text(encoding="utf-8").splitlines()][-a.n:] if a.explain else []
    res = {}
    for name, spec in (("base", a.base), ("tuned", a.tuned)):
        if not spec:
            continue
        c = _client(spec)
        res[name] = {"news": eval_news(c, news), "phrasebank_f1": eval_phrasebank(c, a.n)}
        if expl:
            res[name]["explain"] = eval_explain(c, expl)
    t, b = res["tuned"], res.get("base")
    gate = {"json_valid>=99%": t["news"]["json_valid"] >= GATE1["json_valid"],
            "event_acc>=85%": t["news"]["event_acc"] >= GATE1["event_acc"],
            "latency_p95<15s": (t["news"]["latency_p95"] or 99) < GATE1["latency_p95_max"]}
    if expl:
        gate["invented_numbers<1%"] = t["explain"]["invented_numbers_rate"] < GATE1["invented_numbers_rate_max"]
    if b and t.get("phrasebank_f1") is not None and b.get("phrasebank_f1") is not None:
        gate["sent_f1 +5pp vs base"] = t["phrasebank_f1"] - b["phrasebank_f1"] >= GATE1["sent_f1_gain"]
    res["gate1"] = gate
    res["gate1_pass"] = all(gate.values())
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

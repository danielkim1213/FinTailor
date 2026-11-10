"""Compare teacher LLMs on the same news sample before labeling the full dataset.

Runs the exact production prompt (NEWS_SYSTEM / news_user, batches of 8) against each model via an
OpenAI-compatible endpoint (default: OpenRouter) and reports JSON validity, retries, cost, latency,
and how much the models agree with each other. Outputs side-by-side files for eyeballing.

    python training/compare_teachers.py                       # 3 default models, 48 headlines
    python training/compare_teachers.py --n 96 --models anthropic/claude-haiku-5.5 openai/gpt-6-luna

Key: TEACHER_API_KEY (or OPENROUTER_API_KEY) in .env. Base URL: TEACHER_BASE_URL or OpenRouter.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fintailor.config import env  # noqa: E402
from fintailor.db import connect  # noqa: E402
from fintailor.llm.client import extract_json  # noqa: E402
from fintailor.llm.prompts import NEWS_SYSTEM, news_user  # noqa: E402
from fintailor.llm.schemas import NewsBatch  # noqa: E402

# free tier (no credits). Paid alternatives at similar quality/price: anthropic/claude-haiku-5.5,
# openai/gpt-6-luna, google/gemini-3.5-flash-lite
DEFAULT_MODELS = ["google/gemma-4-26b-a4b-it:free","nvidia/nemotron-3.5-super-vl:free",
                  "nvidia/nemotron-3-ultra-550b-a55b:free"]
OPENROUTER = "https://openrouter.ai/api/v1"


def load_sample(n: int, seed: int) -> list[dict]:
    with connect(read_only=True) as con:
        df = con.execute("select id, ticker, headline, summary from news where headline is not null").df()
    if df.empty:
        raise SystemExit("no news in the store yet - run `python -m fintailor collect news` first")
    # prefer ticker news (what the dataset is mostly made of), fill with general/RSS
    with_t, without = df[df["ticker"].notna()], df[df["ticker"].isna()]
    k = min(n, len(with_t))
    s = pd.concat([with_t.sample(k, random_state=seed),
                   without.sample(min(n - k, len(without)), random_state=seed)])
    s = s.where(pd.notna(s), None)
    return s.to_dict("records")


class RateLimited(Exception):
    pass


class Teacher:
    def __init__(self, model: str, base_url: str, key: str, temperature: float, retries: int):
        self.model, self.base_url, self.key = model, base_url.rstrip("/"), key
        self.temperature, self.retries = temperature, retries
        self.json_mode = True
        self.stats = dict(calls=0, http_errors=0, tok_in=0, tok_out=0, cost=0.0, latency=[])

    def _post(self, messages: list[dict], max_tokens: int) -> str:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": self.temperature, "usage": {"include": True},
                # reasoning models otherwise spend the whole token budget thinking and return empty content
                "reasoning": {"enabled": False}}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        hdr = {"Authorization": f"Bearer {self.key}", "X-Title": "FinTailor teacher compare"}
        for wait in (15, 30, 60, None):  # free models rate-limit hard (429): back off, then give up
            t0 = time.time()
            r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=hdr, timeout=180)
            if r.status_code == 400 and self.json_mode:
                self.json_mode = False
                body.pop("response_format")
                r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=hdr, timeout=180)
            if r.status_code == 429 and wait is None:
                raise RateLimited(f"{self.model}: still rate-limited after backoff")
            if r.status_code != 429:
                break
            print(f"    429 rate limited, waiting {wait}s: {r.text[:150]}", flush=True)
            time.sleep(wait)
        self.stats["calls"] += 1
        if r.status_code in (401, 402, 403):
            raise SystemExit(f"{self.model}: HTTP {r.status_code} {r.text[:300]}")
        r.raise_for_status()
        self.stats["latency"].append(time.time() - t0)
        j = r.json()
        if "error" in j:
            raise requests.RequestException(str(j["error"])[:300])
        u = j.get("usage") or {}
        self.stats["tok_in"] += u.get("prompt_tokens", 0) or 0
        self.stats["tok_out"] += u.get("completion_tokens", 0) or 0
        self.stats["cost"] += float(u.get("cost", 0) or 0)
        return j["choices"][0]["message"]["content"] or ""

    def label(self, chunk: list[dict]) -> tuple[list[dict] | None, int, str]:
        """Returns (items or None, attempts used, last error)."""
        messages = [{"role": "system", "content": NEWS_SYSTEM}, {"role": "user", "content": news_user(chunk)}]
        err = ""
        for attempt in range(1, self.retries + 2):
            try:
                out = NewsBatch.model_validate(extract_json(self._post(messages, 250 * len(chunk))))
                if len(out.items) != len(chunk):
                    raise ValueError(f"expected {len(chunk)} items, got {len(out.items)}")
                return [it.model_dump() for it in out.items], attempt, ""
            except (ValueError, ValidationError, requests.RequestException) as e:
                err = str(e)[:200]
                self.stats["http_errors"] += isinstance(e, requests.RequestException)
                messages = messages + [{"role": "user", "content": f"Your last output was invalid ({err}). "
                                                                   "Reply with valid JSON only."}]
        return None, self.retries + 1, err


def run_model(t: Teacher, sample: list[dict], batch: int) -> tuple[list[dict | None], dict]:
    labels: list[dict | None] = []
    first_ok = final_ok = 0
    errors = []
    nb = (len(sample) + batch - 1) // batch
    limited = 0
    for b, i in enumerate(range(0, len(sample), batch)):
        chunk = sample[i:i + batch]
        if limited >= 2:  # provider keeps refusing: don't burn minutes per batch
            labels += [None] * len(chunk)
            continue
        try:
            items, attempts, err = t.label(chunk)
            limited = 0
        except RateLimited as e:
            items, attempts, err = None, 0, str(e)
            limited += 1
            if limited >= 2:
                print(f"  {t.model}: rate-limited twice in a row, skipping the rest", flush=True)
        first_ok += items is not None and attempts == 1
        final_ok += items is not None
        if err and items is None:
            errors.append(err)
        labels += items if items else [None] * len(chunk)
        print(f"  {t.model}: batch {b + 1}/{nb} {'ok' if items else 'FAILED'}"
              f"{f' (after {attempts} tries)' if items and attempts > 1 else ''}", flush=True)
    lat = t.stats["latency"]
    s = dict(model=t.model, batches=nb, first_try_ok=f"{first_ok}/{nb}", final_ok=f"{final_ok}/{nb}",
             fallback_items=sum(x is None for x in labels), calls=t.stats["calls"],
             tok_in=t.stats["tok_in"], tok_out=t.stats["tok_out"], cost_usd=round(t.stats["cost"], 5),
             cost_per_1k_headlines=round(t.stats["cost"] / max(len(sample), 1) * 1000, 3),
             avg_latency_s=round(float(np.mean(lat)), 1) if lat else None, json_mode=t.json_mode,
             errors=errors[:3])
    ok = [x for x in labels if x]
    if ok:
        s["sentiment_mean"] = round(float(np.mean([x["sentiment"] for x in ok])), 3)
        s["sentiment_zero_share"] = round(float(np.mean([abs(x["sentiment"]) < 0.05 for x in ok])), 2)
        s["event_types"] = len({x["event_type"] for x in ok})
        s["confidence_mean"] = round(float(np.mean([x["confidence"] for x in ok])), 2)
    return labels, s


def agreement(a: list, b: list) -> dict:
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs:
        return {}
    sg = lambda v: 0 if abs(v) < 0.1 else int(np.sign(v))  # noqa: E731
    sx = np.array([x["sentiment"] for x, _ in pairs])
    sy = np.array([y["sentiment"] for _, y in pairs])
    return dict(n=len(pairs),
                sentiment_sign=round(float(np.mean([sg(x["sentiment"]) == sg(y["sentiment"]) for x, y in pairs])), 2),
                sentiment_corr=round(float(np.corrcoef(sx, sy)[0, 1]), 2) if sx.std() > 0 and sy.std() > 0 else None,
                event_type=round(float(np.mean([x["event_type"] == y["event_type"] for x, y in pairs])), 2),
                severity_within_1=round(float(np.mean([abs(x["severity"] - y["severity"]) <= 1 for x, y in pairs])), 2),
                horizon=round(float(np.mean([x["expected_horizon"] == y["expected_horizon"] for x, y in pairs])), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--n", type=int, default=48, help="headlines (multiple of --batch keeps batches equal)")
    ap.add_argument("--batch", type=int, default=8, help="same as production")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="training/compare")
    a = ap.parse_args()

    key = env("OPENROUTER_API_KEY") or env("TEACHER_API_KEY")
    if not key:
        raise SystemExit("set TEACHER_API_KEY (or OPENROUTER_API_KEY) in .env")
    base = env("TEACHER_BASE_URL") or OPENROUTER

    sample = load_sample(a.n, a.seed)
    print(f"{len(sample)} headlines ({sum(1 for s in sample if s['ticker'])} with ticker), "
          f"{len(a.models)} models, endpoint {base}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    labels, summary = {}, []
    for m in a.models:
        labels[m], s = run_model(Teacher(m, base, key, a.temperature, a.retries), sample, a.batch)
        summary.append(s)

    # side-by-side CSV: one row per headline, key fields per model
    rows = []
    for i, it in enumerate(sample):
        r = {"ticker": it["ticker"], "headline": it["headline"]}
        for m in a.models:
            x = labels[m][i]
            short = m.split("/")[-1]
            r[f"{short}_sentiment"] = x["sentiment"] if x else None
            r[f"{short}_event"] = x["event_type"] if x else None
            r[f"{short}_severity"] = x["severity"] if x else None
            r[f"{short}_horizon"] = x["expected_horizon"] if x else None
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "side_by_side.csv", index=False, encoding="utf-8-sig")
    with open(out / "labels.jsonl", "w", encoding="utf-8") as f:
        for i, it in enumerate(sample):
            f.write(json.dumps({"news": it, **{m: labels[m][i] for m in a.models}}, ensure_ascii=False, default=str) + "\n")

    agree = {f"{x.split('/')[-1]} vs {y.split('/')[-1]}": agreement(labels[x], labels[y])
             for x, y in combinations(a.models, 2)}
    report = {"n": len(sample), "summary": summary, "agreement": agree}
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    cols = ["model", "final_ok", "first_try_ok", "fallback_items", "cost_usd", "cost_per_1k_headlines",
            "avg_latency_s", "sentiment_mean", "sentiment_zero_share", "event_types", "confidence_mean"]
    print("\n" + pd.DataFrame(summary)[[c for c in cols if c in pd.DataFrame(summary)]].to_string(index=False))
    print("\nagreement (share of headlines where the two models match):")
    print(pd.DataFrame(agree).T.to_string())
    for s in summary:
        if s["errors"]:
            print(f"\n{s['model']} errors: {s['errors']}")
    print(f"\nfiles: {out / 'side_by_side.csv'}  {out / 'labels.jsonl'}  {out / 'report.json'}")


if __name__ == "__main__":
    main()

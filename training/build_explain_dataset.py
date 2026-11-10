"""FinTailor-Explain dataset: (prediction + client profile) -> plain-English explanation, US/CA clients
(KR/Korean disabled for now - English-first). Teacher drafts are kept ONLY if they pass the production guardrails.

    python training/build_explain_dataset.py --n 3000 --out training/data
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fintailor.llm import guardrails  # noqa: E402
from fintailor.llm.client import LLMClient  # noqa: E402
from fintailor.llm.explainer import FEATURE_LABELS, build_payload, label, template  # noqa: E402
from fintailor.llm.prompts import EXPLAIN_SYSTEM, explain_user  # noqa: E402
from fintailor.signals.profiles import COUNTRY_PACKS  # noqa: E402
from fintailor.utils import log, setup_logging  # noqa: E402

TICKERS = ["XLE", "XLK", "XLF", "XLV", "XLI", "AAPL", "MSFT", "NVDA", "JPM", "XOM", "UNH", "AMZN", "META", "LLY", "CAT"]


def random_signal(rng: random.Random, country: str) -> tuple[dict, dict, str]:
    pack = COUNTRY_PACKS[country]
    lang = "en"  # KR disabled: "ko" if country == "KR" else ("ko" if rng.random() < 0.1 else "en")
    ref = round(rng.uniform(20, 600), 2)
    stop_pct = round(rng.uniform(0.02, 0.12), 4)
    rate = 1.0  # USD-only (was {"USD": 1, "CAD": 1.37, "KRW": 1390}[pack["currency"]])
    pos = round(rng.uniform(500, 20000), 2)
    feats = rng.sample(list(FEATURE_LABELS), 3)
    risks = rng.sample(["earnings inside holding period", "high volatility", "overnight gap risk",
                        "scheduled events: FOMC", "scheduled events: CPI"], 2)
    # KR disabled (English-first):
    # if lang == "ko":
    #     risks = [{"earnings inside holding period": "보유 기간 중 실적 발표", "high volatility": "높은 변동성",
    #               "overnight gap risk": "장 마감 후 갭 위험"}.get(r, r.replace("scheduled events", "예정 일정")) for r in risks]
    sig = {"ticker": rng.choice(TICKERS), "direction": "long", "horizon": rng.choice([1, 5, 10, 30]),
           "prob": round(rng.uniform(0.57, 0.7), 4), "exp_excess_net": round(rng.uniform(0.001, 0.03), 4),
           "ref_price": ref, "stop_pct": stop_pct, "stop_price": round(ref * (1 - stop_pct), 2),
           "max_position_usd": pos, "max_position_local": int(round(pos * rate)), "currency": pack["currency"],
           "drivers": [[f, rng.choice([-1, 1]) * rng.uniform(0.01, 0.2)] for f in feats], "risks": risks,
           "disclaimer": pack["disclaimer"][lang]}
    prof = {"country": country, "age": rng.randint(22, 70), "experience": rng.choice(["intermediate", "advanced"]),
            "risk_score": rng.randint(5, 10)}
    return sig, prof, lang


def payload_for(sig: dict, prof: dict, lang: str) -> dict:
    return build_payload(sig, lang, prof, "")  # same builder as production


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default="training/data")
    ap.add_argument("--teacher", choices=["llm", "template"], default="llm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="parallel teacher requests")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    client = LLMClient(prefix="TEACHER") if a.teacher == "llm" else None
    if client is not None and not client.available:
        raise SystemExit("TEACHER_BASE_URL / TEACHER_MODEL not set (or --teacher template)")
    outd = Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor

    def draft(job):
        sig, payload, msgs, lang = job
        if not client:
            return template(sig, lang)
        try:
            return client.chat(msgs, max_tokens=700)
        except Exception as e:  # noqa: BLE001 - one failed call must not kill the run
            log.warning("teacher call failed: %s", e)
            return None

    kept = rejected = failed = 0
    with open(outd / "explain_train.jsonl", "w", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=a.workers if client else 1) as ex:
        while kept < a.n:
            jobs = []  # generated on the main thread so the rng sequence stays reproducible
            for _ in range(min(a.workers * 4, max(a.n - kept, 1) * 2)):
                country = rng.choice(["US", "CA"])  # "KR" disabled
                sig, prof, lang = random_signal(rng, country)
                payload = payload_for(sig, prof, lang)
                msgs = [{"role": "system", "content": EXPLAIN_SYSTEM[lang]},
                        {"role": "user", "content": explain_user(payload)}]
                jobs.append((sig, payload, msgs, lang, country))
            for (sig, payload, msgs, lang, country), text in zip(jobs, ex.map(lambda j: draft(j[:4]), jobs)):
                if kept >= a.n:
                    break
                if text is None:
                    failed += 1
                    continue
                res = guardrails.check(text, payload, lang)
                if not res["ok"] or sig["disclaimer"][:20] not in text:
                    rejected += 1
                    continue
                f.write(json.dumps({"messages": msgs + [{"role": "assistant", "content": text}],
                                    "meta": {"country": country, "lang": lang}}, ensure_ascii=False) + "\n")
                kept += 1
                if kept % 200 == 0:
                    log.info("explain: kept %d, rejected %d, failed %d", kept, rejected, failed)
            if failed > 50 and kept == 0:
                raise SystemExit("teacher keeps failing - check TEACHER_* in .env")
    log.info("explain dataset: %d kept, %d rejected by guardrails", kept, rejected)


if __name__ == "__main__":
    main()

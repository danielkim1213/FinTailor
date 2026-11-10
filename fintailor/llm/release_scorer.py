"""LLM reading of earnings press releases (data/earnings_releases.py excerpts).

The numbers (EPS, surprise) already come from the earnings table; the LLM judges what numbers miss:
guidance direction, management tone, results vs expectations as described, and whether one-off
items drive the headline. Output is a tiny JSON to keep cost down (~25 output tokens).

Hard budget: stops once the cumulative OpenRouter-reported cost of this run reaches `budget_usd`.
Resumable: (accession, model) pairs already scored are skipped.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from ..config import env
from ..db import connect, upsert
from ..utils import log, utcnow
from .client import extract_json

SCHEMA = """create table if not exists release_scores(
  accession varchar, model varchar, g varchar, t double, s double, x integer, created_at timestamp,
  primary key(accession, model))"""

RELEASE_SYSTEM = """You read the key parts of a US company's quarterly earnings press release (tables removed).
Judge only what the text says; do not use outside knowledge of later events. Reply with compact JSON only:
{"g": guidance vs prior guidance: "up" raised, "down" lowered/withdrawn, "same" reaffirmed, "new" first-time, "none" no guidance,
 "t": management tone -2 very negative .. 2 very positive,
 "s": results vs expectations / prior period as described: -2 big miss or decline .. 2 big beat or strong growth,
 "x": 1 if one-off items (charges, gains, litigation, tariffs) drive the headline numbers, else 0}"""

G_OK = {"up", "down", "same", "new", "none"}


def _parse(text: str) -> dict | None:
    try:
        d = extract_json(text)
    except ValueError:
        return None
    if not isinstance(d, dict):
        return None
    g = str(d.get("g", "none")).lower()
    try:
        t, s, x = float(d.get("t", 0)), float(d.get("s", 0)), int(d.get("x", 0))
    except (TypeError, ValueError):
        return None
    if g not in G_OK:
        return None
    return {"g": g, "t": max(-2.0, min(2.0, t)), "s": max(-2.0, min(2.0, s)), "x": 1 if x else 0}


def score_releases(model: str, accessions: list[str] | None = None, limit: int | None = None, workers: int = 16,
                   budget_usd: float = 0.5, max_chars: int = 3000) -> dict:
    base = (env("TEACHER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
    key = env("TEACHER_API_KEY") or env("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("TEACHER_API_KEY not set")
    with connect() as con:
        con.execute(SCHEMA)
        q = """select r.accession, r.ticker, r.excerpt from earnings_releases r
               where length(r.excerpt) > 200 and not exists
               (select 1 from release_scores s where s.accession = r.accession and s.model = ?)"""
        todo = con.execute(q + " order by r.accession", [model]).df()
    if accessions is not None:
        todo = todo[todo["accession"].isin(set(accessions))]
    if limit:
        todo = todo.head(limit)
    lock = threading.Lock()
    spent = {"usd": 0.0, "tok_in": 0, "tok_out": 0, "ok": 0, "bad": 0, "stopped": False}
    out: list[dict] = []

    def one(rec) -> None:
        if spent["stopped"]:
            return
        body = {"model": model, "temperature": 0, "max_tokens": 80, "usage": {"include": True},
                "reasoning": {"enabled": False}, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": RELEASE_SYSTEM},
                             {"role": "user", "content": f"[{rec.ticker}]\n{rec.excerpt[:max_chars]}"}]}
        try:
            r = requests.post(f"{base}/chat/completions", json=body, timeout=120,
                              headers={"Authorization": f"Bearer {key}", "X-Title": "FinTailor release scoring"})
            if r.status_code == 400:
                body.pop("response_format")
                r = requests.post(f"{base}/chat/completions", json=body, timeout=120,
                                  headers={"Authorization": f"Bearer {key}"})
            j = r.json()
        except Exception as e:  # noqa: BLE001
            log.warning("release %s: %s", rec.accession, e)
            return
        u = j.get("usage") or {}
        res = _parse((j.get("choices") or [{}])[0].get("message", {}).get("content") or "") if "choices" in j else None
        with lock:
            spent["usd"] += float(u.get("cost", 0) or 0)
            spent["tok_in"] += u.get("prompt_tokens", 0) or 0
            spent["tok_out"] += u.get("completion_tokens", 0) or 0
            if res:
                spent["ok"] += 1
                out.append({"accession": rec.accession, "model": model, **res, "created_at": utcnow()})
            else:
                spent["bad"] += 1
            if spent["usd"] >= budget_usd:
                spent["stopped"] = True

    def flush():
        with lock:
            batch, out[:] = list(out), []
        if batch:
            with connect() as con:
                upsert(con, "release_scores", pd.DataFrame(batch))

    recs = list(todo.itertuples())
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(0, len(recs), 500):
            list(ex.map(one, recs[i:i + 500]))
            flush()
            log.info("release scoring %s: %d/%d, $%.3f spent, %d bad", model, min(i + 500, len(recs)), len(recs),
                     spent["usd"], spent["bad"])
            if spent["stopped"]:
                log.warning("budget $%.2f reached - stopping", budget_usd)
                break
    flush()
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in spent.items()} | {"todo": len(recs)}


__all__ = ["score_releases", "RELEASE_SYSTEM", "json"]

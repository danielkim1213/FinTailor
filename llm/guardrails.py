"""Output guardrails: (1) every number in generated text must come from the input,
(2) banned phrases (guarantees, hype, leverage)."""
from __future__ import annotations

import re
from collections.abc import Iterable

NUM = re.compile(r"(?<![A-Za-z0-9.,])[-+]?\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![A-Za-z0-9])|"
                 r"(?<![A-Za-z0-9.,])[-+]?\$?\d+(?:\.\d+)?(?![A-Za-z0-9])")
LIST_MARK = re.compile(r"^\s*\d+[.)]\s", re.M)

BANNED = {
    "en": [r"guarantee", r"risk[- ]free", r"sure thing", r"can't lose", r"cannot lose", r"\bmargin\b", r"leverage",
           r"to the moon", r"\bcertain(ly)? (to|will)\b", r"no[- ]brainer", r"double your"],
    "ko": [r"보장(?!하지\s*않)", r"확실", r"무조건", r"원금\s*보장", r"신용\s*매수", r"미수", r"레버리지", r"대박", r"100%\s*수익", r"손실\s*없"],
}


def _flatten_numbers(obj, out: list[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        for m in NUM.findall(obj):
            try:
                out.append(float(m.replace(",", "").replace("$", "")))
            except ValueError:
                pass
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _flatten_numbers(k, out)
            _flatten_numbers(v, out)
    elif isinstance(obj, Iterable):
        for v in obj:
            _flatten_numbers(v, out)


def allowed_numbers(payload) -> list[float]:
    vals: list[float] = []
    _flatten_numbers(payload, vals)
    extra = []
    for v in vals:
        extra += [v * 100, abs(v), abs(v) * 100]
    return vals + extra


def _matches(x: float, decimals: int, allowed: list[float]) -> bool:
    for a in allowed:
        if round(a, decimals) == x or round(a, decimals) == -x:
            return True
        if decimals == 0 and abs(a) >= 1000 and abs(a - x) / abs(a) < 0.01:  # rounded large amounts
            return True
    return False


def invented_numbers(text: str, payload) -> list[str]:
    allowed = allowed_numbers(payload) + [0.0, 1.0, 100.0]
    clean = LIST_MARK.sub(" ", text)
    bad = []
    for m in NUM.findall(clean):
        s = m.replace(",", "").replace("$", "").lstrip("+")
        try:
            x = float(s)
        except ValueError:
            continue
        dec = len(s.split(".")[1]) if "." in s else 0
        if not _matches(x, dec, allowed):
            bad.append(m)
    return bad


def banned_phrases(text: str, lang: str) -> list[str]:
    hits = []
    for rx in BANNED.get(lang, []) + (BANNED["en"] if lang != "en" else []):
        if re.search(rx, text, flags=re.I):
            hits.append(rx)
    return hits


def check(text: str, payload, lang: str = "en") -> dict:
    nums = invented_numbers(text, payload)
    disc = payload.get("disclaimer") if isinstance(payload, dict) else None
    ban = banned_phrases(text.replace(disc, " ") if disc else text, lang)
    return {"ok": not nums and not ban, "invented_numbers": nums, "banned": ban}

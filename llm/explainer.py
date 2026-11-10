"""Signal explanations in the client's language. LLM with guardrails + regeneration; a deterministic
template is the fallback (and is always guardrail-clean)."""
from __future__ import annotations

from ..config import get_config
from ..utils import log
from . import guardrails
from .client import _THINK, LLMClient
from .prompts import EXPLAIN_SYSTEM, explain_user

FEATURE_LABELS = {
    "ret_1": ("1-day return", "1일 수익률"), "ret_3": ("3-day return", "3일 수익률"),
    "ret_5": ("1-week move (reversal)", "1주 등락(반전)"), "ret_10": ("2-week return", "2주 수익률"),
    "ret_20": ("1-month momentum", "1개월 모멘텀"), "mom_12_1": ("12-month momentum", "12개월 모멘텀"),
    "ma20_pos": ("position vs 20-day average", "20일 평균 대비 위치"), "ma50_pos": ("position vs 50-day average", "50일 평균 대비 위치"),
    "ma200_pos": ("position vs 200-day average", "200일 평균 대비 위치"), "gap": ("opening gap", "시초가 갭"),
    "vol_surge": ("volume surge", "거래량 급증"), "close_loc": ("close within day range", "당일 고저 범위 내 종가 위치"),
    "vwap_dev": ("close vs VWAP", "종가 vs VWAP"), "rv_20": ("recent volatility", "최근 변동성"),
    "atr_14": ("average true range", "ATR 변동폭"), "dist_52w_high": ("distance from 52-week high", "52주 고점 대비 거리"),
    "vs_sector_5": ("1-week strength vs sector", "섹터 대비 1주 강도"), "vs_sector_20": ("1-month strength vs sector", "섹터 대비 1개월 강도"),
    "sector_vs_spy_5": ("sector vs S&P 500 (1 week)", "섹터 vs S&P500 (1주)"), "sector_vs_spy_20": ("sector vs S&P 500 (1 month)", "섹터 vs S&P500 (1개월)"),
    "news_sent_5d": ("5-day news sentiment", "5일 뉴스 감성"), "news_cnt_5d": ("news volume", "뉴스 양"),
    # newer news features: Korean labels not written yet (KR disabled)
    "news_attn_1d": ("unusual news coverage today", ""), "news_attn_5d": ("unusual news coverage this week", ""),
    "news_sent_20d": ("1-month news sentiment", ""), "news_sent_chg": ("shift in news tone", ""),
    "news_csent_5d": ("high-confidence news sentiment", ""), "news_earn_sent_5d": ("earnings/guidance news", ""),
    "news_rating_sent_5d": ("analyst rating changes", ""), "news_legal_5d": ("legal/regulatory news", ""),
    "news_ma_5d": ("M&A news", ""),
    "sector_news_5d": ("sector news impact", "섹터 뉴스 영향"), "llm_sector_view": ("AI sector outlook", "AI 섹터 전망"),
    "days_to_earn": ("days to earnings", "실적 발표까지 일수"), "last_surprise": ("last earnings surprise", "직전 실적 서프라이즈"),
    "vix": ("VIX level", "VIX 수준"), "breadth_50": ("market breadth", "시장 폭"), "beta_60": ("market beta", "시장 베타"),
    "idio_vol_60": ("stock-specific volatility", "개별 변동성"), "earn_yield": ("earnings yield", "이익수익률"),
}


def label(feature: str, lang: str) -> str:
    # second element of each FEATURE_LABELS tuple is the Korean label (KR disabled, kept for later)
    en, _ko = FEATURE_LABELS.get(feature, (feature.replace("_", " "), feature.replace("_", " ")))
    return en


def template(sig: dict, lang: str) -> str:
    drv = ", ".join(f"{label(f, lang)}({'+' if v > 0 else '-'})" for f, v in sig.get("drivers", [])[:3])
    risks = "; ".join(sig.get("risks", [])) or "-"
    p = round(sig["prob"] * 100)
    ex = round(sig["exp_excess_net"] * 100, 2)
    # KR disabled (English-first):
    # if lang == "ko":
    #     return (f"{sig['ticker']} {'매수' if sig['direction'] == 'long' else '매도'} 신호입니다. 다음 거래일 시가에 진입해 "
    #             f"최대 {sig['horizon']}거래일 보유를 가정합니다. 모델이 추정한 S&P500 대비 초과 확률은 {p}%, "
    #             f"비용 차감 후 예상 초과수익은 {ex}%입니다. 손절 가격은 ${sig['stop_price']}이며, "
    #             f"포지션 상한은 {sig['max_position_local']:,} {sig['currency']}입니다. 핵심 요인: {drv}. 위험: {risks}. "
    #             f"{sig['disclaimer']}")
    return (f"{sig['ticker']}: {'buy' if sig['direction'] == 'long' else 'sell short'} at the next open, hold up to "
            f"{sig['horizon']} trading days. Model probability of beating the S&P 500: {p}%; expected excess return after "
            f"costs: {ex}%. Stop price: ${sig['stop_price']}. Position cap: {sig['max_position_local']:,} {sig['currency']}. "
            f"Main drivers: {drv}. Risks: {risks}. {sig['disclaimer']}")


def build_payload(sig: dict, lang: str, profile_summary: dict | None = None, news_summary: str = "") -> dict:
    """The JSON the explainer LLM sees. Shared by inference and training/build_explain_dataset.py so the
    fine-tuned model is trained on exactly what it gets in production. USD-only: every amount is USD."""
    p = {k: sig[k] for k in ("ticker", "direction", "horizon", "prob", "exp_excess_net", "ref_price", "stop_price",
                             "stop_pct") if k in sig}
    if "max_position_usd" in sig or "max_position_local" in sig:
        p["max_position"] = sig.get("max_position_usd", sig.get("max_position_local"))
    p["currency"] = "USD"
    p["risks"] = sig.get("risks", [])
    p["disclaimer"] = sig.get("disclaimer", "")
    p["prob_pct"] = round(sig["prob"] * 100)
    p["exp_excess_pct"] = round(sig["exp_excess_net"] * 100, 2)
    p["drivers"] = [{"name": label(f, lang), "direction": "+" if v > 0 else "-"} for f, v in sig.get("drivers", [])[:3]]
    p["client"] = profile_summary or {}
    p["recent_news"] = (news_summary or "")[:800]
    return p


def explain(sig: dict, lang: str, profile_summary: dict | None = None, news_summary: str = "",
            client: LLMClient | None = None) -> tuple[str, dict]:
    client = client or LLMClient()
    fallback = template(sig, lang)
    if not client.available:
        return fallback, {"source": "template"}
    payload = build_payload(sig, lang, profile_summary, news_summary)
    msgs = [{"role": "system", "content": EXPLAIN_SYSTEM.get(lang, EXPLAIN_SYSTEM["en"])},
            {"role": "user", "content": explain_user(payload)}]
    last = {}
    for attempt in range(get_config()["llm"]["max_retries"] + 1):
        try:
            text = client.chat(msgs, max_tokens=get_config()["llm"]["explain_max_tokens"])
        except Exception as e:  # noqa: BLE001
            log.warning("explain LLM error: %s", e)
            break
        text = _THINK.sub("", text).strip()
        res = guardrails.check(text, payload, lang)
        if res["ok"] and sig["disclaimer"][:20] in text:
            return text, {"source": "llm", "attempts": attempt + 1}
        last = res
        msgs = msgs + [{"role": "assistant", "content": text},
                       {"role": "user", "content": f"Rejected: {res}. Rewrite using only input numbers, no banned words, "
                                                   "and end with the exact disclaimer."}]
    return fallback, {"source": "template", "rejected": last}

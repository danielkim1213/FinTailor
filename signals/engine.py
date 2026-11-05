"""Signal generation. Predictions are identical for everyone; the client profile decides WHO gets a
signal, HOW BIG, and HOW it is explained."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from ..config import cost, env, get_config
from ..db import connect, get_state, upsert
from ..llm.explainer import explain
from ..utils import log, stable_id, utcnow
from .ledger import customer_state
from .profiles import CustomerProfile, load_customers


def tier_for(score: int) -> tuple[str, dict]:
    for name, t in get_config()["signals"]["tiers"].items():
        if t["risk_min"] <= score <= t["risk_max"]:
            return name, t
    return "none", {"max_positions": 0}


def fx_rates(con) -> dict[str, float]:
    out = {"USD": 1.0}
    for cur, sids in (("KRW", ("YF_KRWX", "DEXKOUS")), ("CAD", ("YF_CADX", "DEXCAUS"))):
        v = None
        for sid in sids:
            v = v or con.execute("select value from macro where series_id = ? order by date desc limit 1", [sid]).fetchone()
        out[cur] = float(v[0]) if v else {"KRW": 1400.0, "CAD": 1.38}[cur]
    return out


def candidates(con, run_date: pd.Timestamp, panel_rows: pd.DataFrame) -> pd.DataFrame:
    """Model-level candidate list (before client filters)."""
    scfg = get_config()["signals"]
    pr = con.execute("""select * from predictions where run_date = ? and official""", [run_date.date()]).df()
    if pr.empty:
        return pr
    pr = pr[pr["horizon"].isin(scfg["signal_horizons"])]
    aux = panel_rows.set_index("ticker")[["raw_close", "atr_pct_raw", "dvol20", "is_etf_flag", "days_to_earn_raw", "sector"]]
    pr = pr.join(aux, on="ticker")
    thr = scfg["prob_threshold"]
    long_ = pr[(pr["prob"] >= thr) & (pr["pred_excess"] > scfg["min_edge"])].assign(direction="long")
    short = pr[(pr["prob"] <= 1 - thr) & (pr["pred_excess"] < -scfg["min_edge"] - 2 * cost()) &
               (pr["is_etf_flag"] == False)].assign(direction="short")  # noqa: E712
    c = pd.concat([long_, short], ignore_index=True)
    if c.empty:
        return c
    c["conf"] = np.where(c["direction"] == "long", c["prob"], 1 - c["prob"])
    c = c.sort_values(["conf", "horizon"], ascending=[False, True]).drop_duplicates(["ticker", "direction"])
    # earnings inside the holding window (calendar approx: h trading days ~ h*7/5 days)
    win_days = c["horizon"] * 7 / 5 + 1
    c["earn_in_window"] = c["days_to_earn_raw"] <= win_days
    c["earn_unknown"] = c["days_to_earn_raw"].isna() & ~c["is_etf_flag"].astype(bool)
    if c["earn_unknown"].any() and scfg.get("check_earnings_online", True) and not env("FINTAILOR_OFFLINE"):
        try:
            from ..data.earnings import next_earnings_yf
            nxt = next_earnings_yf(c.loc[c["earn_unknown"], "ticker"].head(40).tolist())
            for t, d in nxt.items():
                m = c["ticker"] == t
                if d is not None:
                    days = (pd.Timestamp(d).tz_localize(None) - run_date).days
                    c.loc[m, "earn_in_window"] = days <= win_days[m]
                    c.loc[m, "earn_unknown"] = False
        except Exception as e:  # noqa: BLE001
            log.warning("online earnings check failed: %s", e)
    ev = con.execute("select date, event from events where date > ? and date <= ?",
                     [run_date.date(), (run_date + pd.Timedelta(days=45)).date()]).df()
    ev["date"] = pd.to_datetime(ev["date"])
    c["macro_events"] = [", ".join(f"{r.event} {r.date.date()}" for r in ev.itertuples()
                                   if r.event != "OPEX" and (r.date - run_date).days <= wd)
                         for wd in win_days]
    mega = set(panel_rows[~panel_rows["is_etf_flag"].astype(bool)].nlargest(get_config()["universe"]["megacap_top_n"], "dvol20")["ticker"])
    c["megacap"] = c["ticker"].isin(mega)
    return c


def _risks(row, lang: str) -> list[str]:
    r = []
    if row["earn_in_window"]:
        r.append("earnings inside holding period" if lang == "en" else "보유 기간 중 실적 발표")
    if row["macro_events"]:
        r.append(("scheduled events: " if lang == "en" else "예정 일정: ") + row["macro_events"])
    if row["atr_pct_raw"] and row["atr_pct_raw"] > 0.04:
        r.append("high volatility" if lang == "en" else "높은 변동성")
    r.append("overnight gap risk" if lang == "en" else "장 마감 후 갭 위험")
    return r


def build_for_customer(c: CustomerProfile, cand: pd.DataFrame, run_date: pd.Timestamp, fx: dict, state: dict,
                       model_version: str, data_asof: str, explain_fn=explain) -> list[dict]:
    scfg = get_config()["signals"]
    lang = c.language
    pack = c.pack
    base = {"run_date": run_date.date(), "customer_id": c.customer_id, "created_at": utcnow()}

    def status_row(status: str, note: str) -> list[dict]:
        return [{**base, "signal_id": stable_id(run_date, c.customer_id, status), "ticker": None, "direction": None,
                 "horizon": None, "status": status, "explanation": note,
                 "payload": json.dumps({"risk_score": c.risk_score, "note": note})}]

    tname, tier = tier_for(c.risk_score)
    edu = (not c.has_emergency_fund or c.has_high_interest_debt or c.experience in scfg["education_experience"]
           or tier.get("max_positions", 0) == 0)
    if edu:
        msg = {"en": "No trade signals: we send education and market commentary until your profile is suitable "
                     "(emergency fund, no high-interest debt, some experience, risk score 5+).",
               "ko": "현재 프로필에서는 매매 신호 대신 교육 자료와 시장 해설만 제공합니다 (비상금 확보, 고금리 부채 없음, "
                     "투자 경험, 위험 점수 5 이상 필요)."}[lang]
        return status_row("education", msg)
    limit = min(tier["weekly_loss_limit"], c.weekly_loss_limit_pct or 1.0)
    if state["weekly_pnl_pct"] <= -limit:
        return status_row("blocked_loss_limit", {"en": "Weekly loss limit reached; no new signals this week.",
                                                 "ko": "주간 손실 한도에 도달해 이번 주에는 새 신호를 보내지 않습니다."}[lang])
    slots = tier["max_positions"] - state["open_positions"]
    if slots <= 0 or cand.empty:
        return status_row("no_signal", {"en": "No new signals today.", "ko": "오늘은 새 신호가 없습니다."}[lang])
    d = cand[~cand["ticker"].isin(state["open_tickers"])]
    if not c.allow_short:
        d = d[d["direction"] == "long"]
    if tier.get("universe") == "etf_megacap":
        d = d[d["is_etf_flag"].astype(bool) | d["megacap"]]
        d = d[d["conf"] >= scfg["prob_threshold"] + scfg["high_conf_extra"]]
    if scfg["exclude_earnings_in_window"]:
        d = d[~d["earn_in_window"].fillna(False).astype(bool) & ~d["earn_unknown"].astype(bool)]
    d = d.head(slots)
    if d.empty:
        return status_row("no_signal", {"en": "No signals passed your filters today.",
                                        "ko": "오늘은 고객님 기준을 통과한 신호가 없습니다."}[lang])
    rate = fx[pack["currency"]]
    capital_usd = c.investable_amount / rate
    out = []
    for r in d.to_dict("records"):
        stop_pct = float(np.clip(scfg["atr_stop_mult"] * (r["atr_pct_raw"] or 0.02), scfg["min_stop_pct"], scfg["max_stop_pct"]))
        risk_usd = capital_usd * tier["risk_per_trade"]
        pos_usd = min(risk_usd / stop_pct, capital_usd * tier["max_position_pct"])
        ref = float(r["raw_close"])
        sign = 1 if r["direction"] == "long" else -1
        sig = {
            "ticker": r["ticker"], "direction": r["direction"], "entry": "next_open", "horizon": int(r["horizon"]),
            "prob": round(float(r["conf"]), 4), "exp_excess_net": round(float(r["pred_excess"]) * sign, 4),
            "ref_price": round(ref, 2), "stop_pct": round(stop_pct, 4), "stop_price": round(ref * (1 - sign * stop_pct), 2),
            "max_position_usd": round(pos_usd, 2), "max_position_local": int(round(pos_usd * rate)),
            "currency": pack["currency"], "shares_cap": int(math.floor(pos_usd / ref)) if ref > 0 else 0,
            "drivers": json.loads(r["drivers"])[:3] if r.get("drivers") else [], "risks": _risks(r, lang),
            "sector": str(r["sector"]), "data_asof": data_asof, "model_version": model_version,
            "tier": tname, "risk_score": c.risk_score, "execution_note": pack["execution_note"][lang],
            "tax_note": pack["tax_notes"][lang], "disclaimer": pack["disclaimer"][lang],
        }
        text, meta = explain_fn(sig, lang, {"country": c.country, "age": c.age, "experience": c.experience,
                                            "risk_score": c.risk_score})
        sig["explain_meta"] = meta
        out.append({**base, "signal_id": stable_id(run_date, c.customer_id, r["ticker"], r["direction"]),
                    "ticker": r["ticker"], "direction": r["direction"], "horizon": int(r["horizon"]),
                    "payload": json.dumps(sig, ensure_ascii=False, default=str), "explanation": text, "status": "sent"})
    return out


def generate_signals(run_date: pd.Timestamp, panel_rows: pd.DataFrame, customers: list[CustomerProfile] | None = None) -> pd.DataFrame:
    from ..models.registry import registry
    with connect(read_only=True) as con:
        halted = get_state(con, "kill_switch", {}).get("halted", False)
        customers = customers if customers is not None else load_customers(con)
        cand = candidates(con, run_date, panel_rows)
        fx = fx_rates(con)
        states = {c.customer_id: customer_state(con, c.customer_id, run_date, c.investable_amount / fx[c.pack["currency"]])
                  for c in customers}
    ver = registry().get("champion") or "?"
    rows: list[dict] = []
    for c in customers:
        if halted:
            rows.append({"run_date": run_date.date(), "customer_id": c.customer_id, "created_at": utcnow(),
                         "signal_id": stable_id(run_date, c.customer_id, "halted"), "ticker": None, "direction": None,
                         "horizon": None, "status": "halted", "explanation": "kill switch active",
                         "payload": "{}"})
            continue
        rows += build_for_customer(c, cand, run_date, fx, states[c.customer_id], ver, str(run_date.date()))
    df = pd.DataFrame(rows)
    with connect() as con:
        upsert(con, "signals", df)
    log.info("signals %s: %s", run_date.date(), df["status"].value_counts().to_dict() if len(df) else {})
    return df

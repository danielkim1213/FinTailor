"""Customer profile schema, risk scoring and country packs (US / CA; KR disabled for now).

NOTE: the risk-score weights and tier rules are placeholders. Have a licensed professional
review them before real clients use the system.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..db import connect, upsert
from ..utils import utcnow

COUNTRY_PACKS = {
    "US": {
        "currency": "USD", "fx_series": None, "language": "en", "timezone": "America/New_York",
        "execution_note": {"en": "Orders execute at the regular US open (9:30 ET)."},
                           # KR disabled: "ko": "미국 정규장 개장(동부 9:30)에 체결을 가정합니다."
        "tax_notes": {"en": "Short-term gains (held under a year) are taxed as ordinary income; watch the wash-sale rule."},
                      # KR disabled: "ko": "1년 미만 보유 차익은 일반 소득으로 과세되며 워시세일 규정에 유의하세요."
        "disclaimer": {"en": "This is model output for information only, not personalized investment advice. Past "
                             "performance does not predict future results and you can lose money."},
                       # KR disabled: "ko": "본 내용은 모델 출력 정보이며 개인 맞춤형 투자 권유가 아닙니다. 과거 성과는 미래를 보장하지 않으며 손실이 발생할 수 있습니다."
        "account_types": ["taxable", "ira", "roth_ira"],
    },
    "CA": {
        # USD-only: all amounts (prices, investable amount, position caps) are USD for every client.
        # was: "currency": "CAD", "fx_series": "DEXCAUS", note "...Consider FX conversion costs (CAD to USD)."
        "currency": "USD", "fx_series": None, "language": "en", "timezone": "America/Toronto",
        "execution_note": {"en": "Orders execute at the US open (9:30 ET). All amounts are in USD."},
                           # KR disabled: "ko": "미국 개장(동부 9:30) 체결 가정. CAD→USD 환전 비용을 고려하세요."
        "tax_notes": {"en": "US dividends: 15% withholding applies in TFSAs (not recoverable) but generally not in RRSPs. "
                            "Frequent trading in a TFSA can be treated as business income."},
                      # KR disabled: "ko": "TFSA 내 미국 배당은 원천징수가 적용되며, RRSP는 일반적으로 면제됩니다. 잦은 매매는 사업소득으로 볼 수 있습니다."
        "disclaimer": {"en": "This is model output for information only, not personalized investment advice. Past "
                             "performance does not predict future results and you can lose money."},
                       # KR disabled: "ko": "본 내용은 모델 출력 정보이며 개인 맞춤형 투자 권유가 아닙니다. 과거 성과는 미래를 보장하지 않으며 손실이 발생할 수 있습니다."
        "account_types": ["non_registered", "tfsa", "rrsp"],
    },
    # --- KR pack disabled for now (English-first). Re-enable together with the ko prompt/guardrails.
    # "KR": {
    #     "currency": "KRW", "fx_series": "DEXKOUS", "language": "ko", "timezone": "Asia/Seoul",
    #     "execution_note": {"en": "Orders execute at the US open (23:30 KST, 22:30 during US daylight saving time)."},
    #                        # KR disabled: "ko": "미국 정규장 개장 시각(한국시간 23:30, 미국 서머타임 기간 22:30)에 체결을 가정합니다."
    #     "tax_notes": {"en": "Overseas stock gains above the annual basic deduction are taxed (capital gains tax filed in May). "
    #                         "FX spreads add to trading cost."},
    #                   # KR disabled: "ko": "해외주식 양도차익은 연간 기본공제를 넘는 금액에 대해 과세되며 다음 해 5월에 신고합니다. 환전 스프레드도 비용입니다."
    #     "disclaimer": {"en": "This is model output for information only, not personalized investment advice. Past "
    #                          "performance does not predict future results and you can lose money."},
    #                    # KR disabled: "ko": "본 내용은 모델 출력 정보이며 개인 맞춤형 투자 권유가 아닙니다. 과거 성과는 미래를 보장하지 않으며 원금 손실이 발생할 수 있습니다."
    #     "account_types": ["general"],
    # },
}


class Questionnaire(BaseModel):
    horizon_years: int = Field(3, ge=0, le=40)           # how long the money can stay invested
    loss_tolerance: int = Field(3, ge=1, le=5)            # 1: sell at -5%, 5: hold through -30%
    income_stability: int = Field(3, ge=1, le=5)
    investment_share_of_assets: int = Field(3, ge=1, le=5)  # 1: >50% of assets, 5: <10%
    knowledge: int = Field(3, ge=1, le=5)


def compute_risk_score(age: int, experience: str, q: Questionnaire) -> int:
    """1..10. Placeholder weights - to be calibrated with an expert."""
    s = 0.0
    s += 2.0 if age < 35 else 1.5 if age < 50 else 1.0 if age < 65 else 0.3
    s += {"none": 0, "beginner": 0.5, "intermediate": 1.5, "advanced": 2.0}[experience]
    s += (q.loss_tolerance - 1) / 4 * 2.5
    s += min(q.horizon_years, 10) / 10 * 1.0
    s += (q.income_stability - 1) / 4 * 1.0
    s += (q.investment_share_of_assets - 1) / 4 * 1.0
    s += (q.knowledge - 1) / 4 * 0.5
    return int(max(1, min(10, round(s))))


class CustomerProfile(BaseModel):
    customer_id: str
    name: str = ""
    country: Literal["US", "CA"]  # "KR" disabled (English-first)
    language: Literal["en"] | None = None  # "ko" disabled
    age: int = Field(ge=18, le=110)
    experience: Literal["none", "beginner", "intermediate", "advanced"]
    investable_amount: float = Field(gt=0)        # USD (all clients, USD-only)
    has_emergency_fund: bool
    has_high_interest_debt: bool
    weekly_loss_limit_pct: float | None = None   # client's own limit (the stricter of this and tier wins)
    allow_short: bool = False
    account_type: str | None = None
    questionnaire: Questionnaire = Field(default_factory=Questionnaire)
    risk_score: int | None = Field(None, ge=1, le=10)

    @model_validator(mode="after")
    def _fill(self):
        if self.language is None:
            self.language = COUNTRY_PACKS[self.country]["language"]
        if self.risk_score is None:
            self.risk_score = compute_risk_score(self.age, self.experience, self.questionnaire)
        return self

    @property
    def pack(self) -> dict:
        return COUNTRY_PACKS[self.country]


def save_customer(p: CustomerProfile) -> None:
    import pandas as pd
    with connect() as con:
        upsert(con, "customers", pd.DataFrame([{"customer_id": p.customer_id, "profile": p.model_dump_json(),
                                                "updated_at": utcnow()}]))


def load_customers(con=None) -> list[CustomerProfile]:
    if con is None:
        with connect(read_only=True) as c:
            return load_customers(c)
    rows = con.execute("select profile from customers").fetchall()
    out = []
    for (raw,) in rows:
        d = json.loads(raw)
        if d.get("country") not in COUNTRY_PACKS:  # e.g. stored KR customers while KR is disabled
            continue
        if d.get("language") not in (None, "en"):
            d["language"] = None
        out.append(CustomerProfile.model_validate(d))
    return out


SAMPLE_CUSTOMERS = [
    CustomerProfile(customer_id="us_aggr", name="US aggressive", country="US", age=31, experience="advanced",
                    investable_amount=50000, has_emergency_fund=True, has_high_interest_debt=False,
                    questionnaire=Questionnaire(horizon_years=10, loss_tolerance=5, income_stability=4,
                                                investment_share_of_assets=4, knowledge=5)),
    CustomerProfile(customer_id="ca_mid", name="CA moderate", country="CA", age=45, experience="intermediate",
                    investable_amount=80000, has_emergency_fund=True, has_high_interest_debt=False, account_type="tfsa",
                    questionnaire=Questionnaire(horizon_years=5, loss_tolerance=3, income_stability=4,
                                                investment_share_of_assets=3, knowledge=3)),
    # CustomerProfile(customer_id="kr_aggr", name="KR 공격형", country="KR", age=29, experience="advanced",
    #                 investable_amount=30000000, has_emergency_fund=True, has_high_interest_debt=False,
    #                 questionnaire=Questionnaire(horizon_years=10, loss_tolerance=5, income_stability=4,
    #                                             investment_share_of_assets=4, knowledge=4)),
    # CustomerProfile(customer_id="kr_new", name="KR 초보", country="KR", age=24, experience="beginner",
    #                 investable_amount=5000000, has_emergency_fund=False, has_high_interest_debt=False),
]

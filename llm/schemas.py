"""Structured outputs. These are also the fine-tuning targets (FinTailor-News)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

SECTORS = ["technology", "financials", "health_care", "consumer_discretionary", "consumer_staples", "energy",
           "industrials", "materials", "utilities", "real_estate", "communication"]

EVENT_TYPES = ["earnings", "guidance", "m_and_a", "analyst_rating", "product", "legal_regulatory", "management",
               "capital_return", "financing", "macro_data", "central_bank", "fiscal_policy", "trade_policy",
               "military_conflict", "geopolitical", "sanctions", "commodity_supply", "natural_disaster",
               "pandemic_health", "labor", "cyber", "market_commentary", "other"]


class NewsScore(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    event_type: str = "other"
    region: str = "US"
    severity: int = Field(2, ge=0, le=5)
    novelty: Literal["new", "follow_up", "repeat"] = "new"
    sentiment: float = Field(0.0, ge=-1, le=1)
    sector_impact: dict[str, Literal["+", "-", "0"]] = Field(default_factory=dict)
    expected_horizon: Literal["intraday", "days", "days_to_weeks", "weeks_to_months", "long_term"] = "days"
    confidence: float = Field(0.5, ge=0, le=1)
    direction_5d: int = Field(0, ge=-1, le=1)   # learned from outcomes during fine-tuning

    @field_validator("event_type")
    @classmethod
    def _et(cls, v: str) -> str:
        return v if v in EVENT_TYPES else "other"

    @field_validator("sector_impact")
    @classmethod
    def _si(cls, v: dict) -> dict:
        return {k: s for k, s in v.items() if k in SECTORS}


class NewsBatch(BaseModel):
    items: list[NewsScore]


class SectorView(BaseModel):
    scores: dict[str, float]          # sector -> -2..+2 (5-day outlook vs market)
    rationale: str = ""

    @field_validator("scores")
    @classmethod
    def _clip(cls, v: dict) -> dict:
        return {k: max(-2.0, min(2.0, float(s))) for k, s in v.items() if k in SECTORS}


class Claim(BaseModel):
    type: Literal["market_move", "sector_move", "earnings", "news_misread", "news_correct", "other"]
    direction: Literal["up", "down", "none"] = "none"
    detail: str = ""


class Review(BaseModel):
    category: Literal["correct", "misread", "market_opposite", "noise", "data_issue"]
    summary: str
    claims: list[Claim] = Field(default_factory=list)
    feature_idea: str = ""
    misread_news_ids: list[str] = Field(default_factory=list)

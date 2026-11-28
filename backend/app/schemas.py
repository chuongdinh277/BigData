"""Hợp đồng dữ liệu vào/ra của API (Pydantic). Khớp contracts/api_contract.md."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Task = Literal["p1", "p2", "p3"]
ReviewStatus = Literal["PENDING", "IN_REVIEW", "REVIEWED", "ESCALATED"]
Decision = Literal["AGREE", "DISAGREE", "NEED_INFO"]


class PredictOptions(BaseModel):
    explain: bool = True
    top_k: int = Field(10, ge=1, le=30)


class PredictRequest(BaseModel):
    case_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.\-]+$")
    features: dict[str, float | int | str | None]
    options: PredictOptions = PredictOptions()


class ValidateRequest(BaseModel):
    features: dict[str, float | int | str | None]


class ReviewPatch(BaseModel):
    status: ReviewStatus
    reviewer: str = Field(..., min_length=1, max_length=64)
    decision: Decision | None = None
    note: str | None = Field(None, max_length=2000)


class AuditOut(BaseModel):
    id: int
    old_status: str | None
    new_status: str
    actor: str
    decision: str | None
    note: str | None
    at: datetime


class ReviewOut(BaseModel):
    id: int
    prediction_id: int
    case_id: str
    task: str
    risk_level: str
    calibrated_probability: float
    trigger_reasons: list[str]
    priority: int
    status: str
    reviewer: str | None
    decision: str | None
    note: str | None
    created_at: datetime
    updated_at: datetime
    prediction: dict[str, Any] | None = None
    audit: list[AuditOut] = []


class PredictionOut(BaseModel):
    id: int
    case_id: str
    task: str
    model_version: str
    calibrated_probability: float
    risk_level: str
    review_flag: bool
    created_at: datetime

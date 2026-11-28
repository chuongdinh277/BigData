"""Kết nối CSDL ứng dụng (PostgreSQL) và 3 bảng: dự đoán, hàng đợi review, nhật ký review.

Chỉ lưu ca demo/giả lập đã khử định danh. KHÔNG lưu dữ liệu MIMIC thật (DUA PhysioNet).
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .config import settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[str] = mapped_column(String(64), index=True)
    task: Mapped[str] = mapped_column(String(4), index=True)
    model_version: Mapped[str] = mapped_column(String(128))
    features_json: Mapped[dict] = mapped_column(JSON)
    probability: Mapped[float] = mapped_column(Float)
    calibrated_probability: Mapped[float] = mapped_column(Float)
    uncertainty_std: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(8), index=True)
    review_flag: Mapped[int] = mapped_column(Integer, default=0)
    result_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    review: Mapped["Review | None"] = relationship(back_populates="prediction", uselist=False)


class Review(Base):
    __tablename__ = "reviews"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), unique=True)
    trigger_reasons: Mapped[list] = mapped_column(JSON)
    priority: Mapped[int] = mapped_column(Integer, index=True)
    status: Mapped[str] = mapped_column(String(16), default="PENDING", index=True)
    reviewer: Mapped[str | None] = mapped_column(String(64), nullable=True)
    decision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    prediction: Mapped[Prediction] = relationship(back_populates="review")
    events: Mapped[list["ReviewAudit"]] = relationship(back_populates="review", order_by="ReviewAudit.id")


class ReviewAudit(Base):
    """Nhật ký bất biến: mỗi lần đổi trạng thái thêm MỘT dòng, không bao giờ sửa/xóa."""
    __tablename__ = "review_audit"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    review_id: Mapped[int] = mapped_column(ForeignKey("reviews.id"), index=True)
    old_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    new_status: Mapped[str] = mapped_column(String(16))
    actor: Mapped[str] = mapped_column(String(64))
    decision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    review: Mapped[Review] = relationship(back_populates="events")


_kw = {"connect_args": {"check_same_thread": False}} if settings.database_url.startswith("sqlite") else {"pool_pre_ping": True}
engine = create_engine(settings.database_url, **_kw)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    if settings.database_url.startswith("sqlite"):
        from pathlib import Path
        Path(settings.database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)


def get_session():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()

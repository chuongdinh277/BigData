"""Hàng đợi xem xét thủ công: ca nguy cơ cao hoặc độ tin cậy thấp.

Vòng đời: PENDING → IN_REVIEW → REVIEWED | ESCALATED. Hệ thống không tự đóng ca.
Mỗi lần đổi trạng thái thêm một dòng nhật ký `review_audit` (ai, lúc nào, quyết định, ghi chú).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import Prediction, Review, ReviewAudit, get_session
from ..schemas import AuditOut, ReviewOut, ReviewPatch

router = APIRouter(tags=["xem xét thủ công"])

TRANSITIONS = {"PENDING": {"IN_REVIEW"}, "IN_REVIEW": {"REVIEWED", "ESCALATED"}, "REVIEWED": set(), "ESCALATED": set()}


def _out(rv: Review, detail: bool = False) -> ReviewOut:
    p = rv.prediction
    return ReviewOut(id=rv.id, prediction_id=rv.prediction_id, case_id=p.case_id, task=p.task, risk_level=p.risk_level,
                     calibrated_probability=p.calibrated_probability, trigger_reasons=rv.trigger_reasons,
                     priority=rv.priority, status=rv.status, reviewer=rv.reviewer, decision=rv.decision, note=rv.note,
                     created_at=rv.created_at, updated_at=rv.updated_at,
                     prediction=p.result_json | {"features": p.features_json} if detail else None,
                     audit=[AuditOut(id=e.id, old_status=e.old_status, new_status=e.new_status, actor=e.actor,
                                     decision=e.decision, note=e.note, at=e.at) for e in rv.events] if detail else [])


@router.get("/reviews", response_model=list[ReviewOut])
def list_reviews(status: str | None = None, task: str | None = None, limit: int = Query(100, le=500),
                 offset: int = 0, db: Session = Depends(get_session)):
    q = select(Review).join(Prediction)
    if status:
        q = q.where(Review.status == status)
    if task:
        q = q.where(Prediction.task == task)
    q = q.order_by(Review.priority, Review.created_at.desc()).limit(limit).offset(offset)
    return [_out(r) for r in db.scalars(q)]


@router.get("/reviews/stats")
def review_stats(db: Session = Depends(get_session)):
    rows = db.execute(select(Review.status, func.count()).group_by(Review.status)).all()
    pr = db.execute(select(Review.priority, func.count()).where(Review.status.in_(["PENDING", "IN_REVIEW"]))
                    .group_by(Review.priority)).all()
    return {"by_status": {s: n for s, n in rows}, "open_by_priority": {int(p): n for p, n in pr}}


@router.get("/reviews/{review_id}", response_model=ReviewOut)
def get_review(review_id: int, db: Session = Depends(get_session)):
    rv = db.get(Review, review_id)
    if rv is None:
        raise HTTPException(404, "Không có ca review này")
    return _out(rv, detail=True)


@router.patch("/reviews/{review_id}", response_model=ReviewOut)
def update_review(review_id: int, body: ReviewPatch, db: Session = Depends(get_session)):
    rv = db.get(Review, review_id)
    if rv is None:
        raise HTTPException(404, "Không có ca review này")
    if body.status not in TRANSITIONS[rv.status]:
        raise HTTPException(409, f"Không thể chuyển {rv.status} → {body.status}. "
                                 f"Hợp lệ: {sorted(TRANSITIONS[rv.status]) or 'không (ca đã đóng)'}")
    if body.status in ("REVIEWED", "ESCALATED") and not body.decision:
        raise HTTPException(422, "Cần ghi quyết định (AGREE | DISAGREE | NEED_INFO) khi đóng ca")
    db.add(ReviewAudit(review_id=rv.id, old_status=rv.status, new_status=body.status, actor=body.reviewer,
                       decision=body.decision, note=body.note))
    rv.status, rv.reviewer = body.status, body.reviewer
    if body.decision:
        rv.decision = body.decision
    if body.note:
        rv.note = body.note
    db.commit()
    db.refresh(rv)
    return _out(rv, detail=True)

"""Cổng QUẢN TRỊ: số liệu toàn hệ thống cho nhóm vận hành — dữ liệu & pipeline Spark, danh mục mô hình,
đánh giá, hàng đợi + nhật ký kiểm toán, cấu hình đang chạy. Chỉ trả số liệu TỔNG HỢP (không dòng bệnh nhân).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import REPO_ROOT, settings
from ..db import Prediction, Review, ReviewAudit, get_session
from ..registry import registry
from .predict import _clean

router = APIRouter(prefix="/admin", tags=["quản trị"])
RESULTS = REPO_ROOT / "results"


def _json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


@router.get("/data")
def data_overview():
    """Dữ liệu và pipeline: số liệu đã kiểm chứng + manifest các lần chạy pipeline + benchmark."""
    data_dir, bench = RESULTS / "data", RESULTS / "benchmark"
    runs = {p.name: _json(p) for p in sorted(data_dir.glob("p*_dataset_*.json"))}
    scaling = []
    if (bench / "scaling.csv").exists():
        with (bench / "scaling.csv").open(encoding="utf-8") as f:
            scaling = list(csv.DictReader(f))
    return _clean({"verified": _json(data_dir / "verified_v1.json"), "pipeline_runs": runs,
                   "benchmark": {"scaling_runs": scaling, "funnel": _json(bench / "funnel.json"),
                                 "pandas_full_load": _json(bench / "pandas_full_load.json")},
                   "layers": [
                       {"layer": "Bronze", "path": "s3://heartguard-mimic-group11/bronze/mimiciv/3.1/", "format": "csv.gz (nguyên bản)", "owner": "Hùng"},
                       {"layer": "Silver", "path": "s3://heartguard-mimic-group11/silver/mimiciv/3.1/", "format": "Parquet, ép kiểu, phân vùng", "owner": "Hùng"},
                       {"layer": "Processed", "path": "s3://heartguard-mimic-group11/processed/{cohort,labels}/", "format": "Parquet", "owner": "Hà"},
                       {"layer": "Gold", "path": "s3://heartguard-mimic-group11/gold/{p1,p2,p3}/<version>/", "format": "Parquet + _manifest.json", "owner": "Khoẻ / Chiến"},
                       {"layer": "Splits", "path": "s3://heartguard-mimic-group11/splits/<task>/v1/", "format": "JSON (chứa subject_id, không commit)", "owner": "Chưởng"},
                       {"layer": "Model artifacts", "path": "s3://heartguard-mimic-group11/model-artifacts/<task>/<version>/", "format": "joblib + JSON + SHA256SUMS", "owner": "Chưởng"},
                   ]})


@router.get("/models")
def models():
    """Danh mục gói mô hình: phiên bản, mô hình, chỉ số, gói nào đang phục vụ."""
    out = []
    for task_dir in sorted(p for p in settings.model_dir.glob("p*") if p.is_dir()):
        serving = registry.get(task_dir.name)
        for b in sorted(p for p in task_dir.iterdir() if (p / "SHA256SUMS").exists()):
            m = _json(b / "metrics.json") or {}
            ch = m.get("chosen")
            cv = (m.get("cv") or {}).get(ch, {})
            test = m.get("test") or {}
            tm = test.get("metrics", {}) if isinstance(test, dict) else {}
            out.append({"task": task_dir.name, "version": b.name, "synthetic": b.name.startswith("synthetic-"),
                        "serving": bool(serving and serving.model_version == b.name), "model": ch,
                        "calibration": m.get("calibration_method"), "cv_pr_auc": cv.get("pr_auc_mean"),
                        "cv_roc_auc": cv.get("roc_auc_mean"), "test_pr_auc": tm.get("pr_auc"),
                        "test_roc_auc": tm.get("roc_auc"), "test_brier": tm.get("brier"),
                        "thresholds": _json(b / "thresholds.json"),
                        "files": sorted(p.name for p in b.iterdir() if p.is_file()),
                        "size_mb": round(sum(p.stat().st_size for p in b.iterdir() if p.is_file()) / 1e6, 2)})
    return _clean(out)


@router.get("/system")
def system(db: Session = Depends(get_session)):
    """Sức khỏe và khối lượng: mô hình nạp, số dự đoán theo bài/mức nguy cơ, hàng đợi, cấu hình (không bí mật)."""
    by_task = db.execute(select(Prediction.task, Prediction.risk_level, func.count()).group_by(Prediction.task, Prediction.risk_level)).all()
    flagged = db.execute(select(Prediction.task, func.sum(Prediction.review_flag), func.count()).group_by(Prediction.task)).all()
    rv = db.execute(select(Review.status, func.count()).group_by(Review.status)).all()
    return {"models_loaded": {t: m.model_version for t, m in registry.models.items()}, "model_errors": registry.errors,
            "predictions": [{"task": t, "risk_level": r, "n": n} for t, r, n in by_task],
            "flag_rate": [{"task": t, "flagged": int(f or 0), "n": n, "rate": (f or 0) / n if n else None} for t, f, n in flagged],
            "reviews": {s: n for s, n in rv},
            "config": {"database": settings.database_url.split("@")[-1].split("://")[-1] if "@" in settings.database_url else "sqlite (dev)",
                       "model_source": settings.model_source, "model_dir": str(settings.model_dir),
                       "model_s3_prefix": settings.model_s3_prefix, "tasks": settings.tasks,
                       "verify_bundle_sha256": settings.verify_bundle}}


@router.get("/audit")
def audit(limit: int = Query(100, le=1000), db: Session = Depends(get_session)):
    """Nhật ký kiểm toán: mọi lần đổi trạng thái review (ai, lúc nào, quyết định)."""
    rows = db.execute(select(ReviewAudit, Review, Prediction).join(Review, ReviewAudit.review_id == Review.id)
                      .join(Prediction, Review.prediction_id == Prediction.id)
                      .order_by(ReviewAudit.at.desc()).limit(limit)).all()
    return [{"id": a.id, "at": a.at, "review_id": r.id, "case_id": p.case_id, "task": p.task, "old_status": a.old_status,
             "new_status": a.new_status, "actor": a.actor, "decision": a.decision, "note": a.note} for a, r, p in rows]

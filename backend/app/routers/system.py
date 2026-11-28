"""Trạng thái hệ thống, danh sách bài toán, lịch sử dự đoán, báo cáo đánh giá, kiểm tra drift theo lô."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from trustworthy_ai import DISCLAIMER

from ..config import TASK_INFO, settings
from ..db import Prediction, get_session
from ..registry import registry
from .predict import _clean
from ..schemas import PredictionOut, Task

router = APIRouter()

# Mỗi "kind" là một khóa trong results/trust/<task>/results.json (do ml.trust.run sinh ra)
REPORT_KEYS = {
    "summary": ["task_name", "data_version", "run_id", "git_commit", "real_data", "dev", "chosen_model", "chosen_by",
                "model_version", "n_features"],
    "metrics": ["cv", "dev_metrics", "test"],
    "baselines": ["baselines_dev", "learning_curve"],
    "calibration": ["calibration", "reliability_dev", "thresholds", "dca_dev", "dca_test", "operating_dev",
                    "operating_test"],
    "explain": ["shap", "explanation_drift"],
    "robustness": ["robustness_dev", "robustness_test"],
    "subgroup": ["subgroup_dev", "subgroup_test", "fairness_dev", "fairness_test"],
    "drift": ["drift", "temporal_validation", "coding_drift", "maintenance"],
    "review": ["review_dev", "review_test", "confidence_validation_dev", "confidence_validation_test",
               "risk_coverage_dev", "risk_coverage_test", "capacity_dev", "review_rules_summary"],
}


@router.get("/health", tags=["hệ thống"])
def health():
    return {"status": "ok", "models_loaded": {t: m.model_version for t, m in registry.models.items()},
            "model_errors": registry.errors, "disclaimer": DISCLAIMER}


@router.get("/tasks", tags=["hệ thống"])
def tasks():
    out = []
    for t in ("p1", "p2", "p3"):
        m = registry.get(t)
        out.append({"task": t, **TASK_INFO[t], "enabled": m is not None and t in settings.tasks,
                    "model_version": m.model_version if m else None, "thresholds": m.thresholds if m else None,
                    "error": registry.errors.get(t)})
    return out


@router.get("/predictions", response_model=list[PredictionOut], tags=["lịch sử"])
def predictions(task: str | None = None, limit: int = Query(50, le=500), offset: int = 0,
                db: Session = Depends(get_session)):
    q = select(Prediction)
    if task:
        q = q.where(Prediction.task == task)
    q = q.order_by(Prediction.created_at.desc()).limit(limit).offset(offset)
    return [PredictionOut(id=p.id, case_id=p.case_id, task=p.task, model_version=p.model_version,
                          calibrated_probability=p.calibrated_probability, risk_level=p.risk_level,
                          review_flag=bool(p.review_flag), created_at=p.created_at) for p in db.scalars(q)]


@router.get("/predictions/{prediction_id}", tags=["lịch sử"])
def prediction_detail(prediction_id: int, db: Session = Depends(get_session)):
    p = db.get(Prediction, prediction_id)
    if p is None:
        raise HTTPException(404, "Không có dự đoán này")
    return {"id": p.id, "case_id": p.case_id, "task": p.task, "features": p.features_json, "result": p.result_json,
            "review_id": p.review.id if p.review else None, "created_at": p.created_at}


def _results_dir(task: str):
    """Kết quả thật (results/trust/<task>) nếu có; nếu chưa chạy trên dữ liệu thật thì dùng bản demo
    (artifacts/demo_results, do tools/make_demo_bundles.py sinh) — UI ghi rõ "DỮ LIỆU GIẢ"."""
    primary = settings.results_dir / task
    demo = settings.results_dir.parents[1] / "artifacts" / "demo_results" / task
    return primary if (primary / "results.json").exists() or not (demo / "results.json").exists() else demo


def _results(task: str) -> dict:
    path = _results_dir(task) / "results.json"
    if not path.exists():
        raise HTTPException(404, f"Chưa có kết quả đánh giá cho {task}. Chạy: HG_TASK={task} python -m ml.trust.run")
    # results.json do json.dumps ghi có thể chứa NaN (chỉ số không xác định) → đổi thành null khi trả API
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/reports/{task}/{kind}", tags=["báo cáo"])
def report(task: Task, kind: str):
    if kind not in REPORT_KEYS:
        raise HTTPException(404, f"kind phải là một trong {sorted(REPORT_KEYS)}")
    R = _results(task)
    return _clean({"task": task, "kind": kind, **{k: R.get(k) for k in REPORT_KEYS[kind]}})


@router.get("/reports/{task}/figures/{name}", tags=["báo cáo"])
def figure(task: Task, name: str):
    if "/" in name or "\\" in name or not name.endswith(".png"):
        raise HTTPException(400, "Tên hình không hợp lệ")
    path = _results_dir(task) / "figures" / name
    if not path.exists():
        raise HTTPException(404, "Không có hình này")
    return FileResponse(path, media_type="image/png")


class DriftBatch(BaseModel):
    records: list[dict[str, float | int | str | None]] = Field(..., min_length=20, max_length=50000)


@router.post("/drift/{task}", tags=["drift"])
def drift_check(task: Task, body: DriftBatch):
    """Giám sát drift theo lô: so một lô ca mới với phân phối tham chiếu lưu trong gói mô hình
    (PSI từng đặc trưng + PSI điểm dự đoán). Không cần nhãn — đây là cảnh báo sớm."""
    m = registry.get(task)
    if m is None:
        raise HTTPException(503, f"Mô hình {task} chưa nạp")
    rep = m.drift_report(body.records)
    clean = lambda v: None if isinstance(v, float) and v != v else v  # noqa: E731
    rep["features"] = [{k: clean(v) for k, v in r.items()} for r in rep["features"]]
    return {"task": task, "model_version": m.model_version, **rep}

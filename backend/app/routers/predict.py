"""Dự đoán, giải thích, kiểm tra đầu vào, lược đồ đầu vào và ca demo."""
from __future__ import annotations

import json
import math

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from trustworthy_ai import DISCLAIMER
from trustworthy_ai.labels import describe

from ..config import REPO_ROOT, TASK_INFO, settings
from ..db import Prediction, Review, ReviewAudit, get_session
from ..demo_cases import demo_cases
from ..registry import registry
from ..schemas import PredictRequest, Task, ValidateRequest

router = APIRouter(tags=["dự đoán"])


def _model(task: str):
    if task not in settings.tasks:
        raise HTTPException(404, f"Bài toán {task} chưa bật")
    m = registry.get(task)
    if m is None:
        raise HTTPException(503, f"Mô hình {task} chưa nạp: {registry.errors.get(task, 'không rõ')}")
    return m


def _check_fields(model, features: dict) -> None:
    unknown = sorted(set(features) - set(model.schema["features"]))
    if unknown:
        raise HTTPException(422, {"message": "Trường không có trong lược đồ đầu vào", "unknown_fields": unknown[:20]})
    for k, v in features.items():
        if isinstance(v, float) and not math.isfinite(v):
            raise HTTPException(422, {"message": "Giá trị không hữu hạn", "field": k})


def _clean(o):
    """NaN/inf → None để JSON hợp lệ."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def _label_explanation(r: dict) -> dict:
    if "explanation" in r:
        for f in r["explanation"]["top_features"]:
            f.update(describe(f["feature"]))
    return r


@router.get("/schema/{task}")
def schema(task: Task):
    m = _model(task)
    contract = json.loads((REPO_ROOT / "contracts" / f"gold_schema_{task}.json").read_text(encoding="utf-8"))
    group_of = {}
    for g, spec in contract["feature_columns"].items():
        if isinstance(spec, list):
            group_of.update({i["name"]: g for i in spec})
        else:
            group_of.update({f"{n}{s}": g for n in spec["items"] for s in spec["suffixes"]})
            group_of.update({i["name"]: g for i in spec.get("extra", [])})
    ranges = m.schema.get("valid_ranges", {})
    return {"task": task, "model_version": m.model_version,
            "fields": [{"name": f, "dtype": m.schema["dtypes"].get(f, "float"), "group": group_of.get(f, "khác"),
                        "valid_range": ranges.get(f), **describe(f)} for f in m.schema["features"]]}


@router.get("/cases/{task}")
def cases(task: Task):
    _model(task)
    return demo_cases(task)


@router.post("/validate/{task}")
def validate(task: Task, body: ValidateRequest):
    m = _model(task)
    _check_fields(m, body.features)
    X = m.to_frame(body.features)
    return {"task": task, "model_version": m.model_version, "input_quality": _clean(m.input_quality(X)[0]),
            "disclaimer": DISCLAIMER}


@router.post("/explain/{task}")
def explain(task: Task, body: PredictRequest):
    m = _model(task)
    _check_fields(m, body.features)
    r = m.predict(body.features, explain=True, top_k=body.options.top_k)[0]
    return _clean({"task": task, "case_id": body.case_id, "model_version": r["model_version"],
                   "explanation": _label_explanation(r)["explanation"], "disclaimer": DISCLAIMER})


@router.post("/predict/{task}")
def predict(task: Task, body: PredictRequest, db: Session = Depends(get_session)):
    m = _model(task)
    _check_fields(m, body.features)
    r = _label_explanation(m.predict(body.features, explain=body.options.explain, top_k=body.options.top_k)[0])
    r = _clean(r)
    row = Prediction(case_id=body.case_id, task=task, model_version=r["model_version"], features_json=_clean(body.features),
                     probability=r["probability"], calibrated_probability=r["calibrated_probability"],
                     uncertainty_std=r["uncertainty_std"], risk_level=r["risk_level"],
                     review_flag=int(r["review"]["flag"]), result_json=r)
    db.add(row)
    db.flush()
    review_id = None
    if r["review"]["flag"]:
        rv = Review(prediction_id=row.id, trigger_reasons=r["review"]["reasons"], priority=r["review"]["priority"])
        db.add(rv)
        db.flush()
        db.add(ReviewAudit(review_id=rv.id, old_status=None, new_status="PENDING", actor="system",
                           note="Tự động gắn cờ: " + ", ".join(r["review"]["reasons"])))
        review_id = rv.id
    db.commit()
    r["review"]["review_id"] = review_id
    return {"task": task, "task_name": TASK_INFO[task]["name"], "case_id": body.case_id, "prediction_id": row.id, **r}

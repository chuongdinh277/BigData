"""Gói mô hình cuối của GĐ5 và hàm dự đoán một ca (dùng cho FastAPI ở GĐ6).

Cấu trúc thư mục gói (artifacts/models/p1/<model_version>/, đồng bộ lên
s3://heartguard-mimic-group11/model-artifacts/p1/<model_version>/):

    model.joblib            pipeline sklearn (tiền xử lý + mô hình) fit trên train+validation
    calibrator.joblib       bộ hiệu chỉnh học từ dự đoán out-of-fold
    fold_models.joblib      5 mô hình fold CV — đo độ bất đồng cho cờ LOW_CONFIDENCE
    explainer_background.joblib  (chỉ LR) trung bình đặc trưng sau tiền xử lý
    feature_schema.json     tên/kiểu cột đầu vào, khoảng hợp lệ
    thresholds.json         ngưỡng LOW/MEDIUM/HIGH
    review_rules.json       luật gắn cờ xem lại
    drift_reference.json    phân phối tham chiếu (biên bin + tỷ lệ) cho drift theo lô
    metrics.json            chỉ số CV / test (tổng hợp)
    model_card.md           mô tả mô hình, giới hạn
    SHA256SUMS              kiểm tra toàn vẹn khi nạp

Gói KHÔNG được commit vào Git (artifacts/models/ đã trong .gitignore).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import DISCLAIMER
from .calibration import risk_level
from .drift import DriftReference
from .explain import Explainer
from .review import flag_cases

LABS_KEYS = ("_first", "_min", "_max")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def save_bundle(out_dir: str | Path, *, model, calibrator, fold_models, feature_columns: list[str],
                dtypes: dict, thresholds: dict, rules: dict, drift_ref: DriftReference, metrics: dict,
                model_card: str, model_version: str, background: pd.DataFrame | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    joblib.dump(calibrator, out / "calibrator.joblib")
    joblib.dump(fold_models, out / "fold_models.joblib")
    if background is not None:
        joblib.dump(background, out / "explainer_background.joblib")
    write_json(out / "feature_schema.json", {"model_version": model_version, "features": feature_columns,
                                             "dtypes": dtypes, "valid_ranges": rules.get("valid_ranges", {})})
    write_json(out / "thresholds.json", thresholds)
    write_json(out / "review_rules.json", rules)
    write_json(out / "drift_reference.json", drift_ref.to_dict())
    write_json(out / "metrics.json", metrics)
    (out / "model_card.md").write_text(model_card, encoding="utf-8")
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{_sha256(p)}  {p.name}\n" for p in files), encoding="utf-8")
    return out


def upload_bundle(local_dir: str | Path, s3_uri: str) -> None:
    import boto3
    bucket, _, prefix = s3_uri.replace("s3a://", "s3://")[len("s3://"):].partition("/")
    s3 = boto3.client("s3")
    for p in Path(local_dir).iterdir():
        if p.is_file():
            s3.upload_file(str(p), bucket, f"{prefix.rstrip('/')}/{p.name}")


class HeartGuardModel:
    """Nạp gói một lần khi khởi động; predict() không bao giờ gọi fit()."""

    def __init__(self, bundle_dir: str | Path, verify: bool = True):
        d = Path(bundle_dir)
        if verify:
            for line in (d / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
                digest, name = line.split("  ", 1)
                if _sha256(d / name) != digest:
                    raise ValueError(f"Gói mô hình bị thay đổi: {name}")
        self.dir = d
        self.model = joblib.load(d / "model.joblib")
        self.calibrator = joblib.load(d / "calibrator.joblib")
        self.fold_models = joblib.load(d / "fold_models.joblib")
        self.schema = json.loads((d / "feature_schema.json").read_text(encoding="utf-8"))
        self.thresholds = json.loads((d / "thresholds.json").read_text(encoding="utf-8"))
        self.rules = json.loads((d / "review_rules.json").read_text(encoding="utf-8"))
        self.drift_ref = DriftReference(json.loads((d / "drift_reference.json").read_text(encoding="utf-8")))
        self.model_version = self.schema["model_version"]
        bg = d / "explainer_background.joblib"
        self.explainer = Explainer(self.model, joblib.load(bg) if bg.exists() else None)

    # ---------- chuẩn hóa đầu vào ----------
    def to_frame(self, records: list[dict] | dict) -> pd.DataFrame:
        records = [records] if isinstance(records, dict) else records
        feats = self.schema["features"]
        df = pd.DataFrame(records).reindex(columns=feats)  # cột không gửi → NaN
        # Cờ thiếu suy ra từ giá trị nếu người dùng không gửi (thiếu ≠ bình thường)
        for c in feats:
            if c.endswith("_missing") and c != "vitals_missing":
                base = c[: -len("_missing")]
                vals = [f"{base}{s}" for s in LABS_KEYS if f"{base}{s}" in df]
                user_sent = [r.get(c) is not None for r in records]
                derived = df[vals].isna().all(axis=1).astype(int) if vals else 1
                df[c] = np.where(user_sent, df[c], derived)
        if "vitals_missing" in feats:
            vit = [c for c in feats if c.startswith(("hr_", "sbp_", "dbp_", "mbp_", "rr_", "spo2_", "temp_"))]
            user_sent = [r.get("vitals_missing") is not None for r in records]
            df["vitals_missing"] = np.where(user_sent, df["vitals_missing"], df[vit].isna().all(axis=1).astype(int))
        for c, t in self.schema["dtypes"].items():
            if c in df and t != "object":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[feats]

    def input_quality(self, X: pd.DataFrame) -> list[dict]:
        out = []
        ranges = self.schema.get("valid_ranges", {})
        for _, row in X.iterrows():
            missing = [c for c in X.columns if not c.endswith("_missing") and pd.isna(row[c])]
            oor = [c for c, (lo, hi) in ranges.items() if c in row and pd.notna(row[c]) and not lo <= row[c] <= hi]
            warn = []
            for lab in self.rules["key_labs"]:
                if row.get(f"{lab}_missing", 0) == 1:
                    warn.append(f"Thiếu xét nghiệm {lab}: kết quả kém tin cậy hơn")
            if row.get("ntprobnp_missing", 0) == 1:
                warn.append("Thiếu NT-proBNP (thường gặp, 68% ca): mô hình dựa vào xét nghiệm khác")
            if row.get("vitals_missing", 0) == 1:
                warn.append("Không có sinh hiệu ICU trong 24h đầu")
            if oor:
                warn.append("Có giá trị ngoài khoảng sinh lý: kiểm tra lại đơn vị/nhập liệu")
            out.append({"missing_fields": missing, "out_of_range_fields": oor, "warnings": warn})
        return out

    # ---------- dự đoán ----------
    def predict_proba(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p_raw = self.model.predict_proba(X)[:, 1]
        p_cal = self.calibrator.transform(p_raw)
        folds = np.column_stack([self.calibrator.transform(m.predict_proba(X)[:, 1]) for m in self.fold_models])
        return p_raw, p_cal, folds.std(axis=1)

    def predict(self, records, explain: bool = True, top_k: int = 10) -> list[dict]:
        X = self.to_frame(records)
        p_raw, p_cal, std = self.predict_proba(X)
        # Giá trị ngoài khoảng: báo cờ nhưng mô hình vẫn nhận NULL để không dự đoán trên số sai
        quality = self.input_quality(X)
        X_model = X.copy()
        for i, q in enumerate(quality):
            for c in q["out_of_range_fields"]:
                X_model.iloc[i, X_model.columns.get_loc(c)] = np.nan
        if any(q["out_of_range_fields"] for q in quality):
            p_raw, p_cal, std = self.predict_proba(X_model)
        flags = flag_cases(X, p_cal, self.rules, std)
        levels = risk_level(p_cal, self.thresholds)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        results = []
        for i in range(len(X)):
            r = {"model_version": self.model_version, "prediction_time": now,
                 "probability": round(float(p_raw[i]), 5), "calibrated_probability": round(float(p_cal[i]), 5),
                 "uncertainty_std": round(float(std[i]), 5), "threshold": self.thresholds["medium_high"],
                 "risk_level": str(levels[i]),
                 "review": {"flag": bool(flags["flag"].iloc[i]), "reasons": flags["reasons"].iloc[i],
                            "priority": int(flags["priority"].iloc[i])},
                 "input_quality": quality[i], "disclaimer": DISCLAIMER}
            if explain:
                r["explanation"] = self.explainer.explain_row(X_model.iloc[[i]], top_k=top_k)
            results.append(r)
        return results

    def drift_report(self, records_or_frame) -> dict:
        X = records_or_frame if isinstance(records_or_frame, pd.DataFrame) else self.to_frame(records_or_frame)
        _, p_cal, _ = self.predict_proba(X)
        rep = self.drift_ref.compare(X, p_cal)
        rep["features"] = rep["features"].head(20).to_dict(orient="records")
        return rep

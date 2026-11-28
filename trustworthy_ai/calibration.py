"""Hiệu chỉnh xác suất và chọn ngưỡng.

Bộ hiệu chỉnh được học trên dự đoán OUT-OF-FOLD của CV theo bệnh nhân (mỗi dự đoán đến từ mô hình
không thấy dòng đó), nên không cần dùng tập test và không bị lạc quan như khi hiệu chỉnh trên train.
Phương pháp (none / platt / isotonic) được chọn bằng CV lồng trên chính các fold đó, tiêu chí Brier.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .evaluation import calibration_stats, discrimination, fit_logistic_1d, logit, threshold_stats

METHODS = ("none", "platt", "isotonic")


class Calibrator:
    def __init__(self, method: str = "isotonic"):
        if method not in METHODS:
            raise ValueError(f"method phải thuộc {METHODS}")
        self.method = method
        self.a_ = 0.0
        self.b_ = 1.0
        self.iso_ = None

    def fit(self, p, y):
        p = np.asarray(p, float)
        y = np.asarray(y, float)
        if self.method == "platt":
            self.a_, self.b_ = fit_logistic_1d(logit(p), y)
        elif self.method == "isotonic":
            self.iso_ = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip").fit(p, y)
        return self

    def transform(self, p):
        p = np.asarray(p, float)
        if self.method == "platt":
            return 1.0 / (1.0 + np.exp(-(self.a_ + self.b_ * logit(p))))
        if self.method == "isotonic":
            return self.iso_.predict(p)
        return p

    def to_dict(self) -> dict:
        d = {"method": self.method, "a": self.a_, "b": self.b_}
        if self.iso_ is not None:
            d["x"] = self.iso_.X_thresholds_.tolist()
            d["y"] = self.iso_.y_thresholds_.tolist()
        return d


def crossfit_calibrate(p_oof, y, folds, method: str) -> np.ndarray:
    """Xác suất hiệu chỉnh 'trung thực': fold j dùng bộ hiệu chỉnh học từ các fold khác."""
    p_oof, y, folds = np.asarray(p_oof, float), np.asarray(y).astype(int), np.asarray(folds)
    out = np.empty_like(p_oof)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        out[te] = Calibrator(method).fit(p_oof[tr], y[tr]).transform(p_oof[te])
    return out


def select_calibrator(p_oof, y, folds, methods=METHODS) -> tuple[str, dict]:
    """Với mỗi phương pháp: học trên các fold khác, áp dụng cho fold j → ghép lại → Brier/ECE."""
    p_oof, y, folds = np.asarray(p_oof, float), np.asarray(y).astype(int), np.asarray(folds)
    report = {}
    for m in methods:
        out = crossfit_calibrate(p_oof, y, folds, m)
        d = discrimination(y, out)
        c = calibration_stats(y, out)
        report[m] = {"brier": d["brier"], "pr_auc": d["pr_auc"], "roc_auc": d["roc_auc"], "ece": c["ece"],
                     "calibration_slope": c["calibration_slope"], "oe_ratio": c["oe_ratio"]}
    best = min(report, key=lambda m: (round(report[m]["brier"], 5), report[m]["ece"]))
    return best, report


def f1_threshold(y, p) -> float:
    from ml.common.metrics import choose_threshold
    return choose_threshold(y, p)


def choose_thresholds(y, p_cal, target_recall: float = 0.80) -> dict:
    """HIGH: ngưỡng tối đa F1 (quy ước chung của nhóm). MEDIUM: ngưỡng lớn nhất vẫn bắt được
    ≥ target_recall ca tử vong. Mọi ngưỡng tính trên xác suất ĐÃ hiệu chỉnh, dự đoán out-of-fold."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p_cal, float)
    t_high = f1_threshold(y, p)
    order = np.argsort(-p, kind="stable")
    recall_curve = np.cumsum(y[order]) / max(y.sum(), 1)
    hit = np.nonzero(recall_curve >= target_recall)[0]
    # ngưỡng lớn nhất vẫn đạt recall mục tiêu (p >= t gồm cả các giá trị bằng nhau → recall chỉ tăng)
    t_med = float(p[order][hit[0]]) if len(hit) else None
    if t_med is None or t_med >= t_high:
        t_med = float(t_high / 2)
    return {"low_medium": t_med, "medium_high": float(t_high), "target_recall_medium": target_recall,
            "at_medium": threshold_stats(y, p, t_med), "at_high": threshold_stats(y, p, t_high)}


def risk_level(p, thresholds: dict):
    p = np.asarray(p, float)
    return np.where(p >= thresholds["medium_high"], "HIGH",
                    np.where(p >= thresholds["low_medium"], "MEDIUM", "LOW"))

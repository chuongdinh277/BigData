"""Lợi ích lâm sàng và vận hành: Decision Curve Analysis, bảng điểm cắt, đường năng lực review.

- Decision Curve Analysis (Vickers & Elkin 2006): lợi ích ròng của việc "can thiệp khi p ≥ pt"
      NB(pt) = TP/n − FP/n · pt/(1 − pt)
  so với hai chiến lược mặc định: can thiệp tất cả, và không can thiệp ai (NB = 0).
- Bảng điểm cắt: tại mỗi ngưỡng, tỷ lệ cảnh báo, độ nhạy, PPV, số ca cần xem để bắt 1 ca (NNE = 1/PPV).
- Đường năng lực: nếu người duyệt chỉ xem được top x% ca theo nguy cơ, bắt được bao nhiêu % ca dương.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_DCA_RANGE = {"p1": (0.01, 0.40), "p2": (0.05, 0.60), "p3": (0.02, 0.50)}


def net_benefit(y, p, pt: float) -> float:
    y = np.asarray(y).astype(int)
    pred = np.asarray(p) >= pt
    n = len(y)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    return tp / n - fp / n * pt / (1 - pt)


def decision_curve(y, p, lo: float = 0.01, hi: float = 0.40, step: float = 0.01) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    prev = float(y.mean())
    rows = []
    for pt in np.round(np.arange(lo, hi + 1e-9, step), 4):
        rows.append({"threshold": float(pt), "net_benefit_model": net_benefit(y, p, pt),
                     "net_benefit_treat_all": prev - (1 - prev) * pt / (1 - pt), "net_benefit_treat_none": 0.0})
    df = pd.DataFrame(rows)
    df["model_beats_defaults"] = df["net_benefit_model"] > df[["net_benefit_treat_all", "net_benefit_treat_none"]].max(axis=1)
    return df


def dca_summary(curve: pd.DataFrame) -> dict:
    """Khoảng ngưỡng mà mô hình có lợi ích ròng cao hơn cả hai chiến lược mặc định."""
    win = curve[curve["model_beats_defaults"]]
    return {"useful_threshold_min": float(win["threshold"].min()) if len(win) else None,
            "useful_threshold_max": float(win["threshold"].max()) if len(win) else None,
            "share_of_range_useful": float(curve["model_beats_defaults"].mean())}


def operating_table(y, p, thresholds=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    rows = []
    for t in thresholds:
        flag = p >= t
        tp = int((flag & (y == 1)).sum())
        n_flag = int(flag.sum())
        ppv = tp / n_flag if n_flag else float("nan")
        rows.append({"threshold": t, "alert_rate": float(flag.mean()), "sensitivity": tp / max(int(y.sum()), 1),
                     "ppv": ppv, "nne": 1 / ppv if ppv and ppv == ppv and ppv > 0 else float("nan")})
    return pd.DataFrame(rows)


def capacity_curve(y, priority_score, shares=(0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)) -> pd.DataFrame:
    """Review top x% ca theo điểm ưu tiên → % ca dương bắt được và tỷ lệ dương trong hàng đợi."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-np.asarray(priority_score, float), kind="stable")
    pos = max(int(y.sum()), 1)
    rows = []
    for s in shares:
        k = max(1, int(round(s * len(y))))
        top = y[order[:k]]
        rows.append({"review_share": s, "n_reviewed": k, "positives_captured_share": float(top.sum() / pos),
                     "positive_rate_in_queue": float(top.mean())})
    return pd.DataFrame(rows)

"""Metric bắt buộc của nhóm. Ngưỡng F1 chọn trên validation, không chọn trên test."""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_recall_curve, roc_auc_score


def choose_threshold(y_true, y_prob) -> float:
    """Ngưỡng làm F1 lớn nhất trên tập được đưa vào (dùng tập validation)."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)[:-1]
    return float(thr[int(np.argmax(f1))]) if len(thr) else 0.5


def evaluate(y_true, y_prob, threshold: float | None = None) -> dict:
    y_true, y_prob = np.asarray(y_true).astype(int), np.asarray(y_prob, dtype=float)
    if threshold is None:
        threshold = choose_threshold(y_true, y_prob)
    both = len(np.unique(y_true)) == 2
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if both else float("nan"),
        "positive_rate": float(y_true.mean()),
        "f1": float(f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)),
        "threshold": float(threshold),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }

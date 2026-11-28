"""Chỉ số đánh giá: phân biệt (ROC/PR), ngưỡng (F1, độ nhạy...), hiệu chỉnh (Brier, ECE, slope, O/E)
và khoảng tin cậy bootstrap theo bệnh nhân (dùng trọng số multinomial, không nhân bản dòng)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

EPS = 1e-6


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_logistic_1d(x, y, *, offset=None, fit_slope=True, weight=None, iters=100):
    """Newton–Raphson cho y ~ a + b·x (+offset). Trả (a, b). Dùng cho Platt và calibration slope."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.ones_like(y) if weight is None else np.asarray(weight, float)
    off = np.zeros_like(y) if offset is None else np.asarray(offset, float)
    X = np.column_stack([np.ones_like(x), x]) if fit_slope else np.ones((len(x), 1))
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = _sigmoid(X @ beta + off)
        grad = X.T @ (w * (y - mu))
        H = X.T @ (X * (w * mu * (1 - mu))[:, None]) + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return (float(beta[0]), float(beta[1])) if fit_slope else (float(beta[0]), 1.0)


def reliability_bins(y, p, n_bins=10, weight=None):
    """Chia theo phân vị của xác suất dự đoán (hợp với dữ liệu lệch). Trả list bin + ECE."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    w = np.ones_like(y) if weight is None else np.asarray(weight, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([p.min(), p.max() + EPS])
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    bins, ece, total = [], 0.0, w.sum()
    for b in range(len(edges) - 1):
        m = (idx == b) & (w > 0)
        if not m.any():
            continue
        wb = w[m].sum()
        mp = float(np.average(p[m], weights=w[m]))
        ob = float(np.average(y[m], weights=w[m]))
        bins.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()),
                     "mean_pred": mp, "obs_rate": ob})
        ece += wb / total * abs(mp - ob)
    return bins, float(ece)


def calibration_stats(y, p, weight=None, n_bins=10) -> dict:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    w = np.ones_like(y) if weight is None else np.asarray(weight, float)
    bins, ece = reliability_bins(y, p, n_bins, w)
    lp = logit(p)
    if len(np.unique(y[w > 0])) == 2:
        _, slope = fit_logistic_1d(lp, y, weight=w)
        citl, _ = fit_logistic_1d(lp, y, offset=lp, fit_slope=False, weight=w)  # intercept khi slope cố định = 1
    else:
        slope = citl = float("nan")
    obs = float(np.average(y, weights=w))
    exp = float(np.average(p, weights=w))
    return {"ece": ece, "calibration_slope": slope, "calibration_intercept": citl,
            "observed_rate": obs, "mean_predicted": exp, "oe_ratio": obs / exp if exp > 0 else float("nan"),
            "bins": bins}


def threshold_stats(y, p, threshold, weight=None) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    w = np.ones(len(y)) if weight is None else np.asarray(weight, float)
    tp = float(w[(pred == 1) & (y == 1)].sum())
    fp = float(w[(pred == 1) & (y == 0)].sum())
    fn = float(w[(pred == 0) & (y == 1)].sum())
    tn = float(w[(pred == 0) & (y == 0)].sum())
    div = lambda a, b: a / b if b > 0 else float("nan")  # noqa: E731
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "specificity": div(tn, tn + fp), "npv": div(tn, tn + fn),
            "fpr": div(fp, fp + tn), "flag_rate": div(tp + fp, tp + fp + fn + tn),
            "f1": div(2 * prec * rec, prec + rec) if prec == prec and rec == rec else float("nan")}


def discrimination(y, p, weight=None) -> dict:
    y = np.asarray(y).astype(int)
    w = None if weight is None else np.asarray(weight, float)
    both = len(np.unique(y if w is None else y[w > 0])) == 2
    prev = float(np.average(y, weights=w))
    brier = float(brier_score_loss(y, p, sample_weight=w))
    ref = prev * (1 - prev)
    return {"n": int(len(y) if w is None else (w > 0).sum()),
            "positive_rate": prev,
            "roc_auc": float(roc_auc_score(y, p, sample_weight=w)) if both else float("nan"),
            "pr_auc": float(average_precision_score(y, p, sample_weight=w)) if both else float("nan"),
            "brier": brier, "brier_skill": 1 - brier / ref if ref > 0 else float("nan")}


def full_metrics(y, p, threshold, weight=None) -> dict:
    out = discrimination(y, p, weight)
    out["pr_auc_lift"] = out["pr_auc"] / out["positive_rate"] if out["positive_rate"] > 0 else float("nan")
    cal = calibration_stats(y, p, weight)
    out.update({k: v for k, v in cal.items() if k != "bins"})
    out.update(threshold_stats(y, p, threshold, weight))
    return out


BOOT_METRICS = ("roc_auc", "pr_auc", "brier", "f1", "recall", "precision", "ece", "oe_ratio")


def bootstrap_ci(y, p, groups, threshold, n_boot=1000, seed=42, alpha=0.05, metrics=BOOT_METRICS) -> dict:
    """CI percentile, lấy mẫu lại THEO BỆNH NHÂN (một bệnh nhân nhiều lượt nhập viện)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    _, gcode = np.unique(np.asarray(groups), return_inverse=True)
    G = gcode.max() + 1
    rng = np.random.default_rng(seed)
    vals = {m: [] for m in metrics}
    for _ in range(n_boot):
        counts = rng.multinomial(G, np.full(G, 1.0 / G))
        w = counts[gcode].astype(float)
        if len(np.unique(y[w > 0])) < 2:
            continue
        r = discrimination(y, p, w)
        r.update(threshold_stats(y, p, threshold, w))
        if "ece" in metrics or "oe_ratio" in metrics:
            c = calibration_stats(y, p, w)
            r["ece"], r["oe_ratio"] = c["ece"], c["oe_ratio"]
        for m in metrics:
            vals[m].append(r[m])
    out = {}
    for m, v in vals.items():
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        out[m] = ([float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))]
                  if len(v) else [float("nan"), float("nan")])
    return out


def paired_bootstrap_diff(y, p_a, p_b, groups, n_boot=1000, seed=42, alpha=0.05) -> dict:
    """CI của HIỆU chỉ số giữa hai mô hình trên CÙNG mẫu bootstrap (lấy mẫu lại theo bệnh nhân).
    Trả ΔPR-AUC, ΔROC-AUC, ΔBrier (a − b). CI chứa 0 ⇒ hai mô hình không khác nhau có ý nghĩa."""
    y = np.asarray(y).astype(int)
    p_a, p_b = np.asarray(p_a, float), np.asarray(p_b, float)
    _, gcode = np.unique(np.asarray(groups), return_inverse=True)
    G = gcode.max() + 1
    rng = np.random.default_rng(seed)
    d = {"pr_auc": [], "roc_auc": [], "brier": []}
    for _ in range(n_boot):
        w = rng.multinomial(G, np.full(G, 1.0 / G))[gcode].astype(float)
        if len(np.unique(y[w > 0])) < 2:
            continue
        a, b = discrimination(y, p_a, w), discrimination(y, p_b, w)
        for k in d:
            d[k].append(a[k] - b[k])
    pa, pb = discrimination(y, p_a), discrimination(y, p_b)
    out = {}
    for k, v in d.items():
        v = np.asarray(v, float)
        lo, hi = float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))
        out[k] = {"diff": pa[k] - pb[k], "ci95": [lo, hi], "significant": not (lo <= 0 <= hi),
                  "p_two_sided": float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())))}
    return out

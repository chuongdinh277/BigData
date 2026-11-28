"""Độ bền khi thiếu dữ liệu: che dần nhóm thông tin đầu vào rồi đo hiệu năng giảm bao nhiêu.

Che một xét nghiệm = gán NULL cả first/min/max và bật cờ <lab>_missing = 1, đúng như khi bệnh nhân
thật sự không được làm xét nghiệm đó (không phải chỉ xóa một ô). Che sinh hiệu = NULL toàn bộ cột
sinh hiệu + vitals_missing = 1. Đánh giá bằng mô hình không thấy dòng đó (out-of-fold hoặc test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.common.config import SCHEMA

from .evaluation import calibration_stats, discrimination

LABS = SCHEMA["feature_columns"]["labs"]["items"]
VITALS = [c["name"] for c in SCHEMA["feature_columns"]["vitals_icu"] if c["name"] != "vitals_missing"]
OMR = [c["name"] for c in SCHEMA["feature_columns"]["history"] if c["name"].startswith("omr_")]

CORE_LABS = ["creatinine", "bun", "sodium", "potassium", "chloride", "bicarbonate", "anion_gap", "glucose",
             "hemoglobin", "hematocrit", "wbc", "platelet", "rdw"]
SPECIAL_LABS = [lab for lab in LABS if lab not in CORE_LABS]

# đơn vị che: tên → (cột giá trị, cột cờ thiếu)
def _units() -> dict[str, tuple[list[str], str | None]]:
    sufs = [s for s in SCHEMA["feature_columns"]["labs"]["suffixes"] if s != "_missing"]  # P2/P3 có thêm _last
    u = {f"lab:{lab}": ([f"{lab}{s}" for s in sufs], f"{lab}_missing") for lab in LABS}
    u["vitals"] = (VITALS, "vitals_missing")
    for c in OMR:
        u[f"omr:{c}"] = ([c], None)
    return u


UNITS = _units()
SCENARIOS = {
    "xet_nghiem_co_ban": [f"lab:{x}" for x in CORE_LABS],
    "xet_nghiem_dac_hieu": [f"lab:{x}" for x in SPECIAL_LABS],
    "sinh_hieu_icu": ["vitals"],
    "omr_ngoai_tru": [f"omr:{c}" for c in OMR],
    "tat_ca_ngau_nhien": list(UNITS),
}


def mask(X: pd.DataFrame, units: list[str], rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """Mỗi (dòng, đơn vị) bị che độc lập với xác suất `rate`."""
    X = X.copy()
    for u in units:
        cols, flag = UNITS[u]
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        hit = rng.random(len(X)) < rate
        if not hit.any():
            continue
        X.loc[hit, cols] = np.nan
        if flag and flag in X.columns:
            X.loc[hit, flag] = 1
    return X


def missingness_curve(predict, X: pd.DataFrame, y, rates=(0, .1, .25, .5, .75, 1.0), repeats=3, seed=42,
                      scenarios: dict | None = None) -> pd.DataFrame:
    """predict(X) → xác suất đã hiệu chỉnh. Trả bảng dài: scenario, rate, repeat, các chỉ số."""
    scenarios = scenarios or SCENARIOS
    rows = []
    rng = np.random.default_rng(seed)
    for name, units in scenarios.items():
        for rate in rates:
            for r in range(1 if rate == 0 else repeats):
                p = predict(mask(X, units, rate, rng) if rate > 0 else X)
                d = discrimination(y, p)
                c = calibration_stats(y, p)
                rows.append({"scenario": name, "rate": rate, "repeat": r, "roc_auc": d["roc_auc"],
                             "pr_auc": d["pr_auc"], "brier": d["brier"], "ece": c["ece"], "oe_ratio": c["oe_ratio"]})
    return pd.DataFrame(rows)


def summarize_curve(curve: pd.DataFrame) -> pd.DataFrame:
    g = curve.groupby(["scenario", "rate"], as_index=False)[["roc_auc", "pr_auc", "brier", "ece", "oe_ratio"]].mean()
    base = g[g["rate"] == 0].set_index("scenario")
    g["pr_auc_drop_pct"] = g.apply(lambda r: 100 * (1 - r["pr_auc"] / base.loc[r["scenario"], "pr_auc"]), axis=1)
    g["roc_auc_drop"] = g.apply(lambda r: base.loc[r["scenario"], "roc_auc"] - r["roc_auc"], axis=1)
    return g


def missing_fraction(X: pd.DataFrame) -> pd.Series:
    """Tỷ lệ đơn vị thông tin bị thiếu của mỗi ca (xét nghiệm, sinh hiệu, OMR)."""
    parts = []
    for cols, flag in UNITS.values():
        if flag and flag in X.columns:
            parts.append(X[flag].fillna(1).astype(float))
        else:
            present = [c for c in cols if c in X.columns]
            if present:
                parts.append(X[present].isna().all(axis=1).astype(float))
    return pd.concat(parts, axis=1).mean(axis=1)

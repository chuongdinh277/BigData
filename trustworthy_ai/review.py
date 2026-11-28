"""Quy trình xem xét thủ công: luật gắn cờ + đánh giá khối lượng hàng đợi.

Lý do gắn cờ (theo api_contract.md):
    HIGH_RISK              xác suất đã hiệu chỉnh ≥ ngưỡng HIGH
    NEAR_THRESHOLD         nằm ngay dưới ngưỡng HIGH (quyết định dễ đổi chiều)
    LOW_CONFIDENCE         các mô hình fold CV bất đồng nhiều (độ lệch chuẩn dự đoán cao)
    MISSING_KEY_LAB:<lab>  thiếu xét nghiệm cơ bản thường luôn có (creatinine, bun, sodium, hemoglobin, wbc)
    HIGH_MISSINGNESS       tỷ lệ thông tin thiếu cao bất thường so với dữ liệu huấn luyện
    OUT_OF_RANGE:<field>   giá trị ngoài khoảng sinh lý (configs/feature_items.yaml)
    OUT_OF_DISTRIBUTION    nhiều đặc trưng nằm ngoài khoảng 0.5–99.5% của dữ liệu huấn luyện
Ưu tiên: 1 = HIGH_RISK, 2 = các lý do độ tin cậy, 3 = chỉ chất lượng dữ liệu.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .robustness import missing_fraction

KEY_LABS = ["creatinine", "bun", "sodium", "hemoglobin", "wbc"]
REPO_ROOT = Path(__file__).resolve().parents[1]


def valid_ranges() -> dict[str, list[float]]:
    # đọc trực tiếp YAML (không qua data_spark) để backend không phải cài PySpark
    cfg = yaml.safe_load((REPO_ROOT / "configs/feature_items.yaml").read_text(encoding="utf-8"))
    r = {f"{lab}_{s}": spec["valid_range"] for lab, spec in cfg["labs"].items() for s in ("first", "min", "max", "last")}
    for spec in cfg["vitals"].values():
        for col in spec["feature_columns"]:
            r[col] = spec["valid_range"]
    r["age"] = [18, 120]
    return r


def build_rules(*, thresholds: dict, X_train: pd.DataFrame, std_train=None, near_band: float = 0.75,
                std_quantile: float = 0.90, missing_quantile: float = 0.95, ood_min_features: int = 3) -> dict:
    num = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])
           and not c.endswith("_missing") and X_train[c].nunique() > 2]
    q = X_train[num].quantile([0.005, 0.995])
    rules = {
        "high_risk_threshold": thresholds["medium_high"],
        "near_threshold_lower": thresholds["medium_high"] * near_band,
        "std_cutoff": float(np.quantile(std_train, std_quantile)) if std_train is not None else None,
        "std_quantile": std_quantile,
        "missing_fraction_cutoff": float(missing_fraction(X_train).quantile(missing_quantile)),
        "key_labs": KEY_LABS,
        "valid_ranges": valid_ranges(),
        "ood_bounds": {c: [float(q.loc[0.005, c]), float(q.loc[0.995, c])] for c in num},
        "ood_min_features": ood_min_features,
    }
    return rules


def flag_cases(X: pd.DataFrame, p_cal, rules: dict, std=None) -> pd.DataFrame:
    p = np.asarray(p_cal, float)
    n = len(X)
    reasons = [[] for _ in range(n)]

    def add(mask, label):
        for i in np.nonzero(np.asarray(mask))[0]:
            reasons[i].append(label)

    add(p >= rules["high_risk_threshold"], "HIGH_RISK")
    add((p >= rules["near_threshold_lower"]) & (p < rules["high_risk_threshold"]), "NEAR_THRESHOLD")
    if std is not None and rules.get("std_cutoff") is not None:
        add(np.asarray(std) >= rules["std_cutoff"], "LOW_CONFIDENCE")
    for lab in rules["key_labs"]:
        col = f"{lab}_missing"
        if col in X:
            add(X[col].fillna(1).to_numpy() == 1, f"MISSING_KEY_LAB:{lab}")
    add(missing_fraction(X).to_numpy() > rules["missing_fraction_cutoff"], "HIGH_MISSINGNESS")
    for c, (lo, hi) in rules["valid_ranges"].items():
        if c in X:
            v = X[c].to_numpy(float)
            add((~np.isnan(v)) & ((v < lo) | (v > hi)), f"OUT_OF_RANGE:{c}")
    out_count = np.zeros(n, int)
    for c, (lo, hi) in rules["ood_bounds"].items():
        if c in X:
            v = X[c].to_numpy(float)
            out_count += ((~np.isnan(v)) & ((v < lo) | (v > hi))).astype(int)
    add(out_count >= rules["ood_min_features"], "OUT_OF_DISTRIBUTION")

    def priority(rs):
        if "HIGH_RISK" in rs:
            return 1
        if any(r in ("NEAR_THRESHOLD", "LOW_CONFIDENCE") for r in rs):
            return 2
        return 3 if rs else 0

    return pd.DataFrame({"flag": [bool(r) for r in reasons], "reasons": reasons,
                         "priority": [priority(r) for r in reasons]}, index=X.index)


LOW_CONFIDENCE_REASONS = ("LOW_CONFIDENCE", "NEAR_THRESHOLD", "MISSING_KEY_LAB", "HIGH_MISSINGNESS",
                          "OUT_OF_RANGE", "OUT_OF_DISTRIBUTION")


def low_confidence_mask(flags: pd.DataFrame) -> np.ndarray:
    """Ca bị gắn cờ vì ĐỘ TIN CẬY THẤP (không tính cờ chỉ vì nguy cơ cao)."""
    return flags["reasons"].map(lambda rs: any(reason_family(r) in LOW_CONFIDENCE_REASONS for r in rs)).to_numpy()


def confidence_validation(y, p, flags: pd.DataFrame) -> dict:
    """Bằng chứng luật "độ tin cậy thấp" có tác dụng: mô hình phải sai NHIỀU HƠN ở nhóm bị gắn cờ.

    So Brier, ECE và |O − E| giữa nhóm bị gắn cờ tin cậy thấp và nhóm còn lại.
    Nếu nhóm bị gắn cờ không tệ hơn thì luật không bắt được ca khó → phải báo cáo trung thực."""
    from .evaluation import calibration_stats, discrimination

    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    m = low_confidence_mask(flags)
    out = {}
    for name, mask in (("low_confidence", m), ("confident", ~m)):
        if mask.sum() < 30 or len(np.unique(y[mask])) < 2:
            out[name] = {"n": int(mask.sum()), "note": "không đủ dữ liệu"}
            continue
        d = discrimination(y[mask], p[mask])
        c = calibration_stats(y[mask], p[mask])
        out[name] = {"n": int(mask.sum()), "share": float(mask.mean()), "positive_rate": d["positive_rate"],
                     "brier": d["brier"], "roc_auc": d["roc_auc"], "ece": c["ece"],
                     "abs_obs_minus_exp": abs(c["observed_rate"] - c["mean_predicted"])}
    lc, cf = out.get("low_confidence", {}), out.get("confident", {})
    if "brier" in lc and "brier" in cf:
        # Brier phụ thuộc tỷ lệ nền → so thêm Brier chuẩn hóa theo p(1−p) của chính nhóm
        norm = lambda d: d["brier"] / max(d["positive_rate"] * (1 - d["positive_rate"]), 1e-9)  # noqa: E731
        out["brier_scaled_ratio_lowconf_vs_confident"] = norm(lc) / norm(cf)
        out["roc_auc_gap_confident_minus_lowconf"] = cf["roc_auc"] - lc["roc_auc"]
    return out


def risk_coverage(y, p, uncertainty, coverages=(1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5)) -> pd.DataFrame:
    """Dự đoán có chọn lọc: chuyển (1 − coverage) ca bất định nhất cho người xem,
    đo chất lượng mô hình trên phần máy tự xử lý. Đường đi xuống = độ bất định có ý nghĩa."""
    from .evaluation import calibration_stats, discrimination

    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    order = np.argsort(np.asarray(uncertainty, float), kind="stable")  # chắc chắn nhất trước
    rows = []
    for c in coverages:
        keep = order[: max(1, int(round(c * len(y))))]
        yy, pp = y[keep], p[keep]
        d = discrimination(yy, pp)
        rows.append({"coverage": c, "n_auto": len(keep), "n_to_review": len(y) - len(keep),
                     "brier": d["brier"], "brier_skill": d["brier_skill"], "roc_auc": d["roc_auc"],
                     "pr_auc": d["pr_auc"], "ece": calibration_stats(yy, pp)["ece"],
                     "positive_rate": d["positive_rate"]})
    return pd.DataFrame(rows)


def reason_family(r: str) -> str:
    return r.split(":", 1)[0]


def queue_report(flags: pd.DataFrame, y) -> dict:
    """Khối lượng hàng đợi và hiệu quả: hàng đợi bắt được bao nhiêu % ca tử vong."""
    y = np.asarray(y).astype(int)
    f = flags["flag"].to_numpy()
    n, pos = len(y), int(y.sum())
    by_reason = {}
    fam = flags["reasons"].map(lambda rs: sorted({reason_family(r) for r in rs}))
    for name in ["HIGH_RISK", "NEAR_THRESHOLD", "LOW_CONFIDENCE", "MISSING_KEY_LAB", "HIGH_MISSINGNESS",
                 "OUT_OF_RANGE", "OUT_OF_DISTRIBUTION"]:
        m = fam.map(lambda rs, k=name: k in rs).to_numpy()
        by_reason[name] = {"share": float(m.mean()), "n": int(m.sum()),
                           "death_rate": float(y[m].mean()) if m.any() else None}
    only_quality = (flags["priority"] == 3).to_numpy()
    return {"n": n, "flagged_share": float(f.mean()), "flagged_n": int(f.sum()),
            "deaths_captured_share": float(y[f].sum() / pos) if pos else None,
            "death_rate_in_queue": float(y[f].mean()) if f.any() else None,
            "death_rate_not_flagged": float(y[~f].mean()) if (~f).any() else None,
            "quality_only_share": float(only_quality.mean()),
            "priority_counts": flags["priority"].value_counts().sort_index().to_dict(),
            "by_reason": by_reason}

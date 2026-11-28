"""Hiệu năng và hiệu chỉnh theo nhóm bệnh nhân + chênh lệch công bằng.

race_group, insurance, hf_primary, icd_version_group chỉ dùng để ĐÁNH GIÁ, không là đầu vào mô hình.
Vì tỷ lệ tử vong khác nhau giữa các nhóm, so PR-AUC phải kèm `pr_auc_lift` = PR-AUC / tỷ lệ dương.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import bootstrap_ci, full_metrics

MIN_EVENTS = 10  # dưới ngưỡng này không tính chỉ số (và che số đếm trong báo cáo)


def add_subgroup_columns(df: pd.DataFrame, missing_frac: pd.Series | None = None) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if "gender_male" in df:
        out["gioi_tinh"] = df["gender_male"].map({1: "Nam", 0: "Nữ"}).fillna("Không rõ")
    if "age" in df:
        out["nhom_tuoi"] = pd.cut(df["age"], [0, 64, 79, 200], labels=["18-64", "65-79", "80+"]).astype(str)
    for c, name in [("anchor_year_group", "giai_doan"), ("race_group", "chung_toc"), ("insurance", "bao_hiem"),
                    ("icd_version_group", "phien_ban_icd"), ("admission_type_group", "loai_nhap_vien")]:
        if c in df:
            out[name] = df[c].astype(str)
    if "in_icu_24h" in df:
        out["icu_24h"] = df["in_icu_24h"].map({1: "Có ICU", 0: "Không ICU"})
    if "icu_any" in df:                                   # P2/P3: có nằm ICU trong đợt
        out["icu_trong_dot"] = df["icu_any"].map({1: "Có ICU", 0: "Không ICU"})
    if "discharge_location_group" in df:
        out["noi_xuat_vien"] = df["discharge_location_group"].astype(str)
    if "hf_primary" in df:
        out["suy_tim_chinh"] = df["hf_primary"].map({1: "Chẩn đoán chính", 0: "Chẩn đoán phụ"})
    if "ntprobnp_missing" in df:
        out["co_ntprobnp"] = df["ntprobnp_missing"].map({0: "Có NT-proBNP", 1: "Không có"})
    # Nhóm GIAO NHAU: chênh lệch thường ẩn ở giao điểm hai thuộc tính (vd nữ cao tuổi)
    if "gioi_tinh" in out and "nhom_tuoi" in out:
        out["gioi_x_tuoi"] = out["gioi_tinh"] + " × " + out["nhom_tuoi"]
    icu = out.get("icu_24h", out.get("icu_trong_dot"))
    if "bao_hiem" in out and icu is not None:
        out["bao_hiem_x_icu"] = out["bao_hiem"] + " × " + icu.astype(str)
    if missing_frac is not None:
        out["muc_thieu_du_lieu"] = pd.qcut(missing_frac.rank(method="first"), 3,
                                           labels=["Thiếu ít", "Thiếu vừa", "Thiếu nhiều"]).astype(str)
    return out


def subgroup_table(groups_df: pd.DataFrame, y, p, subjects, threshold: float, n_boot: int = 200,
                   seed: int = 42) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    subjects = np.asarray(subjects)
    overall = full_metrics(y, p, threshold)
    rows = [{"attribute": "TỔNG THỂ", "group": "Tất cả", **_pick(overall), "roc_auc_ci": None, "pr_auc_ci": None,
             "flags": ""}]
    for attr in groups_df.columns:
        for g, idx in groups_df.groupby(attr, observed=True).indices.items():
            yy, pp = y[idx], p[idx]
            n_pos, n_neg = int(yy.sum()), int(len(yy) - yy.sum())
            row = {"attribute": attr, "group": str(g), "n": int(len(idx)), "n_pos": n_pos,
                   "positive_rate": float(yy.mean())}
            if n_pos < MIN_EVENTS or n_neg < MIN_EVENTS:
                row["flags"] = "quá ít ca để đánh giá"
                rows.append(row)
                continue
            m = full_metrics(yy, pp, threshold)
            ci = bootstrap_ci(yy, pp, subjects[idx], threshold, n_boot=n_boot, seed=seed,
                              metrics=("roc_auc", "pr_auc", "oe_ratio"))
            row.update(_pick(m))
            row["roc_auc_ci"], row["pr_auc_ci"], row["oe_ratio_ci"] = ci["roc_auc"], ci["pr_auc"], ci["oe_ratio"]
            flags = []
            if ci["roc_auc"][1] < overall["roc_auc"]:
                flags.append("ROC-AUC thấp hơn tổng thể (CI không chứa giá trị tổng thể)")
            lo, hi = ci["oe_ratio"]
            if lo > 1.0:
                flags.append("mô hình đánh giá THẤP nguy cơ (O/E > 1)")
            elif hi < 1.0:
                flags.append("mô hình đánh giá CAO nguy cơ (O/E < 1)")
            row["flags"] = "; ".join(flags)
            rows.append(row)
    return pd.DataFrame(rows)


def _pick(m: dict) -> dict:
    keys = ("n", "positive_rate", "roc_auc", "pr_auc", "pr_auc_lift", "brier", "ece", "oe_ratio",
            "calibration_slope", "recall", "precision", "fpr", "flag_rate")
    out = {k: m.get(k) for k in keys}
    out["n_pos"] = int(round(m["tp"] + m["fn"]))
    return out


def fairness_gaps(table: pd.DataFrame) -> pd.DataFrame:
    """Chênh lệch lớn nhất giữa các nhóm của cùng thuộc tính (chỉ nhóm đủ ca).
    recall_gap = equal opportunity difference; fpr_gap thuộc equalized odds."""
    t = table[(table["attribute"] != "TỔNG THỂ") & table["roc_auc"].notna()]
    rows = []
    for attr, g in t.groupby("attribute"):
        if len(g) < 2:
            continue
        rows.append({"attribute": attr, "n_groups": len(g),
                     "roc_auc_gap": g["roc_auc"].max() - g["roc_auc"].min(),
                     "pr_auc_lift_gap": g["pr_auc_lift"].max() - g["pr_auc_lift"].min(),
                     "recall_gap": g["recall"].max() - g["recall"].min(),
                     "fpr_gap": g["fpr"].max() - g["fpr"].min(),
                     "oe_ratio_min": g["oe_ratio"].min(), "oe_ratio_max": g["oe_ratio"].max(),
                     "worst_roc_group": g.loc[g["roc_auc"].idxmin(), "group"]})
    return pd.DataFrame(rows)


# Vì sao mỗi trục là "relevant" (đề: "relevant patient subgroups"). Dùng trong báo cáo.
RELEVANCE = {
    "gioi_tinh": "Công bằng: triệu chứng và điều trị suy tim khác nhau giữa nam và nữ",
    "nhom_tuoi": "Lâm sàng: tuổi là yếu tố tiên lượng mạnh nhất; ≥80 tuổi ít có trong thử nghiệm lâm sàng",
    "giai_doan": "Drift: phác đồ, hệ mã ICD (2015), COVID-19 (2020–2022) thay đổi theo thời gian",
    "chung_toc": "Công bằng: chênh lệch tiếp cận y tế; GWTG-HF có biến chủng tộc",
    "bao_hiem": "Công bằng: proxy kinh tế – xã hội (Medicaid)",
    "phien_ban_icd": "Dữ liệu: bệnh nền mã hóa ICD-9 và ICD-10 khác nhau",
    "loai_nhap_vien": "Lâm sàng: nhập cấp cứu và nhập theo kế hoạch có nguy cơ nền khác nhau",
    "icu_24h": "Dữ liệu: sinh hiệu chỉ có ở ca ICU, mô hình có thể dựa vào việc có/không có sinh hiệu",
    "icu_trong_dot": "Dữ liệu: như trên cho bài toán lúc xuất viện",
    "noi_xuat_vien": "Lâm sàng: ra viện về cơ sở chăm sóc khác có nguy cơ tái nhập/tử vong cao hơn",
    "suy_tim_chinh": "Cohort: suy tim là lý do nhập viện chính hay chỉ là bệnh kèm",
    "co_ntprobnp": "Dữ liệu: dấu ấn sinh học quan trọng nhất của suy tim nhưng thiếu 68.9%",
    "muc_thieu_du_lieu": "Độ tin cậy: hiệu năng ở ca thiếu nhiều dữ liệu",
    "gioi_x_tuoi": "Giao nhau: chênh lệch ẩn ở giao điểm giới × tuổi",
    "bao_hiem_x_icu": "Giao nhau: kinh tế – xã hội × mức độ nặng",
}

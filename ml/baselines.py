"""Mốc so sánh cho mô hình học máy: mốc sàn (Dummy) và thang điểm lâm sàng tính lại từ Gold.

Vì sao cần: mô hình ML chỉ có giá trị khi hơn được (1) đoán theo tỷ lệ nền và (2) công cụ bác sĩ
đang dùng. Các thang điểm dưới đây là BẢN XẤP XỈ dựng từ cột có sẵn trong Gold, ghi rõ biến thiếu;
không được trình bày như bản gốc đã kiểm định.

| Bài | Thang điểm | Nguồn | Biến thiếu/xấp xỉ trong Gold |
|---|---|---|---|
| P1 | GWTG-HF (tử vong trong viện) | Peterson et al., Circ Cardiovasc Qual Outcomes 2010 | Huyết áp, nhịp tim chỉ có ở ca ICU (~77–79% thiếu) → bản đầy đủ chỉ tính trên ca có sinh hiệu; bản rút gọn bỏ 2 biến này |
| P2 | HOSPITAL | Donzé et al., JAMA Intern Med 2013 | Thiếu "ra viện từ khoa ung bướu" và "có thủ thuật" → 5/7 biến |
| P2 | LACE | van Walraven et al., CMAJ 2010 | Charlson xấp xỉ bằng số cờ bệnh nền; lượt cấp cứu 6 tháng xấp xỉ bằng số lần nhập viện 365 ngày |

Bảng điểm GWTG-HF phải được đối chiếu lại với bài gốc trước khi đưa số vào báo cáo
(người phụ trách: xem bảng phân công).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, roc_auc_score

from trustworthy_ai.evaluation import discrimination


def _bins(x: pd.Series, edges: list[float], points: list[int]) -> pd.Series:
    """Điểm theo khoảng [edges[i], edges[i+1]); NaN → NaN."""
    idx = np.searchsorted(np.asarray(edges, float), x.to_numpy(float), side="right") - 1
    idx = np.clip(idx, 0, len(points) - 1)
    out = pd.Series(np.asarray(points, float)[idx], index=x.index)
    return out.where(x.notna())


def gwtg_hf(df: pd.DataFrame, reduced: bool = False) -> pd.Series:
    """GWTG-HF: tuổi, huyết áp tâm thu, BUN, nhịp tim, natri, COPD, chủng tộc không phải da đen.
    reduced=True bỏ huyết áp và nhịp tim (để tính được cho ca không nằm ICU)."""
    age = _bins(df["age"], [-np.inf, 20, 30, 40, 50, 60, 70, 80, 90, 100], [0, 3, 6, 8, 11, 14, 17, 19, 22, 25])
    bun = _bins(df["bun_first"], [-np.inf, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], [0, 2, 4, 6, 8, 9, 11, 13, 15, 17, 19])
    na = _bins(df["sodium_first"], [-np.inf, 131, 135, 139], [4, 2, 1, 0])
    copd = df.get("hx_copd", pd.Series(0, index=df.index)).fillna(0) * 2
    race = df.get("race_group", pd.Series("UNKNOWN", index=df.index)).astype(str)
    nonblack = (race != "BLACK").astype(float) * 3
    score = age + bun + na + copd + nonblack
    if not reduced:
        sbp = _bins(df["sbp_min"], [-np.inf, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
                    [28, 26, 24, 23, 21, 19, 17, 15, 13, 11, 9, 8, 6, 4, 2, 0])
        hr = _bins(df["hr_mean"], [-np.inf, 80, 85, 90, 95, 100, 105], [0, 1, 3, 4, 5, 6, 8])
        score = score + sbp + hr
    return score


def hospital_score(df: pd.DataFrame) -> pd.Series:
    """HOSPITAL rút gọn (5/7 biến): Hb < 12 lúc ra viện, Na < 135 lúc ra viện, nhập viện cấp cứu,
    số lần nhập viện 12 tháng trước, LOS ≥ 5 ngày."""
    hb = (df["hemoglobin_last"].fillna(df["hemoglobin_min"]) < 12).astype(float)
    na = (df["sodium_last"].fillna(df["sodium_min"]) < 135).astype(float)
    urgent = df["admission_type_group"].astype(str).isin(["EMERGENCY", "URGENT"]).astype(float)
    prior = _bins(df["n_prior_adm_365d"].fillna(0), [-np.inf, 2, 6], [0, 2, 5])
    los = (df["los_days"] >= 5).astype(float) * 2
    return hb + na + urgent + prior + los


def lace_score(df: pd.DataFrame) -> pd.Series:
    """LACE xấp xỉ: L = thời gian nằm viện, A = nhập cấp cứu, C = số bệnh nền (xấp xỉ Charlson),
    E = số lần nhập viện 365 ngày trước (xấp xỉ lượt cấp cứu 6 tháng, chặn ở 4)."""
    los = _bins(df["los_days"].fillna(0), [-np.inf, 1, 2, 3, 4, 7, 14], [0, 1, 2, 3, 4, 5, 7])
    acute = df["admission_type_group"].astype(str).isin(["EMERGENCY", "URGENT"]).astype(float) * 3
    dx = [c for c in df.columns if c.startswith("dx_")] or [c for c in df.columns if c.startswith("hx_")]
    n_cm = df[dx].fillna(0).sum(axis=1)
    cm = _bins(n_cm, [-np.inf, 1, 2, 3, 4], [0, 1, 2, 3, 5])
    ed = df["n_prior_adm_365d"].fillna(0).clip(upper=4)
    return los + acute + cm + ed


def clinical_scores(df: pd.DataFrame, task: str) -> dict[str, pd.Series]:
    if task == "p1":
        return {"GWTG-HF (đầy đủ, chỉ ca có sinh hiệu)": gwtg_hf(df), "GWTG-HF (rút gọn, không sinh hiệu)": gwtg_hf(df, True)}
    if task == "p2":
        return {"HOSPITAL (5/7 biến)": hospital_score(df), "LACE (xấp xỉ)": lace_score(df)}
    return {}


def evaluate_baselines(df: pd.DataFrame, y: np.ndarray, task: str, p_model=None) -> list[dict]:
    """Đánh giá mốc trên CÙNG tập (dev, không đụng test). Thang điểm là điểm xếp hạng nên chỉ so
    ROC-AUC và PR-AUC (không có xác suất để so hiệu chuẩn). Nếu có `p_model` (dự đoán ngoài fold của
    mô hình được chọn) thì tính thêm chỉ số của mô hình trên ĐÚNG tập con mà thang điểm tính được,
    để so sánh công bằng."""
    y = np.asarray(y).astype(int)
    prev = float(y.mean())
    rows = [{"baseline": "Dummy (tỷ lệ nền)", "n": len(y), "coverage": 1.0, "positive_rate": prev,
             "roc_auc": 0.5, "pr_auc": prev, "pr_auc_lift": 1.0}]
    for name, s in clinical_scores(df, task).items():
        m = s.notna().to_numpy()
        if m.sum() < 50 or len(np.unique(y[m])) < 2:
            rows.append({"baseline": name, "n": int(m.sum()), "coverage": float(m.mean()), "note": "không đủ dữ liệu"})
            continue
        score, yy = s[m].to_numpy(float), y[m]
        prev_m = float(yy.mean())
        pr = float(average_precision_score(yy, score))
        row = {"baseline": name, "n": int(m.sum()), "coverage": float(m.mean()), "positive_rate": prev_m,
               "roc_auc": float(roc_auc_score(yy, score)), "pr_auc": pr, "pr_auc_lift": pr / prev_m}
        if p_model is not None:
            dm = discrimination(y[m], np.asarray(p_model, float)[m])
            row.update({"model_roc_auc_same_subset": dm["roc_auc"], "model_pr_auc_same_subset": dm["pr_auc"]})
        rows.append(row)
    return rows

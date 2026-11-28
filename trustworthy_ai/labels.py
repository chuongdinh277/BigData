"""Tên dễ hiểu và đơn vị cho đặc trưng, dùng khi hiển thị giải thích cho người dùng (UI, báo cáo).

Ví dụ: "bun_max" → "BUN cao nhất (24h đầu)", đơn vị "mg/dL".
"""
from __future__ import annotations

CONCEPTS: dict[str, tuple[str, str]] = {
    "ntprobnp": ("NT-proBNP", "pg/mL"), "creatinine": ("Creatinine", "mg/dL"), "bun": ("BUN (urê máu)", "mg/dL"),
    "sodium": ("Natri", "mEq/L"), "potassium": ("Kali", "mEq/L"), "chloride": ("Clo", "mEq/L"),
    "bicarbonate": ("Bicarbonate", "mEq/L"), "anion_gap": ("Khoảng trống anion", "mEq/L"),
    "glucose": ("Glucose", "mg/dL"), "hemoglobin": ("Hemoglobin", "g/dL"), "hematocrit": ("Hematocrit", "%"),
    "wbc": ("Bạch cầu", "K/uL"), "platelet": ("Tiểu cầu", "K/uL"), "rdw": ("RDW", "%"),
    "troponin_t": ("Troponin T", "ng/mL"), "lactate": ("Lactate", "mmol/L"), "albumin": ("Albumin", "g/dL"),
    "magnesium": ("Magie", "mg/dL"), "bilirubin": ("Bilirubin", "mg/dL"), "inr": ("INR", ""),
}
SUFFIX = {"_first": "đầu tiên", "_min": "thấp nhất", "_max": "cao nhất", "_last": "48h cuối", "_missing": "không có kết quả"}
SINGLE: dict[str, tuple[str, str]] = {
    "age": ("Tuổi", "năm"), "gender_male": ("Giới nam", ""), "admission_type_group": ("Loại nhập viện", ""),
    "admission_location_group": ("Nơi chuyển đến", ""), "came_from_ed": ("Vào từ cấp cứu", ""),
    "in_icu_24h": ("Nằm ICU trong 24h đầu", ""), "icu_any": ("Có nằm ICU", ""), "icu_days": ("Số ngày ICU", "ngày"),
    "los_days": ("Số ngày nằm viện", "ngày"), "discharge_location_group": ("Nơi xuất viện", ""),
    "n_diagnoses": ("Số chẩn đoán của đợt", ""), "n_prior_adm_365d": ("Số lần nhập viện 12 tháng trước", "lần"),
    "n_prior_hf_adm": ("Số lần nhập viện vì suy tim trước đây", "lần"), "first_admission": ("Lần nhập viện đầu tiên", ""),
    "days_since_last_discharge": ("Số ngày từ lần ra viện trước", "ngày"),
    "n_lab_tests_24h": ("Số xét nghiệm trong 24h đầu", ""), "n_lab_tests_stay": ("Số xét nghiệm cả đợt", ""),
    "omr_bmi_last": ("BMI ngoại trú gần nhất", "kg/m²"), "omr_weight_last": ("Cân nặng ngoại trú gần nhất", "kg"),
    "omr_sbp_last": ("Huyết áp tâm thu ngoại trú gần nhất", "mmHg"), "omr_egfr_last": ("eGFR ngoại trú gần nhất", "mL/phút"),
    "hr_mean": ("Nhịp tim trung bình", "lần/phút"), "hr_max": ("Nhịp tim cao nhất", "lần/phút"),
    "sbp_min": ("Huyết áp tâm thu thấp nhất", "mmHg"), "sbp_mean": ("Huyết áp tâm thu trung bình", "mmHg"),
    "dbp_mean": ("Huyết áp tâm trương trung bình", "mmHg"), "mbp_min": ("Huyết áp trung bình thấp nhất", "mmHg"),
    "rr_mean": ("Nhịp thở trung bình", "lần/phút"), "rr_max": ("Nhịp thở cao nhất", "lần/phút"),
    "spo2_min": ("SpO₂ thấp nhất", "%"), "temp_max_c": ("Nhiệt độ cao nhất", "°C"),
    "vitals_missing": ("Không có sinh hiệu ICU (không nằm ICU)", ""),
}
HX = {"ckd": "bệnh thận mạn", "diabetes": "đái tháo đường", "afib": "rung nhĩ", "copd": "COPD",
      "hypertension": "tăng huyết áp", "cad": "bệnh mạch vành", "prior_mi": "nhồi máu cơ tim", "stroke": "đột quỵ"}


def describe(feature: str) -> dict:
    if feature in SINGLE:
        name, unit = SINGLE[feature]
        return {"label": name, "unit": unit}
    if feature.startswith(("hx_", "dx_")):
        what = HX.get(feature[3:], feature[3:])
        return {"label": ("Tiền sử " if feature.startswith("hx_") else "Chẩn đoán đợt này: ") + what, "unit": ""}
    for suf, text in SUFFIX.items():
        if feature.endswith(suf) and feature[: -len(suf)] in CONCEPTS:
            name, unit = CONCEPTS[feature[: -len(suf)]]
            return {"label": f"{name} {text}", "unit": "" if suf == "_missing" else unit}
    return {"label": feature, "unit": ""}

"""Ca DEMO dựng tay (không lấy từ MIMIC) để trình diễn giao diện: nguy cơ thấp, trung bình, cao,
và một ca thiếu nhiều dữ liệu để thấy luồng "độ tin cậy thấp → xem xét thủ công".

Giá trị nằm trong khoảng sinh lý thường gặp; không đại diện cho bệnh nhân thật nào.
"""
from __future__ import annotations

_LAB_KEYS = ("first", "min", "max")


def _labs(values: dict[str, float], last: bool) -> dict:
    out = {}
    for lab, v in values.items():
        for k in _LAB_KEYS:
            out[f"{lab}_{k}"] = v * {"first": 1.0, "min": 0.92, "max": 1.12}[k]
        if last:
            out[f"{lab}_last"] = v
    return out


_COMMON_LOW = {"creatinine": 1.0, "bun": 18, "sodium": 139, "potassium": 4.1, "chloride": 102, "bicarbonate": 25,
               "anion_gap": 12, "glucose": 115, "hemoglobin": 13.2, "hematocrit": 40, "wbc": 7.5, "platelet": 230,
               "rdw": 13.6, "magnesium": 2.0, "inr": 1.1, "ntprobnp": 900}
_COMMON_HIGH = {"creatinine": 2.6, "bun": 72, "sodium": 128, "potassium": 5.3, "chloride": 94, "bicarbonate": 17,
                "anion_gap": 19, "glucose": 210, "hemoglobin": 9.4, "hematocrit": 29, "wbc": 15.8, "platelet": 140,
                "rdw": 17.8, "magnesium": 2.4, "inr": 1.9, "ntprobnp": 18500, "troponin_t": 0.21, "lactate": 4.2,
                "albumin": 2.6, "bilirubin": 2.1}
_COMMON_MID = {k: (_COMMON_LOW.get(k, v) + v) / 2 for k, v in _COMMON_HIGH.items()}
_VITALS_HIGH = {"hr_mean": 108, "hr_max": 128, "sbp_min": 82, "sbp_mean": 96, "dbp_mean": 52, "mbp_min": 58,
                "rr_mean": 25, "rr_max": 32, "spo2_min": 86, "temp_max_c": 37.9, "vitals_missing": 0}


def _base(task: str, age: int, male: int, icu: int, prior: int, hx: dict) -> dict:
    d = {"age": age, "gender_male": male, "admission_type_group": "EMERGENCY",
         "admission_location_group": "EMERGENCY_ROOM", "came_from_ed": 1,
         "n_prior_adm_365d": prior, "n_prior_hf_adm": max(prior - 1, 0), "first_admission": int(prior == 0),
         "days_since_last_discharge": None if prior == 0 else 45, "omr_bmi_last": 27.5, "omr_weight_last": 78,
         "omr_sbp_last": 128, "omr_egfr_last": 60}
    d.update({f"hx_{k}": v for k, v in hx.items()})
    if task == "p1":
        d["in_icu_24h"] = icu
    else:
        d.update({"icu_any": icu, "icu_days": 3.0 if icu else 0.0, "los_days": 9.0 if icu else 4.0,
                  "discharge_location_group": "FACILITY" if icu else "HOME", "n_diagnoses": 18 if icu else 9})
        d.update({f"dx_{k}": v for k, v in hx.items()})
    return d


def demo_cases(task: str) -> list[dict]:
    last = task != "p1"
    hx_low = {"ckd": 0, "diabetes": 0, "afib": 0, "copd": 0, "hypertension": 1, "cad": 0, "prior_mi": 0, "stroke": 0}
    hx_high = {"ckd": 1, "diabetes": 1, "afib": 1, "copd": 1, "hypertension": 1, "cad": 1, "prior_mi": 1, "stroke": 0}
    low = _base(task, 58, 0, 0, 0, hx_low) | _labs(_COMMON_LOW, last) | {"vitals_missing": 1}
    mid = _base(task, 74, 1, 0, 1, hx_low | {"ckd": 1, "afib": 1}) | _labs(_COMMON_MID, last) | {"vitals_missing": 1}
    high = _base(task, 86, 1, 1, 3, hx_high) | _labs(_COMMON_HIGH, last) | _VITALS_HIGH
    sparse = _base(task, 81, 0, 0, 0, hx_low) | {"creatinine_first": 1.9, "sodium_first": 131, "vitals_missing": 1}
    return [
        {"case_id": f"demo-{task}-low", "title": "Nguy cơ thấp", "description": "58 tuổi, xét nghiệm bình thường", "features": low},
        {"case_id": f"demo-{task}-mid", "title": "Nguy cơ trung bình", "description": "74 tuổi, suy thận mạn, rung nhĩ",
         "features": mid},
        {"case_id": f"demo-{task}-high", "title": "Nguy cơ cao", "description": "86 tuổi, sốc tim, toan lactic, ICU",
         "features": high},
        {"case_id": f"demo-{task}-sparse", "title": "Thiếu dữ liệu",
         "description": "Chỉ có 2 xét nghiệm, để thấy cờ độ tin cậy thấp", "features": sparse},
    ]

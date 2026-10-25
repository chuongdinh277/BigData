"""Tạo bộ MIMIC-IV GIẢ (cùng tên bảng, cùng cột) để code/test khi không có dữ liệu thật.

Không chứa dữ liệu thật — an toàn để commit và dùng cho người chưa có credential PhysioNet.
    python tools/make_synthetic_mimic.py --out sample_data/mimic_fake --patients 800
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABS = [  # itemid, label, fluid, category, uom, mean, sd
    (50963, "NTproBNP", "Blood", "Chemistry", "pg/mL", 4000, 3000),
    (50912, "Creatinine", "Blood", "Chemistry", "mg/dL", 1.5, 0.8),
    (51006, "Urea Nitrogen", "Blood", "Chemistry", "mg/dL", 30, 15),
    (50983, "Sodium", "Blood", "Chemistry", "mEq/L", 138, 4),
    (50971, "Potassium", "Blood", "Chemistry", "mEq/L", 4.2, 0.6),
    (50902, "Chloride", "Blood", "Chemistry", "mEq/L", 101, 5),
    (50882, "Bicarbonate", "Blood", "Chemistry", "mEq/L", 25, 4),
    (50868, "Anion Gap", "Blood", "Chemistry", "mEq/L", 14, 3),
    (50931, "Glucose", "Blood", "Chemistry", "mg/dL", 140, 50),
    (51222, "Hemoglobin", "Blood", "Hematology", "g/dL", 11, 2),
    (51221, "Hematocrit", "Blood", "Hematology", "%", 34, 6),
    (51301, "White Blood Cells", "Blood", "Hematology", "K/uL", 9, 4),
    (51265, "Platelet Count", "Blood", "Hematology", "K/uL", 220, 80),
    (51277, "RDW", "Blood", "Hematology", "%", 15, 2),
    (51003, "Troponin T", "Blood", "Chemistry", "ng/mL", 0.1, 0.2),
    (50813, "Lactate", "Blood", "Blood Gas", "mmol/L", 2, 1),
    (50862, "Albumin", "Blood", "Chemistry", "g/dL", 3.5, 0.5),
    (50960, "Magnesium", "Blood", "Chemistry", "mg/dL", 2, 0.3),
    (50885, "Bilirubin, Total", "Blood", "Chemistry", "mg/dL", 1, 0.8),
    (51237, "INR(PT)", "Blood", "Hematology", "", 1.4, 0.5),
]
VITALS = [  # itemid, label, unit, mean, sd
    (220045, "Heart Rate", "bpm", 85, 15), (220179, "Non Invasive Blood Pressure systolic", "mmHg", 120, 20),
    (220180, "Non Invasive Blood Pressure diastolic", "mmHg", 65, 12),
    (220181, "Non Invasive Blood Pressure mean", "mmHg", 80, 12),
    (220210, "Respiratory Rate", "insp/min", 19, 4), (220277, "O2 saturation pulseoxymetry", "%", 96, 3),
    (223761, "Temperature Fahrenheit", "°F", 98.4, 1),
]
HF10 = ["I5022", "I5023", "I5032", "I5033", "I509", "I110", "I130"]
HF9 = ["4280", "42822", "42832", "40291"]
OTHER10 = ["N184", "E119", "I480", "J449", "I10", "I2510", "J189", "N179"]
OTHER9 = ["5853", "25000", "42731", "496", "4019", "41401", "486", "5849"]
YEAR_GROUPS = ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022"]
ADM_TYPES = ["EW EMER.", "URGENT", "OBSERVATION ADMIT", "ELECTIVE", "DIRECT EMER.", "SURGICAL SAME DAY ADMISSION"]
DISCH = ["HOME", "HOME HEALTH CARE", "SKILLED NURSING FACILITY", "REHAB", "HOSPICE", "AGAINST ADVICE", "ACUTE HOSPITAL"]
ts = lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(x) else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sample_data/mimic_fake")
    ap.add_argument("--patients", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    (out / "hosp").mkdir(parents=True, exist_ok=True)
    (out / "icu").mkdir(parents=True, exist_ok=True)

    pats, adms, dxs, labs, icus, charts, omrs = [], [], [], [], [], [], []
    hadm, stay = 20_000_000, 30_000_000
    for i in range(a.patients):
        sid = 10_000_000 + i
        anchor_year = int(rng.integers(2110, 2190))
        anchor_age = int(rng.integers(18, 92))
        first = pd.Timestamp(f"{anchor_year}-01-01") + pd.Timedelta(days=int(rng.integers(0, 300)))
        n_adm = int(rng.integers(1, 5))
        t = first
        last_disch, died = None, False
        for k in range(n_adm):
            hadm += 1
            admit = t + pd.Timedelta(hours=float(rng.uniform(0, 12)))
            los_h = float(rng.gamma(2.5, 40)) + 2
            disch = admit + pd.Timedelta(hours=los_h)
            hf = rng.random() < 0.6
            death = (rng.random() < (0.12 if hf else 0.04))
            deathtime = disch if death else pd.NaT
            atype = rng.choice(ADM_TYPES, p=[.45, .15, .15, .1, .1, .05])
            dloc = "DIED" if death else rng.choice(DISCH, p=[.35, .25, .2, .08, .05, .04, .03])
            ed = admit - pd.Timedelta(hours=float(rng.uniform(1, 8))) if atype in ("EW EMER.", "URGENT") else pd.NaT
            adms.append([sid, hadm, ts(admit), ts(disch), ts(deathtime), atype, "", "EMERGENCY ROOM" if pd.notna(ed) else "PHYSICIAN REFERRAL",
                         dloc, rng.choice(["Medicare", "Medicaid", "Private", "Other"]), "English", "MARRIED",
                         rng.choice(["WHITE", "BLACK/AFRICAN AMERICAN", "HISPANIC/LATINO", "ASIAN", "OTHER"]),
                         ts(ed), ts(admit) if pd.notna(ed) else "", int(death)])
            v = 9 if rng.random() < 0.4 else 10
            codes = list(rng.choice(OTHER9 if v == 9 else OTHER10, size=int(rng.integers(2, 6)), replace=False))
            if hf:
                pos = int(rng.integers(0, len(codes) + 1))
                codes.insert(pos, rng.choice(HF9 if v == 9 else HF10))
            for s, c in enumerate(codes, 1):
                dxs.append([sid, hadm, s, c, v])
            for itemid, *_r, mu, sd in LABS:
                if rng.random() < (0.35 if itemid == 50963 else 0.85):
                    for _ in range(int(rng.integers(1, 4))):
                        ct = admit + pd.Timedelta(hours=float(rng.uniform(-6, min(los_h, 72))))
                        val = max(0.01, rng.normal(mu * (1.3 if death else 1), sd))
                        labs.append([len(labs) + 1, sid, hadm if rng.random() < .9 else "", "", itemid, "",
                                     ts(ct), ts(ct + pd.Timedelta(minutes=int(rng.integers(20, 240)))),
                                     f"{val:.2f}", round(val, 2), _r[3], "", "", "", "ROUTINE", ""])
            if rng.random() < 0.35:
                stay += 1
                intime = admit + pd.Timedelta(hours=float(rng.uniform(0, 20)))
                outtime = min(disch, intime + pd.Timedelta(hours=float(rng.uniform(12, 120))))
                icus.append([sid, hadm, stay, "Cardiac Vascular Intensive Care Unit (CVICU)", "Cardiac Vascular Intensive Care Unit (CVICU)",
                             ts(intime), ts(outtime), round((outtime - intime).total_seconds() / 86400, 3)])
                for itemid, label, unit, mu, sd in VITALS:
                    for h in range(0, int(min(24, (outtime - intime).total_seconds() / 3600)), 2):
                        ct = intime + pd.Timedelta(hours=h)
                        val = rng.normal(mu, sd)
                        charts.append([sid, hadm, stay, "", ts(ct), ts(ct), itemid, f"{val:.1f}", round(val, 1), unit, 0])
            last_disch = disch
            if death:
                died = True
                break
            t = disch + pd.Timedelta(days=float(rng.exponential(60)))
        for d in range(int(rng.integers(0, 6))):
            cd = first - pd.Timedelta(days=int(rng.integers(1, 400)))
            omrs.append([sid, cd.strftime("%Y-%m-%d"), 1, "Weight (Lbs)", f"{rng.normal(180, 30):.1f}"])
            omrs.append([sid, cd.strftime("%Y-%m-%d"), 1, "BMI (kg/m2)", f"{rng.normal(28, 5):.1f}"])
            omrs.append([sid, cd.strftime("%Y-%m-%d"), 1, "Blood Pressure", f"{int(rng.normal(130, 15))}/{int(rng.normal(75, 10))}"])
        dod = ""
        if died:
            dod = last_disch.strftime("%Y-%m-%d")
        elif rng.random() < 0.15:
            dod = (last_disch + pd.Timedelta(days=int(rng.integers(1, 360)))).strftime("%Y-%m-%d")
        yg = YEAR_GROUPS[int(rng.integers(0, len(YEAR_GROUPS)))]
        pats.append([sid, rng.choice(["M", "F"]), anchor_age, anchor_year, yg, dod])

    def w(rows, cols, path):
        pd.DataFrame(rows, columns=cols).to_csv(out / path, index=False, compression="gzip")

    w(pats, ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"], "hosp/patients.csv.gz")
    w(adms, ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "admission_type", "admit_provider_id",
             "admission_location", "discharge_location", "insurance", "language", "marital_status", "race",
             "edregtime", "edouttime", "hospital_expire_flag"], "hosp/admissions.csv.gz")
    w(dxs, ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"], "hosp/diagnoses_icd.csv.gz")
    codes = sorted({(r[3], r[4]) for r in dxs})
    w([[c, v, f"Synthetic description {c}"] for c, v in codes], ["icd_code", "icd_version", "long_title"], "hosp/d_icd_diagnoses.csv.gz")
    w([[i, l, f, c] for i, l, f, c, *_ in LABS], ["itemid", "label", "fluid", "category"], "hosp/d_labitems.csv.gz")
    w(labs, ["labevent_id", "subject_id", "hadm_id", "specimen_id", "itemid", "order_provider_id", "charttime",
             "storetime", "value", "valuenum", "valueuom", "ref_range_lower", "ref_range_upper", "flag", "priority",
             "comments"], "hosp/labevents.csv.gz")
    w(omrs, ["subject_id", "chartdate", "seq_num", "result_name", "result_value"], "hosp/omr.csv.gz")
    w(icus, ["subject_id", "hadm_id", "stay_id", "first_careunit", "last_careunit", "intime", "outtime", "los"], "icu/icustays.csv.gz")
    w([[i, l, l[:10], "chartevents", "Routine Vital Signs", u, "Numeric", "", ""] for i, l, u, *_ in VITALS],
      ["itemid", "label", "abbreviation", "linksto", "category", "unitname", "param_type", "lownormalvalue", "highnormalvalue"],
      "icu/d_items.csv.gz")
    w(charts, ["subject_id", "hadm_id", "stay_id", "caregiver_id", "charttime", "storetime", "itemid", "value",
               "valuenum", "valueuom", "warning"], "icu/chartevents.csv.gz")
    print(f"[OK] {len(pats)} bệnh nhân, {len(adms)} admission, {len(labs)} lab, {len(charts)} vital → {out}")


if __name__ == "__main__":
    main()

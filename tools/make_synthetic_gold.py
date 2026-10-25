"""Tạo bảng Gold P1 GIẢ đúng contracts/gold_schema_p1.json — để TV2–TV5 code song song
(train/evaluate, SHAP, API, UI) trước khi có Gold thật.

    python tools/make_synthetic_gold.py --rows 1000 --out sample_data/gold_p1_fake.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parents[1]
LAB_PARAMS = {"ntprobnp": (4000, 3000, .55), "creatinine": (1.5, .8, .05), "bun": (30, 15, .05),
              "sodium": (138, 4, .05), "potassium": (4.2, .6, .05), "chloride": (101, 5, .06),
              "bicarbonate": (25, 4, .06), "anion_gap": (14, 3, .08), "glucose": (140, 50, .05),
              "hemoglobin": (11, 2, .04), "hematocrit": (34, 6, .04), "wbc": (9, 4, .04),
              "platelet": (220, 80, .04), "rdw": (15, 2, .06), "troponin_t": (.1, .2, .4),
              "lactate": (2, 1, .6), "albumin": (3.5, .5, .5), "magnesium": (2, .3, .15),
              "bilirubin": (1, .8, .45), "inr": (1.4, .5, .3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--out", default="sample_data/gold_p1_fake.csv")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    schema = json.loads((ROOT / "contracts/gold_schema_p1.json").read_text(encoding="utf-8"))
    n = a.rows
    df = pd.DataFrame({
        "subject_id": 10_000_000 + rng.integers(0, int(n * .7), n),
        "hadm_id": 20_000_000 + np.arange(n),
    })
    df["admittime"] = pd.Timestamp("2150-01-01") + pd.to_timedelta(rng.integers(0, 3650 * 24, n), unit="h")
    df["prediction_time"] = df["admittime"] + pd.Timedelta(hours=24)
    df["anchor_year_group"] = rng.choice(["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022"], n)
    df["race_group"] = rng.choice(["WHITE", "BLACK", "HISPANIC", "ASIAN", "OTHER"], n, p=[.6, .15, .1, .05, .1])
    df["insurance"] = rng.choice(["Medicare", "Medicaid", "Private", "Other"], n, p=[.55, .15, .25, .05])
    df["hf_primary"] = (rng.random(n) < .3).astype(int)
    df["icd_version_group"] = rng.choice(["ICD9", "ICD10"], n)
    df["age"] = np.clip(rng.normal(72, 13, n), 18, 91).round(0)
    df["gender_male"] = (rng.random(n) < .52).astype(int)
    df["admission_type_group"] = rng.choice(["EMERGENCY", "URGENT", "OBSERVATION", "ELECTIVE"], n, p=[.6, .15, .15, .1])
    df["admission_location_group"] = rng.choice(["EMERGENCY_ROOM", "TRANSFER", "REFERRAL", "OTHER"], n, p=[.6, .15, .2, .05])
    df["came_from_ed"] = (df["admission_location_group"] == "EMERGENCY_ROOM").astype(int)
    df["in_icu_24h"] = (rng.random(n) < .3).astype(int)
    df["n_prior_adm_365d"] = rng.poisson(.8, n)
    df["n_prior_hf_adm"] = rng.poisson(.6, n)
    df["first_admission"] = (df["n_prior_adm_365d"] == 0).astype(int)
    df["days_since_last_discharge"] = np.where(df["first_admission"] == 1, np.nan, rng.exponential(120, n).round(1))
    for h, p in [("ckd", .35), ("diabetes", .4), ("afib", .4), ("copd", .2), ("hypertension", .7),
                 ("cad", .45), ("prior_mi", .15), ("stroke", .1)]:
        df[f"hx_{h}"] = (rng.random(n) < p).astype(int)
    for c, (mu, sd, miss) in [("omr_bmi_last", (28, 5, .5)), ("omr_weight_last", (180, 30, .5)),
                              ("omr_sbp_last", (130, 15, .5)), ("omr_egfr_last", (55, 20, .7))]:
        v = rng.normal(mu, sd, n).round(1)
        df[c] = np.where(rng.random(n) < miss, np.nan, v)
    risk = -3.2 + .03 * (df["age"] - 70) + .4 * df["in_icu_24h"] + .3 * df["hx_ckd"]
    for lab, (mu, sd, miss) in LAB_PARAMS.items():
        first = np.abs(rng.normal(mu, sd, n))
        lo, hi = first * rng.uniform(.8, 1, n), first * rng.uniform(1, 1.3, n)
        m = rng.random(n) < miss
        df[f"{lab}_first"] = np.where(m, np.nan, first.round(3))
        df[f"{lab}_min"] = np.where(m, np.nan, lo.round(3))
        df[f"{lab}_max"] = np.where(m, np.nan, hi.round(3))
        df[f"{lab}_missing"] = m.astype(int)
    risk += .02 * np.nan_to_num(df["bun_max"] - 30) + .6 * np.nan_to_num(df["lactate_max"] - 2)
    df["n_lab_tests_24h"] = rng.poisson(35, n)
    icu = df["in_icu_24h"] == 1
    for c, (mu, sd) in {"hr_mean": (85, 12), "hr_max": (105, 15), "sbp_min": (95, 15), "sbp_mean": (118, 15),
                        "dbp_mean": (63, 10), "mbp_min": (62, 10), "rr_mean": (19, 3), "rr_max": (27, 5),
                        "spo2_min": (91, 4), "temp_max_c": (37.3, .6)}.items():
        df[c] = np.where(icu, rng.normal(mu, sd, n).round(1), np.nan)
    df["vitals_missing"] = (~icu).astype(int)
    df["label_mortality"] = (rng.random(n) < 1 / (1 + np.exp(-risk))).astype(int)

    # kiểm tra khớp schema
    expected = [c["name"] for c in schema["id_columns"] + schema["label_columns"] + schema["subgroup_only_columns"]]
    fc = schema["feature_columns"]
    expected += [c["name"] for c in fc["demographic_admission"] + fc["history"] + fc["vitals_icu"]]
    expected += [f"{l}{s}" for l in fc["labs"]["items"] for s in fc["labs"]["suffixes"]] + ["n_lab_tests_24h"]
    missing = set(expected) - set(df.columns)
    extra = set(df.columns) - set(expected)
    assert not missing and not extra, f"Lệch schema: thiếu {missing}, thừa {extra}"
    df = df[expected]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"[OK] {len(df)} dòng × {df.shape[1]} cột, tỷ lệ dương {df['label_mortality'].mean():.3f} → {a.out}")


if __name__ == "__main__":
    main()

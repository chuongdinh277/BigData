"""Fixture quality từ MIMIC GIẢ đã sinh; không phải pipeline cohort/features.

Tạo một tập con P1 có 20 lab, sinh hiệu và lineage từ đúng các event đã dùng.
Chỉ cho phép đọc thư mục có tên mimic_fake để tránh dùng nhầm dữ liệu thật.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
TRACE_COLUMNS = ["subject_id", "hadm_id", "feature_name", "source_table", "source_subject_id",
                 "source_hadm_id", "source_charttime", "source_storetime", "source_dischtime"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="sample_data/mimic_fake")
    ap.add_argument("--out", default="sample_data/quality_fixture")
    ap.add_argument("--rows", type=int, default=100)
    ap.add_argument("--inject-errors", action="store_true")
    args = ap.parse_args()
    raw = Path(args.root)
    if raw.name != "mimic_fake":
        raise SystemExit("Fixture chỉ đọc mimic_fake; không chạy công cụ này trên dữ liệu thật")
    cfg = yaml.safe_load((ROOT / "configs/feature_items.yaml").read_text())
    patients = pd.read_csv(raw / "hosp/patients.csv.gz")
    admissions = pd.read_csv(raw / "hosp/admissions.csv.gz", parse_dates=["admittime", "dischtime"])
    admissions["prediction_time"] = admissions["admittime"] + pd.Timedelta(hours=24)
    admissions = admissions.loc[admissions["dischtime"] > admissions["prediction_time"]].head(args.rows).merge(patients, on="subject_id")
    labs = pd.read_csv(raw / "hosp/labevents.csv.gz", parse_dates=["charttime", "storetime"])
    charts = pd.read_csv(raw / "icu/chartevents.csv.gz", parse_dates=["charttime", "storetime"])
    stays = pd.read_csv(raw / "icu/icustays.csv.gz", parse_dates=["intime", "outtime"])
    rows, traces = [], []

    def record_trace(adm, name, source, events):
        for event in events.itertuples():
            traces.append([int(adm.subject_id), int(adm.hadm_id), name, source, int(event.subject_id),
                           None if pd.isna(event.hadm_id) else int(event.hadm_id),
                           event.charttime if hasattr(event, "charttime") else event.intime,
                           event.storetime if hasattr(event, "storetime") else pd.NaT, pd.NaT])

    for adm in admissions.itertuples():
        row = {"subject_id": int(adm.subject_id), "hadm_id": int(adm.hadm_id), "admittime": adm.admittime,
               "prediction_time": adm.prediction_time, "anchor_year_group": adm.anchor_year_group,
               "age": float(adm.anchor_age + adm.admittime.year - adm.anchor_year),
               "gender_male": int(adm.gender == "M"), "label_mortality": int(adm.hospital_expire_flag)}
        patient_labs = labs.loc[(labs.subject_id == adm.subject_id)
                               & (labs.hadm_id.isna() | (labs.hadm_id == adm.hadm_id))
                               & labs.charttime.between(adm.admittime - pd.Timedelta(hours=6), adm.prediction_time)
                               & (labs.storetime <= adm.prediction_time)]
        count = 0
        for name, spec in cfg["labs"].items():
            events = patient_labs.loc[patient_labs.itemid.isin(spec["itemids"])
                                      & (patient_labs.valueuom.fillna("") == spec["unit_hint"])
                                      & patient_labs.valuenum.between(*spec["valid_range"])].sort_values(["charttime", "labevent_id"])
            count += len(events)
            row[name + "_missing"] = int(events.empty)
            for suffix, value in (("first", events.valuenum.iloc[0] if len(events) else None),
                                  ("min", events.valuenum.min() if len(events) else None),
                                  ("max", events.valuenum.max() if len(events) else None)):
                col = f"{name}_{suffix}"
                row[col] = value
                # Cả first/min/max: giữ mọi event đầu vào để kiểm tra cửa sổ.
                record_trace(adm, col, "labevents", events)
            record_trace(adm, "n_lab_tests_24h", "labevents", events)
        row["n_lab_tests_24h"] = count
        icu = stays.loc[(stays.hadm_id == adm.hadm_id) & stays.intime.between(adm.admittime, adm.prediction_time)]
        row["in_icu_24h"] = int(not icu.empty)
        record_trace(adm, "in_icu_24h", "icustays", icu)
        patient_charts = charts.loc[(charts.hadm_id == adm.hadm_id)
                                   & charts.charttime.between(adm.admittime, adm.prediction_time)
                                   & (charts.storetime <= adm.prediction_time)]
        any_vital = False
        for name, spec in cfg["vitals"].items():
            events = patient_charts.loc[patient_charts.itemid.isin(spec["itemids"])
                                        & (patient_charts.valueuom.fillna("") == spec["unit_hint"])].copy()
            if spec.get("conversion") == "fahrenheit_to_celsius":
                events["valuenum"] = (events.valuenum - 32) * 5 / 9
            events = events.loc[events.valuenum.between(*spec["valid_range"])]
            any_vital |= not events.empty
            for col in spec["feature_columns"]:
                agg = col.rsplit("_", 1)[1] if col != "temp_max_c" else "max"
                row[col] = getattr(events.valuenum, agg)() if len(events) else None
                record_trace(adm, col, "chartevents", events)
        row["vitals_missing"] = int(not any_vital)
        rows.append(row)

    features = pd.DataFrame(rows)
    lineage = pd.DataFrame(traces, columns=TRACE_COLUMNS)
    for col in ["source_charttime", "source_storetime", "source_dischtime"]:
        lineage[col] = pd.to_datetime(lineage[col])
    lineage["source_hadm_id"] = lineage["source_hadm_id"].astype("Int64")
    if args.inject_errors:
        features.loc[features.index[0], "sodium_first"] = 999.0
        features = pd.concat([features, features.iloc[[0]]], ignore_index=True)
        extra = lineage.iloc[[0]].copy()
        prediction = features.loc[features.hadm_id == extra.hadm_id.iloc[0], "prediction_time"].iloc[0]
        extra["source_charttime"] = prediction + pd.Timedelta(seconds=1)
        extra["source_storetime"] = prediction + pd.Timedelta(seconds=2)
        lineage = pd.concat([lineage, extra], ignore_index=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out / "features.parquet", index=False)
    admissions[["subject_id", "hadm_id", "admittime", "dischtime"]].to_parquet(out / "context.parquet", index=False)
    lineage.to_parquet(out / "lineage.parquet", index=False)
    print(f"[OK] Synthetic fixture: {len(features)} dòng, {len(features.columns)} cột, {len(lineage)} provenance rows → {out}")


if __name__ == "__main__":
    main()

"""Kiểm thử Gold P2/P3 (mốc = dischtime): data_spark/features_discharge.py + quality_discharge.py.

Dựng bảng nhỏ bằng createDataFrame hoặc MIMIC giả (tools/make_synthetic_mimic.py); không đọc S3.
"""
import json
import subprocess
import sys
from datetime import datetime, timedelta

import pytest

from data_spark import features as f1
from data_spark import features_discharge as fd
from data_spark.common.io import REPO_ROOT, load_yaml
from data_spark.quality_discharge import check_discharge_gold

ADMIT = datetime(2150, 1, 1)
DISCH = ADMIT + timedelta(days=5)


@pytest.mark.parametrize("task,label", [("p2", "label_readmit30"), ("p3", "label_mort90")])
def test_contract_p2_p3_extends_p1_with_discharge_features(task, label):
    schema = json.loads(fd.schema_path(task).read_text(encoding="utf-8"))
    names = [n for n, _ in f1.schema_columns(schema)]
    assert len(names) == len(set(names))
    assert schema["label"] == label and label in names and "label_mortality" not in names
    for c in ("los_days", "icu_days", "discharge_location_group", "n_diagnoses", "dx_ckd", "icu_any",
              "creatinine_last", "n_lab_tests_stay"):
        assert c in names
    assert "in_icu_24h" not in names and "n_lab_tests_24h" not in names
    assert not set(names) & set(schema["forbidden_columns"])
    assert {"dod", "deathtime", "hospital_expire_flag"} <= set(schema["forbidden_columns"])


def test_build_base_uses_label_population_and_dischtime(spark):
    cohort = spark.createDataFrame(
        [(1, 10, ADMIT, DISCH), (2, 20, ADMIT, DISCH)],
        "subject_id long, hadm_id long, admittime timestamp, dischtime timestamp")
    labels = spark.createDataFrame([(20, 1)], "hadm_id long, label_readmit30 int")
    rows = fd.build_base(cohort, labels, "label_readmit30").collect()
    assert [(r.hadm_id, r.label_readmit30, r.prediction_time) for r in rows] == [(20, 1, DISCH)]


def test_lab_last_is_latest_value_in_final_48h_and_p1_unchanged(spark):
    cfg = {"creatinine": {"itemids": [50912], "valid_range": [0.1, 25]}}
    base = spark.createDataFrame([(1, 10, ADMIT, DISCH)],
                                 "subject_id long, hadm_id long, admittime timestamp, prediction_time timestamp")
    ev = spark.createDataFrame([
        (1, 50912, ADMIT + timedelta(hours=2), ADMIT + timedelta(hours=3), 3.0, "mg/dL"),     # first, max
        (1, 50912, DISCH - timedelta(hours=40), DISCH - timedelta(hours=39), 1.2, "mg/dL"),   # trong 48h
        (1, 50912, DISCH - timedelta(hours=10), DISCH - timedelta(hours=9), 1.0, "mg/dL"),    # last
        (1, 50912, DISCH + timedelta(hours=1), DISCH + timedelta(hours=2), 9.0, "mg/dL"),     # sau xuất viện
    ], "subject_id long, itemid long, charttime timestamp, storetime timestamp, valuenum double, valueuom string")
    lw = f1.lab_window(ev, base, strict_store=True)
    wide, _ = f1.lab_features(lw, cfg, last_hours=48)
    r = wide.first()
    assert (r.creatinine_first, r.creatinine_min, r.creatinine_max, r.creatinine_last) == (3.0, 1.0, 3.0, 1.0)
    p1_cols = f1.lab_features(lw, cfg)[0].columns
    assert "creatinine_last" not in p1_cols and "creatinine_first" in p1_cols


def test_lab_last_is_null_when_no_value_in_final_48h(spark):
    cfg = {"bun": {"itemids": [51006], "valid_range": [1, 300]}}
    base = spark.createDataFrame([(1, 10, ADMIT, DISCH)],
                                 "subject_id long, hadm_id long, admittime timestamp, prediction_time timestamp")
    ev = spark.createDataFrame([(1, 51006, ADMIT + timedelta(hours=1), None, 30.0, "mg/dL")],
                               "subject_id long, itemid long, charttime timestamp, storetime timestamp, "
                               "valuenum double, valueuom string")
    r = f1.lab_features(f1.lab_window(ev, base, True), cfg, last_hours=48)[0].first()
    assert r.bun_first == 30.0 and r.bun_last is None


def test_stay_features_los_icu_clipped_discharge_group_and_current_dx(spark):
    base = spark.createDataFrame(
        [(10, ADMIT, DISCH, "SKILLED NURSING FACILITY"), (20, ADMIT, ADMIT + timedelta(days=2), "HOME")],
        "hadm_id long, admittime timestamp, dischtime timestamp, discharge_location string")
    icu = spark.createDataFrame(
        [(10, ADMIT - timedelta(days=1), ADMIT + timedelta(days=1)),   # 1 ngày trong đợt
         (10, DISCH - timedelta(days=1), DISCH + timedelta(days=3))],  # cắt ở dischtime → 1 ngày
        "hadm_id long, intime timestamp, outtime timestamp")
    dx = spark.createDataFrame([(10, "N185", 10), (10, "I5022", 10), (20, "4280", 9)],
                               "hadm_id long, icd_code string, icd_version int")
    comorb = load_yaml("configs/hf_codes.yaml")["comorbidities"]
    rows = {r.hadm_id: r for r in fd.stay_features(base, icu, dx, comorb).collect()}
    assert rows[10].los_days == pytest.approx(5.0) and rows[10].icu_days == pytest.approx(2.0)
    assert rows[10].discharge_location_group == "FACILITY" and rows[20].discharge_location_group == "HOME"
    assert rows[10].dx_ckd == 1 and rows[20].dx_ckd == 0 and rows[10].n_diagnoses == 2
    assert rows[20].icu_days == 0


@pytest.mark.parametrize("task", ["p2", "p3"])
def test_end_to_end_on_synthetic_mimic_passes_and_detects_leakage(spark, tmp_path, task):
    from pyspark.sql import functions as F
    from data_spark import cohort as cohort_mod
    from data_spark import labels as labels_mod
    from data_spark.common.io import read_table

    mimic_dir = tmp_path / "mimic_fake"
    subprocess.run([sys.executable, str(REPO_ROOT / "tools/make_synthetic_mimic.py"),
                    "--patients", "80", "--seed", "11", "--out", str(mimic_dir)], check=True)
    rt = lambda name: read_table(spark, str(mimic_dir), name, "bronze")  # noqa: E731
    hfc = load_yaml("configs/hf_codes.yaml")
    cohort, _ = cohort_mod.build_cohort(rt("admissions"), rt("patients"), rt("diagnoses_icd"), hfc["heart_failure"])
    cohort = cohort.cache()
    tables, _ = labels_mod.build_labels(cohort, rt("admissions"), rt("patients"))
    labels = tables[task].cache()
    gold, evidence, _, n_base = fd.build(spark, task=task, root=str(mimic_dir), fmt="bronze",
                                         cohort=cohort, labels=labels)
    gold = gold.cache()
    label = fd.TASKS[task]
    assert gold.count() == n_base == labels.join(cohort.select("hadm_id"), "hadm_id").count() > 0
    rep = check_discharge_gold(gold, task=task, cohort=cohort, labels=labels, evidence=evidence)
    assert rep["status"] == "PASSED", rep
    assert gold.filter(F.col("prediction_time") <= F.col("admittime")).count() == 0
    assert set(r[0] for r in gold.select(label).distinct().collect()) <= {0, 1}

    # Cố tình gài một event sau xuất viện và sai mốc dự đoán → phải FAILED
    bad_ev = evidence.withColumn("max_charttime", F.col("max_charttime") + F.expr("INTERVAL 30 DAYS"))
    assert check_discharge_gold(gold, task=task, cohort=cohort, labels=labels, evidence=bad_ev)["status"] == "FAILED"
    shifted = gold.withColumn("prediction_time", F.col("prediction_time") - F.expr("INTERVAL 1 HOURS"))
    rep2 = check_discharge_gold(shifted, task=task, cohort=cohort, labels=labels)
    assert rep2["status"] == "FAILED" and rep2["counts"]["prediction_not_dischtime_rows"] > 0
    flipped = gold.withColumn(label, 1 - F.col(label))
    assert check_discharge_gold(flipped, task=task, cohort=cohort, labels=labels)["counts"]["label_mismatch_rows"] > 0


def test_pipeline_runner_routes_p2_p3_to_discharge_features():
    for task in ("p2", "p3"):
        r = subprocess.run([sys.executable, "-m", "data_spark.pipeline_runner", "--dry-run", "--task", task,
                            "--steps", "features", "--sample-frac", "0.01"],
                           cwd=REPO_ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert f"data_spark.features_discharge --task {task}" in r.stdout and "--sample-frac 0.01" in r.stdout

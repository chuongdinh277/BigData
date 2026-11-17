"""Kiểm thử âm: tương lai, cột cấm, thiếu bằng chứng và tiền sử sai admission."""
from datetime import datetime, timedelta

import pytest

from data_spark.quality import check_leakage, check_prediction_times

ADMIT = datetime(2150, 1, 2)
PREDICT = ADMIT + timedelta(hours=24)
LINEAGE_SCHEMA = ("subject_id long, hadm_id long, feature_name string, source_table string, "
                  "source_subject_id long, source_hadm_id long, source_charttime timestamp, "
                  "source_storetime timestamp, source_dischtime timestamp")


def features(spark, feature="sodium_first", value=138.0):
    return spark.createDataFrame([(1, 10, ADMIT, PREDICT, float(value) if value is not None else None)],
                                 f"subject_id long, hadm_id long, admittime timestamp, prediction_time timestamp, {feature} double")


def lineage(spark, **changes):
    row = dict(subject_id=1, hadm_id=10, feature_name="sodium_first", source_table="labevents",
               source_subject_id=1, source_hadm_id=10, source_charttime=PREDICT,
               source_storetime=PREDICT, source_dischtime=None)
    row.update(changes)
    return spark.createDataFrame([tuple(row.values())], LINEAGE_SCHEMA)


@pytest.mark.parametrize("offset", [-6, 0, 24])
def test_labs_accept_inclusive_window_boundaries(spark, offset):
    event_time = ADMIT + timedelta(hours=offset)
    result = check_leakage(features(spark), lineage(spark, source_charttime=event_time, source_storetime=event_time))
    assert result["status"] == "PASSED"


@pytest.mark.parametrize("changes,metric", [
    ({"source_charttime": PREDICT + timedelta(seconds=1), "source_storetime": PREDICT + timedelta(seconds=1)}, "future_charttime_rows"),
    ({"source_storetime": PREDICT + timedelta(seconds=1)}, "late_storetime_rows"),
    ({"source_charttime": ADMIT - timedelta(hours=6, seconds=1)}, "outside_window_rows"),
    ({"source_charttime": None}, "missing_timestamp_rows"),
    ({"source_subject_id": 2}, "wrong_patient_rows"),
    ({"source_hadm_id": 11}, "wrong_admission_rows"),
    ({"source_table": "diagnoses_icd"}, "invalid_source_rows"),
    ({"hadm_id": 99}, "orphan_lineage_rows"),
])
def test_invalid_source_evidence_fails(spark, changes, metric):
    result = check_leakage(features(spark), lineage(spark, **changes))
    assert result["status"] == "FAILED"
    assert result[metric] == 1


def test_null_storetime_is_accepted_when_charttime_valid(spark):
    lab_res = check_leakage(features(spark), lineage(spark, source_storetime=None))
    assert lab_res["status"] == "PASSED"
    vital_res = check_leakage(
        features(spark, "hr_mean", 80),
        lineage(spark, feature_name="hr_mean", source_table="chartevents", source_charttime=ADMIT, source_storetime=None),
    )
    assert vital_res["status"] == "PASSED"
    missing_ct = check_leakage(features(spark), lineage(spark, source_charttime=None, source_storetime=None))
    assert missing_ct["status"] == "FAILED"
    assert missing_ct["missing_timestamp_rows"] == 1


@pytest.mark.parametrize("name", ["dischtime", "deathtime", "discharge_location", "hospital_expire_flag", "dod", "los", "outtime", "last_careunit", "DISCHTIME"])
def test_forbidden_columns_never_enter_p1(spark, name):
    from pyspark.sql import functions as F
    result = check_leakage(features(spark).withColumn(name, F.lit("forbidden")))
    assert result["status"] == "FAILED"
    assert name in result["forbidden_columns"]


def test_unknown_alias_and_subgroup_model_input_fail(spark):
    from pyspark.sql import functions as F
    assert check_leakage(features(spark).withColumn("future_los", F.lit(2)))["status"] == "FAILED"
    assert check_leakage(features(spark), model_columns=["sodium_first", "label_mortality", "hf_primary"])["status"] == "FAILED"


def test_missing_or_empty_lineage_cannot_pass(spark):
    df = features(spark)
    assert check_leakage(df)["status"] == "NOT_CHECKED"
    result = check_leakage(df, spark.createDataFrame([], LINEAGE_SCHEMA))
    assert result["status"] == "FAILED"
    assert result["uncovered_feature_values"] == 1


@pytest.mark.parametrize("source_id,discharge,expected", [
    (9, ADMIT - timedelta(seconds=1), "PASSED"),
    (10, ADMIT - timedelta(seconds=1), "FAILED"),
    (9, ADMIT, "FAILED"),
    (9, ADMIT + timedelta(seconds=1), "FAILED"),
])
def test_history_must_be_discharged_before_current_admission(spark, source_id, discharge, expected):
    trace = lineage(spark, feature_name="hx_ckd", source_table="diagnoses_icd", source_hadm_id=source_id,
                    source_charttime=ADMIT - timedelta(days=2), source_storetime=None, source_dischtime=discharge)
    assert check_leakage(features(spark, "hx_ckd", 1), trace)["status"] == expected


@pytest.mark.parametrize("hours,expected", [(-1, "FAILED"), (0, "PASSED"), (24, "PASSED"), (25, "FAILED")])
def test_vitals_do_not_use_lab_lookback(spark, hours, expected):
    at = ADMIT + timedelta(hours=hours)
    trace = lineage(spark, feature_name="hr_mean", source_table="chartevents", source_charttime=at, source_storetime=at)
    assert check_leakage(features(spark, "hr_mean", 80), trace)["status"] == expected


@pytest.mark.parametrize("hours,expected", [(-1, "PASSED"), (25, "FAILED")])
def test_icu_presence_can_start_before_admission_but_not_after_prediction(spark, hours, expected):
    trace = lineage(spark, feature_name="in_icu_24h", source_table="icustays",
                    source_charttime=ADMIT + timedelta(hours=hours), source_storetime=None)
    assert check_leakage(features(spark, "in_icu_24h", 1), trace)["status"] == expected


@pytest.mark.parametrize("days,expected", [(-366, "FAILED"), (-365, "PASSED"), (-1, "PASSED"), (0, "FAILED")])
def test_omr_uses_previous_dates_within_one_year(spark, days, expected):
    trace = lineage(spark, feature_name="omr_bmi_last", source_table="omr", source_hadm_id=None,
                    source_charttime=ADMIT + timedelta(days=days), source_storetime=None)
    assert check_leakage(features(spark, "omr_bmi_last", 25), trace)["status"] == expected


@pytest.mark.parametrize("discharge,expected", [(PREDICT, "FAILED"), (PREDICT + timedelta(seconds=1), "PASSED")])
def test_discharge_must_be_strictly_after_prediction(spark, discharge, expected):
    context = spark.createDataFrame([(1, 10, ADMIT, discharge)],
                                    "subject_id long, hadm_id long, admittime timestamp, dischtime timestamp")
    assert check_prediction_times(features(spark), context)["status"] == expected
    assert check_prediction_times(features(spark))["status"] == "NOT_CHECKED"


def test_wrong_prediction_offset_and_duplicate_context_fail(spark):
    from pyspark.sql import functions as F
    context = spark.createDataFrame([(1, 10, ADMIT, PREDICT + timedelta(days=1))],
                                    "subject_id long, hadm_id long, admittime timestamp, dischtime timestamp")
    df = features(spark).withColumn("prediction_time", F.lit(ADMIT + timedelta(hours=23)))
    assert check_prediction_times(df, context)["wrong_prediction_offset_rows"] == 1
    assert check_prediction_times(features(spark), context.unionByName(context))["status"] == "FAILED"


def test_build_lineage_handles_null_and_clock_skew_storetime(spark, tmp_path):
    from data_spark import features as feat_mod
    from data_spark.common.io import load_yaml
    from data_spark.quality import build_lineage

    root = tmp_path / "silver"
    admit = ADMIT
    # 1 bệnh nhân, 1 ca đủ điều kiện P1, 1 lab storetime=NULL và 1 vital storetime < charttime (đều <= prediction_time)
    spark.createDataFrame(
        [(1, 10, admit, admit + timedelta(days=3), None, "EW EMER.", "EMERGENCY ROOM", "HOME", "Medicare", "WHITE", admit, admit, 0)],
        "subject_id long, hadm_id long, admittime timestamp, dischtime timestamp, deathtime timestamp, "
        "admission_type string, admission_location string, discharge_location string, insurance string, race string, "
        "edregtime timestamp, edouttime timestamp, hospital_expire_flag int",
    ).write.mode("overwrite").parquet(str(root / "hosp/admissions"))
    spark.createDataFrame([], "subject_id long, hadm_id long, seq_num int, icd_code string, icd_version int") \
        .write.mode("overwrite").parquet(str(root / "hosp/diagnoses_icd"))
    spark.createDataFrame([], "subject_id long, chartdate date, seq_num int, result_name string, result_value string") \
        .write.mode("overwrite").parquet(str(root / "hosp/omr"))
    spark.createDataFrame([], "subject_id long, hadm_id long, stay_id long, intime timestamp, outtime timestamp, first_careunit string") \
        .write.mode("overwrite").parquet(str(root / "icu/icustays"))
    spark.createDataFrame(
        [(1, 10, 50983, admit + timedelta(hours=2), None, "138", 138.0, "mEq/L")],
        "subject_id long, hadm_id long, itemid int, charttime timestamp, storetime timestamp, value string, valuenum double, valueuom string",
    ).write.mode("overwrite").parquet(str(root / "hosp/labevents"))
    spark.createDataFrame(
        [(1, 10, 100, 220045, admit + timedelta(hours=4), admit + timedelta(hours=3, minutes=55), "82", 82.0, "bpm")],
        "subject_id long, hadm_id long, stay_id long, itemid int, charttime timestamp, storetime timestamp, value string, valuenum double, valueuom string",
    ).write.mode("overwrite").parquet(str(root / "icu/chartevents"))

    items = load_yaml("configs/feature_items.yaml")
    base = spark.createDataFrame([(1, 10, admit, PREDICT)], "subject_id long, hadm_id long, admittime timestamp, prediction_time timestamp")
    lw = feat_mod.lab_window(spark.read.parquet(str(root / "hosp/labevents")), base, strict_store=True)
    labs, _ = feat_mod.lab_features(lw, items["labs"])
    vit, _, _ = feat_mod.vitals_features(spark.read.parquet(str(root / "icu/chartevents")), base, items["vitals"], strict_store=True)
    feat_df = base.join(labs.select("hadm_id", "sodium_first"), "hadm_id", "left").join(vit.select("hadm_id", "hr_mean"), "hadm_id", "left")

    lin = build_lineage(spark, feat_df, root=str(root), fmt="silver", strict_store=True)
    res = check_leakage(feat_df, lin)
    assert res["status"] == "PASSED", res

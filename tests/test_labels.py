from datetime import date, datetime, timedelta

import pytest

from data_spark import labels
from data_spark.labels import build_labels, load_or_build_cohort, write_report


def _tables(spark, cohort_rows, admission_rows=None, patient_rows=None):
    cohort = spark.createDataFrame(
        cohort_rows,
        "subject_id long, hadm_id long, admittime timestamp, dischtime timestamp, "
        "deathtime timestamp, hospital_expire_flag int, discharge_location string",
    )
    admission_rows = admission_rows or [
        (row[0], row[1], row[2], "EMERGENCY")
        for row in cohort_rows
    ]
    admissions = spark.createDataFrame(
        admission_rows,
        "subject_id long, hadm_id long, admittime timestamp, admission_type string",
    )
    patient_rows = patient_rows or [(row[0], None) for row in cohort_rows]
    patients = spark.createDataFrame(
        patient_rows, "subject_id long, dod date"
    )
    return cohort, admissions, patients


def test_input_root_defaults_to_selected_format_path():
    paths = {"bronze": "bronze-root", "silver": "silver-root"}

    assert labels._resolve_input_root(None, "bronze", paths) == "bronze-root"
    assert labels._resolve_input_root(None, "silver", paths) == "silver-root"
    assert labels._resolve_input_root("custom-root", "bronze", paths) == "custom-root"


def test_p1_uses_admission_plus_24h_and_excludes_early_discharge_or_death(spark):
    admit = datetime(2150, 1, 2)
    cohort_rows = [
        (1, 101, admit, admit + timedelta(days=3), admit + timedelta(hours=36), 1, "DIED"),
        (2, 102, admit, admit + timedelta(days=2), None, 0, "HOME"),
        (3, 103, admit, admit + timedelta(hours=24), None, 0, "HOME"),
        (4, 104, admit, admit + timedelta(days=2), admit + timedelta(hours=24), 1, "DIED"),
        (5, 105, admit, admit + timedelta(days=2), admit + timedelta(hours=23), 1, "DIED"),
    ]
    cohort, admissions, patients = _tables(spark, cohort_rows)

    labels, stats = build_labels(cohort, admissions, patients)
    rows = {r.hadm_id: r.asDict() for r in labels["p1"].collect()}

    assert set(rows) == {101, 102}
    assert rows[101]["prediction_time"] == admit + timedelta(hours=24)
    assert rows[101]["label_mortality"] == 1
    assert rows[102]["label_mortality"] == 0
    assert stats["P1"]["excluded"] == {
        "invalid_admission_or_discharge_time": 0,
        "discharged_by_prediction_time": 1,
        "died_by_prediction_time": 2,
    }


def test_p2_includes_day_30_unplanned_readmission_and_excludes_censored_or_planned(spark):
    discharge = datetime(2150, 1, 10)
    cohort_rows = [
        (1, 201, discharge - timedelta(days=2), discharge, None, 0, "HOME"),
        (2, 202, discharge - timedelta(days=2), discharge, None, 0, "HOME"),
        (3, 203, discharge - timedelta(days=2), discharge, None, 0, "HOME"),
        (4, 204, discharge - timedelta(days=2), discharge, None, 0, "HOME"),
        (5, 205, discharge - timedelta(days=2), discharge, None, 0, "ACUTE HOSPITAL"),
        (6, 206, discharge - timedelta(days=2), discharge, discharge, 1, "DIED"),
    ]
    future = [
        (1, 301, discharge + timedelta(days=30), "URGENT"),
        (2, 302, discharge + timedelta(days=30), "ELECTIVE"),
        (3, 303, discharge + timedelta(days=31), "EMERGENCY"),
    ]
    all_admissions = [
        (r[0], r[1], r[2], "EMERGENCY") for r in cohort_rows
    ] + future
    cohort, admissions, patients = _tables(
        spark,
        cohort_rows,
        admission_rows=all_admissions,
        patient_rows=[
            (1, None), (2, None), (3, None), (4, date(2150, 1, 20)),
            (5, None), (6, None),
        ],
    )

    labels, stats = build_labels(cohort, admissions, patients)
    rows = {r.hadm_id: r.asDict() for r in labels["p2"].collect()}

    assert set(rows) == {201, 202, 203}
    assert rows[201]["label_readmit30"] == 1
    assert rows[201]["prediction_time"] == discharge
    assert rows[202]["label_readmit30"] == 0
    assert rows[203]["label_readmit30"] == 0
    assert stats["P2"]["excluded"]["transfer_ama_or_hospice"] == 1
    assert stats["P2"]["excluded"]["died_in_hospital"] == 1
    assert stats["P2"]["excluded"]["death_within_window_without_readmission"] == 1


def test_p3_uses_date_window_days_1_through_90_and_excludes_hospice(spark):
    discharge = datetime(2150, 1, 10, 12)
    cohort_rows = [
        (1, 401, discharge - timedelta(days=2), discharge, None, 0, "HOME"),
        (2, 402, discharge - timedelta(days=2), discharge, None, 0, "HOME"),
        (3, 403, discharge - timedelta(days=2), discharge, None, 0, "HOME"),
        (4, 404, discharge - timedelta(days=2), discharge, None, 0, "HOME"),
        (5, 405, discharge - timedelta(days=2), discharge, None, 0, "HOSPICE"),
    ]
    cohort, admissions, patients = _tables(
        spark,
        cohort_rows,
        patient_rows=[
            (1, date(2150, 1, 10)),
            (2, date(2150, 1, 11)),
            (3, date(2150, 4, 10)),
            (4, date(2150, 4, 11)),
            (5, date(2150, 2, 1)),
        ],
    )

    labels, stats = build_labels(cohort, admissions, patients)
    rows = {r.hadm_id: r.asDict() for r in labels["p3"].collect()}

    assert set(rows) == {401, 402, 403, 404}
    assert rows[401]["label_mort90"] == 0
    assert rows[402]["label_mort90"] == 1
    assert rows[403]["label_mort90"] == 1
    assert rows[404]["label_mort90"] == 0
    assert stats["P3"]["excluded"]["hospice"] == 1


def test_labels_reject_duplicate_admissions_and_invalid_window(spark):
    admit = datetime(2150, 1, 2)
    cohort_rows = [
        (1, 501, admit, admit + timedelta(days=2), None, 0, "HOME"),
        (1, 501, admit, admit + timedelta(days=2), None, 0, "HOME"),
    ]
    cohort, admissions, patients = _tables(spark, cohort_rows)

    with pytest.raises(ValueError, match="cohort.hadm_id không duy nhất"):
        build_labels(cohort, admissions, patients)

    unique_cohort, admissions, patients = _tables(spark, cohort_rows[:1])
    with pytest.raises(ValueError, match="p2_window_days"):
        build_labels(unique_cohort, admissions, patients, p2_window_days=0)


def test_labels_build_cohort_in_memory_without_reading_saved_parquet(spark, monkeypatch):
    admit = datetime(2150, 1, 2)
    admissions = spark.createDataFrame(
        [(
            1, 601, admit, admit + timedelta(days=2), None, 0, "HOME",
            "EMERGENCY", "EMERGENCY ROOM", None, "Medicare", "WHITE",
        )],
        "subject_id long, hadm_id long, admittime timestamp, dischtime timestamp, "
        "deathtime timestamp, hospital_expire_flag int, discharge_location string, "
        "admission_type string, admission_location string, edregtime timestamp, "
        "insurance string, race string",
    )
    patients = spark.createDataFrame(
        [(1, "M", 60, 2140, "2011 - 2013")],
        "subject_id long, gender string, anchor_age int, anchor_year int, anchor_year_group string",
    )
    diagnoses = spark.createDataFrame(
        [(1, 601, 1, "I509", 10)],
        "subject_id long, hadm_id long, seq_num int, icd_code string, icd_version int",
    )
    calls = []

    def fake_read_table(session, root, table, fmt):
        calls.append((root, table, fmt))
        assert table == "diagnoses_icd"
        return diagnoses

    monkeypatch.setattr(labels, "read_table", fake_read_table)

    cohort = load_or_build_cohort(
        spark,
        admissions,
        patients,
        root="silver-root",
        fmt="silver",
        hf_codes={"icd10": ["I50"], "icd9": []},
    )

    assert cohort.select("hadm_id").collect()[0].hadm_id == 601
    assert calls == [("silver-root", "diagnoses_icd", "silver")]


def test_labels_can_reuse_optional_saved_cohort(spark, monkeypatch):
    admit = datetime(2150, 1, 2)
    saved_cohort, admissions, patients = _tables(
        spark,
        [(1, 602, admit, admit + timedelta(days=2), None, 0, "HOME")],
    )

    class SavedCohortReader:
        def parquet(self, path):
            assert path == "saved/cohort"
            return saved_cohort

    class SavedCohortSpark:
        read = SavedCohortReader()

    def unexpected_build(*args, **kwargs):
        pytest.fail("không dựng lại cohort khi đã truyền --cohort")

    monkeypatch.setattr(labels, "build_cohort", unexpected_build)
    monkeypatch.setattr(
        labels,
        "read_table",
        lambda *args: pytest.fail("không đọc diagnosis khi đã truyền --cohort"),
    )

    result = load_or_build_cohort(
        SavedCohortSpark(),
        admissions,
        patients,
        root="unused",
        fmt="silver",
        hf_codes={},
        cohort_path="saved/cohort",
    )

    assert result.select("hadm_id").collect()[0].hadm_id == 602


def test_report_suppresses_small_counts_and_is_aggregate_only(tmp_path):
    report = {
        "generated_at": "2026-11-03T00:00:00+00:00",
        "data_source": "Dữ liệu giả",
        "P1": {
            "summary": {"admissions": 100, "patients": 90, "positives": 5, "positive_rate": 0.05},
            "excluded": {"died_by_prediction_time": 2},
            "label_consistency_mismatch": 0,
        },
        "P2": {
            "summary": {"admissions": 100, "patients": 90, "positives": 20, "positive_rate": 0.2},
            "excluded": {"hospice": 3},
        },
        "P3": {
            "summary": {"admissions": 100, "patients": 90, "positives": 20, "positive_rate": 0.2},
            "excluded": {"hospice": 3},
        },
    }
    path = tmp_path / "labels_result.md"

    write_report(report, path)
    contents = path.read_text(encoding="utf-8")

    assert "<10" in contents
    assert "**Dữ liệu dùng để thử:** Dữ liệu giả." in contents
    assert "| P1 | 100 | 90 | <10 | 0.0500 |" in contents
    assert "subject_id" not in contents
    assert "P1" in contents and "P2" in contents and "P3" in contents

"""Kiểm thử data_spark/features.py (Gold P1). Chạy: pytest tests/test_features.py -v

Mỗi test dựng bảng nhỏ bằng createDataFrame, không đọc S3 hay dữ liệu thật.
Test tự khai báo fixture `spark` (local, 1 core) nên không phụ thuộc conftest.py.
"""

import json
from datetime import date, datetime, timedelta

import pytest
from pyspark.sql import SparkSession

from data_spark import features
from data_spark.common.io import load_yaml

ADMIT = datetime(2150, 1, 1)


@pytest.fixture(scope="module")
def spark():
    s = (
        SparkSession.builder.appName("hg-test-features")
        .master("local[1]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield s
    s.stop()


def frame(spark, rows, ddl):
    return spark.createDataFrame(rows, ddl)


def base_frame(spark, hadm_ids=(10,), admit=ADMIT, subject=1):
    rows = [(subject, h, admit, admit + timedelta(hours=24)) for h in hadm_ids]
    return frame(
        spark,
        rows,
        "subject_id long, hadm_id long, admittime timestamp, prediction_time timestamp",
    )


# ------------------------------------------------------------------ 1. schema
def test_schema_columns_match_contract():
    schema = json.loads(features.SCHEMA_PATH.read_text(encoding="utf-8"))
    cols = features.schema_columns(schema)
    names = [n for n, _ in cols]
    assert (
        len(names) == 124 and len(set(names)) == 124
    )  # 5 id + 1 nhãn + 4 subgroup + 114 đặc trưng
    assert "label_mortality" in names
    assert not set(names) & set(schema["forbidden_columns"])
    assert sum(n.endswith("_missing") for n in names) == 21  # 20 lab + vitals_missing
    assert dict(cols)["n_lab_tests_24h"] == "int" and dict(cols)["age"] == "double"


# ------------------------------------------------------------------ 2. quần thể P1
def test_build_base_keeps_only_admissions_eligible_at_hour_24(spark):
    ddl = (
        "subject_id long, hadm_id long, admittime timestamp, dischtime timestamp, "
        "deathtime timestamp, hospital_expire_flag int"
    )
    cohort = frame(
        spark,
        [
            (1, 10, ADMIT, ADMIT + timedelta(hours=25), None, 0),  # đủ điều kiện
            (
                2,
                20,
                ADMIT,
                ADMIT + timedelta(hours=24),
                None,
                0,
            ),  # xuất viện đúng giờ 24 -> loại
            (
                3,
                30,
                ADMIT,
                ADMIT + timedelta(days=2),
                ADMIT + timedelta(hours=24),
                1,
            ),  # chết đúng giờ 24 -> loại
            (
                4,
                40,
                ADMIT,
                ADMIT + timedelta(days=5),
                ADMIT + timedelta(days=3),
                1,
            ),  # chết sau 24h -> giữ, nhãn 1
            (
                5,
                50,
                ADMIT,
                ADMIT + timedelta(days=2),
                ADMIT + timedelta(hours=5),
                1,
            ),  # chết trước 24h -> loại
        ],
        ddl,
    )
    rows = {r.hadm_id: r for r in features.build_base(cohort).collect()}
    assert sorted(rows) == [10, 40]
    assert rows[10].prediction_time == ADMIT + timedelta(hours=24)
    assert rows[10].label_mortality == 0 and rows[40].label_mortality == 1


def test_build_base_uses_external_labels_to_restrict_population(spark):
    ddl = (
        "subject_id long, hadm_id long, admittime timestamp, dischtime timestamp, "
        "deathtime timestamp, hospital_expire_flag int"
    )
    cohort = frame(
        spark,
        [
            (1, 10, ADMIT, ADMIT + timedelta(days=3), None, 0),
            (2, 20, ADMIT, ADMIT + timedelta(days=3), None, 0),
        ],
        ddl,
    )
    labels = frame(spark, [(20, 1)], "hadm_id long, label int")
    rows = features.build_base(cohort, labels=labels).collect()
    assert [(r.hadm_id, r.label_mortality) for r in rows] == [(20, 1)]


def test_build_base_sampling_is_deterministic_by_subject(spark):
    ddl = (
        "subject_id long, hadm_id long, admittime timestamp, dischtime timestamp, "
        "deathtime timestamp, hospital_expire_flag int"
    )
    rows = [
        (s, s * 10 + k, ADMIT, ADMIT + timedelta(days=3), None, 0)
        for s in range(1, 201)
        for k in (1, 2)
    ]
    cohort = frame(spark, rows, ddl)
    a = features.build_base(cohort, sample_frac=0.3).collect()
    b = features.build_base(cohort, sample_frac=0.3).collect()
    assert {r.hadm_id for r in a} == {r.hadm_id for r in b}
    assert 0 < len(a) < 400
    by_subject = {}
    for r in a:
        by_subject.setdefault(r.subject_id, 0)
        by_subject[r.subject_id] += 1
    assert all(
        n == 2 for n in by_subject.values()
    )  # lấy mẫu theo bệnh nhân, không tách đợt nhập viện


# ------------------------------------------------------------------ 3. nhân khẩu học + subgroup
def test_demographics_groups_icu_and_icd_version(spark):
    ddl = (
        "subject_id long, hadm_id long, admittime timestamp, prediction_time timestamp, gender string, "
        "admission_type string, admission_location string, edregtime timestamp, race string"
    )
    p = ADMIT + timedelta(hours=24)
    base = frame(
        spark,
        [
            (
                1,
                10,
                ADMIT,
                p,
                "M",
                "EW EMER.",
                "EMERGENCY ROOM",
                ADMIT,
                "WHITE - RUSSIAN",
            ),
            (
                2,
                20,
                ADMIT,
                p,
                "F",
                "OBSERVATION ADMIT",
                None,
                None,
                "HISPANIC/LATINO - PUERTO RICAN",
            ),
            (
                3,
                30,
                ADMIT,
                p,
                "F",
                "SURGICAL SAME DAY ADMISSION",
                "TRANSFER FROM HOSPITAL",
                None,
                "BLACK/AFRICAN AMERICAN",
            ),
            (
                4,
                40,
                ADMIT,
                p,
                "M",
                "URGENT",
                "CLINIC REFERRAL",
                None,
                "UNABLE TO OBTAIN",
            ),
            (
                5,
                50,
                ADMIT,
                p,
                "M",
                "SOMETHING NEW",
                "WALK-IN/SELF REFERRAL",
                None,
                "AMERICAN INDIAN/ALASKA NATIVE",
            ),
        ],
        ddl,
    )
    icu = frame(
        spark,
        [
            (
                10,
                ADMIT + timedelta(hours=4),
                ADMIT + timedelta(days=2),
            ),  # vào ICU trước mốc -> 1
            (
                20,
                ADMIT + timedelta(hours=30),
                ADMIT + timedelta(days=3),
            ),  # vào ICU sau mốc -> 0
        ],
        "hadm_id long, intime timestamp, outtime timestamp",
    )
    dx = frame(
        spark, [(10, 10), (30, 9), (30, 10), (20, 9)], "hadm_id long, icd_version int"
    )
    r = {x.hadm_id: x for x in features.demographics(base, icu, dx).collect()}
    assert (r[10].gender_male, r[20].gender_male) == (1, 0)
    assert [r[h].admission_type_group for h in (10, 20, 30, 40, 50)] == [
        "EMERGENCY",
        "OBSERVATION",
        "ELECTIVE",
        "URGENT",
        "UNKNOWN",
    ]
    assert [r[h].admission_location_group for h in (10, 20, 30, 40, 50)] == [
        "EMERGENCY_ROOM",
        "UNKNOWN",
        "TRANSFER",
        "REFERRAL",
        "REFERRAL",
    ]
    assert [r[h].came_from_ed for h in (10, 20)] == [1, 0]
    assert [r[h].in_icu_24h for h in (10, 20, 30)] == [1, 0, 0]
    assert [r[h].race_group for h in (10, 20, 30, 40, 50)] == [
        "WHITE",
        "HISPANIC",
        "BLACK",
        "UNKNOWN",
        "OTHER",
    ]
    assert [r[h].icd_version_group for h in (10, 20, 30)] == ["ICD10", "ICD9", "MIXED"]


# ------------------------------------------------------------------ 4. tiền sử
HF = {"icd9": ["428"], "icd10": ["I50"]}
COMORB = {
    "ckd": {"icd9": ["585"], "icd10": ["N18"]},
    "diabetes": {"icd9": ["250"], "icd10": ["E11"]},
}


def test_history_uses_only_previously_discharged_admissions(spark):
    cur = datetime(2020, 7, 1)
    base = base_frame(spark, hadm_ids=(100,), admit=cur)
    adm = frame(
        spark,
        [
            (
                1,
                90,
                datetime(2019, 8, 1),
                datetime(2019, 8, 5),
            ),  # cách 330 ngày -> trong 365 ngày
            (1, 91, datetime(2020, 1, 1), datetime(2020, 1, 3)),  # cách 180 ngày
            (
                1,
                85,
                datetime(2018, 1, 1),
                datetime(2018, 1, 5),
            ),  # quá 365 ngày: vẫn là đợt trước, không tính vào 365d
            (1, 100, cur, cur + timedelta(days=4)),  # chính nó
            (1, 101, datetime(2021, 1, 1), datetime(2021, 1, 3)),  # tương lai
        ],
        "subject_id long, hadm_id long, admittime timestamp, dischtime timestamp",
    )
    dx = frame(
        spark,
        [
            (90, "42822", 9),
            (90, "5853", 9),
            (91, "25000", 9),
            (85, "I5022", 10),
            (100, "I5022", 10),
            (101, "N189", 10),  # dx của đợt hiện tại/tương lai KHÔNG được dùng
        ],
        "hadm_id long, icd_code string, icd_version int",
    )
    row = features.history(base, adm, dx, HF, COMORB).collect()[0]
    assert row.n_prior_adm_365d == 2
    assert row.n_prior_hf_adm == 2  # 90 (428.22) và 85 (I50.22)
    assert row.first_admission == 0
    assert row.days_since_last_discharge == pytest.approx(180.0)
    assert (row.hx_ckd, row.hx_diabetes) == (1, 1)


def test_history_for_first_admission_has_nulls_and_zero_counts(spark):
    base = base_frame(spark, hadm_ids=(7,))
    adm = frame(
        spark,
        [(1, 7, ADMIT, ADMIT + timedelta(days=3))],
        "subject_id long, hadm_id long, admittime timestamp, dischtime timestamp",
    )
    dx = frame(
        spark, [(7, "I5022", 10)], "hadm_id long, icd_code string, icd_version int"
    )
    row = features.history(base, adm, dx, HF, COMORB).collect()[0]
    assert (
        row.first_admission == 1
        and row.n_prior_adm_365d == 0
        and row.n_prior_hf_adm == 0
    )
    assert row.days_since_last_discharge is None
    assert (row.hx_ckd, row.hx_diabetes) == (0, 0)


# ------------------------------------------------------------------ 5. OMR
def test_omr_uses_latest_valid_value_strictly_before_admission(spark):
    admit = datetime(2150, 1, 10)
    base = base_frame(spark, admit=admit)
    omr = frame(
        spark,
        [
            (1, date(2150, 1, 9), 1, "BMI (kg/m2)", "27.5"),
            (
                1,
                date(2150, 1, 10),
                1,
                "BMI (kg/m2)",
                "31",
            ),  # cùng ngày nhập viện -> loại
            (1, date(2148, 1, 1), 1, "BMI (kg/m2)", "20"),  # quá 365 ngày
            (1, date(2150, 1, 8), 1, "Weight (Lbs)", "200"),
            (1, date(2150, 1, 7), 1, "Blood Pressure", "128/70"),
            (1, date(2150, 1, 6), 1, "eGFR", ">60"),  # chữ -> không parse
            (1, date(2150, 1, 5), 1, "eGFR", "60"),
        ],
        "subject_id long, chartdate date, seq_num int, result_name string, result_value string",
    )
    row = features.omr_features(base, omr).collect()[0]
    assert row.omr_bmi_last == 27.5
    assert row.omr_weight_last == pytest.approx(90.718, abs=1e-3)
    assert row.omr_sbp_last == 128.0
    assert row.omr_egfr_last == 60.0  # lấy giá trị số hợp lệ gần nhất


# ------------------------------------------------------------------ 6. xét nghiệm
LAB_DDL = "subject_id long, itemid int, charttime timestamp, storetime timestamp, valuenum double, valueuom string"
LAB_CFG = {"sodium": {"itemids": [11], "valid_range": [95, 185]}}


def lab_events(spark):
    h = lambda x: ADMIT + timedelta(hours=x)
    return frame(
        spark,
        [
            (1, 11, h(-6), h(-5), 138.0, "mEq/L"),  # đúng biên -6h: giữ, là giá trị đầu
            (1, 11, h(1), h(1), 137.0, "mEq/L"),
            (
                1,
                11,
                h(2),
                h(25),
                136.0,
                "mEq/L",
            ),  # storetime sau mốc -> chưa có kết quả tại 24h -> loại
            (
                1,
                11,
                h(3),
                h(3),
                250.0,
                "mEq/L",
            ),  # ngoài khoảng hợp lệ -> NULL, không vào min/max
            (1, 11, h(24), h(24), 140.0, "mEq/L"),  # đúng biên 24h: giữ
            (1, 11, h(25), h(25), 141.0, "mEq/L"),  # sau mốc -> loại
            (1, 11, h(-7), h(-7), 120.0, "mEq/L"),  # trước biên -6h -> loại
            (1, 11, h(5), h(5), 139.0, "mmol/L"),  # đơn vị khác -> loại, được đếm
            (1, 11, h(6), h(6), 138.5, "mEq/L"),
        ],
        LAB_DDL,
    )


def test_lab_window_and_features_respect_time_availability_range_and_unit(spark):
    base = base_frame(spark)
    lw = features.lab_window(lab_events(spark), base, strict_store=True)
    assert (
        lw.count() == 6
    )  # 9 dòng - (storetime muộn) - (sau mốc 24h) - (trước biên -6h)
    out, stats = features.lab_features(lw, LAB_CFG, min_unit_share=0.5)
    row = out.collect()[0]
    assert row.sodium_first == 138.0
    assert row.sodium_min == 137.0
    assert row.sodium_max == 140.0
    assert (
        stats["sodium"]["out_range"] == 1
        and stats["sodium"]["bad_unit"] == 1
        and stats["sodium"]["unit"] == "mEq/L"
    )


def test_lab_window_without_storetime_filter_keeps_late_results(spark):
    base = base_frame(spark)
    strict = features.lab_window(lab_events(spark), base, strict_store=True).count()
    loose = features.lab_window(lab_events(spark), base, strict_store=False).count()
    assert loose == strict + 1


def test_lab_without_hadm_id_is_attached_by_subject_and_time(spark):
    """Xét nghiệm ở ED trước khi nhập viện (hadm_id lab = NULL) vẫn được tính nhờ nối theo subject_id."""
    base = base_frame(spark)
    ev = frame(
        spark,
        [
            (
                1,
                11,
                ADMIT - timedelta(hours=3),
                ADMIT - timedelta(hours=2),
                133.0,
                "mEq/L",
            )
        ],
        LAB_DDL,
    )
    out, _ = features.lab_features(features.lab_window(ev, base), LAB_CFG, 0.5)
    assert out.collect()[0].sodium_first == 133.0


# ------------------------------------------------------------------ 7. sinh hiệu ICU
VIT_DDL = "hadm_id long, itemid int, charttime timestamp, storetime timestamp, valuenum double"
VIT_CFG = {
    "heart_rate": {
        "itemids": [220045],
        "valid_range": [20, 250],
        "feature_columns": ["hr_mean", "hr_max"],
    },
    "temperature": {
        "itemids": [223761],
        "valid_range": [30, 43],
        "feature_columns": ["temp_max_c"],
    },
}


def test_vitals_use_only_available_values_in_first_24_hours(spark):
    h = lambda x: ADMIT + timedelta(hours=x)
    ce = frame(
        spark,
        [
            (10, 220045, h(1), h(1), 80.0),
            (10, 220045, h(24), h(24), 100.0),  # đúng biên 24h -> giữ
            (10, 220045, h(2), h(25), 60.0),  # storetime sau mốc -> loại
            (10, 220045, h(25), h(25), 200.0),  # sau mốc -> loại
            (10, 220045, h(3), h(3), 300.0),  # ngoài khoảng sinh lý -> loại
            (10, 223761, h(4), h(4), 98.6),  # °F -> 37.0 °C
            (10, 223761, h(5), h(5), 110.0),  # 43.3 °C > 43 -> loại sau quy đổi
        ],
        VIT_DDL,
    )
    out, cols, stats = features.vitals_features(ce, base_frame(spark), VIT_CFG)
    row = out.collect()[0]
    assert cols == ["hr_mean", "hr_max", "temp_max_c"]
    assert row.hr_mean == 90.0 and row.hr_max == 100.0
    assert row.temp_max_c == pytest.approx(37.0)
    assert (
        stats["n_raw"] == 5
    )  # 7 dòng - (sau mốc 24h) - (storetime muộn); n_raw đếm trước khi lọc khoảng hợp lệ


# ------------------------------------------------------------------ 8. ghép đầu–cuối
def test_assemble_end_to_end_matches_contract(spark):
    items = load_yaml("configs/feature_items.yaml")
    schema = json.loads(features.SCHEMA_PATH.read_text(encoding="utf-8"))
    h = lambda x: ADMIT + timedelta(hours=x)
    cohort = frame(
        spark,
        [
            (
                1,
                10,
                ADMIT,
                ADMIT + timedelta(days=4),
                None,
                0,
                "EW EMER.",
                "EMERGENCY ROOM",
                ADMIT,
                "2017 - 2019",
                "M",
                70.0,
                "WHITE",
                "Medicare",
                1,
            ),
            (
                2,
                20,
                ADMIT,
                ADMIT + timedelta(days=4),
                ADMIT + timedelta(days=3),
                1,
                "URGENT",
                "TRANSFER FROM HOSPITAL",
                None,
                "2014 - 2016",
                "F",
                80.0,
                "BLACK",
                "Medicaid",
                0,
            ),
        ],
        "subject_id long, hadm_id long, admittime timestamp, dischtime timestamp, deathtime timestamp, hospital_expire_flag int, "
        "admission_type string, admission_location string, edregtime timestamp, anchor_year_group string, gender string, age double, "
        "race string, insurance string, hf_primary int",
    )
    base = features.build_base(cohort)
    adm = frame(
        spark,
        [
            (1, 10, ADMIT, ADMIT + timedelta(days=4)),
            (2, 20, ADMIT, ADMIT + timedelta(days=4)),
        ],
        "subject_id long, hadm_id long, admittime timestamp, dischtime timestamp",
    )
    dx = frame(
        spark,
        [(10, "I5022", 10), (20, "4280", 9)],
        "hadm_id long, icd_code string, icd_version int",
    )
    icu = frame(
        spark, [(10, h(3), h(40))], "hadm_id long, intime timestamp, outtime timestamp"
    )
    omr = frame(
        spark,
        [(1, date(2149, 12, 1), 1, "BMI (kg/m2)", "30")],
        "subject_id long, chartdate date, seq_num int, result_name string, result_value string",
    )
    labs_ev = frame(
        spark,
        [(1, 50912, h(2), h(3), 1.4, "mg/dL"), (1, 50912, h(10), h(11), 1.9, "mg/dL")],
        LAB_DDL,
    )
    ce = frame(spark, [(10, 220045, h(2), h(2), 88.0)], VIT_DDL)

    demo = features.demographics(base, icu, dx)
    hist = features.history(
        base,
        adm,
        dx,
        load_yaml("configs/hf_codes.yaml")["heart_failure"],
        load_yaml("configs/hf_codes.yaml")["comorbidities"],
    )
    om = features.omr_features(base, omr)
    lw = features.lab_window(labs_ev, base)
    nlab = lw.groupBy("hadm_id").count().withColumnRenamed("count", "n_lab_tests_24h")
    labs, _ = features.lab_features(lw, items["labs"], 0.5)
    vit, vit_cols, _ = features.vitals_features(ce, base, items["vitals"])
    out = features.assemble(
        base, demo, hist, om, labs, nlab, vit, vit_cols, schema, list(items["labs"])
    )

    assert out.columns == [
        n for n, _ in features.schema_columns(schema)
    ]  # đúng tên và đúng thứ tự
    assert {f.name: f.dataType.simpleString() for f in out.schema.fields}[
        "hadm_id"
    ] == "bigint"
    rows = {r.hadm_id: r for r in out.collect()}
    assert sorted(rows) == [10, 20]  # mỗi hadm_id đúng 1 dòng
    r10, r20 = rows[10], rows[20]
    assert (
        r10.creatinine_first == 1.4
        and r10.creatinine_max == 1.9
        and r10.creatinine_missing == 0
    )
    assert (
        r10.sodium_missing == 1 and r10.sodium_first is None
    )  # thiếu để NULL, không điền
    assert r10.n_lab_tests_24h == 2 and r20.n_lab_tests_24h == 0
    assert (r10.in_icu_24h, r20.in_icu_24h) == (1, 0)
    assert (r10.vitals_missing, r20.vitals_missing) == (0, 1) and r10.hr_mean == 88.0
    assert (r10.label_mortality, r20.label_mortality) == (0, 1)
    assert (r10.icd_version_group, r20.icd_version_group) == ("ICD10", "ICD9")
    assert r10.first_admission == 1 and r10.hx_ckd == 0
    assert not set(out.columns) & set(schema["forbidden_columns"])

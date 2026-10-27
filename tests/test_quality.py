import json
from data_spark.common.io import load_yaml
from data_spark.quality import audit_event_units, null_invalid_values, profile_table, redact_counts, run_quality, write_report


def test_profile_missing_nan_blank_duplicate_and_ranges(spark):
    df = spark.createDataFrame([(10, None, "", "A"), (10, float("nan"), "x", "A"), (11, 999.0, " ", "B"), (12, 95.0, "x", "B")],
                               "hadm_id long, sodium_first double, note string, anchor_year_group string")
    report = profile_table(df)
    columns = {r["column"]: r for r in report["columns"]}
    assert columns["sodium_first"]["missing_pct"] == 50
    assert columns["note"]["missing_count"] == 2
    assert columns["sodium_first"]["invalid_count"] == 1
    assert report["duplicates"]["duplicate_keys"] == 1
    assert report["duplicates"]["duplicate_rows"] == 2
    assert report["duplicates"]["excess_rows"] == 1
    assert len(report["missing_by_year"]) == 2
    assert report["status"] == "FAILED"
    assert null_invalid_values(df).where("hadm_id = 11").first()["sodium_first"] is None
    assert df.where("hadm_id = 11").first()["sodium_first"] == 999


def test_empty_table_and_null_key_fail(spark):
    assert profile_table(spark.createDataFrame([], "hadm_id long"))["status"] == "FAILED"
    assert profile_table(spark.createDataFrame([(None,)], "hadm_id long"))["duplicates"]["null_key_rows"] == 1


def test_missing_required_evidence_blocks_overall_status(spark):
    df = spark.createDataFrame([(10, 138.0)], "hadm_id long, sodium_first double")
    assert run_quality(df)["status"] == "NOT_CHECKED"
    assert run_quality(df, p1=False)["status"] == "PASSED"


def test_selected_items_cover_contract_and_spec():
    cfg = load_yaml("configs/feature_items.yaml")
    assert len(cfg["labs"]) == 20
    assert len(cfg["vitals"]) == 7
    ids = [i for group in ("labs", "vitals") for s in cfg[group].values() for i in s["itemids"]]
    assert len(ids) == len(set(ids))
    assert cfg["selection"]["status"] == "synthetic_verified_pending_real"


def test_unit_dominance_threshold_numeric_text_and_temperature(spark):
    events = [(50983, "mEq/L", 138.0, "138")] * 19 + [(50983, "mmol/L", 138.0, "138")]
    df = spark.createDataFrame(events, "itemid int, valueuom string, valuenum double, value string")
    sodium = next(i for i in audit_event_units(df)["items"] if i["feature"] == "sodium")
    assert sodium["status"] == "PASSED"  # đúng 95%
    assert sodium["wrong_unit_rows"] == 1
    bad = df.unionByName(spark.createDataFrame([(50983, "wrong", None, "<0.01")], df.schema))
    sodium = next(i for i in audit_event_units(bad)["items"] if i["feature"] == "sodium")
    assert sodium["status"] == "FAILED"
    assert sodium["numeric_text_rows"] == 1
    temperature = spark.createDataFrame([(223761, "°F", 98.6, "98.6"), (223761, "°F", 200.0, "200")], df.schema)
    item = next(i for i in audit_event_units(temperature, "vitals")["items"] if i["feature"] == "temperature")
    assert item["accepted_rows"] == 1
    assert item["invalid_value_rows"] == 1


def test_reports_contain_aggregates_and_suppress_small_counts(tmp_path):
    report = {"status": "FAILED", "checks": {"profile": {"rows": 100, "columns": [
        {"column": "sodium_first", "missing_count": 3, "missing_pct": 3.0, "invalid_count": 0}]}}}
    safe = redact_counts(report)
    write_report(safe, str(tmp_path))
    payload = json.loads((tmp_path / "quality_report.json").read_text())
    assert payload["checks"]["profile"]["columns"][0]["missing_count"] == "<10"
    assert "subject_id" not in (tmp_path / "quality_report.json").read_text()


def test_old_silver_missing_availability_timestamp_fails(spark, tmp_path):
    from data_spark.common.io import read_table
    import pytest

    directory = tmp_path / "icu/chartevents"
    spark.createDataFrame([(1,)], "subject_id long").write.parquet(str(directory))
    with pytest.raises(ValueError, match="storetime"):
        read_table(spark, str(tmp_path), "chartevents", "silver")

"""Kiểm thử pipeline_runner: phần điều phối (thuần Python) + validate trên Gold giả (Spark local)."""
import json
import subprocess
import sys

import pandas as pd
import pytest

from data_spark import pipeline_runner as pr
from data_spark.common.io import REPO_ROOT, load_yaml

CFG = load_yaml("configs/data.yaml")


def test_resolve_paths_version_and_sample():
    p = pr.resolve_paths(CFG, "v1", None)
    assert p["gold_p1"].endswith("/gold/p1/v1")
    assert p["cohort"].endswith("/processed/cohort")
    assert p["manifest"].endswith("/gold/p1/v1/_manifest.json")
    s = pr.resolve_paths(CFG, "v1", 0.01)
    assert s["gold_p1"].endswith("/gold/p1/_sample_0.01"), "chạy mẫu không được ghi đè v1"


def test_build_commands_order_and_flags():
    p = pr.resolve_paths(CFG, "v1", 0.05)
    cmds = pr.build_commands(["cohort", "labels", "features"], p, version="v1",
                             labels="s3a://b/labels/p1", label_col="y", sample_frac=0.05, skip_verify=True)
    names = [n for n, _ in cmds]
    assert names == ["cohort", "labels", "features"]
    feat = dict(cmds)["features"]
    assert feat[1:3] == ["-m", "data_spark.features"]
    for flag in ("--labels", "--label-col", "--sample-frac", "--skip-verify", "--cohort", "--out"):
        assert flag in feat
    assert feat[feat.index("--out") + 1] == p["gold_p1"]
    labels_cmd = dict(cmds)["labels"]
    assert labels_cmd == [] or labels_cmd[1:] == ["-m", "data_spark.labels"]


def test_build_commands_without_labels_uses_no_label_flag():
    p = pr.resolve_paths(CFG, "v1", None)
    feat = dict(pr.build_commands(["features"], p, version="v1"))["features"]
    assert "--labels" not in feat and "--sample-frac" not in feat


def _summary(**kw):
    s = {"rows": 100, "distinct_hadm": 100, "patients": 80, "positives": 5, "positive_rate": 0.05,
         "label_nulls": 0, "label_values": [0, 1], "bad_prediction_time": 0, "n_columns": 124,
         "forbidden_present": [], "missing_contract_cols": [], "by_year_group": []}
    s.update(kw)
    return s


def test_cross_checks_pass_and_fail():
    assert all(c["ok"] for c in pr.cross_checks(_summary(), 100, {"rows": 100, "positives": 5}))
    bad = pr.cross_checks(_summary(distinct_hadm=99, forbidden_present=["dischtime"], label_nulls=2), 101,
                          {"positives": 6})
    failed = {c["check"] for c in bad if not c["ok"]}
    assert "hadm_id duy nhất" in failed
    assert "Không có cột bị cấm" in failed
    assert "Không có nhãn NULL" in failed
    assert "Số dòng Gold = số ca P1 tính lại từ cohort" in failed
    assert "positives = 6 (kỳ vọng)" in failed


def test_render_markdown_masks_small_cells():
    m = {"data_version": "v1", "created_at": "t", "git_commit": "abc", "status": "PASSED",
         "paths": {"gold_p1": "s3a://x/gold/p1/v1"}, "label_source": "tạm",
         "summary": _summary(by_year_group=[{"group": "2020 - 2022", "n": 50, "positives": 3}]),
         "cross_checks": pr.cross_checks(_summary(), 100), "quality_status": "NOT_CHECKED",
         "quality_note": "chưa có lineage", "steps": [{"step": "features", "status": "OK", "seconds": 1.0}]}
    md = pr.render_markdown(m, threshold=10)
    assert "| 2020 - 2022 | 50 | <10 |" in md
    assert "Ca dương: 5 (5.00%)" in md


def test_dry_run_cli():
    r = subprocess.run([sys.executable, "-m", "data_spark.pipeline_runner", "--dry-run",
                        "--steps", "features", "validate", "--sample-frac", "0.01"],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "data_spark.features" in r.stdout and "_sample_0.01" in r.stdout


def test_validate_on_synthetic_gold(spark, tmp_path):
    gold_csv = tmp_path / "gold.csv"
    subprocess.run([sys.executable, str(REPO_ROOT / "tools/make_synthetic_gold.py"),
                    "--rows", "200", "--out", str(gold_csv)], check=True)
    pdf = pd.read_csv(gold_csv)
    for c in ("admittime", "prediction_time"):
        pdf[c] = pd.to_datetime(pdf[c])
    gold = spark.createDataFrame(pdf)
    gold_path = str(tmp_path / "gold_p1")
    gold.write.mode("overwrite").parquet(gold_path)

    cohort = pdf[["subject_id", "hadm_id", "admittime"]].copy()
    cohort["dischtime"] = cohort["admittime"] + pd.Timedelta(days=5)
    cohort["deathtime"] = pd.NaT
    cohort_path = str(tmp_path / "cohort")
    spark.createDataFrame(cohort.astype({"deathtime": "datetime64[ns]"})) \
        .selectExpr("subject_id", "hadm_id", "admittime", "dischtime",
                    "cast(deathtime as timestamp) as deathtime") \
        .write.mode("overwrite").parquet(cohort_path)

    schema = json.loads((REPO_ROOT / "contracts/gold_schema_p1.json").read_text(encoding="utf-8"))
    paths = {"gold_p1": gold_path, "cohort": cohort_path}
    summary, cohort_p1, qrep = pr.validate(spark, paths, schema, 24)

    assert summary["rows"] == 200 == cohort_p1
    assert summary["positives"] == int(pdf["label_mortality"].sum())
    assert summary["forbidden_present"] == [] and summary["missing_contract_cols"] == []
    assert summary["bad_prediction_time"] == 0
    assert all(c["ok"] for c in pr.cross_checks(summary, cohort_p1))
    assert qrep["status"] in {"PASSED", "NOT_CHECKED"}

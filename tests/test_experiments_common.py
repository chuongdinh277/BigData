"""Kiểm thử khung thí nghiệm chung GĐ3 trên Gold giả (không cần Spark)."""
import csv
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from experiments.common.config import REPO_ROOT
from experiments.common import config
from experiments.common.data import feature_columns, load_gold
from experiments.common.metrics import choose_threshold, evaluate
from experiments.common.results import write_result
from experiments.common.runner import baseline_pipeline, run_case
from experiments.common.split import apply_split, load_manifest, make_split, save_manifest


@pytest.fixture(scope="module")
def gold(tmp_path_factory):
    path = tmp_path_factory.mktemp("gold") / "gold_fake.csv"
    subprocess.run([sys.executable, str(REPO_ROOT / "tools/make_synthetic_gold.py"), "--rows", "1500",
                    "--out", str(path)], check=True)
    return path


def test_feature_columns_exclude_ids_label_subgroups(gold):
    feats = feature_columns(load_gold(str(gold)))
    assert not (set(feats) & config.NON_FEATURE_COLUMNS)
    assert config.LABEL not in feats and "subject_id" not in feats and "race_group" not in feats


def test_split_by_patient_no_overlap_and_deterministic(gold):
    df = load_gold(str(gold))
    m1, m2 = make_split(df), make_split(df)
    assert m1["subjects"] == m2["subjects"] and m1["sha256"] == m2["sha256"]
    s = {k: set(v) for k, v in m1["subjects"].items()}
    assert not (s["train"] & s["val"]) and not (s["train"] & s["test"]) and not (s["val"] & s["test"])
    assert s["train"] | s["val"] | s["test"] == set(df["subject_id"])
    n = df["subject_id"].nunique()
    assert abs(len(s["train"]) / n - 0.70) < 0.02


def test_apply_split_hides_test_by_default(gold, tmp_path):
    df = load_gold(str(gold))
    m = make_split(df)
    save_manifest(m, tmp_path / "split.json")
    parts = apply_split(df, load_manifest(tmp_path / "split.json"))
    assert set(parts) == {"train", "val"}


def test_manifest_tamper_detected(gold, tmp_path):
    df = load_gold(str(gold))
    m = make_split(df)
    m["subjects"]["val"].append(m["subjects"]["test"][0])
    save_manifest(m, tmp_path / "bad.json")
    with pytest.raises(ValueError):
        load_manifest(tmp_path / "bad.json")


def test_metrics_known_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    r = evaluate(y, p, threshold=0.5)
    assert r["roc_auc"] == pytest.approx(0.75)
    assert r["f1"] == pytest.approx(2 / 3)
    assert r["brier"] == pytest.approx(np.mean((p - y) ** 2))
    assert 0.1 <= choose_threshold(y, p) <= 0.8


def test_write_result_rejects_unknown_columns(tmp_path):
    with pytest.raises(ValueError):
        write_result(tmp_path / "r.csv", {"case_id": "x", "bad_col": 1})


def test_run_baseline_end_to_end(gold, tmp_path):
    df = load_gold(str(gold))
    m = make_split(df, data_version="gold_fake")
    save_manifest(m, tmp_path / "split.json")
    out = tmp_path / "results.csv"
    row = run_case(case_id="S0_lr_median_baseline", member="test", strategy="S0", model_name="LogisticRegression",
                   pipeline=baseline_pipeline(df, feature_columns(df)), gold_path=str(gold),
                   manifest_path=str(tmp_path / "split.json"), results_path=str(out), df=df)
    assert 0.5 < row["roc_auc"] <= 1.0
    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == config.RESULT_COLUMNS and len(rows) == 1

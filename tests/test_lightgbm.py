"""LightGBM stage 4: smoke tests chỉ dùng dữ liệu giả local, không truy cập S3/MIMIC."""

from __future__ import annotations

import csv
import subprocess
import sys

import pytest

from experiments.chien.run_lightgbm import (
    CASE_DEFAULT,
    CASE_OPTUNA,
    CASE_WEIGHTED,
    build_pipeline,
    execute,
    expected_features,
    validate_gold,
    validate_manifest,
)
from experiments.common.config import REPO_ROOT, RESULT_COLUMNS
from experiments.common.data import load_gold
from experiments.common.split import load_manifest, make_split, save_manifest


@pytest.fixture(scope="module")
def fake_gold(tmp_path_factory):
    folder = tmp_path_factory.mktemp("lgbm_synthetic")
    gold_path = folder / "gold_fake.csv"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/make_synthetic_gold.py"),
            "--rows",
            "800",
            "--out",
            str(gold_path),
        ],
        check=True,
    )
    df = load_gold(str(gold_path))
    manifest = make_split(df, data_version="fake")
    split_path = folder / "split_fake.json"
    save_manifest(manifest, str(split_path))
    return gold_path, split_path


def test_three_cases_and_exact_schema(fake_gold, tmp_path):
    gold, split = fake_gold
    out = tmp_path / "results.csv"
    rows = execute(
        gold=str(gold),
        split=str(split),
        out=str(out),
        trials=2,
        jobs=2,
        synthetic_split=True,
    )
    assert [r["case_id"] for r in rows] == [CASE_DEFAULT, CASE_WEIGHTED, CASE_OPTUNA]
    with out.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == RESULT_COLUMNS
        saved = list(reader)
    assert len(saved) == 3
    assert [r["strategy"] for r in saved] == ["S1", "S1", "S1"]
    assert all(r["member"] == "chien" and r["model"] == "LightGBM" for r in saved)
    assert all(r["split_seed"] == "42" for r in saved)
    assert all(0 <= float(r["pr_auc"]) <= 1 for r in saved)
    assert float(saved[2]["seconds"]) >= float(rows[2]["seconds"]) - 0.001


def test_no_silent_overwrite_and_no_s3_escape(fake_gold, tmp_path):
    gold, split = fake_gold
    out = tmp_path / "existing.csv"
    out.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        execute(
            gold=str(gold),
            split=str(split),
            out=str(out),
            trials=1,
            jobs=2,
            synthetic_split=True,
        )
    assert out.read_text(encoding="utf-8") == "keep"
    with pytest.raises(ValueError, match="dữ liệu giả cục bộ"):
        execute(
            gold="s3://example/gold",
            split=str(split),
            out=str(tmp_path / "unused.csv"),
            trials=1,
            jobs=2,
            synthetic_split=True,
        )


def test_split_locked_requires_correct_hash(fake_gold):
    _, split = fake_gold
    manifest = load_manifest(str(split))
    with pytest.raises(ValueError, match="SHA256"):
        validate_manifest(manifest, locked=True)
    validate_manifest(manifest, locked=False)


def test_contract_excludes_forbidden_columns(fake_gold):
    gold, _ = fake_gold
    df = load_gold(str(gold))
    feats = [c for c in df.columns if c in expected_features()]
    assert len(expected_features()) == 114
    validate_gold(df, feats, locked=False)
    df["hospital_expire_flag"] = 1
    with pytest.raises(ValueError, match="cấm"):
        validate_gold(df, feats, locked=False)
    df = df.drop(columns=["hospital_expire_flag"])
    with pytest.raises(ValueError, match="contract"):
        validate_gold(df, feats + ["future_los"], locked=False)


def test_preprocessing_is_inside_pipeline():
    pipe = build_pipeline(["age"], ["admission_type_group"], jobs=1)
    assert list(pipe.named_steps) == ["prep", "model"]
    assert pipe.named_steps["prep"].transformers[0][1].named_steps.get("scale") is None
    assert pipe.named_steps["model"].get_params()["random_state"] == 42
    assert pipe.named_steps["model"].get_params().get("scale_pos_weight") is None
    weighted = build_pipeline(["age"], ["admission_type_group"], jobs=1, params={"scale_pos_weight": 19.0})
    assert weighted.named_steps["model"].get_params()["scale_pos_weight"] == 19.0

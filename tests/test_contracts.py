"""Kiểm thử hợp đồng Giai đoạn 1 — chạy không cần Spark."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_configs_load():
    for f in ["data.yaml", "hf_codes.yaml", "feature_items.yaml"]:
        assert yaml.safe_load((ROOT / "configs" / f).read_text(encoding="utf-8"))


def test_hf_codes_have_both_versions():
    hf = yaml.safe_load((ROOT / "configs/hf_codes.yaml").read_text(encoding="utf-8"))["heart_failure"]
    assert any(c.startswith("I50") for c in hf["icd10"])
    assert "428" in hf["icd9"]


def test_synthetic_gold_matches_schema(tmp_path):
    out = tmp_path / "gold.csv"
    subprocess.run([sys.executable, str(ROOT / "tools/make_synthetic_gold.py"), "--rows", "200", "--out", str(out)],
                   check=True)
    df = pd.read_csv(out)
    schema = json.loads((ROOT / "contracts/gold_schema_p1.json").read_text(encoding="utf-8"))
    assert df["hadm_id"].is_unique
    assert set(df[schema["label"]].unique()) <= {0, 1}
    assert not set(schema["forbidden_columns"]) & set(df.columns)
    assert (pd.to_datetime(df["prediction_time"]) > pd.to_datetime(df["admittime"])).all()

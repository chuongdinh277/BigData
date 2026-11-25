"""Kiểm thử GĐ5 (trustworthy_ai) trên dữ liệu giả, không cần Spark hay S3."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from experiments.common.config import LABEL, REPO_ROOT
from experiments.common.data import feature_columns
from experiments.common.runner import make_preprocessor, split_feature_types
from trustworthy_ai.calibration import Calibrator, choose_thresholds, crossfit_calibrate, risk_level, select_calibrator
from trustworthy_ai.drift import DriftReference, psi
from trustworthy_ai.evaluation import bootstrap_ci, calibration_stats, full_metrics, logit
from trustworthy_ai.explain import Explainer, global_importance
from trustworthy_ai.review import build_rules, flag_cases, queue_report
from trustworthy_ai.robustness import UNITS, mask, missing_fraction
from trustworthy_ai.subgroup import add_subgroup_columns, subgroup_table


@pytest.fixture(scope="module")
def gold(tmp_path_factory):
    out = tmp_path_factory.mktemp("gold") / "gold.csv"
    subprocess.run([sys.executable, str(REPO_ROOT / "tools/make_synthetic_gold.py"), "--rows", "2500",
                    "--out", str(out)], check=True, capture_output=True)
    return pd.read_csv(out)


def _pipeline(df, feats, kind):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    num, cat = split_feature_types(df, feats)
    if kind == "xgb":
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=40, max_depth=3, n_jobs=2, random_state=0)
    elif kind == "lgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=40, num_leaves=15, verbosity=-1, random_state=0)
    elif kind == "osma":
        pytest.importorskip("torch")
        from trustworthy_ai.osma_net import osma_pipeline
        return osma_pipeline(epochs=5, n_jobs=2)
    else:
        model = LogisticRegression(max_iter=500)
    return Pipeline([("prep", make_preprocessor(num, cat, scale=kind == "lr")), ("model", model)])


@pytest.mark.parametrize("kind", ["xgb", "lgbm", "lr", "osma"])
def test_shap_is_additive_and_maps_back_to_original_columns(gold, kind):
    feats = feature_columns(gold)
    X, y = gold[feats], gold[LABEL].to_numpy()
    pipe = _pipeline(gold, feats, kind).fit(X, y)
    ex = Explainer(pipe, X.sample(300, random_state=0) if kind == "lr" else None)
    shap_df, base = ex.contributions(X.head(200))
    expected = {c for c in feats if not c.endswith("_missing")} if kind == "osma" else set(feats)
    assert set(shap_df.columns) == expected                        # one-hot đã gộp về cột gốc
    margin = logit(pipe.predict_proba(X.head(200))[:, 1])
    assert np.max(np.abs(base + shap_df.sum(axis=1).to_numpy() - margin)) < 1e-3
    one = ex.explain_row(X.head(1), top_k=5)
    assert len(one["top_features"]) == 5 and {"feature", "value", "shap", "direction"} <= set(one["top_features"][0])
    assert global_importance(shap_df)["mean_abs_shap"].is_monotonic_decreasing
    if kind == "osma":
        assert one["organ_systems"] and one["method"].startswith("ExactAdditive")


def test_osma_net_uses_learned_missing_risk_and_survives_pickle(gold):
    pytest.importorskip("torch")
    import pickle
    from trustworthy_ai.osma_net import organ_system, osma_pipeline
    feats = feature_columns(gold)
    pipe = osma_pipeline(epochs=5, n_jobs=2).fit(gold[feats], gold[LABEL])
    row = gold[feats].head(1).copy()
    miss = row.copy()
    miss[["lactate_first", "lactate_min", "lactate_max"]] = np.nan
    a = pipe.predict_proba(row)[0, 1]
    b = pipe.predict_proba(miss)[0, 1]
    assert np.isfinite(b) and (a != b or row["lactate_first"].isna().all())
    again = pickle.loads(pickle.dumps(pipe))
    assert np.allclose(again.predict_proba(gold[feats].head(50)), pipe.predict_proba(gold[feats].head(50)))
    assert organ_system("creatinine_max") == "than" and organ_system("ntprobnp_first") == "tim_mach"


def test_calibration_fixes_miscalibrated_probabilities():
    rng = np.random.default_rng(0)
    p_true = rng.uniform(0.01, 0.3, 6000)
    y = (rng.random(6000) < p_true).astype(int)
    p_bad = np.clip(p_true * 2.5, 0, 0.99)                         # lệch như khi dùng class weight
    folds = rng.integers(0, 5, 6000)
    method, rep = select_calibrator(p_bad, y, folds)
    assert method != "none" and rep[method]["brier"] < rep["none"]["brier"]
    fixed = crossfit_calibrate(p_bad, y, folds, method)
    assert abs(calibration_stats(y, fixed)["oe_ratio"] - 1) < abs(calibration_stats(y, p_bad)["oe_ratio"] - 1)
    th = choose_thresholds(y, fixed)
    assert th["low_medium"] < th["medium_high"] and th["at_medium"]["recall"] >= 0.8
    assert set(risk_level(fixed, th)) <= {"LOW", "MEDIUM", "HIGH"}
    cal = Calibrator("platt").fit(p_bad, y)
    assert np.all(np.diff(cal.transform(np.linspace(0.01, 0.9, 50))) > 0)  # giữ thứ tự


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 800)
    p = np.clip(y * 0.3 + rng.uniform(0, 0.7, 800), 0, 1)
    groups = rng.integers(0, 300, 800)
    ci = bootstrap_ci(y, p, groups, threshold=0.5, n_boot=200)
    point = full_metrics(y, p, 0.5)
    for m in ("roc_auc", "pr_auc", "brier"):
        assert ci[m][0] <= point[m] <= ci[m][1]


def test_psi_detects_shift_and_missing_bin():
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 5000), "b": rng.choice(["x", "y"], 5000)})
    same = pd.DataFrame({"a": rng.normal(0, 1, 5000), "b": rng.choice(["x", "y"], 5000)})
    moved = pd.DataFrame({"a": rng.normal(1.5, 1, 5000), "b": rng.choice(["x", "y"], 5000, p=[.9, .1])})
    moved.loc[:2000, "a"] = np.nan
    dr = DriftReference.fit(ref, scores=rng.uniform(0, 1, 5000))
    stable = dr.compare(same, scores=rng.uniform(0, 1, 5000))
    shifted = dr.compare(moved, scores=rng.uniform(0.5, 1, 5000))
    assert stable["status"] == "ỔN ĐỊNH" and shifted["status"] == "CẢNH BÁO"
    a = shifted["features"].set_index("feature").loc["a"]
    assert a["psi"] > 0.25 and a["missing_rate_delta"] > 0.3
    assert psi([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0)


def test_masking_sets_values_null_and_flags_on(gold):
    X = gold[feature_columns(gold)]
    m = mask(X, ["lab:creatinine", "vitals"], 1.0, np.random.default_rng(0))
    assert m[["creatinine_first", "creatinine_min", "creatinine_max"]].isna().all().all()
    assert (m["creatinine_missing"] == 1).all() and (m["vitals_missing"] == 1).all()
    assert m["hr_mean"].isna().all() and m["sodium_first"].notna().sum() == X["sodium_first"].notna().sum()
    assert missing_fraction(m).mean() > missing_fraction(X).mean()
    assert "lab:ntprobnp" in UNITS


def test_review_flags_reasons_and_queue_report(gold):
    X = gold[feature_columns(gold)].reset_index(drop=True)
    y = gold[LABEL].to_numpy()
    p = np.full(len(X), 0.02)
    p[:50] = 0.5
    th = {"low_medium": 0.05, "medium_high": 0.2}
    rules = build_rules(thresholds=th, X_train=X, std_train=np.linspace(0, 1, len(X)))
    Xc = X.copy()
    Xc.loc[0, "sodium_first"] = 300                                # ngoài khoảng sinh lý
    Xc.loc[1, ["creatinine_first", "creatinine_min", "creatinine_max"]] = np.nan
    Xc.loc[1, "creatinine_missing"] = 1
    f = flag_cases(Xc, p, rules, std=np.zeros(len(X)))
    assert "HIGH_RISK" in f.loc[0, "reasons"] and "OUT_OF_RANGE:sodium_first" in f.loc[0, "reasons"]
    assert "MISSING_KEY_LAB:creatinine" in f.loc[1, "reasons"] and f.loc[0, "priority"] == 1
    rep = queue_report(f, y)
    assert 0 < rep["flagged_share"] <= 1 and rep["by_reason"]["HIGH_RISK"]["n"] == 50


def test_subgroup_table_masks_small_groups(gold):
    y = gold[LABEL].to_numpy()
    p = np.clip(0.05 + 0.1 * y + np.random.default_rng(3).uniform(0, 0.2, len(y)), 0, 1)
    sg = add_subgroup_columns(gold)
    t = subgroup_table(sg, y, p, gold["subject_id"].to_numpy(), 0.15, n_boot=20)
    assert t.iloc[0]["attribute"] == "TỔNG THỂ"
    small = t[t["n_pos"] < 10]
    assert small["roc_auc"].isna().all()
    assert {"gioi_tinh", "nhom_tuoi", "giai_doan"} <= set(t["attribute"])


def test_stage5_end_to_end_bundle_and_predict(gold, tmp_path):
    """Chạy toàn bộ GĐ5 (bản nhanh) rồi nạp gói mô hình và dự đoán 1 ca như API."""
    from trustworthy_ai.bundle import HeartGuardModel
    gpath = tmp_path / "gold.csv"
    gold.to_csv(gpath, index=False)
    spath = tmp_path / "split.json"
    subprocess.run([sys.executable, "-m", "experiments.common.split", "--gold", str(gpath), "--out", str(spath)],
                   check=True, capture_output=True, cwd=REPO_ROOT)
    from experiments.stage5 import run_stage5
    out = tmp_path / "run"
    assert run_stage5.main(["--gold", str(gpath), "--split", str(spath), "--synthetic", "--quick", "--open-test",
                            "--folds", "3", "--candidates", "XGBoost", "LogisticRegression",
                            *(["OSMANet"] if run_stage5.torch_available() else []), "--out", str(out)]) == 0
    assert (out / "docs/stage5_report.md").exists() and (out / "test_lock.json").exists()
    with pytest.raises(SystemExit):                                # test chỉ được mở một lần
        run_stage5.main(["--gold", str(gpath), "--split", str(spath), "--synthetic", "--quick", "--open-test",
                         "--out", str(out)])
    import json
    bundle = json.loads((out / "stage5_results.json").read_text(encoding="utf-8"))["bundle_dir"]
    model = HeartGuardModel(bundle)
    res = model.predict({"age": 81, "gender_male": 1, "admission_type_group": "EMERGENCY",
                         "creatinine_first": 2.4, "creatinine_max": 3.0, "sodium_first": 300,
                         "ntprobnp_first": None}, top_k=5)[0]
    assert {"probability", "calibrated_probability", "risk_level", "review", "input_quality", "explanation",
            "disclaimer", "model_version"} <= set(res)
    assert 0 <= res["calibrated_probability"] <= 1 and res["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert "sodium_first" in res["input_quality"]["out_of_range_fields"]
    assert any(r.startswith("OUT_OF_RANGE") for r in res["review"]["reasons"])
    assert len(res["explanation"]["top_features"]) == 5
    assert model.drift_report(gold.head(300)[model.schema["features"]])["status"] in {"ỔN ĐỊNH", "THEO DÕI", "CẢNH BÁO"}

    # Nạp gói qua S3 (client giả: "tải" từ thư mục gói cục bộ)
    import shutil
    from pathlib import Path

    class FakeS3:
        def get_paginator(self, _):
            class P:
                def paginate(self, Bucket, Prefix):
                    return [{"Contents": [{"Key": Prefix + f.name} for f in Path(bundle).iterdir()]}]
            return P()

        def download_file(self, bucket, key, dest):
            shutil.copy(Path(bundle) / key.rsplit("/", 1)[-1], dest)

    import boto3
    orig = boto3.client
    boto3.client = lambda *_a, **_k: FakeS3()
    try:
        m2 = HeartGuardModel.from_s3("s3://bucket/model-artifacts/p1/v-test/", cache_dir=tmp_path / "cache")
    finally:
        boto3.client = orig
    assert m2.model_version == model.model_version
    shutil.rmtree(bundle)

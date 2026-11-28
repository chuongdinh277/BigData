"""Test các phân tích độ tin cậy bổ sung: lợi ích lâm sàng, so sánh ghép cặp, drift nâng cao,
kiểm chứng luật review, độ trung thực/hợp lý của giải thích, mốc lâm sàng."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trustworthy_ai.drift import coding_drift, domain_classifier, exceed_null, psi_null_thresholds
from trustworthy_ai.evaluation import paired_bootstrap_diff
from trustworthy_ai.explain import clinical_plausibility
from trustworthy_ai.labels import describe
from trustworthy_ai.review import confidence_validation, risk_coverage
from trustworthy_ai.utility import capacity_curve, dca_summary, decision_curve, net_benefit

RNG = np.random.default_rng(0)


def _data(n=4000, prev=0.1, signal=2.0):
    y = (RNG.random(n) < prev).astype(int)
    score = RNG.normal(size=n) + signal * y
    p = 1 / (1 + np.exp(-(score - 3)))
    return y, p


def test_net_benefit_formula_and_dca():
    y = np.array([1, 0, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1, 0.05])
    # ngưỡng 0.5: TP=1, FP=1 → 1/5 − 1/5 · 0.5/0.5 = 0
    assert net_benefit(y, p, 0.5) == pytest.approx(0.0)
    y, p = _data()
    curve = decision_curve(y, p, 0.02, 0.3)
    assert (curve["net_benefit_treat_none"] == 0).all()
    s = dca_summary(curve)
    assert s["useful_threshold_min"] is not None  # mô hình có tín hiệu thì phải hơn mặc định ở đâu đó


def test_capacity_curve_monotone():
    y, p = _data()
    c = capacity_curve(y, p)
    assert c["positives_captured_share"].is_monotonic_increasing


def test_paired_bootstrap_detects_real_difference_only():
    y, p_good = _data(signal=2.5)
    _, p_noise = _data(signal=0.0)
    groups = np.arange(len(y))
    d = paired_bootstrap_diff(y, p_good, p_noise[: len(y)], groups, n_boot=200)
    assert d["pr_auc"]["significant"] and d["pr_auc"]["diff"] > 0
    same = paired_bootstrap_diff(y, p_good, p_good, groups, n_boot=50)
    assert not same["pr_auc"]["significant"] and same["pr_auc"]["diff"] == 0


def test_psi_null_and_domain_classifier():
    ref = pd.DataFrame({"a": RNG.normal(size=3000), "b": RNG.normal(size=3000), "c": RNG.choice(["x", "y"], 3000)})
    same = pd.DataFrame({"a": RNG.normal(size=800), "b": RNG.normal(size=800), "c": RNG.choice(["x", "y"], 800)})
    shifted = same.assign(a=same["a"] + 1.5)
    null = psi_null_thresholds(ref, sample_size=800, n_boot=10)
    assert set(null["per_feature"]) == {"a", "b", "c"}
    tbl = pd.DataFrame({"feature": ["a", "b"], "psi": [0.9, 0.0]})
    assert exceed_null(tbl, null)["features_over_null"] == ["a"]
    assert domain_classifier(ref, same, folds=2)["auc"] < 0.62
    dc = domain_classifier(ref, shifted, folds=2)
    assert dc["auc"] > 0.75 and dc["top_features"][0]["feature"] == "a"


def test_coding_drift_reports_version_gap():
    df = pd.DataFrame({"anchor_year_group": ["2008 - 2010"] * 50 + ["2017 - 2019"] * 50,
                       "icd_version_group": ["ICD9"] * 50 + ["ICD10"] * 50,
                       "hx_ckd": [0] * 50 + [1] * 25 + [0] * 25})
    r = coding_drift(df)
    assert r["available"] and r["largest_version_gaps"]["hx_ckd"] == pytest.approx(0.5)


def test_confidence_validation_and_risk_coverage():
    y, p = _data(n=3000)
    noisy = RNG.random(len(y)) < 0.2
    p_obs = np.where(noisy, RNG.random(len(y)), p)          # 20% ca bị nhiễu → mô hình "không chắc"
    flags = pd.DataFrame({"reasons": [["LOW_CONFIDENCE"] if f else [] for f in noisy]})
    cv = confidence_validation(y, p_obs, flags)
    assert cv["brier_scaled_ratio_lowconf_vs_confident"] > 1   # luật bắt đúng nhóm khó
    rc = risk_coverage(y, p_obs, noisy.astype(float))
    assert rc.iloc[-1]["brier"] <= rc.iloc[0]["brier"]


def test_clinical_plausibility_and_labels():
    d = pd.DataFrame({"feature": ["bun_max", "sodium_min", "age", "ntprobnp_missing", "omr_bmi_last"],
                      "spearman_value_vs_shap": [0.8, 0.7, 0.05, 0.9, 0.5]})
    pl = clinical_plausibility(d).set_index("feature")
    assert pl.loc["bun_max", "verdict"] == "đúng chiều"
    assert pl.loc["sodium_min", "verdict"] == "NGƯỢC CHIỀU"
    assert pl.loc["age", "verdict"] == "không đơn điệu"
    assert "ntprobnp_missing" not in pl.index and "omr_bmi_last" not in pl.index
    assert describe("bun_max") == {"label": "BUN (urê máu) cao nhất", "unit": "mg/dL"}
    assert describe("hx_ckd")["label"].startswith("Tiền sử")


def test_clinical_baselines_scores():
    from ml.baselines import evaluate_baselines, gwtg_hf, hospital_score, lace_score
    n = 600
    df = pd.DataFrame({"age": RNG.integers(40, 95, n), "bun_first": RNG.uniform(5, 120, n),
                       "sodium_first": RNG.uniform(125, 145, n), "hx_copd": RNG.integers(0, 2, n),
                       "race_group": RNG.choice(["WHITE", "BLACK"], n), "sbp_min": RNG.uniform(60, 180, n),
                       "hr_mean": RNG.uniform(50, 130, n)})
    df.loc[:100, "sbp_min"] = np.nan
    full, red = gwtg_hf(df), gwtg_hf(df, reduced=True)
    assert full.isna().sum() == 101 and red.notna().all()
    y = (RNG.random(n) < 1 / (1 + np.exp(-(red - red.mean()) / 6))).astype(int)
    rows = evaluate_baselines(df, y, "p1", p_model=RNG.random(n))
    assert rows[0]["baseline"].startswith("Dummy") and rows[1]["roc_auc"] > 0.5
    d2 = pd.DataFrame({"hemoglobin_last": [10, 14], "hemoglobin_min": [9, 13], "sodium_last": [130, 140],
                       "sodium_min": [129, 139], "admission_type_group": ["EMERGENCY", "ELECTIVE"],
                       "n_prior_adm_365d": [6, 0], "los_days": [7, 2], "dx_ckd": [1, 0], "dx_copd": [1, 0]})
    assert list(hospital_score(d2)) == [1 + 1 + 1 + 5 + 2, 0]
    assert list(lace_score(d2)) == [5 + 3 + 2 + 4, 2]

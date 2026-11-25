"""HeartGuard AI — Giai đoạn 5: độ tin cậy, chọn mô hình, mở tập test MỘT lần, đóng gói.

Luồng (mọi bước trước "test" chỉ dùng train + validation = tập phát triển "dev"):

  1. CV 5 fold theo bệnh nhân trên dev cho 3 ứng viên GĐ4 (XGBoost, LightGBM, LR với tham số Optuna
     tốt nhất đọc từ experiments/*/results.csv) → chọn mô hình theo PR-AUC trung bình. Sửa điểm yếu
     GĐ4: Optuna chọn tham số VÀ báo điểm trên cùng tập validation.
  2. Hiệu chỉnh xác suất (none/Platt/isotonic) chọn bằng CV trên dự đoán out-of-fold; chọn ngưỡng
     LOW/MEDIUM/HIGH trên xác suất đã hiệu chỉnh.
  3. SHAP toàn cục + độ ổn định giữa các fold; giải thích từng ca có trong gói mô hình.
  4. Độ bền khi thiếu dữ liệu (che dần thông tin, đánh giá out-of-fold).
  5. Nhóm bệnh nhân: hiệu năng + hiệu chỉnh + CI bootstrap theo bệnh nhân, chênh lệch công bằng.
  6. Drift theo lô: PSI dữ liệu/điểm, drift nhãn, huấn luyện giai đoạn cũ → đánh giá giai đoạn mới,
     hiệu chỉnh lại intercept làm biện pháp xử lý, drift thứ hạng SHAP giữa các giai đoạn.
  7. Luật xem xét thủ công + khối lượng hàng đợi.
  8. (chỉ khi --open-test) đánh giá test MỘT lần, CI bootstrap 1000 lần theo bệnh nhân; ghi khóa.
  9. Đóng gói mô hình cuối (+ tùy chọn tải lên S3).

Chạy thật trên EC2 (Hùng bật máy; dữ liệu không rời AWS):
    python -m experiments.stage5.run_stage5 --jobs 8                 # bước 1–7, 9; test còn đóng
    python -m experiments.stage5.run_stage5 --jobs 8 --open-test --upload   # khi nhóm đã chốt

Thử với dữ liệu giả:
    python tools/make_synthetic_gold.py --rows 6000 --out data/gold_fake.csv
    python -m experiments.common.split --gold data/gold_fake.csv --out data/splits/split_fake.json
    python -m experiments.stage5.run_stage5 --gold data/gold_fake.csv --split data/splits/split_fake.json \
        --synthetic --quick --open-test
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from experiments.common.config import GROUP_KEY, LABEL, REPO_ROOT, SEED
from experiments.common.data import feature_columns, load_gold
from experiments.common.runner import split_feature_types
from experiments.common.split import apply_split, load_manifest
from trustworthy_ai.bundle import save_bundle, upload_bundle, write_json
from trustworthy_ai.calibration import Calibrator, choose_thresholds, crossfit_calibrate, risk_level, select_calibrator
from trustworthy_ai.drift import DriftReference, period_order
from trustworthy_ai.evaluation import (bootstrap_ci, calibration_stats, discrimination, fit_logistic_1d,
                                       full_metrics, logit)
from trustworthy_ai.explain import (Explainer, direction_summary, family_importance, global_importance,
                                    importance_stability)
from trustworthy_ai.review import build_rules, flag_cases, queue_report
from trustworthy_ai.robustness import missing_fraction, missingness_curve, summarize_curve
from trustworthy_ai.subgroup import add_subgroup_columns, fairness_gaps, subgroup_table

LOG = logging.getLogger("heartguard.stage5")
DEFAULT_GOLD = "s3://heartguard-mimic-group11/gold/p1/v1/"
DEFAULT_SPLIT = "s3://heartguard-mimic-group11/splits/p1/v1/split_manifest.json"
LOCKED_SPLIT_SHA256 = "584278c737fcbc884e88bb57ed1dc6f5f64fd0c211e645af16cd1e4532390ab5"
S3_ARTIFACTS = "s3://heartguard-mimic-group11/model-artifacts/p1"

RESULTS = {"XGBoost": "experiments/khoe/results.csv", "LightGBM": "experiments/chien/results.csv",
           "LogisticRegression": "experiments/hung/results.csv"}
DROP_KEYS = {"n_jobs", "random_state", "objective", "tree_method", "max_iter", "n_trials", "solver", "verbosity"}


# ----------------------------------------------------------------------------------------------
# Ứng viên
# ----------------------------------------------------------------------------------------------
def best_params(model: str) -> dict:
    """Tham số Optuna tốt nhất của GĐ4 (dòng *optuna* có PR-AUC cao nhất)."""
    path = REPO_ROOT / RESULTS[model]
    if not path.exists():
        return {}
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8")) if "optuna" in r["case_id"]]
    if not rows:
        return {}
    row = max(rows, key=lambda r: float(r["pr_auc"]))
    return {k: v for k, v in json.loads(row["params"]).items() if k not in DROP_KEYS}


def candidate_pipelines(df: pd.DataFrame, feats: list[str], jobs: int, names: list[str]) -> dict:
    num, cat = split_feature_types(df, feats)
    out = {}
    if "XGBoost" in names:
        from experiments.khoe.run_xgboost import build_pipeline as xgb_pipe
        out["XGBoost"] = (xgb_pipe(num, cat, jobs=jobs, params=best_params("XGBoost")), best_params("XGBoost"))
    if "LightGBM" in names:
        from experiments.chien.run_lightgbm import build_pipeline as lgbm_pipe
        out["LightGBM"] = (lgbm_pipe(num, cat, jobs=jobs, params=best_params("LightGBM")), best_params("LightGBM"))
    if "LogisticRegression" in names:
        from experiments.hung.run_lr import lr_pipeline
        p = best_params("LogisticRegression") or {"C": 1.0, "penalty": "l2", "class_weight": None}
        out["LogisticRegression"] = (lr_pipeline(df, feats, solver="liblinear", **p), p)
    return out


# ----------------------------------------------------------------------------------------------
# Các bước
# ----------------------------------------------------------------------------------------------
def make_folds(y, groups, k: int) -> np.ndarray:
    folds = np.full(len(y), -1)
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
    for i, (_, te) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        folds[te] = i
    return folds


def cross_validate(pipe, X, y, folds) -> dict:
    oof = np.zeros(len(y))
    models, per_fold = [], []
    t0 = time.perf_counter()
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        m = clone(pipe).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        models.append(m)
        per_fold.append({"fold": int(k), **discrimination(y[te], oof[te])})
    return {"oof": oof, "models": models, "per_fold": per_fold, "seconds": time.perf_counter() - t0}


def summarize_cv(cv: dict) -> dict:
    pf = pd.DataFrame(cv["per_fold"])
    return {"pr_auc_mean": pf["pr_auc"].mean(), "pr_auc_sd": pf["pr_auc"].std(ddof=1),
            "roc_auc_mean": pf["roc_auc"].mean(), "roc_auc_sd": pf["roc_auc"].std(ddof=1),
            "brier_mean": pf["brier"].mean(), "seconds": cv["seconds"], "per_fold": cv["per_fold"]}


def temporal_validation(pipe, X, y, period, groups, method: str, k: int) -> dict:
    """Huấn luyện trên 2 giai đoạn đầu, đánh giá từng giai đoạn sau. Bộ hiệu chỉnh của mô hình
    thời gian học bằng CV trong chính các giai đoạn huấn luyện (không nhìn dữ liệu tương lai)."""
    periods = period_order(period)
    train_p = periods[:2]
    tr = np.isin(period, train_p)
    folds = make_folds(y[tr], groups[tr], k)
    cv = cross_validate(pipe, X[tr], y[tr], folds)
    cal = Calibrator(method).fit(cv["oof"], y[tr])
    model = clone(pipe).fit(X[tr], y[tr])
    rows = []
    rng = np.random.default_rng(SEED)
    for per in periods[2:]:
        te = period == per
        if te.sum() == 0:
            continue
        p = cal.transform(model.predict_proba(X[te])[:, 1])
        yy = y[te]
        m = full_metrics(yy, p, threshold=0.5)
        row = {"period": per, "n": int(te.sum()), "n_pos": int(yy.sum()), "positive_rate": m["positive_rate"],
               "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "pr_auc_lift": m["pr_auc_lift"], "brier": m["brier"],
               "ece": m["ece"], "oe_ratio": m["oe_ratio"], "mean_predicted": m["mean_predicted"]}
        # Biện pháp: hiệu chỉnh lại intercept trên nửa bệnh nhân của giai đoạn, đánh giá nửa còn lại
        subj = np.unique(groups[te])
        half = set(rng.choice(subj, size=len(subj) // 2, replace=False).tolist())
        a_mask = np.array([g in half for g in groups[te]])
        if yy[a_mask].sum() >= 10 and yy[~a_mask].sum() >= 10:
            a, _ = fit_logistic_1d(logit(p[a_mask]), yy[a_mask], offset=logit(p[a_mask]), fit_slope=False)
            p_new = 1 / (1 + np.exp(-(logit(p[~a_mask]) + a)))
            row["oe_ratio_heldout_before"] = calibration_stats(yy[~a_mask], p[~a_mask])["oe_ratio"]
            row["oe_ratio_heldout_after_recal"] = calibration_stats(yy[~a_mask], p_new)["oe_ratio"]
            row["ece_heldout_before"] = calibration_stats(yy[~a_mask], p[~a_mask])["ece"]
            row["ece_heldout_after_recal"] = calibration_stats(yy[~a_mask], p_new)["ece"]
            row["recal_intercept"] = a
        rows.append(row)
    return {"train_periods": train_p, "rows": rows,
            "train_cv_pr_auc": float(np.mean([f["pr_auc"] for f in cv["per_fold"]]))}


def explanation_drift(pipe, X, y, period, sample: pd.DataFrame, background: pd.DataFrame) -> dict:
    periods = period_order(period)
    imps, tops = [], {}
    for per in periods:
        m = period == per
        if y[m].sum() < 20:
            continue
        model = clone(pipe).fit(X[m], y[m])
        shap_df, _ = Explainer(model, background).contributions(sample)
        imp = global_importance(shap_df)
        imps.append(imp)
        tops[per] = imp["feature"].head(5).tolist()
    stab = []
    for i in range(1, len(imps)):
        s = importance_stability([imps[0], imps[i]])
        stab.append({"period": list(tops)[i], "spearman_vs_ref": s["spearman_mean"], "jaccard_top20": s["jaccard_top20_mean"]})
    return {"reference": list(tops)[0] if tops else None, "top5_by_period": tops, "stability": stab}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ----------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GĐ5: độ tin cậy + chọn mô hình + test một lần + đóng gói")
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--out", default=None, help="Thư mục kết quả chi tiết (mặc định reports/stage5/<run_id>)")
    ap.add_argument("--docs-dir", default=None, help="Nơi ghi báo cáo .md + hình (mặc định docs/stage5 với dữ liệu thật)")
    ap.add_argument("--candidates", nargs="+", default=["XGBoost", "LightGBM", "LogisticRegression"])
    ap.add_argument("--model", default=None, help="Ép chọn mô hình (mặc định: PR-AUC CV cao nhất)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--subgroup-boot", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3, help="Số lần lặp mỗi mức che dữ liệu")
    ap.add_argument("--shap-sample", type=int, default=5000)
    ap.add_argument("--open-test", action="store_true", help="Mở tập test (chỉ một lần)")
    ap.add_argument("--force-reopen-test", action="store_true", help="Chỉ dùng khi nhóm đồng ý, sẽ ghi vào khóa")
    ap.add_argument("--upload", action="store_true", help=f"Tải gói mô hình lên {S3_ARTIFACTS}/<version>/")
    ap.add_argument("--synthetic", action="store_true", help="Dữ liệu giả: bỏ kiểm tra split đã khóa")
    ap.add_argument("--quick", action="store_true", help="Giảm số bootstrap/lặp để thử nhanh")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.quick:
        a.n_boot, a.subgroup_boot, a.repeats, a.shap_sample = 100, 30, 1, 1500

    real = str(a.gold).startswith(("s3://", "s3a://"))
    if a.synthetic and real:
        raise SystemExit("--synthetic không dùng với dữ liệu thật trên S3")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(a.out or REPO_ROOT / "reports/stage5" / run_id)
    docs = Path(a.docs_dir or (REPO_ROOT / "docs/stage5" if real else out / "docs"))
    out.mkdir(parents=True, exist_ok=True)
    lock_path = (REPO_ROOT / "docs/stage5/test_lock.json") if real else out / "test_lock.json"

    # ---------- dữ liệu ----------
    LOG.info("Đọc Gold và split...")
    df = load_gold(a.gold)
    manifest = load_manifest(a.split)
    if not a.synthetic and manifest.get("sha256") != LOCKED_SPLIT_SHA256:
        raise SystemExit("Split không phải bản V1 đã khóa (docs/stage3/split_and_baseline.md)")
    if a.open_test and lock_path.exists() and not a.force_reopen_test:
        raise SystemExit(f"Tập test ĐÃ mở trước đây ({lock_path}). Không mở lại.")
    parts = apply_split(df, manifest, parts=("train", "val", "test") if a.open_test else ("train", "val"))
    dev = pd.concat([parts["train"], parts["val"]], ignore_index=True)
    feats = feature_columns(df)
    X, y, g = dev[feats], dev[LABEL].astype(int).to_numpy(), dev[GROUP_KEY].to_numpy()
    period = dev["anchor_year_group"].astype(str).to_numpy()
    R = {"run_id": run_id, "git_commit": git_commit(), "data_version": manifest.get("data_version"),
         "split_sha256": manifest.get("sha256"), "real_data": real, "n_features": len(feats),
         "dev": {"rows": len(dev), "patients": int(dev[GROUP_KEY].nunique()), "positives": int(y.sum())}}
    folds = make_folds(y, g, a.folds)

    # ---------- 1. CV + chọn mô hình ----------
    cands = candidate_pipelines(df, feats, a.jobs, a.candidates)
    cvs, cv_table = {}, {}
    for name, (pipe, params) in cands.items():
        LOG.info("CV %d fold: %s", a.folds, name)
        cvs[name] = cross_validate(pipe, X, y, folds)
        cv_table[name] = {**summarize_cv(cvs[name]), "params": params}
    chosen = a.model or max(cv_table, key=lambda m: cv_table[m]["pr_auc_mean"])
    best_pf = pd.DataFrame(cv_table[chosen]["per_fold"])["pr_auc"].to_numpy()
    for name in cv_table:  # chênh lệch ghép cặp theo fold so với mô hình được chọn
        d = best_pf - pd.DataFrame(cv_table[name]["per_fold"])["pr_auc"].to_numpy()
        cv_table[name]["pr_auc_diff_vs_chosen_mean"] = float(d.mean())
        cv_table[name]["pr_auc_diff_vs_chosen_sd"] = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    R["cv"] = cv_table
    R["chosen_model"] = chosen
    R["chosen_by"] = "--model (nhóm chọn)" if a.model else "PR-AUC CV cao nhất"
    R["best_by_cv"] = max(cv_table, key=lambda m: cv_table[m]["pr_auc_mean"])
    LOG.info("Chọn mô hình: %s", chosen)

    # ---------- 2. Hiệu chỉnh + ngưỡng ----------
    calib = {}
    for name, cv in cvs.items():
        method, rep = select_calibrator(cv["oof"], y, folds)
        calib[name] = {"method": method, "comparison": rep,
                       "raw": {**discrimination(y, cv["oof"]), **{k: v for k, v in calibration_stats(y, cv["oof"]).items()
                                                                if k != "bins"}}}
    R["calibration"] = calib
    method = calib[chosen]["method"]
    p_raw = cvs[chosen]["oof"]
    p_cal = crossfit_calibrate(p_raw, y, folds, method)   # trung thực, dùng cho mọi phân tích trên dev
    calibrator = Calibrator(method).fit(p_raw, y)          # bộ hiệu chỉnh cuối cho gói mô hình
    R["reliability_dev"] = {"raw": calibration_stats(y, p_raw)["bins"], "calibrated": calibration_stats(y, p_cal)["bins"]}
    thresholds = choose_thresholds(y, p_cal)
    R["thresholds"] = thresholds
    R["dev_metrics"] = full_metrics(y, p_cal, thresholds["medium_high"])
    R["dev_risk_levels"] = pd.Series(risk_level(p_cal, thresholds)).value_counts().to_dict()
    lv = risk_level(p_cal, thresholds)
    R["dev_risk_level_mortality"] = {k: float(y[lv == k].mean()) for k in ("LOW", "MEDIUM", "HIGH") if (lv == k).any()}

    # ---------- mô hình cuối ----------
    final_pipe = cands[chosen][0]
    LOG.info("Huấn luyện mô hình cuối trên train+validation...")
    final = clone(final_pipe).fit(X, y)
    fold_models = cvs[chosen]["models"]
    background = X.sample(min(2000, len(X)), random_state=SEED) if chosen == "LogisticRegression" else None
    std_dev = np.column_stack([calibrator.transform(m.predict_proba(X)[:, 1]) for m in fold_models]).std(axis=1)

    # ---------- 3. SHAP ----------
    LOG.info("SHAP...")
    sample = X.sample(min(a.shap_sample, len(X)), random_state=SEED)
    expl = Explainer(final, background)
    shap_df, base = expl.contributions(sample)
    imp = global_importance(shap_df)
    fold_imps = [global_importance(Explainer(m, background).contributions(sample)[0]) for m in fold_models]
    R["shap"] = {"method": "TreeSHAP" if chosen != "LogisticRegression" else "LinearSHAP",
                 "sample_n": len(sample), "base_value_mean": float(np.mean(base)),
                 "top_features": imp.head(25).to_dict(orient="records"),
                 "families": family_importance(imp).head(15).to_dict(orient="records"),
                 "direction": direction_summary(shap_df, sample, imp["feature"].head(15).tolist()).to_dict(orient="records"),
                 "fold_stability": importance_stability([imp] + fold_imps),
                 "additivity_max_error": float(np.max(np.abs(
                     base + shap_df.sum(axis=1).to_numpy() - logit(final.predict_proba(sample)[:, 1]))))}

    # ---------- 4. Thiếu dữ liệu ----------
    LOG.info("Độ bền khi thiếu dữ liệu...")

    def oof_predict(Xm):
        out_p = np.empty(len(Xm))
        for k, m in enumerate(fold_models):
            te = folds == k
            out_p[te] = calibrator.transform(m.predict_proba(Xm[te])[:, 1])
        return out_p

    curve = missingness_curve(oof_predict, X, y, repeats=a.repeats)
    R["robustness_dev"] = summarize_curve(curve).to_dict(orient="records")

    # ---------- 5. Nhóm bệnh nhân ----------
    LOG.info("Phân tích nhóm bệnh nhân...")
    sg = add_subgroup_columns(dev, missing_fraction(X))
    sg_table = subgroup_table(sg, y, p_cal, g, thresholds["medium_high"], n_boot=a.subgroup_boot)
    R["subgroup_dev"] = sg_table.to_dict(orient="records")
    R["fairness_dev"] = fairness_gaps(sg_table).to_dict(orient="records")

    # ---------- 6. Drift ----------
    LOG.info("Drift theo giai đoạn...")
    periods = period_order(period)
    ref_p = periods[0]
    ref = DriftReference.fit(X[period == ref_p], p_cal[period == ref_p])
    drift_rows = []
    for per in periods:
        m = period == per
        c = ref.compare(X[m], p_cal[m])
        drift_rows.append({"period": per, "n": int(m.sum()), "n_pos": int(y[m].sum()),
                           "positive_rate": float(y[m].mean()), "mean_predicted_cal": float(p_cal[m].mean()),
                           "oe_ratio": float(y[m].mean() / p_cal[m].mean()),
                           "share_features_major": c["share_major"], "share_features_moderate_plus": c["share_moderate_or_more"],
                           "score_psi": c.get("score_psi"), "status": c["status"],
                           "top_features": c["features"].head(8)[["feature", "psi", "level", "missing_rate_delta"]]
                           .to_dict(orient="records"),
                           "psi_all": c["features"][["feature", "psi"]].set_index("feature")["psi"].to_dict()})
    R["drift"] = {"reference_period": ref_p, "periods": drift_rows}
    R["temporal_validation"] = temporal_validation(final_pipe, X, y, period, g, method, a.folds)
    R["explanation_drift"] = explanation_drift(final_pipe, X, y, period, sample, background)

    # ---------- 7. Xem xét thủ công ----------
    rules = build_rules(thresholds=thresholds, X_train=X, std_train=std_dev)
    flags_dev = flag_cases(X, p_cal, rules, std_dev)
    R["review_dev"] = queue_report(flags_dev, y)
    R["review_rules_summary"] = {k: v for k, v in rules.items() if k not in ("valid_ranges", "ood_bounds")}

    # ---------- 8. Test (một lần) ----------
    if a.open_test:
        LOG.info("MỞ TẬP TEST (một lần)...")
        test = parts["test"]
        Xt, yt, gt = test[feats], test[LABEL].astype(int).to_numpy(), test[GROUP_KEY].to_numpy()
        tests = {}
        for name, (pipe, _) in cands.items():
            mdl = final if name == chosen else clone(pipe).fit(X, y)
            cal_n = calibrator if name == chosen else Calibrator(calib[name]["method"]).fit(cvs[name]["oof"], y)
            pr_raw = mdl.predict_proba(Xt)[:, 1]
            pr = cal_n.transform(pr_raw)
            thr = thresholds["medium_high"] if name == chosen else choose_thresholds(
                y, crossfit_calibrate(cvs[name]["oof"], y, folds, calib[name]["method"]))["medium_high"]
            tests[name] = {"metrics": full_metrics(yt, pr, thr),
                           "metrics_uncalibrated": {k: v for k, v in full_metrics(yt, pr_raw, thr).items()
                                                    if k in ("brier", "ece", "oe_ratio", "calibration_slope")},
                           "ci95": bootstrap_ci(yt, pr, gt, thr, n_boot=a.n_boot)}
            if name == chosen:
                p_test = pr
        R["test"] = {"rows": len(test), "patients": int(test[GROUP_KEY].nunique()), "positives": int(yt.sum()),
                     "models": tests,
                     "reliability": {"calibrated": calibration_stats(yt, p_test)["bins"]},
                     "at_medium": full_metrics(yt, p_test, thresholds["low_medium"]),
                     "risk_levels": pd.Series(risk_level(p_test, thresholds)).value_counts().to_dict()}
        lvt = risk_level(p_test, thresholds)
        R["test"]["risk_level_mortality"] = {k: float(yt[lvt == k].mean()) for k in ("LOW", "MEDIUM", "HIGH") if (lvt == k).any()}
        std_t = np.column_stack([calibrator.transform(m.predict_proba(Xt)[:, 1]) for m in fold_models]).std(axis=1)
        R["review_test"] = queue_report(flag_cases(Xt, p_test, rules, std_t), yt)
        sg_t = subgroup_table(add_subgroup_columns(test, missing_fraction(Xt)), yt, p_test, gt,
                              thresholds["medium_high"], n_boot=a.subgroup_boot)
        R["subgroup_test"] = sg_t.to_dict(orient="records")
        R["fairness_test"] = fairness_gaps(sg_t).to_dict(orient="records")
        curve_t = missingness_curve(lambda Z: calibrator.transform(final.predict_proba(Z)[:, 1]), Xt, yt,
                                    repeats=a.repeats)
        R["robustness_test"] = summarize_curve(curve_t).to_dict(orient="records")

    # ---------- 9. Đóng gói ----------
    version = (f"{'synthetic-' if not real else ''}p1-{chosen.lower()}-"
               f"{datetime.now(timezone.utc):%Y%m%d}-{git_commit()}")
    R["model_version"] = version
    metrics_pkg = {"cv": {k: {kk: vv for kk, vv in v.items() if kk != "per_fold"} for k, v in cv_table.items()},
                   "chosen": chosen, "calibration_method": method, "dev_oof": R["dev_metrics"],
                   "test": R.get("test", {}).get("models", {}).get(chosen)}
    from experiments.stage5.report import model_card
    cat_cols = set(split_feature_types(dev, feats)[1])
    bundle_dir = save_bundle(REPO_ROOT / "artifacts/models/p1" / version, model=final, calibrator=calibrator,
                             fold_models=fold_models, feature_columns=feats,
                             dtypes={c: ("object" if c in cat_cols else "float") for c in feats},
                             thresholds=thresholds, rules=rules, drift_ref=ref, metrics=metrics_pkg,
                             model_card=model_card(R), model_version=version, background=background)
    R["bundle_dir"] = str(bundle_dir)
    if a.upload:
        upload_bundle(bundle_dir, f"{S3_ARTIFACTS}/{version}/")
        R["bundle_s3"] = f"{S3_ARTIFACTS}/{version}/"
        LOG.info("Đã tải gói mô hình lên %s", R["bundle_s3"])

    if a.open_test:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {"openings": []}
        prev["openings"].append({"at": run_id, "model_version": version, "git_commit": git_commit(),
                                 "split_sha256": manifest.get("sha256"), "forced": bool(a.force_reopen_test)})
        write_json(lock_path, prev)

    # ---------- báo cáo ----------
    write_json(out / "stage5_results.json", R)
    from experiments.stage5.report import write_report
    md = write_report(R, docs)
    LOG.info("Xong. Kết quả chi tiết: %s · Báo cáo: %s · Gói mô hình: %s", out, md, bundle_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

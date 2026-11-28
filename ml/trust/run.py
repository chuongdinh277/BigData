"""HeartGuard AI — đánh giá độ tin cậy, chọn mô hình, mở tập test MỘT lần, đóng gói (Giai đoạn 5).

Luồng (mọi bước trước "test" chỉ dùng train + validation = tập phát triển "dev"):

  1. CV 5 fold theo bệnh nhân trên dev cho 3 ứng viên GĐ4 (XGBoost, LightGBM, LR với tham số Optuna
     tốt nhất đọc từ results/models/*.csv) → chọn mô hình theo PR-AUC trung bình. Sửa điểm yếu
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

Phân tích bổ sung (mỗi mục trả lời một câu chữ của đề Subject 4):
  - Mốc so sánh: Dummy + thang điểm lâm sàng (GWTG-HF cho P1, HOSPITAL/LACE cho P2)  → "suitable baselines"
  - Learning curve: PR-AUC theo % dữ liệu huấn luyện                                     → giá trị của dữ liệu lớn
  - Decision Curve Analysis, bảng điểm cắt, đường năng lực review                        → hiệu chuẩn có ích lâm sàng
  - Deletion curve + đối chiếu chiều tác động lâm sàng                                    → giải thích đáng tin
  - Ngưỡng PSI bằng bootstrap, bộ phân loại miền, drift mã ICD, 3 chiến lược bảo trì      → data drift + model drift
  - Kiểm chứng luật "độ tin cậy thấp" + risk–coverage                                     → human review có tác dụng

Chạy thật trên EC2 (Hùng bật máy; dữ liệu không rời AWS):
    python -m ml.trust.run --jobs 8                 # bước 1–7, 9; test còn đóng
    python -m ml.trust.run --jobs 8 --open-test --upload   # khi nhóm đã chốt

Thử với dữ liệu giả:
    python tools/make_synthetic_gold.py --rows 6000 --out data/gold_fake.csv
    python -m ml.common.split --gold data/gold_fake.csv --out data/splits/split_fake.json
    python -m ml.trust.run --gold data/gold_fake.csv --split data/splits/split_fake.json \
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

from ml.common.config import GROUP_KEY, LABEL, REPO_ROOT, SEED, TASK, locked_split_sha256
from ml.common.data import data_version as gold_version, feature_columns, load_gold
from ml.common.runner import split_feature_types
from ml.common.split import apply_split, load_manifest
from trustworthy_ai.bundle import save_bundle, upload_bundle, write_json
from trustworthy_ai.calibration import Calibrator, choose_thresholds, crossfit_calibrate, risk_level, select_calibrator
from ml.baselines import evaluate_baselines
from trustworthy_ai.drift import (DriftReference, coding_drift, domain_classifier, exceed_null,
                                  maintenance_strategies, period_order, psi_null_thresholds)
from trustworthy_ai.evaluation import (bootstrap_ci, calibration_stats, discrimination, fit_logistic_1d,
                                       full_metrics, logit, paired_bootstrap_diff)
from trustworthy_ai.explain import (Explainer, clinical_plausibility, deletion_curve, direction_summary,
                                    family_importance, global_importance, importance_stability)
from trustworthy_ai.review import (build_rules, confidence_validation, flag_cases, queue_report,
                                   risk_coverage)
from trustworthy_ai.utility import (DEFAULT_DCA_RANGE, capacity_curve, dca_summary, decision_curve,
                                    operating_table)
from trustworthy_ai.robustness import missing_fraction, missingness_curve, summarize_curve
from trustworthy_ai.subgroup import RELEVANCE, add_subgroup_columns, fairness_gaps, subgroup_table

LOG = logging.getLogger("heartguard.stage5")
# Bài toán chọn bằng HG_TASK (ml/common/config.py); mỗi bài có Gold, split, gói mô hình riêng.
DEFAULT_GOLD = f"s3://heartguard-mimic-group11/gold/{TASK}/v1/"
DEFAULT_SPLIT = f"s3://heartguard-mimic-group11/splits/{TASK}/v1/split_manifest.json"
# Split đã khóa (D-013). P2/P3 khóa khi nhóm tạo split v1 và ghi sha256 vào decision_log.
LOCKED_SPLIT_SHA256 = {t: locked_split_sha256(t) for t in ("p1", "p2", "p3")}  # configs/splits.yaml
S3_ARTIFACTS = f"s3://heartguard-mimic-group11/model-artifacts/{TASK}"
TASK_NAMES = {"p1": "P1 — tử vong trong viện (mốc +24h)",
              "p2": "P2 — tái nhập viện 30 ngày (mốc xuất viện)",
              "p3": "P3 — tử vong 90 ngày sau xuất viện (mốc xuất viện)"}

RESULTS = {"XGBoost": "results/models/xgboost.csv", "LightGBM": "results/models/lightgbm.csv",
           "LogisticRegression": "results/models/logistic.csv"}
LEARNING_FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.0)
# P2/P3: nếu có file results_<task>.csv (Optuna riêng cho bài đó) thì dùng, không thì dùng tham số GĐ4 của P1.
DROP_KEYS = {"n_jobs", "random_state", "objective", "tree_method", "max_iter", "n_trials", "solver", "verbosity"}


# ----------------------------------------------------------------------------------------------
# Ứng viên
# ----------------------------------------------------------------------------------------------
def best_params(model: str) -> dict:
    """Tham số Optuna tốt nhất của GĐ4 (dòng *optuna* có PR-AUC cao nhất)."""
    path = REPO_ROOT / RESULTS[model]
    task_path = path.with_name(f"{path.stem}_{TASK}.csv")
    if TASK != "p1" and task_path.exists():
        path = task_path
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
        from ml.models.xgboost_model import build_pipeline as xgb_pipe
        out["XGBoost"] = (xgb_pipe(num, cat, jobs=jobs, params=best_params("XGBoost")), best_params("XGBoost"))
    if "LightGBM" in names:
        from ml.models.lightgbm_model import build_pipeline as lgbm_pipe
        out["LightGBM"] = (lgbm_pipe(num, cat, jobs=jobs, params=best_params("LightGBM")), best_params("LightGBM"))
    if "LogisticRegression" in names:
        from ml.models.logistic import lr_pipeline
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


def sensitivity_analyses(dev: pd.DataFrame, y, p_cal, pipe, X, folds, skip_extra: bool) -> list[dict]:
    """Phân tích độ nhạy của THIẾT KẾ bài toán (chống câu hỏi vặn khi bảo vệ). Mỗi dòng: chỉ số trên tập con
    hoặc khi bỏ một nhóm đặc trưng, so với toàn bộ (dự đoán ngoài fold, không đụng test)."""
    y = np.asarray(y).astype(int)
    rows = []

    def add(code, question, mask=None, p=None, note=""):
        m = np.ones(len(y), bool) if mask is None else np.asarray(mask, bool)
        pp = p_cal if p is None else p
        if m.sum() < 50 or len(np.unique(y[m])) < 2:
            rows.append({"code": code, "question": question, "n": int(m.sum()), "note": "không đủ dữ liệu"})
            return
        d = discrimination(y[m], pp[m])
        c = calibration_stats(y[m], pp[m])
        rows.append({"code": code, "question": question, "n": int(m.sum()), "n_pos": int(y[m].sum()),
                     "positive_rate": d["positive_rate"], "pr_auc": d["pr_auc"], "roc_auc": d["roc_auc"],
                     "pr_auc_lift": d["pr_auc"] / d["positive_rate"], "oe_ratio": c["oe_ratio"], "note": note})

    add("C0", "Toàn bộ tập phát triển (mốc so sánh)")
    if "n_prior_hf_adm" in dev:
        add("C2", "Chỉ ca đã có chẩn đoán suy tim từ lần nhập viện trước (suy tim đã biết lúc vào viện)",
            dev["n_prior_hf_adm"].fillna(0).to_numpy() >= 1,
            note="Cohort chọn bằng ICD lúc xuất viện; tập con này không phụ thuộc thông tin đó")
    if "admittime" in dev:
        first = dev.groupby(GROUP_KEY)["admittime"].transform("min") == dev["admittime"]
        add("C3", "Chỉ lượt nhập viện đầu tiên của mỗi bệnh nhân (các dòng độc lập)", first.to_numpy())
    if not skip_extra:
        drop = [c for c in X.columns if c.endswith("_missing") or c.startswith("n_lab_tests")]
        keep = [c for c in X.columns if c not in drop]
        if drop and keep:
            # C5: bỏ đặc trưng "hành vi bệnh viện" (số lần xét nghiệm, cờ thiếu) rồi CV lại cùng fold
            num, cat = split_feature_types(X, keep)
            from ml.common.runner import make_preprocessor
            pipe5 = clone(pipe)
            pipe5.steps[0] = ("prep", make_preprocessor(num, cat, scale=pipe5.steps[-1][0] == "model"
                                                         and type(pipe5.steps[-1][1]).__name__ == "LogisticRegression"))
            oof = np.zeros(len(y))
            for k in np.unique(folds):
                tr, te = folds != k, folds == k
                oof[te] = clone(pipe5).fit(X.loc[tr, keep], y[tr]).predict_proba(X.loc[te, keep])[:, 1]
            add("C5", "Bỏ số lần xét nghiệm và mọi cờ _missing (mô hình có học hành vi bệnh viện không?)",
                p=oof, note=f"bỏ {len(drop)} cột; xác suất chưa hiệu chỉnh nên chỉ so PR/ROC-AUC")
    rows.append({"code": "C4", "question": "Tính xuất viện về hospice là tử vong (P1')", "note":
                 "cần nhãn mới ở data_spark.labels (--hospice-as-death), chạy lại Gold; chưa thực hiện trong lần này"})
    rows.append({"code": "C6", "question": "Bỏ phép đo có storetime NULL", "note":
                 "chạy data_spark.features --drop-null-storetime → Gold riêng, rồi chạy lại ml.trust.run và so PR-AUC"})
    return rows


def learning_curve(pipe, X, y, groups, folds, fractions=LEARNING_FRACTIONS) -> list[dict]:
    """PR-AUC ngoài fold khi chỉ huấn luyện trên x% BỆNH NHÂN của mỗi fold train.
    Đường còn đi lên ở 100% → thêm dữ liệu vẫn có ích (lý do kinh tế của pipeline dữ liệu lớn)."""
    rng = np.random.default_rng(SEED)
    rows = []
    for frac in fractions:
        oof = np.full(len(y), np.nan)
        n_train = []
        for k in np.unique(folds):
            tr_idx = np.where(folds != k)[0]
            subj = np.unique(groups[tr_idx])
            keep = set(rng.choice(subj, size=max(1, int(round(frac * len(subj)))), replace=False).tolist())
            sel = tr_idx[np.array([gg in keep for gg in groups[tr_idx]])]
            if y[sel].sum() < 5:
                continue
            m = clone(pipe).fit(X.iloc[sel], y[sel])
            te = folds == k
            oof[te] = m.predict_proba(X[te])[:, 1]
            n_train.append(len(sel))
        ok = ~np.isnan(oof)
        if ok.sum() == 0:
            continue
        d = discrimination(y[ok], oof[ok])
        rows.append({"fraction": frac, "train_rows_mean": float(np.mean(n_train)) if n_train else 0.0,
                     "pr_auc": d["pr_auc"], "roc_auc": d["roc_auc"], "brier": d["brier"]})
    return rows


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
    ap.add_argument("--docs-dir", default=None, help="Nơi ghi báo cáo .md + hình (mặc định results/trust/<task> với dữ liệu thật)")
    ap.add_argument("--no-bundle", action="store_true",
                    help="Không ghi gói mô hình (dùng cho phân tích độ nhạy như C6, tránh lẫn với mô hình chính)")
    ap.add_argument("--skip-extra", action="store_true",
                    help="Bỏ các phân tích tốn thời gian (learning curve, chiến lược bảo trì, bộ phân loại miền)")
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
    if a.no_bundle and (a.upload or a.open_test):
        raise SystemExit("--no-bundle không dùng cùng --upload/--open-test")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.quick:
        a.n_boot, a.subgroup_boot, a.repeats, a.shap_sample = 100, 30, 1, 1500
    null_boot = 10 if a.quick else 50

    real = str(a.gold).startswith(("s3://", "s3a://"))
    if a.synthetic and real:
        raise SystemExit("--synthetic không dùng với dữ liệu thật trên S3")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(a.out or REPO_ROOT / "reports/stage5" / run_id)
    docs_real = REPO_ROOT / "results" / "trust" / TASK
    docs = Path(a.docs_dir or (docs_real if real else out / "report"))
    out.mkdir(parents=True, exist_ok=True)
    lock_path = (docs_real / "test_lock.json") if real else out / "test_lock.json"

    # ---------- dữ liệu ----------
    LOG.info("Đọc Gold và split...")
    df = load_gold(a.gold)
    manifest = load_manifest(a.split)
    locked = LOCKED_SPLIT_SHA256.get(TASK)
    if not a.synthetic and locked and manifest.get("sha256") != locked:
        raise SystemExit("Split không phải bản V1 đã khóa (configs/splits.yaml)")
    if not a.synthetic and not locked:
        LOG.warning("Split %s chưa được khóa trong code: sha256 %s — ghi vào decision_log trước khi mở test",
                    TASK.upper(), manifest.get("sha256"))
    if a.open_test and lock_path.exists() and not a.force_reopen_test:
        raise SystemExit(f"Tập test ĐÃ mở trước đây ({lock_path}). Không mở lại.")
    parts = apply_split(df, manifest, parts=("train", "val", "test") if a.open_test else ("train", "val"))
    dev = pd.concat([parts["train"], parts["val"]], ignore_index=True)
    feats = feature_columns(df)
    X, y, g = dev[feats], dev[LABEL].astype(int).to_numpy(), dev[GROUP_KEY].to_numpy()
    period = dev["anchor_year_group"].astype(str).to_numpy()
    R = {"task": TASK, "task_name": TASK_NAMES[TASK], "label": LABEL, "run_id": run_id, "git_commit": git_commit(), "data_version": gold_version(a.gold), "split_version": manifest.get("data_version"),
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
    best_name = max(cv_table, key=lambda m: cv_table[m]["pr_auc_mean"])
    comp = {}
    for name in cvs:
        if name != best_name:
            comp[name] = paired_bootstrap_diff(y, cvs[best_name]["oof"], cvs[name]["oof"], g, n_boot=min(a.n_boot, 1000))
    R["model_comparison"] = {"reference": best_name, "vs": comp}
    lr = comp.get("LogisticRegression")
    if best_name != "LogisticRegression" and lr is not None:
        R["recommendation"] = (
            f"{best_name} KHÔNG hơn Logistic Regression có ý nghĩa (ΔPR-AUC {lr['pr_auc']['diff']:+.3f}, "
            f"CI95 [{lr['pr_auc']['ci95'][0]:+.3f}, {lr['pr_auc']['ci95'][1]:+.3f}]) → theo luật đặt trước nên chọn "
            "Logistic Regression (đơn giản, nhanh, dễ giải thích): chạy lại với --model LogisticRegression"
            if not lr["pr_auc"]["significant"] else
            f"{best_name} hơn Logistic Regression có ý nghĩa (ΔPR-AUC {lr['pr_auc']['diff']:+.3f}, "
            f"CI95 [{lr['pr_auc']['ci95'][0]:+.3f}, {lr['pr_auc']['ci95'][1]:+.3f}]) → giữ {best_name}")
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

    # ---------- 2b. Mốc so sánh + learning curve + lợi ích lâm sàng ----------
    LOG.info("Mốc so sánh (Dummy, thang điểm lâm sàng)...")
    R["baselines_dev"] = evaluate_baselines(dev, y, TASK, p_model=p_cal)
    lo, hi = DEFAULT_DCA_RANGE[TASK]
    dca = decision_curve(y, p_cal, lo, hi)
    R["dca_dev"] = {"curve": dca.to_dict(orient="records"), "summary": dca_summary(dca)}
    R["operating_dev"] = operating_table(y, p_cal).to_dict(orient="records")
    R["capacity_dev"] = capacity_curve(y, p_cal).to_dict(orient="records")
    LOG.info("Phân tích độ nhạy của thiết kế bài toán...")
    R["sensitivity"] = sensitivity_analyses(dev, y, p_cal, cands[chosen][0], X, folds, a.skip_extra)
    if not a.skip_extra:
        LOG.info("Learning curve...")
        R["learning_curve"] = learning_curve(cands[chosen][0], X, y, g, folds)

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
                 "plausibility": clinical_plausibility(direction_summary(
                     shap_df, sample, [c for c in sample.columns if pd.api.types.is_numeric_dtype(sample[c])]))
                 .to_dict(orient="records"),
                 "deletion_curve": deletion_curve(final, sample.head(500), shap_df.head(500)).to_dict(orient="records"),
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
    R["subgroup_relevance"] = {k: v for k, v in RELEVANCE.items() if k in sg.columns}
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
    smallest = min(int((period == per).sum()) for per in periods)
    null = psi_null_thresholds(X[period == ref_p], sample_size=smallest, n_boot=null_boot)
    for row, per in zip(drift_rows, periods):
        tbl = pd.DataFrame({"feature": list(row["psi_all"]), "psi": list(row["psi_all"].values())})
        row.update(exceed_null(tbl, null))
        if per != ref_p and not a.skip_extra:
            row["domain_classifier"] = domain_classifier(X[period == ref_p], X[period == per])
    R["drift"] = {"reference_period": ref_p, "periods": drift_rows,
                  "null_thresholds": {k: v for k, v in null.items() if k != "per_feature"}}
    R["coding_drift"] = coding_drift(dev)
    if not a.skip_extra:
        LOG.info("So sánh 3 chiến lược bảo trì mô hình...")
        R["maintenance"] = maintenance_strategies(final_pipe, X, y, period, lambda: Calibrator(method))
    R["temporal_validation"] = temporal_validation(final_pipe, X, y, period, g, method, a.folds)
    R["explanation_drift"] = explanation_drift(final_pipe, X, y, period, sample, background)

    # ---------- 7. Xem xét thủ công ----------
    rules = build_rules(thresholds=thresholds, X_train=X, std_train=std_dev)
    flags_dev = flag_cases(X, p_cal, rules, std_dev)
    R["review_dev"] = queue_report(flags_dev, y)
    R["confidence_validation_dev"] = confidence_validation(y, p_cal, flags_dev)
    R["risk_coverage_dev"] = risk_coverage(y, p_cal, std_dev).to_dict(orient="records")
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
        flags_t = flag_cases(Xt, p_test, rules, std_t)
        R["review_test"] = queue_report(flags_t, yt)
        R["confidence_validation_test"] = confidence_validation(yt, p_test, flags_t)
        R["risk_coverage_test"] = risk_coverage(yt, p_test, std_t).to_dict(orient="records")
        dca_t = decision_curve(yt, p_test, lo, hi)
        R["dca_test"] = {"curve": dca_t.to_dict(orient="records"), "summary": dca_summary(dca_t)}
        R["operating_test"] = operating_table(yt, p_test).to_dict(orient="records")
        sg_t = subgroup_table(add_subgroup_columns(test, missing_fraction(Xt)), yt, p_test, gt,
                              thresholds["medium_high"], n_boot=a.subgroup_boot)
        R["subgroup_test"] = sg_t.to_dict(orient="records")
        R["fairness_test"] = fairness_gaps(sg_t).to_dict(orient="records")
        curve_t = missingness_curve(lambda Z: calibrator.transform(final.predict_proba(Z)[:, 1]), Xt, yt,
                                    repeats=a.repeats)
        R["robustness_test"] = summarize_curve(curve_t).to_dict(orient="records")

    # ---------- 9. Đóng gói ----------
    version = (f"{'synthetic-' if not real else ''}{TASK}-{chosen.lower()}-"
               f"{datetime.now(timezone.utc):%Y%m%d}-{git_commit()}")
    R["model_version"] = version
    metrics_pkg = {"cv": {k: {kk: vv for kk, vv in v.items() if kk != "per_fold"} for k, v in cv_table.items()},
                   "chosen": chosen, "calibration_method": method, "dev_oof": R["dev_metrics"],
                   "test": R.get("test", {}).get("models", {}).get(chosen)}
    from ml.trust.report import model_card
    cat_cols = set(split_feature_types(dev, feats)[1])
    bundle_root = (out / "bundle_not_for_serving") if a.no_bundle else REPO_ROOT / "artifacts/models" / TASK
    if a.no_bundle and (a.upload or a.open_test):
        raise SystemExit("--no-bundle không dùng cùng --upload/--open-test")
    bundle_dir = save_bundle(bundle_root / version, model=final, calibrator=calibrator,
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
    docs.mkdir(parents=True, exist_ok=True)
    write_json(docs / "results.json", R)   # chỉ số liệu tổng hợp; backend /reports đọc file này
    from ml.trust.report import write_report
    md = write_report(R, docs)
    LOG.info("Xong. Kết quả chi tiết: %s · Báo cáo: %s · Gói mô hình: %s", out, md, bundle_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

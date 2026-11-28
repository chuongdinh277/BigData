"""Random Forest, Giai đoạn 4: 3 case ghi vào results/models/random_forest.csv.

    S1_rf_default    cấu hình mặc định (300 cây)
    S1_rf_weighted   class_weight="balanced_subsample" để xử lý nhãn lệch
    S1_rf_optuna     Optuna tìm số cây, độ sâu, min_samples_leaf, max_features; mặc định 50 lượt

Kết quả GĐ4 trên dữ liệu thật (results/models/random_forest.csv): PR-AUC 0.3044 nhưng mất 1,142.8 giây,
chậm ~86 lần Logistic Regression (13.3 giây) mà không tốt hơn rõ → bị loại ở D-015. File này giữ lại
để kết quả TÁI LẬP được.

    python -m ml.models.random_forest --gold $G --split $S --trials 50 --jobs 8
"""
from __future__ import annotations

import argparse

import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ml.common.config import SEED
from ml.common.data import feature_columns, load_gold, split_xy
from ml.common.metrics import evaluate
from ml.common.runner import make_preprocessor, run_case, split_feature_types
from ml.common.split import apply_split, load_manifest

MEMBER = "ha"


def rf_pipeline(df, feats: list[str], jobs: int = 4, **params) -> Pipeline:
    num, cat = split_feature_types(df, feats)
    base = {"n_estimators": 300, "min_samples_leaf": 5, "n_jobs": jobs, "random_state": SEED}
    return Pipeline([("prep", make_preprocessor(num, cat, scale=False)),
                     ("model", RandomForestClassifier(**{**base, **params}))])


def tune(df, manifest: dict, feats: list[str], n_trials: int, jobs: int) -> optuna.Study:
    parts = apply_split(df, manifest, parts=("train", "val"))
    X_tr, y_tr = split_xy(parts["train"], feats)
    X_va, y_va = split_xy(parts["val"], feats)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [8, 12, 16, None]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 50, log=True),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.3, 0.5]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced_subsample"]),
        }
        pipe = rf_pipeline(df, feats, jobs=jobs, **params).fit(X_tr, y_tr)
        return evaluate(y_va, pipe.predict_proba(X_va)[:, 1])["pr_auc"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Random Forest - 3 case cho Giai đoạn 4")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", default="results/models/random_forest.csv")
    ap.add_argument("--member", default=MEMBER)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-optuna", action="store_true")
    a = ap.parse_args(argv)

    df = load_gold(a.gold)
    manifest = load_manifest(a.split)
    feats = feature_columns(df)
    common = dict(member=a.member, model_name="RandomForest", gold_path=a.gold, manifest_path=a.split,
                  results_path=a.out, df=df, manifest=manifest)
    run_case(case_id="S1_rf_default", strategy="S1", pipeline=rf_pipeline(df, feats, a.jobs),
             params={"n_estimators": 300, "min_samples_leaf": 5}, note="mặc định; validation", **common)
    run_case(case_id="S1_rf_weighted", strategy="S1",
             pipeline=rf_pipeline(df, feats, a.jobs, class_weight="balanced_subsample"),
             params={"class_weight": "balanced_subsample"}, note="trọng số lớp; validation", **common)
    if a.skip_optuna:
        return
    study = tune(df, manifest, feats, a.trials, a.jobs)
    best = dict(study.best_params)
    run_case(case_id="S1_rf_optuna", strategy="S1", pipeline=rf_pipeline(df, feats, a.jobs, **best),
             params={**best, "n_trials": a.trials}, note=f"Optuna {a.trials} lượt, chọn theo PR-AUC validation", **common)


if __name__ == "__main__":
    main()

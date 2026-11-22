"""Hùng - Logistic Regression, Giai đoạn 4: 3 case ghi vào experiments/hung/results.csv.

    S0_lr_default      cấu hình mặc định của khung chung (so với mốc S0 của Chưởng)
    S1_lr_balanced     class_weight="balanced" để xử lý nhãn lệch (~4.9% dương)
    S2_lr_optuna       Optuna tìm C, penalty, class_weight; mặc định 50 lượt

Chỉ dùng train + validation: apply_split(parts=("train", "val")) không trả tập test.
Mọi tiền xử lý nằm trong pipeline sklearn (make_preprocessor) nên chỉ học từ train.

Thử trên dữ liệu giả (laptop):
    python tools/make_synthetic_gold.py --rows 5000 --out data/gold_fake.csv
    python -m experiments.common.split --gold data/gold_fake.csv --out data/splits/split_fake.json
    python -m experiments.hung.run_lr --gold data/gold_fake.csv --split data/splits/split_fake.json \
        --out /tmp/hung_fake.csv --trials 5

Chạy thật (EC2):
    G=s3://heartguard-mimic-group11/gold/p1/v1/
    S=s3://heartguard-mimic-group11/splits/p1/v1/split_manifest.json
    python -m experiments.hung.run_lr --gold $G --split $S --out experiments/hung/results.csv --trials 50
"""
from __future__ import annotations

import argparse
import warnings

import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from experiments.common.config import SEED
from experiments.common.data import feature_columns, load_gold, split_xy
from experiments.common.metrics import evaluate
from experiments.common.runner import (
    baseline_pipeline,
    make_preprocessor,
    run_case,
    split_feature_types,
)
from experiments.common.split import apply_split, load_manifest

MEMBER = "hung"

# sklearn >= 1.8 đánh dấu tham số `penalty` sắp bỏ; vẫn chạy đúng, chỉ ẩn cảnh báo cho log gọn.
warnings.filterwarnings("ignore", message=".*penalty.*")


def lr_pipeline(df, feats: list[str], **lr_params) -> Pipeline:
    """Giống baseline (median + chuẩn hóa + one-hot) nhưng cho đổi tham số Logistic Regression."""
    num, cat = split_feature_types(df, feats)
    return Pipeline([
        ("prep", make_preprocessor(num, cat, scale=True)),
        ("model", LogisticRegression(max_iter=2000, random_state=SEED, **lr_params)),
    ])


def tune(df, manifest: dict, feats: list[str], n_trials: int) -> optuna.Study:
    """Optuna tối đa hóa PR-AUC trên validation; chỉ fit trên train."""
    parts = apply_split(df, manifest, parts=("train", "val"))
    X_tr, y_tr = split_xy(parts["train"], feats)
    X_va, y_va = split_xy(parts["val"], feats)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
        # liblinear hỗ trợ cả l1 và l2 cho bài toán nhị phân
        pipe = lr_pipeline(df, feats, solver="liblinear", **params)
        pipe.fit(X_tr, y_tr)
        p_va = pipe.predict_proba(X_va)[:, 1]
        return evaluate(y_va, p_va)["pr_auc"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Hùng: Logistic Regression - 3 case cho Giai đoạn 4")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", default="experiments/hung/results.csv")
    ap.add_argument("--member", default=MEMBER)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--skip-optuna", action="store_true", help="chỉ chạy 2 case đầu (để kiểm tra nhanh)")
    a = ap.parse_args(argv)

    df = load_gold(a.gold)
    manifest = load_manifest(a.split)
    feats = feature_columns(df)
    common = dict(member=a.member, model_name="LogisticRegression", gold_path=a.gold,
                  manifest_path=a.split, results_path=a.out, df=df, manifest=manifest)

    # 1) Cấu hình mặc định
    r1 = run_case(case_id="S0_lr_default", strategy="S0", pipeline=baseline_pipeline(df, feats),
                  params={"imputer": "median", "scaler": "standard", "max_iter": 2000},
                  note="mặc định của khung chung, đánh giá trên validation", **common)
    print("S0_lr_default  ", {k: round(r1[k], 4) for k in ("roc_auc", "pr_auc", "f1", "brier")})

    # 2) Xử lý nhãn lệch bằng class_weight
    r2 = run_case(case_id="S1_lr_balanced", strategy="S1",
                  pipeline=lr_pipeline(df, feats, class_weight="balanced"),
                  params={"imputer": "median", "scaler": "standard", "max_iter": 2000,
                          "class_weight": "balanced"},
                  note="class_weight=balanced, đánh giá trên validation", **common)
    print("S1_lr_balanced ", {k: round(r2[k], 4) for k in ("roc_auc", "pr_auc", "f1", "brier")})

    if a.skip_optuna:
        return

    # 3) Optuna: chỉ train + validation
    study = tune(df, manifest, feats, a.trials)
    best = dict(study.best_params)
    print(f"Optuna {len(study.trials)} lượt, PR-AUC tốt nhất trên validation = {study.best_value:.4f}")
    print("Tham số tốt nhất:", best)
    r3 = run_case(case_id=f"S2_lr_optuna{a.trials}", strategy="S2",
                  pipeline=lr_pipeline(df, feats, solver="liblinear", **best),
                  params={**best, "solver": "liblinear", "max_iter": 2000, "n_trials": a.trials},
                  note=f"Optuna {a.trials} lượt (TPE, seed {SEED}), chọn theo PR-AUC validation", **common)
    print("S2_lr_optuna   ", {k: round(r3[k], 4) for k in ("roc_auc", "pr_auc", "f1", "brier")})


if __name__ == "__main__":
    main()

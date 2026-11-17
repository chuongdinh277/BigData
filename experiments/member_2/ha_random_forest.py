import argparse
import csv
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from experiments.common.config import GROUP_KEY, LABEL, RATIOS, RESULT_COLUMNS, SCHEMA, SEED
from experiments.common.data import feature_columns, load_gold, split_xy
from experiments.common.metrics import evaluate
from experiments.common.runner import make_preprocessor, run_case, split_feature_types
from experiments.common.split import apply_split, load_manifest

LOG = logging.getLogger("heartguard.ha.random_forest")

DEFAULT_GOLD = "s3://heartguard-mimic-group11/gold/p1/v1/"
DEFAULT_SPLIT = "s3://heartguard-mimic-group11/splits/p1/v1/split_manifest.json"
DEFAULT_OUT = "experiments/member_2/results.csv"
LOCKED_SPLIT_SHA256 = "584278c737fcbc884e88bb57ed1dc6f5f64fd0c211e645af16cd1e4532390ab5"
MODEL = "RandomForest"
CASES = ("S1_rf_default", "S1_rf_weighted", "S1_rf_optuna")


def expected_features():
    groups = SCHEMA["feature_columns"]
    names = set()
    for group in groups.values():
        if isinstance(group, list):
            names.update(item["name"] for item in group)
        else:
            names.update(
                name + suffix for name in group["items"] for suffix in group["suffixes"]
            )
            names.update(item["name"] for item in group.get("extra", []))
    return names


def validate_manifest(manifest, locked):
    if manifest.get("seed") != SEED or manifest.get("group_key") != GROUP_KEY:
        raise ValueError("Split phải dùng seed 42 và chia theo subject_id.")
    ratios = manifest.get("ratios")
    if (
        not isinstance(ratios, list)
        or len(ratios) != 3
        or not np.allclose(ratios, RATIOS, atol=1e-9)
    ):
        raise ValueError("Split phải theo tỷ lệ 70/15/15.")
    if locked and (
        manifest.get("sha256") != LOCKED_SPLIT_SHA256
        or manifest.get("data_version") != "v1"
    ):
        raise ValueError("Gold/split không khớp Dataset V1 đã khóa của nhóm.")


def validate_gold(df, features, locked):
    if df.empty or df["hadm_id"].isna().any() or df["hadm_id"].duplicated().any():
        raise ValueError("Gold rỗng hoặc hadm_id bị thiếu/trùng.")
    if not set(df[LABEL].unique()).issubset({0, 1}):
        raise ValueError("Gold có nhãn không hợp lệ.")

    forbidden = set(SCHEMA["forbidden_columns"])
    if forbidden & set(df.columns):
        raise ValueError(f"Gold có cột bị cấm: {sorted(forbidden & set(df.columns))}")
    allowed = expected_features()
    unexpected = set(features) - allowed
    missing = allowed - set(features)
    if unexpected or (locked and missing):
        raise ValueError(
            f"Feature không khớp schema; thừa={sorted(unexpected)}, thiếu={sorted(missing)}"
        )


def make_pipeline(numeric, categorical, jobs, params):
    return Pipeline(
        [
            ("prep", make_preprocessor(numeric, categorical, scale=False)),
            (
                "model",
                RandomForestClassifier(random_state=SEED, n_jobs=jobs, **params),
            ),
        ]
    )


def execute(gold, split, out, trials, jobs, overwrite, synthetic_split):
    if trials < 1 or jobs < 1:
        raise ValueError("--trials và --jobs phải lớn hơn 0.")
    if synthetic_split and any(
        path.startswith(("s3://", "s3a://")) for path in (gold, split)
    ):
        raise ValueError("--synthetic-split chỉ dùng với dữ liệu giả cục bộ.")

    target = Path(out)
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} đã tồn tại; dùng --overwrite để thay.")

    df = load_gold(gold)
    manifest = load_manifest(split)
    locked = not synthetic_split
    validate_manifest(manifest, locked)
    features = feature_columns(df)
    validate_gold(df, features, locked)
    numeric, categorical = split_feature_types(df, features)
    parts = apply_split(df, manifest, parts=("train", "val"))
    X_train, y_train = split_xy(parts["train"], features)
    X_val, y_val = split_xy(parts["val"], features)

    if not len(y_train) or not len(y_val):
        raise ValueError("Train hoặc validation rỗng.")
    if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
        raise ValueError("Train và validation phải có đủ hai nhãn.")

    for name in ("train", "val"):
        part = parts[name]
        if set(part[GROUP_KEY].unique()) != set(manifest["subjects"][name]):
            raise ValueError(f"Bệnh nhân {name} không khớp split manifest.")
        summary = manifest.get("summary", {}).get(name, {})
        if summary.get("rows") is not None and len(part) != int(summary["rows"]):
            raise ValueError(f"Số dòng {name} không khớp split manifest.")
        if summary.get("positives") is not None and int(part[LABEL].sum()) != int(
            summary["positives"]
        ):
            raise ValueError(f"Số nhãn dương {name} không khớp split manifest.")

    base_params = {"random_state": SEED, "n_jobs": jobs}
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".rf-ha-", dir=target.parent) as temp_dir:
        temp_csv = Path(temp_dir) / "results.csv"

        def record(case_id, params, note=""):
            result = run_case(
                case_id=case_id,
                member="Ha",
                strategy="S1",
                model_name=MODEL,
                pipeline=make_pipeline(numeric, categorical, jobs, params),
                gold_path=gold,
                manifest_path=split,
                results_path=str(temp_csv),
                params={**base_params, **params},
                note=note,
                df=df,
                manifest=manifest,
                features=features,
            )
            LOG.info(
                "%s: PR-AUC=%.5f; ROC-AUC=%.5f",
                case_id,
                result["pr_auc"],
                result["roc_auc"],
            )
            return result

        record(CASES[0], {}, "Random Forest defaults; validation only")
        record(
            CASES[1],
            {"class_weight": "balanced"},
            "class_weight=balanced; validation only",
        )

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_categorical(
                    "max_depth", [None, 6, 10, 14, 20]
                ),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2"]
                ),
                "class_weight": trial.suggest_categorical(
                    "class_weight", [None, "balanced", "balanced_subsample"]
                ),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            }
            model = make_pipeline(numeric, categorical, jobs, params)
            model.fit(X_train, y_train)
            return evaluate(y_val, model.predict_proba(X_val)[:, 1])["pr_auc"]

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
        )

        def log_progress(current_study, trial):
            count = trial.number + 1
            if count == 1 or count % 5 == 0 or count == trials:
                LOG.info(
                    "Optuna %d/%d; best validation PR-AUC=%.5f",
                    count,
                    trials,
                    current_study.best_value,
                )

        start = time.perf_counter()
        study.optimize(
            objective,
            n_trials=trials,
            callbacks=[log_progress],
            show_progress_bar=False,
        )
        tuning_seconds = time.perf_counter() - start

        best = record(
            CASES[2],
            study.best_params,
            f"Optuna TPE {trials} trials; best validation PR-AUC="
            f"{study.best_value:.5f}; tuning={tuning_seconds:.1f}s",
        )

        with temp_csv.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        if len(rows) != 3 or list(rows[0]) != RESULT_COLUMNS:
            raise RuntimeError("Kết quả không đúng 3 dòng / 15 cột chung.")
        if [row["case_id"] for row in rows] != list(CASES):
            raise RuntimeError("Thứ tự case trong results.csv không hợp lệ.")
        rows[-1]["seconds"] = round(float(rows[-1]["seconds"]) + tuning_seconds, 5)
        with temp_csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_csv, target)

    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description="Random Forest experiment for P1.")
    parser.add_argument("--gold", default=DEFAULT_GOLD)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--synthetic-split", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = execute(
        args.gold,
        args.split,
        args.out,
        args.trials,
        args.jobs,
        args.overwrite,
        args.synthetic_split,
    )
    print(
        f"Optuna best: PR-AUC={result['pr_auc']:.4f}, "
        f"ROC-AUC={result['roc_auc']:.4f}"
    )


if __name__ == "__main__":
    main()

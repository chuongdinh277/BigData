"""HeartGuard AI GĐ4 — LightGBM P1: default, weighted, Optuna.

Sử dụng Gold P1 V1 và patient-level stratified split (70/15/15, seed 42)
đã khóa của nhóm. Chỉ fit train / đánh giá validation, không đánh giá test.

Chạy từ gốc repository:
    python -m ml.models.lightgbm_model --trials 50 --jobs 4

Đầu ra: results/models/lightgbm.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from ml.common.config import (
    locked_split_sha256,
    GROUP_KEY,
    LABEL,
    RATIOS,
    RESULT_COLUMNS,
    SCHEMA,
    SEED,
)
from ml.common.data import feature_columns, load_gold, split_xy
from ml.common.metrics import evaluate
from ml.common.runner import make_preprocessor, run_case, split_feature_types
from ml.common.split import apply_split, load_manifest

LOG = logging.getLogger("heartguard.chien.lightgbm")

DEFAULT_GOLD = "s3://heartguard-mimic-group11/gold/p1/v1/"
DEFAULT_SPLIT = "s3://heartguard-mimic-group11/splits/p1/v1/split_manifest.json"
DEFAULT_OUT = "results/models/lightgbm.csv"
# Nguồn: configs/splits.yaml. SHA256 là hash danh sách bệnh nhân theo split.
LOCKED_SPLIT_SHA256 = locked_split_sha256("p1")  # configs/splits.yaml

MEMBER = "chien"
STRATEGY = "S1"
MODEL = "LightGBM"
CASE_DEFAULT = "S1_lgbm_default"
CASE_WEIGHTED = "S1_lgbm_weighted"
CASE_OPTUNA = "S1_lgbm_optuna"


def expected_features() -> set[str]:
    """Tên 114 input hợp lệ theo contracts/gold_schema_p1.json."""
    groups = SCHEMA["feature_columns"]
    names: set[str] = set()
    for group in groups.values():
        if isinstance(group, list):
            names.update(item["name"] for item in group)
        else:
            names.update(name + suffix for name in group["items"] for suffix in group["suffixes"])
            names.update(item["name"] for item in group.get("extra", []))
    return names


def validate_manifest(manifest: dict, *, locked: bool) -> None:
    """load_manifest đã xác minh SHA256 nội tại; kiểm tra thêm bản V1 đã khóa của nhóm."""
    if manifest.get("seed") != SEED:
        raise ValueError(f"Split seed khác {SEED}: {manifest.get('seed')}")
    if manifest.get("group_key") != GROUP_KEY:
        raise ValueError("Split không chia theo subject_id")
    ratios = manifest.get("ratios")
    if (
        not isinstance(ratios, list)
        or len(ratios) != 3
        or not np.allclose(ratios, RATIOS, atol=1e-9)
    ):
        raise ValueError(f"Tỷ lệ split không phải 70/15/15: {ratios}")
    if locked:
        if manifest.get("sha256") != LOCKED_SPLIT_SHA256:
            raise ValueError(
                "SHA256 split không trùng bản V1 đã khóa trong configs/splits.yaml"
            )
        if manifest.get("data_version") != "v1":
            raise ValueError(f"Split không thuộc Dataset V1: {manifest.get('data_version')}")


def validate_gold(df, features: list[str], *, locked: bool) -> None:
    """Không để nhãn, cột tương lai hoặc subgroup lọt vào mô hình."""
    if not len(df):
        raise ValueError("Gold không có dòng nào")
    if "hadm_id" not in df or df["hadm_id"].isna().any() or df["hadm_id"].duplicated().any():
        raise ValueError("Gold thiếu hadm_id, có NULL hoặc hadm_id trùng")
    if df[GROUP_KEY].isna().any():
        raise ValueError("Gold có subject_id NULL")
    if not set(df[LABEL].dropna().unique()).issubset({0, 1}):
        raise ValueError("Nhãn Gold không chỉ gồm 0 và 1")

    forbidden = {c.casefold() for c in SCHEMA.get("forbidden_columns", [])}
    bad = sorted(c for c in df.columns if c.casefold() in forbidden)
    if bad:
        raise ValueError(f"Gold chứa cột hậu nghiệm bị cấm: {bad}")

    allowed = expected_features()
    unexpected = sorted(set(features) - allowed)
    if unexpected:
        raise ValueError(f"Feature nằm ngoài contract (nguy cơ leakage): {unexpected}")
    if locked:
        missing = sorted(allowed - set(features))
        if missing:
            raise ValueError(f"Dataset V1 thiếu các feature theo contract: {missing}")


def build_pipeline(
    numeric: list[str], categorical: list[str], *, jobs: int, params: dict | None = None
) -> Pipeline:
    """Median/mode + one-hot fit từ train; không scale vì LightGBM là mô hình cây."""
    if not numeric and not categorical:
        raise ValueError("Không tìm thấy đặc trưng hợp lệ")
    model_params = {
        "objective": "binary",
        "random_state": SEED,
        "n_jobs": jobs,
        "verbosity": -1,
    }
    model_params.update(params or {})
    return Pipeline(
        [
            ("prep", make_preprocessor(numeric, categorical, scale=False)),
            ("model", LGBMClassifier(**model_params)),
        ]
    )


def tune_optuna(
    *,
    X_train,
    y_train,
    X_val,
    y_val,
    numeric: list[str],
    categorical: list[str],
    train_weight: float,
    trials: int,
    jobs: int,
) -> tuple[dict, float, float]:
    """TPE tối ưu average precision (PR-AUC của ml.common) trên validation."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": (depth := trial.suggest_int("max_depth", 3, 10)),
            # Cây độ sâu d có tối đa 2^d lá: num_leaves lớn hơn không có tác dụng (GĐ4 từng chọn 101 lá
            # với depth 6, tức chỉ dùng được 64). Ràng buộc để Optuna không phí lượt thử.
            "num_leaves": trial.suggest_int("num_leaves", 7, min(127, 2 ** depth)),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 5),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            # Không dùng thông tin val/test để tính class weight.
            "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1.0, train_weight]),
        }
        pipeline = build_pipeline(numeric, categorical, jobs=jobs, params=params)
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_val)[:, 1]
        score = float(evaluate(y_val, probabilities)["pr_auc"])
        if not np.isfinite(score):
            raise ValueError("Optuna PR-AUC không hữu hạn")
        return score

    def log_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        count = trial.number + 1
        if count == 1 or count % 5 == 0 or count == trials:
            LOG.info("Optuna %d/%d; best validation PR-AUC=%.5f", count, trials, study.best_value)

    t0 = time.perf_counter()
    study.optimize(
        objective,
        n_trials=trials,
        n_jobs=1,
        callbacks=[log_progress],
        show_progress_bar=False,
    )
    tune_seconds = time.perf_counter() - t0
    return dict(study.best_params), float(study.best_value), tune_seconds


def execute(
    *,
    gold: str = DEFAULT_GOLD,
    split: str = DEFAULT_SPLIT,
    out: str = DEFAULT_OUT,
    trials: int = 50,
    jobs: int = 4,
    overwrite: bool = False,
    synthetic_split: bool = False,
) -> list[dict]:
    """Chạy 3 case LightGBM và ghi 3 dòng CSV atomically sau khi hoàn tất."""
    if trials < 1 or jobs < 1:
        raise ValueError("--trials và --jobs phải >= 1")
    if synthetic_split and any(s.startswith(("s3://", "s3a://")) for s in (gold, split)):
        raise ValueError("--synthetic-split chỉ được dùng với dữ liệu giả cục bộ")

    target = Path(out)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"{target} đã tồn tại; để chạy lại dùng --overwrite hoặc đường dẫn --out khác"
        )

    LOG.info("Đọc Gold P1 và split manifest...")
    df = load_gold(gold)
    manifest = load_manifest(split)
    locked = not synthetic_split
    validate_manifest(manifest, locked=locked)

    features = feature_columns(df)
    validate_gold(df, features, locked=locked)
    numeric, categorical = split_feature_types(df, features)
    parts = apply_split(df, manifest, parts=("train", "val"))
    X_train, y_train = split_xy(parts["train"], features)
    X_val, y_val = split_xy(parts["val"], features)
    if min(len(y_train), len(y_val)) == 0:
        raise ValueError("Tập train hoặc validation rỗng")
    if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
        raise ValueError("Train/validation phải có đủ hai lớp 0 và 1")

    for part_name in ("train", "val"):
        actual = parts[part_name]
        expected_ids = set(manifest["subjects"][part_name])
        if set(actual[GROUP_KEY].unique()) != expected_ids:
            raise ValueError(f"Bệnh nhân {part_name} không khớp manifest")
        summary = manifest.get("summary", {}).get(part_name, {})
        if summary.get("rows") is not None and len(actual) != int(summary["rows"]):
            raise ValueError(f"Số lượt nhập viện {part_name} không khớp manifest")
        if summary.get("positives") is not None and int(actual[LABEL].sum()) != int(summary["positives"]):
            raise ValueError(f"Số nhãn dương {part_name} không khớp manifest")

    positive = int(np.count_nonzero(y_train == 1))
    negative = int(np.count_nonzero(y_train == 0))
    weight = negative / positive
    LOG.info(
        "Train %d; validation %d; features %d (numeric %d, categorical %d); scale_pos_weight=%.4f",
        len(y_train),
        len(y_val),
        len(features),
        len(numeric),
        len(categorical),
        weight,
    )

    common = {
        "member": MEMBER,
        "strategy": STRATEGY,
        "model_name": MODEL,
        "gold_path": gold,
        "manifest_path": split,
        "df": df,
        "manifest": manifest,
        "features": features,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict] = []

    with tempfile.TemporaryDirectory(prefix=".lightgbm-chien-", dir=target.parent) as tempdir:
        temporary_csv = Path(tempdir) / "results.csv"

        def run_case_with_params(case_id: str, params: dict, note: str) -> dict:
            row = run_case(
                case_id=case_id,
                pipeline=build_pipeline(numeric, categorical, jobs=jobs, params=params),
                results_path=str(temporary_csv),
                params={"objective": "binary", "random_state": SEED, "n_jobs": jobs, **params},
                note=note,
                **common,
            )
            LOG.info(
                "%s: PR-AUC=%.5f; ROC-AUC=%.5f; F1=%.5f; Brier=%.5f",
                case_id,
                row["pr_auc"],
                row["roc_auc"],
                row["f1"],
                row["brier"],
            )
            result_rows.append(row)
            return row

        LOG.info("[1/3] LightGBM default")
        run_case_with_params(
            CASE_DEFAULT, {}, "Default; validation only; S0 reference PR-AUC=0.299"
        )

        LOG.info("[2/3] LightGBM weighted")
        run_case_with_params(
            CASE_WEIGHTED,
            {"scale_pos_weight": weight},
            "Weighted by negative/positive on train; validation only",
        )

        LOG.info("[3/3] LightGBM Optuna: %d trials", trials)
        best_params, best_ap, tune_seconds = tune_optuna(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            numeric=numeric,
            categorical=categorical,
            train_weight=weight,
            trials=trials,
            jobs=jobs,
        )
        optimized = run_case_with_params(
            CASE_OPTUNA,
            best_params,
            f"Optuna TPE {trials} trials; best validation PR-AUC={best_ap:.5f}; "
            f"tuning={tune_seconds:.1f}s; reuse validation for selection",
        )
        optimized["seconds"] += tune_seconds

        with temporary_csv.open(encoding="utf-8", newline="") as f:
            written = list(csv.DictReader(f))
        if len(written) != 3 or list(written[0]) != RESULT_COLUMNS:
            raise RuntimeError("File kết quả không đúng 3 dòng / 15 cột theo schema chung")
        written[-1]["seconds"] = round(float(optimized["seconds"]), 5)
        with temporary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            writer.writerows(written)
        if [row["case_id"] for row in written] != [CASE_DEFAULT, CASE_WEIGHTED, CASE_OPTUNA]:
            raise RuntimeError("Thứ tự hoặc case_id kết quả không hợp lệ")
        os.replace(temporary_csv, target)

    LOG.info("HOÀN THÀNH: %s (3 cases, 15 columns)", target)
    return result_rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="GĐ4 — LightGBM Default / Weighted / Optuna")
    parser.add_argument("--gold", default=DEFAULT_GOLD)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="Cho phép ghi đè results.csv cũ")
    parser.add_argument(
        "--synthetic-split",
        action="store_true",
        help="Chỉ dùng với dữ liệu giả cục bộ; bỏ kiểm tra SHA256 bản khóa nhóm",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        execute(
            gold=args.gold,
            split=args.split,
            out=args.out,
            trials=args.trials,
            jobs=args.jobs,
            overwrite=args.overwrite,
            synthetic_split=args.synthetic_split,
        )
    except (ValueError, FileExistsError, RuntimeError) as exc:
        LOG.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

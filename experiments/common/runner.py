"""Chạy một case: fit trên train → chọn ngưỡng + đánh giá trên validation → ghi results.csv.

Mọi bước tiền xử lý phải nằm TRONG pipeline sklearn để chỉ học từ train.
"""
import time

import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import SEED
from .data import data_version, feature_columns, load_gold, split_xy
from .metrics import evaluate
from .results import write_result
from .split import apply_split, load_manifest


def split_feature_types(df, features: list[str]) -> tuple[list[str], list[str]]:
    """Tách đặc trưng số và đặc trưng chữ (vd admission_type_group)."""
    num = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in features if c not in num]
    return num, cat


def make_preprocessor(numeric: list[str], categorical: list[str], scale: bool = True) -> ColumnTransformer:
    """Số: điền median (+ chuẩn hóa); chữ: điền giá trị phổ biến + one-hot. Chỉ học khi fit trên train."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    parts = [("num", Pipeline(num_steps), numeric)]
    if categorical:
        parts.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    return ColumnTransformer(parts, remainder="drop")


def baseline_pipeline(df, features: list[str]) -> Pipeline:
    """Mô hình cơ sở chung S0: điền median (học từ train) + chuẩn hóa + one-hot + Logistic Regression."""
    num, cat = split_feature_types(df, features)
    return Pipeline([("prep", make_preprocessor(num, cat)),
                     ("model", LogisticRegression(max_iter=2000, random_state=SEED))])


def run_case(*, case_id: str, member: str, strategy: str, model_name: str, pipeline, gold_path: str,
             manifest_path: str, results_path: str, params: dict | None = None, features: list[str] | None = None,
             note: str = "", df=None, manifest: dict | None = None) -> dict:
    df = load_gold(gold_path) if df is None else df
    manifest = load_manifest(manifest_path) if manifest is None else manifest
    parts = apply_split(df, manifest, parts=("train", "val"))   # không lấy test
    feats = features or feature_columns(df)
    X_tr, y_tr = split_xy(parts["train"], feats)
    X_va, y_va = split_xy(parts["val"], feats)
    model = clone(pipeline)
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    p_va = model.predict_proba(X_va)[:, 1]
    seconds = time.perf_counter() - t0
    m = evaluate(y_va, p_va)
    row = {"case_id": case_id, "member": member, "strategy": strategy, "model": model_name,
           "params": params or {}, "data_version": manifest.get("data_version", data_version(gold_path)),
           "split_seed": manifest.get("seed"), **m, "seconds": seconds, "note": note}
    write_result(results_path, row)
    return row


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Chạy mô hình cơ sở S0 (median + LR) để kiểm tra khung chung")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True, help="vd experiments/member_5/results.csv")
    ap.add_argument("--member", default="common")
    a = ap.parse_args(argv)
    df = load_gold(a.gold)
    feats = feature_columns(df)
    row = run_case(case_id="S0_lr_median_baseline", member=a.member, strategy="S0", model_name="LogisticRegression",
                   pipeline=baseline_pipeline(df, feats), gold_path=a.gold, manifest_path=a.split, results_path=a.out,
                   params={"imputer": "median", "scaler": "standard", "max_iter": 2000}, df=df,
                   note="baseline chung, đánh giá trên validation")
    print({k: row[k] for k in ("case_id", "roc_auc", "pr_auc", "f1", "brier", "seconds")})


if __name__ == "__main__":
    main()

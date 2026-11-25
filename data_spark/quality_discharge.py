"""Kiểm tra chất lượng + chống rò rỉ cho Gold P2/P3 (mốc dự đoán = dischtime).

Không thay thế data_spark/quality.py (của Chiến, chuyên cho P1 với lineage từng event). Ở đây kiểm tra
theo cùng nguyên tắc với bằng chứng gọn hơn: thời điểm event MUỘN NHẤT đã dùng cho mỗi lượt nhập viện.

Một bảng PASSED khi mọi số đếm bằng 0:
    khóa trùng/NULL, nhãn NULL/khác 0-1, nhãn lệch bảng nhãn của Hà, dòng không có trong bảng nhãn,
    prediction_time ≠ dischtime của cohort, admittime lệch, cột bị cấm, thiếu cột hợp đồng,
    event có charttime > dischtime, storetime > dischtime, LOS/ICU-days sai, _last nằm ngoài [min, max],
    cờ _missing mâu thuẫn với giá trị.
"""
from __future__ import annotations

import json

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from data_spark.common.io import REPO_ROOT

LABELS = {"p2": "label_readmit30", "p3": "label_mort90"}


def contract(task: str) -> dict:
    return json.loads((REPO_ROOT / f"contracts/gold_schema_{task}.json").read_text(encoding="utf-8"))


def _sum(cond):
    return F.sum(F.when(cond, 1).otherwise(0))


def check_discharge_gold(gold: DataFrame, *, task: str, cohort: DataFrame, labels: DataFrame | None = None,
                         evidence: DataFrame | None = None, strict_store: bool = True) -> dict:
    schema = contract(task)
    label = LABELS[task]
    from data_spark.features import schema_columns
    expected = {n for n, _ in schema_columns(schema)}
    counts = {
        "missing_contract_columns": len(expected - set(gold.columns)),
        "unexpected_columns": len(set(gold.columns) - expected),
        "forbidden_columns": len(set(schema["forbidden_columns"]) & set(gold.columns)),
    }
    if counts["missing_contract_columns"]:
        return {"status": "FAILED", "counts": counts,
                "missing": sorted(expected - set(gold.columns))}

    labs = schema["feature_columns"]["labs"]["items"]
    ctx = cohort.select("hadm_id", F.col("admittime").alias("c_admit"), F.col("dischtime").alias("c_disch"))
    g = gold.join(ctx, "hadm_id", "left")
    if labels is not None:
        g = g.join(labels.select("hadm_id", F.col(label).alias("l_label"), F.lit(1).alias("in_labels")),
                   "hadm_id", "left")
    if evidence is not None:
        g = g.join(evidence, "hadm_id", "left")
    p, a = F.col("prediction_time"), F.col("admittime")
    conds = {
        "null_key_rows": F.col("hadm_id").isNull() | F.col("subject_id").isNull(),
        "null_label_rows": F.col(label).isNull(),
        "invalid_label_rows": ~F.col(label).isin(0, 1),
        "not_in_cohort_rows": F.col("c_disch").isNull(),
        "prediction_not_dischtime_rows": ~p.eqNullSafe(F.col("c_disch")),
        "admittime_mismatch_rows": ~a.eqNullSafe(F.col("c_admit")),
        "prediction_before_admission_rows": p <= a,
        "los_mismatch_rows": F.abs(F.col("los_days") - (F.unix_timestamp(p) - F.unix_timestamp(a)) / 86400.0) > 1e-3,
        "icu_days_exceed_los_rows": F.col("icu_days") > F.col("los_days") + 1e-3,
        "last_outside_min_max_rows": F.lit(False),
        "missing_flag_inconsistent_rows": F.lit(False),
    }
    for lab in labs:
        last, lo, hi, first, miss = (F.col(f"{lab}{s}") for s in ("_last", "_min", "_max", "_first", "_missing"))
        conds["last_outside_min_max_rows"] = conds["last_outside_min_max_rows"] | (
            last.isNotNull() & ((last < lo - 1e-9) | (last > hi + 1e-9)))
        conds["missing_flag_inconsistent_rows"] = conds["missing_flag_inconsistent_rows"] | (
            (miss == 1) & first.isNotNull()) | ((miss == 0) & first.isNull())
    if labels is not None:
        conds["not_in_labels_rows"] = F.col("in_labels").isNull()
        conds["label_mismatch_rows"] = ~F.col(label).eqNullSafe(F.col("l_label"))
    if evidence is not None:
        conds["future_event_rows"] = F.col("max_charttime") > p
        if strict_store:
            conds["late_storetime_rows"] = F.col("max_storetime") > p
    row = g.agg(F.count(F.lit(1)).alias("rows"), F.countDistinct("hadm_id").alias("distinct_hadm"),
                *[_sum(c).alias(k) for k, c in conds.items()]).first()
    counts.update({k: int(row[k] or 0) for k in conds})
    counts["duplicate_hadm_rows"] = int(row["rows"]) - int(row["distinct_hadm"])
    status = "PASSED" if row["rows"] and not any(counts.values()) else "FAILED"
    return {"status": status, "rows": int(row["rows"]), "counts": counts,
            "evidence_checked": evidence is not None, "labels_checked": labels is not None}

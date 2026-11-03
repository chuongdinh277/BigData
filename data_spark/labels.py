"""Tạo nhãn P1, P2 và P3 theo từng lượt nhập viện.

Bảng đầu ra chỉ có mã bệnh nhân, mã nhập viện, thời điểm dự đoán và nhãn.
Báo cáo chỉ ghi số liệu tổng hợp.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from data_spark.common.io import REPO_ROOT, load_yaml, read_table
from data_spark.common.session import get_spark
from data_spark.cohort import build_cohort

P2_EXCLUDED_DISCHARGE_LOCATIONS = ("HOSPICE", "AGAINST ADVICE", "ACUTE HOSPITAL")
PLANNED_ADMISSION_TYPES = ("ELECTIVE", "SURGICAL SAME DAY ADMISSION")
DEFAULT_PROCESSED_PATH = "s3a://heartguard-mimic-group11/processed"


def _require_columns(df: DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{name} thiếu cột bắt buộc: {sorted(missing)}")


def _assert_unique(df: DataFrame, key: str, name: str) -> None:
    _require_columns(df, {key}, name)
    row = df.agg(
        F.count("*").alias("rows"),
        F.count(key).alias("non_null_keys"),
        F.countDistinct(key).alias("distinct_keys"),
    ).first()
    if row["rows"] != row["non_null_keys"]:
        raise ValueError(f"{name}.{key} có giá trị NULL")
    if row["non_null_keys"] != row["distinct_keys"]:
        raise ValueError(f"{name}.{key} không duy nhất")


def _validate_expire_flags(cohort: DataFrame) -> None:
    invalid = cohort.where(
        F.col("hospital_expire_flag").isNull()
        | ~F.col("hospital_expire_flag").isin(0, 1)
    )
    if invalid.limit(1).count():
        raise ValueError("hospital_expire_flag phải chỉ chứa 0 hoặc 1 trong cohort")


def _summary_and_counts(
    df: DataFrame,
    label: str,
    included,
    counters: dict[str, object] | None = None,
) -> tuple[dict, dict[str, int]]:
    counters = counters or {}
    expressions = [
        F.sum(F.when(included, 1).otherwise(0)).alias("admissions"),
        F.countDistinct(F.when(included, F.col("subject_id"))).alias("patients"),
        F.sum(F.when(included, F.col(label)).otherwise(0)).alias("positives"),
    ]
    expressions.extend(
        F.sum(F.when(condition, 1).otherwise(0)).alias(name)
        for name, condition in counters.items()
    )
    row = df.agg(*expressions).first()
    admissions = int(row["admissions"] or 0)
    positives = int(row["positives"] or 0)
    summary = {
        "admissions": admissions,
        "patients": int(row["patients"] or 0),
        "positives": positives,
        "positive_rate": round(positives / admissions, 4) if admissions else None,
    }
    return summary, {name: int(row[name] or 0) for name in counters}


def _label_columns(df: DataFrame, label: str) -> DataFrame:
    return df.select(
        "hadm_id",
        "subject_id",
        F.col("prediction_time").cast("timestamp").alias("prediction_time"),
        F.col(label).cast("int").alias(label),
    )


def _build_p1(cohort: DataFrame, prediction_hours: int) -> tuple[DataFrame, dict]:
    _require_columns(
        cohort,
        {
            "hadm_id", "subject_id", "admittime", "dischtime", "deathtime",
            "hospital_expire_flag", "discharge_location",
        },
        "cohort",
    )
    base = cohort.withColumn(
        "prediction_time",
        F.col("admittime") + F.expr(f"INTERVAL {prediction_hours} HOURS"),
    )
    valid_time = F.coalesce(
        F.col("admittime").isNotNull()
        & F.col("dischtime").isNotNull()
        & (F.col("dischtime") > F.col("admittime")),
        F.lit(False),
    )
    still_admitted = F.coalesce(F.col("dischtime") > F.col("prediction_time"), F.lit(False))
    alive_at_prediction = (
        F.col("deathtime").isNull() | (F.col("deathtime") > F.col("prediction_time"))
    )
    eligible_condition = valid_time & still_admitted & alive_at_prediction
    base = base.withColumn("label_mortality", F.col("hospital_expire_flag").cast("int"))
    summary, counts = _summary_and_counts(
        base,
        "label_mortality",
        eligible_condition,
        {
            "invalid_admission_or_discharge_time": ~valid_time,
            "discharged_by_prediction_time": valid_time & ~still_admitted,
            "died_by_prediction_time": valid_time & still_admitted & ~alive_at_prediction,
            "label_consistency_mismatch": eligible_condition
            & (F.col("label_mortality") == 1)
            & F.col("deathtime").isNull()
            & (
                F.upper(F.trim(F.coalesce(F.col("discharge_location"), F.lit(""))))
                != "DIED"
            ),
        },
    )
    eligible = base.where(eligible_condition)
    stats = {
        "summary": summary,
        "excluded": {
            key: value
            for key, value in counts.items()
            if key != "label_consistency_mismatch"
        },
        "label_consistency_mismatch": counts["label_consistency_mismatch"],
    }
    return _label_columns(eligible, "label_mortality"), stats


def _attach_death_date(
    df: DataFrame, patients: DataFrame, death_date_column: str = "dod"
) -> DataFrame:
    _require_columns(patients, {"subject_id", death_date_column}, "patients")
    return df.join(
        patients.select(
            "subject_id",
            F.to_date(F.col(death_date_column)).alias("_death_date"),
        ),
        on="subject_id",
        how="left",
    )


def _alive_discharges(cohort: DataFrame) -> tuple[DataFrame, dict[str, int]]:
    _require_columns(
        cohort,
        {
            "hadm_id", "subject_id", "dischtime", "hospital_expire_flag",
            "discharge_location",
        },
        "cohort",
    )
    location = F.upper(F.trim(F.coalesce(F.col("discharge_location"), F.lit(""))))
    valid_time = F.col("dischtime").isNotNull()
    alive_condition = valid_time & (F.col("hospital_expire_flag") == 0)
    row = cohort.agg(
        F.sum(F.when(~valid_time, 1).otherwise(0)).alias("missing_discharge_time"),
        F.sum(F.when(valid_time & (F.col("hospital_expire_flag") == 1), 1).otherwise(0))
        .alias("died_in_hospital"),
        F.sum(
            F.when(
                alive_condition & location.isin(*P2_EXCLUDED_DISCHARGE_LOCATIONS), 1
            ).otherwise(0)
        ).alias("transfer_ama_or_hospice"),
        F.sum(F.when(alive_condition & (location == "HOSPICE"), 1).otherwise(0))
        .alias("hospice"),
    ).first()
    counts = {
        name: int(row[name] or 0)
        for name in (
            "missing_discharge_time",
            "died_in_hospital",
            "transfer_ama_or_hospice",
            "hospice",
        )
    }
    return cohort.where(alive_condition), counts


def _build_p2(
    alive: DataFrame,
    admissions: DataFrame,
    patients: DataFrame,
    window_days: int,
    excluded: dict[str, int],
) -> tuple[DataFrame, dict]:
    _require_columns(
        admissions,
        {"subject_id", "hadm_id", "admittime", "admission_type"},
        "admissions",
    )
    non_transfer = alive.where(
        ~F.upper(F.trim(F.coalesce(F.col("discharge_location"), F.lit("")))).isin(
            *P2_EXCLUDED_DISCHARGE_LOCATIONS
        )
    )
    index = non_transfer.select(
        F.col("hadm_id").alias("_index_hadm_id"),
        F.col("subject_id").alias("_index_subject_id"),
        F.col("dischtime").cast("timestamp").alias("_dischtime"),
    )
    next_admissions = admissions.select(
        F.col("subject_id").alias("_next_subject_id"),
        F.col("hadm_id").alias("_next_hadm_id"),
        F.col("admittime").cast("timestamp").alias("_next_admittime"),
        F.upper(F.trim(F.col("admission_type"))).alias("_next_admission_type"),
    )
    # Không rõ loại nhập viện thì không tính là tái nhập viện ngoài kế hoạch.
    readmission_keys = (
        index.join(
            next_admissions,
            (F.col("_index_subject_id") == F.col("_next_subject_id"))
            & (F.col("_index_hadm_id") != F.col("_next_hadm_id"))
            & (F.col("_next_admittime") > F.col("_dischtime"))
            & (
                F.col("_next_admittime")
                <= F.col("_dischtime") + F.expr(f"INTERVAL {window_days} DAYS")
            )
            & (~F.col("_next_admission_type").isin(*PLANNED_ADMISSION_TYPES)),
            how="inner",
        )
        .select(F.col("_index_hadm_id").alias("hadm_id"))
        .distinct()
        .withColumn("_has_readmission", F.lit(1))
    )
    with_outcome = (
        _attach_death_date(non_transfer, patients)
        .join(readmission_keys, "hadm_id", "left")
        .withColumn(
            "label_readmit30",
            F.coalesce(F.col("_has_readmission"), F.lit(0)).cast("int"),
        )
        .withColumn(
            "_days_to_death",
            F.datediff(F.col("_death_date"), F.to_date(F.col("dischtime"))),
        )
    )
    death_without_readmission = F.coalesce(
        (F.col("label_readmit30") == 0)
        & F.col("_days_to_death").between(1, window_days),
        F.lit(False),
    )
    summary, counts = _summary_and_counts(
        with_outcome,
        "label_readmit30",
        ~death_without_readmission,
        {"death_within_window_without_readmission": death_without_readmission},
    )
    eligible = with_outcome.where(~death_without_readmission).withColumn(
        "prediction_time", F.col("dischtime")
    )
    stats = {"summary": summary, "excluded": {
        key: value
        for key, value in excluded.items()
        if key != "hospice"
    } | counts}
    return _label_columns(eligible, "label_readmit30"), stats


def _build_p3(
    alive: DataFrame,
    patients: DataFrame,
    window_days: int,
    excluded: dict[str, int],
) -> tuple[DataFrame, dict]:
    non_hospice = alive.where(
        F.upper(F.trim(F.coalesce(F.col("discharge_location"), F.lit("")))) != "HOSPICE"
    )
    with_outcome = _attach_death_date(non_hospice, patients).withColumn(
        "_days_to_death",
        F.datediff(F.col("_death_date"), F.to_date(F.col("dischtime"))),
    )
    eligible = with_outcome.withColumn(
        "label_mort90",
        F.when(F.col("_days_to_death").between(1, window_days), 1)
        .otherwise(0)
        .cast("int"),
    ).withColumn("prediction_time", F.col("dischtime"))
    stats = {
        "summary": {
            **_summary_and_counts(
                eligible, "label_mort90", F.lit(True)
            )[0]
        },
        "excluded": {
            key: value
            for key, value in excluded.items()
            if key in {"missing_discharge_time", "died_in_hospital", "hospice"}
        },
    }
    return _label_columns(eligible, "label_mort90"), stats


def build_labels(
    cohort: DataFrame,
    admissions: DataFrame,
    patients: DataFrame,
    *,
    prediction_hours: int = 24,
    p2_window_days: int = 30,
    p3_window_days: int = 90,
) -> tuple[dict[str, DataFrame], dict]:
    """Tạo các bảng nhãn và thống kê số lượt bị loại theo từng lý do."""
    for name, value in (
        ("prediction_hours", prediction_hours),
        ("p2_window_days", p2_window_days),
        ("p3_window_days", p3_window_days),
    ):
        if value <= 0:
            raise ValueError(f"{name} phải là số nguyên dương")
    _assert_unique(cohort, "hadm_id", "cohort")
    _assert_unique(admissions, "hadm_id", "admissions")
    _assert_unique(patients, "subject_id", "patients")
    _validate_expire_flags(cohort)

    alive, discharge_exclusions = _alive_discharges(cohort)
    p1, p1_stats = _build_p1(cohort, prediction_hours)
    p2, p2_stats = _build_p2(
        alive, admissions, patients, p2_window_days, discharge_exclusions
    )
    p3, p3_stats = _build_p3(
        alive, patients, p3_window_days, discharge_exclusions
    )
    return {"p1": p1, "p2": p2, "p3": p3}, {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "P1": p1_stats,
        "P2": p2_stats,
        "P3": p3_stats,
    }


def load_or_build_cohort(
    spark: SparkSession,
    admissions: DataFrame,
    patients: DataFrame,
    *,
    root: str,
    fmt: str,
    hf_codes: dict,
    cohort_path: str | None = None,
) -> DataFrame:
    """Đọc cohort có sẵn nếu được chỉ định, nếu không thì dựng trực tiếp."""
    if cohort_path:
        return spark.read.parquet(cohort_path)

    diagnoses = read_table(spark, root, "diagnoses_icd", fmt)
    cohort, _ = build_cohort(admissions, patients, diagnoses, hf_codes)
    return cohort


def _display_count(value: int, threshold: int) -> str:
    return f"<{threshold}" if 0 < value < threshold else f"{value:,}"


def write_report(report: dict, path: str | Path, threshold: int = 10) -> None:
    """Ghi báo cáo tổng hợp và ẩn các số đếm nhỏ."""
    lines = [
        "# Kết quả chạy thử nhãn P1/P2/P3",
        "",
        f"Thời gian chạy: {report['generated_at']}",
        "",
        f"**Dữ liệu dùng để thử:** {report.get('data_source', 'chưa ghi nhận')}.",
        "",
        "> Báo cáo chỉ ghi số liệu tổng hợp. Các số đếm nhỏ hơn ngưỡng quy định sẽ được ẩn.",
        "",
        "| Nhãn | Số lượt nhập viện | Số bệnh nhân | Ca dương | Tỷ lệ dương |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in ("P1", "P2", "P3"):
        summary = report[task]["summary"]
        counts = [
            _display_count(summary[key], threshold)
            for key in ("admissions", "patients", "positives")
        ]
        rate = "N/A" if summary["positive_rate"] is None else f"{summary['positive_rate']:.4f}"
        lines.append(f"| {task} | {counts[0]} | {counts[1]} | {counts[2]} | {rate} |")

    lines += [
        "",
        "## Số lượt nhập viện bị loại",
        "",
        "| Nhãn | Lý do | Số lượt |",
        "|---|---|---:|",
    ]
    for task in ("P1", "P2", "P3"):
        for reason, count in report[task]["excluded"].items():
            lines.append(f"| {task} | `{reason}` | {_display_count(count, threshold)} |")
    lines += [
        "",
        "Ở P1, mình cũng đối chiếu `hospital_expire_flag` với `deathtime` và "
        "`discharge_location`; có "
        f"{_display_count(report['P1']['label_consistency_mismatch'], threshold)} ca không khớp.",
        "",
    ]
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _resolve_input_root(root: str | None, fmt: str, paths: dict) -> str:
    return root or paths[fmt]


def main() -> None:
    cfg = load_yaml("configs/data.yaml")
    processed = cfg["paths"].get("processed", DEFAULT_PROCESSED_PATH)
    prediction = cfg["prediction"]
    hf_codes = load_yaml("configs/hf_codes.yaml")["heart_failure"]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=None,
        help="Gốc dữ liệu; mặc định lấy từ configs/data.yaml theo --fmt",
    )
    parser.add_argument("--fmt", choices=["silver", "bronze"], default="silver")
    parser.add_argument(
        "--cohort",
        default=None,
        help="Dùng cohort Parquet có sẵn; mặc định dựng trực tiếp từ bảng nguồn, không lưu cohort",
    )
    parser.add_argument("--out", default=f"{processed.rstrip('/')}/labels",
                        help="Thư mục gốc ghi Parquet p1/, p2/, p3/")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Ghi rõ trong báo cáo rằng nguồn dữ liệu là dữ liệu giả",
    )
    parser.add_argument(
        "--report",
        default=str(REPO_ROOT / "docs/stage2/labels_result.md"),
        help="Báo cáo Markdown chỉ gồm thống kê tổng hợp",
    )
    args = parser.parse_args()
    input_root = _resolve_input_root(args.root, args.fmt, cfg["paths"])

    spark: SparkSession = get_spark("hg-labels")
    try:
        admissions = read_table(spark, input_root, "admissions", args.fmt)
        patients = read_table(spark, input_root, "patients", args.fmt)
        cohort = load_or_build_cohort(
            spark,
            admissions,
            patients,
            root=input_root,
            fmt=args.fmt,
            hf_codes=hf_codes,
            cohort_path=args.cohort,
        )
        label_tables, report = build_labels(
            cohort,
            admissions,
            patients,
            prediction_hours=int(prediction["p1_index_hours"]),
            p2_window_days=int(prediction["p2_window_days"]),
            p3_window_days=int(prediction["p3_window_days"]),
        )
        if args.synthetic:
            report["data_source"] = (
                "Dữ liệu giả (synthetic) từ tools/make_synthetic_mimic.py; "
                "không phải dữ liệu MIMIC-IV thật."
            )
        for task, df in label_tables.items():
            destination = f"{args.out.rstrip('/')}/{task}"
            df.repartition(4).write.mode("overwrite").parquet(destination)
            print(f"[OK] Đã ghi nhãn {task.upper()} → {destination}")
        write_report(
            report,
            args.report,
            threshold=int(cfg["privacy"]["small_cell_threshold"]),
        )
        print(f"[OK] Đã ghi báo cáo tổng hợp → {args.report}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

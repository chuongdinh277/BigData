"""GĐ2: kiểm tra bảng Spark và provenance của đặc trưng P1.

Không collect dòng bệnh nhân. Context/lineage chỉ dùng kiểm tra, không ghép vào
Gold xuất ra. Xem docs/stage2/quality_usage.md cho hợp đồng và lệnh chạy.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import NumericType, StringType

from data_spark.common.io import REPO_ROOT, load_yaml
from data_spark.common.session import get_spark


def contract() -> dict:
    return json.loads((REPO_ROOT / "contracts/gold_schema_p1.json").read_text())


def feature_columns() -> set[str]:
    fc = contract()["feature_columns"]
    names = {c["name"] for group in ("demographic_admission", "history", "vitals_icu") for c in fc[group]}
    names.update(f"{lab}{suffix}" for lab in fc["labs"]["items"] for suffix in fc["labs"]["suffixes"])
    return names | {c["name"] for c in fc["labs"]["extra"]}


def value_ranges() -> dict[str, list[float]]:
    cfg = load_yaml("configs/feature_items.yaml")
    ranges = {f"{lab}_{suffix}": spec["valid_range"] for lab, spec in cfg["labs"].items()
              for suffix in ("first", "min", "max", "last")}
    ranges.update({col: spec["valid_range"] for spec in cfg["vitals"].values() for col in spec["feature_columns"]})
    return ranges


def _missing(df: DataFrame, name: str):
    c = F.col(name)
    typ = df.schema[name].dataType
    if isinstance(typ, NumericType):
        return c.isNull() | F.isnan(c.cast("double"))
    if isinstance(typ, StringType):
        return c.isNull() | (F.trim(c) == "")
    return c.isNull()


def _invalid(df: DataFrame, name: str, limits):
    number = F.col(name).cast("double")
    return ~_missing(df, name) & (number.isNull() | F.isnan(number) | ~number.between(*limits))


def _counts(df: DataFrame, conditions: dict) -> dict[str, int]:
    if not conditions:
        return {}
    row = df.agg(*[F.sum(F.when(cond, 1).otherwise(0)).alias(name) for name, cond in conditions.items()]).first()
    return {name: int(row[name] or 0) for name in conditions}


def null_invalid_values(df: DataFrame) -> DataFrame:
    """Gán NULL ngoài khoảng của feature_spec; gọi sau khi quy đổi đơn vị."""
    ranges = value_ranges()
    return df.select(*[F.when(_invalid(df, c, ranges[c]), F.lit(None)).otherwise(F.col(c)).alias(c)
                       if c in ranges else F.col(c) for c in df.columns])


def profile_table(df: DataFrame, keys=("hadm_id",)) -> dict:
    """Missing tổng thể/theo nhóm năm, trùng khóa, khoảng hợp lệ; không sửa input."""
    ranges = value_ranges()
    expressions = [F.count("*").alias("rows")]
    for i, name in enumerate(df.columns):
        expressions.append(F.sum(F.when(_missing(df, name), 1).otherwise(0)).alias(f"m{i}"))
        if name in ranges:
            expressions.append(F.sum(F.when(_invalid(df, name, ranges[name]), 1).otherwise(0)).alias(f"r{i}"))
    row = df.agg(*expressions).first()
    n = int(row["rows"])
    columns = [{"column": c, "missing_count": int(row[f"m{i}"] or 0),
                "missing_pct": round(100 * int(row[f"m{i}"] or 0) / n, 4) if n else None,
                "invalid_count": int(row[f"r{i}"] or 0) if c in ranges else None,
                "valid_range": ranges.get(c)} for i, c in enumerate(df.columns)]
    missing_keys = sorted(set(keys) - set(df.columns))
    duplicates = {"status": "NOT_CHECKED", "missing_columns": missing_keys}
    if not missing_keys:
        counts = _counts(df, {"null_key_rows": F.expr(" OR ".join(f"`{k}` IS NULL" for k in keys))})
        grouped = df.groupBy(*keys).count().where(F.col("count") > 1)
        r = grouped.agg(F.count("*").alias("groups"), F.sum("count").alias("rows"),
                        F.sum(F.col("count") - 1).alias("excess")).first()
        duplicates = {"duplicate_keys": int(r["groups"]), "duplicate_rows": int(r["rows"] or 0),
                      "excess_rows": int(r["excess"] or 0), **counts}
        duplicates["status"] = "FAILED" if duplicates["duplicate_rows"] or counts["null_key_rows"] else "PASSED"
    by_year = []
    if "anchor_year_group" in df.columns:
        exprs = [F.count("*").alias("rows")]
        exprs += [F.sum(F.when(_missing(df, c), 1).otherwise(0)).alias(f"m{i}") for i, c in enumerate(df.columns)]
        for r in df.groupBy("anchor_year_group").agg(*exprs).orderBy("anchor_year_group").collect():
            by_year.append({"anchor_year_group": r["anchor_year_group"], "rows": r["rows"],
                            "columns": [{"column": c, "missing_count": int(r[f"m{i}"] or 0),
                                         "missing_pct": round(100 * int(r[f"m{i}"] or 0) / r["rows"], 4)}
                                        for i, c in enumerate(df.columns)]})
    failed = not n or any(c["invalid_count"] for c in columns) or duplicates["status"] != "PASSED"
    return {"status": "FAILED" if failed else "PASSED", "rows": n, "columns": columns,
            "duplicates": duplicates, "missing_by_year": by_year}


def check_prediction_times(features: DataFrame, context: DataFrame | None = None) -> dict:
    """Context admissions/cohort chứa dischtime; cột này không đi vào Gold P1."""
    required = {"hadm_id", "admittime", "prediction_time"}
    if required - set(features.columns):
        return {"status": "NOT_CHECKED", "reason": "Thiếu hadm_id/admittime/prediction_time"}
    if context is None or {"hadm_id", "admittime", "dischtime"} - set(context.columns):
        return {"status": "NOT_CHECKED", "reason": "Cần context độc lập có hadm_id/admittime/dischtime"}
    if context.where(F.col("hadm_id").isNull()).limit(1).count() or context.groupBy("hadm_id").count().where("count > 1").limit(1).count():
        return {"status": "FAILED", "reason": "Context có khóa NULL/trùng; không thể ghép an toàn"}
    ctx = context.select("hadm_id", F.col("admittime").cast("timestamp").alias("context_admittime"),
                         F.col("dischtime").cast("timestamp").alias("context_dischtime"),
                         *([F.col("subject_id").alias("context_subject_id")] if "subject_id" in context.columns else []))
    joined = features.join(ctx, "hadm_id", "left")
    a, p, d = F.col("admittime").cast("timestamp"), F.col("prediction_time").cast("timestamp"), F.col("context_dischtime")
    conditions = {"missing_time_rows": a.isNull() | p.isNull() | d.isNull() | F.col("context_admittime").isNull(),
                  "invalid_order_rows": ~((a < p) & (p < d)),
                  "wrong_prediction_offset_rows": p != a + F.expr("INTERVAL 24 HOURS"),
                  "admission_mismatch_rows": a != F.col("context_admittime")}
    if "subject_id" in features.columns and "subject_id" in context.columns:
        conditions["patient_mismatch_rows"] = ~F.col("subject_id").eqNullSafe(F.col("context_subject_id"))
    counts = _counts(joined, conditions)
    return {"status": "FAILED" if any(counts.values()) else "PASSED", **counts}


def _temporal_features(features: DataFrame) -> list[str]:
    fc = contract()["feature_columns"]
    static = {c["name"] for c in fc["demographic_admission"]} - {"in_icu_24h"}
    static |= {"first_admission", "vitals_missing"}
    return sorted((set(features.columns) & feature_columns()) - static - {c for c in features.columns if c.endswith("_missing")})


def check_leakage(features: DataFrame, lineage: DataFrame | None = None, model_columns=None) -> dict:
    """Kiểm tra từng event đóng góp và độ phủ bằng chứng từng feature không NULL.

    Lineage là bảng dài, phải được feature pipeline xuất TỪ CÁC EVENT ĐÃ DÙNG,
    không tạo từ timestamp giả định sau khi tổng hợp. Không có lineage => chưa đạt.
    """
    spec = contract()
    allowed_features = feature_columns()
    metadata = {c["name"] for group in ("id_columns", "label_columns", "subgroup_only_columns") for c in spec[group]}
    forbidden = set(spec["forbidden_columns"]) | {"last_careunit", "diagnoses_icd", "procedures_icd", "drgcodes"}
    forbidden_found = sorted(c for c in features.columns if c.lower().split(".")[-1] in forbidden)
    unexpected = sorted(set(features.columns) - allowed_features - metadata)
    invalid_model = sorted(set(model_columns or []) - allowed_features)
    result = {"forbidden_columns": forbidden_found, "unexpected_columns": unexpected, "invalid_model_columns": invalid_model}
    if forbidden_found or unexpected or invalid_model:
        return {"status": "FAILED", **result}
    needed = {"subject_id", "hadm_id", "admittime", "prediction_time"}
    if needed - set(features.columns):
        return {"status": "NOT_CHECKED", "reason": "Thiếu khóa hoặc mốc thời gian của bảng P1", **result}
    missing_ids = _counts(features, {"missing_identity_rows": F.col("subject_id").isNull() | F.col("hadm_id").isNull()})
    if missing_ids["missing_identity_rows"]:
        return {"status": "FAILED", **result, **missing_ids}
    temporal = _temporal_features(features)
    if not temporal:
        return {"status": "PASSED", "scope": "Chỉ có đặc trưng tĩnh", **result}
    cols = {"subject_id", "hadm_id", "feature_name", "source_table", "source_subject_id", "source_hadm_id",
            "source_charttime", "source_storetime", "source_dischtime"}
    if lineage is None or cols - set(lineage.columns):
        return {"status": "NOT_CHECKED", "reason": "Thiếu lineage hoặc thiếu cột provenance",
                "required_lineage_columns": sorted(cols), **result}
    base = features.select("subject_id", "hadm_id", F.col("admittime").cast("timestamp").alias("admittime"),
                           F.col("prediction_time").cast("timestamp").alias("prediction_time"))
    joined = lineage.join(base, ["subject_id", "hadm_id"], "left")
    source = F.regexp_replace(F.lower(F.col("source_table")), r"^(hosp|icu)[./]", "")
    name = F.col("feature_name")
    cfg = load_yaml("configs/feature_items.yaml")
    labs = [c for c in temporal if any(c.startswith(lab + "_") for lab in cfg["labs"])] + ["n_lab_tests_24h"]
    vitals = [c for s in cfg["vitals"].values() for c in s["feature_columns"]]
    is_lab, is_vital, is_icu = name.isin(labs), name.isin(vitals), name == "in_icu_24h"
    is_omr = name.startswith("omr_")
    is_history = name.isin([c["name"] for c in spec["feature_columns"]["history"] if not c["name"].startswith("omr_")])
    ct, st, dt = (F.col(c).cast("timestamp") for c in ("source_charttime", "source_storetime", "source_dischtime"))
    a, p = F.col("admittime"), F.col("prediction_time")
    history_source = ((name.startswith("hx_") | (name == "n_prior_hf_adm")) & (source == "diagnoses_icd")) | (
        name.isin("n_prior_adm_365d", "days_since_last_discharge") & (source == "admissions"))
    valid_source = ((is_lab & (source == "labevents")) | (is_vital & (source == "chartevents"))
                    | (is_icu & (source == "icustays")) | (is_omr & (source == "omr"))
                    | (is_history & history_source))
    same_adm = F.col("source_hadm_id") == F.col("hadm_id")
    conditions = {
        "orphan_lineage_rows": a.isNull() | p.isNull(),
        "unknown_feature_rows": name.isNull() | ~name.isin(temporal),
        "invalid_source_rows": ~F.coalesce(valid_source, F.lit(False)),
        "wrong_patient_rows": ~F.col("source_subject_id").eqNullSafe(F.col("subject_id")),
        "missing_timestamp_rows": ct.isNull() | ((is_lab | is_vital) & st.isNull()) | (is_history & dt.isNull()),
        "future_charttime_rows": ct > p,
        "late_storetime_rows": st > p,
        "reversed_storetime_rows": st < ct,
        "wrong_admission_rows": ((is_vital | is_icu) & ~F.coalesce(same_adm, F.lit(False)))
                                | (is_lab & F.col("source_hadm_id").isNotNull() & ~same_adm),
        "outside_window_rows": (is_lab & (ct < a - F.expr("INTERVAL 6 HOURS")))
                               | (is_vital & (ct < a))
                               | (is_omr & ((F.to_date(ct) >= F.to_date(a)) | (F.to_date(ct) < F.date_sub(F.to_date(a), 365))))
                               | ((name == "n_prior_adm_365d") & (ct < a - F.expr("INTERVAL 365 DAYS"))),
        "invalid_history_rows": is_history & (F.col("source_hadm_id").isNull() | same_adm | (dt >= a) | (ct >= a) | (ct > dt) | (st >= a)),
    }
    counts = _counts(joined, conditions)
    # Cần bằng chứng cho từng giá trị quan sát; cờ thiếu và số đếm bằng 0 không có event.
    pairs = []
    zero_ok = {"in_icu_24h", "n_lab_tests_24h", "n_prior_adm_365d", "n_prior_hf_adm"}
    for c in temporal:
        present = ~_missing(features, c)
        if c in zero_ok or c.startswith("hx_"):
            present = present & (F.col(c) != 0)
        pairs.append(F.when(present, F.lit(c)))
    expected = features.select("subject_id", "hadm_id", F.explode(F.array(*pairs)).alias("feature_name")).where("feature_name IS NOT NULL")
    evidence = lineage.select("subject_id", "hadm_id", "feature_name").distinct()
    counts["uncovered_feature_values"] = expected.join(evidence, ["subject_id", "hadm_id", "feature_name"], "left_anti").count()
    return {"status": "FAILED" if any(counts.values()) else "PASSED", **result, **counts}


def audit_event_units(events: DataFrame, group: str = "labs") -> dict:
    """Đếm đơn vị sai/thiếu, numeric-text và giá trị vô lý cho item đã chọn.

    Chỉ nhóm đơn vị của item đã chọn được collect (không dòng bệnh nhân).
    temperature được đổi F -> C trước kiểm tra khoảng.
    """
    cfg = load_yaml("configs/feature_items.yaml")
    required = {"itemid", "valueuom", "valuenum", "value"}
    if required - set(events.columns):
        raise ValueError(f"Thiếu cột event: {sorted(required - set(events.columns))}")
    definitions = cfg[group]
    mapping = [(int(item), key, s["unit_hint"], float(s["valid_range"][0]), float(s["valid_range"][1]),
                s.get("conversion") == "fahrenheit_to_celsius") for key, s in definitions.items() for item in s["itemids"]]
    meta = events.sparkSession.createDataFrame(mapping, "itemid int, feature string, expected_unit string, lo double, hi double, fahrenheit boolean")
    selected = events.join(F.broadcast(meta), "itemid", "inner")
    unit_ok = F.trim(F.coalesce(F.col("valueuom"), F.lit(""))) == F.col("expected_unit")
    num = F.col("valuenum").cast("double")
    numeric = num.isNotNull() & ~F.isnan(num)
    converted = F.when(F.col("fahrenheit"), (num - 32) * 5 / 9).otherwise(num)
    invalid = numeric & ~converted.between(F.col("lo"), F.col("hi"))
    stats = selected.groupBy("itemid", "feature", "expected_unit").agg(
        F.count("*").alias("rows"), F.sum(F.when(~unit_ok, 1).otherwise(0)).alias("wrong_unit_rows"),
        F.sum(F.when(~numeric, 1).otherwise(0)).alias("missing_numeric_rows"),
        F.sum(F.when(~numeric & (F.length(F.trim(F.col("value"))) > 0), 1).otherwise(0)).alias("numeric_text_rows"),
        F.sum(F.when(unit_ok & invalid, 1).otherwise(0)).alias("invalid_value_rows"),
        F.sum(F.when(unit_ok & numeric & ~invalid, 1).otherwise(0)).alias("accepted_rows")).collect()
    by_id = {r["itemid"]: r.asDict() for r in stats}
    items = []
    for item, name, unit, *_ in mapping:
        r = by_id.get(item, {"itemid": item, "feature": name, "expected_unit": unit, "rows": 0,
                             "wrong_unit_rows": 0, "missing_numeric_rows": 0, "numeric_text_rows": 0,
                             "invalid_value_rows": 0, "accepted_rows": 0})
        share = (r["rows"] - r["wrong_unit_rows"]) / r["rows"] if r["rows"] else None
        r["expected_unit_share"] = round(share, 6) if share is not None else None
        r["status"] = "NOT_CHECKED" if share is None else ("PASSED" if share >= cfg["selection"]["minimum_unit_share"] else "FAILED")
        items.append(r)
    return {"status": _status(r["status"] for r in items), "items": items}


def _status(statuses) -> str:
    statuses = list(statuses)
    return "FAILED" if "FAILED" in statuses else ("NOT_CHECKED" if "NOT_CHECKED" in statuses else "PASSED")


def run_quality(df: DataFrame, *, context=None, lineage=None, p1=True, keys=("hadm_id",)) -> dict:
    checks = {"profile": profile_table(df, keys)}
    if p1:
        checks["prediction_times"] = check_prediction_times(df, context)
        checks["leakage"] = check_leakage(df, lineage)
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "scope": "p1" if p1 else "table",
            "status": _status(c["status"] for c in checks.values()), "checks": checks}


def redact_counts(value, threshold=10, key=""):
    """Che ô đếm nhỏ; dùng trên báo cáo dữ liệu thật trước khi ghi local/S3."""
    if isinstance(value, dict):
        return {k: redact_counts(v, threshold, k) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_counts(v, threshold, key) for v in value]
    is_count = key in {"rows", "duplicate_keys", "uncovered_feature_values"} or key.endswith(("_rows", "_count"))
    if is_count and isinstance(value, int) and 0 < value < threshold:
        return f"<{threshold}"
    return value


def to_markdown(report: dict) -> str:
    lines = ["# Báo cáo kiểm tra chất lượng", "", f"Trạng thái: **{report['status']}** · Nguồn: **{report.get('data_source', 'unspecified')}**", "",
             "Chỉ chứa thống kê tổng hợp. PASSED chỉ áp dụng bảng/lineage đã cung cấp.", ""]
    profile = report.get("checks", {}).get("profile")
    if profile:
        lines += [f"Số dòng: {profile['rows']}", "", "| Cột | Số thiếu | % thiếu | Ngoài khoảng |", "|---|---:|---:|---:|"]
        lines += [f"| {c['column']} | {c['missing_count']} | {c['missing_pct']} | {c['invalid_count'] if c['invalid_count'] is not None else 'N/A'} |" for c in profile["columns"]]
    # JSON tổng hợp giữ đủ missing theo nhóm năm và chi tiết từng phép kiểm.
    detail = {k: v for k, v in report.items() if k != "generated_at"}
    lines += ["", "## Chi tiết kiểm tra (tổng hợp)", "", "```json", json.dumps(detail, ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)


def write_report(report: dict, out: str, spark=None) -> None:
    payloads = {"quality_report.json": json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                "quality_report.md": to_markdown(report)}
    for name, payload in payloads.items():
        if out.startswith("s3a://"):
            if spark is None:
                raise ValueError("Ghi S3 cần SparkSession với IAM Role")
            path = spark._jvm.org.apache.hadoop.fs.Path(out.rstrip("/") + "/" + name)
            fs = path.getFileSystem(spark._jsc.hadoopConfiguration())
            fs.mkdirs(path.getParent())
            stream = fs.create(path, True)
            try:
                stream.write(bytearray(payload.encode("utf-8")))
            finally:
                stream.close()
        else:
            directory = Path(out)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / name).write_text(payload, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", required=True)
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    ap.add_argument("--kind", choices=["p1", "table", "labs", "vitals"], default="p1")
    ap.add_argument("--context", help="Parquet: admissions/cohort, có dischtime")
    ap.add_argument("--lineage", help="Parquet: một dòng cho mỗi nguồn đóng góp/feature")
    ap.add_argument("--keys", nargs="+", default=["hadm_id"])
    ap.add_argument("--out", default="reports/quality")
    ap.add_argument("--synthetic", action="store_true", help="Đánh dấu dữ liệu giả, không che ô nhỏ")
    args = ap.parse_args()
    spark = get_spark("hg-quality")
    try:
        reader = spark.read
        df = reader.parquet(args.table) if args.format == "parquet" else reader.option("header", True).option("inferSchema", True).csv(args.table)
        if args.kind in {"labs", "vitals"}:
            report = {"status": "NOT_CHECKED", "scope": args.kind, "checks": {"units": audit_event_units(df, args.kind)}}
            report["status"] = report["checks"]["units"]["status"]
        else:
            report = run_quality(df, p1=args.kind == "p1", keys=args.keys,
                                 context=reader.parquet(args.context) if args.context else None,
                                 lineage=reader.parquet(args.lineage) if args.lineage else None)
        report["data_source"] = "synthetic" if args.synthetic else "real_or_unspecified"
        report["generated_at"] = datetime.now(timezone.utc).isoformat()
        if not args.synthetic:
            report = redact_counts(report, load_yaml("configs/data.yaml")["privacy"]["small_cell_threshold"])
        write_report(report, args.out, spark)
        print(f"[{report['status']}] quality report → {args.out}")
        if report["status"] != "PASSED":
            raise SystemExit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

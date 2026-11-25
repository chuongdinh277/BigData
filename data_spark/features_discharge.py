"""Feature Engineering — Gold P2 (tái nhập viện 30 ngày) và P3 (tử vong 90 ngày sau xuất viện).

Mốc dự đoán = dischtime của lần nhập viện chỉ mục. Quần thể và nhãn lấy NGUYÊN từ bảng nhãn của Hà
(`<processed>/labels/p2|p3`, D-011): module này không tự định nghĩa lại ai đủ điều kiện.

Dùng lại các hàm của data_spark.features (P1) với prediction_time = dischtime:
    - Nhân khẩu học, nhóm loại/nguồn nhập viện; icu_any = có nằm ICU trong đợt (thay in_icu_24h)
    - Tiền sử: chỉ admission đã xuất viện TRƯỚC admittime hiện tại (như P1)
    - OMR: 365 ngày trước ngày nhập viện (như P1)
    - Xét nghiệm: first/min/max trong [admittime − 6h, dischtime], _last trong [dischtime − 48h, dischtime]
    - Sinh hiệu ICU: tổng hợp cả đợt [admittime, dischtime]
Thêm nhóm "stay" (chỉ hợp lệ vì mốc là lúc xuất viện — feature_spec.md §5):
    los_days, icu_days, discharge_location_group, n_diagnoses, dx_<bệnh kèm> của CHÍNH đợt này.

Chống rò rỉ: mọi event phải có charttime ≤ dischtime và (nếu bật) storetime ≤ dischtime; không dùng
bất cứ thông tin nào của admission sau, patients.dod, deathtime. Xem data_spark/quality_discharge.py.

    python -m data_spark.features_discharge --task p2 --version v1
    python -m data_spark.features_discharge --task p3 --version v1 --sample-frac 0.01
"""
from __future__ import annotations

import argparse
import json
import time
from functools import reduce
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from data_spark import features as f1
from data_spark.common.io import REPO_ROOT, icd_prefix_condition, load_yaml, read_table
from data_spark.common.session import get_spark

TASKS = {"p2": "label_readmit30", "p3": "label_mort90"}
LAST_HOURS = 48


def schema_path(task: str) -> Path:
    return REPO_ROOT / f"contracts/gold_schema_{task}.json"


def build_base(cohort: DataFrame, labels: DataFrame, label_col: str, sample_frac=None) -> DataFrame:
    """Quần thể = đúng các hadm_id trong bảng nhãn; prediction_time = dischtime."""
    lab = labels.select("hadm_id", F.col(label_col).cast("int").alias(label_col))
    b = cohort.join(lab, "hadm_id", "inner").withColumn("prediction_time", F.col("dischtime"))
    if sample_frac:
        b = b.filter(F.pmod(F.hash("subject_id"), F.lit(10000)) < int(sample_frac * 10000))
    return b.filter(F.col("dischtime") > F.col("admittime"))


def discharge_location_group(col):
    loc = F.upper(F.trim(F.coalesce(col, F.lit(""))))
    return (F.when(loc.isin("HOME"), "HOME")
            .when(loc.contains("HOME HEALTH"), "HOME_HEALTH")
            .when(loc.contains("SKILLED NURSING") | loc.contains("REHAB") | loc.contains("CHRONIC")
                  | loc.contains("LONG TERM") | loc.contains("ASSISTED LIVING") | loc.contains("NURSING"), "FACILITY")
            .when(loc == "", "UNKNOWN")
            .otherwise("OTHER"))


def stay_features(base: DataFrame, icustays: DataFrame, dx: DataFrame, comorb: dict) -> DataFrame:
    """Thông tin của chính đợt nằm viện — chỉ dùng được vì mốc dự đoán là lúc xuất viện."""
    b = base.select("hadm_id", "admittime", "dischtime", "discharge_location")
    los = b.select(
        "hadm_id",
        ((F.unix_timestamp("dischtime") - F.unix_timestamp("admittime")) / 86400.0).alias("los_days"),
        discharge_location_group(F.col("discharge_location")).alias("discharge_location_group"),
    )
    # Thời gian ICU, cắt vào [admittime, dischtime] để không tính phần sau xuất viện
    icu = (b.join(icustays.select("hadm_id", "intime", "outtime"), "hadm_id")
           .withColumn("s", F.greatest("intime", "admittime"))
           .withColumn("e", F.least("outtime", "dischtime"))
           .filter(F.col("e") > F.col("s"))
           .groupBy("hadm_id")
           .agg((F.sum(F.unix_timestamp("e") - F.unix_timestamp("s")) / 86400.0).alias("icu_days")))
    cur = dx.join(b.select("hadm_id"), "hadm_id")
    dxf = cur.groupBy("hadm_id").agg(
        F.countDistinct("icd_version", "icd_code").alias("n_diagnoses"),
        *[F.max(icd_prefix_condition(F.col("icd_code"), F.col("icd_version"), codes).cast("int")).alias(f"dx_{n}")
          for n, codes in comorb.items()],
    )
    out = los.join(icu, "hadm_id", "left").join(dxf, "hadm_id", "left")
    for c in ["icu_days", "n_diagnoses"] + [f"dx_{n}" for n in comorb]:
        out = out.withColumn(c, F.coalesce(F.col(c), F.lit(0)))
    return out


def event_max_times(lab_win: DataFrame, vit_events: DataFrame | None) -> DataFrame:
    """Thời điểm muộn nhất của event đã dùng cho mỗi hadm_id (bằng chứng chống rò rỉ, không vào Gold)."""
    parts = [lab_win.groupBy("hadm_id").agg(F.max("charttime").alias("max_charttime"),
                                             F.max("storetime").alias("max_storetime"))]
    if vit_events is not None:
        parts.append(vit_events.groupBy("hadm_id").agg(F.max("charttime").alias("max_charttime"),
                                                       F.max("storetime").alias("max_storetime")))
    u = reduce(lambda a, b: a.unionByName(b), parts)
    return u.groupBy("hadm_id").agg(F.max("max_charttime").alias("max_charttime"),
                                    F.max("max_storetime").alias("max_storetime"))


def vitals_events(chartevents: DataFrame, base: DataFrame, vit_cfg: dict, strict_store: bool) -> DataFrame:
    ids = [i for c in vit_cfg.values() for i in c["itemids"]]
    j = (chartevents.filter(F.col("itemid").isin(ids)).select("hadm_id", "charttime", "storetime")
         .join(F.broadcast(base.select("hadm_id", "admittime", "prediction_time")), "hadm_id")
         .filter((F.col("charttime") >= F.col("admittime")) & (F.col("charttime") <= F.col("prediction_time"))))
    if strict_store:
        j = j.filter(F.col("storetime").isNull() | (F.col("storetime") <= F.col("prediction_time")))
    return j


def assemble(demo, hist, omr, stay, labs, nlab, vit, vit_cols, schema, lab_names) -> DataFrame:
    df = (demo.withColumnRenamed("in_icu_24h", "icu_any")
          .join(hist, "hadm_id", "left").join(omr, "hadm_id", "left").join(stay, "hadm_id", "left")
          .join(labs, "hadm_id", "left").join(nlab, "hadm_id", "left").join(vit, "hadm_id", "left"))
    df = df.withColumn("n_lab_tests_stay", F.coalesce(F.col("n_lab_tests_stay"), F.lit(0)))
    for n in lab_names:
        df = df.withColumn(f"{n}_missing", F.col(f"{n}_first").isNull().cast("int"))
    df = df.withColumn("vitals_missing",
                       reduce(lambda a, b: a & b, [F.col(c).isNull() for c in vit_cols]).cast("int"))
    for c in ["n_prior_adm_365d", "n_prior_hf_adm", "first_admission"] + [c for c in df.columns if c.startswith("hx_")]:
        df = df.withColumn(c, F.coalesce(F.col(c), F.lit(0)))
    spec = f1.schema_columns(schema)
    missing = [n for n, _ in spec if n not in df.columns]
    assert not missing, f"Thiếu cột so với hợp đồng {schema['task']}: {missing}"
    out = df.select(*[F.col(n).cast(f1.SPARK_TYPES[t]).alias(n) for n, t in spec])
    forbidden = set(schema["forbidden_columns"]) & set(out.columns)
    assert not forbidden, f"Cột bị cấm xuất hiện: {forbidden}"
    return out


def build(spark, *, task: str, root: str, fmt: str, cohort: DataFrame, labels: DataFrame,
          sample_frac=None, strict_store: bool = True):
    """Trả (gold, evidence, lab_stats, n_base). evidence = max charttime/storetime theo hadm_id."""
    items = load_yaml("configs/feature_items.yaml")
    hfc = load_yaml("configs/hf_codes.yaml")
    schema = json.loads(schema_path(task).read_text(encoding="utf-8"))
    label = TASKS[task]
    rt = lambda name: read_table(spark, root, name, fmt)  # noqa: E731
    base = build_base(cohort, labels, label, sample_frac).cache()
    n_base = base.count()
    dx = rt("diagnoses_icd")
    icustays = rt("icustays")
    demo = f1.demographics(base, icustays, dx)
    hist = f1.history(base, rt("admissions"), dx, hfc["heart_failure"], hfc["comorbidities"])
    omr = f1.omr_features(base, rt("omr"))
    stay = stay_features(base, icustays, dx, hfc["comorbidities"])
    lw = f1.lab_window(rt("labevents"), base, strict_store).cache()
    nlab = lw.groupBy("hadm_id").agg(F.count(F.lit(1)).alias("n_lab_tests_stay"))
    labs, lab_stats = f1.lab_features(lw, items["labs"], items["selection"].get("minimum_unit_share", 0.95),
                                      last_hours=LAST_HOURS)
    ce = rt("chartevents")
    vit, vit_cols, _ = f1.vitals_features(ce, base, items["vitals"], strict_store)
    gold = assemble(demo, hist, omr, stay, labs, nlab, vit, vit_cols, schema, list(items["labs"]))
    evidence = event_max_times(lw, vitals_events(ce, base, items["vitals"], strict_store))
    return gold, evidence, lab_stats, n_base


def write_report(path: Path, gold: DataFrame, task: str, n_base: int, check: dict, args) -> None:
    label = TASKS[task]
    agg = gold.agg(F.count(F.lit(1)).alias("n"), F.countDistinct("subject_id").alias("p"),
                   F.sum(label).alias("pos"), F.avg("los_days").alias("los"), F.avg("icu_any").alias("icu")).first()
    small = lambda v: "<10" if 0 < v < 10 else f"{v:,}"  # noqa: E731
    L = [f"# Kết quả bảng đặc trưng {task.upper()} (mốc = xuất viện)", "",
         f"- Phiên bản: `{args.version}`; mẫu: {'toàn bộ' if not args.sample_frac else f'{args.sample_frac:.0%} bệnh nhân'}",
         f"- Số ca trong bảng nhãn ∩ cohort: {small(n_base)}",
         f"- Số dòng Gold: {small(agg['n'])}; số bệnh nhân: {small(agg['p'])}",
         f"- Ca dương (`{label}`): {small(int(agg['pos'] or 0))} ({(agg['pos'] or 0) / max(agg['n'], 1):.2%})",
         f"- LOS trung bình: {agg['los'] or 0:.1f} ngày; có nằm ICU: {agg['icu'] or 0:.1%}",
         "", "## Kiểm tra chống rò rỉ (mốc = dischtime)", "",
         "| Kiểm tra | Kết quả |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in check["counts"].items()]
    L += ["", f"Trạng thái: **{check['status']}**", "", "Chỉ chứa số liệu tổng hợp, không có dữ liệu cá nhân.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> None:
    from data_spark.quality_discharge import check_discharge_gold

    cfg = load_yaml("configs/data.yaml")
    ap = argparse.ArgumentParser(description="Gold P2/P3 (mốc = dischtime)")
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    ap.add_argument("--fmt", choices=["silver", "bronze"], default="silver")
    ap.add_argument("--root", default=None)
    ap.add_argument("--cohort", default=None, help="Mặc định <processed>/cohort")
    ap.add_argument("--labels", default=None, help="Mặc định <processed>/labels/<task> (nhãn của Hà)")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--out", default=None, help="Mặc định <gold>/<task>/<version>")
    ap.add_argument("--sample-frac", type=float, default=None)
    ap.add_argument("--no-storetime-filter", action="store_true")
    ap.add_argument("--report", default=None, help="Mặc định docs/stage2/features_<task>.md")
    args = ap.parse_args(argv)

    processed = cfg["paths"].get("processed", "s3a://heartguard-mimic-group11/processed").rstrip("/")
    gold_root = cfg["paths"].get("gold", "s3a://heartguard-mimic-group11/gold").rstrip("/")
    root = args.root or cfg["paths"][args.fmt]
    out = args.out or (f"{gold_root}/{args.task}/{args.version}" if not args.sample_frac
                       else f"{gold_root}/{args.task}/_sample_{args.sample_frac}")
    report = Path(args.report or REPO_ROOT / f"docs/stage2/features_{args.task}.md")
    t0 = time.time()
    spark = get_spark(f"hg-features-{args.task}")
    cohort = spark.read.parquet(args.cohort or f"{processed}/cohort")
    labels = spark.read.parquet(args.labels or f"{processed}/labels/{args.task}")
    gold, evidence, _, n_base = build(spark, task=args.task, root=root, fmt=args.fmt, cohort=cohort,
                                      labels=labels, sample_frac=args.sample_frac,
                                      strict_store=not args.no_storetime_filter)
    gold = gold.cache()
    check = check_discharge_gold(gold, task=args.task, cohort=cohort, labels=labels, evidence=evidence,
                                 strict_store=not args.no_storetime_filter)
    print(f"[quality] {check['status']} {check['counts']}")
    if check["status"] != "PASSED":
        raise SystemExit(f"Gold {args.task.upper()} không qua kiểm tra: {check['counts']}")
    gold.repartition(4).write.mode("overwrite").parquet(out)
    print(f"[OK] Đã ghi → {out}")
    write_report(report, gold, args.task, n_base, check, args)
    print(f"[OK] Báo cáo → {report} · {time.time() - t0:.1f}s")
    spark.stop()


if __name__ == "__main__":
    main()

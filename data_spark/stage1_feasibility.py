"""Giai đoạn 1 — Khảo sát khả thi cohort / nhãn / đặc trưng trên MIMIC-IV.

Chỉ xuất SỐ LIỆU TỔNG HỢP (không có dòng bệnh nhân), các ô < ngưỡng bị che.
Kết quả: <out>/stage1_feasibility.json và <out>/stage1_feasibility.md

Ví dụ trên EC2 (đọc thẳng Bronze csv.gz, lâu vì labevents 2.4 GB):
    python -m data_spark.stage1_feasibility --out reports/stage1
Đọc Silver Parquet (nhanh hơn, sau khi chạy bronze_to_silver.py):
    python -m data_spark.stage1_feasibility --format silver \
        --root s3a://heartguard-mimic-group11/silver/mimiciv/3.1 --out reports/stage1
Chạy thử với dữ liệu giả trên máy cá nhân:
    python tools/make_synthetic_mimic.py --out sample_data/mimic_fake
    HG_LOCAL=1 python -m data_spark.stage1_feasibility --root sample_data/mimic_fake --out reports/stage1_fake
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from data_spark.common.io import icd_prefix_condition, load_yaml, read_table
from data_spark.common.session import get_spark

P2_EXCLUDE_DISCH = ["HOSPICE", "AGAINST ADVICE", "ACUTE HOSPITAL"]
PLANNED_TYPES = ["ELECTIVE", "SURGICAL SAME DAY ADMISSION"]


# ----------------------------------------------------------------------------- helpers
class Report:
    def __init__(self, threshold: int):
        self.t = threshold
        self.data: dict = {"generated_at": datetime.now(timezone.utc).isoformat()}

    def safe(self, n):
        """Che ô nhỏ (n > 0 và n < ngưỡng)."""
        if n is None:
            return None
        n = int(n)
        return f"<{self.t}" if 0 < n < self.t else n

    def put(self, key, value):
        self.data[key] = value


def rate(pos, n):
    return round(pos / n, 4) if n else None


def label_summary(df: DataFrame, label: str, rep: Report) -> dict:
    r = df.agg(F.count("*").alias("n"), F.countDistinct("subject_id").alias("patients"),
               F.sum(label).alias("pos")).first()
    n, pos = int(r["n"]), int(r["pos"] or 0)
    return {"admissions": rep.safe(n), "patients": rep.safe(r["patients"]),
            "positives": rep.safe(pos), "positive_rate": rate(pos, n)}


def breakdown(df: DataFrame, by: str, label: str, rep: Report) -> list[dict]:
    rows = (df.groupBy(by).agg(F.count("*").alias("n"), F.sum(label).alias("pos"))
              .orderBy(by).collect())
    return [{by: r[by], "n": rep.safe(r["n"]), "positives": rep.safe(r["pos"] or 0),
             "positive_rate": rate(int(r["pos"] or 0), int(r["n"]))} for r in rows]


def age_group(col):
    return (F.when(col < 65, "18-64").when(col < 80, "65-79").otherwise("80+"))


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    data_cfg = load_yaml("configs/data.yaml")
    ap.add_argument("--root", default=data_cfg["paths"]["bronze"])
    ap.add_argument("--format", choices=["bronze", "silver"], default="bronze")
    ap.add_argument("--out", default="reports/stage1")
    ap.add_argument("--skip-labs", action="store_true", help="bỏ khảo sát labevents (nhanh)")
    ap.add_argument("--with-chartevents", action="store_true",
                    help="đo độ phủ sinh hiệu từ chartevents (3.3 GB, chậm)")
    args = ap.parse_args()

    pcfg = data_cfg["prediction"]
    hf_cfg = load_yaml("configs/hf_codes.yaml")
    items_cfg = load_yaml("configs/feature_items.yaml")
    rep = Report(int(data_cfg["privacy"]["small_cell_threshold"]))
    rep.put("source", {"root": args.root, "format": args.format,
                       "dataset_version": data_cfg["dataset"]["version"]})
    timings: dict[str, float] = {}

    spark = get_spark("hg-stage1-feasibility")
    t0 = time.time()

    patients = read_table(spark, args.root, "patients", args.format)
    adm = read_table(spark, args.root, "admissions", args.format)
    dx = read_table(spark, args.root, "diagnoses_icd", args.format)

    # --- 1. Admission có mã suy tim ------------------------------------------------
    dx = dx.withColumn("icd_code", F.upper(F.regexp_replace(F.trim("icd_code"), r"\.", "")))
    hf_cond = icd_prefix_condition(F.col("icd_code"), F.col("icd_version"), hf_cfg["heart_failure"])
    hf_adm = (dx.where(hf_cond).groupBy("hadm_id").agg(
        F.max(F.when(F.col("seq_num") == 1, 1).otherwise(0)).alias("hf_primary"),
        F.min("icd_version").alias("v_min"), F.max("icd_version").alias("v_max"))
        .withColumn("icd_version_group",
                    F.when(F.col("v_min") != F.col("v_max"), "MIXED")
                     .when(F.col("v_min") == 9, "ICD9").otherwise("ICD10"))
        .drop("v_min", "v_max"))

    base = (adm.join(patients, "subject_id", "inner")
               .withColumn("age", F.col("anchor_age") + F.year("admittime") - F.col("anchor_year")))
    base.cache()

    # --- 2. Attrition cohort chung ---------------------------------------------------
    n_all = base.count()
    s1 = base.join(hf_adm, "hadm_id", "inner")
    s2 = s1.where(F.col("admittime").isNotNull() & F.col("dischtime").isNotNull()
                  & (F.col("dischtime") > F.col("admittime")))
    s3 = s2.where(F.col("age") >= pcfg["min_age"])
    cohort = s3.withColumn("age_group", age_group(F.col("age")))
    cohort.cache()

    def nn(df):
        r = df.agg(F.count("*").alias("a"), F.countDistinct("subject_id").alias("p")).first()
        return {"admissions": rep.safe(r["a"]), "patients": rep.safe(r["p"])}

    attrition = [
        {"step": "S0 all admissions", **nn(base)},
        {"step": "S1 HF code any position", **nn(s1)},
        {"step": "S2 valid admit/discharge times", **nn(s2)},
        {"step": f"S3 age >= {pcfg['min_age']}", **nn(s3)},
    ]
    rep.put("cohort_attrition", attrition)
    rep.put("cohort_option_B_primary_only", nn(cohort.where(F.col("hf_primary") == 1)))
    rep.put("cohort_by_icd_version", [
        {"icd_version_group": r["icd_version_group"], "admissions": rep.safe(r["n"])}
        for r in cohort.groupBy("icd_version_group").agg(F.count("*").alias("n")).collect()])
    timings["cohort_s"] = round(time.time() - t0, 1)

    # --- 3. P1 tử vong trong viện ----------------------------------------------------
    hours = int(pcfg["p1_index_hours"])
    p1_all = cohort.withColumn("prediction_time",
                               F.col("admittime") + F.expr(f"INTERVAL {hours} HOURS"))
    p1a = p1_all.where(F.col("dischtime") > F.col("prediction_time"))
    p1 = p1a.where(F.col("deathtime").isNull() | (F.col("deathtime") > F.col("prediction_time")))
    p1 = p1.withColumn("label_mortality", F.col("hospital_expire_flag").cast("int"))
    p1.cache()
    mismatch = p1.where(
        (F.col("label_mortality") == 1)
        & F.col("deathtime").isNull()
        & (F.coalesce(F.col("discharge_location"), F.lit("")) != "DIED")).count()
    rep.put("P1", {
        "attrition": [
            {"step": "cohort", **nn(cohort)},
            {"step": f"still admitted at +{hours}h", **nn(p1a)},
            {"step": f"alive at +{hours}h", **nn(p1)},
        ],
        "summary": label_summary(p1, "label_mortality", rep),
        "label_consistency_mismatch": rep.safe(mismatch),
        "by_anchor_year_group": breakdown(p1, "anchor_year_group", "label_mortality", rep),
        "by_gender": breakdown(p1, "gender", "label_mortality", rep),
        "by_age_group": breakdown(p1, "age_group", "label_mortality", rep),
        "by_hf_primary": breakdown(p1, "hf_primary", "label_mortality", rep),
    })
    timings["p1_s"] = round(time.time() - t0, 1)

    # --- 4. P2 tái nhập 30 ngày & P3 tử vong 90 ngày ---------------------------------
    w30, w90 = int(pcfg["p2_window_days"]), int(pcfg["p3_window_days"])
    alive = cohort.where(F.col("hospital_expire_flag") == 0)
    nxt = adm.select(F.col("subject_id").alias("s2"), F.col("hadm_id").alias("h2"),
                     F.col("admittime").alias("adm2"), F.col("admission_type").alias("type2"))
    readm = (alive.select("subject_id", "hadm_id", "dischtime")
             .join(nxt, (F.col("subject_id") == F.col("s2")) & (F.col("hadm_id") != F.col("h2"))
                   & (F.col("adm2") > F.col("dischtime"))
                   & (F.col("adm2") <= F.col("dischtime") + F.expr(f"INTERVAL {w30} DAYS"))
                   & (~F.col("type2").isin(PLANNED_TYPES)), "inner")
             .select("hadm_id").distinct().withColumn("label_readmit30", F.lit(1)))
    p2_all = (alive.join(readm, "hadm_id", "left")
                   .fillna({"label_readmit30": 0})
                   .withColumn("days_to_death",
                               F.datediff(F.col("dod"), F.to_date("dischtime"))))
    p2b = p2_all.where(~F.coalesce(F.col("discharge_location"), F.lit("")).isin(P2_EXCLUDE_DISCH))
    died30_no_readmit = ((F.col("label_readmit30") == 0) & F.col("days_to_death").isNotNull()
                         & (F.col("days_to_death") <= w30))
    p2 = p2b.where(~died30_no_readmit)
    rep.put("P2", {
        "attrition": [
            {"step": "cohort", **nn(cohort)},
            {"step": "discharged alive", **nn(alive)},
            {"step": "exclude hospice/AMA/acute transfer", **nn(p2b)},
            {"step": f"exclude death <= {w30}d without readmission", **nn(p2)},
        ],
        "summary": label_summary(p2, "label_readmit30", rep),
        "sensitivity_keep_deaths_as_0": label_summary(p2b, "label_readmit30", rep),
        "by_anchor_year_group": breakdown(p2, "anchor_year_group", "label_readmit30", rep),
    })

    p3 = (p2_all.where(F.coalesce(F.col("discharge_location"), F.lit("")) != "HOSPICE")
                .withColumn("label_mort90",
                            F.when(F.col("days_to_death").between(0, w90), 1).otherwise(0)))
    rep.put("P3", {
        "summary": label_summary(p3, "label_mort90", rep),
        "by_anchor_year_group": breakdown(p3, "anchor_year_group", "label_mort90", rep),
        "note": "dod observed up to ~1 year after last discharge; 90-day window fully observed",
    })
    timings["p2_p3_s"] = round(time.time() - t0, 1)

    # --- 5. Độ phủ ICU trong 24h đầu (P1) ---------------------------------------------
    icu = read_table(spark, args.root, "icustays", args.format)
    p1_keys = p1.select("subject_id", "hadm_id", "admittime", "prediction_time")
    icu24 = (p1_keys.join(icu.select("hadm_id", "intime", "outtime"), "hadm_id", "inner")
             .where((F.col("intime") <= F.col("prediction_time")) & (F.col("outtime") >= F.col("admittime")))
             .select("hadm_id").distinct())
    n_p1 = p1.count()
    n_icu24 = icu24.count()
    rep.put("P1_icu_within_24h", {"admissions": rep.safe(n_icu24), "rate": rate(n_icu24, n_p1)})

    d_items = read_table(spark, args.root, "d_items", args.format)
    vital_cands = []
    for key, spec in items_cfg["vitals"].items():
        for r in d_items.where(F.col("label").rlike(spec["label_regex"])).collect():
            vital_cands.append({"feature": key, "itemid": r["itemid"], "label": r["label"],
                                "linksto": r["linksto"], "unit": r["unitname"]})
    rep.put("vital_item_candidates", vital_cands)

    if args.with_chartevents and vital_cands:
        ce = read_table(spark, args.root, "chartevents", args.format)
        ids = [c["itemid"] for c in vital_cands]
        feat_of = F.create_map(*[x for c in vital_cands for x in (F.lit(c["itemid"]), F.lit(c["feature"]))])
        cov = (ce.where(F.col("itemid").isin(ids))
                 .join(F.broadcast(p1_keys), ["subject_id", "hadm_id"], "inner")
                 .where((F.col("charttime") >= F.col("admittime")) & (F.col("charttime") <= F.col("prediction_time")))
                 .withColumn("feature", feat_of[F.col("itemid")])
                 .groupBy("feature").agg(F.countDistinct("hadm_id").alias("adm")).collect())
        rep.put("P1_vital_coverage", [{"feature": r["feature"], "admissions": rep.safe(r["adm"]),
                                       "rate": rate(int(r["adm"]), n_p1)} for r in cov])
    timings["icu_s"] = round(time.time() - t0, 1)

    # --- 6. Xét nghiệm: itemid ứng viên + độ phủ trong cửa sổ P1 -----------------------
    d_lab = read_table(spark, args.root, "d_labitems", args.format)
    lab_cands = []
    for key, spec in items_cfg["labs"].items():
        q = d_lab.where(F.col("label").rlike(spec["label_regex"]))
        if spec.get("fluid"):
            q = q.where(F.col("fluid") == spec["fluid"])
        for r in q.collect():
            lab_cands.append({"feature": key, "itemid": r["itemid"], "label": r["label"],
                              "fluid": r["fluid"], "category": r["category"]})
    rep.put("lab_item_candidates_count", len(lab_cands))

    if not args.skip_labs and lab_cands:
        lab = read_table(spark, args.root, "labevents", args.format)
        lb = int(pcfg["lab_lookback_hours"])
        ids = [c["itemid"] for c in lab_cands]
        win = (lab.where(F.col("itemid").isin(ids) & F.col("valuenum").isNotNull())
                  .withColumnRenamed("hadm_id", "lab_hadm_id")
                  .join(F.broadcast(p1_keys), "subject_id", "inner")
                  .where((F.col("charttime") >= F.col("admittime") - F.expr(f"INTERVAL {lb} HOURS"))
                         & (F.col("charttime") <= F.col("prediction_time"))))
        win.cache()
        per_item = (win.groupBy("itemid").agg(
            F.count("*").alias("rows"),
            F.countDistinct("hadm_id").alias("adm"),
            F.sum(F.when(F.col("lab_hadm_id").isNull(), 1).otherwise(0)).alias("rows_null_hadm"),
            F.sum(F.when(F.col("storetime") > F.col("prediction_time"), 1).otherwise(0)).alias("rows_stored_after_pt"),
        ).collect())
        units = (win.groupBy("itemid", "valueuom").agg(F.count("*").alias("n")).collect())
        unit_map: dict[int, list] = {}
        for u in units:
            unit_map.setdefault(u["itemid"], []).append((u["valueuom"], int(u["n"])))
        stats = {r["itemid"]: r for r in per_item}
        for c in lab_cands:
            r = stats.get(c["itemid"])
            if r is None:
                c.update({"rows_in_window": 0, "p1_admissions": 0, "p1_coverage": 0.0})
                continue
            us = sorted(unit_map.get(c["itemid"], []), key=lambda x: -x[1])
            rows = int(r["rows"])
            c.update({
                "rows_in_window": rep.safe(rows),
                "p1_admissions": rep.safe(r["adm"]),
                "p1_coverage": rate(int(r["adm"]), n_p1),
                "share_rows_hadm_null": rate(int(r["rows_null_hadm"]), rows),
                "share_rows_stored_after_prediction_time": rate(int(r["rows_stored_after_pt"]), rows),
                "dominant_unit": us[0][0] if us else None,
                "dominant_unit_share": rate(us[0][1], rows) if us else None,
            })
        # Độ phủ theo feature (gộp các itemid ứng viên)
        feat_of = F.create_map(*[x for c in lab_cands for x in (F.lit(c["itemid"]), F.lit(c["feature"]))])
        cov = (win.withColumn("feature", feat_of[F.col("itemid")])
                  .groupBy("feature").agg(F.countDistinct("hadm_id").alias("adm")).collect())
        rep.put("P1_lab_coverage", sorted(
            [{"feature": r["feature"], "admissions": rep.safe(r["adm"]),
              "coverage": rate(int(r["adm"]), n_p1), "missing_rate": round(1 - (rate(int(r["adm"]), n_p1) or 0), 4)}
             for r in cov], key=lambda x: x["feature"]))
        win.unpersist()
    rep.put("lab_item_candidates", lab_cands)
    timings["labs_s"] = round(time.time() - t0, 1)

    # --- 7. OMR trước nhập viện ---------------------------------------------------------
    omr = read_table(spark, args.root, "omr", args.format)
    names = items_cfg["omr"]["result_names"]
    omr_cov = (omr.where(F.col("result_name").isin(names))
                  .join(F.broadcast(p1_keys), "subject_id", "inner")
                  .where((F.col("chartdate") < F.to_date("admittime"))
                         & (F.col("chartdate") >= F.date_sub(F.to_date("admittime"), 365)))
                  .groupBy("result_name").agg(F.countDistinct("hadm_id").alias("adm")).collect())
    rep.put("P1_omr_prior_365d_coverage", [
        {"result_name": r["result_name"], "admissions": rep.safe(r["adm"]),
         "coverage": rate(int(r["adm"]), n_p1)} for r in omr_cov])

    timings["total_s"] = round(time.time() - t0, 1)
    rep.put("timings_seconds", timings)
    rep.put("spark", {"master": spark.sparkContext.master,
                      "default_parallelism": spark.sparkContext.defaultParallelism,
                      "version": spark.version})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage1_feasibility.json").write_text(json.dumps(rep.data, indent=2, ensure_ascii=False, default=str),
                                                 encoding="utf-8")
    (out / "stage1_feasibility.md").write_text(to_markdown(rep.data), encoding="utf-8")
    print(f"[OK] Đã ghi {out}/stage1_feasibility.json và .md  ({timings['total_s']} s)")
    spark.stop()


# ----------------------------------------------------------------------------- markdown
def _table(rows: list[dict]) -> str:
    if not rows:
        return "_(không có)_\n"
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def to_markdown(d: dict) -> str:
    md = [f"# Báo cáo khả thi Giai đoạn 1\n\nTạo lúc: {d['generated_at']} · Nguồn: `{d['source']['root']}` "
          f"({d['source']['format']}, MIMIC-IV v{d['source']['dataset_version']})\n",
          "> Chỉ chứa số liệu tổng hợp; ô nhỏ hơn ngưỡng đã được che.\n",
          "## Cohort chung\n", _table(d["cohort_attrition"]),
          f"\nPhương án B (HF là chẩn đoán chính): {d['cohort_option_B_primary_only']}\n",
          "\nTheo phiên bản ICD:\n", _table(d["cohort_by_icd_version"])]
    for task, label in (("P1", "tử vong trong viện"), ("P2", "tái nhập 30 ngày"), ("P3", "tử vong 90 ngày")):
        t = d[task]
        md.append(f"\n## {task} — {label}\n")
        if "attrition" in t:
            md.append(_table(t["attrition"]))
        md.append(f"\n**Tổng hợp:** {t['summary']}\n")
        if task == "P1":
            md.append(f"\nNhãn lệch (expire_flag=1 nhưng không có deathtime/DIED): {t['label_consistency_mismatch']}\n")
            for k in ("by_gender", "by_age_group", "by_hf_primary"):
                md.append(f"\n{k}:\n" + _table(t[k]))
        if task == "P2":
            md.append(f"\nĐộ nhạy (giữ ca tử vong, nhãn 0): {t['sensitivity_keep_deaths_as_0']}\n")
        md.append("\nTheo anchor_year_group (cho drift):\n" + _table(t["by_anchor_year_group"]))
    md.append(f"\n## Sinh hiệu\n\nP1 có ICU trong 24h đầu: {d['P1_icu_within_24h']}\n")
    if "P1_vital_coverage" in d:
        md.append(_table(d["P1_vital_coverage"]))
    md.append("\nItem sinh hiệu ứng viên:\n" + _table(d["vital_item_candidates"]))
    if "P1_lab_coverage" in d:
        md.append("\n## Xét nghiệm — độ phủ trong cửa sổ P1\n" + _table(d["P1_lab_coverage"]))
    md.append("\nItemid xét nghiệm ứng viên (TV4 chọn itemid cuối):\n" + _table(d["lab_item_candidates"]))
    md.append("\n## OMR 365 ngày trước nhập viện\n" + _table(d["P1_omr_prior_365d_coverage"]))
    md.append(f"\n## Thời gian chạy\n{d['timings_seconds']}\n\nSpark: {d['spark']}\n")
    return "\n".join(md)


if __name__ == "__main__":
    main()

"""Giai đoạn 2 - Lọc cohort suy tim (phương án A: mã HF ở bất kỳ seq_num).

Các bước (mỗi bước đều đếm số bệnh nhân / số lần nhập viện để làm bảng attrition):
    S0  Toàn bộ admissions
    S1  Có mã suy tim trong diagnoses_icd (theo configs/hf_codes.yaml)
    S2  Có admittime, dischtime và dischtime > admittime
    S3  Có hồ sơ patients và tuổi tại nhập viện >= 18

Đầu ra:
    - Parquet: <out>  (mỗi dòng = 1 lần nhập viện)
    - Markdown: docs/stage2/cohort_result.md  (bảng đếm qua từng bước)

Chạy thử trên dữ liệu giả (laptop), đọc thẳng từ CSV giả:
    python -m data_spark.cohort --fmt bronze --root sample_data/mimic_fake --out sample_data/cohort_fake

Chạy thật trên EC2 (đọc silver Parquet trên S3, ghi lên S3):
    python -m data_spark.cohort
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from data_spark.common.io import REPO_ROOT, icd_prefix_condition, load_yaml, read_table
from data_spark.common.session import get_spark

# Cột đầu ra, đúng theo phân công GĐ2
COHORT_COLS = [
    "subject_id", "hadm_id",
    "admittime", "dischtime", "deathtime", "hospital_expire_flag",
    "discharge_location", "admission_type", "admission_location", "edregtime",
    "anchor_year_group", "gender", "age", "race", "insurance",
    "hf_primary",
]


def count_step(df: DataFrame) -> tuple[int, int]:
    """Trả về (số bệnh nhân, số lần nhập viện)."""
    r = df.agg(
        F.countDistinct("subject_id").alias("n_pat"),
        F.countDistinct("hadm_id").alias("n_adm"),
    ).first()
    return int(r["n_pat"]), int(r["n_adm"])


def build_cohort(adm: DataFrame, pat: DataFrame, dx: DataFrame, hf_codes: dict):
    """Trả về (cohort_df, danh sách các bước attrition)."""
    steps: list[dict] = []

    def record(step: str, desc: str, df: DataFrame) -> None:
        n_pat, n_adm = count_step(df)
        steps.append({"step": step, "desc": desc, "patients": n_pat, "admissions": n_adm})
        print(f"[{step}] {desc:<60s} bệnh nhân={n_pat:>9,d}  nhập viện={n_adm:>9,d}")

    # S0: toàn bộ admissions
    s0 = adm
    record("S0", "Toàn bộ admissions", s0)

    # S1: có mã suy tim. Gộp về 1 dòng / hadm_id để không nhân đôi admission.
    # hf_primary = 1 nếu có mã HF ở seq_num = 1 (chẩn đoán chính), ngược lại 0.
    hf = (
        dx.filter(icd_prefix_condition(F.col("icd_code"), F.col("icd_version"), hf_codes))
        .groupBy("hadm_id")
        .agg(F.max((F.col("seq_num") == 1).cast("int")).alias("hf_primary"))
    )
    s1 = s0.join(hf, on="hadm_id", how="inner")
    record("S1", "Có mã suy tim (HF) ở bất kỳ seq_num", s1)

    # S2: thời gian hợp lệ
    s2 = s1.filter(
        F.col("admittime").isNotNull()
        & F.col("dischtime").isNotNull()
        & (F.col("dischtime") > F.col("admittime"))
    )
    record("S2", "Có admittime/dischtime và dischtime > admittime", s2)

    # S3: ghép patients, tính tuổi tại nhập viện, giữ >= 18
    pat_cols = pat.select("subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group")
    s3 = (
        s2.join(pat_cols, on="subject_id", how="inner")
        .withColumn("age", F.col("anchor_age") + F.year("admittime") - F.col("anchor_year"))
        .filter(F.col("age") >= 18)
    )
    record("S3", "Có hồ sơ patients và tuổi tại nhập viện >= 18", s3)

    cohort = s3.select(*COHORT_COLS)
    return cohort, steps


def write_report(path: Path, steps: list[dict], extra: dict) -> None:
    lines = [
        "# Kết quả cohort suy tim (Giai đoạn 2)",
        "",
        "Phương án A: có mã HF ở bất kỳ `seq_num`. Mã ICD theo `configs/hf_codes.yaml`.",
        "",
        "## Bảng đếm qua từng bước lọc",
        "",
        "| Bước | Điều kiện | Số bệnh nhân | Số lần nhập viện | Bị loại (nhập viện) |",
        "|---|---|---:|---:|---:|",
    ]
    prev = None
    for s in steps:
        dropped = "" if prev is None else f"{prev - s['admissions']:,}"
        lines.append(
            f"| {s['step']} | {s['desc']} | {s['patients']:,} | {s['admissions']:,} | {dropped} |"
        )
        prev = s["admissions"]
    lines += [
        "",
        "## Thông tin thêm về cohort cuối",
        "",
        f"- Số lần nhập viện: {extra['n_adm']:,}",
        f"- Số bệnh nhân: {extra['n_pat']:,}",
        f"- Suy tim là chẩn đoán chính (`hf_primary = 1`): {extra['n_primary']:,}",
        f"- Suy tim là chẩn đoán phụ (`hf_primary = 0`): {extra['n_secondary']:,}",
        f"- Tử vong trong viện (`hospital_expire_flag = 1`): {extra['n_death']:,}",
        "",
        "Chỉ chứa số liệu tổng hợp, không có dữ liệu cá nhân.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = load_yaml("configs/data.yaml")
    hf_codes = load_yaml("configs/hf_codes.yaml")["heart_failure"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", choices=["silver", "bronze"], default="silver",
                    help="silver = Parquet (thật), bronze = CSV (dùng khi thử dữ liệu giả)")
    ap.add_argument("--root", default=None,
                    help="Thư mục gốc dữ liệu; mặc định lấy từ configs/data.yaml theo --fmt")
    ap.add_argument("--out", default=None,
                    help="Nơi ghi cohort Parquet; mặc định <paths.processed>/cohort")
    ap.add_argument("--report", default=str(REPO_ROOT / "docs/stage2/cohort_result.md"))
    args = ap.parse_args()

    root = args.root or cfg["paths"][args.fmt]
    processed = cfg["paths"].get("processed", "s3://heartguard-mimic-group11/processed")
    out = args.out or f"{processed.rstrip('/')}/cohort"

    t0 = time.time()
    spark = get_spark("hg-cohort")

    adm = read_table(spark, root, "admissions", args.fmt)
    pat = read_table(spark, root, "patients", args.fmt)
    dx = read_table(spark, root, "diagnoses_icd", args.fmt)

    cohort, steps = build_cohort(adm, pat, dx, hf_codes)
    cohort = cohort.cache()

    # Kiểm tra: mỗi hadm_id đúng 1 dòng
    n_rows = cohort.count()
    n_hadm = cohort.select("hadm_id").distinct().count()
    assert n_rows == n_hadm, f"hadm_id bị trùng: {n_rows} dòng nhưng {n_hadm} hadm_id"

    n_pat, n_adm = count_step(cohort)
    stats = cohort.agg(
        F.sum("hf_primary").alias("n_primary"),
        F.sum("hospital_expire_flag").alias("n_death"),
    ).first()
    n_primary = int(stats["n_primary"] or 0)
    extra = {
        "n_adm": n_adm,
        "n_pat": n_pat,
        "n_primary": n_primary,
        "n_secondary": n_adm - n_primary,
        "n_death": int(stats["n_death"] or 0),
    }

    cohort.repartition(4).write.mode("overwrite").parquet(out)
    print(f"[OK] Đã ghi cohort → {out}")

    write_report(Path(args.report), steps, extra)
    print(f"[OK] Đã ghi báo cáo → {args.report}")
    print(f"Xong sau {time.time() - t0:.1f}s")
    spark.stop()


if __name__ == "__main__":
    main()
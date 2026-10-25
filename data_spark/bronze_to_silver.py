"""Bronze (csv.gz) → Silver (Parquet đã ép kiểu, repartition). Chạy 1 lần.

csv.gz không chia nhỏ được nên mỗi file chỉ đọc bằng 1 task; Parquet đã repartition
cho phép các job sau chạy song song thật sự. Script ghi lại thời gian và số dòng → dùng cho benchmark.

    python -m data_spark.bronze_to_silver --tables patients admissions diagnoses_icd d_labitems labevents
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from data_spark.common.io import TABLE_SPECS, load_yaml, read_table
from data_spark.common.session import get_spark

# Số partition gợi ý theo kích thước bảng
PARTITIONS = {"labevents": 400, "chartevents": 600, "omr": 16, "diagnoses_icd": 16,
              "admissions": 8, "icustays": 4}


def main() -> None:
    cfg = load_yaml("configs/data.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--bronze", default=cfg["paths"]["bronze"])
    ap.add_argument("--silver", default=cfg["paths"]["silver"])
    ap.add_argument("--tables", nargs="+", default=list(TABLE_SPECS))
    ap.add_argument("--log", default="reports/benchmark/bronze_to_silver.json")
    args = ap.parse_args()

    spark = get_spark("hg-bronze-to-silver")
    log = []
    for t in args.tables:
        t0 = time.time()
        df = read_table(spark, args.bronze, t, "bronze")
        df = df.repartition(PARTITIONS.get(t, 2))
        module = TABLE_SPECS[t]["module"]
        dst = f"{args.silver.rstrip('/')}/{module}/{t}"
        df.write.mode("overwrite").parquet(dst)
        n = spark.read.parquet(dst).count()
        sec = round(time.time() - t0, 1)
        log.append({"table": t, "rows": n, "seconds": sec, "partitions": PARTITIONS.get(t, 2),
                    "master": spark.sparkContext.master})
        print(f"[OK] {t:15s} {n:>12,d} dòng  {sec:>8.1f}s → {dst}")
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).write_text(json.dumps(log, indent=2), encoding="utf-8")
    spark.stop()


if __name__ == "__main__":
    main()

"""Đo tốc độ Spark vs pandas cho tác vụ: tổng hợp xét nghiệm 24h (first/min/max theo hadm_id x lab).

Mỗi lần chạy 1 cấu hình, ghi nối vào CSV:
    python -m data_spark.benchmark --engine spark  --frac 0.05 --master "local[2]" --label spark-2w
    python -m data_spark.benchmark --engine pandas --frac 0.05 --label pandas
Vẽ biểu đồ:
    python -m data_spark.benchmark --plot reports/benchmark/bench.csv
Mẫu: subject_id % 10000 < frac*10000 (cùng quy tắc cho cả hai engine).
Tác vụ giống nhau ở hai engine: cửa sổ [admittime-6h, admittime+24h], lọc 20 itemid, storetime <= mốc,
valuenum không NULL, lấy first/min/max (KHÔNG gồm bước đơn vị/khoảng hợp lệ để hai bên đúng cùng một việc).
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from pathlib import Path

from data_spark.common.io import REPO_ROOT, load_yaml

FIELDS = [
    "engine",
    "label",
    "frac",
    "repeat",
    "seconds",
    "read_seconds",
    "startup_seconds",
    "rows_out",
    "n_subjects",
    "master",
]


def lab_ids(items):
    return {i: n for n, c in items["labs"].items() for i in c["itemids"]}


# ----------------------------------------------------------------------------- Spark
def run_spark(args, cfg, items):
    from pyspark.sql import functions as F

    if args.master:  # PySpark đọc biến này nếu get_spark() không ép master khác
        os.environ["PYSPARK_SUBMIT_ARGS"] = f"--master {args.master} pyspark-shell"
    from data_spark.common.session import get_spark

    t0 = time.time()
    spark = get_spark(f"hg-bench-{args.label}")
    spark.range(1).count()  # ép khởi động xong
    startup = time.time() - t0
    master = spark.sparkContext.master
    print(
        f"[spark] master thực tế = {master}; defaultParallelism = {spark.sparkContext.defaultParallelism}; khởi động {startup:.1f}s"
    )
    root, processed = cfg["paths"]["silver"], cfg["paths"]["processed"].rstrip("/")
    ids = lab_ids(items)
    name_col = F.coalesce(
        *[F.when(F.col("itemid") == i, F.lit(n)) for i, n in ids.items()]
    )
    results = []
    for rep in range(1, args.repeats + 1):
        t = time.time()
        co = (
            spark.read.parquet(args.cohort or f"{processed}/cohort")
            .filter(F.pmod(F.col("subject_id"), F.lit(10000)) < int(args.frac * 10000))
            .select("subject_id", "hadm_id", "admittime")
            .withColumn("pred", F.expr("admittime + INTERVAL 24 HOURS"))
        )
        le = (
            spark.read.parquet(f"{root}/labevents")
            .filter(F.col("itemid").isin(list(ids)))
            .select("subject_id", "itemid", "charttime", "storetime", "valuenum")
        )
        j = (
            le.join(F.broadcast(co), "subject_id")
            .filter(
                (F.col("charttime") >= F.expr("admittime - INTERVAL 6 HOURS"))
                & (F.col("charttime") <= F.col("pred"))
                & (F.col("storetime").isNull() | (F.col("storetime") <= F.col("pred")))
                & F.col("valuenum").isNotNull()
            )
            .withColumn("lab", name_col)
        )
        res = (
            j.groupBy("hadm_id", "lab")
            .agg(
                F.min(F.struct("charttime", "valuenum"))
                .getField("valuenum")
                .alias("first"),
                F.min("valuenum").alias("min"),
                F.max("valuenum").alias("max"),
            )
            .cache()
        )
        n = res.count()  # ép tính toán thật (Spark lazy)
        el = time.time() - t
        res.unpersist()
        results.append(
            dict(
                engine="spark",
                label=args.label,
                frac=args.frac,
                repeat=rep,
                seconds=round(el, 3),
                read_seconds="",
                startup_seconds=round(startup, 3),
                rows_out=n,
                n_subjects=co.select("subject_id").distinct().count(),
                master=master,
            )
        )
        print(f"[spark] lần {rep}: {el:.1f}s, {n:,} dòng")
    spark.stop()
    return results


# ----------------------------------------------------------------------------- pandas
def run_pandas(args, cfg, items):
    import pandas as pd
    import pyarrow.dataset as ds
    import pyarrow.fs as pafs

    root, processed = cfg["paths"]["silver"], cfg["paths"]["processed"].rstrip("/")
    ids = lab_ids(items)

    def open_ds(uri):
        uri = uri.replace("s3a://", "s3://")
        if "://" in uri:
            fs, path = pafs.FileSystem.from_uri(uri)
            return ds.dataset(path, filesystem=fs, format="parquet")
        return ds.dataset(uri, format="parquet")

    results = []
    for rep in range(1, args.repeats + 1):
        t = time.time()
        co = (
            open_ds(args.cohort or f"{processed}/cohort")
            .to_table(columns=["subject_id", "hadm_id", "admittime"])
            .to_pandas()
        )
        co = co[co["subject_id"] % 10000 < int(args.frac * 10000)].copy()
        co["pred"] = co["admittime"] + pd.Timedelta(hours=24)
        subj = co["subject_id"].unique().tolist()
        le = (
            open_ds(f"{root}/labevents")
            .to_table(
                columns=["subject_id", "itemid", "charttime", "storetime", "valuenum"],
                filter=ds.field("itemid").isin(list(ids))
                & ds.field("subject_id").isin(subj),
            )
            .to_pandas()
        )
        read_s = time.time() - t
        j = le.merge(co, on="subject_id")
        j = j[
            (j["charttime"] >= j["admittime"] - pd.Timedelta(hours=6))
            & (j["charttime"] <= j["pred"])
            & (j["storetime"].isna() | (j["storetime"] <= j["pred"]))
            & j["valuenum"].notna()
        ].copy()
        j["lab"] = j["itemid"].map(ids)
        j = j.sort_values(["hadm_id", "lab", "charttime", "valuenum"])
        res = (
            j.groupby(["hadm_id", "lab"])
            .agg(
                first=("valuenum", "first"),
                min=("valuenum", "min"),
                max=("valuenum", "max"),
            )
            .reset_index()
        )
        el = time.time() - t
        results.append(
            dict(
                engine="pandas",
                label=args.label,
                frac=args.frac,
                repeat=rep,
                seconds=round(el, 3),
                read_seconds=round(read_s, 3),
                startup_seconds="",
                rows_out=len(res),
                n_subjects=len(subj),
                master="pandas",
            )
        )
        print(f"[pandas] lần {rep}: {el:.1f}s (đọc {read_s:.1f}s), {len(res):,} dòng")
    return results


# ----------------------------------------------------------------------------- biểu đồ
def plot(csv_path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        groups.setdefault((r["label"], float(r["frac"])), []).append(
            float(r["seconds"])
        )
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label in sorted({k[0] for k in groups}):
        pts = sorted(
            (f, statistics.mean(v), statistics.pstdev(v))
            for (l, f), v in groups.items()
            if l == label
        )
        ax.errorbar(
            [p[0] * 100 for p in pts],
            [p[1] for p in pts],
            yerr=[p[2] for p in pts],
            marker="o",
            capsize=3,
            label=label,
        )
    ax.set_xlabel("Tỷ lệ bệnh nhân lấy mẫu (%)")
    ax.set_ylabel("Thời gian chạy (giây)")
    ax.set_title("Tổng hợp xét nghiệm 24h: Spark vs pandas")
    ax.grid(alpha=0.3)
    ax.legend()
    out = Path(csv_path).with_suffix(".png")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print("[OK] biểu đồ →", out)
    print(
        "| Cấu hình | Mẫu | Trung bình (s) | Độ lệch chuẩn (s) | Số lần |\n|---|---:|---:|---:|---:|"
    )
    for (l, f), v in sorted(groups.items()):
        print(
            f"| {l} | {f:.0%} | {statistics.mean(v):.1f} | {statistics.pstdev(v):.1f} | {len(v)} |"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["spark", "pandas"])
    ap.add_argument("--frac", type=float, default=0.01)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--master", default=None, help='vd "local[2]" hoặc spark://host:7077'
    )
    ap.add_argument("--label", default=None)
    ap.add_argument("--cohort", default=None)
    ap.add_argument("--csv", default=str(REPO_ROOT / "reports/benchmark/bench.csv"))
    ap.add_argument("--plot", default=None, help="đường dẫn CSV để vẽ biểu đồ")
    args = ap.parse_args()
    if args.plot:
        return plot(args.plot)
    args.label = args.label or args.engine
    cfg, items = load_yaml("configs/data.yaml"), load_yaml("configs/feature_items.yaml")
    rows = (run_spark if args.engine == "spark" else run_pandas)(args, cfg, items)
    p = Path(args.csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        new and w.writeheader()
        w.writerows(rows)
    print("[OK] đã ghi", p)


if __name__ == "__main__":
    main()

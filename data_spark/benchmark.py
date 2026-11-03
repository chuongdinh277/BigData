"""Performance Benchmark — Apache Spark vs pandas.

Module thuộc Giai đoạn 2 của dự án HeartGuard AI, chịu trách nhiệm
đánh giá hiệu năng xử lý dữ liệu xét nghiệm bằng Apache Spark và pandas.

1. MỤC TIÊU
   - So sánh thời gian xử lý giữa Spark và pandas trên cùng tác vụ.
   - Đánh giá ảnh hưởng của tỷ lệ dữ liệu đến thời gian thực thi.
   - Đánh giá khả năng tăng tốc khi tăng mức song song của Spark.
   - Cung cấp số liệu và biểu đồ phục vụ báo cáo thực nghiệm.

2. PHẠM VI BENCHMARK
   Tác vụ được đo là tổng hợp 20 loại xét nghiệm trong cửa sổ 24 giờ,
   tương ứng với một phần của pipeline Feature Engineering P1.

   Quy trình xử lý:
   - Đọc cohort và bảng labevents từ dữ liệu Parquet.
   - Lấy mẫu bệnh nhân theo subject_id % 10000.
   - Chỉ chọn các itemid thuộc 20 loại xét nghiệm được cấu hình.
   - Ghép xét nghiệm với cohort theo subject_id.
   - Lọc dữ liệu trong khoảng [admittime - 6h, admittime + 24h].
   - Chỉ giữ bản ghi có storetime NULL hoặc không vượt mốc dự đoán.
   - Loại bỏ bản ghi có valuenum NULL.
   - Nhóm dữ liệu theo (hadm_id, lab).
   - Tính ba giá trị: first, min và max.

   Giá trị first được xác định theo thứ tự (charttime, valuenum).

   Benchmark không bao gồm toàn bộ features.py, không thực hiện
   kiểm tra đơn vị đo, khoảng giá trị hợp lệ hoặc các nhóm đặc trưng
   khác như demographics, medical history, OMR và vital signs.

3. NGUỒN DỮ LIỆU VÀ CẤU HÌNH
   - Cohort: danh sách các đợt nhập viện thuộc quần thể nghiên cứu.
   - labevents: dữ liệu xét nghiệm từ MIMIC-IV.
   - configs/data.yaml: cấu hình đường dẫn dữ liệu.
   - configs/feature_items.yaml: danh sách xét nghiệm và itemid.
   - Dữ liệu được đọc từ Parquet trên Amazon S3.

   Hai engine sử dụng cùng quy tắc lấy mẫu và tổng hợp.
   Tuy nhiên, cơ chế đọc và lọc dữ liệu của Spark và pandas khác nhau.

4. THIẾT KẾ THỰC NGHIỆM
   Tỷ lệ lấy mẫu:
   - 1% bệnh nhân.
   - 5% bệnh nhân.
   - 10% bệnh nhân.

   Cấu hình thực thi:
   - pandas: xử lý dữ liệu trên một máy.
   - Spark local[1]: tối đa 1 luồng thực thi tác vụ Spark.
   - Spark local[2]: tối đa 2 luồng thực thi tác vụ Spark.
   - Spark local[3]: tối đa 3 luồng thực thi tác vụ Spark.

   Mỗi cấu hình được chạy 3 lần để giảm ảnh hưởng của biến động
   trong môi trường thực nghiệm.

   Tổng số phép đo dự kiến:
       3 tỷ lệ dữ liệu x 4 cấu hình x 3 lần = 36 phép đo.

5. PHƯƠNG PHÁP ĐO
   - Sử dụng wall-clock time để đo thời gian thực thi.
   - Bao gồm thời gian đọc dữ liệu và tính toán.
   - Spark được khởi động trước vòng đo; startup time ghi riêng.
   - Buộc Spark thực hiện các phép tổng hợp trước khi kết thúc đo.
   - Ghi số dòng kết quả và số bệnh nhân được xử lý.
   - Ghi tổng first/min/max để hỗ trợ đối chiếu giữa hai engine.
   - Tính thời gian trung bình và độ lệch chuẩn của các lần chạy.

   Các chỉ số được sử dụng:
   - seconds: tổng thời gian xử lý của một lần chạy.
   - read_seconds: thời gian đọc dữ liệu, nếu được ghi nhận riêng.
   - startup_seconds: thời gian khởi động Spark.
   - rows_out: số dòng sau tổng hợp.
   - n_subjects: số bệnh nhân trong mẫu.
   - sum_first, sum_min, sum_max: tổng kiểm tra kết quả tính toán.
   - master: chế độ thực thi của engine.

6. KIỂM TRA TÍNH NHẤT QUÁN
   Kết quả Spark và pandas được đối chiếu thông qua:
   - Số lượng bệnh nhân.
   - Số dòng đầu ra.
   - Tổng các giá trị first/min/max trong sai số cho phép.

   Việc các chỉ số tổng hợp khớp nhau không chứng minh toàn bộ
   giá trị của từng dòng đều giống nhau. Nếu cần xác nhận chính xác,
   phải thực hiện đối chiếu dữ liệu ở cấp (hadm_id, lab).

7. ĐẦU RA
   - CSV: lưu kết quả đo của từng cấu hình và từng lần chạy.
   - PNG: biểu đồ thời gian trung bình và độ lệch chuẩn.
   - Bảng thống kê: tổng hợp thời gian theo cấu hình và tỷ lệ mẫu.

   Đường dẫn kết quả thực nghiệm:
       reports/benchmark/bench_final.csv
       reports/benchmark/bench_final.png

   Tài liệu phân tích:
       docs/stage2/benchmark.md

8. CÁCH CHẠY

   Chạy Spark với 2 luồng, mẫu 1%:
       python -m data_spark.benchmark \\
           --engine spark --frac 0.01 --repeats 3 \\
           --master "local[2]" --label spark-local2 \\
           --csv reports/benchmark/bench_final.csv

   Chạy pandas với mẫu 1%:
       python -m data_spark.benchmark \\
           --engine pandas --frac 0.01 --repeats 3 \\
           --label pandas \\
           --csv reports/benchmark/bench_final.csv

   Tạo biểu đồ và bảng thống kê:
       python -m data_spark.benchmark \\
           --plot reports/benchmark/bench_final.csv

9. GIỚI HẠN THỰC NGHIỆM
   - Spark chạy ở local mode trên một EC2, không phải cluster phân tán.
   - local[1], local[2], local[3] biểu thị mức song song trên một máy,
     không tương đương 1, 2 hoặc 3 worker node độc lập.
   - Hai engine có cơ chế đọc và lọc Parquet khác nhau.
   - Kết quả phản ánh hiệu năng của pipeline được đo, không phải
     so sánh thuần túy tốc độ của hai công cụ tính toán.
   - Thời gian có thể chịu ảnh hưởng của S3 I/O, cache, CPU và RAM.
   - Chưa đánh giá dữ liệu vượt bộ nhớ hoặc khả năng mở rộng cluster.
   - Kết quả Benchmark không đại diện cho toàn bộ Feature Engineering.

10. LƯU Ý
    - Sử dụng cùng cohort và cùng tỷ lệ lấy mẫu cho hai engine.
    - Kiểm tra Spark master thực tế trước khi đo.
    - Không chạy đồng thời các cấu hình trên cùng máy khi đo chuẩn.
    - Kết quả được ghi nối vào CSV; chạy lại có thể tạo bản ghi trùng.
    - Cần kiểm tra tính nhất quán trước khi phân tích hiệu năng.
    - Không đưa dữ liệu bệnh nhân hoặc bản ghi EHR vào báo cáo.

Module: data_spark.benchmark
Giai đoạn: Stage 2 — Performance Benchmark
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
    "sum_first",
    "sum_min",
    "sum_max",
    "master",
]


def lab_ids(items: dict) -> dict[int, str]:
    """Bao dam moi itemid chi map den mot lab."""
    mapping = {}
    for name, cfg in items["labs"].items():
        for raw_id in cfg["itemids"]:
            iid = int(raw_id)
            if iid in mapping and mapping[iid] != name:
                raise ValueError(
                    f"itemid {iid} duoc gan cho ca {mapping[iid]} va {name}"
                )
            mapping[iid] = name
    if not mapping:
        raise ValueError("Khong co lab itemids trong configs/feature_items.yaml")
    return mapping


def run_spark(args, cfg, items):
    from pyspark.sql import functions as F

    # get_spark co the su dung HG_SPARK_MASTER; dat ca hai cach va KIEM TRA master thuc te.
    if args.master:
        os.environ["HG_SPARK_MASTER"] = args.master
        os.environ["PYSPARK_SUBMIT_ARGS"] = f"--master {args.master} pyspark-shell"

    from data_spark.common.session import get_spark

    t0 = time.perf_counter()
    spark = get_spark(f"hg-bench-{args.label}")
    try:
        spark.range(1).count()  # Warm up SparkSession; startup duoc bao cao rieng.
        startup = time.perf_counter() - t0
        master = spark.sparkContext.master
        parallelism = spark.sparkContext.defaultParallelism
        print(
            f"[spark] master thuc te = {master}; defaultParallelism = {parallelism}; khoi dong {startup:.1f}s"
        )
        if args.master and master != args.master:
            raise RuntimeError(
                f"--master={args.master} nhung Spark dang dung {master}. "
                "Dung benchmark de tranh bao cao sai cau hinh."
            )

        root = cfg["paths"]["silver"].rstrip("/")
        processed = (
            cfg["paths"]
            .get("processed", "s3a://heartguard-mimic-group11/processed")
            .rstrip("/")
        )
        ids = lab_ids(items)

        # Spark SQL map literal: khong can tao DataFrame Python (tranh Python 3.14 PicklingError),
        # cung khong can chuoi CASE WHEN dai.
        literals = []
        for iid, name in ids.items():
            literals.extend([F.lit(iid), F.lit(name)])
        name_map = F.create_map(*literals)
        lab_col = F.element_at(name_map, F.col("itemid").cast("int"))

        results = []
        for rep in range(1, args.repeats + 1):
            t = time.perf_counter()
            co = (
                spark.read.parquet(args.cohort or f"{processed}/cohort")
                .filter(
                    F.pmod(F.col("subject_id"), F.lit(10000)) < int(args.frac * 10000)
                )
                .select("subject_id", "hadm_id", "admittime")
                .withColumn("pred", F.expr("admittime + INTERVAL 24 HOURS"))
            )
            le = (
                spark.read.parquet(f"{root}/hosp/labevents")
                .filter(F.col("itemid").isin(list(ids)))
                .select("subject_id", "itemid", "charttime", "storetime", "valuenum")
            )
            j = (
                le.join(F.broadcast(co), "subject_id")
                .filter(
                    (F.col("charttime") >= F.expr("admittime - INTERVAL 6 HOURS"))
                    & (F.col("charttime") <= F.col("pred"))
                    & (
                        F.col("storetime").isNull()
                        | (F.col("storetime") <= F.col("pred"))
                    )
                    & F.col("valuenum").isNotNull()
                )
                .withColumn("lab", lab_col)
            )
            res = j.groupBy("hadm_id", "lab").agg(
                F.min(F.struct("charttime", "valuenum"))
                .getField("valuenum")
                .alias("first"),
                F.min("valuenum").alias("min"),
                F.max("valuenum").alias("max"),
            )
            # Count() mot minh co the khong buoc Spark danh gia first/min/max.
            # Mot action tong hop 3 cot de buoc tinh toan gia tri thuc.
            summary = res.agg(
                F.count(F.lit(1)).alias("rows_out"),
                F.sum("first").alias("sum_first"),
                F.sum("min").alias("sum_min"),
                F.sum("max").alias("sum_max"),
            ).first()
            # Tuong tu pandas: n_subjects la so patient duoc lay mau,
            # khong phai so patient co lab hop le.
            n_subjects = co.select("subject_id").distinct().count()
            elapsed = time.perf_counter() - t
            row = dict(
                engine="spark",
                label=args.label,
                frac=args.frac,
                repeat=rep,
                seconds=round(elapsed, 3),
                read_seconds="",  # Spark lazy evaluation -> khong tach duoc read time tu count.
                startup_seconds=round(startup, 3),
                rows_out=int(summary["rows_out"]),
                n_subjects=int(n_subjects),
                sum_first=float(summary["sum_first"] or 0),
                sum_min=float(summary["sum_min"] or 0),
                sum_max=float(summary["sum_max"] or 0),
                master=master,
            )
            results.append(row)
            print(
                f"[spark] lan {rep}: {elapsed:.1f}s, {row['rows_out']:,} dong, "
                f"{row['n_subjects']:,} benh nhan"
            )
        return results
    finally:
        spark.stop()


def run_pandas(args, cfg, items):
    import pandas as pd
    import pyarrow.dataset as ds
    import pyarrow.fs as pafs

    root = cfg["paths"]["silver"].rstrip("/")
    processed = (
        cfg["paths"]
        .get("processed", "s3a://heartguard-mimic-group11/processed")
        .rstrip("/")
    )
    ids = lab_ids(items)

    def open_ds(uri):
        uri = uri.replace("s3a://", "s3://", 1)
        if "://" in uri:
            fs, path = pafs.FileSystem.from_uri(uri)
            return ds.dataset(path, filesystem=fs, format="parquet")
        return ds.dataset(uri, format="parquet")

    results = []
    for rep in range(1, args.repeats + 1):
        t = time.perf_counter()
        co = (
            open_ds(args.cohort or f"{processed}/cohort")
            .to_table(columns=["subject_id", "hadm_id", "admittime"])
            .to_pandas()
        )
        co = co[co["subject_id"] % 10000 < int(args.frac * 10000)].copy()
        co["pred"] = co["admittime"] + pd.Timedelta(hours=24)
        subj = co["subject_id"].unique().tolist()
        le = (
            open_ds(f"{root}/hosp/labevents")
            .to_table(
                columns=["subject_id", "itemid", "charttime", "storetime", "valuenum"],
                filter=ds.field("itemid").isin(list(ids))
                & ds.field("subject_id").isin(subj),
            )
            .to_pandas()
        )
        read_s = time.perf_counter() - t
        j = le.merge(co, on="subject_id")
        j = j[
            (j["charttime"] >= j["admittime"] - pd.Timedelta(hours=6))
            & (j["charttime"] <= j["pred"])
            & (j["storetime"].isna() | (j["storetime"] <= j["pred"]))
            & j["valuenum"].notna()
        ].copy()
        j["lab"] = j["itemid"].map(ids)
        # Spark min(struct(charttime, valuenum)): tie charttime -> chon valuenum nho nhat.
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
        # Dung cung 3 tong de doi chieu gia tri dau ra, khong chi so dong.
        sums = {c: float(res[c].sum()) for c in ("first", "min", "max")}
        elapsed = time.perf_counter() - t
        row = dict(
            engine="pandas",
            label=args.label,
            frac=args.frac,
            repeat=rep,
            seconds=round(elapsed, 3),
            read_seconds=round(read_s, 3),
            startup_seconds="",
            rows_out=len(res),
            n_subjects=len(subj),
            sum_first=sums["first"],
            sum_min=sums["min"],
            sum_max=sums["max"],
            master="pandas",
        )
        results.append(row)
        print(
            f"[pandas] lan {rep}: {elapsed:.1f}s (doc {read_s:.1f}s), "
            f"{row['rows_out']:,} dong, {row['n_subjects']:,} benh nhan"
        )
    return results


def plot(csv_path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(csv_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"CSV khong co du lieu: {csv_path}")
    groups: dict[tuple[str, float], list[float]] = {}
    for r in rows:
        groups.setdefault((r["label"], float(r["frac"])), []).append(
            float(r["seconds"])
        )

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label in sorted({key[0] for key in groups}):
        pts = sorted(
            (frac, statistics.mean(values), statistics.pstdev(values))
            for (group_label, frac), values in groups.items()
            if group_label == label
        )
        ax.errorbar(
            [p[0] * 100 for p in pts],
            [p[1] for p in pts],
            yerr=[p[2] for p in pts],
            marker="o",
            capsize=3,
            label=label,
        )
    ax.set_xlabel("Ty le benh nhan lay mau (%)")
    ax.set_ylabel("Thoi gian chay (giay)")
    ax.set_title("Tong hop xet nghiem 24h: Spark vs pandas")
    ax.grid(alpha=0.3)
    ax.legend()
    out = Path(csv_path).with_suffix(".png")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("[OK] Bieu do ->", out)
    print("| Cau hinh | Mau | Trung binh (s) | Do lech chuan (s) | So lan |")
    print("|---|---:|---:|---:|---:|")
    for (label, frac), values in sorted(groups.items()):
        print(
            f"| {label} | {frac:.0%} | {statistics.mean(values):.1f} | "
            f"{statistics.pstdev(values):.1f} | {len(values)} |"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", choices=["spark", "pandas"])
    ap.add_argument("--frac", type=float, default=0.01)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--master", default=None, help='vd "local[2]" hoac spark://host:7077'
    )
    ap.add_argument("--label", default=None)
    ap.add_argument("--cohort", default=None)
    ap.add_argument("--csv", default=str(REPO_ROOT / "reports/benchmark/bench_v2.csv"))
    ap.add_argument("--plot", default=None, help="Duong dan CSV de ve bieu do")
    args = ap.parse_args()
    if args.plot:
        return plot(args.plot)
    if not args.engine:
        ap.error("Can --engine spark hoac --engine pandas khi chay benchmark")
    if not 0 < args.frac <= 1:
        ap.error("--frac phai nam trong khoang (0, 1]")
    if args.repeats < 1:
        ap.error("--repeats phai >= 1")
    if args.engine == "pandas" and args.master:
        ap.error("--master chi su dung cho Spark")

    args.label = args.label or (args.master or args.engine)
    cfg = load_yaml("configs/data.yaml")
    items = load_yaml("configs/feature_items.yaml")
    rows = (run_spark if args.engine == "spark" else run_pandas)(args, cfg, items)

    p = Path(args.csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    exists = p.exists() and p.stat().st_size > 0
    if exists:
        with p.open(newline="", encoding="utf-8") as f:
            current_header = next(csv.reader(f), [])
        if current_header != FIELDS:
            raise ValueError(
                f"Header CSV khong khop format moi: {p}. "
                "Hay dung mot --csv moi de khong lam hong du lieu cu."
            )
    with p.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    print("[OK] Da ghi", p)


if __name__ == "__main__":
    main()

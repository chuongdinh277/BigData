"""Benchmark khả năng mở rộng của pipeline Spark — bằng chứng cho yêu cầu "distributed pipeline".

Bản benchmark cũ đo một tác vụ nhỏ trên mẫu 1–10% bệnh nhân: thời gian Spark gần như KHÔNG đổi theo
cỡ mẫu (local[1]: 85.7 s → 77.1 s), nghĩa là chỉ đo chi phí cố định, và pandas chạy trên dữ liệu đã lọc
sẵn nên hai bên không cùng việc. Module này đo đúng thứ cần đo:

  sizes    Số dòng + dung lượng Bronze/Silver của 11 bảng (quy mô thật của dữ liệu).
  funnel   Phễu thu gọn labevents: toàn bộ → 20 itemid → bệnh nhân cohort → cửa sổ [−6h, +24h].
  pandas   Thử nạp TOÀN BỘ labevents bằng pandas trên cùng máy: ghi thời gian, RAM đỉnh hoặc lỗi hết bộ nhớ.
  strong   Strong scaling: cùng dữ liệu (toàn cohort), job features P1 với n = 1, 2, 4, 8 core.
           Speedup S(n) = T(1)/T(n), hiệu suất E(n) = S(n)/n, phần tuần tự ước theo Amdahl.
  weak     Weak scaling: dữ liệu tăng theo số core (25% ↔ 2 core, 50% ↔ 4, 100% ↔ 8).
  ablation Bật/tắt từng tối ưu (AQE, broadcast) trên cấu hình lớn nhất.

Mỗi cấu hình chạy --repeats lần (mặc định 3), lấy trung vị. Kết quả: results/benchmark/scaling.{csv,md,png}.

Chạy trên EC2 (Hùng bật máy; khuyến nghị m5.2xlarge 8 vCPU / 32 GB):
    python -m data_spark.scaling funnel
    python -m data_spark.scaling pandas
    python -m data_spark.scaling strong --cores 1 2 4 8
    python -m data_spark.scaling weak   --cores 2 4 8
    python -m data_spark.scaling ablation --cores 8
    python -m data_spark.scaling report
Trên cụm standalone nhiều máy (deploy/spark/): thêm --master spark://<ip-master>:7077, khi đó n = spark.cores.max.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from data_spark.common.io import REPO_ROOT, load_yaml

OUT = REPO_ROOT / "results" / "benchmark"
CSV = OUT / "scaling.csv"
ABLATIONS = {
    "full": "",
    "no_aqe": "spark.sql.adaptive.enabled=false",
    "no_auto_broadcast": "spark.sql.autoBroadcastJoinThreshold=-1",
}


def _append(row: dict) -> None:
    import csv
    OUT.mkdir(parents=True, exist_ok=True)
    new = not CSV.exists()
    with CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "label", "cores", "frac", "repeat", "seconds", "ok", "note"])
        if new:
            w.writeheader()
        w.writerow(row)


def _run_features(master: str, cores: int, frac: float | None, extra_conf: str, out_suffix: str) -> tuple[float, bool, str]:
    cfg = load_yaml("configs/data.yaml")
    gold = cfg["paths"]["gold"].rstrip("/")
    env = dict(os.environ)
    conf = [extra_conf] if extra_conf else []
    if master.startswith("spark://"):
        env["HG_SPARK_MASTER"] = master
        conf.append(f"spark.cores.max={cores}")
    else:
        env["HG_SPARK_MASTER"] = f"local[{cores}]"
    env["HG_SPARK_CONF"] = ",".join(c for c in conf if c)
    env["HG_SHUFFLE_PARTITIONS"] = str(max(4, 3 * cores))  # 2–3 × số core, AQE gộp tiếp
    cmd = [sys.executable, "-m", "data_spark.features", "--skip-verify", "--out", f"{gold}/p1/_bench_{out_suffix}",
           "--report", str(OUT / f"_features_{out_suffix}.md")]
    if frac:
        cmd += ["--sample-frac", str(frac)]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    return time.perf_counter() - t0, r.returncode == 0, (r.stderr or "")[-300:].replace("\n", " ")


def cmd_strong(a) -> None:
    for n in a.cores:
        for k in range(a.repeats):
            s, ok, err = _run_features(a.master, n, None, "", f"strong{n}")
            _append({"mode": "strong", "label": f"{n} core", "cores": n, "frac": 1.0, "repeat": k, "seconds": round(s, 1),
                     "ok": ok, "note": "" if ok else err})
            print(f"strong n={n} lần {k + 1}: {s:.1f}s {'OK' if ok else 'LỖI'}")


def cmd_weak(a) -> None:
    top = max(a.cores)
    for n in a.cores:
        frac = n / top
        for k in range(a.repeats):
            s, ok, err = _run_features(a.master, n, frac if frac < 1 else None, "", f"weak{n}")
            _append({"mode": "weak", "label": f"{n} core / {frac:.0%}", "cores": n, "frac": frac, "repeat": k,
                     "seconds": round(s, 1), "ok": ok, "note": "" if ok else err})
            print(f"weak n={n} frac={frac:.2f} lần {k + 1}: {s:.1f}s")


def cmd_ablation(a) -> None:
    n = max(a.cores)
    for name, conf in ABLATIONS.items():
        for k in range(a.repeats):
            s, ok, err = _run_features(a.master, n, None, conf, f"abl_{name}")
            _append({"mode": "ablation", "label": name, "cores": n, "frac": 1.0, "repeat": k, "seconds": round(s, 1),
                     "ok": ok, "note": "" if ok else err})
            print(f"ablation {name} lần {k + 1}: {s:.1f}s")


def cmd_funnel(_a) -> None:
    """Đếm số dòng labevents qua từng bước lọc — hình phễu cho báo cáo."""
    from pyspark.sql import functions as F

    from data_spark.common.io import read_table
    from data_spark.common.session import get_spark

    cfg, items = load_yaml("configs/data.yaml"), load_yaml("configs/feature_items.yaml")
    spark = get_spark("hg-funnel")
    labs = read_table(spark, cfg["paths"]["silver"], "labevents", "silver")
    ids = [int(i) for c in items["labs"].values() for i in c["itemids"]]
    cohort = spark.read.parquet(f"{cfg['paths']['processed'].rstrip('/')}/cohort")
    steps = {}
    t0 = time.perf_counter()
    steps["labevents toàn bộ"] = labs.count()
    f1 = labs.filter(F.col("itemid").isin(ids))
    steps["20 xét nghiệm dùng làm đặc trưng"] = f1.count()
    subj = F.broadcast(cohort.select("subject_id").distinct())
    f2 = f1.join(subj, "subject_id")
    steps["thuộc bệnh nhân cohort suy tim"] = f2.count()
    win = F.broadcast(cohort.select("subject_id", "hadm_id", "admittime"))
    f3 = f1.join(win, "subject_id").filter(
        (F.col("charttime") >= F.expr("admittime - INTERVAL 6 HOURS")) & (F.col("charttime") <= F.expr("admittime + INTERVAL 24 HOURS")))
    steps["trong cửa sổ [−6h, +24h] của lượt nhập viện"] = f3.count()
    res = {"steps": steps, "seconds": round(time.perf_counter() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "funnel.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in steps.items():
        print(f"{v:>14,}  {k}")
    spark.stop()


def cmd_sizes(_a) -> None:
    """Quy mô dữ liệu thật: số dòng (Spark count trên Silver) + dung lượng nén Bronze (csv.gz) và Silver
    (Parquet) của từng bảng dùng trong dự án. Đây là con số trả lời "vì sao đây là bài toán Big Data"."""
    from data_spark.common.io import TABLE_SPECS
    from data_spark.common.session import get_spark

    cfg = load_yaml("configs/data.yaml")
    spark = get_spark("hg-sizes")
    jvm, conf = spark._jvm, spark._jsc.hadoopConfiguration()

    def nbytes(uri: str) -> int | None:
        try:
            path = jvm.org.apache.hadoop.fs.Path(uri)
            return int(path.getFileSystem(conf).getContentSummary(path).getLength())
        except Exception:  # noqa: BLE001
            return None

    rows = []
    for table, spec in TABLE_SPECS.items():
        silver = f"{cfg['paths']['silver'].rstrip('/')}/{spec['module']}/{table}"
        bronze = f"{cfg['paths']['bronze'].rstrip('/')}/{spec['module']}/{table}.csv.gz"
        t0 = time.perf_counter()
        n = spark.read.parquet(silver).count()
        rows.append({"table": table, "module": spec["module"], "rows": n, "bronze_gz_bytes": nbytes(bronze),
                     "silver_parquet_bytes": nbytes(silver), "count_seconds": round(time.perf_counter() - t0, 1)})
        print(f"{table:18} {n:>14,} dòng")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table_sizes.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    spark.stop()


def cmd_pandas(_a) -> None:
    """Thử nạp toàn bộ labevents (6 cột) bằng pandas trên cùng máy. Chạy trong tiến trình con để
    nếu hết bộ nhớ thì chỉ tiến trình con chết; ghi thời gian, RAM đỉnh hoặc lý do thất bại."""
    cfg = load_yaml("configs/data.yaml")
    path = f"{cfg['paths']['bronze'].rstrip('/').replace('s3a://', 's3://')}/hosp/labevents.csv.gz"
    code = (
        "import pandas as pd, time, resource, json, sys\n"
        "t=time.perf_counter()\n"
        f"df=pd.read_csv({path!r}, usecols=['subject_id','hadm_id','itemid','charttime','valuenum','valueuom'])\n"
        "print(json.dumps({'rows': len(df), 'seconds': time.perf_counter()-t,"
        " 'peak_rss_gb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024}))\n"
    )
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    res = {"engine": "pandas", "source": path, "returncode": r.returncode, "wall_seconds": round(time.perf_counter() - t0, 1)}
    if r.returncode == 0:
        res.update(json.loads(r.stdout.strip().splitlines()[-1]))
    else:
        res["failure"] = "bị hệ điều hành dừng (hết bộ nhớ, mã -9/137)" if r.returncode in (-9, 137) else (r.stderr or "")[-400:]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pandas_full_load.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


def _median_table():
    import pandas as pd
    df = pd.read_csv(CSV)
    df = df[df["ok"]]
    return df.groupby(["mode", "label", "cores", "frac"], as_index=False)["seconds"].agg(["median", "std", "count"]).reset_index()


def cmd_report(_a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = _median_table()
    L = ["# Benchmark khả năng mở rộng của pipeline Spark", "",
         "Job đo: `data_spark.features` (Gold P1, toàn cohort) — đọc Silver Parquet trên S3, lọc, join, tổng hợp cửa sổ thời gian.",
         "Mỗi cấu hình chạy nhiều lần, báo trung vị. Nhiều core trên một máy KHÔNG tương đương nhiều máy (chung đĩa, mạng tới S3).", ""]
    st = t[t["mode"] == "strong"].sort_values("cores")
    if len(st):
        t1 = float(st.iloc[0]["median"]) * float(st.iloc[0]["cores"])  # chuẩn hóa nếu không có n=1
        L += ["## Strong scaling (cùng dữ liệu, tăng core)", "", "| Core | Trung vị (s) | Speedup S(n) | Hiệu suất E(n) | Lần chạy |", "|---:|---:|---:|---:|---:|"]
        base = float(st.iloc[0]["median"]) if int(st.iloc[0]["cores"]) == 1 else t1
        for r in st.itertuples():
            s = base / r.median
            L.append(f"| {r.cores} | {r.median:.1f} | {s:.2f} | {s / r.cores:.0%} | {r.count} |")
        if len(st) >= 2:
            n, s = float(st.iloc[-1]["cores"]), base / float(st.iloc[-1]["median"])
            serial = (n / s - 1) / (n - 1) if n > 1 else float("nan")
            L += ["", f"Theo định luật Amdahl, S(n) = 1 / (f + (1 − f)/n) → phần tuần tự ước tính f ≈ **{serial:.0%}** "
                      "(đọc file, lập kế hoạch, ghi kết quả)."]
    wk = t[t["mode"] == "weak"].sort_values("cores")
    if len(wk):
        L += ["", "## Weak scaling (dữ liệu tăng theo số core)", "", "| Cấu hình | Trung vị (s) | So với cấu hình nhỏ nhất |", "|---|---:|---:|"]
        b = float(wk.iloc[0]["median"])
        L += [f"| {r.label} | {r.median:.1f} | {r.median / b:.2f}× |" for r in wk.itertuples()]
    ab = t[t["mode"] == "ablation"]
    if len(ab):
        full = float(ab[ab["label"] == "full"]["median"].iloc[0]) if (ab["label"] == "full").any() else None
        L += ["", "## Ablation tối ưu", "", "| Cấu hình | Trung vị (s) | Chậm hơn bản đầy đủ |", "|---|---:|---:|"]
        L += [f"| {r.label} | {r.median:.1f} | {(r.median / full - 1):+.0%} |" if full else f"| {r.label} | {r.median:.1f} | – |" for r in ab.itertuples()]
    for name, title in (("table_sizes.json", "Quy mô từng bảng"), ("funnel.json", "Phễu thu gọn labevents"),
                        ("pandas_full_load.json", "pandas nạp toàn bộ labevents")):
        p = OUT / name
        if p.exists():
            L += ["", f"## {title}", "", "```json", p.read_text(encoding="utf-8"), "```"]
    (OUT / "scaling.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    if len(st) >= 2:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        ax[0].plot(st["cores"], st["median"], "o-")
        ax[0].set_xlabel("Số core")
        ax[0].set_ylabel("Thời gian (s)")
        ax[0].set_title("Strong scaling")
        sp = base / st["median"]
        ax[1].plot(st["cores"], sp, "o-", label="Thực tế")
        ax[1].plot(st["cores"], st["cores"] / st["cores"].iloc[0], "k:", label="Tuyến tính")
        ax[1].set_xlabel("Số core")
        ax[1].set_ylabel("Speedup")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(OUT / "scaling.png", dpi=150)
    print(f"Đã ghi {OUT / 'scaling.md'}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["sizes", "funnel", "pandas", "strong", "weak", "ablation", "report"])
    ap.add_argument("--cores", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--master", default="local", help="local (mặc định) hoặc spark://<ip>:7077")
    a = ap.parse_args(argv)
    {"sizes": cmd_sizes, "funnel": cmd_funnel, "pandas": cmd_pandas, "strong": cmd_strong, "weak": cmd_weak,
     "ablation": cmd_ablation, "report": cmd_report}[a.mode](a)


if __name__ == "__main__":
    main()

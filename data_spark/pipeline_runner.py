"""Pipeline runner — ghép các module GĐ2 thành Dataset V1 (Gold P1).

Thứ tự chạy (mỗi bước là một tiến trình riêng, dùng đúng CLI của từng module):

    cohort   (Hùng)   python -m data_spark.cohort        → <processed>/cohort
    labels   (Hà)     python -m data_spark.labels        → <processed>/labels/p1   (nếu module đã có)
    features (Khoẻ)   python -m data_spark.features      → <gold>/p1/<version>
    validate (Chưởng) đọc Gold + cohort, chạy quality (Chiến) + kiểm tra chéo
                      → manifest.json (S3, cạnh Gold) + docs/stage2/dataset_v1.md

Nhãn P1: nếu có parquet nhãn của Hà (--labels) thì features dùng nhãn đó; nếu không,
features tạm dùng hospital_expire_flag và manifest ghi rõ `label_source` là tạm thời.

Ví dụ trên EC2 (dữ liệu thật, cohort đã có sẵn trên S3):
    python -m data_spark.pipeline_runner --steps features validate --version v1
Chạy đủ, có nhãn của Hà:
    python -m data_spark.pipeline_runner --steps cohort labels features validate \
        --labels s3a://heartguard-mimic-group11/processed/labels/p1 --label-col label_mortality
Thử nhanh trên mẫu 1% bệnh nhân (ghi ra gold/p1/_sample_0.01, không đụng v1):
    python -m data_spark.pipeline_runner --steps features validate --sample-frac 0.01
Chỉ in lệnh sẽ chạy, không chạy gì:
    python -m data_spark.pipeline_runner --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from data_spark.common.io import REPO_ROOT, load_yaml

ALL_STEPS = ["cohort", "labels", "features", "validate"]
DEFAULT_STEPS = ["cohort", "labels", "features", "validate"]


# --------------------------------------------------------------------------- cấu hình & đường dẫn
def resolve_paths(cfg: dict, version: str, sample_frac: float | None,
                  cohort: str | None = None, out: str | None = None) -> dict:
    p = cfg["paths"]
    processed = p.get("processed", "s3a://heartguard-mimic-group11/processed").rstrip("/")
    gold = p.get("gold", "s3a://heartguard-mimic-group11/gold").rstrip("/")
    reports = p.get("reports", "s3a://heartguard-mimic-group11/reports").rstrip("/")
    gold_p1 = out or (f"{gold}/p1/_sample_{sample_frac}" if sample_frac else f"{gold}/p1/{version}")
    return {
        "silver": p["silver"].rstrip("/"),
        "cohort": cohort or f"{processed}/cohort",
        "labels_default": f"{processed}/labels/p1",
        "gold_p1": gold_p1,
        "manifest": f"{gold_p1.rstrip('/')}/_manifest.json",
        "quality_out": f"{reports}/quality/gold_p1_{'sample' if sample_frac else version}",
    }


def module_exists(name: str) -> bool:
    return (REPO_ROOT / "data_spark" / f"{name}.py").exists()


def build_commands(steps: list[str], paths: dict, *, version: str, fmt: str = "silver",
                   labels: str | None = None, label_col: str = "label_mortality",
                   sample_frac: float | None = None, skip_verify: bool = False,
                   no_storetime_filter: bool = False,
                   feature_report: str | None = None) -> list[tuple[str, list[str]]]:
    """Trả về danh sách (tên bước, lệnh) cho các bước chạy bằng tiến trình riêng."""
    py = sys.executable
    cmds: list[tuple[str, list[str]]] = []
    if "cohort" in steps:
        cmds.append(("cohort", [py, "-m", "data_spark.cohort", "--fmt", fmt, "--out", paths["cohort"]]))
    if "labels" in steps:
        if module_exists("labels"):
            cmds.append(("labels", [py, "-m", "data_spark.labels"]))
        else:
            cmds.append(("labels", []))  # rỗng = bỏ qua, chưa có module của Hà
    if "features" in steps:
        c = [py, "-m", "data_spark.features", "--fmt", fmt, "--version", version,
             "--cohort", paths["cohort"], "--out", paths["gold_p1"]]
        if labels:
            c += ["--labels", labels, "--label-col", label_col]
        if sample_frac:
            c += ["--sample-frac", str(sample_frac)]
        if skip_verify:
            c.append("--skip-verify")
        if no_storetime_filter:
            c.append("--no-storetime-filter")
        if feature_report:
            c += ["--report", feature_report]
        cmds.append(("features", c))
    return cmds


def run_step(name: str, cmd: list[str], dry_run: bool) -> dict:
    if not cmd:
        print(f"[{name}] BỎ QUA — chưa có data_spark/{name}.py (nhánh của người phụ trách chưa merge)")
        return {"step": name, "status": "SKIPPED", "seconds": 0.0}
    print(f"[{name}] $ {' '.join(cmd[1:])}")
    if dry_run:
        return {"step": name, "status": "DRY_RUN", "seconds": 0.0}
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
    sec = round(time.time() - t0, 1)
    status = "OK" if rc == 0 else f"FAILED(exit {rc})"
    print(f"[{name}] {status} sau {sec}s")
    return {"step": name, "status": status, "seconds": sec}


# --------------------------------------------------------------------------- kiểm tra chéo Dataset V1
def cross_checks(summary: dict, cohort_p1_rows: int, expected: dict | None = None) -> list[dict]:
    """So khớp Gold với cohort và (tuỳ chọn) số liệu kỳ vọng. Thuần Python để test được."""
    checks = [
        {"check": "hadm_id duy nhất", "ok": summary["rows"] == summary["distinct_hadm"],
         "detail": f"{summary['rows']} dòng / {summary['distinct_hadm']} hadm_id"},
        {"check": "Không có nhãn NULL", "ok": summary["label_nulls"] == 0,
         "detail": f"{summary['label_nulls']} nhãn NULL"},
        {"check": "Nhãn chỉ 0/1", "ok": set(summary["label_values"]) <= {0, 1},
         "detail": f"giá trị: {sorted(summary['label_values'])}"},
        {"check": "Không có cột bị cấm", "ok": not summary["forbidden_present"],
         "detail": ", ".join(summary["forbidden_present"]) or "—"},
        {"check": "Thiếu cột so với contract", "ok": not summary["missing_contract_cols"],
         "detail": ", ".join(summary["missing_contract_cols"][:10]) or "—"},
        {"check": "prediction_time = admittime + 24h", "ok": summary["bad_prediction_time"] == 0,
         "detail": f"{summary['bad_prediction_time']} dòng lệch"},
        {"check": "Số dòng Gold = số ca P1 tính lại từ cohort", "ok": summary["rows"] == cohort_p1_rows,
         "detail": f"Gold {summary['rows']} · cohort P1 {cohort_p1_rows}"},
    ]
    for key, val in (expected or {}).items():
        if val is None:
            continue
        got = summary.get(key)
        checks.append({"check": f"{key} = {val} (kỳ vọng)", "ok": got == val, "detail": f"thực tế {got}"})
    return checks


def mask(n, threshold: int):
    return f"<{threshold}" if isinstance(n, int) and 0 < n < threshold else n


def render_markdown(manifest: dict, threshold: int = 10) -> str:
    s = manifest["summary"]
    lines = [
        f"# Dataset {manifest['data_version']} — Gold P1 (tử vong trong viện, mốc +24h)",
        "",
        f"Tạo lúc: {manifest['created_at']} · Commit: `{manifest['git_commit']}` · "
        f"Trạng thái: **{manifest['status']}**",
        "",
        f"- Vị trí: `{manifest['paths']['gold_p1']}`",
        f"- Nguồn nhãn: {manifest['label_source']}",
        f"- Số dòng (lần nhập viện): {s['rows']:,} · Số bệnh nhân: {s['patients']:,} · Số cột: {s['n_columns']}",
        f"- Ca dương: {s['positives']:,} ({s['positive_rate']:.2%})",
        "",
        "## Kiểm tra chéo",
        "",
        "| Kiểm tra | Kết quả | Chi tiết |",
        "|---|---|---|",
    ]
    lines += [f"| {c['check']} | {'✅' if c['ok'] else '❌'} | {c['detail']} |" for c in manifest["cross_checks"]]
    lines += ["", f"Quality (Chiến): **{manifest['quality_status']}**"
              + (f" — {manifest['quality_note']}" if manifest.get("quality_note") else ""), "",
              "## Theo anchor_year_group (cho drift)", "",
              "| anchor_year_group | n | Ca dương | Tỷ lệ |", "|---|---:|---:|---:|"]
    for r in s["by_year_group"]:
        rate = f"{r['positives'] / r['n']:.2%}" if r["n"] else "—"
        lines.append(f"| {r['group']} | {mask(r['n'], threshold)} | {mask(r['positives'], threshold)} | {rate} |")
    lines += ["", "## Thời gian từng bước", "", "| Bước | Trạng thái | Giây |", "|---|---|---:|"]
    lines += [f"| {r['step']} | {r['status']} | {r['seconds']} |" for r in manifest["steps"]]
    lines += ["", "Chỉ chứa số liệu tổng hợp; ô < 10 đã được che.", ""]
    return "\n".join(lines)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def validate(spark, paths: dict, schema: dict, index_hours: int) -> tuple[dict, int, dict]:
    """Đọc Gold + cohort bằng Spark, trả về (summary, số ca P1 từ cohort, báo cáo quality)."""
    from pyspark.sql import functions as F
    from data_spark import quality

    gold = spark.read.parquet(paths["gold_p1"]).cache()
    cohort = spark.read.parquet(paths["cohort"])
    label = schema.get("label", "label_mortality")

    agg = gold.agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct("hadm_id").alias("distinct_hadm"),
        F.countDistinct("subject_id").alias("patients"),
        F.sum(F.col(label)).alias("positives"),
        F.sum(F.col(label).isNull().cast("int")).alias("label_nulls"),
        F.sum((F.col("prediction_time") != F.col("admittime") + F.expr(f"INTERVAL {index_hours} HOURS"))
              .cast("int")).alias("bad_prediction_time"),
    ).first()
    label_values = [r[0] for r in gold.select(label).distinct().collect() if r[0] is not None]
    by_year = [{"group": r["anchor_year_group"], "n": int(r["n"]), "positives": int(r["pos"] or 0)}
               for r in gold.groupBy("anchor_year_group")
               .agg(F.count(F.lit(1)).alias("n"), F.sum(label).alias("pos"))
               .orderBy("anchor_year_group").collect()]

    contract_cols = set(quality.feature_columns()) | {c["name"] for sec in
                    ("id_columns", "label_columns", "subgroup_only_columns") for c in schema.get(sec, [])}
    forbidden = set(schema.get("forbidden_columns", []))
    rows = int(agg["rows"])
    summary = {
        "rows": rows, "distinct_hadm": int(agg["distinct_hadm"]), "patients": int(agg["patients"]),
        "positives": int(agg["positives"] or 0),
        "positive_rate": (int(agg["positives"] or 0) / rows) if rows else 0.0,
        "label_nulls": int(agg["label_nulls"] or 0), "label_values": [int(v) for v in label_values],
        "bad_prediction_time": int(agg["bad_prediction_time"] or 0),
        "n_columns": len(gold.columns),
        "forbidden_present": sorted(forbidden & set(gold.columns)),
        "missing_contract_cols": sorted(contract_cols - set(gold.columns)),
        "by_year_group": by_year,
    }

    pt = F.col("admittime") + F.expr(f"INTERVAL {index_hours} HOURS")
    cohort_p1 = cohort.filter((F.col("admittime") < pt) & (pt < F.col("dischtime"))
                              & (F.col("deathtime").isNull() | (F.col("deathtime") > pt))).count()

    qrep = quality.run_quality(gold, context=cohort, lineage=None, p1=True)
    return summary, int(cohort_p1), qrep


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    cfg = load_yaml("configs/data.yaml")
    ap = argparse.ArgumentParser(description="Ghép cohort → labels → features → validate thành Dataset V1")
    ap.add_argument("--steps", nargs="+", choices=ALL_STEPS, default=DEFAULT_STEPS)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--fmt", choices=["silver", "bronze"], default="silver")
    ap.add_argument("--cohort", default=None, help="Mặc định <processed>/cohort")
    ap.add_argument("--out", default=None, help="Mặc định <gold>/p1/<version>")
    ap.add_argument("--labels", default=None,
                    help="Parquet nhãn P1 của Hà (hadm_id + --label-col). 'auto' = <processed>/labels/p1")
    ap.add_argument("--label-col", default="label_mortality")
    ap.add_argument("--sample-frac", type=float, default=None)
    ap.add_argument("--skip-verify", action="store_true", help="truyền cho features: bỏ đối chiếu itemid")
    ap.add_argument("--no-storetime-filter", action="store_true")
    ap.add_argument("--expect-rows", type=int, default=None, help="Số dòng kỳ vọng, vd 73768")
    ap.add_argument("--expect-positives", type=int, default=None, help="Số ca dương kỳ vọng, vd 3588")
    ap.add_argument("--report", default=str(REPO_ROOT / "docs/stage2/dataset_v1.md"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    paths = resolve_paths(cfg, args.version, args.sample_frac, args.cohort, args.out)
    labels = paths["labels_default"] if args.labels == "auto" else args.labels
    feature_report = str(REPO_ROOT / ("docs/stage2/features_result_sample.md" if args.sample_frac
                                      else "docs/stage2/features_result_full.md"))
    steps_log: list[dict] = []
    t_all = time.time()

    for name, cmd in build_commands(args.steps, paths, version=args.version, fmt=args.fmt,
                                    labels=labels, label_col=args.label_col, sample_frac=args.sample_frac,
                                    skip_verify=args.skip_verify,
                                    no_storetime_filter=args.no_storetime_filter,
                                    feature_report=feature_report):
        r = run_step(name, cmd, args.dry_run)
        steps_log.append(r)
        if r["status"].startswith("FAILED"):
            print(f"[DỪNG] Bước {name} lỗi — người phụ trách module sửa rồi chạy lại từ bước này.")
            return 1

    if "validate" not in args.steps:
        return 0
    if args.dry_run:
        print(f"[validate] sẽ đọc {paths['gold_p1']} + {paths['cohort']}, ghi {paths['manifest']}")
        return 0

    from data_spark.common.session import get_spark
    from data_spark import quality

    schema = json.loads((REPO_ROOT / "contracts/gold_schema_p1.json").read_text(encoding="utf-8"))
    index_hours = int(cfg["prediction"]["p1_index_hours"])
    t0 = time.time()
    spark = get_spark("hg-pipeline-validate")
    try:
        summary, cohort_p1, qrep = validate(spark, paths, schema, index_hours)
        checks = cross_checks(summary, cohort_p1, {"rows": args.expect_rows, "positives": args.expect_positives})
        steps_log.append({"step": "validate", "status": "OK", "seconds": round(time.time() - t0, 1)})

        q_status = qrep["status"]
        q_note = ""
        if q_status == "NOT_CHECKED":
            q_note = "chưa có lineage của features nên phần leakage theo nguồn chưa kiểm được"
        threshold = int(cfg["privacy"]["small_cell_threshold"])
        qrep["data_source"] = "real_or_unspecified"
        quality.write_report(quality.redact_counts(qrep, threshold), paths["quality_out"], spark)

        all_ok = all(c["ok"] for c in checks) and q_status != "FAILED"
        manifest = {
            "dataset": "heartguard_gold_p1",
            "data_version": args.version if not args.sample_frac else f"sample_{args.sample_frac}",
            "task": "P1",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "source": {"mimic_version": cfg["dataset"]["version"], "silver": paths["silver"]},
            "paths": paths,
            "label_source": (f"nhãn của Hà: {labels} (cột {args.label_col})" if labels
                             else "hospital_expire_flag (TẠM — chờ module labels của Hà)"),
            "schema_version": schema.get("schema_version"),
            "summary": summary,
            "cross_checks": checks,
            "quality_status": q_status,
            "quality_note": q_note,
            "quality_report": paths["quality_out"],
            "steps": steps_log,
            "total_seconds": round(time.time() - t_all, 1),
            "status": "PASSED" if all_ok else "FAILED",
        }
        quality_writer_payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        _write_text(spark, paths["manifest"], quality_writer_payload)
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_markdown(manifest, threshold), encoding="utf-8")
        print(f"[{manifest['status']}] {summary['rows']:,} dòng · {summary['positives']:,} ca dương "
              f"({summary['positive_rate']:.2%}) → {paths['gold_p1']}")
        print(f"[OK] manifest → {paths['manifest']} · báo cáo → {report_path}")
        return 0 if all_ok else 1
    finally:
        spark.stop()


def _write_text(spark, path: str, payload: str) -> None:
    if path.startswith("s3a://"):
        jpath = spark._jvm.org.apache.hadoop.fs.Path(path)
        fs = jpath.getFileSystem(spark._jsc.hadoopConfiguration())
        stream = fs.create(jpath, True)
        try:
            stream.write(bytearray(payload.encode("utf-8")))
        finally:
            stream.close()
    else:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

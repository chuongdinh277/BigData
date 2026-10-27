"""Chạy toàn bộ demo synthetic, xác nhận ca sai bị chặn, xuất báo cáo tổng hợp.

Dùng trong CI và local: HG_LOCAL=1 python tools/verify_quality_demo.py
Không đọc dữ liệu thật hoặc ghi S3.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/quality"


def run(*arguments, expected=0):
    completed = subprocess.run([sys.executable, *arguments], cwd=ROOT, env=os.environ.copy())
    if completed.returncode != expected:
        raise RuntimeError(f"{arguments}: exit {completed.returncode}, expected {expected}")


def read(name):
    return json.loads((OUT / name / "quality_report.json").read_text())


def main():
    os.environ["HG_LOCAL"] = "1"
    os.environ.setdefault("HG_SPARK_MASTER", "local[2]")
    os.environ.setdefault("HG_SHUFFLE_PARTITIONS", "4")
    os.environ.setdefault("HG_DRIVER_MEMORY", "2g")
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ["PYSPARK_PYTHON"] = sys.executable
    OUT.mkdir(parents=True, exist_ok=True)
    run("tools/make_synthetic_mimic.py", "--patients", "800", "--seed", "42")
    run("-m", "data_spark.stage1_feasibility", "--root", "sample_data/mimic_fake", "--out", "reports/stage1_fake", "--with-chartevents")
    for kind, table in (("labs", "hosp/labevents"), ("vitals", "icu/chartevents")):
        run("-m", "data_spark.quality", "--kind", kind, "--format", "csv", "--table",
            f"sample_data/mimic_fake/{table}.csv.gz", "--out", f"reports/quality/{kind}_units", "--synthetic")
    for bad in (False, True):
        suffix = "_bad" if bad else ""
        fixture = f"sample_data/quality_fixture{suffix}"
        run("tools/make_quality_fixture.py", "--out", fixture, *(["--inject-errors"] if bad else []))
        run("-m", "data_spark.quality", "--table", f"{fixture}/features.parquet", "--context", f"{fixture}/context.parquet",
            "--lineage", f"{fixture}/lineage.parquet", "--out", f"reports/quality/p1_fake{suffix}", "--synthetic", expected=1 if bad else 0)
    clean, bad = read("p1_fake"), read("p1_fake_bad")
    assert clean["status"] == "PASSED"
    assert bad["status"] == "FAILED"
    assert bad["checks"]["profile"]["duplicates"]["duplicate_rows"] > 0
    assert any(c["invalid_count"] for c in bad["checks"]["profile"]["columns"])
    for metric in ("future_charttime_rows", "late_storetime_rows"):
        assert bad["checks"]["leakage"][metric] > 0
    feasibility = json.loads((ROOT / "reports/stage1_fake/stage1_feasibility.json").read_text())
    cfg = yaml.safe_load((ROOT / "configs/feature_items.yaml").read_text())
    candidates = {(c["feature"], c["itemid"]) for c in feasibility["lab_item_candidates"]}
    assert all((name, item) in candidates for name, spec in cfg["labs"].items() for item in spec["itemids"])
    vital_candidates = {(c["feature"], c["itemid"]) for c in feasibility["vital_item_candidates"]}
    assert all((name, item) in vital_candidates for name, spec in cfg["vitals"].items() for item in spec["itemids"])
    test_xml = OUT / "pytest.xml"
    tests = None
    if test_xml.exists():
        suites = ElementTree.parse(test_xml).getroot().iter("testsuite")
        tests = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
        for suite in suites:
            for key in tests:
                tests[key] += int(suite.attrib.get(key, 0))
        assert tests["failures"] == tests["errors"] == tests["skipped"] == 0
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    summary = {"commit": sha, "execution": "GitHub Actions" if os.environ.get("GITHUB_ACTIONS") else "local",
               "python": platform.python_version(), "spark": feasibility["spark"], "tests": tests,
               "clean_status": clean["status"], "bad_status": bad["status"], "real_data_status": "NOT_RUN"}
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    missing = clean["checks"]["profile"]
    lab_units, vital_units = read("labs_units"), read("vitals_units")
    actual_patients = len(pd.read_csv(ROOT / "sample_data/mimic_fake/hosp/patients.csv.gz"))
    lines = ["# Báo cáo chất lượng Giai đoạn 2 — Chiến", "", "Ngày: 27/10/2026 · Nhánh: `chien/quality`", "",
             "**Phạm vi: synthetic. Dữ liệu MIMIC thật và ghi S3 chưa chạy.**", "",
             f"Nguồn kiểm chứng: {summary['execution']}, commit `{sha}`; Python {summary['python']}, Spark {summary['spark']['version']}.", "",
             "## Kết quả thực thi", "", "| Hạng mục | Kết quả |", "|---|---|",
             f"| Sinh MIMIC giả, seed 42 | {actual_patients} bệnh nhân, 10 bảng CSV gzip |",
             f"| Khảo sát stage1, gồm chartevents | {feasibility['lab_item_candidates_count']} mã lab ứng viên; chạy thành công |",
             f"| Kiểm tra đơn vị | Lab: {lab_units['status']}; vital: {vital_units['status']} |",
             f"| Fixture P1 từ các event giả đã lọc | {missing['rows']} dòng × {len(missing['columns'])} cột; {clean['status']} |",
             f"| Trùng khóa ở fixture sạch | {missing['duplicates']['duplicate_rows']} dòng |",
             f"| Giá trị ngoài khoảng ở fixture sạch | {sum(c['invalid_count'] or 0 for c in missing['columns'])} |",
             f"| Thời gian và leakage fixture sạch | {clean['checks']['prediction_times']['status']} / {clean['checks']['leakage']['status']} |",
             f"| Fixture cố tình sai | {bad['status']}; CLI trả exit code 1 đúng dự kiến |"]
    if tests:
        lines.append(f"| Pytest | {tests['tests']} tests; {tests['failures']} failures, {tests['errors']} errors, {tests['skipped']} skipped |")
    lines += ["", "## Các lỗi cố tình đưa vào và đã phát hiện", "", "| Kiểm tra | Số phát hiện |", "|---|---:|",
              f"| Dòng nằm trong nhóm trùng hadm_id | {bad['checks']['profile']['duplicates']['duplicate_rows']} |",
              f"| sodium_first ngoài khoảng | {next(c['invalid_count'] for c in bad['checks']['profile']['columns'] if c['column']=='sodium_first')} |",
              f"| Event sau mốc dự đoán | {bad['checks']['leakage']['future_charttime_rows']} |",
              f"| Kết quả có sẵn sau mốc dự đoán | {bad['checks']['leakage']['late_storetime_rows']} |",
              "", "Các số đếm leakage có thể tăng do fixture cố tình nhân đôi admission.", "",
              "## Mã xét nghiệm, đơn vị và độ phủ giả", "",
              "Độ phủ dưới đây thuộc khảo sát synthetic, trước áp dụng bộ lọc chặt storetime/range của fixture quality.", "",
              "| Feature | itemid | Đơn vị | Khoảng hợp lệ | Độ phủ P1 giả |", "|---|---|---|---|---:|"]
    coverage = {r["feature"]: r["coverage"] for r in feasibility["P1_lab_coverage"]}
    for name, spec in cfg["labs"].items():
        lines.append(f"| {name} | {spec['itemids']} | {spec['unit_hint'] or 'không có'} | {spec['valid_range']} | {coverage.get(name)} |")
    lines += ["", "## Chất lượng event đầu vào giả", "",
              "Đây là thống kê trước lọc. Các giá trị ngoài khoảng được loại khi tạo fixture; không được coi là giá trị hợp lệ chỉ vì đơn vị đã đạt.", "",
              "| Nhóm/feature | Số event | Sai đơn vị | Thiếu numeric | Numeric-text | Ngoài khoảng đúng đơn vị |", "|---|---:|---:|---:|---:|---:|"]
    for group in (lab_units, vital_units):
        for r in group["checks"]["units"]["items"]:
            lines.append(f"| {r['feature']} | {r['rows']} | {r['wrong_unit_rows']} | {r['missing_numeric_rows']} | {r['numeric_text_rows']} | {r['invalid_value_rows']} |")
    lines += ["", "## Missing rate từng cột của fixture sạch", "",
              "Thiếu xét nghiệm/sinh hiệu được giữ NULL, không tự impute. Cột ngoài phạm vi khoảng sinh lý ghi N/A.", "",
              "| Cột | Số thiếu | % thiếu | Ngoài khoảng |", "|---|---:|---:|---:|"]
    for c in missing["columns"]:
        lines.append(f"| {c['column']} | {c['missing_count']} | {c['missing_pct']} | {c['invalid_count'] if c['invalid_count'] is not None else 'N/A'} |")
    lines += ["", "## Bàn giao và phần còn cần dữ liệu thật", "",
              "- Cấu hình 20 lab + 7 nhóm vital dùng được để code song song; trạng thái `synthetic_verified_pending_real`.",
              "- Fixture chỉ là tập con feature P1 để kiểm chứng quality, không phải cohort HF hoặc Dataset V1 của nhóm.",
              "- Chờ khảo sát dictionary/đơn vị/độ phủ thật để chốt config thật; chờ cohort và feature/lineage của Hùng, Khoẻ để chạy lại quality.",
              "- Chưa ghi báo cáo lên S3 và chưa đánh dấu Dataset V1 đã chốt.",
              "- Missing theo từng nhóm năm và chi tiết từng phép kiểm có trong JSON tổng hợp ở artifact `quality-evidence`.",
              "- Lệnh tái lập và hợp đồng context/lineage: [quality_usage.md](quality_usage.md).", ""]
    (OUT / "quality_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

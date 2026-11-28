"""Tạo gói mô hình DEMO cho cả 3 bài toán từ dữ liệu GIẢ, để chạy backend/frontend trên máy cá nhân
mà không cần quyền truy cập MIMIC-IV.

    python tools/make_demo_bundles.py                # P1, P2, P3
    python tools/make_demo_bundles.py --tasks p1

Đầu ra:
    artifacts/models/<task>/synthetic-<task>-...      gói mô hình (backend tự nạp nếu chưa có gói thật)
    artifacts/demo_results/<task>/results.json + figures/   (đặt RESULTS_DIR=artifacts/demo_results)
Số liệu từ dữ liệu giả KHÔNG có ý nghĩa y khoa; chỉ để kiểm tra hệ thống chạy đúng.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(env: dict, *args: str) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, env=env, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["p1", "p2", "p3"])
    ap.add_argument("--rows", type=int, default=6000)
    a = ap.parse_args()
    data = ROOT / "data" / "demo"
    data.mkdir(parents=True, exist_ok=True)
    for task in a.tasks:
        env = {**os.environ, "HG_TASK": task, "PYTHONUTF8": "1"}
        gold, split = data / f"gold_{task}.csv", data / f"split_{task}.json"
        run(env, "tools/make_synthetic_gold.py", "--task", task, "--rows", str(a.rows), "--out", str(gold))
        run(env, "-m", "ml.common.split", "--gold", str(gold), "--out", str(split))
        out = ROOT / "artifacts" / "demo_runs" / task
        run(env, "-m", "ml.trust.run", "--gold", str(gold), "--split", str(split), "--synthetic", "--quick",
            "--open-test", "--force-reopen-test", "--out", str(out),
            "--docs-dir", str(ROOT / "artifacts" / "demo_results" / task))
    print("\nXong. Chạy API với dữ liệu demo:\n  RESULTS_DIR=artifacts/demo_results uvicorn backend.app.main:app --port 8000")


if __name__ == "__main__":
    main()

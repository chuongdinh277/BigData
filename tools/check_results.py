"""Cổng kiểm tra chất lượng sau mỗi lượt chạy — sai một mục là DỪNG, chưa phân tích, chưa họp, chưa mở test.

    python tools/check_results.py --stage A   # sau lượt dữ liệu (đọc Gold trên S3: chỉ đếm số dòng/ca dương)
    python tools/check_results.py --stage B   # sau lượt mô hình + đánh giá (đọc results/trust/*)
    python tools/check_results.py --stage C   # sau benchmark
    python tools/check_results.py --stage D   # sau khi mở test

Mỗi kiểm tra so với con số ĐÃ BIẾT (results/data/verified_v1.json) hoặc một ngưỡng hợp lý, ví dụ
ROC-AUC > 0.95 trên dữ liệu bệnh án thường là dấu hiệu rò rỉ thông tin tương lai.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
B3 = "s3://heartguard-mimic-group11"
EXPECTED = {"p1": ("label_mortality", 73768, 3588), "p2": ("label_readmit30", 72666, 17796),
            "p3": ("label_mort90", 75285, 7571)}
TASKS = ("p1", "p2", "p3")
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def stage_a(version: str) -> None:
    import pandas as pd
    for t, (label, rows, pos) in EXPECTED.items():
        try:
            df = pd.read_parquet(f"{B3}/gold/{t}/{version}/", columns=["hadm_id", label])
            check(f"Gold {t.upper()} số dòng = {rows:,}", len(df) == rows, f"thực tế {len(df):,}")
            check(f"Gold {t.upper()} ca dương = {pos:,}", int(df[label].sum()) == pos, f"thực tế {int(df[label].sum()):,}")
            check(f"Gold {t.upper()} hadm_id không trùng", df["hadm_id"].is_unique)
        except Exception as e:  # noqa: BLE001
            check(f"Đọc được Gold {t.upper()} {version}", False, str(e)[:120])
    m = load(ROOT / "results/data" / f"p1_dataset_{version}.json")
    check("Kiểm tra chất lượng P1 PASSED", bool(m) and m.get("status") == "PASSED" and m.get("quality_status") == "PASSED",
          f"status={m and m.get('status')}, quality={m and m.get('quality_status')}")
    for t in ("p2", "p3"):
        check(f"Báo cáo đặc trưng {t.upper()} (kiểm tra rò rỉ lúc xuất viện đã chạy)",
              (ROOT / "results/data" / f"features_{t}.md").exists())
    sp = yaml.safe_load((ROOT / "configs/splits.yaml").read_text(encoding="utf-8"))
    for t in ("p2", "p3"):
        check(f"sha256 chia tập {t.upper()} đã ghi vào configs/splits.yaml", bool((sp.get(t) or {}).get("sha256")))
    check("Có số dòng/dung lượng 11 bảng", (ROOT / "results/benchmark/table_sizes.json").exists())


def stage_b(version: str) -> None:
    for t in TASKS:
        R = load(ROOT / "results/trust" / t / "results.json")
        if not R:
            check(f"{t.upper()}: có results/trust/{t}/results.json", False)
            continue
        check(f"{t.upper()}: chạy trên dữ liệu thật", R.get("real_data") is True)
        check(f"{t.upper()}: đúng phiên bản Gold {version}", R.get("data_version") == version, str(R.get("data_version")))
        prev = R["dev"]["positives"] / R["dev"]["rows"]
        best = max(R["cv"].values(), key=lambda v: v["pr_auc_mean"])
        check(f"{t.upper()}: PR-AUC CV > 1.5 × tỷ lệ nền", best["pr_auc_mean"] > 1.5 * prev,
              f"{best['pr_auc_mean']:.3f} so với nền {prev:.3f}")
        check(f"{t.upper()}: ROC-AUC CV trong (0.60, 0.95) — >0.95 nghi rò rỉ", 0.60 < best["roc_auc_mean"] < 0.95,
              f"{best['roc_auc_mean']:.3f}")
        check(f"{t.upper()}: SHAP cộng đúng (sai số < 1e-3)", R["shap"]["additivity_max_error"] < 1e-3,
              f"{R['shap']['additivity_max_error']:.1e}")
        oe = R["dev_metrics"]["oe_ratio"]
        check(f"{t.upper()}: hiệu chuẩn O/E trong [0.8, 1.25]", 0.8 <= oe <= 1.25, f"{oe:.2f}")
        check(f"{t.upper()}: có so sánh ghép cặp + khuyến nghị chọn mô hình", bool(R.get("model_comparison")))
        check(f"{t.upper()}: drift đủ 5 giai đoạn", len(R["drift"]["periods"]) == 5, str(len(R["drift"]["periods"])))
        check(f"{t.upper()}: có phân tích nhóm phụ, thiếu dữ liệu, review, độ nhạy",
              all(R.get(k) for k in ("subgroup_dev", "robustness_dev", "review_dev", "sensitivity")))
        figs = list((ROOT / "results/trust" / t / "figures").glob("*.png"))
        check(f"{t.upper()}: đủ hình (≥ 12)", len(figs) >= 12, str(len(figs)))
        check(f"{t.upper()}: test CHƯA mở ở lượt B", "test" not in R)
    check("Phân tích C6 đã chạy", (ROOT / "results/trust/p1_c6/results.json").exists())
    cmp_ = ROOT / "results/models/comparison.md"
    check(f"Bảng so sánh mô hình trên {version}", cmp_.exists() and f"Dataset {version}" in cmp_.read_text(encoding="utf-8"))


def stage_c() -> None:
    import pandas as pd
    b = ROOT / "results/benchmark"
    for f in ("table_sizes.json", "funnel.json", "pandas_full_load.json", "scaling.md"):
        check(f"Benchmark: có {f}", (b / f).exists())
    if (b / "scaling.csv").exists():
        s = pd.read_csv(b / "scaling.csv")
        st = s[(s["mode"] == "strong") & s["ok"]]
        check("Strong scaling ≥ 3 mức core", st["cores"].nunique() >= 3, str(sorted(st["cores"].unique())))
        check("Mỗi mức core chạy ≥ 2 lần", bool(len(st)) and st.groupby("cores").size().min() >= 2)
        check("Không có lần chạy lỗi", bool(s["ok"].all()), f"{int((~s['ok']).sum())} lần lỗi")


def stage_d() -> None:
    for t in TASKS:
        R = load(ROOT / "results/trust" / t / "results.json") or {}
        lock = load(ROOT / "results/trust" / t / "test_lock.json")
        check(f"{t.upper()}: test mở đúng 1 lần", bool(lock) and len(lock.get("openings", [])) == 1,
              str(len((lock or {}).get("openings", []))))
        tm = (R.get("test") or {}).get("models", {}).get(R.get("chosen_model"), {})
        roc = (tm.get("metrics") or {}).get("roc_auc")
        check(f"{t.upper()}: có kết quả test kèm CI", bool(tm.get("ci95")))
        check(f"{t.upper()}: ROC-AUC test < 0.95", roc is not None and roc < 0.95, str(roc))
        check(f"{t.upper()}: gói mô hình đã tải lên S3", bool(R.get("bundle_s3")), str(R.get("bundle_s3")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list("ABCD"), required=True)
    ap.add_argument("--version", default="v1_1")
    a = ap.parse_args()
    {"A": lambda: stage_a(a.version), "B": lambda: stage_b(a.version), "C": stage_c, "D": stage_d}[a.stage]()
    width = max(len(n) for n, _, _ in RESULTS)
    for n, ok, d in RESULTS:
        print(f"{'✅' if ok else '❌'} {n:<{width}}  {d}")
    bad = sum(not ok for _, ok, _ in RESULTS)
    print(f"\nCổng {a.stage}: {'ĐẠT' if not bad else f'KHÔNG ĐẠT ({bad} mục)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

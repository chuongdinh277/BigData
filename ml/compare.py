"""Gộp results.csv của cả nhóm thành một bảng so sánh cho Giai đoạn 4.

Chạy từ thư mục gốc repo:
    python -m ml.compare

Đọc results/models/*.csv, kiểm tra cùng data_version và split_seed,
rồi ghi results/models/comparison.md (xếp theo PR-AUC trên validation).
Chỉ dùng kết quả trên validation, không có số liệu tập test.
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
METRICS = ("pr_auc", "roc_auc", "f1", "brier")


def load_rows(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(REPO / pattern))):
        folder = Path(path).stem
        if folder in ("comparison",):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["_source"] = folder
                rows.append(r)
    return rows


def check_consistent(rows: list[dict]) -> None:
    versions = {r["data_version"] for r in rows}
    seeds = {r["split_seed"] for r in rows}
    if len(versions) != 1 or len(seeds) != 1:
        raise SystemExit(f"Kết quả không cùng dữ liệu hoặc cùng bản chia: version={versions}, seed={seeds}")


def fmt(r: dict, k: str) -> str:
    return f"{float(r[k]):.4f}"


def table(rows: list[dict]) -> list[str]:
    out = ["| Hạng | Mô hình | Lần chạy | Người | PR-AUC | ROC-AUC | F1 | Brier | Giây |", "|---:|---|---|---|---:|---:|---:|---:|---:|"]
    for i, r in enumerate(rows, 1):
        out.append(
            f"| {i} | {r['model']} | {r['case_id']} | {r['member']} | {fmt(r, 'pr_auc')} | {fmt(r, 'roc_auc')} | "
            f"{fmt(r, 'f1')} | {fmt(r, 'brier')} | {float(r['seconds']):.1f} |"
        )
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/models/comparison.md")
    ap.add_argument("--data-version", default=None,
                    help="Chỉ so các lần chạy trên phiên bản Gold này (mặc định: phiên bản mới nhất có trong results/models)")
    a = ap.parse_args(argv)

    rows = load_rows("results/models/*.csv")
    if not rows:
        raise SystemExit("Không thấy results/models/*.csv")
    version = a.data_version or max({r["data_version"] for r in rows})
    rows = [r for r in rows if r["data_version"] == version]
    if not rows:
        raise SystemExit(f"Không có kết quả nào trên dữ liệu {version}")
    check_consistent(rows)
    rows.sort(key=lambda r: float(r["pr_auc"]), reverse=True)

    best: dict[str, dict] = {}
    for r in rows:  # đã xếp giảm dần theo PR-AUC nên lần đầu gặp là tốt nhất
        best.setdefault(r["model"], r)
    best_rows = sorted(best.values(), key=lambda r: float(r["pr_auc"]), reverse=True)

    first = rows[0]
    lines = [
        "# Giai đoạn 4: so sánh mô hình",
        "",
        f"Dữ liệu: Dataset {first['data_version']} (Gold P1). Chia tập theo bệnh nhân 70/15/15, seed {first['split_seed']}.",
        "Tất cả số liệu là trên tập validation. Tập test chưa mở.",
        "Chỉ số chính: PR-AUC (tỉ lệ ca tử vong khoảng 4,86%, đoán ngẫu nhiên chỉ được khoảng 0,048).",
        "",
        "## Tốt nhất của từng mô hình",
        "",
        *table(best_rows),
        "",
        "## Tất cả các lần chạy",
        "",
        *table(rows),
        "",
        "Bảng này sinh tự động bằng `python -m ml.compare`. Muốn cập nhật, chạy lại lệnh sau khi có results.csv mới.",
        "",
    ]
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Đã ghi {out} ({len(rows)} lần chạy, {len(best_rows)} mô hình)")


if __name__ == "__main__":
    main()

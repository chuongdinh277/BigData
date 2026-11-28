"""Ghi một dòng kết quả vào results.csv theo đúng cột chung."""
import csv
import json
from pathlib import Path

from .config import RESULT_COLUMNS


def write_result(path: str, row: dict) -> None:
    extra = set(row) - set(RESULT_COLUMNS)
    if extra:
        raise ValueError(f"Cột không có trong mẫu chung: {sorted(extra)}")
    row = {k: row.get(k, "") for k in RESULT_COLUMNS}
    if isinstance(row["params"], dict):
        row["params"] = json.dumps(row["params"], ensure_ascii=False, sort_keys=True)
    for k in ("roc_auc", "pr_auc", "positive_rate", "f1", "threshold", "brier", "seconds"):
        if isinstance(row[k], float):
            row[k] = round(row[k], 5)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)

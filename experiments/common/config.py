"""Hằng số dùng chung, đọc từ contracts/gold_schema_<task>.json để không lệch với Gold.

Bài toán chọn bằng biến môi trường HG_TASK = p1 (mặc định) | p2 | p3, ví dụ:
    HG_TASK=p2 python -m experiments.stage5.run_stage5 --gold s3://.../gold/p2/v1/ --split ...
"""
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK = os.environ.get("HG_TASK", "p1").lower()
if TASK not in ("p1", "p2", "p3"):
    raise ValueError(f"HG_TASK phải là p1, p2 hoặc p3 (đang là {TASK!r})")
SCHEMA = json.loads((REPO_ROOT / "contracts" / f"gold_schema_{TASK}.json").read_text(encoding="utf-8"))

SEED = 42
RATIOS = (0.70, 0.15, 0.15)            # train / validation / test, chia theo bệnh nhân
LABEL = SCHEMA["label"]                 # label_mortality | label_readmit30 | label_mort90
GROUP_KEY = SCHEMA["group_key"]         # subject_id

ID_COLUMNS = [c["name"] for c in SCHEMA["id_columns"]]
SUBGROUP_COLUMNS = [c["name"] for c in SCHEMA["subgroup_only_columns"]]
# Không bao giờ đưa vào mô hình: định danh, thời gian, nhãn, cột chỉ dùng đánh giá nhóm con.
NON_FEATURE_COLUMNS = set(ID_COLUMNS) | set(SUBGROUP_COLUMNS) | {LABEL}

RESULT_COLUMNS = [
    "case_id", "member", "strategy", "model", "params", "data_version", "split_seed",
    "roc_auc", "pr_auc", "positive_rate", "f1", "threshold", "brier", "seconds", "note",
]

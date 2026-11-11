"""Hằng số dùng chung, đọc từ contracts/gold_schema_p1.json để không lệch với Gold."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((REPO_ROOT / "contracts" / "gold_schema_p1.json").read_text(encoding="utf-8"))

SEED = 42
RATIOS = (0.70, 0.15, 0.15)            # train / validation / test, chia theo bệnh nhân
LABEL = SCHEMA["label"]                 # label_mortality
GROUP_KEY = SCHEMA["group_key"]         # subject_id

ID_COLUMNS = [c["name"] for c in SCHEMA["id_columns"]]
SUBGROUP_COLUMNS = [c["name"] for c in SCHEMA["subgroup_only_columns"]]
# Không bao giờ đưa vào mô hình: định danh, thời gian, nhãn, cột chỉ dùng đánh giá nhóm con.
NON_FEATURE_COLUMNS = set(ID_COLUMNS) | set(SUBGROUP_COLUMNS) | {LABEL}

RESULT_COLUMNS = [
    "case_id", "member", "strategy", "model", "params", "data_version", "split_seed",
    "roc_auc", "pr_auc", "positive_rate", "f1", "threshold", "brier", "seconds", "note",
]

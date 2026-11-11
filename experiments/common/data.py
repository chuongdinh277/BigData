"""Đọc bảng Gold P1 (csv giả hoặc parquet thật) và xác định cột đặc trưng."""
from pathlib import Path

import pandas as pd

from .config import GROUP_KEY, LABEL, NON_FEATURE_COLUMNS


def load_gold(path: str) -> pd.DataFrame:
    """Đọc Gold từ .csv, file/thư mục .parquet, hoặc s3://... (cần quyền IAM role, pip install s3fs)."""
    p = str(path)
    if p.startswith("s3a://"):
        p = "s3://" + p[len("s3a://"):]
    if p.endswith(".csv"):
        df = pd.read_csv(p)
    else:
        df = pd.read_parquet(p)
    missing = {LABEL, GROUP_KEY} - set(df.columns)
    if missing:
        raise ValueError(f"Gold thiếu cột bắt buộc: {sorted(missing)}")
    if df[LABEL].isna().any():
        raise ValueError("Gold có nhãn NULL")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Mọi cột trừ định danh, thời gian, nhãn và cột nhóm con (theo hợp đồng Gold)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def split_xy(df: pd.DataFrame, features: list[str] | None = None):
    feats = features or feature_columns(df)
    return df[feats], df[LABEL].astype(int).to_numpy()


def data_version(path: str) -> str:
    """Tên phiên bản từ đường dẫn, ví dụ .../gold/p1/v1 -> 'v1'; file giả -> tên file."""
    name = Path(str(path).rstrip("/")).name
    return name.rsplit(".", 1)[0]

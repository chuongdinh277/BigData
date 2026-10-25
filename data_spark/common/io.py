"""Đọc bảng MIMIC-IV theo một hợp đồng cột cố định (không inferSchema).

Bronze: <root>/<module>/<table>.csv.gz   (module = hosp | icu)
Silver: <root>/<module>/<table>/         (Parquet, do bronze_to_silver.py tạo)
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

REPO_ROOT = Path(__file__).resolve().parents[2]

# Cột cần dùng và kiểu đích. Chỉ ép kiểu những cột này; cột khác giữ dạng string.
TABLE_SPECS: dict[str, dict] = {
    "patients": {"module": "hosp", "cols": {
        "subject_id": "long", "gender": "string", "anchor_age": "int",
        "anchor_year": "int", "anchor_year_group": "string", "dod": "date"}},
    "admissions": {"module": "hosp", "cols": {
        "subject_id": "long", "hadm_id": "long", "admittime": "timestamp",
        "dischtime": "timestamp", "deathtime": "timestamp", "admission_type": "string",
        "admission_location": "string", "discharge_location": "string",
        "insurance": "string", "race": "string", "edregtime": "timestamp",
        "edouttime": "timestamp", "hospital_expire_flag": "int"}},
    "diagnoses_icd": {"module": "hosp", "cols": {
        "subject_id": "long", "hadm_id": "long", "seq_num": "int",
        "icd_code": "string", "icd_version": "int"}},
    "d_icd_diagnoses": {"module": "hosp", "cols": {
        "icd_code": "string", "icd_version": "int", "long_title": "string"}},
    "d_labitems": {"module": "hosp", "cols": {
        "itemid": "int", "label": "string", "fluid": "string", "category": "string"}},
    "labevents": {"module": "hosp", "cols": {
        "subject_id": "long", "hadm_id": "long", "itemid": "int",
        "charttime": "timestamp", "storetime": "timestamp",
        "valuenum": "double", "valueuom": "string"}},
    "omr": {"module": "hosp", "cols": {
        "subject_id": "long", "chartdate": "date", "seq_num": "int",
        "result_name": "string", "result_value": "string"}},
    "icustays": {"module": "icu", "cols": {
        "subject_id": "long", "hadm_id": "long", "stay_id": "long",
        "intime": "timestamp", "outtime": "timestamp", "first_careunit": "string"}},
    "d_items": {"module": "icu", "cols": {
        "itemid": "int", "label": "string", "linksto": "string",
        "category": "string", "unitname": "string"}},
    "chartevents": {"module": "icu", "cols": {
        "subject_id": "long", "hadm_id": "long", "stay_id": "long",
        "itemid": "int", "charttime": "timestamp", "valuenum": "double",
        "valueuom": "string"}},
}


def load_yaml(rel_path: str) -> dict:
    with open(REPO_ROOT / rel_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cast(df: DataFrame, cols: dict[str, str]) -> DataFrame:
    out = []
    for name, typ in cols.items():
        if name not in df.columns:
            raise KeyError(f"Thiếu cột '{name}' — kiểm tra phiên bản MIMIC-IV")
        c = F.col(name)
        if typ == "timestamp":
            c = F.to_timestamp(c)
        elif typ == "date":
            c = F.to_date(c)
        elif typ != "string":
            c = c.cast(typ)
        out.append(c.alias(name))
    return df.select(*out)


def read_table(spark: SparkSession, root: str, table: str, fmt: str = "bronze") -> DataFrame:
    spec = TABLE_SPECS[table]
    module = spec["module"]
    root = root.rstrip("/")
    if fmt == "silver":
        return spark.read.parquet(f"{root}/{module}/{table}").select(*spec["cols"].keys())
    raw = (
        spark.read.option("header", True)
        .option("inferSchema", False)
        .option("multiLine", False)
        .option("escape", '"')
        .csv(f"{root}/{module}/{table}.csv.gz")
    )
    return _cast(raw, spec["cols"])


def icd_prefix_condition(code_col, version_col, codes: dict) -> "F.Column":
    """True nếu icd_code bắt đầu bằng một trong các tiền tố của đúng phiên bản ICD."""
    conds = []
    for ver_key, ver in (("icd9", 9), ("icd10", 10)):
        for p in codes.get(ver_key, []) or []:
            conds.append((version_col == ver) & code_col.startswith(str(p)))
    cond = conds[0]
    for c in conds[1:]:
        cond = cond | c
    return cond

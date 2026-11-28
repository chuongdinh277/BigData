"""Giá trị bị chặn của máy xét nghiệm ("<0.01", ">60") phải được đọc thành số, không bị coi là thiếu.

Trước Dataset v1.1: troponin T mất 31.2% phép đo (24,073/77,209) và eGFR OMR thiếu 99.9% vì lỗi này.
"""
from __future__ import annotations

import os

import pytest

pyspark = pytest.importorskip("pyspark")


@pytest.fixture(scope="module")
def spark():
    os.environ.setdefault("HG_LOCAL", "1")
    from data_spark.common.session import get_spark
    s = get_spark("test-censored")
    yield s


def test_parse_censored_fills_valuenum_and_flags(spark):
    from data_spark.features import parse_censored
    df = spark.createDataFrame(
        [("<0.01", None), ("< 0.03", None), (">60", None), ("0.25", 0.25), ("NEG", None), ("<=5", None)],
        "value string, valuenum double")
    rows = parse_censored(df).collect()
    got = [(r["valuenum"], r["censored"]) for r in rows]
    assert got == [(0.01, 1), (0.03, 1), (60.0, 1), (0.25, 0), (None, 0), (5.0, 1)]


def test_parse_censored_without_value_column_is_noop(spark):
    from data_spark.features import parse_censored
    df = spark.createDataFrame([(1.0,)], "valuenum double")
    out = parse_censored(df).collect()[0]
    assert out["valuenum"] == 1.0 and out["censored"] == 0

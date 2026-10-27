"""Spark thật cho kiểm thử quality/leakage; không truy cập S3."""
import os
import sys

import pytest


@pytest.fixture(scope="session")
def spark():
    from pyspark.sql import SparkSession

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    session = (SparkSession.builder.master("local[2]").appName("quality-tests")
               .config("spark.sql.shuffle.partitions", "2")
               .config("spark.sql.session.timeZone", "UTC")
               .config("spark.ui.enabled", "false").getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()

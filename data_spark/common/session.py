"""SparkSession dùng chung cho cả nhóm.

- Trên EC2: đọc s3a:// bằng IAM Role gắn với instance (không dùng access key).
- Trên máy cá nhân với dữ liệu giả: đặt HG_LOCAL=1 thì không nạp hadoop-aws.

Biến môi trường:
  HG_SPARK_MASTER   mặc định "local[*]"; cluster: "spark://<host>:7077"
  HG_DRIVER_MEMORY  mặc định "8g"
  HG_HADOOP_AWS     phiên bản hadoop-aws khớp Hadoop của Spark (Spark 3.5.x → 3.3.4)
  HG_LOCAL          "1" để chạy hoàn toàn local, không S3
  HG_SPARK_CONF     cấu hình thêm "k1=v1,k2=v2" — dùng cho cụm standalone (spark.executor.memory,
                    spark.executor.cores, spark.cores.max) và thí nghiệm ablation
                    (spark.sql.adaptive.enabled=false, spark.sql.autoBroadcastJoinThreshold=-1)
  HG_EVENT_LOG      thư mục event log để xem lại Spark UI qua History Server (vd file:///tmp/spark-events)
"""
from __future__ import annotations

import os

from pyspark.sql import SparkSession


def get_spark(app_name: str = "heartguard") -> SparkSession:
    master = os.getenv("HG_SPARK_MASTER", "local[*]")
    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", os.getenv("HG_DRIVER_MEMORY", "8g"))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", os.getenv("HG_SHUFFLE_PARTITIONS", "200"))
        .config("spark.sql.adaptive.enabled", "true")
    )
    if os.getenv("HG_LOCAL") != "1":
        hadoop_aws = os.getenv("HG_HADOOP_AWS", "3.3.4")
        builder = (
            builder.config("spark.jars.packages", f"org.apache.hadoop:hadoop-aws:{hadoop_aws}")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config(
                "spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
            )
            .config("spark.hadoop.fs.s3a.endpoint.region", os.getenv("AWS_REGION", "us-east-1"))
        )
    if os.getenv("HG_EVENT_LOG"):
        builder = builder.config("spark.eventLog.enabled", "true").config("spark.eventLog.dir", os.environ["HG_EVENT_LOG"])
    for kv in filter(None, os.getenv("HG_SPARK_CONF", "").split(",")):
        k, _, v = kv.partition("=")
        builder = builder.config(k.strip(), v.strip())
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(os.getenv("HG_LOG_LEVEL", "WARN"))
    return spark

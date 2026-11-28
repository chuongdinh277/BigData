"""Nạp gói mô hình MỘT LẦN khi khởi động (không bao giờ huấn luyện trong API).

Nguồn gói:
  MODEL_SOURCE=local → MODEL_DIR/<task>/<version>/  (version trống = thư mục mới nhất theo tên)
  MODEL_SOURCE=s3    → MODEL_S3_PREFIX/<task>/<version>/ tải về cache rồi kiểm SHA256 (IAM Role của EC2)
Bài toán nào không có gói thì tắt (GET /tasks báo `enabled: false`), API vẫn chạy cho bài khác.
"""
from __future__ import annotations

import logging
from pathlib import Path

from trustworthy_ai.bundle import HeartGuardModel

from .config import settings

LOG = logging.getLogger("heartguard.api.registry")


def _latest_local(task: str) -> Path | None:
    d = settings.model_dir / task
    if not d.exists():
        return None
    # "mới nhất" theo THỜI GIAN đóng gói (mtime của SHA256SUMS), không theo tên: tên bắt đầu bằng tên mô hình
    cands = sorted((p for p in d.iterdir() if p.is_dir() and (p / "SHA256SUMS").exists()),
                   key=lambda p: (p / "SHA256SUMS").stat().st_mtime)
    want = settings.model_versions.get(task)
    if want:
        return next((p for p in cands if p.name == want), None)
    real = [p for p in cands if not p.name.startswith("synthetic-")]
    return (real or cands)[-1] if cands else None


def _latest_s3(task: str) -> str | None:
    import boto3
    prefix = settings.model_s3_prefix.replace("s3a://", "s3://")
    bucket, _, key = prefix[len("s3://"):].partition("/")
    want = settings.model_versions.get(task)
    if want:
        return f"s3://{bucket}/{key.rstrip('/')}/{task}/{want}/"
    # phiên bản mới nhất = thư mục có SHA256SUMS được ghi gần nhất (gói chỉ được tải lên sau khi mở test)
    s3 = boto3.client("s3")
    sums = [o for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{key.rstrip('/')}/{task}/")
            for o in page.get("Contents", []) if o["Key"].endswith("/SHA256SUMS")]
    if not sums:
        return None
    newest = max(sums, key=lambda o: o["LastModified"])["Key"]
    return f"s3://{bucket}/{newest.rsplit('/', 1)[0]}/"


class ModelRegistry:
    def __init__(self) -> None:
        self.models: dict[str, HeartGuardModel] = {}
        self.errors: dict[str, str] = {}

    def load(self) -> None:
        for task in settings.tasks:
            try:
                if settings.model_source == "s3":
                    uri = _latest_s3(task)
                    if not uri:
                        raise FileNotFoundError(f"không có gói trên {settings.model_s3_prefix}/{task}/")
                    self.models[task] = HeartGuardModel.from_s3(uri, cache_dir=settings.model_dir / "cache",
                                                                verify=settings.verify_bundle)
                else:
                    path = _latest_local(task)
                    if not path:
                        raise FileNotFoundError(f"không có gói trong {settings.model_dir / task}")
                    self.models[task] = HeartGuardModel(path, verify=settings.verify_bundle)
                LOG.info("Đã nạp %s: %s", task, self.models[task].model_version)
            except Exception as e:  # noqa: BLE001 — một bài lỗi không làm sập cả API
                self.errors[task] = str(e)
                LOG.warning("Không nạp được mô hình %s: %s", task, e)

    def get(self, task: str) -> HeartGuardModel | None:
        return self.models.get(task)


registry = ModelRegistry()

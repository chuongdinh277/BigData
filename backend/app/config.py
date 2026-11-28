"""Cấu hình backend, đọc từ biến môi trường (xem .env.example ở gốc repo).

Không có giá trị bí mật nào trong code: quyền S3 lấy từ IAM Role của EC2, mật khẩu DB lấy từ .env.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    # PostgreSQL trong Docker Compose; SQLite chỉ dùng cho test/chạy thử trên máy cá nhân
    database_url: str = field(default_factory=lambda: os.getenv(
        "DATABASE_URL", f"sqlite:///{REPO_ROOT / 'artifacts' / 'heartguard_dev.db'}"))
    # local: đọc gói ở MODEL_DIR/<task>/<version>/ ; s3: tải từ MODEL_S3_PREFIX/<task>/<version>/
    model_source: str = field(default_factory=lambda: os.getenv("MODEL_SOURCE", "local"))
    model_dir: Path = field(default_factory=lambda: Path(os.getenv("MODEL_DIR", str(REPO_ROOT / "artifacts" / "models"))))
    model_s3_prefix: str = field(default_factory=lambda: os.getenv(
        "MODEL_S3_PREFIX", "s3://heartguard-mimic-group11/model-artifacts"))
    # Phiên bản cố định từng bài, vd MODEL_VERSION_P1=p1-xgboost-20261201-abc123; trống = bản mới nhất
    model_versions: dict = field(default_factory=lambda: {
        t: os.getenv(f"MODEL_VERSION_{t.upper()}", "") for t in ("p1", "p2", "p3")})
    tasks: list[str] = field(default_factory=lambda: _list("TASKS", "p1,p2,p3"))
    results_dir: Path = field(default_factory=lambda: Path(os.getenv("RESULTS_DIR", str(REPO_ROOT / "results" / "trust"))))
    cors_origins: list[str] = field(default_factory=lambda: _list("CORS_ORIGINS", "http://localhost:5173,http://localhost:8080"))
    verify_bundle: bool = field(default_factory=lambda: os.getenv("VERIFY_BUNDLE", "1") == "1")


settings = Settings()

TASK_INFO = {
    "p1": {"name": "Tử vong trong viện", "prediction_time": "24 giờ sau nhập viện", "outcome": "tử vong trong lần nằm viện này"},
    "p2": {"name": "Tái nhập viện 30 ngày", "prediction_time": "Lúc xuất viện", "outcome": "tái nhập viện ngoài kế hoạch trong 30 ngày"},
    "p3": {"name": "Tử vong 90 ngày sau xuất viện", "prediction_time": "Lúc xuất viện", "outcome": "tử vong trong 90 ngày sau xuất viện"},
}

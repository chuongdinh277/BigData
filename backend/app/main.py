"""HeartGuard AI — API phục vụ dự đoán đáng tin cậy cho bệnh nhân suy tim.

Chạy thử:   uvicorn backend.app.main:app --reload --port 8000   (từ gốc repo)
Tài liệu:   http://localhost:8000/docs  (OpenAPI tự sinh)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from trustworthy_ai import DISCLAIMER

from .config import settings
from .db import init_db
from .registry import registry
from .routers import admin, predict, reviews, system

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    registry.load()          # nạp gói mô hình MỘT lần, không huấn luyện lại
    yield


app = FastAPI(title="HeartGuard AI API", version="1.0.0", lifespan=lifespan,
              description=f"Dự đoán nguy cơ suy tim có giải thích, hiệu chuẩn, giám sát drift và xem xét thủ công. "
                          f"**{DISCLAIMER}.**")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])

API = "/api/v1"
app.include_router(system.router, prefix=API)
app.include_router(predict.router, prefix=API)
app.include_router(reviews.router, prefix=API)
app.include_router(admin.router, prefix=API)

# Container API: chỉ PHỤC VỤ mô hình (không Spark, không huấn luyện). Build từ GỐC repo:
#   docker build -f backend/Dockerfile -t heartguard-api .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 PIP_NO_CACHE_DIR=1
# libgomp1: LightGBM/XGBoost cần OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

WORKDIR /app
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY trustworthy_ai/ trustworthy_ai/
COPY backend/ backend/
COPY contracts/ contracts/
COPY configs/ configs/
RUN mkdir -p artifacts/models results/trust && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD curl -fsS http://localhost:8000/api/v1/health || exit 1
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

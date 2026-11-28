# HeartGuard AI — mọi thao tác của dự án đi qua Makefile để cả nhóm chạy GIỐNG NHAU.
# Trên EC2: source ~/venv-hg/bin/activate trước. Trên Windows dùng Git Bash + make, hoặc gõ thẳng lệnh python bên dưới.
#
#   make help                     xem danh sách lệnh
#   make demo                     chạy toàn hệ thống trên máy cá nhân với dữ liệu GIẢ (không cần AWS)
#   make etl TASK=p1              pipeline Spark Bronze → Silver → Gold (EC2, dữ liệu thật)
#   make trust TASK=p1            đánh giá độ tin cậy + chọn mô hình (test còn đóng)
#   make open-test TASK=p1        mở tập test MỘT lần + tải gói mô hình lên S3
#   make up                       chạy ứng dụng (PostgreSQL + FastAPI + React) bằng Docker Compose

SHELL       := /bin/bash
PY          ?= python
TASK        ?= p1
VERSION     ?= v1_1
JOBS        ?= 8
BUCKET      := s3://heartguard-mimic-group11
GOLD        ?= $(BUCKET)/gold/$(TASK)/$(VERSION)/
SPLIT       ?= $(BUCKET)/splits/$(TASK)/v1/split_manifest.json
COMPOSE     := docker compose -f deploy/docker-compose.yml --env-file .env
export PYTHONUTF8 := 1
export HG_TASK    := $(TASK)

.PHONY: help
help:  ## Danh sách lệnh
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-16s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- môi trường
.PHONY: install install-web check-aws
install:  ## Cài thư viện Python (đầy đủ + Jupyter)
	$(PY) -m pip install -r requirements-dev.txt
install-web:  ## Cài thư viện frontend
	cd frontend && npm ci
check-aws:  ## Kiểm tra EC2 đọc được bucket bằng IAM Role
	aws sts get-caller-identity && aws s3 ls $(BUCKET)/ | head

# ---------------------------------------------------------------- dữ liệu phân tán (Spark)
.PHONY: silver feasibility sizes etl etl-sample quality
silver:  ## Bronze csv.gz → Silver Parquet (một lần)
	$(PY) -m data_spark.bronze_to_silver
feasibility:  ## Khảo sát khả thi cohort/nhãn/đặc trưng (số liệu tổng hợp) trên Silver
	$(PY) -m data_spark.stage1_feasibility --format silver --root $(BUCKET:s3://=s3a://)/silver/mimiciv/3.1 --out reports/stage1
sizes:  ## Số dòng + dung lượng Bronze/Silver của 11 bảng → results/benchmark/table_sizes.json
	$(PY) -m data_spark.scaling sizes
etl:  ## Cohort → nhãn → đặc trưng → kiểm tra chất lượng → Gold (TASK=p1|p2|p3, VERSION=v1_1)
	$(PY) -m data_spark.pipeline_runner --task $(TASK) --steps cohort labels features validate --version $(VERSION)
etl-sample:  ## Như etl nhưng trên 1% bệnh nhân (kiểm tra nhanh)
	$(PY) -m data_spark.pipeline_runner --task $(TASK) --steps features validate --sample-frac 0.01
quality:  ## Chỉ chạy lại kiểm tra chất lượng + rò rỉ của Gold
	$(PY) -m data_spark.pipeline_runner --task $(TASK) --steps validate --version $(VERSION)

# ---------------------------------------------------------------- benchmark Big Data
.PHONY: bench-sizes bench-funnel bench-pandas bench-strong bench-weak bench-ablation bench-report bench
bench-sizes: sizes
bench-funnel:  ## Phễu thu gọn labevents qua từng bước lọc
	$(PY) -m data_spark.scaling funnel
bench-pandas:  ## Thử nạp toàn bộ labevents bằng pandas (ghi RAM / hết bộ nhớ)
	$(PY) -m data_spark.scaling pandas
bench-strong:  ## Strong scaling 1/2/4/8 core, 3 lần mỗi cấu hình
	$(PY) -m data_spark.scaling strong --cores 1 2 4 8
bench-weak:  ## Weak scaling (dữ liệu tăng theo core)
	$(PY) -m data_spark.scaling weak --cores 2 4 8
bench-ablation:  ## Bật/tắt AQE, broadcast
	$(PY) -m data_spark.scaling ablation --cores 8
bench-report:  ## Tổng hợp results/benchmark/scaling.{md,png}
	$(PY) -m data_spark.scaling report
bench: bench-sizes bench-funnel bench-pandas bench-strong bench-weak bench-ablation bench-report  ## Toàn bộ benchmark

# ---------------------------------------------------------------- học máy
.PHONY: split train-lr train-xgb train-lgbm train-rf compare trust open-test
split:  ## Chia tập theo bệnh nhân 70/15/15 (ghi sha256 vào configs/splits.yaml)
	$(PY) -m ml.common.split --gold $(GOLD) --out $(SPLIT)
train-lr:  ## Logistic Regression: mặc định, trọng số, Optuna 50 lượt
	$(PY) -m ml.models.logistic --gold $(GOLD) --split $(SPLIT) --trials 50
train-xgb:  ## XGBoost
	$(PY) -m ml.models.xgboost_model --gold $(GOLD) --split $(SPLIT) --trials 50 --jobs $(JOBS)
train-lgbm:  ## LightGBM
	$(PY) -m ml.models.lightgbm_model --gold $(GOLD) --split $(SPLIT) --trials 50 --jobs $(JOBS)
train-rf:  ## Random Forest (tái lập kết quả GĐ4)
	$(PY) -m ml.models.random_forest --gold $(GOLD) --split $(SPLIT) --trials 50 --jobs $(JOBS)
compare:  ## Bảng so sánh results/models/comparison.md
	$(PY) -m ml.compare
trust:  ## Độ tin cậy + chọn mô hình (test CÒN ĐÓNG) → results/trust/$(TASK)/
	$(PY) -m ml.trust.run --gold $(GOLD) --split $(SPLIT) --jobs $(JOBS)
open-test:  ## Mở test MỘT lần + tải gói mô hình lên S3 (sau khi nhóm chốt)
	$(PY) -m ml.trust.run --gold $(GOLD) --split $(SPLIT) --jobs $(JOBS) --open-test --upload

# ---------------------------------------------------------------- ứng dụng
.PHONY: demo-models api web up down logs
demo-models:  ## Tạo gói mô hình DEMO cho P1/P2/P3 từ dữ liệu giả
	$(PY) tools/make_demo_bundles.py
api:  ## Chạy API ở chế độ phát triển (http://localhost:8000/docs)
	$(PY) -m uvicorn backend.app.main:app --reload --port 8000
web:  ## Chạy giao diện ở chế độ phát triển (http://localhost:5173)
	cd frontend && npm run dev
demo: demo-models  ## Dữ liệu giả → gói mô hình → API (chạy `make web` ở terminal khác)
	$(MAKE) api
up:  ## Docker Compose: db + api + web (http://<ip>:8080)
	$(COMPOSE) up -d --build
down:  ## Dừng ứng dụng
	$(COMPOSE) down
logs:  ## Xem log ứng dụng
	$(COMPOSE) logs -f --tail=100

# ---------------------------------------------------------------- chất lượng mã
.PHONY: test test-fast notebooks
test:  ## Toàn bộ test (cần Java 17 cho phần Spark)
	HG_LOCAL=1 HG_SPARK_MASTER='local[2]' HG_SHUFFLE_PARTITIONS=4 HG_DRIVER_MEMORY=2g $(PY) -m pytest -q tests
test-fast:  ## Test không cần Spark (ML, độ tin cậy, API)
	$(PY) -m pytest -q tests/test_trustworthy.py tests/test_api.py tests/test_experiments_common.py tests/test_xgboost.py tests/test_lightgbm.py tests/test_contracts.py
notebooks:  ## Chạy lại mọi notebook trên dữ liệu giả để kiểm tra chúng còn chạy
	for nb in notebooks/0*.ipynb; do $(PY) -m jupyter nbconvert --to notebook --execute --inplace "$$nb" || exit 1; done

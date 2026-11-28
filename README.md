# HeartGuard AI — Trustworthy Heart Failure Risk Intelligence

Hệ thống dự đoán nguy cơ **đáng tin cậy** cho bệnh nhân suy tim trên MIMIC-IV v3.1: pipeline xử lý dữ liệu phân tán bằng Apache Spark trên AWS, ba bài toán dự đoán, và một lớp độ tin cậy bao quanh mô hình gồm hiệu chuẩn, giải thích SHAP từng bệnh nhân, độ nhạy với dữ liệu thiếu, nhất quán giữa các nhóm bệnh nhân, giám sát drift dữ liệu và drift mô hình, cùng quy trình xem xét thủ công có nhật ký kiểm toán.

> ⚠️ **Research prototype – not for clinical use.**
> ⚠️ Repo **không chứa dữ liệu MIMIC-IV**. Dữ liệu chỉ nằm trong bucket S3 riêng của nhóm (DUA PhysioNet). Không commit, không chia sẻ, không dán dữ liệu thật vào công cụ AI trực tuyến. Mọi số liệu trong `results/` là số **tổng hợp**; ô có ít hơn 10 ca được che.

Môn Big Data (INT3229) · Subject 4 · Nhóm 11: Đinh Văn Quốc Chưởng (trưởng nhóm), Tôn Thiện Khoẻ, Đặng Thái Hà, Bùi Anh Chiến, Đoàn Thái Hùng.

---

## 1. Bài toán

| | P1 · Tử vong trong viện | P2 · Tái nhập viện 30 ngày | P3 · Tử vong 90 ngày sau xuất viện |
|---|---|---|---|
| Mốc dự đoán | `admittime + 24h` | `dischtime` | `dischtime` |
| Lượt nhập viện | 73,768 | 72,666 | 75,285 |
| Bệnh nhân | 30,287 | 27,870 | 28,855 |
| Tỷ lệ dương | 4.86% (3,588) | 24.49% (17,796) | 10.06% (7,571) |
| Đặc trưng (Gold) | 114 | 146 | 146 |

Cohort: 80,746 lượt nhập viện, 31,423 bệnh nhân ≥ 18 tuổi có mã suy tim (từ 546,028 lượt nhập viện của MIMIC-IV). Chi tiết phễu, nhãn, cách chia tập: `results/data/verified_v1.json`.

**Truy vết yêu cầu đề → thành phần hệ thống**

| Yêu cầu Subject 4 | Thành phần | Bằng chứng |
|---|---|---|
| Distributed feature-engineering pipeline | `data_spark/` (Bronze → Silver → Gold, PySpark trên S3) | `results/data/`, `results/benchmark/` |
| Train and compare suitable baselines | `ml/models/`, `ml/baselines.py`, `ml/trust/run.py` | CV 5 fold theo bệnh nhân, bootstrap ghép cặp, mốc GWTG-HF / HOSPITAL / LACE |
| ROC-AUC, PR-AUC, F1, calibration | `trustworthy_ai/evaluation.py`, `calibration.py`, `utility.py` | CI 95% theo bệnh nhân, O/E, slope, ECE, Decision Curve |
| Patient-level explanations | `trustworthy_ai/explain.py`, `POST /explain/{task}` | TreeSHAP, độ ổn định, deletion curve, hợp lý lâm sàng |
| Missing-data sensitivity | `trustworthy_ai/robustness.py` | Đường suy giảm PR-AUC và O/E theo 5 kịch bản che |
| Subgroup consistency | `trustworthy_ai/subgroup.py` | 13 trục (kể cả giao nhau), CI, recall/FPR gap |
| Data drift + model drift | `trustworthy_ai/drift.py`, `POST /drift/{task}` | PSI + ngưỡng null, bộ phân loại miền, drift ICD-9→10, 3 chiến lược bảo trì |
| Human review (high-risk / low-confidence) | `trustworthy_ai/review.py`, `backend/app/routers/reviews.py` | 7 lý do gắn cờ, vòng đời + nhật ký, risk–coverage |

## 2. Kiến trúc

```text
 PhysioNet MIMIC-IV 3.1
        │ aws s3 sync (một lần)
        ▼
 S3 heartguard-mimic-group11 ───────────────────────────────────────────────────────────┐
  bronze/ (csv.gz, 10 GB) ─► silver/ (Parquet) ─► processed/{cohort,labels} ─► gold/{p1,p2,p3}/<ver>/ │
        ▲  Spark trên EC2 (local[n] hoặc cụm standalone: deploy/spark/cluster.sh)       │
        │                                                                                ▼
        │                                               ml/ + trustworthy_ai/ (huấn luyện, độ tin cậy)
        │                                                    │ gói mô hình + SHA256SUMS
        │                                                    ▼
        └────────────────────────────── model-artifacts/<task>/<version>/ ── nạp 1 lần ─► FastAPI
                                                                                          │
                                         React (nginx) ◄──────────── /api/v1 ─────────────┤
                                                                     PostgreSQL ◄─────────┘
                                                        (dự đoán, hàng đợi review, nhật ký)
```

## 3. Cấu trúc thư mục

```text
configs/            data.yaml · hf_codes.yaml · feature_items.yaml · splits.yaml (bản chia tập đã khóa)
contracts/          gold_schema_{p1,p2,p3}.json (hợp đồng cột Gold) · api_contract.md
data_spark/         Pipeline Spark
  common/             session.py (SparkSession, S3A, cụm) · io.py (đọc bảng theo schema cố định)
  bronze_to_silver.py csv.gz → Parquet đã ép kiểu, phân vùng
  cohort.py · labels.py              cohort suy tim; nhãn P1/P2/P3
  features.py · features_discharge.py đặc trưng mốc +24h (P1) và mốc xuất viện (P2/P3)
  quality.py · quality_discharge.py  kiểm tra chất lượng + rò rỉ thời gian; FAILED thì không ghi Gold
  pipeline_runner.py                 điều phối các bước, manifest + báo cáo
  scaling.py · benchmark.py          benchmark khả năng mở rộng (strong/weak scaling, ablation, pandas)
ml/                 Học máy
  common/             chia tập theo bệnh nhân, chỉ số, khung chạy chung
  models/             logistic.py · xgboost_model.py · lightgbm_model.py · random_forest.py
  baselines.py        Dummy + GWTG-HF / HOSPITAL / LACE (xấp xỉ)
  compare.py          bảng so sánh kết quả GĐ4
  trust/run.py        CV, hiệu chuẩn, SHAP, thiếu dữ liệu, nhóm phụ, drift, review, mở test 1 lần, đóng gói
  trust/report.py     báo cáo Markdown + 15 hình
trustworthy_ai/     Thư viện độ tin cậy dùng chung cho đánh giá offline và API
backend/app/        FastAPI: main · config · db (SQLAlchemy) · registry (nạp gói) · routers/{predict,reviews,system,admin}
frontend/           React + TypeScript (Vite) · nginx.conf · Dockerfile
deploy/             docker-compose.yml · spark/{cluster.sh,spark-defaults.conf} · ec2/{bootstrap.sh,ec2.sh} · aws/iam-policy.json
notebooks/          01–09: mỗi phần một notebook để chạy và đọc số liệu
results/            Số liệu tổng hợp được commit: data/ · models/ (GĐ4) · trust/<task>/ · benchmark/
tools/              Dữ liệu giả, gói mô hình demo, sinh notebook
tests/              pytest: Spark ETL, chống rò rỉ, ML, độ tin cậy, API
Makefile            Mọi lệnh vận hành (`make help`)
```

---

## 4. Build

**Yêu cầu:** Python 3.12 (Spark 3.5 hỗ trợ 3.8–3.12), Java 17 (cho PySpark), Node.js 20, Docker + Compose plugin, GNU make. Trên EC2 Ubuntu, `deploy/ec2/bootstrap.sh` cài toàn bộ.

```bash
git clone https://github.com/chuongdinh277/BigData.git HeartGuard-AI && cd HeartGuard-AI
python3.12 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                       # pipeline + ML + API + test + Jupyter
cd frontend && npm ci && npm run build && cd ..           # typecheck + build giao diện
docker build -f backend/Dockerfile -t heartguard-api .    # image API (chỉ phục vụ, không Spark)
docker build -t heartguard-web frontend                   # image web (nginx)
make test                                                 # toàn bộ test (cần Java 17)
```

| File | Dùng khi |
|---|---|
| `requirements.txt` | Pipeline Spark + ML + API + test |
| `requirements-api.txt` | Container API (không Spark, không Optuna) |
| `requirements-dev.txt` | Thêm Jupyter để chạy `notebooks/` |

> Windows: đặt `PYTHONUTF8=1` để in tiếng Việt ra console. Phần Spark nên chạy trên Linux/EC2 (Spark 3.5 trên Windows cần thêm winutils và Java 17).

## 5. Deployment

### 5.1 Hạ tầng AWS

| Thành phần | Cấu hình |
|---|---|
| S3 `heartguard-mimic-group11` (us-east-1) | Riêng tư, Block Public Access, mã hóa SSE-S3; Bronze chỉ đọc |
| EC2 (khuyến nghị m5.2xlarge, 8 vCPU / 32 GB, EBS ≥ 100 GB) | Ubuntu 22.04/24.04; IAM Role theo `deploy/aws/iam-policy.json` (không access key trên máy) |
| Security Group | Chỉ mở 22 (SSH), 8080 (web) cho IP của nhóm; 7077/8081 chỉ trong VPC khi chạy cụm |
| Chi phí | AWS Budget alert; tắt EC2 ngay khi xong việc (`deploy/ec2/ec2.sh stop`) |

```bash
bash deploy/ec2/bootstrap.sh          # một lần trên EC2 mới: Java 17, Python 3.12, Docker, repo, venv
make check-aws                        # xác nhận IAM Role đọc được bucket
```

### 5.2 Cụm Spark standalone (tùy chọn, nhiều máy)

```bash
deploy/spark/cluster.sh master                                    # trên máy master
deploy/spark/cluster.sh worker spark://<ip-private-master>:7077 4 12g   # trên mỗi worker (cùng VPC, cùng IAM Role)
export HG_SPARK_MASTER=spark://<ip-private-master>:7077
export HG_SPARK_CONF="spark.executor.memory=10g,spark.executor.cores=3"
make etl TASK=p1                                                  # job chạy trên cụm
```

Cấu hình gợi ý cho từng tham số (bộ nhớ executor, số partition shuffle, AQE, skew join, event log): `deploy/spark/spark-defaults.conf`.

### 5.3 Ứng dụng (PostgreSQL + FastAPI + React)

```bash
cp .env.example .env                  # đặt POSTGRES_PASSWORD, MODEL_SOURCE=s3
make up                               # = docker compose -f deploy/docker-compose.yml --env-file .env up -d --build
# mở http://<EC2-public-IP>:8080 · tài liệu API: http://<ip>:8080/api/v1/... (OpenAPI ở /docs khi chạy dev)
make logs | make down
```

API nạp gói mô hình **một lần** khi khởi động: `MODEL_SOURCE=s3` tải `model-artifacts/<task>/<version>/` bằng IAM Role rồi kiểm SHA256. Bài toán nào chưa có gói thì tự tắt; các bài khác vẫn chạy.

## 6. Configuration

| Nơi | Nội dung |
|---|---|
| `.env` (từ `.env.example`) | Spark (`HG_SPARK_MASTER`, `HG_DRIVER_MEMORY`, `HG_SHUFFLE_PARTITIONS`, `HG_SPARK_CONF`, `HG_EVENT_LOG`), bài toán (`HG_TASK`), ứng dụng (`POSTGRES_PASSWORD`, `MODEL_SOURCE`, `MODEL_S3_PREFIX`, `MODEL_VERSION_P1..P3`, `TASKS`, `WEB_PORT`) |
| `configs/data.yaml` | Đường dẫn S3 từng lớp, mốc dự đoán (24h, 30 ngày, 90 ngày), cửa sổ xét nghiệm (−6h), ngưỡng che ô nhỏ (10) |
| `configs/hf_codes.yaml` | Mã ICD-9/10 xác định suy tim và 8 bệnh nền |
| `configs/feature_items.yaml` | itemid, đơn vị chủ đạo và khoảng sinh lý của 20 xét nghiệm, 7 sinh hiệu |
| `configs/splits.yaml` | sha256 bản chia tập đã khóa; mọi thí nghiệm kiểm tra khi nạp |
| `contracts/gold_schema_*.json` | Cột Gold được phép, cột cấm (chống rò rỉ), nhãn, cột chỉ dùng phân nhóm |
| `deploy/spark/spark-defaults.conf` | Tham số Spark khuyến nghị cho EC2 |

## 7. Operation

Toàn bộ thao tác đi qua `make` (`make help` để xem danh sách). Thứ tự chạy trên EC2 với dữ liệu thật:

| # | Lệnh | Làm gì | Đầu ra |
|---|---|---|---|
| 1 | `make silver` | Bronze csv.gz → Silver Parquet (một lần) | `s3://…/silver/` |
| 2 | `make feasibility` | Đếm số dòng, dung lượng từng bảng | `reports/stage1/` |
| 3 | `make etl TASK=p1 VERSION=v1_1` | Cohort → nhãn → đặc trưng → kiểm tra chất lượng → Gold | `s3://…/gold/p1/v1_1/`, `results/data/` |
| 4 | `make etl TASK=p2` · `make etl TASK=p3` | Gold mốc xuất viện | `s3://…/gold/{p2,p3}/` |
| 5 | `make split TASK=p2` (và p3) | Chia tập theo bệnh nhân; ghi sha256 vào `configs/splits.yaml` | `s3://…/splits/` |
| 6 | `make bench` | Phễu dữ liệu, pandas vs Spark, strong/weak scaling, ablation | `results/benchmark/` |
| 7 | `make trust TASK=p1` (p2, p3) | CV, mốc lâm sàng, hiệu chuẩn, SHAP, thiếu dữ liệu, nhóm phụ, drift, review — **test còn đóng** | `results/trust/<task>/report.md`, `results.json`, `figures/` |
| 8 | Họp nhóm chốt mô hình (ghi decision log) | Theo luật đặt trước: tree không hơn LR có ý nghĩa thì chọn LR | — |
| 9 | `make open-test TASK=p1` | Mở test **một lần**, tải gói lên S3 | `results/trust/<task>/test_lock.json`, `s3://…/model-artifacts/` |
| 10 | `make up` | Triển khai ứng dụng | http://<ip>:8080 |
| 11 | `deploy/ec2/ec2.sh stop` | Tắt máy, kiểm tra EBS/Elastic IP | — |

Nguyên tắc vận hành:
- **Tập test chỉ mở một lần**. Lần mở được ghi vào `test_lock.json`; mở lại phải dùng `--force-reopen-test` và việc này cũng được ghi lại.
- Gold chỉ được ghi khi kiểm tra chất lượng và rò rỉ **PASSED**.
- Một mình Hùng giữ quyền bật/tắt EC2. Chạy job dài trong `tmux`.
- Muốn cập nhật mô hình đang phục vụ: đặt `MODEL_VERSION_Px` trong `.env` rồi `make up`.

## 8. Usage

**Chạy toàn hệ thống trên máy cá nhân, không cần AWS (dữ liệu giả):**

```bash
python tools/make_demo_bundles.py      # dữ liệu giả → huấn luyện → gói mô hình demo P1/P2/P3 + báo cáo
make api                               # http://localhost:8000/docs
make web                               # http://localhost:5173
```

**API chính** (`/api/v1`, đầy đủ ở `/docs`):

| Endpoint | Chức năng |
|---|---|
| `GET /tasks` · `GET /health` | Bài toán đang bật, phiên bản mô hình, sức khỏe |
| `GET /schema/{task}` · `GET /cases/{task}` | Trường đầu vào (tên dễ hiểu, đơn vị, khoảng hợp lệ) · ca demo |
| `POST /predict/{task}` | Xác suất đã hiệu chuẩn, mức nguy cơ, SHAP top-k, chất lượng đầu vào, cờ review (tự tạo ca trong hàng đợi) |
| `POST /explain/{task}` · `POST /validate/{task}` | Chỉ giải thích · chỉ kiểm tra đầu vào |
| `GET /reviews` · `GET/PATCH /reviews/{id}` | Hàng đợi; chuyển `PENDING → IN_REVIEW → REVIEWED/ESCALATED` (sai thứ tự trả 409), mỗi bước ghi nhật ký |
| `POST /drift/{task}` | Kiểm tra drift của một lô ca mới so với tham chiếu trong gói mô hình (không cần nhãn) |
| `GET /reports/{task}/{kind}` | Báo cáo đánh giá: `metrics`, `baselines`, `calibration`, `explain`, `robustness`, `subgroup`, `drift`, `review` |
| `GET /admin/{data,models,system,audit}` | Cổng quản trị: số liệu pipeline, danh mục mô hình, khối lượng, nhật ký kiểm toán |

```bash
curl -s localhost:8000/api/v1/cases/p1 | jq '.[2]' > case.json
curl -s -X POST localhost:8000/api/v1/predict/p1 -H 'Content-Type: application/json' -d @case.json | jq '{calibrated_probability, risk_level, review}'
```

**Notebook học từng phần** (`jupyter lab notebooks/`): `01` dữ liệu–cohort–nhãn · `02` đặc trưng–chất lượng · `03` mô hình–so sánh · `04` hiệu chuẩn–review · `05` SHAP · `06` thiếu dữ liệu–nhóm phụ · `07` drift · `08` benchmark Big Data · `09` hệ thống–API. Mặc định đọc kết quả demo; trên EC2 đặt `REAL = True`.

## 9. Kiểm thử

| Lệnh | Phạm vi |
|---|---|
| `make test` | Toàn bộ: Spark ETL đầu–cuối trên MIMIC giả (cohort, nhãn, đặc trưng, rò rỉ có gài lỗi), ML, độ tin cậy, API |
| `make test-fast` | Không cần Spark: ML, độ tin cậy, API |
| CI (`.github/workflows/ci.yml`) | Python + Java 17 trên Ubuntu; chặn file dữ liệu bị commit nhầm; build frontend; build 2 Docker image |

## 10. Giới hạn

Dữ liệu một trung tâm (BIDMC, Boston), chưa kiểm định ngoài · cohort chọn bằng ICD lúc xuất viện · 417 ca tử vong trước giờ 24 nằm ngoài P1 · P2 chỉ thấy tái nhập cùng bệnh viện · `dod` chỉ ghi nhận tử vong ngoài viện trong giới hạn của hồ sơ tiểu bang · sinh hiệu chỉ có ở ca ICU · một số phép đo thiếu `storetime` · `anchor_year_group` là khoảng 3 năm gắn theo bệnh nhân nên phân lô thời gian chỉ gần đúng · thang điểm lâm sàng là bản xấp xỉ · SHAP không phải quan hệ nhân quả · người xem xét là thành viên nhóm, chưa phải bác sĩ.

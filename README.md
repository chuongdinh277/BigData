# HeartGuard AI — Trustworthy Heart Failure Risk Intelligence

Dự án môn Big Data (Subject 4) — Nhóm 11. Dự đoán nguy cơ ở bệnh nhân **đã mắc suy tim** trên MIMIC-IV v3.1, gồm pipeline PySpark, thực nghiệm ML, Trustworthy AI (calibration, SHAP, missing, subgroup, drift, human review) và ứng dụng React + FastAPI trên EC2.

> ⚠️ **Research prototype – not for clinical use.**
> ⚠️ **Repo này KHÔNG chứa dữ liệu MIMIC-IV.** Dữ liệu chỉ nằm trong bucket S3 private của nhóm. Không commit, không chia sẻ, không dán dữ liệu thật vào chatbot/LLM online (DUA PhysioNet).

## Trạng thái

| Giai đoạn | Nội dung | Trạng thái |
|---|---|---|
| **1** | Xác định bài toán & thiết kế | 🟡 Bản đề xuất v0.1, chờ nhóm duyệt + chạy số liệu khả thi |
| 2 | Pipeline Spark → Dataset V1 | ⬜ |
| 3 | Split, chiến lược, thiết kế thí nghiệm | ⬜ |
| 4 | Chạy, sàng lọc, tự tuning | ⬜ |
| 5 | Đánh giá độ tin cậy, chọn model | ⬜ |
| 6 | React – FastAPI – Model | ⬜ |
| 7 | Deploy EC2, báo cáo, slide | ⬜ |

## Cấu trúc

```text
configs/            data.yaml · hf_codes.yaml · feature_items.yaml
contracts/          gold_schema_p1.json · api_contract.md
data_spark/         common/ (session, io) · bronze_to_silver.py · stage1_feasibility.py
                    (GĐ2: cohort.py · labels.py · features.py · quality.py · pipeline_runner.py)
docs/               prediction_problem_spec.md · solution_specification.md · decision_log.md
  stage1/           data_inventory.md · cohort_spec.md · feature_spec.md · architecture.md
experiments/        member_1 … member_5
trustworthy_ai/     calibration, SHAP, missing, subgroup, drift (GĐ5)
backend/ frontend/ deployment/     (GĐ6–7)
tools/              make_synthetic_mimic.py · make_synthetic_gold.py
sample_data/        gold_p1_fake.csv (dữ liệu GIẢ, an toàn)
tests/              test_contracts.py
report/ slides/
```

Dữ liệu trên S3:

```text
s3://heartguard-mimic-group11/
├── bronze/mimiciv/3.1/{hosp,icu}/*.csv.gz   ← bản gốc PhysioNet (đã đủ 22 + 9 bảng)
├── silver/mimiciv/3.1/{hosp,icu}/<table>/    ← Parquet (GĐ2)
├── gold/<task>/<data_version>/               ← đặc trưng + nhãn
├── reports/                                  ← chỉ số liệu tổng hợp
└── model-artifacts/<task>/<model_version>/
```

## Giai đoạn 1 — ai làm gì

| Vai | File sở hữu | Việc còn lại sau khi có bản đề xuất |
|---|---|---|
| **TV1** Dữ liệu | `docs/stage1/data_inventory.md`, `data_spark/` | Dựng Spark trên EC2, kiểm tra checksum, **chạy `stage1_feasibility.py`**, điền số dòng |
| **TV2** Cohort | `docs/stage1/cohort_spec.md`, `configs/hf_codes.yaml` | Rà bộ mã HF + trích dẫn, chọn phương án A/B từ số liệu |
| **TV3** Bài toán | `docs/prediction_problem_spec.md` | Điền bảng khả thi §5, gửi câu hỏi §6 cho giảng viên |
| **TV4** Đặc trưng | `docs/stage1/feature_spec.md`, `configs/feature_items.yaml` | Chọn itemid cuối từ `lab_item_candidates`, đánh giá độ phủ |
| **TV5** Kiến trúc | `docs/stage1/architecture.md`, `contracts/` | Wireframe 5 màn hình, chốt API contract |

Mọi quyết định ghi vào `docs/decision_log.md` (đang ở trạng thái **ĐỀ XUẤT**).

## Chạy nhanh

### 1. Môi trường

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# PySpark cần Java 11/17
```

### 2. Thử với dữ liệu GIẢ (không cần credential)

```bash
python tools/make_synthetic_mimic.py --out sample_data/mimic_fake
HG_LOCAL=1 python -m data_spark.stage1_feasibility --root sample_data/mimic_fake --out reports/stage1_fake
python tools/make_synthetic_gold.py --rows 1000 --out sample_data/gold_p1_fake.csv
pytest -q tests
```

### 3. Trên EC2 với dữ liệu thật (TV1)

EC2 đặt ở `us-east-1`, gắn IAM Role có quyền đọc/ghi bucket. Không dùng access key trong code.

```bash
# kiểm tra toàn vẹn (tùy chọn, tải ~10 GB)
aws s3 cp s3://heartguard-mimic-group11/bronze/mimiciv/3.1/ /data/mimic --recursive
(cd /data/mimic && sha256sum -c SHA256SUMS.txt)

# khảo sát khả thi — nhanh (bỏ labevents)
python -m data_spark.stage1_feasibility --skip-labs --out reports/stage1
# đầy đủ (labevents 2.4 GB, chạy lâu)
python -m data_spark.stage1_feasibility --out reports/stage1
# thêm độ phủ sinh hiệu từ chartevents (3.3 GB)
python -m data_spark.stage1_feasibility --with-chartevents --out reports/stage1

# (GĐ2) chuyển Bronze → Silver Parquet, ghi log thời gian cho benchmark
python -m data_spark.bronze_to_silver --tables patients admissions diagnoses_icd d_icd_diagnoses d_labitems omr icustays d_items labevents
```

Output `reports/stage1/stage1_feasibility.md` chỉ chứa số liệu tổng hợp (ô < 10 đã che), được phép đưa vào báo cáo. Thư mục `reports/` bị `.gitignore` chặn mặc định; muốn commit báo cáo tổng hợp thì copy vào `docs/research_results/`.

## Quy tắc làm việc

Phần quality/leakage GĐ2 của Chiến: [hướng dẫn chạy và hợp đồng bàn giao](docs/stage2/quality_usage.md),
[kết quả kiểm chứng](docs/stage2/quality_report.md). Môi trường tối thiểu:
`pip install -r requirements-quality.txt` và Java 17.

- Nhánh `main` luôn chạy được. Mỗi người làm trên nhánh `tv<N>/<việc>`, mở Pull Request, cần ít nhất 1 người khác review.
- Không ai sửa định nghĩa nhãn, cohort hay schema mà không ghi vào `decision_log.md`.
- CI (`.github/workflows/ci.yml`) chặn commit nhầm `.csv.gz` / `.parquet` ngoài `sample_data/`.

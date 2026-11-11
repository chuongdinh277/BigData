# GĐ3 — Chia tập và mô hình cơ sở S0 (P1, Dataset V1)

## Dữ liệu
- Gold: `s3://heartguard-mimic-group11/gold/p1/v1/` (Dataset V1, xem `docs/stage2/dataset_v1.md`)
- 73,768 lượt nhập viện · 30,287 bệnh nhân · 3,588 ca dương (4.86%) · 114 đặc trưng

## Chia tập
- Theo bệnh nhân (`subject_id`), 70/15/15, phân tầng theo bệnh nhân có ca dương, seed 42
- `split_manifest.json` lưu tại `s3://heartguard-mimic-group11/splits/p1/v1/split_manifest.json`
- sha256: `584278c737fcbc884e88bb57ed1dc6f5f64fd0c211e645af16cd1e4532390ab5`
- File chứa subject_id của MIMIC nên **không commit lên Git** (đã chặn trong `.gitignore`)

| Tập | Bệnh nhân | Dòng | Ca dương | Tỷ lệ dương |
|---|---:|---:|---:|---:|
| train | 21,200 | 51,631 | 2,511 | 4.86% |
| validation | 4,543 | 11,129 | 538 | 4.83% |
| test (giữ kín đến GĐ5) | 4,544 | 11,008 | 539 | 4.90% |

## Mô hình cơ sở S0
Median imputer + StandardScaler + Logistic Regression (max_iter 2000), đánh giá trên validation, ngưỡng chọn theo F1 trên validation.

| Chỉ số | Giá trị |
|---|---:|
| ROC-AUC | 0.833 |
| PR-AUC | 0.299 (mốc ngẫu nhiên ≈ 0.048) |
| F1 | 0.342 (ngưỡng 0.177) |
| Brier | 0.040 |
| Thời gian | ~2 giây trên EC2 m5.2xlarge |

Kết quả chi tiết: `experiments/s0/results.csv`. Các chiến lược S1–S4 ở GĐ4 so với mốc này.

## Chạy lại trên EC2
```
G=s3://heartguard-mimic-group11/gold/p1/v1/
S=s3://heartguard-mimic-group11/splits/p1/v1/split_manifest.json
python -m experiments.common.runner --gold $G --split $S --out experiments/<tên>/results.csv --member <tên>
```
Không tạo lại split: dùng đúng file trên S3 (load_manifest kiểm tra sha256).

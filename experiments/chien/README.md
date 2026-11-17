# Thực nghiệm LightGBM (GĐ4) & Kiểm định Quality/Lineage Dataset V1 — Chiến

## 1. Huấn luyện LightGBM trên EC2 (Gold P1 V1)

Báo nhóm trước khi bật `heartguard-ec2`, sau đó chạy từ thư mục gốc repo:

```bash
source ~/venv-hg/bin/activate
G=s3://heartguard-mimic-group11/gold/p1/v1/
S=s3://heartguard-mimic-group11/splits/p1/v1/split_manifest.json

# Kiểm tra nhanh baseline S0 (tùy chọn, không ghi đè results.csv chính)
python -m experiments.common.runner --gold $G --split $S --out /tmp/s0_check.csv --member chien

# Chạy 3 biến thể LightGBM: default, weighted (scale_pos_weight từ train), Optuna 50 trials
python -m experiments.chien.run_lightgbm --gold $G --split $S --out experiments/chien/results.csv --trials 50 --jobs 4 --overwrite
```

Đầu ra: `experiments/chien/results.csv` gồm đúng 3 dòng (`S1_lgbm_default`, `S1_lgbm_weighted`, `S1_lgbm_optuna`) và 15 cột chuẩn `experiments/common/config.py`. Không đánh giá trên tập `test`.

## 2. Chạy kiểm định Quality + Lineage cho Dataset V1 (gỡ `NOT_CHECKED`)

Không ghi đè Gold `gold/p1/v1`. Chạy bước `validate` (chỉ đọc Silver + Cohort + Gold để dựng bảng `lineage` kiểm tra leakage):

```bash
python -m data_spark.pipeline_runner --steps validate --version v1
```

Hoặc chạy CLI độc lập của `data_spark.quality`:

```bash
python -m data_spark.quality --kind p1 \
  --table s3a://heartguard-mimic-group11/gold/p1/v1 \
  --context s3a://heartguard-mimic-group11/processed/cohort \
  --build-lineage --source-format silver \
  --out s3a://heartguard-mimic-group11/reports/quality/gold_p1_v1
```

Sau khi chạy xong trên EC2, Stop EC2 ngay và commit `experiments/chien/results.csv` cùng `docs/stage2/dataset_v1.md` (nếu có cập nhật).

## 3. Kiểm thử cục bộ với dữ liệu giả

```bash
python tools/make_synthetic_gold.py --rows 800 --out /tmp/gold_fake.csv
python -m experiments.common.split --gold /tmp/gold_fake.csv --out /tmp/split_fake.json
python -m experiments.chien.run_lightgbm --gold /tmp/gold_fake.csv --split /tmp/split_fake.json \
  --out /tmp/lgbm_fake.csv --trials 2 --jobs 2 --synthetic-split --overwrite
HG_LOCAL=1 python -m pytest -q tests/test_lightgbm.py tests/test_leakage.py tests/test_quality.py tests/test_pipeline_runner.py
```

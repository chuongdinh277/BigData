# Random Forest P1 — Hà

Chạy từ thư mục gốc repo trên EC2. Script dùng Gold P1 V1 và split manifest đã khóa của nhóm; không đọc hoặc đánh giá tập test.

## Cài thư viện
```bash
python -m pip install -r experiments/member_2/requirements.txt
```

## Chạy
```bash
python -m experiments.member_2.ha_random_forest --trials 50 --jobs 4
```

Mặc định, script đọc:
- Gold: `s3://heartguard-mimic-group11/gold/p1/v1/`
- Split: `s3://heartguard-mimic-group11/splits/p1/v1/split_manifest.json`
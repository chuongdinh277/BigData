# Khung thí nghiệm chung – Giai đoạn 3

Mọi người dùng chung bộ hàm này để kết quả so sánh được với nhau.

## 1. Tạo dữ liệu giả (trong lúc chờ Gold thật)
```bash
python tools/make_synthetic_gold.py --rows 5000 --out data/gold_fake.csv
```

## 2. Tạo cách chia tập theo bệnh nhân (70/15/15, seed 42)
```bash
python -m experiments.common.split --gold data/gold_fake.csv --out data/splits/split_fake.json
```
Gold thật: Chưởng tạo **một lần** `data/splits/split_v1.json` rồi gửi cả nhóm. File chứa subject_id nên **không commit**;
mọi người kiểm tra cùng `sha256` in ra.

## 3. Chạy mô hình cơ sở S0 để kiểm tra
```bash
python -m experiments.common.runner --gold data/gold_fake.csv --split data/splits/split_fake.json \
    --out experiments/member_5/results.csv --member Chuong
```

## 4. Chạy case của mình
```python
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from experiments.common.data import load_gold, feature_columns
from experiments.common.runner import run_case, make_preprocessor, split_feature_types

df = load_gold("data/gold_fake.csv")
feats = feature_columns(df)
num, cat = split_feature_types(df, feats)
pipe = Pipeline([("prep", make_preprocessor(num, cat, scale=False)),
                 ("model", XGBClassifier(scale_pos_weight=19, random_state=42))])
run_case(case_id="S1_xgb_scaleposweight", member="Ha", strategy="S1", model_name="XGBoost",
         pipeline=pipe, gold_path="data/gold_fake.csv", manifest_path="data/splits/split_fake.json",
         results_path="experiments/member_2/results.csv", params={"scale_pos_weight": 19}, df=df)
```

## Quy tắc
- Mọi bước tiền xử lý (điền thiếu, chuẩn hóa, chọn đặc trưng, SMOTE) phải nằm **trong pipeline** → chỉ học từ train.
- `run_case` chọn ngưỡng F1 và đánh giá trên **validation**; `apply_split` mặc định không trả tập test.
- Không đổi cột của `results.csv`; mỗi case một dòng.

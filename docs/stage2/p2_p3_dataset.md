# Dataset P2/P3 — mốc dự đoán lúc xuất viện

> P2: tái nhập viện ngoài kế hoạch trong 30 ngày (`label_readmit30`) · P3: tử vong do mọi nguyên nhân trong 90 ngày sau xuất viện (`label_mort90`).
> Nhãn và quần thể đã chạy trên dữ liệu thật (`docs/stage2/labels_result_real.md`, D-011): P2 72,666 lượt (24.49%), P3 75,285 lượt (10.06%).

## 1. Thành phần

| Thành phần | File | Ghi chú |
|---|---|---|
| Hợp đồng Gold | `contracts/gold_schema_p2.json`, `gold_schema_p3.json` | 146 đặc trưng, cùng tên cột với P1 ở phần chung |
| Tạo đặc trưng (Spark) | `data_spark/features_discharge.py` | Dùng lại hàm của `features.py`; mốc = `dischtime` |
| Kiểm tra chất lượng + rò rỉ | `data_spark/quality_discharge.py` | Chạy tự động trước khi ghi Gold; FAILED thì không ghi |
| Điều phối | `data_spark/pipeline_runner.py --task p2\|p3` | |
| Thí nghiệm + GĐ5 | `HG_TASK=p2\|p3` cho `experiments.common.*` và `experiments.stage5.run_stage5` | |
| Dữ liệu giả | `tools/make_synthetic_gold.py --task p2\|p3` | |
| Test | `tests/test_features_discharge.py`, `tests/test_trustworthy.py` | Gồm test đầu-cuối trên MIMIC giả |

## 2. Đặc trưng (khác P1)

| Nhóm | P1 (mốc +24h) | P2/P3 (mốc = xuất viện) |
|---|---|---|
| Quần thể | Còn nằm viện và còn sống tại +24h | **Đúng các hadm_id trong bảng nhãn của Hà** (không định nghĩa lại) |
| Xét nghiệm | first/min/max trong [admit − 6h, +24h] | first/min/max trong [admit − 6h, dischtime] + **`_last`** trong [dischtime − 48h, dischtime] |
| Số xét nghiệm | `n_lab_tests_24h` | `n_lab_tests_stay` |
| Sinh hiệu ICU | 24h đầu | Cả đợt [admittime, dischtime] |
| ICU | `in_icu_24h` | `icu_any`, `icu_days` (cắt vào [admittime, dischtime]) |
| Đợt nằm viện | — (cấm) | `los_days`, `discharge_location_group` (HOME / HOME_HEALTH / FACILITY / OTHER / UNKNOWN), `n_diagnoses`, `dx_<bệnh kèm>` của **chính đợt này** |
| Tiền sử, OMR, nhân khẩu | Như nhau: chỉ admission đã xuất viện trước admittime; OMR 365 ngày trước ngày nhập viện | |

ICD, LOS và nơi xuất viện của đợt hiện tại bị cấm ở P1, nhưng được dùng ở P2/P3 vì đã có tại thời điểm xuất viện (`feature_spec.md` §5).

## 3. Chống rò rỉ (`quality_discharge.check_discharge_gold`)

Gold chỉ được ghi khi mọi số đếm bằng 0:

- Khóa trùng hoặc NULL; nhãn NULL hoặc khác 0/1; nhãn lệch với bảng nhãn của Hà; dòng không có trong bảng nhãn.
- `prediction_time` ≠ `dischtime` của cohort; `admittime` lệch so với cohort.
- **Event dùng làm đặc trưng có `charttime` > dischtime, hoặc `storetime` > dischtime.** Bằng chứng là thời điểm event muộn nhất của từng lượt nhập viện; bằng chứng này không đưa vào Gold.
- `los_days` không khớp; `icu_days` > `los_days`; `_last` nằm ngoài [min, max]; cờ `_missing` mâu thuẫn với giá trị.
- Cột bị cấm: `deathtime`, `hospital_expire_flag`, `dod`, và mọi thông tin của lần nhập viện sau.

Test tự gài lỗi (event sau xuất viện, lệch mốc dự đoán, đảo nhãn) và xác nhận kiểm tra báo FAILED.

## 4. Chạy trên EC2

```bash
# 1. Gold (nhãn P2/P3 đã có ở processed/labels/p2|p3; cohort ở processed/cohort)
python -m data_spark.pipeline_runner --task p2 --steps features --version v1
python -m data_spark.pipeline_runner --task p3 --steps features --version v1
#   → s3://heartguard-mimic-group11/gold/p2|p3/v1/  +  docs/stage2/features_p2.md, features_p3.md

# 2. Chia tập theo bệnh nhân 70/15/15, seed 42 (manifest chứa subject_id → chỉ lưu S3)
HG_TASK=p2 python -m experiments.common.split --gold s3://heartguard-mimic-group11/gold/p2/v1/ \
    --out s3://heartguard-mimic-group11/splits/p2/v1/split_manifest.json
HG_TASK=p3 python -m experiments.common.split --gold s3://heartguard-mimic-group11/gold/p3/v1/ \
    --out s3://heartguard-mimic-group11/splits/p3/v1/split_manifest.json
#   → ghi sha256 in ra vào decision_log (khóa split P2/P3)

# 3. GĐ5 (test còn đóng) → nhóm chốt → mở test một lần
HG_TASK=p2 python -m experiments.stage5.run_stage5 --jobs 8      # → docs/stage5/p2/
HG_TASK=p3 python -m experiments.stage5.run_stage5 --jobs 8      # → docs/stage5/p3/
HG_TASK=p2 python -m experiments.stage5.run_stage5 --jobs 8 --open-test --upload
HG_TASK=p3 python -m experiments.stage5.run_stage5 --jobs 8 --open-test --upload
```

Siêu tham số: GĐ5 dùng tham số Optuna của P1 (GĐ4). Nếu ai tối ưu riêng cho P2/P3, ghi kết quả vào `experiments/<người>/results_p2.csv` (hoặc `results_p3.csv`) theo cùng cột, và GĐ5 sẽ tự đọc file đó.

## 5. Giới hạn

- P2 chỉ quan sát được tái nhập tại cùng bệnh viện (BIDMC), nên tỷ lệ thật bị đánh giá thấp.
- P3 dựa vào `patients.dod`. Dữ liệu này chỉ có ngày, và có thể thiếu ca tử vong ngoài tiểu bang.
- `_last` thiếu nhiều hơn `_first` vì không phải xét nghiệm nào cũng được làm trong 48h cuối. Cờ `_missing` chỉ phản ánh việc thiếu trong cả đợt.
- Bằng chứng chống rò rỉ của P2/P3 là thời điểm event muộn nhất của từng lượt nhập viện, chưa chi tiết từng event như lineage P1 của Chiến.

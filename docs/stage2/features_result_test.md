# Kết quả bảng đặc trưng P1 (Giai đoạn 2)

- Phiên bản dữ liệu: `v1`; mẫu: 1% bệnh nhân
- Số ca trong cohort sau quy tắc P1 (còn nằm viện, chưa tử vong tại 24h): 743
- Số dòng Gold: 743; số bệnh nhân: 308
- Ca dương (tử vong trong viện): 33 (4.44%)
- Nằm ICU trong 24h đầu: 184 (24.8%); thiếu toàn bộ sinh hiệu: 562
- Sinh hiệu: 21,403 phép đo trong cửa sổ trước khi lọc khoảng hợp lệ

## Loại bỏ theo từng xét nghiệm (cửa sổ -6h → 24h)

| Lab | Số phép đo | Không phải số | Sai đơn vị | Ngoài khoảng | Đơn vị chủ đạo | Tỷ lệ đơn vị |
|---|---:|---:|---:|---:|---|---:|
| magnesium | 1,242 | 0 | 0 | 0 | mg/dL | 100.0% |
| rdw | 1,419 | 6 | 0 | 0 | % | 100.0% |
| wbc | 1,425 | 5 | 0 | 0 | K/uL | 100.0% |
| bilirubin | 527 | 3 | 0 | 0 | mg/dL | 100.0% |
| hemoglobin | 1,431 | 3 | 0 | 0 | g/dL | 100.0% |
| albumin | 352 | 0 | 0 | 0 | g/dL | 100.0% |
| anion_gap | 1,528 | 0 | 0 | 0 | mEq/L | 100.0% |
| bun | 1,554 | 0 | 0 | 0 | mg/dL | 100.0% |
| creatinine | 1,557 | 0 | 0 | 0 | mg/dL | 100.0% |
| inr | 951 | 19 | 0 | 1 | — | 100.0% |
| ntprobnp | 246 | 8 | 0 | 0 | pg/mL | 100.0% |
| platelet | 1,435 | 5 | 0 | 0 | K/uL | 100.0% |
| glucose | 1,506 | 0 | 0 | 0 | mg/dL | 100.0% |
| lactate | 678 | 0 | 0 | 0 | mmol/L | 100.0% |
| chloride | 1,561 | 0 | 0 | 0 | mEq/L | 100.0% |
| sodium | 1,564 | 0 | 0 | 0 | mEq/L | 100.0% |
| potassium | 1,584 | 2 | 0 | 0 | mEq/L | 100.0% |
| troponin_t | 764 | 264 | 0 | 0 | ng/mL | 100.0% |
| hematocrit | 1,479 | 3 | 0 | 0 | % | 100.0% |
| bicarbonate | 1,533 | 0 | 0 | 0 | mEq/L | 100.0% |

## Tỷ lệ NULL của từng đặc trưng (toàn bộ)

| Đặc trưng | % NULL |
|---|---:|
| `days_since_last_discharge` | 26.4% |
| `omr_bmi_last` | 54.2% |
| `omr_weight_last` | 49.8% |
| `omr_sbp_last` | 40.1% |
| `omr_egfr_last` | 100.0% |
| `ntprobnp_first` | 68.2% |
| `ntprobnp_min` | 68.2% |
| `ntprobnp_max` | 68.2% |
| `creatinine_first` | 3.8% |
| `creatinine_min` | 3.8% |
| `creatinine_max` | 3.8% |
| `bun_first` | 3.6% |
| `bun_min` | 3.6% |
| `bun_max` | 3.6% |
| `sodium_first` | 4.6% |
| `sodium_min` | 4.6% |
| `sodium_max` | 4.6% |
| `potassium_first` | 4.0% |
| `potassium_min` | 4.0% |
| `potassium_max` | 4.0% |
| `chloride_first` | 4.3% |
| `chloride_min` | 4.3% |
| `chloride_max` | 4.3% |
| `bicarbonate_first` | 4.7% |
| `bicarbonate_min` | 4.7% |
| `bicarbonate_max` | 4.7% |
| `anion_gap_first` | 4.7% |
| `anion_gap_min` | 4.7% |
| `anion_gap_max` | 4.7% |
| `glucose_first` | 5.0% |
| `glucose_min` | 5.0% |
| `glucose_max` | 5.0% |
| `hemoglobin_first` | 4.0% |
| `hemoglobin_min` | 4.0% |
| `hemoglobin_max` | 4.0% |
| `hematocrit_first` | 3.5% |
| `hematocrit_min` | 3.5% |
| `hematocrit_max` | 3.5% |
| `wbc_first` | 4.0% |
| `wbc_min` | 4.0% |
| `wbc_max` | 4.0% |
| `platelet_first` | 3.6% |
| `platelet_min` | 3.6% |
| `platelet_max` | 3.6% |
| `rdw_first` | 4.2% |
| `rdw_min` | 4.2% |
| `rdw_max` | 4.2% |
| `troponin_t_first` | 65.9% |
| `troponin_t_min` | 65.9% |
| `troponin_t_max` | 65.9% |
| `lactate_first` | 54.8% |
| `lactate_min` | 54.8% |
| `lactate_max` | 54.8% |
| `albumin_first` | 61.2% |
| `albumin_min` | 61.2% |
| `albumin_max` | 61.2% |
| `magnesium_first` | 10.6% |
| `magnesium_min` | 10.6% |
| `magnesium_max` | 10.6% |
| `bilirubin_first` | 48.2% |
| `bilirubin_min` | 48.2% |
| `bilirubin_max` | 48.2% |
| `inr_first` | 21.9% |
| `inr_min` | 21.9% |
| `inr_max` | 21.9% |
| `hr_mean` | 75.6% |
| `hr_max` | 75.6% |
| `sbp_min` | 77.8% |
| `sbp_mean` | 77.8% |
| `dbp_mean` | 77.8% |
| `mbp_min` | 77.9% |
| `rr_mean` | 75.6% |
| `rr_max` | 75.6% |
| `spo2_min` | 75.6% |
| `temp_max_c` | 76.7% |

Chỉ chứa số liệu tổng hợp, không có dữ liệu cá nhân.

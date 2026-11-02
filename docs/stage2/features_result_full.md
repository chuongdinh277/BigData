# Kết quả bảng đặc trưng P1 (Giai đoạn 2)

- Phiên bản dữ liệu: `v1`; mẫu: toàn bộ
- Số ca trong cohort sau quy tắc P1 (còn nằm viện, chưa tử vong tại 24h): 73,768
- Số dòng Gold: 73,768; số bệnh nhân: 30,287
- Ca dương (tử vong trong viện): 3,588 (4.86%)
- Nằm ICU trong 24h đầu: 16,939 (23.0%); thiếu toàn bộ sinh hiệu: 57,089
- Sinh hiệu: 2,089,983 phép đo trong cửa sổ trước khi lọc khoảng hợp lệ

## Loại bỏ theo từng xét nghiệm (cửa sổ -6h → 24h)

| Lab | Số phép đo | Không phải số | Sai đơn vị | Ngoài khoảng | Đơn vị chủ đạo | Tỷ lệ đơn vị |
|---|---:|---:|---:|---:|---|---:|
| magnesium | 119,852 | 17 | 0 | 21 | mg/dL | 100.0% |
| rdw | 139,901 | 511 | 0 | 0 | % | 100.0% |
| wbc | 140,088 | 469 | 0 | 4 | K/uL | 100.0% |
| bilirubin | 50,426 | 517 | 0 | 9 | mg/dL | 100.0% |
| hemoglobin | 140,755 | 411 | 0 | 0 | g/dL | 100.0% |
| albumin | 33,442 | 6 | 0 | 3 | g/dL | 100.0% |
| anion_gap | 150,084 | 6 | 0 | 3 | mEq/L | 100.0% |
| bun | 152,623 | 13 | 0 | 0 | mg/dL | 100.0% |
| creatinine | 153,122 | 19 | 0 | 12 | mg/dL | 100.0% |
| inr | 97,040 | 1,373 | 0 | 14 | — | 100.0% |
| ntprobnp | 24,039 | 550 | 0 | 2 | pg/mL | 100.0% |
| platelet | 141,243 | 619 | 0 | 5 | K/uL | 100.0% |
| glucose | 147,179 | 14 | 0 | 2 | mg/dL | 100.0% |
| chloride | 152,870 | 29 | 0 | 1 | mEq/L | 100.0% |
| lactate | 64,076 | 28 | 0 | 3 | mmol/L | 100.0% |
| sodium | 153,151 | 23 | 0 | 2 | mEq/L | 100.0% |
| potassium | 154,996 | 154 | 0 | 1 | mEq/L | 100.0% |
| troponin_t | 77,209 | 24,073 | 0 | 1 | ng/mL | 100.0% |
| hematocrit | 146,227 | 563 | 0 | 0 | % | 100.0% |
| bicarbonate | 150,521 | 42 | 0 | 3 | mEq/L | 100.0% |

## Tỷ lệ NULL của từng đặc trưng (toàn bộ)

| Đặc trưng | % NULL |
|---|---:|
| `days_since_last_discharge` | 25.2% |
| `omr_bmi_last` | 51.5% |
| `omr_weight_last` | 46.1% |
| `omr_sbp_last` | 41.1% |
| `omr_egfr_last` | 99.9% |
| `ntprobnp_first` | 68.9% |
| `ntprobnp_min` | 68.9% |
| `ntprobnp_max` | 68.9% |
| `creatinine_first` | 3.5% |
| `creatinine_min` | 3.5% |
| `creatinine_max` | 3.5% |
| `bun_first` | 3.6% |
| `bun_min` | 3.6% |
| `bun_max` | 3.6% |
| `sodium_first` | 3.9% |
| `sodium_min` | 3.9% |
| `sodium_max` | 3.9% |
| `potassium_first` | 3.6% |
| `potassium_min` | 3.6% |
| `potassium_max` | 3.6% |
| `chloride_first` | 3.7% |
| `chloride_min` | 3.7% |
| `chloride_max` | 3.7% |
| `bicarbonate_first` | 3.9% |
| `bicarbonate_min` | 3.9% |
| `bicarbonate_max` | 3.9% |
| `anion_gap_first` | 4.0% |
| `anion_gap_min` | 4.0% |
| `anion_gap_max` | 4.0% |
| `glucose_first` | 4.6% |
| `glucose_min` | 4.6% |
| `glucose_max` | 4.6% |
| `hemoglobin_first` | 4.2% |
| `hemoglobin_min` | 4.2% |
| `hemoglobin_max` | 4.2% |
| `hematocrit_first` | 3.8% |
| `hematocrit_min` | 3.8% |
| `hematocrit_max` | 3.8% |
| `wbc_first` | 4.2% |
| `wbc_min` | 4.2% |
| `wbc_max` | 4.2% |
| `platelet_first` | 4.1% |
| `platelet_min` | 4.1% |
| `platelet_max` | 4.1% |
| `rdw_first` | 4.3% |
| `rdw_min` | 4.3% |
| `rdw_max` | 4.3% |
| `troponin_t_first` | 66.1% |
| `troponin_t_min` | 66.1% |
| `troponin_t_max` | 66.1% |
| `lactate_first` | 56.3% |
| `lactate_min` | 56.3% |
| `lactate_max` | 56.3% |
| `albumin_first` | 61.8% |
| `albumin_min` | 61.8% |
| `albumin_max` | 61.8% |
| `magnesium_first` | 11.1% |
| `magnesium_min` | 11.1% |
| `magnesium_max` | 11.1% |
| `bilirubin_first` | 50.1% |
| `bilirubin_min` | 50.1% |
| `bilirubin_max` | 50.1% |
| `inr_first` | 21.9% |
| `inr_min` | 21.9% |
| `inr_max` | 21.9% |
| `hr_mean` | 77.4% |
| `hr_max` | 77.4% |
| `sbp_min` | 79.4% |
| `sbp_mean` | 79.4% |
| `dbp_mean` | 79.4% |
| `mbp_min` | 79.4% |
| `rr_mean` | 77.4% |
| `rr_max` | 77.4% |
| `spo2_min` | 77.4% |
| `temp_max_c` | 78.8% |

Chỉ chứa số liệu tổng hợp, không có dữ liệu cá nhân.

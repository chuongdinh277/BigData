# Báo cáo chất lượng Giai đoạn 2 — Chiến

Ngày: 27/10/2026 · Nhánh: `chien/quality`

**Phạm vi: synthetic. Dữ liệu MIMIC thật và ghi S3 chưa chạy.**

Nguồn kiểm chứng: GitHub Actions, commit `1795c3d5bba84d16ce2aa421b70150b67a862140`; Python 3.11.17, Spark 3.5.3.

Bằng chứng thực thi: [CI thành công](https://github.com/chuongdinh277/BigData/actions/runs/40448883500); artifact `quality-evidence` chứa JSON và kết quả pytest.

## Kết quả thực thi

| Hạng mục | Kết quả |
|---|---|
| Sinh MIMIC giả, seed 42 | 800 bệnh nhân, 10 bảng CSV gzip |
| Khảo sát stage1, gồm chartevents | 20 mã lab ứng viên; chạy thành công |
| Kiểm tra đơn vị | Lab: PASSED; vital: PASSED |
| Fixture P1 từ các event giả đã lọc | 100 dòng × 101 cột; PASSED |
| Trùng khóa ở fixture sạch | 0 dòng |
| Giá trị ngoài khoảng ở fixture sạch | 0 |
| Thời gian và leakage fixture sạch | PASSED / PASSED |
| Fixture cố tình sai | FAILED; CLI trả exit code 1 đúng dự kiến |
| Pytest | 48 tests; 0 failures, 0 errors, 0 skipped |

## Các lỗi cố tình đưa vào và đã phát hiện

| Kiểm tra | Số phát hiện |
|---|---:|
| Dòng nằm trong nhóm trùng hadm_id | 2 |
| sodium_first ngoài khoảng | 2 |
| Event sau mốc dự đoán | 2 |
| Kết quả có sẵn sau mốc dự đoán | 2 |

Các số đếm leakage có thể tăng do fixture cố tình nhân đôi admission.

## Mã xét nghiệm, đơn vị và độ phủ giả

Độ phủ dưới đây thuộc khảo sát synthetic, trước áp dụng bộ lọc chặt storetime/range của fixture quality.

| Feature | itemid | Đơn vị | Khoảng hợp lệ | Độ phủ P1 giả |
|---|---|---|---|---:|
| ntprobnp | [50963] | pg/mL | [5, 70000] | 0.2331 |
| creatinine | [50912] | mg/dL | [0.1, 25] | 0.559 |
| bun | [51006] | mg/dL | [1, 300] | 0.5768 |
| sodium | [50983] | mEq/L | [95, 185] | 0.5674 |
| potassium | [50971] | mEq/L | [1.5, 10] | 0.5365 |
| chloride | [50902] | mEq/L | [60, 150] | 0.5272 |
| bicarbonate | [50882] | mEq/L | [3, 60] | 0.5496 |
| anion_gap | [50868] | mEq/L | [0, 60] | 0.5459 |
| glucose | [50931] | mg/dL | [10, 2000] | 0.5309 |
| hemoglobin | [51222] | g/dL | [2, 25] | 0.5609 |
| hematocrit | [51221] | % | [5, 75] | 0.5787 |
| wbc | [51301] | K/uL | [0.1, 300] | 0.559 |
| platelet | [51265] | K/uL | [1, 2000] | 0.589 |
| rdw | [51277] | % | [5, 40] | 0.5833 |
| troponin_t | [51003] | ng/mL | [0, 50] | 0.5552 |
| lactate | [50813] | mmol/L | [0.1, 30] | 0.5777 |
| albumin | [50862] | g/dL | [0.5, 7] | 0.573 |
| magnesium | [50960] | mg/dL | [0.3, 10] | 0.5459 |
| bilirubin | [50885] | mg/dL | [0.05, 60] | 0.559 |
| inr | [51237] | không có | [0.5, 20] | 0.5515 |

## Chất lượng event đầu vào giả

Đây là thống kê trước lọc. Các giá trị ngoài khoảng được loại khi tạo fixture; không được coi là giá trị hợp lệ chỉ vì đơn vị đã đạt.

| Nhóm/feature | Số event | Sai đơn vị | Thiếu numeric | Numeric-text | Ngoài khoảng đúng đơn vị |
|---|---:|---:|---:|---:|---:|
| ntprobnp | 1343 | 0 | 0 | 0 | 131 |
| creatinine | 3154 | 0 | 0 | 0 | 92 |
| bun | 3151 | 0 | 0 | 0 | 68 |
| sodium | 3219 | 0 | 0 | 0 | 25 |
| potassium | 3120 | 0 | 0 | 0 | 0 |
| chloride | 3129 | 0 | 0 | 0 | 0 |
| bicarbonate | 3088 | 0 | 0 | 0 | 0 |
| anion_gap | 3160 | 0 | 0 | 0 | 0 |
| glucose | 3065 | 0 | 0 | 0 | 11 |
| hemoglobin | 3121 | 0 | 0 | 0 | 0 |
| hematocrit | 3176 | 0 | 0 | 0 | 0 |
| wbc | 3166 | 0 | 0 | 0 | 46 |
| platelet | 3201 | 0 | 0 | 0 | 8 |
| rdw | 3181 | 0 | 0 | 0 | 0 |
| troponin_t | 3116 | 0 | 0 | 0 | 0 |
| lactate | 3151 | 0 | 0 | 0 | 81 |
| albumin | 3126 | 0 | 0 | 0 | 0 |
| magnesium | 3185 | 0 | 0 | 0 | 0 |
| bilirubin | 3132 | 0 | 0 | 0 | 311 |
| inr | 3213 | 0 | 0 | 0 | 109 |
| heart_rate | 7670 | 0 | 0 | 0 | 0 |
| sbp | 7670 | 0 | 0 | 0 | 0 |
| dbp | 7670 | 0 | 0 | 0 | 0 |
| mbp | 7670 | 0 | 0 | 0 | 0 |
| resp_rate | 7670 | 0 | 0 | 0 | 0 |
| spo2 | 7670 | 0 | 0 | 0 | 701 |
| temperature | 7670 | 0 | 0 | 0 | 0 |

## Missing rate từng cột của fixture sạch

Thiếu xét nghiệm/sinh hiệu được giữ NULL, không tự impute. Cột ngoài phạm vi khoảng sinh lý ghi N/A.

| Cột | Số thiếu | % thiếu | Ngoài khoảng |
|---|---:|---:|---:|
| subject_id | 0 | 0.0 | N/A |
| hadm_id | 0 | 0.0 | N/A |
| admittime | 0 | 0.0 | N/A |
| prediction_time | 0 | 0.0 | N/A |
| anchor_year_group | 0 | 0.0 | N/A |
| age | 0 | 0.0 | N/A |
| gender_male | 0 | 0.0 | N/A |
| label_mortality | 0 | 0.0 | N/A |
| ntprobnp_missing | 0 | 0.0 | N/A |
| ntprobnp_first | 77 | 77.0 | 0 |
| ntprobnp_min | 77 | 77.0 | 0 |
| ntprobnp_max | 77 | 77.0 | 0 |
| creatinine_missing | 0 | 0.0 | N/A |
| creatinine_first | 46 | 46.0 | 0 |
| creatinine_min | 46 | 46.0 | 0 |
| creatinine_max | 46 | 46.0 | 0 |
| bun_missing | 0 | 0.0 | N/A |
| bun_first | 51 | 51.0 | 0 |
| bun_min | 51 | 51.0 | 0 |
| bun_max | 51 | 51.0 | 0 |
| sodium_missing | 0 | 0.0 | N/A |
| sodium_first | 56 | 56.0 | 0 |
| sodium_min | 56 | 56.0 | 0 |
| sodium_max | 56 | 56.0 | 0 |
| potassium_missing | 0 | 0.0 | N/A |
| potassium_first | 45 | 45.0 | 0 |
| potassium_min | 45 | 45.0 | 0 |
| potassium_max | 45 | 45.0 | 0 |
| chloride_missing | 0 | 0.0 | N/A |
| chloride_first | 46 | 46.0 | 0 |
| chloride_min | 46 | 46.0 | 0 |
| chloride_max | 46 | 46.0 | 0 |
| bicarbonate_missing | 0 | 0.0 | N/A |
| bicarbonate_first | 49 | 49.0 | 0 |
| bicarbonate_min | 49 | 49.0 | 0 |
| bicarbonate_max | 49 | 49.0 | 0 |
| anion_gap_missing | 0 | 0.0 | N/A |
| anion_gap_first | 41 | 41.0 | 0 |
| anion_gap_min | 41 | 41.0 | 0 |
| anion_gap_max | 41 | 41.0 | 0 |
| glucose_missing | 0 | 0.0 | N/A |
| glucose_first | 47 | 47.0 | 0 |
| glucose_min | 47 | 47.0 | 0 |
| glucose_max | 47 | 47.0 | 0 |
| hemoglobin_missing | 0 | 0.0 | N/A |
| hemoglobin_first | 46 | 46.0 | 0 |
| hemoglobin_min | 46 | 46.0 | 0 |
| hemoglobin_max | 46 | 46.0 | 0 |
| hematocrit_missing | 0 | 0.0 | N/A |
| hematocrit_first | 43 | 43.0 | 0 |
| hematocrit_min | 43 | 43.0 | 0 |
| hematocrit_max | 43 | 43.0 | 0 |
| wbc_missing | 0 | 0.0 | N/A |
| wbc_first | 53 | 53.0 | 0 |
| wbc_min | 53 | 53.0 | 0 |
| wbc_max | 53 | 53.0 | 0 |
| platelet_missing | 0 | 0.0 | N/A |
| platelet_first | 47 | 47.0 | 0 |
| platelet_min | 47 | 47.0 | 0 |
| platelet_max | 47 | 47.0 | 0 |
| rdw_missing | 0 | 0.0 | N/A |
| rdw_first | 36 | 36.0 | 0 |
| rdw_min | 36 | 36.0 | 0 |
| rdw_max | 36 | 36.0 | 0 |
| troponin_t_missing | 0 | 0.0 | N/A |
| troponin_t_first | 46 | 46.0 | 0 |
| troponin_t_min | 46 | 46.0 | 0 |
| troponin_t_max | 46 | 46.0 | 0 |
| lactate_missing | 0 | 0.0 | N/A |
| lactate_first | 50 | 50.0 | 0 |
| lactate_min | 50 | 50.0 | 0 |
| lactate_max | 50 | 50.0 | 0 |
| albumin_missing | 0 | 0.0 | N/A |
| albumin_first | 46 | 46.0 | 0 |
| albumin_min | 46 | 46.0 | 0 |
| albumin_max | 46 | 46.0 | 0 |
| magnesium_missing | 0 | 0.0 | N/A |
| magnesium_first | 47 | 47.0 | 0 |
| magnesium_min | 47 | 47.0 | 0 |
| magnesium_max | 47 | 47.0 | 0 |
| bilirubin_missing | 0 | 0.0 | N/A |
| bilirubin_first | 54 | 54.0 | 0 |
| bilirubin_min | 54 | 54.0 | 0 |
| bilirubin_max | 54 | 54.0 | 0 |
| inr_missing | 0 | 0.0 | N/A |
| inr_first | 53 | 53.0 | 0 |
| inr_min | 53 | 53.0 | 0 |
| inr_max | 53 | 53.0 | 0 |
| n_lab_tests_24h | 0 | 0.0 | N/A |
| in_icu_24h | 0 | 0.0 | N/A |
| hr_mean | 62 | 62.0 | 0 |
| hr_max | 62 | 62.0 | 0 |
| sbp_min | 62 | 62.0 | 0 |
| sbp_mean | 62 | 62.0 | 0 |
| dbp_mean | 62 | 62.0 | 0 |
| mbp_min | 62 | 62.0 | 0 |
| rr_mean | 62 | 62.0 | 0 |
| rr_max | 62 | 62.0 | 0 |
| spo2_min | 62 | 62.0 | 0 |
| temp_max_c | 62 | 62.0 | 0 |
| vitals_missing | 0 | 0.0 | N/A |

## Bàn giao và phần còn cần dữ liệu thật

- Cấu hình 20 lab + 7 nhóm vital dùng được để code song song; trạng thái `synthetic_verified_pending_real`.
- Fixture chỉ là tập con feature P1 để kiểm chứng quality, không phải cohort HF hoặc Dataset V1 của nhóm.
- Chờ khảo sát dictionary/đơn vị/độ phủ thật để chốt config thật; chờ cohort và feature/lineage của Hùng, Khoẻ để chạy lại quality.
- Chưa ghi báo cáo lên S3 và chưa đánh dấu Dataset V1 đã chốt.
- Missing theo từng nhóm năm và chi tiết từng phép kiểm có trong JSON tổng hợp ở artifact `quality-evidence`.
- Lệnh tái lập và hợp đồng context/lineage: [quality_usage.md](quality_usage.md).

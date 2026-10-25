# Đặc tả đặc trưng và quy tắc dữ liệu hợp lệ

> **Chủ sở hữu:** TV4 · **Trạng thái:** v0.1 · Áp dụng trước tiên cho **P1** (mốc `admittime + 24h`). P2/P3 dùng cùng danh sách nhưng cửa sổ kết thúc tại `dischtime` (xem mục 5).
> Danh sách item: `configs/feature_items.yaml`. itemid cuối cùng được chọn từ output `lab_item_candidates` của script Giai đoạn 1.

## 1. Quy ước chung

- Mỗi đặc trưng có: tên, bảng/cột nguồn, cửa sổ, đơn vị, phép tổng hợp, xử lý thiếu, ý nghĩa tham khảo, cảnh báo leakage.
- **Thiếu ≠ bình thường.** Mọi xét nghiệm có thêm cột `<lab>_missing` (1 = không đo trong cửa sổ).
- Không điền thiếu ở bước Spark. Imputation chỉ fit trên **train** ở GĐ3/4.
- Giá trị ngoài khoảng sinh lý (mục 4) được gán NULL và đếm trong báo cáo quality.

## 2. Danh sách đặc trưng cho P1 (≈ 95 cột)

### 2.1 Nhân khẩu học và nhập viện

| Tên | Nguồn | Quy tắc | Thiếu | Leakage |
|---|---|---|---|---|
| `age` | patients + admissions | `anchor_age + year(admittime) − anchor_year` | không thiếu | an toàn |
| `gender_male` | `patients.gender` | M = 1 | không | an toàn |
| `admission_type_group` | `admissions.admission_type` | EMERGENCY (`EW EMER.`, `DIRECT EMER.`) / URGENT / OBSERVATION (`*OBSERVATION*`) / ELECTIVE (`ELECTIVE`, `SURGICAL SAME DAY ADMISSION`) | category `UNKNOWN` | an toàn |
| `came_from_ed` | `admissions.edregtime` | 1 nếu không NULL | 0 | an toàn |
| `admission_location_group` | `admissions.admission_location` | gộp nhóm EMERGENCY ROOM / TRANSFER / REFERRAL / OTHER | `UNKNOWN` | an toàn |
| `in_icu_24h` | `icustays` | 1 nếu có `intime ≤ prediction_time` và `outtime ≥ admittime` | 0 | an toàn |

### 2.2 Tiền sử (chỉ từ admission có `dischtime < admittime` hiện tại)

| Tên | Quy tắc |
|---|---|
| `n_prior_adm_365d` | Số admission trong 365 ngày trước |
| `n_prior_hf_adm` | Số admission HF trước đó (mọi thời điểm) |
| `days_since_last_discharge` | NULL nếu chưa từng nhập viện, kèm cờ `first_admission` |
| `hx_ckd`, `hx_diabetes`, `hx_afib`, `hx_copd`, `hx_hypertension`, `hx_cad`, `hx_prior_mi`, `hx_stroke` | 1 nếu mã có ở **admission trước**, theo `configs/hf_codes.yaml → comorbidities` |
| `omr_bmi_last`, `omr_weight_last`, `omr_sbp_last`, `omr_egfr_last` | Giá trị OMR gần nhất có `chartdate < date(admittime)` trong 365 ngày, kèm cờ `_missing` |

### 2.3 Xét nghiệm — cửa sổ `[admittime − 6h, admittime + 24h]`

Với mỗi xét nghiệm có 4 cột: `<lab>_first`, `<lab>_min`, `<lab>_max`, `<lab>_missing`. Riêng `<lab>_last` để dành cho P2/P3.

| Lab | label (d_labitems, fluid Blood) | Đơn vị | Khoảng hợp lệ | Ý nghĩa tham khảo |
|---|---|---|---|---|
| `ntprobnp` | NTproBNP | pg/mL | 5 – 70 000 | Mức độ quá tải/suy tim |
| `creatinine` | Creatinine | mg/dL | 0.1 – 25 | Chức năng thận |
| `bun` | Urea Nitrogen | mg/dL | 1 – 300 | Chức năng thận, tiên lượng HF |
| `sodium` | Sodium | mEq/L | 95 – 185 | Hạ natri là yếu tố tiên lượng xấu |
| `potassium` | Potassium | mEq/L | 1.5 – 10 | Rối loạn điện giải |
| `chloride` | Chloride | mEq/L | 60 – 150 | |
| `bicarbonate` | Bicarbonate | mEq/L | 3 – 60 | Toan chuyển hóa |
| `anion_gap` | Anion Gap | mEq/L | 0 – 60 | |
| `glucose` | Glucose | mg/dL | 10 – 2 000 | |
| `hemoglobin` | Hemoglobin | g/dL | 2 – 25 | Thiếu máu |
| `hematocrit` | Hematocrit | % | 5 – 75 | |
| `wbc` | White Blood Cells | K/uL | 0.1 – 300 | Nhiễm trùng/viêm |
| `platelet` | Platelet Count | K/uL | 1 – 2 000 | |
| `rdw` | RDW | % | 5 – 40 | |
| `troponin_t` | Troponin T | ng/mL | 0 – 50 | Tổn thương cơ tim |
| `lactate` | Lactate | mmol/L | 0.1 – 30 | Giảm tưới máu |
| `albumin` | Albumin | g/dL | 0.5 – 7 | Dinh dưỡng, tiên lượng |
| `magnesium` | Magnesium | mg/dL | 0.3 – 10 | |
| `bilirubin` | Bilirubin, Total | mg/dL | 0.05 – 60 | Sung huyết gan |
| `inr` | INR(PT) | — | 0.5 – 20 | Đông máu/sung huyết gan |

Thêm: `n_lab_tests_24h` (tổng số lần xét nghiệm, phản ánh mức độ theo dõi; cần cân nhắc vì có thể phản ánh hành vi bác sĩ).

### 2.4 Sinh hiệu ICU — cửa sổ `[admittime, admittime + 24h]`, chỉ ca có `in_icu_24h = 1`

| Tên | Item (d_items) | Tổng hợp | Khoảng hợp lệ |
|---|---|---|---|
| `hr_mean`, `hr_max` | Heart Rate | mean, max | 20 – 250 |
| `sbp_min`, `sbp_mean` | NIBP/ABP systolic | min, mean | 40 – 300 |
| `dbp_mean` | NIBP/ABP diastolic | mean | 10 – 200 |
| `mbp_min` | NIBP/ABP mean | min | 20 – 250 |
| `rr_mean`, `rr_max` | Respiratory Rate | mean, max | 3 – 70 |
| `spo2_min` | O2 saturation pulseoxymetry | min | 50 – 100 |
| `temp_max_c` | Temperature F/C (quy đổi về °C) | max | 30 – 43 °C |
| `vitals_missing` | — | 1 nếu không có ICU/không đo | — |

> ⚠️ Sinh hiệu chỉ có ở khoảng một phần cohort (ca ICU). Phải báo cáo độ phủ, và cân nhắc biến thể mô hình **không dùng sinh hiệu** để so sánh (đây là case cho TV4/TV3 ở GĐ4).

## 3. Danh sách rò rỉ bị cấm (P1)

| # | Biến / nguồn | Lý do |
|---|---|---|
| L1 | `diagnoses_icd`, `procedures_icd`, `drgcodes` **của admission hiện tại** | Gán lúc xuất viện, phản ánh toàn bộ diễn biến |
| L2 | `dischtime`, LOS, `discharge_location` | Biết sau mốc dự đoán; `DIED` chính là nhãn |
| L3 | `deathtime`, `hospital_expire_flag`, `patients.dod` | Là nhãn hoặc biết sau |
| L4 | Mọi event có `charttime > admittime + 24h` | Thông tin tương lai |
| L5 | `storetime > prediction_time` dù `charttime` hợp lệ | Kết quả chưa có tại mốc dự đoán. Phương án chặt: lọc theo `storetime` (TV4 thống kê chênh lệch) |
| L6 | `icustays.los`, `outtime`, `last_careunit` | Biết sau |
| L7 | Bất kỳ thông tin nào từ admission **sau** admission hiện tại | Thời gian đảo ngược |
| L8 | Đặc trưng tổng hợp/chuẩn hóa fit trên toàn bộ dataset | Leakage tiền xử lý: chỉ fit trên train |

**Kiểm thử tự động (GĐ2, `tests/test_leakage.py`):** với mỗi bảng đặc trưng, `max(source_charttime) ≤ prediction_time`; không có cột nằm trong danh sách L1–L3, L6.

## 4. Quy tắc dữ liệu hợp lệ

1. Khóa duy nhất: `hadm_id` trong Gold P1.
2. `prediction_time = admittime + 24h`, và `admittime < prediction_time < dischtime`.
3. Mỗi `itemid` chỉ giữ **đơn vị chủ đạo** (≥ 95% số dòng). Đơn vị khác bị loại và đếm số dòng bị loại.
4. Giá trị ngoài khoảng mục 2.3/2.4 được gán NULL.
5. `valuenum` NULL nhưng `value` có chữ (ví dụ "<0.01") được ghi nhận riêng. GĐ2 quyết định parse hay bỏ.
6. Báo cáo missing rate của từng đặc trưng, tổng thể và theo `anchor_year_group`.

## 5. Khác biệt cho P2/P3 (mốc `dischtime`)

- Cửa sổ xét nghiệm: **48h cuối trước `dischtime`** → `<lab>_last`, cộng thêm `<lab>_first` của cả đợt nằm viện.
- **Được dùng:** LOS, `discharge_location_group`, mã ICD của admission hiện tại (đã có lúc xuất viện), số ngày nằm ICU.
- **Vẫn cấm:** mọi thông tin sau `dischtime`, `patients.dod`.

# Từ điển đặc trưng Gold P1

> Nhánh : khoe/features · Mã: `data_spark/features.py` · Schema nguồn: `contracts/gold_schema_p1.json` (v0.1)
> Đơn vị phân tích: **1 dòng / `hadm_id`**. Mốc dự đoán: `prediction_time = admittime + 24h`.
> Quần thể P1: ca trong cohort còn nằm viện (`dischtime > prediction_time`) và chưa tử vong (`deathtime` NULL hoặc `> prediction_time`).
> Tổng cộng 124 cột = 5 định danh + 1 nhãn + 4 chỉ dùng cho subgroup + **114 đặc trưng** mô hình.

## 1. Quy ước chung

| Quy ước              | Nội dung                                                                                                                                                                                     |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chống rò rỉ          | Chỉ dùng dữ liệu có `charttime ≤ prediction_time`; xét nghiệm và sinh hiệu còn phải có `storetime ≤ prediction_time` (NULL → giữ, vì không kiểm được). Bật/tắt bằng `--no-storetime-filter`. |
| Không điền thiếu     | Thiếu để NULL. Imputation chỉ fit trên train ở GĐ3/4.                                                                                                                                        |
| Cờ thiếu             | Mỗi xét nghiệm có `<lab>_missing` (1 = không có phép đo hợp lệ trong cửa sổ).                                                                                                                |
| Giá trị ngoài khoảng | Gán NULL (khoảng trong `feature_items.yaml`). Số bị loại ghi trong `features_result.md`.                                                                                                     |
| Đơn vị               | Mỗi lab chỉ giữ đơn vị chủ đạo; dòng đơn vị khác bị loại và đếm. Cảnh báo nếu đơn vị chủ đạo < 95%.                                                                                          |
| Cấm                  | Không có cột trong `forbidden_columns` (`dischtime`, `deathtime`, `discharge_location`, `hospital_expire_flag`, `los`, `dod`, `los_icu`, `outtime`). Script `assert` điều này.               |

## 2. Định danh, nhãn và cột chỉ dùng cho subgroup

| Cột                     | Nguồn                                                                       | Cách tính                                                                                                                                      | Ghi chú                                |
| ----------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `subject_id`, `hadm_id` | cohort                                                                      | giữ nguyên                                                                                                                                     | `hadm_id` là khóa chính                |
| `admittime`             | `admissions`                                                                | giữ nguyên                                                                                                                                     |                                        |
| `prediction_time`       | tính                                                                        | `admittime + 24h`                                                                                                                              |                                        |
| `anchor_year_group`     | `patients`                                                                  | giữ nguyên                                                                                                                                     | chỉ dùng chia theo thời kỳ/drift       |
| `label_mortality`       | `admissions.hospital_expire_flag` (hoặc bảng nhãn P1 của Hà qua `--labels`) | 1 nếu tử vong trong đợt nằm viện                                                                                                               | **nhãn**, không phải đầu vào           |
| `race_group`            | `admissions.race`                                                           | HISPANIC (chứa HISPANIC/LATINO) → WHITE → BLACK → ASIAN; UNKNOWN (trống, UNKNOWN, UNABLE TO OBTAIN, PATIENT DECLINED TO ANSWER); còn lại OTHER | chỉ đánh giá công bằng                 |
| `insurance`             | `admissions.insurance`                                                      | giữ nguyên                                                                                                                                     | chỉ đánh giá công bằng                 |
| `hf_primary`            | cohort (`diagnoses_icd.seq_num = 1`)                                        | 1 nếu suy tim là chẩn đoán chính                                                                                                               | mã của **đợt hiện tại** → chỉ subgroup |
| `icd_version_group`     | `diagnoses_icd.icd_version` của đợt hiện tại                                | ICD9 / ICD10 / MIXED                                                                                                                           | chỉ subgroup                           |

## 3. Nhân khẩu học và nhập viện (6 cột)

| Cột                        | Nguồn                           | Cách tính                                                                                                                                                                     | Thiếu       |
| -------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| `age`                      | `patients`, `admissions`        | `anchor_age + year(admittime) − anchor_year` (từ cohort)                                                                                                                      | không thiếu |
| `gender_male`              | `patients.gender`               | 1 nếu `M`                                                                                                                                                                     | 0           |
| `admission_type_group`     | `admissions.admission_type`     | chứa OBSERVATION → OBSERVATION; `EW EMER.`/`DIRECT EMER.`/`EMERGENCY` → EMERGENCY; chứa URGENT → URGENT; `ELECTIVE`/`SURGICAL SAME DAY ADMISSION` → ELECTIVE; còn lại UNKNOWN | UNKNOWN     |
| `admission_location_group` | `admissions.admission_location` | chứa EMERGENCY ROOM → EMERGENCY_ROOM; chứa TRANSFER → TRANSFER; chứa REFERRAL → REFERRAL; NULL/INFORMATION NOT AVAILABLE → UNKNOWN; còn lại OTHER                             | UNKNOWN     |
| `came_from_ed`             | `admissions.edregtime`          | 1 nếu không NULL                                                                                                                                                              | 0           |
| `in_icu_24h`               | `icustays`                      | 1 nếu tồn tại `intime ≤ prediction_time` và `outtime ≥ admittime`                                                                                                             | 0           |

## 4. Tiền sử (16 cột) — chỉ từ các đợt **trước** (`dischtime < admittime` hiện tại)

| Cột                                                                                                    | Cách tính                                                                                              |
| ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| `n_prior_adm_365d`                                                                                     | Số đợt trước có `dischtime` nằm trong 365 ngày trước `admittime` hiện tại (mọi đợt, không chỉ suy tim) |
| `n_prior_hf_adm`                                                                                       | Số đợt trước có mã HF (bộ mã `hf_codes.yaml`, mọi `seq_num`), mọi thời điểm                            |
| `first_admission`                                                                                      | 1 nếu chưa có đợt trước nào                                                                            |
| `days_since_last_discharge`                                                                            | Số ngày (thực) từ `dischtime` gần nhất đến `admittime`; NULL nếu `first_admission = 1`                 |
| `hx_ckd`, `hx_diabetes`, `hx_afib`, `hx_copd`, `hx_hypertension`, `hx_cad`, `hx_prior_mi`, `hx_stroke` | 1 nếu **bất kỳ đợt trước** có mã trong `hf_codes.yaml → comorbidities` (khớp tiền tố)                  |
| `omr_bmi_last`                                                                                         | BMI (kg/m²) gần nhất, `chartdate < ngày nhập viện`, trong 365 ngày; hợp lệ 10–100                      |
| `omr_weight_last`                                                                                      | Cân nặng gần nhất, **quy đổi lb → kg** (× 0.45359237); hợp lệ 20–400 kg                                |
| `omr_sbp_last`                                                                                         | Huyết áp tâm thu (phần trước dấu `/` của "Blood Pressure"); hợp lệ 40–300 mmHg                         |
| `omr_egfr_last`                                                                                        | eGFR gần nhất; chỉ nhận giá trị số thuần (">60" → bỏ); hợp lệ 0–200                                    |

Giá trị OMR chỉ lấy khi hợp lệ, nên "gần nhất" nghĩa là gần nhất **trong các giá trị hợp lệ**.

## 5. Xét nghiệm (81 cột) — cửa sổ `[admittime − 6h, admittime + 24h]`

Mỗi lab có 4 cột: `<lab>_first` (phép đo sớm nhất theo `charttime`), `<lab>_min`, `<lab>_max`, `<lab>_missing`.
Nguồn: `labevents` nối với cohort theo `subject_id` + thời gian (xét nghiệm lấy ở ED trước khi nhập viện có thể chưa có `hadm_id`). Chỉ dùng `valuenum`; giá trị chữ như "<0.01" bị loại và đếm.

| Lab           | itemid | Đơn vị | Khoảng hợp lệ |
| ------------- | ------ | ------ | ------------- |
| `ntprobnp`    | 50963  | pg/mL  | 5 – 70 000    |
| `creatinine`  | 50912  | mg/dL  | 0.1 – 25      |
| `bun`         | 51006  | mg/dL  | 1 – 300       |
| `sodium`      | 50983  | mEq/L  | 95 – 185      |
| `potassium`   | 50971  | mEq/L  | 1.5 – 10      |
| `chloride`    | 50902  | mEq/L  | 60 – 150      |
| `bicarbonate` | 50882  | mEq/L  | 3 – 60        |
| `anion_gap`   | 50868  | mEq/L  | 0 – 60        |
| `glucose`     | 50931  | mg/dL  | 10 – 2 000    |
| `hemoglobin`  | 51222  | g/dL   | 2 – 25        |
| `hematocrit`  | 51221  | %      | 5 – 75        |
| `wbc`         | 51301  | K/uL   | 0.1 – 300     |
| `platelet`    | 51265  | K/uL   | 1 – 2 000     |
| `rdw`         | 51277  | %      | 5 – 40        |
| `troponin_t`  | 51003  | ng/mL  | 0 – 50        |
| `lactate`     | 50813  | mmol/L | 0.1 – 30      |
| `albumin`     | 50862  | g/dL   | 0.5 – 7       |
| `magnesium`   | 50960  | mg/dL  | 0.3 – 10      |
| `bilirubin`   | 50885  | mg/dL  | 0.05 – 60     |
| `inr`         | 51237  | —      | 0.5 – 20      |

Thêm `n_lab_tests_24h`: **tổng số dòng `labevents` (mọi itemid)** trong cùng cửa sổ. Biến này có thể phản ánh hành vi bác sĩ hơn là tình trạng bệnh nhân; nên có case "bỏ biến này" ở GĐ4.

## 6. Sinh hiệu ICU (11 cột) — cửa sổ `[admittime, admittime + 24h]`

Nguồn `chartevents`, chỉ có với ca nằm ICU. Phép đo ngoài khoảng hợp lệ bị loại (nhiệt độ kiểm **sau** khi quy đổi).

| Cột                   | itemid      | Tổng hợp                          | Khoảng hợp lệ |
| --------------------- | ----------- | --------------------------------- | ------------- |
| `hr_mean`, `hr_max`   | 220045      | trung bình, lớn nhất              | 20 – 250 bpm  |
| `sbp_min`, `sbp_mean` | 220179      | nhỏ nhất, trung bình              | 40 – 300 mmHg |
| `dbp_mean`            | 220180      | trung bình                        | 10 – 200 mmHg |
| `mbp_min`             | 220181      | nhỏ nhất                          | 20 – 250 mmHg |
| `rr_mean`, `rr_max`   | 220210      | trung bình, lớn nhất              | 3 – 70        |
| `spo2_min`            | 220277      | nhỏ nhất                          | 50 – 100 %    |
| `temp_max_c`          | 223761 (°F) | lớn nhất sau `(F − 32) × 5/9`     | 30 – 43 °C    |
| `vitals_missing`      | —           | 1 nếu **cả 10 cột trên** đều NULL | —             |

## 7. Điểm cần lưu ý khi dùng

- `itemid` trong `feature_items.yaml` đang ở trạng thái `synthetic_verified_pending_real`. Script tự đối chiếu với `d_labitems`/`d_items` thật trước khi chạy và dừng nếu lệch.
- Sinh hiệu chỉ dùng NIBP (220179/220180/220181) và nhiệt độ °F (223761) theo config; các item huyết áp động mạch xâm lấn và nhiệt độ °C chưa được đưa vào.
- Ca không vào ICU thì toàn bộ sinh hiệu NULL và `vitals_missing = 1`. Cần báo cáo độ phủ và có biến thể mô hình không dùng sinh hiệu.

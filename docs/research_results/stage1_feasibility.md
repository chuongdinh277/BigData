# Báo cáo khả thi Giai đoạn 1

Tạo lúc: 2026-10-31T10:30:22.878264+00:00 · Nguồn: `s3a://heartguard-mimic-group11/silver/mimiciv/3.1` (silver, MIMIC-IV v3.1)

> Chỉ chứa số liệu tổng hợp; ô nhỏ hơn ngưỡng đã được che.

## Cohort chung

| step | admissions | patients |
|---|---|---|
| S0 all admissions | 546028 | 223452 |
| S1 HF code any position | 80767 | 31428 |
| S2 valid admit/discharge times | 80746 | 31423 |
| S3 age >= 18 | 80746 | 31423 |


Phương án B (HF là chẩn đoán chính): {'admissions': 15105, 'patients': 8443}


Theo phiên bản ICD:

| icd_version_group | admissions |
|---|---|
| ICD9 | 37613 |
| ICD10 | 43131 |
| MIXED | <10 |


## P1 — tử vong trong viện

| step | admissions | patients |
|---|---|---|
| cohort | 80746 | 31423 |
| still admitted at +24h | 73769 | 30287 |
| alive at +24h | 73768 | 30287 |


**Tổng hợp:** {'admissions': 73768, 'patients': 30287, 'positives': 3588, 'positive_rate': 0.0486}


Nhãn lệch (expire_flag=1 nhưng không có deathtime/DIED): 0


by_gender:
| gender | n | positives | positive_rate |
|---|---|---|---|
| F | 34780 | 1656 | 0.0476 |
| M | 38988 | 1932 | 0.0496 |


by_age_group:
| age_group | n | positives | positive_rate |
|---|---|---|---|
| 18-64 | 18110 | 537 | 0.0297 |
| 65-79 | 28119 | 1378 | 0.049 |
| 80+ | 27539 | 1673 | 0.0608 |


by_hf_primary:
| hf_primary | n | positives | positive_rate |
|---|---|---|---|
| 0 | 59472 | 3175 | 0.0534 |
| 1 | 14296 | 413 | 0.0289 |


Theo anchor_year_group (cho drift):
| anchor_year_group | n | positives | positive_rate |
|---|---|---|---|
| 2008 - 2010 | 34553 | 1310 | 0.0379 |
| 2011 - 2013 | 14062 | 683 | 0.0486 |
| 2014 - 2016 | 11267 | 619 | 0.0549 |
| 2017 - 2019 | 8769 | 566 | 0.0645 |
| 2020 - 2022 | 5117 | 410 | 0.0801 |


## P2 — tái nhập 30 ngày

| step | admissions | patients |
|---|---|---|
| cohort | 80746 | 31423 |
| discharged alive | 76741 | 29403 |
| exclude hospice/AMA/acute transfer | 74420 | 28574 |
| exclude death <= 30d without readmission | 72602 | 27844 |


**Tổng hợp:** {'admissions': 72602, 'patients': 27844, 'positives': 17796, 'positive_rate': 0.2451}


Độ nhạy (giữ ca tử vong, nhãn 0): {'admissions': 74420, 'patients': 28574, 'positives': 17796, 'positive_rate': 0.2391}


Theo anchor_year_group (cho drift):
| anchor_year_group | n | positives | positive_rate |
|---|---|---|---|
| 2008 - 2010 | 35248 | 9877 | 0.2802 |
| 2011 - 2013 | 13756 | 3123 | 0.227 |
| 2014 - 2016 | 10875 | 2339 | 0.2151 |
| 2017 - 2019 | 8209 | 1701 | 0.2072 |
| 2020 - 2022 | 4514 | 756 | 0.1675 |


## P3 — tử vong 90 ngày


**Tổng hợp:** {'admissions': 75285, 'patients': 28855, 'positives': 7635, 'positive_rate': 0.1014}


Theo anchor_year_group (cho drift):
| anchor_year_group | n | positives | positive_rate |
|---|---|---|---|
| 2008 - 2010 | 36361 | 3672 | 0.101 |
| 2011 - 2013 | 14282 | 1447 | 0.1013 |
| 2014 - 2016 | 11341 | 1137 | 0.1003 |
| 2017 - 2019 | 8540 | 830 | 0.0972 |
| 2020 - 2022 | 4761 | 549 | 0.1153 |


## Sinh hiệu

P1 có ICU trong 24h đầu: {'admissions': 16939, 'rate': 0.2296}

| feature | admissions | rate |
|---|---|---|
| temperature | 16310 | 0.2211 |
| sbp | 16618 | 0.2253 |
| spo2 | 16723 | 0.2267 |
| heart_rate | 16730 | 0.2268 |
| dbp | 16618 | 0.2253 |
| mbp | 16634 | 0.2255 |
| resp_rate | 16710 | 0.2265 |


Item sinh hiệu ứng viên:
| feature | itemid | label | linksto | unit |
|---|---|---|---|---|
| heart_rate | 220045 | Heart Rate | chartevents | bpm |
| sbp | 220179 | Non Invasive Blood Pressure systolic | chartevents | mmHg |
| sbp | 220050 | Arterial Blood Pressure systolic | chartevents | mmHg |
| dbp | 220051 | Arterial Blood Pressure diastolic | chartevents | mmHg |
| dbp | 220180 | Non Invasive Blood Pressure diastolic | chartevents | mmHg |
| mbp | 220052 | Arterial Blood Pressure mean | chartevents | mmHg |
| mbp | 220181 | Non Invasive Blood Pressure mean | chartevents | mmHg |
| resp_rate | 220210 | Respiratory Rate | chartevents | insp/min |
| spo2 | 220277 | O2 saturation pulseoxymetry | chartevents | % |
| temperature | 223762 | Temperature Celsius | chartevents | °C |
| temperature | 223761 | Temperature Fahrenheit | chartevents | °F |


## Xét nghiệm — độ phủ trong cửa sổ P1
| feature | admissions | coverage | missing_rate |
|---|---|---|---|
| albumin | 28699 | 0.389 | 0.611 |
| anion_gap | 71216 | 0.9654 | 0.0346 |
| bicarbonate | 71235 | 0.9657 | 0.0343 |
| bilirubin | 37399 | 0.507 | 0.493 |
| bun | 71515 | 0.9695 | 0.0305 |
| chloride | 71381 | 0.9676 | 0.0324 |
| creatinine | 71551 | 0.9699 | 0.0301 |
| glucose | 71177 | 0.9649 | 0.0351 |
| hematocrit | 71236 | 0.9657 | 0.0343 |
| hemoglobin | 71063 | 0.9633 | 0.0367 |
| inr | 57898 | 0.7849 | 0.2151 |
| lactate | 32285 | 0.4377 | 0.5623 |
| magnesium | 66329 | 0.8992 | 0.1008 |
| ntprobnp | 23453 | 0.3179 | 0.6821 |
| platelet | 71041 | 0.963 | 0.037 |
| potassium | 71496 | 0.9692 | 0.0308 |
| rdw | 70863 | 0.9606 | 0.0394 |
| sodium | 71275 | 0.9662 | 0.0338 |
| troponin_t | 25259 | 0.3424 | 0.6576 |
| wbc | 70929 | 0.9615 | 0.0385 |


Itemid xét nghiệm ứng viên (TV4 chọn itemid cuối):
| feature | itemid | label | fluid | category | rows_in_window | p1_admissions | p1_coverage | share_rows_hadm_null | share_rows_stored_after_prediction_time | dominant_unit | dominant_unit_share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ntprobnp | 50963 | NTproBNP | Blood | Chemistry | 24027 | 23453 | 0.3179 | 0.6862 | 0.0224 | pg/mL | 1.0 |
| creatinine | 50912 | Creatinine | Blood | Chemistry | 157674 | 71551 | 0.9699 | 0.2701 | 0.029 | mg/dL | 1.0 |
| creatinine | 52546 | Creatinine | Blood | Chemistry | 13 | 13 | 0.0002 | 1.0 | 0.0 | mg/dL | 1.0 |
| bun | 52647 | Urea Nitrogen | Blood | Chemistry | 12 | 12 | 0.0002 | 1.0 | 0.0 | mg/dL | 1.0 |
| bun | 51006 | Urea Nitrogen | Blood | Chemistry | 157120 | 71515 | 0.9695 | 0.271 | 0.0287 | mg/dL | 1.0 |
| sodium | 52623 | Sodium | Blood | Chemistry | 14 | 13 | 0.0002 | 1.0 | 0.0 | mEq/L | 1.0 |
| sodium | 50983 | Sodium | Blood | Chemistry | 157752 | 71275 | 0.9662 | 0.2657 | 0.0293 | mEq/L | 1.0 |
| potassium | 50971 | Potassium | Blood | Chemistry | 159556 | 71496 | 0.9692 | 0.2634 | 0.0295 | mEq/L | 1.0 |
| potassium | 52610 | Potassium | Blood | Chemistry | 14 | 13 | 0.0002 | 1.0 | 0.0 | mEq/L | 1.0 |
| chloride | 52535 | Chloride | Blood | Chemistry | 12 | 12 | 0.0002 | 1.0 | 0.0 | mEq/L | 1.0 |
| chloride | 50902 | Chloride | Blood | Chemistry | 157430 | 71381 | 0.9676 | 0.2659 | 0.0291 | mEq/L | 1.0 |
| bicarbonate | 50882 | Bicarbonate | Blood | Chemistry | 154929 | 71235 | 0.9657 | 0.2686 | 0.0287 | mEq/L | 1.0 |
| anion_gap | 50868 | Anion Gap | Blood | Chemistry | 154541 | 71216 | 0.9654 | 0.2688 | 0.0289 | mEq/L | 1.0 |
| anion_gap | 52500 | Anion Gap | Blood | Chemistry | 12 | 12 | 0.0002 | 1.0 | 0.0 | mEq/L | 1.0 |
| glucose | 50931 | Glucose | Blood | Chemistry | 151396 | 70774 | 0.9594 | 0.2684 | 0.0279 | mg/dL | 1.0 |
| glucose | 50809 | Glucose | Blood | Blood Gas | 19410 | 8230 | 0.1116 | 0.2202 | 0.0038 | mg/dL | 1.0 |
| glucose | 52569 | Glucose | Blood | Chemistry | 14 | 13 | 0.0002 | 1.0 | 0.0 | mg/dL | 1.0 |
| hemoglobin | 51222 | Hemoglobin | Blood | Hematology | 142446 | 70928 | 0.9615 | 0.2919 | 0.0148 | g/dL | 1.0 |
| hemoglobin | 51640 | Hemoglobin | Blood | Chemistry | 0 | 0 | 0.0 |  |  |  |  |
| hemoglobin | 50811 | Hemoglobin | Blood | Blood Gas | 13217 | 6402 | 0.0868 | 0.2143 | 0.0033 | g/dL | 1.0 |
| hematocrit | 51639 | Hematocrit | Blood | Chemistry | 0 | 0 | 0.0 |  |  |  |  |
| hematocrit | 51221 | Hematocrit | Blood | Hematology | 147945 | 71236 | 0.9657 | 0.2824 | 0.0154 | % | 1.0 |
| hematocrit | 52028 | Hematocrit | Blood | Blood Gas | 0 | 0 | 0.0 |  |  |  |  |
| hematocrit | 51638 | Hematocrit | Blood | Chemistry | 14 | 13 | 0.0002 | 1.0 | 0.0 | % | 1.0 |
| wbc | 51755 | White Blood Cells | Blood | Chemistry | <10 | <10 | 0.0001 | 1.0 | 0.0 | K/uL | 1.0 |
| wbc | 51301 | White Blood Cells | Blood | Hematology | 141711 | 70929 | 0.9615 | 0.2931 | 0.0148 | K/uL | 1.0 |
| wbc | 51756 | White Blood Cells | Blood | Chemistry | 0 | 0 | 0.0 |  |  |  |  |
| platelet | 51265 | Platelet Count | Blood | Hematology | 142770 | 71041 | 0.963 | 0.2911 | 0.015 | K/uL | 1.0 |
| platelet | 53189 | Platelet Count | Blood | Chemistry | 0 | 0 | 0.0 |  |  |  |  |
| rdw | 51277 | RDW | Blood | Hematology | 141439 | 70863 | 0.9606 | 0.2933 | 0.0145 | % | 1.0 |
| troponin_t | 51003 | Troponin T | Blood | Chemistry | 54523 | 25259 | 0.3424 | 0.3428 | 0.0254 | ng/mL | 1.0 |
| lactate | 52442 | Lactate | Blood | Blood Gas | 0 | 0 | 0.0 |  |  |  |  |
| lactate | 50813 | Lactate | Blood | Blood Gas | 64216 | 32285 | 0.4377 | 0.3816 | 0.0026 | mmol/L | 1.0 |
| lactate | 53154 | Lactate | Blood | Chemistry | <10 | <10 | 0.0 | 1.0 | 0.0 | mmol/L | 1.0 |
| albumin | 53085 | Albumin | Blood | Chemistry | 0 | 0 | 0.0 |  |  |  |  |
| albumin | 50862 | Albumin | Blood | Chemistry | 34214 | 28699 | 0.389 | 0.3974 | 0.0227 | g/dL | 1.0 |
| magnesium | 50960 | Magnesium | Blood | Chemistry | 124066 | 66329 | 0.8992 | 0.1603 | 0.0341 | mg/dL | 1.0 |
| bilirubin | 50885 | Bilirubin, Total | Blood | Chemistry | 51186 | 37399 | 0.507 | 0.3162 | 0.0249 | mg/dL | 1.0 |
| bilirubin | 53089 | Bilirubin, Total | Blood | Chemistry | 0 | 0 | 0.0 |  |  |  |  |
| inr | 51675 | INR(PT) | Blood | Chemistry | <10 | <10 | 0.0 | 1.0 | 0.0 |  | 1.0 |
| inr | 51237 | INR(PT) | Blood | Hematology | 97332 | 57898 | 0.7849 | 0.3026 | 0.0171 |  | 1.0 |


## OMR 365 ngày trước nhập viện
| result_name | admissions | coverage |
|---|---|---|
| BMI (kg/m2) | 35820 | 0.4856 |
| Blood Pressure | 43437 | 0.5888 |
| Height (Inches) | 27269 | 0.3697 |
| Weight (Lbs) | 39771 | 0.5391 |
| eGFR | 58 | 0.0008 |


## Thời gian chạy
{'cohort_s': 38.1, 'p1_s': 48.1, 'p2_p3_s': 62.9, 'icu_s': 82.3, 'labs_s': 109.7, 'total_s': 112.4}

Spark: {'master': 'local[*]', 'default_parallelism': 8, 'version': '3.5.3'}

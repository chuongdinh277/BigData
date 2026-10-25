# Danh mục dữ liệu MIMIC-IV và thiết kế đầu vào Spark

> **Chủ sở hữu:** TV1 · **Trạng thái:** v0.1 (dung lượng đã kiểm tra trên S3 ngày 25/10/2026; số dòng điền sau khi chạy `stage1_feasibility.py --row-counts`)

## 1. Vị trí và phiên bản

- Bucket: `heartguard-mimic-group11` (region `us-east-1`)
- Đường dẫn: `bronze/mimiciv/3.1/` gồm `hosp/` (22 bảng), `icu/` (9 bảng), `CHANGELOG.txt`, `LICENSE.txt`, `SHA256SUMS.txt`
- Đã đủ toàn bộ module `hosp` và `icu` của v3.1. Không có MIMIC-IV-ED và MIMIC-IV-Note (dataset riêng, không bắt buộc).
- Tổng dung lượng nén khoảng **10 GB**. Việc kiểm tra checksum đang chờ làm (`sha256sum -c SHA256SUMS.txt` trên EC2).

## 2. Bảng dùng trong dự án

| Bảng | Module | Dung lượng .gz | Số dòng | Khóa | Vai trò | Dùng cho |
|---|---|---|---|---|---|---|
| `patients` | hosp | 2.7 MB | _ | `subject_id` | Giới, tuổi neo, nhóm năm, `dod` | Cohort, P3 |
| `admissions` | hosp | 19.0 MB | _ | `hadm_id` | Thời gian nhập/xuất, kết cục, loại nhập viện, nhân khẩu học | Cohort, nhãn P1/P2 |
| `diagnoses_icd` | hosp | 32.0 MB | _ | (`hadm_id`, `seq_num`) | Mã ICD-9/10 | Cohort, bệnh kèm |
| `d_icd_diagnoses` | hosp | 0.9 MB | _ | (`icd_code`, `icd_version`) | Tên mã | Tra cứu |
| `labevents` | hosp | **2.4 GB** | _ | `labevent_id` | Xét nghiệm theo thời gian | Đặc trưng chính |
| `d_labitems` | hosp | 13 KB | _ | `itemid` | Tên, fluid, category | Map xét nghiệm |
| `omr` | hosp | 42.0 MB | _ | (`subject_id`, `chartdate`, `seq_num`, `result_name`) | Huyết áp, cân nặng, BMI, eGFR ngoại trú | Đặc trưng lịch sử |
| `transfers` | hosp | 44.0 MB | _ | `transfer_id` | Di chuyển giữa khoa | Tùy chọn |
| `icustays` | icu | 3.2 MB | _ | `stay_id` | Lần nằm ICU | Cờ ICU, độ phủ sinh hiệu |
| `chartevents` | icu | **3.3 GB** | _ | — | Sinh hiệu ICU | Đặc trưng sinh hiệu (chỉ ca ICU) |
| `d_items` | icu | 57 KB | _ | `itemid` | Tên item ICU | Map sinh hiệu |

Chưa dùng ở giai đoạn đầu: `emar*`, `pharmacy`, `prescriptions`, `poe*` (thuốc, khoảng 3 GB), `microbiologyevents`, `inputevents`, `outputevents`, `procedureevents`, `ingredientevents`, `datetimeevents`, `hcpcsevents`, `drgcodes`, `services`, `provider`, `caregiver`, `procedures_icd`. Có thể bổ sung ở vòng sau nếu cần đặc trưng thuốc.

## 3. Sơ đồ khóa liên kết

```text
patients (subject_id)
   │ 1 ── n
admissions (subject_id, hadm_id) ──────────────┐
   │ 1 ── n                                    │ 1 ── n
diagnoses_icd (hadm_id, seq_num, icd_code, icd_version) ── d_icd_diagnoses (icd_code, icd_version)
   │
   ├── labevents (subject_id, hadm_id*, itemid, charttime, valuenum) ── d_labitems (itemid)
   │        *hadm_id có thể NULL (ngoại trú/ED) → join theo subject_id + cửa sổ charttime
   ├── omr (subject_id, chartdate, result_name, result_value)   ← chỉ có ngày, không có hadm_id
   └── icustays (subject_id, hadm_id, stay_id, intime, outtime)
            │ 1 ── n
         chartevents (subject_id, hadm_id, stay_id, itemid, charttime, valuenum) ── d_items (itemid)
```

## 4. Điểm kỹ thuật phải xử lý

| # | Vấn đề | Cách xử lý |
|---|---|---|
| 1 | File `.csv.gz` **không chia nhỏ được**, nên Spark chỉ đọc được bằng 1 task/file | Giai đoạn 2: chạy `bronze_to_silver.py` một lần để chuyển sang Parquet đã repartition. Mọi job sau đọc Silver |
| 2 | `inferSchema` trên file 2–3 GB đọc file hai lần, rất chậm | Đọc chuỗi rồi ép kiểu tường minh các cột cần dùng |
| 3 | Thời gian bị dịch chuyển (date shifting) theo bệnh nhân | Chỉ so sánh thời gian **trong cùng bệnh nhân**. Năm thực chỉ biết qua `anchor_year_group` |
| 4 | Tuổi | `anchor_age + year(admittime) − anchor_year`; `anchor_age` > 89 bị gán 91 |
| 5 | `labevents.hadm_id` có thể NULL | Join theo `subject_id` + `charttime` trong cửa sổ, không chỉ theo `hadm_id` |
| 6 | Trùng đơn vị đo (`valueuom`) | Thống kê đơn vị theo `itemid`, chỉ giữ đơn vị chủ đạo, loại giá trị ngoài khoảng hợp lý |
| 7 | `chartevents` rất lớn và chỉ có cho ICU | Lọc `itemid` + `subject_id` của cohort **trước** khi tổng hợp; báo cáo độ phủ |
| 8 | `omr` chỉ có `chartdate` | Chỉ dùng giá trị có `chartdate < date(admittime)` để tránh leakage |

## 5. Thiết kế dữ liệu đầu vào Spark (bản nháp, chốt ở GĐ2)

```text
Bronze (csv.gz, nguyên bản)
  └─ bronze_to_silver.py   → Silver Parquet, ép kiểu, repartition
       └─ cohort.py        → processed/cohort/        (1 dòng / hadm_id đủ điều kiện)
            ├─ labels.py   → processed/labels/p1|p2|p3
            ├─ features.py → processed/features/p1|p2
            └─ quality.py  → reports/quality/
       └─ pipeline_runner.py ghép → gold/<task>/<data_version>/
```

Hợp đồng cột của Gold: xem `contracts/gold_schema_p1.json`.

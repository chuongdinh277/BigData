# Đặc tả bài toán dự đoán (prediction_problem_spec)

> **Chủ sở hữu:** TV3 · **Trạng thái:** BẢN ĐỀ XUẤT v0.1 — chờ nhóm duyệt và giảng viên xác nhận phạm vi
> **Dữ liệu:** MIMIC-IV v3.1 — `s3://heartguard-mimic-group11/bronze/mimiciv/3.1/`
> **Quy tắc:** không ai được tạo dataset hay huấn luyện trước khi file này được đánh dấu **ĐÃ CHỐT** trong `decision_log.md`.

---

## 0. Tóm tắt đề xuất

| Mã | Bài toán | Đơn vị | Mốc dự đoán | Nhãn | Đề xuất |
|---|---|---|---|---|---|
| **P1** | Tử vong trong viện | 1 đợt nhập viện (`hadm_id`) | `admittime + 24h` | `hospital_expire_flag = 1` | **Bài toán chính** |
| **P2** | Tái nhập viện 30 ngày | 1 lần xuất viện còn sống (`hadm_id`) | `dischtime` | Có admission không theo kế hoạch trong (0, 30] ngày | Bài toán mở rộng 1 |
| **P3** | Tử vong 90 ngày sau xuất viện | 1 lần xuất viện còn sống (`hadm_id`) | `dischtime` | `patients.dod` trong (0, 90] ngày sau `dischtime` | Bài toán mở rộng 2 |

**Lý do chọn P1 làm chính:** nhãn nằm trực tiếp trong `admissions`, không bị kiểm duyệt theo dõi (censoring), mốc dự đoán rõ ràng và đủ dữ liệu xét nghiệm trong 24h đầu. P2 và P3 dùng **chung một tập xuất viện**, nên chi phí làm thêm thấp khi pipeline P1 đã có.

**Vì sao P3 khả thi với MIMIC-IV:** `patients.dod` gộp ngày tử vong từ hồ sơ bệnh viện và hồ sơ hộ tịch tiểu bang, và được theo dõi tối đa khoảng **1 năm sau lần xuất viện cuối** của mỗi bệnh nhân. Cửa sổ 90 ngày nằm gọn trong khoảng đó, nên coi như quan sát đầy đủ (giới hạn: tử vong ngoài tiểu bang có thể bị bỏ sót). Kết cục bất lợi khác ngoài tử vong (biến cố tim mạch, nhập viện vì suy tim) **không** xác minh đầy đủ được ngoài hệ thống bệnh viện, nên không đưa vào P3.

---

## 1. Nhóm nghiên cứu chung (dùng cho cả 3 bài toán)

Chi tiết và thống kê khả thi do TV2 sở hữu trong `docs/stage1/cohort_spec.md`. Tóm tắt:

1. Admission có ít nhất một mã suy tim (theo `configs/hf_codes.yaml`) ở **bất kỳ vị trí** `seq_num`. Phương án so sánh: chỉ chẩn đoán chính (`seq_num = 1`).
2. Tuổi tại nhập viện ≥ 18, với `age = anchor_age + (year(admittime) − anchor_year)`.
3. Mốc thời gian hợp lệ: `dischtime > admittime`.
4. **Giữ mọi admission đủ điều kiện** của cùng một bệnh nhân. Hệ quả bắt buộc: chia train/val/test **theo `subject_id`**, và bootstrap khoảng tin cậy **theo bệnh nhân**.

> ⚠️ **Giới hạn phải ghi trong báo cáo:** mã ICD trong `diagnoses_icd` được gán **lúc xuất viện**. Dùng mã của admission hiện tại để **chọn cohort** là chấp nhận được (đây là thiết kế hồi cứu), nhưng **cấm** dùng chúng làm **đặc trưng** cho P1. Bệnh kèm chỉ được lấy từ các admission **trước** đó.

---

## 2. P1 — Tử vong trong viện (BÀI TOÁN CHÍNH)

| Mục | Định nghĩa |
|---|---|
| **Mục tiêu** | Ước lượng xác suất bệnh nhân suy tim tử vong trong đợt nằm viện hiện tại, tại thời điểm 24h sau nhập viện |
| **Đơn vị dự đoán** | 1 admission (`hadm_id`) |
| **Mốc dự đoán** | `prediction_time = admittime + 24h` |
| **Đối tượng** | Cohort chung (mục 1) **và** bệnh nhân còn nằm viện, còn sống tại mốc dự đoán |
| **Loại trừ** | `dischtime ≤ prediction_time` (đã xuất viện/chuyển viện trước 24h) · `deathtime ≤ prediction_time` (đã tử vong trước 24h) · thiếu `admittime`/`dischtime` |
| **Cửa sổ đặc trưng** | Xét nghiệm: `[admittime − 6h, prediction_time]` (cho phép xét nghiệm làm ở ED ngay trước nhập viện) · Sinh hiệu ICU: `[admittime, prediction_time]` · Lịch sử: các admission có `dischtime < admittime` hiện tại |
| **Nhãn** | `label_mortality = admissions.hospital_expire_flag` (1 = tử vong trong đợt nằm viện) |
| **Ca không xác minh được nhãn** | Không có — nhãn luôn quan sát được trong đợt nằm viện. Kiểm tra chéo: `hospital_expire_flag = 1` phải khớp `deathtime IS NOT NULL` hoặc `discharge_location = 'DIED'`; ca lệch được đếm và báo cáo |
| **Cấm dùng làm đặc trưng** | `dischtime`, độ dài nằm viện (LOS), `discharge_location`, `deathtime`, `hospital_expire_flag`, mã ICD/DRG/thủ thuật của admission hiện tại, bất kỳ sự kiện nào có `charttime > prediction_time`, `patients.dod` |
| **Bảng nguồn** | `admissions`, `patients`, `diagnoses_icd`, `labevents`, `d_labitems`, `icustays`, `chartevents`, `d_items`, `omr` |
| **Đánh giá** | ROC-AUC, PR-AUC (kèm prevalence), F1 tại ngưỡng chọn trên validation, Brier, reliability diagram, CI bootstrap theo bệnh nhân |
| **Đầu ra hệ thống** | `probability`, `calibrated_probability`, `risk_level`, `review_flag`, top đặc trưng SHAP, `model_version`, `prediction_time` |

---

## 3. P2 — Tái nhập viện trong 30 ngày

| Mục | Định nghĩa |
|---|---|
| **Mục tiêu** | Ước lượng xác suất bệnh nhân suy tim tái nhập viện không theo kế hoạch trong 30 ngày sau xuất viện |
| **Đơn vị dự đoán** | 1 lần xuất viện còn sống (`hadm_id` của admission chỉ mục) |
| **Mốc dự đoán** | `prediction_time = dischtime` |
| **Đối tượng** | Cohort chung, `hospital_expire_flag = 0` |
| **Loại trừ admission chỉ mục** | Tử vong trong viện · `discharge_location IN ('HOSPICE', 'AGAINST ADVICE', 'ACUTE HOSPITAL')` (chuyển viện hoặc trốn viện: không quan sát được tái nhập) · **Tử vong trong 30 ngày mà không tái nhập** (`dod` trong (0, 30] ngày và không có readmission) — loại ở phân tích chính, giữ lại làm phân tích độ nhạy với nhãn 0 |
| **Sự kiện tái nhập** | Admission kế tiếp của cùng `subject_id` có `0 < admittime_next − dischtime ≤ 30 ngày` **và** `admission_type NOT IN ('ELECTIVE', 'SURGICAL SAME DAY ADMISSION')` |
| **Nhãn** | `label_readmit30 = 1` nếu có sự kiện tái nhập, ngược lại `0` |
| **Admission đếm 2 lần** | Một admission vừa là "lần tái nhập" của admission trước, vừa là "admission chỉ mục" mới nếu đủ điều kiện — được phép, xử lý bằng group split theo bệnh nhân |
| **Cửa sổ đặc trưng** | Toàn bộ thông tin trong đợt nằm viện đến `dischtime` (xét nghiệm cuối, LOS, `discharge_location`, mã ICD admission hiện tại **được dùng** vì đã có lúc xuất viện) + lịch sử trước đó |
| **Cấm dùng làm đặc trưng** | Mọi thông tin của admission tiếp theo, `patients.dod`, sự kiện sau `dischtime` |
| **Giới hạn** | Chỉ quan sát tái nhập **tại BIDMC**, nên tái nhập ở bệnh viện khác bị tính là 0 (đánh giá thấp tỷ lệ thật). Không xác định được "cuối kỳ dữ liệu" của từng bệnh nhân do thời gian bị dịch chuyển; ghi rõ là giới hạn |
| **Bảng nguồn** | Như P1 + `transfers` (tùy chọn) |
| **Đánh giá** | Như P1 |

---

## 4. P3 — Tử vong 90 ngày sau xuất viện

| Mục | Định nghĩa |
|---|---|
| **Mục tiêu** | Ước lượng xác suất tử vong do mọi nguyên nhân trong 90 ngày sau khi xuất viện còn sống |
| **Đơn vị dự đoán** | 1 lần xuất viện còn sống (cùng tập chỉ mục với P2) |
| **Mốc dự đoán** | `prediction_time = dischtime` |
| **Loại trừ** | Tử vong trong viện · `discharge_location = 'HOSPICE'` (kết cục gần như chắc chắn, không có giá trị dự đoán) |
| **Nhãn** | `label_mort90 = 1` nếu `patients.dod` nằm trong `(date(dischtime), date(dischtime) + 90 ngày]` |
| **Kiểm duyệt (censoring)** | Theo dõi `dod` kéo dài khoảng 1 năm sau lần xuất viện cuối, nên cửa sổ 90 ngày được quan sát đầy đủ. Vì vậy dùng **phân loại nhị phân**, không cần mô hình sống còn |
| **Cửa sổ đặc trưng** | Như P2 |
| **Giới hạn** | `dod` chỉ có ngày (không có giờ); tử vong ngoài tiểu bang có thể thiếu. Mở rộng: 180 ngày cũng khả thi với cùng logic |
| **Đánh giá** | Như P1 |

---

## 5. Điều kiện khả thi (điền bằng số thật sau khi TV1 chạy `stage1_feasibility.py`)

| Chỉ số | P1 | P2 | P3 | Ngưỡng chấp nhận |
|---|---|---|---|---|
| Số admission đủ điều kiện | _ | _ | _ | ≥ 5.000 |
| Số bệnh nhân | _ | _ | _ | — |
| Số ca dương tính | _ | _ | _ | ≥ 500 (để test có ≥ 100 ca dương) |
| Tỷ lệ dương tính | _ | _ | _ | — |
| Tỷ lệ thiếu NT-proBNP / creatinine trong cửa sổ | _ | _ | _ | báo cáo |
| Tỷ lệ có ICU trong 24h đầu (độ phủ sinh hiệu) | _ | — | — | báo cáo |
| Số ca dương mỗi `anchor_year_group` (cho drift) | _ | _ | _ | ≥ 50 mỗi nhóm |

---

## 6. Câu hỏi gửi giảng viên

1. Đề có bắt buộc triển khai đủ 3 bài toán, hay cho phép **P1 làm chính** và P2/P3 là mở rộng?
2. Định nghĩa P3 = **tử vong 90 ngày sau xuất viện** có được chấp nhận là "kết cục bất lợi trung hạn" không?
3. Drift: đánh giá **ngoại tuyến theo lô** (chia theo `anchor_year_group`, có nhãn thật) có đủ cho yêu cầu "drift monitoring" không?
4. Việc chỉ quan sát được tái nhập tại cùng một bệnh viện có cần xử lý thêm, hay chỉ cần nêu giới hạn?

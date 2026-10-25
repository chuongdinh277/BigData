# Đặc tả đối tượng nghiên cứu (cohort) — bệnh nhân suy tim

> **Chủ sở hữu:** TV2 · **Trạng thái:** v0.1 đề xuất · Mã ICD: `configs/hf_codes.yaml`

## 1. Định nghĩa suy tim

Một admission thuộc cohort HF khi có ít nhất một dòng trong `diagnoses_icd` thỏa **một** trong các điều kiện:

| ICD | Mã (khớp theo tiền tố, không có dấu chấm) | Ý nghĩa |
|---|---|---|
| 10 | `I50*` | Suy tim mọi loại |
| 10 | `I110`, `I130`, `I132` | Bệnh tim do tăng huyết áp, có suy tim |
| 10 | `I0981` | Suy tim do thấp |
| 9 | `428*` | Suy tim mọi loại |
| 9 | `40201`, `40211`, `40291`, `40401`, `40403`, `40411`, `40413`, `40491`, `40493` | Bệnh tim/thận do tăng huyết áp, có suy tim |

Tham khảo: cách định nghĩa suy tim theo ICD trong các nghiên cứu EHR (họ mã I50/428 kèm các mã tăng huyết áp có suy tim), thường gặp trong các bộ chỉ số bệnh kèm (Elixhauser/Quan). TV2 bổ sung trích dẫn chính thức vào `references.bib`.

## 2. Hai phương án vị trí mã — chọn sau khi có số liệu

| Phương án | Điều kiện | Ưu | Nhược |
|---|---|---|---|
| **A (đề xuất)** | HF ở **bất kỳ** `seq_num` | Cohort lớn, đúng với khái niệm "bệnh nhân đã mắc suy tim" | Gồm cả ca nhập viện vì lý do khác |
| B | HF là chẩn đoán chính (`seq_num = 1`) | Đồng nhất hơn ("nhập viện vì suy tim") | Cohort nhỏ hơn nhiều, ít ca tử vong |

Phương án B có thể dùng làm **phân tích nhóm con** (subgroup) để kiểm tra độ ổn định.

## 3. Điều kiện chọn / loại (áp dụng theo thứ tự, mỗi bước đếm attrition)

| Bước | Điều kiện | Ghi chú |
|---|---|---|
| S0 | Toàn bộ `admissions` | Điểm xuất phát |
| S1 | Có mã HF (phương án A) | |
| S2 | `dischtime > admittime`, không thiếu thời gian | Loại dữ liệu lỗi |
| S3 | Tuổi tại nhập viện ≥ 18 | `anchor_age + year(admittime) − anchor_year` |
| **Cohort chung** | Kết quả S3 | Dùng cho P1/P2/P3 |
| P1-a | Còn nằm viện tại `admittime + 24h` | `dischtime > admittime + 24h` |
| P1-b | Còn sống tại `admittime + 24h` | `deathtime IS NULL OR deathtime > admittime + 24h` |
| P2/P3-a | Xuất viện còn sống | `hospital_expire_flag = 0` |
| P2-b | Không chuyển viện/trốn viện/hospice | `discharge_location NOT IN ('HOSPICE','AGAINST ADVICE','ACUTE HOSPITAL')` |
| P2-c | Loại ca tử vong ≤ 30 ngày mà không tái nhập | Phân tích chính |
| P3-b | Không xuất viện về hospice | |

## 4. Nhập viện lặp lại

- **Giữ tất cả** admission đủ điều kiện, mỗi admission là một mẫu.
- Hệ quả: chia tập **theo `subject_id`** (không có bệnh nhân nào nằm ở 2 tập), và bootstrap CI **theo bệnh nhân**.
- Phương án độ nhạy: chỉ lấy admission HF **đầu tiên** của mỗi bệnh nhân.

## 5. Thống kê khả thi (điền từ `reports/stage1/stage1_feasibility.md`)

| Chỉ số | Phương án A | Phương án B |
|---|---|---|
| Số admission HF | _ | _ |
| Số bệnh nhân HF | _ | _ |
| Tỷ lệ admission ICD-9 / ICD-10 | _ | _ |
| P1: n / số tử vong / tỷ lệ | _ | _ |
| P2: n / số tái nhập / tỷ lệ | _ | _ |
| P3: n / số tử vong 90 ngày / tỷ lệ | _ | _ |
| Phân bố theo `anchor_year_group` | _ | _ |
| Phân bố giới / nhóm tuổi | _ | _ |

## 6. Nhóm con dự kiến cho đánh giá subgroup (GĐ5)

Giới tính · nhóm tuổi (18–64 / 65–79 / ≥ 80) · `anchor_year_group` · ICD-9 và ICD-10 · nằm ICU trong 24h đầu hay không · HF là chẩn đoán chính hay phụ · `race` (gộp nhóm) và `insurance`. Hai biến cuối **chỉ dùng để đánh giá công bằng, không đưa vào mô hình**.

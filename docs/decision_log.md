# Decision log

Mọi thay đổi về nhãn, cohort, schema hoặc API phải được ghi ở đây, có người đề xuất và người duyệt.
Trạng thái: `ĐỀ XUẤT` → `ĐÃ CHỐT` / `BỊ BÁC`.

| ID | Ngày | Quyết định | Đề xuất | Duyệt | Trạng thái |
|---|---|---|---|---|---|
| D-001 | 25/10/2026 | Dữ liệu: MIMIC-IV **v3.1**, đọc tại `s3://heartguard-mimic-group11/bronze/mimiciv/3.1/` (giữ đường dẫn này, không chuyển sang `raw/`) | TV1 | | ĐỀ XUẤT |
| D-002 | 25/10/2026 | Bài toán chính **P1** = tử vong trong viện, dự đoán tại `admittime + 24h`. P2 (tái nhập 30 ngày) và P3 (tử vong 90 ngày sau xuất viện) là mở rộng | TV3 | | ĐỀ XUẤT |
| D-003 | 25/10/2026 | Cohort: mã HF ở **mọi vị trí** `seq_num` (phương án A), tuổi ≥ 18, giữ mọi admission | TV2 | | ĐỀ XUẤT |
| D-004 | 25/10/2026 | Bệnh kèm chỉ lấy từ **admission trước**. Cấm dùng ICD/DRG của admission hiện tại làm đặc trưng P1 | TV4 | | ĐỀ XUẤT |
| D-005 | 25/10/2026 | Chia tập theo `subject_id` (group split); bootstrap CI theo bệnh nhân | TV3 | | ĐỀ XUẤT |
| D-006 | 25/10/2026 | Drift đánh giá **ngoại tuyến theo lô** `anchor_year_group` (có nhãn thật); chờ giảng viên xác nhận | TV5 | | ĐỀ XUẤT |
| D-007 | 25/10/2026 | Cửa sổ xét nghiệm P1: `[admittime − 6h, admittime + 24h]` | TV4 | | ĐỀ XUẤT |
| D-008 | 25/10/2026 | Ô thống kê < 10 được che khi chia sẻ báo cáo | TV1 | | ĐỀ XUẤT |

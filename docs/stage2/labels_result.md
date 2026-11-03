# Kết quả chạy thử nhãn bằng dữ liệu giả

Dùng script `tools/make_synthetic_mimic.py` để tạo dữ liệu thử, gồm 800
bệnh nhân và 1.851 lượt nhập viện (seed 42). Sau khi lọc cohort suy tim, còn
1.121 lượt. Đây là dữ liệu giả, không phải kết quả trên MIMIC-IV thật.

## Kết quả

| Nhãn | Số lượt nhập viện | Số bệnh nhân | Ca dương | Tỷ lệ ca dương |
|---|---:|---:|---:|---:|
| P1 - tử vong trong viện | 1.068 | 657 | 140 | 13,11% |
| P2 - tái nhập trong 30 ngày | 866 | 549 | 165 | 19,05% |
| P3 - tử vong trong 90 ngày | 937 | 571 | 68 | 7,26% |

Đã kiểm tra để chắc rằng `hadm_id` không bị trùng, các cột khóa và thời
gian dự đoán đều có giá trị, nhãn chỉ là 0 hoặc 1.

## Các lượt nhập viện bị loại

| Bài toán | Lý do | Số lượt |
|---|---|---:|
| P1 | Thời gian nhập viện/xuất viện không hợp lệ | 0 |
| P1 | Đã xuất viện tại hoặc trước mốc dự đoán 24 giờ | 53 |
| P1 | Đã tử vong tại hoặc trước mốc dự đoán 24 giờ | 0 |
| P2 | Thiếu thời gian xuất viện | 0 |
| P2 | Tử vong trong viện | 145 |
| P2 | Chuyển viện, AMA hoặc hospice | 102 |
| P2 | Tử vong trong 30 ngày mà không tái nhập | <10 |
| P3 | Thiếu thời gian xuất viện | 0 |
| P3 | Tử vong trong viện | 145 |
| P3 | Xuất viện về hospice | 39 |


Kết quả này chỉ để thử logic bằng dữ liệu giả; cần chạy lại với MIMIC-IV thật trước khi đưa ra kết luận cho dự án.

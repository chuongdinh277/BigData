# Kết quả gắn nhãn P1/P2/P3 trên dữ liệu thật

**Nguồn dữ liệu:** MIMIC-IV 3.1 (Silver trên S3) sử dụng cohort đã được xây dựng.

Quá trình gắn nhãn trên dữ liệu MIMIC-IV 3.1 đã hoàn thành, tạo thành công ba bộ dữ liệu nhãn P1, P2 và P3 với kết quả thống kê như sau:

| Nhãn | Số lượt nhập viện | Số bệnh nhân | Ca dương | Tỷ lệ dương |
| ---- | ----------------: | -----------: | -------: | ----------: |
| P1   |            73,768 |       30,287 |    3,588 |      0.0486 |
| P2   |            72,666 |       27,870 |   17,796 |      0.2449 |
| P3   |            75,285 |       28,855 |    7,571 |      0.1006 |

## Số lượt nhập viện bị loại

| Nhãn | Lý do                                     | Số lượt |
| ---- | ----------------------------------------- | ------: |
| P1   | `invalid_admission_or_discharge_time`     |       0 |
| P1   | `discharged_by_prediction_time`           |   6,977 |
| P1   | `died_by_prediction_time`                 |     <10 |
| P2   | `missing_discharge_time`                  |       0 |
| P2   | `died_in_hospital`                        |   4,005 |
| P2   | `transfer_ama_or_hospice`                 |   2,321 |
| P2   | `death_within_window_without_readmission` |   1,754 |
| P3   | `missing_discharge_time`                  |       0 |
| P3   | `died_in_hospital`                        |   4,005 |
| P3   | `hospice`                                 |   1,456 |

## Kết luận

Quá trình gắn nhãn P1, P2 và P3 đã hoàn thành trên dữ liệu thực tế. Kết quả thống kê P1 khớp với kết quả Feature Engineering đã thực hiện trước đó. Đồng thời, phép kiểm tra tính nhất quán giữa hospital_expire_flag, deathtime và discharge_location không phát hiện trường hợp không khớp (0 ca).

Cả ba bộ nhãn đã được lưu dưới dạng Parquet trên Amazon S3, phục vụ các bước tích hợp và kiểm tra chất lượng dữ liệu tiếp theo.

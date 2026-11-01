# Đo tốc độ Spark – pandas (Giai đoạn 2)

> Nhánh : khoe/features · Mã: `data_spark/benchmark.py` · **Các ô "…" điền sau khi chạy trên dữ liệu thật.**

## 1. Tác vụ đo

Tổng hợp xét nghiệm 24h: với mỗi `hadm_id` và mỗi trong 20 lab, lấy `first`/`min`/`max` của `valuenum` trong cửa sổ `[admittime − 6h, admittime + 24h]`, có điều kiện `storetime ≤ admittime + 24h`. Hai engine thực hiện **đúng cùng phép tính**, không gồm bước kiểm đơn vị và khoảng hợp lệ. Kết quả đã đối chiếu số dòng đầu ra giữa hai engine (bắt buộc bằng nhau).

| Yếu tố    | Cách thực hiện                                                                                                           |
| --------- | ------------------------------------------------------------------------------------------------------------------------ |
| Mẫu       | `subject_id % 10000 < tỷ lệ × 10000` (1%, 5%, 10% bệnh nhân), cùng mẫu cho mọi cấu hình                                  |
| Spark     | 1, 2, 3 worker (`--master local[k]` hoặc cụm standalone); đọc Parquet trên S3                                            |
| pandas    | pyarrow đọc có lọc `itemid` và `subject_id`, sau đó xử lý bằng pandas trong bộ nhớ                                       |
| Số lần    | 3 lần mỗi cấu hình; ghi từng lần, báo cáo trung bình và độ lệch chuẩn                                                    |
| Thời gian | Từ lúc đọc đến khi có kết quả cuối (gồm I/O). Spark chạy lazy nên ép bằng `count()`; thời gian khởi động Spark ghi riêng |

## 2. Môi trường

| Mục                                       | Giá trị                           |
| ----------------------------------------- | --------------------------------- |
| Máy                                       | … (vd m5.2xlarge: 8 vCPU, 32 GiB) |
| Spark / Python / pandas / pyarrow         | …                                 |
| Cách chạy worker                          | … (local[k] hay cụm standalone)   |
| Số bệnh nhân / dòng `labevents` thuộc mẫu | …                                 |

## 3. Kết quả

| Cấu hình |           Mẫu | Lần 1 (s) | Lần 2 (s) | Lần 3 (s) | Trung bình (s) | Độ lệch chuẩn (s) |
| -------- | ------------: | --------: | --------: | --------: | -------------: | ----------------: |
| pandas   |            1% |         … |         … |         … |              … |                 … |
| pandas   |            5% |         … |         … |         … |              … |                 … |
| pandas   |           10% |         … |         … |         … |              … |                 … |
| spark-1w |            1% |         … |         … |         … |              … |                 … |
| spark-1w |            5% |         … |         … |         … |              … |                 … |
| spark-1w |           10% |         … |         … |         … |              … |                 … |
| spark-2w | 1% / 5% / 10% |         … |         … |         … |              … |                 … |
| spark-3w | 1% / 5% / 10% |         … |         … |         … |              … |                 … |

Thời gian khởi động Spark: … giây.

![Biểu đồ thời gian chạy](../../reports/benchmark/bench.png)

## 4. Nhận xét (viết sau khi có số)

- Với mẫu nhỏ, Spark có thể chậm hơn pandas vì chi phí khởi động và điều phối; ghi rõ điểm giao nhau (nếu có).
- Mức tăng tốc khi thêm worker: … (so với 1 worker; có đạt tuyến tính không và vì sao).
- Pandas có chạy nổi với toàn bộ dữ liệu không (giới hạn bộ nhớ), còn Spark thì sao: …
- Giới hạn của phép đo: chỉ một tác vụ, một máy, mẫu theo `subject_id`; kết quả không suy rộng cho mọi thao tác.

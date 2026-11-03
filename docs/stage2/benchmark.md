### Kết quả và đánh giá Performance Benchmark

Thực nghiệm đánh giá hiệu năng của Apache Spark và pandas trên tác vụ tổng hợp 20 loại xét nghiệm trong cửa sổ từ 6 giờ trước đến 24 giờ sau nhập viện. Các phép tổng hợp gồm giá trị đầu tiên (`first`), nhỏ nhất (`min`) và lớn nhất (`max`) theo từng cặp `(hadm_id, lab)`.

Thử nghiệm sử dụng ba tỷ lệ lấy mẫu bệnh nhân (1%, 5%, 10%) và bốn cấu hình thực thi: pandas, Spark `local[1]`, `local[2]`, `local[3]`. Mỗi cấu hình được chạy ba lần và báo cáo thời gian trung bình cùng độ lệch chuẩn.

| Cấu hình       |      Mẫu 1% |      Mẫu 5% |     Mẫu 10% |
| -------------- | ----------: | ----------: | ----------: |
| pandas         |  4,6 ± 0,1s |  4,8 ± 0,1s |  5,4 ± 0,4s |
| Spark local[1] | 85,7 ± 2,7s | 77,6 ± 5,2s | 77,1 ± 3,9s |
| Spark local[2] | 45,5 ± 3,2s | 41,1 ± 3,3s | 41,1 ± 2,3s |
| Spark local[3] | 31,4 ± 3,4s | 30,6 ± 2,5s | 29,3 ± 2,3s |

**Nhận xét**

- pandas đạt thời gian thấp nhất ở cả ba mức lấy mẫu, cho thấy lợi thế của việc xử lý dữ liệu vừa bộ nhớ và chi phí thực thi thấp.
- Apache Spark có khả năng tận dụng thực thi song song. Với mẫu 10%, cấu hình `local[3]` đạt mức tăng tốc khoảng 2,63 lần so với `local[1]`.
- Thời gian Spark không tăng tương ứng với tỷ lệ lấy mẫu. Điều này cho thấy cần phân tích thêm chi phí đọc dữ liệu, lập kế hoạch và thực thi để xác định nút thắt hiệu năng.

**Giới hạn thực nghiệm**

Các cấu hình Spark chạy ở local mode trên một máy EC2, không phải cụm Spark phân tán với nhiều worker độc lập. Spark và pandas sử dụng phương pháp đọc, lọc dữ liệu khác nhau nên kết quả phản ánh hiệu năng của pipeline hoàn chỉnh, không phải so sánh thuần túy tốc độ tính toán. Thời gian khởi động Spark được ghi nhận riêng và không nằm trong thời gian xử lý báo cáo.

**Kết luận**

Trong phạm vi dữ liệu và tác vụ được khảo sát, pandas có hiệu năng tốt hơn. Tuy nhiên, Spark cho thấy khả năng tăng tốc khi tăng mức song song. Kết quả không đủ cơ sở kết luận hiệu năng của hai công nghệ trên dữ liệu vượt bộ nhớ hoặc trong môi trường cluster phân tán.

### Biểu đồ so sánh hiệu năng

![Spark vs pandas Benchmark](../../reports/benchmark/bench_final.png)

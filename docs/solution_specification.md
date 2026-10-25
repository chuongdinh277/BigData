# Solution Specification — HeartGuard AI

> Bản đầu (v0.1, Giai đoạn 1). **Mỗi người viết phần mình phụ trách**, không dồn cho một người tổng hợp.

| # | Mục | Phụ trách | Nguồn chi tiết | Trạng thái |
|---|---|---|---|---|
| 1 | Bài toán, mục tiêu, phạm vi, ngoài phạm vi | TV3 | `prediction_problem_spec.md` | Nháp |
| 2 | Truy vết yêu cầu đề bài → chức năng → bằng chứng | TV3 | — | Chưa |
| 3 | Dữ liệu, quyền truy cập, đạo đức dữ liệu (DUA PhysioNet) | TV1 | `stage1/data_inventory.md` | Nháp |
| 4 | Cohort và attrition | TV2 | `stage1/cohort_spec.md` | Nháp |
| 5 | Nhãn P1/P2/P3 và chống leakage | TV3 + TV4 | `prediction_problem_spec.md`, `stage1/feature_spec.md` | Nháp |
| 6 | Đặc trưng và quy tắc dữ liệu | TV4 | `stage1/feature_spec.md` | Nháp |
| 7 | Kiến trúc, luồng dữ liệu, hạ tầng AWS | TV5 | `stage1/architecture.md` | Nháp |
| 8 | Thiết kế pipeline Spark và benchmark | TV1 + TV5 | `stage1/data_inventory.md` §5 | Nháp |
| 9 | Thiết kế thí nghiệm ML (split, chiến lược S0–S4, metric) | TV1 | GĐ3 | Chưa |
| 10 | Trustworthy AI: calibration, SHAP, missing, subgroup, drift, review | TV2–TV5 | GĐ5 | Chưa |
| 11 | API và giao diện | TV5 | `../contracts/api_contract.md` | Nháp |
| 12 | Kế hoạch đánh giá và tiêu chí nghiệm thu | Cả nhóm | Kế hoạch 7 giai đoạn | Chưa |
| 13 | Giới hạn nghiên cứu | Cả nhóm | — | Chưa |

## 1. Bài toán (TV3)

Xem `prediction_problem_spec.md`. Tóm tắt: P1 tử vong trong viện tại mốc 24h (chính); P2 tái nhập 30 ngày; P3 tử vong 90 ngày sau xuất viện.

## 3. Dữ liệu (TV1)

_TV1 viết: nguồn, phiên bản, giấy phép, nơi lưu, ai có credential, chính sách không chia sẻ dữ liệu._

## 7. Kiến trúc (TV5)

Xem `stage1/architecture.md`.

## 13. Giới hạn đã biết

- Dữ liệu một trung tâm (BIDMC), thời gian bị dịch chuyển.
- Mã ICD của admission hiện tại được gán lúc xuất viện, nhưng được dùng để chọn cohort.
- P2 chỉ quan sát được tái nhập tại cùng bệnh viện.
- Sinh hiệu chỉ có cho ca ICU.
- Sản phẩm là nguyên mẫu nghiên cứu, không dùng cho quyết định lâm sàng.

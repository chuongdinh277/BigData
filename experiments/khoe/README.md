# Giai đoạn 4 – XGBoost trên Gold P1 V1

## 1. Mục tiêu

Huấn luyện và đánh giá mô hình XGBoost cho bài toán dự đoán tử vong trong viện tại thời điểm 24 giờ sau nhập viện, sử dụng Dataset V1 và chiến lược chia dữ liệu đã thống nhất của nhóm.

Thực hiện ba thí nghiệm:

| Case              | Cấu hình                                     | Mục đích                      |
| ----------------- | -------------------------------------------- | ----------------------------- |
| `S1_xgb_default`  | XGBoost mặc định                             | Thiết lập kết quả cơ sở       |
| `S1_xgb_weighted` | `scale_pos_weight = N_negative / N_positive` | Xử lý mất cân bằng nhãn       |
| `S1_xgb_optuna`   | Optuna TPE, 50 trials                        | Tối ưu PR-AUC trên validation |

## 2. Dữ liệu và phương pháp

- **Dataset:** Gold P1 V1 – MIMIC-IV 3.1.
- **Split:** Patient-level stratified, Train/Validation/Test = 70/15/15, seed 42.
- **Tiền xử lý:** `sklearn.Pipeline`, chỉ fit trên tập train.
- **Huấn luyện:** Sử dụng train, đánh giá và tối ưu trên validation.
- **Đánh giá:** Sử dụng các hàm và schema kết quả trong `experiments/common`.
- **Ràng buộc:** Không sử dụng tập test, không chia lại dữ liệu và không thay đổi code chung.

## 3. Chạy thực nghiệm

Tại thư mục gốc repository trên EC2:

```bash
source ~/venv-hg/bin/activate

python -m experiments.khoe.run_xgboost \
  --trials 50 \
  --jobs 4
```

**Dữ liệu đầu vào:**

- Gold: `s3://heartguard-mimic-group11/gold/p1/v1/`
- Split: `s3://heartguard-mimic-group11/splits/p1/v1/split_manifest.json`

## 4. Kết quả đầu ra

File kết quả:

`experiments/khoe/results.csv`

File chứa đúng **3 dòng kết quả** tương ứng với ba thí nghiệm, sử dụng 15 cột theo `RESULT_COLUMNS` của nhóm.

Các mô hình được đánh giá bằng PR-AUC, ROC-AUC và các chỉ số trong khung đánh giá chung. PR-AUC trên validation là tiêu chí chính để so sánh và lựa chọn cấu hình.

Kết quả được đối chiếu với mô hình S0 (Logistic Regression): PR-AUC = 0.299, ROC-AUC = 0.833.

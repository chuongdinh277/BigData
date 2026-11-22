# XGBoost – Dự đoán tử vong trong viện (Gold P1 V1)

## Tổng quan

Huấn luyện và đánh giá **XGBoost** cho bài toán dự đoán tử vong trong viện tại mốc **24 giờ sau nhập viện**. Các thực nghiệm sử dụng chung Gold P1 V1 và bộ chia dữ liệu đã khóa.

## Các thực nghiệm

| Case              | Cấu hình                                        | Mục đích                                          |
| ----------------- | ----------------------------------------------- | ------------------------------------------------- |
| `S1_xgb_default`  | XGBoost mặc định                                | Thiết lập kết quả cơ sở                           |
| `S1_xgb_weighted` | `scale_pos_weight = N_âm_train / N_dương_train` | Đánh giá tác động của cân bằng trọng số lớp       |
| `S1_xgb_optuna`   | Optuna TPE, 50 trials                           | Tối ưu hyperparameter theo PR-AUC trên validation |

Tiền xử lý được thực hiện trong `sklearn.Pipeline` và chỉ học từ **train**. Sử dụng split theo bệnh nhân, có phân tầng nhãn, tỷ lệ **70/15/15**, seed **42**.

Không chia lại dữ liệu hoặc sử dụng tập test để huấn luyện, tối ưu hay đánh giá.

## Chạy thực nghiệm

Tại thư mục gốc repository trên EC2, với môi trường Python đã cài các thư viện cần thiết (`xgboost`, `optuna`, `scikit-learn`, `pandas`, `pyarrow`, `s3fs`):

```bash
source ~/venv-hg/bin/activate

python -m experiments.khoe.run_xgboost \
  --trials 50 \
  --jobs 4
```

**Đường dẫn mặc định:**

- Gold P1 V1: `s3://heartguard-mimic-group11/gold/p1/v1/`
- Split manifest: `s3://heartguard-mimic-group11/splits/p1/v1/split_manifest.json`
- Kết quả: `experiments/khoe/results.csv`

Nếu cần chạy lại khi đã có file kết quả, sử dụng `--overwrite` hoặc `--out` để lưu vào đường dẫn khác.

## Kết quả và kiểm tra

File `experiments/khoe/results.csv` chứa **3 dòng kết quả và 15 cột** theo schema `RESULT_COLUMNS` trong `experiments/common`.

Các metric đánh giá trên validation gồm **PR-AUC, ROC-AUC, F1 và Brier score**.

Kiểm thử:

```bash
python -m pytest tests/test_xgboost.py -q
```

**Lưu ý:** Kết quả validation dùng để so sánh và lựa chọn cấu hình; không thay thế đánh giá cuối cùng trên test.

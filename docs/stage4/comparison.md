# Giai đoạn 4: so sánh mô hình

Dữ liệu: Dataset v1 (Gold P1). Chia tập theo bệnh nhân 70/15/15, seed 42.
Tất cả số liệu là trên tập validation. Tập test chưa mở.
Chỉ số chính: PR-AUC (tỉ lệ ca tử vong khoảng 4,86%, đoán ngẫu nhiên chỉ được khoảng 0,048).

## Tốt nhất của từng mô hình

| Hạng | Mô hình | Lần chạy | Người | PR-AUC | ROC-AUC | F1 | Brier | Giây |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | XGBoost | S1_xgb_optuna | khoe | 0.3207 | 0.8459 | 0.3662 | 0.0387 | 137.3 |
| 2 | LightGBM | S1_lgbm_optuna | chien | 0.3157 | 0.8479 | 0.3610 | 0.0387 | 197.8 |
| 3 | LogisticRegression | S2_lr_optuna50 | hung | 0.3019 | 0.8344 | 0.3448 | 0.0394 | 13.3 |

## Tất cả các lần chạy

| Hạng | Mô hình | Lần chạy | Người | PR-AUC | ROC-AUC | F1 | Brier | Giây |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | XGBoost | S1_xgb_optuna | khoe | 0.3207 | 0.8459 | 0.3662 | 0.0387 | 137.3 |
| 2 | LightGBM | S1_lgbm_optuna | chien | 0.3157 | 0.8479 | 0.3610 | 0.0387 | 197.8 |
| 3 | LogisticRegression | S2_lr_optuna50 | hung | 0.3019 | 0.8344 | 0.3448 | 0.0394 | 13.3 |
| 4 | LogisticRegression | S0_lr_default | hung | 0.2987 | 0.8331 | 0.3424 | 0.0396 | 2.0 |
| 5 | LogisticRegression | S0_lr_median_baseline | chuong | 0.2987 | 0.8331 | 0.3424 | 0.0396 | 2.2 |
| 6 | LogisticRegression | S1_lr_balanced | hung | 0.2874 | 0.8326 | 0.3400 | 0.1578 | 3.8 |
| 7 | LightGBM | S1_lgbm_default | chien | 0.2812 | 0.8376 | 0.3482 | 0.0398 | 2.1 |
| 8 | LightGBM | S1_lgbm_weighted | chien | 0.2712 | 0.8346 | 0.3339 | 0.1120 | 2.1 |
| 9 | XGBoost | S1_xgb_default | khoe | 0.2372 | 0.8007 | 0.3163 | 0.0426 | 1.5 |
| 10 | XGBoost | S1_xgb_weighted | khoe | 0.1972 | 0.7623 | 0.2594 | 0.0831 | 1.5 |

Bảng này sinh tự động bằng `python -m experiments.chuong.compare_results`. Muốn cập nhật, chạy lại lệnh sau khi có results.csv mới.

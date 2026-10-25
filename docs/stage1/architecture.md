# Kiến trúc hệ thống

> **Chủ sở hữu:** TV5 · v0.1

## 1. Sơ đồ tổng thể

```text
┌──────────────────────── OFFLINE (nghiên cứu / huấn luyện) ────────────────────────┐
│                                                                                    │
│  S3 bronze/mimiciv/3.1 (csv.gz)                                                    │
│        │  bronze_to_silver.py                                                      │
│        ▼                                                                           │
│  S3 silver/ (Parquet) ──► cohort → labels → features → quality (PySpark, EC2)      │
│        │                                                                           │
│        ▼                                                                           │
│  S3 gold/<task>/<data_version>/  +  split_manifest (theo subject_id)               │
│        │                                                                           │
│        ▼                                                                           │
│  experiments/ (5 thành viên) → trustworthy_ai/ (calibration, SHAP, missing,        │
│                                                  subgroup, drift)                  │
│        │                                                                           │
│        ▼                                                                           │
│  S3 model-artifacts/<task>/<model_version>/                                        │
│     model.pkl · preprocessor.pkl · calibrator.pkl · feature_schema.json ·          │
│     thresholds.json · review_rules.json · reports/*.json · SHA256SUMS              │
└────────────────────────────────────────────────────────────────────────────────────┘
                                   │ tải 1 lần khi khởi động
                                   ▼
┌──────────────────────── ONLINE (ứng dụng demo trên EC2) ───────────────────────────┐
│                                                                                    │
│  Trình duyệt ──► Nginx ──► React (SPA)                                             │
│                     │                                                              │
│                     └──► FastAPI /api/v1 ──► Model runtime (RAM)                   │
│                               │                 preprocess → predict →             │
│                               │                 calibrate → SHAP → review flag     │
│                               ▼                                                    │
│                          PostgreSQL (predictions, reviews, audit)                  │
└────────────────────────────────────────────────────────────────────────────────────┘
```

## 2. Thành phần và trách nhiệm

| Thành phần | Công nghệ | Chạy ở đâu | Ghi chú |
|---|---|---|---|
| Lưu trữ dữ liệu | S3 (private, Block Public Access bật) | AWS us-east-1 | Bronze/Silver/Gold/artifacts |
| Xử lý dữ liệu | PySpark 3.5 | EC2 (Docker: master + workers) | Đọc `s3a://` bằng IAM Role, không dùng access key |
| Thực nghiệm | Python, scikit-learn, XGBoost, LightGBM, Optuna, SHAP | EC2 / máy cá nhân với dữ liệu Gold | Không đưa dữ liệu thật lên dịch vụ bên thứ ba hay LLM online |
| API | FastAPI + Uvicorn | EC2 Docker | Không gọi `fit()` khi dự đoán |
| Giao diện | React + Vite | EC2 Docker (Nginx) | Chỉ hiển thị ca giả lập/demo |
| CSDL ứng dụng | PostgreSQL 16 | EC2 Docker | Lịch sử và hàng đợi review |

## 3. Bảo mật và chi phí

- IAM Role gắn cho EC2 chỉ có quyền đọc/ghi bucket của nhóm. Mỗi thành viên dùng IAM user riêng có MFA, **không dùng root**.
- Bật AWS Budget alert. Tắt EC2 khi không chạy job.
- Repo GitHub **không** chứa dữ liệu MIMIC, file `.env` hay key (xem `.gitignore`).

## 4. Màn hình dự kiến (GĐ6)

1. **Dự đoán:** form nhập đặc trưng (hoặc chọn ca demo), hiện xác suất và mức nguy cơ. (TV1)
2. **Hàng đợi review:** danh sách, lý do gắn cờ, cập nhật trạng thái. (TV2)
3. **Giải thích:** SHAP waterfall của ca, cùng SHAP tổng thể. (TV3)
4. **Chất lượng đầu vào và độ bền:** cảnh báo thiếu/bất thường, biểu đồ suy giảm khi thiếu dữ liệu. (TV4)
5. **Lịch sử và báo cáo:** lịch sử dự đoán, bảng so sánh mô hình, subgroup, drift offline. (TV5)

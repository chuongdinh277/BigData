# Hợp đồng API (sơ bộ v0.1)

> **Chủ sở hữu:** TV5 · Backend: FastAPI · Frontend: React · Lưu trữ ứng dụng: PostgreSQL
> Mọi response đều có `task`, `model_version`, `prediction_time`, `disclaimer`.

## Quy ước

- Base URL: `/api/v1`
- `task` ∈ `p1` | `p2` | `p3`. Chỉ các task đã có model hợp lệ mới bật (xem `GET /tasks`).
- Đặc trưng đầu vào **khớp tên cột** trong `contracts/gold_schema_<task>.json`. Trường thiếu thì gửi `null`, không tự điền 0.
- `disclaimer` cố định: `"Research prototype – not for clinical use"`.

## Endpoints

| Method | Path | Mô tả | Phụ trách GĐ6 |
|---|---|---|---|
| GET | `/health` | Trạng thái dịch vụ, model đã nạp | TV1 |
| GET | `/tasks` | Danh sách task đang bật và phiên bản model | TV1 |
| GET | `/schema/{task}` | Danh sách trường đầu vào, kiểu, khoảng hợp lệ | TV4 |
| POST | `/predict/{task}` | Dự đoán + hiệu chỉnh + review flag (+ SHAP tùy chọn) | TV1, TV2 |
| POST | `/explain/{task}` | SHAP chi tiết cho một ca | TV3 |
| POST | `/validate/{task}` | Kiểm tra chất lượng đầu vào, không dự đoán | TV4 |
| GET | `/predictions` | Lịch sử dự đoán (lọc theo task, ngày) | TV5 |
| GET | `/reviews` | Hàng đợi xem xét (lọc theo trạng thái) | TV2 |
| PATCH | `/reviews/{review_id}` | Cập nhật trạng thái/ghi chú | TV2 |
| GET | `/reports/{task}/{kind}` | Báo cáo offline: `metrics`, `calibration`, `subgroup`, `drift`, `robustness` | TV5 |

## `POST /predict/{task}`

**Request**
```json
{
  "case_id": "demo-0001",
  "features": {
    "age": 78, "gender_male": 1, "admission_type_group": "EMERGENCY",
    "came_from_ed": 1, "in_icu_24h": 0,
    "creatinine_first": 2.1, "creatinine_max": 2.4, "creatinine_missing": 0,
    "ntprobnp_first": null, "ntprobnp_missing": 1
  },
  "options": {"explain": true, "top_k": 10}
}
```

**Response 200**
```json
{
  "task": "p1",
  "case_id": "demo-0001",
  "model_version": "p1-xgb-2026.11.29-a1b2c3",
  "prediction_time": "2026-11-30T10:15:00Z",
  "probability": 0.231,
  "calibrated_probability": 0.187,
  "threshold": 0.15,
  "risk_level": "HIGH",
  "review": {
    "flag": true,
    "reasons": ["HIGH_RISK", "MISSING_KEY_LAB:ntprobnp"],
    "review_id": 42
  },
  "input_quality": {
    "missing_fields": ["ntprobnp_first"],
    "out_of_range_fields": [],
    "warnings": ["Thiếu NT-proBNP: kết quả kém tin cậy hơn"]
  },
  "explanation": {
    "method": "TreeSHAP",
    "base_value": 0.08,
    "top_features": [
      {"feature": "bun_max", "value": 68, "shap": 0.041},
      {"feature": "age", "value": 78, "shap": 0.022}
    ]
  },
  "disclaimer": "Research prototype – not for clinical use"
}
```

**Lỗi:** `422` sai kiểu/thiếu trường bắt buộc · `404` task chưa bật · `503` model chưa nạp.

## Review

Trạng thái: `PENDING → IN_REVIEW → REVIEWED | ESCALATED`

`PATCH /reviews/{id}` body: `{"status": "REVIEWED", "reviewer": "tv2", "decision": "AGREE|DISAGREE|NEED_INFO", "note": "..."}`

## Bảng PostgreSQL dự kiến

| Bảng | Cột chính |
|---|---|
| `predictions` | id, case_id, task, model_version, features_json, probability, calibrated_probability, risk_level, created_at |
| `reviews` | id, prediction_id, trigger_reasons, status, reviewer, decision, note, created_at, updated_at |
| `review_audit` | id, review_id, old_status, new_status, actor, at |

Chỉ lưu **ca giả lập hoặc ca demo đã khử định danh**. Không lưu dữ liệu MIMIC thật trong PostgreSQL của ứng dụng.

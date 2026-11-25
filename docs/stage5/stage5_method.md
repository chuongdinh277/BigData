# Giai đoạn 5 — Độ tin cậy và chọn mô hình: phương pháp và cách chạy

> Mã nguồn: `trustworthy_ai/` (thư viện) · `experiments/stage5/run_stage5.py` (chạy toàn bộ) · `tests/test_trustworthy.py`
> Kết quả thật: `docs/stage5/stage5_report.md` (do script sinh ra trên EC2, chỉ có số liệu tổng hợp)

## 1. Mục tiêu đề bài → phần đáp ứng

| Mục tiêu (đề Subject 4) | Cách làm ở GĐ5 | Module |
|---|---|---|
| Huấn luyện, so sánh mô hình cơ bản | CV 5 fold theo bệnh nhân cho XGBoost, LightGBM, LR, dùng tham số Optuna của GĐ4; so sánh ghép cặp theo fold | `run_stage5.cross_validate` |
| ROC-AUC, PR-AUC, F1, hiệu chuẩn | ROC/PR-AUC, F1, độ nhạy, PPV, Brier, ECE, slope, O/E, biểu đồ hiệu chỉnh; CI 95% bootstrap theo bệnh nhân | `evaluation.py` |
| Hiệu chỉnh xác suất | none/Platt/isotonic, chọn bằng CV trên dự đoán out-of-fold; ngưỡng LOW/MEDIUM/HIGH | `calibration.py` |
| Giải thích cấp bệnh nhân | TreeSHAP chính xác của XGBoost/LightGBM, Linear SHAP cho LR; top-k mỗi ca; toàn cục, theo nhóm thông tin, độ ổn định giữa các fold | `explain.py` |
| Độ nhạy với dữ liệu thiếu | Che 0→100% theo 5 kịch bản (xét nghiệm cơ bản, xét nghiệm đặc hiệu, sinh hiệu, OMR, ngẫu nhiên) + phân tầng theo mức thiếu | `robustness.py` |
| Nhất quán giữa nhóm bệnh nhân | 11 thuộc tính (giới, tuổi, giai đoạn, ICU, HF chính/phụ, ICD, loại nhập viện, NT-proBNP, mức thiếu, chủng tộc, bảo hiểm); ROC/PR lift/O/E/độ nhạy kèm CI; recall gap và FPR gap | `subgroup.py` |
| Drift dữ liệu | PSI từng đặc trưng (giá trị thiếu là một bin), thay đổi tỷ lệ thiếu, drift nhãn, PSI điểm dự đoán | `drift.py` |
| Drift mô hình | Huấn luyện 2008–2013 → đánh giá 2014–2022; hiệu chỉnh lại intercept làm biện pháp xử lý; drift thứ hạng SHAP giữa các giai đoạn | `run_stage5.temporal_validation`, `explanation_drift` |
| Xem xét thủ công | 7 lý do gắn cờ + mức ưu tiên; đo khối lượng hàng đợi và % ca tử vong được bắt | `review.py` |
| Đánh giá test, đóng gói | Mở test một lần (có file khóa `docs/stage5/test_lock.json`), đóng gói đủ thành phần cho FastAPI | `bundle.py` |

## 2. Những điểm thiết kế quan trọng

1. **Không đụng test trước bước cuối.** Mọi phân tích dùng dự đoán out-of-fold trên train+validation (dev). Test chỉ mở với `--open-test`. Lần mở được ghi vào `docs/stage5/test_lock.json` (commit Git). Lần thứ hai bị từ chối nếu không có `--force-reopen-test`, và cờ này cũng được ghi lại.
2. **Sửa điểm lạc quan của GĐ4.** Optuna đã chọn tham số trên validation, nên điểm validation cao hơn thực tế. GĐ5 chọn mô hình bằng CV 5 fold theo bệnh nhân.
3. **Hiệu chỉnh trung thực.** Bộ hiệu chỉnh học trên dự đoán out-of-fold (mô hình chưa thấy dòng đó). Ngưỡng và các phân tích dùng xác suất được hiệu chỉnh chéo (fold j dùng bộ hiệu chỉnh học từ các fold khác).
4. **SHAP chính xác, không phụ thuộc gói `shap`.** Dùng `pred_contribs` của XGBoost và `pred_contrib` của LightGBM. Test kiểm tra tính cộng: base + ΣSHAP = logit(p) với sai số < 1e-3. SHAP ở thang log-odds trước hiệu chỉnh. Hiệu chỉnh đơn điệu nên hướng tác động giữ nguyên.
5. **Thiếu ≠ bình thường.** Khi che một xét nghiệm, cờ `_missing` cũng được bật. Gói mô hình tự suy ra cờ thiếu nếu người dùng không gửi.
6. **Quyền riêng tư (DUA PhysioNet).** Báo cáo chỉ có số tổng hợp. Nhóm có < 10 ca bị che cả số đếm lẫn tỷ lệ. Hình chỉ là biểu đồ tổng hợp (không beeswarm từng bệnh nhân). Gói mô hình (`artifacts/models/`) không commit Git, chỉ lưu trên S3.

## 3. Cách chạy trên EC2

```bash
source ~/venv-hg/bin/activate && pip install -r requirements.txt
git pull   # cần các file experiments/{khoe,chien,hung}/results.csv của GĐ4

# Bước A: chạy GĐ5, test VẪN ĐÓNG (~15–30 phút với --jobs 8)
python -m experiments.stage5.run_stage5 --jobs 8
#  → docs/stage5/stage5_report.md + docs/stage5/figures/*.png   (commit được)
#  → reports/stage5/<run_id>/stage5_results.json                  (không commit)
#  → artifacts/models/p1/<version>/                               (gói mô hình, không commit)

# Nhóm đọc báo cáo, chốt mô hình (ghi D-016 trong decision_log). Muốn ép mô hình khác: --model LightGBM

# Bước B: mở test MỘT lần, tải gói mô hình lên S3
python -m experiments.stage5.run_stage5 --jobs 8 --open-test --upload
git add docs/stage5 && git commit -m "GĐ5: kết quả độ tin cậy và test"
```

Thử trên máy cá nhân với dữ liệu giả (không cần quyền AWS):

```bash
python tools/make_synthetic_gold.py --rows 6000 --out data/gold_fake.csv
python -m experiments.common.split --gold data/gold_fake.csv --out data/splits/split_fake.json
python -m experiments.stage5.run_stage5 --gold data/gold_fake.csv --split data/splits/split_fake.json --synthetic --quick --open-test
python -m pytest -q tests/test_trustworthy.py
```

Với dữ liệu giả, báo cáo ghi vào `reports/stage5/<run_id>/docs/` và gói mô hình có tiền tố `synthetic-`, nên không lẫn với kết quả thật.

## 4. Dùng gói mô hình ở GĐ6 (FastAPI)

```python
from trustworthy_ai.bundle import HeartGuardModel
model = HeartGuardModel("artifacts/models/p1/<version>")     # nạp 1 lần lúc khởi động, kiểm SHA256
res = model.predict({"age": 78, "gender_male": 1, "creatinine_first": 2.1, "ntprobnp_first": None})[0]
# res: probability, calibrated_probability, uncertainty_std, threshold, risk_level,
#      review{flag, reasons, priority}, input_quality{missing_fields, out_of_range_fields, warnings},
#      explanation{method, base_value, top_features[...]}, model_version, prediction_time, disclaimer
model.drift_report(batch_dataframe)   # PSI so với tham chiếu, dùng cho GET /reports/p1/drift
```

Kết quả khớp `contracts/api_contract.md`. FastAPI chỉ cần thêm `task`, `case_id`, `review_id` (từ PostgreSQL).

## 5. Phân công GĐ5 (theo đề cương) → phần cần rà và chịu trách nhiệm

| Người | Việc trong đề cương | File rà soát / mở rộng |
|---|---|---|
| Hùng | Tổng hợp ứng viên, đóng gói mô hình cuối, chạy trên EC2 | `run_stage5.py` (CV, chọn mô hình), `bundle.py` |
| Hà | Hiệu chỉnh xác suất, ngưỡng, quy tắc xem lại | `calibration.py`, `review.py` (chỉnh band NEAR_THRESHOLD, phân vị LOW_CONFIDENCE theo khối lượng hàng đợi mong muốn) |
| Khoẻ | Giải thích SHAP | `explain.py`, hình SHAP cho slide |
| Chiến | Độ bền khi thiếu dữ liệu, bộ tiền xử lý | `robustness.py` (thêm kịch bản nếu cần) |
| Chưởng | Nhóm bệnh nhân, drift, định dạng gói, nạp từ S3 | `subgroup.py`, `drift.py`, `report.py` |

## 6. Giới hạn đã biết

- Ngưỡng LOW_CONFIDENCE lấy theo phân vị trên dev, mà mỗi dòng dev đã được 4/5 mô hình fold nhìn thấy. Vì vậy độ bất đồng trên dữ liệu mới cao hơn, tỷ lệ cờ thực tế cao hơn dự kiến. Hãy đọc số trên test.
- Drift chỉ đánh giá theo lô theo `anchor_year_group` (thời gian bị dịch chuyển, chỉ biết nhóm năm). Không giám sát thời gian thực.
- Mô hình cuối huấn luyện trên toàn bộ dev, còn bộ hiệu chỉnh học từ mô hình fold (khác nhau một chút về lượng dữ liệu). Đây là cách cross-fitting thông dụng.
- Chưa có mô hình lõi riêng (OSMA-Net) và chưa có mô hình cho P2/P3. Khung GĐ5 dùng lại được khi có Gold P2/P3.

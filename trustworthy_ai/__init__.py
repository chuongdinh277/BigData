"""HeartGuard AI — Giai đoạn 5: độ tin cậy của mô hình.

Các module:
    evaluation   chỉ số phân loại + hiệu chỉnh (ECE, slope, O/E), bootstrap CI theo bệnh nhân
    calibration  Platt / isotonic, chọn phương pháp bằng CV trên dự đoán out-of-fold; chọn ngưỡng
    explain      SHAP chính xác (TreeSHAP gốc của XGBoost/LightGBM, Linear SHAP cho LR), gộp về cột gốc
    robustness   độ bền khi thiếu dữ liệu (che dần thông tin đầu vào)
    subgroup     hiệu năng và hiệu chỉnh theo nhóm bệnh nhân, chênh lệch công bằng
    drift        PSI dữ liệu/điểm dự đoán, drift nhãn, đánh giá theo thời gian
    review       luật gắn cờ xem lại thủ công và đánh giá khối lượng hàng đợi
    bundle       đóng gói mô hình cuối (model + calibrator + rules + drift reference) và dự đoán 1 ca

Mọi đầu ra ghi vào báo cáo chỉ là số liệu tổng hợp; ô có số ca < 10 được che.
"""

DISCLAIMER = "Research prototype – not for clinical use"

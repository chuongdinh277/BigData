"""Khung thí nghiệm chung GĐ3 (HeartGuard AI, P1).

Luồng: load_gold → make_split/apply_split → fit trên train → chọn ngưỡng trên validation
→ evaluate trên validation → write_result. Tập test KHÔNG được dùng trước GĐ5.

Dùng:  from experiments.common.runner import run_case, baseline_pipeline
       from experiments.common.data import load_gold, feature_columns
"""

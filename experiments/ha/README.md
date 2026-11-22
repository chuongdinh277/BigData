# Random Forest (Hà)

Code của Hà nằm ở nhánh `ha/gd4` (`experiments/member_2/ha_random_forest.py`).
Hà không chạy được trên EC2 nên Chưởng chạy hộ trên `heartguard-ec2` (22/11/2026, commit 736f705 của nhánh `ha/gd4`), dùng Gold P1 V1 và manifest chia tập đã khóa.
`results.csv` là kết quả trên validation, 3 lần chạy: mặc định, class_weight=balanced, Optuna 50 trial.

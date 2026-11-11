# Dataset v1 — Gold P1 (tử vong trong viện, mốc +24h)

Tạo lúc: 2026-11-11T15:48:51+00:00 · Commit: `48e9e63` · Trạng thái: **PASSED**

- Vị trí: `s3a://heartguard-mimic-group11/gold/p1/v1`
- Nguồn nhãn: nhãn của Hà: s3a://heartguard-mimic-group11/processed/labels/p1 (cột label_mortality)
- Số dòng (lần nhập viện): 73,768 · Số bệnh nhân: 30,287 · Số cột: 124
- Ca dương: 3,588 (4.86%)

## Kiểm tra chéo

| Kiểm tra | Kết quả | Chi tiết |
|---|---|---|
| hadm_id duy nhất | ✅ | 73768 dòng / 73768 hadm_id |
| Không có nhãn NULL | ✅ | 0 nhãn NULL |
| Nhãn chỉ 0/1 | ✅ | giá trị: [0, 1] |
| Không có cột bị cấm | ✅ | — |
| Thiếu cột so với contract | ✅ | — |
| prediction_time = admittime + 24h | ✅ | 0 dòng lệch |
| Số dòng Gold = số ca P1 tính lại từ cohort | ✅ | Gold 73768 · cohort P1 73768 |
| rows = 73768 (kỳ vọng) | ✅ | thực tế 73768 |
| positives = 3588 (kỳ vọng) | ✅ | thực tế 3588 |

Quality (Chiến): **NOT_CHECKED** — chưa có lineage của features nên phần leakage theo nguồn chưa kiểm được

## Theo anchor_year_group (cho drift)

| anchor_year_group | n | Ca dương | Tỷ lệ |
|---|---:|---:|---:|
| 2008 - 2010 | 34553 | 1310 | 3.79% |
| 2011 - 2013 | 14062 | 683 | 4.86% |
| 2014 - 2016 | 11267 | 619 | 5.49% |
| 2017 - 2019 | 8769 | 566 | 6.45% |
| 2020 - 2022 | 5117 | 410 | 8.01% |

## Thời gian từng bước

| Bước | Trạng thái | Giây |
|---|---|---:|
| validate | OK | 27.9 |

Chỉ chứa số liệu tổng hợp; ô < 10 đã được che.

# Kết quả cohort suy tim (Giai đoạn 2)

Mã HF ở bất kỳ `seq_num`. Mã ICD theo `configs/hf_codes.yaml`.

## Bảng đếm qua từng bước lọc

| Bước | Điều kiện | Số bệnh nhân | Số lần nhập viện | Bị loại (nhập viện) |
|---|---|---:|---:|---:|
| S0 | Toàn bộ admissions | 223,452 | 546,028 |  |
| S1 | Có mã suy tim (HF) ở bất kỳ seq_num | 31,428 | 80,767 | 465,261 |
| S2 | Có admittime/dischtime và dischtime > admittime | 31,423 | 80,746 | 21 |
| S3 | Có hồ sơ patients và tuổi tại nhập viện >= 18 | 31,423 | 80,746 | 0 |

## Thông tin thêm về cohort cuối

- Số lần nhập viện: 80,746
- Số bệnh nhân: 31,423
- Suy tim là chẩn đoán chính (`hf_primary = 1`): 15,105
- Suy tim là chẩn đoán phụ (`hf_primary = 0`): 65,641
- Tử vong trong viện (`hospital_expire_flag = 1`): 4,005

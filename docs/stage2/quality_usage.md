# Phần việc của Chiến — hướng dẫn chạy và tích hợp

Nhánh: `chien/quality`. Các lệnh dưới đây chạy từ thư mục gốc repo.
Kết quả đo được ghi tại `docs/stage2/quality_report.md`.

## 1. Tạo dữ liệu giả và khảo sát

Dùng Python 3.11/3.12 và Java 17. Môi trường tối thiểu chỉ cài thư viện cho
phần chất lượng, không cần cài toàn bộ thư viện ML/API:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-quality.txt
# JAVA_HOME phải trỏ tới Java 17 đã cài trên máy.
export HG_LOCAL=1
export HG_SPARK_MASTER='local[2]'
export HG_SHUFFLE_PARTITIONS=4
export HG_DRIVER_MEMORY=2g
export SPARK_LOCAL_IP=127.0.0.1

python tools/make_synthetic_mimic.py --patients 800 --seed 42
python -m data_spark.stage1_feasibility \
  --root sample_data/mimic_fake --out reports/stage1_fake --with-chartevents
```

`HG_LOCAL=1` bỏ tải thư viện S3; dữ liệu chỉ được tạo và đọc trên máy.
Đường dẫn và seed được ghi rõ để nhóm tái lập kết quả.
Tham khảo môi trường: [tài liệu chính thức PySpark 3.5.3](https://archive.apache.org/dist/spark/docs/3.5.3/api/python/getting_started/install.html).

## 2. Danh sách xét nghiệm và sinh hiệu

`configs/feature_items.yaml` có 20 xét nghiệm và 7 nhóm sinh hiệu, mỗi mục có:

- `itemids`: mã được chọn để triển khai thử trên synthetic.
- `unit_hint`: đơn vị đầu vào phải khớp; INR không có đơn vị, NULL/rỗng được chuẩn hóa thành chuỗi rỗng.
- `valid_range`: khoảng hợp lệ, lấy nguyên từ `docs/stage1/feature_spec.md`.
- `feature_columns`: tên cột tổng hợp sinh hiệu theo hợp đồng Gold.
- Nhiệt độ dùng Fahrenheit đầu vào; đổi `(F − 32) × 5/9` trước kiểm tra khoảng 30–43 °C.

Giữ `label_regex` và `fluid` để script khảo sát tiếp tục phát hiện mã ứng viên
trên dictionary thật. Các mã ABP/Celsius khác **chưa được thêm vào danh sách
chọn** khi chưa có kết quả khảo sát. Không tự động gộp các mã khác đơn vị.

Trạng thái `synthetic_verified_pending_real` là chủ ý: đã có cấu hình dùng để
code song song, nhưng chưa tuyên bố chốt trên MIMIC thật. Khi có kết quả thật,
đối chiếu dictionary, fluid, `dominant_unit`, `dominant_unit_share >= 0.95`,
độ phủ và tỷ lệ `storetime` sau mốc. Không dùng độ phủ synthetic thay thế số liệu thật.

Kiểm tra đơn vị/giá trị trên event giả:

```bash
python -m data_spark.quality --kind labs --format csv \
  --table sample_data/mimic_fake/hosp/labevents.csv.gz \
  --out reports/quality/lab_units --synthetic
python -m data_spark.quality --kind vitals --format csv \
  --table sample_data/mimic_fake/icu/chartevents.csv.gz \
  --out reports/quality/vital_units --synthetic
```

Các báo cáo đếm riêng đơn vị sai, giá trị số thiếu, giá trị có chữ nhưng không
có `valuenum` (ví dụ `<0.01`), và giá trị ngoài khoảng. GĐ2 chọn bỏ và đếm
numeric-text; không tự chuyển `<0.01` thành `0.01`. Đơn vị đúng phải chiếm ít
nhất 95% tất cả event của từng item. Item không có dòng => `NOT_CHECKED`.
`PASSED` của chế độ này chỉ xác nhận tỷ lệ đơn vị; các số đếm giá trị lỗi vẫn
phải được xử lý trước tạo feature.

## 3. Kiểm tra một bảng bất kỳ

```bash
python -m data_spark.quality --kind table --table /path/to/table.parquet \
  --keys hadm_id --out reports/quality/table
```

`--keys` có thể nhận nhiều cột để kiểm tra bảng có khóa ghép. Báo cáo gồm:
missing theo từng cột và theo `anchor_year_group` nếu có; số khóa trùng,
số dòng nằm trong nhóm trùng, số dòng dư; khóa NULL; số giá trị ngoài khoảng.
NULL, NaN và chuỗi rỗng/trắng được tính thiếu. Các giá trị thiếu được giữ
nguyên để bước imputation sau này chỉ fit trên train.

API `null_invalid_values(df)` trả về DataFrame mới với giá trị ngoài khoảng
được gán NULL. Nó không sửa input và không tự điền giá trị thiếu. Chạy profile
**trước** khi gán NULL để giữ số liệu giá trị lỗi ban đầu.

## 4. Kiểm tra Gold P1 và leakage

```bash
python -m data_spark.quality --kind p1 \
  --table /path/to/features.parquet \
  --context /path/to/cohort.parquet \
  --lineage /path/to/feature_lineage.parquet \
  --out reports/quality/p1
```

Có ba đầu vào riêng:

1. **features**: bảng cần kiểm tra, khóa `hadm_id`, có `subject_id`, `admittime`,
   `prediction_time` và các feature theo contract. Có thể kiểm tra một tập con
   feature khi phát triển; kết quả không chứng nhận đã đủ toàn bộ schema Gold.
2. **context**: bảng admissions/cohort có `hadm_id`, `admittime`, `dischtime`;
   nên có `subject_id` để kiểm tra bệnh nhân khớp. Dùng nó để kiểm tra mốc
   `admittime + 24h` và `admittime < prediction_time < dischtime`. Context phải
   duy nhất theo `hadm_id`. Bảng này không phải input cho mô hình.
3. **lineage**: bảng dài, mỗi dòng ghi một nguồn đóng góp vào một feature.
   Khoẻ cần xuất nó ngay từ các event **thực sự đã dùng** khi tổng hợp.

| Cột lineage | Ý nghĩa |
|---|---|
| `subject_id`, `hadm_id` | Bệnh nhân/lượt nhập viện đích |
| `feature_name` | Tên feature trong contract, ví dụ `sodium_max` |
| `source_table` | `labevents`, `chartevents`, `icustays`, `omr`, `admissions`, `diagnoses_icd`; chấp nhận tiền tố `hosp/`, `icu/`, `hosp.`, `icu.` |
| `source_subject_id` | Bệnh nhân của dòng nguồn, phải khớp đích |
| `source_hadm_id` | Nhập viện nguồn; lab/OMR có thể NULL nếu ghép theo bệnh nhân và cửa sổ |
| `source_charttime` | Thời gian đo; ICU dùng `intime`; tiền sử dùng thời gian nguồn đã có |
| `source_storetime` | Thời gian kết quả có sẵn; lab/vital bắt buộc, OMR/ICU có thể NULL vì nguồn không có trường này |
| `source_dischtime` | Bắt buộc cho tiền sử từ admission trước; các nguồn khác NULL |

Quy tắc chính:

Reader dùng chung đã giữ thêm `labevents.value` và
`chartevents.value/storetime`. Nếu Silver được sinh từ phiên bản cũ, Hùng cần
chuyển lại hai bảng này từ Bronze trước khi chạy quality. Reader sẽ báo rõ
cột thiếu; không tự điền `storetime = charttime`.

- Lab dùng cửa sổ đóng `[admittime − 6h, prediction_time]`.
- Vital và ICU dùng `[admittime, prediction_time]`; không mượn lookback 6h của lab.
- `charttime <= prediction_time` và `storetime <= prediction_time`. Kết quả
  trả muộn, timestamp thiếu hoặc `storetime < charttime` đều bị báo lỗi.
- Tiền sử phải thuộc admission khác và đã xuất viện **trước** admission hiện
  tại. OMR chỉ lấy ngày trước nhập viện, trong 365 ngày.
- Mỗi feature phụ thuộc thời gian có giá trị phải có bằng chứng. Giá trị NULL,
  cờ missing, `first_admission`, và số đếm/cờ bằng 0 không yêu cầu event tồn tại.
- Cột cấm được lấy từ contract và bổ sung `last_careunit`, ICD/DRG hiện tại.
  Cột lạ cũng bị chặn để tránh đổi tên cột leakage rồi đưa vào Gold.
- ID, nhãn và cột subgroup/split có thể tồn tại trong bảng lưu trữ, nhưng
  `check_leakage(..., model_columns=[...])` chỉ cho phép cột feature thực sự
  làm input mô hình; `hf_primary`, nhãn, chủng tộc, bảo hiểm không được lọt vào.

Thiếu context/lineage => **NOT_CHECKED**, CLI trả exit code 1. Phát hiện lỗi
=> **FAILED**, exit code 1. Chỉ đủ kiểm tra và không có lỗi => **PASSED**,
exit code 0. Missing rate được báo cáo nhưng không tự làm thất bại, vì đặc
tả cho phép thiếu xét nghiệm và sinh hiệu.

Giới hạn: bộ kiểm tra xác nhận provenance được cung cấp; không thể phát hiện
pipeline che giấu event đã dùng hoặc preprocessing fit trên toàn bộ dataset.
Các hành vi đó cần review code feature/split ở giai đoạn tương ứng. Kết quả
chất lượng của một fixture không chứng minh toàn bộ dataset thật đã đạt.

## 5. Fixture chạy thử và ca cố tình sai

```bash
python tools/make_quality_fixture.py
python -m data_spark.quality --table sample_data/quality_fixture/features.parquet \
  --context sample_data/quality_fixture/context.parquet \
  --lineage sample_data/quality_fixture/lineage.parquet \
  --out reports/quality/p1_fake --synthetic

python tools/make_quality_fixture.py --inject-errors --out sample_data/quality_fixture_bad
python -m data_spark.quality --table sample_data/quality_fixture_bad/features.parquet \
  --context sample_data/quality_fixture_bad/context.parquet \
  --lineage sample_data/quality_fixture_bad/lineage.parquet \
  --out reports/quality/p1_fake_bad --synthetic
# Lệnh thứ hai phải trả exit code 1.

python -m pytest -q tests
```

Fixture lấy tối đa 100 admission hợp lệ từ MIMIC giả, tạo thống kê lab/vital
và provenance trực tiếp từ event. Nó không lọc cohort suy tim, không thay
thế `cohort.py`, `labels.py`, `features.py` của các thành viên khác. Mẫu lỗi
thêm trùng admission, sodium=999 và event/giờ trả kết quả sau mốc dự đoán.
Dữ liệu sinh ra và reports đều bị Git ignore; Git chỉ giữ code/cấu hình và
báo cáo tổng hợp đã chọn.

## 6. Chạy dữ liệu thật và ghi S3 khi nhóm sẵn sàng

Trên môi trường nhóm được cấp quyền, bỏ `HG_LOCAL=1` và `--synthetic`:

```bash
unset HG_LOCAL
python -m data_spark.quality --kind p1 \
  --table s3a://heartguard-mimic-group11/processed/features/p1/ \
  --context s3a://heartguard-mimic-group11/processed/cohort/ \
  --lineage s3a://heartguard-mimic-group11/processed/features/p1_lineage/ \
  --out s3a://heartguard-mimic-group11/reports/quality/p1/
```

Đường dẫn lineage là hợp đồng bàn giao mới cần Khoẻ xuất; không giả định đã
tồn tại. Nếu nhóm dùng Gold thay cho processed, thay `--table` tương ứng.
Writer dùng Hadoop S3A và IAM Role hiện có, ghi hai file tổng hợp
`quality_report.json`, `quality_report.md`. Ô đếm nhỏ được che theo
`configs/data.yaml`. Không collect event hoặc dòng bệnh nhân vào báo cáo.
Nhánh này chưa xác nhận quyền AWS, chưa đọc MIMIC thật và chưa ghi S3.

#!/usr/bin/env bash
# Chạy dự án trên EC2 theo LƯỢT (chỉ người vận hành chạy). Mỗi bước ghi log riêng; bước tạo dữ liệu lỗi thì
# dừng cả lượt (các bước sau dùng kết quả của nó); bước độc lập lỗi thì ghi lại và chạy tiếp.
#
#   tmux new -s hg                       # chạy trong tmux để mất SSH không chết job
#   source ~/venv-hg/bin/activate
#   bash tools/run_batch.sh A            # dữ liệu: Gold P1 v1.1, P2, P3 + kiểm tra chất lượng + chia tập
#   bash tools/run_batch.sh B            # mô hình + đánh giá độ tin cậy 3 bài + phân tích C6
#   bash tools/run_batch.sh C            # benchmark (máy chỉ chạy việc này)
#   bash tools/run_batch.sh D            # SAU BUỔI HỌP CHỐT: mở test 3 bài + tải gói lên S3
#   bash tools/run_batch.sh AB           # A rồi B liền một mạch (qua đêm)
# Tuỳ chọn:  SHUTDOWN=1 bash tools/run_batch.sh AB   → tự tắt máy khi xong (tránh tốn tiền nếu ngủ quên)
#            BENCH_CORES="1 2 4 8" BENCH_REPEATS=3      → cấu hình benchmark (Hùng đề xuất)
# Sau mỗi lượt chạy: python tools/check_results.py --stage <A|B|D> (cổng kiểm tra chất lượng).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUTF8=1
VERSION="${VERSION:-v1_1}"
B="s3a://heartguard-mimic-group11"
B3="s3://heartguard-mimic-group11"
RUN_ID="$(date -u +%Y%m%dT%H%MZ)"
LOG_DIR="$ROOT/reports/runs/$RUN_ID"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.tsv"
printf "lượt\tbước\ttrạng thái\tgiây\tlog\n" > "$SUMMARY"
FAILED=0

# step <lượt> <tên> <critical:1|0> <lệnh...>
step() {
  local batch="$1" name="$2" critical="$3"; shift 3
  local log="$LOG_DIR/${batch}_${name}.log" t0=$SECONDS
  echo "[$(date +%H:%M:%S)] ▶ $batch/$name"
  if "$@" > "$log" 2>&1; then
    printf "%s\t%s\tOK\t%s\t%s\n" "$batch" "$name" "$((SECONDS - t0))" "$log" >> "$SUMMARY"
    echo "[$(date +%H:%M:%S)] ✓ $batch/$name ($((SECONDS - t0))s)"
  else
    printf "%s\t%s\tLỖI\t%s\t%s\n" "$batch" "$name" "$((SECONDS - t0))" "$log" >> "$SUMMARY"
    echo "[$(date +%H:%M:%S)] ✗ $batch/$name — xem $log"
    FAILED=1
    if [ "$critical" = "1" ]; then
      echo "Bước tạo dữ liệu lỗi → dừng lượt $batch (các bước sau phụ thuộc bước này)."
      finish 1
    fi
  fi
}

finish() {
  echo; column -t -s $'\t' "$SUMMARY" || cat "$SUMMARY"
  aws s3 sync "$LOG_DIR" "$B3/experiments/logs/$RUN_ID/" --quiet || echo "(không đồng bộ được log lên S3)"
  git rev-parse --short HEAD > "$LOG_DIR/commit.txt" 2>/dev/null || true
  if [ "${SHUTDOWN:-0}" = "1" ]; then echo "Tắt máy sau 2 phút (huỷ: sudo shutdown -c)"; sudo shutdown -h +2; fi
  exit "${1:-$FAILED}"
}

batch_A() {
  step A table_sizes 0 python -m data_spark.scaling sizes
  # P1: dùng nhãn đã có của Hà (không chạy lại cohort/nhãn → không ghi đè dữ liệu thật)
  step A gold_p1 1 python -m data_spark.pipeline_runner --task p1 --steps features validate --version "$VERSION" \
       --labels "$B/processed/labels/p1" --label-col label_mortality --expect-rows 73768 --expect-positives 3588
  step A gold_p2 1 python -m data_spark.pipeline_runner --task p2 --steps features --version "$VERSION"
  step A gold_p3 1 python -m data_spark.pipeline_runner --task p3 --steps features --version "$VERSION"
  for t in p2 p3; do
    step A "split_$t" 1 env HG_TASK=$t python -m ml.common.split --gold "$B3/gold/$t/$VERSION/" \
         --out "$B3/splits/$t/v1/split_manifest.json"
  done
  echo "→ Ghi sha256 của split P2/P3 (trong log split_p2/p3) vào configs/splits.yaml rồi commit."
}

batch_B() {
  local G1="$B3/gold/p1/$VERSION/" S1="$B3/splits/p1/v1/split_manifest.json"
  step B train_lr 0 python -m ml.models.logistic --gold "$G1" --split "$S1" --trials 50
  step B train_xgb 0 python -m ml.models.xgboost_model --gold "$G1" --split "$S1" --trials 50 --jobs 8
  step B train_lgbm 0 python -m ml.models.lightgbm_model --gold "$G1" --split "$S1" --trials 50 --jobs 8
  step B compare 0 python -m ml.compare
  for t in p1 p2 p3; do
    step B "trust_$t" 0 env HG_TASK=$t python -m ml.trust.run --gold "$B3/gold/$t/$VERSION/" \
         --split "$B3/splits/$t/v1/split_manifest.json" --jobs 8
  done
  # C6: bỏ phép đo có storetime rỗng → Gold riêng → đánh giá (không đóng gói mô hình)
  step B gold_c6 0 python -m data_spark.features --drop-null-storetime --skip-verify \
       --cohort "$B/processed/cohort" --labels "$B/processed/labels/p1" --label-col label_mortality \
       --out "$B/gold/p1/_c6_no_null_storetime" --report results/data/features_p1_c6.md
  step B trust_c6 0 env HG_TASK=p1 python -m ml.trust.run --gold "$B3/gold/p1/_c6_no_null_storetime/" \
       --split "$B3/splits/p1/v1/split_manifest.json" --jobs 8 --skip-extra --no-bundle \
       --docs-dir results/trust/p1_c6
}

batch_C() {
  step C bench_funnel 0 python -m data_spark.scaling funnel
  step C bench_pandas 0 python -m data_spark.scaling pandas
  step C bench_strong 0 python -m data_spark.scaling strong --cores ${BENCH_CORES:-1 2 4 8} --repeats "${BENCH_REPEATS:-3}"
  step C bench_weak 0 python -m data_spark.scaling weak --cores 2 4 8 --repeats "${BENCH_REPEATS:-3}"
  step C bench_ablation 0 python -m data_spark.scaling ablation --cores 8 --repeats 2
  step C bench_report 0 python -m data_spark.scaling report
}

batch_D() {
  read -r -p "Nhóm đã họp chốt mô hình và check_results --stage B đã PASS? Gõ MO-TEST để tiếp tục: " ok
  [ "$ok" = "MO-TEST" ] || { echo "Huỷ."; exit 1; }
  for t in p1 p2 p3; do
    local extra=(); var="MODEL_${t^^}"; [ -n "${!var:-}" ] && extra=(--model "${!var}")
    step D "open_test_$t" 0 env HG_TASK=$t python -m ml.trust.run --gold "$B3/gold/$t/$VERSION/" \
         --split "$B3/splits/$t/v1/split_manifest.json" --jobs 8 --open-test --upload "${extra[@]}"
  done
}

case "${1:-}" in
  A) batch_A ;; B) batch_B ;; C) batch_C ;; D) batch_D ;;
  AB) batch_A; batch_B ;;
  *) echo "Dùng: bash tools/run_batch.sh A|B|C|D|AB   (D: đặt MODEL_P1=LogisticRegression … nếu nhóm chọn mô hình khác mô hình tốt nhất theo CV)"; exit 1 ;;
esac
finish "$FAILED"

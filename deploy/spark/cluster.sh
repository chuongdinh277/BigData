#!/usr/bin/env bash
# Cụm Spark standalone trên EC2 (1 master + N worker), dùng chính PySpark đã cài trong venv.
#
# Trên máy MASTER (vd heartguard-ec2):
#   deploy/spark/cluster.sh master                       # web UI: http://<ip-master>:8081 (Security Group nội bộ)
#   deploy/spark/cluster.sh worker spark://<ip-private-master>:7077 4 12g   # có thể chạy worker ngay trên master
# Trên mỗi máy WORKER khác (cùng VPC, cùng IAM Role đọc S3):
#   deploy/spark/cluster.sh worker spark://<ip-private-master>:7077 4 12g
# Chạy job lên cụm:
#   export HG_SPARK_MASTER=spark://<ip-private-master>:7077
#   export HG_SPARK_CONF="spark.executor.memory=10g,spark.executor.cores=4"
#   python -m data_spark.pipeline_runner --steps features validate --version v1_1
# Dừng:
#   deploy/spark/cluster.sh stop
# Lịch sử job (Spark UI sau khi job kết thúc):
#   export HG_EVENT_LOG=file:///var/tmp/spark-events ; deploy/spark/cluster.sh history   # http://<ip>:18080
set -euo pipefail

SPARK_HOME="${SPARK_HOME:-$(python -c 'import os, pyspark; print(os.path.dirname(pyspark.__file__))')}"
LOG_DIR="${HG_SPARK_LOG_DIR:-/var/tmp/heartguard-spark}"
EVENTS="${HG_EVENT_LOG:-file:///var/tmp/spark-events}"
mkdir -p "$LOG_DIR" "${EVENTS#file://}"
PRIVATE_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"

case "${1:-}" in
  master)
    nohup "$SPARK_HOME/bin/spark-class" org.apache.spark.deploy.master.Master \
      --host "${PRIVATE_IP}" --port 7077 --webui-port 8081 > "$LOG_DIR/master.log" 2>&1 &
    echo $! > "$LOG_DIR/master.pid"
    echo "Master: spark://${PRIVATE_IP}:7077 · UI http://${PRIVATE_IP}:8081 · log $LOG_DIR/master.log"
    ;;
  worker)
    MASTER="${2:?Cần URL master, vd spark://10.0.1.23:7077}"
    CORES="${3:-4}"
    MEM="${4:-12g}"
    nohup "$SPARK_HOME/bin/spark-class" org.apache.spark.deploy.worker.Worker \
      --cores "$CORES" --memory "$MEM" --webui-port 8082 "$MASTER" > "$LOG_DIR/worker.log" 2>&1 &
    echo $! > "$LOG_DIR/worker.pid"
    echo "Worker ${PRIVATE_IP} (${CORES} core, ${MEM}) → ${MASTER} · log $LOG_DIR/worker.log"
    ;;
  history)
    SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=${EVENTS}" nohup "$SPARK_HOME/bin/spark-class" \
      org.apache.spark.deploy.history.HistoryServer > "$LOG_DIR/history.log" 2>&1 &
    echo $! > "$LOG_DIR/history.pid"
    echo "History Server: http://${PRIVATE_IP}:18080"
    ;;
  stop)
    for p in master worker history; do
      if [ -f "$LOG_DIR/$p.pid" ]; then kill "$(cat "$LOG_DIR/$p.pid")" 2>/dev/null || true; rm -f "$LOG_DIR/$p.pid"; echo "đã dừng $p"; fi
    done
    ;;
  *)
    echo "Dùng: $0 master | worker <spark://ip:7077> [cores] [mem] | history | stop"; exit 1 ;;
esac

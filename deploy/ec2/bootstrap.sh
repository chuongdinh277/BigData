#!/usr/bin/env bash
# Cài đặt một EC2 mới (Ubuntu 22.04/24.04) cho HeartGuard AI: Java 17, Python 3.12 venv, Docker, mã nguồn.
# Chạy MỘT lần sau khi tạo máy:  bash deploy/ec2/bootstrap.sh
# EC2 phải gắn IAM Role có policy deploy/aws/iam-policy.json (không dùng access key trên máy).
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/chuongdinh277/BigData.git}"
APP_DIR="${APP_DIR:-$HOME/HeartGuard-AI}"

sudo apt-get update -y
sudo apt-get install -y --no-install-recommends openjdk-17-jdk-headless git make curl ca-certificates \
  software-properties-common tmux htop
# Python 3.12 (Spark 3.5 hỗ trợ 3.8–3.12)
if ! command -v python3.12 >/dev/null; then
  sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get update -y
  sudo apt-get install -y python3.12 python3.12-venv
fi
# Docker Engine + Compose plugin
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
fi

[ -d "$APP_DIR/.git" ] || git clone "$REPO_URL" "$APP_DIR"
cd "$APP_DIR"
python3.12 -m venv ~/venv-hg
source ~/venv-hg/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
cp -n .env.example .env || true
sudo mkdir -p /var/tmp/spark-local /var/tmp/spark-events && sudo chown -R "$USER" /var/tmp/spark-local /var/tmp/spark-events

echo
echo "Xong. Đăng xuất/đăng nhập lại để dùng docker không cần sudo. Tiếp theo:"
echo "  source ~/venv-hg/bin/activate && cd $APP_DIR && make check-aws && make test"

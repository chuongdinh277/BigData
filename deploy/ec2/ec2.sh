#!/usr/bin/env bash
# Bật/tắt/xem EC2 của nhóm từ máy cá nhân (người giữ quyền: Hùng). Tắt máy ngay khi không dùng để giữ chi phí.
#   deploy/ec2/ec2.sh status | start | stop | ip
# Cần AWS CLI đã cấu hình với tài khoản IAM cá nhân có MFA. Mã máy đặt trong biến HG_EC2_ID (không ghi vào Git).
set -euo pipefail
ID="${HG_EC2_ID:?Đặt HG_EC2_ID=i-xxxxxxxx}"
REGION="${AWS_REGION:-us-east-1}"
q() { aws ec2 describe-instances --region "$REGION" --instance-ids "$ID" --query "$1" --output text; }

case "${1:-status}" in
  status) q 'Reservations[0].Instances[0].[InstanceType,State.Name,PublicIpAddress]' ;;
  start)  aws ec2 start-instances --region "$REGION" --instance-ids "$ID" >/dev/null
          aws ec2 wait instance-running --region "$REGION" --instance-ids "$ID"
          echo "Đang chạy · IP: $(q 'Reservations[0].Instances[0].PublicIpAddress')" ;;
  stop)   aws ec2 stop-instances --region "$REGION" --instance-ids "$ID" >/dev/null
          aws ec2 wait instance-stopped --region "$REGION" --instance-ids "$ID"
          echo "Đã tắt. Nhớ kiểm tra EBS snapshot / Elastic IP không dùng để tránh phát sinh phí." ;;
  ip)     q 'Reservations[0].Instances[0].PublicIpAddress' ;;
  *)      echo "Dùng: $0 status|start|stop|ip"; exit 1 ;;
esac

"""Kiểm tra bucket S3 của dự án — CHỈ ĐỌC METADATA (tên, số file, dung lượng, manifest tổng hợp).

Không đọc nội dung bảng MIMIC (DUA PhysioNet): script chỉ liệt kê object và đọc các file
`_manifest.json` / `split_manifest.json` ở mức tóm tắt (bỏ danh sách subject_id).

    python tools/check_s3.py                      # toàn bộ các lớp
    python tools/check_s3.py --profile heartguard # dùng AWS profile cụ thể (SSO / IAM cá nhân)

Quyền cần: s3:ListBucket + s3:GetObject trên bucket (IAM Role của EC2 hoặc tài khoản IAM cá nhân có MFA).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

BUCKET = "heartguard-mimic-group11"
# Lớp → các tiền tố con mong đợi (dùng để báo thiếu)
EXPECTED = {
    "bronze/mimiciv/3.1/hosp/": ["patients.csv.gz", "admissions.csv.gz", "diagnoses_icd.csv.gz", "labevents.csv.gz", "omr.csv.gz"],
    "bronze/mimiciv/3.1/icu/": ["icustays.csv.gz", "chartevents.csv.gz", "d_items.csv.gz"],
    "silver/mimiciv/3.1/hosp/": ["patients/", "admissions/", "diagnoses_icd/", "d_icd_diagnoses/", "labevents/", "d_labitems/", "omr/", "transfers/"],
    "silver/mimiciv/3.1/icu/": ["icustays/", "chartevents/", "d_items/"],
    "processed/": ["cohort/", "labels/"],
    "gold/": ["p1/", "p2/", "p3/"],
    "splits/": ["p1/", "p2/", "p3/"],
    "model-artifacts/": ["p1/", "p2/", "p3/"],
}


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:,.1f} {u}"
        n /= 1024
    return f"{n:,.1f} PB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None)
    ap.add_argument("--region", default="us-east-1")
    a = ap.parse_args()
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    s3 = boto3.Session(profile_name=a.profile, region_name=a.region).client("s3")
    try:
        ident = boto3.Session(profile_name=a.profile).client("sts").get_caller_identity()
        print(f"Danh tính AWS: {ident['Arn']}\n")
    except (BotoCoreError, ClientError) as e:
        print(f"[LỖI] Không có thông tin đăng nhập AWS hợp lệ: {e}")
        return 2

    # 1. Tổng quan theo thư mục cấp 2–4
    sizes, counts = defaultdict(int), defaultdict(int)
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET):
        for o in page.get("Contents", []):
            k = o["Key"]
            keys.append(k)
            parts = k.split("/")
            group = "/".join(parts[: 4 if k.startswith(("bronze", "silver")) else 3]) + "/"
            sizes[group] += o["Size"]
            counts[group] += 1
    print(f"{'Tiền tố':60} {'Số file':>8} {'Dung lượng':>12}")
    for g in sorted(sizes):
        print(f"{g:60} {counts[g]:>8,} {human(sizes[g]):>12}")
    print(f"{'TỔNG':60} {len(keys):>8,} {human(sum(sizes.values())):>12}\n")

    # 2. Thiếu gì so với cấu trúc mong đợi
    missing = [f"{p}{c}" for p, cs in EXPECTED.items() for c in cs if not any(k.startswith(p + c) for k in keys)]
    print("Thiếu so với cấu trúc dự án:" if missing else "Đủ mọi tiền tố mong đợi.")
    for m in missing:
        print("  -", m)

    # 3. Manifest Gold và split (chỉ phần tổng hợp)
    print("\nManifest Gold:")
    for k in sorted(k for k in keys if k.startswith("gold/") and k.endswith("_manifest.json")):
        m = json.loads(s3.get_object(Bucket=BUCKET, Key=k)["Body"].read())
        s = m.get("summary", {})
        print(f"  {k}: status={m.get('status')} rows={s.get('rows')} patients={s.get('patients')} "
              f"positives={s.get('positives')} cols={s.get('n_columns')} quality={m.get('quality_status')} commit={m.get('git_commit')}")
    print("\nSplit manifest (sha256 + số lượng, KHÔNG in subject_id):")
    for k in sorted(k for k in keys if k.startswith("splits/") and k.endswith(".json")):
        m = json.loads(s3.get_object(Bucket=BUCKET, Key=k)["Body"].read())
        print(f"  {k}: sha256={m.get('sha256')} seed={m.get('seed')} summary={m.get('summary')}")
    print("\nGói mô hình:")
    for v in sorted({"/".join(k.split("/")[:3]) for k in keys if k.startswith("model-artifacts/")}):
        print("  ", v)
    return 0


if __name__ == "__main__":
    sys.exit(main())

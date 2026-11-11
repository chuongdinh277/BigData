"""Chia train/validation/test THEO BỆNH NHÂN, phân tầng theo việc bệnh nhân có ca dương hay không.

Manifest chứa subject_id thật nên lưu ngoài Git (data/splits/ đã nằm trong .gitignore);
chỉ commit mã băm (sha256) và số lượng để cả nhóm kiểm tra cùng một cách chia.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import GROUP_KEY, LABEL, RATIOS, SEED

PARTS = ("train", "val", "test")


def make_split(df: pd.DataFrame, ratios=RATIOS, seed: int = SEED, data_version: str = "unknown") -> dict:
    if abs(sum(ratios) - 1) > 1e-9:
        raise ValueError("Tổng tỉ lệ phải bằng 1")
    patients = df.groupby(GROUP_KEY)[LABEL].max().astype(int)
    ids, strat = patients.index.to_numpy(), patients.to_numpy()
    strat_arg = strat if strat.min() != strat.max() else None
    train_ids, rest_ids, _, rest_y = train_test_split(
        ids, strat, train_size=ratios[0], random_state=seed, stratify=strat_arg)
    val_share = ratios[1] / (ratios[1] + ratios[2])
    rest_strat = rest_y if strat_arg is not None and rest_y.min() != rest_y.max() else None
    val_ids, test_ids = train_test_split(rest_ids, train_size=val_share, random_state=seed, stratify=rest_strat)
    subjects = {k: sorted(int(x) for x in v) for k, v in zip(PARTS, (train_ids, val_ids, test_ids))}
    manifest = {
        "data_version": data_version, "seed": seed, "ratios": list(ratios), "group_key": GROUP_KEY,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "subjects": subjects,
    }
    manifest["summary"] = summarize(df, manifest)
    manifest["sha256"] = manifest_hash(manifest)
    return manifest


def manifest_hash(manifest: dict) -> str:
    payload = json.dumps(manifest["subjects"], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def apply_split(df: pd.DataFrame, manifest: dict, parts=("train", "val")) -> dict:
    """Trả về các phần được yêu cầu. Mặc định KHÔNG trả tập test (chỉ GĐ5 mới dùng)."""
    check_no_overlap(manifest)
    out = {}
    for part in parts:
        if part not in PARTS:
            raise ValueError(f"Phần không hợp lệ: {part}")
        out[part] = df[df[GROUP_KEY].isin(set(manifest["subjects"][part]))].reset_index(drop=True)
    return out


def check_no_overlap(manifest: dict) -> None:
    s = {k: set(v) for k, v in manifest["subjects"].items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        both = s[a] & s[b]
        if both:
            raise AssertionError(f"{len(both)} bệnh nhân nằm ở cả {a} và {b}")


def summarize(df: pd.DataFrame, manifest: dict) -> dict:
    out = {}
    for part in PARTS:
        sub = df[df[GROUP_KEY].isin(set(manifest["subjects"][part]))]
        out[part] = {"patients": int(sub[GROUP_KEY].nunique()), "rows": int(len(sub)),
                     "positives": int(sub[LABEL].sum()),
                     "positive_rate": round(float(sub[LABEL].mean()), 4) if len(sub) else None}
    return out


def _s3_parts(path: str):
    p = str(path)
    for pre in ("s3a://", "s3://"):
        if p.startswith(pre):
            bucket, _, key = p[len(pre):].partition("/")
            return bucket, key
    return None


def save_manifest(manifest: dict, path: str) -> None:
    """Ghi split_manifest ra file cục bộ hoặc s3://. File chứa subject_id: KHÔNG commit lên Git."""
    text = json.dumps(manifest, ensure_ascii=False)
    s3 = _s3_parts(path)
    if s3:
        import boto3
        boto3.client("s3").put_object(Bucket=s3[0], Key=s3[1], Body=text.encode("utf-8"),
                                      ContentType="application/json")
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def load_manifest(path: str) -> dict:
    s3 = _s3_parts(path)
    if s3:
        import boto3
        text = boto3.client("s3").get_object(Bucket=s3[0], Key=s3[1])["Body"].read().decode("utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    m = json.loads(text)
    if manifest_hash(m) != m.get("sha256"):
        raise ValueError("split_manifest bị sửa: sha256 không khớp")
    return m


def main(argv=None):
    import argparse
    from .data import data_version as dv, load_gold
    ap = argparse.ArgumentParser(description="Tạo split_manifest theo bệnh nhân")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", default="data/splits/split_manifest.json",
                    help="Đường dẫn cục bộ (đã gitignore) hoặc s3://.../splits/p1/v1/split_manifest.json")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    df = load_gold(a.gold)
    m = make_split(df, seed=a.seed, data_version=dv(a.gold))
    save_manifest(m, a.out)
    print(json.dumps({"out": a.out, "sha256": m["sha256"], "summary": m["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

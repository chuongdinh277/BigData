"""Phát hiện drift THEO LÔ (không thời gian thực — đúng phạm vi đề cương).

- Drift dữ liệu: PSI từng đặc trưng (giá trị thiếu là một bin riêng) + thay đổi tỷ lệ thiếu.
- Drift nhãn: tỷ lệ tử vong theo giai đoạn.
- Drift điểm dự đoán: PSI của phân phối xác suất.
- Drift mô hình: huấn luyện trên giai đoạn cũ, đánh giá trên giai đoạn mới (hiệu năng + hiệu chỉnh),
  và so thứ hạng tầm quan trọng SHAP giữa mô hình của các giai đoạn.

`DriftReference` lưu phân phối tham chiếu (chỉ biên bin và tỷ lệ — số liệu tổng hợp) vào gói mô hình,
để ứng dụng/job theo lô so sánh một lô dữ liệu mới mà không cần dữ liệu train.
Ngưỡng PSI quen dùng: < 0.10 ổn định, 0.10–0.25 thay đổi vừa, > 0.25 thay đổi lớn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PSI_MODERATE, PSI_MAJOR = 0.10, 0.25
_E = 1e-4


def psi(ref_props, cur_props) -> float:
    a = np.clip(np.asarray(ref_props, float), _E, None)
    b = np.clip(np.asarray(cur_props, float), _E, None)
    a, b = a / a.sum(), b / b.sum()
    return float(np.sum((b - a) * np.log(b / a)))


def level(v: float) -> str:
    if not np.isfinite(v):
        return "N/A"
    return "LỚN" if v > PSI_MAJOR else ("VỪA" if v > PSI_MODERATE else "ỔN ĐỊNH")


class DriftReference:
    def __init__(self, spec: dict | None = None):
        self.spec = spec or {"features": {}, "score": None}

    @classmethod
    def fit(cls, X: pd.DataFrame, scores=None, n_bins: int = 10) -> "DriftReference":
        feats = {}
        for c in X.columns:
            s = X[c]
            if pd.api.types.is_numeric_dtype(s):
                v = s.dropna().to_numpy(float)
                if len(v) == 0:
                    continue
                edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1))[1:-1])
                feats[c] = {"type": "num", "edges": edges.tolist(), "props": _num_props(s, edges).tolist(),
                            "missing_rate": float(s.isna().mean())}
            else:
                vc = s.fillna("__MISSING__").astype(str).value_counts(normalize=True)
                feats[c] = {"type": "cat", "props": vc.to_dict(), "missing_rate": float(s.isna().mean())}
        score = None
        if scores is not None:
            sc = pd.Series(np.asarray(scores, float))
            edges = np.unique(np.quantile(sc, np.linspace(0, 1, n_bins + 1))[1:-1])
            score = {"edges": edges.tolist(), "props": _num_props(sc, edges).tolist()}
        return cls({"features": feats, "score": score})

    def compare(self, X: pd.DataFrame, scores=None) -> dict:
        rows = []
        for c, spec in self.spec["features"].items():
            if c not in X.columns:
                continue
            s = X[c]
            if spec["type"] == "num":
                v = psi(spec["props"], _num_props(s, np.asarray(spec["edges"])))
            else:
                cur = s.fillna("__MISSING__").astype(str).value_counts(normalize=True)
                cats = sorted(set(spec["props"]) | set(cur.index))
                v = psi([spec["props"].get(k, 0) for k in cats], [cur.get(k, 0) for k in cats])
            rows.append({"feature": c, "psi": v, "level": level(v), "missing_rate_ref": spec["missing_rate"],
                         "missing_rate_cur": float(s.isna().mean()),
                         "missing_rate_delta": float(s.isna().mean()) - spec["missing_rate"]})
        table = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
        out = {"n": int(len(X)), "features": table,
               "share_major": float((table["psi"] > PSI_MAJOR).mean()) if len(table) else 0.0,
               "share_moderate_or_more": float((table["psi"] > PSI_MODERATE).mean()) if len(table) else 0.0}
        if scores is not None and self.spec.get("score"):
            sp = self.spec["score"]
            out["score_psi"] = psi(sp["props"], _num_props(pd.Series(np.asarray(scores, float)), np.asarray(sp["edges"])))
            out["score_level"] = level(out["score_psi"])
        out["status"] = ("CẢNH BÁO" if out["share_major"] > 0.10 or out.get("score_psi", 0) > PSI_MAJOR
                         else ("THEO DÕI" if out["share_moderate_or_more"] > 0.10 or out.get("score_psi", 0) > PSI_MODERATE
                               else "ỔN ĐỊNH"))
        return out

    def to_dict(self) -> dict:
        return self.spec


def _num_props(s: pd.Series, edges: np.ndarray) -> np.ndarray:
    s = pd.Series(s)
    v = s.dropna().to_numpy(float)
    idx = np.searchsorted(edges, v, side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    counts = np.append(counts, s.isna().sum())  # bin cuối = thiếu
    return counts / max(len(s), 1)


def period_order(values) -> list[str]:
    return sorted(pd.Series(values).dropna().astype(str).unique())

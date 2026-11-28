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


# ----------------------------------------------------------------------------------------------
# Drift nâng cao: ngưỡng có căn cứ, drift đa biến, drift mã hóa, chiến lược bảo trì mô hình
# ----------------------------------------------------------------------------------------------
def psi_null_thresholds(X_ref: pd.DataFrame, sample_size: int, n_boot: int = 50, q: float = 0.99,
                        seed: int = 42) -> dict:
    """Ngưỡng PSI theo phân phối NULL (không có drift): bootstrap hai mẫu rời nhau cỡ `sample_size`
    lấy từ CHÍNH lô tham chiếu, tính PSI từng đặc trưng; ngưỡng = phân vị q.

    Lý do: ngưỡng quy ước 0.10/0.25 không tính đến cỡ lô, lô nhỏ cho PSI cao chỉ vì nhiễu lấy mẫu.
    """
    rng = np.random.default_rng(seed)
    m = int(min(sample_size, len(X_ref) // 2))
    vals: dict[str, list[float]] = {}
    for _ in range(n_boot):
        idx = rng.permutation(len(X_ref))
        a, b = X_ref.iloc[idx[:m]], X_ref.iloc[idx[m:2 * m]]
        t = DriftReference.fit(a).compare(b)["features"]
        for f, v in zip(t["feature"], t["psi"]):
            vals.setdefault(f, []).append(v)
    per_feature = {f: float(np.quantile(v, q)) for f, v in vals.items()}
    return {"sample_size": m, "n_boot": n_boot, "quantile": q, "per_feature": per_feature,
            "median_threshold": float(np.median(list(per_feature.values()))) if per_feature else float("nan")}


def exceed_null(table: pd.DataFrame, null: dict) -> dict:
    """Số đặc trưng có PSI vượt ngưỡng null của chính nó (báo song song với ngưỡng quy ước)."""
    thr = table["feature"].map(null["per_feature"]).fillna(PSI_MODERATE)
    over = table["psi"] > thr
    return {"n_features_over_null": int(over.sum()), "share_over_null": float(over.mean()) if len(table) else 0.0,
            "features_over_null": table.loc[over, "feature"].head(15).tolist()}


def _as_model_frame(X: pd.DataFrame) -> pd.DataFrame:
    Z = X.copy()
    for c in Z.columns:
        if not pd.api.types.is_numeric_dtype(Z[c]):
            Z[c] = Z[c].astype("category")
    return Z


def domain_classifier(X_ref: pd.DataFrame, X_cur: pd.DataFrame, n_max: int = 20000, folds: int = 3,
                      seed: int = 42, top_k: int = 10) -> dict:
    """Drift ĐA BIẾN: huấn luyện mô hình phân biệt lô tham chiếu với lô hiện tại.

    AUC gần 0.5: không phân biệt được (không drift). AUC càng cao, hai lô càng khác nhau.
    Tầm quan trọng (gain) của mô hình cho biết đặc trưng nào khác nhiều nhất.
    """
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(seed)
    a = X_ref.iloc[rng.permutation(len(X_ref))[:n_max]]
    b = X_cur.iloc[rng.permutation(len(X_cur))[:n_max]]
    Z = _as_model_frame(pd.concat([a, b], ignore_index=True))
    d = np.r_[np.zeros(len(a), int), np.ones(len(b), int)]
    oof = np.zeros(len(d))
    gains = np.zeros(Z.shape[1])
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(Z, d):
        m = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=seed, verbose=-1, importance_type="gain")
        m.fit(Z.iloc[tr], d[tr])
        oof[te] = m.predict_proba(Z.iloc[te])[:, 1]
        gains += m.feature_importances_
    imp = pd.DataFrame({"feature": Z.columns, "gain_share": gains / max(gains.sum(), 1e-12)})
    imp = imp.sort_values("gain_share", ascending=False).head(top_k)
    return {"auc": float(roc_auc_score(d, oof)), "n_ref": len(a), "n_cur": len(b),
            "top_features": imp.to_dict(orient="records")}


def coding_drift(df: pd.DataFrame, period_col: str = "anchor_year_group", version_col: str = "icd_version_group",
                 prefix: str = "hx_") -> dict:
    """Drift do thay đổi HỆ MÃ chẩn đoán (ICD-9 sang ICD-10 từ 10/2015 ở Mỹ, rơi vào lô 2014–2016).

    Bệnh nền trong Gold được mã hóa từ ICD, nên đổi hệ mã có thể làm tỷ lệ cờ bệnh nền nhảy bậc
    mà không phải do bệnh nhân thay đổi.
    """
    flags = [c for c in df.columns if c.startswith(prefix)]
    if not flags or period_col not in df or version_col not in df:
        return {"available": False}
    by_ver = df.groupby(version_col)[flags].mean().round(4)
    by_per = df.groupby(period_col)[flags].mean().round(4)
    mix = pd.crosstab(df[period_col], df[version_col], normalize="index").round(4)
    jump = (by_ver.max() - by_ver.min()).sort_values(ascending=False)
    return {"available": True, "flag_rate_by_icd_version": by_ver.to_dict(orient="index"),
            "flag_rate_by_period": by_per.to_dict(orient="index"),
            "icd_version_mix_by_period": mix.to_dict(orient="index"),
            "largest_version_gaps": jump.head(8).to_dict(),
            "n_by_icd_version": df[version_col].value_counts().to_dict()}


def maintenance_strategies(pipe, X: pd.DataFrame, y, period, calibrator_factory, n_train_periods: int = 2) -> dict:
    """So sánh 3 chiến lược bảo trì mô hình khi dữ liệu trôi theo thời gian, KHÔNG nhìn tương lai.

    Huấn luyện ban đầu trên `n_train_periods` lô đầu. Với mỗi lô đánh giá k (theo thứ tự thời gian):
      (a) static      giữ nguyên mô hình và bộ hiệu chỉnh ban đầu;
      (b) recalibrate giữ mô hình, chỉnh lại intercept bằng nhãn của các lô ĐÃ QUA sau giai đoạn train;
      (c) retrain     huấn luyện lại trên mọi lô trước k (kèm hiệu chỉnh lại).
    Khi đánh giá lô k chỉ dùng nhãn của các lô trước k, đúng điều kiện vận hành thật.
    """
    from sklearn.base import clone

    from .evaluation import calibration_stats, discrimination, fit_logistic_1d, logit

    y = np.asarray(y).astype(int)
    period = np.asarray(period).astype(str)
    periods = period_order(period)
    if len(periods) <= n_train_periods:
        return {"available": False}
    tr = np.isin(period, periods[:n_train_periods])
    base_model = clone(pipe).fit(X[tr], y[tr])
    base_cal = calibrator_factory().fit(base_model.predict_proba(X[tr])[:, 1], y[tr])
    rows = []
    for k in range(n_train_periods, len(periods)):
        cur = period == periods[k]
        seen = np.isin(period, periods[:k])
        seen_post = np.isin(period, periods[n_train_periods:k])
        if cur.sum() == 0 or y[cur].sum() < 5:
            continue
        p_static = base_cal.transform(base_model.predict_proba(X[cur])[:, 1])
        # (b): intercept học trên các lô sau train đã có nhãn; lô đầu tiên sau train chưa có gì để học
        if seen_post.any() and y[seen_post].sum() >= 10:
            p_seen = base_cal.transform(base_model.predict_proba(X[seen_post])[:, 1])
            a, _ = fit_logistic_1d(logit(p_seen), y[seen_post], offset=logit(p_seen), fit_slope=False)
        else:
            a = 0.0
        p_recal = 1 / (1 + np.exp(-(logit(p_static) + a)))
        m_re = clone(pipe).fit(X[seen], y[seen])
        cal_re = calibrator_factory().fit(m_re.predict_proba(X[seen])[:, 1], y[seen])
        p_retrain = cal_re.transform(m_re.predict_proba(X[cur])[:, 1])
        for name, p in (("static", p_static), ("recalibrate", p_recal), ("retrain", p_retrain)):
            d = discrimination(y[cur], p)
            c = calibration_stats(y[cur], p)
            rows.append({"period": periods[k], "strategy": name, "n": int(cur.sum()), "n_pos": int(y[cur].sum()),
                         "positive_rate": d["positive_rate"], "pr_auc": d["pr_auc"], "roc_auc": d["roc_auc"],
                         "brier": d["brier"], "oe_ratio": c["oe_ratio"], "ece": c["ece"],
                         "recal_intercept": a if name == "recalibrate" else None})
    return {"available": True, "train_periods": periods[:n_train_periods], "rows": rows}

"""Giải thích SHAP ở cấp bệnh nhân và toàn cục.

- XGBoost / LightGBM: TreeSHAP chính xác do chính thư viện tính (pred_contribs / pred_contrib),
  không cần gói `shap` và không xấp xỉ.
- Logistic Regression: Linear SHAP = coef · (x − E[x]) trên không gian đã tiền xử lý.

Giá trị SHAP ở thang log-odds của mô hình CHƯA hiệu chỉnh: base_value + tổng SHAP = logit(p thô).
Cột one-hot được cộng gộp về cột gốc (vd admission_type_group).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X, dtype=float)


def _family(name: str) -> str:
    """Gộp đặc trưng về 'họ': creatinine_first/min/max/missing → creatinine; hr_mean → sinh_hieu."""
    for suf in ("_first", "_min", "_max", "_missing", "_last"):
        if name.endswith(suf):
            return name[: -len(suf)]
    if name.startswith(("hr_", "sbp_", "dbp_", "mbp_", "rr_", "spo2_", "temp_")) or name == "vitals_missing":
        return "sinh_hieu"
    if name.startswith("hx_"):
        return "tien_su_benh"
    if name.startswith("omr_"):
        return "omr"
    return name


class Explainer:
    def __init__(self, pipeline, background: pd.DataFrame | None = None):
        self.pipeline = pipeline
        self.prep = pipeline.named_steps["prep"]
        self.model = pipeline.steps[-1][1]
        self.kind = type(self.model).__name__
        names = list(self.prep.get_feature_names_out())
        self.input_columns = list(self.prep.feature_names_in_)
        cat_cols = []
        for name, _, cols in self.prep.transformers_:
            if name == "cat":
                cat_cols = list(cols)
        self.columns_t = names
        self.map_t = []  # cột sau tiền xử lý → cột gốc
        for n in names:
            raw = n.split("__", 1)[1] if "__" in n else n
            if n.startswith("cat__"):
                raw = next((c for c in sorted(cat_cols, key=len, reverse=True) if raw.startswith(c + "_")), raw)
            self.map_t.append(raw)
        self.originals = list(dict.fromkeys(self.map_t))
        self._agg = np.zeros((len(names), len(self.originals)))
        idx = {c: i for i, c in enumerate(self.originals)}
        for j, raw in enumerate(self.map_t):
            self._agg[j, idx[raw]] = 1.0
        self.bg_mean_t = None
        if self.kind == "LogisticRegression":
            if background is None:
                raise ValueError("Logistic Regression cần background (mẫu train) để tính E[x]")
            self.bg_mean_t = _dense(self.prep.transform(background[self.input_columns])).mean(axis=0)

    def contributions(self, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Trả (SHAP theo cột gốc [n × p], base_value [n]) ở thang log-odds."""
        Xt = _dense(self.prep.transform(X[self.input_columns]))
        if self.kind == "XGBClassifier":
            import xgboost as xgb
            raw = self.model.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)
        elif self.kind == "LGBMClassifier":
            raw = np.asarray(self.model.booster_.predict(Xt, pred_contrib=True))
        elif self.kind == "LogisticRegression":
            coef = self.model.coef_.ravel()
            contrib = (Xt - self.bg_mean_t) * coef
            base = float(self.model.intercept_[0] + self.bg_mean_t @ coef)
            raw = np.column_stack([contrib, np.full(len(Xt), base)])
        else:
            raise TypeError(f"Chưa hỗ trợ giải thích cho {self.kind}")
        shap_t, base = raw[:, :-1], raw[:, -1]
        return pd.DataFrame(shap_t @ self._agg, columns=self.originals, index=X.index), base

    def explain_row(self, row: pd.DataFrame, top_k: int = 10) -> dict:
        shap_df, base = self.contributions(row)
        s = shap_df.iloc[0]
        order = s.abs().sort_values(ascending=False).index[:top_k]
        feats = []
        for f in order:
            v = row.iloc[0][f] if f in row.columns else None
            v = None if v is None or (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v)
            feats.append({"feature": f, "value": v, "shap": round(float(s[f]), 5),
                          "direction": "tăng nguy cơ" if s[f] > 0 else "giảm nguy cơ"})
        return {"method": "TreeSHAP" if self.kind != "LogisticRegression" else "LinearSHAP",
                "scale": "log-odds (trước hiệu chỉnh)", "base_value": round(float(base[0]), 5),
                "top_features": feats}


def global_importance(shap_df: pd.DataFrame) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": shap_df.columns,
                        "mean_abs_shap": shap_df.abs().mean().to_numpy(),
                        "mean_shap": shap_df.mean().to_numpy()})
    imp["family"] = imp["feature"].map(_family)
    return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def family_importance(imp: pd.DataFrame) -> pd.DataFrame:
    return (imp.groupby("family", as_index=False)["mean_abs_shap"].sum()
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def direction_summary(shap_df: pd.DataFrame, X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Hướng tác động: tương quan Spearman giữa giá trị đặc trưng và SHAP (chỉ số tổng hợp)."""
    rows = []
    for f in features:
        if f not in X.columns or not pd.api.types.is_numeric_dtype(X[f]):
            continue
        m = X[f].notna()
        if m.sum() < 30 or X.loc[m, f].nunique() < 2 or shap_df.loc[m, f].nunique() < 2:
            continue
        rho = pd.Series(X.loc[m, f].to_numpy()).corr(pd.Series(shap_df.loc[m, f].to_numpy()), method="spearman")
        rows.append({"feature": f, "spearman_value_vs_shap": float(rho),
                     "doc": "giá trị cao → nguy cơ cao" if rho > 0.2 else
                            ("giá trị cao → nguy cơ thấp" if rho < -0.2 else "không đơn điệu")})
    return pd.DataFrame(rows)


def importance_stability(importances: list[pd.DataFrame], top_k: int = 20) -> dict:
    """So sánh thứ hạng tầm quan trọng giữa các mô hình (fold CV hoặc giai đoạn thời gian)."""
    ref = importances[0].set_index("feature")["mean_abs_shap"]
    rhos, jacc = [], []
    for imp in importances[1:]:
        cur = imp.set_index("feature")["mean_abs_shap"].reindex(ref.index).fillna(0)
        rhos.append(float(ref.rank().corr(cur.rank(), method="spearman")))
        a = set(ref.sort_values(ascending=False).index[:top_k])
        b = set(cur.sort_values(ascending=False).index[:top_k])
        jacc.append(len(a & b) / len(a | b))
    return {"spearman_mean": float(np.mean(rhos)) if rhos else float("nan"),
            "spearman_min": float(np.min(rhos)) if rhos else float("nan"),
            f"jaccard_top{top_k}_mean": float(np.mean(jacc)) if jacc else float("nan")}


# ----------------------------------------------------------------------------------------------
# Chất lượng của lời giải thích: trung thực (faithfulness) và hợp lý lâm sàng (plausibility)
# ----------------------------------------------------------------------------------------------
def deletion_curve(pipeline, X: pd.DataFrame, shap_df: pd.DataFrame, ks=(1, 2, 3, 5, 10, 20),
                   n_random: int = 5, seed: int = 42) -> pd.DataFrame:
    """Che (đặt NULL) k đặc trưng có |SHAP| lớn nhất của TỪNG ca, đo thay đổi logit trung bình;
    so với che k đặc trưng ngẫu nhiên. SHAP trung thực ⇔ đường SHAP cao hơn rõ đường ngẫu nhiên.
    Cờ `_missing` không bị che trực tiếp (là hệ quả của việc thiếu, không phải giá trị đo)."""
    from .evaluation import logit

    rng = np.random.default_rng(seed)
    cand = [c for c in X.columns if not c.endswith("_missing") and pd.api.types.is_numeric_dtype(X[c])]
    S = shap_df[cand].abs().to_numpy()
    base = logit(pipeline.predict_proba(X)[:, 1])
    rows = []
    for k in ks:
        k = min(k, len(cand))
        top_idx = np.argsort(-S, axis=1)[:, :k]
        Xs = X.copy()
        arr = Xs[cand].to_numpy(dtype=float, copy=True)
        np.put_along_axis(arr, top_idx, np.nan, axis=1)
        Xs[cand] = arr
        d_shap = float(np.mean(np.abs(logit(pipeline.predict_proba(Xs)[:, 1]) - base)))
        d_rand = []
        for _ in range(n_random):
            Xr = X.copy()
            arr = Xr[cand].to_numpy(dtype=float, copy=True)
            ridx = np.argsort(rng.random(arr.shape), axis=1)[:, :k]
            np.put_along_axis(arr, ridx, np.nan, axis=1)
            Xr[cand] = arr
            d_rand.append(float(np.mean(np.abs(logit(pipeline.predict_proba(Xr)[:, 1]) - base))))
        rows.append({"k": k, "mean_abs_logit_change_shap_order": d_shap,
                     "mean_abs_logit_change_random": float(np.mean(d_rand)),
                     "ratio_shap_vs_random": d_shap / max(float(np.mean(d_rand)), 1e-9)})
    return pd.DataFrame(rows)


# Chiều tác động KỲ VỌNG theo y văn suy tim (+1: giá trị cao → nguy cơ cao; −1: giá trị cao → nguy cơ thấp).
# Dùng cho cả 3 bài toán; với P2 (tái nhập) một số chiều có thể yếu hơn — khi đó báo cáo "không đơn điệu".
EXPECTED_DIRECTION = {
    "age": (+1, "Tuổi cao"),
    "bun": (+1, "Hội chứng tim–thận"),
    "creatinine": (+1, "Suy giảm chức năng thận"),
    "sodium": (-1, "Hạ natri máu là yếu tố tiên lượng xấu"),
    "lactate": (+1, "Giảm tưới máu mô"),
    "sbp": (-1, "Huyết áp tâm thu thấp, nguy cơ sốc tim"),
    "albumin": (-1, "Suy dinh dưỡng, viêm"),
    "ntprobnp": (+1, "Căng giãn cơ tim"),
    "troponin_t": (+1, "Tổn thương cơ tim"),
    "rdw": (+1, "Yếu tố tiên lượng đã ghi nhận trong suy tim"),
    "hr": (+1, "Nhịp nhanh"),
    "anion_gap": (+1, "Toan chuyển hóa"),
    "bicarbonate": (-1, "Toan chuyển hóa"),
    "hemoglobin": (-1, "Thiếu máu"),
    "wbc": (+1, "Viêm, nhiễm trùng"),
}


def _concept(feature: str) -> str | None:
    for k in sorted(EXPECTED_DIRECTION, key=len, reverse=True):
        if feature == k or feature.startswith(k + "_"):
            return k
    return None


def clinical_plausibility(direction: pd.DataFrame) -> pd.DataFrame:
    """Đối chiếu chiều tác động học được (Spearman giá trị ~ SHAP) với chiều kỳ vọng lâm sàng."""
    rows = []
    for _, r in direction.iterrows():
        if r["feature"].endswith("_missing"):
            continue  # cờ thiếu không có "chiều kỳ vọng" theo y văn
        k = _concept(r["feature"])
        if k is None:
            continue
        exp, why = EXPECTED_DIRECTION[k]
        rho = r["spearman_value_vs_shap"]
        verdict = "không đơn điệu" if abs(rho) <= 0.2 else ("đúng chiều" if np.sign(rho) == exp else "NGƯỢC CHIỀU")
        rows.append({"feature": r["feature"], "concept": k, "expected": "↑ nguy cơ" if exp > 0 else "↓ nguy cơ",
                     "spearman_value_vs_shap": float(rho), "verdict": verdict, "rationale": why})
    return pd.DataFrame(rows)

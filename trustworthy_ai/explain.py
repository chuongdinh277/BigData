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
        self.bg_mean_t = None
        if hasattr(self.model, "additive_contributions"):   # OSMA-Net: mô hình cộng tính, giải thích sẵn
            self.kind = "OSMANet"
            self.input_columns = list(self.model.feature_names_in_)
            return
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
        if self.kind == "OSMANet":
            return self.model.additive_contributions(X[self.input_columns])
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
        method = {"LogisticRegression": "LinearSHAP", "OSMANet": "ExactAdditive(OSMA-Net)"}.get(self.kind, "TreeSHAP")
        out = {"method": method,
                "scale": "log-odds (trước hiệu chỉnh)", "base_value": round(float(base[0]), 5),
                "top_features": feats}
        if self.kind == "OSMANet":   # thêm điểm nguy cơ theo hệ cơ quan
            sc = self.model.system_scores(row[self.input_columns]).iloc[0]
            out["organ_systems"] = [{"system": k, "score": round(float(v), 5)}
                                    for k, v in sc.sort_values(key=abs, ascending=False).items()]
        return out


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
        if m.sum() < 30 or X.loc[m, f].nunique() < 2:
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

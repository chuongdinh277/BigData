"""OSMA-Net — Organ-System, Missingness-Aware neural additive network (mô hình lõi của nhóm).

    logit(p) = b + Σ_j h_j(x_j) + Σ_k w_k · onehot_k
    h_j(x)   = f_j(x)  nếu x_j được đo        (f_j: MLP nhỏ riêng cho từng đặc trưng)
             = e_j      nếu x_j bị thiếu        (e_j: "nguy cơ khi thiếu" học được, không điền median)

Vì sao thiết kế như vậy:
- **Cộng tính** → giải thích chính xác: đóng góp của đặc trưng j cho một ca = h_j(x_j) − E_train[h_j];
  đây đúng là giá trị Shapley của mô hình cộng tính (không xấp xỉ như KernelSHAP).
- **Nhận biết thiếu dữ liệu**: khi thiếu, mô hình dùng e_j thay vì giả vờ thấy giá trị trung vị, nên
  “không làm NT-proBNP” có ý nghĩa riêng. Các cờ `_missing` của Gold vì thế không cần làm đầu vào.
- **Theo hệ cơ quan**: các đặc trưng được gộp vào hệ (tim, thận, điện giải, huyết học, gan, tưới máu/hô hấp,
  sinh hiệu, tiền sử, nhân khẩu). Phạt L2 theo nhóm trên đầu ra mỗi hệ để mô hình gọn, và trả thêm
  “điểm nguy cơ theo hệ cơ quan” cho bác sĩ đọc.
- Feature dropout (bỏ ngẫu nhiên cả một h_j khi huấn luyện) giúp bền khi thiếu dữ liệu, như trong NAM.

Giới hạn: không có tương tác giữa các đặc trưng (đánh đổi lấy khả năng giải thích); cần PyTorch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

SYSTEMS = {
    "tim_mach": ("ntprobnp", "troponin_t", "hr_", "sbp_", "dbp_", "mbp_", "hx_cad", "hx_afib", "hx_prior_mi",
                 "hx_hypertension", "omr_sbp", "n_prior_hf_adm"),
    "than": ("creatinine", "bun", "omr_egfr", "hx_ckd"),
    "dien_giai_chuyen_hoa": ("sodium", "potassium", "chloride", "bicarbonate", "anion_gap", "glucose", "magnesium",
                             "hx_diabetes"),
    "huyet_hoc": ("hemoglobin", "hematocrit", "wbc", "platelet", "rdw", "inr"),
    "gan_dinh_duong": ("bilirubin", "albumin", "omr_bmi", "omr_weight"),
    "tuoi_mau_ho_hap": ("lactate", "spo2_", "rr_", "temp_", "hx_copd", "hx_stroke"),
}


def organ_system(name: str) -> str:
    for sys_name, keys in SYSTEMS.items():
        if any(name.startswith(k) for k in keys):
            return sys_name
    if name in ("age", "gender_male"):
        return "nhan_khau"
    return "nhap_vien_tien_su"


class OSMAPreprocessor(BaseEstimator, TransformerMixin):
    """Chuẩn hóa bền (median/IQR, cắt ±5) học từ train; giữ mặt nạ thiếu; one-hot cho cột chữ.
    Đầu ra: [giá trị (0 khi thiếu) | mặt nạ thiếu | one-hot]."""

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.array(X.columns)
        cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        flags = [c for c in X.columns if c.endswith("_missing")]  # suy ra được từ mặt nạ → bỏ
        self.num_ = [c for c in X.columns if c not in cat and c not in flags]
        self.cat_ = cat
        v = X[self.num_].astype(float)
        self.med_ = v.median().fillna(0).to_numpy()
        iqr = (v.quantile(0.75) - v.quantile(0.25)).to_numpy()
        sd = v.std().fillna(1).to_numpy()
        self.scale_ = np.where(np.nan_to_num(iqr) > 0, iqr / 1.349, np.where(sd > 0, sd, 1.0))
        self.levels_ = {c: sorted(X[c].dropna().astype(str).unique().tolist()) for c in cat}
        self.term_names_ = self.num_ + [c for c in cat for _ in self.levels_[c]]
        self.term_original_ = self.term_names_
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X)
        v = X[self.num_].astype(float).to_numpy()
        m = np.isnan(v)
        z = np.clip((v - self.med_) / self.scale_, -5, 5)
        z[m] = 0.0
        oh = [(X[c].astype(str).to_numpy()[:, None] == np.array(self.levels_[c])[None, :]).astype(float)
              for c in self.cat_]
        return np.hstack([z, m.astype(float)] + oh).astype(np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"num__{c}" for c in self.num_] + [f"mask__{c}" for c in self.num_] +
                        [f"cat__{c}_{lv}" for c in self.cat_ for lv in self.levels_[c]])


def _build_module(p_num: int, p_cat: int, hidden: int, n_sys: int):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Parameter(torch.randn(p_num, 1, hidden) * 0.5)
            self.b1 = nn.Parameter(torch.zeros(p_num, hidden))
            self.w2 = nn.Parameter(torch.randn(p_num, hidden, hidden) / hidden ** 0.5)
            self.b2 = nn.Parameter(torch.zeros(p_num, hidden))
            self.w3 = nn.Parameter(torch.randn(p_num, hidden, 1) / hidden ** 0.5 * 0.1)
            self.missing = nn.Parameter(torch.zeros(p_num))     # e_j: nguy cơ khi thiếu
            self.cat = nn.Linear(p_cat, 1, bias=False) if p_cat else None
            self.bias = nn.Parameter(torch.zeros(1))

        def terms(self, z, m):
            h = torch.relu(z.unsqueeze(-1) * self.w1.squeeze(1) + self.b1)          # (n, p, hidden)
            h = torch.relu(torch.einsum("nph,phk->npk", h, self.w2) + self.b2)
            f = torch.einsum("nph,pho->npo", h, self.w3).squeeze(-1)
            return f * (1 - m) + self.missing * m

        def forward(self, z, m, c, drop_mask=None):
            t = self.terms(z, m)
            if drop_mask is not None:
                t = t * drop_mask
            out = self.bias + t.sum(1)
            if self.cat is not None:
                out = out + self.cat(c).squeeze(-1)
            return out, t

    return Net()


class OSMANet(BaseEstimator, ClassifierMixin):
    """Nhận DataFrame thô (như mọi mô hình khác trong pipeline); tiền xử lý học bên trong fit()
    để clone()/CV của sklearn hoạt động đúng."""

    def __init__(self, hidden: int = 16, epochs: int = 40, lr: float = 2e-3, batch_size: int = 512,
                 weight_decay: float = 1e-5, feature_dropout: float = 0.1, output_penalty: float = 1e-3,
                 system_penalty: float = 1e-3, patience: int = 5, random_state: int = 42, n_jobs: int = 4):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.feature_dropout = feature_dropout
        self.output_penalty = output_penalty
        self.system_penalty = system_penalty
        self.patience = patience
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _split(self, X):
        A = self.pre_.transform(X)
        p = self.p_num_
        return A[:, :p], A[:, p:2 * p], A[:, 2 * p:]

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.random_state)
        torch.set_num_threads(max(1, int(self.n_jobs)))
        rng = np.random.default_rng(self.random_state)
        X = pd.DataFrame(X)
        self.pre_ = OSMAPreprocessor().fit(X)
        self.feature_names_in_ = np.array(X.columns)
        self.p_num_ = len(self.pre_.num_)
        names = self.pre_.num_
        # tên cột gốc cho từng "term": cột số → chính nó; mỗi mức one-hot → cột chữ gốc
        self.term_original_ = names + [c for c in self.pre_.cat_ for _ in self.pre_.levels_[c]]
        z, m, c = self._split(X)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        self.systems_ = {}
        for j, n in enumerate(names):
            self.systems_.setdefault(organ_system(n), []).append(j)
        net = _build_module(self.p_num_, c.shape[1], self.hidden, len(self.systems_))
        with torch.no_grad():
            prev = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
            net.bias.fill_(np.log(prev / (1 - prev)))
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # 10% dòng của chính dữ liệu được truyền vào để dừng sớm (không dùng validation/test của nhóm)
        idx = rng.permutation(len(y))
        n_es = max(int(0.1 * len(y)), 1)
        es, tr = idx[:n_es], idx[n_es:].copy()
        T = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))  # noqa: E731
        Z, M, C, Y = T(z), T(m), T(c), T(y)
        sys_idx = [torch.tensor(v, dtype=torch.long) for v in self.systems_.values()]
        lossf = torch.nn.BCEWithLogitsLoss()
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            net.train()
            rng.shuffle(tr)
            for s0 in range(0, len(tr), self.batch_size):
                b = torch.from_numpy(tr[s0:s0 + self.batch_size])
                drop = (torch.rand(len(b), self.p_num_) > self.feature_dropout).float() / (1 - self.feature_dropout)
                out, t = net(Z[b], M[b], C[b], drop)
                loss = lossf(out, Y[b]) + self.output_penalty * (t ** 2).mean()
                if self.system_penalty:
                    loss = loss + self.system_penalty * sum(t[:, i].sum(1).pow(2).mean().sqrt() for i in sys_idx)
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(lossf(net(Z[es], M[es], C[es])[0], Y[es]))
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        self.net_ = net
        self.best_val_loss_ = best
        terms, _ = self._terms(X)
        self.term_mean_ = terms.mean(0)                              # nền E[h_j] cho giải thích
        return self

    def _terms(self, X):
        import torch
        z, m, c = self._split(pd.DataFrame(X))
        with torch.no_grad():
            t = self.net_.terms(torch.from_numpy(z), torch.from_numpy(m)).numpy()
        tc = c * self.net_.cat.weight.detach().numpy().ravel() if self.net_.cat is not None else np.zeros((len(z), 0))
        return np.hstack([t, tc]), float(self.net_.bias.detach().numpy()[0])

    def decision_function(self, X):
        t, b = self._terms(X)
        return b + t.sum(1)

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-np.clip(self.decision_function(X), -50, 50)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def additive_contributions(self, X) -> tuple[pd.DataFrame, np.ndarray]:
        """Đóng góp chính xác (= Shapley của mô hình cộng tính) so với nền trung bình train,
        gộp về cột gốc."""
        t, b = self._terms(X)
        t = t - self.term_mean_
        df = pd.DataFrame(t, columns=self.term_original_, index=pd.DataFrame(X).index)
        df = df.T.groupby(level=0, sort=False).sum().T
        return df, np.full(len(t), b + self.term_mean_.sum())

    def system_scores(self, X) -> pd.DataFrame:
        """Điểm nguy cơ theo hệ cơ quan (log-odds so với ca trung bình)."""
        t, _ = self._terms(X)
        t = t - self.term_mean_
        return pd.DataFrame({s: t[:, idx].sum(1) for s, idx in self.systems_.items()}, index=pd.DataFrame(X).index)

    def __getstate__(self):
        state = self.__dict__.copy()
        if "net_" in state:
            net = state.pop("net_")
            state["net_state_"] = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
            state["net_cat_"] = int(net.cat.in_features) if net.cat is not None else 0
        return state

    def __setstate__(self, state):
        net_state = state.pop("net_state_", None)
        p_cat = state.pop("net_cat_", 0)
        self.__dict__.update(state)
        if net_state is not None:
            import torch
            net = _build_module(self.p_num_, p_cat, self.hidden, len(self.systems_))
            net.load_state_dict({k: torch.from_numpy(v) for k, v in net_state.items()})
            net.eval()
            self.net_ = net


def osma_pipeline(**params):
    """Pipeline([prep=identity, model=OSMA-Net]) để dùng chung CV/hiệu chỉnh/gói mô hình với các mô hình khác."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    return Pipeline([("prep", FunctionTransformer()), ("model", OSMANet(**params))])

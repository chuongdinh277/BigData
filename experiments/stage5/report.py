"""Sinh báo cáo GĐ5 (Markdown + hình) từ kết quả tổng hợp. Ô có số ca < 10 được che."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SMALL = 10


def _f(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return f"{v:.{nd}f}"


def _n(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    v = int(round(v))
    return "<10" if 0 < v < SMALL else f"{v:,}"


def _rate(v, n_pos, nd=1):
    """Tỷ lệ của nhóm có < 10 ca dương cũng bị che (vì n × tỷ lệ = số ca)."""
    return "–" if n_pos is None or (isinstance(n_pos, float) and np.isnan(n_pos)) or 0 < n_pos < SMALL else _pct(v, nd)


def _pct(v, nd=1):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{100 * v:.{nd}f}%"


def _ci(ci, nd=3):
    if not ci or any(c is None or not np.isfinite(c) for c in ci):
        return ""
    return f" ({ci[0]:.{nd}f}–{ci[1]:.{nd}f})"


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------
def figures(R: dict, fig_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    made = {}

    # 1. CV PR-AUC
    cv = R["cv"]
    names = list(cv)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(names, [cv[n]["pr_auc_mean"] for n in names], yerr=[cv[n]["pr_auc_sd"] for n in names],
           color=["#2b6cb0" if n == R["chosen_model"] else "#a0aec0" for n in names], capsize=4)
    ax.axhline(R["dev"]["positives"] / R["dev"]["rows"], color="#c53030", ls="--", lw=1, label="Đoán ngẫu nhiên")
    ax.set_ylabel("PR-AUC (trung bình ± SD, 5 fold)")
    ax.set_title("CV theo bệnh nhân trên train+validation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "cv_pr_auc.png", dpi=150)
    plt.close(fig)
    made["cv"] = "cv_pr_auc.png"

    # 2. Reliability
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    mx = 0
    for key, lab, col in (("raw", "Chưa hiệu chỉnh (OOF)", "#a0aec0"), ("calibrated", "Đã hiệu chỉnh (OOF)", "#2b6cb0")):
        b = pd.DataFrame(R["reliability_dev"][key])
        ax.plot(b["mean_pred"], b["obs_rate"], "o-", label=lab, color=col)
        mx = max(mx, b["mean_pred"].max(), b["obs_rate"].max())
    if "test" in R:
        b = pd.DataFrame(R["test"]["reliability"]["calibrated"])
        ax.plot(b["mean_pred"], b["obs_rate"], "s--", label="Đã hiệu chỉnh (test)", color="#dd6b20")
        mx = max(mx, b["mean_pred"].max(), b["obs_rate"].max())
    ax.plot([0, mx], [0, mx], "k:", lw=1, label="Hoàn hảo")
    ax.set_xlabel("Xác suất dự đoán trung bình (theo thập phân vị)")
    ax.set_ylabel("Tỷ lệ tử vong thực tế")
    ax.set_title(f"Biểu đồ hiệu chỉnh — {R['chosen_model']}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "reliability.png", dpi=150)
    plt.close(fig)
    made["reliability"] = "reliability.png"

    # 3. SHAP
    top = pd.DataFrame(R["shap"]["top_features"]).head(20)[::-1]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#2b6cb0")
    ax.set_xlabel("mean |SHAP| (log-odds)")
    ax.set_title("20 đặc trưng ảnh hưởng nhất")
    fig.tight_layout()
    fig.savefig(fig_dir / "shap_top20.png", dpi=150)
    plt.close(fig)
    made["shap"] = "shap_top20.png"
    fam = pd.DataFrame(R["shap"]["families"]).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(fam["family"], fam["mean_abs_shap"], color="#38a169")
    ax.set_xlabel("Tổng mean |SHAP| của nhóm đặc trưng")
    ax.set_title("Tầm quan trọng theo nhóm thông tin")
    fig.tight_layout()
    fig.savefig(fig_dir / "shap_families.png", dpi=150)
    plt.close(fig)
    made["shap_family"] = "shap_families.png"

    # 4. Robustness
    rob = pd.DataFrame(R.get("robustness_test") or R["robustness_dev"])
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for sc, gdf in rob.groupby("scenario"):
        ax.plot(100 * gdf["rate"], gdf["pr_auc"], "o-", label=sc)
    ax.set_xlabel("% ca bị che thông tin")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Độ bền khi thiếu dữ liệu" + (" (test)" if R.get("robustness_test") else " (out-of-fold)"))
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "robustness.png", dpi=150)
    plt.close(fig)
    made["robustness"] = "robustness.png"

    # 5. Subgroup forest plot
    sg = pd.DataFrame(R.get("subgroup_test") or R["subgroup_dev"])
    sg = sg[(sg["attribute"] != "TỔNG THỂ") & sg["roc_auc"].notna()]
    if len(sg):
        fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(sg) + 1.2))
        labels = [f"{r.attribute}: {r.group}" for r in sg.itertuples()]
        yk = np.arange(len(sg))[::-1]
        lo = [c[0] if isinstance(c, list) else np.nan for c in sg["roc_auc_ci"]]
        hi = [c[1] if isinstance(c, list) else np.nan for c in sg["roc_auc_ci"]]
        ax.errorbar(sg["roc_auc"], yk, xerr=[sg["roc_auc"] - lo, np.array(hi) - sg["roc_auc"]], fmt="o",
                    color="#2b6cb0", ecolor="#a0aec0", capsize=2)
        overall = (R.get("test", {}).get("models", {}).get(R["chosen_model"], {}).get("metrics", {}).get("roc_auc")
                   or R["dev_metrics"]["roc_auc"])
        ax.axvline(overall, color="#c53030", ls="--", lw=1)
        ax.set_yticks(yk)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("ROC-AUC (CI 95% bootstrap theo bệnh nhân)")
        ax.set_title("Hiệu năng theo nhóm bệnh nhân")
        fig.tight_layout()
        fig.savefig(fig_dir / "subgroup_roc.png", dpi=150)
        plt.close(fig)
        made["subgroup"] = "subgroup_roc.png"

    # 6. Drift
    dr = pd.DataFrame(R["drift"]["periods"])
    tv = pd.DataFrame(R["temporal_validation"]["rows"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(dr["period"], 100 * dr["positive_rate"], "o-", label="Tỷ lệ tử vong thực tế")
    axes[0].plot(dr["period"], 100 * dr["mean_predicted_cal"], "s--", label="Dự đoán trung bình")
    axes[0].set_ylabel("%")
    axes[0].set_title("Drift nhãn và điểm dự đoán")
    axes[0].tick_params(axis="x", rotation=30, labelsize=7)
    axes[0].legend(fontsize=7)
    if len(tv):
        axes[1].plot(tv["period"], tv["oe_ratio"], "o-", label="O/E trước hiệu chỉnh lại")
        if "oe_ratio_heldout_after_recal" in tv:
            axes[1].plot(tv["period"], tv["oe_ratio_heldout_after_recal"], "s--", label="O/E sau hiệu chỉnh lại")
        axes[1].axhline(1, color="k", ls=":", lw=1)
        axes[1].set_title("Train giai đoạn cũ → đánh giá giai đoạn mới")
        axes[1].tick_params(axis="x", rotation=30, labelsize=7)
        axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "drift.png", dpi=150)
    plt.close(fig)
    made["drift"] = "drift.png"

    # 7. PSI heatmap (15 đặc trưng drift nhiều nhất)
    psi = pd.DataFrame({r["period"]: r["psi_all"] for r in R["drift"]["periods"]})
    if len(psi):
        top_f = psi.max(axis=1).sort_values(ascending=False).index[:15]
        h = psi.loc[top_f]
        fig, ax = plt.subplots(figsize=(6.5, 5))
        im = ax.imshow(h.to_numpy(), cmap="Oranges", aspect="auto", vmin=0, vmax=max(0.3, float(h.to_numpy().max())))
        ax.set_xticks(range(h.shape[1]))
        ax.set_xticklabels(h.columns, rotation=30, fontsize=7)
        ax.set_yticks(range(h.shape[0]))
        ax.set_yticklabels(h.index, fontsize=7)
        fig.colorbar(im, ax=ax, label="PSI so với giai đoạn tham chiếu")
        ax.set_title("Drift dữ liệu theo đặc trưng")
        fig.tight_layout()
        fig.savefig(fig_dir / "drift_psi.png", dpi=150)
        plt.close(fig)
        made["psi"] = "drift_psi.png"
    return made


# ------------------------------------------------------------------------------------------------
def write_report(R: dict, docs: Path) -> Path:
    docs = Path(docs)
    figs = figures(R, docs / "figures")
    ch = R["chosen_model"]
    L = []
    L.append("# Giai đoạn 5 — Độ tin cậy và chọn mô hình (P1, Dataset V1)\n")
    L.append(f"Run `{R['run_id']}` · commit `{R['git_commit']}` · split sha256 `{str(R['split_sha256'])[:12]}…` · "
             f"{'**dữ liệu thật**' if R['real_data'] else '**DỮ LIỆU GIẢ — số liệu không có ý nghĩa**'} · "
             f"mô hình `{R['model_version']}`\n")
    L.append("> Chỉ số liệu tổng hợp; ô có số ca < 10 được che. Mọi phân tích trước mục 8 chỉ dùng "
             "train+validation (dự đoán out-of-fold), tập test chỉ mở ở mục 8.\n")
    d = R["dev"]
    L.append(f"Tập phát triển (train+validation): {_n(d['rows'])} lượt nhập viện, {_n(d['patients'])} bệnh nhân, "
             f"{_n(d['positives'])} ca tử vong; {R['n_features']} đặc trưng.\n")

    # 1
    L.append("## 1. Chọn mô hình bằng CV 5 fold theo bệnh nhân\n")
    L.append("Ở GĐ4, Optuna chọn tham số và báo điểm trên cùng tập validation nên điểm bị lạc quan. "
             "Ở đây mỗi ứng viên giữ nguyên tham số tốt nhất của GĐ4, rồi được đánh giá lại bằng CV theo bệnh nhân.\n")
    rows = []
    for n, c in R["cv"].items():
        rows.append([("**" + n + "**") if n == ch else n, f"{c['pr_auc_mean']:.3f} ± {c['pr_auc_sd']:.3f}",
                     f"{c['roc_auc_mean']:.3f} ± {c['roc_auc_sd']:.3f}", _f(c["brier_mean"], 4),
                     f"{c['pr_auc_diff_vs_chosen_mean']:+.3f} ± {c['pr_auc_diff_vs_chosen_sd']:.3f}" if n != ch else "–",
                     _f(c["seconds"], 0)])
    L.append(_table(["Mô hình", "PR-AUC", "ROC-AUC", "Brier", "Mô hình chọn hơn (PR-AUC, theo fold)", "Giây"], rows))
    L.append(f"\n**Chọn: {ch}** (PR-AUC trung bình cao nhất). Nếu chênh lệch nhỏ hơn độ lệch chuẩn giữa các fold, "
             "thì nói đúng là các mô hình **ngang nhau**.\n")
    L.append(f"![CV]({'figures/' + figs['cv']})\n")

    # 2
    L.append("## 2. Hiệu chỉnh xác suất và ngưỡng\n")
    cal = R["calibration"][ch]
    rows = [[m + (" ✅" if m == cal["method"] else ""), _f(v["brier"], 5), _f(v["ece"], 4), _f(v["calibration_slope"], 2),
             _f(v["oe_ratio"], 2), _f(v["pr_auc"], 3)] for m, v in cal["comparison"].items()]
    L.append(_table(["Phương pháp", "Brier", "ECE", "Slope", "O/E", "PR-AUC"], rows))
    L.append("\nCác phương pháp được so sánh bằng CV trên dự đoán out-of-fold, rồi chọn theo Brier. "
             "Slope gần 1 và O/E gần 1 nghĩa là xác suất sát thực tế.\n")
    t = R["thresholds"]
    L.append(_table(["Mức", "Ngưỡng xác suất (đã hiệu chỉnh)", "Độ nhạy", "PPV", "% ca bị xếp ≥ mức"],
                    [["MEDIUM", _f(t["low_medium"], 3), _pct(t["at_medium"]["recall"]), _pct(t["at_medium"]["precision"]),
                      _pct(t["at_medium"]["flag_rate"])],
                     ["HIGH (tối đa F1)", _f(t["medium_high"], 3), _pct(t["at_high"]["recall"]),
                      _pct(t["at_high"]["precision"]), _pct(t["at_high"]["flag_rate"])]]))
    L.append("\nTỷ lệ tử vong thực tế theo mức nguy cơ (out-of-fold): " +
             ", ".join(f"{k} {_pct(v)}" for k, v in R["dev_risk_level_mortality"].items()) + "\n")
    L.append(f"![Reliability]({'figures/' + figs['reliability']})\n")

    # 3
    sh = R["shap"]
    L.append("## 3. Giải thích mô hình (SHAP)\n")
    how = "TreeSHAP chính xác do XGBoost/LightGBM tính" if sh["method"] == "TreeSHAP" else "công thức đóng coef·(x − E[x])"
    L.append(f"Phương pháp: **{sh['method']}** ({how}), tính trên mẫu {_n(sh['sample_n'])} ca. "
             f"Kiểm tra cộng tính: sai số lớn nhất |base + ΣSHAP − logit(p)| = {sh['additivity_max_error']:.2e}.\n")
    st = sh["fold_stability"]
    L.append(f"Độ ổn định giữa mô hình cuối và 5 mô hình fold: Spearman thứ hạng trung bình {_f(st['spearman_mean'])} "
             f"(thấp nhất {_f(st['spearman_min'])}), Jaccard top-20 {_f(st['jaccard_top20_mean'])}.\n")
    L.append(_table(["#", "Đặc trưng", "mean |SHAP|", "Hướng"],
                    [[i + 1, r["feature"], _f(r["mean_abs_shap"], 4),
                      next((dd["doc"] for dd in sh["direction"] if dd["feature"] == r["feature"]), "")]
                     for i, r in enumerate(sh["top_features"][:15])]))
    L.append(f"\n![SHAP]({'figures/' + figs['shap']}) ![Nhóm]({'figures/' + figs['shap_family']})\n")
    L.append("Giải thích từng ca: hàm `HeartGuardModel.predict()` trả về `explanation.top_features` "
             "(đặc trưng, giá trị, SHAP, hướng). Màn hình SHAP ở GĐ6 dùng trực tiếp kết quả này.\n")

    # 4
    L.append("## 4. Độ bền khi thiếu dữ liệu\n")
    rob = pd.DataFrame(R.get("robustness_test") or R["robustness_dev"])
    rows = []
    for sc, gdf in rob.groupby("scenario"):
        gdf = gdf.set_index("rate")
        rows.append([sc] + [_f(gdf.loc[r, "pr_auc"]) if r in gdf.index else "–" for r in (0, .25, .5, 1.0)] +
                    [f"{gdf['pr_auc_drop_pct'].max():.1f}%", _f(gdf.loc[gdf.index.max(), "oe_ratio"], 2)])
    L.append(_table(["Thông tin bị che", "PR-AUC 0%", "25%", "50%", "100%", "Giảm tối đa", "O/E khi che 100%"], rows))
    L.append("\nKhi che một xét nghiệm, cờ `_missing` cũng được bật, giống như bệnh nhân thật sự không được làm xét nghiệm đó. "
             f"Hiệu năng được đánh giá trên {'tập test' if R.get('robustness_test') else 'dự đoán out-of-fold'}.\n")
    L.append(f"![Robustness]({'figures/' + figs['robustness']})\n")

    # 5
    L.append("## 5. Nhất quán giữa các nhóm bệnh nhân\n")
    sg = pd.DataFrame(R.get("subgroup_test") or R["subgroup_dev"])
    rows = []
    for r in sg.itertuples():
        rows.append([r.attribute, r.group, _n(r.n), _n(r.n_pos), _rate(r.positive_rate, r.n_pos),
                     (_f(r.roc_auc) + _ci(getattr(r, "roc_auc_ci", None))) if pd.notna(r.roc_auc) else "–",
                     _f(r.pr_auc_lift, 1) if pd.notna(r.pr_auc_lift) else "–",
                     _f(r.oe_ratio, 2) if pd.notna(r.oe_ratio) else "–",
                     _pct(r.recall) if pd.notna(r.recall) else "–", r.flags if isinstance(r.flags, str) else ""])
    L.append(_table(["Thuộc tính", "Nhóm", "n", "Tử vong", "Tỷ lệ", "ROC-AUC (CI95)", "PR lift", "O/E", "Độ nhạy",
                     "Cảnh báo"], rows))
    fg = pd.DataFrame(R.get("fairness_test") or R["fairness_dev"])
    if len(fg):
        L.append("\n**Chênh lệch lớn nhất trong từng thuộc tính** (recall gap là equal opportunity, FPR gap là một phần của equalized odds):\n")
        L.append(_table(["Thuộc tính", "ROC gap", "Recall gap", "FPR gap", "O/E min–max", "Nhóm ROC thấp nhất"],
                        [[r.attribute, _f(r.roc_auc_gap), _f(r.recall_gap), _f(r.fpr_gap),
                          f"{r.oe_ratio_min:.2f}–{r.oe_ratio_max:.2f}", r.worst_roc_group] for r in fg.itertuples()]))
    L.append("\n`chung_toc` và `bao_hiem` chỉ dùng để đánh giá công bằng, không phải đầu vào của mô hình.\n")
    if "subgroup" in figs:
        L.append(f"![Subgroup]({'figures/' + figs['subgroup']})\n")

    # 6
    L.append("## 6. Drift dữ liệu và drift mô hình (theo lô)\n")
    L.append(f"Giai đoạn tham chiếu: `{R['drift']['reference_period']}`. Ngưỡng PSI: < 0.10 ổn định, 0.10–0.25 vừa, > 0.25 lớn.\n")
    L.append(_table(["Giai đoạn", "n", "Tỷ lệ tử vong", "Dự đoán TB", "O/E", "% đặc trưng PSI>0.25", "PSI điểm",
                     "Trạng thái", "Đặc trưng drift nhất"],
                    [[r["period"], _n(r["n"]), _rate(r["positive_rate"], r["n_pos"]), _pct(r["mean_predicted_cal"]), _f(r["oe_ratio"], 2),
                      _pct(r["share_features_major"]), _f(r["score_psi"]), r["status"],
                      ", ".join(f"{x['feature']} ({x['psi']:.2f})" for x in r["top_features"][:3])]
                     for r in R["drift"]["periods"]]))
    tv = R["temporal_validation"]
    L.append(f"\n**Drift mô hình:** huấn luyện trên {', '.join(tv['train_periods'])} "
             f"(PR-AUC CV nội bộ {_f(tv['train_cv_pr_auc'])}), rồi đánh giá trên các giai đoạn sau:\n")
    L.append(_table(["Giai đoạn", "n", "Tỷ lệ", "ROC-AUC", "PR-AUC", "PR lift", "ECE", "O/E", "O/E sau hiệu chỉnh lại"],
                    [[r["period"], _n(r["n"]), _rate(r["positive_rate"], r["n_pos"]), _f(r["roc_auc"]), _f(r["pr_auc"]),
                      _f(r["pr_auc_lift"], 1), _f(r["ece"], 4), _f(r["oe_ratio"], 2),
                      _f(r.get("oe_ratio_heldout_after_recal"), 2)] for r in tv["rows"]]))
    L.append("\nBiện pháp xử lý khi phát hiện drift: cập nhật lại intercept trên một nửa số bệnh nhân của giai đoạn mới, "
             "rồi đánh giá trên nửa còn lại. Cách này không cần huấn luyện lại mô hình.\n")
    ed = R["explanation_drift"]
    if ed["stability"]:
        L.append(f"**Drift giải thích:** so thứ hạng SHAP của mô hình huấn luyện riêng từng giai đoạn với giai đoạn `{ed['reference']}`: " +
                 "; ".join(f"{s['period']} ρ={s['spearman_vs_ref']:.2f}" for s in ed["stability"]) + ".\n")
    L.append(f"![Drift]({'figures/' + figs['drift']})" + (f" ![PSI]({'figures/' + figs['psi']})" if "psi" in figs else "") + "\n")

    # 7
    L.append("## 7. Quy trình xem xét thủ công\n")
    rv = R.get("review_test") or R["review_dev"]
    L.append(f"Đánh giá trên {'tập test' if R.get('review_test') else 'dự đoán out-of-fold'}: "
             f"**{_pct(rv['flagged_share'])}** số ca vào hàng đợi; hàng đợi bắt được **{_pct(rv['deaths_captured_share'])}** "
             f"số ca tử vong. Tỷ lệ tử vong trong hàng đợi là {_pct(rv['death_rate_in_queue'])}, ngoài hàng đợi là "
             f"{_pct(rv['death_rate_not_flagged'])}.\n")
    L.append(_table(["Lý do", "% ca", "Tỷ lệ tử vong trong nhóm"],
                    [[k, _pct(v["share"]), _pct(v["death_rate"]) if v["death_rate"] is not None and v["n"] >= SMALL else "–"]
                     for k, v in rv["by_reason"].items()]))
    L.append("\nƯu tiên: 1 = HIGH_RISK, 2 = NEAR_THRESHOLD/LOW_CONFIDENCE, 3 = chỉ do chất lượng dữ liệu. "
             "Trạng thái review là PENDING → IN_REVIEW → REVIEWED | ESCALATED (`contracts/api_contract.md`). "
             "LOW_CONFIDENCE dùng độ lệch chuẩn giữa 5 mô hình fold. Ngưỡng này lấy theo phân vị trên dữ liệu "
             "phát triển, mà mỗi dòng ở đó đã được 4/5 mô hình nhìn thấy, nên tỷ lệ cờ thực tế trên dữ liệu mới "
             "cao hơn dự kiến; số đo trên test phản ánh đúng hơn.\n")

    # 8
    L.append("## 8. Kết quả trên tập test (mở một lần)\n")
    if "test" in R:
        T = R["test"]
        L.append(f"Test: {_n(T['rows'])} lượt nhập viện, {_n(T['patients'])} bệnh nhân, {_n(T['positives'])} ca tử vong. "
                 "CI 95% bootstrap theo bệnh nhân.\n")
        rows = []
        for n, v in T["models"].items():
            m, ci = v["metrics"], v["ci95"]
            rows.append([("**" + n + "**") if n == ch else n, _f(m["pr_auc"]) + _ci(ci["pr_auc"]),
                         _f(m["roc_auc"]) + _ci(ci["roc_auc"]), _f(m["f1"]) + _ci(ci["f1"]),
                         _f(m["brier"], 4) + _ci(ci["brier"], 4), _f(m["ece"], 4), _f(m["calibration_slope"], 2),
                         _f(m["oe_ratio"], 2), _pct(m["recall"]), _pct(m["precision"])])
        L.append(_table(["Mô hình", "PR-AUC", "ROC-AUC", "F1", "Brier", "ECE", "Slope", "O/E", "Độ nhạy", "PPV"], rows))
        m = T["models"][ch]["metrics"]
        L.append(f"\nMa trận nhầm lẫn của {ch} tại ngưỡng HIGH {m['threshold']:.3f}: TP {_n(m['tp'])}, FP {_n(m['fp'])}, "
                 f"FN {_n(m['fn'])}, TN {_n(m['tn'])}. Hiệu chỉnh: Brier chưa hiệu chỉnh "
                 f"{_f(T['models'][ch]['metrics_uncalibrated']['brier'], 4)} → đã hiệu chỉnh {_f(m['brier'], 4)}.\n")
        L.append("Tỷ lệ tử vong theo mức nguy cơ trên test: " +
                 ", ".join(f"{k} {_pct(v)}" for k, v in T["risk_level_mortality"].items()) + "\n")
    else:
        L.append("Tập test **chưa mở**. Khi nhóm đã chốt mô hình, chạy lại với `--open-test`.\n")

    # 9
    L.append("## 9. Gói mô hình\n")
    L.append(f"`{R['model_version']}`, gồm model + calibrator + 5 mô hình fold + ngưỡng + luật review + tham chiếu drift "
             f"+ SHA256SUMS. Gói nằm cục bộ ở `artifacts/models/p1/{R['model_version']}/`"
             + (f", và đã tải lên `{R['bundle_s3']}`" if R.get("bundle_s3") else "") + ". GĐ6 nạp gói bằng `trustworthy_ai.bundle.HeartGuardModel`.\n")

    path = docs / "stage5_report.md"
    docs.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def model_card(R: dict) -> str:
    ch = R["chosen_model"]
    c = R["cv"][ch]
    test = R.get("test", {}).get("models", {}).get(ch)
    lines = [f"# Model card — {R.get('model_version', ch)}", "",
             "**Mục đích:** ước lượng xác suất tử vong trong viện của bệnh nhân suy tim ≥ 18 tuổi, tại thời điểm 24 giờ "
             "sau nhập viện (P1). Đây là nguyên mẫu nghiên cứu, **không dùng cho quyết định lâm sàng**.", "",
             f"**Dữ liệu:** MIMIC-IV v3.1, Gold P1 {R['data_version']}, split sha256 `{R['split_sha256']}`.", "",
             f"**Mô hình:** {ch}, tham số Optuna từ GĐ4, huấn luyện lại trên train+validation; hiệu chỉnh "
             f"`{R['calibration'][ch]['method']}` học từ dự đoán out-of-fold.", "",
             f"**CV 5 fold theo bệnh nhân:** PR-AUC {c['pr_auc_mean']:.3f} ± {c['pr_auc_sd']:.3f}, "
             f"ROC-AUC {c['roc_auc_mean']:.3f} ± {c['roc_auc_sd']:.3f}.", ""]
    if test:
        m, ci = test["metrics"], test["ci95"]
        lines += [f"**Test (mở một lần):** PR-AUC {m['pr_auc']:.3f}{_ci(ci['pr_auc'])}, ROC-AUC {m['roc_auc']:.3f}"
                  f"{_ci(ci['roc_auc'])}, Brier {m['brier']:.4f}, F1 {m['f1']:.3f}.", ""]
    lines += [f"**Ngưỡng:** MEDIUM ≥ {R['thresholds']['low_medium']:.3f}, HIGH ≥ {R['thresholds']['medium_high']:.3f} "
              "(xác suất đã hiệu chỉnh).", "",
              "**Giới hạn:** dữ liệu từ một bệnh viện; sinh hiệu chỉ có với ca ICU; tỷ lệ tử vong tăng theo thời gian "
              "nên cần hiệu chỉnh lại định kỳ; một số dòng xét nghiệm thiếu storetime (D-014); SHAP giải thích "
              "mô hình, không phải quan hệ nhân quả; race và insurance chỉ dùng để đánh giá công bằng."]
    return "\n".join(lines)

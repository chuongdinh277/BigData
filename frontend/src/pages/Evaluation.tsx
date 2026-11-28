import { useEffect, useState } from "react";
import { api, num, pct } from "../api";
import { TaskPicker, useTask } from "../components/TaskContext";
import { ErrorBox, Table } from "../components/ui";

const TABS = ["Chỉ số", "Hiệu chuẩn", "Thiếu dữ liệu", "Nhóm bệnh nhân", "Review"] as const;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type R = Record<string, any>;

/** Đánh giá mô hình (Chưởng): ROC-AUC, PR-AUC, F1, hiệu chuẩn, mốc lâm sàng, thiếu dữ liệu, nhóm phụ, review. */
export default function Evaluation() {
  const { task } = useTask();
  const [tab, setTab] = useState<(typeof TABS)[number]>("Chỉ số");
  const [d, setD] = useState<Record<string, R>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setD({});
    setError(null);
    ["summary", "metrics", "baselines", "calibration", "robustness", "subgroup", "review"].forEach((k) =>
      api.report(task, k).then((r) => setD((o) => ({ ...o, [k]: r }))).catch((e) => setError(e.message)));
  }, [task]);

  const fig = (name: string) => <img src={api.figure(task, name)} alt={name} onError={(e) => ((e.target as HTMLImageElement).style.display = "none")} />;
  const m = d.metrics;
  const s = d.summary;

  return (
    <>
      <div className="row">
        <div>
          <h1>Đánh giá mô hình</h1>
          <p className="sub">
            {s ? <>Chạy <code>{s.run_id}</code> · commit <code>{s.git_commit}</code> · {s.real_data ? "dữ liệu thật" : <strong>DỮ LIỆU GIẢ (demo)</strong>} · mô hình chọn: <strong>{s.chosen_model}</strong> ({s.chosen_by})</> : "Đang tải…"}
          </p>
        </div>
        <span className="spacer" />
        <TaskPicker />
      </div>
      <ErrorBox error={error} />
      <div className="seg" style={{ marginBottom: 16 }}>
        {TABS.map((t) => <button key={t} className={t === tab ? "on" : ""} onClick={() => setTab(t)}>{t}</button>)}
      </div>

      {tab === "Chỉ số" && m && (
        <div className="grid">
          <div className="card">
            <h2>CV 5 fold theo bệnh nhân (train + validation)</h2>
            <Table head={["Mô hình", "PR-AUC", "ROC-AUC", "Brier", "Chênh với mô hình chọn"]}
              rows={(Object.entries(m.cv) as [string, R][]).map(([k, v]: [string, R]) => [k, `${num(v.pr_auc_mean)} ± ${num(v.pr_auc_sd)}`,
                `${num(v.roc_auc_mean)} ± ${num(v.roc_auc_sd)}`, num(v.brier_mean, 4), v.pr_auc_diff_vs_chosen_mean ? num(v.pr_auc_diff_vs_chosen_mean) : "–"])} />
          </div>
          {d.baselines?.baselines_dev && (
            <div className="card">
              <h2>So với mốc sàn và thang điểm lâm sàng</h2>
              <Table head={["Mốc", "n", "Phủ", "ROC-AUC", "PR-AUC", "Mô hình trên cùng tập con"]}
                rows={d.baselines.baselines_dev.map((b: R) => [b.baseline, b.n, pct(b.coverage), num(b.roc_auc), num(b.pr_auc),
                  b.model_pr_auc_same_subset != null ? `ROC ${num(b.model_roc_auc_same_subset)} · PR ${num(b.model_pr_auc_same_subset)}` : ""])} />
            </div>
          )}
          {m.test ? (
            <div className="card">
              <h2>Tập test (mở một lần) · CI 95% bootstrap theo bệnh nhân</h2>
              <Table head={["Mô hình", "PR-AUC", "ROC-AUC", "F1", "Brier", "Độ nhạy", "PPV"]}
                rows={(Object.entries(m.test.models) as [string, R][]).map(([k, v]: [string, R]) => [k,
                  `${num(v.metrics.pr_auc)} [${num(v.ci95.pr_auc[0])}, ${num(v.ci95.pr_auc[1])}]`,
                  `${num(v.metrics.roc_auc)} [${num(v.ci95.roc_auc[0])}, ${num(v.ci95.roc_auc[1])}]`,
                  num(v.metrics.f1), num(v.metrics.brier, 4), pct(v.metrics.recall), pct(v.metrics.precision)])} />
            </div>
          ) : <div className="note">Tập test chưa mở.</div>}
          <div className="grid g2 figs">{fig("cv_pr_auc.png")}{fig("learning_curve.png")}</div>
        </div>
      )}

      {tab === "Hiệu chuẩn" && d.calibration && (
        <div className="grid">
          <div className="grid g2 figs">{fig("reliability.png")}{fig("decision_curve.png")}</div>
          <div className="card">
            <h2>Bảng điểm cắt</h2>
            <Table head={["Ngưỡng", "% cảnh báo", "Độ nhạy", "PPV", "Số ca cần xem / 1 ca đúng"]}
              rows={(d.calibration.operating_test ?? d.calibration.operating_dev).map((r: R) => [num(r.threshold, 2), pct(r.alert_rate), pct(r.sensitivity), pct(r.ppv), num(r.nne, 1)])} />
          </div>
        </div>
      )}

      {tab === "Thiếu dữ liệu" && d.robustness && (
        <div className="grid">
          <div className="figs">{fig("robustness.png")}</div>
          <div className="card">
            <Table head={["Thông tin bị che", "% che", "PR-AUC", "Giảm", "O/E"]}
              rows={(d.robustness.robustness_test ?? d.robustness.robustness_dev).map((r: R) => [r.scenario, pct(r.rate, 0), num(r.pr_auc), `${num(r.pr_auc_drop_pct, 1)}%`, num(r.oe_ratio, 2)])} />
          </div>
        </div>
      )}

      {tab === "Nhóm bệnh nhân" && d.subgroup && (
        <div className="grid">
          <div className="card">
            <Table head={["Thuộc tính", "Nhóm", "n", "Ca dương", "ROC-AUC", "PR lift", "O/E", "Độ nhạy"]}
              rows={(d.subgroup.subgroup_test ?? d.subgroup.subgroup_dev).map((r: R) => [r.attribute, r.group, r.n, r.n_pos ?? "<10", num(r.roc_auc), num(r.pr_auc_lift, 1), num(r.oe_ratio, 2), pct(r.recall)])} />
          </div>
          <div className="figs">{fig("subgroup_roc.png")}</div>
        </div>
      )}

      {tab === "Review" && d.review && (
        <div className="grid">
          <div className="grid g2 figs">{fig("risk_coverage.png")}</div>
          {(() => {
            const cv = d.review.confidence_validation_test ?? d.review.confidence_validation_dev;
            return cv?.low_confidence?.brier != null ? (
              <div className="card">
                <h2>Luật "độ tin cậy thấp" có bắt đúng ca khó không?</h2>
                <Table head={["Nhóm", "n", "Tỷ lệ dương", "Brier", "ROC-AUC", "ECE"]}
                  rows={([["Bị gắn cờ tin cậy thấp", cv.low_confidence], ["Còn lại", cv.confident]] as [string, R][]).map(([n, v]) => [n, v.n, pct(v.positive_rate), num(v.brier, 4), num(v.roc_auc), num(v.ece, 4)])} />
                <p className="sub" style={{ marginTop: 8 }}>Brier chuẩn hóa nhóm cờ / nhóm còn lại = <strong>{num(cv.brier_scaled_ratio_lowconf_vs_confident, 2)}</strong> (lớn hơn 1 nghĩa là luật bắt đúng ca mô hình hay sai).</p>
              </div>
            ) : null;
          })()}
          <div className="card">
            <h2>Năng lực review: xem top x% ca bắt được bao nhiêu ca dương?</h2>
            <Table head={["% ca được xem", "Số ca", "% ca dương bắt được", "Tỷ lệ dương trong hàng đợi"]}
              rows={(d.review.capacity_dev ?? []).map((r: R) => [pct(r.review_share, 0), r.n_reviewed, pct(r.positives_captured_share), pct(r.positive_rate_in_queue)])} />
          </div>
        </div>
      )}
    </>
  );
}

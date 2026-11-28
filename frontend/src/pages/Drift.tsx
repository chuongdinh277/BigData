import { useEffect, useState } from "react";
import { api, num, pct } from "../api";
import { TaskPicker, useTask } from "../components/TaskContext";
import { ErrorBox, Table } from "../components/ui";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type R = Record<string, any>;
const STATUS_CLS: Record<string, string> = { "ỔN ĐỊNH": "LOW", "THEO DÕI": "MEDIUM", "CẢNH BÁO": "HIGH" };

/** Giám sát drift (Chưởng): drift dữ liệu, drift mô hình, drift mã ICD, chiến lược bảo trì, kiểm tra lô mới. */
export default function Drift() {
  const { task } = useTask();
  const [d, setD] = useState<R | null>(null);
  const [batch, setBatch] = useState<R | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setD(null);
    setBatch(null);
    api.report(task, "drift").then(setD).catch((e) => setError(e.message));
  }, [task]);

  const fig = (name: string) => <img src={api.figure(task, name)} alt={name} onError={(e) => ((e.target as HTMLImageElement).style.display = "none")} />;

  const checkBatch = async () => {
    // Lô demo: nhân bản ca demo và làm lệch có chủ đích (creatinine +40%, mất NT-proBNP) để thấy cảnh báo
    const cases = await api.cases(task);
    const recs = Array.from({ length: 60 }, (_, i) => {
      const f = { ...cases[i % cases.length].features } as R;
      Object.keys(f).filter((k) => k.startsWith("creatinine_") && !k.endsWith("_missing") && f[k] != null).forEach((k) => (f[k] = f[k] * 1.4 * (0.9 + 0.2 * Math.random())));
      Object.keys(f).filter((k) => k.startsWith("ntprobnp_")).forEach((k) => (f[k] = null));
      return f;
    });
    try {
      setBatch(await api.drift(task, recs));
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const periods: R[] = d?.drift?.periods ?? [];
  return (
    <>
      <div className="row">
        <div>
          <h1>Giám sát drift</h1>
          <p className="sub">Drift dữ liệu (đầu vào đổi, đo ngay không cần nhãn) và drift mô hình (hiệu năng, hiệu chuẩn giảm, cần nhãn: P1 khi ra viện, P2 sau 30 ngày, P3 sau 90 ngày). Đánh giá theo lô thời gian <code>anchor_year_group</code>.</p>
        </div>
        <span className="spacer" />
        <TaskPicker />
      </div>
      <ErrorBox error={error} />
      {d && (
        <div className="grid">
          <div className="card">
            <h2>Drift dữ liệu theo lô (tham chiếu {d.drift.reference_period})</h2>
            <Table head={["Giai đoạn", "n", "Tỷ lệ dương", "Dự đoán TB", "O/E", "Đặc trưng PSI > 0.25", "Vượt ngưỡng null", "AUC miền", "PSI điểm", "Trạng thái"]}
              rows={periods.map((r) => [r.period, r.n, pct(r.positive_rate), pct(r.mean_predicted_cal), num(r.oe_ratio, 2), pct(r.share_features_major),
                r.n_features_over_null ?? "–", r.domain_classifier ? num(r.domain_classifier.auc) : "tham chiếu", num(r.score_psi),
                <span key="s" className={`pill ${STATUS_CLS[r.status] ?? "gray"}`}>{r.status}</span>])} />
            {d.drift.null_thresholds && (
              <p className="sub" style={{ marginTop: 8 }}>
                Ngưỡng null: bootstrap {d.drift.null_thresholds.n_boot} lần trong lô tham chiếu, cỡ mẫu {d.drift.null_thresholds.sample_size}, phân vị {pct(d.drift.null_thresholds.quantile, 0)} → trung vị {num(d.drift.null_thresholds.median_threshold)}. AUC miền ≈ 0.5 là không drift.
              </p>
            )}
          </div>
          <div className="grid g2 figs">{fig("drift.png")}{fig("drift_psi.png")}</div>
          <div className="card">
            <h2>Drift mô hình: huấn luyện {d.temporal_validation.train_periods.join(", ")} → đánh giá các lô sau</h2>
            <Table head={["Giai đoạn", "n", "ROC-AUC", "PR-AUC", "ECE", "O/E", "O/E sau hiệu chỉnh lại"]}
              rows={d.temporal_validation.rows.map((r: R) => [r.period, r.n, num(r.roc_auc), num(r.pr_auc), num(r.ece, 4), num(r.oe_ratio, 2), num(r.oe_ratio_heldout_after_recal, 2)])} />
          </div>
          {d.maintenance?.available && (
            <div className="card">
              <h2>Phát hiện drift rồi làm gì: 3 chiến lược bảo trì</h2>
              <div className="figs" style={{ marginBottom: 10 }}>{fig("maintenance.png")}</div>
              <Table head={["Giai đoạn", "Chiến lược", "PR-AUC", "O/E", "ECE"]}
                rows={d.maintenance.rows.map((r: R) => [r.period, r.strategy, num(r.pr_auc), num(r.oe_ratio, 2), num(r.ece, 4)])} />
            </div>
          )}
          {d.coding_drift?.available && (
            <div className="card">
              <h2>Drift do đổi hệ mã ICD-9 → ICD-10 (10/2015)</h2>
              <Table head={["Cờ bệnh nền", "Chênh lệch lớn nhất giữa nhóm phiên bản ICD"]}
                rows={Object.entries(d.coding_drift.largest_version_gaps).map(([k, v]) => [<code key="k">{k}</code>, num(v as number, 3)])} />
            </div>
          )}
          <div className="card">
            <div className="row">
              <h2 style={{ margin: 0 }}>Kiểm tra một lô ca mới (không cần nhãn)</h2>
              <span className="spacer" />
              <button className="btn primary" onClick={checkBatch}>Chạy lô demo bị lệch</button>
            </div>
            <p className="sub">Gửi lô ca lên <code>POST /drift/{task}</code>; API so với phân phối tham chiếu lưu trong gói mô hình.</p>
            {batch && (
              <>
                <div className="row" style={{ marginBottom: 10 }}>
                  <span className={`pill ${STATUS_CLS[batch.status] ?? "gray"}`}>{batch.status}</span>
                  <span>PSI điểm dự đoán: {num(batch.score_psi)}</span>
                  <span>{pct(batch.share_major)} đặc trưng thay đổi lớn</span>
                </div>
                <Table head={["Đặc trưng", "PSI", "Mức", "Tỷ lệ thiếu (tham chiếu → lô)"]}
                  rows={batch.features.slice(0, 10).map((f: R) => [<code key="f">{f.feature}</code>, num(f.psi), f.level, `${pct(f.missing_rate_ref)} → ${pct(f.missing_rate_cur)}`])} />
              </>
            )}
          </div>
        </div>
      )}
    </>
  );
}

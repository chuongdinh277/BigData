import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, num, pct } from "../api";
import { useTask } from "../components/TaskContext";
import { ErrorBox, Kpis, Table } from "../components/ui";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type R = Record<string, any>;

const PILLARS: [string, string, string][] = [
  ["Hiệu chuẩn", "Xác suất 30% có đúng là 30%?", "/evaluation"],
  ["Giải thích", "Vì sao bệnh nhân này nguy cơ cao?", "/explain"],
  ["Dữ liệu thiếu", "Thiếu xét nghiệm thì còn đúng không?", "/evaluation"],
  ["Nhóm phụ", "Có đều với mọi nhóm bệnh nhân?", "/evaluation"],
  ["Drift", "Dữ liệu và mô hình có trôi theo thời gian?", "/drift"],
  ["Xem xét thủ công", "Máy không chắc thì ai quyết?", "/reviews"],
];

/** Tổng quan (Chưởng): bảng so sánh 3 bài toán + hàng đợi + 6 trụ "đáng tin cậy". */
export default function Dashboard() {
  const { tasks, error: ctxError } = useTask();
  const [rep, setRep] = useState<Record<string, R>>({});
  const [stats, setStats] = useState<{ by_status: Record<string, number>; open_by_priority: Record<string, number> } | null>(null);

  useEffect(() => {
    api.reviewStats().then(setStats).catch(() => undefined);
    tasks.filter((t) => t.enabled).forEach((t) =>
      Promise.all([api.report(t.task, "summary"), api.report(t.task, "metrics"), api.report(t.task, "calibration")])
        .then(([s, m, c]) => setRep((o) => ({ ...o, [t.task]: { ...s, ...m, ...c } })))
        .catch(() => undefined));
  }, [tasks]);

  const by = stats?.by_status ?? {};
  const rows = tasks.map((t) => {
    const r = rep[t.task];
    const ch = r?.chosen_model;
    const cv = ch ? r.cv[ch] : null;
    const tm = ch && r?.test ? r.test.models[ch] : null;
    const dev = r?.dev;
    return [
      <strong key="t">{t.task.toUpperCase()}</strong>,
      t.name,
      t.prediction_time,
      dev ? dev.rows.toLocaleString("en-US") : "–",
      dev ? pct(dev.positives / dev.rows, 2) : "–",
      ch ?? "–",
      cv ? `${num(cv.pr_auc_mean)} ±${num(cv.pr_auc_sd)}` : "–",
      cv ? `${num(cv.roc_auc_mean)} ±${num(cv.roc_auc_sd)}` : "–",
      tm ? `${num(tm.metrics.pr_auc)} [${num(tm.ci95.pr_auc[0])}, ${num(tm.ci95.pr_auc[1])}]` : "chưa mở",
      tm ? num(tm.metrics.f1) : "–",
      tm ? num(tm.metrics.brier, 4) : num(r?.dev_metrics?.brier, 4),
      tm ? num(tm.metrics.oe_ratio, 2) : num(r?.dev_metrics?.oe_ratio, 2),
      t.thresholds ? num(t.thresholds.medium_high) : "–",
      r ? (r.real_data ? <span key="d" className="pill LOW">THẬT</span> : <span key="d" className="pill MEDIUM">GIẢ</span>) : "–",
    ];
  });

  return (
    <>
      <h1>HeartGuard AI · Tổng quan hệ thống</h1>
      <p className="sub">Dự đoán nguy cơ ở bệnh nhân suy tim (MIMIC-IV v3.1). Mỗi dự đoán kèm xác suất đã hiệu chuẩn, giải thích SHAP, mức tin cậy, giám sát drift và hàng đợi xem xét thủ công. PR-AUC là chỉ số chính vì lớp dương hiếm; mức ngẫu nhiên bằng tỷ lệ dương.</p>
      <ErrorBox error={ctxError} />
      <div className="card" style={{ marginBottom: 10 }}>
        <h2>Ba bài toán · mô hình đang phục vụ</h2>
        <Table head={["Bài", "Kết cục", "Mốc dự đoán", "Dev (lượt)", "Tỷ lệ dương", "Mô hình", "PR-AUC CV", "ROC-AUC CV", "PR-AUC test [CI95]", "F1 test", "Brier", "O/E", "Ngưỡng HIGH", "Dữ liệu"]} rows={rows} />
        <p className="sub" style={{ margin: "6px 0 0" }}>CV: 5 fold theo bệnh nhân trên train+validation, trung bình ± độ lệch chuẩn. Test mở đúng một lần. O/E = tỷ lệ quan sát / dự đoán trung bình (1 = hiệu chuẩn đúng).</p>
      </div>
      <div style={{ marginBottom: 10 }}>
        <Kpis items={[
          { k: "Chờ xem", v: by.PENDING ?? 0 },
          { k: "Đang xem", v: by.IN_REVIEW ?? 0 },
          { k: "Ưu tiên 1 đang mở", v: stats?.open_by_priority["1"] ?? 0, h: "nguy cơ cao" },
          { k: "Ưu tiên 2 đang mở", v: stats?.open_by_priority["2"] ?? 0, h: "độ tin cậy thấp" },
          { k: "Đã xem", v: by.REVIEWED ?? 0 },
          { k: "Chuyển cấp", v: by.ESCALATED ?? 0 },
        ]} />
      </div>
      <div className="card">
        <h2>Sáu trụ "đáng tin cậy" theo đề Subject 4</h2>
        <Table head={["Trụ", "Câu hỏi", "Xem ở"]} rows={PILLARS.map(([a, q, to]) => [<strong key="a">{a}</strong>, q, <Link key="l" to={to}>{to}</Link>])} />
      </div>
    </>
  );
}

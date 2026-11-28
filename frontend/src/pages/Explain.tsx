import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, num } from "../api";
import { TaskPicker, useTask } from "../components/TaskContext";
import { ErrorBox, Table, Waterfall } from "../components/ui";

/** Màn hình Giải thích (Khoẻ): SHAP của ca vừa dự đoán + SHAP toàn cục + kiểm chứng giải thích. */
export default function Explain() {
  const { task, last } = useTask();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [rep, setRep] = useState<Record<string, any> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setRep(null);
    api.report(task, "explain").then(setRep).catch((e) => setError(e.message));
  }, [task]);

  const sh = rep?.shap;
  const mine = last && last.task === task ? last : null;

  return (
    <>
      <div className="row">
        <div>
          <h1>Giải thích mô hình</h1>
          <p className="sub">SHAP cho biết mỗi yếu tố đẩy nguy cơ của <em>chính bệnh nhân này</em> lên hay xuống bao nhiêu (thang log-odds, trước hiệu chuẩn). SHAP giải thích mô hình, không chứng minh quan hệ nhân quả.</p>
        </div>
        <span className="spacer" />
        <TaskPicker />
      </div>
      <div className="grid g2">
        <div className="card">
          <h2>Ca hiện tại</h2>
          {mine?.explanation ? (
            <>
              <p className="sub" style={{ marginBottom: 10 }}>
                Ca <code>{mine.case_id}</code> · {mine.explanation.method} · giá trị nền (log-odds) {num(mine.explanation.base_value, 2)}
              </p>
              <Waterfall items={mine.explanation.top_features} />
            </>
          ) : (
            <div className="empty">
              Chưa có ca nào. <Link to="/predict">Dự đoán một ca</Link> để xem giải thích.
            </div>
          )}
        </div>
        <div className="card">
          <h2>Toàn cục: mô hình dựa vào gì nói chung</h2>
          <ErrorBox error={error} />
          {sh ? (
            <Table head={["#", "Đặc trưng", "mean |SHAP|", "Hướng"]}
              rows={sh.top_features.slice(0, 15).map((r: { feature: string; mean_abs_shap: number }, i: number) => [
                i + 1, <code key="f">{r.feature}</code>, num(r.mean_abs_shap, 4),
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                sh.direction.find((d: any) => d.feature === r.feature)?.doc ?? "",
              ])} />
          ) : !error && <div className="empty">Đang tải…</div>}
        </div>
      </div>

      {sh && (
        <div className="grid g3" style={{ marginTop: 16 }}>
          <div className="card">
            <h2>Đúng về toán</h2>
            <p className="sub">|giá trị nền + ΣSHAP − logit(p)| lớn nhất</p>
            <div className="stat"><div className="value">{Number(sh.additivity_max_error).toExponential(1)}</div></div>
          </div>
          <div className="card">
            <h2>Ổn định giữa 5 fold</h2>
            <p className="sub">Tương quan thứ hạng Spearman · Jaccard top-20</p>
            <div className="stat"><div className="value">{num(sh.fold_stability.spearman_mean, 2)} · {num(sh.fold_stability.jaccard_top20_mean, 2)}</div></div>
          </div>
          <div className="card">
            <h2>Trung thực</h2>
            <p className="sub">Che 5 yếu tố SHAP lớn nhất so với che 5 yếu tố ngẫu nhiên</p>
            <div className="stat"><div className="value">
              {num(sh.deletion_curve?.find((d: { k: number }) => d.k === 5)?.ratio_shap_vs_random, 1)}×
            </div></div>
          </div>
        </div>
      )}

      {sh?.plausibility?.length > 0 && (
        <div className="card" style={{ marginTop: 16 }}>
          <h2>Hợp lý lâm sàng: chiều tác động có đúng y văn không?</h2>
          <Table head={["Đặc trưng", "Kỳ vọng", "Spearman giá trị ~ SHAP", "Kết luận", "Lý do"]}
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            rows={sh.plausibility.map((p: any) => [
              <code key="f">{p.feature}</code>, p.expected, num(p.spearman_value_vs_shap, 2),
              <span key="v" className={`pill ${p.verdict === "đúng chiều" ? "LOW" : p.verdict === "NGƯỢC CHIỀU" ? "HIGH" : "gray"}`}>{p.verdict}</span>,
              p.rationale,
            ])} />
        </div>
      )}

      {rep?.explanation_drift?.top5_by_period && (
        <div className="card" style={{ marginTop: 16 }}>
          <h2>Lý do của mô hình có đổi theo thời gian không?</h2>
          <Table head={["Giai đoạn", "Top 5 yếu tố"]}
            rows={Object.entries(rep.explanation_drift.top5_by_period).map(([p, f]) => [p, (f as string[]).join(", ")])} />
        </div>
      )}
    </>
  );
}

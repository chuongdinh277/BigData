import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, pct, PredictionRow } from "../api";
import { useTask } from "../components/TaskContext";
import { ErrorBox, RiskPill } from "../components/ui";

/** Lịch sử dự đoán (Chưởng): mọi dự đoán đều truy vết được về phiên bản mô hình và thời điểm. */
export default function History() {
  const { setLast } = useTask();
  const nav = useNavigate();
  const [rows, setRows] = useState<PredictionRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.predictions().then(setRows).catch((e) => setError(e.message));
  }, []);

  const open = async (id: number) => {
    const p = await api.prediction(id);
    setLast({ ...p.result, prediction_id: p.id, case_id: p.case_id, task: p.task });
    nav("/explain");
  };

  return (
    <>
      <h1>Lịch sử dự đoán</h1>
      <p className="sub">Mỗi dự đoán lưu kèm phiên bản mô hình, thời điểm và đầu vào (chỉ ca demo đã khử định danh). Bấm một dòng để xem lại giải thích.</p>
      <ErrorBox error={error} />
      <div className="card">
        {rows.length === 0 ? <div className="empty">Chưa có dự đoán nào.</div> : (
          <table className="t">
            <thead><tr><th>#</th><th>Thời điểm</th><th>Bài</th><th>Ca</th><th>Nguy cơ</th><th className="num">Xác suất</th><th>Review</th><th>Mô hình</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id} className="click" onClick={() => open(r.id)}>
                  <td>{r.id}</td>
                  <td>{new Date(r.created_at).toLocaleString("vi-VN")}</td>
                  <td>{r.task.toUpperCase()}</td>
                  <td><code>{r.case_id}</code></td>
                  <td><RiskPill level={r.risk_level} /></td>
                  <td className="num">{pct(r.calibrated_probability)}</td>
                  <td>{r.review_flag ? <span className="pill HIGH">Có</span> : <span className="pill gray">Không</span>}</td>
                  <td><code style={{ fontSize: 11 }}>{r.model_version}</code></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}

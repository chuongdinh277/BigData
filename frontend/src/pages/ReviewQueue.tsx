import { useCallback, useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, pct, reasonText, Review, ReviewStatus } from "../api";
import { ErrorBox, RiskPill, StatusPill, Waterfall } from "../components/ui";

const NEXT: Record<string, ReviewStatus[]> = { PENDING: ["IN_REVIEW"], IN_REVIEW: ["REVIEWED", "ESCALATED"] };
const STATUS_FILTERS = ["", "PENDING", "IN_REVIEW", "REVIEWED", "ESCALATED"];

/** Màn hình Hàng đợi review (Hà): ca nguy cơ cao hoặc độ tin cậy thấp chờ người xem; có nhật ký đầy đủ. */
export default function ReviewQueue() {
  const [params, setParams] = useSearchParams();
  const [status, setStatus] = useState("PENDING");
  const [rows, setRows] = useState<Review[]>([]);
  const [open, setOpen] = useState<Review | null>(null);
  const [reviewer, setReviewer] = useState(() => localStorage.getItem("hg_reviewer") ?? "");
  const [decision, setDecision] = useState("AGREE");
  const [note, setNote] = useState("");
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    api.reviews(status || undefined).then(setRows).catch((e) => setError(e.message));
  }, [status]);
  useEffect(load, [load]);

  useEffect(() => {
    const id = params.get("open");
    if (id) api.review(Number(id)).then(setOpen).catch((e) => setError(e.message));
  }, [params]);

  const act = async (next: ReviewStatus) => {
    if (!open) return;
    if (!reviewer.trim()) return setError("Nhập tên người duyệt trước khi thao tác.");
    try {
      localStorage.setItem("hg_reviewer", reviewer);
      const body = next === "IN_REVIEW" ? { status: next, reviewer } : { status: next, reviewer, decision, note: note || undefined };
      setOpen(await api.updateReview(open.id, body));
      setNote("");
      setError(null);
      load();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const p = open?.prediction;
  return (
    <>
      <h1>Hàng đợi xem xét thủ công</h1>
      <p className="sub">Ca vào hàng đợi khi nguy cơ cao hoặc dự đoán kém tin cậy (thiếu dữ liệu, ngoài phân phối, các mô hình bất đồng…). Hệ thống không tự đóng ca; mọi thao tác đều được ghi nhật ký.</p>
      <ErrorBox error={error} />
      <div className="card">
        <div className="row" style={{ marginBottom: 10 }}>
          <div className="seg">
            {STATUS_FILTERS.map((s) => (
              <button key={s} className={status === s ? "on" : ""} onClick={() => setStatus(s)}>{s || "Tất cả"}</button>
            ))}
          </div>
          <span className="spacer" />
          <button className="btn" onClick={load}>Làm mới</button>
        </div>
        {rows.length === 0 ? (
          <div className="empty">Không có ca nào ở trạng thái này.</div>
        ) : (
          <table className="t">
            <thead>
              <tr><th>#</th><th>Ưu tiên</th><th>Bài</th><th>Ca</th><th>Nguy cơ</th><th className="num">Xác suất</th><th>Lý do</th><th>Trạng thái</th><th>Tạo lúc</th></tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id} className="click" onClick={() => setParams({ open: String(r.id) })}>
                  <td>{r.id}</td>
                  <td><span className={`pill ${r.priority === 1 ? "HIGH" : r.priority === 2 ? "MEDIUM" : "gray"}`}>P{r.priority}</span></td>
                  <td>{r.task.toUpperCase()}</td>
                  <td><code>{r.case_id}</code></td>
                  <td><RiskPill level={r.risk_level} /></td>
                  <td className="num">{pct(r.calibrated_probability)}</td>
                  <td style={{ maxWidth: 320 }}>{r.trigger_reasons.map(reasonText).join(" · ")}</td>
                  <td><StatusPill status={r.status} /></td>
                  <td>{new Date(r.created_at).toLocaleString("vi-VN")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {open && (
        <div className="drawer" role="dialog" aria-label={`Ca review ${open.id}`}>
          <div className="row">
            <h2 style={{ margin: 0 }}>Ca #{open.id} · <code>{open.case_id}</code></h2>
            <span className="spacer" />
            <button className="btn" onClick={() => { setOpen(null); setParams({}); }}>Đóng</button>
          </div>
          <div className="row" style={{ margin: "10px 0" }}>
            <StatusPill status={open.status} /> <RiskPill level={open.risk_level} />
            <span>{pct(open.calibrated_probability)}</span>
            <span className="pill gray">Ưu tiên {open.priority}</span>
          </div>
          <h3>Lý do gắn cờ</h3>
          <ul className="plain">{open.trigger_reasons.map((r) => <li key={r}>{reasonText(r)}</li>)}</ul>
          {p?.input_quality?.warnings?.length ? (
            <>
              <h3>Cảnh báo dữ liệu đầu vào</h3>
              <ul className="plain">{p.input_quality.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
            </>
          ) : null}
          {p?.explanation && (
            <>
              <h3>Vì sao mô hình dự đoán như vậy</h3>
              <Waterfall items={p.explanation.top_features} />
            </>
          )}
          {NEXT[open.status] && (
            <div className="card" style={{ marginTop: 14 }}>
              <h2>Thao tác</h2>
              <div className="grid g2">
                <label className="f">Người duyệt<input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="vd: ha" /></label>
                {open.status === "IN_REVIEW" && (
                  <label className="f">Quyết định
                    <select value={decision} onChange={(e) => setDecision(e.target.value)}>
                      <option value="AGREE">Đồng ý với mô hình</option>
                      <option value="DISAGREE">Không đồng ý</option>
                      <option value="NEED_INFO">Cần thêm thông tin</option>
                    </select>
                  </label>
                )}
              </div>
              {open.status === "IN_REVIEW" && (
                <label className="f" style={{ marginTop: 8 }}>Ghi chú<textarea rows={3} value={note} onChange={(e) => setNote(e.target.value)} /></label>
              )}
              <div className="row" style={{ marginTop: 10 }}>
                {NEXT[open.status].map((s) => (
                  <button key={s} className={`btn ${s === "REVIEWED" || s === "IN_REVIEW" ? "primary" : ""}`} onClick={() => act(s)}>
                    {{ IN_REVIEW: "Nhận xem", REVIEWED: "Đóng ca (đã xem)", ESCALATED: "Chuyển cấp" }[s as string]}
                  </button>
                ))}
              </div>
            </div>
          )}
          <h3>Nhật ký</h3>
          <ul className="timeline">
            {open.audit.map((e) => (
              <li key={e.id}>
                <strong>{e.old_status ?? "–"} → {e.new_status}</strong> bởi <code>{e.actor}</code>
                {e.decision && <> · {e.decision}</>}
                <br />
                <small>{new Date(e.at).toLocaleString("vi-VN")}</small>
                {e.note && <div>{e.note}</div>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </>
  );
}

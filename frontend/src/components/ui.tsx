import { ReactNode } from "react";
import { Contribution, FeatureValue, num, PredictResult, reasonText } from "../api";

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="card stat">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {hint && <div className="hint">{hint}</div>}
    </div>
  );
}

/** Dải chỉ số: nhiều con số đặt cạnh nhau, ngăn bằng đường kẻ — đọc nhanh, so sánh được. */
export function Kpis({ items }: { items: { k: string; v: ReactNode; h?: ReactNode }[] }) {
  return (
    <div className="kpis">
      {items.map((i) => (
        <div key={i.k}>
          <div className="k">{i.k}</div>
          <div className="v">{i.v}</div>
          {i.h && <div className="h">{i.h}</div>}
        </div>
      ))}
    </div>
  );
}

export function RiskPill({ level }: { level: string }) {
  const text: Record<string, string> = { LOW: "Thấp", MEDIUM: "Trung bình", HIGH: "Cao" };
  return <span className={`pill ${level}`}>{text[level] ?? level}</span>;
}

export function StatusPill({ status }: { status: string }) {
  const text: Record<string, string> = { PENDING: "Chờ xem", IN_REVIEW: "Đang xem", REVIEWED: "Đã xem", ESCALATED: "Chuyển cấp" };
  return <span className={`pill ${status}`}>{text[status] ?? status}</span>;
}

export function ErrorBox({ error }: { error: string | null }) {
  return error ? <div className="err">{error}</div> : null;
}

const fmtValue = (v: FeatureValue, unit: string) =>
  v == null ? "không có" : typeof v === "number" ? `${Number.isInteger(v) ? v : v.toFixed(2)}${unit ? ` ${unit}` : ""}` : String(v);

/** Biểu đồ thác: mỗi dòng là đóng góp SHAP (log-odds) của một đặc trưng cho CHÍNH ca này. */
export function Waterfall({ items, max }: { items: Contribution[]; max?: number }) {
  const m = max ?? Math.max(...items.map((i) => Math.abs(i.shap)), 1e-6);
  return (
    <div>
      {items.map((c) => {
        const w = (Math.abs(c.shap) / m) * 50;
        return (
          <div className="wf-row" key={c.feature}>
            <div className="name">
              {c.label}
              <small>{c.feature.endsWith("_missing") ? (c.value === 1 ? "không làm xét nghiệm" : "có kết quả") : fmtValue(c.value, c.unit)}</small>
            </div>
            <div className="wf-bar">
              <div className="axis" />
              <div className={`bar ${c.shap > 0 ? "up" : "down"}`} style={c.shap > 0 ? { left: "50%", width: `${w}%` } : { right: "50%", width: `${w}%` }} />
            </div>
            <div className="num" style={{ color: c.shap > 0 ? "var(--high)" : "var(--low)" }}>
              {c.shap > 0 ? "+" : ""}
              {c.shap.toFixed(2)}
            </div>
          </div>
        );
      })}
      <div className="row" style={{ fontSize: 12, color: "var(--muted)", marginTop: 6 }}>
        <span>← giảm nguy cơ</span>
        <span className="spacer" />
        <span>tăng nguy cơ →</span>
      </div>
    </div>
  );
}

/** Mức tin cậy tổng hợp từ các lý do gắn cờ (không tính cờ chỉ vì nguy cơ cao). */
export function confidenceOf(r: PredictResult): { level: "CAO" | "VỪA" | "THẤP"; color: string; reasons: string[] } {
  const reasons = r.review.reasons.filter((x) => !x.startsWith("HIGH_RISK"));
  const serious = reasons.filter((x) => /LOW_CONFIDENCE|HIGH_MISSINGNESS|OUT_OF_DISTRIBUTION|OUT_OF_RANGE/.test(x));
  if (serious.length) return { level: "THẤP", color: "var(--high)", reasons };
  if (reasons.length) return { level: "VỪA", color: "var(--mid)", reasons };
  return { level: "CAO", color: "var(--low)", reasons };
}

/** Thẻ tin cậy: gom đủ các trụ "đáng tin cậy" của đề cho MỘT dự đoán. */
export function TrustCard({ r, onOpenReview }: { r: PredictResult; onOpenReview?: (id: number) => void }) {
  const conf = confidenceOf(r);
  const p = r.calibrated_probability;
  const iq = r.input_quality;
  return (
    <div className="card trust">
      <div className={`risk ${r.risk_level}`}>
        <div className="lvl">{r.task_name}</div>
        <div className="p">{(100 * p).toFixed(1)}%</div>
        <div>Mức nguy cơ: <strong>{({ LOW: "THẤP", MEDIUM: "TRUNG BÌNH", HIGH: "CAO" } as Record<string, string>)[r.risk_level]}</strong></div>
        <div className="meter" title="Vạch = ngưỡng HIGH">
          <div className="fill" style={{ width: `${Math.min(100, 100 * p)}%` }} />
          <div className="tick" style={{ left: `${Math.min(100, 100 * r.threshold)}%` }} />
        </div>
        <div style={{ fontSize: 12, opacity: 0.9 }}>Ngưỡng HIGH {num(r.threshold, 3)} · xác suất đã hiệu chuẩn</div>
        <div style={{ fontSize: 12, opacity: 0.8, marginTop: "auto" }}>Xác suất thô {num(r.probability, 3)} · độ lệch giữa 5 mô hình fold ±{num(r.uncertainty_std, 3)}</div>
      </div>
      <div className="body">
        <div className="sect">
          <h3 style={{ marginTop: 0 }}>Độ tin cậy của dự đoán</h3>
          <div className="conf">
            <span className="dot" style={{ background: conf.color }} /> {conf.level}
          </div>
          {conf.reasons.length > 0 && (
            <ul className="plain">
              {conf.reasons.map((x) => (
                <li key={x}>{reasonText(x)}</li>
              ))}
            </ul>
          )}
          {iq.warnings.length > 0 && (
            <ul className="plain" style={{ color: "var(--muted)" }}>
              {iq.warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          )}
        </div>
        {r.explanation && (
          <div className="sect">
            <h3 style={{ marginTop: 0 }}>Vì sao (5 yếu tố ảnh hưởng nhất · {r.explanation.method})</h3>
            <Waterfall items={r.explanation.top_features.slice(0, 5)} />
          </div>
        )}
        <div className="sect row">
          {r.review.flag ? (
            <>
              <span className="pill HIGH">Đã chuyển xem xét thủ công · ưu tiên {r.review.priority}</span>
              {r.review.review_id && onOpenReview && (
                <button className="btn" onClick={() => onOpenReview(r.review.review_id!)}>
                  Mở ca #{r.review.review_id}
                </button>
              )}
            </>
          ) : (
            <span className="pill gray">Không cần xem xét thủ công</span>
          )}
          <span className="spacer" />
          <small style={{ color: "var(--muted)" }}>
            Mô hình <code>{r.model_version}</code>
          </small>
        </div>
      </div>
    </div>
  );
}

/** Ô là số (vd "0.321", "4.86%", "12,345", "0.84 [0.80, 0.87]", "±0.01", "–") → căn phải, phông cố định. */
const NUMERIC = /^[-+±]?[\d.,]+(%|×| ?±.*| \[.*\])?$|^–$/;
const isNum = (c: ReactNode) => typeof c === "number" || (typeof c === "string" && NUMERIC.test(c.trim()));

export function Table({ head, rows }: { head: ReactNode[]; rows: ReactNode[][] }) {
  // cột số = cột mà mọi ô (trừ "–") đều là số → căn phải cả tiêu đề
  const numCols = head.map((_, j) => rows.length > 0 && rows.every((r) => isNum(r[j])));
  return (
    <table className="t">
      <thead>
        <tr>{head.map((h, j) => <th key={j} className={numCols[j] ? "num" : undefined}>{h}</th>)}</tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i}>{r.map((c, j) => <td key={j} className={numCols[j] || isNum(c) ? "num" : undefined}>{c}</td>)}</tr>
        ))}
      </tbody>
    </table>
  );
}

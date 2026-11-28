import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, DemoCase, FeatureValue, Field } from "../api";
import { TaskPicker, useTask } from "../components/TaskContext";
import { ErrorBox, TrustCard } from "../components/ui";

const GROUP_TEXT: Record<string, string> = {
  demographic_admission: "Nhân khẩu học và nhập viện",
  history: "Tiền sử",
  stay: "Đợt nằm viện",
  labs: "Xét nghiệm",
  vitals_icu: "Sinh hiệu ICU",
};

/** Màn hình Dự đoán (Hùng): chọn bài toán, chọn ca demo hoặc nhập tay, nhận Thẻ tin cậy. */
export default function Predict() {
  const { task, setLast, last } = useTask();
  const nav = useNavigate();
  const [fields, setFields] = useState<Field[]>([]);
  const [cases, setCases] = useState<DemoCase[]>([]);
  const [values, setValues] = useState<Record<string, FeatureValue>>({});
  const [caseId, setCaseId] = useState("demo-case");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setError(null);
    Promise.all([api.schema(task), api.cases(task)])
      .then(([s, c]) => {
        setFields(s.fields);
        setCases(c);
        setValues(c[0]?.features ?? {});
        setCaseId(c[0]?.case_id ?? "demo-case");
      })
      .catch((e) => setError(e.message));
  }, [task]);

  const groups = useMemo(() => {
    const g: Record<string, Field[]> = {};
    fields.filter((f) => !f.name.endsWith("_missing")).forEach((f) => (g[f.group] ??= []).push(f));
    return g;
  }, [fields]);

  const set = (f: Field, raw: string) => {
    const v = raw === "" ? null : f.dtype === "object" ? raw : Number(raw);
    setValues((old) => ({ ...old, [f.name]: v }));
  };

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      // không gửi cờ _missing: gói mô hình tự suy ra từ giá trị (thiếu ≠ bình thường)
      const payload = Object.fromEntries(Object.entries(values).filter(([k]) => !k.endsWith("_missing")));
      setLast(await api.predict(task, caseId, payload));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="row">
        <div>
          <h1>Dự đoán nguy cơ</h1>
          <p className="sub">Nhập thông tin bệnh nhân hoặc chọn ca demo. Kết quả gồm xác suất đã hiệu chuẩn, giải thích, độ tin cậy và cờ xem xét thủ công.</p>
        </div>
        <span className="spacer" />
        <TaskPicker />
      </div>
      <ErrorBox error={error} />
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="row">
          <strong>Ca demo:</strong>
          {cases.map((c) => (
            <button key={c.case_id} className={`btn ${caseId === c.case_id ? "primary" : ""}`} title={c.description}
              onClick={() => { setValues(c.features); setCaseId(c.case_id); }}>
              {c.title}
            </button>
          ))}
          <span className="spacer" />
          <button className="btn primary" disabled={busy || !fields.length} onClick={run}>
            {busy ? "Đang dự đoán…" : "Dự đoán"}
          </button>
        </div>
      </div>

      {last && last.task === task && (
        <div style={{ marginBottom: 18 }}>
          <TrustCard r={last} onOpenReview={(id) => nav(`/reviews?open=${id}`)} />
        </div>
      )}

      {Object.entries(groups).map(([g, fs]) => (
        <details className="group" key={g} open={g !== "vitals_icu"}>
          <summary>
            {GROUP_TEXT[g] ?? g} <span style={{ color: "var(--muted)", fontWeight: 400 }}>({fs.length} trường)</span>
          </summary>
          <div className="formgrid" style={{ marginTop: 10 }}>
            {fs.map((f) => (
              <label className="f" key={f.name} title={f.name}>
                {f.label} {f.unit && <span className="u">({f.unit})</span>}
                <input type={f.dtype === "object" ? "text" : "number"} step="any"
                  value={values[f.name] == null ? "" : String(values[f.name])}
                  placeholder={f.valid_range ? `${f.valid_range[0]}–${f.valid_range[1]}` : "trống = không có"}
                  onChange={(e) => set(f, e.target.value)} />
              </label>
            ))}
          </div>
        </details>
      ))}
    </>
  );
}

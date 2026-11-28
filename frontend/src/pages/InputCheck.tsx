import { useEffect, useState } from "react";
import { api, DemoCase, InputQuality } from "../api";
import { TaskPicker, useTask } from "../components/TaskContext";
import { ErrorBox } from "../components/ui";

/** Màn hình Kiểm tra đầu vào (Chiến): phát hiện trường thiếu, giá trị ngoài khoảng sinh lý, lỗi đơn vị
 *  TRƯỚC khi dự đoán. Dán JSON từ hệ thống khác hoặc chọn ca demo. */
export default function InputCheck() {
  const { task } = useTask();
  const [cases, setCases] = useState<DemoCase[]>([]);
  const [text, setText] = useState("{}");
  const [res, setRes] = useState<InputQuality | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.cases(task).then((c) => {
      setCases(c);
      setText(JSON.stringify(c[0]?.features ?? {}, null, 2));
    }).catch((e) => setError(e.message));
    setRes(null);
  }, [task]);

  const check = async () => {
    setError(null);
    try {
      const features = JSON.parse(text);
      setRes((await api.validate(task, features)).input_quality);
    } catch (e) {
      setError(e instanceof SyntaxError ? "JSON không hợp lệ" : (e as Error).message);
    }
  };

  const demoBad = () => {
    const f = { ...(cases[0]?.features ?? {}) } as Record<string, unknown>;
    f.sodium_first = 1390; // nhập nhầm đơn vị / thừa số 0
    f.creatinine_first = null;
    f.age = 130;
    setText(JSON.stringify(f, null, 2));
  };

  return (
    <>
      <div className="row">
        <div>
          <h1>Kiểm tra chất lượng đầu vào</h1>
          <p className="sub">Dữ liệu đầu vào sai thì dự đoán vô nghĩa. Màn hình này kiểm tra trường thiếu, giá trị ngoài khoảng sinh lý và cảnh báo độ tin cậy trước khi gọi mô hình.</p>
        </div>
        <span className="spacer" />
        <TaskPicker />
      </div>
      <ErrorBox error={error} />
      <div className="grid g2">
        <div className="card">
          <div className="row" style={{ marginBottom: 8 }}>
            <h2 style={{ margin: 0 }}>Dữ liệu (JSON)</h2>
            <span className="spacer" />
            {cases.map((c) => <button key={c.case_id} className="btn" onClick={() => setText(JSON.stringify(c.features, null, 2))}>{c.title}</button>)}
            <button className="btn" onClick={demoBad}>Ca nhập sai</button>
          </div>
          <textarea rows={22} style={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }} value={text} onChange={(e) => setText(e.target.value)} />
          <div className="row" style={{ marginTop: 10 }}>
            <span className="spacer" />
            <button className="btn primary" onClick={check}>Kiểm tra</button>
          </div>
        </div>
        <div className="card">
          <h2>Kết quả</h2>
          {!res ? <div className="empty">Bấm "Kiểm tra" để xem kết quả.</div> : (
            <>
              <div className="grid g3" style={{ marginBottom: 12 }}>
                <div className="stat"><div className="label">Trường thiếu</div><div className="value">{res.missing_fields.length}</div></div>
                <div className="stat"><div className="label">Ngoài khoảng</div><div className="value" style={{ color: res.out_of_range_fields.length ? "var(--high)" : undefined }}>{res.out_of_range_fields.length}</div></div>
                <div className="stat"><div className="label">Cảnh báo</div><div className="value">{res.warnings.length}</div></div>
              </div>
              {res.out_of_range_fields.length > 0 && (
                <div className="err">Giá trị ngoài khoảng sinh lý (mô hình sẽ coi là thiếu và gắn cờ review): {res.out_of_range_fields.join(", ")}</div>
              )}
              {res.warnings.length > 0 && <><h3>Cảnh báo</h3><ul className="plain">{res.warnings.map((w) => <li key={w}>{w}</li>)}</ul></>}
              <h3>Trường thiếu ({res.missing_fields.length})</h3>
              <p style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.7 }}>{res.missing_fields.join(", ") || "Không có"}</p>
            </>
          )}
        </div>
      </div>
    </>
  );
}

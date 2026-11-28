import { NavLink, Route, Routes } from "react-router-dom";
import { TaskProvider } from "./components/TaskContext";
import Dashboard from "./pages/Dashboard";
import Predict from "./pages/Predict";
import Explain from "./pages/Explain";
import ReviewQueue from "./pages/ReviewQueue";
import InputCheck from "./pages/InputCheck";
import Evaluation from "./pages/Evaluation";
import Drift from "./pages/Drift";
import History from "./pages/History";

const NAV = [
  { to: "/", label: "Tổng quan", owner: "Chưởng" },
  { to: "/predict", label: "Dự đoán", owner: "Hùng" },
  { to: "/explain", label: "Giải thích", owner: "Khoẻ" },
  { to: "/reviews", label: "Hàng đợi review", owner: "Hà" },
  { to: "/input-check", label: "Kiểm tra đầu vào", owner: "Chiến" },
  { to: "/evaluation", label: "Đánh giá mô hình", owner: "Chưởng" },
  { to: "/drift", label: "Giám sát drift", owner: "Chưởng" },
  { to: "/history", label: "Lịch sử", owner: "Chưởng" },
];

export default function App() {
  return (
    <TaskProvider>
      <div className="layout">
        <aside className="sidebar">
          <div className="brand">
            <img src="/heart.svg" alt="" />
            <div>
              HeartGuard AI
              <small>Nguy cơ suy tim đáng tin cậy</small>
            </div>
          </div>
          <nav className="nav">
            {NAV.map((n) => (
              <NavLink key={n.to} to={n.to} end={n.to === "/"}>
                {n.label}
                <span className="owner">{n.owner}</span>
              </NavLink>
            ))}
          </nav>
          <div className="foot">
            Nhóm 11 · Big Data · MIMIC-IV v3.1
            <br />
            Nguyên mẫu nghiên cứu, không dùng cho quyết định lâm sàng.
          </div>
        </aside>
        <main className="main">
          <div className="banner">Research prototype – not for clinical use. Chỉ dùng ca demo đã khử định danh.</div>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/explain" element={<Explain />} />
            <Route path="/reviews" element={<ReviewQueue />} />
            <Route path="/input-check" element={<InputCheck />} />
            <Route path="/evaluation" element={<Evaluation />} />
            <Route path="/drift" element={<Drift />} />
            <Route path="/history" element={<History />} />
          </Routes>
        </main>
      </div>
    </TaskProvider>
  );
}

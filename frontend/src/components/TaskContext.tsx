import { createContext, ReactNode, useContext, useEffect, useState } from "react";
import { api, PredictResult, Task, TaskInfo } from "../api";

interface Ctx {
  task: Task;
  setTask: (t: Task) => void;
  tasks: TaskInfo[];
  last: PredictResult | null;
  setLast: (r: PredictResult | null) => void;
  error: string | null;
}

const TaskCtx = createContext<Ctx | null>(null);

export function TaskProvider({ children }: { children: ReactNode }) {
  const [task, setTask] = useState<Task>("p1");
  const [tasks, setTasks] = useState<TaskInfo[]>([]);
  const [last, setLast] = useState<PredictResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api
      .tasks()
      .then((ts) => {
        setTasks(ts);
        const first = ts.find((t) => t.enabled);
        if (first) setTask(first.task);
      })
      .catch((e) => setError(`Không kết nối được API: ${e.message}`));
  }, []);
  return <TaskCtx.Provider value={{ task, setTask, tasks, last, setLast, error }}>{children}</TaskCtx.Provider>;
}

export function useTask() {
  const c = useContext(TaskCtx);
  if (!c) throw new Error("useTask ngoài TaskProvider");
  return c;
}

export function TaskPicker() {
  const { task, setTask, tasks } = useTask();
  return (
    <div className="seg" role="tablist" aria-label="Chọn bài toán">
      {tasks.map((t) => (
        <button key={t.task} className={t.task === task ? "on" : ""} disabled={!t.enabled} onClick={() => setTask(t.task)} title={t.error ?? t.outcome}>
          {t.task.toUpperCase()} · {t.name}
        </button>
      ))}
    </div>
  );
}

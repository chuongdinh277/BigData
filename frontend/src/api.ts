// Lớp gọi API duy nhất của frontend. Mọi URL đi qua /api/v1 (proxy Vite khi dev, nginx khi triển khai).

export type Task = "p1" | "p2" | "p3";
export type RiskLevel = "LOW" | "MEDIUM" | "HIGH";
export type ReviewStatus = "PENDING" | "IN_REVIEW" | "REVIEWED" | "ESCALATED";
export type FeatureValue = number | string | null;

export interface TaskInfo {
  task: Task;
  name: string;
  prediction_time: string;
  outcome: string;
  enabled: boolean;
  model_version: string | null;
  thresholds: { low_medium: number; medium_high: number } | null;
  error?: string | null;
}

export interface Field {
  name: string;
  dtype: string;
  group: string;
  label: string;
  unit: string;
  valid_range: [number, number] | null;
}

export interface DemoCase {
  case_id: string;
  title: string;
  description: string;
  features: Record<string, FeatureValue>;
}

export interface Contribution {
  feature: string;
  label: string;
  unit: string;
  value: FeatureValue;
  shap: number;
  direction: string;
}

export interface Explanation {
  method: string;
  scale: string;
  base_value: number;
  top_features: Contribution[];
}

export interface InputQuality {
  missing_fields: string[];
  out_of_range_fields: string[];
  warnings: string[];
}

export interface PredictResult {
  task: Task;
  task_name: string;
  case_id: string;
  prediction_id: number;
  model_version: string;
  prediction_time: string;
  probability: number;
  calibrated_probability: number;
  uncertainty_std: number;
  threshold: number;
  risk_level: RiskLevel;
  review: { flag: boolean; reasons: string[]; priority: number; review_id: number | null };
  input_quality: InputQuality;
  explanation?: Explanation;
  disclaimer: string;
}

export interface Audit {
  id: number;
  old_status: string | null;
  new_status: string;
  actor: string;
  decision: string | null;
  note: string | null;
  at: string;
}

export interface Review {
  id: number;
  prediction_id: number;
  case_id: string;
  task: Task;
  risk_level: RiskLevel;
  calibrated_probability: number;
  trigger_reasons: string[];
  priority: number;
  status: ReviewStatus;
  reviewer: string | null;
  decision: string | null;
  note: string | null;
  created_at: string;
  updated_at: string;
  prediction: (PredictResult & { features: Record<string, FeatureValue> }) | null;
  audit: Audit[];
}

export interface PredictionRow {
  id: number;
  case_id: string;
  task: Task;
  model_version: string;
  calibrated_probability: number;
  risk_level: RiskLevel;
  review_flag: boolean;
  created_at: string;
}

export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api/v1${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(res.status, body?.detail ?? res.statusText);
  return body as T;
}

const post = <T>(path: string, data: unknown) => call<T>(path, { method: "POST", body: JSON.stringify(data) });

export const api = {
  health: () => call<{ status: string; models_loaded: Record<string, string>; model_errors: Record<string, string> }>("/health"),
  tasks: () => call<TaskInfo[]>("/tasks"),
  schema: (t: Task) => call<{ task: Task; model_version: string; fields: Field[] }>(`/schema/${t}`),
  cases: (t: Task) => call<DemoCase[]>(`/cases/${t}`),
  predict: (t: Task, case_id: string, features: Record<string, FeatureValue>) =>
    post<PredictResult>(`/predict/${t}`, { case_id, features, options: { explain: true, top_k: 10 } }),
  validate: (t: Task, features: Record<string, FeatureValue>) =>
    post<{ input_quality: InputQuality; model_version: string }>(`/validate/${t}`, { features }),
  predictions: (t?: Task) => call<PredictionRow[]>(`/predictions${t ? `?task=${t}` : ""}`),
  prediction: (id: number) =>
    call<{ id: number; case_id: string; task: Task; features: Record<string, FeatureValue>; result: PredictResult; review_id: number | null }>(
      `/predictions/${id}`,
    ),
  reviews: (status?: string, task?: string) => {
    const q = new URLSearchParams();
    if (status) q.set("status", status);
    if (task) q.set("task", task);
    return call<Review[]>(`/reviews?${q}`);
  },
  reviewStats: () => call<{ by_status: Record<string, number>; open_by_priority: Record<string, number> }>("/reviews/stats"),
  review: (id: number) => call<Review>(`/reviews/${id}`),
  updateReview: (id: number, body: { status: ReviewStatus; reviewer: string; decision?: string; note?: string }) =>
    call<Review>(`/reviews/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  report: (t: Task, kind: string) => call<Record<string, any>>(`/reports/${t}/${kind}`),
  figure: (t: Task, name: string) => `/api/v1/reports/${t}/figures/${name}`,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  drift: (t: Task, records: Record<string, FeatureValue>[]) => post<Record<string, any>>(`/drift/${t}`, { records }),
};

export const pct = (v: number | null | undefined, d = 1) => (v == null || Number.isNaN(v) ? "–" : `${(100 * v).toFixed(d)}%`);
export const num = (v: number | null | undefined, d = 3) => (v == null || Number.isNaN(v) ? "–" : v.toFixed(d));

export const REASON_TEXT: Record<string, string> = {
  HIGH_RISK: "Nguy cơ cao",
  NEAR_THRESHOLD: "Sát ngưỡng cảnh báo",
  LOW_CONFIDENCE: "Các mô hình bất đồng",
  MISSING_KEY_LAB: "Thiếu xét nghiệm chính",
  HIGH_MISSINGNESS: "Thiếu nhiều dữ liệu",
  OUT_OF_RANGE: "Giá trị ngoài khoảng sinh lý",
  OUT_OF_DISTRIBUTION: "Khác dữ liệu huấn luyện",
};

export const reasonText = (r: string) => {
  const [k, detail] = r.split(":");
  return REASON_TEXT[k] ? `${REASON_TEXT[k]}${detail ? `: ${detail}` : ""}` : r;
};

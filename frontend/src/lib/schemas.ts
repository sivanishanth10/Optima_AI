export type TabId = "diagnostics" | "plan" | "dashboard" | "metrics" | "code" | "kb";

export type UserRole = "beginner" | "intermediate" | "senior";
export type UserPurpose = "cleaning" | "other";
export interface UserPreferences {
  role: UserRole;
  purpose: UserPurpose;
}
// ─── Shared ────────────────────────────────────────────────────────────────────
export interface DatasetFingerprint {
  shape: [number, number];
  columns: string[];
  dtypes: Record<string, string>;
  total_nulls: number;
  duplicate_rows: number;
  safe_sample: any[];
  safe_summary?: string;
}

// ─── Upload ───────────────────────────────────────────────────────────────────
export interface UploadResponse {
  file_id: string;
  file_path: string;
  filename: string;
  rows: number;
  cols: number;
  size_bytes: number;
  columns: string[];
  dtypes: Record<string, string>;
}

// ─── Analyze Init ─────────────────────────────────────────────────────────────
export interface AnalyzeInitResponse {
  file_id: string;
  file_path: string;
  preprocessed_path?: string;
  fingerprint: DatasetFingerprint;
  safe_summary: string;
}

// ─── Diagnose ─────────────────────────────────────────────────────────────────
export interface DiagnoseResponse {
  report: string;
  model_used: string;
}

// ─── Clean ────────────────────────────────────────────────────────────────────
export interface CleanResponse {
  message: string;
  cleaning_report: string; // filename of script
  explanation: string;
  plan: any;
  python_code: string;
  used_kb?: boolean;
  cleaned_data: {
    file_path: string;
    shape: [number, number];
    fingerprint: DatasetFingerprint;
  };
  quality_metrics?: {
    initial: any;
    cleaned: any;
  };
  cleaning_summary?: string;
}

// ─── Metrics ──────────────────────────────────────────────────────────────────
export type MetricsTask = "classification" | "regression" | "clustering";

export interface MetricsResponse {
  task: MetricsTask;
  model: string;
  target?: string;
  // Classification
  accuracy?: number;
  f1?: number;
  precision?: number;
  recall?: number;
  // Regression
  mae?: number;
  mse?: number;
  r2?: number;
  // Common Visuals
  feature_importance?: { feature: string; value: number }[];
  prediction_preview?: { id: number; actual: any; predicted: any }[];
  // Clustering
  cluster_count?: number;
  plot_data?: { x: number; y: number; cluster: number }[];
  silhouette_score?: number;
}

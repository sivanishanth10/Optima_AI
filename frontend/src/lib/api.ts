/**
 * api.ts — Typed API client for all Optima backend endpoints.
 */

import type {
  UploadResponse,
  AnalyzeInitResponse,
  DiagnoseResponse,
  CleanResponse,
  MetricsResponse,
  DatasetFingerprint,
} from "./schemas";

/** 
 * BASE_URL: Dynamic resolution to support both localhost and network IP access. 
 * If accessed via http://10.x.x.x:3000, it hits http://10.x.x.x:8000.
 */
const getBaseUrl = (): string => {
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, ""); // Remove trailing slash if present
  }
  if (typeof window !== "undefined") {
    // Favor the current hostname so network IP access works out-of-the-box
    return `http://${window.location.hostname}:8000`;
  }
  return "http://localhost:8000";
};

const BASE_URL = getBaseUrl();

/** Gets common headers including the Supabase Auth token */
const getHeaders = (isMultipart = false) => {
  const headers: Record<string, string> = {};
  if (!isMultipart) {
    headers["Content-Type"] = "application/json";
  }
  
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("optima_token");
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
  }
  return headers;
};

/** Shared error handler: reads .detail from FastAPI error responses */
async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // ignore JSON parse errors on error responses
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

// ─── Health ──────────────────────────────────────────────────────────────────
/** GET / — check if backend is online */
export async function checkHealth(): Promise<{ status: string }> {
  const res = await fetch(`${BASE_URL}/`, { headers: getHeaders() });
  return handleResponse<{ status: string }>(res);
}

// ─── Upload ───────────────────────────────────────────────────────────────────
/** POST /api/upload — stores the file server-side; returns file_path */
export async function uploadFile(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch(`${BASE_URL}/api/upload`, { 
      method: "POST", 
      body: form,
      headers: getHeaders(true) // multipart
    });
    return handleResponse<UploadResponse>(res);
  } catch (err: any) {
    if (err.message === "Failed to fetch") {
      throw new Error(`Cannot connect to backend at ${BASE_URL}. Is it running?`);
    }
    throw err;
  }
}

// ─── Analyze Init ─────────────────────────────────────────────────────────────
/** POST /api/analyze-init — computes fingerprint + PII-redacted sample */
export async function analyzeInit(filePath: string): Promise<AnalyzeInitResponse> {
  const res = await fetch(`${BASE_URL}/api/analyze-init`, {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify({ file_path: filePath }),
  });
  return handleResponse<AnalyzeInitResponse>(res);
}

// ─── Diagnose ─────────────────────────────────────────────────────────────────
/** POST /api/diagnose — generates AI health report from fingerprint */
export async function diagnose(
  fingerprint: DatasetFingerprint,
  model?: string,
  apiKey?: string
): Promise<DiagnoseResponse> {
  const res = await fetch(`${BASE_URL}/api/diagnose`, {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify({ fingerprint, model, api_key: apiKey }),
  });
  return handleResponse<DiagnoseResponse>(res);
}

// ─── Analyze (Plan only, no execution) ───────────────────────────────────────
/** POST /api/analyze — ask AI for a cleaning plan WITHOUT executing it */
export async function analyzePlan(
  fingerprint: DatasetFingerprint,
  model?: string,
  apiKey?: string
): Promise<{ plan: any; explanation: string }> {
  const res = await fetch(`${BASE_URL}/api/analyze`, {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify({ fingerprint, model, api_key: apiKey }),
  });
  return handleResponse<{ plan: any; explanation: string }>(res);
}

// ─── Clean ────────────────────────────────────────────────────────────────────
/** POST /api/clean — executes a cleaning plan on the dataset. */
export async function cleanDataset(
  filePath: string,
  fingerprint: DatasetFingerprint,
  model?: string,
  apiKey?: string,
  plan?: any,
  enabledActions?: string[]
): Promise<CleanResponse> {
  const res = await fetch(`${BASE_URL}/api/clean`, {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify({
      file_path: filePath,
      fingerprint,
      model,
      api_key: apiKey,
      ...(plan ? { plan } : {}),
      ...(enabledActions ? { enabled_actions: enabledActions } : {}),
    }),
  });
  return handleResponse<CleanResponse>(res);
}

// ─── Chat ─────────────────────────────────────────────────────────────────────
/** POST /api/chat — conversational Q&A with dataset context */
export async function chat(
  prompt: string,
  datasetState: string,
  dataInfo: Record<string, unknown>,
  safeSummary: string,
  model?: string,
  apiKey?: string,
  filePath?: string
): Promise<{ reply: string }> {
  try {
    const res = await fetch(`${BASE_URL}/api/chat`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({
        prompt,
        dataset_state: datasetState,
        data_info: dataInfo,
        safe_summary: safeSummary,
        model,
        api_key: apiKey,
        file_path: filePath,
      }),
    });
    return handleResponse<{ reply: string }>(res);
  } catch (err: any) {
    if (err.message === "Failed to fetch") {
      throw new Error(`Connection failed. Ensure backend is running at ${BASE_URL}.`);
    }
    throw err;
  }
}

// ─── Metrics ──────────────────────────────────────────────────────────────────
/** POST /api/metrics — computes F1/accuracy or MAE/R2 when target is provided */
export async function computeMetrics(
  filePath: string,
  targetColumn: string,
  task: "classification" | "regression",
  model?: string
): Promise<MetricsResponse> {
  const res = await fetch(`${BASE_URL}/api/metrics`, {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify({ file_path: filePath, target_column: targetColumn, task, model }),
  });
  return handleResponse<MetricsResponse>(res);
}

// ─── Quality Metrics ──────────────────────────────────────────────────────────
export async function getQualityMetrics(filePath: string): Promise<any> {
  const res = await fetch(`${BASE_URL}/api/quality-metrics`, {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify({ file_path: filePath }),
  });
  return handleResponse<any>(res);
}

// ─── Downloads ────────────────────────────────────────────────────────────────
/** Returns the direct URL for downloading the cleaned CSV */
export function getCleanedCsvUrl(filename: string): string {
  return `${BASE_URL}/api/downloads/${filename}`;
}

/** Returns the direct URL for a report download (Python script or text report) */
export function getReportDownloadUrl(filename: string): string {
  return `${BASE_URL}/api/download/report/${filename}`;
}

// ─── Knowledge Base ──────────────────────────────────────────────────────────
export async function learnPattern(
  columns: string[],
  fixCode: string,
  description: string,
  sourceFile: string
): Promise<{ message: string; pattern_id: string }> {
  const res = await fetch(`${BASE_URL}/api/learn`, {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify({
      columns,
      fix_code: fixCode,
      description,
      source_file: sourceFile
    }),
  });
  return handleResponse<{ message: string; pattern_id: string }>(res);
}

export async function getPatterns(): Promise<{ verified_patterns: any[]; staged_patterns: any[] }> {
  const res = await fetch(`${BASE_URL}/api/patterns`, { headers: getHeaders() });
  return handleResponse<{ verified_patterns: any[]; staged_patterns: any[] }>(res);
}

export async function verifyPattern(patternId: string): Promise<{ message: string }> {
  const res = await fetch(`${BASE_URL}/api/verify`, {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify({ pattern_id: patternId }),
  });
  return handleResponse<{ message: string }>(res);
}


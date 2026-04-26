import type {
  DesignRequest,
  ExampleDesign,
  InferenceResponse,
  JobResult,
  ModelConfig,
  ModelEntry,
  ValidationResult,
} from "./types";

export const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        message = payload.detail;
      }
    } catch {
      // Keep the HTTP status message.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export async function getModels(): Promise<ModelEntry[]> {
  const payload = await requestJson<{ models: ModelEntry[] }>("/api/models");
  return payload.models;
}

export function getModelConfig(modelId: string): Promise<ModelConfig> {
  return requestJson<ModelConfig>(`/api/models/${encodeURIComponent(modelId)}/config`);
}

export async function getExampleDesigns(): Promise<ExampleDesign[]> {
  const payload = await requestJson<{ examples: ExampleDesign[] }>("/api/example-designs");
  return payload.examples;
}

export function validateDesign(request: DesignRequest): Promise<ValidationResult> {
  return requestJson<ValidationResult>("/api/design/validate", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export function runInference(request: DesignRequest): Promise<InferenceResponse> {
  return requestJson<InferenceResponse>("/api/infer", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export function getJobResult(jobId: string): Promise<JobResult> {
  return requestJson<JobResult>(`/api/jobs/${encodeURIComponent(jobId)}/result`);
}

export function frameUrl(jobId: string, field: string, frameId: number | string): string {
  const normalized = typeof frameId === "number" ? String(frameId).padStart(3, "0") : frameId;
  return `${API_BASE}/api/jobs/${encodeURIComponent(jobId)}/frames/${encodeURIComponent(field)}/${normalized}`;
}

import type {
  DesignRequest,
  ForwardSimulationRequest,
  ForwardSimulationResponse,
  ForwardSimulationResult,
  InferenceResponse,
  InverseModelEntry,
  InverseCandidate,
  InverseResult,
  InverseRunRequest,
  InverseRunResponse,
  JobResult,
  KpiInfo,
  ModelConfig,
  ModelEntry,
  ReferenceCase,
  TargetPreset,
  ValidationResult,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8001";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) message = payload.detail;
    } catch {
      // Keep the HTTP status message.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export function apiUrl(path: string): string {
  return path.startsWith("http") ? path : `${API_BASE}${path}`;
}

export async function getModels(): Promise<ModelEntry[]> {
  const payload = await requestJson<{ models: ModelEntry[] }>("/api/models");
  return payload.models;
}

export async function getModelConfig(modelId: string): Promise<ModelConfig> {
  return requestJson<ModelConfig>(`/api/models/${encodeURIComponent(modelId)}/config`);
}

export async function getReferenceCases(split = "test", limit = 80): Promise<ReferenceCase[]> {
  const params = new URLSearchParams({ split, limit: String(limit) });
  const payload = await requestJson<{ cases: ReferenceCase[] }>(`/api/reference-cases?${params}`);
  return payload.cases;
}

export async function validateDesign(request: DesignRequest): Promise<ValidationResult> {
  return requestJson<ValidationResult>("/api/design/validate", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function runInference(request: DesignRequest): Promise<InferenceResponse> {
  return requestJson<InferenceResponse>("/api/infer", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function getJobResult(jobId: string): Promise<JobResult> {
  return requestJson<JobResult>(`/api/jobs/${encodeURIComponent(jobId)}/result`);
}

export async function runForwardSimulation(request: ForwardSimulationRequest): Promise<ForwardSimulationResponse> {
  return requestJson<ForwardSimulationResponse>("/api/simulate-forward", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function getForwardSimulationStatus(jobId: string): Promise<Record<string, unknown>> {
  return requestJson<Record<string, unknown>>(`/api/simulate-forward/jobs/${encodeURIComponent(jobId)}`);
}

export async function getForwardSimulationResult(jobId: string): Promise<ForwardSimulationResult> {
  return requestJson<ForwardSimulationResult>(`/api/simulate-forward/jobs/${encodeURIComponent(jobId)}/result`);
}

export async function getInverseModels(): Promise<InverseModelEntry[]> {
  const payload = await requestJson<{ models: InverseModelEntry[] }>("/api/inverse/models");
  return payload.models;
}

export async function getTargetPresets(): Promise<TargetPreset[]> {
  const payload = await requestJson<{ presets: TargetPreset[] }>("/api/inverse/target-presets");
  return payload.presets;
}

export async function getInverseKpis(): Promise<KpiInfo[]> {
  const payload = await requestJson<{ kpis: KpiInfo[] }>("/api/inverse/kpis");
  return payload.kpis;
}

export async function runInverse(request: InverseRunRequest): Promise<InverseRunResponse> {
  return requestJson<InverseRunResponse>("/api/inverse/run", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function getInverseStatus(jobId: string): Promise<Record<string, unknown>> {
  return requestJson<Record<string, unknown>>(`/api/inverse/jobs/${encodeURIComponent(jobId)}`);
}

export async function getInverseResult(jobId: string): Promise<InverseResult> {
  return requestJson<InverseResult>(`/api/inverse/jobs/${encodeURIComponent(jobId)}/result`);
}

export async function getInverseCandidates(jobId: string): Promise<{ target: Record<string, unknown>; candidates: InverseCandidate[] }> {
  return requestJson<{ target: Record<string, unknown>; candidates: InverseCandidate[] }>(`/api/inverse/jobs/${encodeURIComponent(jobId)}/candidates`);
}

export async function getInverseDebugFiles(jobId: string): Promise<{ files: Array<{ path: string; size: number; url: string | null }> }> {
  return requestJson<{ files: Array<{ path: string; size: number; url: string | null }> }>(`/api/inverse/jobs/${encodeURIComponent(jobId)}/debug-files`);
}

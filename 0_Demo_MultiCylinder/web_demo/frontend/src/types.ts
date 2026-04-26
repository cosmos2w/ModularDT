export type Mode = "deterministic" | "generative";
export type FieldName = "omega" | "u" | "v" | "p";

export interface Cylinder {
  x: number;
  y: number;
}

export interface GenerativeOptions {
  num_samples: number;
  n_steps: number;
  seed: number | null;
  noise_mode: string;
}

export interface DesignRequest {
  model_id: string;
  mode: Mode;
  re: number;
  cylinders: Cylinder[];
  phase_bins: number;
  resolution_nx: number;
  resolution_ny: number;
  field: FieldName;
  display_smoothing: boolean;
  display_scale: number;
  render_interpolation: "nearest" | "bilinear" | "bicubic" | "lanczos";
  return_hypergraph: boolean;
  return_kpis: boolean;
  generative: GenerativeOptions;
}

export interface ModelEntry {
  id: string;
  label: string;
  mode: Mode;
  enabled: boolean;
  available: boolean;
  preload: boolean;
  stage: number | null;
  run_dir: string;
  checkpoint_path: string;
  config_path: string;
  checkpoint_exists: boolean;
  config_exists: boolean;
  missing_files: string[];
  reason_unavailable: string | null;
  note: string | null;
  status: string;
  error: string | null;
  metadata: Record<string, unknown>;
}

export interface ExampleDesign {
  name: string;
  re: number;
  cylinders: Cylinder[];
}

export interface ModelConfig {
  max_num_cylinders: number;
  domain_length_x: number;
  domain_length_y: number;
  re_scale?: number;
  default_phase_bins: number;
  max_phase_bins: number;
  phase_bin_policy: "cap" | "reject";
  phase_bin_source: string;
  expected_re_min?: number | null;
  expected_re_max?: number | null;
  fields: FieldName[];
  mode: Mode;
  stage?: number | null;
}

export interface ValidationResult {
  valid: boolean;
  warnings: string[];
  max_num_cylinders: number;
  domain_length_x: number;
  domain_length_y: number;
  requested_phase_bins: number;
  effective_phase_bins: number;
  max_phase_bins: number;
  phase_bin_policy: "cap" | "reject";
}

export interface InferenceResponse {
  job_id: string;
  status: string;
  result_url: string;
}

export interface HypergraphCylinder {
  id: string;
  x: number;
  y: number;
}

export interface EnvToken {
  id: number;
  x: number;
  y: number;
  group: number | null;
  confidence: number | null;
}

export interface Hyperedge {
  id: string;
  strength: number | null;
  source: { x: number; y: number } | null;
  wake: { x: number; y: number } | null;
  axis: { x: number; y: number } | null;
  top_cylinders: Array<{ id: string; weight: number }>;
}

export interface HypergraphLink {
  source: string;
  target: string;
  type: string;
  weight: number;
}

export interface Hypergraph {
  cylinders: HypergraphCylinder[];
  env_tokens: EnvToken[];
  hyperedges: Hyperedge[];
  links: HypergraphLink[];
}

export interface KpiData {
  mean_abs_omega?: number[];
  enstrophy?: number[];
  max_abs_omega?: number[];
  kinetic_energy?: number[];
  pressure_range?: number[];
  field_mean?: Record<string, number[]>;
  field_max_abs?: Record<string, number[]>;
  [key: string]: number[] | Record<string, number[]> | undefined;
}

export interface JobResult {
  job_id: string;
  status: string;
  model: ModelEntry;
  validation: ValidationResult;
  domain: {
    length_x: number;
    length_y: number;
    resolution_nx: number;
    resolution_ny: number;
    phase_bins: number;
    requested_phase_bins?: number;
    effective_phase_bins?: number;
    max_phase_bins?: number;
  };
  fields: FieldName[];
  frame_urls: Record<FieldName, string[]>;
  render: {
    frame_count: number;
    fields: Record<FieldName, { vmin: number; vmax: number; frames: string[] }>;
    raw_resolution?: { nx: number; ny: number };
    display_resolution?: { width: number; height: number };
    display_smoothing?: boolean;
    display_scale?: number;
    render_interpolation?: "nearest" | "bilinear" | "bicubic" | "lanczos";
    kpi_source?: string;
    note?: string;
  };
  rendering?: {
    raw_resolution: { nx: number; ny: number };
    display_resolution: { width: number; height: number };
    display_smoothing: boolean;
    render_interpolation: "nearest" | "bilinear" | "bicubic" | "lanczos";
    kpi_source: string;
    note?: string;
  };
  kpis: KpiData | null;
  hypergraph: Hypergraph | null;
  export_npz_url: string;
}

export interface ApiErrorPayload {
  detail?: string;
}

export type DemoMode = "forward" | "inverse";
export type FieldName = "temperature" | "u" | "v" | "p" | "omega";
export type KpiMode = "exact" | "range" | "max" | "min";

export interface ThermalModule {
  x: number;
  y: number;
  heat_power: number;
}

export interface ModelEntry {
  id: string;
  label: string;
  enabled: boolean;
  available: boolean;
  run_dir: string;
  checkpoint_path: string;
  config_path: string;
  checkpoint_exists: boolean;
  config_exists: boolean;
  missing_files: string[];
  reason_unavailable: string | null;
  status: string;
  metadata: Record<string, unknown>;
}

export interface ModelConfig {
  max_num_modules: number;
  domain_length_x: number;
  domain_length_y: number;
  module_radius: number;
  field_names: FieldName[];
  default_reference_split: string;
  default_reference_case_index: number;
  heat_power_min: number;
  heat_power_max: number;
  default_heat_power: number;
  reference_case: {
    case_id: string;
    num_modules: number;
    re: number;
    u_in: number;
  };
  model: ModelEntry;
}

export interface ReferenceCase {
  index: number;
  case_id: string;
  split: string;
  num_modules: number;
  total_heat_power: number;
  re: number;
  u_in: number;
  domain_length_x: number;
  domain_length_y: number;
  module_radius: number;
  modules: ThermalModule[];
}

export interface DesignRequest {
  model_id: string;
  reference_split: string;
  reference_case_index: number;
  reference_case_id: string | null;
  re: number | null;
  u_in: number | null;
  modules: ThermalModule[];
  field: FieldName;
  display_scale: number;
  display_smoothing: boolean;
  render_interpolation: "nearest" | "bilinear" | "bicubic" | "lanczos";
  return_kpis: boolean;
  return_organization: boolean;
}

export interface ValidationResult {
  valid: boolean;
  warnings: string[];
  max_num_modules: number;
  domain_length_x: number;
  domain_length_y: number;
  module_radius: number;
  min_center_distance: number;
  heat_power_min: number;
  heat_power_max: number;
  total_heat_power: number;
}

export interface InferenceResponse {
  job_id: string;
  status: string;
  result_url: string;
}

export interface JobResult {
  job_id: string;
  status: string;
  model: ModelEntry;
  validation: ValidationResult;
  reference_case: {
    case_id: string;
    split: string;
    re: number;
    u_in: number;
  };
  domain: {
    length_x: number;
    length_y: number;
    module_radius: number;
    resolution_nx: number;
    resolution_ny: number;
  };
  fields: FieldName[];
  frame_urls: Record<string, string[]>;
  render: {
    frame_count: number;
    fields: Record<string, { vmin: number; vmax: number; frames: string[] }>;
    raw_resolution: { nx: number; ny: number };
    display_resolution: { width: number; height: number };
  };
  kpis: Record<string, unknown> | null;
  modules: ThermalModule[];
  artifacts: Record<string, string>;
  export_npz_url: string;
}

export interface KpiTargetSpec {
  enabled: boolean;
  name: string;
  mode: KpiMode;
  value: number | null;
  low: number | null;
  high: number | null;
  weight: number;
}

export interface KpiInfo {
  name: string;
  label: string;
  default_mode: KpiMode;
  default_weight: number;
}

export interface InverseModelEntry {
  id: string;
  label: string;
  enabled: boolean;
  available: boolean;
  run_dir: string;
  checkpoint_name: string;
  checkpoint_path: string;
  config_name: string;
  config_path: string;
  checkpoint_exists: boolean;
  config_exists: boolean;
  missing_files: string[];
  reason_unavailable: string | null;
  status: string;
  default_forward_model_id: string | null;
  metadata: Record<string, unknown>;
}

export interface InverseConstraintSpec {
  num_modules_min: number;
  num_modules_max: number;
  min_center_distance: number;
  wall_clearance: number;
  inlet_clearance: number;
  outlet_clearance: number;
  heat_power_total: number | null;
}

export interface InverseSamplingSpec {
  n_samples: number;
  n_steps: number;
  seed: number;
  count_mode: "uniform" | "sample" | "argmax";
}

export interface InverseRunRequest {
  inverse_model_id: string;
  forward_model_id: string;
  target_name: string | null;
  kpis: KpiTargetSpec[];
  constraints: InverseConstraintSpec;
  preferences: Record<string, unknown>;
  sampling: InverseSamplingSpec;
  reference_split: string;
  reference_case_index: number;
}

export interface InverseRunResponse {
  job_id: string;
  status: string;
  status_url: string;
  result_url: string;
}

export interface TargetPreset {
  name: string;
  label: string;
  path: string;
  target: Record<string, unknown>;
}

export interface InverseCandidate {
  rank: number;
  sample_index: number;
  count: number;
  centers: number[][];
  heat_powers?: number[];
  valid: boolean;
  total_score: number;
  kpi_score: number;
  constraint_penalty: number;
  verified_kpis: Record<string, unknown>;
  score_detail: Record<string, unknown>;
  validity: Record<string, unknown>;
}

export interface InverseResult {
  job_id: string;
  status: string;
  summary: Record<string, unknown>;
  target: Record<string, unknown>;
  candidate_count: number;
  candidates: InverseCandidate[];
  artifacts: Record<string, string>;
  stdout_tail: string[];
  stderr_tail: string[];
}

export interface ApiErrorPayload {
  detail?: string;
}

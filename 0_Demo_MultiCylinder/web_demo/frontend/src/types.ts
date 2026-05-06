export type Mode = "deterministic" | "generative";
export type DemoMode = "forward" | "inverse";
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

export interface InverseModelEntry {
  id: string;
  label: string;
  enabled: boolean;
  available: boolean;
  preload: boolean;
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
  default_forward_verifier_id: string | null;
  metadata: Record<string, unknown>;
}

export type KpiTargetMode = "exact" | "range" | "max" | "min" | "minimize" | "maximize";

export interface KpiTargetSpec {
  enabled: boolean;
  name: string;
  mode: KpiTargetMode;
  value: number | null;
  low: number | null;
  high: number | null;
  weight: number;
}

export interface InverseConstraintSpec {
  re: number;
  num_cylinders_min: number;
  num_cylinders_max: number;
  min_center_distance: number;
  min_x_span: number | null;
  min_y_span: number | null;
}

export interface InverseSamplingSpec {
  n_samples: number;
  verify_top_k: number;
  save_verified_top_k: number;
  n_steps: number;
  seed: number | null;
}

export interface InverseVerificationSpec {
  forward_verifier_model_id: string;
  forward_backend: Mode;
  phase_bins: number;
  nx: number;
  ny: number;
  generative_num_samples: number;
  generative_n_steps: number;
  generative_ode_solver: "euler" | "heun";
  uncertainty_penalty_weight: number;
}

export interface InverseRunRequest {
  inverse_model_id: string;
  target_name: string | null;
  kpis: KpiTargetSpec[];
  constraints: InverseConstraintSpec;
  sampling: InverseSamplingSpec;
  verification: InverseVerificationSpec;
  simulation_enabled: boolean;
}

export interface InverseRunResponse {
  job_id: string;
  status: string;
  status_url: string;
  result_url: string;
}

export interface InverseKpiEntry {
  name: string;
  label: string;
  default_mode: KpiTargetMode;
  default_weight: number;
}

export interface InverseTargetPreset {
  name: string;
  label: string;
  path: string;
  target: {
    name?: string;
    Re?: number;
    re?: number;
    num_cylinders_min?: number;
    num_cylinders_max?: number;
    min_center_distance?: number;
    kpis?: Record<string, Partial<KpiTargetSpec> & { mode?: KpiTargetMode }>;
    preferences?: Record<string, unknown>;
  };
}

export interface InverseJobStatus {
  job_id: string;
  status: "queued" | "running" | "parsing_results" | "complete" | "error" | string;
  updated_at?: string;
  created_at?: string;
  started_at?: string;
  completed_at?: string;
  error?: string;
  result_url?: string;
  candidate_count?: number;
  log_tail?: string[];
}

export interface KpiComparisonRow {
  target: {
    mode: KpiTargetMode | string;
    value?: number;
    low?: number;
    high?: number;
    weight?: number;
  };
  achieved: number | null;
  simulation: number | null;
  pass: boolean | null;
}

export interface InverseCandidate {
  id: string;
  rank: number | null;
  sample_index: number;
  score: number | null;
  centers: number[][];
  count: number;
  validity: Record<string, unknown>;
  kpis: Record<string, number>;
  kpis_std?: Record<string, number>;
  kpi_comparison: Record<string, KpiComparisonRow>;
  per_kpi_errors?: Record<string, number>;
  constraint_penalty?: number | null;
  latent_consistency?: number | null;
  verifier_backend?: Mode | string;
  quick_validation_status: string;
  simulation_validation_status: string;
  image_urls?: Record<string, string>;
  artifact_urls?: Record<string, string>;
  frame_urls?: Record<string, string[]>;
  hypergraph?: Hypergraph | null;
  quick_validation?: {
    job_id: string;
    result_url: string;
    frame_urls?: Record<string, string[]>;
    hypergraph?: Hypergraph | null;
  };
  simulation_verification?: {
    ground_truth_kpis?: Record<string, number>;
    kpi_comparison?: Record<string, unknown>;
    ground_truth_score?: number;
    score_delta?: number | null;
  } | null;
  raw?: Record<string, unknown>;
}

export interface InverseJobResult {
  job_id: string;
  status: string;
  target: InverseTargetPreset["target"];
  request: InverseRunRequest;
  sampling: InverseSamplingSpec;
  verification: InverseVerificationSpec;
  constraints: InverseConstraintSpec;
  domain: {
    length_x: number;
    length_y: number;
    max_num_cylinders?: number;
    phase_bins?: number;
    resolution_nx?: number;
    resolution_ny?: number;
  };
  files: Record<string, string>;
  candidates: InverseCandidate[];
}

export interface CandidateValidationResult {
  status: string;
  candidate: InverseCandidate;
  forward_result?: JobResult;
}

export interface CandidateSimulationValidationRequest {
  simulation_mode: "inert" | "active";
  simulation_device?: "cpu" | "gpu" | null;
  simulation_gpu_id?: number | null;
  simulation_preprocess_device?: string | null;
  simulation_nx?: number | null;
  simulation_ny?: number | null;
  simulation_phase_bins?: number | null;
  simulation_warmup_cycles?: number | null;
  simulation_save_cycles?: number | null;
  simulation_frames_per_cycle?: number | null;
  simulation_dt?: number | null;
}

export interface SimulationValidationStatus {
  job_id: string;
  candidate_id: string;
  status: string;
  updated_at?: string;
  created_at?: string;
  completed_at?: string;
  error?: string;
  log_tail?: string[];
  result_url?: string;
  candidate?: InverseCandidate;
}

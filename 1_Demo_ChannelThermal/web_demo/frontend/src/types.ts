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
  design_source: "reference_case" | "diy" | "candidate";
  re: number | null;
  u_in: number | null;
  modules: ThermalModule[];
  field: FieldName;
  display_scale: number;
  display_smoothing: boolean;
  render_interpolation: "nearest" | "bilinear" | "bicubic" | "lanczos";
  return_kpis: boolean;
  return_organization: boolean;
  return_ground_truth: boolean;
  return_error: boolean;
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
  comparison?: FieldComparison | null;
  internal_temperature?: InternalTemperatureResult | null;
  kpis: Record<string, unknown> | null;
  modules: ThermalModule[];
  heat_power_source: string;
  organization: OrganizationSummary | null;
  artifacts: Record<string, string>;
  export_npz_url: string;
}

export interface FieldComparisonMetric {
  rmse: number;
  nrmse: number;
  relative_l2: number;
  mae: number;
  max_abs: number;
  normalizer: number;
}

export interface FieldComparison {
  available: boolean;
  mode: "reference_ground_truth" | "simulation_verification" | "simulation_only" | "inference_only" | string;
  reason?: string | null;
  metrics?: Record<string, FieldComparisonMetric>;
  error_definition?: string;
  error_label?: string;
  ground_truth_frame_urls?: Record<string, string[]>;
  relative_error_frame_urls?: Record<string, string[]>;
  truth_render?: { fields: Record<string, { vmin: number; vmax: number; frames: string[] }> };
  error_render?: { fields: Record<string, { vmin: number; vmax: number; frames: string[] }> };
}

export interface InternalTemperatureModule {
  index: number;
  label: string;
  heat_power: number;
  inferred_url?: string | null;
  ground_truth_url?: string | null;
  simulation_url?: string | null;
  relative_error_url?: string | null;
  metrics?: FieldComparisonMetric | null;
}

export interface InternalTemperatureResult {
  available: boolean;
  quantity: string;
  count: number;
  default_visible_count: number;
  scale?: { vmin: number; vmax: number; label: string } | null;
  error_scale?: { vmin: number; vmax: number; label: string } | null;
  modules: InternalTemperatureModule[];
  reason?: string | null;
}

export interface ForwardSimulationRequest {
  design: DesignRequest;
  prediction_job_id?: string | null;
  max_runtime_seconds?: number;
}

export interface ForwardSimulationResponse {
  job_id: string;
  status: string;
  status_url: string;
  result_url: string;
}

export interface ForwardSimulationResult {
  job_id: string;
  status: string;
  fields: FieldName[];
  frame_urls: Record<string, string[]>;
  predicted_frame_urls?: Record<string, string[]>;
  render?: {
    frame_count: number;
    fields: Record<string, { vmin: number; vmax: number; frames: string[] }>;
    raw_resolution: { nx: number; ny: number };
    display_resolution: { width: number; height: number };
  };
  comparison?: FieldComparison | null;
  internal_temperature?: InternalTemperatureResult | null;
  export_npz_url: string;
  stdout_tail: string[];
  stderr_tail: string[];
}

export interface OrganizationSummary {
  A_mh: number[][];
  A_eh_shape: number[];
  env_token_xy: number[][] | null;
  dominant_module_hyperedge: number[];
  dominant_env_hyperedge: number[];
  hyperedge_strength: number[] | null;
  hyperedge_module_mass: number[] | null;
  hyperedge_env_mass: number[] | null;
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

export interface BoxSpec {
  x: [number, number];
  y: [number, number];
}

export type HeatLoadMode =
  | "per_module"
  | "per_module_range"
  | "uniform"
  | "uniform_range"
  | "total_only"
  | "from_reference"
  | "none";

export interface HeatLoadSpec {
  mode: HeatLoadMode;
  values?: number[] | null;
  ranges?: [number, number][] | null;
  value?: number | null;
  range?: [number, number] | null;
  total?: number | null;
  sort_mode: "heat_desc_then_xy" | "slot_order" | "anonymous";
}

export interface StructureConstraintSpec {
  enabled: boolean;
  strength: number;
  x_span?: [number, number] | null;
  y_span?: [number, number] | null;
  min_x_coverage?: number | null;
  min_y_coverage?: number | null;
  min_mean_pair_distance?: number | null;
  centroid?: [number, number] | null;
  centroid_tolerance?: [number, number] | null;
  avoid_vertical_stack: boolean;
  keepout_boxes: BoxSpec[];
  protected_boxes: BoxSpec[];
  preferred_boxes: BoxSpec[];
  sketch_maps?: Record<string, number[][]> | null;
}

export interface ThermalLimitSpec {
  solid_temperature_max?: number | null;
  module_temperature_spread_max?: number | null;
  pressure_drop_max?: number | null;
  wall_hot_delta_T?: number | null;
  outlet_hot_delta_T?: number | null;
}

export interface ObjectiveWeightsSpec {
  safety: number;
  uniformity: number;
  pressure: number;
  outlet_mixing: number;
  wall_protection: number;
  plume_avoidance: number;
  coverage: number;
}

export interface InverseRunRequest {
  inverse_model_id: string;
  forward_model_id: string;
  target_name: string | null;
  target_mode: "design_intent" | "legacy_kpi";
  kpis: KpiTargetSpec[];
  constraints: InverseConstraintSpec;
  heat_loads: HeatLoadSpec;
  structure_constraints: StructureConstraintSpec;
  thermal_limits: ThermalLimitSpec;
  objective_weights: ObjectiveWeightsSpec;
  field_preferences: Record<string, unknown>;
  preferences: Record<string, unknown>;
  sampling: InverseSamplingSpec;
  guidance_scale?: number | null;
  diversity_rerank_weight?: number | null;
  candidate_pool_multiplier?: number | null;
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
  target_mode: "design_intent" | "legacy_kpi";
  source_dir: string;
}

export interface InverseCandidate {
  rank: number;
  sample_index: number;
  count: number;
  centers: number[][];
  heat_powers?: number[];
  heat_power_source?: string;
  valid: boolean;
  total_score: number;
  design_intent_score?: number | null;
  kpi_score: number;
  constraint_penalty: number;
  hypergraph_consistency_score?: number | null;
  hypergraph_diagnostics_available?: boolean;
  hypergraph_active_count_error?: number | null;
  hypergraph_source_rmse?: number | null;
  hypergraph_thermal_region_rmse?: number | null;
  hypergraph_A_mh_l1?: number | null;
  verified_kpis: Record<string, unknown>;
  score_detail: Record<string, unknown>;
  design_intent_score_detail?: Record<string, unknown>;
  structure_score_detail?: Record<string, unknown>;
  validity: Record<string, unknown>;
  artifacts?: Record<string, string>;
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

import { useCallback, useEffect, useMemo, useState } from "react";
import type { PointerEvent } from "react";
import {
  Activity,
  Download,
  Eraser,
  Flame,
  Layers3,
  Loader2,
  Move,
  Plus,
  RefreshCcw,
  Shuffle,
  Target,
  Trash2,
  Wand2,
} from "lucide-react";
import {
  apiUrl,
  getInverseCandidates,
  getInverseDebugFiles,
  getInverseKpis,
  getInverseModels,
  getInverseResult,
  getInverseStatus,
  getJobResult,
  getForwardSimulationResult,
  getForwardSimulationStatus,
  getModelConfig,
  getModels,
  getReferenceCases,
  getTargetPresets,
  runInference,
  runForwardSimulation,
  runInverse,
  validateDesign,
} from "./api";
import type {
  BoxSpec,
  DemoMode,
  DesignRequest,
  FieldName,
  ForwardSimulationResult,
  HeatLoadMode,
  HeatLoadSpec,
  InverseCandidate,
  InverseConstraintSpec,
  InverseModelEntry,
  InverseResult,
  InverseSamplingSpec,
  JobResult,
  KpiInfo,
  KpiMode,
  KpiTargetSpec,
  ModelConfig,
  ModelEntry,
  ObjectiveWeightsSpec,
  OrganizationSummary,
  ReferenceCase,
  StructureConstraintSpec,
  TargetPreset,
  ThermalLimitSpec,
  ThermalModule,
  ValidationResult,
} from "./types";

const FIELD_LABELS: Record<FieldName, string> = {
  temperature: "Temperature",
  u: "u velocity",
  v: "v velocity",
  p: "Pressure",
  omega: "Vorticity",
};

const FIELD_ORDER: FieldName[] = ["temperature", "u", "v", "p", "omega"];

const KPI_PRIORITY = [
  "max_solid_temperature",
  "p95_solid_temperature",
  "mean_solid_temperature",
  "module_peak_temperature_spread",
  "module_mean_temperature_std",
  "pressure_drop",
  "outlet_temperature_rise_mean",
  "outlet_temperature_nonuniformity",
  "thermal_plume_area",
  "thermal_plume_length",
  "hot_fluid_area_fraction",
  "hot_solid_area_fraction",
  "wall_hot_area_fraction",
  "heat_power_total",
  "num_modules",
];

const DEFAULT_CONSTRAINTS: InverseConstraintSpec = {
  num_modules_min: 4,
  num_modules_max: 8,
  min_center_distance: 1.1,
  wall_clearance: 0.08,
  inlet_clearance: 0.3,
  outlet_clearance: 0.3,
  heat_power_total: null,
};

const DEFAULT_SAMPLING: InverseSamplingSpec = {
  n_samples: 32,
  n_steps: 4,
  seed: 123,
  count_mode: "uniform",
};

const DEFAULT_HEAT_LOADS: HeatLoadSpec = {
  mode: "from_reference",
  values: null,
  ranges: null,
  value: null,
  range: [0.8, 1.8],
  total: null,
  sort_mode: "heat_desc_then_xy",
};

const DEFAULT_STRUCTURE: StructureConstraintSpec = {
  enabled: false,
  strength: 0.5,
  x_span: [1.0, 10.8],
  y_span: [0.65, 3.35],
  min_x_coverage: null,
  min_y_coverage: null,
  min_mean_pair_distance: null,
  centroid: null,
  centroid_tolerance: null,
  avoid_vertical_stack: false,
  keepout_boxes: [],
  protected_boxes: [],
  preferred_boxes: [],
  sketch_maps: null,
};

const DEFAULT_LIMITS: ThermalLimitSpec = {
  solid_temperature_max: 1.8,
  module_temperature_spread_max: 0.3,
  pressure_drop_max: 0.13,
  wall_hot_delta_T: 0.22,
  outlet_hot_delta_T: 0.18,
};

const DEFAULT_WEIGHTS: ObjectiveWeightsSpec = {
  safety: 1.0,
  uniformity: 0.8,
  pressure: 0.4,
  outlet_mixing: 0.6,
  wall_protection: 0.4,
  plume_avoidance: 0.7,
  coverage: 0.3,
};

type FlowMode = "reference" | "u_in" | "re";
type CountMode = "exact" | "range" | "unspecified";
type PlacementMode = "none" | "sketch" | "quantitative" | "reference";
type SketchTool = "preferred" | "keepout" | "protected" | "erase";
type DebugFile = { path: string; size: number; url: string | null };

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function list(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function numeric(value: unknown, fallback = 0): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function maybeNumber(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function pair(value: unknown, fallback: [number, number] | null = null): [number, number] | null {
  if (!Array.isArray(value) || value.length < 2) return fallback;
  const a = numeric(value[0], NaN);
  const b = numeric(value[1], NaN);
  return Number.isFinite(a) && Number.isFinite(b) ? [a, b] : fallback;
}

function formatNumber(value: unknown, digits = 4): string {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "na";
  if (Math.abs(parsed) >= 100 || Math.abs(parsed) < 0.01) return parsed.toExponential(2);
  return parsed.toFixed(digits).replace(/\.?0+$/, "");
}

function heatColor(value: number, min = 0, max = 3): string {
  const t = Math.min(Math.max((value - min) / Math.max(max - min, 1e-6), 0), 1);
  const cool = [35, 126, 158];
  const warm = [204, 64, 72];
  const mid = [221, 164, 55];
  const mix = t < 0.5 ? t * 2 : (t - 0.5) * 2;
  const a = t < 0.5 ? cool : mid;
  const b = t < 0.5 ? mid : warm;
  const rgb = a.map((channel, index) => Math.round(channel + (b[index] - channel) * mix));
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function emptySketch(width = 24, height = 12): number[][] {
  return Array.from({ length: height }, () => Array.from({ length: width }, () => 0));
}

function normalizeBoxes(value: unknown): BoxSpec[] {
  return list(value)
    .map((item) => {
      const raw = record(item);
      const x = pair(raw.x ?? raw.x_span);
      const y = pair(raw.y ?? raw.y_span);
      return x && y ? { x, y } : null;
    })
    .filter((item): item is BoxSpec => item !== null);
}

function modulesFromReference(ref: ReferenceCase | null, config: ModelConfig | null): ThermalModule[] {
  if (ref?.modules?.length) return ref.modules.map((item) => ({ ...item }));
  const lx = config?.domain_length_x ?? 12;
  const ly = config?.domain_length_y ?? 6;
  const heat = config?.default_heat_power ?? 1.2;
  return [
    { x: lx * 0.24, y: ly * 0.34, heat_power: heat },
    { x: lx * 0.38, y: ly * 0.66, heat_power: heat },
    { x: lx * 0.55, y: ly * 0.42, heat_power: heat },
    { x: lx * 0.70, y: ly * 0.60, heat_power: heat },
  ].slice(0, Math.min(4, config?.max_num_modules ?? 4));
}

function makeDesign(modelId: string, config: ModelConfig | null, ref: ReferenceCase | null): DesignRequest {
  return {
    model_id: modelId,
    reference_split: config?.default_reference_split ?? "test",
    reference_case_index: ref?.index ?? config?.default_reference_case_index ?? 0,
    reference_case_id: null,
    design_source: ref ? "reference_case" : "diy",
    re: null,
    u_in: null,
    modules: modulesFromReference(ref, config),
    field: "temperature",
    display_scale: 3,
    display_smoothing: true,
    render_interpolation: "bicubic",
    return_kpis: true,
    return_organization: true,
    return_ground_truth: Boolean(ref),
    return_error: Boolean(ref),
  };
}

function clampModule(module: ThermalModule, config: ModelConfig | null): ThermalModule {
  const lx = config?.domain_length_x ?? 12;
  const ly = config?.domain_length_y ?? 6;
  const radius = config?.module_radius ?? 0.45;
  const heatMin = config?.heat_power_min ?? 0;
  const heatMax = config?.heat_power_max ?? 3;
  return {
    x: Math.min(Math.max(module.x, radius), lx - radius),
    y: Math.min(Math.max(module.y, radius), ly - radius),
    heat_power: Math.min(Math.max(module.heat_power, heatMin), Math.max(heatMax, heatMin)),
  };
}

function randomModules(count: number, config: ModelConfig | null): ThermalModule[] {
  const lx = config?.domain_length_x ?? 12;
  const ly = config?.domain_length_y ?? 6;
  const radius = config?.module_radius ?? 0.45;
  const heatMin = config?.heat_power_min ?? 0.6;
  const heatMax = config?.heat_power_max ?? 2.2;
  const minDistance = Math.max(2.05 * radius, 1.0);
  const modules: ThermalModule[] = [];
  let attempts = 0;
  while (modules.length < count && attempts < 1000) {
    attempts += 1;
    const candidate = {
      x: radius + Math.random() * Math.max(lx - 2 * radius, 0.1),
      y: radius + Math.random() * Math.max(ly - 2 * radius, 0.1),
      heat_power: heatMin + Math.random() * Math.max(heatMax - heatMin, 0.1),
    };
    if (modules.every((item) => Math.hypot(item.x - candidate.x, item.y - candidate.y) >= minDistance)) modules.push(candidate);
  }
  return modules.length ? modules : modulesFromReference(null, config);
}

function kpiRowsFromPreset(preset: TargetPreset | null, kpiInfos: KpiInfo[]): KpiTargetSpec[] {
  const target = preset?.target ?? {};
  const rawKpis = record(target.kpis);
  const rows = Object.entries(rawKpis).map(([name, raw]) => {
    const entry = record(raw);
    const mode = String(entry.mode ?? "max") as KpiMode;
    return {
      enabled: true,
      name,
      mode,
      value: entry.value == null ? null : numeric(entry.value),
      low: entry.low == null ? null : numeric(entry.low),
      high: entry.high == null ? null : numeric(entry.high),
      weight: numeric(entry.weight, 1),
    };
  });
  if (rows.length) return rows;
  return kpiInfos.slice(0, 5).map((item) => ({
    enabled: item.name === "max_solid_temperature" || item.name === "pressure_drop",
    name: item.name,
    mode: item.default_mode,
    value: null,
    low: null,
    high: item.name.includes("temperature") ? 2.0 : 0.1,
    weight: item.default_weight,
  }));
}

function constraintsFromPreset(preset: TargetPreset | null, config: ModelConfig | null): InverseConstraintSpec {
  const target = preset?.target ?? {};
  const scenario = record(target.scenario);
  const geometry = record(target.geometry_constraints);
  return {
    num_modules_min: numeric(scenario.num_modules_min ?? target.num_modules_min, DEFAULT_CONSTRAINTS.num_modules_min),
    num_modules_max: numeric(scenario.num_modules_max ?? target.num_modules_max, DEFAULT_CONSTRAINTS.num_modules_max),
    min_center_distance: numeric(geometry.min_center_distance ?? target.min_center_distance, DEFAULT_CONSTRAINTS.min_center_distance),
    wall_clearance: numeric(geometry.wall_clearance ?? target.wall_clearance, DEFAULT_CONSTRAINTS.wall_clearance),
    inlet_clearance: numeric(geometry.inlet_clearance ?? target.inlet_clearance, DEFAULT_CONSTRAINTS.inlet_clearance),
    outlet_clearance: numeric(geometry.outlet_clearance ?? target.outlet_clearance, DEFAULT_CONSTRAINTS.outlet_clearance),
    heat_power_total: maybeNumber(target.heat_power_total) ?? null,
  };
}

function heatLoadsFromPreset(preset: TargetPreset | null): HeatLoadSpec {
  const raw = record(preset?.target?.heat_loads);
  const rawMode = String(raw.mode ?? DEFAULT_HEAT_LOADS.mode) as HeatLoadMode;
  const mode: HeatLoadMode = ["per_module", "per_module_range", "uniform", "uniform_range", "total_only", "from_reference", "none"].includes(rawMode)
    ? rawMode
    : list(raw.values).length
      ? "per_module"
      : "from_reference";
  return {
    mode,
    values: list(raw.values).map((item) => numeric(item)).filter(Number.isFinite),
    ranges: list(raw.ranges)
      .map((item) => pair(item))
      .filter((item): item is [number, number] => item !== null),
    value: maybeNumber(raw.value),
    range: pair(raw.range, DEFAULT_HEAT_LOADS.range ?? [0.8, 1.8]),
    total: maybeNumber(raw.total),
    sort_mode: String(raw.sort_mode ?? "heat_desc_then_xy") as HeatLoadSpec["sort_mode"],
  };
}

function structureFromPreset(preset: TargetPreset | null, config: ModelConfig | null): StructureConstraintSpec {
  const target = preset?.target ?? {};
  const geometry = record(target.geometry_constraints);
  const structure = record(target.structure_constraints);
  const rawSketch = record(structure.sketch_maps);
  const sketchMaps =
    Object.keys(rawSketch).length > 0
      ? Object.fromEntries(
          Object.entries(rawSketch)
            .filter(([, value]) => Array.isArray(value))
            .map(([key, value]) => [key, value as number[][]]),
        )
      : null;
  return {
    ...DEFAULT_STRUCTURE,
    enabled: Boolean(structure.enabled ?? preset?.target_mode === "design_intent"),
    strength: numeric(structure.strength, DEFAULT_STRUCTURE.strength),
    x_span: pair(structure.x_span ?? geometry.x_span, [config?.module_radius ?? 0.45, (config?.domain_length_x ?? 12) - (config?.module_radius ?? 0.45)]),
    y_span: pair(structure.y_span ?? geometry.y_span, [config?.module_radius ?? 0.45, (config?.domain_length_y ?? 4) - (config?.module_radius ?? 0.45)]),
    min_x_coverage: maybeNumber(structure.min_x_coverage ?? structure.x_coverage_min ?? record(target.field_preferences).min_x_coverage),
    min_y_coverage: maybeNumber(structure.min_y_coverage ?? structure.y_coverage_min ?? record(target.field_preferences).min_y_coverage),
    min_mean_pair_distance: maybeNumber(structure.min_mean_pair_distance ?? structure.mean_pair_distance_min ?? record(target.field_preferences).min_mean_pair_distance),
    centroid: pair(structure.centroid),
    centroid_tolerance: pair(structure.centroid_tolerance),
    avoid_vertical_stack: Boolean(structure.avoid_vertical_stack),
    keepout_boxes: normalizeBoxes(structure.keepout_boxes ?? geometry.keepout_boxes),
    protected_boxes: normalizeBoxes(structure.protected_boxes ?? geometry.protected_boxes),
    preferred_boxes: normalizeBoxes(structure.preferred_boxes ?? geometry.preferred_boxes),
    sketch_maps: sketchMaps,
  };
}

function limitsFromPreset(preset: TargetPreset | null): ThermalLimitSpec {
  const thermal = record(preset?.target?.thermal_limits);
  return {
    solid_temperature_max: maybeNumber(thermal.solid_temperature_max) ?? DEFAULT_LIMITS.solid_temperature_max,
    module_temperature_spread_max: maybeNumber(thermal.module_temperature_spread_max) ?? DEFAULT_LIMITS.module_temperature_spread_max,
    pressure_drop_max: maybeNumber(thermal.pressure_drop_max) ?? DEFAULT_LIMITS.pressure_drop_max,
    wall_hot_delta_T: maybeNumber(thermal.wall_hot_delta_T) ?? DEFAULT_LIMITS.wall_hot_delta_T,
    outlet_hot_delta_T: maybeNumber(thermal.outlet_hot_delta_T) ?? DEFAULT_LIMITS.outlet_hot_delta_T,
  };
}

function weightsFromPreset(preset: TargetPreset | null): ObjectiveWeightsSpec {
  const weights = record(preset?.target?.objective_weights);
  return Object.fromEntries(
    Object.entries(DEFAULT_WEIGHTS).map(([key, value]) => [key, numeric(weights[key], value)]),
  ) as unknown as ObjectiveWeightsSpec;
}

function fieldPrefsFromPreset(preset: TargetPreset | null): Record<string, unknown> {
  const field = record(preset?.target?.field_preferences);
  return {
    avoid_downstream_hot_plumes: field.avoid_downstream_hot_plumes ?? true,
    protect_wall_band: field.protect_wall_band ?? false,
    protect_outlet_uniformity: field.protect_outlet_uniformity ?? false,
    min_x_coverage: field.min_x_coverage ?? null,
    min_y_coverage: field.min_y_coverage ?? null,
    min_mean_pair_distance: field.min_mean_pair_distance ?? null,
  };
}

function repeatOrResize(values: number[], count: number): number[] {
  if (!values.length) return [];
  return Array.from({ length: count }, (_, index) => values[index % values.length]);
}

function candidateHeat(candidate: InverseCandidate, heatLoads: HeatLoadSpec, config: ModelConfig | null): number[] {
  const count = Math.max(Number(candidate.count) || 0, 0);
  if (heatLoads.mode === "per_module" && heatLoads.values?.length) return repeatOrResize(heatLoads.values.map(Number), count);
  if (heatLoads.mode === "per_module_range" && heatLoads.ranges?.length) return repeatOrResize(heatLoads.ranges.map(([low, high]) => (Number(low) + Number(high)) / 2), count);
  if (heatLoads.mode === "uniform" && heatLoads.value != null) return Array.from({ length: count }, () => Number(heatLoads.value));
  if (heatLoads.mode === "uniform_range" && heatLoads.range) return Array.from({ length: count }, () => (Number(heatLoads.range?.[0]) + Number(heatLoads.range?.[1])) / 2);
  if (heatLoads.mode === "total_only" && heatLoads.total != null) return Array.from({ length: count }, () => Number(heatLoads.total) / Math.max(count, 1));
  if (candidate.heat_powers?.length && candidate.heat_powers.some((value) => Math.abs(Number(value)) > 1e-8)) return repeatOrResize(candidate.heat_powers.map(Number), count);
  const fallback = config?.default_heat_power ?? 1.2;
  return Array.from({ length: count }, () => fallback);
}

function DesignDomainEditor({
  modules,
  config,
  validation,
  selected,
  onSelect,
  onChange,
}: {
  modules: ThermalModule[];
  config: ModelConfig | null;
  validation: ValidationResult | null;
  selected: number | null;
  onSelect: (index: number | null) => void;
  onChange: (index: number, module: ThermalModule) => void;
}) {
  const [dragging, setDragging] = useState<number | null>(null);
  const lx = validation?.domain_length_x ?? config?.domain_length_x ?? 12;
  const ly = validation?.domain_length_y ?? config?.domain_length_y ?? 4;
  const radius = validation?.module_radius ?? config?.module_radius ?? 0.45;
  const heatMin = validation?.heat_power_min ?? config?.heat_power_min ?? 0;
  const heatMax = validation?.heat_power_max ?? config?.heat_power_max ?? 3;
  const invalidPairs = useMemo(() => {
    const pairs = new Set<string>();
    const minDistance = validation?.min_center_distance ?? 2 * radius;
    modules.forEach((a, i) => {
      modules.forEach((b, j) => {
        if (j <= i) return;
        if (Math.hypot(a.x - b.x, a.y - b.y) < minDistance) {
          pairs.add(String(i));
          pairs.add(String(j));
        }
      });
    });
    return pairs;
  }, [modules, radius, validation?.min_center_distance]);

  const pointerToModule = useCallback(
    (event: PointerEvent<SVGSVGElement>, index: number) => {
      const rect = event.currentTarget.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * lx;
      const y = ly - ((event.clientY - rect.top) / Math.max(rect.height, 1)) * ly;
      onChange(index, clampModule({ ...modules[index], x, y }, config));
    },
    [config, lx, ly, modules, onChange],
  );

  return (
    <svg
      className="design-domain-canvas"
      viewBox={`0 0 ${lx} ${ly}`}
      role="img"
      onPointerMove={(event) => {
        if (dragging !== null) pointerToModule(event, dragging);
      }}
      onPointerUp={() => setDragging(null)}
      onPointerLeave={() => setDragging(null)}
    >
      <defs>
        <marker id="arrowHead" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L7,3 z" fill="#2d6f8f" />
        </marker>
      </defs>
      <rect x="0" y="0" width={lx} height={ly} fill="#f8faf7" />
      <g opacity="0.25">
        {Array.from({ length: 9 }, (_, i) => (
          <line key={`x-${i}`} x1={(lx * i) / 8} y1="0" x2={(lx * i) / 8} y2={ly} stroke="#9aa79c" strokeWidth="0.012" />
        ))}
        {Array.from({ length: 5 }, (_, i) => (
          <line key={`y-${i}`} x1="0" y1={(ly * i) / 4} x2={lx} y2={(ly * i) / 4} stroke="#9aa79c" strokeWidth="0.012" />
        ))}
      </g>
      <line x1="0.35" y1={ly * 0.5} x2="1.45" y2={ly * 0.5} stroke="#2d6f8f" strokeWidth="0.08" markerEnd="url(#arrowHead)" />
      <line x1={lx - 1.45} y1={ly * 0.5} x2={lx - 0.35} y2={ly * 0.5} stroke="#2d6f8f" strokeWidth="0.08" markerEnd="url(#arrowHead)" />
      <text x={0.25} y={0.35} fontSize="0.24" fill="#53605a">inlet</text>
      <text x={lx - 0.95} y={0.35} fontSize="0.24" fill="#53605a">outlet</text>
      {modules.map((module, index) => {
        const boundaryBad = module.x < radius || module.x > lx - radius || module.y < radius || module.y > ly - radius;
        const invalid = boundaryBad || invalidPairs.has(String(index));
        return (
          <g key={index}>
            <circle
              cx={module.x}
              cy={ly - module.y}
              r={radius}
              fill={heatColor(module.heat_power, heatMin, heatMax)}
              stroke={invalid ? "#d33b45" : selected === index ? "#111827" : "#ffffff"}
              strokeWidth={invalid ? 0.09 : selected === index ? 0.075 : 0.045}
              onPointerDown={(event) => {
                event.currentTarget.setPointerCapture(event.pointerId);
                setDragging(index);
                onSelect(index);
              }}
            />
            <text x={module.x} y={ly - module.y - 0.02} textAnchor="middle" fontSize="0.24" fill="#111827" pointerEvents="none" fontWeight="700">
              M{index + 1}
            </text>
            <text x={module.x} y={ly - module.y + 0.27} textAnchor="middle" fontSize="0.18" fill="#111827" pointerEvents="none">
              q={formatNumber(module.heat_power, 2)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function ModuleHeatTable({
  modules,
  config,
  selected,
  onSelect,
  onChange,
  onAdd,
  onDelete,
  onSetAll,
  onRandomHeat,
  onNormalize,
}: {
  modules: ThermalModule[];
  config: ModelConfig | null;
  selected: number | null;
  onSelect: (index: number) => void;
  onChange: (index: number, patch: Partial<ThermalModule>) => void;
  onAdd: () => void;
  onDelete: (index: number) => void;
  onSetAll: () => void;
  onRandomHeat: () => void;
  onNormalize: () => void;
}) {
  return (
    <div className="module-heat-table">
      <div className="table-actions">
        <button type="button" onClick={onSetAll}>Set all heat</button>
        <button type="button" onClick={onRandomHeat}><Shuffle size={15} /> Random heat</button>
        <button type="button" onClick={onNormalize}>Normalize total</button>
        <button type="button" onClick={onAdd} disabled={modules.length >= (config?.max_num_modules ?? 12)}><Plus size={15} /> Add module</button>
      </div>
      <div className="module-table-grid module-table-head">
        <span>M</span>
        <span>x</span>
        <span>y</span>
        <span>Heat power q</span>
        <span />
      </div>
      {modules.map((module, index) => (
        <div key={index} className={`module-table-grid module-table-row ${selected === index ? "selected" : ""}`}>
          <button type="button" className="module-index" onClick={() => onSelect(index)}>M{index + 1}</button>
          <input type="number" value={module.x.toFixed(3)} step="0.05" onChange={(event) => onChange(index, { x: Number(event.target.value) })} />
          <input type="number" value={module.y.toFixed(3)} step="0.05" onChange={(event) => onChange(index, { y: Number(event.target.value) })} />
          <input type="number" value={module.heat_power.toFixed(3)} step="0.05" onChange={(event) => onChange(index, { heat_power: Number(event.target.value) })} />
          <button type="button" className="icon-button" onClick={() => onDelete(index)} title="Delete selected module"><Trash2 size={15} /></button>
        </div>
      ))}
    </div>
  );
}

function FieldGrid({
  result,
  simulation,
  expandedField,
  onExpand,
}: {
  result: JobResult | null;
  simulation: ForwardSimulationResult | null;
  expandedField: FieldName | null;
  onExpand: (field: FieldName | null) => void;
}) {
  const fields = result?.fields?.length ? FIELD_ORDER.filter((name) => result.fields.includes(name)) : FIELD_ORDER;
  const comparison = simulation?.comparison?.available ? simulation.comparison : result?.comparison;
  const simUrls = simulation?.frame_urls ?? {};
  const inferredUrls = simulation?.predicted_frame_urls ?? result?.frame_urls ?? {};
  return (
    <section className="panel field-grid-panel">
      <div className="panel-heading">
        <h2>Steady Fields</h2>
        <span>{comparison?.available ? comparison.mode.replaceAll("_", " ") : result ? `${result.domain.resolution_nx} x ${result.domain.resolution_ny}` : "run forward"}</span>
      </div>
      <div className="field-grid">
        {fields.map((name) => {
          const url = inferredUrls?.[name]?.[0] ?? result?.frame_urls?.[name]?.[0] ?? null;
          const truthUrl = comparison?.ground_truth_frame_urls?.[name]?.[0] ?? null;
          const simUrl = simUrls[name]?.[0] ?? null;
          const errorUrl = comparison?.relative_error_frame_urls?.[name]?.[0] ?? null;
          const nrmse = comparison?.metrics?.[name]?.nrmse;
          const meta = result?.render?.fields?.[name];
          const simMeta = simulation?.render?.fields?.[name];
          const scaleMeta = simMeta ?? meta;
          const errMeta = simulation?.comparison?.error_render?.fields?.[name] ?? result?.comparison?.error_render?.fields?.[name];
          return (
            <button type="button" key={name} className="field-card" onClick={() => url && onExpand(name)}>
              <span>{FIELD_LABELS[name]}</span>
              <div className={`field-variant-grid ${truthUrl || simUrl || errorUrl ? "has-comparison" : ""}`}>
                <div className="field-variant">
                  <b>Inferred</b>
                  {url ? <img src={apiUrl(url)} alt={`${FIELD_LABELS[name]} inferred field`} /> : <div className="empty-field"><Activity size={22} /></div>}
                </div>
                {truthUrl && <div className="field-variant"><b>Ground truth</b><img src={apiUrl(truthUrl)} alt={`${FIELD_LABELS[name]} ground truth`} /></div>}
                {simUrl && <div className="field-variant"><b>Simulation</b><img src={apiUrl(simUrl)} alt={`${FIELD_LABELS[name]} simulation`} /></div>}
                {errorUrl && <div className="field-variant"><b>Norm error</b><img src={apiUrl(errorUrl)} alt={`${FIELD_LABELS[name]} normalized error`} /></div>}
              </div>
              <small>{nrmse !== undefined ? `NRMSE ${formatNumber(nrmse, 4)}` : meta ? `${formatNumber(meta.vmin, 3)} .. ${formatNumber(meta.vmax, 3)}` : "waiting"}</small>
              {scaleMeta && <ColorScale label="field" vmin={scaleMeta.vmin} vmax={scaleMeta.vmax} palette={fieldPalette(name)} />}
              {errorUrl && errMeta && <ColorScale label="rel error" vmin={errMeta.vmin} vmax={errMeta.vmax} palette="magma" />}
            </button>
          );
        })}
      </div>
      {expandedField && result?.frame_urls?.[expandedField]?.[0] && (
        <div className="modal-scrim" onClick={() => onExpand(null)} role="presentation">
          <div className="field-modal">
            <strong>{FIELD_LABELS[expandedField]}</strong>
            <div className="field-modal-grid">
              <div><span>Inferred</span><img src={apiUrl(result.frame_urls[expandedField][0])} alt={`${FIELD_LABELS[expandedField]} inferred expanded`} /></div>
              {comparison?.ground_truth_frame_urls?.[expandedField]?.[0] && <div><span>Ground truth</span><img src={apiUrl(comparison.ground_truth_frame_urls[expandedField][0])} alt={`${FIELD_LABELS[expandedField]} truth expanded`} /></div>}
              {simUrls[expandedField]?.[0] && <div><span>Simulation</span><img src={apiUrl(simUrls[expandedField][0])} alt={`${FIELD_LABELS[expandedField]} simulation expanded`} /></div>}
              {comparison?.relative_error_frame_urls?.[expandedField]?.[0] && <div><span>Normalized error</span><img src={apiUrl(comparison.relative_error_frame_urls[expandedField][0])} alt={`${FIELD_LABELS[expandedField]} error expanded`} /></div>}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

function fieldPalette(name: FieldName): "inferno" | "magma" | "diverging" | "viridis" {
  if (name === "temperature") return "inferno";
  if (name === "p") return "viridis";
  return "diverging";
}

function ColorScale({ label, vmin, vmax, palette }: { label: string; vmin: number; vmax: number; palette: "inferno" | "magma" | "diverging" | "viridis" }) {
  return (
    <div className="color-scale">
      <span>{formatNumber(vmin, 3)}</span>
      <i className={`color-scale-bar ${palette}`} />
      <span>{formatNumber(vmax, 3)}</span>
      <b>{label}</b>
    </div>
  );
}

function ModuleTemperatureGrid({ result, simulation }: { result: JobResult | null; simulation: ForwardSimulationResult | null }) {
  const internal = simulation?.internal_temperature?.available ? simulation.internal_temperature : result?.internal_temperature;
  const [showAll, setShowAll] = useState(false);
  if (!result) {
    return null;
  }
  if (!internal?.available || !internal.modules.length) {
    return (
      <section className="panel module-temperature-panel">
        <div className="panel-heading"><h2>Module Temperature</h2><span>local disks</span></div>
        <div className="empty-list">{internal?.reason ?? "No module-internal temperature distribution is available for this run."}</div>
      </section>
    );
  }
  const visibleCount = showAll ? internal.modules.length : Math.min(internal.default_visible_count ?? 3, internal.modules.length);
  const visible = internal.modules.slice(0, visibleCount);
  return (
    <section className="panel module-temperature-panel">
      <div className="panel-heading">
        <h2>Module Temperature</h2>
        <span>{internal.count} modules</span>
      </div>
      <div className="module-temperature-grid">
        {visible.map((module) => (
          <article key={module.index} className="module-temperature-card">
            <div className="module-temperature-title">
              <strong>{module.label}</strong>
              <small>q {formatNumber(module.heat_power, 3)} {module.metrics ? `/ NRMSE ${formatNumber(module.metrics.nrmse, 4)}` : ""}</small>
            </div>
            <div className={`field-variant-grid ${module.ground_truth_url || module.simulation_url || module.relative_error_url ? "has-comparison" : ""}`}>
              {module.inferred_url && <div className="field-variant"><b>Inferred</b><img src={apiUrl(module.inferred_url)} alt={`${module.label} inferred internal temperature`} /></div>}
              {module.ground_truth_url && <div className="field-variant"><b>Ground truth</b><img src={apiUrl(module.ground_truth_url)} alt={`${module.label} ground truth internal temperature`} /></div>}
              {module.simulation_url && <div className="field-variant"><b>Simulation</b><img src={apiUrl(module.simulation_url)} alt={`${module.label} simulated internal temperature`} /></div>}
              {module.relative_error_url && <div className="field-variant"><b>Norm error</b><img src={apiUrl(module.relative_error_url)} alt={`${module.label} internal normalized error`} /></div>}
            </div>
          </article>
        ))}
      </div>
      <div className="module-temperature-footer">
        {internal.scale && <ColorScale label={internal.scale.label} vmin={internal.scale.vmin} vmax={internal.scale.vmax} palette="inferno" />}
        {internal.error_scale && <ColorScale label={internal.error_scale.label} vmin={internal.error_scale.vmin} vmax={internal.error_scale.vmax} palette="magma" />}
        {internal.modules.length > 3 && (
          <button type="button" onClick={() => setShowAll((value) => !value)}>
            {showAll ? "Show first three" : `Show all ${internal.modules.length}`}
          </button>
        )}
      </div>
    </section>
  );
}

function OrganizerDomainView({ result }: { result: JobResult | null }) {
  const org = result?.organization;
  const domain = result?.domain;
  if (!result) {
    return (
      <section className="panel organizer-domain-card">
        <div className="panel-heading"><h2>Organizer</h2><span>waiting</span></div>
        <div className="empty-list">Run forward to view the domain-level organizer overlay.</div>
      </section>
    );
  }
  return (
    <section className="panel organizer-domain-card">
      <div className="panel-heading"><h2>Organizer</h2><span>{org ? `${org.A_eh_shape.join(" x ")}` : "overlay artifact"}</span></div>
      {org && domain ? (
        <OrganizerSvg org={org} modules={result.modules} domain={domain} />
      ) : result.artifacts.organization_overlay ? (
        <img className="organizer-overlay-img" src={apiUrl(result.artifacts.organization_overlay)} alt="organization overlay" />
      ) : (
        <div className="empty-list">No organizer arrays were returned by this model.</div>
      )}
      {result.artifacts.organization_matrices && (
        <a className="advanced-link" href={apiUrl(result.artifacts.organization_matrices)} target="_blank" rel="noreferrer">
          <Layers3 size={15} /> Advanced matrices
        </a>
      )}
    </section>
  );
}

function OrganizerSvg({
  org,
  modules,
  domain,
}: {
  org: OrganizationSummary;
  modules: ThermalModule[];
  domain: JobResult["domain"];
}) {
  const lx = domain.length_x;
  const ly = domain.length_y;
  const k = Math.max(org.A_mh?.[0]?.length ?? 1, 1);
  const colors = Array.from({ length: k }, (_, i) => `hsl(${(i * 360) / k} 62% 48%)`);
  const env = org.env_token_xy?.length
    ? org.env_token_xy
    : Array.from({ length: org.A_eh_shape[0] ?? 0 }, (_, i) => {
        const nx = Math.max(Math.round(Math.sqrt((org.A_eh_shape[0] || 1) * (lx / Math.max(ly, 1e-6)))), 1);
        const x = 0.4 + (i % nx) * ((lx - 0.8) / Math.max(nx - 1, 1));
        const y = 0.4 + Math.floor(i / nx) * 0.6;
        return [x, Math.min(y, ly - 0.4)];
      });
  return (
    <svg className="organizer-svg" viewBox={`0 0 ${lx} ${ly}`}>
      <rect x="0" y="0" width={lx} height={ly} fill="#f9faf7" stroke="#172026" strokeWidth="0.035" />
      {env.map(([x, y], index) => {
        const h = org.dominant_env_hyperedge[index] ?? 0;
        return <circle key={index} cx={x} cy={ly - y} r="0.055" fill={colors[h % k]} opacity="0.55" />;
      })}
      {modules.map((module, index) => {
        const h = org.dominant_module_hyperedge[index] ?? 0;
        return (
          <g key={index}>
            <circle cx={module.x} cy={ly - module.y} r={domain.module_radius} fill={colors[h % k]} opacity="0.55" stroke="#111827" strokeWidth="0.055" />
            <text x={module.x} y={ly - module.y + 0.06} textAnchor="middle" fontSize="0.22" fontWeight="700">M{index + 1}</text>
          </g>
        );
      })}
    </svg>
  );
}

function KpiPanel({
  result,
  simulation,
  simulationStatus,
  simulationRunning,
  displayKpis,
  exportUrl,
  onRunSimulation,
}: {
  result: JobResult | null;
  simulation: ForwardSimulationResult | null;
  simulationStatus: Record<string, unknown> | null;
  simulationRunning: boolean;
  displayKpis: [string, unknown][];
  exportUrl: string | null;
  onRunSimulation: () => void;
}) {
  const comparison = simulation?.comparison?.available ? simulation.comparison : result?.comparison;
  const comparisonRows = Object.entries(comparison?.metrics ?? {});
  return (
    <aside className="panel kpi-panel">
      <div className="panel-heading">
        <h2>Run + KPI Summary</h2>
        <span>{result ? result.heat_power_source : "waiting"}</span>
      </div>
      <div className="kpi-list">
        {displayKpis.length ? (
          displayKpis.map(([name, value]) => (
            <div key={name} className="kpi-row">
              <span>{name.replaceAll("_", " ")}</span>
              <strong>{formatNumber(value)}</strong>
            </div>
          ))
        ) : (
          <div className="empty-list">Run a forward pass to populate thermal KPIs.</div>
        )}
      </div>
      <div className="comparison-summary">
        <strong>Field comparison</strong>
        {comparison?.available ? (
          comparisonRows.map(([name, metric]) => (
            <div key={name} className="kpi-row">
              <span>{FIELD_LABELS[name as FieldName] ?? name} NRMSE</span>
              <strong>{formatNumber(metric.nrmse, 4)}</strong>
            </div>
          ))
        ) : (
          <small>{comparison?.reason ?? "Reference cases can show ground truth; DIY designs can be verified with a simulation run."}</small>
        )}
      </div>
      <button type="button" className="full" onClick={onRunSimulation} disabled={!result || simulationRunning}>
        {simulationRunning ? <Loader2 className="spin" size={16} /> : <Activity size={16} />} Run simulation verification
      </button>
      {simulationStatus?.status != null && <small className="muted-line">Simulation: {String(simulationStatus.status)}</small>}
      {exportUrl && (
        <a className="button-link full" href={exportUrl}>
          <Download size={16} /> NPZ export
        </a>
      )}
      {simulation?.export_npz_url && (
        <a className="button-link full" href={apiUrl(simulation.export_npz_url)}>
          <Download size={16} /> Simulation NPZ
        </a>
      )}
    </aside>
  );
}

function PlacementIntentCanvas({
  structure,
  placementMode,
  sketchTool,
  config,
  onMode,
  onTool,
  onStructure,
}: {
  structure: StructureConstraintSpec;
  placementMode: PlacementMode;
  sketchTool: SketchTool;
  config: ModelConfig | null;
  onMode: (mode: PlacementMode) => void;
  onTool: (tool: SketchTool) => void;
  onStructure: (patch: Partial<StructureConstraintSpec>) => void;
}) {
  const lx = config?.domain_length_x ?? 12;
  const ly = config?.domain_length_y ?? 4;
  const maps = structure.sketch_maps ?? { preferred: emptySketch(), keepout: emptySketch(), protected: emptySketch(), reference_soft: emptySketch() };
  const preferred = maps.preferred ?? emptySketch();
  const height = preferred.length || 12;
  const width = preferred[0]?.length || 24;
  const [painting, setPainting] = useState(false);

  const paint = useCallback(
    (event: PointerEvent<SVGSVGElement>) => {
      if (placementMode !== "sketch") return;
      const rect = event.currentTarget.getBoundingClientRect();
      const col = Math.min(Math.max(Math.floor(((event.clientX - rect.left) / Math.max(rect.width, 1)) * width), 0), width - 1);
      const row = Math.min(Math.max(Math.floor(((event.clientY - rect.top) / Math.max(rect.height, 1)) * height), 0), height - 1);
      const next = {
        preferred: (maps.preferred ?? emptySketch(width, height)).map((r) => [...r]),
        keepout: (maps.keepout ?? emptySketch(width, height)).map((r) => [...r]),
        protected: (maps.protected ?? emptySketch(width, height)).map((r) => [...r]),
        reference_soft: (maps.reference_soft ?? emptySketch(width, height)).map((r) => [...r]),
      };
      (["preferred", "keepout", "protected"] as const).forEach((name) => {
        if (sketchTool === "erase") next[name][row][col] = 0;
        else if (name === sketchTool) next[name][row][col] = 1;
      });
      onStructure({ enabled: true, sketch_maps: next });
    },
    [height, maps, onStructure, placementMode, sketchTool, width],
  );

  return (
    <div className="placement-canvas">
      <div className="segmented compact">
        {(["none", "sketch", "quantitative", "reference"] as PlacementMode[]).map((mode) => (
          <button key={mode} type="button" className={placementMode === mode ? "active" : ""} onClick={() => onMode(mode)}>
            {mode === "reference" ? "reference family" : mode}
          </button>
        ))}
      </div>
      {placementMode === "sketch" && (
        <div className="sketch-tools">
          {(["preferred", "keepout", "protected", "erase"] as SketchTool[]).map((tool) => (
            <button key={tool} type="button" className={sketchTool === tool ? "active" : ""} onClick={() => onTool(tool)}>
              {tool === "erase" ? <Eraser size={14} /> : <Move size={14} />} {tool}
            </button>
          ))}
        </div>
      )}
      <svg
        viewBox={`0 0 ${lx} ${ly}`}
        onPointerDown={(event) => {
          setPainting(true);
          paint(event);
        }}
        onPointerMove={(event) => {
          if (painting) paint(event);
        }}
        onPointerUp={() => setPainting(false)}
        onPointerLeave={() => setPainting(false)}
      >
        <rect x="0" y="0" width={lx} height={ly} fill="#f8faf7" stroke="#cfd8d1" strokeWidth="0.03" />
        {(["preferred", "keepout", "protected"] as const).flatMap((name) => {
          const color = name === "preferred" ? "#2f8b65" : name === "keepout" ? "#c83f4d" : "#2d6f8f";
          return (maps[name] ?? emptySketch(width, height)).flatMap((row, r) =>
            row.map((value, c) =>
              value > 0 ? (
                <rect
                  key={`${name}-${r}-${c}`}
                  x={(c / width) * lx}
                  y={(r / height) * ly}
                  width={lx / width}
                  height={ly / height}
                  fill={color}
                  opacity="0.32"
                />
              ) : null,
            ),
          );
        })}
        {structure.keepout_boxes.map((box, index) => <BoxRect key={`k-${index}`} box={box} ly={ly} className="keepout" />)}
        {structure.protected_boxes.map((box, index) => <BoxRect key={`p-${index}`} box={box} ly={ly} className="protected" />)}
        {structure.preferred_boxes.map((box, index) => <BoxRect key={`r-${index}`} box={box} ly={ly} className="preferred" />)}
      </svg>
      <div className="placement-summary">
        <span>strength {formatNumber(structure.strength, 2)}</span>
        <span>{placementMode === "none" ? "no placement conditioning" : `${placementMode} conditioning`}</span>
      </div>
    </div>
  );
}

function BoxRect({ box, ly, className }: { box: BoxSpec; ly: number; className: string }) {
  const x = Math.min(box.x[0], box.x[1]);
  const y0 = Math.min(box.y[0], box.y[1]);
  const w = Math.abs(box.x[1] - box.x[0]);
  const h = Math.abs(box.y[1] - box.y[0]);
  return <rect x={x} y={ly - y0 - h} width={w} height={h} className={`intent-box ${className}`} />;
}

function HeatLoadEditor({
  heatLoads,
  constraints,
  onChange,
}: {
  heatLoads: HeatLoadSpec;
  constraints: InverseConstraintSpec;
  onChange: (patch: Partial<HeatLoadSpec>) => void;
}) {
  const rows = Math.max(constraints.num_modules_max, constraints.num_modules_min, 1);
  const values = heatLoads.values?.length ? heatLoads.values : Array.from({ length: rows }, () => heatLoads.value ?? 1.2);
  return (
    <div className="heat-load-editor">
      <label>
        Heat mode
        <select value={heatLoads.mode} onChange={(event) => onChange({ mode: event.target.value as HeatLoadMode })}>
          <option value="per_module">exact per-module values</option>
          <option value="per_module_range">per-module ranges</option>
          <option value="uniform">shared exact value</option>
          <option value="uniform_range">shared range</option>
          <option value="total_only">total heat only</option>
          <option value="from_reference">from reference</option>
          <option value="none">none</option>
        </select>
      </label>
      {heatLoads.mode === "per_module" && (
        <div className="heat-list">
          {Array.from({ length: rows }, (_, index) => (
            <label key={index}>
              M{index + 1}
              <input
                type="number"
                step="0.05"
                value={(values[index] ?? 1.2).toString()}
                onChange={(event) => {
                  const next = Array.from({ length: rows }, (_, i) => values[i] ?? 1.2);
                  next[index] = Number(event.target.value);
                  onChange({ values: next });
                }}
              />
            </label>
          ))}
        </div>
      )}
      {heatLoads.mode === "uniform" && (
        <label>
          Shared heat q
          <input type="number" step="0.05" value={heatLoads.value ?? 1.2} onChange={(event) => onChange({ value: Number(event.target.value) })} />
        </label>
      )}
      {heatLoads.mode === "uniform_range" && (
        <div className="split-inputs">
          <label>q low<input type="number" step="0.05" value={heatLoads.range?.[0] ?? 0.8} onChange={(event) => onChange({ range: [Number(event.target.value), heatLoads.range?.[1] ?? 1.8] })} /></label>
          <label>q high<input type="number" step="0.05" value={heatLoads.range?.[1] ?? 1.8} onChange={(event) => onChange({ range: [heatLoads.range?.[0] ?? 0.8, Number(event.target.value)] })} /></label>
        </div>
      )}
      {heatLoads.mode === "total_only" && (
        <label>
          Total heat
          <input type="number" step="0.1" value={heatLoads.total ?? ""} onChange={(event) => onChange({ total: event.target.value === "" ? null : Number(event.target.value) })} />
        </label>
      )}
    </div>
  );
}

function CandidateMiniLayout({ candidate, heatLoads, config }: { candidate: InverseCandidate; heatLoads: HeatLoadSpec; config: ModelConfig | null }) {
  const lx = config?.domain_length_x ?? 12;
  const ly = config?.domain_length_y ?? 4;
  const r = config?.module_radius ?? 0.45;
  const heat = candidateHeat(candidate, heatLoads, config);
  return (
    <svg className="candidate-mini-layout" viewBox={`0 0 ${lx} ${ly}`}>
      <rect x="0" y="0" width={lx} height={ly} fill="#f8faf7" stroke="#d6ded7" strokeWidth="0.035" />
      {candidate.centers.slice(0, candidate.count).map(([x, y], index) => (
        <g key={index}>
          <circle cx={numeric(x)} cy={ly - numeric(y)} r={r} fill={heatColor(heat[index] ?? 1.2)} stroke="#111827" strokeWidth="0.045" />
          <text x={numeric(x)} y={ly - numeric(y) + 0.055} textAnchor="middle" fontSize="0.22" fontWeight="700">{index + 1}</text>
        </g>
      ))}
    </svg>
  );
}

function HypergraphDiagnostics({ candidate }: { candidate: InverseCandidate }) {
  const artifacts = candidate.artifacts ?? {};
  const links: Array<[string, string | undefined]> = [
    ["Overlay", artifacts.hypergraph_overlay],
    ["Mismatch heatmap", artifacts.hypergraph_mismatch_heatmap],
    ["Edge table", artifacts.hypergraph_edge_table],
    ["Planned JSON", artifacts.hypergraph_planned],
    ["Realized JSON", artifacts.hypergraph_realized],
  ];
  const availableLinks = links.filter((item): item is [string, string] => Boolean(item[1]));
  if (!candidate.hypergraph_diagnostics_available && !availableLinks.length) return null;
  return (
    <details className="hypergraph-diagnostics">
      <summary>Hypergraph diagnostics</summary>
      <div className="hypergraph-metrics">
        <span>consistency {formatNumber(candidate.hypergraph_consistency_score, 3)}</span>
        <span>active edge error {formatNumber(candidate.hypergraph_active_count_error, 3)}</span>
        <span>source RMSE {formatNumber(candidate.hypergraph_source_rmse, 3)}</span>
        <span>thermal RMSE {formatNumber(candidate.hypergraph_thermal_region_rmse, 3)}</span>
        <span>A_mh L1 {formatNumber(candidate.hypergraph_A_mh_l1, 3)}</span>
      </div>
      {availableLinks.length > 0 && (
        <div className="hypergraph-links">
          {availableLinks.map(([label, url]) => (
            <a key={label} href={apiUrl(url)} target="_blank" rel="noreferrer">{label}</a>
          ))}
        </div>
      )}
      {artifacts.hypergraph_overlay && <img className="hypergraph-thumb" src={apiUrl(artifacts.hypergraph_overlay)} alt="hypergraph overlay" />}
    </details>
  );
}

function CandidateGallery({
  inverseResult,
  heatLoads,
  config,
  debugFiles,
  onUse,
}: {
  inverseResult: InverseResult | null;
  heatLoads: HeatLoadSpec;
  config: ModelConfig | null;
  debugFiles: DebugFile[];
  onUse: (candidate: InverseCandidate) => void;
}) {
  const candidates = inverseResult?.candidates ?? [];
  return (
    <aside className="panel candidate-gallery">
      <div className="panel-heading">
        <h2>Candidate Gallery</h2>
        <span>{inverseResult?.status ?? "waiting"} / {candidates.length}</span>
      </div>
      {candidates.length ? (
        candidates.slice(0, 12).map((candidate) => {
          const heat = candidateHeat(candidate, heatLoads, config);
          return (
            <article key={`${candidate.rank}-${candidate.sample_index}`} className="candidate-card">
              <CandidateMiniLayout candidate={candidate} heatLoads={heatLoads} config={config} />
              <div className="candidate-meta">
                <strong>#{candidate.rank + 1} score {formatNumber(candidate.total_score, 3)}</strong>
                <span>{candidate.count} modules / heat total {formatNumber(heat.reduce((sum, item) => sum + item, 0), 3)}</span>
                <span>Tmax {formatNumber(candidate.verified_kpis?.max_solid_temperature, 3)} / dp {formatNumber(candidate.verified_kpis?.pressure_drop, 3)}</span>
              </div>
              <div className="candidate-actions">
                <button type="button" onClick={() => onUse(candidate)}><Flame size={15} /> Use in forward</button>
                {candidate.artifacts?.preview && <a className="button-link" href={apiUrl(candidate.artifacts.preview)} target="_blank" rel="noreferrer">View details</a>}
              </div>
              <HypergraphDiagnostics candidate={candidate} />
            </article>
          );
        })
      ) : (
        <div className="empty-list">
          {inverseResult?.status === "complete_with_no_candidates" ? "The run completed but no candidates were parsed." : "Run inverse to populate candidate cards."}
        </div>
      )}
      {!candidates.length && debugFiles.length > 0 && (
        <div className="debug-file-list">
          {debugFiles.slice(0, 12).map((file) => (
            <a key={file.path} href={file.url ? apiUrl(file.url) : undefined}>{file.path}</a>
          ))}
        </div>
      )}
    </aside>
  );
}

const PRIMARY_INVERSE_ARTIFACTS = [
  { key: "plots_fields_best_layout_global_fields", label: "Global field distribution" },
  { key: "plots_fields_best_layout_module_internal_disks", label: "Module-internal temperature" },
  { key: "plots_fields_best_layout_interface_curves", label: "Inter-module interface curves" },
];

function InverseResultPanel({
  inverseResult,
  heatLoads,
  config,
  onUse,
}: {
  inverseResult: InverseResult | null;
  heatLoads: HeatLoadSpec;
  config: ModelConfig | null;
  onUse: (candidate: InverseCandidate) => void;
}) {
  const artifacts = inverseResult?.artifacts ?? {};
  const primaryKeys = new Set(PRIMARY_INVERSE_ARTIFACTS.map((item) => item.key));
  const primaryArtifacts = PRIMARY_INVERSE_ARTIFACTS.map((item) => ({ ...item, url: artifacts[item.key] })).filter((item) => item.url);
  const extraArtifacts = Object.entries(artifacts).filter(([name]) => !primaryKeys.has(name));
  const best = inverseResult?.candidates?.[0] ?? null;
  const heat = best ? candidateHeat(best, heatLoads, config) : [];

  return (
    <section className="panel inverse-results">
      <div className="panel-heading">
        <h2>Inferred Distributions</h2>
        <span>{inverseResult ? `score ${formatNumber(inverseResult.summary?.best_score, 3)}` : "run inverse"}</span>
      </div>
      {inverseResult ? (
        <>
          {best && (
            <div className="inverse-best-summary">
              <div>
                <strong>Best candidate #{best.rank + 1}</strong>
                <span>{best.count} modules / heat total {formatNumber(heat.reduce((sum, value) => sum + value, 0), 3)}</span>
              </div>
              <button type="button" onClick={() => onUse(best)}><Flame size={15} /> Use in forward</button>
            </div>
          )}
          {primaryArtifacts.length ? (
            <div className="inverse-primary-artifacts">
              {primaryArtifacts.map((artifact) => (
                <a key={artifact.key} href={apiUrl(artifact.url!)} target="_blank" rel="noreferrer">
                  <img src={apiUrl(artifact.url!)} alt={artifact.label} />
                  <span>{artifact.label}</span>
                </a>
              ))}
            </div>
          ) : (
            <div className="empty-list">The inverse run completed, but no field distribution artifacts were returned.</div>
          )}
          <details className="inverse-extra-artifacts">
            <summary>Additional diagnostics and logs</summary>
            {inverseResult.summary?.best_design_intent_score_breakdown != null && <pre className="score-json">{JSON.stringify(inverseResult.summary.best_design_intent_score_breakdown, null, 2).slice(0, 1800)}</pre>}
            <div className="artifact-grid compact">
              {extraArtifacts.map(([name, url]) => (
                <a key={name} href={apiUrl(url)} target="_blank" rel="noreferrer">
                  {url.endsWith(".png") ? <img src={apiUrl(url)} alt={name.replaceAll("_", " ")} /> : <span className="file-chip">{name}</span>}
                  <span>{name.replaceAll("_", " ")}</span>
                </a>
              ))}
            </div>
            {inverseResult.stdout_tail?.length ? <pre className="log-tail">{inverseResult.stdout_tail.join("\n")}</pre> : null}
            {inverseResult.stderr_tail?.length ? <pre className="log-tail">{inverseResult.stderr_tail.join("\n")}</pre> : null}
          </details>
        </>
      ) : (
        <div className="empty-stage"><Target size={30} /></div>
      )}
    </section>
  );
}

export default function App() {
  const [demoMode, setDemoMode] = useState<DemoMode>("forward");
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [inverseModels, setInverseModels] = useState<InverseModelEntry[]>([]);
  const [presets, setPresets] = useState<TargetPreset[]>([]);
  const [kpiInfos, setKpiInfos] = useState<KpiInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [selectedInverseModelId, setSelectedInverseModelId] = useState("");
  const [config, setConfig] = useState<ModelConfig | null>(null);
  const [referenceCases, setReferenceCases] = useState<ReferenceCase[]>([]);
  const [design, setDesign] = useState<DesignRequest>(() => makeDesign("", null, null));
  const [flowMode, setFlowMode] = useState<FlowMode>("reference");
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [selectedModule, setSelectedModule] = useState<number | null>(0);
  const [result, setResult] = useState<JobResult | null>(null);
  const [simulationJobId, setSimulationJobId] = useState<string | null>(null);
  const [simulationStatus, setSimulationStatus] = useState<Record<string, unknown> | null>(null);
  const [simulationResult, setSimulationResult] = useState<ForwardSimulationResult | null>(null);
  const [simulationRunning, setSimulationRunning] = useState(false);
  const [expandedField, setExpandedField] = useState<FieldName | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [candidateLoaded, setCandidateLoaded] = useState(false);

  const [selectedPresetName, setSelectedPresetName] = useState("");
  const [targetMode, setTargetMode] = useState<"design_intent" | "legacy_kpi">("design_intent");
  const [countMode, setCountMode] = useState<CountMode>("range");
  const [placementMode, setPlacementMode] = useState<PlacementMode>("quantitative");
  const [sketchTool, setSketchTool] = useState<SketchTool>("preferred");
  const [kpiTargets, setKpiTargets] = useState<KpiTargetSpec[]>([]);
  const [constraints, setConstraints] = useState<InverseConstraintSpec>(DEFAULT_CONSTRAINTS);
  const [heatLoads, setHeatLoads] = useState<HeatLoadSpec>(DEFAULT_HEAT_LOADS);
  const [structure, setStructure] = useState<StructureConstraintSpec>(DEFAULT_STRUCTURE);
  const [thermalLimits, setThermalLimits] = useState<ThermalLimitSpec>(DEFAULT_LIMITS);
  const [objectiveWeights, setObjectiveWeights] = useState<ObjectiveWeightsSpec>(DEFAULT_WEIGHTS);
  const [fieldPreferences, setFieldPreferences] = useState<Record<string, unknown>>(fieldPrefsFromPreset(null));
  const [sampling, setSampling] = useState<InverseSamplingSpec>(DEFAULT_SAMPLING);
  const [guidanceScale, setGuidanceScale] = useState(1.0);
  const [diversityWeight, setDiversityWeight] = useState(0.15);
  const [poolMultiplier, setPoolMultiplier] = useState(1.0);
  const [inverseJobId, setInverseJobId] = useState<string | null>(null);
  const [inverseStatus, setInverseStatus] = useState<Record<string, unknown> | null>(null);
  const [inverseResult, setInverseResult] = useState<InverseResult | null>(null);
  const [inverseError, setInverseError] = useState<string | null>(null);
  const [inverseRunning, setInverseRunning] = useState(false);
  const [debugFiles, setDebugFiles] = useState<DebugFile[]>([]);

  const selectedPreset = useMemo(() => presets.find((item) => item.name === selectedPresetName) ?? null, [presets, selectedPresetName]);
  const selectedModel = useMemo(() => models.find((item) => item.id === selectedModelId) ?? null, [models, selectedModelId]);
  const selectedReference = useMemo(() => referenceCases.find((item) => item.index === design.reference_case_index) ?? referenceCases[0] ?? null, [design.reference_case_index, referenceCases]);

  const applyPreset = useCallback(
    (preset: TargetPreset | null, kpis = kpiInfos, nextConfig = config) => {
      setTargetMode(preset?.target_mode ?? "legacy_kpi");
      setKpiTargets(kpiRowsFromPreset(preset, kpis));
      const nextConstraints = constraintsFromPreset(preset, nextConfig);
      setConstraints(nextConstraints);
      setCountMode(nextConstraints.num_modules_min === nextConstraints.num_modules_max ? "exact" : "range");
      setHeatLoads(heatLoadsFromPreset(preset));
      const nextStructure = structureFromPreset(preset, nextConfig);
      setStructure(nextStructure);
      setPlacementMode(preset?.target_mode === "legacy_kpi" ? "none" : nextStructure.sketch_maps ? "sketch" : nextStructure.enabled ? "quantitative" : "none");
      setThermalLimits(limitsFromPreset(preset));
      setObjectiveWeights(weightsFromPreset(preset));
      setFieldPreferences(fieldPrefsFromPreset(preset));
      setInverseResult(null);
      setDebugFiles([]);
      setInverseError(null);
    },
    [config, kpiInfos],
  );

  useEffect(() => {
    let cancelled = false;
    Promise.all([getModels(), getInverseModels(), getTargetPresets(), getInverseKpis()])
      .then(([modelItems, inverseItems, presetItems, kpiItems]) => {
        if (cancelled) return;
        setModels(modelItems);
        setInverseModels(inverseItems);
        setPresets(presetItems);
        setKpiInfos(kpiItems);
        const firstModel = modelItems.find((item) => item.enabled && item.available) ?? modelItems[0];
        const firstInverse = inverseItems.find((item) => item.enabled && item.available) ?? inverseItems[0];
        const firstPreset = presetItems.find((item) => item.source_dir === "inverse_targets_v2" && item.target_mode === "design_intent") ?? presetItems.find((item) => item.target_mode === "design_intent") ?? presetItems[0];
        if (firstModel) setSelectedModelId(firstModel.id);
        if (firstInverse) setSelectedInverseModelId(firstInverse.id);
        if (firstPreset) {
          setSelectedPresetName(firstPreset.name);
          applyPreset(firstPreset, kpiItems, null);
        } else {
          setKpiTargets(kpiRowsFromPreset(null, kpiItems));
        }
      })
      .catch((error: Error) => setLoadError(error.message));
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedModelId) return;
    let cancelled = false;
    getModelConfig(selectedModelId)
      .then(async (nextConfig) => {
        const refs = await getReferenceCases(nextConfig.default_reference_split, 100);
        if (cancelled) return;
        const ref = refs.find((item) => item.index === nextConfig.default_reference_case_index) ?? refs[0] ?? null;
        setConfig(nextConfig);
        setReferenceCases(refs);
        setDesign(makeDesign(selectedModelId, nextConfig, ref));
        setSelectedModule(0);
        setResult(null);
        setLoadError(null);
        if (selectedPreset) applyPreset(selectedPreset, kpiInfos, nextConfig);
      })
      .catch((error: Error) => setLoadError(error.message));
    return () => {
      cancelled = true;
    };
  }, [selectedModelId]);

  useEffect(() => {
    if (!design.model_id) return;
    let cancelled = false;
    const timer = window.setTimeout(() => {
      validateDesign(design)
        .then((payload) => {
          if (!cancelled) setValidation(payload);
        })
        .catch((error: Error) => {
          if (!cancelled) {
            setValidation(null);
            setRunError(error.message);
          }
        });
    }, 180);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [design]);

  useEffect(() => {
    if (!inverseJobId || !inverseRunning) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const status = await getInverseStatus(inverseJobId);
        if (cancelled) return;
        setInverseStatus(status);
        if (status.status === "complete" || status.status === "complete_with_no_candidates") {
          let payload = await getInverseResult(inverseJobId);
          if (!payload.candidates?.length) {
            const candidatePayload = await getInverseCandidates(inverseJobId);
            payload = { ...payload, candidates: candidatePayload.candidates, candidate_count: candidatePayload.candidates.length };
          }
          const debug = !payload.candidates?.length ? await getInverseDebugFiles(inverseJobId).catch(() => ({ files: [] as DebugFile[] })) : { files: [] as DebugFile[] };
          if (!cancelled) {
            setInverseResult(payload);
            setDebugFiles(debug.files);
            setInverseRunning(false);
          }
        } else if (status.status === "failed") {
          setInverseError(String(status.error ?? "Inverse job failed."));
          setInverseRunning(false);
        }
      } catch (error) {
        if (!cancelled) {
          setInverseError(error instanceof Error ? error.message : String(error));
          setInverseRunning(false);
        }
      }
    };
    void poll();
    const timer = window.setInterval(poll, 1500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [inverseJobId, inverseRunning]);

  useEffect(() => {
    if (!simulationJobId || !simulationRunning) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const status = await getForwardSimulationStatus(simulationJobId);
        if (cancelled) return;
        setSimulationStatus(status);
        if (status.status === "complete") {
          const payload = await getForwardSimulationResult(simulationJobId);
          if (!cancelled) {
            setSimulationResult(payload);
            setSimulationRunning(false);
          }
        } else if (status.status === "failed") {
          setRunError(String(status.error ?? "Simulation verification failed."));
          setSimulationRunning(false);
        }
      } catch (error) {
        if (!cancelled) {
          setRunError(error instanceof Error ? error.message : String(error));
          setSimulationRunning(false);
        }
      }
    };
    void poll();
    const timer = window.setInterval(poll, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [simulationJobId, simulationRunning]);

  const displayKpis = useMemo(() => {
    const kpis = result?.kpis ?? null;
    if (!kpis) return [];
    const numericEntries = Object.entries(kpis).filter(([, value]) => typeof value === "number" && Number.isFinite(value));
    const byName = new Map(numericEntries);
    const ordered = KPI_PRIORITY.filter((name) => byName.has(name)).map((name) => [name, byName.get(name)] as [string, unknown]);
    const rest = numericEntries.filter(([name]) => !KPI_PRIORITY.includes(name)).slice(0, 10);
    return [...ordered, ...rest].slice(0, 18);
  }, [result]);

  const updateDesign = useCallback((patch: Partial<DesignRequest>) => {
    setDesign((current) => ({ ...current, ...patch }));
    setRunError(null);
  }, []);

  const updateModule = useCallback(
    (index: number, patch: Partial<ThermalModule>) => {
      setDesign((current) => {
        const modules = current.modules.map((item, itemIndex) => (itemIndex === index ? clampModule({ ...item, ...patch }, config) : item));
        return { ...current, design_source: "diy", return_ground_truth: false, return_error: false, modules };
      });
      setResult(null);
      setSimulationResult(null);
      setRunError(null);
    },
    [config],
  );

  const addModule = useCallback(() => {
    setDesign((current) => {
      const max = config?.max_num_modules ?? 12;
      if (current.modules.length >= max) return current;
      const lx = config?.domain_length_x ?? 12;
      const ly = config?.domain_length_y ?? 4;
      const heat = config?.default_heat_power ?? 1.2;
      const next = clampModule({ x: lx * (0.20 + 0.10 * current.modules.length), y: ly * (0.35 + 0.18 * (current.modules.length % 2)), heat_power: heat }, config);
      setSelectedModule(current.modules.length);
      return { ...current, design_source: "diy", return_ground_truth: false, return_error: false, modules: [...current.modules, next] };
    });
    setResult(null);
    setSimulationResult(null);
  }, [config]);

  const deleteModule = useCallback((index: number) => {
    setDesign((current) => ({ ...current, design_source: "diy", return_ground_truth: false, return_error: false, modules: current.modules.filter((_, itemIndex) => itemIndex !== index) }));
    setSelectedModule((current) => (current === null ? null : Math.max(0, current - (current >= index ? 1 : 0))));
    setResult(null);
    setSimulationResult(null);
  }, []);

  const resetDesign = useCallback(() => {
    setDesign(makeDesign(selectedModelId, config, selectedReference));
    setFlowMode("reference");
    setSelectedModule(0);
    setResult(null);
    setCandidateLoaded(false);
    setRunError(null);
  }, [config, selectedModelId, selectedReference]);

  const randomizeDesign = useCallback(() => {
    setDesign((current) => ({ ...current, design_source: "diy", return_ground_truth: false, return_error: false, modules: randomModules(Math.max(3, Math.min(current.modules.length || 4, config?.max_num_modules ?? 12)), config) }));
    setSelectedModule(0);
    setResult(null);
    setSimulationResult(null);
  }, [config]);

  const handleReferenceChange = useCallback(
    (index: number) => {
      const ref = referenceCases.find((item) => item.index === index) ?? null;
      setDesign((current) => ({ ...current, reference_case_index: index, reference_case_id: null, design_source: "reference_case", re: null, u_in: null, return_ground_truth: true, return_error: true, modules: modulesFromReference(ref, config) }));
      setFlowMode("reference");
      setSelectedModule(0);
      setResult(null);
      setSimulationResult(null);
    },
    [config, referenceCases],
  );

  const handleFlowModeChange = useCallback(
    (mode: FlowMode) => {
      setFlowMode(mode);
      updateDesign({
        design_source: mode === "reference" ? "reference_case" : "diy",
        re: mode === "re" ? selectedReference?.re ?? config?.reference_case.re ?? null : null,
        u_in: mode === "u_in" ? selectedReference?.u_in ?? config?.reference_case.u_in ?? null : null,
        return_ground_truth: mode === "reference",
        return_error: mode === "reference",
      });
    },
    [config?.reference_case.re, config?.reference_case.u_in, selectedReference?.re, selectedReference?.u_in, updateDesign],
  );

  const handleRun = useCallback(async () => {
    if (!selectedModel || !validation?.valid) return;
    setIsRunning(true);
    setRunError(null);
    try {
      const response = await runInference(design);
      const payload = await getJobResult(response.job_id);
      setResult(payload);
      setSimulationResult(null);
      setSimulationStatus(null);
      setCandidateLoaded(false);
    } catch (error) {
      setRunError(error instanceof Error ? error.message : String(error));
    } finally {
      setIsRunning(false);
    }
  }, [design, selectedModel, validation]);

  const handleRunSimulation = useCallback(async () => {
    if (!result || !validation?.valid) return;
    setSimulationRunning(true);
    setRunError(null);
    setSimulationResult(null);
    try {
      const response = await runForwardSimulation({
        design: { ...design, return_ground_truth: false, return_error: false },
        prediction_job_id: result.job_id,
        max_runtime_seconds: 900,
      });
      setSimulationJobId(response.job_id);
      setSimulationStatus({ ...response });
    } catch (error) {
      setRunError(error instanceof Error ? error.message : String(error));
      setSimulationRunning(false);
    }
  }, [design, result, validation]);

  const useCandidateInForward = useCallback(
    (candidate: InverseCandidate) => {
      const heat = candidateHeat(candidate, heatLoads, config);
      const modules = candidate.centers.slice(0, candidate.count).map(([x, y], index) =>
        clampModule({ x: numeric(x), y: numeric(y), heat_power: heat[index] ?? config?.default_heat_power ?? 1.2 }, config),
      );
      setDemoMode("forward");
      setDesign((current) => ({ ...current, model_id: selectedModelId, reference_case_id: null, design_source: "candidate", return_ground_truth: false, return_error: false, modules }));
      setSelectedModule(0);
      setResult(null);
      setSimulationResult(null);
      setCandidateLoaded(true);
      setRunError(null);
    },
    [config, heatLoads, selectedModelId],
  );

  const setAllHeat = useCallback(() => {
    const value = Number(window.prompt("Set all module heat powers to:", String(config?.default_heat_power ?? 1.2)));
    if (!Number.isFinite(value)) return;
    setDesign((current) => ({ ...current, design_source: "diy", return_ground_truth: false, return_error: false, modules: current.modules.map((module) => clampModule({ ...module, heat_power: value }, config)) }));
  }, [config]);

  const randomHeat = useCallback(() => {
    const min = config?.heat_power_min ?? 0.6;
    const max = config?.heat_power_max ?? 2.2;
    setDesign((current) => ({ ...current, design_source: "diy", return_ground_truth: false, return_error: false, modules: current.modules.map((module) => ({ ...module, heat_power: min + Math.random() * Math.max(max - min, 0.1) })) }));
  }, [config]);

  const normalizeHeat = useCallback(() => {
    const currentTotal = design.modules.reduce((sum, item) => sum + item.heat_power, 0);
    const target = Number(window.prompt("Normalize total heat to:", currentTotal.toFixed(3)));
    if (!Number.isFinite(target) || currentTotal <= 0) return;
    const scale = target / currentTotal;
    setDesign((current) => ({ ...current, design_source: "diy", return_ground_truth: false, return_error: false, modules: current.modules.map((module) => clampModule({ ...module, heat_power: module.heat_power * scale }, config)) }));
  }, [config, design.modules]);

  const handlePresetChange = useCallback(
    (name: string) => {
      const preset = presets.find((item) => item.name === name) ?? null;
      setSelectedPresetName(name);
      applyPreset(preset);
    },
    [applyPreset, presets],
  );

  const updateKpiTarget = useCallback((index: number, patch: Partial<KpiTargetSpec>) => {
    setKpiTargets((current) => current.map((item, itemIndex) => (itemIndex === index ? { ...item, ...patch } : item)));
  }, []);

  const addKpiTarget = useCallback(() => {
    const existing = new Set(kpiTargets.map((item) => item.name));
    const next = kpiInfos.find((item) => !existing.has(item.name));
    if (!next) return;
    setKpiTargets((current) => [...current, { enabled: true, name: next.name, mode: next.default_mode, value: null, low: null, high: null, weight: next.default_weight }]);
  }, [kpiInfos, kpiTargets]);

  const deleteKpiTarget = useCallback((index: number) => {
    setKpiTargets((current) => current.filter((_, itemIndex) => itemIndex !== index));
  }, []);

  const handleCountMode = useCallback(
    (mode: CountMode) => {
      setCountMode(mode);
      if (mode === "exact") setConstraints((current) => ({ ...current, num_modules_max: current.num_modules_min }));
      if (mode === "unspecified") setConstraints((current) => ({ ...current, num_modules_min: 1, num_modules_max: config?.max_num_modules ?? 12 }));
    },
    [config?.max_num_modules],
  );

  const handlePlacementMode = useCallback((mode: PlacementMode) => {
    setPlacementMode(mode);
    setStructure((current) => ({ ...current, enabled: mode !== "none", sketch_maps: mode === "sketch" ? current.sketch_maps ?? { preferred: emptySketch(), keepout: emptySketch(), protected: emptySketch(), reference_soft: emptySketch() } : current.sketch_maps }));
    setFieldPreferences((current) => ({ ...current, placement_mode: mode }));
  }, []);

  const handleInverseRun = useCallback(async () => {
    if (!selectedInverseModelId || !selectedModelId) return;
    const effectiveStructure: StructureConstraintSpec = { ...structure, enabled: placementMode !== "none" && structure.enabled };
    if (placementMode === "none") effectiveStructure.sketch_maps = null;
    setInverseRunning(true);
    setInverseError(null);
    setInverseResult(null);
    setInverseStatus(null);
    setDebugFiles([]);
    try {
      const response = await runInverse({
        inverse_model_id: selectedInverseModelId,
        forward_model_id: selectedModelId,
        target_name: selectedPreset?.name ?? "web_demo_target",
        target_mode: targetMode,
        kpis: kpiTargets,
        constraints,
        heat_loads: heatLoads,
        structure_constraints: effectiveStructure,
        thermal_limits: thermalLimits,
        objective_weights: objectiveWeights,
        field_preferences: fieldPreferences,
        preferences: {
          x_span: effectiveStructure.x_span,
          y_span: effectiveStructure.y_span,
          min_x_coverage: effectiveStructure.min_x_coverage,
          min_y_coverage: effectiveStructure.min_y_coverage,
          min_mean_pair_distance: effectiveStructure.min_mean_pair_distance,
          placement_mode: placementMode,
        },
        sampling,
        guidance_scale: guidanceScale,
        diversity_rerank_weight: diversityWeight,
        candidate_pool_multiplier: poolMultiplier,
        reference_split: design.reference_split,
        reference_case_index: design.reference_case_index,
      });
      setInverseJobId(response.job_id);
      setInverseStatus({ status: response.status, job_id: response.job_id });
    } catch (error) {
      setInverseError(error instanceof Error ? error.message : String(error));
      setInverseRunning(false);
    }
  }, [
    constraints,
    design.reference_case_index,
    design.reference_split,
    diversityWeight,
    fieldPreferences,
    guidanceScale,
    heatLoads,
    kpiTargets,
    objectiveWeights,
    placementMode,
    poolMultiplier,
    sampling,
    selectedInverseModelId,
    selectedModelId,
    selectedPreset?.name,
    structure,
    targetMode,
    thermalLimits,
  ]);

  const exportUrl = result?.export_npz_url ? apiUrl(result.export_npz_url) : null;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>ChannelThermal</h1>
          <span>{selectedModel?.label ?? "Thermal design demo"}</span>
        </div>
        <div className="mode-toggle" aria-label="Demo mode selector">
          <button type="button" className={demoMode === "forward" ? "active" : ""} onClick={() => setDemoMode("forward")}><Flame size={16} />Forward</button>
          <button type="button" className={demoMode === "inverse" ? "active" : ""} onClick={() => setDemoMode("inverse")}><Target size={16} />Inverse</button>
        </div>
      </header>

      {demoMode === "forward" ? (
        <div className="forward-page">
          <section className="design-editor-panel panel">
            <div className="panel-heading">
              <h2>Design Domain</h2>
              <div className="model-strip">
                <select value={selectedModelId} onChange={(event) => setSelectedModelId(event.target.value)}>{models.map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}</select>
                <select value={design.reference_case_index} onChange={(event) => handleReferenceChange(Number(event.target.value))}>{referenceCases.map((ref) => <option key={ref.case_id} value={ref.index}>{ref.case_id} / {ref.num_modules} modules</option>)}</select>
                <button type="button" className="icon-button" onClick={resetDesign} title="Reset"><RefreshCcw size={16} /></button>
              </div>
            </div>
            <div className="flow-control-row">
              <label>
                Flow condition
                <select value={flowMode} onChange={(event) => handleFlowModeChange(event.target.value as FlowMode)}>
                  <option value="reference">reference</option>
                  <option value="u_in">u_in override</option>
                  <option value="re">Re override</option>
                </select>
              </label>
              {flowMode === "u_in" && <label>u_in<input type="number" value={design.u_in ?? ""} onChange={(event) => updateDesign({ design_source: "diy", u_in: Number(event.target.value), re: null, return_ground_truth: false, return_error: false })} /></label>}
              {flowMode === "re" && <label>Re<input type="number" value={design.re ?? ""} onChange={(event) => updateDesign({ design_source: "diy", re: Number(event.target.value), u_in: null, return_ground_truth: false, return_error: false })} /></label>}
              <small>The surrogate was trained on dataset flow conditions; override only for diagnostic exploration.</small>
            </div>
            <DesignDomainEditor modules={design.modules} config={config} validation={validation} selected={selectedModule} onSelect={setSelectedModule} onChange={updateModule} />
            <ModuleHeatTable
              modules={design.modules}
              config={config}
              selected={selectedModule}
              onSelect={setSelectedModule}
              onChange={updateModule}
              onAdd={addModule}
              onDelete={deleteModule}
              onSetAll={setAllHeat}
              onRandomHeat={randomHeat}
              onNormalize={normalizeHeat}
            />
          </section>

          <aside className="panel run-panel">
            <div className={`status-box ${validation?.valid ? "valid" : "invalid"}`}>
              <strong>{validation?.valid ? "Ready" : "Needs attention"}</strong>
              <span>{design.modules.length} modules / heat {formatNumber(validation?.total_heat_power, 3)}</span>
              <small>{design.design_source === "reference_case" ? "Ground truth comparison enabled for this test-set case." : "DIY/candidate design: inferred fields only until simulation verification is launched."}</small>
              {candidateLoaded && <button type="button" onClick={handleRun} disabled={!validation?.valid || isRunning}>Run now</button>}
              {validation?.warnings.slice(0, 4).map((warning) => <small key={warning}>{warning}</small>)}
            </div>
            <button type="button" className="primary-action" onClick={handleRun} disabled={!validation?.valid || isRunning || !selectedModel?.available}>
              {isRunning ? <Loader2 className="spin" size={18} /> : <Flame size={18} />} Run forward
            </button>
            <button type="button" onClick={randomizeDesign}><Shuffle size={16} /> Random layout</button>
            {(runError ?? loadError) && <div className="error-text">{runError ?? loadError}</div>}
          </aside>

          <FieldGrid result={result} simulation={simulationResult} expandedField={expandedField} onExpand={setExpandedField} />
          <ModuleTemperatureGrid result={result} simulation={simulationResult} />
          <OrganizerDomainView result={result} />
          <KpiPanel
            result={result}
            simulation={simulationResult}
            simulationStatus={simulationStatus}
            simulationRunning={simulationRunning}
            displayKpis={displayKpis}
            exportUrl={exportUrl}
            onRunSimulation={handleRunSimulation}
          />
        </div>
      ) : (
        <div className="inverse-page">
          <aside className="panel inverse-controls">
            <div className="panel-heading"><h2>Target + Model</h2><span className={`mode-badge ${targetMode}`}>{targetMode === "design_intent" ? "Design intent" : "Legacy KPI"}</span></div>
            <label>Preset<select value={selectedPresetName} onChange={(event) => handlePresetChange(event.target.value)}>{presets.map((preset) => <option key={preset.name} value={preset.name}>{preset.source_dir}: {preset.label}</option>)}</select></label>
            <label>Inverse model<select value={selectedInverseModelId} onChange={(event) => setSelectedInverseModelId(event.target.value)}>{inverseModels.map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}</select></label>
            <label>Forward verifier<select value={selectedModelId} onChange={(event) => setSelectedModelId(event.target.value)}>{models.map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}</select></label>
            <div className="split-inputs">
              <label>samples<input type="number" value={sampling.n_samples} onChange={(event) => setSampling((s) => ({ ...s, n_samples: Number(event.target.value) }))} /></label>
              <label>steps<input type="number" value={sampling.n_steps} onChange={(event) => setSampling((s) => ({ ...s, n_steps: Number(event.target.value) }))} /></label>
            </div>
            <div className="split-inputs">
              <label>seed<input type="number" value={sampling.seed} onChange={(event) => setSampling((s) => ({ ...s, seed: Number(event.target.value) }))} /></label>
              <label>count<select value={sampling.count_mode} onChange={(event) => setSampling((s) => ({ ...s, count_mode: event.target.value as InverseSamplingSpec["count_mode"] }))}><option value="uniform">uniform</option><option value="sample">sample</option><option value="argmax">argmax</option></select></label>
            </div>
            <div className="split-inputs">
              <label>guidance<input type="number" step="0.1" value={guidanceScale} onChange={(event) => setGuidanceScale(Number(event.target.value))} /></label>
              <label>diversity<input type="number" step="0.05" value={diversityWeight} onChange={(event) => setDiversityWeight(Number(event.target.value))} /></label>
            </div>
            <label>candidate pool multiplier<input type="number" step="0.25" value={poolMultiplier} onChange={(event) => setPoolMultiplier(Number(event.target.value))} /></label>
            <button type="button" className="primary-action" onClick={handleInverseRun} disabled={inverseRunning || !selectedInverseModelId || !selectedModelId}>{inverseRunning ? <Loader2 className="spin" size={18} /> : <Wand2 size={18} />} Run inverse</button>
            {inverseStatus && <div className="status-box valid"><strong>{String(inverseStatus.status ?? "queued")}</strong><span>{inverseJobId}</span></div>}
            {inverseError && <div className="error-text">{inverseError}</div>}
          </aside>

          <section className="panel intent-builder">
            <div className="panel-heading"><h2>Design Intent Builder</h2><span>{selectedPreset?.name ?? "custom"}</span></div>
            <section className="intent-section">
              <h3>Module Count + Heat Loads</h3>
              <div className="segmented compact">{(["exact", "range", "unspecified"] as CountMode[]).map((mode) => <button key={mode} type="button" className={countMode === mode ? "active" : ""} onClick={() => handleCountMode(mode)}>{mode}</button>)}</div>
              <div className="split-inputs">
                <label>min modules<input type="number" value={constraints.num_modules_min} onChange={(event) => setConstraints((c) => ({ ...c, num_modules_min: Number(event.target.value), num_modules_max: countMode === "exact" ? Number(event.target.value) : c.num_modules_max }))} /></label>
                <label>max modules<input type="number" value={constraints.num_modules_max} disabled={countMode === "exact"} onChange={(event) => setConstraints((c) => ({ ...c, num_modules_max: Number(event.target.value) }))} /></label>
              </div>
              <HeatLoadEditor heatLoads={heatLoads} constraints={constraints} onChange={(patch) => setHeatLoads((current) => ({ ...current, ...patch }))} />
            </section>
            <section className="intent-section">
              <h3>Placement / Structure Conditioning</h3>
              <PlacementIntentCanvas
                structure={structure}
                placementMode={placementMode}
                sketchTool={sketchTool}
                config={config}
                onMode={handlePlacementMode}
                onTool={setSketchTool}
                onStructure={(patch) => setStructure((current) => ({ ...current, ...patch }))}
              />
              {placementMode === "quantitative" && (
                <div className="quant-grid">
                  <label>x span low<input type="number" value={structure.x_span?.[0] ?? ""} onChange={(event) => setStructure((s) => ({ ...s, x_span: [Number(event.target.value), s.x_span?.[1] ?? 10.8] }))} /></label>
                  <label>x span high<input type="number" value={structure.x_span?.[1] ?? ""} onChange={(event) => setStructure((s) => ({ ...s, x_span: [s.x_span?.[0] ?? 1.0, Number(event.target.value)] }))} /></label>
                  <label>y span low<input type="number" value={structure.y_span?.[0] ?? ""} onChange={(event) => setStructure((s) => ({ ...s, y_span: [Number(event.target.value), s.y_span?.[1] ?? 3.35] }))} /></label>
                  <label>y span high<input type="number" value={structure.y_span?.[1] ?? ""} onChange={(event) => setStructure((s) => ({ ...s, y_span: [s.y_span?.[0] ?? 0.65, Number(event.target.value)] }))} /></label>
                  <label>min x coverage<input type="number" value={structure.min_x_coverage ?? ""} onChange={(event) => setStructure((s) => ({ ...s, min_x_coverage: event.target.value === "" ? null : Number(event.target.value) }))} /></label>
                  <label>min y coverage<input type="number" value={structure.min_y_coverage ?? ""} onChange={(event) => setStructure((s) => ({ ...s, min_y_coverage: event.target.value === "" ? null : Number(event.target.value) }))} /></label>
                  <label>mean pair distance<input type="number" value={structure.min_mean_pair_distance ?? ""} onChange={(event) => setStructure((s) => ({ ...s, min_mean_pair_distance: event.target.value === "" ? null : Number(event.target.value) }))} /></label>
                  <label>strength<input type="range" min="0" max="1" step="0.05" value={structure.strength} onChange={(event) => setStructure((s) => ({ ...s, strength: Number(event.target.value) }))} /></label>
                  <label className="checkbox-line"><input type="checkbox" checked={structure.avoid_vertical_stack} onChange={(event) => setStructure((s) => ({ ...s, avoid_vertical_stack: event.target.checked }))} /> avoid vertical stack</label>
                </div>
              )}
            </section>
            <section className="intent-section">
              <h3>Thermal Objectives</h3>
              <div className="quant-grid">
                {Object.entries(thermalLimits).map(([key, value]) => (
                  <label key={key}>{key.replaceAll("_", " ")}<input type="number" step="0.01" value={value ?? ""} onChange={(event) => setThermalLimits((current) => ({ ...current, [key]: event.target.value === "" ? null : Number(event.target.value) }))} /></label>
                ))}
              </div>
              <div className="weight-grid">
                {Object.entries(objectiveWeights).map(([key, value]) => (
                  <label key={key}>{key.replaceAll("_", " ")}<input type="range" min="0" max="1.5" step="0.05" value={value} onChange={(event) => setObjectiveWeights((current) => ({ ...current, [key]: Number(event.target.value) }))} /><span>{formatNumber(value, 2)}</span></label>
                ))}
              </div>
            </section>
            <details className="legacy-kpi-section" open={targetMode === "legacy_kpi"}>
              <summary>Legacy KPI targets</summary>
              <div className="target-table">
                <div className="target-header"><span>on</span><span>name</span><span>mode</span><span>low</span><span>high/value</span><span>w</span><span /></div>
                {kpiTargets.map((row, index) => (
                  <div className="target-row" key={`${row.name}-${index}`}>
                    <input type="checkbox" checked={row.enabled} onChange={(event) => updateKpiTarget(index, { enabled: event.target.checked })} />
                    <select value={row.name} onChange={(event) => updateKpiTarget(index, { name: event.target.value })}>{kpiInfos.map((item) => <option key={item.name} value={item.name}>{item.name}</option>)}</select>
                    <select value={row.mode} onChange={(event) => updateKpiTarget(index, { mode: event.target.value as KpiMode })}><option value="max">max</option><option value="min">min</option><option value="range">range</option><option value="exact">exact</option></select>
                    <input type="number" value={row.low ?? ""} onChange={(event) => updateKpiTarget(index, { low: event.target.value === "" ? null : Number(event.target.value) })} />
                    <input type="number" value={row.mode === "exact" ? row.value ?? "" : row.high ?? ""} onChange={(event) => updateKpiTarget(index, row.mode === "exact" ? { value: event.target.value === "" ? null : Number(event.target.value) } : { high: event.target.value === "" ? null : Number(event.target.value) })} />
                    <input type="number" value={row.weight} step="0.1" onChange={(event) => updateKpiTarget(index, { weight: Number(event.target.value) })} />
                    <button type="button" className="icon-button" onClick={() => deleteKpiTarget(index)} title="Delete KPI"><Trash2 size={15} /></button>
                  </div>
                ))}
                <button type="button" onClick={addKpiTarget}><Plus size={16} /> Add KPI</button>
              </div>
            </details>
          </section>

          <CandidateGallery inverseResult={inverseResult} heatLoads={heatLoads} config={config} debugFiles={debugFiles} onUse={useCandidateInForward} />

          <InverseResultPanel inverseResult={inverseResult} heatLoads={heatLoads} config={config} onUse={useCandidateInForward} />
        </div>
      )}
    </main>
  );
}

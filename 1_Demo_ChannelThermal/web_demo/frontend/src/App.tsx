import { useCallback, useEffect, useMemo, useState } from "react";
import type { PointerEvent } from "react";
import {
  ChevronDown,
  Download,
  Flame,
  Loader2,
  Plus,
  RefreshCcw,
  Shuffle,
  Target,
  Trash2,
  Wand2,
} from "lucide-react";
import {
  apiUrl,
  getInverseKpis,
  getInverseModels,
  getInverseResult,
  getInverseStatus,
  getJobResult,
  getModelConfig,
  getModels,
  getReferenceCases,
  getTargetPresets,
  runInference,
  runInverse,
  validateDesign,
} from "./api";
import type {
  DemoMode,
  DesignRequest,
  FieldName,
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
  ReferenceCase,
  TargetPreset,
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

function numeric(value: unknown, fallback = 0): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function formatNumber(value: unknown, digits = 4): string {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "na";
  if (Math.abs(parsed) >= 100 || Math.abs(parsed) < 0.01) return parsed.toExponential(2);
  return parsed.toFixed(digits).replace(/\.?0+$/, "");
}

function modulesFromReference(ref: ReferenceCase | null, config: ModelConfig | null): ThermalModule[] {
  if (ref?.modules?.length) {
    return ref.modules.map((item) => ({ ...item }));
  }
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
    re: null,
    u_in: null,
    modules: modulesFromReference(ref, config),
    field: "temperature",
    display_scale: 3,
    display_smoothing: true,
    render_interpolation: "bicubic",
    return_kpis: true,
    return_organization: true,
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
    const ok = modules.every((item) => Math.hypot(item.x - candidate.x, item.y - candidate.y) >= minDistance);
    if (ok) modules.push(candidate);
  }
  return modules.length ? modules : modulesFromReference(null, config);
}

function kpiRowsFromPreset(preset: TargetPreset | null, kpiInfos: KpiInfo[]): KpiTargetSpec[] {
  const target = preset?.target ?? {};
  const rawKpis = (target.kpis ?? {}) as Record<string, Record<string, unknown>>;
  const rows = Object.entries(rawKpis).map(([name, entry]) => {
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

function constraintsFromPreset(preset: TargetPreset | null): InverseConstraintSpec {
  const target = preset?.target ?? {};
  return {
    num_modules_min: numeric(target.num_modules_min, DEFAULT_CONSTRAINTS.num_modules_min),
    num_modules_max: numeric(target.num_modules_max, DEFAULT_CONSTRAINTS.num_modules_max),
    min_center_distance: numeric(target.min_center_distance, DEFAULT_CONSTRAINTS.min_center_distance),
    wall_clearance: numeric(target.wall_clearance, DEFAULT_CONSTRAINTS.wall_clearance),
    inlet_clearance: numeric(target.inlet_clearance, DEFAULT_CONSTRAINTS.inlet_clearance),
    outlet_clearance: numeric(target.outlet_clearance, DEFAULT_CONSTRAINTS.outlet_clearance),
    heat_power_total: target.heat_power_total == null ? null : numeric(target.heat_power_total),
  };
}

function LayoutCanvas({
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
  const ly = validation?.domain_length_y ?? config?.domain_length_y ?? 6;
  const radius = validation?.module_radius ?? config?.module_radius ?? 0.45;
  const heatMin = validation?.heat_power_min ?? config?.heat_power_min ?? 0;
  const heatMax = validation?.heat_power_max ?? config?.heat_power_max ?? 3;

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
      className="layout-canvas"
      viewBox={`0 0 ${lx} ${ly}`}
      role="img"
      onPointerMove={(event) => {
        if (dragging !== null) pointerToModule(event, dragging);
      }}
      onPointerUp={() => setDragging(null)}
      onPointerLeave={() => setDragging(null)}
    >
      <defs>
        <linearGradient id="heatGradient" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stopColor="#3c8fb6" />
          <stop offset="50%" stopColor="#d9a441" />
          <stop offset="100%" stopColor="#c83f4d" />
        </linearGradient>
      </defs>
      <rect x="0" y="0" width={lx} height={ly} rx="0.08" fill="#f5f4ef" />
      <line x1="0" y1={ly * 0.5} x2={lx} y2={ly * 0.5} stroke="#cfd6c8" strokeDasharray="0.16 0.16" strokeWidth="0.025" />
      <g opacity="0.28">
        {Array.from({ length: 7 }, (_, i) => (
          <line key={`x-${i}`} x1={(lx * i) / 6} y1="0" x2={(lx * i) / 6} y2={ly} stroke="#b9c2b1" strokeWidth="0.012" />
        ))}
        {Array.from({ length: 5 }, (_, i) => (
          <line key={`y-${i}`} x1="0" y1={(ly * i) / 4} x2={lx} y2={(ly * i) / 4} stroke="#b9c2b1" strokeWidth="0.012" />
        ))}
      </g>
      <text x={0.18} y={0.32} fontSize="0.22" fill="#516158">inlet</text>
      <text x={lx - 0.92} y={0.32} fontSize="0.22" fill="#516158">outlet</text>
      {modules.map((module, index) => {
        const t = Math.min(Math.max((module.heat_power - heatMin) / Math.max(heatMax - heatMin, 1e-6), 0), 1);
        const fill = `color-mix(in srgb, #3c8fb6 ${Math.round((1 - t) * 100)}%, #c83f4d)`;
        return (
          <g key={index}>
            <circle
              cx={module.x}
              cy={ly - module.y}
              r={radius}
              fill={fill}
              stroke={selected === index ? "#111827" : "#ffffff"}
              strokeWidth={selected === index ? 0.075 : 0.045}
              onPointerDown={(event) => {
                event.currentTarget.setPointerCapture(event.pointerId);
                setDragging(index);
                onSelect(index);
              }}
            />
            <text x={module.x} y={ly - module.y + 0.075} textAnchor="middle" fontSize="0.24" fill="#111827" pointerEvents="none">
              {index + 1}
            </text>
          </g>
        );
      })}
    </svg>
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
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [selectedModule, setSelectedModule] = useState<number | null>(0);
  const [result, setResult] = useState<JobResult | null>(null);
  const [field, setField] = useState<FieldName>("temperature");
  const [isRunning, setIsRunning] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  const [selectedPresetName, setSelectedPresetName] = useState("");
  const [kpiTargets, setKpiTargets] = useState<KpiTargetSpec[]>([]);
  const [constraints, setConstraints] = useState<InverseConstraintSpec>(DEFAULT_CONSTRAINTS);
  const [preferences, setPreferences] = useState<Record<string, unknown>>({});
  const [sampling, setSampling] = useState<InverseSamplingSpec>(DEFAULT_SAMPLING);
  const [inverseJobId, setInverseJobId] = useState<string | null>(null);
  const [inverseStatus, setInverseStatus] = useState<Record<string, unknown> | null>(null);
  const [inverseResult, setInverseResult] = useState<InverseResult | null>(null);
  const [inverseError, setInverseError] = useState<string | null>(null);
  const [inverseRunning, setInverseRunning] = useState(false);

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
        if (firstModel) setSelectedModelId(firstModel.id);
        if (firstInverse) setSelectedInverseModelId(firstInverse.id);
        const firstPreset = presetItems[0] ?? null;
        if (firstPreset) {
          setSelectedPresetName(firstPreset.name);
          setKpiTargets(kpiRowsFromPreset(firstPreset, kpiItems));
          setConstraints(constraintsFromPreset(firstPreset));
          setPreferences(((firstPreset.target.preferences ?? {}) as Record<string, unknown>) ?? {});
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
        const ref =
          refs.find((item) => item.index === nextConfig.default_reference_case_index) ??
          refs[0] ??
          null;
        setConfig(nextConfig);
        setReferenceCases(refs);
        setDesign(makeDesign(selectedModelId, nextConfig, ref));
        setSelectedModule(0);
        setResult(null);
        setField("temperature");
        setLoadError(null);
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
        if (status.status === "complete") {
          const payload = await getInverseResult(inverseJobId);
          if (!cancelled) {
            setInverseResult(payload);
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

  const selectedReference = useMemo(
    () => referenceCases.find((item) => item.index === design.reference_case_index) ?? referenceCases[0] ?? null,
    [design.reference_case_index, referenceCases],
  );

  const selectedModel = useMemo(() => models.find((item) => item.id === selectedModelId) ?? null, [models, selectedModelId]);

  const selectedPreset = useMemo(() => presets.find((item) => item.name === selectedPresetName) ?? null, [presets, selectedPresetName]);

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
        const modules = current.modules.map((item, itemIndex) =>
          itemIndex === index ? clampModule({ ...item, ...patch }, config) : item,
        );
        return { ...current, modules };
      });
      setRunError(null);
    },
    [config],
  );

  const addModule = useCallback(() => {
    setDesign((current) => {
      const max = config?.max_num_modules ?? 12;
      if (current.modules.length >= max) return current;
      const lx = config?.domain_length_x ?? 12;
      const ly = config?.domain_length_y ?? 6;
      const heat = config?.default_heat_power ?? 1.2;
      const next = clampModule(
        {
          x: lx * (0.20 + 0.10 * current.modules.length),
          y: ly * (0.35 + 0.18 * (current.modules.length % 2)),
          heat_power: heat,
        },
        config,
      );
      setSelectedModule(current.modules.length);
      return { ...current, modules: [...current.modules, next] };
    });
    setResult(null);
  }, [config]);

  const deleteModule = useCallback((index: number) => {
    setDesign((current) => ({ ...current, modules: current.modules.filter((_, itemIndex) => itemIndex !== index) }));
    setSelectedModule((current) => (current === null ? null : Math.max(0, current - (current >= index ? 1 : 0))));
    setResult(null);
  }, []);

  const resetDesign = useCallback(() => {
    setDesign(makeDesign(selectedModelId, config, selectedReference));
    setSelectedModule(0);
    setResult(null);
    setRunError(null);
  }, [config, selectedModelId, selectedReference]);

  const randomizeDesign = useCallback(() => {
    setDesign((current) => ({
      ...current,
      modules: randomModules(Math.max(3, Math.min(current.modules.length || 4, config?.max_num_modules ?? 12)), config),
    }));
    setSelectedModule(0);
    setResult(null);
  }, [config]);

  const handleReferenceChange = useCallback(
    (index: number) => {
      const ref = referenceCases.find((item) => item.index === index) ?? null;
      setDesign((current) => ({
        ...current,
        reference_case_index: index,
        reference_case_id: null,
        re: null,
        u_in: null,
        modules: modulesFromReference(ref, config),
      }));
      setSelectedModule(0);
      setResult(null);
    },
    [config, referenceCases],
  );

  const handleRun = useCallback(async () => {
    if (!selectedModel || !validation?.valid) return;
    setIsRunning(true);
    setRunError(null);
    try {
      const response = await runInference(design);
      const payload = await getJobResult(response.job_id);
      setResult(payload);
      setField((payload.fields.includes(design.field) ? design.field : payload.fields[0]) as FieldName);
    } catch (error) {
      setRunError(error instanceof Error ? error.message : String(error));
    } finally {
      setIsRunning(false);
    }
  }, [design, selectedModel, validation]);

  const handlePresetChange = useCallback(
    (name: string) => {
      const preset = presets.find((item) => item.name === name) ?? null;
      setSelectedPresetName(name);
      setKpiTargets(kpiRowsFromPreset(preset, kpiInfos));
      setConstraints(constraintsFromPreset(preset));
      setPreferences(((preset?.target.preferences ?? {}) as Record<string, unknown>) ?? {});
      setInverseResult(null);
      setInverseError(null);
    },
    [kpiInfos, presets],
  );

  const updateKpiTarget = useCallback((index: number, patch: Partial<KpiTargetSpec>) => {
    setKpiTargets((current) => current.map((item, itemIndex) => (itemIndex === index ? { ...item, ...patch } : item)));
  }, []);

  const addKpiTarget = useCallback(() => {
    const existing = new Set(kpiTargets.map((item) => item.name));
    const next = kpiInfos.find((item) => !existing.has(item.name));
    if (!next) return;
    setKpiTargets((current) => [
      ...current,
      {
        enabled: true,
        name: next.name,
        mode: next.default_mode,
        value: null,
        low: null,
        high: null,
        weight: next.default_weight,
      },
    ]);
  }, [kpiInfos, kpiTargets]);

  const deleteKpiTarget = useCallback((index: number) => {
    setKpiTargets((current) => current.filter((_, itemIndex) => itemIndex !== index));
  }, []);

  const handleInverseRun = useCallback(async () => {
    if (!selectedInverseModelId || !selectedModelId) return;
    setInverseRunning(true);
    setInverseError(null);
    setInverseResult(null);
    setInverseStatus(null);
    try {
      const response = await runInverse({
        inverse_model_id: selectedInverseModelId,
        forward_model_id: selectedModelId,
        target_name: selectedPreset?.name ?? "web_demo_target",
        kpis: kpiTargets,
        constraints,
        preferences,
        sampling,
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
    kpiTargets,
    preferences,
    sampling,
    selectedInverseModelId,
    selectedModelId,
    selectedPreset?.name,
  ]);

  const useCandidateInForward = useCallback(
    (candidate: InverseCandidate) => {
      const heatDefault = config?.default_heat_power ?? 1.2;
      const modules = candidate.centers.map(([x, y], index) =>
        clampModule(
          {
            x: numeric(x),
            y: numeric(y),
            heat_power: candidate.heat_powers?.[index] ?? heatDefault,
          },
          config,
        ),
      );
      setDemoMode("forward");
      setDesign((current) => ({
        ...current,
        model_id: selectedModelId,
        reference_case_index: current.reference_case_index,
        reference_case_id: null,
        modules,
      }));
      setSelectedModule(0);
      setResult(null);
      setRunError(null);
    },
    [config, selectedModelId],
  );

  const activeFrameUrl = result?.frame_urls[field]?.[0] ?? null;
  const exportUrl = result?.export_npz_url ? apiUrl(result.export_npz_url) : null;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>ChannelThermal</h1>
          <span>{selectedModel?.label ?? "Thermal design demo"}</span>
        </div>
        <div className="mode-toggle" aria-label="Demo mode selector">
          <button type="button" className={demoMode === "forward" ? "active" : ""} onClick={() => setDemoMode("forward")}>
            <Flame size={16} />
            Forward
          </button>
          <button type="button" className={demoMode === "inverse" ? "active" : ""} onClick={() => setDemoMode("inverse")}>
            <Target size={16} />
            Inverse
          </button>
        </div>
      </header>

      {demoMode === "forward" ? (
        <div className="workspace forward-workspace">
          <aside className="panel controls-panel">
            <div className="panel-heading">
              <h2>Design</h2>
              <button type="button" className="icon-button" onClick={resetDesign} title="Reset">
                <RefreshCcw size={16} />
              </button>
            </div>

            <label>
              Model
              <select value={selectedModelId} onChange={(event) => setSelectedModelId(event.target.value)}>
                {models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.label}
                  </option>
                ))}
              </select>
            </label>

            <label>
              Reference case
              <select value={design.reference_case_index} onChange={(event) => handleReferenceChange(Number(event.target.value))}>
                {referenceCases.map((ref) => (
                  <option key={ref.case_id} value={ref.index}>
                    {ref.case_id} / {ref.num_modules} modules
                  </option>
                ))}
              </select>
            </label>

            <div className="split-inputs">
              <label>
                Re
                <input
                  type="number"
                  value={design.re ?? selectedReference?.re ?? ""}
                  onChange={(event) => updateDesign({ re: Number(event.target.value) })}
                />
              </label>
              <label>
                u_in
                <input
                  type="number"
                  value={design.u_in ?? selectedReference?.u_in ?? ""}
                  onChange={(event) => updateDesign({ u_in: Number(event.target.value) })}
                />
              </label>
            </div>

            <LayoutCanvas
              modules={design.modules}
              config={config}
              validation={validation}
              selected={selectedModule}
              onSelect={setSelectedModule}
              onChange={updateModule}
            />

            <div className="toolbar-row">
              <button type="button" onClick={addModule} disabled={design.modules.length >= (config?.max_num_modules ?? 12)}>
                <Plus size={16} />
                Add
              </button>
              <button type="button" onClick={randomizeDesign}>
                <Shuffle size={16} />
                Random
              </button>
            </div>

            <div className="module-list">
              {design.modules.map((module, index) => (
                <div key={index} className={`module-row ${selectedModule === index ? "selected" : ""}`}>
                  <button type="button" className="module-index" onClick={() => setSelectedModule(index)}>
                    {index + 1}
                  </button>
                  <input type="number" value={module.x.toFixed(3)} step="0.05" onChange={(event) => updateModule(index, { x: Number(event.target.value) })} />
                  <input type="number" value={module.y.toFixed(3)} step="0.05" onChange={(event) => updateModule(index, { y: Number(event.target.value) })} />
                  <input
                    type="number"
                    value={module.heat_power.toFixed(3)}
                    step="0.05"
                    onChange={(event) => updateModule(index, { heat_power: Number(event.target.value) })}
                  />
                  <button type="button" className="icon-button" onClick={() => deleteModule(index)} title="Delete module">
                    <Trash2 size={15} />
                  </button>
                </div>
              ))}
            </div>

            <div className={`status-box ${validation?.valid ? "valid" : "invalid"}`}>
              <strong>{validation?.valid ? "Ready" : "Needs attention"}</strong>
              <span>{design.modules.length} modules / heat {formatNumber(validation?.total_heat_power, 3)}</span>
              {validation?.warnings.slice(0, 3).map((warning) => (
                <small key={warning}>{warning}</small>
              ))}
            </div>

            <button type="button" className="primary-action" onClick={handleRun} disabled={!validation?.valid || isRunning || !selectedModel?.available}>
              {isRunning ? <Loader2 className="spin" size={18} /> : <Flame size={18} />}
              Run forward
            </button>
            {(runError ?? loadError) && <div className="error-text">{runError ?? loadError}</div>}
          </aside>

          <section className="panel field-panel">
            <div className="panel-heading">
              <h2>Field</h2>
              <div className="segmented">
                {(["temperature", "u", "v", "p", "omega"] as FieldName[]).map((name) => (
                  <button key={name} type="button" className={field === name ? "active" : ""} onClick={() => setField(name)}>
                    {FIELD_LABELS[name]}
                  </button>
                ))}
              </div>
            </div>

            <div className="field-stage">
              {activeFrameUrl ? (
                <img src={apiUrl(activeFrameUrl)} alt={`${FIELD_LABELS[field]} field`} />
              ) : (
                <div className="empty-stage">
                  <Flame size={30} />
                </div>
              )}
            </div>

            <div className="result-footer">
              <div>
                <strong>{result?.reference_case.case_id ?? selectedReference?.case_id ?? "No result"}</strong>
                <span>
                  {result ? `${result.domain.resolution_nx} x ${result.domain.resolution_ny}` : `${config?.domain_length_x ?? 12} x ${config?.domain_length_y ?? 6}`}
                </span>
              </div>
              {exportUrl && (
                <a className="button-link" href={exportUrl}>
                  <Download size={16} />
                  NPZ
                </a>
              )}
            </div>

            {result?.artifacts && (
              <div className="artifact-grid">
                {Object.entries(result.artifacts).map(([name, url]) => (
                  <a key={name} href={apiUrl(url)} target="_blank" rel="noreferrer">
                    <img src={apiUrl(url)} alt={name.replaceAll("_", " ")} />
                    <span>{name.replaceAll("_", " ")}</span>
                  </a>
                ))}
              </div>
            )}
          </section>

          <aside className="panel kpi-panel">
            <div className="panel-heading">
              <h2>KPI</h2>
              <span>{result ? "verified surrogate" : "waiting"}</span>
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
          </aside>
        </div>
      ) : (
        <div className="workspace inverse-workspace">
          <aside className="panel inverse-controls">
            <div className="panel-heading">
              <h2>Target</h2>
              <ChevronDown size={16} />
            </div>

            <label>
              Preset
              <select value={selectedPresetName} onChange={(event) => handlePresetChange(event.target.value)}>
                {presets.map((preset) => (
                  <option key={preset.name} value={preset.name}>
                    {preset.label}
                  </option>
                ))}
              </select>
            </label>

            <label>
              Inverse model
              <select value={selectedInverseModelId} onChange={(event) => setSelectedInverseModelId(event.target.value)}>
                {inverseModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.label}
                  </option>
                ))}
              </select>
            </label>

            <div className="split-inputs">
              <label>
                min modules
                <input type="number" value={constraints.num_modules_min} onChange={(event) => setConstraints((c) => ({ ...c, num_modules_min: Number(event.target.value) }))} />
              </label>
              <label>
                max modules
                <input type="number" value={constraints.num_modules_max} onChange={(event) => setConstraints((c) => ({ ...c, num_modules_max: Number(event.target.value) }))} />
              </label>
            </div>

            <div className="split-inputs">
              <label>
                min distance
                <input type="number" step="0.05" value={constraints.min_center_distance} onChange={(event) => setConstraints((c) => ({ ...c, min_center_distance: Number(event.target.value) }))} />
              </label>
              <label>
                wall
                <input type="number" step="0.02" value={constraints.wall_clearance} onChange={(event) => setConstraints((c) => ({ ...c, wall_clearance: Number(event.target.value) }))} />
              </label>
            </div>

            <div className="split-inputs">
              <label>
                samples
                <input type="number" value={sampling.n_samples} onChange={(event) => setSampling((s) => ({ ...s, n_samples: Number(event.target.value) }))} />
              </label>
              <label>
                steps
                <input type="number" value={sampling.n_steps} onChange={(event) => setSampling((s) => ({ ...s, n_steps: Number(event.target.value) }))} />
              </label>
            </div>

            <div className="split-inputs">
              <label>
                seed
                <input type="number" value={sampling.seed} onChange={(event) => setSampling((s) => ({ ...s, seed: Number(event.target.value) }))} />
              </label>
              <label>
                count
                <select value={sampling.count_mode} onChange={(event) => setSampling((s) => ({ ...s, count_mode: event.target.value as InverseSamplingSpec["count_mode"] }))}>
                  <option value="uniform">uniform</option>
                  <option value="sample">sample</option>
                  <option value="argmax">argmax</option>
                </select>
              </label>
            </div>

            <button type="button" className="primary-action" onClick={handleInverseRun} disabled={inverseRunning || !selectedInverseModelId || !selectedModelId}>
              {inverseRunning ? <Loader2 className="spin" size={18} /> : <Wand2 size={18} />}
              Run inverse
            </button>

            {inverseStatus && (
              <div className="status-box valid">
                <strong>{String(inverseStatus.status ?? "queued")}</strong>
                <span>{inverseJobId}</span>
              </div>
            )}
            {inverseError && <div className="error-text">{inverseError}</div>}
          </aside>

          <section className="panel target-panel">
            <div className="panel-heading">
              <h2>KPI Targets</h2>
              <button type="button" onClick={addKpiTarget}>
                <Plus size={16} />
                Add
              </button>
            </div>
            <div className="target-table">
              <div className="target-header">
                <span>on</span>
                <span>name</span>
                <span>mode</span>
                <span>low</span>
                <span>high/value</span>
                <span>w</span>
                <span />
              </div>
              {kpiTargets.map((row, index) => (
                <div className="target-row" key={`${row.name}-${index}`}>
                  <input type="checkbox" checked={row.enabled} onChange={(event) => updateKpiTarget(index, { enabled: event.target.checked })} />
                  <select value={row.name} onChange={(event) => updateKpiTarget(index, { name: event.target.value })}>
                    {kpiInfos.map((item) => (
                      <option key={item.name} value={item.name}>
                        {item.name}
                      </option>
                    ))}
                  </select>
                  <select value={row.mode} onChange={(event) => updateKpiTarget(index, { mode: event.target.value as KpiMode })}>
                    <option value="max">max</option>
                    <option value="min">min</option>
                    <option value="range">range</option>
                    <option value="exact">exact</option>
                  </select>
                  <input type="number" value={row.low ?? ""} onChange={(event) => updateKpiTarget(index, { low: event.target.value === "" ? null : Number(event.target.value) })} />
                  <input
                    type="number"
                    value={row.mode === "exact" ? row.value ?? "" : row.high ?? ""}
                    onChange={(event) =>
                      updateKpiTarget(index, row.mode === "exact" ? { value: event.target.value === "" ? null : Number(event.target.value) } : { high: event.target.value === "" ? null : Number(event.target.value) })
                    }
                  />
                  <input type="number" value={row.weight} step="0.1" onChange={(event) => updateKpiTarget(index, { weight: Number(event.target.value) })} />
                  <button type="button" className="icon-button" onClick={() => deleteKpiTarget(index)} title="Delete KPI">
                    <Trash2 size={15} />
                  </button>
                </div>
              ))}
            </div>
          </section>

          <aside className="panel candidates-panel">
            <div className="panel-heading">
              <h2>Candidates</h2>
              <span>{inverseResult?.candidate_count ?? 0}</span>
            </div>
            <div className="candidate-list">
              {inverseResult?.candidates?.slice(0, 10).map((candidate) => (
                <div key={`${candidate.rank}-${candidate.sample_index}`} className="candidate-row">
                  <div>
                    <strong>#{candidate.rank}</strong>
                    <span>{candidate.count} modules</span>
                  </div>
                  <div>
                    <strong>{formatNumber(candidate.total_score)}</strong>
                    <span>{candidate.valid ? "valid" : "repaired"}</span>
                  </div>
                  <button type="button" onClick={() => useCandidateInForward(candidate)}>
                    <Flame size={15} />
                    Forward
                  </button>
                </div>
              )) ?? <div className="empty-list">No inverse candidates yet.</div>}
            </div>
          </aside>

          <section className="panel inverse-artifacts">
            <div className="panel-heading">
              <h2>Verification</h2>
              <span>{formatNumber(inverseResult?.summary?.best_score)}</span>
            </div>
            <div className="artifact-grid large">
              {inverseResult?.artifacts &&
                Object.entries(inverseResult.artifacts)
                  .filter(([name]) => name.endsWith("field") || name.includes("candidate") || name.includes("verified") || name.includes("temperature") || name.includes("diversity"))
                  .map(([name, url]) => (
                    <a key={name} href={apiUrl(url)} target="_blank" rel="noreferrer">
                      {url.endsWith(".png") ? <img src={apiUrl(url)} alt={name.replaceAll("_", " ")} /> : <span className="file-chip">{name}</span>}
                      <span>{name.replaceAll("_", " ")}</span>
                    </a>
                  ))}
              {!inverseResult && <div className="empty-stage"><Target size={30} /></div>}
            </div>
          </section>
        </div>
      )}
    </main>
  );
}

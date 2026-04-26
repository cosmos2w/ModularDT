import { useCallback, useEffect, useMemo, useState } from "react";
import DomainCanvas from "./components/DomainCanvas";
import FlowViewer from "./components/FlowViewer";
import KpiPanel from "./components/KpiPanel";
import ParameterPanel from "./components/ParameterPanel";
import { getExampleDesigns, getJobResult, getModelConfig, getModels, runInference, validateDesign } from "./api";
import type {
  Cylinder,
  DesignRequest,
  ExampleDesign,
  FieldName,
  JobResult,
  Mode,
  ModelConfig,
  ModelEntry,
  ValidationResult,
} from "./types";

const DEFAULT_GENERATIVE = {
  num_samples: 4,
  n_steps: 16,
  seed: null,
  noise_mode: "harmonic",
};

function defaultCylinders(config: ModelConfig | null): Cylinder[] {
  const lx = config?.domain_length_x ?? 24;
  const ly = config?.domain_length_y ?? 12;
  return [
    { x: lx * 0.24, y: ly * 0.52 },
    { x: lx * 0.38, y: ly * 0.64 },
    { x: lx * 0.52, y: ly * 0.43 },
    { x: lx * 0.66, y: ly * 0.57 },
  ].slice(0, Math.min(4, config?.max_num_cylinders ?? 4));
}

function makeDesign(modelId: string, mode: Mode, config: ModelConfig | null): DesignRequest {
  return {
    model_id: modelId,
    mode,
    re: 100,
    cylinders: defaultCylinders(config),
    phase_bins: config?.default_phase_bins ?? 36,
    resolution_nx: 192,
    resolution_ny: 96,
    field: "omega",
    display_smoothing: true,
    display_scale: 3,
    render_interpolation: "bicubic",
    return_hypergraph: true,
    return_kpis: true,
    generative: DEFAULT_GENERATIVE,
  };
}

function distance(a: Cylinder, b: Cylinder, lx: number, ly: number) {
  const dx = Math.min(Math.abs(a.x - b.x), lx - Math.abs(a.x - b.x));
  const dy = Math.min(Math.abs(a.y - b.y), ly - Math.abs(a.y - b.y));
  return Math.sqrt(dx * dx + dy * dy);
}

function randomDesign(count: number, config: ModelConfig | null): Cylinder[] {
  const lx = config?.domain_length_x ?? 24;
  const ly = config?.domain_length_y ?? 12;
  const maxCount = Math.min(count, config?.max_num_cylinders ?? 8);
  const cylinders: Cylinder[] = [];
  let attempts = 0;
  while (cylinders.length < maxCount && attempts < 500) {
    attempts += 1;
    const candidate = {
      x: 1.2 + Math.random() * Math.max(lx - 2.4, 1),
      y: 1.2 + Math.random() * Math.max(ly - 2.4, 1),
    };
    if (cylinders.every((cyl) => distance(cyl, candidate, lx, ly) >= 1.2)) {
      cylinders.push(candidate);
    }
  }
  return cylinders.length ? cylinders : defaultCylinders(config);
}

export default function App() {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [exampleDesigns, setExampleDesigns] = useState<ExampleDesign[]>([]);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [mode, setMode] = useState<Mode>("deterministic");
  const [config, setConfig] = useState<ModelConfig | null>(null);
  const [design, setDesign] = useState<DesignRequest>(() => makeDesign("", "deterministic", null));
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [selectedCylinder, setSelectedCylinder] = useState<number | null>(0);
  const [result, setResult] = useState<JobResult | null>(null);
  const [field, setField] = useState<FieldName>("omega");
  const [frame, setFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [showHypergraph, setShowHypergraph] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([getModels(), getExampleDesigns()])
      .then(([items, examples]) => {
        if (cancelled) return;
        setModels(items);
        setExampleDesigns(examples);
        const firstDeterministic = items.find((model) => model.mode === "deterministic" && model.enabled) ?? items[0];
        if (firstDeterministic) {
          setSelectedModelId(firstDeterministic.id);
          setMode(firstDeterministic.mode);
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
      .then((nextConfig) => {
        if (cancelled) return;
        setConfig(nextConfig);
        const selectedModel = models.find((model) => model.id === selectedModelId);
        const nextMode = selectedModel?.mode ?? nextConfig.mode ?? "deterministic";
        setMode(nextMode);
        setDesign((current) => ({
          ...makeDesign(selectedModelId, nextMode, nextConfig),
          re: current.model_id ? current.re : 100,
          phase_bins: current.model_id ? current.phase_bins : nextConfig.default_phase_bins,
          resolution_nx: current.resolution_nx,
          resolution_ny: current.resolution_ny,
          cylinders: current.model_id && current.mode === nextMode ? current.cylinders.slice(0, nextConfig.max_num_cylinders) : defaultCylinders(nextConfig),
        }));
        setSelectedCylinder(0);
        setLoadError(null);
      })
      .catch((error: Error) => setLoadError(error.message));
    return () => {
      cancelled = true;
    };
  }, [models, selectedModelId]);

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

  const selectedModel = useMemo(
    () => models.find((model) => model.id === selectedModelId) ?? null,
    [models, selectedModelId],
  );

  const handleModelChange = useCallback(
    (modelId: string) => {
      const model = models.find((item) => item.id === modelId);
      setSelectedModelId(modelId);
      if (model) setMode(model.mode);
      setResult(null);
      setFrame(0);
      setRunError(null);
    },
    [models],
  );

  const handleModeChange = useCallback(
    (nextMode: Mode) => {
      setMode(nextMode);
      const candidate = models.find((model) => model.mode === nextMode && (nextMode === "deterministic" || model.stage === 2 || model.status === "stage2_pending"));
      if (candidate) {
        setSelectedModelId(candidate.id);
      } else {
        setDesign((current) => ({ ...current, mode: nextMode }));
      }
      setResult(null);
      setFrame(0);
      setRunError(null);
    },
    [models],
  );

  const updateDesign = useCallback((patch: Partial<DesignRequest>) => {
    setDesign((current) => ({ ...current, ...patch }));
    setRunError(null);
  }, []);

  const updateCylinder = useCallback((index: number, cylinder: Cylinder) => {
    setDesign((current) => {
      const lx = config?.domain_length_x ?? 24;
      const ly = config?.domain_length_y ?? 12;
      const cylinders = current.cylinders.map((item, itemIndex) =>
        itemIndex === index
          ? {
              x: Math.min(Math.max(cylinder.x, 0), lx - 1e-3),
              y: Math.min(Math.max(cylinder.y, 0), ly - 1e-3),
            }
          : item,
      );
      return { ...current, cylinders };
    });
    setRunError(null);
  }, [config]);

  const addCylinder = useCallback(() => {
    setDesign((current) => {
      const max = config?.max_num_cylinders ?? 8;
      if (current.cylinders.length >= max) return current;
      const lx = config?.domain_length_x ?? 24;
      const ly = config?.domain_length_y ?? 12;
      const next = { x: lx * (0.25 + 0.1 * current.cylinders.length), y: ly * (0.5 + (current.cylinders.length % 2 ? -0.12 : 0.12)) };
      setSelectedCylinder(current.cylinders.length);
      return { ...current, cylinders: [...current.cylinders, next] };
    });
  }, [config]);

  const deleteCylinder = useCallback((index: number) => {
    setDesign((current) => ({
      ...current,
      cylinders: current.cylinders.filter((_, itemIndex) => itemIndex !== index),
    }));
    setSelectedCylinder((current) => (current === null ? null : Math.max(0, Math.min(current, design.cylinders.length - 2))));
  }, [design.cylinders.length]);

  const resetDesign = useCallback(() => {
    setDesign(makeDesign(selectedModelId, mode, config));
    setSelectedCylinder(0);
    setResult(null);
    setFrame(0);
    setIsPlaying(false);
    setRunError(null);
  }, [config, mode, selectedModelId]);

  const randomizeDesign = useCallback(() => {
    setDesign((current) => ({
      ...current,
      cylinders: randomDesign(Math.max(3, Math.min(current.cylinders.length || 4, config?.max_num_cylinders ?? 8)), config),
    }));
    setSelectedCylinder(0);
    setResult(null);
    setFrame(0);
    setRunError(null);
  }, [config]);

  const handleRun = useCallback(async () => {
    if (!selectedModel || !validation?.valid) return;
    setIsRunning(true);
    setRunError(null);
    setIsPlaying(false);
    try {
      const response = await runInference(design);
      const job = await getJobResult(response.job_id);
      setResult(job);
      setFrame(0);
      setField(design.field);
      setIsPlaying(Boolean(job.render.frame_count));
    } catch (error) {
      setRunError(error instanceof Error ? error.message : String(error));
    } finally {
      setIsRunning(false);
    }
  }, [design, selectedModel, validation]);

  const applyExampleDesign = useCallback((exampleName: string) => {
    const example = exampleDesigns.find((item) => item.name === exampleName);
    if (!example) return;
    const max = config?.max_num_cylinders ?? 8;
    setDesign((current) => ({
      ...current,
      re: example.re,
      cylinders: example.cylinders.slice(0, max),
    }));
    setSelectedCylinder(0);
    setResult(null);
    setFrame(0);
    setIsPlaying(false);
    setRunError(null);
  }, [config, exampleDesigns]);

  const safeFrame = result ? Math.min(frame, Math.max(result.render.frame_count - 1, 0)) : frame;

  return (
    <main className="app-shell">
      <ParameterPanel
        models={models}
        selectedModelId={selectedModelId}
        mode={mode}
        config={config}
        design={design}
        validation={validation}
        exampleDesigns={exampleDesigns}
        isRunning={isRunning}
        runError={runError ?? loadError}
        selectedCylinder={selectedCylinder}
        onModelChange={handleModelChange}
        onModeChange={handleModeChange}
        onDesignChange={updateDesign}
        onCylinderChange={updateCylinder}
        onAddCylinder={addCylinder}
        onDeleteCylinder={deleteCylinder}
        onSelectCylinder={setSelectedCylinder}
        onExampleSelect={applyExampleDesign}
        onRun={handleRun}
        onReset={resetDesign}
        onRandomize={randomizeDesign}
      />

      <section className="center-stack">
        <DomainCanvas
          cylinders={design.cylinders}
          config={config}
          re={design.re}
          validation={validation}
          selectedCylinder={selectedCylinder}
          onCylinderChange={updateCylinder}
          onAddCylinder={addCylinder}
          onDeleteCylinder={deleteCylinder}
          onSelectCylinder={setSelectedCylinder}
        />
        <FlowViewer
          result={result}
          field={field}
          frame={safeFrame}
          isPlaying={isPlaying}
          speed={speed}
          showHypergraph={showHypergraph}
          mode={mode}
          onFieldChange={(nextField) => {
            setField(nextField);
            setDesign((current) => ({ ...current, field: nextField }));
          }}
          onFrameChange={setFrame}
          onPlayingChange={setIsPlaying}
          onSpeedChange={setSpeed}
          onShowHypergraphChange={setShowHypergraph}
        />
      </section>

      <KpiPanel
        kpis={result?.kpis}
        frame={safeFrame}
        frameCount={result?.render.frame_count ?? 0}
      />
    </main>
  );
}

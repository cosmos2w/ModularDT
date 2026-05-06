import { AlertTriangle, Play, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getInverseJobResult,
  getInverseJobStatus,
  getInverseKpis,
  getInverseModels,
  getInverseTargetPresets,
  getSimulationStatus,
  quickValidateCandidate,
  runInverseDesign,
  simulationValidateCandidate,
} from "../../api";
import type {
  Cylinder,
  InverseCandidate,
  InverseConstraintSpec,
  InverseJobResult,
  InverseJobStatus,
  InverseKpiEntry,
  InverseModelEntry,
  InverseRunRequest,
  InverseSamplingSpec,
  InverseTargetPreset,
  InverseVerificationSpec,
  KpiTargetMode,
  KpiTargetSpec,
  ModelEntry,
  SimulationValidationStatus,
} from "../../types";
import CandidateCarousel from "./CandidateCarousel";
import CandidateFlowViewer from "./CandidateFlowViewer";
import CandidateKpiComparison from "./CandidateKpiComparison";
import KpiTargetEditor from "./KpiTargetEditor";
import SimulationProgress from "./SimulationProgress";

interface Props {
  forwardModels: ModelEntry[];
  onUseCandidateInForward: (candidate: InverseCandidate, re: number, cylinders: Cylinder[], verifierModelId: string) => void;
}

const defaultSampling: InverseSamplingSpec = {
  n_samples: 64,
  verify_top_k: 16,
  save_verified_top_k: 4,
  n_steps: 32,
  seed: null,
};

const defaultConstraints: InverseConstraintSpec = {
  re: 120,
  num_cylinders_min: 3,
  num_cylinders_max: 6,
  min_center_distance: 1.5,
  min_x_span: 6,
  min_y_span: 2.5,
};

function makeRows(kpis: InverseKpiEntry[]): KpiTargetSpec[] {
  return kpis.map((kpi) => ({
    enabled: false,
    name: kpi.name,
    mode: kpi.default_mode,
    value: null,
    low: kpi.default_mode === "range" ? 0 : null,
    high: kpi.default_mode === "range" || kpi.default_mode === "max" ? 1 : null,
    weight: kpi.default_weight ?? 1,
  }));
}

function specNumber(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function applyPresetToRows(rows: KpiTargetSpec[], preset: InverseTargetPreset): KpiTargetSpec[] {
  const presetKpis = preset.target.kpis ?? {};
  return rows.map((row) => {
    const spec = presetKpis[row.name];
    if (!spec) return { ...row, enabled: false };
    const mode = (spec.mode ?? "exact") as KpiTargetMode;
    return {
      ...row,
      enabled: true,
      mode,
      value: specNumber(spec.value),
      low: specNumber(spec.low),
      high: specNumber(spec.high),
      weight: specNumber(spec.weight) ?? 1,
    };
  });
}

function defaultVerification(forwardModels: ModelEntry[], preferredId?: string | null): InverseVerificationSpec {
  const preferred = preferredId ? forwardModels.find((model) => model.id === preferredId) : null;
  const first = preferred ?? forwardModels.find((model) => model.mode === "deterministic" && model.enabled) ?? forwardModels[0];
  const defaultGrid = modelDefaultGrid(first);
  return {
    forward_verifier_model_id: first?.id ?? "",
    forward_backend: first?.mode ?? "deterministic",
    phase_bins: 12,
    nx: defaultGrid.nx,
    ny: defaultGrid.ny,
    generative_num_samples: 4,
    generative_n_steps: 16,
    generative_ode_solver: "heun",
    uncertainty_penalty_weight: 0.05,
  };
}

function modelDefaultGrid(model: ModelEntry | null | undefined) {
  return {
    nx: Number(model?.metadata?.default_resolution_nx ?? (model?.mode === "generative" ? 256 : 96)),
    ny: Number(model?.metadata?.default_resolution_ny ?? (model?.mode === "generative" ? 128 : 48)),
  };
}

function isRunningStatus(status: string | undefined) {
  return Boolean(status && ["queued", "running", "parsing_results"].includes(status));
}

function isSimulationRunning(status: string | undefined) {
  return Boolean(status && ["queued", "writing_config", "running_simulation", "preprocessing", "computing_kpis"].includes(status));
}

export default function InverseDesignPanel({ forwardModels, onUseCandidateInForward }: Props) {
  const [inverseModels, setInverseModels] = useState<InverseModelEntry[]>([]);
  const [presets, setPresets] = useState<InverseTargetPreset[]>([]);
  const [kpiCatalog, setKpiCatalog] = useState<InverseKpiEntry[]>([]);
  const [kpiRows, setKpiRows] = useState<KpiTargetSpec[]>([]);
  const [selectedInverseId, setSelectedInverseId] = useState("");
  const [selectedPreset, setSelectedPreset] = useState("");
  const [targetName, setTargetName] = useState<string | null>("web_inverse_target");
  const [constraints, setConstraints] = useState<InverseConstraintSpec>(defaultConstraints);
  const [sampling, setSampling] = useState<InverseSamplingSpec>(defaultSampling);
  const [verification, setVerification] = useState<InverseVerificationSpec>(() => defaultVerification(forwardModels));
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<InverseJobStatus | null>(null);
  const [result, setResult] = useState<InverseJobResult | null>(null);
  const [candidates, setCandidates] = useState<InverseCandidate[]>([]);
  const [activeIndex, setActiveIndex] = useState(0);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [quickCandidateId, setQuickCandidateId] = useState<string | null>(null);
  const [simulationStatus, setSimulationStatus] = useState<SimulationValidationStatus | null>(null);
  const [simulationCandidateId, setSimulationCandidateId] = useState<string | null>(null);

  const updateCandidate = useCallback((candidate: InverseCandidate) => {
    setCandidates((current) => current.map((item) => (item.id === candidate.id ? candidate : item)));
    setResult((current) => (current ? { ...current, candidates: current.candidates.map((item) => (item.id === candidate.id ? candidate : item)) } : current));
  }, []);

  useEffect(() => {
    let cancelled = false;
    Promise.all([getInverseModels(), getInverseTargetPresets(), getInverseKpis()])
      .then(([models, nextPresets, nextKpis]) => {
        if (cancelled) return;
        setInverseModels(models);
        setPresets(nextPresets);
        setKpiCatalog(nextKpis);
        const firstModel = models.find((model) => model.available) ?? models[0];
        if (firstModel) {
          setSelectedInverseId(firstModel.id);
          setVerification(defaultVerification(forwardModels, firstModel.default_forward_verifier_id));
        }
        setKpiRows(makeRows(nextKpis));
      })
      .catch((error: Error) => setLoadError(error.message));
    return () => {
      cancelled = true;
    };
  }, [forwardModels]);

  useEffect(() => {
    const selected = inverseModels.find((model) => model.id === selectedInverseId);
    if (selected?.default_forward_verifier_id) {
      setVerification((current) => {
        if (current.forward_verifier_model_id) return current;
        return defaultVerification(forwardModels, selected.default_forward_verifier_id);
      });
    }
  }, [forwardModels, inverseModels, selectedInverseId]);

  useEffect(() => {
    if (!jobId || jobStatus?.status === "complete" || jobStatus?.status === "error") return;
    const timer = window.setInterval(async () => {
      try {
        const status = await getInverseJobStatus(jobId);
        setJobStatus(status);
        if (status.status === "complete") {
          const nextResult = await getInverseJobResult(jobId);
          setResult(nextResult);
          setCandidates(nextResult.candidates ?? []);
          setActiveIndex(0);
        }
      } catch (error) {
        setRunError(error instanceof Error ? error.message : String(error));
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [jobId, jobStatus?.status]);

  useEffect(() => {
    if (!jobId || !simulationCandidateId || !isSimulationRunning(simulationStatus?.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const status = await getSimulationStatus(jobId, simulationCandidateId);
        setSimulationStatus(status);
        if (status.candidate) {
          updateCandidate(status.candidate);
        }
      } catch (error) {
        setRunError(error instanceof Error ? error.message : String(error));
      }
    }, 2200);
    return () => window.clearInterval(timer);
  }, [jobId, simulationCandidateId, simulationStatus?.status]);

  const selectedInverseModel = useMemo(
    () => inverseModels.find((model) => model.id === selectedInverseId) ?? null,
    [inverseModels, selectedInverseId],
  );
  const selectedForwardModel = useMemo(
    () => forwardModels.find((model) => model.id === verification.forward_verifier_model_id) ?? null,
    [forwardModels, verification.forward_verifier_model_id],
  );
  const activeCandidate = candidates[activeIndex] ?? null;
  const activeKpiCount = kpiRows.filter((row) => row.enabled).length;
  const canRun = Boolean(selectedInverseId && verification.forward_verifier_model_id && activeKpiCount > 0 && !isSubmitting && !isRunningStatus(jobStatus?.status));

  const patchConstraints = useCallback((patch: Partial<InverseConstraintSpec>) => {
    setConstraints((current) => ({ ...current, ...patch }));
    setRunError(null);
  }, []);

  const patchSampling = useCallback((patch: Partial<InverseSamplingSpec>) => {
    setSampling((current) => ({ ...current, ...patch }));
    setRunError(null);
  }, []);

  const patchVerification = useCallback((patch: Partial<InverseVerificationSpec>) => {
    setVerification((current) => ({ ...current, ...patch }));
    setRunError(null);
  }, []);

  const handleForwardModelChange = useCallback(
    (modelId: string) => {
      const model = forwardModels.find((item) => item.id === modelId);
      const grid = modelDefaultGrid(model);
      patchVerification({
        forward_verifier_model_id: modelId,
        forward_backend: model?.mode ?? verification.forward_backend,
        nx: grid.nx,
        ny: grid.ny,
      });
    },
    [forwardModels, patchVerification, verification.forward_backend],
  );

  const handleBackendChange = useCallback(
    (backend: InverseVerificationSpec["forward_backend"]) => {
      const model = forwardModels.find((item) => item.mode === backend && item.enabled) ?? forwardModels.find((item) => item.mode === backend);
      const grid = modelDefaultGrid(model);
      patchVerification({
        forward_backend: backend,
        forward_verifier_model_id: model?.id ?? verification.forward_verifier_model_id,
        nx: grid.nx,
        ny: grid.ny,
      });
    },
    [forwardModels, patchVerification, verification.forward_verifier_model_id],
  );

  const handlePresetSelect = useCallback(
    (name: string) => {
      setSelectedPreset(name);
      const preset = presets.find((item) => item.name === name);
      if (!preset) return;
      const preferences = preset.target.preferences ?? {};
      setTargetName(preset.target.name ?? preset.name);
      setConstraints({
        re: Number(preset.target.Re ?? preset.target.re ?? constraints.re),
        num_cylinders_min: Number(preset.target.num_cylinders_min ?? constraints.num_cylinders_min),
        num_cylinders_max: Number(preset.target.num_cylinders_max ?? constraints.num_cylinders_max),
        min_center_distance: Number(preset.target.min_center_distance ?? preferences.min_center_distance ?? constraints.min_center_distance),
        min_x_span: specNumber(preferences.min_x_span),
        min_y_span: specNumber(preferences.min_y_span),
      });
      setKpiRows((rows) => applyPresetToRows(rows.length ? rows : makeRows(kpiCatalog), preset));
    },
    [constraints, kpiCatalog, presets],
  );

  const handleRun = useCallback(async () => {
    if (!canRun) {
      setRunError(activeKpiCount ? "Inverse setup is incomplete." : "Enable at least one KPI target row.");
      return;
    }
    setIsSubmitting(true);
    setRunError(null);
    setJobStatus(null);
    setResult(null);
    setCandidates([]);
    setSimulationStatus(null);
    try {
      const request: InverseRunRequest = {
        inverse_model_id: selectedInverseId,
        target_name: targetName,
        kpis: kpiRows,
        constraints,
        sampling,
        verification,
        simulation_enabled: false,
      };
      const response = await runInverseDesign(request);
      setJobId(response.job_id);
      setJobStatus({ job_id: response.job_id, status: response.status, result_url: response.result_url });
    } catch (error) {
      setRunError(error instanceof Error ? error.message : String(error));
    } finally {
      setIsSubmitting(false);
    }
  }, [activeKpiCount, canRun, constraints, kpiRows, sampling, selectedInverseId, targetName, verification]);

  const handleQuickValidate = useCallback(
    async (candidate: InverseCandidate) => {
      if (!jobId) return;
      setQuickCandidateId(candidate.id);
      setRunError(null);
      try {
        const payload = await quickValidateCandidate(jobId, candidate.id, verification);
        if (payload.candidate) updateCandidate(payload.candidate);
      } catch (error) {
        setRunError(error instanceof Error ? error.message : String(error));
      } finally {
        setQuickCandidateId(null);
      }
    },
    [jobId, updateCandidate, verification],
  );

  const handleSimulationValidate = useCallback(
    async (candidate: InverseCandidate) => {
      if (!jobId) return;
      setSimulationCandidateId(candidate.id);
      setRunError(null);
      try {
        const status = await simulationValidateCandidate(jobId, candidate.id, {
          simulation_mode: "inert",
          simulation_device: null,
          simulation_gpu_id: null,
          simulation_preprocess_device: null,
          simulation_nx: null,
          simulation_ny: null,
          simulation_phase_bins: verification.phase_bins,
          simulation_warmup_cycles: null,
          simulation_save_cycles: null,
          simulation_frames_per_cycle: null,
          simulation_dt: null,
        });
        setSimulationStatus(status);
        if (status.candidate) updateCandidate(status.candidate);
      } catch (error) {
        setRunError(error instanceof Error ? error.message : String(error));
      }
    },
    [jobId, updateCandidate, verification.phase_bins],
  );

  const handleUseCandidate = useCallback(
    (candidate: InverseCandidate) => {
      onUseCandidateInForward(
        candidate,
        constraints.re,
        candidate.centers.map(([x, y]) => ({ x, y })),
        verification.forward_verifier_model_id,
      );
    },
    [constraints.re, onUseCandidateInForward, verification.forward_verifier_model_id],
  );

  const resetJob = useCallback(() => {
    setJobId(null);
    setJobStatus(null);
    setResult(null);
    setCandidates([]);
    setRunError(null);
    setSimulationStatus(null);
    setActiveIndex(0);
  }, []);

  const domainLengthX = result?.domain.length_x ?? Number(selectedInverseModel?.metadata.domain_length_x ?? 24);
  const domainLengthY = result?.domain.length_y ?? Number(selectedInverseModel?.metadata.domain_length_y ?? 12);

  return (
    <section className="inverse-workspace">
      <aside className="panel inverse-setup-panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Inverse</p>
            <h1>Design setup</h1>
          </div>
          <span className="count-pill">{activeKpiCount} KPI</span>
        </div>

        <div className="selector-stack">
          <label className="field-label" htmlFor="inverse-model-select">Inverse model</label>
          <select id="inverse-model-select" className="control" value={selectedInverseId} onChange={(event) => setSelectedInverseId(event.target.value)}>
            {inverseModels.map((model) => (
              <option key={model.id} value={model.id}>
                {model.label} - {model.status}
              </option>
            ))}
          </select>
          {selectedInverseModel?.reason_unavailable && <div className="warning compact">{selectedInverseModel.reason_unavailable}</div>}
        </div>

        <div className="selector-stack">
          <label className="field-label" htmlFor="forward-verifier-select">Forward verifier</label>
          <label>
            <span className="field-label">Verifier backend</span>
            <select className="control" value={verification.forward_backend} onChange={(event) => handleBackendChange(event.target.value as InverseVerificationSpec["forward_backend"])}>
              <option value="deterministic">deterministic</option>
              <option value="generative">generative</option>
            </select>
          </label>
          <select id="forward-verifier-select" className="control" value={verification.forward_verifier_model_id} onChange={(event) => handleForwardModelChange(event.target.value)}>
            {forwardModels.map((model) => (
              <option key={model.id} value={model.id}>
                {model.label} - {model.mode} - {model.status}
              </option>
            ))}
          </select>
          {selectedForwardModel?.reason_unavailable && <div className="warning compact">{selectedForwardModel.reason_unavailable}</div>}
        </div>

        <div className="setting-grid two">
          <label>
            <span className="field-label">Re</span>
            <input className="number-input" type="number" value={constraints.re} onChange={(event) => patchConstraints({ re: Number(event.target.value) })} />
          </label>
          <label>
            <span className="field-label">Min distance</span>
            <input className="number-input" type="number" step="0.05" value={constraints.min_center_distance} onChange={(event) => patchConstraints({ min_center_distance: Number(event.target.value) })} />
          </label>
          <label>
            <span className="field-label">Count min</span>
            <input className="number-input" type="number" min="1" value={constraints.num_cylinders_min} onChange={(event) => patchConstraints({ num_cylinders_min: Number(event.target.value) })} />
          </label>
          <label>
            <span className="field-label">Count max</span>
            <input className="number-input" type="number" min="1" value={constraints.num_cylinders_max} onChange={(event) => patchConstraints({ num_cylinders_max: Number(event.target.value) })} />
          </label>
          <label>
            <span className="field-label">Min x span</span>
            <input className="number-input" type="number" value={constraints.min_x_span ?? ""} onChange={(event) => patchConstraints({ min_x_span: specNumber(event.target.value) })} />
          </label>
          <label>
            <span className="field-label">Min y span</span>
            <input className="number-input" type="number" value={constraints.min_y_span ?? ""} onChange={(event) => patchConstraints({ min_y_span: specNumber(event.target.value) })} />
          </label>
        </div>

        <div className="setting-grid two">
          <label>
            <span className="field-label">Samples</span>
            <input className="number-input" type="number" min="1" max="512" value={sampling.n_samples} onChange={(event) => patchSampling({ n_samples: Number(event.target.value) })} />
          </label>
          <label>
            <span className="field-label">Verify top K</span>
            <input className="number-input" type="number" min="0" max="64" value={sampling.verify_top_k} onChange={(event) => patchSampling({ verify_top_k: Number(event.target.value) })} />
          </label>
          <label>
            <span className="field-label">Save top K</span>
            <input className="number-input" type="number" min="0" max="16" value={sampling.save_verified_top_k} onChange={(event) => patchSampling({ save_verified_top_k: Number(event.target.value) })} />
          </label>
          <label>
            <span className="field-label">Inverse steps</span>
            <input className="number-input" type="number" min="1" value={sampling.n_steps} onChange={(event) => patchSampling({ n_steps: Number(event.target.value) })} />
          </label>
          <label>
            <span className="field-label">Seed</span>
            <input className="number-input" type="number" value={sampling.seed ?? ""} onChange={(event) => patchSampling({ seed: specNumber(event.target.value) })} />
          </label>
          <label>
            <span className="field-label">Phase bins</span>
            <input className="number-input" type="number" min="1" value={verification.phase_bins} onChange={(event) => patchVerification({ phase_bins: Number(event.target.value) })} />
          </label>
          <label>
            <span className="field-label">Grid nx</span>
            <input className="number-input" type="number" min="8" value={verification.nx} onChange={(event) => patchVerification({ nx: Number(event.target.value) })} />
          </label>
          <label>
            <span className="field-label">Grid ny</span>
            <input className="number-input" type="number" min="8" value={verification.ny} onChange={(event) => patchVerification({ ny: Number(event.target.value) })} />
          </label>
        </div>

        {verification.forward_backend === "generative" && (
          <div className="setting-grid two">
            <label>
              <span className="field-label">Gen samples</span>
              <input className="number-input" type="number" min="1" value={verification.generative_num_samples} onChange={(event) => patchVerification({ generative_num_samples: Number(event.target.value) })} />
            </label>
            <label>
              <span className="field-label">Gen steps</span>
              <input className="number-input" type="number" min="1" value={verification.generative_n_steps} onChange={(event) => patchVerification({ generative_n_steps: Number(event.target.value) })} />
            </label>
          </div>
        )}

        <div className="button-row">
          <button className="secondary-button" type="button" onClick={resetJob}>
            <RotateCcw size={16} />
            Clear
          </button>
          <button className="run-button inverse-run-button" type="button" disabled={!canRun} onClick={handleRun}>
            <Play size={18} />
            {isRunningStatus(jobStatus?.status) || isSubmitting ? "Running inverse..." : "Run inverse design"}
          </button>
        </div>
        {(runError || loadError || jobStatus?.error) && (
          <div className="warning">
            <AlertTriangle size={16} />
            <span>{runError ?? loadError ?? jobStatus?.error}</span>
          </div>
        )}
      </aside>

      <div className="inverse-main-stack">
        <KpiTargetEditor rows={kpiRows} presets={presets} selectedPreset={selectedPreset} onRowsChange={setKpiRows} onPresetSelect={handlePresetSelect} />
        <CandidateCarousel
          candidates={candidates}
          activeIndex={activeIndex}
          domainLengthX={domainLengthX}
          domainLengthY={domainLengthY}
          isQuickValidating={quickCandidateId === activeCandidate?.id}
          isSimulating={isSimulationRunning(simulationStatus?.status)}
          onActiveIndexChange={setActiveIndex}
          onQuickValidate={handleQuickValidate}
          onSimulationValidate={handleSimulationValidate}
          onUseInForwardMode={handleUseCandidate}
        />
        <CandidateFlowViewer candidate={activeCandidate} />
      </div>

      <aside className="inverse-side-stack">
        <section className="inverse-card inverse-job-status">
          <div className="section-title-row">
            <h2>Job status</h2>
            <span className="count-pill">{jobStatus?.status ?? "idle"}</span>
          </div>
          <pre className="log-tail">{(jobStatus?.log_tail?.length ? jobStatus.log_tail : ["Inverse job log tail appears here."]).join("\n")}</pre>
        </section>
        <CandidateKpiComparison candidate={activeCandidate} />
        <SimulationProgress status={simulationStatus} />
      </aside>
    </section>
  );
}

import {
  AlertTriangle,
  Dices,
  Play,
  Plus,
  RotateCcw,
  Trash2,
} from "lucide-react";
import CheckpointSelector from "./CheckpointSelector";
import type { Cylinder, DesignRequest, ExampleDesign, Mode, ModelConfig, ModelEntry, ValidationResult } from "../types";

interface Props {
  models: ModelEntry[];
  selectedModelId: string;
  mode: Mode;
  config: ModelConfig | null;
  design: DesignRequest;
  validation: ValidationResult | null;
  exampleDesigns: ExampleDesign[];
  isRunning: boolean;
  runError: string | null;
  selectedCylinder: number | null;
  onModelChange: (modelId: string) => void;
  onModeChange: (mode: Mode) => void;
  onDesignChange: (patch: Partial<DesignRequest>) => void;
  onCylinderChange: (index: number, cylinder: Cylinder) => void;
  onAddCylinder: () => void;
  onDeleteCylinder: (index: number) => void;
  onSelectCylinder: (index: number | null) => void;
  onExampleSelect: (exampleName: string) => void;
  onRun: () => void;
  onReset: () => void;
  onRandomize: () => void;
}

function formatNumber(value: number) {
  return Number.isFinite(value) ? value.toFixed(2) : "0.00";
}

export default function ParameterPanel({
  models,
  selectedModelId,
  mode,
  config,
  design,
  validation,
  exampleDesigns,
  isRunning,
  runError,
  selectedCylinder,
  onModelChange,
  onModeChange,
  onDesignChange,
  onCylinderChange,
  onAddCylinder,
  onDeleteCylinder,
  onSelectCylinder,
  onExampleSelect,
  onRun,
  onReset,
  onRandomize,
}: Props) {
  const selectedModel = models.find((model) => model.id === selectedModelId);
  const selectedModeUnavailable =
    mode === "generative" && (!selectedModel || !selectedModel.enabled || selectedModel.stage !== 2);
  const selectedModelUnavailable = Boolean(selectedModel && !selectedModel.available);
  const maxCylinders = config?.max_num_cylinders ?? validation?.max_num_cylinders ?? 8;
  const maxPhaseBins = config?.max_phase_bins ?? validation?.max_phase_bins ?? 36;
  const phasePolicy = config?.phase_bin_policy ?? validation?.phase_bin_policy ?? "cap";
  const phaseOverMax = design.phase_bins > maxPhaseBins;
  const phaseRejectsRun = phaseOverMax && phasePolicy === "reject";
  const effectivePhaseBins = validation?.effective_phase_bins ?? Math.min(design.phase_bins, maxPhaseBins);
  const canAdd = design.cylinders.length < maxCylinders;
  const runDisabled = isRunning || !validation?.valid || phaseRejectsRun || selectedModeUnavailable || selectedModelUnavailable || !selectedModelId;

  return (
    <aside className="panel parameter-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Panel A</p>
          <h1>Run setup</h1>
        </div>
        <span className="count-pill">{design.cylinders.length}/{maxCylinders}</span>
      </div>

      <CheckpointSelector
        models={models}
        selectedModelId={selectedModelId}
        mode={mode}
        onModelChange={onModelChange}
        onModeChange={onModeChange}
      />

      {selectedModeUnavailable && (
        <div className="warning compact">
          <AlertTriangle size={16} />
          <span>Generative stage-2 checkpoint pending.</span>
        </div>
      )}
      {selectedModelUnavailable && !selectedModeUnavailable && (
        <div className="warning compact">
          <AlertTriangle size={16} />
          <span>{selectedModel?.reason_unavailable ?? "Selected checkpoint is not available."}</span>
        </div>
      )}

      <div className="setting-group">
        <div className="label-row">
          <label className="field-label" htmlFor="re-slider">Reynolds number</label>
          <input
            id="re-input"
            className="number-input small"
            type="number"
            min="1"
            step="1"
            value={design.re}
            onChange={(event) => onDesignChange({ re: Number(event.target.value) })}
          />
        </div>
        <input
          id="re-slider"
          className="slider"
          type="range"
          min={config?.expected_re_min ?? 20}
          max={config?.expected_re_max ?? 300}
          step="1"
          value={design.re}
          onChange={(event) => onDesignChange({ re: Number(event.target.value) })}
        />
      </div>

      <label className="example-select">
        <span className="field-label">Example design</span>
        <select
          className="control"
          defaultValue=""
          onChange={(event) => {
            if (event.target.value) onExampleSelect(event.target.value);
            event.currentTarget.value = "";
          }}
        >
          <option value="">Load example...</option>
          {exampleDesigns.map((example) => (
            <option key={example.name} value={example.name}>
              {example.name}
            </option>
          ))}
        </select>
      </label>

      <div className={`phase-control ${phaseOverMax ? "has-warning" : ""}`}>
        <div className="label-row">
          <label className="field-label" htmlFor="phase-bins-input">Phase bins</label>
          <input
            id="phase-bins-input"
            className="number-input"
            type="number"
            min="1"
            max="512"
            value={design.phase_bins}
            onChange={(event) => onDesignChange({ phase_bins: Number(event.target.value) })}
          />
        </div>
        <input
          className="slider"
          type="range"
          min="1"
          max={maxPhaseBins}
          step="1"
          value={Math.min(design.phase_bins, maxPhaseBins)}
          onChange={(event) => onDesignChange({ phase_bins: Number(event.target.value) })}
        />
        <div className="helper-row">
          <span>Max from selected model config: {maxPhaseBins}</span>
          <span>Effective: {effectivePhaseBins}</span>
        </div>
        {phaseOverMax && (
          <div className={phasePolicy === "cap" ? "notice compact" : "warning compact"}>
            <AlertTriangle size={16} />
            <span>
              {phasePolicy === "cap"
                ? `Will run at max: ${maxPhaseBins}`
                : `Requested phase bins ${design.phase_bins} exceeds configured max ${maxPhaseBins}.`}
            </span>
          </div>
        )}
      </div>

      <div className="setting-grid two">
        <label>
          <span className="field-label">Grid nx</span>
          <input
            className="number-input"
            type="number"
            min="8"
            max="2048"
            value={design.resolution_nx}
            onChange={(event) => onDesignChange({ resolution_nx: Number(event.target.value) })}
          />
        </label>
        <label>
          <span className="field-label">Grid ny</span>
          <input
            className="number-input"
            type="number"
            min="8"
            max="2048"
            value={design.resolution_ny}
            onChange={(event) => onDesignChange({ resolution_ny: Number(event.target.value) })}
          />
        </label>
      </div>

      <section className="table-section">
        <div className="section-title-row">
          <h2>Cylinders</h2>
          <button className="icon-button" type="button" onClick={onAddCylinder} disabled={!canAdd} title="Add cylinder">
            <Plus size={17} />
          </button>
        </div>
        <div className="cylinder-table">
          <div className="table-header">
            <span>ID</span>
            <span>X</span>
            <span>Y</span>
            <span />
          </div>
          {design.cylinders.map((cylinder, index) => (
            <div
              className={`table-row ${selectedCylinder === index ? "selected" : ""}`}
              key={`cylinder-${index}`}
              onClick={() => onSelectCylinder(index)}
            >
              <button type="button" className="cylinder-id" onClick={() => onSelectCylinder(index)}>
                C{index}
              </button>
              <input
                type="number"
                value={formatNumber(cylinder.x)}
                min="0"
                max={config?.domain_length_x ?? 24}
                step="0.05"
                onChange={(event) => onCylinderChange(index, { ...cylinder, x: Number(event.target.value) })}
              />
              <input
                type="number"
                value={formatNumber(cylinder.y)}
                min="0"
                max={config?.domain_length_y ?? 12}
                step="0.05"
                onChange={(event) => onCylinderChange(index, { ...cylinder, y: Number(event.target.value) })}
              />
              <button
                type="button"
                className="icon-button ghost"
                onClick={(event) => {
                  event.stopPropagation();
                  onDeleteCylinder(index);
                }}
                title={`Delete C${index}`}
              >
                <Trash2 size={15} />
              </button>
            </div>
          ))}
        </div>
      </section>

      <div className="button-row">
        <button className="secondary-button" type="button" onClick={onReset}>
          <RotateCcw size={16} />
          Reset
        </button>
        <button className="secondary-button" type="button" onClick={onRandomize}>
          <Dices size={16} />
          Random
        </button>
      </div>

      <button className="run-button" type="button" onClick={onRun} disabled={runDisabled}>
        <Play size={18} />
        {isRunning ? "Running inference..." : "Run inference"}
      </button>

      <div className="validation-list">
        {runError && (
          <div className="warning">
            <AlertTriangle size={16} />
            <span>{runError}</span>
          </div>
        )}
        {validation?.warnings.map((warning) => (
          <div className={validation.valid ? "notice" : "warning"} key={warning}>
            <AlertTriangle size={16} />
            <span>{warning}</span>
          </div>
        ))}
        {validation?.valid && validation.warnings.length === 0 && <div className="success">Design is valid.</div>}
      </div>
    </aside>
  );
}

import { Database, GitBranch } from "lucide-react";
import type { Mode, ModelEntry } from "../types";

interface Props {
  models: ModelEntry[];
  selectedModelId: string;
  mode: Mode;
  onModelChange: (modelId: string) => void;
  onModeChange: (mode: Mode) => void;
}

export default function CheckpointSelector({
  models,
  selectedModelId,
  mode,
  onModelChange,
  onModeChange,
}: Props) {
  const selectedModel = models.find((model) => model.id === selectedModelId);
  const generativeReady = models.some((model) => model.mode === "generative" && model.enabled && model.stage === 2);
  const statusClass = selectedModel?.available
    ? "available"
    : selectedModel?.status === "stage2_pending"
      ? "pending"
      : "error";

  return (
    <div className="selector-stack">
      <label className="field-label" htmlFor="model-select">
        <Database size={15} />
        Model checkpoint
      </label>
      <select
        id="model-select"
        className="control"
        value={selectedModelId}
        onChange={(event) => onModelChange(event.target.value)}
      >
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {model.label} - {model.status}
          </option>
        ))}
      </select>

      <div className="segmented" aria-label="Mode selector">
        <button
          type="button"
          className={mode === "deterministic" ? "active" : ""}
          onClick={() => onModeChange("deterministic")}
        >
          Deterministic
        </button>
        <button
          type="button"
          className={mode === "generative" ? "active" : ""}
          onClick={() => onModeChange("generative")}
          disabled={!generativeReady && selectedModel?.mode !== "generative"}
        >
          <GitBranch size={14} />
          Generative
        </button>
      </div>

      {selectedModel && (
        <div className={`model-status ${selectedModel.available ? "ok" : "warn"}`}>
          <span className={`status-dot ${statusClass}`} />
          <strong>{selectedModel.mode}</strong>
          <span>{selectedModel.status}</span>
          <span>{selectedModel.checkpoint_exists ? "checkpoint ok" : "checkpoint missing"}</span>
          <span>{selectedModel.config_exists ? "config ok" : "config missing"}</span>
          {selectedModel.reason_unavailable && <p>{selectedModel.reason_unavailable}</p>}
          {selectedModel.note && <p>{selectedModel.note}</p>}
          {selectedModel.error && <p>{selectedModel.error}</p>}
        </div>
      )}
    </div>
  );
}

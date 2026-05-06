import { SlidersHorizontal } from "lucide-react";
import type { InverseTargetPreset, KpiTargetMode, KpiTargetSpec } from "../../types";

interface Props {
  rows: KpiTargetSpec[];
  presets: InverseTargetPreset[];
  selectedPreset: string;
  onRowsChange: (rows: KpiTargetSpec[]) => void;
  onPresetSelect: (name: string) => void;
}

const modes: KpiTargetMode[] = ["exact", "range", "max", "min", "minimize", "maximize"];

function numberOrNull(value: string) {
  if (value.trim() === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatInput(value: number | null | undefined) {
  return value === null || value === undefined ? "" : String(value);
}

export default function KpiTargetEditor({
  rows,
  presets,
  selectedPreset,
  onRowsChange,
  onPresetSelect,
}: Props) {
  const patchRow = (index: number, patch: Partial<KpiTargetSpec>) => {
    onRowsChange(rows.map((row, rowIndex) => (rowIndex === index ? { ...row, ...patch } : row)));
  };

  return (
    <section className="inverse-card kpi-target-editor">
      <div className="section-title-row">
        <h2>
          <SlidersHorizontal size={16} />
          KPI targets
        </h2>
        <select className="control compact-control" value={selectedPreset} onChange={(event) => onPresetSelect(event.target.value)}>
          <option value="">Preset...</option>
          {presets.map((preset) => (
            <option key={preset.name} value={preset.name}>
              {preset.label}
            </option>
          ))}
        </select>
      </div>

      <div className="kpi-target-table">
        <div className="kpi-target-header">
          <span />
          <span>KPI</span>
          <span>Mode</span>
          <span>Value</span>
          <span>Low</span>
          <span>High</span>
          <span>Weight</span>
        </div>
        {rows.map((row, index) => (
          <div className={`kpi-target-row ${row.enabled ? "enabled" : ""}`} key={row.name}>
            <input
              type="checkbox"
              checked={row.enabled}
              onChange={(event) => patchRow(index, { enabled: event.target.checked })}
              aria-label={`Enable ${row.name}`}
            />
            <span className="kpi-name-cell" title={row.name}>
              {row.name.replace(/_/g, " ")}
            </span>
            <select className="control" value={row.mode} onChange={(event) => patchRow(index, { mode: event.target.value as KpiTargetMode })}>
              {modes.map((mode) => (
                <option key={mode} value={mode}>
                  {mode}
                </option>
              ))}
            </select>
            <input
              className="number-input"
              type="number"
              step="0.001"
              value={formatInput(row.value)}
              disabled={!["exact", "minimize", "maximize"].includes(row.mode)}
              onChange={(event) => patchRow(index, { value: numberOrNull(event.target.value) })}
            />
            <input
              className="number-input"
              type="number"
              step="0.001"
              value={formatInput(row.low)}
              disabled={!["range", "min"].includes(row.mode)}
              onChange={(event) => patchRow(index, { low: numberOrNull(event.target.value) })}
            />
            <input
              className="number-input"
              type="number"
              step="0.001"
              value={formatInput(row.high)}
              disabled={!["range", "max"].includes(row.mode)}
              onChange={(event) => patchRow(index, { high: numberOrNull(event.target.value) })}
            />
            <input
              className="number-input"
              type="number"
              min="0"
              step="0.1"
              value={row.weight}
              onChange={(event) => patchRow(index, { weight: Number(event.target.value) })}
            />
          </div>
        ))}
      </div>
    </section>
  );
}

import { Activity, Layers } from "lucide-react";
import { useEffect } from "react";
import { API_BASE, frameUrl } from "../api";
import HypergraphOverlay from "./HypergraphOverlay";
import PlaybackControls from "./PlaybackControls";
import type { FieldName, JobResult, Mode } from "../types";

interface Props {
  result: JobResult | null;
  field: FieldName;
  frame: number;
  isPlaying: boolean;
  speed: number;
  showHypergraph: boolean;
  mode: Mode;
  onFieldChange: (field: FieldName) => void;
  onFrameChange: (frame: number) => void;
  onPlayingChange: (playing: boolean) => void;
  onSpeedChange: (speed: number) => void;
  onShowHypergraphChange: (show: boolean) => void;
}

const fields: FieldName[] = ["omega", "u", "v", "p"];

export default function FlowViewer({
  result,
  field,
  frame,
  isPlaying,
  speed,
  showHypergraph,
  mode,
  onFieldChange,
  onFrameChange,
  onPlayingChange,
  onSpeedChange,
  onShowHypergraphChange,
}: Props) {
  const frameCount = result?.render.frame_count ?? result?.domain.phase_bins ?? 0;
  const activeScale = result?.render.fields?.[field];
  const activeFrame = result ? frameUrl(result.job_id, field, frame) : null;
  const rendering = result?.rendering ?? result?.render;
  const smoothing = rendering?.display_smoothing ?? true;
  const renderLabel = result
    ? smoothing
      ? `display: ${rendering?.render_interpolation ?? "bicubic"} x${result.render.display_scale ?? 3}`
      : "raw pixels"
    : null;

  useEffect(() => {
    if (!isPlaying || !frameCount) return;
    const delay = Math.max(35, 240 / speed);
    const timer = window.setInterval(() => onFrameChange((frame + 1) % frameCount), delay);
    return () => window.clearInterval(timer);
  }, [frame, frameCount, isPlaying, onFrameChange, speed]);

  return (
    <section className="panel flow-panel">
      <div className="panel-heading inline">
        <div>
          <p className="eyebrow">Panel C</p>
          <h2>Flow visualization</h2>
        </div>
        <span className={`mode-badge ${mode}`}>{mode}</span>
      </div>

      <div className="flow-toolbar">
        <label className="field-select">
          <span>Field</span>
          <select className="control" value={field} onChange={(event) => onFieldChange(event.target.value as FieldName)}>
            {fields.map((fieldName) => (
              <option key={fieldName} value={fieldName}>
                {fieldName}
              </option>
            ))}
          </select>
        </label>
        <label className="toggle-row">
          <input
            type="checkbox"
            checked={showHypergraph}
            disabled={!result?.hypergraph}
            onChange={(event) => onShowHypergraphChange(event.target.checked)}
          />
          <Layers size={15} />
          Hypergraph
        </label>
        {activeScale && (
          <span className="scale-chip">
            vmin {activeScale.vmin.toFixed(3)} / vmax {activeScale.vmax.toFixed(3)}
          </span>
        )}
        {renderLabel && <span className="scale-chip">{renderLabel}</span>}
      </div>

      <div className="flow-stage">
        {activeFrame ? (
          <>
            <img
              src={activeFrame}
              alt={`${field} field frame ${frame}`}
              className={`flow-frame ${smoothing ? "smooth" : "pixelated"}`}
            />
            {showHypergraph && result?.hypergraph && (
              <HypergraphOverlay
                hypergraph={result.hypergraph}
                width={960}
                height={480}
                domainLengthX={result.domain.length_x}
                domainLengthY={result.domain.length_y}
              />
            )}
          </>
        ) : (
          <div className="empty-state">
            <Activity size={36} />
            <p>Run inference to render the full-cycle field frames.</p>
            <span>Backend: {API_BASE}</span>
          </div>
        )}
      </div>

      <PlaybackControls
        isPlaying={isPlaying}
        frame={frame}
        frameCount={frameCount}
        speed={speed}
        onPlayingChange={onPlayingChange}
        onFrameChange={onFrameChange}
        onSpeedChange={onSpeedChange}
      />
      {result && (
        <div className="flow-caption" title={rendering?.note ?? "KPI calculations use raw model arrays."}>
          KPI computed from raw grid
        </div>
      )}
    </section>
  );
}

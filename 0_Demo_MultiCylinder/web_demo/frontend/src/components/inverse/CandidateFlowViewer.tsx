import { Activity, Film } from "lucide-react";
import { useMemo, useState } from "react";
import { inverseFileUrl } from "../../api";
import type { FieldName, InverseCandidate } from "../../types";

interface Props {
  candidate: InverseCandidate | null;
}

const fields: FieldName[] = ["omega", "u", "v", "p"];

export default function CandidateFlowViewer({ candidate }: Props) {
  const [field, setField] = useState<FieldName>("omega");
  const [frame, setFrame] = useState(0);
  const frameUrls = candidate?.quick_validation?.frame_urls ?? candidate?.frame_urls;
  const imageUrl = useMemo(() => {
    const quick = candidate?.image_urls?.ml_cycle ?? candidate?.image_urls?.ml_flow;
    const simulated = candidate?.image_urls?.ml_vs_simulation_cycle ?? candidate?.image_urls?.simulation_flow;
    return simulated ?? quick ?? null;
  }, [candidate]);
  const activeFrames = frameUrls?.[field] ?? [];
  const activeFrame = activeFrames.length ? inverseFileUrl(activeFrames[Math.min(frame, activeFrames.length - 1)]) : null;

  return (
    <section className="inverse-card inverse-flow-viewer">
      <div className="section-title-row">
        <h2>
          <Film size={16} />
          Candidate flow
        </h2>
        <span className="mode-badge deterministic">{candidate?.verifier_backend ?? "idle"}</span>
      </div>
      {activeFrames.length > 0 && (
        <div className="flow-toolbar inverse-flow-toolbar">
          <label className="field-select">
            <span>Field</span>
            <select className="control" value={field} onChange={(event) => setField(event.target.value as FieldName)}>
              {fields.map((fieldName) => (
                <option key={fieldName} value={fieldName}>
                  {fieldName}
                </option>
              ))}
            </select>
          </label>
          <input
            className="slider"
            type="range"
            min="0"
            max={Math.max(activeFrames.length - 1, 0)}
            value={Math.min(frame, Math.max(activeFrames.length - 1, 0))}
            onChange={(event) => setFrame(Number(event.target.value))}
          />
        </div>
      )}
      <div className="candidate-flow-stage">
        {activeFrame ? (
          <img src={activeFrame} alt={`${field} frame`} className="candidate-flow-image" />
        ) : imageUrl ? (
          <img src={inverseFileUrl(imageUrl)} alt="Candidate verified flow" className="candidate-flow-image" />
        ) : (
          <div className="empty-state">
            <Activity size={32} />
            <p>Select a candidate to inspect its saved quicklook.</p>
          </div>
        )}
      </div>
    </section>
  );
}

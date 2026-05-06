import { ChevronLeft, ChevronRight, ClipboardCheck, Download, FlaskConical, Send } from "lucide-react";
import type { Cylinder, InverseCandidate } from "../../types";

interface Props {
  candidates: InverseCandidate[];
  activeIndex: number;
  domainLengthX: number;
  domainLengthY: number;
  isQuickValidating: boolean;
  isSimulating: boolean;
  onActiveIndexChange: (index: number) => void;
  onQuickValidate: (candidate: InverseCandidate) => void;
  onSimulationValidate: (candidate: InverseCandidate) => void;
  onUseInForwardMode: (candidate: InverseCandidate) => void;
}

function metric(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "n/a";
  return value.toPrecision(4);
}

function DesignPreview({ centers, lx, ly }: { centers: number[][]; lx: number; ly: number }) {
  return (
    <svg className="candidate-design-svg" viewBox={`0 0 ${lx} ${ly}`} role="img" aria-label="Candidate design">
      <rect x="0" y="0" width={lx} height={ly} className="domain-bg" />
      <rect x="0" y="0" width={lx} height={ly} className="domain-border" />
      {centers.map(([x, y], index) => (
        <g key={`${x}-${y}-${index}`} transform={`translate(${x} ${ly - y})`} className="cylinder-node selected">
          <circle r="0.5" />
          <text x="0" y="-0.72">
            C{index}
          </text>
        </g>
      ))}
    </svg>
  );
}

function exportJson(candidate: InverseCandidate) {
  const blob = new Blob([JSON.stringify(candidate.raw ?? candidate, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${candidate.id}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

export default function CandidateCarousel({
  candidates,
  activeIndex,
  domainLengthX,
  domainLengthY,
  isQuickValidating,
  isSimulating,
  onActiveIndexChange,
  onQuickValidate,
  onSimulationValidate,
  onUseInForwardMode,
}: Props) {
  const candidate = candidates[activeIndex] ?? null;
  const next = (delta: number) => {
    if (!candidates.length) return;
    onActiveIndexChange((activeIndex + delta + candidates.length) % candidates.length);
  };

  return (
    <section className="inverse-card candidate-carousel">
      <div className="candidate-carousel-top">
        <button className="icon-button" type="button" onClick={() => next(-1)} disabled={candidates.length < 2} title="Previous candidate">
          <ChevronLeft size={17} />
        </button>
        <div>
          <p className="eyebrow">Candidate</p>
          <h2>
            {candidate ? `Rank ${candidate.rank ?? "-"} · sample ${candidate.sample_index}` : "No candidate yet"}
          </h2>
        </div>
        <button className="icon-button" type="button" onClick={() => next(1)} disabled={candidates.length < 2} title="Next candidate">
          <ChevronRight size={17} />
        </button>
      </div>

      {candidate ? (
        <>
          <DesignPreview centers={candidate.centers} lx={domainLengthX} ly={domainLengthY} />
          <div className="candidate-stat-grid">
            <span>score <strong>{metric(candidate.score)}</strong></span>
            <span>count <strong>{candidate.count}</strong></span>
            <span>valid <strong>{String(candidate.validity?.valid ?? "n/a")}</strong></span>
            <span>min distance <strong>{metric(candidate.validity?.min_pair_distance as number | undefined)}</strong></span>
          </div>
          <div className="candidate-action-row">
            <button className="secondary-button" type="button" disabled={isQuickValidating} onClick={() => onQuickValidate(candidate)}>
              <ClipboardCheck size={16} />
              {isQuickValidating ? "Validating..." : "Quick validate"}
            </button>
            <button className="secondary-button" type="button" disabled={isSimulating} onClick={() => onSimulationValidate(candidate)}>
              <FlaskConical size={16} />
              {isSimulating ? "Starting..." : "Simulation validate"}
            </button>
            <button className="secondary-button" type="button" onClick={() => onUseInForwardMode(candidate)}>
              <Send size={16} />
              Use in forward
            </button>
            <button className="icon-button" type="button" title="Export JSON" onClick={() => exportJson(candidate)}>
              <Download size={16} />
            </button>
          </div>
        </>
      ) : (
        <div className="empty-state">Run inverse design to populate the candidate carousel.</div>
      )}
    </section>
  );
}

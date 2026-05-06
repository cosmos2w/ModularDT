import { BarChart3 } from "lucide-react";
import type { InverseCandidate } from "../../types";

interface Props {
  candidate: InverseCandidate | null;
}

function metric(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "n/a";
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)) return value.toExponential(2);
  return value.toFixed(4);
}

function targetLabel(row: InverseCandidate["kpi_comparison"][string]) {
  const target = row.target;
  if (target.mode === "range") return `${metric(target.low)} to ${metric(target.high)}`;
  if (target.mode === "max") return `<= ${metric(target.high ?? target.value)}`;
  if (target.mode === "min") return `>= ${metric(target.low ?? target.value)}`;
  if (target.value !== undefined) return metric(target.value);
  return String(target.mode);
}

function width(value: number | null | undefined, row: InverseCandidate["kpi_comparison"][string]) {
  const values = [value, row.achieved, row.simulation, row.target.value, row.target.low, row.target.high].filter(
    (item): item is number => typeof item === "number" && Number.isFinite(item),
  );
  const max = Math.max(...values.map((item) => Math.abs(item)), 1e-9);
  return `${Math.min(100, Math.abs(Number(value ?? 0)) / max * 100)}%`;
}

export default function CandidateKpiComparison({ candidate }: Props) {
  const rows = Object.entries(candidate?.kpi_comparison ?? {});

  return (
    <section className="inverse-card kpi-comparison">
      <div className="section-title-row">
        <h2>
          <BarChart3 size={16} />
          KPI comparison
        </h2>
        <span className="count-pill">{rows.length ? `${rows.length} active` : "idle"}</span>
      </div>
      {rows.length ? (
        <div className="comparison-list">
          {rows.map(([name, row]) => (
            <div className={`comparison-row ${row.pass === true ? "pass" : row.pass === false ? "fail" : ""}`} key={name}>
              <div className="comparison-label">
                <strong>{name.replace(/_/g, " ")}</strong>
                <span>target {targetLabel(row)}</span>
              </div>
              <div className="comparison-bars">
                <div className="metric-bar target" style={{ width: width(row.target.value ?? row.target.high ?? row.target.low, row) }}>
                  target
                </div>
                <div className="metric-bar ml" style={{ width: width(row.achieved, row) }}>
                  ML {metric(row.achieved)}
                </div>
                {row.simulation !== null && row.simulation !== undefined && (
                  <div className="metric-bar sim" style={{ width: width(row.simulation, row) }}>
                    sim {metric(row.simulation)}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="empty-mini">Candidate KPI comparison appears after a completed inverse job.</div>
      )}
    </section>
  );
}

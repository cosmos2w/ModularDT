import { TerminalSquare } from "lucide-react";
import type { SimulationValidationStatus } from "../../types";

interface Props {
  status: SimulationValidationStatus | null;
}

export default function SimulationProgress({ status }: Props) {
  return (
    <section className="inverse-card simulation-progress">
      <div className="section-title-row">
        <h2>
          <TerminalSquare size={16} />
          Simulation
        </h2>
        <span className="count-pill">{status?.status ?? "not started"}</span>
      </div>
      {status?.error && <div className="warning compact">{status.error}</div>}
      <pre className="log-tail">
        {(status?.log_tail?.length ? status.log_tail : ["Real simulation validation log tail appears here."]).join("\n")}
      </pre>
    </section>
  );
}

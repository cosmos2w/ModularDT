import type { Hypergraph } from "../types";

interface Props {
  hypergraph: Hypergraph | null;
  width: number;
  height: number;
  domainLengthX: number;
  domainLengthY: number;
}

const PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#7c3aed", "#be123c"];

function sx(x: number, domainLengthX: number, width: number) {
  return (x / domainLengthX) * width;
}

function sy(y: number, domainLengthY: number, height: number) {
  return height - (y / domainLengthY) * height;
}

export default function HypergraphOverlay({ hypergraph, width, height, domainLengthX, domainLengthY }: Props) {
  if (!hypergraph) return null;
  const cylinderById = new Map(hypergraph.cylinders.map((cyl) => [cyl.id, cyl]));

  return (
    <svg className="hypergraph-overlay" viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
      {hypergraph.links
        .filter((link) => link.type === "cylinder-hyperedge")
        .map((link, index) => {
          const cyl = cylinderById.get(link.source);
          const edge = hypergraph.hyperedges.find((candidate) => candidate.id === link.target);
          if (!cyl || !edge?.source) return null;
          return (
            <line
              key={`link-${index}`}
              x1={sx(cyl.x, domainLengthX, width)}
              y1={sy(cyl.y, domainLengthY, height)}
              x2={sx(edge.source.x, domainLengthX, width)}
              y2={sy(edge.source.y, domainLengthY, height)}
              className="hyper-link"
              style={{ opacity: Math.min(0.75, 0.15 + link.weight) }}
            />
          );
        })}

      {hypergraph.env_tokens.map((token) => (
        <circle
          key={`env-${token.id}`}
          cx={sx(token.x, domainLengthX, width)}
          cy={sy(token.y, domainLengthY, height)}
          r="2.6"
          fill={PALETTE[(token.group ?? 0) % PALETTE.length]}
          opacity={Math.max(0.25, token.confidence ?? 0.45)}
        />
      ))}

      {hypergraph.hyperedges.map((edge, index) => {
        const source = edge.source;
        const wake = edge.wake;
        const axis = edge.axis;
        const color = PALETTE[index % PALETTE.length];
        const sourceX = source ? sx(source.x, domainLengthX, width) : null;
        const sourceY = source ? sy(source.y, domainLengthY, height) : null;
        const wakeX = wake ? sx(wake.x, domainLengthX, width) : null;
        const wakeY = wake ? sy(wake.y, domainLengthY, height) : null;
        return (
          <g key={edge.id}>
            {sourceX !== null && sourceY !== null && (
              <circle cx={sourceX} cy={sourceY} r="6" fill="none" stroke={color} strokeWidth="2.2" />
            )}
            {wakeX !== null && wakeY !== null && (
              <circle cx={wakeX} cy={wakeY} r="4.5" fill={color} opacity="0.9" />
            )}
            {wakeX !== null && wakeY !== null && axis && (
              <line
                x1={wakeX}
                y1={wakeY}
                x2={wakeX + axis.x * 34}
                y2={wakeY - axis.y * 34}
                stroke={color}
                strokeWidth="2"
                markerEnd="url(#arrow-head)"
              />
            )}
          </g>
        );
      })}

      <defs>
        <marker id="arrow-head" markerWidth="8" markerHeight="8" refX="5" refY="3" orient="auto">
          <path d="M0,0 L0,6 L6,3 z" fill="#475569" />
        </marker>
      </defs>
    </svg>
  );
}

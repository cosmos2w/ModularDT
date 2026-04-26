import { Grid3X3, Plus, Trash2 } from "lucide-react";
import { useMemo, useRef } from "react";
import type { Cylinder, ModelConfig, ValidationResult } from "../types";

interface Props {
  cylinders: Cylinder[];
  config: ModelConfig | null;
  re: number;
  validation: ValidationResult | null;
  selectedCylinder: number | null;
  onCylinderChange: (index: number, cylinder: Cylinder) => void;
  onAddCylinder: () => void;
  onDeleteCylinder: (index: number) => void;
  onSelectCylinder: (index: number | null) => void;
}

const CYL_RADIUS = 0.5;

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function minImageDistance(a: Cylinder, b: Cylinder, lx: number, ly: number) {
  const dx = Math.min(Math.abs(a.x - b.x), lx - Math.abs(a.x - b.x));
  const dy = Math.min(Math.abs(a.y - b.y), ly - Math.abs(a.y - b.y));
  return Math.sqrt(dx * dx + dy * dy);
}

export default function DomainCanvas({
  cylinders,
  config,
  re,
  validation,
  selectedCylinder,
  onCylinderChange,
  onAddCylinder,
  onDeleteCylinder,
  onSelectCylinder,
}: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const lx = config?.domain_length_x ?? validation?.domain_length_x ?? 24;
  const ly = config?.domain_length_y ?? validation?.domain_length_y ?? 12;
  const maxCylinders = config?.max_num_cylinders ?? validation?.max_num_cylinders ?? 8;

  const invalidIndices = useMemo(() => {
    const invalid = new Set<number>();
    cylinders.forEach((cyl, index) => {
      if (cyl.x < 0 || cyl.x >= lx || cyl.y < 0 || cyl.y >= ly) invalid.add(index);
      cylinders.forEach((other, otherIndex) => {
        if (otherIndex <= index) return;
        if (minImageDistance(cyl, other, lx, ly) < 2 * CYL_RADIUS) {
          invalid.add(index);
          invalid.add(otherIndex);
        }
      });
    });
    return invalid;
  }, [cylinders, lx, ly]);

  function clientToDomain(event: { clientX: number; clientY: number }) {
    const svg = svgRef.current;
    if (!svg) return { x: 0, y: 0 };
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return { x: 0, y: 0 };
    const transformed = point.matrixTransform(ctm.inverse());
    return {
      x: clamp(transformed.x, 0, lx - 1e-3),
      y: clamp(ly - transformed.y, 0, ly - 1e-3),
    };
  }

  function gridLines(count: number, max: number) {
    return Array.from({ length: count + 1 }, (_, index) => (index / count) * max);
  }

  const ghosts = cylinders.flatMap((cyl, index) => {
    const offsets: Array<[number, number]> = [];
    if (cyl.x < CYL_RADIUS * 2) offsets.push([lx, 0]);
    if (lx - cyl.x < CYL_RADIUS * 2) offsets.push([-lx, 0]);
    if (cyl.y < CYL_RADIUS * 2) offsets.push([0, ly]);
    if (ly - cyl.y < CYL_RADIUS * 2) offsets.push([0, -ly]);
    return offsets.map(([dx, dy]) => ({ index, x: cyl.x + dx, y: cyl.y + dy }));
  });

  return (
    <section className="panel domain-panel">
      <div className="panel-heading inline">
        <div>
          <p className="eyebrow">Panel B</p>
          <h2>Computation domain · Re = {Number.isFinite(re) ? re.toFixed(1) : "n/a"}</h2>
        </div>
        <div className="toolbar">
          <button className="secondary-button compact" type="button" onClick={onAddCylinder} disabled={cylinders.length >= maxCylinders}>
            <Plus size={16} />
            Add
          </button>
          <button
            className="icon-button"
            type="button"
            title="Delete selected cylinder"
            disabled={selectedCylinder === null}
            onClick={() => selectedCylinder !== null && onDeleteCylinder(selectedCylinder)}
          >
            <Trash2 size={16} />
          </button>
        </div>
      </div>

      <div className="domain-frame">
        <div className="domain-badge">
          Re = {Number.isFinite(re) ? re.toFixed(1) : "n/a"} | N = {cylinders.length} | Periodic | {lx.toFixed(1)} x {ly.toFixed(1)}
        </div>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${lx} ${ly}`}
          className="domain-svg"
          role="img"
          aria-label="Interactive cylinder domain"
        >
          <rect x="0" y="0" width={lx} height={ly} className="domain-bg" />
          {gridLines(12, lx).map((x) => (
            <line key={`x-${x}`} x1={x} x2={x} y1="0" y2={ly} className="grid-line" />
          ))}
          {gridLines(6, ly).map((y) => (
            <line key={`y-${y}`} x1="0" x2={lx} y1={y} y2={y} className="grid-line" />
          ))}
          <rect x="0" y="0" width={lx} height={ly} className="domain-border" />

          {ghosts.map((ghost) => (
            <circle
              key={`ghost-${ghost.index}-${ghost.x}-${ghost.y}`}
              cx={ghost.x}
              cy={ly - ghost.y}
              r={CYL_RADIUS}
              className="cylinder-ghost"
            />
          ))}

          {cylinders.map((cylinder, index) => (
            <g
              key={`cyl-${index}`}
              transform={`translate(${cylinder.x} ${ly - cylinder.y})`}
              className={`cylinder-node ${selectedCylinder === index ? "selected" : ""} ${invalidIndices.has(index) ? "invalid" : ""}`}
              onPointerDown={(event) => {
                event.preventDefault();
                event.currentTarget.setPointerCapture(event.pointerId);
                onSelectCylinder(index);
              }}
              onPointerMove={(event) => {
                if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
                const next = clientToDomain(event);
                onCylinderChange(index, next);
              }}
              onPointerUp={(event) => event.currentTarget.releasePointerCapture(event.pointerId)}
            >
              <circle r={CYL_RADIUS} />
              <text x="0" y="-0.72">
                C{index}
              </text>
            </g>
          ))}
        </svg>
        <div className="domain-meta">
          <span><Grid3X3 size={14} /> {lx.toFixed(1)} x {ly.toFixed(1)}</span>
          <span>{validation?.valid ? "valid geometry" : "needs attention"}</span>
        </div>
      </div>
    </section>
  );
}

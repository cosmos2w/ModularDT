import { ChevronDown, ChevronUp, Gauge } from "lucide-react";
import { useMemo, useState } from "react";
import type { KpiData } from "../types";

interface Props {
  kpis: KpiData | null | undefined;
  frame: number;
  frameCount: number;
}

interface KpiSeries {
  key: string;
  label: string;
  values: number[];
}

const MAIN_KPIS = [
  ["mean_abs_omega", "Mean |omega|"],
  ["enstrophy", "Enstrophy"],
  ["max_abs_omega", "Max |omega|"],
  ["kinetic_energy", "Kinetic energy"],
  ["pressure_range", "Pressure range"],
] as const;

const LABELS: Record<string, string> = {
  mean_abs_omega: "Mean |omega|",
  enstrophy: "Enstrophy",
  max_abs_omega: "Max |omega|",
  kinetic_energy: "Kinetic energy",
  pressure_range: "Pressure range",
  field_mean_u: "Mean u",
  field_mean_v: "Mean v",
  field_mean_p: "Mean p",
  field_mean_omega: "Mean omega",
  field_max_abs_u: "Max |u|",
  field_max_abs_v: "Max |v|",
  field_max_abs_p: "Max |p|",
  field_max_abs_omega: "Max |omega| field",
};

function isNumberSeries(value: unknown): value is number[] {
  return Array.isArray(value) && value.every((item) => typeof item === "number");
}

function labelForKey(key: string) {
  if (LABELS[key]) return LABELS[key];
  return key
    .replace(/^field_/, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function currentValue(series: number[] | undefined, frame: number) {
  if (!series?.length) return null;
  return series[Math.min(frame, series.length - 1)];
}

function formatMetric(value: number | null) {
  if (value === null || !Number.isFinite(value)) return "n/a";
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)) return value.toExponential(2);
  return value.toFixed(4);
}

function collectSeries(kpis: KpiData | null | undefined): KpiSeries[] {
  if (!kpis) return [];
  const byKey = new Map<string, KpiSeries>();
  const add = (key: string, values: unknown, label = labelForKey(key)) => {
    if (!byKey.has(key) && isNumberSeries(values) && values.length) {
      byKey.set(key, { key, label, values });
    }
  };

  MAIN_KPIS.forEach(([key, label]) => add(key, kpis[key], label));
  Object.entries(kpis).forEach(([key, value]) => {
    if (key === "field_mean" || key === "field_max_abs") return;
    add(key, value);
  });
  Object.entries(kpis.field_mean ?? {}).forEach(([field, values]) => add(`field_mean_${field}`, values));
  Object.entries(kpis.field_max_abs ?? {}).forEach(([field, values]) => add(`field_max_abs_${field}`, values));

  return Array.from(byKey.values());
}

function Sparkline({ series, frame }: { series: number[]; frame: number }) {
  const width = 220;
  const height = 64;
  const pad = 8;
  const min = Math.min(...series);
  const max = Math.max(...series);
  const span = Math.max(max - min, 1e-9);
  const points = series.map((value, index) => {
    const x = pad + (index / Math.max(series.length - 1, 1)) * (width - 2 * pad);
    const y = height - pad - ((value - min) / span) * (height - 2 * pad);
    return { x, y };
  });
  const d = points.map((point, index) => `${index === 0 ? "M" : "L"}${point.x},${point.y}`).join(" ");
  const dot = points[Math.min(frame, points.length - 1)] ?? points[0];
  const gridYs = [0.25, 0.5, 0.75].map((ratio) => pad + ratio * (height - 2 * pad));

  return (
    <svg className="sparkline" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="KPI phase curve">
      {gridYs.map((y) => (
        <line key={y} x1={pad} x2={width - pad} y1={y} y2={y} className="plot-gridline" />
      ))}
      <path d={d} className="plot-line" />
      {dot && <circle cx={dot.x} cy={dot.y} r="4" className="plot-dot" />}
    </svg>
  );
}

function KpiCard({ item, frame }: { item: KpiSeries; frame: number }) {
  return (
    <div className="kpi-card">
      <div className="kpi-card-header">
        <span>{item.label}</span>
        <strong>{formatMetric(currentValue(item.values, frame))}</strong>
      </div>
      <Sparkline series={item.values} frame={frame} />
    </div>
  );
}

export default function KpiPanel({ kpis, frame, frameCount }: Props) {
  const [showMore, setShowMore] = useState(false);
  const series = useMemo(() => collectSeries(kpis), [kpis]);
  const visibleSeries = showMore ? series : series.slice(0, 5);
  const hiddenCount = Math.max(series.length - visibleSeries.length, 0);

  return (
    <aside className="panel kpi-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Panel D</p>
          <h2>KPI targets</h2>
        </div>
        <span className="count-pill">
          {frameCount ? `${Math.min(frame + 1, frameCount)} / ${frameCount}` : "idle"}
        </span>
      </div>

      {visibleSeries.length ? (
        <div className="kpi-grid">
          {visibleSeries.map((item) => (
            <KpiCard item={item} frame={frame} key={item.key} />
          ))}
        </div>
      ) : (
        <div className="empty-mini">
          <Gauge size={28} />
          <span>KPI curves appear after inference.</span>
        </div>
      )}

      {series.length > 5 && (
        <button className="secondary-button kpi-more-button" type="button" onClick={() => setShowMore((current) => !current)}>
          {showMore ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          {showMore ? "Show main five" : `Show ${hiddenCount} more`}
        </button>
      )}
    </aside>
  );
}

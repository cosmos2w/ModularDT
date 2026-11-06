import { useEffect, useMemo, useRef } from 'react';
import { FIELD_SPECS, fieldColor, fieldValue, sampleVelocity, validGrid } from './flowMath.js';
import './scientificPanels.css';

const CANVAS_WIDTH = 540;
const DEFAULT_RANGE = { min: 0, max: 12 };
const SPECS = [
  { name: 'XY', axis: 'z', fixed: 2, horizontal: 0, vertical: 1 },
  { name: 'XZ', axis: 'y', fixed: 1, horizontal: 0, vertical: 2 },
  { name: 'YZ', axis: 'x', fixed: 0, horizontal: 1, vertical: 2 },
];

function labelNumber(value, digits = 2) {
  if (!Number.isFinite(value)) return '—';
  return Number(value.toFixed(digits)).toString();
}

function scaleNumber(value) {
  return Number.isFinite(value) ? Number(value.toPrecision(3)).toString() : '—';
}

function centerOfTurbines(scene, axis) {
  const coordinate = axis === 0 ? 'x' : 'y';
  const values = (scene?.turbines || []).map((turbine) => turbine[coordinate]).filter(Number.isFinite);
  if (values.length) return (Math.min(...values) + Math.max(...values)) / 2;
  const min = scene?.domain?.min?.[axis];
  const max = scene?.domain?.max?.[axis];
  return Number.isFinite(min) && Number.isFinite(max) ? (min + max) / 2 : null;
}

// The slice selector in the 3D view uses the same fraction between the first
// and last returned axis; all bounds and positions remain in rotor diameters.
export function panelGeometry(scene, grid, spec) {
  const coordinate = spec.fixed === 2 ? 0.875 : centerOfTurbines(scene, spec.fixed);
  const title = `${spec.name} · ${spec.axis} = ${labelNumber(coordinate, spec.fixed === 2 ? 3 : 2)} D`;
  if (!validGrid(grid) || coordinate === null) return { ...spec, coordinate, title, available: false };
  const axes = [grid.axes.x, grid.axes.y, grid.axes.z];
  const start = axes[spec.fixed][0];
  const end = axes[spec.fixed][axes[spec.fixed].length - 1];
  if (coordinate < start - 1e-6 || coordinate > end + 1e-6) return { ...spec, coordinate, title, available: false };
  return {
    ...spec, coordinate, title, available: true,
    position: Math.min(1, Math.max(0, (coordinate - start) / (end - start))),
    horizontalBounds: [axes[spec.horizontal][0], axes[spec.horizontal][axes[spec.horizontal].length - 1]],
    verticalBounds: [axes[spec.vertical][0], axes[spec.vertical][axes[spec.vertical].length - 1]],
  };
}

// Equal physical distances have equal display lengths, including vertical
// sections. A maximum height only reduces the plot width for tall domains.
function plotRectangle(panel) {
  const availableWidth = CANVAS_WIDTH - 72;
  const horizontalSpan = panel.horizontalBounds[1] - panel.horizontalBounds[0];
  const verticalSpan = panel.verticalBounds[1] - panel.verticalBounds[0];
  const scale = Math.min(availableWidth / horizontalSpan, 350 / verticalSpan);
  const width = horizontalSpan * scale;
  const height = verticalSpan * scale;
  return { left: 58 + (availableWidth - width) / 2, top: 15, width, height };
}

// Exclude a sample if any positive-weight interpolation corner is outside the
// fluid mask. The same mask applies to displayed predictions and their RMSE.
function validMaskSample(grid, point) {
  const mask = grid.fluid_mask;
  const axes = [grid.axes.x, grid.axes.y, grid.axes.z];
  const [nx, ny, nz] = grid.shape;
  const shape = [nx, ny, nz];
  const corners = axes.map((axis, dimension) => {
    let low = 0;
    let high = shape[dimension] - 1;
    while (high - low > 1) {
      const middle = (low + high) >> 1;
      if (axis[middle] <= point[dimension]) low = middle;
      else high = middle;
    }
    const weight = (point[dimension] - axis[low]) / (axis[high] - axis[low]);
    return [[low, 1 - weight], [high, weight]];
  });
  for (const [ix, wx] of corners[0]) for (const [iy, wy] of corners[1]) for (const [iz, wz] of corners[2]) {
    if (wx * wy * wz > 1e-12 && !mask[(ix * ny + iy) * nz + iz]) return false;
  }
  return true;
}

// Every value used for the colors, isolines, and RMSE is sampled at the same
// physical point. A reference-bearing map shows only paired valid samples.
export function samplePanel(grid, panel, field, range) {
  if (!panel.available || !validGrid(grid) || !Number.isFinite(range?.min) || !Number.isFinite(range?.max)) return null;
  const rectangle = plotRectangle(panel);
  const width = Math.max(2, Math.min(160, Math.ceil(rectangle.width / 2.5)));
  const height = Math.max(2, Math.min(120, Math.ceil(width * rectangle.height / rectangle.width)));
  const pixels = new Uint8ClampedArray(width * height * 4);
  const values = new Float64Array(width * height);
  values.fill(NaN);
  const hasReference = grid.reference_velocity?.length === grid.velocity.length;
  const mask = grid.fluid_mask;
  const partialMask = mask?.length === grid.velocity.length / 3 && mask.some((value) => !value);
  let squaredError = 0;
  let referenceCount = 0;
  let sampleCount = 0;
  let minimum = Infinity;
  let maximum = -Infinity;
  for (let row = 0; row < height; row += 1) for (let column = 0; column < width; column += 1) {
    const point = [0, 0, 0];
    point[panel.fixed] = panel.coordinate;
    point[panel.horizontal] = panel.horizontalBounds[0] + column / (width - 1) * (panel.horizontalBounds[1] - panel.horizontalBounds[0]);
    point[panel.vertical] = panel.verticalBounds[1] - row / (height - 1) * (panel.verticalBounds[1] - panel.verticalBounds[0]);
    if (partialMask && !validMaskSample(grid, point)) continue;
    const velocity = sampleVelocity(grid, point);
    if (!velocity) continue;
    const predicted = fieldValue(velocity, field, point, grid);
    if (!Number.isFinite(predicted)) continue;
    let referenceValue;
    if (hasReference) {
      const reference = sampleVelocity(grid, point, grid.reference_velocity);
      referenceValue = reference && fieldValue(reference, field, point, grid);
      if (!Number.isFinite(referenceValue)) continue;
    }
    const index = row * width + column;
    const offset = index * 4;
    const color = fieldColor(predicted, range, field);
    for (let channel = 0; channel < 3; channel += 1) pixels[offset + channel] = Math.round(255 * color[channel]);
    pixels[offset + 3] = 255;
    values[index] = predicted;
    sampleCount += 1;
    minimum = Math.min(minimum, predicted);
    maximum = Math.max(maximum, predicted);
    if (hasReference) {
      squaredError += (predicted - referenceValue) ** 2;
      referenceCount += 1;
    }
  }
  if (!sampleCount) return null;
  return {
    rectangle, width, height, pixels, values, minimum, maximum, referenceCount,
    rmse: referenceCount ? Math.sqrt(squaredError / referenceCount) : null,
  };
}

// Marching squares works in sample coordinates. Invalid/masked corners skip a
// cell; ambiguous saddles use the bilinear asymptotic decider.
function drawContours(context, sampled, range) {
  const { rectangle, width, height, values } = sampled;
  const low = Math.max(sampled.minimum, range.min);
  const high = Math.min(sampled.maximum, range.max);
  if (!(high > low)) return;
  const xScale = rectangle.width / (width - 1);
  const yScale = rectangle.height / (height - 1);
  const vertices = [[0, 0], [1, 0], [1, 1], [0, 1]];
  const edges = [[0, 1], [1, 2], [2, 3], [3, 0]];
  context.save();
  context.beginPath();
  context.rect(rectangle.left, rectangle.top, rectangle.width, rectangle.height);
  context.clip();
  context.strokeStyle = 'rgba(220, 231, 229, 0.48)';
  context.lineWidth = 1.3;
  context.lineCap = 'round';
  context.lineJoin = 'round';
  for (let levelIndex = 1; levelIndex <= 10; levelIndex += 1) {
    const level = low + (high - low) * levelIndex / 11;
    context.beginPath();
    for (let row = 0; row < height - 1; row += 1) for (let column = 0; column < width - 1; column += 1) {
      const corners = [values[row * width + column], values[row * width + column + 1],
        values[(row + 1) * width + column + 1], values[(row + 1) * width + column]];
      if (!corners.every(Number.isFinite)) continue;
      const intersections = [];
      for (let edge = 0; edge < edges.length; edge += 1) {
        const [a, b] = edges[edge];
        if ((corners[a] < level) === (corners[b] < level)) continue;
        const fraction = (level - corners[a]) / (corners[b] - corners[a]);
        intersections[edge] = [
          column + vertices[a][0] + fraction * (vertices[b][0] - vertices[a][0]),
          row + vertices[a][1] + fraction * (vertices[b][1] - vertices[a][1]),
        ];
      }
      const present = intersections.map((point, edge) => point && edge).filter((edge) => edge !== undefined);
      const segment = (a, b) => {
        if (!a || !b || Math.hypot(a[0] - b[0], a[1] - b[1]) < 1e-9) return;
        context.moveTo(rectangle.left + a[0] * xScale, rectangle.top + a[1] * yScale);
        context.lineTo(rectangle.left + b[0] * xScale, rectangle.top + b[1] * yScale);
      };
      if (present.length === 2) {
        segment(intersections[present[0]], intersections[present[1]]);
      } else if (present.length === 4) {
        const f = corners.map((value) => value - level);
        const determinant = f[0] * f[2] - f[1] * f[3];
        const tolerance = Number.EPSILON * 16 * Math.max(1, Math.abs(f[0] * f[2]), Math.abs(f[1] * f[3]));
        if (Math.abs(determinant) <= tolerance) {
          const denominator = f[0] - f[1] + f[2] - f[3];
          const saddle = denominator ? [column + (f[0] - f[3]) / denominator, row + (f[0] - f[1]) / denominator] : [column + 0.5, row + 0.5];
          intersections.forEach((point) => segment(point, saddle));
        } else {
          const pairs = determinant > 0 ? [[0, 1], [2, 3]] : [[0, 3], [1, 2]];
          pairs.forEach(([a, b]) => segment(intersections[a], intersections[b]));
        }
      }
    }
    context.stroke();
  }
  context.restore();
}

function drawPanel(canvas, panel, sampled, scene, range, error) {
  const context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!sampled) return;
  const { rectangle, width, height, pixels } = sampled;
  const buffer = document.createElement('canvas');
  buffer.width = width;
  buffer.height = height;
  buffer.getContext('2d').putImageData(new ImageData(pixels, width, height), 0, 0);
  context.save();
  context.beginPath();
  context.rect(rectangle.left, rectangle.top, rectangle.width, rectangle.height);
  context.clip();
  context.imageSmoothingEnabled = true;
  const xStep = rectangle.width / (width - 1);
  const yStep = rectangle.height / (height - 1);
  context.drawImage(buffer, rectangle.left - xStep / 2, rectangle.top - yStep / 2, rectangle.width + xStep, rectangle.height + yStep);
  context.restore();
  drawContours(context, sampled, range);

  if (panel.name === 'XY') {
    const [xmin, xmax] = panel.horizontalBounds;
    const [ymin, ymax] = panel.verticalBounds;
    context.save();
    context.beginPath();
    context.rect(rectangle.left, rectangle.top, rectangle.width, rectangle.height);
    context.clip();
    for (const turbine of scene?.turbines || []) {
      if (!Number.isFinite(turbine.x) || !Number.isFinite(turbine.y) || !Number.isFinite(turbine.z) || Math.abs(turbine.z - panel.coordinate) > 1e-5) continue;
      const radius = Number.isFinite(turbine.rotor_diameter) ? turbine.rotor_diameter / 2 : 0.5;
      const x = rectangle.left + (turbine.x - xmin) / (xmax - xmin) * rectangle.width;
      const y1 = rectangle.top + (ymax - (turbine.y - radius)) / (ymax - ymin) * rectangle.height;
      const y2 = rectangle.top + (ymax - (turbine.y + radius)) / (ymax - ymin) * rectangle.height;
      context.beginPath();
      context.moveTo(x, y1);
      context.lineTo(x, y2);
      context.lineWidth = 5;
      context.strokeStyle = 'rgba(19, 42, 48, 0.8)';
      context.stroke();
      context.lineWidth = 2;
      context.strokeStyle = '#eff9f5';
      context.stroke();
    }
    context.restore();
  }

  context.strokeStyle = '#819ba1';
  context.lineWidth = 1.3;
  context.beginPath();
  context.moveTo(rectangle.left, rectangle.top);
  context.lineTo(rectangle.left, rectangle.top + rectangle.height);
  context.lineTo(rectangle.left + rectangle.width, rectangle.top + rectangle.height);
  context.stroke();
  context.fillStyle = '#a9bfc2';
  context.strokeStyle = '#819ba1';
  context.font = '20px system-ui, sans-serif';
  context.lineWidth = 1;
  for (const fraction of [0, 0.5, 1]) {
    const x = rectangle.left + fraction * rectangle.width;
    const y = rectangle.top + (1 - fraction) * rectangle.height;
    context.beginPath();
    context.moveTo(x, rectangle.top + rectangle.height);
    context.lineTo(x, rectangle.top + rectangle.height + 5);
    context.moveTo(rectangle.left - 5, y);
    context.lineTo(rectangle.left, y);
    context.stroke();
    context.textAlign = fraction === 0 ? 'left' : fraction === 1 ? 'right' : 'center';
    context.textBaseline = 'top';
    context.fillText(labelNumber(panel.horizontalBounds[0] + fraction * (panel.horizontalBounds[1] - panel.horizontalBounds[0]), 1), x, rectangle.top + rectangle.height + 7);
    context.textAlign = 'right';
    context.textBaseline = 'middle';
    context.fillText(labelNumber(panel.verticalBounds[0] + fraction * (panel.verticalBounds[1] - panel.verticalBounds[0]), 1), rectangle.left - 9, y);
  }
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(`${'XYZ'[panel.horizontal]} / D`, rectangle.left + rectangle.width / 2, rectangle.top + rectangle.height + 40);
  context.save();
  context.translate(14, rectangle.top + rectangle.height / 2);
  context.rotate(-Math.PI / 2);
  context.fillText(`${'XYZ'[panel.vertical]} / D`, 0, 0);
  context.restore();

  // Keep the measurement on the reconstruction itself without a backplate.
  context.save();
  context.textAlign = 'right';
  context.textBaseline = 'top';
  context.font = '600 18px system-ui, sans-serif';
  context.lineWidth = 3;
  context.strokeStyle = 'rgba(12, 29, 35, 0.8)';
  context.fillStyle = '#f0f7f4';
  const textX = rectangle.left + rectangle.width - 7;
  const textY = rectangle.top + 6;
  context.strokeText(error, textX, textY, rectangle.width - 14);
  context.fillText(error, textX, textY, rectangle.width - 14);
  context.restore();
}

function colorbarGradient(range, field) {
  const colors = Array.from({ length: 33 }, (_, index) => {
    const value = range.min + (range.max - range.min) * index / 32;
    const rgb = fieldColor(value, range, field).map((channel) => Math.round(255 * channel));
    return `rgb(${rgb.join(',')})`;
  });
  return `linear-gradient(90deg, ${colors.join(',')})`;
}

function Panel({ scene, grid, field, range, spec, onPlaneSelect }) {
  const canvas = useRef(null);
  const panel = useMemo(() => panelGeometry(scene, grid, spec), [scene, grid, spec]);
  const sampled = useMemo(() => samplePanel(grid, panel, field, range), [grid, panel, field, range]);
  const unit = FIELD_SPECS[field]?.unit || 'm/s';
  const error = sampled?.rmse == null ? 'Reference error unavailable' : `RMSE ${sampled.rmse.toPrecision(3)} ${unit}`;
  const canvasHeight = sampled ? Math.ceil(sampled.rectangle.top + sampled.rectangle.height + 52) : 120;
  useEffect(() => { if (canvas.current) drawPanel(canvas.current, panel, sampled, scene, range, error); }, [panel, sampled, scene, range, error]);
  return <button type="button" className="scientific-panel" disabled={!sampled}
    aria-label={`${panel.title}, ${error}; show this section in 3D`}
    onClick={() => onPlaneSelect?.({ axis: panel.axis, position: panel.position })}>
    <span className="scientific-panel-title">{panel.title}</span>
    <span className="scientific-canvas-wrap"><canvas ref={canvas} width={CANVAS_WIDTH} height={canvasHeight} aria-hidden="true"/></span>
    {sampled && <span className="scientific-scale" aria-hidden="true">
      <span>{scaleNumber(range.min)}</span><span className="scientific-colorbar" style={{ background: colorbarGradient(range, field) }}/><span>{scaleNumber(range.max)} {unit}</span>
    </span>}
  </button>;
}

export default function ScientificPanels({ scene, result, field = 'speed', range = DEFAULT_RANGE, onPlaneSelect }) {
  const grid = result?.grid;
  return <section className="scientific-panels" aria-label="Scientific flow sections">
    {SPECS.map((spec) => <Panel key={spec.name} scene={scene} grid={grid} field={field} range={range} spec={spec} onPlaneSelect={onPlaneSelect}/>)}
  </section>;
}

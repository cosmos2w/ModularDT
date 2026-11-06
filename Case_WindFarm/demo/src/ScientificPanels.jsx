import { useEffect, useMemo, useRef } from 'react';
import { fieldColor, fieldValue, sampleVelocity, validGrid } from './flowMath.js';
import './scientificPanels.css';

const CANVAS_WIDTH = 540;
const SPECS = [
  { name: 'XY', axis: 'z', fixed: 2, horizontal: 0, vertical: 1, canvasHeight: 360 },
  { name: 'XZ', axis: 'y', fixed: 1, horizontal: 0, vertical: 2, canvasHeight: 240 },
  { name: 'YZ', axis: 'x', fixed: 0, horizontal: 1, vertical: 2, canvasHeight: 240 },
];

function labelNumber(value, digits = 2) {
  if (!Number.isFinite(value)) return '—';
  return Number(value.toFixed(digits)).toString();
}

function centerOfTurbines(scene, axis) {
  const coordinate = axis === 0 ? 'x' : 'y';
  const values = (scene?.turbines || []).map((turbine) => turbine[coordinate]).filter(Number.isFinite);
  if (values.length) return (Math.min(...values) + Math.max(...values)) / 2;
  const min = scene?.domain?.min?.[axis];
  const max = scene?.domain?.max?.[axis];
  return Number.isFinite(min) && Number.isFinite(max) ? (min + max) / 2 : null;
}

// The actual plane is specified in rotor diameters. The slice selector in the
// 3-D view uses the same fraction between the first and last returned axis.
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

function plotRectangle(panel) {
  const plotArea = { left: 60, right: 526, top: 15, bottom: panel.canvasHeight - 97 };
  const availableWidth = plotArea.right - plotArea.left;
  const availableHeight = plotArea.bottom - plotArea.top;
  const horizontalSpan = panel.horizontalBounds[1] - panel.horizontalBounds[0];
  const verticalSpan = panel.verticalBounds[1] - panel.verticalBounds[0];
  const scale = Math.min(availableWidth / horizontalSpan, availableHeight / verticalSpan);
  const width = horizontalSpan * scale;
  const height = verticalSpan * scale;
  return {
    left: plotArea.left + (availableWidth - width) / 2,
    top: plotArea.top + (availableHeight - height) / 2,
    width, height,
  };
}

// When a reference contains masked cells, exclude any sample interpolating a
// masked positive-weight corner from both the displayed map and its metric.
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

// These same sampled points generate both canvas pixels and the displayed RMSE.
export function samplePanel(grid, panel, field, range) {
  if (!panel.available || !validGrid(grid)) return null;
  const rectangle = plotRectangle(panel);
  const width = Math.max(2, Math.min(120, Math.ceil(rectangle.width / 3)));
  const height = Math.max(2, Math.min(80, Math.ceil(width * rectangle.height / rectangle.width)));
  const pixels = new Uint8ClampedArray(width * height * 4);
  const hasReference = Array.isArray(grid.reference_velocity) && grid.reference_velocity.length === grid.velocity.length;
  const mask = grid.fluid_mask;
  const partialMask = hasReference && Array.isArray(mask) && mask.length === grid.velocity.length / 3 && mask.some((value) => !value);
  let squaredError = 0;
  let referenceCount = 0;
  for (let row = 0; row < height; row += 1) for (let column = 0; column < width; column += 1) {
    const point = [0, 0, 0];
    point[panel.fixed] = panel.coordinate;
    point[panel.horizontal] = panel.horizontalBounds[0] + column / (width - 1) * (panel.horizontalBounds[1] - panel.horizontalBounds[0]);
    point[panel.vertical] = panel.verticalBounds[1] - row / (height - 1) * (panel.verticalBounds[1] - panel.verticalBounds[0]);
    const offset = (row * width + column) * 4;
    if (partialMask && !validMaskSample(grid, point)) continue;
    const velocity = sampleVelocity(grid, point);
    if (!velocity) continue;
    const predicted = fieldValue(velocity, field);
    const color = fieldColor(predicted, { min: range[0], max: range[1] });
    for (let channel = 0; channel < 3; channel += 1) pixels[offset + channel] = Math.round(255 * color[channel]);
    pixels[offset + 3] = 255;
    if (hasReference) {
      const reference = sampleVelocity(grid, point, grid.reference_velocity);
      if (reference) {
        squaredError += (predicted - fieldValue(reference, field)) ** 2;
        referenceCount += 1;
      }
    }
  }
  return {
    rectangle, width, height, pixels, referenceCount,
    rmse: referenceCount ? Math.sqrt(squaredError / referenceCount) : null,
  };
}

function drawPanel(canvas, panel, sampled, scene) {
  const context = canvas.getContext('2d');
  context.clearRect(0, 0, CANVAS_WIDTH, panel.canvasHeight);
  context.fillStyle = '#edf4f3';
  context.fillRect(0, 0, CANVAS_WIDTH, panel.canvasHeight);
  if (!sampled) {
    context.fillStyle = '#355965';
    context.font = '20px system-ui, sans-serif';
    context.textAlign = 'center';
    context.fillText('Awaiting reconstruction', CANVAS_WIDTH / 2, panel.canvasHeight / 2);
    return;
  }
  const { rectangle, width, height, pixels } = sampled;
  const buffer = document.createElement('canvas');
  buffer.width = width;
  buffer.height = height;
  buffer.getContext('2d').putImageData(new ImageData(pixels, width, height), 0, 0);
  context.imageSmoothingEnabled = false;
  context.drawImage(buffer, rectangle.left, rectangle.top, rectangle.width, rectangle.height);
  if (panel.name === 'XY') {
    const [xmin, xmax] = panel.horizontalBounds;
    const [ymin, ymax] = panel.verticalBounds;
    context.save();
    context.beginPath();
    context.rect(rectangle.left, rectangle.top, rectangle.width, rectangle.height);
    context.clip();
    for (const turbine of scene?.turbines || []) {
      if (!Number.isFinite(turbine.x) || !Number.isFinite(turbine.y) || !Number.isFinite(turbine.z) || Math.abs(turbine.z - panel.coordinate) > 1e-5) continue;
      const x = rectangle.left + (turbine.x - xmin) / (xmax - xmin) * rectangle.width;
      const y1 = rectangle.top + (ymax - (turbine.y - 0.5)) / (ymax - ymin) * rectangle.height;
      const y2 = rectangle.top + (ymax - (turbine.y + 0.5)) / (ymax - ymin) * rectangle.height;
      context.beginPath();
      context.moveTo(x, y1);
      context.lineTo(x, y2);
      context.lineWidth = 5;
      context.strokeStyle = '#102b37';
      context.stroke();
      context.lineWidth = 2;
      context.strokeStyle = '#ffffff';
      context.stroke();
    }
    context.restore();
  }
  context.strokeStyle = '#456a73';
  context.lineWidth = 1.5;
  context.strokeRect(rectangle.left, rectangle.top, rectangle.width, rectangle.height);
  context.fillStyle = '#163b46';
  context.strokeStyle = '#456a73';
  context.font = '22px system-ui, sans-serif';
  context.lineWidth = 1;
  for (const fraction of [0, 0.5, 1]) {
    const x = rectangle.left + fraction * rectangle.width;
    const y = rectangle.top + (1 - fraction) * rectangle.height;
    context.beginPath();
    context.moveTo(x, rectangle.top + rectangle.height);
    context.lineTo(x, rectangle.top + rectangle.height + 6);
    context.moveTo(rectangle.left - 6, y);
    context.lineTo(rectangle.left, y);
    context.stroke();
    context.textAlign = fraction === 0 ? 'left' : fraction === 1 ? 'right' : 'center';
    context.textBaseline = 'top';
    context.fillText(labelNumber(panel.horizontalBounds[0] + fraction * (panel.horizontalBounds[1] - panel.horizontalBounds[0]), 1), x, rectangle.top + rectangle.height + 9);
    context.textAlign = 'right';
    context.textBaseline = 'middle';
    context.fillText(labelNumber(panel.verticalBounds[0] + fraction * (panel.verticalBounds[1] - panel.verticalBounds[0]), 1), rectangle.left - 10, y);
  }
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(`${'XYZ'[panel.horizontal]} / D`, rectangle.left + rectangle.width / 2, rectangle.top + rectangle.height + 46);
  context.save();
  context.translate(14, rectangle.top + rectangle.height / 2);
  context.rotate(-Math.PI / 2);
  context.fillText(`${'XYZ'[panel.vertical]} / D`, 0, 0);
  context.restore();
}

function Panel({ scene, grid, field, range, spec, onPlaneSelect }) {
  const canvas = useRef(null);
  const panel = useMemo(() => panelGeometry(scene, grid, spec), [scene, grid, spec]);
  const sampled = useMemo(() => samplePanel(grid, panel, field, range), [grid, panel, field, range]);
  useEffect(() => { if (canvas.current) drawPanel(canvas.current, panel, sampled, scene); }, [panel, sampled, scene]);
  const error = sampled?.rmse == null ? 'Reference error unavailable' : `RMSE ${sampled.rmse.toPrecision(3)} m/s`;
  return <button type="button" className="scientific-panel" disabled={!sampled}
    aria-label={`${panel.title}, ${error}; show this section in 3D`}
    onClick={() => onPlaneSelect?.({ axis: panel.axis, position: panel.position })}>
    <span className="scientific-panel-title">{panel.title}</span>
    <span className="scientific-canvas-wrap"><canvas ref={canvas} width={CANVAS_WIDTH} height={panel.canvasHeight} aria-hidden="true"/>
      <span className="scientific-panel-error">{error}</span></span>
  </button>;
}

export default function ScientificPanels({ scene, result, field = 'speed', range = [0, 12], onPlaneSelect }) {
  const grid = result?.grid;
  return <section className="scientific-panels" aria-label="Scientific flow sections">
    {SPECS.map((spec) => <Panel key={spec.name} scene={scene} grid={grid} field={field} range={range} spec={spec} onPlaneSelect={onPlaneSelect}/>)}
  </section>;
}

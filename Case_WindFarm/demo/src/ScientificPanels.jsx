import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Expand, RotateCcw, Scan } from 'lucide-react';
import { FIELD_SPECS, fieldColor, validGrid } from './flowMath.js';
import { cropPanel, panView, plotRectangle, samplePanel, zoomView } from './scientificPlotMath.js';
import './scientificPanels.css';

const DEFAULT_RANGE = { min: 0, max: 12 };
const DEFAULT_VIEW = { zoom: 1, x: 0.5, y: 0.5 };
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

// Marching squares works in sample coordinates. Invalid/masked corners skip a
// cell; ambiguous saddles use the bilinear asymptotic decider.
function drawContours(context, sampled, range) {
  const { rectangle, width, height, values } = sampled;
  const low = Math.max(sampled.minimum, range.min);
  const high = Math.min(sampled.maximum, range.max);
  if (!(high > low)) return;
  const xScale = rectangle.width / (width - 1);
  const yScale = rectangle.height / (height - 1);
  // Reuse the color samples for isolines; a bounded contour mesh avoids
  // repeating large marching-squares work during continuous pointer movement.
  const stride = Math.max(1, Math.ceil(Math.sqrt(width * height / 7000)));
  const vertices = [[0, 0], [1, 0], [1, 1], [0, 1]];
  const edges = [[0, 1], [1, 2], [2, 3], [3, 0]];
  const cells = [];
  for (let row = 0; row < height - 1; row += stride) for (let column = 0; column < width - 1; column += stride) {
    const nextRow = Math.min(row + stride, height - 1);
    const nextColumn = Math.min(column + stride, width - 1);
    let fluid = true;
    for (let y = row; y <= nextRow && fluid; y += 1) for (let x = column; x <= nextColumn; x += 1) {
      if (!Number.isFinite(values[y * width + x])) { fluid = false; break; }
    }
    if (fluid) cells.push({ row, column, dx: nextColumn - column, dy: nextRow - row,
      corners: [values[row * width + column], values[row * width + nextColumn],
        values[nextRow * width + nextColumn], values[nextRow * width + column]] });
  }
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
    for (const { row, column, dx, dy, corners } of cells) {
      const intersections = [];
      for (let edge = 0; edge < edges.length; edge += 1) {
        const [a, b] = edges[edge];
        if ((corners[a] < level) === (corners[b] < level)) continue;
        const fraction = (level - corners[a]) / (corners[b] - corners[a]);
        intersections[edge] = [
          column + dx * (vertices[a][0] + fraction * (vertices[b][0] - vertices[a][0])),
          row + dy * (vertices[a][1] + fraction * (vertices[b][1] - vertices[a][1])),
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
          const saddle = denominator ? [column + dx * (f[0] - f[3]) / denominator, row + dy * (f[0] - f[1]) / denominator] : [column + dx * 0.5, row + dy * 0.5];
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

function drawPanel(canvas, panel, sampled, scene, range, error, size) {
  const pixelRatio = size.pixelRatio;
  const pixelWidth = Math.max(1, Math.round(size.width * pixelRatio));
  const pixelHeight = Math.max(1, Math.round(size.height * pixelRatio));
  if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
  if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
  const context = canvas.getContext('2d');
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, size.width, size.height);
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
      context.lineWidth = 3;
      context.strokeStyle = 'rgba(19, 42, 48, 0.8)';
      context.stroke();
      context.lineWidth = 1.2;
      context.strokeStyle = '#eff9f5';
      context.stroke();
    }
    context.restore();
  }

  context.strokeStyle = '#819ba1';
  context.lineWidth = 0.7;
  context.beginPath();
  context.moveTo(rectangle.left, rectangle.top);
  context.lineTo(rectangle.left, rectangle.top + rectangle.height);
  context.lineTo(rectangle.left + rectangle.width, rectangle.top + rectangle.height);
  context.stroke();
  context.fillStyle = '#a9bfc2';
  context.font = '10px system-ui, sans-serif';
  for (const fraction of [0, 0.5, 1]) {
    const x = rectangle.left + fraction * rectangle.width;
    const y = rectangle.top + (1 - fraction) * rectangle.height;
    context.beginPath();
    context.moveTo(x, rectangle.top + rectangle.height);
    context.lineTo(x, rectangle.top + rectangle.height + 3);
    context.moveTo(rectangle.left - 3, y);
    context.lineTo(rectangle.left, y);
    context.stroke();
    context.textAlign = fraction === 0 ? 'left' : fraction === 1 ? 'right' : 'center';
    context.textBaseline = 'top';
    context.fillText(labelNumber(panel.horizontalBounds[0] + fraction * (panel.horizontalBounds[1] - panel.horizontalBounds[0]), 1), x, rectangle.top + rectangle.height + 4);
    context.textAlign = 'right';
    context.textBaseline = 'middle';
    context.fillText(labelNumber(panel.verticalBounds[0] + fraction * (panel.verticalBounds[1] - panel.verticalBounds[0]), 1), rectangle.left - 5, y);
  }
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(`${'XYZ'[panel.horizontal]} / D`, rectangle.left + rectangle.width / 2, rectangle.top + rectangle.height + 26);
  context.save();
  context.translate(8, rectangle.top + rectangle.height / 2);
  context.rotate(-Math.PI / 2);
  context.fillText(`${'XYZ'[panel.vertical]} / D`, 0, 0);
  context.restore();

  // The visible annotation refers to the displayed crop and paired fluid samples.
  context.save();
  context.textAlign = 'right';
  context.textBaseline = 'top';
  context.font = '600 10px system-ui, sans-serif';
  context.lineWidth = 3;
  context.strokeStyle = 'rgba(12, 29, 35, 0.85)';
  context.fillStyle = '#f0f7f4';
  const textX = rectangle.left + rectangle.width - 4;
  const textY = rectangle.top + 4;
  context.strokeText(error, textX, textY, Math.max(1, rectangle.width - 8));
  context.fillText(error, textX, textY, Math.max(1, rectangle.width - 8));
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

const Panel = memo(function Panel({ scene, grid, field, range, spec, onPlaneSelect, onFocus, focused, view, onViewChange }) {
  const canvasRef = useRef(null);
  const wrapRef = useRef(null);
  const dragRef = useRef(null);
  const frameRef = useRef(null);
  const viewRef = useRef(view);
  const [size, setSize] = useState({ width: 0, height: 0, pixelRatio: 1 });
  const panel = useMemo(() => panelGeometry(scene, grid, spec), [scene, grid, spec]);
  const cropped = useMemo(() => panel.available ? cropPanel(panel, view) : panel, [panel, view]);
  const rectangle = useMemo(() => panel.available && size.width > 0 && size.height > 0 ? plotRectangle(panel, size.width, size.height) : null, [panel, size.width, size.height]);
  const sampled = useMemo(() => samplePanel(grid, cropped, field, range, rectangle, size.pixelRatio), [grid, cropped, field, range, rectangle, size.pixelRatio]);
  const gradient = useMemo(() => colorbarGradient(range, field), [range, field]);
  const unit = FIELD_SPECS[field]?.unit || 'm/s';
  const error = sampled?.rmse == null ? 'Reference error unavailable' : `RMSE ${sampled.rmse.toPrecision(3)} ${unit}`;
  viewRef.current = view;

  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return undefined;
    const measure = () => {
      const bounds = wrap.getBoundingClientRect();
      const width = Math.round(bounds.width);
      const height = Math.round(bounds.height);
      const pixelRatio = window.devicePixelRatio || 1;
      setSize(previous => previous.width === width && previous.height === height && previous.pixelRatio === pixelRatio ? previous : { width, height, pixelRatio });
    };
    const observer = new ResizeObserver(measure);
    observer.observe(wrap);
    window.addEventListener('resize', measure);
    measure();
    return () => { observer.disconnect(); window.removeEventListener('resize', measure); };
  }, []);
  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (canvas && size.width && size.height) drawPanel(canvas, cropped, sampled, scene, range, error, size);
  }, [cropped, sampled, scene, range, error, size]);
  useEffect(() => () => { if (frameRef.current !== null) cancelAnimationFrame(frameRef.current); }, []);

  const queueView = next => {
    if (next.zoom === viewRef.current.zoom && next.x === viewRef.current.x && next.y === viewRef.current.y) return;
    viewRef.current = next;
    if (frameRef.current !== null) return;
    frameRef.current = requestAnimationFrame(() => { frameRef.current = null; onViewChange(spec.name, viewRef.current); });
  };
  const wheelRef = useRef(null);
  wheelRef.current = event => {
    if (!sampled || !rectangle) return;
    const bounds = canvasRef.current.getBoundingClientRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    if (x < rectangle.left || x > rectangle.left + rectangle.width || y < rectangle.top || y > rectangle.top + rectangle.height) return;
    event.preventDefault();
    queueView(zoomView(viewRef.current, viewRef.current.zoom * Math.exp(-event.deltaY * 0.0015), (x - rectangle.left) / rectangle.width, (y - rectangle.top) / rectangle.height));
  };
  useEffect(() => {
    const canvas = canvasRef.current;
    const onWheel = event => wheelRef.current?.(event);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, []);
  const pointerDown = event => {
    if (event.button !== 0 || !sampled) return;
    dragRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, view: viewRef.current, moved: false };
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const pointerMove = event => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId || !rectangle) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    if (Math.hypot(dx, dy) > 3) drag.moved = true;
    if (drag.moved) queueView(panView(drag.view, dx, dy, rectangle));
  };
  const pointerEnd = event => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    if (!drag.moved && event.type === 'pointerup') onFocus(spec.name);
  };
  const resetView = () => {
    if (frameRef.current !== null) { cancelAnimationFrame(frameRef.current); frameRef.current = null; }
    viewRef.current = DEFAULT_VIEW;
    onViewChange(spec.name, DEFAULT_VIEW);
  };
  return <article className={`scientific-panel ${focused ? 'is-focused' : ''}`}>
    <div className="scientific-panel-heading"><span className="scientific-panel-title">{panel.title}</span><div className="scientific-panel-controls">
      <button type="button" onClick={resetView} aria-label={`Reset view ${spec.name}`} title="Reset view"><RotateCcw size={13}/></button>
      <button type="button" disabled={!panel.available} onClick={() => onPlaneSelect?.({ axis: panel.axis, position: panel.position })} aria-label={`Show in 3D: ${spec.name}`} title="Show in 3D"><Scan size={13}/><span>3D</span></button>
      {!focused && <button type="button" disabled={!panel.available} onClick={() => onFocus(spec.name)} aria-label={`Expand map: ${spec.name}`} title="Expand map"><Expand size={13}/></button>}
    </div></div>
    <div ref={wrapRef} className="scientific-canvas-wrap"><canvas ref={canvasRef} style={{ visibility: sampled ? 'visible' : 'hidden' }} aria-label={`Expand map: ${spec.name}`} role="button" tabIndex={panel.available ? 0 : -1}
      data-sample-width={sampled?.width || 0} data-sample-height={sampled?.height || 0} data-zoom={view.zoom} data-crop={panel.available ? [...cropped.horizontalBounds, ...cropped.verticalBounds].join(',') : ''} data-rmse={sampled?.rmse ?? ''}
      onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={pointerEnd} onPointerCancel={pointerEnd}
      onKeyDown={event => { if (sampled && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); onFocus(spec.name); } }}/>{!sampled && <span className="scientific-unavailable">Section unavailable</span>}</div>
    <div className="scientific-scale"><span>{scaleNumber(range.min)}</span><span className="scientific-colorbar" style={{ background: gradient }}/><span>{scaleNumber(range.max)} {unit}</span></div>
  </article>;
});

export default function ScientificPanels({ scene, result, field = 'speed', range, onPlaneSelect, fullscreen, focused, onFocus, views, onViewChange }) {
  const grid = result?.grid;
  const displayRange = range || DEFAULT_RANGE;
  return <div className={`scientific-panels ${fullscreen ? 'is-grid' : ''} ${focused ? 'has-focus' : ''}`} aria-label="Scientific flow sections">
    {SPECS.map(spec => <Panel key={spec.name} scene={scene} grid={grid} field={field} range={displayRange} spec={spec} onPlaneSelect={onPlaneSelect} onFocus={onFocus} focused={focused === spec.name} view={views[spec.name]} onViewChange={onViewChange}/>)}
  </div>;
}

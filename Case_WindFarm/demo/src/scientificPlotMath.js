import { fieldColor, fieldValue, validGrid } from './flowMath.js';

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const maskCache = new WeakMap();
const paletteCache = new Map();

// The axes and the plotted data use the same physical aspect ratio at any size.
export function plotRectangle(panel, canvasWidth, canvasHeight) {
  const availableWidth = Math.max(1, canvasWidth - 47);
  const availableHeight = Math.max(1, canvasHeight - 35);
  const xSpan = panel.horizontalBounds[1] - panel.horizontalBounds[0];
  const ySpan = panel.verticalBounds[1] - panel.verticalBounds[0];
  const scale = Math.min(availableWidth / xSpan, availableHeight / ySpan);
  const width = xSpan * scale;
  const height = ySpan * scale;
  return { left: 39 + (availableWidth - width) / 2, top: 7 + (availableHeight - height) / 2, width, height };
}

export function cropPanel(panel, view) {
  const crop = (bounds, center) => {
    const span = bounds[1] - bounds[0];
    return [bounds[0] + (center - 0.5 / view.zoom) * span, bounds[0] + (center + 0.5 / view.zoom) * span];
  };
  return { ...panel, horizontalBounds: crop(panel.horizontalBounds, view.x), verticalBounds: crop(panel.verticalBounds, view.y) };
}

export function zoomView(view, zoom, xFraction, yFraction) {
  const nextZoom = clamp(zoom, 1, 24);
  const x = clamp(xFraction, 0, 1) - 0.5;
  const y = 0.5 - clamp(yFraction, 0, 1);
  const bound = 0.5 / nextZoom;
  return {
    zoom: nextZoom,
    x: clamp(view.x + x / view.zoom - x / nextZoom, bound, 1 - bound),
    y: clamp(view.y + y / view.zoom - y / nextZoom, bound, 1 - bound),
  };
}

export function panView(view, dx, dy, rectangle) {
  const bound = 0.5 / view.zoom;
  return {
    ...view,
    x: clamp(view.x - dx / rectangle.width / view.zoom, bound, 1 - bound),
    y: clamp(view.y + dy / rectangle.height / view.zoom, bound, 1 - bound),
  };
}

function locate(axis, coordinate) {
  let low = 0;
  let high = axis.length - 1;
  while (high - low > 1) {
    const middle = (low + high) >> 1;
    if (axis[middle] <= coordinate) low = middle;
    else high = middle;
  }
  const fraction = clamp((coordinate - axis[low]) / (axis[high] - axis[low]), 0, 1);
  return { low, high, weights: [1 - fraction, fraction] };
}

function palette(field, range) {
  const key = `${field}:${range.min}:${range.max}`;
  if (paletteCache.has(key)) return paletteCache.get(key);
  const colors = Array.from({ length: 1025 }, (_, i) => fieldColor(range.min + (range.max - range.min) * i / 1024, range, field).map(channel => Math.round(channel * 255)));
  if (paletteCache.size > 12) paletteCache.clear();
  paletteCache.set(key, colors);
  return colors;
}

// Samples are interpolated from the returned grid. The cap bounds pointer-work;
// larger canvases still get more samples up to that cap, at device-pixel density.
// Prediction, reference, mask, contours and RMSE use these exact crop samples.
export function samplePanel(grid, panel, field, range, rectangle, pixelRatio = 1) {
  if (!panel?.available || !validGrid(grid) || !rectangle || rectangle.width < 2 || rectangle.height < 2 || !Number.isFinite(range?.min) || !Number.isFinite(range?.max)) return null;
  let width = Math.max(2, Math.ceil(rectangle.width * pixelRatio / 2));
  let height = Math.max(2, Math.ceil(rectangle.height * pixelRatio / 2));
  const factor = Math.min(1, Math.sqrt(128000 / (width * height)));
  width = Math.max(2, Math.floor(width * factor));
  height = Math.max(2, Math.floor(height * factor));
  const pixels = new Uint8ClampedArray(width * height * 4);
  const values = new Float64Array(width * height);
  values.fill(NaN);
  const hasReference = grid.reference_velocity?.length === grid.velocity.length;
  if (!maskCache.has(grid)) maskCache.set(grid, grid.fluid_mask?.length === grid.velocity.length / 3 && grid.fluid_mask.some(value => !value));
  const partialMask = maskCache.get(grid);
  const axes = [grid.axes.x, grid.axes.y, grid.axes.z];
  const [nx, ny, nz] = grid.shape;
  const xBounds = panel.horizontalBounds;
  const yBounds = panel.verticalBounds;
  const horizontal = Array.from({ length: width }, (_, column) => {
    const coordinate = xBounds[0] + column / (width - 1) * (xBounds[1] - xBounds[0]);
    return { coordinate, location: locate(axes[panel.horizontal], coordinate) };
  });
  const vertical = Array.from({ length: height }, (_, row) => {
    const coordinate = yBounds[1] - row / (height - 1) * (yBounds[1] - yBounds[0]);
    return { coordinate, location: locate(axes[panel.vertical], coordinate) };
  });
  const fixed = locate(axes[panel.fixed], panel.coordinate);
  const colors = palette(field, range);
  let squaredError = 0;
  let referenceCount = 0;
  let sampleCount = 0;
  let minimum = Infinity;
  let maximum = -Infinity;
  for (let row = 0; row < height; row += 1) for (let column = 0; column < width; column += 1) {
    const locations = [];
    locations[panel.fixed] = fixed;
    locations[panel.horizontal] = horizontal[column].location;
    locations[panel.vertical] = vertical[row].location;
    const point = [];
    point[panel.fixed] = panel.coordinate;
    point[panel.horizontal] = horizontal[column].coordinate;
    point[panel.vertical] = vertical[row].coordinate;
    const prediction = [0, 0, 0];
    const reference = [0, 0, 0];
    let fluid = true;
    for (let x = 0; x < 2; x += 1) for (let y = 0; y < 2; y += 1) for (let z = 0; z < 2; z += 1) {
      const weight = locations[0].weights[x] * locations[1].weights[y] * locations[2].weights[z];
      if (weight === 0) continue;
      const ix = x ? locations[0].high : locations[0].low;
      const iy = y ? locations[1].high : locations[1].low;
      const iz = z ? locations[2].high : locations[2].low;
      const index = (ix * ny + iy) * nz + iz;
      if (partialMask && weight > 1e-12 && !grid.fluid_mask[index]) { fluid = false; continue; }
      const offset = index * 3;
      for (let component = 0; component < 3; component += 1) {
        prediction[component] += grid.velocity[offset + component] * weight;
        if (hasReference) reference[component] += grid.reference_velocity[offset + component] * weight;
      }
    }
    if (!fluid || !prediction.every(Number.isFinite) || (hasReference && !reference.every(Number.isFinite))) continue;
    const value = fieldValue(prediction, field, point, grid);
    const referenceValue = hasReference ? fieldValue(reference, field, point, grid) : null;
    if (!Number.isFinite(value) || (hasReference && !Number.isFinite(referenceValue))) continue;
    const index = row * width + column;
    const paletteIndex = range.max === range.min ? 512 : Math.round(1024 * clamp((value - range.min) / (range.max - range.min), 0, 1));
    const color = colors[paletteIndex];
    pixels.set([color[0], color[1], color[2], 255], index * 4);
    values[index] = value;
    sampleCount += 1;
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    if (hasReference) { squaredError += (value - referenceValue) ** 2; referenceCount += 1; }
  }
  if (!sampleCount) return null;
  return { rectangle, width, height, pixels, values, minimum, maximum, referenceCount, sampleCount, rmse: referenceCount ? Math.sqrt(squaredError / referenceCount) : null };
}

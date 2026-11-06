// The API returns x-major, z-fast physical velocity vectors on explicit axes.
// Keep all interpolation and path integration in the physical D coordinate frame.
export const FIELD_COLORS = ['#103b86', '#087bc1', '#19d3dd', '#65e6ad', '#f1e399'];

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

export function validGrid(grid) {
  const shape = grid?.shape;
  const bounds = grid?.bounds;
  return Array.isArray(shape) && shape.length === 3 && shape.every((n) => Number.isInteger(n) && n >= 2)
    && Array.isArray(bounds?.min) && Array.isArray(bounds?.max)
    && bounds.min.length === 3 && bounds.max.length === 3
    && bounds.min.every((v, i) => Number.isFinite(v) && Number.isFinite(bounds.max[i]) && bounds.max[i] > v)
    && ['x', 'y', 'z'].every((name, i) => Array.isArray(grid.axes?.[name]) && grid.axes[name].length === shape[i]
      && grid.axes[name].every((value, j, axis) => Number.isFinite(value) && (j === 0 || value > axis[j - 1])))
    && grid.velocity?.length === shape[0] * shape[1] * shape[2] * 3;
}

export function fieldValue(vector, field) {
  if (field === 'ux') return vector[0];
  if (field === 'uy') return vector[1];
  if (field === 'uz') return vector[2];
  return Math.hypot(vector[0], vector[1], vector[2]);
}

export function getFieldRange(grid, field = 'speed') {
  if (!validGrid(grid)) return null;
  const values = grid.velocity;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < values.length; i += 3) {
    const value = fieldValue([values[i], values[i + 1], values[i + 2]], field);
    if (Number.isFinite(value)) {
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
  }
  return Number.isFinite(min) ? { min, max } : null;
}

export function fieldColor(value, range) {
  const t = clamp((value - range.min) / (range.max - range.min || 1), 0, 1) * (FIELD_COLORS.length - 1);
  const index = Math.min(FIELD_COLORS.length - 2, Math.floor(t));
  const fraction = t - index;
  const a = FIELD_COLORS[index];
  const b = FIELD_COLORS[index + 1];
  return [1, 3, 5].map((start) => {
    const c1 = parseInt(a.slice(start, start + 2), 16);
    const c2 = parseInt(b.slice(start, start + 2), 16);
    return (c1 + (c2 - c1) * fraction) / 255;
  });
}

// Return null outside the represented centre bounds or for a non-finite sample.
export function sampleVelocity(grid, point, velocity = grid.velocity) {
  const { shape, axes } = grid;
  const low = [0, 0, 0];
  const high = [0, 0, 0];
  const fraction = [0, 0, 0];
  for (let a = 0; a < 3; a += 1) {
    const axis = axes[['x', 'y', 'z'][a]];
    if (!Number.isFinite(point[a]) || point[a] < axis[0] - 1e-6 || point[a] > axis[shape[a] - 1] + 1e-6) return null;
    const coordinate = clamp(point[a], axis[0], axis[shape[a] - 1]);
    let left = 0;
    let right = shape[a] - 1;
    while (right - left > 1) {
      const middle = (left + right) >> 1;
      if (axis[middle] <= coordinate) left = middle;
      else right = middle;
    }
    low[a] = left;
    high[a] = right;
    fraction[a] = (coordinate - axis[left]) / (axis[right] - axis[left]);
  }
  const out = [0, 0, 0];
  for (let x = 0; x < 2; x += 1) {
    const ix = x ? high[0] : low[0];
    const wx = x ? fraction[0] : 1 - fraction[0];
    for (let y = 0; y < 2; y += 1) {
      const iy = y ? high[1] : low[1];
      const wy = y ? fraction[1] : 1 - fraction[1];
      for (let z = 0; z < 2; z += 1) {
        const iz = z ? high[2] : low[2];
        const wz = z ? fraction[2] : 1 - fraction[2];
        const weight = wx * wy * wz;
        const offset = ((ix * shape[1] + iy) * shape[2] + iz) * 3;
        for (let component = 0; component < 3; component += 1) out[component] += velocity[offset + component] * weight;
      }
    }
  }
  return out.every(Number.isFinite) ? out : null;
}

// These are exactly the mesh vertices used in the displayed slice, in D.
// Other consumers can interpolate prediction, reference and mask at these same
// coordinates when annotating the displayed reconstruction.
export function sliceSampleCoordinates(grid, axisName = 'z', position = 0) {
  if (!validGrid(grid)) return null;
  const axis = { x: 0, y: 1, z: 2 }[axisName] ?? 2;
  const others = [0, 1, 2].filter((index) => index !== axis);
  const axes = ['x', 'y', 'z'].map((name) => grid.axes[name]);
  const minima = axes.map((values) => values[0]);
  const maxima = axes.map((values) => values[values.length - 1]);
  const countA = Math.min(100, Math.max(24, grid.shape[others[0]] * 2));
  const countB = Math.min(80, Math.max(20, grid.shape[others[1]] * 2));
  const fixedCoordinate = minima[axis] + clamp(Number(position) || 0, 0, 1) * (maxima[axis] - minima[axis]);
  const coordinates = [];
  for (let j = 0; j < countB; j += 1) for (let i = 0; i < countA; i += 1) {
    const point = [0, 0, 0];
    point[axis] = fixedCoordinate;
    point[others[0]] = minima[others[0]] + i / (countA - 1) * (maxima[others[0]] - minima[others[0]]);
    point[others[1]] = minima[others[1]] + j / (countB - 1) * (maxima[others[1]] - minima[others[1]]);
    coordinates.push(point);
  }
  return { coordinates, shape: [countA, countB], axis, others, fixedCoordinate };
}

// Resolve the prepared directory without thresholding, fabricating edges, or
// treating the illustrated row selection as an exhaustive model organization.
export function directoryView(organization, scene, address = null) {
  if (organization?.kind !== 'e_anchor_directory' || !Array.isArray(organization.anchors)) return null;
  const anchors = organization.anchors.filter((anchor) => Array.isArray(anchor.position)
    && anchor.position.length === 3 && anchor.position.every(Number.isFinite));
  const eById = new Map(anchors.map((anchor) => [String(anchor.id), anchor]));
  const mById = new Map((scene?.turbines || []).map((member) => [String(member.id), member]));
  const rows = anchors.filter((anchor) => Array.isArray(anchor.members)
    && (address === null || address === undefined || String(anchor.id) === String(address)));
  const links = [];
  let missing = 0;
  for (const row of rows) {
    for (const member of row.members) {
      if (!(member.weight > 0) || !['M', 'E'].includes(member.role)) continue;
      const donor = member.role === 'M' ? mById.get(String(member.id)) : eById.get(String(member.id));
      const donorPosition = member.role === 'M' ? donor && [donor.x, donor.y, donor.z] : donor?.position;
      if (!donorPosition || donorPosition.length !== 3 || !donorPosition.every(Number.isFinite)) {
        missing += 1;
        continue;
      }
      links.push({
        receiverId: row.id, receiverPosition: row.position,
        donorId: member.id, donorRole: member.role, donorPosition,
        weight: member.weight, exchange: member.exchange, self: member.self,
        isSelf: member.role === 'E' && String(member.id) === String(row.id),
      });
    }
  }
  return { anchors, rows, links, missing };
}

// Midpoint integration follows the model vector direction, using deterministic
// seeds inside the sampled volume. Step size is in physical D, not elapsed time.
export function traceStreamlines(grid, hubHeight = 0.875) {
  if (!validGrid(grid)) return [];
  const axes = ['x', 'y', 'z'].map((name) => grid.axes[name]);
  const min = axes.map((axis) => axis[0]);
  const max = axes.map((axis) => axis[axis.length - 1]);
  const span = max.map((v, i) => v - min[i]);
  const step = Math.max(0.08, Math.min(span[0], span[1]) / 72);
  const paths = [];
  const heights = [...new Set([hubHeight - 0.4, hubHeight, hubHeight + 0.4].map((z) => clamp(z, min[2] + 0.025 * span[2], max[2] - 0.025 * span[2])))];
  const xFractions = [0.025, 0.40];
  for (const xf of xFractions) for (const height of heights) for (let yi = 0; yi < 13; yi += 1) {
    const p = [min[0] + span[0] * xf, min[1] + span[1] * (yi + 0.5) / 13, height];
    const path = [];
    for (let k = 0; k < 120; k += 1) {
      const v = sampleVelocity(grid, p);
      if (!v) break;
      const magnitude = Math.hypot(...v);
      if (magnitude < 1e-8) break;
      path.push({ position: [...p], velocity: v });
      const middle = p.map((coord, i) => coord + 0.5 * step * v[i] / magnitude);
      const vm = sampleVelocity(grid, middle);
      if (!vm) break;
      const midMagnitude = Math.hypot(...vm);
      if (midMagnitude < 1e-8) break;
      for (let i = 0; i < 3; i += 1) p[i] += step * vm[i] / midMagnitude;
    }
    if (path.length > 5) paths.push(path);
  }
  return paths;
}

// The API returns x-major, z-fast physical velocity vectors on explicit axes.
// Keep all interpolation and path integration in the physical D coordinate frame.
export const FIELD_SPECS = {
  speed: { label: '|U|', unit: 'm/s', colors: ['#081a42', '#135ca2', '#19a9c5', '#74d8bd', '#f5c66a'], diverging: false, description: 'Velocity magnitude' },
  ux: { label: 'Ux', unit: 'm/s', colors: ['#171b4d', '#365cac', '#4fa5d3', '#c3d9dc', '#eaa268'], diverging: false, description: 'Downstream velocity' },
  uy: { label: 'Uy', unit: 'm/s', colors: ['#15559a', '#68b0cf', '#e6edef', '#e6a070', '#aa3b35'], diverging: true, description: 'Crosswind velocity' },
  uz: { label: 'Uz', unit: 'm/s', colors: ['#15559a', '#68b0cf', '#e6edef', '#e6a070', '#aa3b35'], diverging: true, description: 'Vertical velocity' },
  deficit: { label: 'Ux deficit', unit: 'm/s', colors: ['#1b588f', '#77b8ce', '#e8edeb', '#eeac6e', '#b63c33'], diverging: true, description: 'Training profile Ux(z) − reconstructed Ux' },
};
// Kept as the speed palette for callers that do not select a field.
export const FIELD_COLORS = FIELD_SPECS.speed.colors;

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

// The saved model adds this same TRAIN-only profile to its predicted residual.
// Match its piecewise-linear interpolation and endpoint clamping exactly.
export function referenceProfileUx(grid, z) {
  const profile = grid?.reference_profile;
  const heights = profile?.z_D;
  const values = profile?.velocity_mps;
  if (!Number.isFinite(z) || !Array.isArray(heights) || !heights.length || !Array.isArray(values) || values.length !== heights.length) return NaN;
  if (z <= heights[0]) return values[0]?.[0] ?? NaN;
  const last = heights.length - 1;
  if (z >= heights[last]) return values[last]?.[0] ?? NaN;
  let left = 0;
  let right = last;
  while (right - left > 1) {
    const middle = (left + right) >> 1;
    if (heights[middle] <= z) left = middle;
    else right = middle;
  }
  const fraction = (z - heights[left]) / (heights[right] - heights[left]);
  return values[left][0] + fraction * (values[right][0] - values[left][0]);
}

export function fieldValue(vector, field, point, grid) {
  if (field === 'deficit') return referenceProfileUx(grid, point?.[2]) - vector[0];
  if (field === 'ux') return vector[0];
  if (field === 'uy') return vector[1];
  if (field === 'uz') return vector[2];
  return Math.hypot(vector[0], vector[1], vector[2]);
}

function percentile(sorted, p) {
  const position = (sorted.length - 1) * p;
  const low = Math.floor(position);
  const fraction = position - low;
  return sorted[low] + fraction * (sorted[Math.ceil(position)] - sorted[low]);
}

// Wake contrast is a display scale computed from real samples near the rotors:
// x = -0.5..8 D downstream, |y-y_t| <= 1.5 D and |z-z_hub| <= 0.65 D.
// Values and metrics are never clipped or changed, only the displayed colors.
export function getFieldRange(grid, field = 'speed', options = {}) {
  if (!validGrid(grid)) return null;
  const values = grid.velocity;
  const turbines = (options.turbines || []).filter((turbine) => [turbine.x, turbine.y, turbine.z].every(Number.isFinite));
  const focused = options.contrast === 'wake' && turbines.length > 0;
  const local = [];
  let min = Infinity;
  let max = -Infinity;
  let finiteCount = 0;
  const [nx, ny, nz] = grid.shape;
  for (let ix = 0; ix < nx; ix += 1) for (let iy = 0; iy < ny; iy += 1) for (let iz = 0; iz < nz; iz += 1) {
    const point = [grid.axes.x[ix], grid.axes.y[iy], grid.axes.z[iz]];
    const offset = ((ix * ny + iy) * nz + iz) * 3;
    const value = fieldValue([values[offset], values[offset + 1], values[offset + 2]], field, point, grid);
    if (Number.isFinite(value)) {
      min = Math.min(min, value);
      max = Math.max(max, value);
      finiteCount += 1;
      if (focused && turbines.some((turbine) => point[0] >= turbine.x - 0.5 && point[0] <= turbine.x + 8
        && Math.abs(point[1] - turbine.y) <= 1.5 && Math.abs(point[2] - turbine.z) <= 0.65)) local.push(value);
    }
  }
  if (!Number.isFinite(min)) return null;
  let lower = min;
  let upper = max;
  const useFocus = local.length >= 8;
  if (useFocus) {
    local.sort((a, b) => a - b);
    lower = percentile(local, 0.02);
    upper = percentile(local, 0.98);
  }
  if (FIELD_SPECS[field]?.diverging) {
    const extent = Math.max(Math.abs(lower), Math.abs(upper));
    lower = -extent;
    upper = extent;
  }
  return {
    min: lower, max: upper, clipped: min < lower || max > upper,
    percentiles: useFocus ? [2, 98] : null,
    scope: useFocus ? 'near-turbine rotor band' : 'full grid',
    sampleCount: useFocus ? local.length : finiteCount,
  };
}

export function fieldColor(value, range, field = 'speed') {
  const palette = FIELD_SPECS[field]?.colors || FIELD_COLORS;
  const scaled = range.max === range.min ? 0.5 : clamp((value - range.min) / (range.max - range.min), 0, 1);
  const t = scaled * (palette.length - 1);
  const index = Math.min(palette.length - 2, Math.floor(t));
  const fraction = t - index;
  const a = palette[index];
  const b = palette[index + 1];
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

// Select only what is drawn in the spatial view. The original rows and weights
// remain available to the model and to the exact-membership Venn view.
export function selectSpatialLinks(data, { mode = 'focused' } = {}) {
  const displayMode = mode === 'all' ? 'all' : 'focused';
  const targetMass = 0.95;
  const selected = new Set();
  const indexedRows = new Map();
  data.links.forEach((link, index) => {
    const key = String(link.receiverId);
    if (!indexedRows.has(key)) indexedRows.set(key, []);
    indexedRows.get(key).push({ link, index });
  });

  let fullMass = 0;
  let shownMass = 0;
  let availableMass = 0;
  const rowSummaries = data.rows.map((row) => {
    const candidates = indexedRows.get(String(row.id)) || [];
    const members = row.members.filter((member) => member.weight > 0 && ['M', 'E'].includes(member.role));
    const rowFullMass = members.reduce((sum, member) => sum + member.weight, 0);
    const rowAvailableMass = candidates.reduce((sum, item) => sum + item.link.weight, 0);
    let rowSelectedMass = 0;

    // Preserve all positive M memberships and the actual positive own-E entry.
    for (const item of candidates) {
      if (displayMode === 'all' || item.link.donorRole === 'M' || item.link.isSelf) {
        selected.add(item.index);
        rowSelectedMass += item.link.weight;
      }
    }
    if (displayMode === 'focused') {
      const optional = candidates.filter((item) => !selected.has(item.index))
        .sort((a, b) => b.link.weight - a.link.weight || a.index - b.index);
      for (const item of optional) {
        if (rowSelectedMass >= targetMass * rowAvailableMass) break;
        selected.add(item.index);
        rowSelectedMass += item.link.weight;
      }
    }

    const omittedMembers = candidates.filter((item) => !selected.has(item.index)).map(({ link }) => ({
      role: link.donorRole, id: link.donorId, weight: link.weight,
    }));
    fullMass += rowFullMass;
    availableMass += rowAvailableMass;
    shownMass += rowSelectedMass;
    return {
      receiverId: row.id,
      totalLinks: candidates.length,
      shownLinks: candidates.length - omittedMembers.length,
      omittedLinks: omittedMembers.length,
      missingLinks: members.length - candidates.length,
      retainedMass: rowFullMass > 0 ? rowSelectedMass / rowFullMass : null,
      availableRetainedMass: rowAvailableMass > 0 ? rowSelectedMass / rowAvailableMass : null,
      omittedMembers,
    };
  });
  const links = data.links.filter((_, index) => selected.has(index));
  return {
    links,
    summary: {
      mode: displayMode,
      targetMass,
      totalLinks: data.links.length,
      shownLinks: links.length,
      omittedLinks: data.links.length - links.length,
      missingLinks: data.missing,
      retainedMass: fullMass > 0 ? shownMass / fullMass : null,
      availableRetainedMass: availableMass > 0 ? shownMass / availableMass : null,
      rows: rowSummaries,
    },
  };
}

// Midpoint integration uses the reconstructed velocity, with seeds around each
// actual rotor and a sparse upstream context. Distance and step size are in D.
export function traceStreamlines(grid, turbines = []) {
  if (!validGrid(grid)) return [];
  const axes = ['x', 'y', 'z'].map((name) => grid.axes[name]);
  const min = axes.map((axis) => axis[0]);
  const max = axes.map((axis) => axis[axis.length - 1]);
  const span = max.map((v, i) => v - min[i]);
  const step = clamp(span[0] / 160, 0.08, 0.18);
  const paths = [];
  const members = Array.isArray(turbines) ? turbines.filter((turbine) => [turbine.x, turbine.y, turbine.z].every(Number.isFinite)) : [];
  const hubHeight = members.length ? members.reduce((sum, turbine) => sum + turbine.z, 0) / members.length : 0.875;
  const seeds = [];
  for (let yi = 0; yi < 7; yi += 1) {
    seeds.push({ point: [min[0] + span[0] * 0.01, min[1] + span[1] * (yi + 0.5) / 7, clamp(hubHeight, min[2], max[2])], kind: 'background', distance: 24 });
  }
  for (const turbine of members) {
    const diameter = Number.isFinite(turbine.rotor_diameter) ? turbine.rotor_diameter : 1;
    for (const y of [-0.32, 0, 0.32]) for (const z of [-0.32, 0, 0.32]) {
      seeds.push({ point: [turbine.x + 0.12 * diameter, turbine.y + y * diameter, turbine.z + z * diameter], kind: 'rotor', distance: 12 * diameter });
    }
  }
  for (const seed of seeds) {
    const p = [...seed.point];
    const path = [];
    for (let k = 0; k < Math.ceil(seed.distance / step); k += 1) {
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
    if (path.length > 5) {
      path.kind = seed.kind;
      paths.push(path);
    }
  }
  return paths;
}

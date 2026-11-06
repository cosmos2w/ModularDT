// Compact native E-address sets. The plotted coordinates are abstract: only
// inside/outside relationships encode data; distance and area do not.
const ROW_COLORS = [0x71e5d6, 0x72c8ef, 0xaaaaf5];
const PROFILE_RESOLUTION = 0.5;
const PROFILE_QUANTILE = 0.9;
const DISPLAY_CAP = 3;
const memberKey = (role, id) => `${role}:${String(id)}`;
const positive = (member) => ['M', 'E'].includes(member.role) && Number.isFinite(member.weight) && member.weight > 0;

function quantile(values, q) {
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * q;
  const lower = Math.floor(index);
  return sorted[lower] + (sorted[Math.ceil(index)] - sorted[lower]) * (index - lower);
}

// The input rows are actual native rows. Self is removed only in the profile
// copy used for choosing examples, never from the displayed memberships.
export function selectVennRows(rows, selectedAddress = null) {
  if (!rows.length) return { rows: [], selection: null };
  const columnKeys = new Set();
  const values = rows.map((row) => {
    const map = new Map();
    for (const member of row.members) {
      if (!positive(member) || (member.role === 'E' && String(member.id) === String(row.id))) continue;
      const key = memberKey(member.role, member.id);
      map.set(key, (map.get(key) || 0) + member.weight);
      columnKeys.add(key);
    }
    return map;
  });
  const columns = [...columnKeys];
  const profiles = values.map((map) => {
    const vector = columns.map((key) => map.get(key) || 0);
    const norm = Math.hypot(...vector);
    return norm ? vector.map((value) => value / norm) : vector;
  });
  const distances = profiles.map((left) => profiles.map((right) => {
    let squared = 0;
    for (let i = 0; i < left.length; i += 1) squared += (left[i] - right[i]) ** 2;
    return Math.sqrt(squared);
  }));
  const pinnedIndex = selectedAddress == null ? -1 : rows.findIndex((row) => String(row.id) === String(selectedAddress));
  let seed = pinnedIndex;
  if (seed < 0) {
    let least = Infinity;
    distances.forEach((row, index) => {
      const total = row.reduce((sum, value) => sum + value, 0);
      if (total < least) { seed = index; least = total; }
    });
  }
  const selected = [seed];
  const history = [];
  let assignments = [];
  let residuals = [];
  const limit = Math.min(DISPLAY_CAP, rows.length);
  const assign = () => {
    assignments = distances.map((row) => {
      let best = 0;
      for (let i = 1; i < selected.length; i += 1) if (row[selected[i]] < row[selected[best]]) best = i;
      return best;
    });
    residuals = distances.map((row, index) => row[selected[assignments[index]]]);
  };
  for (let count = 1; count <= limit; count += 1) {
    for (let iteration = 0; iteration < 30; iteration += 1) {
      assign();
      const refined = selected.map((old, group) => {
        if (pinnedIndex >= 0 && group === 0) return old;
        const members = assignments.flatMap((assignment, index) => assignment === group ? [index] : []);
        if (!members.length) return old;
        let best = members[0];
        let least = Infinity;
        for (const candidate of members) {
          const total = members.reduce((sum, other) => sum + distances[candidate][other], 0);
          if (total < least) { best = candidate; least = total; }
        }
        return best;
      });
      if (refined.every((value, index) => value === selected[index])) break;
      selected.splice(0, selected.length, ...refined);
    }
    assign();
    const residual90 = quantile(residuals, PROFILE_QUANTILE);
    history.push({ count, residual90 });
    if (residual90 <= PROFILE_RESOLUTION || count === limit) break;
    let next = -1;
    let farthest = -1;
    residuals.forEach((distance, index) => {
      if (!selected.includes(index) && distance > farthest) { next = index; farthest = distance; }
    });
    if (next < 0) break;
    selected.push(next);
  }
  const residual90 = quantile(residuals, PROFILE_QUANTILE);
  const coverage = residuals.filter((distance) => distance <= PROFILE_RESOLUTION).length / rows.length;
  return {
    rows: selected.map((index) => rows[index]),
    selection: {
      candidateRows: rows.length, coverage, residual90, cap: DISPLAY_CAP,
      coverageMet: residual90 <= PROFILE_RESOLUTION,
      budgetLimited: residual90 > PROFILE_RESOLUTION && selected.length === limit && limit < rows.length,
      history, pinned: pinnedIndex >= 0 ? rows[pinnedIndex].id : null,
      assignments: assignments.map((index) => rows[selected[index]].id),
      residualDistances: residuals,
    },
  };
}

// Three equal spheres in the picture plane realize every three-set membership
// pattern. This is display geometry, independent of physical member positions.
export function vennSpheres(count) {
  const centers = count === 1 ? [[0, 0, 0]] : count === 2
    ? [[-0.66, 0, 0], [0.66, 0, 0]]
    : [[-0.65, -0.38, 0], [0.65, -0.38, 0], [0, 0.75, 0]];
  return centers.slice(0, count).map((center) => ({ center, radius: 1.25 }));
}

export function sphereMask(position, spheres) {
  return spheres.reduce((mask, sphere, index) => {
    const distance = Math.hypot(...position.map((value, axis) => value - sphere.center[axis]));
    return distance < sphere.radius ? mask | (1 << index) : mask;
  }, 0);
}

function halton(index, base) {
  let fraction = 1;
  let result = 0;
  for (let value = index; value > 0; value = Math.floor(value / base)) {
    fraction /= base;
    result += fraction * (value % base);
  }
  return result;
}

// A deterministic candidate pool is classified against the actual surfaces.
// Margin prevents glyphs from straddling a set boundary. Farthest-point
// placement spreads original identities without altering their membership.
export function placeVennVertices(vertices, spheres) {
  const buckets = Array.from({ length: 1 << spheres.length }, () => []);
  for (let index = 1; index <= 20000; index += 1) {
    // All members share the spheres' central cross-section. This keeps the
    // displayed great-circle boundaries exact under perspective projection;
    // the translucent enclosing surfaces and schematic links remain 3D.
    const position = [(halton(index, 2) * 2 - 1) * 2.2, (halton(index, 3) * 2 - 1) * 2.2, 0];
    const margin = Math.min(...spheres.map((sphere) => Math.abs(Math.hypot(...position.map((value, axis) => value - sphere.center[axis])) - sphere.radius)));
    const mask = sphereMask(position, spheres);
    if (margin >= 0.075) {
      buckets[mask].push({ position, margin, nearest: Infinity, used: false });
    }
  }
  const ordered = [...vertices].sort((a, b) => (a.priority ?? 2) - (b.priority ?? 2));
  const placed = [];
  for (const vertex of ordered) {
    const candidates = buckets[vertex.mask];
    let best = null;
    let bestScore = -1;
    for (const candidate of candidates) {
      if (candidate.used) continue;
      const score = Number.isFinite(candidate.nearest) ? candidate.nearest + 0.02 * candidate.margin : candidate.margin;
      if (score > bestScore) { best = candidate; bestScore = score; }
    }
    if (!best) throw new Error(`No geometric candidate for native membership mask ${vertex.mask}`);
    best.used = true;
    const item = { ...vertex, position: best.position };
    placed.push(item);
    for (const bucket of buckets) for (const candidate of bucket) {
      const dx = candidate.position[0] - best.position[0];
      const dy = candidate.position[1] - best.position[1];
      const dz = candidate.position[2] - best.position[2];
      candidate.nearest = Math.min(candidate.nearest, dx * dx + dy * dy + 0.12 * dz * dz);
    }
  }
  return placed;
}

export function buildVennLayout(organization, scene, selectedAddress = null) {
  if (organization?.kind !== 'e_anchor_directory' || !Array.isArray(organization.anchors)) return null;
  const rows = Array.isArray(organization.directory_rows)
    ? organization.directory_rows.filter((row) => Array.isArray(row.members))
    : organization.anchors.filter((row) => Array.isArray(row.members));
  if (!rows.length) return null;
  const { rows: chosen, selection } = selectVennRows(rows, selectedAddress);
  const selectedIds = new Set(chosen.map((row) => String(row.id)));
  const vertices = new Map();
  for (const member of scene?.turbines || []) vertices.set(memberKey('M', member.id), { role: 'M', id: member.id, mask: 0, known: true, priority: 0 });
  for (const member of organization.anchors) vertices.set(memberKey('E', member.id), { role: 'E', id: member.id, mask: 0, known: true, priority: selectedIds.has(String(member.id)) ? 1 : 2 });
  let missingMembers = 0;
  const missing = new Set();
  const links = [];
  chosen.forEach((row, index) => {
    const receiverKey = memberKey('E', row.id);
    if (!vertices.has(receiverKey)) {
      vertices.set(receiverKey, { role: 'E', id: row.id, mask: 0, known: false, priority: 1 });
      missing.add(receiverKey);
    }
    for (const member of row.members) {
      if (!positive(member)) continue;
      const key = memberKey(member.role, member.id);
      if (!vertices.has(key)) {
        vertices.set(key, { role: member.role, id: member.id, mask: 0, known: false, priority: member.role === 'M' ? 0 : 2 });
        missing.add(key);
      }
      if (missing.has(key)) missingMembers += 1;
      vertices.get(key).mask |= 1 << index;
      links.push({ receiverId: row.id, donorRole: member.role, donorId: member.id, weight: member.weight, isSelf: member.role === 'E' && String(member.id) === String(row.id), rowIndex: index });
    }
  });
  const spheres = vennSpheres(chosen.length);
  const placed = placeVennVertices([...vertices.values()], spheres);
  if (placed.some((vertex) => sphereMask(vertex.position, spheres) !== vertex.mask)) throw new Error('Venn geometry does not match recorded native memberships');
  const intersectionCounts = Object.fromEntries(Array.from({ length: 1 << chosen.length }, (_, index) => [String(index), 0]));
  for (const vertex of placed) intersectionCounts[vertex.mask] += 1;
  const summary = {
    kind: 'venn', rowIds: chosen.map((row) => row.id), shownRows: chosen.length,
    totalRows: organization.selection?.total_rows ?? organization.anchors.length,
    positiveMembers: links.length, drawnMembers: links.length, missingMembers,
    selfMembers: links.filter((link) => link.isSelf).length, intersectionCounts,
    selection, exactMembership: true,
  };
  return { rows: chosen, vertices: placed, spheres, links, summary };
}

function label(THREE, text, color, scale) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 80;
  const context = canvas.getContext('2d');
  context.font = '600 39px Inter, system-ui, sans-serif';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.lineWidth = 7;
  context.strokeStyle = '#061920';
  context.strokeText(text, 128, 40);
  context.fillStyle = color;
  context.fillText(text, 128, 40);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const object = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false, depthTest: false }));
  object.scale.set(1.12 * scale, 0.35 * scale, 1);
  object.renderOrder = 8;
  return object;
}

function ring(THREE, center, radius, plane, color) {
  const points = [];
  for (let i = 0; i < 96; i += 1) {
    const angle = i * Math.PI * 2 / 96;
    const x = Math.cos(angle) * radius;
    const y = Math.sin(angle) * radius;
    points.push(new THREE.Vector3(center[0] + x, center[1] + (plane === 'xy' ? y : 0), center[2] + (plane === 'xy' ? 0 : y)));
  }
  return new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color, transparent: true, opacity: plane === 'xy' ? 0.55 : 0.23, depthWrite: false }));
}

// anchors returned to the caller are already in Three.js world coordinates.
export function buildOrganizationVenn({ THREE, organization, scene, selectedAddress = null }) {
  const layout = buildVennLayout(organization, scene, selectedAddress);
  if (!layout) return null;
  const domain = scene?.domain;
  const extent = domain?.min && domain?.max ? Math.max(domain.max[0] - domain.min[0], domain.max[1] - domain.min[1]) : 30;
  const scale = Math.max(2.8, Math.min(7, extent * 0.14));
  const origin = [0, scale * 1.65, 0];
  const world = (position) => position.map((value, axis) => origin[axis] + value * scale);
  const group = new THREE.Group();
  group.name = 'native-address-sets';
  layout.spheres.forEach((sphere, index) => {
    const center = world(sphere.center);
    const radius = sphere.radius * scale;
    const shell = new THREE.Mesh(new THREE.SphereGeometry(radius, 40, 28), new THREE.MeshBasicMaterial({
      color: ROW_COLORS[index], transparent: true, opacity: 0.075, side: THREE.DoubleSide, depthWrite: false,
    }));
    shell.position.set(...center);
    shell.renderOrder = 0;
    group.add(shell, ring(THREE, center, radius, 'xy', ROW_COLORS[index]), ring(THREE, center, radius, 'xz', ROW_COLORS[index]));
    const marker = label(THREE, `h(E${layout.rows[index].id})`, `#${new THREE.Color(ROW_COLORS[index]).getHexString()}`, scale);
    marker.position.set(center[0], center[1] + radius + 0.14 * scale, center[2]);
    group.add(marker);
  });
  const positions = new Map(layout.vertices.map((vertex) => [memberKey(vertex.role, vertex.id), world(vertex.position)]));
  const eVertices = layout.vertices.filter((vertex) => vertex.role === 'E');
  const eGeometry = new THREE.BufferGeometry();
  eGeometry.setAttribute('position', new THREE.Float32BufferAttribute(eVertices.flatMap((vertex) => positions.get(memberKey('E', vertex.id))), 3));
  const eColors = eVertices.flatMap((vertex) => vertex.mask ? [0.43, 0.81, 0.89] : [0.32, 0.46, 0.50]);
  eGeometry.setAttribute('color', new THREE.Float32BufferAttribute(eColors, 3));
  const ePoints = new THREE.Points(eGeometry, new THREE.PointsMaterial({ size: 4.7, sizeAttenuation: false, vertexColors: true, transparent: true, opacity: 0.9, depthWrite: false }));
  ePoints.userData.members = eVertices.map(({ role, id, mask }) => ({ role, id, mask }));
  ePoints.renderOrder = 4;
  group.add(ePoints);
  for (const vertex of layout.vertices.filter((item) => item.role === 'M')) {
    const position = positions.get(memberKey('M', vertex.id));
    const node = new THREE.Mesh(new THREE.BoxGeometry(scale * 0.12, scale * 0.12, scale * 0.12), new THREE.MeshBasicMaterial({ color: vertex.mask ? 0xb8fff0 : 0x789b9d }));
    node.position.set(...position);
    node.userData.member = { role: 'M', id: vertex.id, mask: vertex.mask };
    node.renderOrder = 5;
    group.add(node);
    const text = label(THREE, `M${vertex.id}`, vertex.mask ? '#c6fff0' : '#9cb9ba', scale * 0.70);
    text.position.set(position[0], position[1] + scale * 0.17, position[2]);
    group.add(text);
  }
  for (const row of layout.rows) {
    const vertex = layout.vertices.find((item) => item.role === 'E' && String(item.id) === String(row.id));
    const position = positions.get(memberKey('E', row.id));
    const node = new THREE.Mesh(new THREE.SphereGeometry(scale * 0.065, 14, 10), new THREE.MeshBasicMaterial({ color: 0xcafaff }));
    node.position.set(...position);
    node.userData.member = { role: 'E', id: row.id, mask: vertex.mask };
    node.renderOrder = 6;
    group.add(node);
    const text = label(THREE, `E${row.id}`, '#c8f5ff', scale * 0.67);
    text.position.set(position[0], position[1] - scale * 0.17, position[2]);
    group.add(text);
  }
  const linePoints = [];
  const lineColors = [];
  for (const link of layout.links.filter((item) => item.donorRole === 'M')) {
    const start = new THREE.Vector3(...positions.get(memberKey('M', link.donorId)));
    const end = new THREE.Vector3(...positions.get(memberKey('E', link.receiverId)));
    const middle = start.clone().add(end).multiplyScalar(0.5);
    middle.z += scale * (0.18 + 0.08 * link.rowIndex);
    const curve = new THREE.QuadraticBezierCurve3(start, middle, end);
    const color = new THREE.Color(ROW_COLORS[link.rowIndex]).multiplyScalar(0.92);
    const add = (a, b) => {
      linePoints.push(...a.toArray(), ...b.toArray());
      lineColors.push(color.r, color.g, color.b, color.r, color.g, color.b);
    };
    for (let segment = 0; segment < 16; segment += 1) add(curve.getPoint(segment / 16), curve.getPoint((segment + 1) / 16));
    const tip = curve.getPoint(0.90);
    const direction = curve.getTangent(0.90).normalize();
    let side = new THREE.Vector3(-direction.y, direction.x, 0);
    if (side.lengthSq() < 1e-8) side = new THREE.Vector3(1, 0, 0);
    side.normalize().multiplyScalar(scale * 0.035);
    const tail = tip.clone().addScaledVector(direction, -scale * 0.085);
    add(tip, tail.clone().add(side));
    add(tip, tail.clone().sub(side));
  }
  if (linePoints.length) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(linePoints, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3));
    const lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.74, depthWrite: false }));
    lines.renderOrder = 3;
    group.add(lines);
  }
  const bounds = [...layout.vertices.map((vertex) => world(vertex.position))];
  for (const sphere of layout.spheres) {
    const center = world(sphere.center);
    const radius = (sphere.radius + 0.3) * scale;
    for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) bounds.push([center[0] + x * radius, center[1] + y * radius, center[2] + z * radius]);
  }
  group.userData.abstractNativeSets = true;
  return { group, anchors: bounds, summary: layout.summary };
}

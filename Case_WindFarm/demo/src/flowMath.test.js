import test from 'node:test';
import assert from 'node:assert/strict';
import { FIELD_SPECS, directoryView, fieldColor, fieldValue, getFieldRange, referenceProfileUx, sampleVelocity, selectSpatialLinks, sliceSampleCoordinates, traceStreamlines, validGrid } from './flowMath.js';

function makeGrid(field) {
  const axes = { x: [-2, 0, 3], y: [0, 1, 5], z: [0, 2, 6] };
  const velocity = [];
  for (const x of axes.x) for (const y of axes.y) for (const z of axes.z) velocity.push(...field(x, y, z));
  return { shape: [3, 3, 3], bounds: { min: [-2, 0, 0], max: [3, 5, 6] }, axes, velocity };
}

test('physical interpolation reproduces an affine field on nonuniform cell centers', () => {
  const grid = makeGrid((x, y, z) => [x + 2 * y - 3 * z, 2 * x - y, 4 + z]);
  assert.ok(validGrid(grid));
  for (const point of [[1, 3, 1], [-2, 0, 0], [3, 5, 6], [0, 1, 2]]) {
    const expected = [point[0] + 2 * point[1] - 3 * point[2], 2 * point[0] - point[1], 4 + point[2]];
    sampleVelocity(grid, point).forEach((value, index) => assert.ok(Math.abs(value - expected[index]) < 1e-12));
  }
  assert.equal(sampleVelocity(grid, [-3, 1, 2]), null);
});

test('rendered sections and reference sampling share exactly the same physical plane', () => {
  const grid = makeGrid(() => [1, 0, 0]);
  const slice = sliceSampleCoordinates(grid, 'y', 0.5);
  assert.equal(slice.fixedCoordinate, 2.5);
  assert.equal(slice.coordinates.length, slice.shape[0] * slice.shape[1]);
  assert.ok(slice.coordinates.every(point => point[1] === 2.5 && sampleVelocity(grid, point) !== null));
});

test('streamlines of a uniform velocity field preserve transverse coordinates', () => {
  const turbine = { x: 0, y: 1, z: 2 };
  const paths = traceStreamlines(makeGrid(() => [1, 0, 0]), [turbine]);
  assert.ok(paths.length > 0);
  const rotorPaths = paths.filter(path => path.kind === 'rotor');
  assert.equal(rotorPaths.length, 9);
  assert.ok(rotorPaths.every(path => Math.abs(path[0].position[0] - 0.12) < 1e-12));
  for (const path of paths) {
    assert.ok(path.every(({ position }) => position[1] === path[0].position[1] && position[2] === path[0].position[2]));
    assert.ok(path.every(({ position }, index) => index === 0 || position[0] > path[index - 1].position[0]));
  }
});

test('deficit uses the checkpoint profile at physical height for both prediction and reference', () => {
  const grid = makeGrid(() => [1, 0, 0]);
  grid.reference_profile = { z_D: [0.5, 1.5, 3], velocity_mps: [[5, 0, 0], [9, 0, 0], [12, 0, 0]] };
  assert.equal(referenceProfileUx(grid, 0), 5);
  assert.equal(referenceProfileUx(grid, 1), 7);
  assert.equal(referenceProfileUx(grid, 4), 12);
  const point = [0, 1, 1];
  const prediction = fieldValue([6, 0, 0], 'deficit', point, grid);
  const reference = fieldValue([5.5, 0, 0], 'deficit', point, grid);
  assert.equal(prediction, 1);
  assert.equal(reference, 1.5);
  assert.equal(Math.abs(prediction - reference), 0.5);
  assert.ok(Number.isNaN(fieldValue([6, 0, 0], 'deficit', point, {})));
});

test('wake display limits use local samples, report clipping and keep signed fields centered on zero', () => {
  const axes = { x: [-1, 0, 1, 4, 9], y: [-3, -1, 0, 1, 3], z: [0, 0.5, 1, 1.5, 4] };
  const velocity = [];
  for (const x of axes.x) for (const y of axes.y) for (const z of axes.z) {
    const local = x >= -0.5 && x <= 8 && Math.abs(y) <= 1.5 && Math.abs(z - 1) <= 0.65;
    velocity.push(local ? 5 + x * 0.1 + y * 0.05 : 100, local ? x - 2 + y : 25, 0);
  }
  const grid = { shape: [5, 5, 5], bounds: { min: [-1, -3, 0], max: [9, 3, 4] }, axes, velocity };
  const options = { contrast: 'wake', turbines: [{ x: 0, y: 0, z: 1 }] };
  const range = getFieldRange(grid, 'ux', options);
  assert.equal(range.scope, 'near-turbine rotor band');
  assert.equal(range.sampleCount, 27);
  assert.deepEqual(range.percentiles, [2, 98]);
  assert.ok(range.max < 10 && range.clipped);
  assert.equal(getFieldRange(grid, 'ux', { contrast: 'global' }).max, 100);
  const signed = getFieldRange(grid, 'uy', options);
  assert.equal(signed.min, -signed.max);
  assert.deepEqual(fieldColor(0, signed, 'uy'), FIELD_SPECS.uy.colors[2].match(/[a-f0-9]{2}/gi).map(channel => parseInt(channel, 16) / 255));
  assert.notDeepEqual(fieldColor(range.min, range, 'ux'), fieldColor(range.min, range, 'speed'));
  assert.equal(getFieldRange(grid, 'ux', { contrast: 'wake', turbines: [] }).scope, 'full grid');
});

test('hyperedge display preserves original identities, self membership and tiny positive weights', () => {
  const scene = { turbines: [{ id: 7, x: 2, y: 3, z: 0.875 }] };
  const organization = { kind: 'e_anchor_directory', anchors: [
    { id: 2, position: [0, 0, 1], members: [
      { role: 'M', id: 7, weight: 1e-12 },
      { role: 'E', id: 2, weight: 0.3, self: 0.3 },
      { role: 'E', id: 9, weight: 0.7 - 1e-12 },
      { role: 'M', id: 99, weight: 0 },
    ] },
    { id: 9, position: [4, 1, 2], members: [{ role: 'M', id: 7, weight: 1 }] },
  ] };
  const before = JSON.stringify(organization);
  const selected = directoryView(organization, scene, 2);
  assert.equal(selected.rows.length, 1);
  assert.equal(selected.links.length, 3);
  assert.equal(selected.missing, 0);
  assert.deepEqual(selected.links[0].donorPosition, [2, 3, 0.875]);
  assert.equal(selected.links[0].weight, 1e-12);
  assert.ok(selected.links.find(link => link.donorId === 2).isSelf);
  assert.equal(directoryView(organization, scene).links.length, 4);
  assert.equal(JSON.stringify(organization), before);
});

test('focused spatial view retains 95% of each row and preserves positive M and own-E members', () => {
  const scene = { turbines: [{ id: 7, x: 2, y: 3, z: 0.875 }] };
  const organization = { kind: 'e_anchor_directory', anchors: [
    { id: 2, position: [0, 0, 1], members: [
      { role: 'M', id: 7, weight: 1e-12 },
      { role: 'E', id: 2, weight: 2e-12 },
      { role: 'E', id: 9, weight: 0.7 },
      { role: 'E', id: 10, weight: 0.26 },
      { role: 'E', id: 11, weight: 0.025 },
      { role: 'E', id: 12, weight: 0.015 },
    ] },
    { id: 9, position: [1, 0, 1], members: [
      { role: 'M', id: 7, weight: 0.5 },
      { role: 'E', id: 9, weight: 0.1 },
      { role: 'E', id: 2, weight: 0.2 },
      { role: 'E', id: 10, weight: 0.16 },
      { role: 'E', id: 11, weight: 0.04 },
    ] },
    ...[10, 11, 12].map((id) => ({ id, position: [id, 0, 1] })),
  ] };
  const data = directoryView(organization, scene);
  const before = JSON.stringify({ organization, data });
  const focused = selectSpatialLinks(data, { mode: 'focused' });
  assert.deepEqual(focused.links.map(({ receiverId, donorId }) => [receiverId, donorId]), [
    [2, 7], [2, 2], [2, 9], [2, 10], [9, 7], [9, 9], [9, 2], [9, 10],
  ]);
  assert.equal(focused.summary.totalLinks, 11);
  assert.equal(focused.summary.shownLinks, 8);
  assert.equal(focused.summary.omittedLinks, 3);
  assert.equal(focused.summary.missingLinks, 0);
  assert.ok(focused.summary.rows.every((row) => row.retainedMass >= 0.95));
  assert.deepEqual(focused.summary.rows[0].omittedMembers, [
    { role: 'E', id: 11, weight: 0.025 }, { role: 'E', id: 12, weight: 0.015 },
  ]);
  assert.deepEqual(focused.summary.rows[1].omittedMembers, [{ role: 'E', id: 11, weight: 0.04 }]);
  assert.ok(Math.abs(focused.summary.retainedMass - (1.92 + 3e-12) / (2 + 3e-12)) < 1e-12);

  const all = selectSpatialLinks(data, { mode: 'all' });
  assert.notEqual(all.links, data.links);
  assert.deepEqual(all.links, data.links);
  assert.ok(all.links.every((link, index) => link === data.links[index]));
  assert.equal(all.summary.omittedLinks, 0);
  assert.equal(all.summary.retainedMass, 1);
  assert.equal(JSON.stringify({ organization, data }), before);
});

test('spatial mass coverage accounts for members with unresolved coordinates', () => {
  const organization = { kind: 'e_anchor_directory', anchors: [
    { id: 2, position: [0, 0, 1], members: [
      { role: 'M', id: 7, weight: 0.5 },
      { role: 'E', id: 2, weight: 0.25 },
      { role: 'E', id: 404, weight: 0.25 },
    ] },
  ] };
  const data = directoryView(organization, { turbines: [{ id: 7, x: 0, y: 0, z: 1 }] });
  const { links, summary } = selectSpatialLinks(data, { mode: 'focused' });
  assert.equal(links.length, 2);
  assert.equal(summary.missingLinks, 1);
  assert.equal(summary.rows[0].missingLinks, 1);
  assert.equal(summary.omittedLinks, 0);
  assert.equal(summary.retainedMass, 0.75);
  assert.equal(summary.availableRetainedMass, 1);
});

import test from 'node:test';
import assert from 'node:assert/strict';
import { directoryView, sampleVelocity, sliceSampleCoordinates, traceStreamlines, validGrid } from './flowMath.js';

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
  const paths = traceStreamlines(makeGrid(() => [1, 0, 0]));
  assert.ok(paths.length > 0);
  for (const path of paths) {
    assert.ok(path.every(({ position }) => position[1] === path[0].position[1] && position[2] === path[0].position[2]));
    assert.ok(path.every(({ position }, index) => index === 0 || position[0] > path[index - 1].position[0]));
  }
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

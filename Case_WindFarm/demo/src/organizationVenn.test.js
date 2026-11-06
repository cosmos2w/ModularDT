import test from 'node:test';
import assert from 'node:assert/strict';
import { buildVennLayout, placeVennVertices, selectVennRows, sphereMask, vennSpheres } from './organizationVenn.js';

test('each original vertex is strictly inside exactly the recorded native sets', () => {
  for (let count = 1; count <= 3; count += 1) {
    const spheres = vennSpheres(count);
    const vertices = Array.from({ length: (1 << count) * 8 }, (_, index) => ({ id: index, mask: index % (1 << count) }));
    const placed = placeVennVertices(vertices, spheres);
    assert.equal(placed.length, vertices.length);
    for (const vertex of placed) {
      assert.equal(sphereMask(vertex.position, spheres), vertex.mask);
      assert.equal(sphereMask([vertex.position[0], vertex.position[1], 0], spheres), vertex.mask);
      for (const sphere of spheres) {
        const distance = Math.hypot(...vertex.position.map((value, axis) => value - sphere.center[axis]));
        assert.ok(Math.abs(distance - sphere.radius) >= 0.075 - 1e-12);
      }
    }
  }
});

test('selection uses actual non-self profiles and collapses equivalent rows', () => {
  const rows = [2, 5, 9].map((id) => ({ id, members: [
    { role: 'M', id: 7, weight: 1 }, { role: 'E', id, weight: 0.00001 },
  ] }));
  const before = JSON.stringify(rows);
  const auto = selectVennRows(rows);
  assert.deepEqual(auto.rows.map((row) => row.id), [2]);
  assert.equal(auto.selection.coverage, 1);
  assert.equal(auto.selection.residual90, 0);
  const pinned = selectVennRows(rows, 9);
  assert.deepEqual(pinned.rows.map((row) => row.id), [9]);
  assert.equal(pinned.selection.pinned, 9);
  assert.equal(JSON.stringify(rows), before);
});

test('directory rows preserve zero, tiny positive, own-E, original roles and every intersection', () => {
  const masks = [1, 2, 3, 4, 5, 6, 7];
  const rows = [0, 1, 2].map((index) => ({ id: 10 + index, members: [
    ...masks.filter((mask) => mask & (1 << index)).map((mask) => ({ role: 'M', id: mask, weight: mask === (1 << index) ? 1 : 1e-12 })),
    { role: 'E', id: 10 + index, weight: 1e-14 },
    { role: 'E', id: 99, weight: index === 1 ? 1e-30 : 0 },
  ] }));
  const organization = { kind: 'e_anchor_directory', directory_rows: rows,
    anchors: [10, 11, 12, 99, 100].map((id) => ({ id, position: [0, 0, 0] })), selection: { total_rows: 5 } };
  const scene = { turbines: [0, ...masks].map((id) => ({ id, x: 0, y: 0, z: 0 })) };
  const layout = buildVennLayout(organization, scene);
  assert.equal(layout.rows.length, 3);
  assert.equal(layout.summary.selection.candidateRows, 3);
  assert.equal(layout.summary.selfMembers, 3);
  assert.equal(layout.summary.missingMembers, 0);
  assert.equal(layout.summary.positiveMembers, rows.flatMap((row) => row.members.filter((member) => member.weight > 0)).length);
  for (const vertex of layout.vertices) {
    let expected = 0;
    layout.rows.forEach((row, index) => {
      if (row.members.some((member) => member.role === vertex.role && member.id === vertex.id && member.weight > 0)) expected |= 1 << index;
    });
    assert.equal(vertex.mask, expected);
    assert.equal(sphereMask(vertex.position, layout.spheres), expected);
  }
  assert.equal(layout.vertices.find((vertex) => vertex.role === 'M' && vertex.id === 0).mask, 0);
  assert.notEqual(layout.vertices.find((vertex) => vertex.role === 'E' && vertex.id === 99).mask, 0);
  assert.equal(layout.vertices.find((vertex) => vertex.role === 'E' && vertex.id === 100).mask, 0);
  assert.equal(Object.values(layout.summary.intersectionCounts).reduce((sum, count) => sum + count, 0), layout.vertices.length);
});

import assert from 'node:assert/strict';
import { clampTurbinePosition, domainLengthFromDrag, isPointerClick, withinDomain } from './designMath.js';

const domain = { min: [-5, -4, 0], max: [5, 4, 3] };
assert.deepEqual(clampTurbinePosition(domain, 9, -7), [4.5, -3.5]);
assert.deepEqual(clampTurbinePosition(domain, 1.25, -2), [1.25, -2]);
assert.equal(withinDomain(domain, 5, 4), true);
assert.equal(withinDomain(domain, 5.01, 4), false);

// Both opposing handles resize symmetrically around the unchanged center.
assert.equal(domainLengthFromDrag(10, 5, 6.5, 1, [4, 20]), 13);
assert.equal(domainLengthFromDrag(10, -5, -6.5, -1, [4, 20]), 13);
assert.equal(domainLengthFromDrag(10, 5, -5, 1, [4, 20]), 4);
assert.equal(domainLengthFromDrag(10, -5, -50, -1, [4, 20]), 20);
assert.equal(isPointerClick({ x: 5, y: 5 }, { x: 7, y: 7 }), true);
assert.equal(isPointerClick({ x: 5, y: 5 }, { x: 9, y: 5 }), false);

console.log('designMath tests passed');

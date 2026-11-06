// Domain resizing keeps the center fixed, so an edge moves by half the length change.
export function domainLengthFromDrag(initialLength, startCoordinate, currentCoordinate, direction, limits) {
  const candidate = initialLength + 2 * direction * (currentCoordinate - startCoordinate);
  return Math.min(limits[1], Math.max(limits[0], candidate));
}

// Mirror the model editor's half-diameter clearance at the horizontal boundary.
export function clampTurbinePosition(domain, x, y, margin = 0.5) {
  return [x, y].map((value, axis) => {
    const low = domain.min[axis] + margin;
    const high = domain.max[axis] - margin;
    return Math.min(high, Math.max(low, value));
  });
}

export function withinDomain(domain, x, y) {
  return x >= domain.min[0] && x <= domain.max[0] && y >= domain.min[1] && y <= domain.max[1];
}

export function isPointerClick(start, end, threshold = 4) {
  return Math.hypot(end.x - start.x, end.y - start.y) < threshold;
}

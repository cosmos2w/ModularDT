"""Finite area contact quadrature with exact discrete normalization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def rasterize_contacts(
    contacts: Sequence[dict], ny: int, nx: int, order: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Return cell-average chi[M,ny,nx] and pre-normalization integrals.

    Only cells intersecting each disk's bounding box are integrated. The
    normalized discrete integral is exactly one to floating-point precision.
    """
    if min(nx, ny) < 2 or order not in (2, 4, 8):
        raise ValueError("grid must have at least two cells and quadrature order must be 2, 4, or 8")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes = (nodes + 1) / 2
    weights = weights / 2
    chi = np.zeros((len(contacts), ny, nx), dtype=np.float64)
    raw_integrals = np.zeros(len(contacts), dtype=np.float64)
    for k, contact in enumerate(contacts):
        cx, cy, radius = (float(contact[key]) for key in ("x", "y", "r"))
        if not (0 < radius and radius <= cx <= 1 - radius and radius <= cy <= 1 - radius):
            raise ValueError("contact disk must lie within the unit sheet")
        i0 = max(0, int(np.floor((cx - radius) * nx)))
        i1 = min(nx, int(np.ceil((cx + radius) * nx)))
        j0 = max(0, int(np.floor((cy - radius) * ny)))
        j1 = min(ny, int(np.ceil((cy + radius) * ny)))
        ix = np.arange(i0, i1, dtype=np.float64)
        iy = np.arange(j0, j1, dtype=np.float64)
        if not len(ix) or not len(iy):
            raise ValueError("contact has empty support")
        cell = np.zeros((len(iy), len(ix)), dtype=np.float64)
        for ay, wy in zip(nodes, weights):
            yy = (iy[:, None] + ay) / ny
            for ax, wx in zip(nodes, weights):
                xx = (ix[None, :] + ax) / nx
                cell += wx * wy * (3 / (np.pi * radius**2)) * np.maximum(
                    1 - ((xx - cx) ** 2 + (yy - cy) ** 2) / radius**2, 0
                ) ** 2
        chi[k, j0:j1, i0:i1] = cell
        raw_integrals[k] = float(cell.sum() / (nx * ny))
        if not np.isfinite(raw_integrals[k]) or raw_integrals[k] <= 0:
            raise ArithmeticError("contact quadrature yielded zero/nonfinite integral")
        chi[k] /= raw_integrals[k]
    return chi, raw_integrals

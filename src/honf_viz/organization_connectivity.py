"""Select actual native organization rows and disclose displayed connections.

Selections and clusters are display summaries; they do not alter native
memberships or assert a learned number of groups.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def select_native_rows(
    matrices: list[np.ndarray], row_labels: list[str], *,
    own_columns: np.ndarray | None = None, resolution: float = .5,
    coverage: float = .9, max_edges: int = 4, edges: int | None = None,
) -> tuple[list[int], dict]:
    """Select exact native rows from aligned matrices using equal-model profiles.

    Automatic selection stops at the first refined medoid count whose stated
    nearest-profile distance quantile is at most ``resolution``. Explicit
    ``edges`` requests that many actual rows, independent of that threshold.
    """
    if not matrices:
        raise ValueError("At least one native membership matrix is required.")
    arrays = [np.asarray(matrix, dtype=np.float64) for matrix in matrices]
    shape = arrays[0].shape
    if len(shape) != 2 or min(shape) < 1 or any(a.shape != shape for a in arrays):
        raise ValueError("Native matrices must be nonempty and have aligned rows and columns.")
    if any(not np.isfinite(a).all() or np.any(a < 0) for a in arrays):
        raise ValueError("Native membership weights must be finite and nonnegative.")
    count = shape[0]
    if len(row_labels) != count or len(set(row_labels)) != count or not all(isinstance(x, str) for x in row_labels):
        raise ValueError("Each native row needs a unique string label.")
    if not math.isfinite(resolution) or resolution < 0 or not math.isfinite(coverage) or not 0 < coverage <= 1:
        raise ValueError("Resolution must be finite and nonnegative and coverage in (0, 1].")
    if isinstance(max_edges, bool) or not isinstance(max_edges, int) or max_edges < 1:
        raise ValueError("max_edges must be a positive integer.")
    if edges is not None and (isinstance(edges, bool) or not isinstance(edges, int) or not 1 <= edges <= count):
        raise ValueError("Explicit edge count must be between one and the native row count.")
    own = None
    if own_columns is not None:
        own = np.asarray(own_columns)
        if own.shape != (count,) or not np.issubdtype(own.dtype, np.integer) or np.any(own < 0) or np.any(own >= shape[1]):
            raise ValueError("Own-member columns must give one valid integer column per row.")
    features = []
    for array in arrays:
        profile = array.copy()
        if own is not None:
            profile[np.arange(count), own] = 0
        norms = np.linalg.norm(profile, axis=1, keepdims=True)
        features.append(np.divide(profile, norms, out=np.zeros_like(profile), where=norms > 0))
    features = np.concatenate(features, axis=1) / math.sqrt(len(arrays))
    squared = np.einsum("ij,ij->i", features, features)
    distances = np.sqrt(np.maximum(squared[:, None] + squared[None, :] - 2 * features @ features.T, 0))
    np.fill_diagonal(distances, 0)
    limit = min(count, edges if edges is not None else max_edges)
    selected = [int(np.argmin(distances.sum(axis=1)))]
    history = []
    for k in range(1, limit + 1):
        # Medoids must remain distinct even when duplicate profiles leave a
        # cluster empty; the original row order resolves every exact tie.
        for _ in range(30):
            assignments = np.argmin(distances[:, selected], axis=1)
            refined = []
            for group, old in enumerate(selected):
                members = np.flatnonzero(assignments == group)
                refined.append(int(members[np.argmin(distances[np.ix_(members, members)].sum(axis=1))])
                               if len(members) else old)
            if refined == selected:
                break
            selected = refined
        assignments = np.argmin(distances[:, selected], axis=1)
        residuals = distances[np.arange(count), np.asarray(selected)[assignments]]
        quantile = float(np.quantile(residuals, coverage))
        history.append({"edge_count": k, "quantile_distance": quantile})
        if edges is None and quantile <= resolution:
            break
        if k == limit:
            break
        remaining = residuals.copy()
        remaining[selected] = -1
        selected.append(int(np.argmax(remaining)))
    met = quantile <= resolution
    return selected, {
        "method": "equal-model row-L2-normalized native profiles; deterministic medoid and farthest-first refinement",
        "selection_only_own_columns_removed": own is not None,
        "native_row_count": count, "model_count": len(arrays),
        "selected_rows": selected.copy(), "selected_labels": [row_labels[i] for i in selected],
        "assignments": assignments.tolist(), "residual_distances": residuals.tolist(),
        "resolution": float(resolution), "coverage_quantile": float(coverage),
        "covered_rows": int(np.count_nonzero(residuals <= resolution)),
        "covered_fraction": float(np.mean(residuals <= resolution)),
        "quantile_distance": quantile, "coverage_met": bool(met),
        "budget_limited": bool(edges is None and not met and limit < count),
        "explicit_edges": edges, "max_edges": max_edges, "selection_history": history,
        "interpretation": "Actual native rows selected for display; assignments are not learned groups.",
    }


def membership_overlap(module_by_edge: np.ndarray) -> np.ndarray:
    """Return a symmetric, derived shared-membership association in [0, 1]."""
    weights = np.asarray(module_by_edge, dtype=np.float64)
    if weights.ndim != 2 or min(weights.shape) < 1 or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Memberships must be a finite, nonnegative module-by-edge matrix.")
    totals = weights.sum(axis=1, keepdims=True)
    probabilities = np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0)
    # The theoretical overlap is bounded by one; clip only floating-point
    # summation noise so the connection validator accepts valid distributions.
    return np.clip(np.minimum(probabilities[:, None, :], probabilities[None, :, :]).sum(axis=2), 0, 1)


def select_connections(
    weights: np.ndarray, module_labels: Sequence[str], *, kind: str = "actual_mm",
    max_incoming: int = 1, max_connections: int = 8,
) -> dict:
    """Select bounded positive connections, preserving direction and values.

    Actual MM uses ``weights[receiver, donor]``. Membership association is
    symmetric and undirected; it is explicitly not an applied gate.
    """
    values = np.asarray(weights, dtype=np.float64)
    n = len(module_labels)
    if n < 1 or values.shape != (n, n) or len(set(module_labels)) != n or not all(isinstance(x, str) for x in module_labels):
        raise ValueError("Weights must be square and labels unique strings.")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Connection strengths must be finite and in [0, 1].")
    if kind not in ("actual_mm", "association"):
        raise ValueError("kind must be actual_mm or association.")
    if kind == "association" and not np.allclose(values, values.T, rtol=0, atol=1e-12):
        raise ValueError("Derived associations must be symmetric.")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (max_incoming, max_connections)):
        raise ValueError("Display limits must be nonnegative integers.")
    universe = [(receiver, donor, float(values[receiver, donor])) for receiver in range(n)
                for donor in range(n) if receiver != donor and values[receiver, donor] > 0
                and (kind == "actual_mm" or donor < receiver)]
    candidates = set()
    for receiver in range(n):
        incoming = [(receiver, donor, float(values[receiver, donor])) for donor in range(n)
                    if donor != receiver and values[receiver, donor] > 0]
        incoming.sort(key=lambda item: (-item[2], item[1]))
        for r, d, _ in incoming[:max_incoming]:
            candidates.add((r, d) if kind == "actual_mm" else (max(r, d), min(r, d)))
    eligible = [item for item in universe if item[:2] in candidates]
    eligible.sort(key=lambda item: (-item[2], item[0], item[1]))
    chosen = eligible[:max_connections]
    def item(r: int, d: int, v: float) -> dict:
        index = 0 if v < 1 / 3 else 1 if v < 2 / 3 else 2
        return {"receiver_index": r, "donor_index": d, "receiver": module_labels[r],
                "donor": module_labels[d], "weight": v,
                "bin": ("weak", "medium", "strong")[index], "width": (1.2, 2.6, 4.4)[index]}
    selected_mass = sum(v for _, _, v in chosen)
    total_mass = sum(v for _, _, v in universe)
    return {"kind": kind, "directed": kind == "actual_mm",
            "meaning": "actual receiver <- donor MM gate" if kind == "actual_mm" else "derived symmetric shared-membership association; not an applied gate",
            "bin_boundaries": [0, 1 / 3, 2 / 3, 1], "max_incoming": max_incoming,
            "max_connections": max_connections, "connections": [item(*v) for v in chosen],
            "positive_count": len(universe), "eligible_after_incoming_limit": len(eligible),
            "omitted_count": len(universe) - len(chosen), "total_weight": total_mass,
            "selected_weight": selected_mass, "omitted_weight": total_mass - selected_mass,
            "selected_mass_fraction": selected_mass / total_mass if total_mass else None}

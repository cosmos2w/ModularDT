"""Mask-aware comparisons of aligned Thermal physical response observations."""
from __future__ import annotations

import numpy as np

from honf_runtime.state_response import compare_prevalidated_response, field_metrics, summarize_response_actions


def compare_native_response(baseline, edited):
    """Compare observations whose native IDs, coordinates and units were aligned by the case."""
    return compare_prevalidated_response(baseline, edited)


def fieldwise_flow_diagnostics(baseline, repeated, heat_changed, mask):
    """No-op and heat-null differences on each channel's own native units."""
    base, repeat, heat = (np.asarray(x, dtype=np.float64) for x in (baseline, repeated, heat_changed))
    mask = np.asarray(mask, dtype=bool)
    if base.shape != repeat.shape or base.shape != heat.shape or base.ndim != 2 or base.shape[1] < 4 or mask.shape != base.shape[:1] or not mask.any():
        raise ValueError('Flow diagnostics require aligned nonempty native fluid receivers.')
    output = {}
    for i, (name, units) in enumerate((('u','dataset velocity units'), ('v','dataset velocity units'),
                                        ('p','dataset pressure units'), ('omega','dataset vorticity units'))):
        values = {}
        for label, other in (('zero_edit', repeat), ('heat_only', heat)):
            diff = (other[mask, i] - base[mask, i])
            if not np.isfinite(diff).all():
                raise ValueError('Nonfinite prediction on a valid flow diagnostic receiver.')
            values[label] = {'max_abs': float(np.max(np.abs(diff))),
                             'rmse': float(np.sqrt(np.mean(diff * diff)))}
        output[name] = {'units': units, 'valid_count': int(mask.sum()), **values}
    return output


def summarize_actions(rows, floors, *, family_ids=None):
    """Retain the Thermal J/P policy over the shared family reductions."""
    return summarize_response_actions(rows, {name: floors[name] for name in ('J', 'P')},
                                      family_ids=family_ids)


def intermodule_fluid_mask(fluid_xy, fluid_valid, module_xy):
    """Fluid receivers inside the convex hull of at least three noncollinear centres."""
    from scipy.spatial import Delaunay, QhullError
    xy = np.asarray(fluid_xy, dtype=np.float64)
    valid = np.asarray(fluid_valid, dtype=bool)
    centers = np.asarray(module_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or valid.shape != (len(xy),) or centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError('Native fluid and module coordinates must be aligned 2D points.')
    if len(centers) < 3 or np.linalg.matrix_rank(centers - centers.mean(axis=0)) < 2:
        return np.zeros(len(xy), dtype=bool), 'fewer than three noncollinear module centers'
    try:
        mask = valid & (Delaunay(centers).find_simplex(xy) >= 0)
    except QhullError:
        return np.zeros(len(xy), dtype=bool), 'degenerate convex hull'
    return mask, None if mask.any() else 'no valid fluid receivers within module-center hull'


def evaluate_thermal_candidate_pool(predicted, physical, *, objective, constraint_caps,
                                    baseline='baseline', objective_scale, sensitivity):
    """Evaluate a requested finite pool of Thermal J/P response deltas.

    J is the native material smooth maximum and P the native inlet-minus-outlet
    pressure; the caller supplies their physical-unit scales, sensitivities and
    requested caps. Each candidate mapping must contain both J and P.
    """
    from collections.abc import Mapping
    from honf_runtime.state_response import evaluate_candidate_pool

    names = {'J', 'P'}
    if (objective not in names or not isinstance(constraint_caps, Mapping)
            or not set(constraint_caps).issubset(names)
            or not isinstance(sensitivity, Mapping) or set(sensitivity) != names):
        raise ValueError('Thermal pool requests require declared J/P functionals and sensitivities.')
    for dataset in (predicted, physical):
        if dataset is not None and (not isinstance(dataset, Mapping) or any(
                not isinstance(row, Mapping) or set(row) != names for row in dataset.values())):
            raise ValueError('Each Thermal candidate needs exactly J and P deltas.')
    return evaluate_candidate_pool(predicted, physical, objective=objective,
        constraint_caps=constraint_caps, baseline=baseline,
        objective_scale=objective_scale, sensitivity=sensitivity)

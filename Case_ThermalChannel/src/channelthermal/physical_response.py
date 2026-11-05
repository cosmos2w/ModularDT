"""Mask-aware comparisons of aligned Thermal physical response observations."""
from __future__ import annotations

import itertools
import math

import numpy as np


def compare_native_response(baseline, edited):
    """Compare each endpoint on its own support and changes on common support.

    Each endpoint maps the same field names to ``prediction``, ``reference``
    and boolean ``mask`` arrays. Callers must align original members, native
    receivers, coordinates and units before calling. Optional local observations
    can be supplied as additional named fields with the same contract.
    """
    if not baseline or set(baseline) != set(edited):
        raise ValueError('Both endpoints must contain the same nonempty field set.')
    result = {}
    for name in baseline:
        base, edit = baseline[name], edited[name]
        bp, br, bm = (np.asarray(base[k]) for k in ('prediction', 'reference', 'mask'))
        ep, er, em = (np.asarray(edit[k]) for k in ('prediction', 'reference', 'mask'))
        if not (bp.shape == br.shape == bm.shape == ep.shape == er.shape == em.shape):
            raise ValueError(f'Native observations for {name!r} must be aligned.')
        common = bm.astype(bool) & em.astype(bool)
        result[name] = {
            'baseline': field_metrics(bp, br, bm),
            'edited': field_metrics(ep, er, em),
            'response': field_metrics(ep - bp, er - br, common),
        }
    return result


def field_metrics(prediction, target, mask):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError('Prediction, target and physical validity mask must align.')
    if not mask.any() or not np.isfinite(prediction[mask]).all() or not np.isfinite(target[mask]).all():
        raise ValueError('A physical field comparison needs finite, valid receivers.')
    diff = prediction[mask] - target[mask]
    rmse = float(np.sqrt(np.mean(diff * diff)))
    reference_rms = float(np.sqrt(np.mean(target[mask] ** 2)))
    return {'count': int(mask.sum()), 'rmse': rmse,
            'reference_rms': reference_rms,
            'rmse_over_reference_rms': rmse / reference_rms if reference_rms > 0 else None,
            'mae': float(np.mean(np.abs(diff))), 'prediction_mean': float(prediction[mask].mean()),
            'physical_mean': float(target[mask].mean())}


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
    """Equal-family all-action and resolved metrics; retain no-resolution coverage."""
    result = {}
    observed = {row['case_id'] for row in rows}
    families = sorted(observed if family_ids is None else set(family_ids))
    if not observed.issubset(families):
        raise ValueError('Observed actions contain a family outside the declared cohort.')
    for key in ('J', 'P'):
        floor = float(floors[key])
        if not math.isfinite(floor) or floor <= 0:
            raise ValueError('TRAIN-derived resolution floors must be positive and finite.')
        if any(not math.isfinite(float(row['delta'][key][source]))
               for row in rows for source in ('model', 'physical')):
            raise ValueError('Response values must be finite.')
        family_rows, resolved_rows, near_null_rows = [], [], []
        signs = signs_total = order_wrong = order_total = 0
        all_signs = all_signs_total = all_order_wrong = all_order_total = 0
        for family in families:
            cases = [r for r in rows if r['case_id'] == family]
            if not cases:
                continue
            errors = [abs(r['delta'][key]['model'] - r['delta'][key]['physical']) for r in cases]
            family_rows.append(float(np.mean(errors)))
            resolved = [r for r in cases if abs(r['delta'][key]['physical']) > floor]
            near_null = [r for r in cases if abs(r['delta'][key]['physical']) <= floor]
            if near_null:
                near_null_rows.append(float(np.mean([abs(r['delta'][key]['model']-r['delta'][key]['physical']) for r in near_null])))
            if resolved:
                resolved_rows.append({'case_id': family,
                    'mae': float(np.mean([abs(r['delta'][key]['model']-r['delta'][key]['physical']) for r in resolved])),
                    'mean_effect': float(np.mean([abs(r['delta'][key]['physical']) for r in resolved]))})
            for row in resolved:
                signs += int(np.sign(row['delta'][key]['model']) == np.sign(row['delta'][key]['physical']))
                signs_total += 1
            for row in cases:
                all_signs += int(np.sign(row['delta'][key]['model']) == np.sign(row['delta'][key]['physical']))
                all_signs_total += 1
            for a, b in itertools.combinations(cases, 2):
                physical = a['delta'][key]['physical'] - b['delta'][key]['physical']
                model = a['delta'][key]['model'] - b['delta'][key]['model']
                all_order_wrong += int(np.sign(model) != np.sign(physical)); all_order_total += 1
                if abs(physical) > 2 * floor:
                    order_wrong += int(np.sign(model) != np.sign(physical)); order_total += 1
        resolved_mae = float(np.mean([r['mae'] for r in resolved_rows])) if resolved_rows else None
        effect = float(np.mean([r['mean_effect'] for r in resolved_rows])) if resolved_rows else None
        result[key] = {'floor': floor, 'family_count': len(families),
            'families_with_actions': len(family_rows), 'families_without_actions': len(families)-len(family_rows),
            'all_action_count': all_signs_total, 'all_sign_correct': all_signs,
            'all_order_wrong': all_order_wrong, 'all_order_count': all_order_total,
            'all_action_equal_family_mae': float(np.mean(family_rows)) if family_rows else None,
            'resolved_family_count': len(resolved_rows), 'unresolved_family_count': len(families)-len(resolved_rows),
            'resolved_equal_family_mae': resolved_mae, 'resolved_equal_family_mean_abs_effect': effect,
            'near_null_family_count': len(near_null_rows),
            'near_null_equal_family_mae': float(np.mean(near_null_rows)) if near_null_rows else None,
            'resolved_mae_over_effect': resolved_mae/effect if effect else None,
            'resolved_sign_correct': signs, 'resolved_sign_count': signs_total,
            'resolved_order_wrong': order_wrong, 'resolved_order_count': order_total}
    return result


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

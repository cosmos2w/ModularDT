"""Case-neutral reductions for physical state and finite-response supervision.

Cases supply physical observations, receiver identity/alignment, masks, scales,
functionals and decision requests. These helpers do not create or select data.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _nonnegative(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative.')
    return float(value)


def _positive_scale(value: Any) -> float:
    if isinstance(value, Mapping):
        value = value['scale']
    value = _nonnegative(value, 'Response scale')
    if value == 0:
        raise ValueError('Response scale must be positive.')
    return value


@dataclass(frozen=True)
class ResponseObjectiveConfig:
    """Weights for a new optional state/response objective.

    Endpoint and field weights are case-neutral. Functional names and weights
    are supplied by the case, and may be empty. Historical recipes retain their
    own explicit objective and are not implicitly migrated to this config.
    """

    enabled: bool = True
    endpoint_weight: float = 1.0
    response_weight: float = 1.0
    field_response_weight: float = 1.0
    functional_weights: Mapping[str, float] = field(default_factory=dict)
    warmup_epochs: int = 10
    ramp_epochs: int = 40
    coefficient_mode: str = 'gradient_norm'
    gradient_norm_ratios: Mapping[str, float] = field(
        default_factory=lambda: {'endpoint': 0.25, 'response': 0.10})

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('State/response enabled must be boolean.')
        for name in ('endpoint_weight', 'response_weight', 'field_response_weight'):
            _nonnegative(getattr(self, name), name)
        if type(self.warmup_epochs) is not int or self.warmup_epochs < 0:
            raise ValueError('State/response warmup must be nonnegative epochs.')
        if type(self.ramp_epochs) is not int or self.ramp_epochs < 1:
            raise ValueError('State/response ramp must be positive epochs.')
        if self.coefficient_mode not in ('gradient_norm', 'fixed'):
            raise ValueError('Coefficient mode must be gradient_norm or fixed.')
        if not isinstance(self.gradient_norm_ratios, Mapping) or set(self.gradient_norm_ratios) != {'endpoint', 'response'}:
            raise ValueError('Gradient norm ratios must specify endpoint and response.')
        for name, value in self.gradient_norm_ratios.items():
            _nonnegative(value, f'Gradient norm ratio {name!r}')
        if not isinstance(self.functional_weights, Mapping):
            raise TypeError('Functional weights must be a name-to-weight mapping.')
        for name, weight in self.functional_weights.items():
            if not isinstance(name, str) or not name:
                raise ValueError('Functional names must be nonempty strings.')
            _nonnegative(weight, f'Functional weight {name!r}')

    @classmethod
    def from_mapping(cls, values):
        """Parse a generic recipe section without changing legacy recipes."""
        if not isinstance(values, Mapping):
            raise TypeError('State/response configuration must be a mapping.')
        return cls(**values)

    def ramp(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError('State/response epoch must be a nonnegative integer.')
        return min(1., max(0., (epoch - self.warmup_epochs) / self.ramp_epochs))

    def coefficients(self, native_norm=None, auxiliary_norms=None):
        """Return effective outer weights; gradient mode requires measured norms."""
        multipliers = {'endpoint': float(self.endpoint_weight), 'response': float(self.response_weight)}
        if not self.enabled:
            return {name: 0. for name in multipliers}
        if self.coefficient_mode == 'fixed':
            return multipliers
        active_ratios = {name: self.gradient_norm_ratios[name] if multipliers[name] else 0.
                         for name in multipliers}
        if not any(active_ratios.values()):
            return {name: 0. for name in multipliers}
        if native_norm is None or auxiliary_norms is None:
            raise ValueError('Gradient-norm balancing requires TRAIN-measured component norms.')
        calibrated = calibrate_response_coefficients(native_norm, auxiliary_norms, active_ratios)
        return {name: multipliers[name] * calibrated[name] for name in multipliers}


def calibrate_response_coefficients(native_norm, auxiliary_norms, target_ratios):
    """Set auxiliary/native gradient norm ratios from case-selected TRAIN panels."""
    native_norm = _positive_scale(native_norm)
    if not isinstance(auxiliary_norms, Mapping) or not isinstance(target_ratios, Mapping) or set(auxiliary_norms) != set(target_ratios):
        raise ValueError('Auxiliary norms and desired ratios must have matching names.')
    coefficients = {}
    for name, ratio in target_ratios.items():
        ratio = _nonnegative(ratio, f'Gradient norm ratio {name!r}')
        auxiliary = _nonnegative(auxiliary_norms[name], f'Auxiliary gradient norm {name!r}')
        if ratio and not auxiliary:
            raise ValueError(f'Cannot calibrate a positive ratio with a zero {name!r} gradient norm.')
        coefficients[name] = ratio * native_norm / auxiliary if ratio else 0.
    return coefficients


def equal_mean(values):
    """Arithmetic mean, retaining tensor gradients and equal member weights."""
    values = tuple(values)
    if not values:
        raise ValueError('An equal mean needs at least one value.')
    return sum(values) / len(values)


def weighted_mean(values, weights=None):
    """Weighted mean for scalar values; weights need not sum to one."""
    values = tuple(values)
    if weights is None:
        return equal_mean(values)
    weights = tuple(weights)
    if not values or len(values) != len(weights):
        raise ValueError('Weighted values and weights must be nonempty and aligned.')
    weights = tuple(_nonnegative(weight, 'Weight') for weight in weights)
    denominator = sum(weights)
    if denominator == 0:
        raise ValueError('A weighted mean needs positive total weight.')
    return sum(weight * value for value, weight in zip(values, weights)) / denominator


def masked_weighted_mse(error, mask, weights):
    """Mean squared residual with explicit nonnegative receiver mass."""
    if not torch.is_tensor(error):
        raise TypeError('Residual must be a tensor.')
    mask = torch.as_tensor(mask, device=error.device, dtype=torch.bool)
    weights = torch.as_tensor(weights, device=error.device, dtype=error.dtype)
    if mask.shape != error.shape or weights.shape != error.shape:
        raise ValueError('Residual, mask and receiver weights must align.')
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError('Receiver weights must be finite and nonnegative.')
    active = mask & (weights > 0)
    if not bool(active.any()) or not bool(torch.isfinite(error[active]).all()):
        raise ValueError('Weighted support must contain finite residuals with positive mass.')
    return (error[active].square() * weights[active]).sum() / weights[active].sum()


def equal_family_mean(values, family_ids):
    """Average observations within each family, then average families equally."""
    values, family_ids = tuple(values), tuple(family_ids)
    if not values or len(values) != len(family_ids):
        raise ValueError('Family IDs must align with a nonempty observation sequence.')
    grouped = {}
    for family, value in zip(family_ids, values):
        grouped.setdefault(family, []).append(value)
    return equal_mean(equal_mean(group) for group in grouped.values())


def validate_aligned_receiver_ids(baseline_ids, edited_ids):
    """Validate explicit original receiver identities in their current order.

    Cases align or reorder the observations first and check their own coordinate
    convention; matching shapes alone cannot establish physical alignment.
    """
    baseline_ids, edited_ids = tuple(baseline_ids), tuple(edited_ids)
    if (not baseline_ids or len(set(baseline_ids)) != len(baseline_ids)
            or len(set(edited_ids)) != len(edited_ids) or baseline_ids != edited_ids):
        raise ValueError('Aligned receiver IDs must be unique and match in order.')
    return baseline_ids


def normalized_response_mse(baseline_prediction, edited_prediction, baseline_reference,
                            edited_reference, mask, scale, *, group_dim=None):
    """Finite-response squared residual on common physical support.

    With ``group_dim`` the chosen receiver dimension is reduced first, then
    nonempty groups are weighted equally. Without it, all valid entries have
    equal weight. The case must establish receiver-ID and coordinate alignment.
    """
    tensors = (baseline_prediction, edited_prediction, baseline_reference, edited_reference)
    if any(not torch.is_tensor(value) for value in tensors):
        raise TypeError('Response observations must be tensors.')
    if any(value.shape != tensors[0].shape for value in tensors[1:]):
        raise ValueError('Response predictions and references must be aligned.')
    mask = torch.as_tensor(mask, dtype=torch.bool, device=tensors[0].device)
    if mask.shape != tensors[0].shape or not bool(mask.any()):
        raise ValueError('Response mask must align and contain valid receivers.')
    scale = _positive_scale(scale)
    # Keep this operation order for compatibility with existing fitted recipes.
    error = (edited_prediction - baseline_prediction - edited_reference + baseline_reference) / scale
    if group_dim is None:
        return error[mask].square().mean()
    if not isinstance(group_dim, int) or not -error.ndim <= group_dim < error.ndim:
        raise ValueError('Invalid receiver group dimension.')
    count = mask.sum(group_dim)
    active = count > 0
    per_group = torch.where(mask, error, torch.zeros_like(error)).square().sum(group_dim) / count.clamp_min(1)
    return per_group[active].mean()


def normalized_delta_mse(predicted_delta, reference_delta, scale):
    """Normalized squared functional-delta error; inputs remain case-owned."""
    if not torch.is_tensor(predicted_delta):
        raise TypeError('Predicted functional response must be a tensor.')
    return ((predicted_delta - reference_delta) / _positive_scale(scale)).square().mean()


def combine_response_losses(field_loss, functional_losses, config: ResponseObjectiveConfig):
    """Combine a field loss with case-named functional losses at declared weights."""
    if set(functional_losses) != set(config.functional_weights):
        raise ValueError('Functional losses must match the configured functional weights.')
    if not config.enabled:
        return field_loss * 0
    combined = float(config.field_response_weight) * field_loss
    for name, loss in functional_losses.items():
        combined = combined + float(config.functional_weights[name]) * loss
    return combined


def field_metrics(prediction, target, mask):
    """Physical field metrics on the supplied validity mask and in its units."""
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
    return {'count': int(mask.sum()), 'rmse': rmse, 'reference_rms': reference_rms,
            'rmse_over_reference_rms': rmse / reference_rms if reference_rms > 0 else None,
            'mae': float(np.mean(np.abs(diff))), 'prediction_mean': float(prediction[mask].mean()),
            'physical_mean': float(target[mask].mean())}


def compare_prevalidated_response(baseline, edited):
    """Compare case-prevalidated endpoints and changes on common support.

    This compatibility seam requires the case to have already checked original
    receiver IDs, coordinate conventions and units before supplying observations.
    New generic callers should use ``compare_aligned_response`` with IDs.
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
        result[name] = {'baseline': field_metrics(bp, br, bm), 'edited': field_metrics(ep, er, em),
                        'response': field_metrics(ep - bp, er - br, common)}
    return result


def compare_aligned_response(baseline, edited):
    """Compare endpoints on their supports and changes on their common support.

    Each named field must supply prediction, reference, mask and ordered
    ``receiver_ids`` for both endpoints. The case owns physical units and checks
    any coordinate requirements beyond those explicit original IDs.
    """
    if not baseline or set(baseline) != set(edited):
        raise ValueError('Both endpoints must contain the same nonempty field set.')
    for name in baseline:
        base, edit = baseline[name], edited[name]
        ids = validate_aligned_receiver_ids(base['receiver_ids'], edit['receiver_ids'])
        for observation in (base, edit):
            shape = np.asarray(observation['prediction']).shape
            if not shape or shape[0] != len(ids):
                raise ValueError(f'Native observations for {name!r} must align with receiver IDs.')
    return compare_prevalidated_response(baseline, edited)


def summarize_response_actions(rows, floors, *, family_ids=None):
    """Equal-family action and resolution metrics for case-named functionals."""
    result = {}
    observed = {row['case_id'] for row in rows}
    families = sorted(observed if family_ids is None else set(family_ids))
    if not observed.issubset(families):
        raise ValueError('Observed actions contain a family outside the declared cohort.')
    for key in floors:
        floor = _positive_scale(floors[key])
        if any(not math.isfinite(float(row['delta'][key][source]))
               for row in rows for source in ('model', 'physical')):
            raise ValueError('Response values must be finite.')
        family_rows, resolved_rows, near_null_rows = [], [], []
        signs = signs_total = order_wrong = order_total = 0
        all_signs = all_signs_total = all_order_wrong = all_order_total = 0
        for family in families:
            cases = [row for row in rows if row['case_id'] == family]
            if not cases:
                continue
            errors = [abs(row['delta'][key]['model'] - row['delta'][key]['physical']) for row in cases]
            family_rows.append(float(np.mean(errors)))
            resolved = [row for row in cases if abs(row['delta'][key]['physical']) > floor]
            near_null = [row for row in cases if abs(row['delta'][key]['physical']) <= floor]
            if near_null:
                near_null_rows.append(float(np.mean([abs(row['delta'][key]['model'] - row['delta'][key]['physical']) for row in near_null])))
            if resolved:
                resolved_rows.append({'case_id': family,
                    'mae': float(np.mean([abs(row['delta'][key]['model'] - row['delta'][key]['physical']) for row in resolved])),
                    'mean_effect': float(np.mean([abs(row['delta'][key]['physical']) for row in resolved]))})
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
        resolved_mae = float(np.mean([row['mae'] for row in resolved_rows])) if resolved_rows else None
        effect = float(np.mean([row['mean_effect'] for row in resolved_rows])) if resolved_rows else None
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


def finite_pool_decision_losses(predictions, physical_deltas, *, observable_names, sensitivities,
                                research_scales, response_scales, requests,
                                resolution_multiplier=4.0, margin_fraction=0.25):
    """Differentiate finite-pool feasibility, ranking and near-null losses.

    Predictions have axes [view, baseline plus actions, observable]. The case
    supplies observable definitions, sensitivity, request objective/constraint
    pairs and band multipliers; this routine never creates design labels.
    """
    names, requests = tuple(observable_names), tuple(requests)
    if not names or len(set(names)) != len(names) or not requests:
        raise ValueError('Decision pools need unique observables and requests.')
    if any(set(mapping) != set(names) for mapping in (sensitivities, research_scales, response_scales)):
        raise ValueError('Decision metadata must match the declared observables.')
    y = np.asarray(physical_deltas, dtype=np.float64)
    if (not torch.is_tensor(predictions) or predictions.ndim != 3 or predictions.shape[0] < 1
            or predictions.shape[1] < 2 or predictions.shape[2] != len(names)
            or y.shape != (predictions.shape[1] - 1, len(names)) or not np.isfinite(y).all()):
        raise ValueError('Decision predictions and physical deltas must align and be finite.')
    resolution_multiplier = _positive_scale(resolution_multiplier)
    margin_fraction = _nonnegative(margin_fraction, 'Decision margin fraction')
    radii = [resolution_multiplier * _positive_scale(sensitivities[name]) for name in names]
    m = [_positive_scale(research_scales[name]) for name in names]
    s = [_positive_scale(response_scales[name]) for name in names]
    indices = {name: index for index, name in enumerate(names)}
    for objective, constraint, band in requests:
        if objective not in indices or constraint not in indices or objective == constraint:
            raise ValueError('Each decision request needs distinct declared objective and constraint.')
        if isinstance(band, bool) or not isinstance(band, Real) or not math.isfinite(band):
            raise ValueError('Decision band must be finite.')
    delta = predictions.to(torch.float64) - predictions[:, :1, :].to(torch.float64)
    observed_delta = delta.detach().cpu().numpy()
    truth = np.concatenate((np.zeros((1, len(names)), dtype=np.float64), y), axis=0)
    zero = delta.sum() * 0
    feasibility, ranking, nulls = [], [], []
    counts = {'requests': 0, 'feasible_actions': 0, 'infeasible_actions': 0,
              'ambiguous_actions': 0, 'ranking_pairs': 0,
              **{f'null_{name}': 0 for name in names},
              'active_feasible_hinges': 0, 'active_infeasible_hinges': 0,
              'active_ranking_hinges': 0, **{f'active_null_{name}_hinges': 0 for name in names}}
    for view in range(predictions.shape[0]):
        for objective_name, constraint_name, band in requests:
            objective, constraint = indices[objective_name], indices[constraint_name]
            bound = band * m[constraint]
            observed = y[:, constraint]
            feasible = observed + radii[constraint] <= bound
            infeasible = observed - radii[constraint] > bound
            counts['requests'] += 1
            counts['feasible_actions'] += int(feasible.sum())
            counts['infeasible_actions'] += int(infeasible.sum())
            counts['ambiguous_actions'] += int((~feasible & ~infeasible).sum())
            margin = np.minimum(margin_fraction * m[constraint], np.abs(observed - bound) - radii[constraint])
            observed_prediction = observed_delta[view, 1:, constraint]
            counts['active_feasible_hinges'] += int((feasible & (observed_prediction - bound + margin > 0)).sum())
            counts['active_infeasible_hinges'] += int((infeasible & (bound + margin - observed_prediction > 0)).sum())
            predicted = delta[view, 1:, constraint]
            classes = []
            for mask, sign in ((feasible, 1.), (infeasible, -1.)):
                selected = np.flatnonzero(mask).tolist()
                if selected:
                    residual = sign * (predicted[selected] - bound) + predicted.new_tensor(margin[selected])
                    classes.append(F.huber_loss(torch.relu(residual) / m[constraint],
                                                torch.zeros_like(residual), reduction='mean'))
            feasibility.append(sum(classes) / len(classes) if classes else zero)
            eligible = [0] + [i + 1 for i in range(len(y)) if bool(feasible[i])]
            pair_values, pair_weights = [], []
            for i in eligible:
                for j in eligible:
                    ri = 0. if i == 0 else radii[objective]
                    rj = 0. if j == 0 else radii[objective]
                    gap = truth[j, objective] - truth[i, objective] - ri - rj
                    if float(gap) <= 0:
                        continue
                    distance = min(margin_fraction * m[objective], float(gap))
                    weight = min(1., float(gap) / s[objective])
                    counts['active_ranking_hinges'] += int(
                        observed_delta[view, i, objective] - observed_delta[view, j, objective] + distance > 0)
                    hinge = torch.relu(delta[view, i, objective] - delta[view, j, objective] + distance) / m[objective]
                    pair_values.append(F.huber_loss(hinge, torch.zeros_like(hinge), reduction='sum') * weight)
                    pair_weights.append(weight)
            counts['ranking_pairs'] += len(pair_values)
            ranking.append(sum(pair_values) / sum(pair_weights) if pair_values else zero)
        for index, name in enumerate(names):
            mask = np.abs(y[:, index]) <= radii[index]
            counts[f'null_{name}'] += int(mask.sum())
            counts[f'active_null_{name}_hinges'] += int(
                (mask & (np.abs(observed_delta[view, 1:, index]) > radii[index])).sum())
            error = torch.relu(delta[view, 1:, index].abs() - radii[index]) / m[index]
            selected = np.flatnonzero(mask).tolist()
            nulls.append(F.huber_loss(error[selected], torch.zeros_like(error[selected]), reduction='mean')
                         if selected else zero)
    return ({'feasibility': sum(feasibility) / len(feasibility),
             'ranking': sum(ranking) / len(ranking), 'near_null': sum(nulls) / len(nulls)}, counts)


def evaluate_candidate_pool(predicted_deltas, physical_deltas, *, objective, constraint_caps,
                            baseline='baseline', objective_scale, sensitivity):
    """Select by predicted deltas, then assess a matched finite physical pool.

    Minimize the named objective subject to upper caps on named observables.
    Both mappings use ``action -> {observable: delta from baseline}``. When
    physical deltas are unavailable, only the model choice is returned; no
    acceptance or physical guarantee is inferred. With matched physical truth,
    an infeasible model choice falls back only to a physically feasible baseline.
    Best feasible and regret apply only to the supplied finite pool.
    """
    if not isinstance(predicted_deltas, Mapping) or not predicted_deltas:
        raise ValueError('Candidate predictions must be a nonempty action mapping.')
    if (not isinstance(objective, str) or not objective or not isinstance(baseline, str)
            or baseline not in predicted_deltas or not isinstance(constraint_caps, Mapping)):
        raise ValueError('A named objective, baseline action, and constraint caps are required.')
    actions = tuple(predicted_deltas)
    if any(not isinstance(action, str) or not action for action in actions):
        raise ValueError('Candidate action labels must be nonempty strings.')
    constraints = tuple(constraint_caps)
    if any(not isinstance(name, str) or not name for name in constraints):
        raise ValueError('Constraint observable names must be nonempty strings.')
    constraints = tuple(sorted(constraints))
    names = tuple(dict.fromkeys((objective, *constraints)))

    def finite(value, label):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f'{label} must be a finite number.')
        return float(value)

    caps = {name: finite(constraint_caps[name], f'Constraint cap {name!r}') for name in constraints}
    if not isinstance(sensitivity, Mapping) or any(name not in sensitivity for name in names):
        raise ValueError('Sensitivity must supply every objective and constraint observable.')
    thresholds = {name: _nonnegative(sensitivity[name], f'Sensitivity {name!r}') for name in names}
    scale = _positive_scale(objective_scale)

    def validate(values, label):
        if not isinstance(values, Mapping) or set(values) != set(actions):
            raise ValueError(f'{label} action IDs must match the predicted pool.')
        result = {}
        for action in actions:
            row = values[action]
            if not isinstance(row, Mapping) or any(name not in row for name in names):
                raise ValueError(f'{label} observables must match the decision request.')
            result[action] = {name: finite(row[name], f'{label} {action}/{name}') for name in names}
        if any(result[baseline][name] != 0. for name in names):
            raise ValueError(f'{label} baseline deltas must be zero.')
        return result

    predicted = validate(predicted_deltas, 'Predicted')
    physical = None if physical_deltas is None else validate(physical_deltas, 'Physical')

    def feasible(values):
        return all(values[name] <= caps[name] for name in constraints)

    def ranking(values, action):
        return (values[action][objective], *(values[action][name] for name in constraints), action)

    predicted_admitted = sorted((action for action in actions if feasible(predicted[action])),
                                key=lambda action: ranking(predicted, action))
    selected = predicted_admitted[0] if predicted_admitted else None
    result = {'objective': objective, 'constraint_caps': caps, 'baseline_action': baseline,
              'model_selected_action': selected, 'predicted_feasible_actions': predicted_admitted,
              'selected_predicted_delta': predicted[selected] if selected is not None else None,
              'physical_verification': 'unavailable' if physical is None else 'matched finite pool',
              'selected_physical_delta': None, 'raw_constraint_violation': None,
              'raw_constraint_excess': None, 'accepted_action': None, 'fallback_to_baseline': None,
              'best_admitted_physically_feasible_action': None, 'physical_best_action': None,
              'physical_optimum_excluded_by_model': None, 'objective_ordering_failure': None,
              'false_feasible_actions': None, 'false_rejected_actions': None,
              'near_null_false_rejected_actions': None, 'raw_improvement': None,
              'accepted_improvement': None,
              'physically_improving': None, 'numerically_resolved_improvement': None,
              'regret': None, 'normalized_regret': None}
    if physical is None:
        return result
    physically_feasible = sorted((action for action in actions if feasible(physical[action])),
                                 key=lambda action: ranking(physical, action))
    best = physically_feasible[0] if physically_feasible else None
    admitted_feasible = [action for action in predicted_admitted if feasible(physical[action])]
    best_admitted = min(admitted_feasible, key=lambda action: ranking(physical, action)) if admitted_feasible else None
    violated = selected is not None and not feasible(physical[selected])
    accepted = (selected if selected is not None and not violated else
                baseline if feasible(physical[baseline]) else None)
    false_feasible = [action for action in actions if action != baseline
                      and feasible(predicted[action]) and not feasible(physical[action])]
    false_rejected = [action for action in actions if action != baseline
                      and not feasible(predicted[action]) and feasible(physical[action])]
    near_null = [action for action in false_rejected
                 if any(predicted[action][name] > caps[name]
                        and abs(physical[action][name]) <= thresholds[name] for name in constraints)]
    result.update({
        'selected_physical_delta': physical[selected] if selected is not None else None,
        'raw_constraint_violation': bool(violated) if selected is not None else None,
        'raw_constraint_excess': ({name: max(0., physical[selected][name] - caps[name])
                                   for name in constraints} if selected is not None else None),
        'accepted_action': accepted,
        'fallback_to_baseline': bool(accepted == baseline and (selected is None or violated)),
        'raw_improvement': -physical[selected][objective] if selected is not None else None,
        'best_admitted_physically_feasible_action': best_admitted,
        'physical_best_action': best,
        'physical_optimum_excluded_by_model': (not feasible(predicted[best]) if best is not None else None),
        'objective_ordering_failure': (physical[selected][objective] > physical[best_admitted][objective]
                                       if selected is not None and not violated and best_admitted is not None else None),
        'false_feasible_actions': false_feasible, 'false_rejected_actions': false_rejected,
        'near_null_false_rejected_actions': near_null,
    })
    if accepted is not None:
        accepted_delta = physical[accepted][objective]
        result.update({'accepted_improvement': -accepted_delta, 'physically_improving': accepted_delta < 0.,
                       'numerically_resolved_improvement': accepted_delta < -4. * thresholds[objective]})
        if best is not None:
            regret = accepted_delta - physical[best][objective]
            result.update({'regret': regret, 'normalized_regret': regret / scale})
    return result

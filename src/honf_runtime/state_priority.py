"""One-sided response-gradient projection for optional auxiliary supervision."""
import math

import torch

STAT_DEFINITIONS = {
    "active_count": ("updates", "Auxiliary updates with state-priority projection"),
    "absent_count": ("updates", "Updates without an auxiliary episode"),
    "state_norm_sum": ("gradient units", "Sum of active weighted state-gradient norms"),
    "response_norm_sum": ("gradient units", "Sum of active weighted response-gradient norms"),
    "projected_count": ("updates", "Active updates with negative alignment projected"),
    "suppressed_count": ("updates", "Active updates with state norm at most 1e-12"),
    "zero_response_count": ("updates", "Active updates with exactly zero response gradient"),
    "cosine_sum": ("1", "Sum of defined state-response gradient cosines"),
    "cosine_count": ("updates", "Active updates with both gradients nonzero"),
    "retained_fraction_sum": ("1", "Sum of retained response norm fractions where defined"),
    "retained_fraction_count": ("updates", "Active updates with nonzero response gradient"),
    "clip_norm_sum": ("gradient units", "Sum of active combined norms before global clipping"),
    "clipped_count": ("updates", "Active updates with global clipping"),
    "displacement_dot_sum": ("gradient times parameter", "Sampled state-gradient dot actual AdamW parameter displacement, including weight decay"),
    "displacement_count": ("updates", "Sampled displacement measurements at ages 100, 500, and 1000"),
}


def recording_columns_and_definitions(provider):
    """Return the same optional columns and meanings for branch and resume."""
    columns = tuple(provider.training_history_columns())
    reader = getattr(provider, "training_column_definitions", None)
    definitions = dict(reader()) if callable(reader) else {}
    if getattr(provider, "state_priority_response_terms", None) is not None:
        columns += tuple(f"work_counts.state_priority_{name}" for name in STAT_DEFINITIONS)
        definitions.update({f"work_counts.state_priority_{name}": {
            "dtype": "float", "unit": unit, "meaning": meaning}
            for name, (unit, meaning) in STAT_DEFINITIONS.items()})
    return columns, definitions


def combine_state_response(parameters, response_gradients, *, state_floor=1e-12):
    """Combine accumulated state gradients and response gradients in place.

    A state L2 norm at or below 1e-12 suppresses the response, avoiding a
    division by numerical noise. Parameters unused by both objectives stay None.
    """
    state = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
    def dot(left, right):
        return sum(float((a.double() * b.double()).sum()) for a, b in zip(left, right)
                   if a is not None and b is not None)
    state_sq = dot(state, state)
    response_sq = dot(response_gradients, response_gradients)
    alignment = dot(state, response_gradients)
    if not all(math.isfinite(v) for v in (state_sq, response_sq, alignment)):
        raise FloatingPointError("Nonfinite state or response gradient.")
    suppressed = state_sq <= state_floor ** 2
    projected = not suppressed and alignment < 0.0
    factor = alignment / state_sq if projected else 0.0
    retained_sq = 0.0
    for p, state_gradient, response_gradient in zip(parameters, state, response_gradients):
        if suppressed:
            continue
        if response_gradient is None and (not projected or state_gradient is None):
            continue
        contribution = (torch.zeros_like(state_gradient) if response_gradient is None
                        else response_gradient.detach().clone())
        if projected and state_gradient is not None:
            contribution.add_(state_gradient, alpha=-factor)
        retained_sq += float(contribution.double().square().sum())
        if p.grad is None:
            p.grad = contribution
        else:
            p.grad.add_(contribution)
    result = {"active_count": 1.0, "state_norm_sum": math.sqrt(state_sq),
              "response_norm_sum": math.sqrt(response_sq),
              "projected_count": float(projected), "suppressed_count": float(suppressed),
              "zero_response_count": float(response_sq == 0.0)}
    if state_sq > 0 and response_sq > 0:
        result.update(cosine_sum=alignment / math.sqrt(state_sq * response_sq), cosine_count=1.0)
    if response_sq > 0:
        result.update(retained_fraction_sum=math.sqrt(retained_sq / response_sq), retained_fraction_count=1.0)
    return state, result

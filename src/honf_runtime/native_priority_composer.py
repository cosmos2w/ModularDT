"""Compose one effective batch's native and observed-response gradients."""
from __future__ import annotations

import math

import torch


STAT_DEFINITIONS = {
    "updates": ("count", "Effective batches composed"),
    "native_norm_sum": ("gradient units", "Sum of complete native gradient norms"),
    "selected_native_norm_sum": ("gradient units", "Sum of native gradient norms on the selected subset"),
    "auxiliary_norm_sum": ("gradient units", "Sum of raw weighted auxiliary gradient norms on the selected subset"),
    "preprojection_cosine_sum": ("1", "Sum of defined native and auxiliary subset gradient cosines"),
    "preprojection_cosine_count": ("count", "Effective batches with a defined subset gradient cosine"),
    "projection_count": ("count", "Effective batches with a negative subset dot product"),
    "cap_count": ("count", "Effective batches whose projected auxiliary gradient reached the norm cap"),
    "effective_auxiliary_norm_sum": ("gradient units", "Sum of projected and capped auxiliary gradient norms"),
    "clip_factor_sum": ("1", "Sum of final global clipping multipliers"),
}


def compose_native_priority(parameters, selected, response_gradients, maximum_native_norm_ratio):
    """Modify native ``.grad`` once after both complete-batch sums are known.

    ``response_gradients`` correspond to ``selected``; a missing gradient is a
    zero vector. Reductions remain on device until one compact statistics copy.
    """
    parameters, selected, response_gradients = tuple(parameters), tuple(selected), tuple(response_gradients)
    if (not parameters or not selected or len(selected) != len(response_gradients)
            or len({id(p) for p in parameters}) != len(parameters)
            or len({id(p) for p in selected}) != len(selected)
            or not {id(p) for p in selected} <= {id(p) for p in parameters}
            or isinstance(maximum_native_norm_ratio, bool)
            or not math.isfinite(float(maximum_native_norm_ratio)) or maximum_native_norm_ratio <= 0):
        raise ValueError("Invalid native-priority parameter inventory or norm ratio.")
    devices = {p.device for p in parameters}
    if len(devices) != 1:
        raise ValueError("Native-priority composition requires one parameter device.")
    zero = torch.zeros((), device=parameters[0].device, dtype=torch.float64)

    def squared(values):
        pieces = [value.detach().to(dtype=torch.float64).square().sum() for value in values if value is not None]
        return torch.stack(pieces).sum() if pieces else zero

    native_sq = squared(p.grad for p in parameters)
    selected_sq = squared(p.grad for p in selected)
    response_sq = squared(response_gradients)
    products = [(p.grad.detach().to(torch.float64) * response.detach().to(torch.float64)).sum()
                for p, response in zip(selected, response_gradients)
                if p.grad is not None and response is not None]
    dot = torch.stack(products).sum() if products else zero
    coefficient = torch.minimum(dot, zero) / torch.where(selected_sq > 0, selected_sq, torch.ones_like(selected_sq))
    projected = []
    for parameter, response in zip(selected, response_gradients):
        if response is None and parameter.grad is None:
            projected.append(None)
            continue
        value = torch.zeros_like(parameter) if response is None else response.detach().clone()
        if parameter.grad is not None:
            value.sub_((coefficient * parameter.grad.detach().to(torch.float64)).to(dtype=value.dtype))
        projected.append(value)
    projected_sq = squared(projected)
    native_norm, selected_norm = native_sq.sqrt(), selected_sq.sqrt()
    response_norm, projected_norm = response_sq.sqrt(), projected_sq.sqrt()
    cap = float(maximum_native_norm_ratio) * native_norm
    cap_factor = torch.where((native_norm > 0) & (projected_norm > 0),
        torch.minimum(torch.ones_like(cap), cap / torch.where(projected_norm > 0, projected_norm,
                                                               torch.ones_like(projected_norm))), zero)
    cosine_defined = (selected_norm > 0) & (response_norm > 0)
    cosine = torch.where(cosine_defined,
        dot / torch.where(cosine_defined, selected_norm * response_norm, torch.ones_like(dot)), zero)
    record = torch.stack((native_norm, selected_norm, response_norm, cosine,
                          cosine_defined.to(torch.float64), (dot < 0).to(torch.float64),
                          ((projected_norm > cap) & (projected_norm > 0)).to(torch.float64),
                          projected_norm * cap_factor))
    corrections = [None if value is None else value * cap_factor.to(dtype=value.dtype) for value in projected]
    # AdamW distinguishes a missing gradient from an allocated zero: the latter
    # advances moments and applies decay. Batch all such checks into the same
    # device-to-host transfer as the aggregate statistics, without one host
    # synchronization per parameter.
    absent_indices = [index for index, (parameter, correction) in enumerate(zip(selected, corrections))
                      if parameter.grad is None and correction is not None]
    nonzero_flags = [torch.count_nonzero(corrections[index]).ne(0).to(torch.float64) for index in absent_indices]
    values = torch.cat((record, torch.stack(nonzero_flags))) if nonzero_flags else record
    values = values.cpu().tolist()
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("Nonfinite native or auxiliary gradient in native-priority composition.")
    has_correction = dict(zip(absent_indices, values[len(record):]))
    for index, (parameter, correction) in enumerate(zip(selected, corrections)):
        if correction is None or (parameter.grad is None and not has_correction[index]):
            continue
        if parameter.grad is None:
            parameter.grad = correction
        else:
            parameter.grad.add_(correction)
    return dict(zip(("native_norm_sum", "selected_native_norm_sum", "auxiliary_norm_sum",
                     "preprojection_cosine_sum", "preprojection_cosine_count", "projection_count",
                     "cap_count", "effective_auxiliary_norm_sum"), values[:len(record)]), updates=1.0)

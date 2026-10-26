"""Exact sparse routing projections and denominator-free two-hop algebra."""

from __future__ import annotations

import torch

from .types import MeasureQueryProjection, TypedSourceIncidence


def _move_last(value: torch.Tensor, dim: int) -> tuple[torch.Tensor, int]:
    dim = dim if dim >= 0 else value.ndim + dim
    if dim < 0 or dim >= value.ndim:
        raise ValueError(f"dim={dim} is out of range for a {value.ndim}-D tensor.")
    return value.movedim(dim, -1), dim


def masked_sparsemax(
    logits: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Project logits onto a simplex with exact zeros outside ``valid``.

    Rows with no valid coordinate return all zeros.  A nonempty source row is
    expected to have at least one valid candidate and therefore has unit mass.
    The implementation uses finite sentinels for masked sort entries and does
    not form products such as ``0 * (-inf)``.
    """

    if not isinstance(logits, torch.Tensor) or logits.ndim == 0:
        raise ValueError("Sparsemax logits must be a tensor with at least one dimension.")
    if not logits.is_floating_point():
        raise TypeError("Sparsemax logits must use a floating-point dtype.")
    values, normalized_dim = _move_last(logits, dim)
    if valid is None:
        valid_values = torch.ones_like(values, dtype=torch.bool)
    else:
        if tuple(valid.shape) != tuple(logits.shape):
            raise ValueError("Sparsemax valid mask must have the same shape as logits.")
        valid_values, _ = _move_last(valid.to(device=logits.device, dtype=torch.bool), dim)
    count = int(values.shape[-1])
    if count == 0:
        return torch.zeros_like(logits)

    # Route scalars are cheap to promote and the wider prefix arithmetic avoids
    # overflow for a finite sentinel or tiny positive measures.  H-wide source
    # states remain in their model dtype.
    work_dtype = (
        torch.float64
        if values.dtype in {torch.float16, torch.bfloat16, torch.float32}
        else values.dtype
    )
    work_values = values.to(dtype=work_dtype)
    # A finite minimum avoids invalid arithmetic in prefix products while
    # remaining below every ordinary finite logit.  Invalid coordinates never
    # receive a gradient because their final result comes from torch.where.
    sentinel = torch.finfo(work_dtype).min if work_dtype.is_floating_point else -1.0e30
    sorted_values, sort_order = torch.sort(
        torch.where(valid_values, work_values, work_values.new_full((), sentinel)),
        dim=-1,
        descending=True,
        stable=True,
    )
    sorted_valid = torch.gather(valid_values, -1, sort_order)
    safe_sorted = torch.where(sorted_valid, sorted_values, torch.zeros_like(sorted_values))
    cumulative = safe_sorted.cumsum(dim=-1)
    ranks = torch.arange(1, count + 1, device=values.device, dtype=work_dtype)
    support_sorted = sorted_valid & (ranks * safe_sorted > cumulative - 1.0)
    support_count = support_sorted.sum(dim=-1)
    safe_count = support_count.clamp_min(1)
    threshold_index = (safe_count - 1).unsqueeze(-1)
    threshold = (cumulative.gather(-1, threshold_index).squeeze(-1) - 1.0) / safe_count.to(work_dtype)
    threshold = torch.where(support_count > 0, threshold, torch.zeros_like(threshold))
    projected = torch.where(
        valid_values,
        torch.clamp(work_values - threshold.unsqueeze(-1), min=0.0),
        torch.zeros_like(work_values),
    )
    return projected.movedim(-1, normalized_dim).to(dtype=logits.dtype)


def ordinary_source_sparsemax(
    logits: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Ordinary sparsemax used for each source-to-hub incidence row."""

    return masked_sparsemax(logits, valid, dim=dim)


def source_measure_sparsemax(
    logits: torch.Tensor,
    hub_measure: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    dim: int = -1,
) -> MeasureQueryProjection:
    """Apply the source-measure query projection from Equations (7)--(9).

    For occupied hubs the returned density is
    ``d = relu(logits - tau)`` and probability is ``alpha = hub_measure * d``.
    The threshold is solved from ``sum(mu * relu(z - tau)) == 1`` on each
    occupied row.  Source measures and selected route weights remain live in
    the autograd graph; only the support/ranking decisions are integer masks.
    """

    if not isinstance(logits, torch.Tensor) or logits.ndim == 0:
        raise ValueError("Query sparsemax logits must be a tensor with at least one dimension.")
    if not logits.is_floating_point():
        raise TypeError("Query sparsemax logits must use a floating-point dtype.")
    if not isinstance(hub_measure, torch.Tensor):
        raise TypeError("hub_measure must be a tensor.")
    values, normalized_dim = _move_last(logits, dim)
    if hub_measure.ndim != 2 and logits.ndim == 3:
        # Keep the general dimensional implementation below, but provide a
        # clearer error for the production [B,Q,K] shape.
        raise ValueError("hub_measure must have shape [B,K].")
    measure_values = hub_measure
    if int(measure_values.shape[-1]) != int(values.shape[-1]):
        raise ValueError("hub_measure must align with the projected hub dimension.")
    # Query rows are expected to share the batch/hub axes with mu.  Broadcast
    # a [B,K] measure over arbitrary intermediate query axes.
    if measure_values.ndim > values.ndim:
        raise ValueError("hub_measure has more dimensions than logits.")
    while measure_values.ndim < values.ndim:
        measure_values = measure_values.unsqueeze(-2)
    try:
        measure_values = torch.broadcast_to(measure_values, values.shape)
    except RuntimeError as error:
        raise ValueError("hub_measure must broadcast over logits leading dimensions.") from error
    if valid is None:
        valid_values = torch.ones_like(values, dtype=torch.bool)
    else:
        if tuple(valid.shape) != tuple(logits.shape):
            raise ValueError("Query sparsemax valid mask must have the same shape as logits.")
        valid_values, _ = _move_last(valid.to(device=logits.device, dtype=torch.bool), dim)
    occupied = valid_values & (measure_values > 0.0)
    count = int(values.shape[-1])
    if count == 0:
        zeros = torch.zeros_like(logits)
        return MeasureQueryProjection(zeros, zeros, logits.new_zeros(logits.shape[:-1]), occupied)

    work_dtype = (
        torch.float64
        if values.dtype in {torch.float16, torch.bfloat16, torch.float32}
        else values.dtype
    )
    work_values = values.to(dtype=work_dtype)
    work_measure = measure_values.to(dtype=work_dtype)
    sentinel = torch.finfo(work_dtype).min if work_dtype.is_floating_point else -1.0e30
    masked_values = torch.where(occupied, work_values, work_values.new_full((), sentinel))
    sorted_values, order = torch.sort(masked_values, dim=-1, descending=True, stable=True)
    sorted_occupied = torch.gather(occupied, -1, order)
    sorted_measure = torch.gather(work_measure, -1, order)
    # Do not multiply zero mass by the finite sentinel.  The safe value is only
    # used for prefix sums after invalid coordinates have been selected.
    safe_measure = torch.where(sorted_occupied, sorted_measure, torch.zeros_like(sorted_measure))
    safe_values = torch.where(sorted_occupied, sorted_values, torch.zeros_like(sorted_values))
    prefix_measure = safe_measure.cumsum(dim=-1)
    prefix_value_measure = (safe_measure * safe_values).cumsum(dim=-1)
    safe_prefix_measure = torch.where(
        prefix_measure > 0.0,
        prefix_measure,
        torch.ones_like(prefix_measure),
    )
    prefix_threshold = (prefix_value_measure - 1.0) / safe_prefix_measure
    support_sorted = sorted_occupied & (sorted_values > prefix_threshold)
    # The KKT support is a prefix under sorted logits.  Cumulative products
    # make that assumption explicit even in finite-precision tie cases.
    support_prefix = torch.cumprod(support_sorted.to(torch.int64), dim=-1)
    support_count = support_prefix.sum(dim=-1)
    safe_count = support_count.clamp_min(1)
    threshold_index = (safe_count - 1).unsqueeze(-1)
    threshold = prefix_threshold.gather(-1, threshold_index).squeeze(-1)
    threshold = torch.where(support_count > 0, threshold, torch.zeros_like(threshold))
    density = torch.where(
        occupied,
        torch.clamp(work_values - threshold.unsqueeze(-1), min=0.0),
        torch.zeros_like(work_values),
    )
    probability = torch.where(occupied, work_measure * density, torch.zeros_like(work_values))
    return MeasureQueryProjection(
        density=density.movedim(-1, normalized_dim),
        probability=probability.movedim(-1, normalized_dim),
        threshold=threshold,
        support=(occupied & (density > 0.0)).movedim(-1, normalized_dim),
    )


# A short alias mirrors the notation in the mathematical plan and makes call
# sites self-documenting when both ordinary and measure-aware projections are
# imported together.
query_source_measure_sparsemax = source_measure_sparsemax


def normalized_source_measure(
    weights: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize module/environment source measures while preserving gradients."""

    if weights.ndim != 2:
        raise ValueError("Source weights must have shape [B,N].")
    if valid is None:
        valid_values = weights > 0.0
    else:
        if tuple(valid.shape) != tuple(weights.shape):
            raise ValueError("Source validity must align with source weights.")
        valid_values = valid.to(device=weights.device, dtype=torch.bool) & (weights > 0.0)
    work_dtype = (
        torch.float64
        if weights.dtype in {torch.float16, torch.bfloat16, torch.float32}
        else weights.dtype
    )
    work_weights = weights.to(dtype=work_dtype)
    positive = torch.where(valid_values, work_weights, torch.zeros_like(work_weights))
    total = positive.sum(dim=-1, keepdim=True)
    safe_total = torch.where(total > 0.0, total, torch.ones_like(total))
    normalized = positive / safe_total
    return torch.where(total > 0.0, normalized, torch.zeros_like(normalized))


def module_source_measure(module_present: torch.Tensor) -> torch.Tensor:
    """Return the uniform measure over active module slots."""

    if module_present.ndim != 2:
        raise ValueError("module_present must have shape [B,M].")
    return normalized_source_measure(torch.ones_like(module_present), module_present > 0.5)


def environment_source_measure(env_weights: torch.Tensor) -> torch.Tensor:
    """Return normalized positive quadrature mass over environment samples."""

    return normalized_source_measure(env_weights, env_weights > 0.0)


def induced_hub_measure(
    source_weights: torch.Tensor,
    membership: torch.Tensor,
) -> torch.Tensor:
    """Compute ``mu_k = sum_i omega_i A_ik`` without detaching either input."""

    if source_weights.ndim != 2 or membership.ndim != 3:
        raise ValueError("source_weights/membership must have shapes [B,N] and [B,N,K].")
    if tuple(source_weights.shape) != tuple(membership.shape[:2]):
        raise ValueError("Source weights and membership must align on [B,N].")
    work_dtype = (
        torch.float64
        if source_weights.dtype in {torch.float16, torch.bfloat16, torch.float32}
        or membership.dtype in {torch.float16, torch.bfloat16, torch.float32}
        else torch.promote_types(source_weights.dtype, membership.dtype)
    )
    return torch.einsum(
        "bn,bnk->bk",
        source_weights.to(dtype=work_dtype),
        membership.to(dtype=work_dtype),
    )


def build_typed_source_incidence(
    source_weights: torch.Tensor,
    membership: torch.Tensor,
    *,
    source_valid: torch.Tensor | None = None,
) -> TypedSourceIncidence:
    """Create a typed incidence container and its positive inverted view."""

    from .pair_join import build_inverted_source_incidence

    hub_measure = induced_hub_measure(source_weights, membership)
    inverted = build_inverted_source_incidence(
        membership,
        source_weights=source_weights,
        source_valid=source_valid,
    )
    return TypedSourceIncidence(source_weights, membership, hub_measure, inverted)

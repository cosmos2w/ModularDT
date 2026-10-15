"""Pure-Torch source-read scoring, support plans, and packet layouts.

The utilities here do not change the maintained A2-D reader. They provide a
small controller and source-preserving plans for callers that opt into the
Round 3 sparse-read path. In particular, source identity is represented only
by tensor-axis indices in plans; source-ID values are never scorer inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _positive_scalar(value: float | Tensor, name: str) -> Tensor:
    result = torch.as_tensor(value)
    if result.numel() != 1:
        raise ValueError(f"{name} must be a scalar.")
    _require_all(torch.isfinite(result) & (result > 0), f"{name} must be finite and positive.")
    return result.reshape(())


def _require_all(condition: Tensor, message: str) -> None:
    """Validate synchronously on CPU and without a CUDA host round-trip."""
    valid = condition.all()
    if valid.device.type == "cuda":
        torch._assert_async(valid, message)
    elif not bool(valid):
        raise ValueError(message)


def _positive_broadcast(value: float | Tensor, name: str, reference: Tensor) -> Tensor:
    result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    try:
        broadcast_shape = torch.broadcast_shapes(tuple(result.shape), tuple(reference.shape))
    except RuntimeError as exc:
        raise ValueError(f"{name} must broadcast to the reference shape {tuple(reference.shape)}.") from exc
    if tuple(broadcast_shape) != tuple(reference.shape):
        raise ValueError(f"{name} must broadcast to the reference shape {tuple(reference.shape)}.")
    _require_all(torch.isfinite(result) & (result > 0), f"{name} must be finite and positive.")
    return result


def _expand_source_mask(mask: Tensor, batch: int, queries: int, sources: int, name: str) -> Tensor:
    if mask.dtype != torch.bool:
        raise ValueError(f"{name} must be a boolean tensor.")
    if tuple(mask.shape) == (batch, sources):
        return mask[:, None, :].expand(batch, queries, sources)
    if tuple(mask.shape) == (batch, queries, sources):
        return mask
    raise ValueError(f"{name} must have shape [B,M] or [B,Q,M].")


def _smoothstep5(value: Tensor) -> Tensor:
    t = value.clamp(0.0, 1.0)
    return t.pow(3) * (10.0 + t * (-15.0 + 6.0 * t))


def smooth_threshold_gate(scores: Tensor, threshold: float | Tensor) -> Tensor:
    """Endpoint-flat threshold gate with support only where ``scores > threshold``.

    The transition occupies ``[threshold, 2*threshold]``. The fixed threshold
    is supplied by the caller and is never estimated from the current batch.
    """
    tau = _positive_scalar(threshold, "threshold").to(device=scores.device, dtype=scores.dtype)
    return _smoothstep5((scores - tau) / tau)


def smooth_near_gate(distance: Tensor, radius: float | Tensor) -> Tensor:
    """Smooth geometry-only protection with scalar or per-pair radius values.

    For example, a radius shaped ``[B,1,M]`` broadcasts over query positions.
    The gate is one through half-radius and zero at radius.
    """
    r = _positive_broadcast(radius, "radius", distance)
    return _smoothstep5(2.0 * (r - distance) / r)


def combine_source_gates(utility_gate: Tensor, near_gate: Tensor) -> Tensor:
    """Combine utility and smooth near protection without a Boolean override."""
    if utility_gate.shape != near_gate.shape:
        raise ValueError("utility_gate and near_gate must have equal shapes.")
    _require_all(
        torch.isfinite(utility_gate) & (utility_gate >= 0.0) & (utility_gate <= 1.0),
        "utility_gate must be finite and lie in [0,1].",
    )
    _require_all(
        torch.isfinite(near_gate) & (near_gate >= 0.0) & (near_gate <= 1.0),
        "near_gate must be finite and lie in [0,1].",
    )
    return 1.0 - (1.0 - utility_gate) * (1.0 - near_gate)


class SourceReadControllerV1(nn.Module):
    """Small learned source-utility scorer and rank-r source-preserving completion.

    ``utility_scores`` receives only projected contextual states and the
    existing ``2*C+1`` lightweight geometry descriptor. It has no current
    heating-vector argument, so Thermal support is independent of the current
    control call. Projection results can be prepared once and reused by the
    caller while their parameter/version binding remains valid.
    """

    def __init__(
        self,
        width: int,
        coordinate_dim: int,
        heads: int = 4,
        thermal: bool = True,
        rank: int = 16,
        scorer_width: int = 32,
        completion: bool = True,
    ) -> None:
        super().__init__()
        if width < 1 or coordinate_dim < 1 or heads < 1 or rank < 1 or scorer_width < 1:
            raise ValueError("width, coordinate_dim, heads, rank, and scorer_width must be positive.")
        if width % heads:
            raise ValueError("width must be divisible by heads.")
        self.width = int(width)
        self.coordinate_dim = int(coordinate_dim)
        self.heads = int(heads)
        self.rank = int(rank)
        self.thermal = bool(thermal)
        self.completion_enabled = bool(completion)
        self.scorer_width = int(scorer_width)
        self.score_channels = self.heads + int(self.thermal)

        # These projections are small enough to prepare and reuse independently
        # of the expensive nonlinear pair-value path.
        self.source_projection = nn.Linear(self.width, 8, bias=False)
        self.query_projection = nn.Linear(self.width, 8, bias=False)
        score_input_width = 8 + 8 + 2 * self.coordinate_dim + 1
        self.utility_scorer = nn.Sequential(
            nn.Linear(score_input_width, self.scorer_width),
            nn.SiLU(),
            nn.Linear(self.scorer_width, self.score_channels),
        )

        if self.completion_enabled:
            self.completion_query_projection: nn.Linear | None = nn.Linear(
                self.width, self.rank, bias=False
            )
            self.completion_source_projection: nn.Linear | None = nn.Linear(
                self.width, self.rank, bias=False
            )
            self.completion_output: nn.Linear | None = nn.Linear(self.rank, self.width, bias=False)
        else:
            # S-Utility has no low-rank completion and therefore no unused
            # trainable weights, optimizer moments, or checkpoint tensors.
            self.completion_query_projection = None
            self.completion_source_projection = None
            self.completion_output = None

        # Fixed-size buffers keep state-dict shapes stable. ``threshold_count``
        # marks the active prefix; at most two calibrated operating points fit.
        self.register_buffer("thresholds", torch.zeros(2, dtype=torch.float32))
        self.register_buffer("threshold_count", torch.zeros((), dtype=torch.int64))
        self.register_buffer("context_scales", torch.ones(self.heads, dtype=torch.float32))
        self.register_buffer("heating_envelope", torch.ones((), dtype=torch.float32))
        self.register_buffer("trained", torch.zeros((), dtype=torch.bool))
        self.register_buffer("calibrated", torch.zeros((), dtype=torch.bool))
        self.register_buffer("metadata_frozen", torch.zeros((), dtype=torch.bool))

    def project_source_states(self, source_states: Tensor) -> Tensor:
        if source_states.ndim != 3 or source_states.shape[-1] != self.width:
            raise ValueError("source_states must have shape [B,M,width].")
        return self.source_projection(source_states)

    def project_query_states(self, query_states: Tensor) -> Tensor:
        if query_states.ndim != 3 or query_states.shape[-1] != self.width:
            raise ValueError("query_states must have shape [B,Q,width].")
        return self.query_projection(query_states)

    def utility_scores(
        self,
        source_projection: Tensor,
        query_projection: Tensor,
        geometry_descriptor: Tensor,
    ) -> Tensor:
        """Return nonnegative ``[B,Q,M,heads(+thermal)]`` utility estimates."""
        if source_projection.ndim != 3 or source_projection.shape[-1] != 8:
            raise ValueError("source_projection must have shape [B,M,8].")
        if query_projection.ndim != 3 or query_projection.shape[-1] != 8:
            raise ValueError("query_projection must have shape [B,Q,8].")
        batch, sources, _ = source_projection.shape
        if query_projection.shape[0] != batch:
            raise ValueError("query and source projections must share the batch axis.")
        queries = query_projection.shape[1]
        expected_descriptor = (batch, queries, sources, 2 * self.coordinate_dim + 1)
        if tuple(geometry_descriptor.shape) != expected_descriptor:
            raise ValueError(
                "geometry_descriptor must have shape "
                f"[B,Q,M,{2 * self.coordinate_dim + 1}]."
            )
        query_pairs = query_projection[:, :, None, :].expand(batch, queries, sources, 8)
        source_pairs = source_projection[:, None, :, :].expand(batch, queries, sources, 8)
        features = torch.cat((query_pairs, source_pairs, geometry_descriptor), dim=-1)
        return F.softplus(self.utility_scorer(features))

    def forward(
        self, query_states: Tensor, source_states: Tensor, geometry_descriptor: Tensor
    ) -> Tensor:
        """Convenience scorer call; no current heat or source-ID values are accepted."""
        return self.utility_scores(
            self.project_source_states(source_states),
            self.project_query_states(query_states),
            geometry_descriptor,
        )

    def install_calibration_(
        self,
        *,
        thresholds: Sequence[float],
        context_scales: Sequence[float] | Tensor,
        heating_envelope: float,
    ) -> None:
        """Install the fixed TRAIN-calibrated operating metadata once, pre-freeze."""
        _require_all(~self.metadata_frozen, "Source-read calibration metadata is frozen.")
        _require_all(~self.calibrated, "Source-read calibration metadata was already installed.")
        threshold_values = tuple(float(value) for value in thresholds)
        if not 1 <= len(threshold_values) <= 2:
            raise ValueError("One or two positive operating thresholds are required.")
        if any(not math.isfinite(value) or value <= 0 for value in threshold_values):
            raise ValueError("Operating thresholds must be finite and positive.")
        scales = torch.as_tensor(context_scales, dtype=self.context_scales.dtype)
        if tuple(scales.shape) != (self.heads,):
            raise ValueError(f"context_scales must contain exactly {self.heads} values.")
        _require_all(torch.isfinite(scales) & (scales > 0), "context_scales must be finite and positive.")
        envelope = _positive_scalar(heating_envelope, "heating_envelope")
        with torch.no_grad():
            self.thresholds.zero_()
            self.thresholds[: len(threshold_values)].copy_(
                torch.tensor(threshold_values, dtype=self.thresholds.dtype, device=self.thresholds.device)
            )
            self.threshold_count.fill_(len(threshold_values))
            self.context_scales.copy_(scales.to(device=self.context_scales.device))
            self.heating_envelope.copy_(
                envelope.to(device=self.heating_envelope.device, dtype=self.heating_envelope.dtype)
            )
            self.calibrated.fill_(True)

    def mark_trained_(self) -> None:
        """Record training completion without changing forward behavior."""
        _require_all(~self.metadata_frozen, "Frozen source-read metadata cannot be changed.")
        self.trained.fill_(True)

    def freeze_metadata_(self) -> None:
        """Freeze the persisted operating metadata after fitting/calibration."""
        _require_all(self.calibrated, "Install TRAIN calibration before freezing source-read metadata.")
        self.metadata_frozen.fill_(True)

    def dense_utility_targets(
        self,
        original_alpha: Tensor,
        fine_messages: Tensor,
        thermal_k: Tensor | None,
        temperature_scale: float | Tensor,
        *,
        completion_messages: Tensor | None = None,
        completion_k: Tensor | None = None,
    ) -> Tensor:
        """Build dense TRAIN-teacher utility magnitudes for S-Utility/S-Complete.

        ``fine_messages`` and ``completion_messages`` are post-``W_o`` values;
        differencing them therefore cancels the shared output projection bias.
        Thermal K is differenced directly and is not multiplied by attention.
        Callers must restrict teacher construction to the declared TRAIN scope.
        """
        if original_alpha.ndim != 4 or original_alpha.shape[-1] != self.heads:
            raise ValueError("original_alpha must have shape [B,Q,M,heads].")
        batch, queries, sources, _ = original_alpha.shape
        expected_messages = (batch, queries, sources, self.width)
        if tuple(fine_messages.shape) != expected_messages:
            raise ValueError("fine_messages must have shape [B,Q,M,width].")
        if completion_messages is not None and tuple(completion_messages.shape) != expected_messages:
            raise ValueError("completion_messages must match fine_messages.")
        residual_messages = fine_messages if completion_messages is None else fine_messages - completion_messages
        head_width = self.width // self.heads
        residual_heads = residual_messages.reshape(batch, queries, sources, self.heads, head_width)
        flow_targets = (original_alpha[..., None] * residual_heads).norm(dim=-1)
        flow_targets = flow_targets / self.context_scales.to(flow_targets)[None, None, None, :]
        blocks = [flow_targets]

        if self.thermal:
            if thermal_k is None or tuple(thermal_k.shape) != (batch, queries, sources):
                raise ValueError("Thermal controller targets require thermal_k with shape [B,Q,M].")
            if completion_k is not None and tuple(completion_k.shape) != (batch, queries, sources):
                raise ValueError("completion_k must match thermal_k.")
            temp_scale = _positive_scalar(temperature_scale, "temperature_scale").to(thermal_k)
            residual_k = thermal_k if completion_k is None else thermal_k - completion_k
            thermal_target = self.heating_envelope.to(thermal_k) * residual_k.abs() / temp_scale
            blocks.append(thermal_target.unsqueeze(-1))
        elif thermal_k is not None or completion_k is not None:
            raise ValueError("thermal_k/completion_k were supplied to a controller without Thermal output.")
        return torch.cat(blocks, dim=-1).detach()

    def project_completion_queries(self, query_states: Tensor) -> Tensor:
        self._require_completion()
        if query_states.ndim != 3 or query_states.shape[-1] != self.width:
            raise ValueError("query_states must have shape [B,Q,width].")
        assert self.completion_query_projection is not None
        return self.completion_query_projection(query_states)

    def project_completion_sources(self, module_states: Tensor) -> Tensor:
        self._require_completion()
        if module_states.ndim != 3 or module_states.shape[-1] != self.width:
            raise ValueError("module_states must have shape [B,M,width].")
        assert self.completion_source_projection is not None
        return self.completion_source_projection(module_states)

    def _require_completion(self) -> None:
        if not self.completion_enabled:
            raise ValueError("Rank completion is disabled for this source-read controller.")

    def completion_pair_hidden_from_projections(
        self,
        query_projection: Tensor,
        source_projection: Tensor,
        flat_batch_index: Tensor,
        flat_query_index: Tensor,
        flat_source_index: Tensor,
    ) -> Tensor:
        """Evaluate U(a_q*b_i) only for the supplied physical-pair indices."""
        self._require_completion()
        if query_projection.ndim != 3 or query_projection.shape[-1] != self.rank:
            raise ValueError("query_projection must have shape [B,Q,rank].")
        if source_projection.ndim != 3 or source_projection.shape[-1] != self.rank:
            raise ValueError("source_projection must have shape [B,M,rank].")
        indices = (flat_batch_index, flat_query_index, flat_source_index)
        if any(index.ndim != 1 or index.dtype != torch.long for index in indices):
            raise ValueError("flat batch/query/source indices must be rank-1 int64 tensors.")
        if len({index.numel() for index in indices}) != 1:
            raise ValueError("flat pair index arrays must have equal lengths.")
        if query_projection.shape[0] != source_projection.shape[0]:
            raise ValueError("query and source completion projections must share the batch axis.")
        q = query_projection[flat_batch_index, flat_query_index]
        s = source_projection[flat_batch_index, flat_source_index]
        assert self.completion_output is not None
        return self.completion_output(q * s)

    def completion_headwise_sum_from_projections(
        self, query_projection: Tensor, source_projection: Tensor, original_alpha: Tensor
    ) -> Tensor:
        """Return ``[B,Q,H,width]`` completion sums without a ``[B,Q,M,width]`` tensor."""
        self._require_completion()
        if original_alpha.ndim != 4 or original_alpha.shape[-1] != self.heads:
            raise ValueError("original_alpha must have shape [B,Q,M,heads].")
        batch, queries, sources, _ = original_alpha.shape
        if tuple(query_projection.shape) != (batch, queries, self.rank):
            raise ValueError("query_projection does not match alpha batch/query axes and rank.")
        if tuple(source_projection.shape) != (batch, sources, self.rank):
            raise ValueError("source_projection does not match alpha batch/source axes and rank.")
        weighted_source = torch.einsum("bqmh,bmr->bqhr", original_alpha, source_projection)
        assert self.completion_output is not None
        return self.completion_output(query_projection[:, :, None, :] * weighted_source)

    def completion_weighted_sum_from_projections(
        self, query_projection: Tensor, source_projection: Tensor, weights: Tensor
    ) -> Tensor:
        """Contract arbitrary source weights through the rank-r completion."""
        self._require_completion()
        if weights.ndim != 3:
            raise ValueError("weights must have shape [B,Q,M].")
        batch, queries, sources = weights.shape
        if tuple(query_projection.shape) != (batch, queries, self.rank):
            raise ValueError("query_projection does not match weights and completion rank.")
        if tuple(source_projection.shape) != (batch, sources, self.rank):
            raise ValueError("source_projection does not match weights and completion rank.")
        weighted_source = torch.einsum("bqm,bmr->bqr", weights, source_projection)
        assert self.completion_output is not None
        return self.completion_output(query_projection * weighted_source)

    def completion_pair_hidden(
        self,
        query_states: Tensor,
        source_states: Tensor,
        flat_batch_index: Tensor,
        flat_query_index: Tensor,
        flat_source_index: Tensor,
    ) -> Tensor:
        return self.completion_pair_hidden_from_projections(
            self.project_completion_queries(query_states),
            self.project_completion_sources(source_states),
            flat_batch_index,
            flat_query_index,
            flat_source_index,
        )

    def completion_headwise_sum(
        self, query_states: Tensor, source_states: Tensor, original_alpha: Tensor
    ) -> Tensor:
        return self.completion_headwise_sum_from_projections(
            self.project_completion_queries(query_states),
            self.project_completion_sources(source_states),
            original_alpha,
        )


@dataclass(frozen=True)
class PackedSourcePairsV1:
    """Flat admitted ``(batch, query, source)`` pairs and original read weights."""

    batch_index: Tensor
    query_index: Tensor
    source_index: Tensor
    gate: Tensor
    original_alpha: Tensor

    @property
    def pair_count(self) -> int:
        return self.source_index.numel()


@dataclass(frozen=True)
class SourceReadPlanV1:
    """Immutable source support, smooth gates, and non-renormalized alpha values."""

    support_mask: Tensor
    gates: Tensor
    original_alpha: Tensor
    source_present_mask: Tensor
    query_valid_mask: Tensor
    protected_support: Tensor
    packed_pairs: PackedSourcePairsV1

    @property
    def batch_size(self) -> int:
        return self.gates.shape[0]

    @property
    def query_count(self) -> int:
        return self.gates.shape[1]

    @property
    def source_count(self) -> int:
        return self.gates.shape[2]

    @property
    def pair_count(self) -> int:
        return self.packed_pairs.pair_count


def build_source_read_plan_v1(
    gates: Tensor,
    original_alpha: Tensor,
    source_present_mask: Tensor,
    *,
    query_valid_mask: Tensor | None = None,
    protected_mask: Tensor | None = None,
) -> SourceReadPlanV1:
    """Bind support to gates, valid source/query masks, and original alpha.

    ``protected_mask`` is a declared hard-protection diagnostic/control input;
    it may keep a present zero-attention source in support. Smooth near-source
    protection belongs in ``gates`` via ``combine_source_gates`` and must not
    be replaced by this Boolean mask on a deployed path.
    """
    if gates.ndim != 3 or not gates.is_floating_point():
        raise ValueError("gates must be a floating tensor with shape [B,Q,M].")
    batch, queries, sources = gates.shape
    if original_alpha.ndim != 4 or tuple(original_alpha.shape[:3]) != (batch, queries, sources):
        raise ValueError("original_alpha must have shape [B,Q,M,H] matching gates.")
    if not original_alpha.is_floating_point() or original_alpha.shape[-1] < 1:
        raise ValueError("original_alpha must have a positive feature-head axis.")
    _require_all(torch.isfinite(gates), "gates must be finite.")
    _require_all((gates >= 0.0) & (gates <= 1.0), "gates must lie in [0,1].")
    _require_all(torch.isfinite(original_alpha), "original_alpha must be finite.")
    _require_all(original_alpha >= 0.0, "original_alpha must be nonnegative.")
    if tuple(source_present_mask.shape) != (batch, sources) or source_present_mask.dtype != torch.bool:
        raise ValueError("source_present_mask must be boolean [B,M].")
    if query_valid_mask is None:
        query_valid_mask = torch.ones((batch, queries), dtype=torch.bool, device=gates.device)
    if tuple(query_valid_mask.shape) != (batch, queries) or query_valid_mask.dtype != torch.bool:
        raise ValueError("query_valid_mask must be boolean [B,Q].")
    if gates.device != original_alpha.device or gates.device != source_present_mask.device:
        raise ValueError("gates, alpha, and source masks must share a device.")
    if query_valid_mask.device != gates.device:
        raise ValueError("query_valid_mask must share the gates device.")
    if protected_mask is None:
        protected = torch.zeros((batch, queries, sources), dtype=torch.bool, device=gates.device)
    else:
        protected = _expand_source_mask(protected_mask, batch, queries, sources, "protected_mask")
        if protected.device != gates.device:
            raise ValueError("protected_mask must share the gates device.")
    present = source_present_mask[:, None, :].expand(batch, queries, sources)
    query_valid = query_valid_mask[:, :, None].expand(batch, queries, sources)
    valid_pairs = present & query_valid
    protected = protected & valid_pairs
    admitted_gates = gates * valid_pairs.to(gates.dtype)
    admitted_gates = torch.where(protected, torch.ones_like(admitted_gates), admitted_gates)
    support = valid_pairs & ((admitted_gates > 0.0) | protected)
    flat_batch, flat_query, flat_source = support.nonzero(as_tuple=True)
    packed = PackedSourcePairsV1(
        batch_index=flat_batch,
        query_index=flat_query,
        source_index=flat_source,
        gate=admitted_gates[flat_batch, flat_query, flat_source],
        original_alpha=original_alpha[flat_batch, flat_query, flat_source],
    )
    return SourceReadPlanV1(
        support_mask=support,
        gates=admitted_gates,
        original_alpha=original_alpha,
        source_present_mask=source_present_mask,
        query_valid_mask=query_valid_mask,
        protected_support=protected,
        packed_pairs=packed,
    )


def pack_source_pairs(plan: SourceReadPlanV1) -> PackedSourcePairsV1:
    """Return every admitted physical pair exactly once, in row-major order."""
    return plan.packed_pairs


@dataclass(frozen=True)
class DegreeBucketPacketV1:
    """One group of query packets padded to a shared small source degree."""

    width: int
    batch_index: Tensor
    query_index: Tensor
    source_indices: Tensor
    padding_mask: Tensor
    valid_pair_count: int

    @property
    def query_count(self) -> int:
        return self.query_index.numel()

    @property
    def allocated_pair_slots(self) -> int:
        return self.source_indices.numel()


@dataclass(frozen=True)
class DegreeBucketPacketLayoutV1:
    """General index-list packet layout with explicit padding and work counts."""

    buckets: tuple[DegreeBucketPacketV1, ...]
    empty_batch_index: Tensor
    empty_query_index: Tensor
    valid_pair_count: int
    allocated_pair_slots: int
    padding_pair_count: int
    bucket_assignment_query_count: int
    bucket_membership_comparison_count: int
    query_sort_count: int
    gather_pair_count: int
    bucket_widths: tuple[int, ...]


def _degree_bucket_widths(source_count: int) -> tuple[int, ...]:
    if source_count <= 0:
        return ()
    widths: list[int] = []
    width = 1
    while width < source_count:
        widths.append(width)
        width *= 2
    widths.append(source_count if widths and widths[-1] != source_count else width)
    # The loop's final power-of-two can equal source_count; avoid a duplicate.
    return tuple(dict.fromkeys(widths))


def build_degree_bucket_packet_layout(plan: SourceReadPlanV1) -> DegreeBucketPacketLayoutV1:
    """Group query packets into 1/2/4/.../M degree buckets without query loops.

    The only Python loop is over the fixed logarithmic set of bucket widths.
    Each query's source list remains individual and source-preserving; padding
    is reported explicitly and uses ``-1`` plus a True ``padding_mask``.
    """
    batch, queries, sources = plan.support_mask.shape
    flat_query_count = batch * queries
    device = plan.support_mask.device
    degrees = plan.support_mask.reshape(flat_query_count, sources).sum(dim=-1)
    flat_rows = torch.arange(flat_query_count, device=device, dtype=torch.long)
    flat_batch = torch.div(flat_rows, max(queries, 1), rounding_mode="floor")
    flat_query = flat_rows.remainder(max(queries, 1))
    widths = _degree_bucket_widths(sources)
    buckets: list[DegreeBucketPacketV1] = []
    lower = 0
    for width in widths:
        in_bucket = (degrees > lower) & (degrees <= width)
        row_ids = torch.nonzero(in_bucket, as_tuple=False).flatten()
        row_degrees = degrees.index_select(0, row_ids)
        row_batch = flat_batch.index_select(0, row_ids)
        row_query = flat_query.index_select(0, row_ids)
        row_support = plan.support_mask.reshape(flat_query_count, sources).index_select(0, row_ids)
        pair_rows, pair_sources = row_support.nonzero(as_tuple=True)
        source_indices = torch.full(
            (row_ids.numel(), width), -1, dtype=torch.long, device=device
        )
        row_pair_starts = row_degrees.cumsum(0) - row_degrees
        pair_rank = torch.arange(pair_sources.numel(), device=device, dtype=torch.long)
        pair_rank = pair_rank - torch.repeat_interleave(
            row_pair_starts, row_degrees, output_size=pair_sources.numel()
        )
        linear_slot = pair_rows * width + pair_rank
        source_indices.reshape(-1).index_copy_(0, linear_slot, pair_sources)
        padding_mask = torch.arange(width, device=device)[None, :] >= row_degrees[:, None]
        buckets.append(
            DegreeBucketPacketV1(
                width=width,
                batch_index=row_batch,
                query_index=row_query,
                source_indices=source_indices,
                padding_mask=padding_mask,
                valid_pair_count=pair_sources.numel(),
            )
        )
        lower = width

    empty_rows = torch.nonzero(degrees == 0, as_tuple=False).flatten()
    empty_batch = flat_batch.index_select(0, empty_rows)
    empty_query = flat_query.index_select(0, empty_rows)
    allocated = sum(bucket.allocated_pair_slots for bucket in buckets)
    valid = plan.pair_count
    return DegreeBucketPacketLayoutV1(
        buckets=tuple(buckets),
        empty_batch_index=empty_batch,
        empty_query_index=empty_query,
        valid_pair_count=valid,
        allocated_pair_slots=allocated,
        padding_pair_count=allocated - valid,
        bucket_assignment_query_count=flat_query_count,
        bucket_membership_comparison_count=2 * flat_query_count * len(widths),
        query_sort_count=0,
        gather_pair_count=valid,
        bucket_widths=widths,
    )


def s_attn_support(
    original_alpha: Tensor,
    valid_source_mask: Tensor,
    *,
    retention: float = 0.95,
    protected_mask: Tensor | None = None,
) -> Tensor:
    """Union the smallest per-head attention-mass supports at 95% or 99%."""
    if not math.isclose(float(retention), 0.95, abs_tol=1.0e-12) and not math.isclose(
        float(retention), 0.99, abs_tol=1.0e-12
    ):
        raise ValueError("S-Attn retention must be exactly 0.95 or 0.99.")
    if original_alpha.ndim != 4 or not original_alpha.is_floating_point():
        raise ValueError("original_alpha must have shape [B,Q,M,H].")
    _require_all(torch.isfinite(original_alpha), "original_alpha must be finite.")
    _require_all(original_alpha >= 0.0, "original_alpha must be nonnegative.")
    batch, queries, sources, heads = original_alpha.shape
    valid = _expand_source_mask(valid_source_mask, batch, queries, sources, "valid_source_mask")
    if valid.device != original_alpha.device:
        raise ValueError("valid_source_mask must share the alpha device.")
    masked_alpha = torch.where(valid[..., None], original_alpha, torch.zeros_like(original_alpha))
    order = torch.argsort(masked_alpha, dim=2, descending=True, stable=True)
    sorted_alpha = masked_alpha.gather(2, order)
    total = sorted_alpha.sum(dim=2, keepdim=True)
    before = sorted_alpha.cumsum(dim=2) - sorted_alpha
    keep_sorted = (total > 0.0) & (before < total * float(retention)) & valid[..., None].expand(
        batch, queries, sources, heads
    ).gather(2, order)
    per_head = torch.zeros_like(keep_sorted)
    per_head.scatter_(2, order, keep_sorted)
    support = per_head.any(dim=-1)
    if protected_mask is not None:
        protected = _expand_source_mask(protected_mask, batch, queries, sources, "protected_mask")
        if protected.device != original_alpha.device:
            raise ValueError("protected_mask must share the alpha device.")
        support = support | (protected & valid)
    return support & valid


def s_attn95_support(
    original_alpha: Tensor, valid_source_mask: Tensor, *, protected_mask: Tensor | None = None
) -> Tensor:
    return s_attn_support(original_alpha, valid_source_mask, retention=0.95, protected_mask=protected_mask)


def s_attn99_support(
    original_alpha: Tensor, valid_source_mask: Tensor, *, protected_mask: Tensor | None = None
) -> Tensor:
    return s_attn_support(original_alpha, valid_source_mask, retention=0.99, protected_mask=protected_mask)


def _target_counts(
    target_support: Tensor | int, batch: int, queries: int, sources: int, device
) -> Tensor:
    if isinstance(target_support, int):
        if target_support < 0:
            raise ValueError("target support count cannot be negative.")
        return torch.full((batch, queries), target_support, dtype=torch.long, device=device)
    if target_support.dtype == torch.bool and tuple(target_support.shape) == (batch, queries, sources):
        if target_support.device != device:
            raise ValueError("target_support must share the valid-source device.")
        return target_support.sum(dim=-1)
    if tuple(target_support.shape) == (batch, queries):
        integral = (
            target_support == target_support.round()
            if target_support.is_floating_point()
            else torch.ones_like(target_support, dtype=torch.bool)
        )
        _require_all(
            torch.isfinite(target_support) & (target_support >= 0) & integral,
            "target support counts must be finite nonnegative integers.",
        )
        return target_support.to(device=device, dtype=torch.long)
    raise ValueError("target_support must be an integer, [B,Q] counts, or boolean [B,Q,M].")


def _ranked_equal_count_support(
    scores: Tensor,
    valid_source_mask: Tensor,
    target_support: Tensor | int,
    *,
    protected_mask: Tensor | None = None,
) -> Tensor:
    if scores.ndim != 3:
        raise ValueError("control scores must have shape [B,Q,M].")
    batch, queries, sources = scores.shape
    valid = _expand_source_mask(valid_source_mask, batch, queries, sources, "valid_source_mask")
    if valid.device != scores.device:
        raise ValueError("valid_source_mask must share the score device.")
    if protected_mask is None:
        protected = torch.zeros_like(valid)
    else:
        protected = _expand_source_mask(protected_mask, batch, queries, sources, "protected_mask")
        if protected.device != scores.device:
            raise ValueError("protected_mask must share the score device.")
        protected = protected & valid
    desired = _target_counts(target_support, batch, queries, sources, scores.device)
    valid_count = valid.sum(dim=-1)
    protected_count = protected.sum(dim=-1)
    desired = torch.minimum(torch.maximum(desired, protected_count), valid_count)
    additional = desired - protected_count
    candidates = valid & ~protected
    ranked_scores = torch.where(candidates, scores, torch.full_like(scores, -torch.inf))
    order = torch.argsort(ranked_scores, dim=-1, descending=True, stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(-1, order, torch.arange(sources, device=scores.device).expand_as(order))
    chosen = candidates & (rank < additional[..., None])
    return protected | chosen


def nearest_support(
    query_coordinates: Tensor,
    source_coordinates: Tensor,
    valid_source_mask: Tensor,
    target_support: Tensor | int,
    *,
    protected_mask: Tensor | None = None,
) -> Tensor:
    """Equal-count nearest-source control, including protected pairs in count."""
    if query_coordinates.ndim != 3 or source_coordinates.ndim != 3:
        raise ValueError("query/source coordinates must have shapes [B,Q,C] and [B,M,C].")
    if (
        query_coordinates.shape[0] != source_coordinates.shape[0]
        or query_coordinates.shape[-1] != source_coordinates.shape[-1]
    ):
        raise ValueError("query/source coordinate batch and coordinate axes must match.")
    delta = source_coordinates[:, None, :, :] - query_coordinates[:, :, None, :]
    scores = -torch.linalg.vector_norm(delta, dim=-1)
    return _ranked_equal_count_support(
        scores, valid_source_mask, target_support, protected_mask=protected_mask
    )


def directional_support(
    query_coordinates: Tensor,
    source_coordinates: Tensor,
    direction: Tensor,
    valid_source_mask: Tensor,
    target_support: Tensor | int,
    *,
    protected_mask: Tensor | None = None,
) -> Tensor:
    """Equal-count control choosing sources furthest along a declared direction.

    ``direction`` may be ``[C]``, ``[B,C]`` or ``[B,Q,C]``. The score is the
    signed projection of ``source - query`` on the normalized direction.
    """
    if query_coordinates.ndim != 3 or source_coordinates.ndim != 3:
        raise ValueError("query/source coordinates must have shapes [B,Q,C] and [B,M,C].")
    batch, queries, coordinate_dim = query_coordinates.shape
    if source_coordinates.shape[0] != batch or source_coordinates.shape[-1] != coordinate_dim:
        raise ValueError("query/source coordinate batch and coordinate axes must match.")
    if direction.ndim == 1 and tuple(direction.shape) == (coordinate_dim,):
        direction = direction.reshape(1, 1, coordinate_dim).expand(batch, queries, coordinate_dim)
    elif direction.ndim == 2 and tuple(direction.shape) == (batch, coordinate_dim):
        direction = direction[:, None, :].expand(batch, queries, coordinate_dim)
    elif tuple(direction.shape) != (batch, queries, coordinate_dim):
        raise ValueError("direction must have shape [C], [B,C], or [B,Q,C].")
    direction = F.normalize(direction, dim=-1, eps=1.0e-12)
    delta = source_coordinates[:, None, :, :] - query_coordinates[:, :, None, :]
    scores = torch.einsum("bqmc,bqc->bqm", delta, direction)
    return _ranked_equal_count_support(
        scores, valid_source_mask, target_support, protected_mask=protected_mask
    )


def shuffle_support(
    valid_source_mask: Tensor,
    target_support: Tensor | int,
    *,
    protected_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
    shuffle_scores: Tensor | None = None,
) -> Tensor:
    """Equal-count source-score-shuffle control; never consumes model outputs."""
    if valid_source_mask.ndim not in (2, 3) or valid_source_mask.dtype != torch.bool:
        raise ValueError("valid_source_mask must be boolean [B,M] or [B,Q,M].")
    if valid_source_mask.ndim == 2:
        batch, sources = valid_source_mask.shape
        if isinstance(target_support, Tensor) and target_support.ndim == 3:
            queries = target_support.shape[1]
        elif isinstance(target_support, Tensor) and target_support.ndim == 2:
            queries = target_support.shape[1]
        elif isinstance(protected_mask, Tensor) and protected_mask.ndim == 3:
            queries = protected_mask.shape[1]
        else:
            queries = 1
        valid = valid_source_mask[:, None, :].expand(batch, queries, sources)
    else:
        batch, queries, sources = valid_source_mask.shape
        valid = valid_source_mask
    if shuffle_scores is None:
        scores = torch.rand(valid.shape, dtype=torch.float32, device=valid.device, generator=generator)
    else:
        if tuple(shuffle_scores.shape) != tuple(valid.shape):
            raise ValueError("shuffle_scores must match the expanded [B,Q,M] valid mask.")
        scores = shuffle_scores
    return _ranked_equal_count_support(
        scores, valid, target_support, protected_mask=protected_mask
    )

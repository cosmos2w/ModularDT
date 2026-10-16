"""Tensor contracts for attention-inherited gathering.

These containers describe correspondence and tensor axes. Dataset-specific
units and native geometry remain owned by the adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import Tensor

TokenKind = Literal["module", "environment"]
NormalizationKind = Literal["softmax", "entmax15", "softmax_entmax15_blend", "diagnostic_intervention"]


@dataclass(frozen=True)
class RouteScoreSnapshot:
    """Last-cycle EM or EE score data plus source/destination correspondence.

    ``query_scaled`` is L2-normalized Q multiplied by the route's live bounded
    temperature; ``key_normalized`` is L2-normalized K. ``raw_logits`` already
    contains their content score and the additive geometric score, but excludes
    the donor log measure. The route that produced these scores is distinct
    from the final token states later used as payloads.
    """

    route: Literal["em", "ee"]
    query_scaled: Tensor  # [B,H,E,D]
    key_normalized: Tensor  # [B,H,N,D]
    geometry_bias: Tensor  # [B,H,E,N]
    raw_logits: Tensor  # [B,H,E,N]
    donor_mask: Tensor  # bool [B,N]
    destination_mask: Tensor  # bool [B,E]
    donor_measure: Tensor  # [B,N], unnormalized native measure
    anchor_ids: Tensor  # integer [B,E]
    donor_ids: Tensor  # integer [B,N]
    score_origin: str

    def __post_init__(self) -> None:
        if self.route not in ("em", "ee"):
            raise ValueError("IAG snapshots are restricted to EM and EE routes.")
        if self.query_scaled.ndim != 4 or self.key_normalized.ndim != 4:
            raise ValueError("Route Q/K tensors must have shapes [B,H,N,D].")
        batch, heads, anchors, head_width = self.query_scaled.shape
        if self.key_normalized.shape[:2] != (batch, heads) or self.key_normalized.shape[-1] != head_width:
            raise ValueError("Route Q/K batch, head and head-width axes must match.")
        donors = self.key_normalized.shape[2]
        expected_scores = (batch, heads, anchors, donors)
        if self.geometry_bias.shape != expected_scores or self.raw_logits.shape != expected_scores:
            raise ValueError("Route score tensors must have shape [B,H,E,N].")
        if self.donor_mask.shape != (batch, donors) or self.donor_mask.dtype != torch.bool:
            raise ValueError("donor_mask must be boolean [B,N].")
        if self.destination_mask.shape != (batch, anchors) or self.destination_mask.dtype != torch.bool:
            raise ValueError("destination_mask must be boolean [B,E].")
        if self.donor_measure.shape != (batch, donors):
            raise ValueError("donor_measure must have shape [B,N].")
        if self.anchor_ids.shape != (batch, anchors) or self.donor_ids.shape != (batch, donors):
            raise ValueError("Route IDs must match the destination/donor axes.")
        if self.anchor_ids.dtype not in (torch.int32, torch.int64) or self.donor_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError("Route IDs must use an integer tensor dtype.")


@dataclass(frozen=True)
class TypedTokenValues:
    """Final-context payload values for one physical token type."""

    kind: TokenKind
    values: Tensor  # [B,H,N,D]
    token_ids: Tensor  # integer [B,N]
    valid: Tensor  # bool [B,N]
    measure: Tensor  # [B,N], unnormalized native measure

    def __post_init__(self) -> None:
        if self.kind not in ("module", "environment"):
            raise ValueError("Token kind must be 'module' or 'environment'.")
        if self.values.ndim != 4:
            raise ValueError("Typed values must have shape [B,H,N,D].")
        batch, _heads, count, _width = self.values.shape
        if self.token_ids.shape != (batch, count) or self.valid.shape != (batch, count):
            raise ValueError("Token IDs and validity must match the value token axis.")
        if self.measure.shape != (batch, count):
            raise ValueError("Token measures must match the value token axis.")
        if self.valid.dtype != torch.bool:
            raise TypeError("Token validity must be boolean.")
        if self.token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("Token IDs must use an integer tensor dtype.")


@dataclass(frozen=True)
class TypedAssociationAtlas:
    """One typed environment-anchor to source-token association atlas."""

    kind: TokenKind
    raw_logits: Tensor  # [B,H,E,N], score excludes log donor measure
    probabilities: Tensor  # [B,H,E,N], zero on invalid entries
    anchor_ids: Tensor  # integer [B,E]
    donor_ids: Tensor  # integer [B,N]
    donor_valid: Tensor  # bool [B,N]
    destination_valid: Tensor  # bool [B,E]
    normalized_measure: Tensor  # [B,N]
    normalization: NormalizationKind

    def __post_init__(self) -> None:
        if self.kind not in ("module", "environment"):
            raise ValueError("Atlas kind must be 'module' or 'environment'.")
        if self.normalization not in ("softmax", "entmax15", "softmax_entmax15_blend", "diagnostic_intervention"):
            raise ValueError("Unsupported atlas normalization.")
        if self.raw_logits.ndim != 4 or self.probabilities.shape != self.raw_logits.shape:
            raise ValueError("Atlas logits and probabilities must both have shape [B,H,E,N].")
        batch, _heads, anchors, donors = self.raw_logits.shape
        if self.anchor_ids.shape != (batch, anchors) or self.donor_ids.shape != (batch, donors):
            raise ValueError("Atlas IDs must match the anchor/donor axes.")
        if self.donor_valid.shape != (batch, donors) or self.donor_valid.dtype != torch.bool:
            raise ValueError("donor_valid must be boolean [B,N].")
        if self.destination_valid.shape != (batch, anchors) or self.destination_valid.dtype != torch.bool:
            raise ValueError("destination_valid must be boolean [B,E].")
        if self.normalized_measure.shape != (batch, donors):
            raise ValueError("normalized_measure must have shape [B,N].")


@dataclass(frozen=True)
class PreparedAnchorValues:
    """Scene-reused typed value aggregates at environment anchors."""

    kind: TokenKind
    values: Tensor  # [B,H,E,D]
    anchor_ids: Tensor  # integer [B,E]
    executed_edge_counts: Tensor | None = None  # integer [B,H,E], positive atlas edges executed
    candidate_edge_counts: Tensor | None = None  # integer [B,H,E], valid donor edges available

    def __post_init__(self) -> None:
        if self.kind not in ("module", "environment"):
            raise ValueError("Prepared value kind must be 'module' or 'environment'.")
        if self.values.ndim != 4 or self.anchor_ids.shape != (
            self.values.shape[0], self.values.shape[2]
        ):
            raise ValueError("Prepared anchor values must be [B,H,E,D] with IDs [B,E].")
        expected_counts = self.values.shape[:3]
        for name, counts in (
            ("executed_edge_counts", self.executed_edge_counts),
            ("candidate_edge_counts", self.candidate_edge_counts),
        ):
            if counts is not None and counts.shape != expected_counts:
                raise ValueError(f"{name} must have shape [B,H,E].")


@dataclass(frozen=True)
class ReceiverCover:
    """Sparse continuous query-to-anchor partition with explicit ID mapping.

    ``anchor_indices`` are execution offsets into the prepared environment
    token order. ``anchor_ids`` preserve native correspondence and are checked
    against those offsets at the public boundary. A cover contains at most
    eight anchor slots per query (4 in 2-D or 8 in 3-D).
    """

    anchor_indices: Tensor  # integer [B,Q,C]
    anchor_ids: Tensor  # integer [B,Q,C]
    weights: Tensor  # [B,Q,C]
    validity: Tensor  # bool [B,Q,C]
    query_ids: Tensor  # integer [B,Q]
    frame_key: Any
    query_mask: Tensor | None = None  # bool [B,Q]
    _tensor_identity_versions: tuple[tuple[str, int, int], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.anchor_indices.ndim != 3:
            raise ValueError("anchor_indices must have shape [B,Q,C].")
        if not 1 <= self.anchor_indices.shape[-1] <= 8:
            raise ValueError("A receiver cover must contain between one and eight slots.")
        if self.anchor_ids.shape != self.anchor_indices.shape or self.weights.shape != self.anchor_indices.shape:
            raise ValueError("Cover IDs and weights must match [B,Q,C].")
        if self.validity.shape != self.anchor_indices.shape or self.validity.dtype != torch.bool:
            raise ValueError("Cover validity must be boolean [B,Q,C].")
        batch, queries, _count = self.anchor_indices.shape
        if self.query_ids.shape != (batch, queries) or self.query_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("query_ids must be integer [B,Q].")
        if self.anchor_indices.dtype not in (torch.int32, torch.int64) or self.anchor_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError("Cover anchor indices and IDs must use integer tensors.")
        if self.query_mask is not None and (
            self.query_mask.shape != (batch, queries) or self.query_mask.dtype != torch.bool
        ):
            raise ValueError("query_mask must be boolean [B,Q].")
        object.__setattr__(self, "_tensor_identity_versions", self._capture_tensor_versions())

    def _capture_tensor_versions(self) -> tuple[tuple[str, int, int], ...]:
        tensors = (
            ("anchor_indices", self.anchor_indices),
            ("anchor_ids", self.anchor_ids),
            ("weights", self.weights),
            ("validity", self.validity),
            ("query_ids", self.query_ids),
            ("query_mask", self.query_mask),
        )
        return tuple(
            (name, id(tensor), -1 if torch.is_inference(tensor) else int(tensor._version))
            for name, tensor in tensors
            if tensor is not None
        )

    @property
    def tensor_versions(self) -> tuple[tuple[str, int], ...]:
        """Captured PyTorch version counters; inference tensors report ``-1``."""
        return tuple((name, version) for name, _identity, version in self._tensor_identity_versions)

    def validate_unchanged(self) -> None:
        """Reject in-place edits after this cover acquired its identity."""
        current = self._capture_tensor_versions()
        if current != self._tensor_identity_versions:
            raise RuntimeError("Receiver-cover tensors changed after cover construction.")

    @property
    def batch_size(self) -> int:
        return int(self.anchor_indices.shape[0])

    @property
    def query_count(self) -> int:
        return int(self.anchor_indices.shape[1])

    @property
    def slots(self) -> int:
        return int(self.anchor_indices.shape[2])

    def validate_partition(self, *, atol: float = 2.0e-6, rtol: float = 2.0e-6) -> None:
        """Validate finite nonnegative weights and unit mass on live queries."""
        self.validate_unchanged()
        if not torch.isfinite(self.weights).all():
            raise ValueError("Receiver-cover weights must be finite.")
        if (self.weights < 0).any():
            raise ValueError("Receiver-cover weights must be nonnegative.")
        effective = self.weights * self.validity.to(self.weights.dtype)
        mass = effective.sum(dim=-1)
        live = (
            torch.ones_like(mass, dtype=torch.bool)
            if self.query_mask is None
            else self.query_mask
        )
        if not torch.allclose(mass[live], torch.ones_like(mass[live]), atol=atol, rtol=rtol):
            raise ValueError("Every live receiver cover must have unit mass.")
        if (mass[~live] != 0).any():
            raise ValueError("Padded queries must have zero receiver-cover mass.")


@dataclass(frozen=True)
class PreparedSourceIndexedResponse:
    """Anchor-space signed response to one supplied heating vector."""

    values: Tensor  # [B,H,E,R]
    anchor_ids: Tensor  # integer [B,E]
    executed_edge_counts: Tensor | None = None  # integer [B,H,E]
    candidate_edge_counts: Tensor | None = None  # integer [B,H,E]

    def __post_init__(self) -> None:
        if self.values.ndim != 4 or self.anchor_ids.shape != (
            self.values.shape[0], self.values.shape[2]
        ):
            raise ValueError("Prepared source response must be [B,H,E,R] with IDs [B,E].")
        expected_counts = self.values.shape[:3]
        for name, counts in (
            ("executed_edge_counts", self.executed_edge_counts),
            ("candidate_edge_counts", self.candidate_edge_counts),
        ):
            if counts is not None and counts.shape != expected_counts:
                raise ValueError(f"{name} must have shape [B,H,E].")


# Delay this package import until the route/token contracts above are defined.
# The historical content_hypergraph package eagerly imports its model, whose
# staged-attention module imports RouteScoreSnapshot from this module.
from dataclasses import fields as dataclass_fields

from ..content_hypergraph.contracts import QueryBatch, SceneBatch, _require_tensor_condition


@dataclass(frozen=True)
class InheritedSceneBatch(SceneBatch):
    """Native scene contract with optional authoritative receiver bounds."""

    receiver_legal_bounds: Tensor | None = None  # [B,2,D], lower then upper

    def __post_init__(self) -> None:
        self._validate_receiver_legal_bounds()

    def _validate_receiver_legal_bounds(self) -> None:
        bounds = self.receiver_legal_bounds
        if bounds is None:
            return
        if not isinstance(bounds, Tensor):
            raise TypeError("receiver_legal_bounds must be a tensor or None.")
        if bounds.ndim != 3 or bounds.shape != (
            self.environment_coordinates.shape[0], 2, self.environment_coordinates.shape[-1]
        ):
            raise ValueError("receiver_legal_bounds must have shape [B,2,D].")
        if bounds.device != self.environment_coordinates.device:
            raise ValueError("receiver_legal_bounds must share the environment coordinate device.")
        if bounds.dtype != self.environment_coordinates.dtype or not bounds.is_floating_point():
            raise TypeError("receiver_legal_bounds must share the floating environment coordinate dtype.")
        _require_tensor_condition(
            torch.isfinite(bounds).all() & (bounds[:, 0] <= bounds[:, 1]).all(),
            "receiver_legal_bounds must be finite and ordered lower-to-upper.",
        )

    def validate(
        self,
        *,
        coordinate_dim: int,
        module_input_dim: int,
        environment_input_dim: int,
        operating_input_dim: int,
    ) -> None:
        super().validate(
            coordinate_dim=coordinate_dim,
            module_input_dim=module_input_dim,
            environment_input_dim=environment_input_dim,
            operating_input_dim=operating_input_dim,
        )
        self._validate_receiver_legal_bounds()

    def tensor_version_stamp(self) -> tuple[tuple[int, int, int], ...]:
        stamp = super().tensor_version_stamp()
        bounds = self.receiver_legal_bounds
        if bounds is None:
            return stamp
        try:
            version = int(bounds._version)
        except RuntimeError:  # Inference tensors do not expose a version counter.
            version = -1
        return (*stamp, (id(bounds), int(bounds.data_ptr()), version))


@dataclass(frozen=True)
class InheritedQueryBatch(QueryBatch):
    """Historical query contract plus the optional IAG receiver cover."""

    receiver_cover: ReceiverCover | None = None


def with_receiver_cover(query: QueryBatch, cover: ReceiverCover) -> InheritedQueryBatch:
    """Wrap an existing query without changing the historical QueryBatch API."""
    cover.validate_unchanged()
    if query.coordinates.shape[:2] != (cover.batch_size, cover.query_count):
        raise ValueError("Receiver cover batch/query axes must match the query batch.")
    if query.query_ids is not None and not torch.equal(query.query_ids, cover.query_ids):
        raise ValueError("Receiver cover query IDs do not match the native query IDs.")
    if query.query_mask is not None and cover.query_mask is not None and not torch.equal(
        query.query_mask, cover.query_mask
    ):
        raise ValueError("Receiver cover validity does not match the native query mask.")
    if query.coordinate_frame_key is not None and query.coordinate_frame_key != cover.frame_key:
        raise ValueError("Receiver cover frame does not match the native query frame.")
    values = {item.name: getattr(query, item.name) for item in dataclass_fields(QueryBatch)}
    if values["query_ids"] is None:
        values["query_ids"] = cover.query_ids
    if values["query_mask"] is None:
        values["query_mask"] = cover.query_mask
    if values["coordinate_frame_key"] is None:
        values["coordinate_frame_key"] = cover.frame_key
    return InheritedQueryBatch(**values, receiver_cover=cover)

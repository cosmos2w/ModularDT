"""Typed containers and runtime protocol for the generic routing index.

The routing index only uses geometry as an ephemeral input to score candidate
paths.  In particular, :class:`RoutingGeometry` does not expose a field value
or a learned physical state to a routing hub.  Keeping these small containers
in a separate package lets the routed field backend share the sparse algebra
with case adapters without importing any case-specific code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class RoutingGeometry(Protocol):
    """Runtime geometry contract consumed by generic routing code.

    ``features`` may accept any leading batch/point dimensions and returns a
    matching tensor of low-dimensional, case-owned descriptors.  ``bounds``
    is represented as ``((lo_0, hi_0), ...)`` and ``length_scale`` contains
    one strictly positive value per coordinate dimension.  The resistance
    method is an optional *finite penalty* in the numerical implementation;
    adapters without a certified resistance may omit it and the generic
    helpers use a neutral zero penalty.
    """

    bounds: Sequence[Sequence[float]] | torch.Tensor
    length_scale: Sequence[float] | torch.Tensor

    def features(self, points: torch.Tensor) -> torch.Tensor:
        """Return adapter-owned descriptors aligned with ``points``."""

    def resistance(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        relation_type: str = "routing",
    ) -> torch.Tensor:
        """Return a finite nonnegative path-resistance penalty."""



@runtime_checkable
class RoutingResistanceField(Protocol):
    """Optional extension for adapters exposing pointwise resistance ``rho``."""

    def resistance_field(
        self,
        points: torch.Tensor,
        relation_type: str = "routing",
    ) -> torch.Tensor:
        """Return pointwise nonnegative resistance values."""


# A descriptive alias makes the intended boundary explicit for callers that
# prefer the word "provider" while retaining the shorter public name used by
# the implementation plan.
RoutingGeometryProvider = RoutingGeometry


@dataclass(frozen=True)
class RoutingCandidates:
    """Candidate routing hubs for one prepared physical state.

    ``descriptors`` and ``propensity`` are routing-only tensors.  They are
    deliberately kept separate from all physical source values; a hub must
    never be passed to the field head as a replacement source state.
    """

    coords: torch.Tensor
    descriptors: torch.Tensor
    propensity: torch.Tensor
    valid: torch.Tensor
    candidate_origin: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.coords.ndim != 3:
            raise ValueError("RoutingCandidates.coords must have shape [B,K,d].")
        if self.descriptors.ndim != 3:
            raise ValueError("RoutingCandidates.descriptors must have shape [B,K,D].")
        if self.propensity.ndim != 2:
            raise ValueError("RoutingCandidates.propensity must have shape [B,K].")
        if self.valid.ndim != 2:
            raise ValueError("RoutingCandidates.valid must have shape [B,K].")
        batch, count = self.coords.shape[:2]
        if tuple(self.descriptors.shape[:2]) != (batch, count):
            raise ValueError("Routing candidate coordinates/descriptors must align on [B,K].")
        if tuple(self.propensity.shape) != (batch, count):
            raise ValueError("Routing candidate propensity must align with [B,K].")
        if tuple(self.valid.shape) != (batch, count):
            raise ValueError("Routing candidate validity must align with [B,K].")
        if self.candidate_origin is not None and tuple(self.candidate_origin.shape) != (batch, count):
            raise ValueError("Routing candidate origin metadata must align with [B,K].")


@dataclass(frozen=True)
class InvertedSourceIncidence:
    """Positive source-to-hub entries used by the two-hop join.

    The integer indices are safe to sort/group.  ``values`` is gathered from
    the live membership tensor and is intentionally not detached, so the
    eventual pair prior remains connected to source routing logits.
    """

    batch_index: torch.Tensor
    source_index: torch.Tensor
    hub_index: torch.Tensor
    values: torch.Tensor
    source_count: int
    hub_count: int
    # Optional fixed ``(batch, hub)`` CSR offsets.  Builders populate this so
    # a prepared source index is sorted once and can be joined repeatedly over
    # receiver chunks.  The default keeps the small container convenient for
    # hand-authored numerical fixtures.
    group_offsets: torch.Tensor | None = None

    def __post_init__(self) -> None:
        rows = int(self.values.numel())
        for name, value in (
            ("batch_index", self.batch_index),
            ("source_index", self.source_index),
            ("hub_index", self.hub_index),
        ):
            if value.ndim != 1 or int(value.numel()) != rows:
                raise ValueError(f"InvertedSourceIncidence.{name} must align with values.")
        if self.source_count < 0 or self.hub_count < 0:
            raise ValueError("Inverted source dimensions must be nonnegative.")
        if self.group_offsets is not None and (
            self.group_offsets.ndim != 1 or int(self.group_offsets.numel()) < 1
        ):
            # The exact batch count is not recoverable from an empty incidence
            # alone, so empty builders may still supply a correctly sized
            # [B*K+1] tensor.  Nonempty offsets only need to be one-dimensional
            # with a sentinel at the end; the join performs the full alignment
            # check against its query tensor.
            raise ValueError("Inverted incidence group_offsets must be a nonempty 1-D tensor.")


@dataclass(frozen=True)
class TypedSourceIncidence:
    """Source measure, ordinary memberships, and their inverted join view."""

    source_weights: torch.Tensor
    membership: torch.Tensor
    hub_measure: torch.Tensor
    inverted: InvertedSourceIncidence

    def __post_init__(self) -> None:
        if self.source_weights.ndim != 2:
            raise ValueError("TypedSourceIncidence.source_weights must have shape [B,N].")
        if self.membership.ndim != 3:
            raise ValueError("TypedSourceIncidence.membership must have shape [B,N,K].")
        if self.hub_measure.ndim != 2:
            raise ValueError("TypedSourceIncidence.hub_measure must have shape [B,K].")
        if tuple(self.membership.shape[:2]) != tuple(self.source_weights.shape):
            raise ValueError("Source weights and membership must align on [B,N].")
        if tuple(self.hub_measure.shape) != tuple(self.membership.shape[:1] + self.membership.shape[2:]):
            raise ValueError("Hub measure must align with membership on [B,K].")

    @property
    def hub_to_source_indices(self) -> InvertedSourceIncidence:
        """Return the prepared positive inverted incidence lists."""

        return self.inverted

    @property
    def membership_values(self) -> torch.Tensor:
        """Return live positive memberships used by the packed join."""

        return self.inverted.values


@dataclass(frozen=True)
class MeasureQueryProjection:
    """Result of source-measure query sparsemax."""

    density: torch.Tensor
    probability: torch.Tensor
    threshold: torch.Tensor
    support: torch.Tensor

    @property
    def alpha(self) -> torch.Tensor:
        """Alias matching the mathematical notation ``alpha``."""

        return self.probability


@dataclass(frozen=True)
class PackedPairs:
    """Unique positive receiver-source pairs and their live two-hop priors."""

    batch_index: torch.Tensor
    receiver_index: torch.Tensor
    source_index: torch.Tensor
    prior: torch.Tensor
    raw_path_count: int
    unique_pair_count: int

    def __post_init__(self) -> None:
        rows = int(self.prior.numel())
        for name, value in (
            ("batch_index", self.batch_index),
            ("receiver_index", self.receiver_index),
            ("source_index", self.source_index),
        ):
            if value.ndim != 1 or int(value.numel()) != rows:
                raise ValueError(f"PackedPairs.{name} must align with prior.")
        if self.raw_path_count < 0 or self.unique_pair_count < 0:
            raise ValueError("Packed pair counts must be nonnegative.")
        if self.unique_pair_count != rows:
            raise ValueError("PackedPairs.unique_pair_count must equal prior length.")
        if self.raw_path_count < self.unique_pair_count:
            raise ValueError("Raw path count cannot be below unique pair count.")

    @property
    def duplicate_expansion(self) -> float:
        """Raw path/unique-pair expansion, useful for the execution ledger."""

        if self.unique_pair_count == 0:
            return 0.0
        return float(self.raw_path_count) / float(self.unique_pair_count)


@dataclass(frozen=True)
class PreparedRoutingIndex:
    """Shared candidate and typed source state for one physical preparation."""

    candidates: RoutingCandidates
    module_incidence: TypedSourceIncidence | None
    environment_incidence: TypedSourceIncidence | None
    diagnostics: dict[str, Any] | None = None

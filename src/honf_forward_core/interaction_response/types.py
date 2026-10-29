"""Typed inputs for baseline-conditioned interaction-response models.

The tensors in :class:`BaselineResponseCache` describe one accepted design.
Trial changes are supplied separately, keyed by physical module ID, and are
read only for the donors named by a response factor.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

EvidenceSource = Literal[
    "reference_solver",
    "stored_reference",
    "surrogate_teacher",
    "analytic_synthetic",
]

VALID_EVIDENCE_SOURCES = frozenset(
    {
        "reference_solver",
        "stored_reference",
        "surrogate_teacher",
        "analytic_synthetic",
    }
)

DEFAULT_OUTPUT_CHANNELS: dict[str, tuple[str, ...]] = {
    "fluid_fields": ("u", "v", "p", "omega", "temperature"),
    "interface": ("T_surface", "q_normal"),
    "solid_temperature": ("temperature",),
}

DEFAULT_QUERY_FEATURE_DIMS: dict[str, int] = {
    # Geometry-only queries by default. In particular, solved T_outside and
    # h fields from the stored interface_condition are not silent inputs.
    "fluid_fields": 2,
    "interface": 3,
    "solid_temperature": 2,
}

ModuleId = Hashable


@dataclass(frozen=True)
class BaselineResponseCache:
    """Immutable-by-contract features for one fixed baseline design.

    ``module_ids`` gives stable physical identities for rows of the three
    module tensors. Padding is represented by ``None`` and is never a valid
    factor donor. A cache can batch repeated responses around the same module
    identities; different anchor families should be batched only when their
    IDs have the same design-variable meaning.
    """

    module_ids: tuple[ModuleId | None, ...]
    module_features: torch.Tensor  # [B, M, F]
    baseline_design: torch.Tensor  # [B, M, D]
    baseline_context: torch.Tensor  # [B, C]

    def __post_init__(self) -> None:
        if self.module_features.ndim != 3:
            raise ValueError("module_features must have shape [B,M,F].")
        if self.baseline_design.ndim != 3:
            raise ValueError("baseline_design must have shape [B,M,D].")
        if self.baseline_context.ndim != 2:
            raise ValueError("baseline_context must have shape [B,C].")
        batch, module_count, _ = self.module_features.shape
        if len(self.module_ids) != module_count:
            raise ValueError("module_ids must contain one stable ID per module row.")
        if self.baseline_design.shape[:2] != (batch, module_count):
            raise ValueError("baseline_design must align with module_features on [B,M].")
        if self.baseline_context.shape[0] != batch:
            raise ValueError("baseline_context must align with module_features on B.")
        if not all(
            torch.is_floating_point(value)
            for value in (self.module_features, self.baseline_design, self.baseline_context)
        ):
            raise ValueError("Baseline response tensors must be floating point.")
        if not (self.module_features.device == self.baseline_design.device == self.baseline_context.device):
            raise ValueError("Baseline response tensors must be on the same device.")
        if not (self.module_features.dtype == self.baseline_design.dtype == self.baseline_context.dtype):
            raise ValueError("Baseline response tensors must use the same floating-point dtype.")
        active_ids = [module_id for module_id in self.module_ids if module_id is not None]
        if len(set(active_ids)) != len(active_ids):
            raise ValueError("Non-padding physical module IDs must be unique within a cache.")

    @property
    def batch_size(self) -> int:
        return int(self.module_features.shape[0])

    @property
    def module_count(self) -> int:
        return int(self.module_features.shape[1])

    def indices_for(self, module_ids: Sequence[ModuleId]) -> tuple[int, ...]:
        """Resolve physical IDs to cache rows, rejecting padding or unknown IDs."""

        row_by_id = {module_id: index for index, module_id in enumerate(self.module_ids) if module_id is not None}
        missing = [module_id for module_id in module_ids if module_id not in row_by_id]
        if missing:
            raise KeyError(f"Physical module IDs are absent from the baseline cache: {missing!r}.")
        return tuple(row_by_id[module_id] for module_id in module_ids)

    def to(self, device: torch.device | str) -> BaselineResponseCache:
        """Move the baseline tensors together while preserving physical IDs."""

        return BaselineResponseCache(
            module_ids=self.module_ids,
            module_features=self.module_features.to(device),
            baseline_design=self.baseline_design.to(device),
            baseline_context=self.baseline_context.to(device),
        )


@dataclass(frozen=True)
class ValidityNeighborhood:
    """Measured local design/context neighborhood for a response factor.

    Step radii are in the same normalized coordinates used by the model.
    Empty radii mean that the neighborhood has not yet been quantified; this
    is an explicit unknown rather than evidence for zero response.
    """

    # A ``None`` coordinate is deliberately unmeasured.  For example, a
    # horizontal stencil on one donor does not establish a vertical radius
    # for that same donor.
    max_abs_delta_by_module: Mapping[ModuleId, tuple[float | None, ...]]
    baseline_design_center_by_module: Mapping[ModuleId, tuple[float, ...]] | None = None
    context_radius: float | None = None
    context_center: tuple[float, ...] = ()
    anchor_family_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for module_id, radii in self.max_abs_delta_by_module.items():
            measured = [value for value in radii if value is not None]
            if not radii or any(not math.isfinite(value) or value <= 0.0 for value in measured):
                raise ValueError(f"Measured validity radii for {module_id!r} must be finite and positive.")
        for module_id, center in (self.baseline_design_center_by_module or {}).items():
            if not center or any(not math.isfinite(value) for value in center):
                raise ValueError(f"Baseline design center for {module_id!r} must contain finite coordinates.")
        if any(not math.isfinite(value) for value in self.context_center):
            raise ValueError("context_center must contain finite coordinates.")
        if self.context_radius is not None and (not math.isfinite(self.context_radius) or self.context_radius <= 0.0):
            raise ValueError("context_radius must be finite and positive when supplied.")


@dataclass(frozen=True)
class ReceiverSupport:
    """Baseline-fixed receiver support, indexed independently of query order.

    ``all_receivers=True`` is the explicit unlocalized control. Otherwise a
    factor is evaluated only at the declared stable query IDs and/or physical
    receiver module IDs for each output role.
    """

    all_receivers: bool = True
    receiver_query_ids_by_role: Mapping[str, tuple[Hashable, ...]] | None = None
    receiver_module_ids_by_role: Mapping[str, tuple[ModuleId, ...]] | None = None

    def __post_init__(self) -> None:
        query_map = self.receiver_query_ids_by_role or {}
        module_map = self.receiver_module_ids_by_role or {}
        if self.all_receivers and (query_map or module_map):
            raise ValueError("All-receiver support cannot also declare a restricted receiver set.")
        if not self.all_receivers and not (query_map or module_map):
            raise ValueError("Restricted receiver support must name query IDs or physical module IDs.")
        for role, ids in (*query_map.items(), *module_map.items()):
            if len(set(ids)) != len(ids):
                raise ValueError(f"Receiver IDs for role {role!r} must be unique.")


@dataclass(frozen=True)
class ResponseFactor:
    """A directed response factor indexed by physical donor IDs and roles."""

    factor_id: str
    donor_ids: tuple[ModuleId, ...]
    output_roles: tuple[str, ...]
    validity: ValidityNeighborhood
    receiver_support: ReceiverSupport = ReceiverSupport()
    evidence_sources: tuple[EvidenceSource, ...] = ()
    evidence_anchor_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.factor_id:
            raise ValueError("factor_id must be nonempty.")
        if len(self.donor_ids) not in (1, 2):
            raise ValueError("The initial anchored response operator supports unary and pair factors only.")
        if len(set(self.donor_ids)) != len(self.donor_ids):
            raise ValueError("A factor cannot contain the same physical donor more than once.")
        if not self.output_roles or len(set(self.output_roles)) != len(self.output_roles):
            raise ValueError("A factor must declare one or more unique receiver/output roles.")
        unknown = set(self.evidence_sources) - VALID_EVIDENCE_SOURCES
        if unknown:
            raise ValueError(f"Unknown evidence source labels: {sorted(unknown)!r}.")


@dataclass(frozen=True)
class ResponseQueryBatch:
    """Fixed baseline or material-coordinate queries for one output role."""

    features: torch.Tensor  # [B, Q, Fq]
    receiver_module_ids: tuple[ModuleId | None, ...] | None = None
    query_ids: tuple[Hashable, ...] | None = None
    mask: torch.Tensor | None = None  # [B, Q], carried for loss/evaluation
    quadrature_weights: torch.Tensor | None = None  # [B, Q], carried for loss/evaluation

    def __post_init__(self) -> None:
        if self.features.ndim != 3:
            raise ValueError("Response query features must have shape [B,Q,Fq].")
        batch, query_count, _ = self.features.shape
        if self.receiver_module_ids is not None and len(self.receiver_module_ids) != query_count:
            raise ValueError("receiver_module_ids must align with the query dimension.")
        if self.query_ids is not None and len(self.query_ids) != query_count:
            raise ValueError("query_ids must align with the query dimension.")
        if self.query_ids is not None and len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError("Stable query IDs must be unique within a query batch.")
        for name, value in (("mask", self.mask), ("quadrature_weights", self.quadrature_weights)):
            if value is not None and value.shape != (batch, query_count):
                raise ValueError(f"{name} must have shape [B,Q].")


ResponseQueries = Mapping[str, ResponseQueryBatch]


def make_baseline_cache(
    module_ids: Sequence[ModuleId | None],
    module_features: torch.Tensor,
    baseline_design: torch.Tensor,
    baseline_context: torch.Tensor,
) -> BaselineResponseCache:
    """Create an owned baseline cache without detaching training gradients."""

    return BaselineResponseCache(
        module_ids=tuple(module_ids),
        module_features=module_features.clone(),
        baseline_design=baseline_design.clone(),
        baseline_context=baseline_context.clone(),
    )

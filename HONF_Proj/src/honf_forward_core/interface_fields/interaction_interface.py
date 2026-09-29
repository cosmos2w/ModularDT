"""Minimal frozen forward-organizer interface for conditional reuse."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import torch

from .adaptive_interaction_cover import (
    INTERACTION_MECHANISMS,
    InteractionPermissionKey,
    MechanismPlan,
)
from .input_cover_organizer import OrganizerScores
from .types import EncodedInterfaceCase


def _snapshot(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach().clone()


def _check_ids(
    name: str,
    values: Sequence[Hashable] | None,
    expected: int,
) -> tuple[Hashable, ...] | None:
    if values is None:
        return None
    result = tuple(values)
    if len(result) != expected:
        raise ValueError(f"{name} must contain one identity per physical row")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate physical identities")
    return result


@dataclass(frozen=True)
class InteractionInterface:
    """Detached candidate-built incidence used by downstream inverse models.

    The source rows retain native physical order. Receiver anchors describe
    the model-side geometry used to build the tree; `receiver_source_weights`
    evaluates that same hard tree at a newly supplied candidate sensor panel.
    Model weights remain external and frozen by the caller.
    """

    module_descriptors: Mapping[str, torch.Tensor]
    environment_descriptors: Mapping[str, torch.Tensor]
    receiver_coordinates: torch.Tensor
    receiver_weights: torch.Tensor
    receiver_roles: torch.Tensor
    receiver_access: torch.Tensor  # [A,N]
    typed_permissions: Mapping[str, torch.Tensor]  # canonical key -> [N,S]
    permission_statuses: Mapping[str, str]
    frontier_ids: tuple[int, ...]
    module_embeddings: torch.Tensor | None
    environment_embeddings: torch.Tensor | None
    node_embeddings: torch.Tensor | None
    numerical_state_version: str
    evidence_scope: str
    module_source_ids: tuple[Hashable, ...] | None = None
    environment_source_ids: tuple[Hashable, ...] | None = None
    receiver_physical_ids: tuple[Hashable, ...] | None = None
    tree: object = field(default=None, repr=False, compare=False)
    split_gates: torch.Tensor = field(default_factory=lambda: torch.empty(0), repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.numerical_state_version.strip():
            raise ValueError("numerical_state_version cannot be empty")
        if not self.evidence_scope.strip():
            raise ValueError("evidence_scope cannot be empty")
        if self.receiver_coordinates.ndim != 2:
            raise ValueError("receiver coordinates must have shape [A,d]")
        if self.receiver_access.ndim != 2 or self.receiver_access.shape[0] != self.receiver_coordinates.shape[0]:
            raise ValueError("receiver access must have shape [A,N]")
        if self.receiver_weights.shape != (self.receiver_coordinates.shape[0],):
            raise ValueError("receiver weights must align with physical anchors")
        if self.receiver_roles.shape != (self.receiver_coordinates.shape[0],):
            raise ValueError("receiver roles must align with physical anchors")
        if self.receiver_roles.dtype not in (torch.int32, torch.int64):
            raise TypeError("receiver roles must have integer dtype")
        if bool((self.receiver_access < 0).any()) or not bool(torch.isfinite(self.receiver_access).all()):
            raise ValueError("receiver access must be finite and nonnegative")
        module_valid = self.module_descriptors.get("valid")
        module_coordinates = self.module_descriptors.get("coordinates")
        environment_weights = self.environment_descriptors.get("weights")
        environment_coordinates = self.environment_descriptors.get("coordinates")
        if module_valid is None or module_coordinates is None:
            raise ValueError("module descriptors must expose coordinates and validity")
        if environment_weights is None or environment_coordinates is None:
            raise ValueError("environment descriptors must expose coordinates and weights")
        if module_valid.ndim != 1 or module_coordinates.ndim != 2 or module_coordinates.shape[0] != module_valid.shape[0]:
            raise ValueError("module source descriptors must share their physical source axis")
        if environment_weights.ndim != 1 or environment_coordinates.ndim != 2 or environment_coordinates.shape[0] != environment_weights.shape[0]:
            raise ValueError("environment source descriptors must share their physical source axis")
        if self.module_source_ids is not None and len(self.module_source_ids) != module_valid.numel():
            raise ValueError("module physical IDs do not align with module descriptors")
        if self.environment_source_ids is not None and len(self.environment_source_ids) != environment_weights.numel():
            raise ValueError("environment physical IDs do not align with environment descriptors")
        if self.receiver_physical_ids is not None and len(self.receiver_physical_ids) != self.receiver_coordinates.shape[0]:
            raise ValueError("receiver physical IDs do not align with receiver anchors")

    @property
    def module_validity(self) -> torch.Tensor:
        return self.module_descriptors["valid"]

    @property
    def environment_validity(self) -> torch.Tensor:
        return self.environment_descriptors["weights"] > 0.0

    @property
    def explicit_bypass_routes(self) -> tuple[str, ...]:
        return tuple(
            key for key, status in sorted(self.permission_statuses.items())
            if status == "full_access_bypass_missing_key"
        )

    def permission_matrix(self, mechanism: str, *, phase: str | None = None) -> torch.Tensor:
        key = InteractionPermissionKey(mechanism, phase)
        if phase is not None and key.canonical_name in self.typed_permissions:
            return self.typed_permissions[key.canonical_name]
        base = InteractionPermissionKey(mechanism).canonical_name
        try:
            return self.typed_permissions[base]
        except KeyError as exc:
            raise ValueError(f"unknown typed permission {key.canonical_name!r}") from exc

    def permission_status(self, mechanism: str, *, phase: str | None = None) -> str:
        key = InteractionPermissionKey(mechanism, phase)
        if key.canonical_name in self.permission_statuses:
            return self.permission_statuses[key.canonical_name]
        base = InteractionPermissionKey(mechanism).canonical_name
        try:
            return self.permission_statuses[base]
        except KeyError as exc:
            raise ValueError(f"unknown typed permission {key.canonical_name!r}") from exc

    def receiver_source_weights(
        self,
        mechanism: str,
        receiver_coordinates: torch.Tensor,
        *,
        phase: str | None = None,
        receiver_validity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate hard physical receiver/source weights at current inputs."""

        tree = self.tree
        if tree is None:
            raise RuntimeError("interface did not retain its candidate-built receiver tree")
        coordinates = receiver_coordinates.to(
            device=self.split_gates.device,
            dtype=self.split_gates.dtype,
        ).detach()
        if coordinates.ndim != 2 or coordinates.shape[1] != self.receiver_coordinates.shape[1]:
            raise ValueError("receiver coordinates must match the native spatial dimension")
        access = tree.access(coordinates, self.split_gates).detach()
        membership = self.permission_matrix(mechanism, phase=phase)
        if not bool(((membership == 0.0) | (membership == 1.0)).all()):
            raise ValueError("frozen inverse interfaces require hard binary permissions")
        result = access @ membership
        if receiver_validity is not None:
            if receiver_validity.shape != (coordinates.shape[0],):
                raise ValueError("receiver validity must align with supplied coordinates")
            result = result * (receiver_validity.to(device=result.device) > 0.5).to(result.dtype)[:, None]
        return result.detach()

    def module_pair_weights(self, *, phase: str | None = None) -> torch.Tensor:
        """Return hard MM access in physical module order, with invalid/self pairs removed."""

        coordinates = self.module_descriptors["coordinates"]
        result = self.receiver_source_weights("MM", coordinates, phase=phase)
        valid = self.module_validity > 0.5
        result = result * valid.to(result.dtype)[:, None] * valid.to(result.dtype)[None, :]
        if result.shape[0] == result.shape[1]:
            result = result.masked_fill(torch.eye(result.shape[0], device=result.device, dtype=torch.bool), 0.0)
        return result.detach()

    def route_is_full(self, mechanism: str, *, phase: str | None = None) -> bool:
        """Report a route that carries all eligible sources on the hard frontier."""

        matrix = self.permission_matrix(mechanism, phase=phase)
        rows = matrix[list(self.frontier_ids)]
        if mechanism.upper() in {"MM", "EM", "QM"}:
            expected = self.module_validity.to(matrix.dtype)
        else:
            expected = self.environment_validity.to(matrix.dtype)
        return bool(torch.equal(rows, expected[None, :].expand_as(rows)))


def interaction_interface_from_plan(
    plan: MechanismPlan,
    encoded: EncodedInterfaceCase,
    scores: OrganizerScores,
    *,
    numerical_state_version: str,
    evidence_scope: str = "forward-trained",
    case_index: int = 0,
    module_source_ids: Sequence[Hashable] | None = None,
    environment_source_ids: Sequence[Hashable] | None = None,
    receiver_physical_ids: Sequence[Hashable] | None = None,
) -> InteractionInterface:
    """Export a detached, hard candidate graph with source-order metadata."""

    if plan.direct_access or plan.direct_policies:
        raise ValueError(
            "a direct-pair control cannot be exported as a grouped frozen interface; "
            "freeze the selected forward organizer plan instead"
        )

    batch_size = int(encoded.module_present.shape[0])
    if case_index < 0 or case_index >= batch_size:
        raise IndexError("case_index is outside the encoded batch")
    if int(plan.module_present.numel()) != int(encoded.module_present.shape[1]):
        raise ValueError("plan module source axis does not match encoded native order")
    if int(plan.environment_count) != int(encoded.env_coords.shape[1]):
        raise ValueError("plan environment source axis does not match encoded native order")
    expected_present = encoded.module_present[case_index].to(
        device=plan.module_present.device, dtype=plan.module_present.dtype
    )
    if not torch.equal(plan.module_present, expected_present):
        raise ValueError("plan module validity does not match the encoded candidate design")
    if not bool(((plan.split_gates == 0.0) | (plan.split_gates == 1.0)).all()):
        raise ValueError("frozen inverse interfaces require hard binary frontier gates")

    frontier: list[int] = []
    pending = [0]
    while pending:
        node_index = pending.pop()
        node = plan.tree.nodes[node_index]
        if node.left is None or float(plan.split_gates[node_index]) == 0.0:
            frontier.append(node_index)
        else:
            assert node.right is not None
            pending.extend((node.right, node.left))
    frontier_ids = tuple(frontier)
    if not frontier_ids:
        raise ValueError("hard plan has no active receiver frontier")

    module_features = encoded.module_features[case_index]
    environment_features = None if encoded.env_features is None else encoded.env_features[case_index]
    module_valid = (encoded.module_present[case_index] > 0.5).to(encoded.module_present.dtype)
    module_descriptors = MappingProxyType({
        "coordinates": _snapshot(encoded.module_centers[case_index]),
        "features": _snapshot(module_features),
        "state": _snapshot(encoded.module_tokens[case_index]),
        "valid": _snapshot(module_valid),
    })
    environment_descriptors = MappingProxyType({
        "coordinates": _snapshot(encoded.env_coords[case_index]),
        "features": _snapshot(environment_features),
        "state": _snapshot(encoded.env_tokens[case_index]),
        "weights": _snapshot(encoded.env_weights[case_index]),
    })

    phases = sorted({key.phase for key in plan.permissions if key.phase is not None})
    contexts = [(mechanism, None) for mechanism in INTERACTION_MECHANISMS]
    contexts.extend(
        (mechanism, phase)
        for phase in phases
        for mechanism in INTERACTION_MECHANISMS
    )
    permissions: dict[str, torch.Tensor] = {}
    statuses: dict[str, str] = {}
    for mechanism, phase in contexts:
        key = InteractionPermissionKey(mechanism, phase).canonical_name
        matrix = plan.permission_matrix(mechanism, phase=phase).detach()
        if not bool(((matrix == 0.0) | (matrix == 1.0)).all()):
            raise ValueError("frozen inverse interfaces require hard binary permissions")
        permissions[key] = matrix.clone()
        statuses[key] = plan.permission_status(mechanism, phase=phase)

    anchor_coordinates = _snapshot(plan.tree.universe.coordinates)
    assert anchor_coordinates is not None
    access = plan.tree.access(anchor_coordinates, plan.split_gates.detach()).detach()
    module_ids = _check_ids("module_source_ids", module_source_ids, int(module_valid.numel()))
    environment_ids = _check_ids(
        "environment_source_ids", environment_source_ids, int(encoded.env_coords.shape[1])
    )
    receiver_ids = _check_ids(
        "receiver_physical_ids", receiver_physical_ids, int(anchor_coordinates.shape[0])
    )
    return InteractionInterface(
        module_descriptors=module_descriptors,
        environment_descriptors=environment_descriptors,
        receiver_coordinates=anchor_coordinates,
        receiver_weights=_snapshot(plan.tree.universe.weights),
        receiver_roles=_snapshot(plan.tree.universe.roles),
        receiver_access=_snapshot(access),
        typed_permissions=MappingProxyType(permissions),
        permission_statuses=MappingProxyType(statuses),
        frontier_ids=frontier_ids,
        module_embeddings=_snapshot(scores.module_embeddings),
        environment_embeddings=_snapshot(scores.environment_embeddings),
        node_embeddings=_snapshot(scores.node_embeddings),
        numerical_state_version=str(numerical_state_version),
        evidence_scope=str(evidence_scope),
        module_source_ids=module_ids,
        environment_source_ids=environment_ids,
        receiver_physical_ids=receiver_ids,
        tree=plan.tree,
        split_gates=_snapshot(plan.split_gates),
    )


__all__ = ["InteractionInterface", "interaction_interface_from_plan"]

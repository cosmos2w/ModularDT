"""Typed provenance and validity records for a frozen native interface step.

These records describe model transport and finite receiver responses. They do
not place reference outputs inside a forward organizer or treat a grouping as
a physical interaction claim.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from torch import nn

from channelthermal.response_control.contracts import DesignInput, RoleQuery
from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    INTERACTION_MECHANISMS,
    MechanismPlan,
)
from honf_inverse_core.contracts import NamedContext, PhysicalDesign


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def role_schema_hash(queries: Mapping[str, RoleQuery]) -> str:
    """Hash target-free role/coordinate conventions, excluding target values."""

    return _digest({
        name: {
            "role": query.role,
            "channels": query.channel_names,
            "units": query.channel_units,
            "coordinate_kind": query.coordinate_kind,
            "feature_shape": tuple(query.query_features.shape),
            "receiver_slots": query.receiver_slots,
        }
        for name, query in sorted(queries.items())
    })


def role_support_hash(queries: Mapping[str, RoleQuery]) -> str:
    """Bind a local plan to the exact target-free receiver coordinates."""

    digest = hashlib.sha256()
    digest.update(role_schema_hash(queries).encode("ascii"))
    for name, query in sorted(queries.items()):
        features = query.query_features.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(features.dtype).encode("ascii"))
        digest.update(features.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class InterfacePacket:
    mechanism: str
    source_physical_ids: tuple[Hashable, ...]
    receiver_support_id: str
    permission_parameters: Mapping[str, Any]
    evidence_scope: str
    phase: str | None = None

    def __post_init__(self) -> None:
        if self.mechanism not in {"MM", "ME", "EM", "QM", "QE"}:
            raise ValueError("Interface mechanism must identify one native message or read.")
        if not self.receiver_support_id:
            raise ValueError("Receiver support needs an explicit role and coordinate convention.")
        if self.evidence_scope not in {"model_transport", "teacher_response", "local_reference"}:
            raise ValueError("Interface evidence scope must be explicit.")
        if len(set(self.source_physical_ids)) != len(self.source_physical_ids):
            raise ValueError("One packet must list each physical source only once.")


def packets_from_cover(
    plan: MechanismPlan,
    module_ids_by_slot: Sequence[Hashable | None],
    *,
    phase: str | None,
) -> tuple[InterfacePacket, ...]:
    """Export active model packets with physical module and quadrature supports.

    Environment coordinates identify support locations, while local column
    indices retain distinct quadrature atoms at duplicate locations. Neither
    the support inventory nor its model permission is physical causal evidence.
    """

    source_slots = int(plan.module_present.numel())
    if len(module_ids_by_slot) > source_slots:
        raise ValueError("Module physical IDs exceed the cover source slots.")
    # The native core pads the physical design to checkpoint capacity. Padded
    # slots have no physical identity and may never become permitted sources.
    module_ids_by_slot = tuple(module_ids_by_slot) + (None,) * (source_slots - len(module_ids_by_slot))
    present = tuple(bool(value) for value in (plan.module_present > 0.5).detach().cpu().tolist())
    if any(active != (module_id is not None) for active, module_id in zip(
        present, module_ids_by_slot, strict=True
    )):
        raise ValueError("Cover physical IDs must match active module sources.")
    if len({module_id for module_id in module_ids_by_slot if module_id is not None}) != sum(present):
        raise ValueError("Physical module source IDs must be unique.")
    universe = plan.tree.universe
    coordinates = universe.coordinates.detach().cpu()
    roles = universe.roles.detach().cpu().tolist()
    environmental = coordinates[universe.roles.detach().cpu() == 0]
    if len(environmental) != plan.environment_count:
        raise ValueError("Cover receiver universe does not identify every environment support.")
    active_nodes = plan.active_node_mask().detach().cpu().tolist()
    plan_hash = plan.canonical_hash()
    packets: list[InterfacePacket] = []
    for mechanism in INTERACTION_MECHANISMS:
        permissions = plan.permission_matrix(mechanism, phase=phase).detach().cpu()
        for node_index, active in enumerate(active_nodes):
            if not active:
                continue
            selected = [index for index, value in enumerate(permissions[node_index]) if float(value) > 0.0]
            if not selected:
                continue
            supports = tuple(sorted({
                (int(roles[index]), *tuple(float(value) for value in coordinates[index].tolist()))
                for index in plan.tree.nodes[node_index].anchor_indices
            }))
            receiver_id = "receiver_support:" + _digest({"anchors": supports})
            if mechanism in {"MM", "EM", "QM"}:
                source_ids: tuple[Hashable, ...] = tuple(
                    ("module", module_ids_by_slot[index]) for index in selected
                )
            else:
                source_ids = tuple(dict.fromkeys(
                    ("environment_support_xy", *tuple(float(value) for value in environmental[index].tolist()))
                    for index in selected
                ))
            packets.append(InterfacePacket(
                mechanism=mechanism,
                source_physical_ids=source_ids,
                receiver_support_id=receiver_id,
                permission_parameters={
                    "plan_hash": plan_hash,
                    "node_index": node_index,
                    "source_column_indices": tuple(selected),
                    "source_permission_weights": tuple(float(permissions[node_index, index]) for index in selected),
                    "environment_support_is_quadrature": mechanism in {"ME", "QE"},
                },
                evidence_scope="model_transport",
                phase=phase,
            ))
    return tuple(packets)


class AllAccessNativeCoverPolicy(nn.Module):
    """Explicit native full-access control that constructs the case-local tree."""

    def plan_cases(self, encoded: Any, prepared_state: Any, trees: Sequence[Any]) -> tuple[MechanismPlan, ...]:
        del prepared_state
        if len(trees) != int(encoded.module_present.shape[0]):
            raise ValueError("Native all-access trees must align with the encoded batch.")
        environment_count = int(encoded.env_coords.shape[1])
        return tuple(
            MechanismPlan.full_access(tree, encoded.module_present[index], environment_count)
            for index, tree in enumerate(trees)
        )


@dataclass(frozen=True)
class LocalInterfacePlan:
    """One frozen combinatorial plan with live trial-state validity checks."""

    anchor_input_hash: str
    checkpoint_hash: str
    cover_plans: tuple[Any, ...]
    cover_plan_hashes: tuple[str, ...]
    anchor_positions: tuple[tuple[float, float], ...]
    anchor_heating: tuple[float, ...]
    module_present: tuple[int, ...]
    module_ids_by_slot: tuple[Hashable | None, ...]
    module_family_id: str
    context_hash: str
    role_schema_hash: str
    role_support_hash: str
    max_position_delta: float
    allow_heat_changes: bool
    declared_global_paths: tuple[str, ...]
    validity_schema: str = "native_receiver_local_v1"

    @classmethod
    def from_anchor(
        cls,
        *,
        baseline: PhysicalDesign,
        module_ids_by_slot: Sequence[Hashable | None],
        context: NamedContext,
        role_queries: Mapping[str, RoleQuery],
        checkpoint_hash: str,
        cover_plans: Sequence[Any],
        max_position_delta: float,
        allow_heat_changes: bool = False,
        declared_global_paths: Sequence[str] = (
            "checkpoint_native_encoder",
            "checkpoint_native_coarse",
            "checkpoint_native_local",
            "checkpoint_native_local_surrogate",
            "checkpoint_native_port_refinement",
        ),
    ) -> LocalInterfacePlan:
        if not checkpoint_hash or len(checkpoint_hash) != 64:
            raise ValueError("A checkpoint SHA256 is required for a frozen native plan.")
        if not np.isfinite(max_position_delta) or max_position_delta < 0.0:
            raise ValueError("The frozen-plan trust radius must be nonnegative and finite.")
        plans = tuple(cover_plans)
        if len(plans) != 1 or not isinstance(plans[0], MechanismPlan):
            raise ValueError("The one-case native call requires one typed mechanism cover plan.")
        module_ids = tuple(module_ids_by_slot)
        present = tuple(int(value) for value in baseline.module_present)
        if len(module_ids) != len(present) or any(
            bool(active) != (module_id is not None)
            for active, module_id in zip(present, module_ids, strict=True)
        ):
            raise ValueError("Frozen cover physical module IDs must align with active slots.")
        if len({module_id for module_id in module_ids if module_id is not None}) != sum(present):
            raise ValueError("Frozen cover physical module IDs must be unique.")
        plan_present = tuple(int(value) for value in (plans[0].module_present > 0.5).detach().cpu().tolist())
        if plan_present[:len(present)] != present or any(plan_present[len(present):]):
            raise ValueError("Frozen cover module source catalogue differs from the anchor design.")
        return cls(
            anchor_input_hash=_digest({"design": baseline.to_dict(), "context": context.to_dict()}),
            checkpoint_hash=checkpoint_hash,
            cover_plans=plans,
            cover_plan_hashes=tuple(plan.canonical_hash() for plan in plans),
            anchor_positions=tuple(tuple(map(float, pair)) for pair in baseline.module_centers),
            anchor_heating=tuple(map(float, baseline.heat_powers)),
            module_present=present,
            module_ids_by_slot=module_ids,
            module_family_id=baseline.module_family_id,
            context_hash=_digest(context.to_dict()),
            role_schema_hash=role_schema_hash(role_queries),
            role_support_hash=role_support_hash(role_queries),
            max_position_delta=float(max_position_delta),
            allow_heat_changes=bool(allow_heat_changes),
            declared_global_paths=tuple(declared_global_paths),
        )

    def validate_trial(
        self,
        design: PhysicalDesign | DesignInput,
        context: NamedContext,
        role_queries: Mapping[str, RoleQuery],
        *,
        checkpoint_hash: str,
        module_ids_by_slot: Sequence[Hashable | None],
        module_family_id: str | None = None,
    ) -> None:
        if checkpoint_hash != self.checkpoint_hash:
            raise ValueError("Frozen cover checkpoint identity changed.")
        if _digest(context.to_dict()) != self.context_hash:
            raise ValueError("Frozen cover operating context changed.")
        if role_schema_hash(role_queries) != self.role_schema_hash:
            raise ValueError("Frozen cover receiver-role schema changed.")
        if role_support_hash(role_queries) != self.role_support_hash:
            raise ValueError("Frozen cover receiver coordinates changed.")
        if tuple(module_ids_by_slot) != self.module_ids_by_slot:
            raise ValueError("Frozen cover physical module-to-slot identities changed.")
        if isinstance(design, PhysicalDesign):
            if design.module_family_id != self.module_family_id:
                raise ValueError("Frozen cover module family changed.")
            positions = np.asarray(design.module_centers)
            heating = np.asarray(design.heat_powers)
            present = np.asarray(design.module_present)
        else:
            if module_family_id != self.module_family_id:
                raise ValueError("Frozen cover module family changed.")
            positions = design.module_positions.detach().cpu().numpy()
            heating = design.module_heating.detach().cpu().numpy()
            present = design.module_present.detach().cpu().numpy()
        if positions.shape != (len(self.module_present), 2):
            raise ValueError("Frozen cover module slots changed.")
        if tuple(int(value) for value in present) != self.module_present:
            raise ValueError("Frozen cover module presence changed.")
        if not self.allow_heat_changes and not np.array_equal(heating, np.asarray(self.anchor_heating)):
            raise ValueError("Frozen cover heating changed outside its declared validity schema.")
        if np.max(np.abs(positions - np.asarray(self.anchor_positions))) > self.max_position_delta + 1e-7:
            raise ValueError("Trial left the frozen combinatorial trust region.")
        if tuple(plan.canonical_hash() for plan in self.cover_plans) != self.cover_plan_hashes:
            raise ValueError("A frozen cover plan was mutated after anchoring.")

    def metadata(self) -> dict[str, Any]:
        return {
            "anchor_input_hash": self.anchor_input_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "cover_plan_hashes": self.cover_plan_hashes,
            "context_hash": self.context_hash,
            "role_schema_hash": self.role_schema_hash,
            "role_support_hash": self.role_support_hash,
            "module_ids_by_slot": tuple(None if value is None else str(value) for value in self.module_ids_by_slot),
            "max_position_delta": self.max_position_delta,
            "allow_heat_changes": self.allow_heat_changes,
            "declared_global_paths": self.declared_global_paths,
            "validity_schema": self.validity_schema,
        }

    def packet_inventory(
        self,
        module_ids_by_slot: Sequence[Hashable | None],
        *,
        phases: Sequence[str | None] = ("P0", "P1", "P2"),
    ) -> tuple[InterfacePacket, ...]:
        if len(self.cover_plans) != 1 or not isinstance(self.cover_plans[0], MechanismPlan):
            raise TypeError("Physical packet export requires one typed native cover plan.")
        if tuple(module_ids_by_slot) != self.module_ids_by_slot:
            raise ValueError("Packet export physical IDs differ from the frozen cover anchor.")
        return tuple(
            packet
            for phase in phases
            for packet in packets_from_cover(self.cover_plans[0], module_ids_by_slot, phase=phase)
        )


@dataclass(frozen=True)
class ResponseAssessment:
    receiver_quantity: str
    changed_coordinates: tuple[str, ...]
    predicted_increment: float
    numerical_discrepancy: float | None
    model_error_indicator: float | None
    evidence_scope: str
    supported_move_scale: float | None


@dataclass(frozen=True)
class DecisionGroup:
    changed_coordinates: tuple[str, ...]
    protected_receivers: tuple[str, ...]
    protected_constraints: tuple[str, ...]
    rationale: str
    supporting_response_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.rationale not in {
            "shared_receiver", "constraint_coordination", "resolved_mixed_response",
            "direct_sensitivity_control",
        }:
            raise ValueError("A decision group requires a supported rationale.")
        if not self.changed_coordinates or len(set(self.changed_coordinates)) != len(self.changed_coordinates):
            raise ValueError("Decision coordinates must be nonempty and unique.")


__all__ = [
    "AllAccessNativeCoverPolicy", "DecisionGroup", "InterfacePacket", "LocalInterfacePlan",
    "ResponseAssessment", "packets_from_cover", "role_schema_hash", "role_support_hash",
]

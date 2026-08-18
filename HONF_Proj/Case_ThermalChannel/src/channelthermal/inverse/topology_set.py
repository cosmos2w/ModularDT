"""ThermalChannel set targets derived from forward topology signatures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from honf_forward_core.evaluation.topology_signature import validate_topology_signature
from honf_inverse_core.contracts import NamedContext, PhysicalDesign, finite_array, jsonable

from .compact_plan import COMPACT_PLAN_FEATURE_NAMES, normalize_compact_plan


TOPOLOGY_SET_SCHEMA_NAME = "thermalchannel_topology_set"
TOPOLOGY_SET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TopologySetPlan:
    """One unordered padded topology set with signature-bound provenance."""

    raw: np.ndarray
    normalized: np.ndarray
    active_mask: np.ndarray
    relations: np.ndarray
    relation_feature_names: tuple[str, ...]
    forward_checkpoint_sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw = finite_array(self.raw, name="topology set raw")
        normalized = finite_array(self.normalized, name="topology set normalized")
        active = finite_array(self.active_mask, name="topology set active mask").reshape(-1)
        relations = finite_array(self.relations, name="topology set relations")
        if raw.ndim != 2 or raw.shape != normalized.shape or raw.shape[1] != 12:
            raise ValueError("Topology set raw/normalized tokens must share shape [K,12].")
        if active.shape != (raw.shape[0],) or not np.isin(active, [0.0, 1.0]).all():
            raise ValueError("Topology set active_mask must be binary with K entries.")
        if not np.array_equal(raw[:, 0] > 0.5, active > 0.5):
            raise ValueError("Topology set token activity and active_mask disagree.")
        names = tuple(str(value) for value in self.relation_feature_names)
        if relations.shape != (raw.shape[0], raw.shape[0], len(names)):
            raise ValueError("Topology set relations must have shape [K,K,F_r].")
        checkpoint = str(self.forward_checkpoint_sha256)
        if len(checkpoint) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in checkpoint
        ):
            raise ValueError("Topology set requires a forward checkpoint SHA-256.")
        object.__setattr__(self, "raw", raw.astype(np.float32))
        object.__setattr__(self, "normalized", normalized.astype(np.float32))
        object.__setattr__(self, "active_mask", active.astype(np.float32))
        object.__setattr__(self, "relations", relations.astype(np.float32))
        object.__setattr__(self, "relation_feature_names", names)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def edge_capacity(self) -> int:
        return int(self.raw.shape[0])

    def to_dict(self, *, include_arrays: bool = True) -> dict[str, Any]:
        payload = {
            "schema_name": TOPOLOGY_SET_SCHEMA_NAME,
            "schema_version": TOPOLOGY_SET_SCHEMA_VERSION,
            "topology_signature_schema_name": "honf_topology_signature",
            "topology_signature_schema_version": 3,
            "forward_checkpoint_sha256": self.forward_checkpoint_sha256,
            "edge_capacity": self.edge_capacity,
            "feature_names": list(COMPACT_PLAN_FEATURE_NAMES),
            "relation_feature_names": list(self.relation_feature_names),
            "metadata": jsonable(self.metadata),
        }
        if include_arrays:
            payload.update(
                raw=self.raw.tolist(),
                normalized=self.normalized.tolist(),
                active_mask=self.active_mask.tolist(),
                relations=self.relations.tolist(),
            )
        return payload


def extract_topology_set(
    signature: Mapping[str, Any],
    design: PhysicalDesign,
    context: NamedContext,
) -> TopologySetPlan:
    """Convert schema-v3 tokens to the unordered 12-value inverse plan ABI."""

    validate_topology_signature(signature)
    names = [str(value) for value in np.asarray(signature["edge_feature_names"]).tolist()]
    features = np.asarray(signature["edge_features"], dtype=np.float64)

    def column(name: str) -> np.ndarray:
        return features[:, names.index(name)]

    active = np.asarray(signature["edge_mask"], dtype=np.float64)
    module_incidence = np.asarray(signature["module_incidence"], dtype=np.float64)
    if module_incidence.shape[0] != design.max_modules:
        raise ValueError("Topology signature module slots do not match the physical design.")
    if int(np.asarray(signature["active_module_count"])) != design.module_count:
        raise ValueError("Topology signature active module count does not match the physical design.")
    module_mass = column("module_mass")
    environment_mass = column("environment_mass")
    strength = np.sqrt(np.maximum(module_mass * environment_mass, 0.0) + 1.0e-6)

    absolute_heat = np.abs(design.heat_powers) * design.module_present
    heat_mass = (module_incidence * absolute_heat[:, None]).sum(axis=0)
    if float(heat_mass.sum()) <= 1.0e-12:
        heat_mass = module_incidence.sum(axis=0)
    heat_fraction = heat_mass / max(float(heat_mass.sum()), 1.0e-12)
    hard_fraction = np.zeros(features.shape[0], dtype=np.float64)
    active_modules = np.flatnonzero(design.module_present > 0.5)
    if active_modules.size:
        owners = module_incidence[active_modules].argmax(axis=1)
        hard_fraction = (
            np.bincount(owners, minlength=features.shape[0]).astype(np.float64)
            / float(active_modules.size)
        )
    raw = np.column_stack(
        (
            active,
            column("source_x"),
            column("source_y"),
            column("region_x"),
            column("region_y"),
            module_mass,
            environment_mass,
            strength,
            column("region_scale_x"),
            column("region_scale_y"),
            heat_fraction,
            hard_fraction,
        )
    ).astype(np.float32)
    raw[active <= 0.5] = 0.0
    normalized = normalize_compact_plan(raw, context)
    checkpoint = str(np.asarray(signature.get("forward_checkpoint_sha256", "")).item())
    return TopologySetPlan(
        raw=raw,
        normalized=normalized,
        active_mask=active,
        relations=np.asarray(signature["edge_relations"], dtype=np.float32),
        relation_feature_names=tuple(
            str(value) for value in np.asarray(signature["relation_feature_names"]).tolist()
        ),
        forward_checkpoint_sha256=checkpoint,
        metadata={
            "case_id": str(np.asarray(signature.get("case_id", "")).item()),
            "reference_measure": str(np.asarray(signature.get("reference_measure", "")).item()),
            "reference_query_digest": str(
                np.asarray(signature.get("reference_query_digest", "")).item()
            ),
            "serialization_permutation": np.asarray(
                signature["serialization_permutation"], dtype=np.int64
            ).tolist(),
        },
    )


def topology_set_dataset_arrays(plans: Sequence[TopologySetPlan]) -> dict[str, np.ndarray]:
    """Stack compatible topology sets for the topology-set HDF5 root group."""

    plans = tuple(plans)
    if not plans:
        raise ValueError("At least one topology set is required.")
    reference = plans[0]
    for plan in plans[1:]:
        if (
            plan.edge_capacity != reference.edge_capacity
            or plan.relation_feature_names != reference.relation_feature_names
            or plan.forward_checkpoint_sha256 != reference.forward_checkpoint_sha256
        ):
            raise ValueError("Topology sets must share capacity, relation schema, and checkpoint provenance.")
    return {
        "tokens_raw": np.stack([plan.raw for plan in plans]),
        "tokens_normalized": np.stack([plan.normalized for plan in plans]),
        "active_mask": np.stack([plan.active_mask for plan in plans]),
        "relations": np.stack([plan.relations for plan in plans]),
        "feature_names": np.asarray(COMPACT_PLAN_FEATURE_NAMES, dtype=object),
        "relation_feature_names": np.asarray(reference.relation_feature_names, dtype=object),
    }


def topology_set_dataset_attributes(plan: TopologySetPlan) -> dict[str, Any]:
    """Return the strict HDF5 provenance attributes required for set training."""

    return {
        "topology_signature_schema_name": "honf_topology_signature",
        "topology_signature_schema_version": 3,
        "forward_topology_checkpoint_sha256": plan.forward_checkpoint_sha256,
        "topology_set_schema_name": TOPOLOGY_SET_SCHEMA_NAME,
        "topology_set_schema_version": TOPOLOGY_SET_SCHEMA_VERSION,
    }


__all__ = [
    "TOPOLOGY_SET_SCHEMA_NAME",
    "TOPOLOGY_SET_SCHEMA_VERSION",
    "TopologySetPlan",
    "extract_topology_set",
    "topology_set_dataset_attributes",
    "topology_set_dataset_arrays",
]

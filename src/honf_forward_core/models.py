"""Stable public model names over the original checkpoint-compatible cores.

A public name selects a concrete core; it never rewrites serialized architecture
IDs, tensor names, physical transforms, or a case adapter. Pass that core's
original configuration, as returned by ``get_config()`` / ``to_dict()``.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn
    from .config import UnifiedForwardConfig


@dataclass(frozen=True)
class ModelFamily:
    name: str
    display_name: str
    development_names: tuple[str, ...]
    contract: str


MODEL_FAMILIES = (
    ModelFamily("hypergraph_routed", "Hypergraph-routed neural field", ("1404", "run1404"),
                "Fixed-slot organization and original-member pairwise reads; legacy configurations remain loadable."),
    ModelFamily("dense_pairwise", "Dense pairwise neural field", ("1804", "run1804"),
                "Dense contextual record exchange and pairwise field reads."),
    ModelFamily("content_attention", "Content-attention neural field", ("a2d", "a2-d"),
                "Direct contextual content reads; historical physical heads belong to the case adapter."),
    ModelFamily("unified_hypergraph", "Unified hypergraph neural field", ("u-h", "uh"),
                "Unified contextual exchange and hypergraph-controlled original-member reads."),
    ModelFamily("relation_anchored", "Relation-anchored neural field", ("s", "r13-11s", "13-11s"),
                "Detail-preserving joint-field prediction with applied relation reuse and inventory-normalized access."),
    ModelFamily("relation_anchored_active", "Active-domain relation-anchored neural field", ("x", "r13-11x", "13-11x"),
                "The same joint-field path with physical access measure normalized on the active domain."),
)


def model_family(name: str) -> ModelFamily:
    """Resolve a public name or a documented development alias."""
    key = name.casefold()
    for family in MODEL_FAMILIES:
        if key in (family.name, *family.development_names):
            return family
    raise ValueError(f"Unknown model {name!r}; choose one of {', '.join(x.name for x in MODEL_FAMILIES)}.")


def relation_anchored_config(name: str, base_model_config: Mapping[str, Any], *,
                             input_dataset: str = "native_records", query_tile_size: int = 256,
                             retained_mass: float = 0.95, ee_initial_tau: float = 1.4907824993133545) -> dict[str, Any]:
    """Create a fresh detail-preserving configuration for either anchored family.

    The base is the case-independent content core configuration. Case adapters
    supply continuous E-cover indices and weights in their query batches.
    Saved configurations must instead be passed through unchanged to build_model.
    """
    family = model_family(name).name
    if family not in ("relation_anchored", "relation_anchored_active"):
        raise ValueError("A relation-anchored family is required.")
    from .interaction_reused_member_routing import DetailAnchoredMemberRoutingCore

    cls = DetailAnchoredMemberRoutingCore
    base = deepcopy(dict(base_model_config))
    # The joint core retains the encoder/contextualizer, not the old reader.
    # Match the normalized base configuration returned by get_config().
    base["thermal_response_rank"] = 0
    base.pop("source_value_conditioning", None)
    base.pop("source_read_policy", None)
    config = dict(schema=cls.schema, architecture=cls._anchored_arm_name("local_e_cover"),
                  locator_mode="local_e_cover", base_model_config=base,
                  input_dataset=input_dataset, input_capability=cls.input_capability,
                  query_tile_size=query_tile_size, retained_mass=retained_mass,
                  score_calibration="inherited", ee_incidence_aggregation="post_gate",
                  ee_gate_mode="relative_quadratic", ee_initial_tau=ee_initial_tau)
    if family == "relation_anchored_active":
        config["access_measure"] = "active_domain"
    cls.validate_config(config)
    return config


def build_model(name: str, config: Mapping[str, Any] | UnifiedForwardConfig) -> nn.Module:
    """Construct a core from its unchanged, strict original configuration.

    The two historical pairwise cores accept ``UnifiedForwardConfig`` or its
    dictionary. Other cores accept their versioned ``get_config()`` mapping.
    Case adapters supply input records, coordinate frames and physical targets.
    Loading a case checkpoint still goes through that case's maintained loader.
    """
    family = model_family(name).name
    if family in ("hypergraph_routed", "dense_pairwise"):
        from .config import UnifiedForwardConfig
        values = config if isinstance(config, UnifiedForwardConfig) else UnifiedForwardConfig.from_dict(dict(config))
        expected = "legacy_honf" if family == "hypergraph_routed" else "dense_pairwise_field"
        if values.forward_architecture != expected:
            raise ValueError(f"{family} requires the saved forward_architecture={expected!r}.")
        if family == "hypergraph_routed":
            from .model import HONFNeuralField
            return HONFNeuralField(values)
        from .interface_fields.core import InterfaceFieldCore
        return InterfaceFieldCore(values)
    values = dict(config)
    if family == "content_attention":
        from .content_hypergraph.model_v2 import ContentHypergraphCoreV2
        if values.get("architecture") != "A2-D":
            raise ValueError("content_attention requires the original A2-D configuration.")
        return ContentHypergraphCoreV2.from_config(values)
    if family == "unified_hypergraph":
        from .continuous_joint.unified import UnifiedJointCore
        if values.get("routing_mode") != "hypergraph":
            raise ValueError("unified_hypergraph requires routing_mode='hypergraph'.")
        UnifiedJointCore.validate_config(values)
        return UnifiedJointCore.from_config(values)
    from .interaction_reused_member_routing import anchored_core_from_config
    expected = "active_domain" if family == "relation_anchored_active" else "inventory_unit_total"
    if (values.get("access_measure", "inventory_unit_total") != expected
            or values.get("ee_gate_mode") != "relative_quadratic"
            or values.get("ee_incidence_aggregation") != "post_gate"
            or values.get("score_calibration") != "inherited"
            or values.get("mediated_mode", "none") != "none"):
        raise ValueError(f"{family} requires its original relative-quadratic EE, inherited-calibration and {expected} access configuration.")
    return anchored_core_from_config(values)

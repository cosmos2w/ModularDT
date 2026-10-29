"""Stable public model names over the original checkpoint-compatible cores.

A public name selects a concrete core; it never rewrites serialized architecture
IDs, tensor names, physical transforms, or a case adapter. Pass that core's
original configuration, as returned by ``get_config()`` / ``to_dict()``.
"""
from __future__ import annotations

from collections.abc import Mapping
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
                "Joint-field prediction with applied relation reuse and inventory-normalized E-addressed access."),
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
    from .interaction_reused_member_routing.anchored import AnchoredMemberRoutingCore
    expected = "active_domain" if family == "relation_anchored_active" else "inventory_unit_total"
    if (values.get("access_measure", "inventory_unit_total") != expected
            or values.get("ee_gate_mode") != "relative_quadratic"
            or values.get("ee_incidence_aggregation") != "post_gate"
            or values.get("score_calibration") != "inherited"
            or values.get("mediated_mode", "none") != "none"):
        raise ValueError(f"{family} requires its original relative-quadratic EE, inherited-calibration and {expected} access configuration.")
    return AnchoredMemberRoutingCore.from_config(values)

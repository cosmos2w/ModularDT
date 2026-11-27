"""Identify the parameters that exclusively govern HONF relationships.

Ownership is declared from concrete module roles, then resolved by parameter
identity. A tied tensor is shared/content whenever any of its registered
owners is outside those roles.
"""
from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from honf_forward_core.content_hypergraph.staged_attention_v2 import SpatialRouteAttentionV2
from honf_forward_core.interaction_reused_member_routing.anchored import AnchoredMemberRoutingCore
from honf_forward_core.interaction_reused_member_routing.core import (
    AppliedInteractionIncidence,
    InteractionMemberRouter,
)


@dataclass(frozen=True)
class ParameterOwnership:
    organization_names: tuple[str, ...]
    content_names: tuple[str, ...]
    organization_parameters: tuple[nn.Parameter, ...]
    content_parameters: tuple[nn.Parameter, ...]

    def metadata(self):
        return {
            "organization_parameters": list(self.organization_names),
            "content_parameters": list(self.content_names),
            "meaning": "Exclusive relationship parameters; tied or value-path parameters are content/shared.",
        }


def partition_organization_parameters(model: nn.Module) -> ParameterOwnership:
    """Partition every trainable tensor once using concrete module ownership."""
    relationship_paths: set[str] = set()

    def add_role(owner_path, owner, role):
        child = getattr(owner, role, None)
        if isinstance(child, nn.Module):
            prefix = f"{owner_path}.{role}" if owner_path else role
            relationship_paths.update(f"{prefix}.{name}" for name, _ in
                                      child.named_parameters(remove_duplicate=False))
        elif isinstance(child, nn.Parameter):
            relationship_paths.add(f"{owner_path}.{role}" if owner_path else role)

    for path, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, SpatialRouteAttentionV2):
            for role in ("query", "key", "geometry_score", "raw_temperature"):
                add_role(path, module, role)
            # The donor key normalization also feeds the value projection.
            # Optional message replacements additionally use receiver_norm.
            if module.pair_message is None and module.content_replacement is None:
                add_role(path, module, "query_norm")
        elif isinstance(module, AppliedInteractionIncidence):
            relationship_paths.update(
                f"{path}.{name}" if path else name
                for name, _ in module.named_parameters(remove_duplicate=False)
            )
        elif isinstance(module, InteractionMemberRouter):
            for role in ("key_M", "key_E", "proposal_M", "member_M", "member_E"):
                add_role(path, module, role)
        if isinstance(module, AnchoredMemberRoutingCore):
            for role in ("typed_read_gates", "evidence_strength"):
                add_role(path, module, role)

    aliases: dict[int, list[str]] = {}
    values: dict[int, nn.Parameter] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if parameter.requires_grad:
            aliases.setdefault(id(parameter), []).append(name)
            values[id(parameter)] = parameter
    organization_names, content_names = [], []
    organization, content = [], []
    for parameter_id, names in aliases.items():
        exclusive = all(name in relationship_paths for name in names)
        (organization_names if exclusive else content_names).append(names[0])
        (organization if exclusive else content).append(values[parameter_id])
    if not organization or not content:
        raise ValueError("The model must expose both relationship and content/shared trainable parameters.")
    return ParameterOwnership(tuple(organization_names), tuple(content_names),
                              tuple(organization), tuple(content))

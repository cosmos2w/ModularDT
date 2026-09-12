"""Small tensor containers for the matched interface-field architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .response_hierarchy import EnvironmentHierarchy, HierarchyGeometry


@dataclass(frozen=True)
class EncodedInterfaceCase:
    """Case encodings shared by all new interaction backends."""

    module_tokens: torch.Tensor
    env_tokens: torch.Tensor
    global_token: torch.Tensor
    module_centers: torch.Tensor
    env_coords: torch.Tensor
    module_present: torch.Tensor
    module_features: torch.Tensor
    env_features: torch.Tensor | None
    env_weights: torch.Tensor
    coordinate_scale: torch.Tensor
    # Optional adapter-supplied physical region IDs aligned with ``env_coords``.
    # IDs are intentionally kept separate from encoded tokens so the common
    # coarse route can continue to consume the original fine environment.
    env_region_ids: torch.Tensor | None = None
    env_hierarchy: EnvironmentHierarchy | None = None
    env_hierarchy_geometry: HierarchyGeometry | None = None


@dataclass(frozen=True)
class PreparedInterfaceField:
    """Differentiable backend state for one physical coupling pass."""

    encoded: EncodedInterfaceCase
    module_states: torch.Tensor
    backend_state: Any
    coarse_state: torch.Tensor
    interaction_aux: Dict[str, Any]


@dataclass(frozen=True)
class InterfaceRead:
    """A continuous interaction-context read and its diagnostics."""

    context: torch.Tensor
    interaction_aux: Dict[str, torch.Tensor]

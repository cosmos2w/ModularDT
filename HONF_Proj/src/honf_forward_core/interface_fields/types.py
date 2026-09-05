"""Small tensor containers for the matched interface-field architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch


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

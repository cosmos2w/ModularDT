"""Minimal three-term context for the Run-1405 field reader.

The fixed-group backend supplies the two fine terms.  This module supplies
only the query/global background term and the field head, keeping the common
``InterfaceFieldCore`` facade intact while removing the historical coarse
latent bank and local-neighbour branch for the opt-in profile.
"""

from __future__ import annotations

import torch
from torch import nn

from honf_forward_core.nn import LazyMLP


class ThreeTermInterfaceContext(nn.Module):
    """Implement ``C_g + C_pair,M + C_pair,E`` behind the common API."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        field_dim: int,
        query_fourier_frequencies: int | None = None,
    ) -> None:
        super().__init__()
        del query_fourier_frequencies
        self.hidden_dim = int(hidden_dim)
        self.field_dim = int(field_dim)
        if self.hidden_dim <= 0 or self.field_dim <= 0:
            raise ValueError("hidden_dim and field_dim must be positive.")

        # Both receiver Fourier features and the global token are fed to this
        # network by read_coarse.  LazyMLP keeps the existing adapter-defined
        # query-feature width boundary without exposing it in the context API.
        self.global_background = LazyMLP(self.hidden_dim, num_layers=2)
        self.context_norm = nn.LayerNorm(self.hidden_dim)
        self.field_head = LazyMLP(
            self.hidden_dim,
            out_dim=self.field_dim,
            num_layers=3,
        )

    def prepare_coarse(
        self,
        module_states: torch.Tensor,
        env_tokens: torch.Tensor,
        module_present: torch.Tensor,
        env_weights: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Return a tiny shape-only state; source tokens are intentionally unused."""

        del env_tokens, module_present, env_weights, args, kwargs
        return module_states.new_empty((int(module_states.shape[0]), 0, self.hidden_dim))

    def read_coarse(
        self,
        receiver_features: torch.Tensor,
        global_token: torch.Tensor,
        coarse_state: torch.Tensor | None,
    ) -> torch.Tensor:
        """Evaluate ``C_g(q)=F_g[Phi_Q(q/s), g]`` only."""

        del coarse_state
        if receiver_features.ndim != 3:
            raise ValueError("receiver_features must have shape [B,Q,F].")
        if global_token.ndim != 2 or int(global_token.shape[0]) != int(receiver_features.shape[0]):
            raise ValueError("global_token must have shape [B,H] aligned with receiver_features.")
        global_values = global_token[:, None, :].expand(
            -1,
            int(receiver_features.shape[1]),
            -1,
        )
        return self.global_background(torch.cat([receiver_features, global_values], dim=-1))

    def read_local(
        self,
        receivers: torch.Tensor,
        module_states: torch.Tensor,
        module_centers: torch.Tensor,
        module_features: torch.Tensor,
        module_present: torch.Tensor,
        coordinate_scale: torch.Tensor,
        module_radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact zeros without constructing distances or local messages."""

        del module_centers, module_features, module_present, coordinate_scale, module_radius
        if receivers.ndim != 3 or module_states.ndim != 3:
            raise ValueError("receivers and module_states must have shape [B,N,*].")
        if int(receivers.shape[0]) != int(module_states.shape[0]):
            raise ValueError("receivers and module_states must share the batch dimension.")
        zeros = module_states.new_zeros(
            (int(receivers.shape[0]), int(receivers.shape[1]), self.hidden_dim)
        )
        counts = module_states.new_zeros(
            (int(receivers.shape[0]), int(receivers.shape[1]))
        )
        return zeros, counts

    def predict_field(
        self,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        context: torch.Tensor,
        global_token: torch.Tensor,
        query_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply only ``D_theta[LayerNorm(C_g+C_pair,M+C_pair,E)]``."""

        del receivers, receiver_features, global_token, query_features
        if context.ndim != 3 or int(context.shape[-1]) != self.hidden_dim:
            raise ValueError("context must have shape [B,Q,hidden_dim].")
        return self.field_head(self.context_norm(context))


__all__ = ["ThreeTermInterfaceContext"]

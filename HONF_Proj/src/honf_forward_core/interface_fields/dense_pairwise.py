"""Contextualized dense pairwise neural-integral field adaptation."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from honf_forward_core.nn import FourierFeatures, LazyMLP

from .common import BiasedMultiheadAttention
from .types import EncodedInterfaceCase


class DensePairwiseField(nn.Module):
    """Strong collective pairwise backend; not an exact paper reproduction."""

    def __init__(self, hidden_dim: int, message_hidden_dim: int, num_heads: int, fourier_frequencies: int, *, activation_checkpointing: bool = False):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.relative_fourier = FourierFeatures(None, fourier_frequencies)
        self.mm_message = LazyMLP(message_hidden_dim, out_dim=hidden_dim, num_layers=3)
        self.me_message = LazyMLP(message_hidden_dim, out_dim=hidden_dim, num_layers=3)
        self.em_message = LazyMLP(message_hidden_dim, out_dim=hidden_dim, num_layers=3)
        self.module_update = LazyMLP(hidden_dim, num_layers=3)
        self.env_update = LazyMLP(hidden_dim, num_layers=3)
        self.query_module_message = LazyMLP(message_hidden_dim, out_dim=hidden_dim, num_layers=3)
        self.query_module_output = nn.Linear(hidden_dim, hidden_dim)
        self.env_query = LazyMLP(hidden_dim, num_layers=2)
        self.env_attention = BiasedMultiheadAttention(hidden_dim, num_heads)
        self.env_geometry_bias = LazyMLP(message_hidden_dim, out_dim=num_heads, num_layers=2)

    def _mlp(self, module: nn.Module, values: torch.Tensor) -> torch.Tensor:
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, values, use_reentrant=False)
        return module(values)

    def prepare_fine_messages(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Prepare Dense's simultaneous fine MM/ME/EM messages.

        The regional backend reuses this exact preparation and only changes
        what happens after the fine ``EM`` reduction.  In particular, all
        three typed messages consume the input ``module_states``; the module
        update is not fed back into the environmental message in the same
        pass.
        """

        centers = encoded.module_centers
        env_coords = encoded.env_coords
        scale = encoded.coordinate_scale
        present = encoded.module_present
        _batch, modules, _ = module_states.shape
        env_count = int(encoded.env_tokens.shape[1])
        active_count = present.sum(dim=1, keepdim=True).clamp_min(0.0)

        mm_rel = (centers[:, :, None, :] - centers[:, None, :, :]) / scale
        mm_features = self.relative_fourier(mm_rel)
        zi = module_states[:, :, None, :].expand(-1, -1, modules, -1)
        zl = module_states[:, None, :, :].expand(-1, modules, -1, -1)
        mm = self._mlp(self.mm_message, torch.cat([zi, zl, mm_features], dim=-1))
        eye = torch.eye(modules, device=present.device, dtype=torch.bool)[None]
        mm_mask = (present[:, :, None] > 0.5) & (present[:, None, :] > 0.5) & ~eye
        a_mm = (mm * mm_mask[..., None]).sum(dim=2) / (1.0 + active_count[..., None])

        me_rel = (centers[:, :, None, :] - env_coords[:, None, :, :]) / scale
        me_features = self.relative_fourier(me_rel)
        zi_env = module_states[:, :, None, :].expand(-1, -1, env_count, -1)
        ej_mod = encoded.env_tokens[:, None, :, :].expand(-1, modules, -1, -1)
        me = self._mlp(self.me_message, torch.cat([zi_env, ej_mod, me_features], dim=-1))
        weighted_me = me * encoded.env_weights[:, None, :, None]
        a_me = weighted_me.sum(dim=2) / encoded.env_weights.sum(dim=1)[:, None, None].clamp_min(1.0e-12)
        a_me = a_me * present[..., None]
        global_modules = encoded.global_token[:, None, :].expand(-1, modules, -1)
        contextual_modules = (
            module_states + self.module_update(torch.cat([module_states, a_mm, a_me, global_modules], dim=-1))
        ) * present[..., None]

        em_rel = -me_rel.transpose(1, 2)
        em_features = self.relative_fourier(em_rel)
        ej_sources = encoded.env_tokens[:, :, None, :].expand(-1, -1, modules, -1)
        zi_for_env = module_states[:, None, :, :].expand(-1, env_count, -1, -1)
        em = self._mlp(self.em_message, torch.cat([ej_sources, zi_for_env, em_features], dim=-1))
        a_em = (em * present[:, None, :, None]).sum(dim=2) / (1.0 + active_count[:, :, None])
        return {
            "module_tokens": contextual_modules,
            # Keep the fine encoded environment available to the common
            # coarse route and to frozen evaluation utilities.
            "env_tokens": encoded.env_tokens,
            "environment_messages": a_em,
        }

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> dict[str, torch.Tensor]:
        del return_routing_maps
        fine = self.prepare_fine_messages(encoded, module_states)
        global_env = encoded.global_token[:, None, :].expand(-1, fine["env_tokens"].shape[1], -1)
        contextual_env = fine["env_tokens"] + self.env_update(
            torch.cat([fine["env_tokens"], fine["environment_messages"], global_env], dim=-1)
        )
        return {"module_tokens": fine["module_tokens"], "env_tokens": contextual_env}

    def read_module(
        self,
        state: dict[str, torch.Tensor],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
    ) -> torch.Tensor:
        """Read Dense's direct nonlinear query--module contribution."""

        modules = int(state["module_tokens"].shape[1])
        relative = (receivers[:, :, None, :] - encoded.module_centers[:, None, :, :]) / encoded.coordinate_scale
        relative_features = self.relative_fourier(relative)
        source = state["module_tokens"][:, None, :, :].expand(-1, receivers.shape[1], -1, -1)
        global_features = encoded.global_token[:, None, None, :].expand(-1, receivers.shape[1], modules, -1)
        module_messages = self._mlp(
            self.query_module_message,
            torch.cat([source, relative_features, global_features], dim=-1),
        )
        return self.query_module_output(
            (module_messages * encoded.module_present[:, None, :, None]).sum(dim=2)
            / (1.0 + encoded.module_present.sum(dim=1)[:, None, None])
        )

    def project_environment_sources(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project prepared environmental sources for reuse across reads."""

        return self.env_attention.project_source(source)

    def read_environment_projected(
        self,
        projected_key: torch.Tensor,
        projected_value: torch.Tensor,
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        source_coords: torch.Tensor,
        source_weights: torch.Tensor,
        *,
        source_mask: torch.Tensor | None = None,
        return_routing_maps: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Read projected environmental sources with supplied geometry/mass."""

        source_relative = (receivers[:, :, None, :] - source_coords[:, None, :, :]) / encoded.coordinate_scale
        source_bias = self._mlp(
            self.env_geometry_bias,
            self.relative_fourier(source_relative),
        ).permute(0, 3, 1, 2)
        env_query = self.env_query(receiver_features)
        projected_query = self.env_attention.project_query(env_query)
        if source_mask is None:
            # Caller-owned sources are expected to have positive quadrature
            # masses.  Keep their exact ratios, including subnormal values.
            log_weights = torch.log(source_weights)
        else:
            # Padded regional slots are masked by the attention reader.  Give
            # those slots a finite placeholder only for log evaluation so a
            # zero padding mass cannot produce ``log(0)``; retained positive
            # masses are never clamped or otherwise altered.
            safe_weights = torch.where(
                source_mask > 0.5,
                source_weights,
                torch.ones_like(source_weights),
            )
            log_weights = torch.log(safe_weights)
        return self.env_attention.read_projected(
            projected_query,
            projected_key,
            projected_value,
            bias=source_bias,
            source_mask=source_mask,
            log_weights=log_weights,
            return_attention=bool(return_routing_maps),
        )

    def read_environment(
        self,
        state: dict[str, torch.Tensor],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Read Dense's environmental contribution independently."""

        env_relative = (receivers[:, :, None, :] - encoded.env_coords[:, None, :, :]) / encoded.coordinate_scale
        env_bias = self._mlp(self.env_geometry_bias, self.relative_fourier(env_relative)).permute(0, 3, 1, 2)
        env_query = self.env_query(receiver_features)
        return self.env_attention(
            env_query,
            state["env_tokens"],
            bias=env_bias,
            log_weights=torch.log(encoded.env_weights.clamp_min(torch.finfo(encoded.env_weights.dtype).tiny)),
            return_attention=bool(return_routing_maps),
        )

    def read(
        self,
        state: dict[str, torch.Tensor],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        module_context = self.read_module(state, encoded, receivers, receiver_features)

        env_context, env_attention = self.read_environment(
            state,
            encoded,
            receivers,
            receiver_features,
            return_routing_maps=bool(return_routing_maps),
        )
        aux: dict[str, torch.Tensor] = {
            # This scalar per receiver is retained for training summaries.  The
            # potentially large attention tensor is only materialized when the
            # caller explicitly asks for routing maps.
            "dense_module_context_norm": torch.linalg.vector_norm(module_context, dim=-1),
        }
        if return_routing_maps and env_attention is not None:
            aux["dense_environment_attention"] = env_attention
        return module_context + env_context, aux

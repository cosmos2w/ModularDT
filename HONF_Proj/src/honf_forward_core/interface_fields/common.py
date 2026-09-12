"""Shared attention, coarse communication, local correction, and field head."""

from __future__ import annotations

import math

import torch
from torch import nn

from honf_forward_core.nn import MLP, FourierFeatures, LazyMLP


class BiasedMultiheadAttention(nn.Module):
    """Pre-normalized multi-head attention with geometric and quadrature bias."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.source_norm = nn.LayerNorm(self.hidden_dim)
        self.query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.hidden_dim)

    def project_query(self, query: torch.Tensor) -> torch.Tensor:
        """Return the projected query in ``[B, heads, Q, head_dim]`` form.

        The projection is intentionally exposed for readers that prepare a
        source once and decode several receiver chunks.  It is a normal
        differentiable module call; callers must not detach or reuse the
        result after the source state has changed.
        """

        batch, query_count, _ = query.shape
        return self.query(self.query_norm(query)).reshape(
            batch, query_count, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def project_source(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return projected source keys and values in attention-head form."""

        batch, source_count, _ = source.shape
        normalized = self.source_norm(source)
        key = self.key(normalized).reshape(
            batch, source_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = self.value(normalized).reshape(
            batch, source_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        return key, value

    def read_projected(
        self,
        projected_query: torch.Tensor,
        projected_key: torch.Tensor,
        projected_value: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        source_mask: torch.Tensor | None = None,
        log_weights: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Read already projected sources with the usual biased attention.

        ``projected_query``, ``projected_key`` and ``projected_value`` use
        ``[B, heads, query/source, head_dim]``.  This helper keeps source
        LayerNorm and projections outside receiver chunks while retaining
        the same softmax, mask, quadrature-weight and output projection as
        :meth:`forward`.
        """

        if projected_query.ndim != 4 or projected_key.ndim != 4 or projected_value.ndim != 4:
            raise ValueError("Projected attention tensors must have four dimensions.")
        batch, heads, query_count, head_dim = projected_query.shape
        if heads != self.num_heads or head_dim != self.head_dim:
            raise ValueError("Projected query shape does not match attention heads.")
        if projected_key.shape[:2] != (batch, heads) or projected_value.shape[:2] != (batch, heads):
            raise ValueError("Projected source batch/head dimensions do not match the query.")
        source_count = int(projected_key.shape[2])
        if projected_value.shape[2] != source_count:
            raise ValueError("Projected key/value source counts must match.")

        scores = torch.matmul(projected_query, projected_key.transpose(-1, -2)) / math.sqrt(float(head_dim))
        if bias is not None:
            scores = scores + (bias[:, None] if bias.ndim == 3 else bias)
        if log_weights is not None:
            scores = scores + log_weights[:, None, None, :]
        if source_mask is not None:
            valid = source_mask[:, None, None, :] > 0.5
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        if source_mask is not None:
            valid_float = (source_mask[:, None, None, :] > 0.5).to(weights.dtype)
            weights = weights * valid_float
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        result = torch.matmul(weights, projected_value).transpose(1, 2).reshape(batch, query_count, self.hidden_dim)
        return self.output(result), weights if return_attention else None

    def forward(
        self,
        query: torch.Tensor,
        source: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        source_mask: torch.Tensor | None = None,
        log_weights: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, query_count, _ = query.shape
        source_count = int(source.shape[1])
        q = self.query(self.query_norm(query)).reshape(batch, query_count, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(self.source_norm(source)).reshape(batch, source_count, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(self.source_norm(source)).reshape(batch, source_count, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(self.head_dim))
        if bias is not None:
            scores = scores + (bias[:, None] if bias.ndim == 3 else bias)
        if log_weights is not None:
            scores = scores + log_weights[:, None, None, :]
        if source_mask is not None:
            valid = source_mask[:, None, None, :] > 0.5
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        if source_mask is not None:
            valid_float = (source_mask[:, None, None, :] > 0.5).to(weights.dtype)
            weights = weights * valid_float
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        result = torch.matmul(weights, v).transpose(1, 2).reshape(batch, query_count, self.hidden_dim)
        return self.output(result), weights if return_attention else None


class PreNormAttentionBlock(nn.Module):
    """One standard latent self-attention/residual MLP block."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.attention = BiasedMultiheadAttention(hidden_dim, num_heads)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers=2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(tokens, tokens)
        tokens = tokens + attended
        return tokens + self.mlp(self.mlp_norm(tokens))


class SharedInterfaceContext(nn.Module):
    """Common G-token coarse route, compact local route, and physical field head."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        field_dim: int,
        num_heads: int,
        coarse_latent_count: int,
        coarse_blocks: int,
        local_radius_factor: float,
        fourier_frequencies: int,
        coarse_module_source: str = "module_states",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.local_radius_factor = float(local_radius_factor)
        self.coarse_module_source = str(coarse_module_source)
        if self.coarse_module_source not in {"module_states", "group_states"}:
            raise ValueError(
                "coarse_module_source must be either 'module_states' or 'group_states'."
            )
        self.relative_fourier = FourierFeatures(None, fourier_frequencies)
        self.receiver_fourier = FourierFeatures(None, fourier_frequencies)
        self.coarse_seeds = nn.Parameter(torch.randn(coarse_latent_count, hidden_dim) / math.sqrt(float(hidden_dim)))
        # Keep the historical module names and parameter ownership exactly in
        # the default mode.  In group-source mode, do not construct a second
        # source network: the experiment replaces the module source route
        # with one same-width/head-count attention module.
        if self.coarse_module_source == "module_states":
            self.coarse_module_attention = BiasedMultiheadAttention(hidden_dim, num_heads)
        else:
            self.coarse_group_attention = BiasedMultiheadAttention(hidden_dim, num_heads)
        self.coarse_env_attention = BiasedMultiheadAttention(hidden_dim, num_heads)
        self.coarse_blocks = nn.ModuleList(
            PreNormAttentionBlock(hidden_dim, num_heads) for _ in range(int(coarse_blocks))
        )
        self.coarse_query = LazyMLP(hidden_dim, num_layers=2)
        self.coarse_read = BiasedMultiheadAttention(hidden_dim, num_heads)
        self.local_message = LazyMLP(hidden_dim, num_layers=3)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.field_head = LazyMLP(hidden_dim, out_dim=field_dim, num_layers=3)

    def prepare_coarse(
        self,
        module_states: torch.Tensor,
        env_tokens: torch.Tensor,
        module_present: torch.Tensor,
        env_weights: torch.Tensor,
        packed_group_states: torch.Tensor | None = None,
        packed_group_occupancy: torch.Tensor | None = None,
        packed_group_valid: torch.Tensor | None = None,
        *,
        group_source_enabled: bool = True,
    ) -> torch.Tensor:
        seeds = self.coarse_seeds.unsqueeze(0).expand(module_states.shape[0], -1, -1)
        if self.coarse_module_source == "module_states":
            # Preserve the historical call and arithmetic verbatim when the
            # option is absent.  Extra group-source arguments are ignored in
            # this branch so old callers/checkpoints remain compatible.
            module_context, _ = self.coarse_module_attention(
                seeds, module_states, source_mask=module_present
            )
        else:
            if not group_source_enabled:
                # This is the diagnostic group-source-only intervention.  It
                # happens before the unchanged coarse processor, while the
                # environmental background below remains active.
                module_context = torch.zeros_like(seeds)
            else:
                if packed_group_states is None:
                    raise ValueError(
                        "group_states coarse preparation requires packed_group_states."
                    )
                if packed_group_occupancy is None:
                    raise ValueError(
                        "group_states coarse preparation requires packed_group_occupancy."
                    )
                if packed_group_valid is None:
                    packed_group_valid = packed_group_occupancy > 0.0
                self._validate_packed_group_sources(
                    packed_group_states,
                    packed_group_occupancy,
                    packed_group_valid,
                    batch_size=int(module_states.shape[0]),
                )
                source_occupancy = packed_group_occupancy.to(
                    device=packed_group_states.device,
                    dtype=packed_group_states.dtype,
                )
                module_context, _ = self.coarse_group_attention(
                    seeds,
                    packed_group_states,
                    source_mask=packed_group_valid,
                    log_weights=torch.log(
                        source_occupancy.clamp_min(torch.finfo(source_occupancy.dtype).tiny)
                    ),
                )
                # BiasedMultiheadAttention's output projection has a bias. If
                # a batch item has no occupied groups, masking all rows would
                # otherwise expose that bias as fictitious group information.
                # Keep the supported rows unchanged and make empty cases
                # genuinely background-only.
                has_group_source = packed_group_valid.to(dtype=torch.bool).any(dim=-1)
                module_context = module_context * has_group_source[:, None, None].to(
                    dtype=module_context.dtype
                )
        env_context, _ = self.coarse_env_attention(
            seeds, env_tokens, log_weights=torch.log(env_weights.clamp_min(torch.finfo(env_weights.dtype).tiny))
        )
        coarse = seeds + module_context + env_context
        for block in self.coarse_blocks:
            coarse = block(coarse)
        return coarse

    def _validate_packed_group_sources(
        self,
        group_states: torch.Tensor,
        occupancy: torch.Tensor,
        valid: torch.Tensor,
        *,
        batch_size: int,
    ) -> None:
        if group_states.ndim != 3:
            raise ValueError("packed_group_states must have shape [B,G,H].")
        if occupancy.ndim != 2 or valid.ndim != 2:
            raise ValueError("packed_group_occupancy and packed_group_valid must have shape [B,G].")
        if group_states.shape[:2] != occupancy.shape or occupancy.shape != valid.shape:
            raise ValueError("Packed group states, occupancy, and validity masks must align.")
        if int(group_states.shape[0]) != batch_size:
            raise ValueError("Packed group sources must match the module-state batch size.")
        if int(group_states.shape[-1]) != self.hidden_dim:
            raise ValueError(
                f"packed_group_states hidden width must be {self.hidden_dim}, got {group_states.shape[-1]}."
            )
        if not torch.is_floating_point(group_states) or not torch.is_floating_point(occupancy):
            raise ValueError("Packed group states and occupancy must use floating-point tensors.")
        if valid.dtype not in {torch.bool, torch.float16, torch.float32, torch.float64, torch.bfloat16}:
            raise ValueError("packed_group_valid must be boolean or floating point.")
        if group_states.device != occupancy.device or occupancy.device != valid.device:
            raise ValueError("Packed group states, occupancy, and validity masks must share a device.")
        if not bool(torch.isfinite(occupancy).all()):
            raise ValueError("packed_group_occupancy must be finite.")
        if bool((occupancy < 0.0).any()):
            raise ValueError("packed_group_occupancy must be nonnegative.")
        occupied = valid > 0.5
        if bool((occupied & (occupancy <= 0.0)).any()):
            raise ValueError("Valid packed group sources must have strictly positive occupancy.")

    def read_coarse(
        self,
        receiver_features: torch.Tensor,
        global_token: torch.Tensor,
        coarse_state: torch.Tensor,
    ) -> torch.Tensor:
        query = self.coarse_query(torch.cat([receiver_features, global_token[:, None, :].expand(-1, receiver_features.shape[1], -1)], dim=-1))
        context, _ = self.coarse_read(query, coarse_state)
        return context

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
        """Gather neighbours before evaluating the local message network."""

        batch, query_count, _spatial_dim = receivers.shape
        output_batches = []
        count_batches = []
        support_radius = max(float(module_radius) * self.local_radius_factor, 1.0e-8)
        for batch_index in range(batch):
            batch_output = module_states.new_zeros(query_count, self.hidden_dim)
            batch_counts = module_states.new_zeros(query_count)
            relative = receivers[batch_index, :, None, :] - module_centers[batch_index, None, :, :]
            distances = torch.linalg.vector_norm(relative, dim=-1)
            neighbour_mask = (distances < support_radius) & (module_present[batch_index, None, :] > 0.5)
            query_index, module_index = torch.nonzero(neighbour_mask, as_tuple=True)
            if query_index.numel() == 0:
                output_batches.append(batch_output)
                count_batches.append(batch_counts)
                continue
            selected_relative = relative[query_index, module_index] / coordinate_scale.reshape(-1)
            relative_features = self.relative_fourier(selected_relative)
            messages = self.local_message(
                torch.cat(
                    [
                        module_states[batch_index, module_index],
                        relative_features,
                        module_features[batch_index, module_index],
                    ],
                    dim=-1,
                )
            )
            normalized_distance = distances[query_index, module_index] / support_radius
            kernel = ((1.0 - normalized_distance).clamp_min(0.0).square() * (1.0 + 2.0 * normalized_distance))
            batch_output = torch.index_add(batch_output, 0, query_index, kernel[:, None] * messages)
            mass = torch.index_add(module_states.new_zeros(query_count), 0, query_index, kernel)
            batch_counts = torch.index_add(batch_counts, 0, query_index, torch.ones_like(kernel))
            output_batches.append(batch_output / (1.0 + mass[:, None]))
            count_batches.append(batch_counts)
        return torch.stack(output_batches, dim=0), torch.stack(count_batches, dim=0)

    def predict_field(
        self,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        context: torch.Tensor,
        global_token: torch.Tensor,
        query_features: torch.Tensor | None,
    ) -> torch.Tensor:
        pieces = [receiver_features, self.context_norm(context), global_token[:, None, :].expand(-1, receivers.shape[1], -1)]
        if query_features is not None:
            pieces.append(query_features)
        return self.field_head(torch.cat(pieces, dim=-1))

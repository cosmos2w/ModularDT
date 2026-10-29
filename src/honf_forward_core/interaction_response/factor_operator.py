"""Anchored unary and pair response factors with no trial-design bypass."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Literal

import torch
from torch import nn

from .types import (
    DEFAULT_OUTPUT_CHANNELS,
    DEFAULT_QUERY_FEATURE_DIMS,
    BaselineResponseCache,
    ResponseFactor,
    ResponseQueries,
    ResponseQueryBatch,
)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _normalize_channel_spec(
    output_channels: Mapping[str, int | Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    source = DEFAULT_OUTPUT_CHANNELS if output_channels is None else output_channels
    result: dict[str, tuple[str, ...]] = {}
    for role, spec in source.items():
        if isinstance(spec, int):
            if spec < 1:
                raise ValueError(f"Output width for role {role!r} must be positive.")
            result[role] = tuple(f"{role}_{index}" for index in range(spec))
        else:
            names = tuple(spec)
            if not names or len(set(names)) != len(names):
                raise ValueError(f"Output channels for role {role!r} must be nonempty and unique.")
            result[role] = names
    if not result:
        raise ValueError("At least one output role must be configured.")
    return result


class AnchoredResponseFactorOperator(nn.Module):
    """Predict local multi-output changes through physical donor factors.

    The only trial-dependent argument is ``delta_by_module_id``. For each
    factor, the implementation looks up only its declared donor IDs and
    constructs unary or pair responses by batched inclusion--exclusion. The
    cached baseline, factor graph, receiver support, and query coordinates are
    shared by every sign variant.
    """

    def __init__(
        self,
        *,
        module_feature_dim: int,
        design_dim: int,
        context_dim: int,
        delta_dim: int,
        query_feature_dims: Mapping[str, int] | None = None,
        output_channels: Mapping[str, int | Sequence[str]] | None = None,
        hidden_dim: int = 96,
        decoder_mode: Literal["inclusion_exclusion", "delta_gated"] = "inclusion_exclusion",
    ) -> None:
        super().__init__()
        if min(module_feature_dim, design_dim, delta_dim, hidden_dim) <= 0 or context_dim < 0:
            raise ValueError("Feature widths and hidden_dim must be positive; context_dim may be zero.")
        if context_dim == 0:
            # A learned constant baseline context is still useful while keeping
            # the public cache shape explicit and consistent.
            context_dim = 1
            self.accept_empty_context = True
        else:
            self.accept_empty_context = False

        self.module_feature_dim = int(module_feature_dim)
        self.design_dim = int(design_dim)
        self.context_dim = int(context_dim)
        self.delta_dim = int(delta_dim)
        self.hidden_dim = int(hidden_dim)
        if decoder_mode not in {"inclusion_exclusion", "delta_gated"}:
            raise ValueError("decoder_mode must be 'inclusion_exclusion' or 'delta_gated'.")
        self.decoder_mode = decoder_mode
        self.output_channel_names = _normalize_channel_spec(output_channels)
        requested_query_dims = DEFAULT_QUERY_FEATURE_DIMS if query_feature_dims is None else query_feature_dims
        self.query_feature_dims = {role: int(dim) for role, dim in requested_query_dims.items()}
        missing_dims = set(self.output_channel_names) - set(self.query_feature_dims)
        if missing_dims:
            raise ValueError(f"Every configured output role needs query features: {sorted(missing_dims)!r}.")
        if any(dim < 1 for dim in self.query_feature_dims.values()):
            raise ValueError("Query feature dimensions must be positive.")

        hidden = self.hidden_dim
        baseline_input_dim = self.module_feature_dim + self.design_dim
        self.baseline_module_encoder = nn.Sequential(
            nn.Linear(baseline_input_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.baseline_context_encoder = nn.Sequential(
            nn.Linear(self.context_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.pair_geometry_encoder = nn.Sequential(
            nn.Linear(self.design_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.edge_context_encoder = _mlp(hidden * 4, hidden, hidden)
        self.donor_change_encoder = _mlp(hidden + self.delta_dim, hidden, hidden)
        self.role_names = tuple(self.output_channel_names)
        self.role_to_index = {role: index for index, role in enumerate(self.role_names)}
        self.role_embedding = nn.Embedding(len(self.role_names), hidden)
        self.order_embedding = nn.Embedding(3, hidden)
        self.query_encoders = nn.ModuleDict(
            {
                role: nn.Sequential(
                    nn.Linear(feature_dim + hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
                )
                for role, feature_dim in self.query_feature_dims.items()
                if role in self.output_channel_names
            }
        )
        if self.decoder_mode == "inclusion_exclusion":
            self.psi_decoders = nn.ModuleDict(
                {
                    "1": _mlp(hidden * 4, hidden, hidden),
                    "2": _mlp(hidden * 4, hidden, hidden),
                }
            )
            self.output_heads = nn.ModuleDict(
                {
                    role: nn.Linear(hidden, len(channel_names))
                    for role, channel_names in self.output_channel_names.items()
                }
            )
        else:
            # A direct delta gate prevents a small design displacement from
            # disappearing inside the difference of nearly equal deep-MLP
            # outputs. Pair factors retain additive donor terms plus a
            # bilinear cross term; the order-2 inclusion-exclusion stencil
            # cancels the additive terms exactly.
            self.delta_gated_pair_change_encoder = nn.Sequential(
                nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
            )
            self.delta_gated_unary_decoders = nn.ModuleDict(
                {
                    role: _mlp(hidden * 4, hidden, self.delta_dim * len(channel_names))
                    for role, channel_names in self.output_channel_names.items()
                }
            )
            self.delta_gated_pair_decoders = nn.ModuleDict(
                {
                    role: _mlp(hidden * 4, hidden, self.delta_dim * self.delta_dim * len(channel_names))
                    for role, channel_names in self.output_channel_names.items()
                }
            )
            for decoder in (*self.delta_gated_unary_decoders.values(), *self.delta_gated_pair_decoders.values()):
                final_layer = decoder[-1]
                if isinstance(final_layer, nn.Linear):
                    nn.init.zeros_(final_layer.weight)
                    nn.init.ones_(final_layer.bias)

        self._response_scale_buffer_by_role: dict[str, str] = {}
        for index, (role, channel_names) in enumerate(self.output_channel_names.items()):
            buffer_name = f"response_output_scale_{index}"
            self.register_buffer(
                buffer_name,
                torch.ones(len(channel_names)),
                persistent=self.decoder_mode == "delta_gated",
            )
            self._response_scale_buffer_by_role[role] = buffer_name

    def set_response_scales(self, scales: Mapping[str, torch.Tensor | Sequence[float] | float]) -> None:
        """Set fixed, train-only channel scales used to restore physical units.

        This is enabled only for the delta-gated decoder, which predicts
        normalized amplitudes before applying the measured scale. The scale
        tensors are buffers, never trainable parameters.
        """

        if self.decoder_mode != "delta_gated":
            raise ValueError("Fixed response scales are only used by the delta-gated decoder.")
        if set(scales) != set(self.output_channel_names):
            raise ValueError("Response scales must be provided for every configured output role.")
        for role, channel_names in self.output_channel_names.items():
            value = torch.as_tensor(scales[role], device=next(self.parameters()).device, dtype=next(self.parameters()).dtype)
            if value.ndim == 0:
                value = value.expand(len(channel_names))
            if value.shape != (len(channel_names),):
                raise ValueError(f"Response scale for {role!r} must be scalar or one value per output channel.")
            if not bool(torch.isfinite(value).all()) or bool((value <= 0.0).any()):
                raise ValueError(f"Response scales for {role!r} must be finite and positive.")
            getattr(self, self._response_scale_buffer_by_role[role]).copy_(value)

    def _validate_cache(self, cache: BaselineResponseCache) -> None:
        if cache.module_features.shape[-1] != self.module_feature_dim:
            raise ValueError(
                f"module_features width must be {self.module_feature_dim}, got {cache.module_features.shape[-1]}."
            )
        if cache.baseline_design.shape[-1] != self.design_dim:
            raise ValueError(f"baseline_design width must be {self.design_dim}, got {cache.baseline_design.shape[-1]}.")
        expected_context = 0 if self.accept_empty_context else self.context_dim
        if cache.baseline_context.shape[-1] != expected_context:
            raise ValueError(
                f"baseline_context width must be {expected_context}, got {cache.baseline_context.shape[-1]}."
            )
        if cache.module_features.device != next(self.parameters()).device:
            raise ValueError("Baseline cache and response operator must be on the same device.")

    def _baseline_tokens(self, cache: BaselineResponseCache) -> tuple[torch.Tensor, torch.Tensor]:
        module_design = torch.cat([cache.module_features, cache.baseline_design], dim=-1)
        module_tokens = self.baseline_module_encoder(module_design)
        if self.accept_empty_context:
            context = cache.baseline_context.new_zeros(cache.batch_size, 1)
        else:
            context = cache.baseline_context
        context_token = self.baseline_context_encoder(context)
        return module_tokens, context_token

    def _edge_context(
        self,
        cache: BaselineResponseCache,
        module_tokens: torch.Tensor,
        context_token: torch.Tensor,
        factor: ResponseFactor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
        row_indices = cache.indices_for(factor.donor_ids)
        rows = torch.as_tensor(row_indices, device=module_tokens.device, dtype=torch.long)
        donor_tokens = module_tokens.index_select(1, rows)
        donor_design = cache.baseline_design.index_select(1, rows)
        mean_token = donor_tokens.mean(dim=1)
        max_token = donor_tokens.amax(dim=1)
        if len(row_indices) == 2:
            pair_relative = torch.abs(donor_design[:, 0] - donor_design[:, 1])
            pair_token = self.pair_geometry_encoder(pair_relative)
        else:
            pair_token = torch.zeros_like(mean_token)
        order_index = torch.full((cache.batch_size,), len(row_indices), device=module_tokens.device, dtype=torch.long)
        # Order has a separate embedding from output role, using fixed learned
        # vectors so donor count is visible but donor labels are not.
        order_token = self.order_embedding(order_index)
        edge = self.edge_context_encoder(
            torch.cat([context_token, mean_token, max_token, pair_token + order_token], dim=-1)
        )
        return edge, donor_tokens, row_indices

    def _receiver_tokens(
        self,
        cache: BaselineResponseCache,
        module_tokens: torch.Tensor,
        query: ResponseQueryBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, query_count, _ = query.features.shape
        if batch != cache.batch_size:
            raise ValueError("Query batches must align with the baseline cache batch size.")
        if query.receiver_module_ids is None:
            tokens = module_tokens.new_zeros(batch, query_count, self.hidden_dim)
            present = module_tokens.new_zeros(batch, query_count, 1)
            return tokens, present
        rows = []
        present_values = []
        row_by_id = {module_id: index for index, module_id in enumerate(cache.module_ids) if module_id is not None}
        for module_id in query.receiver_module_ids:
            if module_id is None:
                rows.append(0)
                present_values.append(0.0)
            else:
                if module_id not in row_by_id:
                    raise KeyError(f"Receiver module ID {module_id!r} is absent from the baseline cache.")
                rows.append(row_by_id[module_id])
                present_values.append(1.0)
        row_tensor = torch.as_tensor(rows, device=module_tokens.device, dtype=torch.long)
        present = module_tokens.new_tensor(present_values).reshape(1, query_count, 1).expand(batch, -1, -1)
        tokens = module_tokens.index_select(1, row_tensor) * present
        return tokens, present

    def _supported_query_indices(
        self,
        factor: ResponseFactor,
        role: str,
        query: ResponseQueryBatch,
    ) -> list[int]:
        if factor.receiver_support.all_receivers:
            return list(range(query.features.shape[1]))

        support = factor.receiver_support
        query_ids = (support.receiver_query_ids_by_role or {}).get(role, ())
        module_ids = (support.receiver_module_ids_by_role or {}).get(role, ())
        query_id_set = set(query_ids)
        module_id_set = set(module_ids)
        if query_id_set and query.query_ids is None:
            raise ValueError(f"Role {role!r} needs stable query_ids for restricted query receiver support.")
        if module_id_set and query.receiver_module_ids is None:
            raise ValueError(f"Role {role!r} needs receiver_module_ids for restricted module receiver support.")
        selected = []
        for index in range(query.features.shape[1]):
            query_match = query.query_ids is not None and query.query_ids[index] in query_id_set
            module_match = query.receiver_module_ids is not None and query.receiver_module_ids[index] in module_id_set
            if query_match or module_match:
                selected.append(index)
        return selected

    def _decode_delta_gated_chunk(
        self,
        *,
        role: str,
        query_tokens: torch.Tensor,
        edge_context: torch.Tensor,
        donor_tokens: torch.Tensor,
        donor_deltas: torch.Tensor,
        role_token: torch.Tensor,
    ) -> torch.Tensor:
        """Decode normalized amplitudes behind exact unary/pair delta gates."""

        batch, variant_count, donor_count, _ = donor_deltas.shape
        query_count = query_tokens.shape[1]
        channels = len(self.output_channel_names[role])
        donor_context = self.donor_change_encoder(
            torch.cat(
                [
                    donor_tokens[:, None].expand(-1, variant_count, -1, -1),
                    donor_deltas,
                ],
                dim=-1,
            )
        )
        response = query_tokens.new_zeros(batch, variant_count, query_count, channels)
        unary_decoder = self.delta_gated_unary_decoders[role]
        for donor_index in range(donor_count):
            decoder_input = torch.cat(
                [
                    query_tokens[:, None].expand(-1, variant_count, -1, -1),
                    edge_context[:, None, None].expand(-1, variant_count, query_count, -1),
                    donor_context[:, :, donor_index, None].expand(-1, -1, query_count, -1),
                    role_token.expand(batch, variant_count, query_count, -1),
                ],
                dim=-1,
            )
            amplitude = unary_decoder(decoder_input).reshape(
                batch, variant_count, query_count, self.delta_dim, channels
            )
            response = response + (
                amplitude * donor_deltas[:, :, donor_index, None, :, None]
            ).sum(dim=-2)

        if donor_count == 2:
            pair_context = self.delta_gated_pair_change_encoder(
                torch.cat(
                    [
                        donor_context.mean(dim=2),
                        torch.abs(donor_context[:, :, 0] - donor_context[:, :, 1]),
                    ],
                    dim=-1,
                )
            )
            pair_decoder_input = torch.cat(
                [
                    query_tokens[:, None].expand(-1, variant_count, -1, -1),
                    edge_context[:, None, None].expand(-1, variant_count, query_count, -1),
                    pair_context[:, :, None].expand(-1, -1, query_count, -1),
                    role_token.expand(batch, variant_count, query_count, -1),
                ],
                dim=-1,
            )
            pair_amplitude = self.delta_gated_pair_decoders[role](pair_decoder_input).reshape(
                batch,
                variant_count,
                query_count,
                self.delta_dim,
                self.delta_dim,
                channels,
            )
            first_delta = donor_deltas[:, :, 0]
            second_delta = donor_deltas[:, :, 1]
            cross_gate = first_delta[:, :, None, :, None] * second_delta[:, :, None, None, :]
            response = response + (
                pair_amplitude * cross_gate.unsqueeze(-1)
            ).sum(dim=(-3, -2))

        output_scale = getattr(self, self._response_scale_buffer_by_role[role]).to(
            device=response.device, dtype=response.dtype
        )
        return response * output_scale.reshape(1, 1, 1, channels)

    def _predict_role(
        self,
        *,
        cache: BaselineResponseCache,
        factor: ResponseFactor,
        role: str,
        query: ResponseQueryBatch,
        module_tokens: torch.Tensor,
        donor_tokens: torch.Tensor,
        edge_context: torch.Tensor,
        donor_deltas: torch.Tensor,
        query_chunk_size: int | None,
    ) -> torch.Tensor:
        expected_dim = self.query_feature_dims[role]
        if query.features.shape[-1] != expected_dim:
            raise ValueError(
                f"Query role {role!r} feature width must be {expected_dim}, got {query.features.shape[-1]}."
            )
        if query.features.device != donor_tokens.device or query.features.dtype != donor_tokens.dtype:
            raise ValueError("Query tensors and baseline cache must share device and dtype.")
        selected_positions = self._supported_query_indices(factor, role, query)
        query_indices = torch.as_tensor(selected_positions, device=query.features.device, dtype=torch.long)
        batch, full_query_count, _ = query.features.shape
        output_dim = len(self.output_channel_names[role])
        if not selected_positions:
            return query.features.new_zeros(batch, full_query_count, output_dim)

        selected_query = query.features.index_select(1, query_indices)
        selected_receiver_ids = None
        if query.receiver_module_ids is not None:
            selected_receiver_ids = tuple(query.receiver_module_ids[index] for index in selected_positions)
        selected_query_ids = None
        if query.query_ids is not None:
            selected_query_ids = tuple(query.query_ids[index] for index in selected_positions)
        selected = ResponseQueryBatch(
            features=selected_query,
            receiver_module_ids=selected_receiver_ids,
            query_ids=selected_query_ids,
        )
        _, variant_count, donor_count, _ = donor_deltas.shape
        subset_bits_list = [[(subset >> donor) & 1 for donor in range(donor_count)] for subset in range(variant_count)]
        signs = torch.tensor(
            [(-1.0) ** (donor_count - sum(bits)) for bits in subset_bits_list],
            device=donor_deltas.device,
            dtype=donor_deltas.dtype,
        )
        # Inclusion-exclusion variants share identical baseline donor/query
        # tokens, context, graph and receiver support.
        role_index = torch.tensor(self.role_to_index[role], device=donor_tokens.device, dtype=torch.long)
        role_token = self.role_embedding(role_index).reshape(1, 1, 1, self.hidden_dim)
        chunk = int(selected.features.shape[1] if query_chunk_size is None else query_chunk_size)
        if chunk <= 0:
            raise ValueError("query_chunk_size must be positive when supplied.")
        outputs = []
        if self.decoder_mode == "inclusion_exclusion":
            variant_donor_tokens = donor_tokens[:, None].expand(-1, variant_count, -1, -1)
            delta_tokens = self.donor_change_encoder(
                torch.cat([variant_donor_tokens, donor_deltas], dim=-1)
            ).sum(dim=2)
            psi = self.psi_decoders[str(donor_count)]
            head = self.output_heads[role]
        for start in range(0, selected.features.shape[1], chunk):
            stop = min(start + chunk, selected.features.shape[1])
            query_part = ResponseQueryBatch(
                features=selected.features[:, start:stop],
                receiver_module_ids=(
                    None if selected.receiver_module_ids is None else selected.receiver_module_ids[start:stop]
                ),
                query_ids=None if selected.query_ids is None else selected.query_ids[start:stop],
            )
            receiver_tokens, receiver_present = self._receiver_tokens(cache, module_tokens, query_part)
            query_encoded = self.query_encoders[role](
                torch.cat([query_part.features, receiver_tokens, receiver_present], dim=-1)
            )
            if self.decoder_mode == "delta_gated":
                variant_response = self._decode_delta_gated_chunk(
                    role=role,
                    query_tokens=query_encoded,
                    edge_context=edge_context,
                    donor_tokens=donor_tokens,
                    donor_deltas=donor_deltas,
                    role_token=role_token,
                )
                outputs.append((variant_response * signs[None, :, None, None]).sum(dim=1))
                continue
            query_count = query_encoded.shape[1]
            psi_input = torch.cat(
                [
                    query_encoded[:, None].expand(-1, variant_count, -1, -1),
                    edge_context[:, None, None].expand(-1, variant_count, query_count, -1),
                    delta_tokens[:, :, None].expand(-1, -1, query_count, -1),
                    role_token.expand(batch, variant_count, query_count, -1),
                ],
                dim=-1,
            )
            decoded = head(psi(psi_input))
            outputs.append((decoded * signs[None, :, None, None]).sum(dim=1))
        selected_output = torch.cat(outputs, dim=1)
        if len(selected_positions) == full_query_count:
            return selected_output
        return query.features.new_zeros(batch, full_query_count, output_dim).index_copy(
            dim=1, index=query_indices, source=selected_output
        )

    def _predict_factor_with_tokens(
        self,
        cache: BaselineResponseCache,
        factor: ResponseFactor,
        delta_by_module_id: Mapping[Hashable, torch.Tensor],
        queries: ResponseQueries,
        module_tokens: torch.Tensor,
        context_token: torch.Tensor,
        *,
        query_chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        edge_context, donor_tokens, _ = self._edge_context(cache, module_tokens, context_token, factor)
        model_donor_ids = factor.donor_ids
        if self.decoder_mode == "delta_gated" and len(model_donor_ids) == 2:
            model_donor_ids = tuple(
                sorted(model_donor_ids, key=lambda value: (type(value).__qualname__, repr(value)))
            )
            canonical_rows = torch.as_tensor(
                cache.indices_for(model_donor_ids), device=module_tokens.device, dtype=torch.long
            )
            donor_tokens = module_tokens.index_select(1, canonical_rows)
        donor_rows = []
        for donor_id in model_donor_ids:
            if donor_id not in delta_by_module_id:
                raise KeyError(f"Trial delta for factor donor {donor_id!r} is missing.")
            delta = delta_by_module_id[donor_id]
            if delta.shape != (cache.batch_size, self.delta_dim):
                raise ValueError(f"Delta for donor {donor_id!r} must have shape [{cache.batch_size},{self.delta_dim}].")
            if delta.device != cache.module_features.device or delta.dtype != cache.module_features.dtype:
                raise ValueError("Trial deltas and baseline cache must share device and dtype.")
            donor_rows.append(delta)
        donor_delta = torch.stack(donor_rows, dim=1)
        variant_count = 1 << len(model_donor_ids)
        subset_bits = torch.tensor(
            [[(subset >> donor) & 1 for donor in range(len(model_donor_ids))] for subset in range(variant_count)],
            device=donor_delta.device,
            dtype=donor_delta.dtype,
        )
        variant_deltas = donor_delta[:, None] * subset_bits[None, :, :, None]

        result: dict[str, torch.Tensor] = {}
        for role in factor.output_roles:
            if role not in self.output_channel_names:
                raise KeyError(f"Factor {factor.factor_id!r} declares unknown output role {role!r}.")
            if role not in queries:
                continue
            query = queries[role]
            if query.features.shape[0] != cache.batch_size:
                raise ValueError(f"Query role {role!r} does not align with the baseline cache batch size.")
            result[role] = self._predict_role(
                cache=cache,
                factor=factor,
                role=role,
                query=query,
                module_tokens=module_tokens,
                donor_tokens=donor_tokens,
                edge_context=edge_context,
                donor_deltas=variant_deltas,
                query_chunk_size=query_chunk_size,
            )
        return result

    def predict_factor_response(
        self,
        cache: BaselineResponseCache,
        factor: ResponseFactor,
        delta_by_module_id: Mapping[Hashable, torch.Tensor],
        queries: ResponseQueries,
        *,
        query_chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict one factor's signed response at its declared output roles.

        The mapping may contain deltas for other modules; those values are
        never read. No perturbed layout, dense trial field, case identifier, or
        inverse objective is accepted by this method.
        """

        self._validate_cache(cache)
        module_tokens, context_token = self._baseline_tokens(cache)
        return self._predict_factor_with_tokens(
            cache,
            factor,
            delta_by_module_id,
            queries,
            module_tokens,
            context_token,
            query_chunk_size=query_chunk_size,
        )

    def predict_response(
        self,
        cache: BaselineResponseCache,
        factors: Sequence[ResponseFactor],
        delta_by_module_id: Mapping[Hashable, torch.Tensor],
        queries: ResponseQueries,
        *,
        factor_weights: Mapping[str, float | torch.Tensor] | None = None,
        query_chunk_size: int | None = None,
        skip_inactive: bool = True,
        return_evaluated_factor_ids: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], tuple[str, ...]]:
        """Sum selected response factors; an empty graph returns exact zeros."""

        self._validate_cache(cache)
        if len({factor.factor_id for factor in factors}) != len(factors):
            raise ValueError("factor_id values must be unique within a response graph.")
        unknown_query_roles = set(queries) - set(self.output_channel_names)
        if unknown_query_roles:
            raise KeyError(f"Unconfigured response query roles: {sorted(unknown_query_roles)!r}.")
        outputs = {
            role: query.features.new_zeros(
                query.features.shape[0], query.features.shape[1], len(self.output_channel_names[role])
            )
            for role, query in queries.items()
        }
        evaluated: list[str] = []
        # Prepare the immutable baseline once for the complete response call;
        # all factors and all anchored sign variants share these tensors.
        module_tokens, context_token = self._baseline_tokens(cache)
        for factor in factors:
            raw_weight = 1.0 if factor_weights is None else factor_weights.get(factor.factor_id, 1.0)
            weight = torch.as_tensor(raw_weight, device=cache.module_features.device, dtype=cache.module_features.dtype)
            if weight.ndim not in (0, 1) or (weight.ndim == 1 and weight.shape[0] != cache.batch_size):
                raise ValueError("Each factor weight must be a scalar or a per-example [B] tensor.")
            if skip_inactive and not bool(torch.count_nonzero(weight).item()):
                continue
            factor_output = self._predict_factor_with_tokens(
                cache,
                factor,
                delta_by_module_id,
                queries,
                module_tokens,
                context_token,
                query_chunk_size=query_chunk_size,
            )
            evaluated.append(factor.factor_id)
            scale = weight.reshape(cache.batch_size, 1, 1) if weight.ndim == 1 else weight
            for role, value in factor_output.items():
                outputs[role] = outputs[role] + value * scale
        if return_evaluated_factor_ids:
            return outputs, tuple(evaluated)
        return outputs

    def validity_status(
        self,
        cache: BaselineResponseCache,
        factor: ResponseFactor,
        delta_by_module_id: Mapping[Hashable, torch.Tensor],
        *,
        atol: float = 1.0e-6,
    ) -> bool | None:
        """Return validity, or ``None`` when the evidence neighborhood is unknown."""

        self._validate_cache(cache)
        neighborhood = factor.validity
        known = True
        for donor_id in factor.donor_ids:
            radii = neighborhood.max_abs_delta_by_module.get(donor_id)
            if radii is None:
                known = False
                continue
            delta = delta_by_module_id[donor_id]
            if len(radii) != self.delta_dim:
                raise ValueError(f"Validity radius for {donor_id!r} must have {self.delta_dim} coordinates.")
            if delta.shape != (cache.batch_size, self.delta_dim):
                raise ValueError(f"Delta for {donor_id!r} must have shape [{cache.batch_size},{self.delta_dim}].")
            for coordinate, radius in enumerate(radii):
                if radius is None:
                    known = False
                elif bool((delta[:, coordinate].abs() > radius + atol).any()):
                    return False
            centers = neighborhood.baseline_design_center_by_module or {}
            center = centers.get(donor_id)
            if center is None:
                known = False
            else:
                if len(center) != self.design_dim:
                    raise ValueError(
                        f"Baseline design center for {donor_id!r} must have {self.design_dim} coordinates."
                    )
                row = cache.indices_for((donor_id,))[0]
                if bool(
                    (cache.baseline_design[:, row] - cache.baseline_design.new_tensor(center)).abs().gt(atol).any()
                ):
                    return False
        if neighborhood.context_radius is None or not neighborhood.context_center:
            known = False
        else:
            if len(neighborhood.context_center) != self.context_dim:
                raise ValueError("context_center width does not match the configured baseline context.")
            if self.accept_empty_context:
                known = False
            else:
                center = cache.baseline_context.new_tensor(neighborhood.context_center)
                distance = torch.linalg.vector_norm(cache.baseline_context - center, dim=-1)
                if bool((distance > neighborhood.context_radius + atol).any()):
                    return False
        return True if known else None

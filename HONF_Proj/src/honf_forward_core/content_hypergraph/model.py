"""Shared A-D/A-H6 content-hypergraph forward core."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from .attention import TypedAttentionBlock
from .contracts import CoreOutput, PreparedOrganization, QueryBatch, SceneBatch
from .decoding import (
    DirectSourceReader,
    EdgeQueryReader,
    FieldHead,
    LocalDetailReader,
    LocalPairPack,
    QueryEncoder,
)
from .organization import HypergraphOrganizer, SourceCoefficientBank


class ContentHypergraphCore(nn.Module):
    """New typed-attention family with a direct A-D and organized A-H6 arm.

    Dataset adapters own physical feature construction, coordinate frames,
    units, normalization, and native observations. The shared core receives
    only transformed prescribed tensors and returns model-space fields.

    A-D omits the final environment destination update because no downstream
    block consumes that state; its final simultaneous module update still
    reads the same pre-update environment states as a full MM/ME/EM block.
    A-H6 keeps the final environment update because its organizer consumes it.
    """

    def __init__(
        self,
        module_input_dim: int,
        environment_input_dim: int,
        coordinate_dim: int,
        query_input_dim: int = 0,
        operating_input_dim: int = 0,
        output_dim: int = 3,
        architecture: str = "A-H6",
        context_dim: int = 128,
        num_heads: int = 4,
        context_depth: int = 2,
        num_hyperedges: int = 6,
        thermal_response_rank: int = 4,
        local_hidden_dim: int = 64,
        dropout: float = 0.0,
        init_seed: int = 0,
        final_environment_update: bool | None = None,
    ) -> None:
        super().__init__()
        architecture = str(architecture).upper()
        if architecture not in {"A-D", "A-H6"}:
            raise ValueError("architecture must be 'A-D' or 'A-H6'.")
        if (
            min(
                module_input_dim,
                environment_input_dim,
                coordinate_dim,
                output_dim,
                context_dim,
                num_heads,
                context_depth,
                local_hidden_dim,
            )
            < 1
        ):
            raise ValueError("Feature widths, coordinate_dim, depth, and head counts must be positive.")
        if context_dim % num_heads:
            raise ValueError("context_dim must be divisible by num_heads.")
        if query_input_dim < 0 or operating_input_dim < 0 or thermal_response_rank < 0:
            raise ValueError("Optional feature widths and thermal rank cannot be negative.")
        if architecture == "A-H6" and num_hyperedges < 1:
            raise ValueError("A-H6 needs at least one hyperedge.")
        expected_final_environment_update = architecture == "A-H6"
        if final_environment_update is not None and bool(final_environment_update) != expected_final_environment_update:
            raise ValueError("final_environment_update is fixed by the selected architecture.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0,1).")

        self.module_input_dim = int(module_input_dim)
        self.environment_input_dim = int(environment_input_dim)
        self.coordinate_dim = int(coordinate_dim)
        self.query_input_dim = int(query_input_dim)
        self.operating_input_dim = int(operating_input_dim)
        self.output_dim = int(output_dim)
        self.architecture = architecture
        self.final_environment_update = expected_final_environment_update
        self.context_dim = int(context_dim)
        self.num_heads = int(num_heads)
        self.context_depth = int(context_depth)
        self.num_hyperedges = int(num_hyperedges) if architecture == "A-H6" else 0
        self.thermal_response_rank = int(thermal_response_rank)
        self.local_hidden_dim = int(local_hidden_dim)
        self.dropout_probability = float(dropout)
        self.init_seed = int(init_seed)
        self._owner_token = id(self)

        # Restore the caller's RNG state after constructors; final parameter
        # values are then initialized from stable names below.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.module_encoder = nn.Linear(self.module_input_dim, self.context_dim)
            self.environment_encoder = nn.Linear(self.environment_input_dim, self.context_dim)
            self.operating_encoder = (
                nn.Linear(self.operating_input_dim, self.context_dim) if self.operating_input_dim else None
            )
            self.context_blocks = nn.ModuleList(
                [
                    TypedAttentionBlock(
                        self.context_dim,
                        self.num_heads,
                        self.coordinate_dim,
                        self.dropout_probability,
                        update_environment=(self.final_environment_update or block_index < self.context_depth - 1),
                    )
                    for block_index in range(self.context_depth)
                ]
            )
            self.query_encoder = QueryEncoder(self.query_input_dim, self.coordinate_dim, self.context_dim)
            self.local_reader = LocalDetailReader(
                self.context_dim,
                self.coordinate_dim,
                self.local_hidden_dim,
                thermal=self.thermal_response_rank > 0,
            )
            self.field_head = FieldHead(self.context_dim, self.local_hidden_dim, self.output_dim)

            if architecture == "A-H6":
                self.organizer = HypergraphOrganizer(self.context_dim, self.num_hyperedges, self.coordinate_dim)
                self.edge_reader = EdgeQueryReader(
                    self.context_dim,
                    self.coordinate_dim,
                    self.local_hidden_dim,
                    self.thermal_response_rank,
                )
                self.coefficient_bank = (
                    SourceCoefficientBank(self.context_dim, self.coordinate_dim, self.thermal_response_rank)
                    if self.thermal_response_rank
                    else None
                )
                self.direct_reader = None
            else:
                # A-D has no dormant edge or coefficient parameters. It is the
                # explicit dense source-read control under the same contextualizer.
                self.organizer = None
                self.edge_reader = None
                self.coefficient_bank = None
                self.direct_reader = DirectSourceReader(
                    self.context_dim,
                    self.coordinate_dim,
                    self.local_hidden_dim,
                    thermal=self.thermal_response_rank > 0,
                )
        self._initialize_named_parameters()

    def _stable_seed(self, name: str) -> int:
        payload = f"{self.init_seed}:{name}".encode()
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**31 - 1)

    def _initialize_named_parameters(self) -> None:
        """Initialize parameters by stable path, independent of module order."""
        with torch.no_grad():
            initialized = set()
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(self._stable_seed(f"{name}.weight"))
                    weight = torch.empty(tuple(module.weight.shape), dtype=torch.float32, device="cpu")
                    nn.init.xavier_uniform_(weight, generator=generator)
                    if name.endswith(("geometry_bias", "geometry_score")):
                        weight.mul_(0.01)
                    module.weight.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))
                    initialized.add(id(module.weight))
                    if module.bias is not None:
                        module.bias.zero_()
                        initialized.add(id(module.bias))
                elif isinstance(module, nn.LayerNorm):
                    if module.weight is not None:
                        module.weight.fill_(1.0)
                        initialized.add(id(module.weight))
                    if module.bias is not None:
                        module.bias.zero_()
                        initialized.add(id(module.bias))

            for name, parameter in self.named_parameters():
                if id(parameter) in initialized:
                    continue
                if name.endswith("edge_seeds"):
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(self._stable_seed(name))
                    sample = torch.randn(tuple(parameter.shape), generator=generator, dtype=torch.float32)
                    parameter.copy_((0.02 * sample).to(device=parameter.device, dtype=parameter.dtype))
                # Anchor offsets and edge log-scales intentionally start at zero.

    def get_config(self) -> dict[str, Any]:
        """Return standalone, JSON-safe constructor metadata."""
        return {
            "module_input_dim": self.module_input_dim,
            "environment_input_dim": self.environment_input_dim,
            "coordinate_dim": self.coordinate_dim,
            "query_input_dim": self.query_input_dim,
            "operating_input_dim": self.operating_input_dim,
            "output_dim": self.output_dim,
            "architecture": self.architecture,
            "context_dim": self.context_dim,
            "num_heads": self.num_heads,
            "context_depth": self.context_depth,
            "num_hyperedges": self.num_hyperedges,
            "thermal_response_rank": self.thermal_response_rank,
            "local_hidden_dim": self.local_hidden_dim,
            "dropout": self.dropout_probability,
            "init_seed": self.init_seed,
            "final_environment_update": self.final_environment_update,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ContentHypergraphCore:
        return cls(**dict(config))

    def _config_key(self) -> tuple[Any, ...]:
        config = self.get_config()
        return tuple((key, config[key]) for key in sorted(config))

    def _parameter_version_stamp(self) -> tuple[tuple[int, int], ...]:
        return tuple((id(parameter), int(parameter._version)) for parameter in self.parameters())

    def _buffer_version_stamp(self) -> tuple[tuple[int, int], ...]:
        stamp = []
        for buffer in self.buffers():
            try:
                version = int(buffer._version)
            except RuntimeError:
                version = -1
            stamp.append((id(buffer), version))
        return tuple(stamp)

    def _frame(self, scene: SceneBatch, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (scene.coordinate_origin is None) != (scene.coordinate_scale is None):
            raise ValueError("coordinate_origin and coordinate_scale must be supplied together.")
        if scene.coordinate_origin is not None:
            origin = scene.coordinate_origin.to(dtype=dtype)
            scale = scene.coordinate_scale.to(dtype=dtype)
            return origin, scale
        module_coordinates = scene.module_coordinates.to(dtype=dtype)
        environment_coordinates = scene.environment_coordinates.to(dtype=dtype)
        coordinates = torch.cat((module_coordinates, environment_coordinates), dim=1)
        valid = torch.cat((scene.module_mask, scene.environment_mask), dim=1)
        positive_inf = torch.full_like(coordinates, torch.inf)
        negative_inf = torch.full_like(coordinates, -torch.inf)
        lower = torch.where(valid[:, :, None], coordinates, positive_inf).amin(dim=1)
        upper = torch.where(valid[:, :, None], coordinates, negative_inf).amax(dim=1)
        has_any = valid.any(dim=1, keepdim=True)
        lower = torch.where(has_any, lower, torch.zeros_like(lower))
        upper = torch.where(has_any, upper, torch.ones_like(upper))
        span = upper - lower
        scale = torch.where(span > 1.0e-6, span, torch.ones_like(span))
        return lower, scale

    def _operating_state(self, scene: SceneBatch, dtype: torch.dtype) -> Tensor:
        batch = scene.batch_size
        if self.operating_encoder is None:
            return scene.module_features.new_zeros(batch, self.context_dim, dtype=dtype)
        return self.operating_encoder(scene.operating_features.to(dtype=dtype))

    def _empty_organization(
        self,
        scene: SceneBatch,
        module_states: Tensor,
        environment_states: Tensor,
        operating_state: Tensor,
        frame_origin: Tensor,
        frame_scale: Tensor,
        trace: bool,
    ) -> dict[str, Tensor | None]:
        batch, modules, _ = module_states.shape
        environment_count = environment_states.shape[1]
        return {
            "edges": module_states.new_zeros(batch, 0, self.context_dim),
            "module_incidence": module_states.new_zeros(batch, modules, 0),
            "environment_incidence": module_states.new_zeros(batch, environment_count, 0),
            "module_pooling_weights": module_states.new_zeros(batch, 0, modules),
            "environment_pooling_weights": module_states.new_zeros(batch, 0, environment_count),
            "module_edge_mass": module_states.new_zeros(batch, 0),
            "environment_edge_mass": module_states.new_zeros(batch, 0),
            "edge_anchors": module_states.new_zeros(batch, 0, self.coordinate_dim),
            "edge_scales": module_states.new_zeros(batch, 0, self.coordinate_dim),
        }

    def prepare(self, scene: SceneBatch, *, return_trace: bool = False) -> PreparedOrganization:
        """Contextualize one target-free scene and prepare its organization."""
        scene.validate(
            coordinate_dim=self.coordinate_dim,
            module_input_dim=self.module_input_dim,
            environment_input_dim=self.environment_input_dim,
            operating_input_dim=self.operating_input_dim,
        )
        parameter = next(self.parameters())
        dtype = parameter.dtype
        device = parameter.device
        for value in (
            scene.module_features,
            scene.environment_features,
            scene.module_coordinates,
            scene.environment_coordinates,
            scene.module_mask,
            scene.environment_mask,
            scene.module_measure,
            scene.environment_measure,
            scene.module_ids,
            scene.environment_ids,
            scene.local_radius,
        ):
            if value.device != device:
                raise ValueError("All scene tensors must already be on the model device.")
        for value in (
            scene.operating_features,
            scene.coordinate_origin,
            scene.coordinate_scale,
            scene.module_measure_reference,
            scene.environment_measure_reference,
        ):
            if value is not None and value.device != device:
                raise ValueError("All supplied optional scene tensors must be on the model device.")

        frame_origin, frame_scale = self._frame(scene, dtype)
        operating_state = self._operating_state(scene, dtype)
        module_states = self.module_encoder(scene.module_features.to(dtype=dtype)) + operating_state[:, None, :]
        environment_states = (
            self.environment_encoder(scene.environment_features.to(dtype=dtype)) + operating_state[:, None, :]
        )
        module_states = module_states * scene.module_mask[:, :, None].to(dtype)
        environment_states = environment_states * scene.environment_mask[:, :, None].to(dtype)
        module_coordinates = scene.module_coordinates.to(dtype=dtype)
        environment_coordinates = scene.environment_coordinates.to(dtype=dtype)
        module_measure = scene.module_measure.to(dtype=dtype)
        environment_measure = scene.environment_measure.to(dtype=dtype)
        for block in self.context_blocks:
            module_states, environment_states = block(
                module_states,
                environment_states,
                module_coordinates,
                environment_coordinates,
                module_measure,
                environment_measure,
                scene.module_mask,
                scene.environment_mask,
                frame_scale,
            )

        if self.organizer is None:
            organization = self._empty_organization(
                scene, module_states, environment_states, operating_state, frame_origin, frame_scale, return_trace
            )
            source_coefficients = None
            edge_keys = None
            edge_values = None
            direct_source_keys, direct_source_values = self.direct_reader.prepare_sources(module_states)
        else:
            organization = self.organizer(
                scene,
                module_states,
                environment_states,
                operating_state,
                frame_origin,
                frame_scale,
            )
            if self.coefficient_bank is not None:
                source_coefficients = self.coefficient_bank(
                    scene,
                    module_states,
                    organization["edges"],
                    organization["module_incidence"],
                    operating_state,
                    organization["edge_anchors"],
                    frame_scale,
                )
            else:
                source_coefficients = None
            edge_keys, edge_values = self.edge_reader.prepare_edges(organization["edges"])
            direct_source_keys = None
            direct_source_values = None

        trace = None
        if return_trace:
            trace = {
                "module_incidence": organization["module_incidence"],
                "environment_incidence": organization["environment_incidence"],
                "module_pooling_weights": organization["module_pooling_weights"],
                "environment_pooling_weights": organization["environment_pooling_weights"],
                "module_edge_mass": organization["module_edge_mass"],
                "environment_edge_mass": organization["environment_edge_mass"],
            }
        return PreparedOrganization(
            scene=scene,
            module_states=module_states,
            environment_states=environment_states,
            edges=organization["edges"],
            module_incidence=organization["module_incidence"],
            environment_incidence=organization["environment_incidence"],
            module_pooling_weights=organization["module_pooling_weights"],
            environment_pooling_weights=organization["environment_pooling_weights"],
            module_edge_mass=organization["module_edge_mass"],
            environment_edge_mass=organization["environment_edge_mass"],
            edge_anchors=organization["edge_anchors"],
            edge_scales=organization["edge_scales"],
            edge_keys=edge_keys,
            edge_values=edge_values,
            direct_source_keys=direct_source_keys,
            direct_source_values=direct_source_values,
            source_coefficients=source_coefficients,
            model_owner=self._owner_token,
            model_parameter_versions=self._parameter_version_stamp(),
            model_buffer_versions=self._buffer_version_stamp(),
            scene_tensor_versions=scene.tensor_version_stamp(),
            scene_identity_snapshot=(scene.scene_key, scene.normalizer_key, scene.coordinate_frame_key),
            frame_origin=frame_origin,
            frame_scale=frame_scale,
            operating_state=operating_state,
            model_config_key=self._config_key(),
            trace=trace,
        )

    def _validate_prepared_state(self, organization: PreparedOrganization) -> None:
        if organization.model_owner != self._owner_token:
            raise ValueError("Prepared organization belongs to a different model instance.")
        if organization.model_config_key != self._config_key():
            raise ValueError("Prepared organization configuration is stale.")
        if organization.model_parameter_versions != self._parameter_version_stamp():
            raise ValueError("Model parameters changed after scene preparation; rebuild the organization.")
        if organization.model_buffer_versions != self._buffer_version_stamp():
            raise ValueError("Model buffers changed after scene preparation; rebuild the organization.")
        if organization.scene_tensor_versions != organization.scene.tensor_version_stamp():
            raise ValueError("Scene tensors changed after preparation; rebuild the organization.")
        if organization.scene_identity_snapshot != (
            organization.scene.scene_key,
            organization.scene.normalizer_key,
            organization.scene.coordinate_frame_key,
        ):
            raise ValueError("Scene identity or preprocessing keys changed after preparation.")

    def _validate_query(self, organization: PreparedOrganization, query: QueryBatch) -> None:
        query.validate(
            batch_size=organization.scene.batch_size,
            coordinate_dim=self.coordinate_dim,
            query_input_dim=self.query_input_dim,
        )
        if query.coordinates.device != organization.module_states.device:
            raise ValueError("Query tensors must be on the prepared organization's device.")
        for value in (
            query.features,
            query.query_mask,
            query.query_ids,
            query.local_batch_indices,
            query.local_query_indices,
            query.local_module_indices,
            query.local_weights,
        ):
            if value is not None and value.device != organization.module_states.device:
                raise ValueError("All supplied query tensors must be on the prepared organization's device.")
        if query.local_batch_indices is not None:
            batch_size, query_count = query.coordinates.shape[:2]
            module_count = organization.scene.num_modules
            indices_ok = (
                (query.local_batch_indices >= 0)
                & (query.local_batch_indices < batch_size)
                & (query.local_query_indices >= 0)
                & (query.local_query_indices < query_count)
                & (query.local_module_indices >= 0)
                & (query.local_module_indices < module_count)
            ).all()
            if indices_ok.device.type == "cuda":
                torch._assert_async(indices_ok, "Packed local-pair indices are outside the scene/query axes.")
            elif not bool(indices_ok):
                raise ValueError("Packed local-pair indices are outside the scene/query axes.")
        if (
            organization.scene.scene_key is not None
            and query.scene_key is not None
            and organization.scene.scene_key != query.scene_key
        ):
            raise ValueError("Query scene key does not match the prepared organization.")
        if (
            organization.scene.coordinate_frame_key is not None
            and query.coordinate_frame_key is not None
            and organization.scene.coordinate_frame_key != query.coordinate_frame_key
        ):
            raise ValueError("Query coordinate frame does not match the prepared organization.")

    def validate_prepared_state(self, organization: PreparedOrganization, query: QueryBatch) -> None:
        """Validate model/scene ownership and one complete query before tiled reads.

        Adapters may call this once before slicing a full receiver query, then
        use the private ``_decode_validated`` path for slices they create
        without mutating the scene or query tensors. Ordinary callers should
        use ``decode``, which always validates at its public boundary.
        """
        self._validate_prepared_state(organization)
        self._validate_query(organization, query)

    def _validate_prepared(self, organization: PreparedOrganization, query: QueryBatch) -> None:
        self.validate_prepared_state(organization, query)

    def _module_measure_reference(self, scene: SceneBatch, dtype: torch.dtype) -> Tensor:
        if scene.module_measure_reference is None:
            return scene.module_measure.new_ones(scene.batch_size, dtype=dtype)
        return scene.module_measure_reference.to(dtype=dtype)

    def _query_states(self, organization: PreparedOrganization, query: QueryBatch) -> Tensor:
        return self.query_encoder(
            query,
            organization.operating_state,
            organization.frame_origin,
            organization.frame_scale,
        )

    def _validate_edge_intervention(
        self,
        organization: PreparedOrganization,
        edge_gates: Tensor | None,
        uniform_access: bool,
    ) -> None:
        if self.architecture != "A-H6":
            if edge_gates is not None or uniform_access:
                raise ValueError("Edge interventions are only available for A-H6.")
            return
        if edge_gates is not None:
            expected = (organization.scene.batch_size, organization.num_hyperedges)
            if tuple(edge_gates.shape) != expected:
                raise ValueError(f"edge_gates must have shape {expected}.")
            if edge_gates.device != organization.module_states.device:
                raise ValueError("edge_gates must be on the prepared organization's device.")
            valid = torch.isfinite(edge_gates).all() & (edge_gates >= 0).all() & (edge_gates <= 1).all()
            if valid.device.type == "cuda":
                torch._assert_async(valid, "edge_gates must contain finite values in [0, 1].")
            elif not bool(valid):
                raise ValueError("edge_gates must contain finite values in [0, 1].")

    def _decode_features(
        self,
        organization: PreparedOrganization,
        query: QueryBatch,
        *,
        return_trace: bool,
        edge_gates: Tensor | None = None,
        uniform_access: bool = False,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor,
        LocalPairPack,
        dict[str, Tensor],
    ]:
        scene = organization.scene
        query_states = self._query_states(organization, query)
        local_context, local_pairs, local_temperature_coefficients = self.local_reader(
            scene,
            query,
            organization.module_states,
            query_states,
            organization.frame_scale,
            self._module_measure_reference(scene, query_states.dtype),
        )
        trace: dict[str, Tensor] = {}
        if self.architecture == "A-H6":
            global_context, learned_access, base_access, effective_access, basis, messages = self.edge_reader(
                query,
                query_states,
                organization,
                edge_gates=edge_gates,
                uniform_access=uniform_access,
            )
            direct_temperature_coefficients = None
            if return_trace:
                trace["edge_access"] = learned_access
                trace["edge_base_access"] = base_access
                trace["edge_used_access"] = effective_access
                trace["edge_gates"] = (
                    organization.edges.new_ones(organization.scene.batch_size, organization.num_hyperedges)
                    if edge_gates is None
                    else edge_gates
                )
                trace["edge_messages"] = messages
        else:
            global_context, access, direct_temperature_coefficients, messages = self.direct_reader(
                scene,
                query,
                query_states,
                organization.module_states,
                organization.frame_scale,
                self._module_measure_reference(scene, query_states.dtype),
                organization.direct_source_keys,
                organization.direct_source_values,
            )
            basis = None
            if return_trace:
                trace["direct_source_access"] = access
                trace["direct_source_messages"] = messages
        nonlinear_field = self.field_head(query_states, global_context, local_context)
        query_mask = query.query_mask
        if query_mask is not None:
            nonlinear_field = nonlinear_field * query_mask[:, :, None].to(nonlinear_field.dtype)
        if return_trace:
            trace["local_pair_batch_indices"] = local_pairs.batch_index
            trace["local_pair_query_indices"] = local_pairs.query_index
            trace["local_pair_module_indices"] = local_pairs.module_index
            trace["local_pair_support_weights"] = local_pairs.support_weight
            if local_temperature_coefficients.numel():
                trace["local_temperature_coefficients"] = local_temperature_coefficients
            if organization.trace is not None:
                trace.update(organization.trace)
        return (
            nonlinear_field,
            query_states,
            local_context,
            basis,
            direct_temperature_coefficients,
            local_temperature_coefficients,
            local_pairs,
            trace,
        )

    @staticmethod
    def _accumulation_dtype(dtype: torch.dtype | None, fallback: torch.dtype) -> torch.dtype:
        if dtype is None:
            dtype = fallback
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("accumulation_dtype must be torch.float32 or torch.float64.")
        return dtype

    @staticmethod
    def _local_apply(
        local_pairs,
        local_coefficients: Tensor,
        controls: Tensor,
        batch_size: int,
        query_count: int,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        result = controls.new_zeros(batch_size * query_count, dtype=dtype)
        if local_pairs.count == 0:
            return result.reshape(batch_size, query_count)
        b = local_pairs.batch_index
        q = local_pairs.query_index
        m = local_pairs.module_index
        values = local_pairs.support_weight.to(dtype) * local_coefficients.to(dtype) * controls[b, m].to(dtype)
        return result.index_add(0, b * query_count + q, values).reshape(batch_size, query_count)

    def _factorized_apply(
        self,
        organization: PreparedOrganization,
        query: QueryBatch,
        basis: Tensor,
        local_pairs,
        local_coefficients: Tensor,
        controls: Tensor,
        *,
        accumulation_dtype: torch.dtype | None,
    ) -> Tensor:
        if organization.source_coefficients is None:
            raise ValueError("This prepared organization has no affine source coefficient bank.")
        dtype = self._accumulation_dtype(accumulation_dtype, controls.dtype)
        coefficients = organization.source_coefficients.to(dtype=dtype)
        grouped_controls = torch.einsum("bhrm,bm->bhr", coefficients, controls.to(dtype=dtype))
        edge_response = torch.einsum("bqhr,bhr->bq", basis.to(dtype=dtype), grouped_controls)
        local_response = self._local_apply(
            local_pairs,
            local_coefficients,
            controls,
            query.batch_size,
            query.num_queries,
            dtype=dtype,
        )
        return edge_response + local_response

    def decode(
        self,
        organization: PreparedOrganization,
        query: QueryBatch,
        controls: Tensor | None = None,
        *,
        return_basis: bool = False,
        return_trace: bool = False,
        accumulation_dtype: torch.dtype | None = None,
        edge_gates: Tensor | None = None,
        uniform_access: bool = False,
    ) -> CoreOutput:
        """Decode independent receivers and optionally apply affine controls."""
        self.validate_prepared_state(organization, query)
        return self._decode_validated(
            organization,
            query,
            controls,
            return_basis=return_basis,
            return_trace=return_trace,
            accumulation_dtype=accumulation_dtype,
            edge_gates=edge_gates,
            uniform_access=uniform_access,
        )

    def _decode_validated(
        self,
        organization: PreparedOrganization,
        query: QueryBatch,
        controls: Tensor | None = None,
        *,
        return_basis: bool = False,
        return_trace: bool = False,
        accumulation_dtype: torch.dtype | None = None,
        edge_gates: Tensor | None = None,
        uniform_access: bool = False,
    ) -> CoreOutput:
        """Decode a query slice after adapter-owned full-query validation.

        This private entry point skips model/scene/query boundary scans. It is
        only for adapter loops that have just called
        ``validate_prepared_state`` on the complete query and create immutable
        slices from that exact query during the loop.
        """
        self._validate_edge_intervention(organization, edge_gates, uniform_access)
        (
            nonlinear_field,
            _query_states,
            _local_context,
            basis,
            direct_temperature_coefficients,
            local_temperature_coefficients,
            local_pairs,
            trace,
        ) = self._decode_features(
            organization,
            query,
            return_trace=return_trace,
            edge_gates=edge_gates,
            uniform_access=uniform_access,
        )
        temperature = None
        if controls is not None:
            expected = (query.batch_size, organization.scene.num_modules)
            if tuple(controls.shape) != expected:
                raise ValueError(f"controls must have shape {expected}.")
            if controls.device != organization.module_states.device:
                raise ValueError("controls must be on the prepared organization's device.")
            dtype = self._accumulation_dtype(accumulation_dtype, controls.dtype)
            if self.architecture == "A-H6":
                if basis is None:
                    raise ValueError("A-H6 needs a thermal basis reader to apply controls.")
                temperature = self._factorized_apply(
                    organization,
                    query,
                    basis,
                    local_pairs,
                    local_temperature_coefficients,
                    controls,
                    accumulation_dtype=dtype,
                )
            else:
                if direct_temperature_coefficients is None:
                    raise ValueError("This A-D model has no affine thermal response head.")
                direct = torch.einsum(
                    "bqm,bm->bq",
                    direct_temperature_coefficients.to(dtype=dtype),
                    controls.to(dtype=dtype),
                )
                local = self._local_apply(
                    local_pairs,
                    local_temperature_coefficients,
                    controls,
                    query.batch_size,
                    query.num_queries,
                    dtype=dtype,
                )
                temperature = direct + local
            if return_trace and basis is not None:
                amplitudes = torch.einsum(
                    "bhrm,bm->bhr",
                    organization.source_coefficients.to(temperature.dtype),
                    controls.to(temperature.dtype),
                )
                trace["edge_temperature_contributions"] = basis.to(temperature.dtype) * amplitudes[:, None, :, :]
        if query.query_mask is not None and temperature is not None:
            temperature = temperature * query.query_mask.to(temperature.dtype)
        if return_trace and direct_temperature_coefficients is not None:
            trace["direct_temperature_coefficients"] = direct_temperature_coefficients
        work_counts = {
            "receiver_rows": int(query.batch_size * query.num_queries),
            "edge_rows": int(query.batch_size * query.num_queries * organization.num_hyperedges),
            "direct_source_rows": int(query.batch_size * query.num_queries * organization.scene.num_modules)
            if self.architecture == "A-D"
            else 0,
            "local_rows": int(local_pairs.count),
            "local_candidate_rows": 0
            if query.local_batch_indices is not None
            else int(query.batch_size * query.num_queries * organization.scene.num_modules),
        }
        return CoreOutput(
            nonlinear_field=nonlinear_field,
            thermal_temperature=temperature,
            thermal_basis=basis if return_basis else None,
            trace=trace if return_trace else None,
            work_counts=work_counts,
        )

    def _read_for_response(
        self,
        organization: PreparedOrganization,
        query: QueryBatch,
        *,
        edge_gates: Tensor | None = None,
        uniform_access: bool = False,
    ):
        self.validate_prepared_state(organization, query)
        self._validate_edge_intervention(organization, edge_gates, uniform_access)
        query_states = self._query_states(organization, query)
        _, local_pairs, local_coefficients = self.local_reader(
            organization.scene,
            query,
            organization.module_states,
            query_states,
            organization.frame_scale,
            self._module_measure_reference(organization.scene, query_states.dtype),
        )
        if self.architecture == "A-H6":
            _, _, _, _, basis, _ = self.edge_reader(
                query,
                query_states,
                organization,
                edge_gates=edge_gates,
                uniform_access=uniform_access,
            )
            return basis, None, local_pairs, local_coefficients
        _, _, direct, _ = self.direct_reader(
            organization.scene,
            query,
            query_states,
            organization.module_states,
            organization.frame_scale,
            self._module_measure_reference(organization.scene, query_states.dtype),
            organization.direct_source_keys,
            organization.direct_source_values,
        )
        return None, direct, local_pairs, local_coefficients

    def apply_increment(
        self,
        organization: PreparedOrganization,
        query: QueryBatch,
        delta_controls: Tensor,
        *,
        accumulation_dtype: torch.dtype | None = None,
        edge_gates: Tensor | None = None,
        uniform_access: bool = False,
    ) -> Tensor:
        """Apply a finite affine control increment without exporting dense K."""
        expected = (query.batch_size, organization.scene.num_modules)
        if tuple(delta_controls.shape) != expected:
            raise ValueError(f"delta_controls must have shape {expected}.")
        if delta_controls.device != organization.module_states.device:
            raise ValueError("delta_controls must be on the prepared organization's device.")
        basis, direct, local_pairs, local_coefficients = self._read_for_response(
            organization, query, edge_gates=edge_gates, uniform_access=uniform_access
        )
        if self.architecture == "A-H6":
            if basis is None:
                raise ValueError("A-H6 needs a thermal basis reader to apply increments.")
            return self._factorized_apply(
                organization,
                query,
                basis,
                local_pairs,
                local_coefficients,
                delta_controls,
                accumulation_dtype=accumulation_dtype,
            )
        if direct is None:
            raise ValueError("This A-D model has no affine thermal response head.")
        dtype = self._accumulation_dtype(accumulation_dtype, delta_controls.dtype)
        direct_response = torch.einsum("bqm,bm->bq", direct.to(dtype), delta_controls.to(dtype))
        return direct_response + self._local_apply(
            local_pairs,
            local_coefficients,
            delta_controls,
            query.batch_size,
            query.num_queries,
            dtype=dtype,
        )

    def heat_jvp(
        self,
        organization: PreparedOrganization,
        query: QueryBatch,
        tangent: Tensor,
        *,
        accumulation_dtype: torch.dtype | None = None,
        edge_gates: Tensor | None = None,
        uniform_access: bool = False,
    ) -> Tensor:
        """Exact control JVP for the declared affine Thermal capability."""
        return self.apply_increment(
            organization,
            query,
            tangent,
            accumulation_dtype=accumulation_dtype,
            edge_gates=edge_gates,
            uniform_access=uniform_access,
        )

    def prepare_affine_query(self, organization, query, *, edge_gates=None, uniform_access=False):
        """Opt in to spatial-factor reuse for repeated affine applications."""
        from .response import prepare_affine_query
        return prepare_affine_query(self, organization, query, edge_gates=edge_gates,
                                    uniform_access=uniform_access)

    def apply_prepared_affine(self, prepared_query, controls, *, accumulation_dtype=None):
        """Apply retained spatial factors without repeating neural readers."""
        from .response import apply_prepared_affine
        return apply_prepared_affine(self, prepared_query, controls, accumulation_dtype=accumulation_dtype)

    def heat_vjp_prepared_affine(self, prepared_query, cotangent, *, accumulation_dtype=None):
        """Transpose product of retained spatial factors without exporting K."""
        from .response import heat_vjp_prepared_affine
        return heat_vjp_prepared_affine(self, prepared_query, cotangent, accumulation_dtype=accumulation_dtype)

    def heat_vjp(
        self,
        organization: PreparedOrganization,
        query: QueryBatch,
        cotangent: Tensor,
        *,
        accumulation_dtype: torch.dtype | None = None,
        edge_gates: Tensor | None = None,
        uniform_access: bool = False,
    ) -> Tensor:
        """Exact transpose product of the affine control operator."""
        expected = (query.batch_size, query.num_queries)
        if tuple(cotangent.shape) != expected:
            raise ValueError(f"cotangent must have shape {expected}.")
        if cotangent.device != organization.module_states.device:
            raise ValueError("cotangent must be on the prepared organization's device.")
        basis, direct, local_pairs, local_coefficients = self._read_for_response(
            organization, query, edge_gates=edge_gates, uniform_access=uniform_access
        )
        dtype = self._accumulation_dtype(accumulation_dtype, cotangent.dtype)
        if self.architecture == "A-H6":
            if basis is None or organization.source_coefficients is None:
                raise ValueError("A-H6 needs source and spatial factors for a heat VJP.")
            weighted_modes = torch.einsum("bq,bqhr->bhr", cotangent.to(dtype), basis.to(dtype))
            gradient = torch.einsum("bhr,bhrm->bm", weighted_modes, organization.source_coefficients.to(dtype))
        else:
            if direct is None:
                raise ValueError("This A-D model has no affine thermal response head.")
            gradient = torch.einsum("bq,bqm->bm", cotangent.to(dtype), direct.to(dtype))
        if local_pairs.count:
            b = local_pairs.batch_index
            q = local_pairs.query_index
            m = local_pairs.module_index
            contribution = (
                cotangent[b, q].to(dtype) * local_pairs.support_weight.to(dtype) * local_coefficients.to(dtype)
            )
            gradient = (
                gradient.reshape(-1)
                .index_add(0, b * organization.scene.num_modules + m, contribution)
                .reshape(query.batch_size, organization.scene.num_modules)
            )
        return gradient

    def export_source_kernels(
        self,
        organization: PreparedOrganization,
        query: QueryBatch,
        *,
        accumulation_dtype: torch.dtype | None = None,
        edge_gates: Tensor | None = None,
        uniform_access: bool = False,
    ) -> Tensor:
        """Explicitly materialize dense ``[B,Q,M]`` source kernels."""
        basis, direct, local_pairs, local_coefficients = self._read_for_response(
            organization, query, edge_gates=edge_gates, uniform_access=uniform_access
        )
        dtype = self._accumulation_dtype(accumulation_dtype, organization.module_states.dtype)
        if self.architecture == "A-H6":
            if basis is None or organization.source_coefficients is None:
                raise ValueError("A-H6 needs source and spatial factors to export kernels.")
            kernels = torch.einsum("bqhr,bhrm->bqm", basis.to(dtype), organization.source_coefficients.to(dtype))
        else:
            if direct is None:
                raise ValueError("This A-D model has no affine thermal response head.")
            kernels = direct.to(dtype)
        if local_pairs.count:
            batch_size, query_count = query.coordinates.shape[:2]
            module_count = organization.scene.num_modules
            flat = kernels.reshape(-1)
            indices = (
                local_pairs.batch_index * query_count + local_pairs.query_index
            ) * module_count + local_pairs.module_index
            values = local_pairs.support_weight.to(dtype) * local_coefficients.to(dtype)
            kernels = flat.index_add(0, indices, values).reshape(batch_size, query_count, module_count)
        return kernels

"""Versioned geometry/content implementation with exact affine contracts."""

from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .content_bank_v2 import ContentBankV2, SourceCoefficientBankV2
from .contracts import CoreOutput, PreparedOrganizationV2, QueryBatch, SceneBatch
from .decoding import LocalPairPack
from .readers_v2 import (
    DirectSourceReaderV2,
    FieldHeadV2,
    HeadwiseEdgeReaderV2,
    LocalDetailReaderV2,
    QueryEncoderV2,
)
from .spatial_features_v2 import coordinate_gamma, pair_score_features
from .staged_attention_v2 import StagedContextCycleV2


_SCHEMA = "content_hypergraph_core_v2"
_TRACE_LIMIT = 500_000
_PREP_TRACE_DEFAULTS = (
    "module_gather_weights",
    "environment_gather_weights",
    "edges",
    "edge_anchors",
    "edge_scales",
    "source_coefficients",
)
_DECODE_TRACE_DEFAULTS = (
    "edge_access",
    "edge_base_access",
    "edge_used_access",
    "edge_gates",
    "thermal_basis",
    "edge_temperature_contributions",
    "local_pair_batch_indices",
    "local_pair_query_indices",
    "local_pair_module_indices",
    "local_pair_support_weights",
    "local_temperature_coefficients",
    "direct_temperature_coefficients",
)
_DECODE_TRACE_AVAILABLE = (
    "joint_hard_support",
    "joint_environment_ids",
    "joint_record_types",
    "joint_self_membership",
    "joint_cross_membership",
    "joint_protected_mask",
    "joint_mandatory_support",
    "joint_environment_support",
    "joint_environment_access",
    "joint_environment_context",
    "joint_conditioned_query",
    "native_hard_support",
    "native_module_ids",
    "native_dispatch_seed",
    "native_dispatch_seed_ids",
    "native_membership_hard",
    "native_membership_probabilities",
    "native_protected_direct",
    "native_protected_exceptions",
    "native_group_member_ids",
    "native_group_member_valid",
    "native_group_query_ids",
    "native_group_query_valid",
    "native_seed_permission",
    "native_module_permission",
    "native_selected_batch",
    "native_selected_query",
    "native_selected_source",
    "native_selected_source_ids",
    "edge_access",
    "edge_base_access",
    "edge_used_access",
    "edge_gates",
    "edge_messages",
    "thermal_basis",
    "edge_temperature_contributions",
    "direct_source_access",
    "direct_source_messages",
    "source_read_gates",
    "source_read_support",
    "source_read_selected_batch",
    "source_read_selected_query",
    "source_read_selected_source",
    "direct_temperature_coefficients",
    "local_pair_batch_indices",
    "local_pair_query_indices",
    "local_pair_module_indices",
    "local_pair_support_weights",
    "local_temperature_coefficients",
)


def _tensor_version(value: Tensor) -> tuple[int, int, int]:
    try:
        version = int(value._version)
    except RuntimeError:
        version = -1
    return id(value), int(value.data_ptr()), version


def _query_tensor_versions(query: QueryBatch) -> tuple[tuple[int, int, int], ...]:
    values = (
        query.coordinates,
        query.features,
        query.query_mask,
        query.query_ids,
        query.local_batch_indices,
        query.local_query_indices,
        query.local_module_indices,
        query.local_weights,
        query.anchor_indices,
        query.anchor_weights,
    )
    return tuple(_tensor_version(value) for value in values if value is not None)


def _factor_versions(prepared) -> tuple[tuple[int, int, int], ...]:
    pack = prepared.local_pairs
    values = (
        prepared.organization.source_coefficients,
        prepared.basis,
        prepared.direct_coefficients,
        prepared.local_coefficients,
        prepared.nonlinear_field,
        pack.batch_index,
        pack.query_index,
        pack.module_index,
        pack.support_weight,
        prepared.edge_gates,
        prepared.local_gates,
    )
    versions = []
    for value in values:
        if isinstance(value, Tensor):
            versions.append(_tensor_version(value))
        elif value is not None:
            versions.extend(_tensor_version(item) for item in value.tensor_values().values())
    return tuple(versions)


@dataclass(frozen=True)
class PreparedAffineQueryV2:
    """Opt-in retained factors for repeated controls on one v2 query set."""

    organization: PreparedOrganizationV2
    query: QueryBatch
    basis: Tensor | None
    direct_coefficients: Any
    local_pairs: LocalPairPack
    local_coefficients: Tensor
    nonlinear_field: Tensor
    query_versions: tuple[tuple[int, int, int], ...]
    factor_versions: tuple[tuple[int, int, int], ...]
    query_identity: tuple[Any, ...]
    edge_gates: Tensor | None
    uniform_access: bool
    local_gates: Tensor | None
    architecture: str
    source_read_work: Mapping[str, int] | None = None


def _trace_request(
    return_trace: bool,
    request,
    defaults: Sequence[str],
    available: Sequence[str],
) -> tuple[set[str], int, Tensor | None, bool]:
    """Resolve selectable trace names, query indices, and a hard element cap."""
    explicit = request is not None
    indices = None
    max_elements = _TRACE_LIMIT
    if request is None or request is False:
        selected = set(defaults if return_trace else ())
    elif request is True:
        selected = set(defaults)
    elif isinstance(request, str):
        selected = {request}
    elif isinstance(request, Mapping):
        raw_include = request.get("include", defaults if return_trace else ())
        if isinstance(raw_include, str):
            selected = {raw_include}
        else:
            selected = {str(name) for name in raw_include}
        raw_cap = request.get("max_elements", _TRACE_LIMIT)
        if isinstance(raw_cap, bool) or not isinstance(raw_cap, int) or raw_cap < 1:
            raise ValueError("trace_request.max_elements must be a positive integer.")
        max_elements = min(raw_cap, _TRACE_LIMIT)
        if "query_indices" in request:
            raw_indices = torch.as_tensor(request["query_indices"], dtype=torch.long)
            if raw_indices.ndim != 1 or raw_indices.numel() > 4096:
                raise ValueError("trace_request.query_indices must be a one-dimensional selection of at most 4096 rows.")
            if raw_indices.unique().numel() != raw_indices.numel():
                raise ValueError("trace_request.query_indices cannot contain duplicates.")
            indices = raw_indices
            if not selected:
                selected = {"edge_access", "edge_messages", "edge_temperature_contributions"}
    else:
        selected = {str(name) for name in request}
    unknown = selected.difference(available)
    if unknown:
        raise ValueError(f"Unknown v2 trace names: {sorted(unknown)}.")
    return selected, max_elements, indices, explicit


class ContentHypergraphCoreV2(nn.Module):
    """A2-D/A2-H6 core with explicit geometry and a versioned state schema."""

    schema = _SCHEMA

    def __init__(
        self,
        module_input_dim: int,
        environment_input_dim: int,
        coordinate_dim: int,
        query_input_dim: int = 0,
        operating_input_dim: int = 0,
        output_dim: int = 3,
        architecture: str = "A2-H6",
        context_dim: int = 128,
        num_heads: int = 4,
        context_depth: int = 2,
        num_hyperedges: int = 6,
        thermal_response_rank: int = 4,
        local_hidden_dim: int = 128,
        dropout: float = 0.0,
        init_seed: int = 0,
        query_edge_geometry_mode: str | None = None,
        reader_execution_backend: str = "legacy",
        source_value_conditioning: str = "none",
        reader_query_chunk_size: int | None = None,
        source_read_policy: str = "full",
        source_read_executor: str = "packed",
        source_read_rank: int = 16,
        source_read_scorer_width: int = 32,
    ) -> None:
        super().__init__()
        architecture = str(architecture).strip().upper()
        if architecture not in {"A2-D", "A2-H6"}:
            raise ValueError("architecture must be 'A2-D' or 'A2-H6'.")
        if query_edge_geometry_mode not in (None, "typed_w_centroids_v1"):
            raise ValueError("Unsupported query_edge_geometry_mode.")
        if query_edge_geometry_mode is not None and architecture != "A2-H6":
            raise ValueError("query_edge_geometry_mode='typed_w_centroids_v1' is supported only by A2-H6.")
        if reader_execution_backend not in {"legacy", "reduced_projection"}:
            raise ValueError("reader_execution_backend must be 'legacy' or 'reduced_projection'.")
        if source_value_conditioning not in {"none", "source_scale"}:
            raise ValueError("source_value_conditioning must be 'none' or 'source_scale'.")
        if reader_execution_backend == "reduced_projection" and architecture != "A2-D":
            raise ValueError("reader_execution_backend='reduced_projection' is currently supported only by A2-D.")
        if source_value_conditioning != "none" and architecture != "A2-D":
            raise ValueError("source_value_conditioning='source_scale' is supported only by A2-D.")
        if reader_query_chunk_size is not None and (
            type(reader_query_chunk_size) is not int or reader_query_chunk_size < 1 or architecture != "A2-D"
        ):
            raise ValueError("reader_query_chunk_size must be positive and is supported only by A2-D.")
        if source_read_policy not in ("full", "utility", "completion") or (
            source_read_policy != "full" and architecture != "A2-D"
        ):
            raise ValueError("Selected source policies require A2-D.")
        if source_read_executor not in ("packed", "packet", "dense_masked"):
            raise ValueError("Unknown source_read_executor.")
        if source_read_rank != 16 or source_read_scorer_width != 32:
            raise ValueError("The bounded source reader uses rank16/scorer32.")
        if min(
            module_input_dim,
            environment_input_dim,
            coordinate_dim,
            output_dim,
            context_dim,
            num_heads,
            context_depth,
            local_hidden_dim,
        ) < 1:
            raise ValueError("Feature widths, coordinate_dim, context depth and head count must be positive.")
        if context_dim % num_heads or local_hidden_dim % num_heads:
            raise ValueError("Context and local widths must be divisible by num_heads.")
        if query_input_dim < 0 or operating_input_dim < 0 or thermal_response_rank < 0:
            raise ValueError("Optional feature widths and thermal response rank cannot be negative.")
        if architecture == "A2-H6" and num_hyperedges != 6:
            raise ValueError("A2-H6 has exactly six content edges.")
        if architecture == "A2-H6" and thermal_response_rank not in (0, num_heads):
            raise ValueError("A2-H6 uses one signed thermal mode per feature head.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0,1).")

        self.module_input_dim = int(module_input_dim)
        self.environment_input_dim = int(environment_input_dim)
        self.coordinate_dim = int(coordinate_dim)
        self.query_input_dim = int(query_input_dim)
        self.operating_input_dim = int(operating_input_dim)
        self.output_dim = int(output_dim)
        self.architecture = architecture
        self.context_dim = int(context_dim)
        self.num_heads = int(num_heads)
        self.context_depth = int(context_depth)
        self.num_hyperedges = 6 if architecture == "A2-H6" else 0
        self.thermal_response_rank = int(thermal_response_rank)
        self.local_hidden_dim = int(local_hidden_dim)
        self.dropout_probability = float(dropout)
        self.init_seed = int(init_seed)
        self.query_edge_geometry_mode = query_edge_geometry_mode
        self.reader_execution_backend = reader_execution_backend
        self.source_value_conditioning = source_value_conditioning
        self.reader_query_chunk_size = reader_query_chunk_size
        self.source_read_policy = source_read_policy
        self.source_read_executor = source_read_executor
        self.source_read_rank = source_read_rank
        self.source_read_scorer_width = source_read_scorer_width
        self._source_read_override = None
        self._source_read_router_generation = 0
        self._source_read_dense_warmup = False
        self._owner_token = id(self)

        module_encoder_input = self.module_input_dim + 8 * self.coordinate_dim
        environment_encoder_input = self.environment_input_dim + 8 * self.coordinate_dim
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.module_encoder = nn.Sequential(
                nn.Linear(module_encoder_input, context_dim),
                nn.SiLU(),
                nn.Linear(context_dim, context_dim),
            )
            self.environment_encoder = nn.Sequential(
                nn.Linear(environment_encoder_input, context_dim),
                nn.SiLU(),
                nn.Linear(context_dim, context_dim),
            )
            self.operating_encoder = (
                nn.Linear(operating_input_dim, context_dim) if operating_input_dim else None
            )
            self.scene_summary_net = nn.Sequential(
                nn.Linear(3 * context_dim + 3 + 2 * coordinate_dim, 2 * context_dim),
                nn.SiLU(),
                nn.Linear(2 * context_dim, context_dim),
            )
            self.module_raw_projection = nn.Linear(context_dim, context_dim, bias=False)
            self.environment_raw_projection = nn.Linear(context_dim, context_dim, bias=False)
            self.module_scene_projection = nn.Linear(context_dim, context_dim, bias=False)
            self.environment_scene_projection = nn.Linear(context_dim, context_dim, bias=False)
            self.module_raw_gate = nn.Parameter(torch.tensor(0.1))
            self.environment_raw_gate = nn.Parameter(torch.tensor(0.1))
            self.module_scene_gate = nn.Parameter(torch.tensor(0.1))
            self.environment_scene_gate = nn.Parameter(torch.tensor(0.1))
            self.context_blocks = nn.ModuleList(
                StagedContextCycleV2(context_dim, num_heads, coordinate_dim, dropout)
                for _ in range(context_depth)
            )
            self.query_encoder = QueryEncoderV2(query_input_dim, coordinate_dim, context_dim)
            self.local_reader = LocalDetailReaderV2(
                context_dim, coordinate_dim, local_hidden_dim, thermal=thermal_response_rank > 0
            )
            self.field_head = FieldHeadV2(context_dim, local_hidden_dim, output_dim)
            if architecture == "A2-H6":
                self.content_bank = ContentBankV2(
                    context_dim, coordinate_dim, edge_count=6, heads=num_heads, dropout=dropout
                )
                self.edge_reader = HeadwiseEdgeReaderV2(
                    local_hidden_dim,
                    coordinate_dim,
                    thermal_response_rank,
                    heads=num_heads,
                    query_edge_geometry_mode=query_edge_geometry_mode,
                )
                self.coefficient_bank = (
                    SourceCoefficientBankV2(context_dim, coordinate_dim, thermal_response_rank)
                    if thermal_response_rank
                    else None
                )
                self.direct_reader = None
            else:
                self.content_bank = None
                self.edge_reader = None
                self.coefficient_bank = None
                self.direct_reader = DirectSourceReaderV2(
                    local_hidden_dim,
                    coordinate_dim,
                    heads=num_heads,
                    thermal=thermal_response_rank > 0,
                    source_value_conditioning=source_value_conditioning,
                    source_read_policy=source_read_policy,
                    source_read_rank=source_read_rank,
                    source_read_scorer_width=source_read_scorer_width,
                )
        self._initialize_named_parameters()
        if self.direct_reader is not None and self.direct_reader.source_scale_projection is not None:
            # Keep the D-scale initial function equal to D0 after the shared
            # name-based initializer has assigned every ordinary parameter.
            with torch.no_grad():
                self.direct_reader.source_scale_projection.weight.zero_()

    def _stable_seed(self, name: str) -> int:
        payload = f"{self.init_seed}:{name}".encode()
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**31 - 1)

    def _initialize_named_parameters(self) -> None:
        """Name-based initialization makes common A2 arms exactly comparable."""
        with torch.no_grad():
            initialized: set[int] = set()
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(self._stable_seed(f"{name}.weight"))
                    weight = torch.empty(tuple(module.weight.shape), dtype=torch.float32, device="cpu")
                    nn.init.xavier_uniform_(weight, generator=generator)
                    if name.endswith("geometry_score.2"):
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
                elif name.endswith("anchor_offsets"):
                    parameter.zero_()
                elif name.endswith("raw_scale_fraction"):
                    parameter.fill_(math.log(0.2 / 0.8))
                elif name.endswith("_gate"):
                    parameter.fill_(0.1)
                # Bounded attention temperatures are initialized in their
                # modules and remain live tensors throughout training.

    def get_config(self) -> dict[str, Any]:
        config = {
            "schema": self.schema,
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
        }
        if self.query_edge_geometry_mode is not None:
            config["query_edge_geometry_mode"] = self.query_edge_geometry_mode
        if self.reader_execution_backend != "legacy":
            config["reader_execution_backend"] = self.reader_execution_backend
        if self.source_value_conditioning != "none":
            config["source_value_conditioning"] = self.source_value_conditioning
        if self.reader_query_chunk_size is not None:
            config["reader_query_chunk_size"] = self.reader_query_chunk_size
        if self.source_read_policy != "full":
            config.update(source_read_policy=self.source_read_policy, source_read_executor=self.source_read_executor,
                          source_read_rank=self.source_read_rank, source_read_scorer_width=self.source_read_scorer_width)
        return config

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ContentHypergraphCoreV2:
        values = dict(config)
        schema = values.pop("schema", None)
        if schema != cls.schema:
            raise ValueError(f"Expected {cls.schema!r} config schema, got {schema!r}.")
        return cls(**values)

    def _config_key(self) -> tuple[Any, ...]:
        config = self.get_config()
        if self.source_read_policy != "full" or self._source_read_override is not None or self._source_read_router_generation:
            config["source_read_router_generation"] = self._source_read_router_generation
            config["source_read_override"] = self._source_read_override
        return tuple((key, config[key]) for key in sorted(config))

    def _parameter_version_stamp(self) -> tuple[tuple[int, int], ...]:
        return tuple((id(parameter), int(parameter._version)) for parameter in self.parameters())

    def set_source_read_training_epoch(self, epoch: int) -> None:
        """Persist the absolute child age; epoch zero preserves a saved stage."""
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("Source-read training epoch must be a nonnegative integer.")
        if not epoch or self.source_read_policy == "full":
            return
        controller = self.direct_reader.source_read
        # 500 is the common dense checkpoint. 501 starts at lambda=0 and
        # 600 reaches lambda=1; subsequent updates remain selected.
        transition = 0. if epoch <= 501 else min(1., (epoch - 501) / 99.)
        with torch.no_grad():
            if int(controller.training_epoch) != epoch:
                controller.training_epoch.fill_(epoch)
                controller.transition_lambda.fill_(transition)

    def _uses_source_selection(self) -> bool:
        if self._source_read_override is not None:
            return self._source_read_override[0] != "full"
        if self.source_read_policy == "full":
            return False
        epoch = int(self.direct_reader.source_read.training_epoch)
        return epoch < 0 or epoch > 500

    def _validate_source_read_endpoint(self, controls: Tensor, present_mask: Tensor) -> None:
        if self.source_read_policy == "full" or not self._uses_source_selection():
            return
        if (self._source_read_override is not None
                and self._source_read_override[0] != "fitted"
                and not self._source_read_override[0].endswith("_fitted")):
            return
        present = controls[present_mask]
        envelope = self.direct_reader.source_read.heating_envelope.to(controls)
        condition = torch.isfinite(controls).all() & (present.abs() <= envelope).all()
        if condition.is_cuda:
            torch._assert_async(condition, "Sparse endpoint heat exceeds the declared TRAIN envelope; use explicit full mode.")
        elif not bool(condition):
            raise ValueError("Sparse endpoint heat exceeds the declared TRAIN envelope; use explicit full mode.")

    @contextmanager
    def source_read_override_v1(self, mode: str, *, executor: str | None = None,
                                operating_point: int | None = None):
        """Explicit read-only policy/cost control with invalidating ownership."""
        modes = {"full", "fitted", "attn95", "attn99", "nearest95", "nearest99",
                 "directional95", "directional99", "shuffle95", "shuffle99",
                 "nearest_fitted", "directional_fitted", "shuffle_fitted"}
        chosen_executor = self.source_read_executor if executor is None else executor
        if self.architecture != "A2-D" or mode not in modes:
            raise ValueError("Source-read controls apply only to A2-D and declared policies.")
        if chosen_executor not in ("packed", "packet", "dense_masked"):
            raise ValueError("Unknown source-read executor.")
        if operating_point is not None and operating_point not in (0, 1):
            raise ValueError("Only the two preregistered operating points are available.")
        previous = self._source_read_override
        self._source_read_override = (mode, chosen_executor, operating_point)
        self._source_read_router_generation += 1
        try:
            yield self
        finally:
            self._source_read_override = previous
            self._source_read_router_generation += 1

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
        if scene.coordinate_origin is None or scene.coordinate_scale is None:
            raise ValueError("content_hypergraph_core_v2 requires explicit coordinate_origin and coordinate_scale.")
        return scene.coordinate_origin.to(dtype=dtype), scene.coordinate_scale.to(dtype=dtype)

    def _check_scene(self, scene: SceneBatch) -> None:
        scene.validate(
            coordinate_dim=self.coordinate_dim,
            module_input_dim=self.module_input_dim,
            environment_input_dim=self.environment_input_dim,
            operating_input_dim=self.operating_input_dim,
        )
        if scene.module_characteristic_lengths is None:
            raise ValueError("content_hypergraph_core_v2 requires module_characteristic_lengths [B,M,D].")
        if scene.environment_characteristic_lengths is None:
            raise ValueError("content_hypergraph_core_v2 requires environment_characteristic_lengths [B,E,D].")
        if scene.coordinate_origin is None or scene.coordinate_scale is None:
            raise ValueError("content_hypergraph_core_v2 requires explicit coordinate_origin and coordinate_scale.")

    def _operating_state(self, scene: SceneBatch, dtype: torch.dtype) -> Tensor:
        if self.operating_encoder is None:
            return scene.module_features.new_zeros(scene.batch_size, self.context_dim, dtype=dtype)
        return self.operating_encoder(scene.operating_features.to(dtype=dtype))

    @staticmethod
    def _measure_summary(states: Tensor, measure: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        # Typed summaries are normalized measures. Accumulate their short
        # weighted sums in FP64 so extreme positive measures remain live in
        # the quotient while model states retain their configured dtype.
        work_dtype = torch.float64 if states.dtype == torch.float32 else states.dtype
        weights = torch.where(valid, measure.to(work_dtype), torch.zeros_like(measure, dtype=work_dtype))
        total_work = weights.sum(dim=1)
        numerator = torch.einsum("bn,bnd->bd", weights, states.to(work_dtype))
        mean = numerator / total_work[:, None].clamp_min(torch.finfo(work_dtype).tiny)
        mean = torch.where(total_work[:, None] > 0, mean, torch.zeros_like(mean))
        return mean.to(states.dtype), total_work.to(states.dtype)

    @staticmethod
    def _context_geometry_features(scene: SceneBatch, frame_scale: Tensor, dtype: torch.dtype) -> dict[str, Tensor]:
        """Build the four immutable route descriptors once for this preparation."""
        module_coordinates = scene.module_coordinates.to(dtype)
        environment_coordinates = scene.environment_coordinates.to(dtype)
        module_lengths = scene.module_characteristic_lengths.to(dtype)
        environment_lengths = scene.environment_characteristic_lengths.to(dtype)

        def descriptor(destination, donor, destination_lengths, donor_lengths):
            return pair_score_features(
                destination,
                donor,
                frame_scale,
                destination_lengths,
                donor_lengths,
            )[0]

        return {
            "mm": descriptor(module_coordinates, module_coordinates, module_lengths, module_lengths),
            "em": descriptor(environment_coordinates, module_coordinates, environment_lengths, module_lengths),
            "ee": descriptor(
                environment_coordinates, environment_coordinates, environment_lengths, environment_lengths
            ),
            "me": descriptor(module_coordinates, environment_coordinates, module_lengths, environment_lengths),
        }

    @staticmethod
    def _log_measure_ratio(total: Tensor, reference: Tensor) -> Tensor:
        total64 = total.to(torch.float64)
        reference64 = reference.to(torch.float64).clamp_min(torch.finfo(torch.float64).tiny)
        positive = total64 > 0
        log_total = torch.log(torch.where(positive, total64, torch.ones_like(total64)))
        value = torch.nn.functional.softplus(log_total - torch.log(reference64))
        return torch.where(positive, value, torch.zeros_like(value)).to(total.dtype)

    def _scene_summary(
        self,
        scene: SceneBatch,
        module_states: Tensor,
        environment_states: Tensor,
        operating_state: Tensor,
        frame_scale: Tensor,
        module_reference: Tensor,
        environment_reference: Tensor,
    ) -> Tensor:
        module_measure = scene.module_measure.to(module_states.dtype)
        environment_measure = scene.environment_measure.to(environment_states.dtype)
        module_valid = scene.module_mask & (module_measure > 0)
        environment_valid = scene.environment_mask & (environment_measure > 0)
        module_mean, module_total = self._measure_summary(module_states, module_measure, module_valid)
        environment_mean, environment_total = self._measure_summary(
            environment_states, environment_measure, environment_valid
        )
        module_size = scene.module_characteristic_lengths.to(module_states.dtype) / frame_scale[:, None, :]
        environment_size = (
            scene.environment_characteristic_lengths.to(environment_states.dtype) / frame_scale[:, None, :]
        )
        module_size_mean, _ = self._measure_summary(module_size, module_measure, module_valid)
        environment_size_mean, _ = self._measure_summary(environment_size, environment_measure, environment_valid)
        active_module_count = scene.module_mask.sum(dim=1, dtype=module_states.dtype)[:, None]
        module_log_mass = self._log_measure_ratio(module_total, module_reference)[:, None]
        environment_log_mass = self._log_measure_ratio(environment_total, environment_reference)[:, None]
        summary_input = torch.cat(
            (
                module_mean,
                environment_mean,
                operating_state,
                active_module_count,
                module_log_mass,
                environment_log_mass,
                module_size_mean,
                environment_size_mean,
            ),
            dim=-1,
        )
        return self.scene_summary_net(summary_input)

    def _initial_states(self, scene: SceneBatch, operating_state: Tensor, origin: Tensor, scale: Tensor):
        dtype = next(self.parameters()).dtype
        module_gamma = coordinate_gamma(scene.module_coordinates.to(dtype), origin, scale)
        environment_gamma = coordinate_gamma(scene.environment_coordinates.to(dtype), origin, scale)
        module_size = scene.module_characteristic_lengths.to(dtype) / scale[:, None, :]
        environment_size = scene.environment_characteristic_lengths.to(dtype) / scale[:, None, :]
        module_inputs = torch.cat((scene.module_features.to(dtype), module_gamma, module_size), dim=-1)
        environment_inputs = torch.cat((scene.environment_features.to(dtype), environment_gamma, environment_size), dim=-1)
        raw_module = self.module_encoder(module_inputs) + operating_state[:, None, :]
        raw_environment = self.environment_encoder(environment_inputs) + operating_state[:, None, :]
        raw_module = raw_module * scene.module_mask[:, :, None].to(dtype)
        raw_environment = raw_environment * scene.environment_mask[:, :, None].to(dtype)
        return raw_module, raw_environment

    def _device_values(self, scene: SceneBatch) -> tuple[Tensor, ...]:
        values = (
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
            scene.module_characteristic_lengths,
            scene.environment_characteristic_lengths,
            scene.operating_features,
            scene.coordinate_origin,
            scene.coordinate_scale,
            scene.module_measure_reference,
            scene.environment_measure_reference,
        )
        return tuple(value for value in values if value is not None)

    @staticmethod
    def _measure_references(scene: SceneBatch, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        batch = scene.batch_size
        module = scene.module_measure_reference
        environment = scene.environment_measure_reference
        if module is None:
            module = scene.module_measure.new_ones(batch)
        if environment is None:
            environment = scene.environment_measure.new_ones(batch)
        return module.to(dtype=dtype), environment.to(dtype=dtype)

    def prepare(
        self,
        scene: SceneBatch,
        *,
        return_trace: bool = False,
        trace_request=None,
    ) -> PreparedOrganizationV2:
        """Prepare live scene context and typed edge organization exactly once."""
        self._check_scene(scene)
        parameter = next(self.parameters())
        dtype, device = parameter.dtype, parameter.device
        if any(value.device != device for value in self._device_values(scene)):
            raise ValueError("All scene tensors must already be on the v2 model device.")
        frame_origin, frame_scale = self._frame(scene, dtype)
        if not torch.isfinite(frame_origin).all() or not torch.isfinite(frame_scale).all() or not bool(
            (frame_scale > 0).all()
        ):
            raise ValueError("coordinate_origin must be finite and coordinate_scale finite and positive.")
        operating_state = self._operating_state(scene, dtype)
        context_geometry_features = self._context_geometry_features(scene, frame_scale, dtype)
        raw_module_states, raw_environment_states = self._initial_states(
            scene, operating_state, frame_origin, frame_scale
        )
        module_states, environment_states = raw_module_states, raw_environment_states
        module_reference, environment_reference = self._measure_references(scene, dtype)
        use_summary = self._uses_scene_summary()
        context_value_policy = self._context_value_policy()
        scene_summary = (self._scene_summary(
            scene, module_states, environment_states, operating_state, frame_scale,
            module_reference, environment_reference,
        ) if use_summary else None)

        prep_trace_available = [
            "module_gather_weights", "environment_gather_weights", "edges", "edge_anchors", "edge_scales",
            "source_coefficients", "first_content", "mixed_content", "second_content",
            "module_weights_first", "environment_weights_first", "module_content_first",
            "environment_content_first", "module_content_second", "environment_content_second",
            "module_states", "environment_states", "raw_module_states", "raw_environment_states",
            "scene_summary", "scene_summary_initial", "edge_edge_attention",
        ]
        if self.query_edge_geometry_mode == "typed_w_centroids_v1":
            prep_trace_available.extend((
                "module_content_centroids",
                "environment_content_centroids",
                "module_content_valid",
                "environment_content_valid",
            ))
        for cycle in range(self.context_depth):
            prep_trace_available.extend(
                f"cycle{cycle}/{name}"
                for name in (
                    "module_boundary", "environment_boundary", "module_after_mm", "environment_after_em",
                    "environment_after_ee", "module_after_me", "mm_attention", "em_attention",
                    "ee_attention", "me_attention",
                )
            )
        selected, trace_cap, trace_query_indices, trace_explicit = _trace_request(
            return_trace, trace_request, _PREP_TRACE_DEFAULTS, prep_trace_available
        )
        request_attention = any(name.endswith("_attention") for name in selected)
        organization_trace: dict[str, Tensor] = {}
        if "raw_module_states" in selected:
            organization_trace["raw_module_states"] = raw_module_states
        if "raw_environment_states" in selected:
            organization_trace["raw_environment_states"] = raw_environment_states
        if "scene_summary_initial" in selected:
            organization_trace["scene_summary_initial"] = scene_summary

        native_cycle_trace = None
        applied_context_stages = []
        context_work_summaries = []
        context_stage_metadata = []
        for cycle_index, block in enumerate(self.context_blocks):
            module_boundary = module_states
            environment_boundary = environment_states
            module_states = module_states + self.module_raw_gate * self.module_raw_projection(raw_module_states)
            if use_summary:
                module_states = module_states + self.module_scene_gate * self.module_scene_projection(scene_summary)[:, None, :]
            environment_states = environment_states + self.environment_raw_gate * self.environment_raw_projection(
                raw_environment_states
            )
            if use_summary:
                environment_states = environment_states + self.environment_scene_gate * self.environment_scene_projection(
                    scene_summary
                )[:, None, :]
            module_states = module_states * scene.module_mask[:, :, None].to(dtype)
            environment_states = environment_states * scene.environment_mask[:, :, None].to(dtype)
            if f"cycle{cycle_index}/module_boundary" in selected:
                organization_trace[f"cycle{cycle_index}/module_boundary"] = module_states
            if f"cycle{cycle_index}/environment_boundary" in selected:
                organization_trace[f"cycle{cycle_index}/environment_boundary"] = environment_states
            native_capture = self._captures_native_routes() and (
                cycle_index == self.context_depth - 1 or context_value_policy is not None)
            native_options = ({
                "capture_native_routes": True,
                "native_capture_route_names": self._native_capture_route_names(),
                "cycle_index": cycle_index,
                "owner_token": self._owner_token,
                "parameter_versions": self._parameter_version_stamp(),
                "buffer_versions": self._buffer_version_stamp(),
                "scene_identity": (scene.scene_key, scene.normalizer_key, scene.coordinate_frame_key),
                "model_config_key": self._config_key(),
                "scene_tensor_versions": scene.tensor_version_stamp(),
            } if native_capture else {})
            module_states, environment_states, cycle_trace = block(
                module_states,
                environment_states,
                scene,
                frame_scale,
                return_attention=request_attention,
                geometry_features=context_geometry_features,
                **({"value_policy": context_value_policy, "cycle_index": cycle_index}
                   if context_value_policy is not None and not native_capture else
                   {"value_policy": context_value_policy} if context_value_policy is not None else {}),
                **native_options,
            )
            if context_value_policy is not None:
                from ..continuous_joint.preparation import compact_context_work, context_stage_manifest
                context_work_summaries.append(compact_context_work(cycle_trace, getattr(self, "_joint_epoch", 3000)))
                context_stage_metadata.extend(context_stage_manifest(cycle_trace))
                if return_trace or (trace_request is not None and trace_request is not False):
                    applied_context_stages.append({key: value for key, value in cycle_trace.items()
                        if key.endswith("_native_route_snapshot") or key.endswith("_applied_incidence")})
            if native_capture:
                native_cycle_trace = cycle_trace
            for name, value in cycle_trace.items():
                key = f"cycle{cycle_index}/{name}"
                if key in selected:
                    organization_trace[key] = value
            # Recompute the typed, measure-normalized scene summary at every
            # cycle boundary. G^{l+1} sees the final M/E states of this cycle.
            if use_summary:
                scene_summary = self._scene_summary(
                    scene, module_states, environment_states, operating_state, frame_scale,
                    module_reference, environment_reference,
                )

        if "module_states" in selected:
            organization_trace["module_states"] = module_states
        if "environment_states" in selected:
            organization_trace["environment_states"] = environment_states
        if "scene_summary" in selected:
            organization_trace["scene_summary"] = scene_summary

        batch, modules, _ = module_states.shape
        environment_count = environment_states.shape[1]
        content_result: dict[str, Tensor] = {}
        module_content_centroids = environment_content_centroids = None
        module_content_valid = environment_content_valid = None
        if self.architecture == "A2-H6":
            organization = self.content_bank(
                scene,
                module_states,
                environment_states,
                scene_summary,
                operating_state,
                frame_origin,
                frame_scale,
                module_reference,
                environment_reference,
                return_trace="edge_edge_attention" in selected,
            )
            content_result = organization
            edges = organization["edges"]
            module_weights = organization["module_gather_weights"]
            environment_weights = organization["environment_gather_weights"]
            anchors = organization["edge_anchors"]
            scales = organization["edge_scales"]
            source_coefficients = None
            if self.coefficient_bank is not None:
                _kappa, source_coefficients = self.coefficient_bank(
                    scene,
                    module_states,
                    edges,
                    module_weights,
                    scene_summary,
                    operating_state,
                    anchors,
                    frame_scale,
                )
            edge_keys, edge_values = self.edge_reader.prepare_edges(edges)
            if self.query_edge_geometry_mode == "typed_w_centroids_v1":
                module_coordinates = scene.module_coordinates.to(dtype=module_weights.dtype)
                environment_coordinates = scene.environment_coordinates.to(dtype=environment_weights.dtype)
                module_content_valid = (
                    scene.module_mask & (scene.module_measure > 0)
                ).any(dim=-1, keepdim=True).expand(-1, module_weights.shape[1])
                environment_content_valid = (
                    scene.environment_mask & (scene.environment_measure > 0)
                ).any(dim=-1, keepdim=True).expand(-1, environment_weights.shape[1])
                module_content_centroids = torch.bmm(module_weights, module_coordinates)
                environment_content_centroids = torch.bmm(environment_weights, environment_coordinates)
                module_content_centroids = torch.where(
                    module_content_valid[:, :, None],
                    module_content_centroids,
                    torch.zeros_like(module_content_centroids),
                )
                environment_content_centroids = torch.where(
                    environment_content_valid[:, :, None],
                    environment_content_centroids,
                    torch.zeros_like(environment_content_centroids),
                )
            direct_source_keys = direct_source_values = None
            for key in prep_trace_available:
                if key not in selected:
                    continue
                if key in organization:
                    organization_trace[key] = organization[key]
                elif key == "source_coefficients" and source_coefficients is not None:
                    organization_trace[key] = source_coefficients
                elif key == "second_content":
                    organization_trace[key] = edges
            prep_work = {
                "context_mm_rows": int(self.context_depth * scene.batch_size * modules * modules),
                "context_em_rows": int(self.context_depth * scene.batch_size * environment_count * modules),
                "context_ee_rows": int(self.context_depth * scene.batch_size * environment_count * environment_count),
                "context_me_rows": int(self.context_depth * scene.batch_size * modules * environment_count),
                "context_dense_pair_rows": int(
                    self.context_depth * scene.batch_size * (modules * modules + 2 * modules * environment_count + environment_count * environment_count)
                ),
                "context_geometry_pair_rows": int(
                    scene.batch_size * (modules * modules + 2 * modules * environment_count + environment_count * environment_count)
                ),
                "content_gather_first_pass_rows": int(scene.batch_size * 6 * (modules + environment_count)),
                "content_gather_second_pass_rows": int(scene.batch_size * 6 * (modules + environment_count)),
                "content_gather_score_rows": int(2 * scene.batch_size * 6 * (modules + environment_count)),
                "content_geometry_pair_rows": int(scene.batch_size * 6 * (modules + environment_count)),
                "edge_interaction_pair_rows": int(scene.batch_size * 6 * 6),
                "edge_interaction_head_pair_rows": int(scene.batch_size * self.num_heads * 6 * 6),
                "context_attention_trace_head_pair_rows": int(
                    scene.batch_size * self.context_depth * self.num_heads
                    * (modules * modules + 2 * modules * environment_count + environment_count * environment_count)
                ) if request_attention else 0,
                "edge_interaction_attention_trace_head_pair_rows": int(
                    scene.batch_size * self.num_heads * 6 * 6
                ) if "edge_edge_attention" in selected else 0,
                "scene_summary_module_entries": int((self.context_depth + 1) * scene.batch_size * modules),
                "scene_summary_environment_entries": int((self.context_depth + 1) * scene.batch_size * environment_count),
                "raw_reinjection_rows": int(self.context_depth * scene.batch_size * (modules + environment_count)),
                "scene_reinjection_rows": int(self.context_depth * scene.batch_size),
                "source_coefficient_entries": int(0 if source_coefficients is None else source_coefficients.numel()),
                "direct_source_projection_rows": 0,
                "direct_source_projection_entries": 0,
            }
            if self.query_edge_geometry_mode == "typed_w_centroids_v1":
                prep_work.update({
                    "content_centroid_coordinate_products": int(
                        scene.batch_size * 6 * (modules + environment_count) * self.coordinate_dim
                    ),
                    "content_centroid_output_coordinates": int(
                        2 * scene.batch_size * 6 * self.coordinate_dim
                    ),
                })
        else:
            edges = module_states.new_zeros(scene.batch_size, 0, self.context_dim)
            module_weights = module_states.new_zeros(scene.batch_size, 0, modules)
            environment_weights = module_states.new_zeros(scene.batch_size, 0, environment_count)
            anchors = module_states.new_zeros(scene.batch_size, 0, self.coordinate_dim)
            scales = module_states.new_zeros(scene.batch_size, 0, self.coordinate_dim)
            source_coefficients = None
            edge_keys = edge_values = None
            direct_source_keys, direct_source_values = self.direct_reader.prepare_sources(module_states)
            prep_work = {
                "context_mm_rows": int(self.context_depth * scene.batch_size * modules * modules),
                "context_em_rows": int(self.context_depth * scene.batch_size * environment_count * modules),
                "context_ee_rows": int(self.context_depth * scene.batch_size * environment_count * environment_count),
                "context_me_rows": int(self.context_depth * scene.batch_size * modules * environment_count),
                "context_dense_pair_rows": int(
                    self.context_depth * scene.batch_size * (modules * modules + 2 * modules * environment_count + environment_count * environment_count)
                ),
                "context_geometry_pair_rows": int(
                    scene.batch_size * (modules * modules + 2 * modules * environment_count + environment_count * environment_count)
                ),
                "content_gather_score_rows": 0,
                "content_geometry_pair_rows": 0,
                "edge_interaction_pair_rows": 0,
                "edge_interaction_head_pair_rows": 0,
                "context_attention_trace_head_pair_rows": int(
                    scene.batch_size * self.context_depth * self.num_heads
                    * (modules * modules + 2 * modules * environment_count + environment_count * environment_count)
                ) if request_attention else 0,
                "edge_interaction_attention_trace_head_pair_rows": 0,
                "scene_summary_module_entries": int((self.context_depth + 1) * scene.batch_size * modules),
                "scene_summary_environment_entries": int((self.context_depth + 1) * scene.batch_size * environment_count),
                "raw_reinjection_rows": int(self.context_depth * scene.batch_size * (modules + environment_count)),
                "scene_reinjection_rows": int(self.context_depth * scene.batch_size),
                "source_coefficient_entries": 0,
                "direct_source_projection_rows": int(scene.batch_size * modules),
                "direct_source_projection_entries": int(scene.batch_size * modules * self.context_dim),
            }

        if self.query_edge_geometry_mode == "typed_w_centroids_v1":
            candidate_trace_values = {
                "module_content_centroids": module_content_centroids,
                "environment_content_centroids": environment_content_centroids,
                "module_content_valid": module_content_valid,
                "environment_content_valid": environment_content_valid,
            }
            for name, value in candidate_trace_values.items():
                if name in selected:
                    if value is None:
                        raise ValueError("Typed content centroid trace requested without prepared centroids.")
                    organization_trace[name] = value

        for key, value in (
            ("edges", edges),
            ("module_gather_weights", module_weights),
            ("environment_gather_weights", environment_weights),
            ("edge_anchors", anchors),
            ("edge_scales", scales),
        ):
            if key in selected:
                organization_trace[key] = value
        if trace_query_indices is not None:
            # Query indices are accepted uniformly by prepare/decode, but
            # preparation has no receiver axis and therefore uses no slice.
            raise ValueError("trace_request.query_indices is only valid on decode/affine reader traces.")
        if not use_summary:
            organization_trace.pop("scene_summary", None)
            organization_trace.pop("scene_summary_initial", None)
            prep_work["scene_summary_module_entries"] = 0
            prep_work["scene_summary_environment_entries"] = 0
            prep_work["scene_reinjection_rows"] = 0
        if context_work_summaries:
            native_cycle_trace = dict(native_cycle_trace or {})
            native_cycle_trace["applied_context_stages"] = tuple(applied_context_stages)
            native_cycle_trace["context_work_summary"] = sum(context_work_summaries)
            native_cycle_trace["context_stage_metadata"] = tuple(context_stage_metadata)
        bounded_trace = self._bounded_trace(organization_trace, selected, trace_cap, trace_explicit)
        prep_work["trace_payload_elements"] = int(
            sum(value.numel() for value in (bounded_trace or {}).values())
        )
        source_read_projection = source_read_completion = None
        if self.architecture == "A2-D" and self.direct_reader.source_read is not None:
            controller = self.direct_reader.source_read
            source_read_projection = controller.project_source_states(module_states)
            if controller.completion_enabled:
                source_read_completion = controller.project_completion_sources(module_states)
            prep_work["source_read_score_projection_entries"] = source_read_projection.numel()
            prep_work["source_read_completion_projection_entries"] = (
                0 if source_read_completion is None else source_read_completion.numel())
        prepared = PreparedOrganizationV2(
            scene=scene,
            module_states=module_states,
            environment_states=environment_states,
            raw_module_states=raw_module_states,
            raw_environment_states=raw_environment_states,
            scene_summary=scene_summary,
            edges=edges,
            first_content=content_result.get("first_content", edges),
            mixed_content=content_result.get("mixed_content", edges),
            second_content=content_result.get("second_content", edges),
            module_gather_weights=module_weights,
            environment_gather_weights=environment_weights,
            edge_anchors=anchors,
            edge_scales=scales,
            edge_keys=edge_keys,
            edge_values=edge_values,
            direct_source_keys=direct_source_keys,
            direct_source_values=direct_source_values,
            source_read_source_projection=source_read_projection,
            source_read_completion_source=source_read_completion,
            source_read_projection_versions=tuple(_tensor_version(value) for value in
                (source_read_projection, source_read_completion) if value is not None),
            source_read_input_versions=(tuple(_tensor_version(value) for value in
                (direct_source_keys, direct_source_values, module_states, frame_origin, frame_scale,
                 operating_state) if value is not None)
                if self.architecture == "A2-D" and (self.source_read_policy != "full"
                    or self._source_read_override is not None) else ()),
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
            work_counts=prep_work,
            trace=bounded_trace,
            module_content_centroids=module_content_centroids,
            environment_content_centroids=environment_content_centroids,
            module_content_valid=module_content_valid,
            environment_content_valid=environment_content_valid,
            content_geometry_versions=(
                tuple(
                    _tensor_version(value)
                    for value in (
                        module_content_centroids,
                        environment_content_centroids,
                        module_content_valid,
                        environment_content_valid,
                    )
                    if value is not None
                )
                if self.query_edge_geometry_mode == "typed_w_centroids_v1"
                else ()
            ),
        )
        return self._finalize_native_preparation(prepared, native_cycle_trace)

    def _uses_scene_summary(self) -> bool:
        """Legacy default; unified local-input models remove this value path."""
        return True

    def _context_value_policy(self):
        """Optional applied typed incidence, evaluated before contextual reduction."""
        return None

    def _captures_native_routes(self) -> bool:
        """Narrow opt-in hook; historical families retain their exact path."""
        return False

    def _native_capture_route_names(self):
        return ("mm", "me")

    def _finalize_native_preparation(self, organization, cycle_trace):
        return organization

    def _uses_native_retrieval(self) -> bool:
        return False

    def _decode_native_retrieval(self, organization, query, query_states, local_pairs, *, return_messages=False):
        raise NotImplementedError("Native retrieval is an explicit core capability.")

    def _prepare_native_decode(self, organization, query, query_states, *, trace_enabled=False):
        """Default-preserving seam for conditioning every physical decode path."""
        return query_states, None

    def _decode_native_values(self, organization, query, query_states, local_pairs, *,
                              return_messages=False, decode_context=None):
        return self._decode_native_retrieval(organization, query, query_states, local_pairs,
                                             return_messages=return_messages)

    @staticmethod
    def _bounded_trace(values: Mapping[str, Tensor], selected: set[str], max_elements: int, explicit: bool):
        trace: dict[str, Tensor] = {}
        omitted = 0
        for name, value in values.items():
            if name not in selected:
                continue
            if value.numel() > max_elements:
                if explicit:
                    raise ValueError(
                        f"Requested trace {name!r} has {value.numel()} elements, above max_elements={max_elements}."
                    )
                omitted += 1
                continue
            trace[name] = value
        if omitted:
            trace["trace_omitted_count"] = torch.tensor(omitted, dtype=torch.int64)
        return trace or None

    def _validate_prepared_state(self, organization: PreparedOrganizationV2) -> None:
        if not isinstance(organization, PreparedOrganizationV2):
            raise TypeError("Expected a PreparedOrganizationV2 from this core family.")
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
        source_projections = (organization.source_read_source_projection,
                              organization.source_read_completion_source)
        if organization.source_read_projection_versions != tuple(
                _tensor_version(value) for value in source_projections if value is not None):
            raise ValueError("Prepared source-read projections changed; rebuild the organization.")
        if organization.source_read_input_versions and organization.source_read_input_versions != tuple(
                _tensor_version(value) for value in (organization.direct_source_keys,
                organization.direct_source_values, organization.module_states, organization.frame_origin,
                organization.frame_scale, organization.operating_state) if value is not None):
            raise ValueError("Prepared source-read inputs changed; rebuild the organization.")
        if self.query_edge_geometry_mode == "typed_w_centroids_v1":
            content_geometry = (
                organization.module_content_centroids,
                organization.environment_content_centroids,
                organization.module_content_valid,
                organization.environment_content_valid,
            )
            if any(not isinstance(value, Tensor) for value in content_geometry):
                raise ValueError("Prepared content geometry centroids or validity masks are missing.")
            batch = organization.scene.batch_size
            expected_shapes = (
                (batch, 6, self.coordinate_dim),
                (batch, 6, self.coordinate_dim),
                (batch, 6),
                (batch, 6),
            )
            if any(tuple(value.shape) != shape for value, shape in zip(content_geometry, expected_shapes)):
                raise ValueError("Prepared content geometry centroid or validity shape changed.")
            if content_geometry[2].dtype != torch.bool or content_geometry[3].dtype != torch.bool:
                raise ValueError("Prepared content geometry validity masks must remain boolean.")
            current_versions = tuple(_tensor_version(value) for value in content_geometry)
            if len(organization.content_geometry_versions) != 4 or (
                organization.content_geometry_versions != current_versions
            ):
                raise ValueError("Prepared content geometry centroid state changed; rebuild the organization.")

    def _validate_query(self, organization: PreparedOrganizationV2, query: QueryBatch) -> None:
        query.validate(
            batch_size=organization.scene.batch_size,
            coordinate_dim=self.coordinate_dim,
            query_input_dim=self.query_input_dim,
        )
        device = organization.module_states.device
        if query.coordinates.device != device:
            raise ValueError("Query tensors must be on the prepared organization's device.")
        for value in (
            query.features,
            query.query_mask,
            query.query_ids,
            query.local_batch_indices,
            query.local_query_indices,
            query.local_module_indices,
            query.local_weights,
            query.anchor_indices,
            query.anchor_weights,
        ):
            if value is not None and value.device != device:
                raise ValueError("All supplied query tensors must be on the prepared organization's device.")
        if query.local_batch_indices is not None:
            batch_size, query_count = query.coordinates.shape[:2]
            module_count = organization.scene.num_modules
            in_bounds = (
                (query.local_batch_indices >= 0)
                & (query.local_batch_indices < batch_size)
                & (query.local_query_indices >= 0)
                & (query.local_query_indices < query_count)
                & (query.local_module_indices >= 0)
                & (query.local_module_indices < module_count)
            ).all()
            if in_bounds.device.type == "cuda":
                torch._assert_async(in_bounds, "Packed local-pair indices are outside the scene/query axes.")
            elif not bool(in_bounds):
                raise ValueError("Packed local-pair indices are outside the scene/query axes.")
        if query.anchor_indices is not None:
            active = query.anchor_weights > 0
            in_bounds = (((query.anchor_indices >= 0)
                          & (query.anchor_indices < organization.scene.num_environment_tokens))
                         | ~active).all()
            if in_bounds.device.type == "cuda":
                torch._assert_async(in_bounds, "Sparse query anchor slots are outside the environment axis.")
            elif not bool(in_bounds):
                raise ValueError("Sparse query anchor slots are outside the environment axis.")
        if organization.scene.scene_key is not None and query.scene_key is not None:
            if organization.scene.scene_key != query.scene_key:
                raise ValueError("Query scene key does not match the prepared organization.")
        if organization.scene.coordinate_frame_key is not None and query.coordinate_frame_key is not None:
            if organization.scene.coordinate_frame_key != query.coordinate_frame_key:
                raise ValueError("Query coordinate frame does not match the prepared organization.")

    def validate_prepared_state(self, organization: PreparedOrganizationV2, query: QueryBatch) -> None:
        """Validate ownership and a full receiver request before internal tiling."""
        self._validate_prepared_state(organization)
        self._validate_query(organization, query)

    def _validate_local_intervention(self, organization, local_gates: Tensor | None) -> None:
        if local_gates is None:
            return
        expected = (organization.scene.batch_size, organization.scene.num_modules)
        if tuple(local_gates.shape) != expected:
            raise ValueError(f"local_gates must have shape {expected}.")
        if local_gates.device != organization.module_states.device:
            raise ValueError("local_gates must be on the prepared organization's device.")
        valid = torch.isfinite(local_gates).all() & (local_gates >= 0).all() & (local_gates <= 1).all()
        if valid.device.type == "cuda":
            torch._assert_async(valid, "local_gates must contain finite values in [0,1].")
        elif not bool(valid):
            raise ValueError("local_gates must contain finite values in [0,1].")

    def _validate_edge_intervention(
        self,
        organization: PreparedOrganizationV2,
        edge_gates: Tensor | None,
        uniform_access: bool,
    ) -> None:
        if self.architecture != "A2-H6":
            if edge_gates is not None or uniform_access:
                raise ValueError("Edge interventions are only available for A2-H6.")
            return
        if edge_gates is not None:
            expected = (organization.scene.batch_size, 6)
            if tuple(edge_gates.shape) != expected:
                raise ValueError(f"edge_gates must have shape {expected}.")
            if edge_gates.device != organization.module_states.device:
                raise ValueError("edge_gates must be on the prepared organization's device.")
            valid = torch.isfinite(edge_gates).all() & (edge_gates >= 0).all() & (edge_gates <= 1).all()
            if valid.device.type == "cuda":
                torch._assert_async(valid, "edge_gates must contain finite values in [0,1].")
            elif not bool(valid):
                raise ValueError("edge_gates must contain finite values in [0,1].")

    def _module_measure_reference(self, scene: SceneBatch, dtype: torch.dtype) -> Tensor:
        if scene.module_measure_reference is None:
            return scene.module_measure.new_ones(scene.batch_size, dtype=dtype)
        return scene.module_measure_reference.to(dtype=dtype)

    def _query_states(self, organization: PreparedOrganizationV2, query: QueryBatch) -> Tensor:
        return self.query_encoder(
            query, organization.operating_state, organization.frame_origin, organization.frame_scale
        )

    def source_read_teacher_trace_v1(self, organization: PreparedOrganizationV2, query: QueryBatch) -> dict[str, Any]:
        """Bounded direct-reader model targets, never physical source labels.

        The caller owns TRAIN membership and the teacher-row budget. This
        seam refuses a full-field query dump and does no local/native read.
        It preserves the qualified legacy output projection for its labels.
        """
        if self.architecture != "A2-D" or query.num_queries > 256:
            raise ValueError("Source-read teachers require A2-D and at most256 receiver rows.")
        self.validate_prepared_state(organization, query)
        from .source_read_sparse import _fine_hidden, original_source_scores_v1

        query_states = self._query_states(organization, query)
        scene, reader = organization.scene, self.direct_reader
        descriptor, alpha = original_source_scores_v1(
            reader, scene, query, query_states, organization.frame_scale,
            self._module_measure_reference(scene, query_states.dtype), organization.direct_source_keys)
        hidden = _fine_hidden(reader, descriptor, reader.query_value(query_states)[:, :, None],
                              organization.direct_source_values[:, None])
        messages = reader.message_output(hidden)
        qmask = torch.ones(query.coordinates.shape[:2], dtype=torch.bool, device=query.coordinates.device)
        if query.query_mask is not None:
            qmask = query.query_mask
        thermal_k = None
        if reader.thermal_output is not None:
            thermal_k = reader.thermal_output(messages).squeeze(-1)
            thermal_k = thermal_k * scene.module_mask[:, None].to(thermal_k.dtype) * qmask[:, :, None]
        delta = descriptor[..., :self.coordinate_dim] * organization.frame_scale[:, None, None]
        return {
            "query_states": query_states, "module_states": organization.module_states,
            "descriptor": descriptor, "alpha": alpha, "fine_hidden": hidden,
            "fine_messages": messages, "thermal_k": thermal_k,
            "query_mask": qmask, "source_present_mask": scene.module_mask,
            "local_radius": scene.local_radius,
            "distance_to_sources": torch.linalg.vector_norm(delta, dim=-1),
            "fine_executed_rows": query.batch_size * query.num_queries * scene.num_modules,
            "fine_present_rows": int((qmask[:, :, None] & scene.module_mask[:, None]).sum().item()),
        }

    @staticmethod
    def _accumulation_dtype(dtype: torch.dtype | None, fallback: torch.dtype) -> torch.dtype:
        if dtype is None:
            dtype = fallback
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("accumulation_dtype must be torch.float32 or torch.float64.")
        return dtype

    @staticmethod
    def _local_apply(
        local_pairs: LocalPairPack,
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
        b, q, m = local_pairs.batch_index, local_pairs.query_index, local_pairs.module_index
        values = (
            local_pairs.support_weight.to(dtype)
            * local_coefficients.to(dtype)
            * controls[b, m].to(dtype)
        )
        return result.index_add(0, b * query_count + q, values).reshape(batch_size, query_count)

    def _factorized_apply(
        self,
        organization: PreparedOrganizationV2,
        query: QueryBatch,
        basis: Tensor,
        local_pairs: LocalPairPack,
        local_coefficients: Tensor,
        controls: Tensor,
        *,
        accumulation_dtype: torch.dtype | None,
    ) -> Tensor:
        if organization.source_coefficients is None:
            raise ValueError("This prepared organization has no H6 source coefficient bank.")
        dtype = self._accumulation_dtype(accumulation_dtype, controls.dtype)
        source_amplitudes = torch.einsum(
            "bhrm,bm->bhr", organization.source_coefficients.to(dtype), controls.to(dtype)
        )
        edge_response = torch.einsum("bqhr,bhr->bq", basis.to(dtype), source_amplitudes)
        local_response = self._local_apply(
            local_pairs,
            local_coefficients,
            controls,
            query.batch_size,
            query.num_queries,
            dtype=dtype,
        )
        return edge_response + local_response

    def _decode_features(
        self,
        organization: PreparedOrganizationV2,
        query: QueryBatch,
        *,
        return_trace: bool,
        trace_request=None,
        edge_gates: Tensor | None = None,
        uniform_access: bool = False,
        local_gates: Tensor | None = None,
    ):
        query_states, decode_context = self._prepare_native_decode(
            organization, query, self._query_states(organization, query),
            trace_enabled=return_trace or trace_request is not None)
        local_context, local_pairs, local_coefficients = self.local_reader(
            organization.scene,
            query,
            organization.module_states,
            query_states,
            organization.frame_scale,
            self._module_measure_reference(organization.scene, query_states.dtype),
            local_gates=local_gates,
            execution_backend=self.reader_execution_backend,
        )
        trace_available = list(_DECODE_TRACE_AVAILABLE)
        if organization.trace is not None:
            trace_available.extend(name for name in organization.trace if name not in trace_available)
        selected, trace_cap, query_indices, trace_explicit = _trace_request(
            return_trace, trace_request, _DECODE_TRACE_DEFAULTS, trace_available
        )
        if query_indices is not None:
            if query_indices.numel() and (
                int(query_indices.min()) < 0 or int(query_indices.max()) >= query.num_queries
            ):
                raise ValueError("trace_request.query_indices are outside the query axis.")
            query_indices = query_indices.to(device=query.coordinates.device)
        want_messages = "edge_messages" in selected or "direct_source_messages" in selected
        trace: dict[str, Tensor] = {}
        source_read_work = None
        direct_messages_preselected = False
        if self.architecture == "A2-H6":
            global_context, base_access, used_access, basis, messages = self.edge_reader(
                query,
                query_states,
                organization,
                organization.frame_scale,
                edge_gates=edge_gates,
                uniform_access=uniform_access,
                return_messages=want_messages,
            )
            direct_coefficients = None
            gates = (
                organization.edges.new_ones(organization.scene.batch_size, 6)
                if edge_gates is None
                else edge_gates.to(dtype=organization.edges.dtype)
            )
            if "edge_access" in selected:
                # Public access reflects the coefficients actually used by
                # the reader (base attention multiplied by intervention
                # gates, without post-gate renormalization).
                trace["edge_access"] = used_access.permute(0, 1, 3, 2)
            if "edge_base_access" in selected:
                trace["edge_base_access"] = base_access.permute(0, 1, 3, 2)
            if "edge_used_access" in selected:
                trace["edge_used_access"] = used_access.permute(0, 1, 3, 2)
            if "edge_gates" in selected:
                trace["edge_gates"] = gates
            if messages is not None and "edge_messages" in selected:
                trace["edge_messages"] = messages
            if basis is not None and "thermal_basis" in selected:
                trace["thermal_basis"] = basis
        else:
            direct_message_indices = (
                query_indices
                if self.reader_execution_backend == "reduced_projection"
                and "direct_source_messages" in selected
                else None
            )
            if self._uses_native_retrieval():
                result = self._decode_native_values(
                    organization, query, query_states, local_pairs, return_messages=want_messages,
                    decode_context=decode_context)
                global_context, direct_access = result.context, result.alpha
                direct_coefficients, messages = result.response, result.messages
                source_read_work = result.counts
                trace.update({name: value for name, value in result.trace.items() if name in selected})
                direct_message_indices = None
            elif self._uses_source_selection():
                from .source_read_decoder_v1 import decode_selected_sources_v1
                result = decode_selected_sources_v1(self, organization, query, query_states,
                                                    return_messages=want_messages)
                global_context, direct_access = result.context, result.alpha
                direct_coefficients, messages = result.response, result.messages
                source_read_work = result.counts
                trace.update({name: value for name, value in result.trace.items() if name in selected})
                direct_message_indices = None
            else:
                global_context, direct_access, direct_coefficients, messages = self.direct_reader(
                    organization.scene, query, query_states, organization.frame_scale,
                    self._module_measure_reference(organization.scene, query_states.dtype),
                    organization.direct_source_keys, organization.direct_source_values,
                    return_messages=want_messages, execution_backend=self.reader_execution_backend,
                    message_query_indices=direct_message_indices)
            direct_messages_preselected = direct_message_indices is not None
            basis = None
            if "direct_source_access" in selected:
                trace["direct_source_access"] = direct_access
            if messages is not None and "direct_source_messages" in selected:
                trace["direct_source_messages"] = messages
            if "direct_temperature_coefficients" in selected:
                trace["direct_temperature_coefficients"] = (
                    direct_coefficients.explicit_dense_kernel()
                    if direct_coefficients is not None and not isinstance(direct_coefficients, Tensor)
                    else direct_coefficients)

        nonlinear_field = self.field_head(query_states, global_context, local_context)
        if query.query_mask is not None:
            nonlinear_field = nonlinear_field * query.query_mask[:, :, None].to(nonlinear_field.dtype)
        if local_pairs.count:
            if "local_pair_batch_indices" in selected:
                trace["local_pair_batch_indices"] = local_pairs.batch_index
            if "local_pair_query_indices" in selected:
                trace["local_pair_query_indices"] = local_pairs.query_index
            if "local_pair_module_indices" in selected:
                trace["local_pair_module_indices"] = local_pairs.module_index
            if "local_pair_support_weights" in selected:
                trace["local_pair_support_weights"] = local_pairs.support_weight
            if "local_temperature_coefficients" in selected:
                trace["local_temperature_coefficients"] = local_coefficients
        if organization.trace is not None and (return_trace or trace_request is not None):
            trace.update(organization.trace)
            if return_trace and trace_request is None:
                selected.update(organization.trace)

        if query_indices is not None:
            for name, value in tuple(trace.items()):
                if name in {
                    "edge_access", "edge_base_access", "edge_used_access", "edge_messages",
                    "thermal_basis", "edge_temperature_contributions", "direct_source_access",
                    "direct_source_messages", "direct_temperature_coefficients",
                    "source_read_gates", "source_read_support",
                    "native_module_permission", "native_seed_permission",
                    "native_hard_support", "native_dispatch_seed", "native_dispatch_seed_ids",
                    "native_protected_direct", "native_protected_exceptions",
                    "joint_environment_support", "joint_environment_access",
                    "joint_environment_context", "joint_conditioned_query",
                    "joint_hard_support", "joint_protected_mask", "joint_mandatory_support",
                } and value.ndim >= 2 and value.shape[1] == query.num_queries and not (
                    direct_messages_preselected and name == "direct_source_messages"
                ):
                    trace[name] = value.index_select(1, query_indices)
            trace["trace_query_indices"] = query_indices
            selected.add("trace_query_indices")
        bounded_trace = self._bounded_trace(trace, selected, trace_cap, trace_explicit)
        return (
            nonlinear_field,
            query_states,
            local_context,
            basis,
            direct_coefficients,
            local_coefficients,
            local_pairs,
            bounded_trace or {},
            source_read_work,
        )

    def decode(
        self,
        organization: PreparedOrganizationV2,
        query: QueryBatch,
        controls: Tensor | None = None,
        *,
        return_basis: bool = False,
        return_trace: bool = False,
        trace_request=None,
        accumulation_dtype: torch.dtype | None = None,
        edge_gates: Tensor | None = None,
        uniform_access: bool = False,
        local_gates: Tensor | None = None,
    ) -> CoreOutput:
        """Decode receivers, with optional separately applicable linear controls."""
        self.validate_prepared_state(organization, query)
        if self.reader_query_chunk_size is not None and not (return_trace or trace_request is not None):
            from .query_chunks_v1 import decode_query_chunks_v1

            return decode_query_chunks_v1(
                query, self.reader_query_chunk_size,
                lambda tile: self._decode_validated(
                    organization, tile, controls, return_basis=return_basis,
                    accumulation_dtype=accumulation_dtype, edge_gates=edge_gates,
                    uniform_access=uniform_access, local_gates=local_gates,
                ),
            )
        return self._decode_validated(
            organization,
            query,
            controls,
            return_basis=return_basis,
            return_trace=return_trace,
            trace_request=trace_request,
            accumulation_dtype=accumulation_dtype,
            edge_gates=edge_gates,
            uniform_access=uniform_access,
            local_gates=local_gates,
        )

    def _reader_work_counts(
        self,
        organization: PreparedOrganizationV2,
        query: QueryBatch,
        local_pairs: LocalPairPack,
        *,
        field_head: bool,
        apply_controls: bool,
        trace: Mapping[str, Tensor] | None = None,
        trace_selected: set[str] | None = None,
        source_read_work: Mapping[str, int] | None = None,
    ) -> dict[str, int]:
        """Count operations performed by one reader pass and its contraction."""
        batch, query_count = query.coordinates.shape[:2]
        modules = organization.scene.num_modules
        edges = organization.num_hyperedges if self.architecture == "A2-H6" else 0
        edge_rows = int(batch * query_count * edges)
        direct_rows = int(batch * query_count * modules) if self.architecture == "A2-D" else 0
        trace_selected = set() if trace_selected is None else trace_selected
        result = {
            "receiver_rows": int(batch * query_count),
            # Each edge row emits exactly one local_hidden_dim-wide message;
            # the message is split into heads after that single projection.
            "edge_rows": edge_rows,
            "edge_message_rows": edge_rows,
            "edge_feature_message_entries": int(edge_rows * self.local_hidden_dim),
            "edge_access_rows": int(edge_rows * self.num_heads),
            "edge_temperature_mode_rows": int(
                edge_rows * self.thermal_response_rank
            ) if self.architecture == "A2-H6" else 0,
            "direct_source_rows": direct_rows,
            "direct_source_message_entries": int(direct_rows * self.local_hidden_dim),
            "direct_temperature_coefficient_entries": int(
                direct_rows
            ) if self.architecture == "A2-D" and self.thermal_response_rank else 0,
            # A2-D constructs its signed [B,Q,M] source coefficients during
            # every reader call when thermal response is enabled, even when
            # the caller has not yet supplied controls.
            "direct_dense_reference_entries": direct_rows
            if self.architecture == "A2-D" and self.thermal_response_rank else 0,
            "local_rows": int(local_pairs.count),
            "local_flow_feature_entries": int(local_pairs.count * self.local_hidden_dim),
            "local_temperature_coefficient_rows": int(local_pairs.count)
            if self.thermal_response_rank else 0,
            "local_candidate_rows": 0 if query.local_batch_indices is not None
            else int(batch * query_count * modules),
            "field_head_rows": int(batch * query_count) if field_head else 0,
            "field_head_input_entries": int(
                batch * query_count * (2 * self.context_dim + self.local_hidden_dim)
            ) if field_head else 0,
            "source_amplitude_contraction_rows": int(
                batch * edges * self.thermal_response_rank * modules
            ) if apply_controls and self.architecture == "A2-H6" else 0,
            "edge_response_contraction_rows": int(
                batch * query_count * edges * self.thermal_response_rank
            ) if apply_controls and self.architecture == "A2-H6" else 0,
            "direct_response_contraction_rows": direct_rows
            if apply_controls and self.architecture == "A2-D" else 0,
            "local_contraction_rows": int(local_pairs.count) if apply_controls else 0,
            "trace_edge_temperature_amplitude_rows": int(
                batch * edges * self.thermal_response_rank * modules
            ) if apply_controls and self.architecture == "A2-H6"
            and "edge_temperature_contributions" in trace_selected else 0,
            "trace_edge_temperature_product_rows": int(
                batch * query_count * edges * self.thermal_response_rank
            ) if apply_controls and self.architecture == "A2-H6"
            and "edge_temperature_contributions" in trace_selected else 0,
            "trace_payload_elements": int(
                sum(value.numel() for value in (trace or {}).values())
            ),
        }
        if self.query_edge_geometry_mode == "typed_w_centroids_v1":
            centroid_descriptor_rows = int(2 * batch * query_count * edges)
            result.update({
                "content_centroid_descriptor_rows": centroid_descriptor_rows,
                "content_centroid_geometry_feature_entries": int(
                    centroid_descriptor_rows * (2 * self.coordinate_dim + 1)
                ),
                "content_centroid_relative_gamma_entries": int(
                    centroid_descriptor_rows * 7 * self.coordinate_dim
                ),
            })
        if source_read_work is not None:
            fine_rows = int(source_read_work["fine_executed_rows"])
            message_rows = int(source_read_work.get("fine_message_projection_rows", fine_rows))
            selected_rows = int(source_read_work["fine_selected_rows"])
            full_fastpath = bool(source_read_work.get("source_full_fastpath", 0))
            result.update({"direct_source_rows": fine_rows,
                           "direct_source_message_entries": message_rows * self.local_hidden_dim,
                           "direct_dense_reference_entries": direct_rows if full_fastpath
                               and self.thermal_response_rank else 0,
                           "direct_temperature_coefficient_entries": (direct_rows if full_fastpath else selected_rows)
                               if self.thermal_response_rank else 0,
                           "direct_response_contraction_rows": (direct_rows if full_fastpath else selected_rows)
                               if apply_controls else 0})
            result.update({"source_read_" + name: int(value) for name, value in source_read_work.items()})
        return result

    def _decode_validated(
        self,
        organization: PreparedOrganizationV2,
        query: QueryBatch,
        controls: Tensor | None = None,
        *,
        return_basis: bool = False,
        return_trace: bool = False,
        trace_request=None,
        accumulation_dtype: torch.dtype | None = None,
        edge_gates: Tensor | None = None,
        uniform_access: bool = False,
        local_gates: Tensor | None = None,
    ) -> CoreOutput:
        """Decode an adapter-created query tile after full-request validation."""
        self._validate_edge_intervention(organization, edge_gates, uniform_access)
        self._validate_local_intervention(organization, local_gates)
        (
            nonlinear_field,
            _query_states,
            _local_context,
            basis,
            direct_coefficients,
            local_coefficients,
            local_pairs,
            trace,
            source_read_work,
        ) = self._decode_features(
            organization,
            query,
            return_trace=return_trace,
            trace_request=trace_request,
            edge_gates=edge_gates,
            uniform_access=uniform_access,
            local_gates=local_gates,
        )
        temperature = None
        if controls is not None:
            if not self.thermal_response_rank:
                raise ValueError("This v2 core was configured without affine thermal response.")
            expected = (query.batch_size, organization.scene.num_modules)
            if tuple(controls.shape) != expected:
                raise ValueError(f"controls must have shape {expected}.")
            if controls.device != organization.module_states.device:
                raise ValueError("controls must be on the prepared organization's device.")
            self._validate_source_read_endpoint(controls, organization.scene.module_mask)
            dtype = self._accumulation_dtype(accumulation_dtype, controls.dtype)
            if self.architecture == "A2-H6":
                if basis is None or organization.source_coefficients is None:
                    raise ValueError("A2-H6 requires its spatial and source coefficient factors.")
                temperature = self._factorized_apply(
                    organization,
                    query,
                    basis,
                    local_pairs,
                    local_coefficients,
                    controls,
                    accumulation_dtype=dtype,
                )
                if return_trace or trace_request is not None:
                    selected, cap, indices, explicit = _trace_request(
                        return_trace,
                        trace_request,
                        _DECODE_TRACE_DEFAULTS,
                        [
                            *_DECODE_TRACE_AVAILABLE,
                            *(organization.trace.keys() if organization.trace is not None else ()),
                            "trace_query_indices",
                        ],
                    )
                    if "edge_temperature_contributions" in selected:
                        amplitudes = torch.einsum(
                            "bhrm,bm->bhr", organization.source_coefficients.to(dtype), controls.to(dtype)
                        )
                        contributions = basis.to(dtype) * amplitudes[:, None, :, :]
                        if indices is not None:
                            indices = indices.to(device=contributions.device)
                            if indices.numel() and (
                                int(indices.min()) < 0 or int(indices.max()) >= query.num_queries
                            ):
                                raise ValueError("trace_request.query_indices are outside the query axis.")
                            contributions = contributions.index_select(1, indices)
                        if contributions.numel() > cap and explicit:
                            raise ValueError(
                                "Requested edge_temperature_contributions exceeds trace_request.max_elements."
                            )
                        if contributions.numel() <= cap:
                            trace["edge_temperature_contributions"] = contributions
                            if indices is not None:
                                trace["trace_query_indices"] = indices
            else:
                if direct_coefficients is None:
                    raise ValueError("A2-D direct signed response coefficients are missing.")
                direct = (torch.einsum("bqm,bm->bq", direct_coefficients.to(dtype), controls.to(dtype))
                          if isinstance(direct_coefficients, Tensor) else
                          direct_coefficients.apply(controls, accumulation_dtype=dtype))
                local = self._local_apply(
                    local_pairs,
                    local_coefficients,
                    controls,
                    query.batch_size,
                    query.num_queries,
                    dtype=dtype,
                )
                temperature = direct + local
        if query.query_mask is not None and temperature is not None:
            temperature = temperature * query.query_mask.to(temperature.dtype)

        trace_selected = set()
        if return_trace or trace_request is not None:
            trace_available = [
                *_DECODE_TRACE_AVAILABLE,
                *(organization.trace.keys() if organization.trace is not None else ()),
                "trace_query_indices",
            ]
            trace_selected, _, _, _ = _trace_request(
                return_trace, trace_request, _DECODE_TRACE_DEFAULTS, trace_available
            )
        work_counts = self._reader_work_counts(
            organization,
            query,
            local_pairs,
            field_head=True,
            apply_controls=controls is not None,
            trace=trace,
            trace_selected=trace_selected,
            source_read_work=source_read_work,
        )
        return CoreOutput(
            nonlinear_field=nonlinear_field,
            thermal_temperature=temperature,
            thermal_basis=basis if return_basis else None,
            trace=trace if return_trace or trace_request is not None else None,
            work_counts=work_counts,
        )

    def _read_for_response(
        self,
        organization: PreparedOrganizationV2,
        query: QueryBatch,
        *,
        edge_gates: Tensor | None = None,
        uniform_access: bool = False,
        local_gates: Tensor | None = None,
    ):
        self.validate_prepared_state(organization, query)
        self._validate_edge_intervention(organization, edge_gates, uniform_access)
        self._validate_local_intervention(organization, local_gates)
        query_states, decode_context = self._prepare_native_decode(
            organization, query, self._query_states(organization, query))
        _, local_pairs, local_coefficients = self.local_reader(
            organization.scene,
            query,
            organization.module_states,
            query_states,
            organization.frame_scale,
            self._module_measure_reference(organization.scene, query_states.dtype),
            local_gates=local_gates,
            execution_backend=self.reader_execution_backend,
        )
        if self.architecture == "A2-H6":
            _, _, _, basis, _ = self.edge_reader(
                query,
                query_states,
                organization,
                organization.frame_scale,
                edge_gates=edge_gates,
                uniform_access=uniform_access,
                return_messages=False,
            )
            return basis, None, local_pairs, local_coefficients, None
        if self._uses_native_retrieval():
            result = self._decode_native_values(organization, query, query_states, local_pairs,
                                                decode_context=decode_context)
            return None, result.response, local_pairs, local_coefficients, result.counts
        if self._uses_source_selection():
            from .source_read_decoder_v1 import decode_selected_sources_v1
            result = decode_selected_sources_v1(self, organization, query, query_states)
            return None, result.response, local_pairs, local_coefficients, result.counts
        _, _, direct, _ = self.direct_reader(
            organization.scene,
            query,
            query_states,
            organization.frame_scale,
            self._module_measure_reference(organization.scene, query_states.dtype),
            organization.direct_source_keys,
            organization.direct_source_values,
            return_messages=False,
            execution_backend=self.reader_execution_backend,
        )
        return None, direct, local_pairs, local_coefficients, None

    def apply_increment(
        self,
        organization: PreparedOrganizationV2,
        query: QueryBatch,
        delta_controls: Tensor,
        *,
        accumulation_dtype: torch.dtype | None = None,
        edge_gates: Tensor | None = None,
        uniform_access: bool = False,
        local_gates: Tensor | None = None,
        return_work_counts: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, int]]:
        expected = (query.batch_size, organization.scene.num_modules)
        if tuple(delta_controls.shape) != expected:
            raise ValueError(f"delta_controls must have shape {expected}.")
        if delta_controls.device != organization.module_states.device:
            raise ValueError("delta_controls must be on the prepared organization's device.")
        if not self.thermal_response_rank:
            raise ValueError("This v2 core was configured without affine thermal response.")
        basis, direct, local_pairs, local_coefficients, source_read_work = self._read_for_response(
            organization,
            query,
            edge_gates=edge_gates,
            uniform_access=uniform_access,
            local_gates=local_gates,
        )
        if self.architecture == "A2-H6":
            if basis is None:
                raise ValueError("A2-H6 needs spatial thermal modes to apply increments.")
            increment = self._factorized_apply(
                organization,
                query,
                basis,
                local_pairs,
                local_coefficients,
                delta_controls,
                accumulation_dtype=accumulation_dtype,
            )
        else:
            if direct is None:
                raise ValueError("A2-D needs signed per-source thermal coefficients.")
            dtype = self._accumulation_dtype(accumulation_dtype, delta_controls.dtype)
            direct_increment = (torch.einsum("bqm,bm->bq", direct.to(dtype), delta_controls.to(dtype))
                                if isinstance(direct, Tensor) else direct.apply(
                                    delta_controls, accumulation_dtype=dtype, check_envelope=False))
            increment = direct_increment + self._local_apply(
                local_pairs,
                local_coefficients,
                delta_controls,
                query.batch_size,
                query.num_queries,
                dtype=dtype,
            )
        if not return_work_counts:
            return increment
        work_counts = self._reader_work_counts(
            organization,
            query,
            local_pairs,
            field_head=False,
            apply_controls=True,
            source_read_work=source_read_work,
        )
        return increment, work_counts

    def heat_jvp(
        self,
        organization: PreparedOrganizationV2,
        query: QueryBatch,
        tangent: Tensor,
        *,
        accumulation_dtype: torch.dtype | None = None,
        edge_gates: Tensor | None = None,
        uniform_access: bool = False,
        local_gates: Tensor | None = None,
    ) -> Tensor:
        return self.apply_increment(
            organization,
            query,
            tangent,
            accumulation_dtype=accumulation_dtype,
            edge_gates=edge_gates,
            uniform_access=uniform_access,
            local_gates=local_gates,
        )

    def heat_vjp(
        self,
        organization: PreparedOrganizationV2,
        query: QueryBatch,
        cotangent: Tensor,
        *,
        accumulation_dtype: torch.dtype | None = None,
        edge_gates: Tensor | None = None,
        uniform_access: bool = False,
        local_gates: Tensor | None = None,
    ) -> Tensor:
        if cotangent.shape != (query.batch_size, query.num_queries):
            raise ValueError("cotangent must match [B,Q].")
        if cotangent.device != organization.module_states.device:
            raise ValueError("cotangent must be on the prepared organization's device.")
        if not self.thermal_response_rank:
            raise ValueError("This v2 core was configured without affine thermal response.")
        basis, direct, local_pairs, local_coefficients, _source_read_work = self._read_for_response(
            organization,
            query,
            edge_gates=edge_gates,
            uniform_access=uniform_access,
            local_gates=local_gates,
        )
        dtype = self._accumulation_dtype(accumulation_dtype, cotangent.dtype)
        if self.architecture == "A2-H6":
            if basis is None or organization.source_coefficients is None:
                raise ValueError("A2-H6 needs both spatial and source factors for a heat VJP.")
            weighted_modes = torch.einsum("bq,bqhr->bhr", cotangent.to(dtype), basis.to(dtype))
            gradient = torch.einsum("bhr,bhrm->bm", weighted_modes, organization.source_coefficients.to(dtype))
        else:
            if direct is None:
                raise ValueError("A2-D needs signed per-source thermal coefficients.")
            gradient = (torch.einsum("bq,bqm->bm", cotangent.to(dtype), direct.to(dtype))
                        if isinstance(direct, Tensor) else direct.transpose(cotangent, accumulation_dtype=dtype))
        if local_pairs.count:
            b, q, m = local_pairs.batch_index, local_pairs.query_index, local_pairs.module_index
            contribution = (
                cotangent[b, q].to(dtype)
                * local_pairs.support_weight.to(dtype)
                * local_coefficients.to(dtype)
            )
            gradient = gradient.reshape(-1).index_add(
                0, b * organization.scene.num_modules + m, contribution
            ).reshape(query.batch_size, organization.scene.num_modules)
        return gradient

    def prepare_affine_query(
        self,
        organization: PreparedOrganizationV2,
        query: QueryBatch,
        *,
        edge_gates: Tensor | None = None,
        uniform_access: bool = False,
        local_gates: Tensor | None = None,
    ) -> PreparedAffineQueryV2:
        if not self.thermal_response_rank:
            raise ValueError("This v2 model has no affine-control capability.")
        self.validate_prepared_state(organization, query)
        self._validate_edge_intervention(organization, edge_gates, uniform_access)
        self._validate_local_intervention(organization, local_gates)
        field, _, _, basis, direct, local_coefficients, local_pairs, _, source_read_work = self._decode_features(
            organization,
            query,
            return_trace=False,
            edge_gates=edge_gates,
            uniform_access=uniform_access,
            local_gates=local_gates,
        )
        prepared = PreparedAffineQueryV2(
            organization=organization,
            query=query,
            basis=basis,
            direct_coefficients=direct,
            local_pairs=local_pairs,
            local_coefficients=local_coefficients,
            nonlinear_field=field,
            query_versions=_query_tensor_versions(query),
            factor_versions=(),
            query_identity=(query.scene_key, query.coordinate_frame_key, query.receiver_key),
            edge_gates=edge_gates,
            uniform_access=uniform_access,
            local_gates=local_gates,
            architecture=self.architecture,
            source_read_work=source_read_work,
        )
        if self._uses_source_selection():
            tensors = (*_query_tensor_versions(query), *organization.scene.tensor_version_stamp(),
                       *_factor_versions(prepared))
            if any(stamp[-1] < 0 for stamp in tensors):
                raise ValueError("Reusable source plans require versioned tensors; prepare under no_grad, not inference_mode.")
        return PreparedAffineQueryV2(**{**vars(prepared), "factor_versions": _factor_versions(prepared)})

    def _validate_affine_query(self, prepared: PreparedAffineQueryV2) -> None:
        if not isinstance(prepared, PreparedAffineQueryV2) or prepared.architecture != self.architecture:
            raise TypeError("Expected a PreparedAffineQueryV2 from this architecture.")
        self.validate_prepared_state(prepared.organization, prepared.query)
        if _query_tensor_versions(prepared.query) != prepared.query_versions:
            raise ValueError("Receiver/support tensors changed after affine query preparation.")
        if _factor_versions(prepared) != prepared.factor_versions:
            raise ValueError("Prepared affine factors changed; rebuild the query.")
        if prepared.query_identity != (
            prepared.query.scene_key,
            prepared.query.coordinate_frame_key,
            prepared.query.receiver_key,
        ):
            raise ValueError("Prepared affine query identity changed; rebuild the query.")

    def apply_prepared_affine(
        self,
        prepared_query: PreparedAffineQueryV2,
        controls: Tensor,
        *,
        accumulation_dtype: torch.dtype | None = None,
        check_envelope: bool = True,
    ) -> CoreOutput:
        self._validate_affine_query(prepared_query)
        organization, query = prepared_query.organization, prepared_query.query
        if controls.shape != (query.batch_size, organization.scene.num_modules):
            raise ValueError("Affine controls must match the prepared physical source axis.")
        if controls.device != query.coordinates.device:
            raise ValueError("Affine controls must be on the prepared query device.")
        if check_envelope:
            self._validate_source_read_endpoint(controls, organization.scene.module_mask)
        dtype = self._accumulation_dtype(accumulation_dtype, controls.dtype)
        if prepared_query.basis is not None:
            temperature = self._factorized_apply(
                organization,
                query,
                prepared_query.basis,
                prepared_query.local_pairs,
                prepared_query.local_coefficients,
                controls,
                accumulation_dtype=dtype,
            )
            direct_dense_entries = 0
        else:
            if prepared_query.direct_coefficients is None:
                raise ValueError("A2-D affine cache is missing explicit direct coefficients.")
            direct_factors = prepared_query.direct_coefficients
            temperature = (torch.einsum("bqm,bm->bq", direct_factors.to(dtype), controls.to(dtype))
                           if isinstance(direct_factors, Tensor) else
                           direct_factors.apply(controls, accumulation_dtype=dtype,
                                                check_envelope=check_envelope))
            temperature = temperature + self._local_apply(
                prepared_query.local_pairs,
                prepared_query.local_coefficients,
                controls,
                query.batch_size,
                query.num_queries,
                dtype=dtype,
            )
            direct_dense_entries = int(direct_factors.numel()) if isinstance(direct_factors, Tensor) else 0
        if query.query_mask is not None:
            temperature = temperature * query.query_mask.to(dtype)
        return CoreOutput(
            nonlinear_field=prepared_query.nonlinear_field,
            thermal_temperature=temperature,
            work_counts={
                "receiver_rows": int(query.batch_size * query.num_queries),
                "edge_rows": 0,
                "direct_source_rows": 0,
                "direct_dense_reference_entries": direct_dense_entries,
                "local_rows": 0,
                "local_contraction_rows": int(prepared_query.local_pairs.count),
                "spatial_factor_entries": 0
                if prepared_query.basis is None
                else int(prepared_query.basis.numel()),
            },
        )

    def heat_vjp_prepared_affine(
        self,
        prepared_query: PreparedAffineQueryV2,
        cotangent: Tensor,
        *,
        accumulation_dtype: torch.dtype | None = None,
    ) -> Tensor:
        self._validate_affine_query(prepared_query)
        organization, query = prepared_query.organization, prepared_query.query
        if cotangent.shape != (query.batch_size, query.num_queries) or cotangent.device != query.coordinates.device:
            raise ValueError("Affine cotangent must match the prepared receiver axis and device.")
        dtype = self._accumulation_dtype(accumulation_dtype, cotangent.dtype)
        weights = cotangent.to(dtype)
        if query.query_mask is not None:
            weights = weights * query.query_mask.to(dtype)
        if prepared_query.basis is not None:
            modes = torch.einsum("bq,bqhr->bhr", weights, prepared_query.basis.to(dtype))
            gradient = torch.einsum("bhr,bhrm->bm", modes, organization.source_coefficients.to(dtype))
        else:
            if prepared_query.direct_coefficients is None:
                raise ValueError("A2-D affine cache is missing explicit direct coefficients.")
            direct_factors = prepared_query.direct_coefficients
            gradient = (torch.einsum("bq,bqm->bm", weights, direct_factors.to(dtype))
                        if isinstance(direct_factors, Tensor) else
                        direct_factors.transpose(weights, accumulation_dtype=dtype))
        pack = prepared_query.local_pairs
        if pack.count:
            b, q, m = pack.batch_index, pack.query_index, pack.module_index
            local_values = weights[b, q] * pack.support_weight.to(dtype) * prepared_query.local_coefficients.to(dtype)
            gradient = gradient.reshape(-1).index_add(
                0, b * organization.scene.num_modules + m, local_values
            ).reshape(query.batch_size, organization.scene.num_modules)
        return gradient

    def export_source_kernels(
        self,
        organization: PreparedOrganizationV2,
        query: QueryBatch,
        *,
        accumulation_dtype: torch.dtype | None = None,
        edge_gates: Tensor | None = None,
        uniform_access: bool = False,
        local_gates: Tensor | None = None,
    ) -> Tensor:
        """Explicitly materialize dense ``[B,Q,M]`` K for bounded diagnostics."""
        basis, direct, local_pairs, local_coefficients, _source_read_work = self._read_for_response(
            organization,
            query,
            edge_gates=edge_gates,
            uniform_access=uniform_access,
            local_gates=local_gates,
        )
        dtype = self._accumulation_dtype(accumulation_dtype, organization.module_states.dtype)
        if self.architecture == "A2-H6":
            if basis is None or organization.source_coefficients is None:
                raise ValueError("A2-H6 needs both spatial and source factors to export kernels.")
            kernels = torch.einsum(
                "bqhr,bhrm->bqm", basis.to(dtype), organization.source_coefficients.to(dtype)
            )
        else:
            if direct is None:
                raise ValueError("A2-D needs direct signed source coefficients to export kernels.")
            kernels = (direct.to(dtype) if isinstance(direct, Tensor) else
                       direct.explicit_dense_kernel(accumulation_dtype=dtype))
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

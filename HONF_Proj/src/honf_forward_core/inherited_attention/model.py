"""Attention-inherited gathering with scene-reused, typed anchor values.

The contextualizer is the existing ordered MM/EM/EE/ME implementation. Only
the final EM/EE *scores* are inherited; payload projections see final token
states. There is no query/source pair-value reader in this model.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..content_hypergraph.contracts import CoreOutput, QueryBatch, SceneBatch
from ..content_hypergraph.decoding import LocalPairPack
from ..content_hypergraph.readers_v2 import FieldHeadV2, LocalDetailReaderV2, QueryEncoderV2
from ..content_hypergraph.spatial_features_v2 import coordinate_gamma, pair_score_features
from ..content_hypergraph.staged_attention_v2 import StagedContextCycleV2
from .atlas import build_typed_atlas, validate_cover_anchor_identity
from .contracts import (
    InheritedQueryBatch,
    PreparedAnchorValues,
    ReceiverCover,
    RouteScoreSnapshot,
    TypedAssociationAtlas,
    TypedTokenValues,
)
from .query_reader import build_cell_query_plan, query_conditioned_gather
from .readers import gather_cover_dense_reference, gather_cover_indexed, prepare_anchor_values


def _stamp(tensor: Tensor) -> tuple[int, int, int]:
    try:
        version = int(tensor._version)
    except RuntimeError:
        version = -1
    return id(tensor), tensor.data_ptr(), version


def _tensors(value: Any) -> tuple[Tensor, ...]:
    """Small contract containers only; never hash/copy numerical payloads."""
    if isinstance(value, Tensor):
        return (value,)
    if hasattr(value, "__dataclass_fields__"):
        return tuple(t for field in fields(value) for t in _tensors(getattr(value, field.name)))
    if isinstance(value, (tuple, list)):
        return tuple(t for item in value for t in _tensors(item))
    return ()


def _versions(value: Any) -> tuple[tuple[int, int, int], ...]:
    return tuple(_stamp(tensor) for tensor in _tensors(value))


@dataclass
class _GraphLifetime:
    consumed: bool = False

    def mark_consumed(self, gradient: Tensor) -> Tensor:
        self.consumed = True
        return gradient


@dataclass
class PreparedInheritedOrganization:
    """One scene's live atlas, final values and ownership boundary."""

    scene: SceneBatch
    module_states: Tensor
    environment_states: Tensor
    operating_state: Tensor
    frame_origin: Tensor
    frame_scale: Tensor
    scene_summary: Tensor
    module_score_snapshot: RouteScoreSnapshot
    environment_score_snapshot: RouteScoreSnapshot
    module_atlas: TypedAssociationAtlas
    environment_atlas: TypedAssociationAtlas
    module_values: TypedTokenValues
    environment_values: TypedTokenValues
    module_anchor_values: PreparedAnchorValues | None
    environment_anchor_values: PreparedAnchorValues | None
    source_carriers: Tensor | None  # [B,H,M,R], heat independent
    source_edge_indices: tuple[Tensor, Tensor, Tensor, Tensor] | None
    normalization_blend: float
    owner_token: int
    parameter_versions: tuple
    buffer_versions: tuple
    scene_versions: tuple
    factor_versions: tuple
    identity: tuple
    configuration: tuple
    graph_lifetime: _GraphLifetime
    work_counts: dict[str, int]
    trace: dict[str, Any] | None = None
    module_query_keys: Tensor | None = None  # Q only: final-token [B,H,M,D]
    environment_query_keys: Tensor | None = None  # Q only: final-token [B,H,E,D]

    @property
    def scene_key(self) -> str:
        """Expose the prepared-scene identity to native adapter facades."""
        return self.scene.scene_key


@dataclass(frozen=True)
class PreparedInheritedAffineQuery:
    """Heat-independent query factors; no ordinary dense K[Q,M] cache."""

    organization: PreparedInheritedOrganization
    query: InheritedQueryBatch
    phi: Tensor
    local_pairs: LocalPairPack
    local_coefficients: Tensor
    nonlinear_field: Tensor
    query_versions: tuple
    factor_versions: tuple
    identity: tuple
    work_counts: dict[str, int]


class InheritedAttentionCoreV1(nn.Module):
    """IAG-C with an explicit geometry-owned receiver cover.

    ``soft`` is a fresh representation. ``measure_entmax15`` has the same
    parameters and an absolute-age soft/entmax blend (501--800), so its e500
    child can inherit every physical weight and optimizer moment exactly.
    """

    schema = "inherited_attention_core_v1"

    def __init__(
        self,
        *,
        module_input_dim: int,
        environment_input_dim: int,
        coordinate_dim: int,
        query_input_dim: int = 0,
        operating_input_dim: int = 0,
        output_dim: int = 3,
        architecture: str = "IAG-C",
        context_dim: int = 128,
        num_heads: int = 4,
        context_depth: int = 2,
        local_hidden_dim: int = 128,
        dropout: float = 0.0,
        init_seed: int = 0,
        signed_carrier_width: int = 8,
        thermal_response: bool = False,
        normalization: str = "soft",
        scene_executor: str = "dense",
        query_executor: str = "indexed",
    ) -> None:
        super().__init__()
        if architecture not in ("IAG-C", "IAG-Q"):
            raise ValueError("Unknown inherited-attention representation.")
        if architecture == "IAG-Q" and thermal_response:
            raise ValueError("The diagnosed IAG-Q replacement is Wind-only; Thermal retains IAG-C.")
        if min(module_input_dim, environment_input_dim, coordinate_dim, output_dim,
               context_dim, num_heads, context_depth, local_hidden_dim) < 1:
            raise ValueError("Model dimensions and context depth must be positive.")
        if context_dim % num_heads or query_input_dim < 0 or operating_input_dim < 0:
            raise ValueError("Invalid head or optional feature dimensions.")
        if signed_carrier_width != 8 or type(thermal_response) is not bool:
            raise ValueError("IAG-C uses eight signed carriers per head and an explicit thermal capability.")
        if normalization not in ("soft", "measure_entmax15"):
            raise ValueError("normalization must be soft or measure_entmax15.")
        valid_executors = ((scene_executor in ("dense", "selected") and query_executor in ("indexed", "dense_reference"))
                           if architecture == "IAG-C" else
                           (scene_executor == "cached_tokens" and query_executor in ("cell_selected", "dense_reference")))
        if not valid_executors:
            raise ValueError("Unknown inherited gathering executor.")
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
        self.local_hidden_dim = int(local_hidden_dim)
        self.dropout_probability = float(dropout)
        self.init_seed = int(init_seed)
        self.signed_carrier_width = int(signed_carrier_width)
        self.thermal_response = thermal_response
        self.normalization = normalization
        self.scene_executor = scene_executor
        self.query_executor = query_executor
        self._owner_token = id(self)
        self.register_buffer("normalization_epoch", torch.tensor(0, dtype=torch.int64))

        width = self.context_dim
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            # Common module names deliberately match the D0 initialization.
            self.module_encoder = nn.Sequential(
                nn.Linear(module_input_dim + 8 * coordinate_dim, width), nn.SiLU(), nn.Linear(width, width)
            )
            self.environment_encoder = nn.Sequential(
                nn.Linear(environment_input_dim + 8 * coordinate_dim, width), nn.SiLU(), nn.Linear(width, width)
            )
            self.operating_encoder = nn.Linear(operating_input_dim, width) if operating_input_dim else None
            self.scene_summary_net = nn.Sequential(
                nn.Linear(3 * width + 3 + 2 * coordinate_dim, 2 * width), nn.SiLU(), nn.Linear(2 * width, width)
            )
            self.module_raw_projection = nn.Linear(width, width, bias=False)
            self.environment_raw_projection = nn.Linear(width, width, bias=False)
            self.module_scene_projection = nn.Linear(width, width, bias=False)
            self.environment_scene_projection = nn.Linear(width, width, bias=False)
            self.module_raw_gate = nn.Parameter(torch.tensor(0.1))
            self.environment_raw_gate = nn.Parameter(torch.tensor(0.1))
            self.module_scene_gate = nn.Parameter(torch.tensor(0.1))
            self.environment_scene_gate = nn.Parameter(torch.tensor(0.1))
            self.context_blocks = nn.ModuleList(
                StagedContextCycleV2(width, num_heads, coordinate_dim, dropout) for _ in range(context_depth)
            )
            self.query_encoder = QueryEncoderV2(query_input_dim, coordinate_dim, width)
            self.local_reader = LocalDetailReaderV2(width, coordinate_dim, local_hidden_dim, thermal=thermal_response)
            self.field_head = FieldHeadV2(width, local_hidden_dim, output_dim)
            self.module_value_norm = nn.LayerNorm(width)
            self.environment_value_norm = nn.LayerNorm(width)
            self.module_value_projection = nn.Linear(width, width)
            self.environment_value_projection = nn.Linear(width, width)
            # The bias is applied exactly once, after separate typed gathers.
            self.gather_output = nn.Linear(2 * width, width)
            self.source_carrier = (
                nn.Sequential(nn.LayerNorm(width), nn.Linear(width, num_heads * signed_carrier_width))
                if thermal_response else None
            )
            self.response_phi = (
                nn.Sequential(nn.Linear(2 * width + local_hidden_dim, width), nn.SiLU(),
                              nn.Linear(width, num_heads * signed_carrier_width))
                if thermal_response else None
            )
        self._initialize_named_parameters()

    @property
    def thermal_response_rank(self) -> int:
        """Native adapter capability; not an H6 mode-count configuration."""
        return self.signed_carrier_width if self.thermal_response else 0

    def _initialize_named_parameters(self) -> None:
        with torch.no_grad():
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    digest = hashlib.blake2b(f"{self.init_seed}:{name}.weight".encode(), digest_size=8).digest()
                    generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest, "little") % (2**31 - 1))
                    weight = torch.empty(tuple(module.weight.shape), dtype=torch.float32)
                    nn.init.xavier_uniform_(weight, generator=generator)
                    if name.endswith("geometry_score.2"):
                        weight.mul_(0.01)
                    module.weight.copy_(weight.to(module.weight))
                    if module.bias is not None:
                        module.bias.zero_()
                elif isinstance(module, nn.LayerNorm):
                    if module.weight is not None:
                        module.weight.fill_(1.0)
                    if module.bias is not None:
                        module.bias.zero_()

    def get_config(self) -> dict[str, Any]:
        return {
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
            "local_hidden_dim": self.local_hidden_dim,
            "dropout": self.dropout_probability,
            "init_seed": self.init_seed,
            "signed_carrier_width": self.signed_carrier_width,
            "thermal_response": self.thermal_response,
            "normalization": self.normalization,
            "scene_executor": self.scene_executor,
            "query_executor": self.query_executor,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> InheritedAttentionCoreV1:
        values = dict(config)
        if values.pop("schema", None) != cls.schema:
            raise ValueError(f"Expected {cls.schema} configuration.")
        return cls(**values)

    def set_inherited_attention_training_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Absolute training age must be a nonnegative integer.")
        # Epoch zero is a read-only loaded-checkpoint evaluation, not a reset.
        if epoch and int(self.normalization_epoch) != epoch:
            self.normalization_epoch.fill_(epoch)

    def _blend(self) -> float:
        if self.normalization == "soft":
            return 0.0
        epoch = int(self.normalization_epoch)
        return min(1.0, max(0.0, (epoch - 500) / 300.0))

    def _configuration(self) -> tuple:
        return (*tuple(sorted(self.get_config().items())), ("training_mode", bool(self.training)))

    def _parameter_versions(self) -> tuple:
        return tuple(_stamp(parameter) for parameter in self.parameters())

    def _buffer_versions(self) -> tuple:
        return tuple(_stamp(buffer) for buffer in self.buffers())

    @staticmethod
    def _identity(scene: SceneBatch) -> tuple:
        return scene.scene_key, scene.normalizer_key, scene.coordinate_frame_key

    @staticmethod
    def _measure_summary(states: Tensor, measure: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        dtype = torch.float64 if states.dtype == torch.float32 else states.dtype
        weights = torch.where(valid, measure.to(dtype), torch.zeros_like(measure, dtype=dtype))
        total = weights.sum(dim=1)
        numerator = torch.einsum("bn,bnd->bd", weights, states.to(dtype))
        mean = numerator / total[:, None].clamp_min(torch.finfo(dtype).tiny)
        mean = torch.where(total[:, None] > 0, mean, torch.zeros_like(mean))
        return mean.to(states.dtype), total.to(states.dtype)

    @staticmethod
    def _log_measure_ratio(total: Tensor, reference: Tensor) -> Tensor:
        total64 = total.to(torch.float64)
        reference64 = reference.to(torch.float64).clamp_min(torch.finfo(torch.float64).tiny)
        positive = total64 > 0
        value = torch.nn.functional.softplus(
            torch.log(torch.where(positive, total64, torch.ones_like(total64))) - torch.log(reference64)
        )
        return torch.where(positive, value, torch.zeros_like(value)).to(total.dtype)

    def _summary(self, scene: SceneBatch, module: Tensor, environment: Tensor,
                 operating: Tensor, scale: Tensor) -> Tensor:
        mv = scene.module_mask & (scene.module_measure > 0)
        ev = scene.environment_mask & (scene.environment_measure > 0)
        mm, mt = self._measure_summary(module, scene.module_measure, mv)
        em, et = self._measure_summary(environment, scene.environment_measure, ev)
        ms, _ = self._measure_summary(scene.module_characteristic_lengths.to(module) / scale[:, None], scene.module_measure, mv)
        es, _ = self._measure_summary(scene.environment_characteristic_lengths.to(environment) / scale[:, None], scene.environment_measure, ev)
        mr = scene.module_measure_reference
        er = scene.environment_measure_reference
        if mr is None:
            mr = mt.new_ones(scene.batch_size)
        if er is None:
            er = et.new_ones(scene.batch_size)
        return self.scene_summary_net(torch.cat((
            mm, em, operating, scene.module_mask.sum(dim=1, dtype=module.dtype)[:, None],
            self._log_measure_ratio(mt, mr)[:, None], self._log_measure_ratio(et, er)[:, None], ms, es,
        ), dim=-1))

    def _atlas(self, snapshot: RouteScoreSnapshot, blend: float) -> TypedAssociationAtlas:
        if blend == 0.0:
            return build_typed_atlas(snapshot, normalization="softmax")
        sparse = build_typed_atlas(snapshot, normalization="entmax15")
        if blend == 1.0:
            return sparse
        soft = build_typed_atlas(snapshot, normalization="softmax")
        return replace(sparse, probabilities=(1.0 - blend) * soft.probabilities + blend * sparse.probabilities,
                       normalization="softmax_entmax15_blend")

    def _typed_values(self, states: Tensor, scene: SceneBatch, kind: str) -> TypedTokenValues:
        module = kind == "module"
        norm = self.module_value_norm if module else self.environment_value_norm
        projection = self.module_value_projection if module else self.environment_value_projection
        valid = scene.module_mask if module else scene.environment_mask
        values = projection(norm(states)) * valid[:, :, None].to(states.dtype)
        values = values.reshape(scene.batch_size, states.shape[1], self.num_heads, self.context_dim // self.num_heads)
        return TypedTokenValues(kind=kind, values=values.permute(0, 2, 1, 3),
                                token_ids=scene.module_ids if module else scene.environment_ids,
                                valid=valid, measure=scene.module_measure if module else scene.environment_measure)

    @staticmethod
    def _factors(organization: PreparedInheritedOrganization) -> tuple:
        return _versions((organization.module_score_snapshot, organization.environment_score_snapshot,
                          organization.module_atlas, organization.environment_atlas,
                          organization.module_values, organization.environment_values,
                          organization.module_anchor_values, organization.environment_anchor_values,
                          organization.module_states, organization.environment_states, organization.operating_state,
                          organization.frame_origin, organization.frame_scale, organization.source_carriers,
                          organization.source_edge_indices, organization.module_query_keys,
                          organization.environment_query_keys))

    def _route_keys(self, route: nn.Module, states: Tensor) -> Tensor:
        projected = route.key(route.key_norm(states))
        shaped = projected.reshape(states.shape[0], states.shape[1], self.num_heads, self.context_dim // self.num_heads)
        return F.normalize(shaped.transpose(1, 2), dim=-1, eps=1.0e-6)

    def _route_queries(self, route: nn.Module, queries: Tensor) -> Tensor:
        projected = route.query(route.query_norm(queries))
        shaped = projected.reshape(queries.shape[0], queries.shape[1], self.num_heads, self.context_dim // self.num_heads)
        return F.normalize(shaped.transpose(1, 2), dim=-1, eps=1.0e-6) * route.temperature[None, :, None, None]

    def prepare(self, scene: SceneBatch, *, return_trace: bool = False) -> PreparedInheritedOrganization:
        if torch.is_inference_mode_enabled() or any(_stamp(tensor)[-1] < 0 for tensor in _tensors(scene)):
            raise ValueError("IAG prepared organizations require versioned tensors; use no_grad, not inference_mode.")
        scene.validate(coordinate_dim=self.coordinate_dim, module_input_dim=self.module_input_dim,
                       environment_input_dim=self.environment_input_dim, operating_input_dim=self.operating_input_dim)
        if any(value is None for value in (scene.module_characteristic_lengths, scene.environment_characteristic_lengths,
                                          scene.coordinate_origin, scene.coordinate_scale)):
            raise ValueError("IAG requires explicit token sizes and a fixed native coordinate frame.")
        parameter = next(self.parameters())
        if any(tensor.device != parameter.device for tensor in _tensors(scene)):
            raise ValueError("Every scene tensor must be on the model device.")
        dtype = parameter.dtype
        origin, scale = scene.coordinate_origin.to(dtype), scene.coordinate_scale.to(dtype)
        if not torch.isfinite(origin).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("The native frame must be finite and have positive scale.")
        operating = (self.operating_encoder(scene.operating_features.to(dtype)) if self.operating_encoder is not None
                     else scene.module_features.new_zeros(scene.batch_size, self.context_dim, dtype=dtype))
        raw_module = self.module_encoder(torch.cat((scene.module_features.to(dtype),
            coordinate_gamma(scene.module_coordinates.to(dtype), origin, scale),
            scene.module_characteristic_lengths.to(dtype) / scale[:, None]), dim=-1)) + operating[:, None]
        raw_environment = self.environment_encoder(torch.cat((scene.environment_features.to(dtype),
            coordinate_gamma(scene.environment_coordinates.to(dtype), origin, scale),
            scene.environment_characteristic_lengths.to(dtype) / scale[:, None]), dim=-1)) + operating[:, None]
        raw_module = raw_module * scene.module_mask[:, :, None].to(dtype)
        raw_environment = raw_environment * scene.environment_mask[:, :, None].to(dtype)
        module, environment = raw_module, raw_environment
        summary = self._summary(scene, module, environment, operating, scale)
        mc, ec = scene.module_coordinates.to(dtype), scene.environment_coordinates.to(dtype)
        ml, el = scene.module_characteristic_lengths.to(dtype), scene.environment_characteristic_lengths.to(dtype)
        geometry = {
            name: pair_score_features(destination, donor, scale, dl, sl)[0]
            for name, destination, donor, dl, sl in (
                ("mm", mc, mc, ml, ml), ("em", ec, mc, el, ml),
                ("ee", ec, ec, el, el), ("me", mc, ec, ml, el),
            )
        }
        traces = []
        final_trace = None
        for index, block in enumerate(self.context_blocks):
            module = module + self.module_raw_gate * self.module_raw_projection(raw_module)
            module = module + self.module_scene_gate * self.module_scene_projection(summary)[:, None]
            environment = environment + self.environment_raw_gate * self.environment_raw_projection(raw_environment)
            environment = environment + self.environment_scene_gate * self.environment_scene_projection(summary)[:, None]
            module = module * scene.module_mask[:, :, None].to(dtype)
            environment = environment * scene.environment_mask[:, :, None].to(dtype)
            module, environment, final_trace = block(
                module, environment, scene, scale, return_attention=return_trace,
                export_route_scores=index == self.context_depth - 1, geometry_features=geometry,
            )
            summary = self._summary(scene, module, environment, operating, scale)
            if return_trace:
                traces.append(final_trace)
        assert final_trace is not None
        em = final_trace["em_score_snapshot"]
        ee = final_trace["ee_score_snapshot"]
        blend = self._blend()
        ma, ea = self._atlas(em, blend), self._atlas(ee, blend)
        mv, ev = self._typed_values(module, scene, "module"), self._typed_values(environment, scene, "environment")
        mg = (prepare_anchor_values(ma, mv, executor=self.scene_executor) if self.architecture == "IAG-C" else None)
        eg = (prepare_anchor_values(ea, ev, executor=self.scene_executor) if self.architecture == "IAG-C" else None)
        mk = self._route_keys(self.context_blocks[-1].em, module) if self.architecture == "IAG-Q" else None
        ek = self._route_keys(self.context_blocks[-1].ee, environment) if self.architecture == "IAG-Q" else None
        carriers = None
        if self.source_carrier is not None:
            carriers = self.source_carrier(module).reshape(scene.batch_size, scene.num_modules,
                self.num_heads, self.signed_carrier_width).permute(0, 2, 1, 3)
            carriers = carriers * ma.donor_valid[:, None, :, None].to(carriers.dtype)
        source_edges = (torch.nonzero(ma.probabilities > 0, as_tuple=True)
                        if self.scene_executor == "selected" and carriers is not None else None)
        life = _GraphLifetime()
        for tensor in (module, environment):
            if tensor.requires_grad:
                tensor.register_hook(life.mark_consumed)
        batch, modules, anchors = scene.batch_size, scene.num_modules, scene.num_environment_tokens
        eligible = int((ma.donor_valid.sum(1) * ma.destination_valid.sum(1)).sum()
                       + (ea.donor_valid.sum(1) * ea.destination_valid.sum(1)).sum()) * self.num_heads
        positive = int((ma.probabilities > 0).sum() + (ea.probabilities > 0).sum())
        dense = batch * self.num_heads * anchors * (modules + anchors)
        work = {
            "contextualizer_score_entries": batch * self.num_heads * self.context_depth *
                (modules * modules + 2 * anchors * modules + anchors * anchors),
            "atlas_score_entries": dense, "atlas_eligible_associations": eligible,
            "atlas_positive_associations": positive,
            "scene_value_executed_associations": (0 if self.architecture == "IAG-Q" else
                                                   dense if self.scene_executor == "dense" else positive),
            "typed_projected_token_rows": batch * (modules + anchors),
            "typed_projected_query_key_rows": batch * (modules + anchors) if self.architecture == "IAG-Q" else 0,
            "far_pair_value_rows": 0, "direct_source_rows": 0,
        }
        organization = PreparedInheritedOrganization(
            scene, module, environment, operating, origin, scale, summary, em, ee, ma, ea, mv, ev, mg, eg,
            carriers, source_edges, blend, self._owner_token, self._parameter_versions(), self._buffer_versions(),
            scene.tensor_version_stamp(), (), self._identity(scene), self._configuration(), life, work,
            {"context_cycles": traces, "score_stage": "last_em_ee", "payload_stage": "final_tokens"}
            if return_trace else None,
            module_query_keys=mk, environment_query_keys=ek,
        )
        organization.factor_versions = self._factors(organization)
        return organization

    def intervene_associations(
        self,
        organization: PreparedInheritedOrganization,
        module_probabilities: Tensor,
        environment_probabilities: Tensor,
        *,
        label: str,
    ) -> PreparedInheritedOrganization:
        """Rebuild read-only control factors from explicitly supplied atlases.

        Context, token values, geometry and all learned weights stay fixed.
        Sub-probability rows permit a declared contribution removal without
        renormalizing it into a different normalized policy. These factors
        have their own ownership stamps; the original cache is untouched.
        """
        if self.training or torch.is_grad_enabled():
            raise ValueError("Association interventions require eval mode and no_grad; they are not a training arm.")
        if torch.is_inference_mode_enabled() or any(
            _stamp(tensor)[-1] < 0 for tensor in (module_probabilities, environment_probabilities)
        ):
            raise ValueError("Association interventions require versioned tensors; use no_grad, not inference_mode.")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("An association intervention needs an explicit descriptive label.")
        self.validate_prepared_state(organization)
        atlases = []
        for original, probabilities in (
            (organization.module_atlas, module_probabilities),
            (organization.environment_atlas, environment_probabilities),
        ):
            if (probabilities.shape != original.probabilities.shape or
                    probabilities.device != original.probabilities.device or
                    probabilities.dtype != original.probabilities.dtype):
                raise ValueError("Intervention probabilities must retain the exact typed axes, dtype and device.")
            if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
                raise ValueError("Intervention probabilities must be finite and nonnegative.")
            allowed = original.destination_valid[:, None, :, None] & original.donor_valid[:, None, None, :]
            if (probabilities.masked_select(~allowed.expand_as(probabilities)) != 0).any():
                raise ValueError("An intervention cannot introduce masked source or destination associations.")
            if (probabilities.sum(-1) > 1.0 + 3.0e-6).any():
                raise ValueError("Intervention rows must be probabilities or declared unnormalized removals.")
            atlases.append(replace(original, probabilities=probabilities, normalization="diagnostic_intervention"))
        module, environment = atlases
        module_values = (prepare_anchor_values(module, organization.module_values, executor=self.scene_executor)
                         if self.architecture == "IAG-C" else None)
        environment_values = (prepare_anchor_values(environment, organization.environment_values, executor=self.scene_executor)
                              if self.architecture == "IAG-C" else None)
        edges = (torch.nonzero(module.probabilities > 0, as_tuple=True)
                 if self.scene_executor == "selected" and organization.source_carriers is not None else None)
        positive = int((module.probabilities > 0).sum() + (environment.probabilities > 0).sum())
        work = dict(organization.work_counts)
        work.update(contextualizer_score_entries=0, atlas_score_entries=0, typed_projected_token_rows=0,
                    atlas_positive_associations=positive,
                    scene_value_executed_associations=(0 if self.architecture == "IAG-Q" else
                        module.probabilities.numel() + environment.probabilities.numel()
                        if self.scene_executor == "dense" else positive))
        result = replace(organization, module_atlas=module, environment_atlas=environment,
                         module_anchor_values=module_values, environment_anchor_values=environment_values,
                         source_edge_indices=edges, graph_lifetime=_GraphLifetime(), work_counts=work,
                         trace={**(organization.trace or {}), "association_intervention": label,
                                "intervention_reuses_fixed_context_and_token_values": True})
        result.factor_versions = self._factors(result)
        return result

    def validate_prepared_state(self, organization: PreparedInheritedOrganization, query: QueryBatch | None = None) -> None:
        if not isinstance(organization, PreparedInheritedOrganization) or organization.owner_token != self._owner_token:
            raise ValueError("Prepared IAG organization belongs to another model instance.")
        if organization.graph_lifetime.consumed:
            raise ValueError("Prepared autograd graph was consumed; prepare a new scene forward.")
        if organization.parameter_versions != self._parameter_versions() or organization.buffer_versions != self._buffer_versions():
            raise ValueError("Model weights or normalization buffers changed; prepare the scene again.")
        if organization.configuration != self._configuration() or organization.identity != self._identity(organization.scene):
            raise ValueError("Prepared configuration or native scene identity changed.")
        if organization.scene_versions != organization.scene.tensor_version_stamp() or organization.factor_versions != self._factors(organization):
            raise ValueError("Prepared scene/atlas/value factors changed; prepare the scene again.")
        if query is None:
            return
        query.validate(batch_size=organization.scene.batch_size, coordinate_dim=self.coordinate_dim, query_input_dim=self.query_input_dim)
        if torch.is_inference_mode_enabled() or any(_stamp(tensor)[-1] < 0 for tensor in _tensors(query)):
            raise ValueError("IAG receiver caches require versioned tensors; use no_grad, not inference_mode.")
        if any(tensor.device != organization.module_states.device for tensor in _tensors(query)):
            raise ValueError("Every receiver tensor must be on the prepared model device.")
        if not isinstance(query, InheritedQueryBatch) or query.receiver_cover is None:
            raise ValueError("IAG queries require the native geometry-owned ReceiverCover.")
        if query.scene_key is not None and query.scene_key != organization.scene.scene_key:
            raise ValueError("Receiver scene identity does not match the prepared scene.")
        if query.coordinate_frame_key != organization.scene.coordinate_frame_key or query.receiver_cover.frame_key != query.coordinate_frame_key:
            raise ValueError("Receiver cover and scene require the same native frame identity.")
        if query.query_ids is None or not torch.equal(query.query_ids, query.receiver_cover.query_ids):
            raise ValueError("Receiver cover IDs do not match the native query order.")
        if query.query_mask is not None and (query.receiver_cover.query_mask is None or
                                            not torch.equal(query.query_mask, query.receiver_cover.query_mask)):
            raise ValueError("Receiver cover validity does not match the query mask.")
        query.receiver_cover.validate_partition()
        validate_cover_anchor_identity(query.receiver_cover, organization.scene.environment_ids)

    def _gather(self, values: PreparedAnchorValues | Tensor, cover: ReceiverCover, anchor_ids: Tensor) -> Tensor:
        reader = gather_cover_indexed if self.query_executor == "indexed" else gather_cover_dense_reference
        return reader(values, cover, anchor_ids=anchor_ids, validate_identity=False)

    def _decode_features(self, organization: PreparedInheritedOrganization, query: InheritedQueryBatch,
                         local_gates: Tensor | None = None, *, field_head: bool = True,
                         return_query_trace: bool = False) -> tuple:
        self.validate_prepared_state(organization, query)
        scene, cover = organization.scene, query.receiver_cover
        assert cover is not None
        if local_gates is not None and (local_gates.shape != scene.module_mask.shape or not torch.isfinite(local_gates).all()):
            raise ValueError("Local intervention gates must be finite [B,M].")
        qs = self.query_encoder(query, organization.operating_state, organization.frame_origin, organization.frame_scale)
        query_work, query_trace = {}, None
        if self.architecture == "IAG-C":
            assert organization.module_anchor_values is not None and organization.environment_anchor_values is not None
            mc = self._gather(organization.module_anchor_values, cover, scene.environment_ids).reshape(query.batch_size, query.num_queries, self.context_dim)
            ec = self._gather(organization.environment_anchor_values, cover, scene.environment_ids).reshape_as(mc)
        else:
            assert organization.module_query_keys is not None and organization.environment_query_keys is not None
            plan = build_cell_query_plan(cover) if self.query_executor == "cell_selected" else None
            mc, mw, mt = query_conditioned_gather(organization.module_atlas, organization.module_values,
                self._route_queries(self.context_blocks[-1].em, qs), organization.module_query_keys,
                cover, executor=self.query_executor, plan=plan, return_trace=return_query_trace)
            ec, ew, et = query_conditioned_gather(organization.environment_atlas, organization.environment_values,
                self._route_queries(self.context_blocks[-1].ee, qs), organization.environment_query_keys,
                cover, executor=self.query_executor, plan=plan, return_trace=return_query_trace)
            mc = mc.reshape(query.batch_size, query.num_queries, self.context_dim)
            ec = ec.reshape_as(mc)
            query_work = {**{f"module_{key}": value for key, value in mw.items()},
                          **{f"environment_{key}": value for key, value in ew.items()},
                          "query_geometry_grouping_operations": int(plan is not None),
                          "query_geometry_cell_count": 0 if plan is None else int(plan.counts.numel())}
            if return_query_trace:
                query_trace = {"module": mt, "environment": et,
                               "query_weight_law": "normalize(beta*exp(shared_final_route_QK))",
                               "diagnostic_query_limit": 16}
        global_context = self.gather_output(torch.cat((mc, ec), dim=-1))
        reference = scene.module_measure_reference
        if reference is None:
            reference = scene.module_measure.new_ones(scene.batch_size)
        local, pairs, coefficients = self.local_reader(scene, query, organization.module_states, qs,
            organization.frame_scale, reference.to(qs), local_gates=local_gates, execution_backend="legacy")
        nonlinear = self.field_head(qs, global_context, local) if field_head else None
        phi = (None if self.response_phi is None else self.response_phi(torch.cat((qs, global_context, local), dim=-1))
               .reshape(query.batch_size, query.num_queries, self.num_heads, self.signed_carrier_width))
        if query.query_mask is not None:
            if nonlinear is not None:
                nonlinear = nonlinear * query.query_mask[:, :, None].to(nonlinear.dtype)
            if phi is not None:
                phi = phi * query.query_mask[:, :, None, None].to(phi.dtype)
        return nonlinear, phi, pairs, coefficients, mc, ec, query_work, query_trace

    @staticmethod
    def _dtype(requested: torch.dtype | None, fallback: torch.dtype) -> torch.dtype:
        dtype = fallback if requested is None else requested
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Affine accumulation supports float32 or float64.")
        return dtype

    @staticmethod
    def _local_apply(pack: LocalPairPack, coefficients: Tensor, controls: Tensor,
                     batch: int, queries: int, dtype: torch.dtype) -> Tensor:
        result = controls.new_zeros(batch * queries, dtype=dtype)
        if pack.count:
            values = pack.support_weight.to(dtype) * coefficients.to(dtype) * controls[pack.batch_index, pack.module_index].to(dtype)
            result = result.index_add(0, pack.batch_index * queries + pack.query_index, values)
        return result.reshape(batch, queries)

    def _validate_controls(self, organization: PreparedInheritedOrganization, controls: Tensor) -> None:
        if controls.shape != organization.scene.module_mask.shape or controls.device != organization.module_states.device:
            raise ValueError("Heating controls must match the prepared [B,M] physical source axis and device.")
        if not torch.isfinite(controls).all():
            raise ValueError("Heating controls must be finite.")

    @staticmethod
    def _density(organization: PreparedInheritedOrganization, dtype: torch.dtype) -> Tensor:
        atlas = organization.module_atlas
        measure = atlas.normalized_measure.to(dtype)
        safe = torch.where(atlas.donor_valid, measure, torch.ones_like(measure))
        density = atlas.probabilities.to(dtype) / safe[:, None, None, :]
        density = density * atlas.donor_valid[:, None, None, :].to(dtype)
        if not torch.isfinite(density).all():
            raise ValueError("Measure-relative response is ill-conditioned in the requested arithmetic; no measure clamp is applied.")
        return density

    def _anchor_response(self, organization: PreparedInheritedOrganization, controls: Tensor, dtype: torch.dtype) -> Tensor:
        density = self._density(organization, dtype)
        assert organization.source_carriers is not None
        carriers = organization.source_carriers.to(dtype) * controls.to(dtype)[:, None, :, None]
        if self.scene_executor == "dense":
            return torch.einsum("bhem,bhmr->bher", density, carriers)
        assert organization.source_edge_indices is not None
        b, h, e, m = organization.source_edge_indices
        batch, heads, anchors, _modules = density.shape
        values = density[b, h, e, m, None] * carriers[b, h, m]
        output = carriers.new_zeros(batch * heads * anchors, self.signed_carrier_width)
        return output.index_add(0, (b * heads + h) * anchors + e, values).reshape(batch, heads, anchors, self.signed_carrier_width)

    def _apply_controls(self, organization: PreparedInheritedOrganization, query: InheritedQueryBatch, phi: Tensor,
               pack: LocalPairPack, coefficients: Tensor, controls: Tensor, dtype: torch.dtype) -> Tensor:
        self._validate_controls(organization, controls)
        response = self._anchor_response(organization, controls, dtype)
        assert query.receiver_cover is not None
        cover = replace(query.receiver_cover, weights=query.receiver_cover.weights.to(dtype))
        gathered = self._gather(response, cover, organization.scene.environment_ids)
        far = (gathered * phi.to(dtype)).sum(dim=(-1, -2))
        temperature = far + self._local_apply(pack, coefficients, controls, query.batch_size, query.num_queries, dtype)
        if query.query_mask is not None:
            temperature = temperature * query.query_mask.to(dtype)
        return temperature

    def _query_work(self, organization: PreparedInheritedOrganization, query: InheritedQueryBatch,
                    pack: LocalPairPack, *, controls: bool, field: bool, features: bool = True) -> dict[str, int]:
        cover = query.receiver_cover
        assert cover is not None
        allocated = query.batch_size * query.num_queries * cover.slots * self.num_heads
        positive = int(((cover.weights > 0) & cover.validity).sum()) * self.num_heads
        actual = allocated if self.query_executor == "indexed" else query.batch_size * query.num_queries * organization.scene.num_environment_tokens * self.num_heads
        return {
            "receiver_rows": query.batch_size * query.num_queries,
            "cover_slots": allocated, "cover_positive_slots": positive,
            "cover_executed_anchor_reads": (actual * (2 * int(features) + int(controls))
                                             if self.architecture == "IAG-C" else 0),
            "scene_response_executed_associations": (0 if not controls else
                int(organization.module_atlas.probabilities.numel()) if self.scene_executor == "dense" else
                int((organization.module_atlas.probabilities > 0).sum())),
            "local_rows": pack.count if features else 0, "local_contraction_rows": pack.count if controls else 0,
            "field_head_rows": query.batch_size * query.num_queries if field else 0,
            "far_pair_value_rows": 0, "direct_source_rows": 0, "edge_rows": 0,
            "direct_dense_reference_entries": 0,
        }

    def decode(self, organization: PreparedInheritedOrganization, query: InheritedQueryBatch,
               controls: Tensor | None = None, *, accumulation_dtype: torch.dtype | None = None,
               return_trace: bool = False, local_gates: Tensor | None = None) -> CoreOutput:
        nonlinear, phi, pairs, coefficients, mc, ec, query_work, query_trace = self._decode_features(
            organization, query, local_gates, return_query_trace=return_trace)
        temperature = None
        if controls is not None:
            if phi is None:
                raise ValueError("This IAG core has no affine thermal response.")
            temperature = self._apply_controls(organization, query, phi, pairs, coefficients, controls,
                                      self._dtype(accumulation_dtype, controls.dtype))
        work = self._query_work(organization, query, pairs, controls=controls is not None, field=True)
        work.update(query_work)
        trace = ({"receiver_cover": query.receiver_cover, "module_context": mc, "environment_context": ec,
                  "thermal_phi": phi, "local_pairs": pairs, "local_coefficients": coefficients,
                  "query_conditioned_gather": query_trace}
                 if return_trace else None)
        return CoreOutput(nonlinear_field=nonlinear, thermal_temperature=temperature,
                          thermal_basis=phi, trace=trace, work_counts=work)

    def forward(self, scene: SceneBatch, query: InheritedQueryBatch, controls: Tensor | None = None,
                *, return_trace: bool = False, accumulation_dtype: torch.dtype | None = None) -> CoreOutput:
        return self.decode(self.prepare(scene, return_trace=return_trace), query, controls,
                           return_trace=return_trace, accumulation_dtype=accumulation_dtype)

    def apply_increment(self, organization: PreparedInheritedOrganization, query: InheritedQueryBatch,
                        delta_controls: Tensor, *, accumulation_dtype: torch.dtype | None = None,
                        local_gates: Tensor | None = None, return_work_counts: bool = False) -> Tensor | tuple:
        if not self.thermal_response:
            raise ValueError("This IAG core has no affine thermal response.")
        _field, phi, pairs, coefficients, _mc, _ec, _work, _trace = self._decode_features(
            organization, query, local_gates, field_head=False)
        assert phi is not None
        result = self._apply_controls(organization, query, phi, pairs, coefficients, delta_controls,
                             self._dtype(accumulation_dtype, delta_controls.dtype))
        return (result, self._query_work(organization, query, pairs, controls=True, field=False)) if return_work_counts else result

    def heat_jvp(self, organization: PreparedInheritedOrganization, query: InheritedQueryBatch,
                 tangent: Tensor, *, accumulation_dtype: torch.dtype | None = None) -> Tensor:
        return self.apply_increment(organization, query, tangent, accumulation_dtype=accumulation_dtype)

    @staticmethod
    def _affine_factors(prepared: PreparedInheritedAffineQuery) -> tuple:
        return _versions((prepared.phi, prepared.local_pairs, prepared.local_coefficients, prepared.nonlinear_field))

    def prepare_affine_query(self, organization: PreparedInheritedOrganization, query: InheritedQueryBatch,
                             *, local_gates: Tensor | None = None) -> PreparedInheritedAffineQuery:
        if not self.thermal_response:
            raise ValueError("This IAG core has no affine thermal response.")
        field, phi, pairs, coefficients, _mc, _ec, _work, _trace = self._decode_features(organization, query, local_gates)
        assert phi is not None
        query_versions = _versions(query)
        if any(stamp[-1] < 0 for stamp in (*query_versions, *organization.scene_versions, *organization.factor_versions)):
            raise ValueError("Reusable affine caches require versioned tensors; prepare under no_grad, not inference_mode.")
        prepared = PreparedInheritedAffineQuery(organization, query, phi, pairs, coefficients, field,
            query_versions, (), (query.scene_key, query.coordinate_frame_key, query.receiver_key),
            self._query_work(organization, query, pairs, controls=False, field=True))
        return replace(prepared, factor_versions=self._affine_factors(prepared))

    def _validate_affine(self, prepared: PreparedInheritedAffineQuery) -> None:
        if not isinstance(prepared, PreparedInheritedAffineQuery):
            raise TypeError("Expected an IAG prepared affine query.")
        self.validate_prepared_state(prepared.organization, prepared.query)
        if prepared.query_versions != _versions(prepared.query) or prepared.factor_versions != self._affine_factors(prepared):
            raise ValueError("Receiver, cover or prepared affine factors changed; prepare the query again.")
        if prepared.identity != (prepared.query.scene_key, prepared.query.coordinate_frame_key, prepared.query.receiver_key):
            raise ValueError("Prepared native receiver identity changed.")

    def apply_prepared_affine(self, prepared_query: PreparedInheritedAffineQuery, controls: Tensor,
                              *, accumulation_dtype: torch.dtype | None = None, check_envelope: bool = True) -> CoreOutput:
        # IAG-C has full inherited access and no sparse heat-envelope policy.
        self._validate_affine(prepared_query)
        temperature = self._apply_controls(prepared_query.organization, prepared_query.query, prepared_query.phi,
            prepared_query.local_pairs, prepared_query.local_coefficients, controls,
            self._dtype(accumulation_dtype, controls.dtype))
        work = self._query_work(prepared_query.organization, prepared_query.query, prepared_query.local_pairs,
                                controls=True, field=False, features=False)
        work["reused_query_factor_entries"] = prepared_query.phi.numel()
        return CoreOutput(nonlinear_field=prepared_query.nonlinear_field, thermal_temperature=temperature,
                          thermal_basis=prepared_query.phi, work_counts=work)

    def heat_vjp_prepared_affine(self, prepared_query: PreparedInheritedAffineQuery, cotangent: Tensor,
                                 *, accumulation_dtype: torch.dtype | None = None) -> Tensor:
        self._validate_affine(prepared_query)
        organization, query = prepared_query.organization, prepared_query.query
        if cotangent.shape != query.coordinates.shape[:2] or cotangent.device != query.coordinates.device:
            raise ValueError("Cotangent must match native [B,Q] receivers and device.")
        if not torch.isfinite(cotangent).all():
            raise ValueError("Cotangent must be finite.")
        dtype = self._dtype(accumulation_dtype, cotangent.dtype)
        cover = query.receiver_cover
        assert cover is not None and organization.source_carriers is not None
        batch, heads, anchors, _ = organization.module_atlas.probabilities.shape
        weights = cotangent.to(dtype)
        if query.query_mask is not None:
            weights = weights * query.query_mask.to(dtype)
        adjoint = weights[:, :, None, None] * prepared_query.phi.to(dtype)
        # Indexed transpose of Lambda: Q*C rows, never Q*E*R expansion.
        indices = cover.anchor_indices[:, None].expand(batch, heads, query.num_queries, cover.slots)
        b = torch.arange(batch, device=indices.device)[:, None, None, None]
        h = torch.arange(heads, device=indices.device)[None, :, None, None]
        flat_indices = ((b * heads + h) * anchors + indices).reshape(-1)
        values = adjoint.permute(0, 2, 1, 3)[:, :, :, None, :] * (
            cover.weights.to(dtype) * cover.validity.to(dtype))[:, None, :, :, None]
        anchor_adjoint = weights.new_zeros(batch * heads * anchors, self.signed_carrier_width)
        anchor_adjoint = anchor_adjoint.index_add(0, flat_indices, values.reshape(-1, self.signed_carrier_width))
        anchor_adjoint = anchor_adjoint.reshape(batch, heads, anchors, self.signed_carrier_width)
        density = self._density(organization, dtype)
        if self.scene_executor == "dense":
            donor_adjoint = torch.einsum("bhem,bher->bhmr", density, anchor_adjoint)
        else:
            assert organization.source_edge_indices is not None
            sb, sh, se, sm = organization.source_edge_indices
            contributions = density[sb, sh, se, sm, None] * anchor_adjoint[sb, sh, se]
            donor_adjoint = weights.new_zeros(batch * heads * organization.scene.num_modules, self.signed_carrier_width)
            donor_adjoint = donor_adjoint.index_add(0,
                (sb * heads + sh) * organization.scene.num_modules + sm, contributions).reshape(
                    batch, heads, organization.scene.num_modules, self.signed_carrier_width)
        gradient = (donor_adjoint * organization.source_carriers.to(dtype)).sum(dim=(1, 3))
        pack = prepared_query.local_pairs
        if pack.count:
            local = weights[pack.batch_index, pack.query_index] * pack.support_weight.to(dtype) * prepared_query.local_coefficients.to(dtype)
            gradient = gradient.reshape(-1).index_add(0, pack.batch_index * organization.scene.num_modules + pack.module_index,
                local).reshape(batch, organization.scene.num_modules)
        return gradient

    def heat_vjp(self, organization: PreparedInheritedOrganization, query: InheritedQueryBatch,
                 cotangent: Tensor, *, accumulation_dtype: torch.dtype | None = None) -> Tensor:
        return self.heat_vjp_prepared_affine(self.prepare_affine_query(organization, query), cotangent,
                                            accumulation_dtype=accumulation_dtype)

    def export_source_kernels(self, organization: PreparedInheritedOrganization, query: InheritedQueryBatch,
                              *, accumulation_dtype: torch.dtype | None = None,
                              local_gates: Tensor | None = None, max_entries: int = 1_048_576) -> Tensor:
        """Explicit bounded diagnostic K export; never used by ordinary decode."""
        if query.batch_size * query.num_queries * organization.scene.num_modules > max_entries:
            raise ValueError("Explicit source-kernel export exceeds its bounded diagnostic budget.")
        if not self.thermal_response:
            raise ValueError("This IAG core has no affine thermal response.")
        _field, phi, pack, coefficients, _mc, _ec, _work, _trace = self._decode_features(
            organization, query, local_gates, field_head=False)
        assert phi is not None and query.receiver_cover is not None and organization.source_carriers is not None
        dtype = self._dtype(accumulation_dtype, organization.module_states.dtype)
        cover = replace(query.receiver_cover, weights=query.receiver_cover.weights.to(dtype))
        density_at_query = self._gather(self._density(organization, dtype), cover, organization.scene.environment_ids)
        kernels = torch.einsum("bqhm,bhmr,bqhr->bqm", density_at_query,
                               organization.source_carriers.to(dtype), phi.to(dtype))
        if pack.count:
            indices = (pack.batch_index * query.num_queries + pack.query_index) * organization.scene.num_modules + pack.module_index
            kernels = kernels.reshape(-1).index_add(0, indices, pack.support_weight.to(dtype) * coefficients.to(dtype)).reshape_as(kernels)
        if query.query_mask is not None:
            kernels = kernels * query.query_mask[:, :, None].to(dtype)
        return kernels

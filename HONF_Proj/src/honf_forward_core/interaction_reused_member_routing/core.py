"""Round13-3 interaction-reused routing over original M/E records.

The contextualizer is the existing ordered two-cycle MM/EM/EE/ME operator.
Its final applied MM and ME gates become live address-to-member coefficients;
query values are then read from the original prepared records.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..content_hypergraph.model_v2 import ContentHypergraphCoreV2
from ..content_hypergraph.contracts import PreparedOrganizationV2, QueryBatch, SceneBatch
from ..content_hypergraph.query_chunks_v1 import slice_query_v1
from ..continuous_joint.primitives import entmax15, positive_quadratic_gate
from ..joint_native.routing import ScoreCalibration


_SCHEMA = "interaction_reused_member_routing_v1"
_CAPABILITY = "joint_prescribed_records_v1"
_ROUTE_TOPOLOGY = ("mm", "em", "ee", "me")


def _tensor_version(value: Tensor) -> tuple[int, int, int]:
    try:
        version = int(value._version)
    except RuntimeError:
        version = -1
    return id(value), int(value.data_ptr()), version


def _work_count_tensor(value: int | Tensor, reference: Tensor) -> Tensor:
    """Return one detached device-side integer count without reading it on CPU."""
    if isinstance(value, Tensor):
        return value.detach().to(device=reference.device, dtype=torch.long).reshape(())
    return torch.tensor(int(value), device=reference.device, dtype=torch.long)


def _pack_work_counts(counts: Mapping[str, int | Tensor], keys: tuple[str, ...],
                      reference: Tensor) -> Tensor:
    """Pack logical work counts for device-side tile aggregation."""
    if not keys:
        return torch.empty((0,), device=reference.device, dtype=torch.long)
    static_values: list[int] = []
    dynamic_indices: list[int] = []
    dynamic_values: list[Tensor] = []
    for index, name in enumerate(keys):
        value = counts.get(name, 0)
        if isinstance(value, Tensor):
            dynamic_indices.append(index)
            dynamic_values.append(_work_count_tensor(value, reference))
            static_values.append(0)
        else:
            static_values.append(int(value))
    packed = torch.tensor(static_values, device=reference.device, dtype=torch.long)
    if dynamic_values:
        indices = torch.tensor(dynamic_indices, device=reference.device, dtype=torch.long)
        packed.index_copy_(0, indices, torch.stack(dynamic_values))
    return packed


def _unpack_work_counts(packed: Tensor, keys: tuple[str, ...]) -> dict[str, int]:
    """Convert aggregated counts to the public integer dictionary once."""
    return {name: int(value) for name, value in zip(keys, packed.detach().cpu().tolist())}


def _entmax_with_null(logits: Tensor, allowed: Tensor, temperature: float) -> Tensor:
    """Entmax over allowed records and one always-available zero-logit null."""
    if logits.shape != allowed.shape or allowed.dtype != torch.bool:
        raise ValueError("Typed scores and candidate mask must have equal shapes.")
    scaled = logits / temperature
    null = scaled.new_zeros(*scaled.shape[:-1], 1)
    probabilities = entmax15(torch.cat((scaled, null), dim=-1),
                             torch.cat((allowed, torch.ones_like(null, dtype=torch.bool)), dim=-1))
    return probabilities


def _positive_softmax(logits: Tensor, allowed: Tensor) -> Tensor:
    """All-member softmax with a representable positive mass per eligible row."""
    if logits.shape != allowed.shape:
        raise ValueError("Full typed score and eligibility axes differ.")
    if not logits.shape[-1]:
        return logits.clone()
    any_allowed = allowed.any(-1, keepdim=True)
    maximum = logits.masked_fill(~allowed, -torch.inf).amax(-1, keepdim=True)
    maximum = torch.where(any_allowed, maximum, torch.zeros_like(maximum))
    positive = torch.exp(torch.where(allowed, logits - maximum, torch.zeros_like(logits)))
    positive = positive.clamp_min(torch.finfo(logits.dtype).tiny) * allowed.to(logits.dtype)
    total = positive.sum(-1, keepdim=True)
    return torch.where(any_allowed, positive / total.clamp_min(torch.finfo(logits.dtype).tiny),
                       torch.zeros_like(positive))


class AppliedInteractionIncidence(nn.Module):
    """Live positive gates used by the contextual value reductions."""

    def __init__(self, heads: int):
        super().__init__()
        self.routes = nn.ModuleDict({name: ScoreCalibration(heads) for name in _ROUTE_TOPOLOGY})
        # A moderate active offset avoids the +7 saturated gate calibration.
        with torch.no_grad():
            for calibration in self.routes.values():
                calibration.bias.fill_(1.5)

    def forward(self, raw_logits: Tensor, allowed: Tensor, route_name: str, **metadata) -> Tensor:
        return positive_quadratic_gate(self.routes[route_name](raw_logits, allowed)) * allowed


class InteractionMemberRouter(nn.Module):
    """Shared M/E keys, values, cheap scores and one joint physical head."""

    def __init__(self, width: int, coordinate_dim: int, output_dim: int):
        super().__init__()
        geometry_width = 2 * coordinate_dim + 1
        pair_width = 2 * width + geometry_width
        self.key_M = nn.Linear(width, width, bias=False)
        self.value_M = nn.Linear(width, width, bias=False)
        self.key_E = nn.Linear(width, width, bias=False)
        self.value_E = nn.Linear(width, width, bias=False)
        self.proposal_M = nn.Sequential(nn.Linear(pair_width, width), nn.SiLU(), nn.Linear(width, 1))
        self.member_M = nn.Sequential(nn.Linear(pair_width, width), nn.SiLU(), nn.Linear(width, 1))
        self.member_E = nn.Sequential(nn.Linear(pair_width, width), nn.SiLU(), nn.Linear(width, 1))
        self.joint_trunk = nn.Sequential(
            nn.LayerNorm(3 * width), nn.Linear(3 * width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.joint_head = nn.Linear(width, output_dim)

class _SourceProjectionFacade:
    """A2-D preparation shim that reuses the routed M projections only."""

    source_read = None

    def __init__(self, router: InteractionMemberRouter):
        self.router = router

    def prepare_sources(self, states: Tensor) -> tuple[Tensor, Tensor]:
        return self.router.key_M(states), self.router.value_M(states)


@dataclass
class InteractionJointOutput:
    """Joint channels with direct Thermal adapter compatibility slices."""

    joint_fields: Tensor
    trace: dict[str, Tensor] | None = None
    work_counts: dict[str, int] | None = None

    @property
    def nonlinear_field(self) -> Tensor:
        return self.joint_fields[..., :3]

    @property
    def thermal_temperature(self) -> Tensor:
        if self.joint_fields.shape[-1] < 4:
            raise AttributeError("This joint output has no fourth compatibility channel.")
        return self.joint_fields[..., 3]

    @property
    def thermal_basis(self):
        return None


class InteractionReusedMemberRoutingCore(ContentHypergraphCoreV2):
    """Full/Route Round13-3 model with original-record readout and no heat affine."""

    schema = _SCHEMA
    input_capability = _CAPABILITY

    def __init__(self, base_model_config: Mapping[str, Any], *, member_routing_mode: str = "route",
                 input_dataset: str = "native_records", input_capability: str = _CAPABILITY,
                 entmax_temperature: float = 1.0, query_tile_size: int = 256):
        if member_routing_mode not in ("full", "route"):
            raise ValueError("member_routing_mode must be 'full' or 'route'.")
        if input_capability != self.input_capability:
            raise ValueError(f"Input capability must be {self.input_capability!r}.")
        if not isinstance(input_dataset, str) or not input_dataset:
            raise ValueError("input_dataset must be a nonempty dataset identity.")
        if not isinstance(entmax_temperature, (int, float)) or not 0.05 <= float(entmax_temperature) <= 20:
            raise ValueError("entmax_temperature must be finite and in [0.05,20].")
        if type(query_tile_size) is not int or query_tile_size < 1:
            raise ValueError("query_tile_size must be a positive integer.")
        base = dict(base_model_config)
        if base.pop("schema", None) != ContentHypergraphCoreV2.schema or base.get("architecture") != "A2-D":
            raise ValueError("Round13-3 requires a versioned A2-D encoder/context base.")
        for key in ("module_input_dim", "environment_input_dim", "coordinate_dim", "output_dim",
                    "context_dim", "num_heads", "context_depth"):
            if int(base.get(key, 0)) < 1:
                raise ValueError(f"Round13-3 requires a positive {key}.")
        if int(base["context_depth"]) != 2:
            raise ValueError("Round13-3 requires exactly two applied contextual cycles.")
        # These old reader-only settings are deliberately retired. The encoder
        # and ordered contextualizer remain the common A2-D computation.
        base["thermal_response_rank"] = 0
        base["source_value_conditioning"] = "none"
        base["source_read_policy"] = "full"
        base["architecture"] = "A2-D"
        self.member_routing_mode = member_routing_mode
        self.input_dataset = input_dataset
        self.entmax_temperature = float(entmax_temperature)
        self.query_tile_size = query_tile_size
        super().__init__(**base)

        # Retire pooled summary and historical local/heat-affine decoder paths.
        for name in ("scene_summary_net", "module_scene_projection", "environment_scene_projection"):
            if hasattr(self, name):
                delattr(self, name)
        for name in ("module_scene_gate", "environment_scene_gate"):
            if name in self._parameters:
                del self._parameters[name]
        if hasattr(self, "local_reader"):
            delattr(self, "local_reader")
        if hasattr(self, "field_head"):
            delattr(self, "field_head")

        # Keep the A2-D preparation seam, but make it use the exact M keys and
        # values cached for the new reader. It performs no query reads.
        if hasattr(self, "direct_reader"):
            delattr(self, "direct_reader")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._stable_seed("interaction_reused_member_routing"))
            self.member_router = InteractionMemberRouter(self.context_dim, self.coordinate_dim, self.output_dim)
            self.applied_incidence = AppliedInteractionIncidence(self.num_heads)
        self._initialize_member_router()
        with torch.no_grad():
            # Native M1 logits were near 2.2 at zero bias, leaving almost all
            # proposal mass on one address and a vanishing task gradient.
            self.member_router.proposal_M[-1].bias.fill_(-1.5)
        self.direct_reader = _SourceProjectionFacade(self.member_router)
        self._context_message_removal: Mapping[str, Any] | None = None
        self._joint_epoch = 0
        self._joint_update_index = 0
        self._joint_macro_clock = 0
        self.architecture = self._arm_name(member_routing_mode)
        self._set_mode_trainability()

    @staticmethod
    def _arm_name(mode: str) -> str:
        return "R13-3-Full" if mode == "full" else "R13-3-Route"

    def _initialize_member_router(self) -> None:
        with torch.no_grad():
            for name, module in self.member_router.named_modules():
                if isinstance(module, nn.Linear):
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(self._stable_seed("member_router." + name + ".weight"))
                    weight = torch.empty(tuple(module.weight.shape), dtype=torch.float32)
                    nn.init.xavier_uniform_(weight, generator=generator)
                    module.weight.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))
                    if module.bias is not None:
                        module.bias.zero_()
                elif isinstance(module, nn.LayerNorm):
                    if module.weight is not None:
                        module.weight.fill_(1.)
                    if module.bias is not None:
                        module.bias.zero_()

    @property
    def architecture_name(self) -> str:
        return self._arm_name(self.member_routing_mode)

    @property
    def dropped_parent_parameter_names(self) -> tuple[str, ...]:
        return ("scene_summary_net.*", "module_scene_projection.*", "environment_scene_projection.*",
                "module_scene_gate", "environment_scene_gate", "local_reader.*", "field_head.*",
                "direct_reader.*")

    def set_member_routing_mode(self, mode: str) -> None:
        if mode not in ("full", "route"):
            raise ValueError("member-routing mode must be 'full' or 'route'.")
        self.member_routing_mode = mode
        self.architecture = self._arm_name(mode)
        self._set_mode_trainability()

    def _set_mode_trainability(self) -> None:
        for parameter in self.member_router.proposal_M.parameters():
            parameter.requires_grad_(self.member_routing_mode == "route")

    def get_config(self) -> dict[str, Any]:
        base = ContentHypergraphCoreV2.get_config(self)
        base["architecture"] = "A2-D"
        base["schema"] = ContentHypergraphCoreV2.schema
        return dict(schema=self.schema, architecture=self.architecture_name,
                    member_routing_mode=self.member_routing_mode, base_model_config=base,
                    input_dataset=self.input_dataset, input_capability=self.input_capability,
                    entmax_temperature=self.entmax_temperature, query_tile_size=self.query_tile_size)

    @classmethod
    def validate_config(cls, config: Mapping[str, Any]) -> None:
        required = {"schema", "architecture", "member_routing_mode", "base_model_config", "input_dataset",
                    "input_capability", "entmax_temperature", "query_tile_size"}
        if set(config) != required or config.get("schema") != cls.schema:
            raise ValueError("Round13-3 configuration inventory/schema differs.")
        mode = config["member_routing_mode"]
        if mode not in ("full", "route") or config["architecture"] != cls._arm_name(mode):
            raise ValueError("Round13-3 arm name and member_routing_mode disagree.")
        base = dict(config["base_model_config"])
        if base.pop("schema", None) != ContentHypergraphCoreV2.schema:
            raise ValueError("Round13-3 base configuration schema differs.")
        ContentHypergraphCoreV2(**base)
        if config["input_capability"] != cls.input_capability:
            raise ValueError("Round13-3 input capability differs.")
        if not isinstance(config["input_dataset"], str) or not config["input_dataset"]:
            raise ValueError("Round13-3 input dataset identity is required.")
        if type(config["query_tile_size"]) is not int or config["query_tile_size"] < 1:
            raise ValueError("Round13-3 query tile size must be positive.")
        if not isinstance(config["entmax_temperature"], (int, float)) or not 0.05 <= float(config["entmax_temperature"]) <= 20:
            raise ValueError("Round13-3 entmax temperature is outside its valid range.")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]):
        cls.validate_config(config)
        values = dict(config)
        values.pop("schema")
        values.pop("architecture")
        return cls(**values)

    def _config_key(self) -> tuple[Any, ...]:
        config = self.get_config()
        return tuple((key, config[key]) for key in sorted(config))

    def _uses_scene_summary(self) -> bool:
        return False

    def _captures_native_routes(self) -> bool:
        return True

    def _native_capture_route_names(self):
        return _ROUTE_TOPOLOGY

    def _context_value_policy(self):
        return self._AppliedMessagePolicy(self)

    class _AppliedMessagePolicy:
        execution_backend = "segment"

        def __init__(self, core: "InteractionReusedMemberRoutingCore"):
            self.core = core

        def __call__(self, raw_logits, allowed, route_name, **metadata):
            gates = self.core.applied_incidence(raw_logits, allowed, route_name, **metadata)
            removal = self.core._context_message_removal
            if removal is None or metadata.get("stage_identity") != removal.get("stage") or route_name != removal.get("route"):
                return gates
            destination_ids = metadata.get("destination_ids")
            donor_ids = metadata.get("donor_ids")
            if destination_ids is None or donor_ids is None:
                raise ValueError("Applied-message removal requires preserved original IDs.")
            hit = ((destination_ids == int(removal["destination_id"]))[:, :, None]
                   & (donor_ids == int(removal["donor_id"]))[:, None, :])
            if not bool(hit.any()):
                raise ValueError("Requested contextual message IDs are absent from the selected stage.")
            return gates * ~hit

    @contextmanager
    def _base_architecture_scope(self):
        public = self.architecture
        self.architecture = "A2-D"
        try:
            yield
        finally:
            self.architecture = public

    def prepare(self, scene: SceneBatch, **options) -> PreparedOrganizationV2:
        """Prepare adapter-owned prescribed M/E records and applied directed gates."""
        removal = options.pop("context_message_removal", None)
        if removal is not None:
            required = {"stage", "route", "destination_id", "donor_id"}
            if set(removal) != required or removal["route"] not in _ROUTE_TOPOLOGY:
                raise ValueError("context_message_removal needs stage, route, destination_id and donor_id.")
            if not isinstance(removal["stage"], str) or not removal["stage"].endswith("/" + removal["route"]):
                raise ValueError("Context-message stage must identify the requested directed route.")
        previous = self._context_message_removal
        self._context_message_removal = removal
        try:
            with self._base_architecture_scope():
                prepared = super().prepare(scene, **options)
        finally:
            self._context_message_removal = previous
        return prepared

    @staticmethod
    def _safe_snapshot_gate(cycle_trace: Mapping[str, Any], route: str, scene: SceneBatch) -> Tensor:
        snapshot = cycle_trace.get(route + "_native_route_snapshot")
        gate = cycle_trace.get(route + "_applied_incidence")
        if snapshot is None:
            raise ValueError(f"Context stage lacks the actual {route.upper()} route snapshot.")
        expected = (scene.batch_size, snapshot.destination_ids.shape[1], snapshot.donor_ids.shape[1])
        if gate is None:
            # Empty donor inventories use a true zero-width matrix, never an
            # invented edge or a reduction over an empty tensor.
            if all(size > 0 for size in expected[1:]):
                raise ValueError(f"Nonempty applied {route.upper()} stage has no captured gate tensor.")
            gate = scene.module_features.new_zeros(expected)
        if tuple(gate.shape) != expected:
            raise ValueError(f"Applied {route.upper()} incidence has an invalid shape.")
        return gate

    def _finalize_native_preparation(self, organization: PreparedOrganizationV2, cycle_trace):
        cycle_trace = {} if cycle_trace is None else cycle_trace
        scene = organization.scene
        gate_mm = self._safe_snapshot_gate(cycle_trace, "mm", scene)
        gate_me = self._safe_snapshot_gate(cycle_trace, "me", scene)
        present_M = scene.module_mask.to(gate_mm.dtype)
        identity = torch.diag_embed(present_M)
        off_diagonal = ~torch.eye(scene.num_modules, device=gate_mm.device, dtype=torch.bool)[None]
        organization.B_M = gate_mm * off_diagonal.to(gate_mm.dtype) + identity
        organization.B_E = gate_me * scene.environment_mask[:, None, :].to(gate_me.dtype)
        organization.B_M_edge_index = (organization.B_M > 0).nonzero(as_tuple=False)
        organization.B_E_edge_index = (organization.B_E > 0).nonzero(as_tuple=False)
        organization.B_M_edge_values = organization.B_M[
            organization.B_M_edge_index.unbind(-1)] if organization.B_M_edge_index.numel() else organization.B_M.reshape(-1)[:0]
        organization.B_E_edge_values = organization.B_E[
            organization.B_E_edge_index.unbind(-1)] if organization.B_E_edge_index.numel() else organization.B_E.reshape(-1)[:0]

        # Final-read E information measure is dimensionless: positive physical
        # measures divided once by their scene positive mean. M is a discrete
        # present-record count measure, so each present M has nu=1, including
        # a zero donor-measure M whose prescribed identity remains addressable.
        physical_E = scene.environment_measure.to(organization.environment_states.dtype)
        valid_E = scene.environment_mask & (physical_E > 0)
        positive_count = valid_E.sum(-1, keepdim=True).clamp_min(1).to(physical_E.dtype)
        mean_E = (physical_E * valid_E.to(physical_E.dtype)).sum(-1, keepdim=True) / positive_count
        organization.nu_M = scene.module_mask.to(organization.module_states.dtype)
        organization.nu_E = torch.where(valid_E, physical_E / mean_E.clamp_min(torch.finfo(physical_E.dtype).tiny),
                                        torch.zeros_like(physical_E))
        organization.member_keys_M = organization.direct_source_keys
        organization.member_values_M = organization.direct_source_values
        organization.member_keys_E = self.member_router.key_E(organization.environment_states)
        organization.member_values_E = self.member_router.value_E(organization.environment_states)
        organization.member_projection_versions = tuple(_tensor_version(value) for value in (
            organization.member_keys_M, organization.member_values_M,
            organization.member_keys_E, organization.member_values_E))
        organization.member_route_state_versions = tuple(_tensor_version(value) for value in (
            organization.B_M, organization.B_E, organization.nu_M, organization.nu_E,
            organization.B_M_edge_index, organization.B_M_edge_values,
            organization.B_E_edge_index, organization.B_E_edge_values))
        organization.work_counts.update({
            "cached_scene_projection_rows_M": int(scene.batch_size * scene.num_modules),
            "cached_scene_projection_rows_E": int(scene.batch_size * scene.num_environment_tokens),
            "member_key_projection_rows_M": int(scene.batch_size * scene.num_modules),
            "member_value_projection_rows_M": int(scene.batch_size * scene.num_modules),
            "member_key_projection_rows_E": int(scene.batch_size * scene.num_environment_tokens),
            "member_value_projection_rows_E": int(scene.batch_size * scene.num_environment_tokens),
        })
        metadata = tuple(cycle_trace.get("context_stage_metadata", ()))
        for route in _ROUTE_TOPOLOGY:
            rows = [row for row in metadata if row.get("route") == route]
            organization.work_counts[f"context_{route}_score_rows"] = int(sum(
                scene.batch_size * row["destination_ids"].shape[1] * row["donor_ids"].shape[1] for row in rows))
            organization.work_counts[f"context_{route}_positive_value_edges"] = int(sum(
                int(row["positive_value_edges"]) for row in rows))
            organization.work_counts[f"context_{route}_eligible_edges"] = int(sum(
                int(row["eligible_edges"]) for row in rows))
        organization.context_stage_metadata = metadata
        organization.context_stage_traces = tuple(cycle_trace.get("applied_context_stages", ()))
        organization.final_applied_context_stages = {
            route: {"snapshot": cycle_trace.get(route + "_native_route_snapshot"),
                    "incidence": cycle_trace.get(route + "_applied_incidence")}
            for route in _ROUTE_TOPOLOGY
        }
        return organization

    def _validate_prepared_state(self, organization: PreparedOrganizationV2) -> None:
        super()._validate_prepared_state(organization)
        current = tuple(_tensor_version(value) for value in (
            organization.member_keys_M, organization.member_values_M,
            organization.member_keys_E, organization.member_values_E))
        if current != organization.member_projection_versions:
            raise ValueError("Prepared interaction member projections changed; rebuild the scene.")
        route_state = tuple(_tensor_version(value) for value in (
            organization.B_M, organization.B_E, organization.nu_M, organization.nu_E,
            organization.B_M_edge_index, organization.B_M_edge_values,
            organization.B_E_edge_index, organization.B_E_edge_values))
        if route_state != organization.member_route_state_versions:
            raise ValueError("Prepared interaction routing state changed; rebuild the scene.")

    def validate_prepared_state(self, organization: PreparedOrganizationV2, query: QueryBatch) -> None:
        self._validate_prepared_state(organization)
        self._validate_query(organization, query)

    def _query_mask(self, query: QueryBatch) -> Tensor:
        return (torch.ones(query.coordinates.shape[:2], dtype=torch.bool, device=query.coordinates.device)
                if query.query_mask is None else query.query_mask)

    def _query_states(self, organization: PreparedOrganizationV2, query: QueryBatch) -> Tensor:
        # Operating controls enter their prescribed M/E records during scene
        # preparation. The receiver itself carries geometry and optional native
        # query features, not a separate raw-control bypass.
        no_direct_control = organization.operating_state.new_zeros(
            organization.scene.batch_size, self.context_dim)
        return self.query_encoder(query, no_direct_control, organization.frame_origin,
                                  organization.frame_scale)

    def _pair_geometry(self, scene: SceneBatch, query: QueryBatch, source: str,
                       rows: Tensor) -> Tensor:
        if not rows.numel():
            return query.coordinates.new_empty((0, 2 * self.coordinate_dim + 1))
        b, q, s = rows.unbind(-1)
        source_coordinates = scene.module_coordinates if source == "M" else scene.environment_coordinates
        source_lengths = scene.module_characteristic_lengths if source == "M" else scene.environment_characteristic_lengths
        delta = query.coordinates[b, q].to(source_coordinates.dtype) - source_coordinates[b, s]
        normalized = delta / scene.coordinate_scale[b].to(delta.dtype)
        distance = torch.linalg.vector_norm(normalized, dim=-1, keepdim=True)
        local = torch.asinh(delta / source_lengths[b, s].to(delta.dtype))
        return torch.cat((normalized, distance, local), dim=-1)

    def _pair_rows(self, mask: Tensor) -> Tensor:
        return mask.nonzero(as_tuple=False)

    def _member_base_logits(self, prepared: PreparedOrganizationV2, query: QueryBatch,
                            qstates: Tensor, source: str, candidates: Tensor,
                            score_mode: str) -> tuple[Tensor, int]:
        scene = prepared.scene
        keys = prepared.member_keys_M if source == "M" else prepared.member_keys_E
        n = keys.shape[1]
        output = qstates.new_zeros(query.batch_size, query.num_queries, n)
        rows = self._pair_rows(candidates)
        if not rows.numel():
            return output, 0
        geometry = self._pair_geometry(scene, query, source, rows)
        batch, query_index, source_index = rows.unbind(-1)
        if score_mode == "geometry":
            logits = -torch.linalg.vector_norm(geometry[:, :self.coordinate_dim], dim=-1)
        elif score_mode == "learned":
            scorer = self.member_router.member_M if source == "M" else self.member_router.member_E
            features = torch.cat((qstates[batch, query_index], keys[batch, source_index], geometry), dim=-1)
            logits = scorer(features).squeeze(-1)
        else:
            raise ValueError("member_score_mode must be 'learned' or 'geometry'.")
        return output.index_put((batch, query_index, source_index), logits), int(rows.shape[0])

    def _proposal_logits(self, prepared: PreparedOrganizationV2, query: QueryBatch,
                         qstates: Tensor) -> tuple[Tensor, int]:
        scene = prepared.scene
        allowed = scene.module_mask[:, None, :].expand(-1, query.num_queries, -1) & self._query_mask(query)[..., None]
        rows = self._pair_rows(allowed)
        scores = qstates.new_zeros(query.batch_size, query.num_queries, scene.num_modules)
        if not rows.numel():
            return scores, 0
        b, q, m = rows.unbind(-1)
        geometry = self._pair_geometry(scene, query, "M", rows)
        features = torch.cat((qstates[b, q], prepared.member_keys_M[b, m], geometry), dim=-1)
        logits = self.member_router.proposal_M(features).squeeze(-1)
        return scores.index_put((b, q, m), logits), int(rows.shape[0])

    def _validate_dense_control(self, value: Tensor | None, shape: tuple[int, ...], name: str,
                                reference: Tensor) -> Tensor | None:
        if value is None:
            return None
        if tuple(value.shape) != shape or value.device != reference.device:
            raise ValueError(f"{name} must have shape {shape} on the prepared device.")
        valid = torch.isfinite(value).all() & (value >= 0).all()
        if not bool(valid):
            raise ValueError(f"{name} must be finite and nonnegative.")
        return value.to(reference)

    @staticmethod
    def _expand_live_coefficients(pi: Tensor, matrix: Tensor, Q: int, N: int,
                                  *, cached_index: Tensor | None = None,
                                  cached_values: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Coalesce live address paths without unordered CUDA atomic sums.

        Each output member has one fixed-order address reduction. Unordered
        index_add here can move an entmax weight across exact zero on a
        checkpoint replay, changing the packed value-row shape in backward.
        Batched products retain gradients to pi and B and need no Q*M*N bank.
        Compact edge references still supply original-ID directory metadata.
        """
        edge_index = (matrix > 0).nonzero(as_tuple=False) if cached_index is None else cached_index
        edge_values = (matrix[edge_index.unbind(-1)] if cached_values is None and edge_index.numel()
                       else matrix.reshape(-1)[:0] if cached_values is None else cached_values)
        b, address, member = edge_index.unbind(-1) if edge_index.numel() else (
            edge_index.new_empty(0), edge_index.new_empty(0), edge_index.new_empty(0))
        coefficients = torch.bmm(pi, matrix)
        support = torch.bmm((pi > 0).to(pi.dtype), (matrix > 0).to(pi.dtype)) > 0
        if edge_index.numel():
            positive = ((pi[b, :, address] > 0) & (edge_values[:, None] > 0)).to(pi.dtype)
            active_rows = (positive.detach() > 0).sum(dtype=torch.long)
        else:
            active_rows = torch.zeros((), device=pi.device, dtype=torch.long)
        return coefficients, support, edge_index, edge_values, active_rows

    def _typed_read_state(self, prepared: PreparedOrganizationV2, query: QueryBatch,
                          qstates: Tensor, intervention: Mapping[str, Any] | None,
                          *, return_trace: bool) -> tuple[Tensor, dict[str, Tensor], dict[str, int | Tensor], Tensor]:
        scene = prepared.scene
        intervention = {} if intervention is None else dict(intervention)
        allowed_controls = {"member_score_mode", "value_execution", "fixed_B_M", "fixed_B_E", "fixed_pi",
                            "fixed_candidate_mask_M", "fixed_candidate_mask_E", "fixed_member_logits_M",
                            "fixed_member_logits_E"}
        unknown_controls = set(intervention) - allowed_controls
        if unknown_controls:
            raise ValueError(f"Unsupported interaction routing interventions: {sorted(unknown_controls)}.")
        score_mode = intervention.get("member_score_mode", "learned")
        if intervention.get("value_execution", "selected") not in ("selected", "dense_candidate_union"):
            raise ValueError("value_execution must be selected or dense_candidate_union.")
        dense_union = intervention.get("value_execution", "selected") == "dense_candidate_union"
        Q, M, E = query.num_queries, scene.num_modules, scene.num_environment_tokens
        qmask = self._query_mask(query)
        counts: dict[str, int | Tensor] = {}
        trace: dict[str, Tensor] = {}

        B_M = self._validate_dense_control(intervention.get("fixed_B_M"),
            (scene.batch_size, M, M), "fixed_B_M", prepared.B_M)
        B_E = self._validate_dense_control(intervention.get("fixed_B_E"),
            (scene.batch_size, M, E), "fixed_B_E", prepared.B_E)
        if B_M is not None:
            diagonal = B_M.diagonal(dim1=1, dim2=2)
            expected_identity = scene.module_mask.to(diagonal.dtype)
            invalid_support = (B_M > 0) & ~(scene.module_mask[:, :, None] & scene.module_mask[:, None, :])
            if not torch.equal(diagonal, expected_identity) or bool(invalid_support.any()):
                raise ValueError("fixed_B_M must preserve prescribed identity and present-member support.")
        if B_E is not None:
            invalid_support = (B_E > 0) & ~(scene.module_mask[:, :, None] & scene.environment_mask[:, None, :])
            if bool(invalid_support.any()):
                raise ValueError("fixed_B_E cannot create relations to absent M/E records.")
        B_M = prepared.B_M if B_M is None else B_M
        B_E = prepared.B_E if B_E is None else B_E
        if self.member_routing_mode == "full":
            proposal_logits = qstates.new_zeros(scene.batch_size, Q, M)
            proposal_rows = 0
            pi = qstates.new_zeros(scene.batch_size, Q, M)
            proposal_null = qstates.new_ones(scene.batch_size, Q)
        else:
            proposal_logits, proposal_rows = self._proposal_logits(prepared, query, qstates)
            proposal_allowed = scene.module_mask[:, None, :].expand(-1, Q, -1) & qmask[..., None]
            fixed_pi = intervention.get("fixed_pi")
            if fixed_pi is None:
                proposal_distribution = _entmax_with_null(proposal_logits, proposal_allowed, self.entmax_temperature)
                pi = proposal_distribution[..., :M]
                proposal_null = proposal_distribution[..., M]
            else:
                if tuple(fixed_pi.shape) != (scene.batch_size, Q, M) or fixed_pi.device != qstates.device:
                    raise ValueError("fixed_pi must be [B,Q,M] on the query device.")
                if not bool((torch.isfinite(fixed_pi).all() & (fixed_pi >= 0).all()
                             & (fixed_pi.sum(-1) <= 1 + 1.e-5).all())):
                    raise ValueError("fixed_pi must be nonnegative and sum to at most one.")
                if bool((fixed_pi.masked_select(~scene.module_mask[:, None, :].expand_as(fixed_pi)) > 0).any()):
                    raise ValueError("fixed_pi cannot propose absent original modules.")
                if bool((fixed_pi.masked_select(~qmask[..., None].expand_as(fixed_pi)) > 0).any()):
                    raise ValueError("fixed_pi cannot propose from masked query rows.")
                pi = fixed_pi.to(qstates)
                proposal_null = (1. - pi.sum(-1)).clamp_min(0.)
        if B_M is prepared.B_M:
            edge_M = (prepared.B_M_edge_index, prepared.B_M_edge_values)
        else:
            edge_M = (None, None)
        if B_E is prepared.B_E:
            edge_E = (prepared.B_E_edge_index, prepared.B_E_edge_values)
        else:
            edge_E = (None, None)
        a_M, inferred_M, edge_index_M, edge_values_M, expansion_rows_M = self._expand_live_coefficients(
            pi, B_M, Q, M, cached_index=edge_M[0], cached_values=edge_M[1])
        a_E, inferred_E, edge_index_E, edge_values_E, expansion_rows_E = self._expand_live_coefficients(
            pi, B_E, Q, E, cached_index=edge_E[0], cached_values=edge_E[1])
        if self.member_routing_mode == "full":
            candidate_M = scene.module_mask[:, None, :] & qmask[..., None]
            candidate_E = (scene.environment_mask & (scene.environment_measure > 0))[:, None, :] & qmask[..., None]
        else:
            candidate_M = inferred_M & scene.module_mask[:, None, :] & qmask[..., None]
            candidate_E = inferred_E & (scene.environment_mask & (scene.environment_measure > 0))[:, None, :] & qmask[..., None]
        fixed_candidate_M = intervention.get("fixed_candidate_mask_M")
        fixed_candidate_E = intervention.get("fixed_candidate_mask_E")
        if fixed_candidate_M is not None:
            if self.member_routing_mode != "route":
                raise ValueError("Fixed relation candidates are available only for the routed arm.")
            if tuple(fixed_candidate_M.shape) != tuple(candidate_M.shape) or fixed_candidate_M.dtype != torch.bool:
                raise ValueError("fixed_candidate_mask_M must be a boolean [B,Q,M] tensor.")
            if bool((fixed_candidate_M & ~inferred_M).any()):
                raise ValueError("Fixed M candidates must remain reachable through a positive B path.")
            candidate_M = fixed_candidate_M.to(device=qstates.device) & scene.module_mask[:, None, :] & qmask[..., None]
        if fixed_candidate_E is not None:
            if self.member_routing_mode != "route":
                raise ValueError("Fixed relation candidates are available only for the routed arm.")
            if tuple(fixed_candidate_E.shape) != tuple(candidate_E.shape) or fixed_candidate_E.dtype != torch.bool:
                raise ValueError("fixed_candidate_mask_E must be a boolean [B,Q,E] tensor.")
            if bool((fixed_candidate_E & ~inferred_E).any()):
                raise ValueError("Fixed E candidates must remain reachable through a positive B path.")
            candidate_E = fixed_candidate_E.to(device=qstates.device) & (scene.environment_mask & (scene.environment_measure > 0))[:, None, :] & qmask[..., None]

        logits_M, rows_M = self._member_base_logits(prepared, query, qstates, "M", candidate_M, score_mode)
        logits_E, rows_E = self._member_base_logits(prepared, query, qstates, "E", candidate_E, score_mode)
        for name, value, shape in (("fixed_member_logits_M", logits_M, (scene.batch_size, Q, M)),
                                   ("fixed_member_logits_E", logits_E, (scene.batch_size, Q, E))):
            fixed = intervention.get(name)
            if fixed is not None:
                if tuple(fixed.shape) != shape or fixed.device != qstates.device or not bool(torch.isfinite(fixed).all()):
                    raise ValueError(f"{name} must be a finite tensor with shape {shape} on the query device.")
                if name.endswith("_M"):
                    logits_M = fixed.to(qstates)
                else:
                    logits_E = fixed.to(qstates)

        # Address coefficients contribute a logarithmic information prior.
        # nu_M is the discrete present-record count measure; nu_E is one
        # dimensionless mean-normalized physical measure, applied exactly once.
        prior_M = torch.where(candidate_M, a_M.clamp_min(torch.finfo(qstates.dtype).tiny).log(),
                              torch.zeros_like(a_M)) + prepared.nu_M[:, None, :].clamp_min(1.).log()
        prior_E = torch.where(candidate_E, a_E.clamp_min(torch.finfo(qstates.dtype).tiny).log(),
                              torch.zeros_like(a_E)) + prepared.nu_E[:, None, :].clamp_min(torch.finfo(qstates.dtype).tiny).log()
        effective_M = logits_M + prior_M
        effective_E = logits_E + prior_E
        if self.member_routing_mode == "full":
            allowed_M = scene.module_mask[:, None, :] & qmask[..., None]
            allowed_E = (scene.environment_mask & (scene.environment_measure > 0))[:, None, :] & qmask[..., None]
            beta_M = _positive_softmax(logits_M + prepared.nu_M[:, None, :].clamp_min(1.).log(), allowed_M)
            beta_E = _positive_softmax(logits_E + prepared.nu_E[:, None, :].clamp_min(torch.finfo(qstates.dtype).tiny).log(), allowed_E)
            null_M = qstates.new_zeros(scene.batch_size, Q)
            null_E = qstates.new_zeros(scene.batch_size, Q)
            candidate_M, candidate_E = allowed_M, allowed_E
        else:
            beta_null_M = _entmax_with_null(effective_M, candidate_M, self.entmax_temperature)
            beta_null_E = _entmax_with_null(effective_E, candidate_E, self.entmax_temperature)
            beta_M, null_M = beta_null_M[..., :M], beta_null_M[..., M]
            beta_E, null_E = beta_null_E[..., :E], beta_null_E[..., E]

        execute_M = candidate_M if dense_union else (beta_M > 0)
        execute_E = candidate_E if dense_union else (beta_E > 0)
        c_M = self._reduce_selected(prepared.member_values_M, beta_M, execute_M)
        c_E = self._reduce_selected(prepared.member_values_E, beta_E, execute_E)
        joint = self.member_router.joint_trunk(torch.cat((qstates, c_M, c_E), dim=-1))
        joint_fields = self.member_router.joint_head(joint)
        joint_fields = joint_fields * qmask[..., None].to(joint_fields.dtype)
        candidate_M_count = candidate_M.detach().sum(dtype=torch.long)
        candidate_E_count = candidate_E.detach().sum(dtype=torch.long)
        selected_M_count = (beta_M.detach() > 0).sum(dtype=torch.long)
        selected_E_count = (beta_E.detach() > 0).sum(dtype=torch.long)
        execution_M_count = execute_M.detach().sum(dtype=torch.long)
        execution_E_count = execute_E.detach().sum(dtype=torch.long)
        counts.update(
            query_M_proposal_score_rows=proposal_rows,
            query_M_cheap_score_rows=rows_M,
            query_E_cheap_score_rows=rows_E,
            query_M_selected_value_rows=selected_M_count,
            query_E_selected_value_rows=selected_E_count,
            query_M_value_execution_rows=execution_M_count,
            query_E_value_execution_rows=execution_E_count,
            query_M_dense_candidate_value_rows=candidate_M_count if dense_union else 0,
            query_E_dense_candidate_value_rows=candidate_E_count if dense_union else 0,
            query_M_candidate_rows=candidate_M_count,
            query_E_candidate_rows=candidate_E_count,
            query_address_coefficient_pair_rows=int(scene.batch_size * Q * M * (M + E)),
            query_address_relation_rows=expansion_rows_M + expansion_rows_E,
            query_receiver_rows=int(scene.batch_size * Q),
        )
        if return_trace:
            trace = dict(
                proposal_logits_M=proposal_logits, proposal_weights_M=pi, proposal_null=proposal_null,
                candidate_mask_M=candidate_M, candidate_mask_E=candidate_E,
                member_prior_M=prior_M, member_prior_E=prior_E,
                member_logits_M=logits_M, member_logits_E=logits_E,
                member_effective_scores_M=effective_M, member_effective_scores_E=effective_E,
                member_weights_M=beta_M, member_weights_E=beta_E,
                member_null_M=null_M, member_null_E=null_E,
                original_module_ids=scene.module_ids, original_environment_ids=scene.environment_ids,
                B_M_edge_index=edge_index_M, B_M_edge_values=edge_values_M,
                B_E_edge_index=edge_index_E, B_E_edge_values=edge_values_E,
                B_M=B_M, B_E=B_E,
                address_coefficients_M=a_M, address_coefficients_E=a_E,
                selected_value_indices_M=(beta_M > 0).nonzero(as_tuple=False),
                selected_value_indices_E=(beta_E > 0).nonzero(as_tuple=False),
            )
        return torch.cat((qstates, c_M, c_E), dim=-1), trace, counts, joint_fields

    @staticmethod
    def _reduce_selected(values: Tensor, weights: Tensor, execute_mask: Tensor) -> Tensor:
        batch, queries, sources = weights.shape
        width = values.shape[-1]
        rows = execute_mask.nonzero(as_tuple=False)
        reduced = values.new_zeros(batch * queries, width)
        if rows.numel():
            b, q, source = rows.unbind(-1)
            selected_values = values[b, source]
            contribution = selected_values * weights[b, q, source, None]
            reduced = reduced.index_add(0, b * queries + q, contribution)
        else:
            reduced = reduced + values.reshape(-1, width)[:0].sum()
        return reduced.reshape(batch, queries, width)

    def decode(self, organization: PreparedOrganizationV2, query: QueryBatch,
               controls: Tensor | None = None, *, intervention: Mapping[str, Any] | None = None,
               return_trace: bool = False, trace_request=None, **kwargs) -> InteractionJointOutput:
        if controls is not None:
            raise ValueError("Round13-3 emits one joint four-field tensor; no affine controls are accepted.")
        if kwargs:
            unknown = sorted(set(kwargs) - {"return_basis", "accumulation_dtype", "edge_gates", "uniform_access", "local_gates"})
            if unknown:
                raise TypeError(f"Unsupported Round13-3 decode options: {unknown}.")
            if any(kwargs.get(name) is not None and name != "uniform_access" for name in kwargs):
                raise ValueError("Historical affine, edge and local gates are not Round13-3 controls.")
        self.validate_prepared_state(organization, query)
        if trace_request is not None:
            return_trace = True
        if return_trace and query.num_queries > self.query_tile_size:
            raise ValueError("Interaction routing traces require an explicit query slice no larger than query_tile_size.")
        fields_by_tile = []
        trace = None
        # Preparation belongs to PreparedScene and is reported separately by
        # the native task. Cached decode must not charge those rows again.
        counts: dict[str, int] = {}
        packed_count_total = None
        count_keys = ("query_M_proposal_score_rows", "query_M_cheap_score_rows", "query_E_cheap_score_rows",
                      "query_M_selected_value_rows", "query_E_selected_value_rows", "query_M_value_execution_rows",
                      "query_E_value_execution_rows", "query_M_dense_candidate_value_rows",
                      "query_E_dense_candidate_value_rows", "query_M_candidate_rows", "query_E_candidate_rows",
                      "query_address_relation_rows", "query_address_coefficient_pair_rows", "query_receiver_rows")
        count_keys += getattr(self, "_additional_decode_count_keys", ())
        starts = range(0, max(query.num_queries, 1), self.query_tile_size)
        for start in starts:
            stop = min(start + self.query_tile_size, query.num_queries)
            if query.num_queries == 0:
                break
            tile = query if query.num_queries <= self.query_tile_size else slice_query_v1(query, start, stop)
            tile_intervention = self._slice_intervention(intervention, query.num_queries, start, stop)
            if self.training and torch.is_grad_enabled() and not return_trace:
                def run_tile(coordinates, tile=tile, tile_intervention=tile_intervention):
                    local_query = replace(tile, coordinates=coordinates)
                    local_qstates = self._query_states(organization, local_query)
                    _, _, local_counts, local_fields = self._typed_read_state(
                        organization, local_query, local_qstates, tile_intervention, return_trace=False)
                    count_values = tuple(local_counts.get(name, 0) for name in count_keys)
                    return local_fields, count_values
                local_fields, count_values = checkpoint(run_tile, tile.coordinates, use_reentrant=False)
                local_counts = dict(zip(count_keys, count_values))
            else:
                local_qstates = self._query_states(organization, tile)
                _, local_trace, local_counts, local_fields = self._typed_read_state(
                    organization, tile, local_qstates, tile_intervention, return_trace=return_trace)
                if return_trace:
                    trace = local_trace
            packed_counts = _pack_work_counts(local_counts, count_keys, local_fields)
            fields_by_tile.append(local_fields)
            packed_count_total = (packed_counts.detach() if packed_count_total is None else
                                  packed_count_total + packed_counts.detach())
        fields = torch.cat(fields_by_tile, dim=1) if fields_by_tile else next(self.parameters()).new_zeros(
            query.batch_size, 0, self.output_dim)
        if packed_count_total is not None:
            counts = _unpack_work_counts(packed_count_total, count_keys)
        if return_trace:
            trace["joint_fields"] = fields
        return InteractionJointOutput(fields, trace if return_trace else None, counts)

    @staticmethod
    def _slice_intervention(intervention: Mapping[str, Any] | None, full_queries: int,
                            start: int, stop: int) -> dict[str, Any] | None:
        if intervention is None:
            return None
        sliced = dict(intervention)
        query_keys = ("fixed_pi", "fixed_candidate_mask_M", "fixed_candidate_mask_E",
                      "fixed_member_logits_M", "fixed_member_logits_E")
        for name in query_keys:
            value = sliced.get(name)
            if value is None:
                continue
            if not isinstance(value, Tensor) or value.ndim != 3:
                raise ValueError(f"{name} must be a rank-three [B,Q,N] tensor.")
            if value.shape[1] == full_queries:
                sliced[name] = value[:, start:stop]
            elif full_queries > stop - start:
                raise ValueError(f"{name} must cover all {full_queries} native queries before tiled decoding.")
        return sliced

    def capture_interaction_routing_trace(self, prepared: PreparedOrganizationV2, query: QueryBatch,
                                          intervention: Mapping[str, Any] | None = None) -> dict[str, Tensor]:
        output = self.decode(prepared, query, intervention=intervention, return_trace=True)
        return output.trace or {}

    def dense_candidate_union(self, prepared: PreparedOrganizationV2, query: QueryBatch,
                              intervention: Mapping[str, Any] | None = None) -> InteractionJointOutput:
        controls = {} if intervention is None else dict(intervention)
        controls["value_execution"] = "dense_candidate_union"
        return self.decode(prepared, query, intervention=controls, return_trace=False)

    def _geometry_B_control(self, prepared: PreparedOrganizationV2) -> tuple[Tensor, Tensor]:
        """Reweight each applied support row by geometry, preserving its count."""
        scene = prepared.scene
        B_M = torch.zeros_like(prepared.B_M)
        B_E = torch.zeros_like(prepared.B_E)
        for target, applied, source_coords, source_lengths in (
            (B_M, prepared.B_M, scene.module_coordinates, scene.module_characteristic_lengths),
            (B_E, prepared.B_E, scene.environment_coordinates, scene.environment_characteristic_lengths),
        ):
            edges = applied > 0
            for batch in range(scene.batch_size):
                for address in range(scene.num_modules):
                    members = edges[batch, address].nonzero(as_tuple=False).flatten()
                    if not members.numel():
                        continue
                    delta = source_coords[batch, members] - scene.module_coordinates[batch, address]
                    scale = torch.sqrt(source_lengths[batch, members] * scene.module_characteristic_lengths[batch, address])
                    distance = torch.linalg.vector_norm(delta / scale.clamp_min(torch.finfo(scale.dtype).tiny), dim=-1)
                    weights = torch.softmax(-distance, dim=0).clamp_min(torch.finfo(distance.dtype).tiny)
                    weights = weights / weights.sum()
                    target[batch, address, members] = weights
        # Geometry B control cannot rewrite prescribed M identity coefficients.
        if scene.num_modules:
            identity = torch.diag_embed(scene.module_mask.to(B_M.dtype))
            B_M = torch.where(torch.eye(scene.num_modules, device=B_M.device, dtype=torch.bool)[None], identity, B_M)
        return B_M, B_E

    @staticmethod
    def _scaled_B_control(prepared: PreparedOrganizationV2, scale: float = 0.7) -> tuple[Tensor, Tensor]:
        """Change applied coefficients while preserving each typed relation support."""
        if not 0 < scale < 1:
            raise ValueError("B coefficient scale must be between zero and one.")
        scene = prepared.scene
        B_M = prepared.B_M * scale
        if scene.num_modules:
            identity = torch.diag_embed(scene.module_mask.to(B_M.dtype))
            diagonal = torch.eye(scene.num_modules, device=B_M.device, dtype=torch.bool)[None]
            B_M = torch.where(diagonal, identity, B_M)
        B_E = prepared.B_E * scale
        return B_M, B_E

    def member_routing_intervention(self, probe_name: str, prepared: PreparedOrganizationV2,
                                    query: QueryBatch, query_indices=None):
        """Build one fixed-checkpoint probe and any required preparation replay."""
        self.validate_prepared_state(prepared, query)
        if query_indices is not None:
            indices = torch.as_tensor(query_indices, device=query.coordinates.device, dtype=torch.long)
            if indices.ndim != 1 or not indices.numel() or bool((indices < 0).any()) or bool((indices >= query.num_queries).any()):
                raise ValueError("query_indices must be a nonempty in-range vector.")
        if probe_name == "b_fixed_candidates_scores":
            scaled_M, scaled_E = self._scaled_B_control(prepared)
            # This control changes only positive B coefficients: support stays
            # fixed, pi is recomputed by the unchanged proposal scorer, and
            # member logits use the same learned scorer over the same IDs.
            return ({"fixed_B_M": scaled_M, "fixed_B_E": scaled_E}, None)
        if probe_name == "geometry_b_comparable_counts":
            geometry_M, geometry_E = self._geometry_B_control(prepared)
            return ({"fixed_B_M": geometry_M, "fixed_B_E": geometry_E}, None)
        if probe_name == "geometry_member_scores_fixed_b_pi":
            return ({"fixed_B_M": prepared.B_M, "fixed_B_E": prepared.B_E,
                     "member_score_mode": "geometry"}, None)
        if probe_name == "all_candidates_open":
            return ({"value_execution": "dense_candidate_union"}, None)
        if probe_name == "remove_contextual_message_recompute":
            context_stages = getattr(prepared, "context_stage_traces", ())
            if not context_stages:
                prepared = self.prepare(prepared.scene, trace_request={"include": []})
                context_stages = getattr(prepared, "context_stage_traces", ())
            first_cycle_me = next((stage for stage in context_stages
                if getattr(stage.get("me_native_route_snapshot"), "stage_identity", "") == "cycle0/me"), None)
            if first_cycle_me is None:
                raise ValueError("A captured first-cycle ME stage is required for a dependent-stage removal probe.")
            snapshot = first_cycle_me["me_native_route_snapshot"]
            gates = first_cycle_me.get("me_applied_incidence")
            if gates is None or not bool((gates > 0).any()):
                raise ValueError("The first-cycle applied ME stage has no positive message to remove.")
            edge = (gates > 0).nonzero(as_tuple=False)[0]
            batch, destination, donor = (int(v) for v in edge)
            return ({}, {"context_message_removal": {"stage": snapshot.stage_identity, "route": "me",
                    "destination_id": int(snapshot.destination_ids[batch, destination]),
                    "donor_id": int(snapshot.donor_ids[batch, donor])}})
        raise ValueError(f"Unknown interaction routing probe: {probe_name}.")

    # No joint-budget hooks are exposed. The generic training engine therefore
    # never constructs an inherited structural-loss macro for this core.
    joint_begin_update = None
    joint_begin_microbatch = None
    joint_microbatch_result = None
    joint_constraint_duals = None
    joint_after_update = None

    def set_organization_clock(self, epoch: int, update_index: int = 0) -> None:
        if type(epoch) is not int or type(update_index) is not int or epoch < 0 or update_index < 0:
            raise ValueError("Organization clock values must be nonnegative integers.")
        self._joint_epoch = epoch
        self._joint_update_index = update_index

    def set_training_clock(self, epoch: int, update_index: int = 0) -> None:
        self.set_organization_clock(epoch, update_index)

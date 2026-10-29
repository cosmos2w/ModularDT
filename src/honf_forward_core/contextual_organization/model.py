"""Joint physical-field core with contextual typed forests and nullable reads."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Hashable, Mapping

import torch
from torch import Tensor, nn

from ..content_hypergraph.contracts import CoreOutput, QueryBatch, SceneBatch
from ..content_hypergraph.spatial_features_v2 import coordinate_gamma
from ..content_hypergraph.staged_attention_v2 import StagedContextCycleV2
from .forest import ContextualFrontierPolicy, SceneForest, build_forest
from .routing import (
    null_normalized_group_access,
    null_normalized_weights,
    positive_gate,
    quintic_window,
    smooth_union,
)


_SCHEMA = "contextual_organization_joint_v1"


@dataclass
class ContextualJointOutput(CoreOutput):
    """Compatibility view over one generic jointly predicted field vector."""

    joint_fields: Tensor | None = None


@dataclass(frozen=True)
class PreparedRecordBank:
    """One scene's selected typed frontier and its direct original-record C."""

    bank: str
    forest: SceneForest | None
    frontier: Any
    scene_rows: Tensor
    context_states: Tensor
    keys: Tensor
    coordinates: Tensor
    lengths: Tensor
    measure: Tensor
    information_measure: Tensor
    member_radii: Tensor
    group_states: Tensor
    group_centers: Tensor
    group_extents: Tensor
    group_outer_radii: Tensor
    group_mass: Tensor
    group_ancestry: tuple[tuple[int, ...], ...]
    membership: Tensor
    donor_local_indices: tuple[tuple[int, ...], ...]

    @property
    def present_count(self) -> int:
        return int(self.scene_rows.numel())

    @property
    def live_count(self) -> int:
        return int(self.membership.shape[0])


@dataclass(frozen=True)
class PreparedContextualOrganization:
    """Query-independent preparation for a complete physical input scene."""

    scene: SceneBatch
    module_states: Tensor
    environment_states: Tensor
    module_keys: Tensor
    environment_keys: Tensor
    module_banks: tuple[PreparedRecordBank, ...]
    environment_banks: tuple[PreparedRecordBank, ...]
    model_owner: int
    model_parameter_versions: tuple[tuple[int, int], ...]
    model_buffer_versions: tuple[tuple[int, int], ...]
    scene_tensor_versions: tuple[tuple[int, int, int], ...]
    scene_identity_snapshot: tuple[Any, Any, Any]
    model_config_key: tuple[Any, ...]
    frame_origin: Tensor
    frame_scale: Tensor
    work_counts: Mapping[str, int]

    @property
    def scene_key(self) -> Any:
        return self.scene.scene_key

    @property
    def module_frontiers(self) -> tuple[Any, ...]:
        return tuple(bank.frontier for bank in self.module_banks)

    @property
    def environment_frontiers(self) -> tuple[Any, ...]:
        return tuple(bank.frontier for bank in self.environment_banks)


def _stamp_tensor(value: Tensor) -> tuple[int, int, int]:
    try:
        version = int(value._version)
    except RuntimeError:
        version = -1
    return id(value), int(value.data_ptr()), version


def _scene_tensors(scene: SceneBatch) -> tuple[Tensor, ...]:
    values = (
        scene.module_features, scene.environment_features, scene.module_coordinates,
        scene.environment_coordinates, scene.module_mask, scene.environment_mask,
        scene.module_measure, scene.environment_measure, scene.module_ids,
        scene.environment_ids, scene.local_radius, scene.operating_features,
        scene.coordinate_origin, scene.coordinate_scale,
        scene.module_measure_reference, scene.environment_measure_reference,
        scene.module_characteristic_lengths, scene.environment_characteristic_lengths,
    )
    return tuple(value for value in values if value is not None)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, *, final_bias: float = 0.0) -> nn.Sequential:
    result = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )
    nn.init.constant_(result[-1].bias, final_bias)
    return result


class ContextualOrganizationJointCore(nn.Module):
    """Shared Transformer context, typed adaptive groups, and one joint field.

    The core owns no case channel names or physical units. A case adapter puts
    prescribed per-record controls in the corresponding record features and
    maps the returned joint vector to its primitive channels.
    """

    schema = _SCHEMA

    def __init__(
        self,
        module_input_dim: int,
        environment_input_dim: int,
        coordinate_dim: int,
        query_input_dim: int = 0,
        operating_input_dim: int = 0,
        output_dim: int = 4,
        architecture: str = "J-H",
        context_dim: int = 128,
        num_heads: int = 4,
        context_depth: int = 2,
        local_hidden_dim: int = 128,
        dropout: float = 0.0,
        init_seed: int = 0,
        forest_neighbors: int = 4,
        query_tile_size: int = 512,
    ) -> None:
        super().__init__()
        architecture = str(architecture).upper()
        if architecture not in {"J-D", "J-H"}:
            raise ValueError("architecture must be 'J-D' or 'J-H'.")
        if min(module_input_dim, environment_input_dim, coordinate_dim, output_dim,
               context_dim, num_heads, context_depth, local_hidden_dim) < 1:
            raise ValueError("Input/output widths, context depth and head count must be positive.")
        if context_dim % num_heads:
            raise ValueError("context_dim must be divisible by num_heads.")
        if local_hidden_dim != context_dim:
            raise ValueError("The typed readers and joint trunk use context_dim as their shared width.")
        if query_input_dim < 0 or operating_input_dim < 0:
            raise ValueError("Optional input widths cannot be negative.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0,1).")
        if isinstance(forest_neighbors, bool) or int(forest_neighbors) != forest_neighbors or forest_neighbors < 0:
            raise ValueError("forest_neighbors must be a nonnegative integer.")
        if isinstance(query_tile_size, bool) or int(query_tile_size) != query_tile_size or query_tile_size < 1:
            raise ValueError("query_tile_size must be a positive integer.")

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
        self.forest_neighbors = int(forest_neighbors)
        self.query_tile_size = int(query_tile_size)
        self._organization_epoch = 1
        self._organization_update = 0

        gamma_dim = 7 * self.coordinate_dim
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.init_seed)
            self.module_encoder = _mlp(self.module_input_dim + gamma_dim + self.coordinate_dim,
                                       self.context_dim, self.context_dim)
            self.environment_encoder = _mlp(self.environment_input_dim + gamma_dim + self.coordinate_dim,
                                            self.context_dim, self.context_dim)
            self.operating_encoder = (
                _mlp(self.operating_input_dim, self.context_dim, self.context_dim)
                if self.operating_input_dim else None
            )
            self.context_cycles = nn.ModuleList([
                StagedContextCycleV2(self.context_dim, self.num_heads, self.coordinate_dim,
                                     self.dropout_probability)
                for _ in range(self.context_depth)
            ])
            self.module_key = nn.Linear(self.context_dim, self.context_dim)
            self.environment_key = nn.Linear(self.context_dim, self.context_dim)
            self.module_value = nn.Linear(self.context_dim, self.context_dim)
            self.environment_value = nn.Linear(self.context_dim, self.context_dim)
            self.module_query_key = nn.Linear(self.context_dim, self.context_dim)
            self.environment_query_key = nn.Linear(self.context_dim, self.context_dim)
            self.module_group_membership = _mlp(2 * self.context_dim + 2 * self.coordinate_dim + 2,
                                                 self.context_dim, 1, final_bias=1.5)
            self.environment_group_membership = _mlp(2 * self.context_dim + 2 * self.coordinate_dim + 2,
                                                      self.context_dim, 1, final_bias=1.5)
            self.module_group_router = _mlp(2 * self.context_dim + 2 * self.coordinate_dim + 1,
                                            self.context_dim, 1, final_bias=1.5)
            self.environment_group_router = _mlp(2 * self.context_dim + 2 * self.coordinate_dim + 1,
                                                 self.context_dim, 1, final_bias=1.5)
            source_geo_dim = 2 * self.coordinate_dim + 1
            self.module_source_geometry = _mlp(source_geo_dim, self.context_dim, self.num_heads)
            self.environment_source_geometry = _mlp(source_geo_dim, self.context_dim, self.num_heads)
            self.environment_conditioning = nn.Linear(self.context_dim, self.context_dim, bias=False)
            self.query_encoder = _mlp(self.query_input_dim + gamma_dim, self.context_dim, self.context_dim)
            self.query_norm = nn.LayerNorm(self.context_dim)
            self.fusion_trunk = nn.Sequential(
                nn.Linear(3 * self.context_dim, self.context_dim),
                nn.LayerNorm(self.context_dim),
                nn.SiLU(),
                nn.Dropout(self.dropout_probability),
                nn.Linear(self.context_dim, self.context_dim),
                nn.SiLU(),
            )
            self.field_head = nn.Linear(self.context_dim, self.output_dim)
            self.frontier_policies = nn.ModuleDict({
                "module": ContextualFrontierPolicy(
                    self.context_dim, self.coordinate_dim, run_seed=self.init_seed, warmup_epochs=20
                ),
                "environment": ContextualFrontierPolicy(
                    self.context_dim, self.coordinate_dim, run_seed=self.init_seed, warmup_epochs=20
                ),
            })

        self.register_buffer("organization_epoch", torch.tensor(1, dtype=torch.long))
        self.register_buffer("organization_update", torch.tensor(0, dtype=torch.long))

    @classmethod
    def validate_config(cls, config: Mapping[str, Any]) -> dict[str, Any]:
        values = dict(config)
        if values.pop("schema", None) != cls.schema:
            raise ValueError("Contextual joint core configuration schema differs.")
        cls(**values)
        return dict(config)

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
            "forest_neighbors": self.forest_neighbors,
            "query_tile_size": self.query_tile_size,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ContextualOrganizationJointCore":
        values = dict(config)
        if values.pop("schema", None) != cls.schema:
            raise ValueError("Contextual joint core configuration schema differs.")
        return cls(**values)

    def _config_key(self) -> tuple[Any, ...]:
        return tuple((key, value) for key, value in sorted(self.get_config().items())) + (
            ("organization_epoch", int(self.organization_epoch)),
            ("organization_update", int(self.organization_update)),
        )

    def _parameter_version_stamp(self) -> tuple[tuple[int, int], ...]:
        return tuple((id(parameter), int(parameter._version)) for parameter in self.parameters())

    def _buffer_version_stamp(self) -> tuple[tuple[int, int], ...]:
        output = []
        for value in self.buffers():
            try:
                version = int(value._version)
            except RuntimeError:
                version = -1
            output.append((id(value), version))
        return tuple(output)

    def set_organization_clock(self, epoch: int, update: int = 0) -> None:
        """Set the ordinary run epoch/update used by keyed frontier sampling."""
        if isinstance(epoch, bool) or int(epoch) != epoch or epoch < 1:
            raise ValueError("organization epoch must be a positive integer.")
        if isinstance(update, bool) or int(update) != update or update < 0:
            raise ValueError("organization update must be a nonnegative integer.")
        self.organization_epoch.fill_(int(epoch))
        self.organization_update.fill_(int(update))
        self._organization_epoch = int(epoch)
        self._organization_update = int(update)

    def _check_scene(self, scene: SceneBatch) -> None:
        scene.validate(
            coordinate_dim=self.coordinate_dim,
            module_input_dim=self.module_input_dim,
            environment_input_dim=self.environment_input_dim,
            operating_input_dim=self.operating_input_dim,
        )
        if scene.coordinate_origin is None or scene.coordinate_scale is None:
            raise ValueError("ContextualOrganizationJointCore requires coordinate_origin and coordinate_scale.")
        if scene.module_characteristic_lengths is None or scene.environment_characteristic_lengths is None:
            raise ValueError("The contextual joint core requires characteristic lengths for both typed banks.")

    def _scene_states(self, scene: SceneBatch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        dtype = next(self.parameters()).dtype
        origin = scene.coordinate_origin.to(dtype=dtype)
        scale = scene.coordinate_scale.to(dtype=dtype).clamp_min(torch.finfo(dtype).eps)
        module_coordinates = scene.module_coordinates.to(dtype=dtype)
        environment_coordinates = scene.environment_coordinates.to(dtype=dtype)
        module_lengths = scene.module_characteristic_lengths.to(dtype=dtype)
        environment_lengths = scene.environment_characteristic_lengths.to(dtype=dtype)
        module_gamma = coordinate_gamma(module_coordinates, origin, scale)
        environment_gamma = coordinate_gamma(environment_coordinates, origin, scale)
        module_inputs = torch.cat((scene.module_features.to(dtype=dtype), module_gamma,
                                   module_lengths / scale[:, None, :]), dim=-1)
        environment_inputs = torch.cat((scene.environment_features.to(dtype=dtype), environment_gamma,
                                        environment_lengths / scale[:, None, :]), dim=-1)
        module_states = self.module_encoder(module_inputs)
        environment_states = self.environment_encoder(environment_inputs)
        if self.operating_encoder is not None:
            operating = self.operating_encoder(scene.operating_features.to(dtype=dtype))
            module_states = module_states + operating[:, None, :]
            environment_states = environment_states + operating[:, None, :]
        module_states = module_states * scene.module_mask[:, :, None].to(dtype)
        environment_states = environment_states * scene.environment_mask[:, :, None].to(dtype)
        for cycle_index, cycle in enumerate(self.context_cycles):
            module_states, environment_states, _ = cycle(
                module_states,
                environment_states,
                scene,
                scale,
                cycle_index=cycle_index,
            )
        return module_states, environment_states, origin, scale

    @staticmethod
    def _project_present(values: Tensor, present: Tensor, projection: nn.Module) -> Tensor:
        """Project only real records; padded rows do not incur neural work."""
        output = values.new_zeros(values.shape)
        if bool(present.any()):
            output[present] = projection(values[present])
        return output

    @staticmethod
    def _scene_case_key(scene_key: Any, batch_index: int, batch_size: int) -> Hashable:
        if batch_size > 1 and isinstance(scene_key, (tuple, list)) and len(scene_key) == batch_size:
            value = scene_key[batch_index]
        else:
            value = scene_key
        if value is None:
            return ("batch_scene", int(batch_index))
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError("scene_key values used for frontier sampling must be hashable.") from exc
        return value

    def _prepare_bank(
        self,
        *,
        bank: str,
        states: Tensor,
        keys: Tensor,
        coordinates: Tensor,
        lengths: Tensor,
        measures: Tensor,
        ids: Tensor,
        present: Tensor,
        scene_rows_all: Tensor,
        information_reference: Tensor,
        local_radius: Tensor | None,
        case_key: Hashable,
        frame_scale: Tensor,
        forest_override: SceneForest | None = None,
        forced_subtree: tuple[int, str] | None = None,
    ) -> tuple[PreparedRecordBank, dict[str, int]]:
        compact_rows = torch.nonzero(present, as_tuple=False).flatten()
        scene_rows = scene_rows_all.index_select(0, compact_rows)
        compact_states = states.index_select(0, scene_rows)
        compact_keys = keys.index_select(0, scene_rows)
        compact_coordinates = coordinates.index_select(0, scene_rows)
        compact_lengths = lengths.index_select(0, scene_rows)
        compact_measure = measures.index_select(0, scene_rows)
        compact_ids_tensor = ids.index_select(0, scene_rows)
        if bank == "module":
            information_measure = torch.ones_like(compact_measure)
            forest_measure = information_measure
            member_radii = (local_radius.index_select(0, scene_rows).clamp_min(0.0)
                            + 0.5 * torch.linalg.vector_norm(compact_lengths, dim=-1))
            mass_scale = information_reference.new_ones(())
        else:
            mass_scale = information_reference.clamp_min(torch.finfo(compact_measure.dtype).tiny)
            information_measure = compact_measure / mass_scale
            forest_measure = information_measure
            member_radii = 0.5 * torch.linalg.vector_norm(compact_lengths, dim=-1)
        n = int(scene_rows.numel())
        if n:
            forest = forest_override or build_forest(
                    compact_states,
                    compact_coordinates,
                    compact_lengths,
                    forest_measure,
                    compact_ids_tensor,
                    bank,
                    k=self.forest_neighbors,
                )
            force_leaves = self.architecture == "J-D"
            frontier = self.frontier_policies[bank](
                forest,
                epoch=self._organization_epoch,
                update=self._organization_update,
                case_key=(case_key, bank),
                training=bool(self.training),
                force_leaves=force_leaves,
                forced_subtree=forced_subtree,
            )
        else:
            forest = None
            frontier = None

        live_indices = () if frontier is None else tuple(int(v) for v in frontier.node_indices)
        group_states: list[Tensor] = []
        group_centers: list[Tensor] = []
        group_extents: list[Tensor] = []
        group_outer_radii: list[Tensor] = []
        group_masses: list[Tensor] = []
        group_ancestry: list[tuple[int, ...]] = []
        donor_local_indices: list[tuple[int, ...]] = []
        membership_rows: list[Tensor] = []
        membership_net = self.module_group_membership if bank == "module" else self.environment_group_membership
        id_order = {value: index for index, value in enumerate(forest.original_ids)} if forest is not None else {}
        for node_index in live_indices:
            node = forest.nodes[node_index]
            ancestry = tuple(int(value) for value in node.ancestry_local_indices)
            ancestry_tensor = torch.as_tensor(ancestry, device=compact_states.device, dtype=torch.long)
            node_state = node.contextual_state.to(device=compact_states.device, dtype=compact_states.dtype)
            member_coordinates = compact_coordinates.index_select(0, ancestry_tensor)
            member_lengths = compact_lengths.index_select(0, ancestry_tensor)
            half_lengths = 0.5 * member_lengths.abs()
            lower = (member_coordinates - half_lengths).amin(dim=0)
            upper = (member_coordinates + half_lengths).amax(dim=0)
            center = 0.5 * (lower + upper)
            extent = (upper - lower).clamp_min(0.0)
            if bank == "module":
                mass = compact_measure.new_tensor(float(len(ancestry)))
            else:
                mass = information_measure.index_select(0, ancestry_tensor).sum()
            group_states.append(node_state)
            group_centers.append(center)
            group_extents.append(extent)
            group_outer_radii.append(member_radii.index_select(0, ancestry_tensor).amax())
            group_masses.append(mass)
            group_ancestry.append(ancestry)

            donor_set = set(ancestry)
            if forest is not None:
                for member in ancestry:
                    donor_set.update(forest.spatial_neighbors[member])
            donors = tuple(sorted(donor_set, key=lambda value: id_order[forest.original_ids[value]]))
            donor_tensor = torch.as_tensor(donors, device=compact_states.device, dtype=torch.long)
            donor_states = compact_states.index_select(0, donor_tensor)
            donor_coordinates = compact_coordinates.index_select(0, donor_tensor)
            donor_info = information_measure.index_select(0, donor_tensor)
            donor_info_for_log = donor_info.clamp_min(torch.finfo(compact_states.dtype).tiny)
            scale = frame_scale.to(device=compact_states.device, dtype=compact_states.dtype).clamp_min(
                torch.finfo(compact_states.dtype).eps
            )
            relative = (donor_coordinates - center) / scale
            distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
            extent_features = (extent / scale).expand_as(relative)
            membership_features = torch.cat((
                node_state.expand_as(donor_states),
                donor_states,
                relative,
                extent_features,
                distance,
                torch.log(donor_info_for_log).unsqueeze(-1),
            ), dim=-1)
            gates = positive_gate(membership_net(membership_features).squeeze(-1))
            gates = gates * (donor_info > 0).to(gates.dtype)
            membership_rows.append(torch.zeros(n, device=compact_states.device, dtype=compact_states.dtype)
                                   .index_copy(0, donor_tensor, gates))
            donor_local_indices.append(donors)

        k_live = len(live_indices)
        membership = (torch.stack(membership_rows) if membership_rows else
                      compact_states.new_zeros((0, n)))
        group_state_tensor = (torch.stack(group_states) if group_states else
                              compact_states.new_zeros((0, self.context_dim)))
        group_center_tensor = (torch.stack(group_centers) if group_centers else
                               compact_coordinates.new_zeros((0, self.coordinate_dim)))
        group_extent_tensor = (torch.stack(group_extents) if group_extents else
                               compact_coordinates.new_zeros((0, self.coordinate_dim)))
        group_outer_radius_tensor = (torch.stack(group_outer_radii) if group_outer_radii else
                                     compact_measure.new_zeros((0,)))
        group_mass_tensor = (torch.stack(group_masses) if group_masses else compact_measure.new_zeros((0,)))
        result = PreparedRecordBank(
            bank=bank,
            forest=forest,
            frontier=frontier,
            scene_rows=scene_rows,
            context_states=compact_states,
            keys=compact_keys,
            coordinates=compact_coordinates,
            lengths=compact_lengths,
            measure=compact_measure,
            information_measure=information_measure,
            member_radii=member_radii,
            group_states=group_state_tensor,
            group_centers=group_center_tensor,
            group_extents=group_extent_tensor,
            group_outer_radii=group_outer_radius_tensor,
            group_mass=group_mass_tensor,
            group_ancestry=tuple(group_ancestry),
            membership=membership,
            donor_local_indices=tuple(donor_local_indices),
        )
        stats = {
            f"{bank}_present_records": n,
            f"{bank}_eligible_records": int((information_measure > 0).sum().item()),
            f"{bank}_candidate_nodes": 0 if forest is None else len(forest.nodes),
            f"{bank}_candidate_edges": 0 if forest is None else len(forest.candidate_edges),
            f"{bank}_live_groups": k_live,
            f"{bank}_allocated_groups": k_live,
            f"{bank}_membership_cells": int(membership.numel()),
            f"{bank}_membership_donor_cells": sum(len(values) for values in donor_local_indices),
        }
        if forest is not None:
            stats[f"{bank}_spatial_neighbor_edges"] = sum(len(row) for row in forest.spatial_neighbors) // 2
            stats[f"{bank}_context_neighbor_edges"] = sum(len(row) for row in forest.context_neighbors) // 2
            pair_count = n * max(0, n - 1) // 2
            stats[f"{bank}_candidate_geometry_distance_pairs"] = pair_count
            stats[f"{bank}_candidate_context_distance_pairs"] = pair_count
            stats[f"{bank}_candidate_distance_pairs"] = 2 * pair_count
        else:
            stats[f"{bank}_spatial_neighbor_edges"] = 0
            stats[f"{bank}_context_neighbor_edges"] = 0
            stats[f"{bank}_candidate_distance_pairs"] = 0
            stats[f"{bank}_candidate_geometry_distance_pairs"] = 0
            stats[f"{bank}_candidate_context_distance_pairs"] = 0
        return result, stats

    def prepare(self, scene: SceneBatch, *, return_trace: bool = False) -> PreparedContextualOrganization:
        self._check_scene(scene)
        parameter_versions = self._parameter_version_stamp()
        buffer_versions = self._buffer_version_stamp()
        scene_versions = tuple(_stamp_tensor(value) for value in _scene_tensors(scene))
        identity = (scene.scene_key, scene.normalizer_key, scene.coordinate_frame_key)
        module_states, environment_states, origin, scale = self._scene_states(scene)
        module_keys = self._project_present(module_states, scene.module_mask, self.module_key)
        environment_keys = self._project_present(
            environment_states, scene.environment_mask, self.environment_key
        )
        batch = scene.batch_size
        module_banks: list[PreparedRecordBank] = []
        environment_banks: list[PreparedRecordBank] = []
        counts: dict[str, int] = {
            "contextual_module_records": int(scene.module_mask.sum().item()),
            "contextual_environment_records": int(scene.environment_mask.sum().item()),
            "contextual_attention_cycles": self.context_depth,
            "module_projected_key_rows": int(scene.module_mask.sum().item()),
            "environment_projected_key_rows": int(scene.environment_mask.sum().item()),
        }
        module_reference = scene.module_measure_reference
        environment_reference = scene.environment_measure_reference
        if module_reference is None:
            module_reference = scene.module_measure.new_ones(batch)
        if environment_reference is None:
            environment_reference = scene.environment_measure.new_ones(batch)
        for batch_index in range(batch):
            case_key = self._scene_case_key(scene.scene_key, batch_index, batch)
            module_bank, module_counts = self._prepare_bank(
                bank="module",
                states=module_states[batch_index],
                keys=module_keys[batch_index],
                coordinates=scene.module_coordinates[batch_index],
                lengths=scene.module_characteristic_lengths[batch_index],
                measures=scene.module_measure[batch_index],
                ids=scene.module_ids[batch_index],
                present=scene.module_mask[batch_index],
                scene_rows_all=torch.arange(scene.num_modules, device=scene.module_mask.device),
                information_reference=module_reference[batch_index],
                local_radius=scene.local_radius[batch_index],
                case_key=case_key,
                frame_scale=scale[batch_index],
            )
            environment_bank, environment_counts = self._prepare_bank(
                bank="environment",
                states=environment_states[batch_index],
                keys=environment_keys[batch_index],
                coordinates=scene.environment_coordinates[batch_index],
                lengths=scene.environment_characteristic_lengths[batch_index],
                measures=scene.environment_measure[batch_index],
                ids=scene.environment_ids[batch_index],
                present=scene.environment_mask[batch_index],
                scene_rows_all=torch.arange(scene.num_environment_tokens, device=scene.environment_mask.device),
                information_reference=environment_reference[batch_index],
                local_radius=None,
                case_key=case_key,
                frame_scale=scale[batch_index],
            )
            module_banks.append(module_bank)
            environment_banks.append(environment_bank)
            for key, value in (*module_counts.items(), *environment_counts.items()):
                counts[key] = counts.get(key, 0) + int(value)
        return PreparedContextualOrganization(
            scene=scene,
            module_states=module_states,
            environment_states=environment_states,
            module_keys=module_keys,
            environment_keys=environment_keys,
            module_banks=tuple(module_banks),
            environment_banks=tuple(environment_banks),
            model_owner=id(self),
            model_parameter_versions=parameter_versions,
            model_buffer_versions=buffer_versions,
            scene_tensor_versions=scene_versions,
            scene_identity_snapshot=identity,
            model_config_key=self._config_key(),
            frame_origin=origin,
            frame_scale=scale,
            work_counts=counts,
        )

    def _assert_prepared(self, prepared: PreparedContextualOrganization, query: QueryBatch) -> None:
        if prepared.model_owner != id(self):
            raise ValueError("Prepared contextual organization belongs to a different core instance.")
        if prepared.model_parameter_versions != self._parameter_version_stamp():
            raise ValueError("Core parameters changed after contextual preparation.")
        if prepared.model_buffer_versions != self._buffer_version_stamp():
            raise ValueError("Core buffers changed after contextual preparation.")
        if prepared.model_config_key != self._config_key():
            raise ValueError("Core configuration or organization clock changed after preparation.")
        if prepared.scene_tensor_versions != tuple(_stamp_tensor(value) for value in _scene_tensors(prepared.scene)):
            raise ValueError("Scene tensors changed after contextual preparation.")
        if prepared.scene_identity_snapshot != (
            prepared.scene.scene_key, prepared.scene.normalizer_key, prepared.scene.coordinate_frame_key
        ):
            raise ValueError("Scene identity changed after contextual preparation.")
        if query.scene_key is not None and query.scene_key != prepared.scene.scene_key:
            raise ValueError("Query scene_key does not match the prepared physical inputs.")
        if query.coordinate_frame_key is not None and query.coordinate_frame_key != prepared.scene.coordinate_frame_key:
            raise ValueError("Query coordinate frame does not match the prepared scene.")
        query.validate(
            batch_size=prepared.scene.batch_size,
            coordinate_dim=self.coordinate_dim,
            query_input_dim=self.query_input_dim,
        )

    def _query_initial(self, prepared: PreparedContextualOrganization, query: QueryBatch) -> Tensor:
        dtype = next(self.parameters()).dtype
        coordinates = query.coordinates.to(dtype=dtype)
        gamma = coordinate_gamma(coordinates, prepared.frame_origin.to(dtype), prepared.frame_scale.to(dtype))
        if self.query_input_dim:
            features = query.features.to(dtype=dtype)
            inputs = torch.cat((features, gamma), dim=-1)
        else:
            inputs = gamma
        result = self.query_encoder(inputs)
        scene = prepared.scene
        if self.operating_encoder is not None:
            operating = self.operating_encoder(scene.operating_features.to(dtype=dtype))
            result = result + operating[:, None, :]
        return self.query_norm(result)

    def _group_support_tile(
        self,
        bank: PreparedRecordBank,
        query_coordinates: Tensor,
        query_mask: Tensor,
    ) -> tuple[Tensor, int]:
        """Geometry-prefilter group/query pairs, then evaluate member envelopes."""
        qn = int(query_coordinates.shape[0])
        groups = bank.live_count
        if groups == 0 or qn == 0:
            return query_coordinates.new_zeros((qn, groups)), qn * groups
        lower = bank.group_centers - 0.5 * bank.group_extents
        upper = bank.group_centers + 0.5 * bank.group_extents
        outside = (lower[None, :, :] - query_coordinates[:, None, :]).clamp_min(0.0) + (
            query_coordinates[:, None, :] - upper[None, :, :]
        ).clamp_min(0.0)
        box_distance = torch.linalg.vector_norm(outside, dim=-1)
        possible = (box_distance <= bank.group_outer_radii[None, :]) & query_mask[:, None]
        columns: list[Tensor] = []
        for group_index, ancestry in enumerate(bank.group_ancestry):
            query_rows = torch.nonzero(possible[:, group_index], as_tuple=False).flatten()
            if query_rows.numel() == 0:
                columns.append(query_coordinates.new_zeros((qn,)))
                continue
            member_rows = torch.as_tensor(ancestry, device=bank.coordinates.device, dtype=torch.long)
            member_coordinates = bank.coordinates.index_select(0, member_rows)
            member_radii = bank.member_radii.index_select(0, member_rows)
            selected_coordinates = query_coordinates.index_select(0, query_rows)
            distance = torch.linalg.vector_norm(
                selected_coordinates[:, None, :] - member_coordinates[None, :, :], dim=-1
            )
            member_windows = quintic_window(
                distance, member_radii[None, :], member_radii[None, :] * 0.2
            )
            exact_support = smooth_union(member_windows, dim=-1)
            columns.append(query_coordinates.new_zeros((qn,)).index_copy(0, query_rows, exact_support))
        return torch.stack(columns, dim=-1), qn * groups

    @staticmethod
    def _aggregate_sparse_edges(
        query_rows: Tensor,
        source_rows: Tensor,
        gate_values: Tensor,
        source_count: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Coalesce differentiable duplicate (query, source) edges."""
        if query_rows.numel() == 0:
            return query_rows, source_rows, gate_values
        linear = query_rows * source_count + source_rows
        unique, inverse = torch.unique(linear, sorted=True, return_inverse=True)
        values = gate_values.new_zeros((unique.numel(),)).index_add(0, inverse, gate_values)
        keep = values > 0
        unique = unique[keep]
        values = values[keep]
        return torch.div(unique, source_count, rounding_mode="floor"), unique.remainder(source_count), values

    def _route_bank_tiled(
        self,
        bank: PreparedRecordBank,
        query_states: Tensor,
        query_coordinates: Tensor,
        frame_scale: Tensor,
        query_mask: Tensor,
    ) -> tuple[Tensor, dict[str, int]]:
        """Score only geometrically supported query/group pairs in bounded tiles."""
        qn, groups = int(query_states.shape[0]), bank.live_count
        group_access = query_states.new_zeros((qn, groups))
        if groups == 0 or qn == 0:
            return group_access, {"group_bounds_tested": qn * groups, "group_score_rows": 0}
        route_net = self.module_group_router if bank.bank == "module" else self.environment_group_router
        eps = torch.finfo(query_states.dtype).eps
        scale = frame_scale.to(device=query_states.device, dtype=query_states.dtype).clamp_min(eps)
        scored_rows = 0
        for start in range(0, qn, self.query_tile_size):
            stop = min(qn, start + self.query_tile_size)
            qstate = query_states[start:stop]
            qcoord = query_coordinates[start:stop]
            qmask = query_mask[start:stop]
            support, _ = self._group_support_tile(bank, qcoord, qmask)
            eligible = (support > 0) & (bank.group_mass[None, :] > 0)
            pairs = torch.nonzero(eligible, as_tuple=False)
            if pairs.numel() == 0:
                continue
            qlocal, group_rows = pairs.unbind(dim=-1)
            global_q = qlocal + start
            centers = bank.group_centers.index_select(0, group_rows)
            extents = bank.group_extents.index_select(0, group_rows)
            descriptors = bank.group_states.index_select(0, group_rows)
            relative = (query_coordinates.index_select(0, global_q) - centers) / scale
            extent_features = extents / scale
            distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
            features = torch.cat((
                query_states.index_select(0, global_q), descriptors, relative, extent_features, distance,
            ), dim=-1)
            logits = route_net(features).squeeze(-1)
            omega = (
                bank.group_mass.index_select(0, group_rows)
                * support[qlocal, group_rows]
                * positive_gate(logits)
            )
            denominators = query_states.new_ones((stop - start,)).index_add(0, qlocal, omega)
            access = omega / denominators.index_select(0, qlocal)
            group_access = group_access.index_put((global_q, group_rows), access, accumulate=True)
            scored_rows += int(pairs.shape[0])
        return group_access, {
            "group_bounds_tested": qn * groups,
            "group_score_rows": scored_rows,
        }

    def _group_record_edges(
        self,
        bank: PreparedRecordBank,
        access: Tensor,
        *,
        query_offset: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Expand selected group access through direct ancestry-plus-halo C rows."""
        query_parts: list[Tensor] = []
        source_parts: list[Tensor] = []
        value_parts: list[Tensor] = []
        for group_index, donors in enumerate(bank.donor_local_indices):
            donor_rows = torch.as_tensor(donors, device=access.device, dtype=torch.long)
            if donor_rows.numel() == 0:
                continue
            c_values = bank.membership[group_index].index_select(0, donor_rows)
            valid = c_values > 0
            donor_rows = donor_rows[valid]
            c_values = c_values[valid]
            if donor_rows.numel() == 0:
                continue
            selected_queries = torch.nonzero(access[:, group_index] > 0, as_tuple=False).flatten()
            if selected_queries.numel() == 0:
                continue
            query_parts.append(
                (selected_queries[:, None].expand(-1, donor_rows.numel()) + query_offset).reshape(-1)
            )
            source_parts.append(donor_rows[None, :].expand(selected_queries.numel(), -1).reshape(-1))
            value_parts.append(
                (access.index_select(0, selected_queries)[:, group_index, None] * c_values[None, :]).reshape(-1)
            )
        if not query_parts:
            empty = torch.empty((0,), device=access.device, dtype=torch.long)
            return empty, empty, access.new_zeros((0,))
        return self._aggregate_sparse_edges(
            torch.cat(query_parts), torch.cat(source_parts), torch.cat(value_parts), bank.present_count
        )

    def _local_module_edges(
        self,
        scene: SceneBatch,
        query: QueryBatch,
        batch_index: int,
        query_start: int,
        query_stop: int,
        bank: PreparedRecordBank,
        query_coordinates: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, int, int]:
        """Return smooth near-field M edges and exact local-work counters.

        ``geometry_rows`` counts the dense query-by-module envelope evaluations
        performed by the fallback path, including pairs outside the envelope.
        ``packed_pair_rows`` counts supplied native local pairs mapped by the
        decoder; that path consumes precomputed geometry and does not claim to
        have reevaluated the envelope.
        """
        if bank.present_count == 0 or query_coordinates.shape[0] == 0:
            empty = torch.empty((0,), device=query_coordinates.device, dtype=torch.long)
            return empty, empty, query_coordinates.new_zeros((0,)), 0, 0
        if query.local_batch_indices is not None:
            keep = (
                (query.local_batch_indices == batch_index)
                & (query.local_query_indices >= query_start)
                & (query.local_query_indices < query_stop)
            )
            physical_query = query.local_query_indices[keep].to(torch.long)
            physical_module = query.local_module_indices[keep].to(torch.long)
            values = query.local_weights[keep].to(dtype=query_coordinates.dtype)
            if physical_query.numel() == 0:
                empty = torch.empty((0,), device=query_coordinates.device, dtype=torch.long)
                return empty, empty, query_coordinates.new_zeros((0,)), 0, 0
            scene_to_local = torch.full(
                (scene.num_modules,), -1, device=bank.scene_rows.device, dtype=torch.long
            )
            scene_to_local[bank.scene_rows] = torch.arange(bank.present_count, device=bank.scene_rows.device)
            local_source = scene_to_local.index_select(0, physical_module)
            valid = local_source >= 0
            local_query = physical_query[valid] - query_start
            local_source = local_source[valid]
            values = values[valid]
            return self._aggregate_sparse_edges(local_query, local_source, values, bank.present_count) + (
                0, int(physical_query.numel())
            )

        radii = scene.local_radius[batch_index].index_select(0, bank.scene_rows).to(query_coordinates.dtype)
        displacement = query_coordinates[:, None, :] - bank.coordinates[None, :, :]
        distance = torch.linalg.vector_norm(displacement, dim=-1)
        window = quintic_window(distance, radii[None, :], radii[None, :] * 0.2)
        local_query, local_source = torch.nonzero(window > 0, as_tuple=True)
        if local_query.numel() == 0:
            empty = torch.empty((0,), device=query_coordinates.device, dtype=torch.long)
            return empty, empty, query_coordinates.new_zeros((0,)), int(window.numel()), 0
        edges = self._aggregate_sparse_edges(
            local_query, local_source, window[local_query, local_source], bank.present_count
        )
        return (*edges, int(window.numel()), 0)

    def _combine_module_edges(
        self,
        bank: PreparedRecordBank,
        route_edges: tuple[Tensor, Tensor, Tensor],
        local_edges: tuple[Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        r_q, r_s, r_v = route_edges
        l_q, l_s, l_v = local_edges
        if r_q.numel() == 0 and l_q.numel() == 0:
            return r_q, r_s, r_v
        all_q = torch.cat((r_q, l_q))
        all_s = torch.cat((r_s, l_s))
        all_v = torch.cat((r_v, l_v))
        local_indicator = torch.cat((r_v.new_zeros(r_v.shape), l_v.new_ones(l_v.shape)))
        linear = all_q * bank.present_count + all_s
        unique, inverse = torch.unique(linear, sorted=True, return_inverse=True)
        route_values = r_v.new_zeros((unique.numel(),)).index_add(
            0, inverse, all_v * (1.0 - local_indicator)
        )
        local_values = r_v.new_zeros((unique.numel(),)).index_add(
            0, inverse, all_v * local_indicator
        ).clamp(0.0, 1.0)
        gates = local_values + (1.0 - local_values) * route_values
        keep = gates > 0
        unique = unique[keep]
        gates = gates[keep]
        return torch.div(unique, bank.present_count, rounding_mode="floor"), unique.remainder(bank.present_count), gates

    def _source_context_tiled(
        self,
        *,
        bank: PreparedRecordBank,
        query_states: Tensor,
        query_coordinates: Tensor,
        frame_scale: Tensor,
        edge_query_rows: Tensor,
        edge_source_rows: Tensor,
        edge_gates: Tensor,
        union_indices: Tensor,
        union_values: Tensor,
        collect_trace: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor | int]]:
        """Score only selected query/source pairs and normalize against a null read."""
        qn = int(query_states.shape[0])
        contexts = query_states.new_zeros((qn, self.context_dim))
        if edge_query_rows.numel() == 0:
            return contexts, {
                "query_key_score_rows": 0,
                "query_projection_rows": 0,
                "value_read_rows": 0,
                "edge_query_rows": edge_query_rows if collect_trace else None,
                "edge_source_rows": edge_source_rows if collect_trace else None,
                "weights": query_states.new_zeros((0, self.num_heads)) if collect_trace else None,
            }
        key = edge_query_rows * bank.present_count + edge_source_rows
        order = torch.argsort(key, stable=True)
        edge_query_rows = edge_query_rows.index_select(0, order)
        edge_source_rows = edge_source_rows.index_select(0, order)
        edge_gates = edge_gates.index_select(0, order)
        output_parts: list[tuple[int, int, Tensor]] = []
        weight_parts: list[Tensor] = []
        query_projection_rows = 0
        eps = torch.finfo(query_states.dtype).eps
        scale = frame_scale.to(device=query_states.device, dtype=query_states.dtype).clamp_min(eps)
        value_positions = torch.searchsorted(union_indices, edge_source_rows)
        pair_values = union_values.index_select(0, value_positions).reshape(
            -1, self.num_heads, self.context_dim // self.num_heads
        )
        query_projector = self.module_query_key if bank.bank == "module" else self.environment_query_key
        geometry_net = self.module_source_geometry if bank.bank == "module" else self.environment_source_geometry
        source_info = bank.information_measure.index_select(0, edge_source_rows)
        valid_edge = source_info > 0
        edge_query_rows = edge_query_rows[valid_edge]
        edge_source_rows = edge_source_rows[valid_edge]
        edge_gates = edge_gates[valid_edge]
        pair_values = pair_values[valid_edge]
        source_info = source_info[valid_edge]
        if edge_query_rows.numel() == 0:
            return contexts, {
                "query_key_score_rows": 0,
                "query_projection_rows": 0,
                "value_read_rows": 0,
                "edge_query_rows": edge_query_rows if collect_trace else None,
                "edge_source_rows": edge_source_rows if collect_trace else None,
                "weights": query_states.new_zeros((0, self.num_heads)) if collect_trace else None,
            }
        # Query/source rows are sorted by query, so process contiguous query tiles.
        for start in range(0, qn, self.query_tile_size):
            stop = min(qn, start + self.query_tile_size)
            left = torch.searchsorted(edge_query_rows, torch.tensor(start, device=query_states.device), right=False)
            right = torch.searchsorted(edge_query_rows, torch.tensor(stop, device=query_states.device), right=False)
            if int(right - left) == 0:
                continue
            local_q = edge_query_rows[left:right]
            local_sources = edge_source_rows[left:right]
            local_gates = edge_gates[left:right]
            local_values = pair_values[left:right]
            local_measure = source_info[left:right]
            used_q, inverse_q = torch.unique(local_q, sorted=True, return_inverse=True)
            query_keys = query_projector(query_states.index_select(0, used_q)).reshape(
                used_q.numel(), self.num_heads, self.context_dim // self.num_heads
            )
            query_projection_rows += int(used_q.numel())
            pair_query_keys = query_keys.index_select(0, inverse_q)
            source_keys = bank.keys.index_select(0, local_sources).reshape(
                -1, self.num_heads, self.context_dim // self.num_heads
            )
            logits = (pair_query_keys * source_keys).sum(dim=-1) / (self.context_dim // self.num_heads) ** 0.5
            delta = bank.coordinates.index_select(0, local_sources) - query_coordinates.index_select(0, local_q)
            normalized = delta / scale
            local = torch.asinh(delta / bank.lengths.index_select(0, local_sources).abs().clamp_min(eps))
            distance = torch.linalg.vector_norm(normalized, dim=-1, keepdim=True)
            geometry = torch.cat((normalized, local, distance), dim=-1)
            logits = logits + geometry_net(geometry)
            log_weight = (
                torch.log(local_gates.clamp_min(torch.finfo(local_gates.dtype).tiny))[:, None]
                + torch.log(local_measure.clamp_min(torch.finfo(local_measure.dtype).tiny))[:, None]
                + logits
            )
            segment = (local_q - start)[:, None] * self.num_heads + torch.arange(
                self.num_heads, device=query_states.device
            )[None, :]
            segment_flat = segment.reshape(-1)
            log_flat = log_weight.reshape(-1)
            tile_queries = stop - start
            max_by_segment = query_states.new_zeros((tile_queries * self.num_heads,)).scatter_reduce(
                0, segment_flat, log_flat, reduce="amax", include_self=True
            )
            exp_pair = torch.exp(log_flat - max_by_segment.index_select(0, segment_flat))
            sum_by_segment = torch.exp(-max_by_segment).index_add(
                0, segment_flat, exp_pair
            )
            weights = (exp_pair / sum_by_segment.index_select(0, segment_flat)).reshape(-1, self.num_heads)
            head_rows = (local_q - start)[:, None] * self.num_heads + torch.arange(
                self.num_heads, device=query_states.device
            )[None, :]
            aggregate = query_states.new_zeros((tile_queries * self.num_heads, self.context_dim // self.num_heads)).index_add(
                0,
                head_rows.reshape(-1),
                (weights[:, :, None] * local_values).reshape(-1, self.context_dim // self.num_heads),
            )
            tile_context = aggregate.reshape(tile_queries, self.num_heads, self.context_dim // self.num_heads).reshape(
                tile_queries, self.context_dim
            )
            output_parts.append((start, stop, tile_context))
            if collect_trace:
                weight_parts.append(weights)
        for start, stop, context in output_parts:
            contexts = contexts.index_copy(0, torch.arange(start, stop, device=query_states.device), context)
        all_weights = (torch.cat(weight_parts, dim=0) if weight_parts else
                       query_states.new_zeros((0, self.num_heads))) if collect_trace else None
        return contexts, {
            "query_key_score_rows": int(edge_query_rows.numel()),
            "query_projection_rows": query_projection_rows,
            "value_read_rows": int(edge_query_rows.numel()),
            "edge_query_rows": edge_query_rows if collect_trace else None,
            "edge_source_rows": edge_source_rows if collect_trace else None,
            "weights": all_weights,
        }

    def decode_reference(
        self,
        prepared: PreparedContextualOrganization,
        query: QueryBatch,
    ) -> ContextualJointOutput:
        """Small-panel per-query reference for the tiled sparse decoder.

        This intentionally simple implementation is useful for CPU parity
        checks on small scenes. Production calls should use :meth:`decode`,
        which tiles query/group routing and coalesces selected source pairs.
        """
        self._assert_prepared(prepared, query)
        dtype = next(self.parameters()).dtype
        query_states = self._query_initial(prepared, query)
        query_coordinates = query.coordinates.to(dtype=dtype)
        batch_size, query_count = query_states.shape[:2]
        environment_context_batches: list[Tensor] = []
        module_context_batches: list[Tensor] = []

        def route_one(bank: PreparedRecordBank, qstate: Tensor, qcoord: Tensor, frame_scale: Tensor):
            groups = bank.live_count
            if groups == 0:
                return qstate.new_zeros((0,)), qstate.new_zeros((bank.present_count,))
            support, _ = self._group_support_tile(
                bank, qcoord.reshape(1, -1), torch.ones((1,), dtype=torch.bool, device=qcoord.device)
            )
            eligible = torch.nonzero((support[0] > 0) & (bank.group_mass > 0), as_tuple=False).flatten()
            access = qstate.new_zeros((groups,))
            if eligible.numel() == 0:
                return access, access @ bank.membership
            centers = bank.group_centers.index_select(0, eligible)
            extents = bank.group_extents.index_select(0, eligible)
            descriptors = bank.group_states.index_select(0, eligible)
            scale = frame_scale.to(qstate).clamp_min(torch.finfo(qstate.dtype).eps)
            relative = (qcoord[None, :] - centers) / scale
            extent_features = extents / scale
            distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
            features = torch.cat((qstate.expand_as(descriptors), descriptors, relative,
                                  extent_features, distance), dim=-1)
            route_net = self.module_group_router if bank.bank == "module" else self.environment_group_router
            logits = route_net(features).squeeze(-1)
            selected_access = null_normalized_group_access(
                bank.group_mass.index_select(0, eligible), support[0].index_select(0, eligible), logits
            )
            access = access.index_copy(0, eligible, selected_access)
            return access, access @ bank.membership

        def read_one(
            bank: PreparedRecordBank,
            gates: Tensor,
            qstate: Tensor,
            qcoord: Tensor,
            frame_scale: Tensor,
        ) -> Tensor:
            selected = torch.nonzero((gates > 0) & (bank.information_measure > 0), as_tuple=False).flatten()
            if selected.numel() == 0:
                return qstate.new_zeros((self.context_dim,))
            query_projector = self.module_query_key if bank.bank == "module" else self.environment_query_key
            geometry_net = self.module_source_geometry if bank.bank == "module" else self.environment_source_geometry
            value_projector = self.module_value if bank.bank == "module" else self.environment_value
            head_width = self.context_dim // self.num_heads
            query_key = query_projector(qstate).reshape(self.num_heads, head_width)
            keys = bank.keys.index_select(0, selected).reshape(-1, self.num_heads, head_width)
            logits = torch.einsum("hd,nhd->hn", query_key, keys) / (head_width ** 0.5)
            delta = bank.coordinates.index_select(0, selected) - qcoord[None, :]
            normalized = delta / frame_scale.to(delta).clamp_min(torch.finfo(delta.dtype).eps)
            local = torch.asinh(delta / bank.lengths.index_select(0, selected).abs().clamp_min(
                torch.finfo(delta.dtype).eps
            ))
            distance = torch.linalg.vector_norm(normalized, dim=-1, keepdim=True)
            geometry = torch.cat((normalized, local, distance), dim=-1)
            logits = logits + geometry_net(geometry).transpose(0, 1)
            selected_gates = gates.index_select(0, selected)
            selected_measure = bank.information_measure.index_select(0, selected)
            weights = null_normalized_weights(
                selected_gates[None, :].expand(self.num_heads, -1), selected_measure, logits
            )
            values = value_projector(bank.context_states.index_select(0, selected)).reshape(
                -1, self.num_heads, head_width
            )
            return torch.einsum("hn,nhd->hd", weights, values).reshape(self.context_dim)

        for batch_index in range(batch_size):
            module_bank = prepared.module_banks[batch_index]
            environment_bank = prepared.environment_banks[batch_index]
            query_mask = (query.query_mask[batch_index] if query.query_mask is not None else
                          torch.ones(query_count, dtype=torch.bool, device=query.coordinates.device))
            env_contexts: list[Tensor] = []
            mod_contexts: list[Tensor] = []
            for query_index in range(query_count):
                qstate = query_states[batch_index, query_index]
                qcoord = query_coordinates[batch_index, query_index]
                if not bool(query_mask[query_index]):
                    env_contexts.append(qstate.new_zeros((self.context_dim,)))
                    mod_contexts.append(qstate.new_zeros((self.context_dim,)))
                    continue
                env_access, env_gates = route_one(
                    environment_bank, qstate, qcoord, prepared.frame_scale[batch_index]
                )
                mod_access, mod_gates = route_one(
                    module_bank, qstate, qcoord, prepared.frame_scale[batch_index]
                )
                local_q, local_source, local_values, _, _ = self._local_module_edges(
                    prepared.scene, query, batch_index, query_index, query_index + 1,
                    module_bank, qcoord.reshape(1, -1),
                )
                local = qstate.new_zeros((module_bank.present_count,))
                if local_source.numel():
                    local = local.index_add(0, local_source, local_values).clamp(0.0, 1.0)
                mod_gates = local + (1.0 - local) * mod_gates
                env_context = read_one(
                    environment_bank, env_gates, qstate, qcoord, prepared.frame_scale[batch_index]
                )
                conditioned = self.query_norm(qstate + self.environment_conditioning(env_context))
                mod_context = read_one(
                    module_bank, mod_gates, conditioned, qcoord, prepared.frame_scale[batch_index]
                )
                env_contexts.append(env_context)
                mod_contexts.append(mod_context)
            environment_context_batches.append(torch.stack(env_contexts) if env_contexts else
                                                query_states.new_zeros((0, self.context_dim)))
            module_context_batches.append(torch.stack(mod_contexts) if mod_contexts else
                                           query_states.new_zeros((0, self.context_dim)))
        environment_context = torch.stack(environment_context_batches)
        module_context = torch.stack(module_context_batches)
        fused = torch.cat((query_states, environment_context, module_context), dim=-1)
        fields = self.field_head(self.fusion_trunk(fused))
        if query.query_mask is not None:
            fields = fields * query.query_mask[:, :, None].to(fields.dtype)
        return ContextualJointOutput(
            nonlinear_field=fields[..., :min(3, self.output_dim)],
            thermal_temperature=fields[..., 3:4] if self.output_dim >= 4 else None,
            thermal_basis=None,
            trace=None,
            work_counts={"decode_reference": 1},
            joint_fields=fields,
        )

    def decode(
        self,
        prepared: PreparedContextualOrganization,
        query: QueryBatch,
        *,
        return_trace: bool = False,
        trace_request: Any = None,
    ) -> ContextualJointOutput:
        self._assert_prepared(prepared, query)
        scene = prepared.scene
        dtype = next(self.parameters()).dtype
        query_states = self._query_initial(prepared, query)
        query_coordinates = query.coordinates.to(dtype=dtype)
        batch, query_count = query_states.shape[:2]
        counts_to_add = {
            "module_group_score_rows": 0,
            "environment_group_score_rows": 0,
            "module_group_bounds_tested": 0,
            "environment_group_bounds_tested": 0,
            "module_query_key_score_rows": 0,
            "environment_query_key_score_rows": 0,
            "module_query_projection_rows": 0,
            "environment_query_projection_rows": 0,
            "module_selected_value_projection_rows": 0,
            "environment_selected_value_projection_rows": 0,
            "module_value_read_rows": 0,
            "environment_value_read_rows": 0,
            "module_cross_query_reused_value_rows": 0,
            "environment_cross_query_reused_value_rows": 0,
            "module_query_group_access_rows": 0,
            "environment_query_group_access_rows": 0,
            "module_query_source_edges": 0,
            "environment_query_source_edges": 0,
            "module_local_geometry_rows": 0,
            "module_local_packed_pair_rows": 0,
        }
        per_batch_module_pi: list[Tensor] = []
        per_batch_environment_pi: list[Tensor] = []
        per_batch_module_edges: list[tuple[Tensor, Tensor, Tensor]] = []
        per_batch_environment_edges: list[tuple[Tensor, Tensor, Tensor]] = []
        per_batch_module_reads: list[dict[str, Tensor | int]] = []
        per_batch_environment_reads: list[dict[str, Tensor | int]] = []
        context_environment: list[Tensor] = []
        context_module: list[Tensor] = []
        collect_trace = bool(return_trace or trace_request)

        for batch_index in range(batch):
            module_bank = prepared.module_banks[batch_index]
            environment_bank = prepared.environment_banks[batch_index]
            query_mask = (query.query_mask[batch_index] if query.query_mask is not None
                          else torch.ones(query_count, dtype=torch.bool, device=query.coordinates.device))
            batch_queries = query_states[batch_index]
            batch_coordinates = query_coordinates[batch_index]
            env_pi, env_work = self._route_bank_tiled(
                environment_bank, batch_queries, batch_coordinates,
                prepared.frame_scale[batch_index], query_mask,
            )
            mod_pi, mod_work = self._route_bank_tiled(
                module_bank, batch_queries, batch_coordinates,
                prepared.frame_scale[batch_index], query_mask,
            )
            counts_to_add["environment_group_score_rows"] += env_work["group_score_rows"]
            counts_to_add["environment_group_bounds_tested"] += env_work["group_bounds_tested"]
            counts_to_add["module_group_score_rows"] += mod_work["group_score_rows"]
            counts_to_add["module_group_bounds_tested"] += mod_work["group_bounds_tested"]
            env_edges = self._group_record_edges(environment_bank, env_pi)
            mod_route_edges = self._group_record_edges(module_bank, mod_pi)

            local_q_parts: list[Tensor] = []
            local_source_parts: list[Tensor] = []
            local_gate_parts: list[Tensor] = []
            local_geometry_rows = 0
            local_packed_pair_rows = 0
            for start in range(0, query_count, self.query_tile_size):
                stop = min(query_count, start + self.query_tile_size)
                local_q, local_source, local_gate, geometry_rows, packed_pair_rows = self._local_module_edges(
                    scene, query, batch_index, start, stop, module_bank, batch_coordinates[start:stop]
                )
                if local_q.numel():
                    local_q_parts.append(local_q + start)
                    local_source_parts.append(local_source)
                    local_gate_parts.append(local_gate)
                local_geometry_rows += geometry_rows
                local_packed_pair_rows += packed_pair_rows
            if local_q_parts:
                local_edges = self._aggregate_sparse_edges(
                    torch.cat(local_q_parts), torch.cat(local_source_parts), torch.cat(local_gate_parts),
                    module_bank.present_count,
                )
            else:
                empty = torch.empty((0,), device=batch_coordinates.device, dtype=torch.long)
                local_edges = (empty, empty, batch_coordinates.new_zeros((0,)))
            mod_edges = self._combine_module_edges(module_bank, mod_route_edges, local_edges)

            env_union = torch.unique(env_edges[1], sorted=True)
            mod_union = torch.unique(mod_edges[1], sorted=True)
            env_values = (
                self.environment_value(environment_bank.context_states.index_select(0, env_union))
                if env_union.numel() else environment_bank.context_states.new_zeros((0, self.context_dim))
            )
            mod_values = (
                self.module_value(module_bank.context_states.index_select(0, mod_union))
                if mod_union.numel() else module_bank.context_states.new_zeros((0, self.context_dim))
            )
            env_contexts, env_read = self._source_context_tiled(
                bank=environment_bank,
                query_states=batch_queries,
                query_coordinates=batch_coordinates,
                frame_scale=prepared.frame_scale[batch_index],
                edge_query_rows=env_edges[0],
                edge_source_rows=env_edges[1],
                edge_gates=env_edges[2],
                union_indices=env_union,
                union_values=env_values,
                collect_trace=collect_trace,
            )
            qconditioned = self.query_norm(batch_queries + self.environment_conditioning(env_contexts))
            mod_contexts, mod_read = self._source_context_tiled(
                bank=module_bank,
                query_states=qconditioned,
                query_coordinates=batch_coordinates,
                frame_scale=prepared.frame_scale[batch_index],
                edge_query_rows=mod_edges[0],
                edge_source_rows=mod_edges[1],
                edge_gates=mod_edges[2],
                union_indices=mod_union,
                union_values=mod_values,
                collect_trace=collect_trace,
            )
            env_contexts = env_contexts * query_mask[:, None].to(env_contexts.dtype)
            mod_contexts = mod_contexts * query_mask[:, None].to(mod_contexts.dtype)
            counts_to_add["environment_query_key_score_rows"] += int(env_read["query_key_score_rows"])
            counts_to_add["environment_value_read_rows"] += int(env_read["value_read_rows"])
            counts_to_add["module_query_key_score_rows"] += int(mod_read["query_key_score_rows"])
            counts_to_add["module_value_read_rows"] += int(mod_read["value_read_rows"])
            counts_to_add["environment_query_projection_rows"] += int(env_read["query_projection_rows"])
            counts_to_add["module_query_projection_rows"] += int(mod_read["query_projection_rows"])
            counts_to_add["environment_selected_value_projection_rows"] += int(env_union.numel())
            counts_to_add["module_selected_value_projection_rows"] += int(mod_union.numel())
            counts_to_add["module_query_group_access_rows"] += int((mod_pi > 0).sum().item())
            counts_to_add["environment_query_group_access_rows"] += int((env_pi > 0).sum().item())
            counts_to_add["module_query_source_edges"] += int(mod_edges[0].numel())
            counts_to_add["environment_query_source_edges"] += int(env_edges[0].numel())
            counts_to_add["module_local_geometry_rows"] += local_geometry_rows
            counts_to_add["module_local_packed_pair_rows"] += local_packed_pair_rows

            per_batch_module_pi.append(mod_pi)
            per_batch_environment_pi.append(env_pi)
            per_batch_module_edges.append(mod_edges)
            per_batch_environment_edges.append(env_edges)
            per_batch_module_reads.append(mod_read)
            per_batch_environment_reads.append(env_read)
            context_environment.append(env_contexts)
            context_module.append(mod_contexts)

        environment_context = torch.stack(context_environment) if context_environment else query_states.new_zeros((0, query_count, self.context_dim))
        module_context = torch.stack(context_module) if context_module else query_states.new_zeros((0, query_count, self.context_dim))
        fused = torch.cat((query_states, environment_context, module_context), dim=-1)
        joint_fields = self.field_head(self.fusion_trunk(fused))
        if query.query_mask is not None:
            joint_fields = joint_fields * query.query_mask[:, :, None].to(joint_fields.dtype)

        # Keep decode counters scoped to this call. Preparation counters live
        # separately on PreparedContextualOrganization.work_counts; duplicating
        # them here makes provider-level totals silently double count scene work.
        total_counts = {key: int(value) for key, value in counts_to_add.items()}
        total_counts["module_cross_query_reused_value_rows"] = max(
            0, counts_to_add["module_value_read_rows"] - counts_to_add["module_selected_value_projection_rows"]
        )
        total_counts["environment_cross_query_reused_value_rows"] = max(
            0, counts_to_add["environment_value_read_rows"]
            - counts_to_add["environment_selected_value_projection_rows"]
        )
        trace = None
        if return_trace or trace_request:
            def sparse_edge_trace(
                bank: PreparedRecordBank, edges: tuple[Tensor, Tensor, Tensor]
            ) -> dict[str, Tensor | int]:
                return {
                    "query_indices": edges[0],
                    "scene_record_indices": bank.scene_rows.index_select(0, edges[1]),
                    "record_local_indices": edges[1],
                    "gate_values": edges[2],
                    "query_count": query_count,
                }

            trace = {
                "module_membership": tuple(bank.membership for bank in prepared.module_banks),
                "environment_membership": tuple(bank.membership for bank in prepared.environment_banks),
                "module_live_node_indices": tuple(
                    () if bank.frontier is None else tuple(bank.frontier.node_indices) for bank in prepared.module_banks
                ),
                "environment_live_node_indices": tuple(
                    () if bank.frontier is None else tuple(bank.frontier.node_indices) for bank in prepared.environment_banks
                ),
                "module_group_access": tuple(per_batch_module_pi),
                "environment_group_access": tuple(per_batch_environment_pi),
                "module_query_gates": tuple(
                    sparse_edge_trace(bank, edges)
                    for bank, edges in zip(prepared.module_banks, per_batch_module_edges)
                ),
                "environment_query_gates": tuple(
                    sparse_edge_trace(bank, edges)
                    for bank, edges in zip(prepared.environment_banks, per_batch_environment_edges)
                ),
                "module_selected_scene_rows": tuple(
                    bank.scene_rows.index_select(0, edges[1])
                    for bank, edges in zip(prepared.module_banks, per_batch_module_edges)
                ),
                "environment_selected_scene_rows": tuple(
                    bank.scene_rows.index_select(0, edges[1])
                    for bank, edges in zip(prepared.environment_banks, per_batch_environment_edges)
                ),
                "module_source_read_attention": tuple(per_batch_module_reads),
                "environment_source_read_attention": tuple(per_batch_environment_reads),
                "cross_query_value_reuse": {
                    "module": total_counts["module_cross_query_reused_value_rows"],
                    "environment": total_counts["environment_cross_query_reused_value_rows"],
                },
            }
        nonlinear = joint_fields[..., :min(3, self.output_dim)]
        thermal = joint_fields[..., 3:4] if self.output_dim >= 4 else None
        return ContextualJointOutput(
            nonlinear_field=nonlinear,
            thermal_temperature=thermal,
            thermal_basis=None,
            trace=trace,
            work_counts=total_counts,
            joint_fields=joint_fields,
        )

    def organizer_terms(self, prepared: PreparedContextualOrganization) -> dict[str, Any]:
        """Expose typed structural expectations and policy evidence to runtime."""
        module_expected = []
        environment_expected = []
        module_logprob = []
        environment_logprob = []
        module_risk = []
        environment_risk = []
        for bank in prepared.module_banks:
            if bank.frontier is not None:
                module_expected.append(bank.frontier.expected_K)
                module_logprob.append(bank.frontier.log_prob)
                module_risk.append(bank.frontier.risk_logits)
        for bank in prepared.environment_banks:
            if bank.frontier is not None:
                environment_expected.append(bank.frontier.expected_K)
                environment_logprob.append(bank.frontier.log_prob)
                environment_risk.append(bank.frontier.risk_logits)

        def sum_or_zero(values: list[Tensor], device: torch.device) -> Tensor:
            return torch.stack(values).sum() if values else torch.zeros((), device=device)

        device = prepared.module_states.device
        module_count = sum(bank.present_count for bank in prepared.module_banks)
        environment_count = sum(bank.present_count for bank in prepared.environment_banks)
        return {
            "expected_k_module": sum_or_zero(module_expected, device),
            "expected_k_environment": sum_or_zero(environment_expected, device),
            "expected_k_module_fraction": sum_or_zero(module_expected, device) / max(1, module_count),
            "expected_k_environment_fraction": sum_or_zero(environment_expected, device) / max(1, environment_count),
            "log_probability_module": sum_or_zero(module_logprob, device),
            "log_probability_environment": sum_or_zero(environment_logprob, device),
            "risk_logits_module": tuple(module_risk),
            "risk_logits_environment": tuple(environment_risk),
            "frontiers_module": prepared.module_frontiers,
            "frontiers_environment": prepared.environment_frontiers,
        }

    def organizer_parameters(self, bank: str) -> dict[str, nn.Parameter]:
        """Return the complete named trainable inventory for one typed policy.

        Shared context/query parameters are intentionally present in both
        inventories; callers calibrate M and E contributions independently.
        """
        if bank not in {"module", "environment"}:
            raise ValueError("bank must be 'module' or 'environment'.")
        prefixes = (
            "module_encoder.", "environment_encoder.", "context_cycles.",
            "query_encoder.", "query_norm.", "frontier_policies." + bank + ".",
            bank + "_key.", bank + "_query_key.", bank + "_source_geometry.",
            bank + "_group_membership.", bank + "_group_router.",
        )
        if self.operating_encoder is not None:
            prefixes += ("operating_encoder.",)
        if bank == "module":
            prefixes += ("environment_conditioning.",)
        return {
            name: parameter for name, parameter in self.named_parameters()
            if any(name.startswith(prefix) for prefix in prefixes)
        }

    def _assert_prepared_state(self, prepared: PreparedContextualOrganization) -> None:
        if prepared.model_owner != id(self):
            raise ValueError("Prepared contextual organization belongs to a different core instance.")
        if prepared.model_parameter_versions != self._parameter_version_stamp():
            raise ValueError("Core parameters changed after contextual preparation.")
        if prepared.model_buffer_versions != self._buffer_version_stamp():
            raise ValueError("Core buffers changed after contextual preparation.")
        if prepared.model_config_key != self._config_key():
            raise ValueError("Core configuration or organization clock changed after preparation.")
        if prepared.scene_tensor_versions != tuple(_stamp_tensor(value) for value in _scene_tensors(prepared.scene)):
            raise ValueError("Scene tensors changed after contextual preparation.")
        if prepared.scene_identity_snapshot != (
            prepared.scene.scene_key, prepared.scene.normalizer_key, prepared.scene.coordinate_frame_key
        ):
            raise ValueError("Scene identity changed after contextual preparation.")

    def intervene_frontier(
        self,
        prepared: PreparedContextualOrganization,
        bank: str,
        scene_index: int,
        node_id: int,
        *,
        stop: bool,
    ) -> PreparedContextualOrganization:
        """Rebuild one typed frontier while reusing the exact encoded scene.

        ``stop=True`` selects the node as a coarse live group; ``False`` splits
        it to its current descendant frontier. Explicit counterfactuals are
        also available on J-D for matched functional-sensitivity probes, while
        ordinary J-D preparation remains fixed to leaves. Every other reached
        decision is replayed with the same keyed policy identity. The contextual
        encoder and candidate forest are reused, while the affected typed C rows
        and access directory are rebuilt from original records.
        """
        self._assert_prepared_state(prepared)
        if bank not in {"module", "environment"}:
            raise ValueError("bank must be 'module' or 'environment'.")
        if isinstance(scene_index, bool) or not 0 <= int(scene_index) < prepared.scene.batch_size:
            raise ValueError("scene_index is outside the prepared batch.")
        scene_index = int(scene_index)
        current = (prepared.module_banks if bank == "module" else prepared.environment_banks)[scene_index]
        if current.forest is None:
            raise ValueError("Cannot intervene on an empty typed bank.")
        if node_id not in current.forest.internal_indices:
            raise ValueError("node_id must name an internal candidate in this scene forest.")
        scene = prepared.scene
        case_key = self._scene_case_key(scene.scene_key, scene_index, scene.batch_size)
        if bank == "module":
            states = prepared.module_states[scene_index]
            keys = prepared.module_keys[scene_index]
            coordinates = scene.module_coordinates[scene_index]
            lengths = scene.module_characteristic_lengths[scene_index]
            measures = scene.module_measure[scene_index]
            ids = scene.module_ids[scene_index]
            present = scene.module_mask[scene_index]
            reference = scene.module_measure_reference
            if reference is None:
                reference = scene.module_measure.new_ones(scene.batch_size)
            radius = scene.local_radius[scene_index]
        else:
            states = prepared.environment_states[scene_index]
            keys = prepared.environment_keys[scene_index]
            coordinates = scene.environment_coordinates[scene_index]
            lengths = scene.environment_characteristic_lengths[scene_index]
            measures = scene.environment_measure[scene_index]
            ids = scene.environment_ids[scene_index]
            present = scene.environment_mask[scene_index]
            reference = scene.environment_measure_reference
            if reference is None:
                reference = scene.environment_measure.new_ones(scene.batch_size)
            radius = None
        new_bank, bank_counts = self._prepare_bank(
            bank=bank,
            states=states,
            keys=keys,
            coordinates=coordinates,
            lengths=lengths,
            measures=measures,
            ids=ids,
            present=present,
            scene_rows_all=torch.arange(present.shape[0], device=present.device),
            information_reference=reference[scene_index],
            local_radius=radius,
            case_key=case_key,
            frame_scale=prepared.frame_scale[scene_index],
            forest_override=current.forest,
            forced_subtree=(int(node_id), "coarse" if stop else "split"),
        )
        module_banks = list(prepared.module_banks)
        environment_banks = list(prepared.environment_banks)
        (module_banks if bank == "module" else environment_banks)[scene_index] = new_bank
        counts = dict(prepared.work_counts)
        counts.update(bank_counts)
        counts["frontier_interventions"] = counts.get("frontier_interventions", 0) + 1
        return replace(
            prepared,
            module_banks=tuple(module_banks),
            environment_banks=tuple(environment_banks),
            work_counts=counts,
        )

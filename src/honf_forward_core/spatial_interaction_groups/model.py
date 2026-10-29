"""Spatial group functions shared by interaction relay and native queries."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from ..content_hypergraph.contracts import QueryBatch, SceneBatch
from ..content_hypergraph.spatial_features_v2 import coordinate_gamma
from ..contextual_organization.model import (
    ContextualJointOutput,
    _mlp,
    _scene_tensors,
    _stamp_tensor,
)
from ..interaction_groups.model import Exchange, nullable
from .frontier import (
    BalancedGridTemplate,
    BalancedGridTree,
    build_balanced_grid_template,
    geometry_key,
)


_SCHEMA = "spatial_interaction_group_fields_v1"
_RIDGE = 1.0e-6
_C1_EXTENSION_LIMIT = 2.0


def extend_spatial_coordinate(value: Tensor) -> Tensor:
    """C1-bounded linear coordinate used for group-function extrapolation."""
    magnitude = value.abs()
    outside = value.sign() * (_C1_EXTENSION_LIMIT - torch.exp(1.0 - magnitude))
    return torch.where(magnitude <= 1.0, value, outside)


@dataclass
class SpatialFrontier:
    """Selected cut plus its geometry-only candidate tree."""

    nodes: tuple[int, ...]
    children: tuple[tuple[int, int] | None, ...]
    labels: Tensor
    utilities: Tensor
    reached: tuple[int, ...]
    depths: tuple[int, ...]
    sizes: tuple[int, ...]
    edge_count: int
    policy_rows: int
    tree: BalancedGridTree
    selection_mode: str
    exploration_depth: int | None

    @property
    def active_nodes(self) -> tuple[int, ...]:
        return self.nodes

    @property
    def K(self) -> int:
        return len(self.nodes)

    @property
    def candidate_nodes(self) -> tuple[int, ...]:
        return self.tree.internal_nodes

    @property
    def max_depth(self) -> int:
        return self.tree.max_depth

    @property
    def exploration_selected(self) -> bool:
        return self.selection_mode == "scale_exploration"

    def cover_at_depth(self, depth: int) -> tuple[int, ...]:
        return self.tree.cover_at_depth(depth)

    def cover_nearest_k(self, target: int) -> tuple[int, ...]:
        return self.tree.cover_nearest_k(target)


@dataclass
class _NodeStatistics:
    mass: Tensor
    centers: Tensor
    scales: Tensor
    means: Tensor
    slopes: Tensor
    latent_constant_residual: Tensor
    latent_residual: Tensor
    relation_mean: Tensor
    relation_scatter: Tensor
    relation_magnitude: Tensor
    positive_count: Tensor
    tree: BalancedGridTree


@dataclass
class PreparedSpatialGroupField:
    """One stage's selected groups and all-node organizer statistics."""

    frontier: SpatialFrontier
    hierarchy: BalancedGridTree
    group_nodes: tuple[int, ...]
    live_group_count: int
    labels: Tensor
    group_means: Tensor
    group_slopes: Tensor
    latent_constant_residual: Tensor
    latent_residual: Tensor
    mass: Tensor
    centers: Tensor
    scales: Tensor
    keys: Tensor
    value_coefficients: Tensor
    relation_mean: Tensor
    module_states: Tensor
    environment_states: Tensor
    all_node_means: Tensor
    all_node_slopes: Tensor
    all_node_constant_residual: Tensor
    all_node_residual: Tensor
    all_node_mass: Tensor
    all_node_centers: Tensor
    all_node_scales: Tensor
    all_node_relation_mean: Tensor
    all_node_relation_scatter: Tensor
    all_node_relation_magnitude: Tensor
    utility_candidate_nodes: tuple[int, ...]
    utility_features: Tensor
    work_counts: dict[str, int]

    @property
    def cut(self) -> tuple[int, ...]:
        return self.group_nodes

    @property
    def nodes(self) -> tuple[int, ...]:
        return self.group_nodes

    @property
    def depths(self) -> tuple[int, ...]:
        return tuple(self.frontier.depths[node] for node in self.group_nodes)

    @property
    def levels(self) -> tuple[int, ...]:
        return self.depths


@dataclass
class PreparedSpatialInteractionGroups:
    scene: SceneBatch
    layers: tuple[tuple[PreparedSpatialGroupField, ...], ...]
    module_states: Tensor
    environment_states: Tensor
    frame_origin: Tensor
    frame_scale: Tensor
    model_owner: int
    parameter_versions: tuple
    buffer_versions: tuple
    scene_versions: tuple
    frontier_context: tuple[int, int, int, int]
    zero_slopes: bool
    work_counts: dict[str, int]
    relay_trace: tuple[dict[str, Tensor], ...] | None = None

    @property
    def final_groups(self) -> tuple[PreparedSpatialGroupField, ...]:
        return self.layers[-1]

    @property
    def scene_key(self) -> Any:
        return self.scene.scene_key


class _InteractionStage(nn.Module):
    """One retained MM/ME/EE stage and the legacy-compatible pointwise map."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.mm = Exchange(width, heads)
        self.me = Exchange(width, heads)
        self.ee = Exchange(width, heads)
        # Keep the established singleton-value map exactly pointwise.
        self.group_transform = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.group_projection = _SpatialGroupProjection(width, heads)


class _SpatialGroupProjection(nn.Module):
    """Stage-specific bias-free projections of group-function coefficients."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.h = heads
        self.d = width
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)

    def coefficients(
        self, means: Tensor, slopes: Tensor, *, num_modes: int
    ) -> tuple[Tensor, Tensor]:
        keys = self.k(means).reshape(-1, self.h, self.d // self.h)
        mean_values = self.v(means).reshape(-1, self.h, self.d // self.h)
        if num_modes == 1:
            values = mean_values[:, None]
        else:
            slope_values = self.v(slopes).reshape(-1, 2, self.h, self.d // self.h)
            values = torch.cat((mean_values[:, None], slope_values), dim=1)
        return keys, values


class _SpatialGroupReader(nn.Module):
    """One coefficient-factorized read, reused by the M relay and queries."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.h = heads
        self.d = width
        self.norm = nn.LayerNorm(width)
        self.q = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.geo = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, heads, bias=False))
        self.ff = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, width),
        )

    def query_projection(self, states: Tensor) -> Tensor:
        return self.q(self.norm(states)).reshape(*states.shape[:-1], self.h, self.d // self.h)

    def read(
        self,
        projected_query: Tensor,
        receiver_coordinates: Tensor,
        group: PreparedSpatialGroupField,
        *,
        prior: Tensor | None = None,
        allowed_receivers: Tensor | None = None,
        zero_slopes: bool = False,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Read all positive-mass groups without materializing function values."""
        count = group.mass.numel()
        heads = self.h
        head_width = self.d // heads
        keys = group.keys
        coefficients = group.value_coefficients
        modes = 1 if zero_slopes else coefficients.shape[1]
        q = projected_query
        delta = receiver_coordinates[:, None, :] - group.centers[None, :, :]
        relative = delta / group.scales[None, :, :].clamp_min(1.0e-12)
        log_extent = group.scales.clamp_min(1.0e-12).log().mean(-1)
        geo_features = torch.cat((relative, log_extent[None, :, None].expand(len(q), -1, -1)), dim=-1)
        geo = self.geo(geo_features).movedim(-1, -3)
        logits = torch.einsum("qhd,khd->hqk", q, keys) / math.sqrt(head_width)
        logits = logits + geo + group.mass.clamp_min(1.0e-30).log()[None, None, :]
        if prior is not None:
            logits = logits + prior
        valid = (group.mass > 0)[None, None, :].expand(heads, len(q), count)
        if allowed_receivers is not None:
            valid = valid & allowed_receivers[None, :, None]
        weights = nullable(logits, valid)

        basis = extend_spatial_coordinate(relative)
        context = torch.einsum("hqk,khd->qhd", weights, coefficients[:, 0])
        if modes == 3:
            for axis in range(2):
                weighted_basis = weights * basis[..., axis][None, :, :]
                context = context + torch.einsum(
                    "hqk,khd->qhd", weighted_basis, coefficients[:, axis + 1]
                )
        message = self.out(context.flatten(-2))
        trace = {"weights": weights, "logits": logits, "basis": basis}
        return message, weights, trace


def _interval_totals(values: Tensor, order: Tensor, intervals: Tensor) -> Tensor:
    ordered = values.index_select(0, order)
    prefix = torch.cat((ordered.new_zeros((1, *ordered.shape[1:])), ordered.cumsum(0)), dim=0)
    starts, ends = intervals.unbind(-1)
    return prefix.index_select(0, ends) - prefix.index_select(0, starts)


def _node_statistics(
    latent: Tensor,
    relation: Tensor,
    coordinates: Tensor,
    mass: Tensor,
    tree: BalancedGridTree,
    frame_origin: Tensor,
    frame_scale: Tensor,
) -> _NodeStatistics:
    """Compute weighted moments for every tree node by prefix interval sums."""
    dtype = torch.float64 if latent.dtype in (torch.float16, torch.bfloat16, torch.float32) else latent.dtype
    z = latent.to(dtype)
    xy = coordinates.to(dtype)
    mu = mass.to(dtype)
    rel = relation.permute(1, 0, 2).to(dtype)
    order = torch.as_tensor(tree.leaf_order, dtype=torch.long, device=z.device)
    intervals = torch.as_tensor(tree.intervals, dtype=torch.long, device=z.device)

    total_mass = _interval_totals(mu, order, intervals)
    positive_count = _interval_totals((mu > 0).to(dtype), order, intervals)
    safe_mass = total_mass.clamp_min(torch.finfo(dtype).tiny)
    sum_xy = _interval_totals(xy * mu[:, None], order, intervals)
    sum_z = _interval_totals(z * mu[:, None], order, intervals)
    sum_z2 = _interval_totals(z.square().sum(-1) * mu, order, intervals)
    xy_outer = xy[:, :, None] * xy[:, None, :]
    sum_xy2 = _interval_totals(xy_outer * mu[:, None, None], order, intervals)
    sum_xyz = _interval_totals(xy[:, :, None] * z[:, None, :] * mu[:, None, None], order, intervals)
    sum_rel = _interval_totals(rel * mu[:, None, None], order, intervals)
    sum_rel2 = _interval_totals(rel.square().mean(dim=(-1, -2)) * mu, order, intervals)

    active = total_mass > 0
    centers = sum_xy / safe_mass[:, None]
    means = sum_z / safe_mass[:, None]
    box_min_raw = torch.as_tensor(tree.box_min, dtype=dtype, device=z.device)
    box_max_raw = torch.as_tensor(tree.box_max, dtype=dtype, device=z.device)
    origin = frame_origin.to(dtype)
    scale = frame_scale.to(dtype)
    box_min = (box_min_raw - origin) / scale
    box_max = (box_max_raw - origin) / scale
    fallback_centers = (box_min + box_max) * 0.5
    centers = torch.where(active[:, None], centers, fallback_centers)

    covariance = sum_xy2 / safe_mass[:, None, None] - centers[:, :, None] * centers[:, None, :]
    cross = sum_xyz / safe_mass[:, None, None] - centers[:, :, None] * means[:, None, :]
    scales = torch.maximum((centers - box_min).abs(), (box_max - centers).abs())
    grid_step = torch.as_tensor(tree.grid_step, dtype=dtype, device=z.device) / scale
    scales = torch.maximum(scales, grid_step[None] * 0.5)
    covariance = covariance / (scales[:, :, None] * scales[:, None, :])
    cross = cross / scales[:, :, None]

    axis_present = covariance.diagonal(dim1=-2, dim2=-1) > 1.0e-14
    covariance = covariance * axis_present[:, :, None] * axis_present[:, None, :]
    cross = torch.where(axis_present[:, :, None], cross, torch.zeros_like(cross))
    identity = torch.eye(2, dtype=dtype, device=z.device)
    slopes = torch.linalg.solve_ex(
        covariance + _RIDGE * identity,
        cross,
        check_errors=False,
    ).result
    enough_support = (positive_count > 1) & active
    slopes = torch.where(
        enough_support[:, None, None] & axis_present[:, :, None], slopes, 0.0
    )

    variance_z = sum_z2 / safe_mass - means.square().sum(-1)
    constant_residual = variance_z.clamp_min(0.0)
    constant_residual = torch.where(active, constant_residual, 0.0)
    covariance_slopes = torch.einsum("nij,njd->nid", covariance, slopes)
    residual = variance_z - 2.0 * (cross * slopes).sum(dim=(-1, -2))
    residual = residual + (slopes * covariance_slopes).sum(dim=(-1, -2))
    residual = residual.clamp_min(0.0)
    residual = torch.where(active, residual, 0.0)

    relation_mean = sum_rel / safe_mass[:, None, None]
    relation_magnitude = relation_mean.square().mean(dim=(-1, -2))
    relation_scatter = sum_rel2 / safe_mass - relation_magnitude
    relation_scatter = relation_scatter.clamp_min(0.0)
    relation_mean = torch.where(active[:, None, None], relation_mean, 0.0)
    relation_magnitude = torch.where(active, relation_magnitude, 0.0)
    relation_scatter = torch.where(active, relation_scatter, 0.0)

    out_dtype = latent.dtype
    return _NodeStatistics(
        mass=total_mass.to(out_dtype),
        centers=centers.to(out_dtype),
        scales=scales.to(out_dtype),
        means=torch.where(active[:, None], means, 0.0).to(out_dtype),
        slopes=slopes.to(out_dtype),
        latent_constant_residual=constant_residual.to(out_dtype),
        latent_residual=residual.to(out_dtype),
        relation_mean=relation_mean.to(out_dtype),
        relation_scatter=relation_scatter.to(out_dtype),
        relation_magnitude=relation_magnitude.to(out_dtype),
        positive_count=positive_count,
        tree=tree,
    )


def _utility_feature_rows(
    stats: _NodeStatistics,
    *,
    relation_features: bool,
) -> tuple[tuple[int, ...], Tensor]:
    tree = stats.tree
    nodes = tree.internal_nodes
    device = stats.means.device
    node_ids = torch.as_tensor(nodes, dtype=torch.long, device=device)
    children = torch.as_tensor(
        [tree.children[node] for node in nodes], dtype=torch.long, device=device
    )
    left, right = children.unbind(-1)
    means = stats.means
    slopes = stats.slopes
    relation_mean = stats.relation_mean
    residual = stats.latent_residual
    mass = stats.mass
    sizes = torch.as_tensor(tree.sizes, dtype=means.dtype, device=device)
    scale_features = stats.scales[node_ids].clamp_min(1.0e-12).log()
    log_mass = mass[node_ids].clamp_min(1.0e-30).log()[:, None]
    log_size = (sizes[node_ids] / tree.leaf_count).clamp_min(1.0e-12).log()[:, None]
    child_mass = mass[left] + mass[right]
    left_fraction = (mass[left] / child_mass.clamp_min(1.0e-30))[:, None]
    right_fraction = (mass[right] / child_mass.clamp_min(1.0e-30))[:, None]
    relation_magnitude = stats.relation_magnitude[node_ids, None]
    relation_scatter = stats.relation_scatter[node_ids, None]
    relation_difference = (relation_mean[left] - relation_mean[right]).square().mean(dim=(-1, -2))[:, None]
    if not relation_features:
        relation_magnitude = torch.zeros_like(relation_magnitude)
        relation_scatter = torch.zeros_like(relation_scatter)
        relation_difference = torch.zeros_like(relation_difference)

    features = torch.cat(
        (
            means[node_ids],
            slopes[node_ids].flatten(1),
            means[right] - means[left],
            (slopes[right] - slopes[left]).flatten(1),
            residual[node_ids, None],
            residual[left, None],
            residual[right, None],
            scale_features,
            log_mass,
            log_size,
            left_fraction,
            right_fraction,
            relation_magnitude,
            relation_scatter,
            relation_difference,
        ),
        dim=-1,
    )
    return nodes, features


def _case_key_number(scene_key: Any, batch_index: int) -> int:
    if isinstance(scene_key, (tuple, list)):
        value = scene_key[batch_index]
    else:
        value = scene_key
    text = str(value)
    return sum((position + 1) * ord(character) for position, character in enumerate(text)) & 0xFFFFFFFF


def _seed_values(seed: int, epoch: int, update: int, case: int) -> list[int]:
    return [int(value) & 0xFFFFFFFF for value in (seed, epoch, update, case)]


class SpatialInteractionGroupCore(nn.Module):
    """Two-stage shared-cut core with first-order spatial group functions."""

    schema = _SCHEMA

    def __init__(
        self,
        module_input_dim: int = 9,
        environment_input_dim: int = 6,
        coordinate_dim: int = 2,
        query_input_dim: int = 0,
        operating_input_dim: int = 12,
        output_dim: int = 4,
        architecture: str = "F13-1",
        context_dim: int = 128,
        num_heads: int = 4,
        context_depth: int = 2,
        local_hidden_dim: int = 128,
        dropout: float = 0.0,
        init_seed: int = 0,
        query_tile_size: int = 512,
        environment_nx: int = 24,
        environment_ny: int = 8,
        ridge_epsilon: float = _RIDGE,
    ) -> None:
        super().__init__()
        architecture = str(architecture).upper()
        if architecture not in {"F13-0", "F13-1"}:
            raise ValueError("architecture must be 'F13-0' or 'F13-1'.")
        if coordinate_dim != 2 or context_depth != 2 or dropout != 0:
            raise ValueError("Round13 requires two stages, 2-D coordinates, and dropout=0.")
        if context_dim % num_heads:
            raise ValueError("context_dim must be divisible by num_heads.")
        if local_hidden_dim != context_dim:
            raise ValueError("local_hidden_dim must equal context_dim for the shared field trunk.")
        if min(module_input_dim, environment_input_dim, context_dim, num_heads, output_dim) < 1:
            raise ValueError("Model dimensions must be positive.")
        if query_tile_size < 1 or environment_nx < 1 or environment_ny < 1:
            raise ValueError("Query tile and grid dimensions must be positive.")
        if ridge_epsilon != _RIDGE:
            raise ValueError("Round13 fixes dimensionless moment regularization at 1e-6.")
        self.config = {
            "module_input_dim": module_input_dim,
            "environment_input_dim": environment_input_dim,
            "coordinate_dim": coordinate_dim,
            "query_input_dim": query_input_dim,
            "operating_input_dim": operating_input_dim,
            "output_dim": output_dim,
            "architecture": architecture,
            "context_dim": context_dim,
            "num_heads": num_heads,
            "context_depth": context_depth,
            "local_hidden_dim": local_hidden_dim,
            "dropout": dropout,
            "init_seed": init_seed,
            "query_tile_size": query_tile_size,
            "environment_nx": environment_nx,
            "environment_ny": environment_ny,
            "ridge_epsilon": ridge_epsilon,
        }
        self.__dict__.update(self.config)
        feature_dim = 6 * context_dim + 12
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            self.module_encoder = _mlp(module_input_dim + 16, context_dim, context_dim)
            self.environment_encoder = _mlp(environment_input_dim + 16, context_dim, context_dim)
            self.operating_encoder = (
                _mlp(operating_input_dim, context_dim, context_dim) if operating_input_dim else None
            )
            self.layers = nn.ModuleList(
                [_InteractionStage(context_dim, num_heads) for _ in range(context_depth)]
            )
            self.group_reader = _SpatialGroupReader(context_dim, num_heads)
            self.query_encoder = _mlp(query_input_dim + 14, context_dim, context_dim)
            self.fusion_trunk = _mlp(2 * context_dim, context_dim, context_dim)
            self.field_head = nn.Linear(context_dim, output_dim)
            utility_width = max(32, min(128, context_dim))
            self.utility_head = nn.Sequential(
                nn.Linear(feature_dim, utility_width),
                nn.SiLU(),
                nn.Linear(utility_width, 1),
            )
            nn.init.zeros_(self.utility_head[-1].bias)
        self.register_buffer("organization_epoch", torch.tensor(1, dtype=torch.long))
        self.register_buffer("organization_update", torch.tensor(0, dtype=torch.long))
        self.register_buffer("organization_seed", torch.tensor(int(init_seed), dtype=torch.long))
        self.register_buffer("exploration_override", torch.tensor(-1, dtype=torch.long))
        self._frontier_context = (1, 0, int(init_seed), -1)
        self._geometry_templates: OrderedDict[tuple, BalancedGridTemplate] = OrderedDict()
        self._utility_feature_dim = feature_dim

    @property
    def num_modes(self) -> int:
        return 1 if self.architecture == "F13-0" else 3

    @property
    def utility_feature_dim(self) -> int:
        return self._utility_feature_dim

    def get_config(self) -> dict[str, Any]:
        return {"schema": self.schema, **self.config}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SpatialInteractionGroupCore":
        values = dict(config)
        if values.pop("schema", None) != cls.schema:
            raise ValueError("Wrong spatial interaction-group schema.")
        return cls(**values)

    def set_frontier_context(
        self,
        epoch: int,
        update: int,
        seed: int,
        *,
        exploration: bool | None = None,
    ) -> None:
        """Set the deterministic key for exploration without advancing RNG state."""
        if min(int(epoch), int(update)) < 0:
            raise ValueError("Frontier epoch and update must be nonnegative.")
        flag = -1 if exploration is None else int(bool(exploration))
        self.organization_epoch.fill_(int(epoch))
        self.organization_update.fill_(int(update))
        self.organization_seed.fill_(int(seed))
        self.exploration_override.fill_(flag)
        self._frontier_context = (int(epoch), int(update), int(seed), flag)

    def set_organization_clock(self, epoch: int, update: int = 0) -> None:
        self.set_frontier_context(epoch, update, self.init_seed)

    def _parameter_version_stamp(self) -> tuple:
        return tuple((id(parameter), parameter._version) for parameter in self.parameters())

    def _buffer_version_stamp(self) -> tuple:
        return tuple(_stamp_tensor(value) for value in self.buffers())

    def _tree_for_coordinates(self, coordinates: np.ndarray) -> BalancedGridTree:
        key = geometry_key(coordinates)
        template = self._geometry_templates.get(key)
        if template is None:
            template = build_balanced_grid_template(coordinates, self.environment_nx, self.environment_ny)
            self._geometry_templates[key] = template
            if len(self._geometry_templates) > 32:
                self._geometry_templates.popitem(last=False)
        else:
            self._geometry_templates.move_to_end(key)
        return template.bind(coordinates)

    def _run_stage(
        self,
        stage: _InteractionStage,
        m: Tensor,
        e: Tensor,
        scene: SceneBatch,
        xy: Tensor,
        exy: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        present = scene.module_mask
        allowed_mm = present[:, :, None] & present[:, None, :]
        allowed_mm = allowed_mm & ~torch.eye(m.shape[1], device=m.device, dtype=torch.bool)[None]
        mm, _ = stage.mm(
            m,
            m,
            xy,
            xy,
            scene.module_measure / scene.module_measure_reference[:, None],
            allowed_mm,
        )
        m = torch.where((present.sum(-1) > 1)[:, None, None], mm, m) * present[..., None]
        e, relation = stage.me(
            e,
            m,
            exy,
            xy,
            scene.module_measure / scene.module_measure_reference[:, None],
            scene.environment_mask[:, :, None] & present[:, None, :],
        )
        e, _ = stage.ee(
            e,
            e,
            exy,
            exy,
            scene.environment_measure / scene.environment_measure_reference[:, None],
            scene.environment_mask[:, :, None] & scene.environment_mask[:, None, :],
        )
        e = e * scene.environment_mask[..., None]
        return m, e, relation

    def _features(
        self,
        stats: _NodeStatistics,
        *,
        relation_features: bool,
    ) -> tuple[tuple[int, ...], Tensor]:
        return _utility_feature_rows(stats, relation_features=relation_features)

    def predict_utility(self, features: Tensor) -> Tensor:
        """Return signed parent-minus-descendant normalized loss advantages."""
        return 2.0 * torch.tanh(self.utility_head(features).squeeze(-1))

    def utility_candidates(
        self,
        prepared: PreparedSpatialInteractionGroups,
        *,
        batch_index: int = 0,
    ) -> tuple[int, ...]:
        self._assert_prepared(prepared, None)
        return prepared.layers[0][batch_index].utility_candidate_nodes

    def utility_candidate_metadata(
        self,
        prepared: PreparedSpatialInteractionGroups,
        *,
        batch_index: int = 0,
    ) -> tuple[dict[str, int], ...]:
        self._assert_prepared(prepared, None)
        group = prepared.layers[0][batch_index]
        tree = group.hierarchy
        return tuple(
            {"node_id": node, "depth": tree.depths[node], "size": tree.sizes[node]}
            for node in group.utility_candidate_nodes
        )

    def utility_candidate_metadata_for_scene(
        self,
        scene: SceneBatch,
        *,
        batch_index: int = 0,
    ) -> tuple[dict[str, int], ...]:
        """Return stable geometry-only candidates without running the model."""
        coordinates = scene.environment_coordinates
        if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
            raise ValueError("Scene environment coordinates must have shape [B,E,2].")
        if batch_index < 0 or batch_index >= coordinates.shape[0]:
            raise IndexError("batch_index is outside the scene batch.")
        raw = coordinates[batch_index].detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
        tree = self._tree_for_coordinates(raw)
        return tuple(
            {"node_id": node, "depth": tree.depths[node], "size": tree.sizes[node]}
            for node in tree.internal_nodes
        )

    def utility_features(
        self,
        prepared: PreparedSpatialInteractionGroups,
        node: int,
        *,
        batch_index: int = 0,
        detach: bool = False,
    ) -> Tensor:
        self._assert_prepared(prepared, None)
        group = prepared.layers[0][batch_index]
        try:
            index = group.utility_candidate_nodes.index(int(node))
        except ValueError as exc:
            raise ValueError("Utility features require an internal hierarchy node.") from exc
        result = group.utility_features[index]
        return result.detach() if detach else result

    def _select_frontier(
        self,
        tree: BalancedGridTree,
        utility_nodes: tuple[int, ...],
        utilities: Sequence[float],
        *,
        batch_index: int,
        scene_key: Any,
        forced: Any,
        explore: bool,
        context: tuple[int, int, int, int],
    ) -> tuple[tuple[int, ...], tuple[int, ...], int | None]:
        split = [False] * tree.node_count
        for node, value in zip(utility_nodes, utilities):
            split[node] = float(value) > 0.0
        active: tuple[int, ...]
        exploration_depth: int | None = None
        if explore and forced is None:
            epoch, update, seed, _ = context
            case = _case_key_number(scene_key, batch_index)
            sequence = np.random.SeedSequence(_seed_values(seed, epoch, update, case))
            generator = np.random.default_rng(sequence)
            exposure_bucket = int(sequence.generate_state(1, dtype=np.uint32)[0]) % 4
            if exposure_bucket == 0:
                depth = int(generator.integers(tree.max_depth + 1))
                active = tree.cover_at_depth(depth)
                exploration_depth = depth
            else:
                active = tree.cut_from_decisions(split, forced)
        else:
            active = tree.cut_from_decisions(split, forced)
        active_set = set(active)
        reached: list[int] = []
        stack = [tree.root]
        while stack:
            node = stack.pop()
            children = tree.children[node]
            if children is None or node in active_set:
                continue
            reached.append(node)
            stack.extend(reversed(children))
        return active, tuple(reached), exploration_depth

    def _make_group_field(
        self,
        stats: _NodeStatistics,
        frontier: SpatialFrontier,
        module_states: Tensor,
        environment_states: Tensor,
        projection: _SpatialGroupProjection,
        *,
        zero_slopes: bool,
        retain_environment_states: bool,
        live_group_count: int,
    ) -> PreparedSpatialGroupField:
        selected = torch.as_tensor(frontier.nodes, dtype=torch.long, device=stats.means.device)
        means = stats.means.index_select(0, selected)
        slopes = stats.slopes.index_select(0, selected)
        mass = stats.mass.index_select(0, selected)
        centers = stats.centers.index_select(0, selected)
        scales = stats.scales.index_select(0, selected)
        relation_mean = stats.relation_mean.index_select(0, selected)
        value_slopes = torch.zeros_like(slopes) if zero_slopes else slopes
        keys, values = projection.coefficients(means, value_slopes, num_modes=self.num_modes)
        labels = torch.as_tensor(frontier.labels, dtype=torch.long, device=means.device)
        return PreparedSpatialGroupField(
            frontier=frontier,
            hierarchy=frontier.tree,
            group_nodes=frontier.nodes,
            live_group_count=int(live_group_count),
            labels=labels,
            group_means=means,
            group_slopes=slopes,
            latent_constant_residual=stats.latent_constant_residual.index_select(0, selected),
            latent_residual=stats.latent_residual.index_select(0, selected),
            mass=mass,
            centers=centers,
            scales=scales,
            keys=keys,
            value_coefficients=values,
            relation_mean=relation_mean,
            module_states=module_states,
            environment_states=(
                environment_states
                if retain_environment_states
                else environment_states.new_empty((0, environment_states.shape[-1]))
            ),
            all_node_means=stats.means,
            all_node_slopes=stats.slopes,
            all_node_constant_residual=stats.latent_constant_residual,
            all_node_residual=stats.latent_residual,
            all_node_mass=stats.mass,
            all_node_centers=stats.centers,
            all_node_scales=stats.scales,
            all_node_relation_mean=stats.relation_mean,
            all_node_relation_scatter=stats.relation_scatter,
            all_node_relation_magnitude=stats.relation_magnitude,
            utility_candidate_nodes=frontier.tree.internal_nodes,
            utility_features=frontier.utilities.new_empty((0, self.utility_feature_dim)),
            work_counts={},
        )

    def prepare(
        self,
        scene: SceneBatch,
        *,
        clamp_layer: int | None = None,
        forced_frontier: Mapping[int, Any] | Any | None = None,
        query_chunk_size: int | None = None,
        zero_slopes: bool = False,
        relation_features: bool = True,
        exploration: bool | None = None,
        return_trace: bool = False,
    ) -> PreparedSpatialInteractionGroups:
        """Prepare both stages; one stage0 cut is reused through stage1."""
        if query_chunk_size is not None and int(query_chunk_size) < 1:
            raise ValueError("query_chunk_size must be positive when supplied.")
        if clamp_layer not in (None, 0):
            raise ValueError("Round13 has only the stage0 group-to-M relay (clamp_layer=0).")
        scene.validate(
            coordinate_dim=2,
            module_input_dim=self.module_input_dim,
            environment_input_dim=self.environment_input_dim,
            operating_input_dim=self.operating_input_dim,
        )
        live_counts = torch.stack(
            (
                scene.module_mask.sum(dim=-1, dtype=torch.long),
                scene.environment_mask.sum(dim=-1, dtype=torch.long),
            ),
            dim=-1,
        ).detach().to(device="cpu").tolist()
        module_live_counts = [int(row[0]) for row in live_counts]
        environment_live_counts = [int(row[1]) for row in live_counts]
        origin = scene.coordinate_origin
        frame_scale = scene.coordinate_scale
        xy = (scene.module_coordinates - origin[:, None]) / frame_scale[:, None]
        exy = (scene.environment_coordinates - origin[:, None]) / frame_scale[:, None]
        module_input = torch.cat(
            (
                scene.module_features,
                coordinate_gamma(scene.module_coordinates, origin, frame_scale),
                scene.module_characteristic_lengths / frame_scale[:, None],
            ),
            dim=-1,
        )
        environment_input = torch.cat(
            (
                scene.environment_features,
                coordinate_gamma(scene.environment_coordinates, origin, frame_scale),
                scene.environment_characteristic_lengths / frame_scale[:, None],
            ),
            dim=-1,
        )
        m = self.module_encoder(module_input)
        e = self.environment_encoder(environment_input)
        if self.operating_encoder is not None:
            operating = self.operating_encoder(scene.operating_features)
            m = m + operating[:, None]
            e = e + operating[:, None]
        m = m * scene.module_mask[..., None]
        e = e * scene.environment_mask[..., None]

        work: dict[str, int] = {
            "module_embedding_rows": int(scene.module_mask.numel()),
            "module_embedding_live_rows": sum(module_live_counts),
            "environment_embedding_rows": int(scene.environment_mask.numel()),
            "environment_embedding_live_rows": sum(environment_live_counts),
            "mask_count_host_rows": 2 * scene.batch_size
            if scene.environment_coordinates.device.type != "cpu"
            else 0,
            "geometry_host_coordinate_rows": 0,
            "frontier_host_utility_rows": 0,
            "live_group_count_host_rows": 0,
            "cut_label_device_rows": 0,
            "group_index_device_rows": 0,
            "hierarchy_metadata_device_rows": 0,
        }
        m, e, relation0 = self._run_stage(self.layers[0], m, e, scene, xy, exy)
        latent0 = self.layers[0].group_transform(e)
        mass = (
            scene.environment_measure / scene.environment_measure_reference[:, None]
        ) * scene.environment_mask.to(scene.environment_measure.dtype)

        override = forced_frontier
        if isinstance(forced_frontier, Mapping) and not any(
            key in forced_frontier for key in ("cover", "depth")
        ):
            indices = set(forced_frontier)
            if indices - {0, 1}:
                raise ValueError("Only the shared stage0/stage1 frontier may be overridden.")
            if 0 in forced_frontier and 1 in forced_frontier and forced_frontier[0] != forced_frontier[1]:
                raise ValueError("Both stages share one frontier; conflicting overrides were supplied.")
            override = forced_frontier.get(0, forced_frontier.get(1))
        if isinstance(forced_frontier, Mapping) and any(
            key in forced_frontier for key in ("cover", "depth")
        ):
            override = forced_frontier

        context = self._frontier_context
        use_exploration = (
            bool(self.training and context[3] != 0)
            if exploration is None
            else bool(exploration and self.training)
        )
        stats_by_case: list[_NodeStatistics] = []
        trees: list[BalancedGridTree] = []
        utility_rows: list[Tensor] = []
        candidate_nodes: tuple[int, ...] | None = None
        raw_geometry = scene.environment_coordinates.detach().to(
            device="cpu", dtype=torch.float64
        ).contiguous().numpy()
        if scene.environment_coordinates.device.type != "cpu":
            work["geometry_host_coordinate_rows"] = int(
                scene.batch_size * scene.num_environment_tokens
            )
        for b in range(scene.batch_size):
            tree = self._tree_for_coordinates(raw_geometry[b])
            if candidate_nodes is None:
                candidate_nodes = tree.internal_nodes
            elif candidate_nodes != tree.internal_nodes:
                raise ValueError("Batch scenes have incompatible canonical tree topology.")
            stats = _node_statistics(
                latent0[b],
                relation0[b],
                exy[b],
                mass[b],
                tree,
                origin[b],
                frame_scale[b],
            )
            nodes, features = self._features(stats, relation_features=relation_features)
            if nodes != candidate_nodes:
                raise RuntimeError("Utility node ordering differs from its hierarchy.")
            stats_by_case.append(stats)
            trees.append(tree)
            utility_rows.append(features)
        utility_features = torch.stack(utility_rows)
        utilities = self.predict_utility(utility_features)
        live_node_matrix = torch.stack(
            [stats.positive_count > 0 for stats in stats_by_case], dim=0
        ).to(dtype=utilities.dtype)
        frontier_metadata = torch.cat((utilities.detach(), live_node_matrix), dim=1)
        metadata_cpu = frontier_metadata.to(device="cpu").tolist()
        candidate_count = utilities.shape[1]
        utility_cpu = [row[:candidate_count] for row in metadata_cpu]
        live_node_rows = [row[candidate_count:] for row in metadata_cpu]
        if utilities.device.type != "cpu":
            work["frontier_host_utility_rows"] = int(utilities.numel())
            work["live_group_count_host_rows"] = int(live_node_matrix.numel())

        stage0_groups: list[PreparedSpatialGroupField] = []
        frontiers: list[SpatialFrontier] = []
        relay_states: list[Tensor] = []
        relay_traces: list[dict[str, Tensor]] = []
        for b, (tree, stats) in enumerate(zip(trees, stats_by_case)):
            case_override = override
            if isinstance(override, Mapping) and not any(key in override for key in ("cover", "depth")):
                # A batch-indexed override is also accepted for evaluator panels.
                if b in override and isinstance(override[b], (Mapping, tuple, list)):
                    case_override = override[b]
            active, reached, exploration_depth = self._select_frontier(
                tree,
                candidate_nodes or (),
                utility_cpu[b],
                batch_index=b,
                scene_key=scene.scene_key,
                forced=case_override,
                explore=use_exploration,
                context=context,
            )
            labels = tree.labels_for_cut(active)
            if latent0.device.type != "cpu":
                work["cut_label_device_rows"] += len(labels)
            if case_override is not None:
                selection_mode = "fixed_cover" if isinstance(case_override, Mapping) else "forced"
            elif exploration_depth is not None:
                selection_mode = "scale_exploration"
            else:
                selection_mode = "learned"
            frontier = SpatialFrontier(
                nodes=active,
                children=tree.children,
                labels=torch.as_tensor(labels, dtype=torch.long, device=latent0.device),
                utilities=utilities[b],
                reached=reached,
                depths=tree.depths,
                sizes=tree.sizes,
                edge_count=tree.node_count - tree.leaf_count,
                policy_rows=len(tree.internal_nodes),
                tree=tree,
                selection_mode=selection_mode,
                exploration_depth=exploration_depth,
            )
            group = self._make_group_field(
                stats,
                frontier,
                m[b],
                e[b],
                self.layers[0].group_projection,
                zero_slopes=zero_slopes,
                retain_environment_states=return_trace,
                live_group_count=sum(bool(live_node_rows[b][node]) for node in active),
            )
            group.utility_features = utility_features[b]
            group.work_counts = {
                "balanced_hierarchy_nodes": tree.node_count,
                "frontier_internal_candidates": len(tree.internal_nodes),
                "stage0_original_e_rows": len(e[b]),
                "stage0_pointwise_value_transform_rows": len(e[b]),
                "stage0_all_node_moment_rows": tree.node_count,
                "stage0_group_key_projection_rows": len(active),
                "stage0_group_value_projection_rows": len(active) * self.num_modes,
                "stage0_selected_group_count": len(active),
                "stage0_group_member_rows": len(e[b]),
            }
            frontiers.append(frontier)
            stage0_groups.append(group)

            if clamp_layer == 0:
                message = torch.zeros_like(m[b])
                relay_weights = message.new_zeros((self.num_heads, m.shape[1], len(active)))
                relay_trace = {
                    "weights": relay_weights,
                    "group_nodes": torch.as_tensor(active, dtype=torch.long, device=m.device),
                }
            else:
                projected = self.group_reader.query_projection(m[b])
                prior = group.relation_mean.permute(1, 2, 0)
                message, relay_weights, relay_trace = self.group_reader.read(
                    projected,
                    xy[b],
                    group,
                    prior=prior,
                    allowed_receivers=scene.module_mask[b],
                    zero_slopes=zero_slopes,
                )
            m_relay = m[b] + message
            m_relay = (m_relay + self.group_reader.ff(m_relay)) * scene.module_mask[b, :, None]
            relay_states.append(m_relay)
            if return_trace:
                relay_traces.append(
                    {
                        **relay_trace,
                        "group_nodes": torch.as_tensor(active, dtype=torch.long, device=m.device),
                    }
                )
            relay_executed = clamp_layer != 0
            group.work_counts["relay_group_scores"] = (
                int(m.shape[1] * len(active) * self.num_heads) if relay_executed else 0
            )
            group.work_counts["relay_group_value_reads"] = (
                int(m.shape[1] * len(active)) if relay_executed else 0
            )
            group.work_counts["relay_group_value_mode_reads"] = (
                int(m.shape[1] * len(active) * self.num_modes) if relay_executed else 0
            )
            group.work_counts["relay_live_receivers"] = module_live_counts[b]
            group.work_counts["relay_relation_prior_entries"] = int(
                self.num_heads * m.shape[1] * len(active) if relay_executed else 0
            )
            group.work_counts["stage0_relation_pool_edges"] = int(
                len(e[b]) * m.shape[1] * self.num_heads
            )
            group.work_counts["stage0_relation_pool_live_edges"] = (
                environment_live_counts[b] * module_live_counts[b] * self.num_heads
            )
            if clamp_layer == 0:
                group.work_counts["relay_bypassed"] = 1
            if not self.training and not return_trace:
                group.keys = group.keys.new_empty((0, self.num_heads, self.context_dim // self.num_heads))
                group.value_coefficients = group.value_coefficients.new_empty(
                    (0, self.num_modes, self.num_heads, self.context_dim // self.num_heads)
                )

        m = torch.stack(relay_states)
        m, e, relation1 = self._run_stage(self.layers[1], m, e, scene, xy, exy)
        latent1 = self.layers[1].group_transform(e)
        stage1_groups: list[PreparedSpatialGroupField] = []
        for b, (tree, stats, frontier) in enumerate(zip(trees, stats_by_case, frontiers)):
            # The second stage recomputes latent coefficients under the exact stage0 cut.
            final_stats = _node_statistics(
                latent1[b],
                relation1[b],
                exy[b],
                mass[b],
                tree,
                origin[b],
                frame_scale[b],
            )
            final_frontier = SpatialFrontier(
                nodes=frontier.nodes,
                children=frontier.children,
                labels=frontier.labels,
                utilities=frontier.utilities,
                reached=frontier.reached,
                depths=frontier.depths,
                sizes=frontier.sizes,
                edge_count=frontier.edge_count,
                policy_rows=frontier.policy_rows,
                tree=tree,
                selection_mode=frontier.selection_mode,
                exploration_depth=frontier.exploration_depth,
            )
            group = self._make_group_field(
                final_stats,
                final_frontier,
                m[b],
                e[b],
                self.layers[1].group_projection,
                zero_slopes=zero_slopes,
                retain_environment_states=return_trace,
                live_group_count=stage0_groups[b].live_group_count,
            )
            group.utility_features = stage0_groups[b].utility_features
            group.work_counts = {
                "stage1_original_e_rows": len(e[b]),
                "stage1_pointwise_value_transform_rows": len(e[b]),
                "stage1_all_node_moment_rows": tree.node_count,
                "stage1_group_key_projection_rows": len(frontier.nodes),
                "stage1_group_value_projection_rows": len(frontier.nodes) * self.num_modes,
                "stage1_selected_group_count": len(frontier.nodes),
                "stage1_group_member_rows": len(e[b]),
                "stage1_relation_pool_edges": len(e[b]) * m.shape[1] * self.num_heads,
                "stage1_relation_pool_live_edges": (
                    environment_live_counts[b] * module_live_counts[b] * self.num_heads
                ),
            }
            stage1_groups.append(group)

        selected_groups = sum(group.frontier.K for group in stage1_groups)
        if scene.environment_coordinates.device.type != "cpu":
            work["hierarchy_metadata_device_rows"] = scene.batch_size * (
                2 * (scene.num_environment_tokens + 3 * trees[0].node_count + 1)
                + 2 * len(trees[0].internal_nodes)
                + trees[0].node_count
                + scene.num_environment_tokens
            )
            work["group_index_device_rows"] = sum(
                group.frontier.K for group in stage0_groups
            ) + selected_groups
        work.update(
            {
                "mm_pairs": 2 * scene.batch_size * m.shape[1] ** 2 * self.num_heads,
                "me_pairs": 2 * scene.batch_size * m.shape[1] * e.shape[1] * self.num_heads,
                "ee_pairs": 2 * scene.batch_size * e.shape[1] ** 2 * self.num_heads,
                "mm_live_pairs": sum(count * (count - 1) for count in module_live_counts)
                * 2
                * self.num_heads,
                "me_live_pairs": sum(
                    module_live_counts[b] * environment_live_counts[b]
                    for b in range(scene.batch_size)
                )
                * 2
                * self.num_heads,
                "ee_live_pairs": sum(count * count for count in environment_live_counts)
                * 2
                * self.num_heads,
                "policy_rows": sum(group.frontier.policy_rows for group in stage0_groups),
                "metadata_nodes": sum(group.frontier.tree.node_count for group in stage0_groups),
                "utility_head_rows": sum(group.frontier.policy_rows for group in stage0_groups),
                "group_relay_pairs": sum(
                    group.work_counts["relay_group_scores"] for group in stage0_groups
                ),
                "group_read_modes_per_selected_group": self.num_modes,
                "stage0_group_value_modes_total": sum(
                    group.frontier.K for group in stage0_groups
                ) * self.num_modes,
                "stage1_group_value_modes_total": selected_groups * self.num_modes,
                "selected_groups_total": selected_groups,
                "original_e_rows": scene.batch_size * e.shape[1],
                "pool_source_rows": scene.batch_size * e.shape[1],
            }
        )
        for groups in (stage0_groups, stage1_groups):
            for group in groups:
                for name, value in group.work_counts.items():
                    work[name] = work.get(name, 0) + int(value)
        return PreparedSpatialInteractionGroups(
            scene=scene,
            layers=(tuple(stage0_groups), tuple(stage1_groups)),
            module_states=m,
            environment_states=(
                e if return_trace else e.new_empty((scene.batch_size, 0, e.shape[-1]))
            ),
            frame_origin=origin,
            frame_scale=frame_scale,
            model_owner=id(self),
            parameter_versions=self._parameter_version_stamp(),
            buffer_versions=self._buffer_version_stamp(),
            scene_versions=tuple(_stamp_tensor(value) for value in _scene_tensors(scene)),
            frontier_context=self._frontier_context,
            zero_slopes=bool(zero_slopes),
            work_counts=work,
            relay_trace=tuple(relay_traces) if return_trace else None,
        )

    def _assert_prepared(self, prepared: PreparedSpatialInteractionGroups, query: QueryBatch | None) -> None:
        if prepared.model_owner != id(self) or prepared.parameter_versions != self._parameter_version_stamp():
            raise ValueError("Prepared spatial groups have a different model owner/state.")
        if prepared.buffer_versions != self._buffer_version_stamp():
            raise ValueError("Spatial frontier context changed after prepare.")
        if prepared.scene_versions != tuple(_stamp_tensor(value) for value in _scene_tensors(prepared.scene)):
            raise ValueError("Scene tensors changed after spatial group preparation.")
        if prepared.frontier_context != self._frontier_context:
            raise ValueError("Frontier context changed after spatial group preparation.")
        if query is not None:
            if query.scene_key is not None and query.scene_key != prepared.scene_key:
                raise ValueError("Query scene differs from prepared scene.")
            if query.coordinate_frame_key is not None and query.coordinate_frame_key != prepared.scene.coordinate_frame_key:
                raise ValueError("Query coordinate frame differs from prepared scene.")
            query.validate(
                batch_size=prepared.scene.batch_size,
                coordinate_dim=2,
                query_input_dim=self.query_input_dim,
            )

    def _read_tile(
        self,
        prepared: PreparedSpatialInteractionGroups,
        batch_index: int,
        q_state: Tensor,
        coordinates: Tensor,
        *,
        zero_slopes: bool,
    ) -> tuple[Tensor, dict[str, int], dict[str, Tensor]]:
        group = prepared.final_groups[batch_index]
        frame_xy = (coordinates - prepared.frame_origin[batch_index]) / prepared.frame_scale[batch_index]
        projected = self.group_reader.query_projection(q_state)
        context, weights, trace = self.group_reader.read(
            projected,
            frame_xy,
            group,
            zero_slopes=zero_slopes,
        )
        fields = self.field_head(self.fusion_trunk(torch.cat((q_state, context), dim=-1)))
        count = len(group.mass)
        read_modes = 1 if zero_slopes else int(group.value_coefficients.shape[1])
        work = {
            "group_scores": len(q_state) * count * self.num_heads,
            "group_value_reads": len(q_state) * count,
            "group_value_mode_reads": len(q_state) * count * read_modes,
            "group_value_modes_per_group": read_modes,
            "group_live_scores": group.live_group_count * len(q_state) * self.num_heads,
            "group_live_value_reads": group.live_group_count * len(q_state),
            "original_e_query_reads": 0,
            "module_value_query_reads": 0,
            "empty_queries": len(q_state) if group.live_group_count == 0 else 0,
            "native_query_rows": len(q_state),
        }
        trace = {
            **trace,
            "group_nodes": torch.as_tensor(group.group_nodes, dtype=torch.long, device=q_state.device),
            "group_centers": group.centers,
            "group_scales": group.scales,
        }
        return fields, work, trace

    def decode(
        self,
        prepared: PreparedSpatialInteractionGroups,
        query: QueryBatch,
        *,
        exhaustive: bool = False,
        return_trace: bool = False,
        candidate_traces: Any = None,
        association_scope: str = "postings",
        zero_slopes: bool | None = None,
    ):
        """Read the final group functions directly at every receiver."""
        if association_scope not in {"postings", "all"}:
            raise ValueError("Unknown legacy association_scope value.")
        if candidate_traces is not None:
            raise ValueError("Spatial group reads do not use candidate traces.")
        del exhaustive  # Every positive-mass group is always read.
        self._assert_prepared(prepared, query)
        use_zero_slopes = prepared.zero_slopes if zero_slopes is None else bool(zero_slopes)
        gamma = coordinate_gamma(query.coordinates, prepared.frame_origin, prepared.frame_scale)
        query_features = (
            torch.cat((query.features, gamma), dim=-1) if self.query_input_dim else gamma
        )
        q = self.query_encoder(query_features)
        outputs: list[Tensor] = []
        trace_rows: list[list[dict[str, Tensor]]] = []
        work: dict[str, int] = {}
        for batch_index in range(len(q)):
            pieces: list[Tensor] = []
            traces: list[dict[str, Tensor]] = []
            for start in range(0, q.shape[1], self.query_tile_size):
                end = min(q.shape[1], start + self.query_tile_size)
                fields, counters, trace = self._read_tile(
                    prepared,
                    batch_index,
                    q[batch_index, start:end],
                    query.coordinates[batch_index, start:end],
                    zero_slopes=use_zero_slopes,
                )
                pieces.append(fields)
                traces.append(trace)
                for name, value in counters.items():
                    if name == "group_value_modes_per_group":
                        value = int(value)
                        previous = work.setdefault(name, value)
                        if previous != value:
                            raise RuntimeError("Group value-mode count changed across decode tiles.")
                    else:
                        work[name] = work.get(name, 0) + int(value)
            if pieces:
                outputs.append(torch.cat(pieces, dim=0))
            else:
                outputs.append(q.new_empty((0, self.output_dim)))
            trace_rows.append(traces)
        fields = torch.stack(outputs)
        if query.query_mask is not None:
            fields = fields * query.query_mask[..., None]
        output = ContextualJointOutput(
            nonlinear_field=fields[..., :3],
            thermal_temperature=fields[..., 3:4] if fields.shape[-1] >= 4 else None,
            work_counts=work,
            joint_fields=fields,
        )
        return (output, trace_rows) if return_trace else output

    def forward(self, scene: SceneBatch, query: QueryBatch) -> ContextualJointOutput:
        return self.decode(self.prepare(scene), query)


def map_compatible_parent_tensors(
    parent_state: Mapping[str, Tensor], target_model: SpatialInteractionGroupCore
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Map shape-compatible singleton-core tensors and report every name.

    This is an initialization mapping only. It does not copy optimizer state or
    assert that a copied tensor has identical semantics after the model change.
    """
    if not isinstance(parent_state, Mapping):
        raise TypeError("parent_state must be a state-dict mapping.")
    source: dict[str, Tensor] = {}
    for raw_name, value in parent_state.items():
        name = str(raw_name)
        if name.startswith("core."):
            name = name[5:]
        if name.startswith("module."):
            name = name[7:]
        source[name] = value
    target = target_model.state_dict()
    mapped: dict[str, Tensor] = {name: value.detach().clone() for name, value in target.items()}
    rename = {
        **{
            f"group_reader.{suffix}": f"layers.0.gm.{suffix}"
            for suffix in ("norm.weight", "norm.bias", "q.weight", "out.weight")
        },
        "group_reader.geo.0.weight": "layers.0.gm.geo.0.weight",
        "group_reader.geo.0.bias": "layers.0.gm.geo.0.bias",
        "group_reader.geo.2.weight": "layers.0.gm.geo.2.weight",
        "group_reader.ff.0.weight": "layers.0.gm.ff.0.weight",
        "group_reader.ff.0.bias": "layers.0.gm.ff.0.bias",
        "group_reader.ff.1.weight": "layers.0.gm.ff.1.weight",
        "group_reader.ff.1.bias": "layers.0.gm.ff.1.bias",
        "group_reader.ff.3.weight": "layers.0.gm.ff.3.weight",
        "group_reader.ff.3.bias": "layers.0.gm.ff.3.bias",
    }
    copied: list[dict[str, str]] = []
    new: list[dict[str, Any]] = []
    used_source: set[str] = set()
    clock_buffers = {
        "organization_epoch",
        "organization_update",
        "organization_seed",
        "exploration_override",
    }
    for target_name, target_value in target.items():
        if target_name in clock_buffers:
            new.append(
                {
                    "target": target_name,
                    "reason": "child control clock reset to its local initialization",
                }
            )
            continue
        projection_source = None
        if target_name.startswith("layers.") and ".group_projection." in target_name:
            stage = target_name.split(".", 2)[1]
            suffix = target_name.split(".group_projection.", 1)[1]
            projection_source = f"layers.{stage}.gm.{suffix}"
        source_name = projection_source or rename.get(target_name, target_name)
        candidate = source.get(source_name)
        if candidate is None:
            new.append({"target": target_name, "reason": "no compatible parent tensor"})
            continue
        if tuple(candidate.shape) != tuple(target_value.shape):
            new.append(
                {
                    "target": target_name,
                    "source": source_name,
                    "reason": f"shape mismatch {tuple(candidate.shape)} != {tuple(target_value.shape)}",
                }
            )
            continue
        mapped[target_name] = candidate.detach().to(device=target_value.device, dtype=target_value.dtype).clone()
        used_source.add(source_name)
        copied.append({"source": source_name, "target": target_name})
    dropped = sorted(set(source) - used_source)
    return mapped, {
        "schema": "spatial_interaction_group_parent_tensor_map_v1",
        "copied": copied,
        "new": new,
        "dropped": dropped,
        "control_clock_reset": sorted(clock_buffers),
    }

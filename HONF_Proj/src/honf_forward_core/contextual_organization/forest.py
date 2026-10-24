"""Contextual candidate forests and keyed learned frontier decisions.

Topology is built from detached geometry/context distances. Node descriptors
are pooled from the original tensors so the learned frontier remains
differentiable with respect to contextual states and geometry metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Hashable, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CandidateEdge:
    """One undirected edge in the typed candidate graph, indexed by input rows."""

    left: int
    right: int
    geometry_distance: float
    context_distance: float
    cost: float
    spatial_knn: bool
    context_knn: bool
    spatial_spanning: bool


@dataclass(frozen=True)
class ForestStats:
    records: int
    candidates: int
    candidate_edges: int
    spatial_knn_edges: int
    context_knn_edges: int
    spatial_spanning_edges: int
    spatial_only_edges: int
    context_only_edges: int
    shared_knn_edges: int
    mean_geometry_distance: float
    mean_context_distance: float
    mean_candidate_cost: float
    tied_candidate_costs: int
    topology_detached: bool = True


@dataclass(frozen=True)
class ForestNode:
    """A leaf or binary candidate node; ancestry always refers to original rows."""

    index: int
    children: tuple[int, int] | None
    parent: int | None
    ancestry_local_indices: tuple[int, ...]
    ancestry_ids: tuple[Hashable, ...]
    identity: tuple[Hashable, ...]
    contextual_state: Tensor
    descriptor: Tensor
    position_mean: Tensor
    lower: Tensor
    upper: Tensor
    extent: Tensor
    mass: Tensor
    mean_measure: Tensor
    leaf_count: int
    candidate_edge: CandidateEdge | None = None

    @property
    def is_leaf(self) -> bool:
        return self.children is None


@dataclass(frozen=True)
class SceneForest:
    """One unbatched scene and one typed record bank."""

    bank: Hashable
    nodes: tuple[ForestNode, ...]
    root: int | None
    original_ids: tuple[Hashable, ...]
    spatial_neighbors: tuple[tuple[int, ...], ...]
    context_neighbors: tuple[tuple[int, ...], ...]
    candidate_edges: tuple[CandidateEdge, ...]
    stats: ForestStats
    states: Tensor
    coordinates: Tensor
    lengths: Tensor
    measure: Tensor
    geometry_scale: Tensor

    @property
    def record_count(self) -> int:
        return len(self.original_ids)

    @property
    def descriptor_dim(self) -> int:
        if self.nodes:
            return int(self.nodes[0].descriptor.shape[-1])
        return int(self.states.shape[-1] + 2 * self.coordinates.shape[-1] + 3)

    @property
    def internal_indices(self) -> tuple[int, ...]:
        return tuple(node.index for node in self.nodes if not node.is_leaf)


@dataclass(frozen=True)
class FrontierStats:
    candidate_nodes: int
    internal_nodes: int
    reached_internal_nodes: int
    actual_k: int
    expected_k_value: float
    sampled: bool
    capacity_active: bool
    score_function_active: bool


@dataclass(frozen=True)
class Frontier:
    """A hard exact-cover frontier plus differentiable policy summaries."""

    node_indices: tuple[int, ...]
    ancestry_local_indices: tuple[tuple[int, ...], ...]
    ancestry_ids: tuple[tuple[Hashable, ...], ...]
    K: int
    expected_K: Tensor
    log_prob: Tensor
    stop_logits: Mapping[int, Tensor]
    risk_logits: Mapping[int, Tensor]
    decisions: Mapping[int, bool]
    reached_internal: tuple[int, ...]
    mode: str
    sampling_identity: Mapping[str, Any]
    stats: FrontierStats
    capacity_active: bool
    score_function_active: bool


@dataclass
class _NodeDraft:
    index: int
    children: tuple[int, int] | None
    parent: int | None
    ancestry_local_indices: tuple[int, ...]
    ancestry_ids: tuple[Hashable, ...]
    identity_key: tuple[tuple[str, Any], ...]
    candidate_edge: CandidateEdge | None


def _native_id(value: Any) -> Hashable:
    if isinstance(value, np.generic):
        value = value.item()
    try:
        hash(value)
    except TypeError as error:
        raise TypeError("record IDs must be immutable and hashable") from error
    return value


def _id_order_key(value: Hashable) -> tuple[str, Any]:
    """Natural ordering for common scalar IDs, stable type/repr ordering otherwise."""
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    if isinstance(value, bool):
        return kind, int(value)
    if isinstance(value, int):
        return kind, value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("record IDs cannot be non-finite floats")
        return kind, value
    if isinstance(value, str):
        return kind, value
    if isinstance(value, bytes):
        return kind, value.hex()
    return kind, repr(value)


def _id_token(value: Hashable) -> str:
    kind, token = _id_order_key(value)
    return f"{kind}:{token!r}"


def _stable_seed(parts: Sequence[Any]) -> int:
    """Small deterministic FNV-1a seed for keyed draws; not an integrity hash."""
    payload = repr(tuple(parts)).encode("utf-8")
    state = 14695981039346656037
    for byte in payload:
        state = ((state ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return state & 0x7FFFFFFFFFFFFFFF


def _as_id_tuple(ids: Sequence[Hashable] | Tensor, count: int) -> tuple[Hashable, ...]:
    if isinstance(ids, Tensor):
        values = ids.detach().cpu().tolist()
    else:
        values = list(ids)
    if len(values) != count:
        raise ValueError(f"ids must contain one immutable ID per row; got {len(values)} for N={count}")
    normalized = tuple(_native_id(value) for value in values)
    order_keys = tuple(_id_order_key(value) for value in normalized)
    if len(set(normalized)) != len(normalized):
        raise ValueError("record IDs must be unique within a bank")
    if len(set(order_keys)) != len(order_keys):
        raise ValueError("record IDs need distinct stable ordering keys")
    return normalized


def _validate_inputs(
    states: Tensor,
    coordinates: Tensor,
    lengths: Tensor,
    measure: Tensor,
    ids: Sequence[Hashable] | Tensor,
) -> tuple[tuple[Hashable, ...], Tensor, Tensor, Tensor]:
    if not all(isinstance(value, Tensor) for value in (states, coordinates, lengths, measure)):
        raise TypeError("states, coordinates, lengths, and measure must be torch tensors")
    if states.ndim != 2 or states.shape[1] < 1:
        raise ValueError("states must have shape [N, H] for one unbatched bank")
    if coordinates.ndim != 2 or coordinates.shape[0] != states.shape[0] or coordinates.shape[1] < 1:
        raise ValueError("coordinates must have shape [N, D] with D >= 1")
    if lengths.shape != coordinates.shape:
        raise ValueError("lengths must have the same [N, D] shape as coordinates")
    if measure.shape != (states.shape[0],):
        raise ValueError("measure must have shape [N]")
    ids_tuple = _as_id_tuple(ids, states.shape[0])
    device = states.device
    coordinates = coordinates.to(device=device)
    lengths = lengths.to(device=device, dtype=coordinates.dtype)
    measure = measure.to(device=device, dtype=states.dtype)
    if not bool(torch.isfinite(states.detach()).all()):
        raise ValueError("contextual states must be finite before candidate construction")
    if not bool(torch.isfinite(coordinates.detach()).all()):
        raise ValueError("coordinates must be finite")
    if not bool(torch.isfinite(lengths.detach()).all()) or bool((lengths.detach() < 0).any()):
        raise ValueError("record lengths must be finite and nonnegative")
    if not bool(torch.isfinite(measure.detach()).all()) or bool((measure.detach() < 0).any()):
        raise ValueError("organizational measure must be finite and nonnegative")
    return ids_tuple, coordinates, lengths, measure


def _detached_distances(
    states: Tensor, coordinates: Tensor, lengths: Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized geometry, cosine-context distances, and per-axis scale."""
    context = states.detach().to(device="cpu", dtype=torch.float64).numpy()
    points = coordinates.detach().to(device="cpu", dtype=torch.float64).numpy()
    widths = lengths.detach().to(device="cpu", dtype=torch.float64).numpy()
    count, coordinate_dim = points.shape
    if count == 0:
        return (
            np.zeros((0, 0), dtype=np.float64),
            np.zeros((0, 0), dtype=np.float64),
            np.ones((coordinate_dim,), dtype=np.float64),
        )

    scene_span = np.ptp(points, axis=0)
    typical_width = np.zeros((coordinate_dim,), dtype=np.float64)
    for dim in range(coordinate_dim):
        positive = widths[:, dim][widths[:, dim] > 0.0]
        if positive.size:
            typical_width[dim] = float(np.median(positive))
    axis_scale = np.maximum(np.maximum(scene_span, typical_width), 1.0e-12)

    geometry_distance = np.zeros((count, count), dtype=np.float64)
    for left in range(count):
        normalized_delta = (points[left + 1 :] - points[left]) / axis_scale
        distances = 2.0 * np.linalg.norm(normalized_delta, axis=1) / math.sqrt(coordinate_dim)
        geometry_distance[left, left + 1 :] = distances
        geometry_distance[left + 1 :, left] = distances

    norms = np.linalg.norm(context, axis=1)
    denominator = norms[:, None] * norms[None, :]
    cosine = np.zeros((count, count), dtype=np.float64)
    np.divide(context @ context.T, denominator, out=cosine, where=denominator > 0.0)
    cosine = np.clip(cosine, -1.0, 1.0)
    context_distance = 1.0 - cosine
    np.fill_diagonal(geometry_distance, 0.0)
    np.fill_diagonal(context_distance, 0.0)
    return geometry_distance, context_distance, axis_scale


def _pair_key(left: int, right: int, id_keys: Sequence[tuple[str, Any]]) -> tuple[int, int]:
    if id_keys[left] <= id_keys[right]:
        return left, right
    return right, left


def build_forest(
    states: Tensor,
    coordinates: Tensor,
    lengths: Tensor,
    measure: Tensor,
    ids: Sequence[Hashable] | Tensor,
    bank: Hashable,
    *,
    k: int = 4,
) -> SceneForest:
    """Build one typed contextual candidate forest from `[N, ...]` inputs.

    Spatial distances use scene-span or typical record-width normalization;
    context distances are ``1 - cosine`` over detached contextual states. The
    candidate edge cost is the fixed sum of those two recorded components.
    Four-or-fewer spatial/context neighbors are symmetrized, and a spatial MST
    supplies connectivity without becoming part of the one-hop kNN halo.
    """
    if isinstance(k, bool) or int(k) != k or k < 0:
        raise ValueError("k must be a nonnegative integer")
    try:
        hash(bank)
    except TypeError as error:
        raise TypeError("bank must be a stable hashable type label") from error
    ids_tuple, coordinates, lengths, measure = _validate_inputs(
        states, coordinates, lengths, measure, ids
    )
    count, context_dim = states.shape
    coordinate_dim = coordinates.shape[1]
    id_keys = tuple(_id_order_key(value) for value in ids_tuple)
    geometry, context, axis_scale = _detached_distances(states, coordinates, lengths)

    spatial_sets = [set() for _ in range(count)]
    context_sets = [set() for _ in range(count)]
    edge_flags: dict[tuple[int, int], list[bool]] = {}

    def mark(left: int, right: int, flag_index: int) -> None:
        pair = _pair_key(left, right, id_keys)
        flags = edge_flags.setdefault(pair, [False, False, False])
        flags[flag_index] = True

    for row in range(count):
        others = [other for other in range(count) if other != row]
        spatial_order = sorted(others, key=lambda other: (geometry[row, other], id_keys[other]))
        context_order = sorted(others, key=lambda other: (context[row, other], id_keys[other]))
        for neighbor in spatial_order[: min(int(k), len(spatial_order))]:
            spatial_sets[row].add(neighbor)
            spatial_sets[neighbor].add(row)
            mark(row, neighbor, 0)
        for neighbor in context_order[: min(int(k), len(context_order))]:
            context_sets[row].add(neighbor)
            context_sets[neighbor].add(row)
            mark(row, neighbor, 1)

    # A geometry-only minimum spanning tree connects the graph, including
    # coincident records whose ID ordering resolves all ties deterministically.
    spatial_spanning_edges = 0
    disjoint_parent = list(range(count))
    disjoint_rank = [0] * count

    def find(value: int) -> int:
        while disjoint_parent[value] != value:
            disjoint_parent[value] = disjoint_parent[disjoint_parent[value]]
            value = disjoint_parent[value]
        return value

    all_spatial_pairs = sorted(
        ((geometry[left, right], *_pair_key(left, right, id_keys))
         for left in range(count) for right in range(left + 1, count)),
        key=lambda item: (item[0], id_keys[item[1]], id_keys[item[2]]),
    )
    for distance, left, right in all_spatial_pairs:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            continue
        if disjoint_rank[left_root] < disjoint_rank[right_root]:
            left_root, right_root = right_root, left_root
        disjoint_parent[right_root] = left_root
        if disjoint_rank[left_root] == disjoint_rank[right_root]:
            disjoint_rank[left_root] += 1
        mark(left, right, 2)
        spatial_spanning_edges += 1
        if spatial_spanning_edges == max(0, count - 1):
            break

    edges = tuple(
        CandidateEdge(
            left=left,
            right=right,
            geometry_distance=float(geometry[left, right]),
            context_distance=float(context[left, right]),
            cost=float(geometry[left, right] + context[left, right]),
            spatial_knn=flags[0],
            context_knn=flags[1],
            spatial_spanning=flags[2],
        )
        for (left, right), flags in sorted(
            edge_flags.items(),
            key=lambda item: (
                geometry[item[0][0], item[0][1]] + context[item[0][0], item[0][1]],
                id_keys[item[0][0]],
                id_keys[item[0][1]],
            ),
        )
    )

    drafts: list[_NodeDraft] = []
    for row, original_id in enumerate(ids_tuple):
        drafts.append(
            _NodeDraft(
                index=row,
                children=None,
                parent=None,
                ancestry_local_indices=(row,),
                ancestry_ids=(original_id,),
                identity_key=(id_keys[row],),
                candidate_edge=None,
            )
        )

    if count:
        component_parent = list(range(count))
        component_rank = [0] * count
        component_tree = {row: row for row in range(count)}

        def component_find(value: int) -> int:
            while component_parent[value] != value:
                component_parent[value] = component_parent[component_parent[value]]
                value = component_parent[value]
            return value

        for edge in edges:
            left_root = component_find(edge.left)
            right_root = component_find(edge.right)
            if left_root == right_root:
                continue
            left_child = component_tree[left_root]
            right_child = component_tree[right_root]
            left_draft, right_draft = drafts[left_child], drafts[right_child]
            if right_draft.identity_key < left_draft.identity_key:
                left_child, right_child = right_child, left_child
                left_draft, right_draft = right_draft, left_draft
            ancestry_local = tuple(
                sorted(
                    left_draft.ancestry_local_indices + right_draft.ancestry_local_indices,
                    key=lambda row: id_keys[row],
                )
            )
            ancestry_ids = tuple(ids_tuple[row] for row in ancestry_local)
            identity_key = tuple(id_keys[row] for row in ancestry_local)
            node_index = len(drafts)
            node = _NodeDraft(
                index=node_index,
                children=(left_child, right_child),
                parent=None,
                ancestry_local_indices=ancestry_local,
                ancestry_ids=ancestry_ids,
                identity_key=identity_key,
                candidate_edge=edge,
            )
            drafts.append(node)
            drafts[left_child].parent = node_index
            drafts[right_child].parent = node_index

            if component_rank[left_root] < component_rank[right_root]:
                left_root, right_root = right_root, left_root
            component_parent[right_root] = left_root
            if component_rank[left_root] == component_rank[right_root]:
                component_rank[left_root] += 1
            # The spatial MST already connected the graph. This independent
            # union structure follows combined candidate cost for the hierarchy.
            component_tree[left_root] = node_index
            component_tree.pop(right_root, None)
            if len(drafts) == 2 * count - 1:
                break

        roots = [draft.index for draft in drafts if draft.parent is None]
        if len(roots) != 1 or len(drafts) != 2 * count - 1:
            raise RuntimeError("spatial spanning edges failed to produce one full binary hierarchy")
        root = roots[0]
    else:
        root = None

    if drafts:
        membership = torch.zeros((len(drafts), count), dtype=torch.bool, device=states.device)
        for draft in drafts:
            membership[draft.index, list(draft.ancestry_local_indices)] = True
        context_weights = membership.to(dtype=states.dtype)
        counts = context_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        node_states = context_weights @ states / counts

        coordinate_weights = membership.to(dtype=coordinates.dtype)
        coordinate_counts = coordinate_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        node_position_mean = coordinate_weights @ coordinates / coordinate_counts
        member_lower = coordinates - 0.5 * lengths
        member_upper = coordinates + 0.5 * lengths
        expanded_membership = membership.unsqueeze(-1)
        node_lower = torch.where(
            expanded_membership, member_lower.unsqueeze(0), torch.full_like(member_lower.unsqueeze(0), float("inf"))
        ).amin(dim=1)
        node_upper = torch.where(
            expanded_membership, member_upper.unsqueeze(0), torch.full_like(member_upper.unsqueeze(0), float("-inf"))
        ).amax(dim=1)
        node_extent = node_upper - node_lower
        node_mass = context_weights @ measure
        node_mean_measure = node_mass / counts.squeeze(-1).to(dtype=node_mass.dtype)
        scale_tensor = torch.as_tensor(axis_scale, dtype=coordinates.dtype, device=coordinates.device)
        coordinate_origin = coordinates.amin(dim=0)
        normalized_position = (node_position_mean - coordinate_origin) / scale_tensor
        normalized_extent = node_extent / scale_tensor
        log_mass = torch.log1p(node_mass.clamp_min(0.0)).unsqueeze(-1)
        log_count = torch.log1p(counts.squeeze(-1)).unsqueeze(-1).to(dtype=states.dtype)
        measure_mean_feature = node_mean_measure.unsqueeze(-1).to(dtype=states.dtype)
        descriptors = torch.cat(
            (
                node_states,
                normalized_position.to(dtype=states.dtype),
                normalized_extent.to(dtype=states.dtype),
                log_mass.to(dtype=states.dtype),
                measure_mean_feature,
                log_count,
            ),
            dim=-1,
        )
        nodes = tuple(
            ForestNode(
                index=draft.index,
                children=draft.children,
                parent=draft.parent,
                ancestry_local_indices=draft.ancestry_local_indices,
                ancestry_ids=draft.ancestry_ids,
                identity=draft.ancestry_ids,
                contextual_state=node_states[draft.index],
                descriptor=descriptors[draft.index],
                position_mean=node_position_mean[draft.index],
                lower=node_lower[draft.index],
                upper=node_upper[draft.index],
                extent=node_extent[draft.index],
                mass=node_mass[draft.index],
                mean_measure=node_mean_measure[draft.index],
                leaf_count=len(draft.ancestry_local_indices),
                candidate_edge=draft.candidate_edge,
            )
            for draft in drafts
        )
    else:
        nodes = ()
        scale_tensor = torch.as_tensor(axis_scale, dtype=coordinates.dtype, device=coordinates.device)

    edge_costs = [edge.cost for edge in edges]
    sorted_costs = sorted(edge_costs)
    tied_costs = sum(left == right for left, right in zip(sorted_costs, sorted_costs[1:]))
    stats = ForestStats(
        records=count,
        candidates=len(drafts),
        candidate_edges=len(edges),
        spatial_knn_edges=sum(bool(edge.spatial_knn) for edge in edges),
        context_knn_edges=sum(bool(edge.context_knn) for edge in edges),
        spatial_spanning_edges=spatial_spanning_edges,
        spatial_only_edges=sum(edge.spatial_knn and not edge.context_knn for edge in edges),
        context_only_edges=sum(edge.context_knn and not edge.spatial_knn for edge in edges),
        shared_knn_edges=sum(edge.spatial_knn and edge.context_knn for edge in edges),
        mean_geometry_distance=float(np.mean([edge.geometry_distance for edge in edges])) if edges else 0.0,
        mean_context_distance=float(np.mean([edge.context_distance for edge in edges])) if edges else 0.0,
        mean_candidate_cost=float(np.mean(edge_costs)) if edge_costs else 0.0,
        tied_candidate_costs=tied_costs,
    )
    spatial_neighbors = tuple(
        tuple(sorted(neighbors, key=lambda row: id_keys[row])) for neighbors in spatial_sets
    )
    context_neighbors = tuple(
        tuple(sorted(neighbors, key=lambda row: id_keys[row])) for neighbors in context_sets
    )
    return SceneForest(
        bank=bank,
        nodes=nodes,
        root=root,
        original_ids=ids_tuple,
        spatial_neighbors=spatial_neighbors,
        context_neighbors=context_neighbors,
        candidate_edges=edges,
        stats=stats,
        states=states,
        coordinates=coordinates,
        lengths=lengths,
        measure=measure,
        geometry_scale=scale_tensor,
    )


class ContextualFrontierPolicy(nn.Module):
    """Shared contextual stop/risk scorer with exact-cover frontier sampling."""

    sampling_scheme = "fnv1a64-node-key-v1"

    def __init__(
        self,
        context_dim: int,
        coordinate_dim: int,
        *,
        hidden_dim: int | None = None,
        risk_dim: int = 1,
        run_seed: int = 0,
        warmup_epochs: int = 20,
    ) -> None:
        super().__init__()
        if context_dim < 1 or coordinate_dim < 1 or risk_dim < 1:
            raise ValueError("context_dim, coordinate_dim, and risk_dim must be positive")
        if isinstance(warmup_epochs, bool) or int(warmup_epochs) != warmup_epochs or warmup_epochs < 0:
            raise ValueError("warmup_epochs must be a nonnegative integer")
        if isinstance(run_seed, bool) or int(run_seed) != run_seed:
            raise ValueError("run_seed must be an integer")
        descriptor_dim = context_dim + 2 * coordinate_dim + 3
        width = int(hidden_dim or max(32, context_dim))
        if width < 1:
            raise ValueError("hidden_dim must be positive")
        self.context_dim = int(context_dim)
        self.coordinate_dim = int(coordinate_dim)
        self.descriptor_dim = descriptor_dim
        self.risk_dim = int(risk_dim)
        self.run_seed = int(run_seed)
        self.warmup_epochs = int(warmup_epochs)
        self.hidden_dim = width
        self.shared_scorer = nn.Sequential(
            nn.Linear(3 * descriptor_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        self.stop_head = nn.Linear(width, 1)
        self.risk_head = nn.Linear(width, risk_dim)
        nn.init.zeros_(self.stop_head.weight)
        nn.init.constant_(self.stop_head.bias, -4.0)

    def get_extra_state(self) -> dict[str, Any]:
        """Keep sampling identity/config in ordinary PyTorch module state."""
        return {
            "version": 1,
            "run_seed": self.run_seed,
            "sampling_scheme": self.sampling_scheme,
            "warmup_epochs": self.warmup_epochs,
        }

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("frontier policy extra state must be a mapping")
        if state.get("version") != 1 or state.get("sampling_scheme") != self.sampling_scheme:
            raise ValueError("frontier policy sampling state is incompatible")
        self.run_seed = int(state["run_seed"])
        self.warmup_epochs = int(state["warmup_epochs"])

    @staticmethod
    def temperature_for_epoch(epoch: int) -> float:
        t = min(1.0, max(0.0, (int(epoch) - 1) / 199.0))
        return 0.5 + 0.75 * (1.0 + math.cos(math.pi * t))

    @staticmethod
    def structural_multiplier_for_epoch(epoch: int) -> float:
        t = min(1.0, max(0.0, (int(epoch) - 1) / 199.0))
        return 0.1 + 0.45 * (1.0 - math.cos(math.pi * t))

    def _node_scores(self, forest: SceneForest) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        if forest.descriptor_dim != self.descriptor_dim:
            raise ValueError(
                f"forest descriptor width {forest.descriptor_dim} does not match policy width {self.descriptor_dim}"
            )
        weight = self.shared_scorer[0].weight
        internal = tuple(node.index for node in forest.nodes if node.children is not None)
        if not internal:
            return {}, {}
        descriptors = torch.stack([node.descriptor for node in forest.nodes]).to(
            device=weight.device, dtype=weight.dtype
        )
        left_indices = torch.as_tensor(
            [forest.nodes[index].children[0] for index in internal],
            device=weight.device,
            dtype=torch.long,
        )
        right_indices = torch.as_tensor(
            [forest.nodes[index].children[1] for index in internal],
            device=weight.device,
            dtype=torch.long,
        )
        node_indices = torch.as_tensor(internal, device=weight.device, dtype=torch.long)
        descriptor = descriptors.index_select(0, node_indices)
        left = descriptors.index_select(0, left_indices)
        right = descriptors.index_select(0, right_indices)
        child_mean = 0.5 * (left + right)
        child_difference = torch.abs(left - right)
        features = torch.cat((descriptor, child_mean, child_difference), dim=-1)
        shared = self.shared_scorer(features)
        stop_values = self.stop_head(shared).reshape(-1)
        risk_values = self.risk_head(shared)
        stop_logits = {index: stop_values[row] for row, index in enumerate(internal)}
        risk_logits = {index: risk_values[row] for row, index in enumerate(internal)}
        return stop_logits, risk_logits

    @staticmethod
    def _keyed_uniform(key_parts: Sequence[Any]) -> float:
        seed = _stable_seed(key_parts)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return torch.rand((), generator=generator, dtype=torch.float64).item()

    @classmethod
    def _keyed_bernoulli(cls, probability: Tensor, key_parts: Sequence[Any]) -> bool:
        draw = cls._keyed_uniform(key_parts)
        threshold = float(probability.detach().to(device="cpu", dtype=torch.float64))
        return draw < threshold

    def forward(
        self,
        forest: SceneForest,
        *,
        epoch: int,
        case_key: Hashable,
        update: int = 0,
        training: bool,
        force_leaves: bool = False,
        forced_decisions: Mapping[int, bool] | None = None,
        forced_subtree: tuple[int, Literal["coarse", "split"]] | None = None,
    ) -> Frontier:
        """Select a hard cover and expose expected-K/log-probability gradients.

        Training epochs through ``warmup_epochs`` force all leaves. Later
        training samples reached decisions with node-keyed Bernoulli draws;
        evaluation uses the deterministic ``stop_logit >= 0`` rule.
        ``forced_subtree=(node_index, 'coarse'|'split')`` changes only that
        subtree and opens its ancestor path as needed for isolated comparison.
        """
        if isinstance(epoch, bool) or int(epoch) != epoch or epoch < 1:
            raise ValueError("epoch must be a positive integer")
        if isinstance(update, bool) or int(update) != update or update < 0:
            raise ValueError("update must be a nonnegative integer")
        if forced_decisions is not None and forced_subtree is not None:
            raise ValueError("forced_decisions and forced_subtree are separate execution modes")
        try:
            hash(case_key)
        except TypeError as error:
            raise TypeError("case_key must be a stable hashable case identity") from error

        stop_logits, risk_logits = self._node_scores(forest)
        tau_value = self.temperature_for_epoch(int(epoch))
        sample_phase = bool(training and not force_leaves and epoch > self.warmup_epochs)
        warmup_phase = bool(training and epoch <= self.warmup_epochs)
        leaves_only = bool(force_leaves or warmup_phase)
        mode = "forced_leaves" if leaves_only else "keyed_sample" if sample_phase else "deterministic_eval"
        case_token = _id_token(case_key)

        if forced_decisions is not None:
            unknown = set(forced_decisions) - set(forest.internal_indices)
            if unknown:
                raise ValueError(f"forced decisions reference non-internal node indices: {sorted(unknown)}")
            mode = "enumerated_frontier"

        target_node: int | None = None
        target_choice: str | None = None
        forced_path: set[int] = set()
        if forced_subtree is not None:
            if len(forced_subtree) != 2:
                raise ValueError("forced_subtree must be (node_index, 'coarse'|'split')")
            target_node, target_choice = int(forced_subtree[0]), forced_subtree[1]
            if target_node < 0 or target_node >= len(forest.nodes):
                raise ValueError("forced_subtree node index is outside the forest")
            if target_choice not in {"coarse", "split"}:
                raise ValueError("forced_subtree choice must be 'coarse' or 'split'")
            if forest.nodes[target_node].children is None:
                raise ValueError("forced_subtree must select an internal candidate node")
            ancestor = forest.nodes[target_node].parent
            while ancestor is not None:
                forced_path.add(ancestor)
                ancestor = forest.nodes[ancestor].parent
            mode = f"forced_subtree_{target_choice}"

        if forced_decisions is not None and leaves_only:
            raise ValueError("forced_decisions cannot be applied during forced-leaf warmup")

        log_probability_terms: list[Tensor] = []
        if forest.root is None:
            reference = next(self.parameters())
            expected_k = reference.new_zeros(())
            log_prob = reference.new_zeros(())
            selected: list[int] = []
            reached: list[int] = []
            decisions: dict[int, bool] = {}
        else:
            reference = next(iter(stop_logits.values()), forest.nodes[forest.root].descriptor.reshape(-1)[0])
            temperature = reference.new_tensor(tau_value)
            internal_indices = tuple(stop_logits)
            internal_positions = {index: position for position, index in enumerate(internal_indices)}
            internal_logits = (
                torch.stack([stop_logits[index] for index in internal_indices])
                if internal_indices else reference.new_empty((0,))
            )
            scaled_logits = internal_logits / temperature
            probability_values = torch.sigmoid(scaled_logits)
            scaled_by_node = {
                index: scaled_logits[position] for position, index in enumerate(internal_indices)
            }
            probability_by_node = {
                index: probability_values[position] for position, index in enumerate(internal_indices)
            }

            # Each node's reach is the product of split probabilities on its
            # ancestor path. A padded path matrix lets one tensor reduction
            # replace a recursive chain of per-node CUDA scalar operations.
            path_positions = []
            for node in forest.nodes:
                path = []
                ancestor = node.parent
                while ancestor is not None:
                    path.append(internal_positions[ancestor])
                    ancestor = forest.nodes[ancestor].parent
                path_positions.append(path)
            max_depth = max((len(path) for path in path_positions), default=0)
            if max_depth:
                path_indices = np.zeros((len(forest.nodes), max_depth), dtype=np.int64)
                path_mask = np.zeros((len(forest.nodes), max_depth), dtype=np.bool_)
                for row, path in enumerate(path_positions):
                    if path:
                        path_indices[row, :len(path)] = path
                        path_mask[row, :len(path)] = True
                path_indices_tensor = torch.as_tensor(path_indices, device=reference.device)
                path_mask_tensor = torch.as_tensor(path_mask, device=reference.device)
                ancestor_probabilities = probability_values.index_select(
                    0, path_indices_tensor.reshape(-1)
                ).reshape(len(forest.nodes), max_depth)
                path_open_probabilities = torch.where(
                    path_mask_tensor,
                    1.0 - ancestor_probabilities,
                    torch.ones_like(ancestor_probabilities),
                )
                reaches = path_open_probabilities.prod(dim=1)
            else:
                reaches = reference.new_ones((len(forest.nodes),))
            internal_node_tensor = torch.as_tensor(
                internal_indices, device=reference.device, dtype=torch.long
            )
            leaves = tuple(node.index for node in forest.nodes if node.children is None)
            leaf_tensor = torch.as_tensor(leaves, device=reference.device, dtype=torch.long)
            internal_expected = (
                reaches.index_select(0, internal_node_tensor) * probability_values
            ).sum() if internal_indices else reference.new_zeros(())
            leaf_expected = reaches.index_select(0, leaf_tensor).sum() if leaves else reference.new_zeros(())
            expected_k = internal_expected + leaf_expected

            if sample_phase:
                cpu_probabilities = probability_values.detach().to(
                    device="cpu", dtype=torch.float64
                ).tolist()
                sample_probability_by_node = {
                    index: cpu_probabilities[position]
                    for position, index in enumerate(internal_indices)
                }
            elif not leaves_only and forced_decisions is None:
                cpu_logits = internal_logits.detach().to(device="cpu").tolist()
                deterministic_stop_by_node = {
                    index: value >= 0.0 for index, value in zip(internal_indices, cpu_logits)
                }

            selected = []
            reached = []
            decisions = {}
            def visit(node_index: int) -> None:
                node = forest.nodes[node_index]
                if node.children is None:
                    selected.append(node_index)
                    return
                reached.append(node_index)
                scaled = scaled_by_node[node_index]
                probability = probability_by_node[node_index]

                # A counterfactual intervention must be able to open the path
                # to its target even for the fixed-leaf J-D arm. Apply the
                # target and its forced ancestors before the general
                # leaves-only rule; all unrelated branches remain at leaves.
                if target_node == node_index:
                    should_stop = target_choice == "coarse"
                    is_forced = True
                elif node_index in forced_path:
                    should_stop = False
                    is_forced = True
                elif leaves_only:
                    should_stop = False
                    is_forced = True
                elif forced_decisions is not None:
                    if node_index not in forced_decisions:
                        raise ValueError(f"forced_decisions missing reached node {node_index}")
                    should_stop = bool(forced_decisions[node_index])
                    is_forced = False
                elif sample_phase:
                    node_key = tuple(_id_token(value) for value in node.identity)
                    key_parts = (
                            self.run_seed,
                            case_token,
                            int(epoch),
                            int(update),
                            _id_token(forest.bank),
                            node_key,
                    )
                    draw = self._keyed_uniform(key_parts)
                    should_stop = draw < sample_probability_by_node[node_index]
                    is_forced = False
                else:
                    should_stop = deterministic_stop_by_node[node_index]
                    is_forced = True

                decisions[node_index] = should_stop
                if sample_phase and not is_forced:
                    log_probability_terms.append(F.logsigmoid(scaled if should_stop else -scaled))
                if should_stop:
                    selected.append(node_index)
                    return
                left, right = node.children
                visit(left)
                visit(right)

            visit(forest.root)
            log_prob = torch.stack(log_probability_terms).sum() if log_probability_terms else reference.new_zeros(())

        selected_ancestry = tuple(forest.nodes[index].ancestry_local_indices for index in selected)
        selected_ids = tuple(forest.nodes[index].ancestry_ids for index in selected)
        flattened = [row for ancestry in selected_ancestry for row in ancestry]
        if sorted(flattened) != list(range(forest.record_count)) or len(flattened) != forest.record_count:
            raise RuntimeError("selected frontier does not exactly cover every original record once")

        capacity_active = bool(training and not force_leaves and epoch > self.warmup_epochs)
        score_function_active = bool(sample_phase and forced_subtree is None)
        actual_k = len(selected)
        expected_k_value = float(expected_k.detach().to(device="cpu", dtype=torch.float64))
        frontier_stats = FrontierStats(
            candidate_nodes=len(forest.nodes),
            internal_nodes=len(forest.internal_indices),
            reached_internal_nodes=len(reached),
            actual_k=actual_k,
            expected_k_value=expected_k_value,
            sampled=bool(sample_phase and forced_decisions is None and forced_subtree is None),
            capacity_active=capacity_active,
            score_function_active=score_function_active,
        )
        sampling_identity = {
            "version": 1,
            "scheme": self.sampling_scheme,
            "run_seed": self.run_seed,
            "case_key": case_token,
            "epoch": int(epoch),
            "update": int(update),
            "bank": _id_token(forest.bank),
            "mode": mode,
            "decisions": tuple(
                (tuple(_id_token(value) for value in forest.nodes[index].identity), bool(choice))
                for index, choice in decisions.items()
            ),
        }
        return Frontier(
            node_indices=tuple(selected),
            ancestry_local_indices=selected_ancestry,
            ancestry_ids=selected_ids,
            K=actual_k,
            expected_K=expected_k,
            log_prob=log_prob,
            stop_logits=stop_logits,
            risk_logits=risk_logits,
            decisions=dict(decisions),
            reached_internal=tuple(reached),
            mode=mode,
            sampling_identity=sampling_identity,
            stats=frontier_stats,
            capacity_active=capacity_active,
            score_function_active=score_function_active,
        )

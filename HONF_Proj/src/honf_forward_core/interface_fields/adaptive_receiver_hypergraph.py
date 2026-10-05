"""Input-only, task-trained receiver hierarchy for the shared fine reader.

The geometry tree indexes receivers, not source values. Every frontier action
retains physical source order and is evaluated by the common deduplicated read.
The hard operator uses sparse source densities and a deterministic frontier;
``soft=True`` supplies positive restoration permissions for the training VJP.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .adaptive_interaction_cover import CaseLocalReceiverTree, ReceiverAnchorUniverse, canonical_receiver_index
from .receiver_tree_access import build_receiver_tree_geometry, receiver_tree_access
from .routing_index.sparse_projection import source_measure_sparsemax
from .topology_probe import FixedTopologyInvalid, fixed_active_projection, validate_catalogue
from .typed_hypergraph_state import (
    TypedHypergraphState,
    TypedSourceAccess,
    build_source_catalogue,
    pad_geometry,
    source_moments,
)
from .types import EncodedInterfaceCase

MECHANISMS = ("MM", "ME", "EM", "QM", "QE")


def _pool(states: torch.Tensor, mass: torch.Tensor, *, soft_precision: bool = False,
          exact_mass: bool = False) -> torch.Tensor:
    if not soft_precision:
        denominator = mass.sum(-1, keepdim=True)
        safe = torch.where(mass[..., None] > 0, states, torch.zeros_like(states))
        denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator)) if exact_mass else denominator.clamp_min(1e-12)
        return (safe * mass[..., None]).sum(-2) / denominator
    floor = states.new_tensor(1e-12).to(torch.float64)
    mass = mass.to(torch.float64)
    denominator = mass.sum(-1, keepdim=True)
    safe = torch.where(mass[..., None] > 0, states, torch.zeros_like(states)).to(torch.float64)
    denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator)) if exact_mass else denominator.clamp_min(floor)
    summary = (safe * mass[..., None]).sum(-2) / denominator
    return summary.to(states.dtype)


def _membership(scores: torch.Tensor, measure: torch.Tensor, *, soft: bool,
                temperature: float) -> torch.Tensor:
    """Measure-aware density: splitting an identical source atom preserves it."""
    score_floor = torch.finfo(scores.dtype).tiny
    measure_floor = measure.new_tensor(1e-12)
    if soft:
        scores = scores.to(torch.float64)
        measure = measure.to(torch.float64)
    normalized = measure / measure.sum(-1, keepdim=True).clamp_min(measure_floor.to(measure.dtype))
    valid = (measure > 0)[:, None, :].expand_as(scores)
    if scores.shape[-1] == 0:
        return scores
    if not soft:
        return source_measure_sparsemax(scores, normalized, valid).density.to(scores.dtype)
    masked = torch.where(valid, scores / temperature, scores.new_full((), -1e4))
    shifted = masked - masked.amax(-1, keepdim=True)
    positive = torch.where(valid, shifted.exp(), torch.zeros_like(scores))
    # A positive permission on every eligible source allows restoration of a
    # source whose hard sparsemax entry has become exactly zero.
    positive = torch.where(valid, positive.clamp_min(score_floor), positive)
    denominator = (positive * normalized[:, None]).sum(-1, keepdim=True)
    return positive / denominator.clamp_min(score_floor)


def _frontier_admission(tree: CaseLocalReceiverTree, gates: torch.Tensor,
                        capacity: int) -> torch.Tensor:
    """Expected frontier activity, including nodes outside a sampled Q batch."""
    incoming = {0: gates.new_ones(())}
    active = []
    for index, node in enumerate(tree.nodes):
        gate = gates[index]
        active.append(incoming[index] if node.is_leaf else incoming[index] * (1 - gate))
        if not node.is_leaf:
            incoming[node.left] = incoming[index] * gate
            incoming[node.right] = incoming[index] * gate
    return F.pad(torch.stack(active), (0, capacity - len(tree.nodes)))


def _receiver_node_extrema(tree: CaseLocalReceiverTree, coords: torch.Tensor,
                           indices: torch.Tensor, valid: torch.Tensor):
    """Extrema over distinct coordinate/role blocks preserve tie derivatives.

    Fine-row extrema have identical values under exact atom refinement, but
    their equal-tie subgradient depends on the number of duplicate rows.
    Canonical coordinates pull each block's derivative back by child mass.
    Historical trees retain their original fine-row convention.
    """
    if tree.measure_consistent:
        view = tree.canonical_index
        atom_blocks = {atom: block for block, atoms in enumerate(view.block_atoms) for atom in atoms}
        node_blocks = [sorted({atom_blocks[atom] for atom in node.anchor_indices}) for node in tree.nodes]
        width = max(map(len, node_blocks))
        indices = torch.tensor([row + [0] * (width - len(row)) for row in node_blocks],
                               device=coords.device, dtype=torch.long)
        valid = torch.arange(width, device=coords.device)[None] < torch.tensor(
            [len(row) for row in node_blocks], device=coords.device)[:, None]
        coords = view.coordinates
    selected = coords[indices]
    low = torch.where(valid[..., None], selected, torch.full_like(selected, torch.inf)).amin(1)
    high = torch.where(valid[..., None], selected, torch.full_like(selected, -torch.inf)).amax(1)
    return low, high


class AdaptiveReceiverHypergraph(nn.Module):
    """Five typed receiver hierarchies, independent of requested query batches.

    ``set_epoch`` selects the absolute-epoch curriculum. Its counters are
    persistent buffers and ordinary checkpoint/RNG resume preserves it.
    Evaluation always selects the deterministic learned frontier.
    """

    def __init__(self, hidden_dim: int, *, spatial_dim: int | None = None,
                 organizer_dim: int = 64, control_dim: int = 16,
                 max_depth: int = 3, overlap_fraction: float = 0.06,
                 temperature: float = 1.0, faithful_controls: bool = False,
                 measure_consistent: bool = False,
                 structural_measure_policy_version: int = 1) -> None:
        super().__init__()
        if spatial_dim not in (None, 2, 3):
            raise ValueError("receiver hierarchies support spatial dimensions 2 and 3")
        if min(hidden_dim, organizer_dim, control_dim) < 1 or not 0 <= max_depth <= 3:
            raise ValueError("positive dimensions and receiver depth between zero and three are required")
        if temperature <= 0 or not 0 < overlap_fraction < 1:
            raise ValueError("positive temperature and overlap fraction in (0,1) are required")
        if isinstance(structural_measure_policy_version, bool) or structural_measure_policy_version not in (1, 2):
            raise ValueError("Structural measure policy version must be 1 or 2")
        self.hidden_dim = int(hidden_dim)
        self.control_dim = int(control_dim)
        self.max_depth = int(max_depth)
        self.capacity = 2 ** (max_depth + 1) - 1
        self.overlap_fraction = float(overlap_fraction)
        self.temperature = float(temperature)
        self.faithful_controls = bool(faithful_controls)
        self.measure_consistent = bool(measure_consistent)
        self.structural_measure_policy_version = int(structural_measure_policy_version)
        self.node_encoder = nn.Sequential(nn.Linear(4 * hidden_dim + 28, organizer_dim),
                                          nn.GELU(), nn.Linear(organizer_dim, organizer_dim), nn.GELU())
        self.source_encoder = nn.Linear(hidden_dim, organizer_dim)
        self.split_head = nn.Linear(organizer_dim, 1)
        self.source_scores = nn.ModuleDict({tau: nn.Sequential(
            nn.Linear(2 * organizer_dim + 8, organizer_dim), nn.GELU(),
            nn.Linear(organizer_dim, 1)) for tau in MECHANISMS})
        if self.faithful_controls:
            # Center/extent (xyz with presence), depth and receiver roles only.
            # The rich node embedding remains exclusively in the planner.
            self.control_geometry_encoder = nn.Sequential(nn.Linear(21, organizer_dim), nn.GELU())
        self.control_heads = nn.ModuleDict({tau: nn.Sequential(
            nn.Linear((3 if self.faithful_controls else 2) * hidden_dim + organizer_dim + 7, organizer_dim), nn.GELU(),
            nn.Linear(organizer_dim, control_dim)) for tau in MECHANISMS})
        self.geometry_strength = nn.ParameterDict({tau: nn.Parameter(torch.zeros(())) for tau in MECHANISMS})
        # Parent and identical children initially expose all physical sources
        # with unit density and identity control, regardless of frontier depth.
        for head in (*self.source_scores.values(), *self.control_heads.values()):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        # The physical gain projections begin at zero. Identical nonzero
        # controls let those projections learn on step one, then send useful
        # task gradients into the collective heads on subsequent steps.
        for head in self.control_heads.values():
            with torch.no_grad():
                head[-1].bias.copy_(torch.linspace(-0.1, 0.1, control_dim))
        nn.init.zeros_(self.split_head.weight)
        nn.init.zeros_(self.split_head.bias)
        self.register_buffer("training_epoch", torch.tensor(1, dtype=torch.long))
        self.register_buffer("exercise_counter", torch.tensor(0, dtype=torch.long))

    def set_epoch(self, epoch: int) -> None:
        if epoch < 1:
            raise ValueError("curriculum epochs start at one")
        self.training_epoch.fill_(int(epoch))

    def _continuous_input(self, value: torch.Tensor) -> torch.Tensor:
        # Training keeps the established organizer/physical gradient boundary.
        # Frozen inverse evaluation still differentiates continuous physics at
        # the selected discrete topology, including collective control moments.
        return value.detach() if self.training else value

    def _exercise_depth(self) -> int | None:
        if not self.training:
            return None
        epoch = int(self.training_epoch.item())
        exploration = (1.0 if epoch <= 25 else
                       1.0 - 0.75 * min((epoch - 25) / 75.0, 1.0) if epoch <= 100 else
                       0.25 if epoch <= 300 else 0.10)
        if epoch <= 25 or bool(torch.rand((), device=self.training_epoch.device) < exploration):
            depth = int(self.exercise_counter.item()) % (self.max_depth + 1)
            self.exercise_counter.add_(1)
            return depth
        return None

    def _catalogue(self, encoded: EncodedInterfaceCase, states: torch.Tensor, case: int,
                   mechanism: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mechanism in ("MM", "ME"):
            valid = encoded.module_present[case] > 0.5
            return encoded.module_centers[case, valid], encoded.module_present[case, valid], states[case, valid]
        if mechanism == "EM":
            valid = encoded.env_weights[case] > 0
            return encoded.env_coords[case, valid], encoded.env_weights[case, valid], self._continuous_input(encoded.env_tokens[case, valid])
        coords, mass = encoded.receiver_anchor_coords, encoded.receiver_anchor_weights
        if coords is None:
            valid = encoded.env_weights[case] > 0
            return encoded.env_coords[case, valid], encoded.env_weights[case, valid], self._continuous_input(encoded.env_tokens[case, valid])
        coords = coords if coords.ndim == 2 else coords[case]
        mass = coords.new_ones(coords.shape[0]) if mass is None else (mass if mass.ndim == 1 else mass[case])
        valid = mass > 0
        tokens = self._continuous_input(encoded.global_token[case]).expand(int(valid.sum()), -1)
        return coords[valid], mass[valid], tokens

    def _receiver_roles(self, encoded, case, mechanism, count):
        if mechanism in ("QM", "QE") and encoded.receiver_anchor_roles is not None:
            roles = encoded.receiver_anchor_roles
            roles = roles if roles.ndim == 1 else roles[case]
            mass = encoded.receiver_anchor_weights
            if mass is not None:
                mass = mass if mass.ndim == 1 else mass[case]
                roles = roles[mass > 0]
        else:
            role = 0 if mechanism in ("MM", "ME") else 1 if mechanism == "EM" else 2
            roles = torch.full((count,), role, device=encoded.module_tokens.device, dtype=torch.long)
        if bool(((roles < 0) | (roles >= 8)).any()):
            raise ValueError("receiver role codes must lie in [0,7]")
        return roles

    def _index(self, encoded, states, case, tau, case_scale, summaries, global_token, measures, phase_features,
               fixed_tree=None, *, encode_nodes=True):
        coords, mass, receiver_states = self._catalogue(encoded, self._continuous_input(states), case, tau)
        roles = self._receiver_roles(encoded, case, tau, coords.shape[0])
        if coords.shape[0] == 0:
            coords = states.new_zeros((1, encoded.module_centers.shape[-1]))
            mass = coords.new_ones(1)
            roles = torch.zeros(1, device=coords.device, dtype=torch.long)
            receiver_states = states.new_zeros((1, self.hidden_dim))
        universe = ReceiverAnchorUniverse(coords, mass, roles, case_scale)
        if fixed_tree is None:
            tree = CaseLocalReceiverTree.build(universe, max_nodes=self.capacity, min_leaf_anchors=1,
                overlap_fraction=self.overlap_fraction, max_depth=self.max_depth, measure_consistent=self.measure_consistent)
        else:
            if coords.shape != fixed_tree.universe.coordinates.shape or not torch.equal(roles, fixed_tree.universe.roles.to(roles)):
                raise FixedTopologyInvalid("Fixed receiver-tree anchor IDs/roles changed")
            tree = CaseLocalReceiverTree(universe, fixed_tree.nodes, fixed_tree.overlap_fraction, fixed_tree.capacity_saturated,
                fixed_tree.measure_consistent,
                canonical_receiver_index(coords, mass, roles) if fixed_tree.measure_consistent else None)
        depths = []
        node_depths = {0: 0}
        for index, node in enumerate(tree.nodes):
            depth = node_depths[index]
            depths.append(depth)
            if node.left is not None:
                node_depths[node.left] = depth + 1
                node_depths[node.right] = depth + 1
        # Batch the same weighted node reductions. The fixed gather follows
        # each node's original anchor order; padding contributes exact zeros.
        # One transfer builds all integer node rows, instead of many tiny
        # device indexing/reduction launches for every hierarchy node.
        width = max(len(node.anchor_indices) for node in tree.nodes)
        rows = [list(node.anchor_indices) + [0] * (width - len(node.anchor_indices)) for node in tree.nodes]
        indices = torch.tensor(rows, device=coords.device, dtype=torch.long)
        valid = torch.arange(width, device=coords.device)[None] < torch.tensor(
            [len(node.anchor_indices) for node in tree.nodes], device=coords.device)[:, None]
        node_mass = torch.where(valid, mass[indices], torch.zeros_like(mass[indices]))
        selected_coords = coords[indices]
        centers = _pool(selected_coords, node_mass, exact_mass=self.measure_consistent)
        low, high = _receiver_node_extrema(tree, coords, indices, valid)
        extent = high - low
        receiver_summary = _pool(receiver_states[indices], node_mass, exact_mass=self.measure_consistent)
        role_summary = _pool(F.one_hot(roles[indices], 8).to(states.dtype), node_mass, exact_mass=self.measure_consistent)
        module_mass, env_mass = measures["M"][case].sum(), measures["E"][case].sum()
        statistics = torch.stack((module_mass.log1p(), env_mass.log1p(), (module_mass > 0).to(states.dtype),
                                  (env_mass > 0).to(states.dtype)))
        count = len(tree.nodes)
        descriptors = torch.cat((receiver_summary, summaries["M"][case].expand(count, -1),
            summaries["E"][case].expand(count, -1), global_token[case].expand(count, -1),
            pad_geometry(centers / case_scale), pad_geometry(extent / case_scale), statistics.expand(count, -1),
            centers.new_tensor(depths)[:, None] / max(self.max_depth, 1), phase_features.expand(count, -1), role_summary), -1)
        return tree, centers, self.node_encoder(descriptors) if encode_nodes else descriptors, depths

    def _content_control(self, tau, summaries, masses, geometry, context, phase_features):
        """Control content has exactly the declared donors and prescribed context."""
        statistics = torch.stack((masses["M"].log1p(), masses["E"].log1p(),
            (masses["M"] > 0).to(geometry.dtype), (masses["E"] > 0).to(geometry.dtype)), -1)
        inputs = [summaries["M"], summaries["E"], geometry, statistics.to(geometry.dtype)]
        if self.faithful_controls:
            inputs.append(context)
        inputs.append(phase_features.expand(*geometry.shape[:-1], -1))
        return self.control_heads[tau](torch.cat(inputs, -1))

    def recompute_controls(self, state: TypedHypergraphState, encoded: EncodedInterfaceCase,
                           module_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """Diagnostic fixed-membership attribution, not an inverse freeze policy.

        Hold both donor weights, planner decisions and receiver geometry fixed;
        recompute source content and prescribed context. An omitted P0 donor
        then has no control path. P1/P2 source states retain upstream ancestry.
        """
        if not self.faithful_controls or not state.control_memberships:
            raise ValueError("Conditional content attribution requires faithful controls.")
        sources = {"M": self._continuous_input(module_states), "E": self._continuous_input(encoded.env_tokens)}
        phase_features = F.one_hot(torch.tensor(state.phase, device=module_states.device), 3).to(module_states.dtype)
        result = {}
        for tau in MECHANISMS:
            summaries, masses = {}, {}
            for kind in ("M", "E"):
                mass = state.control_memberships[tau][kind] * state.source_measures[kind][:, None]
                summaries[kind] = _pool(sources[kind][:, None].expand(-1, mass.shape[1], -1, -1), mass, exact_mass=True)
                masses[kind] = mass.sum(-1)
            result[tau] = self._content_control(tau, summaries, masses,
                state.strategy_data["control_geometry"][tau],
                self._continuous_input(encoded.global_token)[:, None].expand(-1, state.group_count, -1), phase_features)
            result[tau] = torch.where(state.strategy_data["node_valid"][tau][..., None], result[tau], torch.zeros_like(result[tau]))
        return result

    def prepare(self, encoded: EncodedInterfaceCase, module_states: torch.Tensor,
                *, phase: int = 0, soft: bool = False, capture_topology: bool = False,
                fixed_topology: TypedHypergraphState | None = None,
                topology_mode: str = "fixed_active_set") -> TypedHypergraphState:
        """Prepare ordinary actions or an explicitly named local continuation.

        Historical fixed-active-set replay continues the source projection
        affinely on its recorded positive set and can reject negative density.
        ``fixed_frontier_live_membership`` retains tree identities and split
        gates while projecting current source scores normally. Its donor
        support may change; it is not the historical affine continuation.
        Receiver connectivity is bound separately by ``access`` replay.
        """
        if topology_mode not in {"fixed_active_set", "fixed_frontier_live_membership"}:
            raise ValueError(f"Unknown topology continuation mode: {topology_mode!r}")
        if fixed_topology is None and topology_mode != "fixed_active_set":
            raise ValueError("Fixed-frontier continuation requires a reference topology")
        phase = int(phase)
        if phase not in (0, 1, 2):
            raise ValueError("physical phase must be 0, 1 or 2")
        if encoded.module_centers.shape[-1] not in (2, 3):
            raise ValueError("receiver hierarchies support 2-D and 3-D geometry")
        catalogue = build_source_catalogue(encoded, module_states)
        if fixed_topology is not None:
            if self.training or soft or fixed_topology.phase != phase:
                raise ValueError("Fixed topology is evaluation-only and requires a matching hard physical phase")
            validate_catalogue(catalogue, fixed_topology)
        sources = {"M": self._continuous_input(module_states), "E": self._continuous_input(encoded.env_tokens)}
        global_token = self._continuous_input(encoded.global_token)
        measures = catalogue["source_measures"]
        sources = {kind: torch.where(catalogue["source_valid"][kind][..., None], z,
                                     torch.zeros_like(z)) for kind, z in sources.items()}
        summaries = {kind: _pool(sources[kind], measures[kind], soft_precision=soft,
                                exact_mass=self.measure_consistent) for kind in ("M", "E")}
        phase_features = F.one_hot(torch.tensor(phase, device=module_states.device), 3).to(module_states.dtype)
        batch = module_states.shape[0]
        memberships, controls, typed_centres, trees, gates, split_logits, typed_admission = {}, {}, {}, {}, {}, {}, {}
        frontier_counts = {}
        exercise_depth = None if soft else self._exercise_depth()
        scale = encoded.coordinate_scale.reshape(-1, encoded.module_centers.shape[-1])
        if scale.shape[0] not in (1, batch):
            raise ValueError("coordinate scale must be global or supplied once per case")
        # Geometry remains case-owned. Encode all real node descriptors in
        # one call; typed physics and source normalizations stay independent.
        index_cache = {}
        geometry_cache = {}
        for tau in ("MM", "EM", "QM"):
            receiver_kind = "M" if tau == "MM" else "E" if tau == "EM" else "Q"
            for case in range(batch):
                case_scale = scale[0 if scale.shape[0] == 1 else case]
                index_cache[case, receiver_kind] = self._index(encoded, module_states, case, tau, case_scale,
                    summaries, global_token, measures, phase_features,
                    None if fixed_topology is None else fixed_topology.strategy_data["trees"][tau][case],
                    encode_nodes=False)
                if self.faithful_controls:
                    descriptor = index_cache[case, receiver_kind][2]
                    offset = 4 * self.hidden_dim
                    geometry_cache[case, receiver_kind] = self.control_geometry_encoder(torch.cat((
                        descriptor[:, offset:offset + 12], descriptor[:, offset + 16:offset + 17],
                        descriptor[:, -8:]), -1))
        counts = [len(item[0].nodes) for item in index_cache.values()]
        node_embeddings = self.node_encoder(torch.cat([item[2] for item in index_cache.values()])).split(counts)
        for key, embedding in zip(index_cache, node_embeddings):
            tree, centre, _, depths = index_cache[key]
            index_cache[key] = tree, centre, embedding, depths
        all_source_logits, all_control_memberships = ({}, {}) if capture_topology else (None, None)
        source_embeddings = {kind: self.source_encoder(z) for kind, z in sources.items()}
        valid_coords = {kind: torch.where(catalogue["source_valid"][kind][..., None], catalogue["source_coords"][kind],
                                        torch.zeros_like(catalogue["source_coords"][kind])) for kind in ("M", "E")}
        case_scale = scale.expand(batch, -1)[:, None, None]
        control_memberships, control_presence, control_geometry, all_node_valid = {}, {}, {}, {}
        for tau in MECHANISMS:
            receiver_kind = "M" if tau in ("MM", "ME") else "E" if tau == "EM" else "Q"
            indices = [index_cache[case, receiver_kind] for case in range(batch)]
            route_trees = tuple(item[0] for item in indices)
            lengths = [len(tree.nodes) for tree in route_trees]
            node_cases = torch.tensor([case for case, length in enumerate(lengths) for _ in range(length)],
                                      device=module_states.device)
            packed_centers = torch.cat([item[1] for item in indices])
            packed_embeddings = torch.cat([item[2] for item in indices])
            packed_geometry = torch.cat([geometry_cache[case, receiver_kind] for case in range(batch)]) if self.faithful_controls else packed_embeddings
            node_valid = torch.arange(self.capacity, device=module_states.device)[None] < torch.tensor(
                lengths, device=module_states.device)[:, None]
            centers = torch.stack([F.pad(item[1], (0, 0, 0, self.capacity-length)) for item, length in zip(indices, lengths)])
            embeddings = torch.stack([F.pad(item[2], (0, 0, 0, self.capacity-length)) for item, length in zip(indices, lengths)])
            logits = self.split_head(embeddings).squeeze(-1)
            logits = torch.where(node_valid, logits, torch.zeros_like(logits))
            leaf_mask = torch.tensor([[not node.is_leaf for node in tree.nodes] + [False]*(self.capacity-length)
                                      for tree, length in zip(route_trees, lengths)], device=logits.device)
            if fixed_topology is not None:
                split = fixed_topology.strategy_data["gates"][tau].to(logits)
            elif soft:
                split = torch.sigmoid(logits.to(torch.float64) / self.temperature) * leaf_mask
            elif exercise_depth is not None:
                depth_mask = torch.tensor([[depth < exercise_depth for depth in item[3]] + [False]*(self.capacity-length)
                                          for item, length in zip(indices, lengths)], device=logits.device)
                split = depth_mask.to(logits.dtype) * leaf_mask
            else:
                split = (logits >= 0).to(logits.dtype) * leaf_mask
            typed_membership, typed_summary, typed_mass, score_features, distances = {}, {}, {}, {}, {}
            for kind, type_index in (("M", 0), ("E", 1)):
                z = source_embeddings[kind][node_cases]
                relative = (packed_centers[:, None] - valid_coords[kind][node_cases]) / case_scale[:, 0][node_cases]
                kind_flag = logits.new_tensor([float(type_index == 0), float(type_index == 1)]).expand(relative.shape[:-1] + (2,))
                score_features[kind] = torch.cat((packed_embeddings[:, None].expand(-1, z.shape[1], -1),
                    z, pad_geometry(relative), kind_flag), -1)
                distances[kind] = relative.square().sum(-1)
            source_counts = [sources[kind].shape[1] for kind in ("M", "E")]
            raw_scores = self.source_scores[tau](torch.cat((score_features["M"], score_features["E"]), 1)).squeeze(-1).split(source_counts, 1)
            recorded_logits, recorded_memberships = ({}, {}) if capture_topology else (None, None)
            for kind, raw in zip(("M", "E"), raw_scores):
                scores = raw - self.geometry_strength[tau] * distances[kind]
                if fixed_topology is None or topology_mode == "fixed_frontier_live_membership":
                    density = _membership(scores[:, None], measures[kind][node_cases], soft=soft, temperature=self.temperature)[:, 0]
                else:
                    densities = []
                    for case, case_scores in enumerate(scores.split(lengths)):
                        reference_logits = fixed_topology.strategy_data["all_source_logits"][tau][kind][case]
                        reference_density = fixed_topology.strategy_data["all_control_memberships"][tau][kind][case]
                        mu = measures[kind][case] / measures[kind][case].sum().clamp_min(1e-12)
                        density = fixed_active_projection(case_scores, reference_logits, reference_density, mu)
                        densities.append(density)
                    density = torch.cat(densities)
                if capture_topology:
                    recorded_logits[kind] = list(scores.split(lengths))
                    recorded_memberships[kind] = list(density.split(lengths))
                member_mass = density * measures[kind][node_cases]
                typed_membership[kind] = density
                typed_summary[kind] = _pool(sources[kind][node_cases], member_mass, soft_precision=soft,
                                            exact_mass=self.faithful_controls)
                typed_mass[kind] = member_mass.sum(-1)
            control = self._content_control(tau, typed_summary, typed_mass, packed_geometry,
                                           global_token[node_cases], phase_features)
            controls[tau] = torch.stack([F.pad(value, (0, 0, 0, self.capacity-length))
                                        for value, length in zip(control.split(lengths), lengths)])
            control_memberships[tau] = {
                donor: torch.stack([torch.cat((value, value.new_zeros((self.capacity-length, value.shape[-1]))), 0)
                    for value, length in zip(typed_membership[donor].split(lengths), lengths)])
                for donor in ("M", "E")
            }
            control_presence[tau] = {
                donor: torch.stack([F.pad(value > 0, (0, self.capacity-length))
                    for value, length in zip(typed_mass[donor].split(lengths), lengths)])
                for donor in ("M", "E")
            }
            all_node_valid[tau] = node_valid
            if self.faithful_controls:
                control_geometry[tau] = torch.stack([F.pad(value, (0, 0, 0, self.capacity-length))
                    for value, length in zip(packed_geometry.split(lengths), lengths)])
            kind = "M" if tau in ("MM", "EM", "QM") else "E"
            memberships[tau] = torch.stack([torch.cat((value, value.new_zeros((self.capacity-length, value.shape[-1]))), 0)
                                           for value, length in zip(typed_membership[kind].split(lengths), lengths)])
            typed_centres[tau], trees[tau], gates[tau], split_logits[tau] = centers, route_trees, split, logits
            counts = []
            split_values = split.detach().cpu().tolist()
            for tree, values in zip(route_trees, split_values):
                pending, frontier = [0], 0
                while pending:
                    index = pending.pop()
                    node = tree.nodes[index]
                    if node.is_leaf or values[index] == 0:
                        frontier += 1
                    else:
                        pending.extend((node.left, node.right))
                counts.append(frontier)
            frontier_counts[tau] = module_states.new_tensor(counts)
            typed_admission[tau] = torch.stack([_frontier_admission(tree, split[case], self.capacity)
                                               for case, tree in enumerate(route_trees)])
            if capture_topology:
                all_source_logits[tau], all_control_memberships[tau] = recorded_logits, recorded_memberships
        access_geometry = {kind: build_receiver_tree_geometry(trees[tau], self.capacity)
                           for kind, tau in (("M", "MM"), ("E", "EM"), ("Q", "QM"))}
        admission = typed_admission["QM"]
        diagnostics = {"allocated_capacity": module_states.new_full((batch,), self.capacity),
                       "exploration": module_states.new_full((batch,), float(exercise_depth is not None)),
                       **{f"{tau}_frontier_groups": count for tau, count in frontier_counts.items()}}
        probe_data = {"all_source_logits": all_source_logits, "all_control_memberships": all_control_memberships} if capture_topology else {}
        provenance = {
            "mode": "membership_local_v1" if self.faithful_controls else "legacy_rich_node_v1",
            "planning": ("all current module states", "all current environmental states", "receiver state/geometry/role",
                         "prescribed global context" if self.faithful_controls else "adapter global context (may include aggregate heat)", "physical phase"),
            "value_donors": {tau: "M" if tau in ("MM", "EM", "QM") else "E" for tau in MECHANISMS},
            "control_donors": {tau: ("M", "E") for tau in MECHANISMS},
            "control_content": ("membership-weighted module state", "membership-weighted environmental state",
                "receiver geometry/role", "explicit donor mass/presence", "prescribed global context", "physical phase")
                if self.faithful_controls else ("membership-weighted M/E state", "rich all-source planner embedding with adapter global context/source summaries", "phase"),
            "upstream_ancestry": "P0 input encodings" if phase == 0 else
                "phase-current module/environment states may contain earlier transport, predicted ports and Stage-A refinement",
            "interpretation": "conditional computational dependencies; learned support is not physical causality",
        }
        return TypedHypergraphState(memberships=memberships, controls=controls, centres=typed_centres["QM"],
                                    admission=admission, phase=phase, diagnostics=diagnostics,
                                    control_memberships=control_memberships, control_presence=control_presence,
                                    dependency_provenance=provenance,
                                    strategy_data={"trees": trees, "gates": gates, "split_logits": split_logits,
                                                   "typed_centres": typed_centres, "typed_admission": typed_admission,
                                                   "soft": soft, "access_geometry": access_geometry,
                                                   "measure_consistent": self.measure_consistent,
                                                   "structural_measure_policy_version": self.structural_measure_policy_version,
                                                   "topology_continuation_mode": topology_mode if fixed_topology is not None else "ordinary",
                                                   "control_geometry": control_geometry, "node_valid": all_node_valid,
                                                   **probe_data}, **catalogue)

    def access(self, state: TypedHypergraphState, receivers: torch.Tensor, mechanism: str,
               receiver_tokens: torch.Tensor | None = None, *, soft: bool = False,
               pair_valid: torch.Tensor | None = None, capture_topology: bool = False,
               fixed_receiver_access: dict | None = None, prepared_action=None, include_diagnostics=True, detach_permissions=False) -> TypedSourceAccess:
        del receiver_tokens, soft
        tau = str(mechanism).upper()
        if tau not in MECHANISMS:
            raise ValueError(f"unknown typed mechanism {mechanism!r}")
        receiver_valid = None
        receiver_kind = "M" if tau in ("MM", "ME") else "E" if tau == "EM" else None
        if receiver_kind is not None and receivers.shape[:2] == state.source_valid[receiver_kind].shape:
            receiver_valid = state.source_valid[receiver_kind]
            receivers = torch.where(receiver_valid[..., None], receivers, torch.zeros_like(receivers))
        index_kind = "M" if tau in ("MM", "ME") else "E" if tau == "EM" else "Q"
        edge_access = receiver_tree_access(receivers, state.strategy_data["gates"][tau],
                                    state.strategy_data["access_geometry"][index_kind], self.max_depth,
                                    soft_precision=bool(state.strategy_data["soft"]))
        if receiver_valid is not None:
            edge_access = torch.where(receiver_valid[..., None], edge_access, torch.zeros_like(edge_access))
        if fixed_receiver_access is not None:
            active = fixed_receiver_access["edge_access"].to(edge_access) > 0
            if bool((active & (edge_access <= 0)).any()):
                raise FixedTopologyInvalid("Recorded tree receiver connection reached zero along the local path")
            edge_access = torch.where(active, edge_access, torch.zeros_like(edge_access))
            total = edge_access.sum(-1, keepdim=True)
            original_total = fixed_receiver_access["edge_access"].to(edge_access).sum(-1, keepdim=True)
            edge_access = edge_access * (original_total / torch.where(total > 0, total, torch.ones_like(total)))
        kind = "M" if tau in ("MM", "EM", "QM") else "E"
        if tau == "MM" and pair_valid is None:
            if receivers.shape[1] != state.source_coords["M"].shape[1]:
                raise ValueError("MM access requires the canonical physical module receiver axis")
            pair_valid = state.source_valid["M"][:, :, None] & ~torch.eye(receivers.shape[1], device=receivers.device, dtype=torch.bool)[None]
        elif tau == "ME" and pair_valid is None and receivers.shape[1] == state.source_valid["M"].shape[1]:
            pair_valid = state.source_valid["M"][:, :, None]
        elif tau == "EM" and pair_valid is None and receivers.shape[1] == state.source_valid["E"].shape[1]:
            pair_valid = state.source_valid["E"][:, :, None]
        if detach_permissions:
            edge_access = edge_access.detach()
        access = source_moments(edge_access, state.memberships[tau], state.controls[tau],
                              state.source_measures[kind], state.source_valid[kind], pair_valid=pair_valid, prepared_action=prepared_action, include_diagnostics=include_diagnostics)
        if self.measure_consistent:
            # Native M/E routes use their owned physical measure. Queried
            # points use an explicitly declared equal-point counting measure.
            receiver_measure = state.source_measures[index_kind] if index_kind in ("M", "E") else receivers.new_ones(receivers.shape[:2])
            access.diagnostics["receiver_measures"] = receiver_measure
        return access


__all__ = ["AdaptiveReceiverHypergraph"]

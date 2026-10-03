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

from .adaptive_interaction_cover import CaseLocalReceiverTree, ReceiverAnchorUniverse
from .routing_index.sparse_projection import source_measure_sparsemax
from .typed_hypergraph_state import (
    TypedHypergraphState,
    TypedSourceAccess,
    build_source_catalogue,
    pad_geometry,
    source_moments,
)
from .types import EncodedInterfaceCase

MECHANISMS = ("MM", "ME", "EM", "QM", "QE")


def _pool(states: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    denominator = mass.sum(-1, keepdim=True)
    safe = torch.where(mass[..., None] > 0, states, torch.zeros_like(states))
    return (safe * mass[..., None]).sum(-2) / denominator.clamp_min(1e-12)


def _membership(scores: torch.Tensor, measure: torch.Tensor, *, soft: bool,
                temperature: float) -> torch.Tensor:
    """Measure-aware density: splitting an identical source atom preserves it."""
    normalized = measure / measure.sum(-1, keepdim=True).clamp_min(1e-12)
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
    positive = torch.where(valid, positive.clamp_min(torch.finfo(scores.dtype).tiny), positive)
    denominator = (positive * normalized[:, None]).sum(-1, keepdim=True)
    return positive / denominator.clamp_min(torch.finfo(scores.dtype).tiny)


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


class AdaptiveReceiverHypergraph(nn.Module):
    """Five typed receiver hierarchies, independent of requested query batches.

    ``set_epoch`` selects the absolute-epoch curriculum. Its counters are
    persistent buffers and ordinary checkpoint/RNG resume preserves it.
    Evaluation always selects the deterministic learned frontier.
    """

    def __init__(self, hidden_dim: int, *, spatial_dim: int | None = None,
                 organizer_dim: int = 64, control_dim: int = 16,
                 max_depth: int = 3, overlap_fraction: float = 0.06,
                 temperature: float = 1.0) -> None:
        super().__init__()
        if spatial_dim not in (None, 2, 3):
            raise ValueError("receiver hierarchies support spatial dimensions 2 and 3")
        if min(hidden_dim, organizer_dim, control_dim) < 1 or not 0 <= max_depth <= 3:
            raise ValueError("positive dimensions and receiver depth between zero and three are required")
        if temperature <= 0 or not 0 < overlap_fraction < 1:
            raise ValueError("positive temperature and overlap fraction in (0,1) are required")
        self.hidden_dim = int(hidden_dim)
        self.control_dim = int(control_dim)
        self.max_depth = int(max_depth)
        self.capacity = 2 ** (max_depth + 1) - 1
        self.overlap_fraction = float(overlap_fraction)
        self.temperature = float(temperature)
        self.node_encoder = nn.Sequential(nn.Linear(4 * hidden_dim + 28, organizer_dim),
                                          nn.GELU(), nn.Linear(organizer_dim, organizer_dim), nn.GELU())
        self.source_encoder = nn.Linear(hidden_dim, organizer_dim)
        self.split_head = nn.Linear(organizer_dim, 1)
        self.source_scores = nn.ModuleDict({tau: nn.Sequential(
            nn.Linear(2 * organizer_dim + 8, organizer_dim), nn.GELU(),
            nn.Linear(organizer_dim, 1)) for tau in MECHANISMS})
        self.control_heads = nn.ModuleDict({tau: nn.Sequential(
            nn.Linear(2 * hidden_dim + organizer_dim + 7, organizer_dim), nn.GELU(),
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
            return encoded.env_coords[case, valid], encoded.env_weights[case, valid], encoded.env_tokens[case, valid].detach()
        coords, mass = encoded.receiver_anchor_coords, encoded.receiver_anchor_weights
        if coords is None:
            valid = encoded.env_weights[case] > 0
            return encoded.env_coords[case, valid], encoded.env_weights[case, valid], encoded.env_tokens[case, valid].detach()
        coords = coords if coords.ndim == 2 else coords[case]
        mass = coords.new_ones(coords.shape[0]) if mass is None else (mass if mass.ndim == 1 else mass[case])
        valid = mass > 0
        tokens = encoded.global_token[case].detach().expand(int(valid.sum()), -1)
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

    def _index(self, encoded, states, case, tau, case_scale, summaries, global_token, measures, phase_features):
        coords, mass, receiver_states = self._catalogue(encoded, states.detach(), case, tau)
        roles = self._receiver_roles(encoded, case, tau, coords.shape[0])
        if coords.shape[0] == 0:
            coords = states.new_zeros((1, encoded.module_centers.shape[-1]))
            mass = coords.new_ones(1)
            roles = torch.zeros(1, device=coords.device, dtype=torch.long)
            receiver_states = states.new_zeros((1, self.hidden_dim))
        tree = CaseLocalReceiverTree.build(ReceiverAnchorUniverse(coords, mass, roles, case_scale),
            max_nodes=self.capacity, min_leaf_anchors=1, overlap_fraction=self.overlap_fraction,
            max_depth=self.max_depth)
        centers, descriptors, depths = [], [], []
        node_depths = {0: 0}
        for index, node in enumerate(tree.nodes):
            ids = list(node.anchor_indices)
            node_mass = mass[ids]
            center = _pool(coords[ids], node_mass)
            extent = coords[ids].amax(0) - coords[ids].amin(0)
            receiver_summary = _pool(receiver_states[ids], node_mass)
            role_summary = _pool(F.one_hot(roles[ids], 8).to(states.dtype), node_mass)
            statistics = torch.stack((measures["M"][case].sum().log1p(), measures["E"][case].sum().log1p(),
                                      (measures["M"][case].sum() > 0).to(states.dtype),
                                      (measures["E"][case].sum() > 0).to(states.dtype)))
            depth = node_depths[index]
            descriptors.append(torch.cat((receiver_summary, summaries["M"][case], summaries["E"][case], global_token[case],
                                          pad_geometry(center / case_scale), pad_geometry(extent / case_scale), statistics,
                                          center.new_tensor([depth / max(self.max_depth, 1)]), phase_features, role_summary)))
            centers.append(center)
            depths.append(depth)
            if node.left is not None:
                node_depths[node.left] = depth + 1
                node_depths[node.right] = depth + 1
        return tree, torch.stack(centers), self.node_encoder(torch.stack(descriptors)), depths

    def prepare(self, encoded: EncodedInterfaceCase, module_states: torch.Tensor,
                *, phase: int = 0, soft: bool = False) -> TypedHypergraphState:
        phase = int(phase)
        if phase not in (0, 1, 2):
            raise ValueError("physical phase must be 0, 1 or 2")
        if encoded.module_centers.shape[-1] not in (2, 3):
            raise ValueError("receiver hierarchies support 2-D and 3-D geometry")
        catalogue = build_source_catalogue(encoded, module_states)
        sources = {"M": module_states.detach(), "E": encoded.env_tokens.detach()}
        global_token = encoded.global_token.detach()
        measures = catalogue["source_measures"]
        sources = {kind: torch.where(catalogue["source_valid"][kind][..., None], z,
                                     torch.zeros_like(z)) for kind, z in sources.items()}
        summaries = {kind: _pool(sources[kind], measures[kind]) for kind in ("M", "E")}
        phase_features = F.one_hot(torch.tensor(phase, device=module_states.device), 3).to(module_states.dtype)
        batch = module_states.shape[0]
        memberships, controls, typed_centres, trees, gates, split_logits, typed_admission = {}, {}, {}, {}, {}, {}, {}
        frontier_counts = {}
        exercise_depth = None if soft else self._exercise_depth()
        scale = encoded.coordinate_scale.reshape(-1, encoded.module_centers.shape[-1])
        if scale.shape[0] not in (1, batch):
            raise ValueError("coordinate scale must be global or supplied once per case")
        # Three case-local catalogues; mechanisms retain separate source
        # permissions/controls. No state is cached across coupling phases.
        index_cache = {}
        source_embeddings = {kind: self.source_encoder(z) for kind, z in sources.items()}
        for tau in MECHANISMS:
            route_memberships, route_controls, route_centres, route_trees, route_gates, route_logits, counts = [], [], [], [], [], [], []
            for case in range(batch):
                case_scale = scale[0 if scale.shape[0] == 1 else case]
                receiver_kind = "M" if tau in ("MM", "ME") else "E" if tau == "EM" else "Q"
                key = case, receiver_kind
                if key not in index_cache:
                    index_cache[key] = self._index(encoded, module_states, case, tau, case_scale,
                                                   summaries, global_token, measures, phase_features)
                tree, centers, embeddings, depths = index_cache[key]
                logits = self.split_head(embeddings).squeeze(-1)
                leaf_mask = torch.tensor([not node.is_leaf for node in tree.nodes], device=logits.device)
                if soft:
                    split = torch.sigmoid(logits / self.temperature) * leaf_mask
                elif exercise_depth is not None:
                    split = torch.tensor([depth < exercise_depth for depth in depths], device=logits.device, dtype=logits.dtype) * leaf_mask
                else:
                    split = (logits >= 0).to(logits.dtype) * leaf_mask
                typed_membership, typed_summary, typed_mass = {}, {}, {}
                for kind, type_index in (("M", 0), ("E", 1)):
                    z = source_embeddings[kind][case]
                    source_coords = torch.where(catalogue["source_valid"][kind][case, :, None],
                                                catalogue["source_coords"][kind][case],
                                                torch.zeros_like(catalogue["source_coords"][kind][case]))
                    relative = (centers[:, None] - source_coords[None]) / case_scale
                    kind_flag = F.one_hot(torch.tensor(type_index, device=logits.device), 2).to(logits.dtype).expand(relative.shape[:2] + (2,))
                    score_features = torch.cat((embeddings[:, None].expand(-1, z.shape[0], -1),
                                                z[None].expand(len(tree.nodes), -1, -1), pad_geometry(relative), kind_flag), -1)
                    scores = self.source_scores[tau](score_features).squeeze(-1) - self.geometry_strength[tau] * relative.square().sum(-1)
                    density = _membership(scores[None], measures[kind][case:case+1], soft=soft,
                                          temperature=self.temperature)[0]
                    member_mass = density * measures[kind][case, None]
                    typed_membership[kind] = density
                    typed_summary[kind] = _pool(sources[kind][case, None].expand(len(tree.nodes), -1, -1), member_mass)
                    typed_mass[kind] = member_mass.sum(-1)
                statistics = torch.stack((typed_mass["M"].log1p(), typed_mass["E"].log1p(),
                                          (typed_mass["M"] > 0).to(logits.dtype), (typed_mass["E"] > 0).to(logits.dtype)), -1)
                control = self.control_heads[tau](torch.cat((typed_summary["M"], typed_summary["E"], embeddings,
                                                            statistics, phase_features.expand(len(tree.nodes), -1)), -1))
                kind = "M" if tau in ("MM", "EM", "QM") else "E"
                pad = self.capacity - len(tree.nodes)
                membership = typed_membership[kind]
                route_memberships.append(torch.cat((membership, membership.new_zeros((pad, membership.shape[-1]))), 0))
                route_controls.append(F.pad(control, (0, 0, 0, pad)))
                route_centres.append(F.pad(centers, (0, 0, 0, pad)))
                route_trees.append(tree)
                route_gates.append(F.pad(split, (0, pad)))
                route_logits.append(F.pad(logits, (0, pad)))
                # Count the actual frontier, excluding descendants of closed nodes.
                pending, frontier = [0], 0
                while pending:
                    index = pending.pop()
                    node = tree.nodes[index]
                    if node.is_leaf or float(split[index].detach()) == 0:
                        frontier += 1
                    else:
                        pending.extend((node.left, node.right))
                counts.append(frontier)
            memberships[tau], controls[tau] = torch.stack(route_memberships), torch.stack(route_controls)
            typed_centres[tau], trees[tau] = torch.stack(route_centres), tuple(route_trees)
            gates[tau], split_logits[tau] = torch.stack(route_gates), torch.stack(route_logits)
            frontier_counts[tau] = module_states.new_tensor(counts)
            typed_admission[tau] = torch.stack([_frontier_admission(tree, gates[tau][case], self.capacity)
                                                 for case, tree in enumerate(trees[tau])])
        admission = typed_admission["QM"]
        diagnostics = {"allocated_capacity": module_states.new_full((batch,), self.capacity),
                       "exploration": module_states.new_full((batch,), float(exercise_depth is not None)),
                       **{f"{tau}_frontier_groups": count for tau, count in frontier_counts.items()}}
        return TypedHypergraphState(memberships=memberships, controls=controls, centres=typed_centres["QM"],
                                    admission=admission, phase=phase, diagnostics=diagnostics,
                                    strategy_data={"trees": trees, "gates": gates, "split_logits": split_logits,
                                                   "typed_centres": typed_centres, "typed_admission": typed_admission,
                                                   "soft": soft}, **catalogue)

    def access(self, state: TypedHypergraphState, receivers: torch.Tensor, mechanism: str,
               receiver_tokens: torch.Tensor | None = None, *, soft: bool = False,
               pair_valid: torch.Tensor | None = None) -> TypedSourceAccess:
        del receiver_tokens, soft
        tau = str(mechanism).upper()
        if tau not in MECHANISMS:
            raise ValueError(f"unknown typed mechanism {mechanism!r}")
        receiver_valid = None
        receiver_kind = "M" if tau in ("MM", "ME") else "E" if tau == "EM" else None
        if receiver_kind is not None and receivers.shape[:2] == state.source_valid[receiver_kind].shape:
            receiver_valid = state.source_valid[receiver_kind]
            receivers = torch.where(receiver_valid[..., None], receivers, torch.zeros_like(receivers))
        access = []
        for case, tree in enumerate(state.strategy_data["trees"][tau]):
            weights = tree.access(receivers[case], state.strategy_data["gates"][tau][case, :len(tree.nodes)])
            access.append(F.pad(weights, (0, self.capacity - len(tree.nodes))))
        edge_access = torch.stack(access)
        if receiver_valid is not None:
            edge_access = torch.where(receiver_valid[..., None], edge_access, torch.zeros_like(edge_access))
        kind = "M" if tau in ("MM", "EM", "QM") else "E"
        if tau == "MM" and pair_valid is None:
            if receivers.shape[1] != state.source_coords["M"].shape[1]:
                raise ValueError("MM access requires the canonical physical module receiver axis")
            pair_valid = state.source_valid["M"][:, :, None] & ~torch.eye(receivers.shape[1], device=receivers.device, dtype=torch.bool)[None]
        elif tau == "ME" and pair_valid is None and receivers.shape[1] == state.source_valid["M"].shape[1]:
            pair_valid = state.source_valid["M"][:, :, None]
        elif tau == "EM" and pair_valid is None and receivers.shape[1] == state.source_valid["E"].shape[1]:
            pair_valid = state.source_valid["E"][:, :, None]
        return source_moments(edge_access, state.memberships[tau], state.controls[tau],
                              state.source_measures[kind], state.source_valid[kind], pair_valid=pair_valid)


__all__ = ["AdaptiveReceiverHypergraph"]

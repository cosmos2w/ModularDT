"""Historical per-case receiver organizer equations for first-gradient parity."""
import torch
from torch.nn import functional as F

from honf_forward_core.interface_fields.adaptive_receiver_hypergraph import _frontier_admission, _membership, _pool
from honf_forward_core.interface_fields.topology_probe import fixed_active_projection, validate_catalogue
from honf_forward_core.interface_fields.typed_hypergraph_state import (
    MECHANISMS,
    TypedHypergraphState,
    build_source_catalogue,
    pad_geometry,
)
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def case_loop_prepare(self, encoded: EncodedInterfaceCase, module_states: torch.Tensor,
            *, phase: int = 0, soft: bool = False, capture_topology: bool = False,
            fixed_topology: TypedHypergraphState | None = None) -> TypedHypergraphState:
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
    all_source_logits, all_control_memberships = ({}, {}) if capture_topology else (None, None)
    source_embeddings = {kind: self.source_encoder(z) for kind, z in sources.items()}
    for tau in MECHANISMS:
        route_memberships, route_controls, route_centres, route_trees, route_gates, route_logits, counts = [], [], [], [], [], [], []
        recorded_logits, recorded_memberships = ({"M": [], "E": []}, {"M": [], "E": []}) if capture_topology else (None, None)
        for case in range(batch):
            case_scale = scale[0 if scale.shape[0] == 1 else case]
            receiver_kind = "M" if tau in ("MM", "ME") else "E" if tau == "EM" else "Q"
            key = case, receiver_kind
            if key not in index_cache:
                index_cache[key] = self._index(encoded, module_states, case, tau, case_scale,
                                               summaries, global_token, measures, phase_features,
                                               None if fixed_topology is None else fixed_topology.strategy_data["trees"][tau][case])
            tree, centers, embeddings, depths = index_cache[key]
            logits = self.split_head(embeddings).squeeze(-1)
            leaf_mask = torch.tensor([not node.is_leaf for node in tree.nodes], device=logits.device)
            if fixed_topology is not None:
                split = fixed_topology.strategy_data["gates"][tau][case, :len(tree.nodes)].to(logits)
            elif soft:
                split = torch.sigmoid(logits.to(torch.float64) / self.temperature) * leaf_mask
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
                if fixed_topology is None:
                    density = _membership(scores[None], measures[kind][case:case+1], soft=soft,
                                          temperature=self.temperature)[0]
                else:
                    reference_logits = fixed_topology.strategy_data["all_source_logits"][tau][kind][case]
                    reference_density = fixed_topology.strategy_data["all_control_memberships"][tau][kind][case]
                    mu = measures[kind][case] / measures[kind][case].sum().clamp_min(1e-12)
                    density = fixed_active_projection(scores, reference_logits, reference_density, mu)
                if capture_topology:
                    recorded_logits[kind].append(scores)
                    recorded_memberships[kind].append(density)
                member_mass = density * measures[kind][case, None]
                typed_membership[kind] = density
                typed_summary[kind] = _pool(sources[kind][case, None].expand(len(tree.nodes), -1, -1), member_mass,
                                            soft_precision=soft)
                typed_mass[kind] = member_mass.sum(-1)
            statistics = torch.stack((typed_mass["M"].log1p(), typed_mass["E"].log1p(),
                                      (typed_mass["M"] > 0).to(logits.dtype), (typed_mass["E"] > 0).to(logits.dtype)), -1)
            control = self.control_heads[tau](torch.cat((typed_summary["M"], typed_summary["E"], embeddings,
                                                        statistics.to(logits.dtype), phase_features.expand(len(tree.nodes), -1)), -1))
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
            split_values = split.detach().cpu().tolist()
            while pending:
                index = pending.pop()
                node = tree.nodes[index]
                if node.is_leaf or split_values[index] == 0:
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
        if capture_topology:
            all_source_logits[tau], all_control_memberships[tau] = recorded_logits, recorded_memberships
    admission = typed_admission["QM"]
    diagnostics = {"allocated_capacity": module_states.new_full((batch,), self.capacity),
                   "exploration": module_states.new_full((batch,), float(exercise_depth is not None)),
                   **{f"{tau}_frontier_groups": count for tau, count in frontier_counts.items()}}
    probe_data = {"all_source_logits": all_source_logits, "all_control_memberships": all_control_memberships} if capture_topology else {}
    return TypedHypergraphState(memberships=memberships, controls=controls, centres=typed_centres["QM"],
                                admission=admission, phase=phase, diagnostics=diagnostics,
                                strategy_data={"trees": trees, "gates": gates, "split_logits": split_logits,
                                               "typed_centres": typed_centres, "typed_admission": typed_admission,
                                               "soft": soft, **probe_data}, **catalogue)

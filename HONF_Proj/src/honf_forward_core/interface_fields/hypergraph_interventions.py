"""Explicit same-weight controls for typed plans; never a training policy."""

from dataclasses import replace

import torch

from .typed_hypergraph_state import SOURCE_TYPE, TypedHypergraphState


def membership_intervention(state: TypedHypergraphState, mode: str) -> TypedHypergraphState:
    """Preserve membership multisets while changing their physical assignment.

    Geometry ranks sources by distance to each physical group centre; rewire
    cyclically moves active source columns. Exchange moves group source sets
    while keeping receiver access and controls fixed. Unequal source measures
    can change group mass and are reported by the evaluator.
    """
    if mode not in {"geometry", "rewire", "exchange"}:
        return state
    members = {}
    for mechanism, original in state.memberships.items():
        kind = SOURCE_TYPE[mechanism]
        if mode == "exchange":
            changed = original.clone()
            admission = state.strategy_data.get("typed_admission", {}).get(mechanism, state.admission)
            for case in range(original.shape[0]):
                active = torch.nonzero(admission[case] > 0, as_tuple=False).flatten()
                if active.numel():
                    changed[case, active] = original[case, active].roll(1, dims=0)
            members[mechanism] = changed
            continue
        changed = torch.zeros_like(original)
        centres = state.strategy_data.get("typed_centres", {}).get(mechanism, state.centres)
        for case in range(original.shape[0]):
            indices = torch.nonzero(state.source_valid[kind][case], as_tuple=False).flatten()
            if not indices.numel():
                continue
            values = original[case, :, indices]
            if mode == "rewire":
                changed[case, :, indices] = values.roll(1, dims=-1)
            else:
                distance = (centres[case, :, None] - state.source_coords[kind][case, indices][None]).square().sum(-1)
                nearest = torch.argsort(distance, dim=-1, stable=True)
                sorted_values = torch.sort(values, dim=-1, descending=True, stable=True).values
                reassigned = torch.zeros_like(values).scatter(-1, nearest, sorted_values)
                changed[case, :, indices] = reassigned
        members[mechanism] = changed
    return replace(state, memberships=members)


def fixed_structure_intervention(state: TypedHypergraphState, *, count: int = 2, depth: int = 1) -> TypedHypergraphState:
    """A labelled fixed frontier/admission intervention at unchanged weights."""
    data = dict(state.strategy_data)
    if "trees" in data:
        from .adaptive_receiver_hypergraph import _frontier_admission
        gates, admitted = {}, {}
        for mechanism, trees in data["trees"].items():
            gates[mechanism] = torch.zeros_like(data["gates"][mechanism])
            for case, tree in enumerate(trees):
                pending = [(0, 0)]
                while pending:
                    index, level = pending.pop()
                    node = tree.nodes[index]
                    gates[mechanism][case, index] = float(not node.is_leaf and level < depth)
                    if not node.is_leaf:
                        pending.extend(((node.left, level + 1), (node.right, level + 1)))
            admitted[mechanism] = torch.stack([
                _frontier_admission(tree, gates[mechanism][case], state.group_count)
                for case, tree in enumerate(trees)
            ])
        data.update(gates=gates, typed_admission=admitted)
        return replace(state, admission=admitted["QM"], strategy_data=data)
    logits = data["gate_logits"]
    admission = torch.zeros_like(state.admission)
    chosen = logits.topk(min(max(1, count), logits.shape[-1]), dim=-1).indices
    admission.scatter_(-1, chosen, 1.0)
    return replace(state, admission=admission)


__all__ = ["fixed_structure_intervention", "membership_intervention"]

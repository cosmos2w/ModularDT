"""Observed-only bounded proposals, physical feasibility and charged work."""

import pytest
import torch

from honf_inverse_core.heat_inference import bounded_trust_heat_inference
from honf_forward_core.interface_fields.topology_probe import FixedTopologyInvalid


def _predictor(heat):
    return {"observed": heat[:2], "held": heat[2:], "groups": ([0, 1],),
            "peaks": heat + 1, "pressure": heat.sum()}


def _args():
    return dict(predictor=_predictor, observed=torch.tensor([.2, 1.8]), held=torch.tensor([0.]),
        active=torch.ones(3, dtype=torch.bool), total=torch.tensor(3.), initial_fraction=torch.ones(3) / 3,
        steps=10, learning_rate=.05)


def test_trust_trail_is_feasible_monotone_observed_and_held_blind():
    first = bounded_trust_heat_inference(**_args())
    changed = {**_args(), "held": torch.tensor([1000.])}
    second = bounded_trust_heat_inference(**changed)
    assert torch.equal(torch.stack(first.heat), torch.stack(second.heat))
    assert first.optimizer_steps == 10 and first.accepted_steps + first.rejected_steps == 10
    assert len(first.heat) == 11 and first.vjp_calls == tuple(range(11))
    assert first.forward_calls[-1] == 1 + 10 + sum(first.trial_evaluations)
    assert max(first.trial_evaluations) <= 2
    assert first.observed_rmse[-1] < first.observed_rmse[0]
    assert all(a >= b for a, b in zip(first.observed_rmse, first.observed_rmse[1:]))
    for index, heat in enumerate(first.heat):
        torch.testing.assert_close(heat.sum(), torch.tensor(3.))
        assert (heat >= 0).all()
        if index:
            assert ((heat - first.heat[index-1]) / 3.).abs().max() <= .0500001


def test_invalid_topology_trials_reject_keep_design_and_charge_attempted_calls():
    count = 0

    def invalid(_heat, _reference):
        nonlocal count
        count += 1
        raise FixedTopologyInvalid("outside native active-set region")

    trail = bounded_trust_heat_inference(**{**_args(), "steps": 3}, proposal_predictor=invalid)
    assert count == 6 and trail.accepted_steps == 0 and trail.rejected_steps == 3
    assert trail.forward_calls[-1] == 10 and trail.vjp_calls[-1] == 3
    assert torch.equal(torch.stack(trail.heat), trail.heat[0].expand(4, -1))
    assert all(not row["topology_valid"] for receipt in trail.step_receipts for row in receipt["trials"])


def test_graph_random_blocks_match_sizes_and_preserve_outside_allocations():
    args = {**_args(), "steps": 3,
        "permutation_stream": (torch.tensor([1, 2, 0]),) * 3}
    graph = bounded_trust_heat_inference(**args, mode="graph")
    sizes = tuple(int(block.numel()) for block in graph.selected_modules)
    random = bounded_trust_heat_inference(**args, mode="ungrouped", block_size_stream=sizes)
    assert sizes == tuple(int(block.numel()) for block in random.selected_modules)
    for trail in (graph, random):
        for step, block in enumerate(trail.selected_modules):
            outside = torch.ones(3, dtype=torch.bool)
            outside[block] = False
            torch.testing.assert_close(trail.heat[step+1][outside], trail.heat[step][outside])


def test_ordinary_rebuild_regression_does_not_get_accepted():
    def false_improvement(heat, _reference):
        return {"observed": torch.tensor([.2, 1.8]), "held": heat[2:]}

    # The gradient proposes movement while the ordinary real predictor moves
    # uphill; an optimistic frozen-trial surrogate must not bypass acceptance.
    calls = 0

    def switched(heat):
        nonlocal calls
        calls += 1
        return {"observed": heat[:2] if calls == 1 else heat[:2] + 100., "held": heat[2:]}

    trail = bounded_trust_heat_inference(**{**_args(), "predictor": switched, "steps": 1}, proposal_predictor=false_improvement)
    assert trail.accepted_steps == 0 and trail.rejected_steps == 1
    torch.testing.assert_close(trail.heat[0], trail.heat[1])
    assert all("rejected_reason" in row for row in trail.step_receipts[0]["trials"])


def test_step_and_radius_scope_are_explicit():
    with pytest.raises(ValueError, match="at most ten"):
        bounded_trust_heat_inference(**{**_args(), "steps": 11})

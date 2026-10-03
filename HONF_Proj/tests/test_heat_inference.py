"""Fixed-total identifiability and constrained surrogate optimization."""

import pytest
import torch

from honf_inverse_core.heat_inference import (
    fixed_total_basis,
    fixed_total_heat_inference,
    observation_identifiability,
)


def test_fixed_total_basis_keeps_physical_padding_and_orthonormality():
    active = torch.tensor([True, False, True, True])
    basis = fixed_total_basis(active, dtype=torch.float64)
    assert basis.shape == (4, 2)
    torch.testing.assert_close(basis.sum(0), torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(basis.T @ basis, torch.eye(2, dtype=torch.float64))
    assert not basis[1].any()


def test_rank_detects_nonunique_observations_before_inference():
    active = torch.ones(3, dtype=torch.bool)
    heat = torch.ones(3, requires_grad=True)
    informative = observation_identifiability(lambda value: {"observed": value[:2]}, heat, active)
    assert informative["rank"] == 2 and informative["free_dimensions"] == 2
    ambiguous = observation_identifiability(lambda value: {"observed": value.sum().reshape(1)}, heat, active)
    assert ambiguous["rank"] == 0
    assert ambiguous["condition_number"] == float("inf")


def _predictor(heat):
    return {"observed": heat[:2], "held": heat[2:], "peaks": heat + 2,
            "pressure": heat.sum().reshape(1), "groups": ([0, 1], [1, 2])}


def test_joint_recovery_has_complete_feasible_trail_and_live_heat_gradient():
    trail = fixed_total_heat_inference(_predictor, torch.tensor([.2, .8]), torch.tensor([2.]),
        torch.ones(3, dtype=torch.bool), torch.tensor(3.), torch.ones(3) / 3, steps=30, learning_rate=.04)
    assert len(trail.heat) == 31 and trail.optimizer_steps == 30
    assert trail.iterations == tuple(range(31))
    assert trail.observed_rmse[-1] < trail.observed_rmse[0]
    for heat in trail.heat:
        assert (heat >= 0).all()
        torch.testing.assert_close(heat.sum(), torch.tensor(3.))
    assert len(trail.peaks) == len(trail.pressure) == 31
    assert all(left <= right for left, right in zip(trail.elapsed_seconds, trail.elapsed_seconds[1:]))


def test_group_and_random_blocks_share_size_budget_and_keep_untouched_heat_fixed():
    common = {"predictor": _predictor, "observed": torch.tensor([.2, .8]), "held": torch.tensor([2.]),
        "active": torch.ones(3, dtype=torch.bool), "total": torch.tensor(3.), "initial_fraction": torch.ones(3) / 3,
        "steps": 4, "block_stream": (0., .9, 0., .9),
        "permutation_stream": (torch.tensor([0, 2, 1]),) * 4}
    graph = fixed_total_heat_inference(**common, mode="graph")
    ungrouped = fixed_total_heat_inference(**common, mode="ungrouped")
    assert graph.meaningful_graph_steps == ungrouped.meaningful_graph_steps == 4
    assert graph.group_fallback_steps == ungrouped.group_fallback_steps == 0
    for trail in (graph, ungrouped):
        for index, selected in enumerate(trail.selected_modules):
            assert selected.numel() == 2
            outside = torch.ones(3, dtype=torch.bool)
            outside[selected] = False
            torch.testing.assert_close(trail.heat[index+1][outside], trail.heat[index][outside])


def test_single_module_is_deterministic_zero_free_dimension():
    active = torch.tensor([True])
    identifiability = observation_identifiability(lambda value: {"observed": value}, torch.ones(1), active)
    assert identifiability["free_dimensions"] == 0 and identifiability["rank"] == 0
    trail = fixed_total_heat_inference(lambda value: {"observed": value, "held": value},
        torch.tensor([3.]), torch.tensor([3.]), active, torch.tensor(3.), torch.tensor([1.]), steps=30)
    assert trail.optimizer_steps == 0 and trail.iterations == (0,)
    torch.testing.assert_close(trail.heat[0], torch.tensor([3.]))


def test_missing_gradient_and_bad_permutation_are_not_accepted():
    with pytest.raises(RuntimeError):
        fixed_total_heat_inference(lambda heat: {"observed": heat[:1].detach(), "held": heat[1:]},
            torch.zeros(1), torch.zeros(2), torch.ones(3, dtype=torch.bool), torch.tensor(1.),
            torch.ones(3) / 3, steps=1)
    with pytest.raises(ValueError, match="permute"):
        fixed_total_heat_inference(_predictor, torch.zeros(2), torch.zeros(1), torch.ones(3, dtype=torch.bool),
            torch.tensor(1.), torch.ones(3) / 3, mode="ungrouped", steps=1,
            permutation_stream=(torch.tensor([0, 0, 2]),))

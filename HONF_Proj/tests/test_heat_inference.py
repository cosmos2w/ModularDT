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
    matched = fixed_total_heat_inference(lambda value: {"observed": value, "held": value},
        torch.tensor([3.]), torch.tensor([3.]), active, torch.tensor(3.), torch.tensor([1.]),
        mode="ungrouped", steps=30, block_size_stream=())
    assert matched.optimizer_steps == 0


def test_recorded_graph_cardinalities_match_even_when_candidate_groups_change():
    def conditional(heat):
        group = [0, 1] if float(heat[0].detach()) >= 1 else [0, 1, 2]
        return {"observed": heat[:2], "held": heat[2:], "groups": (group,)}

    arguments = {"predictor": conditional, "observed": torch.tensor([.2, 1.8]), "held": torch.tensor([1., 1.]),
        "active": torch.ones(4, dtype=torch.bool), "total": torch.tensor(4.), "initial_fraction": torch.ones(4) / 4,
        "steps": 5, "permutation_stream": (torch.tensor([3, 2, 1, 0]),) * 5}
    graph = fixed_total_heat_inference(**arguments, mode="graph")
    sizes = tuple(int(group.numel()) for group in graph.selected_modules)
    assert set(sizes) == {2, 3}
    ungrouped = fixed_total_heat_inference(**arguments, mode="ungrouped", block_size_stream=sizes)
    assert tuple(int(group.numel()) for group in ungrouped.selected_modules) == sizes
    unmatched = fixed_total_heat_inference(**arguments, mode="ungrouped")
    assert tuple(int(group.numel()) for group in unmatched.selected_modules) != sizes
    for step, chosen in enumerate(ungrouped.selected_modules):
        outside = torch.ones(4, dtype=torch.bool)
        outside[chosen] = False
        torch.testing.assert_close(ungrouped.heat[step+1][outside], ungrouped.heat[step][outside])
    with pytest.raises(ValueError, match="Recorded block sizes"):
        fixed_total_heat_inference(**arguments, mode="ungrouped", block_size_stream=(2,))


def test_missing_gradient_and_bad_permutation_are_not_accepted():
    with pytest.raises(RuntimeError):
        fixed_total_heat_inference(lambda heat: {"observed": heat[:1].detach(), "held": heat[1:]},
            torch.zeros(1), torch.zeros(2), torch.ones(3, dtype=torch.bool), torch.tensor(1.),
            torch.ones(3) / 3, steps=1)
    with pytest.raises(ValueError, match="permute"):
        fixed_total_heat_inference(_predictor, torch.zeros(2), torch.zeros(1), torch.ones(3, dtype=torch.bool),
            torch.tensor(1.), torch.ones(3) / 3, mode="ungrouped", steps=1,
            permutation_stream=(torch.tensor([0, 0, 2]),))


def test_full_joint_fallback_matches_joint_with_a_permuted_control_stream():
    def nonlinear(heat):
        return {"observed": (heat[:3] + heat[:3].square()), "held": heat[3:], "groups": ()}

    arguments = {"predictor": nonlinear, "observed": torch.tensor([.2, .8, .5]),
        "held": torch.tensor([.5]), "active": torch.ones(4, dtype=torch.bool), "total": torch.tensor(2.),
        "initial_fraction": torch.tensor([.100001, .299999, .350001, .249999]), "steps": 30,
        "permutation_stream": (torch.tensor([2, 0, 3, 1]),) * 30}
    joint = fixed_total_heat_inference(**arguments, mode="joint")
    graph = fixed_total_heat_inference(**arguments, mode="graph")
    random = fixed_total_heat_inference(**arguments, mode="ungrouped", block_size_stream=(4,) * 30)
    for trail in (graph, random):
        assert trail.group_fallback_steps == 30 and trail.meaningful_graph_steps == 0
        assert torch.equal(torch.stack(trail.heat), torch.stack(joint.heat))
        assert torch.equal(torch.stack(trail.observed_predictions), torch.stack(joint.observed_predictions))

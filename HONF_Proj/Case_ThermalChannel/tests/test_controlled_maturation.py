from __future__ import annotations

from types import SimpleNamespace

import torch

from channelthermal.response_control.training import (
    StagedTrainingConfig,
    TrainingStage,
    run_staged_fit,
)
from channelthermal.response_control.losses import ThermalLossScales
from channelthermal.response_control.maturation import (
    THERMAL_ACTION_PATHS,
    ThermalMaturationSchedule,
    available_frontier_for_paths,
)
from test_response_control import _AbsoluteField, _stencil


def _tree(*children: tuple[int | None, int | None]):
    return SimpleNamespace(nodes=tuple(SimpleNamespace(left=left, right=right) for left, right in children))


def test_maturation_uses_two_scaffold_passes_then_two_passes_per_action() -> None:
    families = tuple(f"family-{index}" for index in range(8))
    schedule = ThermalMaturationSchedule(families, start_update=200, seed=11)

    plans = [schedule.plan(200 + update) for update in range(100)]
    sparse = [plan for plan in plans if not plan.full_access_replay]
    assert len(sparse) == 80
    assert sum(plan.full_access_replay for plan in plans) == 20
    assert [plan.relative_update for plan in plans if plan.full_access_replay] == list(range(4, 100, 5))

    for primary_pass in range(8):
        pass_rows = [plan for plan in sparse if plan.primary_pass == primary_pass]
        assert {plan.family_id for plan in pass_rows} == set(families)
        assert len(pass_rows) == len(families)
    assert all(plan.action == "two_packet_scaffold" for plan in sparse if plan.primary_pass < 2)
    for primary_pass, expected_action in zip(range(2, 8), ("root", "two_packet", "four_packet") * 2, strict=True):
        pass_rows = [plan for plan in sparse if plan.primary_pass == primary_pass]
        assert {plan.action for plan in pass_rows} == {expected_action}
        assert all(plan.requested_cut_paths == THERMAL_ACTION_PATHS[expected_action] for plan in pass_rows)


def test_full_replay_reuses_prior_family_and_does_not_consume_sparse_pass() -> None:
    schedule = ThermalMaturationSchedule(("a", "b", "c"), start_update=200)
    sparse_before = [schedule.plan(200 + update) for update in range(4)]
    replay = schedule.plan(204)
    next_sparse = schedule.plan(205)

    assert replay.full_access_replay is True
    assert replay.capacity_fraction == 1.0
    assert replay.family_id == sparse_before[-1].family_id
    assert replay.relative_sparse_update == sparse_before[-1].relative_sparse_update
    assert next_sparse.full_access_replay is False
    assert next_sparse.relative_sparse_update == 4


def test_early_leaf_resolves_to_available_complete_cut_without_increasing_k() -> None:
    # Root splits, but its left branch is already a physical leaf.
    early_left = _tree((1, 2), (None, None), (3, 4), (None, None), (None, None))
    frontier, paths = available_frontier_for_paths(
        early_left, THERMAL_ACTION_PATHS["four_packet"]
    )
    assert frontier == (1, 3, 4)
    assert paths == ("L", "RL", "RR")
    assert len(frontier) < len(THERMAL_ACTION_PATHS["four_packet"])

    root_only = _tree((None, None))
    frontier, paths = available_frontier_for_paths(
        root_only, THERMAL_ACTION_PATHS["four_packet"]
    )
    assert frontier == (0,)
    assert paths == ("",)


def test_early_leaf_left_refined_right_coarse_preserves_left_to_right_cut_order() -> None:
    # Root splits; its left child has two children while its right child is a
    # terminal leaf. Native node order is LL=3, LR=4, R=2, not numeric order.
    left_refined = _tree(
        (1, 2),
        (3, 4),
        (None, None),
        (None, None),
        (None, None),
    )
    frontier, paths = available_frontier_for_paths(
        left_refined, THERMAL_ACTION_PATHS["four_packet"]
    )
    assert frontier == (3, 4, 2)
    assert paths == ("LL", "LR", "R")


def test_duplicate_family_ids_are_rejected() -> None:
    try:
        ThermalMaturationSchedule(("same", "same"))
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate response families must not silently collapse the exposure denominator")


def test_auxiliary_anchor_changes_update_in_blockwise_projection_branch() -> None:
    scales = ThermalLossScales(
        value={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        finite={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        mixed={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        pressure_value=1.0,
        pressure_response=1.0,
        pressure_limit=3.0,
        pressure_boundary=0.5,
        solid_temperature=1.0,
        smooth_peak_beta=1.0,
        near_limit_band=0.5,
    )
    config = StagedTrainingConfig(
        arm="R_response",
        max_optimizer_updates=1,
        max_epochs=1,
        total_optimizer_update_ceiling=1,
        checkpoint_every_updates=1,
        review_updates=(1,),
        project_response_gradient_blockwise=True,
        stages=(TrainingStage("projected_response", 0, 1, ("value", "finite")),),
    )
    starting_state = _AbsoluteField().state_dict()

    def fit(with_anchor: bool) -> float:
        model = _AbsoluteField()
        model.load_state_dict(starting_state)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-5)

        def anchor_loss(_completed, _stencil, _predictions, _terms):
            if not with_anchor:
                return None, {}
            # Nonzero train-only incumbent-distortion gradient at this start.
            return 100.0 * torch.square(model.gain - 0.1), {"anchor_loss": 100.0 * float((model.gain.detach() - 0.1).square())}

        run_staged_fit(
            model,
            model,
            optimizer,
            [_stencil()],
            scales=scales,
            loss_weights={"value": 1.0, "finite": 1.0},
            config=config,
            auxiliary_loss_fn=anchor_loss,
        )
        return float(model.gain.detach())

    without_anchor = fit(False)
    with_anchor = fit(True)
    assert with_anchor != without_anchor

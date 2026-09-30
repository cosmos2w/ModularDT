from __future__ import annotations

from types import SimpleNamespace

from windfarm.workflows.maturation import (
    BASE_LANE_GPU_BUDGET_SECONDS,
    BASE_RECIPE_STAGE,
    MAX_LANE_GPU_BUDGET_SECONDS,
    QE_FULL_RECIPE_STAGE,
    REMEDY_ACTION_ORDER,
    WIND_ACTION_PATHS,
    WindMaturationSchedule,
    available_frontier_for_paths,
    sparse_capacity_for_recipe,
    validated_lane_gpu_budget_seconds,
)


def _tree(*children: tuple[int | None, int | None]):
    return SimpleNamespace(nodes=tuple(SimpleNamespace(left=left, right=right) for left, right in children))


def test_wind_maturation_has_two_scaffold_then_two_complete_passes_per_action() -> None:
    rows = tuple(range(408))
    schedule = WindMaturationSchedule(
        rows,
        start_update=100,
        seed=13,
        remedy_start_after_update=None,
    )
    plans = [schedule.plan(100 + update) for update in range(4080)]
    sparse = [plan for plan in plans if not plan.full_access_replay]
    replays = [plan for plan in plans if plan.full_access_replay]

    assert len(sparse) == 3264
    assert len(replays) == 816
    assert all(plan.capacity_vector == (("QE", 0.95), ("MM", 0.90)) for plan in sparse)
    assert all(plan.capacity_vector == (("QE", 1.0), ("MM", 1.0)) for plan in replays)
    assert [plan.relative_update for plan in replays[:4]] == [4, 9, 14, 19]

    for primary_pass in range(8):
        pass_rows = [plan for plan in sparse if plan.primary_pass == primary_pass]
        assert len(pass_rows) == len(rows)
        assert {plan.row_id for plan in pass_rows} == set(rows)
        assert len({plan.row_id for plan in pass_rows}) == len(rows)

    assert all(plan.action == "two_packet_scaffold" for plan in sparse if plan.primary_pass < 2)
    expected = ("root", "two_packet", "four_packet", "root", "two_packet", "four_packet")
    for primary_pass, expected_action in zip(range(2, 8), expected, strict=True):
        pass_rows = [plan for plan in sparse if plan.primary_pass == primary_pass]
        assert {plan.action for plan in pass_rows} == {expected_action}
        assert all(plan.requested_cut_paths == WIND_ACTION_PATHS[expected_action] for plan in pass_rows)


def test_replay_reuses_row_but_samples_a_fresh_query_without_consuming_pass() -> None:
    schedule = WindMaturationSchedule((10, 20, 30), start_update=100, seed=17)
    prior = schedule.plan(103)
    replay = schedule.plan(104)
    next_sparse = schedule.plan(105)

    assert prior.full_access_replay is False
    assert replay.full_access_replay is True
    assert replay.row_id == prior.row_id
    assert replay.relative_sparse_update == prior.relative_sparse_update
    assert replay.replay_sparse_update == prior.relative_sparse_update
    assert replay.query_seed != prior.query_seed
    assert next_sparse.relative_sparse_update == prior.relative_sparse_update + 1


def test_remedy_switch_preserves_old_stream_and_starts_the_versioned_recipe() -> None:
    rows = tuple(range(408))
    amended = WindMaturationSchedule(rows, start_update=100, seed=2112)
    historical = WindMaturationSchedule(
        rows,
        start_update=100,
        seed=2112,
        remedy_start_after_update=None,
    )
    assert [amended.plan(update) for update in range(100, 1850)] == [
        historical.plan(update) for update in range(100, 1850)
    ]

    last_old = amended.plan(1849)
    first_remedy = amended.plan(1850)
    assert last_old.recipe_stage == BASE_RECIPE_STAGE
    assert last_old.full_access_replay
    assert last_old.relative_sparse_update == 1399
    assert first_remedy.recipe_stage == QE_FULL_RECIPE_STAGE
    assert first_remedy.phase == "action_family"
    assert first_remedy.action == "four_packet"
    assert first_remedy.primary_pass == 4
    assert first_remedy.pass_position == 0
    assert first_remedy.recipe_pass == 0
    assert first_remedy.recipe_pass_position == 0
    assert first_remedy.relative_sparse_update == 1400
    assert first_remedy.capacity_vector == (("QE", 1.0), ("MM", 0.90))
    assert first_remedy.sparse_capacity_vector == sparse_capacity_for_recipe(QE_FULL_RECIPE_STAGE)
    assert first_remedy.query_seed == 2112 + 1750 * 104_729


def test_remedy_schedule_has_six_complete_action_passes_and_4_to_1_replays() -> None:
    rows = tuple(range(408))
    schedule = WindMaturationSchedule(rows, start_update=100, seed=2112)
    plans = [schedule.plan(1850 + offset) for offset in range(3060)]
    sparse = [plan for plan in plans if not plan.full_access_replay]
    replays = [plan for plan in plans if plan.full_access_replay]

    assert len(sparse) == 2448
    assert len(replays) == 612
    assert tuple(
        next(plan.action for plan in sparse if plan.recipe_pass == recipe_pass)
        for recipe_pass in range(6)
    ) == REMEDY_ACTION_ORDER
    assert all(plan.recipe_stage == QE_FULL_RECIPE_STAGE for plan in plans)
    assert all(plan.phase == "action_family" for plan in sparse)
    assert all(plan.capacity_vector == (("QE", 1.0), ("MM", 0.90)) for plan in sparse)
    assert all(plan.capacity_vector == (("QE", 1.0), ("MM", 1.0)) for plan in replays)
    assert [plan.relative_sparse_update for plan in sparse] == list(range(1400, 3848))
    assert len({plan.query_seed for plan in plans}) == 3060

    for recipe_pass, expected_action in enumerate(REMEDY_ACTION_ORDER):
        pass_rows = [plan for plan in sparse if plan.recipe_pass == recipe_pass]
        assert len(pass_rows) == len(rows)
        assert {plan.row_id for plan in pass_rows} == set(rows)
        assert len({plan.row_id for plan in pass_rows}) == len(rows)
        assert {plan.primary_pass for plan in pass_rows} == {recipe_pass + 4}
        assert {plan.action for plan in pass_rows} == {expected_action}
        assert all(plan.phase == "action_family" for plan in pass_rows)

    for index, replay in enumerate(plans):
        if not replay.full_access_replay:
            continue
        previous = plans[index - 1]
        assert previous.row_id == replay.row_id
        assert previous.relative_sparse_update == replay.relative_sparse_update
        assert previous.query_seed != replay.query_seed
        assert replay.replay_sparse_update == previous.relative_sparse_update
    assert plans[-1].full_access_replay
    assert plans[-1].relative_update == 4809
    try:
        schedule.plan(4910)
    except ValueError as exc:
        assert "complete at u4910" in str(exc)
    else:
        raise AssertionError("the bounded remedy schedule must stop at u4910")


def test_lane_gpu_budget_defaults_to_14_hours_and_caps_reallocation_at_120_percent() -> None:
    assert BASE_LANE_GPU_BUDGET_SECONDS == 50_400
    assert MAX_LANE_GPU_BUDGET_SECONDS == 60_480
    assert validated_lane_gpu_budget_seconds({}) == BASE_LANE_GPU_BUDGET_SECONDS

    extension = 2_400
    manifest = {
        "lane_gpu_budget_seconds": BASE_LANE_GPU_BUDGET_SECONDS + extension,
        "resource_reallocation_history": [{
            "old_lane_gpu_budget_seconds": BASE_LANE_GPU_BUDGET_SECONDS,
            "new_lane_gpu_budget_seconds": BASE_LANE_GPU_BUDGET_SECONDS + extension,
            "donor_allocation": "measured lane reserve",
            "donor_allocation_seconds": extension,
            "measured_pilot_forecast_seconds": 2_200,
            "evidence_path": "diagnostics/pilot_forecast.json",
            "recorded_at_utc": "2026-09-30T00:00:00+00:00",
        }],
    }
    assert validated_lane_gpu_budget_seconds(manifest) == BASE_LANE_GPU_BUDGET_SECONDS + extension

    at_limit_extension = MAX_LANE_GPU_BUDGET_SECONDS - BASE_LANE_GPU_BUDGET_SECONDS
    at_limit = {
        "lane_gpu_budget_seconds": MAX_LANE_GPU_BUDGET_SECONDS,
        "resource_reallocation_history": [{
            **manifest["resource_reallocation_history"][0],
            "new_lane_gpu_budget_seconds": MAX_LANE_GPU_BUDGET_SECONDS,
            "donor_allocation_seconds": at_limit_extension,
        }],
    }
    assert validated_lane_gpu_budget_seconds(at_limit) == MAX_LANE_GPU_BUDGET_SECONDS

    invalid = dict(manifest, lane_gpu_budget_seconds=MAX_LANE_GPU_BUDGET_SECONDS + 1)
    try:
        validated_lane_gpu_budget_seconds(invalid)
    except ValueError as exc:
        assert "between" in str(exc)
    else:
        raise AssertionError("Wind allocation must not exceed 120 percent of its base cap")

    unrecorded = {"lane_gpu_budget_seconds": BASE_LANE_GPU_BUDGET_SECONDS + 1}
    try:
        validated_lane_gpu_budget_seconds(unrecorded)
    except ValueError as exc:
        assert "history" in str(exc)
    else:
        raise AssertionError("a larger Wind allocation must have an explicit reallocation record")

    discontinuous = dict(manifest)
    discontinuous["resource_reallocation_history"] = [
        dict(manifest["resource_reallocation_history"][0], old_lane_gpu_budget_seconds=1)
    ]
    try:
        validated_lane_gpu_budget_seconds(discontinuous)
    except ValueError as exc:
        assert "incomplete or discontinuous" in str(exc)
    else:
        raise AssertionError("resource reallocation records must form an exact old-to-new chain")


def test_early_leaf_resolves_to_an_available_complete_cut() -> None:
    early_left = _tree((1, 2), (None, None), (3, 4), (None, None), (None, None))
    frontier, paths = available_frontier_for_paths(early_left, WIND_ACTION_PATHS["four_packet"])
    assert frontier == (1, 3, 4)
    assert paths == ("L", "RL", "RR")

    root_only = _tree((None, None))
    frontier, paths = available_frontier_for_paths(root_only, WIND_ACTION_PATHS["four_packet"])
    assert frontier == (0,)
    assert paths == ("",)


def test_left_refined_right_coarse_cut_uses_path_order_not_node_ids() -> None:
    # CaseLocalReceiverTree allocates both root children before the left
    # grandchildren, so the native cut's left-to-right order is (3, 4, 2).
    left_refined = _tree((1, 2), (3, 4), (None, None), (None, None), (None, None))
    frontier, paths = available_frontier_for_paths(left_refined, ("LL", "LR", "R"))

    assert frontier == (3, 4, 2)
    assert paths == ("LL", "LR", "R")


def test_duplicate_rows_are_rejected() -> None:
    try:
        WindMaturationSchedule((1, 2, 1))
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate row IDs would corrupt the without-replacement exposure clock")

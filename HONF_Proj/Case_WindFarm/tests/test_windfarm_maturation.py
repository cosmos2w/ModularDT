from __future__ import annotations

from types import SimpleNamespace

from windfarm.workflows.maturation import (
    WIND_ACTION_PATHS,
    WindMaturationSchedule,
    available_frontier_for_paths,
)


def _tree(*children: tuple[int | None, int | None]):
    return SimpleNamespace(nodes=tuple(SimpleNamespace(left=left, right=right) for left, right in children))


def test_wind_maturation_has_two_scaffold_then_two_complete_passes_per_action() -> None:
    rows = tuple(range(408))
    schedule = WindMaturationSchedule(rows, start_update=100, seed=13)
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

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "scripts"))

from run_selected_weight_controls import (
    FixedTrainPopulationOrganizer,
    TrainPopulationFeatures,
    _fraction_for_live_target,
    _full_access_packet_count_fields,
    _packet_support_summary,
    _preflight_native_forward_budget,
    _rewire_binary_matrix,
    _root_source_union_plan,
    _route_live_rows,
    _train_population_features,
    _validate_cuda_gpu_binding,
)

from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    CaseLocalReceiverTree,
    MechanismPlan,
    ReceiverAnchorUniverse,
)
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def _encoded(value: float) -> EncodedInterfaceCase:
    return EncodedInterfaceCase(
        module_tokens=torch.full((1, 2, 2), value),
        env_tokens=torch.full((1, 3, 2), value + 1),
        global_token=torch.full((1, 2), value + 2),
        module_centers=torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]),
        env_coords=torch.tensor([[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]]),
        module_present=torch.tensor([[1.0, 1.0]]),
        module_features=torch.full((1, 2, 1), value + 3),
        env_features=torch.full((1, 3, 1), value + 4),
        env_weights=torch.tensor([[1.0, 2.0, 1.0]]),
        coordinate_scale=torch.ones(2),
    )


def _population_capture(value: float) -> dict[str, object]:
    return {
        "encoded": _encoded(value),
        "p0_module_states": torch.full((1, 2, 2), value + 5),
    }


def test_train_population_uses_equal_family_weighted_input_means() -> None:
    captures = {f"family_{index:02d}": _population_capture(float(index)) for index in range(8)}

    population = _train_population_features(captures)

    assert population.family_ids == tuple(f"family_{index:02d}" for index in range(8))
    assert torch.equal(population.module_feature, torch.tensor([6.5], dtype=torch.float64))
    assert torch.equal(population.module_state, torch.tensor([8.5, 8.5], dtype=torch.float64))
    assert torch.equal(population.environment_feature, torch.tensor([7.5], dtype=torch.float64))
    assert torch.equal(population.global_token, torch.tensor([5.5, 5.5], dtype=torch.float64))


class _RecordingOrganizer:
    def __init__(self) -> None:
        self.encoded = None
        self.prepared_state = None
        self.plans_args = None

    def score_cases(self, encoded, prepared_state, trees, *, budgets=None):
        self.encoded = encoded
        self.prepared_state = prepared_state
        self.trees = trees
        self.budgets = budgets
        return ("scores",)

    def plans_from_scores(self, *args, **kwargs):
        self.plans_args = args
        return ("plans",)


def test_fixed_population_features_preserve_case_geometry_and_delegate_planning() -> None:
    current = _encoded(99.0)
    population = TrainPopulationFeatures(
        module_token=torch.tensor([1.0, 2.0]),
        environment_token=torch.tensor([3.0, 4.0]),
        global_token=torch.tensor([5.0, 6.0]),
        module_state=torch.tensor([7.0, 8.0]),
        environment_state=torch.tensor([9.0, 10.0]),
        module_feature=torch.tensor([11.0]),
        environment_feature=torch.tensor([12.0]),
        family_ids=tuple(f"family_{index:02d}" for index in range(8)),
    )
    route = _RecordingOrganizer()
    wrapper = FixedTrainPopulationOrganizer(route, population)
    current_state = {
        "module_states": torch.full((1, 2, 2), 100.0),
        "environment_states": torch.full((1, 3, 2), 101.0),
        "global_state": torch.full((1, 2), 102.0),
    }
    trees = (object(),)

    result = wrapper.score_cases(current, current_state, trees, budgets={"QE": 0.9})

    assert result == ("scores",)
    assert torch.equal(route.encoded.module_tokens, torch.tensor([[[1.0, 2.0], [1.0, 2.0]]]))
    assert torch.equal(route.encoded.env_tokens, torch.tensor([[[3.0, 4.0]] * 3]))
    assert torch.equal(route.encoded.global_token, torch.tensor([[5.0, 6.0]]))
    assert torch.equal(route.encoded.module_features, torch.full((1, 2, 1), 11.0))
    assert torch.equal(route.encoded.env_features, torch.full((1, 3, 1), 12.0))
    assert torch.equal(route.prepared_state["module_states"], torch.tensor([[[7.0, 8.0]] * 2]))
    assert torch.equal(route.prepared_state["environment_states"], torch.tensor([[[9.0, 10.0]] * 3]))
    assert torch.equal(route.prepared_state["global_state"], torch.tensor([[5.0, 6.0]]))
    for field in ("module_centers", "env_coords", "module_present", "env_weights", "coordinate_scale"):
        assert torch.equal(getattr(route.encoded, field), getattr(current, field))
    assert wrapper.plans_from_scores("scores", current, trees, hard=True) == ("plans",)
    assert route.plans_args[1] is current


def test_validity_aware_rewire_preserves_degrees_and_reports_effective_change() -> None:
    membership = np.asarray(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.int64,
    )
    active = np.ones(4, dtype=bool)
    valid = np.asarray([True, True, True, False])

    rewired, detail = _rewire_binary_matrix(membership, active, valid, seed=9)

    assert np.array_equal(rewired.sum(axis=1), membership.sum(axis=1))
    assert np.array_equal(rewired.sum(axis=0), membership.sum(axis=0))
    assert np.array_equal(rewired[:, ~valid], membership[:, ~valid])
    assert detail["row_degrees_preserved"]
    assert detail["column_degrees_preserved"]
    assert detail["invalid_source_columns_unchanged"]
    assert detail["accepted_switches"] > 0
    assert detail["changed_link_entries"] == int(np.count_nonzero(rewired != membership))
    assert detail["available"] == (detail["changed_link_entries"] > 0)


def test_rewire_reports_inactive_when_no_effective_switch_is_possible() -> None:
    membership = np.asarray([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.int64)
    rewired, detail = _rewire_binary_matrix(
        membership,
        np.ones(3, dtype=bool),
        np.ones(3, dtype=bool),
        seed=21,
    )

    assert np.array_equal(rewired, membership)
    assert detail["accepted_switches"] == 0
    assert detail["changed_link_entries"] == 0
    assert detail["status"] == "inactive_no_valid_switch"


def test_rewire_rejects_permissions_on_invalid_source_padding() -> None:
    membership = np.asarray([[1, 0], [0, 1]], dtype=np.int64)

    try:
        _rewire_binary_matrix(
            membership,
            np.ones(2, dtype=bool),
            np.asarray([True, False]),
            seed=5,
        )
    except ValueError as error:
        assert "Invalid source columns" in str(error)
    else:
        raise AssertionError("Invalid-source membership must be rejected before rewiring.")


def test_live_rows_prefer_executed_counters_without_double_counting_aliases() -> None:
    row = _route_live_rows({
        "cover_prepare_mm_executed_rows": 12,
        "cover_prepare_mm_actual_rows": 12,
        "cover_qe_actual_rows": 5,
        "cover_prepare_me_padded_rows": 99,
    })

    assert row["status"] == "measured_executor_rows"
    assert row["by_mechanism"]["MM"] == 12
    assert row["by_mechanism"]["QE"] == 5
    assert row["by_mechanism"]["ME"] is None
    assert row["measured_total_executor_rows"] == 17
    assert row["counter_keys"] == ["cover_prepare_mm_executed_rows", "cover_qe_actual_rows"]


def test_live_match_fraction_updates_use_measured_rows_only() -> None:
    assert _fraction_for_live_target(0.90, 45, 90) == 0.45
    assert _fraction_for_live_target(0.40, 0, 20) == 1.0e-3
    assert _fraction_for_live_target(0.40, 30, 0) == 0.8
    assert _fraction_for_live_target(0.90, 90, 45) == 1.0


def test_packet_support_reports_distinct_joint_k_and_raw_cut_separately(monkeypatch) -> None:
    import run_selected_weight_controls as controls

    observed = {}

    def fake_support(encoded, tree, plan, cut):
        observed["args"] = (encoded, tree, plan, tuple(cut))
        return (b"pattern-a", b"pattern-b"), {
            "joint_nonempty_permission_patterns": 2,
            "hard_permission_matrices_binary": True,
        }

    monkeypatch.setattr(controls.action_eval, "_support_signature_and_details", fake_support)
    encoded, tree, plan = object(), object(), object()
    result = _packet_support_summary(
        {"baseline": {"encoded": encoded, "trees": (tree,), "plans": (plan,)}},
        {"frontier": [1, 2, 3, 4]},
    )

    assert observed["args"] == (encoded, tree, plan, (1, 2, 3, 4))
    assert result["raw_packet_cut_size"] == 4
    assert result["nonredundant_packet_count"] == 2
    assert result["nonredundant_packet_count_status"] == "measured_combined_MM_QE_source_patterns"


def test_full_access_has_explicit_not_applicable_packet_k() -> None:
    fields = _full_access_packet_count_fields()

    assert fields["raw_packet_cut_size"] is None
    assert fields["nonredundant_packet_count"] is None
    assert fields["nonredundant_packet_count_status"] == "not_applicable_full_access"


def test_root_source_union_collapses_only_mm_qe_to_one_root_and_preserves_bypasses() -> None:
    encoded = _encoded(0.0)
    coords = torch.cat((encoded.env_coords[0], encoded.module_centers[0]), dim=0)
    universe = ReceiverAnchorUniverse(
        coordinates=coords,
        weights=torch.ones(coords.shape[0]),
        roles=torch.zeros(coords.shape[0], dtype=torch.long),
        coordinate_scale=torch.ones(2),
    )
    tree = CaseLocalReceiverTree.build(universe, max_nodes=7, min_leaf_anchors=1)
    gates = torch.zeros(len(tree.nodes))
    if not tree.nodes[0].is_leaf:
        gates[0] = 1.0
    mm = torch.zeros((len(tree.nodes), 2))
    qe = torch.zeros((len(tree.nodes), 3))
    mm[1] = torch.tensor([1.0, 0.0])
    mm[2] = torch.tensor([0.0, 1.0])
    qe[1] = torch.tensor([1.0, 0.0, 0.0])
    qe[2] = torch.tensor([0.0, 1.0, 0.0])
    plan = MechanismPlan(
        tree,
        gates,
        encoded.module_present[0],
        int(encoded.env_coords.shape[1]),
        permissions={"MM": mm, "QE": qe},
    )

    root, detail = _root_source_union_plan(plan, encoded, (1, 2))

    root_mm = root.permission_matrix("MM", module_present=encoded.module_present[0])
    root_qe = root.permission_matrix("QE")
    assert torch.equal(root_mm[0], torch.tensor([1.0, 1.0]))
    assert torch.equal(root_qe[0], torch.tensor([1.0, 1.0, 0.0]))
    assert torch.count_nonzero(root_mm[1:]) == 0
    assert torch.count_nonzero(root_qe[1:]) == 0
    assert float(root.split_gates[0]) == 0.0
    assert root.explicit_bypass_keys == plan.explicit_bypass_keys
    assert detail["MM"]["root_source_count"] == 2
    assert detail["QE"]["root_source_count"] == 2
    assert detail["root_split_closed"]
    assert detail["full_access_bypass_routes_preserved"] == sorted(plan.explicit_bypass_keys)


def test_selected_weight_preflight_counts_full_and_worst_case_sparse_state_calls() -> None:
    class Stencil:
        def __init__(self, family: str, state_count: int) -> None:
            self.physical_family_id = family
            self.records = tuple(range(state_count))

    train = [Stencil(f"train-{index}", 11) for index in range(8)]
    dev = [Stencil(f"dev-{index}", 10 + index) for index in range(2)]

    result = _preflight_native_forward_budget(
        train, dev, train_control_count=2, dev_control_count=2
    )

    expected = 11 + 8 * 11 + 21 + 43 + 12 * 43 + 12 * 43
    assert result["worst_case_native_state_forwards"] == expected
    assert result["G_sparse_controls_per_case"] == 12
    assert result["P_live_count_matched_actions_per_case"] == 3


def test_cuda_controls_require_manifest_lane_uuid_and_physical_gpu2_binding() -> None:
    manifest = {
        "physical_gpu": 2,
        "physical_gpu_uuid": "GPU-example",
        "cuda_visible_devices": "2",
    }
    lanes = [
        {"event": "arm_start", "arm": arm, "physical_gpu_index": 2,
         "physical_gpu_uuid": "GPU-example", "cuda_visible_devices": "2"}
        for arm in ("G", "P")
    ]

    result = _validate_cuda_gpu_binding(
        manifest,
        torch.device("cuda:0"),
        cuda_visible_devices="2",
        queried_identity=("GPU-example", "GPU test"),
        lane_rows=lanes,
    )

    assert result["status"] == "verified_physical_gpu2"
    assert result["physical_gpu_uuid"] == "GPU-example"
    assert result["lane_rows_verified"] == 2
    try:
        _validate_cuda_gpu_binding(
            manifest,
            torch.device("cuda:0"),
            cuda_visible_devices="2",
            queried_identity=("GPU-wrong", "GPU test"),
            lane_rows=lanes,
        )
    except ValueError as error:
        assert "differs from controlled manifest" in str(error)
    else:
        raise AssertionError("A wrong physical GPU UUID must fail closed.")
    bad_lanes = [dict(lanes[0]), {**lanes[1], "physical_gpu_uuid": "GPU-wrong"}]
    try:
        _validate_cuda_gpu_binding(
            manifest,
            torch.device("cuda:0"),
            cuda_visible_devices="2",
            queried_identity=("GPU-example", "GPU test"),
            lane_rows=bad_lanes,
        )
    except ValueError as error:
        assert "lane execution ledger" in str(error)
    else:
        raise AssertionError("A lane UUID mismatch must fail closed.")

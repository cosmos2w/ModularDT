from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import evaluate_matured_action_selector as selector

from honf_forward_core.interface_fields.action_risk_fit import ActionPolicyCaseResult


def test_policy_summary_separates_selected_errors_from_candidate_diagnostics() -> None:
    def case(
        key: str, *, selected: str, oracle: str | None, supported: bool,
        false_safe: tuple[str, ...] = (), false_reject: tuple[str, ...] = (),
    ) -> ActionPolicyCaseResult:
        full = selected == "full_access"
        return ActionPolicyCaseResult(
            case_key=key, family_key=key, selected_action_key=selected,
            measured_oracle_action_key=oracle, unsupported_at_budget=full,
            selected_supported_sparse=supported,
            measured_adequate_sparse=() if oracle is None else (oracle,),
            predicted_safe_sparse=() if full else tuple(dict.fromkeys((selected, *false_safe))),
            false_safe_sparse=false_safe, false_reject_sparse=false_reject,
            selected_work=1.0 if full else 0.8,
            measured_oracle_sparse_work=None if oracle is None else 0.75,
            selected_nonredundant_k=None if full else 2,
        )

    result = selector.summarize_policy_results([
        case("a", selected="two_packet", oracle="root", supported=False,
             false_safe=("two_packet",), false_reject=("root",)),
        case("b", selected="full_access", oracle="root", supported=False,
             false_reject=("root",)),
        case("c", selected="root", oracle="root", supported=True,
             false_safe=("four_packet",)),
    ], split="development", model_name="neural")
    assert result["false_safe_choice_count"] == 1
    assert result["false_reject_choice_count"] == 1
    assert result["candidate_false_safe_count"] == 2
    assert result["candidate_false_reject_count"] == 2
    assert [row["selected_false_safe"] for row in result["per_case"]] == [True, False, False]
    assert [row["full_fallback_with_adequate_sparse"] for row in result["per_case"]] == [False, True, False]


def test_training_exposure_requires_executed_baseline_cuts(tmp_path: Path) -> None:
    steps = tmp_path / "training_steps.jsonl"
    routes = tmp_path / "route_work.jsonl"
    def records(update: int, primary_pass: int, paths: tuple[str, ...], frontier: tuple[int, ...]):
        baseline = {
            "frontier": list(frontier),
            "frontier_paths": list(paths),
            "raw_frontier_k": len(frontier),
            "nonredundant_k": len(frontier),
            "nonredundant_k_status": "source_signature_quotient",
            "case_index": 0,
        }
        attempted = 1000 + update
        step = {
            "completed_update": update,
            "attempted_optimizer_steps_including_old_branch": attempted,
            "training_family_id": "family-a",
            "historical_case_id": f"history-{update}",
            "training_metadata": {
                "action": "four_packet",
                "phase": "action_family",
                "primary_pass": primary_pass,
                "capacity_fraction": selector.PRIMARY_CAPACITY,
                "requested_cut_paths": list(selector.THERMAL_ACTION_PATHS["four_packet"]),
                "resolved_cut_paths": list(paths),
                "resolved_cut_paths_status": "actual_baseline_hard_route_record",
                "realized_cut_evidence": {
                    "status": "realized_sparse_cut",
                    "completed_updates_before_attempt": update - 1,
                    "attempted_optimizer_step_including_old_branch": attempted,
                    "baseline": baseline,
                },
                "full_access_replay": False,
            },
        }
        route = {
            "arm": "G",
            "state": "baseline",
            "case_index": 0,
            "optimizer_update": update,
            "completed_updates_before_attempt": update - 1,
            "attempted_optimizer_step_including_old_branch": attempted,
            **baseline,
            "budgets": {"QE": selector.PRIMARY_CAPACITY, "MM": selector.PRIMARY_CAPACITY},
            "routes": {
                "QE": {"requested_fraction": selector.PRIMARY_CAPACITY},
                "MM": {"requested_fraction": selector.PRIMARY_CAPACITY},
            },
        }
        return step, route

    step_rows, route_rows = zip(
        records(1, 2, ("L", "R"), (3, 4)),
        records(2, 5, ("LL", "LR", "RL", "RR"), (1, 2, 3, 4)),
        strict=True,
    )
    steps.write_text("".join(json.dumps(row) + "\n" for row in step_rows), encoding="utf-8")
    routes.write_text("".join(json.dumps(row) + "\n" for row in route_rows), encoding="utf-8")

    measured = selector.summarize_training_exposure(
        steps, checkpoint_update=2, family_ids=("family-a",), route_work_path=routes,
        extra_route="MM",
    )["per_action"]["four_packet"]
    assert measured["trained_sparse"]
    assert measured["realized_baseline_raw_k_counts"] == {2: 1, 4: 1}
    assert measured["realized_baseline_path_counts"] == {"L|R": 1, "LL|LR|RL|RR": 1}
    assert measured["requested_action_is_path_resolving_family"]
    assert measured["per_family_raw_scheduled_action_capacity_updates"] == {"family-a": 2}
    assert measured["per_family_audited_action_capacity_updates"] == {"family-a": 2}
    assert measured["realized_baseline_cut_audited"]

    mismatch = dict(route_rows[0])
    mismatch["frontier_paths"] = ["LL", "RR"]
    routes.write_text(json.dumps(mismatch) + "\n" + json.dumps(route_rows[1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its executed baseline cut"):
        selector.summarize_training_exposure(
            steps, checkpoint_update=2, family_ids=("family-a",), route_work_path=routes,
            extra_route="MM",
        )

    routes.write_text(json.dumps(route_rows[0]) + "\n", encoding="utf-8")
    missing = selector.summarize_training_exposure(
        steps, checkpoint_update=2, family_ids=("family-a",), route_work_path=routes,
        extra_route="MM",
    )["per_action"]["four_packet"]
    assert not missing["trained_sparse"]
    assert missing["missing_realized_baseline_updates"] == [2]
    assert not missing["realized_baseline_cut_audited"]

    legacy_route = {
        "arm": "G",
        "state": "baseline",
        "optimizer_update": 1,
        "frontier": [0],
    }
    later_step, later_route = records(3, 8, ("LL", "LR", "RL", "RR"), (1, 2, 3, 4))
    steps.write_text(
        "".join(json.dumps(row) + "\n" for row in step_rows)
        + json.dumps(later_step) + "\n",
        encoding="utf-8",
    )
    routes.write_text(
        json.dumps(legacy_route) + "\n" + json.dumps(route_rows[1]) + "\n"
        + json.dumps(later_route) + "\n",
        encoding="utf-8",
    )
    legacy = selector.summarize_training_exposure(
        steps, checkpoint_update=3, family_ids=("family-a",), route_work_path=routes,
        extra_route="MM",
    )["per_action"]["four_packet"]
    assert legacy["complete_primary_pass_ids"] == [5, 8]
    assert legacy["trained_sparse"]
    assert legacy["missing_realized_baseline_updates"] == [1]
    assert legacy["uncertified_realized_baseline_reasons"] == {
        "1": "legacy_or_incomplete_route_metric_binding_fields"
    }
    assert legacy["per_family_raw_scheduled_action_capacity_updates"] == {"family-a": 3}
    assert legacy["per_family_audited_action_capacity_updates"] == {"family-a": 2}
    assert legacy["per_family_uncertified_action_capacity_updates"] == {"family-a": 1}
    assert not legacy["realized_baseline_cut_audited"]


def test_query_repeat_audit_measures_intersections_and_labels_missing_ids() -> None:
    def stencil(**role_ids: tuple[str, ...] | None) -> SimpleNamespace:
        roles = {
            name: SimpleNamespace(query_ids=ids)
            for name, ids in role_ids.items()
        }
        return SimpleNamespace(
            baseline=SimpleNamespace(output=SimpleNamespace(roles=roles))
        )

    audit = selector._query_id_overlap_audit(
        stencil(fluid_fields=("f0", "f1"), solid_temperature=("s0", "s1")),
        stencil(fluid_fields=("f1", "f2"), solid_temperature=("s2",)),
    )
    assert audit["status"] == "measured_query_id_intersections"
    assert audit["role_overlap"]["fluid_fields"]["intersection_count"] == 1
    assert audit["role_overlap"]["solid_temperature"]["intersection_count"] == 0
    assert not audit["sampled_roles_disjoint"]

    unavailable = selector._query_id_overlap_audit(
        stencil(fluid_fields=("f0",), solid_temperature=None),
        stencil(fluid_fields=("f1",), solid_temperature=None),
    )
    assert unavailable["status"] == "independently_reseeded_overlap_unavailable"
    assert unavailable["sampled_roles_disjoint"] is None


def test_query_repeat_is_excluded_from_primary_training_rows() -> None:
    primary = SimpleNamespace(family_key="family-a", case_key="family-a|primary_0")
    repeat = SimpleNamespace(family_key="family-a", case_key="family-a|query_repeat_0")
    dev = SimpleNamespace(family_key="family-b", case_key="family-b|primary_0")

    fit_rows = selector._primary_training_rows(
        [primary, repeat, dev], train_families={"family-a"}
    )
    primary_rows, repeat_rows = selector._partition_query_repeat_rows([primary, repeat, dev])
    assert fit_rows == [primary]
    assert primary_rows == [primary, dev]
    assert repeat_rows == [repeat]


def test_exact_context_partition_allows_correlated_reynolds_contexts_and_catches_known_alias() -> None:
    selector.validate_family_partitions({
        "train": ("family-a", "0001"),
        "dev": ("active_packet:case0319:Re70",),
        "held_family_audit": ("active_packet:case0319:Re50",),
    })
    assert selector.shared_geometry_across_splits({
        "dev": ("active_packet:case0319:Re70",),
        "held_family_audit": ("active_packet:case0319:Re50",),
    }) == {"0319": ["dev", "held_family_audit"]}
    with pytest.raises(ValueError, match=r"duplicate_family:0001\+0273"):
        selector.validate_family_partitions({
            "train": ("0001",),
            "dev": ("active_packet:case0273:Re50",),
        })


def test_packet_signature_and_canonical_pairs_use_different_receiver_axes() -> None:
    class FakePlan:
        split_gates = torch.zeros(3)

        def permission_matrix(self, mechanism: str, *, module_present=None):
            assert mechanism == "MM"
            return torch.tensor([
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
            ])

        def access_for(self, mechanism: str, receivers: torch.Tensor, *, module_present=None):
            assert mechanism == "MM"
            assert receivers.shape == (4, 2)
            return torch.tensor([
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ])

    catalog = SimpleNamespace(
        receiver_coordinates=torch.zeros((4, 2)),
        source_validity=torch.tensor([True, True, False]),
        pair_validity=torch.tensor([
            [False, True, False],
            [True, False, False],
            [True, True, False],
            [False, False, False],
        ]),
    )
    packet_sources, canonical_support, detail = selector._route_support_for_catalog(
        FakePlan(), "MM", catalog, (0, 1), module_present=torch.tensor([1.0, 1.0, 0.0]),
    )
    assert packet_sources.shape == (2, 3)  # selected packet count K
    assert catalog.pair_validity.shape == (4, 3)  # canonical receiver count R
    assert packet_sources.tolist() == [[True, False, False], [False, True, False]]
    assert canonical_support.shape == (4, 3)
    assert canonical_support.sum().item() == 2
    assert not canonical_support[0, 0] and not canonical_support[1, 1]
    assert detail["selected_hard_pairs"] == 2
    assert detail["active_receiver_rows"] == 2
    assert detail["hard_permission_matrix_binary"] and detail["canonical_hard_access_binary"]


def test_baseline_selector_features_ignore_permuted_saved_target_values(monkeypatch: pytest.MonkeyPatch) -> None:
    encoded = SimpleNamespace(module_tokens=torch.zeros((1, 1, 2)))
    monkeypatch.setattr(
        selector,
        "describe_realized_plan",
        lambda scores, plan, encoded_value, cut: torch.tensor([[0.2, 0.7], [0.4, 0.9]]),
    )
    monkeypatch.setattr(
        selector,
        "receiver_role_descriptors",
        lambda tree, *, role_count: torch.arange(role_count * 2, dtype=torch.float32).reshape(role_count, 2),
    )
    monkeypatch.setattr(
        selector,
        "_support_signature_and_details",
        lambda encoded_value, tree, plan, cut: ((b"route-pattern",), {
            "selected_hard_pairs": 3,
            "eligible_pairs": 9,
        }),
    )
    monkeypatch.setattr(selector, "frontier_paths", lambda tree, cut, max_depth: tuple(f"node:{i}" for i in cut))
    snapshot = {
        "encoded": encoded,
        "trees": (object(),),
        "plans": (object(),),
        "scores": (object(),),
        "cut": (0, 1),
        "records": ({"frontier": (0, 1)},),
        "budget_fractions": {"MM": 0.9, "QE": 0.9},
    }
    targets = np.arange(12, dtype=np.float64)
    first = selector._fit_panel_features(
        {**snapshot, "saved_target_values": targets.copy()}, action="two_packet", role_count=2,
    )
    permuted = selector._fit_panel_features(
        {**snapshot, "saved_target_values": targets[::-1].copy()}, action="two_packet", role_count=2,
    )
    for first_value, permuted_value in zip(first[:4], permuted[:4], strict=True):
        if torch.is_tensor(first_value):
            torch.testing.assert_close(first_value, permuted_value, rtol=0, atol=0)
        else:
            assert first_value == permuted_value


def test_resource_censored_sparse_actions_do_not_create_a_selector_fit() -> None:
    rows = (SimpleNamespace(trained_sparse=False, full_access=True),)
    assert selector._fit_selector_models(
        rows,
        rows,
        train_families=("train-family",),
        current_forward_sha256="fixed-endpoint",
        seed=23819,
    ) is None


def test_predict_maps_leaves_unexposed_train_actions_for_explicit_full_fallback() -> None:
    case_key = "train-family|primary_0"
    rows = []
    for action, trained, full in (
        ("root", True, False),
        ("two_packet", False, False),
        ("four_packet", False, False),
        ("full_access", False, True),
    ):
        rows.append(selector.ActionEvidenceRow(
            family_key="train-family",
            case_key=case_key,
            action_key=action,
            forward_sha256="fixed-state",
            packet_rows=torch.tensor([[0.2, 0.3]]),
            budget_vector=torch.tensor([0.9, 0.9]),
            receiver_role_features=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            candidate_role_error=torch.tensor([1.0, 1.0]),
            incumbent_role_error=torch.tensor([1.0, 1.0]),
            numerical_floor=torch.tensor([1.0e-6, 1.0e-6]),
            exact_work=0.5 if trained else 1.0,
            nonredundant_k=1,
            trained_sparse=trained,
            full_access=full,
        ))

    class NeuralModel:
        def __call__(self, packet_rows, budget_vector, receiver_role_features, *, nonredundant_k):
            return torch.tensor([0.1, 0.2])

    class RidgeModel:
        def predict(self, row):
            return torch.tensor([0.3, 0.4])

    crossfit = SimpleNamespace(
        row_keys=((case_key, "root"), (case_key, "full_access")),
        neural=torch.tensor([[0.1, 0.2], [0.5, 0.6]]),
        ridge=torch.tensor([[0.3, 0.4], [0.7, 0.8]]),
    )
    neural, ridge = selector._predict_maps(
        rows,
        neural_crossfit=crossfit,
        ridge_crossfit=crossfit,
        neural_final=SimpleNamespace(model=NeuralModel()),
        ridge_final=RidgeModel(),
        train_families={"train-family"},
    )
    assert set(neural) == set(ridge) == {(case_key, "root"), (case_key, "full_access")}

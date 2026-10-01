from __future__ import annotations

import json
import subprocess
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

from honf_forward_core.interface_fields.action_aware_frontier import ActionAwareRiskHead
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
    assert len(measured["certified_realized_path_k_observations"]) == 2
    # The action was scheduled in two complete passes, but neither realized
    # path/K was actually repeated. It is not a matured candidate yet.
    exposure = {"per_action": {"four_packet": measured}}
    qualified, lineage = selector._certified_sparse_case(
        exposure=exposure, action="four_packet", family_id="family-a",
        split="train", realized_paths=("LL", "LR", "RL", "RR"),
        nonredundant_k=4, exact_work=90.0, full_work=100.0,
    )
    assert not qualified
    assert lineage["exact_path_and_k_complete_pass_ids"] == [5]

    repeated = dict(measured)
    repeated["certified_realized_path_k_observations"] = [
        {"primary_pass": pass_id, "training_family_id": "family-a",
         "realized_cut_paths": ["LL", "LR", "RL", "RR"],
         "realized_nonredundant_k": 4}
        for pass_id in (2, 5)
    ]
    repeated_exposure = {"per_action": {"four_packet": repeated}}
    def qualify(split: str, family: str, work: float) -> bool:
        return selector._certified_sparse_case(
            exposure=repeated_exposure, action="four_packet", family_id=family,
            split=split, realized_paths=("LL", "LR", "RL", "RR"),
            nonredundant_k=4, exact_work=work, full_work=100.0,
        )[0]
    assert qualify("train", "family-a", 90.0)
    assert qualify("dev", "family-held", 90.0)
    assert not qualify("train", "family-held", 90.0)
    assert not qualify("dev", "family-held", 100.0)

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


def test_inference_bundle_round_trip_selects_from_saved_input_features(tmp_path: Path) -> None:
    torch.manual_seed(19)
    roles = ("fluid", "solid")
    coordinates = torch.tensor([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
    ])
    from honf_forward_core.interface_fields.action_aware_frontier import (
        describe_realized_plan,
        receiver_role_descriptors,
    )
    from honf_forward_core.interface_fields.adaptive_interaction_cover import (
        CaseLocalReceiverTree,
        MechanismPlan,
        ReceiverAnchorUniverse,
    )
    from honf_forward_core.interface_fields.input_cover_organizer import OrganizerScores

    receiver_universe = ReceiverAnchorUniverse(
        coordinates,
        torch.ones(4),
        torch.tensor([0, 0, 1, 1]),
        torch.ones(2),
    )
    tree = CaseLocalReceiverTree.build(receiver_universe, max_nodes=7, min_leaf_anchors=1)
    node_count = len(tree.nodes)
    scores = OrganizerScores(
        split_logits=torch.zeros(node_count),
        module_logits=torch.zeros(node_count, 2),
        environment_logits=torch.zeros(node_count, 2),
        mechanism_logits={
            "MM": torch.arange(node_count * 2, dtype=torch.float32).reshape(node_count, 2) / 10,
            "QE": torch.flip(torch.arange(node_count * 2, dtype=torch.float32).reshape(node_count, 2), (1,)) / 10,
        },
        node_embeddings=torch.arange(node_count * 4, dtype=torch.float32).reshape(node_count, 4) / 10,
        module_embeddings=torch.arange(8, dtype=torch.float32).reshape(2, 4) / 7,
        environment_embeddings=torch.arange(8, dtype=torch.float32).reshape(2, 4) / 9,
        budget_vector=torch.tensor([0.9, 0.9, 1.0, 1.0, 1.0]),
    )
    encoded = SimpleNamespace(
        module_centers=coordinates[:2][None],
        module_present=torch.ones(1, 2),
        env_coords=coordinates[2:][None],
        env_weights=torch.ones(1, 2),
    )
    left, right = tree.nodes[0].left, tree.nodes[0].right
    assert left is not None and right is not None
    grandchildren = (
        tree.nodes[left].left, tree.nodes[left].right,
        tree.nodes[right].left, tree.nodes[right].right,
    )
    assert all(index is not None for index in grandchildren)
    cuts = {
        "root": (0,),
        "two_packet": (left, right),
        "four_packet": tuple(int(index) for index in grandchildren),
        "full_access": tuple(int(index) for index in grandchildren),
    }
    role_features = receiver_role_descriptors(tree, role_count=len(roles))
    action_features = {}
    for action, cut in cuts.items():
        if action == "full_access":
            mm_permission = torch.ones(node_count, 2)
            qe_permission = torch.ones(node_count, 2)
        else:
            mm_permission = torch.zeros(node_count, 2)
            qe_permission = torch.zeros(node_count, 2)
            mm_permission[:, 0] = 1.0
            qe_permission[:, 1] = 1.0
        plan = MechanismPlan(
            tree,
            torch.zeros(node_count),
            encoded.module_present[0],
            int(encoded.env_coords.shape[1]),
            permissions={"MM": mm_permission, "QE": qe_permission},
        )
        action_features[action] = {
            "packet_rows": describe_realized_plan(scores, plan, encoded, cut),
            "budget_vector": scores.budget_vector,
            "receiver_role_features": role_features,
        }
    packet_width = action_features["root"]["packet_rows"].shape[1]
    role_width = role_features.shape[1]
    model = ActionAwareRiskHead(
        packet_feature_dim=packet_width,
        budget_dim=len(selector.MECHANISM_ORDER),
        receiver_role_dim=role_width,
        hidden_dim=12,
    ).eval()

    checkpoint_path = tmp_path / "selected_g.pt"
    source_path = tmp_path / "source.pt"
    manifest_path = tmp_path / "controlled_manifest.json"
    action_table_path = tmp_path / "action_table.jsonl"
    provenance_path = tmp_path / "provenance.json"
    bundle_path = tmp_path / "action_risk_selector.pt"
    checkpoint_path.write_bytes(b"selected G endpoint fixture")
    source_path.write_bytes(b"fixed Run1804 source fixture")
    manifest_path.write_text('{"run_id":"fixture"}\n', encoding="utf-8")
    action_table_path.write_text(
        '{"action":"root","split":"train","reference_label":3.0}\n', encoding="utf-8"
    )
    provenance_path.write_text(
        json.dumps({
            "run_id": "fixture",
            "checkpoint_arm": "G",
            "selector_inference_bundle_path": str(bundle_path.resolve()),
        }) + "\n",
        encoding="utf-8",
    )
    forward = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": selector._sha256(checkpoint_path),
        "checkpoint_update": 900,
        "checkpoint_arm": "G",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": selector._sha256(source_path),
        "controlled_manifest": str(manifest_path),
        "controlled_manifest_sha256": selector._sha256(manifest_path),
    }

    packet_counts = {action: len(cuts[action]) for action in selector.ACTION_KEYS}
    feature_path = tmp_path / "candidate_features.pt"
    torch.save({
        "action_features": action_features,
        "exact_work_by_action": {"root": 7.0, "two_packet": 3.0, "four_packet": 5.0, "full_access": 10.0},
        "nonredundant_k_by_action": packet_counts,
        "trained_sparse_by_action": {"root": True, "two_packet": True, "four_packet": False, "full_access": False},
    }, feature_path)
    saved_features = torch.load(feature_path, map_location="cpu", weights_only=True)
    runtime_policy = {
        "action_order": list(selector.ACTION_KEYS),
        "exposure_qualified_sparse_actions": ["root", "two_packet"],
        "role_order": list(roles),
        "fixed_role_log_limits": [1.0e6, 1.0e6],
        "empirical_role_margin": [0.1, 0.1],
    }
    expected = selector.select_action_with_risk_bundle(
        model,
        {"policy": runtime_policy},
        saved_features["action_features"],
        exact_work_by_action=saved_features["exact_work_by_action"],
        nonredundant_k_by_action=saved_features["nonredundant_k_by_action"],
        trained_sparse_by_action=saved_features["trained_sparse_by_action"],
    )
    assert expected["selected_action"] == "two_packet"

    artifact = selector.save_action_risk_selector_bundle(
        bundle_path,
        model=model,
        selected_forward=forward,
        action_table_path=action_table_path,
        provenance_path=provenance_path,
        role_order=roles,
        train_family_ids=("train-family-a", "train-family-b"),
        split_manifest_sha256="a" * 64,
        fixed_role_log_limits=runtime_policy["fixed_role_log_limits"],
        empirical_margin=runtime_policy["empirical_role_margin"],
        fit_seed=31,
        fit_updates=120,
        crossfit_folds=2,
        exposure_qualified_sparse_actions=("root", "two_packet"),
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite selector bundle"):
        selector.save_action_risk_selector_bundle(
            bundle_path,
            model=model,
            selected_forward=forward,
            action_table_path=action_table_path,
            provenance_path=provenance_path,
            role_order=roles,
            train_family_ids=("train-family-a", "train-family-b"),
            split_manifest_sha256="a" * 64,
            fixed_role_log_limits=runtime_policy["fixed_role_log_limits"],
            empirical_margin=runtime_policy["empirical_role_margin"],
            fit_seed=31,
            fit_updates=120,
            crossfit_folds=2,
            exposure_qualified_sparse_actions=("root", "two_packet"),
        )
    loaded_model, loaded_bundle = selector.load_action_risk_selector_bundle(
        bundle_path,
        expected_bundle_sha256=artifact["sha256"],
        expected_forward_sha256=forward["checkpoint_sha256"],
    )
    actual = selector.select_action_with_risk_bundle(
        loaded_model,
        loaded_bundle,
        saved_features["action_features"],
        exact_work_by_action=saved_features["exact_work_by_action"],
        nonredundant_k_by_action=saved_features["nonredundant_k_by_action"],
        trained_sparse_by_action=saved_features["trained_sparse_by_action"],
    )
    assert actual == expected
    assert loaded_bundle["policy"]["action_order"] == list(selector.ACTION_KEYS)
    assert loaded_bundle["policy"]["exposure_qualified_sparse_actions"] == ["root", "two_packet"]
    assert loaded_bundle["training"]["train_family_ids"] == ["train-family-a", "train-family-b"]
    assert "candidate_role_error" not in loaded_bundle
    assert "reference_labels" not in loaded_bundle
    assert "action_features" not in loaded_bundle
    assert artifact["action_table_sha256"] == selector._sha256(action_table_path)
    assert artifact["provenance_sha256"] == selector._sha256(provenance_path)
    assert loaded_bundle["feature_schema"]["packet_rows"]["tensor_shape"] == ["realized_cut_k", packet_width]
    assert loaded_bundle["feature_schema"]["receiver_role_features"]["tensor_shape"] == [len(roles), role_width]

    # Re-run the serialized decision in an independent interpreter to guard
    # against hidden model state or import-process state.
    child_code = r"""
import json, sys, torch
bundle_path, bundle_sha, forward_sha, features_path, case_src, project_src, script_src = sys.argv[1:]
sys.path[:0] = [case_src, project_src, script_src]
import evaluate_matured_action_selector as selector
model, bundle = selector.load_action_risk_selector_bundle(
    __import__('pathlib').Path(bundle_path),
    expected_bundle_sha256=bundle_sha,
    expected_forward_sha256=forward_sha,
)
features = torch.load(features_path, map_location='cpu', weights_only=True)
decision = selector.select_action_with_risk_bundle(
    model,
    bundle,
    features['action_features'],
    exact_work_by_action=features['exact_work_by_action'],
    nonredundant_k_by_action=features['nonredundant_k_by_action'],
    trained_sparse_by_action=features['trained_sparse_by_action'],
    device='cpu',
)
print(json.dumps(decision, sort_keys=True))
"""
    fresh = subprocess.run(
        [
            sys.executable,
            "-c",
            child_code,
            str(bundle_path),
            artifact["sha256"],
            forward["checkpoint_sha256"],
            str(feature_path),
            str(selector.CASE_ROOT / "src"),
            str(selector.PROJECT_ROOT / "src"),
            str(selector.CASE_ROOT / "scripts"),
        ],
        cwd=selector.PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    fresh_decision = json.loads(fresh.stdout)
    assert fresh_decision["selected_action"] == expected["selected_action"]
    np.testing.assert_array_equal(
        np.asarray(fresh_decision["predicted_log_risk_by_action"]),
        np.asarray(expected["predicted_log_risk_by_action"]),
    )

    unauthorized_actions = dict(saved_features["trained_sparse_by_action"])
    unauthorized_actions["four_packet"] = True
    with pytest.raises(ValueError, match="exposure-qualified set"):
        selector.select_action_with_risk_bundle(
            loaded_model,
            loaded_bundle,
            saved_features["action_features"],
            exact_work_by_action=saved_features["exact_work_by_action"],
            nonredundant_k_by_action=saved_features["nonredundant_k_by_action"],
            trained_sparse_by_action=unauthorized_actions,
        )
    malformed_roles = dict(saved_features["action_features"])
    malformed_roles["root"] = dict(malformed_roles["root"])
    malformed_roles["root"]["receiver_role_features"] = role_features[:1]
    with pytest.raises(ValueError, match="feature shapes"):
        selector.select_action_with_risk_bundle(
            loaded_model,
            loaded_bundle,
            malformed_roles,
            exact_work_by_action=saved_features["exact_work_by_action"],
            nonredundant_k_by_action=saved_features["nonredundant_k_by_action"],
            trained_sparse_by_action=saved_features["trained_sparse_by_action"],
        )

    with pytest.raises(ValueError, match="different selected G checkpoint"):
        selector.load_action_risk_selector_bundle(
            bundle_path,
            expected_bundle_sha256=artifact["sha256"],
            expected_forward_sha256="b" * 64,
        )

    action_table_path.write_text('{"action":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="action_table identity changed"):
        selector.load_action_risk_selector_bundle(
            bundle_path,
            expected_bundle_sha256=artifact["sha256"],
            expected_forward_sha256=forward["checkpoint_sha256"],
        )

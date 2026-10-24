"""Pure metric and input-selection checks for bounded soft-only reviews."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import pytest
import torch

from honf_runtime.thermal_soft_reviews import (
    _candidate_is_promising,
    _accumulate_query_access,
    _canonical_val89_ids,
    _new_query_access_accumulator,
    _native_checkpoint_validation,
    _normalized_pair_discrepancy,
    _response_inventory,
    _response_payload_for_softonly,
    _select_intervention_candidates,
    _select_train_panel,
    DEVELOPMENT_RESPONSE_IDS,
    TRAIN_RESPONSE_IDS,
)


def test_parameter_and_buffer_hashes_accept_scalar_native_temperatures():
    from honf_runtime.thermal_soft_reviews import _weights_sha256, _state_sha256
    model = torch.nn.Module()
    model.register_parameter("raw_temperature", torch.nn.Parameter(torch.tensor(1.)))
    model.register_buffer("clock", torch.tensor(100, dtype=torch.long))
    before = _weights_sha256(model)
    assert before == _weights_sha256(model)
    assert _state_sha256(model.state_dict())
    with torch.no_grad():
        model.raw_temperature.add_(1.)
    assert before != _weights_sha256(model)


def test_review_snapshot_preserves_dictionary_controller_extra_state():
    from honf_runtime.thermal_soft_reviews import _state_snapshot, _state_sha256
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))
            self.calls = 0
        def get_extra_state(self):
            return {"calls": self.calls, "history": [torch.tensor(2.)]}
        def set_extra_state(self, state):
            self.calls = state["calls"]
    model = Model()
    snapshot = _state_snapshot(model)
    digest = _state_sha256(snapshot)
    model.calls = 9
    assert _state_sha256(model.state_dict()) != digest
    model.load_state_dict(snapshot, strict=True)
    assert model.calls == 0 and _state_sha256(model.state_dict()) == digest


def _case(case_id, present):
    return {"case_id": case_id, "structure": {"module_present": np.asarray(present, dtype=np.float32)}}


def test_normalized_membership_tv_ignores_row_amplitude_and_retains_magnitudes():
    eligible = np.ones(4, dtype=bool)
    result = _normalized_pair_discrepancy(
        np.asarray([1.0, 0.0, 5.0, 0.0]),
        np.asarray([10.0, 0.0, 50.0, 0.0]),
        eligible,
        module_count=2,
    )
    assert result["nondegenerate"] is True
    assert result["normalized_rowdiff"] == 0.0
    assert result["raw_row_l1_difference"] == 54.0
    assert result["left_row_l1_mass"] == 6.0
    assert result["right_row_l1_mass"] == 60.0


def test_normalized_membership_tv_equal_weights_available_donor_types():
    result = _normalized_pair_discrepancy(
        np.asarray([1.0, 0.0, 2.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0, 3.0]),
        np.ones(4, dtype=bool),
        module_count=2,
    )
    assert result["normalized_rowdiff"] == 1.0
    assert result["typed_donors"]["M"]["normalized_total_variation"] == 1.0
    assert result["typed_donors"]["E"]["normalized_total_variation"] == 1.0


def test_normalized_membership_reports_empty_and_degenerate_typed_slices():
    result = _normalized_pair_discrepancy(
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 0.0, 0.0]),
        np.asarray([True, True, False, False]),
        module_count=2,
    )
    assert result["nondegenerate"] is False
    assert result["normalized_rowdiff"] is None
    assert result["typed_empty_or_degenerate_skipped"] == ["M", "E"]


def test_eligible_degenerate_donor_type_blocks_non_degenerate_pair_gate():
    result = _normalized_pair_discrepancy(
        np.asarray([0.0, 0.0, 2.0, 0.0]),
        np.asarray([0.0, 0.0, 0.0, 2.0]),
        np.ones(4, dtype=bool),
        module_count=2,
    )
    assert result["normalized_rowdiff"] == 1.0
    assert result["typed_donors"]["E"]["available"] is True
    assert result["degenerate_eligible_types"] == ["M"]
    assert result["nondegenerate"] is False


def test_query_access_accumulation_matches_full_and_tiled_query_panels():
    gates = np.asarray([[[1.0, 0.0, 0.5], [0.0, 0.25, 0.0],
                         [0.2, 0.3, 0.4], [0.0, 0.0, 0.0]]], dtype=np.float32)
    weights = np.asarray([[[0.75, 0.25], [0.0, 1.0],
                           [0.4, 0.6], [0.0, 0.0]]], dtype=np.float32)
    logits = np.asarray([[[1.0, 2.0], [3.0, 4.0],
                          [5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    query_mask = np.asarray([[True, True, True, False]])
    eligible = np.asarray([[True, True, False]])
    group_mask = np.asarray([[True, True]])

    def collect(slices):
        state = _new_query_access_accumulator()
        for start, stop in slices:
            _accumulate_query_access(
                state, torch.as_tensor(gates[:, start:stop]), torch.as_tensor(weights[:, start:stop]),
                torch.as_tensor(logits[:, start:stop]), torch.as_tensor(query_mask[:, start:stop]),
                torch.as_tensor(eligible), torch.as_tensor(group_mask),
            )
        return state

    full = collect([(0, 4)])
    tiled = collect([(0, 1), (1, 3), (3, 4)])
    assert tiled == full


def test_tiny_native_adapter_query_access_uses_core_bounded_tiles_and_matches_full():
    from honf_forward_core.content_hypergraph.decoding import pack_local_pairs
    from tests.test_thermal_compact_adapter import _compact_adapter
    from honf_runtime.thermal_soft_reviews import _query_access_summary

    model, structure, fluid, local = _compact_adapter(3)
    model.eval()
    core = model.core
    core.query_tile_size = 5
    prepared = model.prepare_native(structure, fluid, local_query_points=local, ntheta=16,
                                    scene_key=("soft-review-tiled-query-test",))
    organization, query = prepared.organization, prepared.query_batch
    qmask = query.query_mask if query.query_mask is not None else torch.ones_like(query.query_ids, dtype=torch.bool)
    states = core._query_states(organization, query)
    local_pairs = pack_local_pairs(organization.scene, query, organization.frame_scale)
    envelope = states.new_zeros(states.shape[0], states.shape[1], organization.scene.num_modules)
    if local_pairs.count:
        envelope = envelope.index_put((local_pairs.batch_index, local_pairs.query_index,
                                       local_pairs.module_index), local_pairs.support_weight)
    gates, weights, logits = core.compact_retrieval.query_access(
        organization.compact_groups, organization, query, states, envelope, qmask,
    )
    expected = _new_query_access_accumulator()
    _accumulate_query_access(expected, gates, weights, logits, qmask,
                             organization.joint_records.eligible, organization.compact_groups.group_mask)
    observed = _query_access_summary(model, organization, query)
    assert observed["query_tile_size"] == 5
    assert observed["query_tile_count"] == (query.query_ids.shape[1] + 4) // 5
    assert observed["query_count"] == expected["query_count"]
    assert observed["eligible_query_record_pairs"] == expected["eligible_query_record_pairs"]
    assert observed["positive_logical_access_pairs"] == expected["positive_logical_access_pairs"]
    assert observed["positive_logical_access_fraction"] == pytest.approx(
        expected["positive_logical_access_pairs"] / expected["eligible_query_record_pairs"]
    )
    assert observed["mean_logical_access_value"] == pytest.approx(
        expected["logical_access_sum"] / expected["eligible_query_record_pairs"], rel=1e-6
    )


def test_tiny_native_adapter_scoped_full_shared_keeps_pair_slots_and_probability():
    from tests.test_thermal_compact_adapter import _compact_adapter

    model, structure, fluid, local = _compact_adapter(12)
    model.eval()
    core = model.core
    baseline = model.prepare_native(structure, fluid, local_query_points=local, ntheta=16,
                                    scene_key=("soft-review-full-shared-test",))
    base_org, base_compact = baseline.organization, baseline.organization.compact_groups
    valid = torch.nonzero(base_compact.pair_mask[0], as_tuple=False).flatten()
    assert valid.numel() > 0
    pair_index = int(valid[0])
    left, right = map(int, base_compact.pair_slots[0, pair_index].tolist())
    with core.compact_soft_intervention_scope("full_shared", pair_slots=[(left, right)]):
        intervened = model.prepare_native(structure, fluid, local_query_points=local, ntheta=16,
                                           scene_key=("soft-review-full-shared-test",))
    changed_org, changed_compact = intervened.organization, intervened.organization.compact_groups
    assert torch.equal(base_compact.pair_slots, changed_compact.pair_slots)
    assert torch.equal(base_compact.pair_mask, changed_compact.pair_mask)
    assert torch.equal(base_compact.pair_types, changed_compact.pair_types)
    assert torch.equal(base_compact.pair_logits, changed_compact.pair_logits)
    assert torch.equal(base_org.joint_records.ids, changed_org.joint_records.ids)
    assert int(changed_compact.group_mask[0].sum()) == int(base_compact.group_mask[0].sum())
    assert torch.equal(changed_compact.controls[0, left], changed_compact.controls[0, right])


def test_promising_gate_is_conjunctive_and_uses_prespecified_boundaries():
    eligible = {"normalized_rowdiff": 0.1, "nondegenerate": True}
    assert _candidate_is_promising(0.8, eligible)
    assert not _candidate_is_promising(0.799, eligible)
    assert not _candidate_is_promising(0.8, {**eligible, "normalized_rowdiff": 0.10001})
    assert not _candidate_is_promising(0.9, {**eligible, "nondegenerate": False})


def test_intervention_selection_fills_bounded_quota_with_unqualified_probes():
    candidates = [
        {"case_id": "case-b", "bank": "M", "slots": [3, 4], "candidate_index": 1,
         "probability": 0.99, "promising": False},
        {"case_id": "case-a", "bank": "M", "slots": [1, 2], "candidate_index": 0,
         "probability": 0.82, "promising": True},
        {"case_id": "case-a", "bank": "M", "slots": [5, 6], "candidate_index": 2,
         "probability": 0.91, "promising": False},
        {"case_id": "case-c", "bank": "E", "slots": [7, 8], "candidate_index": 3,
         "probability": 0.75, "promising": False},
        {"case_id": "case-d", "bank": "E", "slots": [9, 10], "candidate_index": 4,
         "probability": 0.70, "promising": False},
        {"case_id": "case-e", "bank": "E", "slots": [11, 12], "candidate_index": 5,
         "probability": 0.60, "promising": False},
    ]
    selected = _select_intervention_candidates(candidates, limit_per_bank=2)
    selected_by_bank = {
        bank: [(row, basis) for row, basis in selected if row["bank"] == bank]
        for bank in ("M", "E")
    }
    assert [row["slots"] for row, _ in selected_by_bank["M"]] == [[1, 2], [3, 4]]
    assert [basis for _, basis in selected_by_bank["M"]] == [
        "promising_threshold", "highest_probability_unqualified_probe"
    ]
    assert [row["slots"] for row, _ in selected_by_bank["E"]] == [[7, 8], [9, 10]]
    assert all(basis == "highest_probability_unqualified_probe" for _, basis in selected_by_bank["E"])
    assert all(not row["promising"] for row, basis in selected_by_bank["E"])
    assert len(selected_by_bank["M"]) == len(selected_by_bank["E"]) == 2


def test_train_panel_uses_input_counts_and_stable_id_ties():
    cases = [
        _case("m4-z", [1, 1, 1, 1]), _case("m2-z", [1, 1]),
        _case("m1-z", [1]), _case("m3-z", [1, 1, 1]),
        _case("m1-a", [1]), _case("m2-a", [1, 1]),
        _case("m3-a", [1, 1, 1]), _case("m4-a", [1, 1, 1, 1]),
    ]
    assert _select_train_panel(cases) == ("m1-a", "m2-a", "m3-a", "m4-a")


def test_retained_response_inventory_checks_train4_and_heldout_dev4_metadata_read_only():
    atlas = Path(
        "/data2/wanglz/ModularDT/relocated_workspace/HONF_Proj/diagnostics/generated/interactions/"
        "physical_response_atlas_20260926/families"
    )
    if not atlas.is_dir():
        pytest.skip("Retained DATA2 response atlas is not present in this environment.")
    provider = SimpleNamespace(atlas_directory=atlas)
    base_inventory = {"independent_heat_null": None}
    for group, family_ids in (("train", TRAIN_RESPONSE_IDS),
                              ("development", DEVELOPMENT_RESPONSE_IDS)):
        for family_id in family_ids:
            result = _response_inventory(provider, family_id, response_group=group,
                                         base_inventory=base_inventory)
            source = result["thermal_families"][0]
            assert source["source_split"] == "original TRAIN"
            assert source["response_training_exposure"] == (
                "original TRAIN response supervision" if group == "train"
                else "held out from fit; separately identified response-development family"
            )
            assert source["external_metadata_identity"]["split"] == "train"
            assert source["embedded_npz_metadata_identity"]["source_dataset_split"] == "train"
            assert source["embedded_npz_metadata_identity"]["physical_family_id"] == (
                "duplicate_family:0001+0273" if family_id == "0001" else f"stored_family:{family_id}"
            )
            assert {"fluid_fields", "interface", "solid_temperature"} == set(source["role_contract"])
            assert {"baseline", "i_minus", "i_plus", "j_minus", "j_plus", "mm", "mp", "pm", "pp",
                    "heat_transfer_plus", "heat_transfer_minus"}.issubset(source["all_labels"])


def test_softonly_response_label_is_derived_without_rewriting_checkpoint_identity():
    original = {
        "epoch": 1000,
        "experiment_identity": {
            "config_name": "thermal_compact_softonly_full1000_scratch_v2",
            "source_arm": "scratch_random",
            "provider_identity": {"frozen": True},
        },
    }
    derived = _response_payload_for_softonly(original)
    assert "treatment" not in original["experiment_identity"]
    assert original["experiment_identity"]["source_arm"] == "scratch_random"
    assert derived["experiment_identity"]["treatment"] == "thermal_compact_softonly_full1000_scratch_v2"
    assert derived["epoch"] == original["epoch"]
    assert derived["experiment_identity"]["provider_identity"] == original["experiment_identity"]["provider_identity"]


def test_native_board_rejects_noncanonical_exposed_val89_identity_without_target_reads(tmp_path):
    canonical = _canonical_val89_ids()
    provider = SimpleNamespace(validation_case_ids=tuple(reversed(canonical)))
    result = _native_checkpoint_validation(
        torch.nn.Identity(), provider, selector="latest", checkpoint_path=tmp_path / "latest.pt",
        checkpoint_sha256="a" * 64, epoch=1000, output=tmp_path,
        deadline=100.0, global_deadline=100.0,
    )
    assert result["status"] == "unavailable_wrong_exposed_validation_identity"
    assert result["expected_val89_ids"] == list(canonical)
    assert result["exposed_validation_case_ids"] == list(reversed(canonical))
    assert result["locked_test_targets_read"] is False

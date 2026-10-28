from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))

from honf_runtime.r13_8_anchored_organization_evaluation import (
    _accumulate_numeric_counts,
    _collect_u_organization_diagnostics,
    _invoke_anchored_boundary_sweep,
    _live_probe_coordinates_match,
    _native_metric_deltas,
    _paired_read_penalties,
    _reassignment_intervention,
    _summarize_provider_boundary,
    _thermal_boundary_coordinates,
    evaluate,
    require_complete_anchored_work_counts,
    select_deterministic_anchor_panel,
    summarize_anchored_work_counts,
    summarize_boundary_sweep,
)


def test_anchor_panel_is_deterministic_spatially_spread_and_ignores_padding() -> None:
    ids = np.asarray([40, 10, 30, 20, 50, -1])
    coordinates = np.asarray([
        [1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [np.nan, np.nan],
    ])
    mask = np.asarray([1, 1, 1, 1, 1, 0], dtype=bool)

    selected = select_deterministic_anchor_panel(ids, coordinates, mask, count=4)
    permutation = np.asarray([4, 2, 0, 5, 3, 1])
    selected_permuted = select_deterministic_anchor_panel(
        ids[permutation], coordinates[permutation], mask[permutation], count=4,
    )

    assert selected == selected_permuted
    assert len(selected) == len(set(selected)) == 4
    assert -1 not in selected
    selected_xy = {int(ids[index]): coordinates[index].tolist() for index in range(len(ids))}
    selected_points = [selected_xy[int(original_id)] for original_id in selected]
    assert len({tuple(point) for point in selected_points}) == 4


def test_anchor_panel_rejects_duplicate_valid_original_ids_and_bad_mask() -> None:
    coordinates = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="unique"):
        select_deterministic_anchor_panel([7, 7], coordinates)
    with pytest.raises(ValueError, match="shape"):
        select_deterministic_anchor_panel([7, 8], coordinates, [True])


def test_boundary_sweep_keeps_decoder_and_native_levels_separate() -> None:
    offsets = [-0.5, -0.1, 0.0, 0.1, 0.5]
    smooth = np.asarray([[x] for x in offsets])
    decoder = summarize_boundary_sweep(
        offsets, smooth, output_scale=1.0, evaluation_level="decoder",
    )
    native = summarize_boundary_sweep(
        offsets, smooth + np.asarray([[0.0], [0.0], [0.0], [0.01], [0.0]]),
        output_scale=2.0, evaluation_level="native_extraction",
    )

    assert decoder["evaluation_level"] == "decoder"
    assert native["evaluation_level"] == "native_extraction"
    assert decoder["locator_rebuild_required"] is True
    assert [row["distance_in_E_spacing"] for row in decoder["continuity_and_derivative_pairs"]] == [0.1, 0.5]
    assert decoder["sampled_values"] == smooth.tolist()
    assert decoder["near_face_affine_fit"]["positive_probe_offsets_in_spacing"] == [0.1, 0.5]
    assert decoder["one_sided_affine_fit"]["estimated_limit_gap_max_abs"] == pytest.approx(0.0)
    assert decoder["one_sided_affine_fit"]["slope_gap_rmse_per_spacing"] == pytest.approx(0.0)
    assert native["one_sided_affine_fit"]["estimated_limit_gap_max_abs"] > 0.0


def test_boundary_sweep_moves_device_output_scales_to_host_before_numpy() -> None:
    import torch

    offsets = [-0.5, -0.1, 0.0, 0.1, 0.5]
    values = np.asarray([[x, 2.0 * x] for x in offsets])
    summary = summarize_boundary_sweep(
        offsets,
        values,
        output_scale=torch.tensor([2.0, 3.0], requires_grad=True),
        evaluation_level="native_extraction",
    )

    assert summary["output_scale"] == [2.0, 3.0]


def test_broad_affine_fit_can_show_a_false_gap_for_a_smooth_cubic() -> None:
    offsets = np.asarray(
        [-0.5, -0.2, -0.05, -0.005, -0.0005, 0.0, 0.0005, 0.005, 0.05, 0.2, 0.5],
    )
    samples = (offsets**2 + offsets**3)[:, None]
    summary = summarize_boundary_sweep(offsets, samples, evaluation_level="decoder")

    broad_gap = summary["one_sided_affine_fit"]["estimated_limit_gap_max_abs"]
    near = summary["near_face_affine_fit"]
    assert summary["one_sided_affine_fit"]["fit_scope"].startswith("broad diagnostic")
    assert near["positive_probe_offsets_in_spacing"] == [0.0005, 0.005, 0.05]
    assert near["negative_probe_offsets_in_spacing"] == [-0.0005, -0.005, -0.05]
    assert broad_gap > 0.02
    assert near["estimated_limit_gap_max_abs"] < 1.0e-4


def test_boundary_dispatch_forwards_each_read_mode() -> None:
    calls = []

    def boundary_fn(model, batch, query_index, coordinates, *, offsets_in_spacing, read_mode):
        calls.append({
            "query_index": query_index,
            "coordinates": coordinates,
            "offsets": offsets_in_spacing,
            "read_mode": read_mode,
        })
        return {"read_mode": read_mode}

    probe = {"probe_coordinates": [[0.0, 0.0]], "offsets_in_spacing": [0.0]}
    for mode in ("full", "restricted"):
        result = _invoke_anchored_boundary_sweep(boundary_fn, "model", "batch", 9, probe, mode)
        assert result["read_mode"] == mode
    assert [row["read_mode"] for row in calls] == ["full", "restricted"]


def test_live_probe_coordinate_check_accepts_only_dtype_scale_rounding() -> None:
    import torch

    planned = np.asarray([[5.75, 3.375], [5.975, 3.375], [6.25, 3.375]], dtype=np.float64)
    returned = torch.tensor(planned, dtype=torch.float32)
    assert _live_probe_coordinates_match(returned, planned)
    perturbed = returned.clone()
    perturbed[1, 0] += 1.0e-3
    assert not _live_probe_coordinates_match(perturbed, planned)


def test_work_summary_separates_unique_union_scoring_from_per_anchor_reads() -> None:
    summary = summarize_anchored_work_counts({
        "core_prepare_primary_prepared_e_directory_rows": 12,
        "core_prepare_primary_prepared_e_directory_full_entries": 120,
        "core_prepare_primary_prepared_e_directory_joint95_entries": 47,
        "core_prepare_primary_prepared_e_directory_sort_entries": 120,
        "core_prepare_primary_prepared_e_directory_compile_pair_rows": 120,
        "core_prepare_primary_prepared_e_directory_positive_exchange_entries": 80,
        "thermal_prepare_primary_query_anchor_locator_rows": 50,
        "thermal_prepare_primary_query_anchor_index_rows": 200,
        "core_decode_primary_query_unique_union_raw_score_pairs_M": 9,
        "core_decode_primary_query_unique_union_raw_score_pairs_E": 11,
        "core_decode_primary_query_anchor_member_pairs_M": 17,
        "core_decode_primary_query_anchor_member_pairs_E": 19,
        "core_decode_primary_query_anchor_normalization_rows_M": 8,
        "core_decode_primary_query_anchor_normalization_rows_E": 8,
        "core_decode_primary_query_anchor_typed_gate_rows_M": 8,
        "core_decode_primary_query_anchor_typed_gate_rows_E": 8,
        "core_decode_primary_query_anchor_coalesced_value_pairs_M": 7,
        "core_decode_primary_query_anchor_coalesced_value_pairs_E": 9,
        "core_decode_primary_query_anchor_value_reduction_pairs_M": 17,
        "core_decode_primary_query_anchor_value_reduction_pairs_E": 19,
        "core_decode_primary_query_M_member_query_affine_rows": 9,
        "core_decode_primary_query_E_member_query_affine_rows": 11,
        "core_decode_primary_query_joint_field_head_rows": 8,
    })

    observed = summary["observed"]
    assert observed["unique_raw_score_pairs_M"] == 9
    assert observed["unique_raw_score_pairs_E"] == 11
    assert observed["anchor_member_pairs_M"] == 17
    assert observed["anchor_member_pairs_E"] == 19
    assert observed["anchor_value_reduction_pairs_M"] > observed["unique_raw_score_pairs_M"]
    assert summary["missing_counters"] == []
    assert require_complete_anchored_work_counts(summary["source_work_counts"])["missing_counters"] == []


def test_complete_work_summary_rejects_missing_measured_counter() -> None:
    with pytest.raises(ValueError, match="missing measured work counters"):
        require_complete_anchored_work_counts({"prepared_e_directory_rows": 1})


def test_paired_read_penalty_keeps_signed_casewise_changes() -> None:
    full = {"metrics": {"temperature_surface": {
        "unit": "K",
        "per_case": [
            {"case_id": "a", "module_count": 1, "value": 2.0},
            {"case_id": "b", "module_count": 2, "value": 1.0},
        ],
    }}}
    restricted = {"metrics": {"temperature_surface": {
        "unit": "K",
        "per_case": [
            {"case_id": "a", "module_count": 1, "value": 1.5},
            {"case_id": "b", "module_count": 2, "value": 1.2},
        ],
    }}}
    summary = _paired_read_penalties(full, restricted)["temperature_surface"]
    assert summary["mean_restricted_minus_full"] == pytest.approx(-0.15)
    assert summary["improved_case_count"] == 1
    assert summary["regressed_case_count"] == 1
    assert summary["worst_positive_penalty"] == pytest.approx(0.2)


def test_same_state_metric_deltas_are_paired_and_count_hierarchy_stays_separate() -> None:
    baseline = {"u": {"unit": "m/s", "per_case": [{"case_id": "case-1", "value": 2.0}]}}
    intervention = {"u": {"unit": "m/s", "per_case": [{"case_id": "case-1", "value": 1.5}]}}
    delta = _native_metric_deltas(baseline, intervention)["u"]
    assert delta["intervention_minus_baseline_rmse"] == pytest.approx(-0.5)

    total: dict[str, float] = {}
    _accumulate_numeric_counts(total, {
        "baseline": {"query_anchor_member_pairs_M": 10},
        "intervention": {"query_anchor_member_pairs_M": 7},
        "probe": [{"query_anchor_member_pairs_E": 3}, {"query_anchor_member_pairs_E": 4}],
    })
    assert total == {
        "baseline__query_anchor_member_pairs_M": 10.0,
        "intervention__query_anchor_member_pairs_M": 7.0,
        "probe__0000__query_anchor_member_pairs_E": 3.0,
        "probe__0001__query_anchor_member_pairs_E": 4.0,
    }


def test_reassignment_intervention_uses_trace_members_and_an_absent_same_role_target() -> None:
    from types import SimpleNamespace

    scene = SimpleNamespace(
        environment_ids=np.asarray([[10, 11]]),
        module_ids=np.asarray([[1, 2]]),
        environment_mask=np.asarray([[True, True]]),
        module_mask=np.asarray([[True, True]]),
    )
    trace = {
        "anchor_indices": np.asarray([[[0]]]),
        "anchor_weights": np.asarray([[[1.0]]]),
        "anchor_read_indices": np.asarray([[0, 0, 0]]),
        "anchor_read_member_read_index": np.asarray([0, 0]),
        "anchor_read_E_slots": np.asarray([0]),
        "anchor_read_original_id": np.asarray([1, 10]),
        "anchor_read_is_module": np.asarray([True, False]),
    }
    assert _reassignment_intervention(scene, {"trace": trace}) == {
        "kind": "reassign_member",
        "source_role": "M",
        "source_id": 1,
        "target_role": "M",
        "target_id": 2,
        "anchor_id": 10,
    }


def test_reassignment_falls_back_to_actual_nonself_removal_for_saturated_row() -> None:
    from types import SimpleNamespace

    scene = SimpleNamespace(
        environment_ids=np.asarray([[10, 11]]),
        module_ids=np.asarray([[1, 2]]),
        environment_mask=np.asarray([[True, True]]),
        module_mask=np.asarray([[True, True]]),
    )
    trace = {
        "anchor_indices": np.asarray([[[0]]]),
        "anchor_weights": np.asarray([[[1.0]]]),
        "anchor_read_indices": np.asarray([[0, 0, 0]]),
        "anchor_read_E_slots": np.asarray([0]),
        "anchor_read_member_read_index": np.asarray([0, 0, 0, 0]),
        "anchor_read_original_id": np.asarray([1, 2, 10, 11]),
        "anchor_read_is_module": np.asarray([True, True, False, False]),
    }

    intervention = _reassignment_intervention(scene, {"trace": trace})
    assert intervention == {
        "kind": "remove_member",
        "requested_kind": "reassign_member",
        "fallback_reason": "the selected full E row already contains every distinct eligible present target member",
        "member_role": "M",
        "member_id": 1,
        "anchor_id": 10,
    }


def test_u_diagnostics_retain_matching_e_rows_and_actual_full_restricted_pi() -> None:
    from types import SimpleNamespace
    import torch
    from honf_forward_core.content_hypergraph.contracts import QueryBatch

    module_ids = np.asarray([[1, 2]])
    environment_ids = np.asarray([[10, 20, 30, 40]])
    module_mask = np.ones((1, 2), dtype=bool)
    environment_mask = np.ones((1, 4), dtype=bool)
    environment_coordinates = np.asarray([[[0., 0.], [1., 0.], [0., 1.], [1., 1.]]])
    scene = SimpleNamespace(
        module_features=np.asarray([[[0.1, 0.2], [0.3, 0.4]]]),
        environment_features=np.asarray([[[0.5], [0.6], [0.7], [0.8]]]),
        module_coordinates=np.asarray([[[0.1, 0.2], [0.8, 0.9]]]),
        environment_coordinates=environment_coordinates,
        module_ids=module_ids,
        environment_ids=environment_ids,
        module_mask=module_mask,
        environment_mask=environment_mask,
        module_measure=np.asarray([[1.0, 2.0]]),
        environment_measure=np.asarray([[0.25, 0.25, 0.25, 0.25]]),
        module_measure_reference=np.asarray([[1.5, 2.5]]),
        environment_measure_reference=np.asarray([[0.25, 0.25, 0.25, 0.25]]),
        local_radius=np.asarray([[0.1, 0.2]]),
        module_characteristic_lengths=np.asarray([[0.3, 0.4]]),
        environment_characteristic_lengths=np.asarray([[0.1, 0.1, 0.1, 0.1]]),
        coordinate_origin=np.asarray([[0.0, 0.0]]),
        coordinate_scale=np.asarray([[1.0, 1.0]]),
        operating_features=np.asarray([[0.75]]),
        coordinate_frame_key="physical-x-y",
    )
    query_batch = QueryBatch(
        coordinates=torch.tensor([[[0., 0.], [0.3, 0.2], [0.7, 0.8], [1., 1.]]]),
        query_mask=torch.ones((1, 4), dtype=torch.bool),
    )
    member_axis = 6
    H = np.arange(member_axis * member_axis, dtype=np.float32).reshape(1, member_axis, member_axis)
    retrieval = np.eye(member_axis, dtype=np.float32)[None]
    organization = SimpleNamespace(
        H_EM_applied=np.arange(8, dtype=np.float32).reshape(1, 4, 2),
        H_EE_applied=np.arange(16, dtype=np.float32).reshape(1, 4, 4),
        H=H,
        retrieval_R=retrieval,
        H_self=np.zeros_like(H),
    )
    prepared = SimpleNamespace(
        organization=organization,
        scene_batch=scene,
        query_batch=query_batch,
    )

    class FakeCore:
        def decode(self, _organization, query, *, intervention, return_trace):
            assert return_trace is True
            pi = np.tile(np.asarray([[0.1, 0.2, 0.15, 0.15, 0.2, 0.2]], dtype=np.float32),
                         (1, query.num_queries, 1))
            return SimpleNamespace(
                trace={"proposal_weights_joint": pi, "pi_joint": pi,
                       "address_original_ids": np.asarray([[1, 2, 10, 20, 30, 40]])},
                work_counts={f"read_{intervention['read_mode']}": int(query.num_queries)},
            )

    model = SimpleNamespace(core=FakeCore(), architecture="U-test")
    batch = SimpleNamespace(
        targets=SimpleNamespace(case_ids=("case-1",)),
        case_keys=("case-1",),
        scene_inputs=object(),
        receivers=SimpleNamespace(),
    )

    class Provider:
        @staticmethod
        def native_evaluation_batches(*, split, case_keys, microbatch_size):
            assert (split, tuple(case_keys), microbatch_size) == ("validation", ("case-1",), 1)
            return iter((batch,))

        @staticmethod
        def predict_native(*_args, **_kwargs):
            return SimpleNamespace(native_prepared=prepared, work={"prepared": 1}), {}

    result = _collect_u_organization_diagnostics(model, Provider(), {"epoch": 100}, ("case-1",), "cpu")
    assert result["status"] == "completed_global_entry_E_row_views"
    assert result["deterministic_E_row_views"]["case-1"]["deterministic_E_row_panel"] == [10, 40, 20, 30]
    first_row = result["deterministic_E_row_views"]["case-1"]["rows"][0]
    assert first_row["H_global_row_at_M_plus_E_slot"] == H[0, 2].tolist()
    assert first_row["retrieval_R_global_address_row_at_M_plus_E_slot"] == retrieval[0, 2].tolist()
    assert "not an anchor choice" in first_row["view_semantics"]
    scene_geometry = result["deterministic_E_row_views"]["case-1"]["scene_geometry"]
    assert scene_geometry["module"]["coordinates"] == [[0.1, 0.2], [0.8, 0.9]]
    assert scene_geometry["environment"]["coordinates"] == environment_coordinates[0].tolist()
    assert scene_geometry["module"]["physical_measure"] == [1.0, 2.0]
    for mode in ("full", "restricted"):
        trace = result["global_query_distribution_traces"][mode]["case-1"]
        assert np.asarray(trace["actual_global_proposal_pi"]).shape == (1, 3, 6)
    assert result["anchored_work_counters"]["status"] == "not_applicable_global_entry"


def test_u_native_evaluator_skips_legacy_member_route_collection(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace
    import honf_runtime.contextual_frontier_evaluation as checkpoint_runtime
    import honf_runtime.member_routing_evaluation as native_runtime
    import honf_runtime.r13_8_anchored_organization_evaluation as evaluation_runtime

    checkpoint = tmp_path / "u500.pt"
    checkpoint.write_bytes(b"fixture")
    output_root = tmp_path / "round"
    output_root.mkdir()
    output_parent = output_root / "evaluations"
    output_parent.mkdir()
    provider = SimpleNamespace(
        training_case_ids=("0006", "0051", "0322", "0231"),
        validation_case_ids=("0277",),
        recipe={
            "storage": {
                "output_root": str(output_root),
                "max_output_bytes": 10_000_000,
                "max_peak_temporary_bytes": 1_000_000,
                "minimum_free_headroom_bytes": 1_000_000,
            },
            "evaluation": {"max_output_bytes": 10_000_000},
        },
    )
    model = SimpleNamespace(
        architecture="R13-7-U-ME-SingleDistributionJoint95",
        core=SimpleNamespace(schema="interaction_reused_single_distribution_member_routing_v1"),
    )
    calls = []

    monkeypatch.setattr(checkpoint_runtime, "load_contextual_checkpoint",
                        lambda *_args: (model, provider, {"epoch": 500}))
    monkeypatch.setattr(checkpoint_runtime, "_checkpoint_selection", lambda *_args: {})
    monkeypatch.setattr(evaluation_runtime, "_validate_checkpoint_population", lambda *_args: {
        "arm": "U", "validation_case_ids": ["0277"],
        "validation_population": "fixed25_VAL22",
    })
    monkeypatch.setattr(evaluation_runtime, "_select_validation_panel", lambda *_args: ("0277",))
    monkeypatch.setattr(native_runtime, "_storage_snapshot", lambda *_args, **_kwargs: {
        "free_before_bytes": 10**12,
        "free_after_bytes": 10**12,
    })

    def fake_native_split(_model, _provider, _payload, split, case_ids, *, export_ids=frozenset(), collect_routes=False):
        calls.append((split, bool(collect_routes)))
        return {
            "case_ids": list(case_ids),
            "case_count": len(case_ids),
            "metrics": {},
            "maintained_native_reduction": {},
            "executed_work_counts": {},
            "representative_exports": [],
            "representative_arrays": {},
        }

    monkeypatch.setattr(native_runtime, "_evaluate_native_split", fake_native_split)
    monkeypatch.setattr(evaluation_runtime, "_write_representative_arrays", lambda *_args: {
        "compressed_bytes": 0, "uncompressed_bytes": 0,
    })
    monkeypatch.setattr(evaluation_runtime, "_collect_u_organization_diagnostics", lambda *_args: {
        "status": "completed_global_entry_E_row_views",
    })

    result = evaluate(checkpoint, "cpu", output_parent / "u500", representative_case_ids=("0277",))

    assert result["status"] == "completed"
    assert calls == [
        ("validation", False), ("train", False),
        ("validation", False), ("train", False),
    ]
    assert (output_parent / "u500" / "result.json").is_file()


def test_boundary_provider_summary_retains_each_probe_cost_and_artificial_face_label() -> None:
    offsets = [-0.5, -0.2, -0.05, 0.0, 0.05, 0.2, 0.5]
    values = np.asarray(offsets, dtype=np.float64)[:, None]
    counts = [{"prepare_prepared_e_directory_rows": 1, "decoder_query_joint_field_head_rows": 1}
              for _offset in offsets]
    score_trace = [{"member_score_union": {"E": {"indices": [0], "raw_scores": [0.25]}},
                    "anchor_read": {"calibrated_scores": [0.5]}}
                   for _offset in offsets]
    anchor_indices = np.tile(np.asarray([[5, 6, 29, 30]]), (len(offsets), 1))
    anchor_weights = np.full((len(offsets), 4), 0.25)
    summary = _summarize_provider_boundary({
        "locator_rebuilt_for_each_probe": True,
        "read_mode": "restricted",
        "locator_mode": "local_e_cover",
        "offsets_in_spacing": offsets,
        "probe_coordinates": [[value, 0.0] for value in offsets],
        "environment_original_ids": list(range(40)),
        "anchor_indices": anchor_indices,
        "anchor_weights": anchor_weights,
        "decoder": {"values": values, "output_scale": [1.0]},
        "native_extraction": {
            "values": values, "output_scale": [1.0],
            "native_grid_nx_ny": [128, 64], "native_grid_cell_count": 8192,
            "interpolation_valid_mask": np.ones(len(offsets), dtype=bool),
            "sampling_method": "fixed-grid piecewise bilinear",
            "off_grid_curl_evaluated": False,
        },
        "physical_interface_control": {
            "kind": "artificial_coverage_face", "prescribed_controls_held_fixed": True,
        },
        "score_trace_by_probe": score_trace,
        "work_counts_by_probe": counts,
        "shared_fixed_work_counts": {
            "prepare_prepared_e_directory_rows": 192,
            "prepare_prepared_e_directory_full_entries": 37440,
            "prepare_prepared_e_directory_joint95_entries": 33600,
            "prepare_prepared_e_directory_sort_entries": 37440,
            "prepare_prepared_e_directory_compile_pair_rows": 37440,
            "prepare_prepared_e_directory_positive_exchange_entries": 37440,
            "prepare_query_anchor_locator_rows": 8192,
            "prepare_query_anchor_index_rows": 8192,
            "native_query_unique_union_raw_score_pairs_M": 24576,
            "native_query_unique_union_raw_score_pairs_E": 1409024,
            "native_query_anchor_member_pairs_M": 24576,
            "native_query_anchor_member_pairs_E": 1409024,
            "native_query_anchor_normalization_rows_M": 8192,
            "native_query_anchor_normalization_rows_E": 8192,
            "native_query_anchor_typed_gate_rows_M": 8192,
            "native_query_anchor_typed_gate_rows_E": 8192,
            "native_query_anchor_coalesced_value_pairs_M": 24576,
            "native_query_anchor_coalesced_value_pairs_E": 1409024,
            "native_query_anchor_value_reduction_pairs_M": 24576,
            "native_query_anchor_value_reduction_pairs_E": 1409024,
            "native_query_M_member_query_affine_rows": 8192,
            "native_query_E_member_query_affine_rows": 8192,
            "native_query_joint_field_head_rows": 8192,
        },
    })
    assert summary["physical_interface_claim"].startswith("artificial E-coverage boundary")
    assert summary["read_mode"] == "restricted"
    assert summary["locator_cover_regularity_expectation"].startswith("smoothstep cover is expected to be C1")
    assert summary["live_anchor_carrier_by_probe"][0]["environment_original_ids"] == [5, 6, 29, 30]
    reuse = summary["active_anchor_reuse_across_probes"]
    assert reuse["unique_original_ids"] == 4
    assert reuse["duplicate_positive_ids_within_probe"] == 0
    assert {row["environment_original_id"]: row["probe_count"]
            for row in reuse["reused_original_ids"]} == {5: 7, 6: 7, 29: 7, 30: 7}
    assert len(summary["work_counts_by_probe"]) == len(offsets)
    assert summary["raw_score_trace_by_probe"] == score_trace
    assert summary["separated_work_counts_by_probe"][0]["observed"]["directory_rows"] == 1
    assert summary["native_extraction"]["regularity_assessment"] == (
        "assessed_all_probes_within_native_cell_center_hull"
    )
    assert summary["native_extraction"]["interpolation_valid_mask"] == [True] * len(offsets)
    assert summary["native_extraction"]["native_grid_nx_ny"] == [128, 64]
    assert summary["native_extraction"]["native_grid_cell_count"] == 8192
    assert summary["separated_shared_fixed_work_counts"]["observed"]["unique_raw_score_pairs_E"] == 1409024


def test_boundary_summary_preserves_partial_native_hull_without_fitting_masked_values() -> None:
    offsets = [-0.5, -0.2, -0.05, 0.0, 0.05, 0.2, 0.5]
    values = np.asarray(offsets, dtype=np.float64)[:, None]
    shared = {
        "prepare_prepared_e_directory_rows": 1,
        "prepare_prepared_e_directory_full_entries": 1,
        "prepare_prepared_e_directory_joint95_entries": 1,
        "prepare_prepared_e_directory_sort_entries": 1,
        "prepare_prepared_e_directory_compile_pair_rows": 1,
        "prepare_prepared_e_directory_positive_exchange_entries": 1,
        "prepare_query_anchor_locator_rows": 1,
        "prepare_query_anchor_index_rows": 1,
    }
    for suffix in (
        "query_unique_union_raw_score_pairs_M", "query_unique_union_raw_score_pairs_E",
        "query_anchor_member_pairs_M", "query_anchor_member_pairs_E",
        "query_anchor_normalization_rows_M", "query_anchor_normalization_rows_E",
        "query_anchor_typed_gate_rows_M", "query_anchor_typed_gate_rows_E",
        "query_anchor_coalesced_value_pairs_M", "query_anchor_coalesced_value_pairs_E",
        "query_anchor_value_reduction_pairs_M", "query_anchor_value_reduction_pairs_E",
        "query_M_member_query_affine_rows", "query_E_member_query_affine_rows",
        "query_joint_field_head_rows",
    ):
        shared[f"native_{suffix}"] = 1
    valid = np.asarray([False, False, True, True, True, True, True])
    boundary = {
        "locator_rebuilt_for_each_probe": True,
        "read_mode": "full",
        "locator_mode": "local_e_cover",
        "offsets_in_spacing": offsets,
        "probe_coordinates": [[value, 0.0] for value in offsets],
        "environment_original_ids": list(range(40)),
        "anchor_indices": np.tile(np.asarray([[5, 6, 29, 30]]), (len(offsets), 1)),
        "anchor_weights": np.full((len(offsets), 4), 0.25),
        "decoder": {"values": values, "output_scale": [1.0]},
        "native_extraction": {
            "values": np.asarray([[0.0], [0.0], [0.2], [0.0], [0.2], [0.4], [0.6]]),
            "output_scale": [1.0],
            "interpolation_valid_mask": valid,
            "sampling_method": "fixed-grid piecewise bilinear",
            "off_grid_curl_evaluated": False,
        },
        "physical_interface_control": {
            "kind": "artificial_coverage_face", "prescribed_controls_held_fixed": True,
        },
        "score_trace_by_probe": [{} for _ in offsets],
        "work_counts_by_probe": [{"query_joint_field_head_rows": 1} for _ in offsets],
        "shared_fixed_work_counts": shared,
    }

    summary = _summarize_provider_boundary(boundary)

    native = summary["native_extraction"]
    assert native["regularity_assessment"] == "not_assessed_partial_native_center_hull"
    assert native["interpolation_valid_mask"] == valid.tolist()
    assert native["outside_native_cell_center_hull_probe_indices"] == [0, 1]
    assert native["sampled_values"] == boundary["native_extraction"]["values"].tolist()
    assert "one_sided_affine_fit" not in native
    assert "continuity_and_derivative_pairs" not in native
    assert summary["decoder"]["one_sided_affine_fit"]["estimated_limit_gap_max_abs"] == pytest.approx(0.0)


def test_boundary_probe_coordinates_match_hard_faces_and_smooth_knots() -> None:
    from types import SimpleNamespace

    scene = SimpleNamespace(
        environment_coordinates=np.zeros((1, 24 * 8, 2)),
        coordinate_origin=np.asarray([[0.0, -2.0]]),
        coordinate_scale=np.asarray([[12.0, 8.0]]),
    )
    hard_x, hard_y = _thermal_boundary_coordinates(scene, "e_anchor")
    smooth_x, smooth_y, owner_x, owner_y, outer_x, outer_y = _thermal_boundary_coordinates(
        scene, "local_e_cover",
    )

    offsets = np.asarray([-0.5, -0.2, -0.05, -0.005, -0.0005, 0.0, 0.0005, 0.005, 0.05, 0.2, 0.5])
    assert hard_x["boundary_kind"] == "hard_cell_ownership_face"
    assert hard_x["face_coordinate"] == pytest.approx(6.0)
    assert np.asarray(hard_x["probe_coordinates"])[:, 0] == pytest.approx(6.0 + offsets * 0.5)
    assert hard_y["face_coordinate"] == pytest.approx(2.0)
    assert smooth_x["boundary_kind"] == "smooth_cover_center_knot"
    assert smooth_x["face_coordinate"] == pytest.approx(6.25)
    assert smooth_y["face_coordinate"] == pytest.approx(2.5)
    assert np.asarray(smooth_x["probe_coordinates"])[:, 1] == pytest.approx(np.full(11, 2.5))
    assert np.asarray(smooth_y["probe_coordinates"])[:, 0] == pytest.approx(np.full(11, 6.25))
    assert owner_x["boundary_kind"] == "smooth_cover_former_ownership_face"
    assert owner_x["face_coordinate"] == pytest.approx(6.0)
    assert owner_y["boundary_kind"] == "smooth_cover_former_ownership_face"
    assert owner_y["face_coordinate"] == pytest.approx(2.0)
    assert np.asarray(owner_x["probe_coordinates"])[:, 1] == pytest.approx(np.full(11, 2.5))
    assert np.asarray(owner_y["probe_coordinates"])[:, 0] == pytest.approx(np.full(11, 6.25))
    assert outer_x["boundary_kind"] == "smooth_cover_outer_strip_junction"
    assert outer_x["probe_coordinates"][0][0] == pytest.approx(0.0)
    assert outer_y["probe_coordinates"][0][1] == pytest.approx(-2.0)


@pytest.mark.parametrize(
    ("offsets", "values", "message"),
    [
        ([-0.2, 0.2, -0.5, 0.5], [[0.0], [1.0], [0.0], [1.0]], "exactly one face sample"),
        ([-0.2, 0.0, 0.3], [[0.0], [0.0], [1.0]], "symmetric negative probe"),
        ([-0.2, 0.0, 0.2], [[0.0], [0.0], [np.nan]], "finite values"),
    ],
)
def test_boundary_sweep_rejects_incomplete_or_nonfinite_probe_sets(offsets, values, message) -> None:
    with pytest.raises(ValueError, match=message):
        summarize_boundary_sweep(offsets, values, evaluation_level="decoder")

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from compact_retrieval_evaluate import (
    REPRESENTATIVE_IDS,
    TRAIN24_IDS,
    VAL89_IDS,
    _compact_trace_sample_arrays,
    _prediction_difference,
    choose_legal_module_translation,
    _topology_from_organization,
    compare_native_board_pair,
    compare_native_board_pair_arrays,
    native_response_coefficient_arrays,
    prepare_copy_with_disabled_group,
    response_payload_for_arm,
    resolve_native_board_case_ids,
    summarize_cached_heat_reuse_arrays,
    summarize_parent_heat_reuse_cli,
    summarize_compact_trace,
    summarize_native_case_rows,
    summarize_role_group_mass,
    translated_structure,
    _geometry_probe_one_case,
    _probe_topology_seam,
    validate_w6_ranking_reference,
    validate_case_panel,
)


def _metric_row(case_id: str, module_count: int, offset: float) -> dict:
    fields = {
        name: {"rmse": offset + index}
        for index, name in enumerate(("u", "v", "p", "omega", "temperature_fluid",
                                      "temperature_surface", "temperature_material", "q_proxy"))
    }
    module_peaks = [
        {"native_module_id": index, "prediction": offset + 8, "target": 0.0,
         "error": offset + 8}
        for index in range(2)
    ]
    return {"case_id": case_id, "module_count": module_count,
            "fields": fields, "module_peaks": module_peaks}


def test_round12_panel_binds_exact_canonical_val89() -> None:
    assert len(VAL89_IDS) == 89
    assert VAL89_IDS[0] == "0274" and VAL89_IDS[28] == "0302"
    assert VAL89_IDS[29] == "0633" and VAL89_IDS[-1] == "0692"
    assert validate_case_panel(reversed(VAL89_IDS)) == tuple(reversed(VAL89_IDS))
    with pytest.raises(ValueError, match="duplicate"):
        validate_case_panel((*VAL89_IDS[:-1], VAL89_IDS[0]))
    with pytest.raises(ValueError, match="differs"):
        validate_case_panel(VAL89_IDS[:-1])
    assert resolve_native_board_case_ids("validation", None) == VAL89_IDS
    with pytest.raises(ValueError, match="canonical ordered VAL89"):
        resolve_native_board_case_ids("validation", tuple(reversed(VAL89_IDS)))
    assert resolve_native_board_case_ids("train", None) == TRAIN24_IDS
    assert resolve_native_board_case_ids("train", TRAIN24_IDS) == TRAIN24_IDS
    with pytest.raises(ValueError, match="ordered fixed TRAIN24"):
        resolve_native_board_case_ids("train", ("0006", "0254"))


def test_native_metric_summary_keeps_equal_case_tails_and_m_strata() -> None:
    ids = ("a", "b", "c")
    rows = [_metric_row("a", 3, 1.0), _metric_row("b", 3, 3.0), _metric_row("c", 5, 2.0)]
    summary = summarize_native_case_rows(rows, expected_case_ids=ids)
    assert summary["case_count"] == 3
    assert summary["module_count_strata"] == {"M3": 2, "M5": 1}
    assert summary["fields"]["u"]["equal_case_mean"] == 2.0
    assert summary["fields"]["u"]["worst_case_id"] == "b"
    assert summary["fields"]["q_proxy"]["by_module_count"]["M3"]["case_count"] == 2
    assert summary["fields"]["module_peak_endpoint"]["equal_case_mean"] == 10.0


def test_compact_trace_summary_uses_exact_positive_support() -> None:
    weights = np.asarray([[[1.0, 0.0], [0.0, 0.5]]])
    membership = np.asarray([[[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]])
    final_access = np.asarray([[[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]])
    summary = summarize_compact_trace({
        "compact_group_weights": weights,
        "compact_membership": membership,
        "compact_group_mask": np.asarray([[True, True]]),
        "compact_group_types": np.asarray([["M", "E"]]),
        "local_module_envelope": np.zeros((1, 2, 0)),
        "original_record_present": np.asarray([[True, True, True]]),
        "original_record_eligible": np.asarray([[True, True, True]]),
        "original_record_count": np.asarray([3]),
        "gates": final_access,
    })
    row = summary["rows"][0]
    assert row["positive_C_memberships"] == 4
    assert row["group_types_live_counts"] == {"M": 1, "E": 1}
    assert row["positive_final_access_entries"] == 4
    assert row["source_count_N"] == 3
    assert row["allocated_source_slots_N"] == 3
    assert row["represented_support_equals_final_positive_support"] is True


def test_compact_trace_composes_local_module_envelope_and_masks_padding() -> None:
    trace = {
        "compact_group_weights": np.asarray([[[1.0], [1.0]]]),
        "compact_membership": np.asarray([[[1.0, 0.0, 1.0, 0.0]]]),
        "compact_group_mask": np.asarray([[True]]),
        "local_module_envelope": np.asarray([[[0.0, 0.25], [0.0, 0.25]]]),
        "query_valid_mask": np.asarray([[True, False]]),
        "original_record_present": np.asarray([[True, True, True, False]]),
        "original_record_eligible": np.asarray([[True, True, True, False]]),
        "original_record_count": np.asarray([3]),
        "gates": np.asarray([[[1.0, 0.25, 1.0, 0.0], [1.0, 0.25, 1.0, 1.0]]]),
    }
    row = summarize_compact_trace(trace)["rows"][0]
    assert row["source_count_N"] == 3
    assert row["allocated_source_slots_N"] == 4
    assert row["valid_query_count"] == 1
    assert row["piC_positive_support_entries"] == 2
    assert row["support_added_by_local_module_envelope"] == 1
    assert row["positive_final_access_entries"] == 3
    assert row["represented_support_equals_final_positive_support"] is True


def test_compact_trace_support_respects_record_eligibility() -> None:
    trace = {
        "compact_group_weights": np.asarray([[[1.0]]]),
        "compact_membership": np.asarray([[[1.0, 1.0, 1.0, 0.0]]]),
        "compact_group_mask": np.asarray([[True]]),
        "local_module_envelope": np.asarray([[[0.0, 0.25, 0.0]]]),
        "original_record_present": np.asarray([[True, True, True, False]]),
        "original_record_eligible": np.asarray([[True, False, True, False]]),
        "original_record_count": np.asarray([3]),
        "gates": np.asarray([[[1.0, 0.0, 1.0, 0.0]]]),
    }
    row = summarize_compact_trace(trace)["rows"][0]
    assert row["source_count_N"] == 3
    assert row["source_eligible_count"] == 2
    assert row["positive_C_memberships"] == 2
    assert row["positive_C_memberships_on_ineligible_present_records"] == 1
    assert row["piC_positive_support_entries"] == 2
    assert row["local_module_envelope_positive_support_entries"] == 0
    assert row["positive_final_access_entries"] == 2
    assert row["represented_support_equals_final_positive_support"] is True


def test_response_arm_label_is_derived_without_mutating_sealed_identity() -> None:
    payload = {"epoch": 1000, "experiment_identity": {"config_name": "thermal_u_h_full5000_v1"}}
    labeled = response_payload_for_arm(payload)
    assert labeled["experiment_identity"]["treatment"] == "U-H"
    assert "treatment" not in payload["experiment_identity"]
    with pytest.raises(ValueError, match="conflicts"):
        response_payload_for_arm(payload, "U-S")


def test_saved_ranking_controls_bind_original_six_id_pool_tau_and_pressure_limit() -> None:
    from compact_retrieval_evaluate import W6_CANDIDATE_IDS

    rows = [{"candidate_id": candidate_id} for candidate_id in W6_CANDIDATE_IDS]
    saved = {"original_constraints": {"tau_temperature": 2.0}}
    study = {"pressure_drop_limit": 0.13649363704785286}
    assert tuple(row["candidate_id"] for row in validate_w6_ranking_reference(rows, saved, study)) == W6_CANDIDATE_IDS
    with pytest.raises(ValueError, match="ordered six-ID"):
        validate_w6_ranking_reference(list(reversed(rows)), saved, study)
    with pytest.raises(ValueError, match="tau=2.0"):
        validate_w6_ranking_reference(rows, {"original_constraints": {"tau_temperature": 1.0}}, study)
    with pytest.raises(ValueError, match="pressure limit"):
        validate_w6_ranking_reference(rows, saved, {"pressure_drop_limit": 0.14})


def test_cached_heat_reuse_reports_absolute_errors_for_all_saved_heat_vectors() -> None:
    arrays = {}
    for label in ("heat_transfer_plus", "heat_transfer_minus"):
        arrays[f"{label}__fluid_fields__promoted_increment"] = np.asarray(
            [[0, 0, 0, 0, 1e-8], [100, 200, 300, 400, 500]], dtype=np.float64,
        )
        arrays[f"{label}__fluid_fields__prediction_delta"] = np.zeros((2, 5), dtype=np.float64)
        arrays[f"{label}__fluid_fields__reference_delta"] = np.zeros((2, 5), dtype=np.float64)
        arrays["heat__fluid_fields__valid"] = np.asarray([True, False])
        arrays[f"{label}__interface__promoted_increment"] = np.asarray([[1, 2], [100, 200]], dtype=np.float64)
        arrays[f"{label}__interface__prediction_delta"] = np.zeros((2, 2), dtype=np.float64)
        arrays[f"{label}__interface__reference_delta"] = np.zeros((2, 2), dtype=np.float64)
        arrays["heat__interface__valid"] = np.asarray([[True, True], [False, False]])
        arrays[f"{label}__solid_temperature__promoted_increment"] = np.asarray([[3], [30]], dtype=np.float64)
        arrays[f"{label}__solid_temperature__prediction_delta"] = np.zeros((2, 1), dtype=np.float64)
        arrays[f"{label}__solid_temperature__reference_delta"] = np.zeros((2, 1), dtype=np.float64)
        arrays["heat__solid_temperature__valid"] = np.asarray([True, False])
    summary = summarize_cached_heat_reuse_arrays(arrays, "0001")
    assert summary["comparison_count"] == 6
    first = summary["comparisons"][0]
    assert first["valid_values"] == 5
    assert "heterogeneous channel units" in first["aggregate_interpretation"]
    assert first["cached_K_vs_fresh_native"]["absolute_rmse"] == pytest.approx(1e-8 / np.sqrt(5))
    assert first["component_comparisons"]["temperature"]["valid_values"] == 1
    assert first["component_comparisons"]["temperature"]["cached_K_vs_fresh_native"]["absolute_rmse"] == pytest.approx(1e-8)
    assert first["component_comparisons"]["temperature"]["cached_K_vs_physical_reference"]["absolute_rmse"] == pytest.approx(1e-8)
    assert first["component_comparisons"]["u"]["cached_K_vs_fresh_native"]["absolute_rmse"] == 0.0
    interface = summary["comparisons"][1]
    assert interface["component_comparisons"]["temperature"]["cached_K_vs_physical_reference"]["absolute_rmse"] == pytest.approx(1.0)
    assert interface["component_comparisons"]["q_proxy"]["cached_K_vs_physical_reference"]["absolute_rmse"] == pytest.approx(2.0)
    material = summary["comparisons"][2]
    assert material["component_comparisons"]["temperature"]["valid_values"] == 1
    assert "relative_l2" not in first["cached_K_vs_fresh_native"]


def test_heat_reuse_cli_accepts_arbitrary_arms_and_recorded_epochs(tmp_path: Path) -> None:
    from types import SimpleNamespace
    import json

    response_results = []
    for arm, epoch in (("R0", 40), ("R1", 1000)):
        arm_dir = tmp_path / arm
        arm_dir.mkdir()
        families = []
        for family_id in ("0001", "0318", "0333", "0348"):
            arrays = {}
            for heat_label in ("heat_transfer_plus", "heat_transfer_minus"):
                for role, shape, valid in (
                    ("fluid_fields", (2, 5), np.asarray([True, False])),
                    ("interface", (2, 2), np.asarray([[True, True], [False, False]])),
                    ("solid_temperature", (2, 1), np.asarray([True, False])),
                ):
                    prefix = f"{heat_label}__{role}__"
                    arrays[prefix + "promoted_increment"] = np.zeros(shape, dtype=np.float64)
                    arrays[prefix + "prediction_delta"] = np.zeros(shape, dtype=np.float64)
                    arrays[prefix + "reference_delta"] = np.zeros(shape, dtype=np.float64)
                    arrays[f"heat__{role}__valid"] = valid
            np.savez_compressed(arm_dir / f"family_{family_id}.npz", **arrays)
            families.append({"family_id": family_id, "arrays": f"family_{family_id}.npz"})
        result_path = arm_dir / "response.json"
        result_path.write_text(json.dumps({
            "schema": "round9_saved_physical_response_replay_v1",
            "dataset": "thermal",
            "completed_epoch": epoch,
            "optimizer_updates": 0,
            "new_solver_attempts": 0,
            "families": families,
        }))
        response_results.append(f"{arm}={result_path}")

    output = tmp_path / "summary"
    summarize_parent_heat_reuse_cli(SimpleNamespace(response_result=response_results, output=output))
    summary = json.loads((output / "result.json").read_text())
    assert summary["schema"] == "round12_fixed_layout_K_reuse_summary_v2"
    assert {arm: row["checkpoint_epoch"] for arm, row in summary["arms"].items()} == {"R0": 40, "R1": 1000}
    assert all(len(row["families"]) == 4 for row in summary["arms"].values())


def test_representative_panels_are_explicit_and_exact() -> None:
    assert resolve_native_board_case_ids("validation", None, representative_panel=True) == REPRESENTATIVE_IDS["validation"]
    assert resolve_native_board_case_ids("train", None, representative_panel=True) == REPRESENTATIVE_IDS["train"]
    with pytest.raises(ValueError, match="exact ordered panel"):
        resolve_native_board_case_ids("validation", ("0274", "0293", "0658", "0692"), representative_panel=True)
    with pytest.raises(ValueError, match="canonical ordered VAL89"):
        resolve_native_board_case_ids("validation", REPRESENTATIVE_IDS["validation"])


def test_role_group_mass_uses_interpolated_native_receivers_and_equal_role_rule() -> None:
    from types import SimpleNamespace

    trace = {
        "compact_group_weights": np.asarray([[[0.8, 0.2], [0.1, 0.9]]]),
        "compact_group_mask": np.asarray([[True, True]]),
        "query_valid_mask": np.asarray([[True, True]]),
    }
    prepared = SimpleNamespace(
        stencils={"fluid": SimpleNamespace(
            indices=np.asarray([[[0, 0, 0, 0], [1, 1, 1, 1]]]),
            weights=np.asarray([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]),
            valid=np.asarray([[True, True]]),
        )},
        source_present=np.asarray([[True]]),
    )
    summary = summarize_role_group_mass(trace, prepared)
    assert summary["roles"]["fluid"]["valid_native_receivers"] == 2
    assert summary["roles"]["fluid"]["mean_group_mass_per_native_receiver"] == pytest.approx([0.45, 0.55])
    assert summary["selected_group_index"] == 1


def test_copy_only_group_intervention_rebuilds_masks_and_preserves_original() -> None:
    from dataclasses import dataclass
    import torch
    from honf_forward_core.native_retrieval.contracts import tensor_versions
    from compact_retrieval_evaluate import _compact_group_version_stamp

    @dataclass(frozen=True)
    class Compact:
        membership: object
        group_mask: object
        group_types: object
        controls: object
        centroids: object
        lengths: object
        extents: object
        pair_logits: object
        pair_mask: object
        pair_slots: object
        bank_group_indices: object
        bank_group_masks: object

    @dataclass
    class Organization:
        compact_groups: object
        compact_group_versions: object

    @dataclass(frozen=True)
    class Prepared:
        organization: object

    compact = Compact(
        membership=torch.ones(1, 2, 3), group_mask=torch.tensor([[True, True]]),
        group_types=torch.tensor([[0, 1]]), controls=torch.ones(1, 2, 4),
        centroids=torch.ones(1, 2, 2), lengths=torch.ones(1, 2, 2), extents=torch.ones(1, 2, 2),
        pair_logits=torch.zeros(1, 1), pair_mask=torch.ones(1, 1, dtype=torch.bool),
        pair_slots=torch.zeros(1, 1, 2, dtype=torch.long),
        bank_group_indices=(torch.tensor([[0]]), torch.tensor([[1]])),
        bank_group_masks=(torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool)),
    )
    tensors = [compact.membership, compact.group_mask, compact.group_types, compact.controls,
               compact.centroids, compact.lengths, compact.extents, compact.pair_logits,
               compact.pair_mask, compact.pair_slots, *compact.bank_group_indices, *compact.bank_group_masks]
    original = Prepared(Organization(compact, tensor_versions(*tensors)))
    changed, transition = prepare_copy_with_disabled_group(original, batch_index=0, group_index=0)
    assert original.organization.compact_groups.group_mask.tolist() == [[True, True]]
    assert changed.organization.compact_groups.group_mask.tolist() == [[False, True]]
    assert [mask.tolist() for mask in changed.organization.compact_groups.bank_group_masks] == [[[False]], [[True]]]
    assert changed.organization.compact_group_versions == _compact_group_version_stamp(changed.organization.compact_groups)
    assert transition["live_K_after"] == 1 and transition["allocated_K_after"] == 2


def test_native_response_coefficients_recompute_present_module_columns() -> None:
    import torch
    from types import SimpleNamespace

    class FakeAdapter:
        def apply_native_with_work_counts(self, prepared, heat, *, increment=False):
            assert increment is True
            output = {
                "fluid_temperature": (heat[:, 0:1] + 3.0 * heat[:, 2:3])[:, None, :, None],
                "pred_interface": heat[:, :, None, None].expand(-1, -1, 2, 1),
                "pred_internal_temperature": heat[:, :, None, None].expand(-1, -1, 3, 1),
            }
            return output, {"decode_calls": 1}

    prepared = SimpleNamespace(
        source_present=torch.tensor([[True, False, True]]),
        scene_batch=SimpleNamespace(module_features=torch.zeros(1, 3, 2)),
    )
    arrays = native_response_coefficient_arrays(FakeAdapter(), prepared)
    assert arrays["response_K_source_module_indices"].tolist() == [0, 2]
    assert arrays["response_K_fluid_temperature"].reshape(2, -1)[:, 0].tolist() == [1.0, 3.0]
    assert len(arrays["response_K_column_work_counts"]) == 2


def test_legal_translation_is_copy_only_and_keeps_geometry_in_domain() -> None:
    import torch

    structure = {
        "module_centers": torch.tensor([[[0.25, 0.5], [0.70, 0.5]]]),
        "module_present": torch.tensor([[True, True]]),
        "material_params": torch.tensor([[0.01, 0.02, 0.03, 1.0, 2.0, 0.04]]),
        "domain_length_x": torch.tensor([[1.0]]),
    }
    selected = choose_legal_module_translation(structure)
    moved = translated_structure(structure, selected["module_index"], 0, selected["nominal_shift"])
    assert selected["maximum_legal_shift_magnitude"] > abs(selected["nominal_shift"]) > 0
    assert moved["module_centers"][0, selected["module_index"], 0].item() != structure["module_centers"][0, selected["module_index"], 0].item()
    assert torch.equal(structure["module_centers"], torch.tensor([[[0.25, 0.5], [0.70, 0.5]]]))


def test_geometry_probe_rebuilds_live_input_and_separates_fixed_cut_from_seam() -> None:
    from types import SimpleNamespace
    import torch
    from honf_forward_core.content_hypergraph.contracts import QueryBatch

    class Topology:
        def __init__(self, merged):
            self.merged = merged

        def as_dict(self):
            return {
                "candidate_pairs": [[0, 1, 0]],
                "accepted_pairs": [[0, 1, 0]] if self.merged else [],
                "group_members": [[0, 1]] if self.merged else [[0], [1]],
                "group_types": [0] if self.merged else [0, 1],
                "active_group_keys": [[0, [1, 2]]] if self.merged else [[0, [1]], [1, [2]]],
            }

    class FakeCore:
        query_tile_size = 8

        def prepare(self, structure):
            x = structure["module_centers"][0, 0, 0]
            merged = bool((x > 0.60).detach().cpu())
            membership = (torch.tensor([[[0.5, 0.5]]], device=x.device) if merged else
                          torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], device=x.device))
            types = torch.tensor([[0]] if merged else [[0, 1]], device=x.device)
            group_mask = torch.ones_like(types, dtype=torch.bool)
            compact = SimpleNamespace(
                topology=(Topology(merged),),
                pair_logits=(x - 0.60).reshape(1, 1),
                membership=membership,
                group_mask=group_mask,
                group_types=types,
                controls=x.expand(1, types.shape[1], 2),
                centroids=x.expand(1, types.shape[1], 2),
                lengths=(x.abs() + 1).expand(1, types.shape[1], 2),
                extents=torch.zeros((1, types.shape[1], 2), device=x.device),
            )
            return SimpleNamespace(compact_groups=compact, center=x)

        def capture_continuous_trace(self, organization, query):
            compact = organization.compact_groups
            groups = compact.group_mask.shape[1]
            count = query.num_queries
            x = organization.center
            if groups == 1:
                weights = torch.ones((1, count, 1), device=x.device)
            else:
                weights = torch.stack((x.expand(count) * 0 + 0.6, x.expand(count) * 0 + 0.4), dim=-1)[None]
            gates = torch.stack((x.expand(count), (1 - x).expand(count)), dim=-1)[None]
            lineage = compact.topology[0].as_dict()
            return {
                "compact_group_weights": weights,
                "gates": gates,
                "local_module_envelope": torch.zeros((1, count, 1), device=x.device),
                "query_valid_mask": torch.ones((1, count), dtype=torch.bool, device=x.device),
                "query_ids": query.query_ids,
                "compact_membership": compact.membership,
                "compact_group_mask": compact.group_mask,
                "compact_group_types": compact.group_types,
                "compact_group_controls": compact.controls,
                "compact_group_centroids": compact.centroids,
                "compact_group_lengths": compact.lengths,
                "compact_group_extents": compact.extents,
                "compact_live_K": compact.group_mask.sum(-1),
                "compact_allocated_K": compact.group_mask.shape[1],
                "compact_lineage": (lineage,),
                "original_record_present": torch.ones((1, 2), dtype=torch.bool, device=x.device),
                "original_record_eligible": torch.ones((1, 2), dtype=torch.bool, device=x.device),
                "original_record_count": torch.tensor([2], device=x.device),
                "module_ids": torch.tensor([[1]], device=x.device),
                "environment_ids": torch.tensor([[2]], device=x.device),
            }

    class FakeModel:
        def __init__(self):
            self.core = FakeCore()

        def scene_batch(self, structure, *, scene_key=None):
            return structure

        def prepare_native(self, structure, fluid_xy, *, local_query_points, ntheta, scene_key):
            org = self.core.prepare(structure)
            query = QueryBatch(coordinates=fluid_xy,
                               query_ids=torch.tensor([[10, 20]], device=fluid_xy.device),
                               query_mask=torch.ones((1, 2), dtype=torch.bool, device=fluid_xy.device))
            return SimpleNamespace(
                organization=org,
                stencils={"fluid": SimpleNamespace(
                    valid=torch.ones(1, 2, dtype=torch.bool, device=fluid_xy.device),
                    indices=torch.tensor([[[0, 1, 0, 1], [0, 1, 0, 1]]], device=fluid_xy.device),
                    weights=torch.full((1, 2, 4), 0.25, device=fluid_xy.device),
                )},
                query_batch=query,
                center=org.center,
            )

        def apply_native(self, prepared, heat):
            value = prepared.center.square().expand(1, 2, 1)
            return {"fluid_temperature": value}

    structure = {
        "module_centers": torch.tensor([[[0.25, 0.5]]]),
        "module_present": torch.tensor([[True]]),
        "material_params": torch.tensor([[0.01, 0.02, 0.03, 1.0, 2.0, 0.04]]),
        "domain_length_x": torch.tensor([[1.0]]),
    }
    batch = SimpleNamespace(
        scene_inputs=SimpleNamespace(structure=structure, case_ids=("0006",)),
        receivers=SimpleNamespace(
            fluid_xy=torch.tensor([[[0.1, 0.1], [0.9, 0.9]]]),
            local_query_points=torch.zeros(1, 1, 2, 2),
            heat=torch.ones(1, 1),
        ),
    )
    model = FakeModel()
    row = _geometry_probe_one_case(model, batch, "0006")
    assert row["fixed_topology_confirmed_for_derivative"] is True
    assert len(row["finite_difference_h_sweep"]) == 4
    assert row["absolute_derivative_difference"] < 1e-5
    route = row["original_to_translated_live_routing"]
    assert route["receiver_correspondence_checks"]["same_grid_neighbor_ids"]
    assert route["group_ancestry_alignment"]["matched_group_count"] == 2
    assert route["final_access_delta_on_fixed_exterior_fluid_receivers"]["delta_max_abs"] > 0
    seam = _probe_topology_seam(model, batch, row["translation"])
    assert seam["status"] == "accepted_ancestry_change_found"
    gaps = [item["translation_gap"] for item in seam["shrinking_gap_outputs"]]
    jumps = [abs(item["left_to_right_output_jump"]) for item in seam["shrinking_gap_outputs"]]
    assert gaps == sorted(gaps, reverse=True)
    assert jumps[-1] < jumps[0]


def test_geometry_topology_reports_slot_order_separately_from_ancestry() -> None:
    from types import SimpleNamespace
    import torch

    class Topology:
        def __init__(self, members, keys, candidate):
            self.members, self.keys, self.candidate = members, keys, candidate

        def as_dict(self):
            return {"candidate_pairs": self.candidate, "accepted_pairs": [],
                    "group_members": self.members, "group_types": [0, 0],
                    "active_group_keys": self.keys}

    before = SimpleNamespace(compact_groups=SimpleNamespace(
        topology=(Topology([[0], [1]], [[0, [10]], [0, [20]]], [[0, 1, 0]]),),
        pair_logits=torch.tensor([[-1.0]])))
    after = SimpleNamespace(compact_groups=SimpleNamespace(
        topology=(Topology([[1], [0]], [[0, [20]], [0, [10]]], [[1, 0, 0]]),),
        pair_logits=torch.tensor([[-1.0]])))
    left, right = _topology_from_organization(before), _topology_from_organization(after)
    assert left["ordering_signature"] != right["ordering_signature"]
    assert left["ancestry_signature"] == right["ancestry_signature"]


def test_query_trace_export_is_small_and_marks_solid_neural_probes() -> None:
    from types import SimpleNamespace
    import torch
    from honf_forward_core.content_hypergraph.contracts import QueryBatch

    query = QueryBatch(coordinates=torch.arange(8, dtype=torch.float32).reshape(1, 4, 2),
                       query_ids=torch.tensor([[0, 1, 2, 3]]),
                       query_mask=torch.ones((1, 4), dtype=torch.bool))
    stencil = lambda rows: SimpleNamespace(
        indices=torch.tensor([[rows]], dtype=torch.long).reshape(1, len(rows), 4),
        weights=torch.tensor([[[0.7, 0.1, 0.1, 0.1]] * len(rows)]),
        valid=torch.ones((1, len(rows)), dtype=torch.bool),
    )
    prepared = SimpleNamespace(
        query_batch=query,
        source_present=torch.tensor([[True]]),
        native_solid_mask=torch.tensor([[[False, True, False, False]]]),
        stencils={"fluid": stencil([[0, 1, 2, 3]] * 3),
                  "surface": stencil([[1, 2, 3, 0]] * 2),
                  "outside": stencil([[2, 3, 0, 1]] * 2),
                  "material": stencil([[3, 0, 1, 2]] * 2)},
    )
    trace = {
        "compact_group_weights": torch.ones((1, 4, 2)),
        "gates": torch.ones((1, 4, 1)),
        "local_module_envelope": torch.zeros((1, 4, 1)),
        "query_valid_mask": torch.ones((1, 4), dtype=torch.bool),
        "query_ids": torch.tensor([[0, 1, 2, 3]]),
    }
    arrays, manifest = _compact_trace_sample_arrays(
        trace, prepared, fluid_receiver_count=2, local_receiver_count=1,
    )
    assert len(manifest) == 5
    assert arrays["trace_sample_compact_group_weights"].shape == (5, 2)
    assert all(row["physical_fluid_error_receiver"] is False for row in manifest)
    assert any(row["inside_module_solid_neural_probe"] for row in manifest)


def test_representative_boundary_summary_binds_exact_panels_and_support_fields() -> None:
    def board(split, ids, delta):
        rows = []
        traces = {}
        for case_id in ids:
            rows.append({"case_id": case_id, "module_count": 1,
                         "fields": {field: {"rmse": 1.0 + delta} for field in
                                    ("u", "v", "p", "omega", "temperature_fluid", "temperature_surface",
                                     "temperature_material", "q_proxy")},
                         "module_peaks": [{"prediction": 1.0 + delta, "target": 0.0}]})
            traces[case_id] = {"summary": {"rows": [{
                "live_group_count_K": 2 + delta, "allocated_group_count_K": 3,
                "positive_C_memberships": 4 + delta, "piC_positive_support_entries": 5 + delta,
                "positive_final_access_entries": 6 + delta,
                "support_added_by_local_module_envelope": 1,
            }]}}
        return {"split": split, "case_ids": list(ids), "records": rows,
                "organization_traces": traces, "checkpoint": "checkpoint.pt", "checkpoint_epoch": 40}

    ids = REPRESENTATIVE_IDS["validation"]
    result = compare_native_board_pair(board("validation", ids, 0), board("validation", ids, 1))
    assert result["organization_support_changes"][ids[0]]["live_group_count_K"]["delta"] == 1
    assert result["field_error_changes"]["u"][0]["error_delta_after_minus_before"] == 1
    with pytest.raises(ValueError, match="exact representative panel"):
        compare_native_board_pair(board("validation", ids[1:] + ids[:1], 0), board("validation", ids[1:] + ids[:1], 1))


def test_boundary_array_comparison_includes_physical_and_response_K_deltas(tmp_path: Path) -> None:
    ids = REPRESENTATIVE_IDS["train"]

    def board(delta):
        return {
            "split": "train", "case_ids": list(ids), "checkpoint": f"e{delta}.pt", "checkpoint_epoch": 40,
            "records": [{"case_id": case_id, "module_count": 1,
                         "fields": {name: {"rmse": 1.0 + delta} for name in
                                    ("u", "v", "p", "omega", "temperature_fluid", "temperature_surface",
                                     "temperature_material", "q_proxy")},
                         "module_peaks": [{"prediction": 1.0 + delta, "target": 0.0}]}
                        for case_id in ids],
            "organization_traces": {case_id: {"summary": {"rows": [{
                "live_group_count_K": 2, "allocated_group_count_K": 3,
                "positive_C_memberships": 4, "piC_positive_support_entries": 5,
                "positive_final_access_entries": 6, "support_added_by_local_module_envelope": 1,
            }]}} for case_id in ids},
        }

    before_dir, after_dir = tmp_path / "before", tmp_path / "after"
    before_dir.mkdir()
    after_dir.mkdir()
    for case_id in ids:
        for directory, delta in ((before_dir, 0.0), (after_dir, 1.0)):
            field = np.zeros((2, 5), dtype=np.float32)
            interface = np.zeros((1, 2, 2), dtype=np.float32)
            material = np.zeros((1, 2), dtype=np.float32)
            response_fluid = np.zeros((1, 1, 2, 1), dtype=np.float32)
            response_interface = np.zeros((1, 1, 1, 2, 2), dtype=np.float32)
            response_material = np.zeros((1, 1, 1, 2, 1), dtype=np.float32)
            if delta:
                field[:] = np.asarray([1, 2, 3, 4, 5], dtype=np.float32)
                interface[0] = [[1, 10], [100, 20]]
                material[0, :] = [3, 30]
                response_fluid[0, 0, :, 0] = [4, 40]
                response_interface[0, 0, 0] = [[5, 6], [50, 60]]
                response_material[0, 0, 0, :, 0] = [7, 70]
            np.savez_compressed(
                directory / f"case_{case_id}.npz",
                field_prediction=field,
                fluid_valid=np.asarray([True, False]),
                interface_prediction=interface,
                interface_valid=np.asarray([[True, True]]),
                outside_valid=np.asarray([[False, True]]),
                material_prediction=material,
                material_valid=np.asarray([[True, False]]),
                response_K_fluid_temperature=response_fluid,
                response_K_pred_interface=response_interface,
                response_K_pred_internal_temperature=response_material,
                response_K_source_module_indices=np.asarray([0]),
            )
    result = compare_native_board_pair_arrays(board(0), board(1), before_dir, after_dir)
    row = result["per_case_prediction_and_response_K_deltas"][ids[0]]
    assert row["physical_outputs"]["field_prediction"]["delta_rmse"] == pytest.approx(np.sqrt(11.0))
    assert "descriptive only" in row["physical_outputs"]["field_prediction"]["interpretation"]
    field_components = row["physical_outputs"]["field_prediction_by_component"]
    assert list(field_components) == ["u", "v", "p", "omega", "temperature"]
    assert [field_components[name]["delta_max_abs"] for name in field_components] == [1, 2, 3, 4, 5]
    assert all(field_components[name]["finite_values"] == 1 for name in field_components)
    interface_components = row["physical_outputs"]["interface_prediction_by_component"]
    assert interface_components["temperature"]["finite_values"] == 2
    assert interface_components["temperature"]["delta_max_abs"] == pytest.approx(100.0)
    assert interface_components["q_proxy"]["finite_values"] == 1
    assert interface_components["q_proxy"]["delta_max_abs"] == pytest.approx(20.0)
    assert row["physical_outputs"]["material_prediction"]["finite_values"] == 1
    assert row["response_K"]["response_K_fluid_temperature"]["finite_values"] == 1
    assert row["response_K"]["response_K_fluid_temperature"]["delta_max_abs"] == pytest.approx(4.0)
    response_interface = row["response_K"]["response_K_pred_interface_by_component"]
    assert response_interface["temperature"]["finite_values"] == 2
    assert response_interface["temperature"]["delta_max_abs"] == pytest.approx(50.0)
    assert response_interface["q_proxy"]["finite_values"] == 1
    assert response_interface["q_proxy"]["delta_max_abs"] == pytest.approx(60.0)
    assert row["response_K"]["response_K_pred_internal_temperature"]["finite_values"] == 1
    assert row["response_K"]["response_K_pred_internal_temperature"]["delta_max_abs"] == pytest.approx(7.0)


def test_native_prediction_difference_separates_mixed_channels_and_uses_receiver_masks() -> None:
    before = {
        "pred_field": np.zeros((1, 2, 5), dtype=np.float32),
        "pred_interface": np.zeros((1, 1, 2, 2), dtype=np.float32),
        "pred_internal_temperature": np.zeros((1, 1, 2, 1), dtype=np.float32),
    }
    after = {
        "pred_field": np.broadcast_to(np.asarray([1, 2, 3, 4, 5], dtype=np.float32), (1, 2, 5)).copy(),
        "pred_interface": np.asarray([[[[1, 10], [100, 20]]]], dtype=np.float32),
        "pred_internal_temperature": np.asarray([[[[3], [30]]]], dtype=np.float32),
    }
    native_masks = {
        "fluid_valid": np.asarray([True, False]),
        "interface_valid": np.asarray([[True, True]]),
        "outside_valid": np.asarray([[False, True]]),
        "material_valid": np.asarray([[True, False]]),
    }
    result = _prediction_difference(
        before, after, before_masks=native_masks, after_masks=native_masks,
    )
    fields = result["pred_field_by_component"]
    assert list(fields) == ["u", "v", "p", "omega", "temperature"]
    assert [fields[name]["delta_max_abs"] for name in fields] == [1, 2, 3, 4, 5]
    assert all(fields[name]["finite_values"] == 1 for name in fields)
    assert "descriptive only" in result["pred_field"]["interpretation"]
    interface = result["pred_interface_by_component"]
    assert interface["temperature"]["finite_values"] == 2
    assert interface["temperature"]["delta_max_abs"] == pytest.approx(100.0)
    assert interface["q_proxy"]["finite_values"] == 1
    assert interface["q_proxy"]["delta_max_abs"] == pytest.approx(20.0)
    assert result["pred_internal_temperature_by_component"]["temperature"]["finite_values"] == 1

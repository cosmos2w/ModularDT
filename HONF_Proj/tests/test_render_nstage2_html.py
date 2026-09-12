"""Focused CPU checks for the bounded NStage2 evidence renderer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "tools" / "diagnostics"), str(_ROOT / "src")]

import render_nstage2_html as renderer


def _map_arrays(*, with_mask: bool) -> dict[str, np.ndarray]:
    x, y = np.meshgrid(np.arange(3, dtype=np.float32), np.arange(2, dtype=np.float32))
    gt = np.zeros((2, 3, 5), dtype=np.float32)
    pred = np.ones_like(gt)
    arrays = {"x_grid": x, "y_grid": y, "gt_field_grid": gt, "pred_field_grid": pred}
    if with_mask:
        arrays["fluid_mask"] = np.array([[True, True, False], [True, False, True]])
    return arrays


def test_old_sources_are_model_owned_and_dense_uses_canonical_mask(tmp_path: Path) -> None:
    project = tmp_path / "project"
    study = tmp_path / "nstage2"
    dense_dir = project / "Trained_Results/ThermalChannel/HONF_Forward_Runs/CompareModels/Stage2_Run1401_1804_1801_1802_Epoch500_20Case/debug_npz"
    reader_dir = project / "diagnostics/generated/interface_operator_study/group_reader_recovery/endpoint/debug_npz"
    regional_dir = project / "diagnostics/generated/interface_operator_study/regional_response/endpoint500/debug_npz"
    for directory in (dense_dir, reader_dir, regional_dir):
        directory.mkdir(parents=True)
    (study / "comparison/missing_parent_anchors/debug_npz").mkdir(parents=True)
    (study / "comparison/missing_dense_anchors/debug_npz").mkdir(parents=True)
    np.savez(dense_dir / "Dense_pairwise_adaptation__500__0273.npz", **_map_arrays(with_mask=False))
    np.savez(reader_dir / "Geometry-envelope_sparse_HONF__500__0273.npz", **_map_arrays(with_mask=True))
    np.savez(regional_dir / "Regional_response_HONF__500__0273.npz", **_map_arrays(with_mask=True))
    # A differently named architecture in the Reader directory must not be
    # selected merely because it shares the case token.
    np.savez(reader_dir / "Regional_response_HONF__500__0273.npz", **_map_arrays(with_mask=True))
    reader = renderer._npz_candidates(project, study, "Reader", "0283")
    regional = renderer._npz_candidates(project, study, "Regional", "0283")
    dense = renderer._npz_candidates(project, study, "Dense", "0273")
    assert reader == regional == []
    assert len(dense) == 1
    reader = renderer._npz_candidates(project, study, "Reader", "0273")
    regional = renderer._npz_candidates(project, study, "Regional", "0273")
    assert len(reader) == len(regional) == 1
    assert reader[0].name.startswith("Geometry-envelope_sparse_HONF__500__")
    assert regional[0].name.startswith("Regional_response_HONF__500__")
    assert dense[0].name.startswith("Dense_pairwise_adaptation__500__")
    canonical = renderer._canonical_mask_for_case(project, study, "0273")
    entry, status = renderer._map_entry(dense[0], project, model="Dense", canonical_mask=canonical)
    assert entry is not None
    assert status["mask_source"].startswith("canonical stored fluid_mask from")


def test_timing_keeps_synthetic_shape_and_memory(tmp_path: Path) -> None:
    study = tmp_path / "nstage2"
    (study / "track_a").mkdir(parents=True)
    candidate = {
        "models": [
            {
                "architecture": "hierarchical_regional_honf",
                "checkpoint": {"label": "A1807_at500", "epoch": 500},
                "real_cases": [{"case_id": "0273", "receiver_chunk_size": 2048, "normal": {"phases": {"full_forward": {"median_ms": 2.0, "peak_allocated_bytes": 8}}}}],
                "synthetic_shapes": [{"shape": {"E": 3072, "M": 128, "Q": 262144}, "receiver_chunk_size": 2048, "normal": {"median_ms": 4.0, "peak_allocated_bytes": 16}}],
            }
        ]
    }
    (study / "track_a/timing_chunk2048.json").write_text(json.dumps(candidate), encoding="utf-8")
    rows = renderer._timing_rows(study)
    synthetic = [row for row in rows if row["kind"] == "synthetic"]
    assert synthetic
    assert {row["shape_label"] for row in synthetic} == {"E=3072 M=128 Q=262144"}
    assert all(row["checkpoint_context"] == "candidate endpoint500" for row in synthetic)
    assert all(row["peak_allocated_mib"] is not None for row in synthetic)


def test_parent_chunk128_timing_is_ingested_as_exact500(tmp_path: Path) -> None:
    study = tmp_path / "nstage2"
    comparison = study / "comparison"
    comparison.mkdir(parents=True)
    phases = {
        name: {"median_ms": float(index + 1), "peak_allocated_bytes": 2 * 1024 * 1024}
        for index, name in enumerate(renderer.TIMING_PHASES)
    }
    payload = {
        "models": [
            {
                "architecture": "dense_pairwise_field",
                "checkpoint": {"label": "Dense1804_at500", "epoch": 500},
                "real_anchors": [{"case_id": "0273", "query_count": 8192, "receiver_chunk_size": 128, "normal": {"phases": phases}}],
                "synthetic_shapes": [
                    {
                        "shape": {"E": 768, "M": 32, "Q": 65536},
                        "receiver_chunk_size": 128,
                        "normal": {"median_ms": 7.0, "peak_allocated_bytes": 4 * 1024 * 1024},
                    }
                ],
            }
        ]
    }
    path = comparison / "parent_timing_chunk128.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rows = renderer._timing_rows(study)
    rows = [row for row in rows if row["source"] == str(path)]
    assert rows
    assert {row["checkpoint_context"] for row in rows} == {"parent exact500"}
    assert {row["timing_chunk"] for row in rows} == {128}
    assert {row["query_count"] for row in rows} == {8192, 65536}
    assert {row["kind"] for row in rows} == {"real", "synthetic"}
    assert {row["case_id"] for row in rows if row["kind"] == "real"} == {"0273"}


def test_formal_intervention_reader_ignores_epoch10_and_uses_gt_deltas(tmp_path: Path) -> None:
    study = tmp_path / "nstage2"
    (study / "track_a").mkdir(parents=True)
    (study / "track_a" / "intervention_execution_check_epoch10.json").write_text("{}", encoding="utf-8")
    payload = {
        "checkpoint": {"label": "A1807_at500"},
        "results": [
            {
                "case_id": "0273",
                "query_count": 8192,
                "ground_truth_errors": {"normal": {"metrics": {"global_field_fluid_norm_l2": 1.0}}},
                "interventions": {
                    "p0": {"error_deltas": {"global_field_fluid_norm_l2": 0.25}},
                },
            }
        ],
    }
    (study / "track_a" / "interventions.json").write_text(json.dumps(payload), encoding="utf-8")
    rows = renderer._formal_intervention_rows(study)
    assert rows == [
        {
            "scope": "P0",
            "mode": "p0",
            "model": "A",
            "case_id": "0273",
            "metric": "global_field_fluid_norm_l2",
            "value": 0.25,
            "query_count": 8192,
            "checkpoint": "A1807_at500",
            "source": str(study / "track_a" / "interventions.json"),
        }
    ]


def test_a_record_requires_exact_tree_keys(tmp_path: Path) -> None:
    path = tmp_path / "A__500__0273.npz"
    np.savez(
        path,
        tree_levels=np.array([1, 0]),
        tree_children=np.array([[1, -1], [-1, -1]]),
        tree_coords=np.array([[0.0, 0.0], [1.0, 0.0]]),
        tree_bounds_min=np.zeros((2, 2)),
        tree_bounds_max=np.ones((2, 2)),
        tree_states=np.ones((2, 3)),
        tree_mass=np.array([2.0, 1.0]),
        tree_valid=np.array([True, True]),
        interaction__hierarchical_incidence_query=np.array([0]),
    )
    record = renderer._organization_record(path, _ROOT, "A", "0273")
    assert record["status"] == "available"
    assert "tree_response_norm" in record
    assert record["incidence_keys"] == ["interaction__hierarchical_incidence_query"]
    assert "membership" not in record

    bad = tmp_path / "A_bad.npz"
    np.savez(
        bad,
        tree_levels=np.array([1, 0]),
        tree_children=np.array([[1, -1], [-1, -1]]),
        tree_coords=np.array([[0.0, 0.0], [1.0, 0.0]]),
        tree_bounds_min=np.zeros((2, 2)),
        tree_bounds_max=np.ones((2, 2)),
        tree_states=np.ones((1, 3)),
        tree_mass=np.array([2.0, 1.0]),
        tree_valid=np.array([True, True]),
    )
    bad_record = renderer._organization_record(bad, _ROOT, "A", "0273")
    assert bad_record["status"] == "unavailable"
    assert "misaligned exact arrays" in bad_record["reason"]


def test_b_topology_preserves_eight_shared_coarse_seeds() -> None:
    record = {
        "model": "B",
        "case_id": "0273",
        "status": "available",
        "support_x": [0.0, 1.0],
        "support_y": [0.0, 1.0],
        "topology": {
            "interaction__module_group_indices": [[0, 1], [0, 1]],
            "interaction__environment_group_indices": [[0, 0], [0, 1]],
            "interaction__initial_port_group_read_group_index": [[0, 1]],
        },
    }
    figure = renderer._b_topology_plot([record], "0273")
    assert figure is not None
    assert any("modules → groups" == trace.get("name") for trace in figure["data"])
    assert any("fine environment → coarse source" == trace.get("name") for trace in figure["data"])
    labels = [item.get("text", "") for item in figure["layout"]["annotations"]]
    assert any("8 coarse seeds" in label for label in labels)
    assert not any("coarse seed 0" in label for label in labels)


def test_accuracy_cost_prefers_endpoint500_candidate_and_fixed_large_shape() -> None:
    headline = [
        {"run": "1804", "status": "available", "global_field_fluid_norm_pooled_relative_l2": "0.1"},
        {"run": "1807", "status": "available", "global_field_fluid_norm_pooled_relative_l2": "0.2"},
    ]
    timing = [
        {
            "run": "1804",
            "architecture": "dense_pairwise_field",
            "kind": "synthetic",
            "phase": "full_forward",
            "checkpoint_context": "parent exact500",
            "shape": {"E": 3072, "M": 128, "Q": 262144},
            "timing_chunk": 2048,
            "median_ms": 10.0,
        },
        {
            "run": "1807",
            "architecture": "hierarchical_regional_honf",
            "kind": "synthetic",
            "phase": "full_forward",
            "checkpoint_context": "candidate endpoint500",
            "shape": {"E": 3072, "M": 128, "Q": 262144},
            "timing_chunk": 2048,
            "median_ms": 5.0,
        },
        {
            "run": "1807",
            "architecture": "hierarchical_regional_honf",
            "kind": "synthetic",
            "phase": "full_forward",
            "checkpoint_context": "parent mature checkpoint5000",
            "shape": {"E": 768, "M": 32, "Q": 65536},
            "timing_chunk": 2048,
            "median_ms": 1.0,
        },
        {
            "run": "parent",
            "architecture": "dense_pairwise_field",
            "kind": "synthetic",
            "phase": "full_forward",
            "checkpoint_context": "parent exact500",
            "shape": {"E": 3072, "M": 128, "Q": 262144},
            "timing_chunk": 128,
            "median_ms": 0.1,
        },
    ]
    figure = renderer._accuracy_cost_plot(headline, timing)
    assert figure is not None
    assert "Q=262144" in figure["layout"]["title"]["text"]
    assert {trace["name"] for trace in figure["data"]} == {"Dense", "A"}
    dense = next(trace for trace in figure["data"] if trace["name"] == "Dense")
    assert dense["x"] == [10.0]


def test_timing_plot_deduplicates_parent_exact500_before_mature(tmp_path: Path) -> None:
    rows = [
        {
            "track": "parent",
            "run": "parent",
            "architecture": "dense_pairwise_field",
            "model": "Dense1804_at500",
            "checkpoint_context": "parent exact500",
            "query_count": 8192,
            "timing_chunk": 2048,
            "kind": "real",
            "case_id": "0273",
            "shape": {},
            "shape_label": "",
            "phase": "full_forward",
            "median_ms": 10.0,
            "source": str(tmp_path / "exact.json"),
        },
        {
            "track": "parent",
            "run": "parent",
            "architecture": "dense_pairwise_field",
            "model": "Dense1804_at5000",
            "checkpoint_context": "parent mature checkpoint5000",
            "query_count": 8192,
            "timing_chunk": 2048,
            "kind": "real",
            "case_id": "0273",
            "shape": {},
            "shape_label": "",
            "phase": "full_forward",
            "median_ms": 1.0,
            "source": str(tmp_path / "mature.json"),
        },
    ]
    figure = renderer._timing_plot(rows, 2048, "real:0273", "full_forward")
    assert figure is not None
    assert figure["data"][0]["x"] == ["Dense"]
    assert figure["data"][0]["y"] == [10.0]
    assert figure["data"][0]["customdata"][0][0] == "parent exact500"

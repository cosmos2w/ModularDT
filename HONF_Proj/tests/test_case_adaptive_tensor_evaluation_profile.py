"""Focused profile/evaluation tests for the Phase-2 tensor residual mode."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from channelthermal.evaluation.results import (
    extract_organization_arrays,
    interaction_tensor_diagnostics,
    support_diagnostics,
)
from channelthermal.evaluation_tools.organizer_visualization import (
    render_case_adaptive_residual_summary,
)

from honf_runtime.config_loader import load_config_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_evaluator():
    path = PROJECT_ROOT / "tools" / "diagnostics" / "evaluate_case_adaptive_residual.py"
    spec = importlib.util.spec_from_file_location("evaluate_case_adaptive_residual_phase2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample() -> dict[str, object]:
    x_grid, y_grid = np.meshgrid(
        np.linspace(0.0, 4.0, 6, dtype=np.float32),
        np.linspace(0.0, 2.0, 4, dtype=np.float32),
    )
    return {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "steady_field": np.zeros((*x_grid.shape, 5), dtype=np.float32),
        "structure": {
            "module_centers": np.asarray(
                [[0.7, 0.5], [2.0, 1.0], [3.3, 1.5]], dtype=np.float32
            ),
            "module_present": np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
            "heat_powers": np.asarray([1.0, -0.5, 0.0], dtype=np.float32),
            "domain_length_x": np.asarray(4.0, dtype=np.float32),
            "domain_length_y": np.asarray(2.0, dtype=np.float32),
        },
    }


def _phase2_aux() -> dict[str, np.ndarray]:
    env_coords = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]], dtype=np.float32
    )
    tensor = np.zeros((3, 4, 5), dtype=np.float32)
    tensor[0, :, 0] = np.asarray([1.0, 0.8, 0.5, 0.2], dtype=np.float32)
    tensor[1, :, 1] = np.asarray([0.2, 0.5, 0.8, 1.0], dtype=np.float32)
    return {
        "A_me": np.full((3, 4), 1.0 / 4.0, dtype=np.float32),
        "A_mh": np.asarray(
            [[0.8, 0.2, 0.0], [0.2, 0.8, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
        ),
        "A_eh": np.asarray(
            [[0.8, 0.2, 0.0], [0.6, 0.4, 0.0], [0.3, 0.7, 0.0], [0.1, 0.9, 0.0]],
            dtype=np.float32,
        ),
        "env_coords": env_coords,
        "hyper_strength": np.asarray([0.7, 0.4, 0.2], dtype=np.float32),
        "hyper_module_mass": np.asarray([0.5, 0.5, 0.0], dtype=np.float32),
        "hyper_env_mass": np.asarray([0.5, 0.5, 0.0], dtype=np.float32),
        "hyper_source_coords": np.asarray(
            [[1.0, 0.7], [2.1, 1.0], [0.0, 0.0]], dtype=np.float32
        ),
        "hyper_region_coords": np.asarray(
            [[1.5, 0.8], [2.5, 1.4], [0.0, 0.0]], dtype=np.float32
        ),
        "active_hyperedge_mask": np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        "hard_case_edge_mask": np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        "effective_edge_mask": np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        "edge_survival_soft": np.asarray([1.0, 0.4, 0.1], dtype=np.float32),
        "case_adaptive_edge_count": np.asarray(2.0, dtype=np.float32),
        "case_adaptive_edge_cap": np.asarray(3.0, dtype=np.float32),
        "case_adaptive_cap_hit": np.asarray(0.0, dtype=np.float32),
        "case_adaptive_stop_reached": np.asarray(1.0, dtype=np.float32),
        "residual_stop_fraction": np.asarray(0.01, dtype=np.float32),
        "residual_fraction_trace": np.asarray([1.0, 0.4, 0.01, 0.0], dtype=np.float32),
        "residual_marginal_explained_fraction": np.asarray(
            [0.6, 0.39, 0.01], dtype=np.float32
        ),
        "residual_mechanism_strength": np.asarray([0.7, 0.4, 0.2], dtype=np.float32),
        "residual_module_factor": np.asarray(
            [[0.8, 0.2, 0.0], [0.2, 0.8, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
        ),
        "residual_environment_factor": np.asarray(
            [[0.8, 0.2, 0.0], [0.6, 0.4, 0.0], [0.3, 0.7, 0.0], [0.1, 0.9, 0.0]],
            dtype=np.float32,
        ),
        "residual_content_factor": np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "residual_interaction_tensor": tensor,
        "residual_coupling_row_mass": np.asarray([0.5, 0.5, 0.0], dtype=np.float32),
    }


def test_phase2_profile_is_complete_and_keeps_recommendation() -> None:
    profile_path = PROJECT_ROOT / "src/config_core/forward/case_adaptive_tensor_residual_context.json"
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    core = payload["model"]["core_honf"]
    expected = {
        "organizer_mode": "case_adaptive_tensor_residual",
        "num_hyperedges": 0,
        "edge_capacity": 0,
        "minimum_active_edges": 1,
        "residual_interaction_dim": 32,
        "residual_mechanism_cap_multiplier": 1.5,
        "residual_stop_fraction": 0.01,
        "residual_soft_stop_temperature": 0.002,
        "residual_factor_refinement_steps": 1,
        "residual_coupling_fourier_frequencies": 4,
        "field_assembly_mode": "context_fusion",
        "pairwise_aggregation_mode": "fused_query_module",
        "pairwise_kernel_mode": "legacy_mlp",
        "routing_execution": "dense",
        "query_module_retained_mass_floor": 1.0,
        "hidden_dim": 256,
        "dropout": 0.0,
        "num_env_tokens_x": 24,
        "num_env_tokens_y": 8,
    }
    for key, value in expected.items():
        assert core[key] == value
    assert payload["profile_name"] == "case_adaptive_tensor_residual_context"
    assert payload["training"]["epochs"] == 500
    assert payload["training"]["seed"] == 0
    assert payload["training"]["learning_rate"] == pytest.approx(3.0e-4)
    assert payload["training"]["weight_decay"] == pytest.approx(1.0e-5)
    assert payload["checkpointing"]["save_epoch_milestones"] == [100, 250, 500, 1000, 2500, 5000]
    assert payload["run"]["id"] == "1701"
    assert payload["run"]["name"] == "case_adaptive_tensor_residual_v2"
    case_config = json.loads(
        (PROJECT_ROOT / "Case_ThermalChannel/configs/case_default.json").read_text(encoding="utf-8")
    )
    assert case_config["loss"]["organizer_regularization"]["enabled"] is False

    registry = json.loads(
        (PROJECT_ROOT / "src/config_core/forward/profile_registry.json").read_text(encoding="utf-8")
    )
    entry = next(item for item in registry["profiles"] if item["name"] == payload["profile_name"])
    assert entry["status"] == "candidate"
    assert entry["base"] is None
    assert registry["recommended_forward_profile"] == "stage7_structured_context"
    bundle = load_config_bundle("project://src/config_core/forward/case_adaptive_tensor_residual_context.json")
    assert bundle.effective["model"]["core_honf"]["organizer_mode"] == "case_adaptive_tensor_residual"


def test_phase2_arrays_keep_hard_support_and_tensor_rank_facts() -> None:
    arrays = extract_organization_arrays(_sample(), _phase2_aux())
    np.testing.assert_array_equal(arrays["active_hyperedge_mask"], [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(arrays["hard_case_edge_mask"], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(arrays["edge_survival_soft"], [1.0, 0.4, 0.1])
    assert arrays["residual_interaction_tensor"].shape == (3, 4, 5)

    diagnostics = interaction_tensor_diagnostics(
        arrays["residual_interaction_tensor"], arrays["present"]
    )
    assert diagnostics["interaction_tensor_shape"] == [3, 4, 5]
    assert diagnostics["interaction_tensor_finite"] is True
    assert diagnostics["interaction_tensor_nonnegative"] is True
    assert diagnostics["interaction_tensor_inactive_module_max"] == pytest.approx(0.0)
    assert diagnostics["interaction_tensor_module_effective_rank"] >= 1.0
    assert diagnostics["interaction_tensor_environment_effective_rank"] >= 1.0
    assert diagnostics["interaction_tensor_content_effective_rank"] >= 1.0

    support = support_diagnostics(_phase2_aux())
    assert support["soft_hard_support_gap_mean"] == pytest.approx((0.0 + 0.6 + 0.1) / 3.0)
    assert support["soft_hard_support_gap_max"] == pytest.approx(0.6)
    assert support["hard_forward_support_exact"] is True
    assert support["hard_support_count"] == pytest.approx(2.0)


def test_phase2_summary_reports_k_by_module_and_tensor_availability() -> None:
    evaluator = _load_evaluator()
    rows = [
        {
            "checkpoint": "tensor",
            "module_count": 2,
            "case_adaptive_edge_count": 2.0,
            "case_adaptive_edge_cap": 3.0,
            "case_adaptive_k_over_module_count": 1.0,
            "interaction_tensor_available": True,
            "interaction_tensor_finite": True,
            "interaction_tensor_nonnegative": True,
            "hard_forward_support_exact": True,
        },
        {
            "checkpoint": "tensor",
            "module_count": 3,
            "case_adaptive_edge_count": 3.0,
            "case_adaptive_edge_cap": 5.0,
            "case_adaptive_k_over_module_count": 1.0,
            "interaction_tensor_available": True,
            "interaction_tensor_finite": True,
            "interaction_tensor_nonnegative": True,
            "hard_forward_support_exact": True,
        },
    ]
    summary = evaluator.summarize_rows(rows)
    assert summary["interaction_tensor_available_count"] == 2
    assert summary["interaction_tensor_finite_fraction"] == pytest.approx(1.0)
    assert summary["case_adaptive_k_by_module_count"]["2"]["hard_k_histogram"] == {"2": 1}
    assert summary["case_adaptive_k_by_module_count"]["3"]["cap_mean"] == pytest.approx(5.0)
    assert summary["k_by_module_count"] == summary["case_adaptive_k_by_module_count"]


def test_phase2_tensor_request_is_opt_in_and_visual_is_one_summary(tmp_path: Path) -> None:
    evaluator = _load_evaluator()

    class Facade:
        def forward(self, *, return_organizer_diagnostics: bool = False):
            del return_organizer_diagnostics

    assert evaluator._interaction_tensor_request_kwargs(Facade(), False) == {}
    assert evaluator._interaction_tensor_request_kwargs(Facade(), True) == {
        "return_organizer_diagnostics": True
    }

    arrays = extract_organization_arrays(_sample(), _phase2_aux())
    output = tmp_path / "tensor_residual_summary.png"
    render_case_adaptive_residual_summary(output, arrays)
    assert output.is_file()
    assert output.stat().st_size > 0

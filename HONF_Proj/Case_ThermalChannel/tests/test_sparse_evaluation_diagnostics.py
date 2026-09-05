"""Focused tests for Stage-2 sparse comparison diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from channelthermal.evaluation.prepared import serialize_interaction_aux
from channelthermal.workflows.compare_models import (
    interaction_metrics,
    plot_anchor_physical_predictions,
    plot_sparse_interface_groups,
    reconstruction_metrics,
    save_debug_npz,
)


class _IdentityNormalizer:
    def normalize_fields(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values)

    def normalize_internal_temperature(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values)

    def normalize_interface_targets(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values)


def _physical_case() -> tuple[dict[str, object], dict[str, object]]:
    x_grid, y_grid = np.meshgrid(
        np.asarray([0.0, 1.0], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
    )
    steady_field = np.zeros((2, 2, 5), dtype=np.float32)
    internal_target = np.asarray([[10.0, 12.0], [500.0, 600.0]], dtype=np.float32)
    interface_target = np.zeros((2, 2, 2), dtype=np.float32)
    teacher_ports = np.zeros((2, 2, 5), dtype=np.float32)
    teacher_ports[0, :, 3] = [10.0, 20.0]
    teacher_ports[0, :, 4] = [100.0, 200.0]
    teacher_ports[1, :, 3:] = 900.0
    raw_sample = {
        "case_id": "0273",
        "x_grid": x_grid,
        "y_grid": y_grid,
        "steady_field": steady_field,
        "module_mask": np.zeros((2, 2), dtype=bool),
        "module_internal_temperature_points": internal_target,
        "interface_target": interface_target,
        "teacher_port_tokens": teacher_ports,
        "interface_condition_valid_mask": np.asarray(
            [[1.0, 0.0], [1.0, 1.0]], dtype=np.float32
        ),
        "structure": {
            "module_centers": np.asarray([[0.25, 0.25], [0.75, 0.75]], dtype=np.float32),
            "module_present": np.asarray([1.0, 0.0], dtype=np.float32),
            "material_params": np.asarray([0, 0, 0, 0, 0, 0.1], dtype=np.float32),
        },
    }
    pred_field = steady_field + 1.0
    pred_internal = internal_target[..., None] + 1.0
    pred_internal[1] = 9999.0
    pred_interface = interface_target + np.asarray([2.0, 4.0], dtype=np.float32)
    pred_interface[1] = 9999.0
    final_ports = teacher_ports.copy()
    final_ports[0, :, 3] = [12.0, 18.0]
    final_ports[0, :, 4] = [110.0, 999.0]
    final_ports[1, :, 3:] = 9999.0
    provisional_ports = teacher_ports.copy()
    provisional_ports[0, :, 3] = [11.0, 19.0]
    provisional_ports[0, :, 4] = [105.0, 999.0]
    predictions = {
        "pred_field_grid": pred_field,
        "pred_internal_temperature": pred_internal,
        "pred_interface": pred_interface,
        "pred_port_condition": final_ports,
        "pred_port_condition_raw": provisional_ports,
    }
    return raw_sample, predictions


def test_sparse_interaction_serialization_preserves_flattened_topology() -> None:
    aux = {
        "forward_architecture": "sparse_interface_honf",
        "support_centres": torch.arange(10, dtype=torch.float32).reshape(5, 2),
        "module_group_indices": torch.tensor([[0, 0, 1], [0, 2, 4]]),
        "group_module_degree": torch.arange(5, dtype=torch.float32),
        "module_support_degree": torch.tensor([[2.0, 1.0, 0.0]]),
        "group_count_per_case": torch.tensor([5]),
        "initial_port_group_read_group_index": torch.tensor(
            [[[[0, 1, -1, -1], [2, 3, -1, -1]]]], dtype=torch.long
        ),
    }

    serialized = serialize_interaction_aux(aux)

    assert serialized["support_centres"].shape == (5, 2)
    assert serialized["module_group_indices"].shape == (2, 3)
    assert serialized["group_module_degree"].shape == (5,)
    assert serialized["module_support_degree"].shape == (3,)
    assert np.asarray(serialized["group_count_per_case"]).shape == ()
    assert serialized["initial_port_group_read_group_index"].shape == (1, 2, 4)


def test_sparse_interaction_metrics_report_real_incidence_and_shared_ids() -> None:
    predictions = {
        "interaction_aux": {
            "forward_architecture": "sparse_interface_honf",
            "group_count_per_case": np.asarray(3),
            "support_spacing": 1.5,
            "support_centres": np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            "module_group_indices": np.asarray([[0, 0, 1, 1], [0, 1, 1, 2]]),
            "environment_group_indices": np.asarray([[0, 1, 2], [0, 1, 2]]),
            "module_group_incidence_count_per_case": np.asarray(4),
            "environment_group_incidence_count_per_case": np.asarray(3),
            "group_module_degree": np.asarray([1.0, 2.0, 1.0]),
            "group_environment_degree": np.asarray([1.0, 1.0, 1.0]),
            "module_support_degree": np.asarray([2.0, 2.0]),
            "group_occupancy": np.asarray([1.0, 2.0, 1.0]),
            "initial_port_group_read_group_index": np.asarray(
                [[[0, 1, -1, -1], [2, -1, -1, -1]]]
            ),
            "initial_port_group_read_degree": np.asarray([[2.0, 1.0]]),
            "initial_port_group_read_weight_mass": np.asarray([[1.0, 1.0]]),
            "initial_port_group_read_max_weight": np.asarray([[0.7, 1.0]]),
            "initial_port_main_context_norm": np.asarray([[4.0, 6.0]]),
            "initial_port_main_context_fraction": np.asarray([[0.5, 0.75]]),
        },
        "routing_maps": {
            "group_read_group_index": np.asarray(
                [[0, 1, -1, -1], [1, 2, -1, -1], [2, -1, -1, -1]]
            ),
            "group_read_degree": np.asarray([2.0, 2.0, 1.0]),
            "group_read_weight_mass": np.ones(3),
            "group_read_max_weight": np.asarray([0.6, 0.7, 1.0]),
            "main_context_norm": np.asarray([2.0, 4.0, 6.0]),
            "main_context_fraction": np.asarray([0.2, 0.4, 0.6]),
        },
    }

    row = interaction_metrics({"model_label": "sparse"}, predictions)

    assert row["support_group_count"] == 3.0
    assert row["support_module_group_incidence_count"] == 4.0
    assert row["support_environment_group_incidence_count"] == 3.0
    assert row["support_p2_query_read_incidence_count"] == 5.0
    assert row["support_p0_port_read_incidence_count"] == 3.0
    assert row["support_p0_p2_shared_group_count"] == 3.0
    assert row["support_p0_p2_shared_group_id_namespace_valid"] == 1.0
    assert row["support_group_unique_module_degree_mean"] == pytest.approx(4.0 / 3.0)
    assert row["query_main_context_norm_fraction_mean"] == pytest.approx(0.4)
    assert row["port_main_context_norm_fraction_mean"] == pytest.approx(0.625)

    predictions["routing_maps"]["group_read_group_index"][0, 2] = -2
    invalid = interaction_metrics({"model_label": "sparse"}, predictions)
    assert invalid["support_p0_p2_shared_group_id_namespace_valid"] == 0.0


def test_reconstruction_metrics_include_masked_physical_port_and_interface_errors() -> None:
    raw_sample, predictions = _physical_case()
    dataset = SimpleNamespace(normalizer=_IdentityNormalizer())

    row, _ = reconstruction_metrics(
        base_row={"model_label": "candidate"},
        predictions=predictions,
        raw_sample=raw_sample,
        dataset=dataset,
        checkpoint_targets_normalized=False,
        channel_order=["u", "v", "p", "omega", "temperature"],
    )

    assert row["field_temperature_fluid_physical_mae"] == pytest.approx(1.0)
    assert row["internal_temperature_physical_mae"] == pytest.approx(1.0)
    assert row["interface_t_surface_physical_mae"] == pytest.approx(2.0)
    assert row["interface_q_normal_physical_mae"] == pytest.approx(4.0)
    assert row["port_t_env_final_physical_mae"] == pytest.approx(2.0)
    assert row["port_t_env_provisional_physical_mae"] == pytest.approx(1.0)
    assert row["port_h_effective_final_physical_mae"] == pytest.approx(10.0)
    assert row["port_h_effective_provisional_physical_mae"] == pytest.approx(5.0)
    assert row["port_t_env_final_physical_num_values"] == 2.0
    assert row["port_h_effective_final_physical_num_values"] == 1.0


def test_debug_npz_keeps_sparse_route_ids_and_port_evidence(tmp_path) -> None:
    raw_sample, predictions = _physical_case()
    predictions.update(
        {
            "routing_maps": {
                "group_read_group_index": np.asarray([[0, 1, -1, -1]], dtype=np.int64),
                "group_read_normalized_weight": np.asarray(
                    [[0.6, 0.4, 0.0, 0.0]], dtype=np.float32
                ),
            },
            "interaction_aux": {
                "support_centres": np.asarray([[0.0, 0.0], [1.0, 0.0]]),
                "module_group_indices": np.asarray([[0, 0], [0, 1]], dtype=np.int64),
            },
        }
    )
    path = tmp_path / "case.npz"

    save_debug_npz(path, predictions, raw_sample)

    with np.load(path) as payload:
        assert payload["group_read_group_index"].dtype == np.int64
        assert payload["interaction__module_group_indices"].shape == (2, 2)
        assert payload["pred_port_condition"].shape == (2, 2, 5)
        assert payload["gt_port_condition"].shape == (2, 2, 5)


def test_anchor_plots_render_physical_evidence_and_shared_sparse_routes(tmp_path) -> None:
    raw_sample, predictions = _physical_case()
    dataset = SimpleNamespace(normalizer=_IdentityNormalizer())
    predictions.update(
        {
            "interaction_aux": {
                "support_centres": np.asarray([[0.0, 0.0], [1.0, 0.0]]),
                "support_spacing": 0.5,
                "module_group_indices": np.asarray([[0, 0], [0, 1]], dtype=np.int64),
                "module_geometric_membership": np.asarray([0.6, 0.4]),
                "module_learned_membership": np.asarray([0.7, 0.3]),
                "group_module_degree": np.asarray([1.0, 1.0]),
                "initial_port_group_read_group_index": np.asarray(
                    [
                        [[0, 1, -1, -1], [0, -1, -1, -1]],
                        [[-1, -1, -1, -1], [-1, -1, -1, -1]],
                    ]
                ),
                "initial_port_group_read_normalized_weight": np.asarray(
                    [
                        [[0.6, 0.4, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    ]
                ),
            },
            "routing_maps": {
                "group_read_group_index": np.asarray(
                    [
                        [0, 1, -1, -1],
                        [0, -1, -1, -1],
                        [1, -1, -1, -1],
                        [0, 1, -1, -1],
                    ]
                ),
                "group_read_normalized_weight": np.asarray(
                    [
                        [0.6, 0.4, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.4, 0.6, 0.0, 0.0],
                    ]
                ),
                "group_read_degree": np.asarray([2.0, 1.0, 1.0, 2.0]),
            },
        }
    )
    physical_path = tmp_path / "physical.png"
    sparse_path = tmp_path / "sparse.png"

    plot_anchor_physical_predictions(
        physical_path,
        predictions,
        raw_sample,
        dataset,
        checkpoint_targets_normalized=False,
        title="physical anchor",
    )
    plot_sparse_interface_groups(
        sparse_path, predictions, raw_sample, title="sparse anchor"
    )

    assert physical_path.stat().st_size > 0
    assert sparse_path.stat().st_size > 0

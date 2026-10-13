"""Checkpoint-native global input transforms match the public HDF5 reader."""

import json

import h5py
import numpy as np
import pytest
from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
from channelthermal.evaluation.loading import checkpoint_native_inputs_from_physical_sample


def _write_minimal_case(path):
    case_id = "case001"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("case_ids", data=np.asarray([case_id], dtype="S"))
        h5.create_dataset("splits", data=np.asarray(["test"], dtype="S"))
        h5.attrs.update(field_dim=5, n_interface_points=3, max_modules=2, local_grid_size=2)
        h5.create_dataset("channel_order", data=np.asarray(["u", "v", "p", "omega", "temperature"], dtype="S"))
        h5.create_dataset(
            "interface_condition_feature_names",
            data=np.asarray(
                ["theta", "normal_x", "normal_y", "T_outside", "u_normal", "u_tangent", "h_proxy", "h_effective"],
                dtype="S",
            ),
        )
        case = h5.create_group("cases").create_group(case_id)
        case.attrs.update(split="test", converged=True)
        case.create_dataset("sampled_points", data=np.arange(4 * 7, dtype=np.float32).reshape(4, 7))
        case.create_dataset("module_centers", data=np.asarray([[2.0, 3.0], [8.0, 4.0]], dtype=np.float32))
        case.create_dataset("module_present", data=np.asarray([1.0, 0.0], dtype=np.float32))
        case.create_dataset("heat_powers", data=np.asarray([3.0, 0.0], dtype=np.float32))
        condition = np.zeros((2, 3, 8), dtype=np.float32)
        condition[0] = np.asarray(
            [[0.0, 1.0, 0.0, 10.0, 0.1, -0.2, 15.0, 14.0],
             [1.0, 0.0, 1.0, 12.0, 0.2, 0.3, 16.0, 15.0],
             [2.0, -1.0, 0.0, 14.0, -0.3, 0.4, 17.0, 16.0]],
            dtype=np.float32,
        )
        case.create_dataset("interface_condition", data=condition)
        case.create_dataset("interface_condition_valid_mask", data=np.ones((2, 3), dtype=np.float32))
        case.create_dataset("interface_target", data=np.ones((2, 3, 2), dtype=np.float32) * 5.0)
        case.create_dataset("module_internal_temperature", data=np.ones((2, 2, 2), dtype=np.float32) * 8.0)
        case.create_dataset("module_internal_mask", data=np.ones((2, 2), dtype=np.float32))
        material = case.create_group("material_parameters")
        material.attrs.update(nu=0.006, solid_alpha=0.01, fluid_alpha=0.02, solid_k=1.0, fluid_k=1.0,
                              module_radius=0.45, re=100.0, u_in=1.0)
        case.create_dataset("case_config_json", data=np.bytes_(json.dumps({"domain": {"lx": 12.0, "ly": 6.0}})))
        case.create_dataset("x_grid", data=np.asarray([[3.0, 9.0], [3.0, 9.0]], dtype=np.float32))
        case.create_dataset("y_grid", data=np.asarray([[1.5, 1.5], [4.5, 4.5]], dtype=np.float32))
        case.create_dataset("steady_field", data=np.ones((2, 2, 5), dtype=np.float32) * 7.0)
        case.create_dataset("rms_field", data=np.ones((2, 2, 5), dtype=np.float32))
        case.create_dataset("module_mask", data=np.zeros((2, 2), dtype=np.uint8))
    return case_id


def _checkpoint(stats, *, normalize_inputs=True):
    return {
        "train_config": {"dataset": {"normalize_inputs": normalize_inputs, "normalize_targets": True}},
        "global_normalization_stats": stats,
    }


def test_checkpoint_native_input_transform_matches_public_dataset_reader(tmp_path):
    path = tmp_path / "minimal_thermal.h5"
    case_id = _write_minimal_case(path)
    stats = {
        "heat_power_mean": np.asarray([1.0], dtype=np.float32),
        "heat_power_std": np.asarray([2.0], dtype=np.float32),
        "interface_condition_mean": np.arange(8, dtype=np.float32),
        "interface_condition_std": np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32),
        "field_mean_by_channel": np.zeros(5, dtype=np.float32),
        "field_std_by_channel": np.ones(5, dtype=np.float32),
        "interface_target_mean": np.zeros(2, dtype=np.float32),
        "interface_target_std": np.ones(2, dtype=np.float32),
        "internal_temperature_mean": np.zeros(1, dtype=np.float32),
        "internal_temperature_std": np.ones(1, dtype=np.float32),
    }
    checkpoint = _checkpoint(stats)
    normalizer = H5Normalizer(stats)
    raw_reader = GlobalChannelThermalDataset(
        path, split="test", points_per_case=1, normalize_inputs=False, normalize_targets=False,
        random_point_sampling=False, include_grid=True, normalizer=normalizer, case_ids=[case_id],
    )
    normalized_reader = GlobalChannelThermalDataset(
        path, split="test", points_per_case=1, normalize_inputs=True, normalize_targets=True,
        random_point_sampling=False, include_grid=True, normalizer=normalizer, case_ids=[case_id],
    )
    try:
        physical = raw_reader[0]
        physical_heat = physical["structure"]["heat_powers"].copy()
        physical_condition = physical["interface_condition"].copy()
        transformed = checkpoint_native_inputs_from_physical_sample(physical, checkpoint)
        public = normalized_reader[0]

        np.testing.assert_array_equal(transformed["structure"]["heat_powers"], public["structure"]["heat_powers"])
        np.testing.assert_array_equal(transformed["interface_condition"], public["interface_condition"])
        np.testing.assert_array_equal(transformed["local_module_params"], public["local_module_params"])
        np.testing.assert_array_equal(transformed["teacher_port_tokens"], public["teacher_port_tokens"])
        np.testing.assert_array_equal(transformed["steady_field"], physical["steady_field"])
        np.testing.assert_array_equal(transformed["interface_target"], physical["interface_target"])
        np.testing.assert_array_equal(transformed["module_internal_temperature_points"],
                                      physical["module_internal_temperature_points"])
        np.testing.assert_array_equal(physical["structure"]["heat_powers"], physical_heat)
        np.testing.assert_array_equal(physical["interface_condition"], physical_condition)
        assert transformed is not physical
        assert transformed["structure"] is not physical["structure"]
    finally:
        raw_reader.close()
        normalized_reader.close()


def test_checkpoint_native_input_transform_preserves_disabled_legacy_path():
    sample = {"structure": {"heat_powers": np.asarray([2.0], dtype=np.float32)},
              "interface_condition": np.ones((1, 2, 8), dtype=np.float32)}

    result = checkpoint_native_inputs_from_physical_sample(sample, _checkpoint({}, normalize_inputs=False))

    np.testing.assert_array_equal(result["structure"]["heat_powers"], sample["structure"]["heat_powers"])
    np.testing.assert_array_equal(result["interface_condition"], sample["interface_condition"])
    assert result is not sample


def test_checkpoint_native_input_transform_rejects_missing_required_stats():
    sample = {"structure": {"heat_powers": np.asarray([2.0], dtype=np.float32)},
              "interface_condition": np.ones((1, 2, 8), dtype=np.float32)}

    with pytest.raises(ValueError, match="missing required saved statistics"):
        checkpoint_native_inputs_from_physical_sample(sample, _checkpoint({}))

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from honf_runtime.unified_training import SamplingKey, TaskBatch

from windfarm.normalization import VelocityNormalizer, VerticalProfileBaseline
from windfarm.shared_interaction import WindFarmRefinedInteractionModel
from windfarm.splits import make_group_split
from windfarm.training.unified_task import (
    DEFAULT_ROLE_QUERY_COUNTS,
    WindReceiverInputs,
    WindRefinementTask,
    WindTargets,
    _as_scene_batch,
    _query_sampling_sha256,
    _read_fixed_rows,
    _role_ids,
    _scene_inputs,
    _stable_json_sha256,
)


def _normalizer() -> VelocityNormalizer:
    return VelocityNormalizer(
        mean=np.asarray([1.0, 0.0, 0.0]),
        std=np.asarray([0.2, 0.1, 0.1]),
        safe_std=np.asarray([0.2, 0.1, 0.1]),
        source_rows=72,
    )


def _profile() -> VerticalProfileBaseline:
    return VerticalProfileBaseline(
        bin_centers_D=np.asarray([0.0, 2.0], dtype=np.float64),
        values_mps=np.asarray([[4.0, 0.1, -0.2], [12.0, 0.5, 0.2]], dtype=np.float64),
        counts=np.asarray([100, 100], dtype=np.int64),
        z_min_D=0.0,
        z_max_D=2.0,
    )


def _case(row: int = 0, layout: int = 0, direction: float = 270.0):
    from windfarm.geometry import environment_representation, global_geometry_features, support_geometry

    support = support_geometry(
        np.asarray([-400.0, 0.0, 800.0], dtype=np.float32),
        np.asarray([-320.0, 0.0, 320.0], dtype=np.float32),
        np.asarray([20.0, 70.0, 130.0], dtype=np.float32),
    )
    environment = environment_representation(support, token_shape=(2, 2, 2))
    centers = np.asarray([[0.0, 0.0, 0.875], [3.0, 0.5, 0.875]], dtype=np.float32)
    return SimpleNamespace(
        index=row,
        case=f"gen_{row:04d}_wd{int(direction)}",
        layout=f"layout-{layout}",
        layout_index=layout,
        wind_direction_deg=direction,
        n_turbines=2,
        support=support,
        module_centers=centers,
        module_present=np.ones(2, dtype=np.float32),
        module_features=np.asarray([[0.5, 0.875], [0.5, 0.875]], dtype=np.float32),
        global_context=global_geometry_features(support, direction, 2),
        env_coords=environment.coords_D,
        env_features=environment.features,
        env_weights=environment.weights_D3,
    )


class _View:
    def __init__(self):
        self.metadata = {
            "layout_index": np.repeat(np.arange(14, dtype=np.int64), 3),
            "wd_deg": np.tile(np.asarray([270.0, 285.0, 300.0]), 14),
        }

    def run(self, row: int):
        return _case(row, int(self.metadata["layout_index"][row]), float(self.metadata["wd_deg"][row]))


def _provider() -> WindRefinementTask:
    view = _View()
    train_rows = np.arange(0, 18, dtype=np.int64)
    validation_rows = np.arange(18, 42, dtype=np.int64)
    manifest = {
        "subset_id": "wind_shared_fixed24_v1",
        "manifest_sha256": "test-manifest-sha",
        "train_layout_indices": list(range(6)),
        "validation_layout_indices": list(range(6, 14)),
        "train_layout_selection": {
            "selected_turbine_counts": [6, 7, 8, 9],
            "uncovered_turbine_counts": [18],
        },
    }
    return WindRefinementTask(
        view,  # type: ignore[arg-type]
        train_rows=train_rows,
        validation_rows=validation_rows,
        normalizer=_normalizer(),
        background_profile=_profile(),
        manifest=manifest,
        manifest_path=Path("/tmp/fixed_subset_manifest.json"),
        normalization_path=Path("/tmp/train_only_normalization.json"),
        role_scales={name: 1.0 for name in DEFAULT_ROLE_QUERY_COUNTS},
        role_scale_sha256="test-role-scale-sha",
    )


def _fixed_manifest(groups: np.ndarray, directions: np.ndarray) -> tuple[dict, object]:
    split = make_group_split(groups, seed=42)
    train_layouts = np.sort(np.unique(groups[split.train]))[:24]
    validation_layouts = np.sort(np.unique(groups[split.validation]))[:8]
    train_rows = np.flatnonzero(np.isin(groups, train_layouts)).astype(np.int64)
    validation_rows = np.flatnonzero(np.isin(groups, validation_layouts)).astype(np.int64)

    def selection(rows: np.ndarray, layouts: np.ndarray) -> dict:
        return {
            "used_target_values_for_selection": False,
            "row_indices": rows.tolist(),
            "layout_indices": layouts.tolist(),
        }

    payload = {
        "subset_id": "wind_shared_fixed24_v1",
        "used_targets_for_subset_selection": False,
        "source_split": {
            "seed": 42,
            "group_key": "layout_index",
            "test_target_values_read": False,
            "train_row_indices_sha256": split.metadata["partitions"]["train"]["row_indices_sha256"],
            "validation_row_indices_sha256": split.metadata["partitions"]["validation"]["row_indices_sha256"],
            "original_train_layout_count": split.metadata["partitions"]["train"]["groups"],
            "original_validation_layout_count": split.metadata["partitions"]["validation"]["groups"],
            "original_test_layout_count": split.metadata["partitions"]["test"]["groups"],
        },
        "train_row_indices": train_rows.tolist(),
        "validation_row_indices": validation_rows.tolist(),
        "train_row_count": int(train_rows.size),
        "validation_row_count": int(validation_rows.size),
        "direction_categories": [270.0, 285.0, 300.0],
        "train_layout_indices": train_layouts.tolist(),
        "validation_layout_indices": validation_layouts.tolist(),
        "train_layout_selection": selection(train_rows, train_layouts),
        "validation_layout_selection": selection(validation_rows, validation_layouts),
    }
    payload["manifest_sha256"] = _stable_json_sha256(payload)
    return payload, split


def test_fixed_wind_manifest_binds_source_splits_and_all_direction_rows(tmp_path: Path):
    groups = np.repeat(np.arange(60, dtype=np.int64), 3)
    directions = np.tile(np.asarray([270.0, 285.0, 300.0]), 60)
    manifest, split = _fixed_manifest(groups, directions)
    view = SimpleNamespace(metadata={"layout_index": groups, "wd_deg": directions})
    path = tmp_path / "fixed_subset_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    _, train_rows, validation_rows = _read_fixed_rows(view, split, path)  # type: ignore[arg-type]
    assert train_rows.shape == (72,)
    assert validation_rows.shape == (24,)

    corrupted = dict(manifest)
    corrupted["train_layout_selection"] = dict(manifest["train_layout_selection"])
    corrupted["train_layout_selection"]["row_indices"] = list(reversed(train_rows.tolist()))
    corrupted.pop("manifest_sha256")
    corrupted["manifest_sha256"] = _stable_json_sha256(corrupted)
    path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="nested layout/row inventory"):
        _read_fixed_rows(view, split, path)  # type: ignore[arg-type]


def test_query_receipt_binds_native_cells_but_not_matched_arm():
    sample = SimpleNamespace(
        coordinates_D=np.zeros((1024, 3), dtype=np.float32),
        flat_indices=np.arange(1024, dtype=np.int64),
        role_sample_counts=DEFAULT_ROLE_QUERY_COUNTS,
    )
    full_key = SamplingKey(42, 10, 2, 3, "warmup", "full_detail")
    adaptive_key = SamplingKey(42, 10, 2, 3, "warmup", "adaptive_detail")
    receipt = _query_sampling_sha256((6,), (sample,), full_key)
    assert len(receipt) == 64
    assert _query_sampling_sha256((6,), (sample,), adaptive_key) == receipt
    changed = SimpleNamespace(
        coordinates_D=np.ones((1024, 3), dtype=np.float32),
        flat_indices=sample.flat_indices,
        role_sample_counts=DEFAULT_ROLE_QUERY_COUNTS,
    )
    assert _query_sampling_sha256((6,), (changed,), full_key) != receipt


def test_wind_predictor_does_not_claim_an_unpaid_adaptive_full_replay():
    torch.manual_seed(8)
    provider = _provider()
    case = _case()
    scene = _as_scene_batch((_scene_inputs(case),), torch.device("cpu"))
    receivers = WindReceiverInputs(
        coordinates_D=np.asarray([[[0.5, 0.2, 0.875], [4.0, 1.0, 1.1]]], dtype=np.float32),
        role_ids=np.asarray([[0, 1]], dtype=np.int8),
    )
    model = WindFarmRefinedInteractionModel(
        velocity_transform=_normalizer(), background_profile=_profile(), hidden=16, message=12
    )
    model.eval()

    adaptive, adaptive_state = provider.predict_native(
        model, scene, receivers, "adaptive_detail", "soft", epoch=700, temperature=0.55
    )
    assert adaptive.full_mps is None
    assert adaptive_state["phase"] == "hard"
    assert adaptive_state["temperature"] == pytest.approx(0.55)

    full, _ = provider.predict_native(model, scene, receivers, "full_detail", "soft", epoch=700)
    assert full.full_mps is full.main_mps


def test_wind_stage_hook_accepts_and_checks_shared_engine_temperature():
    provider = _provider()
    receipt = provider.on_phase_start(
        object(), phase="warmup", epoch=1, arm="warmup", temperature=1.0
    )
    assert receipt["temperature"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="shared engine schedule"):
        provider.on_phase_start(
            object(), phase="soft", epoch=601, arm="adaptive_detail", temperature=0.5
        )


def test_wind_cost_calibration_is_train_only_bounded_and_checkpointable():
    torch.manual_seed(11)
    provider = _provider()
    case = _case()
    scene_inputs = _scene_inputs(case)
    coordinates = np.resize(
        np.asarray([[0.5, 0.2, 0.875], [4.0, 1.0, 1.1], [2.0, -1.0, 2.5]], dtype=np.float32),
        (1024, 3),
    )[None]
    roles = _role_ids(DEFAULT_ROLE_QUERY_COUNTS)[None]
    targets = WindTargets(
        velocity_mps=np.zeros((1, 1024, 3), dtype=np.float32),
        row_indices=(0,),
        case_metadata=({"row_index": 0, "layout_index": 0, "wind_direction_deg": 270.0},),
    )
    batch = TaskBatch(
        scene_inputs=(scene_inputs,),
        receivers=WindReceiverInputs(coordinates, roles),
        targets=targets,
        case_keys=(0,),
    )
    provider.message_calibration_rows = (0,)
    provider._message_scale = 0.2
    provider._expected_work_calibration_batches = lambda: (batch,)  # type: ignore[method-assign]
    model = WindFarmRefinedInteractionModel(
        velocity_transform=_normalizer(), background_profile=_profile(), hidden=16, message=12
    )
    model.core.refinement.residual_scale.fill_(0.2)

    receipt = provider.calibrate_expected_work_weight(model)
    assert receipt["row_indices"] == [0]
    assert receipt["validation_values_read"] is False
    assert 0.0 <= receipt["coefficient"] <= 0.1
    assert math_isfinite(receipt["router_native_gradient_norm"])
    state = provider.training_state_dict()
    restored = _provider()
    restored.load_training_state_dict(state)
    assert restored.training_state_dict() == state


def test_wind_checkpoint_restore_can_replace_only_device_local_prefit_rms():
    saved = _provider()
    saved._message_scale = 0.2
    saved._message_scale_calibration = {
        "method": "pre-fit fine source-message RMS",
        "layout_indices": [0, 1, 2, 3],
        "row_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "query_count_per_row": 1024,
        "query_coordinates_sha256": "a" * 64,
        "active_message_component_count": 24,
        "message_rms_latent": 0.2,
        "minimum_scale": 1.0e-6,
        "target_values_read": False,
    }
    checkpoint_state = saved.training_state_dict()

    cpu_provider = _provider()
    cpu_provider._message_scale = 0.20000000025
    cpu_provider._message_scale_calibration = {
        **saved._message_scale_calibration,
        "message_rms_latent": cpu_provider._message_scale,
    }
    with pytest.raises(ValueError, match="active fixed TRAIN calibration"):
        cpu_provider.load_training_state_dict(checkpoint_state)

    cpu_provider.load_training_state_dict(
        checkpoint_state, allow_prefit_calibration_replace=True
    )
    assert cpu_provider._message_scale == checkpoint_state["message_scale"]
    assert cpu_provider._message_scale_calibration == checkpoint_state["message_scale_calibration"]


def test_wind_prefit_calibration_replace_rejects_different_query_panel():
    saved = _provider()
    saved._message_scale = 0.2
    saved._message_scale_calibration = {
        "method": "pre-fit fine source-message RMS",
        "layout_indices": [0, 1, 2, 3],
        "row_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "query_count_per_row": 1024,
        "query_coordinates_sha256": "a" * 64,
        "active_message_component_count": 24,
        "message_rms_latent": 0.2,
        "minimum_scale": 1.0e-6,
        "target_values_read": False,
    }
    checkpoint_state = saved.training_state_dict()
    changed_provider = _provider()
    changed_provider._message_scale = 0.21
    changed_provider._message_scale_calibration = {
        **saved._message_scale_calibration,
        "query_coordinates_sha256": "b" * 64,
        "message_rms_latent": 0.21,
    }
    with pytest.raises(ValueError, match="same pre-fit TRAIN calibration"):
        changed_provider.load_training_state_dict(
            checkpoint_state, allow_prefit_calibration_replace=True
        )


def math_isfinite(value: float) -> bool:
    return bool(np.isfinite(float(value)))

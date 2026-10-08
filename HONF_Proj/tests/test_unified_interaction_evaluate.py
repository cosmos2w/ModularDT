from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from honf_runtime.unified_training import EngineConfig, SelectionPolicy, _engine_config_payload

TOOL_PATH = Path(__file__).resolve().parents[1] / "tools/unified_interaction_evaluate.py"
SPEC = importlib.util.spec_from_file_location("unified_interaction_evaluate", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def test_checkpoint_age_comes_from_payload_and_literal_labels_are_enforced():
    config = SimpleNamespace(seed=0, total_epochs=2500)
    selection = SimpleNamespace(field_metric="field_score")
    provider_identity = {"manifest": "fixed", "normalization": "train-only", "device": "cpu"}
    saved_provider_identity = {**provider_identity, "device": "cuda:0"}
    identity = {
        "workflow": "unified_interaction_refinement",
        "task": "ThermalChannel",
        "development_profile": "fixed25_v1",
        "seed": 0,
        "initial_model_state_sha256": "initial-state",
        "engine_profile": "warmup500_open600_soft800_total2500",
        "provider_identity": saved_provider_identity,
        "engine_config": config.__dict__,
        "selection_policy": selection.__dict__,
        "optimizer_schedule_contract": [],
        "run_id": "run-a",
    }
    payload = {
        "workflow": "unified_interaction_refinement",
        "checkpoint_schema_version": 1,
        "epoch": 1200,
        "current_epoch": 1200,
        "experiment_identity": identity,
        "arm": "adaptive_detail",
    }
    result = evaluator.validate_checkpoint_binding(
        payload,
        task="thermal",
        provider_identity=provider_identity,
        initial_state_sha256="initial-state",
        engine_config=config,
        selection=selection,
        label="selected",
    )
    assert result["epoch"] == 1200
    assert result["requested_age"] is None
    assert result["saved_training_device"] == "cuda:0"
    assert result["evaluation_provider_device"] == "cpu"
    with pytest.raises(ValueError, match="requires literal payload age 1000"):
        evaluator.validate_checkpoint_binding(
            payload,
            task="thermal",
            provider_identity=provider_identity,
            initial_state_sha256="initial-state",
            engine_config=config,
            selection=selection,
            label="literal1000",
        )
    assert evaluator._expected_literal_age("adaptive_literal1000") == 1000
    assert evaluator._expected_literal_age("full-epoch2500") == 2500


def test_checkpoint_identity_rejects_other_normalizer_and_task_profile():
    config = SimpleNamespace(seed=42, total_epochs=2500)
    selection = SimpleNamespace(field_metric="field_score")
    identity = {
        "workflow": "unified_interaction_refinement",
        "task": "WindFarm",
        "development_profile": "fixed24_v1",
        "seed": 42,
        "initial_model_state_sha256": "initial-state",
        "engine_profile": "warmup500_open600_soft800_total2500",
        "provider_identity": {"normalization": "other"},
        "engine_config": config.__dict__,
        "selection_policy": selection.__dict__,
        "optimizer_schedule_contract": [],
    }
    payload = {
        "workflow": "unified_interaction_refinement",
        "checkpoint_schema_version": 1,
        "epoch": 1000,
        "current_epoch": 1000,
        "experiment_identity": identity,
        "arm": "full_detail",
    }
    with pytest.raises(ValueError, match="provider identity differs"):
        evaluator.validate_checkpoint_binding(
            payload,
            task="wind",
            provider_identity={"normalization": "fixed-train-only"},
            initial_state_sha256="initial-state",
            engine_config=config,
            selection=selection,
            label="literal1000",
        )


def test_evaluator_normalizes_legacy_checkpoint_sampler_default():
    config = EngineConfig(seed=42, microbatch_cases=4, effective_cases=24, total_epochs=2500,
                          warmup_epochs=500, open_through_epoch=600, soft_through_epoch=800)
    selection = SelectionPolicy(field_metric="field_score", response_guard_metric=None)
    saved_engine = _engine_config_payload(config)
    assert "sampling_version" not in saved_engine  # Prior sealed checkpoints predate the explicit sampler field.
    provider_identity = {"dataset": "WindFarm", "subset_manifest_sha256": "fixed24", "device": "cpu"}
    identity = {
        "workflow": "unified_interaction_refinement",
        "task": "WindFarm",
        "development_profile": "fixed24_v1",
        "seed": 42,
        "initial_model_state_sha256": "initial-state",
        "engine_profile": "warmup500_open600_soft800_total2500",
        "provider_identity": provider_identity,
        "engine_config": saved_engine,
        "selection_policy": selection.__dict__,
        "optimizer_schedule_contract": [],
    }
    payload = {
        "workflow": "unified_interaction_refinement",
        "checkpoint_schema_version": 1,
        "epoch": 1000,
        "current_epoch": 1000,
        "experiment_identity": identity,
        "arm": "full_detail",
    }
    result = evaluator.validate_checkpoint_binding(
        payload,
        task="wind",
        provider_identity=provider_identity,
        initial_state_sha256="initial-state",
        engine_config=config,
        selection=selection,
        label="literal1000",
    )
    assert result["epoch"] == 1000


def test_representative_export_scope_binds_controls_and_field_only_to_their_arms():
    evaluator._validate_export_scope("adaptive_detail", detailed=True, field_only=False)
    evaluator._validate_export_scope("full_detail", detailed=False, field_only=True)
    with pytest.raises(ValueError, match="selected adaptive-detail"):
        evaluator._validate_export_scope("full_detail", detailed=True, field_only=False)
    with pytest.raises(ValueError, match="Full-detail checkpoint"):
        evaluator._validate_export_scope("adaptive_detail", detailed=False, field_only=True)
    with pytest.raises(ValueError, match="cannot request both"):
        evaluator._validate_export_scope("adaptive_detail", detailed=True, field_only=True)


def test_cli_accepts_separate_field_and_detailed_checkpoint_labels():
    parser = evaluator.build_parser()
    args = parser.parse_args([
        "--task", "thermal",
        "--checkpoint", "selected_adaptive=/tmp/adaptive.pt",
        "--checkpoint", "selected_full=/tmp/full.pt",
        "--detailed-label", "selected_adaptive",
        "--field-label", "selected_full",
    ])
    assert args.detailed_label == ["selected_adaptive"]
    assert args.field_label == ["selected_full"]
    assert args.output_dir is None


def test_default_evaluator_index_belongs_to_source_run(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / "Run_fixture/checkpoints/latest_model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(evaluator, "_load_task", lambda *_: (None, None, None, None, "fixture"))
    monkeypatch.setattr(evaluator, "_load_checkpoint_payload", lambda _path: {"experiment_identity": {}})

    def completed_evaluation(task, label, path, **kwargs):
        assert kwargs["output_root"] is None
        return {"checkpoint": {"label": label, "epoch": 100, "arm": "adaptive_detail"},
                "summary_path": str(tmp_path / "Run_fixture/evaluations/unified" / label / "evaluation_summary.json")}

    monkeypatch.setattr(evaluator, "evaluate_checkpoint", completed_evaluation)
    assert evaluator.main(["--task", "wind", "--checkpoint", f"selected={checkpoint}"]) == 0
    capsys.readouterr()
    index = tmp_path / "Run_fixture/comparisons/unified_evaluation_index.json"
    assert json.loads(index.read_text())["checkpoint_results"][0]["epoch"] == 100


def test_six_observed_three_held_interface_is_geometry_only_and_disjoint():
    x, y = np.meshgrid(np.linspace(-2, 2, 17), np.linspace(-1, 1, 11))
    coordinates = np.column_stack((x.reshape(-1), y.reshape(-1)))
    prediction = np.column_stack((coordinates[:, 0] + coordinates[:, 1], coordinates[:, 0] - coordinates[:, 1]))
    reference = prediction + 0.25
    first = evaluator.split_native_receiver_interface(coordinates, prediction, reference)
    changed_targets = evaluator.split_native_receiver_interface(coordinates, prediction, reference * 17)
    assert first["observed_indices"].shape == (6,)
    assert first["held_indices"].shape == (3,)
    assert set(first["observed_indices"]).isdisjoint(set(first["held_indices"]))
    np.testing.assert_array_equal(first["observed_indices"], changed_targets["observed_indices"])
    np.testing.assert_array_equal(first["held_indices"], changed_targets["held_indices"])
    np.testing.assert_allclose(first["held_prediction"], prediction[first["held_indices"]])
    np.testing.assert_allclose(first["held_reference"], reference[first["held_indices"]])


def test_receiver_selection_and_tail_summary_reject_invalid_or_empty_inputs():
    with pytest.raises(ValueError, match="at least the requested count"):
        evaluator.farthest_receiver_indices(np.zeros((2, 3)), 3)
    assert evaluator._tail_summary(np.asarray([np.nan])) == {"count": 0, "available": False}
    summary = evaluator._tail_summary(np.asarray([-1.0, 2.0, 3.0]))
    assert summary["count"] == 3
    assert summary["max_abs"] == 3.0
    assert summary["rmse"] == pytest.approx((14.0 / 3.0) ** 0.5)


def test_json_default_serializes_numpy_vectors_and_scalars():
    assert evaluator._json_default(np.asarray([70.0, 80.0])) == [70.0, 80.0]
    assert evaluator._json_default(np.float64(2.5)) == 2.5


def test_thermal_native_region_masks_use_geometry_inlet_sign_and_declared_radius():
    query = np.asarray([[3.5, 0.0], [3.0, 0.0], [3.5, 1.1], [0.5, 0.0]])
    centers = np.asarray([[2.0, 0.0]])
    present = np.asarray([True])
    valid = np.asarray([True, True, True, True])
    positive = evaluator._thermal_geometry_region_masks(query, centers, present, 0.5, 1.0, valid)
    np.testing.assert_array_equal(positive["near"], [False, True, False, False])
    np.testing.assert_array_equal(positive["downstream"], [True, False, False, False])
    negative = evaluator._thermal_geometry_region_masks(query, centers, present, 0.5, -1.0, valid)
    assert negative["downstream"][3]
    no_flow = evaluator._thermal_geometry_region_masks(query, centers, present, 0.5, 0.0, valid)
    assert not no_flow["downstream"].any()


def test_thermal_all_base_retains_fine_far_path_on_protected_pairs():
    import torch

    base = torch.tensor([[[[2.0], [3.0], [4.0]]]])
    fine = torch.tensor([[[[12.0], [13.0], [14.0]]]])
    protected = torch.tensor([[[True, False, True]]])
    effective = evaluator._thermal_effective_all_base_far(base, fine, protected)
    torch.testing.assert_close(effective, torch.tensor([[[[12.0], [3.0], [14.0]]]]))
    action = torch.tensor([0.5, 0.25, 0.75])
    near_weight = torch.tensor([[[0.6, 0.0, 0.2]]])
    delta = (fine - effective) * (1.0 - near_weight[..., None]) * action[None, None, :, None]
    torch.testing.assert_close(delta, torch.tensor([[[[0.0], [2.5], [0.0]]]]))


def test_wind_direction_pair_source_map_preserves_ids_and_physical_rotation():
    angle = np.deg2rad(15.0)
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    centers_from = np.asarray([
        [-2.0, 0.0, 0.875], [0.5, 1.0, 0.875], [3.0, -1.0, 0.875], [1.0, 4.0, 0.875]
    ])
    centers_to = centers_from.copy()
    centers_to[:, :2] = centers_from[:, :2] @ rotation.T + np.asarray([0.2, -0.1])
    ids = np.arange(4, dtype=np.int64)
    fitted = evaluator._fit_wind_source_rotation(ids, centers_from, ids.copy(), centers_to)
    assert fitted["rotation_degrees_ccw"] == pytest.approx(15.0, abs=1.0e-10)
    assert fitted["max_center_error_D"] < 1.0e-12
    mapped = evaluator._map_wind_direction_coordinates([[1.25, -0.5, 0.875]], fitted)
    expected = np.asarray([[1.25, -0.5]]) @ rotation.T + np.asarray([0.2, -0.1])
    np.testing.assert_allclose(mapped[0, :2], expected[0], atol=1.0e-12)
    with pytest.raises(ValueError, match="identical physical order"):
        evaluator._fit_wind_source_rotation(ids, centers_from, ids[::-1], centers_to)
    changed = centers_to.copy()
    changed[0, 0] += 0.01
    with pytest.raises(ValueError, match="source correspondence failed"):
        evaluator._fit_wind_source_rotation(ids, centers_from, ids, changed)


def test_wind_trilinear_native_reference_is_exact_for_linear_fields():
    x = np.asarray([-2.0, -0.5, 1.0, 4.0])
    y = np.asarray([-3.0, 0.0, 2.0])
    z = np.asarray([60.0, 70.0, 90.0])
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    # Convert to native C-order (z,y,x,3) and flatten by the documented map.
    grid = np.stack((xx + 2 * yy + 3 * zz, 2 * xx - yy + zz, -xx + yy - zz), axis=-1)
    field = np.transpose(grid, (2, 1, 0, 3)).reshape(-1, 3).astype(np.float32)
    run = SimpleNamespace(
        x_m=x, y_m=y, z_m=z, nx=x.size, ny=y.size, U=field,
    )
    queries_m = np.asarray([[-1.25, -1.5, 65.0], [2.5, 1.0, 82.0]])
    result = evaluator._wind_trilinear_reference(run, queries_m / 80.0)
    expected = np.column_stack((
        queries_m[:, 0] + 2 * queries_m[:, 1] + 3 * queries_m[:, 2],
        2 * queries_m[:, 0] - queries_m[:, 1] + queries_m[:, 2],
        -queries_m[:, 0] + queries_m[:, 1] - queries_m[:, 2],
    ))
    np.testing.assert_allclose(result["velocity_mps"], expected, rtol=1.0e-6, atol=1.0e-5)
    np.testing.assert_allclose(result["interpolation_weights"].sum(axis=1), 1.0, atol=1.0e-12)


def test_wind_same_layout_probe_uses_nine_target_free_receivers_and_reads_targets_after_both_rows(
    monkeypatch,
):
    evaluator._add_import_paths()
    import windfarm.geometry

    diameter = float(windfarm.geometry.D_M)
    counter = {"predictions": 0}

    class Run:
        def __init__(self, offset: float):
            self.x_m = np.linspace(-10 * diameter, 10 * diameter, 21)
            self.y_m = np.linspace(-10 * diameter, 10 * diameter, 21)
            self.z_m = np.asarray([60.0, 70.0, 90.0])
            self.nx = self.x_m.size
            self.ny = self.y_m.size
            self._offset = offset
            xx, yy, zz = np.meshgrid(self.x_m, self.y_m, self.z_m, indexing="ij")
            grid = np.stack((xx + 2 * yy + 3 * zz + offset, yy - zz, xx + offset), axis=-1)
            self._field = np.transpose(grid, (2, 1, 0, 3)).reshape(-1, 3).astype(np.float32)

        @property
        def U(self):
            assert counter["predictions"] == 6, "reference values were read before both scenes were predicted"
            return self._field

    angle = np.deg2rad(15.0)
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    centers_from = np.column_stack((np.linspace(-4.0, 4.0, 23), np.sin(np.arange(23)), np.full(23, 0.875)))
    centers_to = centers_from.copy()
    centers_to[:, :2] = centers_from[:, :2] @ rotation.T
    ids = np.arange(23, dtype=np.int64)
    scenes = {
        6: SimpleNamespace(source_ids=ids[None], present=np.ones((1, 23)), centers=centers_from[None],
                           environment_coords=np.zeros((1, 8, 3)), dependency=None),
        7: SimpleNamespace(source_ids=ids[None], present=np.ones((1, 23)), centers=centers_to[None],
                           environment_coords=np.zeros((1, 8, 3)), dependency=None),
    }
    cases = {
        6: SimpleNamespace(index=6, case="gen_0002_wd270", layout_index=2, n_turbines=23,
                           wind_direction_deg=270.0, run=Run(0.0),
                           support=SimpleNamespace(lower_D=np.zeros(3), upper_D=np.ones(3), extent_D=np.ones(3)),
                           module_centers=centers_from, module_present=np.ones(23),
                           module_features=np.zeros((23, 2)), global_context=np.zeros(11),
                           env_coords=np.zeros((8, 3)), env_features=np.zeros((8, 4)), env_weights=np.ones(8)),
        7: SimpleNamespace(index=7, case="gen_0002_wd285", layout_index=2, n_turbines=23,
                           wind_direction_deg=285.0, run=Run(1.0),
                           support=SimpleNamespace(lower_D=np.zeros(3), upper_D=np.ones(3), extent_D=np.ones(3)),
                           module_centers=centers_to, module_present=np.ones(23),
                           module_features=np.zeros((23, 2)), global_context=np.zeros(11),
                           env_coords=np.zeros((8, 3)), env_features=np.zeros((8, 4)), env_weights=np.ones(8)),
    }

    class Provider:
        def __init__(self):
            self.train_rows = np.asarray([6, 7])
            self.manifest = {"manifest_sha256": "fixed24-test"}
            self.view = SimpleNamespace(run=lambda row: cases[int(row)])

        def make_scene(self, scene_inputs):
            return scenes[int(scene_inputs[0].row_index)]

    axis = np.linspace(-2.0, 2.0, 5, dtype=np.float32)
    qx, qy = np.meshgrid(axis, axis)
    candidate_coordinates = np.column_stack((qx.reshape(-1), qy.reshape(-1), np.full(qx.size, 70.0 / diameter)))
    monkeypatch.setattr(
        evaluator,
        "_wind_plane_geometry",
        lambda run, requested_height_m=70.0: {
            "coords_D": candidate_coordinates,
            "flat_indices": np.arange(candidate_coordinates.shape[0]),
            "shape": (5, 5),
            "index": 1,
            "actual_m": 70.0,
            "requested_m": 70.0,
        },
    )

    def predict(_model, scene, coordinates, *, mode, threshold, temperature):
        counter["predictions"] += 1
        offset = float(scene.centers[0, 0, 0])
        velocity = np.full((1, len(coordinates), 3), offset, dtype=np.float32)
        velocity += {"all_fine": 2.0, "all_base": 1.0, "adaptive": 1.5}[mode]
        result = {"velocity_mps": velocity, "query_coordinates_D": np.asarray(coordinates)[None]}
        if mode == "adaptive":
            result.update({
                "route/keep": np.ones((1, len(coordinates), 23), dtype=np.float32),
                "route/protected": np.zeros((1, len(coordinates), 23), dtype=np.float32),
                "route/probability": np.full((1, len(coordinates), 23), 0.75, dtype=np.float32),
            })
        return result, {"active_pairs": len(coordinates) * 23}

    monkeypatch.setattr(evaluator, "_wind_predict_plane", predict)
    monkeypatch.setattr(
        evaluator,
        "_wind_partition",
        lambda _provider, _row: {
            "original_split": "original_train",
            "fixed24_v1_membership": "TRAIN",
            "development_metrics_population": False,
        },
    )
    report, arrays = evaluator._evaluate_wind_same_layout_direction_pair(
        SimpleNamespace(), Provider(), temperature=1.0
    )
    assert counter["predictions"] == 6
    assert report["selection"]["receiver_count"] == 9
    assert report["selection"]["uses_target_values"] is False
    assert report["source_correspondence"]["matched_source_count"] == 23
    np.testing.assert_allclose(
        arrays["layout2_direction_pair/row6/F_minus_B/velocity_mps"], 1.0
    )
    assert arrays["layout2_direction_pair/row6/adaptive/keep"].shape == (1, 9, 23)
    assert arrays["layout2_direction_pair/row7/target/velocity_mps"].shape == (9, 3)
    assert arrays["layout2_direction_pair/row7/target/interpolation_flat_indices"].shape == (9, 8)


@pytest.mark.parametrize("canonical", [False, True])
def test_monitor_cache_reuse_requires_exact_run_arm_epoch_and_ordered_thermal_dev_membership(tmp_path, canonical):
    run_id = "thermal-fixed25"
    run_directory = tmp_path / "runs" / "thermal" / run_id / "adaptive_detail"
    run_directory.mkdir(parents=True)
    checkpoint = run_directory / ("checkpoints/best_by_field_mse_model.pt" if canonical else "best_by_field_mse_model.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"trusted-checkpoint")
    case_ids = [f"{index:04d}" for index in range(22)]
    cache = {
        "validation_execution_mode": "adaptive_detail",
        "validation_route_phase": "hard",
        "case_count": 22,
        "case_ids": case_ids,
        "field_score": 0.5,
    }
    cache_path = run_directory / ("evaluations/validation/validation_epoch_1200.json" if canonical else "validation_epoch_1200.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    payload = {
        "epoch": 1200,
        "arm": "adaptive_detail",
        "experiment_identity": {"run_id": run_id},
        "history": [{"epoch": 1200, "validation": dict(cache)}],
    }
    provider = SimpleNamespace(validation_cases=[{"case_id": case_id} for case_id in case_ids])
    loaded = evaluator._load_training_validation_cache(
        payload,
        task="thermal",
        provider=provider,
        checkpoint_path=checkpoint,
        runs_root=tmp_path / "runs",
    )
    assert loaded is not None
    assert loaded[0]["field_score"] == 0.5
    assert loaded[1]["checkpoint_embeds_exact_validation_metrics"] is True
    edited_cache = {**cache, "field_score": 0.6}
    cache_path.write_text(json.dumps(edited_cache), encoding="utf-8")
    assert evaluator._load_training_validation_cache(
        payload,
        task="thermal",
        provider=provider,
        checkpoint_path=checkpoint,
        runs_root=tmp_path / "runs",
    ) is None
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    provider.validation_cases[0], provider.validation_cases[1] = (
        provider.validation_cases[1], provider.validation_cases[0]
    )
    assert evaluator._load_training_validation_cache(
        payload,
        task="thermal",
        provider=provider,
        checkpoint_path=checkpoint,
        runs_root=tmp_path / "runs",
    ) is None


def test_monitor_cache_reuse_checks_exact_ordered_wind_rows(tmp_path):
    run_id = "wind-fixed24"
    run_directory = tmp_path / "runs" / "wind" / run_id / "full_detail"
    run_directory.mkdir(parents=True)
    checkpoint = run_directory / "epoch_1000_model.pt"
    checkpoint.write_bytes(b"trusted-checkpoint")
    expected_rows = list(range(24))
    cache = {
        "validation_execution_mode": "full_detail",
        "validation_route_phase": "hard",
        "row_count": 24,
        "rows": [{"row_index": row} for row in expected_rows],
    }
    (run_directory / "validation_epoch_1000.json").write_text(json.dumps(cache), encoding="utf-8")
    payload = {
        "epoch": 1000,
        "arm": "full_detail",
        "experiment_identity": {"run_id": run_id},
        "history": [{"epoch": 1000, "validation": cache}],
    }
    provider = SimpleNamespace(validation_rows=np.asarray(expected_rows))
    assert evaluator._load_training_validation_cache(
        payload,
        task="wind",
        provider=provider,
        checkpoint_path=checkpoint,
        runs_root=tmp_path / "runs",
    ) is not None
    provider.validation_rows = np.asarray(expected_rows[::-1])
    assert evaluator._load_training_validation_cache(
        payload,
        task="wind",
        provider=provider,
        checkpoint_path=checkpoint,
        runs_root=tmp_path / "runs",
    ) is None


def test_fresh_thermal_dev_reducer_handles_variable_source_counts_without_physical_concat():
    import torch

    batches = []
    expected_source_rows = 0
    for index in range(22):
        source_count = 1 + index % 5
        expected_source_rows += source_count
        q = 3
        field_targets = torch.ones((1, q, 5))
        field_targets[0, :, 4] = torch.tensor([1.0, 2.0, 3.0])
        structure = {
            "module_centers": np.zeros((1, source_count, 2), dtype=np.float32),
            "module_present": np.ones((1, source_count), dtype=np.float32),
            "material_params": np.asarray([[0.0, 0.0, 0.0, 1.0, 1.0, 0.5]], dtype=np.float32),
            "u_in": np.asarray([[1.0]], dtype=np.float32),
        }
        targets = SimpleNamespace(
            case_ids=(f"case-{index:02d}",),
            field_targets=field_targets,
            point_weights=torch.tensor([[1.0, 1.5, 3.0]]),
            interface_target=torch.ones((1, source_count, 1, 2)),
            material_targets=torch.ones((1, source_count, 1)),
        )
        receivers = SimpleNamespace(fluid_xy=torch.zeros((1, q, 2)))
        batches.append(SimpleNamespace(
            scene_inputs={"index": index, "structure": structure}, receivers=receivers, targets=targets,
        ))

    class FakeProvider:
        def validation_batches(self):
            return batches

        def make_scene(self, inputs):
            return inputs

        def predict_native(self, model, scene, receivers, *args, **kwargs):
            del model, args, kwargs
            source_count = 1 + int(scene["index"]) % 5
            native_main = {
                "fluid_temperature": torch.zeros((1, 3, 1)),
                "pred_interface": torch.zeros((1, source_count, 1, 2)),
                "pred_internal_temperature": torch.zeros((1, source_count, 1, 1)),
            }
            prepared = SimpleNamespace(
                source_present=torch.ones((1, source_count)),
                stencils={
                    "fluid": SimpleNamespace(valid=torch.ones((1, 3), dtype=torch.bool)),
                    "surface": SimpleNamespace(valid=torch.ones((1, source_count, 1), dtype=torch.bool)),
                    "material": SimpleNamespace(valid=torch.ones((1, source_count, 1), dtype=torch.bool)),
                },
            )
            prediction = SimpleNamespace(
                native_main=native_main,
                native_prepared=prepared,
                work={"fine_rows": 1, "cheap_rows": 2, "gate_rows": 3, "near_rows": 4,
                      "selected_detail_rows": 5, "active_pairs": source_count * 3},
            )
            return prediction, {}

        def validation_metrics(self, predictions, targets, auxiliary):
            del predictions, auxiliary
            return {"case_rows": [{"case_id": targets.case_ids[0]}]}

        def reduce_native_metrics(self, records):
            rows = [row for record in records for row in record["case_rows"]]
            return {"case_count": len(rows), "case_rows": rows}

    metrics = evaluator._collect_validation(
        "thermal", object(), FakeProvider(), mode="all_fine", threshold=0.5,
        epoch=1000, temperature=1.0,
    )
    assert metrics["case_count"] == 22
    assert metrics["residual_tails"]["fluid_temperature"]["count"] == 22 * 3
    assert metrics["residual_tails"]["fluid_temperature"]["rmse"] == pytest.approx(np.sqrt(14.0 / 3.0))
    assert metrics["residual_tail_definition"]["aggregation"] == (
        "unweighted point-pooled residuals across the provider's validation samples"
    )
    assert metrics["residual_tail_definition"]["fluid_temperature"]["point_weight_application"].startswith(
        "positive weights are an inclusion mask only"
    )
    fluid_boundary = metrics["data_boundary"]["thermal_fluid_validation_sample"]
    assert fluid_boundary["sampled_case_rows"] == 22
    assert fluid_boundary["sampled_query_rows"] == 22 * 3
    assert fluid_boundary["positive_point_weight_rows"] == 22 * 3
    assert fluid_boundary["eligible_residual_tail_rows"] == 22 * 3
    assert fluid_boundary["observed_finite_point_weight_min"] == 1.0
    assert fluid_boundary["observed_finite_point_weight_max"] == 3.0
    assert not fluid_boundary["point_weight_magnitudes_applied_to_residual_tails"]
    assert metrics["residual_tails"]["interface"]["count"] == expected_source_rows
    assert metrics["residual_tails"]["material_temperature"]["count"] == expected_source_rows
    assert metrics["residual_tails"]["near/temperature"]["count"] == 22 * 3
    assert metrics["residual_tails"]["downstream/temperature"]["count"] == 0
    assert metrics["near_downstream_diagnostics"]["equal_case"]["near"]["available_case_count"] == 22
    assert metrics["near_downstream_diagnostics"]["equal_case"]["downstream"]["available_case_count"] == 0


def test_thermal_fixed_case_conversion_batches_only_whitelisted_fields():
    import torch

    evaluator._add_import_paths()
    structure = {
        "module_centers": np.zeros((3, 2), dtype=np.float64),
        "module_present": np.ones(3, dtype=np.float32),
        "module_source_ids": np.asarray([11, 12, 13], dtype=np.int64),
        "material_params": np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        "re": np.asarray([100.0]),
        "heat_powers": np.asarray([4.0, 5.0, 6.0]),
        "field_targets": np.ones((4, 5)),
    }
    converted = evaluator._thermal_context_tensors(structure, torch.device("cpu"))
    assert set(converted) == {"module_centers", "module_present", "module_source_ids", "material_params", "re"}
    assert converted["module_centers"].shape == (1, 3, 2)
    assert converted["module_present"].shape == (1, 3)
    assert converted["module_source_ids"].shape == (1, 3)
    assert converted["module_source_ids"].dtype == torch.int64
    assert converted["module_centers"].dtype == torch.float32


@pytest.mark.parametrize("batched_material", [False, True])
@pytest.mark.parametrize(("catalogue_changes", "expected_valid"), [(False, True), (True, False)])
def test_thermal_geometry_ad_fd_maps_native_receiver_ids_and_checks_full_route_masks(
    catalogue_changes, expected_valid, batched_material,
):
    import torch

    class FakeProvider:
        @staticmethod
        def _set_execution(*_args, **_kwargs):
            return None

    class FakeModel:
        def __init__(self):
            self.catalogue_changes = catalogue_changes

        def prepare_native(self, structure, query, *, ntheta):
            del query, ntheta
            center = structure["module_centers"]
            ids = [99, 222, 88]
            if self.catalogue_changes and float(center[0, 0, 0].detach()) > 1.0004:
                ids = [99, 222, 77]
            route_keep = torch.tensor([[[False, False], [True, False], [False, True]]])
            protected = torch.zeros_like(route_keep)
            response = SimpleNamespace(
                receiver_ids=torch.tensor([ids], dtype=torch.int64),
                refinement_aux={"keep": route_keep, "protected": protected},
            )
            return SimpleNamespace(response=response, structure=structure)

        @staticmethod
        def apply_native(prepared, heat, *, accumulation_dtype):
            del heat, accumulation_dtype
            center = prepared.structure["module_centers"]
            return {"fluid_temperature": center[0, 0, 0].square().reshape(1, 1, 1)}

    material_params = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.5])
    if batched_material:
        material_params = material_params.unsqueeze(0)
    scene = SimpleNamespace(structure={
        "module_centers": torch.tensor([[[1.0, 0.0], [2.0, 0.0]]]),
        "module_source_ids": torch.tensor([[17, 29]], dtype=torch.int64),
        "material_params": material_params,
    })
    neural_coordinates = np.asarray([[[0.0, 0.0], [1.0, 0.0]]], dtype=np.float32)
    native_grid_indices = np.asarray([[222, 111]], dtype=np.int64)
    adaptive_route = {
        "route/probability": np.asarray([[[0.9, 0.1], [0.7, 0.3]]]),
        "route/protected": np.zeros((1, 2, 2), dtype=bool),
        "route/keep": np.asarray([[[True, False], [True, False]]]),
        "source_present": np.ones((1, 2), dtype=bool),
    }
    arrays, receipt = evaluator._thermal_geometry_ad_fd_demo(
        FakeModel(), FakeProvider(), scene, torch.ones((1, 2)), neural_coordinates,
        native_grid_indices, adaptive_route, temperature=1.0,
    )
    np.testing.assert_array_equal(arrays["native_grid_id"], [222])
    np.testing.assert_array_equal(arrays["route_keep_base"], [[[True, False]]])
    assert receipt["rebuilt_base_route_matches_saved_route"] is True
    assert receipt["rebuilt_native_receiver_catalogue_unchanged_at_both_endpoints"] is expected_valid
    assert receipt["local_fixed_route_fd_check_valid"] is expected_valid
    assert receipt["step_native_geometry_units"] == pytest.approx(5.0e-4)
    assert np.isfinite(receipt["relative_difference"])

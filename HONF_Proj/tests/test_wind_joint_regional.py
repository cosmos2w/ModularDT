from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from windfarm.geometry import POSITIONAL_SCALE_D
from windfarm.joint_regional import WindFarmJointRegionalModel
from windfarm.normalization import VelocityNormalizer, VerticalProfileBaseline
from windfarm.training.joint_task import (
    WindJointReceivers,
    WindJointRegionalTask,
    WindJointSceneInputs,
    WindJointTargets,
    _calibration_cache_payload,
    _fit_or_load_formal_transforms,
    _joint_scene_batch,
    _read_role_scale_cache,
    _read_transform_cache,
    _role_counts_for_query_count,
    _sampling_dataset_id,
    _transform_cache_payload,
    build_wind_joint_task,
)

from honf_runtime.unified_training import SamplingKey, TaskBatch
from honf_runtime.unified_training import _sampling_dataset_id as runtime_sampling_dataset_id


def _scene_input(row: int = 0, *, sources: int = 2, shift: float = 0.0) -> WindJointSceneInputs:
    lower = np.asarray([-3.0 + shift, -2.0, 0.0], dtype=np.float32)
    extent = np.asarray([10.0, 8.0, 6.25], dtype=np.float32)
    upper = lower + extent
    scale = np.asarray(POSITIONAL_SCALE_D, dtype=np.float32)
    centers = np.zeros((sources, 3), dtype=np.float32)
    centers[:, 0] = lower[0] + np.linspace(2.0, 7.0, sources, dtype=np.float32)
    centers[:, 1] = lower[1] + np.linspace(2.0, 6.0, sources, dtype=np.float32)
    centers[:, 2] = 0.875
    present = np.ones(sources, dtype=np.float32)
    features = np.column_stack(
        (np.full(sources, 0.5, dtype=np.float32), np.full(sources, 0.875, dtype=np.float32))
    )
    grid_axes = [lower[i] + (np.arange(4, dtype=np.float32) + 0.5) * extent[i] / 4.0 for i in range(3)]
    mesh = np.meshgrid(*grid_axes, indexing="ij")
    env = np.stack([axis.reshape(-1) for axis in mesh], axis=-1).astype(np.float32)
    lower_norm = np.broadcast_to(lower / scale, (64, 3))
    features_env = np.concatenate((env / scale, lower_norm, np.full((64, 1), 1.0 / 64.0)), axis=-1)
    weights = np.full(64, float(np.prod(extent) / 64.0), dtype=np.float32)
    context = np.zeros(11, dtype=np.float32)
    context[row % 3] = 1.0
    context[3] = float(sources / 30.0)
    context[4] = 1.0
    context[5:8] = lower / scale
    context[8:11] = extent / scale
    for value in (centers, present, features, env, features_env, weights, context, lower, upper, extent):
        value.setflags(write=False)
    return WindJointSceneInputs(
        row_index=row,
        case_name=f"synthetic-{row}",
        layout_index=row,
        wind_direction_deg=(270.0, 285.0, 300.0)[row % 3],
        module_centers=centers,
        module_present=present,
        module_features=features,
        module_ids=np.arange(sources, dtype=np.int64),
        global_context=context,
        support_lower_D=lower,
        support_upper_D=upper,
        support_extent_D=extent,
        env_coords=env,
        env_features=features_env,
        env_weights=weights,
    )


def _transforms() -> tuple[VelocityNormalizer, VerticalProfileBaseline]:
    normalizer = VelocityNormalizer(
        mean=np.asarray([0.3, -0.1, 0.0]),
        std=np.asarray([0.4, 0.3, 0.2]),
        safe_std=np.asarray([0.4, 0.3, 0.2]),
        u_ref_mps=10.0,
        sample_count_per_row=8192,
        source_rows=72,
        seed=42,
    )
    centers = np.linspace(0.0, 6.25, 8, dtype=np.float64)
    values = np.column_stack((8.0 + centers * 0.1, 1.0 - centers * 0.05, centers * 0.02))
    profile = VerticalProfileBaseline(centers, values, np.full(8, 100, dtype=np.int64))
    return normalizer, profile


def _model(mode: str = "J-H", *, locality_prior_strength: float = 0.0) -> WindFarmJointRegionalModel:
    torch.manual_seed(42)
    normalizer, profile = _transforms()
    return WindFarmJointRegionalModel(
        velocity_transform=normalizer,
        background_profile=profile,
        mode=mode,
        hidden=16,
        message=16,
        regional_anchors=4,
        depth=2,
        receiver_tile=2,
        seed=42,
        locality_prior_strength=locality_prior_strength,
    )


def _joint_cli_module():
    cli_path = Path(__file__).resolve().parents[1] / "tools" / "joint_regional_train.py"
    spec = importlib.util.spec_from_file_location("wind_joint_regional_train_test", cli_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _receivers(batch: int = 1) -> torch.Tensor:
    points = torch.tensor(
        [[[-2.5, -1.0, 1.0], [0.0, 1.0, 2.5], [3.0, 3.0, 4.5], [5.0, 5.0, 6.0]]],
        dtype=torch.float32,
    )
    return points.expand(batch, -1, -1).contiguous()


def test_wind_joint_model_preserves_native_profile_frame_and_standalone_load() -> None:
    model = _model()
    scene = _joint_scene_batch((_scene_input(),), torch.device("cpu"))
    prepared = model.prepare_scene(scene)
    receivers = _receivers()
    baseline = model.profile_at_receivers(receivers)
    with torch.no_grad():
        final = model.core.field_head[-1]
        final.weight.zero_()
        final.bias.zero_()
    prepared = model.prepare_scene(scene)
    prediction = model.predict_physical(prepared, receivers)
    torch.testing.assert_close(prediction, baseline, rtol=0.0, atol=1.0e-6)
    assert prediction.shape == (1, 4, 3)
    assert prediction[0, 0].tolist() == pytest.approx(baseline[0, 0].tolist(), abs=1.0e-6)

    config = model.export_config()
    assert "locality_prior_strength" not in config["core"]
    loaded = WindFarmJointRegionalModel.from_config(config)
    loaded.load_state_dict(model.state_dict())
    loaded.eval()
    loaded_prediction = loaded.predict_physical(loaded.prepare_scene(scene), receivers)
    torch.testing.assert_close(loaded_prediction, prediction, rtol=0.0, atol=1.0e-6)
    assert config["exact_affine_increments"] is False
    with pytest.raises(ValueError, match="does not support exact affine"):
        loaded.apply_increment(torch.ones(1))


def test_wind_locality_prior_roundtrip_and_native_input_query_derivatives() -> None:
    assert inspect.signature(build_wind_joint_task).parameters["locality_prior_strength"].default == 0.0
    model = _model(locality_prior_strength=1.0)
    no_prior = _model()
    scene = _joint_scene_batch((_scene_input(),), torch.device("cpu"))
    receivers = _receivers()
    with torch.no_grad():
        prior_prepared = model.prepare_scene(scene)
        baseline_prepared = no_prior.prepare_scene(scene)
        assert not torch.allclose(
            prior_prepared.core_context.source_membership,
            baseline_prepared.core_context.source_membership,
        )
        assert not torch.allclose(
            prior_prepared.core_context.environment_membership,
            baseline_prepared.core_context.environment_membership,
        )

    config = model.export_config()
    assert config["core"]["locality_prior_strength"] == 1.0
    loaded = WindFarmJointRegionalModel.from_config(config)
    loaded.load_state_dict(model.state_dict(), strict=True)
    loaded.eval()
    torch.testing.assert_close(
        loaded.predict_physical(loaded.prepare_scene(scene), receivers),
        model.predict_physical(model.prepare_scene(scene), receivers),
        rtol=0.0,
        atol=0.0,
    )
    assert loaded.locality_prior_strength == 1.0

    grad_scene = replace(scene, centers=scene.centers.detach().clone().requires_grad_(True))
    grad_receivers = receivers.detach().clone().requires_grad_(True)
    scalar = model.predict_physical(model.prepare_scene(grad_scene), grad_receivers)[0, 1, 0]
    center_grad, query_grad = torch.autograd.grad(scalar, (grad_scene.centers, grad_receivers))
    center_value = center_grad[0, 0, 0]
    query_value = query_grad[0, 1, 0]
    assert torch.isfinite(center_value) and torch.isfinite(query_value)
    assert abs(float(center_value)) > 1.0e-7
    assert abs(float(query_value)) > 1.0e-7

    epsilon = 1.0e-2
    with torch.no_grad():
        center_plus, center_minus = scene.centers.clone(), scene.centers.clone()
        center_plus[0, 0, 0] += epsilon
        center_minus[0, 0, 0] -= epsilon
        finite_center = (
            model.predict_physical(model.prepare_scene(replace(scene, centers=center_plus)), receivers)[0, 1, 0]
            - model.predict_physical(model.prepare_scene(replace(scene, centers=center_minus)), receivers)[0, 1, 0]
        ) / (2.0 * epsilon)
        query_plus, query_minus = receivers.clone(), receivers.clone()
        query_plus[0, 1, 0] += epsilon
        query_minus[0, 1, 0] -= epsilon
        finite_query = (
            model.predict_physical(model.prepare_scene(scene), query_plus)[0, 1, 0]
            - model.predict_physical(model.prepare_scene(scene), query_minus)[0, 1, 0]
        ) / (2.0 * epsilon)
    torch.testing.assert_close(center_value, finite_center, rtol=2.0e-2, atol=2.0e-4)
    torch.testing.assert_close(query_value, finite_query, rtol=2.0e-2, atol=2.0e-4)
    with pytest.raises(ValueError, match="only for J-H"):
        _model("J-geometry", locality_prior_strength=1.0)


def test_wind_joint_predictions_are_source_environment_query_and_batch_order_invariant() -> None:
    model = _model()
    first, second = _scene_input(0, sources=2), _scene_input(1, sources=3, shift=0.3)
    scene = _joint_scene_batch((first, second), torch.device("cpu"))
    receivers = _receivers(batch=2)
    with torch.no_grad():
        pair = model.predict_physical(model.prepare_scene(scene), receivers)
        alone_first = model.predict_physical(model.prepare_scene(_joint_scene_batch((first,), torch.device("cpu"))), receivers[:1])
        alone_second = model.predict_physical(model.prepare_scene(_joint_scene_batch((second,), torch.device("cpu"))), receivers[1:])
    torch.testing.assert_close(pair[:1], alone_first, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(pair[1:], alone_second, rtol=1.0e-5, atol=1.0e-5)

    source_order = torch.tensor([2, 0, 1])
    permuted_source = replace(
        scene,
        sources=scene.sources[:, source_order],
        centers=scene.centers[:, source_order],
        present=scene.present[:, source_order],
        source_lengths=scene.source_lengths[:, source_order],
        source_measures=scene.source_measures[:, source_order],
        source_ids=scene.source_ids[:, source_order],
    )
    environment_order = torch.arange(63, -1, -1)
    permuted_environment = replace(
        scene,
        environment_tokens=scene.environment_tokens[:, environment_order],
        environment_coords=scene.environment_coords[:, environment_order],
        environment_present=scene.environment_present[:, environment_order],
        environment_measures=scene.environment_measures[:, environment_order],
    )
    query_order = torch.tensor([3, 1, 0, 2])
    with torch.no_grad():
        source_prediction = model.predict_physical(model.prepare_scene(permuted_source), receivers)
        environment_prediction = model.predict_physical(model.prepare_scene(permuted_environment), receivers)
        query_prediction = model.predict_physical(model.prepare_scene(scene), receivers[:, query_order])
    torch.testing.assert_close(source_prediction, pair, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(environment_prediction, pair, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(query_prediction, pair[:, query_order], rtol=1.0e-5, atol=1.0e-5)


def test_wind_joint_fresh_optimizer_step_trains_all_executed_core_parameters() -> None:
    model = _model()
    scene = _joint_scene_batch((_scene_input(),), torch.device("cpu"))
    receivers = _receivers()
    target = model.profile_at_receivers(receivers) + torch.tensor(
        [[[0.3, -0.1, 0.05], [-0.2, 0.2, 0.1], [0.1, 0.1, -0.2], [-0.15, -0.1, 0.2]]]
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    prediction = model(scene, receivers)
    (prediction - target).square().mean().backward()
    missing_gradients = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    zero_gradients = [
        name for name, parameter in model.named_parameters()
        if parameter.grad is not None and not bool(torch.count_nonzero(parameter.grad))
    ]
    assert not missing_gradients
    assert not zero_gradients
    optimizer.step()
    unchanged = [
        name for name, parameter in model.named_parameters()
        if torch.equal(before[name], parameter.detach())
    ]
    assert not unchanged
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_wind_joint_prepared_state_rebuilds_after_geometry_or_weight_change() -> None:
    model = _model()
    scene = _joint_scene_batch((_scene_input(),), torch.device("cpu"))
    receivers = _receivers()
    prepared = model.prepare_scene(scene)
    with torch.no_grad():
        before = model.predict_physical(prepared, receivers)
        changed_scene = replace(scene, centers=scene.centers + torch.tensor([0.2, 0.0, 0.0]))
        changed = model.predict_physical(model.prepare_scene(changed_scene), receivers)
    assert not torch.allclose(before, changed)
    with torch.no_grad():
        scene.centers[0, 0, 0].add_(0.1)
    with pytest.raises(ValueError, match="changed after preparation|stale"):
        model.predict_physical(prepared, receivers)

    rebuilt = model.prepare_scene(scene)
    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    with pytest.raises(ValueError, match="parameters changed after preparation|stale"):
        model.predict_physical(rebuilt, receivers)


def test_wind_joint_target_poisoning_cannot_enter_scene_or_prediction() -> None:
    model = _model()
    item = _scene_input()
    provider = object.__new__(WindJointRegionalTask)
    provider.device = torch.device("cpu")
    receivers_np = _receivers().numpy()
    role_ids = np.asarray([[0, 1, 2, 3]], dtype=np.int8)
    batch = TaskBatch(
        scene_inputs=(item,),
        receivers=WindJointReceivers(receivers_np, role_ids),
        targets=WindJointTargets(np.zeros((1, 4, 3), dtype=np.float32), (0,), ({"row_index": 0},)),
        case_keys=(0,),
    )
    scene = provider.make_scene(batch.scene_inputs)
    pred, _ = provider.predict_native(model, scene, batch.receivers)
    poisoned_batch = replace(
        batch,
        targets=WindJointTargets(
            np.full((1, 4, 3), 1.0e8, dtype=np.float32), (0,), ({"row_index": 0},)
        ),
    )
    poisoned_scene = provider.make_scene(poisoned_batch.scene_inputs)
    poisoned_pred, _ = provider.predict_native(model, poisoned_scene, poisoned_batch.receivers)
    torch.testing.assert_close(pred.main_mps, poisoned_pred.main_mps, rtol=0.0, atol=0.0)
    assert not hasattr(item, "velocity_mps")
    assert item.env_coords.shape == (64, 3)
    assert item.env_features.shape == (64, 7)
    assert item.env_weights.sum() == pytest.approx(float(np.prod(item.support_extent_D)), rel=1.0e-6)


def test_wind_joint_role_mixture_and_native_vector_objective_are_exact() -> None:
    counts = _role_counts_for_query_count(4096)
    assert counts == {
        "volume": 820,
        "hub_slab": 820,
        "downstream_envelope": 820,
        "near_turbine": 820,
        "background": 816,
    }
    assert sum(counts.values()) == 4096
    assert len(counts) == 5


def test_wind_joint_task_provider_keeps_case_epoch_queries_and_losses_packing_invariant(monkeypatch) -> None:
    from windfarm.training import joint_task

    items = [
        _scene_input(0, sources=2),
        _scene_input(1, sources=3, shift=0.3),
        _scene_input(2, sources=1),
        _scene_input(3, sources=2, shift=0.15),
    ]
    cases = []
    for item in items:
        support = SimpleNamespace(
            lower_D=item.support_lower_D,
            upper_D=item.support_upper_D,
            extent_D=item.support_extent_D,
        )
        cases.append(
            SimpleNamespace(
                index=item.row_index,
                case=item.case_name,
                layout_index=item.layout_index,
                wind_direction_deg=item.wind_direction_deg,
                module_centers=item.module_centers,
                module_present=item.module_present,
                module_features=item.module_features,
                global_context=item.global_context,
                support=support,
                env_coords=item.env_coords,
                env_features=item.env_features,
                env_weights=item.env_weights,
            )
        )

    class FakeView:
        def __init__(self):
            self.metadata = {"layout_index": np.asarray([0, 1, 2, 3], dtype=np.int64)}

        @staticmethod
        def run(row: int):
            return cases[int(row)]

    def fake_sample(case, rng, role_counts, *, rng_by_role, **_kwargs):
        del rng
        pieces = []
        for role in joint_task.ROLE_NAMES:
            role_rng = rng_by_role[role]
            lower = np.asarray(case.support.lower_D, dtype=np.float32)
            extent = np.asarray(case.support.extent_D, dtype=np.float32)
            pieces.append(lower + role_rng.random((role_counts[role], 3)).astype(np.float32) * extent)
        coordinates = np.concatenate(pieces, axis=0)
        target = np.column_stack(
            (
                8.0 + coordinates[:, 0] * 0.01 + case.index,
                1.0 + coordinates[:, 1] * 0.02,
                coordinates[:, 2] * 0.03,
            )
        ).astype(np.float32)
        return SimpleNamespace(
            coordinates_D=coordinates,
            target_mps=target,
            role_sample_counts=dict(role_counts),
        )

    monkeypatch.setattr(joint_task, "sample_native_role_queries", fake_sample)
    model = _model()
    role_scales = {role: (0.4, 0.3, 0.2) for role in joint_task.ROLE_NAMES}
    fingerprint = "fixed-test-training-fingerprint"
    manifest_fingerprint = "fixed-test-manifest-fingerprint"
    provider = WindJointRegionalTask(
        FakeView(),
        model=model,
        train_rows=[0, 1],
        validation_rows=[2, 3],
        manifest={"subset_id": "wind_shared_fixed24_v1", "manifest_sha256": manifest_fingerprint},
        profile_id="wind_shared_fixed24_v1",
        training_fingerprint=fingerprint,
        role_component_scales=role_scales,
        role_scale_calibration={"method": "synthetic test calibration"},
        role_scale_cache_identity={"cache_key_sha256": "synthetic-cache-key"},
        role_scale_source_sha256="scale-test-sha",
        normalizer_source={"kind": "synthetic test transforms"},
        seed=42,
        primary_queries=10,
        microbatch_size=2,
        effective_batch_size=24,
        total_epochs=2500,
        device="cpu",
        catalogue_cache=SimpleNamespace(
            max_cached_bytes=0,
            summary=lambda: {
                "catalogue_count": 0,
                "catalogue_build_count": 0,
                "cache_hit_count": 0,
                "cache_miss_count": 0,
                "cache_eviction_count": 0,
                "cache_oversize_bypass_count": 0,
                "cache_capacity_bytes": 0,
                "cached_bytes": 0,
            },
        ),
    )
    assert provider.sampling_dataset_id == runtime_sampling_dataset_id(provider.identity_payload())
    key = SamplingKey(
        42,
        7,
        0,
        0,
        "train",
        "J-H",
        sampling_version=SamplingKey.CASE_EPOCH_VERSION,
        dataset_id=_sampling_dataset_id(provider.profile_id, provider.manifest["manifest_sha256"]),
    )
    packed = provider.make_batch((0, 1), key)
    first = provider.make_batch((0,), key)
    second = provider.make_batch((1,), key)
    assert packed.receivers.role_ids.shape == (2, 10)
    assert np.array_equal(packed.receivers.coordinates_D[0], first.receivers.coordinates_D[0])
    assert np.array_equal(packed.receivers.coordinates_D[1], second.receivers.coordinates_D[0])

    packed_scene = provider.make_scene(packed.scene_inputs)
    first_scene = provider.make_scene(first.scene_inputs)
    second_scene = provider.make_scene(second.scene_inputs)
    with torch.no_grad():
        packed_prediction, _ = provider.predict_native(model, packed_scene, packed.receivers)
        first_prediction, _ = provider.predict_native(model, first_scene, first.receivers)
        second_prediction, _ = provider.predict_native(model, second_scene, second.receivers)
    torch.testing.assert_close(packed_prediction.main_mps[:1], first_prediction.main_mps, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(packed_prediction.main_mps[1:], second_prediction.main_mps, rtol=1.0e-5, atol=1.0e-5)

    packed_denominators = provider.loss_denominators((packed,))
    split_denominators = provider.loss_denominators((first, second))
    assert packed_denominators == split_denominators
    terms = provider.loss_terms(packed_prediction, packed.targets, "joint", {})
    assert set(terms) == {f"native_role/{role}" for role in joint_task.ROLE_NAMES}
    assert all(term.weight == 0.2 for term in terms.values())
    assert provider.phase_metadata() == {
        "training_mode": "joint", "phase": "joint", "gate_schedule": None, "active_parameters": "all"
    }
    validation_batches = tuple(provider.validation_batches())
    assert len(validation_batches) == 1
    assert validation_batches[0].case_keys == (2, 3)
    assert validation_batches[0].targets.row_indices == (2, 3)

    from honf_runtime.unified_training import EngineConfig, SelectionPolicy, TrainingEngine

    cli = _joint_cli_module()
    engine = TrainingEngine(
        EngineConfig(
            seed=42,
            microbatch_cases=2,
            effective_cases=24,
            total_epochs=2500,
            training_mode="joint",
            sampling_version=SamplingKey.CASE_EPOCH_VERSION,
        ),
        device="cpu",
        selection=SelectionPolicy(field_metric="field_score"),
    )
    setup = provider.preparation_summary()
    assert setup["status"] == "prepared_only"
    assert setup["optimizer_started"] is False
    assert setup["training_query_samples"] == provider.training_samples * provider.primary_queries
    payload = cli.summary(model, provider, engine, {"recipe": {"mode": "J-H"}})
    serialized = json.dumps(payload, allow_nan=False)
    assert json.loads(serialized)["preparation"]["wind_test_target_values_read"] is False
    preflight = engine.preflight_one_update(model, provider, arm="J-H")
    assert preflight["optimizer_updates"] == 1
    assert preflight["case_keys"] == [0]
    validation = engine._evaluate(model, provider, "J-H", "joint", epoch=1)
    assert validation["row_count"] == 2
    assert [row["row_index"] for row in validation["rows"]] == [2, 3]


def test_wind_role_scale_cache_is_checksummed_and_exact_source_bound(tmp_path) -> None:
    cache_key = {
        "schema_version": 1,
        "calibration_id": "wind_joint_e64_native_role_scales_v1",
        "calibration_queries": {
            "target_values_from_windtest": False,
            "target_partition": "selected TRAIN calibration rows only",
        },
        "train_row_indices_sha256": "rows-train-sha",
        "training_fingerprint": "wind-train-source-sha",
        "background_profile_sha256": "profile-sha",
        "calibration_layout_indices": [2, 10, 25, 46],
        "calibration_row_indices": [0, 1, 2, 3],
    }
    calibration = {
        "training_rows_sha256": "rows-train-sha",
        "training_fingerprint": "wind-train-source-sha",
        "background_profile_sha256": "profile-sha",
        "calibration_layout_indices": [2, 10, 25, 46],
        "calibration_row_indices": [0, 1, 2, 3],
        "calibration_query_count_per_row": 1024,
        "target_values_read": "TRAIN calibration panel only",
        "solver_attempts": 0,
        "component_role_scales_mps": {role: [0.4, 0.3, 0.2] for role in (
            "volume", "hub_slab", "downstream_envelope", "near_turbine", "background"
        )},
        "target_sample_sha256": "1" * 64,
        "query_sampling_sha256": "2" * 64,
    }
    cache_file = tmp_path / "wind-role-scales.json"
    cache_file.write_text(
        json.dumps(_calibration_cache_payload(cache_key, calibration)),
        encoding="utf-8",
    )
    assert _read_role_scale_cache(cache_file, cache_key) == calibration
    changed_source = {**cache_key, "training_fingerprint": "different-target-source"}
    assert _read_role_scale_cache(cache_file, changed_source) is None

    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    payload["calibration"]["component_role_scales_mps"]["volume"][0] = 100.0
    cache_file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        _read_role_scale_cache(cache_file, cache_key)


def test_wind_formal_transform_cache_binds_full_train_recipe_without_test_targets(tmp_path) -> None:
    normalizer, profile = _transforms()
    rows = list(range(72))
    rows_sha = hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()
    cache_key = {
        "dataset_profile": "wind_formal_fulltrain_v1",
        "train_row_indices": rows,
        "train_row_indices_sha256": rows_sha,
        "samples_per_row": 8192,
        "profile_bins": 8,
        "seed": 42,
        "wind_test_target_values_read": False,
        "target_partition_access": "all selected original TRAIN rows only",
        "solver_attempts": 0,
    }
    cache_file = tmp_path / "formal-transforms.json"
    cache_file.write_text(
        json.dumps(_transform_cache_payload(cache_key, normalizer, profile)), encoding="utf-8"
    )
    loaded = _read_transform_cache(cache_file, cache_key)
    assert loaded is not None
    assert loaded[0].source_rows == 72
    assert loaded[1].values_mps.shape == (8, 3)
    changed_rows = {**cache_key, "train_row_indices": list(range(71)) + [72]}
    assert _read_transform_cache(cache_file, changed_rows) is None


def test_wind_fulltrain_transform_fit_cache_reuses_only_exact_source_membership(tmp_path, monkeypatch) -> None:
    from windfarm.training import joint_task

    volume_root = tmp_path / "family_volume"
    volume_root.mkdir()
    array_names = (
        "U.npy", "run_cell_offsets.npy", "run_shape.npy", "run_x_offsets.npy", "run_y_offsets.npy",
        "run_z_offsets.npy", "x_cell_m.npy", "y_cell_m.npy", "z_cell_m.npy", "case.npy",
        "layout_index.npy", "wd_deg.npy", "completed.npy",
    )
    for name in array_names:
        (volume_root / name).write_bytes(b"source")
    metadata = {
        "case": np.asarray([f"case-{row}" for row in range(72)]),
        "layout_index": np.repeat(np.arange(24, dtype=np.int64), 3),
        "wd_deg": np.tile(np.asarray([270.0, 285.0, 300.0]), 24),
        "n_turbines": np.full(72, 12, dtype=np.int16),
        "turbine_xy_D": np.zeros((72, 30, 2), dtype=np.float32),
        "U_ref": np.asarray(9.0),
        "D_m": np.asarray(80.0),
        "hub_height_m": np.asarray(70.0),
    }
    view = SimpleNamespace(volume=SimpleNamespace(volume_root=volume_root), metadata=metadata)
    calls = []

    def fake_fit(_dataset, rows, *, samples_per_row, seed, profile_bins):
        calls.append((tuple(map(int, rows)), samples_per_row, seed, profile_bins))
        normalizer = VelocityNormalizer(
            mean=np.asarray([0.3, -0.1, 0.0]),
            std=np.asarray([0.4, 0.3, 0.2]),
            safe_std=np.asarray([0.4, 0.3, 0.2]),
            u_ref_mps=9.0,
            sample_count_per_row=samples_per_row,
            source_rows=len(rows),
            seed=seed,
        )
        centers = np.linspace(0.0, 6.25, profile_bins, dtype=np.float64)
        values = np.column_stack((8.0 + centers * 0.1, centers * 0.01, centers * 0.02))
        profile = VerticalProfileBaseline(centers, values, np.full(profile_bins, 10, dtype=np.int64))
        return normalizer, profile

    monkeypatch.setattr(joint_task, "fit_velocity_statistics", fake_fit)
    rows = np.arange(72, dtype=np.int64)
    manifest = {"manifest_sha256": "manual-fulltrain-manifest-sha"}
    first = _fit_or_load_formal_transforms(
        view=view,
        train_rows=rows,
        train_layouts=np.arange(24, dtype=np.int64),
        manifest=manifest,
        training_fingerprint="manual-fulltrain-fingerprint",
        seed=42,
        cache_dir=tmp_path / "cache",
    )
    second = _fit_or_load_formal_transforms(
        view=view,
        train_rows=rows,
        train_layouts=np.arange(24, dtype=np.int64),
        manifest=manifest,
        training_fingerprint="manual-fulltrain-fingerprint",
        seed=42,
        cache_dir=tmp_path / "cache",
    )
    assert len(calls) == 1
    assert first[2] == second[2]
    assert second[2]["wind_test_target_values_read"] is False
    assert second[0].source_rows == 72

    changed_rows = rows.copy()
    changed_rows[-1] = 73
    _fit_or_load_formal_transforms(
        view=view,
        train_rows=changed_rows,
        train_layouts=np.arange(24, dtype=np.int64),
        manifest=manifest,
        training_fingerprint="manual-fulltrain-fingerprint",
        seed=42,
        cache_dir=tmp_path / "cache",
    )
    os.utime(volume_root / "U.npy", ns=(1_790_000_000_000_000_000, 1_790_000_000_000_000_001))
    _fit_or_load_formal_transforms(
        view=view,
        train_rows=rows,
        train_layouts=np.arange(24, dtype=np.int64),
        manifest=manifest,
        training_fingerprint="manual-fulltrain-fingerprint",
        seed=42,
        cache_dir=tmp_path / "cache",
    )
    assert len(calls) == 3


def test_joint_cli_rejects_manual_formal_start_without_explicit_launch_flag(tmp_path, capsys) -> None:
    cli = _joint_cli_module()
    recipe_path = tmp_path / "manual-wind-formal.json"
    recipe_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "wind",
                "mode": "J-H",
                "seed": 42,
                "hidden": 128,
                "message": 128,
                "regional_anchors": 32,
                "depth": 2,
                "receiver_tile": 512,
                "primary_queries": 4096,
                "microbatch_cases": 24,
                "effective_cases": 24,
                "formal_full": True,
                "total_epochs": 5000,
                "initialization": "fresh_all_trainable",
                "launch_policy": "manual_only",
                "dataset_protocol": "original420_train",
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "must-not-start"
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "start",
                "--recipe-json",
                str(recipe_path),
                "--output-dir",
                str(output_dir),
                "--stop-after",
                "100",
            ]
        )
    assert error.value.code == 2
    assert "Formal optimization requires explicit --manual-formal-launch" in capsys.readouterr().err
    assert not output_dir.exists()

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from honf_forward_core.content_hypergraph import ContentHypergraphCore

from windfarm.content_hypergraph import (
    WIND_CONTENT_COORDINATE_FRAME,
    WIND_CONTENT_ENVIRONMENT_COUNT,
    WindContentHypergraphAdapter,
    WindContentReceivers,
    WindContentSceneInputs,
)
from windfarm.normalization import VelocityNormalizer, VerticalProfileBaseline
from windfarm.training.content_hypergraph_task import (
    _WIND_CONTENT_CATALOGUE_CACHES,
    WIND_CONTENT_ROLE_CACHE_DEVELOPMENT_BYTES,
    WIND_CONTENT_ROLE_CACHE_FULL_BYTES,
    WindContentHypergraphTask,
    WindContentPredictions,
    WindContentTargets,
    _role_counts_for_query_count,
    _role_ids,
    _shared_read_only_catalogue_cache,
    _validate_recipe,
)
from windfarm.workflows.joint_forward import ROLE_NAMES, NativeRoleCatalogueCache


def _transforms() -> tuple[VelocityNormalizer, VerticalProfileBaseline]:
    normalizer = VelocityNormalizer(
        mean=np.asarray([0.25, -0.5, 0.1]),
        std=np.asarray([0.3, 0.2, 0.4]),
        safe_std=np.asarray([0.3, 0.2, 0.4]),
        u_ref_mps=9.0,
        source_rows=72,
        seed=42,
    )
    profile = VerticalProfileBaseline(
        bin_centers_D=np.asarray([0.0, 1.0, 2.0]),
        values_mps=np.asarray([[2.0, 0.0, 1.0], [4.0, 1.0, 2.0], [6.0, 2.0, 3.0]]),
        counts=np.asarray([10, 10, 10]),
    )
    return normalizer, profile


def _scene_case():
    env_coords = np.column_stack((np.arange(64), np.arange(64) * 0.1, np.arange(64) * 0.01))
    # Keep feature values recognizable so the adapter's literal input binding
    # can be checked without relying on model outputs.
    env_features = np.arange(64 * 7, dtype=np.float32).reshape(64, 7)
    env_weights = np.linspace(0.01, 0.64, 64, dtype=np.float64)
    return SimpleNamespace(
        index=5,
        case="wind-layout-2-wd285",
        layout_index=2,
        wind_direction_deg=285.0,
        module_centers=np.asarray([[0.0, 0.0, 0.875], [10.0, 2.0, 0.875]], dtype=np.float32),
        module_present=np.asarray([1.0, 0.0], dtype=np.float32),
        module_features=np.asarray([[0.5, 0.875], [0.5, 0.875]], dtype=np.float32),
        global_context=np.asarray([0.0, 1.0, 0.0, 2.0 / 30.0, 1.0, -1.0, -1.0, 0.0, 4.0, 2.0, 2.0]),
        support=SimpleNamespace(
            lower_D=np.asarray([-1.0, -2.0, 0.0]),
            upper_D=np.asarray([4.0, 3.0, 2.0]),
            extent_D=np.asarray([5.0, 5.0, 2.0]),
        ),
        environment=SimpleNamespace(
            token_shape=(4, 4, 4),
            coords_D=env_coords.astype(np.float32),
            features=env_features,
            weights_D3=env_weights,
        ),
        # Scene construction must never inspect this target-bearing object.
        run=_TargetReadForbidden(),
    )


def _native_query_fixture(adapter, inputs, coordinates_D):
    coordinates = np.asarray(coordinates_D, dtype=np.float32)
    query_indices, module_indices = adapter.packed_local_pair_indices(inputs, coordinates)
    support = SimpleNamespace(
        lower_D=inputs.support_lower_D,
        upper_D=inputs.support_upper_D,
        extent_D=inputs.support_extent_D,
    )
    receivers = WindContentReceivers(
        coordinates_D=coordinates[None],
        features=adapter.query_features(coordinates, support)[None],
        flat_indices=np.arange(100, 100 + coordinates.shape[0], dtype=np.int64)[None],
        role_ids=np.zeros((1, coordinates.shape[0]), dtype=np.int8),
        scene_keys=(inputs.scene_key,),
        local_batch_indices=np.zeros(query_indices.size, dtype=np.int64),
        local_query_indices=query_indices,
        local_module_indices=module_indices,
    )
    return receivers


def _small_content_core():
    return ContentHypergraphCore(
        module_input_dim=2,
        environment_input_dim=7,
        coordinate_dim=3,
        query_input_dim=7,
        operating_input_dim=11,
        output_dim=3,
        architecture="A-H6",
        context_dim=16,
        num_heads=4,
        context_depth=1,
        num_hyperedges=6,
        thermal_response_rank=0,
        local_hidden_dim=8,
        init_seed=42,
    ).eval()


class _TargetReadForbidden:
    @property
    def U(self):
        raise AssertionError("WindContentSceneInputs.from_case must not read native velocity targets")


def test_native_scene_preserves_geometry_measures_and_literal_ids_without_target_reads():
    case = _scene_case()
    inputs = WindContentSceneInputs.from_case(case)
    normalizer, profile = _transforms()
    adapter = WindContentHypergraphAdapter(normalizer, profile)
    scene = adapter.make_scene_batch((inputs,))

    assert tuple(scene.environment_features.shape) == (1, WIND_CONTENT_ENVIRONMENT_COUNT, 7)
    assert tuple(scene.module_features.shape) == (1, 2, 2)
    assert scene.module_ids.tolist() == [[0, 1]]
    assert scene.environment_ids.tolist() == [list(range(64))]
    assert scene.module_mask.tolist() == [[True, False]]
    np.testing.assert_allclose(
        scene.environment_measure.cpu().numpy()[0],
        case.environment.weights_D3.astype(np.float32),
        rtol=0,
        atol=0,
    )
    np.testing.assert_array_equal(scene.environment_coordinates.cpu().numpy()[0], case.environment.coords_D)
    np.testing.assert_array_equal(scene.environment_features.cpu().numpy()[0], case.environment.features)
    np.testing.assert_array_equal(scene.module_coordinates.cpu().numpy()[0], case.module_centers)
    assert torch.all(scene.local_radius == 1.5)
    assert scene.coordinate_frame_key == WIND_CONTENT_COORDINATE_FRAME
    assert scene.scene_key == (inputs.scene_key,)


def test_query_batch_keeps_native_ids_and_packs_only_1p5d_local_rows():
    case = _scene_case()
    inputs = WindContentSceneInputs.from_case(case)
    normalizer, profile = _transforms()
    adapter = WindContentHypergraphAdapter(normalizer, profile)
    scene = adapter.make_scene_batch((inputs,))
    coords = np.asarray([[0.0, 0.0, 0.875], [1.5, 0.0, 0.875], [1.49, 0.0, 0.875]], dtype=np.float32)
    query_indices, module_indices = adapter.packed_local_pair_indices(inputs, coords)
    receivers = WindContentReceivers(
        coordinates_D=coords[None],
        features=np.zeros((1, 3, 7), dtype=np.float32),
        flat_indices=np.asarray([[17, 23, 30]], dtype=np.int64),
        role_ids=np.asarray([[0, 1, 2]], dtype=np.int8),
        scene_keys=(inputs.scene_key,),
        local_batch_indices=np.zeros(query_indices.size, dtype=np.int64),
        local_query_indices=query_indices,
        local_module_indices=module_indices,
    )
    query = adapter.make_query_batch(scene, receivers)
    query.validate(batch_size=1, coordinate_dim=3, query_input_dim=7)

    assert query.query_ids.tolist() == [[17, 23, 30]]
    np.testing.assert_array_equal(query.coordinates.cpu().numpy()[0], coords)
    assert query.local_batch_indices.tolist() == [0, 0]
    assert query.local_query_indices.tolist() == [0, 2]
    assert query.local_module_indices.tolist() == [0, 0]
    ratios = torch.tensor([0.0, 1.49 / 1.5])
    expected = (1.0 - ratios).pow(4) * (4.0 * ratios + 1.0)
    torch.testing.assert_close(query.local_weights, expected)


def test_native_adapter_drives_shared_core_and_returns_physical_profile_residual():
    case = _scene_case()
    inputs = WindContentSceneInputs.from_case(case)
    normalizer, profile = _transforms()
    adapter = WindContentHypergraphAdapter(normalizer, profile)
    scene = adapter.make_scene_batch((inputs,))
    coords = np.asarray([[0.0, 0.0, 0.875], [1.49, 0.0, 0.875]], dtype=np.float32)
    query_indices, module_indices = adapter.packed_local_pair_indices(inputs, coords)
    receivers = WindContentReceivers(
        coordinates_D=coords[None],
        features=np.zeros((1, 2, 7), dtype=np.float32),
        flat_indices=np.asarray([[17, 30]], dtype=np.int64),
        role_ids=np.asarray([[0, 1]], dtype=np.int8),
        scene_keys=(inputs.scene_key,),
        local_batch_indices=np.zeros(query_indices.size, dtype=np.int64),
        local_query_indices=query_indices,
        local_module_indices=module_indices,
    )
    task = object.__new__(WindContentHypergraphTask)
    task.adapter = adapter
    model = ContentHypergraphCore(
        module_input_dim=2,
        environment_input_dim=7,
        coordinate_dim=3,
        query_input_dim=7,
        operating_input_dim=11,
        output_dim=3,
        architecture="A-H6",
        context_dim=16,
        num_heads=4,
        context_depth=1,
        num_hyperedges=6,
        thermal_response_rank=0,
        local_hidden_dim=8,
        init_seed=42,
    ).eval()
    predictions, auxiliary = task.predict_native(model, scene, receivers, "A-H6")

    assert predictions.main_mps.shape == (1, 2, 3)
    assert auxiliary["prepared_hyperedge_count"] == 6
    assert auxiliary["core_output"].thermal_temperature is None
    torch.testing.assert_close(
        predictions.main_mps,
        predictions.profile_mps + predictions.residual_mps,
    )


def test_adapter_state_roundtrip_and_profile_residual_inverse_are_self_contained():
    normalizer, profile = _transforms()
    adapter = WindContentHypergraphAdapter(normalizer, profile)
    restored = WindContentHypergraphAdapter.from_state_payload(adapter.state_payload())
    assert restored.state_payload() == adapter.state_payload()

    normalized = torch.tensor([[[0.5, -1.0, 0.25], [0.0, 0.0, 0.0]]], dtype=torch.float64)
    coords = torch.tensor([[[0.5, 0.0, 0.5], [0.0, 0.0, 3.0]]], dtype=torch.float64)
    physical, baseline, residual = restored.physical_velocity(normalized, coords)
    expected_residual = (
        normalized * torch.tensor(normalizer.safe_std, dtype=torch.float64)
        + torch.tensor(normalizer.mean, dtype=torch.float64)
    ) * normalizer.u_ref_mps
    expected_profile = torch.tensor([[[3.0, 0.5, 1.5], [6.0, 2.0, 3.0]]], dtype=torch.float64)
    torch.testing.assert_close(residual, expected_residual)
    torch.testing.assert_close(baseline, expected_profile)
    torch.testing.assert_close(physical, expected_profile + expected_residual)
    coords.requires_grad_(True)
    restored.profile_at(coords).sum().backward()
    assert coords.grad is not None and torch.isfinite(coords.grad).all()


def test_bound_transform_arrays_reject_direct_and_forced_mutation():
    normalizer, profile = _transforms()
    adapter = WindContentHypergraphAdapter(normalizer, profile)
    with pytest.raises(ValueError):
        adapter.normalizer.mean[0] = 1.0
    adapter.normalizer.mean.setflags(write=True)
    adapter.normalizer.mean[0] += 1.0
    with pytest.raises(RuntimeError, match="normalizer array 'mean' changed"):
        adapter.state_payload()

    profile_adapter = WindContentHypergraphAdapter(*_transforms())
    profile_adapter.background_profile.values_mps.setflags(write=True)
    profile_adapter.background_profile.values_mps[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="background-profile array 'values_mps' changed"):
        profile_adapter.physical_velocity(
            torch.zeros((1, 1, 3)),
            torch.zeros((1, 1, 3)),
        )


def test_native_source_xy_geometry_vjp_matches_two_central_fd_steps():
    """Use fixed Eulerian receivers and support while moving source x/y only."""

    inputs = WindContentSceneInputs.from_case(_scene_case())
    normalizer, profile = _transforms()
    adapter = WindContentHypergraphAdapter(normalizer, profile)
    scene = adapter.make_scene_batch((inputs,))
    # These receiver coordinates stay fixed in the native Wind frame across
    # all source perturbations. They are interior to the unchanged support box.
    coordinates = np.asarray(
        [[0.35, 0.20, 0.875], [0.65, -0.25, 0.875], [1.05, 0.30, 0.875]],
        dtype=np.float32,
    )
    receivers = _native_query_fixture(adapter, inputs, coordinates)
    model = _small_content_core()
    objective_weights = torch.tensor(
        [[[0.2, -0.4, 0.7], [-0.3, 0.8, 0.1], [0.5, 0.2, -0.6]]],
        dtype=torch.float32,
    )

    source_xy = scene.module_coordinates.detach().clone()
    source_xy.requires_grad_(True)
    differentiable_scene = replace(scene, module_coordinates=source_xy)
    query = adapter.make_query_batch(differentiable_scene, receivers)
    organization = model.prepare(differentiable_scene)
    output = model.decode(organization, query)
    physical, profile_at_q, _ = adapter.physical_velocity(output.nonlinear_field, query.coordinates)
    objective = torch.sum(physical * objective_weights)
    gradient = torch.autograd.grad(objective, source_xy)[0][0, 0, :2]
    assert torch.isfinite(gradient).all()
    assert torch.linalg.vector_norm(gradient) > 1.0e-7
    assert torch.all(query.coordinates[..., 2] == 0.875)
    assert torch.all(profile_at_q == adapter.profile_at(query.coordinates))

    direction = torch.tensor([0.6, -0.8], dtype=torch.float32)

    def evaluate(displacement_xy):
        moved = scene.module_coordinates.detach().clone()
        moved[0, 0, :2] += torch.as_tensor(displacement_xy, dtype=moved.dtype)
        moved_scene = replace(scene, module_coordinates=moved)
        moved_query = adapter.make_query_batch(moved_scene, receivers)
        moved_output = model.decode(model.prepare(moved_scene), moved_query)
        moved_physical, moved_profile, _ = adapter.physical_velocity(
            moved_output.nonlinear_field,
            moved_query.coordinates,
        )
        torch.testing.assert_close(moved_query.coordinates, query.coordinates, rtol=0.0, atol=0.0)
        torch.testing.assert_close(moved_profile, profile_at_q, rtol=0.0, atol=0.0)
        return torch.sum(moved_physical * objective_weights)

    vjp_directional = torch.dot(gradient, direction)
    for step in (2.0e-2, 1.0e-2):
        delta = direction * step
        finite_difference = (evaluate(delta) - evaluate(-delta)) / (2.0 * step)
        torch.testing.assert_close(finite_difference, vjp_directional, rtol=5.0e-2, atol=2.0e-4)


def test_native_query_subset_and_source_permutation_preserve_receiver_predictions():
    inputs = WindContentSceneInputs.from_case(_scene_case())
    normalizer, profile = _transforms()
    adapter = WindContentHypergraphAdapter(normalizer, profile)
    scene = adapter.make_scene_batch((inputs,))
    coordinates = np.asarray(
        [[0.35, 0.20, 0.875], [0.65, -0.25, 0.875], [1.05, 0.30, 0.875], [2.0, 0.0, 0.875]],
        dtype=np.float32,
    )
    receivers = _native_query_fixture(adapter, inputs, coordinates)
    query = adapter.make_query_batch(scene, receivers)
    model = _small_content_core()
    organization = model.prepare(scene)
    full = model.decode(organization, query).nonlinear_field

    subset_rows = np.asarray([0, 2], dtype=np.int64)
    subset_coordinates = coordinates[subset_rows]
    subset_receivers = _native_query_fixture(adapter, inputs, subset_coordinates)
    subset_query = adapter.make_query_batch(scene, subset_receivers)
    subset = model.decode(organization, subset_query).nonlinear_field
    torch.testing.assert_close(subset, full[:, subset_rows], rtol=1.0e-6, atol=1.0e-6)

    permutation = np.asarray([1, 0], dtype=np.int64)
    permuted_inputs = replace(
        inputs,
        module_centers_D=inputs.module_centers_D[permutation].copy(),
        module_present=inputs.module_present[permutation].copy(),
        module_features=inputs.module_features[permutation].copy(),
        module_ids=inputs.module_ids[permutation].copy(),
    )
    permuted_scene = adapter.make_scene_batch((permuted_inputs,))
    permuted_receivers = _native_query_fixture(adapter, permuted_inputs, coordinates)
    permuted_query = adapter.make_query_batch(permuted_scene, permuted_receivers)
    permuted = model.decode(model.prepare(permuted_scene), permuted_query).nonlinear_field
    torch.testing.assert_close(permuted, full, rtol=2.0e-5, atol=2.0e-6)


@pytest.mark.parametrize("geometry_field", ["module_coordinates", "environment_coordinates"])
def test_native_scene_geometry_mutation_invalidates_prepared_organization(geometry_field):
    inputs = WindContentSceneInputs.from_case(_scene_case())
    normalizer, profile = _transforms()
    adapter = WindContentHypergraphAdapter(normalizer, profile)
    scene = adapter.make_scene_batch((inputs,))
    receivers = _native_query_fixture(adapter, inputs, np.asarray([[0.35, 0.20, 0.875]], dtype=np.float32))
    query = adapter.make_query_batch(scene, receivers)
    model = _small_content_core()
    organization = model.prepare(scene)
    with torch.no_grad():
        getattr(scene, geometry_field)[0, 0, 0].add_(0.01)
    with pytest.raises(ValueError, match="Scene tensors changed after preparation"):
        model.decode(organization, query)


def test_role_counts_and_physical_metrics_keep_equal_role_objective_and_per_role_rl2():
    counts = _role_counts_for_query_count(4096)
    assert sum(counts.values()) == 4096
    assert tuple(counts) == ROLE_NAMES
    role_ids = _role_ids(counts)
    assert role_ids.size == 4096
    assert {name: int(np.count_nonzero(role_ids == index)) for index, name in enumerate(ROLE_NAMES)} == counts

    task = object.__new__(WindContentHypergraphTask)
    task.role_component_scales = {role: np.asarray([1.0, 2.0, 4.0]) for role in ROLE_NAMES}
    task.validation_rows = np.asarray([5], dtype=np.int64)
    q = 10
    roles = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4]])
    actual = np.full((1, q, 3), 2.0, dtype=np.float32)
    target = WindContentTargets(actual, (5,), ({"row_index": 5, "layout_index": 2, "case": "fixture"},))
    prediction = WindContentPredictions(
        main_mps=torch.ones((1, q, 3)),
        normalized_residual=torch.zeros((1, q, 3)),
        residual_mps=torch.zeros((1, q, 3)),
        profile_mps=torch.zeros((1, q, 3)),
        role_ids=roles,
    )
    terms = task._loss_terms(prediction, target)
    expected_role_numerator = 2.0 * (1.0 + 0.25 + 0.0625)
    for role in ROLE_NAMES:
        assert float(terms[f"native_role/{role}"].numerator) == pytest.approx(expected_role_numerator)
        assert terms[f"native_role/{role}"].denominator == 6.0
    metrics = task.validation_metrics(prediction, target, {})
    row = metrics["rows"][0]
    assert row["roles"]["volume"]["relative_l2"] == pytest.approx(0.5)
    assert row["roles"]["volume"]["height_profile_relative_l2"] == pytest.approx(1.0)
    expected_score = np.mean((np.asarray([1.0, 1.0, 1.0]) ** 2) / np.asarray([1.0, 4.0, 16.0]))
    assert row["field_score"] == expected_score
    reduced = task.reduce_native_metrics((metrics,))
    assert reduced["row_count"] == 1 and reduced["field_score"] == expected_score


def test_recipe_resolves_dataset_default_thermal_rank_and_protocol_schedule():
    base = {
        "schema": "content_hypergraph_recipe_v1",
        "dataset": "wind",
        "architecture": "A-H6",
        "seed": 42,
        "primary_queries": 4096,
        "microbatch_size": 8,
        "effective_batch_size": 24,
        "model": {"thermal_response_rank": 4},
    }
    development, architecture, _ = _validate_recipe(
        {
            **base,
            "protocol": "development",
            "total_epochs": 2500,
        }
    )
    assert architecture == "A-H6" and development["model"]["thermal_response_rank"] == 4
    assert development["catalogue_cache_ram_bytes"] == 2 * 1024**3
    development_explicit, _, _ = _validate_recipe(
        {
            **base,
            "protocol": "development",
            "total_epochs": 2500,
            "catalogue_cache_ram_bytes": WIND_CONTENT_ROLE_CACHE_DEVELOPMENT_BYTES,
        }
    )
    assert development_explicit["catalogue_cache_ram_bytes"] == WIND_CONTENT_ROLE_CACHE_DEVELOPMENT_BYTES
    dev_task = object.__new__(WindContentHypergraphTask)
    dev_task.recipe = {"protocol": "development"}
    dev_task.total_epochs = 2500
    assert dev_task._schedule().hold_through_epoch == 1000

    full_recipe, _, _ = _validate_recipe(
        {
            **base,
            "protocol": "full_train1000",
            "total_epochs": 5000,
            "approved_stop_after": 1000,
        }
    )
    assert full_recipe["protocol"] == "full_train1000"
    assert full_recipe["catalogue_cache_ram_bytes"] == 2 * 1024**3
    full_explicit, _, _ = _validate_recipe(
        {
            **base,
            "protocol": "full_train1000",
            "total_epochs": 5000,
            "approved_stop_after": 1000,
            "catalogue_cache_ram_bytes": WIND_CONTENT_ROLE_CACHE_FULL_BYTES,
        }
    )
    assert full_explicit["catalogue_cache_ram_bytes"] == WIND_CONTENT_ROLE_CACHE_FULL_BYTES
    full_task = object.__new__(WindContentHypergraphTask)
    full_task.recipe = {"protocol": "full_train1000"}
    full_task.total_epochs = 5000
    assert full_task._schedule().hold_through_epoch == 2000
    full_task.recipe["schedule"] = {
        "peak_lr": 3.0e-4,
        "warmup_start_lr": 3.0e-5,
        "warmup_epochs": 20,
        "hold_through_epoch": 1000,
        "total_epochs": 5000,
        "final_lr": 3.0e-6,
    }
    with pytest.raises(ValueError, match="hold through epoch 2000"):
        full_task._schedule()


@pytest.mark.parametrize("limit", [1, 2 * 1024**3, 32 * 1024**3, 128 * 1024**3])
def test_catalogue_cache_capacity_is_recipe_bound_and_reuses_same_read_only_store(monkeypatch, limit):
    from windfarm.training import content_hypergraph_task as task_module

    _WIND_CONTENT_CATALOGUE_CACHES.clear()
    monkeypatch.setattr(task_module, "training_catalogue_cache_directory", lambda: None)
    first = _shared_read_only_catalogue_cache(limit)
    assert _shared_read_only_catalogue_cache(limit) is first
    assert first.max_cached_bytes == limit
    assert first.persistent_dir is None and first.read_only_persistent
    task = object.__new__(WindContentHypergraphTask)
    task.catalogue_cache = first
    identity = task._catalogue_cache_identity()
    assert identity["max_ram_bytes"] == limit
    assert first.disk_write_count == 0


@pytest.mark.parametrize("limit", [0, -1, True, 128 * 1024**3 + 1, 1.5])
def test_catalogue_cache_capacity_rejects_invalid_recipe_values(limit):
    recipe = {
        "schema": "content_hypergraph_recipe_v1",
        "dataset": "wind",
        "architecture": "A-H6",
        "protocol": "development",
        "seed": 42,
        "primary_queries": 4096,
        "microbatch_size": 8,
        "effective_batch_size": 24,
        "total_epochs": 2500,
        "catalogue_cache_ram_bytes": limit,
    }
    with pytest.raises(ValueError, match="catalogue_cache_ram_bytes"):
        _validate_recipe(recipe)


def test_native_endpoint_rows_are_input_selected_and_use_fixed_q8192_streams():
    task = object.__new__(WindContentHypergraphTask)
    task.train_rows = np.asarray([6, 2, 4, 8], dtype=np.int64)
    task.validation_rows = np.asarray([1, 3, 5], dtype=np.int64)
    task.view = SimpleNamespace(metadata={"n_turbines": np.asarray([0, 1, 0, 2, 3, 2, 5, 0, 5])})
    task.max_microbatch_cases = 2
    task.seed = 42
    task.sampling_version = "case_epoch_v1"
    task.sampling_dataset_id = "wind:test-membership"
    calls = []

    def record_batch(rows, key, **kwargs):
        calls.append((tuple(int(value) for value in rows), key, kwargs))
        return calls[-1]

    task._make_batch = record_batch
    assert task.native_evaluation_rows("train") == (2, 8)
    assert task.native_evaluation_rows("train", [6]) == (6,)
    assert task.native_evaluation_rows("validation") == (1, 3, 5)
    with pytest.raises(ValueError, match="members of the sealed split"):
        task.native_evaluation_rows("train", [1])

    train_batches = list(task.native_evaluation_batches("train", microbatch_size=2))
    assert len(train_batches) == 1
    assert train_batches[0][0] == (2, 8)
    assert train_batches[0][2] == {
        "validation": False,
        "stream_validation": True,
        "query_count": 8192,
        "track_samples": False,
    }
    assert train_batches[0][1].sampling_version == "case_epoch_v1"

    calls.clear()
    dev_batches = list(task.native_evaluation_batches("validation", microbatch_size=1))
    assert [batch[0] for batch in dev_batches] == [(1,), (3,), (5,)]
    assert all(batch[2]["validation"] and batch[2]["query_count"] == 8192 for batch in dev_batches)


def _small_native_case():
    return SimpleNamespace(
        index=7,
        layout_index=7,
        diameter_m=2.0,
        hub_height_m=2.0,
        module_centers=np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
        module_present=np.ones(1, dtype=np.float32),
        run=SimpleNamespace(
            nx=4,
            ny=3,
            nz=3,
            cell_count=36,
            x_m=np.asarray([0.0, 2.0, 4.0, 6.0]),
            y_m=np.asarray([0.0, 2.0, 4.0]),
            z_m=np.asarray([1.0, 3.0, 9.0]),
            U=np.arange(108, dtype=np.float32).reshape(36, 3),
        ),
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    import hashlib

    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_readonly_role_cache_reuses_exact_entry_and_keeps_cold_miss_in_ram(tmp_path):
    case = _small_native_case()
    writable = NativeRoleCatalogueCache(persistent_dir=tmp_path)
    reference = writable.get(case)
    before = _tree_hashes(tmp_path)

    readonly = NativeRoleCatalogueCache(persistent_dir=tmp_path, read_only_persistent=True)
    reused = readonly.get(case)
    assert readonly.disk_hit_count == 1 and readonly.build_count == 0
    np.testing.assert_array_equal(reused.coordinates_D, reference.coordinates_D)

    cold_case = _small_native_case()
    cold_case.index = 8
    cold_case.layout_index = 8
    cold = NativeRoleCatalogueCache(persistent_dir=tmp_path, read_only_persistent=True)
    built = cold.get(cold_case)
    assert built.geometry_sha256 != reference.geometry_sha256
    assert cold.build_count == 1 and cold.disk_write_count == 0
    assert cold.summary()["persistent_read_only"] is True
    assert _tree_hashes(tmp_path) == before

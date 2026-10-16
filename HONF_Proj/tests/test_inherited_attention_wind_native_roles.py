from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / "src", PROJECT / "Case_WindFarm/src"):
    sys.path.insert(0, str(directory))

from windfarm.content_hypergraph import (
    WIND_CONTENT_COORDINATE_FRAME,
    WindContentReceivers,
    WindContentSceneInputs,
    WindInheritedAttentionAdapterV1,
)
from windfarm.normalization import VelocityNormalizer, VerticalProfileBaseline
from windfarm.training import inherited_attention_task as iag_wind
from windfarm.training.content_hypergraph_task import (
    ROLE_NAMES,
    WindContentHypergraphTask,
    WindContentTargets,
    WindFormalA2DTaskV1,
)

ROLE_COUNTS = (1640, 1640, 1640, 1640, 1632)
ROLE_IDS = np.repeat(np.arange(5, dtype=np.int8), ROLE_COUNTS)


def _row533_adapter_fixture(query_y: np.float32 | None = None):
    lower = np.asarray(
        [-17.85594367980957, -14.279596328735352, 0.022507892921566963], dtype=np.float32,
    )
    upper = np.asarray(
        [28.7056827545166, 14.740605354309082, 6.159968376159668], dtype=np.float32,
    )
    extent = np.asarray([46.56162643432617, 29.020200729370117, 6.137460708618164], dtype=np.float32)
    axes = [np.linspace(lo, hi, 4, dtype=np.float32) for lo, hi in zip(lower, upper)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    environment_coordinates = np.stack((xx, yy, zz), axis=-1).reshape(64, 3)
    scene_input = WindContentSceneInputs(
        row_index=533,
        case_name="gen_0177_wd300",
        layout_index=177,
        wind_direction_deg=300.0,
        module_centers_D=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        module_present=np.asarray([1.0], dtype=np.float32),
        module_features=np.asarray([[0.02, 0.1]], dtype=np.float32),
        module_ids=np.asarray([0], dtype=np.int64),
        operating_features=np.zeros(11, dtype=np.float32),
        support_lower_D=lower,
        support_upper_D=upper,
        support_extent_D=extent,
        environment_coordinates_D=environment_coordinates,
        environment_features=np.zeros((64, 7), dtype=np.float32),
        environment_measure_D3=np.ones(64, dtype=np.float32),
    )
    normalizer = VelocityNormalizer(
        mean=np.zeros(3), std=np.ones(3), safe_std=np.ones(3),
    )
    profile = VerticalProfileBaseline(
        bin_centers_D=np.asarray([0.0, 6.25]),
        values_mps=np.zeros((2, 3)),
        counts=np.ones(2, dtype=np.int64),
    )
    adapter = WindInheritedAttentionAdapterV1(normalizer, profile)
    scene = adapter.make_scene_batch((scene_input,))
    coordinates = upper[None, None, :].copy()
    if query_y is not None:
        coordinates[0, 0, 1] = query_y
    receivers = WindContentReceivers(
        coordinates_D=coordinates,
        features=np.zeros((1, 1, 7), dtype=np.float32),
        flat_indices=np.zeros((1, 1), dtype=np.int64),
        role_ids=np.zeros((1, 1), dtype=np.int8),
        scene_keys=(scene_input.scene_key,),
        local_batch_indices=np.empty(0, dtype=np.int64),
        local_query_indices=np.empty(0, dtype=np.int64),
        local_module_indices=np.empty(0, dtype=np.int64),
    )
    return adapter, scene, receivers, lower, upper


def _empty_provider() -> iag_wind.WindInheritedAttentionTaskV1:
    return object.__new__(iag_wind.WindInheritedAttentionTaskV1)


def test_prediction_consumes_the_scene_prepared_by_the_training_engine():
    from honf_forward_core.content_hypergraph.contracts import QueryBatch, SceneBatch
    from honf_forward_core.inherited_attention import (
        InheritedAttentionCoreV1,
        smoothstep_grid_receiver_cover,
        with_receiver_cover,
    )

    provider = _empty_provider()
    anchors = torch.cartesian_prod(*(torch.tensor([0.0, 1.0]) for _ in range(3)))[None]
    scene = SceneBatch(
        module_features=torch.ones(1, 2, 2), environment_features=torch.ones(1, 8, 7),
        module_coordinates=torch.tensor([[[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]]]),
        environment_coordinates=anchors,
        module_mask=torch.ones(1, 2, dtype=torch.bool), environment_mask=torch.ones(1, 8, dtype=torch.bool),
        module_measure=torch.ones(1, 2), environment_measure=torch.ones(1, 8),
        module_ids=torch.tensor([[13, 29]]), environment_ids=torch.arange(8)[None],
        local_radius=torch.ones(1, 2), operating_features=torch.ones(1, 11),
        coordinate_origin=torch.zeros(1, 3), coordinate_scale=torch.ones(1, 3),
        module_measure_reference=torch.ones(1), environment_measure_reference=torch.ones(1),
        scene_key="wind-engine-contract", normalizer_key="test", coordinate_frame_key="native",
        module_characteristic_lengths=torch.full((1, 2, 3), 0.2),
        environment_characteristic_lengths=torch.ones(1, 8, 3),
    )
    base = QueryBatch(
        coordinates=torch.tensor([[[0.3, 0.4, 0.5], [0.7, 0.6, 0.5]]]),
        features=torch.ones(1, 2, 7), query_mask=torch.ones(1, 2, dtype=torch.bool),
        query_ids=torch.tensor([[41, 43]]), scene_key=scene.scene_key,
        coordinate_frame_key=scene.coordinate_frame_key, receiver_key="test-query",
    )
    cover = smoothstep_grid_receiver_cover(
        base.coordinates, anchors, scene.environment_ids, scene.environment_mask, base.query_ids,
        frame_key=scene.coordinate_frame_key, query_mask=base.query_mask,
        legal_bounds=torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]),
    )
    query = with_receiver_cover(base, cover)
    scene_calls = []

    def make_scene_batch(inputs):
        scene_calls.append(inputs)
        assert len(scene_calls) == 1
        return scene

    provider.adapter = SimpleNamespace(
        make_scene_batch=make_scene_batch,
        make_query_batch=lambda prepared_scene, receivers: query,
        physical_velocity=lambda residual, coordinates: (residual, torch.zeros_like(residual), residual),
    )
    core = InheritedAttentionCoreV1(
        module_input_dim=2, environment_input_dim=7, coordinate_dim=3, query_input_dim=7,
        operating_input_dim=11, context_dim=8, num_heads=2, context_depth=1, local_hidden_dim=8,
    ).eval()
    provider.model = core
    prepared_scene = provider.make_scene(("native-input",))
    with torch.no_grad():
        predictions, auxiliary = provider.predict_native(
            core, prepared_scene, SimpleNamespace(role_ids=np.zeros((1, 2), dtype=np.int8)),
        )
    assert len(scene_calls) == 1
    assert auxiliary["organization"].scene is prepared_scene
    assert predictions.main_mps.shape == (1, 2, 3)
    assert torch.isfinite(predictions.main_mps).all()


def test_native_role_batches_use_fixed_q8192_and_preserve_formal_monitor_api(monkeypatch: pytest.MonkeyPatch):
    provider = _empty_provider()
    monkeypatch.setattr(provider, "native_evaluation_rows", lambda split, case_keys: (17, 23))
    calls = []

    def batch_builder(received_provider, row, *, split, batch_index):
        calls.append((received_provider, row, split, batch_index))
        return SimpleNamespace(
            case_keys=(row,),
            receivers=SimpleNamespace(role_ids=ROLE_IDS[None, :]),
            auxiliary={"role_query_counts": dict(zip(ROLE_NAMES, ROLE_COUNTS))},
        )

    monkeypatch.setattr(iag_wind, "make_native_role_panel_batch", batch_builder)
    validation = tuple(provider.native_role_evaluation_batches("validation"))
    train = tuple(provider.native_role_evaluation_batches("train"))

    assert [batch.case_keys for batch in validation] == [(17,), (23,)]
    assert [batch.case_keys for batch in train] == [(17,), (23,)]
    assert [(call[1], call[2], call[3]) for call in calls] == [
        (17, "validation", 0), (23, "validation", 1),
        (17, "train-fixture", 0), (23, "train-fixture", 1),
    ]
    assert WindFormalA2DTaskV1.VALIDATION_QUERY_COUNT == 10240
    assert iag_wind.WindInheritedAttentionTaskV1.validation_batches is WindFormalA2DTaskV1.validation_batches
    assert iag_wind.WindInheritedAttentionTaskV1.native_evaluation_batches is WindContentHypergraphTask.native_evaluation_batches


def test_native_role_metrics_keep_hash_and_physical_five_role_tails(monkeypatch: pytest.MonkeyPatch):
    provider = _empty_provider()
    metadata = {"row_index": 17, "layout_index": 4, "n_turbines": 8}
    targets = WindContentTargets(
        np.zeros((1, 8192, 3), dtype=np.float32), (17,), (metadata,),
    )
    predictions = SimpleNamespace(
        main_mps=torch.ones((1, 8192, 3)),
        role_ids=torch.as_tensor(ROLE_IDS[None, :]),
    )
    query_hash = "a" * 64
    record = provider.native_role_evaluation_metrics(
        predictions,
        targets,
        {"query_sampling_sha256": query_hash},
    )
    monkeypatch.setattr(provider, "native_evaluation_rows", lambda split, case_keys: (17,))
    metrics = provider.reduce_native_role_evaluation_metrics([record], split="validation")

    assert metrics["query_count_per_row"] == 8192
    assert metrics["role_query_counts"] == dict(zip(ROLE_NAMES, ROLE_COUNTS))
    assert metrics["row_query_sampling_sha256"] == {"17": query_hash}
    assert metrics["overall"]["roles"][ROLE_NAMES[0]]["vector_rmse_mps"]["mean"] == pytest.approx(np.sqrt(3.0))
    assert metrics["overall"]["roles"][ROLE_NAMES[-1]]["component_rmse_mps"]["Ux"]["count"] == 1


def test_native_role_metrics_reject_bad_sampler_receipt_or_row_inventory(monkeypatch: pytest.MonkeyPatch):
    provider = _empty_provider()
    metadata = {"row_index": 17, "layout_index": 4, "n_turbines": 8}
    targets = WindContentTargets(np.zeros((1, 8192, 3), dtype=np.float32), (17,), (metadata,))
    predictions = SimpleNamespace(
        main_mps=torch.zeros((1, 8192, 3)),
        role_ids=torch.as_tensor(ROLE_IDS[None, :]),
    )
    with pytest.raises(ValueError, match="query SHA256"):
        provider.native_role_evaluation_metrics(predictions, targets, {"query_sampling_sha256": "bad"})
    record = provider.native_role_evaluation_metrics(
        predictions, targets, {"query_sampling_sha256": "b" * 64},
    )
    monkeypatch.setattr(provider, "native_evaluation_rows", lambda split, case_keys: (23,))
    with pytest.raises(ValueError, match="exact selected rows"):
        provider.reduce_native_role_evaluation_metrics([record], split="validation")


def test_wind_iag_scene_uses_exact_native_support_bounds_and_versions_them():
    from honf_forward_core.inherited_attention import InheritedAttentionCoreV1, InheritedSceneBatch

    adapter, scene, receivers, _lower, upper = _row533_adapter_fixture()
    assert isinstance(scene, InheritedSceneBatch)
    assert scene.coordinate_frame_key == WIND_CONTENT_COORDINATE_FRAME
    assert torch.equal(scene.receiver_legal_bounds[0, 0], torch.as_tensor(
        [-17.85594367980957, -14.279596328735352, 0.022507892921566963], dtype=torch.float32,
    ))
    assert torch.equal(scene.receiver_legal_bounds[0, 1], torch.as_tensor(upper))
    # The old rounded scene metadata is preserved exactly, even though y's
    # computed upper endpoint is one FP32 ULP inward from the native bound.
    assert (scene.coordinate_origin + scene.coordinate_scale)[0, 1].item() < upper[1]
    query = adapter.make_query_batch(scene, receivers)
    assert query.receiver_cover is not None
    assert query.receiver_cover.weights.shape == (1, 1, 8)

    core = InheritedAttentionCoreV1(
        module_input_dim=2, environment_input_dim=7, coordinate_dim=3, query_input_dim=7,
        operating_input_dim=11, context_dim=8, num_heads=2, context_depth=1, local_hidden_dim=8,
    ).eval()
    with torch.no_grad():
        prepared = core.prepare(scene)
        scene.receiver_legal_bounds[0, 1, 1].add_(1.0e-6)
        with pytest.raises(ValueError, match="Prepared scene/atlas/value factors changed"):
            core.validate_prepared_state(prepared, query)


def test_wind_iag_scene_rejects_receiver_outside_exact_native_support():
    adapter, scene, _receivers, _lower, upper = _row533_adapter_fixture()
    outside_y = np.nextafter(np.float32(upper[1]), np.float32(np.inf))
    _adapter, _scene, outside_receivers, _lower, _upper = _row533_adapter_fixture(outside_y)
    with pytest.raises(ValueError, match="within the adapter's legal domain"):
        adapter.make_query_batch(scene, outside_receivers)


def test_inherited_scene_bounds_validate_shape_device_dtype_and_domain():
    _adapter, scene, _receivers, _lower, _upper = _row533_adapter_fixture()
    with pytest.raises(TypeError, match="must be a tensor"):
        replace(scene, receiver_legal_bounds=[[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="shape"):
        replace(scene, receiver_legal_bounds=torch.zeros((1, 1, 3), dtype=torch.float32))
    with pytest.raises(ValueError, match="device"):
        replace(scene, receiver_legal_bounds=torch.empty((1, 2, 3), dtype=torch.float32, device="meta"))
    with pytest.raises(TypeError, match="floating environment coordinate dtype"):
        replace(scene, receiver_legal_bounds=torch.zeros((1, 2, 3), dtype=torch.float64))
    nonfinite = scene.receiver_legal_bounds.clone()
    nonfinite[0, 1, 0] = torch.nan
    with pytest.raises(ValueError, match="finite and ordered"):
        replace(scene, receiver_legal_bounds=nonfinite)
    reversed_bounds = scene.receiver_legal_bounds.flip(1)
    with pytest.raises(ValueError, match="finite and ordered"):
        replace(scene, receiver_legal_bounds=reversed_bounds)

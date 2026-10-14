from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from honf_runtime.unified_training import SamplingKey
from windfarm.content_hypergraph import WindContentReceivers, WindContentSceneInputs
from windfarm.normalization import VelocityNormalizer
from windfarm.training.content_hypergraph_task import (
    WindContentPredictions,
    WindContentTargets,
    WindFormalA2DTaskV1,
    _validate_formal_wind_a2d_recipe_v1,
    _validate_recipe_v2,
)


RECIPE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/config_core/forward/content_hypergraph/wind_a2-d_formal_full_train_v1.json"
)


def formal_recipe() -> dict:
    return json.loads(RECIPE_PATH.read_text(encoding="utf-8"))


def _scene_inputs(row: int = 7) -> WindContentSceneInputs:
    return WindContentSceneInputs(
        row_index=row,
        case_name=f"wind-{row}",
        layout_index=2,
        wind_direction_deg=285.0,
        module_centers_D=np.zeros((1, 3), dtype=np.float32),
        module_present=np.ones(1, dtype=np.float32),
        module_features=np.ones((1, 2), dtype=np.float32),
        module_ids=np.zeros(1, dtype=np.int64),
        operating_features=np.zeros(11, dtype=np.float32),
        support_lower_D=np.zeros(3, dtype=np.float32),
        support_upper_D=np.ones(3, dtype=np.float32),
        support_extent_D=np.ones(3, dtype=np.float32),
        environment_coordinates_D=np.zeros((64, 3), dtype=np.float32),
        environment_features=np.zeros((64, 7), dtype=np.float32),
        environment_measure_D3=np.ones(64, dtype=np.float32),
    )


class _SampleCase:
    def __init__(self, row: int) -> None:
        self.index = row
        self.case = f"wind-{row}"
        self.layout_index = 2
        self.wind_direction_deg = 285.0
        self.module_present = np.ones(1, dtype=np.float32)
        self.support = object()
        self.sample_calls: list[tuple[int, str, int]] = []

    def sample_queries(self, count: int, rng: np.random.Generator, *, mode: str):
        assert mode in ("volume", "hub_band")
        marker = int(rng.integers(0, 2**31))
        self.sample_calls.append((count, mode, marker))
        coordinates = np.zeros((count, 3), dtype=np.float32)
        coordinates[:, 0] = marker % 1000
        return SimpleNamespace(
            flat_indices=np.full(count, marker, dtype=np.int64),
            coords_D=coordinates,
            velocity_mps=np.full((count, 3), marker / 1000.0, dtype=np.float32),
        )


class _OneParameterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.bias = nn.Parameter(torch.ones(1))

    def get_config(self):
        return {"architecture": "A2-D"}


def _provider_without_data() -> WindFormalA2DTaskV1:
    provider = object.__new__(WindFormalA2DTaskV1)
    row = 7
    manifest_sha = "a" * 64
    identity = {"dataset": "WindFarm", "subset_manifest_sha256": manifest_sha}
    from honf_runtime.unified_training import _sampling_dataset_id

    expected_dataset_id = _sampling_dataset_id(identity)
    case = _SampleCase(row)
    scene_inputs = _scene_inputs(row)
    provider.view = SimpleNamespace(run=lambda selected: case)
    provider.adapter = SimpleNamespace(
        scene_inputs=lambda ignored: scene_inputs,
        query_features=lambda coordinates, support: np.zeros((len(coordinates), 7), dtype=np.float32),
        packed_local_pair_indices=lambda inputs, coordinates: (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        ),
    )
    provider.sampling_seed = 42
    provider.seed = 42
    provider.primary_queries = 8192
    provider.validation_query_count = provider.VALIDATION_QUERY_COUNT
    provider.max_microbatch_cases = 8
    provider.validation_rows = np.asarray([row], dtype=np.int64)
    provider._validation_cache = {}
    provider.sampling_version = SamplingKey.CASE_EPOCH_VERSION
    provider.sampling_dataset_id = expected_dataset_id
    provider._train_row_set = {row}
    provider._validation_row_set = {row}
    provider.training_samples = 0
    provider.validation_samples = 0
    provider.role_query_counts = {"volume": 6144, "hub_band": 2048}
    provider.validation_role_query_counts = {"volume": 8192, "hub_band": 2048}
    provider.identity_payload = lambda: identity
    provider._test_expected_dataset_id = expected_dataset_id
    provider._test_case = case
    return provider


def test_formal_recipe_is_a_separate_strict_a2d_training_protocol():
    recipe = formal_recipe()
    validated, architecture, microbatch = _validate_formal_wind_a2d_recipe_v1(recipe)
    assert validated == recipe
    assert architecture == "A2-D"
    assert microbatch == 8
    legacy_curve = copy.deepcopy(recipe)
    legacy_curve["execution"]["curve_every"] = 100
    assert _validate_formal_wind_a2d_recipe_v1(legacy_curve)[0] == legacy_curve
    for unsupported_curve in (1, 5, 20, True):
        invalid_curve = copy.deepcopy(recipe)
        invalid_curve["execution"]["curve_every"] = unsupported_curve
        with pytest.raises(ValueError, match="curve_every"):
            _validate_formal_wind_a2d_recipe_v1(invalid_curve)
    with pytest.raises(ValueError, match="development or full_train1000"):
        _validate_recipe_v2(recipe)
    for key, invalid in (
        ("seed", 42),
        ("sampling_seed", 0),
        ("primary_queries", 4096),
        ("validation_volume_queries", 6144),
        ("effective_batch_size", 24),
        ("microbatch_size", 12),
        ("optimizer_bias_norm_exclusion", True),
        ("approved_stop_after", 1000),
    ):
        mutated = copy.deepcopy(recipe)
        mutated[key] = invalid
        with pytest.raises(ValueError):
            _validate_formal_wind_a2d_recipe_v1(mutated)
    mutated = copy.deepcopy(recipe)
    mutated["architecture"] = "A2-H6"
    with pytest.raises(ValueError, match="A2-D"):
        _validate_formal_wind_a2d_recipe_v1(mutated)
    mutated = copy.deepcopy(recipe)
    mutated["optimizer"]["bias_and_norm_weight_decay"] = 0.0
    with pytest.raises(ValueError, match="AdamW"):
        _validate_formal_wind_a2d_recipe_v1(mutated)


def test_formal_sampling_uses_exact_two_region_native_panels_and_fixed_validation():
    provider = _provider_without_data()
    key = SamplingKey(
        42,
        17,
        0,
        0,
        "joint",
        "A2-D",
        sampling_version=SamplingKey.CASE_EPOCH_VERSION,
        dataset_id=provider._test_expected_dataset_id,
    )
    batch = provider._make_batch((7,), key, validation=False, query_count=8192)
    assert batch.receivers.role_ids.shape == (1, 8192)
    assert np.count_nonzero(batch.receivers.role_ids[0] == 0) == 6144
    assert np.count_nonzero(batch.receivers.role_ids[0] == 1) == 2048
    assert provider._test_case.sample_calls[:2][0][:2] == (6144, "volume")
    assert provider._test_case.sample_calls[:2][1][:2] == (2048, "hub_band")
    train_hash = batch.auxiliary["query_sampling_sha256"]
    repeated = provider._make_batch((7,), key, validation=False, query_count=8192)
    assert repeated.auxiliary["query_sampling_sha256"] == train_hash
    assert provider.loss_denominators((batch,)) == {provider.LOSS_NAME: 1.0}

    validation_key = SamplingKey(
        42,
        17,
        9,
        3,
        "validation",
        "A2-D",
        sampling_version=SamplingKey.CASE_EPOCH_VERSION,
        dataset_id=provider._test_expected_dataset_id,
    )
    validation_a = provider._make_batch((7,), validation_key, validation=True, query_count=10240)
    validation_b = provider._make_batch((7,), validation_key, validation=True, query_count=10240)
    assert validation_a.receivers.role_ids.shape == (1, 10240)
    assert np.count_nonzero(validation_a.receivers.role_ids[0] == 0) == 8192
    assert np.count_nonzero(validation_a.receivers.role_ids[0] == 1) == 2048
    assert validation_a.auxiliary["query_sampling_sha256"] == validation_b.auxiliary["query_sampling_sha256"]
    assert provider._test_case.sample_calls[-4][0:2] == (8192, "volume")
    assert provider._test_case.sample_calls[-3][0:2] == (2048, "hub_band")
    assert provider.validation_samples == 2
    # Engine validation must use the historical fixed Q10240 selector panel.
    validation_batches = tuple(provider.validation_batches())
    assert len(validation_batches) == 1
    assert validation_batches[0].receivers.role_ids.shape == (1, 10240)
    provider._validation_row_set.add(8)
    multirow_array_batch = provider._make_batch(
        np.asarray([7, 8], dtype=np.int64), validation_key, validation=True, query_count=10240
    )
    assert multirow_array_batch.receivers.role_ids.shape == (2, 10240)
    with pytest.raises(ValueError, match="exactly Q8192"):
        provider._make_batch((7,), key, validation=False, query_count=4096)
    with pytest.raises(ValueError, match="exactly Q10240"):
        provider._make_batch((7,), validation_key, validation=True, query_count=8192)


def test_formal_loss_is_case_equal_weighted_normalized_mse_with_historical_region_weights():
    provider = object.__new__(WindFormalA2DTaskV1)
    provider.primary_queries = 8192
    provider.adapter = SimpleNamespace(
        normalizer=VelocityNormalizer(
            mean=np.zeros(3),
            std=np.asarray([0.5, 0.25, 0.75]),
            safe_std=np.asarray([0.5, 0.25, 0.75]),
            u_ref_mps=10.0,
        )
    )
    target = np.zeros((2, 8192, 3), dtype=np.float32)
    normalized_error = torch.zeros((2, 8192, 3), dtype=torch.float32)
    normalized_error[:, :6144] = torch.tensor([1.0, 2.0, 3.0])
    normalized_error[:, 6144:] = torch.tensor([2.0, 1.0, 4.0])
    scale = torch.tensor([5.0, 2.5, 7.5])
    physical_prediction = normalized_error * scale
    metadata = tuple({"row_index": row} for row in (10, 11))
    targets = WindContentTargets(target, (10, 11), metadata)
    predictions = WindContentPredictions(
        main_mps=physical_prediction,
        normalized_residual=normalized_error,
        residual_mps=physical_prediction,
        profile_mps=torch.zeros_like(physical_prediction),
        role_ids=torch.cat(
            (
                torch.zeros((2, 6144), dtype=torch.long),
                torch.ones((2, 2048), dtype=torch.long),
            ),
            dim=1,
        ),
    )
    term = provider._formal_loss_terms(predictions, targets)[provider.LOSS_NAME]
    expected_volume = float(torch.tensor([1.0, 2.0, 3.0]).square().mean())
    expected_band = float(torch.tensor([2.0, 1.0, 4.0]).square().mean())
    expected_case = 0.75 * expected_volume + 0.25 * expected_band
    assert term.denominator == 2.0
    assert float(term.numerator) == pytest.approx(2.0 * expected_case)

    validation_target = np.zeros((2, 10240, 3), dtype=np.float32)
    validation_error = torch.zeros((2, 10240, 3), dtype=torch.float32)
    validation_error[:, :8192] = torch.tensor([1.0, 2.0, 3.0])
    validation_error[:, 8192:] = torch.tensor([2.0, 1.0, 4.0])
    validation_prediction = validation_error * scale
    validation_targets = WindContentTargets(validation_target, (10, 11), metadata)
    validation_predictions = WindContentPredictions(
        main_mps=validation_prediction,
        normalized_residual=validation_error,
        residual_mps=validation_prediction,
        profile_mps=torch.zeros_like(validation_prediction),
        role_ids=torch.cat(
            (
                torch.zeros((2, 8192), dtype=torch.long),
                torch.ones((2, 2048), dtype=torch.long),
            ),
            dim=1,
        ),
    )
    val_term = provider._formal_loss_terms(validation_predictions, validation_targets, validation=True)[
        provider.LOSS_NAME
    ]
    expected_validation_case = 0.8 * expected_volume + 0.2 * expected_band
    assert float(val_term.numerator) == pytest.approx(2.0 * expected_validation_case)
    records = provider.validation_metrics(validation_predictions, validation_targets, {})
    reduced = provider.reduce_native_metrics((records,), expected_rows=(10, 11))
    assert reduced["val_volume_mse"] == pytest.approx(expected_volume)
    assert reduced["val_band_mse"] == pytest.approx(expected_band)
    assert reduced["val_weighted_mse"] == pytest.approx(expected_validation_case)
    with pytest.raises(ValueError, match="exact VALID rows"):
        provider.reduce_native_metrics((records,), expected_rows=(10,))


def test_formal_optimizer_keeps_one_group_and_decays_bias_and_norm_parameters():
    provider = object.__new__(WindFormalA2DTaskV1)
    provider.model = _OneParameterModel()
    provider.recipe = formal_recipe()
    groups = provider.optimizer_groups(provider.model)
    assert len(groups) == 1
    assert groups[0].name == "wind_formal_adamw_all_parameters"
    assert groups[0].parameter_names == ("weight", "bias")
    assert groups[0].weight_decay == 1.0e-5
    assert groups[0].schedule.value(1) == 3.0e-4
    assert groups[0].schedule.value(2500) == 3.0e-4
    with pytest.raises(ValueError, match="only to this A2-D"):
        provider.optimizer_groups(provider.model, arm="A2-H6")

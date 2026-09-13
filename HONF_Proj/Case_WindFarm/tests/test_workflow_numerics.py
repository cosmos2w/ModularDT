"""Real-data numerical checks for the bounded WindFarm workflow."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from honf_forward_core.config import UnifiedForwardConfig

from windfarm.data import WindFarmNativeDataset, WindFarmNativeView, collate_windfarm
from windfarm.model import WindFarmForwardModel
from windfarm.normalization import read_normalization_json
from windfarm.splits import make_group_split
from windfarm.workflows.train_forward import _as_device_batch, _weighted_loss

CASE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CASE_ROOT.parent
VOLUME_ROOT = CASE_ROOT / "Dataset" / "links" / "wind_farm"
NORMALIZATION_PATH = CASE_ROOT / "Dataset" / "derived" / "forward_velocity_v1" / "normalization.json"


def _real_view() -> tuple[WindFarmNativeView, np.ndarray]:
    view = WindFarmNativeView(VOLUME_ROOT)
    split = make_group_split(np.asarray(view.volume.array("layout_index")), seed=42)
    return view, split.train


def test_sampled_loss_is_exactly_the_documented_volume_band_mixture() -> None:
    prediction = torch.zeros((2, 5, 3), dtype=torch.float32)
    target = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [1.0, 1.0, 0.0], [0.0, 0.0, 4.0]],
            [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [1.0, 2.0, 3.0], [4.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    raw_batch = {
        "query_loss_weight": np.asarray(
            [[0.75 / 3.0, 0.75 / 3.0, 0.75 / 3.0, 0.25 / 2.0, 0.25 / 2.0],
             [0.75 / 3.0, 0.75 / 3.0, 0.75 / 3.0, 0.25 / 2.0, 0.25 / 2.0]],
            dtype=np.float32,
        ),
    }
    loss, per_case, weighted_squared_error, point_error = _weighted_loss(
        prediction,
        target,
        raw_batch,
        torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32),
    )

    # Case 0: volume mean=3/2, band mean=11/6, mixture=19/12.
    # Case 1: volume mean=8/9, band mean=59/12, mixture=91/48.
    torch.testing.assert_close(per_case, torch.tensor([19.0 / 12.0, 91.0 / 48.0]))
    torch.testing.assert_close(loss, torch.tensor(167.0 / 96.0))
    assert weighted_squared_error.shape == target.shape
    torch.testing.assert_close(point_error, weighted_squared_error.mean(dim=-1))

    equal_mixture, _, _, _ = _weighted_loss(
        prediction,
        target,
        {"query_loss_weight": np.full((2, 5), 0.2, dtype=np.float32)},
        torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32),
    )
    assert not torch.allclose(loss, equal_mixture)


def test_real_validation_sampling_is_fixed_and_training_sampling_changes_by_epoch() -> None:
    view, train_rows = _real_view()
    normalizer, _ = read_normalization_json(NORMALIZATION_PATH)
    validation = WindFarmNativeDataset(
        view,
        train_rows[:1],
        normalizer=normalizer,
        queries_per_case=32,
        volume_fraction=0.75,
        seed=42,
        fixed_sampling=True,
    )
    fixed_first = validation[0]
    fixed_second = validation[0]
    np.testing.assert_array_equal(fixed_first["query_xy"], fixed_second["query_xy"])
    np.testing.assert_array_equal(fixed_first["target_field"], fixed_second["target_field"])

    training = WindFarmNativeDataset(
        view,
        train_rows[:1],
        normalizer=normalizer,
        queries_per_case=32,
        volume_fraction=0.75,
        seed=42,
        fixed_sampling=False,
    )
    training.set_epoch(1)
    epoch_one = training[0]["query_xy"]
    training.set_epoch(2)
    epoch_two = training[0]["query_xy"]
    assert not np.array_equal(epoch_one, epoch_two)


@pytest.mark.parametrize("profile_name", ("windfarm_classic_k6.json", "windfarm_dense_pairwise.json"))
def test_real_model_prediction_does_not_read_target_field(profile_name: str) -> None:
    view, train_rows = _real_view()
    normalizer, _ = read_normalization_json(NORMALIZATION_PATH)
    config_payload = json.loads((PROJECT_ROOT / "src" / "config_core" / "forward" / profile_name).read_text())
    config = UnifiedForwardConfig.from_dict(dict(config_payload["model"]["core_honf"]))
    dataset = WindFarmNativeDataset(
        view,
        train_rows[:1],
        normalizer=normalizer,
        queries_per_case=32,
        volume_fraction=0.75,
        seed=42,
        fixed_sampling=True,
    )
    raw = collate_windfarm([dataset[0]])
    batch = _as_device_batch(raw, torch.device("cpu"))
    model = WindFarmForwardModel(config, velocity_transform=normalizer).eval()
    model.materialize(batch)
    with torch.no_grad():
        prepared = model.prepare_case(batch)
        reference = model.predict_standardized(
            prepared,
            batch.query_xy,
            batch.query_features,
            receiver_chunk_size=128,
        )
        altered_target = torch.full_like(batch.target_field, 123.0)
        altered_batch = replace(batch, target_field=altered_target)
        altered_prepared = model.prepare_case(altered_batch)
        altered = model.predict_standardized(
            altered_prepared,
            altered_batch.query_xy,
            altered_batch.query_features,
            receiver_chunk_size=128,
        )
    torch.testing.assert_close(reference, altered, rtol=0.0, atol=0.0)


__all__ = [
    "test_real_model_prediction_does_not_read_target_field",
    "test_real_validation_sampling_is_fixed_and_training_sampling_changes_by_epoch",
    "test_sampled_loss_is_exactly_the_documented_volume_band_mixture",
]

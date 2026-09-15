"""Bounded real-data execution checks for the WindFarm forward study.

These tests deliberately use a few training-layout rows and small query
counts.  They exercise the native reader, wrapper, diagnostic reductions,
figures, timing primitive, and trusted checkpoint loader without running a
native-volume sweep or a managed training run.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from honf_forward_core.config import UnifiedForwardConfig
from windfarm.data import WindFarmNativeView, case_batch
from windfarm.geometry import support_weights
from windfarm.model import WindFarmForwardModel
from windfarm.normalization import VelocityNormalizer, VerticalProfileBaseline
from windfarm.splits import GroupSplit, make_group_split
from windfarm.study_cost import measure
from windfarm.study_diagnostics import geometry_diagnostics
from windfarm.study_spatial import (
    native_coordinates,
    native_plane,
    render_native_plane,
    stream_native_errors,
)
from windfarm.workflows.evaluate_forward import load_checkpoint
from windfarm.workflows.study_evidence import _predictor, _routing_summary, _vertical_profile
from windfarm.workflows.train_forward import _save_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOLUME_ROOT = PROJECT_ROOT / "Dataset" / "links" / "wind_farm"
DERIVED_ROOT = PROJECT_ROOT / "Dataset" / "derived" / "forward_velocity_v1"
CONFIG_ROOT = PROJECT_ROOT.parent / "src" / "config_core" / "forward"


@pytest.fixture(scope="session")
def native_view() -> WindFarmNativeView:
    if not VOLUME_ROOT.exists():
        pytest.skip(f"WindFarm volume is unavailable at {VOLUME_ROOT}")
    return WindFarmNativeView(VOLUME_ROOT)


@pytest.fixture(scope="session")
def seed42_split(native_view: WindFarmNativeView) -> GroupSplit:
    split = make_group_split(np.asarray(native_view.volume.array("layout_index")), seed=42)
    assert (len(split.train), len(split.validation), len(split.test)) == (420, 90, 90)
    assert len(np.unique(np.asarray(native_view.volume.array("layout_index"))[split.train])) == 140
    assert len(np.unique(np.asarray(native_view.volume.array("layout_index"))[split.validation])) == 30
    assert len(np.unique(np.asarray(native_view.volume.array("layout_index"))[split.test])) == 30
    return split


@pytest.fixture(scope="session")
def normalizer() -> VelocityNormalizer:
    path = DERIVED_ROOT / "normalization.json"
    if not path.is_file():
        pytest.skip(f"WindFarm normalization sidecar is unavailable at {path}")
    return VelocityNormalizer.from_dict(json.loads(path.read_text(encoding="utf-8")))


@pytest.fixture(scope="session")
def vertical_baseline() -> VerticalProfileBaseline:
    path = DERIVED_ROOT / "normalization.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    profile = payload.get("vertical_profile_baseline")
    if not isinstance(profile, dict):
        pytest.skip("WindFarm normalization sidecar has no vertical-profile baseline")
    return VerticalProfileBaseline.from_dict(profile)


@pytest.fixture(scope="session", params=("classic", "dense"))
def actual_model(request: pytest.FixtureRequest, native_view: WindFarmNativeView,
                 seed42_split: GroupSplit, normalizer: VelocityNormalizer):
    profile_name = str(request.param)
    config_path = CONFIG_ROOT / (
        "windfarm_classic_k6.json" if profile_name == "classic" else "windfarm_dense_pairwise.json"
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = UnifiedForwardConfig.from_dict(payload["model"]["core_honf"])
    case = native_view.run(int(seed42_split.train[0]))
    torch.manual_seed(20260913 + (0 if profile_name == "classic" else 1))
    model = WindFarmForwardModel(config, velocity_transform=normalizer).cpu().eval()
    geometry_only = case_batch(case, case.module_centers)
    prepared = model.materialize(geometry_only)
    return SimpleNamespace(
        name=profile_name,
        config=config,
        model=model,
        case=case,
        prepared=prepared,
        geometry_only=geometry_only,
    )


def test_real_native_index_stream_and_slice_figure(
    native_view: WindFarmNativeView,
    seed42_split: GroupSplit,
    normalizer: VelocityNormalizer,
    vertical_baseline: VerticalProfileBaseline,
    tmp_path: Path,
) -> None:
    case = native_view.run(int(seed42_split.train[0]))
    run = case.run
    assert isinstance(run.U, np.memmap)

    flat = np.asarray([0, run.nx - 1, run.nx, run.nx * run.ny, run.cell_count - 1], dtype=np.int64)
    coordinates = native_coordinates(run, flat, case.diameter_m) * case.diameter_m
    np.testing.assert_allclose(coordinates[2], [run.x_m[0], run.y_m[1], run.z_m[0]], rtol=1e-6)
    np.testing.assert_allclose(coordinates[-1], [run.x_m[-1], run.y_m[-1], run.z_m[-1]], rtol=1e-6)

    plane = native_plane(run, "z", case.hub_height_m)
    index = int(plane["index"])
    np.testing.assert_array_equal(plane["target"], run.U_structured[index])
    render_native_plane(
        plane,
        np.zeros_like(plane["target"]),
        case.module_centers,
        "bounded native slice",
        tmp_path / "native_z_slice.png",
    )
    assert (tmp_path / "native_z_slice.png").stat().st_size > 0

    # Stream only two native z planes while retaining exact C-order fields.
    z_count = min(2, run.nz)
    short_run = replace(
        run,
        shape_nxyz=(run.nx, run.ny, z_count),
        z_m=run.z_m[:z_count],
        U=run.U[: run.nx * run.ny * z_count],
        p=run.p[: run.nx * run.ny * z_count],
        k=run.k[: run.nx * run.ny * z_count],
        epsilon=run.epsilon[: run.nx * run.ny * z_count],
    )
    axis_weights = tuple(axis / case.diameter_m for axis in support_weights(
        short_run.x_m, short_run.y_m, short_run.z_m
    ))
    reduction = stream_native_errors(
        short_run,
        lambda coords: np.zeros((len(coords), 3), dtype=np.float32),
        vertical_baseline.predict,
        axis_weights,
        case.module_centers,
        normalizer.safe_std * normalizer.u_ref_mps,
        diameter=case.diameter_m,
        hub_height=case.hub_height_m,
        chunk_size=257,
    )
    assert reduction["volume"]["available"] is True
    assert reduction["volume"]["count"] == short_run.cell_count
    expected_volume = (
        (short_run.x_m[-1] - short_run.x_m[0])
        * (short_run.y_m[-1] - short_run.y_m[0])
        * (short_run.z_m[-1] - short_run.z_m[0])
        / case.diameter_m**3
    )
    np.testing.assert_allclose(reduction["volume"]["quadrature_volume_D3"], expected_volume, rtol=1e-6)
    assert np.isfinite(reduction["volume"]["standardized_mse"])


def test_real_wrapper_diagnostics_figures_and_cost(
    actual_model: SimpleNamespace,
    native_view: WindFarmNativeView,
    seed42_split: GroupSplit,
    normalizer: VelocityNormalizer,
    vertical_baseline: VerticalProfileBaseline,
    tmp_path: Path,
) -> None:
    model = actual_model.model
    case = actual_model.case
    prepared = actual_model.prepared
    sample = case.sample_queries(8, np.random.default_rng(4200 + case.index), mode="volume")
    geometry_only = case_batch(case, sample.coords_D)
    assert geometry_only.target_field is None
    assert tuple(geometry_only.env_coords.shape) == (1, 512, 3)
    assert tuple(geometry_only.env_features.shape) == (1, 512, 7)
    assert tuple(geometry_only.env_weights.shape) == (1, 512)
    np.testing.assert_allclose(float(geometry_only.env_weights.sum()), case.support.volume_D3, rtol=1e-6)

    output = model.decode(
        prepared,
        geometry_only.query_xy,
        geometry_only.query_features,
        receiver_chunk_size=3,
        return_routing_maps=True,
    )
    prediction = output["pred_field"]
    assert tuple(prediction.shape) == (1, 8, 3)
    assert torch.isfinite(prediction).all()
    if actual_model.name == "classic":
        assert output["query_hyper_attention"].shape[1] == 8
    else:
        assert output["dense_environment_attention"].shape[2] == 8

    chunked = model.predict_standardized(
        prepared, geometry_only.query_xy, geometry_only.query_features, receiver_chunk_size=8
    )
    torch.testing.assert_close(prediction, chunked, rtol=1e-5, atol=1e-6)
    physical = model.predict_physical(
        prepared, geometry_only.query_xy, geometry_only.query_features, receiver_chunk_size=4
    )
    expected_physical = (
        chunked * torch.as_tensor(normalizer.safe_std, dtype=chunked.dtype)
        + torch.as_tensor(normalizer.mean, dtype=chunked.dtype)
    ) * normalizer.u_ref_mps
    torch.testing.assert_close(physical, expected_physical, rtol=1e-5, atol=1e-5)

    # The prepared case contains encoded geometry only; neither target nor a
    # solved-field statistic is allowed to become a model input.
    encoded = prepared.encoded
    names = set(encoded) if isinstance(encoded, dict) else {
        field.name for field in getattr(encoded, "__dataclass_fields__", {}).values()
    }
    assert not any(any(term in name.lower() for term in ("target", "wake", "loss")) for name in names)

    diagnostic_batches = []
    for row in np.asarray(seed42_split.train[:3], dtype=np.int64):
        diagnostic_case = native_view.run(int(row))
        diagnostic_sample = diagnostic_case.sample_queries(
            8, np.random.default_rng(8100 + int(row)), mode="volume"
        )
        diagnostic_batches.append(case_batch(
            diagnostic_case,
            diagnostic_sample.coords_D,
            normalizer=normalizer,
            velocity_mps=diagnostic_sample.velocity_mps,
        ))
    records = geometry_diagnostics(model, diagnostic_batches)
    assert len(records) == 3
    for record in records:
        assert record["query_count"] == 8
        assert np.isfinite(record["environment_volume_D3"])
        assert np.isfinite(record["same_chunk_repeat"]["rms"])
        assert np.isfinite(record["split_weight_duplication"]["rms"])

    predict = _predictor(model, case, prepared, torch.device("cpu"), chunk=4)
    predicted = predict(sample.coords_D)
    assert predicted.shape == (8, 3)
    assert np.isfinite(predicted).all()
    profile = _vertical_profile(case, predict, vertical_baseline, tmp_path / f"{actual_model.name}_profile.png")
    assert len(profile["z_D"]) == case.run.nz
    assert (tmp_path / f"{actual_model.name}_profile.png").stat().st_size > 0
    routing = _routing_summary(
        model, case, prepared, torch.device("cpu"), tmp_path / f"{actual_model.name}_routing.png"
    )
    assert routing["receiver_count"] == 512
    assert (tmp_path / f"{actual_model.name}_routing.png").stat().st_size > 0

    timing = measure(
        lambda: model.predict_standardized(
            prepared, geometry_only.query_xy, geometry_only.query_features, receiver_chunk_size=4
        ),
        torch.device("cpu"),
        warmups=1,
        repeats=2,
    )
    assert len(timing["seconds"]) == 2
    assert all(np.isfinite(timing["seconds"]))
    assert all(value > 0.0 for value in timing["seconds"])


def test_tiny_real_update_trusted_checkpoint_round_trip(
    native_view: WindFarmNativeView,
    seed42_split: GroupSplit,
    normalizer: VelocityNormalizer,
    vertical_baseline: VerticalProfileBaseline,
    tmp_path: Path,
) -> None:
    profile_path = CONFIG_ROOT / "windfarm_classic_k6.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    core_payload = copy.deepcopy(profile["model"]["core_honf"])
    core_payload.update(
        {
            "hidden_dim": 16,
            "pairwise_kernel_hidden_dim": 16,
            "pairwise_kernel_num_layers": 2,
            "query_fourier_frequencies": 1,
            "position_fourier_frequencies": 1,
            "pairwise_kernel_fourier_frequencies": 1,
        }
    )
    config = UnifiedForwardConfig.from_dict(core_payload)
    torch.manual_seed(20260931)
    model = WindFarmForwardModel(config, velocity_transform=normalizer).cpu()
    case = native_view.run(int(seed42_split.train[0]))
    sample = case.sample_queries(4, np.random.default_rng(921), mode="volume")
    batch = case_batch(
        case,
        sample.coords_D,
        normalizer=normalizer,
        velocity_mps=sample.velocity_mps,
    )
    geometry_only = case_batch(case, case.module_centers)
    model.materialize(geometry_only)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(batch)["pred_field"]
    loss = (prediction - batch.target_field).square().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    assert float(loss.detach()) >= 0.0

    checkpoint_path = tmp_path / "tiny_real_update.pt"
    training_config = {
        "dataset": {"dataset_id": "wind_farm_volume_v1", "dataset_schema": 1},
        "model": {"core_honf": config.to_dict()},
    }
    split_metadata = seed42_split
    _save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        best_metric=float(loss.detach()),
        best_epoch=1,
        config=training_config,
        normalizer=normalizer,
        split=split_metadata,
        profile=vertical_baseline,
        update_count=1,
    )
    assert checkpoint_path.is_file()

    loaded, payload = load_checkpoint(
        checkpoint_path,
        device="cpu",
        materialization_batch=geometry_only,
    )
    assert payload["case_id"] == "WindFarm"
    assert payload["model_family"] == "honf_forward"
    loaded_prepared = loaded.materialize(geometry_only)
    model.eval()
    with torch.no_grad():
        expected = model.predict_standardized(
            model.materialize(geometry_only), batch.query_xy, batch.query_features, receiver_chunk_size=2
        )
        restored = loaded.predict_standardized(
            loaded_prepared, batch.query_xy, batch.query_features, receiver_chunk_size=2
        )
    torch.testing.assert_close(restored, expected, rtol=1e-6, atol=1e-6)

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from channelthermal.inverse.heat_allocation import (
    ThermalHeatFeatureScaler,
    build_thermal_heat_tasks,
    centered_log_fractions,
    collate_thermal_heat_tasks,
    select_fluid_sensor_indices,
)
from channelthermal.inverse.packet_reuse import (
    ThermalCandidatePacketProvider,
    _frontier_evidence_for_selection,
    packet_links_from_interfaces,
    provider_known_inputs,
    train_matched_heat_diffusion,
)

from honf_forward_core.interface_fields.budgeted_frontier import FrontierSelection

from honf_inverse_core.models.frozen_packet_diffusion import (
    ConditionalPacketDenoiser,
    PacketLinks,
    heat_from_logits,
)


def _write_case(handle: h5py.File, case_id: str, *, split: str, heat: tuple[float, ...]) -> None:
    group = handle["cases"].create_group(case_id)
    group.attrs["converged"] = True
    group.attrs["split"] = split
    config = {
        "domain": {"lx": 12.0, "ly": 6.0, "module_radius": 0.45},
        "flow": {"re": 50.0, "u_in": 1.0, "nu": 0.018},
        "thermal": {"solid_alpha": 0.01, "fluid_alpha": 0.02, "solid_k": 1.0, "fluid_k": 1.5},
    }
    group.create_dataset("case_config_json", data=np.bytes_(json.dumps(config)))
    centers = np.zeros((12, 2), dtype=np.float32)
    centers[: len(heat)] = np.asarray([[2.0 + 2.0 * i, 1.5 + 0.25 * i] for i in range(len(heat))])
    present = np.zeros((12,), dtype=np.float32)
    present[: len(heat)] = 1.0
    powers = np.zeros((12,), dtype=np.float32)
    powers[: len(heat)] = heat
    group.create_dataset("module_centers", data=centers)
    group.create_dataset("module_present", data=present)
    group.create_dataset("heat_powers", data=powers)
    material = group.create_group("material_parameters")
    for key, value in {
        "nu": 0.018,
        "solid_alpha": 0.01,
        "fluid_alpha": 0.02,
        "solid_k": 1.0,
        "fluid_k": 1.5,
        "module_radius": 0.45,
    }.items():
        material.attrs[key] = value
    y, x = np.mgrid[0:64, 0:128]
    x_grid = (12.0 * x / 127.0).astype(np.float32)
    y_grid = (6.0 * y / 63.0).astype(np.float32)
    mask = np.zeros((64, 128), dtype=np.uint8)
    for cx, cy in centers[: len(heat)]:
        mask[((x_grid - cx) ** 2 + (y_grid - cy) ** 2) < 0.45**2] = 1
    group.create_dataset("x_grid", data=x_grid)
    group.create_dataset("y_grid", data=y_grid)
    group.create_dataset("module_mask", data=mask)
    field = np.zeros((64, 128, 5), dtype=np.float32)
    field[..., 4] = 280.0 + 3.0 * x_grid + 2.0 * y_grid
    group.create_dataset("steady_field", data=field)


def _write_dataset(path) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("case_ids", data=np.asarray([b"train_a", b"train_b", b"train_single", b"test_a"]))
        handle.create_dataset("splits", data=np.asarray([b"train", b"train", b"train", b"test"]))
        cases = handle.create_group("cases")
        _write_case(handle, "train_a", split="train", heat=(1.0, 2.0, 3.0))
        _write_case(handle, "train_b", split="train", heat=(2.0, 2.0, 2.0, 2.0, 2.0))
        _write_case(handle, "train_single", split="train", heat=(4.0,))
        _write_case(handle, "test_a", split="test", heat=(1.0, 4.0, 2.0, 1.0))


def test_centered_log_fractions_decode_to_the_supplied_nonnegative_total() -> None:
    heat = np.asarray([1.0, 2.0, 5.0], dtype=np.float32)
    logits = centered_log_fractions(heat)
    decoded = heat_from_logits(
        torch.as_tensor(logits).reshape(1, -1, 1),
        torch.tensor([[heat.sum()]]),
        torch.ones((1, heat.size), dtype=torch.bool),
    )
    torch.testing.assert_close(decoded.flatten(), torch.as_tensor(heat), rtol=1e-6, atol=1e-6)
    assert abs(float(logits.sum())) < 1e-6


def test_unsupported_frontier_keeps_least_risk_predictions_explicit() -> None:
    prediction = SimpleNamespace(
        predicted_work=torch.tensor([0.2, 0.7]),
        role_distortion=torch.tensor([[0.1, 0.3], [0.4, 0.9]]),
    )
    selection = FrontierSelection(
        selected_frontier=None,
        selected_index=None,
        unsupported_at_budget=True,
        least_risk_frontier=(4, 5),
        least_risk_index=1,
    )
    evidence = _frontier_evidence_for_selection(selection, prediction)
    assert evidence.selected_frontier is None
    assert evidence.unsupported_at_budget is True
    assert evidence.predicted_frontier == (4, 5)
    assert evidence.prediction_scope == "least_risk_research_only"
    assert evidence.predicted_work == pytest.approx(0.7)
    assert evidence.predicted_role_distortion == pytest.approx((0.4, 0.9))


def test_sensor_catalogue_is_spatially_broad_and_geometry_only() -> None:
    y, x = np.mgrid[0:64, 0:128]
    mask = np.zeros((64, 128), dtype=bool)
    mask[(x - 60) ** 2 + (y - 30) ** 2 < 8**2] = True
    x_grid = 12.0 * x / 127.0
    y_grid = 6.0 * y / 63.0
    first = select_fluid_sensor_indices(mask, x_grid, y_grid, sensor_count=12)
    second = select_fluid_sensor_indices(mask.copy(), x_grid.copy(), y_grid.copy(), sensor_count=12)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (12, 2)
    assert not mask[first[:, 0], first[:, 1]].any()
    assert np.unique(first, axis=0).shape[0] == 12


def test_heat_tasks_use_train_split_and_keep_hidden_heat_out_of_condition(tmp_path) -> None:
    path = tmp_path / "thermal.h5"
    _write_dataset(path)
    train = build_thermal_heat_tasks(path, split="train", sensor_count=8)
    test = build_thermal_heat_tasks(path, split="test", sensor_count=8)
    assert tuple(task.case_id for task in train) == ("train_a", "train_b")
    assert tuple(task.case_id for task in test) == ("test_a",)
    all_train = build_thermal_heat_tasks(path, split="train", sensor_count=8, min_modules=1)
    assert tuple(task.case_id for task in all_train) == ("train_a", "train_b", "train_single")
    scaler = ThermalHeatFeatureScaler.fit(train)
    left = collate_thermal_heat_tasks([train[0]], scaler=scaler)
    poisoned_target = replace(
        train[0],
        clean_heat=np.asarray([0.5, 2.5, 3.0], dtype=np.float32),
        clean_state=centered_log_fractions(np.asarray([0.5, 2.5, 3.0]))[:, None],
    )
    right = collate_thermal_heat_tasks([poisoned_target], scaler=scaler)
    for name in ("known_state", "design_mask", "module_valid", "module_features", "sensor_features"):
        torch.testing.assert_close(getattr(left.condition, name), getattr(right.condition, name))
    assert not torch.equal(left.clean_heat, right.clean_heat)
    assert left.condition.module_features.shape[-1] == 13
    assert left.condition.sensor_features.shape[-1] == 3
    assert torch.allclose(left.total_heat, torch.tensor([[6.0]]))
    torch.testing.assert_close(left.module_centers[0, :3], torch.as_tensor(train[0].module_centers))
    np.testing.assert_allclose(left.physical_context[0].numpy(), np.asarray([50.0, 1.0, 12.0, 6.0, 0.018, 0.01, 0.02, 1.0, 1.5, 0.45]))
    assert np.isfinite(left.sensor_xy.numpy()).all()
    with pytest.raises(ValueError, match="train-only"):
        ThermalHeatFeatureScaler.fit(test)


def test_requested_case_ids_cannot_silently_cross_splits(tmp_path) -> None:
    path = tmp_path / "thermal.h5"
    _write_dataset(path)
    with pytest.raises(ValueError, match="absent from split"):
        build_thermal_heat_tasks(path, split="train", case_ids=("test_a",))


def test_inverse_development_excludes_forward_fit_duplicate_family_alias() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/run_active_packet_inverse.py"
    spec = importlib.util.spec_from_file_location("thermal_active_packet_inverse_for_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tasks = tuple(
        SimpleNamespace(
            case_id=case_id,
            module_valid=np.ones(3, dtype=bool),
            physical_context=np.asarray([float(index + 1)], dtype=np.float32),
        )
        for index, case_id in enumerate(("0273", "test_a", "test_b", "test_c", "test_d", "test_e", "test_f", "test_g", "test_h"))
    )
    selected = module._development_tasks(tasks)
    assert tuple(task.case_id for task in selected) == (
        "test_a", "test_b", "test_c", "test_d", "test_e", "test_f", "test_g", "test_h"
    )


def test_paired_fraction_shift_uses_collated_validity_for_mixed_module_counts() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/run_active_packet_inverse.py"
    spec = importlib.util.spec_from_file_location("thermal_active_packet_inverse_for_shift_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = np.zeros((2, 2, 10), dtype=np.float32)
    changed = baseline.copy()
    changed[:, 0, :3] = (1.0, 2.0, 3.0)
    changed[:, 1, :5] = (1.0, 1.0, 1.0, 1.0, 1.0)
    changed[:, :, 5:] = 1000.0  # Padding must never affect the metric.
    valid = np.zeros((2, 10), dtype=bool)
    valid[0, :3] = True
    valid[1, :5] = True
    total = np.asarray([[6.0], [5.0]], dtype=np.float32)
    assert module._paired_fraction_shift(changed, baseline, valid, total) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="padded case batch"):
        module._paired_fraction_shift(changed, baseline, valid[:, :3], total)


class _TypedFakeInterface:
    def __init__(self, centers: torch.Tensor, valid: torch.Tensor) -> None:
        self.module_descriptors = {"coordinates": centers, "valid": valid}
        self.environment_descriptors = {
            "coordinates": torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            "weights": torch.ones(4),
        }
        self.module_embeddings = torch.arange(centers.shape[0] * 4, dtype=torch.float32).reshape(-1, 4)
        self.environment_embeddings = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        self.environment_validity = torch.ones(4, dtype=torch.bool)
        self.valid = valid

    @property
    def module_validity(self) -> torch.Tensor:
        return self.valid

    def module_pair_weights(self, *, phase=None) -> torch.Tensor:
        value = self.valid[:, None] & self.valid[None, :]
        value.fill_diagonal_(False)
        return value.float()

    def receiver_source_weights(self, mechanism, receiver_coordinates, *, phase=None):
        if mechanism == "ME":
            return torch.full((self.valid.numel(), 4), 0.3)
        if mechanism == "EM":
            return torch.full((4, self.valid.numel()), 0.4)
        if mechanism == "QM":
            return torch.full((receiver_coordinates.shape[0], self.valid.numel()), 0.2)
        if mechanism == "QE":
            return torch.full((receiver_coordinates.shape[0], 4), 0.5)
        raise AssertionError(mechanism)


def test_thermal_packet_adapter_uses_typed_routes_and_exact_module_validity(tmp_path) -> None:
    path = tmp_path / "thermal.h5"
    _write_dataset(path)
    tasks = build_thermal_heat_tasks(path, split="train", sensor_count=8)
    scaler = ThermalHeatFeatureScaler.fit(tasks)
    batch = collate_thermal_heat_tasks(tasks, scaler=scaler)
    known = provider_known_inputs(batch)
    assert not hasattr(known, "clean_heat")
    assert not hasattr(known, "case_ids")

    built_heat: list[torch.Tensor] = []

    def make_interfaces(candidate_heat, known_inputs, condition):
        assert not hasattr(known_inputs, "clean_heat")
        built_heat.append(candidate_heat.detach().clone())
        return tuple(
            _TypedFakeInterface(known_inputs.module_centers[row], condition.module_valid[row])
            for row in range(candidate_heat.shape[0])
        )

    provider = ThermalCandidatePacketProvider(known, make_interfaces)
    state = torch.zeros_like(batch.clean_state)
    links = provider(state, batch.condition)
    assert links.module_source.shape == (2, 5, 5)
    assert links.sensor_source.shape == (2, 8, 5)
    assert links.module_environment.shape == (2, 5, 4)
    assert links.environment_module.shape == (2, 4, 5)
    assert links.sensor_environment.shape == (2, 8, 4)
    torch.testing.assert_close(links.module_environment, torch.full_like(links.module_environment, 0.3))
    torch.testing.assert_close(links.environment_module, torch.full_like(links.environment_module, 0.4))
    torch.testing.assert_close(links.sensor_environment, torch.full_like(links.sensor_environment, 0.5))
    torch.testing.assert_close(links.sensor_source, torch.full_like(links.sensor_source, 0.2))
    torch.testing.assert_close(built_heat[-1].sum(dim=1).squeeze(-1), batch.total_heat.squeeze(-1))
    assert not torch.equal(built_heat[-1], batch.clean_heat.unsqueeze(-1))

    invalid = _TypedFakeInterface(known.module_centers[0], torch.ones(5, dtype=torch.bool))
    with pytest.raises(ValueError, match="validity changed"):
        packet_links_from_interfaces(
            (invalid,),
            known.module_centers[:1],
            known.sensor_xy[:1],
            expected_module_valid=batch.condition.module_valid[:1],
        )


def test_matched_heat_training_replays_data_and_keeps_forward_state_frozen(tmp_path) -> None:
    path = tmp_path / "thermal.h5"
    _write_dataset(path)
    tasks = build_thermal_heat_tasks(path, split="train", sensor_count=8)
    scaler = ThermalHeatFeatureScaler.fit(tasks)
    frozen_forward = torch.nn.BatchNorm1d(4)
    frozen_forward.train()
    denoiser = ConditionalPacketDenoiser(
        design_dim=1,
        module_dim=13,
        sensor_dim=3,
        embedding_dim=4,
        hidden_dim=16,
        layers=1,
    )

    def provider_factory(known):
        def provider(state, condition):
            batch, modules, _ = state.shape
            mm = torch.ones(batch, modules, modules, device=state.device)
            mm.diagonal(dim1=1, dim2=2).zero_()
            mm = mm * condition.module_valid[:, :, None] * condition.module_valid[:, None, :]
            qm = condition.module_valid[:, None, :].expand(-1, condition.sensor_features.shape[1], -1).float()
            embeddings = state.new_zeros((batch, modules, 4))
            embeddings[..., 0] = known.module_centers[..., 0]
            return PacketLinks(mm, qm, embeddings)

        return provider

    saved_checkpoints = []
    attempted_rows = []
    fit_kwargs = dict(
        scaler=scaler,
        denoiser_template=denoiser,
        provider_factory=provider_factory,
        frozen_modules={"forward": frozen_forward},
        batch_size=2,
        steps=4,
        seed=77,
    )
    first_segment = train_matched_heat_diffusion(
        tasks,
        updates=2,
        checkpoint_callback=saved_checkpoints.append,
        attempt_callback=attempted_rows.append,
        checkpoint_every_updates=1,
        **fit_kwargs,
    )
    assert len(first_segment.graph_losses) == len(first_segment.dense_losses) == 2
    assert all(np.isfinite(first_segment.graph_losses)) and all(np.isfinite(first_segment.dense_losses))
    assert first_segment.updates_per_arm == 2
    resumed = train_matched_heat_diffusion(
        tasks,
        updates=3,
        resume_payload=saved_checkpoints[-1],
        checkpoint_every_updates=1,
        **fit_kwargs,
    )
    uninterrupted = train_matched_heat_diffusion(
        tasks,
        updates=3,
        checkpoint_every_updates=1,
        **fit_kwargs,
    )
    assert resumed.graph_losses == uninterrupted.graph_losses
    assert resumed.dense_losses == uninterrupted.dense_losses
    for arm in ("graph_model", "dense_model"):
        for key, value in getattr(resumed, arm).state_dict().items():
            assert torch.equal(value, getattr(uninterrupted, arm).state_dict()[key])
    assert [row["next_attempted_call"] for row in attempted_rows] == [1, 1, 2, 2]
    assert all(row["recorded_before_optimizer_step"] for row in attempted_rows)
    assert resumed.updates_per_arm == 3
    assert frozen_forward.training is False
    assert resumed.frozen_state_hashes_before == resumed.frozen_state_hashes_after

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch
import windfarm.inverse.bounded_velocity_completion as stable
from windfarm.data import NativeCase
from windfarm.geometry import EnvironmentRepresentation, SupportGeometry
from windfarm.inverse.packet_completion import (
    SENSOR_OBSERVED_INDICES,
    FixedNativeSensorPanel,
    WindCompletionTask,
    intended_wind_sensor_coordinates,
)

from honf_inverse_core.models.bounded_velocity import BoundedVelocityPacketDiffusion
from honf_inverse_core.models.frozen_packet_diffusion import PacketLinks


def test_fixed_observed_array_slot_14_maps_to_raw_sensor_index_22() -> None:
    coordinates = intended_wind_sensor_coordinates()
    raw_sensor_index = SENSOR_OBSERVED_INDICES[14]
    assert raw_sensor_index == 22
    np.testing.assert_allclose(coordinates[raw_sensor_index], [12.2, 5.0, 0.875], rtol=0, atol=1.0e-6)


class _TinyTrainingDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.design_dim = 2
        self.module_dim = 13
        self.sensor_dim = 6
        self.embedding_dim = 3
        self.output = torch.nn.Linear(2 + self.module_dim + self.sensor_dim + 3, 2)

    def forward(self, state, time, condition, links):
        del time
        source = links.module_source
        message = torch.bmm(source, links.module_embeddings) / source.sum(-1, keepdim=True).clamp_min(1.0)
        valid = condition.sensor_valid.to(condition.sensor_features.dtype).unsqueeze(-1)
        sensor_mean = (condition.sensor_features * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        sensor = sensor_mean[:, None].expand(-1, state.shape[1], -1)
        features = torch.cat((state, condition.module_features, sensor, message), dim=-1)
        return self.output(features) * condition.design_mask.unsqueeze(-1)


def _task(*, hidden_xy=(3.0, -2.0), row_index: int = 12) -> WindCompletionTask:
    lower = np.asarray([-20.0, -19.0, 0.0], dtype=np.float32)
    upper = np.asarray([30.0, 19.0, 6.25], dtype=np.float32)
    support = SupportGeometry(
        lower_m=lower * 80.0,
        upper_m=upper * 80.0,
        lower_D=lower,
        upper_D=upper,
        extent_D=upper - lower,
        volume_m3=float(np.prod((upper - lower) * 80.0)),
        volume_D3=float(np.prod(upper - lower)),
    )
    env_coords = np.zeros((2, 3), dtype=np.float32)
    environment = EnvironmentRepresentation(
        coords_D=env_coords,
        features=np.zeros((2, 4), dtype=np.float32),
        weights_D3=np.ones((2, 1), dtype=np.float32),
        token_shape=(1, 1, 2),
    )
    clean = np.asarray(
        [[1.0, 1.5, 0.875], [hidden_xy[0], hidden_xy[1], 0.875], [5.0, -1.0, 0.875]],
        dtype=np.float32,
    )
    visible = np.asarray([True, False, True])
    candidate_centers = clean.copy()
    candidate_centers[~visible, :2] = 0.0
    case = NativeCase(
        index=row_index,
        case=f"wind_{row_index}",
        layout=f"layout_{row_index}",
        layout_index=row_index,
        wind_direction_deg=285.0,
        n_turbines=3,
        run=None,
        support=support,
        environment=environment,
        module_centers=candidate_centers,
        module_present=np.ones(3, dtype=np.float32),
        module_features=np.arange(18, dtype=np.float32).reshape(3, 6),
        global_context=np.asarray([1.0, 0.2, 285.0, 3.0], dtype=np.float32),
    )
    intended = intended_wind_sensor_coordinates()
    sensors = FixedNativeSensorPanel(
        intended_coordinates_D=intended,
        native_coordinates_D=intended.copy(),
        reference_velocity_mps=np.arange(72, dtype=np.float32).reshape(24, 3) / 10.0,
        native_flat_indices=np.arange(24, dtype=np.int64),
        snap_distance_D=np.zeros(24, dtype=np.float32),
    )
    return WindCompletionTask(
        row_index=row_index,
        partition="train",
        clean_centers_D=clean,
        visible_mask=visible,
        template_case=case,
        sensors=sensors,
    )


def test_hidden_target_poison_changes_only_separate_supervision_target() -> None:
    clean_task = _task()
    poisoned = _task(hidden_xy=(9.0, -8.0))
    center = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    scale = np.asarray([2.0, 3.0, 4.0], dtype=np.float32)
    clean_target, clean_condition, clean_known = stable.wind_velocity_condition_from_task(
        clean_task,
        sensor_velocity_center_mps=center,
        sensor_velocity_scale_mps=scale,
    )
    poison_target, poison_condition, poison_known = stable.wind_velocity_condition_from_task(
        poisoned,
        sensor_velocity_center_mps=center,
        sensor_velocity_scale_mps=scale,
    )
    assert not torch.equal(clean_target, poison_target)
    target_free_condition = stable.wind_velocity_condition_from_known(
        clean_known,
        sensor_velocity_center_mps=center,
        sensor_velocity_scale_mps=scale,
    )
    for name in (
        "known_state",
        "design_mask",
        "module_valid",
        "module_features",
        "sensor_features",
        "sensor_valid",
        "design_lower",
        "design_upper",
    ):
        assert torch.equal(getattr(clean_condition, name), getattr(target_free_condition, name))
    for name in (
        "known_state",
        "design_mask",
        "module_valid",
        "module_features",
        "sensor_features",
        "sensor_valid",
        "design_lower",
        "design_upper",
    ):
        assert torch.equal(getattr(clean_condition, name), getattr(poison_condition, name))
    assert np.array_equal(clean_known.template_case.module_centers, poison_known.template_case.module_centers)
    assert np.array_equal(clean_known.observed_velocity_mps, poison_known.observed_velocity_mps)
    assert torch.all(clean_target.abs() <= 1.0) and torch.all(poison_target.abs() <= 1.0)
    # The visible states are normalized, while hidden entries are absent from the condition.
    torch.testing.assert_close(
        clean_condition.known_state[0, clean_task.visible_mask],
        clean_target[0, clean_task.visible_mask],
        rtol=0,
        atol=0,
    )
    assert torch.equal(clean_condition.known_state[0, ~clean_task.visible_mask], torch.zeros(1, 2))


def test_provider_links_are_unchanged_when_hidden_target_is_poisoned(monkeypatch) -> None:
    clean_task = _task()
    poisoned = _task(hidden_xy=(9.0, -8.0))
    center = np.zeros(3, dtype=np.float32)
    scale = np.ones(3, dtype=np.float32)
    _, clean_condition, clean_known = stable.wind_velocity_condition_from_task(
        clean_task, sensor_velocity_center_mps=center, sensor_velocity_scale_mps=scale
    )
    _, poison_condition, poison_known = stable.wind_velocity_condition_from_task(
        poisoned, sensor_velocity_center_mps=center, sensor_velocity_scale_mps=scale
    )

    class _Builder:
        def __init__(self) -> None:
            self.candidates: list[torch.Tensor] = []

        def __call__(self, centers, _known):
            self.candidates.append(centers.detach().clone())
            return SimpleNamespace(centers_D=centers.detach().clone())

    def fake_links(interface, observed_coordinates):
        centers = interface.centers_D
        modules = centers.shape[0]
        observed = torch.as_tensor(observed_coordinates, dtype=centers.dtype, device=centers.device)
        return PacketLinks(
            module_source=torch.ones(1, modules, modules, device=centers.device),
            sensor_source=torch.ones(1, observed.shape[0], modules, device=centers.device),
            module_embeddings=centers[None],
            module_coordinates=centers[None],
            sensor_coordinates=observed[None],
        )

    monkeypatch.setattr(stable, "packet_links_from_wind_interface", fake_links)
    clean_builder = _Builder()
    poison_builder = _Builder()
    clean_provider = stable.WindVelocityCandidatePacketProvider(clean_known, clean_builder)
    poison_provider = stable.WindVelocityCandidatePacketProvider(poison_known, poison_builder)
    state = clean_condition.known_state.clone()
    state[0, 1] = torch.tensor([1.7, -1.4])
    clean_links = clean_provider(state, clean_condition)
    poison_links = poison_provider(state, poison_condition)
    torch.testing.assert_close(clean_builder.candidates[0], poison_builder.candidates[0], rtol=0, atol=0)
    for name in (
        "module_source",
        "sensor_source",
        "module_embeddings",
        "module_coordinates",
        "sensor_coordinates",
    ):
        assert torch.equal(getattr(clean_links, name), getattr(poison_links, name))
    assert clean_links.module_coordinates.shape[-1] == 2
    assert clean_links.sensor_coordinates.shape[-1] == 2
    decoded = clean_builder.candidates[0][1, :2]
    assert torch.equal(decoded, torch.tensor([15.0, -15.0]))


def test_sensor_normalization_is_training_only_and_unique_row_weighted() -> None:
    first = _task(row_index=12)
    alternate_visible = np.asarray([False, True, True])
    alternate_template_centers = first.clean_centers_D.copy()
    alternate_template_centers[~alternate_visible, :2] = 0.0
    duplicate_mask = replace(
        first,
        visible_mask=alternate_visible,
        template_case=replace(first.template_case, module_centers=alternate_template_centers),
    )
    second = _task(row_index=13)
    center, scale = stable.fit_wind_sensor_velocity_transform((first, duplicate_mask, second))
    observed = np.asarray(SENSOR_OBSERVED_INDICES, dtype=np.int64)
    values = np.concatenate(
        [first.sensors.reference_velocity_mps[observed], second.sensors.reference_velocity_mps[observed]],
        axis=0,
    )
    np.testing.assert_allclose(center, values.mean(axis=0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(scale, np.maximum(values.std(axis=0), 0.05), rtol=1e-6, atol=1e-6)


def test_one_matched_update_has_equal_exposure_shared_links_and_endpoint_count(tmp_path) -> None:
    tasks: list[WindCompletionTask] = []
    for layout in range(24):
        base = _task(row_index=100 + layout)
        base = replace(base, template_case=replace(base.template_case, layout_index=layout))
        tasks.append(base)
        second_visible = np.asarray([False, True, True])
        second_centers = base.clean_centers_D.copy()
        second_centers[~second_visible, :2] = 0.0
        second = replace(
            base,
            visible_mask=second_visible,
            template_case=replace(base.template_case, module_centers=second_centers),
        )
        tasks.append(second)

    calls = {"count": 0}
    review_updates: list[int] = []

    def provider_factory(_known):
        def provider(state, condition):
            calls["count"] += 1
            base = state[..., 0].sigmoid()
            module_source = base[:, None, :].expand(-1, state.shape[1], -1).clone()
            module_source.diagonal(dim1=1, dim2=2).zero_()
            sensor_source = base[:, None, :].expand(-1, condition.sensor_features.shape[1], -1).clone()
            embeddings = torch.cat((state, torch.ones_like(state[..., :1])), dim=-1)
            return PacketLinks(module_source, sensor_source, embeddings)

        return provider

    frozen = torch.nn.Linear(1, 1)
    center, scale = stable.fit_wind_sensor_velocity_transform(tasks)
    attempt_log = tmp_path / "optimizer_attempts.jsonl"
    matched = stable.train_matched_wind_velocity(
        tasks,
        denoiser_template=_TinyTrainingDenoiser(),
        provider_factory=provider_factory,
        frozen_modules={"frozen": frozen},
        sensor_velocity_center_mps=center,
        sensor_velocity_scale_mps=scale,
        source_freeze_id="selected-G-u4910-forced-two-packet",
        updates=1,
        seed=47,
        device="cpu",
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_every=1,
        attempt_log_path=attempt_log,
        review_callback=lambda update, _graph, _dense: review_updates.append(update),
    )

    assert matched.updates_per_arm == 1
    assert matched.batch_size == 4
    assert matched.training_examples_per_arm == 4
    assert matched.endpoint_examples_per_arm == 1
    assert matched.shared_provider_calls == 4
    assert review_updates == [1]
    assert matched.optimizer_elapsed_seconds > 0.0
    assert matched.wall_elapsed_seconds >= matched.optimizer_elapsed_seconds
    assert calls["count"] == 4
    record = json.loads(attempt_log.read_text(encoding="utf-8").splitlines()[0])
    assert record["graph_loss"] >= 0.0 and record["dense_loss"] >= 0.0
    assert record["pure_noise_endpoint_examples"] == 1
    assert record["shared_provider_calls_this_update"] == 4
    assert record["examples_seen_per_arm"] == 4
    assert (tmp_path / "checkpoints" / "updates_000001.pt").is_file()


def test_bounded_overfit_diagnostic_uses_single_fresh_graph_model_without_sampling() -> None:
    tasks = []
    for layout in range(8):
        task = _task(row_index=300 + layout)
        tasks.append(replace(task, template_case=replace(task.template_case, layout_index=layout)))

    calls = {"count": 0}

    def provider_factory(_known):
        def provider(state, condition):
            calls["count"] += 1
            base = state[..., 0].sigmoid()
            module_source = base[:, None, :].expand(-1, state.shape[1], -1).clone()
            module_source.diagonal(dim1=1, dim2=2).zero_()
            sensor_source = base[:, None, :].expand(-1, condition.sensor_features.shape[1], -1).clone()
            embeddings = torch.cat((state, torch.ones_like(state[..., :1])), dim=-1)
            return PacketLinks(module_source, sensor_source, embeddings)

        return provider

    frozen = torch.nn.Linear(1, 1)
    center, scale = stable.fit_wind_sensor_velocity_transform(tasks)
    diagnostic = stable.train_wind_velocity_overfit_diagnostic(
        tasks,
        denoiser_template=_TinyTrainingDenoiser(),
        provider_factory=provider_factory,
        frozen_modules={"frozen": frozen},
        sensor_velocity_center_mps=center,
        sensor_velocity_scale_mps=scale,
        updates=1,
        seed=93,
        device="cpu",
    )
    assert diagnostic.updates == 1
    assert diagnostic.examples_seen == 4
    assert diagnostic.endpoint_examples == 1
    assert diagnostic.provider_calls == 4
    assert calls["count"] == 4 + 2 * len(tasks) * len(diagnostic.audit_times)
    assert set(diagnostic.clean_mse_before_by_time) == {"0.25", "0.5", "0.75", "1.0"}
    assert all(np.isfinite(tuple(diagnostic.clean_mse_after_by_time.values())))
    assert all(np.isfinite(tuple(diagnostic.observation_response_after_by_time.values())))
    assert not hasattr(diagnostic.model, "primary_checkpoint")


def test_primary_denoising_review_uses_fixed_shared_noise_and_one_provider_per_task_time() -> None:
    train_task = _task(row_index=701)
    development_task = replace(_task(row_index=702), partition="development")
    calls = {"count": 0}

    def provider_factory(_known):
        def provider(state, condition):
            calls["count"] += 1
            base = state[..., 0].clamp(-1.0, 1.0)
            module_source = base[:, None, :].expand(-1, state.shape[1], -1).clone()
            module_source.diagonal(dim1=1, dim2=2).zero_()
            sensor_source = base[:, None, :].expand(-1, condition.sensor_features.shape[1], -1).clone()
            embeddings = torch.cat((state, torch.ones_like(state[..., :1])), dim=-1)
            return PacketLinks(module_source, sensor_source, embeddings)

        return provider

    graph = BoundedVelocityPacketDiffusion(_TinyTrainingDenoiser())
    dense = BoundedVelocityPacketDiffusion(deepcopy(graph.denoiser))
    center, scale = stable.fit_wind_sensor_velocity_transform((train_task,))
    report = stable.evaluate_matched_wind_velocity_denoising(
        graph,
        dense,
        {"train": (train_task,), "development": (development_task,)},
        provider_factory=provider_factory,
        sensor_velocity_center_mps=center,
        sensor_velocity_scale_mps=scale,
        seed=112,
        audit_times=(0.25, 1.0),
    )

    assert calls["count"] == 4
    assert report["candidate_provider_calls_shared_between_arms"] == 4
    assert report["task_counts"] == {"train": 1, "development": 1}
    assert report["evaluation_type"].startswith("fixed-noise masked-v")
    assert report["coordinate_counts"] == {"train": 2, "development": 2}
    for arm_metrics in report["arms"].values():
        for partition_metrics in arm_metrics.values():
            assert set(partition_metrics) == {"0.25", "1.0"}
            for metrics in partition_metrics.values():
                assert all(np.isfinite(value) for value in metrics.values())
    assert graph.training and dense.training

"""No-leakage and matched-control checks for bounded Wind completion."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from honf_inverse_core.models.frozen_packet_diffusion import (
    ConditionalPacketDenoiser,
    SpatialConditionalPacketDenoiser,
)
from torch import nn

from windfarm.data import NativeCase
from windfarm.inverse.packet_completion import (
    WindCandidatePacketProvider,
    _degree_preserving_packet_rewire,
    candidate_centers_from_state,
    evaluate_matched_wind_completion,
    fixed_native_sensor_panel,
    hidden_set_error_D,
    make_wind_completion_task,
    native_clearance_design_bounds,
    permute_wind_task,
    task_native_support_bounds_D,
    train_matched_wind_diffusion,
    wind_condition_from_task,
    wind_geometry_validity,
    wind_sample_control,
)


def _case() -> NativeCase:
    diameter = 80.0
    x = np.linspace(-10.0, 20.0, 61, dtype=np.float32) * diameter
    y = np.linspace(-8.0, 8.0, 33, dtype=np.float32) * diameter
    z = np.linspace(0.05, 6.0, 13, dtype=np.float32) * diameter
    shape = (len(x), len(y), len(z))
    size = int(np.prod(shape))
    run = SimpleNamespace(
        x_m=x,
        y_m=y,
        z_m=z,
        nx=shape[0],
        ny=shape[1],
        nz=shape[2],
        U=np.column_stack(
            (
                np.linspace(7.0, 8.0, size, dtype=np.float32),
                np.zeros(size, dtype=np.float32),
                np.zeros(size, dtype=np.float32),
            )
        ),
    )
    centers = np.asarray([[-3.0, 2.0, 0.875], [2.0, -1.0, 0.875], [9.0, 4.0, 0.875]], dtype=np.float32)
    support = SimpleNamespace(
        lower_D=np.asarray((-10.0, -8.0, 0.05), dtype=np.float32),
        upper_D=np.asarray((20.0, 8.0, 6.0), dtype=np.float32),
    )
    return NativeCase(
        index=7,
        case="synthetic-test",
        layout="synthetic-test",
        layout_index=2,
        wind_direction_deg=270.0,
        n_turbines=3,
        run=run,
        support=support,
        environment=SimpleNamespace(),
        module_centers=centers,
        module_present=np.ones(3, dtype=np.float32),
        module_features=np.tile(np.asarray((0.5, 0.875), dtype=np.float32), (3, 1)),
        global_context=np.asarray((0, 0, 1, 3 / 30, 1, 0, 0, 0, 1, 1, 1), dtype=np.float32),
        diameter_m=diameter,
        hub_height_m=70.0,
    )


class _TinyInterface:
    def __init__(self, centers: torch.Tensor) -> None:
        self.module_descriptors = {"coordinates": centers}
        self.environment_descriptors = {"coordinates": centers.new_tensor([[0.0, 0.0, 0.875]])}
        self.module_embeddings = centers.new_ones((centers.shape[0], 8))
        self.environment_embeddings = centers.new_ones((1, 8))
        self.environment_validity = centers.new_ones((1,), dtype=torch.bool)

    def module_pair_weights(self) -> torch.Tensor:
        size = self.module_descriptors["coordinates"].shape[0]
        return 1.0 - torch.eye(size)

    def receiver_source_weights(self, mechanism: str, receivers: torch.Tensor) -> torch.Tensor:
        source_count = (
            self.module_descriptors["coordinates"].shape[0]
            if mechanism in {"MM", "EM", "QM"}
            else 1
        )
        return receivers.new_ones((receivers.shape[0], source_count))


def _provider_factory(known):
    return WindCandidatePacketProvider(known, lambda centers, _known: _TinyInterface(centers))


def test_fixed_sensors_and_known_provider_do_not_expose_hidden_layout() -> None:
    case = _case()
    panel = fixed_native_sensor_panel(case)
    assert panel.native_coordinates_D.shape == (24, 3)
    assert panel.reference_velocity_mps.shape == (24, 3)
    assert float(panel.snap_distance_D.max()) < 0.4
    task = make_wind_completion_task(case, partition="train", hidden_count=1, seed=19)
    task_lower, task_upper = task_native_support_bounds_D(task)
    np.testing.assert_array_equal(task_lower, case.support.lower_D[:2])
    np.testing.assert_array_equal(task_upper, case.support.upper_D[:2])
    another_mask = make_wind_completion_task(case, partition="train", hidden_count=2, seed=123)
    np.testing.assert_array_equal(task.sensors.intended_coordinates_D, another_mask.sensors.intended_coordinates_D)
    np.testing.assert_array_equal(task.sensors.native_flat_indices, another_mask.sensors.native_flat_indices)
    clean, condition, known = wind_condition_from_task(task)
    assert known.template_case.run is None
    assert known.template_case.receiver_anchor_coords is None
    assert condition.sensor_features.shape == (1, 16, 6)
    assert condition.module_features.shape == (1, 3, 16)
    assert not bool(condition.design_mask[0, task.visible_mask].any())

    provider = _provider_factory(known)
    assert not hasattr(provider.known, "held_velocity_mps")
    assert not hasattr(provider.known, "held_coordinates_D")
    assert not hasattr(provider.known, "observed_velocity_mps")
    links_before = provider(clean, condition)
    centers_before = provider.last_candidate_centers_D.clone()
    task.clean_centers_D[~task.visible_mask, :2] = 14.0  # poison forbidden clean target
    links_after = provider(clean, condition)
    torch.testing.assert_close(links_after.module_source, links_before.module_source)
    torch.testing.assert_close(provider.last_candidate_centers_D, centers_before)
    reconstructed = candidate_centers_from_state(clean, condition, known)
    torch.testing.assert_close(reconstructed[task.visible_mask, :2], condition.known_state[0, task.visible_mask].sigmoid() * 30 - 15, atol=1e-5, rtol=1e-5)


def test_train_only_centered_sensor_features_use_explicit_per_sensor_statistics() -> None:
    task = make_wind_completion_task(_case(), partition="train", hidden_count=1, seed=2111)
    center = np.full((16, 3), 7.0, dtype=np.float32)
    scale = np.full((16, 3), 2.0, dtype=np.float32)
    _, condition, known = wind_condition_from_task(
        task,
        sensor_velocity_center_mps=center,
        sensor_velocity_scale_mps=scale,
    )
    expected = (known.observed_velocity_mps - center) / scale
    np.testing.assert_allclose(condition.sensor_features[0, :, 3:].cpu().numpy(), expected)
    _, legacy, _ = wind_condition_from_task(task)
    np.testing.assert_allclose(
        legacy.sensor_features[0, :, 3:].cpu().numpy(),
        known.observed_velocity_mps / 9.0,
    )


def test_native_clearance_bounds_generate_inside_public_support_and_keep_visible() -> None:
    task = make_wind_completion_task(_case(), partition="train", hidden_count=1, seed=9)
    public_bounds = (
        np.asarray((-10.0, -8.0), dtype=np.float32),
        np.asarray((20.0, 8.0), dtype=np.float32),
    )
    lower, upper = native_clearance_design_bounds(
        public_bounds[0], public_bounds[1], rotor_radius_D=0.5,
    )
    np.testing.assert_allclose(lower, [-9.5, -7.5])
    np.testing.assert_allclose(upper, [15.0, 7.5])
    clean, condition, known = wind_condition_from_task(
        task, public_support_bounds_D=public_bounds,
    )
    condition.validate(design_dim=2, module_dim=16, sensor_dim=6)
    visible_tensor = torch.as_tensor(task.visible_mask)
    assert torch.equal(
        condition.known_state[0, visible_tensor],
        torch.zeros_like(condition.known_state[0, visible_tensor]),
    )
    torch.testing.assert_close(
        condition.module_features[0, visible_tensor, -3:-1],
        torch.as_tensor(task.clean_centers_D[task.visible_mask, :2])
        / torch.tensor((50.0, 38.0)),
    )
    state = clean.clone()
    state[:, ~torch.as_tensor(task.visible_mask), :] = 20.0
    candidate = candidate_centers_from_state(state, condition, known)
    hidden = candidate[~torch.as_tensor(task.visible_mask)]
    assert bool((hidden[:, :2] >= torch.as_tensor(lower)).all())
    assert bool((hidden[:, :2] <= torch.as_tensor(upper)).all())
    torch.testing.assert_close(
        candidate[torch.as_tensor(task.visible_mask)],
        torch.as_tensor(task.clean_centers_D[task.visible_mask]),
    )
    _, legacy_condition, _ = wind_condition_from_task(task)
    legacy = candidate_centers_from_state(state, legacy_condition, known)
    assert bool((legacy[~torch.as_tensor(task.visible_mask), 1] > upper[1]).any())
    task.clean_centers_D[~task.visible_mask, 1] = 8.0
    with pytest.raises(ValueError, match="Clean hidden target"):
        wind_condition_from_task(task, public_support_bounds_D=public_bounds)


def test_hidden_set_matching_geometry_and_permutation() -> None:
    task = make_wind_completion_task(_case(), partition="train", hidden_count=2, seed=22)
    permuted = permute_wind_task(task, seed=700)
    assert permuted.template_case.run is None
    assert int((~permuted.visible_mask).sum()) == 2
    generated = task.clean_centers_D.copy()
    generated[~task.visible_mask] = generated[~task.visible_mask][::-1]
    assert hidden_set_error_D(generated, task.clean_centers_D, task.visible_mask) == 0.0
    validity = wind_geometry_validity(task.clean_centers_D, rotor_radius_D=0.5)
    assert validity["inside_design_box"] and validity["rotors_nonoverlap"]
    outside = wind_geometry_validity(
        task.clean_centers_D,
        rotor_radius_D=0.5,
        support_lower_D=np.asarray((-1.0, -8.0, 0.05)),
        support_upper_D=np.asarray((20.0, 8.0, 6.0)),
    )
    assert not outside["inside_native_domain"]
    unknown = wind_geometry_validity(task.clean_centers_D, rotor_radius_D=0.5)
    assert unknown["inside_native_domain"] is None
    assert unknown["native_support_status"] == "unknown"
    generated[0, :2] = generated[1, :2]
    assert not wind_geometry_validity(generated, rotor_radius_D=0.5)["rotors_nonoverlap"]


def test_one_matched_diffusion_update_keeps_frozen_weights() -> None:
    task = make_wind_completion_task(_case(), partition="train", hidden_count=1, seed=8)
    torch.manual_seed(8)
    denoiser = ConditionalPacketDenoiser(
        design_dim=2, module_dim=16, sensor_dim=6, embedding_dim=8, hidden_dim=16, layers=1
    )
    frozen = nn.Linear(2, 2)
    before = {key: value.detach().clone() for key, value in frozen.state_dict().items()}
    public_bounds = (np.asarray((-10.0, -8.0)), np.asarray((20.0, 8.0)))
    seen_generation_bounds: list[tuple[list[float], list[float]]] = []

    def capturing_provider_factory(known):
        provider = _provider_factory(known)

        def capture(state, condition):
            assert condition.design_lower is not None and condition.design_upper is not None
            seen_generation_bounds.append((
                condition.design_lower[0].detach().cpu().tolist(),
                condition.design_upper[0].detach().cpu().tolist(),
            ))
            return provider(state, condition)

        return capture

    result = train_matched_wind_diffusion(
        [task],
        denoiser_template=denoiser,
        provider_factory=capturing_provider_factory,
        frozen_modules={"forward": frozen},
        updates=1,
        steps=4,
        device="cpu",
        public_support_bounds_for_task=lambda _task: public_bounds,
        public_support_input_label="common_public_domain_control",
    )
    assert result.updates_per_arm == 1
    assert result.organizer_calls == 1
    assert seen_generation_bounds == [([-9.5, -7.5], [15.0, 7.5])]
    assert np.isfinite(result.graph_losses[0]) and np.isfinite(result.dense_losses[0])
    assert result.frozen_state_hashes_before == result.frozen_state_hashes_after
    for key, value in frozen.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_spatial_conditioner_runs_small_matched_candidate_rebuild() -> None:
    task = make_wind_completion_task(_case(), partition="train", hidden_count=1, seed=18)
    torch.manual_seed(18)
    denoiser = SpatialConditionalPacketDenoiser(
        design_dim=2, module_dim=16, sensor_dim=6, embedding_dim=8,
        hidden_dim=16, layers=1, coordinate_dim=3,
    )
    result = train_matched_wind_diffusion(
        [task], denoiser_template=denoiser, provider_factory=_provider_factory,
        frozen_modules={"forward": nn.Linear(2, 2)}, updates=2, steps=3,
        device="cpu",
    )
    assert result.updates_per_arm == 2 and result.organizer_calls == 2
    assert all(np.isfinite(value) for value in (*result.graph_losses, *result.dense_losses))
    assert result.frozen_state_hashes_before == result.frozen_state_hashes_after


def test_matched_sampling_retains_all_attempts_and_identical_seeds() -> None:
    source = _case()
    training = make_wind_completion_task(source, partition="train", hidden_count=1, seed=8)
    development = [
        make_wind_completion_task(source, partition="development", hidden_count=1, seed=100 + k)
        for k in range(8)
    ]
    _, condition, _ = wind_condition_from_task(development[0])
    empty = wind_sample_control(condition, "no_observations")
    assert not bool(empty.sensor_valid.any())
    changed = wind_sample_control(condition, "changed_observations")
    torch.testing.assert_close(changed.sensor_features[:, :, :3], condition.sensor_features[:, :, :3])
    assert not torch.equal(changed.sensor_features[:, :, 3:], condition.sensor_features[:, :, 3:])
    denoiser = ConditionalPacketDenoiser(
        design_dim=2, module_dim=16, sensor_dim=6, embedding_dim=8, hidden_dim=16, layers=1
    )
    matched = train_matched_wind_diffusion(
        [training],
        denoiser_template=denoiser,
        provider_factory=_provider_factory,
        frozen_modules={"forward": nn.Linear(2, 2)},
        updates=1,
        steps=2,
        device="cpu",
    )
    rows = evaluate_matched_wind_completion(
        development,
        matched=matched,
        provider_factory=_provider_factory,
        surrogate_predictor=lambda _candidate, coordinates: np.zeros_like(coordinates),
        controls=("as_observed", "no_observations"),
        device="cpu",
        public_support_bounds_for_task=lambda _task: (
            np.asarray((-10.0, -8.0)), np.asarray((20.0, 8.0))
        ),
        public_support_input_label="common_public_domain_control",
    )
    assert len(rows) == 8 * 8 * 2 * 2
    for graph, dense in zip(rows[0::2], rows[1::2], strict=True):
        assert graph["arm"] == "I-G" and dense["arm"] == "I-dense"
        assert graph["public_support_input"] == dense["public_support_input"] == "common_public_domain_control"
        assert graph["generation_design_lower_D"] == dense["generation_design_lower_D"] == [-9.5, -7.5]
        assert graph["generation_design_upper_D"] == dense["generation_design_upper_D"] == [15.0, 7.5]
        assert graph["sample_seed"] == dense["sample_seed"]
        assert graph["organizer_calls"] == dense["organizer_calls"] == 3
        assert graph["full_forward_calls"] == dense["full_forward_calls"] == 1
        assert len(graph["trail_centers_D"]) == len(graph["trail_timesteps"])
        assert np.isfinite(graph["hidden_set_error_D"])
    seeds = {
        (row["row_index"], row["sample_index"], row["arm"], row["control"]): row["sample_seed"]
        for row in rows
    }
    for row in rows:
        assert seeds[(row["row_index"], row["sample_index"], row["arm"], "as_observed")] == seeds[
            (row["row_index"], row["sample_index"], row["arm"], "no_observations")
        ]


def test_inverse_checkpoint_resume_preserves_noise_and_case_stream(tmp_path) -> None:
    task = make_wind_completion_task(_case(), partition="train", hidden_count=1, seed=81)
    denoiser = ConditionalPacketDenoiser(
        design_dim=2, module_dim=16, sensor_dim=6, embedding_dim=8, hidden_dim=16, layers=1
    )
    frozen = {"forward": nn.Linear(2, 2)}
    common = {
        "denoiser_template": denoiser,
        "provider_factory": _provider_factory,
        "frozen_modules": frozen,
        "steps": 2,
        "seed": 19,
        "device": "cpu",
    }
    uninterrupted = train_matched_wind_diffusion([task], updates=2, **common)
    checkpoints = tmp_path / "inverse"
    attempts = tmp_path / "inverse_optimizer_attempts.jsonl"
    train_matched_wind_diffusion(
        [task], updates=1, checkpoint_dir=checkpoints, checkpoint_every=1,
        attempt_log_path=attempts, **common
    )
    resumed = train_matched_wind_diffusion(
        [task],
        updates=2,
        checkpoint_dir=checkpoints,
        checkpoint_every=1,
        resume_checkpoint=checkpoints / "updates_000001.pt",
        attempt_log_path=attempts,
        **common,
    )
    attempted_rows = [json.loads(line) for line in attempts.read_text().splitlines()]
    assert [(row["cumulative_update"], row["arm"]) for row in attempted_rows] == [
        (1, "I-G"), (1, "I-dense"), (2, "I-G"), (2, "I-dense"),
    ]
    assert resumed.graph_losses == uninterrupted.graph_losses
    assert resumed.dense_losses == uninterrupted.dense_losses
    for arm in ("graph_model", "dense_model"):
        baseline = getattr(uninterrupted, arm).state_dict()
        restarted = getattr(resumed, arm).state_dict()
        for key, value in baseline.items():
            torch.testing.assert_close(value, restarted[key], rtol=0, atol=0)
    evaluation_only = train_matched_wind_diffusion(
        [task], updates=2, checkpoint_dir=checkpoints,
        resume_checkpoint=checkpoints / "updates_000002.pt",
        attempt_log_path=attempts, **common,
    )
    assert len(attempts.read_text().splitlines()) == 4
    assert evaluation_only.graph_losses == uninterrupted.graph_losses
    assert evaluation_only.dense_losses == uninterrupted.dense_losses
    for arm in ("graph_model", "dense_model"):
        expected = getattr(resumed, arm).state_dict()
        actual = getattr(evaluation_only, arm).state_dict()
        for key, value in expected.items():
            torch.testing.assert_close(value, actual[key], rtol=0, atol=0)


def test_packet_incidence_rewire_preserves_packet_size_and_source_degree() -> None:
    permissions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    rewired, swaps = _degree_preserving_packet_rewire(permissions, seed=2)
    assert swaps == 1
    assert not torch.equal(rewired, permissions)
    torch.testing.assert_close((rewired > 0).sum(dim=1), (permissions > 0).sum(dim=1))
    torch.testing.assert_close((rewired > 0).sum(dim=0), (permissions > 0).sum(dim=0))
    full, no_swaps = _degree_preserving_packet_rewire(torch.ones_like(permissions), seed=2)
    assert no_swaps == 0
    torch.testing.assert_close(full, torch.ones_like(permissions))

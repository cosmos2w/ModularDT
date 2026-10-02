from __future__ import annotations

import pytest
import torch
from torch import nn

from honf_inverse_core.models.bounded_velocity import (
    VELOCITY_CHECKPOINT_FORMAT,
    VELOCITY_PREDICTION_TYPE,
    VELOCITY_STATE_PARAMETERIZATION,
    BoundedVelocityPacketDiffusion,
    cosine_velocity_coefficients,
    require_velocity_checkpoint,
)
from honf_inverse_core.models.frozen_packet_diffusion import DiffusionCondition, PacketLinks


class _FixedVelocityDenoiser(nn.Module):
    def __init__(self, value: float = -10.0) -> None:
        super().__init__()
        self.design_dim = 2
        self.module_dim = 1
        self.sensor_dim = 1
        self.value = float(value)
        self.seen_states: list[torch.Tensor] = []

    def forward(self, state, time, condition, links):
        self.seen_states.append(state.detach().clone())
        return torch.full_like(state, self.value) * condition.design_mask.unsqueeze(-1)


class _OracleVelocityDenoiser(nn.Module):
    """CPU-only oracle used to check the complete deterministic DDIM path."""

    def __init__(self, clean: torch.Tensor) -> None:
        super().__init__()
        self.design_dim = 2
        self.module_dim = 1
        self.sensor_dim = 1
        self.register_buffer("clean", clean.clone())

    def forward(self, state, time, condition, links):
        del links
        a, b = cosine_velocity_coefficients(time)
        a = a.reshape(-1, 1, 1)
        b = b.reshape(-1, 1, 1)
        generated = condition.design_mask.unsqueeze(-1)
        clean = torch.where(generated, self.clean, condition.known_state)
        noise = torch.where(generated, (state - a * clean) / b, torch.zeros_like(state))
        return (a * noise - b * clean) * generated


def _condition(module_count: int, *, invalid_last: bool = False) -> DiffusionCondition:
    known = torch.zeros(1, module_count, 2)
    valid = torch.ones(1, module_count, dtype=torch.bool)
    if module_count > 1:
        known[0, 0] = torch.tensor([0.25, -0.5])
    design = torch.zeros(1, module_count, dtype=torch.bool)
    design[0, -2 if invalid_last else -1] = True
    if invalid_last:
        valid[0, -1] = False
    return DiffusionCondition(
        known_state=known,
        design_mask=design,
        module_valid=valid,
        module_features=torch.zeros(1, module_count, 1),
        sensor_features=torch.zeros(1, 1, 1),
        sensor_valid=torch.ones(1, 1, dtype=torch.bool),
    )


def _links(state: torch.Tensor) -> PacketLinks:
    batch, modules, _ = state.shape
    return PacketLinks(
        module_source=torch.ones(batch, modules, modules),
        sensor_source=torch.ones(batch, 1, modules),
        module_embeddings=torch.zeros(batch, modules, 3),
    )


def test_velocity_transform_reconstructs_clean_and_noise_at_both_endpoints() -> None:
    times = torch.tensor([0.0, 0.37, 1.0], dtype=torch.float64)
    a, b = cosine_velocity_coefficients(times)
    assert torch.equal(a[[0, 2]], torch.tensor([1.0, 0.0], dtype=torch.float64))
    assert torch.equal(b[[0, 2]], torch.tensor([0.0, 1.0], dtype=torch.float64))
    clean = torch.tensor([[0.2, -0.7], [0.4, 0.1], [-0.8, 0.9]], dtype=torch.float64)
    noise = torch.tensor([[-0.3, 0.8], [0.6, 0.2], [0.1, -0.5]], dtype=torch.float64)
    noisy = a[:, None] * clean + b[:, None] * noise
    velocity = a[:, None] * noise - b[:, None] * clean
    clean_recovered = a[:, None] * noisy - b[:, None] * velocity
    noise_recovered = b[:, None] * noisy + a[:, None] * velocity
    torch.testing.assert_close(clean_recovered, clean, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(noise_recovered, noise, atol=1e-12, rtol=1e-12)
    assert torch.equal(velocity[-1], -clean[-1])


def test_training_noise_is_zero_on_visible_and_invalid_slots() -> None:
    condition = _condition(4, invalid_last=True)
    model = BoundedVelocityPacketDiffusion(_FixedVelocityDenoiser(), steps=20)
    clean = torch.zeros_like(condition.known_state)
    generated_index = int(condition.design_mask[0].nonzero()[0])
    clean[0, generated_index] = torch.tensor([0.3, -0.2])
    noise = torch.full_like(clean, 7.0)
    time = torch.tensor([1.0])
    noisy, velocity = model.noisy_state_and_velocity_target(clean, condition, time, noise)
    torch.testing.assert_close(noisy[0, 0], condition.known_state[0, 0], rtol=0, atol=0)
    torch.testing.assert_close(noisy[0, generated_index], torch.full((2,), 7.0), rtol=0, atol=0)
    torch.testing.assert_close(velocity[0, generated_index], -clean[0, generated_index], rtol=0, atol=0)
    torch.testing.assert_close(noisy[0, 3], torch.zeros(2), rtol=0, atol=0)
    torch.testing.assert_close(velocity[0, 3], torch.zeros(2), rtol=0, atol=0)


@pytest.mark.parametrize(("module_count", "invalid_last"), [(1, False), (4, False), (5, True)])
def test_terminal_projection_fraction_counts_generated_coordinates_only(
    module_count: int,
    invalid_last: bool,
) -> None:
    condition = _condition(module_count, invalid_last=invalid_last)
    denoiser = _FixedVelocityDenoiser()
    model = BoundedVelocityPacketDiffusion(denoiser, steps=20)
    initial = torch.zeros_like(condition.known_state)
    generated_index = int(condition.design_mask[0].nonzero()[0])
    initial[0, generated_index] = torch.tensor([2.0, -2.0])
    provider_inputs: list[torch.Tensor] = []

    def provider(state, _condition):
        provider_inputs.append(state.detach().clone())
        return _links(state)

    trail = model.sample(condition, provider, initial_noise=initial)
    # The denoiser sees the raw state, while the provider sees its clipped proxy.
    assert torch.equal(denoiser.seen_states[0][0, generated_index], initial[0, generated_index])
    assert torch.equal(provider_inputs[0][0, generated_index], torch.tensor([1.0, -1.0]))
    assert trail.projection_coordinate_fraction == 1.0
    assert bool((trail.final_state[0, generated_index].abs() > 1.0).all())
    assert bool((trail.projected_final_state[0, generated_index].abs() <= 1.0).all())
    assert trail.generated_coordinate_count == 2
    assert len(trail.states) == 21 and len(trail.clean_estimates) == 20
    assert trail.state_timesteps == tuple(float(step) / 20.0 for step in range(20, -1, -1))
    assert trail.clean_estimate_timesteps == tuple(float(step) / 20.0 for step in range(20, 0, -1))
    for state in trail.states:
        torch.testing.assert_close(
            state[0, ~condition.design_mask[0]],
            condition.known_state[0, ~condition.design_mask[0]],
            rtol=0,
            atol=0,
        )


def test_cpu_oracle_recovers_bounded_target_through_all_twenty_steps() -> None:
    condition = _condition(4)
    generated_index = int(condition.design_mask[0].nonzero()[0])
    clean = torch.zeros_like(condition.known_state)
    clean[0, generated_index] = torch.tensor([0.37, -0.62])
    initial_noise = torch.zeros_like(clean)
    initial_noise[0, generated_index] = torch.tensor([-0.43, 0.81])
    model = BoundedVelocityPacketDiffusion(_OracleVelocityDenoiser(clean), steps=20)
    times_seen: list[int] = []

    trail = model.sample(
        condition,
        lambda state, _condition: _links(state),
        initial_noise=initial_noise,
        progress_callback=lambda step, *_args: times_seen.append(step),
    )

    assert times_seen == list(range(21))
    generated = condition.design_mask.unsqueeze(-1)
    complete_clean = torch.where(generated, clean, condition.known_state)
    complete_noise = torch.where(generated, initial_noise, torch.zeros_like(initial_noise))
    for index, timestep in enumerate(trail.state_timesteps):
        a, b = cosine_velocity_coefficients(torch.tensor([timestep]))
        expected = torch.where(
            generated,
            a.reshape(1, 1, 1) * complete_clean + b.reshape(1, 1, 1) * complete_noise,
            condition.known_state,
        )
        torch.testing.assert_close(trail.states[index], expected, atol=2e-6, rtol=2e-6)
    for estimate in trail.clean_estimates:
        expected_clean = torch.where(generated, complete_clean, condition.known_state)
        torch.testing.assert_close(estimate, expected_clean, atol=2e-6, rtol=2e-6)
    expected_final = torch.where(generated, complete_clean, condition.known_state)
    torch.testing.assert_close(trail.final_state, expected_final, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(trail.projected_final_state, expected_final, atol=2e-6, rtol=2e-6)
    assert trail.projection_coordinate_fraction == 0.0


def test_old_epsilon_checkpoint_cannot_pass_velocity_schema() -> None:
    require_velocity_checkpoint(
        {
            "checkpoint_format": VELOCITY_CHECKPOINT_FORMAT,
            "prediction_type": VELOCITY_PREDICTION_TYPE,
            "state_parameterization": VELOCITY_STATE_PARAMETERIZATION,
        }
    )
    with pytest.raises(ValueError, match="bounded normalized-coordinate velocity"):
        require_velocity_checkpoint({"task": "wind_position_completion", "model_state_dict": {}})
    with pytest.raises(ValueError, match="prediction target"):
        require_velocity_checkpoint(
            {
                "checkpoint_format": VELOCITY_CHECKPOINT_FORMAT,
                "prediction_type": "epsilon",
                "state_parameterization": VELOCITY_STATE_PARAMETERIZATION,
            }
        )

from dataclasses import replace

import pytest
import torch
from torch import nn

from honf_inverse_core.models.bounded_velocity import (
    BoundedVelocityPacketDiffusion,
    cosine_velocity_coefficients,
    require_velocity_checkpoint,
)
from honf_inverse_core.models.centered_simplex_velocity import (
    HEAT_CHECKPOINT_FORMAT,
    HEAT_PREDICTION_TYPE,
    HEAT_STATE_PARAMETERIZATION,
    CenteredSimplexVelocityDiffusion,
    HeatSimplexCondition,
    centered_heat_noise,
    decode_heat_simplex,
    encode_heat_simplex,
    heat_provider_proxy,
    require_heat_simplex_checkpoint,
)
from honf_inverse_core.models.frozen_packet_diffusion import PacketLinks


def condition(modules=4, batch=1, invalid_last=False):
    valid = torch.ones(batch, modules, dtype=torch.bool)
    if invalid_last:
        valid[:, -1] = False
    return HeatSimplexCondition(
        known_state=torch.zeros(batch, modules, 1), design_mask=valid.clone(), module_valid=valid,
        module_features=torch.randn(batch, modules, 2), sensor_features=torch.randn(batch, 3, 2),
        sensor_valid=torch.ones(batch, 3, dtype=torch.bool), total_heat=torch.full((batch, 1), 100.),
    )


def links(state, _condition=None):
    batch, modules, _ = state.shape
    return PacketLinks(torch.ones(batch, modules, modules), torch.ones(batch, 3, modules), torch.ones(batch, modules, 4))


class Denoiser(nn.Module):
    design_dim, module_dim, sensor_dim = 1, 2, 2

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self.seen = []

    def forward(self, state, time, condition, access):
        self.seen.append(state.detach().clone())
        return self.layer(state)


class Oracle(Denoiser):
    def __init__(self, target):
        super().__init__()
        self.register_buffer('target', target)

    def forward(self, state, time, condition, access):
        a, b = cosine_velocity_coefficients(time)
        a, b = a[:, None, None], b[:, None, None]
        epsilon = (state - a * self.target) / b
        return a * epsilon - b * self.target


def test_concentrated_heat_above_position_bounds_roundtrips_with_physical_gradients():
    given = condition(16)
    heat = torch.zeros_like(given.known_state)
    heat[:, 0] = 100
    heat.requires_grad_()
    state = encode_heat_simplex(heat, given.total_heat, given.module_valid)
    assert state.max() > 1 and state.sum().abs() < 1e-6
    decoded = decode_heat_simplex(state, given.total_heat, given.module_valid)
    torch.testing.assert_close(decoded, heat)
    raw = decode_heat_simplex(state, given.total_heat, given.module_valid, project=False)
    gradient = torch.autograd.grad((raw * torch.arange(16.)[None, :, None]).sum(), heat)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    model = CenteredSimplexVelocityDiffusion(Denoiser())
    model.noisy_state_and_velocity_target(state, given, torch.tensor([.4]), torch.randn_like(state))
    # The separate historical position contract remains strict.
    with pytest.raises(ValueError, match=r'\[-1, 1\]'):
        BoundedVelocityPacketDiffusion(Denoiser()).noisy_state_and_velocity_target(state, given, torch.tensor([.4]), torch.randn_like(state))


def test_zero_sum_noise_padding_and_velocity_endpoints():
    given = condition(5, batch=3, invalid_last=True)
    heat = torch.zeros_like(given.known_state)
    heat[:, :4] = 25
    clean = encode_heat_simplex(heat, given.total_heat, given.module_valid)
    noise = torch.randn_like(clean)
    epsilon = centered_heat_noise(clean, given.module_valid, noise)
    assert epsilon[:, -1].abs().sum() == 0
    torch.testing.assert_close(epsilon.sum(1), torch.zeros(3, 1), atol=1e-6, rtol=0)
    times = torch.tensor([0., .4, 1.])
    noisy, velocity = CenteredSimplexVelocityDiffusion(Denoiser()).noisy_state_and_velocity_target(clean, given, times, noise)
    a, b = cosine_velocity_coefficients(times)
    torch.testing.assert_close(a[:, None, None] * noisy - b[:, None, None] * velocity, clean, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(b[:, None, None] * noisy + a[:, None, None] * velocity, epsilon, atol=1e-6, rtol=1e-6)


def test_projection_distinguishes_raw_negative_heat_and_feasible_proxy():
    given = condition()
    state = torch.tensor([[[10.], [-4.], [-3.], [-3.]]])
    raw = decode_heat_simplex(state, given.total_heat, given.module_valid, project=False)
    proxy = heat_provider_proxy(state, given)
    assert raw.min() < 0
    assert proxy.min() >= 0
    torch.testing.assert_close(raw.sum(1), given.total_heat)
    torch.testing.assert_close(proxy.sum(1), given.total_heat)
    torch.testing.assert_close(proxy[0, 0], torch.tensor([100.]))
    # Legitimate concentrated allocations are never clipped to r<=1.
    assert encode_heat_simplex(proxy, given.total_heat, given.module_valid).max() > 1


def test_oracle_complete_reverse_trail_recovers_heat_and_exact_time_indices():
    given = condition(4)
    heat = torch.tensor([[[10.], [20.], [30.], [40.]]])
    clean = encode_heat_simplex(heat, given.total_heat, given.module_valid)
    initial = torch.randn_like(clean)
    epsilon = centered_heat_noise(clean, given.module_valid, initial)
    seen = []
    def provider(value, context):
        seen.append(value.clone())
        assert value.min() >= 0
        torch.testing.assert_close(value.sum(1), context.total_heat, atol=1e-5, rtol=1e-5)
        return links(value)
    model = CenteredSimplexVelocityDiffusion(Oracle(clean), steps=20)
    trail = model.sample(given, provider, initial_noise=initial)
    assert len(trail.states) == 21 and len(trail.clean_estimates) == 20
    assert trail.state_timesteps == tuple(i / 20 for i in range(20, -1, -1))
    assert trail.clean_estimate_timesteps == tuple(i / 20 for i in range(20, 0, -1))
    assert trail.organizer_calls == 20
    for state, time in zip(trail.states, trail.state_timesteps):
        a, b = cosine_velocity_coefficients(torch.tensor([time]))
        torch.testing.assert_close(state, a[:, None, None] * clean + b[:, None, None] * epsilon, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(trail.raw_final_heat, heat, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(trail.projected_final_heat, heat, atol=1e-5, rtol=1e-5)


def test_one_module_is_deterministic_without_provider_or_denoiser_calls():
    given = condition(1)
    denoiser = Denoiser()
    model = CenteredSimplexVelocityDiffusion(denoiser, steps=3)
    def forbidden(*args):
        raise AssertionError('M=1 needs no conditional inference')
    clean = encode_heat_simplex(given.total_heat[:, None], given.total_heat, given.module_valid)
    assert clean.abs().sum() == 0
    loss = model.training_loss(clean, given, forbidden, time=torch.tensor([.5]), noise=torch.ones_like(clean))
    loss.backward()
    assert loss == 0 and not denoiser.seen
    trail = model.sample(given, forbidden, initial_noise=torch.ones_like(clean))
    assert trail.organizer_calls == 0 and trail.free_dimensions.tolist() == [0]
    torch.testing.assert_close(trail.projected_final_heat, torch.tensor([[[100.]]]))


def test_training_updates_denoiser_and_shared_links_keep_forward_frozen():
    given = condition()
    heat = torch.tensor([[[10.], [20.], [30.], [40.]]])
    clean = encode_heat_simplex(heat, given.total_heat, given.module_valid)
    module_embeddings = torch.ones(1, 4, 4, requires_grad=True)
    shared = replace(links(clean), module_embeddings=module_embeddings)
    model = CenteredSimplexVelocityDiffusion(Denoiser())
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    before = model.denoiser.layer.weight.detach().clone()
    noise, time = torch.randn_like(clean), torch.tensor([.4])
    optimizer.zero_grad(set_to_none=True)
    loss = model.training_loss(clean, given, time=time, noise=noise, links=shared)
    loss.backward()
    assert module_embeddings.grad is None
    assert model.denoiser.layer.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.denoiser.layer.weight)
    assert torch.isfinite(model.training_loss(clean, given, time=time, noise=noise, links=shared, dense=True))


def test_one_candidate_provider_call_is_shared_between_matched_training_arms():
    given = condition()
    clean = encode_heat_simplex(torch.tensor([[[10.], [20.], [30.], [40.]]]), given.total_heat, given.module_valid)
    model = CenteredSimplexVelocityDiffusion(Denoiser())
    noise, time = torch.randn_like(clean), torch.tensor([.6])
    calls = []
    def provider(physical_heat, known):
        calls.append(physical_heat)
        return links(physical_heat)
    noisy, _ = model.noisy_state_and_velocity_target(clean, given, time, noise)
    shared = model.prepare_candidate_links(noisy, given, provider)
    graph = model.training_loss(clean, given, time=time, noise=noise, links=shared)
    full = model.training_loss(clean, given, time=time, noise=noise, links=shared, dense=True)
    assert len(calls) == 1
    assert not shared.module_embeddings.requires_grad
    assert torch.isfinite(graph + full)


def test_heat_checkpoint_contract_is_separate_and_invalid_conditions_rejected():
    payload = {'checkpoint_format': HEAT_CHECKPOINT_FORMAT, 'prediction_type': HEAT_PREDICTION_TYPE, 'state_parameterization': HEAT_STATE_PARAMETERIZATION}
    require_heat_simplex_checkpoint(payload)
    with pytest.raises(ValueError):
        require_velocity_checkpoint(payload)
    with pytest.raises(ValueError):
        require_heat_simplex_checkpoint({'checkpoint_format': 'wind_normalized_xy_velocity_v1'})
    given = condition()
    for invalid in (replace(given, total_heat=torch.tensor([[-1.]])), replace(given, known_state=torch.ones_like(given.known_state)), replace(given, design_mask=torch.tensor([[1, 1, 1, 0]], dtype=torch.bool))):
        with pytest.raises(ValueError):
            CenteredSimplexVelocityDiffusion(Denoiser()).sample(invalid, links)


def test_zero_total_canonical_state_and_projection_stay_zero():
    given = replace(condition(), total_heat=torch.zeros(1, 1))
    heat = torch.zeros_like(given.known_state)
    clean = encode_heat_simplex(heat, given.total_heat, given.module_valid)
    assert clean.abs().sum() == 0
    torch.testing.assert_close(decode_heat_simplex(torch.randn_like(clean), given.total_heat, given.module_valid), heat)

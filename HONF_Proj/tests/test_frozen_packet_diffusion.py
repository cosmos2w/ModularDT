"""Behavioral checks for the bounded frozen-packet inverse pilot."""

from __future__ import annotations

import os
from dataclasses import replace

import pytest
import torch

from honf_inverse_core.models.frozen_packet_diffusion import (
    ConditionalPacketDenoiser,
    DiffusionCondition,
    FrozenPacketDiffusion,
    PacketLinks,
    SpatialConditionalPacketDenoiser,
    dense_access,
    heat_from_logits,
)


def _condition(task: str) -> DiffusionCondition:
    batch, modules, sensors = 2, 4, 3
    design_dim = 1 if task == "heat" else 2
    known = torch.zeros(batch, modules, design_dim)
    design_mask = torch.ones(batch, modules, dtype=torch.bool)
    if task == "position":
        design_mask[:, :2] = False
        known[:, 0] = torch.tensor([0.2, 0.3])
        known[:, 1] = torch.tensor([0.7, 0.8])
    return DiffusionCondition(
        known_state=known,
        design_mask=design_mask,
        module_valid=torch.ones(batch, modules, dtype=torch.bool),
        module_features=torch.randn(batch, modules, 3),
        sensor_features=torch.randn(batch, sensors, 4),
        sensor_valid=torch.ones(batch, sensors, dtype=torch.bool),
    )


class CandidateProvider:
    def __init__(self) -> None:
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.calls: list[torch.Tensor] = []
        self.forbidden_clean_cache = torch.tensor(float("nan"))

    def __call__(self, candidate: torch.Tensor, condition: DiffusionCondition) -> PacketLinks:
        self.calls.append(candidate.clone())
        batch, modules, _ = candidate.shape
        sensors = condition.sensor_features.shape[1]
        # Depend on the current candidate. The poisoned clean cache above is
        # intentionally inaccessible through the provider's call arguments.
        base = torch.sigmoid(candidate[..., 0] * self.scale)
        mm = base[:, None, :].expand(batch, modules, modules).clone()
        mm.diagonal(dim1=1, dim2=2).zero_()
        sm = base[:, None, :].expand(batch, sensors, modules).clone()
        embed = candidate.mean(dim=-1, keepdim=True).expand(batch, modules, 2)
        return PacketLinks(mm, sm, embed)


def _model(task: str) -> FrozenPacketDiffusion:
    denoiser = ConditionalPacketDenoiser(
        design_dim=1 if task == "heat" else 2,
        module_dim=3,
        sensor_dim=4,
        embedding_dim=2,
        hidden_dim=16,
        layers=2,
    )
    return FrozenPacketDiffusion(denoiser, task=task, steps=4)


def test_sample_progress_callback_persists_terminal_before_later_interruption(tmp_path) -> None:
    torch.manual_seed(37)
    condition = _condition("position")
    model = _model("position")
    provider = CandidateProvider()
    initial_noise = torch.randn_like(condition.known_state)
    terminal_path = tmp_path / "first-terminal.pt"
    callback_times: list[int] = []

    def save_first_trail(timestep: int, state: torch.Tensor) -> None:
        assert not state.requires_grad
        assert state.grad_fn is None
        callback_times.append(timestep)
        if timestep == 0:
            temporary = terminal_path.with_suffix(".tmp")
            torch.save(
                {"timestep": timestep, "state": state.detach().cpu().clone()},
                temporary,
            )
            os.replace(temporary, terminal_path)

    first = model.sample(
        condition,
        provider,
        initial_noise=initial_noise,
        save_every=2,
        progress_callback=save_first_trail,
    )
    assert callback_times == [4, 3, 2, 1, 0]
    assert first.timesteps == (4, 2, 0)
    saved_before = torch.load(terminal_path, map_location="cpu", weights_only=True)
    torch.testing.assert_close(saved_before["state"], first.final_state.detach().cpu())

    def interrupt_second_trail(timestep: int, _state: torch.Tensor) -> None:
        if timestep == model.steps - 1:
            raise RuntimeError("simulated interruption after one reverse update")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        model.sample(
            condition,
            provider,
            initial_noise=initial_noise,
            progress_callback=interrupt_second_trail,
        )

    saved_after = torch.load(terminal_path, map_location="cpu", weights_only=True)
    assert saved_after["timestep"] == 0
    torch.testing.assert_close(saved_after["state"], saved_before["state"])


def test_heat_allocation_and_frozen_graph_training_gradient() -> None:
    torch.manual_seed(23)
    condition = _condition("heat")
    model = _model("heat")
    provider = CandidateProvider()
    logits = torch.tensor([[[0.2], [0.4], [-0.1], [0.3]]] * 2)
    allocation = heat_from_logits(logits, torch.tensor([[5.0], [7.0]]), condition.module_valid)
    assert torch.all(allocation >= 0)
    torch.testing.assert_close(allocation.sum(dim=1), torch.tensor([[5.0], [7.0]]))

    loss = model.training_loss(
        logits,
        condition,
        provider,
        timesteps=torch.tensor([1, 2]),
        noise=torch.randn_like(logits),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    assert provider.scale.grad is None
    assert len(provider.calls) == 1

    initial = torch.randn_like(logits)
    trail = model.sample(condition, provider, initial_noise=initial, save_every=1)
    assert trail.organizer_calls == model.steps
    assert len(provider.calls) == 1 + model.steps
    assert not torch.equal(provider.calls[1], provider.calls[-1])
    torch.testing.assert_close(trail.final_state.sum(dim=1), torch.zeros(2, 1), atol=5e-5, rtol=0)
    assert torch.isfinite(trail.final_state).all()


def test_position_visible_modules_fixed_and_matched_dense_links() -> None:
    torch.manual_seed(29)
    condition = _condition("position")
    model = _model("position")
    provider = CandidateProvider()
    noise = torch.randn_like(condition.known_state)
    graph_trail = model.sample(condition, provider, initial_noise=noise)
    dense_trail = model.sample(condition, provider, initial_noise=noise, dense=True)
    torch.testing.assert_close(graph_trail.final_state[:, :2], condition.known_state[:, :2])
    torch.testing.assert_close(dense_trail.final_state[:, :2], condition.known_state[:, :2])
    assert not torch.equal(graph_trail.final_state[:, 2:], dense_trail.final_state[:, 2:])

    links = provider(noise, condition)
    full = dense_access(links, condition)
    assert torch.equal(full.module_source.diagonal(dim1=1, dim2=2), torch.zeros(2, 4))
    torch.testing.assert_close(full.module_embeddings, links.module_embeddings)


def test_heat_denoiser_respects_physical_module_permutation() -> None:
    torch.manual_seed(31)
    condition = _condition("heat")
    model = _model("heat")
    provider = CandidateProvider()
    state = torch.randn_like(condition.known_state)
    links = provider(state, condition)
    time = torch.tensor([0.3, 0.7])
    original = model.denoiser(state, time, condition, links)

    order = torch.tensor([2, 0, 3, 1])
    rearranged = DiffusionCondition(
        known_state=condition.known_state[:, order],
        design_mask=condition.design_mask[:, order],
        module_valid=condition.module_valid[:, order],
        module_features=condition.module_features[:, order],
        sensor_features=condition.sensor_features,
        sensor_valid=condition.sensor_valid,
    )
    permuted_links = PacketLinks(
        module_source=links.module_source[:, order][:, :, order],
        sensor_source=links.sensor_source[:, :, order],
        module_embeddings=links.module_embeddings[:, order],
    )
    permuted = model.denoiser(state[:, order], time, rearranged, permuted_links)
    torch.testing.assert_close(permuted, original[:, order], rtol=1e-5, atol=1e-6)


def test_typed_environment_routes_change_output_and_dense_control_keeps_embeddings() -> None:
    torch.manual_seed(37)
    condition = _condition("heat")
    model = _model("heat")
    state = torch.randn_like(condition.known_state)
    base = CandidateProvider()(state, condition)
    batch, modules, _ = state.shape
    sensors = condition.sensor_features.shape[1]
    links = PacketLinks(
        module_source=base.module_source,
        sensor_source=base.sensor_source,
        module_embeddings=base.module_embeddings,
        module_environment=torch.rand(batch, modules, 2),
        environment_module=torch.rand(batch, 2, modules),
        sensor_environment=torch.rand(batch, sensors, 2),
        environment_embeddings=torch.randn(batch, 2, 2),
        environment_valid=torch.tensor([[True, False], [True, False]]),
    )
    time = torch.tensor([0.2, 0.6])
    with_environment = model.denoiser(state, time, condition, links)
    without_environment = model.denoiser(state, time, condition, base)
    assert not torch.equal(with_environment, without_environment)
    dense = dense_access(links, condition)
    torch.testing.assert_close(dense.environment_embeddings, links.environment_embeddings)
    assert torch.all(dense.module_environment[..., 1] == 0)
    assert torch.all(dense.sensor_environment[..., 0] == 1)


def test_spatial_conditioner_binds_values_to_locations_without_sensor_order() -> None:
    torch.manual_seed(47)
    condition = DiffusionCondition(
        known_state=torch.zeros(1, 3, 2),
        design_mask=torch.tensor([[False, True, True]]),
        module_valid=torch.ones(1, 3, dtype=torch.bool),
        module_features=torch.randn(1, 3, 3),
        sensor_features=torch.tensor([[[0.0, 0.0, 1.0], [1.0, 0.0, -1.0], [0.0, 1.0, 0.5]]]),
        sensor_valid=torch.ones(1, 3, dtype=torch.bool),
    )
    coords = torch.tensor([[[0.1, 0.1], [0.7, 0.2], [0.2, 0.9]]])
    links = PacketLinks(
        module_source=torch.ones(1, 3, 3) - torch.eye(3)[None],
        sensor_source=torch.ones(1, 3, 3),
        module_embeddings=torch.randn(1, 3, 4),
        module_coordinates=coords,
        sensor_coordinates=condition.sensor_features[..., :2],
    )
    denoiser = SpatialConditionalPacketDenoiser(
        design_dim=2, module_dim=3, sensor_dim=3, embedding_dim=4,
        hidden_dim=24, layers=2, coordinate_dim=2,
    )
    state = torch.randn_like(condition.known_state)
    time = torch.tensor([0.4])
    original = denoiser(state, time, condition, links)
    order = torch.tensor([2, 0, 1])
    shuffled = replace(
        condition,
        sensor_features=condition.sensor_features[:, order],
        sensor_valid=condition.sensor_valid[:, order],
    )
    shuffled_links = replace(
        links,
        sensor_source=links.sensor_source[:, order],
        sensor_coordinates=links.sensor_coordinates[:, order],
    )
    torch.testing.assert_close(
        denoiser(state, time, shuffled, shuffled_links), original, atol=1e-6, rtol=1e-5
    )
    swapped_features = condition.sensor_features.clone()
    swapped_features[:, [0, 1], 2] = swapped_features[:, [1, 0], 2]
    swapped = denoiser(state, time, replace(condition, sensor_features=swapped_features), links)
    assert float((swapped - original).abs().max()) > 1e-4
    # Full-link G and Dense are exactly the same conditioner and access action.
    torch.testing.assert_close(denoiser(state, time, condition, dense_access(links, condition)), original)


def test_spatial_conditioner_handles_missing_sensors_and_frozen_links() -> None:
    torch.manual_seed(53)
    condition = DiffusionCondition(
        known_state=torch.zeros(1, 2, 1),
        design_mask=torch.ones(1, 2, dtype=torch.bool),
        module_valid=torch.ones(1, 2, dtype=torch.bool),
        module_features=torch.zeros(1, 2, 2),
        sensor_features=torch.tensor([[[0.0, 0.0, 0.8], [1.0, 0.0, -0.2]]]),
        sensor_valid=torch.tensor([[False, False]]),
    )
    links = PacketLinks(
        module_source=torch.ones(1, 2, 2) - torch.eye(2)[None],
        sensor_source=torch.ones(1, 2, 2),
        module_embeddings=torch.randn(1, 2, 3),
        module_coordinates=torch.tensor([[[0.2, 0.1], [0.8, 0.1]]]),
        sensor_coordinates=condition.sensor_features[..., :2],
    )
    denoiser = SpatialConditionalPacketDenoiser(
        design_dim=1, module_dim=2, sensor_dim=3, embedding_dim=3,
        hidden_dim=16, layers=1, coordinate_dim=2,
    )
    model = FrozenPacketDiffusion(denoiser, task="heat", steps=3)
    def provider(_candidate: torch.Tensor, _condition: DiffusionCondition) -> PacketLinks:
        return links
    clean = torch.randn_like(condition.known_state)
    loss = model.training_loss(
        clean, condition, provider, timesteps=torch.tensor([1]), noise=torch.ones_like(clean)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    assert links.module_embeddings.grad is None
    changed = replace(condition, sensor_features=condition.sensor_features + torch.tensor([0.0, 0.0, 7.0]))
    time = torch.tensor([0.2])
    state = torch.randn_like(clean)
    torch.testing.assert_close(denoiser(state, time, condition, links), denoiser(state, time, changed, links))

    dropped = torch.tensor([[[True, True, False], [True, True, True]]])
    one_missing_value = condition.sensor_features.clone()
    one_missing_value[0, 0, 2] = float("nan")
    observed = replace(
        condition, sensor_valid=torch.ones_like(condition.sensor_valid),
        sensor_features=one_missing_value, sensor_channel_valid=dropped,
    )
    dropped_output = denoiser(state, time, observed, links)
    assert torch.isfinite(dropped_output).all()
    alternate = one_missing_value.clone()
    alternate[0, 0, 2] = 1000.0
    torch.testing.assert_close(
        denoiser(state, time, replace(observed, sensor_features=alternate), links),
        dropped_output,
    )
    with pytest.raises(ValueError, match="Available sensor"):
        denoiser(state, time, replace(observed, sensor_channel_valid=None), links)

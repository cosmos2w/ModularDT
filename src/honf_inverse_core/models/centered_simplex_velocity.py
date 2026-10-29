"""Fixed-total heat diffusion in a zero-sum simplex-fraction subspace.

This is a distinct model identity from bounded position diffusion. Clean heat
states may exceed [-1,1]. Providers receive explicitly projected physical heat
allocations; denoisers receive the raw noisy centered fraction state. No hidden
individual heat values enter the condition or the forward provider.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields

import torch
from torch import nn

from .bounded_velocity import cosine_velocity_coefficients
from .frozen_packet_diffusion import DiffusionCondition, PacketLinks, centered_active, dense_access

HEAT_CHECKPOINT_FORMAT = "fixed_total_heat_centered_simplex_velocity_v1"
HEAT_PREDICTION_TYPE = "v=a*epsilon-b*r0"
HEAT_STATE_PARAMETERIZATION = "r0=sqrt(M)*(heat/total-1/M);active_zero_sum"


@dataclass(frozen=True)
class HeatSimplexCondition(DiffusionCondition):
    total_heat: torch.Tensor | None = None  # public [B,1], in adapter physical units

    def validate(self, design_dim: int, module_dim: int, sensor_dim: int) -> None:
        super().validate(design_dim, module_dim, sensor_dim)
        if design_dim != 1:
            raise ValueError("Centered-simplex heat uses exactly one scalar per module.")
        if not torch.equal(self.design_mask.bool(), self.module_valid.bool()):
            raise ValueError("Fixed-total heat design generates every active allocation.")
        if self.total_heat is None or self.total_heat.shape != (self.known_state.shape[0], 1):
            raise ValueError("Supplied total heat must have shape [B,1].")
        if not bool(torch.isfinite(self.total_heat).all()) or bool((self.total_heat < 0).any()):
            raise ValueError("Supplied total heat must be finite and nonnegative.")
        if not bool(torch.isfinite(self.known_state).all()) or bool((self.known_state != 0).any()):
            raise ValueError("Heat condition contains no known individual allocations; known state must be zero.")

    @property
    def free_dimensions(self) -> torch.Tensor:
        return (self.module_valid.bool().sum(dim=-1) - 1).clamp_min(0)


def _validate_heat_shapes(value: torch.Tensor, total: torch.Tensor, active: torch.Tensor) -> None:
    if value.ndim != 3 or value.shape[-1] != 1 or active.shape != value.shape[:2] or total.shape != (value.shape[0], 1):
        raise ValueError("Heat state must be [B,M,1], active mask [B,M], total [B,1].")
    if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(total).all()) or bool((total < 0).any()):
        raise ValueError("Heat state and supplied nonnegative total must be finite.")
    if bool((active.bool().sum(-1) == 0).any()):
        raise ValueError("Heat design needs at least one active module.")


def encode_heat_simplex(heat: torch.Tensor, total: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Differentiable caller-owned heat -> centered state; no NumPy conversion."""
    _validate_heat_shapes(heat, total, active)
    valid = active.bool()[..., None]
    if bool((heat < 0).any()) or bool((heat.masked_select(~valid) != 0).any()):
        raise ValueError("Clean heat must be nonnegative and zero on invalid modules.")
    if not torch.allclose(heat.sum(dim=1), total, atol=1e-5, rtol=1e-5):
        raise ValueError("Clean allocation does not match the supplied total heat.")
    count = active.bool().sum(-1, keepdim=True).to(heat.dtype)[..., None]
    # Zero supplied total has one canonical unidentifiable state: uniform r=0.
    fraction = torch.where(total[..., None] > 0, heat / total[..., None].clamp_min(torch.finfo(heat.dtype).tiny), 1.0 / count)
    return centered_active(count.sqrt() * (fraction - 1.0 / count), active)


def _project_simplex(fractions: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Euclidean simplex projection on active entries; no saturation logits."""
    valid = active.bool()
    sentinel = torch.finfo(fractions.dtype).min
    ordered, order = torch.sort(torch.where(valid, fractions, fractions.new_full((), sentinel)), dim=-1, descending=True)
    sorted_valid = valid.gather(-1, order)
    cumulative = torch.where(sorted_valid, ordered, torch.zeros_like(ordered)).cumsum(-1)
    ranks = torch.arange(1, fractions.shape[-1] + 1, device=fractions.device, dtype=fractions.dtype)
    occupied = sorted_valid & (ordered * ranks > cumulative - 1.0)
    count = occupied.sum(-1, keepdim=True).clamp_min(1)
    threshold = (cumulative.gather(-1, count - 1) - 1.0) / count.to(fractions.dtype)
    return torch.where(valid, (fractions - threshold).clamp_min(0), torch.zeros_like(fractions))


def decode_heat_simplex(state: torch.Tensor, total: torch.Tensor, active: torch.Tensor, *, project: bool = True) -> torch.Tensor:
    """Affine raw allocations or exact-simplex projected physical allocations.

    Both retain the supplied total. Raw allocations can be negative and remain
    separately labelled; only projected ones are sent to a physical provider.
    """
    _validate_heat_shapes(state, total, active)
    count = active.bool().sum(-1, keepdim=True).to(state.dtype)[..., None]
    centred = centered_active(state, active)
    fractions = torch.where(active.bool()[..., None], centred / count.sqrt() + 1.0 / count, torch.zeros_like(state))
    if project:
        fractions = _project_simplex(fractions.squeeze(-1), active).unsqueeze(-1)
    return fractions * total[..., None]


def heat_provider_proxy(state: torch.Tensor, condition: HeatSimplexCondition) -> torch.Tensor:
    """Projected physical heat, with a distinct contract from position clips."""
    return decode_heat_simplex(state, condition.total_heat, condition.module_valid, project=True)


def centered_heat_noise(reference: torch.Tensor, active: torch.Tensor, noise: torch.Tensor | None = None, *, generator: torch.Generator | None = None) -> torch.Tensor:
    value = torch.randn(reference.shape, device=reference.device, dtype=reference.dtype, generator=generator) if noise is None else noise
    if value.shape != reference.shape or not bool(torch.isfinite(value).all()):
        raise ValueError("Heat noise must be finite and match the state shape.")
    return centered_active(value, active)


@dataclass(frozen=True)
class HeatSimplexSampleTrail:
    final_state: torch.Tensor
    projected_final_state: torch.Tensor
    raw_final_heat: torch.Tensor
    projected_final_heat: torch.Tensor
    states: tuple[torch.Tensor, ...]
    state_timesteps: tuple[float, ...]
    clean_estimates: tuple[torch.Tensor, ...]
    clean_estimate_timesteps: tuple[float, ...]
    projection_mask: torch.Tensor
    free_dimensions: torch.Tensor
    organizer_calls: int


HeatLinksForCandidate = Callable[[torch.Tensor, HeatSimplexCondition], PacketLinks]


class CenteredSimplexVelocityDiffusion(nn.Module):
    checkpoint_format = HEAT_CHECKPOINT_FORMAT
    prediction_type = HEAT_PREDICTION_TYPE
    state_parameterization = HEAT_STATE_PARAMETERIZATION

    def __init__(self, denoiser: nn.Module, *, steps: int = 20):
        super().__init__()
        if getattr(denoiser, "design_dim", None) != 1 or not 2 <= steps <= 100:
            raise ValueError("Heat velocity diffusion requires scalar heat denoiser and 2..100 steps.")
        self.denoiser = denoiser
        self.steps = int(steps)
        self.design_dim = 1

    def _validate_condition(self, condition: HeatSimplexCondition) -> None:
        if not isinstance(condition, HeatSimplexCondition):
            raise TypeError("Heat sampler requires the separate HeatSimplexCondition.")
        condition.validate(1, int(self.denoiser.module_dim), int(self.denoiser.sensor_dim))

    def _validate_target(self, clean: torch.Tensor, condition: HeatSimplexCondition):
        self._validate_condition(condition)
        if clean.shape != condition.known_state.shape or not bool(torch.isfinite(clean).all()):
            raise ValueError("Clean heat state must be finite and match the condition.")
        if not torch.allclose(clean, centered_active(clean, condition.module_valid), atol=1e-6, rtol=1e-6):
            raise ValueError("Clean heat state must be zero-sum on active modules and zero on padding.")
        heat = decode_heat_simplex(clean, condition.total_heat, condition.module_valid, project=False)
        if bool((heat < -1e-5).any()):
            raise ValueError("Clean heat state lies outside the nonnegative allocation simplex.")
        if bool(((condition.total_heat == 0)[..., None] & (clean != 0)).any()):
            raise ValueError("Zero-total heat uses canonical zero centered state.")

    def noisy_state_and_velocity_target(self, clean_state, condition, time, noise):
        self._validate_target(clean_state, condition)
        if time.shape != (clean_state.shape[0],):
            raise ValueError("Training times must have shape [B].")
        epsilon = centered_heat_noise(clean_state, condition.module_valid, noise)
        a, b = cosine_velocity_coefficients(time)
        a, b = a[:, None, None], b[:, None, None]
        return a * clean_state + b * epsilon, a * epsilon - b * clean_state

    @staticmethod
    def _detach_links(links: PacketLinks) -> PacketLinks:
        return PacketLinks(**{field.name: None if getattr(links, field.name) is None else getattr(links, field.name).detach() for field in fields(links)})

    def _links(self, state, condition, provider, *, dense=False, links=None):
        if links is None:
            if provider is None:
                raise ValueError("Heat diffusion needs a provider or shared frozen links.")
            with torch.no_grad():
                links = provider(heat_provider_proxy(state.detach(), condition), condition)
        detached = self._detach_links(links)
        return dense_access(detached, condition) if dense else detached

    def _velocity(self, state, time, condition, links):
        predicted = self.denoiser(state, time, condition, links)
        if predicted.shape != state.shape or not bool(torch.isfinite(predicted).all()):
            raise ValueError("Heat denoiser velocity must be finite and match the state shape.")
        return centered_active(predicted, condition.module_valid)

    def prepare_candidate_links(self, noisy_state, condition, provider) -> PacketLinks:
        """One frozen provider call to share between matched graph/full arms."""
        self._validate_condition(condition)
        return self._links(noisy_state, condition, provider)

    def training_loss(self, clean_state, condition, provider=None, *, dense=False, time, noise, links=None):
        """Shared times/noise/links allow matched graph/full inverse arms."""
        noisy, target = self.noisy_state_and_velocity_target(clean_state, condition, time, noise)
        if not bool((condition.free_dimensions > 0).any()):
            # M=1 is a deterministic task with no learned degree of freedom.
            parameters = tuple(self.denoiser.parameters())
            return sum((parameter.sum() * 0 for parameter in parameters), clean_state.sum() * 0)
        frozen = self._links(noisy, condition, provider, dense=dense, links=links)
        predicted = self._velocity(noisy, time, condition, frozen)
        return (predicted - target).square().sum() / condition.free_dimensions.sum().clamp_min(1)

    @torch.no_grad()
    def sample(self, condition, provider, *, dense=False, initial_noise=None, generator=None,
               progress_callback=None, access_callback=None, pre_step_check=None) -> HeatSimplexSampleTrail:
        self._validate_condition(condition)
        state = centered_heat_noise(condition.known_state, condition.module_valid, initial_noise, generator=generator).detach()
        states, state_times, estimates, estimate_times = [state.clone()], [1.0], [], []
        calls = 0
        if progress_callback is not None:
            progress_callback(0, 1.0, 1.0, state.clone(), None)
        deterministic = not bool((condition.free_dimensions > 0).any())
        for index in range(self.steps):
            current, following = (self.steps - index) / self.steps, (self.steps - index - 1) / self.steps
            if pre_step_check is not None:
                pre_step_check(index, current)
            time = state.new_full((state.shape[0],), current)
            if deterministic:
                velocity = torch.zeros_like(state)
            else:
                links = self._links(state, condition, provider, dense=dense)
                calls += 1
                if access_callback is not None:
                    access_callback(index, current, links)
                velocity = self._velocity(state, time, condition, links)
            a, b = cosine_velocity_coefficients(time)
            a, b = a[:, None, None], b[:, None, None]
            clean, epsilon = a * state - b * velocity, b * state + a * velocity
            next_a, next_b = cosine_velocity_coefficients(time.new_full(time.shape, following))
            state = centered_active(next_a[:, None, None] * clean + next_b[:, None, None] * epsilon, condition.module_valid)
            if not bool(torch.isfinite(state).all()):
                raise FloatingPointError("Nonfinite heat reverse trajectory.")
            estimates.append(clean.clone())
            estimate_times.append(current)
            states.append(state.clone())
            state_times.append(following)
            if progress_callback is not None:
                progress_callback(index + 1, current, following, state.clone(), clean.clone())
        raw_heat = decode_heat_simplex(state, condition.total_heat, condition.module_valid, project=False)
        projected_heat = decode_heat_simplex(state, condition.total_heat, condition.module_valid, project=True)
        projected_state = encode_heat_simplex(projected_heat, condition.total_heat, condition.module_valid)
        changed = ~torch.isclose(raw_heat, projected_heat, rtol=1e-6, atol=1e-6)
        return HeatSimplexSampleTrail(state, projected_state, raw_heat, projected_heat,
                                     tuple(states), tuple(state_times), tuple(estimates), tuple(estimate_times),
                                     changed, condition.free_dimensions, calls)


def require_heat_simplex_checkpoint(payload: Mapping[str, object]) -> None:
    if payload.get("checkpoint_format") != HEAT_CHECKPOINT_FORMAT:
        raise ValueError("Checkpoint is not a fixed-total centered-simplex heat model.")
    if payload.get("prediction_type") != HEAT_PREDICTION_TYPE or payload.get("state_parameterization") != HEAT_STATE_PARAMETERIZATION:
        raise ValueError("Heat checkpoint prediction/state parameterization does not match.")


__all__ = ["HEAT_CHECKPOINT_FORMAT", "HEAT_PREDICTION_TYPE", "HEAT_STATE_PARAMETERIZATION", "CenteredSimplexVelocityDiffusion", "HeatSimplexCondition", "HeatSimplexSampleTrail", "centered_heat_noise", "decode_heat_simplex", "encode_heat_simplex", "heat_provider_proxy", "require_heat_simplex_checkpoint"]

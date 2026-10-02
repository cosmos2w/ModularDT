"""Opt-in normalized-coordinate velocity diffusion for inverse designs.

The historical epsilon-prediction implementation remains in
``frozen_packet_diffusion``. This module uses a separate checkpoint identity,
an affine clean-coordinate representation, and the exact cosine/sine velocity
parameterization described by the stable inverse plan.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from torch import nn

from .frozen_packet_diffusion import (
    DiffusionCondition,
    LinksForCandidate,
    PacketLinks,
    dense_access,
)

VELOCITY_CHECKPOINT_FORMAT = "wind_normalized_xy_velocity_v1"
VELOCITY_PREDICTION_TYPE = "v=a*epsilon-b*z0"
VELOCITY_STATE_PARAMETERIZATION = "affine_normalized_xy[-1,1]"


def cosine_velocity_coefficients(time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact endpoint coefficients for ``a=cos(pi*t/2), b=sin(pi*t/2)``."""

    if not torch.is_floating_point(time):
        raise TypeError("Diffusion times must be floating-point tensors.")
    if not bool(torch.isfinite(time).all()) or bool(torch.any((time < 0.0) | (time > 1.0))):
        raise ValueError("Diffusion times must be finite and lie in [0, 1].")
    a = torch.cos(time * (math.pi / 2.0))
    b = torch.sin(time * (math.pi / 2.0))
    zero = time == 0.0
    one = time == 1.0
    a = torch.where(zero, torch.ones_like(a), torch.where(one, torch.zeros_like(a), a))
    b = torch.where(zero, torch.zeros_like(b), torch.where(one, torch.ones_like(b), b))
    return a, b


def bounded_provider_proxy(
    state: torch.Tensor,
    condition: DiffusionCondition,
) -> torch.Tensor:
    """Clip only generated normalized coordinates for candidate-link building.

    The denoiser continues to receive the unbounded noisy state. Visible module
    values are restored exactly, and invalid slots receive no generated value.
    """

    if state.shape != condition.known_state.shape:
        raise ValueError("Candidate state and known state must have identical shapes.")
    if not bool(torch.isfinite(state).all()):
        raise ValueError("Candidate state must be finite before provider construction.")
    generated = condition.design_mask.bool().unsqueeze(-1)
    valid = condition.module_valid.bool().unsqueeze(-1)
    if bool(torch.any(condition.known_state.masked_select(~valid.expand_as(state)) != 0.0)):
        raise ValueError("Invalid module slots must have zero known state.")
    clipped = state.clamp(-1.0, 1.0)
    return torch.where(generated, clipped, condition.known_state)


def _masked_noise_like(
    reference: torch.Tensor,
    condition: DiffusionCondition,
    noise: torch.Tensor | None,
) -> torch.Tensor:
    value = torch.randn_like(reference) if noise is None else noise
    if value.shape != reference.shape:
        raise ValueError("Diffusion noise must have the same shape as the design state.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Diffusion noise must be finite.")
    return value * condition.design_mask.to(value.dtype).unsqueeze(-1)


def _expand_coefficients(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], *([1] * 2))


@dataclass(frozen=True)
class VelocitySampleTrail:
    """Complete deterministic reverse trajectory and separate terminal projection."""

    final_state: torch.Tensor
    projected_final_state: torch.Tensor
    states: tuple[torch.Tensor, ...]
    state_timesteps: tuple[float, ...]
    clean_estimates: tuple[torch.Tensor, ...]
    clean_estimate_timesteps: tuple[float, ...]
    state_norms: tuple[torch.Tensor, ...]
    clean_estimate_norms: tuple[torch.Tensor, ...]
    projection_mask: torch.Tensor
    generated_coordinate_count: int
    organizer_calls: int

    @property
    def projection_coordinate_fraction(self) -> float:
        if self.generated_coordinate_count == 0:
            return 0.0
        return float(self.projection_mask.sum().cpu()) / self.generated_coordinate_count


class BoundedVelocityPacketDiffusion(nn.Module):
    """Velocity-prediction diffusion over normalized bounded design coordinates."""

    checkpoint_format = VELOCITY_CHECKPOINT_FORMAT
    prediction_type = VELOCITY_PREDICTION_TYPE
    state_parameterization = VELOCITY_STATE_PARAMETERIZATION

    def __init__(self, denoiser: nn.Module, *, steps: int = 20) -> None:
        super().__init__()
        if not 2 <= int(steps) <= 100:
            raise ValueError("Stable velocity diffusion requires 2..100 reverse steps.")
        self.denoiser = denoiser
        self.steps = int(steps)
        self.design_dim = int(getattr(denoiser, "design_dim", 0))
        if self.design_dim <= 0:
            raise ValueError("Velocity denoiser must expose a positive design_dim.")

    def _validate_target(
        self,
        clean_state: torch.Tensor,
        condition: DiffusionCondition,
    ) -> None:
        if clean_state.shape != condition.known_state.shape:
            raise ValueError("Clean target and condition state have incompatible shapes.")
        if clean_state.shape[-1] != self.design_dim:
            raise ValueError("Clean target width differs from the denoiser design width.")
        if not bool(torch.isfinite(clean_state).all()):
            raise ValueError("Clean normalized design targets must be finite.")
        condition.validate(
            self.design_dim,
            int(self.denoiser.module_dim),
            int(self.denoiser.sensor_dim),
        )
        generated = condition.design_mask.bool().unsqueeze(-1)
        if bool(torch.any(generated & ((clean_state < -1.0 - 1e-6) | (clean_state > 1.0 + 1e-6)))):
            raise ValueError("Clean normalized generated coordinates must lie in [-1, 1].")

    def noisy_state_and_velocity_target(
        self,
        clean_state: torch.Tensor,
        condition: DiffusionCondition,
        time: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct ``z_t`` and ``v_t`` with visible and invalid noise set to zero."""

        self._validate_target(clean_state, condition)
        batch = clean_state.shape[0]
        if time.shape != (batch,) or not torch.is_floating_point(time):
            raise ValueError("Training times must be floating-point with shape [batch].")
        generated_noise = _masked_noise_like(clean_state, condition, noise)
        generated = condition.design_mask.bool().unsqueeze(-1)
        z0 = torch.where(generated, clean_state, condition.known_state)
        a, b = cosine_velocity_coefficients(time)
        a = _expand_coefficients(a)
        b = _expand_coefficients(b)
        noisy = torch.where(generated, a * z0 + b * generated_noise, condition.known_state)
        velocity = torch.where(
            generated,
            a * generated_noise - b * z0,
            torch.zeros_like(generated_noise),
        )
        return noisy, velocity

    def _links(
        self,
        state: torch.Tensor,
        condition: DiffusionCondition,
        provider: LinksForCandidate,
        *,
        dense: bool,
    ) -> PacketLinks:
        # The provider gets only a clipped proxy. The denoiser still sees the
        # unmodified noisy coordinates in ``state``.
        proxy = bounded_provider_proxy(state.detach(), condition)
        with torch.no_grad():
            links = provider(proxy, condition)
        detached = PacketLinks(
            links.module_source.detach(),
            links.sensor_source.detach(),
            links.module_embeddings.detach(),
            *(
                None if value is None else value.detach()
                for value in (
                    links.module_environment,
                    links.environment_module,
                    links.sensor_environment,
                    links.environment_embeddings,
                    links.environment_valid,
                    links.module_coordinates,
                    links.sensor_coordinates,
                )
            ),
        )
        return dense_access(detached, condition) if dense else detached

    def training_loss(
        self,
        clean_state: torch.Tensor,
        condition: DiffusionCondition,
        provider: LinksForCandidate,
        *,
        dense: bool = False,
        time: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the design-mask-only mean squared velocity error."""

        noisy, target_velocity = self.noisy_state_and_velocity_target(clean_state, condition, time, noise)
        links = self._links(noisy, condition, provider, dense=dense)
        predicted = self.denoiser(noisy, time, condition, links)
        mask = condition.design_mask.to(predicted.dtype).unsqueeze(-1)
        return ((predicted - target_velocity).square() * mask).sum() / (mask.sum() * self.design_dim).clamp_min(1.0)

    @staticmethod
    def _rms(value: torch.Tensor, condition: DiffusionCondition) -> torch.Tensor:
        mask = condition.design_mask.to(value.dtype).unsqueeze(-1)
        count = mask.sum(dim=(1, 2)).clamp_min(1.0) * value.shape[-1]
        return ((value.square() * mask).sum(dim=(1, 2)) / count).sqrt()

    @torch.no_grad()
    def sample(
        self,
        condition: DiffusionCondition,
        provider: LinksForCandidate,
        *,
        dense: bool = False,
        initial_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        progress_callback: Callable[[int, float, float, torch.Tensor, torch.Tensor | None], None] | None = None,
        access_callback: Callable[[int, float, PacketLinks], None] | None = None,
        pre_step_check: Callable[[int, float], None] | None = None,
    ) -> VelocitySampleTrail:
        """Run a deterministic 20-step DDIM-style path and retain every estimate."""

        condition.validate(
            self.design_dim,
            int(self.denoiser.module_dim),
            int(self.denoiser.sensor_dim),
        )
        if condition.known_state.shape[-1] != self.design_dim:
            raise ValueError("Condition design width differs from the velocity denoiser.")
        state = (
            torch.randn(
                condition.known_state.shape,
                dtype=condition.known_state.dtype,
                device=condition.known_state.device,
                generator=generator,
            )
            if initial_noise is None
            else initial_noise
        )
        if state.shape != condition.known_state.shape or not bool(torch.isfinite(state).all()):
            raise ValueError("Initial diffusion noise must be finite and match the state shape.")
        generated = condition.design_mask.bool().unsqueeze(-1)
        state = torch.where(generated, state, condition.known_state).detach().clone()
        states: list[torch.Tensor] = [state.clone()]
        state_times: list[float] = [1.0]
        clean_estimates: list[torch.Tensor] = []
        clean_times: list[float] = []
        state_norms: list[torch.Tensor] = [self._rms(state, condition).detach().cpu()]
        clean_norms: list[torch.Tensor] = []
        if progress_callback is not None:
            progress_callback(0, 1.0, 1.0, state.clone(), None)

        for index in range(self.steps):
            current_time = float(self.steps - index) / float(self.steps)
            next_time = float(self.steps - index - 1) / float(self.steps)
            if pre_step_check is not None:
                pre_step_check(index, current_time)
            time = state.new_full((state.shape[0],), current_time)
            links = self._links(state, condition, provider, dense=dense)
            if access_callback is not None:
                access_callback(index, current_time, links)
            predicted_velocity = self.denoiser(state, time, condition, links)
            a, b = cosine_velocity_coefficients(time)
            a = _expand_coefficients(a)
            b = _expand_coefficients(b)
            raw_clean = torch.where(
                generated,
                a * state - b * predicted_velocity,
                condition.known_state,
            )
            predicted_noise = torch.where(
                generated,
                b * state + a * predicted_velocity,
                torch.zeros_like(state),
            )
            next_time_tensor = state.new_full((state.shape[0],), next_time)
            next_a, next_b = cosine_velocity_coefficients(next_time_tensor)
            next_a = _expand_coefficients(next_a)
            next_b = _expand_coefficients(next_b)
            state = torch.where(
                generated,
                next_a * raw_clean + next_b * predicted_noise,
                condition.known_state,
            )
            state = state.detach()
            clean_estimates.append(raw_clean.detach().clone())
            clean_times.append(current_time)
            state_norms.append(self._rms(state, condition).detach().cpu())
            clean_norms.append(self._rms(raw_clean, condition).detach().cpu())
            states.append(state.clone())
            state_times.append(next_time)
            if progress_callback is not None:
                progress_callback(index + 1, current_time, next_time, state.clone(), raw_clean.clone())

        raw_terminal = state.detach().clone()
        projection_mask = generated & ((raw_terminal < -1.0) | (raw_terminal > 1.0))
        projected = torch.where(
            generated,
            raw_terminal.clamp(-1.0, 1.0),
            condition.known_state,
        )
        return VelocitySampleTrail(
            final_state=raw_terminal,
            projected_final_state=projected.detach().clone(),
            states=tuple(states),
            state_timesteps=tuple(state_times),
            clean_estimates=tuple(clean_estimates),
            clean_estimate_timesteps=tuple(clean_times),
            state_norms=tuple(state_norms),
            clean_estimate_norms=tuple(clean_norms),
            projection_mask=projection_mask.detach().clone(),
            generated_coordinate_count=int(generated.expand_as(state).sum().item()),
            organizer_calls=self.steps,
        )


def require_velocity_checkpoint(payload: Mapping[str, object]) -> None:
    """Reject epsilon checkpoints and mismatched state/target semantics."""

    if payload.get("checkpoint_format") != VELOCITY_CHECKPOINT_FORMAT:
        raise ValueError("Checkpoint is not a bounded normalized-coordinate velocity model.")
    if payload.get("prediction_type") != VELOCITY_PREDICTION_TYPE:
        raise ValueError("Velocity checkpoint prediction target does not match this sampler.")
    if payload.get("state_parameterization") != VELOCITY_STATE_PARAMETERIZATION:
        raise ValueError("Velocity checkpoint coordinate parameterization does not match this sampler.")


__all__ = [
    "VELOCITY_CHECKPOINT_FORMAT",
    "VELOCITY_PREDICTION_TYPE",
    "VELOCITY_STATE_PARAMETERIZATION",
    "BoundedVelocityPacketDiffusion",
    "VelocitySampleTrail",
    "bounded_provider_proxy",
    "cosine_velocity_coefficients",
    "require_velocity_checkpoint",
]

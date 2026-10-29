"""Small conditional diffusion model for frozen forward-packet reuse.

The forward interface is evaluated from the *current* noisy design by a caller
supplied function. The clean design is used only as the denoising target. Graph
and dense controls use the same network and frozen module embeddings; only the
physical access matrices differ. No slot-index embedding is used.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .rectified_flow import SinusoidalTimeEmbedding

TaskKind = Literal["heat", "position"]


@dataclass(frozen=True)
class PacketLinks:
    """Candidate-specific, physically indexed links from a frozen organizer.

    Rows of ``module_source`` are receiving modules; columns are source modules.
    Rows of ``sensor_source`` are fixed physical observation receivers. Optional
    embeddings come from the same frozen forward model in both inverse arms.
    """

    module_source: torch.Tensor  # [B, M, M]
    sensor_source: torch.Tensor  # [B, S, M]
    module_embeddings: torch.Tensor  # [B, M, E]
    module_environment: torch.Tensor | None = None  # ME [B, M, E_source]
    environment_module: torch.Tensor | None = None  # EM [B, E_receiver, M]
    sensor_environment: torch.Tensor | None = None  # QE [B, S, E_source]
    environment_embeddings: torch.Tensor | None = None  # [B, E, embedding_dim]
    environment_valid: torch.Tensor | None = None  # [B, E]
    module_coordinates: torch.Tensor | None = None  # [B, M, spatial_dim]
    sensor_coordinates: torch.Tensor | None = None  # [B, S, spatial_dim]


@dataclass(frozen=True)
class DiffusionCondition:
    """Only known inputs, with clean values represented by the separate target."""

    known_state: torch.Tensor  # [B, M, D]; visible coordinates or zeros
    design_mask: torch.Tensor  # [B, M]; generated modules
    module_valid: torch.Tensor  # [B, M]
    module_features: torch.Tensor  # [B, M, F]; known geometry/context only
    sensor_features: torch.Tensor  # [B, S, O]; fixed-location observations
    sensor_valid: torch.Tensor  # [B, S]
    design_lower: torch.Tensor | None = None  # optional public native design bounds [B,D]
    design_upper: torch.Tensor | None = None  # paired with design_lower
    sensor_channel_valid: torch.Tensor | None = None  # optional [B,S,O] availability flags

    def validate(self, design_dim: int, module_dim: int, sensor_dim: int) -> None:
        batch, modules, width = self.known_state.shape
        if width != design_dim or self.design_mask.shape != (batch, modules):
            raise ValueError("Known state or design mask has an incompatible shape.")
        if self.module_valid.shape != (batch, modules):
            raise ValueError("Module validity has an incompatible shape.")
        if self.module_features.shape != (batch, modules, module_dim):
            raise ValueError("Known module features have an incompatible shape.")
        if self.sensor_features.ndim != 3 or self.sensor_features.shape[0] != batch:
            raise ValueError("Sensor features have an incompatible batch shape.")
        if self.sensor_features.shape[-1] != sensor_dim:
            raise ValueError("Sensor features have an incompatible width.")
        if self.sensor_valid.shape != self.sensor_features.shape[:2]:
            raise ValueError("Sensor validity has an incompatible shape.")
        if self.sensor_channel_valid is not None:
            if self.sensor_channel_valid.shape != self.sensor_features.shape:
                raise ValueError("Sensor-channel validity has an incompatible shape.")
            if self.sensor_channel_valid.dtype != torch.bool:
                raise TypeError("Sensor-channel availability flags must be boolean.")
        if (self.design_lower is None) != (self.design_upper is None):
            raise ValueError("Public design bounds must be supplied together.")
        if self.design_lower is not None:
            if self.design_lower.shape != (batch, design_dim) or self.design_upper.shape != (batch, design_dim):
                raise ValueError("Public design bounds have an incompatible shape.")
            if (not torch.isfinite(self.design_lower).all()
                    or not torch.isfinite(self.design_upper).all()
                    or torch.any(self.design_lower >= self.design_upper)):
                raise ValueError("Public design bounds must be finite and ordered.")
        if torch.any(self.design_mask.bool() & ~self.module_valid.bool()):
            raise ValueError("Generated modules must be valid modules.")
        if torch.any(self.design_mask.sum(dim=1) == 0):
            raise ValueError("Every condition needs at least one generated module.")


LinksForCandidate = Callable[[torch.Tensor, DiffusionCondition], PacketLinks]


def centered_active(value: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Project heat logits/noise into the zero-sum active-module subspace."""

    weight = active.to(value.dtype).unsqueeze(-1)
    mean = (value * weight).sum(dim=1, keepdim=True) / weight.sum(dim=1, keepdim=True).clamp_min(1)
    return (value - mean) * weight


def heat_from_logits(logits: torch.Tensor, total: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Return nonnegative allocation with the supplied total, never a hidden total."""

    if logits.shape[-1] != 1 or total.shape != (logits.shape[0], 1):
        raise ValueError("Heat logits must be [B,M,1] and supplied totals [B,1].")
    if active.shape != logits.shape[:2] or torch.any(active.sum(dim=1) == 0):
        raise ValueError("Every heat allocation needs an active module.")
    if torch.any(total < 0):
        raise ValueError("Supplied heat totals must be nonnegative.")
    masked = logits.squeeze(-1).masked_fill(~active.bool(), torch.finfo(logits.dtype).min)
    fractions = torch.softmax(masked, dim=1)
    return (fractions * total).unsqueeze(-1)


def dense_access(links: PacketLinks, condition: DiffusionCondition) -> PacketLinks:
    """Matched full-access inverse control, retaining frozen forward embeddings."""

    valid = condition.module_valid.bool()
    module = valid[:, :, None] & valid[:, None, :]
    diagonal = torch.eye(valid.shape[1], device=valid.device, dtype=torch.bool)
    module = module & ~diagonal[None]
    sensor = condition.sensor_valid.bool()[:, :, None] & valid[:, None, :]
    environment = _environment_tensors(links)
    if environment is None:
        extra = {}
    else:
        env_valid = links.environment_valid.bool()
        extra = {
            "module_environment": (
                valid[:, :, None] & env_valid[:, None, :]
            ).to(links.module_environment.dtype),
            "environment_module": (
                env_valid[:, :, None] & valid[:, None, :]
            ).to(links.environment_module.dtype),
            "sensor_environment": (
                condition.sensor_valid.bool()[:, :, None] & env_valid[:, None, :]
            ).to(links.sensor_environment.dtype),
            "environment_embeddings": links.environment_embeddings,
            "environment_valid": links.environment_valid,
        }
    return PacketLinks(
        module_source=module.to(links.module_source.dtype),
        sensor_source=sensor.to(links.sensor_source.dtype),
        module_embeddings=links.module_embeddings,
        module_coordinates=links.module_coordinates,
        sensor_coordinates=links.sensor_coordinates,
        **extra,
    )


def _environment_tensors(links: PacketLinks) -> tuple[torch.Tensor, ...] | None:
    values = (
        links.module_environment,
        links.environment_module,
        links.sensor_environment,
        links.environment_embeddings,
        links.environment_valid,
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("Typed environment links, embeddings, and validity must be supplied together.")
    return values


def _normalized_message(weights: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    denominator = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.bmm(weights, source) / denominator


class ConditionalPacketDenoiser(nn.Module):
    """Equivariant set denoiser with packet-mediated module and sensor messages."""

    def __init__(
        self,
        *,
        design_dim: int,
        module_dim: int,
        sensor_dim: int,
        embedding_dim: int,
        hidden_dim: int = 96,
        layers: int = 3,
    ) -> None:
        super().__init__()
        if min(design_dim, module_dim, sensor_dim, embedding_dim, hidden_dim, layers) <= 0:
            raise ValueError("Denoiser dimensions and depth must be positive.")
        self.design_dim = int(design_dim)
        self.module_dim = int(module_dim)
        self.sensor_dim = int(sensor_dim)
        self.embedding_dim = int(embedding_dim)
        self.module_input = nn.Linear(design_dim + module_dim + embedding_dim + 2, hidden_dim)
        self.environment_input = nn.Linear(embedding_dim, hidden_dim)
        self.sensor_input = nn.Linear(sensor_dim, hidden_dim)
        self.time_input = SinusoidalTimeEmbedding(hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim * 4),
                    nn.Linear(hidden_dim * 4, hidden_dim * 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(layers)
            ]
        )
        self.environment_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim * 3),
                    nn.Linear(hidden_dim * 3, hidden_dim * 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(layers)
            ]
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, design_dim))

    def _sensor_tokens(self, condition: DiffusionCondition) -> torch.Tensor:
        """Historical linear token encoder, retained for old inverse checkpoints."""

        return self.sensor_input(condition.sensor_features)

    def _module_observation(
        self,
        hidden: torch.Tensor,
        sensed: torch.Tensor,
        sensor_weights: torch.Tensor,
        condition: DiffusionCondition,
        links: PacketLinks,
    ) -> torch.Tensor:
        return _normalized_message(sensor_weights.transpose(1, 2), sensed)

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        condition: DiffusionCondition,
        links: PacketLinks,
    ) -> torch.Tensor:
        condition.validate(self.design_dim, self.module_dim, self.sensor_dim)
        batch, modules, _ = state.shape
        sensors = condition.sensor_features.shape[1]
        if state.shape != condition.known_state.shape or time.shape != (batch,):
            raise ValueError("Noisy state or diffusion time has an incompatible shape.")
        if links.module_source.shape != (batch, modules, modules):
            raise ValueError("Module-source links have an incompatible shape.")
        if links.sensor_source.shape != (batch, sensors, modules):
            raise ValueError("Sensor-source links have an incompatible shape.")
        if links.module_embeddings.shape != (batch, modules, self.embedding_dim):
            raise ValueError("Frozen module embeddings have an incompatible shape.")
        environment = _environment_tensors(links)
        if environment is not None:
            me, em, qe, env_embeddings, env_valid = environment
            env_count = env_embeddings.shape[1]
            if (
                me.shape != (batch, modules, env_count)
                or em.shape != (batch, env_count, modules)
                or qe.shape != (batch, sensors, env_count)
                or env_embeddings.shape != (batch, env_count, self.embedding_dim)
                or env_valid.shape != (batch, env_count)
            ):
                raise ValueError("Typed environment links have incompatible source/receiver axes.")
        for tensor in (links.module_source, links.sensor_source, links.module_embeddings, *(environment or ())):
            if not torch.isfinite(tensor).all():
                raise ValueError("Frozen interface contains nonfinite values.")
        if torch.any(links.module_source < 0) or torch.any(links.sensor_source < 0):
            raise ValueError("Frozen access weights must be nonnegative.")
        if environment is not None and any(torch.any(value < 0) for value in environment[:3]):
            raise ValueError("Typed environment access weights must be nonnegative.")

        valid = condition.module_valid.to(state.dtype)
        module_input = torch.cat(
            (
                state,
                condition.module_features,
                links.module_embeddings,
                condition.design_mask.to(state.dtype).unsqueeze(-1),
                valid.unsqueeze(-1),
            ),
            dim=-1,
        )
        hidden = self.module_input(module_input) + self.time_input(time).unsqueeze(1)
        hidden = hidden * valid.unsqueeze(-1)
        sensor = self._sensor_tokens(condition)
        sensor = sensor * condition.sensor_valid.to(sensor.dtype).unsqueeze(-1)
        module_weights = links.module_source.to(state.dtype) * (
            valid[:, :, None] * valid[:, None, :]
        )
        sensor_weights = links.sensor_source.to(state.dtype) * (
            condition.sensor_valid.to(state.dtype)[:, :, None] * valid[:, None, :]
        )
        if environment is None:
            environment_hidden = None
            module_env_weights = env_module_weights = sensor_env_weights = None
        else:
            me, em, qe, env_embeddings, env_valid = environment
            env_mask = env_valid.to(state.dtype)
            environment_hidden = self.environment_input(env_embeddings) * env_mask.unsqueeze(-1)
            module_env_weights = me.to(state.dtype) * (valid[:, :, None] * env_mask[:, None, :])
            env_module_weights = em.to(state.dtype) * (env_mask[:, :, None] * valid[:, None, :])
            sensor_env_weights = qe.to(state.dtype) * (
                condition.sensor_valid.to(state.dtype)[:, :, None] * env_mask[:, None, :]
            )
        for block, env_block in zip(self.blocks, self.environment_blocks, strict=True):
            module_message = _normalized_message(module_weights, hidden)
            # A sensor receives through its physical packet links. The same
            # incidence returns its known observation to associated modules.
            sensed = sensor + _normalized_message(sensor_weights, hidden)
            if environment_hidden is not None:
                sensed = sensed + _normalized_message(sensor_env_weights, environment_hidden)
                env_observation = _normalized_message(sensor_env_weights.transpose(1, 2), sensed)
                env_message = _normalized_message(env_module_weights, hidden)
                environment_hidden = environment_hidden + env_block(
                    torch.cat((environment_hidden, env_message, env_observation), dim=-1)
                )
                environment_hidden = environment_hidden * env_mask.unsqueeze(-1)
                environment_message = _normalized_message(module_env_weights, environment_hidden)
            else:
                environment_message = torch.zeros_like(hidden)
            observation = self._module_observation(
                hidden, sensed, sensor_weights, condition, links
            )
            hidden = hidden + block(
                torch.cat((hidden, module_message, environment_message, observation), dim=-1)
            )
            hidden = hidden * valid.unsqueeze(-1)
        return self.output(hidden) * condition.design_mask.to(state.dtype).unsqueeze(-1)


class SpatialConditionalPacketDenoiser(ConditionalPacketDenoiser):
    """Sensor-bound inverse conditioner for new runs; legacy weights stay loadable.

    Sensor tokens are nonlinear in their coordinate/value pair. Every module
    then attends to sensor tokens with a candidate-specific relative-coordinate
    bias and its frozen graph access as a prior. Dense uses the identical
    conditioner with full admissible access. Coordinates are supplied by the
    candidate-only interface and never inferred from a clean hidden target.
    """

    def __init__(self, *, coordinate_dim: int, **kwargs: int) -> None:
        super().__init__(**kwargs)
        if not 1 <= coordinate_dim <= self.sensor_dim:
            raise ValueError("Spatial coordinate width must fit the sensor features.")
        self.coordinate_dim = int(coordinate_dim)
        width = self.module_input.out_features
        self.sensor_input = nn.Sequential(
            nn.Linear(2 * self.sensor_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.sensor_query = nn.Linear(width, width, bias=False)
        self.sensor_key = nn.Linear(width, width, bias=False)
        self.sensor_value = nn.Linear(width, width, bias=False)
        self.relative_bias = nn.Sequential(
            nn.Linear(self.coordinate_dim, width // 2 or 1),
            nn.SiLU(),
            nn.Linear(width // 2 or 1, 1),
        )

    def _sensor_tokens(self, condition: DiffusionCondition) -> torch.Tensor:
        present = condition.sensor_valid.bool()[..., None]
        channels = (
            torch.ones_like(condition.sensor_features, dtype=torch.bool)
            if condition.sensor_channel_valid is None else condition.sensor_channel_valid
        )
        if bool((present & ~channels[..., : self.coordinate_dim]).any()):
            raise ValueError("Observed sensors require valid location channels.")
        available = present & channels
        if not bool(torch.isfinite(condition.sensor_features[available]).all()):
            raise ValueError("Available sensor coordinates and values must be finite.")
        values = torch.where(available, condition.sensor_features, 0.0)
        tokens = torch.cat((values, available.to(values.dtype)), dim=-1)
        return self.sensor_input(tokens)

    def _module_observation(
        self,
        hidden: torch.Tensor,
        sensed: torch.Tensor,
        sensor_weights: torch.Tensor,
        condition: DiffusionCondition,
        links: PacketLinks,
    ) -> torch.Tensor:
        module_xy = links.module_coordinates
        sensor_xy = links.sensor_coordinates
        batch, modules, width = hidden.shape
        sensors = sensed.shape[1]
        expected_module = (batch, modules, self.coordinate_dim)
        expected_sensor = (batch, sensors, self.coordinate_dim)
        if module_xy is None or module_xy.shape != expected_module:
            raise ValueError("Spatial conditioner needs candidate module coordinates.")
        if sensor_xy is None or sensor_xy.shape != expected_sensor:
            raise ValueError("Spatial conditioner needs fixed sensor coordinates.")
        if not torch.isfinite(module_xy).all() or not torch.isfinite(sensor_xy).all():
            raise ValueError("Spatial conditioner coordinates must be finite.")
        # The input-only span makes relative coordinates comparable across
        # native units without indexing statistics by a future task or sensor.
        locations = torch.cat((module_xy, sensor_xy), dim=1)
        location_valid = torch.cat(
            (condition.module_valid.bool(), condition.sensor_valid.bool()), dim=1
        )
        maximum = locations.masked_fill(~location_valid[..., None], -torch.inf).amax(dim=1)
        minimum = locations.masked_fill(~location_valid[..., None], torch.inf).amin(dim=1)
        span = (maximum - minimum).clamp_min(1e-6)
        relative = (module_xy[:, :, None, :] - sensor_xy[:, None, :, :]) / span[:, None, None]
        logits = torch.bmm(
            self.sensor_query(hidden), self.sensor_key(sensed).transpose(1, 2)
        ) / math.sqrt(width)
        logits = logits + self.relative_bias(relative).squeeze(-1)
        prior = sensor_weights.transpose(1, 2)
        allowed = (prior > 0) & condition.module_valid.bool()[:, :, None]
        logits = logits + prior.clamp_min(1e-8).log()
        logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1) * allowed.to(logits.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.bmm(attention, self.sensor_value(sensed))


@dataclass(frozen=True)
class SampleTrail:
    final_state: torch.Tensor
    states: tuple[torch.Tensor, ...]
    timesteps: tuple[int, ...]
    organizer_calls: int


class FrozenPacketDiffusion(nn.Module):
    """Twenty-step DDPM pilot with a frozen, candidate-rebuilt graph callback."""

    def __init__(
        self,
        denoiser: ConditionalPacketDenoiser,
        *,
        task: TaskKind,
        steps: int = 20,
    ) -> None:
        super().__init__()
        if task not in {"heat", "position"} or not (2 <= steps <= 20):
            raise ValueError("Use heat or position with 2..20 reverse steps.")
        if task == "heat" and denoiser.design_dim != 1:
            raise ValueError("Heat logits need a single state channel.")
        if task == "position" and denoiser.design_dim != 2:
            raise ValueError("Position completion needs two state channels.")
        self.denoiser = denoiser
        self.task = task
        self.steps = int(steps)
        indices = torch.arange(steps + 1, dtype=torch.float64)
        cosine = torch.cos(((indices / steps + 0.008) / 1.008) * math.pi / 2).square()
        alpha_bar = (cosine / cosine[0]).clamp_min(1e-5)
        betas = (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-4, 0.999)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alpha_bar", torch.cumprod(alphas.float(), dim=0))

    def _project(self, value: torch.Tensor, condition: DiffusionCondition) -> torch.Tensor:
        if self.task == "heat":
            return centered_active(value, condition.design_mask)
        mask = condition.design_mask.to(value.dtype).unsqueeze(-1)
        return value * mask + condition.known_state * (1.0 - mask)

    def _links(
        self,
        state: torch.Tensor,
        condition: DiffusionCondition,
        provider: LinksForCandidate,
        *,
        dense: bool,
    ) -> PacketLinks:
        # The provider receives the current state and known inputs only. It is
        # never passed the clean target; no organizer weight gets an inverse
        # gradient. A caller must rebuild design-dependent encodings here.
        with torch.no_grad():
            links = provider(state.detach(), condition)
        links = PacketLinks(
            links.module_source.detach(),
            links.sensor_source.detach(),
            links.module_embeddings.detach(),
            *(None if value is None else value.detach() for value in (
                links.module_environment,
                links.environment_module,
                links.sensor_environment,
                links.environment_embeddings,
                links.environment_valid,
                links.module_coordinates,
                links.sensor_coordinates,
            )),
        )
        return dense_access(links, condition) if dense else links

    def training_loss(
        self,
        clean_state: torch.Tensor,
        condition: DiffusionCondition,
        provider: LinksForCandidate,
        *,
        dense: bool = False,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition.validate(
            self.denoiser.design_dim, self.denoiser.module_dim, self.denoiser.sensor_dim
        )
        if clean_state.shape != condition.known_state.shape:
            raise ValueError("Clean target has an incompatible shape.")
        batch = clean_state.shape[0]
        if timesteps is None:
            timesteps = torch.randint(self.steps, (batch,), device=clean_state.device)
        if timesteps.shape != (batch,) or torch.any(timesteps < 0) or torch.any(timesteps >= self.steps):
            raise ValueError("Diffusion timesteps are invalid.")
        if noise is None:
            noise = torch.randn_like(clean_state)
        if noise.shape != clean_state.shape:
            raise ValueError("Diffusion noise has an incompatible shape.")
        target = self._project(clean_state, condition)
        noise = self._project(noise, condition)
        alpha_bar = self.alpha_bar[timesteps].reshape(batch, 1, 1)
        state = self._project(alpha_bar.sqrt() * target + (1 - alpha_bar).sqrt() * noise, condition)
        links = self._links(state, condition, provider, dense=dense)
        predicted = self.denoiser(state, timesteps.float() / self.steps, condition, links)
        weight = condition.design_mask.to(predicted.dtype).unsqueeze(-1)
        return ((predicted - noise).square() * weight).sum() / (
            weight.sum() * self.denoiser.design_dim
        ).clamp_min(1)

    @torch.no_grad()
    def sample(
        self,
        condition: DiffusionCondition,
        provider: LinksForCandidate,
        *,
        dense: bool = False,
        initial_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        save_every: int = 4,
        progress_callback: Callable[[int, torch.Tensor], None] | None = None,
    ) -> SampleTrail:
        """Sample one trail and optionally report detached state snapshots.

        The callback receives the initial ``steps`` state, then each
        post-update index down through zero. Callback exceptions propagate and
        stop the sample, leaving any caller-persisted snapshots intact.
        """
        condition.validate(
            self.denoiser.design_dim, self.denoiser.module_dim, self.denoiser.sensor_dim
        )
        if save_every <= 0:
            raise ValueError("save_every must be positive.")
        if initial_noise is None:
            initial_noise = torch.randn(
                condition.known_state.shape,
                dtype=condition.known_state.dtype,
                device=condition.known_state.device,
                generator=generator,
            )
        if initial_noise.shape != condition.known_state.shape:
            raise ValueError("Initial noise has an incompatible shape.")
        state = self._project(initial_noise, condition)
        trail: list[torch.Tensor] = [state.detach().clone()]
        times: list[int] = [self.steps]
        if progress_callback is not None:
            progress_callback(self.steps, state.detach().clone())
        for index in range(self.steps - 1, -1, -1):
            links = self._links(state, condition, provider, dense=dense)
            time = torch.full(
                (state.shape[0],), index / self.steps, device=state.device, dtype=state.dtype
            )
            prediction = self.denoiser(state, time, condition, links)
            beta = self.betas[index]
            alpha = self.alphas[index]
            alpha_bar = self.alpha_bar[index]
            mean = (state - beta * prediction / (1 - alpha_bar).sqrt()) / alpha.sqrt()
            if index:
                previous_alpha_bar = self.alpha_bar[index - 1]
                variance = beta * (1 - previous_alpha_bar) / (1 - alpha_bar)
                next_noise = torch.randn(
                    state.shape, dtype=state.dtype, device=state.device, generator=generator
                )
                mean = mean + variance.sqrt() * self._project(next_noise, condition)
            state = self._project(mean, condition)
            if progress_callback is not None:
                progress_callback(index, state.detach().clone())
            if index == 0 or index % save_every == 0:
                trail.append(state.detach().clone())
                times.append(index)
        return SampleTrail(state, tuple(trail), tuple(times), self.steps)


__all__ = [
    "ConditionalPacketDenoiser",
    "DiffusionCondition",
    "FrozenPacketDiffusion",
    "PacketLinks",
    "SampleTrail",
    "SpatialConditionalPacketDenoiser",
    "centered_active",
    "dense_access",
    "heat_from_logits",
]

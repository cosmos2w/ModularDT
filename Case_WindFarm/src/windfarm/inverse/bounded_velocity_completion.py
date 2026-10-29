"""Target-free Wind adapters and matched training for normalized velocity diffusion.

This opt-in API deliberately lives beside the historical packet-completion
implementation. It uses the same task schema, frozen u4910 action builder, and
native sensor measurements, while keeping normalized coordinates distinct
from the old sigmoid-logit parameterization.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from honf_inverse_core.models.bounded_velocity import (
    VELOCITY_CHECKPOINT_FORMAT,
    VELOCITY_PREDICTION_TYPE,
    VELOCITY_STATE_PARAMETERIZATION,
    BoundedVelocityPacketDiffusion,
    bounded_provider_proxy,
    cosine_velocity_coefficients,
    require_velocity_checkpoint,
)
from honf_inverse_core.models.frozen_packet_diffusion import (
    DiffusionCondition,
    LinksForCandidate,
    PacketLinks,
    dense_access,
)
from honf_runtime.compat import load_trusted_checkpoint
from torch import nn

from windfarm.geometry import HUB_HEIGHT_D, POSITIONAL_SCALE_D
from windfarm.inverse.packet_completion import (
    DESIGN_LOWER_D,
    DESIGN_UPPER_D,
    SENSOR_OBSERVED_INDICES,
    WindCandidateKnown,
    WindCompletionKnown,
    WindCompletionTask,
    WindFixedActionCandidateInterfaceBuilder,
    candidate_only_known,
    known_from_wind_task,
    packet_links_from_wind_interface,
    permute_wind_task,
)


def fit_wind_sensor_velocity_transform(
    train_tasks: Sequence[WindCompletionTask],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit fixed three-channel input normalization from unique training rows only."""

    if not train_tasks or any(task.partition != "train" for task in train_tasks):
        raise ValueError("Wind sensor normalization accepts training tasks only")
    rows: dict[int, np.ndarray] = {}
    observed = np.asarray(SENSOR_OBSERVED_INDICES, dtype=np.int64)
    for task in train_tasks:
        if int(task.row_index) not in rows:
            panel = np.asarray(task.sensors.reference_velocity_mps, dtype=np.float64)
            if panel.shape != (24, 3) or not np.isfinite(panel).all():
                raise ValueError("Training sensor values must be finite [24,3] arrays")
            rows[int(task.row_index)] = panel[observed]
    values = np.concatenate(tuple(rows.values()), axis=0)
    center = values.mean(axis=0).astype(np.float32)
    scale = np.maximum(values.std(axis=0), 0.05).astype(np.float32)
    return center, scale


def normalized_wind_design(
    centers_D: np.ndarray,
    *,
    lower_D: Sequence[float] = DESIGN_LOWER_D,
    upper_D: Sequence[float] = DESIGN_UPPER_D,
) -> np.ndarray:
    """Affine map from physical XY to the declared common normalized box."""

    centers = np.asarray(centers_D, dtype=np.float32)
    lower = np.asarray(lower_D, dtype=np.float32)
    upper = np.asarray(upper_D, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] < 2:
        raise ValueError("Wind centers must have shape [M,>=2]")
    if lower.shape != (2,) or upper.shape != (2,) or np.any(lower >= upper):
        raise ValueError("Target-free Wind design bounds must be ordered XY vectors")
    xy = centers[:, :2]
    if not np.isfinite(xy).all() or np.any(xy < lower[None, :]) or np.any(xy > upper[None, :]):
        raise ValueError("Clean Wind centers fall outside the declared common design box")
    return (2.0 * (xy - lower[None, :]) / (upper - lower)[None, :] - 1.0).astype(np.float32)


def decode_normalized_wind_design(
    normalized_xy: torch.Tensor,
    lower_D: torch.Tensor,
    upper_D: torch.Tensor,
) -> torch.Tensor:
    """Decode normalized coordinates affinely; no sigmoid or hidden row bounds."""

    if normalized_xy.shape[-1] != 2 or lower_D.shape[-1] != 2 or upper_D.shape != lower_D.shape:
        raise ValueError("Normalized Wind design and bounds must have XY width")
    if not bool(torch.isfinite(normalized_xy).all()):
        raise ValueError("Normalized Wind design must be finite")
    if bool(torch.any(lower_D >= upper_D)):
        raise ValueError("Wind design bounds must be strictly ordered")
    return lower_D + 0.5 * (normalized_xy + 1.0) * (upper_D - lower_D)


def wind_velocity_condition_from_task(
    task: WindCompletionTask,
    *,
    sensor_velocity_center_mps: Sequence[float],
    sensor_velocity_scale_mps: Sequence[float],
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, DiffusionCondition, WindCompletionKnown]:
    """Return normalized training target separately from target-free condition.

    The physical design bounds are the single shared dataset contract
    ``[-15,15]^2 D``. They do not depend on row-native meshes or hidden centers.
    Sensor normalization is caller supplied and must have been fitted from
    training observations only.
    """

    known = known_from_wind_task(task)
    condition = wind_velocity_condition_from_known(
        known,
        sensor_velocity_center_mps=sensor_velocity_center_mps,
        sensor_velocity_scale_mps=sensor_velocity_scale_mps,
        device=device,
    )
    target = torch.as_tensor(
        normalized_wind_design(np.asarray(task.clean_centers_D, dtype=np.float32)[:, :2])[None],
        device=torch.device(device),
    )
    return target, condition, known


def wind_velocity_condition_from_known(
    known: WindCompletionKnown,
    *,
    sensor_velocity_center_mps: Sequence[float],
    sensor_velocity_scale_mps: Sequence[float],
    device: torch.device | str = "cpu",
) -> DiffusionCondition:
    """Build a Wind condition from visible geometry and sensors alone.

    This entry point has no clean hidden-coordinate argument and is the only
    condition builder used by the native development sampler.
    """

    center = np.asarray(sensor_velocity_center_mps, dtype=np.float32)
    scale = np.asarray(sensor_velocity_scale_mps, dtype=np.float32)
    if center.shape != (3,) or scale.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Wind velocity normalization must contain three finite channel centers/scales")
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("Wind velocity normalization scales must be finite and positive")
    visible = np.asarray(known.visible_mask, dtype=bool)
    visible_centers = np.asarray(known.template_case.module_centers, dtype=np.float32)
    if visible_centers.ndim != 2 or visible_centers.shape[1] != 3:
        raise ValueError("Target-free Wind template centers must be [M,3]")
    count = int(visible_centers.shape[0])
    if visible.shape != (count,) or not visible.any() or visible.all():
        raise ValueError("Wind task must retain visible turbines and at least one generated turbine")
    if np.any(visible_centers[~visible, :2] != 0.0):
        raise ValueError("Target-free Wind template must zero every hidden center placeholder")

    module_features = np.concatenate(
        (
            np.asarray(known.template_case.module_features, dtype=np.float32),
            np.broadcast_to(
                np.asarray(known.template_case.global_context, dtype=np.float32)[None, :],
                (count, known.template_case.global_context.size),
            ),
            np.asarray(known.template_case.module_centers[:, :2], dtype=np.float32)
            / np.asarray((50.0, 38.0), dtype=np.float32),
            visible[:, None].astype(np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    observed_velocity = np.asarray(known.observed_velocity_mps, dtype=np.float32)
    if observed_velocity.shape != (len(SENSOR_OBSERVED_INDICES), 3):
        raise ValueError("Wind task observed values do not match the fixed sensor role")
    sensor_features = np.concatenate(
        (
            np.asarray(known.observed_coordinates_D, dtype=np.float32)
            / np.asarray(POSITIONAL_SCALE_D, dtype=np.float32),
            (observed_velocity - center[None, :]) / scale[None, :],
        ),
        axis=1,
    ).astype(np.float32)
    target_device = torch.device(device)
    visible_state = np.zeros((count, 2), dtype=np.float32)
    visible_state[visible] = normalized_wind_design(visible_centers[visible, :2])
    condition = DiffusionCondition(
        known_state=torch.as_tensor(visible_state[None], device=target_device),
        design_mask=torch.as_tensor((~visible)[None], device=target_device, dtype=torch.bool),
        module_valid=torch.ones((1, count), device=target_device, dtype=torch.bool),
        module_features=torch.as_tensor(module_features[None], device=target_device),
        sensor_features=torch.as_tensor(sensor_features[None], device=target_device),
        sensor_valid=torch.ones((1, len(SENSOR_OBSERVED_INDICES)), device=target_device, dtype=torch.bool),
        design_lower=torch.as_tensor(DESIGN_LOWER_D[None], device=target_device),
        design_upper=torch.as_tensor(DESIGN_UPPER_D[None], device=target_device),
    )
    condition.validate(2, int(module_features.shape[-1]), int(sensor_features.shape[-1]))
    return condition


class WindVelocityCandidatePacketProvider:
    """Build frozen packet links from affine normalized coordinates only."""

    def __init__(
        self,
        known: WindCompletionKnown,
        builder: WindFixedActionCandidateInterfaceBuilder,
    ) -> None:
        if known.template_case.run is not None:
            raise ValueError("Velocity candidate provider cannot retain a clean native field view")
        if np.any(np.asarray(known.template_case.module_centers)[~known.visible_mask, :2] != 0.0):
            raise ValueError("Velocity candidate template must zero hidden center placeholders")
        self.known: WindCandidateKnown = candidate_only_known(known)
        self.builder = builder
        self.calls = 0
        self.last_candidate_centers_D: torch.Tensor | None = None
        self.last_interface: Any | None = None

    def candidate_centers(
        self,
        state: torch.Tensor,
        condition: DiffusionCondition,
    ) -> torch.Tensor:
        """Decode a clipped provider proxy and restore exact visible positions."""

        if state.shape[0] != 1 or state.shape[-1] != 2:
            raise ValueError("Wind velocity candidate state must have shape [1,M,2]")
        count = int(self.known.template_case.n_turbines)
        if state.shape[1] != count or condition.design_mask.shape != (1, count):
            raise ValueError("Velocity candidate state and Wind module count differ")
        if condition.design_lower is None or condition.design_upper is None:
            raise ValueError("Stable Wind candidate decoding requires supplied common design bounds")
        lower = condition.design_lower[0].to(device=state.device, dtype=state.dtype)
        upper = condition.design_upper[0].to(device=state.device, dtype=state.dtype)
        if not torch.equal(lower, state.new_tensor(DESIGN_LOWER_D)) or not torch.equal(
            upper, state.new_tensor(DESIGN_UPPER_D)
        ):
            raise ValueError("Stable Wind adapter accepts only the frozen common [-15,15]^2 D domain")
        proxy = bounded_provider_proxy(state.detach(), condition)
        xy = decode_normalized_wind_design(proxy[0], lower, upper)
        visible = torch.as_tensor(self.known.visible_mask, device=state.device, dtype=torch.bool)
        exact_visible = torch.as_tensor(
            self.known.template_case.module_centers[:, :2], device=state.device, dtype=state.dtype
        )
        xy = torch.where(visible[:, None], exact_visible, xy)
        height = state.new_full((count, 1), float(HUB_HEIGHT_D))
        return torch.cat((xy, height), dim=-1)

    @torch.no_grad()
    def __call__(self, state: torch.Tensor, condition: DiffusionCondition) -> PacketLinks:
        centers = self.candidate_centers(state, condition)
        interface = self.builder(centers, self.known)
        observed = torch.as_tensor(self.known.observed_coordinates_D, device=centers.device, dtype=centers.dtype)
        links = packet_links_from_wind_interface(interface, observed)
        # The new spatial conditioner explicitly consumes the horizontal
        # physical frame. Keep the complete XYZ geometry inside the frozen
        # provider, while exposing only XY link coordinates to the denoiser.
        links = replace(
            links,
            module_coordinates=links.module_coordinates[..., :2],
            sensor_coordinates=links.sensor_coordinates[..., :2],
        )
        self.calls += 1
        self.last_candidate_centers_D = centers.detach().clone()
        self.last_interface = interface
        return links


def _task_id(task: WindCompletionTask) -> dict[str, Any]:
    return {
        "row_index": int(task.row_index),
        "layout_index": int(task.template_case.layout_index),
        "wind_direction_deg": float(task.template_case.wind_direction_deg),
        "turbine_count": int(task.template_case.n_turbines),
        "visible_mask": np.asarray(task.visible_mask, dtype=np.uint8).tolist(),
    }


def _frozen_snapshots(modules: Mapping[str, nn.Module]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        for name, module in modules.items()
    }


def _assert_frozen_equal(before: Mapping[str, Mapping[str, torch.Tensor]], modules: Mapping[str, nn.Module]) -> None:
    for name, module in modules.items():
        current = module.state_dict()
        saved = before[name]
        if set(current) != set(saved) or any(not torch.equal(saved[key], current[key].detach().cpu()) for key in saved):
            raise RuntimeError(f"Frozen inverse provider module {name!r} changed during training")


@dataclass(frozen=True)
class WindMatchedVelocityResult:
    graph_model: BoundedVelocityPacketDiffusion
    dense_model: BoundedVelocityPacketDiffusion
    graph_losses: tuple[float, ...]
    dense_losses: tuple[float, ...]
    updates_per_arm: int
    batch_size: int
    training_examples_per_arm: int
    endpoint_examples_per_arm: int
    shared_provider_calls: int
    optimizer_elapsed_seconds: float
    review_elapsed_seconds: float
    wall_elapsed_seconds: float


@dataclass(frozen=True)
class WindVelocityOverfitResult:
    model: BoundedVelocityPacketDiffusion
    updates: int
    batch_size: int
    examples_seen: int
    endpoint_examples: int
    provider_calls: int
    losses: tuple[float, ...]
    clean_mse_before_by_time: Mapping[str, float]
    clean_mse_after_by_time: Mapping[str, float]
    clean_mae_before_by_time: Mapping[str, float]
    clean_mae_after_by_time: Mapping[str, float]
    observation_response_before_by_time: Mapping[str, float]
    observation_response_after_by_time: Mapping[str, float]
    audit_times: tuple[float, ...]


def train_matched_wind_velocity(
    train_tasks: Sequence[WindCompletionTask],
    *,
    denoiser_template: nn.Module,
    provider_factory: Callable[[WindCompletionKnown], LinksForCandidate],
    frozen_modules: Mapping[str, nn.Module],
    sensor_velocity_center_mps: Sequence[float],
    sensor_velocity_scale_mps: Sequence[float],
    source_freeze_id: str,
    updates: int,
    batch_size: int = 4,
    steps: int = 20,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-2,
    pure_noise_fraction: float = 0.25,
    seed: int = 20261002,
    device: torch.device | str = "cpu",
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = 50,
    resume_checkpoint: Path | None = None,
    attempt_log_path: Path | None = None,
    stop_after_utc: datetime | None = None,
    review_callback: Callable[[int, BoundedVelocityPacketDiffusion, BoundedVelocityPacketDiffusion], None]
    | None = None,
) -> WindMatchedVelocityResult:
    """Train matched graph/dense arms with shared initialization and four-task steps."""

    if not train_tasks or any(task.partition != "train" for task in train_tasks):
        raise ValueError("Stable Wind inverse fitting accepts training tasks only")
    layouts = {int(task.template_case.layout_index) for task in train_tasks}
    masks_by_layout: dict[int, set[tuple[int, ...]]] = defaultdict(set)
    for task in train_tasks:
        layout = int(task.template_case.layout_index)
        masks_by_layout[layout].add(tuple(map(int, np.asarray(task.visible_mask, dtype=bool))))
    if len(layouts) < 24 or any(len(masks_by_layout[layout]) < 2 for layout in layouts):
        raise ValueError("Stable inverse cohort needs 24 layouts and multiple hidden-slot choices per layout")
    if not source_freeze_id.strip():
        raise ValueError("Stable inverse training needs the predeclared forward-only freeze identity")
    if not frozen_modules:
        raise ValueError("Frozen forward and organizer modules must be supplied")
    if updates < 1 or batch_size != 4 or steps != 20:
        raise ValueError("Stable inverse uses positive updates, batch four, and exactly 20 DDIM steps")
    if learning_rate <= 0.0 or weight_decay < 0.0 or not 0.0 < pure_noise_fraction <= 0.5:
        raise ValueError("Stable inverse optimizer or endpoint-stratification settings are invalid")
    if checkpoint_every < 1:
        raise ValueError("Stable inverse checkpoint interval must be positive")
    center = np.asarray(sensor_velocity_center_mps, dtype=np.float32)
    scale = np.asarray(sensor_velocity_scale_mps, dtype=np.float32)
    if center.shape != (3,) or scale.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Training-owned sensor center/scale must have three finite values")
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("Training-owned sensor scales must be finite and positive")

    target_device = torch.device(device)
    frozen_before = _frozen_snapshots(frozen_modules)
    for module in frozen_modules.values():
        module.eval().requires_grad_(False)
    graph = BoundedVelocityPacketDiffusion(deepcopy(denoiser_template), steps=steps).to(target_device)
    dense = BoundedVelocityPacketDiffusion(deepcopy(denoiser_template), steps=steps).to(target_device)
    if set(graph.denoiser.state_dict()) != set(dense.denoiser.state_dict()) or any(
        not torch.equal(graph.denoiser.state_dict()[key], dense.denoiser.state_dict()[key])
        for key in graph.denoiser.state_dict()
    ):
        raise RuntimeError("Matched stable inverse arms did not start from identical weights")
    graph.train()
    dense.train()
    optimizers = (
        torch.optim.AdamW(graph.denoiser.parameters(), lr=learning_rate, weight_decay=weight_decay),
        torch.optim.AdamW(dense.denoiser.parameters(), lr=learning_rate, weight_decay=weight_decay),
    )
    index_rng = np.random.default_rng(int(seed))
    noise_rng = torch.Generator(device="cpu").manual_seed(int(seed) ^ 0x391D)
    graph_losses: list[float] = []
    dense_losses: list[float] = []
    endpoint_examples = 0
    shared_provider_calls = 0
    start_update = 0
    elapsed_before = 0.0
    optimizer_elapsed_before = 0.0
    review_elapsed_before = 0.0
    task_ids = [_task_id(task) for task in train_tasks]
    expected_checkpoint_contract: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": "wind_stable_velocity_inverse",
        "source_freeze_id": source_freeze_id,
        "seed": int(seed),
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "pure_noise_fraction": float(pure_noise_fraction),
        "train_task_ids": task_ids,
        "sensor_velocity_center_mps": center.tolist(),
        "sensor_velocity_scale_mps": scale.tolist(),
        "prediction_type": VELOCITY_PREDICTION_TYPE,
        "state_parameterization": VELOCITY_STATE_PARAMETERIZATION,
    }
    if resume_checkpoint is not None:
        if checkpoint_dir is None:
            raise ValueError("Velocity resume needs its checkpoint directory")
        checkpoint_path = Path(resume_checkpoint).resolve()
        if not checkpoint_path.is_relative_to(Path(checkpoint_dir).resolve()):
            raise ValueError("Velocity resume checkpoint must belong to this experiment directory")
        saved = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
        if not isinstance(saved, Mapping):
            raise TypeError("Velocity checkpoint must be a mapping")
        require_velocity_checkpoint(saved)
        for key, value in expected_checkpoint_contract.items():
            if saved.get(key) != value:
                raise ValueError(f"Velocity checkpoint contract mismatch for {key}")
        start_update = int(saved.get("update_count", -1))
        if start_update < 1 or start_update > updates:
            raise ValueError("Velocity checkpoint update count is outside the requested cumulative target")
        graph.load_state_dict(saved["graph_state_dict"], strict=True)
        dense.load_state_dict(saved["dense_state_dict"], strict=True)
        optimizers[0].load_state_dict(saved["graph_optimizer_state_dict"])
        optimizers[1].load_state_dict(saved["dense_optimizer_state_dict"])
        index_rng.bit_generator.state = saved["index_rng_state"]
        noise_rng.set_state(saved["noise_rng_state"])
        graph_losses = list(map(float, saved["graph_losses"]))
        dense_losses = list(map(float, saved["dense_losses"]))
        endpoint_examples = int(saved.get("endpoint_examples", -1))
        shared_provider_calls = int(saved.get("shared_provider_calls", -1))
        if endpoint_examples < 0 or shared_provider_calls != start_update * batch_size:
            raise ValueError("Velocity checkpoint exposure/provider counts are inconsistent")
        if len(graph_losses) != start_update or len(dense_losses) != start_update:
            raise ValueError("Velocity checkpoint loss histories do not match its update count")
        elapsed_before = float(saved.get("elapsed_seconds_total", 0.0))
        optimizer_elapsed_before = float(saved.get("optimizer_elapsed_seconds_total", 0.0))
        review_elapsed_before = float(saved.get("review_elapsed_seconds_total", 0.0))

    started = time.monotonic()
    completed_update = start_update
    optimizer_elapsed = optimizer_elapsed_before
    review_elapsed = review_elapsed_before

    def save_checkpoint(update_count: int) -> None:
        if checkpoint_dir is None:
            return
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        destination = checkpoint_dir / f"updates_{update_count:06d}.pt"
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        payload = {
            **expected_checkpoint_contract,
            "checkpoint_format": VELOCITY_CHECKPOINT_FORMAT,
            "update_count": int(update_count),
            "graph_state_dict": graph.state_dict(),
            "dense_state_dict": dense.state_dict(),
            "graph_optimizer_state_dict": optimizers[0].state_dict(),
            "dense_optimizer_state_dict": optimizers[1].state_dict(),
            "index_rng_state": index_rng.bit_generator.state,
            "noise_rng_state": noise_rng.get_state(),
            "graph_losses": list(graph_losses),
            "dense_losses": list(dense_losses),
            "endpoint_examples": int(endpoint_examples),
            "shared_provider_calls": int(shared_provider_calls),
            "elapsed_seconds_total": elapsed_before + time.monotonic() - started,
            "optimizer_elapsed_seconds_total": optimizer_elapsed,
            "review_elapsed_seconds_total": review_elapsed,
            "matched_initialization_exact": True,
            "forward_and_organizer_frozen": True,
            "examples_seen_per_arm": int(update_count * batch_size),
        }
        torch.save(payload, temporary)
        os.replace(temporary, destination)

    for update_index in range(start_update, int(updates)):
        if stop_after_utc is not None and datetime.now(timezone.utc) >= stop_after_utc:
            if completed_update > 0:
                save_checkpoint(completed_update)
            break
        update_started = time.monotonic()
        chosen = index_rng.choice(len(train_tasks), size=batch_size, replace=False)
        selected = [train_tasks[int(index)] for index in chosen]
        update_times = torch.rand((batch_size,), generator=noise_rng)
        endpoint_count = max(1, round(batch_size * pure_noise_fraction))
        endpoint_slots = index_rng.choice(batch_size, size=endpoint_count, replace=False)
        update_times[endpoint_slots] = 1.0
        endpoint_examples += int(torch.sum(update_times == 1.0).item())
        provider_calls_this_update = 0
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss_totals = [0.0, 0.0]
        for sample_index, source_task in enumerate(selected):
            target_np = np.asarray(source_task.clean_centers_D, dtype=np.float32)
            if target_np.shape[0] != int(source_task.template_case.n_turbines):
                raise ValueError("Stable inverse task turbine axis changed after selection")
            task = permute_wind_task(source_task, int(index_rng.integers(0, 2**31 - 1)))
            target, condition, known = wind_velocity_condition_from_task(
                task,
                sensor_velocity_center_mps=center,
                sensor_velocity_scale_mps=scale,
                device=target_device,
            )
            candidate_provider = provider_factory(known)
            cached_state: torch.Tensor | None = None
            cached_links: PacketLinks | None = None

            def shared_provider(
                proxy_state: torch.Tensor,
                current_condition: DiffusionCondition,
                provider_fn: LinksForCandidate = candidate_provider,
            ) -> PacketLinks:
                nonlocal cached_state, cached_links, provider_calls_this_update
                if cached_state is None:
                    cached_state = proxy_state.detach().clone()
                    cached_links = provider_fn(proxy_state, current_condition)
                    provider_calls_this_update += 1
                elif not torch.equal(proxy_state.detach(), cached_state):
                    raise RuntimeError("Matched velocity arms received different candidate proxies")
                assert cached_links is not None
                return cached_links

            noise = torch.randn(target.shape, generator=noise_rng, dtype=target.dtype).to(target_device)
            time_value = update_times[sample_index].to(device=target_device, dtype=target.dtype).reshape(1)
            for arm_index, (model, is_dense) in enumerate(((graph, False), (dense, True))):
                loss = model.training_loss(
                    target,
                    condition,
                    shared_provider,
                    dense=is_dense,
                    time=time_value,
                    noise=noise,
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"Stable Wind inverse {('I-v-G', 'I-v-dense')[arm_index]} loss became nonfinite "
                        f"at update {update_index + 1}"
                    )
                (loss / float(batch_size)).backward()
                loss_totals[arm_index] += float(loss.detach().cpu()) / float(batch_size)
            if cached_links is None:
                raise RuntimeError("Matched velocity provider failed to build shared links")
        shared_provider_calls += provider_calls_this_update
        if provider_calls_this_update != batch_size:
            raise RuntimeError("Each paired batch must use one shared candidate provider call per example")
        grad_norms = [
            float(
                torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), max_norm=1.0, error_if_nonfinite=True)
                .detach()
                .cpu()
            )
            for model in (graph, dense)
        ]
        for optimizer in optimizers:
            optimizer.step()
        optimizer_step_seconds = time.monotonic() - update_started
        optimizer_elapsed += optimizer_step_seconds
        graph_losses.append(loss_totals[0])
        dense_losses.append(loss_totals[1])
        update_count = update_index + 1
        completed_update = update_count
        review_seconds = 0.0
        if review_callback is not None and update_count in (1, 200, 500, 1000):
            review_started = time.monotonic()
            review_callback(update_count, graph, dense)
            review_seconds = time.monotonic() - review_started
            review_elapsed += review_seconds
        if attempt_log_path is not None:
            attempt_log_path.parent.mkdir(parents=True, exist_ok=True)
            with attempt_log_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "experiment_id": "wind_stable_velocity_inverse",
                            "update_count": update_count,
                            "batch_size": batch_size,
                            "examples_seen_per_arm": update_count * batch_size,
                            "pure_noise_endpoint_exposure": int(update_times.max().item() == 1.0),
                            "pure_noise_endpoint_examples": int(torch.sum(update_times == 1.0).item()),
                            "cumulative_endpoint_examples": endpoint_examples,
                            "shared_provider_calls_this_update": provider_calls_this_update,
                            "cumulative_shared_provider_calls": shared_provider_calls,
                            "graph_loss": loss_totals[0],
                            "dense_loss": loss_totals[1],
                            "graph_gradient_norm_preclip": grad_norms[0],
                            "dense_gradient_norm_preclip": grad_norms[1],
                            "optimizer_step_seconds": optimizer_step_seconds,
                            "review_seconds": review_seconds,
                            "elapsed_seconds": elapsed_before + time.monotonic() - started,
                        },
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
        if update_count % checkpoint_every == 0 or update_count == updates:
            save_checkpoint(update_count)

    graph.eval()
    dense.eval()
    _assert_frozen_equal(frozen_before, frozen_modules)
    return WindMatchedVelocityResult(
        graph_model=graph,
        dense_model=dense,
        graph_losses=tuple(graph_losses),
        dense_losses=tuple(dense_losses),
        updates_per_arm=int(completed_update),
        batch_size=int(batch_size),
        training_examples_per_arm=int(completed_update * batch_size),
        endpoint_examples_per_arm=int(endpoint_examples),
        shared_provider_calls=int(shared_provider_calls),
        optimizer_elapsed_seconds=float(optimizer_elapsed),
        review_elapsed_seconds=float(review_elapsed),
        wall_elapsed_seconds=float(elapsed_before + time.monotonic() - started),
    )


@torch.no_grad()
def evaluate_matched_wind_velocity_denoising(
    graph_model: BoundedVelocityPacketDiffusion,
    dense_model: BoundedVelocityPacketDiffusion,
    tasks_by_partition: Mapping[str, Sequence[WindCompletionTask]],
    *,
    provider_factory: Callable[[WindCompletionKnown], LinksForCandidate],
    sensor_velocity_center_mps: Sequence[float],
    sensor_velocity_scale_mps: Sequence[float],
    seed: int,
    audit_times: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
) -> dict[str, Any]:
    """Compare fixed-noise masked-v and clean-coordinate errors by partition.

    This is a teacher-supervised denoising review, not a reverse trajectory.
    Both arms receive identical target-free conditions, q-noisy states, and the
    same graph links; dense access is derived from that same provider result.
    Development targets are read only here and never enter model conditions or
    optimizer batches.
    """

    if not tasks_by_partition or any(partition not in {"train", "development"} for partition in tasks_by_partition):
        raise ValueError("Denoising review accepts only predeclared train/development partitions")
    if any(task.partition != partition for partition, tasks in tasks_by_partition.items() for task in tasks):
        raise ValueError("Denoising review task partition labels do not match their panel")
    if not audit_times or any(not 0.0 < float(value) <= 1.0 for value in audit_times):
        raise ValueError("Denoising review times must lie in (0, 1]")
    center = np.asarray(sensor_velocity_center_mps, dtype=np.float32)
    scale = np.asarray(sensor_velocity_scale_mps, dtype=np.float32)
    if center.shape != (3,) or scale.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Review input normalization must use the fixed three-channel train transform")
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("Review input scales must be finite and positive")
    if graph_model.steps != dense_model.steps or graph_model.design_dim != dense_model.design_dim:
        raise ValueError("Matched denoising review models have incompatible diffusion contracts")

    device = next(graph_model.parameters()).device
    graph_training, dense_training = graph_model.training, dense_model.training
    graph_model.eval()
    dense_model.eval()
    metric_sums: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    coordinate_counts: dict[str, int] = {}
    task_records: dict[str, list[dict[str, Any]]] = {}
    provider_calls = 0
    coordinate_scale_D = float(np.mean((DESIGN_UPPER_D[:2] - DESIGN_LOWER_D[:2]) / 2.0))

    for partition, tasks in tasks_by_partition.items():
        if not tasks:
            raise ValueError(f"Denoising review {partition} panel is empty")
        metric_sums[partition] = {
            arm: {
                str(float(current_time)): {
                    key: 0.0
                    for key in (
                        "v_squared_error_sum",
                        "v_absolute_error_sum",
                        "clean_squared_error_sum",
                        "clean_absolute_error_sum",
                    )
                }
                for current_time in audit_times
            }
            for arm in ("I-v-G", "I-v-dense")
        }
        coordinate_counts[partition] = 0
        task_records[partition] = [_task_id(task) for task in tasks]
        for task_index, task in enumerate(tasks):
            target, condition, known = wind_velocity_condition_from_task(
                task,
                sensor_velocity_center_mps=center,
                sensor_velocity_scale_mps=scale,
                device=device,
            )
            provider = provider_factory(known)
            noise_generator = torch.Generator(device="cpu").manual_seed(
                int(seed) + 1_000_003 * (0 if partition == "train" else 1) + task_index
            )
            fixed_noise = torch.randn(
                target.shape,
                generator=noise_generator,
                dtype=target.dtype,
                device="cpu",
            ).to(device)
            generated = condition.design_mask.bool().unsqueeze(-1)
            generated_coordinates = int(condition.design_mask.sum().item()) * target.shape[-1]
            if generated_coordinates < 1:
                raise ValueError("Denoising review panel must contain a hidden design coordinate")
            coordinate_counts[partition] += generated_coordinates

            for current_time in audit_times:
                time_value = target.new_full((target.shape[0],), float(current_time))
                noisy, target_velocity = graph_model.noisy_state_and_velocity_target(
                    target, condition, time_value, fixed_noise
                )
                shared_links = graph_model._links(noisy, condition, provider, dense=False)
                provider_calls += 1
                full_links = dense_access(shared_links, condition)
                predictions = (
                    graph_model.denoiser(noisy, time_value, condition, shared_links),
                    dense_model.denoiser(noisy, time_value, condition, full_links),
                )
                a, b = cosine_velocity_coefficients(time_value)
                for arm, predicted_velocity in zip(("I-v-G", "I-v-dense"), predictions):
                    clean_estimate = (
                        a.reshape(target.shape[0], 1, 1) * noisy - b.reshape(target.shape[0], 1, 1) * predicted_velocity
                    )
                    velocity_error = (predicted_velocity - target_velocity).masked_select(generated)
                    clean_error = (clean_estimate - target).masked_select(generated)
                    values = metric_sums[partition][arm][str(float(current_time))]
                    values["v_squared_error_sum"] += float(velocity_error.square().sum().cpu())
                    values["v_absolute_error_sum"] += float(velocity_error.abs().sum().cpu())
                    values["clean_squared_error_sum"] += float(clean_error.square().sum().cpu())
                    values["clean_absolute_error_sum"] += float(clean_error.abs().sum().cpu())

    result: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_type": "fixed-noise masked-v and clean-coordinate denoising review; no reverse trajectories",
        "task_counts": {partition: len(tasks) for partition, tasks in tasks_by_partition.items()},
        "task_masks": task_records,
        "audit_times": [float(value) for value in audit_times],
        "fixed_noise_seed": int(seed),
        "sensor_normalization_source": "unique training rows only",
        "coordinate_scale_D_per_normalized_unit": coordinate_scale_D,
        "candidate_provider_calls_shared_between_arms": provider_calls,
        "coordinate_counts": coordinate_counts,
        "arms": {},
    }
    for arm in ("I-v-G", "I-v-dense"):
        result["arms"][arm] = {}
        for partition in tasks_by_partition:
            count = coordinate_counts[partition]
            by_time: dict[str, Any] = {}
            for current_time in audit_times:
                key = str(float(current_time))
                sums = metric_sums[partition][arm][key]
                v_mse = sums["v_squared_error_sum"] / count
                clean_mse = sums["clean_squared_error_sum"] / count
                by_time[key] = {
                    "masked_v_mse_normalized": v_mse,
                    "masked_v_mae_normalized": sums["v_absolute_error_sum"] / count,
                    "clean_coordinate_mse_normalized": clean_mse,
                    "clean_coordinate_mae_normalized": sums["clean_absolute_error_sum"] / count,
                    "clean_coordinate_rmse_D": float(np.sqrt(clean_mse) * coordinate_scale_D),
                    "clean_coordinate_mae_D": (sums["clean_absolute_error_sum"] / count * coordinate_scale_D),
                }
            result["arms"][arm][partition] = by_time
    if graph_training:
        graph_model.train()
    if dense_training:
        dense_model.train()
    return result


def train_wind_velocity_overfit_diagnostic(
    diagnostic_tasks: Sequence[WindCompletionTask],
    *,
    denoiser_template: nn.Module,
    provider_factory: Callable[[WindCompletionKnown], LinksForCandidate],
    frozen_modules: Mapping[str, nn.Module],
    sensor_velocity_center_mps: Sequence[float],
    sensor_velocity_scale_mps: Sequence[float],
    updates: int = 100,
    batch_size: int = 4,
    steps: int = 20,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-2,
    pure_noise_fraction: float = 0.25,
    seed: int = 20261002,
    device: torch.device | str = "cpu",
    attempt_log_path: Path | None = None,
    stop_after_utc: datetime | None = None,
) -> WindVelocityOverfitResult:
    """Fit one fresh graph model on at most eight train tasks as a bounded diagnostic.

    The returned model is diagnostic-only. No checkpoint is emitted, and the
    primary matched pair must be rebuilt from the untouched caller template.
    Fixed-noise single-step clean estimates audit conditioning without a reverse
    trajectory or a development target.
    """

    if not 4 <= len(diagnostic_tasks) <= 8:
        raise ValueError("Velocity overfit diagnostic accepts at most eight tasks")
    if any(task.partition != "train" for task in diagnostic_tasks):
        raise ValueError("Velocity overfit diagnostic accepts training tasks only")
    layouts = {int(task.template_case.layout_index) for task in diagnostic_tasks}
    if len(layouts) != len(diagnostic_tasks):
        raise ValueError("Diagnostic tasks must use distinct layouts")
    if not frozen_modules or not 1 <= int(updates) <= 100 or batch_size != 4 or steps != 20:
        raise ValueError("Diagnostic needs frozen providers, 1..100 updates, batch four, and 20 steps")
    if learning_rate <= 0.0 or weight_decay < 0.0 or not 0.0 < pure_noise_fraction <= 0.5:
        raise ValueError("Diagnostic optimizer or endpoint-stratification settings are invalid")
    center = np.asarray(sensor_velocity_center_mps, dtype=np.float32)
    scale = np.asarray(sensor_velocity_scale_mps, dtype=np.float32)
    if center.shape != (3,) or scale.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Diagnostic input normalization must have three finite channel centers/scales")
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("Diagnostic input scales must be finite and positive")
    target_device = torch.device(device)
    frozen_before = _frozen_snapshots(frozen_modules)
    for module in frozen_modules.values():
        module.eval().requires_grad_(False)
    model = BoundedVelocityPacketDiffusion(deepcopy(denoiser_template), steps=steps).to(target_device)
    model.train()
    optimizer = torch.optim.AdamW(model.denoiser.parameters(), lr=learning_rate, weight_decay=weight_decay)
    index_rng = np.random.default_rng(int(seed))
    noise_rng = torch.Generator(device="cpu").manual_seed(int(seed) ^ 0x391D)
    audit_times = (0.25, 0.5, 0.75, 1.0)
    audit_noise_by_task = {
        index: torch.randn(
            (int(task.template_case.n_turbines), 2),
            generator=torch.Generator(device="cpu").manual_seed(int(seed) + 8000 + index),
        )
        .unsqueeze(0)
        .to(target_device)
        for index, task in enumerate(diagnostic_tasks)
    }

    def audit() -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        model.eval()
        squared: dict[str, list[float]] = {str(value): [] for value in audit_times}
        absolute: dict[str, list[float]] = {str(value): [] for value in audit_times}
        response: dict[str, list[float]] = {str(value): [] for value in audit_times}
        with torch.no_grad():
            for task_index, task in enumerate(diagnostic_tasks):
                target, condition, known = wind_velocity_condition_from_task(
                    task,
                    sensor_velocity_center_mps=center,
                    sensor_velocity_scale_mps=scale,
                    device=target_device,
                )
                provider = provider_factory(known)
                noise = audit_noise_by_task[task_index]
                generated = condition.design_mask.to(target.dtype).unsqueeze(-1)
                denominator = (generated.sum() * target.shape[-1]).clamp_min(1.0)
                for current_time in audit_times:
                    time_tensor = target.new_full((1,), current_time)
                    noisy, _target_velocity = model.noisy_state_and_velocity_target(
                        target, condition, time_tensor, noise
                    )
                    links = model._links(noisy, condition, provider, dense=False)
                    predicted = model.denoiser(noisy, time_tensor, condition, links)
                    a, b = cosine_velocity_coefficients(time_tensor)
                    raw_clean = a.reshape(1, 1, 1) * noisy - b.reshape(1, 1, 1) * predicted
                    error = (raw_clean - target) * generated
                    label = str(current_time)
                    squared[label].append(float((error.square().sum() / denominator).cpu()))
                    absolute[label].append(float((error.abs().sum() / denominator).cpu()))
                    no_observations = replace(condition, sensor_valid=torch.zeros_like(condition.sensor_valid))
                    no_observed_prediction = model.denoiser(noisy, time_tensor, no_observations, links)
                    no_observed_clean = a.reshape(1, 1, 1) * noisy - b.reshape(1, 1, 1) * no_observed_prediction
                    response[label].append(
                        float((((raw_clean - no_observed_clean).abs() * generated).sum() / denominator).cpu())
                    )
        model.train()
        return (
            {key: float(np.mean(value)) for key, value in squared.items()},
            {key: float(np.mean(value)) for key, value in absolute.items()},
            {key: float(np.mean(value)) for key, value in response.items()},
        )

    mse_before, mae_before, response_before = audit()
    losses: list[float] = []
    endpoint_examples = 0
    provider_calls = 0
    started = time.monotonic()
    completed_updates = 0
    for update_index in range(int(updates)):
        if stop_after_utc is not None and datetime.now(timezone.utc) >= stop_after_utc:
            break
        chosen = index_rng.choice(len(diagnostic_tasks), size=batch_size, replace=False)
        selected = [diagnostic_tasks[int(index)] for index in chosen]
        times = torch.rand((batch_size,), generator=noise_rng)
        endpoint_count = max(1, round(batch_size * pure_noise_fraction))
        endpoint_slots = index_rng.choice(batch_size, size=endpoint_count, replace=False)
        times[endpoint_slots] = 1.0
        endpoint_examples += int(torch.sum(times == 1.0).item())
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for sample_index, source_task in enumerate(selected):
            task = permute_wind_task(source_task, int(index_rng.integers(0, 2**31 - 1)))
            target, condition, known = wind_velocity_condition_from_task(
                task,
                sensor_velocity_center_mps=center,
                sensor_velocity_scale_mps=scale,
                device=target_device,
            )
            provider = provider_factory(known)
            noise = torch.randn(target.shape, generator=noise_rng, dtype=target.dtype).to(target_device)
            time_value = times[sample_index].to(device=target_device, dtype=target.dtype).reshape(1)
            loss = model.training_loss(target, condition, provider, time=time_value, noise=noise)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Velocity diagnostic loss became nonfinite at update {update_index + 1}")
            (loss / float(batch_size)).backward()
            loss_total += float(loss.detach().cpu()) / float(batch_size)
            provider_calls += 1
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), max_norm=1.0, error_if_nonfinite=True)
            .detach()
            .cpu()
        )
        optimizer.step()
        losses.append(loss_total)
        completed_updates = update_index + 1
        if attempt_log_path is not None:
            attempt_log_path.parent.mkdir(parents=True, exist_ok=True)
            with attempt_log_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "experiment_id": "wind_stable_velocity_overfit_diagnostic",
                            "update_count": update_index + 1,
                            "optimizer_steps": 1,
                            "batch_size": batch_size,
                            "examples_seen": (update_index + 1) * batch_size,
                            "endpoint_examples_this_update": int(torch.sum(times == 1.0).item()),
                            "cumulative_endpoint_examples": endpoint_examples,
                            "candidate_provider_calls_this_update": batch_size,
                            "cumulative_candidate_provider_calls": provider_calls,
                            "loss": loss_total,
                            "gradient_norm_preclip": gradient_norm,
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
    mse_after, mae_after, response_after = audit()
    model.eval()
    _assert_frozen_equal(frozen_before, frozen_modules)
    return WindVelocityOverfitResult(
        model=model,
        updates=int(completed_updates),
        batch_size=batch_size,
        examples_seen=int(completed_updates * batch_size),
        endpoint_examples=endpoint_examples,
        provider_calls=provider_calls,
        losses=tuple(losses),
        clean_mse_before_by_time=mse_before,
        clean_mse_after_by_time=mse_after,
        clean_mae_before_by_time=mae_before,
        clean_mae_after_by_time=mae_after,
        observation_response_before_by_time=response_before,
        observation_response_after_by_time=response_after,
        audit_times=audit_times,
    )


__all__ = [
    "WindMatchedVelocityResult",
    "WindVelocityCandidatePacketProvider",
    "WindVelocityOverfitResult",
    "decode_normalized_wind_design",
    "fit_wind_sensor_velocity_transform",
    "normalized_wind_design",
    "train_matched_wind_velocity",
    "train_wind_velocity_overfit_diagnostic",
    "wind_velocity_condition_from_known",
    "wind_velocity_condition_from_task",
]

"""Bounded staged fitting with explicit optimizer-update accounting."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from channelthermal.interaction_evidence.response_dataset import ResponseStencil
from channelthermal.interaction_evidence.types import EvidenceSplit

from .algebra import MixedResponseSpec, predict_stencil
from .contracts import AbsoluteOperator
from .historical import HistoricalValueSource
from .losses import ThermalLossScales, compute_stencil_loss_terms, historical_absolute_value_loss

RESPONSE_TERMS = ("value", "finite", "mixed", "decision", "constraint")


@dataclass(frozen=True)
class TrainingStage:
    """An inclusive start and exclusive stop in completed optimizer updates."""

    name: str
    start_update: int
    stop_update: int
    active_terms: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or self.start_update < 0 or self.stop_update <= self.start_update:
            raise ValueError("Training stages need a name and a positive update interval.")
        terms = tuple(dict.fromkeys(self.active_terms))
        if not terms or any(term not in RESPONSE_TERMS for term in terms):
            raise ValueError(f"Training stages may use only {RESPONSE_TERMS}.")
        if "value" not in terms:
            raise ValueError("Every stage must retain the absolute value loss.")
        object.__setattr__(self, "active_terms", terms)


@dataclass(frozen=True)
class StagedTrainingConfig:
    """Small fit-recipe limits subordinate to the round-wide resource ceiling."""

    arm: str = "B_response"
    max_optimizer_updates: int = 2000
    max_epochs: int = 500
    total_optimizer_update_ceiling: int = 20000
    checkpoint_every_updates: int = 25
    max_wall_seconds: float | None = None
    review_updates: tuple[int, ...] = (100, 300, 1000, 2000)
    random_seed: int = 2317
    stages: tuple[TrainingStage, ...] = (
        TrainingStage("value_warmup", 0, 100, ("value",)),
        TrainingStage("finite_response", 100, 300, ("value", "finite")),
        TrainingStage(
            "paired_response_and_decision",
            300,
            2000,
            ("value", "finite", "mixed", "decision", "constraint"),
        ),
    )

    def __post_init__(self) -> None:
        if self.arm not in {"B_value", "B_response"}:
            raise ValueError("arm must be B_value or B_response.")
        if min(self.max_optimizer_updates, self.max_epochs, self.total_optimizer_update_ceiling) <= 0:
            raise ValueError("Training update and epoch ceilings must be positive.")
        if self.max_optimizer_updates > self.total_optimizer_update_ceiling:
            raise ValueError("Fit-recipe updates cannot exceed the formal arm update ceiling.")
        if self.checkpoint_every_updates <= 0:
            raise ValueError("checkpoint_every_updates must be positive.")
        if self.max_wall_seconds is not None and (
            not np.isfinite(self.max_wall_seconds) or self.max_wall_seconds <= 0.0
        ):
            raise ValueError("max_wall_seconds must be positive and finite when supplied.")
        reviews = tuple(sorted({int(value) for value in self.review_updates}))
        if any(value <= 0 or value > self.max_optimizer_updates for value in reviews):
            raise ValueError("Review updates must lie within the configured update ceiling.")
        stages = tuple(self.stages)
        if not stages or stages[0].start_update != 0 or stages[-1].stop_update < self.max_optimizer_updates:
            raise ValueError("Stages must cover update zero through the configured ceiling.")
        for left, right in pairwise(stages):
            if left.stop_update != right.start_update:
                raise ValueError("Training stages must be contiguous and nonoverlapping.")
        object.__setattr__(self, "review_updates", reviews)
        object.__setattr__(self, "stages", stages)

    def active_terms(self, completed_updates: int) -> tuple[str, ...]:
        if completed_updates < 0:
            raise ValueError("completed_updates must be nonnegative.")
        if self.arm == "B_value":
            return ("value",)
        for stage in self.stages:
            if stage.start_update <= completed_updates < stage.stop_update:
                return stage.active_terms
        # Keep the last explicitly approved objective set beyond its declared
        # interval. In particular, do not silently make unresolved mixed
        # labels active through this fallback.
        return self.stages[-1].active_terms

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> StagedTrainingConfig:
        raw = dict(values)
        stages = tuple(
            TrainingStage(
                name=str(item["name"]),
                start_update=int(item["start_update"]),
                stop_update=int(item["stop_update"]),
                active_terms=tuple(item["active_terms"]),
            )
            for item in raw.pop("stages", ())
        )
        if stages:
            raw["stages"] = stages
        if "review_updates" in raw:
            raw["review_updates"] = tuple(raw["review_updates"])
        return cls(**raw)


def load_staged_training_config(path: str | None = None, *, arm: str = "B_response") -> StagedTrainingConfig:
    """Load the checked-in paired-arm schedule without accepting ad hoc defaults."""

    import json
    from pathlib import Path

    config_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[3] / "configs" / "response_control_staged.json"
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    fields = {
        "arm": arm,
        "max_optimizer_updates": payload["max_optimizer_updates"],
        "max_epochs": payload["max_epochs"],
        "total_optimizer_update_ceiling": payload["formal_arm_update_ceiling"],
        "checkpoint_every_updates": payload["checkpoint_every_updates"],
        "max_wall_seconds": payload.get("max_wall_seconds"),
        "review_updates": tuple(payload["review_updates"]),
        "random_seed": payload["random_seed"],
        "stages": tuple(payload["stages"]),
    }
    return StagedTrainingConfig.from_mapping(fields)


def calibrate_gradient_weights(
    loss_batches: Sequence[Mapping[str, torch.Tensor]],
    parameters: Iterable[torch.nn.Parameter],
    *,
    reference_term: str = "value",
) -> Mapping[str, float]:
    """Freeze term multipliers from training-only shared-parameter gradients.

    Each term is normalized to the RMS gradient norm of the reference term.
    The caller must build these losses only from training families; a missing
    or zero-gradient term is surfaced rather than assigned an arbitrary value.
    """

    params = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not params or not loss_batches:
        raise ValueError("Gradient calibration needs trainable parameters and training batches.")
    squared_norms: dict[str, list[float]] = {}
    saw_reference = False
    for batch in loss_batches:
        if reference_term in batch:
            saw_reference = True
        batch_norms = _gradient_squared_norms(batch, params)
        for name, norm_squared in batch_norms.items():
            squared_norms.setdefault(name, []).append(norm_squared)
    return _weights_from_squared_norms(squared_norms, reference_term, saw_reference)


def _gradient_squared_norms(
    batch: Mapping[str, torch.Tensor], parameters: Sequence[torch.nn.Parameter]
) -> dict[str, float]:
    norms: dict[str, float] = {}
    scalar_losses = [
        (name, loss)
        for name, loss in batch.items()
        if isinstance(loss, torch.Tensor) and loss.numel() == 1 and loss.requires_grad
    ]
    if any(not isinstance(loss, torch.Tensor) or loss.numel() != 1 for loss in batch.values()):
        bad = next(name for name, loss in batch.items() if not isinstance(loss, torch.Tensor) or loss.numel() != 1)
        raise TypeError(f"Calibration loss {bad!r} must be a scalar torch tensor.")
    for index, (name, loss) in enumerate(scalar_losses):
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=index + 1 < len(scalar_losses),
            allow_unused=True,
        )
        norm_squared = sum(
            float(gradient.detach().double().square().sum().cpu())
            for gradient in gradients
            if gradient is not None
        )
        if norm_squared > 0.0 and np.isfinite(norm_squared):
            norms[name] = norm_squared
    return norms


def _weights_from_squared_norms(
    squared_norms: Mapping[str, Sequence[float]],
    reference_term: str,
    saw_reference: bool,
) -> Mapping[str, float]:
    if not saw_reference or reference_term not in squared_norms:
        raise ValueError(f"Reference loss {reference_term!r} has no finite nonzero gradient.")
    rms_norm = {
        name: float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64))))
        for name, values in squared_norms.items()
    }
    reference_norm = rms_norm[reference_term]
    if not np.isfinite(reference_norm) or reference_norm <= 0.0:
        raise ValueError("Reference gradient norm must be positive and finite.")
    # A missing response can be a physically unresolved target, and a zero
    # gradient can be a model-side insensitivity. Neither case becomes a
    # zero-trained weight. Only objectives with a measured nonzero gradient
    # receive calibrated multipliers; callers leave the rest unknown.
    return MappingProxyType({name: reference_norm / norm for name, norm in rms_norm.items()})


def calibrate_operator_weights(
    operator: AbsoluteOperator,
    training_stencils: Sequence[ResponseStencil],
    *,
    scales: ThermalLossScales,
    mixed_specs: Sequence[MixedResponseSpec] = (),
    historical_value_source: HistoricalValueSource | None = None,
    parameters: Iterable[torch.nn.Parameter],
    device: torch.device | str | None = None,
) -> Mapping[str, float]:
    """Calibrate against the same value objective used by each fit update."""

    if not training_stencils:
        raise ValueError("At least one training stencil is required for calibration.")
    if any(stencil.split is not EvidenceSplit.TRAIN for stencil in training_stencils):
        raise ValueError("Gradient scaling may use training stencils only.")
    params = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not params:
        raise ValueError("Gradient calibration needs trainable parameters.")
    historical_record = None
    if historical_value_source is not None:
        if not historical_value_source.case_ids:
            raise ValueError("Historical gradient calibration needs a train case.")
        historical_record = historical_value_source.load(historical_value_source.case_ids[0])
    squared_norms: dict[str, list[float]] = {}
    saw_reference = False
    for stencil in training_stencils:
        predictions = predict_stencil(operator, stencil, device=device)
        terms = compute_stencil_loss_terms(
            predictions, stencil, scales=scales, mixed_specs=mixed_specs
        )
        calibrated_terms = dict(terms.terms)
        if historical_record is not None:
            if "value" not in calibrated_terms:
                raise ValueError("Historical calibration requires a stencil value loss.")
            calibrated_terms["value"] = calibrated_terms["value"] + historical_absolute_value_loss(
                operator, historical_record, scales=scales, device=device
            )
        saw_reference = saw_reference or "value" in calibrated_terms
        for name, norm_squared in _gradient_squared_norms(calibrated_terms, params).items():
            squared_norms.setdefault(name, []).append(norm_squared)
        del predictions, terms
    return _weights_from_squared_norms(squared_norms, "value", saw_reference)


@dataclass(frozen=True)
class TrainingStep:
    completed_update: int
    attempted_optimizer_step: int
    training_stencil_index: int
    stage: str
    active_terms: tuple[str, ...]
    total_loss: float
    term_losses: Mapping[str, float]
    historical_case_id: str | None = None


@dataclass(frozen=True)
class StagedFitResult:
    arm: str
    initial_update: int
    actual_optimizer_updates: int
    attempted_optimizer_steps: int
    final_update: int
    total_attempted_optimizer_steps: int
    completed_epochs: int
    calibrated_weights: Mapping[str, float]
    history: tuple[TrainingStep, ...]
    stopped_at_review: bool
    stopped_for_wall_time: bool
    wall_seconds: float


class TrainingCheckpointError(RuntimeError):
    """A checkpoint callback failed at a known optimizer-update boundary."""

    def __init__(self, reason: str, completed_updates: int, attempted_steps: int):
        super().__init__(reason)
        self.completed_updates = int(completed_updates)
        self.attempted_steps = int(attempted_steps)


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    arm: str,
    completed_updates: int,
    config: StagedTrainingConfig,
    calibrated_weights: Mapping[str, float],
    sampler_rng_state: object | None = None,
    remaining_order: Sequence[int] = (),
    attempted_optimizer_steps: int = 0,
    historical_case_order: Sequence[str] = (),
    historical_next_index: int = 0,
) -> dict[str, Any]:
    """Construct resumable model/optimizer/RNG state for atomic caller storage."""

    import random

    model_devices = {parameter.device for parameter in model.parameters() if parameter.device.type == "cuda"}
    cuda_rng = (
        {str(device): torch.cuda.get_rng_state(device) for device in model_devices}
        if model_devices
        else None
    )
    import copy

    return {
        "schema_version": 1,
        "arm": arm,
        "actual_optimizer_updates": int(completed_updates),
        "attempted_optimizer_steps": int(attempted_optimizer_steps),
        "model": {name: value.detach().clone() for name, value in model.state_dict().items()},
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": copy.deepcopy(np.random.get_state()),
        "torch_rng_state": torch.get_rng_state().clone(),
        "cuda_rng_state_by_model_device": cuda_rng,
        "sampler_rng_state": copy.deepcopy(sampler_rng_state),
        "sampler_remaining_order": [int(index) for index in remaining_order],
        "historical_case_order": [str(case_id) for case_id in historical_case_order],
        "historical_next_index": int(historical_next_index),
        "calibrated_loss_weights": dict(calibrated_weights),
        "training_config": {
            "arm": config.arm,
            "max_optimizer_updates": config.max_optimizer_updates,
            "max_epochs": config.max_epochs,
            "total_optimizer_update_ceiling": config.total_optimizer_update_ceiling,
            "checkpoint_every_updates": config.checkpoint_every_updates,
            "max_wall_seconds": config.max_wall_seconds,
            "review_updates": list(config.review_updates),
            "random_seed": config.random_seed,
            "stages": [
                {
                    "name": stage.name,
                    "start_update": stage.start_update,
                    "stop_update": stage.stop_update,
                    "active_terms": list(stage.active_terms),
                }
                for stage in config.stages
            ],
        },
    }


def restore_checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    payload: Mapping[str, Any],
    *,
    config: StagedTrainingConfig,
) -> tuple[int, int, random.Random, list[int], Mapping[str, float]]:
    """Restore model, optimizer, global RNG, and local sampler state exactly."""

    if payload.get("schema_version") != 1 or payload.get("arm") != config.arm:
        raise ValueError("Resume checkpoint schema/arm does not match this training run.")
    saved_config = payload.get("training_config", {})
    expected_config = {
        "arm": config.arm,
        "max_optimizer_updates": config.max_optimizer_updates,
        "max_epochs": config.max_epochs,
        "total_optimizer_update_ceiling": config.total_optimizer_update_ceiling,
        "checkpoint_every_updates": config.checkpoint_every_updates,
        "max_wall_seconds": config.max_wall_seconds,
        "review_updates": list(config.review_updates),
        "random_seed": config.random_seed,
        "stages": [
            {
                "name": stage.name,
                "start_update": stage.start_update,
                "stop_update": stage.stop_update,
                "active_terms": list(stage.active_terms),
            }
            for stage in config.stages
        ],
    }
    if any(saved_config.get(key) != value for key, value in expected_config.items()):
        raise ValueError("Resume checkpoint training schedule differs from the requested recipe.")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    for device_name, state in (payload.get("cuda_rng_state_by_model_device") or {}).items():
        device = torch.device(device_name)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.set_rng_state(state, device=device)
    sampler = random.Random()
    sampler_state = payload.get("sampler_rng_state")
    if sampler_state is None:
        raise ValueError("Resume checkpoint has no exact response-panel sampler RNG state.")
    sampler.setstate(sampler_state)
    order = [int(index) for index in payload.get("sampler_remaining_order", ())]
    weights = {str(name): float(value) for name, value in payload.get("calibrated_loss_weights", {}).items()}
    completed = int(payload.get("actual_optimizer_updates", -1))
    attempted = int(payload.get("attempted_optimizer_steps", -1))
    if completed < 0 or attempted < completed:
        raise ValueError("Resume checkpoint update counts are invalid.")
    return completed, attempted, sampler, order, MappingProxyType(weights)


def run_staged_fit(
    operator: AbsoluteOperator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_stencils: Sequence[ResponseStencil],
    *,
    scales: ThermalLossScales,
    loss_weights: Mapping[str, float] | None = None,
    mixed_specs: Sequence[MixedResponseSpec] = (),
    historical_value_source: HistoricalValueSource | None = None,
    config: StagedTrainingConfig | None = None,
    initial_update: int = 0,
    resume_payload: Mapping[str, Any] | None = None,
    stop_at_update: int | None = None,
    device: torch.device | str | None = None,
    on_checkpoint: Callable[[Mapping[str, Any], str], None] | None = None,
    on_review: Callable[[TrainingStep], str | None] | None = None,
) -> StagedFitResult:
    """Fit a fixed training panel and count each successful optimizer step.

    ``operator`` is an absolute native or historical adapter. Both arms can
    consume the same fixed historical case order and role-aware value batch.
    Checkpoint serialization is delegated to ``on_checkpoint``; its zero-update
    invocation runs before any training step, so a schema/write failure cannot
    consume optimizer work.
    """

    config = StagedTrainingConfig() if config is None else config
    if not training_stencils or any(stencil.split is not EvidenceSplit.TRAIN for stencil in training_stencils):
        raise ValueError("run_staged_fit accepts a nonempty training-split stencil panel only.")
    if initial_update < 0 or initial_update > config.max_optimizer_updates:
        raise ValueError("initial_update must be within the configured update range.")
    historical_order = tuple(historical_value_source.case_ids) if historical_value_source is not None else ()
    if historical_value_source is not None and (not historical_order or len(set(historical_order)) != len(historical_order)):
        raise ValueError("Historical replay needs a nonempty, unique train case order.")

    total_cap = min(
        config.max_optimizer_updates,
        config.total_optimizer_update_ceiling,
        config.max_epochs * len(training_stencils),
    )
    if initial_update > total_cap:
        raise ValueError("initial_update exceeds the effective update cap for this panel.")
    if stop_at_update is not None:
        stop_at_update = int(stop_at_update)
        if stop_at_update not in config.review_updates:
            raise ValueError("stop_at_update must be one of the configured review gates.")
        if stop_at_update <= initial_update or stop_at_update > total_cap:
            raise ValueError("stop_at_update must be ahead of the resume point and within the effective cap.")

    params = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not params:
        raise ValueError("The supplied model has no trainable parameters.")
    if resume_payload is None:
        sampler = random.Random(config.random_seed)
        order: list[int] = []
        attempted_total = 0
        weights = dict(loss_weights or {"value": 1.0})
    else:
        restored_update, attempted_total, sampler, order, saved_weights = restore_checkpoint_payload(
            model, optimizer, resume_payload, config=config
        )
        if initial_update not in (0, restored_update):
            raise ValueError("initial_update disagrees with the resume checkpoint update.")
        initial_update = restored_update
        if any(index < 0 or index >= len(training_stencils) for index in order):
            raise ValueError("Resume checkpoint contains a panel sampler index outside this training panel.")
        if len(order) != len(set(order)):
            raise ValueError("Resume checkpoint sampler order contains duplicate panel indices.")
        if loss_weights is not None and dict(loss_weights) != dict(saved_weights):
            raise ValueError("Explicit loss weights differ from the calibrated resume weights.")
        weights = dict(saved_weights)
        if historical_value_source is not None:
            if tuple(resume_payload.get("historical_case_order", ())) != historical_order:
                raise ValueError("Resume checkpoint historical train cohort/order differs.")
            if int(resume_payload.get("historical_next_index", -1)) != initial_update % len(historical_order):
                raise ValueError("Resume checkpoint historical case cursor differs.")
        elif resume_payload.get("historical_case_order"):
            raise ValueError("Resume checkpoint requires its historical value source.")
    if initial_update > total_cap:
        raise ValueError("Resume update exceeds the effective update cap for this panel.")
    if stop_at_update is not None and (
        stop_at_update <= initial_update or stop_at_update > total_cap
    ):
        raise ValueError("stop_at_update must be ahead of the resume point and within the effective cap.")
    if config.arm == "B_value":
        weights = {"value": 1.0}
    if "value" not in weights:
        raise ValueError("Every staged fit requires a calibrated absolute-value loss weight.")
    for name, value in weights.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"Loss multiplier {name!r} must be finite and nonnegative.")

    wall_started = time.monotonic()
    if on_checkpoint is not None:
        try:
            on_checkpoint(
                checkpoint_payload(
                    model,
                    optimizer,
                    arm=config.arm,
                    completed_updates=initial_update,
                    config=config,
                    calibrated_weights=weights,
                    sampler_rng_state=sampler.getstate(),
                    remaining_order=order,
                    attempted_optimizer_steps=attempted_total,
                    historical_case_order=historical_order,
                    historical_next_index=initial_update % len(historical_order) if historical_order else 0,
                ),
                "resume_preflight" if resume_payload is not None else "zero_update_preflight",
            )
        except Exception as exc:
            raise TrainingCheckpointError(
                f"Initial checkpoint preflight failed: {exc}", initial_update, 0
            ) from exc

    completed = int(initial_update)
    attempted = 0
    history: list[TrainingStep] = []
    stopped_at_review = False
    stopped_for_wall_time = False
    model.train()
    while completed < total_cap and (stop_at_update is None or completed < stop_at_update):
        if (
            config.max_wall_seconds is not None
            and time.monotonic() - wall_started >= config.max_wall_seconds
        ):
            stopped_for_wall_time = True
            break
        if not order:
            order = list(range(len(training_stencils)))
            sampler.shuffle(order)
        sample_index = order.pop()
        stencil = training_stencils[sample_index]
        active_terms = config.active_terms(completed)
        predictions = predict_stencil(operator, stencil, device=device)
        losses = compute_stencil_loss_terms(
            predictions,
            stencil,
            scales=scales,
            mixed_specs=mixed_specs,
            enabled_terms=active_terms,
        )
        active_weights = {
            name: weights[name]
            for name in active_terms
            if name in weights and name in losses.terms and weights[name] > 0.0
        }
        if not active_weights:
            if "value" not in losses.terms:
                raise ValueError("The training stencil has no observed absolute-value objective.")
            if weights["value"] <= 0.0:
                raise ValueError("The absolute-value loss multiplier must be positive.")
            active_weights = {"value": weights["value"]}
        total = losses.total(active_weights)
        historical_case_id = None
        historical_loss = None
        if historical_value_source is not None:
            historical_case_id = historical_order[completed % len(historical_order)]
            historical_loss = historical_absolute_value_loss(
                operator,
                historical_value_source.load(historical_case_id),
                scales=scales,
                device=device,
            )
            total = total + weights["value"] * historical_loss
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"Non-finite total loss before optimizer update {completed + 1}.")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in params):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"Non-finite gradient before optimizer update {completed + 1}.")
        attempted += 1
        attempted_total += 1
        optimizer.step()
        completed += 1
        term_losses = {name: float(value.detach().cpu()) for name, value in losses.terms.items()}
        if historical_loss is not None:
            term_losses["historical_value"] = float(historical_loss.detach().cpu())
        step = TrainingStep(
            completed_update=completed,
            attempted_optimizer_step=attempted,
            training_stencil_index=sample_index,
            stage=_stage_name(config, completed - 1),
            active_terms=tuple(active_weights),
            total_loss=float(total.detach().cpu()),
            term_losses=term_losses,
            historical_case_id=historical_case_id,
        )
        history.append(step)
        review_decision = None
        is_review = completed in config.review_updates
        if on_review is not None and is_review:
            review_decision = on_review(step)
            if review_decision not in (None, "continue", "stop"):
                raise ValueError("on_review must return None, 'continue', or 'stop'.")
        elif is_review:
            review_decision = "stop"
        if on_checkpoint is not None and (
            completed % config.checkpoint_every_updates == 0
            or completed in config.review_updates
            or completed == total_cap
            or completed == stop_at_update
        ):
            try:
                on_checkpoint(
                    checkpoint_payload(
                        model,
                        optimizer,
                        arm=config.arm,
                        completed_updates=completed,
                        config=config,
                        calibrated_weights=weights,
                        sampler_rng_state=sampler.getstate(),
                        remaining_order=order,
                        attempted_optimizer_steps=attempted_total,
                        historical_case_order=historical_order,
                        historical_next_index=completed % len(historical_order) if historical_order else 0,
                    ),
                    "training_checkpoint",
                )
            except Exception as exc:
                raise TrainingCheckpointError(
                    f"Checkpoint callback failed after {completed} optimizer updates: {exc}",
                    completed,
                    attempted,
                ) from exc
        wall_expired = (
            config.max_wall_seconds is not None
            and time.monotonic() - wall_started >= config.max_wall_seconds
        )
        if wall_expired:
            stopped_for_wall_time = True
            if on_checkpoint is not None and completed not in config.review_updates:
                try:
                    on_checkpoint(
                        checkpoint_payload(
                            model,
                            optimizer,
                            arm=config.arm,
                            completed_updates=completed,
                            config=config,
                            calibrated_weights=weights,
                            sampler_rng_state=sampler.getstate(),
                            remaining_order=order,
                            attempted_optimizer_steps=attempted_total,
                            historical_case_order=historical_order,
                            historical_next_index=completed % len(historical_order) if historical_order else 0,
                        ),
                        "wall_time_checkpoint",
                    )
                except Exception as exc:
                    raise TrainingCheckpointError(
                        f"Wall-time checkpoint failed after {completed} optimizer updates: {exc}",
                        completed,
                        attempted,
                    ) from exc
        if completed == stop_at_update or review_decision == "stop" or wall_expired:
            stopped_at_review = completed == stop_at_update or review_decision == "stop"
            break

    return StagedFitResult(
        arm=config.arm,
        initial_update=initial_update,
        actual_optimizer_updates=completed - initial_update,
        attempted_optimizer_steps=attempted,
        final_update=completed,
        total_attempted_optimizer_steps=attempted_total,
        completed_epochs=completed // len(training_stencils),
        calibrated_weights=MappingProxyType(weights),
        history=tuple(history),
        stopped_at_review=stopped_at_review,
        stopped_for_wall_time=stopped_for_wall_time,
        wall_seconds=float(time.monotonic() - wall_started),
    )


def _stage_name(config: StagedTrainingConfig, update: int) -> str:
    for stage in config.stages:
        if stage.start_update <= update < stage.stop_update:
            return stage.name
    return config.stages[-1].name


__all__ = [
    "RESPONSE_TERMS",
    "StagedFitResult",
    "StagedTrainingConfig",
    "TrainingCheckpointError",
    "TrainingStage",
    "TrainingStep",
    "calibrate_gradient_weights",
    "calibrate_operator_weights",
    "checkpoint_payload",
    "load_staged_training_config",
    "restore_checkpoint_payload",
    "run_staged_fit",
]

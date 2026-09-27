"""Paired B_value/B_response fitting from one materialized full-access state."""

from __future__ import annotations

import csv
import os
import random
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from channelthermal.interaction_evidence.response_dataset import ResponseStencil

from .algebra import MixedResponseSpec
from .contracts import AbsoluteOperator
from .historical import HistoricalValueSource
from .losses import ThermalLossScales
from .training import (
    StagedFitResult,
    StagedTrainingConfig,
    TrainingStep,
    calibrate_operator_weights,
    run_staged_fit,
)


@dataclass(frozen=True)
class PairedFitResult:
    """Results for the two arms fitted from the same initial state and panel."""

    arms: Mapping[str, StagedFitResult]
    calibrated_response_weights: Mapping[str, float]
    review_decisions: Mapping[int, str]

    def __post_init__(self) -> None:
        if set(self.arms) != {"B_value", "B_response"}:
            raise ValueError("A paired fit must contain exactly B_value and B_response.")
        values = dict(self.arms)
        if values["B_value"].actual_optimizer_updates != values["B_response"].actual_optimizer_updates:
            raise ValueError("Paired arms did not complete the same number of optimizer updates.")
        first = values["B_value"].history
        second = values["B_response"].history
        if tuple(step.training_stencil_index for step in first) != tuple(
            step.training_stencil_index for step in second
        ):
            raise ValueError("Paired arms did not use the same value-sample order.")
        object.__setattr__(self, "arms", MappingProxyType(values))
        object.__setattr__(self, "calibrated_response_weights", MappingProxyType(dict(self.calibrated_response_weights)))
        object.__setattr__(self, "review_decisions", MappingProxyType(dict(self.review_decisions)))


@dataclass(frozen=True)
class _RNGSnapshot:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: torch.Tensor
    torch_cuda: Mapping[str, torch.Tensor]


def _snapshot_rng(model: torch.nn.Module) -> _RNGSnapshot:
    cuda_devices = sorted(
        {parameter.device for parameter in model.parameters() if parameter.device.type == "cuda"},
        key=lambda device: (device.type, device.index or 0),
    )
    cuda_states = {
        str(device): torch.cuda.get_rng_state(device).clone()
        for device in cuda_devices
    }
    return _RNGSnapshot(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=MappingProxyType(cuda_states),
    )


def _restore_rng(snapshot: _RNGSnapshot) -> None:
    random.setstate(snapshot.python)
    np.random.set_state(snapshot.numpy)
    torch.set_rng_state(snapshot.torch_cpu)
    for device, state in snapshot.torch_cuda.items():
        torch.cuda.set_rng_state(state, device=torch.device(device))


def _materialized_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        if not isinstance(value, torch.Tensor) or value.device.type == "meta":
            raise TypeError(f"Model state {name!r} is not a materialized tensor.")
        state[name] = value.detach().clone()
    if not state:
        raise ValueError("A materialized target model must have a nonempty state dictionary.")
    return state


def run_paired_staged_fits(
    model: torch.nn.Module,
    operator_factory: Callable[[torch.nn.Module], AbsoluteOperator],
    optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
    training_stencils: Sequence[ResponseStencil],
    *,
    historical_value_source: HistoricalValueSource | None = None,
    scales: ThermalLossScales,
    mixed_specs: Sequence[MixedResponseSpec],
    config: StagedTrainingConfig,
    device: torch.device | str | None = None,
    on_checkpoint: Callable[[str, Mapping[str, Any], str], None] | None = None,
    on_review: Callable[[str, torch.nn.Module, TrainingStep], str | None] | None = None,
    stop_at_update: int | None = None,
    review_continuations: Sequence[int] = (),
    resume_payloads: Mapping[str, Mapping[str, Any]] | None = None,
    fixed_response_weights: Mapping[str, float] | None = None,
) -> PairedFitResult:
    """Run the matched value-only and response-aware arms.

    The input model must already be materialized and loaded from the selected
    checkpoint/refit state. The response multipliers are calibrated once from
    the training families at that state. Both arms then restart from an exact
    copy of the same model state and RNG stream, share the same deterministic
    stencil order, and use the same update/epoch caps.
    """

    if not training_stencils or any(stencil.split.value != "train" for stencil in training_stencils):
        raise ValueError("Paired fits accept training-split stencils only.")
    continuations = {int(value) for value in review_continuations}
    if not continuations.issubset(set(config.review_updates)):
        raise ValueError("Review continuations must name configured review gates.")
    if stop_at_update is not None and stop_at_update not in config.review_updates:
        raise ValueError("stop_at_update must name one configured review gate.")
    resumed = dict(resume_payloads or {})
    if resumed and set(resumed) != {"B_value", "B_response"}:
        raise ValueError("Paired resume requires both B_value and B_response checkpoint payloads.")
    if resumed:
        value_payload = resumed["B_value"]
        response_payload = resumed["B_response"]
        if int(value_payload.get("actual_optimizer_updates", -1)) != int(
            response_payload.get("actual_optimizer_updates", -2)
        ):
            raise ValueError("Paired resume checkpoints must have the same completed update.")
        if value_payload.get("sampler_rng_state") != response_payload.get("sampler_rng_state"):
            raise ValueError("Paired resume checkpoints have different panel sampler RNG states.")
        if value_payload.get("sampler_remaining_order") != response_payload.get("sampler_remaining_order"):
            raise ValueError("Paired resume checkpoints have different remaining panel orders.")
        for field in ("historical_case_order", "historical_next_index"):
            if value_payload.get(field) != response_payload.get(field):
                raise ValueError(f"Paired resume checkpoints have different {field}.")
    if resumed and fixed_response_weights is not None:
        raise ValueError("A paired resume already carries its frozen response weights.")
    initial_state = _materialized_state(model)
    rng_snapshot = _snapshot_rng(model)
    model_was_training = model.training
    if resumed:
        response_weights = {
            str(name): float(value)
            for name, value in resumed["B_response"].get("calibrated_loss_weights", {}).items()
        }
        if not response_weights:
            raise ValueError("B_response resume checkpoint has no calibrated loss weights.")
    elif fixed_response_weights is not None:
        response_weights = {
            str(name): float(value) for name, value in fixed_response_weights.items()
        }
        if not {"value", "finite", "decision", "constraint"}.issubset(response_weights):
            raise ValueError(
                "Frozen B_response weights must include value, finite, decision, and constraint terms."
            )
        if any(not torch.isfinite(torch.tensor(value)) or value <= 0.0 for value in response_weights.values()):
            raise ValueError("Frozen B_response weights must be positive and finite.")
    else:
        try:
            model.eval()
            response_operator = operator_factory(model)
            response_weights = calibrate_operator_weights(
                response_operator,
                training_stencils[:1],
                scales=scales,
                mixed_specs=mixed_specs,
                historical_value_source=historical_value_source,
                parameters=model.parameters(),
                device=device,
            )
        finally:
            model.load_state_dict(initial_state, strict=True)
            model.train(model_was_training)
            _restore_rng(rng_snapshot)

    results: dict[str, StagedFitResult] = {}
    for arm in ("B_value", "B_response"):
        if not resumed:
            model.load_state_dict(initial_state, strict=True)
        _restore_rng(rng_snapshot)
        operator = operator_factory(model)
        optimizer = optimizer_factory(model)
        arm_config = StagedTrainingConfig(
            arm=arm,
            max_optimizer_updates=config.max_optimizer_updates,
            max_epochs=config.max_epochs,
            total_optimizer_update_ceiling=config.total_optimizer_update_ceiling,
            checkpoint_every_updates=config.checkpoint_every_updates,
            max_wall_seconds=config.max_wall_seconds,
            review_updates=config.review_updates,
            random_seed=config.random_seed,
            stages=config.stages,
        )
        checkpoint_callback = None
        if on_checkpoint is not None:
            def checkpoint_callback(
                payload: Mapping[str, Any], label: str, *, current_arm: str = arm
            ) -> None:
                on_checkpoint(current_arm, payload, label)

        def review_callback(step: TrainingStep, *, current_arm: str = arm) -> str:
            callback_decision = on_review(current_arm, model, step) if on_review is not None else None
            if callback_decision not in (None, "continue", "stop"):
                raise ValueError("Paired on_review must return None, 'continue', or 'stop'.")
            if callback_decision == "continue" and step.completed_update not in continuations:
                raise ValueError(
                    "A review callback cannot continue without a recorded review_continuations entry."
                )
            if callback_decision == "stop":
                return "stop"
            if step.completed_update == stop_at_update:
                return "stop"
            return "continue" if step.completed_update in continuations else "stop"
        result = run_staged_fit(
            operator,
            model,
            optimizer,
            training_stencils,
            historical_value_source=historical_value_source,
            scales=scales,
            loss_weights=response_weights if arm == "B_response" else None,
            mixed_specs=mixed_specs,
            config=arm_config,
            resume_payload=resumed.get(arm),
            stop_at_update=stop_at_update,
            device=device,
            on_checkpoint=checkpoint_callback,
            on_review=review_callback,
        )
        results[arm] = result
        # The next arm reloads every trainable/buffer tensor from this exact
        # pre-fit state; no B_value output can initialize B_response.
        if not resumed:
            model.load_state_dict(initial_state, strict=True)
        model.train(model_was_training)
        _restore_rng(rng_snapshot)

    decision_gates = sorted(set(config.review_updates))
    decision_record = {
        gate: (
            "continue"
            if gate in continuations and (stop_at_update is None or gate < stop_at_update)
            else "stop"
        )
        for gate in decision_gates
        if stop_at_update is None or gate <= stop_at_update
    }
    return PairedFitResult(results, response_weights, decision_record)


def write_paired_training_curves(path: str | Path, result: PairedFitResult) -> Path:
    """Atomically save every optimizer-step loss as a compact CSV curve."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    term_names = sorted(
        {
            term
            for arm_result in result.arms.values()
            for step in arm_result.history
            for term in step.term_losses
        }
    )
    fields = [
        "arm",
        "completed_update",
        "attempted_optimizer_step",
        "training_stencil_index",
        "historical_case_id",
        "stage",
        "active_terms",
        "total_loss",
        *[f"loss_{term}" for term in term_names],
    ]
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.tmp-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for arm, arm_result in result.arms.items():
                for step in arm_result.history:
                    writer.writerow(_step_row(arm, step, term_names))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def _step_row(arm: str, step: TrainingStep, term_names: Sequence[str]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "arm": arm,
        "completed_update": step.completed_update,
        "attempted_optimizer_step": step.attempted_optimizer_step,
        "training_stencil_index": step.training_stencil_index,
        "historical_case_id": step.historical_case_id or "",
        "stage": step.stage,
        "active_terms": ";".join(step.active_terms),
        "total_loss": step.total_loss,
    }
    row.update({f"loss_{name}": step.term_losses.get(name, "") for name in term_names})
    return row


__all__ = ["PairedFitResult", "run_paired_staged_fits", "write_paired_training_curves"]

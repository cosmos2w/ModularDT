"""Matched value/response fitting from one materialized native model state."""

from __future__ import annotations

import csv
import json
import os
import random
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from channelthermal.interaction_evidence.response_dataset import ResponseStencil

from .algebra import MixedResponseSpec
from .contracts import AbsoluteOperator
from .historical import HistoricalValueSource
from .losses import FixedHeatNullControl, ThermalLossScales
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
    gradient_calibration: Mapping[str, Any] = field(default_factory=dict)
    buffer_integrity: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        keys = set(self.arms)
        pair_names = next(
            (prefix for prefix in ("B", "R") if keys == {f"{prefix}_value", f"{prefix}_response"}),
            None,
        )
        if pair_names is None:
            raise ValueError("A paired fit must contain exactly matching B or R value/response arms.")
        values = dict(self.arms)
        value_arm = f"{pair_names}_value"
        response_arm = f"{pair_names}_response"
        if values[value_arm].actual_optimizer_updates != values[response_arm].actual_optimizer_updates:
            raise ValueError("Paired arms did not complete the same number of optimizer updates.")
        first = values[value_arm].history
        second = values[response_arm].history
        if tuple(step.training_stencil_index for step in first) != tuple(
            step.training_stencil_index for step in second
        ):
            raise ValueError("Paired arms did not use the same value-sample order.")
        object.__setattr__(self, "arms", MappingProxyType(values))
        object.__setattr__(self, "calibrated_response_weights", MappingProxyType(dict(self.calibrated_response_weights)))
        object.__setattr__(self, "review_decisions", MappingProxyType(dict(self.review_decisions)))
        object.__setattr__(self, "gradient_calibration", MappingProxyType(dict(self.gradient_calibration)))
        object.__setattr__(self, "buffer_integrity", MappingProxyType(dict(self.buffer_integrity)))


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


def _materialized_named_buffers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot persistent and nonpersistent buffers by their registered names."""

    buffers: dict[str, torch.Tensor] = {}
    for name, value in model.named_buffers():
        if not isinstance(value, torch.Tensor) or value.device.type == "meta":
            raise TypeError(f"Model buffer {name!r} is not a materialized tensor.")
        buffers[name] = value.detach().clone()
    return buffers


def _restore_materialized_state(
    model: torch.nn.Module,
    snapshot: dict[str, torch.Tensor],
) -> list[str]:
    """Restore common persistent tensors and capture newly materialized keys.

    A strict ``load_state_dict`` can fail when calibration triggers a lazy
    persistent buffer that was absent from the first input. Existing tensors
    are restored exactly; new materialized keys are inventoried and become
    part of the shared pre-fit state used by both arms.
    """

    current = model.state_dict()
    missing = sorted(set(snapshot) - set(current))
    if missing:
        raise RuntimeError(f"Persistent model state disappeared during calibration: {missing}.")
    added = sorted(set(current) - set(snapshot))
    with torch.no_grad():
        for name, value in snapshot.items():
            current[name].copy_(value)
    for name in added:
        snapshot[name] = current[name].detach().clone()
    return added


def _named_buffer_integrity(
    model: torch.nn.Module,
    snapshot: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Compare registered buffers independently of ``state_dict`` persistence."""

    current = dict(model.named_buffers())
    previous_names = set(snapshot)
    current_names = set(current)
    missing = sorted(previous_names - current_names)
    added = sorted(current_names - previous_names)
    changed = sorted(
        name for name in previous_names & current_names
        if not torch.equal(current[name], snapshot[name])
    )
    nonpersistent_names = sorted(
        name for name in current_names if name not in model.state_dict()
    )
    persistent_additions = sorted(set(added) - set(nonpersistent_names))
    return {
        # A nonpersistent buffer may be registered lazily by a positional
        # encoder after the first input. It is inventoried and becomes part of
        # the next arm's exact baseline; persistent additions remain failures.
        "passed": not missing and not changed and not persistent_additions,
        "buffer_count": len(snapshot),
        "missing_names": missing,
        "added_names": added,
        "persistent_added_names": persistent_additions,
        "changed_names": changed,
        "nonpersistent_names": nonpersistent_names,
    }


def _restore_named_buffers(
    model: torch.nn.Module,
    snapshot: Mapping[str, torch.Tensor],
) -> None:
    """Restore all snapshotted buffers, including nonpersistent buffers."""

    current = dict(model.named_buffers())
    if set(current) != set(snapshot):
        raise RuntimeError(
            "Paired arm buffer inventory changed before restoration: "
            f"missing={sorted(set(snapshot) - set(current))}, "
            f"added={sorted(set(current) - set(snapshot))}."
        )
    with torch.no_grad():
        for name, value in snapshot.items():
            current[name].copy_(value)


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
    fixed_heat_controls: Sequence[FixedHeatNullControl] = (),
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
    prefix = config.arm[0]
    value_arm = f"{prefix}_value"
    response_arm = f"{prefix}_response"
    if not config.arm.endswith("_response"):
        raise ValueError("Paired fitting configuration must name its response arm.")
    if resumed and set(resumed) != {value_arm, response_arm}:
        raise ValueError(f"Paired resume requires both {value_arm} and {response_arm} checkpoint payloads.")
    if resumed:
        value_payload = resumed[value_arm]
        response_payload = resumed[response_arm]
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
    initial_buffers = _materialized_named_buffers(model)
    rng_snapshot = _snapshot_rng(model)
    model_was_training = model.training
    gradient_calibration: dict[str, Any] = {}
    calibration_buffer_additions: list[str] = []
    if resumed:
        response_weights = {
            str(name): float(value)
            for name, value in resumed[response_arm].get("calibrated_loss_weights", {}).items()
        }
        if not response_weights:
            raise ValueError(f"{response_arm} resume checkpoint has no calibrated loss weights.")
        gradient_calibration = {"scope": "resume_checkpoint_weights"}
    elif fixed_response_weights is not None:
        response_weights = {
            str(name): float(value) for name, value in fixed_response_weights.items()
        }
        required_weights = (
            {"value", *config.required_response_terms, *config.required_control_terms}
            if prefix == "R"
            else {"value", "finite", "decision", "constraint"}
        )
        if not required_weights.issubset(response_weights):
            raise ValueError(
                f"Frozen {response_arm} weights must include {sorted(required_weights)}."
            )
        if any(not torch.isfinite(torch.tensor(value)) or value <= 0.0 for value in response_weights.values()):
            raise ValueError(f"Frozen {response_arm} weights must be positive and finite.")
        gradient_calibration = {"scope": "frozen_response_weights"}
    else:
        try:
            model.eval()
            response_operator = operator_factory(model)
            calibration_diagnostics: dict[str, Any] = {}
            response_weights = calibrate_operator_weights(
                response_operator,
                training_stencils,
                scales=scales,
                mixed_specs=mixed_specs,
                historical_value_source=historical_value_source,
                parameters=model.parameters(),
                device=device,
                enabled_terms=tuple(
                    dict.fromkeys(
                        term
                        for stage in config.stages
                        for term in stage.active_terms
                    )
                ),
                include_feasibility_bce=config.include_feasibility_bce,
                diagnostic_sink=calibration_diagnostics,
                fixed_heat_controls=fixed_heat_controls,
            )
            gradient_calibration = {
                "scope": "equal_family_train_calibration",
                "diagnostics": calibration_diagnostics,
            }
        finally:
            persistent_additions = _restore_materialized_state(model, initial_state)
            current_buffers = _materialized_named_buffers(model)
            old_names = set(initial_buffers)
            missing_buffers = sorted(old_names - set(current_buffers))
            changed_buffers = sorted(
                name for name in old_names & set(current_buffers)
                if not torch.equal(current_buffers[name], initial_buffers[name])
            )
            if missing_buffers or changed_buffers:
                raise RuntimeError(
                    "Gradient calibration changed an existing model buffer: "
                    f"missing={missing_buffers}, changed={changed_buffers}."
                )
            calibration_buffer_additions = sorted(
                set(current_buffers) - old_names | set(persistent_additions)
            )
            initial_buffers = current_buffers
            model.train(model_was_training)
            _restore_rng(rng_snapshot)

    missing_required = set(config.required_response_terms) - set(response_weights)
    if missing_required:
        raise ValueError(
            "Train-only response calibration found no usable gradient for required terms: "
            f"{sorted(missing_required)}."
        )

    results: dict[str, StagedFitResult] = {}
    buffer_integrity: dict[str, Any] = {}
    for arm in (value_arm, response_arm):
        if not resumed:
            model.load_state_dict(initial_state, strict=True)
            _restore_named_buffers(model, initial_buffers)
        _restore_rng(rng_snapshot)
        operator = operator_factory(model)
        optimizer = optimizer_factory(model)
        arm_config = replace(config, arm=arm)
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
            loss_weights=response_weights if arm == response_arm else None,
            mixed_specs=mixed_specs,
            config=arm_config,
            resume_payload=resumed.get(arm),
            stop_at_update=stop_at_update,
            device=device,
            on_checkpoint=checkpoint_callback,
            on_review=review_callback,
            fixed_heat_controls=fixed_heat_controls,
        )
        results[arm] = result
        integrity = _named_buffer_integrity(model, initial_buffers)
        integrity["calibration_added_names"] = list(calibration_buffer_additions)
        buffer_integrity[arm] = integrity
        if not integrity["passed"]:
            raise RuntimeError(f"Paired {arm} fit changed frozen buffers: {integrity}.")
        for name in integrity["added_names"]:
            initial_buffers[name] = dict(model.named_buffers())[name].detach().clone()
        # The next arm reloads every trainable/buffer tensor from this exact
        # pre-fit state; the value-only arm never initializes the response arm.
        if not resumed:
            model.load_state_dict(initial_state, strict=True)
            _restore_named_buffers(model, initial_buffers)
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
    return PairedFitResult(
        results, response_weights, decision_record, gradient_calibration, buffer_integrity
    )


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
        "active_term_weights",
        "response_gradient_projection_blocks",
        "response_gradient_dot_before",
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
        "active_term_weights": json.dumps(dict(step.active_term_weights), sort_keys=True),
        "response_gradient_projection_blocks": ";".join(step.response_gradient_projection_blocks),
        "response_gradient_dot_before": json.dumps(dict(step.response_gradient_dot_before), sort_keys=True),
        "total_loss": step.total_loss,
    }
    row.update({f"loss_{name}": step.term_losses.get(name, "") for name in term_names})
    return row


__all__ = [
    "PairedFitResult",
    "run_paired_staged_fits",
    "write_paired_training_curves",
]

"""One case-neutral optimizer engine for matched HONF refinement campaigns.

Dataset adapters construct target-separated batches, native predictions, loss
terms, and physical metrics. This module owns visitation, accumulation,
optimizers, schedules, checkpointing, selection, and exact epoch-boundary
resume for both Thermal and Wind.
"""

from __future__ import annotations

import copy
import fcntl
import functools
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from .paths import PROJECT_ROOT
from .reproducibility import environment_snapshot, source_state_snapshot
from .run_layout import RunLayout, resolve_checkpoint


def _exclusive_training_run(function):
    """Hold one process-owned output lock across startup, fit and failure."""

    @functools.wraps(function)
    def locked(self, model, provider, output_dir, **kwargs):
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".training.lock").open("a+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Another training process owns this run directory.") from exc
            try:
                return function(self, model, provider, output, **kwargs)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    return locked


@dataclass(frozen=True)
class SamplingKey:
    """Stable key for data and query streams, independent of global RNG state."""

    seed: int
    epoch: int
    update_index: int
    microbatch_index: int
    stage: str
    arm: str

    def seed_for(self, *coordinates: Any) -> int:
        """Return a stable query/data seed that is identical across matched arms."""

        payload = json.dumps(
            [self.seed, self.epoch, self.update_index, self.microbatch_index, self.stage, coordinates],
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**32)

    def numpy_rng(self, *coordinates: Any) -> np.random.Generator:
        return np.random.default_rng(self.seed_for(*coordinates))

    def torch_generator(self, *coordinates: Any, device: torch.device | str = "cpu") -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed_for(*coordinates))
        return generator


@dataclass(frozen=True)
class TaskBatch:
    """A microbatch with an explicit inference/supervision boundary."""

    scene_inputs: Any
    receivers: Any
    targets: Any
    auxiliary: Any = None
    case_keys: tuple[Any, ...] = ()


@dataclass(frozen=True)
class LossTerm:
    """A summed differentiable numerator and its exact valid-element count."""

    numerator: torch.Tensor
    denominator: float | int | torch.Tensor
    weight: float = 1.0


@dataclass(frozen=True)
class ScheduleSpec:
    """Linear warmup, hold, then cosine decay on the absolute epoch clock."""

    peak_lr: float
    warmup_start_lr: float = 0.0
    warmup_epochs: int = 0
    hold_through_epoch: int = 1000
    total_epochs: int = 2500
    final_lr: float = 3.0e-6

    def __post_init__(self) -> None:
        if self.peak_lr <= 0 or self.final_lr <= 0 or self.warmup_start_lr < 0:
            raise ValueError("Learning rates must be finite and positive, except a zero warmup start.")
        if not all(math.isfinite(value) for value in (self.peak_lr, self.warmup_start_lr, self.final_lr)):
            raise ValueError("Learning rates must be finite.")
        if self.warmup_epochs < 0 or self.hold_through_epoch < self.warmup_epochs:
            raise ValueError("Schedule warmup/hold boundaries are inconsistent.")
        if self.total_epochs <= self.hold_through_epoch:
            raise ValueError("The schedule horizon must extend beyond its hold boundary.")
        if self.final_lr > self.peak_lr:
            raise ValueError("The final learning rate cannot exceed its peak.")

    def value(self, epoch: int) -> float:
        epoch = int(epoch)
        if epoch < 1:
            raise ValueError("Schedule epochs are one-based.")
        if self.warmup_epochs and epoch <= self.warmup_epochs:
            fraction = 1.0 if self.warmup_epochs == 1 else (epoch - 1) / (self.warmup_epochs - 1)
            return self.warmup_start_lr + fraction * (self.peak_lr - self.warmup_start_lr)
        if epoch <= self.hold_through_epoch:
            return self.peak_lr
        fraction = min(max((epoch - self.hold_through_epoch) /
                           (self.total_epochs - self.hold_through_epoch), 0.0), 1.0)
        return self.final_lr + 0.5 * (self.peak_lr - self.final_lr) * (1.0 + math.cos(math.pi * fraction))


@dataclass(frozen=True)
class OptimizerGroupSpec:
    """Named parameter selection and schedule for one AdamW parameter group."""

    name: str
    parameter_names: tuple[str, ...]
    schedule: ScheduleSpec
    weight_decay: float = 1.0e-5
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8


@dataclass(frozen=True)
class EngineConfig:
    seed: int
    microbatch_cases: int
    effective_cases: int
    total_epochs: int = 2500
    warmup_epochs: int = 500
    open_through_epoch: int = 600
    soft_through_epoch: int = 800
    monitor_every: int = 100
    gradient_clip: float = 1.0
    deterministic_case_order: bool = True
    monitor_epochs: tuple[int, ...] = ()
    checkpoint_epochs: tuple[int, ...] | None = None
    latest_every: int | None = None
    curve_every: int | None = None

    def __post_init__(self) -> None:
        if min(self.microbatch_cases, self.effective_cases, self.total_epochs, self.monitor_every) < 1:
            raise ValueError("Batch sizes, horizon, and monitoring interval must be positive.")
        if self.effective_cases < self.microbatch_cases:
            raise ValueError("Effective batch size must be at least the microbatch size.")
        if not 1 <= self.warmup_epochs <= self.open_through_epoch < self.soft_through_epoch < self.total_epochs:
            raise ValueError("Warmup/open/soft stages must be ordered inside the declared horizon.")
        if self.gradient_clip <= 0 or not math.isfinite(self.gradient_clip):
            raise ValueError("Gradient clipping must be positive and finite.")
        for name in ("monitor_epochs", "checkpoint_epochs"):
            epochs = getattr(self, name)
            if epochs is not None and (not isinstance(epochs, tuple)
                    or any(type(epoch) is not int or not 1 <= epoch <= self.total_epochs for epoch in epochs)
                    or tuple(sorted(set(epochs))) != epochs):
                raise ValueError("Declared monitoring/checkpoint epochs must be sorted unique horizon epochs.")
        for interval in (self.latest_every, self.curve_every):
            if interval is not None and (type(interval) is not int or interval < 1):
                raise ValueError("Latest-state and loss-curve intervals must be positive integers.")

    def stage_for_epoch(self, epoch: int) -> str:
        if epoch <= self.warmup_epochs:
            return "warmup"
        if epoch <= self.open_through_epoch:
            return "open"
        if epoch <= self.soft_through_epoch:
            return "soft"
        return "hard"

    def temperature_for_epoch(self, epoch: int) -> float:
        """Anneal the smooth phase from 1.0 to 0.1 on absolute epochs."""

        soft_start = self.open_through_epoch + 1
        if epoch <= soft_start:
            return 1.0
        if epoch >= self.soft_through_epoch:
            return 0.1
        fraction = (epoch - soft_start) / (self.soft_through_epoch - soft_start)
        return 1.0 + fraction * (0.1 - 1.0)


def _engine_config_payload(config: EngineConfig) -> dict[str, Any]:
    """Keep legacy identities unchanged when new optional output controls are unused."""
    payload = asdict(config)
    for name, default in (("monitor_epochs", ()), ("checkpoint_epochs", None),
                          ("latest_every", None), ("curve_every", None)):
        if payload[name] == default:
            payload.pop(name)
    return payload


def _resume_identity_amendment(saved: Mapping[str, Any], current: Mapping[str, Any],
                               *, allow_microbatch_change: bool = False) -> dict[str, Any] | None:
    """Permit only an explicit microbatch amendment; all scientific bindings stay sealed."""
    if saved == current:
        return None
    message = "Checkpoint identity, engine config, selection or optimizer schedule differs."
    if not allow_microbatch_change:
        raise ValueError(message)
    previous = saved.get("engine_config", {})
    proposed = current.get("engine_config", {})
    before, after = previous.get("microbatch_cases"), proposed.get("microbatch_cases")
    if (type(before) is not int or type(after) is not int
            or not 1 <= after <= proposed.get("effective_cases", 0) or before == after):
        raise ValueError(message)
    amended = copy.deepcopy(saved)
    amended["engine_config"]["microbatch_cases"] = after
    if amended != current:
        raise ValueError(message)
    return {"kind": "explicit_microbatch_continuation", "source_microbatch_cases": before,
            "microbatch_cases": after, "effective_cases": proposed["effective_cases"],
            "trajectory_note": "Query seeds include microbatch index; future samples and FP32 accumulation can change."}


def _render_loss_curves(history: Sequence[Mapping[str, Any]], output: Path, field_metric: str) -> None:
    """Write grouped TRAIN/VALIDATION curves and preserve historical metric meaning."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    def objective_at(row: Mapping[str, Any]) -> Mapping[str, Any]:
        objective = row.get("validation_objective")
        if not isinstance(objective, Mapping):
            validation = row.get("validation", {})
            objective = validation.get("loss_objective") if isinstance(validation, Mapping) else None
        return objective if isinstance(objective, Mapping) else {}

    def term_group(name: str) -> int:
        if name in {"operator_residual", "response"}:
            return 1
        if name in {"base_loss", "router_importance_loss", "expected_work"}:
            return 2
        if name == "reconstruction" or name.startswith("native_role/"):
            return 0
        return 2

    def term_label(name: str) -> str:
        if name == "reconstruction":
            return "Field prediction loss"
        if name == "response":
            return "Response prediction loss"
        if name == "operator_residual":
            return "Heat-balance residual loss"
        if name == "base_loss":
            return "Coarse-vs-fine prediction loss"
        if name == "router_importance_loss":
            return "Router probability loss"
        if name == "expected_work":
            return "Expected selected-read penalty"
        if name.startswith("native_role/"):
            role = name.split("/", 1)[1]
            role_labels = {
                "downstream_envelope": "Downstream-region",
                "hub_slab": "Hub-height slab",
                "near_turbine": "Near-turbine hub-slab",
                "volume": "Full-volume",
                "background": "Background-region",
            }
            return f"{role_labels.get(role, role.replace('_', ' ').title())} velocity loss"
        return name.replace("_", " ")

    wind_role_regions = {
        "volume": "All sampled native-volume cells; D is the reference rotor diameter.",
        "hub_slab": "Native cells within 0.5D vertically of hub height: |z - z_hub| <= 0.5D; D is the reference rotor diameter.",
        "downstream_envelope": (
            "Geometry-defined sampling region from each active hub: 0 < dx <= 10D and "
            "sqrt(dy^2 + dz^2) <= 1.5D; D is the reference rotor diameter. This is not a physical wake label."
        ),
        "near_turbine": (
            "Native cells within 1.5D horizontal x-y distance of an active hub and within the hub-height slab; "
            "D is the reference rotor diameter."
        ),
        "background": "Complement of the downstream region and hub-height slab; D is the reference rotor diameter.",
    }

    def term_definition(name: str) -> dict[str, str]:
        if name == "reconstruction":
            return {
                "panel": "native_prediction",
                "label": term_label(name),
                "unit": "dimensionless weighted scaled squared-error contribution",
                "formula": "Per case: mean of the scaled fluid, surface, and material mean-square errors, plus 0.05 times scaled q-proxy mean-square error. Adaptive TRAIN may blend 75% main-path loss and 25% already-computed fine-path loss; hard VALIDATION uses the main path only.",
                "weight": "Overall provider weight 1.0; q-proxy coefficient 0.05.",
            }
        if name == "response":
            return {
                "panel": "physics",
                "label": term_label(name),
                "unit": "dimensionless weighted scaled squared-error contribution",
                "formula": "Mean-square response prediction error on eligible TRAIN response targets; no response target is available in exposed VALIDATION.",
                "weight": "Recipe response_coefficient.",
            }
        if name == "operator_residual":
            return {
                "panel": "physics",
                "label": term_label(name),
                "unit": "dimensionless weighted normalized squared-residual contribution",
                "formula": "Mean-square discrete heat-balance residual normalized by the fixed TRAIN temperature scale; stored residual supervision is TRAIN-only.",
                "weight": "Recipe operator_coefficient.",
            }
        if name == "base_loss":
            return {
                "panel": "routing",
                "label": term_label(name),
                "unit": "dimensionless weighted prediction squared-error contribution",
                "formula": "Mean-square error between coarse-path and fine-path predictions, normalized by the fixed TRAIN scale; omitted on hard VALIDATION because no extra fine reference is computed.",
                "weight": "Configured approximation coefficient 0.1 divided by the squared fixed TRAIN scale.",
            }
        if name == "router_importance_loss":
            return {
                "panel": "routing",
                "label": term_label(name),
                "unit": "dimensionless weighted probability squared-error contribution",
                "formula": "Mean-square router probability error against detached residual-importance targets; omitted on hard VALIDATION because target construction requires fine-detail values.",
                "weight": "0.01.",
            }
        if name == "expected_work":
            return {
                "panel": "routing",
                "label": term_label(name),
                "unit": "dimensionless weighted expected selected-read fraction",
                "formula": "Expected selected fine reads divided by active receiver-source pairs, multiplied by the calibrated TRAIN penalty weight.",
                "weight": "TRAIN-calibrated coefficient, capped by the provider configuration.",
            }
        if name.startswith("native_role/"):
            role = name.split("/", 1)[1]
            role_region = wind_role_regions.get(role, "fixed native role sample; see task recipe")
            return {
                "panel": "native_prediction",
                "label": term_label(name),
                "unit": "dimensionless weighted normalized velocity squared-error contribution",
                "formula": "Mean-square xyz velocity error divided by the squared fixed TRAIN scale s_role^2 (s_role is in m/s); hard VALIDATION uses the deployed main path without fine-path replay. "
                          "Adaptive TRAIN blends 75% main and 25% fine-path losses outside warmup.",
                "weight": "0.2 per native role.",
                "role_region": role_region,
            }
        return {
            "panel": "routing",
            "label": term_label(name),
            "unit": "provider-defined weighted loss contribution",
            "formula": "Provider objective term; see run configuration for its specific prediction, residual, or routing definition.",
            "weight": "Provider-defined; validation weights by epoch are listed separately.",
        }

    def value_at(row: Mapping[str, Any], name: str, source: str) -> float | None:
        if source == "train":
            value = row.get("train_losses", {}).get(name)
        else:
            value = objective_at(row).get("terms", {}).get(name)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def plot_series(
        axis: Any, epochs: Sequence[Any], values: Sequence[Any], *, label: str,
        color: Any = None, linestyle: str = "-", marker: str = ".",
    ) -> int:
        numeric = np.asarray(values, dtype=np.float64)
        invalid = ~np.isfinite(numeric) | (numeric <= 0.0)
        masked_count = int(invalid.sum())
        axis.set_yscale("log")
        if masked_count < len(values):
            axis.plot(epochs, np.ma.masked_where(invalid, numeric), linestyle=linestyle,
                      marker=marker, label=label, color=color)
        elif len(values):
            axis.set_ylim(1.0, 10.0)
        return masked_count

    names: set[str] = set()
    for row in history:
        names.update(str(name) for name in row.get("train_losses", {}))
        names.update(str(name) for name in objective_at(row).get("terms", {}))
    all_names = sorted(names)
    figure = Figure(figsize=(15.5, 8.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3, squeeze=False).ravel()
    colors = {name: f"C{index % 10}" for index, name in enumerate(all_names)}
    masked_count_total = 0

    def set_panel_title(axis: Any, heading: str, explanation: str, *, fontsize: float = 9.5) -> None:
        axis.set_title(f"{heading}\n{explanation}", fontsize=fontsize, linespacing=1.1, pad=5)

    def plot_term_group(axis: Any, group_index: int) -> None:
        nonlocal masked_count_total
        group_names = [name for name in all_names if term_group(name) == group_index]
        for name in group_names:
            color = colors[name]
            for source, series_label in (("train", "TRAIN"), ("validation", "HELD-OUT VALIDATION")):
                points = [(int(row["epoch"]), value_at(row, name, source)) for row in history
                          if type(row.get("epoch")) is int and value_at(row, name, source) is not None]
                if points:
                    masked_count_total += plot_series(
                        axis, [item[0] for item in points], [item[1] for item in points],
                        label=f"{series_label}: {term_label(name)}", color=color,
                        linestyle="-" if source == "train" else "--",
                        marker="." if source == "train" else "o",
                    )
        axis.set_xlabel("Completed epoch")
        axis.grid(alpha=0.25, which="both")
        if axis.lines:
            axis.legend(fontsize=7.0, loc="best", frameon=False)
        else:
            axis.set_yscale("log")
            axis.set_ylim(1.0, 10.0)
            empty_message = {
                0: "No prediction losses saved",
                1: "No physics or response losses saved",
                2: "No approximation or routing losses saved",
            }.get(group_index, "No losses saved for this group")
            axis.text(0.5, 0.5, empty_message, ha="center", va="center",
                      transform=axis.transAxes)

    # The objective subtotal compares exactly the terms recorded on both sides.
    total_axis = axes[0]
    comparable_rows = []
    for row in history:
        objective = objective_at(row)
        if objective.get("matched_terms") and type(row.get("epoch")) is int:
            comparable_rows.append((int(row["epoch"]), objective))
    if comparable_rows:
        for key, label, color, style, marker in (
            ("training_matched_total", "TRAIN shared-name sum", "C0", "-", "."),
            ("validation_matched_total", "HELD-OUT VALIDATION shared-name sum", "C1", "--", "o"),
        ):
            points = [(epoch, objective.get(key)) for epoch, objective in comparable_rows
                      if objective.get(key) is not None]
            if points:
                masked_count_total += plot_series(
                    total_axis, [point[0] for point in points], [point[1] for point in points],
                    label=label, color=color, linestyle=style, marker=marker,
                )
        if total_axis.lines:
            total_axis.legend(fontsize=8, loc="best", frameon=False)
    else:
        total_axis.set_yscale("log")
        total_axis.set_ylim(1.0, 10.0)
        no_validation_objective = all(not objective_at(row).get("terms") for row in history)
        total_axis.text(
            0.5, 0.5,
            ("Held-out loss was not recorded in this history."
             if no_validation_objective else "No shared held-out loss terms were saved."),
            ha="center", va="center", transform=total_axis.transAxes,
        )
    adaptive_run = any(str(row.get("arm", "")) == "adaptive_detail" for row in history)
    train_validation_path_note = (
        "TRAIN: mean of per-update means\nVALIDATION: pooled valid-element mean\n"
        "Adaptive TRAIN: 25% fine-path loss replay\nVALIDATION: hard main path"
        if adaptive_run else
        "TRAIN: mean of per-update means\nVALIDATION: pooled valid-element mean\n"
        "TRAIN: phase-specific optimization path\nVALIDATION: hard main path"
    )
    set_panel_title(
        total_axis, "Shared-name loss summaries",
        f"{train_validation_path_note}",
        fontsize=8.2,
    )
    total_axis.set_xlabel("Completed epoch")
    total_axis.set_ylabel("Shared-name weighted loss sum (dimensionless)")
    total_axis.grid(alpha=0.25, which="both")

    native_axis, physics_axis, selector_axis, routing_axis, work_axis = axes[1:]
    plot_term_group(native_axis, 0)
    wind_role_names = [name for name in all_names if name.startswith("native_role/")]
    if wind_role_names:
        set_panel_title(
            native_axis, "Wind velocity losses by sampled region",
            "Hub slab: |z-z_hub|<=0.5D; downstream: 0<dx<=10D\n"
            "Downstream yz radius<=1.5D; D = reference rotor diameter",
            fontsize=8.2,
        )
    else:
        set_panel_title(native_axis, "Prediction losses", "Prediction vs reference; scaled squared error")
    native_axis.set_ylabel("Weighted loss contribution (dimensionless)")
    plot_term_group(physics_axis, 1)
    set_panel_title(physics_axis, "Physics and response losses",
                    "Discrete stored-flow residual; response prediction")
    physics_axis.set_ylabel("Weighted loss contribution (dimensionless)")
    plot_term_group(routing_axis, 2)
    set_panel_title(
        routing_axis, "Approximation and routing losses",
        "Coarse/fine prediction mismatch; router probability error vs residual-importance target\n"
        "Penalty on expected selected fine reads",
        fontsize=8.9,
    )
    routing_axis.set_ylabel("Weighted loss contribution (dimensionless)")

    selector_points = []
    for row in history:
        validation = row.get("validation", {})
        value = validation.get(field_metric) if isinstance(validation, Mapping) else None
        try:
            selector_points.append((int(row["epoch"]), float(value)))
        except (KeyError, TypeError, ValueError):
            continue
    if selector_points:
        selector_label = ("HELD-OUT VALIDATION: Held-out field prediction error"
                          if field_metric == "field_score" else f"HELD-OUT VALIDATION: {field_metric}")
        masked_count_total += plot_series(
            selector_axis, [point[0] for point in selector_points], [point[1] for point in selector_points],
            label=selector_label, color="C0", linestyle="-", marker="o",
        )
        selector_axis.legend(fontsize=8, loc="best", frameon=False)
    else:
        selector_axis.set_yscale("log")
        selector_axis.set_ylim(1.0, 10.0)
        selector_axis.text(0.5, 0.5, "No saved validation selector metric", ha="center", va="center",
                           transform=selector_axis.transAxes)
    selector_explanation = ("Normalized field MSE for checkpoint selection; not a loss"
                            if field_metric == "field_score"
                            else "Checkpoint selector, shown separately from loss")
    set_panel_title(selector_axis, "Held-out field prediction error" if field_metric == "field_score"
                    else "Held-out checkpoint metric", selector_explanation)
    selector_axis.set_xlabel("Completed epoch")
    selector_axis.set_ylabel(
        "Normalized mean-square field error (dimensionless)" if field_metric == "field_score"
        else "Provider-defined metric units")
    selector_axis.grid(alpha=0.25, which="both")

    work_definitions = {
        "active_pairs": "Active receiver-source pairs processed",
        "fine_rows": "Fine detail-head rows executed",
        "selected_detail_rows": "Rows selected by the detail route",
        "cheap_rows": "Cheap-path rows evaluated",
        "gate_rows": "Router gate rows evaluated",
        "operator_rows": "Discrete thermal residual rows evaluated",
    }
    work_keys = sorted({key for row in history for key in row.get("work_counts", {})}
                       & set(work_definitions))
    for index, key in enumerate(work_keys):
        points = []
        for row in history:
            try:
                points.append((int(row["epoch"]), float(row["work_counts"][key])))
            except (KeyError, TypeError, ValueError):
                continue
        if points:
            masked_count_total += plot_series(
                work_axis, [point[0] for point in points], [point[1] for point in points],
                label=f"{work_definitions[key]} (TRAIN)", color=f"C{index % 10}",
            )
    if work_keys and work_axis.lines:
        work_axis.legend(fontsize=7.0, loc="best", frameon=False)
    else:
        work_axis.set_yscale("log")
        work_axis.set_ylim(1.0, 10.0)
        work_axis.text(0.5, 0.5, "No recorded work counters", ha="center", va="center",
                       transform=work_axis.transAxes)
    set_panel_title(work_axis, "Recorded TRAIN work counters",
                    "Counter sums; not measured executor time or savings")
    work_axis.set_xlabel("Completed epoch")
    work_axis.set_ylabel("Provider counter sum (rows or pairs / epoch)")
    work_axis.grid(alpha=0.25, which="both")
    if masked_count_total:
        routing_axis.text(
            0.02, 0.02,
            f"{masked_count_total} nonpositive/nonfinite values masked for log display; history unchanged.",
            ha="left", va="bottom", fontsize=8, transform=routing_axis.transAxes,
        )
    arms = sorted({str(row["arm"]) for row in history if row.get("arm")})
    case_counts = sorted({int(row["case_visits"]) for row in history if row.get("case_visits") is not None})
    scope = ""
    if len(case_counts) == 1:
        scope = f"{case_counts[0]} TRAIN cases/epoch"
    elif case_counts:
        scope = f"{case_counts[0]}–{case_counts[-1]} TRAIN cases/epoch"
    title_parts = [", ".join(arms) if arms else "training"]
    if scope:
        title_parts.append(scope)
    title_parts.append("TRAIN optimization vs exposed VALIDATION hard inference")
    figure.suptitle(" | ".join(title_parts), fontsize=13)
    epochs = sorted({int(row["epoch"]) for row in history if type(row.get("epoch")) is int})
    if epochs:
        limits = (epochs[0] - 0.5, epochs[-1] + 0.5) if len(epochs) == 1 else (epochs[0], epochs[-1])
        for axis in axes:
            axis.set_xlim(*limits)

    validation_weights: dict[str, dict[str, float]] = {}
    for row in history:
        objective = objective_at(row)
        try:
            epoch = int(row["epoch"])
        except (KeyError, TypeError, ValueError):
            continue
        for name, weight in objective.get("term_weights", {}).items():
            validation_weights.setdefault(str(name), {})[str(epoch)] = float(weight)
    metadata = {
        "figure": "loss_curves",
        "layout": "2x3 grouped overview",
        "panel_definitions": {
        "shared_name_loss_summary": "Sum same-named weighted loss terms recorded in both TRAIN and hard VALIDATION; this summary is not directly comparable because TRAIN uses a mean of macro-update means and validation pools valid elements. Adaptive TRAIN may include 25% fine-path loss replay; validation uses hard main-path loss only.",
            "native_prediction": "Visible note: prediction versus reference, scaled squared error. Validation uses the deployed hard-route main prediction and omits full-detail replay.",
            "physics": "Visible note: discrete heat-balance residual and response prediction losses. Thermal response and residual targets are TRAIN-only when absent from the exposed validation batch.",
            "field_selector": f"Visible note: normalized mean-square field error for {field_metric}; held-out selector used for checkpoint selection, shown separately from loss.",
            "routing": "Visible note: coarse-versus-fine prediction error, router probability error against residual-importance targets, and penalty on expected selected fine reads. Fine references and importance targets are unavailable on hard validation.",
            "train_work": "Visible note: TRAIN counter sums only, not measured executor time or savings. These counts do not establish sparse executor savings. Training includes full-detail replay where configured; validation work is not recorded.",
        },
        "term_definitions": {name: term_definition(name) for name in all_names},
        "validation_objective_weights_by_epoch": validation_weights,
        "recorded_value_convention": "TRAIN curves show weighted macro-update mean loss contributions. Validation curves apply the recorded provider weight after summing numerators and valid-element denominators over exposed validation batches.",
        "selector_metric_unit": "dimensionless normalized mean-square field error for field_score; otherwise provider-defined units",
        "training_loss_weight_note": "Each term definition lists its formula and fixed or calibrated provider weight. Validation objective weights by epoch are listed separately. TRAIN histories record already-weighted contributions.",
        "training_reduction": "Arithmetic mean of weighted macro-update means; each optimizer update has equal weight, including a partial final update.",
        "validation_reduction": "Pooled numerator divided by pooled valid-element denominator, then multiplied by provider term weight.",
        "prediction_path_comparison": "TRAIN follows the phase-specific optimization path and may include 25% fine-path loss replay; held-out loss uses deployed hard inference on the main prediction only.",
        "directly_comparable": False,
        "work_counter_definitions": {key: work_definitions[key] for key in work_keys},
        "validation_scope": "exposed held-out validation with hard inference; not an independent TEST result",
    }
    _atomic_json(output / "loss_curves_metadata.json", metadata)
    for suffix in ("pdf", "png"):
        destination = output / f"loss_curves.{suffix}"
        temporary = output / f".loss_curves.{os.getpid()}.tmp.{suffix}"
        figure.savefig(temporary, format=suffix, dpi=140)
        os.replace(temporary, destination)
    figure.clear()

@dataclass(frozen=True)
class SelectionPolicy:
    field_metric: str = "field_score"
    response_guard_metric: str | None = None
    maximum_response_ratio: float = 1.10

    def __post_init__(self) -> None:
        if not self.field_metric:
            raise ValueError("A field metric is required for checkpoint selection.")
        if not math.isfinite(self.maximum_response_ratio) or self.maximum_response_ratio <= 0:
            raise ValueError("The response guard must be a positive finite ratio.")


class TaskProvider(Protocol):
    """Dataset-owned data, native physics, objective and metric contract."""

    def identity_payload(self) -> Mapping[str, Any]: ...

    def epoch_cases(self, epoch: int, seed: int) -> Sequence[Any]: ...

    def make_batch(self, case_keys: Sequence[Any], key: SamplingKey) -> TaskBatch: ...

    def loss_denominators(
        self, batches: Sequence[TaskBatch], phase: str, arm: str,
    ) -> Mapping[str, float]: ...

    def make_scene(self, scene_inputs: Any) -> Any: ...

    def predict_native(
        self, model: nn.Module, scene: Any, receivers: Any, execution_mode: str, phase: str,
        epoch: int, temperature: float,
    ) -> tuple[Any, Any]: ...

    def loss_terms(
        self, predictions: Any, targets: Any, phase: str, auxiliary_state: Any,
    ) -> Mapping[str, LossTerm]: ...

    def validation_batches(self) -> Iterable[TaskBatch]: ...

    def validation_metrics(self, predictions: Any, targets: Any, auxiliary_state: Any) -> Mapping[str, Any]: ...

    def reduce_native_metrics(self, records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]: ...

    def validation_loss_terms(
        self, predictions: Any, targets: Any, auxiliary_state: Any, *, batch: TaskBatch, arm: str,
    ) -> Mapping[str, LossTerm]: ...

    def optimizer_groups(self, model: nn.Module, arm: str, stage: str) -> Sequence[OptimizerGroupSpec]: ...

    def work_counts(self, batch: TaskBatch, predictions: Any, auxiliary_state: Any) -> Mapping[str, int | float]: ...


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return copy.deepcopy(value)


def capture_rng_state() -> dict[str, Any]:
    """Capture all process RNGs needed for exact epoch-boundary continuation."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def named_optimizer_state(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Serialize AdamW moments by stable model parameter name."""

    name_by_parameter = {id(parameter): name for name, parameter in model.named_parameters()}
    state: dict[str, Any] = {}
    group_names = []
    for group in optimizer.param_groups:
        names = []
        for parameter in group["params"]:
            name = name_by_parameter.get(id(parameter))
            if name is None:
                raise ValueError("Optimizer contains a parameter not owned by the model.")
            names.append(name)
            if parameter in optimizer.state:
                state[name] = _cpu_tree(optimizer.state[parameter])
        group_names.append({"name": group["group_name"], "parameter_names": names})
    return {"state_by_name": state, "groups": group_names}


def _load_named_optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    payload: Mapping[str, Any] | None,
) -> None:
    if payload is None:
        return
    states = payload.get("state_by_name", payload)
    if not isinstance(states, Mapping):
        raise TypeError("Named optimizer seed must map parameter names to optimizer states.")
    parameters = dict(model.named_parameters())
    for name, saved_state in states.items():
        if name not in parameters:
            raise ValueError(f"Optimizer seed refers to missing model parameter {name!r}.")
        if not isinstance(saved_state, Mapping):
            raise TypeError(f"Optimizer state for {name!r} is not a mapping.")
        parameter = parameters[name]
        restored = {}
        for key, value in saved_state.items():
            if torch.is_tensor(value) and key != "step":
                restored[key] = value.to(parameter.device).clone()
            elif torch.is_tensor(value):
                restored[key] = value.detach().cpu().clone()
            else:
                restored[key] = copy.deepcopy(value)
        optimizer.state[parameter] = restored


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_clean_stop_request(output: Path, name: str) -> dict[str, Any] | None:
    request_path = output / name
    if not request_path.is_file():
        return None
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request, dict) or not isinstance(request.get("request_id"), str):
        raise TypeError("Clean-stop requests must be JSON objects with a stable request_id.")
    if not request["request_id"]:
        raise ValueError("Clean-stop request_id must be nonempty.")
    return request


def _consume_acknowledged_clean_stop(output: Path, name: str, layout: RunLayout) -> None:
    """Remove only a leftover request whose exact identity was already acked."""

    request = _read_clean_stop_request(output, name)
    if request is None:
        return
    acknowledgement_path = layout.read_path("clean_stop_acknowledged.json")
    if not acknowledgement_path.is_file():
        return
    acknowledgement = json.loads(acknowledgement_path.read_text(encoding="utf-8"))
    if (not isinstance(acknowledgement, dict)
            or acknowledgement.get("request_id") != request["request_id"]):
        return
    request_tag = hashlib.sha256(request["request_id"].encode("utf-8")).hexdigest()[:16]
    _atomic_json(layout.write_path(f"clean_stop_consumed_{request_tag}.json"), {
        "request": request,
        "acknowledgement": acknowledgement,
        "consumed_on_resume": True,
    })
    (output / name).unlink(missing_ok=True)


def _denominator(value: float | torch.Tensor) -> float:
    result = float(value.detach().item()) if torch.is_tensor(value) else float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("Loss denominators must be finite and nonnegative.")
    return result


def _macro_update_slices(count: int, effective_cases: int) -> Iterable[slice]:
    for start in range(0, count, effective_cases):
        yield slice(start, min(start + effective_cases, count))


def _case_order_digest(case_keys: Sequence[Any]) -> str:
    encoded = json.dumps(list(case_keys), sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _optimizer_spec_payload(spec: OptimizerGroupSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "parameter_names": list(spec.parameter_names),
        "schedule": asdict(spec.schedule),
        "weight_decay": float(spec.weight_decay),
        "betas": list(spec.betas),
        "eps": float(spec.eps),
    }


class TrainingEngine:
    """Own the single training/evaluation loop used by all dataset adapters."""

    def __init__(
        self,
        config: EngineConfig,
        *,
        device: torch.device | str,
        selection: SelectionPolicy | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.selection = selection or SelectionPolicy()

    def _make_optimizer(
        self,
        model: nn.Module,
        provider: TaskProvider,
        arm: str,
        initial_epoch: int,
        optimizer_seed: Mapping[str, Any] | None,
        *,
        group_specs: Sequence[OptimizerGroupSpec] | None = None,
    ) -> tuple[torch.optim.Optimizer, tuple[OptimizerGroupSpec, ...]]:
        specs = tuple(group_specs or provider.optimizer_groups(
            model, arm, self.config.stage_for_epoch(initial_epoch)))
        if not specs or len({spec.name for spec in specs}) != len(specs):
            raise ValueError("Providers must declare at least one uniquely named optimizer group.")
        named = dict(model.named_parameters())
        used: set[str] = set()
        groups: list[dict[str, Any]] = []
        for spec in specs:
            if not spec.parameter_names:
                raise ValueError(f"Optimizer group {spec.name!r} has no parameters.")
            if not math.isfinite(spec.weight_decay) or spec.weight_decay < 0:
                raise ValueError(f"Optimizer group {spec.name!r} has invalid weight decay.")
            params = []
            for name in spec.parameter_names:
                if name not in named:
                    raise ValueError(f"Optimizer group {spec.name!r} references missing parameter {name!r}.")
                if name in used:
                    raise ValueError(f"Parameter {name!r} appears in more than one optimizer group.")
                if not named[name].requires_grad:
                    raise ValueError(f"Optimizer parameter {name!r} is unexpectedly frozen at engine creation.")
                params.append(named[name])
                used.add(name)
            groups.append({
                "params": params,
                "lr": spec.schedule.value(initial_epoch),
                "weight_decay": spec.weight_decay,
                "betas": spec.betas,
                "eps": spec.eps,
                "group_name": spec.name,
            })
        trainable = {name for name, parameter in named.items() if parameter.requires_grad}
        omitted = trainable - used
        if omitted:
            raise ValueError(f"Trainable model parameters lack optimizer groups: {sorted(omitted)}")
        optimizer = torch.optim.AdamW(groups)
        _load_named_optimizer_state(model, optimizer, optimizer_seed)
        return optimizer, specs

    def preflight_one_update(
        self,
        model: nn.Module,
        provider: TaskProvider,
        *,
        arm: str = "warmup",
        optimizer_seed: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Exercise one disposable warmup optimizer update without saving a run.

        The maintained CLI uses this for a CPU-only dry run. It deliberately
        delegates to the exact same batch, denominator, forward, loss,
        accumulation, clipping, and optimizer helpers as :meth:`fit`, while
        leaving no checkpoint or training history behind.
        """

        if arm != "warmup":
            raise ValueError("A disposable preflight update must use the shared warmup arm.")
        model.to(self.device)
        cases = tuple(provider.epoch_cases(1, self.config.seed))
        if not cases:
            raise ValueError("The task provider has no TRAIN case for a dry run.")
        cases = cases[:1]
        phase = self.config.stage_for_epoch(1)
        key = SamplingKey(self.config.seed, 1, 0, 0, phase, arm)
        batch = provider.make_batch(cases, key)
        if tuple(batch.case_keys) and list(batch.case_keys) != list(cases):
            raise ValueError("Provider changed case keys during the disposable preflight.")
        if not tuple(batch.case_keys):
            batch = TaskBatch(batch.scene_inputs, batch.receivers, batch.targets, batch.auxiliary, cases)
        on_phase_start = getattr(provider, "on_phase_start", None)
        stage_receipt = None
        if callable(on_phase_start):
            stage_receipt = on_phase_start(
                model=model, arm=arm, epoch=1, phase=phase,
                temperature=self.config.temperature_for_epoch(1))
        model.train(True)
        optimizer, specs = self._make_optimizer(
            model, provider, arm, 1, optimizer_seed)
        self._set_schedules(optimizer, specs, 1)
        losses, work, denominators, query_hash = self._run_update(
            model, provider, optimizer, (batch,), phase=phase, arm=arm,
            epoch=1, update_index=0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {
            "status": "disposable_cpu_preflight_update" if self.device.type == "cpu" else "disposable_preflight_update",
            "arm": arm,
            "phase": phase,
            "case_keys": list(cases),
            "optimizer_updates": 1,
            "losses": losses,
            "denominators": denominators,
            "work_counts": work,
            "query_sampling_sha256": query_hash,
            "stage_receipt": stage_receipt,
        }

    @staticmethod
    def _set_schedules(optimizer: torch.optim.Optimizer, specs: Sequence[OptimizerGroupSpec], epoch: int) -> None:
        by_name = {spec.name: spec for spec in specs}
        if {group["group_name"] for group in optimizer.param_groups} != set(by_name):
            raise ValueError("Optimizer groups changed after the absolute schedule was sealed.")
        for group in optimizer.param_groups:
            group["lr"] = by_name[group["group_name"]].schedule.value(epoch)

    def _run_update(
        self,
        model: nn.Module,
        provider: TaskProvider,
        optimizer: torch.optim.Optimizer,
        microbatches: Sequence[TaskBatch],
        *,
        phase: str,
        arm: str,
        epoch: int,
        update_index: int,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], str | None]:
        planned = {
            name: _denominator(value)
            for name, value in provider.loss_denominators(microbatches, phase, arm).items()
        }
        if any(not name for name in planned):
            raise ValueError("Loss term names must be nonempty.")
        if any(value <= 0 for value in planned.values()):
            raise ValueError("A declared optimization term must have a positive macro-update denominator.")
        observed: dict[str, float] = {name: 0.0 for name in planned}
        losses: dict[str, float] = {name: 0.0 for name in planned}
        work: dict[str, float] = {}
        sampling_records: list[str] = []
        optimizer.zero_grad(set_to_none=True)
        for batch in microbatches:
            batch_auxiliary = batch.auxiliary if isinstance(batch.auxiliary, Mapping) else {}
            sample_hash = batch_auxiliary.get("query_sampling_sha256")
            if sample_hash is not None:
                if not isinstance(sample_hash, str) or len(sample_hash) != 64:
                    raise ValueError("Provider query-sampling receipt must be a SHA256 hex string.")
                sampling_records.append(sample_hash)
            scene = provider.make_scene(batch.scene_inputs)
            predictions, auxiliary_state = provider.predict_native(
                model, scene, batch.receivers, arm, phase, epoch=epoch,
                temperature=self.config.temperature_for_epoch(epoch))
            terms = provider.loss_terms(predictions, batch.targets, phase, auxiliary_state)
            if not isinstance(terms, Mapping):
                raise TypeError("loss_terms must return a mapping of names to LossTerm values.")
            objective: torch.Tensor | None = None
            for name, term in terms.items():
                if name not in planned:
                    raise ValueError(f"Loss term {name!r} has no target-only macro denominator.")
                if not torch.is_tensor(term.numerator) or term.numerator.numel() != 1:
                    raise ValueError(f"Loss numerator {name!r} must be a scalar tensor.")
                denominator = _denominator(term.denominator)
                if not math.isfinite(float(term.weight)):
                    raise ValueError(f"Loss weight {name!r} is not finite.")
                if denominator == 0:
                    if float(term.numerator.detach()) != 0.0:
                        raise ValueError(f"Loss numerator {name!r} is nonzero with an empty valid mask.")
                    continue
                observed[name] += denominator
                piece = term.numerator * (float(term.weight) / planned[name])
                objective = piece if objective is None else objective + piece
                losses[name] += float(term.numerator.detach()) / planned[name] * float(term.weight)
            for name, value in provider.work_counts(batch, predictions, auxiliary_state).items():
                work[name] = work.get(name, 0.0) + float(value)
            if objective is None:
                raise ValueError("A microbatch produced no differentiable loss terms.")
            if not bool(torch.isfinite(objective.detach())):
                raise FloatingPointError("Nonfinite optimizer objective.")
            objective.backward()
        for name, expected in planned.items():
            actual = observed.get(name, 0.0)
            if not math.isclose(actual, expected, rel_tol=1.0e-6, abs_tol=1.0e-7):
                raise ValueError(f"Actual denominator for {name!r} ({actual}) differs from prepass ({expected}).")
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]
                      if parameter.grad is not None]
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip)
        if not torch.isfinite(norm) or norm <= 0:
            raise FloatingPointError("Nonfinite or zero actual parameter gradient.")
        optimizer.step()
        work["gradient_norm"] = float(norm.detach())
        sampling_hash = hashlib.sha256(
            json.dumps(sampling_records, separators=(",", ":")).encode("utf-8")
        ).hexdigest() if sampling_records else None
        return losses, work, {name: observed[name] for name in observed}, sampling_hash

    @torch.no_grad()
    def _evaluate(
        self, model: nn.Module, provider: TaskProvider, arm: str, phase: str, epoch: int,
    ) -> dict[str, Any]:
        del phase
        previous_mode = model.training
        rng_state = capture_rng_state()
        records: list[Mapping[str, Any]] = []
        loss_totals: dict[str, dict[str, float]] = {}
        loss_batch_counts: dict[str, int] = {}
        batch_count = 0
        model.eval()
        try:
            for batch in provider.validation_batches():
                batch_count += 1
                scene = provider.make_scene(batch.scene_inputs)
                # The candidate's deployed hard route is always measured here; a
                # warmup/open or soft training phase must not leak into validation.
                predictions, auxiliary_state = provider.predict_native(
                    model, scene, batch.receivers, arm, "hard", epoch=epoch,
                    temperature=self.config.temperature_for_epoch(epoch))
                records.append(provider.validation_metrics(predictions, batch.targets, auxiliary_state))
                validation_loss_fn = getattr(provider, "validation_loss_terms", None)
                if callable(validation_loss_fn):
                    state_reader = getattr(provider, "training_state_dict", None)
                    state_restorer = getattr(provider, "load_training_state_dict", None)
                    provider_state = copy.deepcopy(state_reader()) if callable(state_reader) else None
                    try:
                        terms = validation_loss_fn(
                            predictions, batch.targets, auxiliary_state, batch=batch, arm=arm)
                    finally:
                        if provider_state is not None and callable(state_restorer):
                            state_restorer(provider_state)
                    if not isinstance(terms, Mapping):
                        raise TypeError("validation_loss_terms must return a mapping of names to LossTerm values.")
                    for name, term in terms.items():
                        if not isinstance(name, str) or not name:
                            raise ValueError("Validation loss term names must be nonempty strings.")
                        if not isinstance(term, LossTerm) or not torch.is_tensor(term.numerator) or term.numerator.numel() != 1:
                            raise ValueError(f"Validation loss {name!r} must be a scalar LossTerm.")
                        weight = float(term.weight)
                        denominator = _denominator(term.denominator)
                        numerator = float(term.numerator.detach())
                        if not math.isfinite(weight) or not math.isfinite(numerator):
                            raise FloatingPointError(f"Validation loss {name!r} is nonfinite.")
                        if denominator == 0:
                            if numerator != 0.0:
                                raise ValueError(f"Validation loss {name!r} has a nonzero numerator with no valid elements.")
                            continue
                        aggregate = loss_totals.setdefault(
                            name, {"numerator": 0.0, "denominator": 0.0, "weight": weight})
                        if not math.isclose(aggregate["weight"], weight, rel_tol=0.0, abs_tol=1.0e-12):
                            raise ValueError(f"Validation objective weight for {name!r} changed across batches.")
                        aggregate["numerator"] += numerator
                        aggregate["denominator"] += denominator
                        loss_batch_counts[name] = loss_batch_counts.get(name, 0) + 1
            if batch_count == 0:
                raise ValueError("The task provider returned no exposed validation batches.")
            if any(count != batch_count for count in loss_batch_counts.values()):
                raise ValueError("Each validation objective term must be present in every validation batch.")
            result = dict(provider.reduce_native_metrics(records))
            extra_metrics = getattr(provider, "extra_validation_metrics", None)
            if callable(extra_metrics):
                result.update(extra_metrics(model, arm, "hard"))
            if loss_totals:
                result["validation_loss_terms"] = {
                    name: (values["numerator"] / values["denominator"]) * values["weight"]
                    for name, values in loss_totals.items()
                }
                result["validation_loss_term_weights"] = {
                    name: values["weight"] for name, values in loss_totals.items()
                }
                result["validation_loss_aggregation"] = (
                    "sum of numerators divided by sum of valid elements across exposed validation batches, "
                    "then multiplied by the provider objective weight"
                )
        finally:
            model.train(previous_mode)
            restore_rng_state(rng_state)
        result["validation_execution_mode"] = arm
        result["validation_route_phase"] = "hard"
        return result

    def _checkpoint_payload(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        identity: Mapping[str, Any],
        arm: str,
        epoch: int,
        history: Sequence[Mapping[str, Any]],
        best_field: float,
        best_guarded: float,
        sampler_state: Mapping[str, Any],
        group_specs: Sequence[OptimizerGroupSpec],
        provider_training_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "checkpoint_schema_version": 1,
            "workflow": "unified_interaction_refinement",
            "experiment_identity": _cpu_tree(identity),
            "arm": arm,
            "epoch": int(epoch),
            "current_epoch": int(epoch),
            "model_state_dict": _cpu_tree(model.state_dict()),
            "optimizer_state_by_name": named_optimizer_state(model, optimizer),
            "rng_state": _cpu_tree(capture_rng_state()),
            "sampler_state": _cpu_tree(sampler_state),
            "history": _cpu_tree(list(history)),
            "best_field_score": float(best_field),
            "best_response_guarded_score": float(best_guarded),
            "optimizer_group_names": [group["group_name"] for group in optimizer.param_groups],
            "optimizer_schedule_contract": [_optimizer_spec_payload(spec) for spec in group_specs],
            "provider_training_state": _cpu_tree(provider_training_state or {}),
        }

    @_exclusive_training_run
    def fit(
        self,
        model: nn.Module,
        provider: TaskProvider,
        output_dir: str | Path,
        *,
        identity: Mapping[str, Any],
        arm: str,
        stop_after: int,
        optimizer_seed: Mapping[str, Any] | None = None,
        resume_checkpoint: str | Path | None = None,
        branch_from_checkpoint: str | Path | None = None,
        clean_stop_name: str = "CLEAN_STOP_REQUEST.json",
        allow_microbatch_change: bool = False,
    ) -> dict[str, Any]:
        """Fit one warmup or arm segment and save exact epoch-boundary state."""

        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        layout = RunLayout(output)
        layout.ensure()
        if resume_checkpoint is None:
            markers = (
                "latest_model.pt",
                "last.pt",
                "history.json",
                "fit_summary.json",
                "active_process.json",
                "progress.json",
            )
            has_epoch_checkpoints = any((output / "checkpoints").glob("epoch_*_model.pt")) or any(
                output.glob("epoch_*_model.pt")
            )
            if any(layout.read_path(name).exists() for name in markers) or has_epoch_checkpoints:
                raise ValueError("A new fit requires a fresh run directory; use exact resume for existing history.")
        else:
            requested_latest = resolve_checkpoint(output, resume_checkpoint)
            current_latest = resolve_checkpoint(output, "latest")
            if requested_latest != current_latest:
                raise ValueError("Exact resume requires this run's current latest checkpoint; historical checkpoints cannot rewind history.")
        identity = dict(identity)
        provider_identity = dict(provider.identity_payload())
        if "provider_identity" in identity and identity["provider_identity"] != provider_identity:
            raise ValueError("Sealed experiment provider identity differs from the active task adapter.")
        if "arm" in identity:
            raise ValueError("Matched execution arm is checkpoint metadata, not part of shared identity.")
        if resume_checkpoint is not None and branch_from_checkpoint is not None:
            raise ValueError("A run cannot be both an exact resume and a new branch.")
        if allow_microbatch_change and resume_checkpoint is None:
            raise ValueError("A microbatch amendment requires an existing run's latest checkpoint.")
        source_path = (
            resolve_checkpoint(output, resume_checkpoint)
            if resume_checkpoint is not None
            else resolve_checkpoint(output, branch_from_checkpoint)
            if branch_from_checkpoint is not None
            else None
        )
        source_payload: Mapping[str, Any] | None = None
        restored_rng: Mapping[str, Any] | None = None
        if source_path is not None:
            from honf_runtime.compat import load_trusted_checkpoint

            source_payload = load_trusted_checkpoint(source_path, map_location="cpu")
        start_epoch = 1 if source_payload is None else int(source_payload["epoch"]) + 1
        if resume_checkpoint is not None and int(source_payload["epoch"]) >= stop_after:
            raise ValueError("Exact resume stop must advance the saved epoch.")
        if not start_epoch <= stop_after <= self.config.total_epochs:
            raise ValueError("Requested stop must advance the saved age and remain inside the common horizon.")
        if not identity:
            raise ValueError("A sealed nonempty experiment identity is required.")
        model.to(self.device)
        # The warmup stage declares the stable parameter groups for the whole
        # run, including router parameters while warmup executes all fine
        # reads. All-fine execution does not imply frozen router gradients.
        group_specs = tuple(provider.optimizer_groups(model, arm, "warmup"))
        sealed_identity = {
            **dict(identity),
            "provider_identity": provider_identity,
            "engine_config": _engine_config_payload(self.config),
            "selection_policy": asdict(self.selection),
            "optimizer_schedule_contract": [_optimizer_spec_payload(spec) for spec in group_specs],
        }
        amendment = None
        if source_payload is not None:
            amendment = _resume_identity_amendment(source_payload.get("experiment_identity", {}), sealed_identity,
                allow_microbatch_change=allow_microbatch_change)
            if resume_checkpoint is not None:
                if source_payload.get("arm") != arm:
                    raise ValueError("Exact resume arm differs from the saved checkpoint.")
            else:
                if source_payload.get("arm") != "warmup":
                    raise ValueError("Matched arms may branch only from the shared warmup checkpoint.")
                if int(source_payload.get("epoch", -1)) != self.config.warmup_epochs:
                    raise ValueError("Matched arms must branch at the declared common warmup boundary.")
            model.load_state_dict(source_payload["model_state_dict"], strict=True)
            optimizer_seed = source_payload["optimizer_state_by_name"]
            restored_rng = source_payload["rng_state"]
            restore_provider_state = getattr(provider, "load_training_state_dict", None)
            if callable(restore_provider_state):
                restore_provider_state(source_payload.get("provider_training_state", {}))
        if resume_checkpoint is not None:
            _consume_acknowledged_clean_stop(output, clean_stop_name, layout)
        optimizer, optimizer_specs = self._make_optimizer(
            model, provider, arm, start_epoch, optimizer_seed, group_specs=group_specs)
        if resume_checkpoint is not None:
            saved_group_names = source_payload.get("optimizer_group_names")
            current_group_names = [group["group_name"] for group in optimizer.param_groups]
            saved_groups = source_payload["optimizer_state_by_name"].get("groups", [])
            current_groups = named_optimizer_state(model, optimizer).get("groups", [])
            if (saved_group_names != current_group_names or saved_groups != current_groups
                    or source_payload.get("optimizer_schedule_contract")
                    != [_optimizer_spec_payload(spec) for spec in optimizer_specs]):
                raise ValueError("Exact resume optimizer group membership/order differs from its checkpoint.")
        if restored_rng is not None:
            restore_rng_state(restored_rng)
        history = list(source_payload.get("history", [])) if source_payload is not None else []
        resume_amendments = list(source_payload.get("resume_amendments", [])) if source_payload is not None else []
        if amendment is not None:
            amendment = {**amendment, "source_epoch": int(source_payload["epoch"]), "next_epoch": start_epoch,
                "source_checkpoint_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "preserved_case_visits": sum(int(row["case_visits"]) for row in history),
                "preserved_optimizer_updates": sum(int(row["optimizer_updates"]) for row in history),
                "recorded_unix": time.time()}
            resume_amendments.append(amendment)
            _atomic_json(layout.write_path(f"microbatch_amendment_epoch_{start_epoch:04d}.json"), amendment)
        best_field = float(source_payload.get("best_field_score", float("inf"))) if source_payload else float("inf")
        best_guarded = float(source_payload.get("best_response_guarded_score", float("inf"))) if source_payload else float("inf")
        branch_parent = None
        if branch_from_checkpoint is not None:
            branch_parent = {
                "path": str(source_path),
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "identity": source_payload.get("experiment_identity"),
                "epoch": int(source_payload["epoch"]),
            }
            history = []
            best_field = best_guarded = float("inf")
        active = {
            "status": "running", "arm": arm, "identity": sealed_identity, "device": str(self.device),
            "pid": os.getpid(),
            "process_start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19],
            "start_epoch": start_epoch, "requested_stop": int(stop_after),
            "started_unix": float(os.getenv("PROCESS_STARTED_UNIX", time.time())),
            "engine_started_unix": time.time(),
            "branch_parent": branch_parent,
        }
        if resume_amendments:
            active["resume_amendments"] = resume_amendments
        if resume_checkpoint is None and layout.canonical_writes:
            identity_path = layout.path("experiment_identity.json")
            marker_path = layout.path("artifact_layout.json")
            if not marker_path.exists():
                _atomic_json(marker_path, {
                    "artifact_layout_version": 1,
                    "run_dir": str(output),
                })
            if not identity_path.exists():
                _atomic_json(identity_path, sealed_identity)
            software_path = layout.path("software.json")
            if not software_path.exists():
                _atomic_json(software_path, environment_snapshot())
            source_state_path = layout.path("source_state.json")
            if not source_state_path.exists():
                _atomic_json(source_state_path, source_state_snapshot(PROJECT_ROOT))
        _atomic_json(layout.write_path("active_process.json"), active)
        if resume_checkpoint is not None:
            restored = {"event": "checkpoint_restored", "source_epoch": start_epoch - 1,
                "next_epoch": start_epoch, "microbatch_cases": self.config.microbatch_cases,
                "effective_cases": self.config.effective_cases,
                "preserved_case_visits": sum(int(row["case_visits"]) for row in history),
                "preserved_optimizer_updates": sum(int(row["optimizer_updates"]) for row in history)}
            print(json.dumps(restored, sort_keys=True), flush=True)
            if bool(getattr(sys.stderr, "isatty", lambda: False)()):
                print(f"Restored epoch {start_epoch - 1}; next {start_epoch}/{stop_after}; "
                      f"microbatch {self.config.microbatch_cases}, effective {self.config.effective_cases}; "
                      f"retained {restored['preserved_case_visits']} TRAIN visits, "
                      f"{restored['preserved_optimizer_updates']} optimizer updates", file=sys.stderr, flush=True)
        total_started = time.perf_counter()
        stopped_at = start_epoch - 1
        latest_epoch = int(source_payload["epoch"]) if resume_checkpoint is not None else 0
        completed_visits = sum(int(row["case_visits"]) for row in history)
        completed_updates = sum(int(row["optimizer_updates"]) for row in history)
        clean_stopped = False
        previous_phase = self.config.stage_for_epoch(start_epoch - 1) if start_epoch > 1 else None
        try:
            for epoch in range(start_epoch, int(stop_after) + 1):
                # Callbacks (preparation, validation, or a prior calibration hook)
                # may leave the module in evaluation mode. The engine owns the
                # mode for every optimization epoch.
                model.train(True)
                phase = self.config.stage_for_epoch(epoch)
                stage_receipt = None
                if phase != previous_phase:
                    on_phase_start = getattr(provider, "on_phase_start", None)
                    if callable(on_phase_start):
                        stage_receipt = on_phase_start(
                            model=model, arm=arm, epoch=epoch, phase=phase,
                            temperature=self.config.temperature_for_epoch(epoch))
                    if stage_receipt is not None:
                        _atomic_json(layout.write_path(f"stage_receipt_epoch_{epoch:04d}_{phase}.json"), stage_receipt)
                # A stage callback may run calibration/evaluation in no-grad mode.
                # Reassert training mode after it and before any optimizer work.
                model.train(True)
                previous_phase = phase
                self._set_schedules(optimizer, group_specs, epoch)
                epoch_started = time.perf_counter()
                ordered_cases = list(provider.epoch_cases(epoch, self.config.seed))
                if not ordered_cases:
                    raise ValueError("Each development epoch must visit at least one selected TRAIN case.")
                if self.config.deterministic_case_order:
                    rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, epoch, 0x484F4E46]))
                    order = rng.permutation(len(ordered_cases)).tolist()
                    ordered_cases = [ordered_cases[index] for index in order]
                order_hash = _case_order_digest(ordered_cases)
                update_rows: list[dict[str, Any]] = []
                update_count = math.ceil(len(ordered_cases) / self.config.effective_cases)
                progress = tqdm(
                    total=len(ordered_cases), desc=f"epoch {epoch}/{stop_after}", unit="case",
                    dynamic_ncols=True, file=sys.stderr,
                    disable=not bool(getattr(sys.stderr, "isatty", lambda: False)()), leave=False,
                )
                try:
                    for update_index, group_slice in enumerate(
                        _macro_update_slices(len(ordered_cases), self.config.effective_cases)
                    ):
                        update_cases = ordered_cases[group_slice]
                        microbatches = []
                        for micro_index, start in enumerate(range(0, len(update_cases), self.config.microbatch_cases)):
                            case_chunk = update_cases[start:start + self.config.microbatch_cases]
                            key = SamplingKey(self.config.seed, epoch, update_index, micro_index, phase, arm)
                            batch = provider.make_batch(case_chunk, key)
                            if tuple(batch.case_keys) and list(batch.case_keys) != list(case_chunk):
                                raise ValueError("Provider changed the ordered case keys inside a microbatch.")
                            if not tuple(batch.case_keys):
                                batch = TaskBatch(batch.scene_inputs, batch.receivers, batch.targets, batch.auxiliary,
                                                  tuple(case_chunk))
                            microbatches.append(batch)
                        losses, work, _denominators, sampling_hash = self._run_update(model, provider, optimizer, microbatches,
                            phase=phase, arm=arm, epoch=epoch, update_index=update_index)
                        update_rows.append({"losses": losses, "work": work, "case_count": len(update_cases),
                                            "microbatch_count": len(microbatches),
                                            "query_sampling_sha256": sampling_hash})
                        loss_value = sum(float(value) for value in losses.values())
                        lr_values = ",".join(
                            f"{group['group_name']}:{float(group['lr']):.1e}" for group in optimizer.param_groups
                        )
                        progress.set_postfix({
                            "updates": f"{len(update_rows)}/{update_count}", "loss": f"{loss_value:.4g}",
                            "lr": lr_values, "phase": phase,
                        }, refresh=False)
                        progress.update(len(update_cases))
                        _atomic_json(layout.write_path("progress.json"), {
                            "status": "training", "epoch_in_progress": epoch, "completed_epoch": epoch - 1,
                            "phase": phase, "updates_this_epoch": len(update_rows),
                            "case_visits_this_epoch": sum(item["case_count"] for item in update_rows),
                            "completed_case_visits": completed_visits,
                            "completed_optimizer_updates": completed_updates,
                            "latest_checkpoint_epoch": latest_epoch, "updated_unix": time.time(),
                        })
                finally:
                    progress.close()
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                train_seconds = time.perf_counter() - epoch_started
                row: dict[str, Any] = {
                    "epoch": epoch,
                    "phase": phase,
                    "temperature": self.config.temperature_for_epoch(epoch),
                    "arm": arm,
                    "case_visits": len(ordered_cases),
                    "optimizer_updates": len(update_rows),
                    "microbatches": sum(item["microbatch_count"] for item in update_rows),
                    "partial_update_cases": len(ordered_cases) % self.config.effective_cases,
                    "learning_rates": {group["group_name"]: float(group["lr"]) for group in optimizer.param_groups},
                    "train_seconds": train_seconds,
                    "train_losses": _mean_mapping([item["losses"] for item in update_rows]),
                    "work_counts": _sum_mapping([item["work"] for item in update_rows]),
                    "case_order_sha256": order_hash,
                }
                sample_hashes = [item["query_sampling_sha256"] for item in update_rows
                                 if item["query_sampling_sha256"] is not None]
                if sample_hashes:
                    row["query_sampling_sha256"] = hashlib.sha256(
                        json.dumps(sample_hashes, separators=(",", ":")).encode("utf-8")
                    ).hexdigest()
                if stage_receipt is not None:
                    row["stage_receipt"] = stage_receipt
                clean_stop_request = _read_clean_stop_request(output, clean_stop_name)
                clean_stop_requested = clean_stop_request is not None
                review = (epoch % self.config.monitor_every == 0 or epoch in self.config.monitor_epochs
                          or epoch == stop_after or clean_stop_requested)
                improved_field = improved_guarded = False
                if review:
                    validation_started = time.perf_counter()
                    metrics = self._evaluate(model, provider, arm, phase, epoch)
                    validation_terms = dict(metrics.pop("validation_loss_terms", {}))
                    validation_weights = dict(metrics.pop("validation_loss_term_weights", {}))
                    aggregation = metrics.pop("validation_loss_aggregation", None)
                    train_terms = dict(row["train_losses"])
                    matched_terms = sorted(set(train_terms) & set(validation_terms))
                    training_only_terms = sorted(set(train_terms) - set(validation_terms))
                    validation_only_terms = sorted(set(validation_terms) - set(train_terms))
                    training_reduction = (
                        "arithmetic mean of macro-update means; each optimizer update has equal weight, "
                        "including any partial final update"
                    )
                    validation_reduction = aggregation or "No held-out objective losses were recorded by this provider."
                    objective = {
                        "scope": "exposed VALIDATION panel evaluated with deployed hard inference",
                        "training_phase": phase,
                        "validation_route_phase": "hard",
                        "training_reduction": training_reduction,
                        "validation_reduction": validation_reduction,
                        "training_prediction_mode": (
                            "phase-specific optimization forward; provider loss terms may blend main and fine paths"
                        ),
                        "validation_prediction_mode": (
                            "deployed hard inference; validation terms use the provider-defined hard-route outputs"
                        ),
                        "directly_comparable": False,
                        "comparability_note": (
                            "Shared term names do not imply identical predictions or reductions: TRAIN losses are "
                            "averaged per optimizer update over changing model states; VALIDATION losses pool "
                            "valid elements at the review epoch using hard inference."
                        ),
                        "terms": validation_terms,
                        "term_weights": validation_weights,
                        "matched_terms": matched_terms,
                        "training_only_terms": training_only_terms,
                        "validation_only_terms": validation_only_terms,
                        "training_matched_total": (sum(train_terms[name] for name in matched_terms)
                                                   if matched_terms else None),
                        "validation_matched_total": (sum(validation_terms[name] for name in matched_terms)
                                                     if matched_terms else None),
                        "is_full_training_objective": not training_only_terms and not validation_only_terms,
                        "is_full_training_objective_note": (
                            "This flag reports term-name coverage only; directly_comparable separately describes "
                            "whether values use the same prediction path and reduction."
                        ),
                        "aggregation": aggregation,
                        "definition": (
                            "Shared-name subtotal includes only terms measured on both sides, using the provider's "
                            "objective weights. It is not a directly comparable objective: reductions differ and "
                            "provider prediction paths may differ. Hard validation omits full-detail replay and any "
                            "train-only PDE or auxiliary terms."
                        ),
                    }
                    row["validation_objective"] = objective
                    metrics["loss_objective"] = objective
                    row["validation"] = metrics
                    row["validation_seconds"] = time.perf_counter() - validation_started
                    field_score = float(metrics[self.selection.field_metric])
                    if not math.isfinite(field_score):
                        raise FloatingPointError("Nonfinite declared field selector.")
                    improved_field = field_score < best_field
                    if improved_field:
                        best_field = field_score
                    guard_key = self.selection.response_guard_metric
                    guard_pass = guard_key is None or float(metrics[guard_key]) <= self.selection.maximum_response_ratio
                    improved_guarded = guard_key is not None and guard_pass and field_score < best_guarded
                    if improved_guarded:
                        best_guarded = field_score
                history.append(row)
                print(json.dumps(row, sort_keys=True, default=str, allow_nan=False), flush=True)
                milestone = ((review if self.config.checkpoint_epochs is None
                              else epoch in self.config.checkpoint_epochs)
                             or epoch == stop_after or improved_guarded)
                curve_due = self.config.curve_every is not None and (
                    epoch == 1 or epoch % self.config.curve_every == 0 or epoch == stop_after or clean_stop_requested)
                should_save = (review or milestone or curve_due
                               or (self.config.latest_every is not None and epoch % self.config.latest_every == 0))
                if should_save:
                    sampler_state = {
                        "seed": self.config.seed,
                        "completed_epoch": epoch,
                        "next_epoch": epoch + 1,
                        "next_case_cursor": 0,
                        "last_case_order_sha256": order_hash,
                        "case_visits_this_epoch": len(ordered_cases),
                        "optimizer_updates_this_epoch": len(update_rows),
                    }
                    payload = self._checkpoint_payload(model, optimizer, identity=sealed_identity, arm=arm, epoch=epoch,
                        history=history, best_field=best_field, best_guarded=best_guarded, sampler_state=sampler_state,
                        group_specs=optimizer_specs,
                        provider_training_state=getattr(provider, "training_state_dict", dict)())
                    if resume_amendments:
                        payload["resume_amendments"] = resume_amendments
                    _atomic_json(layout.write_path("history.json"), history)
                    if milestone:
                        _atomic_torch_save(layout.write_path(f"epoch_{epoch:04d}_model.pt"), payload)
                    if review:
                        _atomic_json(layout.write_path(f"validation_epoch_{epoch:04d}.json"), row["validation"])
                        if improved_field:
                            _atomic_torch_save(layout.write_path("best_by_field_mse_model.pt"), payload)
                        if improved_guarded:
                            selected_checkpoint = layout.write_path(f"epoch_{epoch:04d}_model.pt")
                            _atomic_json(layout.write_path("best_by_response_guarded_selection.json"), {
                                "selector": "response_guarded_field_score",
                                "field_metric": self.selection.field_metric,
                                "field_score": field_score,
                                "guard_metric": guard_key,
                                "guard_value": float(metrics[guard_key]),
                                "maximum_guard_ratio": self.selection.maximum_response_ratio,
                                "epoch": int(epoch),
                                "checkpoint": str(selected_checkpoint.relative_to(output)),
                                "checkpoint_sha256": hashlib.sha256(selected_checkpoint.read_bytes()).hexdigest(),
                            })
                    # Advance the resume cursor only after required milestone
                    # and selector artifacts exist, including the terminal age.
                    _atomic_torch_save(layout.write_path("latest_model.pt"), payload)
                    latest_epoch = epoch
                if curve_due:
                    _atomic_json(layout.write_path("history.json"), history)
                    curve_rng = capture_rng_state()
                    try:
                        _render_loss_curves(
                            history,
                            layout.write_path("loss_curves.png").parent,
                            self.selection.field_metric,
                        )
                    finally:
                        restore_rng_state(curve_rng)
                completed_visits += len(ordered_cases)
                completed_updates += len(update_rows)
                _atomic_json(layout.write_path("progress.json"), {
                    "status": "epoch_completed", "completed_epoch": epoch, "phase": phase,
                    "completed_case_visits": completed_visits, "completed_optimizer_updates": completed_updates,
                    "latest_checkpoint_epoch": latest_epoch, "updated_unix": time.time(),
                })
                stopped_at = epoch
                if clean_stop_requested:
                    acknowledgement = {
                        "request_id": clean_stop_request["request_id"],
                        "requested_unix": clean_stop_request.get("requested_unix"),
                        "requested_by": clean_stop_request.get("requested_by"),
                        "acknowledged_unix": time.time(),
                        "epoch": epoch,
                        "phase": phase,
                    }
                    _atomic_json(layout.write_path("clean_stop_acknowledged.json"), acknowledgement)
                    request_tag = hashlib.sha256(clean_stop_request["request_id"].encode("utf-8")).hexdigest()[:16]
                    _atomic_json(layout.write_path(f"clean_stop_consumed_{request_tag}.json"), {
                        "request": clean_stop_request,
                        "acknowledgement": acknowledgement,
                        "consumed_on_resume": False,
                    })
                    (output / clean_stop_name).unlink(missing_ok=True)
                    clean_stopped = True
                    break
            elapsed = time.perf_counter() - total_started
            active.update(status="clean_stopped" if clean_stopped else "completed",
                          completed_epoch=stopped_at, ended_unix=time.time(), engine_seconds=elapsed,
                          process_seconds=max(0.0, time.time() - float(active["started_unix"])))
            _atomic_json(layout.write_path("active_process.json"), active)
            guarded_summary = best_guarded if math.isfinite(best_guarded) else None
            _atomic_json(layout.write_path("fit_summary.json"), {
                **active,
                "case_visits": sum(int(item["case_visits"]) for item in history if int(item["epoch"]) >= start_epoch),
                "optimizer_updates": sum(int(item["optimizer_updates"]) for item in history if int(item["epoch"]) >= start_epoch),
                "cumulative_case_visits": completed_visits,
                "cumulative_optimizer_updates": completed_updates,
                "cumulative_microbatches": sum(int(item["microbatches"]) for item in history),
                "cumulative_partial_update_cases": sum(int(item["partial_update_cases"]) for item in history),
                "best_field_score": best_field,
                "best_response_guarded_score": guarded_summary,
                "branch_parent": branch_parent,
            })
            return {**active, "completed_epoch": stopped_at, "best_field_score": best_field,
                    "best_response_guarded_score": guarded_summary, "branch_parent": branch_parent}
        except BaseException as exc:
            active.update(status="failed", completed_epoch=stopped_at, ended_unix=time.time(),
                          engine_seconds=time.perf_counter() - total_started,
                          process_seconds=max(0.0, time.time() - float(active["started_unix"])),
                          exception_type=type(exc).__name__, exception_message=str(exc))
            try:
                _atomic_json(layout.write_path("active_process.json"), active)
            except OSError:
                pass  # Preserve the original failure when its receipt cannot be written.
            raise


def _mean_mapping(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([float(row[key]) for row in rows if key in row])) for key in keys}


def _sum_mapping(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            result[key] = result.get(key, 0.0) + float(value)
    return result

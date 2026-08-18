"""Readable staged trainer for the hierarchical inverse generator.

Request/context `R,c` condition plan flow `G` and layout flow `D`. Stage four
may call a case-owned frozen-HONF hook on a configured subset to compare
realized `G_hat`; flow matching remains the dominant objective. No stage runs
iterative design optimization.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Any, Callable, Mapping

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from honf_inverse_core.models.hierarchical_inverse import HierarchicalInverseDesigner
from honf_inverse_core.models.matching import token_assignment
from honf_inverse_core.models.rectified_flow import flow_interpolation
from honf_inverse_core.models.request_encoder import RequestEncoding

from .checkpointing import load_inverse_checkpoint, save_inverse_checkpoint
from .losses import layout_training_losses, plan_training_losses
from .stages import configure_stage, generated_plan_probability


JointLossHook = Callable[
    [HierarchicalInverseDesigner, Mapping[str, Any], Any, torch.Tensor, torch.Tensor],
    Mapping[str, torch.Tensor],
]


def recursive_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {name: recursive_to_device(item, device) for name, item in value.items()}
    return value


def recursive_batch_slice(value: Any, count: int) -> Any:
    if torch.is_tensor(value):
        return value[:count]
    if isinstance(value, Mapping):
        return {name: recursive_batch_slice(item, count) for name, item in value.items()}
    return value


def slice_encoding(encoding: RequestEncoding, count: int) -> RequestEncoding:
    return RequestEncoding(
        encoding.global_embedding[:count],
        encoding.token_embeddings[:count],
        encoding.token_mask[:count],
    )


@dataclass
class StageResult:
    stage: str
    best_validation_loss: float
    best_epoch: int
    epochs: int
    global_step: int


class InverseTrainer:
    """Train four explicit stages and preserve familiar checkpoint aliases."""

    ALIASES = {
        "stage_plan": "best_plan_model.pt",
        "stage_layout_teacher_plan": "best_layout_model.pt",
        "stage_layout_mixed_plan": "best_unguided_model.pt",
        "stage_joint_consistency": "best_corrected_model.pt",
    }

    def __init__(
        self,
        designer: HierarchicalInverseDesigner,
        *,
        device: str | torch.device,
        run_dir: str | Path,
        checkpoint_provenance: Mapping[str, Any],
        joint_loss_hook: JointLossHook | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.designer = designer.to(self.device)
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        self.provenance = dict(checkpoint_provenance)
        self.joint_loss_hook = joint_loss_hook
        self.global_step = 0
        self.metrics_path = self.run_dir / "metrics.csv"
        self.loss_curve_path = self.run_dir / "loss_curve.png"
        self.status_path = self.run_dir / "training_status.json"

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write_status(self, payload: Mapping[str, Any]) -> None:
        """Atomically publish the latest human- and machine-readable state."""

        destination = self.status_path
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        value = {"updated_at": self._utc_now(), **dict(payload)}
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _read_metric_rows(self) -> list[dict[str, str]]:
        if not self.metrics_path.exists():
            return []
        with self.metrics_path.open("r", newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def _save_loss_curve(self) -> None:
        """Atomically refresh a compact cross-stage loss plot from metrics.csv."""

        rows = self._read_metric_rows()
        if not rows:
            return
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-honf-inverse")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
        panels = (
            (axes[0], "Total loss", "train_total", "validation_total"),
            (axes[1], "Flow loss", "train_flow", "validation_flow"),
        )
        stages = list(dict.fromkeys(row.get("stage", "unknown") for row in rows))
        for axis, title, train_key, validation_key in panels:
            positive = True
            for stage in stages:
                selected = [row for row in rows if row.get("stage", "unknown") == stage]
                x_values: list[float] = []
                train_values: list[float] = []
                validation_values: list[float] = []
                for row in selected:
                    try:
                        x_value = float(row["global_step"])
                        train_value = float(row[train_key])
                        validation_value = float(row[validation_key])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not all(math.isfinite(value) for value in (x_value, train_value, validation_value)):
                        continue
                    x_values.append(x_value)
                    train_values.append(train_value)
                    validation_values.append(validation_value)
                    positive = positive and train_value > 0.0 and validation_value > 0.0
                if x_values:
                    short = stage.removeprefix("stage_")
                    axis.plot(x_values, train_values, label=f"{short} train")
                    axis.plot(x_values, validation_values, linestyle="--", label=f"{short} validation")
            axis.set_title(title)
            axis.set_xlabel("global optimizer step")
            axis.set_ylabel("loss")
            if positive:
                axis.set_yscale("log")
            axis.grid(True, alpha=0.25)
            if axis.lines:
                axis.legend(fontsize=7)
        figure.suptitle("Hierarchical inverse training (updated after every epoch)")
        temporary = self.loss_curve_path.with_name(f".{self.loss_curve_path.name}.tmp-{os.getpid()}")
        try:
            figure.savefig(temporary, format="png", dpi=150)
            os.replace(temporary, self.loss_curve_path)
        finally:
            plt.close(figure)
            if temporary.exists():
                temporary.unlink()

    def _encoding(self, batch: Mapping[str, Any]):
        return self.designer.encode_request(
            batch["request"],
            batch["context"],
            batch["geometry_constraints"],
            batch["geometry_constraint_mask"],
        )

    def _plan_loss(self, batch: Mapping[str, Any], encoding: Any) -> dict[str, torch.Tensor]:
        target_plan = batch["plan"].float()
        target = self.designer.plan_flow.continuous_target(target_plan)
        if self.designer.plan_flow.plan_token_mode == "exchangeable_set":
            noise = torch.randn_like(target)
            assignment = token_assignment(
                noise,
                target,
                method=str(self.designer.model_config["matching_mode"]),
            )
            target = assignment @ target
            activity_target = (assignment @ target_plan[..., 0:1]).squeeze(-1)
            state, velocity_target, time, _ = flow_interpolation(target, noise=noise)
        else:
            activity_target = target_plan[..., 0]
            state, velocity_target, time, _ = flow_interpolation(target)
        output = self.designer.plan_flow(state, time, encoding)
        endpoint = state + (1.0 - time[:, None, None]) * output.velocity
        return plan_training_losses(
            predicted_velocity=output.velocity,
            target_velocity=velocity_target,
            activity_logits=output.activity_logits,
            activity_target=activity_target,
            endpoint_estimate=endpoint,
        )

    def _layout_loss(
        self,
        batch: Mapping[str, Any],
        encoding: Any,
        plan_condition: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        target = batch["layout"].float()
        state, velocity_target, time, _ = flow_interpolation(target)
        output = self.designer.layout_flow(state, time, plan_condition, encoding)
        endpoint = state + (1.0 - time[:, None, None]) * output.velocity
        losses = layout_training_losses(
            predicted_velocity=output.velocity,
            target_velocity=velocity_target,
            presence_logits=output.presence_logits,
            presence_target=batch["module_present"],
            count_logits=output.count_logits,
            count_target=batch["module_count"],
            endpoint_estimate=endpoint,
            layout_target=target,
            geometry_constraints=batch["geometry_constraints"],
            compact_plan=plan_condition,
        )
        return losses, endpoint

    def compute_losses(
        self,
        batch: Mapping[str, Any],
        *,
        stage: str,
        epoch: int,
        stage_epochs: int,
        joint_batch: bool,
        joint_sample_count: int,
    ) -> dict[str, torch.Tensor]:
        encoding = self._encoding(batch)
        if stage == "stage_plan":
            return self._plan_loss(batch, encoding)
        plan_condition = batch["plan"].float()
        if stage == "stage_layout_mixed_plan":
            probability = generated_plan_probability(epoch, stage_epochs)
            if torch.rand((), device=self.device) < probability:
                with torch.no_grad():
                    plan_condition = self.designer.plan_flow.sample(
                        encoding,
                        steps=min(8, self.designer.plan_flow.sampling_steps),
                        method="euler",
                    ).compact_plan
        losses, endpoint = self._layout_loss(batch, encoding, plan_condition)
        if stage == "stage_joint_consistency" and joint_batch:
            if self.joint_loss_hook is None:
                raise RuntimeError("stage_joint_consistency selected a verifier batch without joint_loss_hook.")
            sample_count = min(max(int(joint_sample_count), 1), endpoint.shape[0])
            joint_batch_data = recursive_batch_slice(batch, sample_count)
            joint_encoding = slice_encoding(encoding, sample_count)
            joint_plan = plan_condition[:sample_count]
            joint_layout = endpoint[:sample_count]
            corrector_only = not any(
                parameter.requires_grad for parameter in self.designer.layout_flow.parameters()
            )
            if corrector_only:
                # Corrector-only refinement must see its real deployment
                # distribution, not the teacher-plan interpolation proxy used
                # to retain layout-flow gradients in ordinary stage four.
                with torch.no_grad():
                    sampled_plan = self.designer.plan_flow.sample(
                        joint_encoding,
                        steps=min(8, self.designer.plan_flow.sampling_steps),
                        method="euler",
                    )
                    sampled_layout = self.designer.layout_flow.sample(
                        sampled_plan.compact_plan,
                        joint_encoding,
                        geometry_constraints=joint_batch_data["geometry_constraints"],
                        steps=min(8, self.designer.layout_flow.sampling_steps),
                        method="euler",
                    )
                target_module_present = joint_batch_data["module_present"]
                joint_plan = sampled_plan.compact_plan
                joint_layout = sampled_layout.layout
                joint_batch_data = {
                    **joint_batch_data,
                    # Keep the feasible dataset layout mask separate from the
                    # sampled design mask.  The corrector cannot change module
                    # topology, so unmatched sampled slots must not be pulled
                    # toward zero-padded target coordinates.
                    "target_module_present": target_module_present,
                    "module_present": sampled_layout.module_present,
                }
            extra = self.joint_loss_hook(
                self.designer,
                joint_batch_data,
                joint_encoding,
                joint_plan,
                joint_layout,
            )
            extra_total = extra["total"]
            cap = losses["flow"].detach().clamp_min(1.0e-6) * 0.5
            scale = torch.clamp(cap / extra_total.detach().clamp_min(1.0e-8), max=1.0)
            losses = {**losses, **{f"joint_{name}": value for name, value in extra.items()}}
            losses["total"] = losses["total"] + extra_total * scale
        return losses

    def _run_epoch(
        self,
        loader: DataLoader,
        *,
        stage: str,
        epoch: int,
        stage_epochs: int,
        optimizer: torch.optim.Optimizer | None,
        joint_batch_fraction: float,
        joint_sample_count: int,
        phase: str,
    ) -> dict[str, float]:
        training = optimizer is not None
        self.designer.train(training)
        totals: dict[str, float] = {}
        batches = 0
        description = f"{stage} {phase} {epoch + 1}/{stage_epochs}"
        progress = tqdm(loader, desc=description, unit="batch", dynamic_ncols=True, leave=False)
        with progress:
            for batch_index, raw_batch in enumerate(progress):
                batch = recursive_to_device(raw_batch, self.device)
                joint_period = max(int(round(1.0 / max(joint_batch_fraction, 1.0e-9))), 1)
                joint_batch = (
                    stage == "stage_joint_consistency"
                    and joint_batch_fraction > 0.0
                    and batches % joint_period == 0
                )
                with torch.set_grad_enabled(training):
                    losses = self.compute_losses(
                        batch,
                        stage=stage,
                        epoch=epoch,
                        stage_epochs=stage_epochs,
                        joint_batch=joint_batch,
                        joint_sample_count=joint_sample_count,
                    )
                    nonfinite = [
                        name for name, value in losses.items()
                        if not bool(torch.isfinite(value.detach()).all())
                    ]
                    if nonfinite:
                        raise FloatingPointError(
                            f"Non-finite inverse loss at stage={stage}, phase={phase}, "
                            f"epoch={epoch + 1}, batch={batch_index + 1}: {nonfinite}"
                        )
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        if losses["total"].requires_grad:
                            losses["total"].backward()
                            torch.nn.utils.clip_grad_norm_(
                                [parameter for parameter in self.designer.parameters() if parameter.requires_grad], 1.0
                            )
                            optimizer.step()
                            self.global_step += 1
                batch_values = {name: float(value.detach()) for name, value in losses.items()}
                for name, value in batch_values.items():
                    totals[name] = totals.get(name, 0.0) + value
                batches += 1
                postfix = {
                    "loss": f"{batch_values['total']:.4g}",
                    "mean": f"{totals['total'] / batches:.4g}",
                    "flow": f"{totals['flow'] / batches:.4g}",
                }
                if joint_batch and "joint_total" in batch_values:
                    postfix["joint"] = f"{batch_values['joint_total']:.4g}"
                progress.set_postfix(postfix, refresh=False)
        if batches == 0:
            raise ValueError(f"Empty loader for inverse stage {stage}.")
        return {name: value / batches for name, value in totals.items()}

    def _append_metrics(self, row: Mapping[str, Any]) -> None:
        exists = self.metrics_path.exists()
        with self.metrics_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def train_stage(
        self,
        stage: str,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        *,
        epochs: int,
        learning_rate: float,
        weight_decay: float = 1.0e-5,
        joint_batch_fraction: float = 0.25,
        joint_sample_count: int = 4,
        corrector_only: bool = False,
    ) -> StageResult:
        configure_stage(self.designer, stage)
        if corrector_only:
            if stage != "stage_joint_consistency" or self.designer.corrector is None:
                raise ValueError("corrector_only is valid only for joint stage with a corrector.")
            for parameter in self.designer.layout_flow.parameters():
                parameter.requires_grad_(False)
        parameters = [parameter for parameter in self.designer.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=float(learning_rate), weight_decay=float(weight_decay))
        best_loss = float("inf")
        best_epoch = -1
        trainable_count = sum(parameter.numel() for parameter in parameters)
        try:
            train_batches = len(train_loader)
        except TypeError:
            train_batches = None
        try:
            validation_batches = len(validation_loader)
        except TypeError:
            validation_batches = None
        tqdm.write(
            f"[stage:start] {stage} epochs={epochs} lr={learning_rate:.3g} "
            f"train_batches={train_batches if train_batches is not None else '?'} "
            f"validation_batches={validation_batches if validation_batches is not None else '?'} "
            f"trainable_parameters={trainable_count:,}"
        )
        self._write_status(
            {
                "status": "running",
                "stage": stage,
                "epoch": 0,
                "stage_epochs": int(epochs),
                "global_step": self.global_step,
                "learning_rate": float(learning_rate),
                "train_batches": train_batches,
                "validation_batches": validation_batches,
            }
        )
        epoch_progress = tqdm(range(int(epochs)), desc=f"{stage} epochs", unit="epoch", dynamic_ncols=True)
        active_epoch = 0
        try:
            for epoch in epoch_progress:
                active_epoch = epoch + 1
                started = time.monotonic()
                train = self._run_epoch(
                    train_loader, stage=stage, epoch=epoch, stage_epochs=epochs,
                    optimizer=optimizer, joint_batch_fraction=joint_batch_fraction,
                    joint_sample_count=joint_sample_count, phase="train",
                )
                validation = self._run_epoch(
                    validation_loader, stage=stage, epoch=epoch, stage_epochs=epochs,
                    optimizer=None,
                    # Stage-four checkpoint selection must observe the same sparse
                    # HONF/correction objective; otherwise "best_corrected" would
                    # be selected by layout flow alone. Other stages ignore it.
                    joint_batch_fraction=joint_batch_fraction,
                    joint_sample_count=joint_sample_count, phase="validation",
                )
                improved = validation["total"] < best_loss
                if improved:
                    best_loss = validation["total"]
                    best_epoch = epoch
                row = {
                    "stage": stage,
                    "epoch": epoch,
                    "epoch_display": epoch + 1,
                    "stage_epochs": int(epochs),
                    "global_step": self.global_step,
                    "learning_rate": float(learning_rate),
                    "train_total": train["total"],
                    "train_flow": train["flow"],
                    "validation_total": validation["total"],
                    "validation_flow": validation["flow"],
                    "best_validation_total": best_loss,
                    "is_best": int(improved),
                    "epoch_seconds": time.monotonic() - started,
                }
                self._append_metrics(row)
                latest_path = save_inverse_checkpoint(
                    self.run_dir / "latest_model.pt",
                    designer=self.designer,
                    stage=stage,
                    epoch=epoch,
                    global_step=self.global_step,
                    provenance=self.provenance,
                    optimizer=optimizer,
                    metrics=row,
                )
                best_path = self.run_dir / self.ALIASES[stage]
                if improved:
                    best_path = save_inverse_checkpoint(
                        best_path,
                        designer=self.designer,
                        stage=stage,
                        epoch=epoch,
                        global_step=self.global_step,
                        provenance=self.provenance,
                        optimizer=optimizer,
                        metrics=row,
                    )
                plot_error = None
                try:
                    self._save_loss_curve()
                except Exception as error:  # Plotting must not invalidate a valid checkpoint.
                    plot_error = f"{type(error).__name__}: {error}"
                    tqdm.write(f"[plot:warning] {plot_error}")
                self._write_status(
                    {
                        "status": "running",
                        "stage": stage,
                        "epoch": epoch + 1,
                        "stage_epochs": int(epochs),
                        "global_step": self.global_step,
                        "train_losses": train,
                        "validation_losses": validation,
                        "best_validation_loss": best_loss,
                        "best_epoch": best_epoch + 1,
                        "latest_checkpoint": str(latest_path),
                        "best_checkpoint": str(best_path),
                        "loss_curve": str(self.loss_curve_path),
                        "loss_curve_error": plot_error,
                        "epoch_seconds": row["epoch_seconds"],
                    }
                )
                epoch_progress.set_postfix(
                    train=f"{train['total']:.4g}",
                    validation=f"{validation['total']:.4g}",
                    best=f"{best_loss:.4g}",
                    refresh=True,
                )
                tqdm.write(
                    f"[epoch] {stage} {epoch + 1}/{epochs} step={self.global_step} "
                    f"train={train['total']:.6g} validation={validation['total']:.6g} "
                    f"best={best_loss:.6g}@{best_epoch + 1} seconds={row['epoch_seconds']:.1f}"
                )
                tqdm.write(f"[checkpoint:latest] {latest_path}")
                if improved:
                    tqdm.write(f"[checkpoint:best] {best_path} validation={best_loss:.6g}")
        except BaseException as error:
            self._write_status(
                {
                    "status": "failed",
                    "stage": stage,
                    "epoch": active_epoch,
                    "stage_epochs": int(epochs),
                    "global_step": self.global_step,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            tqdm.write(f"[stage:error] {stage}: {type(error).__name__}: {error}")
            raise
        finally:
            epoch_progress.close()
        # Each stage hands its best validation state to the next stage. This is
        # especially important for small diagnostic datasets where the last
        # teacher-layout epoch can overfit badly. ``latest_model.pt`` still
        # preserves the literal last-epoch resume state.
        best_checkpoint = load_inverse_checkpoint(self.run_dir / self.ALIASES[stage])
        self.designer.load_state_dict(best_checkpoint["model_state_dict"])
        self._write_status(
            {
                "status": "stage_complete",
                "stage": stage,
                "epoch": int(epochs),
                "stage_epochs": int(epochs),
                "global_step": self.global_step,
                "best_validation_loss": best_loss,
                "best_epoch": best_epoch + 1,
                "latest_checkpoint": str(self.run_dir / "latest_model.pt"),
                "best_checkpoint": str(self.run_dir / self.ALIASES[stage]),
                "loss_curve": str(self.loss_curve_path),
            }
        )
        tqdm.write(
            f"[stage:complete] {stage} best_validation={best_loss:.6g} "
            f"best_epoch={best_epoch + 1} global_step={self.global_step}"
        )
        return StageResult(stage, best_loss, best_epoch, int(epochs), self.global_step)

    def mark_training_complete(self, results: list[Mapping[str, Any]]) -> None:
        """Publish final status after the workflow has completed every selected stage."""

        self._write_status(
            {
                "status": "complete",
                "global_step": self.global_step,
                "stages": [dict(result) for result in results],
                "latest_checkpoint": str(self.run_dir / "latest_model.pt"),
                "loss_curve": str(self.loss_curve_path),
                "metrics": str(self.metrics_path),
            }
        )
        tqdm.write(
            f"[training:complete] stages={len(results)} global_step={self.global_step} "
            f"metrics={self.metrics_path} loss_curve={self.loss_curve_path}"
        )


__all__ = [
    "InverseTrainer", "JointLossHook", "StageResult", "recursive_batch_slice",
    "recursive_to_device", "slice_encoding",
]

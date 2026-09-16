#!/usr/bin/env python3
"""Bounded CUDA memory diagnosis using the real ThermalChannel epoch path.

Run from ``HONF_Proj/`` with the physical GPU selected by the caller, for
example ``CUDA_VISIBLE_DEVICES=0 python tools/diagnostics/diagnose_routing_memory.py
--device cuda:0 ...``. The script loads one trusted checkpoint, runs a short
fixed-small-M and alternating-small/large-M training/validation sequence via
the maintained ``run_epoch`` function, and writes scalar allocator snapshots.
It never reserves a managed run directory or saves model/optimizer weights.

Forward hooks record allocator counters and weak references only. They do not
retain outputs or install autograd saved-tensor hooks. ``empty_cache`` runs once
at the end as a diagnostic comparison; no cache clearing is done between
phases.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
import weakref
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from channelthermal.data.collation import ChannelThermalBatchCollator
from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
from channelthermal.evaluation.loading import load_model
from channelthermal.training.checkpoints import _validate_resume_checkpoint
from channelthermal.training.epoch import (
    effective_local_loss_weights,
    effective_port_condition_settings,
    predicted_consistency_weight_for_epoch,
    run_epoch,
)
from channelthermal.training.optimizer import (
    _validate_optimizer_resume_compatibility,
    build_forward_optimizer,
)
from honf_runtime.compat import make_grad_scaler, strip_module_prefix
from honf_runtime.config_loader import load_config_bundle


class SequenceBatchSampler(Sampler[list[int]]):
    """Yield a finite, reproducible sequence of already-sized index batches."""

    def __init__(self, batches: Sequence[Sequence[int]]) -> None:
        self.batches = [list(batch) for batch in batches]

    def __iter__(self) -> Iterator[list[int]]:
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


class MemoryTracker:
    """Store allocator scalars and weak references, never live output tensors."""

    AUXILIARY_NORMS = ("main_context_norm", "coarse_context_norm", "local_context_norm")

    def __init__(self, device: torch.device, model: torch.nn.Module) -> None:
        self.device = device
        self.model = model
        self.phase = "setup"
        self.training = False
        self.clear_grad_before_forward = False
        self.optimizer: torch.optim.Optimizer | None = None
        self.current_batch: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []
        self.output_refs: list[weakref.ReferenceType[torch.Tensor]] = []
        self.output_tensor_count = 0
        self._forward_handles: list[Any] = []
        self._original_step: Any = None
        self._original_zero_grad: Any = None
        self.hook_counts_before: dict[str, int] | None = None

    def hook_counts(self) -> dict[str, int]:
        return {
            "forward_pre_hooks": len(getattr(self.model, "_forward_pre_hooks", {})),
            "forward_hooks": len(getattr(self.model, "_forward_hooks", {})),
        }

    def _memory_values(self) -> dict[str, Any]:
        sync_error = None
        try:
            torch.cuda.synchronize(self.device)
        except RuntimeError as exc:
            sync_error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        stats = torch.cuda.memory_stats(self.device)
        keys = {
            "active_bytes": "active_bytes.all.current",
            "inactive_split_bytes": "inactive_split_bytes.all.current",
            "peak_active_bytes": "active_bytes.all.peak",
            "peak_inactive_split_bytes": "inactive_split_bytes.all.peak",
            "alloc_retries": "num_alloc_retries",
            "oom_count": "num_ooms",
        }
        result: dict[str, int | None] = {
            "allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
        }
        result.update({name: int(stats[key]) if key in stats else None for name, key in keys.items()})
        if sync_error is not None:
            result["synchronize_error"] = sync_error
        return result

    def _live_output_refs(self) -> int:
        return sum(reference() is not None for reference in self.output_refs)

    def record(self, event: str, **extra: Any) -> None:
        record: dict[str, Any] = {
            "elapsed_seconds": round(time.perf_counter() - _START_TIME, 6),
            "phase": self.phase,
            "event": event,
            "batch": None if self.current_batch is None else dict(self.current_batch),
            "forward_output_weakrefs_seen": self.output_tensor_count,
            "forward_output_weakrefs_alive": self._live_output_refs(),
            "model_hook_counts": self.hook_counts(),
        }
        record.update(self._memory_values())
        record.update(extra)
        self.records.append(record)

    def _output_tensor_summary(self, output: Any) -> dict[str, Any]:
        seen_objects: set[int] = set()
        visited_containers: set[int] = set()
        byte_count = 0
        count = 0

        def visit(value: Any) -> None:
            nonlocal byte_count, count
            if torch.is_tensor(value):
                object_id = id(value)
                if object_id not in seen_objects:
                    seen_objects.add(object_id)
                    self.output_refs.append(weakref.ref(value))
                    self.output_tensor_count += 1
                    count += 1
                    byte_count += int(value.numel()) * int(value.element_size())
                return
            if isinstance(value, dict):
                identity = id(value)
                if identity in visited_containers:
                    return
                visited_containers.add(identity)
                for child in value.values():
                    visit(child)
            elif isinstance(value, (list, tuple)):
                identity = id(value)
                if identity in visited_containers:
                    return
                visited_containers.add(identity)
                for child in value:
                    visit(child)

        visit(output)
        return {"tensor_count": count, "tensor_bytes": byte_count}

    def _auxiliary_norm_summary(self, output: Any) -> dict[str, Any]:
        if not isinstance(output, dict):
            return {}
        aux = output.get("interaction_aux")
        if not isinstance(aux, dict):
            return {}
        summary: dict[str, Any] = {}
        for key in self.AUXILIARY_NORMS:
            value = aux.get(key)
            if torch.is_tensor(value):
                summary[key] = {
                    "shape": [int(size) for size in value.shape],
                    "numel": int(value.numel()),
                    "bytes": int(value.numel()) * int(value.element_size()),
                    "requires_grad": bool(value.requires_grad),
                    "grad_fn": None if value.grad_fn is None else type(value.grad_fn).__name__,
                }
        return summary

    def set_batch(self, context: dict[str, Any]) -> None:
        self.current_batch = dict(context)

    def _install_optimizer_observers(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self._original_step = optimizer.step
        self._original_zero_grad = optimizer.zero_grad

        def observed_step(*args: Any, **kwargs: Any) -> Any:
            self.record("before_optimizer_step")
            result = self._original_step(*args, **kwargs)
            self.record("after_optimizer_step")
            return result

        def observed_zero_grad(*args: Any, **kwargs: Any) -> Any:
            self.record("before_zero_grad")
            result = self._original_zero_grad(*args, **kwargs)
            self.record("after_zero_grad")
            return result

        optimizer.step = observed_step  # type: ignore[method-assign]
        optimizer.zero_grad = observed_zero_grad  # type: ignore[method-assign]

    def restore_optimizer_observers(self) -> None:
        if self.optimizer is not None:
            self.optimizer.step = self._original_step  # type: ignore[method-assign]
            self.optimizer.zero_grad = self._original_zero_grad  # type: ignore[method-assign]
        self.optimizer = None
        self._original_step = None
        self._original_zero_grad = None

    def start_phase(
        self,
        phase: str,
        *,
        training: bool,
        optimizer: torch.optim.Optimizer | None,
        clear_grad_before_forward: bool,
    ) -> None:
        self.phase = phase
        self.training = training
        self.clear_grad_before_forward = clear_grad_before_forward
        self.hook_counts_before = self.hook_counts()
        if optimizer is not None:
            self._install_optimizer_observers(optimizer)

        def before_forward(_module: torch.nn.Module, _inputs: tuple[Any, ...]) -> None:
            if (
                self.clear_grad_before_forward
                and self.training
                and self.optimizer is not None
            ):
                self.optimizer.zero_grad(set_to_none=True)
            self.record("before_forward")

        def after_forward(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            output_summary = self._output_tensor_summary(output)
            aux_summary = self._auxiliary_norm_summary(output)
            self.record(
                "after_forward",
                current_forward_output_tensor_count=output_summary["tensor_count"],
                current_forward_output_tensor_bytes=output_summary["tensor_bytes"],
                auxiliary_norms=aux_summary,
            )

        self._forward_handles = [
            self.model.register_forward_pre_hook(before_forward),
            self.model.register_forward_hook(after_forward),
        ]

    def stop_phase_hooks(self) -> None:
        for handle in self._forward_handles:
            handle.remove()
        self._forward_handles.clear()


class ObservedLoader:
    """Set scalar batch-shape context before yielding each ordinary batch."""

    def __init__(
        self,
        loader: DataLoader,
        tracker: MemoryTracker,
        labels: Sequence[str],
    ) -> None:
        self.loader = loader
        self.tracker = tracker
        self.labels = tuple(labels)

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for batch_index, batch in enumerate(self.loader, start=1):
            present = batch["structure"]["module_present"]
            counts = [int(value) for value in present.sum(dim=1).tolist()]
            query_xy = batch["query_xy"]
            case_ids = batch.get("case_id", ())
            if isinstance(case_ids, (tuple, list)):
                case_ids = [str(case_id) for case_id in case_ids]
            else:
                case_ids = [str(case_ids)]
            self.tracker.set_batch(
                {
                    "batch_index": batch_index,
                    "label": self.labels[batch_index - 1],
                    "batch_size": int(present.shape[0]),
                    "padded_module_width": int(present.shape[1]),
                    "active_modules_min": min(counts, default=0),
                    "active_modules_max": max(counts, default=0),
                    "query_points": int(query_xy.shape[1]),
                    "case_ids": case_ids,
                }
            )
            yield batch


def _route_identity(config: dict[str, Any]) -> tuple[str, str]:
    core = dict(config.get("model", {}).get("core_honf", {}))
    interface = dict(core.get("interface_model", {}))
    routing = dict(interface.get("routing", {}))
    return str(core.get("forward_architecture", "")), str(routing.get("strategy", ""))


def _exact_module_pool(dataset: GlobalChannelThermalDataset, module_count: int) -> list[int]:
    return [index for index, count in enumerate(dataset.selected_module_counts) if int(count) == module_count]


def _one_batch(pool: Sequence[int], batch_size: int, offset: int) -> list[int]:
    if not pool:
        raise ValueError("Cannot build a batch from an empty module-count group.")
    return [int(pool[(offset + index) % len(pool)]) for index in range(batch_size)]


def _make_plan(
    dataset: GlobalChannelThermalDataset,
    *,
    pattern: str,
    count: int,
    batch_size: int,
    small_m: int,
    large_m: int,
    fixed_m: int,
) -> tuple[list[list[int]], list[str]]:
    if count <= 0:
        return [], []
    if pattern == "fixed":
        groups = [(fixed_m, f"fixed_m{fixed_m}")] * count
    else:
        groups = [
            (small_m, f"small_m{small_m}") if index % 2 == 0 else (large_m, f"large_m{large_m}")
            for index in range(count)
        ]
    batches: list[list[int]] = []
    labels: list[str] = []
    for index, (module_count, label) in enumerate(groups):
        pool = _exact_module_pool(dataset, module_count)
        offset = 0 if pattern == "fixed" else index * batch_size
        batches.append(_one_batch(pool, batch_size, offset=offset))
        labels.append(label)
    return batches, labels


def _scenario_counts(dataset: GlobalChannelThermalDataset) -> tuple[int, int, int]:
    unique = sorted(set(int(value) for value in dataset.selected_module_counts))
    if not unique:
        raise ValueError("The selected dataset split is empty.")
    return unique[0], unique[0], unique[-1]


def _build_loader(
    dataset: GlobalChannelThermalDataset,
    *,
    batches: Sequence[Sequence[int]],
    labels: Sequence[str],
    batch_size: int,
    device: torch.device,
    collator: ChannelThermalBatchCollator,
    tracker: MemoryTracker,
) -> ObservedLoader:
    sampler = SequenceBatchSampler(batches)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    if any(len(batch) != batch_size for batch in batches):
        raise ValueError("Every diagnostic batch must match --batch-size.")
    return ObservedLoader(loader, tracker, labels)


def _parameter_finiteness(model: torch.nn.Module) -> dict[str, Any]:
    """Check parameters through temporary CPU copies, returning scalar metadata."""
    parameter_tensor_count = 0
    parameter_scalar_count = 0
    nonfinite_tensor_count = 0
    uninitialized_tensor_count = 0
    for parameter in model.parameters():
        try:
            numel = int(parameter.numel())
        except ValueError:
            uninitialized_tensor_count += 1
            continue
        if numel == 0:
            continue
        parameter_tensor_count += 1
        parameter_scalar_count += numel
        cpu_parameter = parameter.detach().to(device="cpu")
        if not bool(np.isfinite(cpu_parameter.numpy()).all()):
            nonfinite_tensor_count += 1
        del cpu_parameter
    return {
        "parameter_tensor_count": parameter_tensor_count,
        "parameter_scalar_count": parameter_scalar_count,
        "nonfinite_tensor_count": nonfinite_tensor_count,
        "uninitialized_tensor_count": uninitialized_tensor_count,
        "all_finite": nonfinite_tensor_count == 0,
    }


def _run_phase(
    tracker: MemoryTracker,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any,
    loader: ObservedLoader,
    phase: str,
    training: bool,
    amp: bool,
    loss_cfg: dict[str, Any],
    local_mode: str,
    teacher_ratio: float,
    internal_weight: float,
    interface_weight: float,
    predicted_weight: float,
    gradient_clip_norm: float,
    capture_update: bool,
    clear_grad_before_forward: bool,
) -> dict[str, Any]:
    tracker.start_phase(
        phase,
        training=training,
        optimizer=optimizer,
        clear_grad_before_forward=clear_grad_before_forward,
    )
    torch.cuda.reset_peak_memory_stats(tracker.device)
    tracker.record("before_run_epoch")
    error: str | None = None
    metrics: dict[str, float] | None = None
    try:
        metrics = run_epoch(
            model,  # type: ignore[arg-type]
            loader,  # type: ignore[arg-type]
            tracker.device,
            loss_cfg,
            optimizer=optimizer,
            scaler=scaler if training else None,
            amp=amp,
            max_batches=len(loader),
            local_port_condition_mode=local_mode,
            mixed_teacher_ratio=teacher_ratio,
            effective_internal_temperature_weight=internal_weight,
            effective_interface_weight=interface_weight,
            predicted_consistency_weight=predicted_weight,
            gradient_clip_norm=gradient_clip_norm,
            record_gradient_diagnostics=bool(capture_update and training),
        )
    except torch.cuda.OutOfMemoryError as exc:
        error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        exc.__traceback__ = None
    finally:
        tracker.record("after_run_epoch_return" if error is None else "after_run_epoch_oom", error=error)
        tracker.stop_phase_hooks()
        tracker.record("after_hook_clear_before_gc", hook_counts_before=tracker.hook_counts_before)

    if training and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    tracker.record("after_zero_grad_before_gc" if training else "after_validation_return_before_gc")
    result = {
        "phase": phase,
        "status": "oom" if error is not None else "complete",
        "error": error,
        "metrics": {
            key: float(metrics[key])
            for key in ("loss_total", "field_mse", "temperature_mse")
            if metrics is not None and key in metrics
        },
    }
    del metrics
    tracker.record("after_del_before_gc")
    collected = int(gc.collect())
    tracker.record("after_gc", collected_objects=collected)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trusted local forward checkpoint path.")
    parser.add_argument("--profile", required=True, help="Core profile URI/path used to validate routing identity.")
    parser.add_argument("--device", required=True, help="CUDA device visible to this process; use CUDA_VISIBLE_DEVICES in the caller.")
    parser.add_argument("--output", required=True, help="JSON output path; no run/checkpoint artifacts are written.")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--query-points", type=int, default=1024)
    parser.add_argument("--pattern", choices=("both", "fixed", "alternating"), default="both")
    parser.add_argument("--fixed-train-batches", type=int, default=2)
    parser.add_argument("--alternating-train-batches", type=int, default=4)
    parser.add_argument("--fixed-val-batches", type=int, default=1)
    parser.add_argument("--alternating-val-batches", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=None, help="Training epoch schedule; defaults to checkpoint epoch + 1.")
    parser.add_argument(
        "--capture-update",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the production first-batch capture-update schedule.",
    )
    parser.add_argument("--zero-grad-before-forward", action="store_true", help="Diagnostic timing intervention; clear previous gradients at the next forward boundary.")
    parser.add_argument(
        "--restore-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore checkpoint AdamW state (default); use --no-restore-optimizer for a cold optimizer.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    global _START_TIME
    _START_TIME = time.perf_counter()
    args = parse_args(argv)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose the physical GPU with caller-side CUDA_VISIBLE_DEVICES.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must name a CUDA device.")
    if args.batch_size <= 0 or args.query_points <= 0:
        raise ValueError("--batch-size and --query-points must be positive.")

    profile = load_config_bundle(args.profile, project_root=PROJECT_ROOT).effective
    if profile.get("workflow") != "forward" or profile.get("model_family") != "honf_forward":
        raise ValueError("--profile must resolve to a ThermalChannel HONF forward profile.")
    if str(profile.get("case", {}).get("id", "")) != "ThermalChannel":
        raise ValueError("--profile must target ThermalChannel.")

    # The maintained evaluator uses the trusted checkpoint loader, checks model
    # identity, and strictly restores the checkpoint state_dict.
    model, checkpoint = load_model(checkpoint_path, device)
    checkpoint_model_config = dict(checkpoint.get("model_config") or {})
    profile_route = _route_identity(profile)
    checkpoint_route = _route_identity({"model": checkpoint_model_config})
    if profile_route != checkpoint_route:
        raise ValueError(
            f"Profile route {profile_route} does not match checkpoint route {checkpoint_route}."
        )
    if profile_route[0] != "routed_pairwise_honf" or profile_route[1] not in {"module_hubs", "mean_shift"}:
        raise ValueError(f"This diagnosis expects the routed ThermalChannel profiles, got {profile_route}.")

    train_config = dict(checkpoint.get("train_config") or {})
    dataset_cfg = dict(train_config.get("dataset") or {})
    training_cfg = dict(train_config.get("training") or {})
    loss_cfg = dict(train_config.get("loss") or {})
    checkpoint_stats = {
        name: value for name, value in dict(checkpoint.get("global_normalization_stats") or {}).items()
    }
    normalizer = H5Normalizer(checkpoint_stats) if checkpoint_stats else None
    train_dataset = GlobalChannelThermalDataset(
        dataset_cfg.get("packed_h5_path", "./Case_ThermalChannel/Dataset/links/thermal_channel_global_v1.h5"),
        split=dataset_cfg.get("train_split", "train"),
        points_per_case=int(args.query_points),
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=bool(dataset_cfg.get("random_point_sampling", True)),
        seed=int(training_cfg.get("seed", 42)),
        include_grid=False,
        require_converged=bool(dataset_cfg.get("require_converged", False)),
        normalizer=normalizer,
    )
    val_dataset = GlobalChannelThermalDataset(
        dataset_cfg.get("packed_h5_path", "./Case_ThermalChannel/Dataset/links/thermal_channel_global_v1.h5"),
        split=dataset_cfg.get("val_split", "test"),
        points_per_case=int(args.query_points),
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=False,
        seed=int(training_cfg.get("seed", 42)) + 1000,
        include_grid=False,
        require_converged=bool(dataset_cfg.get("require_converged", False)),
        normalizer=train_dataset.normalizer,
    )
    if len(val_dataset) == 0:
        if bool(dataset_cfg.get("allow_train_as_validation", False)):
            val_dataset = train_dataset
        else:
            raise ValueError("The checkpoint validation split is empty.")
    _validate_resume_checkpoint(
        checkpoint,
        model=model,
        model_config=model.config,
        dataset=train_dataset,
        dataset_config=dict(checkpoint.get("train_config", {}).get("dataset", {}) or {}),
    )

    optimizer, optimizer_inventory = build_forward_optimizer(model, training_cfg)
    optimizer_state = checkpoint.get("optimizer_state_dict")
    optimizer_restored = bool(args.restore_optimizer and optimizer_state)
    if optimizer_restored:
        _validate_optimizer_resume_compatibility(checkpoint, optimizer_inventory)
        optimizer.load_state_dict(optimizer_state)
    scaler = make_grad_scaler(device, bool(training_cfg.get("amp", False)))
    tracker = MemoryTracker(device, model)
    tracker.record("setup_complete", optimizer_restored=optimizer_restored)

    checkpoint_epoch = int(checkpoint.get("epoch", checkpoint.get("current_epoch", 0)) or 0)
    epoch = int(args.epoch) if args.epoch is not None else checkpoint_epoch + 1
    total_epochs = int(training_cfg.get("epochs", epoch))
    capture_update = (
        epoch in {1, 2, 5, 10, 20} or epoch % 50 == 0
        if args.capture_update is None
        else bool(args.capture_update)
    )
    seed = int(training_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.set_training_progress(epoch=epoch, total_epochs=total_epochs)
    local_mode, teacher_ratio = effective_port_condition_settings(epoch, training_cfg)
    internal_weight, interface_weight = effective_local_loss_weights(loss_cfg, local_mode, teacher_ratio)
    predicted_weight = predicted_consistency_weight_for_epoch(epoch, loss_cfg)
    gradient_clip_norm = float(training_cfg.get("gradient_clip_norm", 0.0) or 0.0)
    amp = bool(training_cfg.get("amp", False))

    train_fixed_m, train_small_m, train_large_m = _scenario_counts(train_dataset)
    val_fixed_m, val_small_m, val_large_m = _scenario_counts(val_dataset)
    collator = ChannelThermalBatchCollator(
        dynamic_module_padding=bool(dataset_cfg.get("dynamic_module_padding", True)),
        max_modules_per_batch=(
            None if dataset_cfg.get("max_modules_per_batch") is None else int(dataset_cfg["max_modules_per_batch"])
        ),
    )
    patterns = ("fixed", "alternating") if args.pattern == "both" else (args.pattern,)
    pattern_results: list[dict[str, Any]] = []
    oom_seen = False
    try:
        for pattern in patterns:
            # Start both shape policies from the exact same in-memory checkpoint
            # state and optimizer state, then run train followed by validation.
            model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]), strict=True)
            if optimizer_restored:
                optimizer.load_state_dict(optimizer_state)
            optimizer.zero_grad(set_to_none=True)
            train_dataset.set_epoch(epoch)
            train_count = args.fixed_train_batches if pattern == "fixed" else args.alternating_train_batches
            train_batches, train_labels = _make_plan(
                train_dataset,
                pattern=pattern,
                count=train_count,
                batch_size=args.batch_size,
                small_m=train_small_m,
                large_m=train_large_m,
                fixed_m=train_fixed_m,
            )
            train_loader = _build_loader(
                train_dataset,
                batches=train_batches,
                labels=train_labels,
                batch_size=args.batch_size,
                device=device,
                collator=collator,
                tracker=tracker,
            )
            train_result = _run_phase(
                tracker,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                loader=train_loader,
                phase=f"train_{pattern}",
                training=True,
                amp=amp,
                loss_cfg=loss_cfg,
                local_mode=local_mode,
                teacher_ratio=teacher_ratio,
                internal_weight=internal_weight,
                interface_weight=interface_weight,
                predicted_weight=predicted_weight,
                gradient_clip_norm=gradient_clip_norm,
                capture_update=capture_update,
                clear_grad_before_forward=bool(args.zero_grad_before_forward),
            )
            train_result["parameter_finiteness"] = _parameter_finiteness(model)
            tracker.restore_optimizer_observers()
            del train_loader, train_batches, train_labels
            if train_result["status"] == "oom":
                oom_seen = True
                pattern_results.append({"pattern": pattern, "train": train_result, "validation": None})
                break

            val_dataset.set_epoch(epoch)
            val_count = args.fixed_val_batches if pattern == "fixed" else args.alternating_val_batches
            val_batches, val_labels = _make_plan(
                val_dataset,
                pattern=pattern,
                count=val_count,
                batch_size=args.batch_size,
                small_m=val_small_m,
                large_m=val_large_m,
                fixed_m=val_fixed_m,
            )
            val_loader = _build_loader(
                val_dataset,
                batches=val_batches,
                labels=val_labels,
                batch_size=args.batch_size,
                device=device,
                collator=collator,
                tracker=tracker,
            )
            val_result = _run_phase(
                tracker,
                model=model,
                optimizer=None,
                scaler=None,
                loader=val_loader,
                phase=f"validation_{pattern}",
                training=False,
                amp=amp,
                loss_cfg=loss_cfg,
                local_mode=local_mode,
                teacher_ratio=teacher_ratio,
                internal_weight=internal_weight,
                interface_weight=interface_weight,
                predicted_weight=predicted_weight,
                gradient_clip_norm=gradient_clip_norm,
                capture_update=False,
                clear_grad_before_forward=False,
            )
            val_result["parameter_finiteness"] = _parameter_finiteness(model)
            del val_loader, val_batches, val_labels
            pattern_results.append({"pattern": pattern, "train": train_result, "validation": val_result})
            if val_result["status"] == "oom":
                oom_seen = True
                break
    finally:
        tracker.restore_optimizer_observers()
        tracker.stop_phase_hooks()
        try:
            tracker.record("before_final_empty_cache")
            torch.cuda.empty_cache()
            tracker.record("after_final_empty_cache")
        except RuntimeError:
            pass

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    module_counts = {
        "train": {
            "fixed_small_m": train_fixed_m,
            "alternating_small_m": train_small_m,
            "alternating_large_m": train_large_m,
            "case_count": len(train_dataset),
        },
        "validation": {
            "fixed_small_m": val_fixed_m,
            "alternating_small_m": val_small_m,
            "alternating_large_m": val_large_m,
            "case_count": len(val_dataset),
        },
    }
    payload = {
        "schema_version": 1,
        "status": "oom" if oom_seen else "complete",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "profile": str(args.profile),
        "forward_architecture": profile_route[0],
        "routing_strategy": profile_route[1],
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "cuda_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "allocator_backend": (
            str(torch.cuda.get_allocator_backend())
            if hasattr(torch.cuda, "get_allocator_backend")
            else None
        ),
        "allocator_environment": {
            key: os.environ.get(key)
            for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
            if os.environ.get(key) is not None
        },
        "batch_size": int(args.batch_size),
        "query_points": int(args.query_points),
        "training_epoch_schedule": epoch,
        "validation_mode": local_mode,
        "restore_optimizer": optimizer_restored,
        "capture_update_diagnostics": capture_update,
        "zero_grad_before_forward_intervention": bool(args.zero_grad_before_forward),
        "scenario_module_counts": module_counts,
        "scenarios": pattern_results,
        "events": tracker.records,
        "interpretation": {
            "allocated_bytes": "live CUDA tensor allocations at the sampling point",
            "reserved_bytes": "allocator-managed CUDA memory including reusable cached blocks",
            "inactive_split_bytes": "reserved split blocks currently inactive",
            "weakrefs": "forward output tensors tracked by weak reference only; any survivors after GC are live elsewhere",
            "empty_cache": "final diagnostic release of unused cache; does not free live tensors",
        },
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"[done] memory diagnosis saved: {output_path}")
    return 1 if oom_seen else 0


_START_TIME = time.perf_counter()


if __name__ == "__main__":
    raise SystemExit(main())

"""Benchmark routed execution on matched real ThermalChannel batches.

This is a bounded, read-only execution benchmark.  It loads explicitly named
checkpoints, samples the same case IDs and fixed ``Q`` query points for every
variant, and runs one-batch ``run_epoch`` steps so the timing includes forward,
backward, and the optimizer update.  It never creates a managed run or writes
weights.  The output is written only to the path supplied with ``--output``.

The routed variants are:

``reference``
    Historical path compiler and packed QE reader.
``compileroptimized``
    Integer-only path union followed by the tiled live-prior compiler.
``compiler+denseQE``
    ``compileroptimized`` plus the opt-in complete-support QE reader.  The
    complete reader is eligible only when the compiled environmental support
    contains every ``(batch, query, environment)`` pair.  A one-step counter
    pass records whole-batch and hybrid complete-support rows separately from
    packed fallback tiles.
``batched+denseQE``
    ``compiler+denseQE`` with the bounded batched FP64 scalar compiler
    (``scalar_tile_size=262144``, ``use_checkpoint=False``).  Its scalar tile
    budget is independent of the fine neural pair tile size.

The intended endpoint command supplies the four matched labels (1401, 1804,
2000, and 2100), ``--batch-size 48``, and ``--query-points 1024``.  GPU work is
deliberately caller-selected; importing or running with ``--plan-only`` does
not launch training.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import math
import re
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "Case_ThermalChannel" / "src",
    PROJECT_ROOT / "tools",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from channelthermal.data.collation import ChannelThermalBatchCollator
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
from run_stage3_interface_study import (
    CheckpointSpec,
    _load_dataset,
    _load_model_spec,
    parse_checkpoint_specs,
    select_device,
    write_json,
)

from honf_runtime.compat import load_trusted_checkpoint

VARIANTS = ("reference", "compileroptimized", "compiler+denseQE", "batched+denseQE")
EXPECTED_RUN_IDS = ("1401", "1804", "2000", "2100")
BATCHED_SCALAR_TILE_SIZE = 262_144


@dataclass(frozen=True)
class Bucket:
    """One fixed module-width bucket shared by every checkpoint."""

    label: str
    module_count: int
    case_ids: tuple[str, ...]


def _json_value(value: Any) -> Any:
    """Convert tensors, NumPy values, paths, and nonfinite floats to JSON."""

    if torch.is_tensor(value):
        if value.numel() == 1:
            return _json_value(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_value(value.item())
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _synchronize(device: torch.device) -> None:
    """Synchronize only at measured step boundaries."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _training_schedule(checkpoint: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str, float, float, float, float]:
    """Resolve the maintained loss schedule used by ``run_epoch``."""

    train_config = dict(checkpoint.get("train_config") or {})
    training_config = dict(train_config.get("training") or {})
    loss_config = dict(train_config.get("loss") or {})
    epoch = int(checkpoint.get("epoch", checkpoint.get("current_epoch", 0)) or 0)
    mode, ratio = effective_port_condition_settings(epoch, training_config)
    internal_weight, interface_weight = effective_local_loss_weights(loss_config, mode, ratio)
    predicted_weight = predicted_consistency_weight_for_epoch(epoch, loss_config)
    return (
        training_config,
        loss_config,
        str(mode),
        float(ratio),
        float(internal_weight),
        float(interface_weight),
        float(predicted_weight),
    )


def _dataset_args(args: argparse.Namespace) -> argparse.Namespace:
    """Provide the small argument surface used by the canonical dataset loader."""

    return SimpleNamespace(
        dataset=args.dataset,
        split=args.split,
    )


def _case_to_index(dataset: Any) -> dict[str, int]:
    return {str(case_id): index for index, case_id in enumerate(dataset.selected_case_ids)}


def _common_case_ids(datasets: Sequence[Any]) -> list[str]:
    if not datasets:
        raise ValueError("At least one dataset is required.")
    common = {str(case_id) for case_id in datasets[0].selected_case_ids}
    for dataset in datasets[1:]:
        common.intersection_update(str(case_id) for case_id in dataset.selected_case_ids)
    if not common:
        raise ValueError("The supplied checkpoints have no common case IDs in the selected split.")
    return sorted(common)


def _case_count(dataset: Any, case_id: str) -> int:
    index = _case_to_index(dataset).get(str(case_id))
    if index is None:
        raise KeyError(f"case_id={case_id!r} is absent from split={dataset.split!r}.")
    return int(dataset.selected_module_counts[index])


def _make_buckets(
    datasets: Sequence[Any],
    *,
    batch_size: int,
    requested: str,
) -> list[Bucket]:
    """Select minimum and maximum real module widths before timing anything."""

    common = _common_case_ids(datasets)
    counts = {case_id: _case_count(datasets[0], case_id) for case_id in common}
    for dataset in datasets[1:]:
        for case_id in common:
            if _case_count(dataset, case_id) != counts[case_id]:
                raise ValueError(
                    "Matched checkpoint datasets disagree on module count for "
                    f"case_id={case_id!r}."
                )
    distinct = sorted(set(counts.values()))
    if requested == "min":
        selected = distinct[:1]
    elif requested == "max":
        selected = distinct[-1:]
    else:
        selected = distinct if len(distinct) == 1 else [distinct[0], distinct[-1]]

    buckets: list[Bucket] = []
    for module_count in selected:
        candidates = sorted(case_id for case_id, count in counts.items() if count == module_count)
        if not candidates:
            raise RuntimeError(f"No common real case has module count M={module_count}.")
        # Cycling is explicit metadata, and keeps B=48 fixed even when a
        # width bucket contains fewer than 48 physical cases.  Every model
        # receives the identical ordered case-ID sequence.
        case_ids = tuple(candidates[index % len(candidates)] for index in range(batch_size))
        buckets.append(Bucket(f"M{module_count}", module_count, case_ids))
    return buckets


def _build_loader(
    dataset: Any,
    checkpoint: Mapping[str, Any],
    bucket: Bucket,
    *,
    batch_size: int,
) -> DataLoader:
    index_by_case = _case_to_index(dataset)
    indices = [index_by_case[case_id] for case_id in bucket.case_ids]
    train_config = dict(checkpoint.get("train_config") or {})
    dataset_config = dict(train_config.get("dataset") or {})
    collator = ChannelThermalBatchCollator(
        dynamic_module_padding=bool(dataset_config.get("dynamic_module_padding", True)),
        max_modules_per_batch=dataset_config.get("max_modules_per_batch"),
    )
    return DataLoader(
        Subset(dataset, indices),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collator,
    )


def _validate_batch_shape(loader: DataLoader, device: torch.device, query_points: int) -> dict[str, Any]:
    """Check the real collated batch once, before any timed step."""

    iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration as exc:
        raise ValueError("The benchmark loader produced no batch.") from exc
    query_xy = batch.get("query_xy")
    structure = batch.get("structure", {})
    present = structure.get("module_present")
    if not torch.is_tensor(query_xy) or query_xy.ndim != 3:
        raise ValueError("The collated real batch has no [B,Q,d] query_xy tensor.")
    if int(query_xy.shape[1]) != int(query_points):
        raise ValueError(
            f"Real loader produced Q={int(query_xy.shape[1])}; expected --query-points={query_points}."
        )
    if not torch.is_tensor(present) or present.ndim != 2:
        raise ValueError("The collated real batch has no [B,M] module_present tensor.")
    result = {
        "batch_size": int(query_xy.shape[0]),
        "query_points": int(query_xy.shape[1]),
        "padded_module_width": int(present.shape[1]),
        "active_modules_min": int(present.sum(dim=1).min().item()),
        "active_modules_max": int(present.sum(dim=1).max().item()),
    }
    del batch
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


@contextlib.contextmanager
def _routing_variant(
    model: torch.nn.Module,
    variant: str,
    *,
    pair_tile_size: int,
    dense_pair_tile_size: int,
) -> Iterator[None]:
    """Select the reference/optimized compiler and optional dense QE path."""

    if variant not in VARIANTS:
        raise ValueError(f"Unknown routing variant {variant!r}; expected one of {VARIANTS!r}.")
    from honf_forward_core.interface_fields import routed_pairwise
    from honf_forward_core.interface_fields.routing_index.pair_join import (
        compile_two_hop_pairs_batched,
        compile_two_hop_pairs_optimized,
        compile_two_hop_pairs_reference,
    )

    original_dispatch = routed_pairwise.compile_positive_pairs
    original_optimized_dispatch = getattr(routed_pairwise, "compile_two_hop_pairs_optimized", None)
    original_batched_dispatch = routed_pairwise.compile_two_hop_pairs_batched
    if variant == "reference":
        compiler = compile_two_hop_pairs_reference
    elif variant == "batched+denseQE":
        compiler = compile_two_hop_pairs_batched
    else:
        compiler = compile_two_hop_pairs_optimized

    def dispatch(density: torch.Tensor, incidence: Any) -> Any:
        if compiler is compile_two_hop_pairs_reference:
            return compiler(density, incidence)
        if compiler is compile_two_hop_pairs_batched:
            return compiler(
                density,
                incidence,
                scalar_tile_size=BATCHED_SCALAR_TILE_SIZE,
                use_checkpoint=False,
            )
        return compiler(
            density,
            incidence,
            pair_tile_size=int(pair_tile_size),
            use_checkpoint=True,
        )

    routed_pairwise.compile_positive_pairs = dispatch

    def batched_dispatch(density: torch.Tensor, incidence: Any, **kwargs: Any) -> Any:
        """Keep legacy optimized variants independent of production default."""

        del kwargs
        if compiler is compile_two_hop_pairs_batched:
            return compiler(
                density,
                incidence,
                scalar_tile_size=BATCHED_SCALAR_TILE_SIZE,
                use_checkpoint=False,
            )
        if compiler is compile_two_hop_pairs_reference:
            return compiler(density, incidence)
        return compiler(
            density,
            incidence,
            pair_tile_size=int(pair_tile_size),
            use_checkpoint=True,
        )

    # Current production ``optimized_exact`` calls the batched symbol
    # directly.  Patching both symbols keeps the earlier optimized variants
    # comparable with their pre-batched benchmark definition.
    routed_pairwise.compile_two_hop_pairs_batched = batched_dispatch

    def optimized_dispatch(density: torch.Tensor, incidence: Any, **kwargs: Any) -> Any:
        # The production optimized reader passes fine_pair_chunk_size as the
        # prior tile size.  Patch only the compiler symbol here so a benchmark
        # tile-size sweep does not also alter the fine MLP tile size.  The
        # batched candidate has its own scalar budget and deliberately ignores
        # the optimized compiler's pair-tile keyword.
        kwargs = dict(kwargs)
        if compiler is compile_two_hop_pairs_batched:
            del kwargs
            return compiler(
                density,
                incidence,
                scalar_tile_size=BATCHED_SCALAR_TILE_SIZE,
                use_checkpoint=False,
            )
        kwargs["pair_tile_size"] = int(pair_tile_size)
        return compiler(density, incidence, **kwargs)

    if original_optimized_dispatch is not None:
        routed_pairwise.compile_two_hop_pairs_optimized = optimized_dispatch
    saved_fields: list[tuple[Any, str | None, bool, int]] = []
    for field in model.modules():
        if hasattr(field, "routing_execution") and hasattr(field, "dense_environment_fast_path"):
            saved_fields.append(
                (
                    field,
                    str(field.routing_execution),
                    bool(field.dense_environment_fast_path),
                    int(getattr(field, "dense_environment_pair_tile_size", dense_pair_tile_size)),
                )
            )
            field.routing_execution = "gathered" if variant == "reference" else "optimized_exact"
            field.dense_environment_fast_path = variant in ("compiler+denseQE", "batched+denseQE")
            field.dense_environment_pair_tile_size = int(dense_pair_tile_size)
        elif hasattr(field, "dense_environment_fast_path"):
            saved_fields.append(
                (
                    field,
                    None,
                    bool(field.dense_environment_fast_path),
                    int(getattr(field, "dense_environment_pair_tile_size", dense_pair_tile_size)),
                )
            )
            field.dense_environment_fast_path = variant in ("compiler+denseQE", "batched+denseQE")
            field.dense_environment_pair_tile_size = int(dense_pair_tile_size)
    try:
        yield
    finally:
        routed_pairwise.compile_positive_pairs = original_dispatch
        if original_optimized_dispatch is not None:
            routed_pairwise.compile_two_hop_pairs_optimized = original_optimized_dispatch
        routed_pairwise.compile_two_hop_pairs_batched = original_batched_dispatch
        for field, old_execution, old_fast_path, old_tile_size in saved_fields:
            if old_execution is not None:
                field.routing_execution = old_execution
            field.dense_environment_fast_path = old_fast_path
            field.dense_environment_pair_tile_size = old_tile_size


class _RouteCounter:
    """One untimed forward counter pass for the hybrid QE dispatch.

    ``read_environment_pairs`` receives one packed union for the whole batch.
    The hybrid reader can nevertheless send complete-support rows through the
    dense callback while sending partial rows through the packed evaluator.
    Keep those cases separate so a zero whole-batch count is not misread as a
    zero complete-row count.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.environment_read_calls = 0
        # Retain this historical field as the exact whole-batch fast-path
        # count.  The explicit names below are used in new output.
        self.full_support_read_calls = 0
        self.complete_support_read_calls = 0
        self.whole_batch_complete_read_calls = 0
        self.hybrid_read_calls = 0
        self.packed_only_read_calls = 0
        self.complete_support_batch_rows = 0
        self.packed_fallback_batch_rows = 0
        self.complete_tile_calls = 0
        self.packed_score_tile_calls = 0
        self._restorations: list[tuple[Any, str, Any]] = []

    def install(self) -> None:
        for field in self.model.modules():
            if hasattr(field, "read_environment_pairs") and hasattr(field, "_environment_complete_tile"):
                original_read = field.read_environment_pairs
                self._restorations.append((field, "read_environment_pairs", original_read))

                def counted_read(
                    state: dict[str, Any],
                    encoded: Any,
                    receivers: torch.Tensor,
                    receiver_features: torch.Tensor,
                    pairs: Any,
                    *,
                    _original: Any = original_read,
                    _field: Any = field,
                ) -> Any:
                    self.environment_read_calls += 1
                    key = state.get("env_keys")
                    source_count = (
                        int(key.shape[-2])
                        if torch.is_tensor(key)
                        else int(encoded.env_coords.shape[1])
                    )
                    batch = int(receivers.shape[0])
                    query_count = int(receivers.shape[1])
                    pair_counts = torch.bincount(
                        pairs.batch_index,
                        minlength=batch,
                    )
                    complete_rows = pair_counts == query_count * source_count
                    complete_row_count = int(complete_rows.sum().item())
                    fallback_row_count = batch - complete_row_count
                    fast_path = bool(getattr(_field, "dense_environment_fast_path", False))
                    if fast_path and complete_row_count:
                        self.complete_support_read_calls += 1
                        self.complete_support_batch_rows += complete_row_count
                    if fast_path and fallback_row_count:
                        self.packed_fallback_batch_rows += fallback_row_count
                    if not fast_path or complete_row_count == 0:
                        self.packed_only_read_calls += 1
                    elif fallback_row_count == 0:
                        self.full_support_read_calls += 1
                        self.whole_batch_complete_read_calls += 1
                    else:
                        self.hybrid_read_calls += 1
                    return _original(state, encoded, receivers, receiver_features, pairs)

                field.read_environment_pairs = counted_read
                original_complete = field._environment_complete_tile
                self._restorations.append((field, "_environment_complete_tile", original_complete))

                def counted_complete(*args: Any, _original: Any = original_complete, **kwargs: Any) -> Any:
                    self.complete_tile_calls += 1
                    return _original(*args, **kwargs)

                field._environment_complete_tile = counted_complete
                original_score = field._environment_score_tile
                self._restorations.append((field, "_environment_score_tile", original_score))

                def counted_score(*args: Any, _original: Any = original_score, **kwargs: Any) -> Any:
                    self.packed_score_tile_calls += 1
                    return _original(*args, **kwargs)

                field._environment_score_tile = counted_score

    def restore(self) -> None:
        for field, name, original in reversed(self._restorations):
            setattr(field, name, original)
        self._restorations.clear()

    def __enter__(self) -> Self:
        self.install()
        return self

    def __exit__(self, *_: object) -> None:
        self.restore()


def _compiler_policy(variant: str, *, pair_tile_size: int) -> dict[str, Any]:
    """Return the exact compiler controls used by one benchmark variant."""

    if variant == "reference":
        return {
            "compiler": "compile_two_hop_pairs_reference",
            "pair_tile_size": None,
            "scalar_tile_size": None,
            "use_checkpoint": False,
            "dense_environment_fast_path": False,
        }
    if variant == "compileroptimized":
        return {
            "compiler": "compile_two_hop_pairs_optimized",
            "pair_tile_size": int(pair_tile_size),
            "scalar_tile_size": None,
            "use_checkpoint": True,
            "dense_environment_fast_path": False,
        }
    if variant == "compiler+denseQE":
        return {
            "compiler": "compile_two_hop_pairs_optimized",
            "pair_tile_size": int(pair_tile_size),
            "scalar_tile_size": None,
            "use_checkpoint": True,
            "dense_environment_fast_path": True,
        }
    if variant == "batched+denseQE":
        return {
            "compiler": "compile_two_hop_pairs_batched",
            "pair_tile_size": None,
            "scalar_tile_size": BATCHED_SCALAR_TILE_SIZE,
            "use_checkpoint": False,
            "dense_environment_fast_path": True,
        }
    raise ValueError(f"Unknown routing variant {variant!r}.")


def _run_step(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    checkpoint: Mapping[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None,
    training_schedule: tuple[dict[str, Any], dict[str, Any], str, float, float, float, float],
) -> dict[str, float]:
    training_config, loss_config, mode, ratio, internal_weight, interface_weight, predicted_weight = training_schedule
    gradient_clip_norm = float(
        training_config.get("gradient_clip_norm", training_config.get("grad_clip_norm", 0.0))
    )
    del checkpoint
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return run_epoch(
            model,
            loader,
            device,
            loss_config,
            optimizer=optimizer,
            scaler=None,
            amp=False,
            max_batches=1,
            local_port_condition_mode=mode,
            mixed_teacher_ratio=ratio,
            effective_internal_temperature_weight=internal_weight,
            effective_interface_weight=interface_weight,
            predicted_consistency_weight=predicted_weight,
            gradient_clip_norm=gradient_clip_norm,
            record_gradient_diagnostics=False,
        )


def _measure_variant(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    checkpoint: Mapping[str, Any],
    *,
    variant: str,
    warmups: int,
    repetitions: int,
    pair_tile_size: int,
    dense_pair_tile_size: int,
) -> dict[str, Any]:
    schedule = _training_schedule(checkpoint)
    optimizer, optimizer_inventory = build_forward_optimizer(model, schedule[0])
    optimizer_state_policy = "cold_no_checkpoint_state"
    optimizer_state_error: str | None = None
    checkpoint_optimizer_state = checkpoint.get("optimizer_state_dict")
    if checkpoint_optimizer_state is None:
        checkpoint_optimizer_state = checkpoint.get("optimizer_state")
    if checkpoint_optimizer_state is not None:
        try:
            _validate_optimizer_resume_compatibility(
                dict(checkpoint),
                optimizer_inventory,
            )
            optimizer.load_state_dict(checkpoint_optimizer_state)
            optimizer_state_policy = "checkpoint_restored"
        except Exception as exc:  # noqa: BLE001 - preserve a cold fallback as explicit evidence
            optimizer_state_policy = "cold_incompatible_checkpoint_state"
            optimizer_state_error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    model.train()
    warmup_metrics: list[dict[str, float]] = []
    with _routing_variant(
        model,
        variant,
        pair_tile_size=pair_tile_size,
        dense_pair_tile_size=dense_pair_tile_size,
    ):
        for _ in range(int(warmups)):
            _synchronize(device)
            warmup_metrics.append(_run_step(model, loader, device, checkpoint, optimizer=optimizer, training_schedule=schedule))
            _synchronize(device)

        samples: list[dict[str, Any]] = []
        for repetition in range(int(repetitions)):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            _synchronize(device)
            started = time.perf_counter()
            error: str | None = None
            metrics: dict[str, float] = {}
            try:
                metrics = _run_step(
                    model,
                    loader,
                    device,
                    checkpoint,
                    optimizer=optimizer,
                    training_schedule=schedule,
                )
            except torch.cuda.OutOfMemoryError as exc:
                error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            finally:
                _synchronize(device)
            elapsed = time.perf_counter() - started
            sample: dict[str, Any] = {
                "repetition": repetition + 1,
                "elapsed_seconds": float(elapsed),
                "metrics": metrics,
                "status": "oom" if error is not None else "complete",
                "error": error,
                "peak_allocated_mib": (
                    float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
                    if device.type == "cuda"
                    else None
                ),
                "peak_reserved_mib": (
                    float(torch.cuda.max_memory_reserved(device) / (1024.0**2))
                    if device.type == "cuda"
                    else None
                ),
            }
            samples.append(sample)
            if error is not None:
                break

        # Count route eligibility in one untimed validation step.  Timing has
        # no callbacks or per-forward synchronization; this separate pass
        # supplies the evidence needed to interpret compiler+denseQE.
        counter_payload: dict[str, Any] = {
            "status": "not_applicable",
            "environment_read_calls": 0,
            "full_support_read_calls": 0,
            "complete_support_read_calls": 0,
            "whole_batch_complete_read_calls": 0,
            "hybrid_read_calls": 0,
            "packed_only_read_calls": 0,
            "complete_support_batch_rows": 0,
            "packed_fallback_batch_rows": 0,
            "complete_tile_calls": 0,
            "packed_score_tile_calls": 0,
        }
        try:
            model.eval()
            with _RouteCounter(model) as counter:
                _run_step(model, loader, device, checkpoint, optimizer=None, training_schedule=schedule)
            counter_payload = {
                "status": "complete",
                "environment_read_calls": int(counter.environment_read_calls),
                "full_support_read_calls": int(counter.full_support_read_calls),
                "complete_support_read_calls": int(counter.complete_support_read_calls),
                "whole_batch_complete_read_calls": int(counter.whole_batch_complete_read_calls),
                "hybrid_read_calls": int(counter.hybrid_read_calls),
                "packed_only_read_calls": int(counter.packed_only_read_calls),
                "complete_support_batch_rows": int(counter.complete_support_batch_rows),
                "packed_fallback_batch_rows": int(counter.packed_fallback_batch_rows),
                "complete_tile_calls": int(counter.complete_tile_calls),
                "packed_score_tile_calls": int(counter.packed_score_tile_calls),
            }
        except torch.cuda.OutOfMemoryError as exc:
            counter_payload.update(status="oom", error=f"{type(exc).__name__}: {str(exc).splitlines()[0]}")
        finally:
            model.train()

    elapsed_values = [
        float(sample["elapsed_seconds"])
        for sample in samples
        if sample.get("status") == "complete"
    ]
    all_complete = bool(samples) and all(sample.get("status") == "complete" for sample in samples)
    summary: dict[str, Any] = {
        "status": "complete" if all_complete else (
            "oom" if any(sample.get("status") == "oom" for sample in samples) else "error"
        ),
        "compiler_policy": _compiler_policy(variant, pair_tile_size=pair_tile_size),
        "warmups": int(warmups),
        "repetitions": int(repetitions),
        "warmup_metrics": warmup_metrics,
        "samples": samples,
        "optimizer_state_policy": optimizer_state_policy,
        "optimizer_state_error": optimizer_state_error,
        "counter_pass": counter_payload,
        "latency_seconds": {
            "min": min(elapsed_values) if elapsed_values else None,
            "median": float(np.median(elapsed_values)) if elapsed_values else None,
            "max": max(elapsed_values) if elapsed_values else None,
        },
        "peak_allocated_mib": max(
            (float(sample["peak_allocated_mib"]) for sample in samples if sample["peak_allocated_mib"] is not None),
            default=None,
        ),
        "peak_reserved_mib": max(
            (float(sample["peak_reserved_mib"]) for sample in samples if sample["peak_reserved_mib"] is not None),
            default=None,
        ),
    }
    del optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def _architecture(model: Any) -> str:
    return str(
        getattr(
            getattr(getattr(model, "config", None), "core_honf", None),
            "forward_architecture",
            "unknown",
        )
    )


def _variant_list(model: torch.nn.Module, requested: Sequence[str]) -> tuple[str, ...]:
    architecture = _architecture(model)
    if architecture == "routed_pairwise_honf":
        return tuple(requested)
    # Dense/legacy 1401 and 1804 checkpoints remain matched model baselines;
    # route compiler switches cannot affect their execution graph.
    return ("reference",)


def _run_records(
    specs: Sequence[CheckpointSpec],
    datasets: Sequence[Any],
    buckets: Sequence[Bucket],
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec, dataset in zip(specs, datasets, strict=True):
        checkpoint_payload = None
        for bucket in buckets:
            # A fresh model and optimizer per bucket/variant prevents the
            # earlier bucket's optimizer state from biasing a later timing.
            model, checkpoint = _load_model_spec(spec, device)
            if checkpoint_payload is None:
                checkpoint_payload = {
                    "label": spec.label,
                    "path": str(spec.path),
                    "epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))),
                    "architecture": _architecture(model),
                }
            actual_variants = _variant_list(model, args.variant)
            del model, checkpoint
            for variant in actual_variants:
                model, checkpoint = _load_model_spec(spec, device)
                loader = _build_loader(dataset, checkpoint, bucket, batch_size=int(args.batch_size))
                shape = _validate_batch_shape(loader, device, int(args.query_points))
                try:
                    measured = _measure_variant(
                        model,
                        loader,
                        device,
                        checkpoint,
                        variant=variant,
                        warmups=int(args.warmup),
                        repetitions=int(args.repetitions),
                        pair_tile_size=int(args.pair_tile_size),
                        dense_pair_tile_size=int(args.dense_pair_tile_size),
                    )
                    status = measured.get("status", "error")
                    error = None
                except torch.cuda.OutOfMemoryError as exc:
                    status = "oom"
                    error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
                    measured = {"status": status, "error": error}
                    torch.cuda.empty_cache() if device.type == "cuda" else None
                except Exception as exc:  # noqa: BLE001 - preserve per-variant evidence
                    status = "error"
                    error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
                    measured = {"status": status, "error": error}
                records.append(
                    {
                        "checkpoint": checkpoint_payload,
                        "bucket": {
                            "label": bucket.label,
                            "module_count": bucket.module_count,
                            "case_ids": list(bucket.case_ids),
                        },
                        "shape": shape,
                        "variant": variant,
                        "status": status,
                        "error": error,
                        "measurement": measured,
                    }
                )
                del loader, model, checkpoint
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    return records


def _matched_run_ids(specs: Sequence[CheckpointSpec]) -> dict[str, Any]:
    labels = [spec.label for spec in specs]
    matched = {
        run_id: [label for label in labels if re.search(rf"(?<!\d){re.escape(run_id)}(?!\d)", label)]
        for run_id in EXPECTED_RUN_IDS
    }
    return {
        "expected": list(EXPECTED_RUN_IDS),
        "matched": matched,
        "missing": [run_id for run_id, values in matched.items() if not values],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", required=True, help="e.g. cuda:0; GPU visibility is caller-selected.")
    parser.add_argument("--dataset", default=None, help="Optional packed HDF5 override.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--query-points", type=int, default=1024)
    parser.add_argument("--bucket", choices=("min", "max", "both"), default="both")
    parser.add_argument("--variant", action="append", choices=VARIANTS, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--pair-tile-size", type=int, default=16384)
    parser.add_argument("--dense-pair-tile-size", type=int, default=262144)
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Permit a planning/smoke run without all four 1401/1804/2000/2100 labels.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate arguments and checkpoint labels without loading models or launching GPU work.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.variant = tuple(args.variant or VARIANTS)
    if args.batch_size <= 0 or args.query_points <= 0:
        raise ValueError("--batch-size and --query-points must be positive.")
    if args.warmup < 0 or args.repetitions <= 0:
        raise ValueError("--warmup must be nonnegative and --repetitions must be positive.")
    if args.pair_tile_size <= 0 or args.dense_pair_tile_size <= 0:
        raise ValueError("Pair tile sizes must be positive.")
    specs = parse_checkpoint_specs(args.checkpoint)
    matching = _matched_run_ids(specs)
    if matching["missing"] and not args.allow_subset:
        raise ValueError(
            "The physical benchmark requires labels containing all four matched runs "
            f"{EXPECTED_RUN_IDS}; missing {matching['missing']}. Use --allow-subset only for a bounded smoke/planning run."
        )
    if args.plan_only:
        write_json(
            args.output,
            {
                "schema_version": 1,
                "task": "routing_optimization_benchmark",
                "status": "plan_only",
                "checkpoints": [
                    {"label": spec.label, "path": str(spec.path)} for spec in specs
                ],
                "matched_run_ids": matching,
                "requested": {
                    "device": str(args.device),
                    "batch_size": int(args.batch_size),
                    "query_points": int(args.query_points),
                    "variants": list(args.variant),
                    "compiler_policies": {
                        variant: _compiler_policy(
                            variant,
                            pair_tile_size=int(args.pair_tile_size),
                        )
                        for variant in args.variant
                    },
                },
            },
        )
        return 0

    device = select_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose an available physical GPU.")
    loader_args = _dataset_args(args)
    datasets: list[Any] = []
    dataset_paths: list[str] = []
    try:
        for spec in specs:
            checkpoint = load_trusted_checkpoint(spec.path, map_location="cpu")
            dataset, dataset_path = _load_dataset(
                checkpoint,
                loader_args,
                points_per_case_override=int(args.query_points),
                random_point_sampling_override=False,
            )
            # The shared Stage-3 loader requests grid fields for evaluation
            # studies. Training batches do not consume them; drop that extra
            # payload before the first real batch is collated.
            dataset.include_grid = False
            datasets.append(dataset)
            dataset_paths.append(str(dataset_path))
        buckets = _make_buckets(
            datasets,
            batch_size=int(args.batch_size),
            requested=str(args.bucket),
        )
        records = _run_records(specs, datasets, buckets, device, args)
    finally:
        for dataset in datasets:
            dataset.close()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_records_complete = bool(records) and all(
        record.get("status") == "complete" for record in records
    )
    overall_status = "complete" if all_records_complete else "incomplete"
    write_json(
        args.output,
        {
            "schema_version": 1,
            "task": "routing_optimization_benchmark",
            "status": overall_status,
            "matched_run_ids": matching,
            "dataset_paths": dataset_paths,
            "split": str(args.split),
            "protocol": {
                "batch_size": int(args.batch_size),
                "query_points": int(args.query_points),
                "bucket_policy": str(args.bucket),
                "variants": list(args.variant),
                "warmups": int(args.warmup),
                "repetitions": int(args.repetitions),
                "pair_tile_size": int(args.pair_tile_size),
                "dense_pair_tile_size": int(args.dense_pair_tile_size),
                "batched_scalar_tile_size": BATCHED_SCALAR_TILE_SIZE,
                "batched_use_checkpoint": False,
                "compiler_policies": {
                    variant: _compiler_policy(
                        variant,
                        pair_tile_size=int(args.pair_tile_size),
                    )
                    for variant in args.variant
                },
                "timed_step": "canonical run_epoch one-batch forward+backward+optimizer.step",
                "synchronization": "one CUDA synchronize before and after each measured wall-clock step",
                "timed_hooks": False,
                "route_counter_pass": "one untimed forward after timing; no synchronization hook per forward",
                "optimizer_state": "checkpoint state restored when compatible; explicit cold fallback is recorded",
            },
            "records": records,
            "limitations": [
                "This is a bounded one-batch latency/memory benchmark, not a training run or convergence claim.",
                "Case buckets are selected by common real case IDs and module count; repeated IDs are used only when a bucket has fewer than B cases.",
                "Peak allocated and reserved values are reported only on CUDA; CPU records keep these fields null.",
                "1401/1804 or other non-routed architectures are recorded as reference baselines because route switches cannot affect them.",
                "The untimed route counter pass reports whole-batch and hybrid complete-support rows plus packed fallback tile calls; it is excluded from latency statistics.",
            ],
        },
    )
    return 0 if all_records_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())

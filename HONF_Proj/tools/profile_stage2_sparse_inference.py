#!/usr/bin/env python3
"""Measure synchronized sparse-interface preparation and prepared decoding."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from channelthermal.data.collation import ChannelThermalBatchCollator
from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
from channelthermal.evaluation.loading import load_model, make_batch
from channelthermal.interface_field_coupling import _port_coordinates
from channelthermal.training.epoch import (
    effective_local_loss_weights,
    effective_port_condition_settings,
    predicted_consistency_weight_for_epoch,
    run_epoch,
)
from channelthermal.training.optimizer import build_forward_optimizer
from torch.utils.data import DataLoader

from honf_forward_core.config import BatchData
from honf_runtime.compat import make_grad_scaler, recursive_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        "--checkpoint-path",
        dest="checkpoints",
        type=Path,
        action="append",
        required=True,
        help="Checkpoint path; repeat for a matched multi-checkpoint profile.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Label corresponding to each --checkpoint; repeat in the same order.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument(
        "--prepared-query-count",
        type=int,
        default=8192,
        help="Prepared-decode query count (default: 8192).",
    )
    parser.add_argument(
        "--receiver-chunk-size",
        type=int,
        action="append",
        default=None,
        help=(
            "Runtime receiver chunk size for interface-field inference. Repeat to compare sizes; "
            "omitting it uses the checkpoint-configured value (normally 128)."
        ),
    )
    parser.add_argument(
        "--routing-mode",
        choices=("summary", "detailed"),
        action="append",
        default=None,
        help=(
            "Inference diagnostics mode: summary keeps cheap per-receiver scalars, while detailed "
            "also returns routing maps. Repeat to measure both modes."
        ),
    )
    parser.add_argument(
        "--training-step",
        action="store_true",
        help=(
            "Measure one nonpersistent canonical training step on an actual packed "
            "48-case x 1024-query batch."
        ),
    )
    return parser.parse_args()


@contextmanager
def _runtime_receiver_chunk_size(model: Any, chunk_size: int | None):
    """Temporarily override only the execution chunk, without changing config."""

    core = getattr(model, "core", None)
    if chunk_size is None or not hasattr(core, "receiver_chunk_size"):
        yield
        return
    previous = int(core.receiver_chunk_size)
    core.receiver_chunk_size = int(chunk_size)
    try:
        yield
    finally:
        core.receiver_chunk_size = previous


def measure(
    function: Callable[[], Any],
    device: torch.device,
    *,
    warmups: int,
    repetitions: int,
    inference: bool = True,
) -> dict[str, Any]:
    context_factory = torch.inference_mode if inference else nullcontext
    for _ in range(warmups):
        with context_factory():
            value = function()
        del value
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    baseline_reserved = int(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)
    samples_ms: list[float] = []
    for _ in range(repetitions):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with context_factory():
            value = function()
        torch.cuda.synchronize(device)
        samples_ms.append((time.perf_counter() - started) * 1000.0)
        del value
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        "median_ms": float(statistics.median(samples_ms)),
        "mean_ms": float(statistics.mean(samples_ms)),
        "p05_ms": float(np.quantile(samples_ms, 0.05)),
        "p95_ms": float(np.quantile(samples_ms, 0.95)),
        "samples_ms": samples_ms,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_allocated_bytes": max(peak_allocated - baseline_allocated, 0),
        "incremental_peak_reserved_bytes": max(peak_reserved - baseline_reserved, 0),
    }


_OUTPUT_AGREEMENT_KEYS = (
    ("pred_field", "field"),
    ("pred_port_condition", "port"),
    ("pred_interface", "interface"),
    ("pred_internal_temperature", "internal"),
)


def _output_agreement(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Compare public checkpoint outputs without retaining diagnostic maps."""

    agreement: dict[str, Any] = {}
    for key, label in _OUTPUT_AGREEMENT_KEYS:
        reference_value = reference.get(key)
        candidate_value = candidate.get(key)
        entry: dict[str, Any] = {
            "rtol": float(rtol),
            "atol": float(atol),
            "available": False,
            "allclose": None,
            "max_abs": None,
        }
        if not torch.is_tensor(reference_value) or not torch.is_tensor(candidate_value):
            agreement[label] = entry
            continue
        entry["available"] = True
        entry["reference_shape"] = list(reference_value.shape)
        entry["candidate_shape"] = list(candidate_value.shape)
        if reference_value.shape != candidate_value.shape:
            entry["allclose"] = False
            agreement[label] = entry
            continue
        reference_cpu = reference_value.detach().float().cpu()
        candidate_cpu = candidate_value.detach().float().cpu()
        difference = (candidate_cpu - reference_cpu).abs()
        entry["max_abs"] = float(difference.max().cpu()) if difference.numel() else 0.0
        reference_norm = torch.linalg.vector_norm(reference_cpu)
        entry["relative_l2"] = float(
            (torch.linalg.vector_norm(difference) / reference_norm.clamp_min(1.0e-12)).cpu()
        )
        entry["allclose"] = bool(
            torch.allclose(
                reference_cpu,
                candidate_cpu,
                rtol=float(rtol),
                atol=float(atol),
            )
        )
        agreement[label] = entry
    return agreement


def _public_output_snapshot(outputs: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Retain only public output tensors on CPU for numerical comparisons."""

    return {
        key: outputs[key].detach().cpu()
        for key, _ in _OUTPUT_AGREEMENT_KEYS
        if torch.is_tensor(outputs.get(key))
    }


def _agreement_tolerances(routing_mode: str, receiver_chunk_size: int) -> tuple[float, float]:
    """Use strict map-only tolerance when the execution chunk is unchanged."""

    if routing_mode == "detailed" and int(receiver_chunk_size) == 128:
        return 2.0e-6, 2.0e-7
    return 2.0e-5, 2.0e-6


def _checkpoint_labels(checkpoints: list[Path], labels: list[str]) -> list[str]:
    """Resolve explicit labels while preserving the historical one-checkpoint CLI."""

    if labels and len(labels) != len(checkpoints):
        raise ValueError(
            "Each --checkpoint must have one matching --label; "
            f"received {len(checkpoints)} checkpoints and {len(labels)} labels."
        )
    if labels:
        return [str(label) for label in labels]
    return [path.stem for path in checkpoints]


def _new_family_encoding_and_layout(
    model: Any,
    batch: dict[str, Any],
) -> tuple[Any, Any]:
    """Run only adapter/environment encoding and sparse layout construction.

    This follows the maintained interface-field path in
    ``channelthermal.interface_field_coupling.forward_interface_field``.  It is
    intentionally a profiler hook rather than a new model API.  Legacy HONF
    has no matched ``encode_case``/layout phase and is not sent through this
    helper.
    """

    architecture = str(model.config.core_honf.forward_architecture)
    if architecture == "legacy_honf":
        raise ValueError("The new-family encoding/layout phase is not defined for legacy_honf.")
    structure = batch["structure"]
    query_xy = batch["query_xy"][:, :1]
    device, dtype = query_xy.device, query_xy.dtype
    batch_size = int(query_xy.shape[0])
    re = structure.get("re")
    u_in = structure.get("u_in")
    module_centers = structure["module_centers"]
    heat_powers = structure["heat_powers"]
    module_present = structure["module_present"]
    material_params = structure.get("material_params")
    domain_length_x = structure.get("domain_length_x")
    domain_length_y = structure.get("domain_length_y")
    re = query_xy.new_zeros(batch_size, 1) if re is None else re
    u_in = query_xy.new_zeros(batch_size, 1) if u_in is None else u_in
    if material_params is None:
        material_params = query_xy.new_zeros(batch_size, int(model.config.channelthermal.material_param_dim))
    adapter = model.input_adapter(
        re=re.to(device=device, dtype=dtype),
        u_in=u_in.to(device=device, dtype=dtype),
        module_centers=module_centers.to(device=device, dtype=dtype),
        heat_powers=heat_powers.to(device=device, dtype=dtype)
        * float(model.config.channelthermal.heat_scale),
        module_present=module_present.to(device=device, dtype=dtype),
        material_params=material_params.to(device=device, dtype=dtype),
        domain_length_x=None
        if domain_length_x is None
        else domain_length_x.to(device=device, dtype=dtype),
        domain_length_y=None
        if domain_length_y is None
        else domain_length_y.to(device=device, dtype=dtype),
    )
    env = model.environment_builder(
        batch_size=batch_size,
        num_env_tokens_x=int(model.config.core_honf.num_env_tokens_x),
        num_env_tokens_y=int(model.config.core_honf.num_env_tokens_y),
        domain_length_x=float(model.config.core_honf.domain_length_x),
        domain_length_y=float(model.config.core_honf.domain_length_y),
        device=device,
        dtype=dtype,
    )
    encoded = model.core.encode_case(
        BatchData(
            module_centers=adapter.module_centers,
            module_present=adapter.module_present,
            module_features=adapter.module_features,
            global_context=adapter.global_context,
            query_xy=query_xy.float(),
            query_time=None,
            target_field=None,
            case_name="channelthermal",
            metadata={},
            env_coords=env.env_coords,
            env_features=env.env_features,
        )
    )
    interface_condition = batch.get("interface_condition")
    teacher_port_tokens = batch.get("teacher_port_tokens")
    ntheta = model._infer_ntheta(interface_condition, teacher_port_tokens)
    physical_port_xy = _port_coordinates(model, adapter.module_centers, ntheta)
    layout_cache = model.core.build_layout(encoded, physical_port_xy)
    return encoded, layout_cache


def _training_step_measurement(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    device: torch.device,
    *,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    """Measure one canonical, nonpersistent 48x1024 training step.

    The step reuses ``training.epoch.run_epoch`` so the forward, loss terms,
    AMP policy, clipping, and AdamW construction are the maintained training
    path rather than a profiler approximation.  The loaded model/optimizer
    are private to this function and are discarded in ``finally``.
    """

    dataset = None
    train_model = None
    optimizer = None
    try:
        train_config = checkpoint.get("train_config", {})
        if not isinstance(train_config, dict):
            raise RuntimeError("checkpoint.train_config is missing or is not a mapping")
        dataset_config = train_config.get("dataset", {})
        training_config = train_config.get("training", {})
        loss_config = train_config.get("loss", {})
        if not isinstance(dataset_config, dict) or not isinstance(training_config, dict):
            raise RuntimeError("checkpoint train_config.dataset/training must be mappings")
        if not isinstance(loss_config, dict):
            loss_config = {}
        statistics_payload = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in checkpoint.get("global_normalization_stats", {}).items()
        }
        dataset_path = dataset_config.get(
            "packed_h5_path",
            "./Case_ThermalChannel/Dataset/links/thermal_channel_global_v1.h5",
        )
        dataset = GlobalChannelThermalDataset(
            dataset_path,
            split=dataset_config.get("train_split", "train"),
            points_per_case=1024,
            normalize_inputs=bool(dataset_config.get("normalize_inputs", False)),
            normalize_targets=bool(dataset_config.get("normalize_targets", False)),
            random_point_sampling=False,
            seed=int(training_config.get("seed", 42)),
            normalizer=H5Normalizer(statistics_payload) if statistics_payload else None,
        )
        if len(dataset) < 48:
            raise RuntimeError(
                "packed training split contains fewer than 48 cases "
                f"({len(dataset)} available)"
            )
        collator = ChannelThermalBatchCollator(
            dynamic_module_padding=bool(dataset_config.get("dynamic_module_padding", True)),
            max_modules_per_batch=(
                None
                if dataset_config.get("max_modules_per_batch") is None
                else int(dataset_config["max_modules_per_batch"])
            ),
        )
        loader = DataLoader(
            dataset,
            batch_size=48,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
            collate_fn=collator,
        )
        batch = next(iter(loader))
        query_count = int(batch["query_xy"].shape[-2])
        if query_count != 1024:
            raise RuntimeError(
                "packed training batch did not preserve 1024 queries per case "
                f"(got {query_count})"
            )
        batch = recursive_to_device(batch, device)

        # A second strict load prevents the timed optimizer step from mutating
        # the inference model used by the other phases.  This never enters the
        # run allocator/resume path.  When present, the checkpoint optimizer
        # state is loaded directly into this disposable optimizer so the
        # measured step matches its recorded AdamW moments.
        train_model, _ = load_model(checkpoint_path, device)
        optimizer, _ = build_forward_optimizer(train_model, training_config)
        optimizer_state_loaded = False
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if isinstance(optimizer_state, dict) and optimizer_state:
            optimizer.load_state_dict(optimizer_state)
            optimizer_state_loaded = True
        scaler = make_grad_scaler(device, bool(training_config.get("amp", False)))
        checkpoint_epoch = checkpoint.get(
            "epoch",
            checkpoint.get("current_epoch", training_config.get("epochs", 1)),
        )
        epoch = int(checkpoint_epoch if checkpoint_epoch is not None else 1)
        mode, ratio = effective_port_condition_settings(epoch, training_config)
        effective_internal, effective_interface = effective_local_loss_weights(
            loss_config,
            mode,
            ratio,
        )
        predicted_weight = predicted_consistency_weight_for_epoch(epoch, loss_config)
        gradient_clip_norm = float(training_config.get("gradient_clip_norm", 0.0) or 0.0)

        def training_step() -> Any:
            return run_epoch(
                train_model,
                [batch],
                device,
                loss_config,
                optimizer=optimizer,
                scaler=scaler,
                amp=bool(training_config.get("amp", False)),
                max_batches=1,
                local_port_condition_mode=mode,
                mixed_teacher_ratio=ratio,
                effective_internal_temperature_weight=effective_internal,
                effective_interface_weight=effective_interface,
                predicted_consistency_weight=predicted_weight,
                gradient_clip_norm=gradient_clip_norm,
                record_gradient_diagnostics=False,
            )

        result = measure(
            training_step,
            device,
            warmups=warmups,
            repetitions=repetitions,
            inference=False,
        )
        result.update(
            {
                "status": "ok",
                "training_batch_size": 48,
                "training_points_per_case": 1024,
                "training_query_count": query_count,
                "training_mode": mode,
                "training_mixed_teacher_ratio": float(ratio),
                "training_amp": bool(training_config.get("amp", False)),
                "training_gradient_clip_norm": gradient_clip_norm,
                "optimizer_state_loaded": optimizer_state_loaded,
            }
        )
        return result
    except Exception as exc:  # noqa: BLE001 - preserve the exact safe blocker in the artifact.
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "status": "blocked",
            "blocker": f"{type(exc).__name__}: {exc}",
            "training_batch_size": 48,
            "training_points_per_case": 1024,
            "optimizer_state_loaded": False,
        }
    finally:
        if dataset is not None:
            dataset.close()
        optimizer = None
        train_model = None


def main() -> int:
    args = parse_args()
    if args.warmups < 0 or args.repetitions < 1:
        raise ValueError("--warmups must be nonnegative and --repetitions must be positive.")
    if args.prepared_query_count < 1:
        raise ValueError("--prepared-query-count must be positive.")
    receiver_chunk_sizes = [None] if args.receiver_chunk_size is None else [int(value) for value in args.receiver_chunk_size]
    if any(value is not None and value <= 0 for value in receiver_chunk_sizes):
        raise ValueError("--receiver-chunk-size values must be positive.")
    routing_modes = ["summary"] if args.routing_mode is None else [str(value) for value in args.routing_mode]
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This utility records synchronized CUDA timings; select a CUDA device.")

    checkpoint_paths = [path.resolve() for path in args.checkpoints]
    labels = _checkpoint_labels(checkpoint_paths, list(args.label))
    case_ids = args.case_id or ["0273", "0653"]
    rows: list[dict[str, Any]] = []
    for checkpoint_path, label in zip(checkpoint_paths, labels, strict=True):
        model, checkpoint = load_model(checkpoint_path, device)
        model.eval()
        dataset = None
        try:
            dataset_config = checkpoint.get("train_config", {}).get("dataset", {})
            if not isinstance(dataset_config, dict):
                raise RuntimeError("checkpoint.train_config.dataset is missing or is not a mapping")
            statistics_payload = {
                key: np.asarray(value, dtype=np.float32)
                for key, value in checkpoint.get("global_normalization_stats", {}).items()
            }
            dataset = GlobalChannelThermalDataset(
                dataset_config.get(
                    "packed_h5_path",
                    "./Case_ThermalChannel/Dataset/links/thermal_channel_global_v1.h5",
                ),
                split="test",
                points_per_case=1,
                normalize_inputs=bool(dataset_config.get("normalize_inputs", False)),
                normalize_targets=bool(dataset_config.get("normalize_targets", False)),
                random_point_sampling=False,
                include_grid=True,
                normalizer=H5Normalizer(statistics_payload) if statistics_payload else None,
            )
            case_indices = {
                str(case_id): index for index, case_id in enumerate(dataset.selected_case_ids)
            }
            missing = [case_id for case_id in case_ids if str(case_id) not in case_indices]
            if missing:
                raise KeyError(f"Cases are absent from the test split: {missing}")

            architecture = str(model.config.core_honf.forward_architecture)
            for case_id in case_ids:
                sample = dataset[case_indices[str(case_id)]]
                query_xy = np.stack(
                    (sample["x_grid"].reshape(-1), sample["y_grid"].reshape(-1)), axis=-1
                ).astype(np.float32)
                batch = make_batch(sample, query_xy, device)
                forward_kwargs = {
                    "interface_condition": batch.get("interface_condition"),
                    "local_module_params": batch.get("local_module_params"),
                    "teacher_port_tokens": batch.get("teacher_port_tokens"),
                    "local_query_points": batch.get("module_internal_query_points"),
                    "local_port_condition_mode": "predicted",
                    "mixed_teacher_ratio": 0.0,
                }

                # Keep one untimed chunk-128 summary output as the numerical
                # reference.  Each measured full-forward mode below gets one
                # corresponding untimed output comparison; timing calls remain
                # unchanged and never retain the comparison tensors.
                with torch.inference_mode(), _runtime_receiver_chunk_size(model, 128):
                    agreement_reference_output = model(
                        batch["structure"],
                        batch["query_xy"],
                        return_routing_maps=False,
                        **forward_kwargs,
                    )
                agreement_reference = _public_output_snapshot(agreement_reference_output)
                del agreement_reference_output

                for routing_mode in routing_modes:
                    request_routing_maps = routing_mode == "detailed"
                    for receiver_chunk_size in receiver_chunk_sizes:
                        def configured_full_forward(
                            model: torch.nn.Module = model,
                            batch: dict[str, Any] = batch,
                            forward_kwargs: dict[str, Any] = forward_kwargs,
                            request_routing_maps: bool = request_routing_maps,
                            receiver_chunk_size: int | None = receiver_chunk_size,
                        ) -> Any:
                            with _runtime_receiver_chunk_size(model, receiver_chunk_size):
                                return model(
                                    batch["structure"],
                                    batch["query_xy"],
                                    return_routing_maps=request_routing_maps,
                                    **forward_kwargs,
                                )

                        def configured_prepare_with_one_query(
                            model: torch.nn.Module = model,
                            batch: dict[str, Any] = batch,
                            forward_kwargs: dict[str, Any] = forward_kwargs,
                            request_routing_maps: bool = request_routing_maps,
                            receiver_chunk_size: int | None = receiver_chunk_size,
                        ) -> Any:
                            with _runtime_receiver_chunk_size(model, receiver_chunk_size):
                                return model(
                                    batch["structure"],
                                    batch["query_xy"][:, :1],
                                    return_prepared_state=True,
                                    return_routing_maps=request_routing_maps,
                                    **forward_kwargs,
                                )

                        effective_receiver_chunk_size = (
                            int(model.core.receiver_chunk_size)
                            if receiver_chunk_size is None
                            else int(receiver_chunk_size)
                        )
                        with torch.inference_mode():
                            agreement_candidate = configured_full_forward()
                        agreement_rtol, agreement_atol = _agreement_tolerances(
                            routing_mode,
                            effective_receiver_chunk_size,
                        )
                        output_agreement = _output_agreement(
                            agreement_reference,
                            agreement_candidate,
                            rtol=agreement_rtol,
                            atol=agreement_atol,
                        )
                        del agreement_candidate

                        phase_specs = (
                            ("full_forward", int(query_xy.shape[0]), configured_full_forward),
                            ("physical_preparation_plus_one_query", 1, configured_prepare_with_one_query),
                        )
                        for phase, query_count, function in phase_specs:
                            result = measure(
                                function,
                                device,
                                warmups=int(args.warmups),
                                repetitions=int(args.repetitions),
                            )
                            agreement_fields = (
                                {"output_agreement": output_agreement}
                                if phase == "full_forward"
                                else {}
                            )
                            rows.append(
                                {
                                    "label": label,
                                    "checkpoint": str(checkpoint_path),
                                    "architecture": architecture,
                                    "case_id": str(case_id),
                                    "phase": phase,
                                    "query_count": query_count,
                                    "receiver_chunk_size": (
                                        int(model.core.receiver_chunk_size)
                                        if receiver_chunk_size is None and hasattr(model.core, "receiver_chunk_size")
                                        else receiver_chunk_size
                                    ),
                                    "routing_mode": routing_mode,
                                    "status": "ok",
                                    **agreement_fields,
                                    **result,
                                }
                            )

                if architecture != "legacy_honf":
                    result = measure(
                        lambda model=model, batch=batch: _new_family_encoding_and_layout(model, batch),
                        device,
                        warmups=int(args.warmups),
                        repetitions=int(args.repetitions),
                    )
                    rows.append(
                        {
                            "label": label,
                            "checkpoint": str(checkpoint_path),
                            "architecture": architecture,
                            "case_id": str(case_id),
                            "phase": "encoding_plus_layout_construction",
                            "query_count": 0,
                            "status": "ok",
                            **result,
                        }
                    )

                with torch.inference_mode():
                    prepared = model(
                        batch["structure"],
                        batch["query_xy"][:, :1],
                        return_prepared_state=True,
                        **forward_kwargs,
                    )["prepared_state"]
                prepared_query_count = min(int(args.prepared_query_count), int(query_xy.shape[0]))
                prepared_query = batch["query_xy"][:, :prepared_query_count]

                for routing_mode in routing_modes:
                    request_routing_maps = routing_mode == "detailed"
                    for receiver_chunk_size in receiver_chunk_sizes:
                        def prepared_decode(
                            model: torch.nn.Module = model,
                            prepared: Any = prepared,
                            prepared_query: torch.Tensor = prepared_query,
                            request_routing_maps: bool = request_routing_maps,
                            receiver_chunk_size: int | None = receiver_chunk_size,
                        ) -> Any:
                            return model.decode_prepared(
                                prepared,
                                prepared_query,
                                return_routing_maps=request_routing_maps,
                                receiver_chunk_size=receiver_chunk_size,
                            )

                        result = measure(
                            prepared_decode,
                            device,
                            warmups=int(args.warmups),
                            repetitions=int(args.repetitions),
                        )
                        rows.append(
                            {
                                "label": label,
                                "checkpoint": str(checkpoint_path),
                                "architecture": architecture,
                                "case_id": str(case_id),
                                "phase": "prepared_decode",
                                "query_count": prepared_query_count,
                                "receiver_chunk_size": (
                                    int(model.core.receiver_chunk_size)
                                    if receiver_chunk_size is None and hasattr(model.core, "receiver_chunk_size")
                                    else receiver_chunk_size
                                ),
                                "routing_mode": routing_mode,
                                "status": "ok",
                                **result,
                            }
                        )
                del prepared, batch
                del agreement_reference

            if args.training_step:
                training_result = _training_step_measurement(
                    checkpoint_path,
                    checkpoint,
                    device,
                    warmups=int(args.warmups),
                    repetitions=int(args.repetitions),
                )
                rows.append(
                    {
                        "label": label,
                        "checkpoint": str(checkpoint_path),
                        "architecture": architecture,
                        "case_id": "train_batch_48x1024",
                        "phase": "training_step",
                        "query_count": 1024,
                        **training_result,
                    }
                )
        finally:
            if dataset is not None:
                dataset.close()
            del model
            del checkpoint

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": str(device),
        "checkpoint": str(checkpoint_paths[0]) if len(checkpoint_paths) == 1 else [str(path) for path in checkpoint_paths],
        "checkpoints": [
            {"label": label, "path": str(path)}
            for label, path in zip(labels, checkpoint_paths, strict=True)
        ],
        "warmups": int(args.warmups),
        "repetitions": int(args.repetitions),
        "prepared_query_count": int(args.prepared_query_count),
        "receiver_chunk_sizes": [
            "configured" if value is None else int(value) for value in receiver_chunk_sizes
        ],
        "routing_modes": routing_modes,
        "output_agreement": {
            "reference": "untimed full_forward with receiver_chunk_size=128 and routing_mode=summary",
            "keys": [label for _, label in _OUTPUT_AGREEMENT_KEYS],
            "metrics": ["max_abs", "relative_l2", "allclose"],
            "tolerances": {
                "chunk_change_or_summary": {"rtol": 2.0e-5, "atol": 2.0e-6},
                "same_chunk_detailed": {"rtol": 2.0e-6, "atol": 2.0e-7},
            },
        },
        "training_step_requested": bool(args.training_step),
        "synchronized": True,
        "scope": (
            "full_forward, physical_preparation_plus_one_query, and prepared_decode "
            "are synchronized CUDA phases; prepared_decode uses the first configured "
            "8192 queries (or the full case when smaller). New-family models also "
            "report adapter/encoding plus real layout construction. Optional training_step "
            "uses the canonical run_epoch path on a nonpersistent packed 48x1024 batch. "
            "Routing modes and receiver chunk sizes are runtime execution controls; "
            "they do not rewrite checkpoint configuration."
        ),
        "rows": rows,
    }
    json_path = output_dir / "gpu0_phase_timing.json"
    csv_path = output_dir / "gpu0_phase_timing.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "label",
            "checkpoint",
            "architecture",
            "case_id",
            "phase",
            "query_count",
            "receiver_chunk_size",
            "routing_mode",
            "status",
            "blocker",
            "training_batch_size",
            "training_points_per_case",
            "training_query_count",
            "training_mode",
            "training_mixed_teacher_ratio",
            "training_amp",
            "training_gradient_clip_norm",
            "optimizer_state_loaded",
            "median_ms",
            "mean_ms",
            "p05_ms",
            "p95_ms",
            "baseline_allocated_bytes",
            "baseline_reserved_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "incremental_peak_allocated_bytes",
            "incremental_peak_reserved_bytes",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            [{key: value for key, value in row.items() if key != "samples_ms"} for row in rows]
        )
    print(json_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

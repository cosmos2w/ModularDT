"""Run the bounded CPU prelaunch checks for occupancy-adaptive Run 1409.

The command uses the maintained ThermalChannel dataset/collator and training
loss path, evaluates two real predicted-port batches, then performs exactly
one in-memory optimizer update.  It requires an explicit profile and output
path, never allocates a managed run, and never writes a checkpoint.  CUDA is
rejected deliberately: this check is intended to establish data flow and
gradient connectivity on a read-only CPU path before any managed launch.

Example::

    PYTHONPATH=src:Case_ThermalChannel/src \
      python tools/diagnostics/run_run1409_occupancy_prelaunch.py \
      --profile src/config_core/forward/occupancy_adaptive_group_control_honf_context.json \
      --output /path/to/Run_1409/evaluations/prelaunch_predicted_port_cpu.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _jsonable(value: Any) -> Any:
    import numpy as np
    import torch

    if torch.is_tensor(value):
        if value.numel() == 1:
            return _jsonable(value.detach().cpu().item())
        flattened = value.detach().float().cpu().reshape(-1)
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.numel()),
            "finite": bool(torch.isfinite(flattened).all()),
            "min": _jsonable(flattened.min()),
            "max": _jsonable(flattened.max()),
            "mean": _jsonable(flattened.mean()),
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(value) for key, value in metrics.items()}


def _router_update_stats(model: Any, before: Mapping[str, Any] | None = None) -> dict[str, Any]:
    import torch

    gradients: list[torch.Tensor] = []
    updates: list[torch.Tensor] = []
    names: list[str] = []
    prototype_gradient_rows: list[float] = []
    prototype_parameter_name: str | None = None
    for name, parameter in model.named_parameters():
        if ".backend.router." not in f".{name}.":
            continue
        names.append(name)
        if parameter.grad is not None:
            gradients.append(parameter.grad.detach().float().reshape(-1))
        if before is not None and name in before:
            updates.append((parameter.detach() - before[name]).float().reshape(-1))
        if name.endswith(("backend.router.group_codes", "router.group_codes")):
            prototype_parameter_name = name
            if parameter.grad is not None and parameter.grad.ndim >= 2:
                prototype_gradient_rows = parameter.grad.detach().float().reshape(
                    int(parameter.grad.shape[0]), -1
                ).double().norm(dim=1).cpu().tolist()
    gradient = torch.cat(gradients) if gradients else torch.empty(0)
    update = torch.cat(updates) if updates else torch.empty(0)
    occupied_rows = [index for index, value in enumerate(prototype_gradient_rows) if value > 1.0e-12]
    occupied_values = [prototype_gradient_rows[index] for index in occupied_rows]
    return {
        "parameter_count": len(names),
        "parameter_names": names,
        "gradient_norm": float(gradient.double().norm()) if gradient.numel() else 0.0,
        "update_norm": float(update.double().norm()) if update.numel() else 0.0,
        "finite_gradient": bool(torch.isfinite(gradient).all()) if gradient.numel() else True,
        "finite_update": bool(torch.isfinite(update).all()) if update.numel() else True,
        "nonzero_gradient": bool(torch.count_nonzero(gradient)) if gradient.numel() else False,
        "nonzero_update": bool(torch.count_nonzero(update)) if update.numel() else False,
        "group_codes_parameter_name": prototype_parameter_name,
        "group_codes_gradient_row_norms": prototype_gradient_rows,
        "group_codes_nonzero_gradient_rows": occupied_rows,
        "group_codes_occupied_row_gradient_norms_nonidentical": bool(
            len(occupied_values) > 1
            and max(occupied_values) - min(occupied_values) > 1.0e-12
        ),
    }


def _build_loader(cfg: Mapping[str, Any], dataset: Any, device: Any) -> Any:
    from channelthermal.data.collation import ChannelThermalBatchCollator, ModuleCountBucketBatchSampler
    from torch.utils.data import DataLoader

    dataset_cfg = cfg["dataset"]
    training_cfg = cfg["training"]
    collator = ChannelThermalBatchCollator(
        dynamic_module_padding=bool(dataset_cfg.get("dynamic_module_padding", True)),
        max_modules_per_batch=dataset_cfg.get("max_modules_per_batch"),
    )
    batch_size = int(dataset_cfg.get("batch_size", training_cfg.get("batch_size", 4)))
    if bool(dataset_cfg.get("bucket_by_module_count", True)):
        sampler = ModuleCountBucketBatchSampler(
            dataset.selected_module_counts,
            batch_size=batch_size,
            bucket_size_multiplier=int(dataset_cfg.get("module_count_bucket_size_multiplier", 8)),
            seed=int(training_cfg.get("seed", 42)),
        )
        return DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=False, collate_fn=collator)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collator)


def _preview_batch(loader: Any, dataset: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize two real batches through the ordinary collator."""

    iterator = iter(loader)
    batches: list[dict[str, Any]] = []
    for _ in range(2):
        try:
            batch = next(iterator)
        except StopIteration as exc:
            raise RuntimeError("dataset loader did not provide two real batches") from exc
        batches.append(batch)
    first = batches[0]
    return batches, {
        "batch_count": len(batches),
        "batch_sizes": [int(batch["query_xy"].shape[0]) for batch in batches],
        "query_counts": [int(batch["query_xy"].shape[1]) for batch in batches],
        "case_ids": [list(batch.get("case_id", ())) for batch in batches],
        "module_counts": [batch["structure"]["module_present"].sum(dim=-1).detach().cpu().tolist() for batch in batches],
        "dataset_split": str(dataset.split),
        "dataset_path": str(dataset.path),
        "dataset_epoch": int(dataset.epoch),
        "first_batch_device": str(first["query_xy"].device),
    }


def run_prelaunch(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.device) != "cpu":
        raise ValueError("group-control prelaunch is CPU-only; pass --device cpu")
    for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel" / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch
    from channelthermal.data.datasets import GlobalChannelThermalDataset
    from channelthermal.model import ChannelThermalHONFModel
    from channelthermal.training.epoch import (
        effective_local_loss_weights,
        effective_port_condition_settings,
        predicted_consistency_weight_for_epoch,
        run_epoch,
    )
    from channelthermal.training.optimizer import build_forward_optimizer
    from channelthermal.workflows.train_forward import build_model_config, resolve_auto_internal_mode

    from honf_runtime.case_protocol import WorkflowRequest
    from honf_runtime.compat import make_grad_scaler, set_seed
    from honf_runtime.config_loader import load_config_bundle
    from honf_runtime.registry import load_case_plugin

    profile = Path(args.profile).expanduser().resolve()
    if not profile.is_file():
        raise FileNotFoundError(profile)
    overlay = None if args.overlay is None else Path(args.overlay).expanduser().resolve()
    if overlay is not None and not overlay.is_file():
        raise FileNotFoundError(overlay)
    bundle = load_config_bundle(str(profile), experiment_overlay=None if overlay is None else str(overlay))
    plugin = load_case_plugin(str(bundle.case["plugin"]))
    plugin.validate_config(bundle)
    request = WorkflowRequest(workflow="forward", device="cpu", epochs=50)
    cfg = plugin._forward_config(bundle, request, Path(args.output).expanduser().resolve().parent)
    architecture = str(cfg["model"]["core_honf"].get("forward_architecture", ""))
    expected_architecture = str(
        getattr(args, "expected_architecture", "occupancy_adaptive_group_control_honf")
    )
    if architecture != expected_architecture:
        raise RuntimeError(
            f"resolved profile architecture is {architecture!r}, expected {expected_architecture!r}"
        )
    dataset_cfg = cfg["dataset"]
    training_cfg = cfg["training"]
    loss_cfg = dict(cfg["loss"])
    seed = int(training_cfg.get("seed", 42))
    set_seed(seed)
    dataset = GlobalChannelThermalDataset(
        dataset_cfg["packed_h5_path"],
        split=dataset_cfg.get("train_split", "train"),
        points_per_case=int(dataset_cfg.get("points_per_case", 4096)),
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=bool(dataset_cfg.get("random_point_sampling", True)),
        seed=seed,
        require_converged=bool(dataset_cfg.get("require_converged", False)),
    )
    dataset.set_epoch(1)
    model_config = build_model_config(cfg, dataset)
    # Keep the maintained constructor path so the configured local surrogate
    # checkpoint is attached before the predicted-port pass.  Passing
    # ``attach_local_from_checkpoint=False`` would make this prelaunch check
    # fail by construction even when the profile points at the trusted local
    # model artifact.
    model = ChannelThermalHONFModel(model_config).to(torch.device("cpu"))
    model.set_global_target_normalization(dataset.normalizer.stats, normalize_targets=bool(dataset_cfg.get("normalize_targets", False)))
    resolve_auto_internal_mode(model_config, model)
    if not model.local_surrogate_attached:
        raise RuntimeError("the maintained local-surrogate checkpoint was not attached")
    loader = _build_loader(cfg, dataset, torch.device("cpu"))
    batches, batch_summary = _preview_batch(loader, dataset)
    for batch in batches:
        if str(batch["query_xy"].device) != "cpu":
            raise RuntimeError("real predicted-port batch unexpectedly moved off CPU")
    optimizer, _optimizer_inventory = build_forward_optimizer(model, training_cfg)
    scaler = make_grad_scaler(torch.device("cpu"), False)
    mode, ratio = effective_port_condition_settings(1, training_cfg)
    if str(mode) != "predicted":
        raise RuntimeError(f"resolved training mode is {mode!r}, expected predicted")
    internal_weight, interface_weight = effective_local_loss_weights(loss_cfg, mode, ratio)
    predicted_weight = predicted_consistency_weight_for_epoch(1, loss_cfg)
    # Use the maintained epoch path for exactly two real read-only batches.
    # This pass has no optimizer and therefore cannot alter model parameters.
    if hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(1)
    eval_metrics = run_epoch(
        model,
        loader,
        torch.device("cpu"),
        loss_cfg,
        optimizer=None,
        scaler=None,
        amp=False,
        max_batches=2,
        local_port_condition_mode=str(mode),
        mixed_teacher_ratio=float(ratio),
        effective_internal_temperature_weight=float(internal_weight),
        effective_interface_weight=float(interface_weight),
        predicted_consistency_weight=float(predicted_weight),
        gradient_clip_norm=0.0,
    )
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    if hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(1)
    update_metrics = run_epoch(
        model,
        loader,
        torch.device("cpu"),
        loss_cfg,
        optimizer=optimizer,
        scaler=scaler,
        amp=False,
        max_batches=1,
        local_port_condition_mode=str(mode),
        mixed_teacher_ratio=float(ratio),
        effective_internal_temperature_weight=float(internal_weight),
        effective_interface_weight=float(interface_weight),
        predicted_consistency_weight=float(predicted_weight),
        gradient_clip_norm=float(training_cfg.get("gradient_clip_norm", 0.0) or 0.0),
    )
    stats = _router_update_stats(model, before)
    if not stats["finite_gradient"] or not stats["finite_update"]:
        raise RuntimeError(f"non-finite group-router gradient/update: {stats}")
    payload = {
        "schema_version": 1,
        "task": str(getattr(args, "task_name", "run1409_occupancy_adaptive_prelaunch")),
        "status": "complete",
        "profile": str(profile),
        "experiment_overlay": None if overlay is None else str(overlay),
        "architecture": architecture,
        "device": "cpu",
        "seed": seed,
        "mode": str(mode),
        "mixed_teacher_ratio": float(ratio),
        "effective_internal_temperature_weight": float(internal_weight),
        "effective_interface_weight": float(interface_weight),
        "predicted_consistency_weight": float(predicted_weight),
        "batch": batch_summary,
        "optimizer": {
            "name": optimizer.__class__.__name__,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "weight_decay": float(optimizer.param_groups[0].get("weight_decay", 0.0)),
            "steps": 1,
        },
        "two_batch_read_metrics": _scalar_metrics(eval_metrics),
        "one_update_metrics": _scalar_metrics(update_metrics),
        str(getattr(args, "router_stats_key", "occupancy_router_gradient_update")): stats,
        "checkpoint_written": False,
        "managed_run_allocated": False,
        "cuda_used": False,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--overlay", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    payload = run_prelaunch(build_parser().parse_args(argv))
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_prelaunch"]

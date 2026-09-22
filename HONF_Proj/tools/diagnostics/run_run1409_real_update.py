"""Run the historical v1 Run-1409 update without a managed run.

The script follows the maintained ThermalChannel forward workflow for config
composition, dataset construction, batch sampling, optimizer construction,
loss assembly, and ``run_epoch``.  It performs exactly one B=48, Q=1024
predicted-port AdamW step, then evaluates the same first training batch once
after the update.  No checkpoint or run directory is created.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = PROJECT_ROOT / "src/config_core/forward/budgeted_group_control_honf_context.json"
DEFAULT_OVERLAY = PROJECT_ROOT / "src/config_core/forward/experiments/run1409_case_group_budget_calibrated.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "diagnostics/generated/run1409_budgeted_group_control/real_update.json"
EXPECTED_WEIGHT = 0.005783974924700852


def _jsonable(value: Any) -> Any:
    """Convert tensors and non-finite scalars into bounded JSON values."""

    import torch

    if torch.is_tensor(value):
        if value.numel() == 1:
            return _jsonable(value.detach().cpu().item())
        flat = value.detach().float().cpu().reshape(-1)
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.numel()),
            "min": _jsonable(flat.min()),
            "max": _jsonable(flat.max()),
            "mean": _jsonable(flat.mean()),
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(value) for key, value in metrics.items()}


def _gate_stats(model: Any, before: Mapping[str, Any] | None = None) -> dict[str, Any]:
    import torch

    gradients: list[torch.Tensor] = []
    updates: list[torch.Tensor] = []
    parameter_names: list[str] = []
    finite_gradients = True
    finite_updates = True
    for name, parameter in model.named_parameters():
        if not name.startswith("core.backend.router.case_gate."):
            continue
        parameter_names.append(name)
        if parameter.grad is not None:
            gradient = parameter.grad.detach().float()
            gradients.append(gradient.reshape(-1))
            finite_gradients = finite_gradients and bool(torch.isfinite(gradient).all())
        if before is not None:
            update = (parameter.detach() - before[name]).float()
            updates.append(update.reshape(-1))
            finite_updates = finite_updates and bool(torch.isfinite(update).all())
    gradient_vector = torch.cat(gradients) if gradients else torch.empty(0)
    update_vector = torch.cat(updates) if updates else torch.empty(0)
    return {
        "parameter_count": len(parameter_names),
        "parameter_names": parameter_names,
        "gradient_norm": float(gradient_vector.double().norm()) if gradient_vector.numel() else 0.0,
        "update_norm": float(update_vector.double().norm()) if update_vector.numel() else 0.0,
        "finite_gradient": bool(finite_gradients),
        "finite_update": bool(finite_updates),
        "nonzero_gradient": bool(torch.count_nonzero(gradient_vector)) if gradient_vector.numel() else False,
        "nonzero_update": bool(torch.count_nonzero(update_vector)) if update_vector.numel() else False,
    }


def _build_loader(cfg: Mapping[str, Any], dataset: Any, device: Any) -> Any:
    from torch.utils.data import DataLoader

    from channelthermal.data.collation import ChannelThermalBatchCollator, ModuleCountBucketBatchSampler

    dataset_cfg = cfg["dataset"]
    training_cfg = cfg["training"]
    collator = ChannelThermalBatchCollator(
        dynamic_module_padding=bool(dataset_cfg.get("dynamic_module_padding", True)),
        max_modules_per_batch=(
            None
            if dataset_cfg.get("max_modules_per_batch") is None
            else int(dataset_cfg["max_modules_per_batch"])
        ),
    )
    batch_size = int(dataset_cfg.get("batch_size", training_cfg.get("batch_size", 4)))
    num_workers = int(dataset_cfg.get("num_workers", 0))
    if bool(dataset_cfg.get("bucket_by_module_count", True)):
        sampler = ModuleCountBucketBatchSampler(
            dataset.selected_module_counts,
            batch_size=batch_size,
            bucket_size_multiplier=int(dataset_cfg.get("module_count_bucket_size_multiplier", 8)),
            seed=int(training_cfg.get("seed", 42)),
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )


def run_update(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from channelthermal.data.datasets import GlobalChannelThermalDataset
    from channelthermal.training.epoch import (
        effective_local_loss_weights,
        effective_port_condition_settings,
        predicted_consistency_weight_for_epoch,
        run_epoch,
    )
    from channelthermal.training.optimizer import build_forward_optimizer
    from channelthermal.workflows.train_forward import build_model_config, resolve_auto_internal_mode
    from honf_runtime.case_protocol import WorkflowRequest
    from honf_runtime.compat import make_grad_scaler, recursive_to_device, set_seed
    from honf_runtime.config_loader import load_config_bundle
    from honf_runtime.registry import load_case_plugin
    from channelthermal.model import ChannelThermalHONFModel

    device = torch.device(args.device)
    if device.type != "cuda" or device.index != 0:
        raise ValueError("This bounded check must use GPU 0 via --device cuda:0.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the requested real-batch update.")
    torch.cuda.set_device(device)
    torch.cuda.synchronize(device)

    profile = Path(args.profile).expanduser().resolve()
    overlay = Path(args.overlay).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    bundle = load_config_bundle(str(profile), experiment_overlay=str(overlay))
    plugin = load_case_plugin(str(bundle.case["plugin"]))
    plugin.validate_config(bundle)
    request = WorkflowRequest(workflow="forward", device=str(device), epochs=50)
    cfg = plugin._forward_config(bundle, request, output.parent)
    if str(cfg["model"]["core_honf"]["forward_architecture"]) != "budgeted_group_control_honf":
        raise ValueError("resolved profile is not budgeted_group_control_honf")
    loss_cfg = dict(cfg["loss"])
    weight = float(loss_cfg["case_group_budget_weight"])
    if not math.isclose(weight, EXPECTED_WEIGHT, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError(f"unexpected calibrated case_group_budget_weight={weight!r}")
    dataset_cfg = cfg["dataset"]
    training_cfg = cfg["training"]
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
    model = ChannelThermalHONFModel(model_config).to(device)
    if str(getattr(model, "budgeted_schedule_mode", "static")) == "dense_to_sparse_v2":
        raise ValueError(
            "run_run1409_real_update.py is the historical v1 gate-update check; "
            "use run_run1409_v2_prelaunch_audit.py for the rescue schedule"
        )
    model.set_global_target_normalization(
        dataset.normalizer.stats,
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
    )
    resolve_auto_internal_mode(model_config, model)
    local_path = model.local_coupling.local_surrogate_checkpoint_path
    if not model.local_surrogate_attached or not local_path or not Path(local_path).is_file():
        raise RuntimeError("the maintained local-surrogate checkpoint was not attached")
    loader = _build_loader(cfg, dataset, device)
    if hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(1)
    # Preview the deterministic first sampler batch for identity/shape evidence;
    # run_epoch below consumes the same loader at the same sampler epoch.
    first_indices = next(iter(loader.batch_sampler))
    preview_collator = loader.collate_fn
    preview_batch = preview_collator([dataset[index] for index in first_indices])
    batch_size = int(preview_batch["query_xy"].shape[0])
    query_count = int(preview_batch["query_xy"].shape[1])
    if batch_size != 48 or query_count != 1024:
        raise RuntimeError(f"expected real B48/Q1024 batch, got B{batch_size}/Q{query_count}")
    batch_case_ids = list(preview_batch["case_id"])

    optimizer, _optimizer_inventory = build_forward_optimizer(model, training_cfg)
    scaler = make_grad_scaler(device, bool(training_cfg.get("amp", False)))
    mode, ratio = effective_port_condition_settings(1, training_cfg)
    if str(mode) != "predicted":
        raise RuntimeError(f"resolved training mode is {mode!r}, expected predicted")
    internal_weight, interface_weight = effective_local_loss_weights(loss_cfg, mode, ratio)
    predicted_weight = predicted_consistency_weight_for_epoch(1, loss_cfg)
    gate_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("core.backend.router.case_gate.")
    }
    if hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(1)
    train_metrics = run_epoch(
        model,
        loader,
        device,
        loss_cfg,
        optimizer=optimizer,
        scaler=scaler,
        amp=bool(training_cfg.get("amp", False)),
        max_batches=1,
        local_port_condition_mode=str(mode),
        mixed_teacher_ratio=float(ratio),
        effective_internal_temperature_weight=float(internal_weight),
        effective_interface_weight=float(interface_weight),
        predicted_consistency_weight=float(predicted_weight),
        gradient_clip_norm=float(training_cfg.get("gradient_clip_norm", 0.0) or 0.0),
    )
    torch.cuda.synchronize(device)
    gate_stats = _gate_stats(model, gate_before)
    if not gate_stats["finite_gradient"] or not gate_stats["finite_update"]:
        raise RuntimeError(f"non-finite gate gradient/update: {gate_stats}")
    if not gate_stats["nonzero_gradient"] or not gate_stats["nonzero_update"]:
        raise RuntimeError(f"missing gate gradient/update: {gate_stats}")

    # The same one-batch loader is evaluated after the single optimizer step;
    # no second optimizer step or checkpoint operation occurs.
    if hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(1)
    post_metrics = run_epoch(
        model,
        loader,
        device,
        loss_cfg,
        optimizer=None,
        scaler=None,
        amp=False,
        max_batches=1,
        local_port_condition_mode=str(mode),
        mixed_teacher_ratio=float(ratio),
        effective_internal_temperature_weight=float(internal_weight),
        effective_interface_weight=float(interface_weight),
        predicted_consistency_weight=float(predicted_weight),
        gradient_clip_norm=0.0,
    )
    torch.cuda.synchronize(device)
    output_payload = {
        "schema_version": 1,
        "task": "run1409_real_predicted_port_update",
        "status": "complete",
        "profile": str(profile),
        "experiment_overlay": str(overlay),
        "architecture": str(model.config.core_honf.forward_architecture),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "seed": seed,
        "mode": str(mode),
        "mixed_teacher_ratio": float(ratio),
        "effective_internal_temperature_weight": float(internal_weight),
        "effective_interface_weight": float(interface_weight),
        "predicted_consistency_weight": float(predicted_weight),
        "case_group_budget_weight": weight,
        "batch": {
            "batch_size": batch_size,
            "query_count": query_count,
            "case_ids": batch_case_ids,
            "split": str(dataset.split),
            "points_per_case": int(dataset.points_per_case),
            "dataset_path": str(dataset.path),
            "dataset_epoch": int(dataset.epoch),
            "module_counts": preview_batch["module_count"].tolist(),
        },
        "local_surrogate": {
            "attached": bool(model.local_surrogate_attached),
            "checkpoint_path": str(local_path),
            "frozen": bool(model.local_coupling.local_surrogate_frozen),
        },
        "optimizer": {
            "name": optimizer.__class__.__name__,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "weight_decay": float(optimizer.param_groups[0].get("weight_decay", 0.0)),
            "steps": 1,
        },
        "pre_update_metrics_run_epoch": _scalar_metrics(train_metrics),
        "post_update_metrics_run_epoch": _scalar_metrics(post_metrics),
        "pre_update_total_loss": _jsonable(train_metrics.get("loss_total")),
        "pre_update_physical_loss": _jsonable(train_metrics.get("loss_physical")),
        "post_update_total_loss": _jsonable(post_metrics.get("loss_total")),
        "post_update_physical_loss": _jsonable(post_metrics.get("loss_physical")),
        "gate_gradient_update": gate_stats,
        "checkpoint_written": False,
        "managed_run_allocated": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(output_payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable(output_payload), indent=2, sort_keys=True))
    print(f"evidence={output}")
    return output_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--overlay", default=str(DEFAULT_OVERLAY))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.path.insert(0, str(PROJECT_ROOT / "Case_ThermalChannel" / "src"))
    run_update(parse_args())

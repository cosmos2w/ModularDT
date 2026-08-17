"""Four-stage ThermalChannel hierarchical inverse training workflow."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any

import h5py
from torch.utils.data import DataLoader

from honf_inverse_core.config import validate_config_keys
from honf_inverse_core.models.hierarchical_inverse import HierarchicalInverseDesigner
from honf_inverse_core.training.checkpointing import load_inverse_checkpoint
from honf_inverse_core.training.stages import TRAINING_STAGES
from honf_inverse_core.training.trainer import InverseTrainer
from channelthermal.inverse.dataset_io import InverseH5Dataset, validate_inverse_hdf5
from channelthermal.inverse.diagnostics import artifact_sha256
from channelthermal.inverse.differentiable_verifier import DifferentiableThermalChannelVerifier
from channelthermal.inverse.verifier import FrozenThermalChannelVerifier


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Inverse training config must be a JSON object: {path}")
    return value


def _next_run_dir(root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in root.glob("Run_*_"):
        try:
            indices.append(int(path.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    # Glob without relying on suffix shape because names may contain underscores.
    for path in root.glob("Run_*"):
        try:
            indices.append(int(path.name.split("_")[1]))
        except (IndexError, ValueError):
            pass
    index = max(indices, default=0) + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in name)
    return root / f"Run_{index:04d}_{timestamp}_{safe_name}"


def _provenance(dataset_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with h5py.File(dataset_path, "r") as h5:
        raw = h5["provenance/json"][()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        source = json.loads(str(raw))
        normalization = {}
        for name in h5["normalization"].keys():
            value = h5[f"normalization/{name}"][...]
            if value.dtype.kind in {"S", "O", "U"}:
                normalization[name] = [
                    item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    for item in value.reshape(-1)
                ]
            else:
                normalization[name] = value.tolist()
        provenance = {
            "forward_checkpoint_id": source.get("forward_checkpoint_identifier"),
            "forward_checkpoint_sha256": source.get("forward_checkpoint_sha256"),
            "inverse_dataset_version": int(h5.attrs["schema_version"]),
            "inverse_dataset_hash": artifact_sha256(dataset_path),
            "request_schema_version": int(h5.attrs["request_schema_version"]),
            "compact_plan_schema_version": int(h5.attrs["compact_plan_schema_version"]),
            "normalization_stats": normalization,
        }
        return provenance, source


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    config = validate_config_keys(
        config,
        allowed={
            "dataset_path", "forward_checkpoint", "device", "output_root", "run_name",
            "batch_size", "num_workers", "learning_rate", "weight_decay", "allow_partial_debug",
            "joint_query_grid", "joint_local_grid_size", "initialize_from", "model", "stages",
        },
        required={
            "dataset_path", "forward_checkpoint", "device", "output_root", "run_name",
            "batch_size", "model", "stages",
        },
        label="inverse hierarchical training",
    )
    dataset_path = Path(config["dataset_path"]).expanduser().resolve()
    summary = validate_inverse_hdf5(dataset_path)
    with h5py.File(dataset_path, "r") as h5:
        if bool(h5.attrs.get("partial_debug", False)) and not bool(config.get("allow_partial_debug", False)):
            raise ValueError("Formal inverse training rejects partial_debug datasets unless explicitly allowed.")
    train_dataset = InverseH5Dataset(dataset_path, split="train")
    validation_dataset = InverseH5Dataset(dataset_path, split="validation")
    if not train_dataset or not validation_dataset:
        raise ValueError("Inverse training requires nonempty train and validation splits.")
    model_config = dict(config.get("model", {}))
    model_config.update(num_edges=summary["num_edges"], max_modules=summary["max_modules"])
    designer = HierarchicalInverseDesigner.from_config(model_config)
    provenance, source = _provenance(dataset_path)
    initialize_from = config.get("initialize_from")
    initialization = None
    if initialize_from:
        initialization_path = Path(str(initialize_from)).expanduser().resolve()
        warm_start = load_inverse_checkpoint(initialization_path)
        for key in (
            "forward_checkpoint_sha256", "inverse_dataset_hash", "request_schema_version",
            "compact_plan_schema_version",
        ):
            if warm_start["provenance"].get(key) != provenance.get(key):
                raise ValueError(f"Warm-start checkpoint provenance mismatch for {key}.")
        designer.load_compatible_state_dict(warm_start["model_state_dict"])
        initialization = {
            "checkpoint": str(initialization_path),
            "source_stage": warm_start["stage"],
            "source_epoch": int(warm_start["epoch"]),
            "source_global_step": int(warm_start["global_step"]),
        }
    run_dir = _next_run_dir(Path(config["output_root"]).expanduser().resolve(), str(config.get("run_name", "hierarchical_inverse")))
    run_dir.mkdir(parents=True)
    if initialize_from:
        for alias in ("best_plan_model.pt", "best_layout_model.pt", "best_unguided_model.pt"):
            source_alias = Path(str(initialize_from)).expanduser().resolve().parent / alias
            if source_alias.is_file():
                shutil.copy2(source_alias, run_dir / alias)
    with (run_dir / "config_resolved.json").open("w", encoding="utf-8") as stream:
        json.dump({**config, "model": model_config, "run_dir": str(run_dir)}, stream, indent=2, sort_keys=True)
        stream.write("\n")
    batch_size = int(config.get("batch_size", 16))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=int(config.get("num_workers", 0)), pin_memory=str(config["device"]).startswith("cuda"),
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False,
        num_workers=int(config.get("num_workers", 0)), pin_memory=str(config["device"]).startswith("cuda"),
    )
    stages = validate_config_keys(
        config["stages"],
        allowed=set(TRAINING_STAGES),
        required=set(TRAINING_STAGES),
        label="inverse training stages",
    )
    for stage, raw_settings in stages.items():
        settings = validate_config_keys(
            raw_settings,
            allowed={
                "epochs", "learning_rate", "forward_batch_fraction",
                "forward_sample_count", "corrector_only",
            },
            required={"epochs", "learning_rate"},
            label=stage,
        )
        if int(settings["epochs"]) < 0 or float(settings["learning_rate"]) <= 0.0:
            raise ValueError(f"{stage} requires epochs >= 0 and learning_rate > 0.")
        fraction = float(settings.get("forward_batch_fraction", 0.25))
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"{stage}.forward_batch_fraction must be in [0,1].")
        if int(settings.get("forward_sample_count", 4)) <= 0:
            raise ValueError(f"{stage}.forward_sample_count must be positive.")
        stages[stage] = settings
    joint_adapter = None
    if int(stages.get("stage_joint_consistency", {}).get("epochs", 0)) > 0:
        checkpoint = config.get("forward_checkpoint") or source.get("forward_checkpoint_path")
        if not checkpoint:
            raise ValueError("stage_joint_consistency requires a frozen forward checkpoint path.")
        frozen = FrozenThermalChannelVerifier(checkpoint, device=config["device"], dataset_path=source.get("dataset_path"))
        stats = provenance["normalization_stats"]
        joint_adapter = DifferentiableThermalChannelVerifier(
            frozen,
            inverse_heat_mean=float(stats["active_heat_power_mean"]),
            inverse_heat_std=float(stats["active_heat_power_std"]),
            functional_mean=stats["functional_mean"],
            functional_std=stats["functional_std"],
            query_grid=tuple(config.get("joint_query_grid", [32, 16])),
            local_grid_size=int(config.get("joint_local_grid_size", 18)),
            matching=str(model_config.get("matching_mode", "canonical")),
        )
    trainer = InverseTrainer(
        designer,
        device=config["device"],
        run_dir=run_dir,
        checkpoint_provenance=provenance,
        joint_loss_hook=None if joint_adapter is None else joint_adapter.joint_loss_hook,
    )
    if initialization is not None:
        trainer.global_step = int(initialization["source_global_step"])
    print(
        f"[inverse:start] run_dir={run_dir} train_rows={len(train_dataset)} "
        f"validation_rows={len(validation_dataset)} batch_size={batch_size} device={config['device']}",
        flush=True,
    )
    results = []
    for stage in (
        "stage_plan", "stage_layout_teacher_plan", "stage_layout_mixed_plan", "stage_joint_consistency"
    ):
        settings = dict(stages.get(stage, {}))
        epochs = int(settings.get("epochs", 0))
        if epochs <= 0:
            continue
        result = trainer.train_stage(
            stage,
            train_loader,
            validation_loader,
            epochs=epochs,
            learning_rate=float(settings.get("learning_rate", config.get("learning_rate", 1.0e-4))),
            weight_decay=float(config.get("weight_decay", 1.0e-5)),
            joint_batch_fraction=float(settings.get("forward_batch_fraction", 0.25)),
            joint_sample_count=int(settings.get("forward_sample_count", 4)),
            corrector_only=bool(settings.get("corrector_only", False)),
        )
        results.append(result.__dict__)
    trainer.mark_training_complete(results)
    run_summary = {
        "status": "complete",
        "run_dir": str(run_dir),
        "dataset_path": str(dataset_path),
        "dataset_summary": summary,
        "stage_results": results,
        "checkpoint_provenance": provenance,
        "initialization": initialization,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(run_summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    train_dataset.close()
    validation_dataset.close()
    return run_summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", required=True)
    value.add_argument("--dataset-path")
    value.add_argument("--device")
    value.add_argument("--output-root")
    value.add_argument("--run-name")
    value.add_argument("--allow-partial-debug", action="store_true")
    value.add_argument("--smoke", action="store_true", help="Use one epoch/stage and a small model for diagnostics.")
    value.add_argument("--yes", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = _json(Path(args.config).expanduser().resolve())
    for name in ("dataset_path", "device", "output_root", "run_name"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if args.allow_partial_debug:
        config["allow_partial_debug"] = True
    if args.smoke:
        config["allow_partial_debug"] = True
        config["batch_size"] = 2
        config["joint_query_grid"] = [12, 8]
        config["joint_local_grid_size"] = 10
        config["model"] = {
            **dict(config.get("model", {})),
            "request_hidden_dim": 64,
            "plan_hidden_dim": 96,
            "plan_layers": 2,
            "plan_sampling_steps": 4,
            "layout_hidden_dim": 96,
            "layout_layers": 2,
            "layout_sampling_steps": 4,
            "corrector_hidden_dim": 96,
            "corrector_blocks": 1,
        }
        config["stages"] = {
            stage: {**dict(settings), "epochs": 1}
            for stage, settings in config["stages"].items()
        }
    if not args.yes:
        raise SystemExit("Refusing to start inverse training without --yes.")
    print(json.dumps(run_training(config), indent=2, sort_keys=True))
    return 0


__all__ = ["main", "run_training"]

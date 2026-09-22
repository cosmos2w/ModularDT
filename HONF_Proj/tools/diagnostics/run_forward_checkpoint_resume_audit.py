"""Exercise the strict forward-training resume path without running an epoch."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _imports() -> tuple[Any, ...]:
    import sys

    for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel" / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch
    from channelthermal.data.datasets import GlobalChannelThermalDataset
    from channelthermal.evaluation.loading import load_model
    from channelthermal.training.checkpoints import _restore_rng_state, _validate_resume_checkpoint
    from channelthermal.training.optimizer import (
        _validate_optimizer_resume_compatibility,
        build_forward_optimizer,
        refresh_optimizer_group_inventory,
    )

    return (
        torch,
        GlobalChannelThermalDataset,
        load_model,
        _restore_rng_state,
        _validate_resume_checkpoint,
        _validate_optimizer_resume_compatibility,
        build_forward_optimizer,
        refresh_optimizer_group_inventory,
    )


def _finite_optimizer_state(torch: Any, optimizer: Any) -> tuple[int, bool]:
    tensor_count = 0
    finite = True
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                tensor_count += 1
                finite = finite and bool(torch.isfinite(value).all().item())
    return tensor_count, finite


def run(checkpoint_path: Path) -> dict[str, Any]:
    (
        torch,
        GlobalChannelThermalDataset,
        load_model,
        restore_rng_state,
        validate_resume_checkpoint,
        validate_optimizer_resume_compatibility,
        build_forward_optimizer,
        refresh_optimizer_group_inventory,
    ) = _imports()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    model, checkpoint = load_model(checkpoint_path, torch.device("cpu"))
    train_config = dict(checkpoint.get("train_config") or {})
    dataset_config = dict(train_config.get("dataset") or {})
    training_config = dict(train_config.get("training") or {})
    dataset = GlobalChannelThermalDataset(
        dataset_config["packed_h5_path"],
        split=dataset_config.get("train_split", "train"),
        points_per_case=dataset_config.get("points_per_case", 4096),
        normalize_inputs=bool(dataset_config.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_config.get("normalize_targets", False)),
        random_point_sampling=bool(dataset_config.get("random_point_sampling", True)),
        seed=int(training_config.get("seed", 42)),
        require_converged=bool(dataset_config.get("require_converged", False)),
    )
    try:
        validate_resume_checkpoint(
            checkpoint,
            model=model,
            model_config=model.config,
            dataset=dataset,
            dataset_config=dataset_config,
        )
        optimizer, inventory = build_forward_optimizer(model, training_config)
        validate_optimizer_resume_compatibility(checkpoint, inventory)
        inventory = refresh_optimizer_group_inventory(model, inventory)
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if not isinstance(optimizer_state, Mapping):
            raise TypeError("checkpoint does not contain optimizer state")
        optimizer.load_state_dict(optimizer_state)
        restore_rng_state(checkpoint)
        optimizer_tensor_count, optimizer_state_finite = _finite_optimizer_state(
            torch, optimizer
        )
        checkpoint_epoch = int(
            checkpoint.get("epoch", checkpoint.get("current_epoch", 0)) or 0
        )
        return {
            "schema_version": 1,
            "task": "strict_forward_checkpoint_resume_audit",
            "status": "complete",
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint_epoch,
            "next_epoch": checkpoint_epoch + 1,
            "architecture": str(model.config.core_honf.forward_architecture),
            "model_state_strict_load": True,
            "checkpoint_identity_valid": True,
            "model_config_valid": True,
            "dataset_schema_valid": True,
            "dataset_normalization_valid": True,
            "optimizer_group_structure_valid": True,
            "optimizer_state_loaded": True,
            "optimizer_group_count": len(optimizer.param_groups),
            "optimizer_parameter_state_count": len(optimizer.state),
            "optimizer_state_tensor_count": optimizer_tensor_count,
            "optimizer_state_finite": optimizer_state_finite,
            "optimizer_mode": inventory["mode"],
            "rng_state_restored": True,
            "training_epoch_executed": False,
            "checkpoint_written": False,
        }
    finally:
        dataset.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args.checkpoint)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

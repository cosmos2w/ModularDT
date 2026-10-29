#!/usr/bin/env python3
"""Add versioned HONF metadata to a trusted historical checkpoint.

This migration deliberately leaves tensor keys and numerical values untouched.
PyTorch checkpoints use pickle; only migrate files from a trusted source.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from honf_runtime.compat import load_trusted_checkpoint


def classify(payload: dict) -> str:
    """Classify the two checkpoint families supported by schema version 1."""

    if payload.get("stage") or payload.get("channel_order"):
        return "forward"
    if payload.get("dataset_feature_names") or payload.get("local_normalization_stats"):
        return "local_module"
    raise ValueError("Checkpoint is neither a recognized forward nor ThermalChannel local-module artifact.")


def migrate(payload: dict) -> tuple[dict, list[str]]:
    """Return a shallow migrated payload and the metadata fields added."""

    migrated = dict(payload)
    workflow = classify(migrated)
    defaults = {
        "checkpoint_schema_version": 1,
        "case_id": "ThermalChannel",
        "model_family": "honf_forward",
        "workflow": workflow,
    }
    if workflow == "local_module":
        defaults["local_module_id"] = "thermal_disk"
    added = []
    for key, value in defaults.items():
        if key not in migrated:
            migrated[key] = value
            added.append(key)
    if int(migrated.get("checkpoint_schema_version", 0)) != 1:
        raise ValueError("Only checkpoint schema version 1 is supported by this release.")
    return migrated, added


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Trusted historical .pt checkpoint.")
    parser.add_argument("--output", help="New .pt path; required unless --dry-run is used.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = Path(args.checkpoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = load_trusted_checkpoint(source, map_location="cpu")
    migrated, added = migrate(payload)
    print(f"source={source}")
    print(f"workflow={migrated['workflow']}")
    print(f"schema_version={migrated['checkpoint_schema_version']}")
    print(f"added={added}")
    if args.dry_run:
        return 0
    if not args.output:
        raise ValueError("--output is required unless --dry-run is used.")
    destination = Path(args.output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(migrated, temporary)
    temporary.replace(destination)
    print(f"output={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

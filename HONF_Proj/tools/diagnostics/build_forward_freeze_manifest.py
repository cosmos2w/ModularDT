#!/usr/bin/env python3
"""Build the Stage-1--7 checkpoint freeze manifest without copying weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT_ROOT / "Trained_Results" / "ThermalChannel" / "HONF_Forward_Runs"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "experiments" / "stage1_7_freeze_manifest.json"
RUNS = {
    "run1000": "Run_1000_20260817_214356_enhanced_honf_pairwise",
    "run1005": "Run_1005_20260818_114516_stage1_fixed_additive_soft_init",
    "run1007": "Run_1007_20260818_161422_adaptive_sparse_additive",
    "run1102": "Run_1102_20260820_002237_adaptive_sparse_additive_formal",
    "run1202": "Run_1202_20260820_185410_stage4_split_lr_dense_background",
    "run1301": "Run_1301_20260821_142454_stage5_exchangeable_soft_organized",
    "run1302": "Run_1302_20260821_142505_stage5_fixed_softmax_modern",
    "run1304": "Run_1304_20260822_232649_stage5_fixed_residual_concat_uniform_lr3e4",
    "run1401": "Run_1401_20260823_151126_stage7_modern_structured_context",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def resolve_artifact(run_dir: Path, canonical: str, historical: str) -> Path:
    candidate = run_dir / canonical
    return candidate if candidate.exists() else run_dir / historical


def state_inventory_digest(state: dict[str, torch.Tensor]) -> str:
    inventory = [
        [name, list(value.shape), str(value.dtype)]
        for name, value in sorted(state.items())
    ]
    payload = json.dumps(inventory, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def checkpoint_record(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    return {
        "path": relative(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))),
        "state_key_count": len(state),
        "state_inventory_sha256": state_inventory_digest(state),
        "checkpoint_schema_version": checkpoint.get("checkpoint_schema_version"),
    }


def run_record(label: str, directory_name: str) -> dict[str, Any]:
    run_dir = RUN_ROOT / directory_name
    best_path = resolve_artifact(run_dir, "checkpoints/best_field.pt", "best_by_field_mse_model.pt")
    latest_path = resolve_artifact(run_dir, "checkpoints/latest.pt", "latest_model.pt")
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    config_path = resolve_artifact(run_dir, "configs/resolved.json", "config_resolved.json")
    manifest_path = run_dir / "run_manifest.json"
    local_path_value = best_checkpoint.get(
        "local_checkpoint_provenance",
        best_checkpoint.get("local_surrogate_checkpoint_path"),
    )
    local_path = Path(local_path_value) if local_path_value else None
    return {
        "run_directory": relative(run_dir),
        "status": "current_scientific_baseline" if label == "run1401" else "historical_reference",
        "best_field": checkpoint_record(best_path),
        "latest": checkpoint_record(latest_path),
        "resolved_config": {
            "path": relative(config_path),
            "sha256": file_sha256(config_path),
        },
        "run_manifest": {
            "path": relative(manifest_path),
            "sha256": file_sha256(manifest_path),
        },
        "dataset": {
            "id": best_checkpoint.get("dataset_id"),
            "sha256": best_checkpoint.get("dataset_fingerprint"),
        },
        "stage_a": (
            {
                "path": relative(local_path),
                "sha256": file_sha256(local_path),
            }
            if local_path is not None and local_path.exists()
            else {"path": str(local_path_value), "sha256": None}
        ),
    }


def main() -> int:
    args = parse_args()
    runs = {label: run_record(label, directory) for label, directory in RUNS.items()}
    payload = {
        "schema_version": 1,
        "freeze_scope": "HONF forward research Stages 1-7",
        "source_sha_at_cleanup_start": args.source_sha,
        "checkpoint_bytes_stored_in_git": False,
        "accepted_primary_model": {
            "run": "run1401",
            "checkpoint": "best_field",
            "epoch": 4585,
            "organizer": "fixed K=6 softmax",
            "mechanism_state": "residual/raw hyperedge state",
            "field_assembly": "context_fusion",
        },
        "historical_compatibility_reference": {
            "run": "run1000",
            "checkpoint": "best_field",
            "epoch": 9655,
        },
        "runs": runs,
        "golden_references": {
            "run1000_best_field_e9655": {
                "checkpoint": runs["run1000"]["best_field"]["path"],
                "checkpoint_sha256": runs["run1000"]["best_field"]["sha256"],
                "epoch": 9655,
                "case_id": "0653",
                "split": "test",
                "query_batch_size": 8192,
            },
            "run1401_best_field_e4585": {
                "checkpoint": runs["run1401"]["best_field"]["path"],
                "checkpoint_sha256": runs["run1401"]["best_field"]["sha256"],
                "epoch": 4585,
                "case_id": "0653",
                "split": "test",
                "query_batch_size": 8192,
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

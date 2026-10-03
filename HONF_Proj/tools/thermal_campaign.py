#!/usr/bin/env python3
"""Generate the finite native Thermal portfolio and print ordinary launch commands.

Configuration generation never starts an optimizer or a subprocess trainer.
Stages use the same absolute 5000-epoch schedule and native full-case split.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "B-native": "dense_pairwise_field",
    "B-fine": "three_term_full_access_honf",
    "H-tree": "adaptive_receiver_hypergraph_honf",
    "H-overlap": "overlap_control_hypergraph_honf",
    "H-local": "local_overlap_hypergraph_honf",
}


def portfolio_profiles(*, first_run_id: int = 2201, stage: int = 100, microbatch_size: int = 8) -> dict[str, dict]:
    """Build matched profiles from the maintained native physical initializer."""

    if stage not in (100, 500, 1000, 5000):
        raise ValueError("Campaign stage must be 100, 500, 1000 or manual 5000.")
    base = json.loads((PROJECT_ROOT / "src/config_core/forward/dense_pairwise_interface_context.json").read_text())
    profiles = {}
    for offset, (arm, architecture) in enumerate(ARMS.items()):
        config = copy.deepcopy(base)
        config["profile_name"] = f"thermal_shared_core_{arm.lower()}_v1"
        config["model"]["core_honf"]["forward_architecture"] = architecture
        config["case"]["config"] = "project://Case_ThermalChannel/configs/case_default.json"
        config["training"].update(seed=0, epochs=stage, learning_rate=3e-4, organizer_learning_rate=None,
                                  weight_decay=1e-5, amp=False, gradient_clip_norm=1.0,
                                  max_train_batches_per_epoch=None, max_val_batches=None)
        config["training"]["campaign"] = {
            "name": "thermal_shared_core_v1", "arm": arm, "version": 1, "parent": None,
            "schedule_total_epochs": 5000, "require_full_epoch": True,
            "matched_fresh_initialization": True, "gpu_telemetry": True,
            "microbatch_size": microbatch_size,
            "structural_weight": 0.001, "structural_ramp_start": 26, "structural_ramp_end": 100,
            "response_stencils": [f"project://diagnostics/generated/interactions/physical_response_atlas_20260926/families/train_{case}_responses.npz"
                                  for case in ("0001", "0304", "0318", "0320", "0333", "0335", "0348", "0350")],
        }
        config["checkpointing"].update(save_latest_every_epochs=25,
                                       save_epoch_milestones=[*range(25, 1001, 25), 2500, 5000])
        # The first screen remains policy 1; reviewed stage continuations
        # explicitly amend the common physical objective at epoch 101.
        if stage > 100:
            config["training"]["campaign"].update(
                physical_loss_policy_version=2, native_loss_denominators_start_epoch=101,
            )
        config["run"].update(id=f"{first_run_id + offset:04d}", name=f"thermal_{arm.lower()}_v1")
        config["_note"] = (
            "Native full-case seed0 campaign; Stage-A frozen; same effective batch48 and Q1024. "
            "Shared physical tensors come from materialized fresh B-fine. Stage stops do not reset the schedule. "
            "Use --resume-checkpoint at stages500/1000; 5000 is manual only. "
            "Primary query counters exclude auxiliary P0/P1/P2 queries."
        )
        profiles[arm] = config
    return profiles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-run-id", type=int, default=None)
    parser.add_argument("--stage", type=int, choices=(100, 500, 1000, 5000), default=100)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--fresh", action="store_true", help="Manual independent fresh 5000 recipe; requires explicit unused run IDs.")
    args = parser.parse_args()
    if args.fresh and (args.stage != 5000 or args.first_run_id is None):
        parser.error("--fresh is a manual 5000 recipe and requires --stage 5000 plus explicit --first-run-id.")
    profiles = portfolio_profiles(first_run_id=2201 if args.first_run_id is None else args.first_run_id,
                                  stage=args.stage, microbatch_size=args.microbatch_size)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == 100 or args.fresh:
        for config in profiles.values():
            run_id = config["run"]["id"]
            conflicts = list((PROJECT_ROOT / "Trained_Results").rglob(f"Run_{run_id}_*"))
            if conflicts:
                raise FileExistsError(f"Run ID {run_id} already exists: {conflicts}")
    for arm, config in profiles.items():
        path = output / f"{arm.lower()}_e{args.stage}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        suffix = "" if args.stage == 100 or args.fresh else " --resume-checkpoint PATH_TO_EXACT_PREVIOUS_STAGE_CHECKPOINT"
        print(f"{arm}: CUDA_VISIBLE_DEVICES=1 python train.py --config {path} --device cuda:0{suffix} --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

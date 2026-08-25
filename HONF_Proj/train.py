#!/usr/bin/env python3
"""Generic HONF training entry point.

This file intentionally knows nothing about ThermalChannel tensors, losses, or
models.  It composes configuration, loads the selected case plugin, prints the
fully resolved launch, reserves a standard run directory, and delegates the
case-specific workflow.
"""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import replace
from pathlib import Path

from honf_runtime.case_protocol import WorkflowRequest
from honf_runtime.config_loader import load_config_bundle
from honf_runtime.launch_summary import confirm_launch, print_launch_summary
from honf_runtime.paths import resolve_path
from honf_runtime.registry import load_case_plugin, require_model_family
from honf_runtime.run_store import RunStore


DEFAULT_CONFIG = "project://src/config_core/forward/enhanced_honf_pairwise.json"


def _validate_managed_resume(run_dir: Path, bundle, workflow: str) -> dict | None:
    """Validate immutable run identity/config sections before managed resume."""

    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "case_id": bundle.effective["case"]["id"],
        "model_family": bundle.effective["model_family"],
        "workflow": workflow,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Resume run identity mismatch: {mismatches}")
    resolved_path = run_dir / "configs" / "resolved_config.json"
    if resolved_path.exists():
        saved = json.loads(resolved_path.read_text(encoding="utf-8"))
        for section in ("model", "dataset", "loss", "case"):
            if saved.get(section) != bundle.effective.get(section):
                raise ValueError(
                    f"Resume configuration section {section!r} differs from the original run. "
                    "Only runtime/training-duration overrides may change."
                )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a registered HONF case/model workflow.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Core launch profile.")
    parser.add_argument("--workflow", choices=["forward", "local_module"], default=None)
    parser.add_argument("--experiment-overlay", default=None, help="Strict overlay for an archived/ablation profile.")
    parser.add_argument("--device", default=None, help="Torch device override, for example cpu or cuda:0.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--run-id", "--Run_ID", dest="run_id", default=None)
    parser.add_argument("--run-name", default=None)
    checkpoint_mode = parser.add_mutually_exclusive_group()
    checkpoint_mode.add_argument("--resume-checkpoint", default=None)
    checkpoint_mode.add_argument(
        "--initialize-checkpoint",
        default=None,
        help="Start a new forward run from compatible name-and-shape matched checkpoint weights.",
    )
    parser.add_argument(
        "--local-checkpoint",
        default=None,
        help="Forward: Stage-A dependency override. Local workflow: optional initialization checkpoint.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the launch without writing anything.")
    parser.add_argument("--yes", action="store_true", help="Confirm an unattended launch after reviewing --dry-run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overrides = {
        "device": args.device,
        "epochs": args.epochs,
        "run_id": args.run_id,
        "run_name": args.run_name,
    }
    bundle = load_config_bundle(
        args.config,
        overrides=overrides,
        experiment_overlay=args.experiment_overlay,
    )
    workflow = str(args.workflow or bundle.effective["workflow"])
    if args.workflow is not None and args.workflow != bundle.effective["workflow"]:
        raise ValueError(
            f"--workflow={args.workflow!r} conflicts with profile workflow={bundle.effective['workflow']!r}; "
            "choose the matching core profile."
        )
    require_model_family(bundle.effective["model_family"], workflow)
    request = WorkflowRequest(
        workflow=workflow,
        device=args.device,
        run_id=bundle.effective["Run_ID"],
        run_name=str(bundle.effective["run"].get("name", "run")),
        epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        resume_checkpoint=args.resume_checkpoint,
        initialize_checkpoint=args.initialize_checkpoint,
        local_checkpoint=args.local_checkpoint,
    )
    plugin_path = str(bundle.case.get("plugin", ""))
    plugin = load_case_plugin(plugin_path)
    if plugin.case_id != bundle.effective["case"]["id"]:
        raise ValueError(
            f"Loaded plugin case_id={plugin.case_id!r} does not match configured "
            f"case_id={bundle.effective['case']['id']!r}."
        )
    plugin.validate_config(bundle)
    case_facts = plugin.inspect_launch(bundle, request)

    if args.resume_checkpoint:
        run_dir = resolve_path(args.resume_checkpoint).parent
        resume_manifest = _validate_managed_resume(run_dir, bundle, workflow)
        managed_resume = resume_manifest is not None
        if resume_manifest is not None:
            request = replace(
                request,
                run_id=str(resume_manifest.get("run_id", request.run_id)),
                run_name=run_dir.name,
            )
        proposed_output = str(run_dir)
        proposal = None
        store = None
    else:
        managed_resume = False
        run_cfg = bundle.effective["run"]
        store = RunStore(run_cfg.get("output_root", "project://Trained_Results"))
        local_module_id = bundle.effective["case"].get("selection", {}).get("local_module_id")
        proposal = store.propose(
            case_id=plugin.case_id,
            workflow=workflow,
            model_family=bundle.effective["model_family"],
            run_id=bundle.effective["Run_ID"],
            run_name=str(run_cfg.get("name", "run")),
            local_module_id=None if local_module_id is None else str(local_module_id),
        )
        run_dir = proposal.path
        proposed_output = str(proposal.path)

    print_launch_summary(bundle, request, case_facts, proposed_output=proposed_output)
    if not confirm_launch(assume_yes=bool(args.yes), dry_run=bool(args.dry_run)):
        if not args.dry_run:
            print("[cancelled] no run directory was created.")
        return 0

    if proposal is not None and store is not None:
        run_dir = store.create(proposal, bundle, launch_facts=case_facts)
        store.update_status(run_dir, "running")
    elif managed_resume:
        RunStore.update_status(Path(run_dir), "running", resumed_from=str(resolve_path(args.resume_checkpoint)))
    try:
        status = int(plugin.train(bundle, request, run_dir=Path(run_dir)))
    except BaseException as exc:
        if (Path(run_dir) / "run_manifest.json").exists():
            last_completed_epoch = None
            for history_name in ("metrics.csv", "loss_history.csv"):
                history_path = Path(run_dir) / history_name
                if history_path.is_file():
                    try:
                        lines = [line for line in history_path.read_text(encoding="utf-8").splitlines()[1:] if line]
                        if lines:
                            last_completed_epoch = int(float(lines[-1].split(",", 1)[0]))
                    except (OSError, ValueError):
                        pass
                    break
            RunStore.update_status(
                Path(run_dir),
                "failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback=traceback.format_exc(),
                last_completed_epoch=last_completed_epoch,
            )
        raise
    if (Path(run_dir) / "run_manifest.json").exists():
        checkpoint_inventory = RunStore.finalize_artifacts(Path(run_dir))
        metric_summary = RunStore.metric_summary(Path(run_dir))
        RunStore.update_status(
            Path(run_dir),
            "completed" if status == 0 else "failed",
            exit_code=status,
            checkpoints=checkpoint_inventory,
            **metric_summary,
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())

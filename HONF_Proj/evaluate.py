#!/usr/bin/env python3
"""Generic HONF evaluation and post-processing entry point."""

from __future__ import annotations

import argparse

from honf_runtime.case_protocol import WorkflowRequest
from honf_runtime.config_loader import load_config_bundle
from honf_runtime.registry import load_case_plugin, require_model_family


DEFAULT_CONFIG = "project://src/config_core/forward/enhanced_honf_pairwise.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a registered HONF case workflow.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Core profile used to locate the case plugin.")
    parser.add_argument("--workflow", choices=["forward", "local_module", "compare"], default="forward")
    parser.add_argument("--experiment-overlay", default=None)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint selector/path understood by the case workflow.")
    parser.add_argument(
        "--run-id",
        "--Run_ID",
        dest="run_ids",
        action="append",
        default=[],
        help="Run ID; repeat for the compare workflow.",
    )
    parser.add_argument("--saved-root", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    args, case_args = parser.parse_known_args()
    if args.workflow == "compare":
        for run_id in args.run_ids:
            case_args.extend(["--Run_ID", str(run_id)])
        args.run_id = None
    else:
        if len(args.run_ids) > 1:
            parser.error("forward/local_module evaluation accepts only one --run-id")
        args.run_id = args.run_ids[0] if args.run_ids else None
    args.case_args = case_args
    return args


def main() -> int:
    args = parse_args()
    bundle = load_config_bundle(
        args.config,
        overrides={"device": args.device},
        experiment_overlay=args.experiment_overlay,
    )
    require_model_family(bundle.effective["model_family"], args.workflow)
    plugin = load_case_plugin(str(bundle.case.get("plugin", "")))
    plugin.validate_config(bundle)
    request = WorkflowRequest(
        workflow=args.workflow,
        device=args.device,
        run_id=args.run_id,
        checkpoint=args.checkpoint,
        saved_root=args.saved_root,
        dataset=args.dataset,
        output_dir=args.output_dir,
        extra_args=tuple(args.case_args),
    )
    return int(plugin.evaluate(bundle, request))


if __name__ == "__main__":
    raise SystemExit(main())

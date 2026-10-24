#!/usr/bin/env python3
"""Ordinary user-directed Thermal research training and evaluation."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / "src", PROJECT / "Case_ThermalChannel/src", PROJECT / "Case_WindFarm/src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from honf_runtime import research_training as research


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="start one ordinary research run")
    train.add_argument("--config", required=True)
    train.add_argument("--device", required=True, help="visible device ordinal, e.g. cuda:0 or cpu")
    train.add_argument("--devices", nargs="+", help="optional permitted fallback/preference list")
    train.add_argument("--epochs", type=int, help="local stop; defaults to the configured first segment")
    train.add_argument("--output", required=True)
    train.add_argument("--diagnostic-only", action="store_true")
    train.add_argument("--case-limit", type=int)
    train.add_argument("--case-id", action="append", dest="case_ids")

    qualify = commands.add_parser("qualify-softonly", help="bounded no-retained-training soft-only qualification")
    qualify.add_argument("--config", required=True)
    qualify.add_argument("--device", required=True)
    qualify.add_argument("--output", required=True)

    resume = commands.add_parser("resume", help="continue this run from its latest complete checkpoint")
    resume.add_argument("--run", required=True)
    resume.add_argument("--device", required=True)
    resume.add_argument("--devices", nargs="+")
    resume.add_argument("--extend-horizon", action="store_true",
                        help="explicitly extend a terminal soft-only run; preserve its original schedules")
    resume_stop = resume.add_mutually_exclusive_group(required=True)
    resume_stop.add_argument("--additional-epochs", type=int)
    resume_stop.add_argument("--stop-after-epoch", type=int,
                             help="absolute local epoch to reach, including the checkpoint epoch")
    resume.add_argument("--microbatch-cases", type=int,
                        help="execution-only runtime packing; does not change run identity")
    resume.add_argument("--query-tile-size", type=int,
                        help="execution-only query tile override")
    resume.add_argument("--checkpoint", choices=("off", "values", "tile"),
                        help="execution-only activation-checkpoint mode")
    resume.add_argument("--backend", choices=("segment", "dense_weighted", "source_order", "packed", "dense"),
                        help="execution-only native execution backend")

    cut = commands.add_parser("activate-cut", help="activate and persist the Round12 R1 hard cut after local e40")
    cut.add_argument("--run", required=True)

    fork_common = commands.add_parser("fork-common", help="calibrate and fork Round12 R0/R1 from common local e20")
    fork_common.add_argument("--run", required=True)
    fork_common.add_argument("--r0-config", required=True)
    fork_common.add_argument("--r0-output", required=True)
    fork_common.add_argument("--r1-config", required=True)
    fork_common.add_argument("--r1-output", required=True)
    fork_common.add_argument("--device", required=True)

    fork_terminal = commands.add_parser("fork-terminal", help="create an explicit longer-horizon R0/R1 continuation")
    fork_terminal.add_argument("--run", required=True)
    fork_terminal.add_argument("--output", required=True)
    fork_terminal.add_argument("--local-horizon", required=True, type=int)
    fork_terminal.add_argument("--device", required=True)

    evaluate = commands.add_parser("evaluate", help="evaluate latest, best, or an explicit checkpoint")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--checkpoint", default="latest")

    assess = commands.add_parser("assess", help="summarize history and available checkpoints")
    assess.add_argument("--run", required=True)
    assess.add_argument("--dataset", default="thermal")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "train":
        result = research.train(
            args.config,
            device=args.device,
            devices=args.devices,
            epochs=args.epochs,
            output=args.output,
            case_limit=args.case_limit,
            case_ids=args.case_ids,
            diagnostic_only=args.diagnostic_only,
        )
    elif args.command == "qualify-softonly":
        from honf_runtime.thermal_soft_organization import qualify_softonly
        result = qualify_softonly(args.config, args.device, args.output)
    elif args.command == "resume":
        result = research.resume(
            args.run,
            device=args.device,
            devices=args.devices,
            additional_epochs=args.additional_epochs,
            extend_horizon=args.extend_horizon,
            stop_after_epoch=args.stop_after_epoch,
            microbatch_cases=args.microbatch_cases,
            query_tile_size=args.query_tile_size,
            checkpoint=args.checkpoint,
            backend=args.backend,
        )
    elif args.command == "activate-cut":
        result = research.activate_round12_cut(args.run)
    elif args.command == "fork-common":
        result = research.fork_round12_common(
            args.run,
            r0_config_path=args.r0_config,
            r0_output=args.r0_output,
            r1_config_path=args.r1_config,
            r1_output=args.r1_output,
            device=args.device,
        )
    elif args.command == "fork-terminal":
        result = research.fork_round12_terminal(
            args.run, output=args.output, local_horizon=args.local_horizon, device=args.device,
        )
    elif args.command == "evaluate":
        result = research.evaluate(args.run, device=args.device, checkpoint=args.checkpoint)
    else:
        result = research.assess(args.run, dataset=args.dataset)
    if isinstance(result, dict):
        print(json.dumps(result, sort_keys=True, default=str, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

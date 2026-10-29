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

    train_member_routing = commands.add_parser(
        "train-member-routing",
        help="start one maintained Thermal member-routing run",
    )
    train_member_routing.add_argument("--config", required=True)
    train_member_routing.add_argument("--device", required=True)
    train_member_routing.add_argument("--devices", nargs="+")
    train_member_routing.add_argument("--epochs", type=int)
    train_member_routing.add_argument("--output", required=True)
    train_member_routing.add_argument("--dry-run", action="store_true",
                                      help="build the provider and print its identity without creating run output")

    for producer in (train_member_routing,):
        producer.add_argument("--compact-history", action=argparse.BooleanOptionalAction, default=True,
                              help="store complete epoch values with shared metadata recorded once")

    resume = commands.add_parser("resume", help="continue this run from its latest complete checkpoint")
    resume.add_argument("--run", required=True)
    resume.add_argument("--device", required=True)
    resume.add_argument("--devices", nargs="+")
    resume.add_argument("--extend-horizon", action="store_true",
                        help="explicitly extend a completed research run; preserve its original schedules")
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

    resume_member_routing = commands.add_parser(
        "resume-member-routing",
        help="resume a maintained Thermal member-routing run",
    )
    resume_member_routing.add_argument("--run", required=True)
    resume_member_routing.add_argument("--device", required=True)
    resume_member_routing.add_argument("--devices", nargs="+")
    resume_member_routing.add_argument("--stop-after-epoch", type=int, required=True)
    resume_member_routing.add_argument(
        "--extend-horizon", action="store_true",
        help="explicitly continue a completed configured endpoint with the same optimizer schedule",
    )
    resume_member_routing.add_argument(
        "--wall-clock-budget-seconds", type=int,
        help="absolute cumulative provider elapsed-time cap for Relative full-TRAIN600 continuation",
    )

    for resumer in (resume_member_routing,):
        resumer.add_argument("--compact-history", action=argparse.BooleanOptionalAction, default=None,
                             help="output-only history format; omitted preserves the run's current format")

    evaluate = commands.add_parser("evaluate", help="evaluate latest, best, or an explicit checkpoint")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--checkpoint", default="latest")

    anchored = commands.add_parser("evaluate-r13-8", aliases=["evaluate-member-routing"],
                                   help="evaluate an anchored checkpoint")
    anchored.add_argument("--checkpoint", required=True)
    anchored.add_argument("--device", default="cpu")
    anchored.add_argument("--output", required=True)
    anchored.add_argument("--representative-case", action="append", dest="representative_cases")

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
    elif args.command == "train-member-routing":
        import torch
        from honf_runtime.contextual_frontier_training import (
            _arm, _build_task, load_contextual_frontier_config, train_contextual_frontier,
        )
        if args.dry_run:
            config, config_path = load_contextual_frontier_config(args.config)
            arm = _arm(config)
            model, provider = _build_task(config, arm, torch.device(args.device))
            result = {
                "event": "member_routing_dry_run_ready",
                "config_path": str(config_path),
                "arm": arm,
                "architecture": model.architecture,
                "training_cases": len(provider.training_case_ids),
                "validation_cases": len(provider.validation_case_ids),
                "validation_scope": provider.validation_scope,
                "provider_identity": provider.identity_payload(),
                "output_created": False,
            }
        else:
            result = train_contextual_frontier(
                args.config, device=args.device, devices=args.devices,
                epochs=args.epochs, output=args.output, compact_history=args.compact_history,
            )
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
    elif args.command == "resume-member-routing":
        from honf_runtime.contextual_frontier_training import resume_contextual_frontier
        result = resume_contextual_frontier(
            args.run, device=args.device, devices=args.devices,
            stop_after_epoch=args.stop_after_epoch, extend_horizon=args.extend_horizon,
            wall_clock_budget_seconds=args.wall_clock_budget_seconds,
            compact_history=args.compact_history,
        )
    elif args.command == "evaluate":
        result = research.evaluate(args.run, device=args.device, checkpoint=args.checkpoint)
    elif args.command in ("evaluate-r13-8", "evaluate-member-routing"):
        from honf_runtime.r13_8_anchored_organization_evaluation import evaluate
        result = evaluate(args.checkpoint, args.device, args.output,
                          representative_case_ids=args.representative_cases)
    else:
        result = research.assess(args.run, dataset=args.dataset)
    if isinstance(result, dict):
        print(json.dumps(result, sort_keys=True, default=str, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

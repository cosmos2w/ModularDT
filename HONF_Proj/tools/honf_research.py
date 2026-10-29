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

    train_contextual = commands.add_parser(
        "train-contextual", help="start one Thermal contextual-frontier joint run",
    )
    train_contextual.add_argument("--config", required=True)
    train_contextual.add_argument("--device", required=True, help="visible device ordinal, e.g. cuda:0 or cpu")
    train_contextual.add_argument("--devices", nargs="+", help="optional permitted fallback/preference list")
    train_contextual.add_argument("--epochs", type=int, help="local stop; defaults to the configured first segment")
    train_contextual.add_argument("--output", required=True)

    train_member_routing = commands.add_parser(
        "train-member-routing", aliases=["train-single-distribution-member-routing"],
        help="start one maintained Thermal member-routing run",
    )
    train_member_routing.add_argument("--config", required=True)
    train_member_routing.add_argument("--device", required=True)
    train_member_routing.add_argument("--devices", nargs="+")
    train_member_routing.add_argument("--epochs", type=int)
    train_member_routing.add_argument("--output", required=True)
    train_member_routing.add_argument("--dry-run", action="store_true",
                                      help="build the provider and print its identity without creating run output")

    for producer in (train_contextual, train_member_routing):
        producer.add_argument("--compact-history", action=argparse.BooleanOptionalAction, default=True,
                              help="store complete epoch values with shared metadata recorded once")

    continue_contextual = commands.add_parser(
        "continue-contextual", help="continue an F13 development checkpoint with full state in a fresh namespace",
    )
    continue_contextual.add_argument("--config", required=True)
    continue_contextual.add_argument("--device", required=True)
    continue_contextual.add_argument("--devices", nargs="+")
    continue_contextual.add_argument("--epochs", type=int, help="absolute local stop, including the source epoch")
    continue_contextual.add_argument("--output", required=True)

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

    resume_contextual = commands.add_parser(
        "resume-contextual", help="continue a contextual-frontier run from its latest complete checkpoint",
    )
    resume_contextual.add_argument("--run", required=True)
    resume_contextual.add_argument("--device", required=True)
    resume_contextual.add_argument("--devices", nargs="+")
    resume_contextual.add_argument("--stop-after-epoch", type=int, required=True,
                                   help="absolute local epoch to reach")
    resume_contextual.add_argument(
        "--wall-clock-budget-seconds", type=int,
        help=("absolute cumulative provider elapsed-time cap for a supported F13 full-TRAIN resume; "
              "omitting it reuses any saved run-manifest value"),
    )

    resume_member_routing = commands.add_parser(
        "resume-member-routing", aliases=["resume-single-distribution-member-routing"],
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

    for resumer in (resume_contextual, resume_member_routing):
        resumer.add_argument("--compact-history", action=argparse.BooleanOptionalAction, default=None,
                             help="output-only history format; omitted preserves the run's current format")

    evaluate_member_routing = commands.add_parser(
        "evaluate-member-routing", help="evaluate native VAL89, routing probes, and measured work for a checkpoint",
    )
    evaluate_member_routing.add_argument("--checkpoint", required=True)
    evaluate_member_routing.add_argument("--device", default="cpu")
    evaluate_member_routing.add_argument("--output", required=True)

    evaluate_joint_member_routing = commands.add_parser(
        "evaluate-joint-member-routing",
        help="evaluate both full and joint95 native paths on VAL89 plus the fixed R13-5 detail panel",
    )
    evaluate_joint_member_routing.add_argument("--checkpoint", required=True)
    evaluate_joint_member_routing.add_argument("--device", default="cpu")
    evaluate_joint_member_routing.add_argument("--output", required=True)

    evaluate_r13_6_member_routing = commands.add_parser(
        "evaluate-r13-6-member-routing", aliases=["evaluate-r13-7-single-distribution-member-routing"],
        help="evaluate a maintained R13-6 C/H or R13-7 U checkpoint on all canonical VAL89 cases",
    )
    evaluate_r13_6_member_routing.add_argument("--checkpoint", required=True)
    evaluate_r13_6_member_routing.add_argument("--device", default="cpu")
    evaluate_r13_6_member_routing.add_argument("--output", required=True)

    evaluate_supplement = commands.add_parser(
        "evaluate-supplement-member-routing",
        help="evaluate the R13-5 supplement full and joint95 native paths on the fixed panel",
    )
    evaluate_supplement.add_argument("--checkpoint", required=True)
    evaluate_supplement.add_argument("--device", default="cpu")
    evaluate_supplement.add_argument("--output", required=True)

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

    evaluate_contextual = commands.add_parser(
        "evaluate-contextual", help="evaluate a contextual-frontier or interaction-groups checkpoint without training",
    )
    evaluate_contextual.add_argument("--checkpoint", required=True)
    evaluate_contextual.add_argument("--device", required=True)
    evaluate_contextual.add_argument("--output", required=True)
    evaluate_contextual.add_argument(
        "--kind",
        choices=("native", "responses", "organization", "group-execution", "group-spatial-diagnostic", "spatial-groups"),
        required=True,
    )
    evaluate_contextual.add_argument(
        "--include-cost",
        action="store_true",
        help="include the bounded Round13 complete-native timing panel (spatial-groups only)",
    )

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
    elif args.command == "train-contextual":
        from honf_runtime.contextual_frontier_training import train_contextual_frontier
        result = train_contextual_frontier(
            args.config,
            device=args.device,
            devices=args.devices,
            epochs=args.epochs,
            output=args.output,
        )
    elif args.command in ("train-member-routing", "train-single-distribution-member-routing"):
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
    elif args.command == "continue-contextual":
        from honf_runtime.contextual_frontier_training import continue_contextual_frontier
        result = continue_contextual_frontier(
            args.config,
            device=args.device,
            devices=args.devices,
            epochs=args.epochs,
            output=args.output,
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
    elif args.command == "resume-contextual":
        from honf_runtime.contextual_frontier_training import resume_contextual_frontier
        result = resume_contextual_frontier(
            args.run,
            device=args.device,
            devices=args.devices,
            stop_after_epoch=args.stop_after_epoch,
            wall_clock_budget_seconds=args.wall_clock_budget_seconds,
            compact_history=args.compact_history,
        )
    elif args.command in ("resume-member-routing", "resume-single-distribution-member-routing"):
        from honf_runtime.contextual_frontier_training import resume_contextual_frontier
        result = resume_contextual_frontier(
            args.run, device=args.device, devices=args.devices,
            stop_after_epoch=args.stop_after_epoch, extend_horizon=args.extend_horizon,
            wall_clock_budget_seconds=args.wall_clock_budget_seconds,
            compact_history=args.compact_history,
        )
    elif args.command == "evaluate-joint-member-routing":
        from honf_runtime.joint_member_routing_evaluation import evaluate
        result = evaluate(args.checkpoint, args.device, args.output)
    elif args.command in ("evaluate-r13-6-member-routing",
                          "evaluate-r13-7-single-distribution-member-routing"):
        from honf_runtime.r13_6_member_routing_evaluation import evaluate
        result = evaluate(args.checkpoint, args.device, args.output)
    elif args.command == "evaluate-supplement-member-routing":
        from honf_runtime.supplement_member_routing import evaluate
        result = evaluate(args.checkpoint, args.device, args.output)
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
    elif args.command == "evaluate-contextual":
        if args.kind == "organization":
            from honf_runtime.compat import load_trusted_checkpoint
            from honf_runtime.contextual_frontier_evaluation import _contextual_checkpoint_parts
            payload = load_trusted_checkpoint(args.checkpoint, map_location="cpu")
            identity, _provider, _saved = _contextual_checkpoint_parts(payload)
            if identity.get("workflow") == "interaction_groups_joint_training_v1":
                raise ValueError("The contextual organization evaluator does not support interaction_groups_v1 checkpoints; use native, responses, or group-execution evaluation.")
        if args.kind == "group-execution":
            from honf_runtime.interaction_group_evaluation import evaluate_execution as evaluate
        elif args.kind == "group-spatial-diagnostic":
            from honf_runtime.spatial_interaction_group_evaluation import evaluate_legacy_group_diagnostic as evaluate
        elif args.kind == "spatial-groups":
            from honf_runtime.spatial_interaction_group_evaluation import evaluate_spatial_groups as evaluate
        elif args.kind == "native":
            from honf_runtime.contextual_frontier_evaluation import evaluate_native as evaluate
        elif args.kind == "responses":
            from honf_runtime.contextual_frontier_responses import evaluate_responses as evaluate
        else:
            from honf_runtime.contextual_frontier_organization import evaluate_organization as evaluate
        if args.kind == "spatial-groups":
            result = evaluate(
                args.checkpoint, args.device, args.output, include_cost=args.include_cost,
            )
        else:
            result = evaluate(args.checkpoint, args.device, args.output)
    elif args.command == "evaluate-member-routing":
        from honf_runtime.member_routing_evaluation import evaluate_member_routing
        result = evaluate_member_routing(args.checkpoint, args.device, args.output)
    else:
        result = research.assess(args.run, dataset=args.dataset)
    if isinstance(result, dict):
        print(json.dumps(result, sort_keys=True, default=str, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

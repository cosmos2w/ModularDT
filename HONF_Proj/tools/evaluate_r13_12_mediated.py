#!/usr/bin/env python
"""Evaluate literal R13-12 native checkpoints without fitting or solver calls."""
from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, help="New DATA2 evaluation directory")
    parser.add_argument("--task", choices=("native", "organization", "interventions", "cost"), required=True)
    parser.add_argument("--cases", nargs="+", help="Explicit fixed case IDs; native statistics stay inside checkpoint VAL membership")
    parser.add_argument("--modes", nargs="+", choices=("full", "all_e95", "source95_all", "restricted"))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cpu-smoke", action="store_true", help="Small cost API execution check, not benchmark evidence")
    parser.add_argument("--no-collective", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    import torch
    from honf_runtime.r13_12_mediated_evaluation import (
        benchmark_native, evaluate_interventions, evaluate_native, evaluate_organization,
    )
    torch.set_num_threads(args.threads)
    options = {}
    if args.modes is not None:
        options["modes"] = args.modes
    if args.cases is not None:
        if args.task == "cost":
            parser.error("Cost uses the inherited fixed low/high-M and TRAIN24 workloads.")
        options["case_ids"] = args.cases
    if args.task == "cost":
        options.update(warmups=args.warmups, repeats=args.repeats, cpu_smoke=args.cpu_smoke)
    if args.task == "interventions":
        options["collective"] = not args.no_collective
    function = {"native": evaluate_native, "organization": evaluate_organization,
                "interventions": evaluate_interventions, "cost": benchmark_native}[args.task]
    result = function(args.checkpoint, args.device, args.output, **options)
    print(json.dumps({"status": result["status"], "output": args.output,
                      "checkpoint_epoch": result["checkpoint_epoch"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

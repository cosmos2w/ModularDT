#!/usr/bin/env python3
"""Compare placement proposals using saved, frozen ThermalChannel predictors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from channelthermal.inverse.placement_benchmark import benchmark_placement
from channelthermal.member_routing import ThermalDetailAnchoredMemberRoutingAdapter
from honf_runtime.compat import load_trusted_checkpoint


def restore_checkpoint(path, device):
    saved = load_trusted_checkpoint(path, map_location="cpu")
    adapter = saved.get("adapter_state")
    if adapter is None:
        adapter = saved["experiment_identity"]["provider_identity"]["adapter_state"]
    model = ThermalDetailAnchoredMemberRoutingAdapter.from_state_payload(
        adapter, state_dict=saved["model_state_dict"], device=device,
    )
    return model.requires_grad_(False).eval(), saved.get("epoch")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--budget", type=int, required=True, help="Candidate scores per method and family.")
    parser.add_argument("--tau", type=float, required=True, help="Native material smooth-peak temperature scale.")
    parser.add_argument("--pressure-allowance", type=float, required=True, help="Maximum pressure rise from the baseline.")
    parser.add_argument("--j-resolution", type=float, required=True, help="Minimum resolved reference decrease in J.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("restricted", "full"), default="restricted")
    args = parser.parse_args(argv)
    panel = json.loads(args.panel.read_text())
    results = {}
    for item in args.checkpoint:
        label, separator, path = item.partition("=")
        if not separator or not label or not path or label in results:
            parser.error("Each checkpoint must use a distinct LABEL=PATH.")
        checkpoint = Path(path).expanduser().resolve(strict=True)
        model, epoch = restore_checkpoint(checkpoint, args.device)
        result = benchmark_placement(
            model, panel, budget=args.budget, pressure_allowance=args.pressure_allowance,
            tau=args.tau, resolution=args.j_resolution, seed=args.seed, device=args.device, mode=args.mode,
        )
        results[label] = {"checkpoint": str(checkpoint), "epoch": epoch, **result}
        del model
    print(json.dumps({"panel": str(args.panel.resolve()), "device": args.device, "checkpoints": results},
                     indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Bounded scalar-compiler allocation probe; no training or saved weights."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from honf_forward_core.interface_fields.routing_index.pair_join import (
    build_inverted_source_incidence,
    compile_two_hop_pairs,
)


def memory() -> dict[str, int]:
    torch.cuda.synchronize()
    stats = torch.cuda.memory_stats()
    return {key: stats[key] for key in (
        "allocated_bytes.all.current", "reserved_bytes.all.current",
        "allocated_bytes.all.peak", "inactive_split_bytes.all.current",
        "num_alloc_retries", "num_ooms",
    )}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for hubs in (1, 6, 12):
        gc.collect()
        torch.cuda.empty_cache()
        a = torch.full((48, 192, hubs), 1 / hubs, device="cuda", requires_grad=True)
        d = torch.full((48, 128, hubs), 1 / 192, device="cuda", requires_grad=True)
        weights = torch.ones((48, 192), device="cuda")
        incidence = build_inverted_source_incidence(a, source_weights=weights)
        torch.cuda.reset_peak_memory_stats()
        row = {"B": 48, "Q": 128, "E": 192, "K": hubs, "before": memory()}
        pairs = compile_two_hop_pairs(d, incidence, weights)
        row.update(raw_paths=pairs.raw_path_count, unique_pairs=pairs.unique_pair_count,
                   after_forward=memory())
        pairs.prior.square().sum().backward()
        row["after_backward"] = memory()
        row["gradients_finite"] = bool(torch.isfinite(a.grad).all() & torch.isfinite(d.grad).all())
        del pairs, incidence, a, d, weights
        gc.collect()
        row["after_release"] = memory()
        torch.cuda.empty_cache()
        row["after_empty_cache"] = memory()
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scope": "isolated scalar compiler, fully positive memberships; not a training or bandwidth sweep", "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()

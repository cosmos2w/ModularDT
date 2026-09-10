#!/usr/bin/env python3
"""Profile and validate the Wind Farm dataset.

Examples
--------
    python scripts/inspect_windfarm.py \
        --dataset-root Dataset/links/wind_farm

The command writes only compact, reproducible diagnostics.  It does not load
the convenience ``family_tensor.npz`` bundle and it does not write figures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SCRIPT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from windfarm.io import WindFarmDataset
from windfarm.profile import profile_dataset, write_profile_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root or family_volume directory (usually the workspace symlink).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_ROOT / "diagnostics" / "generated",
        help="Ignored diagnostics directory for JSON/CSV/NPZ outputs.",
    )
    parser.add_argument("--sample-points-per-run", type=int, default=256)
    parser.add_argument(
        "--full-scan-fields",
        action="store_true",
        help="Stream every value of U/p/k/epsilon; default samples evenly spaced points per run.",
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample_points_per_run < 0:
        raise SystemExit("--sample-points-per-run must be non-negative")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")
    dataset = WindFarmDataset(args.dataset_root)
    profile = profile_dataset(
        dataset,
        sample_points_per_run=args.sample_points_per_run,
        full_scan=args.full_scan_fields,
        chunk_size=args.chunk_size,
    )
    paths = write_profile_outputs(
        profile,
        args.output_dir,
        group_values=dataset.array("layout_index"),
        seed=args.seed,
    )
    print(f"runs={profile['dataset']['runs']}")
    print(f"layout_groups={profile['dataset']['layout_groups']}")
    print(f"directions={profile['dataset']['directions']}")
    print(f"total_cells={profile['dataset']['total_cells']}")
    print(f"validation_ok={profile['validation']['ok']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0 if profile["validation"]["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

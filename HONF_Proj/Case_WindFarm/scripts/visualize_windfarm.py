#!/usr/bin/env python3
"""Render deterministic 3-D cut-plane showcases for selected CFD cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SCRIPT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from windfarm.io import WindFarmDataset
from windfarm.visualize import (
    default_figure_path,
    render_case_showcase,
    select_representative_indices,
)


def _indices(value: str) -> list[int]:
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case indices must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one case index is required")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_ROOT / "diagnostics" / "generated",
        help="Ignored diagnostics directory for showcase figures.",
    )
    parser.add_argument(
        "--cases",
        type=_indices,
        default=None,
        help="Comma-separated source row indices; default is 0,middle,last.",
    )
    parser.add_argument("--count", type=int, default=3, help="Number of default representative cases.")
    parser.add_argument("--field", choices=("speed", "ux", "uy", "uz"), default="speed")
    parser.add_argument("--x-cut-m", type=float, default=0.0)
    parser.add_argument("--y-cut-m", type=float, default=0.0)
    parser.add_argument("--z-cut-m", type=float, default=70.0)
    parser.add_argument("--max-points", type=int, default=180)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--vmin", type=float, default=None, help="Optional shared lower color limit.")
    parser.add_argument("--vmax", type=float, default=None, help="Optional shared upper color limit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count <= 0 or args.max_points <= 0 or args.dpi <= 0:
        raise SystemExit("--count, --max-points, and --dpi must be positive")
    if (args.vmin is None) != (args.vmax is None):
        raise SystemExit("--vmin and --vmax must be supplied together")
    color_limits = None if args.vmin is None else (args.vmin, args.vmax)
    dataset = WindFarmDataset(args.dataset_root)
    compact = dataset.compact_metadata(allow_npz_fallback=True)
    turbine_xy = compact.get("turbine_xy_D")
    turbine_counts = compact.get("n_turbines")
    rotor_diameter = float(compact.get("D_m", 80.0))
    hub_height = float(compact.get("hub_height_m", args.z_cut_m))
    selected = args.cases if args.cases is not None else select_representative_indices(dataset, args.count)
    for index in selected:
        if not 0 <= index < dataset.n_runs:
            raise SystemExit(f"case index {index} is outside [0, {dataset.n_runs - 1}]")
    for index in selected:
        output = default_figure_path(args.output_dir, dataset, index)
        turbine_xy_m = None
        if turbine_xy is not None and turbine_counts is not None:
            turbine_xy_m = turbine_xy[index, : int(turbine_counts[index])] * rotor_diameter
        rendered = render_case_showcase(
            dataset,
            index,
            output,
            field=args.field,
            x_cut_m=args.x_cut_m,
            y_cut_m=args.y_cut_m,
            z_cut_m=args.z_cut_m,
            max_points=args.max_points,
            dpi=args.dpi,
            turbine_xy_m=turbine_xy_m,
            turbine_hub_height_m=hub_height,
            color_limits=color_limits,
        )
        run = dataset.run(index)
        print(
            f"case_index={index} case={run.case} layout_index={run.layout_index} "
            f"wind_direction_deg={run.wind_direction_deg:g} output={rendered}"
        )
    manifest = {
        "schema_version": 1,
        "dataset_root": str(dataset.volume_root),
        "field": args.field,
        "cuts_requested_m": {"x": args.x_cut_m, "y": args.y_cut_m, "z": args.z_cut_m},
        "max_points": args.max_points,
        "color_limits": None if color_limits is None else list(color_limits),
        "camera": {"elevation_deg": 25, "azimuth_deg": -55},
        "cases": [
            {
                "index": int(index),
                "case": dataset.run(index).case,
                "layout_index": dataset.run(index).layout_index,
                "wind_direction_deg": dataset.run(index).wind_direction_deg,
                "cuts_actual_m": {
                    "x": float(dataset.run(index).x_m[abs(dataset.run(index).x_m - args.x_cut_m).argmin()]),
                    "y": float(dataset.run(index).y_m[abs(dataset.run(index).y_m - args.y_cut_m).argmin()]),
                    "z": float(dataset.run(index).z_m[abs(dataset.run(index).z_m - args.z_cut_m).argmin()]),
                },
                "turbines_overlaid": bool(turbine_xy is not None and turbine_counts is not None),
                "output": str(default_figure_path(args.output_dir, dataset, index)),
            }
            for index in selected
        ],
    }
    manifest_path = args.output_dir.expanduser().resolve() / "windfarm_visualizations.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

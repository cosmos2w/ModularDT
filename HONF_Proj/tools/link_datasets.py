#!/usr/bin/env python3
"""Create optional browsing symlinks for configured dataset resources."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from honf_runtime.paths import PROJECT_ROOT, resolve_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--locations",
        default="project://Case_ThermalChannel/Dataset/dataset_locations.local.json",
    )
    parser.add_argument("--replace", action="store_true", help="Replace links previously created by this tool.")
    args = parser.parse_args()
    location_path = resolve_path(args.locations)
    with location_path.open("r", encoding="utf-8") as stream:
        locations = json.load(stream)
    link_root = PROJECT_ROOT / "Case_ThermalChannel" / "Dataset" / "links"
    link_root.mkdir(parents=True, exist_ok=True)
    for dataset_id, raw_path in locations.items():
        target = resolve_path(raw_path)
        if not target.is_file():
            raise FileNotFoundError(f"Dataset target does not exist: {target}")
        link = link_root / f"{dataset_id}.h5"
        if link.is_symlink() and args.replace:
            link.unlink()
        elif link.exists() or link.is_symlink():
            raise FileExistsError(f"Link already exists: {link}; pass --replace to refresh it.")
        # A relative target keeps links readable when source and data trees move
        # together, while the loader remains independent of these conveniences.
        link.symlink_to(Path(os.path.relpath(target, start=link.parent)))
        print(f"{dataset_id}: {link} -> {link.readlink()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

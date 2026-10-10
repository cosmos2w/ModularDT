"""Plan or prebuild geometry-only Wind TRAIN/VALID catalogues on the CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_WindFarm/src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
from windfarm.data import WindFarmNativeView
from windfarm.training.unified_task import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DERIVED_ROOT,
    DEFAULT_PILOT_ROOT,
    _load_original_split,
    _read_fixed_rows,
)
from windfarm.workflows.joint_forward import NativeRoleCatalogueCache
from windfarm.workflows.native_role_cache import training_catalogue_cache_directory


def select_rows(view, split, scope: str, manifest_path: Path):
    """Reuse sealed memberships; never include original TEST geometry or targets."""
    if scope == "full":
        train, validation = np.asarray(split.train), np.asarray(split.validation)
        if train.shape != (420,) or validation.shape != (90,):
            raise ValueError("Full Wind catalogue requires original TRAIN420/VALID90.")
        manifest_sha256 = None
    elif scope == "segmented":
        manifest, train, validation = _read_fixed_rows(view, split, manifest_path)
        manifest_sha256 = manifest["manifest_sha256"]
    else:
        raise ValueError("Wind catalogue scope must be full or segmented.")
    rows = np.concatenate((train, validation)).astype(np.int64)
    if np.unique(rows).size != rows.size or set(rows) & set(split.test):
        raise ValueError("Wind catalogue selection overlaps itself or protected TEST.")
    if not set(train).issubset(set(split.train)) or not set(validation).issubset(set(split.validation)):
        raise ValueError("Wind catalogue rows escaped the original source partitions.")
    groups = np.asarray(view.metadata["layout_index"])
    if set(groups[rows]) & set(groups[split.test]):
        raise ValueError("Wind catalogue layouts overlap protected TEST.")
    return rows, {
        "scope": scope,
        "train_row_count": len(train),
        "validation_row_count": len(validation),
        "row_indices_sha256": hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest(),
        "subset_manifest_sha256": manifest_sha256,
        "test_rows_prepared": 0,
        "target_values_read": False,
    }


def prepare_rows(view, rows, cache, *, progress=None):
    """Touch only geometry arrays; do not sample queries, targets or train state."""
    payload_bytes = 0
    for number, row in enumerate(rows, start=1):
        case = view.run(int(row))
        catalogue = cache.get(case)
        payload_bytes += catalogue.cached_nbytes
        if progress is not None:
            progress(number, int(row), len(rows))
        del catalogue, case
    return {"selected_payload_array_bytes": payload_bytes, **cache.summary()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("full", "segmented"), required=True)
    parser.add_argument(
        "--prepare", action="store_true", help="Build missing geometry entries; otherwise only show the selection."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_PILOT_ROOT / "fixed_subset_manifest.json")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    directory = args.cache_dir or training_catalogue_cache_directory()
    if directory is None:
        parser.error("Persistent cache is disabled; specify --cache-dir or unset HONF_WIND_CATALOGUE_CACHE_DIR=off.")
    view = WindFarmNativeView(args.data_root, allow_npz_metadata_fallback=True)
    split = _load_original_split(view, args.derived_root)
    rows, result = select_rows(view, split, args.scope, args.manifest_path)
    result.update({"cache_directory": str(directory.expanduser().resolve()), "prepared": args.prepare})
    if args.prepare:
        started = time.perf_counter()
        cache = NativeRoleCatalogueCache(max_cached_bytes=1, persistent_dir=directory)
        result.update(
            prepare_rows(
                view,
                rows,
                cache,
                progress=lambda number, row, total: print(
                    f"Geometry catalogue {number}/{total}: native row {row}",
                    file=sys.stderr,
                    flush=True,
                ),
            )
        )
        result["preparation_seconds"] = time.perf_counter() - started
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

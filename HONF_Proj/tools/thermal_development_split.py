#!/usr/bin/env python3
"""Create one immutable input-stratified native Thermal development manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / "src", PROJECT / "Case_ThermalChannel/src"):
    sys.path.insert(0, str(directory))

from channelthermal.data.development_split import (
    build_development_manifest,
    validate_development_manifest,
)

from honf_runtime.compat import resolve_demo_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--spec", type=Path, default=PROJECT / "src/config_core/data/thermal_development25_v1.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    spec = json.loads(args.spec.read_text())
    if spec.get("schema_version") != 1 or set(spec) != {"schema_version", "policy"}:
        parser.error("Selection spec must contain only schema_version=1 and policy.")
    manifest = build_development_manifest(resolve_demo_path(args.dataset), policy=spec["policy"])
    validate_development_manifest(manifest, args.dataset, expected_fingerprint=manifest["manifest_sha256"])
    if not args.dry_run:
        if args.output is None:
            parser.error("--output is required unless --dry-run is set.")
        output = args.output.expanduser().resolve()
        repository = PROJECT.parent
        if output == repository or repository in output.parents:
            parser.error("Generated case-ID manifests must be outside the source project; do not upload generated data.")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        output.chmod(0o444)
    print(json.dumps({"manifest_path": str(args.output.resolve()) if args.output else None,
                      "manifest_sha256": manifest["manifest_sha256"], "policy": manifest["policy"],
                      "source": manifest["source"], "partitions": manifest["partitions"],
                      "scope": manifest["scope"], "training_launched": False}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

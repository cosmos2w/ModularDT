#!/usr/bin/env python3
"""Validate a named ThermalChannel dataset against its committed manifest."""

from __future__ import annotations

import argparse

from channelthermal.resources import DatasetRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_id", nargs="?", help="Logical dataset ID (positional compatibility form).")
    parser.add_argument("--dataset-id", dest="dataset_id_option", help="Logical dataset ID.")
    parser.add_argument("--sha256", action="store_true", help="Read the full file and verify its SHA-256 digest.")
    args = parser.parse_args()
    dataset_id = args.dataset_id_option or args.dataset_id
    if not dataset_id:
        parser.error("provide DATASET_ID or --dataset-id DATASET_ID")
    if args.dataset_id_option and args.dataset_id and args.dataset_id_option != args.dataset_id:
        parser.error("positional DATASET_ID and --dataset-id disagree")
    registry = DatasetRegistry(
        "project://Case_ThermalChannel/Dataset/dataset_manifest.json",
        "project://Case_ThermalChannel/Dataset/dataset_locations.local.json",
    )
    resource = registry.resolve(dataset_id)
    registry.validate(resource, verify_sha256=args.sha256)
    print(f"[ok] {resource.dataset_id}: {resource.path}")
    print(f"schema={resource.record.get('schema')} sha256={resource.fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

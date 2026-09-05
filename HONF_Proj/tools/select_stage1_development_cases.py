#!/usr/bin/env python3
"""Select a deterministic 20-case development subset from physical geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from channelthermal.data.datasets import GlobalChannelThermalDataset
from honf_runtime.compat import resolve_demo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--anchor", action="append", default=["0273", "0653"])
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def geometry_descriptor(sample: dict) -> np.ndarray:
    structure = sample["structure"]
    present = np.asarray(structure["module_present"]) > 0.5
    centers = np.asarray(structure["module_centers"], dtype=np.float64)[present]
    count = len(centers)
    if count <= 1:
        return np.asarray([count, 12.0, 12.0, 0.0, 0.0], dtype=np.float64)
    distance = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    pairs = distance[np.triu_indices(count, 1)]
    nearest = np.where(np.eye(count, dtype=bool), np.inf, distance).min(axis=1)
    spread = np.sqrt(np.mean(np.sum((centers - centers.mean(axis=0)) ** 2, axis=-1)))
    close_fraction = np.mean(pairs < 2.5)
    return np.asarray([count, np.min(nearest), np.median(pairs), spread, close_fraction], dtype=np.float64)


def main() -> int:
    args = parse_args()
    dataset = GlobalChannelThermalDataset(
        args.dataset,
        split=args.split,
        points_per_case=1,
        normalize_inputs=False,
        normalize_targets=False,
        random_point_sampling=False,
    )
    case_ids = [str(value) for value in dataset.selected_case_ids]
    descriptors = np.stack([geometry_descriptor(dataset[index]) for index in range(len(dataset))])
    median = np.median(descriptors, axis=0)
    scale = np.quantile(descriptors, 0.75, axis=0) - np.quantile(descriptors, 0.25, axis=0)
    normalized = (descriptors - median) / np.maximum(scale, 1.0e-8)
    selected = []
    for anchor in args.anchor:
        if anchor not in case_ids:
            raise KeyError(f"Anchor case {anchor!r} is absent from split {args.split!r}.")
        index = case_ids.index(anchor)
        if index not in selected:
            selected.append(index)
    # Seed coverage with one geometrically central case per available module
    # count, then greedily maximize distance from the selected descriptor set.
    for module_count in sorted({int(value) for value in descriptors[:, 0]}):
        candidates = np.flatnonzero(descriptors[:, 0] == module_count)
        center = np.median(normalized[candidates], axis=0)
        index = int(candidates[np.argmin(np.linalg.norm(normalized[candidates] - center, axis=1))])
        if index not in selected and len(selected) < int(args.count):
            selected.append(index)
    while len(selected) < int(args.count):
        remaining = np.asarray([index for index in range(len(dataset)) if index not in selected])
        nearest_distance = np.min(
            np.linalg.norm(normalized[remaining, None, :] - normalized[np.asarray(selected)][None, :, :], axis=-1),
            axis=1,
        )
        selected.append(int(remaining[np.argmax(nearest_distance)]))
    selected = selected[: int(args.count)]
    names = ["active_module_count", "minimum_nearest_spacing", "median_pair_spacing", "layout_spread", "close_pair_fraction"]
    payload = {
        "selection": "physical_descriptor_maximin",
        "split": args.split,
        "count": len(selected),
        "anchors": list(dict.fromkeys(args.anchor)),
        "descriptor_names": names,
        "case_ids": [case_ids[index] for index in selected],
        "cases": [
            {
                "case_id": case_ids[index],
                "dataset_index": index,
                "descriptors": {name: float(value) for name, value in zip(names, descriptors[index])},
            }
            for index in selected
        ],
    }
    output = resolve_demo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

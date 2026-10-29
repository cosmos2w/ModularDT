"""Fixed input-stratified development subsets of existing native partitions.

Only catalogue and small physical-input arrays are read. Generated manifests
belong outside Git; their semantic fingerprint binds the selection policy,
source metadata and exact membership, independently of file placement.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from honf_runtime.compat import decode_string_array, resolve_demo_path

SCHEMA_VERSION = 1
DEFAULT_POLICY = {
    "seed": 20261004,
    "fraction": 0.25,
    "excluded_test_case_ids": ["0273"],
    "re_band_bounds": [70.0, 130.0],
    "heat_strata": 3,
}


def _plain(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_case_catalog(dataset_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read original membership and physical input descriptors, never targets."""
    path = resolve_demo_path(dataset_path).resolve()
    before = path.stat()
    records = []
    with h5py.File(path, "r") as h5:
        ids = decode_string_array(h5["case_ids"][...])
        splits = decode_string_array(h5["splits"][...])
        if len(ids) != len(splits) or len(ids) != len(set(ids)):
            raise ValueError("Native case IDs/splits must be aligned and unique.")
        root_metadata = _plain(dict(h5.attrs))
        for case_id, split in zip(ids, splits):
            group = h5["cases"][case_id]
            present = np.asarray(group["module_present"][...]) > 0.5
            heat = np.asarray(group["heat_powers"][...])
            centers = np.asarray(group["module_centers"][...])
            material = _plain(dict(group["material_parameters"].attrs))
            re = float(material["re"])
            total = float(np.asarray(heat[present], dtype=np.float64).sum())
            if not present.any() or not np.isfinite(heat).all() or not np.isfinite(centers).all() or not math.isfinite(re):
                raise ValueError(f"Invalid physical selection metadata for case {case_id}.")
            records.append({"case_id": case_id, "split": split.lower(), "module_count": int(present.sum()),
                            "re": re, "total_heat": total,
                            "input_sha256": _digest({"present": present, "heat": heat, "centers": centers, "material": material})})
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("Native dataset changed while reading selection metadata.")
    source = {"dataset_path": str(path), "dataset_id": root_metadata.get("dataset_id"),
              "catalog_checksum": {key: root_metadata[key] for key in ("catalog_checksum", "catalog_sha256", "dataset_fingerprint") if key in root_metadata},
              "metadata_sha256": _digest({"root_metadata": root_metadata, "records": records}),
              "stat": {"size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns}}
    return source, records


def _policy(settings: Mapping[str, Any] | None) -> dict[str, Any]:
    policy = {**DEFAULT_POLICY, **dict(settings or {})}
    if set(policy) != set(DEFAULT_POLICY):
        raise ValueError("Unknown development selection policy fields.")
    if isinstance(policy["seed"], bool) or not isinstance(policy["seed"], int) or policy["seed"] < 0:
        raise ValueError("Development seed must be a nonnegative integer.")
    fraction = float(policy["fraction"])
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Development fraction must lie in (0,1].")
    bounds = [float(value) for value in policy["re_band_bounds"]]
    if not bounds or not all(math.isfinite(value) for value in bounds) or bounds != sorted(set(bounds)):
        raise ValueError("Re band bounds must be finite, unique and increasing.")
    heat_strata = policy["heat_strata"]
    if isinstance(heat_strata, bool) or not isinstance(heat_strata, int) or heat_strata < 1:
        raise ValueError("Heat strata must be a positive integer.")
    excluded = policy["excluded_test_case_ids"]
    if isinstance(excluded, str) or not all(isinstance(value, str) for value in excluded) or len(set(excluded)) != len(excluded):
        raise ValueError("Excluded test IDs must be a unique sequence of strings.")
    return {"seed": policy["seed"], "fraction": fraction, "excluded_test_case_ids": sorted(excluded),
            "re_band_bounds": bounds, "heat_strata": heat_strata}


def _allocate(counts: Sequence[int], target: int, *, minimum: bool, rng: np.random.Generator | None = None) -> list[int]:
    """Bounded largest-remainder allocation, with coverage where affordable."""
    if not counts or target < 0 or target > sum(counts):
        raise ValueError("Invalid stratum allocation.")
    ideal = np.asarray(counts, dtype=np.float64) * target / sum(counts)
    allocation = np.floor(ideal).astype(int)
    lower = np.ones(len(counts), dtype=int) if minimum and target >= len(counts) else np.zeros(len(counts), dtype=int)
    allocation = np.maximum(allocation, lower)
    tie = np.arange(len(counts)) if rng is None else rng.permutation(len(counts))
    priority = {int(index): rank for rank, index in enumerate(tie)}
    while int(allocation.sum()) != target:
        add = int(allocation.sum()) < target
        eligible = [i for i, n in enumerate(counts) if allocation[i] < n] if add else [i for i in range(len(counts)) if allocation[i] > lower[i]]
        best = min(eligible, key=lambda i: (-(ideal[i] - allocation[i]) if add else -(allocation[i] - ideal[i]), priority[i]))
        allocation[best] += 1 if add else -1
    return allocation.tolist()


def _choose(records: Sequence[dict[str, Any]], target: int, policy: Mapping[str, Any], rng: np.random.Generator) -> tuple[list[str], dict[str, Any]]:
    modules = sorted({record["module_count"] for record in records})
    if target < len(modules):
        raise ValueError("Development target cannot retain every module-count stratum.")
    groups = [[record for record in records if record["module_count"] == m] for m in modules]
    quotas = _allocate([len(group) for group in groups], target, minimum=True)
    selected, strata = [], []
    for m, group, quota in zip(modules, groups, quotas):
        bands = defaultdict(list)
        for record in group:
            bands[int(np.searchsorted(policy["re_band_bounds"], record["re"], side="left"))].append(record)
        band_ids = sorted(bands)
        band_quotas = _allocate([len(bands[b]) for b in band_ids], quota, minimum=True, rng=rng)
        for band, band_quota in zip(band_ids, band_quotas):
            ordered = sorted(bands[band], key=lambda record: (record["total_heat"], record["case_id"]))
            heat_groups = [list(indices) for indices in np.array_split(np.arange(len(ordered)), min(policy["heat_strata"], len(ordered)))]
            heat_quotas = _allocate([len(indices) for indices in heat_groups], band_quota, minimum=True, rng=rng)
            for heat_bin, indices, heat_quota in zip(range(len(heat_groups)), heat_groups, heat_quotas):
                chosen = rng.choice(indices, size=heat_quota, replace=False)
                selected.extend(ordered[int(index)]["case_id"] for index in chosen)
                strata.append({"module_count": m, "re_band": band, "heat_rank_bin": heat_bin, "eligible": len(indices), "selected": heat_quota})
    return sorted(selected), {"module_quotas": {str(m): q for m, q in zip(modules, quotas)}, "strata": strata}


def manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Hash semantic content; relocation/stat provenance is validated separately."""
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    source = dict(payload["source"])
    source.pop("dataset_path", None)
    source.pop("stat", None)
    payload["source"] = source
    return _digest(payload)


def build_development_manifest(dataset_path: str | Path, *, policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Generate reproducible without-replacement membership from input metadata."""
    source, records = read_case_catalog(dataset_path)
    policy = _policy(policy)
    original = {split: [record for record in records if record["split"] == split] for split in ("train", "test")}
    if sum(map(len, original.values())) != len(records) or not all(original.values()):
        raise ValueError("Development selection requires original nonempty train/test partitions.")
    excluded = set(policy["excluded_test_case_ids"])
    if not excluded <= {record["case_id"] for record in original["test"]}:
        raise ValueError("Excluded test IDs must belong to the original test partition.")
    partitions = {}
    for offset, split in enumerate(("train", "test")):
        eligible = [record for record in original[split] if split != "test" or record["case_id"] not in excluded]
        target = math.floor(len(eligible) * policy["fraction"] + 0.5)
        chosen, summary = _choose(eligible, target, policy, np.random.default_rng(np.random.SeedSequence([policy["seed"], offset])))
        partitions[split] = {"original_count": len(original[split]), "eligible_count": len(eligible), "selected_count": len(chosen),
                             "fraction_of_eligible": len(chosen) / len(eligible), "fraction_of_original": len(chosen) / len(original[split]),
                             "case_ids": chosen, **summary}
    manifest = {"schema_version": SCHEMA_VERSION, "scope": "fixed_input_stratified_development_only",
                "selection_algorithm": "module_largest_remainder_then_Re_bands_then_heat_rank_strata_v1",
                "normalization_scope": "selected_original_training_cases_only", "policy": policy, "source": source, "partitions": partitions}
    manifest["manifest_sha256"] = manifest_fingerprint(manifest)
    return manifest


def validate_development_manifest(manifest: Mapping[str, Any], dataset_path: str | Path, *, expected_fingerprint: str | None = None) -> dict[str, Any]:
    """Bind membership and source provenance, including path and size/mtime.

    A copied/relocated dataset needs an explicit regenerated manifest binding;
    the relocation-independent semantic SHA alone never bypasses stat checks.
    """
    value = json.loads(json.dumps(manifest, allow_nan=False))
    if value.get("schema_version") != SCHEMA_VERSION or value.get("manifest_sha256") != manifest_fingerprint(value):
        raise ValueError("Development manifest schema/fingerprint mismatch.")
    if expected_fingerprint is not None and value["manifest_sha256"] != expected_fingerprint:
        raise ValueError("Development manifest differs from the configured fingerprint.")
    regenerated = build_development_manifest(dataset_path, policy=value["policy"])
    if value != regenerated:
        raise ValueError("Development manifest membership or native dataset metadata/stat differs from deterministic regeneration.")
    train, test = (set(value["partitions"][split]["case_ids"]) for split in ("train", "test"))
    if train & test:
        raise ValueError("Development train/test memberships overlap.")
    return value


def load_development_manifest(path: str | Path, dataset_path: str | Path, expected_fingerprint: str | None = None) -> dict[str, Any]:
    return validate_development_manifest(json.loads(resolve_demo_path(path).read_text()), dataset_path, expected_fingerprint=expected_fingerprint)


def development_case_ids(manifest: Mapping[str, Any], split: str) -> tuple[str, ...]:
    split = "test" if split.lower() == "val" else split.lower()
    if split not in ("train", "test"):
        raise ValueError("Development subsets only preserve original train/test partitions.")
    return tuple(manifest["partitions"][split]["case_ids"])


def resolve_development_manifest(dataset_config: Mapping[str, Any], dataset_path: str | Path, *, allow_embedded: bool = False) -> dict[str, Any] | None:
    """Training requires the fixed external file; readers may use saved binding."""
    path = dataset_config.get("development_manifest")
    embedded = dataset_config.get("development_subset")
    expected = dataset_config.get("development_manifest_sha256")
    if not path and embedded is None and expected is None:
        return None
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Explicit development selection requires its fixed semantic SHA256.")
    if path and resolve_demo_path(path).is_file():
        result = load_development_manifest(path, dataset_path, expected)
        if embedded is not None and result != validate_development_manifest(embedded, dataset_path, expected_fingerprint=expected):
            raise ValueError("External and checkpoint development memberships differ.")
        return result
    if allow_embedded and embedded is not None:
        return validate_development_manifest(embedded, dataset_path, expected_fingerprint=expected)
    raise ValueError("Training/resume requires the original fixed development manifest file.")

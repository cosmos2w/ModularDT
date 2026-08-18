"""Strict case-owned HDF5 I/O for hierarchical inverse data.

Physical design ``D``, context ``c``, request variants ``R``, target compact
plan ``G``, and later verified ``G_hat`` share the reusable contracts, while
HDF5 remains a ThermalChannel dependency rather than a core dependency.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
from torch.utils.data import Dataset


INVERSE_DATASET_SCHEMA_NAME = "thermalchannel_hierarchical_inverse_dataset"
INVERSE_DATASET_SCHEMA_VERSION = 1
TOPOLOGY_SET_DATASET_SCHEMA_NAME = "thermalchannel_hierarchical_inverse_topology_set"
TOPOLOGY_SET_DATASET_SCHEMA_VERSION = 1


def _string_dtype() -> np.dtype[Any]:
    return h5py.string_dtype(encoding="utf-8")


def _write_tree(group: h5py.Group, values: Mapping[str, Any]) -> None:
    for name, value in values.items():
        if isinstance(value, Mapping):
            _write_tree(group.create_group(str(name)), value)
            continue
        array = np.asarray(value)
        if array.dtype.kind in {"U", "O"}:
            data = np.asarray(array, dtype=object)
            group.create_dataset(str(name), data=data, dtype=_string_dtype())
            continue
        compression = "gzip" if array.ndim > 0 and array.size >= 64 else None
        group.create_dataset(
            str(name),
            data=array,
            compression=compression,
            shuffle=bool(compression),
        )


def write_inverse_hdf5_atomic(
    path: str | Path,
    *,
    arrays: Mapping[str, Any],
    attributes: Mapping[str, Any],
) -> Path:
    """Write one artifact to a sibling temporary file and atomically replace."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    try:
        with h5py.File(temporary, "w") as h5:
            for name, value in attributes.items():
                if isinstance(value, (dict, list, tuple)):
                    h5.attrs[str(name)] = json.dumps(value, sort_keys=True)
                else:
                    h5.attrs[str(name)] = value
            _write_tree(h5, arrays)
            h5.flush()
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    validate_inverse_hdf5(destination)
    return destination


def _decode(values: Any) -> list[str]:
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in np.asarray(values)]


def validate_inverse_hdf5(path: str | Path) -> dict[str, Any]:
    """Validate required schema, shapes, finite data, and split disjointness."""

    artifact = Path(path).expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"Inverse dataset not found: {artifact}")
    with h5py.File(artifact, "r") as h5:
        schema_name = str(h5.attrs.get("schema_name", ""))
        topology_set = schema_name == TOPOLOGY_SET_DATASET_SCHEMA_NAME
        if schema_name not in {INVERSE_DATASET_SCHEMA_NAME, TOPOLOGY_SET_DATASET_SCHEMA_NAME}:
            raise ValueError("Inverse dataset schema_name mismatch.")
        expected_version = (
            TOPOLOGY_SET_DATASET_SCHEMA_VERSION if topology_set else INVERSE_DATASET_SCHEMA_VERSION
        )
        if int(h5.attrs.get("schema_version", 0)) != expected_version:
            raise ValueError("Inverse dataset schema_version mismatch.")
        required = {
            "case_id", "source_split", "inverse_split", "variant_id", "variant_seed",
            "design", "context", "functionals", "requests", "geometry", "normalization", "splits",
        }
        required.add("topology" if topology_set else "plan")
        missing = sorted(required - set(h5))
        if missing:
            raise ValueError(f"Inverse dataset is missing root entries: {missing}")
        case_ids = _decode(h5["case_id"][...])
        splits = _decode(h5["inverse_split"][...])
        n = len(case_ids)
        variants = int(h5.attrs["variants_per_case"])
        if n != int(h5.attrs["num_cases"]) or len(set(case_ids)) != n:
            raise ValueError("Inverse dataset case IDs/count are inconsistent.")
        if h5["variant_seed"].shape != (n, variants):
            raise ValueError("variant_seed must have shape [N,V].")
        if h5["requests/active_mask"].shape != (n, variants, 4):
            raise ValueError("requests/active_mask must have shape [N,V,4].")
        if topology_set:
            if str(h5.attrs.get("topology_signature_schema_name", "")) != "honf_topology_signature":
                raise ValueError("Topology-set dataset requires honf_topology_signature provenance.")
            if int(h5.attrs.get("topology_signature_schema_version", 0)) != 3:
                raise ValueError("Topology-set dataset requires topology signature schema version 3.")
            checkpoint_sha256 = str(h5.attrs.get("forward_topology_checkpoint_sha256", ""))
            if len(checkpoint_sha256) != 64 or any(
                character not in "0123456789abcdefABCDEF" for character in checkpoint_sha256
            ):
                raise ValueError("Topology-set dataset requires a forward checkpoint SHA-256.")
            tokens = h5["topology/tokens_raw"]
            normalized_tokens = h5["topology/tokens_normalized"]
            active_mask = h5["topology/active_mask"]
            relations = h5["topology/relations"]
            if tokens.shape[:1] != (n,) or tokens.shape[-1] != 12 or normalized_tokens.shape != tokens.shape:
                raise ValueError("Topology tokens must share shape [N,K,12].")
            if active_mask.shape != tokens.shape[:2]:
                raise ValueError("topology/active_mask must have shape [N,K].")
            if relations.ndim != 4 or relations.shape[:3] != (n, tokens.shape[1], tokens.shape[1]):
                raise ValueError("topology/relations must have shape [N,K,K,F_r].")
            if not np.array_equal((tokens[..., 0] > 0.5), (active_mask[...] > 0.5)):
                raise ValueError("Topology token activity and active_mask disagree.")
            plan_raw_path = "topology/tokens_raw"
            plan_normalized_path = "topology/tokens_normalized"
        else:
            if h5["plan/compact_raw"].shape[:1] != (n,) or h5["plan/compact_raw"].shape[-1] != 12:
                raise ValueError("plan/compact_raw must have shape [N,K,12].")
            plan_raw_path = "plan/compact_raw"
            plan_normalized_path = "plan/compact_normalized"
        if h5["design/model/module_present"].shape[0] != n:
            raise ValueError("Design case dimension mismatch.")
        for name in (
            plan_raw_path, plan_normalized_path, "context/vector",
            "context/normalized_vector", "requests/target_normalized",
            "design/model/normalized_module_centers", "design/model/normalized_heat_powers",
        ):
            if not np.isfinite(h5[name][...]).all():
                raise ValueError(f"Inverse dataset contains non-finite values in {name}.")
        split_sets = {
            name: {case_ids[index] for index, split in enumerate(splits) if split == name}
            for name in ("train", "validation", "test")
        }
        if any(split_sets[a] & split_sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise ValueError("Inverse dataset has case-ID split leakage.")
        return {
            "num_cases": n,
            "variants_per_case": variants,
            "num_edges": int(h5[plan_raw_path].shape[1]),
            "max_modules": int(h5["design/model/module_present"].shape[1]),
            "split_counts": {name: len(values) for name, values in split_sets.items()},
            "plan_token_mode": "exchangeable_set" if topology_set else "indexed",
            "topology_schema_name": (
                str(h5.attrs["topology_signature_schema_name"]) if topology_set else ""
            ),
            "topology_schema_version": (
                int(h5.attrs["topology_signature_schema_version"]) if topology_set else 0
            ),
            "forward_topology_checkpoint_sha256": (
                str(h5.attrs["forward_topology_checkpoint_sha256"]) if topology_set else ""
            ),
        }


class InverseH5Dataset(Dataset):
    """Flatten a strict case-major inverse artifact into `(case, variant)` rows."""

    REQUEST_NAMES = (
        "type_id", "relation_id", "target_normalized", "target_mask",
        "tolerance_normalized", "range_normalized", "range_mask", "priority",
        "weight", "region", "region_mask", "active_mask",
    )

    def __init__(self, path: str | Path, *, split: str = "train", preload: bool = True) -> None:
        self.path = Path(path).expanduser().resolve()
        self.summary = validate_inverse_hdf5(self.path)
        self.split = str(split)
        if self.split not in {"train", "validation", "test", "all"}:
            raise ValueError(f"Unknown inverse split: {self.split!r}")
        self._h5: h5py.File | None = None
        self._arrays: dict[str, np.ndarray] | None = None
        with h5py.File(self.path, "r") as h5:
            inverse_splits = _decode(h5["inverse_split"][...])
            variants = int(h5.attrs["variants_per_case"])
            case_indices = [
                index for index, value in enumerate(inverse_splits)
                if self.split == "all" or value == self.split
            ]
            self.records = tuple((case_index, variant) for case_index in case_indices for variant in range(variants))
            self.request_schema_version = int(h5.attrs["request_schema_version"])
            self.compact_plan_schema_version = int(h5.attrs["compact_plan_schema_version"])
            self.dataset_hash = str(h5.attrs.get("dataset_hash", ""))
            self.plan_token_mode = str(self.summary["plan_token_mode"])
            self.plan_path = (
                "topology/tokens_normalized"
                if self.plan_token_mode == "exchangeable_set"
                else "plan/compact_normalized"
            )
            if preload:
                paths = (
                    "context/normalized_vector", "context/vector",
                    "geometry/constraint_normalized", "geometry/constraint_raw",
                    "geometry/constraint_mask", self.plan_path,
                    "design/model/normalized_module_centers",
                    "design/model/normalized_heat_powers",
                    "design/model/module_present", "design/module_count",
                    *(f"requests/{name}" for name in self.REQUEST_NAMES),
                )
                self._arrays = {name: h5[name][...] for name in paths}

    @property
    def h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_h5"] = None
        return state

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> dict[str, Any]:
        case_index, variant_index = self.records[int(item)]
        arrays = self._arrays

        def read(name: str, *indices: int) -> Any:
            source = arrays[name] if arrays is not None else self.h5[name]
            return source[indices]

        return {
            "case_index": np.asarray(case_index, dtype=np.int64),
            "variant_index": np.asarray(variant_index, dtype=np.int64),
            "context": read("context/normalized_vector", case_index).astype(np.float32),
            "context_raw": read("context/vector", case_index).astype(np.float32),
            "geometry_constraints": read("geometry/constraint_normalized", case_index, variant_index).astype(np.float32),
            "geometry_constraints_raw": read("geometry/constraint_raw", case_index, variant_index).astype(np.float32),
            "geometry_constraint_mask": read("geometry/constraint_mask", case_index, variant_index).astype(np.float32),
            "request": {
                name: read(f"requests/{name}", case_index, variant_index)
                for name in self.REQUEST_NAMES
            },
            "plan": read(self.plan_path, case_index).astype(np.float32),
            "layout": np.concatenate(
                [
                    read("design/model/normalized_module_centers", case_index).astype(np.float32),
                    read("design/model/normalized_heat_powers", case_index).astype(np.float32)[:, None],
                ],
                axis=-1,
            ),
            "module_present": read("design/model/module_present", case_index).astype(np.float32),
            "module_count": np.asarray(read("design/module_count", case_index), dtype=np.int64),
        }


__all__ = [
    "INVERSE_DATASET_SCHEMA_NAME",
    "INVERSE_DATASET_SCHEMA_VERSION",
    "TOPOLOGY_SET_DATASET_SCHEMA_NAME",
    "TOPOLOGY_SET_DATASET_SCHEMA_VERSION",
    "InverseH5Dataset",
    "validate_inverse_hdf5",
    "write_inverse_hdf5_atomic",
]

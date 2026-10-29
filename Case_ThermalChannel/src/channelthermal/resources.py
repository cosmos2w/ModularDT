"""Resolve and validate named ThermalChannel dataset resources."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py

from honf_runtime.paths import PROJECT_ROOT, resolve_path


@dataclass(frozen=True)
class DatasetResource:
    """One resolved dataset plus its committed manifest record."""

    dataset_id: str
    path: Path
    record: Mapping[str, Any]

    @property
    def fingerprint(self) -> str:
        return str(self.record.get("sha256", ""))


class DatasetRegistry:
    """Dataset-ID resolver backed by a committed manifest and local map."""

    def __init__(
        self,
        manifest_path: str | Path,
        locations_path: str | Path | None,
        *,
        project_root: str | Path = PROJECT_ROOT,
    ):
        self.project_root = Path(project_root).resolve()
        self.manifest_path = resolve_path(manifest_path, project_root=self.project_root)
        with self.manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if int(manifest.get("schema_version", 0)) != 1:
            raise ValueError(f"Unsupported dataset manifest schema: {self.manifest_path}")
        self.records = dict(manifest.get("datasets", {}))

        self.locations: dict[str, str] = {}
        if locations_path:
            resolved_locations = resolve_path(locations_path, project_root=self.project_root)
            if resolved_locations.exists():
                with resolved_locations.open("r", encoding="utf-8") as stream:
                    payload = json.load(stream)
                if not isinstance(payload, dict):
                    raise TypeError(f"Dataset location map must be an object: {resolved_locations}")
                self.locations = {str(key): str(value) for key, value in payload.items()}

    @classmethod
    def from_case_config(cls, dataset_config: Mapping[str, Any]) -> "DatasetRegistry":
        return cls(dataset_config["manifest"], dataset_config.get("locations"))

    def resolve(self, dataset_id: str, *, explicit_path: str | Path | None = None) -> DatasetResource:
        """Resolve one logical ID without depending on the current directory."""

        dataset_id = str(dataset_id)
        if dataset_id not in self.records:
            raise KeyError(f"Unknown ThermalChannel dataset ID {dataset_id!r}; available={sorted(self.records)}")
        record = self.records[dataset_id]
        if explicit_path is not None:
            path = resolve_path(explicit_path, project_root=self.project_root)
        elif dataset_id in self.locations:
            path = resolve_path(self.locations[dataset_id], project_root=self.project_root)
        else:
            data_root = os.environ.get("HONF_DATA_ROOT")
            if not data_root:
                raise FileNotFoundError(
                    f"No location configured for dataset {dataset_id!r}. Copy "
                    "Dataset/dataset_locations.example.json to dataset_locations.local.json "
                    "or set HONF_DATA_ROOT."
                )
            relative = record.get("relative_path", record["filename"])
            path = Path(data_root).expanduser().resolve() / str(relative)
        return DatasetResource(dataset_id=dataset_id, path=path, record=record)

    def validate(self, resource: DatasetResource, *, verify_sha256: bool = False) -> None:
        """Check file identity and the required HDF5 root contract."""

        path = resource.path
        if not path.is_file():
            raise FileNotFoundError(f"Dataset {resource.dataset_id!r} not found: {path}")
        expected_size = resource.record.get("size_bytes")
        if expected_size is not None and path.stat().st_size != int(expected_size):
            raise ValueError(
                f"Dataset size mismatch for {resource.dataset_id!r}: "
                f"expected {expected_size}, found {path.stat().st_size}."
            )
        with h5py.File(path, "r") as h5_file:
            missing = [key for key in resource.record.get("required_keys", []) if key not in h5_file]
        if missing:
            raise ValueError(f"Dataset {resource.dataset_id!r} is missing HDF5 keys: {missing}")
        if verify_sha256:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            expected = str(resource.record.get("sha256", ""))
            if expected and digest.hexdigest() != expected:
                raise ValueError(
                    f"Dataset SHA-256 mismatch for {resource.dataset_id!r}: "
                    f"expected {expected}, found {digest.hexdigest()}."
                )

"""Atomic, checksummed persistence for geometry-only native role arrays."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_NATIVE_ROLE_CACHE_DIR = Path(__file__).resolve().parents[3] / "diagnostics/generated/native_role_catalogues_v1"


def training_catalogue_cache_directory() -> Path | None:
    """Share one physical-geometry store across future full and segmented fits."""
    value = os.environ.get("HONF_WIND_CATALOGUE_CACHE_DIR")
    if value == "off":
        return None
    return Path(value).expanduser().resolve() if value else DEFAULT_NATIVE_ROLE_CACHE_DIR


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class NativeRoleCatalogueStore:
    """Store immutable .npy arrays; never store targets, draws or learned state."""

    def __init__(self, directory: Path, recipe: dict[str, Any], roles: tuple[str, ...]):
        self.recipe = recipe
        self.roles = roles
        recipe_hash = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.directory = Path(directory).expanduser().resolve() / recipe_hash
        self.array_names = (
            "coordinates",
            *(f"cdf_{role}" for role in roles),
            *(f"indices_{role}" for role in roles if role != "volume"),
        )

    def entry(self, layout: int, geometry: str) -> Path:
        if len(geometry) != 64 or any(char not in "0123456789abcdef" for char in geometry):
            raise ValueError("Native catalogue geometry must have a SHA256 identity.")
        return self.directory / f"layout_{int(layout)}_{geometry}"

    @contextmanager
    def locked(self, layout: int, geometry: str):
        """A per-entry lock prevents duplicate builds across concurrent starts."""
        entry = self.entry(layout, geometry)
        locks = self.directory / ".locks"
        locks.mkdir(parents=True, exist_ok=True)
        with (locks / f"{entry.name}.lock").open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def load(self, layout: int, geometry: str, cell_count: int):
        """Reject stale, partial or corrupt payloads before returning mapped arrays."""
        entry = self.entry(layout, geometry)
        try:
            manifest = json.loads((entry / "manifest.json").read_text())
            if (
                manifest["schema_version"] != 1
                or manifest["recipe"] != self.recipe
                or manifest["layout_index"] != layout
                or manifest["geometry_sha256"] != geometry
                or manifest["native_cell_count"] != cell_count
                or set(manifest["arrays"]) != set(self.array_names)
                or set(manifest["role_support_volume_m3"]) != set(self.roles)
            ):
                return None
            arrays = {}
            for name in self.array_names:
                path = entry / f"{name}.npy"
                binding = manifest["arrays"][name]
                if path.stat().st_size != binding["file_bytes"] or _sha256(path) != binding["sha256"]:
                    return None
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                dtype = np.dtype(
                    "float32" if name == "coordinates" else "uint32" if name.startswith("indices_") else "float64"
                )
                if array.dtype != dtype or list(array.shape) != binding["shape"]:
                    return None
                arrays[name] = array
            if arrays["coordinates"].shape != (cell_count, 3):
                return None
            for role in self.roles:
                cdf = arrays[f"cdf_{role}"]
                support = manifest["role_support_volume_m3"][role]
                if cdf.ndim != 1 or not cdf.size or cdf[-1] != 1.0 or not np.isfinite(support) or support <= 0:
                    return None
                if role == "volume":
                    if cdf.shape != (cell_count,):
                        return None
                elif arrays[f"indices_{role}"].shape != cdf.shape:
                    return None
            return arrays, manifest["role_support_volume_m3"]
        except (OSError, ValueError, KeyError, TypeError, EOFError):
            return None

    def write(self, layout: int, geometry: str, arrays: dict[str, np.ndarray], supports: dict[str, float]):
        """Publish complete entries by atomic directory rename under the entry lock."""
        if set(arrays) != set(self.array_names):
            raise ValueError("Persistent native catalogue has an unexpected array inventory.")
        self.directory.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".building-", dir=self.directory))
        manifest = {
            "schema_version": 1,
            "recipe": self.recipe,
            "layout_index": layout,
            "geometry_sha256": geometry,
            "native_cell_count": len(arrays["coordinates"]),
            "role_support_volume_m3": supports,
            "arrays": {},
        }
        for name, array in arrays.items():
            path = staging / f"{name}.npy"
            with path.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            manifest["arrays"][name] = {
                "shape": list(array.shape),
                "file_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        with (staging / "manifest.json").open("w") as stream:
            json.dump(manifest, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        entry = self.entry(layout, geometry)
        if entry.exists():
            # Retain invalid evidence; never delete source data or other entries.
            os.replace(entry, entry.with_name(f".invalid-{entry.name}-{uuid.uuid4().hex}"))
        os.replace(staging, entry)
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

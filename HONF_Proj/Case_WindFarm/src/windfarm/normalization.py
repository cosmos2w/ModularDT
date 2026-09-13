"""Training-only velocity scaling and the vertical-profile diagnostic baseline."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import U_REF_MPS, gather_native_sample

NORMALIZATION_SCHEMA_VERSION = 1
DEFAULT_SAMPLES_PER_ROW = 8192
DEFAULT_PROFILE_BINS = 32


def _seed_sequence(seed: int, epoch: int, row: int, stream_id: int) -> np.random.SeedSequence:
    """Compose a reproducible integer seed without process-randomized hashes."""

    return np.random.SeedSequence((int(seed), int(epoch), int(row), int(stream_id)))


def _rng_for(seed: int, row: int, *, stream_id: int) -> np.random.Generator:
    return np.random.default_rng(_seed_sequence(seed, 0, row, stream_id))


def _safe_std(values: np.ndarray, floor: float = 1.0e-3) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=np.float64), float(floor))


@dataclass(frozen=True)
class VelocityNormalizer:
    """Global three-channel transform fitted from training native samples."""

    mean: np.ndarray
    std: np.ndarray
    safe_std: np.ndarray
    u_ref_mps: float = U_REF_MPS
    sample_count_per_row: int = DEFAULT_SAMPLES_PER_ROW
    source_rows: int = 0
    seed: int = 42
    std_floor: float = 1.0e-3

    def __post_init__(self) -> None:
        for name in ("mean", "std", "safe_std"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite length-3 vector")
        if np.any(self.safe_std <= 0.0) or float(self.u_ref_mps) <= 0.0:
            raise ValueError("safe_std and u_ref_mps must be positive")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VelocityNormalizer:
        return cls(
            mean=np.asarray(payload["mean_dimensionless"], dtype=np.float64),
            std=np.asarray(payload["std_dimensionless"], dtype=np.float64),
            safe_std=np.asarray(payload["safe_std_dimensionless"], dtype=np.float64),
            u_ref_mps=float(payload.get("u_ref_mps", U_REF_MPS)),
            sample_count_per_row=int(payload.get("samples_per_training_row", DEFAULT_SAMPLES_PER_ROW)),
            source_rows=int(payload.get("training_rows", 0)),
            seed=int(payload.get("seed", 42)),
            std_floor=float(payload.get("std_floor", 1.0e-3)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NORMALIZATION_SCHEMA_VERSION,
            "target_components": ["Ux", "Uy", "Uz"],
            "target_units": "m/s",
            "u_ref_mps": float(self.u_ref_mps),
            "mean_dimensionless": self.mean.astype(float).tolist(),
            "std_dimensionless": self.std.astype(float).tolist(),
            "safe_std_dimensionless": self.safe_std.astype(float).tolist(),
            "std_floor": float(self.std_floor),
            "samples_per_training_row": int(self.sample_count_per_row),
            "training_rows": int(self.source_rows),
            "seed": int(self.seed),
            "fitted_from": "native volume quadrature samples from training rows only",
        }

    def normalize(self, values_mps: Any) -> np.ndarray:
        values = np.asarray(values_mps, dtype=np.float32)
        if values.shape[-1:] != (3,):
            raise ValueError(f"velocity values must end in 3 channels, got {values.shape}")
        return ((values / float(self.u_ref_mps) - self.mean) / self.safe_std).astype(np.float32)

    def denormalize(self, values_dimensionless: Any) -> np.ndarray:
        values = np.asarray(values_dimensionless, dtype=np.float32)
        if values.shape[-1:] != (3,):
            raise ValueError(f"normalized velocity values must end in 3 channels, got {values.shape}")
        return ((values * self.safe_std + self.mean) * float(self.u_ref_mps)).astype(np.float32)


@dataclass
class _VectorMoments:
    count: int = 0
    mean: np.ndarray = None  # type: ignore[assignment]
    m2: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.mean is None:
            self.mean = np.zeros(3, dtype=np.float64)
        if self.m2 is None:
            self.m2 = np.zeros(3, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"moments expect [N,3] values, got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("velocity samples contain non-finite values")
        count = int(array.shape[0])
        if count == 0:
            return
        batch_mean = np.mean(array, axis=0, dtype=np.float64)
        centered = array - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0, dtype=np.float64)
        if self.count == 0:
            self.count = count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        total = self.count + count
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + delta * delta * (self.count * count / total)
        self.mean += delta * (count / total)
        self.count = total

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise ValueError("at least two finite samples are required for normalization")
        variance = np.maximum(self.m2 / float(self.count), 0.0)
        return self.mean.copy(), np.sqrt(variance)


@dataclass(frozen=True)
class VerticalProfileBaseline:
    """Training-population altitude profile used only as a diagnostic."""

    bin_centers_D: np.ndarray
    values_mps: np.ndarray
    counts: np.ndarray
    z_min_D: float = 0.0
    z_max_D: float = 6.25

    def __post_init__(self) -> None:
        centers = np.asarray(self.bin_centers_D)
        values = np.asarray(self.values_mps)
        counts = np.asarray(self.counts)
        if centers.ndim != 1 or values.shape != (centers.size, 3) or counts.shape != (centers.size,):
            raise ValueError("vertical profile arrays have inconsistent shapes")
        if np.any(~np.isfinite(values)) or np.any(counts < 0):
            raise ValueError("vertical profile must have finite values and non-negative bin counts")

    def predict(self, z_D: Any) -> np.ndarray:
        z = np.asarray(z_D, dtype=np.float64)
        flat = z.reshape(-1)
        output = np.column_stack(
            [np.interp(flat, self.bin_centers_D, self.values_mps[:, channel]) for channel in range(3)]
        )
        return output.reshape(z.shape + (3,)).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NORMALIZATION_SCHEMA_VERSION,
            "kind": "altitude_conditioned_training_profile",
            "bin_centers_D": np.asarray(self.bin_centers_D, dtype=np.float64).tolist(),
            "values_mps": np.asarray(self.values_mps, dtype=np.float64).tolist(),
            "counts": np.asarray(self.counts, dtype=np.int64).tolist(),
            "filled_empty_bins": np.flatnonzero(np.asarray(self.counts) == 0).astype(int).tolist(),
            "z_min_D": float(self.z_min_D),
            "z_max_D": float(self.z_max_D),
            "fit_scope": "same equal-per-training-row native volume samples as velocity normalization",
            "interpretation": "training-population background including its average wakes; diagnostic only",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VerticalProfileBaseline:
        return cls(
            bin_centers_D=np.asarray(payload["bin_centers_D"], dtype=np.float64),
            values_mps=np.asarray(payload["values_mps"], dtype=np.float64),
            counts=np.asarray(payload["counts"], dtype=np.int64),
            z_min_D=float(payload.get("z_min_D", 0.0)),
            z_max_D=float(payload.get("z_max_D", 6.25)),
        )


@dataclass
class _ProfileMoments:
    """Streaming sums for the small altitude profile."""

    bins: int
    z_min_D: float = 0.0
    z_max_D: float = 6.25

    def __post_init__(self) -> None:
        self.edges = np.linspace(self.z_min_D, self.z_max_D, int(self.bins) + 1, dtype=np.float64)
        self.sums = np.zeros((int(self.bins), 3), dtype=np.float64)
        self.counts = np.zeros(int(self.bins), dtype=np.int64)

    def update(self, coords_D: np.ndarray, velocity_mps: np.ndarray) -> None:
        coords = np.asarray(coords_D, dtype=np.float64)
        values = np.asarray(velocity_mps, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3 or values.shape != coords.shape:
            raise ValueError("profile samples must have matching [N,3] coordinates and velocities")
        index = np.searchsorted(self.edges, coords[:, 2], side="right") - 1
        index = np.clip(index, 0, int(self.bins) - 1)
        for channel in range(3):
            self.sums[:, channel] += np.bincount(index, weights=values[:, channel], minlength=int(self.bins))
        self.counts += np.bincount(index, minlength=int(self.bins)).astype(np.int64)

    def finish(self) -> VerticalProfileBaseline:
        if not np.any(self.counts):
            raise ValueError("vertical profile received no samples")
        centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        values = np.empty_like(self.sums)
        occupied = self.counts > 0
        for channel in range(3):
            values[:, channel] = np.interp(
                centers,
                centers[occupied],
                (self.sums[occupied, channel] / self.counts[occupied]),
            )
        # Empty bins use linear interpolation between occupied bin centres;
        # np.interp uses the nearest occupied value at either endpoint.  Keep
        # their true zero counts so this policy remains visible to reports.
        return VerticalProfileBaseline(centers, values, self.counts.copy(), self.z_min_D, self.z_max_D)


def fit_velocity_statistics(
    dataset: Any,
    training_rows: Iterable[int],
    *,
    samples_per_row: int = DEFAULT_SAMPLES_PER_ROW,
    seed: int = 42,
    profile_bins: int = DEFAULT_PROFILE_BINS,
) -> tuple[VelocityNormalizer, VerticalProfileBaseline]:
    """Fit normalization and profile from equal-count samples per train row."""

    rows = tuple(int(row) for row in training_rows)
    if not rows:
        raise ValueError("training_rows cannot be empty")
    count = int(samples_per_row)
    if count <= 0:
        raise ValueError("samples_per_row must be positive")
    moments = _VectorMoments()
    profile_moments = _ProfileMoments(int(profile_bins))
    for row in rows:
        rng = _rng_for(seed, row, stream_id=17)
        sample = gather_native_sample(dataset.run(row), count, rng, mode="volume")
        dimensionless = sample.velocity_mps.astype(np.float64) / float(U_REF_MPS)
        moments.update(dimensionless)
        profile_moments.update(sample.coords_D, sample.velocity_mps)
    mean, std = moments.finish()
    normalizer = VelocityNormalizer(
        mean=mean,
        std=std,
        safe_std=_safe_std(std),
        u_ref_mps=U_REF_MPS,
        sample_count_per_row=count,
        source_rows=len(rows),
        seed=int(seed),
    )
    return normalizer, profile_moments.finish()


def write_normalization_json(
    path: str | Path,
    normalizer: VelocityNormalizer,
    profile: VerticalProfileBaseline | None = None,
) -> Path:
    """Write the small train-derived transform/profile sidecar."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = normalizer.to_dict()
    if profile is not None:
        payload["vertical_profile_baseline"] = profile.to_dict()
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def read_normalization_json(path: str | Path) -> tuple[VelocityNormalizer, VerticalProfileBaseline | None]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    profile_payload = payload.get("vertical_profile_baseline")
    profile = None if profile_payload is None else VerticalProfileBaseline.from_dict(profile_payload)
    return VelocityNormalizer.from_dict(payload), profile


__all__ = [
    "DEFAULT_PROFILE_BINS",
    "DEFAULT_SAMPLES_PER_ROW",
    "NORMALIZATION_SCHEMA_VERSION",
    "VelocityNormalizer",
    "VerticalProfileBaseline",
    "fit_velocity_statistics",
    "read_normalization_json",
    "write_normalization_json",
]

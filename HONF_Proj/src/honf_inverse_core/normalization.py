"""Versioned affine normalization used by HONF inverse contracts.

Physical design ``D`` contains padded module geometry and heat, context ``c``
contains operating/material values, request ``R`` contains requested physical
functionals, compact plan ``G`` is the generated canonical mechanism, and
``G_hat`` is the same compact schema re-extracted by the frozen forward HONF.
This module only owns immutable numeric transformations shared by those objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


MIN_STD = 1.0e-8


def _finite_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


@dataclass(frozen=True)
class ScalarStats:
    """Mean/std/count for one named scalar physical quantity."""

    mean: float
    std: float
    count: int

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.mean)):
            raise ValueError("Normalization mean must be finite.")
        if not np.isfinite(float(self.std)) or float(self.std) < MIN_STD:
            raise ValueError(f"Normalization std must be finite and >= {MIN_STD}.")
        if int(self.count) < 0:
            raise ValueError("Normalization count must be nonnegative.")

    def normalize(self, value: Any) -> np.ndarray:
        return (_finite_array(value, name="normalization input") - float(self.mean)) / float(self.std)

    def normalize_width(self, value: Any) -> np.ndarray:
        return _finite_array(value, name="normalization width") / float(self.std)

    def denormalize(self, value: Any) -> np.ndarray:
        return _finite_array(value, name="denormalization input") * float(self.std) + float(self.mean)

    def denormalize_width(self, value: Any) -> np.ndarray:
        return _finite_array(value, name="denormalization width") * float(self.std)

    def to_dict(self) -> dict[str, Any]:
        return {"mean": float(self.mean), "std": float(self.std), "count": int(self.count)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScalarStats":
        unknown = sorted(set(payload) - {"mean", "std", "count"})
        if unknown:
            raise ValueError(f"Unknown scalar-normalization keys: {unknown}")
        return cls(mean=float(payload["mean"]), std=float(payload["std"]), count=int(payload.get("count", 0)))


@dataclass(frozen=True)
class VectorStats:
    """Named vector mean/std learned from the inverse-training split only."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    count: int

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.feature_names)
        mean = _finite_array(self.mean, name="vector mean").reshape(-1)
        std = _finite_array(self.std, name="vector std").reshape(-1)
        if not names or len(set(names)) != len(names):
            raise ValueError("Vector normalization requires unique non-empty feature names.")
        if mean.shape != (len(names),) or std.shape != mean.shape:
            raise ValueError("Vector mean/std shape must equal the feature-name count.")
        if np.any(std < MIN_STD):
            raise ValueError(f"Every vector std must be >= {MIN_STD}.")
        if int(self.count) < 0:
            raise ValueError("Normalization count must be nonnegative.")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "mean", mean.astype(np.float32))
        object.__setattr__(self, "std", std.astype(np.float32))

    def _values(self, value: Any, *, name: str) -> np.ndarray:
        values = _finite_array(value, name=name)
        if values.shape[-1:] != (len(self.feature_names),):
            raise ValueError(
                f"{name} final dimension {values.shape[-1:]} does not match "
                f"{len(self.feature_names)} features."
            )
        return values

    def normalize(self, value: Any) -> np.ndarray:
        values = self._values(value, name="vector normalization input")
        return ((values - self.mean) / self.std).astype(np.float32)

    def normalize_width(self, value: Any) -> np.ndarray:
        values = self._values(value, name="vector normalization width")
        return (values / self.std).astype(np.float32)

    def denormalize(self, value: Any) -> np.ndarray:
        values = self._values(value, name="vector denormalization input")
        return (values * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "count": int(self.count),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VectorStats":
        unknown = sorted(set(payload) - {"feature_names", "mean", "std", "count"})
        if unknown:
            raise ValueError(f"Unknown vector-normalization keys: {unknown}")
        return cls(
            feature_names=tuple(str(value) for value in payload["feature_names"]),
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
            count=int(payload.get("count", 0)),
        )


def fit_scalar(values: Sequence[float] | np.ndarray) -> ScalarStats:
    """Fit finite scalar statistics and clamp only a degenerate standard deviation."""

    array = _finite_array(values, name="fit values").reshape(-1)
    if array.size == 0:
        raise ValueError("Cannot fit scalar normalization from no values.")
    observed_std = float(np.std(array, ddof=0))
    std = observed_std if observed_std >= MIN_STD else 1.0
    return ScalarStats(mean=float(np.mean(array)), std=std, count=int(array.size))


def fit_vector(values: Any, feature_names: Sequence[str]) -> VectorStats:
    """Fit vector statistics over the first dimension."""

    array = _finite_array(values, name="fit vectors")
    names = tuple(str(name) for name in feature_names)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != len(names):
        raise ValueError("fit_vector expects a non-empty [N,F] array matching feature_names.")
    std = np.std(array, axis=0, ddof=0)
    std = np.where(std < MIN_STD, 1.0, std)
    return VectorStats(names, np.mean(array, axis=0), std, int(array.shape[0]))

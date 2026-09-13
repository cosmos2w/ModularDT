"""Native variable-domain case view, sampled Dataset, and batch collation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .geometry import (
    D_M,
    ENV_TOKEN_SHAPE,
    HUB_HEIGHT_M,
    U_REF_MPS,
    EnvironmentRepresentation,
    NativeQuerySample,
    SupportGeometry,
    environment_representation,
    gather_native_sample,
    global_geometry_features,
    module_geometry,
    support_features,
    support_geometry,
)
from .io import WindFarmDataset
from .normalization import VelocityNormalizer

COMPACT_GEOMETRY_KEYS = (
    "case",
    "layout",
    "layout_index",
    "wd_deg",
    "n_turbines",
    "turbine_xy_D",
    "U_ref",
    "D_m",
    "hub_height_m",
)


def _seed_sequence(seed: int, epoch: int, row: int, stream_id: int) -> np.random.SeedSequence:
    return np.random.SeedSequence((int(seed), int(epoch), int(row), int(stream_id)))


def _as_float_array(value: Any, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if shape is not None and result.shape != shape:
        raise ValueError(f"expected shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("geometry array contains non-finite values")
    return result


@dataclass(frozen=True)
class NativeCase:
    """One row's exact native fields plus known geometry representation."""

    index: int
    case: str
    layout: str
    layout_index: int
    wind_direction_deg: float
    n_turbines: int
    run: Any
    support: SupportGeometry
    environment: EnvironmentRepresentation
    module_centers: np.ndarray
    module_present: np.ndarray
    module_features: np.ndarray
    global_context: np.ndarray
    diameter_m: float = D_M
    hub_height_m: float = HUB_HEIGHT_M

    @property
    def shape_nxyz(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.run.shape_nxyz)

    @property
    def x_m(self) -> np.ndarray:
        return self.run.x_m

    @property
    def y_m(self) -> np.ndarray:
        return self.run.y_m

    @property
    def z_m(self) -> np.ndarray:
        return self.run.z_m

    @property
    def env_coords(self) -> np.ndarray:
        return self.environment.coords_D

    @property
    def env_features(self) -> np.ndarray:
        return self.environment.features

    @property
    def env_weights(self) -> np.ndarray:
        return self.environment.weights_D3

    def geometry_for_queries(self, query_coords_D: Any) -> dict[str, np.ndarray]:
        """Return query coordinates/features for arbitrary valid coordinates."""

        coords = _as_float_array(query_coords_D)
        if coords.ndim not in (2, 3) or coords.shape[-1] != 3:
            raise ValueError(f"query coordinates must be [Q,3] or [B,Q,3], got {coords.shape}")
        # The support box is explicit.  Do not clip or extrapolate silently;
        # callers need to know when an inference receiver lies outside it.
        lower = self.support.lower_D.astype(np.float32)
        upper = self.support.upper_D.astype(np.float32)
        if np.any(coords < lower) or np.any(coords > upper):
            raise ValueError("query coordinates lie outside the represented centre support")
        return {
            "query_xy": coords.copy(),
            "query_features": support_features(coords, self.support),
        }

    # The shorter name is convenient for streaming diagnostics.
    geometry_batch = geometry_for_queries

    def sample_queries(
        self,
        count: int,
        rng: np.random.Generator,
        *,
        mode: Literal["volume", "hub_band"] = "volume",
    ) -> NativeQuerySample:
        """Sample exact native receiver coordinates and matching U targets."""

        return gather_native_sample(
            self.run,
            int(count),
            rng,
            mode=mode,
            hub_height_m=self.hub_height_m,
            diameter_m=self.diameter_m,
        )


class WindFarmNativeView:
    """Shared mmap-backed view used by training and native diagnostics."""

    def __init__(
        self,
        root: str | Path | WindFarmDataset,
        *,
        compact_metadata: Mapping[str, Any] | None = None,
        allow_npz_metadata_fallback: bool = True,
        diameter_m: float = D_M,
        hub_height_m: float = HUB_HEIGHT_M,
        reference_speed_mps: float = U_REF_MPS,
        token_shape: tuple[int, int, int] = ENV_TOKEN_SHAPE,
    ) -> None:
        self.volume = root if isinstance(root, WindFarmDataset) else WindFarmDataset(root)
        metadata = (
            dict(compact_metadata)
            if compact_metadata is not None
            else self.volume.compact_metadata(
                allow_npz_fallback=allow_npz_metadata_fallback,
                names=COMPACT_GEOMETRY_KEYS,
            )
        )
        missing = [name for name in COMPACT_GEOMETRY_KEYS if name not in metadata]
        if missing:
            raise ValueError(
                "WindFarm forward geometry requires compact metadata keys "
                f"{missing}; provide family_tensor.npz or individual metadata arrays"
            )
        self.metadata = {name: np.asarray(metadata[name]) for name in metadata}
        self.diameter_m = float(diameter_m)
        self.hub_height_m = float(hub_height_m)
        self.reference_speed_mps = float(reference_speed_mps)
        self.token_shape = tuple(int(value) for value in token_shape)
        for name, expected in (
            ("D_m", self.diameter_m),
            ("hub_height_m", self.hub_height_m),
            ("U_ref", self.reference_speed_mps),
        ):
            supplied = np.asarray(self.metadata[name])
            if supplied.shape == () and not np.isclose(float(supplied), expected):
                raise ValueError(f"compact metadata {name}={float(supplied)} disagrees with configured {expected}")
        self._validate_join()

    def _validate_join(self) -> None:
        n = self.volume.n_runs
        for name in COMPACT_GEOMETRY_KEYS:
            values = self.metadata[name]
            if values.ndim > 0 and values.shape[0] != n:
                raise ValueError(f"compact metadata {name} has {values.shape[0]} rows, expected {n}")
        compact_case = np.asarray(self.metadata["case"]).astype(str)
        volume_case = np.asarray(self.volume.array("case")).astype(str)
        if not np.array_equal(compact_case, volume_case):
            raise ValueError("compact and volume case order differ")
        for name in ("layout_index", "wd_deg"):
            if not np.array_equal(np.asarray(self.metadata[name]), np.asarray(self.volume.array(name))):
                raise ValueError(f"compact and volume {name} order differs")
        n_turbines = np.asarray(self.metadata["n_turbines"])
        turbine_xy = np.asarray(self.metadata["turbine_xy_D"])
        if n_turbines.ndim != 1 or turbine_xy.ndim != 3 or turbine_xy.shape[:2] != (n, 30) or turbine_xy.shape[-1] != 2:
            raise ValueError("turbine metadata must have shapes [N] and [N,30,2]")
        if np.any(n_turbines < 1) or np.any(n_turbines > 30):
            raise ValueError("n_turbines contains values outside [1,30]")

    @property
    def n_cases(self) -> int:
        return self.volume.n_runs

    @property
    def volume_dataset(self) -> WindFarmDataset:
        """Alias used by callers that already refer to the raw view."""

        return self.volume

    @property
    def native(self) -> WindFarmNativeView:
        """Self-alias for study code that names the native view explicitly."""

        return self

    def run(self, index: int) -> NativeCase:
        """Build a case view without copying its raw field volume."""

        row = int(index)
        if row < 0 or row >= self.n_cases:
            raise IndexError(f"case index {row} outside [0, {self.n_cases})")
        run = self.volume.run(row)
        support = support_geometry(run.x_m, run.y_m, run.z_m, diameter_m=self.diameter_m)
        environment = environment_representation(support, token_shape=self.token_shape)
        n_turbines = int(np.asarray(self.metadata["n_turbines"])[row])
        centers, present, features = module_geometry(
            np.asarray(self.metadata["turbine_xy_D"])[row],
            n_turbines,
            hub_height_D=self.hub_height_m / self.diameter_m,
        )
        global_context = global_geometry_features(
            support,
            float(np.asarray(self.metadata["wd_deg"])[row]),
            n_turbines,
            reference_speed_mps=self.reference_speed_mps,
        )
        return NativeCase(
            index=row,
            case=str(np.asarray(self.metadata["case"])[row]),
            layout=str(np.asarray(self.metadata["layout"])[row]),
            layout_index=int(np.asarray(self.metadata["layout_index"])[row]),
            wind_direction_deg=float(np.asarray(self.metadata["wd_deg"])[row]),
            n_turbines=n_turbines,
            run=run,
            support=support,
            environment=environment,
            module_centers=centers,
            module_present=present,
            module_features=features,
            global_context=global_context,
            diameter_m=self.diameter_m,
            hub_height_m=self.hub_height_m,
        )


class WindFarmNativeDataset:
    """Map-style samples over exact native cells and geometry-only tokens."""

    def __init__(
        self,
        view: WindFarmNativeView,
        row_indices: Iterable[int],
        *,
        normalizer: VelocityNormalizer | None = None,
        queries_per_case: int = 1024,
        volume_fraction: float = 0.75,
        seed: int = 42,
        fixed_sampling: bool = False,
        include_query_weights: bool = True,
    ) -> None:
        self.view = view
        self.row_indices = tuple(int(index) for index in row_indices)
        if any(index < 0 or index >= view.n_cases for index in self.row_indices):
            raise IndexError("row_indices contains a case outside the native view")
        self.normalizer = normalizer
        self.queries_per_case = int(queries_per_case)
        self.volume_fraction = float(volume_fraction)
        if self.queries_per_case <= 0 or not 0.0 < self.volume_fraction < 1.0:
            raise ValueError("queries_per_case must be positive and volume_fraction must lie in (0,1)")
        self.volume_queries = round(self.queries_per_case * self.volume_fraction)
        self.band_queries = self.queries_per_case - self.volume_queries
        if self.volume_queries <= 0 or self.band_queries <= 0:
            raise ValueError("both volume and hub-band sample counts must be positive")
        self.seed = int(seed)
        self.fixed_sampling = bool(fixed_sampling)
        self.include_query_weights = bool(include_query_weights)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.row_indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _sample_rng(self, row: int, stream_id: int) -> np.random.Generator:
        epoch = 0 if self.fixed_sampling else self.epoch
        return np.random.default_rng(_seed_sequence(self.seed, epoch, row, stream_id))

    def __getitem__(self, position: int) -> dict[str, Any]:
        row = self.row_indices[position]
        case = self.view.run(row)
        volume_sample = case.sample_queries(
            self.volume_queries,
            self._sample_rng(row, 101),
            mode="volume",
        )
        band_sample = case.sample_queries(
            self.band_queries,
            self._sample_rng(row, 102),
            mode="hub_band",
        )
        query_xy = np.concatenate((volume_sample.coords_D, band_sample.coords_D), axis=0).astype(np.float32)
        physical_target = np.concatenate((volume_sample.velocity_mps, band_sample.velocity_mps), axis=0)
        target_field = (
            self.normalizer.normalize(physical_target)
            if self.normalizer is not None
            else physical_target.astype(np.float32, copy=True)
        )
        query_features = support_features(query_xy, case.support)
        loss_weights = np.concatenate(
            (
                np.full(self.volume_queries, self.volume_fraction / self.volume_queries, dtype=np.float32),
                np.full(self.band_queries, (1.0 - self.volume_fraction) / self.band_queries, dtype=np.float32),
            )
        )
        measure_weights = np.concatenate(
            (
                np.full(
                    self.volume_queries,
                    volume_sample.distribution_volume_m3 / self.volume_queries,
                    dtype=np.float32,
                ),
                np.full(
                    self.band_queries,
                    band_sample.distribution_volume_m3 / self.band_queries,
                    dtype=np.float32,
                ),
            )
        )
        metadata = {
            "source_index": row,
            "case": case.case,
            "layout": case.layout,
            "layout_index": case.layout_index,
            "wind_direction_deg": case.wind_direction_deg,
            "shape_nxyz": case.shape_nxyz,
            "support_lower_D": case.support.lower_D.astype(np.float32),
            "support_upper_D": case.support.upper_D.astype(np.float32),
            "support_volume_m3": case.support.volume_m3,
            "sampling_modes": ("volume", "hub_band"),
            "volume_query_count": self.volume_queries,
            "hub_band_query_count": self.band_queries,
        }
        sample: dict[str, Any] = {
            "module_centers": case.module_centers.copy(),
            "module_present": case.module_present.copy(),
            "module_features": case.module_features.copy(),
            "global_context": case.global_context.copy(),
            "env_coords": case.env_coords.copy(),
            "env_features": case.env_features.copy(),
            "env_weights": case.env_weights.copy(),
            "query_xy": query_xy,
            "query_features": query_features,
            "query_time": None,
            "target_field": target_field,
            "case_name": case.case,
            "metadata": metadata,
        }
        if self.include_query_weights:
            sample["query_loss_weight"] = loss_weights
            sample["query_measure_m3"] = measure_weights
        return sample


def collate_windfarm(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate samples while padding modules only to the local batch maximum."""

    if not samples:
        raise ValueError("cannot collate an empty WindFarm batch")
    first = samples[0]
    module_width = max(int(np.asarray(sample["module_centers"]).shape[0]) for sample in samples)
    batch_size = len(samples)
    centers = np.zeros((batch_size, module_width, 3), dtype=np.float32)
    present = np.zeros((batch_size, module_width), dtype=np.float32)
    features = np.zeros((batch_size, module_width, 2), dtype=np.float32)
    for batch_index, sample in enumerate(samples):
        count = int(np.asarray(sample["module_centers"]).shape[0])
        centers[batch_index, :count] = np.asarray(sample["module_centers"], dtype=np.float32)
        present[batch_index, :count] = np.asarray(sample["module_present"], dtype=np.float32)
        features[batch_index, :count] = np.asarray(sample["module_features"], dtype=np.float32)
    result: dict[str, Any] = {
        "module_centers": centers,
        "module_present": present,
        "module_features": features,
        "global_context": np.stack([np.asarray(sample["global_context"], dtype=np.float32) for sample in samples]),
        "env_coords": np.stack([np.asarray(sample["env_coords"], dtype=np.float32) for sample in samples]),
        "env_features": np.stack([np.asarray(sample["env_features"], dtype=np.float32) for sample in samples]),
        "env_weights": np.stack([np.asarray(sample["env_weights"], dtype=np.float32) for sample in samples]),
        "query_xy": np.stack([np.asarray(sample["query_xy"], dtype=np.float32) for sample in samples]),
        "query_features": np.stack([np.asarray(sample["query_features"], dtype=np.float32) for sample in samples]),
        "query_time": None,
        "target_field": np.stack([np.asarray(sample["target_field"], dtype=np.float32) for sample in samples]),
        "case_name": [str(sample["case_name"]) for sample in samples],
        "metadata": [sample["metadata"] for sample in samples],
    }
    for key in ("query_loss_weight", "query_measure_m3"):
        if key in first:
            result[key] = np.stack([np.asarray(sample[key], dtype=np.float32) for sample in samples])
    result["module_count"] = np.asarray(
        [int(np.asarray(sample["module_present"]).sum()) for sample in samples], dtype=np.int64
    )
    return result


def batch_to_batch_data(batch: Mapping[str, Any]) -> Any:
    """Build the reusable core's BatchData lazily, retaining auxiliary weights."""

    from honf_forward_core.config import BatchData

    fields = set(getattr(BatchData, "__dataclass_fields__", {}))
    payload = {
        name: batch[name]
        for name in (
            "module_centers",
            "module_present",
            "module_features",
            "global_context",
            "query_xy",
            "query_time",
            "target_field",
            "case_name",
            "metadata",
            "env_coords",
            "env_features",
            "query_features",
        )
        if name in fields
    }
    if "env_weights" in fields and batch.get("env_weights") is not None:
        payload["env_weights"] = batch["env_weights"]
    return BatchData(**payload)


def case_batch(
    case: NativeCase,
    query_coords_D: Any,
    *,
    normalizer: VelocityNormalizer | None = None,
    velocity_mps: Any | None = None,
) -> Any:
    """Build a tensor ``BatchData`` for one case and arbitrary geometry-only queries.

    This is the inference/native-streaming counterpart to
    :class:`WindFarmNativeDataset`: it never needs target values, while an
    optional ``velocity_mps`` argument creates the training-standardized
    target field for a measured batch.  The helper intentionally returns only
    canonical model fields; sampling measures stay in Dataset samples for
    losses and native reductions.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - the core package requires torch
        raise RuntimeError("case_batch requires PyTorch to construct BatchData") from exc
    query_input = (
        query_coords_D.detach().cpu().numpy()
        if torch.is_tensor(query_coords_D)
        else query_coords_D
    )
    geometry = case.geometry_for_queries(query_input)
    query = np.asarray(geometry["query_xy"], dtype=np.float32)
    if query.ndim != 2:
        raise ValueError("case_batch expects query_coords_D with shape [Q,3]")
    target = None
    if velocity_mps is not None:
        physical_input = (
            velocity_mps.detach().cpu().numpy()
            if torch.is_tensor(velocity_mps)
            else velocity_mps
        )
        physical = np.asarray(physical_input, dtype=np.float32)
        if physical.shape != (query.shape[0], 3):
            raise ValueError("velocity_mps must align with query_coords_D as [Q,3]")
        target = normalizer.normalize(physical) if normalizer is not None else physical.copy()
    from honf_forward_core.config import BatchData

    return BatchData(
        module_centers=torch.from_numpy(case.module_centers[None].copy()),
        module_present=torch.from_numpy(case.module_present[None].copy()),
        module_features=torch.from_numpy(case.module_features[None].copy()),
        global_context=torch.from_numpy(case.global_context[None].copy()),
        query_xy=torch.from_numpy(query[None].copy()),
        query_time=None,
        target_field=None if target is None else torch.from_numpy(target[None].copy()),
        case_name=case.case,
        metadata={
            "source_index": case.index,
            "case": case.case,
            "layout": case.layout,
            "layout_index": case.layout_index,
            "wind_direction_deg": case.wind_direction_deg,
            "support_lower_D": case.support.lower_D.astype(np.float32),
            "support_upper_D": case.support.upper_D.astype(np.float32),
        },
        env_coords=torch.from_numpy(case.env_coords[None].copy()),
        env_features=torch.from_numpy(case.env_features[None].copy()),
        query_features=torch.from_numpy(geometry["query_features"][None].copy()),
        env_weights=torch.from_numpy(case.env_weights[None].copy()),
    )


__all__ = [
    "COMPACT_GEOMETRY_KEYS",
    "NativeCase",
    "WindFarmNativeDataset",
    "WindFarmNativeView",
    "batch_to_batch_data",
    "case_batch",
    "collate_windfarm",
]

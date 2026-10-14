"""Native WindFarm adapter for the shared content-hypergraph core.

The adapter owns the Wind coordinate frame, geometry-only token features,
TRAIN-fitted residual transform, and height-profile conversion. It never reads
velocity targets while constructing a scene.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from honf_forward_core.content_hypergraph import QueryBatch, SceneBatch

from .geometry import support_features
from .normalization import VelocityNormalizer, VerticalProfileBaseline

WIND_CONTENT_ENVIRONMENT_SHAPE = (4, 4, 4)
WIND_CONTENT_ENVIRONMENT_COUNT = 64
WIND_CONTENT_MODULE_FEATURES = 2
WIND_CONTENT_ENVIRONMENT_FEATURES = 7
WIND_CONTENT_OPERATING_FEATURES = 11
WIND_CONTENT_QUERY_FEATURES = 7
WIND_CONTENT_LOCAL_RADIUS_D = 1.5
WIND_CONTENT_COORDINATE_FRAME = "wind_aligned_rotor_diameters_v1"


def _readonly(value: Any, dtype: Any = np.float32) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _stable_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class WindContentSceneInputs:
    """Target-free, native-frame geometry for one direction row."""

    row_index: int
    case_name: str
    layout_index: int
    wind_direction_deg: float
    module_centers_D: np.ndarray
    module_present: np.ndarray
    module_features: np.ndarray
    module_ids: np.ndarray
    operating_features: np.ndarray
    support_lower_D: np.ndarray
    support_upper_D: np.ndarray
    support_extent_D: np.ndarray
    environment_coordinates_D: np.ndarray
    environment_features: np.ndarray
    environment_measure_D3: np.ndarray

    @classmethod
    def from_case(cls, case: Any) -> WindContentSceneInputs:
        """Copy the prescribed geometry/context fields and exclude native U."""

        centers = np.asarray(case.module_centers)
        count = int(centers.shape[0])
        environment = case.environment
        if tuple(environment.token_shape) != WIND_CONTENT_ENVIRONMENT_SHAPE:
            raise ValueError("Wind content-hypergraph scenes require the geometry-only E64 token shape (4,4,4).")
        if centers.shape != (count, 3) or count < 1 or count > 30:
            raise ValueError("Wind physical module centers must have shape [M,3] with 1<=M<=30.")
        if np.asarray(case.module_present).shape != (count,) or np.asarray(case.module_features).shape != (count, 2):
            raise ValueError("Wind physical module presence/features must align as [M] and [M,2].")
        if np.asarray(case.global_context).shape != (WIND_CONTENT_OPERATING_FEATURES,):
            raise ValueError("Wind operating context must retain its prescribed width 11.")
        if (
            np.asarray(environment.coords_D).shape != (WIND_CONTENT_ENVIRONMENT_COUNT, 3)
            or np.asarray(environment.features).shape
            != (WIND_CONTENT_ENVIRONMENT_COUNT, WIND_CONTENT_ENVIRONMENT_FEATURES)
            or np.asarray(environment.weights_D3).shape != (WIND_CONTENT_ENVIRONMENT_COUNT,)
        ):
            raise ValueError("Wind native E64 geometry must have coordinate/feature/measure shapes [64,3]/[64,7]/[64].")
        if not all(
            np.isfinite(np.asarray(value)).all()
            for value in (
                centers,
                case.module_present,
                case.module_features,
                case.global_context,
                case.support.lower_D,
                case.support.upper_D,
                case.support.extent_D,
                environment.coords_D,
                environment.features,
                environment.weights_D3,
            )
        ):
            raise ValueError("Wind target-free native scene contains nonfinite geometry or features.")
        if np.any(np.asarray(environment.weights_D3) <= 0.0):
            raise ValueError("Wind native E64 centre-box measures must be positive.")
        module_ids = np.arange(count, dtype=np.int64)
        return cls(
            row_index=int(case.index),
            case_name=str(case.case),
            layout_index=int(case.layout_index),
            wind_direction_deg=float(case.wind_direction_deg),
            module_centers_D=_readonly(centers),
            module_present=_readonly(case.module_present),
            module_features=_readonly(case.module_features),
            # Native turbine metadata is stored in stable row order; these IDs
            # preserve that correspondence and are never learned embeddings.
            module_ids=_readonly(module_ids, dtype=np.int64),
            operating_features=_readonly(case.global_context),
            support_lower_D=_readonly(case.support.lower_D),
            support_upper_D=_readonly(case.support.upper_D),
            support_extent_D=_readonly(case.support.extent_D),
            environment_coordinates_D=_readonly(environment.coords_D),
            environment_features=_readonly(environment.features),
            environment_measure_D3=_readonly(environment.weights_D3),
        )

    @property
    def scene_key(self) -> tuple[str, int, int, float]:
        return (self.case_name, self.row_index, self.layout_index, self.wind_direction_deg)


@dataclass(frozen=True)
class WindContentReceivers:
    """A five-role native receiver sample with exact cell and role identity."""

    coordinates_D: np.ndarray
    features: np.ndarray
    flat_indices: np.ndarray
    role_ids: np.ndarray
    scene_keys: tuple[tuple[str, int, int, float], ...]
    local_batch_indices: np.ndarray | None = None
    local_query_indices: np.ndarray | None = None
    local_module_indices: np.ndarray | None = None

    def __post_init__(self) -> None:
        coordinates = _readonly(self.coordinates_D)
        features = _readonly(self.features)
        flat_indices = _readonly(self.flat_indices, dtype=np.int64)
        role_ids = _readonly(self.role_ids, dtype=np.int8)
        if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
            raise ValueError("Wind receivers must have shape [B,Q,3].")
        expected = coordinates.shape[:2]
        if features.shape != (*expected, WIND_CONTENT_QUERY_FEATURES):
            raise ValueError("Wind receiver features must have shape [B,Q,7].")
        if flat_indices.shape != expected or role_ids.shape != expected:
            raise ValueError("Wind receiver flat IDs and role IDs must align with [B,Q].")
        if not np.isfinite(coordinates).all() or not np.isfinite(features).all() or np.any(flat_indices < 0):
            raise ValueError("Wind receiver coordinates/features/IDs must be finite and valid.")
        if len(self.scene_keys) != coordinates.shape[0]:
            raise ValueError("Wind receiver scene keys must match the batch dimension.")
        if np.any((role_ids < 0) | (role_ids >= 5)):
            raise ValueError("Wind receiver role IDs must encode one of the five canonical roles.")
        packed = (
            self.local_batch_indices,
            self.local_query_indices,
            self.local_module_indices,
        )
        if any(value is not None for value in packed):
            if any(value is None for value in packed):
                raise ValueError("Packed Wind local support requires batch, query, and module indices together.")
            local_batch, local_query, local_module = (_readonly(value, dtype=np.int64) for value in packed)
            if (
                local_batch.ndim != 1
                or local_query.shape != local_batch.shape
                or local_module.shape != local_batch.shape
            ):
                raise ValueError("Packed Wind local-support indices must be aligned one-dimensional arrays.")
            if (
                np.any(local_batch < 0)
                or np.any(local_batch >= expected[0])
                or np.any(local_query < 0)
                or np.any(local_query >= expected[1])
                or np.any(local_module < 0)
            ):
                raise ValueError("Packed Wind local-support indices contain out-of-range entries.")
            object.__setattr__(self, "local_batch_indices", local_batch)
            object.__setattr__(self, "local_query_indices", local_query)
            object.__setattr__(self, "local_module_indices", local_module)
        object.__setattr__(self, "coordinates_D", coordinates)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "flat_indices", flat_indices)
        object.__setattr__(self, "role_ids", role_ids)


class WindContentHypergraphAdapter:
    """Convert target-free Wind scenes and normalized residuals to physical m/s."""

    def __init__(
        self,
        normalizer: VelocityNormalizer,
        background_profile: VerticalProfileBaseline,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        if not isinstance(normalizer, VelocityNormalizer):
            raise TypeError("Wind content adapter requires the serialized TRAIN VelocityNormalizer.")
        if not isinstance(background_profile, VerticalProfileBaseline):
            raise TypeError("Wind content adapter requires the TRAIN-fitted height profile.")
        # Own exact transform values instead of retaining mutable arrays from
        # a fit/cache container. Public read-only flags catch ordinary writes;
        # boundary snapshots also catch callers who deliberately re-enable
        # NumPy writes behind a prepared scene's normalizer key.
        self.normalizer = VelocityNormalizer(
            mean=np.array(normalizer.mean, dtype=np.float64, copy=True),
            std=np.array(normalizer.std, dtype=np.float64, copy=True),
            safe_std=np.array(normalizer.safe_std, dtype=np.float64, copy=True),
            u_ref_mps=float(normalizer.u_ref_mps),
            sample_count_per_row=int(normalizer.sample_count_per_row),
            source_rows=int(normalizer.source_rows),
            seed=int(normalizer.seed),
            std_floor=float(normalizer.std_floor),
        )
        self.background_profile = VerticalProfileBaseline(
            bin_centers_D=np.array(background_profile.bin_centers_D, dtype=np.float64, copy=True),
            values_mps=np.array(background_profile.values_mps, dtype=np.float64, copy=True),
            counts=np.array(background_profile.counts, dtype=np.int64, copy=True),
            z_min_D=float(background_profile.z_min_D),
            z_max_D=float(background_profile.z_max_D),
        )
        for array in (
            self.normalizer.mean,
            self.normalizer.std,
            self.normalizer.safe_std,
            self.background_profile.bin_centers_D,
            self.background_profile.values_mps,
            self.background_profile.counts,
        ):
            array.setflags(write=False)
        self._normalizer_arrays = {
            name: np.array(getattr(self.normalizer, name), copy=True) for name in ("mean", "std", "safe_std")
        }
        self._profile_arrays = {
            name: np.array(getattr(self.background_profile, name), copy=True)
            for name in ("bin_centers_D", "values_mps", "counts")
        }
        self._normalizer_scalars = (
            self.normalizer.u_ref_mps,
            self.normalizer.sample_count_per_row,
            self.normalizer.source_rows,
            self.normalizer.seed,
            self.normalizer.std_floor,
        )
        self._profile_scalars = (self.background_profile.z_min_D, self.background_profile.z_max_D)
        self.device = torch.device(device)
        self._state_payload = {
            "schema": "wind_content_hypergraph_adapter_v2",
            "coordinate_frame": WIND_CONTENT_COORDINATE_FRAME,
            "environment_token_shape": list(WIND_CONTENT_ENVIRONMENT_SHAPE),
            "module_features": ["rotor_radius_D", "hub_height_D"],
            "environment_features": [
                "distance_from_lower_x_over_positional_scale_D",
                "distance_from_lower_y_over_positional_scale_D",
                "distance_from_lower_z_over_positional_scale_D",
                "distance_to_upper_x_over_positional_scale_D",
                "distance_to_upper_y_over_positional_scale_D",
                "distance_to_upper_z_over_positional_scale_D",
                "absolute_height_over_positional_scale_D",
            ],
            "operating_features": [
                "wind_direction_one_hot_270",
                "wind_direction_one_hot_285",
                "wind_direction_one_hot_300",
                "turbine_count_fraction",
                "reference_speed_fraction",
                "support_lower_x_over_positional_scale_D",
                "support_lower_y_over_positional_scale_D",
                "support_lower_z_over_positional_scale_D",
                "support_extent_x_over_positional_scale_D",
                "support_extent_y_over_positional_scale_D",
                "support_extent_z_over_positional_scale_D",
            ],
            "query_features": [
                "distance_from_lower_x_over_positional_scale_D",
                "distance_from_lower_y_over_positional_scale_D",
                "distance_from_lower_z_over_positional_scale_D",
                "distance_to_upper_x_over_positional_scale_D",
                "distance_to_upper_y_over_positional_scale_D",
                "distance_to_upper_z_over_positional_scale_D",
                "absolute_height_over_positional_scale_D",
            ],
            "normalizer": normalizer.to_dict(),
            "background_profile": background_profile.to_dict(),
            "physical_output": "TRAIN height profile m/s + normalized residual * safe_std * u_ref_mps",
            "residual_transform": {
                "kind": "native_profile_residual_scale_v1",
                "centering": "TRAIN height profile only",
                "absolute_velocity_mean_added": False,
            },
            "wind_direction_handling": "native wind-aligned coordinates; direction remains prescribed operating input; no second rotation",
            "local_radius_D": WIND_CONTENT_LOCAL_RADIUS_D,
        }
        self.normalizer_key = _stable_sha256(self._state_payload)

    def _assert_transforms_unchanged(self) -> None:
        for name, snapshot in self._normalizer_arrays.items():
            current = np.asarray(getattr(self.normalizer, name))
            if (
                current.dtype != snapshot.dtype
                or current.shape != snapshot.shape
                or not np.array_equal(current, snapshot)
            ):
                raise RuntimeError(f"Wind adapter TRAIN normalizer array {name!r} changed after binding.")
        for name, snapshot in self._profile_arrays.items():
            current = np.asarray(getattr(self.background_profile, name))
            if (
                current.dtype != snapshot.dtype
                or current.shape != snapshot.shape
                or not np.array_equal(current, snapshot)
            ):
                raise RuntimeError(f"Wind adapter TRAIN background-profile array {name!r} changed after binding.")
        normalizer_scalars = (
            self.normalizer.u_ref_mps,
            self.normalizer.sample_count_per_row,
            self.normalizer.source_rows,
            self.normalizer.seed,
            self.normalizer.std_floor,
        )
        if normalizer_scalars != self._normalizer_scalars:
            raise RuntimeError("Wind adapter TRAIN normalizer metadata changed after binding.")
        if (self.background_profile.z_min_D, self.background_profile.z_max_D) != self._profile_scalars:
            raise RuntimeError("Wind adapter TRAIN background-profile bounds changed after binding.")

    @classmethod
    def from_state_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> WindContentHypergraphAdapter:
        if payload.get("schema") != "wind_content_hypergraph_adapter_v2":
            raise ValueError("Unsupported Wind content-hypergraph adapter state schema.")
        if payload.get("coordinate_frame") != WIND_CONTENT_COORDINATE_FRAME:
            raise ValueError("Wind adapter checkpoint uses a different native coordinate frame.")
        normalizer = VelocityNormalizer.from_dict(dict(payload["normalizer"]))
        profile = VerticalProfileBaseline.from_dict(dict(payload["background_profile"]))
        adapter = cls(normalizer, profile, device=device)
        if adapter.state_payload() != dict(payload):
            raise ValueError("Wind adapter state does not match its canonical serialized transforms.")
        return adapter

    def state_payload(self) -> dict[str, Any]:
        """Return all transforms needed for inference without reopening source data."""

        self._assert_transforms_unchanged()
        return json.loads(json.dumps(self._state_payload, sort_keys=True))

    @staticmethod
    def scene_inputs(case: Any) -> WindContentSceneInputs:
        return WindContentSceneInputs.from_case(case)

    @staticmethod
    def packed_local_pair_indices(
        inputs: WindContentSceneInputs,
        coordinates_D: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select native query/module pairs inside the prescribed 1.5D radius."""

        coordinates = np.asarray(coordinates_D, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[-1] != 3 or not np.isfinite(coordinates).all():
            raise ValueError("Wind local-support selection requires finite native coordinates [Q,3].")
        active = np.flatnonzero(np.asarray(inputs.module_present) > 0.5)
        query_rows: list[np.ndarray] = []
        module_rows: list[np.ndarray] = []
        for module_index in active:
            relative = coordinates - np.asarray(inputs.module_centers_D[module_index], dtype=np.float64)
            within = np.linalg.norm(relative, axis=-1) < WIND_CONTENT_LOCAL_RADIUS_D
            selected = np.flatnonzero(within)
            if selected.size:
                query_rows.append(selected.astype(np.int64, copy=False))
                module_rows.append(np.full(selected.size, int(module_index), dtype=np.int64))
        if not query_rows:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        return np.concatenate(query_rows), np.concatenate(module_rows)

    def make_scene_batch(self, inputs: Sequence[WindContentSceneInputs]) -> SceneBatch:
        self._assert_transforms_unchanged()
        if not inputs:
            raise ValueError("Wind content-hypergraph scene batches cannot be empty.")
        if any(not isinstance(item, WindContentSceneInputs) for item in inputs):
            raise TypeError("Wind content-hypergraph scene construction accepts only target-free scene inputs.")
        batch = len(inputs)
        module_count = max(int(item.module_centers_D.shape[0]) for item in inputs)
        if module_count > 30:
            raise ValueError("Wind physical source capacity exceeds 30 turbines.")
        module_features = np.zeros((batch, module_count, WIND_CONTENT_MODULE_FEATURES), dtype=np.float32)
        module_coordinates = np.zeros((batch, module_count, 3), dtype=np.float32)
        module_mask = np.zeros((batch, module_count), dtype=bool)
        module_measure = np.zeros((batch, module_count), dtype=np.float32)
        module_ids = np.full((batch, module_count), -1, dtype=np.int64)
        local_radius = np.full((batch, module_count), WIND_CONTENT_LOCAL_RADIUS_D, dtype=np.float32)
        environment_features = np.empty(
            (batch, WIND_CONTENT_ENVIRONMENT_COUNT, WIND_CONTENT_ENVIRONMENT_FEATURES), dtype=np.float32
        )
        environment_coordinates = np.empty((batch, WIND_CONTENT_ENVIRONMENT_COUNT, 3), dtype=np.float32)
        environment_mask = np.ones((batch, WIND_CONTENT_ENVIRONMENT_COUNT), dtype=bool)
        environment_measure = np.empty((batch, WIND_CONTENT_ENVIRONMENT_COUNT), dtype=np.float32)
        environment_ids = np.broadcast_to(
            np.arange(WIND_CONTENT_ENVIRONMENT_COUNT, dtype=np.int64),
            (batch, WIND_CONTENT_ENVIRONMENT_COUNT),
        ).copy()
        operating = np.empty((batch, WIND_CONTENT_OPERATING_FEATURES), dtype=np.float32)
        origin = np.empty((batch, 3), dtype=np.float32)
        scale = np.empty((batch, 3), dtype=np.float32)
        for batch_index, item in enumerate(inputs):
            count = int(item.module_centers_D.shape[0])
            if item.module_features.shape != (count, WIND_CONTENT_MODULE_FEATURES):
                raise ValueError("Wind module token features must be [M,2] = radius and hub height in D units.")
            if item.module_present.shape != (count,) or item.module_ids.shape != (count,):
                raise ValueError("Wind module masks and literal IDs must align with module geometry.")
            if item.environment_coordinates_D.shape != (64, 3) or item.environment_features.shape != (64, 7):
                raise ValueError("Wind E64 coordinates/features must have shapes [64,3] and [64,7].")
            if item.environment_measure_D3.shape != (64,) or np.any(item.environment_measure_D3 <= 0.0):
                raise ValueError("Wind E64 environment measures must be 64 positive native D^3 masses.")
            if item.operating_features.shape != (WIND_CONTENT_OPERATING_FEATURES,):
                raise ValueError("Wind prescribed global geometry/direction context must have width 11.")
            if not np.isfinite(item.module_centers_D).all() or not np.isfinite(item.environment_features).all():
                raise ValueError("Wind target-free scene geometry contains nonfinite values.")
            module_features[batch_index, :count] = item.module_features
            module_coordinates[batch_index, :count] = item.module_centers_D
            active = np.asarray(item.module_present) > 0.5
            module_mask[batch_index, :count] = active
            module_measure[batch_index, :count] = active.astype(np.float32)
            module_ids[batch_index, :count] = item.module_ids
            environment_features[batch_index] = item.environment_features
            environment_coordinates[batch_index] = item.environment_coordinates_D
            environment_measure[batch_index] = item.environment_measure_D3
            operating[batch_index] = item.operating_features
            origin[batch_index] = item.support_lower_D
            scale[batch_index] = item.support_extent_D
        if not np.isfinite(environment_measure).all() or np.any(environment_measure <= 0.0):
            raise ValueError("Wind native environment geometry must carry finite positive D^3 measures.")
        env_reference = environment_measure.mean(axis=1, dtype=np.float64).astype(np.float32)
        tensor = lambda value, dtype=None: torch.as_tensor(value, device=self.device, dtype=dtype)
        keys = tuple(item.scene_key for item in inputs)
        return SceneBatch(
            module_features=tensor(module_features),
            environment_features=tensor(environment_features),
            module_coordinates=tensor(module_coordinates),
            environment_coordinates=tensor(environment_coordinates),
            module_mask=tensor(module_mask, torch.bool),
            environment_mask=tensor(environment_mask, torch.bool),
            module_measure=tensor(module_measure),
            environment_measure=tensor(environment_measure),
            module_ids=tensor(module_ids, torch.long),
            environment_ids=tensor(environment_ids, torch.long),
            local_radius=tensor(local_radius),
            operating_features=tensor(operating),
            coordinate_origin=tensor(origin),
            coordinate_scale=tensor(scale),
            module_measure_reference=tensor(np.ones(batch, dtype=np.float32)),
            environment_measure_reference=tensor(env_reference),
            scene_key=keys,
            normalizer_key=self.normalizer_key,
            coordinate_frame_key=WIND_CONTENT_COORDINATE_FRAME,
        )

    def make_query_batch(
        self,
        scene: SceneBatch,
        receivers: WindContentReceivers,
    ) -> QueryBatch:
        self._assert_transforms_unchanged()
        if not isinstance(receivers, WindContentReceivers):
            raise TypeError("Wind query construction requires native WindContentReceivers.")
        if receivers.scene_keys != scene.scene_key:
            raise ValueError("Wind receiver batch is not bound to the supplied prescribed scene rows.")
        # Explicit copies avoid exposing immutable mmap/batch arrays through
        # writable Torch views while keeping the copy bounded to this batch.
        coordinates = torch.tensor(receivers.coordinates_D, device=self.device, dtype=torch.float32)
        features = torch.tensor(receivers.features, device=self.device, dtype=torch.float32)
        query_ids = torch.tensor(receivers.flat_indices, device=self.device, dtype=torch.long)
        if coordinates.ndim != 3 or coordinates.shape[0] != scene.batch_size or coordinates.shape[-1] != 3:
            raise ValueError("Wind native query coordinates must have shape [B,Q,3].")
        expected = (scene.batch_size, coordinates.shape[1])
        if features.shape != (*expected, WIND_CONTENT_QUERY_FEATURES):
            raise ValueError("Wind native query geometry features must have shape [B,Q,7].")
        if query_ids.shape != expected:
            raise ValueError("Wind native flat cell IDs must match every requested query.")
        if receivers.local_batch_indices is None:
            raise ValueError("Wind native receivers must include the adapter's packed local-support rows.")
        local_batch = torch.tensor(receivers.local_batch_indices, device=self.device, dtype=torch.long)
        local_query = torch.tensor(receivers.local_query_indices, device=self.device, dtype=torch.long)
        local_module = torch.tensor(receivers.local_module_indices, device=self.device, dtype=torch.long)
        if local_batch.numel():
            delta = coordinates[local_batch, local_query] - scene.module_coordinates[local_batch, local_module]
            radius = scene.local_radius[local_batch, local_module].clamp_min(torch.finfo(coordinates.dtype).tiny)
            ratio = torch.linalg.vector_norm(delta, dim=-1) / radius
            # Same endpoint-flat compact support window as the core's sparse
            # fallback, evaluated only at packed rows so the weights remain
            # differentiable without materializing a BxQxM distance tensor.
            local_weights = (1.0 - ratio).clamp_min(0.0).pow(4) * (4.0 * ratio + 1.0)
        else:
            local_weights = coordinates.new_empty((0,))
        return QueryBatch(
            coordinates=coordinates,
            features=features,
            query_mask=torch.ones(expected, device=self.device, dtype=torch.bool),
            query_ids=query_ids,
            local_batch_indices=local_batch,
            local_query_indices=local_query,
            local_module_indices=local_module,
            local_weights=local_weights,
            scene_key=scene.scene_key,
            coordinate_frame_key=WIND_CONTENT_COORDINATE_FRAME,
            receiver_key=(scene.scene_key, tuple(int(v) for v in coordinates.shape[1:2])),
        )

    def query_features(self, coords_D: Any, support: Any) -> np.ndarray:
        """Use the maintained seven native support features, with no rotation."""

        self._assert_transforms_unchanged()
        return support_features(coords_D, support)

    def profile_at(self, coordinates_D: torch.Tensor) -> torch.Tensor:
        """Differentiable piecewise-linear height-profile lookup in physical m/s."""

        self._assert_transforms_unchanged()
        centers = torch.tensor(
            self.background_profile.bin_centers_D,
            device=coordinates_D.device,
            dtype=coordinates_D.dtype,
        )
        values = torch.tensor(
            self.background_profile.values_mps,
            device=coordinates_D.device,
            dtype=coordinates_D.dtype,
        )
        z = coordinates_D[..., 2]
        if centers.numel() == 1:
            return values[0].expand(*z.shape, 3)
        right = torch.searchsorted(centers, z.contiguous(), right=True).clamp(1, centers.numel() - 1)
        left = right - 1
        low = centers[left]
        high = centers[right]
        fraction = ((z - low) / (high - low)).clamp(0.0, 1.0)
        interpolated = values[left] + fraction.unsqueeze(-1) * (values[right] - values[left])
        return torch.where(
            (z <= centers[0]).unsqueeze(-1),
            values[0],
            torch.where((z >= centers[-1]).unsqueeze(-1), values[-1], interpolated),
        )

    def physical_velocity(
        self,
        normalized_residual: torch.Tensor,
        coordinates_D: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the stored residual transform, then add the TRAIN profile."""

        self._assert_transforms_unchanged()
        if normalized_residual.shape != (*coordinates_D.shape[:-1], 3):
            raise ValueError("Wind normalized residual and receiver coordinates must align as [B,Q,3].")
        safe_std = torch.tensor(
            self.normalizer.safe_std, device=normalized_residual.device, dtype=normalized_residual.dtype
        )
        # The stored normalizer mean describes absolute velocity. Native Wind
        # residuals are centered by the height profile and use only its scale.
        residual_mps = normalized_residual * safe_std * float(self.normalizer.u_ref_mps)
        profile_mps = self.profile_at(coordinates_D)
        return profile_mps + residual_mps, profile_mps, residual_mps


__all__ = [
    "WIND_CONTENT_COORDINATE_FRAME",
    "WIND_CONTENT_ENVIRONMENT_COUNT",
    "WIND_CONTENT_ENVIRONMENT_SHAPE",
    "WIND_CONTENT_LOCAL_RADIUS_D",
    "WindContentHypergraphAdapter",
    "WindContentReceivers",
    "WindContentSceneInputs",
]

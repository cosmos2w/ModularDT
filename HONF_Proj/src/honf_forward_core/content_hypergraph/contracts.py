"""Tensor contracts for the content-hypergraph forward family.

The containers in this module deliberately describe tensor axes and ownership;
they do not assign physical meaning to dataset-specific features or units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


def _require_shape(name: str, value: Tensor | None, shape: tuple[int | None, ...]) -> None:
    if value is None:
        return
    if value.ndim != len(shape):
        raise ValueError(f"{name} must have rank {len(shape)}, got {value.ndim}.")
    for axis, (actual, expected) in enumerate(zip(value.shape, shape)):
        if expected is not None and actual != expected:
            raise ValueError(f"{name} axis {axis} must have size {expected}, got {actual}.")


def _require_tensor_condition(condition: Tensor, message: str) -> None:
    """Check tensor predicates without a Python sync on accelerator devices."""
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


@dataclass(frozen=True)
class SceneBatch:
    """Prescribed, target-free scene tensors for a batch of independent scenes.

    Features are already transformed by the dataset-owned, TRAIN-fitted input
    transforms. ``module_ids`` and ``environment_ids`` preserve correspondence
    only; the core never embeds their numeric values. ``local_radius`` is a
    per-module support radius in the scene coordinate units and is supplied by
    the adapter (for example, three module radii or 1.5 rotor diameters).
    """

    module_features: Tensor
    environment_features: Tensor
    module_coordinates: Tensor
    environment_coordinates: Tensor
    module_mask: Tensor
    environment_mask: Tensor
    module_measure: Tensor
    environment_measure: Tensor
    module_ids: Tensor
    environment_ids: Tensor
    local_radius: Tensor
    operating_features: Tensor | None = None
    coordinate_origin: Tensor | None = None
    coordinate_scale: Tensor | None = None
    module_measure_reference: Tensor | None = None
    environment_measure_reference: Tensor | None = None
    scene_key: Any = None
    normalizer_key: Any = None
    coordinate_frame_key: Any = None
    # Explicit physical lengths used by the versioned geometry-aware v2 core.
    # These remain optional for exact v1 replay; v2 requires both tensors.
    module_characteristic_lengths: Tensor | None = None
    environment_characteristic_lengths: Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.module_features.shape[0])

    @property
    def num_modules(self) -> int:
        return int(self.module_features.shape[1])

    @property
    def num_environment_tokens(self) -> int:
        return int(self.environment_features.shape[1])

    @property
    def coordinate_dim(self) -> int:
        return int(self.module_coordinates.shape[-1])

    def validate(
        self, *, coordinate_dim: int, module_input_dim: int, environment_input_dim: int, operating_input_dim: int
    ) -> None:
        """Validate the public scene boundary once before prepared computation."""
        if self.module_features.ndim != 3 or self.module_features.shape[-1] != module_input_dim:
            raise ValueError("module_features must have shape [B,M,module_input_dim].")
        if self.environment_features.ndim != 3 or self.environment_features.shape[-1] != environment_input_dim:
            raise ValueError("environment_features must have shape [B,E,environment_input_dim].")
        batch, modules = self.module_features.shape[:2]
        env_batch, env_count = self.environment_features.shape[:2]
        if batch != env_batch:
            raise ValueError("Module and environment batches must match.")
        _require_shape("module_coordinates", self.module_coordinates, (batch, modules, coordinate_dim))
        _require_shape("environment_coordinates", self.environment_coordinates, (batch, env_count, coordinate_dim))
        _require_shape("module_mask", self.module_mask, (batch, modules))
        _require_shape("environment_mask", self.environment_mask, (batch, env_count))
        _require_shape("module_measure", self.module_measure, (batch, modules))
        _require_shape("environment_measure", self.environment_measure, (batch, env_count))
        _require_shape("module_ids", self.module_ids, (batch, modules))
        _require_shape("environment_ids", self.environment_ids, (batch, env_count))
        _require_shape("local_radius", self.local_radius, (batch, modules))
        if self.module_mask.dtype != torch.bool or self.environment_mask.dtype != torch.bool:
            raise TypeError("module_mask and environment_mask must be boolean tensors.")
        if self.module_ids.dtype not in (torch.int32, torch.int64) or self.environment_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError("physical token IDs must use an integer tensor dtype.")
        if operating_input_dim:
            if self.operating_features is None:
                raise ValueError("operating_features are required by this core configuration.")
            _require_shape("operating_features", self.operating_features, (batch, operating_input_dim))
        elif self.operating_features is not None and self.operating_features.shape != (batch, 0):
            raise ValueError("operating_features were supplied but the core has no operating encoder.")
        _require_shape("coordinate_origin", self.coordinate_origin, (batch, coordinate_dim))
        _require_shape("coordinate_scale", self.coordinate_scale, (batch, coordinate_dim))
        _require_shape("module_measure_reference", self.module_measure_reference, (batch,))
        _require_shape("environment_measure_reference", self.environment_measure_reference, (batch,))
        _require_shape(
            "module_characteristic_lengths", self.module_characteristic_lengths,
            (batch, modules, coordinate_dim),
        )
        _require_shape(
            "environment_characteristic_lengths", self.environment_characteristic_lengths,
            (batch, env_count, coordinate_dim),
        )
        for name, value in (
            ("module_measure", self.module_measure),
            ("environment_measure", self.environment_measure),
        ):
            _require_tensor_condition(
                torch.isfinite(value).all() & (value >= 0).all(),
                f"{name} must contain only finite, nonnegative values.",
            )
        _require_tensor_condition(
            torch.isfinite(self.local_radius).all() & (self.local_radius > 0).all(),
            "local_radius must contain only finite, positive values.",
        )
        for name, value in (
            ("module_coordinates", self.module_coordinates),
            ("environment_coordinates", self.environment_coordinates),
        ):
            _require_tensor_condition(torch.isfinite(value).all(), f"{name} must be finite.")
        for name, value in (
            ("module_measure_reference", self.module_measure_reference),
            ("environment_measure_reference", self.environment_measure_reference),
        ):
            if value is not None:
                _require_tensor_condition(
                    torch.isfinite(value).all() & (value > 0).all(),
                    f"{name} must contain only finite, positive values.",
                )
        for name, value in (
            ("module_characteristic_lengths", self.module_characteristic_lengths),
            ("environment_characteristic_lengths", self.environment_characteristic_lengths),
        ):
            if value is not None:
                _require_tensor_condition(
                    torch.isfinite(value).all() & (value > 0).all(),
                    f"{name} must contain only finite, positive values.",
                )
        if self.coordinate_origin is not None:
            _require_tensor_condition(
                torch.isfinite(self.coordinate_origin).all(),
                "coordinate_origin must be finite.",
            )
            _require_tensor_condition(
                torch.isfinite(self.coordinate_scale).all() & (self.coordinate_scale > 0).all(),
                "coordinate_scale must contain only finite, positive values.",
            )

    def tensor_version_stamp(self) -> tuple[tuple[int, int, int], ...]:
        """Cheap in-place mutation stamp for public prepared-state boundaries."""
        values = (
            self.module_features,
            self.environment_features,
            self.module_coordinates,
            self.environment_coordinates,
            self.module_mask,
            self.environment_mask,
            self.module_measure,
            self.environment_measure,
            self.module_ids,
            self.environment_ids,
            self.local_radius,
            self.operating_features,
            self.coordinate_origin,
            self.coordinate_scale,
            self.module_measure_reference,
            self.environment_measure_reference,
            self.module_characteristic_lengths,
            self.environment_characteristic_lengths,
        )
        stamp = []
        for value in values:
            if value is None:
                continue
            try:
                version = int(value._version)
            except RuntimeError:  # Inference tensors do not expose a version counter.
                version = -1
            stamp.append((id(value), int(value.data_ptr()), version))
        return tuple(stamp)


@dataclass(frozen=True)
class QueryBatch:
    """Receiver coordinates and optional native features for one scene batch.

    Local pairs may be supplied as packed rows. If omitted, the core constructs
    all pairs inside each source's smooth compact-support radius from the live
    scene and query coordinates. Supplied ``local_weights`` are the adapter's
    differentiable support weights and are not detached by the core.
    """

    coordinates: Tensor
    features: Tensor | None = None
    query_mask: Tensor | None = None
    query_ids: Tensor | None = None
    local_batch_indices: Tensor | None = None
    local_query_indices: Tensor | None = None
    local_module_indices: Tensor | None = None
    local_weights: Tensor | None = None
    scene_key: Any = None
    coordinate_frame_key: Any = None
    receiver_key: Any = None

    @property
    def batch_size(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def num_queries(self) -> int:
        return int(self.coordinates.shape[1])

    def validate(self, *, batch_size: int, coordinate_dim: int, query_input_dim: int) -> None:
        if self.coordinates.ndim != 3:
            raise ValueError("query coordinates must have shape [B,Q,D].")
        _require_shape("coordinates", self.coordinates, (batch_size, None, coordinate_dim))
        if self.features is not None:
            if self.features.ndim != 3 or self.features.shape[:2] != self.coordinates.shape[:2]:
                raise ValueError("query features must have shape [B,Q,Fq].")
            if self.features.shape[-1] != query_input_dim:
                raise ValueError(f"query feature width must be {query_input_dim}.")
        elif query_input_dim:
            raise ValueError("query features are required by this core configuration.")
        _require_shape("query_mask", self.query_mask, (batch_size, self.num_queries))
        _require_shape("query_ids", self.query_ids, (batch_size, self.num_queries))
        _require_tensor_condition(torch.isfinite(self.coordinates).all(), "query coordinates must be finite.")
        if self.query_mask is not None and self.query_mask.dtype != torch.bool:
            raise TypeError("query_mask must be a boolean tensor.")
        packed = (
            self.local_batch_indices,
            self.local_query_indices,
            self.local_module_indices,
            self.local_weights,
        )
        if any(value is not None for value in packed):
            if not all(value is not None for value in packed):
                raise ValueError("All packed local-pair tensors must be supplied together.")
            if self.local_batch_indices.ndim != 1:
                raise ValueError("Packed local-pair indices must be one-dimensional.")
            count = int(self.local_batch_indices.shape[0])
            _require_shape("local_batch_indices", self.local_batch_indices, (count,))
            _require_shape("local_query_indices", self.local_query_indices, (count,))
            _require_shape("local_module_indices", self.local_module_indices, (count,))
            _require_shape("local_weights", self.local_weights, (count,))
            if any(value.dtype not in (torch.int32, torch.int64) for value in packed[:3]):
                raise TypeError("Packed local-pair indices must use an integer tensor dtype.")
            _require_tensor_condition(
                torch.isfinite(self.local_weights).all() & (self.local_weights >= 0).all(),
                "Packed local support weights must be finite and nonnegative.",
            )


@dataclass
class PreparedOrganization:
    """Live, query-independent organization for one scene and model state."""

    scene: SceneBatch
    module_states: Tensor
    environment_states: Tensor
    edges: Tensor
    module_incidence: Tensor
    environment_incidence: Tensor
    module_pooling_weights: Tensor
    environment_pooling_weights: Tensor
    module_edge_mass: Tensor
    environment_edge_mass: Tensor
    edge_anchors: Tensor
    edge_scales: Tensor
    edge_keys: Tensor | None
    edge_values: Tensor | None
    direct_source_keys: Tensor | None
    direct_source_values: Tensor | None
    source_coefficients: Tensor | None
    model_owner: int
    model_parameter_versions: tuple[tuple[int, int], ...]
    model_buffer_versions: tuple[tuple[int, int], ...]
    scene_tensor_versions: tuple[tuple[int, int, int], ...]
    scene_identity_snapshot: tuple[Any, Any, Any]
    frame_origin: Tensor
    frame_scale: Tensor
    operating_state: Tensor
    model_config_key: tuple[Any, ...]
    trace: dict[str, Tensor] | None = None

    @property
    def num_hyperedges(self) -> int:
        return int(self.edges.shape[1])

    @property
    def source_ids(self) -> Tensor:
        return self.scene.module_ids

    @property
    def scene_key(self) -> Any:
        return self.scene.scene_key


@dataclass
class PreparedOrganizationV2:
    """Prepared v2 state with typed donor-normalized gathers.

    ``module_gather_weights`` and ``environment_gather_weights`` are the
    executed W tensors, normalized over donors independently for each edge.
    The object deliberately has no v1 incidence/pooling aliases: any display
    incidence derived from W belongs in an explicitly labelled exporter.
    """

    scene: SceneBatch
    module_states: Tensor
    environment_states: Tensor
    raw_module_states: Tensor
    raw_environment_states: Tensor
    scene_summary: Tensor
    edges: Tensor
    first_content: Tensor
    mixed_content: Tensor
    second_content: Tensor
    module_gather_weights: Tensor
    environment_gather_weights: Tensor
    edge_anchors: Tensor
    edge_scales: Tensor
    edge_keys: Tensor | None
    edge_values: Tensor | None
    direct_source_keys: Tensor | None
    direct_source_values: Tensor | None
    source_coefficients: Tensor | None
    model_owner: int
    model_parameter_versions: tuple[tuple[int, int], ...]
    model_buffer_versions: tuple[tuple[int, int], ...]
    scene_tensor_versions: tuple[tuple[int, int, int], ...]
    scene_identity_snapshot: tuple[Any, Any, Any]
    frame_origin: Tensor
    frame_scale: Tensor
    operating_state: Tensor
    model_config_key: tuple[Any, ...]
    work_counts: dict[str, int]
    trace: dict[str, Tensor] | None = None

    @property
    def num_hyperedges(self) -> int:
        return int(self.edges.shape[1])

    @property
    def source_ids(self) -> Tensor:
        return self.scene.module_ids

    @property
    def scene_key(self) -> Any:
        return self.scene.scene_key


@dataclass
class CoreOutput:
    """Model-space field outputs; physical transforms remain adapter-owned."""

    nonlinear_field: Tensor
    thermal_temperature: Tensor | None = None
    thermal_basis: Tensor | None = None
    trace: dict[str, Tensor] | None = None
    work_counts: dict[str, int] | None = None

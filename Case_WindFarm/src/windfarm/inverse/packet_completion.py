"""Frozen packet reuse for stored-CFD turbine-position completion.

The target positions and full stored field are held by :class:`WindCompletionTask`.
The provider receives a separate candidate-only record with no run view or
hidden clean coordinates. Sensor locations are predeclared in physical D and
snapped once to native cell centres, independently of which turbines are hidden.
No candidate layout is described as having a new CFD reference.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
from honf_forward_core.interface_fields.budgeted_frontier import (
    FrontierSelection,
    enumerate_frontier_cuts,
    select_frontier_by_predictions,
)
from honf_forward_core.interface_fields.adaptive_interaction_cover import MechanismPlan
from honf_forward_core.interface_fields.input_cover_organizer import InputOnlyCoverOrganizer
from honf_forward_core.interface_fields.interaction_interface import (
    InteractionInterface,
    interaction_interface_from_plan,
)
from honf_inverse_core.models.frozen_packet_diffusion import (
    ConditionalPacketDenoiser,
    DiffusionCondition,
    FrozenPacketDiffusion,
    LinksForCandidate,
    PacketLinks,
)
from honf_runtime.compat import load_trusted_checkpoint
from torch import nn

from windfarm.data import NativeCase, case_batch
from windfarm.geometry import (
    ENV_TOKEN_SHAPE,
    HUB_HEIGHT_D,
    POSITIONAL_SCALE_D,
    U_REF_MPS,
    EnvironmentRepresentation,
    SupportGeometry,
    environment_representation,
    global_geometry_features,
    module_geometry,
    support_geometry,
)

DESIGN_LOWER_D = np.asarray((-15.0, -15.0), dtype=np.float32)
DESIGN_UPPER_D = np.asarray((15.0, 15.0), dtype=np.float32)
SENSOR_X_D = np.linspace(-7.0, 17.0, 6, dtype=np.float32)
SENSOR_Y_D = np.linspace(-5.0, 5.0, 4, dtype=np.float32)
SENSOR_OBSERVED_INDICES = tuple(index for index in range(24) if index % 3 != 0)
SENSOR_HELD_INDICES = tuple(index for index in range(24) if index % 3 == 0)


def intended_wind_sensor_coordinates() -> np.ndarray:
    """Return the fixed public sensor grid before any row-native snapping."""

    return np.asarray(
        [(float(x), float(y), HUB_HEIGHT_D) for y in SENSOR_Y_D for x in SENSOR_X_D],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class FixedNativeSensorPanel:
    """Twenty-four declared physical sensors and their stored native values."""

    intended_coordinates_D: np.ndarray  # [24,3], same across every layout
    native_coordinates_D: np.ndarray  # [24,3], snapped cell centres
    reference_velocity_mps: np.ndarray  # [24,3], stored OpenFOAM U
    native_flat_indices: np.ndarray  # [24]
    snap_distance_D: np.ndarray  # [24]


def fixed_native_sensor_panel(case: NativeCase) -> FixedNativeSensorPanel:
    """Select a layout-independent sensor grid and gather its nearest native cells.

    The intended grid is inside the measured intersection of all 600 source
    domains. The actual snapped coordinate and displacement are retained, so
    nearest-cell sampling is never called an exact fixed-coordinate solve.
    Neither the hidden-turbine choice nor turbine-neighbourhood coordinates
    enter sensor selection.
    """

    run = case.run
    if run is None:
        raise ValueError("Sensor reference extraction needs an intact stored native run")
    intended = intended_wind_sensor_coordinates()
    lower, upper = case.support.lower_D, case.support.upper_D
    if np.any(intended < lower[None, :]) or np.any(intended > upper[None, :]):
        raise ValueError("Predeclared Wind sensors are outside this native support")
    axes = (np.asarray(run.x_m), np.asarray(run.y_m), np.asarray(run.z_m))
    nearest = tuple(
        np.abs(axis[None, :] / case.diameter_m - intended[:, dim, None]).argmin(axis=1)
        for dim, axis in enumerate(axes)
    )
    ix, iy, iz = nearest
    flat = ix + int(run.nx) * (iy + int(run.ny) * iz)
    if len(set(map(int, flat))) != intended.shape[0]:
        raise ValueError("Native sensor snapping collapsed distinct declared sensors")
    native = np.column_stack(
        [axes[dim][nearest[dim]] / case.diameter_m for dim in range(3)]
    ).astype(np.float32)
    velocity = np.asarray(run.U[flat], dtype=np.float32).copy()
    if velocity.shape != (24, 3) or not np.all(np.isfinite(velocity)):
        raise ValueError("Stored native sensor velocities are invalid")
    displacement = np.linalg.norm(native - intended, axis=1).astype(np.float32)
    return FixedNativeSensorPanel(intended, native, velocity, flat.astype(np.int64), displacement)


def _sanitized_template(case: NativeCase, positions_D: np.ndarray, visible: np.ndarray) -> NativeCase:
    centers = np.asarray(positions_D, dtype=np.float32).copy()
    centers[~visible, :2] = 0.0
    # Remove the field view and any anchor geometry derived from clean hidden
    # positions. case_batch reconstructs anchors from the current candidate.
    return replace(
        case,
        run=None,
        module_centers=centers,
        receiver_anchor_coords=None,
        receiver_anchor_weights=None,
        receiver_anchor_roles=None,
    )


@dataclass(frozen=True)
class WindCompletionTask:
    """One supervised clean layout and a physically independent sensor panel."""

    row_index: int
    partition: str
    clean_centers_D: np.ndarray  # [M,3], training/evaluation target only
    visible_mask: np.ndarray  # [M]
    template_case: NativeCase  # no run, no hidden clean coordinates/anchors
    sensors: FixedNativeSensorPanel
    native_support_lower_D: np.ndarray | None = None
    native_support_upper_D: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.partition not in {"train", "development"}:
            raise ValueError("Wind completion tasks require an explicit train/development label")
        centers = np.asarray(self.clean_centers_D)
        visible = np.asarray(self.visible_mask)
        if centers.ndim != 2 or centers.shape[1] != 3 or visible.shape != centers.shape[:1]:
            raise ValueError("Clean centers and visible mask must align")
        if not 1 <= int((~visible).sum()) <= 2 or not bool(visible.any()):
            raise ValueError("This bounded task hides one or two of at least two turbines")
        if self.template_case.run is not None or self.template_case.receiver_anchor_coords is not None:
            raise ValueError("Candidate template must not retain clean CFD fields or anchors")
        if not np.array_equal(self.template_case.module_centers[visible], centers[visible]):
            raise ValueError("Template must preserve every visible turbine exactly")
        if not np.array_equal(
            self.template_case.module_centers[~visible, :2],
            np.zeros((int((~visible).sum()), 2), dtype=np.float32),
        ):
            raise ValueError("Candidate template must replace hidden positions with a fixed placeholder")
        if (self.native_support_lower_D is None) != (self.native_support_upper_D is None):
            raise ValueError("Native row support bounds must be supplied together")
        lower = (
            np.asarray(self.template_case.support.lower_D, dtype=np.float32).copy()
            if self.native_support_lower_D is None
            else np.asarray(self.native_support_lower_D, dtype=np.float32).copy()
        )
        upper = (
            np.asarray(self.template_case.support.upper_D, dtype=np.float32).copy()
            if self.native_support_upper_D is None
            else np.asarray(self.native_support_upper_D, dtype=np.float32).copy()
        )
        if (
            lower.ndim != 1
            or upper.shape != lower.shape
            or lower.size != 3
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower >= upper)
        ):
            raise ValueError("Native row support bounds must be finite ordered XYZ vectors")
        lower.setflags(write=False)
        upper.setflags(write=False)
        object.__setattr__(self, "native_support_lower_D", lower)
        object.__setattr__(self, "native_support_upper_D", upper)


@dataclass(frozen=True)
class WindCandidateGeometryContext:
    """Fixed public geometry used by candidate-only inverse construction."""

    support: SupportGeometry
    environment: EnvironmentRepresentation
    train_case_indices: tuple[int, ...]
    train_axis_sha256: str
    intended_sensor_coordinates_D: np.ndarray
    context_source: str = "train_axis_intersection"
    context_source_sha256: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "source_partition": (
                "train_only"
                if self.context_source == "train_axis_intersection"
                else "public_dataset_geometry"
            ),
            "context_source": self.context_source,
            "context_source_sha256": self.context_source_sha256,
            "train_case_indices": list(self.train_case_indices),
            "train_axis_sha256": self.train_axis_sha256,
            "support_lower_D": self.support.lower_D.tolist(),
            "support_upper_D": self.support.upper_D.tolist(),
            "environment_token_shape": list(self.environment.token_shape),
            "environment_coordinates_sha256": hashlib.sha256(
                np.ascontiguousarray(self.environment.coords_D).tobytes()
            ).hexdigest(),
            "environment_features_sha256": hashlib.sha256(
                np.ascontiguousarray(self.environment.features).tobytes()
            ).hexdigest(),
            "environment_weights_sha256": hashlib.sha256(
                np.ascontiguousarray(self.environment.weights_D3).tobytes()
            ).hexdigest(),
            "intended_sensor_coordinates_sha256": hashlib.sha256(
                np.ascontiguousarray(self.intended_sensor_coordinates_D).tobytes()
            ).hexdigest(),
        }


# Preserve the train-axis-specific import name for existing callers.
TrainDerivedWindCandidateContext = WindCandidateGeometryContext


def build_train_derived_wind_candidate_context(
    training_cases: Sequence[tuple[str, NativeCase]],
    *,
    intended_sensor_coordinates_D: np.ndarray | None = None,
) -> WindCandidateGeometryContext:
    """Build one candidate context from the intersection of train-only axes.

    Callers provide ``(partition, case)`` pairs so a validation/development or
    test row cannot be used accidentally.  Only axis bounds and the existing
    public sensor grid are retained; row-native environment tokens are not
    copied into the common context.
    """

    if not training_cases or any(partition != "train" for partition, _ in training_cases):
        raise ValueError("Common Wind support may be built from training cases only")
    cases = [case for _, case in training_cases]
    if any(case.run is None for case in cases):
        raise ValueError("Train-derived Wind support requires intact native axes")
    token_shapes = {
        tuple(int(value) for value in getattr(case.environment, "token_shape", ENV_TOKEN_SHAPE))
        for case in cases
    }
    if len(token_shapes) != 1:
        raise ValueError("Train-layout environment token shapes differ")
    axis_digest = hashlib.sha256()
    lower_m: list[float] = []
    upper_m: list[float] = []
    for axis_name in ("x_m", "y_m", "z_m"):
        axes = [np.asarray(getattr(case.run, axis_name), dtype=np.float64) for case in cases]
        if any(axis.ndim != 1 or axis.size < 2 or not np.all(np.isfinite(axis))
               or not np.all(np.diff(axis) > 0.0) for axis in axes):
            raise ValueError(f"Train native {axis_name} axes are not finite and increasing")
        common_lower = max(float(axis[0]) for axis in axes)
        common_upper = min(float(axis[-1]) for axis in axes)
        if common_lower >= common_upper:
            raise ValueError(f"Train native {axis_name} axes have no positive common interval")
        lower_m.append(common_lower)
        upper_m.append(common_upper)
        for case, axis in zip(cases, axes, strict=True):
            axis_digest.update(np.asarray([case.index], dtype=np.int64).tobytes())
            axis_digest.update(axis_name.encode("ascii"))
            axis_digest.update(np.asarray(axis.shape, dtype=np.int64).tobytes())
            axis_digest.update(axis.tobytes())
    bounds = [np.asarray((lo, hi), dtype=np.float64) for lo, hi in zip(lower_m, upper_m, strict=True)]
    support = support_geometry(*bounds, diameter_m=cases[0].diameter_m)
    sensors = (
        intended_wind_sensor_coordinates()
        if intended_sensor_coordinates_D is None
        else np.asarray(intended_sensor_coordinates_D, dtype=np.float32).copy()
    )
    if sensors.ndim != 2 or sensors.shape[1] != 3 or not np.all(np.isfinite(sensors)):
        raise ValueError("Fixed intended Wind sensor coordinates must be finite [S,3]")
    if np.any(sensors < support.lower_D[None, :]) or np.any(sensors > support.upper_D[None, :]):
        raise ValueError("Fixed intended Wind sensors fall outside train-derived common support")
    sensors.setflags(write=False)
    environment = environment_representation(support, token_shape=next(iter(token_shapes)))
    return WindCandidateGeometryContext(
        support=support,
        environment=environment,
        train_case_indices=tuple(int(case.index) for case in cases),
        train_axis_sha256=axis_digest.hexdigest(),
        intended_sensor_coordinates_D=sensors,
        context_source="train_axis_intersection",
        context_source_sha256=axis_digest.hexdigest(),
    )


def build_compact_public_wind_candidate_context(
    training_cases: Sequence[tuple[str, NativeCase]],
    *,
    compact_support_axes_D: tuple[Sequence[float], Sequence[float], Sequence[float]],
    compact_axes_sha256: str,
    intended_sensor_coordinates_D: np.ndarray | None = None,
) -> WindCandidateGeometryContext:
    """Build candidate context from fixed, documented compact raster axes.

    The compact raster coordinate extent is shared across all cases, but its
    per-case validity mask is not a claim that every coordinate is native CFD.
    The caller must bind the supplied axes to the compact dataset manifest and
    record row-native support checks separately.
    """

    if not training_cases or any(partition != "train" for partition, _ in training_cases):
        raise ValueError("Compact public Wind context requires explicitly labeled train cases")
    cases = [case for _, case in training_cases]
    if any(case.run is None for case in cases):
        raise ValueError("Compact public Wind context requires intact train cases")
    if len({float(case.diameter_m) for case in cases}) != 1:
        raise ValueError("Training cases disagree on the rotor diameter")
    token_shapes = {
        tuple(int(value) for value in getattr(case.environment, "token_shape", ENV_TOKEN_SHAPE))
        for case in cases
    }
    if len(token_shapes) != 1:
        raise ValueError("Train-layout environment token shapes differ")
    if len(compact_axes_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in compact_axes_sha256.lower()
    ):
        raise ValueError("Compact public axis digest must be a SHA-256 hex string")

    axes_D = tuple(np.asarray(axis, dtype=np.float64) for axis in compact_support_axes_D)
    if len(axes_D) != 3 or any(
        axis.ndim != 1
        or axis.size < 2
        or not np.all(np.isfinite(axis))
        or not np.all(np.diff(axis) > 0.0)
        for axis in axes_D
    ):
        raise ValueError("Compact public support axes must be finite, increasing XYZ vectors")
    diameter_m = float(cases[0].diameter_m)
    support = support_geometry(*(axis * diameter_m for axis in axes_D), diameter_m=diameter_m)
    sensors = (
        intended_wind_sensor_coordinates()
        if intended_sensor_coordinates_D is None
        else np.asarray(intended_sensor_coordinates_D, dtype=np.float32).copy()
    )
    if sensors.ndim != 2 or sensors.shape[1] != 3 or not np.all(np.isfinite(sensors)):
        raise ValueError("Fixed intended Wind sensor coordinates must be finite [S,3]")
    if np.any(sensors < support.lower_D[None, :]) or np.any(sensors > support.upper_D[None, :]):
        raise ValueError("Fixed intended Wind sensors fall outside the compact public domain")
    sensors.setflags(write=False)
    environment = environment_representation(support, token_shape=next(iter(token_shapes)))

    train_axis_digest = hashlib.sha256()
    for case in cases:
        for axis_name in ("x_m", "y_m", "z_m"):
            axis = np.asarray(getattr(case.run, axis_name), dtype=np.float64)
            train_axis_digest.update(np.asarray([case.index], dtype=np.int64).tobytes())
            train_axis_digest.update(axis_name.encode("ascii"))
            train_axis_digest.update(np.asarray(axis.shape, dtype=np.int64).tobytes())
            train_axis_digest.update(axis.tobytes())
    return WindCandidateGeometryContext(
        support=support,
        environment=environment,
        train_case_indices=tuple(int(case.index) for case in cases),
        train_axis_sha256=train_axis_digest.hexdigest(),
        intended_sensor_coordinates_D=sensors,
        context_source="compact_public_coordinate_axes",
        context_source_sha256=compact_axes_sha256.lower(),
    )


def apply_train_derived_wind_candidate_context(
    task: WindCompletionTask,
    context: WindCandidateGeometryContext,
) -> WindCompletionTask:
    """Replace row-native support/features with a frozen public context."""

    if not np.array_equal(task.sensors.intended_coordinates_D, context.intended_sensor_coordinates_D):
        raise ValueError("Task sensor grid differs from the frozen public sensor grid")
    template = task.template_case
    if template.run is not None or template.receiver_anchor_coords is not None:
        raise ValueError("Wind candidate context cannot be applied to a live or anchored case")
    centers, present, features = module_geometry(
        np.asarray(template.module_centers[:, :2], dtype=np.float32),
        int(template.n_turbines),
        hub_height_D=float(template.hub_height_m / template.diameter_m),
    )
    global_context = global_geometry_features(
        context.support,
        float(template.wind_direction_deg),
        int(template.n_turbines),
    )
    candidate_template = replace(
        template,
        support=context.support,
        environment=context.environment,
        module_centers=centers,
        module_present=present,
        module_features=features,
        global_context=global_context,
        receiver_anchor_coords=None,
        receiver_anchor_weights=None,
        receiver_anchor_roles=None,
    )
    return replace(task, template_case=candidate_template)


apply_wind_candidate_context = apply_train_derived_wind_candidate_context


def make_wind_completion_task(
    case: NativeCase,
    *,
    partition: str,
    hidden_count: int,
    seed: int,
) -> WindCompletionTask:
    """Build one task; permutation and hidden choice are independent of slots."""

    count = int(case.n_turbines)
    if hidden_count not in {1, 2} or hidden_count >= count:
        raise ValueError("Hide one or two turbines, leaving at least one visible")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(count)
    centers = np.asarray(case.module_centers, dtype=np.float32)[order].copy()
    features = np.asarray(case.module_features, dtype=np.float32)[order].copy()
    visible = np.ones(count, dtype=bool)
    visible[rng.choice(count, size=hidden_count, replace=False)] = False
    if np.any(centers[:, :2] < DESIGN_LOWER_D) or np.any(centers[:, :2] > DESIGN_UPPER_D):
        raise ValueError("Stored layout exceeds the declared train-supported design box")
    permuted = replace(case, module_centers=centers, module_features=features)
    return WindCompletionTask(
        row_index=int(case.index),
        partition=partition,
        clean_centers_D=centers,
        visible_mask=visible,
        template_case=_sanitized_template(permuted, centers, visible),
        sensors=fixed_native_sensor_panel(case),
    )


def permute_wind_task(task: WindCompletionTask, seed: int) -> WindCompletionTask:
    """Randomize same-type turbine token order on every optimizer update."""

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(task.clean_centers_D.shape[0])
    clean = task.clean_centers_D[order].copy()
    visible = task.visible_mask[order].copy()
    original = task.template_case
    permuted = replace(original, module_features=original.module_features[order].copy())
    return replace(
        task,
        clean_centers_D=clean,
        visible_mask=visible,
        template_case=_sanitized_template(permuted, clean, visible),
    )


@dataclass(frozen=True)
class WindCompletionKnown:
    """Everything the candidate graph may see, with the clean target removed."""

    template_case: NativeCase
    visible_mask: np.ndarray
    observed_coordinates_D: np.ndarray
    observed_velocity_mps: np.ndarray
    held_coordinates_D: np.ndarray
    held_velocity_mps: np.ndarray


@dataclass(frozen=True)
class WindCandidateKnown:
    """Only public design inputs and sensor coordinates enter the organizer."""

    template_case: NativeCase
    visible_mask: np.ndarray
    observed_coordinates_D: np.ndarray


def candidate_only_known(known: WindCompletionKnown) -> WindCandidateKnown:
    return WindCandidateKnown(
        template_case=known.template_case,
        visible_mask=known.visible_mask.copy(),
        observed_coordinates_D=known.observed_coordinates_D.copy(),
    )


def known_from_wind_task(task: WindCompletionTask) -> WindCompletionKnown:
    panel = task.sensors
    observed = np.asarray(SENSOR_OBSERVED_INDICES, dtype=np.int64)
    held = np.asarray(SENSOR_HELD_INDICES, dtype=np.int64)
    return WindCompletionKnown(
        template_case=task.template_case,
        visible_mask=task.visible_mask.copy(),
        # Native snapped coordinates and distances remain in ``task.sensors``
        # for audit.  The candidate conditioner always receives the declared
        # fixed locations, never coordinates that vary with a row's hidden
        # layout or native support.
        observed_coordinates_D=panel.intended_coordinates_D[observed].copy(),
        observed_velocity_mps=panel.reference_velocity_mps[observed].copy(),
        held_coordinates_D=panel.intended_coordinates_D[held].copy(),
        held_velocity_mps=panel.reference_velocity_mps[held].copy(),
    )


def task_native_support_bounds_D(
    task: WindCompletionTask,
) -> tuple[np.ndarray, np.ndarray]:
    """Return this task's native XY bounds for an explicit public-input mode.

    The bounds come from the row's exact native mesh and may depend on how
    that mesh was selected for the clean layout. Callers must opt in and
    record that provenance; this helper is not a common layout-independent
    domain.
    """

    lower = np.asarray(task.template_case.support.lower_D, dtype=np.float32)
    upper = np.asarray(task.template_case.support.upper_D, dtype=np.float32)
    if lower.ndim != 1 or upper.shape != lower.shape or lower.size < 2:
        raise ValueError("Task native support must provide aligned XYZ bounds")
    if not np.all(np.isfinite(lower[:2])) or not np.all(np.isfinite(upper[:2])):
        raise ValueError("Task native support XY bounds must be finite")
    return lower[:2].copy(), upper[:2].copy()


def native_clearance_design_bounds(
    support_lower_D: Sequence[float],
    support_upper_D: Sequence[float],
    rotor_radius_D: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect declared public support with the design box and rotor clearance.

    The caller supplies support independently of hidden positions. In
    particular, row-specific CFD axes are not presumed to be public inverse
    inputs when the row's native domain varies with the hidden layout.
    """

    radius_D = float(rotor_radius_D)
    if not np.isfinite(radius_D) or radius_D < 0:
        raise ValueError("Native rotor clearance must be finite and nonnegative")
    supplied_lower = np.asarray(support_lower_D, dtype=np.float32)
    supplied_upper = np.asarray(support_upper_D, dtype=np.float32)
    if supplied_lower.shape != (2,) or supplied_upper.shape != (2,):
        raise ValueError("Public native support must give two XY bounds")
    lower = np.maximum(DESIGN_LOWER_D, supplied_lower + radius_D)
    upper = np.minimum(DESIGN_UPPER_D, supplied_upper - radius_D)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or np.any(lower >= upper):
        raise ValueError("Public design and native rotor-clearance domains do not overlap")
    return lower.astype(np.float32), upper.astype(np.float32)


def _position_logits(
    xy_D: np.ndarray,
    lower_D: np.ndarray = DESIGN_LOWER_D,
    upper_D: np.ndarray = DESIGN_UPPER_D,
) -> np.ndarray:
    fraction = (np.asarray(xy_D, dtype=np.float32) - lower_D) / (upper_D - lower_D)
    fraction = np.clip(fraction, 1.0e-4, 1.0 - 1.0e-4)
    return np.log(fraction / (1.0 - fraction)).astype(np.float32)


def wind_condition_from_task(
    task: WindCompletionTask,
    *,
    device: torch.device | str = "cpu",
    sensor_velocity_center_mps: Sequence[Sequence[float]] | Sequence[float] | None = None,
    sensor_velocity_scale_mps: Sequence[Sequence[float]] | Sequence[float] | None = None,
    public_support_bounds_D: tuple[Sequence[float], Sequence[float]] | None = None,
) -> tuple[torch.Tensor, DiffusionCondition, WindCompletionKnown]:
    """Build the condition separately from clean targets and disclose bound inputs.

    Clean target coordinates never enter the condition. With no support bounds,
    the historical design-box parameterization is retained. Supplied support
    bounds are an additional task input whose source and scope must be audited
    by the caller. Row-native bounds are forbidden candidate inputs.
    """

    known = known_from_wind_task(task)
    count = int(task.clean_centers_D.shape[0])
    visible = known.visible_mask
    public_bounded = public_support_bounds_D is not None
    lower_D, upper_D = (
        native_clearance_design_bounds(
            public_support_bounds_D[0], public_support_bounds_D[1],
            float(known.template_case.module_features[0, 0]),
        ) if public_bounded else (DESIGN_LOWER_D, DESIGN_UPPER_D)
    )
    hidden_xy_D = task.clean_centers_D[~visible, :2]
    if public_bounded and np.any((hidden_xy_D < lower_D) | (hidden_xy_D > upper_D)):
        raise ValueError("Clean hidden target falls outside public support-clearance bounds")
    clean = _position_logits(task.clean_centers_D[:, :2], lower_D, upper_D)
    visible_logits = np.zeros_like(clean)
    if not public_bounded:
        # Preserve the historical inverse protocol. Under a narrower public
        # hidden-design domain, visible turbines may lie outside that domain;
        # their exact physical XY is supplied separately in module_features
        # and kept fixed by candidate_centers_from_state.
        visible_logits[visible] = clean[visible]
    template = known.template_case
    visible_xy = np.zeros_like(task.clean_centers_D[:, :2])
    visible_xy[visible] = task.clean_centers_D[visible, :2]
    module_features = np.concatenate(
        (
            template.module_features,
            np.broadcast_to(template.global_context[None, :], (count, template.global_context.size)),
            visible_xy / np.asarray((50.0, 38.0), dtype=np.float32),
            visible[:, None].astype(np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    if (sensor_velocity_center_mps is None) != (sensor_velocity_scale_mps is None):
        raise ValueError("Sensor velocity centering and scaling must be supplied together")
    if sensor_velocity_center_mps is None:
        normalized_sensor_velocity = known.observed_velocity_mps / float(U_REF_MPS)
    else:
        center = np.asarray(sensor_velocity_center_mps, dtype=np.float32)
        scale = np.asarray(sensor_velocity_scale_mps, dtype=np.float32)
        allowed_shapes = {(3,), tuple(known.observed_velocity_mps.shape)}
        if center.shape not in allowed_shapes or scale.shape != center.shape:
            raise ValueError("Sensor velocity statistics must have shape [3] or [observed_sensor_count,3]")
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("Sensor velocity centering/scaling statistics must be finite with positive scales")
        normalized_sensor_velocity = (known.observed_velocity_mps - center) / scale
    sensor_features = np.concatenate(
        (
            known.observed_coordinates_D / POSITIONAL_SCALE_D.astype(np.float32),
            normalized_sensor_velocity,
        ),
        axis=1,
    ).astype(np.float32)
    target_device = torch.device(device)
    condition = DiffusionCondition(
        known_state=torch.as_tensor(visible_logits[None], device=target_device),
        design_mask=torch.as_tensor((~visible)[None], device=target_device),
        module_valid=torch.ones((1, count), device=target_device, dtype=torch.bool),
        module_features=torch.as_tensor(module_features[None], device=target_device),
        sensor_features=torch.as_tensor(sensor_features[None], device=target_device),
        sensor_valid=torch.ones((1, len(SENSOR_OBSERVED_INDICES)), device=target_device, dtype=torch.bool),
        design_lower=(torch.as_tensor(lower_D[None], device=target_device) if public_bounded else None),
        design_upper=(torch.as_tensor(upper_D[None], device=target_device) if public_bounded else None),
    )
    return torch.as_tensor(clean[None], device=target_device), condition, known


def candidate_centers_from_state(
    state: torch.Tensor,
    condition: DiffusionCondition,
    known: WindCompletionKnown | WindCandidateKnown,
) -> torch.Tensor:
    """Map unconstrained hidden logits to the declared physical design box."""

    if state.shape[0] != 1 or state.shape[-1] != 2:
        raise ValueError("Wind candidate state must have shape [1,M,2]")
    count = int(known.template_case.n_turbines)
    if state.shape[1] != count or condition.design_mask.shape != (1, count):
        raise ValueError("Candidate state and known turbine count do not align")
    lower = (
        condition.design_lower[0].to(device=state.device, dtype=state.dtype)
        if condition.design_lower is not None else state.new_tensor(DESIGN_LOWER_D)
    )
    upper = (
        condition.design_upper[0].to(device=state.device, dtype=state.dtype)
        if condition.design_upper is not None else state.new_tensor(DESIGN_UPPER_D)
    )
    xy = lower + torch.sigmoid(state[0]) * (upper - lower)
    xy = torch.maximum(torch.minimum(xy, upper), lower)
    visible = torch.as_tensor(known.visible_mask, device=state.device)
    original_visible = torch.as_tensor(
        known.template_case.module_centers[:, :2], device=state.device, dtype=state.dtype
    )
    xy = torch.where(visible[:, None], original_visible, xy)
    height = state.new_full((count, 1), float(HUB_HEIGHT_D))
    return torch.cat((xy, height), dim=-1)


def wind_geometry_validity(
    centers_D: np.ndarray,
    rotor_radius_D: float,
    *,
    support_lower_D: np.ndarray | None = None,
    support_upper_D: np.ndarray | None = None,
    support_name: str = "native_domain",
) -> dict[str, object]:
    """Record design, named-support, and rotor-overlap checks before rejection.

    ``inside_native_domain`` remains reserved for exact row-native support.
    Other public boxes are returned as ``inside_support`` and are not mislabeled
    as native CFD validity.
    """

    label = str(support_name).strip()
    if not label:
        label = "native_domain"

    centers = np.asarray(centers_D, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or not np.all(np.isfinite(centers)):
        return {
            "finite": False,
            "inside_design_box": False,
            "inside_native_domain": False if label == "native_domain" else None,
            "native_support_status": "invalid_geometry" if label == "native_domain" else "not_checked_here",
            "inside_support": False,
            "support_name": label,
            "rotors_nonoverlap": False,
            "min_spacing_D": float("nan"),
        }
    inside = bool(np.all((centers[:, :2] >= DESIGN_LOWER_D) & (centers[:, :2] <= DESIGN_UPPER_D)))
    if (support_lower_D is None) != (support_upper_D is None):
        raise ValueError("Native support bounds must be supplied together")
    native = None if support_lower_D is None else bool(np.all(
        (centers >= np.asarray(support_lower_D)[None, :])
        & (centers <= np.asarray(support_upper_D)[None, :])
    ))
    inside_native = native if label == "native_domain" else None
    native_status = (
        "invalid_geometry"
        if label == "native_domain" and not np.all(np.isfinite(centers))
        else "checked"
        if label == "native_domain" and native is not None
        else "unknown"
        if label == "native_domain"
        else "not_checked_here"
    )
    difference = centers[:, None, :2] - centers[None, :, :2]
    distance = np.linalg.norm(difference, axis=-1)
    np.fill_diagonal(distance, np.inf)
    minimum = float(np.min(distance)) if len(centers) > 1 else float("inf")
    return {
        "finite": True,
        "inside_design_box": inside,
        "inside_native_domain": inside_native,
        "native_support_status": native_status,
        "inside_support": native,
        "support_name": label,
        "rotors_nonoverlap": bool(minimum >= 2.0 * float(rotor_radius_D)),
        "min_spacing_D": minimum,
    }


def hidden_set_error_D(generated_D: np.ndarray, clean_D: np.ndarray, visible_mask: np.ndarray) -> float:
    """Mean optimal hidden-position match for one or two exchangeable turbines."""

    hidden = ~np.asarray(visible_mask, dtype=bool)
    generated = np.asarray(generated_D, dtype=np.float64)[hidden]
    clean = np.asarray(clean_D, dtype=np.float64)[hidden]
    if generated.shape != clean.shape or generated.ndim != 2 or generated.shape[1] != 3:
        raise ValueError("Hidden generated and clean turbine sets must align")
    return float(min(
        np.linalg.norm(generated[list(order), :2] - clean[:, :2], axis=1).mean()
        for order in permutations(range(len(generated)))
    ))


@dataclass(frozen=True)
class WindFrontierEvidence:
    selected_frontier: tuple[int, ...] | None
    least_risk_frontier: tuple[int, ...]
    unsupported_at_budget: bool
    predicted_work_fraction: float | None
    predicted_role_risk: tuple[float, ...] | None
    least_risk_predicted_work_fraction: float
    least_risk_predicted_role_risk: tuple[float, ...]


class WindCandidateInterfaceBuilder:
    """Rebuild a trained, frozen forward organizer from each candidate layout."""

    def __init__(
        self,
        model: nn.Module,
        organizer: InputOnlyCoverOrganizer,
        *,
        budget_fractions: Mapping[str, float],
        role_tolerance: Sequence[float],
        numerical_state_version: str,
        max_frontier_depth: int = 3,
    ) -> None:
        if organizer.frontier_utility_head is None:
            raise ValueError("Wind inverse reuse requires a trained frontier utility head")
        if organizer.frontier_utility_head.role_count != len(role_tolerance):
            raise ValueError("Frontier utility and physical-role tolerances differ")
        if not numerical_state_version.strip():
            raise ValueError("Frozen forward/organizer numerical version is required")
        self.model = model.eval().requires_grad_(False)
        self.organizer = organizer.eval().requires_grad_(False)
        self.budget_fractions = {str(key).upper(): float(value) for key, value in budget_fractions.items()}
        if set(self.budget_fractions) not in ({"QE", "MM"}, {"QE", "EM"}):
            raise ValueError("Wind reuse requires QE plus one selected MM or EM capacity route")
        self.role_tolerance = torch.as_tensor(tuple(role_tolerance), dtype=torch.float32)
        self.numerical_state_version = numerical_state_version
        self.max_frontier_depth = int(max_frontier_depth)
        self.last_frontier_evidence: WindFrontierEvidence | None = None

    @torch.no_grad()
    def __call__(self, centers_D: torch.Tensor, known: WindCandidateKnown) -> InteractionInterface:
        if centers_D.ndim != 2 or centers_D.shape != known.template_case.module_centers.shape:
            raise ValueError("Current candidate centers must align with known turbine count")
        if not bool(torch.isfinite(centers_D).all()):
            raise ValueError("Candidate turbine centers must be finite")
        centers = centers_D.detach().to(device="cpu", dtype=torch.float32).numpy()
        candidate = replace(
            known.template_case,
            module_centers=centers.copy(),
            receiver_anchor_coords=None,
            receiver_anchor_weights=None,
            receiver_anchor_roles=None,
        )
        device = next(self.model.parameters()).device
        batch = case_batch(
            candidate,
            known.observed_coordinates_D,
            velocity_mps=None,
            include_receiver_anchors=True,
        ).to(device)
        core = self.model.core
        encoded = core.encode_case(batch)
        trees = core.backend.build_case_trees(encoded)
        score_inputs = {
            "module_states": encoded.module_tokens,
            "environment_states": encoded.env_tokens,
            "global_state": encoded.global_token,
        }
        scores = self.organizer.score_cases(
            encoded, score_inputs, trees, budgets=self.budget_fractions
        )[0]
        cuts = enumerate_frontier_cuts(trees[0], max_depth=self.max_frontier_depth)
        prediction = self.organizer.score_frontiers(
            scores, trees[0], cuts=cuts, max_depth=self.max_frontier_depth
        )
        selection: FrontierSelection = select_frontier_by_predictions(
            prediction,
            role_tolerance=self.role_tolerance.to(
                device=prediction.role_distortion.device,
                dtype=prediction.role_distortion.dtype,
            ),
        )
        chosen = (
            selection.least_risk_frontier
            if selection.unsupported_at_budget
            else selection.selected_frontier
        )
        if chosen is None:
            raise RuntimeError("Frontier utility supplied no deployment or research cut")
        self.last_frontier_evidence = WindFrontierEvidence(
            selected_frontier=selection.selected_frontier,
            least_risk_frontier=selection.least_risk_frontier,
            unsupported_at_budget=selection.unsupported_at_budget,
            predicted_work_fraction=(
                None
                if selection.selected_index is None
                else float(prediction.predicted_work[selection.selected_index].detach().cpu())
            ),
            predicted_role_risk=(
                None
                if selection.selected_index is None
                else tuple(
                    float(value)
                    for value in prediction.role_distortion[selection.selected_index].detach().cpu().tolist()
                )
            ),
            least_risk_predicted_work_fraction=float(
                prediction.predicted_work[selection.least_risk_index].detach().cpu()
            ),
            least_risk_predicted_role_risk=tuple(
                float(value)
                for value in prediction.role_distortion[selection.least_risk_index].detach().cpu().tolist()
            ),
        )
        plan = self.organizer.plans_from_scores(
            (scores,),
            encoded,
            trees,
            hard=True,
            frontier_cuts=(chosen,),
            budget_fractions=self.budget_fractions,
        )[0]
        return interaction_interface_from_plan(
            plan,
            encoded,
            scores,
            numerical_state_version=self.numerical_state_version,
            evidence_scope=(
                "frozen-forward-research-cut-unsupported"
                if selection.unsupported_at_budget
                else "frozen-forward-selected-cut"
            ),
        )


def packet_links_from_wind_interface(
    interface: InteractionInterface,
    observed_coordinates_D: torch.Tensor,
) -> PacketLinks:
    """Retain native typed routes and physical source order for one case."""

    if observed_coordinates_D.ndim != 2 or observed_coordinates_D.shape[1] != 3:
        raise ValueError("Wind observation coordinates must be [S,3]")
    module = interface.module_descriptors["coordinates"]
    environment = interface.environment_descriptors["coordinates"]
    module_embedding = interface.module_embeddings
    environment_embedding = interface.environment_embeddings
    if module_embedding is None or environment_embedding is None:
        raise ValueError("Frozen forward source embeddings are required for matched inverse controls")
    observed = observed_coordinates_D.to(device=module.device, dtype=module.dtype)
    return PacketLinks(
        module_source=interface.module_pair_weights()[None],
        sensor_source=interface.receiver_source_weights("QM", observed)[None],
        module_embeddings=module_embedding[None],
        module_environment=interface.receiver_source_weights("ME", module)[None],
        environment_module=interface.receiver_source_weights("EM", environment)[None],
        sensor_environment=interface.receiver_source_weights("QE", observed)[None],
        environment_embeddings=environment_embedding[None],
        environment_valid=interface.environment_validity[None],
        module_coordinates=module[None],
        sensor_coordinates=observed[None],
    )


InterfaceForWindCandidate = Callable[[torch.Tensor, WindCandidateKnown], InteractionInterface]


@dataclass(frozen=True)
class WindFixedActionEvidence:
    action_key: str
    requested_paths: tuple[str, ...]
    realized_paths: tuple[str, ...]
    policy_selected: bool
    selection_status: str
    nonredundant_k: int | None
    public_support_source: str


def _wind_action_nonredundant_k(plan: MechanismPlan, encoded: Any, cut: tuple[int, ...]) -> int:
    """Count distinct realized packet source signatures for one sparse cut."""

    indices = torch.as_tensor(cut, device=plan.split_gates.device, dtype=torch.long)
    module = (plan.permission_matrix("MM")[indices] > 0.0) & (
        encoded.module_present[0] > 0.5
    )[None, :]
    environment = (plan.permission_matrix("QE")[indices] > 0.0) & (
        encoded.env_weights[0] > 0.0
    )[None, :]
    signatures = torch.cat((module, environment), dim=1)
    bearing = signatures.any(dim=1)
    return len({tuple(row) for row in signatures[bearing].detach().cpu().tolist()})


class WindFixedActionCandidateInterfaceBuilder:
    """Build selected-u4910 candidate links for one declared fixed cut.

    This adapter deliberately does not consult the legacy Stage-C frontier
    utility head.  The action is a predeclared research intervention resolved
    against the current candidate tree, while feature context and sensor
    locations come from the frozen public context recorded by the caller.
    """

    def __init__(
        self,
        model: nn.Module,
        organizer: InputOnlyCoverOrganizer,
        *,
        context: WindCandidateGeometryContext,
        action_key: str,
        budget_fractions: Mapping[str, float],
        numerical_state_version: str,
        max_frontier_depth: int = 3,
    ) -> None:
        if action_key not in {"root", "two_packet", "full_access"}:
            raise ValueError("Fixed Wind inverse action must be root, two_packet, or full_access")
        if not numerical_state_version.strip():
            raise ValueError("Frozen forward/organizer numerical version is required")
        budgets = {str(key).upper(): float(value) for key, value in budget_fractions.items()}
        if set(budgets) != {"QE", "MM"} or any(not 0.0 <= value <= 1.0 for value in budgets.values()):
            raise ValueError("Wind inverse reuse requires valid frozen QE and MM capacities")
        self.model = model.eval().requires_grad_(False)
        self.organizer = organizer.eval().requires_grad_(False)
        self.context = context
        self.action_key = str(action_key)
        self.budget_fractions = budgets
        self.numerical_state_version = numerical_state_version
        self.max_frontier_depth = int(max_frontier_depth)
        self.calls = 0
        self.last_frontier_evidence: WindFixedActionEvidence | None = None

    @torch.no_grad()
    def __call__(self, centers_D: torch.Tensor, known: WindCandidateKnown) -> InteractionInterface:
        from windfarm.workflows.maturation import WIND_ACTION_PATHS, available_frontier_for_paths

        if centers_D.ndim != 2 or centers_D.shape != known.template_case.module_centers.shape:
            raise ValueError("Candidate centers must align with the known Wind template")
        if not bool(torch.isfinite(centers_D).all()):
            raise ValueError("Candidate centers must be finite before interface encoding")
        if known.template_case.run is not None:
            raise ValueError("Candidate-only interface must not retain a native field view")
        template = known.template_case
        support_lower = torch.as_tensor(
            self.context.support.lower_D, device=centers_D.device, dtype=centers_D.dtype
        )
        support_upper = torch.as_tensor(
            self.context.support.upper_D, device=centers_D.device, dtype=centers_D.dtype
        )
        if bool(((centers_D < support_lower) | (centers_D > support_upper)).any()):
            raise ValueError("Candidate centers fall outside the frozen public support")
        if not np.array_equal(template.support.lower_D, self.context.support.lower_D) or not np.array_equal(
            template.support.upper_D, self.context.support.upper_D
        ):
            raise ValueError("Candidate template does not use the frozen public support")
        expected_environment = self.context.environment
        for name in ("coords_D", "features", "weights_D3"):
            actual = getattr(template.environment, name, None)
            expected = getattr(expected_environment, name)
            if actual is None or not np.array_equal(actual, expected):
                raise ValueError(
                    f"Candidate template does not use the frozen public environment {name}"
                )
        expected_global_context = global_geometry_features(
            self.context.support,
            float(template.wind_direction_deg),
            int(template.n_turbines),
        )
        if not np.array_equal(template.global_context, expected_global_context):
            raise ValueError("Candidate template does not use the frozen public global context")
        expected_observed_coordinates = self.context.intended_sensor_coordinates_D[
            np.asarray(SENSOR_OBSERVED_INDICES, dtype=np.int64)
        ]
        if not np.array_equal(known.observed_coordinates_D, expected_observed_coordinates):
            raise ValueError("Candidate observed sensors differ from the declared fixed sensor subset")
        candidate = replace(
            template,
            module_centers=centers_D.detach().to(device="cpu", dtype=torch.float32).numpy().copy(),
            support=self.context.support,
            environment=self.context.environment,
            receiver_anchor_coords=None,
            receiver_anchor_weights=None,
            receiver_anchor_roles=None,
        )
        batch = case_batch(
            candidate,
            known.observed_coordinates_D,
            velocity_mps=None,
            include_receiver_anchors=True,
        ).to(next(self.model.parameters()).device)
        core = self.model.core
        encoded = core.encode_case(batch)
        trees = core.backend.build_case_trees(encoded)
        score_inputs = {
            "module_states": encoded.module_tokens,
            "environment_states": encoded.env_tokens,
            "global_state": encoded.global_token,
        }
        scores = self.organizer.score_cases(
            encoded, score_inputs, trees, budgets=self.budget_fractions
        )
        if self.action_key == "full_access":
            plan = MechanismPlan.full_access(
                trees[0], encoded.module_present[0], int(encoded.env_coords.shape[1])
            )
            requested_paths: tuple[str, ...] = ()
            realized_paths = tuple()
            nonredundant_k = None
        else:
            requested_paths = tuple(WIND_ACTION_PATHS[self.action_key])
            frontier, realized_paths = available_frontier_for_paths(
                trees[0], requested_paths, max_depth=self.max_frontier_depth
            )
            plan = self.organizer.plans_from_scores(
                scores,
                encoded,
                trees,
                hard=True,
                frontier_cuts=(frontier,),
                budget_fractions=self.budget_fractions,
            )[0]
            nonredundant_k = _wind_action_nonredundant_k(plan, encoded, tuple(frontier))
        interface = interaction_interface_from_plan(
            plan,
            encoded,
            scores[0],
            numerical_state_version=self.numerical_state_version,
            evidence_scope=f"selected-u4910-forced-{self.action_key}-research-only",
        )
        self.calls += 1
        self.last_frontier_evidence = WindFixedActionEvidence(
            action_key=self.action_key,
            requested_paths=requested_paths,
            realized_paths=tuple(realized_paths),
            policy_selected=False,
            selection_status="forced_research_action_not_policy_selection",
            nonredundant_k=nonredundant_k,
            public_support_source=self.context.context_source,
        )
        return interface


class WindCandidatePacketProvider:
    """Supply current-proposal graph links, with no reference target in scope."""

    def __init__(self, known: WindCompletionKnown, builder: InterfaceForWindCandidate) -> None:
        if known.template_case.run is not None:
            raise ValueError("Candidate provider must not retain a clean native field view")
        self.known = candidate_only_known(known)
        self.builder = builder
        self.last_candidate_centers_D: torch.Tensor | None = None
        self.last_interface: InteractionInterface | None = None

    def __call__(self, state: torch.Tensor, condition: DiffusionCondition) -> PacketLinks:
        centers = candidate_centers_from_state(state.detach(), condition, self.known)
        interface = self.builder(centers, self.known)
        observed = torch.as_tensor(
            self.known.observed_coordinates_D,
            device=centers.device,
            dtype=centers.dtype,
        )
        links = packet_links_from_wind_interface(interface, observed)
        self.last_candidate_centers_D = centers.detach().clone()
        self.last_interface = interface
        return links


def _degree_preserving_packet_rewire(
    membership: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, int]:
    """Swap node/source incidences, retaining each packet size and source degree."""

    if membership.ndim != 2 or not bool(torch.isfinite(membership).all()):
        raise ValueError("Packet permissions must be a finite node/source matrix")
    values = membership.detach().cpu().numpy().copy()
    edges = np.argwhere(values > 0.0)
    if len(edges) < 2:
        return membership.detach().clone(), 0
    original_rows = (values > 0).sum(axis=1)
    original_columns = (values > 0).sum(axis=0)
    rng = np.random.default_rng(int(seed))
    swaps = 0
    target_swaps = min(31, 2 * max(0, len(edges) // 4) + 1)
    for _ in range(min(10 * len(edges), 2000)):
        first, second = rng.choice(len(edges), size=2, replace=False)
        row_a, source_a = map(int, edges[first])
        row_b, source_b = map(int, edges[second])
        if row_a == row_b or source_a == source_b:
            continue
        if values[row_a, source_b] > 0.0 or values[row_b, source_a] > 0.0:
            continue
        strength_a = values[row_a, source_a]
        strength_b = values[row_b, source_b]
        values[row_a, source_a] = values[row_b, source_b] = 0.0
        values[row_a, source_b] = strength_a
        values[row_b, source_a] = strength_b
        edges[first] = (row_a, source_b)
        edges[second] = (row_b, source_a)
        swaps += 1
        if swaps >= target_swaps:
            break
    if not np.array_equal((values > 0).sum(axis=1), original_rows) or not np.array_equal(
        (values > 0).sum(axis=0), original_columns
    ):
        raise AssertionError("Packet rewiring changed a packet size or source degree")
    return torch.as_tensor(values, device=membership.device, dtype=membership.dtype), swaps


def rewire_wind_interface(
    interface: InteractionInterface,
    *,
    seed: int,
) -> tuple[InteractionInterface, dict[str, int]]:
    """Make a fixed-weight packet topology control from the same candidate.

    The candidate encoder, embeddings, frontier and target-free score pass are
    identical; each typed node/source permission undergoes degree-preserving
    edge swaps. A full or too-small matrix can have zero effective swaps, which
    must be reported rather than treated as a changed-graph control.
    """

    permissions: dict[str, torch.Tensor] = {}
    swaps: dict[str, int] = {}
    for route_index, (key, membership) in enumerate(sorted(interface.typed_permissions.items())):
        rewired, count = _degree_preserving_packet_rewire(
            membership, seed=int(seed) + 7919 * route_index
        )
        permissions[key] = rewired
        swaps[key] = count
    return replace(
        interface,
        typed_permissions=permissions,
        evidence_scope=f"{interface.evidence_scope};fixed-weight-packet-incidence-rewire",
    ), swaps


class WindRewiredPacketProvider:
    """Rebuild current-candidate packets, then rewire their typed memberships."""

    def __init__(
        self,
        known: WindCompletionKnown,
        builder: InterfaceForWindCandidate,
        *,
        seed: int,
    ) -> None:
        self.known = candidate_only_known(known)
        self.builder = builder
        self.seed = int(seed)
        self.calls = 0
        self.last_interface: InteractionInterface | None = None
        self.last_swap_counts: dict[str, int] = {}
        self.total_swap_counts: dict[str, int] = {}

    def __call__(self, state: torch.Tensor, condition: DiffusionCondition) -> PacketLinks:
        centers = candidate_centers_from_state(state.detach(), condition, self.known)
        interface = self.builder(centers, self.known)
        rewired, counts = rewire_wind_interface(interface, seed=self.seed + self.calls * 104729)
        observed = torch.as_tensor(
            self.known.observed_coordinates_D, device=centers.device, dtype=centers.dtype
        )
        self.calls += 1
        self.last_interface = rewired
        self.last_swap_counts = counts
        for key, value in counts.items():
            self.total_swap_counts[key] = self.total_swap_counts.get(key, 0) + value
        return packet_links_from_wind_interface(rewired, observed)


def _state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class WindMatchedDiffusionResult:
    graph_model: FrozenPacketDiffusion
    dense_model: FrozenPacketDiffusion
    graph_losses: tuple[float, ...]
    dense_losses: tuple[float, ...]
    initial_denoiser_hash: str
    frozen_state_hashes_before: Mapping[str, str]
    frozen_state_hashes_after: Mapping[str, str]
    updates_per_arm: int
    organizer_calls: int


def train_matched_wind_diffusion(
    train_tasks: Sequence[WindCompletionTask],
    *,
    denoiser_template: ConditionalPacketDenoiser,
    provider_factory: Callable[[WindCompletionKnown], LinksForCandidate],
    frozen_modules: Mapping[str, nn.Module],
    updates: int,
    steps: int = 20,
    learning_rate: float = 2.0e-4,
    seed: int = 21119,
    device: torch.device | str = "cpu",
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = 50,
    resume_checkpoint: Path | None = None,
    attempt_log_path: Path | None = None,
    sensor_velocity_center_mps: Sequence[Sequence[float]] | Sequence[float] | None = None,
    sensor_velocity_scale_mps: Sequence[Sequence[float]] | Sequence[float] | None = None,
    public_support_bounds_for_task: Callable[
        [WindCompletionTask], tuple[Sequence[float], Sequence[float]]
    ] | None = None,
    public_support_input_label: str = "none",
    allow_sensor_transform_transition: bool = False,
) -> WindMatchedDiffusionResult:
    """Fit graph/full-link arms with equal weights, cases, noise and updates."""

    if not train_tasks or any(task.partition != "train" for task in train_tasks):
        raise ValueError("Wind inverse fitting accepts training layouts only")
    if not 1 <= int(updates) <= 800 or not 2 <= int(steps) <= 20:
        raise ValueError("Wind inverse pilot exceeds bounded updates or diffusion steps")
    if learning_rate <= 0.0:
        raise ValueError("Inverse learning rate must be positive")
    if checkpoint_every < 1:
        raise ValueError("Inverse checkpoint period must be positive")
    if resume_checkpoint is not None and checkpoint_dir is None:
        raise ValueError("Resume requires the checkpoint directory for provenance")
    if public_support_bounds_for_task is None and public_support_input_label != "none":
        raise ValueError("Public support bounds and their explicit input label must be supplied together")
    if public_support_bounds_for_task is not None and (
        not public_support_input_label.strip() or public_support_input_label == "none"
    ):
        raise ValueError("An explicit public support bounds provider needs a non-'none' provenance label")
    if (sensor_velocity_center_mps is None) != (sensor_velocity_scale_mps is None):
        raise ValueError("Sensor velocity centering and scaling must be supplied together")
    transform_record = None
    if sensor_velocity_center_mps is not None:
        transform_record = {
            "center_mps": np.asarray(sensor_velocity_center_mps, dtype=np.float32).tolist(),
            "scale_mps": np.asarray(sensor_velocity_scale_mps, dtype=np.float32).tolist(),
        }
    target_device = torch.device(device)
    frozen_before = {
        name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        for name, module in frozen_modules.items()
    }
    frozen_hashes_before = {name: _state_hash(module) for name, module in frozen_modules.items()}
    for module in frozen_modules.values():
        module.eval().requires_grad_(False)
    graph = FrozenPacketDiffusion(deepcopy(denoiser_template), task="position", steps=steps).to(target_device)
    dense = FrozenPacketDiffusion(deepcopy(denoiser_template), task="position", steps=steps).to(target_device)
    graph.train()
    dense.train()
    initial_hash = _state_hash(graph.denoiser)
    if _state_hash(dense.denoiser) != initial_hash:
        raise RuntimeError("Matched inverse denoisers did not start identically")
    optimizers = (
        torch.optim.AdamW(graph.denoiser.parameters(), lr=learning_rate),
        torch.optim.AdamW(dense.denoiser.parameters(), lr=learning_rate),
    )
    index_rng = np.random.default_rng(int(seed))
    noise_rng = torch.Generator(device="cpu").manual_seed(int(seed) ^ 0x7197)
    losses: tuple[list[float], list[float]] = ([], [])
    task_hash = hashlib.sha256()
    for task in train_tasks:
        task_hash.update(str(task.row_index).encode("ascii"))
        task_hash.update(np.asarray(task.clean_centers_D, dtype=np.float32).tobytes())
        task_hash.update(np.asarray(task.visible_mask, dtype=np.uint8).tobytes())
        if public_support_bounds_for_task is not None:
            support_lower, support_upper = public_support_bounds_for_task(task)
            task_hash.update(np.asarray(support_lower, dtype=np.float32).tobytes())
            task_hash.update(np.asarray(support_upper, dtype=np.float32).tobytes())
    task_digest = task_hash.hexdigest()
    start_update = 0
    elapsed_before = 0.0
    if resume_checkpoint is not None:
        checkpoint_path = Path(resume_checkpoint).resolve()
        checkpoint_root = Path(checkpoint_dir).resolve()
        if not checkpoint_path.is_relative_to(checkpoint_root):
            raise ValueError("Inverse resume checkpoint must belong to the declared directory")
        saved = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
        expected = {
            "schema_version": 1,
            "task": "wind_position_completion",
            "seed": int(seed),
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "train_task_sha256": task_digest,
            "initial_denoiser_hash": initial_hash,
            "frozen_state_hashes": frozen_hashes_before,
        }
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("Inverse resume checkpoint does not match task, initialization, or frozen forward")
        if saved.get("public_support_input_label", "none") != public_support_input_label:
            raise ValueError("Inverse resume checkpoint public support input differs from this run")
        saved_transform = saved.get("sensor_velocity_transform")
        if saved_transform != transform_record and not (
            allow_sensor_transform_transition and saved_transform is None and transform_record is not None
        ):
            raise ValueError("Inverse resume checkpoint sensor velocity transform differs from this run")
        start_update = int(saved["update_count"])
        if not 0 < start_update <= updates:
            raise ValueError("Inverse resume update must not exceed the cumulative target")
        graph.load_state_dict(saved["graph_state_dict"], strict=True)
        dense.load_state_dict(saved["dense_state_dict"], strict=True)
        optimizers[0].load_state_dict(saved["graph_optimizer_state_dict"])
        optimizers[1].load_state_dict(saved["dense_optimizer_state_dict"])
        index_rng.bit_generator.state = saved["index_rng_state"]
        noise_rng.set_state(saved["noise_rng_state"])
        losses = (list(saved["graph_losses"]), list(saved["dense_losses"]))
        if len(losses[0]) != start_update or len(losses[1]) != start_update:
            raise ValueError("Inverse resume loss ledger does not match optimizer updates")
        elapsed_before = float(saved["elapsed_seconds_total"])
    started = time.monotonic()

    def save_checkpoint(update: int) -> None:
        if checkpoint_dir is None:
            return
        target = Path(checkpoint_dir) / f"updates_{update:06d}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        torch.save(
            {
                "schema_version": 1,
                "task": "wind_position_completion",
                "seed": int(seed),
                "steps": int(steps),
                "learning_rate": float(learning_rate),
                "public_support_input_label": public_support_input_label,
                "train_task_sha256": task_digest,
                "initial_denoiser_hash": initial_hash,
                "frozen_state_hashes": frozen_hashes_before,
                "update_count": update,
                "graph_state_dict": graph.state_dict(),
                "dense_state_dict": dense.state_dict(),
                "graph_optimizer_state_dict": optimizers[0].state_dict(),
                "dense_optimizer_state_dict": optimizers[1].state_dict(),
                "index_rng_state": index_rng.bit_generator.state,
                "noise_rng_state": noise_rng.get_state(),
                "sensor_velocity_transform": transform_record,
                "graph_losses": list(losses[0]),
                "dense_losses": list(losses[1]),
                "elapsed_seconds_total": elapsed_before + time.monotonic() - started,
            },
            temporary,
        )
        os.replace(temporary, target)

    for update in range(start_update, int(updates)):
        source = train_tasks[int(index_rng.integers(0, len(train_tasks)))]
        task = permute_wind_task(source, int(index_rng.integers(0, 2**31 - 1)))
        public_support_bounds = (
            None
            if public_support_bounds_for_task is None
            else public_support_bounds_for_task(task)
        )
        clean, condition, known = wind_condition_from_task(
            task, device=target_device,
            sensor_velocity_center_mps=sensor_velocity_center_mps,
            sensor_velocity_scale_mps=sensor_velocity_scale_mps,
            public_support_bounds_D=public_support_bounds,
        )
        candidate_provider = provider_factory(known)
        cached_state: torch.Tensor | None = None
        cached_links: PacketLinks | None = None

        def provider(
            state: torch.Tensor,
            current_condition: DiffusionCondition,
            provider_fn: LinksForCandidate = candidate_provider,
        ) -> PacketLinks:
            nonlocal cached_state, cached_links
            if cached_state is None:
                cached_state = state.detach().clone()
                cached_links = provider_fn(state, current_condition)
            elif not torch.equal(state.detach(), cached_state):
                raise RuntimeError("Matched inverse arms received different noisy candidate designs")
            assert cached_links is not None
            return cached_links

        timesteps = torch.randint(steps, (1,), generator=noise_rng).to(target_device)
        noise = torch.randn(clean.shape, generator=noise_rng, dtype=clean.dtype).to(target_device)
        for arm, model, optimizer, is_dense, history in zip(
            ("I-G", "I-dense"), (graph, dense), optimizers, (False, True), losses, strict=True
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(
                clean, condition, provider, dense=is_dense, timesteps=timesteps, noise=noise
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Wind inverse loss became nonfinite at update {update + 1}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.denoiser.parameters(), max_norm=1.0, error_if_nonfinite=True
            )
            if attempt_log_path is not None:
                attempt_path = Path(attempt_log_path)
                attempt_path.parent.mkdir(parents=True, exist_ok=True)
                with attempt_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({
                        "arm": arm,
                        "cumulative_update": update + 1,
                        "loss": float(loss.detach().cpu()),
                        "gradient_norm_preclip": float(gradient_norm.detach().cpu()),
                        "recorded_before_optimizer_step": True,
                        "timestamp_unix": time.time(),
                    }, sort_keys=True) + "\n")
            optimizer.step()
            history.append(float(loss.detach().cpu()))
        if (update + 1) % checkpoint_every == 0 or update + 1 == updates:
            save_checkpoint(update + 1)
    graph.eval()
    dense.eval()
    frozen_after = {name: _state_hash(module) for name, module in frozen_modules.items()}
    if frozen_hashes_before != frozen_after:
        raise RuntimeError("Forward/organizer weights changed during frozen inverse training")
    for name, module in frozen_modules.items():
        if any(
            not torch.equal(value, module.state_dict()[key].detach().cpu())
            for key, value in frozen_before[name].items()
        ):
            raise RuntimeError(f"Frozen {name} state changed during inverse training")
    return WindMatchedDiffusionResult(
        graph, dense, tuple(losses[0]), tuple(losses[1]),
        initial_hash, frozen_hashes_before, frozen_after, int(updates), int(updates),
    )


def wind_surrogate_predictor(model: nn.Module) -> Callable[[NativeCase, np.ndarray], np.ndarray]:
    """Make a frozen full-access candidate predictor for sensor consistency.

    Generated layouts have no OpenFOAM reference. This function never reads
    ``NativeCase.run`` and reports the model's prediction, not physical truth.
    """

    model.eval().requires_grad_(False)
    device = next(model.parameters()).device

    @torch.no_grad()
    def predict(candidate: NativeCase, coordinates_D: np.ndarray) -> np.ndarray:
        if candidate.run is not None:
            raise ValueError("A generated-layout surrogate candidate must not contain a CFD run")
        batch = case_batch(candidate, coordinates_D).to(device)
        backend = model.core.backend
        original_mode = backend.cover_mode
        try:
            backend.set_cover_mode("full_access")
            prepared = model.prepare_case(batch)
            output = model.predict_physical(
                prepared,
                batch.query_xy,
                batch.query_features,
                receiver_chunk_size=1024,
            )
        finally:
            backend.set_cover_mode(original_mode)
        return output[0].detach().cpu().numpy().astype(np.float32)

    return predict


def wind_sample_control(
    condition: DiffusionCondition,
    control: str,
) -> DiffusionCondition:
    """Apply declared observation ablations without changing hidden targets."""

    if control == "as_observed":
        return condition
    if control == "no_observations":
        return replace(condition, sensor_valid=torch.zeros_like(condition.sensor_valid))
    if control == "half_observations":
        valid = condition.sensor_valid.clone()
        valid[:, 1::2] = False
        return replace(condition, sensor_valid=valid)
    if control == "streamwise_only":
        features = condition.sensor_features.clone()
        features[:, :, 4:6] = 0.0
        return replace(condition, sensor_features=features)
    if control == "changed_observations":
        features = condition.sensor_features.clone()
        # Deterministic cyclic reassignment changes the field values while
        # retaining the declared observation coordinates and native units.
        features[:, :, 3:6] = features[:, :, 3:6].roll(shifts=3, dims=1)
        return replace(condition, sensor_features=features)
    raise ValueError(f"Unknown Wind observation control: {control}")


def _candidate_from_centers(known: WindCompletionKnown, centers_D: np.ndarray) -> NativeCase:
    return replace(
        known.template_case,
        module_centers=np.asarray(centers_D, dtype=np.float32).copy(),
        receiver_anchor_coords=None,
        receiver_anchor_weights=None,
        receiver_anchor_roles=None,
    )


def _wind_observation_rmse(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> float:
    if prediction.shape != reference.shape or prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError("Wind sensor prediction and stored reference must align")
    return float(np.sqrt(np.mean(np.square(prediction.astype(np.float64) - reference))))


def evaluate_matched_wind_completion(
    tasks: Sequence[WindCompletionTask],
    *,
    matched: WindMatchedDiffusionResult,
    provider_factory: Callable[[WindCompletionKnown], LinksForCandidate],
    surrogate_predictor: Callable[[NativeCase, np.ndarray], np.ndarray],
    independent_surrogate_predictor: Callable[[NativeCase, np.ndarray], np.ndarray] | None = None,
    surrogate_auditors: Mapping[str, Callable[[NativeCase, np.ndarray], np.ndarray]] | None = None,
    samples_per_task: int = 8,
    controls: Sequence[str] = (
        "as_observed",
        "no_observations",
        "changed_observations",
        "half_observations",
        "streamwise_only",
    ),
    seed: int = 21120,
    acceptable_observation_slack_mps: float = 0.5,
    include_rewired: bool = False,
    device: torch.device | str = "cpu",
    sensor_velocity_center_mps: Sequence[Sequence[float]] | Sequence[float] | None = None,
    sensor_velocity_scale_mps: Sequence[Sequence[float]] | Sequence[float] | None = None,
    public_support_bounds_for_task: Callable[
        [WindCompletionTask], tuple[Sequence[float], Sequence[float]]
    ] | None = None,
    public_support_input_label: str | None = None,
    completed_sample_keys: set[tuple[int, str, int, str]] | None = None,
    row_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> list[dict[str, object]]:
    """Sample matched arms and retain every attempted design and proposal trail.

    Development tasks and stored CFD are read only by this evaluator. An
    acceptable sample passes native geometry and a declared surrogate sensor
    threshold relative to the clean-layout surrogate error; this is not CFD
    validation of a new layout. Invalid attempts remain in the returned rows.
    """

    if not 8 <= len(tasks) <= 12 or any(task.partition != "development" for task in tasks):
        raise ValueError("Use 8..12 held development tasks for Wind inverse evaluation")
    if samples_per_task != 8:
        raise ValueError("The matched Wind inverse evaluation uses eight samples per task")
    if acceptable_observation_slack_mps < 0:
        raise ValueError("Observation slack must be nonnegative")
    if public_support_input_label is not None and not public_support_input_label.strip():
        raise ValueError("Public support input label must be nonempty when supplied")
    if public_support_bounds_for_task is None and public_support_input_label not in (None, "none"):
        raise ValueError("A public support label requires a task support bounds provider")
    if public_support_bounds_for_task is not None and public_support_input_label == "none":
        raise ValueError("A public support bounds provider cannot be labelled 'none'")
    audit_predictors = dict(surrogate_auditors or {})
    if any(not str(name).strip() for name in audit_predictors):
        raise ValueError("Surrogate audit predictor names must be nonempty")
    completed_keys = set(completed_sample_keys or ())
    target_device = torch.device(device)
    ordinary_arms = (("I-G", matched.graph_model, False), ("I-dense", matched.dense_model, True))
    rows: list[dict[str, object]] = []
    for task_index, task in enumerate(tasks):
        public_support_bounds = (
            None
            if public_support_bounds_for_task is None
            else public_support_bounds_for_task(task)
        )
        _, base_condition, known = wind_condition_from_task(
            task, device=target_device,
            sensor_velocity_center_mps=sensor_velocity_center_mps,
            sensor_velocity_scale_mps=sensor_velocity_scale_mps,
            public_support_bounds_D=public_support_bounds,
        )
        clean_case = _candidate_from_centers(known, task.clean_centers_D)
        clean_pred = surrogate_predictor(clean_case, known.observed_coordinates_D)
        clean_observed_rmse = _wind_observation_rmse(clean_pred, known.observed_velocity_mps)
        rotor_radius_D = float(known.template_case.module_features[0, 0])
        for control in controls:
            condition = wind_sample_control(base_condition, control)
            arms = ordinary_arms + (
                (("I-G-rewired", matched.graph_model, False),)
                if include_rewired and control == "as_observed"
                else ()
            )
            for sample_index in range(samples_per_task):
                # Reuse the same noise and reverse stream across controls as
                # well as arms, so observation ablations have paired draws.
                sample_seed = int(seed + task_index * 100003 + sample_index * 101)
                initial_generator = torch.Generator(device=target_device).manual_seed(sample_seed)
                initial_noise = torch.randn(
                    condition.known_state.shape,
                    device=target_device,
                    dtype=condition.known_state.dtype,
                    generator=initial_generator,
                )
                for arm, model, dense in arms:
                    sample_key = (int(task.row_index), str(control), int(sample_index), str(arm))
                    if sample_key in completed_keys:
                        continue
                    if arm == "I-G-rewired":
                        ordinary_provider = provider_factory(known)
                        if not isinstance(ordinary_provider, WindCandidatePacketProvider):
                            raise TypeError("Packet rewiring needs a Wind candidate packet provider")
                        provider = WindRewiredPacketProvider(
                            known, ordinary_provider.builder, seed=sample_seed ^ 0x7331
                        )
                    else:
                        provider = provider_factory(known)
                    reverse_generator = torch.Generator(device=target_device).manual_seed(sample_seed ^ 0x711A)
                    if target_device.type == "cuda":
                        torch.cuda.synchronize(target_device)
                    started = time.perf_counter()
                    trail = model.sample(
                        condition,
                        provider,
                        dense=dense,
                        initial_noise=initial_noise,
                        generator=reverse_generator,
                    )
                    final_links = provider(trail.final_state, condition)
                    frontier_evidence = getattr(provider.builder, "last_frontier_evidence", None)
                    final_interface = provider.last_interface
                    if target_device.type == "cuda":
                        torch.cuda.synchronize(target_device)
                    generation_s = time.perf_counter() - started
                    centers = candidate_centers_from_state(
                        trail.final_state, condition, known
                    ).detach().cpu().numpy()
                    validity = wind_geometry_validity(
                        centers,
                        rotor_radius_D,
                        support_lower_D=known.template_case.support.lower_D,
                        support_upper_D=known.template_case.support.upper_D,
                        support_name=(
                            "native_domain"
                            if public_support_bounds_for_task is None
                            else public_support_input_label or "public_candidate_domain"
                        ),
                    )
                    if public_support_bounds_for_task is not None:
                        native_validity = wind_geometry_validity(
                            centers,
                            rotor_radius_D,
                            support_lower_D=task.native_support_lower_D,
                            support_upper_D=task.native_support_upper_D,
                        )
                        validity["inside_native_domain"] = native_validity["inside_native_domain"]
                        validity["native_support_status"] = native_validity["native_support_status"]
                        validity["native_row_support_lower_D"] = task.native_support_lower_D.tolist()
                        validity["native_row_support_upper_D"] = task.native_support_upper_D.tolist()
                        validity["candidate_public_domain_valid"] = validity["inside_support"]
                    else:
                        native_validity = validity
                    generation_lower = (
                        condition.design_lower[0].detach().cpu().numpy()
                        if condition.design_lower is not None
                        else DESIGN_LOWER_D
                    )
                    generation_upper = (
                        condition.design_upper[0].detach().cpu().numpy()
                        if condition.design_upper is not None
                        else DESIGN_UPPER_D
                    )
                    inside_generation_bounds = bool(np.all(
                        (centers[:, :2] >= generation_lower[None, :])
                        & (centers[:, :2] <= generation_upper[None, :])
                    ))
                    native_geometry_valid = bool(
                        validity["finite"]
                        and inside_generation_bounds
                        and native_validity["inside_native_domain"]
                        and native_validity["rotors_nonoverlap"]
                    )
                    public_geometry_valid = bool(
                        validity["finite"]
                        and validity["inside_design_box"]
                        and inside_generation_bounds
                        and validity["inside_support"]
                        and validity["rotors_nonoverlap"]
                    )
                    candidate = _candidate_from_centers(known, centers)
                    query = np.concatenate(
                        (known.observed_coordinates_D, known.held_coordinates_D), axis=0
                    )
                    predicted = surrogate_predictor(candidate, query)
                    observed_count = known.observed_coordinates_D.shape[0]
                    observed_rmse = _wind_observation_rmse(
                        predicted[:observed_count], known.observed_velocity_mps
                    )
                    held_rmse = _wind_observation_rmse(
                        predicted[observed_count:], known.held_velocity_mps
                    )
                    independent_observed_rmse = None
                    independent_held_rmse = None
                    independent_disagreement = None
                    independent_predictions = None
                    if independent_surrogate_predictor is not None and control == "as_observed":
                        independent = independent_surrogate_predictor(candidate, query)
                        independent_predictions = independent
                        independent_observed_rmse = _wind_observation_rmse(
                            independent[:observed_count], known.observed_velocity_mps
                        )
                        independent_held_rmse = _wind_observation_rmse(
                            independent[observed_count:], known.held_velocity_mps
                        )
                        independent_disagreement = _wind_observation_rmse(
                            independent, predicted
                        )
                    audit_results = {}
                    if control == "as_observed":
                        for audit_name, audit_predictor in audit_predictors.items():
                            audited = audit_predictor(candidate, query)
                            audit_results[audit_name] = {
                                "predictions_mps": np.asarray(audited, dtype=np.float32).tolist(),
                                "observed_rmse_mps": _wind_observation_rmse(
                                    audited[:observed_count], known.observed_velocity_mps
                                ),
                                "held_rmse_mps": _wind_observation_rmse(
                                    audited[observed_count:], known.held_velocity_mps
                                ),
                                "disagreement_with_primary_rmse_mps": _wind_observation_rmse(
                                    audited, predicted
                                ),
                            }
                    complete_candidate_s = time.perf_counter() - started
                    acceptable = bool(
                        validity["finite"]
                        and validity["inside_design_box"]
                        and inside_generation_bounds
                        and validity["inside_support"]
                        and validity["inside_native_domain"]
                        and validity["rotors_nonoverlap"]
                        and observed_rmse <= clean_observed_rmse + acceptable_observation_slack_mps
                    )
                    trail_centers = [
                        candidate_centers_from_state(state, condition, known)
                        .detach().cpu().numpy().tolist()
                        for state in trail.states
                    ]
                    trail_geometry_statuses = []
                    for timestep, state_centers in zip(
                        trail.timesteps, trail_centers, strict=True
                    ):
                        state_array = np.asarray(state_centers, dtype=np.float32)
                        state_public = wind_geometry_validity(
                            state_array,
                            rotor_radius_D,
                            support_lower_D=known.template_case.support.lower_D,
                            support_upper_D=known.template_case.support.upper_D,
                            support_name=public_support_input_label or "public_candidate_domain",
                        )
                        state_native = wind_geometry_validity(
                            state_array,
                            rotor_radius_D,
                            support_lower_D=task.native_support_lower_D,
                            support_upper_D=task.native_support_upper_D,
                        )
                        trail_geometry_statuses.append({
                            "timestep": int(timestep),
                            "finite": bool(state_public["finite"]),
                            "inside_design_box": bool(state_public["inside_design_box"]),
                            "inside_generation_design_bounds": bool(np.all(
                                (state_array[:, :2] >= generation_lower[None, :])
                                & (state_array[:, :2] <= generation_upper[None, :])
                            )),
                            "inside_compact_public_domain": bool(state_public["inside_support"]),
                            "compact_rotors_nonoverlap": bool(state_public["rotors_nonoverlap"]),
                            "inside_native_row_support": bool(state_native["inside_native_domain"]),
                            "native_rotors_nonoverlap": bool(state_native["rotors_nonoverlap"]),
                        })
                    row_result = {
                            "row_index": task.row_index,
                            "partition": task.partition,
                            "control": control,
                            "sample_index": sample_index,
                            "sample_seed": sample_seed,
                            "arm": arm,
                            "hidden_count": int((~known.visible_mask).sum()),
                            "rotor_radius_D": rotor_radius_D,
                            "visible_mask": known.visible_mask.tolist(),
                            "clean_centers_D": task.clean_centers_D.tolist(),
                            "generated_centers_D": centers.tolist(),
                            "candidate_public_support_lower_D": known.template_case.support.lower_D.tolist(),
                            "candidate_public_support_upper_D": known.template_case.support.upper_D.tolist(),
                            "native_support_lower_D": task.native_support_lower_D.tolist(),
                            "native_support_upper_D": task.native_support_upper_D.tolist(),
                            "public_support_input": (
                                "none"
                                if public_support_bounds_for_task is None
                                else public_support_input_label or "caller_supplied_public_support"
                            ),
                            "generation_design_lower_D": (
                                condition.design_lower[0].detach().cpu().tolist()
                                if condition.design_lower is not None
                                else DESIGN_LOWER_D.tolist()
                            ),
                            "generation_design_upper_D": (
                                condition.design_upper[0].detach().cpu().tolist()
                                if condition.design_upper is not None
                                else DESIGN_UPPER_D.tolist()
                            ),
                            "inside_generation_design_bounds": inside_generation_bounds,
                            "hidden_set_error_D": hidden_set_error_D(
                                centers, task.clean_centers_D, known.visible_mask
                            ),
                            "observed_sensor_coordinates_D": known.observed_coordinates_D.tolist(),
                            "held_sensor_coordinates_D": known.held_coordinates_D.tolist(),
                            "surrogate_query_coordinates_D": query.tolist(),
                            "geometry": validity,
                            "geometry_valid_before_repair": public_geometry_valid,
                            "geometry_valid_after_repair": public_geometry_valid,
                            "native_geometry_valid_before_repair": native_geometry_valid,
                            "native_geometry_valid_after_repair": native_geometry_valid,
                            "repair_applied": False,
                            "repair_method": "none_identity",
                            "trail_geometry_statuses": trail_geometry_statuses,
                            "clean_surrogate_observed_rmse_mps": clean_observed_rmse,
                            "surrogate_observed_rmse_mps": observed_rmse,
                            "surrogate_held_rmse_mps": held_rmse,
                            "surrogate_primary_predictions_mps": predicted.tolist(),
                            "independent_surrogate_observed_rmse_mps": independent_observed_rmse,
                            "independent_surrogate_held_rmse_mps": independent_held_rmse,
                            "independent_surrogate_disagreement_rmse_mps": independent_disagreement,
                            "independent_surrogate_predictions_mps": (
                                None
                                if independent_predictions is None
                                else independent_predictions.tolist()
                            ),
                            "surrogate_audit_results": audit_results,
                            "acceptable": acceptable,
                            "trail_timesteps": list(trail.timesteps),
                            "trail_centers_D": trail_centers,
                            "final_module_source_access": final_links.module_source[0].detach().cpu().tolist(),
                            "final_sensor_source_access": final_links.sensor_source[0].detach().cpu().tolist(),
                            "final_frontier_evidence": (
                                None if frontier_evidence is None else asdict(frontier_evidence)
                            ),
                            "final_permission_statuses": (
                                None
                                if final_interface is None or not hasattr(final_interface, "permission_statuses")
                                else dict(final_interface.permission_statuses)
                            ),
                            "fixed_weight_packet_rewire_swaps": (
                                dict(provider.total_swap_counts)
                                if isinstance(provider, WindRewiredPacketProvider)
                                else None
                            ),
                            "organizer_calls": trail.organizer_calls + 1,
                            "full_forward_calls": (
                                1
                                + int(independent_disagreement is not None)
                                + len(audit_results)
                            ),
                            "generation_seconds": generation_s,
                            "complete_candidate_seconds": complete_candidate_s,
                    }
                    rows.append(row_result)
                    if row_callback is not None:
                        row_callback(row_result)
    return rows

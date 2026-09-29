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
from windfarm.geometry import HUB_HEIGHT_D, POSITIONAL_SCALE_D, U_REF_MPS

DESIGN_LOWER_D = np.asarray((-15.0, -15.0), dtype=np.float32)
DESIGN_UPPER_D = np.asarray((15.0, 15.0), dtype=np.float32)
SENSOR_X_D = np.linspace(-7.0, 17.0, 6, dtype=np.float32)
SENSOR_Y_D = np.linspace(-5.0, 5.0, 4, dtype=np.float32)
SENSOR_OBSERVED_INDICES = tuple(index for index in range(24) if index % 3 != 0)
SENSOR_HELD_INDICES = tuple(index for index in range(24) if index % 3 == 0)


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
    intended = np.asarray(
        [(float(x), float(y), HUB_HEIGHT_D) for y in SENSOR_Y_D for x in SENSOR_X_D],
        dtype=np.float32,
    )
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
        observed_coordinates_D=panel.native_coordinates_D[observed].copy(),
        observed_velocity_mps=panel.reference_velocity_mps[observed].copy(),
        held_coordinates_D=panel.native_coordinates_D[held].copy(),
        held_velocity_mps=panel.reference_velocity_mps[held].copy(),
    )


def _position_logits(xy_D: np.ndarray) -> np.ndarray:
    fraction = (np.asarray(xy_D, dtype=np.float32) - DESIGN_LOWER_D) / (
        DESIGN_UPPER_D - DESIGN_LOWER_D
    )
    fraction = np.clip(fraction, 1.0e-4, 1.0 - 1.0e-4)
    return np.log(fraction / (1.0 - fraction)).astype(np.float32)


def wind_condition_from_task(
    task: WindCompletionTask,
    *,
    device: torch.device | str = "cpu",
    sensor_velocity_center_mps: Sequence[Sequence[float]] | Sequence[float] | None = None,
    sensor_velocity_scale_mps: Sequence[Sequence[float]] | Sequence[float] | None = None,
) -> tuple[torch.Tensor, DiffusionCondition, WindCompletionKnown]:
    """Build no-leakage condition and a separate clean denoising target."""

    known = known_from_wind_task(task)
    count = int(task.clean_centers_D.shape[0])
    visible = known.visible_mask
    clean = _position_logits(task.clean_centers_D[:, :2])
    visible_logits = np.zeros_like(clean)
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
    lower = state.new_tensor(DESIGN_LOWER_D)
    upper = state.new_tensor(DESIGN_UPPER_D)
    xy = lower + torch.sigmoid(state[0]) * (upper - lower)
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
) -> dict[str, float | bool]:
    """Record support-box and rotor-overlap checks before any rejection."""

    centers = np.asarray(centers_D, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or not np.all(np.isfinite(centers)):
        return {"finite": False, "inside_design_box": False, "inside_native_domain": False,
                "rotors_nonoverlap": False, "min_spacing_D": float("nan")}
    inside = bool(np.all((centers[:, :2] >= DESIGN_LOWER_D) & (centers[:, :2] <= DESIGN_UPPER_D)))
    if (support_lower_D is None) != (support_upper_D is None):
        raise ValueError("Native support bounds must be supplied together")
    native = True if support_lower_D is None else bool(np.all(
        (centers >= np.asarray(support_lower_D)[None, :])
        & (centers <= np.asarray(support_upper_D)[None, :])
    ))
    difference = centers[:, None, :2] - centers[None, :, :2]
    distance = np.linalg.norm(difference, axis=-1)
    np.fill_diagonal(distance, np.inf)
    minimum = float(np.min(distance)) if len(centers) > 1 else float("inf")
    return {
        "finite": True,
        "inside_design_box": inside,
        "inside_native_domain": native,
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
    )


InterfaceForWindCandidate = Callable[[torch.Tensor, WindCandidateKnown], InteractionInterface]


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
        builder: WindCandidateInterfaceBuilder,
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
        clean, condition, known = wind_condition_from_task(
            task, device=target_device,
            sensor_velocity_center_mps=sensor_velocity_center_mps,
            sensor_velocity_scale_mps=sensor_velocity_scale_mps,
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
    target_device = torch.device(device)
    ordinary_arms = (("I-G", matched.graph_model, False), ("I-dense", matched.dense_model, True))
    rows: list[dict[str, object]] = []
    for task_index, task in enumerate(tasks):
        _, base_condition, known = wind_condition_from_task(
            task, device=target_device,
            sensor_velocity_center_mps=sensor_velocity_center_mps,
            sensor_velocity_scale_mps=sensor_velocity_scale_mps,
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
                    if arm == "I-G-rewired":
                        ordinary_provider = provider_factory(known)
                        if not isinstance(ordinary_provider, WindCandidatePacketProvider) or not isinstance(
                            ordinary_provider.builder, WindCandidateInterfaceBuilder
                        ):
                            raise TypeError("Packet rewiring needs the frozen Wind candidate interface")
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
                    if independent_surrogate_predictor is not None and control == "as_observed":
                        independent = independent_surrogate_predictor(candidate, query)
                        independent_observed_rmse = _wind_observation_rmse(
                            independent[:observed_count], known.observed_velocity_mps
                        )
                        independent_held_rmse = _wind_observation_rmse(
                            independent[observed_count:], known.held_velocity_mps
                        )
                        independent_disagreement = _wind_observation_rmse(
                            independent, predicted
                        )
                    complete_candidate_s = time.perf_counter() - started
                    acceptable = bool(
                        validity["finite"]
                        and validity["inside_design_box"]
                        and validity["inside_native_domain"]
                        and validity["rotors_nonoverlap"]
                        and observed_rmse <= clean_observed_rmse + acceptable_observation_slack_mps
                    )
                    trail_centers = [
                        candidate_centers_from_state(state, condition, known)
                        .detach().cpu().numpy().tolist()
                        for state in trail.states
                    ]
                    rows.append(
                        {
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
                            "native_support_lower_D": known.template_case.support.lower_D.tolist(),
                            "native_support_upper_D": known.template_case.support.upper_D.tolist(),
                            "hidden_set_error_D": hidden_set_error_D(
                                centers, task.clean_centers_D, known.visible_mask
                            ),
                            "geometry": validity,
                            "clean_surrogate_observed_rmse_mps": clean_observed_rmse,
                            "surrogate_observed_rmse_mps": observed_rmse,
                            "surrogate_held_rmse_mps": held_rmse,
                            "independent_surrogate_observed_rmse_mps": independent_observed_rmse,
                            "independent_surrogate_held_rmse_mps": independent_held_rmse,
                            "independent_surrogate_disagreement_rmse_mps": independent_disagreement,
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
                            "full_forward_calls": 1 + int(independent_disagreement is not None),
                            "generation_seconds": generation_s,
                            "complete_candidate_seconds": complete_candidate_s,
                        }
                    )
    return rows

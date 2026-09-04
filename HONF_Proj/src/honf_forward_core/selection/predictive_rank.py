"""Deterministic predictive-rank selection for a prepared fixed edge bank.

The selector is intentionally outside the organizer.  It compares query-field
predictions from every non-empty support mask with the complete prepared bank
on deterministic case probes, then caches only a query-routing mask.  Learned
edge states and incidence matrices are never recomputed or modified.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


EPS = 1.0e-12
MAX_EXHAUSTIVE_EDGES = 12
DEFAULT_MASK_BATCH_SIZE = 8


def _evenly_spaced_indices(count: int, limit: int, device: torch.device) -> torch.Tensor:
    """Return deterministic integer indices including both endpoints."""

    if limit >= count:
        return torch.arange(count, device=device)
    if limit <= 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    numerator = torch.arange(limit, dtype=torch.long, device=device) * (count - 1)
    return torch.div(numerator, limit - 1, rounding_mode="floor")


def _lexicographic_xy(points: torch.Tensor) -> torch.Tensor:
    """Sort points by x then y without depending on module input order."""

    if points.shape[0] <= 1:
        return points
    by_y = torch.argsort(points[:, 1], stable=True)
    ordered = points.index_select(0, by_y)
    by_x = torch.argsort(ordered[:, 0], stable=True)
    return ordered.index_select(0, by_x)


def build_deterministic_case_probes(
    env_coords: torch.Tensor,
    module_centers: torch.Tensor,
    module_present: torch.Tensor,
    *,
    module_radius: float,
    limit: int = 256,
    source: str = "environment_plus_module_local",
    domain_length_x: float | None = None,
    domain_length_y: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build padded environment and eight-direction module-local probes.

    Returns coordinates ``[B,P,2]`` plus a boolean validity mask ``[B,P]``.
    Active module-local probes are lexicographically sorted, making the probe
    set invariant to a consistent permutation or trailing padding of modules.
    """

    if source != "environment_plus_module_local":
        raise ValueError("Unsupported case probe source.")
    if int(limit) <= 0:
        raise ValueError("Probe limit must be positive.")
    if module_centers.ndim != 3 or module_centers.shape[-1] != 2:
        raise ValueError("module_centers must have shape [B,M,2].")
    if module_present.shape != module_centers.shape[:2]:
        raise ValueError("module_present must have shape [B,M].")
    batch = int(module_centers.shape[0])
    if env_coords.ndim == 2:
        env_coords = env_coords.unsqueeze(0).expand(batch, -1, -1)
    if env_coords.ndim != 3 or env_coords.shape[0] != batch or env_coords.shape[-1] != 2:
        raise ValueError("env_coords must have shape [E,2] or [B,E,2].")

    # Fixed directions avoid trigonometric device differences and cover axes
    # plus diagonals at a radius just outside the physical module surface.
    diagonal = 2.0**-0.5
    directions = module_centers.new_tensor(
        [
            [1.0, 0.0],
            [diagonal, diagonal],
            [0.0, 1.0],
            [-diagonal, diagonal],
            [-1.0, 0.0],
            [-diagonal, -diagonal],
            [0.0, -1.0],
            [diagonal, -diagonal],
        ]
    )
    radius = 1.05 * float(module_radius)
    cases: list[torch.Tensor] = []
    for case_index in range(batch):
        active = module_present[case_index] > 0.5
        centers = module_centers[case_index, active]
        local = centers[:, None, :] + radius * directions[None, :, :]
        local = local.reshape(-1, 2)
        if domain_length_x is not None:
            local[:, 0] = local[:, 0].clamp(0.0, float(domain_length_x))
        if domain_length_y is not None:
            local[:, 1] = local[:, 1].clamp(0.0, float(domain_length_y))
        local = _lexicographic_xy(local)
        points = torch.cat([env_coords[case_index], local], dim=0)
        indices = _evenly_spaced_indices(
            int(points.shape[0]), min(int(limit), int(points.shape[0])), points.device
        )
        cases.append(points.index_select(0, indices))

    width = max(int(points.shape[0]) for points in cases)
    probes = module_centers.new_zeros(batch, width, 2)
    valid = torch.zeros(batch, width, dtype=torch.bool, device=module_centers.device)
    for case_index, points in enumerate(cases):
        count = int(points.shape[0])
        probes[case_index, :count] = points
        valid[case_index, :count] = True
    return probes, valid


def enumerate_nonempty_edge_masks(
    edge_count: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Enumerate all non-empty masks by increasing K and stable bit code."""

    count = int(edge_count)
    if count <= 0:
        raise ValueError("edge_count must be positive.")
    if count > MAX_EXHAUSTIVE_EDGES:
        raise ValueError(
            f"Exhaustive support search is limited to {MAX_EXHAUSTIVE_EDGES} edges."
        )
    codes = sorted(range(1, 1 << count), key=lambda code: (code.bit_count(), code))
    return torch.tensor(
        [[bool(code & (1 << edge)) for edge in range(count)] for code in codes],
        dtype=torch.bool,
        device=device,
    )


def _repeat_case_tensor(value: torch.Tensor, batch: int, repeats: int) -> torch.Tensor:
    """Repeat a batch-leading tensor while preserving shared/scalar tensors."""

    if value.ndim == 0 or int(value.shape[0]) != batch:
        return value
    expanded = value.unsqueeze(1).expand(batch, repeats, *value.shape[1:])
    return expanded.reshape(batch * repeats, *value.shape[1:])


def _repeat_organizer(
    organizer: dict[str, Any], batch: int, repeats: int
) -> dict[str, Any]:
    return {
        key: _repeat_case_tensor(value, batch, repeats) if torch.is_tensor(value) else value
        for key, value in organizer.items()
    }


def _relative_rms(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return overall ``[B,S]`` and per-channel ``[B,S,F]`` discrepancies."""

    weights = valid[:, None, :, None].to(dtype=reference.dtype)
    difference_square = (candidate - reference[:, None, :, :]).square() * weights
    reference_square = reference[:, None, :, :].square() * weights
    channel_numerator = difference_square.sum(dim=2)
    channel_denominator = reference_square.sum(dim=2).clamp_min(EPS)
    channel = torch.sqrt(channel_numerator / channel_denominator)
    overall = torch.sqrt(
        difference_square.sum(dim=(2, 3))
        / reference_square.sum(dim=(2, 3)).clamp_min(EPS)
    )
    return overall, channel


def select_probe_fidelity_support(
    organizer: dict[str, Any],
    global_token: torch.Tensor,
    probe_xy: torch.Tensor,
    probe_valid: torch.Tensor,
    decode_fn: Callable[[torch.Tensor, dict[str, Any], torch.Tensor], torch.Tensor],
    *,
    relative_rms_tolerance: float,
    channel_tolerance: float,
    search: str = "exhaustive_small_bank",
    mask_batch_size: int = DEFAULT_MASK_BATCH_SIZE,
) -> dict[str, Any]:
    """Select the smallest routing mask faithful to the complete edge bank."""

    if search != "exhaustive_small_bank":
        raise ValueError("Unsupported predictive-rank search mode.")
    if float(relative_rms_tolerance) < 0.0 or float(channel_tolerance) < 0.0:
        raise ValueError("Probe fidelity tolerances must be nonnegative.")
    hyper_state = organizer.get("hyper_state")
    if not torch.is_tensor(hyper_state) or hyper_state.ndim != 3:
        raise ValueError("organizer must contain hyper_state [B,K,H].")
    batch, edge_count, _ = hyper_state.shape
    if global_token.ndim != 2 or global_token.shape[0] != batch:
        raise ValueError("global_token must have shape [B,H].")
    if probe_xy.ndim != 3 or probe_xy.shape[0] != batch or probe_xy.shape[-1] != 2:
        raise ValueError("probe_xy must have shape [B,P,2].")
    if probe_valid.shape != probe_xy.shape[:2]:
        raise ValueError("probe_valid must have shape [B,P].")
    if bool((probe_valid.sum(dim=-1) <= 0).any()):
        raise ValueError("Every case must contain at least one valid probe.")

    masks = enumerate_nonempty_edge_masks(int(edge_count), device=hyper_state.device)
    full_mask = torch.ones(edge_count, dtype=torch.bool, device=hyper_state.device)
    candidate_masks = masks[:-1]
    overall_parts: list[torch.Tensor] = []
    channel_parts: list[torch.Tensor] = []
    with torch.no_grad():
        reference = decode_fn(probe_xy, organizer, global_token)
        if reference.ndim != 3 or reference.shape[:2] != probe_xy.shape[:2]:
            raise ValueError("decode_fn must return fields with shape [B,P,F].")
        for start in range(0, int(candidate_masks.shape[0]), max(int(mask_batch_size), 1)):
            mask_chunk = candidate_masks[start : start + max(int(mask_batch_size), 1)]
            repeats = int(mask_chunk.shape[0])
            repeated_organizer = _repeat_organizer(organizer, int(batch), repeats)
            repeated_mask = mask_chunk[None, :, :].expand(batch, -1, -1).reshape(
                batch * repeats, edge_count
            )
            repeated_organizer["predictive_edge_mask"] = repeated_mask.to(
                dtype=hyper_state.dtype
            )
            repeated_probe = probe_xy[:, None, :, :].expand(
                batch, repeats, *probe_xy.shape[1:]
            ).reshape(batch * repeats, *probe_xy.shape[1:])
            repeated_global = _repeat_case_tensor(global_token, int(batch), repeats)
            prediction = decode_fn(repeated_probe, repeated_organizer, repeated_global)
            prediction = prediction.reshape(batch, repeats, probe_xy.shape[1], -1)
            overall, channel = _relative_rms(prediction, reference, probe_valid)
            overall_parts.append(overall)
            channel_parts.append(channel)

    if overall_parts:
        candidate_overall = torch.cat(overall_parts, dim=1)
        candidate_channel = torch.cat(channel_parts, dim=1)
    else:
        candidate_overall = reference.new_empty((batch, 0))
        candidate_channel = reference.new_empty((batch, 0, reference.shape[-1]))
    selected_masks: list[torch.Tensor] = []
    selected_overall: list[torch.Tensor] = []
    selected_channel: list[torch.Tensor] = []
    selected_codes: list[int] = []
    candidate_ranks = candidate_masks.sum(dim=-1)
    for case_index in range(int(batch)):
        feasible = (candidate_overall[case_index] <= float(relative_rms_tolerance)) & (
            candidate_channel[case_index].amax(dim=-1) <= float(channel_tolerance)
        )
        feasible_indices = torch.nonzero(feasible, as_tuple=False).reshape(-1)
        if feasible_indices.numel() == 0:
            chosen_mask = full_mask
            chosen_overall = reference.new_zeros(())
            chosen_channel = reference.new_zeros(reference.shape[-1])
            chosen_code = (1 << int(edge_count)) - 1
        else:
            ranks = candidate_ranks.index_select(0, feasible_indices)
            minimum_rank = ranks.amin()
            smallest = feasible_indices[ranks == minimum_rank]
            errors = candidate_overall[case_index].index_select(0, smallest)
            chosen_index = smallest[torch.argmin(errors)]
            chosen_mask = candidate_masks[chosen_index]
            chosen_overall = candidate_overall[case_index, chosen_index]
            chosen_channel = candidate_channel[case_index, chosen_index]
            powers = 1 << torch.arange(edge_count, device=chosen_mask.device)
            chosen_code = int((chosen_mask.to(torch.long) * powers).sum().item())
        selected_masks.append(chosen_mask)
        selected_overall.append(chosen_overall)
        selected_channel.append(chosen_channel)
        selected_codes.append(chosen_code)

    selected_mask = torch.stack(selected_masks, dim=0)
    output = dict(organizer)
    output.update(
        {
            "predictive_selected_mask": selected_mask.to(dtype=hyper_state.dtype).detach(),
            "predictive_edge_count": selected_mask.sum(dim=-1).to(dtype=hyper_state.dtype),
            "predictive_edge_selected_code": torch.tensor(
                selected_codes, device=hyper_state.device, dtype=torch.long
            ),
            "predictive_probe_relative_rms": torch.stack(selected_overall).detach(),
            "predictive_probe_channel_relative_rms": torch.stack(selected_channel).detach(),
            "predictive_probe_count": probe_valid.sum(dim=-1).to(dtype=hyper_state.dtype),
            "predictive_full_edge_count": hyper_state.new_full(
                (batch,), float(edge_count)
            ),
            "predictive_candidate_count": hyper_state.new_full(
                (batch,), float(masks.shape[0])
            ),
            "predictive_probe_relative_rms_tolerance": hyper_state.new_full(
                (batch,), float(relative_rms_tolerance)
            ),
            "predictive_probe_channel_tolerance": hyper_state.new_full(
                (batch,), float(channel_tolerance)
            ),
            "case_edge_selection_mode": "probe_fidelity",
            "case_edge_probe_source": "environment_plus_module_local",
            "case_edge_probe_search": search,
        }
    )
    if bool((~selected_mask).any()):
        # Keep an all-edge fallback on the byte-identical historical decoder
        # path. The all-ones selection remains cached above as metadata, while
        # this routing-only key exists only when work can actually be removed.
        output["predictive_edge_mask"] = selected_mask.to(
            dtype=hyper_state.dtype
        ).detach()
    return output

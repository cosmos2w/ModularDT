"""Matched receiver-conditioned readouts over prepared hyperedge states.

The primary pool-first and edge-first maps share one parameter inventory and
the same prepared anchor/query/geometry information.  This module does not
read physical source values; its payload is the completed scene-prepared
hyperedge state supplied by the owning model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .atlas import validate_cover_anchor_identity
from .contracts import ReceiverCover

ReceiverReduction = Literal["pool_first", "edge_first"]
ReceiverAccess = Literal["local_cover", "full_bank_diagnostic"]
GeometryMode = Literal["native_signed", "scene_frame_normalized"]

READOUT_WIDTH = 64
READOUT_QUERY_TILE = 512
READOUT_FULL_BANK_QUERY_TILE = 64
READOUT_ANCHOR_TILE = 32
READOUT_TRACE_MAX_ELEMENTS = 1_048_576


def signed_geometry_features(relative_coordinates: Tensor) -> Tensor:
    """Return the shared seven-feature map for each signed coordinate."""
    if relative_coordinates.ndim < 1 or relative_coordinates.shape[-1] < 1:
        raise ValueError("relative_coordinates must have a nonempty coordinate axis.")
    if not relative_coordinates.is_floating_point():
        raise TypeError("relative_coordinates must use a floating point dtype.")
    pi = relative_coordinates.new_tensor(torch.pi)
    pieces = []
    for frequency in (1.0, 0.5, 0.25):
        phase = pi * frequency * relative_coordinates
        pieces.extend((torch.sin(phase), torch.cos(phase)))
    # The asinh channel is first by schema, followed by the three sin/cos pairs.
    result = torch.stack((torch.asinh(relative_coordinates), *pieces), dim=-1)
    return result.flatten(start_dim=-2)


def _tensor_stamps(tensors: tuple[Tensor, ...]) -> tuple[tuple[int, int, int], ...]:
    stamps = []
    for tensor in tensors:
        try:
            version = int(tensor._version)
        except RuntimeError:
            version = -1
        stamps.append((id(tensor), int(tensor.data_ptr()), version))
    return tuple(stamps)


@dataclass(frozen=True)
class PreparedHyperedgeReadoutValues:
    """Owner-bound ``A_Z Z`` projection of one completed H scene state."""

    values: Tensor  # [B,E,64], projected from the exact completed H state
    anchor_ids: Tensor  # integer [B,E]
    source_state: Tensor  # exact H ``processed_state`` object
    frame_key: object
    scene_version: object
    owner_token: object
    _tensor_versions: tuple[tuple[int, int, int], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.values.ndim != 3 or self.values.shape[-1] != READOUT_WIDTH:
            raise ValueError("Prepared readout values must have shape [B,E,64].")
        if self.source_state.ndim != 3 or self.source_state.shape[:2] != self.values.shape[:2]:
            raise ValueError("Prepared readout source must be the matching [B,E,D] state.")
        if self.anchor_ids.shape != self.values.shape[:2] or self.anchor_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Prepared readout IDs must be integer [B,E].")
        if any(value.device != self.values.device for value in (self.anchor_ids, self.source_state)):
            raise ValueError("Prepared readout values, source and IDs must share a device.")
        object.__setattr__(self, "_tensor_versions", _tensor_stamps((self.values, self.anchor_ids, self.source_state)))

    def validate_unchanged(self) -> None:
        if _tensor_stamps((self.values, self.anchor_ids, self.source_state)) != self._tensor_versions:
            raise RuntimeError("Prepared hyperedge readout values changed after construction.")

    def validate_binding(self, *, frame_key=None, scene_version=None, owner_token=None) -> None:
        self.validate_unchanged()
        if frame_key is not None and frame_key != self.frame_key:
            raise ValueError("Prepared hyperedge readout uses a different coordinate frame.")
        if scene_version is not None and scene_version != self.scene_version:
            raise ValueError("Prepared hyperedge readout is stale for the current scene.")
        if owner_token is not None and owner_token != self.owner_token:
            raise ValueError("Prepared hyperedge readout belongs to another model cache owner.")

    def tensors(self) -> tuple[Tensor, Tensor, Tensor]:
        return self.values, self.anchor_ids, self.source_state


class HyperedgeReceiverReadout(nn.Module):
    """Width-64 matched pool-first/edge-first receiver correction.

    ``geometry_mode='native_signed'`` retains the dataset's supplied signed
    coordinate units. ``scene_frame_normalized`` divides query-minus-anchor
    offsets by the scene's explicit frame scale, as required by Thermal.
    """

    def __init__(
        self,
        *,
        coordinate_dim: int,
        reduction: ReceiverReduction,
        access: ReceiverAccess,
        geometry_mode: GeometryMode,
        anchor_state_dim: int = 128,
        query_state_dim: int = 128,
        context_dim: int = 128,
        init_seed: int = 0,
    ) -> None:
        super().__init__()
        if coordinate_dim not in (2, 3):
            raise ValueError("The receiver readout supports two or three native coordinates.")
        if (anchor_state_dim, query_state_dim, context_dim) != (128, 128, 128):
            raise ValueError("Round6 fixes anchor/query/context widths at 128 and reader width at 64.")
        if reduction not in ("pool_first", "edge_first"):
            raise ValueError("reduction must be pool_first or edge_first.")
        if access not in ("local_cover", "full_bank_diagnostic"):
            raise ValueError("access must be local_cover or full_bank_diagnostic.")
        if access == "full_bank_diagnostic" and reduction != "edge_first":
            raise ValueError("The full-bank diagnostic is defined only for edge_first reduction.")
        if geometry_mode not in ("native_signed", "scene_frame_normalized"):
            raise ValueError("Unknown receiver geometry scale convention.")
        if type(init_seed) is not int:
            raise TypeError("init_seed must be an integer.")
        self.coordinate_dim = int(coordinate_dim)
        self.reduction = reduction
        self.access = access
        self.geometry_mode = geometry_mode
        self.init_seed = init_seed
        self.anchor_projection = nn.Linear(anchor_state_dim, READOUT_WIDTH, bias=False)
        self.query_projection = nn.Linear(query_state_dim, READOUT_WIDTH, bias=False)
        self.geometry_projection = nn.Linear(7 * coordinate_dim, READOUT_WIDTH, bias=False)
        self.output_projection = nn.Linear(READOUT_WIDTH, context_dim, bias=False)
        self.hidden_bias = nn.Parameter(torch.zeros(READOUT_WIDTH))
        self._initialize_input_projections()
        nn.init.zeros_(self.output_projection.weight)

    def _initialize_input_projections(self) -> None:
        # Stable named initialization makes pool_first and edge_first births
        # tensor-identical while keeping every input map nonzero.
        with torch.no_grad():
            for name in ("anchor_projection", "query_projection", "geometry_projection"):
                layer = getattr(self, name)
                generator = torch.Generator(device="cpu").manual_seed(
                    self.init_seed_for(name)
                )
                values = torch.empty(tuple(layer.weight.shape), dtype=torch.float32)
                nn.init.xavier_uniform_(values, generator=generator)
                layer.weight.copy_(values.to(layer.weight))
            self.hidden_bias.zero_()

    def init_seed_for(self, name: str) -> int:
        import hashlib

        digest = hashlib.blake2b(f"{self.init_seed}:receiver_readout.{name}.weight".encode(), digest_size=8).digest()
        return int.from_bytes(digest, "little") % (2**31 - 1)

    def prepare_scene(
        self,
        source_state: Tensor,
        anchor_ids: Tensor,
        *,
        frame_key,
        scene_version,
        owner_token,
    ) -> PreparedHyperedgeReadoutValues:
        if source_state.ndim != 3 or source_state.shape[-1] != 128:
            raise ValueError("The receiver readout must project the completed [B,E,128] H state.")
        if source_state.shape[0] != anchor_ids.shape[0] or source_state.shape[1] != anchor_ids.shape[1]:
            raise ValueError("The completed H state and anchor IDs must have matching [B,E] axes.")
        projected = self.anchor_projection(source_state)
        return PreparedHyperedgeReadoutValues(
            values=projected,
            anchor_ids=anchor_ids,
            source_state=source_state,
            frame_key=frame_key,
            scene_version=scene_version,
            owner_token=owner_token,
        )

    def _relative_coordinates(
        self,
        query_coordinates: Tensor,
        selected_anchor_coordinates: Tensor,
        frame_scale: Tensor,
    ) -> Tensor:
        relative = query_coordinates[:, :, None, :] - selected_anchor_coordinates
        if self.geometry_mode == "scene_frame_normalized":
            if frame_scale.shape != (query_coordinates.shape[0], self.coordinate_dim):
                raise ValueError("Frame-normalized geometry requires explicit scene scale [B,d].")
            relative = relative / frame_scale[:, None, None, :]
        return relative

    @staticmethod
    def _slice_cover(cover: ReceiverCover, start: int, stop: int) -> ReceiverCover:
        return ReceiverCover(
            anchor_indices=cover.anchor_indices[:, start:stop],
            anchor_ids=cover.anchor_ids[:, start:stop],
            weights=cover.weights[:, start:stop],
            validity=cover.validity[:, start:stop],
            query_ids=cover.query_ids[:, start:stop],
            frame_key=cover.frame_key,
            query_mask=None if cover.query_mask is None else cover.query_mask[:, start:stop],
            topology=cover.topology,
            scene_version=cover.scene_version,
            owner_token=cover.owner_token,
        )

    def local_correction(
        self,
        *,
        prepared: PreparedHyperedgeReadoutValues,
        query_states: Tensor,
        query_coordinates: Tensor,
        anchor_coordinates: Tensor,
        anchor_mask: Tensor | None = None,
        frame_scale: Tensor,
        cover: ReceiverCover,
        query_mask: Tensor | None,
        collect_exact_work_counts: bool,
        return_trace: bool = False,
        activation: Callable[[Tensor], Tensor] = F.silu,
        expected_frame_key=None,
        expected_scene_version=None,
        expected_owner_token=None,
    ) -> tuple[Tensor, dict[str, Tensor] | None, dict[str, int]]:
        """Apply one of the matched local-cover corrections in bounded tiles."""
        if self.access != "local_cover":
            raise ValueError("local_correction is valid only for local_cover access.")
        if not isinstance(prepared, PreparedHyperedgeReadoutValues):
            raise TypeError("prepared must be owner-bound PreparedHyperedgeReadoutValues.")
        if not isinstance(cover, ReceiverCover):
            raise TypeError("cover must be a native ReceiverCover.")
        if query_states.ndim != 3:
            raise ValueError("query_states must have shape [B,Q,128].")
        batch, queries, state_dim = query_states.shape
        if state_dim != 128 or query_coordinates.shape != (batch, queries, self.coordinate_dim):
            raise ValueError("Query state/coordinate axes do not match the receiver schema.")
        if anchor_coordinates.shape[:2] != prepared.values.shape[:2] or anchor_coordinates.shape[-1] != self.coordinate_dim:
            raise ValueError("Anchor coordinates do not match the prepared hyperedge bank.")
        if anchor_mask is None and cover.topology is not None:
            anchor_mask = cover.topology.environment_mask
        if anchor_mask is not None and (
            anchor_mask.shape != prepared.values.shape[:2] or anchor_mask.dtype != torch.bool
        ):
            raise ValueError("anchor_mask must be boolean [B,E] and match the prepared bank.")
        if (cover.batch_size, cover.query_count) != (batch, queries):
            raise ValueError("Receiver cover axes must match query states.")
        if query_mask is not None and query_mask.shape != (batch, queries):
            raise ValueError("query_mask must match [B,Q].")
        prepared.validate_binding(
            frame_key=expected_frame_key,
            scene_version=expected_scene_version,
            owner_token=expected_owner_token,
        )
        if cover.frame_key != prepared.frame_key:
            raise ValueError("Receiver cover and prepared hyperedge state use different frames.")
        if any(tensor.device != prepared.values.device for tensor in (
            query_states, query_coordinates, anchor_coordinates, frame_scale,
            cover.anchor_indices, cover.anchor_ids, cover.query_ids, cover.weights, cover.validity,
            *(() if anchor_mask is None else (anchor_mask,)),
        )):
            raise ValueError("Prepared values, coordinates, query state and cover must share one device.")
        cover.validate_unchanged()
        if cover.topology is not None:
            cover.validate_topology(
                frame_key=prepared.frame_key,
                scene_version=prepared.scene_version,
                anchor_coordinates=anchor_coordinates,
                environment_ids=prepared.anchor_ids,
            )
        else:
            cover.validate_topology(frame_key=prepared.frame_key)
        validate_cover_anchor_identity(cover, prepared.anchor_ids)
        if anchor_mask is not None:
            selected_anchor_validity = torch.gather(
                anchor_mask[:, None, :].expand(batch, queries, -1),
                2,
                cover.anchor_indices.to(torch.long),
            )
            invalid_masked_anchor = ((~selected_anchor_validity) & cover.validity).any()
            if invalid_masked_anchor.device.type == "cuda":
                torch._assert_async(
                    ~invalid_masked_anchor,
                    "A receiver cover cannot mark an invalid environment anchor as valid.",
                )
            elif bool(invalid_masked_anchor):
                raise ValueError("A receiver cover cannot mark an invalid environment anchor as valid.")
        if query_mask is not None and cover.query_mask is not None and not torch.equal(cover.query_mask, query_mask):
            raise ValueError("Query mask and receiver-cover mask must match exactly.")
        if query_mask is not None and cover.query_mask is None:
            raise ValueError("A masked query batch requires a cover carrying the same query mask.")
        if query_mask is None and cover.query_mask is not None:
            query_mask = cover.query_mask
        effective_query_mask = (
            torch.ones((batch, queries), dtype=torch.bool, device=query_states.device)
            if query_mask is None
            else query_mask
        )
        if effective_query_mask.dtype != torch.bool:
            raise TypeError("query_mask must be boolean [B,Q].")
        if not cover.weights.is_floating_point():
            raise TypeError("Receiver-cover weights must use a floating dtype.")
        invalid_weights = (~torch.isfinite(cover.weights) | (cover.weights < 0)).any()
        if invalid_weights.device.type == "cuda":
            torch._assert_async(~invalid_weights, "Receiver-cover weights must be finite and nonnegative.")
        elif bool(invalid_weights):
            raise ValueError("Receiver-cover weights must be finite and nonnegative.")
        effective_cover_mass = (cover.weights * cover.validity.to(cover.weights.dtype)).sum(dim=-1)
        invalid_padding = (effective_cover_mass[~effective_query_mask] != 0).any()
        if invalid_padding.device.type == "cuda":
            torch._assert_async(~invalid_padding, "Padded queries must have zero receiver-cover mass.")
        elif bool(invalid_padding):
            raise ValueError("Padded queries must have zero receiver-cover mass.")
        if not query_states.is_floating_point() or query_states.dtype != prepared.values.dtype:
            raise ValueError("Query and prepared hyperedge states must use the same floating dtype.")
        if any(not tensor.is_floating_point() for tensor in (query_coordinates, anchor_coordinates)):
            raise TypeError("Receiver coordinates must use floating point dtypes.")
        if query_coordinates.dtype != anchor_coordinates.dtype:
            raise ValueError("Query and anchor coordinates must use the same dtype.")
        if query_coordinates.dtype != query_states.dtype:
            raise ValueError("Receiver coordinates and hidden states must use the same dtype.")
        if self.geometry_mode == "scene_frame_normalized":
            if frame_scale.shape != (batch, self.coordinate_dim):
                raise ValueError("Frame-normalized geometry requires explicit scene scale [B,d].")
            if frame_scale.dtype != query_coordinates.dtype:
                raise ValueError("Frame scale and receiver coordinates must use the same dtype.")
            invalid_scale = (~torch.isfinite(frame_scale) | (frame_scale <= 0)).any()
            if invalid_scale.device.type == "cuda":
                torch._assert_async(~invalid_scale, "Receiver frame scale must be finite and positive.")
            elif bool(invalid_scale):
                raise ValueError("Receiver frame scale must be finite and positive.")

        # The query projection is computed once for each receiver. The scene
        # projection was computed once when the model prepared this scene.
        projected_query = self.query_projection(query_states)
        corrections: list[Tensor] = []
        trace_parts: list[dict[str, Tensor]] = []
        max_live_elements = 0
        checkpointed_query_tiles = 0
        positive_weight_total = (
            torch.zeros((), dtype=torch.int64, device=query_states.device)
            if collect_exact_work_counts
            else None
        )
        retain_detailed_trace = batch * queries * cover.slots * READOUT_WIDTH <= READOUT_TRACE_MAX_ELEMENTS
        tracks_autograd = torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (
                query_states,
                query_coordinates,
                anchor_coordinates,
                prepared.values,
                self.geometry_projection.weight,
                self.output_projection.weight,
            )
        )
        for start in range(0, queries, READOUT_QUERY_TILE):
            stop = min(start + READOUT_QUERY_TILE, queries)
            tile_cover = self._slice_cover(cover, start, stop)
            qterm = projected_query[:, start:stop] + self.hidden_bias
            live_weights = effective_query_mask[:, start:stop].to(query_states.dtype)

            def compute_tile(
                qtile, qcoords, zprojection, anchors, scale, indices, raw_weights, validity,
                live_mask, geometry_weight, output_weight,
            ):
                batch_indices = torch.arange(batch, device=indices.device)[:, None, None]
                selected_projection = zprojection[batch_indices, indices.to(torch.long)]
                selected_coordinates = anchors[batch_indices, indices.to(torch.long)]
                relative_coordinates = qcoords[:, :, None, :] - selected_coordinates
                if self.geometry_mode == "scene_frame_normalized":
                    relative_coordinates = relative_coordinates / scale[:, None, None, :]
                geometry_features = signed_geometry_features(relative_coordinates)
                edge_terms = selected_projection + F.linear(geometry_features, geometry_weight)
                effective_weights = raw_weights.to(edge_terms.dtype) * validity.to(edge_terms.dtype)
                mass = effective_weights.sum(dim=-1, keepdim=True)
                if self.reduction == "pool_first":
                    preactivation = mass * qtile + (effective_weights[..., None] * edge_terms).sum(dim=-2)
                    hidden = activation(preactivation)
                else:
                    messages = activation(qtile[:, :, None, :] + edge_terms)
                    hidden = (effective_weights[..., None] * messages).sum(dim=-2)
                correction_value = F.linear(hidden, output_weight)
                correction_value = correction_value * live_mask[..., None].to(correction_value.dtype)
                return correction_value

            indices = tile_cover.anchor_indices
            weights = tile_cover.weights
            validity = tile_cover.validity
            can_checkpoint = tracks_autograd and (
                not return_trace or not retain_detailed_trace
            )
            if can_checkpoint:
                checkpointed_query_tiles += 1
                correction = checkpoint(
                    compute_tile,
                    qterm,
                    query_coordinates[:, start:stop],
                    prepared.values,
                    anchor_coordinates,
                    frame_scale,
                    indices,
                    weights,
                    validity,
                    live_weights,
                    self.geometry_projection.weight,
                    self.output_projection.weight,
                    use_reentrant=False,
                )
                relative = geometry = projected_selected = pooled = preactivation = messages = None
            else:
                batch_indices = torch.arange(batch, device=indices.device)[:, None, None]
                projected_selected = prepared.values[batch_indices, indices.to(torch.long)]
                selected_coordinates = anchor_coordinates[batch_indices, indices.to(torch.long)]
                relative = query_coordinates[:, start:stop, None, :] - selected_coordinates
                if self.geometry_mode == "scene_frame_normalized":
                    if frame_scale.shape != (batch, self.coordinate_dim):
                        raise ValueError("Frame-normalized geometry requires explicit scene scale [B,d].")
                    relative = relative / frame_scale[:, None, None, :]
                geometry_features = signed_geometry_features(relative)
                geometry = self.geometry_projection(geometry_features)
                edge_terms = projected_selected + geometry
                effective_weights = weights.to(edge_terms.dtype) * validity.to(edge_terms.dtype)
                mass = effective_weights.sum(dim=-1, keepdim=True)
                if self.reduction == "pool_first":
                    preactivation = mass * qterm + (effective_weights[..., None] * edge_terms).sum(dim=-2)
                    pooled = activation(preactivation)
                    messages = None
                else:
                    messages = activation(qterm[:, :, None, :] + edge_terms)
                    pooled = (effective_weights[..., None] * messages).sum(dim=-2)
                    preactivation = None
                correction = self.output_projection(pooled) * live_weights[..., None]
            corrections.append(correction)
            max_live_elements = max(max_live_elements, batch * (stop - start) * cover.slots * READOUT_WIDTH)
            if collect_exact_work_counts:
                tile_positive = ((tile_cover.weights > 0) & tile_cover.validity).sum()
                assert positive_weight_total is not None
                positive_weight_total = positive_weight_total + tile_positive
            if return_trace and retain_detailed_trace:
                part = {
                    "anchor_indices": tile_cover.anchor_indices,
                    "anchor_ids": tile_cover.anchor_ids,
                    "weights": weights,
                    "signed_relative_coordinates": relative,
                    "geometry_features": signed_geometry_features(relative),
                    "anchor_state_projection": projected_selected,
                    "query_projection": qterm,
                    "weighted_hidden_reduction": pooled,
                    "correction": correction,
                }
                if preactivation is not None:
                    part["pooled_preactivation"] = preactivation
                if messages is not None and messages.numel() <= READOUT_TRACE_MAX_ELEMENTS:
                    part["edge_messages"] = messages
                elif messages is not None:
                    part["edge_messages_retained"] = torch.tensor(False, device=messages.device)
                trace_parts.append(part)

        correction = torch.cat(corrections, dim=1) if corrections else query_states.new_zeros(batch, 0, 128)
        trace = None
        if return_trace:
            # Query tiles are bounded above. Native trace exports are intended
            # for small fixed probes; large message tensors are never rebuilt.
            trace = {
                "reduction": self.reduction,
                "access": self.access,
                "parts": tuple(trace_parts),
                "detailed_tensors_retained": retain_detailed_trace,
            }
        slots = batch * queries * cover.slots
        nonlinear_entries = batch * queries * READOUT_WIDTH * (cover.slots if self.reduction == "edge_first" else 1)
        geometry_entries = batch * queries * cover.slots * 7 * self.coordinate_dim
        work = {
            "receiver_readout_query_projection_rows": batch * queries,
            "receiver_readout_selected_slots": slots,
            "receiver_readout_geometry_feature_entries": geometry_entries,
            "receiver_readout_nonlinear_64_entries": nonlinear_entries,
            "receiver_readout_max_live_tile_bytes_per_call": max_live_elements * prepared.values.element_size(),
            "receiver_readout_checkpointed_query_tiles": checkpointed_query_tiles,
            "receiver_readout_nonlinear_activation_bytes_for_autograd": (
                2 * nonlinear_entries * prepared.values.element_size()
                if tracks_autograd and checkpointed_query_tiles < (
                    queries + READOUT_QUERY_TILE - 1
                ) // READOUT_QUERY_TILE
                else 0
            ),
            "receiver_readout_trace_message_bytes": (
                batch * queries * cover.slots * READOUT_WIDTH * prepared.values.element_size()
                if return_trace and retain_detailed_trace and self.reduction == "edge_first"
                else 0
            ),
            "receiver_readout_exact_positive_weight_count_available": int(collect_exact_work_counts),
        }
        if collect_exact_work_counts:
            assert positive_weight_total is not None
            work["receiver_readout_positive_weight_count"] = int(positive_weight_total.detach().cpu())
        return correction, trace, work

    def full_bank_correction(
        self,
        *,
        prepared: PreparedHyperedgeReadoutValues,
        query_states: Tensor,
        query_coordinates: Tensor,
        anchor_coordinates: Tensor,
        anchor_mask: Tensor,
        anchor_measure: Tensor,
        frame_scale: Tensor,
        query_mask: Tensor | None,
        collect_exact_work_counts: bool,
        return_trace: bool = False,
        activation: Callable[[Tensor], Tensor] = F.silu,
        expected_frame_key=None,
        expected_scene_version=None,
        expected_owner_token=None,
    ) -> tuple[Tensor, dict[str, Tensor] | None, dict[str, int]]:
        """Apply the fixed-measure full-bank access diagnostic in bounded tiles."""
        if self.access != "full_bank_diagnostic" or self.reduction != "edge_first":
            raise ValueError("full_bank_correction requires edge_first full_bank_diagnostic access.")
        if not isinstance(prepared, PreparedHyperedgeReadoutValues):
            raise TypeError("prepared must be owner-bound PreparedHyperedgeReadoutValues.")
        if query_states.ndim != 3:
            raise ValueError("query_states must have shape [B,Q,128].")
        batch, queries, state_dim = query_states.shape
        if state_dim != 128 or query_coordinates.shape != (batch, queries, self.coordinate_dim):
            raise ValueError("Query state/coordinate axes do not match the receiver schema.")
        if anchor_coordinates.shape[:2] != prepared.values.shape[:2] or anchor_coordinates.shape[-1] != self.coordinate_dim:
            raise ValueError("Anchor coordinates do not match the prepared hyperedge bank.")
        if anchor_mask.shape != prepared.values.shape[:2] or anchor_mask.dtype != torch.bool:
            raise ValueError("anchor_mask must be boolean [B,E].")
        if anchor_measure.shape != prepared.values.shape[:2]:
            raise ValueError("anchor_measure must match [B,E].")
        if query_mask is not None and (
            query_mask.shape != (batch, queries) or query_mask.dtype != torch.bool
        ):
            raise ValueError("query_mask must be boolean [B,Q].")
        prepared.validate_binding(
            frame_key=expected_frame_key,
            scene_version=expected_scene_version,
            owner_token=expected_owner_token,
        )
        if any(tensor.device != prepared.values.device for tensor in (
            query_states, query_coordinates, anchor_coordinates, anchor_mask, anchor_measure, frame_scale,
        )):
            raise ValueError("Prepared values, coordinates and measures must share one device.")
        if not query_states.is_floating_point() or query_states.dtype != prepared.values.dtype:
            raise ValueError("Query and prepared hyperedge states must use the same floating dtype.")
        if any(not tensor.is_floating_point() for tensor in (query_coordinates, anchor_coordinates, anchor_measure)):
            raise TypeError("Receiver coordinates and native measures must use floating point dtypes.")
        if query_coordinates.dtype != anchor_coordinates.dtype or query_coordinates.dtype != query_states.dtype:
            raise ValueError("Receiver coordinates and hidden states must use the same dtype.")
        if self.geometry_mode == "scene_frame_normalized":
            if frame_scale.shape != (batch, self.coordinate_dim):
                raise ValueError("Frame-normalized geometry requires explicit scene scale [B,d].")
            if frame_scale.dtype != query_coordinates.dtype:
                raise ValueError("Frame scale and receiver coordinates must use the same dtype.")
            invalid_scale = (~torch.isfinite(frame_scale) | (frame_scale <= 0)).any()
            if invalid_scale.device.type == "cuda":
                torch._assert_async(~invalid_scale, "Receiver frame scale must be finite and positive.")
            elif bool(invalid_scale):
                raise ValueError("Receiver frame scale must be finite and positive.")
        invalid_measure = (~torch.isfinite(anchor_measure) | (anchor_measure < 0) |
                           (anchor_mask & (anchor_measure <= 0))).any()
        if invalid_measure.device.type == "cuda":
            torch._assert_async(~invalid_measure, "Valid environment anchors require finite positive measure.")
        elif bool(invalid_measure):
            raise ValueError("Valid environment anchors require finite positive measure.")
        measure = anchor_measure.to(prepared.values.dtype) * anchor_mask.to(prepared.values.dtype)
        total = measure.sum(dim=-1, keepdim=True)
        nu = torch.where(total > 0, measure / total.clamp_min(torch.finfo(measure.dtype).tiny), torch.zeros_like(measure))
        projected_query = self.query_projection(query_states) + self.hidden_bias
        correction_tiles: list[Tensor] = []
        trace_parts: list[dict[str, Tensor]] = []
        max_live_elements = (
            batch * min(queries, READOUT_FULL_BANK_QUERY_TILE)
            * min(prepared.values.shape[1], READOUT_ANCHOR_TILE) * READOUT_WIDTH
        )
        candidate_slots = batch * queries * prepared.values.shape[1]
        retain_detailed_trace = (
            batch * queries * prepared.values.shape[1] * READOUT_WIDTH <= READOUT_TRACE_MAX_ELEMENTS
        )
        checkpointed_query_tiles = 0
        tracks_autograd = torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (
                query_states,
                query_coordinates,
                anchor_coordinates,
                prepared.values,
                self.geometry_projection.weight,
                self.output_projection.weight,
            )
        )
        for qstart in range(0, queries, READOUT_FULL_BANK_QUERY_TILE):
            qstop = min(qstart + READOUT_FULL_BANK_QUERY_TILE, queries)
            qcoords = query_coordinates[:, qstart:qstop]
            qterm = projected_query[:, qstart:qstop]
            qcoordinates = query_coordinates[:, qstart:qstop]

            def compute_full_bank_tile(
                qtile, qcoords, zprojection, anchors, scale, fixed_measure, valid_anchors,
                live_mask, geometry_weight, output_weight,
            ):
                tile_reduction = qtile.new_zeros(batch, qtile.shape[1], READOUT_WIDTH)
                for anchor_start in range(0, zprojection.shape[1], READOUT_ANCHOR_TILE):
                    anchor_stop = min(anchor_start + READOUT_ANCHOR_TILE, zprojection.shape[1])
                    relative_coordinates = qcoords[:, :, None, :] - anchors[:, None, anchor_start:anchor_stop, :]
                    if self.geometry_mode == "scene_frame_normalized":
                        relative_coordinates = relative_coordinates / scale[:, None, None, :]
                    geometry_features = signed_geometry_features(relative_coordinates)
                    geometry = F.linear(geometry_features, geometry_weight)
                    edge_input = qtile[:, :, None, :] + zprojection[:, None, anchor_start:anchor_stop, :] + geometry
                    messages = activation(edge_input)
                    weights = fixed_measure[:, None, anchor_start:anchor_stop] * valid_anchors[
                        :, None, anchor_start:anchor_stop
                    ].to(fixed_measure.dtype)
                    tile_reduction = tile_reduction + (messages * weights[..., None]).sum(dim=-2)
                tile_context = F.linear(tile_reduction, output_weight)
                return tile_context * live_mask[..., None].to(tile_context.dtype)

            can_checkpoint = tracks_autograd and (
                not return_trace or not retain_detailed_trace
            )
            live_weights = (torch.ones(batch, qstop - qstart, device=query_states.device, dtype=query_states.dtype)
                            if query_mask is None else query_mask[:, qstart:qstop].to(query_states.dtype))
            if can_checkpoint:
                checkpointed_query_tiles += 1
                correction = checkpoint(
                    compute_full_bank_tile,
                    qterm,
                    qcoordinates,
                    prepared.values,
                    anchor_coordinates,
                    frame_scale,
                    nu,
                    anchor_mask,
                    live_weights,
                    self.geometry_projection.weight,
                    self.output_projection.weight,
                    use_reentrant=False,
                )
                reduction = None
            else:
                reduction = query_states.new_zeros(batch, qstop - qstart, READOUT_WIDTH)
            message_parts: list[Tensor] = []
            geometry_parts: list[Tensor] = []
            relative_parts: list[Tensor] = []
            if not can_checkpoint:
                for estart in range(0, prepared.values.shape[1], READOUT_ANCHOR_TILE):
                    estop = min(estart + READOUT_ANCHOR_TILE, prepared.values.shape[1])
                    rel = qcoords[:, :, None, :] - anchor_coordinates[:, None, estart:estop, :]
                    if self.geometry_mode == "scene_frame_normalized":
                        if frame_scale.shape != (batch, self.coordinate_dim):
                            raise ValueError("Frame-normalized geometry requires explicit scene scale [B,d].")
                        rel = rel / frame_scale[:, None, None, :]
                    geometry_features = signed_geometry_features(rel)
                    geometry = self.geometry_projection(geometry_features)
                    edge_input = qterm[:, :, None, :] + prepared.values[:, None, estart:estop, :] + geometry
                    messages = activation(edge_input)
                    weights = nu[:, None, estart:estop] * anchor_mask[:, None, estart:estop].to(nu.dtype)
                    reduction = reduction + (messages * weights[..., None]).sum(dim=-2)
                    if return_trace and retain_detailed_trace:
                        message_parts.append(messages)
                        geometry_parts.append(geometry_features)
                        relative_parts.append(rel)
            if not can_checkpoint:
                correction = self.output_projection(reduction) * live_weights[..., None]
            correction_tiles.append(correction)
            if return_trace:
                part = {"correction": correction, "detailed_tensors_retained": retain_detailed_trace}
                if reduction is not None:
                    part["weighted_hidden_reduction"] = reduction
                if message_parts:
                    part["edge_messages"] = torch.cat(message_parts, dim=2)
                    part["geometry_features"] = torch.cat(geometry_parts, dim=2)
                    part["signed_relative_coordinates"] = torch.cat(relative_parts, dim=2)
                else:
                    part["edge_messages_retained"] = torch.tensor(False, device=correction.device)
                    part["detailed_tensors_retained"] = retain_detailed_trace
                trace_parts.append(part)
        correction = torch.cat(correction_tiles, dim=1) if correction_tiles else query_states.new_zeros(batch, 0, 128)
        trace = ({"reduction": self.reduction, "access": self.access, "nu": nu, "parts": tuple(trace_parts)}
                 if return_trace else None)
        work = {
            "receiver_readout_query_projection_rows": batch * queries,
            "receiver_readout_selected_slots": candidate_slots,
            "receiver_readout_geometry_feature_entries": batch * queries * prepared.values.shape[1] * 7 * self.coordinate_dim,
            "receiver_readout_nonlinear_64_entries": batch * queries * prepared.values.shape[1] * READOUT_WIDTH,
            "receiver_readout_max_live_tile_bytes_per_call": max_live_elements * prepared.values.element_size(),
            "receiver_readout_checkpointed_query_tiles": checkpointed_query_tiles,
            "receiver_readout_nonlinear_activation_bytes_for_autograd": (
                2 * batch * queries * prepared.values.shape[1] * READOUT_WIDTH * prepared.values.element_size()
                if tracks_autograd and checkpointed_query_tiles < (
                    queries + READOUT_FULL_BANK_QUERY_TILE - 1
                ) // READOUT_FULL_BANK_QUERY_TILE
                else 0
            ),
            "receiver_readout_trace_message_bytes": (
                batch * queries * prepared.values.shape[1] * READOUT_WIDTH * prepared.values.element_size()
                if return_trace and retain_detailed_trace
                else 0
            ),
            "receiver_readout_exact_valid_anchor_count_available": int(collect_exact_work_counts),
            "receiver_readout_full_bank_fixed_measure": 1,
        }
        if collect_exact_work_counts:
            work["receiver_readout_valid_anchor_count"] = int(anchor_mask.sum().detach().cpu())
        return correction, trace, work

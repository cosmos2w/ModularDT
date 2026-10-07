"""Receiver-bound source packets for affine and nonlinear interaction fields.

Packets are advisory by default. Their bindings name the exact scene, receiver
catalogue, physical sources, units, output/control capabilities, and complete
context ancestry. The module deliberately contains no dataset physics or
full-kernel proposal path.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

import torch
from torch import nn

PACKET_MODES = ("full", "packet_advisory", "packet_experimental")
DEFAULT_PACKET_MODE = "full"


def _ids(values: Sequence[str], name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if (not allow_empty and not result) or any(not value for value in result):
        raise ValueError(f"{name} must contain nonempty physical IDs")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate IDs")
    return result


def tensor_fingerprint(values: Sequence[torch.Tensor]) -> str:
    """Hash exact tensor values and metadata for a small scene/context binding."""
    digest = hashlib.sha256()
    for index, value in enumerate(values):
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
            raise ValueError("Scene binding tensors must be finite torch tensors")
        canonical = value.detach().contiguous().cpu()
        digest.update(str(index).encode())
        digest.update(str(canonical.dtype).encode())
        digest.update(json.dumps(list(canonical.shape), separators=(",", ":")).encode())
        digest.update(canonical.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def prepared_context_fingerprints(prepared_context) -> tuple[str, str]:
    """Return cached fingerprints bound to prepared tensor identities and versions."""
    prepared_context.assert_fresh()
    geometry_values = (
        prepared_context.centers,
        prepared_context.present,
        prepared_context.lengths,
        prepared_context.source_lengths,
    )
    context_values = (
        prepared_context.source_states,
        prepared_context.environment_states,
        prepared_context.global_state,
        prepared_context.environment_coords,
        prepared_context.environment_present,
        prepared_context.source_measure,
        prepared_context.environment_measure,
    )
    values = geometry_values + context_values
    if not all(torch.is_tensor(value) for value in values):
        raise ValueError("Prepared context must expose all source and environment binding tensors")
    signature = tuple(
        (id(value), value.data_ptr(), int(value._version), tuple(value.shape), str(value.dtype), str(value.device))
        for value in values
    )
    cached = getattr(prepared_context, "_receiver_packet_fingerprint_cache", None)
    if cached is not None and cached[0] == signature:
        return cached[1], cached[2]
    geometry_fingerprint = tensor_fingerprint(geometry_values)
    context_fingerprint = tensor_fingerprint(context_values)
    prepared_context._receiver_packet_fingerprint_cache = (signature, geometry_fingerprint, context_fingerprint)
    return geometry_fingerprint, context_fingerprint


@dataclass(frozen=True)
class InteractionBinding:
    """Identity and capability contract for one immutable prepared scene."""

    dataset_id: str
    scene_id: str
    geometry_fingerprint: str
    context_fingerprint: str
    source_ids: tuple[str, ...]
    source_slots: tuple[int, ...]
    prepared_source_ids: tuple[str, ...]
    receiver_ids: tuple[str, ...]
    receiver_coordinates: tuple[tuple[float, ...], ...]
    environment_context_ids: tuple[str, ...]
    output_roles: tuple[str, ...]
    control_roles: tuple[str, ...]
    units: tuple[tuple[str, str], ...]
    capability: str
    ancestry: tuple[str, ...]
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("dataset_id", "scene_id", "geometry_fingerprint", "context_fingerprint", "capability"):
            if not getattr(self, name):
                raise ValueError(f"Binding {name} must be explicit")
        object.__setattr__(self, "source_ids", _ids(self.source_ids, "source_ids"))
        source_slots = tuple(int(value) for value in self.source_slots)
        if (
            len(source_slots) != len(self.source_ids)
            or any(value < 0 for value in source_slots)
            or len(set(source_slots)) != len(source_slots)
        ):
            raise ValueError("Binding source slots must be unique nonnegative indices aligned with physical IDs")
        object.__setattr__(self, "source_slots", source_slots)
        prepared_source_ids = _ids(self.prepared_source_ids, "prepared_source_ids")
        if len(prepared_source_ids) != len(self.source_ids):
            raise ValueError("Prepared native source IDs must align with the physical source catalog")
        object.__setattr__(self, "prepared_source_ids", prepared_source_ids)
        object.__setattr__(self, "receiver_ids", _ids(self.receiver_ids, "receiver_ids"))
        receiver_coordinates = tuple(tuple(float(value) for value in row) for row in self.receiver_coordinates)
        if (
            len(receiver_coordinates) != len(self.receiver_ids)
            or not receiver_coordinates
            or not receiver_coordinates[0]
            or any(len(row) != len(receiver_coordinates[0]) for row in receiver_coordinates)
            or not all(math.isfinite(value) for row in receiver_coordinates for value in row)
        ):
            raise ValueError("Receiver coordinates must be finite, dimension-consistent, and aligned with IDs")
        object.__setattr__(self, "receiver_coordinates", receiver_coordinates)
        object.__setattr__(
            self,
            "environment_context_ids",
            _ids(self.environment_context_ids, "environment_context_ids", allow_empty=True),
        )
        object.__setattr__(self, "output_roles", _ids(self.output_roles, "output_roles"))
        object.__setattr__(self, "control_roles", _ids(self.control_roles, "control_roles", allow_empty=True))
        object.__setattr__(self, "ancestry", _ids(self.ancestry, "ancestry", allow_empty=True))
        units = tuple(sorted((str(key), str(value)) for key, value in self.units))
        if not units or any(not key or not value for key, value in units):
            raise ValueError("Bindings require explicit nonempty unit names and values")
        if len({key for key, _ in units}) != len(units):
            raise ValueError("Binding unit names must be unique")
        object.__setattr__(self, "units", units)
        payload = {
            "dataset_id": self.dataset_id,
            "scene_id": self.scene_id,
            "geometry_fingerprint": self.geometry_fingerprint,
            "context_fingerprint": self.context_fingerprint,
            "source_ids": self.source_ids,
            "source_slots": self.source_slots,
            "prepared_source_ids": self.prepared_source_ids,
            "receiver_ids": self.receiver_ids,
            "receiver_coordinates": self.receiver_coordinates,
            "environment_context_ids": self.environment_context_ids,
            "output_roles": self.output_roles,
            "control_roles": self.control_roles,
            "units": self.units,
            "capability": self.capability,
            "ancestry": self.ancestry,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        object.__setattr__(self, "_fingerprint", hashlib.sha256(encoded).hexdigest())

    @property
    def fingerprint(self) -> str:
        return self._fingerprint


def validate_bound_receiver_coordinates(
    binding: InteractionBinding,
    receivers: torch.Tensor,
    receiver_ids: Sequence[str],
) -> float:
    """Validate a whole or subset read against immutable ID-to-coordinate bindings."""
    started = perf_counter()
    requested = _ids(receiver_ids, "requested receiver_ids")
    if receivers.ndim != 3 or receivers.shape[1] != len(requested):
        raise ValueError("Receiver coordinates and requested physical IDs must have aligned [B,Q,D] shapes")
    if receivers.shape[2] != len(binding.receiver_coordinates[0]):
        raise ValueError("Receiver coordinate dimension differs from the bound receiver catalogue")
    catalog = torch.as_tensor(binding.receiver_coordinates, device=receivers.device, dtype=receivers.dtype)
    if requested == binding.receiver_ids:
        expected_catalog = catalog
    else:
        positions = {receiver_id: index for index, receiver_id in enumerate(binding.receiver_ids)}
        expected_indices = [positions.get(receiver_id, -1) for receiver_id in requested]
        if any(index < 0 for index in expected_indices):
            unknown = [requested[index] for index, position in enumerate(expected_indices) if position < 0]
            raise ValueError(f"Requested receiver IDs are outside the bound coordinate catalogue: {unknown[:3]}")
        expected_index_tensor = torch.tensor(expected_indices, device=receivers.device, dtype=torch.long)
        expected_catalog = catalog.index_select(0, expected_index_tensor)
    expected = expected_catalog.unsqueeze(0).expand(receivers.shape[0], -1, -1)
    if not torch.isfinite(receivers).all() or not torch.equal(receivers.detach(), expected):
        raise ValueError("Requested receiver coordinates differ from the bound physical ID-coordinate catalogue")
    return perf_counter() - started


@dataclass(frozen=True)
class ActionDomain:
    """A TRAIN-qualified componentwise action box in physical units."""

    control_role: str
    unit: str
    source_ids: tuple[str, ...]
    radii: tuple[float, ...]
    require_balanced: bool = False

    def __post_init__(self) -> None:
        if not self.control_role or not self.unit:
            raise ValueError("Action domains require an explicit control role and unit")
        source_ids = _ids(self.source_ids, "action source_ids")
        radii = tuple(float(value) for value in self.radii)
        if len(radii) != len(source_ids) or any(not math.isfinite(value) or value < 0 for value in radii):
            raise ValueError("Action radii must be finite, nonnegative, and source aligned")
        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(self, "radii", radii)

    @property
    def fingerprint(self) -> str:
        payload = (self.control_role, self.unit, self.source_ids, self.radii, self.require_balanced)
        return hashlib.sha256(repr(payload).encode()).hexdigest()

    def radii_tensor(self, *, device=None, dtype=torch.float32) -> torch.Tensor:
        return torch.tensor(self.radii, device=device, dtype=dtype)

    def accepts(self, increment: torch.Tensor, source_ids: Sequence[str], *, tolerance: float = 0.0) -> bool:
        ids = tuple(str(value) for value in source_ids)
        if ids != self.source_ids or increment.shape[-1] != len(ids) or not bool(torch.isfinite(increment).all()):
            return False
        radii = self.radii_tensor(device=increment.device, dtype=increment.dtype)
        if bool((increment.abs() > radii + tolerance).any()):
            return False
        if self.require_balanced:
            epsilon = max(tolerance, torch.finfo(increment.dtype).eps * 64)
            if bool((increment.sum(-1).abs() > epsilon * increment.abs().sum(-1).clamp_min(1)).any()):
                return False
        return True


@dataclass(frozen=True)
class ReceiverPacket:
    """One directed receiver region to physical-source permission list."""

    packet_id: str
    binding_fingerprint: str
    action_domain_fingerprint: str
    receiver_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    environment_ancestry_ids: tuple[str, ...]
    ancestry_ids: tuple[str, ...]
    output_roles: tuple[str, ...]
    control_roles: tuple[str, ...]
    units: tuple[tuple[str, str], ...]
    proposal_status: str = "empirical_packet_proposal"
    residual_margin: float | None = None

    def __post_init__(self) -> None:
        if not self.packet_id or not self.binding_fingerprint or not self.action_domain_fingerprint:
            raise ValueError("Packets require stable identity and scene/action bindings")
        for name in (
            "receiver_ids",
            "source_ids",
            "environment_ancestry_ids",
            "ancestry_ids",
            "output_roles",
            "control_roles",
        ):
            object.__setattr__(
                self,
                name,
                _ids(
                    getattr(self, name),
                    name,
                    allow_empty=name in {"source_ids", "environment_ancestry_ids", "ancestry_ids", "control_roles"},
                ),
            )
        units = tuple(sorted((str(key), str(value)) for key, value in self.units))
        if not units or any(not key or not value for key, value in units):
            raise ValueError("Packets must carry explicit unit bindings")
        object.__setattr__(self, "units", units)
        if self.residual_margin is not None and (not math.isfinite(self.residual_margin) or self.residual_margin < 0):
            raise ValueError("Packet residual margins must be finite and nonnegative")


@dataclass(frozen=True)
class PacketUnion:
    """Per-request receiver/source union after duplicate removal and fallback."""

    receiver_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    binding_fingerprint: str
    action_domain_fingerprint: str
    source_keep: torch.Tensor
    fallback: tuple[bool, ...]
    reasons: tuple[str, ...]
    selected_source_ids: tuple[tuple[str, ...], ...]
    _source_keep_identity: int = field(init=False, repr=False, compare=False)
    _source_keep_data_ptr: int = field(init=False, repr=False, compare=False)
    _source_keep_version: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "receiver_ids", _ids(self.receiver_ids, "packet-union receiver_ids"))
        object.__setattr__(self, "source_ids", _ids(self.source_ids, "packet-union source_ids"))
        if not self.binding_fingerprint or not self.action_domain_fingerprint:
            raise ValueError("Packet unions must bind the exact scene and action domain")
        if self.source_keep.shape != (len(self.receiver_ids), len(self.source_ids)):
            raise ValueError("Packet union mask must be [requested receivers, physical sources]")
        if self.source_keep.dtype != torch.bool:
            raise ValueError("Packet union source mask must be boolean")
        if not (len(self.fallback) == len(self.reasons) == len(self.selected_source_ids) == len(self.receiver_ids)):
            raise ValueError("Packet union receipts must align with requested receivers")
        fallback = tuple(bool(value) for value in self.fallback)
        reasons = tuple(str(value) for value in self.reasons)
        selected = tuple(
            _ids(row, f"packet-union selected_source_ids[{index}]", allow_empty=True)
            for index, row in enumerate(self.selected_source_ids)
        )
        if any(not reason for reason in reasons):
            raise ValueError("Packet union reasons must be explicit for every receiver")
        source_column = {source_id: index for index, source_id in enumerate(self.source_ids)}
        expected = torch.zeros_like(self.source_keep)
        for row, (is_fallback, selected_ids) in enumerate(zip(fallback, selected)):
            if any(source_id not in source_column for source_id in selected_ids):
                raise ValueError("Packet union selected source identity is outside its physical catalog")
            expected_order = tuple(source_id for source_id in self.source_ids if source_id in set(selected_ids))
            if selected_ids != expected_order:
                raise ValueError("Packet union selected sources must preserve the bound physical catalog order")
            if is_fallback and selected_ids != self.source_ids:
                raise ValueError("Full-source fallback rows must select the complete physical source catalog")
            if selected_ids:
                expected[row, [source_column[source_id] for source_id in selected_ids]] = True
        if not torch.equal(expected, self.source_keep):
            raise ValueError("Packet union mask disagrees with its immutable fallback/source-list metadata")
        object.__setattr__(self, "fallback", fallback)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "selected_source_ids", selected)
        object.__setattr__(self, "_source_keep_identity", id(self.source_keep))
        object.__setattr__(self, "_source_keep_data_ptr", self.source_keep.data_ptr())
        object.__setattr__(self, "_source_keep_version", int(self.source_keep._version))


def _validate_packet_union_integrity(packet_union: PacketUnion) -> None:
    if (
        id(packet_union.source_keep) != packet_union._source_keep_identity
        or packet_union.source_keep.data_ptr() != packet_union._source_keep_data_ptr
        or int(packet_union.source_keep._version) != packet_union._source_keep_version
    ):
        raise ValueError("Packet union source mask was modified after immutable source-selection metadata was bound")
    if (
        packet_union.source_keep.shape != (len(packet_union.receiver_ids), len(packet_union.source_ids))
        or packet_union.source_keep.dtype != torch.bool
        or not (
            len(packet_union.fallback)
            == len(packet_union.reasons)
            == len(packet_union.selected_source_ids)
            == len(packet_union.receiver_ids)
        )
    ):
        raise ValueError("Packet union catalog, reasons, and selection metadata no longer align")
    source_column = {source_id: index for index, source_id in enumerate(packet_union.source_ids)}
    if len(source_column) != len(packet_union.source_ids) or any(not reason for reason in packet_union.reasons):
        raise ValueError("Packet union source catalog or fallback reasons are invalid")
    expected = torch.zeros_like(packet_union.source_keep)
    for row, (is_fallback, selected_ids) in enumerate(zip(packet_union.fallback, packet_union.selected_source_ids)):
        if any(source_id not in source_column for source_id in selected_ids):
            raise ValueError("Packet union selection contains a source outside its bound catalog")
        selected_set = set(selected_ids)
        expected_order = tuple(source_id for source_id in packet_union.source_ids if source_id in selected_set)
        if tuple(selected_ids) != expected_order:
            raise ValueError("Packet union selection changed its bound physical source order")
        if is_fallback and tuple(selected_ids) != packet_union.source_ids:
            raise ValueError("Full-source fallback metadata must select every active physical source")
        if selected_ids:
            expected[row, [source_column[source_id] for source_id in selected_ids]] = True
    if not torch.equal(expected, packet_union.source_keep):
        raise ValueError("Packet union mask no longer matches its immutable source-selection metadata")


def _packet_valid(packet: ReceiverPacket, binding: InteractionBinding, action: ActionDomain) -> str | None:
    if packet.proposal_status.startswith("full_fallback") or packet.residual_margin is None:
        return "proposal_uncalibrated_full_fallback"
    if packet.binding_fingerprint != binding.fingerprint:
        return "scene_receiver_source_or_capability_binding_mismatch"
    if packet.action_domain_fingerprint != action.fingerprint:
        return "action_domain_mismatch"
    if packet.environment_ancestry_ids != binding.environment_context_ids:
        return "environment_context_ancestry_mismatch"
    if packet.ancestry_ids != binding.ancestry:
        return "context_ancestry_mismatch"
    if packet.output_roles != binding.output_roles or packet.control_roles != binding.control_roles:
        return "output_or_control_type_mismatch"
    if packet.units != binding.units:
        return "unit_mismatch"
    if not _action_matches_binding(action, binding):
        return "action_control_or_unit_binding_mismatch"
    if any(source not in binding.source_ids for source in packet.source_ids):
        return "unknown_physical_source"
    if any(receiver not in binding.receiver_ids for receiver in packet.receiver_ids):
        return "unknown_receiver"
    return None


def _action_matches_binding(action: ActionDomain, binding: InteractionBinding) -> bool:
    """Check physical control role, units, and full physical source catalogue."""
    if action.control_role not in binding.control_roles or action.source_ids != binding.source_ids:
        return False
    units = dict(binding.units)
    # Bindings may state a generic control unit or name the concrete control role.
    declared_unit = units.get(action.control_role, units.get("control"))
    return declared_unit == action.unit


def _prepared_source_id(value) -> str:
    """Canonicalize a native context identity independently from its physical alias."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return str(int(value))
    return str(value)


def _validate_prepared_source_binding(binding: InteractionBinding, prepared_context) -> None:
    """Map the ordered physical catalogue onto every active native prepared slot."""
    present = getattr(prepared_context, "present", None)
    native_ids = getattr(prepared_context, "source_ids", None)
    if (
        not torch.is_tensor(present)
        or not torch.is_tensor(native_ids)
        or present.ndim != 2
        or native_ids.shape != present.shape
    ):
        raise ValueError("Prepared source catalog must expose aligned [B,M] presence and native source IDs")
    for batch in range(present.shape[0]):
        active_slots = tuple(torch.nonzero(present[batch] > 0.5, as_tuple=False).flatten().tolist())
        if active_slots != binding.source_slots:
            raise ValueError("Binding source slots must exactly cover the prepared active source catalog in order")
        native_ids_at_slots = tuple(_prepared_source_id(native_ids[batch, slot]) for slot in active_slots)
        if native_ids_at_slots != binding.prepared_source_ids:
            raise ValueError("Physical source aliases do not map to the prepared native source IDs at bound slots")


def full_access_union(
    binding: InteractionBinding,
    action: ActionDomain,
    receiver_ids: Sequence[str] | None = None,
    *,
    device=None,
    fallback: bool = True,
    reason: str = "full_access",
) -> PacketUnion:
    """Build the deterministic full-source route for every requested receiver."""
    receivers = _ids(binding.receiver_ids if receiver_ids is None else receiver_ids, "requested receiver_ids")
    keep = torch.ones(len(receivers), len(binding.source_ids), dtype=torch.bool, device=device)
    all_sources = tuple(binding.source_ids)
    return PacketUnion(
        receiver_ids=receivers,
        source_ids=binding.source_ids,
        binding_fingerprint=binding.fingerprint,
        action_domain_fingerprint=action.fingerprint,
        source_keep=keep,
        fallback=(fallback,) * len(receivers),
        reasons=(reason,) * len(receivers),
        selected_source_ids=(all_sources,) * len(receivers),
    )


def union_receiver_packets(
    binding: InteractionBinding,
    action: ActionDomain,
    packets: Sequence[ReceiverPacket],
    receiver_ids: Sequence[str],
    *,
    device=None,
) -> PacketUnion:
    """Union overlapping packet memberships once per receiver; invalid rows fall back fully."""
    requested = _ids(receiver_ids, "requested receiver_ids")
    source_order = tuple(binding.source_ids)
    source_column = {source_id: index for index, source_id in enumerate(source_order)}
    packet_receivers: dict[str, list[ReceiverPacket]] = {}
    # Metadata is immutable for this call. Validate each packet once, rather
    # than rescanning its complete receiver catalogue for every covered row.
    packet_validity: dict[int, str | None] = {}
    for packet in packets:
        if id(packet) not in packet_validity:
            packet_validity[id(packet)] = _packet_valid(packet, binding, action)
        for receiver_id in packet.receiver_ids:
            packet_receivers.setdefault(receiver_id, []).append(packet)
    bound_receivers = set(binding.receiver_ids)
    keep = torch.zeros(len(requested), len(source_order), dtype=torch.bool, device=device)
    fallback: list[bool] = []
    reasons: list[str] = []
    selected: list[tuple[str, ...]] = []
    for row, receiver_id in enumerate(requested):
        candidates = packet_receivers.get(receiver_id, [])
        invalid = None
        for packet in candidates:
            invalid = packet_validity[id(packet)]
            if invalid is not None:
                break
        if receiver_id not in bound_receivers:
            invalid = "receiver_outside_bound_catalogue"
        if not candidates:
            invalid = "no_packet_covers_receiver"
        if invalid is not None:
            keep[row] = True
            fallback.append(True)
            reasons.append(invalid)
            selected.append(source_order)
            continue
        union = {source_id for packet in candidates for source_id in packet.source_ids}
        ordered = tuple(source_id for source_id in source_order if source_id in union)
        for source_id in ordered:
            keep[row, source_column[source_id]] = True
        fallback.append(False)
        reasons.append("packet_union")
        selected.append(ordered)
    return PacketUnion(
        receiver_ids=requested,
        source_ids=source_order,
        binding_fingerprint=binding.fingerprint,
        action_domain_fingerprint=action.fingerprint,
        source_keep=keep,
        fallback=tuple(fallback),
        reasons=tuple(reasons),
        selected_source_ids=tuple(selected),
    )


def execution_source_mask(
    mode: Literal["full", "packet_advisory", "packet_experimental"] = DEFAULT_PACKET_MODE,
    *,
    proposed: PacketUnion | None = None,
    device=None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Resolve the three explicit modes; full access remains the ordinary default."""
    if mode not in PACKET_MODES:
        raise ValueError(f"Unknown receiver-packet mode {mode!r}")
    if proposed is None:
        raise ValueError("Resolve a concrete receiver/source shape before selecting an execution mode")
    proposal_mask = proposed.source_keep.to(device=device)
    if mode == "packet_experimental":
        selected = proposal_mask
    else:
        selected = torch.ones_like(proposal_mask)
    receipt = {
        "mode": mode,
        "packet_proposal_used": True,
        "packet_mask_executed": mode == "packet_experimental",
        "fallback_receivers": int(sum(proposed.fallback)),
        "fallback_reasons": list(proposed.reasons),
        "proposed_active_source_pair_rows": int(proposal_mask.sum().item()),
        "selected_active_source_pair_rows": int(selected.sum().item()),
    }
    return selected, receipt


def response_rms_budget(fraction: float, train_response_rms: float) -> float:
    """Return only the plan's declared one- or two-percent TRAIN response budget."""
    if fraction not in (0.01, 0.02):
        raise ValueError("Receiver cover budget fraction must be exactly 1% or 2%")
    if not math.isfinite(train_response_rms) or train_response_rms <= 0:
        raise ValueError("TRAIN response RMS must be finite and positive")
    return fraction * train_response_rms


def exact_teacher_sensitivities(
    kernel: torch.Tensor,
    patch_weights: torch.Tensor,
    action_radii: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw s_i and radius-weighted q_i from an explicitly supplied exact K."""
    if kernel.ndim != 2 or patch_weights.shape != (kernel.shape[0],) or action_radii.shape != (kernel.shape[1],):
        raise ValueError("Exact K, normalized patch weights, and action radii must align")
    if not all(bool(torch.isfinite(value).all()) for value in (kernel, patch_weights, action_radii)):
        raise ValueError("Teacher sensitivity inputs must be finite")
    if bool((patch_weights < 0).any()) or not torch.isclose(
        patch_weights.sum(), patch_weights.new_tensor(1.0), rtol=0.0, atol=1e-6
    ):
        raise ValueError("Patch weights must be nonnegative and sum to one")
    if bool((action_radii < 0).any()):
        raise ValueError("Action radii must be nonnegative")
    raw = torch.sqrt((kernel.square() * patch_weights[:, None]).sum(0))
    return raw, raw * action_radii


def deterministic_cover(
    importance: torch.Tensor,
    source_ids: Sequence[str],
    budget: float,
    *,
    protected_source_ids: Sequence[str] = (),
) -> tuple[tuple[str, ...], float]:
    """Choose the smallest score-ranked set meeting the omitted triangle budget."""
    ids = _ids(source_ids, "cover source_ids")
    if importance.ndim != 1 or importance.numel() != len(ids) or not bool(torch.isfinite(importance).all()):
        raise ValueError("Cover importance must be a finite source-aligned vector")
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("Cover budget must be finite and nonnegative")
    protected = set(_ids(protected_source_ids, "protected_source_ids", allow_empty=True))
    if not protected.issubset(ids):
        raise ValueError("Protected near-source IDs must belong to this source catalogue")
    score_values = [float(value) for value in importance.detach().cpu().tolist()]
    ordered = sorted(range(len(ids)), key=lambda index: (-score_values[index], ids[index]))
    chosen: set[str] = set(protected)
    for index in ordered:
        omitted = sum(score_values[j] for j in range(len(ids)) if ids[j] not in chosen)
        if omitted <= budget:
            break
        chosen.add(ids[index])
    # Preserve scene source order in the exported packet.
    selected = tuple(source_id for source_id in ids if source_id in chosen)
    omitted_bound = sum(score_values[index] for index, source_id in enumerate(ids) if source_id not in chosen)
    return selected, float(omitted_bound)


def geometry_equal_k_controls(
    source_coordinates: torch.Tensor,
    receiver_center: torch.Tensor,
    source_ids: Sequence[str],
    flow_direction: torch.Tensor,
    k: int,
    *,
    protected_source_ids: Sequence[str] = (),
) -> dict[str, tuple[str, ...]]:
    """Return nearest and nearest-upstream controls with physical-ID tie breaks."""
    ids = _ids(source_ids, "geometry-control source_ids")
    if source_coordinates.ndim != 2 or receiver_center.shape != (source_coordinates.shape[1],):
        raise ValueError("Geometry controls require [M,D] sources and [D] receiver coordinates")
    if flow_direction.shape != receiver_center.shape or not bool(torch.isfinite(flow_direction).all()):
        raise ValueError("Geometry control flow direction must be finite and dimension aligned")
    if source_coordinates.shape[0] != len(ids) or not bool(torch.isfinite(source_coordinates).all()):
        raise ValueError("Geometry source coordinates must align with physical IDs")
    direction_norm = torch.linalg.vector_norm(flow_direction)
    if float(direction_norm) <= 0 or k < 1 or k > len(ids):
        raise ValueError("Geometry controls require nonzero flow direction and 1 <= K <= M")
    protected = set(_ids(protected_source_ids, "protected_source_ids", allow_empty=True))
    if not protected.issubset(ids) or len(protected) > k:
        raise ValueError("Protected near sources must fit within the shared equal-K count")
    relative = source_coordinates - receiver_center[None]
    distance = torch.linalg.vector_norm(relative, dim=-1)
    upstream = (relative * flow_direction[None]).sum(-1) <= 0
    base_order = sorted(range(len(ids)), key=lambda index: (float(distance[index]), ids[index]))
    upstream_order = sorted(
        (index for index in base_order if bool(upstream[index])), key=lambda index: (float(distance[index]), ids[index])
    )

    def select(order: Sequence[int]) -> tuple[str, ...]:
        chosen = set(protected)
        for index in order:
            if len(chosen) >= k:
                break
            chosen.add(ids[index])
        return tuple(source_id for source_id in ids if source_id in chosen)

    return {
        "nearest": select(base_order),
        "upstream": select(upstream_order + [index for index in base_order if index not in set(upstream_order)]),
    }


def build_receiver_pair_features(
    *,
    source_states: torch.Tensor,
    global_state: torch.Tensor,
    environment_states: torch.Tensor | None = None,
    environment_measure: torch.Tensor | None = None,
    source_coordinates: torch.Tensor,
    receiver_coordinates: torch.Tensor,
    domain_lengths: torch.Tensor,
    source_lengths: torch.Tensor,
    receiver_features: torch.Tensor,
    output_type_features: torch.Tensor,
    control_type_features: torch.Tensor,
) -> torch.Tensor:
    """Build cheap context/geometry features without evaluating a fine readout."""
    if source_states.ndim != 3 or global_state.ndim != 2:
        raise ValueError("Source states must be [B,M,H] and global state [B,H]")
    batch, modules, hidden = source_states.shape
    queries = receiver_coordinates.shape[1]
    dimension = receiver_coordinates.shape[-1]
    if source_coordinates.shape != (batch, modules, dimension):
        raise ValueError("Source coordinates must be [B,M,D] aligned to source states")
    if receiver_coordinates.shape != (batch, queries, dimension):
        raise ValueError("Receiver coordinates must be [B,P,D]")
    if global_state.shape != (batch, hidden) or domain_lengths.shape != (batch, dimension):
        raise ValueError("Global state and domain lengths must align with source context")
    if source_lengths.shape != (batch, modules) or bool((source_lengths <= 0).any()):
        raise ValueError("Every physical source requires a positive characteristic length")
    if bool((domain_lengths <= 0).any()):
        raise ValueError("Physical domain lengths must be positive")
    if (environment_states is None) != (environment_measure is None):
        raise ValueError("Prepared environment states and measures must be supplied together")
    if environment_states is None:
        environment_context = torch.zeros_like(global_state)
    else:
        if (
            environment_states.ndim != 3
            or environment_states.shape[0] != batch
            or environment_states.shape[2] != hidden
        ):
            raise ValueError("Prepared environment states must be [B,E,H] with the source hidden width")
        if environment_measure.shape != environment_states.shape[:2] or bool((environment_measure < 0).any()):
            raise ValueError("Prepared environment measures must be nonnegative [B,E]")
        environment_context = (environment_states * environment_measure[..., None]).sum(1)
        environment_context = environment_context / environment_measure.sum(1, keepdim=True).clamp_min(
            torch.finfo(environment_context.dtype).tiny
        )
    if receiver_features.ndim != 3 or receiver_features.shape[:2] != (batch, queries):
        raise ValueError("Receiver-region features must be [B,P,F]")
    for name, features in (("output", output_type_features), ("control", control_type_features)):
        if features.ndim != 3 or features.shape[:2] != (batch, queries):
            raise ValueError(f"{name} type features must be [B,P,F]")
    tensors = (
        source_states,
        global_state,
        source_coordinates,
        receiver_coordinates,
        domain_lengths,
        source_lengths,
        receiver_features,
        output_type_features,
        control_type_features,
    )
    if environment_states is not None:
        tensors += (environment_states, environment_measure, environment_context)
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("Receiver-pair feature inputs must be finite")
    relative = (source_coordinates[:, None] - receiver_coordinates[:, :, None]) / domain_lengths[:, None, None]
    phase = relative * (2.0 * math.pi)
    normalized_distance = torch.linalg.vector_norm(
        (source_coordinates[:, None] - receiver_coordinates[:, :, None]) / source_lengths[:, None, :, None],
        dim=-1,
        keepdim=True,
    )
    normalized_length = (source_lengths[:, :, None] / domain_lengths[:, None]).clamp_min(1e-12).log()
    source_part = source_states[:, None].expand(-1, queries, -1, -1)
    global_part = global_state[:, None, None].expand(-1, queries, modules, -1)
    environment_part = environment_context[:, None, None].expand(-1, queries, modules, -1)

    def repeat_receiver(value: torch.Tensor) -> torch.Tensor:
        return value[:, :, None].expand(-1, -1, modules, -1)

    return torch.cat(
        (
            source_part,
            global_part,
            environment_part,
            relative,
            phase.sin(),
            phase.cos(),
            normalized_distance,
            normalized_length[:, None].expand(-1, queries, -1, -1),
            repeat_receiver(receiver_features),
            repeat_receiver(output_type_features),
            repeat_receiver(control_type_features),
        ),
        dim=-1,
    )


class ReceiverPairScorer(nn.Module):
    """Small shared patch/source scorer trained against frozen response sensitivity."""

    def __init__(self, input_width: int, hidden_width: int = 32, parameter_cap: int = 50_000):
        super().__init__()
        if input_width < 1 or hidden_width < 1 or parameter_cap < 1:
            raise ValueError("Scorer widths and parameter cap must be positive")
        self.network = nn.Sequential(nn.Linear(input_width, hidden_width), nn.SiLU(), nn.Linear(hidden_width, 1))
        self.input_width = int(input_width)
        self.hidden_width = int(hidden_width)
        self.parameter_cap = int(parameter_cap)
        if self.parameter_count > parameter_cap:
            raise ValueError(f"Receiver pair scorer has {self.parameter_count} parameters, above cap {parameter_cap}")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.input_width:
            raise ValueError("Receiver-pair feature width differs from the frozen scorer")
        return self.network(features).squeeze(-1)


def scorer_parameter_sha256(scorer: ReceiverPairScorer) -> str:
    """Hash canonical named scorer tensors so calibration is bound to exact weights."""
    digest = hashlib.sha256()
    for name, value in sorted(scorer.state_dict().items()):
        canonical = value.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(canonical.dtype).encode())
        digest.update(json.dumps(list(canonical.shape), separators=(",", ":")).encode())
        digest.update(canonical.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _sync_tensor_device(value: torch.Tensor) -> None:
    if value.device.type == "cuda":
        torch.cuda.synchronize(value.device)


@dataclass(frozen=True)
class ProposerFitReceipt:
    seed: int
    epochs: int
    training_layout_ids: tuple[str, ...]
    calibration_layout_ids: tuple[str, ...]
    train_only_log_floor: float
    train_only_residual_margin: float
    calibration_max_positive_residual: float
    fit_losses: tuple[float, ...]
    optimizer_updates: int
    parameter_count: int
    parameter_sha256: str
    calibration_scope: str = "explicit TRAIN whole-layout holdout; empirical, not a uniform unseen-layout certificate"


def fit_receiver_pair_scorer(
    scorer: ReceiverPairScorer,
    features: torch.Tensor,
    raw_sensitivities: torch.Tensor,
    layout_ids: Sequence[str],
    *,
    epochs: int = 500,
    seed: int = 0,
    calibration_fraction: float = 0.15,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    review: Callable[[int, float], None] | None = None,
    fit_accounting: dict | None = None,
) -> ProposerFitReceipt:
    """Fit one small scorer and calibrate a TRAIN-only whole-layout residual margin."""
    total_started = perf_counter()
    if epochs < 1 or epochs > 500 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Packet fit allows 1..500 epochs with positive batch size and learning rate")
    if features.ndim != 2 or raw_sensitivities.shape != (features.shape[0],):
        raise ValueError("Packet training features and raw sensitivities must be [N,F] and [N]")
    if len(layout_ids) != features.shape[0] or not bool(torch.isfinite(features).all()):
        raise ValueError("Every finite TRAIN pair needs its original layout identity")
    if not bool(torch.isfinite(raw_sensitivities).all()) or bool((raw_sensitivities < 0).any()):
        raise ValueError("Frozen teacher raw sensitivities must be finite and nonnegative")
    if not 0 < calibration_fraction < 0.5:
        raise ValueError("Whole-layout calibration fraction must lie in (0, 0.5)")
    split_started = perf_counter()
    unique_layouts = sorted({str(value) for value in layout_ids})
    if len(unique_layouts) < 2:
        raise ValueError("Packet training requires separate TRAIN fitting and calibration layouts")
    rank = sorted(unique_layouts, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    calibration_count = max(1, min(len(unique_layouts) - 1, round(len(unique_layouts) * calibration_fraction)))
    calibration_layouts = tuple(sorted(rank[:calibration_count]))
    calibration_set = set(calibration_layouts)
    layout_tensor = tuple(str(value) for value in layout_ids)
    calibration_mask_cpu = torch.tensor([value in calibration_set for value in layout_tensor], dtype=torch.bool)
    fit_mask_cpu = ~calibration_mask_cpu
    if not bool(fit_mask_cpu.any()) or not bool(calibration_mask_cpu.any()):
        raise ValueError("Whole-layout split left no fitting or calibration rows")
    split_seconds = perf_counter() - split_started
    device = next(scorer.parameters()).device
    transfer_started = perf_counter()
    features = features.to(device=device, dtype=next(scorer.parameters()).dtype)
    raw_sensitivities = raw_sensitivities.to(device=device, dtype=features.dtype)
    fit_indices = torch.nonzero(fit_mask_cpu, as_tuple=False).flatten().to(device)
    cal_indices = torch.nonzero(calibration_mask_cpu, as_tuple=False).flatten().to(device)
    _sync_tensor_device(features)
    transfer_seconds = perf_counter() - transfer_started
    positive = raw_sensitivities.index_select(0, fit_indices)
    positive = positive[positive > 0]
    if positive.numel() == 0:
        raise ValueError("TRAIN fitting layouts contain no positive teacher sensitivity")
    floor = max(torch.finfo(features.dtype).tiny, float(positive.median()) * 1e-6)
    target_log = raw_sensitivities.clamp_min(floor).log()
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    updates = 0
    scorer.train()
    training_started = perf_counter()
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(fit_indices.numel(), generator=generator).to(device)
        epoch_loss = 0.0
        pair_count = 0
        for positions in permutation.split(batch_size):
            selected = fit_indices.index_select(0, positions)
            prediction = scorer(features.index_select(0, selected))
            loss = torch.nn.functional.smooth_l1_loss(prediction, target_log.index_select(0, selected))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = int(selected.numel())
            epoch_loss += float(loss.detach()) * count
            pair_count += count
            updates += 1
        mean_loss = epoch_loss / max(pair_count, 1)
        losses.append(mean_loss)
        if review is not None and epoch % 100 == 0:
            review(epoch, mean_loss)
    _sync_tensor_device(features)
    training_seconds = perf_counter() - training_started
    scorer.eval()
    calibration_started = perf_counter()
    with torch.no_grad():
        residual = target_log.index_select(0, cal_indices) - scorer(features.index_select(0, cal_indices))
        positive_residual = residual.clamp_min(0)
        margin = float(positive_residual.max())
    _sync_tensor_device(features)
    calibration_seconds = perf_counter() - calibration_started
    parameter_hash_started = perf_counter()
    parameter_sha256 = scorer_parameter_sha256(scorer)
    parameter_hash_seconds = perf_counter() - parameter_hash_started
    if fit_accounting is not None:
        fit_accounting.update(
            {
                "whole_layout_split_seconds": split_seconds,
                "input_device_transfer_seconds": transfer_seconds,
                "scorer_training_seconds": training_seconds,
                "whole_layout_calibration_seconds": calibration_seconds,
                "calibrated_scorer_parameter_hash_seconds": parameter_hash_seconds,
                "fit_total_seconds": perf_counter() - total_started,
                "optimizer_updates": updates,
                "training_layout_count": len(unique_layouts) - len(calibration_layouts),
                "calibration_layout_count": len(calibration_layouts),
            }
        )
    return ProposerFitReceipt(
        seed,
        epochs,
        tuple(value for value in unique_layouts if value not in calibration_set),
        calibration_layouts,
        floor,
        margin,
        margin,
        tuple(losses),
        updates,
        scorer.parameter_count,
        parameter_sha256,
    )


@torch.no_grad()
def propose_receiver_packets(
    scorer: ReceiverPairScorer,
    fit_receipt: ProposerFitReceipt | None,
    pair_features: torch.Tensor,
    binding: InteractionBinding,
    action_domain: ActionDomain,
    patch_receiver_ids: Sequence[Sequence[str]],
    *,
    budget_fraction: float,
    train_response_rms: float,
    protected_near_source_ids: Sequence[Sequence[str]] | None = None,
    proposal_accounting: dict | None = None,
) -> tuple[ReceiverPacket, ...]:
    """Build empirical packets from cheap features only; this function accepts no K."""
    total_started = perf_counter()
    if pair_features.ndim != 3 or pair_features.shape[0] != len(patch_receiver_ids):
        raise ValueError("Pair features must be [P,M,F] and align with patch receiver groups")
    if pair_features.shape[2] != scorer.input_width or pair_features.shape[1] < 1:
        raise ValueError("Pair feature width or patch count differs from the scorer contract")
    modules = len(binding.source_ids)
    if pair_features.shape[1] != modules:
        raise ValueError("Proposal pair features must be [patches, physical sources, feature width]")
    if len(action_domain.source_ids) != modules or action_domain.source_ids != binding.source_ids:
        raise ValueError("Action radii and binding must use the same physical source order")
    if protected_near_source_ids is None:
        protected_near_source_ids = [()] * len(patch_receiver_ids)
    if len(protected_near_source_ids) != len(patch_receiver_ids):
        raise ValueError("Protected near-source lists must align with receiver patches")
    budget = response_rms_budget(budget_fraction, train_response_rms)
    radii = action_domain.radii_tensor(device=pair_features.device, dtype=pair_features.dtype)
    hash_seconds = 0.0
    if fit_receipt is None or not math.isfinite(fit_receipt.train_only_residual_margin):
        margin, status = None, "full_fallback_missing_train_calibration"
    else:
        hash_started = perf_counter()
        parameter_hash = scorer_parameter_sha256(scorer)
        hash_seconds = perf_counter() - hash_started
        if not fit_receipt.parameter_sha256 or parameter_hash != fit_receipt.parameter_sha256:
            raise ValueError("Proposer weights differ from the TRAIN calibration receipt")
        margin, status = fit_receipt.train_only_residual_margin, "empirical_packet_proposal"
    packets = []
    forward_started = perf_counter()
    with torch.no_grad():
        predicted_log = scorer(pair_features)
        raw_upper = torch.exp((predicted_log + (margin or 0.0)).clamp(max=80.0))
    _sync_tensor_device(pair_features)
    forward_seconds = perf_counter() - forward_started
    selection_started = perf_counter()
    for patch_index, receiver_group in enumerate(patch_receiver_ids):
        receiver_group_ids = _ids(receiver_group, f"patch[{patch_index}] receiver_ids")
        if not set(receiver_group_ids).issubset(binding.receiver_ids):
            raise ValueError("Receiver patch contains IDs outside its bound catalogue")
        if status != "empirical_packet_proposal" or not bool(torch.isfinite(raw_upper[patch_index]).all()):
            selected = binding.source_ids
            omitted_bound = 0.0
            packet_status = "full_fallback_nonfinite_or_uncalibrated"
        else:
            importance_upper = raw_upper[patch_index] * radii
            selected, omitted_bound = deterministic_cover(
                importance_upper,
                binding.source_ids,
                budget,
                protected_source_ids=protected_near_source_ids[patch_index],
            )
            packet_status = status
        packets.append(
            ReceiverPacket(
                packet_id=f"{binding.scene_id}:patch:{patch_index}",
                binding_fingerprint=binding.fingerprint,
                action_domain_fingerprint=action_domain.fingerprint,
                receiver_ids=receiver_group_ids,
                source_ids=tuple(selected),
                environment_ancestry_ids=binding.environment_context_ids,
                ancestry_ids=binding.ancestry,
                output_roles=binding.output_roles,
                control_roles=binding.control_roles,
                units=binding.units,
                proposal_status=f"{packet_status};upper_action_mass_omitted={omitted_bound:.12g}",
                residual_margin=margin,
            )
        )
    selection_seconds = perf_counter() - selection_started
    if proposal_accounting is not None:
        proposal_accounting.update(
            {
                "scorer_weight_hash_validation_seconds": hash_seconds,
                "batched_scorer_forward_seconds": forward_seconds,
                "deterministic_packet_selection_seconds": selection_seconds,
                "proposal_total_seconds": perf_counter() - total_started,
                "proposal_full_K_calls": 0,
            }
        )
    return tuple(packets)


@torch.no_grad()
def export_packets(
    scorer: ReceiverPairScorer,
    fit_receipt: ProposerFitReceipt | None,
    pair_features: torch.Tensor,
    binding: InteractionBinding,
    action_domain: ActionDomain,
    patch_receiver_ids: Sequence[Sequence[str]],
    *,
    budget_fraction: float,
    train_response_rms: float,
    protected_near_source_ids: Sequence[Sequence[str]] | None = None,
    proposal_accounting: dict | None = None,
) -> tuple[ReceiverPacket, ...]:
    """Public export endpoint for learned receiver packets; never evaluates K."""
    return propose_receiver_packets(
        scorer,
        fit_receipt,
        pair_features,
        binding,
        action_domain,
        patch_receiver_ids,
        budget_fraction=budget_fraction,
        train_response_rms=train_response_rms,
        protected_near_source_ids=protected_near_source_ids,
        proposal_accounting=proposal_accounting,
    )


@torch.no_grad()
def apply_affine_packet_increment(
    operator,
    prepared_context,
    receivers: torch.Tensor,
    exact_baseline: torch.Tensor,
    delta_control: torch.Tensor,
    binding: InteractionBinding,
    current_binding: InteractionBinding,
    action_domain: ActionDomain,
    packet_union: PacketUnion | None = None,
    *,
    mode: Literal["full", "packet_advisory", "packet_experimental"] = DEFAULT_PACKET_MODE,
    requested_receiver_ids: Sequence[str] | None = None,
    baseline_binding: InteractionBinding | None = None,
    receiver_features: torch.Tensor | None = None,
    receiver_ids: torch.Tensor | None = None,
    accumulation_dtype=None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Apply a fixed-scene affine increment to a caller-supplied exact baseline.

    This consumer supports only geometry-defined receivers (``query_width=0``).
    Feature-dependent kernels need an exact-baseline binding that records the
    receiver features used for both baseline and increment.
    Grouped readouts are also outside this consumer until their derived group
    caches are included in the binding fingerprint.
    """
    if getattr(operator, "mode", None) != "direct":
        raise ValueError(
            "The packet increment consumer currently supports direct readouts only; "
            "grouped readouts require derived-cache binding"
        )
    query_width = int(getattr(operator, "query_width", 0))
    if query_width != 0 or (
        receiver_features is not None
        and (not torch.is_tensor(receiver_features) or receiver_features.shape[-1] != 0)
    ):
        raise ValueError(
            "Nonzero-width receiver features require a receiver-feature-aware exact-baseline binding; "
            "the packet consumer supports query_width=0 only"
        )
    baseline_binding = binding if baseline_binding is None else baseline_binding
    if baseline_binding.fingerprint != current_binding.fingerprint:
        raise ValueError("Exact baseline belongs to a stale scene; rebuild the baseline before applying this increment")
    if requested_receiver_ids is None:
        if packet_union is not None:
            requested_receiver_ids = packet_union.receiver_ids
        elif receivers.shape[1] == len(current_binding.receiver_ids):
            requested_receiver_ids = current_binding.receiver_ids
        else:
            raise ValueError("Subset receiver reads require explicit physical IDs bound to their coordinates")
    receiver_binding_validation_seconds = validate_bound_receiver_coordinates(
        current_binding, receivers, requested_receiver_ids
    )
    _validate_prepared_source_binding(current_binding, prepared_context)
    prepared_context_validation_started = perf_counter()
    prepared_geometry_fingerprint, prepared_context_fingerprint = prepared_context_fingerprints(prepared_context)
    prepared_context_fingerprint_validation_seconds = perf_counter() - prepared_context_validation_started
    if (
        prepared_geometry_fingerprint != current_binding.geometry_fingerprint
        or prepared_context_fingerprint != current_binding.context_fingerprint
    ):
        raise ValueError("Prepared context fingerprints differ from the current scene binding")
    stale_binding = current_binding.fingerprint != binding.fingerprint
    if packet_union is not None:
        if packet_union.binding_fingerprint != binding.fingerprint:
            raise ValueError("Packet union belongs to a different scene binding than the supplied packet binding")
        if packet_union.source_ids != binding.source_ids:
            raise ValueError("Packet union physical source order differs from its scene binding")
        if packet_union.receiver_ids != tuple(str(value) for value in requested_receiver_ids):
            raise ValueError("Packet union receiver IDs or order differ from the requested receiver read")
        if not stale_binding and packet_union.action_domain_fingerprint != action_domain.fingerprint:
            raise ValueError("Packet union belongs to a different action domain")
    if stale_binding:
        # The stale packet is invalid. Full access is deterministic and remains safe.
        requested_ids = requested_receiver_ids
        if requested_ids is None:
            requested_ids = current_binding.receiver_ids if packet_union is None else packet_union.receiver_ids
        packet_union = full_access_union(current_binding, action_domain, requested_ids, device=receivers.device)
        binding = current_binding
    if binding.capability != "affine_scalar_increment":
        raise ValueError("Finite packet increments require an explicitly affine scalar-control capability")
    if not _action_matches_binding(action_domain, binding):
        raise ValueError("Increment action role, physical unit, or source order differs from its scene binding")
    if packet_union is None:
        if mode != "full":
            raise ValueError("Advisory and experimental modes require a concrete packet proposal")
        requested_ids = binding.receiver_ids if requested_receiver_ids is None else requested_receiver_ids
        packet_union = full_access_union(
            binding, action_domain, requested_ids, device=receivers.device, fallback=False, reason="full_mode"
        )
    if packet_union.binding_fingerprint != binding.fingerprint:
        raise ValueError("Resolved packet union does not match the scene binding")
    if packet_union.action_domain_fingerprint != action_domain.fingerprint:
        raise ValueError("Resolved packet union does not match the action domain")
    if packet_union.receiver_ids != tuple(str(value) for value in requested_receiver_ids):
        raise ValueError("Resolved packet union receiver order differs from the request")
    if packet_union.source_ids != binding.source_ids:
        raise ValueError("Resolved packet union source order differs from the bound physical sources")
    packet_union_integrity_started = perf_counter()
    _validate_packet_union_integrity(packet_union)
    packet_union_integrity_validation_seconds = perf_counter() - packet_union_integrity_started
    if hasattr(operator, "assert_owned"):
        operator.assert_owned(prepared_context)
    elif hasattr(prepared_context, "assert_fresh"):
        prepared_context.assert_fresh()
    source_ids = binding.source_ids
    if action_domain.source_ids != source_ids:
        raise ValueError("Action-domain source order differs from the bound physical sources")
    if delta_control.ndim != 2 or delta_control.shape[-1] != len(source_ids):
        raise ValueError("Affine increment must align with active physical source IDs")
    action_ok = action_domain.accepts(delta_control, source_ids)
    if not action_ok and mode != "full":
        packet_union = full_access_union(binding, action_domain, packet_union.receiver_ids, device=receivers.device)
    proposed_mask, mode_receipt = execution_source_mask(mode, proposed=packet_union, device=receivers.device)
    if mode == "full" and packet_union.source_keep.shape[0] != receivers.shape[1]:
        proposed_mask = torch.ones(
            receivers.shape[0], receivers.shape[1], len(source_ids), dtype=torch.bool, device=receivers.device
        )
    else:
        proposed_mask = proposed_mask.unsqueeze(0).expand(receivers.shape[0], -1, -1)
    if proposed_mask.shape != (receivers.shape[0], receivers.shape[1], len(source_ids)):
        raise ValueError("Packet receiver rows or physical source columns do not align with the read request")
    # Packet identities use the active physical catalog, while the preserved
    # adapter may carry padded source slots. Expand only at the executor edge.
    full_source_count = int(prepared_context.centers.shape[1])
    if any(slot >= full_source_count for slot in binding.source_slots):
        raise ValueError("Bound active physical source slot exceeds the prepared source capacity")
    active_source_rows = proposed_mask
    source_mask = torch.zeros(
        receivers.shape[0],
        receivers.shape[1],
        full_source_count,
        dtype=torch.bool,
        device=receivers.device,
    )
    source_mask[:, :, list(binding.source_slots)] = active_source_rows
    has_packet_fallback = any(packet_union.fallback)
    has_active_source_omission = not bool(active_source_rows.all())
    use_subset = mode == "packet_experimental" and has_active_source_omission and not has_packet_fallback
    if use_subset:
        if not hasattr(operator, "prepare_receivers_subset"):
            raise RuntimeError("Shared interaction core does not expose the required pre-MLP subset gather API")
        response = operator.prepare_receivers_subset(
            prepared_context,
            receivers,
            source_keep=source_mask,
            receiver_features=receiver_features,
            receiver_ids=receiver_ids,
        )
    else:
        response = operator.prepare_receivers(
            prepared_context,
            receivers,
            receiver_features=receiver_features,
            receiver_ids=receiver_ids,
        )
        near = operator.near_weight(
            receivers,
            prepared_context.centers,
            prepared_context.source_lengths,
            prepared_context.present,
        )
        full_far_rows = int(receivers.shape[0] * receivers.shape[1] * prepared_context.centers.shape[1])
        response.execution_receipt = {
            "mode": "full_access"
            if mode == "full"
            else (
                "packet_experimental_full_fallback" if mode == "packet_experimental" else "packet_advisory_full_access"
            ),
            "fine_rows": full_far_rows,
            "near_rows": int((near > 0).sum().item()),
            "full_far_rows": full_far_rows,
            "context_ancestry": "full source/source and source/environment context",
            "validity": "exact baseline plus affine increment; full source access",
        }
        mode_receipt["packet_mask_executed"] = False
    mode_receipt["selected_active_source_pair_rows"] = int(active_source_rows.sum().item())
    padded_delta = delta_control.new_zeros(delta_control.shape[0], full_source_count)
    padded_delta[:, list(binding.source_slots)] = delta_control
    increment = operator.apply_increment(response, padded_delta, accumulation_dtype=accumulation_dtype)
    if exact_baseline.shape != increment.shape:
        raise ValueError("Exact baseline and affine increment must have the same receiver/output shape")
    receipt = {
        **mode_receipt,
        "packet_proposal_used": bool(mode != "full"),
        "action_domain_valid": bool(action_ok),
        "stale_binding_fallback": stale_binding,
        "used_pre_mlp_subset_gather": use_subset,
        "baseline_binding": baseline_binding.fingerprint,
        "scene_binding": current_binding.fingerprint,
        "receiver_binding_validation_seconds": receiver_binding_validation_seconds,
        "prepared_context_fingerprint_validation_seconds": prepared_context_fingerprint_validation_seconds,
        "packet_union_integrity_validation_seconds": packet_union_integrity_validation_seconds,
        "increment_accumulation_dtype": str(accumulation_dtype or delta_control.dtype),
        "exact_baseline_reused": True,
        "proposal_full_kernel_calls": 0,
        "executed_subset_receipt": getattr(response, "execution_receipt", {}),
        "source_ids": list(source_ids),
        "source_slots": list(binding.source_slots),
        "prepared_source_ids": list(binding.prepared_source_ids),
        "proposed_active_source_pair_rows": int(packet_union.source_keep.sum().item()),
        "selected_active_source_pair_rows": int(active_source_rows.sum().item()),
        "executed_source_rows": int(getattr(response, "execution_receipt", {}).get("fine_rows", 0)),
        "full_far_rows": int(getattr(response, "execution_receipt", {}).get("full_far_rows", 0)),
        "output_capability": binding.capability,
    }
    return exact_baseline + increment, receipt

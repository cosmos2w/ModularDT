"""Type-qualified native records and immutable Round9 routing results."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..native_retrieval.contracts import _require, _validate_ids, tensor_versions


@dataclass(frozen=True)
class CanonicalGeometry:
    """Scene-only geometry ordering and duplicate groups; IDs never break ties."""

    ranks: Tensor
    duplicate_groups: Tensor
    keys: Tensor

    @classmethod
    def build(cls, keys: Tensor) -> "CanonicalGeometry":
        if keys.ndim != 3 or not keys.shape[-1]:
            raise ValueError("Canonical geometry requires [B,N,G] physical keys.")
        batch, count, dimensions = keys.shape
        order = torch.arange(count, device=keys.device).expand(batch, count)
        for dim in range(dimensions - 1, -1, -1):
            order = order.gather(1, keys[:, :, dim].gather(1, order).argsort(dim=1, stable=True))
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, torch.arange(count, device=keys.device).expand_as(order))
        sorted_keys = keys.gather(1, order[:, :, None].expand(batch, count, dimensions))
        starts = torch.ones((batch, count), dtype=torch.bool, device=keys.device)
        if count > 1:
            starts[:, 1:] = (sorted_keys[:, 1:] != sorted_keys[:, :-1]).any(-1)
        sorted_groups = starts.long().cumsum(-1) - 1
        groups = torch.empty_like(order)
        groups.scatter_(1, order, sorted_groups)
        return cls(ranks, groups, keys)


@dataclass(frozen=True)
class TypedNativeRecords:
    """M followed by E, with disjoint (type, native-ID) identities.

    M presence includes zero-information-measure physical heaters. E retrieval
    eligibility requires positive native quadrature measure. Raw E presence is
    retained separately and still determines the native seed catalogue.
    """

    module_ids: Tensor
    environment_ids: Tensor
    module_present: Tensor
    environment_present: Tensor
    module_coordinates: Tensor
    environment_coordinates: Tensor
    module_lengths: Tensor
    environment_lengths: Tensor
    module_measure: Tensor
    environment_measure: Tensor
    geometry: CanonicalGeometry = field(init=False)
    _versions: tuple = field(init=False, repr=False)

    def __post_init__(self) -> None:
        batch = self.module_ids.shape[0]
        for kind in ("module", "environment"):
            ids, valid = getattr(self, kind + "_ids"), getattr(self, kind + "_present")
            coordinates = getattr(self, kind + "_coordinates")
            lengths, measure = getattr(self, kind + "_lengths"), getattr(self, kind + "_measure")
            _validate_ids(ids, valid, kind + " IDs")
            if ids.shape[0] != batch or coordinates.ndim != 3 or coordinates.shape[:2] != ids.shape:
                raise ValueError("Native coordinates must preserve [B,N] correspondence.")
            if lengths.shape != coordinates.shape or measure.shape != ids.shape:
                raise ValueError("Native lengths/measures must match their record axes.")
            _require(torch.isfinite(coordinates), "Coordinates must be finite.")
            _require(torch.isfinite(lengths) & (lengths > 0), "Lengths must be finite and positive.")
            _require(torch.isfinite(measure) & (measure >= 0), "Measures must be finite and nonnegative.")
        if self.module_coordinates.shape[-1] != self.environment_coordinates.shape[-1]:
            raise ValueError("M and E coordinate dimensions must agree.")
        keys = torch.cat((self.coordinates, self.lengths, self.types[..., None].to(self.coordinates)), -1)
        object.__setattr__(self, "geometry", CanonicalGeometry.build(keys))
        object.__setattr__(self, "_versions", tensor_versions(*self._tensors()))

    @classmethod
    def from_scene(cls, scene) -> "TypedNativeRecords":
        return cls(scene.module_ids, scene.environment_ids, scene.module_mask, scene.environment_mask,
                   scene.module_coordinates, scene.environment_coordinates,
                   scene.module_characteristic_lengths, scene.environment_characteristic_lengths,
                   scene.module_measure, scene.environment_measure)

    def _tensors(self) -> tuple[Tensor, ...]:
        original = tuple(getattr(self, kind + "_" + name) for kind in ("module", "environment")
                         for name in ("ids", "present", "coordinates", "lengths", "measure"))
        return original + (self.geometry.keys, self.geometry.ranks, self.geometry.duplicate_groups)

    def assert_current(self) -> None:
        if tensor_versions(*self._tensors()) != self._versions:
            raise ValueError("Typed native records changed after preparation.")

    @property
    def module_count(self) -> int:
        return self.module_ids.shape[1]

    @property
    def ids(self) -> Tensor:
        return torch.cat((self.module_ids, self.environment_ids), -1)

    @property
    def types(self) -> Tensor:
        return torch.cat((torch.zeros_like(self.module_ids), torch.ones_like(self.environment_ids)), -1)

    @property
    def present(self) -> Tensor:
        return torch.cat((self.module_present, self.environment_present), -1)

    @property
    def eligible(self) -> Tensor:
        return torch.cat((self.module_present, self.environment_present & (self.environment_measure > 0)), -1)

    @property
    def coordinates(self) -> Tensor:
        return torch.cat((self.module_coordinates, self.environment_coordinates), 1)

    @property
    def lengths(self) -> Tensor:
        return torch.cat((self.module_lengths, self.environment_lengths), 1)


@dataclass(frozen=True)
class JointNativeDirectory:
    records: TypedNativeRecords
    membership_logits: Tensor
    membership_probabilities: Tensor
    membership_hard: Tensor
    membership_straight_through: Tensor
    self_membership: Tensor
    cross_membership: Tensor
    _versions: tuple = field(init=False, repr=False)

    def __post_init__(self) -> None:
        batch, count = self.records.ids.shape
        for value in self._tensors():
            if value.shape != (batch, count, count):
                raise ValueError("Joint incidence must preserve [B,M+E,M+E].")
        for value in (self.membership_hard, self.self_membership, self.cross_membership):
            if value.dtype != torch.bool:
                raise TypeError("Hard incidence and mandatory memberships must be boolean.")
        object.__setattr__(self, "_versions", tensor_versions(*self._tensors()))

    def _tensors(self):
        return (self.membership_logits, self.membership_probabilities, self.membership_hard,
                self.membership_straight_through, self.self_membership, self.cross_membership)

    def assert_current(self) -> None:
        self.records.assert_current()
        if tensor_versions(*self._tensors()) != self._versions:
            raise ValueError("Joint directory changed after sampling.")

    @property
    def mandatory_membership(self) -> Tensor:
        return self.self_membership | self.cross_membership

    @property
    def deterministic_hard(self) -> Tensor:
        allowed = self.records.present[:, :, None] & self.records.eligible[:, None, :]
        return ((self.membership_logits >= 0) & allowed) | self.mandatory_membership


@dataclass(frozen=True)
class JointReadPlan:
    """Receiver supports plus hard-forward gates, never pooled physical values."""

    hard_support: Tensor
    value_gates: Tensor
    mandatory_support: Tensor
    protected_mask: Tensor
    module_count: int
    dispatch: object | None = None
    probabilities: Tensor | None = None

    @property
    def read_mask(self) -> Tensor:
        return self.hard_support

    @property
    def M_mask(self) -> Tensor:
        return self.hard_support[..., :self.module_count]

    @property
    def E_mask(self) -> Tensor:
        return self.hard_support[..., self.module_count:]

    @property
    def seed_index(self) -> Tensor | None:
        return None if self.dispatch is None else self.dispatch.dispatch_seed

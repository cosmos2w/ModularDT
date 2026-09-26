"""Family-safe stencils and mask-aware finite response examples."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from .types import EvidenceSource, EvidenceSplit, RoleOutput, SolveRecord, SolveStatus


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _same_context(left: Any, right: Any) -> bool:
    return json.dumps(_jsonable(left.values), sort_keys=True) == json.dumps(_jsonable(right.values), sort_keys=True)


@dataclass(frozen=True)
class ResponseBlock:
    """A finite response on a stencil-wide common receiver set."""

    role: str
    label: str
    query_features: np.ndarray
    delta: np.ndarray
    valid_mask: np.ndarray
    quadrature_weights: np.ndarray
    channel_names: tuple[str, ...]
    channel_units: tuple[str, ...]
    query_ids: tuple[str, ...]
    receiver_module_ids: tuple[str, ...] | None
    noise_floor: np.ndarray | None = None

    def __post_init__(self) -> None:
        features = _readonly(self.query_features, dtype=np.float64)
        delta = _readonly(self.delta, dtype=np.float64)
        mask = _readonly(self.valid_mask, dtype=bool)
        weights = _readonly(self.quadrature_weights, dtype=np.float64).reshape(-1)
        if features.ndim != 2 or delta.ndim != 2 or delta.shape[0] != features.shape[0]:
            raise ValueError("ResponseBlock features and delta must align as [N,D], [N,C].")
        if mask.shape != delta.shape or weights.shape != (features.shape[0],):
            raise ValueError("ResponseBlock mask and quadrature weights must align to samples.")
        if np.any(~np.isfinite(delta[mask])):
            raise ValueError("Resolved finite changes must be finite.")
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("ResponseBlock weights must be finite and nonnegative.")
        if delta.shape[1] != len(self.channel_names) or len(self.channel_names) != len(self.channel_units):
            raise ValueError("ResponseBlock channel metadata is inconsistent.")
        if len(self.query_ids) != features.shape[0]:
            raise ValueError("ResponseBlock query IDs are not aligned.")
        floor = None
        if self.noise_floor is not None:
            floor = _readonly(self.noise_floor, dtype=np.float64)
            if floor.shape != delta.shape or np.any(floor < 0.0) or not np.isfinite(floor).all():
                raise ValueError("ResponseBlock noise floor must be finite [N,C].")
        object.__setattr__(self, "query_features", features)
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(self, "quadrature_weights", weights)
        object.__setattr__(self, "channel_names", tuple(self.channel_names))
        object.__setattr__(self, "channel_units", tuple(self.channel_units))
        object.__setattr__(self, "query_ids", tuple(self.query_ids))
        object.__setattr__(self, "receiver_module_ids", None if self.receiver_module_ids is None else tuple(self.receiver_module_ids))
        object.__setattr__(self, "noise_floor", floor)

    @property
    def observed_mask(self) -> np.ndarray:
        """Common geometric/source support; finite-response loss may use this mask."""

        return self.valid_mask

    @property
    def resolved_sign_mask(self) -> np.ndarray | None:
        """Sign/derivative labels resolved above a measured floor, or None if unknown."""

        if self.noise_floor is None:
            return None
        return self.valid_mask & (np.abs(self.delta) > self.noise_floor)

    @property
    def unresolved_sign_mask(self) -> np.ndarray | None:
        """Observed values whose sign is at or below a supplied numerical floor."""

        resolved = self.resolved_sign_mask
        return None if resolved is None else self.valid_mask & ~resolved


@dataclass(frozen=True)
class ResponseExample:
    """One anchored trial delta with role-wise resolved finite-change labels."""

    record_id: str
    anchor_id: str
    physical_family_id: str
    split: EvidenceSplit
    source: EvidenceSource
    base_design: Any
    context: Any
    perturbation_by_module: Mapping[str, tuple[float, ...]]
    physical_step_scales: Mapping[str, float]
    roles: Mapping[str, ResponseBlock]

    def __post_init__(self) -> None:
        object.__setattr__(self, "perturbation_by_module", MappingProxyType(dict(self.perturbation_by_module)))
        object.__setattr__(self, "physical_step_scales", MappingProxyType(dict(self.physical_step_scales)))
        object.__setattr__(self, "roles", MappingProxyType(dict(self.roles)))


@dataclass(frozen=True)
class ResponseStencil:
    """A baseline and its perturbation family, held to one source and split."""

    baseline: SolveRecord
    variants: Mapping[str, SolveRecord]

    def __post_init__(self) -> None:
        variants = dict(self.variants)
        if not variants:
            raise ValueError("A response stencil requires at least one perturbed solve.")
        if "baseline" in variants:
            raise ValueError("The baseline belongs in baseline, not variants.")
        records = [self.baseline, *variants.values()]
        for record in records:
            if record.status is not SolveStatus.CONVERGED or record.output is None:
                raise ValueError(f"Stencil member {record.record_id!r} is not converged.")
        reference = self.baseline
        for record in variants.values():
            if record.source is not reference.source:
                raise ValueError("A stencil cannot mix evidence sources.")
            if record.design.anchor_id != reference.design.anchor_id:
                raise ValueError("All stencil members must retain one anchor ID.")
            if record.design.physical_family_id != reference.design.physical_family_id:
                raise ValueError("All stencil members must retain one physical family.")
            if record.design.split is not reference.design.split:
                raise ValueError("A physical family cannot cross evidence splits.")
            if record.design.active_module_ids != reference.design.active_module_ids:
                raise ValueError("Perturbations must preserve active module identities.")
            if not _same_context(record.context, reference.context):
                raise ValueError("Every stencil member must use the same operating context.")
        object.__setattr__(self, "variants", MappingProxyType(variants))

    @property
    def anchor_id(self) -> str:
        return self.baseline.design.anchor_id

    @property
    def physical_family_id(self) -> str:
        return self.baseline.design.physical_family_id

    @property
    def split(self) -> EvidenceSplit:
        return EvidenceSplit(self.baseline.design.split)

    @property
    def source(self) -> EvidenceSource:
        return EvidenceSource(self.baseline.source)

    @property
    def records(self) -> tuple[SolveRecord, ...]:
        return (self.baseline, *self.variants.values())

    def common_mask(self, role: str) -> np.ndarray:
        """Intersect receiver/channel support over every stencil member."""

        role_outputs = [record.output.roles[role] for record in self.records]  # type: ignore[union-attr]
        first = role_outputs[0]
        for output in role_outputs[1:]:
            _validate_aligned_role(first, output)
        return _readonly(np.logical_and.reduce([output.valid_mask for output in role_outputs]), dtype=bool)

    def finite_change(self, variant_label: str, role: str) -> ResponseBlock:
        """Return trial minus this stencil's own solved baseline, without zero filling."""

        if variant_label not in self.variants:
            raise KeyError(f"Unknown stencil variant {variant_label!r}.")
        baseline = self.baseline.output.roles[role]  # type: ignore[union-attr]
        trial = self.variants[variant_label].output.roles[role]  # type: ignore[union-attr]
        _validate_aligned_role(baseline, trial)
        common = self.common_mask(role)
        delta = np.full(trial.values.shape, np.nan, dtype=np.float64)
        delta[common] = trial.values[common] - baseline.values[common]
        floor = _combined_noise_floor([self.baseline.output.roles[role], self.variants[variant_label].output.roles[role]])  # type: ignore[union-attr]
        return ResponseBlock(
            role=role,
            label=variant_label,
            query_features=baseline.query_features,
            delta=delta,
            valid_mask=common,
            quadrature_weights=baseline.quadrature_weights,
            channel_names=baseline.channel_names,
            channel_units=baseline.channel_units,
            query_ids=baseline.query_ids,
            receiver_module_ids=baseline.receiver_module_ids,
            noise_floor=floor,
        )

    def anchored_interaction(
        self,
        *,
        role: str,
        joint_variant: str,
        first_variant: str,
        second_variant: str,
        label: str = "anchored_mixed_response",
    ) -> ResponseBlock:
        """Compute y_ij - y_i - y_j + y_0 on the full-stencil common support."""

        required = (joint_variant, first_variant, second_variant)
        missing = [name for name in required if name not in self.variants]
        if missing:
            raise KeyError(f"Missing interaction corners: {missing}.")
        outputs = [self.baseline.output.roles[role]]  # type: ignore[union-attr]
        outputs.extend(self.variants[name].output.roles[role] for name in required)  # type: ignore[union-attr]
        for output in outputs[1:]:
            _validate_aligned_role(outputs[0], output)
        common = self.common_mask(role)
        values = np.full(outputs[0].values.shape, np.nan, dtype=np.float64)
        base, joint, first, second = (output.values for output in outputs)
        values[common] = joint[common] - first[common] - second[common] + base[common]
        return ResponseBlock(
            role=role,
            label=label,
            query_features=outputs[0].query_features,
            delta=values,
            valid_mask=common,
            quadrature_weights=outputs[0].quadrature_weights,
            channel_names=outputs[0].channel_names,
            channel_units=outputs[0].channel_units,
            query_ids=outputs[0].query_ids,
            receiver_module_ids=outputs[0].receiver_module_ids,
            noise_floor=_combined_noise_floor(outputs),
        )

    def centered_mixed_difference(
        self,
        *,
        role: str,
        joint_pp: str,
        joint_pm: str,
        joint_mp: str,
        joint_mm: str,
        first_step: float,
        second_step: float,
    ) -> ResponseBlock:
        """Compute (y++ - y+- - y-+ + y--)/(4 h_i h_j) from joint corners."""

        labels = (joint_pp, joint_pm, joint_mp, joint_mm)
        if any(label not in self.variants for label in labels):
            raise KeyError("All four centered mixed-difference variants must exist.")
        if not np.isfinite([first_step, second_step]).all() or first_step <= 0.0 or second_step <= 0.0:
            raise ValueError("Mixed-difference step sizes must be positive and finite.")
        outputs = [self.baseline.output.roles[role]]  # type: ignore[union-attr]
        outputs.extend(self.variants[label].output.roles[role] for label in labels)  # type: ignore[union-attr]
        for output in outputs[1:]:
            _validate_aligned_role(outputs[0], output)
        common = self.common_mask(role)
        pp, pm, mp, mm = (self.variants[label].output.roles[role].values for label in labels)  # type: ignore[union-attr]
        mixed = np.full(outputs[0].values.shape, np.nan, dtype=np.float64)
        mixed[common] = (pp[common] - pm[common] - mp[common] + mm[common]) / (4.0 * first_step * second_step)
        mixed_noise_floor = _combined_noise_floor(outputs[1:])
        if mixed_noise_floor is not None:
            mixed_noise_floor = mixed_noise_floor / (4.0 * first_step * second_step)
        return ResponseBlock(
            role=role,
            label="centered_mixed_derivative",
            query_features=outputs[0].query_features,
            delta=mixed,
            valid_mask=common,
            quadrature_weights=outputs[0].quadrature_weights,
            channel_names=outputs[0].channel_names,
            channel_units=outputs[0].channel_units,
            query_ids=outputs[0].query_ids,
            receiver_module_ids=outputs[0].receiver_module_ids,
            noise_floor=mixed_noise_floor,
        )

    def examples(self) -> tuple[ResponseExample, ...]:
        """Materialize one geometry-only-query finite-change example per trial."""

        examples = []
        role_names = tuple(self.baseline.output.roles)  # type: ignore[union-attr]
        for label, record in self.variants.items():
            roles = {name: self.finite_change(label, name) for name in role_names}
            examples.append(
                ResponseExample(
                    record_id=record.record_id,
                    anchor_id=record.design.anchor_id,
                    physical_family_id=record.design.physical_family_id,
                    split=EvidenceSplit(record.design.split),
                    source=EvidenceSource(record.source),
                    base_design=self.baseline.design,
                    context=self.baseline.context,
                    perturbation_by_module=record.perturbation_by_module,
                    physical_step_scales=record.physical_step_scales,
                    roles=roles,
                )
            )
        return tuple(examples)


def _combined_noise_floor(outputs: list[RoleOutput]) -> np.ndarray | None:
    if any(output.noise_floor is None for output in outputs):
        return None
    floors = np.stack([output.noise_floor for output in outputs], axis=0)  # type: ignore[arg-type]
    return np.sqrt(np.sum(np.square(floors), axis=0))


def _validate_aligned_role(reference: RoleOutput, candidate: RoleOutput) -> None:
    if reference.role != candidate.role:
        raise ValueError("Output roles differ.")
    if reference.values.shape != candidate.values.shape or reference.query_features.shape != candidate.query_features.shape:
        raise ValueError(f"Role {reference.role!r} receiver shapes differ across stencil members.")
    if reference.channel_names != candidate.channel_names or reference.channel_units != candidate.channel_units:
        raise ValueError(f"Role {reference.role!r} channel schemas differ across stencil members.")
    if reference.query_ids != candidate.query_ids or reference.receiver_module_ids != candidate.receiver_module_ids:
        raise ValueError(f"Role {reference.role!r} material receiver identities changed across the stencil.")
    if not np.array_equal(reference.query_features, candidate.query_features):
        raise ValueError(f"Role {reference.role!r} fixed query/material coordinates differ across stencil members.")
    if not np.array_equal(reference.quadrature_weights, candidate.quadrature_weights):
        raise ValueError(f"Role {reference.role!r} quadrature weights differ across stencil members.")


def validate_family_splits(records: list[SolveRecord] | tuple[SolveRecord, ...]) -> None:
    """Reject any physical family that appears under more than one split."""

    seen: dict[str, EvidenceSplit] = {}
    for record in records:
        family = record.design.physical_family_id
        split = EvidenceSplit(record.design.split)
        previous = seen.setdefault(family, split)
        if previous is not split:
            raise ValueError(f"Physical family {family!r} crosses splits {previous.value!r} and {split.value!r}.")


def attach_role_noise_floors(
    record: SolveRecord,
    floors_by_role: Mapping[str, np.ndarray],
) -> SolveRecord:
    """Return a copy of a record with measured role/channel floors attached.

    Floors must align with the corresponding role's sampled values. Unknown
    roles and non-finite or negative values are rejected; no missing floor is
    replaced with zero.
    """

    if record.output is None:
        raise ValueError("Noise floors can be attached only to a record with output values.")
    unknown = sorted(set(floors_by_role) - set(record.output.roles))
    if unknown:
        raise KeyError(f"Noise floors reference unknown roles: {unknown}.")
    roles = dict(record.output.roles)
    for name, floor in floors_by_role.items():
        roles[name] = replace(roles[name], noise_floor=floor)
    output = replace(record.output, roles=roles)
    return replace(record, output=output)


def estimate_tightening_response_floor(
    nominal: ResponseBlock,
    tightened: ResponseBlock,
) -> ResponseBlock:
    """Attach the measured change in a response under tighter convergence.

    The pointwise floor is ``abs(delta_tight - delta_nominal)`` on the
    intersection of the two observed supports. The returned block preserves
    ``valid_mask`` as geometric/source support; sign claims can separately use
    ``resolved_sign_mask``. A zero outside that support is only a storage
    placeholder and is never marked observed.
    """

    if nominal.role != tightened.role or nominal.channel_names != tightened.channel_names:
        raise ValueError("Tightening response blocks must have matching roles and channels.")
    if nominal.channel_units != tightened.channel_units:
        raise ValueError("Tightening response blocks must use matching channel units.")
    if nominal.query_ids != tightened.query_ids or nominal.receiver_module_ids != tightened.receiver_module_ids:
        raise ValueError("Tightening response blocks must preserve receiver identities.")
    if nominal.query_features.shape != tightened.query_features.shape or not np.array_equal(
        nominal.query_features, tightened.query_features
    ):
        raise ValueError("Tightening response blocks must use fixed, aligned queries.")
    if nominal.delta.shape != tightened.delta.shape:
        raise ValueError("Tightening response arrays must be aligned.")
    common = nominal.observed_mask & tightened.observed_mask
    floor = np.zeros_like(nominal.delta, dtype=np.float64)
    floor[common] = np.abs(tightened.delta[common] - nominal.delta[common])
    return replace(nominal, valid_mask=common, noise_floor=floor)


__all__ = [
    "ResponseBlock",
    "ResponseExample",
    "ResponseStencil",
    "attach_role_noise_floors",
    "estimate_tightening_response_floor",
    "validate_family_splits",
]

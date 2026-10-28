"""E-anchored access rows and the row-local typed member read.

The directory stores only positive original-member entries in packed rows.
Its shortlist indices are discrete, while the access coefficients stored at
those indices remain connected to the applied final-cycle EM/EE tensors.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..content_hypergraph.query_chunks_v1 import slice_query_v1
from .core import InteractionJointOutput, InteractionReusedMemberRoutingCore, _pack_work_counts, _tensor_version
from .guided import prior_weighted_scores
from .joint import build_joint_member_measure
from .relative import read_probability
from .single_distribution_core import SingleDistributionMemberRoutingCore


@dataclass(frozen=True)
class EAccessRow:
    """One packed E-receiver row over original M/E records."""

    member_index: Tensor
    original_id: Tensor
    is_module: Tensor
    access_weight: Tensor
    applied_exchange: Tensor
    structural_self: Tensor
    measure: Tensor


@dataclass(frozen=True)
class PackedEAccessRows:
    """Ragged E rows represented by flat tensors and small CPU offsets."""

    offsets: Tensor
    member_index: Tensor
    original_id: Tensor
    is_module: Tensor
    access_weight: Tensor
    applied_exchange: Tensor
    structural_self: Tensor
    measure: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.offsets.shape[0])

    @property
    def anchor_count(self) -> int:
        return int(self.offsets.shape[1] - 1)

    def row(self, batch_index: int, anchor_index: int) -> EAccessRow:
        if not 0 <= batch_index < self.batch_size:
            raise IndexError("E-access batch index is outside the packed directory.")
        if not 0 <= anchor_index < self.anchor_count:
            raise IndexError("E-access anchor index is outside the packed directory.")
        start = int(self.offsets[batch_index, anchor_index])
        stop = int(self.offsets[batch_index, anchor_index + 1])
        return EAccessRow(
            member_index=self.member_index[start:stop],
            original_id=self.original_id[start:stop],
            is_module=self.is_module[start:stop],
            access_weight=self.access_weight[start:stop],
            applied_exchange=self.applied_exchange[start:stop],
            structural_self=self.structural_self[start:stop],
            measure=self.measure[start:stop],
        )


@dataclass(frozen=True)
class EAccessDirectory:
    """Scene-conditioned full and joint-95 E receiver directories."""

    full: PackedEAccessRows
    restricted: PackedEAccessRows
    own_e_member_index: Tensor
    module_count: int
    environment_count: int
    retained_mass: float


@dataclass(frozen=True)
class AnchoredRoleRead:
    """One typed, row-local read and its exact probability/statistic state."""

    probabilities: Tensor
    calibrated_scores: Tensor
    expectation: Tensor
    normalized_entropy: Tensor
    gate_logits: Tensor
    strength: Tensor
    context_before_strength: Tensor | None
    context: Tensor | None


def _validate_measure(value: Tensor, shape: tuple[int, ...], name: str) -> None:
    if tuple(value.shape) != shape or not value.is_floating_point():
        raise ValueError(f"{name} must be floating point with shape {shape}.")
    if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
        raise ValueError(f"{name} must be finite and nonnegative.")


def _validate_ids(value: Tensor, shape: tuple[int, ...], name: str) -> None:
    if tuple(value.shape) != shape or value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be an integer tensor with shape {shape}.")


def _pack_masked_rows(
    candidate_indices: Tensor,
    selected: Tensor,
    access: Tensor,
    exchange: Tensor,
    self_access: Tensor,
    member_measure: Tensor,
    member_ids: Tensor,
    is_module: Tensor,
) -> PackedEAccessRows:
    """Pack selected ``[B,E,N]`` entries in row-major order without row loops."""
    batch_size, anchor_count = selected.shape[:2]
    coordinates = selected.nonzero(as_tuple=False)
    batch_index, anchor_index, candidate_position = coordinates.unbind(-1)
    member_index = candidate_indices[batch_index, anchor_index, candidate_position]

    # Offsets remain the small CPU-side ragged index used by row() and the
    # packed decoder. Counts are reduced over device rows and transferred once.
    row_counts = selected.sum(-1)
    batch_counts = row_counts.sum(-1)
    batch_starts = torch.cat((batch_counts.new_zeros((1,)), batch_counts.cumsum(0)[:-1]))
    row_ends = row_counts.cumsum(-1) + batch_starts[:, None]
    offsets = torch.cat((batch_starts[:, None], row_ends), dim=-1).to(device="cpu", dtype=torch.long)

    return PackedEAccessRows(
        offsets=offsets,
        member_index=member_index,
        original_id=member_ids[batch_index, member_index],
        is_module=is_module[member_index],
        access_weight=access[batch_index, anchor_index, member_index],
        applied_exchange=exchange[batch_index, anchor_index, member_index],
        structural_self=self_access[batch_index, anchor_index, member_index],
        measure=member_measure[batch_index, member_index],
    )


def compile_e_access_directory(
    applied_em: Tensor,
    applied_ee: Tensor,
    module_measure: Tensor,
    environment_measure: Tensor,
    module_mask: Tensor,
    environment_mask: Tensor,
    module_ids: Tensor,
    environment_ids: Tensor,
    *,
    own_e_member_index: Tensor | None = None,
    retained_mass: float = 0.95,
) -> EAccessDirectory:
    """Compile receiver-first ``(applied EM + applied EE + self) * μ`` rows.

    Each E slot is one directory receiver. The applied blocks are the actual
    final-cycle ``E←M`` and ``E←E`` relations. ``module_measure`` and
    ``environment_measure`` are the caller's already prepared μ values and
    are multiplied exactly once. The 95% prefix is selected over the whole
    normalized M+E row, with role/original-ID tie ordering; the anchor E
    member is appended if it did not enter that prefix.
    """
    if applied_em.ndim != 3 or applied_ee.ndim != 3:
        raise ValueError("Applied EM/EE blocks must have [B,E,M] and [B,E,E] axes.")
    batch_size, environment_count, module_count = applied_em.shape
    if tuple(applied_ee.shape) != (batch_size, environment_count, environment_count):
        raise ValueError("Applied EE must align with the E receiver and donor inventories.")
    if not (applied_em.is_floating_point() and applied_ee.is_floating_point()):
        raise TypeError("Applied EM/EE relation blocks must be floating point.")
    if (applied_em.device != applied_ee.device or applied_em.dtype != applied_ee.dtype):
        raise ValueError("Applied EM/EE blocks must share device and dtype.")
    if (not bool(torch.isfinite(applied_em).all()) or not bool(torch.isfinite(applied_ee).all())
            or bool((applied_em < 0).any()) or bool((applied_ee < 0).any())):
        raise ValueError("Applied EM/EE blocks must be finite and nonnegative.")
    _validate_measure(module_measure, (batch_size, module_count), "module_measure")
    _validate_measure(environment_measure, (batch_size, environment_count), "environment_measure")
    if module_measure.device != applied_em.device or environment_measure.device != applied_em.device:
        raise ValueError("Measures and applied relations must share a device.")
    if module_mask.shape != module_measure.shape or module_mask.dtype != torch.bool:
        raise ValueError("module_mask must be boolean and match module_measure.")
    if environment_mask.shape != environment_measure.shape or environment_mask.dtype != torch.bool:
        raise ValueError("environment_mask must be boolean and match environment_measure.")
    if module_mask.device != applied_em.device or environment_mask.device != applied_em.device:
        raise ValueError("Masks and applied relations must share a device.")
    _validate_ids(module_ids, (batch_size, module_count), "module_ids")
    _validate_ids(environment_ids, (batch_size, environment_count), "environment_ids")
    if module_ids.device != applied_em.device or environment_ids.device != applied_em.device:
        raise ValueError("Original IDs and applied relations must share a device.")
    if (isinstance(retained_mass, bool) or not isinstance(retained_mass, (float, int))
            or not 0.0 < float(retained_mass) <= 1.0):
        raise ValueError("retained_mass must be in (0,1].")

    if own_e_member_index is None:
        own_e_member_index = torch.arange(environment_count, device=applied_em.device, dtype=torch.long)
        own_e_member_index = own_e_member_index[None, :].expand(batch_size, -1)
    if (tuple(own_e_member_index.shape) != (batch_size, environment_count)
            or own_e_member_index.dtype not in (torch.int32, torch.int64)
            or own_e_member_index.device != applied_em.device):
        raise ValueError("own_e_member_index must be integer [B,E] on the relation device.")
    if environment_count:
        valid_anchor = environment_mask & (environment_measure > 0)
        anchor_index = own_e_member_index
        if bool(((anchor_index < 0) & valid_anchor).any()) or bool(
                ((anchor_index >= environment_count) & valid_anchor).any()):
            raise ValueError("Every valid E receiver must map to a present E self member.")
        safe_anchor = anchor_index.clamp(0, max(environment_count - 1, 0))
        mapped_valid = environment_mask.gather(1, safe_anchor) & (environment_measure.gather(1, safe_anchor) > 0)
        if bool((valid_anchor & ~mapped_valid).any()):
            raise ValueError("A valid E receiver's structural self must have positive measure.")

    total_members = module_count + environment_count
    member_measure = torch.cat((module_measure, environment_measure), dim=-1)
    member_valid = torch.cat((module_mask, environment_mask), dim=-1)
    member_measure = member_measure * member_valid.to(member_measure.dtype)
    member_ids = torch.cat((module_ids, environment_ids), dim=-1)
    is_module = torch.arange(total_members, device=applied_em.device) < module_count
    exchange = torch.cat((applied_em, applied_ee), dim=-1)
    receiver_valid = environment_mask & (environment_measure > 0)
    exchange = exchange * member_valid[:, None, :].to(exchange.dtype)
    exchange = exchange * receiver_valid[:, :, None].to(exchange.dtype)
    self_access = torch.zeros_like(exchange)
    if environment_count:
        rows = torch.arange(environment_count, device=applied_em.device)[None, :].expand(batch_size, -1)
        batches = torch.arange(batch_size, device=applied_em.device)[:, None].expand_as(rows)
        destinations = module_count + own_e_member_index.clamp(0, max(environment_count - 1, 0))
        valid_self = (environment_mask & (environment_measure > 0)).to(exchange.dtype)
        self_access[batches, rows, destinations] = valid_self
    unnormalized = (exchange + self_access) * member_measure[:, None, :]
    row_total = unnormalized.sum(-1, keepdim=True)
    tiny = torch.finfo(unnormalized.dtype).tiny
    access = torch.where(row_total > 0, unnormalized / row_total.clamp_min(tiny),
                         torch.zeros_like(unnormalized))

    # Stable secondary order is role then original ID. The primary order is
    # descending live access mass; only the discrete permutation is detached.
    tie_orders: list[Tensor] = []
    for batch_index in range(batch_size):
        ids_cpu = member_ids[batch_index].detach().cpu().tolist()
        tie_indices = sorted(range(total_members), key=lambda idx: (0 if idx < module_count else 1, ids_cpu[idx]))
        tie_orders.append(torch.tensor(tie_indices, device=applied_em.device, dtype=torch.long))

    own_global = module_count + own_e_member_index.to(torch.long)
    if batch_size and environment_count and total_members:
        tie_order = torch.stack(tie_orders, dim=0)
        tie_order_rows = tie_order[:, None, :].expand(batch_size, environment_count, total_members)
        tied_weights = access.gather(-1, tie_order_rows)
        order_in_ties = torch.argsort(tied_weights.detach(), dim=-1, descending=True, stable=True)
        sorted_indices = tie_order_rows.gather(-1, order_in_ties)
        sorted_weights = access.gather(-1, sorted_indices)
        full_counts = (sorted_weights.detach() > 0).sum(-1)
        cumulative = sorted_weights.detach().cumsum(-1)
        target = cumulative[..., -1:] * float(retained_mass)
        prefix_counts = torch.searchsorted(cumulative.contiguous(), target.contiguous(), right=False).squeeze(-1) + 1
        prefix_counts = torch.minimum(prefix_counts, full_counts)
        positions = torch.arange(total_members, device=applied_em.device)[None, None, :]
        own_in_prefix = ((sorted_indices == own_global[..., None]) &
                         (positions < prefix_counts[..., None])).any(-1)
    else:
        sorted_indices = torch.empty((batch_size, environment_count, 0),
                                     device=applied_em.device, dtype=torch.long)
        full_counts = torch.zeros((batch_size, environment_count),
                                  device=applied_em.device, dtype=torch.long)
        prefix_counts = full_counts
        own_in_prefix = torch.zeros((batch_size, environment_count),
                                    device=applied_em.device, dtype=torch.bool)

    # Convert the ragged choices to row-major masks, then gather all columns in
    # two batched passes. The final restricted column is reserved for structural
    # self when it was outside the positive joint-95 prefix.
    positions = torch.arange(total_members, device=applied_em.device)[None, None, :]
    full_selected = positions < full_counts[..., None]
    restricted_indices = torch.cat((sorted_indices, own_global[..., None]), dim=-1)
    own_appended = receiver_valid & ~own_in_prefix
    restricted_selected = torch.cat((positions < prefix_counts[..., None], own_appended[..., None]), dim=-1)

    return EAccessDirectory(
        full=_pack_masked_rows(
            sorted_indices, full_selected, access, exchange, self_access,
            member_measure, member_ids, is_module,
        ),
        restricted=_pack_masked_rows(
            restricted_indices, restricted_selected, access, exchange, self_access,
            member_measure, member_ids, is_module,
        ),
        own_e_member_index=own_e_member_index,
        module_count=module_count,
        environment_count=environment_count,
        retained_mass=float(retained_mass),
    )


def anchored_row_read(
    raw_scores: Tensor,
    access_prior: Tensor,
    candidate: Tensor,
    values: Tensor | None,
    normalized_query: Tensor,
    typed_gate: nn.Module,
    scene_eligible_count: Tensor | int,
    query_mask: Tensor,
    *,
    score_calibration: str = "inherited",
    reduce_values: bool = True,
) -> AnchoredRoleRead:
    """Read one role from one E row using one final p for stats and values.

    ``raw_scores`` is gathered once from the shared query/member union and has
    shape ``[Q,S]``. ``access_prior`` and ``candidate`` describe this one
    compiled row and role; values are the corresponding original records.
    The typed gate receives the usual query, p-weighted calibrated score mean,
    and entropy normalized by the full scene eligible count.
    """
    if raw_scores.ndim != 2:
        raise ValueError("raw_scores must have [Q,S] axes for one anchor row.")
    queries, members = raw_scores.shape
    if access_prior.shape != (members,) or candidate.shape != (members,) or candidate.dtype != torch.bool:
        raise ValueError("access_prior and boolean candidate must match one row's member axis.")
    if not access_prior.is_floating_point() or access_prior.device != raw_scores.device:
        raise ValueError("access_prior must be floating point on the score device.")
    if not bool(torch.isfinite(access_prior).all()) or bool((access_prior < 0).any()):
        raise ValueError("access_prior must be finite and nonnegative.")
    if reduce_values:
        if values is None or values.ndim not in (2, 3) or values.shape[-2] != members:
            raise ValueError("values must have [S,D] or [Q,S,D] axes when reducing a row context.")
        if values.ndim == 3 and values.shape[0] != queries:
            raise ValueError("Per-query values must match the raw score query count.")
        if values.device != raw_scores.device or values.dtype != raw_scores.dtype:
            raise ValueError("Values and raw scores must share device and dtype.")
    elif values is not None and (values.device != raw_scores.device or values.dtype != raw_scores.dtype):
        raise ValueError("Optional values and raw scores must share device and dtype.")
    if normalized_query.ndim != 2 or normalized_query.shape[0] != queries:
        raise ValueError("normalized_query must have [Q,Dq] axes.")
    if normalized_query.device != raw_scores.device or normalized_query.dtype != raw_scores.dtype:
        raise ValueError("normalized_query and raw scores must share device and dtype.")
    if query_mask.shape != (queries,) or query_mask.dtype != torch.bool or query_mask.device != raw_scores.device:
        raise ValueError("query_mask must be boolean [Q] on the score device.")
    if score_calibration not in ("inherited", "bounded_tanh"):
        raise ValueError("score_calibration must be 'inherited' or 'bounded_tanh'.")
    if isinstance(scene_eligible_count, Tensor):
        if scene_eligible_count.shape not in ((), (queries,)) or scene_eligible_count.device != raw_scores.device:
            raise ValueError("scene_eligible_count must be scalar or [Q] on the score device.")
        eligible_count = scene_eligible_count.to(raw_scores)
    else:
        eligible_count = raw_scores.new_full((queries,), int(scene_eligible_count))
    if bool((eligible_count < 0).any()) or not bool(torch.isfinite(eligible_count).all()):
        raise ValueError("scene_eligible_count must be finite and nonnegative.")
    if eligible_count.ndim == 0:
        eligible_count = eligible_count.expand(queries)

    candidate = candidate & (access_prior > 0)
    candidate_rows = candidate[None, :].expand(queries, members)
    prior_rows = access_prior[None, :].expand(queries, members)
    if score_calibration == "inherited":
        calibrated = prior_weighted_scores(raw_scores, prior_rows, candidate_rows)
    else:
        calibrated = 4.0 * torch.tanh(raw_scores / 4.0) * candidate_rows.to(raw_scores.dtype)

    if members:
        valid = candidate_rows & query_mask[:, None]
        log_prior = torch.where(valid, prior_rows.clamp_min(torch.finfo(raw_scores.dtype).tiny).log(),
                                torch.zeros_like(raw_scores))
        log_weight = torch.where(valid, log_prior + calibrated,
                                 torch.full_like(raw_scores, -torch.inf))
        has_members = valid.any(-1, keepdim=True)
        maximum = log_weight.amax(-1, keepdim=True)
        maximum = torch.where(has_members, maximum, torch.zeros_like(maximum))
        shifted = torch.where(valid, log_weight - maximum, torch.zeros_like(log_weight))
        unnormalized = torch.exp(shifted).clamp_min(torch.finfo(raw_scores.dtype).tiny)
        unnormalized = unnormalized * valid.to(raw_scores.dtype)
        normalizer = unnormalized.sum(-1, keepdim=True)
        probabilities = torch.where(normalizer > 0,
                                    unnormalized / normalizer.clamp_min(torch.finfo(raw_scores.dtype).tiny),
                                    torch.zeros_like(unnormalized))
    else:
        has_members = torch.zeros((queries, 1), dtype=torch.bool, device=raw_scores.device)
        probabilities = raw_scores.new_zeros((queries, 0))
    expectation = (probabilities * calibrated).sum(-1)
    tiny = torch.finfo(raw_scores.dtype).tiny
    entropy = -(probabilities * probabilities.clamp_min(tiny).log()).sum(-1)
    normalized_entropy = torch.where(
        eligible_count > 1,
        entropy / eligible_count.clamp_min(2.0).log(),
        torch.zeros_like(entropy),
    )
    gate_inputs = torch.cat((normalized_query, expectation[:, None], normalized_entropy[:, None]), dim=-1)
    gate_logits = typed_gate(gate_inputs)
    if gate_logits.shape != (queries, 2):
        raise ValueError("typed_gate must return two logits per query.")
    contrast = gate_logits[:, 0] - gate_logits[:, 1]
    strength = read_probability(contrast) * has_members[:, 0].to(raw_scores.dtype)
    strength = strength * query_mask.to(strength.dtype)

    if reduce_values:
        if values.ndim == 2:
            value_read = torch.einsum("qs,sd->qd", probabilities, values)
        else:
            value_read = torch.einsum("qs,qsd->qd", probabilities, values)
        context = value_read * strength[:, None]
    else:
        value_read = None
        context = None
    return AnchoredRoleRead(
        probabilities=probabilities,
        calibrated_scores=calibrated,
        expectation=expectation,
        normalized_entropy=normalized_entropy,
        gate_logits=gate_logits,
        strength=strength,
        context_before_strength=value_read,
        context=context,
    )


def blend_anchored_contexts(contexts: Tensor, anchor_weights: Tensor) -> Tensor:
    """Blend completed row contexts with the adapter's unnormalized λ sum."""
    if contexts.ndim != 4 or anchor_weights.shape != contexts.shape[:3]:
        raise ValueError("contexts and anchor_weights must have [B,Q,K,D] and [B,Q,K] axes.")
    if not (contexts.is_floating_point() and anchor_weights.is_floating_point()):
        raise TypeError("Contexts and anchor weights must be floating point.")
    if contexts.device != anchor_weights.device or contexts.dtype != anchor_weights.dtype:
        raise ValueError("Contexts and anchor weights must share device and dtype.")
    if not bool(torch.isfinite(contexts).all()) or not bool(torch.isfinite(anchor_weights).all()):
        raise ValueError("Contexts and anchor weights must be finite.")
    if bool((anchor_weights < 0).any()):
        raise ValueError("Anchor weights must be nonnegative.")
    return (contexts * anchor_weights[..., None]).sum(dim=2)


@dataclass(frozen=True)
class _AnchorReadGroup:
    batch_index: int
    anchor_index: int
    query_indices: Tensor
    anchor_slots: Tensor
    anchor_weights: Tensor
    row: EAccessRow


class AnchoredMemberRoutingCore(SingleDistributionMemberRoutingCore):
    """R13-8 A/L core with actual E exchange rows governing member reads."""

    schema = "interaction_reused_e_anchored_member_routing_v1"
    _ANCHOR_COUNT_KEYS = (
        "query_unique_union_raw_score_pairs_M", "query_unique_union_raw_score_pairs_E",
        "query_anchor_member_pairs_M", "query_anchor_member_pairs_E",
        "query_anchor_normalization_rows_M", "query_anchor_normalization_rows_E",
        "query_anchor_typed_gate_rows_M", "query_anchor_typed_gate_rows_E",
        "query_anchor_coalesced_value_pairs_M", "query_anchor_coalesced_value_pairs_E",
        "query_anchor_value_reduction_pairs_M", "query_anchor_value_reduction_pairs_E",
        "query_M_member_query_affine_rows", "query_E_member_query_affine_rows",
        "query_joint_field_head_rows",
    )
    _additional_decode_count_keys = tuple(getattr(
        InteractionReusedMemberRoutingCore, "_additional_decode_count_keys", (),
    )) + _ANCHOR_COUNT_KEYS

    def __init__(self, base_model_config: Mapping[str, Any], *, locator_mode: str = "e_anchor",
                 input_dataset: str = "native_records", query_tile_size: int = 256,
                 retained_mass: float = 0.95, score_calibration: str = "inherited"):
        if locator_mode not in ("e_anchor", "local_e_cover"):
            raise ValueError("locator_mode must be 'e_anchor' or 'local_e_cover'.")
        if (isinstance(retained_mass, bool) or not isinstance(retained_mass, (float, int))
                or not 0 < float(retained_mass) <= 1):
            raise ValueError("retained_mass must be in (0,1].")
        if score_calibration not in ("inherited", "bounded_tanh"):
            raise ValueError("score_calibration must be 'inherited' or 'bounded_tanh'.")
        self.locator_mode = locator_mode
        self.retained_mass = float(retained_mass)
        self.score_calibration = score_calibration
        super().__init__(base_model_config, address_mode="ME", input_dataset=input_dataset,
                         query_tile_size=query_tile_size, retained_prior_law="none")
        self.score_calibration = score_calibration
        self.joint_member_coverage = self.retained_mass
        # Initialization used the exact U parameter inventory and name seeds;
        # remove the learned entrance scorer from the actual A/L state dict.
        if hasattr(self.member_router, "proposal_M"):
            del self.member_router.proposal_M
        self.architecture = self._anchored_arm_name(locator_mode)

    @staticmethod
    def _anchored_arm_name(locator_mode: str) -> str:
        return "R13-8-A-EAnchor" if locator_mode == "e_anchor" else "R13-8-L-LocalECover"

    @property
    def architecture_name(self) -> str:
        return self._anchored_arm_name(self.locator_mode)

    @property
    def active_parameter_inventory(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.named_parameters())

    def get_config(self) -> dict[str, Any]:
        base = InteractionReusedMemberRoutingCore.get_config(self)
        return {
            "schema": self.schema,
            "architecture": self.architecture_name,
            "locator_mode": self.locator_mode,
            "base_model_config": base["base_model_config"],
            "input_dataset": self.input_dataset,
            "input_capability": self.input_capability,
            "query_tile_size": self.query_tile_size,
            "retained_mass": self.retained_mass,
            "score_calibration": self.score_calibration,
        }

    @classmethod
    def validate_config(cls, config: Mapping[str, Any]) -> None:
        required = {"schema", "architecture", "locator_mode", "base_model_config", "input_dataset",
                    "input_capability", "query_tile_size", "retained_mass", "score_calibration"}
        if set(config) != required or config.get("schema") != cls.schema:
            raise ValueError("R13-8 anchored routing configuration inventory/schema differs.")
        locator_mode = config["locator_mode"]
        if (locator_mode not in ("e_anchor", "local_e_cover")
                or config["architecture"] != cls._anchored_arm_name(locator_mode)):
            raise ValueError("R13-8 architecture and locator_mode disagree.")
        if config["input_capability"] != InteractionReusedMemberRoutingCore.input_capability:
            raise ValueError("R13-8 input capability differs from the joint-record core.")
        if not isinstance(config["input_dataset"], str) or not config["input_dataset"]:
            raise ValueError("R13-8 input dataset identity is required.")
        if type(config["query_tile_size"]) is not int or config["query_tile_size"] < 1:
            raise ValueError("R13-8 query tile size must be positive.")
        if (isinstance(config["retained_mass"], bool) or not isinstance(config["retained_mass"], (int, float))
                or not 0 < float(config["retained_mass"]) <= 1):
            raise ValueError("R13-8 retained mass must be in (0,1].")
        if config["score_calibration"] not in ("inherited", "bounded_tanh"):
            raise ValueError("R13-8 score calibration is unsupported.")
        base_config = {
            "schema": InteractionReusedMemberRoutingCore.schema,
            "architecture": InteractionReusedMemberRoutingCore._arm_name("route"),
            "member_routing_mode": "route",
            "base_model_config": config["base_model_config"],
            "input_dataset": config["input_dataset"],
            "input_capability": config["input_capability"],
            "entmax_temperature": 1.0,
            "query_tile_size": config["query_tile_size"],
        }
        InteractionReusedMemberRoutingCore.validate_config(base_config)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]):
        cls.validate_config(config)
        return cls(config["base_model_config"], locator_mode=config["locator_mode"],
                   input_dataset=config["input_dataset"], query_tile_size=config["query_tile_size"],
                   retained_mass=config["retained_mass"], score_calibration=config["score_calibration"])

    @staticmethod
    def _directory_versions(directory: EAccessDirectory) -> tuple[int, ...]:
        tensors = []
        for rows in (directory.full, directory.restricted):
            tensors.extend((rows.offsets, rows.member_index, rows.original_id, rows.is_module,
                            rows.access_weight, rows.applied_exchange, rows.structural_self, rows.measure))
        tensors.append(directory.own_e_member_index)
        return tuple(_tensor_version(value) for value in tensors)

    def prepare(self, scene, **options):
        """Prepare the shared encoder and only member scorer key terms."""
        prepared = InteractionReusedMemberRoutingCore.prepare(self, scene, **options)
        prepared._anchored_normalized_member_keys_M = self._routing_features(prepared.member_keys_M)
        prepared._anchored_normalized_member_keys_E = self._routing_features(prepared.member_keys_E)
        width = self.context_dim
        specs = (("member_M_M", self.member_router.member_M, "M"),
                 ("member_E_E", self.member_router.member_E, "E"))
        prepared._anchored_affine_key_terms = {}
        affine_versions = []
        for name, scorer, role in specs:
            keys = (prepared._anchored_normalized_member_keys_M if role == "M"
                    else prepared._anchored_normalized_member_keys_E)
            term = F.linear(keys, scorer[0].weight[:, width:2 * width])
            prepared._anchored_affine_key_terms[name] = term
            affine_versions.append((name, _tensor_version(term), _tensor_version(scorer[0].weight)))
        prepared._anchored_affine_key_versions = tuple(affine_versions)
        prepared._anchored_normalized_key_versions = (
            _tensor_version(prepared.member_keys_M), _tensor_version(prepared.member_keys_E),
            _tensor_version(prepared._anchored_normalized_member_keys_M),
            _tensor_version(prepared._anchored_normalized_member_keys_E),
        )
        directory = getattr(prepared, "e_access_directory", None)
        if not isinstance(directory, EAccessDirectory):
            raise ValueError("Prepared anchored scene lacks its compiled E access directory.")
        prepared._anchored_directory_versions = self._directory_versions(directory)
        prepared._anchored_exchange_versions = (
            _tensor_version(prepared.anchored_applied_EM),
            _tensor_version(prepared.anchored_applied_EE),
        )
        prepared.work_counts.update({
            "normalized_member_key_rows_M": int(scene.batch_size * scene.num_modules),
            "normalized_member_key_rows_E": int(scene.batch_size * scene.num_environment_tokens),
            "prepared_e_directory_rows": int(scene.batch_size * scene.num_environment_tokens),
            "prepared_e_directory_full_entries": int(directory.full.member_index.numel()),
            "prepared_e_directory_joint95_entries": int(directory.restricted.member_index.numel()),
            "prepared_e_directory_sort_entries": int(
                scene.batch_size * scene.num_environment_tokens *
                (scene.num_modules + scene.num_environment_tokens)
            ),
        })
        return prepared

    def _finalize_native_preparation(self, organization, cycle_trace):
        """Capture applied final-cycle E←M/E←E and compile the live directory."""
        organization = InteractionReusedMemberRoutingCore._finalize_native_preparation(
            self, organization, cycle_trace,
        )
        scene = organization.scene
        cycle_trace = {} if cycle_trace is None else cycle_trace
        applied_em = self._safe_snapshot_gate(cycle_trace, "em", scene)
        applied_ee = self._safe_snapshot_gate(cycle_trace, "ee", scene)
        snapshots = {route: cycle_trace.get(route + "_native_route_snapshot") for route in ("em", "ee")}
        expected = {"em": (scene.environment_ids, scene.module_ids),
                    "ee": (scene.environment_ids, scene.environment_ids)}
        cycles = set()
        for route, snapshot in snapshots.items():
            if snapshot is None:
                raise ValueError(f"Final contextual cycle lacks the actual {route.upper()} route snapshot.")
            stage = str(getattr(snapshot, "stage_identity", ""))
            if getattr(snapshot, "route", None) != route or not stage.endswith("/" + route):
                raise ValueError(f"Captured {route.upper()} provenance does not match its applied route.")
            if (not torch.equal(snapshot.destination_ids, expected[route][0])
                    or not torch.equal(snapshot.donor_ids, expected[route][1])):
                raise ValueError(f"Captured {route.upper()} IDs do not preserve original record identity.")
            cycles.add(stage.rsplit("/", 1)[0])
        if len(cycles) != 1:
            raise ValueError("Anchored EM and EE must come from one actual final contextual cycle.")
        organization.anchored_applied_EM = applied_em
        organization.anchored_applied_EE = applied_ee
        organization.anchored_route_provenance = {
            route: {"stage_identity": snapshots[route].stage_identity, "route": route,
                    "receiver_role": "E", "donor_role": "M" if route == "em" else "E",
                    "destination_ids": snapshots[route].destination_ids,
                    "donor_ids": snapshots[route].donor_ids}
            for route in ("em", "ee")
        }
        member_measure = build_joint_member_measure(
            scene.module_mask, scene.environment_mask, scene.environment_measure,
            dtype=applied_em.dtype,
        )
        directory = compile_e_access_directory(
            applied_em, applied_ee, member_measure[:, :scene.num_modules],
            member_measure[:, scene.num_modules:],
            scene.module_mask, scene.environment_mask, scene.module_ids, scene.environment_ids,
            retained_mass=self.retained_mass,
        )
        organization.e_access_directory = directory
        organization.work_counts.update({
            "prepared_e_directory_compile_pair_rows": int(applied_em.numel() + applied_ee.numel()),
            "prepared_e_directory_positive_exchange_entries": int((directory.full.applied_exchange > 0).sum()),
            "prepared_e_directory_total_access_entries": int(directory.full.member_index.numel()),
        })
        return organization

    def _validate_prepared_state(self, organization):
        InteractionReusedMemberRoutingCore._validate_prepared_state(self, organization)
        normalized_m = getattr(organization, "_anchored_normalized_member_keys_M", None)
        normalized_e = getattr(organization, "_anchored_normalized_member_keys_E", None)
        if normalized_m is None or normalized_e is None:
            raise ValueError("Prepared anchored scene lacks normalized member keys; rebuild it.")
        current = (_tensor_version(organization.member_keys_M), _tensor_version(organization.member_keys_E),
                   _tensor_version(normalized_m), _tensor_version(normalized_e))
        if current != organization._anchored_normalized_key_versions:
            raise ValueError("Prepared anchored member-key views changed; rebuild the scene.")
        affine = getattr(organization, "_anchored_affine_key_terms", None)
        if not isinstance(affine, dict) or set(affine) != {"member_M_M", "member_E_E"}:
            raise ValueError("Prepared anchored scene lacks member-only scorer terms.")
        current_affine = tuple(
            (name, _tensor_version(affine[name]), _tensor_version(scorer[0].weight))
            for name, scorer in (("member_M_M", self.member_router.member_M),
                                 ("member_E_E", self.member_router.member_E))
        )
        if current_affine != organization._anchored_affine_key_versions:
            raise ValueError("Prepared anchored scorer terms changed; rebuild the scene.")
        directory = getattr(organization, "e_access_directory", None)
        if (not isinstance(directory, EAccessDirectory)
                or self._directory_versions(directory) != organization._anchored_directory_versions):
            raise ValueError("Prepared E access directory changed; rebuild the scene.")
        exchange_versions = (
            _tensor_version(organization.anchored_applied_EM),
            _tensor_version(organization.anchored_applied_EE),
        )
        if exchange_versions != organization._anchored_exchange_versions:
            raise ValueError("Prepared applied E exchange blocks changed; rebuild the scene.")

    def validate_prepared_state(self, organization, query):
        self._validate_prepared_state(organization)
        self._validate_query(organization, query)
        self._validate_anchor_carrier(organization.scene, query)

    @staticmethod
    def _directory_trace(prepared) -> dict[str, Tensor]:
        """Expose compiled access columns separately from their applied source blocks."""
        directory = prepared.e_access_directory
        result = {
            "e_access_own_e_member_index": directory.own_e_member_index,
            "e_access_applied_EM": prepared.anchored_applied_EM,
            "e_access_applied_EE": prepared.anchored_applied_EE,
        }
        for label, rows in (("full", directory.full), ("restricted", directory.restricted)):
            for column in ("offsets", "member_index", "original_id", "is_module", "access_weight",
                           "applied_exchange", "structural_self", "measure"):
                result[f"e_access_{label}_{column}"] = getattr(rows, column)
        for role in ("em", "ee"):
            provenance = prepared.anchored_route_provenance[role]
            result[f"e_access_{role}_destination_ids"] = provenance["destination_ids"]
            result[f"e_access_{role}_donor_ids"] = provenance["donor_ids"]
        return result

    def _validate_anchor_carrier(self, scene, query) -> tuple[Tensor, Tensor]:
        indices = getattr(query, "anchor_indices", None)
        weights = getattr(query, "anchor_weights", None)
        if not isinstance(indices, Tensor) or not isinstance(weights, Tensor):
            raise ValueError("R13-8 queries require adapter-supplied anchor_indices and anchor_weights.")
        if indices.ndim != 3 or weights.shape != indices.shape or indices.shape[:2] != (
                query.batch_size, query.num_queries):
            raise ValueError("Anchor indices and weights must have matching [B,Q,K] axes.")
        if indices.dtype not in (torch.int32, torch.int64) or not weights.is_floating_point():
            raise TypeError("Anchor indices must be integers and anchor weights floating point.")
        if indices.device != query.coordinates.device or weights.device != query.coordinates.device:
            raise ValueError("Anchor carrier and query coordinates must share a device.")
        if weights.dtype != query.coordinates.dtype:
            raise ValueError("Anchor weights and query coordinates must share a floating dtype.")
        slots = indices.shape[-1]
        if (slots < 1 or (self.locator_mode == "e_anchor" and slots != 1)
                or (self.locator_mode == "local_e_cover" and slots > 4)):
            raise ValueError("The E locator must supply one A anchor or at most four L anchors.")
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("Anchor weights must be finite and nonnegative.")
        query_mask = self._query_mask(query)
        positive = weights > 0
        if scene.num_environment_tokens == 0 and bool((positive & query_mask[..., None]).any()):
            raise ValueError("A query cannot select an E anchor from an empty E inventory.")
        if scene.num_environment_tokens and bool(
                (((indices < 0) | (indices >= scene.num_environment_tokens)) & positive).any()):
            raise ValueError("A positive anchor weight references an absent E slot.")
        if scene.num_environment_tokens:
            safe_indices = indices.clamp(0, scene.num_environment_tokens - 1).to(torch.long)
            selected_present = scene.environment_mask.gather(1, safe_indices.reshape(scene.batch_size, -1))
            selected_measure = scene.environment_measure.gather(1, safe_indices.reshape(scene.batch_size, -1))
            selected_present = selected_present.reshape_as(positive)
            selected_measure = selected_measure.reshape_as(weights)
            if bool((positive & ~(selected_present & (selected_measure > 0))).any()):
                raise ValueError("A positive anchor weight references an absent or zero-measure E slot.")
        if bool((positive & ~query_mask[..., None]).any()):
            raise ValueError("Masked queries cannot carry positive E anchor weights.")
        sums = weights.sum(-1)
        if bool((query_mask & ((sums - 1.0).abs() > 1.0e-5)).any()):
            raise ValueError("Every valid query's E anchor weights must sum to one.")
        if bool((~query_mask & (sums.abs() > 1.0e-7)).any()):
            raise ValueError("Masked queries must have zero E anchor weight.")
        for index in range(slots - 1):
            duplicate = ((indices[..., index, None] == indices[..., index + 1:])
                         & positive[..., index, None] & positive[..., index + 1:])
            if bool(duplicate.any()):
                raise ValueError("Positive E anchor slots must be deduplicated by the adapter.")
        return indices, weights


    @staticmethod
    def _segment_sum(values, indices, count):
        return values.new_zeros((count,) + values.shape[1:]).index_add(0, indices, values)

    def _packed_memberships(self, prepared, query, table, intervention):
        """Expand only located directory rows, retaining live selected priors."""
        anchors, weights = query.anchor_indices, query.anchor_weights
        batch, queries, slots = anchors.shape
        active = (weights > 0) & self._query_mask(query)[..., None]
        reads = active.nonzero(as_tuple=False)
        b, q, k = reads.unbind(-1)
        a = anchors[b, q, k]
        offsets = table.offsets.to(device=anchors.device)
        starts, lengths = offsets[b, a], offsets[b, a + 1] - offsets[b, a]
        read_id = torch.repeat_interleave(torch.arange(len(reads), device=anchors.device), lengths)
        previous = lengths.cumsum(0) - lengths
        local = torch.arange(read_id.numel(), device=anchors.device) - previous[read_id]
        entries = starts[read_id] + local
        members = table.member_index[entries]
        prior = table.access_weight[entries]
        original = table.original_id[entries]
        is_module = table.is_module[entries]
        structural = table.structural_self[entries]
        control = intervention.get("access_intervention", intervention.get("directory_intervention"))
        if control is not None:
            control = ({**intervention, "kind": control} if isinstance(control, str) else dict(control))
            kind = control.get("kind", control.get("mode"))
            keep = torch.ones_like(members, dtype=torch.bool)
            if kind == "self_only":
                own = prepared.e_access_directory.own_e_member_index[b[read_id], a[read_id]]
                keep = members == (prepared.scene.num_modules + own)
            elif kind in ("remove_member", "reassign_member"):
                role = control.get("source_role", control.get("member_role", control.get("role")))
                identity = control.get("source_id", control.get("member_id", control.get("original_id")))
                if role not in ("M", "E") or identity is None:
                    raise ValueError("Access interventions require source_role and source_id.")
                hit = (is_module == (role == "M")) & (original == int(identity)) & (structural == 0)
                if control.get("anchor_id") is not None:
                    hit = hit & (prepared.scene.environment_ids[b[read_id], a[read_id]] == int(control["anchor_id"]))
                if kind == "remove_member":
                    keep = ~hit
                else:
                    target_role, target_id = control.get("target_role", role), control.get("target_id")
                    if target_role not in ("M", "E") or target_id is None:
                        raise ValueError("Reassignment requires target_role and target_id.")
                    ids = prepared.scene.module_ids if target_role == "M" else prepared.scene.environment_ids
                    mask = prepared.scene.module_mask if target_role == "M" else prepared.scene.environment_mask
                    if target_role == "E":
                        mask = mask & (prepared.scene.environment_measure > 0)
                    matches = (ids == int(target_id)) & mask
                    if not bool((matches.sum(-1) == 1).all()):
                        raise ValueError("Reassignment target must identify one present original member per scene.")
                    target_slot = matches.to(torch.long).argmax(-1)
                    if target_role == "E":
                        target_slot = target_slot + prepared.scene.num_modules
                    members = torch.where(hit, target_slot[b[read_id]], members)
                    original = torch.where(hit, original.new_full((), int(target_id)), original)
                    is_module = torch.where(hit, torch.full_like(is_module, target_role == "M"), is_module)
            else:
                raise ValueError("Unknown access intervention.")
            read_id, entries, members, prior, original, is_module = (
                value[keep] for value in (read_id, entries, members, prior, original, is_module))
        total = prepared.scene.num_modules + prepared.scene.num_environment_tokens
        pair_key = ((b[read_id] * queries + q[read_id]) * total + members)
        return {"reads": reads, "anchors": a, "weights": weights[b, q, k], "read_id": read_id,
                "entries": entries, "members": members, "prior": prior, "original": original,
                "is_module": is_module, "pair_key": pair_key}

    def _raw_union_scores(self, prepared, query, qstates, keys):
        """Evaluate the unchanged factored scorer exactly once per union pair."""
        total = prepared.scene.num_modules + prepared.scene.num_environment_tokens
        query_index = torch.div(keys, total, rounding_mode="floor")
        member = keys.remainder(total)
        b = torch.div(query_index, query.num_queries, rounding_mode="floor")
        q = query_index.remainder(query.num_queries)
        module = member < prepared.scene.num_modules
        qnorm = self._routing_features(qstates)
        raw = qstates.new_zeros(keys.shape)
        counts = {}
        for role, use in (("M", module), ("E", ~module)):
            positions = use.nonzero(as_tuple=False).flatten()
            rb, rq, rm = b[positions], q[positions], member[positions]
            if role == "E":
                rm = rm - prepared.scene.num_modules
            scorer = self.member_router.member_M if role == "M" else self.member_router.member_E
            query_term = F.linear(qnorm, scorer[0].weight[:, :self.context_dim], scorer[0].bias)
            key_term = prepared._anchored_affine_key_terms[f"member_{role}_{role}"]
            rows = torch.stack((rb, rq, rm), dim=-1)
            geometry = self._pair_geometry(prepared.scene, query, role, rows)
            hidden = query_term[rb, rq] + key_term[rb, rm]
            hidden = hidden + F.linear(geometry, scorer[0].weight[:, 2 * self.context_dim:])
            scores = scorer[2](scorer[1](hidden)).squeeze(-1)
            raw = raw.index_copy(0, positions, scores)
            counts[f"query_unique_union_raw_score_pairs_{role}"] = positions.numel()
            counts[f"query_{role}_member_query_affine_rows"] = query.batch_size * query.num_queries
        return raw, qnorm, counts

    def _read_packed(self, prepared, query, qstates, qnorm, packed, union_keys, raw, *, return_trace):
        """Calibrate and normalize each located row/role, then coalesce values."""
        reads, rid = packed["reads"], packed["read_id"]
        read_count, groups = len(reads), 2 * len(reads)
        group = rid * 2 + (~packed["is_module"]).to(torch.long)
        union_index = torch.searchsorted(union_keys, packed["pair_key"])
        scores, prior = raw[union_index], packed["prior"]
        tiny = torch.finfo(scores.dtype).tiny
        mass = self._segment_sum(prior, group, groups)
        conditional = prior / mass[group].clamp_min(tiny)
        means = self._segment_sum(conditional * scores, group, groups)
        centered = scores - means[group]
        variance = self._segment_sum(conditional * centered.square(), group, groups)
        ell = 4. * torch.tanh(centered / (4. * torch.sqrt(variance[group] + 1.)))
        if self.score_calibration == "bounded_tanh":
            ell = 4. * torch.tanh(scores / 4.)
        log_weight = prior.clamp_min(tiny).log() + ell
        maximum = scores.new_full((groups,), -torch.inf).scatter_reduce(
            0, group, log_weight, reduce="amax", include_self=True)
        numerator = torch.exp(log_weight - maximum[group]).clamp_min(tiny)
        normalizer = self._segment_sum(numerator, group, groups)
        probabilities = numerator / normalizer[group].clamp_min(tiny)
        expectation = self._segment_sum(probabilities * ell, group, groups).reshape(read_count, 2)
        entropy = self._segment_sum(-probabilities * probabilities.clamp_min(tiny).log(), group, groups)
        entropy = entropy.reshape(read_count, 2)
        b, q, k = reads.unbind(-1)
        scene_count = torch.stack((prepared.scene.module_mask.sum(-1),
                                  (prepared.scene.environment_mask & (prepared.scene.environment_measure > 0)).sum(-1)), -1)
        eligible = scene_count[b].to(scores)
        entropy = torch.where(eligible > 1, entropy / eligible.clamp_min(2).log(), torch.zeros_like(entropy))
        strength_parts, gate_parts = [], []
        for role_index, role in enumerate(("M", "E")):
            gate_input = torch.cat((qnorm[b, q], expectation[:, role_index, None], entropy[:, role_index, None]), -1)
            logits = self.typed_read_gates[role](gate_input)
            strength_parts.append(read_probability(logits[:, 0] - logits[:, 1]) * (mass.reshape(read_count, 2)[:, role_index] > 0))
            gate_parts.append(logits)
        strength = torch.stack(strength_parts, -1)
        beta = packed["weights"][rid] * strength.reshape(-1)[group] * probabilities
        coalesced = self._segment_sum(beta, union_index, len(union_keys))
        total = prepared.scene.num_modules + prepared.scene.num_environment_tokens
        uq = torch.div(union_keys, total, rounding_mode="floor")
        um = union_keys.remainder(total)
        ub = torch.div(uq, query.num_queries, rounding_mode="floor")
        contexts, counts = {}, {}
        for role_index, role in enumerate(("M", "E")):
            selected = ((um < prepared.scene.num_modules) if role == "M" else (um >= prepared.scene.num_modules))
            # Paired restricted reads share paid full scores, but reduce values
            # only for their own actual positive member coefficients.
            selected = selected & (coalesced > 0)
            positions = selected.nonzero(as_tuple=False).flatten()
            slot = um[positions] - (prepared.scene.num_modules if role == "E" else 0)
            values = prepared.member_values_M if role == "M" else prepared.member_values_E
            contributions = values[ub[positions], slot] * coalesced[positions, None]
            context = self._segment_sum(contributions, uq[positions], query.batch_size * query.num_queries)
            contexts[role] = context.reshape(query.batch_size, query.num_queries, self.context_dim)
            memberships = packed["is_module"] if role == "M" else ~packed["is_module"]
            counts[f"query_anchor_member_pairs_{role}"] = memberships.sum().detach()
            counts[f"query_anchor_normalization_rows_{role}"] = read_count
            counts[f"query_anchor_typed_gate_rows_{role}"] = read_count
            counts[f"query_anchor_coalesced_value_pairs_{role}"] = positions.numel()
            counts[f"query_anchor_value_reduction_pairs_{role}"] = positions.numel()
        fused = torch.cat((qstates, contexts["M"], contexts["E"]), -1)
        fields = self.member_router.joint_head(self.member_router.joint_trunk(fused))
        fields = fields * self._query_mask(query)[..., None].to(fields)
        counts["query_joint_field_head_rows"] = query.batch_size * query.num_queries
        trace = None
        if return_trace:
            trace = {"anchor_indices": query.anchor_indices, "anchor_weights": query.anchor_weights,
                     "anchor_read_indices": reads, "anchor_read_E_slots": packed["anchors"],
                     "anchor_read_member_read_index": rid, "anchor_read_member_index": packed["members"],
                     "anchor_read_original_id": packed["original"], "anchor_read_is_module": packed["is_module"],
                     "anchor_read_access_prior": prior, "anchor_read_probabilities": probabilities,
                     "anchor_read_calibrated_scores": ell, "anchor_read_strength": strength,
                     "anchor_read_coalesced_union_keys": union_keys,
                     "anchor_read_coalesced_member_coefficients": coalesced,
                     "original_module_ids": prepared.scene.module_ids,
                     "original_environment_ids": prepared.scene.environment_ids,
                     "anchored_applied_EM": prepared.anchored_applied_EM,
                     "anchored_applied_EE": prepared.anchored_applied_EE,
                     "context_M": contexts["M"], "context_E": contexts["E"], "joint_fields": fields}
            trace.update(self._directory_trace(prepared))
            for role_index, role in enumerate(("M", "E")):
                rho = fields.new_zeros(query.batch_size, query.num_queries, query.anchor_indices.shape[-1])
                rho = rho.index_put((b, q, k), strength[:, role_index])
                trace[f"anchor_read_strength_{role}"] = rho
                trace[f"member_read_strength_{role}"] = (rho * query.anchor_weights).sum(-1)
                role_mask = um < prepared.scene.num_modules if role == "M" else um >= prepared.scene.num_modules
                positions = role_mask.nonzero(as_tuple=False).flatten()
                trace[f"member_score_union_indices_{role}"] = union_keys[positions]
                trace[f"member_score_union_values_{role}"] = raw[positions]
                trace[f"anchor_read_coalesced_member_coefficients_{role}"] = coalesced[positions]
                n = prepared.scene.num_modules if role == "M" else prepared.scene.num_environment_tokens
                slot = um[positions] - (prepared.scene.num_modules if role == "E" else 0)
                member_weights = fields.new_zeros(query.batch_size * query.num_queries, n)
                member_weights = member_weights.index_put((uq[positions], slot), coalesced[positions])
                trace[f"member_weights_{role}"] = member_weights.reshape(query.batch_size, query.num_queries, n)
                trace[f"candidate_mask_{role}"] = trace[f"member_weights_{role}"] > 0
        return fields, trace, counts

    def _decode_tile(self, prepared, query, read_modes, *, return_trace=False, intervention=None):
        controls = {} if intervention is None else dict(intervention)
        if set(controls) - {"read_mode", "access_intervention", "directory_intervention", "member_role",
                            "member_id", "source_role", "source_id", "target_role", "target_id", "anchor_id"}:
            raise ValueError("Unsupported anchored read intervention.")
        qstates = self._query_states(prepared, query)
        directory = prepared.e_access_directory
        packed = {mode: self._packed_memberships(prepared, query, getattr(directory, mode), controls)
                  for mode in read_modes}
        union_keys = torch.unique(torch.cat([row["pair_key"] for row in packed.values()]), sorted=True)
        raw, qnorm, shared = self._raw_union_scores(prepared, query, qstates, union_keys)
        outputs = {}
        for mode in read_modes:
            fields, trace, counts = self._read_packed(prepared, query, qstates, qnorm, packed[mode],
                                                     union_keys, raw, return_trace=return_trace)
            outputs[mode] = (fields, trace, counts)
        return outputs, shared

    def _decode_requests(self, prepared, query, modes, *, intervention=None, return_trace=False):
        self.validate_prepared_state(prepared, query)
        if return_trace and query.num_queries > self.query_tile_size:
            raise ValueError("Anchored traces require an explicit query slice within query_tile_size.")
        fields = {mode: [] for mode in modes}
        counts = {mode: {} for mode in modes}
        shared_counts, traces = {}, {}
        count_keys = self._ANCHOR_COUNT_KEYS
        for start in range(0, query.num_queries, self.query_tile_size):
            tile = slice_query_v1(query, start, min(start + self.query_tile_size, query.num_queries))
            if self.training and torch.is_grad_enabled() and not return_trace:
                def execute(coordinates, anchor_weights, tile=tile):
                    local = replace(tile, coordinates=coordinates, anchor_weights=anchor_weights)
                    result, shared = self._decode_tile(prepared, local, modes, intervention=intervention)
                    tensors = tuple(result[mode][0] for mode in modes)
                    packed_counts = tuple(_pack_work_counts(result[mode][2], count_keys, coordinates) for mode in modes)
                    return tensors + packed_counts + (_pack_work_counts(shared, count_keys, coordinates),)
                result = checkpoint(execute, tile.coordinates, tile.anchor_weights, use_reentrant=False)
                output_fields = result[:len(modes)]
                local_counts = result[len(modes):2 * len(modes)]
                local_shared = result[-1]
            else:
                result, shared = self._decode_tile(prepared, tile, modes, return_trace=return_trace, intervention=intervention)
                output_fields = tuple(result[mode][0] for mode in modes)
                local_counts = tuple(_pack_work_counts(result[mode][2], count_keys, tile.coordinates) for mode in modes)
                local_shared = _pack_work_counts(shared, count_keys, tile.coordinates)
                if return_trace:
                    traces = {mode: result[mode][1] for mode in modes}
            for mode, output, local_count in zip(modes, output_fields, local_counts):
                fields[mode].append(output)
                counts[mode] = local_count.detach() if not isinstance(counts[mode], Tensor) else counts[mode] + local_count.detach()
            shared_counts = local_shared.detach() if not isinstance(shared_counts, Tensor) else shared_counts + local_shared.detach()
        def unpack(value):
            return ({name: int(number) for name, number in zip(count_keys, value.cpu().tolist())}
                    if isinstance(value, Tensor) else {})
        outputs = {}
        for mode in modes:
            output = torch.cat(fields[mode], 1) if fields[mode] else query.coordinates.new_zeros(query.batch_size, 0, self.output_dim)
            outputs[mode] = InteractionJointOutput(output, traces.get(mode), unpack(counts[mode]))
        return outputs, unpack(shared_counts)

    def decode(self, organization, query, controls=None, *, intervention=None, return_trace=False,
               trace_request=None, **kwargs):
        if controls is not None or any(value is not None and value is not False for value in kwargs.values()):
            raise ValueError("Anchored joint prediction accepts no historical affine controls.")
        intervention = {} if intervention is None else dict(intervention)
        mode = intervention.get("read_mode", self._read_execution_override or "full")
        if mode not in ("full", "restricted"):
            raise ValueError("Anchored read mode must be full or restricted.")
        outputs, shared = self._decode_requests(organization, query, (mode,), intervention=intervention,
                                                return_trace=return_trace or trace_request is not None)
        output = outputs[mode]
        for key, value in shared.items():
            output.work_counts[key] = output.work_counts.get(key, 0) + value
        return output

    def decode_paired(self, organization, query, *, intervention=None, return_trace=False, trace_request=None):
        outputs, shared = self._decode_requests(organization, query, ("full", "restricted"),
                                                intervention=intervention,
                                                return_trace=return_trace or trace_request is not None)
        return outputs["full"], outputs["restricted"], shared


__all__ = [
    "AnchoredMemberRoutingCore",
    "AnchoredRoleRead",
    "EAccessDirectory",
    "EAccessRow",
    "PackedEAccessRows",
    "anchored_row_read",
    "blend_anchored_contexts",
    "compile_e_access_directory",
]

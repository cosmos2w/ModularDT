"""Native-token hypergraph construction from captured MM and ME routes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .contracts import tensor_versions
from .route_capture import NativeRouteSnapshot


def _positive_threshold(value: float, name: str) -> float:
    threshold = float(value)
    if not math.isfinite(threshold) or threshold < 0.0 or threshold >= 0.5:
        raise ValueError(f"{name} must be finite and lie in [0,0.5).")
    return threshold


def _s5(value: Tensor) -> Tensor:
    t = value.clamp(0.0, 1.0)
    return (t.pow(3) * (10.0 + t * (-15.0 + 6.0 * t))).clamp(0.0, 1.0)


def relative_density(raw_scores: Tensor, allowed: Tensor) -> Tensor:
    """Return exp(score - row max), zero outside each route's eligible set."""
    if raw_scores.ndim != 4:
        raise ValueError("raw_scores must have shape [B,H,D,S].")
    if allowed.shape != (raw_scores.shape[0], raw_scores.shape[2], raw_scores.shape[3]):
        raise ValueError("allowed must match [B,D,S] route axes.")
    if raw_scores.shape[-1] == 0:
        return torch.zeros_like(raw_scores)
    mask = allowed[:, None, :, :]
    row_has_donor = mask.any(dim=-1, keepdim=True)
    masked = raw_scores.masked_fill(~mask, -torch.inf)
    maximum = masked.amax(dim=-1, keepdim=True)
    maximum = torch.where(row_has_donor, maximum, torch.zeros_like(maximum))
    delta = torch.where(mask & row_has_donor, raw_scores - maximum, torch.zeros_like(raw_scores))
    return torch.exp(delta) * mask.to(raw_scores.dtype)


def density_permission(raw_scores: Tensor, allowed: Tensor, threshold: float) -> tuple[Tensor, Tensor]:
    """Map dimensionless row-relative density to an endpoint-flat permission."""
    tau = _positive_threshold(threshold, "threshold")
    relative = relative_density(raw_scores, allowed)
    if tau == 0.0:
        return relative, allowed[:, None, :, :].to(raw_scores.dtype).expand_as(raw_scores)
    permission = _s5((relative - tau) / tau)
    permission = permission * allowed[:, None, :, :].to(permission.dtype)
    return relative, permission


def measure_attention_probabilities(snapshot: NativeRouteSnapshot) -> Tensor:
    """Reconstruct measure-weighted route probabilities for inspection only."""
    if snapshot.raw_scores.shape[-1] == 0:
        return torch.zeros_like(snapshot.raw_scores)
    measure = snapshot.donor_measure.to(snapshot.raw_scores.dtype)
    valid = snapshot.donor_mask & (measure > 0)
    safe_measure = torch.where(valid, measure, torch.ones_like(measure))
    logits = snapshot.raw_scores + torch.log(safe_measure)[:, None, None, :]
    allowed = snapshot.allowed[:, None, :, :]
    row_has_donor = allowed.any(dim=-1, keepdim=True)
    logits = logits.masked_fill(~allowed, -torch.inf)
    logits = torch.where(row_has_donor, logits, torch.zeros_like(logits))
    weights = torch.softmax(logits, dim=-1) * allowed.to(logits.dtype)
    return torch.where(row_has_donor, weights, torch.zeros_like(weights))


@dataclass(frozen=True)
class NativeRetrievalHypergraph:
    """M actual module-seeded hyperedges over native M/E IDs.

    The graph contains scores and permissions only. It has no pooled token
    value, centroid, moment, edge embedding, or decoder payload.
    """

    mm_snapshot: NativeRouteSnapshot
    me_snapshot: NativeRouteSnapshot
    module_relative_density: Tensor  # [B,H,Mseed,M]
    environment_relative_density: Tensor  # [B,H,Mseed,E]
    module_member_permissions: Tensor  # [B,H,Mseed,M], retrieval self included
    environment_member_permissions: Tensor  # [B,H,Mseed,E]
    module_attention_probabilities: Tensor  # [B,H,Mseed,M], original context rule
    environment_attention_probabilities: Tensor  # [B,H,Mseed,E], original context rule
    module_context_allowed: Tensor  # bool [B,Mseed,M], MM diagonal remains excluded
    module_retrieval_allowed: Tensor  # bool [B,Mseed,M], adds seed self
    environment_allowed: Tensor  # bool [B,Mseed,E]
    tau_mm: float
    tau_me: float
    _derived_versions: tuple[tuple[int, int, int], ...] = field(default=(), init=False, repr=False)

    @property
    def seed_ids(self) -> Tensor:
        return self.mm_snapshot.destination_ids

    @property
    def module_ids(self) -> Tensor:
        return self.mm_snapshot.donor_ids

    @property
    def environment_ids(self) -> Tensor:
        return self.me_snapshot.donor_ids

    @property
    def seed_valid(self) -> Tensor:
        return self.mm_snapshot.destination_mask

    @property
    def module_valid(self) -> Tensor:
        return self.mm_snapshot.donor_mask

    @property
    def environment_valid(self) -> Tensor:
        return self.me_snapshot.donor_mask

    @property
    def module_measure(self) -> Tensor:
        return self.mm_snapshot.donor_measure

    @property
    def environment_measure(self) -> Tensor:
        return self.me_snapshot.donor_measure

    def __post_init__(self) -> None:
        object.__setattr__(self, "_derived_versions", tensor_versions(
            self.module_relative_density,
            self.environment_relative_density,
            self.module_member_permissions,
            self.environment_member_permissions,
            self.module_attention_probabilities,
            self.environment_attention_probabilities,
            self.module_context_allowed,
            self.module_retrieval_allowed,
            self.environment_allowed,
        ))

    def assert_current(
        self,
        *,
        owner_token: int | None = None,
        parameter_versions: tuple[tuple[int, int], ...] | None = None,
        buffer_versions: tuple[tuple[int, int], ...] | None = None,
        scene_identity: tuple[object, object, object] | None = None,
        scene_tensor_versions: tuple[tuple[int, int, int], ...] | None = None,
        model_config_key: tuple[object, ...] | None = None,
    ) -> None:
        """Reject a graph whose score source, members, or ownership has changed."""
        self.mm_snapshot.assert_current()
        self.me_snapshot.assert_current()
        current = tensor_versions(
            self.module_relative_density,
            self.environment_relative_density,
            self.module_member_permissions,
            self.environment_member_permissions,
            self.module_attention_probabilities,
            self.environment_attention_probabilities,
            self.module_context_allowed,
            self.module_retrieval_allowed,
            self.environment_allowed,
        )
        if current != self._derived_versions:
            raise ValueError("Native hypergraph memberships, probabilities, or masks changed in place.")
        for name, expected in (
            ("owner", owner_token),
            ("parameter versions", parameter_versions),
            ("buffer versions", buffer_versions),
            ("scene identity", scene_identity),
            ("scene tensor versions", scene_tensor_versions),
            ("model configuration", model_config_key),
        ):
            if expected is None:
                continue
            actual = {
                "owner": self.mm_snapshot.owner_token,
                "parameter versions": self.mm_snapshot.parameter_versions,
                "buffer versions": self.mm_snapshot.buffer_versions,
                "scene identity": self.mm_snapshot.scene_identity,
                "scene tensor versions": self.mm_snapshot.scene_tensor_versions,
                "model configuration": self.mm_snapshot.model_config_key,
            }[name]
            if actual != expected:
                raise ValueError(f"Native hypergraph {name} binding is stale.")


def build_native_retrieval_hypergraph(
    mm_snapshot: NativeRouteSnapshot,
    me_snapshot: NativeRouteSnapshot,
    *,
    tau_mm: float,
    tau_me: float,
) -> NativeRetrievalHypergraph:
    """Build overlapping seed memberships without changing contextualization.

    The MM contextualizer retains its diagonal exclusion. This graph adds each
    valid seed's own native module ID with permission one for retrieval only.
    """
    if mm_snapshot.route != "mm" or me_snapshot.route != "me":
        raise ValueError("Native hypergraph construction requires typed MM and ME snapshots.")
    tau_mm = _positive_threshold(tau_mm, "tau_mm")
    tau_me = _positive_threshold(tau_me, "tau_me")
    batch, heads, seeds, modules = mm_snapshot.raw_scores.shape
    if batch != me_snapshot.raw_scores.shape[0] or heads != me_snapshot.raw_scores.shape[1]:
        raise ValueError("MM and ME captures must share batch and preparation-head axes.")
    if me_snapshot.raw_scores.shape[2] != seeds:
        raise ValueError("MM and ME captures must have the same module-seed axis.")
    if mm_snapshot.destination_ids.shape != (batch, seeds) or mm_snapshot.donor_ids.shape != (batch, modules):
        raise ValueError("MM snapshot IDs do not match its route axes.")
    if not torch.equal(mm_snapshot.destination_ids, mm_snapshot.donor_ids):
        raise ValueError("MM destination and donor IDs must use the same native module ordering.")
    if not torch.equal(me_snapshot.destination_ids, mm_snapshot.destination_ids):
        raise ValueError("ME module destinations must match the MM native seed ordering.")
    if not torch.equal(me_snapshot.destination_mask, mm_snapshot.destination_mask):
        raise ValueError("MM and ME seed validity masks must match.")
    if mm_snapshot.stage_identity.split("/")[-1] != "mm" or me_snapshot.stage_identity.split("/")[-1] != "me":
        raise ValueError("Captured stage identities must identify the actual MM and ME routes.")
    if mm_snapshot.owner_token != me_snapshot.owner_token:
        raise ValueError("MM and ME snapshots must have the same model owner.")
    if mm_snapshot.parameter_versions != me_snapshot.parameter_versions:
        raise ValueError("MM and ME snapshots must share the same model parameter version binding.")
    if mm_snapshot.buffer_versions != me_snapshot.buffer_versions:
        raise ValueError("MM and ME snapshots must share the same model buffer version binding.")
    if mm_snapshot.scene_identity != me_snapshot.scene_identity:
        raise ValueError("MM and ME snapshots must share the same scene/frame/normalizer binding.")
    if mm_snapshot.scene_tensor_versions != me_snapshot.scene_tensor_versions:
        raise ValueError("MM and ME snapshots must share the same scene tensor version binding.")
    if mm_snapshot.model_config_key != me_snapshot.model_config_key:
        raise ValueError("MM and ME snapshots must share the same model configuration binding.")
    if mm_snapshot.measure_convention != me_snapshot.measure_convention:
        raise ValueError("MM and ME snapshots must share an explicit measure convention.")
    if mm_snapshot.stage_identity.rsplit("/", 1)[0] != me_snapshot.stage_identity.rsplit("/", 1)[0]:
        raise ValueError("MM and ME snapshots must come from the same preparation cycle.")
    mm_snapshot.assert_current()
    me_snapshot.assert_current()

    module_relative, module_permission = density_permission(
        mm_snapshot.raw_scores, mm_snapshot.allowed, tau_mm
    )
    environment_relative, environment_permission = density_permission(
        me_snapshot.raw_scores, me_snapshot.allowed, tau_me
    )
    seed_is_native_module = (
        mm_snapshot.destination_ids[:, :, None] == mm_snapshot.donor_ids[:, None, :]
    )
    seed_valid = mm_snapshot.destination_mask[:, :, None]
    valid_module = mm_snapshot.donor_mask[:, None, :]
    retrieval_self = seed_is_native_module & seed_valid & valid_module
    module_permission = torch.maximum(
        module_permission,
        retrieval_self[:, None, :, :].to(module_permission.dtype),
    )
    module_context_allowed = mm_snapshot.allowed
    module_retrieval_allowed = module_context_allowed | retrieval_self
    return NativeRetrievalHypergraph(
        mm_snapshot=mm_snapshot,
        me_snapshot=me_snapshot,
        module_relative_density=module_relative,
        environment_relative_density=environment_relative,
        module_member_permissions=module_permission,
        environment_member_permissions=environment_permission,
        module_attention_probabilities=measure_attention_probabilities(mm_snapshot),
        environment_attention_probabilities=measure_attention_probabilities(me_snapshot),
        module_context_allowed=module_context_allowed,
        module_retrieval_allowed=module_retrieval_allowed,
        environment_allowed=me_snapshot.allowed,
        tau_mm=tau_mm,
        tau_me=tau_me,
    )


def build_native_hypergraph(
    mm_snapshot: NativeRouteSnapshot,
    me_snapshot: NativeRouteSnapshot,
    *,
    tau_mm: float,
    tau_me: float,
) -> NativeRetrievalHypergraph:
    """Short public spelling used by the native core integration."""
    return build_native_retrieval_hypergraph(
        mm_snapshot, me_snapshot, tau_mm=tau_mm, tau_me=tau_me
    )


__all__ = [
    "NativeRetrievalHypergraph",
    "build_native_hypergraph",
    "build_native_retrieval_hypergraph",
    "density_permission",
    "measure_attention_probabilities",
    "relative_density",
]

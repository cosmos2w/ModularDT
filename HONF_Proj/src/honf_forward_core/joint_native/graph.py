"""All-native typed seed directory and exact unique-row support algebra."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..native_retrieval.contracts import _require
from .contracts import JointNativeDirectory, JointReadPlan, TypedNativeRecords, CanonicalGeometry
from .routing import (ScoreCalibration, JointDispatchParameters, DirectGateParameters,
                      binary_sample, canonical_winner, direct_typed_gates, dispatch_joint_native_seed)


class JointGraphParameters(nn.Module):
    def __init__(self, heads: int, *, dtype=torch.float32):
        super().__init__()
        self.routes = nn.ModuleDict({name: ScoreCalibration(heads, dtype=dtype)
                                     for name in ("mm", "me", "em", "ee")})


def calibrate_bias(logits_without_bias: Tensor, eligible_nonmandatory: Tensor,
                   target=.98, *, iterations=64) -> Tensor:
    """One input-only birth calibration; callers restrict data to fixed TRAIN batches."""
    if not 0 < target < 1 or iterations < 1:
        raise ValueError("Calibration target must lie in (0,1) and iterations must be positive.")
    values = logits_without_bias[eligible_nonmandatory].detach()
    if not values.numel():
        raise ValueError("No nonmandatory eligible pairs; record unsupported calibration explicitly.")
    _require(torch.isfinite(values), "Calibration scores must be finite.")
    span = values.abs().amax() + 32
    lower, upper = -span, span
    for _ in range(iterations):
        midpoint = (lower + upper) / 2
        lower = torch.where((values + midpoint).sigmoid().mean() < target, midpoint, lower)
        upper = torch.where((values + midpoint).sigmoid().mean() >= target, midpoint, upper)
    return (lower + upper) / 2


def _validated_raw(capture, records, name, destination, donor):
    snapshot = capture[name]
    if isinstance(snapshot, Tensor):
        # Tensor fixtures are an explicit data-free primitive boundary. Live
        # preparation uses owner-bound snapshots and validates them below.
        return snapshot
    snapshot.assert_current()
    if snapshot.route != name or snapshot.stage_identity.rsplit("/", 1)[-1] != name:
        raise ValueError("Typed native capture route mismatch.")
    if not torch.equal(snapshot.destination_ids, records.ids[:, destination]) or not torch.equal(
            snapshot.donor_ids, records.ids[:, donor]):
        raise ValueError("Native capture IDs do not match typed directory correspondence.")
    if not torch.equal(snapshot.destination_mask, records.present[:, destination]) or not torch.equal(
            snapshot.donor_mask, records.present[:, donor]):
        raise ValueError("Native capture presence differs from the prepared scene.")
    return snapshot.raw_scores


def build_joint_native_graph(capture: dict, records: TypedNativeRecords, parameters: JointGraphParameters,
                             *, deterministic=False, straight_through=False, temperature=1.,
                             noise_scale=1., logistic_noise=None, generators=None) -> JointNativeDirectory:
    """Sample once per scene, independently in each typed route namespace.

    This function is pure with respect to counters. Backward checkpointing must
    reuse the returned incidence/noise rather than rebuilding a directory.
    """
    records.assert_current()
    if set(capture) != {"mm", "me", "em", "ee"}:
        raise ValueError("A joint graph requires actual native MM/ME/EM/EE captures.")
    snapshots = [value for value in capture.values() if not isinstance(value, Tensor)]
    if snapshots:
        if len(snapshots) != 4:
            raise ValueError("Do not mix unbound raw tensors and bound route snapshots.")
        for field in ("owner_token", "parameter_versions", "buffer_versions", "scene_identity",
                      "scene_tensor_versions", "model_config_key"):
            if any(getattr(value, field) != getattr(snapshots[0], field) for value in snapshots[1:]):
                raise ValueError("Typed native snapshots must share owner, weights, scene and config.")
        cycles = {value.stage_identity.rsplit("/", 1)[0] for value in snapshots}
        if len(cycles) != 1:
            raise ValueError("Joint native snapshots must originate from the same contextual cycle.")
    m, n = records.module_count, records.ids.shape[1]
    batch = records.ids.shape[0]
    slices = {"m": slice(0, m), "e": slice(m, n)}
    blocks = {}
    for name in ("mm", "me", "em", "ee"):
        destination, donor = slices[name[0]], slices[name[1]]
        raw = _validated_raw(capture, records, name, destination, donor)
        dest_present = records.present[:, destination]
        donor_eligible = records.eligible[:, donor]
        allowed = dest_present[:, :, None] & donor_eligible[:, None, :]
        self_floor = torch.zeros_like(allowed)
        if name[0] == name[1]:
            count = allowed.shape[-1]
            self_floor = torch.eye(count, device=allowed.device, dtype=torch.bool)[None] & allowed
            # MM's contextual scores exclude self; retrieval restores it below.
            score_allowed = allowed & ~self_floor if name == "mm" else allowed
        else:
            score_allowed = allowed
        logits = parameters.routes[name](raw, score_allowed)
        cross_floor = torch.zeros_like(allowed)
        if name[0] != name[1]:
            geometry = CanonicalGeometry.build(records.geometry.keys[:, donor])
            winner, _, unresolved = canonical_winner(logits, allowed, geometry)
            _require(~unresolved, "Indistinguishable mandatory cross-type donor ties are indeterminate.")
            if allowed.shape[-1]:
                cross_floor.scatter_(2, winner.clamp_min(0)[..., None], (winner >= 0)[..., None])
        mandatory = self_floor | cross_floor
        hard, gate, probability = binary_sample(logits, allowed, mandatory, deterministic=deterministic,
            straight_through=straight_through, temperature=temperature, noise_scale=noise_scale,
            logistic_noise=None if logistic_noise is None else logistic_noise[name],
            generator=None if generators is None else generators[name])
        blocks[name] = (logits, probability, hard, gate, self_floor, cross_floor)
    assembled = []
    for index in range(6):
        assembled.append(torch.cat((torch.cat((blocks["mm"][index], blocks["me"][index]), -1),
                                    torch.cat((blocks["em"][index], blocks["ee"][index]), -1)), 1))
    return JointNativeDirectory(records, *assembled)


@torch.no_grad()
def calibrate_graph_birth(directories: list[JointNativeDirectory], parameters: JointGraphParameters,
                          target=.98) -> dict:
    """Calibrate each route once over up to eight fixed TRAIN input batches.

    The caller owns the TRAIN/provenance boundary. Returned diagnostics include
    unsupported empty/nonmandatory routes explicitly, rather than claiming that
    an unmeasured default was calibrated.
    """
    if not 1 <= len(directories) <= 8:
        raise ValueError("Birth calibration requires one to eight fixed TRAIN batches.")
    result = {}
    for name in ("mm", "me", "em", "ee"):
        values = []
        for directory in directories:
            directory.assert_current()
            m = directory.records.module_count
            ds = slice(None,m) if name[0] == "m" else slice(m,None)
            ss = slice(None,m) if name[1] == "m" else slice(m,None)
            allowed = directory.records.present[:,ds,None] & directory.records.eligible[:,None,ss]
            nonmandatory = allowed & ~directory.mandatory_membership[:,ds,ss]
            values.append((directory.membership_logits[:,ds,ss]-parameters.routes[name].bias)[nonmandatory])
        scores = torch.cat(values)
        if not scores.numel():
            result[name] = {"status":"no_nonmandatory_pairs", "pairs":0}
            continue
        bias = calibrate_bias(scores,torch.ones_like(scores,dtype=torch.bool),target)
        parameters.routes[name].bias.copy_(bias)
        result[name] = {"status":"calibrated", "pairs":scores.numel(), "bias":float(bias),
                        "mean_probability":float((scores+bias).sigmoid().mean())}
    return result


def final_support(dispatch_seed: Tensor, directory: JointNativeDirectory, protected_mask: Tensor,
                  *, query_mask=None, mandatory_only=False) -> Tensor:
    directory.assert_current()
    batch, queries = dispatch_seed.shape
    n = directory.records.ids.shape[1]
    if protected_mask.shape != (batch, queries, n) or protected_mask.dtype != torch.bool:
        raise ValueError("Protections must be boolean [B,Q,M+E].")
    if query_mask is None:
        query_mask = torch.ones_like(dispatch_seed, dtype=torch.bool)
    _require(~protected_mask | directory.records.eligible[:, None], "Protections require eligible records.")
    _require((dispatch_seed >= -1) & (dispatch_seed < n), "Invalid joint native seed index.")
    if not n:
        return protected_mask & query_mask[..., None]
    valid = (dispatch_seed >= 0) & query_mask
    bi = torch.arange(batch, device=dispatch_seed.device)[:, None]
    _require(~valid | directory.records.present[bi, dispatch_seed.clamp_min(0)], "Dispatched seed must be present.")
    membership = directory.mandatory_membership if mandatory_only else directory.membership_hard
    selected = membership[bi, dispatch_seed.clamp_min(0)] & valid[..., None]
    return (selected | protected_mask) & query_mask[..., None]


def dense_st_gate(dispatch_gate: Tensor, membership_gate: Tensor, protected_mask: Tensor,
                  *, query_mask=None) -> Tensor:
    """Multiplicative hard-zero gates retain omitted-member/router derivatives."""
    gate = torch.bmm(dispatch_gate, membership_gate)
    gate = torch.where(protected_mask, torch.ones_like(gate), gate)
    return gate if query_mask is None else gate * query_mask[..., None]


class JointNativeRouter(nn.Module):
    """Narrow scene-build / pre-E receiver-route interface shared by H and S."""

    def __init__(self, heads: int, mode="hypergraph", *, dtype=torch.float32):
        super().__init__()
        if mode not in ("hypergraph", "direct"):
            raise ValueError("Joint native mode must be hypergraph or direct.")
        self.mode = mode
        if mode == "hypergraph":
            self.graph_parameters = JointGraphParameters(heads, dtype=dtype)
            self.dispatch_parameters = JointDispatchParameters(heads, dtype=dtype)
        else:
            self.direct_parameters = DirectGateParameters(heads, dtype=dtype)

    def build(self, capture, scene, **options):
        records = TypedNativeRecords.from_scene(scene)
        if self.mode == "direct":
            return records
        return build_joint_native_graph(capture, records, self.graph_parameters, **options)

    def route(self, directory, raw_M, raw_E, protected_M, protected_E, *, query_mask=None,
              deterministic=False, straight_through=False, temperature=1., noise_scale=1.,
              gumbel_noise=None, direct_logistic_noise=None, generator=None, generators=None,
              query_ids=None):
        # query_ids label an externally keyed RNG draw; they never enter logits.
        protected = torch.cat((protected_M, protected_E), -1)
        common = dict(query_mask=query_mask, deterministic=deterministic,
                      straight_through=straight_through, temperature=temperature, noise_scale=noise_scale)
        if self.mode == "direct":
            return direct_typed_gates(raw_M, raw_E, directory, self.direct_parameters, protected,
                                     logistic_noise=direct_logistic_noise, generators=generators, **common)
        dispatch = dispatch_joint_native_seed(raw_M, raw_E, directory.records, self.dispatch_parameters,
                                              gumbel_noise=gumbel_noise, generator=generator, **common)
        _require(~dispatch.tie_indeterminate, "Indistinguishable native seed maxima are indeterminate.")
        support = final_support(dispatch.dispatch_seed, directory, protected, query_mask=query_mask)
        floor = final_support(dispatch.dispatch_seed, directory, protected, query_mask=query_mask, mandatory_only=True)
        gates = dense_st_gate(dispatch.straight_through, directory.membership_straight_through,
                              protected, query_mask=query_mask)
        return JointReadPlan(support, gates, floor, protected, directory.records.module_count, dispatch)

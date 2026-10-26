"""Compact TRAIN-only health observations for the maintained R13-3 provider."""
from __future__ import annotations

import copy
import math
import random
import time
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import torch


HEALTH_SCHEMA = "thermal_member_routing_health_v1"
HEALTH_STATE_SCHEMA = "thermal_member_routing_health_state_v1"
NATIVE_ROLE_NAMES = (
    "flow/u", "flow/v", "flow/p", "flow/omega", "temperature/fluid",
    "temperature/surface", "temperature/material", "q_proxy",
)
TYPED_PATHS = ("M", "E")


def _tensor_norm(value: torch.Tensor | None) -> float:
    if value is None or not torch.is_tensor(value) or not value.numel():
        return 0.0
    result = float(value.detach().double().square().sum().sqrt().cpu())
    if not math.isfinite(result):
        raise FloatingPointError("Member-routing health observed a nonfinite tensor norm.")
    return result


def _scalar(value: torch.Tensor | float | int) -> float:
    result = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
    if not math.isfinite(result):
        raise FloatingPointError("Member-routing health observed a nonfinite scalar.")
    return result


def _masked_rms(value: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    value = value.detach().double()
    valid = torch.isfinite(value)
    if mask is not None:
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        valid = valid & mask.to(device=value.device, dtype=torch.bool).expand_as(value)
    selected = value[valid]
    return 0.0 if not selected.numel() else _scalar(selected.square().mean().sqrt())


def _support_summary(weights: torch.Tensor | None) -> dict[str, float | int]:
    if weights is None:
        return {"positive_pairs": 0, "queries_with_support": 0, "query_count": 0,
                "mean_positive_per_query": 0.0}
    positive = weights.detach() > 0
    counts = positive.sum(-1)
    return {
        "positive_pairs": int(positive.sum().item()),
        "queries_with_support": int((counts > 0).sum().item()),
        "query_count": int(counts.numel()),
        "mean_positive_per_query": 0.0 if not counts.numel() else _scalar(counts.float().mean()),
    }


def _support_with_ids(weights: torch.Tensor | None, original_ids: torch.Tensor | None) -> dict[str, Any]:
    summary = _support_summary(weights)
    if weights is None or original_ids is None:
        summary["original_ids_with_support"] = []
        return summary
    supported = (weights.detach() > 0).any(dim=1)
    ids = original_ids.detach()
    summary["original_ids_with_support"] = [
        int(value) if float(value).is_integer() else float(value)
        for value in ids[supported].reshape(-1).cpu().tolist()
    ]
    return summary


def _score_summary(scores: torch.Tensor | None, allowed: torch.Tensor | None,
                   null_weights: torch.Tensor | None) -> dict[str, Any]:
    if scores is None or allowed is None:
        return {"available": False}
    scores = scores.detach().double()
    allowed = allowed.detach().to(device=scores.device, dtype=torch.bool)
    raw: list[torch.Tensor] = []
    centered: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    zero_null_margins: list[torch.Tensor] = []
    for row, mask in zip(scores.reshape(-1, scores.shape[-1]),
                         allowed.expand_as(scores).reshape(-1, scores.shape[-1])):
        values = row[mask]
        if not values.numel():
            continue
        raw.append(values)
        centered.append(values - values.mean())
        ordered = values.sort(descending=True).values
        zero_null_margins.append(ordered[0])
        if ordered.numel() > 1:
            runner_up = ordered[1]
        elif null_weights is not None:
            runner_up = ordered.new_zeros(())
        else:
            continue
        margins.append(ordered[0] - runner_up)

    def describe(values):
        if not values:
            return {"count": 0, "mean": None, "min": None, "max": None, "rms": None}
        packed = torch.cat(values)
        return {"count": int(packed.numel()), "mean": _scalar(packed.mean()),
                "min": _scalar(packed.min()), "max": _scalar(packed.max()),
                "rms": _scalar(packed.square().mean().sqrt())}

    margin_values = torch.stack(margins) if margins else None
    return {
        "available": True,
        "raw": describe(raw),
        "centered_per_query": describe(centered),
        "top_score_minus_zero_reference": describe([torch.stack(zero_null_margins)] if zero_null_margins else []),
        "top_margin_over_runner_up_or_null": (
            {"count": 0, "mean": None, "min": None, "max": None}
            if margin_values is None else {
                "count": int(margin_values.numel()), "mean": _scalar(margin_values.mean()),
                "min": _scalar(margin_values.min()), "max": _scalar(margin_values.max()),
            }
        ),
    }


def _read_vector(prepared, trace: Mapping[str, torch.Tensor], path: str) -> torch.Tensor:
    suffix = "M" if path == "M" else "E"
    weights = trace.get(f"member_weights_{suffix}")
    values = getattr(prepared.organization, f"member_values_{suffix}", None)
    if weights is None or values is None:
        return prepared.query_batch.coordinates.new_zeros((prepared.query_batch.batch_size,
                                                            prepared.query_batch.num_queries, 0))
    return torch.einsum("bqn,bnd->bqd", weights, values)


def _native_field_delta(base: Mapping[str, torch.Tensor], other: Mapping[str, torch.Tensor],
                        model, prepared) -> dict[str, float]:
    field_std = model.field_std.to(device=base["pred_field"].device, dtype=base["pred_field"].dtype)
    output: dict[str, float] = {}
    fluid_mask = prepared.stencils["fluid"].valid
    for channel, role in enumerate(("flow/u", "flow/v", "flow/p", "flow/omega", "temperature/fluid")):
        difference = (other["pred_field"][..., channel] - base["pred_field"][..., channel]) / field_std[channel].clamp_min(1e-12)
        output[role] = _masked_rms(difference, fluid_mask)

    present = prepared.source_present.to(torch.bool)
    surface_valid = prepared.stencils["surface"].valid.reshape(base["pred_interface"].shape[:-1])
    outside_valid = prepared.stencils["outside"].valid.reshape(base["pred_interface"].shape[:-1])
    interface_scale = model.interface_target_std.to(device=base["pred_interface"].device,
                                                     dtype=base["pred_interface"].dtype)
    surface_mask = present[..., None] & surface_valid
    q_mask = surface_mask & outside_valid
    surface_difference = ((other["pred_interface"][..., 0] - base["pred_interface"][..., 0])
                          / interface_scale[0].clamp_min(1e-12))
    q_difference = ((other["pred_interface"][..., 1] - base["pred_interface"][..., 1])
                    / interface_scale[1].clamp_min(1e-12))
    output["temperature/surface"] = _masked_rms(surface_difference, surface_mask)
    output["q_proxy"] = _masked_rms(q_difference, q_mask)

    material_valid = prepared.stencils["material"].valid.reshape(base["pred_internal_temperature"].shape[:-1])
    material_mask = present[..., None] & material_valid
    material_scale = model.internal_temperature_std.to(
        device=base["pred_internal_temperature"].device,
        dtype=base["pred_internal_temperature"].dtype,
    )[0]
    material_difference = ((other["pred_internal_temperature"][..., 0]
                            - base["pred_internal_temperature"][..., 0])
                           / material_scale.clamp_min(1e-12))
    output["temperature/material"] = _masked_rms(material_difference, material_mask)
    return output


def _score_targets(trace: Mapping[str, torch.Tensor], prepared) -> dict[str, tuple[torch.Tensor | None, torch.Tensor | None,
                                                                                 torch.Tensor | None]]:
    query = prepared.query_batch
    query_mask = (torch.ones(query.coordinates.shape[:2], dtype=torch.bool,
                             device=query.coordinates.device)
                  if query.query_mask is None else query.query_mask)
    trace_queries = trace["proposal_logits_M"].shape[1]
    query_mask = query_mask[:, :trace_queries]
    scene = prepared.organization.scene
    def address_log(role):
        value = trace.get(f"address_coefficients_{role}")
        return None if value is None else value.clamp_min(torch.finfo(value.dtype).tiny).log()

    def measure_log(role):
        value = getattr(prepared.organization, f"nu_{role}", None)
        shape = trace.get(f"candidate_mask_{role}")
        return (None if value is None or shape is None else
                value.clamp_min(torch.finfo(value.dtype).tiny).log()[:, None, :].expand_as(shape))
    proposal_allowed = scene.module_mask[:, None, :] & query_mask[..., None]
    return {
        "proposal_M": (trace.get("proposal_logits_M"), proposal_allowed, trace.get("proposal_null")),
        "proposal_M_effective": (trace.get("proposal_effective_scores_M"), proposal_allowed,
                                 trace.get("proposal_null")),
        "member_M_raw": (trace.get("member_logits_M"), trace.get("candidate_mask_M"), trace.get("member_null_M")),
        "member_M_centered": (trace.get("member_normalized_scores_M"), trace.get("candidate_mask_M"),
                              trace.get("member_null_M")),
        "member_M_effective": (trace.get("member_effective_scores_M"), trace.get("candidate_mask_M"), trace.get("member_null_M")),
        "member_M_log_address": (address_log("M"), trace.get("candidate_mask_M"), None),
        "member_M_log_measure": (measure_log("M"), trace.get("candidate_mask_M"), None),
        "member_E_raw": (trace.get("member_logits_E"), trace.get("candidate_mask_E"), trace.get("member_null_E")),
        "member_E_centered": (trace.get("member_normalized_scores_E"), trace.get("candidate_mask_E"),
                              trace.get("member_null_E")),
        "member_E_effective": (trace.get("member_effective_scores_E"), trace.get("candidate_mask_E"), trace.get("member_null_E")),
        "member_E_log_address": (address_log("E"), trace.get("candidate_mask_E"), None),
        "member_E_log_measure": (measure_log("E"), trace.get("candidate_mask_E"), None),
    }


def _parameter_groups(model) -> dict[str, tuple[torch.nn.Parameter, ...]]:
    core = getattr(model, "core", model)
    named = dict(core.named_parameters())
    group_names = {
        "contextual_interaction": lambda name: not name.startswith(("member_router.", "typed_read_gates.")),
        "proposal_score": lambda name: name.startswith("member_router.proposal_M."),
        "typed_member_scores": lambda name: name.startswith(("member_router.member_M.", "member_router.member_E.")),
        "typed_keys": lambda name: name.startswith(("member_router.key_M.", "member_router.key_E.")),
        "typed_values": lambda name: name.startswith(("member_router.value_M.", "member_router.value_E.")),
        "typed_read_gates": lambda name: name.startswith("typed_read_gates."),
        "joint_field_head": lambda name: name.startswith(("member_router.joint_trunk.", "member_router.joint_head.")),
    }
    return {group: tuple(parameter for name, parameter in named.items()
                         if select(name) and parameter.requires_grad)
            for group, select in group_names.items()}


def _gradient_observation(model, objective: torch.Tensor, prepared,
                          trace: Mapping[str, torch.Tensor]) -> tuple[dict[str, float], dict[str, float]]:
    groups = _parameter_groups(model)
    targets: list[torch.Tensor] = []
    target_groups: list[tuple[str, Any]] = []
    for group, parameters in groups.items():
        for parameter in parameters:
            targets.append(parameter)
            target_groups.append(("parameter", group))

    path_targets = {
        "proposal_score_M": trace.get("proposal_logits_M"),
        "score_M": trace.get("member_effective_scores_M"),
        "score_E": trace.get("member_effective_scores_E"),
        "B_M": getattr(prepared.organization, "B_M", None),
        "B_E": getattr(prepared.organization, "B_E", None),
        "key_M": getattr(prepared.organization, "member_keys_M", None),
        "key_E": getattr(prepared.organization, "member_keys_E", None),
        "value_M": getattr(prepared.organization, "member_values_M", None),
        "value_E": getattr(prepared.organization, "member_values_E", None),
    }
    for name, target in path_targets.items():
        if torch.is_tensor(target) and target.requires_grad:
            targets.append(target)
            target_groups.append(("live", name))

    group_norms = {name: 0.0 for name in groups}
    live_norms = {name: 0.0 for name in path_targets}
    if objective.requires_grad and targets:
        gradients = torch.autograd.grad(objective, tuple(targets), allow_unused=True, retain_graph=False)
        squared = {name: 0.0 for name in groups}
        live_squared = {name: 0.0 for name in path_targets}
        for gradient, (kind, name) in zip(gradients, target_groups):
            if gradient is None:
                continue
            norm_sq = float(gradient.detach().double().square().sum().cpu())
            if not math.isfinite(norm_sq):
                raise FloatingPointError("Member-routing health observed a nonfinite native gradient.")
            if kind == "parameter":
                squared[name] += norm_sq
            else:
                live_squared[name] += norm_sq
        group_norms = {name: math.sqrt(value) for name, value in squared.items()}
        live_norms = {name: math.sqrt(value) for name, value in live_squared.items()}
    return group_norms, live_norms


def _score_gradient_observation(objective: torch.Tensor,
                                trace: Mapping[str, torch.Tensor]) -> dict[str, float]:
    targets = {
        "proposal_score_M": trace.get("proposal_logits_M"),
        "score_M": trace.get("member_effective_scores_M"),
        "score_E": trace.get("member_effective_scores_E"),
        "read_strength_M": trace.get("member_read_strength_M"),
        "read_strength_E": trace.get("member_read_strength_E"),
    }
    live = [(name, value) for name, value in targets.items()
            if torch.is_tensor(value) and value.requires_grad]
    result = {name: 0.0 for name in targets}
    if objective.requires_grad and live:
        gradients = torch.autograd.grad(objective, tuple(value for _, value in live),
                                        allow_unused=True, retain_graph=False)
        for (name, _value), gradient in zip(live, gradients):
            result[name] = _tensor_norm(gradient)
    return result


def _native_with_first_tile_trace(model, prepared):
    """Keep native extraction on the full union and trace its first legal tile.

    The traced scores are the actual tensors used by that tile's prediction,
    so its native-loss gradients remain meaningful. Other tiles retain the
    normal no-trace decoder, including its training checkpoint behavior.
    """
    from honf_forward_core.content_hypergraph.query_chunks_v1 import slice_query_v1

    fields, counts, first_trace = [], {}, None
    query = prepared.query_batch
    size = int(model.core.query_tile_size)
    for start in range(0, query.num_queries, size):
        tile = slice_query_v1(query, start, min(start + size, query.num_queries))
        output = model.core.decode(prepared.organization, tile, return_trace=start == 0)
        fields.append(output.joint_fields)
        for name, value in output.work_counts.items():
            counts[name] = counts.get(name, 0) + int(value)
        if start == 0:
            first_trace = output.trace
    joint = torch.cat(fields, dim=1)
    native = model._extract_native_roles(prepared, joint[..., 3:4], joint[..., :3])
    return native, counts, first_trace


def _first_query_rows(trace, count):
    rows = int(trace["proposal_logits_M"].shape[1])
    return {name: (value[:, :count] if torch.is_tensor(value) and value.ndim >= 2
                   and value.shape[1] == rows and name not in {
                       "B_M", "B_E", "original_module_ids", "original_environment_ids"}
                   and not name.startswith(("selected_value_indices_", "B_M_edge", "B_E_edge")) else value)
            for name, value in trace.items()}


def _path_summary(case_records: list[dict[str, Any]], path: str) -> dict[str, Any]:
    key_names = (("score_M", "key_M", "value_M", "B_M") if path == "M"
                 else ("score_E", "key_E", "value_E", "B_E"))
    positive_pairs = sum(int(record["support"][f"final_{path}"]["positive_pairs"])
                         for record in case_records)
    norms = {name: max((float(record["live_gradient_norms"].get(name, 0.0))
                        for record in case_records), default=0.0)
             for name in key_names}
    return {
        "positive_weight_pairs": positive_pairs,
        "active_in_any_query": positive_pairs > 0,
        "semantic_gradient_norms_max_across_panel": norms,
        "semantic_gradients_exact_zero_across_panel": all(value == 0.0 for value in norms.values()),
        "typed_read_strength_l2_mean": (
            None if not case_records else float(np.mean([
                float(record["typed_read_strength_l2"][path]) for record in case_records
            ]))
        ),
        "typed_read_heat_sensitivity_l2_max": max((
            float(record["typed_read_heat_sensitivity_l2"][path]) for record in case_records
        ), default=0.0),
        "typed_read_repeatability_l2_max": max((
            float(record["typed_read_repeatability_l2"][path]) for record in case_records
        ), default=0.0),
    }


def evaluate_pause(initial: Mapping[str, Any] | None, previously_active: Mapping[str, bool],
                   current: Mapping[str, Any], panel_module_counts: tuple[int, ...] | list[int]) -> dict[str, Any] | None:
    """Return an evidence-based collapse reason; a single empty query never pauses."""
    records = current.get("cases", ())
    counts = {int(record["module_count"]) for record in records}
    low, high = min(panel_module_counts), max(panel_module_counts)
    if low not in counts or high not in counts:
        return None

    for path in TYPED_PATHS:
        path_info = current.get("paths", {}).get(path, {})
        if (bool(previously_active.get(path, False))
                and int(path_info.get("positive_weight_pairs", 0)) == 0
                and bool(path_info.get("semantic_gradients_exact_zero_across_panel", False))):
            return {"kind": "previously_active_typed_path_collapsed",
                    "path": path, "panel_module_counts": sorted(counts)}

    if initial is None:
        return None
    initial_by_count = {int(record["module_count"]): record for record in initial.get("cases", ())}
    current_by_count = {int(record["module_count"]): record for record in records}
    low_high = (low, high)
    if any(count not in initial_by_count or count not in current_by_count for count in low_high):
        return None

    def path_gradient_zero(record, path):
        gradients = record.get("live_gradient_norms", {})
        names = (("score_M", "key_M", "value_M", "B_M") if path == "M"
                 else ("score_E", "key_E", "value_E", "B_E"))
        return all(float(gradients.get(name, 0.0)) == 0.0 for name in names)

    def initial_signal(record, path):
        if int(record["support"][f"final_{path}"]["positive_pairs"]) > 0:
            return True
        names = (("score_M", "key_M", "value_M", "B_M") if path == "M"
                 else ("score_E", "key_E", "value_E", "B_E"))
        if any(float(record.get("live_gradient_norms", {}).get(name, 0.0)) > 0.0 for name in names):
            return True
        return (float(record["typed_read_heat_sensitivity_l2"][path])
                > float(record["typed_read_repeatability_l2"][path]))

    def current_silent(record, path):
        return (path_gradient_zero(record, path)
                and float(record["typed_read_heat_sensitivity_l2"][path])
                <= float(record["typed_read_repeatability_l2"][path]))

    both_paths_silent = all(
        all(initial_signal(initial_by_count[count], path) and current_silent(current_by_count[count], path)
            for count in low_high)
        for path in TYPED_PATHS
    )
    if both_paths_silent:
        return {"kind": "both_typed_paths_lost_native_gradient_and_scene_sensitivity",
                "panel_module_counts": list(low_high),
                "comparison": "each path is compared with its own initial observation and measured repeatability"}
    return None


class MemberRoutingHealth:
    """R13-3-specific fixed-TRAIN panel, native objective, and clean-stop evidence."""

    def __init__(self, provider, config: Mapping[str, Any]):
        if not isinstance(config, Mapping) or config.get("enabled") is not True:
            raise ValueError("Enabled member-routing health requires a mapping with enabled=true.")
        self.provider = provider
        self.observation_epochs = int(config.get("observation_epochs", 12))
        self.later_every = int(config.get("later_every", 10))
        self.panel_module_counts = tuple(int(value) for value in config.get("panel_module_counts", (1, 2, 5, 12)))
        self.pause_on_collapse = bool(config.get("pause_on_collapse", True))
        if (self.observation_epochs < 1 or self.later_every < 1 or not self.panel_module_counts
                or any(value < 1 for value in self.panel_module_counts)
                or len(set(self.panel_module_counts)) != len(self.panel_module_counts)):
            raise ValueError("Member-routing health observation cadence/panel is invalid.")
        self.panel_case_ids = self._select_panel_case_ids()
        self.state = {
            "schema": HEALTH_STATE_SCHEMA,
            "panel_case_ids": list(self.panel_case_ids),
            "initial_observation": None,
            "last_observation": None,
            "last_observed_epoch": 0,
            "previously_active_paths": {path: False for path in TYPED_PATHS},
            "clean_stop": None,
        }

    def _select_panel_case_ids(self) -> tuple[str, ...]:
        from ..data.development_split import development_case_ids

        fixed_train = set(map(str, development_case_ids(self.provider.manifest, "train")))
        admitted = {str(case["case_id"]): case for case in self.provider.training_cases}
        candidates: dict[int, list[str]] = {count: [] for count in self.panel_module_counts}
        for case_id in fixed_train & set(admitted):
            present = np.asarray(admitted[case_id]["structure"]["module_present"]) > 0.5
            count = int(present.sum())
            if count in candidates:
                candidates[count].append(case_id)
        missing = [count for count, values in candidates.items() if not values]
        if missing:
            raise ValueError(f"Fixed TRAIN health panel has no cases at module counts {missing}.")
        return tuple(min(candidates[count]) for count in self.panel_module_counts)

    @property
    def initial_observation(self):
        return self.state.get("initial_observation")

    def scheduled(self, epoch: int) -> bool:
        epoch = int(epoch)
        if epoch <= self.observation_epochs:
            return True
        return (epoch - self.observation_epochs) % self.later_every == 0

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if state is None:
            return
        if not isinstance(state, Mapping) or state.get("schema") != HEALTH_STATE_SCHEMA:
            raise ValueError("Saved member-routing health state has an unknown schema.")
        if list(state.get("panel_case_ids", ())) != list(self.panel_case_ids):
            raise ValueError("Saved member-routing health panel differs from fixed TRAIN selection.")
        self.state = copy.deepcopy(dict(state))

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.state)

    def record_initial(self, observation: dict[str, Any]) -> None:
        self.state["initial_observation"] = copy.deepcopy(observation)
        self.state["last_observation"] = copy.deepcopy(observation)
        self.state["last_observed_epoch"] = int(observation["training_epoch"])
        self.state["previously_active_paths"] = {
            path: bool(observation.get("paths", {}).get(path, {}).get("active_in_any_query", False))
            for path in TYPED_PATHS
        }

    def record_epoch(self, observation: dict[str, Any]) -> dict[str, Any] | None:
        previous = dict(self.state.get("previously_active_paths", {}))
        reason = evaluate_pause(self.initial_observation, previous, observation, self.panel_module_counts)
        for path in TYPED_PATHS:
            previous[path] = (bool(previous.get(path, False))
                              or bool(observation.get("paths", {}).get(path, {}).get("active_in_any_query", False)))
        self.state["previously_active_paths"] = previous
        self.state["last_observation"] = copy.deepcopy(observation)
        self.state["last_observed_epoch"] = int(observation["training_epoch"])
        return reason

    def _run_case(self, model, case_id: str) -> dict[str, Any]:
        from honf_runtime.unified_training import SamplingKey
        from .content_hypergraph_task import ThermalContentPredictions

        provider = self.provider
        if int(provider.primary_queries) != 1024:
            raise ValueError("Member-routing health uses the maintained fixed Q=1024 TRAIN sample.")
        key = SamplingKey(seed=int(provider.seed), epoch=1, update_index=0, microbatch_index=0,
                          stage="joint", arm=str(model.architecture),
                          sampling_version=provider.sampling_version,
                          dataset_id=provider.sampling_dataset_id)
        full_batch = provider.make_batch((case_id,), key)
        query_count = min(8, int(full_batch.receivers.fluid_xy.shape[1]))
        if query_count < 3:
            raise ValueError("Member-routing health requires at least three fixed TRAIN queries.")
        indices = torch.arange(query_count, device=full_batch.receivers.fluid_xy.device)
        probe_batch = replace(
            full_batch,
            receivers=replace(full_batch.receivers,
                              fluid_xy=full_batch.receivers.fluid_xy.index_select(1, indices)),
            targets=replace(full_batch.targets,
                            field_targets=full_batch.targets.field_targets.index_select(1, indices),
                            point_weights=full_batch.targets.point_weights.index_select(1, indices)),
        )

        def prepare_and_run(batch, *, capture_trace):
            prepared = provider._prepare_for_probe(model, batch)
            if capture_trace:
                native, decode, trace = _native_with_first_tile_trace(model, prepared)
            else:
                native, decode, output = model.apply_native_with_work_counts(
                    prepared, batch.receivers.heat,
                )
                trace = getattr(output, "trace", None)
            prediction = ThermalContentPredictions(prepared, native, work=decode)
            roles = provider._role_losses(prediction, batch.targets)
            if capture_trace and not isinstance(trace, Mapping):
                raise RuntimeError("Native member-routing observation did not expose its actual route trace.")
            return prepared, native, trace or {}, roles

        # Compute objective errors and parameter/B/key/value gradients over the
        # complete provider-sampled Q=1024 batch. The bounded trace below only
        # captures eight fixed receiver rows for support and score evidence.
        full_prepared, full_native, _full_trace, full_roles = prepare_and_run(full_batch, capture_trace=False)
        weighted_full_loss = sum(float(provider.native_weights[name]) * value.mean()
                                 for name, value in full_roles.items())
        group_norms, live_norms = _gradient_observation(
            model, weighted_full_loss, full_prepared, {},
        )

        trace_rng_cpu = torch.random.get_rng_state()
        trace_rng_cuda = (torch.cuda.get_rng_state(torch.device(provider.device))
                          if torch.device(provider.device).type == "cuda" else None)
        prepared, native, trace, roles = prepare_and_run(probe_batch, capture_trace=True)
        weighted_trace_loss = sum(float(provider.native_weights[name]) * value.mean()
                                  for name, value in roles.items())
        live_norms.update(_score_gradient_observation(weighted_trace_loss, trace))
        score_gradient_queries = int(trace["proposal_logits_M"].shape[1])
        trace = _first_query_rows(trace, query_count)

        torch.random.set_rng_state(trace_rng_cpu)
        if trace_rng_cuda is not None:
            torch.cuda.set_rng_state(trace_rng_cuda, torch.device(provider.device))
        with torch.no_grad():
            repeat_prepared, repeat_native, repeat_trace, _repeat_roles = prepare_and_run(probe_batch, capture_trace=True)
        repeat_trace = _first_query_rows(repeat_trace, query_count)
        perturbed_receivers = replace(probe_batch.receivers, heat=probe_batch.receivers.heat * 1.1)
        heat_batch = replace(probe_batch, receivers=perturbed_receivers)
        torch.random.set_rng_state(trace_rng_cpu)
        if trace_rng_cuda is not None:
            torch.cuda.set_rng_state(trace_rng_cuda, torch.device(provider.device))
        with torch.no_grad():
            heat_prepared, heat_native, heat_trace, _heat_roles = prepare_and_run(heat_batch, capture_trace=True)
        heat_trace = _first_query_rows(heat_trace, query_count)

        supports = {
            "proposal_M": _support_with_ids(trace.get("proposal_weights_M"),
                                             trace.get("original_module_ids")),
            "candidate_M": _support_with_ids(trace.get("candidate_mask_M"),
                                              trace.get("original_module_ids")),
            "candidate_E": _support_with_ids(trace.get("candidate_mask_E"),
                                              trace.get("original_environment_ids")),
            "final_M": _support_with_ids(trace.get("member_weights_M"),
                                          trace.get("original_module_ids")),
            "final_E": _support_with_ids(trace.get("member_weights_E"),
                                          trace.get("original_environment_ids")),
        }
        nulls = {
            name: (None if trace.get(name) is None else _scalar(trace[name].detach().mean()))
            for name in ("proposal_null", "member_null_M", "member_null_E")
        }
        scene = prepared.organization.scene
        score_summaries = {
            name: _score_summary(scores, allowed, null)
            for name, (scores, allowed, null) in _score_targets(trace, prepared).items()
        }

        sensitivity = {
            "repeatability_fieldwise_native_delta": _native_field_delta(native, repeat_native, model, prepared),
            "heat_plus10_fieldwise_native_delta": _native_field_delta(native, heat_native, model, prepared),
        }
        path_read = {path: _read_vector(prepared, trace, path) for path in TYPED_PATHS}
        path_repeat = {path: _read_vector(repeat_prepared, repeat_trace, path) for path in TYPED_PATHS}
        path_heat = {path: _read_vector(heat_prepared, heat_trace, path) for path in TYPED_PATHS}
        read_strength = {path: _scalar(path_read[path].detach().double().norm()) for path in TYPED_PATHS}
        read_noise = {path: _tensor_norm(path_repeat[path] - path_read[path]) for path in TYPED_PATHS}
        read_heat_delta = {path: _tensor_norm(path_heat[path] - path_read[path]) for path in TYPED_PATHS}

        module_count = int(scene.module_mask[0].sum().item())
        role_errors = {name: _scalar(value.detach().mean()) for name, value in full_roles.items()}
        expected_roles = set(NATIVE_ROLE_NAMES)
        if set(role_errors) != expected_roles:
            raise RuntimeError(f"Native health role inventory changed: {sorted(role_errors)}.")
        return {
            "case_id": str(case_id), "module_count": module_count,
            "native_primary_query_count": int(full_batch.receivers.fluid_xy.shape[1]),
            "trace_query_count": int(query_count),
            "fieldwise_error_query_scope": "full provider-sampled Q=1024 native roles",
            "fieldwise_native_error_normalized_mse": role_errors,
            "support": supports,
            "null_usage_mean": nulls,
            "raw_centered_scores_and_margins": score_summaries,
            "objective_gradient_group_norms": group_norms,
            "live_gradient_norms": live_norms,
            "score_gradient_query_scope": "actual first decoder tile under eight-fluid-query plus full native surface/material objective",
            "score_gradient_query_count": score_gradient_queries,
            "typed_read_strength_l2": read_strength,
            "typed_read_repeatability_l2": read_noise,
            "typed_read_heat_sensitivity_l2": read_heat_delta,
            "heat_sensitivity": sensitivity,
            "optional_corev2_read_fields_mean": {
                str(key): _scalar(value.detach().mean())
                for key, value in trace.items()
                if str(key).startswith(("typed_read_strength", "member_read_strength_",
                                        "member_read_contrast_", "member_gate_derivative_",
                                        "member_omitted_mass_")) and torch.is_tensor(value)
            },
        }

    def observe(self, model, *, training_epoch: int) -> dict[str, Any]:
        provider = self.provider
        device = torch.device(provider.device)
        modules = tuple(model.modules())
        training_flags = tuple(module.training for module in modules)
        parameters = tuple(model.parameters())
        gradients = tuple((parameter.grad, None if parameter.grad is None else parameter.grad.detach().clone())
                          for parameter in parameters)
        rng_python = random.getstate()
        rng_numpy = np.random.get_state()
        rng_torch = torch.random.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        counters = {name: getattr(provider, name) for name in ("training_samples", "validation_samples")
                    if hasattr(provider, name)}
        core = getattr(model, "core", model)
        clocks = {name: getattr(core, name) for name in
                  ("_joint_epoch", "_joint_update_index", "_joint_macro_clock") if hasattr(core, name)}
        cuda_start = cuda_end = None
        if device.type == "cuda":
            cuda_start, cuda_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            cuda_start.record()
        wall_start = time.perf_counter()
        try:
            with torch.enable_grad():
                case_records = [self._run_case(model, case_id) for case_id in self.panel_case_ids]
            if cuda_end is not None:
                cuda_end.record()
                torch.cuda.synchronize(device)
            wall_seconds = time.perf_counter() - wall_start
            cuda_seconds = None if cuda_start is None else float(cuda_start.elapsed_time(cuda_end)) / 1000.0
            paths = {path: _path_summary(case_records, path) for path in TYPED_PATHS}
            core = getattr(model, "core", model)
            read_override = getattr(core, "_read_execution_override", None)
            training_read_mode = (
                read_override or ("continuous" if model.training else "sparse")
                if hasattr(core, "sparse_mass_tolerance") else
                ("native_training" if model.training else "native_evaluation")
            )
            return {
                "schema": HEALTH_SCHEMA,
                "training_epoch": int(training_epoch),
                "sampling_epoch": 1,
                "panel_case_ids": list(self.panel_case_ids),
                "panel_module_counts": [int(record["module_count"]) for record in case_records],
                "cases": case_records,
                "paths": paths,
                "execution_scope": {
                    "native_training_path_observed": True,
                    "training_read_mode": training_read_mode,
                    "training_weights_are_not_sparse_execution_evidence": True,
                    "sparse_execution_measured": False,
                    "sparse_executor_savings_measured": False,
                    "trace_query_cap": 8,
                },
                "diagnostic_runtime": {
                    "wall_seconds": float(wall_seconds),
                    "cuda_seconds": cuda_seconds,
                    "included_in_training_optimizer_runtime": False,
                },
            }
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.set_rng_state(rng_cuda, device)
            torch.random.set_rng_state(rng_torch)
            np.random.set_state(rng_numpy)
            random.setstate(rng_python)
            for name, value in counters.items():
                setattr(provider, name, value)
            for name, value in clocks.items():
                setattr(core, name, value)
            for parameter, (original, saved) in zip(parameters, gradients):
                if original is None:
                    parameter.grad = None
                else:
                    if parameter.grad is not original:
                        parameter.grad = original
                    with torch.no_grad():
                        original.copy_(saved)
            for module, flag in zip(modules, training_flags):
                module.training = flag


def request_clean_stop(output_dir, *, epoch: int, reason: Mapping[str, Any]) -> dict[str, Any]:
    """Use the runner's ordinary clean-stop request without replacing another request."""
    from honf_runtime.unified_training import _atomic_json

    path = output_dir / "CLEAN_STOP_REQUEST.json"
    if path.exists():
        return {"written": False, "existing_request_preserved": True}
    request = {
        "schema": "ordinary_clean_stop_request_v1",
        "request_id": f"member-routing-health-epoch-{int(epoch)}",
        "requested_unix": time.time(),
        "requested_by": "member_routing_health_v1",
        "reason": "fixed TRAIN panel detected widespread native path collapse",
        "health_evidence": dict(reason),
    }
    _atomic_json(path, request)
    return {"written": True, "request_id": request["request_id"], "path": str(path)}


__all__ = ["HEALTH_SCHEMA", "HEALTH_STATE_SCHEMA", "MemberRoutingHealth", "evaluate_pause", "request_clean_stop"]

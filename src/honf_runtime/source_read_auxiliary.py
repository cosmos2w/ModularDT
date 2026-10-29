"""Bounded TRAIN-only source-read calibration and sparse-child auxiliary fit.

The module is deliberately separate from the physical provider state. It fits
only newly attached ``direct_reader.source_read`` parameters, uses dense model
targets from the declared TRAIN cohort, and retains at most one compact
128-query refresh buffer for exact continuation.
"""

from __future__ import annotations

import hashlib
import math
import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from honf_forward_core.content_hypergraph.source_read_plans import (
    SourceReadControllerV1,
    combine_source_gates,
    smooth_near_gate,
    smooth_threshold_gate,
)

from .source_read_child import (
    AUXILIARY_FIT_SCHEMA,
    AUXILIARY_OPTIMIZER_ALGORITHM,
    MAX_AUXILIARY_UPDATES,
)
from .unified_training import SamplingKey, _sampling_dataset_id


SCHEMA = "source_read_auxiliary_manager_v1"
STATE_SCHEMA = "source_read_auxiliary_state_v5"
MAX_INITIAL_QUERY_ROWS = 256
MAX_REFRESH_QUERY_ROWS = 128
DEFAULT_RAM_CACHE_BUDGET_BYTES = 8 * 1024**3
_ALLOWED_VARIANTS = frozenset(("S-Utility", "S-Complete"))
_SOURCE_READ_PARAMETER_MARKER = "direct_reader.source_read."


def _finite_positive_scalar(value: Any, name: str) -> float:
    result = torch.as_tensor(value, dtype=torch.float64)
    if result.numel() != 1 or not bool(torch.isfinite(result).all()) or not bool((result > 0).all()):
        raise ValueError(f"{name} must be a finite positive scalar.")
    return float(result.reshape(()).item())


def _tensor_digest(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _model_parts(model: nn.Module) -> tuple[nn.Module, nn.Module, SourceReadControllerV1]:
    core = getattr(model, "core", model)
    direct_reader = getattr(core, "direct_reader", None)
    controller = getattr(direct_reader, "source_read", None)
    if not isinstance(controller, SourceReadControllerV1):
        raise ValueError("Source-read auxiliary calibration requires DirectSourceReaderV2.source_read.")
    if direct_reader is None or not isinstance(getattr(direct_reader, "message_output", None), nn.Linear):
        raise ValueError("Source-read auxiliary calibration requires the native message_output projection.")
    if controller.thermal and not isinstance(getattr(direct_reader, "thermal_output", None), nn.Linear):
        raise ValueError("Thermal source-read calibration requires the native thermal_output projection.")
    return core, direct_reader, controller


def _named_controller_parameters(
    model: nn.Module, controller: nn.Module,
) -> dict[str, nn.Parameter]:
    all_parameters = dict(model.named_parameters())
    controller_ids = {id(parameter) for parameter in controller.parameters()}
    result = {name: parameter for name, parameter in all_parameters.items()
              if id(parameter) in controller_ids}
    if len(result) != len(controller_ids) or not result:
        raise ValueError("Source-read controller parameters are not uniquely named by the model.")
    if any(_SOURCE_READ_PARAMETER_MARKER not in name for name in result):
        raise ValueError("Every new source-read parameter must live below direct_reader.source_read.*.")
    return result


@dataclass(frozen=True)
class _TeacherCase:
    case_key: Any
    query_states: Tensor
    module_states: Tensor
    descriptor: Tensor
    alpha: Tensor
    fine_messages: Tensor
    thermal_k: Tensor | None
    query_mask: Tensor
    source_present_mask: Tensor
    local_radius: Tensor
    distance_to_sources: Tensor
    fine_executed_rows: int
    fine_present_rows: int

    @property
    def batch_size(self) -> int:
        return int(self.query_states.shape[0])

    @property
    def query_count(self) -> int:
        return int(self.query_states.shape[1])

    @property
    def source_count(self) -> int:
        return int(self.module_states.shape[1])

    @property
    def pair_rows(self) -> int:
        source_mask = _source_mask_3d(self.source_present_mask, self.batch_size, self.query_count,
                                      self.source_count)
        valid = self.query_mask[:, :, None] & source_mask
        return int(valid.sum().item())

    @property
    def byte_count(self) -> int:
        values = (self.query_states, self.module_states, self.descriptor, self.alpha,
                  self.fine_messages, self.query_mask, self.source_present_mask,
                  self.local_radius, self.distance_to_sources)
        total = sum(value.numel() * value.element_size() for value in values)
        if self.thermal_k is not None:
            total += self.thermal_k.numel() * self.thermal_k.element_size()
        return int(total)

    def state_dict(self) -> dict[str, Any]:
        return {
            "case_key": self.case_key,
            "query_states": self.query_states,
            "module_states": self.module_states,
            "descriptor": self.descriptor,
            "alpha": self.alpha,
            "fine_messages": self.fine_messages,
            "thermal_k": self.thermal_k,
            "query_mask": self.query_mask,
            "source_present_mask": self.source_present_mask,
            "local_radius": self.local_radius,
            "distance_to_sources": self.distance_to_sources,
            "fine_executed_rows": self.fine_executed_rows,
            "fine_present_rows": self.fine_present_rows,
        }


def _source_mask_3d(mask: Tensor, batch: int, queries: int, sources: int) -> Tensor:
    if mask.dtype != torch.bool:
        raise ValueError("source_present_mask must be boolean.")
    if tuple(mask.shape) == (batch, sources):
        return mask[:, None, :].expand(batch, queries, sources)
    if tuple(mask.shape) == (batch, queries, sources):
        return mask
    raise ValueError("source_present_mask must have shape [B,M] or [B,Q,M].")


def select_evenly_spaced_teacher_rows(query, max_query_rows: int):
    """Select at most the bound from valid rows and preserve native query IDs."""
    from honf_forward_core.content_hypergraph.query_chunks_v1 import select_query_rows_v1

    if type(max_query_rows) is not int or not 1 <= max_query_rows <= MAX_INITIAL_QUERY_ROWS:
        raise ValueError("Dense teacher query selection must be bounded to [1,256] rows.")
    coordinates = getattr(query, "coordinates", None)
    query_mask = getattr(query, "query_mask", None)
    if not isinstance(coordinates, Tensor) or coordinates.ndim != 3 or coordinates.shape[0] != 1:
        raise ValueError("Dense teacher query selection requires a single-case QueryBatch.")
    if query_mask is None:
        valid = torch.arange(coordinates.shape[1], device=coordinates.device, dtype=torch.long)
    else:
        if not isinstance(query_mask, Tensor) or query_mask.dtype != torch.bool or tuple(query_mask.shape) != tuple(coordinates.shape[:2]):
            raise ValueError("Dense teacher query mask must be boolean and match [1,Q].")
        valid = torch.nonzero(query_mask[0], as_tuple=False).reshape(-1)
    if valid.numel() == 0:
        raise ValueError("Dense teacher query batch contains no valid native union rows.")
    count = min(int(valid.numel()), max_query_rows)
    if count == 1:
        positions = torch.zeros(1, device=valid.device, dtype=torch.long)
    else:
        positions = torch.div(
            torch.arange(count, device=valid.device, dtype=torch.long) * (valid.numel() - 1),
            count - 1,
            rounding_mode="floor",
        )
    indices = valid.index_select(0, positions)
    if indices.unique().numel() != indices.numel():
        raise RuntimeError("Evenly spaced source-read teacher query indices are not unique.")
    return select_query_rows_v1(query, indices)


def _trace_value(trace: Mapping[str, Any], *keys: str, required: bool = True) -> Any:
    for key in keys:
        if key in trace:
            return trace[key]
    if required:
        raise ValueError(f"Source-read teacher trace is missing {keys[0]!r}.")
    return None


def _cpu_tensor(value: Any, name: str, *, boolean: bool = False) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"Source-read teacher trace {name} must be a tensor.")
    result = value.detach().to(device="cpu").contiguous()
    if boolean and result.dtype != torch.bool:
        raise ValueError(f"Source-read teacher trace {name} must be boolean.")
    if not boolean and not torch.isfinite(result).all():
        raise ValueError(f"Source-read teacher trace {name} contains nonfinite values.")
    return result


def _normalize_teacher_case(
    case_key: Any, trace: Mapping[str, Any], controller: SourceReadControllerV1,
    *, max_query_rows: int,
) -> _TeacherCase:
    if not isinstance(trace, Mapping):
        raise ValueError("The source-read teacher factory must return a trace mapping.")
    query_states = _cpu_tensor(_trace_value(trace, "query_states"), "query_states")
    module_states = _cpu_tensor(_trace_value(trace, "module_states"), "module_states")
    descriptor = _cpu_tensor(_trace_value(trace, "descriptor", "geometry_descriptor"), "descriptor")
    alpha = _cpu_tensor(_trace_value(trace, "alpha", "original_alpha"), "alpha")
    fine_messages = _cpu_tensor(_trace_value(trace, "fine_messages"), "fine_messages")
    thermal_value = _trace_value(trace, "thermal_k", "thermal_coefficients", required=controller.thermal)
    thermal_k = None if thermal_value is None else _cpu_tensor(thermal_value, "thermal_k")
    query_mask = _cpu_tensor(_trace_value(trace, "query_mask"), "query_mask", boolean=True)
    source_mask = _cpu_tensor(_trace_value(trace, "source_present_mask"), "source_present_mask", boolean=True)
    local_radius = _cpu_tensor(_trace_value(trace, "local_radius"), "local_radius")
    distance = _cpu_tensor(_trace_value(trace, "distance_to_sources"), "distance_to_sources")
    fine_executed_rows = _trace_value(trace, "fine_executed_rows")
    fine_present_rows = _trace_value(trace, "fine_present_rows")

    if query_states.ndim != 3 or module_states.ndim != 3:
        raise ValueError("Teacher query/module states must have shapes [B,Q,D] and [B,M,D].")
    batch, queries, width = query_states.shape
    if batch != 1 or module_states.shape[0] != 1 or module_states.shape[-1] != width:
        raise ValueError("Each source-read teacher trace must describe exactly one TRAIN case.")
    sources = int(module_states.shape[1])
    if queries > max_query_rows:
        raise ValueError(f"Teacher trace exceeded its {max_query_rows}-query row bound.")
    if width != controller.width:
        raise ValueError("Teacher state width differs from the source-read controller.")
    if tuple(descriptor.shape) != (batch, queries, sources, 2 * controller.coordinate_dim + 1):
        raise ValueError("Teacher descriptor has the wrong [B,Q,M,2*C+1] shape.")
    if tuple(alpha.shape) != (batch, queries, sources, controller.heads):
        raise ValueError("Teacher alpha has the wrong [B,Q,M,heads] shape.")
    if tuple(fine_messages.shape) != (batch, queries, sources, width):
        raise ValueError("Teacher fine_messages has the wrong [B,Q,M,width] shape.")
    if query_mask.dtype != torch.bool or tuple(query_mask.shape) != (batch, queries):
        raise ValueError("query_mask must have shape [1,Q] and boolean dtype.")
    _source_mask_3d(source_mask, batch, queries, sources)
    if tuple(local_radius.shape) == (batch, sources):
        local_radius = local_radius[:, None, :]
    elif tuple(local_radius.shape) not in ((batch, 1, sources), (batch, queries, sources)):
        raise ValueError("local_radius must have shape [B,M], [B,1,M], or [B,Q,M].")
    if tuple(distance.shape) != (batch, queries, sources):
        raise ValueError("distance_to_sources must have shape [B,Q,M].")
    valid_source = _source_mask_3d(source_mask, batch, queries, sources)
    if type(fine_executed_rows) is not int or fine_executed_rows != batch * queries * sources:
        raise ValueError("Teacher fine_executed_rows must include every dense BQM pair, including padding.")
    expected_present_rows = int((query_mask[:, :, None] & valid_source).sum().item())
    if type(fine_present_rows) is not int or fine_present_rows != expected_present_rows:
        raise ValueError("Teacher fine_present_rows must match valid queries and present source columns.")
    radius_valid = torch.isfinite(local_radius) & (local_radius > 0)
    if not radius_valid.expand_as(valid_source).logical_or(~valid_source).all():
        raise ValueError("Present sources require finite positive local radii.")
    distance_valid = torch.isfinite(distance) & (distance >= 0)
    if not (distance_valid.logical_or(~valid_source)).all():
        raise ValueError("Present source distances must be finite and nonnegative.")
    if controller.thermal:
        if thermal_k is None or tuple(thermal_k.shape) != (batch, queries, sources):
            raise ValueError("Thermal teacher trace requires thermal_k with shape [B,Q,M].")
    elif thermal_k is not None:
        raise ValueError("Nonthermal source-read controller cannot receive thermal coefficients.")

    # Padded-source radii are irrelevant to the gate. Replace them by a benign
    # positive value so the shared smooth gate remains well-defined.
    local_radius = torch.where(radius_valid, local_radius, torch.ones_like(local_radius))
    distance = torch.where(distance_valid, distance, torch.zeros_like(distance))
    return _TeacherCase(
        case_key=case_key,
        query_states=query_states,
        module_states=module_states,
        descriptor=descriptor,
        alpha=alpha,
        fine_messages=fine_messages,
        thermal_k=thermal_k,
        query_mask=query_mask,
        source_present_mask=source_mask,
        local_radius=local_radius,
        distance_to_sources=distance,
        fine_executed_rows=fine_executed_rows,
        fine_present_rows=fine_present_rows,
    )


def _device_dtype(controller: SourceReadControllerV1) -> tuple[torch.device, torch.dtype]:
    parameter = next(controller.parameters(), None)
    if parameter is None:
        return torch.device("cpu"), torch.float32
    dtype = parameter.dtype if torch.is_floating_point(parameter) else torch.float32
    return parameter.device, dtype


def _case_to_device(case: _TeacherCase, device: torch.device, dtype: torch.dtype) -> dict[str, Tensor | None]:
    return {
        "query_states": case.query_states.to(device=device, dtype=dtype),
        "module_states": case.module_states.to(device=device, dtype=dtype),
        "descriptor": case.descriptor.to(device=device, dtype=dtype),
        "alpha": case.alpha.to(device=device, dtype=dtype),
        "fine_messages": case.fine_messages.to(device=device, dtype=dtype),
        "thermal_k": None if case.thermal_k is None else case.thermal_k.to(device=device, dtype=dtype),
        "query_mask": case.query_mask.to(device=device),
        "source_present_mask": case.source_present_mask.to(device=device),
        "local_radius": case.local_radius.to(device=device, dtype=dtype),
        "distance_to_sources": case.distance_to_sources.to(device=device, dtype=dtype),
    }


def _valid_pair_mask(data: Mapping[str, Tensor | None]) -> Tensor:
    query_mask = data["query_mask"]
    source_mask = data["source_present_mask"]
    assert isinstance(query_mask, Tensor) and isinstance(source_mask, Tensor)
    batch, queries = query_mask.shape
    sources = data["module_states"].shape[1]  # type: ignore[union-attr]
    return query_mask[:, :, None] & _source_mask_3d(source_mask, batch, queries, sources)


def _near_gate(data: Mapping[str, Tensor | None]) -> Tensor:
    radius = data["local_radius"]
    distance = data["distance_to_sources"]
    assert isinstance(radius, Tensor) and isinstance(distance, Tensor)
    physical_distance = distance
    return smooth_near_gate(physical_distance, radius)


def _completion_outputs(
    controller: SourceReadControllerV1, direct_reader: nn.Module,
    query_states: Tensor, module_states: Tensor,
) -> tuple[Tensor, Tensor | None]:
    """Evaluate the rank-r base only inside bounded auxiliary teacher buffers."""
    if not controller.completion_enabled:
        raise ValueError("Completion outputs requested for an S-Utility controller.")
    batch, queries, _ = query_states.shape
    sources = module_states.shape[1]
    q_projection = controller.project_completion_queries(query_states)
    s_projection = controller.project_completion_sources(module_states)
    b_index = torch.arange(batch, device=query_states.device)[:, None, None].expand(
        batch, queries, sources
    ).reshape(-1)
    q_index = torch.arange(queries, device=query_states.device)[None, :, None].expand(
        batch, queries, sources
    ).reshape(-1)
    s_index = torch.arange(sources, device=query_states.device)[None, None, :].expand(
        batch, queries, sources
    ).reshape(-1)
    hidden = controller.completion_pair_hidden_from_projections(
        q_projection, s_projection, b_index, q_index, s_index
    ).reshape(batch, queries, sources, controller.width)
    message_output = getattr(direct_reader, "message_output")
    # During child initialization/co-adaptation only newly added controller
    # parameters may receive auxiliary gradients. Keep inherited Wo/K weights
    # as fixed linear maps for this auxiliary signal.
    messages = F.linear(
        hidden,
        message_output.weight.detach(),
        None if message_output.bias is None else message_output.bias.detach(),
    )
    thermal_output = getattr(direct_reader, "thermal_output", None)
    thermal_k = None
    if controller.thermal:
        assert isinstance(thermal_output, nn.Linear)
        thermal_k = F.linear(
            messages,
            thermal_output.weight.detach(),
            None if thermal_output.bias is None else thermal_output.bias.detach(),
        ).squeeze(-1)
    return messages, thermal_k


def _utility_targets(
    controller: SourceReadControllerV1,
    alpha: Tensor,
    fine_messages: Tensor,
    fine_k: Tensor | None,
    *,
    context_scales: Tensor,
    heating_envelope: float,
    temperature_scale: float,
    completion_messages: Tensor | None = None,
    completion_k: Tensor | None = None,
) -> Tensor:
    """Build the declared output-relevant dense teacher targets from TRAIN."""
    if alpha.ndim != 4 or alpha.shape[-1] != controller.heads:
        raise ValueError("Teacher attention must have shape [B,Q,M,heads].")
    batch, queries, sources, _ = alpha.shape
    if tuple(fine_messages.shape) != (batch, queries, sources, controller.width):
        raise ValueError("Teacher fine messages do not align with attention and controller width.")
    if completion_messages is not None and completion_messages.shape != fine_messages.shape:
        raise ValueError("Completion messages must align with fine teacher messages.")
    residual = fine_messages if completion_messages is None else fine_messages - completion_messages
    head_width = controller.width // controller.heads
    residual_heads = residual.reshape(batch, queries, sources, controller.heads, head_width)
    flow = (alpha[..., None] * residual_heads).norm(dim=-1)
    flow = flow / context_scales.to(flow).view(1, 1, 1, controller.heads)
    blocks = [flow]
    if controller.thermal:
        if fine_k is None or tuple(fine_k.shape) != (batch, queries, sources):
            raise ValueError("Thermal teacher targets require fine K with shape [B,Q,M].")
        if completion_k is not None and completion_k.shape != fine_k.shape:
            raise ValueError("Completion K must align with fine thermal teacher values.")
        residual_k = fine_k if completion_k is None else fine_k - completion_k
        blocks.append((heating_envelope * residual_k.abs() / temperature_scale).unsqueeze(-1))
    elif fine_k is not None or completion_k is not None:
        raise ValueError("Nonthermal source-read targets cannot include thermal coefficients.")
    return torch.cat(blocks, dim=-1).detach()


def _scorer_and_completion_losses(
    controller: SourceReadControllerV1,
    direct_reader: nn.Module,
    data: Mapping[str, Tensor | None],
    *,
    context_scales: Tensor,
    heating_envelope: float,
    temperature_scale: float = 1.0,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    query_states = data["query_states"]
    module_states = data["module_states"]
    descriptor = data["descriptor"]
    alpha = data["alpha"]
    fine_messages = data["fine_messages"]
    fine_k = data["thermal_k"]
    assert all(isinstance(value, Tensor) for value in (
        query_states, module_states, descriptor, alpha, fine_messages
    ))
    assert fine_k is None or isinstance(fine_k, Tensor)
    valid = _valid_pair_mask(data)
    completion_messages: Tensor | None = None
    completion_k: Tensor | None = None
    completion_loss = fine_messages.new_zeros(())
    if controller.completion_enabled:
        completion_messages, completion_k = _completion_outputs(
            controller, direct_reader, query_states, module_states
        )
        # The completion learns the output-relevant head contexts and signed
        # thermal coefficients, not an arbitrary hidden-vector norm.
        batch, queries, sources, _ = fine_messages.shape
        head_width = controller.width // controller.heads
        scale = context_scales.to(fine_messages).view(1, 1, controller.heads, 1)
        fine_heads = fine_messages.reshape(batch, queries, sources, controller.heads, head_width)
        base_heads = completion_messages.reshape(batch, queries, sources, controller.heads, head_width)
        weights = alpha[:, :, :, :, None]
        fine_context = (weights * fine_heads).sum(dim=2) / scale
        base_context = (weights * base_heads).sum(dim=2) / scale
        valid_queries = data["query_mask"]
        assert isinstance(valid_queries, Tensor)
        if bool(valid_queries.any()):
            flow_completion_loss = (fine_context - base_context).square()[valid_queries].mean()
        else:
            flow_completion_loss = fine_messages.new_zeros(())
        if controller.thermal:
            assert fine_k is not None and completion_k is not None
            thermal_residual = heating_envelope * (fine_k - completion_k) / temperature_scale
            thermal_completion_loss = thermal_residual.square()[valid].mean() if bool(valid.any()) else fine_messages.new_zeros(())
        else:
            thermal_completion_loss = fine_messages.new_zeros(())
        completion_loss = flow_completion_loss + thermal_completion_loss
    targets = _utility_targets(
        controller,
        alpha,
        fine_messages,
        fine_k,
        context_scales=context_scales,
        heating_envelope=heating_envelope,
        temperature_scale=temperature_scale,
        completion_messages=None if completion_messages is None else completion_messages.detach(),
        completion_k=None if completion_k is None else completion_k.detach(),
    )
    source_projection = controller.project_source_states(module_states)
    query_projection = controller.project_query_states(query_states)
    predictions = controller.utility_scores(source_projection, query_projection, descriptor)
    transformed_prediction = torch.log1p(predictions.clamp_min(0.0))
    transformed_target = torch.log1p(targets.clamp_min(0.0))
    pair_loss = F.smooth_l1_loss(transformed_prediction, transformed_target, reduction="none").mean(dim=-1)
    scorer_loss = pair_loss[valid].mean() if bool(valid.any()) else pair_loss.sum() * 0.0
    components = {
        "scorer": scorer_loss.detach(),
        "completion": completion_loss.detach(),
        "total_unweighted": (scorer_loss + completion_loss).detach(),
    }
    return scorer_loss, completion_loss, components


def _context_scales(cases: Sequence[_TeacherCase], controller: SourceReadControllerV1) -> Tensor:
    """RMS dense direct-context norm by head over valid fixed TRAIN queries."""
    totals = torch.zeros(controller.heads, dtype=torch.float64)
    counts = torch.zeros(controller.heads, dtype=torch.float64)
    head_width = controller.width // controller.heads
    for case in cases:
        batch, queries, sources, _ = case.fine_messages.shape
        messages = case.fine_messages.reshape(batch, queries, sources, controller.heads, head_width)
        direct_context = (case.alpha[..., None] * messages).sum(dim=2)
        magnitude_sq = direct_context.square().sum(dim=-1).to(torch.float64)
        valid_queries = case.query_mask.to(torch.bool)
        valid_per_head = valid_queries[..., None].expand_as(magnitude_sq)
        totals += magnitude_sq.masked_fill(~valid_per_head, 0.0).sum(dim=(0, 1))
        counts += valid_per_head.sum(dim=(0, 1)).to(torch.float64)
    scales = torch.sqrt(totals / counts.clamp_min(1.0)).to(torch.float32)
    # The declared fixed TRAIN floor keeps silent heads finite without
    # replacing their scale by an unrelated unit magnitude.
    scales = torch.where(torch.isfinite(scales), scales.clamp_min(1.0e-8), torch.full_like(scales, 1.0e-8))
    return scales


def _stable_int(*parts: Any) -> int:
    payload = "\x1f".join(str(value) for value in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _make_sampling_key(
    *, seed: int, epoch: int, update_index: int, stage: str, variant: str,
    sampling_version: str, dataset_id: str,
) -> SamplingKey:
    return SamplingKey(
        seed=int(seed), epoch=int(epoch), update_index=int(update_index), microbatch_index=0,
        stage=stage, arm=variant, sampling_version=sampling_version, dataset_id=dataset_id,
    )


def _factory_trace(
    provider, model: nn.Module, case_key: Any, key: SamplingKey, max_query_rows: int,
    factory: Callable[[Any, nn.Module, Any, SamplingKey, int], Mapping[str, Any]] | None,
) -> Mapping[str, Any]:
    state_method = getattr(provider, "training_state_dict", None)
    provider_state = copy.deepcopy(state_method()) if callable(state_method) else None
    scalar_names = (
        "physical_training_samples", "physical_training_queries",
        "training_samples", "training_queries",
    )
    counters = {}
    for name in scalar_names:
        if not hasattr(provider, name):
            continue
        value = getattr(provider, name)
        if isinstance(value, Tensor) and value.numel() == 1:
            value = value.detach().cpu().clone()
        elif isinstance(value, (int, float, bool, str)) or value is None:
            value = copy.deepcopy(value)
        else:
            continue
        counters[name] = value
    try:
        if factory is not None:
            result = factory(provider, model, case_key, key, max_query_rows)
        else:
            method = getattr(provider, "source_read_teacher_trace", None)
            if not callable(method):
                raise ValueError("Provider must implement source_read_teacher_trace for sparse-child calibration.")
            result = method(model, case_key, key, max_query_rows)
    finally:
        if callable(state_method):
            after_state = state_method()
            if not _nested_equal(provider_state, after_state):
                raise ValueError("TRAIN teacher probe changed provider training_state_dict counters/state.")
        for name, before in counters.items():
            after = getattr(provider, name)
            if isinstance(after, Tensor) and after.numel() == 1:
                after = after.detach().cpu()
            if not _nested_equal(before, after):
                raise ValueError(f"TRAIN teacher probe changed provider physical sampling counter {name!r}.")
    return result


def _nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return isinstance(left, Tensor) and isinstance(right, Tensor) and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (isinstance(left, Mapping) and isinstance(right, Mapping) and left.keys() == right.keys()
                and all(_nested_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_nested_equal(a, b) for a, b in zip(left, right)))
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _factory_physical_loss(
    provider, model: nn.Module, case_key: Any, key: SamplingKey,
    factory: Callable[[Any, nn.Module, Any, SamplingKey], Tensor] | None,
) -> Tensor:
    state_method = getattr(provider, "training_state_dict", None)
    provider_state = copy.deepcopy(state_method()) if callable(state_method) else None
    scalar_names = (
        "physical_training_samples", "physical_training_queries",
        "training_samples", "training_queries",
    )
    counters: dict[str, Any] = {}
    for name in scalar_names:
        if not hasattr(provider, name):
            continue
        value = getattr(provider, name)
        if isinstance(value, Tensor) and value.numel() == 1:
            value = value.detach().cpu().clone()
        elif isinstance(value, (int, float, bool, str)) or value is None:
            value = copy.deepcopy(value)
        else:
            continue
        counters[name] = value
    try:
        if factory is not None:
            result = factory(provider, model, case_key, key)
        else:
            method = getattr(provider, "source_read_physical_calibration_loss", None)
            if not callable(method):
                raise ValueError("Provider must implement source_read_physical_calibration_loss.")
            result = method(model, case_key, key)
    finally:
        if callable(state_method):
            after_state = state_method()
            if not _nested_equal(provider_state, after_state):
                raise ValueError("Native physical calibration changed provider training_state_dict counters/state.")
        for name, before in counters.items():
            after = getattr(provider, name)
            if isinstance(after, Tensor) and after.numel() == 1:
                after = after.detach().cpu()
            if not _nested_equal(before, after):
                raise ValueError(f"Native physical calibration changed sampling counter {name!r}.")
    if not isinstance(result, Tensor) or result.numel() != 1 or not result.requires_grad:
        raise ValueError("Native physical calibration factory must return a differentiable scalar tensor.")
    if not torch.isfinite(result).all():
        raise ValueError("Native physical calibration loss is nonfinite.")
    return result.reshape(())


def _gradient_norm(gradients: Sequence[Tensor | None]) -> float:
    squared = 0.0
    for gradient in gradients:
        if gradient is not None:
            squared += float(gradient.detach().to(torch.float64).square().sum().item())
    return math.sqrt(squared)


def _relative_rms(diff_squared: float, reference_squared: float) -> float:
    if diff_squared < 0 or reference_squared < 0 or not math.isfinite(diff_squared + reference_squared):
        raise ValueError("TRAIN fidelity screen accumulated an invalid squared norm.")
    if reference_squared <= 1.0e-24:
        return 0.0 if diff_squared <= 1.0e-24 else math.sqrt(diff_squared) / 1.0e-12
    return math.sqrt(diff_squared / reference_squared)


def _validate_auxiliary_optimizer(optimizer: torch.optim.Optimizer,
                                  new_parameters: Mapping[str, nn.Parameter]) -> None:
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("Source-read auxiliary initialization requires the inherited joint AdamW optimizer.")
    new_ids = {id(parameter) for parameter in new_parameters.values()}
    found_ids: set[int] = set()
    for group in optimizer.param_groups:
        group_ids = {id(parameter) for parameter in group["params"]}
        if not (group_ids & new_ids):
            continue
        found_ids.update(group_ids & new_ids)
        expected = {
            "lr": AUXILIARY_OPTIMIZER_ALGORITHM["learning_rate"],
            "weight_decay": AUXILIARY_OPTIMIZER_ALGORITHM["weight_decay"],
            "betas": tuple(AUXILIARY_OPTIMIZER_ALGORITHM["betas"]),
            "eps": AUXILIARY_OPTIMIZER_ALGORITHM["eps"],
        }
        if any(group.get(name) != value for name, value in expected.items()):
            raise ValueError("Joint AdamW settings differ from the sealed source-read auxiliary algorithm.")
    if found_ids != new_ids:
        raise ValueError("Every new source-read parameter must belong to the inherited joint AdamW optimizer.")


def _calibrate_thresholds(
    model: nn.Module,
    controller: SourceReadControllerV1,
    direct_reader: nn.Module,
    cases: Sequence[_TeacherCase],
    *,
    device: torch.device,
    dtype: torch.dtype,
    context_scales: Tensor,
    heating_envelope: float,
) -> tuple[tuple[float, float], dict[str, Any]]:
    score_blocks: list[Tensor] = []
    near_blocks: list[Tensor] = []
    union_blocks: list[Tensor] = []
    with torch.no_grad():
        for case in cases:
            data = _case_to_device(case, device, dtype)
            q = data["query_states"]
            s = data["module_states"]
            descriptor = data["descriptor"]
            alpha = data["alpha"]
            fine_messages = data["fine_messages"]
            fine_k = data["thermal_k"]
            assert all(isinstance(value, Tensor) for value in (q, s, descriptor, alpha, fine_messages))
            completion_messages = None
            completion_k = None
            if controller.completion_enabled:
                completion_messages, completion_k = _completion_outputs(controller, direct_reader, q, s)
            targets = _utility_targets(
                controller, alpha, fine_messages, fine_k,
                context_scales=context_scales,
                heating_envelope=heating_envelope,
                temperature_scale=1.0,
                completion_messages=completion_messages, completion_k=completion_k,
            )
            scores = controller(
                q, s, descriptor
            ).sum(dim=-1)
            valid = _valid_pair_mask(data)
            near = _near_gate(data)
            positive = targets.sum(dim=-1) > torch.finfo(targets.dtype).eps
            union = valid & (positive | (near > 0))
            score_blocks.append(scores.reshape(-1).cpu())
            near_blocks.append(((near > 0) & valid).reshape(-1).cpu())
            union_blocks.append(union.reshape(-1).cpu())
    if not score_blocks:
        raise ValueError("No TRAIN rows were available for fixed operating-threshold calibration.")
    scores = torch.cat(score_blocks).to(torch.float64)
    near = torch.cat(near_blocks).to(torch.bool)
    union = torch.cat(union_blocks).to(torch.bool)
    scores = scores[union]
    near = near[union]
    if scores.numel() == 0:
        raise ValueError("No positive or near-protected TRAIN source pairs were available for calibration.")
    thresholds: list[float] = []
    target_union_rows: list[int] = []
    for target_fraction in (0.75, 0.50):
        desired = max(1, int(math.ceil(target_fraction * scores.numel())))
        target_union_rows.append(desired)
        protected_count = int(near.sum().item())
        if protected_count >= desired:
            tau = float(scores.max().item()) + max(1.0e-6, abs(float(scores.max().item())) * 1.0e-6)
        else:
            remaining = min(scores.numel() - protected_count, desired - protected_count)
            candidates = scores[~near].sort(descending=True).values
            pivot = float(candidates[max(0, remaining - 1)].item())
            tau = max(torch.finfo(torch.float32).tiny, math.nextafter(pivot, -math.inf))
        thresholds.append(float(tau))
    # Scores are nonnegative and the nominal 75% retention threshold cannot
    # exceed the nominal 50% threshold except where near protection dominates;
    # preserve target order so the persisted operating point stays auditable.
    if thresholds[0] > thresholds[1]:
        raise ValueError("TRAIN calibration inverted the fixed 75%/50% threshold ordering.")

    heads = controller.heads
    physical_pair_count = 0
    union_pair_count = int(union.sum().item())
    near_union_count = int(near.sum().item())
    active_union = [0, 0]
    active_physical = [0, 0]
    flow_diff_sq = torch.zeros((2, heads), dtype=torch.float64)
    flow_reference_sq = torch.zeros((2, heads), dtype=torch.float64)
    thermal_endpoint_diff_sq = [0.0, 0.0]
    thermal_endpoint_reference_sq = [0.0, 0.0]
    thermal_kernel_diff_sq = [0.0, 0.0]
    thermal_kernel_reference_sq = [0.0, 0.0]
    query_total = 0

    with torch.no_grad():
        for case in cases:
            data = _case_to_device(case, device, dtype)
            q, s = data["query_states"], data["module_states"]
            descriptor, alpha = data["descriptor"], data["alpha"]
            fine_messages, fine_k = data["fine_messages"], data["thermal_k"]
            assert all(isinstance(value, Tensor) for value in (q, s, descriptor, alpha, fine_messages))
            valid = _valid_pair_mask(data)
            query_mask = data["query_mask"]
            assert isinstance(query_mask, Tensor)
            near_case = _near_gate(data)
            completion_messages = None
            completion_k = None
            if controller.completion_enabled:
                completion_messages, completion_k = _completion_outputs(controller, direct_reader, q, s)
            if completion_messages is None:
                completion_messages = torch.zeros_like(fine_messages)
            source_alpha = alpha * valid[..., None].to(alpha.dtype)
            source_mask = valid[..., None].to(fine_messages.dtype)
            head_width = controller.width // controller.heads
            fine_heads = (fine_messages * source_mask).reshape(
                case.batch_size, case.query_count, case.source_count, heads, head_width
            )
            base_heads = (completion_messages * source_mask).reshape_as(fine_heads)
            residual_heads = fine_heads - base_heads
            dense_context = (source_alpha[..., None] * fine_heads).sum(dim=2)
            base_context = (source_alpha[..., None] * base_heads).sum(dim=2)
            valid_queries = query_mask[..., None, None]
            physical_pair_count += int(valid.sum().item())
            query_total += int(query_mask.sum().item())

            scores_case = controller(q, s, descriptor).sum(dim=-1)
            case_union = valid & (( _utility_targets(
                controller, alpha, fine_messages, fine_k,
                context_scales=context_scales, heating_envelope=heating_envelope,
                temperature_scale=1.0,
                completion_messages=completion_messages if controller.completion_enabled else None,
                completion_k=completion_k,
            ).sum(dim=-1) > torch.finfo(fine_messages.dtype).eps) | (near_case > 0))
            for point, threshold in enumerate(thresholds):
                utility_gate = smooth_threshold_gate(scores_case, threshold)
                gate = combine_source_gates(utility_gate, near_case)
                gate = gate * valid.to(gate.dtype)
                active_union[point] += int(((gate > 0) & case_union).sum().item())
                active_physical[point] += int((gate > 0).sum().item())
                sparse_context = base_context + (
                    source_alpha[..., None] * gate[..., None, None] * residual_heads
                ).sum(dim=2)
                context_error = (sparse_context - dense_context).to(torch.float64)
                context_reference = dense_context.to(torch.float64)
                masked_error = context_error * valid_queries
                masked_reference = context_reference * valid_queries
                # The screen keeps only tiny FP64 totals on the host, even
                # when the controller and teacher reductions execute on CUDA.
                flow_diff_sq[point] += masked_error.square().sum(dim=(0, 1, 3)).cpu()
                flow_reference_sq[point] += masked_reference.square().sum(dim=(0, 1, 3)).cpu()

                if controller.thermal:
                    assert fine_k is not None
                    if completion_k is None:
                        base_k = torch.zeros_like(fine_k)
                    else:
                        base_k = completion_k
                    residual_k = (fine_k - base_k) * valid.to(fine_k.dtype)
                    candidate_k = base_k * valid.to(base_k.dtype) + gate * residual_k
                    dense_k = fine_k * valid.to(fine_k.dtype)
                    k_error = (candidate_k - dense_k).to(torch.float64)
                    k_reference = dense_k.to(torch.float64)
                    k_mask = valid.to(torch.float64)
                    thermal_kernel_diff_sq[point] += float((k_error.square() * k_mask).sum().item())
                    thermal_kernel_reference_sq[point] += float((k_reference.square() * k_mask).sum().item())
                    dense_endpoint = (dense_k * heating_envelope).sum(dim=-1)
                    candidate_endpoint = (candidate_k * heating_envelope).sum(dim=-1)
                    endpoint_error = (candidate_endpoint - dense_endpoint).to(torch.float64)
                    endpoint_reference = dense_endpoint.to(torch.float64)
                    qvalid = query_mask.to(torch.float64)
                    thermal_endpoint_diff_sq[point] += float((endpoint_error.square() * qvalid).sum().item())
                    thermal_endpoint_reference_sq[point] += float((endpoint_reference.square() * qvalid).sum().item())

    union_retention = [count / union_pair_count for count in active_union]
    physical_retention = [count / physical_pair_count for count in active_physical]
    head_relative = [
        [_relative_rms(float(flow_diff_sq[point, head]), float(flow_reference_sq[point, head]))
         for head in range(heads)]
        for point in range(2)
    ]
    flow_relative = [_relative_rms(float(flow_diff_sq[point].sum()),
                                   float(flow_reference_sq[point].sum())) for point in range(2)]
    if controller.thermal:
        thermal_relative: list[float | None] = [
            _relative_rms(thermal_endpoint_diff_sq[point], thermal_endpoint_reference_sq[point])
            for point in range(2)
        ]
        kernel_relative: list[float | None] = [
            _relative_rms(thermal_kernel_diff_sq[point], thermal_kernel_reference_sq[point])
            for point in range(2)
        ]
    else:
        thermal_relative = [None, None]
        kernel_relative = [None, None]

    def finite_nonworse(candidate: int, reference: int) -> bool:
        values_candidate = head_relative[candidate] + [flow_relative[candidate]]
        values_reference = head_relative[reference] + [flow_relative[reference]]
        if controller.thermal:
            values_candidate += [float(thermal_relative[candidate]), float(kernel_relative[candidate])]
            values_reference += [float(thermal_relative[reference]), float(kernel_relative[reference])]
        return all(math.isfinite(value) for value in values_candidate) and all(
            after <= before + 1.0e-12 for after, before in zip(values_candidate, values_reference)
        )

    useful_75 = active_physical[0] < physical_pair_count
    if useful_75:
        primary, reason = 0, "least_aggressive_useful_75"
    elif active_physical[1] < active_physical[0] and finite_nonworse(1, 0):
        primary, reason = 1, "75_no_savings_use_50_without_worse_screen"
    else:
        primary, reason = 0, "no_useful_point_default75"
    screen = {
        "schema": "source_read_train_fidelity_screen_v1",
        "partition": "TRAIN",
        "scope": "bounded_dense_teacher_rows_at_most_256_per_train_case",
        "pair_denominator": "positive_or_near_protected_physical_pair_union",
        "thresholds": [float(value) for value in thresholds],
        "target_union_retention": [0.75, 0.50],
        "target_union_pair_counts": target_union_rows,
        "union_pair_count": union_pair_count,
        "physical_pair_count": physical_pair_count,
        "query_count": query_total,
        "near_protected_union_pair_count": near_union_count,
        "actual_union_pair_counts": active_union,
        "actual_physical_pair_counts": active_physical,
        "actual_union_retention": union_retention,
        "actual_physical_pair_retention": physical_retention,
        "target_infeasible_due_to_near_protection": [
            near_union_count > target_union_rows[point] for point in range(2)
        ],
        "actual_target_overrun_pair_counts": [
            max(0, active_union[point] - target_union_rows[point]) for point in range(2)
        ],
        "flow_direct_context_relative_rms": flow_relative,
        "flow_direct_context_head_relative_rms": head_relative,
        "thermal_positive_H_endpoint_relative_rms": thermal_relative if controller.thermal else None,
        "thermal_kernel_relative_rms": kernel_relative if controller.thermal else None,
        "thermal_temperature_scale": 1.0 if controller.thermal else None,
        "thermal_target_scale_convention": (
            "native_core_K_is_TRAIN_standardized_temperature_per_heat; auxiliary_sT=1"
            if controller.thermal else None
        ),
        "primary_operating_point": primary,
        "selection_reason": reason,
    }
    return (float(thresholds[0]), float(thresholds[1])), screen


class SourceReadAuxiliaryManagerV1:
    """Fixed-TRAIN teacher fitting and bounded current-model label refreshes."""

    def __init__(
        self,
        controller: SourceReadControllerV1,
        *,
        variant: str,
        train_case_keys: Sequence[Any],
        seed: int,
        sampling_dataset_id: str,
        sampling_version: str = SamplingKey.CASE_EPOCH_VERSION,
        teacher_trace_factory: Callable[[Any, nn.Module, Any, SamplingKey, int], Mapping[str, Any]] | None = None,
        physical_loss_factory: Callable[[Any, nn.Module, Any, SamplingKey], Tensor] | None = None,
        initial_query_rows: int = MAX_INITIAL_QUERY_ROWS,
        refresh_query_rows: int = MAX_REFRESH_QUERY_ROWS,
        refresh_every: int = 8,
        teacher_row_budget_fraction: float = 0.05,
        ram_cache_budget_bytes: int = DEFAULT_RAM_CACHE_BUDGET_BYTES,
        parent_sha256: str | None = None,
    ) -> None:
        if variant not in _ALLOWED_VARIANTS:
            raise ValueError("Sparse-child auxiliary variant must be S-Utility or S-Complete.")
        if controller.completion_enabled != (variant == "S-Complete"):
            raise ValueError("Source-read variant and controller completion capacity disagree.")
        keys = tuple(train_case_keys)
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("Source-read auxiliary calibration requires a nonempty unique TRAIN cohort.")
        if type(seed) is not int or seed < 0 or not sampling_dataset_id:
            raise ValueError("Source-read auxiliary sampling requires a nonnegative seed and dataset identity.")
        if sampling_version not in SamplingKey.SUPPORTED_VERSIONS:
            raise ValueError("Unsupported source-read auxiliary sampling version.")
        if type(initial_query_rows) is not int or not 1 <= initial_query_rows <= MAX_INITIAL_QUERY_ROWS:
            raise ValueError("Initial source-read teacher rows must be in [1,256].")
        if type(refresh_query_rows) is not int or not 1 <= refresh_query_rows <= MAX_REFRESH_QUERY_ROWS:
            raise ValueError("Refresh source-read teacher rows must be in [1,128].")
        if type(refresh_every) is not int or refresh_every < 1:
            raise ValueError("Source-read teacher refresh cadence must be a positive update count.")
        if (not math.isfinite(teacher_row_budget_fraction) or teacher_row_budget_fraction <= 0
                or teacher_row_budget_fraction >= 0.05 + 1e-15):
            raise ValueError("Co-adaptation teacher-row budget must be positive and at most 5%.")
        if type(ram_cache_budget_bytes) is not int or ram_cache_budget_bytes < 1:
            raise ValueError("In-memory teacher cache needs a positive byte cap.")
        self.controller = controller
        self.variant = variant
        self.train_case_keys = keys
        self.seed = seed
        self.sampling_dataset_id = sampling_dataset_id
        self.sampling_version = sampling_version
        self.teacher_trace_factory = teacher_trace_factory
        self.physical_loss_factory = physical_loss_factory
        self.initial_query_rows = initial_query_rows
        self.refresh_query_rows = refresh_query_rows
        self.refresh_every = refresh_every
        self.teacher_row_budget_fraction = float(teacher_row_budget_fraction)
        self.ram_cache_budget_bytes = ram_cache_budget_bytes
        if parent_sha256 is not None and (
            len(parent_sha256) != 64 or any(character not in "0123456789abcdef" for character in parent_sha256)
        ):
            raise ValueError("Source-read auxiliary parent binding must be a lowercase SHA256.")
        self.parent_sha256 = parent_sha256
        self.source_epoch = 500
        self.context_scales: Tensor | None = None
        self.heating_envelope = 1.0
        self.temperature_scale = 1.0
        self.auxiliary_coefficient: float | None = None
        self.gradient_calibration: dict[str, float | str] | None = None
        self.primary_threshold_index = 0
        self.threshold_train_density: tuple[float, float] | None = None
        self.train_fidelity_screen: dict[str, Any] | None = None
        self.initial_teacher_rows = 0
        self.initial_teacher_present_rows = 0
        self.coadaptation_teacher_rows = 0
        self.coadaptation_teacher_present_rows = 0
        self.last_ordinary_update_index = 0
        self.last_absolute_epoch = 500
        self.last_primary_fine_rows = 0
        self.max_source_count = 0
        self.refresh_case_key: Any | None = None
        self._refresh_cache: _TeacherCase | None = None
        self.last_loss_components: dict[str, float] = {}
        self.last_gradient_balance: dict[str, float | None] | None = None
        self.last_refresh: dict[str, Any] = {"refreshed": False, "reason": "not_started"}

    def _new_sampling_key(self, *, epoch: int, update_index: int, stage: str) -> SamplingKey:
        return _make_sampling_key(
            seed=self.seed, epoch=epoch, update_index=update_index, stage=stage,
            variant=self.variant, sampling_version=self.sampling_version,
            dataset_id=self.sampling_dataset_id,
        )

    def _resolve_envelope(self, provider) -> float:
        if not self.controller.thermal:
            return 1.0
        method = getattr(provider, "source_read_positive_train_envelope", None)
        if not callable(method):
            raise ValueError("Thermal provider must declare its positive TRAIN source-read envelope.")
        value = method(self.train_case_keys)
        envelope = _finite_positive_scalar(value, "TRAIN heating envelope")
        self.heating_envelope = envelope
        return envelope

    def _collect_initial_teacher_cases(
        self, model: nn.Module, provider,
    ) -> list[_TeacherCase]:
        cases: list[_TeacherCase] = []
        used_bytes = 0
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for index, case_key in enumerate(self.train_case_keys):
                    key = self._new_sampling_key(
                        epoch=500, update_index=index, stage="source_read_auxiliary_initial"
                    )
                    trace = _factory_trace(
                        provider, model, case_key, key, self.initial_query_rows,
                        self.teacher_trace_factory,
                    )
                    case = _normalize_teacher_case(
                        case_key, trace, self.controller, max_query_rows=self.initial_query_rows
                    )
                    used_bytes += case.byte_count
                    if used_bytes > self.ram_cache_budget_bytes:
                        raise MemoryError(
                            "TRAIN-only source-read teacher cache exceeded its declared in-memory byte cap; "
                            "reduce the bounded cohort/query rows rather than writing a cache."
                        )
                    self.initial_teacher_rows += case.fine_executed_rows
                    self.initial_teacher_present_rows += case.fine_present_rows
                    self.max_source_count = max(self.max_source_count, case.source_count)
                    if case.pair_rows:
                        cases.append(case)
        finally:
            model.train(was_training)
        if not cases or self.initial_teacher_rows <= 0:
            raise ValueError("TRAIN source-read teacher cohort contains no valid source/query pairs.")
        return cases

    def _calibrate_coefficient(
        self, model: nn.Module, provider, direct_reader: nn.Module,
        case: _TeacherCase, added_parameters: Mapping[str, nn.Parameter],
    ) -> dict[str, float | str]:
        parameters = dict(model.named_parameters())
        inherited = [(name, parameter) for name, parameter in parameters.items()
                     if name not in added_parameters]
        flags = {name: parameter.requires_grad for name, parameter in parameters.items()}
        for parameter in parameters.values():
            parameter.requires_grad_(True)
        try:
            key = self._new_sampling_key(
                epoch=500, update_index=0, stage="source_read_native_gradient_calibration"
            )
            physical_loss = _factory_physical_loss(
                provider, model, case.case_key, key, self.physical_loss_factory
            )
            physical_gradients = torch.autograd.grad(
                physical_loss, [parameter for _, parameter in inherited],
                allow_unused=True, retain_graph=False, create_graph=False,
            )
            physical_norm = _gradient_norm(physical_gradients)
            data = _case_to_device(case, *_device_dtype(self.controller))
            scorer_loss, completion_loss, _ = _scorer_and_completion_losses(
                self.controller,
                direct_reader,
                data,
                context_scales=self.context_scales,
                heating_envelope=self.heating_envelope,
                temperature_scale=self.temperature_scale,
            )
            auxiliary_loss = scorer_loss + completion_loss
            auxiliary_gradients = torch.autograd.grad(
                auxiliary_loss, list(added_parameters.values()),
                allow_unused=True, retain_graph=False, create_graph=False,
            )
            auxiliary_norm = _gradient_norm(auxiliary_gradients)
        finally:
            for name, parameter in parameters.items():
                parameter.requires_grad_(flags[name])
        if not math.isfinite(physical_norm) or physical_norm < 0 or not math.isfinite(auxiliary_norm) or auxiliary_norm <= 0:
            raise ValueError("Source-read fixed TRAIN gradient calibration produced invalid norms.")
        coefficient = min(1.0, 0.05 * physical_norm / auxiliary_norm)
        weighted_share = coefficient * auxiliary_norm / physical_norm if physical_norm > 0 else 0.0
        calibration: dict[str, float | str] = {
            "schema": "source_read_gradient_calibration_v1",
            "partition": "TRAIN",
            "physical_all_inherited_parameter_gradient_l2_norm": physical_norm,
            "auxiliary_all_new_parameter_gradient_l2_norm": auxiliary_norm,
            "fixed_auxiliary_coefficient": coefficient,
            "realized_weighted_auxiliary_to_physical_gradient_share": weighted_share,
            "maximum_gradient_share": 0.05,
        }
        if not all(math.isfinite(float(value)) for key, value in calibration.items()
                   if key not in ("schema", "partition")):
            raise ValueError("Source-read gradient calibration receipt contains nonfinite values.")
        if weighted_share > 0.05 + 1.0e-12:
            raise RuntimeError("Calibrated source-read gradient share exceeds the fixed 5% limit.")
        self.gradient_calibration = calibration
        return calibration

    def _select_refresh_case(self, epoch: int, update_index: int) -> Any:
        position = _stable_int(self.seed, self.sampling_dataset_id, epoch, update_index, "refresh")
        return self.train_case_keys[position % len(self.train_case_keys)]

    def fit_initial(
        self,
        model: nn.Module,
        provider,
        optimizer: torch.optim.Optimizer,
        *,
        added_parameter_names: Sequence[str],
        train_case_keys: Sequence[Any],
        max_updates: int,
        parent_sha256: str,
        source_epoch: int,
    ) -> dict[str, Any]:
        """Fit and TRAIN-calibrate the attached controller with the joint optimizer."""
        if tuple(train_case_keys) != self.train_case_keys:
            raise ValueError("Auxiliary callback TRAIN cases differ from the manager's sealed cohort.")
        if type(max_updates) is not int or not 1 <= max_updates <= MAX_AUXILIARY_UPDATES:
            raise ValueError("Source-read auxiliary optimizer updates must be in [1,256].")
        if type(source_epoch) is not int or source_epoch != 500:
            raise ValueError("Source-read auxiliary fitting must attach to the exact dense e500 parent.")
        if (not isinstance(parent_sha256, str) or len(parent_sha256) != 64
                or any(character not in "0123456789abcdef" for character in parent_sha256)):
            raise ValueError("Source-read auxiliary fitting requires the sealed parent checkpoint SHA256.")
        if self.parent_sha256 is not None and self.parent_sha256 != parent_sha256:
            raise ValueError("Source-read manager is bound to a different dense e500 parent checkpoint.")
        if self.auxiliary_coefficient is not None or self._refresh_cache is not None:
            raise ValueError("Source-read initial auxiliary fitting cannot be repeated on an attached child.")
        self.parent_sha256 = parent_sha256
        core, direct_reader, controller = _model_parts(model)
        if controller is not self.controller:
            raise ValueError("Source-read manager is bound to a different model controller.")
        named_controller = _named_controller_parameters(model, controller)
        if set(named_controller) != set(added_parameter_names):
            raise ValueError("Declared new source-read parameter names differ from the attached controller.")
        _validate_auxiliary_optimizer(optimizer, named_controller)
        if any(not parameter.requires_grad for parameter in named_controller.values()):
            raise ValueError("Engine must enable all new source-read parameters for auxiliary fitting.")
        if any(parameter.requires_grad for name, parameter in model.named_parameters()
               if name not in named_controller):
            raise ValueError("Inherited physical parameters must be frozen during initial auxiliary fitting.")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise ValueError("Initial source-read auxiliary fitting requires clean parameter gradients.")
        buffers_before = {
            name: value.detach().cpu().clone() for name, value in model.named_buffers()
        }

        envelope = self._resolve_envelope(provider)
        cases = self._collect_initial_teacher_cases(model, provider)
        self.context_scales = _context_scales(cases, controller)
        device, dtype = _device_dtype(controller)
        # Fit on deterministic cohort cycling. Each dense teacher case is
        # bounded to 256 query rows and held only in CPU RAM for this callback.
        optimizer_updates = 0
        used_case_keys: set[Any] = set()
        initial_training = model.training
        model.train(True)
        try:
            for update in range(max_updates):
                case = cases[update % len(cases)]
                data = _case_to_device(case, device, dtype)
                scorer_loss, completion_loss, components = _scorer_and_completion_losses(
                    controller,
                    direct_reader,
                    data,
                    context_scales=self.context_scales,
                    heating_envelope=envelope,
                    temperature_scale=self.temperature_scale,
                )
                loss = scorer_loss + completion_loss
                if not torch.isfinite(loss):
                    raise ValueError("Source-read auxiliary fitting produced a nonfinite loss.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(parameter.grad is None for parameter in named_controller.values()):
                    missing = sorted(name for name, parameter in named_controller.items()
                                     if parameter.grad is None)
                    raise ValueError(f"Source-read auxiliary fit left declared parameters unused: {missing}.")
                if any(parameter.grad is not None for name, parameter in model.named_parameters()
                       if name not in named_controller):
                    raise ValueError("Auxiliary gradients escaped into inherited physical parameters.")
                nonfinite_gradients = sorted(
                    name for name, parameter in named_controller.items()
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                )
                if nonfinite_gradients:
                    raise ValueError(
                        "Source-read auxiliary fitting produced nonfinite new-parameter gradients: "
                        f"{nonfinite_gradients}."
                    )
                torch.nn.utils.clip_grad_norm_(
                    list(named_controller.values()),
                    max_norm=float(AUXILIARY_OPTIMIZER_ALGORITHM["gradient_clip_norm"]),
                    error_if_nonfinite=True,
                )
                optimizer.step()
                optimizer_updates += 1
                used_case_keys.add(case.case_key)
                self.last_loss_components = {
                    "scorer": float(components["scorer"].item()),
                    "completion": float(components["completion"].item()),
                    "total_unweighted": float(components["total_unweighted"].item()),
                }
        finally:
            model.train(initial_training)

        thresholds, screen = _calibrate_thresholds(
            model, controller, direct_reader, cases, device=device, dtype=dtype,
            context_scales=self.context_scales, heating_envelope=envelope,
        )
        controller.install_calibration_(
            thresholds=thresholds,
            context_scales=self.context_scales,
            heating_envelope=envelope,
        )
        controller.mark_trained_()
        controller.freeze_metadata_()
        self.primary_threshold_index = int(screen["primary_operating_point"])
        controller.primary_operating_point.fill_(self.primary_threshold_index)
        self.train_fidelity_screen = screen
        self.gradient_calibration = self._calibrate_coefficient(
            model, provider, direct_reader, cases[0], named_controller
        )
        self.auxiliary_coefficient = float(self.gradient_calibration["fixed_auxiliary_coefficient"])
        self.threshold_train_density = (0.75, 0.50)
        self.last_absolute_epoch = source_epoch
        self.last_ordinary_update_index = 0
        self.last_primary_fine_rows = 0

        # Keep one case with no more than 128 query rows for updates before the
        # first scheduled current-model refresh. Release the rest immediately.
        refresh_index = _stable_int(self.seed, self.sampling_dataset_id, "initial_refresh") % len(cases)
        refresh_case = cases[refresh_index]
        self._refresh_cache = _truncate_case(refresh_case, self.refresh_query_rows)
        self.refresh_case_key = refresh_case.case_key
        del cases

        updated_names: list[str] = []
        steps: dict[str, int] = {}
        for name, parameter in named_controller.items():
            state = optimizer.state.get(parameter, {})
            step_value = state.get("step", 0)
            step_count = int(step_value.item()) if isinstance(step_value, Tensor) else int(step_value)
            if step_count > 0:
                updated_names.append(name)
                steps[name] = step_count
        updated_names.sort()
        if not updated_names:
            raise ValueError("No source-read controller parameter received an auxiliary AdamW update.")
        buffers_after = dict(model.named_buffers())
        if set(buffers_before) != set(buffers_after):
            raise ValueError("Initial source-read auxiliary fit changed the model buffer inventory.")
        buffer_changed_names = []
        for name, before in buffers_before.items():
            after = buffers_after[name].detach().cpu()
            if before.shape != after.shape or before.dtype != after.dtype:
                raise ValueError(f"Initial source-read auxiliary fit replaced model buffer {name!r}.")
            if not torch.equal(before, after):
                if _SOURCE_READ_PARAMETER_MARKER not in name:
                    raise ValueError(f"Initial source-read auxiliary fit changed unrelated buffer {name!r}.")
                buffer_changed_names.append(name)
        buffer_changed_names.sort()
        if not buffer_changed_names:
            raise ValueError("Source-read initial fit did not install its fixed TRAIN calibration buffers.")
        fit_case_keys = [key for key in self.train_case_keys if key in used_case_keys]
        if not fit_case_keys:
            raise ValueError("No TRAIN case contributed to source-read auxiliary fitting.")
        # Provider may be a wrapper with its own `core`; the object identity
        # lookup above is authoritative and avoids modifying native state.
        setattr(provider, "source_read_auxiliary_manager", self)
        receipt = {
            "schema": AUXILIARY_FIT_SCHEMA,
            "partition": "TRAIN",
            "source_checkpoint_sha256": parent_sha256,
            "source_epoch": source_epoch,
            "train_case_keys": list(self.train_case_keys),
            "train_case_count": len(self.train_case_keys),
            "auxiliary_fit_case_keys": fit_case_keys,
            "auxiliary_fit_case_count": len(fit_case_keys),
            "auxiliary_fit_case_share": float(len(fit_case_keys) / len(self.train_case_keys)),
            "teacher_rows": int(self.initial_teacher_rows),
            "teacher_present_rows": int(self.initial_teacher_present_rows),
            "optimizer_updates": int(optimizer_updates),
            "updated_parameter_names": updated_names,
            "changed_buffer_names": buffer_changed_names,
            "optimizer_steps_by_parameter": steps,
            "optimizer_algorithm": dict(AUXILIARY_OPTIMIZER_ALGORITHM),
            "train_fidelity_screen": copy.deepcopy(screen),
            "gradient_calibration": copy.deepcopy(self.gradient_calibration),
        }
        return receipt

    def initialize_inherited_exposure(
        self,
        *,
        ordinary_update_index: int,
        primary_fine_rows: int,
        absolute_epoch: int = 500,
    ) -> None:
        """Seed the sparse child's clocks with its exact inherited dense prefix."""
        if (type(absolute_epoch) is not int or absolute_epoch != 500
                or self.last_absolute_epoch != 500 or self.last_ordinary_update_index != 0
                or self.last_primary_fine_rows != 0 or self.coadaptation_teacher_rows != 0):
            raise ValueError("Inherited source-read exposure can only initialize a fresh e500 manager.")
        if type(ordinary_update_index) is not int or ordinary_update_index < 1:
            raise ValueError("Inherited dense prefix must contain at least one ordinary optimizer update.")
        if type(primary_fine_rows) is not int or primary_fine_rows < 1:
            raise ValueError("Inherited dense prefix must contain measured primary fine rows.")
        self.last_ordinary_update_index = ordinary_update_index
        self.last_primary_fine_rows = primary_fine_rows

    def record_primary_fine_rows(self, primary_fine_rows: int) -> None:
        """Persist the work just completed so an epoch-boundary resume is exact."""
        if type(primary_fine_rows) is not int or primary_fine_rows < self.last_primary_fine_rows:
            raise ValueError("Cumulative primary fine-row exposure cannot move backwards.")
        self.last_primary_fine_rows = primary_fine_rows

    def record_gradient_balance(self, *, native_norm: float, weighted_auxiliary_norm: float) -> dict[str, float | None]:
        """Store the current update's native/weighted-auxiliary gradient share."""
        if (not math.isfinite(native_norm) or native_norm < 0
                or not math.isfinite(weighted_auxiliary_norm) or weighted_auxiliary_norm < 0):
            raise ValueError("Source-read gradient-balance telemetry must contain finite nonnegative norms.")
        result: dict[str, float | None] = {
            "native_all_parameter_gradient_l2_norm": float(native_norm),
            "weighted_auxiliary_new_parameter_gradient_l2_norm": float(weighted_auxiliary_norm),
            "weighted_auxiliary_to_native_gradient_share": (
                float(weighted_auxiliary_norm / native_norm) if native_norm > 0 else None
            ),
        }
        self.last_gradient_balance = result
        return dict(result)

    def _current_auxiliary_loss(
        self, model: nn.Module, case: _TeacherCase,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        _, direct_reader, controller = _model_parts(model)
        if controller is not self.controller:
            raise ValueError("Source-read manager is bound to a different controller than the model.")
        device, dtype = _device_dtype(controller)
        data = _case_to_device(case, device, dtype)
        scorer_loss, completion_loss, components = _scorer_and_completion_losses(
            controller,
            direct_reader,
            data,
            context_scales=controller.context_scales,
            heating_envelope=self.heating_envelope,
            temperature_scale=self.temperature_scale,
        )
        total = scorer_loss + completion_loss
        assert self.auxiliary_coefficient is not None
        return total * self.auxiliary_coefficient, components

    def before_update(
        self,
        model: nn.Module,
        provider,
        *,
        absolute_epoch: int,
        ordinary_update_index: int,
        primary_fine_rows: int,
    ) -> dict[str, Any]:
        """Refresh one bounded TRAIN teacher buffer on the fixed eight-step clock."""
        if self._refresh_cache is None or self.auxiliary_coefficient is None:
            raise ValueError("Source-read auxiliary manager has not completed its initial fit.")
        if type(absolute_epoch) is not int or absolute_epoch < 501:
            raise ValueError("Source-read current-model refreshes begin at absolute epoch 501.")
        if type(ordinary_update_index) is not int or ordinary_update_index < 1:
            raise ValueError("Source-read refresh needs a positive ordinary child-update index.")
        if type(primary_fine_rows) is not int or primary_fine_rows < 0:
            raise ValueError("Cumulative primary fine-row exposure must be a nonnegative integer.")
        if ordinary_update_index < self.last_ordinary_update_index:
            raise ValueError("Source-read ordinary update clock moved backwards across resume.")
        if absolute_epoch < self.last_absolute_epoch or primary_fine_rows < self.last_primary_fine_rows:
            raise ValueError("Source-read epoch or primary fine-row exposure moved backwards.")
        if ordinary_update_index == self.last_ordinary_update_index:
            return dict(self.last_refresh)
        self.last_ordinary_update_index = ordinary_update_index
        self.last_absolute_epoch = absolute_epoch
        self.last_primary_fine_rows = primary_fine_rows
        if ordinary_update_index % self.refresh_every:
            self.last_refresh = {"refreshed": False, "reason": "cadence"}
            return dict(self.last_refresh)
        worst_case_rows = self.refresh_query_rows * max(1, self.max_source_count)
        budget_limit = self.teacher_row_budget_fraction * primary_fine_rows
        if self.coadaptation_teacher_rows + worst_case_rows >= budget_limit:
            self.last_refresh = {"refreshed": False, "reason": "five_percent_teacher_row_budget"}
            return dict(self.last_refresh)
        case_key = self._select_refresh_case(absolute_epoch, ordinary_update_index)
        key = self._new_sampling_key(
            epoch=absolute_epoch,
            update_index=ordinary_update_index,
            stage="source_read_auxiliary_refresh",
        )
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                trace = _factory_trace(
                    provider, model, case_key, key, self.refresh_query_rows,
                    self.teacher_trace_factory,
                )
                _, _, controller = _model_parts(model)
                case = _normalize_teacher_case(
                    case_key, trace, controller, max_query_rows=self.refresh_query_rows
                )
        finally:
            model.train(was_training)
        actual_rows = case.fine_executed_rows
        actual_present_rows = case.fine_present_rows
        if self.coadaptation_teacher_rows + actual_rows >= budget_limit:
            # The preflight uses the maximum source count observed over the
            # fixed cohort; reject a malformed/provider-shifted trace before
            # installing it or counting it as an accepted refresh.
            raise ValueError("Source-read refresh exceeded its preflight 5% dense-teacher row budget.")
        if case.source_count > self.max_source_count:
            raise ValueError("Source-read refresh introduced a source count outside the sealed TRAIN cohort.")
        self._refresh_cache = case
        self.refresh_case_key = case_key
        self.coadaptation_teacher_rows += actual_rows
        self.coadaptation_teacher_present_rows += actual_present_rows
        self.last_refresh = {
            "refreshed": True,
            "reason": "cadence",
            "case_key": case_key,
            "epoch": absolute_epoch,
            "ordinary_update_index": ordinary_update_index,
            "teacher_rows": actual_rows,
            "teacher_present_rows": actual_present_rows,
            "cumulative_teacher_rows": self.coadaptation_teacher_rows,
            "cumulative_teacher_present_rows": self.coadaptation_teacher_present_rows,
            "primary_fine_rows": primary_fine_rows,
            "teacher_row_share": float(self.coadaptation_teacher_rows / primary_fine_rows)
            if primary_fine_rows else 0.0,
        }
        return dict(self.last_refresh)

    def auxiliary_loss(self, model: nn.Module) -> Tensor:
        if self._refresh_cache is None:
            raise ValueError("Source-read auxiliary loss requires its TRAIN refresh buffer.")
        result, components = self._current_auxiliary_loss(model, self._refresh_cache)
        self.last_loss_components = {
            "scorer": float(components["scorer"].item()),
            "completion": float(components["completion"].item()),
            "total_unweighted": float(components["total_unweighted"].item()),
            "coefficient": float(self.auxiliary_coefficient or 0.0),
        }
        return result

    def state_dict(self) -> dict[str, Any]:
        """Return compact manager-only continuation state, separate from provider state."""
        return {
            "schema": STATE_SCHEMA,
            "variant": self.variant,
            "train_case_keys": list(self.train_case_keys),
            "seed": self.seed,
            "sampling_dataset_id": self.sampling_dataset_id,
            "sampling_version": self.sampling_version,
            "initial_query_rows": self.initial_query_rows,
            "refresh_query_rows": self.refresh_query_rows,
            "refresh_every": self.refresh_every,
            "teacher_row_budget_fraction": self.teacher_row_budget_fraction,
            "ram_cache_budget_bytes": self.ram_cache_budget_bytes,
            "parent_sha256": self.parent_sha256,
            "source_epoch": self.source_epoch,
            "context_scales": None if self.context_scales is None else self.context_scales.clone(),
            "heating_envelope": self.heating_envelope,
            "temperature_scale": self.temperature_scale,
            "auxiliary_coefficient": self.auxiliary_coefficient,
            "gradient_calibration": copy.deepcopy(self.gradient_calibration),
            "primary_threshold_index": self.primary_threshold_index,
            "threshold_train_density": self.threshold_train_density,
            "train_fidelity_screen": copy.deepcopy(self.train_fidelity_screen),
            "initial_teacher_rows": self.initial_teacher_rows,
            "initial_teacher_present_rows": self.initial_teacher_present_rows,
            "coadaptation_teacher_rows": self.coadaptation_teacher_rows,
            "coadaptation_teacher_present_rows": self.coadaptation_teacher_present_rows,
            "last_ordinary_update_index": self.last_ordinary_update_index,
            "last_absolute_epoch": self.last_absolute_epoch,
            "last_primary_fine_rows": self.last_primary_fine_rows,
            "max_source_count": self.max_source_count,
            "refresh_case_key": self.refresh_case_key,
            "refresh_cache": None if self._refresh_cache is None else self._refresh_cache.state_dict(),
            "last_loss_components": dict(self.last_loss_components),
            "last_gradient_balance": copy.deepcopy(self.last_gradient_balance),
            "last_refresh": dict(self.last_refresh),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get("schema") != STATE_SCHEMA:
            raise ValueError("Malformed source-read auxiliary manager continuation state.")
        expected = {
            "variant": self.variant,
            "train_case_keys": list(self.train_case_keys),
            "seed": self.seed,
            "sampling_dataset_id": self.sampling_dataset_id,
            "sampling_version": self.sampling_version,
            "initial_query_rows": self.initial_query_rows,
            "refresh_query_rows": self.refresh_query_rows,
            "refresh_every": self.refresh_every,
            "teacher_row_budget_fraction": self.teacher_row_budget_fraction,
            "ram_cache_budget_bytes": self.ram_cache_budget_bytes,
            "source_epoch": 500,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("Source-read auxiliary manager state belongs to a different child or TRAIN cohort.")
        saved_parent = state.get("parent_sha256")
        if (not isinstance(saved_parent, str) or len(saved_parent) != 64
                or any(character not in "0123456789abcdef" for character in saved_parent)
                or (self.parent_sha256 is not None and self.parent_sha256 != saved_parent)):
            raise ValueError("Source-read auxiliary manager state has a different dense parent SHA256.")
        self.parent_sha256 = saved_parent
        scales = state.get("context_scales")
        if scales is not None:
            scales = _cpu_tensor(scales, "context_scales")
            if tuple(scales.shape) != (self.controller.heads,) or not torch.all(scales > 0):
                raise ValueError("Saved source-read auxiliary context scales are malformed.")
            self.context_scales = scales.to(torch.float32)
        self.heating_envelope = _finite_positive_scalar(state.get("heating_envelope"), "heating envelope")
        self.temperature_scale = _finite_positive_scalar(state.get("temperature_scale"), "temperature scale")
        coefficient = state.get("auxiliary_coefficient")
        if coefficient is not None and (type(coefficient) not in (int, float) or not math.isfinite(coefficient)
                                        or coefficient < 0 or coefficient > 1):
            raise ValueError("Saved source-read auxiliary coefficient is malformed.")
        self.auxiliary_coefficient = None if coefficient is None else float(coefficient)
        calibration = state.get("gradient_calibration")
        if calibration is not None:
            if not isinstance(calibration, Mapping) or calibration.get("schema") != "source_read_gradient_calibration_v1" or calibration.get("partition") != "TRAIN":
                raise ValueError("Saved source-read gradient calibration receipt is malformed.")
            try:
                physical_norm = float(calibration["physical_all_inherited_parameter_gradient_l2_norm"])
                auxiliary_norm = float(calibration["auxiliary_all_new_parameter_gradient_l2_norm"])
                fixed_coefficient = float(calibration["fixed_auxiliary_coefficient"])
                realized_share = float(calibration["realized_weighted_auxiliary_to_physical_gradient_share"])
                maximum_share = float(calibration["maximum_gradient_share"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Saved source-read gradient calibration receipt is incomplete.") from exc
            numeric_values = (physical_norm, auxiliary_norm, fixed_coefficient, realized_share, maximum_share)
            if (not all(math.isfinite(value) for value in numeric_values)
                    or physical_norm < 0 or auxiliary_norm <= 0
                    or not 0 <= fixed_coefficient <= 1 or maximum_share != 0.05
                    or realized_share < 0 or realized_share > maximum_share + 1.0e-12
                    or fixed_coefficient != (min(1.0, 0.05 * physical_norm / auxiliary_norm))
                    or realized_share != (fixed_coefficient * auxiliary_norm / physical_norm if physical_norm > 0 else 0.0)
                    or self.auxiliary_coefficient != fixed_coefficient):
                raise ValueError("Saved source-read gradient calibration receipt violates its fixed TRAIN rule.")
            self.gradient_calibration = dict(calibration)
        else:
            if self.auxiliary_coefficient is not None:
                raise ValueError("A fitted source-read manager is missing its fixed gradient calibration receipt.")
            self.gradient_calibration = None
        for name in ("initial_teacher_rows", "initial_teacher_present_rows", "coadaptation_teacher_rows",
                     "coadaptation_teacher_present_rows", "last_ordinary_update_index",
                     "last_absolute_epoch", "last_primary_fine_rows", "max_source_count"):
            value = state.get(name)
            if type(value) is not int or value < 0:
                raise ValueError(f"Saved source-read manager counter {name} is malformed.")
            setattr(self, name, value)
        self.primary_threshold_index = state.get("primary_threshold_index")
        if type(self.primary_threshold_index) is not int or self.primary_threshold_index not in (0, 1):
            raise ValueError("Saved source-read primary operating point is malformed.")
        density = state.get("threshold_train_density")
        if density is not None and tuple(density) != (0.75, 0.50):
            raise ValueError("Saved source-read threshold density differs from the fixed two-point calibration.")
        self.threshold_train_density = None if density is None else (0.75, 0.50)
        screen = state.get("train_fidelity_screen")
        if screen is not None and (not isinstance(screen, Mapping)
                                   or screen.get("schema") != "source_read_train_fidelity_screen_v1"
                                   or screen.get("partition") != "TRAIN"
                                   or screen.get("primary_operating_point") != self.primary_threshold_index):
            raise ValueError("Saved source-read TRAIN fidelity screen is malformed or inconsistent.")
        self.train_fidelity_screen = None if screen is None else copy.deepcopy(dict(screen))
        stored_primary = getattr(self.controller, "primary_operating_point", None)
        if stored_primary is not None and int(stored_primary) != self.primary_threshold_index:
            raise ValueError("Source-read manager operating point differs from the restored controller buffer.")
        cache = state.get("refresh_cache")
        case_key = state.get("refresh_case_key")
        if cache is None:
            self._refresh_cache = None
            self.refresh_case_key = None
        else:
            if not isinstance(cache, Mapping):
                raise ValueError("Saved source-read refresh cache is malformed.")
            restored = _normalize_teacher_case(case_key, cache, self.controller,
                                               max_query_rows=self.refresh_query_rows)
            if restored.case_key not in self.train_case_keys:
                raise ValueError("Saved source-read refresh cache is outside the TRAIN cohort.")
            self._refresh_cache = restored
            self.refresh_case_key = case_key
        components = state.get("last_loss_components", {})
        refresh = state.get("last_refresh", {})
        if not isinstance(components, Mapping) or not isinstance(refresh, Mapping):
            raise ValueError("Saved source-read manager telemetry is malformed.")
        self.last_loss_components = {str(key): float(value) for key, value in components.items()}
        self.last_refresh = dict(refresh)
        gradient_balance = state.get("last_gradient_balance")
        if gradient_balance is not None:
            if not isinstance(gradient_balance, Mapping):
                raise ValueError("Saved source-read gradient-balance telemetry is malformed.")
            try:
                native_norm = float(gradient_balance["native_all_parameter_gradient_l2_norm"])
                auxiliary_norm = float(gradient_balance["weighted_auxiliary_new_parameter_gradient_l2_norm"])
                share = gradient_balance["weighted_auxiliary_to_native_gradient_share"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Saved source-read gradient-balance telemetry is incomplete.") from exc
            if (not math.isfinite(native_norm) or native_norm < 0
                    or not math.isfinite(auxiliary_norm) or auxiliary_norm < 0
                    or (share is None) != (native_norm == 0)):
                raise ValueError("Saved source-read gradient-balance telemetry has invalid norms.")
            expected_share = auxiliary_norm / native_norm if native_norm > 0 else None
            if share is not None and (not math.isfinite(float(share)) or float(share) != expected_share):
                raise ValueError("Saved source-read gradient-balance share does not match its norms.")
            self.last_gradient_balance = {
                "native_all_parameter_gradient_l2_norm": native_norm,
                "weighted_auxiliary_new_parameter_gradient_l2_norm": auxiliary_norm,
                "weighted_auxiliary_to_native_gradient_share": expected_share,
            }
        else:
            self.last_gradient_balance = None


def _truncate_case(case: _TeacherCase, max_query_rows: int) -> _TeacherCase:
    rows = min(case.query_count, max_query_rows)
    if rows == case.query_count:
        return case
    return _TeacherCase(
        case_key=case.case_key,
        query_states=case.query_states[:, :rows].contiguous(),
        module_states=case.module_states,
        descriptor=case.descriptor[:, :rows].contiguous(),
        alpha=case.alpha[:, :rows].contiguous(),
        fine_messages=case.fine_messages[:, :rows].contiguous(),
        thermal_k=None if case.thermal_k is None else case.thermal_k[:, :rows].contiguous(),
        query_mask=case.query_mask[:, :rows].contiguous(),
        source_present_mask=(case.source_present_mask if case.source_present_mask.ndim == 2
                             else case.source_present_mask[:, :rows].contiguous()),
        local_radius=(case.local_radius if case.local_radius.shape[1] == 1
                      else case.local_radius[:, :rows].contiguous()),
        distance_to_sources=case.distance_to_sources[:, :rows].contiguous(),
        fine_executed_rows=case.batch_size * rows * case.source_count,
        fine_present_rows=int((case.query_mask[:, :rows, None] & _source_mask_3d(
            case.source_present_mask if case.source_present_mask.ndim == 2 else case.source_present_mask[:, :rows],
            case.batch_size, rows, case.source_count)).sum().item()),
    )


def fit_source_read_child_auxiliary(
    model: nn.Module,
    provider,
    optimizer: torch.optim.Optimizer,
    *,
    added_parameter_names: Sequence[str],
    train_case_keys: Sequence[Any],
    max_updates: int,
    parent_sha256: str,
    source_epoch: int,
    variant: str | None = None,
    teacher_trace_factory: Callable[[Any, nn.Module, Any, SamplingKey, int], Mapping[str, Any]] | None = None,
    physical_loss_factory: Callable[[Any, nn.Module, Any, SamplingKey], Tensor] | None = None,
) -> dict[str, Any]:
    """Engine callback implementing bounded dense-e500 to learned-child fit.

    Optional factories receive ``(provider, model, case_key, SamplingKey, ... )``
    and exist for provider-owned dispatch and fixture-only tests. Without them,
    the exact provider hooks named by the Round3 protocol are used.
    """
    _, _, controller = _model_parts(model)
    inferred_variant = "S-Complete" if controller.completion_enabled else "S-Utility"
    if variant is None:
        variant = inferred_variant
    if variant != inferred_variant:
        raise ValueError("Auxiliary callback variant disagrees with controller completion configuration.")
    sampling_dataset_id = getattr(provider, "sampling_dataset_id", None)
    if not isinstance(sampling_dataset_id, str) or not sampling_dataset_id:
        identity_method = getattr(provider, "identity_payload", None)
        if not callable(identity_method):
            raise ValueError("Source-read provider must expose its exact sampling dataset identity.")
        sampling_dataset_id = _sampling_dataset_id(identity_method())
    seed = getattr(provider, "seed", 0)
    if type(seed) is not int or seed < 0:
        raise ValueError("Source-read provider seed must be a nonnegative integer.")
    sampling_version = getattr(provider, "sampling_version", SamplingKey.CASE_EPOCH_VERSION)
    manager = SourceReadAuxiliaryManagerV1(
        controller,
        variant=variant,
        train_case_keys=train_case_keys,
        seed=seed,
        sampling_dataset_id=sampling_dataset_id,
        sampling_version=sampling_version,
        teacher_trace_factory=teacher_trace_factory,
        physical_loss_factory=physical_loss_factory,
        parent_sha256=parent_sha256,
    )
    return manager.fit_initial(
        model,
        provider,
        optimizer,
        added_parameter_names=added_parameter_names,
        train_case_keys=train_case_keys,
        max_updates=max_updates,
        parent_sha256=parent_sha256,
        source_epoch=source_epoch,
    )


def make_source_read_auxiliary_manager(
    model: nn.Module,
    provider,
    *,
    train_case_keys: Sequence[Any],
    parent_sha256: str | None = None,
) -> SourceReadAuxiliaryManagerV1:
    """Reconstruct a child manager shell for strict auxiliary-state resume."""
    _, _, controller = _model_parts(model)
    variant = getattr(provider, "source_reader_variant", None)
    expected_variant = "S-Complete" if controller.completion_enabled else "S-Utility"
    if variant != expected_variant:
        raise ValueError("Sparse source-read provider variant differs from its attached controller.")
    sampling_dataset_id = getattr(provider, "sampling_dataset_id", None)
    if not isinstance(sampling_dataset_id, str) or not sampling_dataset_id:
        identity_method = getattr(provider, "identity_payload", None)
        if not callable(identity_method):
            raise ValueError("Source-read provider must expose its exact sampling dataset identity.")
        sampling_dataset_id = _sampling_dataset_id(identity_method())
    seed = getattr(provider, "seed", None)
    if type(seed) is not int or seed < 0:
        raise ValueError("Source-read provider seed must be a nonnegative integer.")
    return SourceReadAuxiliaryManagerV1(
        controller,
        variant=variant,
        train_case_keys=train_case_keys,
        seed=seed,
        sampling_dataset_id=sampling_dataset_id,
        sampling_version=getattr(provider, "sampling_version", SamplingKey.CASE_EPOCH_VERSION),
        parent_sha256=parent_sha256,
    )


__all__ = [
    "DEFAULT_RAM_CACHE_BUDGET_BYTES",
    "MAX_INITIAL_QUERY_ROWS",
    "MAX_REFRESH_QUERY_ROWS",
    "make_source_read_auxiliary_manager",
    "SCHEMA",
    "STATE_SCHEMA",
    "SourceReadAuxiliaryManagerV1",
    "fit_source_read_child_auxiliary",
]

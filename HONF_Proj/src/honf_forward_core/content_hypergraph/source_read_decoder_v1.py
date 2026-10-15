"""Source-preserving selected decoding for the separately labelled Round3 study.

Only the light descriptor, original attention and utility scores are dense in
the query/source axes. Fine Fourier features and fine values follow selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .decoding import _query_mask
from .source_read_plans import (
    build_source_read_plan_v1, combine_source_gates, directional_support,
    nearest_support, s_attn_support, shuffle_support, smooth_near_gate,
    smooth_threshold_gate,
)
from .source_read_response import SourceReadAffineFactorsV1
from .source_read_sparse import (
    evaluate_fine_source_rows_v1, original_source_scores_v1,
    reduce_selected_headwise_v1,
)


@dataclass(frozen=True)
class SelectedSourceReadV1:
    context: Tensor
    alpha: Tensor
    response: SourceReadAffineFactorsV1 | Tensor | None
    messages: Tensor | None
    counts: dict[str, int]
    trace: dict[str, Tensor]


def decode_selected_sources_v1(core, organization, query, query_states, *, return_messages=False):
    """Build and execute one heat-independent plan; never renormalize alpha."""
    reader = core.direct_reader
    scene = organization.scene
    controller = reader.source_read
    override = core._source_read_override
    mode = override[0] if override is not None else "fitted"
    executor = override[1] if override is not None else core.source_read_executor
    point = override[2] if override is not None else None
    batch, queries = query_states.shape[:2]
    sources = scene.num_modules
    descriptor, alpha = original_source_scores_v1(
        reader, scene, query, query_states, organization.frame_scale,
        core._module_measure_reference(scene, query_states.dtype), organization.direct_source_keys)
    # Information attention uses positive measure, but a physically present
    # zero-measure source still owns its signed Thermal control column.
    present = scene.module_mask if reader.thermal_output is not None else (
        scene.module_mask & (scene.module_measure > 0))
    valid = _query_mask(query)
    distance = torch.linalg.vector_norm(
        descriptor[..., :core.coordinate_dim] * organization.frame_scale[:, None, None, :], dim=-1)
    near = smooth_near_gate(distance, scene.local_radius[:, None, :])
    protected = (near > 0) & present[:, None, :] & valid[:, :, None]
    fitted_count = mode == "fitted" or mode.endswith("_fitted")
    completion = fitted_count and core.source_read_policy == "completion"
    target = None
    if fitted_count:
        if controller is None:
            raise ValueError("A fitted source policy requires a trained source controller.")
        # The check is one scalar per request, never a per-query CPU mask loop.
        if not bool(controller.calibrated) or not bool(controller.trained):
            raise ValueError("Source controller must be TRAIN-fitted and calibrated before selection.")
        score = controller.utility_scores(
            organization.source_read_source_projection,
            controller.project_query_states(query_states), descriptor).sum(dim=-1)
        index = int(controller.primary_operating_point) if point is None else int(point)
        if index < 0 or index >= int(controller.threshold_count):
            raise ValueError("Unknown TRAIN-calibrated operating point.")
        gates = combine_source_gates(smooth_threshold_gate(score, controller.thresholds[index]), near)
        transition = controller.transition_lambda.to(gates)
        gates = 1 - transition + transition * gates
        if mode != "fitted":
            # Population controls match the fitted physical-pair count. Their
            # hard unit gates are a different approximation from smooth fitted
            # attenuation; score/count discovery remains measured work.
            target = (gates > 0) & present[:, None, :] & valid[:, :, None]
    else:
        # These hard supports are read-only controls. TRAIN deployment always
        # uses the smooth utility/near gates above.
        if mode not in ("attn95", "attn99", "nearest95", "nearest99", "directional95",
                        "directional99", "shuffle95", "shuffle99"):
            raise ValueError("Unknown read-only source policy.")
        retention = .95 if mode.endswith("95") else .99
        target = s_attn_support(alpha, present, retention=retention, protected_mask=protected)
    if target is not None:
        if mode.startswith("nearest"):
            target = nearest_support(query.coordinates, scene.module_coordinates, present, target,
                                     protected_mask=protected)
        elif mode.startswith("directional"):
            # The source scene's operating direction is input metadata; the
            # caller declares its physical axis explicitly through this mode.
            direction = query.coordinates.new_zeros(core.coordinate_dim)
            direction[0] = 1
            target = directional_support(query.coordinates, scene.module_coordinates, direction,
                                         present, target, protected_mask=protected)
        elif mode.startswith("shuffle"):
            # Deterministic physical-position score; no global RNG or target
            # information and no dependence on source-array ordering.
            phase = scene.module_coordinates.to(query_states.dtype).sum(-1)
            scores = torch.frac(torch.sin(phase * 12.9898 + 78.233) * 43758.5453)
            target = shuffle_support(present, target, protected_mask=protected,
                                     shuffle_scores=scores[:, None, :].expand(batch, queries, sources))
        gates = target.to(query_states.dtype)
    pair_valid = present[:, None, :] & valid[:, :, None]
    # Exact legacy recovery is deliberate: no completion/cancellation roundoff
    # and no unqualified projection reassociation at the full-access endpoint.
    if bool(((gates == 1) | ~pair_valid).all()):
        context, old_alpha, response, messages = reader(
            scene, query, query_states, organization.frame_scale,
            core._module_measure_reference(scene, query_states.dtype),
            organization.direct_source_keys, organization.direct_source_values,
            return_messages=return_messages, execution_backend=core.reader_execution_backend)
        selected_count = int(pair_valid.sum())
        return SelectedSourceReadV1(context, old_alpha, response, messages,
            {"fine_selected_rows": selected_count,
             "fine_executed_rows": batch * queries * sources,
             "fine_padding_rows": batch * queries * sources - selected_count,
             "source_score_rows": 2 * batch * queries * sources,
             "source_utility_score_rows": batch * queries * sources if fitted_count else 0,
             "source_full_fastpath": 1},
            {"source_read_gates": gates * pair_valid, "source_read_support": pair_valid})
    plan = build_source_read_plan_v1(gates, alpha, present, query_valid_mask=valid)
    query_values = reader.query_value(query_states)
    fine = evaluate_fine_source_rows_v1(
        reader, plan, descriptor, query_values, organization.direct_source_values, executor=executor)
    pairs = plan.packed_pairs
    b, q, m = pairs.batch_index, pairs.query_index, pairs.source_index
    hidden = fine.hidden
    base_head = None
    cq = cs = base_selected = None
    if completion:
        cq = controller.project_completion_queries(query_states)
        cs = organization.source_read_completion_source
        base_selected = controller.completion_pair_hidden_from_projections(cq, cs, b, q, m)
        hidden = hidden - base_selected
        base_head = controller.completion_headwise_sum_from_projections(cq, cs, alpha)
    context = reduce_selected_headwise_v1(hidden, plan, reader.message_output,
                                         base_head_hidden=base_head)
    response = None
    if reader.thermal_output is not None:
        # The Thermal law is signed and source-wise, independently of alpha.
        wk = (reader.thermal_output.weight @ reader.message_output.weight).reshape(-1)
        bk = reader.thermal_output.bias.reshape(())
        if reader.message_output.bias is not None:
            bk = bk + (reader.thermal_output.weight @ reader.message_output.bias).reshape(())
        coefficient = F.linear(hidden, wk[None]).squeeze(-1)
        rank_weight = None
        if completion:
            rank_weight = (wk[None] @ controller.completion_output.weight).reshape(-1)
        else:
            coefficient = coefficient + bk
        response = SourceReadAffineFactorsV1(
            batch, queries, sources, b, q, m, coefficient * pairs.gate, present,
            controller.heating_envelope if controller is not None else 1.0e30,
            query_mask=valid, completion_query=cq, completion_source=cs,
            thermal_rank_weight=rank_weight, scalar_bias=bk if completion else 0.)
    messages = None
    if return_messages:
        # Explicit bounded trace only. This allocation is counted separately.
        values = reader.message_output(fine.hidden)
        messages = values.new_zeros((batch, queries, sources, core.local_hidden_dim))
        messages[b, q, m] = values
    counts = dict(fine.counts)
    counts.update(source_score_rows=batch * queries * sources,
                  source_utility_score_rows=batch * queries * sources if fitted_count else 0,
                  source_full_fastpath=0, completion_rank=16 if completion else 0,
                  response_selected_coefficient_entries=0 if response is None else pairs.pair_count,
                  completion_query_factor_entries=0 if cq is None else cq.numel(),
                  completion_source_factor_entries=0 if cs is None else cs.numel())
    return SelectedSourceReadV1(context, alpha, response, messages, counts,
        {"source_read_gates": plan.gates, "source_read_support": plan.support_mask,
         "source_read_selected_batch": b, "source_read_selected_query": q,
         "source_read_selected_source": m})

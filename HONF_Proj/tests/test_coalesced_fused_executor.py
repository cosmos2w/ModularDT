"""B-correct fused QE selection for controlled Run-1503-v2 quotients."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from test_coalesced_sparse_incidence_quotient import (
    _batch,
    _device,
    _force_first_pair_quotient,
    _packed_state,
    _payload,
)

from honf_forward_core.config import UnifiedForwardConfig
from honf_forward_core.interface_fields import InterfaceFieldCore
from honf_forward_core.interface_fields.kernels.qe_triton import is_triton_qe_available
from honf_forward_core.interface_fields.sparse_incidence_group_control import (
    SparseIncidenceGroupControlPairwiseField,
)


@pytest.mark.skipif(
    not is_triton_qe_available("cuda"),
    reason="coalesced fused QE requires CUDA and Triton",
)
def test_nonzero_quotient_fused_executor_matches_full_core_and_gradients() -> None:
    device = _device()
    torch.manual_seed(15030331)
    core = InterfaceFieldCore(
        UnifiedForwardConfig.from_dict(_payload())
    ).to(device).train()
    core.set_training_progress(epoch=50, total_epochs=500)
    batch = _batch(device)
    encoded = core.encode_case(batch)
    prepared_candidate = core.prepare(
        encoded,
        encoded.module_tokens,
        return_routing_maps=True,
    )
    backend = core.backend
    candidate_state = prepared_candidate.backend_state
    generated_plan = candidate_state["coalescence_plan"]
    plan = _force_first_pair_quotient(generated_plan)
    assert (plan.case_group_count < plan.active_proposal_count).all()
    assert (plan.multiplicity == 2).sum(dim=-1).eq(1).all()

    provisional_state = SparseIncidenceGroupControlPairwiseField.prepare(
        backend,
        encoded,
        encoded.module_tokens,
        return_routing_maps=False,
    )
    provisional = provisional_state["group_control_state"]
    query = plan.module_query_centers[:, :1, :].expand(-1, 5, -1).contiguous()
    receiver_features = backend.router.query_fourier(
        query / backend.router._scale(encoded)
    )
    query_input = torch.cat(
        [
            receiver_features,
            provisional.global_control[:, None, :].expand(
                -1, query.shape[1], -1
            ),
        ],
        dim=-1,
    )
    scaled_query = backend.router._rms_scale(
        backend.router.query_projection(query_input)
    )
    geometry = -0.5 * (
        torch.linalg.vector_norm(
            query[:, :, None, :] - plan.module_query_centers[:, None, :, :],
            dim=-1,
        )
        + torch.linalg.vector_norm(
            query[:, :, None, :] - plan.environment_query_centers[:, None, :, :],
            dim=-1,
        )
    ) / plan.geometry_scale[:, None, None]
    target_logits = torch.full_like(geometry[:, 0, :], -2.0)
    target_logits[:, :2] = 0.5
    key_direction = scaled_query[:, 0, :].detach()
    key_norm = key_direction.square().sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    key_block = key_direction[:, None, :] * (
        (target_logits - geometry[:, 0, :].detach()) / key_norm
    )[..., None]
    controlled_descriptors = plan.fused_descriptors.clone()
    controlled_descriptors[..., : plan.control_dim] = key_block
    plan = replace(plan, fused_descriptors=controlled_descriptors)
    packed_state, _packed_controls = _packed_state(
        backend,
        provisional_state,
        plan,
        encoded,
    )
    prepared = replace(prepared_candidate, backend_state=packed_state)

    features = core._receiver_features(prepared, query.float())
    route = backend._route(packed_state, encoded, query, features)
    assert (route.assignment[..., 0] > 0.0).all()
    assert (route.assignment[..., 1] > 0.0).all()

    backend.executor_policy = "rectangular_reference"
    backend.diagnostic_executor_independent = True
    rectangular = core.decode_queries(
        prepared,
        query,
        receiver_chunk_size=4,
    )["pred_field"]

    calls = {"module_selected": 0, "environment_fused": 0}
    original_module = backend._read_module_partial
    original_environment = backend._read_environment_fused_unique_pairs

    def counted_module(*args, **kwargs):
        calls["module_selected"] += 1
        return original_module(*args, **kwargs)

    def counted_environment(*args, **kwargs):
        calls["environment_fused"] += 1
        return original_environment(*args, **kwargs)

    backend._read_module_partial = counted_module
    backend._read_environment_fused_unique_pairs = counted_environment
    backend.executor_policy = "fused_unique_pairs"
    backend.diagnostic_executor_independent = False
    try:
        selected = core.decode_queries(
            prepared,
            query,
            receiver_chunk_size=4,
        )["pred_field"]
    finally:
        backend._read_module_partial = original_module
        backend._read_environment_fused_unique_pairs = original_environment
        backend.executor_policy = "rectangular_reference"
        backend.diagnostic_executor_independent = True

    assert calls["module_selected"] > 0
    assert calls["environment_fused"] > 0
    torch.testing.assert_close(selected, rectangular, atol=3.0e-5, rtol=3.0e-5)

    parameters = tuple(parameter for parameter in core.parameters() if parameter.requires_grad)
    targets = (query, *parameters)
    rectangular_gradients = torch.autograd.grad(
        rectangular.square().sum(),
        targets,
        allow_unused=True,
        retain_graph=True,
    )
    selected_gradients = torch.autograd.grad(
        selected.square().sum(),
        targets,
        allow_unused=True,
    )
    assert len(rectangular_gradients) == len(selected_gradients)
    for expected, actual in zip(
        rectangular_gradients,
        selected_gradients,
        strict=True,
    ):
        if expected is None:
            assert actual is None
            continue
        assert actual is not None and torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=3.0e-4, rtol=3.0e-3)

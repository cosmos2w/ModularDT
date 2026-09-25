"""End-to-end parent and physical-reader contracts for Run 1503-v4."""

from __future__ import annotations

import copy
import importlib
import json
from dataclasses import fields, replace
from itertools import pairwise
from pathlib import Path

import torch

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.interface_fields import InterfaceFieldCore
from honf_forward_core.interface_fields.group_control_router import GroupQueryRoute
from honf_forward_core.interface_fields.sparse_incidence_group_control import (
    SparseIncidenceGroupControlPairwiseField,
)
from honf_forward_core.organization.functional_fusion_tree import (
    FunctionalProbeCatalogue,
    FunctionalProbeFamily,
    contracted_logits,
)
from honf_forward_core.organization.group_fusion import weighted_sparsemax

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V4_PROFILE = PROJECT_ROOT / "src/config_core/forward/continuous_functional_coalescence_honf_context.json"
PARENT_PROFILE = PROJECT_ROOT / "src/config_core/forward/sparse_incidence_group_control_honf_context.json"
V4_ARCHITECTURE = "continuous_functional_coalescence_honf"
PARENT_ARCHITECTURE = "sparse_incidence_group_control_honf"


def _payload(architecture: str) -> dict[str, object]:
    source = V4_PROFILE if architecture == V4_ARCHITECTURE else PARENT_PROFILE
    payload = copy.deepcopy(json.loads(source.read_text(encoding="utf-8"))["model"]["core_honf"])
    payload.update(
        {
            "field_dim": 5,
            "hidden_dim": 16,
            "domain_length_x": 12.0,
            "domain_length_y": 6.0,
            "module_radius": 0.45,
            "query_fourier_frequencies": 2,
            "position_fourier_frequencies": 2,
            "forward_architecture": architecture,
        }
    )
    interface = payload["interface_model"]
    interface.update(
        {
            "message_hidden_dim": 8,
            "attention_heads": 2,
            "relative_fourier_frequencies": 2,
            "receiver_chunk_size": 4,
            "activation_checkpointing": False,
            "environment_refinement_normalizer": "sparsemax",
        }
    )
    return payload


def _batch(seed: int, queries: int = 8) -> BatchData:
    generator = torch.Generator().manual_seed(seed)
    extent = torch.tensor([12.0, 6.0])
    return BatchData(
        module_centers=torch.rand(2, 5, 2, generator=generator) * extent,
        module_present=torch.tensor(
            [[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]]
        ),
        module_features=torch.randn(2, 5, 3, generator=generator),
        global_context=torch.randn(2, 4, generator=generator),
        query_xy=torch.rand(2, queries, 2, generator=generator) * extent,
        query_time=None,
        target_field=torch.randn(2, queries, 5, generator=generator),
        case_name="run1503-v4-functional-test",
        metadata={},
        env_coords=torch.rand(2, 13, 2, generator=generator) * extent,
        env_features=torch.randn(2, 13, 2, generator=generator),
        env_weights=torch.rand(2, 13, generator=generator) + 0.2,
    )


def _on_cpu(batch: BatchData) -> BatchData:
    updates = {
        field.name: value.cpu()
        for field in fields(batch)
        if torch.is_tensor(value := getattr(batch, field.name))
    }
    return replace(batch, **updates)


def _port_probe_catalogue(batch: BatchData) -> FunctionalProbeCatalogue:
    batch_size, modules = batch.module_present.shape
    coordinates = batch.module_centers[:, :, None, :].expand(-1, -1, 2, -1).reshape(
        batch_size, modules * 2, 2
    )
    valid = (batch.module_present > 0.5)[:, :, None].expand(-1, -1, 2).reshape(
        batch_size, modules * 2
    )
    weights = valid.to(dtype=coordinates.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    family = FunctionalProbeFamily(coordinates, valid, weights)
    return FunctionalProbeCatalogue("P2", family, family)


def _same_first_gradients(
    left_loss: torch.Tensor,
    right_loss: torch.Tensor,
    inputs: tuple[torch.Tensor, ...],
    parameters: tuple[torch.nn.Parameter, ...],
    *,
    atol: float,
    rtol: float,
) -> int:
    values = (*inputs, *parameters)
    left = torch.autograd.grad(left_loss, values, retain_graph=True, allow_unused=True)
    right = torch.autograd.grad(right_loss, values, retain_graph=True, allow_unused=True)
    nonzero_parameter_tensors = 0
    for index, (left_grad, right_grad) in enumerate(zip(left, right, strict=True)):
        assert (left_grad is None) == (right_grad is None)
        if left_grad is None:
            continue
        assert torch.isfinite(left_grad).all()
        assert torch.isfinite(right_grad).all()
        torch.testing.assert_close(left_grad, right_grad, atol=atol, rtol=rtol)
        if index >= len(inputs) and torch.count_nonzero(left_grad).item() > 0:
            nonzero_parameter_tensors += 1
    return nonzero_parameter_tensors


def test_epoch_fifty_uses_bitwise_parent_path_and_first_gradients() -> None:
    torch.manual_seed(15035001)
    parent = InterfaceFieldCore(UnifiedForwardConfig.from_dict(_payload(PARENT_ARCHITECTURE))).cpu().eval()
    candidate = InterfaceFieldCore(UnifiedForwardConfig.from_dict(_payload(V4_ARCHITECTURE))).cpu().eval()
    candidate.set_training_progress(epoch=50, total_epochs=500)

    # Materialize lazy layers before transferring the exact parent weights.
    warmup = _batch(15035002, queries=3)
    with torch.no_grad():
        for core in (parent, candidate):
            encoded = core.encode_case(warmup)
            prepared = core.prepare(encoded, encoded.module_tokens)
            core.decode_queries(prepared, warmup.query_xy)
    incompatible = candidate.load_state_dict(parent.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert set(incompatible.missing_keys) == {"backend.functional_tree_membership"}

    batch = _batch(15035003, queries=5)
    batch = replace(
        batch,
        query_xy=batch.query_xy.clone().requires_grad_(),
        module_centers=batch.module_centers.clone().requires_grad_(),
        env_coords=batch.env_coords.clone().requires_grad_(),
    )
    parent_encoded = parent.encode_case(batch)
    candidate_encoded = candidate.encode_case(batch)
    parent_prepared = parent.prepare(parent_encoded, parent_encoded.module_tokens)
    candidate_prepared = candidate.prepare(candidate_encoded, candidate_encoded.module_tokens)
    assert "functional_tree_plan" not in candidate_prepared.backend_state
    parent_output = parent.decode_queries(parent_prepared, batch.query_xy)
    candidate_output = candidate.decode_queries(candidate_prepared, batch.query_xy)
    torch.testing.assert_close(
        parent_output["pred_field"], candidate_output["pred_field"], atol=0.0, rtol=0.0
    )

    parent_parameters = dict(parent.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    assert parent_parameters.keys() == candidate_parameters.keys()
    selected_names = (
        "backend.router.query_projection.0.weight",
        "backend.router.group_control.net.0.weight",
    )
    selected = tuple(candidate_parameters[name] for name in selected_names)
    parent_selected = tuple(parent_parameters[name] for name in selected_names)
    weights = torch.linspace(0.7, 1.3, parent_output["pred_field"].numel()).reshape_as(
        parent_output["pred_field"]
    )
    parent_grads = torch.autograd.grad(
        (parent_output["pred_field"] * weights).sum(),
        (batch.query_xy, batch.module_centers, batch.env_coords, *parent_selected),
        retain_graph=True,
    )
    candidate_grads = torch.autograd.grad(
        (candidate_output["pred_field"] * weights).sum(),
        (batch.query_xy, batch.module_centers, batch.env_coords, *selected),
        retain_graph=True,
    )
    for left, right in zip(parent_grads, candidate_grads, strict=True):
        torch.testing.assert_close(left, right, atol=0.0, rtol=0.0)


def test_active_identity_tree_retains_parent_reader_and_first_gradients(monkeypatch) -> None:
    """An active, uncontracted tree must preserve the actual parent reader."""

    torch.manual_seed(15035004)
    parent = InterfaceFieldCore(UnifiedForwardConfig.from_dict(_payload(PARENT_ARCHITECTURE))).cpu().eval()
    candidate = InterfaceFieldCore(UnifiedForwardConfig.from_dict(_payload(V4_ARCHITECTURE))).cpu().eval()
    candidate.set_training_progress(epoch=50, total_epochs=500)
    warmup = _batch(15035005, queries=3)
    with torch.no_grad():
        for core in (parent, candidate):
            encoded = core.encode_case(warmup)
            prepared = core.prepare(encoded, encoded.module_tokens)
            core.decode_queries(prepared, warmup.query_xy)
    incompatible = candidate.load_state_dict(parent.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert set(incompatible.missing_keys) == {"backend.functional_tree_membership"}
    candidate.set_training_progress(epoch=150, total_epochs=500)

    backend_module = importlib.import_module(
        "honf_forward_core.interface_fields.continuous_functional_coalescence"
    )
    original_scores = backend_module.node_source_action_scores

    def keep_distinct(*args, **kwargs):
        scores = original_scores(*args, **kwargs)
        return replace(scores, score2=torch.ones_like(scores.score2))

    monkeypatch.setattr(backend_module, "node_source_action_scores", keep_distinct)
    batch = _batch(15035006, queries=7)
    batch = replace(
        batch,
        query_xy=batch.query_xy.clone().requires_grad_(),
        module_centers=batch.module_centers.clone().requires_grad_(),
        env_coords=batch.env_coords.clone().requires_grad_(),
    )
    parent_encoded = parent.encode_case(batch)
    candidate_encoded = candidate.encode_case(batch)
    parent_prepared = parent.prepare(parent_encoded, parent_encoded.module_tokens)
    candidate_prepared = candidate.prepare(
        candidate_encoded,
        candidate_encoded.module_tokens,
        functional_probes=_port_probe_catalogue(batch),
    )
    plan = candidate_prepared.backend_state["functional_tree_plan"]
    assert not bool(plan.closed_nodes.any())
    assert not bool((plan.node_gamma > 0.0).any())
    torch.testing.assert_close(
        plan.valid.sum(dim=-1), plan.parent_controls.phase_occupied.sum(dim=-1)
    )

    parent_field = parent.decode_queries(parent_prepared, batch.query_xy)["pred_field"]
    candidate_field = candidate.decode_queries(candidate_prepared, batch.query_xy)["pred_field"]
    torch.testing.assert_close(candidate_field, parent_field, atol=1.0e-6, rtol=1.0e-6)
    parent_parameter = dict(parent.named_parameters())["backend.router.query_projection.0.weight"]
    candidate_parameter = dict(candidate.named_parameters())["backend.router.query_projection.0.weight"]
    weights = torch.linspace(0.7, 1.3, parent_field.numel()).reshape_as(parent_field)
    parent_gradients = torch.autograd.grad(
        (parent_field * weights).sum(),
        (batch.query_xy, batch.module_centers, batch.env_coords, parent_parameter),
        retain_graph=True,
    )
    candidate_gradients = torch.autograd.grad(
        (candidate_field * weights).sum(),
        (batch.query_xy, batch.module_centers, batch.env_coords, candidate_parameter),
        retain_graph=True,
    )
    for parent_gradient, candidate_gradient in zip(parent_gradients, candidate_gradients, strict=True):
        torch.testing.assert_close(candidate_gradient, parent_gradient, atol=1.0e-5, rtol=1.0e-5)


def test_active_packed_physical_reads_match_virtual_tree_and_gradients() -> None:
    torch.manual_seed(717)
    core = InterfaceFieldCore(UnifiedForwardConfig.from_dict(_payload(V4_ARCHITECTURE))).cpu().eval()
    core.set_training_progress(epoch=151, total_epochs=500)
    batch = _on_cpu(_batch(717, queries=8))
    batch = replace(
        batch,
        query_xy=batch.query_xy.clone().requires_grad_(),
        module_centers=batch.module_centers.clone().requires_grad_(),
        env_coords=batch.env_coords.clone().requires_grad_(),
    )
    encoded = core.encode_case(batch)
    probes = _port_probe_catalogue(batch)
    compact = core.prepare(
        encoded,
        encoded.module_tokens,
        functional_probes=probes,
        return_routing_maps=True,
    )
    state = compact.backend_state
    plan = state["functional_tree_plan"]
    assert bool(plan.closed_nodes.any())
    assert bool((plan.valid.sum(dim=-1) < plan.parent_controls.phase_occupied.sum(dim=-1)).all())

    receiver_features = core._receiver_features(compact, batch.query_xy.float())
    compact_route = core.backend._route(
        state, encoded, batch.query_xy, receiver_features
    )
    parent_route = core.backend.router.route_queries(
        encoded, plan.parent_controls, batch.query_xy, receiver_features
    )
    virtual_logits = contracted_logits(parent_route.logits, plan.transform)
    virtual_mass, virtual_alpha = weighted_sparsemax(
        virtual_logits,
        torch.ones_like(plan.parent_controls.phase_occupied, dtype=torch.long),
        plan.parent_controls.phase_occupied[:, None, :],
    )
    virtual_route = GroupQueryRoute(
        query_control=parent_route.query_control,
        assignment=virtual_alpha,
        logits=virtual_logits,
        query_keys=parent_route.query_keys,
    )
    del virtual_mass
    expanded_compact_alpha = torch.einsum(
        "bqr,bkr->bqk", compact_route.query_density, plan.membership
    )
    torch.testing.assert_close(expanded_compact_alpha, virtual_alpha, atol=5.0e-7, rtol=5.0e-6)

    compact_controls = state["group_control_state"]
    parent_controls = plan.parent_controls
    module_rho_compact = torch.bmm(
        compact_route.query_density, compact_controls.module_membership.transpose(1, 2)
    )
    module_rho_virtual = torch.bmm(
        virtual_alpha, parent_controls.module_membership.transpose(1, 2)
    )
    environment_rho_compact = torch.bmm(
        compact_route.query_density,
        compact_controls.environment_membership.transpose(1, 2),
    )
    environment_rho_virtual = torch.bmm(
        virtual_alpha, parent_controls.environment_membership.transpose(1, 2)
    )
    module_zeta_compact = torch.einsum(
        "bqr,bsrd->bqsd", compact_route.query_density, compact_controls.module_source_moment
    )
    module_zeta_virtual = torch.einsum(
        "bqk,bsk,bkd->bqsd",
        virtual_alpha,
        parent_controls.module_membership,
        parent_controls.group_control,
    )
    environment_zeta_compact = torch.einsum(
        "bqr,breh->bqeh",
        compact_route.query_density,
        state["environment_head_source_control"],
    )
    parent_state = SparseIncidenceGroupControlPairwiseField.prepare(
        core.backend, encoded, encoded.module_tokens, return_routing_maps=False
    )
    environment_zeta_virtual = torch.einsum(
        "bqk,bkeh->bqeh",
        virtual_alpha,
        parent_state["environment_head_source_control"],
    )
    torch.testing.assert_close(module_rho_compact, module_rho_virtual, atol=5.0e-7, rtol=5.0e-6)
    torch.testing.assert_close(environment_rho_compact, environment_rho_virtual, atol=5.0e-7, rtol=5.0e-6)
    torch.testing.assert_close(module_zeta_compact, module_zeta_virtual, atol=5.0e-7, rtol=5.0e-6)
    torch.testing.assert_close(environment_zeta_compact, environment_zeta_virtual, atol=2.0e-6, rtol=2.0e-5)

    compact_output = core.decode_queries(compact, batch.query_xy)
    virtual_prepared = replace(compact, backend_state=parent_state)
    original_route = core.backend._route

    def transformed_parent_route(state_arg, encoded_arg, receivers, receiver_features=None):
        if state_arg is not parent_state:
            return original_route(state_arg, encoded_arg, receivers, receiver_features)
        parent = core.backend.router.route_queries(
            encoded_arg, plan.parent_controls, receivers, receiver_features
        )
        logits = contracted_logits(parent.logits, plan.transform)
        _, assignment = weighted_sparsemax(
            logits,
            torch.ones_like(plan.parent_controls.phase_occupied, dtype=torch.long),
            plan.parent_controls.phase_occupied[:, None, :],
        )
        return GroupQueryRoute(
            query_control=parent.query_control,
            assignment=assignment,
            logits=logits,
            query_keys=parent.query_keys,
        )

    core.backend._route = transformed_parent_route  # type: ignore[method-assign]
    try:
        virtual_output = core.decode_queries(virtual_prepared, batch.query_xy)
    finally:
        core.backend._route = original_route  # type: ignore[method-assign]
    torch.testing.assert_close(
        compact_output["pred_field"], virtual_output["pred_field"], atol=2.0e-6, rtol=2.0e-5
    )
    compact_route_for_read = core.backend._route(
        state, encoded, batch.query_xy, receiver_features
    )
    virtual_route_for_read = virtual_route
    compact_qm, _ = core.backend._read_module(
        state, encoded, batch.query_xy, compact_route_for_read, include_diagnostics=False
    )
    virtual_qm, _ = core.backend._read_module(
        parent_state, encoded, batch.query_xy, virtual_route_for_read, include_diagnostics=False
    )
    compact_qe, _ = core.backend._read_environment(
        state,
        encoded,
        batch.query_xy,
        receiver_features,
        compact_route_for_read,
        include_diagnostics=False,
    )
    virtual_qe, _ = core.backend._read_environment(
        parent_state,
        encoded,
        batch.query_xy,
        receiver_features,
        virtual_route_for_read,
        include_diagnostics=False,
    )
    torch.testing.assert_close(compact_qm, virtual_qm, atol=2.0e-6, rtol=2.0e-5)
    torch.testing.assert_close(compact_qe, virtual_qe, atol=2.0e-6, rtol=2.0e-5)

    parameter_map = dict(core.named_parameters())
    parameters = tuple(parameter_map.values())
    weights = torch.linspace(0.8, 1.2, compact_output["pred_field"].numel()).reshape_as(
        compact_output["pred_field"]
    )
    nonzero_parameter_tensors = _same_first_gradients(
        (compact_output["pred_field"] * weights).sum(),
        (virtual_output["pred_field"] * weights).sum(),
        (batch.query_xy, batch.module_centers, batch.env_coords),
        parameters,
        atol=8.0e-5,
        rtol=5.0e-4,
    )
    assert nonzero_parameter_tensors >= 20


def test_core_reader_closure_boundary_has_one_sided_flat_gradients(monkeypatch) -> None:
    """Exercise the real prepare/reader path while controlling only its score input."""
    torch.manual_seed(717)
    core = InterfaceFieldCore(UnifiedForwardConfig.from_dict(_payload(V4_ARCHITECTURE))).cpu().eval()
    core.set_training_progress(epoch=151, total_epochs=500)
    batch = _on_cpu(_batch(717, queries=8))
    encoded = core.encode_case(batch)
    probes = _port_probe_catalogue(batch)

    backend_module = importlib.import_module(
        "honf_forward_core.interface_fields.continuous_functional_coalescence"
    )
    original_score_fn = backend_module.node_source_action_scores
    score_input: list[torch.Tensor] = []

    def controlled_node_scores(*args, **kwargs):
        scores = original_score_fn(*args, **kwargs)
        scalar_score = score_input[0].to(device=scores.score2.device, dtype=scores.score2.dtype)
        return replace(scores, score2=scalar_score.expand_as(scores.score2))

    monkeypatch.setattr(backend_module, "node_source_action_scores", controlled_node_scores)

    close_score2 = 0.02**2
    brackets = (1.0e-4, 5.0e-5, 2.0e-5, 1.0e-5)
    weights = torch.linspace(0.8, 1.2, batch.target_field.numel()).reshape_as(batch.target_field)
    right_gradients: list[float] = []
    right_field_deltas: list[float] = []
    boundary_output: torch.Tensor | None = None

    for delta in (0.0, *(-value for value in brackets), *brackets):
        score = torch.tensor(close_score2 + delta, dtype=torch.float32, requires_grad=True)
        score_input[:] = [score]
        compact = core.prepare(
            encoded,
            encoded.module_tokens,
            functional_probes=probes,
            return_routing_maps=True,
        )
        plan = compact.backend_state["functional_tree_plan"]
        parent_state = SparseIncidenceGroupControlPairwiseField.prepare(
            core.backend, encoded, encoded.module_tokens, return_routing_maps=False
        )

        compact_output = core.decode_queries(compact, batch.query_xy)
        virtual_prepared = replace(compact, backend_state=parent_state)
        original_route = core.backend._route

        def transformed_parent_route(
            state_arg,
            encoded_arg,
            receivers,
            receiver_features=None,
            *,
            parent_state_arg=parent_state,
            original_route_fn=original_route,
            plan_arg=plan,
        ):
            if state_arg is not parent_state_arg:
                return original_route_fn(state_arg, encoded_arg, receivers, receiver_features)
            parent_route = core.backend.router.route_queries(
                encoded_arg, plan_arg.parent_controls, receivers, receiver_features
            )
            virtual_logits = contracted_logits(parent_route.logits, plan_arg.transform)
            _, assignment = weighted_sparsemax(
                virtual_logits,
                torch.ones_like(plan_arg.parent_controls.phase_occupied, dtype=torch.long),
                plan_arg.parent_controls.phase_occupied[:, None, :],
            )
            return GroupQueryRoute(
                query_control=parent_route.query_control,
                assignment=assignment,
                logits=virtual_logits,
                query_keys=parent_route.query_keys,
            )

        core.backend._route = transformed_parent_route  # type: ignore[method-assign]
        try:
            virtual_output = core.decode_queries(virtual_prepared, batch.query_xy)
        finally:
            core.backend._route = original_route  # type: ignore[method-assign]

        torch.testing.assert_close(
            compact_output["pred_field"],
            virtual_output["pred_field"],
            atol=2.0e-6,
            rtol=2.0e-5,
        )
        compact_objective = (compact_output["pred_field"] * weights).sum()
        virtual_objective = (virtual_output["pred_field"] * weights).sum()
        compact_gradient, = torch.autograd.grad(compact_objective, score, retain_graph=True)
        virtual_gradient, = torch.autograd.grad(virtual_objective, score, retain_graph=True)
        assert torch.isfinite(compact_gradient)
        torch.testing.assert_close(compact_gradient, virtual_gradient, atol=2.0e-5, rtol=2.0e-4)

        if delta <= 0.0:
            assert bool(plan.closed_nodes.all())
            assert bool((plan.node_gamma == 1.0).all())
            assert torch.count_nonzero(compact_gradient).item() == 0
            if delta == 0.0:
                boundary_output = compact_output["pred_field"].detach()
            else:
                torch.testing.assert_close(
                    compact_output["pred_field"], boundary_output, atol=2.0e-7, rtol=2.0e-6
                )
        else:
            assert not bool(plan.closed_nodes.any())
            assert bool((plan.node_gamma < 1.0).all())
            right_gradients.append(float(compact_gradient.detach()))
            right_field_deltas.append(
                float((compact_output["pred_field"].detach() - boundary_output).norm())
            )

    assert all(value > 0.0 for value in right_gradients)
    assert all(left > right for left, right in pairwise(right_gradients))
    assert all(left > right for left, right in pairwise(right_field_deltas))


def _thermal_optimizer_fixture(module_count: int, device: torch.device):
    from channelthermal.config import ChannelThermalHONFConfig
    from channelthermal.local_surrogate.model import LocalModuleConfig, LocalModuleSurrogate
    from channelthermal.model import ChannelThermalHONFModel

    core_payload = _payload(V4_ARCHITECTURE)
    core_payload.update(
        {
            "coordinate_scale": [12.0, 6.0],
            "boundary_feature_mode": "none",
            "num_env_tokens_x": 4,
            "num_env_tokens_y": 3,
        }
    )
    config = ChannelThermalHONFConfig.from_dict(
        {
            "core_honf": core_payload,
            "channelthermal": {
                "use_local_surrogate": True,
                "internal_prediction_mode": "local_surrogate",
                "local_surrogate_latent_dim": 16,
                "interaction_refinement_steps": 1,
                "default_num_interface_points": 4,
                "port_global_consistency_num_points": 4,
            },
        }
    )
    model = ChannelThermalHONFModel(config, attach_local_from_checkpoint=False)
    local_surrogate = LocalModuleSurrogate(
        LocalModuleConfig(
            hidden_dim=16,
            latent_dim=16,
            num_port_latents=2,
            num_heads=4,
            num_layers=1,
            coord_fourier_frequencies=2,
            dropout=0.0,
        )
    )
    model.local_coupling.set_local_surrogate(
        local_surrogate,
        freeze=True,
        normalization_config={},
        normalization_stats={},
    )
    model.to(device)
    model.set_training_progress(epoch=151, total_epochs=500)

    if module_count == 2:
        centers = torch.tensor([[[3.0, 1.5], [8.0, 4.5]]])
    else:
        centers = torch.tensor(
            [[
                [1.0, 1.0],
                [3.5, 1.0],
                [6.0, 1.0],
                [8.5, 1.0],
                [11.0, 1.0],
                [1.0, 5.0],
                [3.5, 5.0],
                [6.0, 5.0],
                [8.5, 5.0],
                [11.0, 5.0],
            ]]
        )
    centers = centers.to(device)
    batch_size, _, _ = centers.shape
    generator = torch.Generator().manual_seed(151000 + module_count)
    modules = int(centers.shape[1])
    ports = 4
    theta = torch.arange(ports, dtype=torch.float32) * (2.0 * torch.pi / ports)
    teacher = torch.zeros((batch_size, modules, ports, 5), dtype=torch.float32)
    teacher[..., 0] = theta[None, None, :]
    teacher[..., 1] = torch.cos(theta)[None, None, :]
    teacher[..., 2] = torch.sin(theta)[None, None, :]
    teacher[..., 3] = 23.0
    teacher[..., 4] = 5.0
    local_points = torch.tensor(
        [[[-0.2, 0.0], [0.15, 0.12], [0.1, -0.16]]], dtype=torch.float32
    ).expand(batch_size, modules, -1, -1).clone()
    query_xy = torch.tensor(
        [[[1.0, 0.7], [6.0, 2.1], [11.0, 3.1], [4.0, 5.4]]], dtype=torch.float32
    )
    batch = {
        "structure": {
            "module_centers": centers,
            "module_present": torch.ones((batch_size, modules), dtype=torch.float32),
            "heat_powers": torch.linspace(1.0, 2.0, modules)[None, :],
            "material_params": torch.tensor([[0.01, 0.02, 0.03, 1.0, 0.5, 0.45]]),
            "re": torch.tensor([[100.0]]),
            "u_in": torch.tensor([[1.0]]),
        },
        "query_xy": query_xy,
        "field_targets": torch.randn((batch_size, query_xy.shape[1], 5), generator=generator) * 0.1,
        "module_internal_query_points": local_points,
        "module_internal_temperature_points": 22.0 + torch.rand(
            (batch_size, modules, local_points.shape[-2]), generator=generator
        ),
        "interface_target": torch.stack(
            [torch.full((batch_size, modules, ports), 24.0), torch.full((batch_size, modules, ports), 0.1)],
            dim=-1,
        ),
        "teacher_port_tokens": teacher,
        "interface_condition_valid_mask": torch.ones(
            (batch_size, modules, ports), dtype=torch.float32
        ),
    }
    batch = {
        key: (
            {subkey: value.to(device) for subkey, value in value.items()}
            if key == "structure"
            else value.to(device)
        )
        for key, value in batch.items()
    }
    return model, batch


def _run_predicted_port_optimizer_update(module_count: int, device: torch.device) -> dict[str, object]:
    from channelthermal.training.epoch import make_model_inputs, run_epoch
    from channelthermal.training.optimizer import build_forward_optimizer

    torch.manual_seed(151100 + module_count)
    model, batch = _thermal_optimizer_fixture(module_count, device)
    model.eval()
    with torch.no_grad():
        model(
            **make_model_inputs(
                batch,
                local_port_condition_mode="predicted",
                mixed_teacher_ratio=0.0,
                return_predicted_port_outputs=True,
                return_port_global_consistency=True,
            )
        )
    optimizer, _ = build_forward_optimizer(
        model, {"learning_rate": 5.0e-4, "weight_decay": 0.0}
    )
    phase_plans: list[tuple[str, object]] = []
    original_prepare = model.core.prepare

    def capture_preparation(*args, **kwargs):
        prepared = original_prepare(*args, **kwargs)
        plan = prepared.backend_state.get("functional_tree_plan")
        catalogue = kwargs.get("functional_probes")
        if plan is not None:
            if plan.node_gamma.requires_grad:
                plan.node_gamma.retain_grad()
            phase_plans.append((catalogue.phase, plan))
        return prepared

    model.core.prepare = capture_preparation  # type: ignore[method-assign]
    metrics = run_epoch(
        model=model,
        loader=[batch],
        device=device,
        loss_cfg={
            "field_mse_weight": 1.0,
            "port_supervised_weight": 0.1,
            "port_smoothness_weight": 0.01,
            "port_global_consistency_weight": 0.05,
            "port_temperature_scale": 30.0,
            "port_h_weight": 0.1,
        },
        optimizer=optimizer,
        scaler=None,
        amp=False,
        max_batches=1,
        local_port_condition_mode="predicted",
        mixed_teacher_ratio=0.0,
        effective_internal_temperature_weight=0.2,
        effective_interface_weight=0.1,
        predicted_consistency_weight=0.05,
        gradient_clip_norm=0.0,
        record_gradient_diagnostics=True,
    )
    assert {phase for phase, _ in phase_plans} >= {"P0", "P1", "P2"}
    assert torch.isfinite(torch.tensor(metrics["loss_total"]))
    assert metrics["parameter_update_norm"] > 0.0
    assert metrics["parameter_update_norm_head"] > 0.0
    assert metrics["parameter_update_norm_backend"] > 0.0

    named_parameters = dict(model.named_parameters())
    port_head = [
        parameter
        for name, parameter in named_parameters.items()
        if name.startswith("local_coupling.port_head.") and parameter.requires_grad
    ]
    assert port_head
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in port_head)
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in port_head
    )
    action_parameters = [
        named_parameters[name]
        for name in (
            "core.backend.module_control_gain.weight",
            "core.backend.environment_score_control.weight",
            "core.backend.router.query_projection.0.weight",
        )
        if name in named_parameters and named_parameters[name].requires_grad
    ]
    assert action_parameters
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in action_parameters)
    action_gradient_norm = torch.sqrt(
        sum(parameter.grad.detach().double().square().sum() for parameter in action_parameters)
    )
    assert action_gradient_norm.item() > 0.0

    transition_gradient = 0.0
    transition_count = 0
    for _phase, plan in phase_plans:
        transition = (plan.node_gamma > 0.0) & (plan.node_gamma < 1.0)
        if bool(transition.any()):
            transition_count += int(transition.sum().detach().cpu())
            if plan.node_gamma.grad is not None:
                transition_gradient = max(
                    transition_gradient,
                    float(plan.node_gamma.grad[transition].abs().max().detach().cpu()),
                )
    if transition_count > 0:
        assert transition_gradient > 0.0
    return {
        "module_count": module_count,
        "loss": float(metrics["loss_total"]),
        "parameter_update_norm": float(metrics["parameter_update_norm"]),
        "port_head_update_norm": float(metrics["parameter_update_norm_head"]),
        "action_gradient_norm": float(action_gradient_norm.detach().cpu()),
        "phase_count": len(phase_plans),
        "transition_nodes": transition_count,
        "transition_gamma_gradient_max": transition_gradient,
        "device": str(device),
    }


def test_predicted_port_optimizer_updates_low_m_and_m10() -> None:
    summaries = [
        _run_predicted_port_optimizer_update(2, torch.device("cpu")),
        _run_predicted_port_optimizer_update(10, torch.device("cpu")),
    ]
    assert [entry["module_count"] for entry in summaries] == [2, 10]
    assert all(entry["parameter_update_norm"] > 0.0 for entry in summaries)

"""Physical-reader parity and no-probe contracts for Run-1503-v5."""

from __future__ import annotations

import copy
import json
from dataclasses import fields, replace
from pathlib import Path

import torch

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.interface_fields import InterfaceFieldCore
from honf_forward_core.interface_fields.types import PreparedInterfaceField


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V5_PROFILE = PROJECT_ROOT / "src/config_core/forward/task_trained_functional_coalescence_honf_context.json"
PARENT_PROFILE = PROJECT_ROOT / "src/config_core/forward/sparse_incidence_group_control_honf_context.json"
V5_ARCHITECTURE = "task_trained_functional_coalescence_honf"
PARENT_ARCHITECTURE = "sparse_incidence_group_control_honf"


def _config_payload(architecture: str) -> dict[str, object]:
    source = V5_PROFILE if architecture == V5_ARCHITECTURE else PARENT_PROFILE
    payload = copy.deepcopy(
        json.loads(source.read_text(encoding="utf-8"))["model"]["core_honf"]
    )
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
            "receiver_chunk_size": 8,
            "activation_checkpointing": False,
            "environment_refinement_normalizer": "sparsemax",
        }
    )
    return payload


def _batch(seed: int = 15037001, queries: int = 5) -> BatchData:
    generator = torch.Generator().manual_seed(seed)
    extent = torch.tensor([12.0, 6.0])
    return BatchData(
        module_centers=torch.rand(1, 12, 2, generator=generator) * extent,
        module_present=torch.ones(1, 12),
        module_features=torch.randn(1, 12, 3, generator=generator),
        global_context=torch.randn(1, 4, generator=generator),
        query_xy=torch.rand(1, queries, 2, generator=generator) * extent,
        query_time=None,
        target_field=torch.randn(1, queries, 5, generator=generator),
        case_name="run1503-v5-backend-test",
        metadata={},
        env_coords=torch.rand(1, 9, 2, generator=generator) * extent,
        env_features=torch.randn(1, 9, 2, generator=generator),
        env_weights=torch.rand(1, 9, generator=generator) + 0.5,
    )


def _core(architecture: str) -> InterfaceFieldCore:
    return InterfaceFieldCore(
        UnifiedForwardConfig.from_dict(_config_payload(architecture))
    ).cpu().eval()


def _backend_terms(
    core: InterfaceFieldCore,
    prepared: PreparedInterfaceField,
    receivers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, object]:
    backend = core.backend
    features = core._receiver_features(prepared, receivers)
    route = backend._route(
        prepared.backend_state,
        prepared.encoded,
        receivers,
        features,
    )
    module, _ = backend._read_module(
        prepared.backend_state,
        prepared.encoded,
        receivers,
        route,
        include_diagnostics=False,
    )
    environment, _ = backend._read_environment(
        prepared.backend_state,
        prepared.encoded,
        receivers,
        features,
        route,
        include_diagnostics=False,
    )
    return module, environment, route


def _prepare(
    core: InterfaceFieldCore,
    encoded,
    *,
    mode: str | None = None,
) -> PreparedInterfaceField:
    if mode is not None:
        core.backend.functional_detail_inference_mode = mode
    return core.prepare(encoded, encoded.module_tokens)


def _force_uniform_source_routes(core: InterfaceFieldCore) -> None:
    """Keep all K leaves active in a controlled physical-reader test."""

    router = core.backend.router
    with torch.no_grad():
        # Small distinct prototypes keep source and query logits in the dense
        # entmax/sparsemax region while retaining a live contrast signal.
        router.group_codes.mul_(1.0e-2)
    router.geometry_fraction = 1.0e6


def _set_controller_bias(core: InterfaceFieldCore, value: float) -> None:
    controller = core.backend.functional_detail_controller
    with torch.no_grad():
        controller.network[-1].weight.zero_()
        controller.network[-1].bias.fill_(float(value))


def _grads(
    loss: torch.Tensor,
    inputs: tuple[torch.Tensor, ...],
    parameters: tuple[torch.nn.Parameter, ...],
) -> tuple[torch.Tensor | None, ...]:
    return torch.autograd.grad(
        loss,
        (*inputs, *parameters),
        retain_graph=True,
        allow_unused=True,
    )


def _assert_gradients_close(
    left: tuple[torch.Tensor | None, ...],
    right: tuple[torch.Tensor | None, ...],
    *,
    atol: float = 2.0e-5,
    rtol: float = 2.0e-5,
) -> None:
    for left_gradient, right_gradient in zip(left, right, strict=True):
        assert (left_gradient is None) == (right_gradient is None)
        if left_gradient is not None:
            assert torch.isfinite(left_gradient).all()
            assert torch.isfinite(right_gradient).all()
            torch.testing.assert_close(
                left_gradient, right_gradient, atol=atol, rtol=rtol
            )


def test_virtual_all_open_matches_direct_parent_and_has_no_probe_scoring(
    monkeypatch,
) -> None:
    torch.manual_seed(15037002)
    parent = _core(PARENT_ARCHITECTURE)
    torch.manual_seed(15037002)
    candidate = _core(V5_ARCHITECTURE)
    candidate.set_training_progress(epoch=50, total_epochs=500)
    _force_uniform_source_routes(parent)
    _force_uniform_source_routes(candidate)
    _set_controller_bias(candidate, 8.0)

    batch = _batch()
    batch = replace(
        batch,
        module_centers=batch.module_centers.clone().requires_grad_(),
        env_coords=batch.env_coords.clone().requires_grad_(),
        query_xy=batch.query_xy.clone().requires_grad_(),
    )
    encoded = candidate.encode_case(batch)
    parent.backend.eval()
    candidate.backend.eval()
    # First pass materializes every lazy reader layer in the same order.
    parent_prepared = parent.prepare(encoded, encoded.module_tokens)
    candidate_prepared = _prepare(candidate, encoded, mode="virtual")
    receivers = batch.query_xy
    with torch.no_grad():
        for model, prepared in ((parent, parent_prepared), (candidate, candidate_prepared)):
            model.backend.read(
                prepared.backend_state,
                encoded,
                receivers,
                model._receiver_features(prepared, receivers),
            )
    # Reuse the Run-1502 backend tensors as the matched inherited checkpoint.
    parent_state = parent.backend.state_dict()
    candidate_state = candidate.backend.state_dict()
    shared = {
        name: value
        for name, value in parent_state.items()
        if name in candidate_state and tuple(value.shape) == tuple(candidate_state[name].shape)
    }
    incompatible = candidate.backend.load_state_dict(shared, strict=False)
    assert incompatible.unexpected_keys == []
    assert all(
        name.startswith("functional_detail_controller.")
        for name in incompatible.missing_keys
    )

    # At e50 the candidate delegates to the unchanged parent prepare/read
    # path, including bitwise predictions and first derivatives.
    candidate_e50 = _prepare(candidate, encoded, mode="virtual")
    assert "functional_detail_plan" not in candidate_e50.backend_state
    parent_context, _ = parent.backend.read(
        parent_prepared.backend_state,
        encoded,
        receivers,
        parent._receiver_features(parent_prepared, receivers),
    )
    candidate_context, _ = candidate.backend.read(
        candidate_e50.backend_state,
        encoded,
        receivers,
        candidate._receiver_features(candidate_e50, receivers),
    )
    torch.testing.assert_close(candidate_context, parent_context, atol=0.0, rtol=0.0)
    parent_module, parent_environment, _ = _backend_terms(
        parent, parent_prepared, receivers
    )
    candidate_module, candidate_environment, _ = _backend_terms(
        candidate, candidate_e50, receivers
    )
    e50_weights = torch.linspace(0.5, 1.5, parent_module.numel()).reshape_as(parent_module)
    e50_parent_loss = (parent_module * e50_weights).sum() + (
        parent_environment * e50_weights.flip(1)
    ).sum()
    e50_candidate_loss = (candidate_module * e50_weights).sum() + (
        candidate_environment * e50_weights.flip(1)
    ).sum()
    parameter_names = (
        "router.query_projection.0.weight",
        "module_control_gain.weight",
        "environment_value_control.weight",
        "environment_score_control.weight",
        "query_module_output.weight",
    )
    parent_parameters = dict(parent.backend.named_parameters())
    candidate_parameters = dict(candidate.backend.named_parameters())
    parent_selected = tuple(parent_parameters[name] for name in parameter_names)
    candidate_selected = tuple(candidate_parameters[name] for name in parameter_names)
    inputs = (
        encoded.module_tokens,
        encoded.env_tokens,
        encoded.module_centers,
        encoded.env_coords,
        receivers,
    )
    _assert_gradients_close(
        _grads(e50_parent_loss, inputs, parent_selected),
        _grads(e50_candidate_loss, inputs, candidate_selected),
        atol=0.0,
        rtol=0.0,
    )
    candidate.set_training_progress(epoch=150, total_epochs=500)

    router = candidate.backend.router
    logits_calls = 0
    original_query_logits = router.query_logits

    def count_logits(*args, **kwargs):
        nonlocal logits_calls
        logits_calls += 1
        return original_query_logits(*args, **kwargs)

    def forbidden_parent_sparsemax(*_args, **_kwargs):
        raise AssertionError("v5 must route from finite logits without parent sparsemax")

    monkeypatch.setattr(router, "query_logits", count_logits)
    monkeypatch.setattr(router, "route_queries", forbidden_parent_sparsemax)
    # Preparation computes conditional gates and exact R only.  It does not
    # evaluate persistent probe/query scores.
    candidate_prepared = _prepare(candidate, encoded, mode="virtual")
    assert logits_calls == 0
    detail = candidate_prepared.backend_state["functional_detail_plan"].detail
    assert torch.equal(detail.transform, torch.eye(12)[None])
    assert detail.actual_R.tolist() == [12]
    virtual_diagnostics = candidate.backend.preparation_aux(
        candidate_prepared.backend_state,
        include_diagnostics=True,
    )
    controls = candidate_prepared.backend_state["group_control_state"]
    expected_module_moment = torch.einsum(
        "bsk,bkd->bskd", controls.module_membership, controls.group_control
    )
    expected_environment_moment = torch.einsum(
        "bsk,bkd->bskd", controls.environment_membership, controls.group_control
    )
    torch.testing.assert_close(
        virtual_diagnostics["functional_detail_module_source_moment"],
        expected_module_moment.detach(),
    )
    torch.testing.assert_close(
        virtual_diagnostics["functional_detail_environment_source_moment"],
        expected_environment_moment.detach(),
    )

    parent_context, _ = parent.backend.read(
        parent_prepared.backend_state,
        encoded,
        receivers,
        parent._receiver_features(parent_prepared, receivers),
    )
    v5_context, _ = candidate.backend.read(
        candidate_prepared.backend_state,
        encoded,
        receivers,
        candidate._receiver_features(candidate_prepared, receivers),
    )
    assert logits_calls == 1
    torch.testing.assert_close(v5_context, parent_context, atol=2.0e-6, rtol=2.0e-6)

    # Measure first derivatives through source geometry, receiver geometry,
    # query routing, QM, QE, and the environmental value control.
    parent_module, parent_environment, _ = _backend_terms(
        parent, parent_prepared, receivers
    )
    v5_module, v5_environment, _ = _backend_terms(
        candidate, candidate_prepared, receivers
    )
    torch.testing.assert_close(v5_module, parent_module, atol=2.0e-6, rtol=2.0e-6)
    torch.testing.assert_close(
        v5_environment, parent_environment, atol=2.0e-6, rtol=2.0e-6
    )
    torch.testing.assert_close(
        candidate_prepared.backend_state["environment_values"],
        parent_prepared.backend_state["environment_values"],
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    weights = torch.linspace(0.5, 1.5, parent_module.numel()).reshape_as(parent_module)
    parent_loss = (parent_module * weights).sum() + (
        parent_environment * weights.flip(1)
    ).sum()
    candidate_loss = (v5_module * weights).sum() + (
        v5_environment * weights.flip(1)
    ).sum()
    parent_grads = _grads(parent_loss, inputs, parent_selected)
    candidate_grads = _grads(candidate_loss, inputs, candidate_selected)
    _assert_gradients_close(parent_grads, candidate_grads)


def test_exact_compact_and_virtual_readers_match_values_and_first_gradients() -> None:
    torch.manual_seed(15037003)
    core = _core(V5_ARCHITECTURE)
    core.set_training_progress(epoch=150, total_epochs=500)
    _force_uniform_source_routes(core)
    _set_controller_bias(core, -8.0)

    batch = _batch(15037004)
    batch = replace(
        batch,
        module_centers=batch.module_centers.clone().requires_grad_(),
        env_coords=batch.env_coords.clone().requires_grad_(),
        query_xy=batch.query_xy.clone().requires_grad_(),
    )
    encoded = core.encode_case(batch)
    core.backend.eval()
    virtual = _prepare(core, encoded, mode="virtual")
    compact = _prepare(core, encoded, mode="compact")
    virtual_plan = virtual.backend_state["functional_detail_plan"]
    compact_plan = compact.backend_state["functional_detail_plan"]
    assert virtual_plan.detail.actual_R.tolist() == [1]
    assert compact_plan.detail.actual_R.tolist() == [1]
    assert not virtual_plan.compact
    assert compact_plan.compact
    assert int(compact_plan.membership.shape[-1]) == 1
    assert "coalescence_plan" not in virtual.backend_state
    assert "coalescence_plan" in compact.backend_state
    assert compact.backend_state["functional_detail_execution_width"].tolist() == [1.0]

    receivers = batch.query_xy
    virtual_module, virtual_environment, virtual_route = _backend_terms(
        core, virtual, receivers
    )
    compact_module, compact_environment, compact_route = _backend_terms(
        core, compact, receivers
    )
    torch.testing.assert_close(
        virtual_route.assignment.sum(dim=-1), torch.ones(1, 5)
    )
    torch.testing.assert_close(
        compact_route.assignment.sum(dim=-1), torch.full((1, 5), 1.0 / 12.0)
    )
    torch.testing.assert_close(
        compact_route.query_mass,
        torch.ones_like(compact_route.query_mass),
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    for compact_value, virtual_value in (
        (compact_module, virtual_module),
        (compact_environment, virtual_environment),
        (compact.backend_state["environment_values"], virtual.backend_state["environment_values"]),
    ):
        torch.testing.assert_close(
            compact_value, virtual_value, atol=2.0e-5, rtol=2.0e-5
        )

    weights = torch.linspace(0.5, 1.5, virtual_module.numel()).reshape_as(virtual_module)
    virtual_loss = (virtual_module * weights).sum() + (
        virtual_environment * weights.flip(1)
    ).sum()
    compact_loss = (compact_module * weights).sum() + (
        compact_environment * weights.flip(1)
    ).sum()
    names = (
        "router.query_projection.0.weight",
        "module_control_gain.weight",
        "environment_value_control.weight",
        "environment_score_control.weight",
        "query_module_output.weight",
    )
    parameters = dict(core.backend.named_parameters())
    selected = tuple(parameters[name] for name in names)
    inputs = (
        encoded.module_tokens,
        encoded.env_tokens,
        encoded.module_centers,
        encoded.env_coords,
        receivers,
    )
    _assert_gradients_close(
        _grads(virtual_loss, inputs, selected),
        _grads(compact_loss, inputs, selected),
    )


def test_mixed_compaction_preserves_live_node_logit_gradients(monkeypatch) -> None:
    """A hard child closure must not detach the other live Haar scales."""

    torch.manual_seed(15037005)
    core = _core(V5_ARCHITECTURE)
    core.set_training_progress(epoch=150, total_epochs=500)
    _force_uniform_source_routes(core)
    batch = _batch(15037006)
    encoded = core.encode_case(batch)
    core.backend.eval()
    controller = core.backend.functional_detail_controller
    membership = controller.node_membership
    closed_index = int(torch.nonzero(membership.sum(dim=-1) == 2)[0, 0])
    base_logits = torch.zeros((1, controller.node_count))
    base_logits[0, closed_index] = -8.0
    node_logit_proxy = torch.zeros_like(base_logits, requires_grad=True)

    def proxy_logits(descriptors: torch.Tensor) -> torch.Tensor:
        return (base_logits + node_logit_proxy).expand(int(descriptors.shape[0]), -1)

    monkeypatch.setattr(controller, "forward", proxy_logits)
    virtual = _prepare(core, encoded, mode="virtual")
    compact = _prepare(core, encoded, mode="compact")
    virtual_detail = virtual.backend_state["functional_detail_plan"].detail
    compact_plan = compact.backend_state["functional_detail_plan"]
    assert int(virtual_detail.closed_nodes.sum()) == 1
    assert int(compact_plan.detail.closed_nodes.sum()) == 1
    assert virtual_detail.actual_R.tolist() == [11]
    assert compact_plan.detail.actual_R.tolist() == [11]
    assert bool(((virtual_detail.retention > 0.0) & (virtual_detail.retention < 1.0)).any())
    assert int(compact_plan.membership.shape[-1]) == 11

    receivers = batch.query_xy
    virtual_module, virtual_environment, _ = _backend_terms(core, virtual, receivers)
    compact_module, compact_environment, _ = _backend_terms(core, compact, receivers)
    torch.testing.assert_close(compact_module, virtual_module, atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(
        compact_environment, virtual_environment, atol=2.0e-5, rtol=2.0e-5
    )
    weights = torch.linspace(0.4, 1.6, virtual_module.numel()).reshape_as(virtual_module)
    virtual_loss = (virtual_module * weights).sum() + (
        virtual_environment * weights.flip(1)
    ).sum()
    compact_loss = (compact_module * weights).sum() + (
        compact_environment * weights.flip(1)
    ).sum()
    virtual_gradient = torch.autograd.grad(
        virtual_loss, node_logit_proxy, retain_graph=True
    )[0]
    compact_gradient = torch.autograd.grad(
        compact_loss, node_logit_proxy, retain_graph=True
    )[0]
    assert torch.isfinite(virtual_gradient).all()
    assert torch.isfinite(compact_gradient).all()
    assert torch.count_nonzero(virtual_gradient).item() > 0
    torch.testing.assert_close(
        compact_gradient, virtual_gradient, atol=3.0e-5, rtol=3.0e-5
    )


def test_auto_inference_uses_virtual_width_for_exact_closure() -> None:
    torch.manual_seed(15037007)
    core = _core(V5_ARCHITECTURE)
    core.set_training_progress(epoch=150, total_epochs=500)
    _force_uniform_source_routes(core)
    _set_controller_bias(core, -8.0)
    encoded = core.encode_case(_batch(15037008))
    core.backend.eval()
    assert core.backend.functional_detail_inference_mode == "auto"

    automatic = _prepare(core, encoded)
    automatic_plan = automatic.backend_state["functional_detail_plan"]
    assert automatic_plan.detail.actual_R.tolist() == [1]
    assert int(automatic_plan.detail.closed_nodes.sum()) == 11
    assert not automatic_plan.compact
    assert automatic.backend_state["functional_detail_execution_width"].tolist() == [
        12.0
    ]
    assert int(automatic_plan.membership.shape[-1]) == 12
    assert "coalescence_plan" not in automatic.backend_state

    virtual = _prepare(core, encoded, mode="virtual")
    virtual_plan = virtual.backend_state["functional_detail_plan"]
    torch.testing.assert_close(automatic_plan.membership, virtual_plan.membership)

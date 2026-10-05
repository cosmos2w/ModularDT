"""Native-context dispatch, full-access identity and live global task controls."""

from dataclasses import replace

import pytest
import torch

from honf_forward_core.config import BatchData, InterfaceFieldConfig, UnifiedForwardConfig
from honf_forward_core.interface_fields.common import SharedInterfaceContext
from honf_forward_core.interface_fields.core import InterfaceFieldCore
from honf_forward_core.interface_fields.global_control_hypergraph import GlobalControlHypergraph
from honf_forward_core.interface_fields.topology_probe import OrganizerTopologyProbe
from honf_forward_core.interface_fields.typed_hypergraph_field import TypedHypergraphField
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def _encoded(dimension=2):
    generator = torch.Generator().manual_seed(731)
    return EncodedInterfaceCase(
        module_tokens=torch.randn(2, 4, 8, generator=generator),
        env_tokens=torch.randn(2, 7, 8, generator=generator),
        global_token=torch.randn(2, 8, generator=generator),
        module_centers=torch.rand(2, 4, dimension, generator=generator),
        env_coords=torch.rand(2, 7, dimension, generator=generator),
        module_present=torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.]]),
        module_features=torch.randn(2, 4, 3, generator=generator), env_features=None,
        env_weights=torch.tensor([[1., .5, 2., 1., 3., .75, 1.]]).expand(2, -1),
        coordinate_scale=torch.ones(1, 1, dimension))


def _core(architecture, dimension):
    return InterfaceFieldCore(UnifiedForwardConfig(
        spatial_dim=dimension, coordinate_scale=[1.] * dimension, boundary_feature_mode="none",
        periodic_axes=[], field_dim=3, hidden_dim=8, forward_architecture=architecture,
        interface_model=InterfaceFieldConfig(message_hidden_dim=12, attention_heads=2,
            coarse_latent_count=4, coarse_blocks=1, relative_fourier_frequencies=2,
            receiver_chunk_size=3, hypergraph_options={"organizer_dim": 16})))


@pytest.mark.parametrize("dimension", [2, 3])
def test_native_context_initial_identity_matches_dense_common_and_gradients(dimension):
    encoded = _encoded(dimension)
    batch = BatchData(module_centers=encoded.module_centers, module_present=encoded.module_present,
        module_features=encoded.module_features, global_context=encoded.global_token,
        query_xy=torch.rand(2, 5, dimension), target_field=None, query_time=None,
        case_name="native-context", metadata={}, env_coords=encoded.env_coords,
        env_features=torch.randn(2, 7, 3), env_weights=encoded.env_weights)
    models = [_core(name, dimension).eval() for name in (
        "dense_pairwise_field", "native_context_tree_honf", "native_context_global_control_honf")]
    for model in models:
        assert isinstance(model.common, SharedInterfaceContext)
        record = model.encode_case(batch)
        model.decode_queries(model.prepare(record, record.module_tokens), batch.query_xy)
    reference = models[0].state_dict()
    for model in models[1:]:
        model.load_state_dict({**model.state_dict(), **reference})
    results = []
    for model in models:
        query = batch.query_xy.clone().requires_grad_()
        record = model.encode_case(batch)
        prepared = model.prepare(record, record.module_tokens)
        assert prepared.interaction_aux["coarse_latent_count"] == 4
        output = model.decode_queries(prepared, query)["pred_field"]
        output.square().sum().backward()
        results.append((output, query.grad))
    for output, gradient in results[1:]:
        torch.testing.assert_close(output, results[0][0], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(gradient, results[0][1], atol=3e-6, rtol=3e-5)
    for name, parameter in models[0].named_parameters():
        for candidate in models[1:]:
            other = dict(candidate.named_parameters())[name]
            if parameter.grad is None:
                assert other.grad is None
            else:
                torch.testing.assert_close(other.grad, parameter.grad, atol=3e-6, rtol=3e-5)


def test_global_unit_density_and_measure_weighted_control_refinement():
    encoded = _encoded()
    organizer = GlobalControlHypergraph(8, organizer_dim=16).eval()
    with torch.no_grad():
        for head in organizer.control_heads.values():
            head[-1].weight.normal_(std=0.1)
    state = organizer.prepare(encoded, encoded.module_tokens, phase=1)
    assert state.group_count == 1
    for tau, points in (("MM", encoded.module_centers), ("ME", encoded.module_centers),
                        ("EM", encoded.env_coords), ("QM", torch.rand(2, 5, 2)),
                        ("QE", torch.rand(2, 5, 2))):
        access = organizer.access(state, points, tau)
        assert torch.equal(access.density[access.support], torch.ones_like(access.density[access.support]))
        torch.testing.assert_close(access.weight[access.support], torch.ones_like(access.weight[access.support]))
    # Split every quadrature atom into two identical half-mass atoms.
    refined = replace(encoded, env_tokens=encoded.env_tokens.repeat_interleave(2, 1),
        env_coords=encoded.env_coords.repeat_interleave(2, 1),
        env_weights=encoded.env_weights.repeat_interleave(2, 1) / 2)
    refined_state = organizer.prepare(refined, refined.module_tokens, phase=1)
    for tau in state.controls:
        torch.testing.assert_close(state.controls[tau], refined_state.controls[tau], atol=2e-7, rtol=2e-6)
    assert state.export()["control_input_provenance"]
    assert set(state.strategy_data["typed_centres"]) == set(state.controls)


@pytest.mark.parametrize("dimension", [2, 3])
def test_global_unequal_atom_refinement_preserves_geometry_vjp_and_content_dependencies(dimension):
    encoded = _encoded(dimension)
    organizer = GlobalControlHypergraph(8, spatial_dim=dimension, organizer_dim=16).double().eval()
    with torch.no_grad():
        for head in organizer.control_heads.values():
            head[-1].weight.normal_(std=0.1)
    coords = encoded.env_coords.double().requires_grad_()
    tokens = encoded.env_tokens.double().requires_grad_()
    record = replace(encoded, module_tokens=encoded.module_tokens.double(), env_tokens=tokens,
        global_token=encoded.global_token.double(), module_centers=encoded.module_centers.double(),
        env_coords=coords, env_weights=encoded.env_weights.double(),
        module_present=encoded.module_present.double(), coordinate_scale=encoded.coordinate_scale.double())
    state = organizer.prepare(record, record.module_tokens)
    loss = sum(value.square().sum() for value in state.controls.values())
    parent_coords, parent_tokens = torch.autograd.grad(loss, (coords, tokens))
    child_coords = coords.detach().repeat_interleave(2, 1).requires_grad_()
    child_tokens = tokens.detach().repeat_interleave(2, 1).requires_grad_()
    fractions = coords.new_tensor([0.3, 0.7]).repeat(coords.shape[1])
    refined = replace(record, env_coords=child_coords, env_tokens=child_tokens,
        env_weights=record.env_weights.repeat_interleave(2, 1) * fractions)
    child_state = organizer.prepare(refined, refined.module_tokens)
    child_loss = sum(value.square().sum() for value in child_state.controls.values())
    coordinate_grad, token_grad = torch.autograd.grad(child_loss, (child_coords, child_tokens))
    for child_grad, parent_grad in ((coordinate_grad, parent_coords), (token_grad, parent_tokens)):
        torch.testing.assert_close(child_grad[:, ::2], parent_grad * 0.3, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(child_grad[:, 1::2], parent_grad * 0.7, atol=1e-12, rtol=1e-10)
    recomputed = organizer.recompute_controls(state, record, record.module_tokens)
    for tau in state.controls:
        torch.testing.assert_close(recomputed[tau], state.controls[tau], rtol=0, atol=0)
    changed = organizer.recompute_controls(state, replace(record, env_tokens=tokens + 1), record.module_tokens)
    assert any(not torch.equal(changed[tau], state.controls[tau]) for tau in state.controls)


@pytest.mark.parametrize("dimension", [2, 3])
def test_global_nonidentity_fine_read_refinement_and_topology_export(dimension):
    torch.manual_seed(39)
    model = TypedHypergraphField(8, 12, 2, 2, architecture="native_context_global_control_honf",
        spatial_dim=dimension, module_characteristic_length=0.1,
        options={"organizer_dim": 16}).double().eval()
    encoded = _encoded(dimension)
    record = replace(encoded, module_tokens=encoded.module_tokens.double(),
        global_token=encoded.global_token.double(), module_centers=encoded.module_centers.double(),
        module_present=encoded.module_present.double(), env_tokens=encoded.env_tokens.double(),
        env_coords=encoded.env_coords.double(), env_weights=encoded.env_weights.double(),
        coordinate_scale=encoded.coordinate_scale.double())
    query, features = torch.rand(2, 5, dimension, dtype=torch.float64), torch.rand(2, 5, 6, dtype=torch.float64)
    model.read(model.prepare(record, record.module_tokens), record, query, features)
    with torch.no_grad():
        for head in model.organizer.control_heads.values():
            head[-1].weight.normal_(std=0.1)
        for gain in model.control_gain.values():
            gain.weight.normal_(std=0.1)
    coords = record.env_coords.clone().requires_grad_()
    tokens = record.env_tokens.clone().requires_grad_()
    parent = replace(record, env_coords=coords, env_tokens=tokens)
    with OrganizerTopologyProbe(model.organizer) as capture:
        prepared = model.prepare(parent, parent.module_tokens)
        value, _ = model.read(prepared, parent, query, features)
    assert model.export_typed_state(prepared)["dependency_provenance"]["mode"] == "global_measure_weighted_v1"
    with OrganizerTopologyProbe(model.organizer, capture.record):
        replay, _ = model.read(model.prepare(parent, parent.module_tokens), parent, query, features)
    torch.testing.assert_close(value, replay, atol=0, rtol=0)
    parent_grad = torch.autograd.grad(value.square().sum(), (coords, tokens))
    children = [source.detach().repeat_interleave(2, 1).requires_grad_() for source in (coords, tokens)]
    fractions = coords.new_tensor([0.3, 0.7]).repeat(coords.shape[1])
    refined = replace(parent, env_coords=children[0], env_tokens=children[1],
        env_weights=parent.env_weights.repeat_interleave(2, 1) * fractions)
    child_value, _ = model.read(model.prepare(refined, refined.module_tokens), refined, query, features)
    torch.testing.assert_close(child_value, value, atol=1e-12, rtol=1e-10)
    child_grad = torch.autograd.grad(child_value.square().sum(), children)
    for derivative, original in zip(child_grad, parent_grad):
        torch.testing.assert_close(derivative[:, ::2], original * 0.3, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(derivative[:, 1::2], original * 0.7, atol=1e-12, rtol=1e-10)


def test_historical_faithful_core_keeps_three_term_context():
    from honf_forward_core.interface_fields.three_term_context import ThreeTermInterfaceContext
    assert isinstance(_core("faithful_receiver_hypergraph_honf", 2).common, ThreeTermInterfaceContext)


@pytest.mark.parametrize("execution", ["projected", "full_control"])
def test_global_ordinary_task_updates_reach_heads_and_geometry(execution):
    torch.manual_seed(59)
    model = TypedHypergraphField(8, 12, 2, 2, architecture="native_context_global_control_honf",
        spatial_dim=2, module_characteristic_length=0.1, options={"organizer_dim": 16}).train()
    model.control_execution = execution
    assert model.training_gradient_mode == "ordinary_task_v1"
    assert not model.supports_local_context_shadow
    encoded, query, features = _encoded(), torch.rand(2, 5, 2), torch.randn(2, 5, 6)
    model.read(model.prepare(encoded, encoded.module_tokens), encoded, query, features)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimizer_ids == {id(p) for p in model.parameters() if p.requires_grad}
    useful = set()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        states = encoded.module_tokens.clone().requires_grad_()
        record = replace(encoded, module_tokens=states)
        prepared = model.prepare(record, states)
        output, aux = model.read(prepared, record, query, features)
        assert "hypergraph_structural_numerator" not in aux
        assert model.last_structural_cost is None
        output.square().mean().backward()
        assert states.grad is not None and torch.isfinite(states.grad).all()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and parameter.grad.abs().sum() > 0:
                useful.add(name)
        optimizer.step()
    for tau in ("MM", "ME", "EM", "QM", "QE"):
        assert f"organizer.control_heads.{tau}.2.weight" in useful
        assert f"control_gain.{tau}.weight" in useful
    assert "organizer.control_geometry_encoder.0.weight" in useful
    for parameter in model.control_score.parameters():
        assert not parameter.requires_grad and parameter.grad is None
        assert torch.count_nonzero(parameter) == 0

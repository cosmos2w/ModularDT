"""Generic dimensional/invariance and physical adapter contracts for routed reads."""
from dataclasses import replace

import pytest
import torch

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.interface_fields.core import InterfaceFieldCore


class BoxGeometry:
    def __init__(self, dimension):
        self.length_scale = (1.8,) * dimension
        self.bounds = ((0., 8.),) * dimension

    def features(self, points):
        return points / 8.

    def resistance(self, a, b, relation_type):
        return (a - b).sum(-1) * 0.


def make_case(dimension):
    torch.manual_seed(41)
    batch = BatchData(
        module_centers=torch.rand(2, 4, dimension) * 8.,
        module_present=torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 1.]]),
        module_features=torch.randn(2, 4, 3), global_context=torch.randn(2, 5),
        query_xy=torch.rand(2, 7, dimension) * 8., query_time=None, target_field=None,
        case_name='generic', metadata={}, env_coords=torch.rand(2, 9, dimension) * 8.,
        env_features=torch.randn(2, 9, 3), env_weights=torch.rand(2, 9) + .2,
        routing_geometry=BoxGeometry(dimension),
    )
    config = UnifiedForwardConfig.from_dict({
        'forward_architecture': 'routed_pairwise_honf', 'spatial_dim': dimension,
        'hidden_dim': 16, 'field_dim': 3, 'coordinate_scale': [8.] * dimension,
        'boundary_feature_mode': 'none', 'periodic_axes': [],
        'interface_model': {'message_hidden_dim': 12, 'attention_heads': 2,
            'coarse_latent_count': 3, 'receiver_chunk_size': 3,
            'routing': {'descriptor_dim': 8, 'router_hidden_dim': 12, 'fine_pair_chunk_size': 11}},
    })
    return InterfaceFieldCore(config).eval(), batch


def predict(core, batch):
    encoded = core.encode_case(batch)
    prepared = core.prepare(encoded, encoded.module_tokens)
    return core.decode_queries(prepared, batch.query_xy)['pred_field']


@pytest.mark.parametrize('dimension', [2, 3])
def test_generic_chunk_batch_module_permutation_and_padding(dimension):
    core, batch = make_case(dimension)
    with torch.no_grad():
        expected = predict(core, batch)
        permutation = torch.tensor([3, 1, 0, 2])
        permuted = replace(batch, module_centers=batch.module_centers[:, permutation],
            module_features=batch.module_features[:, permutation], module_present=batch.module_present[:, permutation])
        torch.testing.assert_close(predict(core, permuted), expected, atol=2e-6, rtol=2e-5)
        padded = replace(batch,
            module_centers=torch.cat((batch.module_centers, torch.full((2, 2, dimension), float('nan'))), 1),
            module_features=torch.cat((batch.module_features, torch.full((2, 2, 3), float('nan'))), 1),
            module_present=torch.cat((batch.module_present, torch.zeros(2, 2)), 1))
        torch.testing.assert_close(predict(core, padded), expected, atol=2e-6, rtol=2e-5)
        singleton = replace(batch, **{key: value[:1] for key, value in vars(batch).items() if torch.is_tensor(value)})
        torch.testing.assert_close(predict(core, singleton), expected[:1], atol=2e-6, rtol=2e-5)
        encoded = core.encode_case(batch)
        prepared = core.prepare(encoded, encoded.module_tokens)
        whole = core.decode_queries(prepared, batch.query_xy, receiver_chunk_size=64)['pred_field']
        torch.testing.assert_close(whole, expected, atol=2e-6, rtol=2e-5)
    query = batch.query_xy.clone().requires_grad_()
    centers = batch.module_centers.clone().requires_grad_()
    output = predict(core, replace(batch, query_xy=query, module_centers=centers))
    gradients = torch.autograd.grad(output.square().sum(), (query, centers))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)


def test_empty_module_case_uses_only_neutral_environment_candidate():
    core, batch = make_case(2)
    batch = replace(batch, module_present=torch.tensor([[0., 0., 0., 0.], [1., 0., 0., 0.]]))
    with torch.no_grad():
        encoded = core.encode_case(batch)
        prepared = core.prepare(encoded, encoded.module_tokens)
        index = prepared.backend_state['routing_index']
        assert index.candidates.valid.sum(-1).tolist() == [1, 1]
        assert index.candidates.candidate_origin[0, index.candidates.valid[0]].tolist() == [-1]
        assert torch.count_nonzero(index.module_incidence.membership[0]) == 0
        torch.testing.assert_close(index.environment_incidence.hub_measure.sum(-1), torch.ones(2, dtype=torch.float64))
        output = core.decode_queries(prepared, batch.query_xy)['pred_field']
        assert torch.isfinite(output).all()
        empty_alone = replace(batch, **{key: value[:1] for key, value in vars(batch).items() if torch.is_tensor(value)})
        torch.testing.assert_close(predict(core, empty_alone), output[:1], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("strategy", ["module_hubs", "mean_shift"])
def test_physical_passes_refresh_live_routing_with_frozen_local_surrogate(strategy):
    from pathlib import Path

    from channelthermal.config import ChannelThermalHONFConfig
    from channelthermal.model import ChannelThermalHONFModel

    local_checkpoint = Path(__file__).resolve().parents[1] / 'Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_0000_base/best_model.pt'
    if not local_checkpoint.is_file():
        pytest.skip('The established frozen local surrogate is unavailable.')
    config = ChannelThermalHONFConfig.from_dict({
        'core_honf': {'forward_architecture': 'routed_pairwise_honf', 'hidden_dim': 32,
            'domain_length_x': 12., 'domain_length_y': 4., 'coordinate_scale': [12., 6.],
            'module_radius': .45, 'num_env_tokens_x': 4, 'num_env_tokens_y': 3,
            'boundary_feature_mode': 'none', 'interface_model': {
                'message_hidden_dim': 24, 'attention_heads': 4, 'receiver_chunk_size': 8,
                'routing': {'strategy': strategy, 'descriptor_dim': 8, 'router_hidden_dim': 12,
                            'fine_pair_chunk_size': 32}}},
        'channelthermal': {'use_local_surrogate': True, 'freeze_local_surrogate': True,
            'local_surrogate_checkpoint_path': str(local_checkpoint),
            'internal_prediction_mode': 'local_surrogate', 'interaction_refinement_steps': 1,
            'default_num_interface_points': 8},
    })
    model = ChannelThermalHONFModel(config).eval()
    original_prepare = model.core.backend.prepare
    preparations = []
    def capture(*args, **kwargs):
        state = original_prepare(*args, **kwargs)
        preparations.append((getattr(model.core, '_interface_read_role', None), state))
        return state
    model.core.backend.prepare = capture
    centers = torch.tensor([[[3., 1.5], [8., 2.5]]], requires_grad=True)
    heat = torch.tensor([[1., 2.]], requires_grad=True)
    output = model(query_xy=torch.tensor([[[1., 1.], [6., 2.], [11., 3.]]]),
        re=torch.tensor([[100.]]), u_in=torch.tensor([[1.]]), module_centers=centers,
        heat_powers=heat, module_present=torch.ones(1, 2),
        material_params=torch.tensor([[.01, .02, .03, 1., .5, .45]]),
        local_port_condition_mode='predicted', return_prepared_state=True)
    assert [role for role, _ in preparations] == ['p0_port', 'p1_refinement', 'p2_field']
    assert len({id(state) for _, state in preparations}) == 3
    assert all(state['module_tokens'].requires_grad and state['env_tokens'].requires_grad for _, state in preparations)
    objective = output['pred_field'].square().mean() + output['pred_port_condition'].square().mean()
    gradients = torch.autograd.grad(objective, (centers, heat))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    assert all(not p.requires_grad for p in model.local_coupling.local_surrogate.parameters())


def test_checkpointed_tiles_preserve_multi_tile_outputs_and_gradients():
    import copy
    core, batch = make_case(2)
    with torch.no_grad():
        predict(core, batch)
    checkpointed = copy.deepcopy(core).train()
    core.train()
    checkpointed.backend.activation_checkpointing = True
    core.backend.activation_checkpointing = False
    a = predict(core, batch)
    b = predict(checkpointed, batch)
    torch.testing.assert_close(a, b)
    a.square().sum().backward()
    b.square().sum().backward()
    reference = dict(core.named_parameters())
    for name, parameter in checkpointed.named_parameters():
        expected = reference[name].grad
        if expected is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, expected, atol=2e-6, rtol=2e-5, msg=name)

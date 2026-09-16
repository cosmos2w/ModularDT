"""Routing configuration stays opt-in and preserves historical serialization."""
import copy

import pytest
import torch
from channelthermal.environment import ChannelThermalEnvironmentBuilder
from channelthermal.routing_geometry import ChannelThermalRoutingGeometry

from honf_forward_core.config import BatchData, RoutingIndexConfig, UnifiedForwardConfig
from honf_runtime.config_loader import load_config_bundle


def test_routing_profile_preserves_dense_scientific_settings():
    dense = load_config_bundle('project://src/config_core/forward/dense_pairwise_interface_context.json').effective
    routed = load_config_bundle('project://src/config_core/forward/routing_module_hubs_context.json').effective
    a, b = copy.deepcopy(dense['model']['core_honf']), copy.deepcopy(routed['model']['core_honf'])
    assert b.pop('forward_architecture') == 'routed_pairwise_honf'
    a.pop('forward_architecture')
    assert b['interface_model'].pop('routing')['query_normalizer'] == 'source_measure_sparsemax'
    assert a == b
    assert dense['training'] == routed['training']
    assert routed['run']['id'] == '2000'
    assert routed['training']['epochs'] == 500


def test_routing_serialization_and_invalid_settings():
    for architecture in ('legacy_honf', 'dense_pairwise_field'):
        args = {} if architecture == 'legacy_honf' else {'forward_architecture': architecture, 'interface_model': {}}
        config = UnifiedForwardConfig.from_dict(args)
        assert 'routing' not in config.to_dict().get('interface_model', {})
    config = UnifiedForwardConfig.from_dict({'forward_architecture': 'routed_pairwise_honf', 'interface_model': {'routing': {}}})
    assert isinstance(config.interface_model.routing, RoutingIndexConfig)
    assert UnifiedForwardConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()
    for bad in ({'temperature': 0}, {'temperature': float('nan')}, {'strategy': 'dictionary'}, {'fine_pair_chunk_size': 0}, {'query_normalizer': 'sparsemax'}, {'execution': 'dense'}):
        with pytest.raises(ValueError):
            RoutingIndexConfig.from_dict(bad)
    optimized = RoutingIndexConfig.from_dict({'execution': 'optimized_exact'})
    assert optimized.execution == 'optimized_exact'
    roundtrip = UnifiedForwardConfig.from_dict(
        {'forward_architecture': 'routed_pairwise_honf', 'interface_model': {'routing': vars(optimized)}}
    )
    assert UnifiedForwardConfig.from_dict(roundtrip.to_dict()).to_dict() == roundtrip.to_dict()
    with pytest.raises(ValueError, match='requires interface_model.routing'):
        UnifiedForwardConfig.from_dict({'forward_architecture': 'routed_pairwise_honf', 'interface_model': {}})


def test_optimized_exact_quickcheck_profiles_are_explicit_and_matched():
    profiles = (
        ('routing_module_hubs_optimized_context.json', 'module_hubs', '2001', 'routed_module_hubs_quickcheck_optimized'),
        ('routing_mean_shift_optimized_context.json', 'mean_shift', '2101', 'routed_mean_shift_quickcheck_optimized'),
    )
    for filename, strategy, run_id, run_name in profiles:
        bundle = load_config_bundle(f'project://src/config_core/forward/{filename}')
        core_honf = bundle.core['model']['core_honf']
        routing = core_honf['interface_model']['routing']
        assert routing['strategy'] == strategy
        assert routing['execution'] == 'optimized_exact'
        assert bundle.core['training']['epochs'] == 50
        assert bundle.core['run']['id'] == run_id
        assert bundle.core['run']['name'] == run_name
        config = UnifiedForwardConfig.from_dict(core_honf)
        assert config.interface_model.routing.execution == 'optimized_exact'

    historical = load_config_bundle('project://src/config_core/forward/routing_module_hubs_context.json')
    assert historical.core['model']['core_honf']['interface_model']['routing']['execution'] == 'gathered'


def test_channel_geometry_is_known_descriptors_and_neutral_resistance():
    geometry = ChannelThermalRoutingGeometry(12., 4., .45)
    env = ChannelThermalEnvironmentBuilder()(batch_size=2, num_env_tokens_x=4, num_env_tokens_y=3,
        domain_length_x=12., domain_length_y=4., device=torch.device('cpu'), dtype=torch.float64)
    torch.testing.assert_close(geometry.features(env.env_coords), env.env_features)
    a = env.env_coords.clone().requires_grad_()
    b = torch.zeros_like(a).requires_grad_()
    resistance = geometry.resistance(a, b, 'query')
    assert torch.count_nonzero(resistance) == 0
    grads = torch.autograd.grad(resistance.sum(), (a, b))
    assert all(torch.count_nonzero(g) == 0 for g in grads)
    batch = BatchData(a, torch.ones(2,12), a, torch.zeros(2,1), a, None, None, 'test', {}, routing_geometry=geometry)
    assert batch.to('cpu').routing_geometry is geometry
    assert BatchData.from_dict(vars(batch)).routing_geometry is geometry
    assert 'routing_geometry' not in batch.to_dict()

"""Real native Wind scene observations without dataset copies or fitting."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2
from honf_forward_core.models import relation_anchored_config
from honf_runtime.observation_blocks import (
    extract_observations,
    observation_response_losses,
    prepare_observation_sampling,
    sample_observation_blocks,
    union_observation_dependencies,
)
from honf_runtime.observation_response_task import execute_observation_dependencies
from windfarm.data import WindFarmNativeView
from windfarm.normalization import VelocityNormalizer, VerticalProfileBaseline
from windfarm.relation_anchored import WindRelationAnchoredAdapter
from windfarm.splits import make_group_split
from windfarm.training.observation_response import WindObservationBinding


def test_real_train_scenes_extraction_and_joint_predictor_without_fitting():
    scipy_interpolate = pytest.importorskip('scipy.interpolate')
    source = Path(__file__).resolve().parents[1] / 'Dataset/links/wind_farm'
    if not (source / 'family_volume/run_shape.npy').is_file():
        pytest.skip('The separately stored Wind source dataset is unavailable.')
    view = WindFarmNativeView(source)
    split = make_group_split(np.asarray(view.metadata['layout_index']), seed=42)
    normalizer = VelocityNormalizer(np.zeros(3), np.ones(3), np.ones(3), u_ref_mps=9.)
    provider = SimpleNamespace(view=view, train_rows=split.train, validation_rows=split.validation,
                               seed=42, adapter=SimpleNamespace(normalizer=normalizer))
    binding = WindObservationBinding(provider, population_size=12)
    pair = binding.training_pairs()[0]
    assert float(view.metadata['wd_deg'][int(pair[0])]) == float(view.metadata['wd_deg'][int(pair[1])])
    block, = binding.observation_blocks(pair)
    assert block.references.shape == (2, 12, 3)
    assert block.partition == 'train'
    assert block.time == 'steady_final_solution'
    assert block.source_times == tuple(view.volume.run(int(row)).source_time for row in pair)
    assert block.absolute_mask.all() and block.pair_mask.all()
    positions_m = np.asarray(block.coordinates, dtype=np.float64) * view.diameter_m
    for index, row in enumerate(pair):
        run = view.volume.run(int(row))
        interpolation = scipy_interpolate.RegularGridInterpolator((run.z_m, run.y_m, run.x_m),
                                                                  run.U_structured, bounds_error=True)
        independently_gathered = interpolation(positions_m[:, ::-1])
        np.testing.assert_allclose(block.references[index], independently_gathered, atol=1.e-12, rtol=1.e-12)

    blocks = (block,)
    plan = prepare_observation_sampling(blocks)
    draws = sample_observation_blocks(plan, 3, np.random.default_rng(9))
    union = union_observation_dependencies(blocks, draws)
    native_values = torch.tensor(np.asarray([view.volume.run(int(dep.scene_id)).U[int(dep.native_id)]
                                             for dep in union.dependencies]), dtype=torch.float64)[None]
    gathered = torch.stack([extract_observations(block, scene, native_values, union.mappings[0][scene])
                            for scene in range(2)], dim=1)
    used = {draw.seed_index for draw in draws.draws} | {draw.neighbor_index for draw in draws.draws
                                                        if draw.neighbor_index is not None}
    for scene in range(2):
        np.testing.assert_allclose(gathered.detach().numpy()[0, scene, sorted(used)],
                                   block.references[scene, sorted(used)], atol=1.e-12, rtol=1.e-12)

    profile = VerticalProfileBaseline(np.array([0., 7.]), np.zeros((2, 3)), np.ones(2, dtype=np.int64))
    base = ContentHypergraphCoreV2(module_input_dim=2, environment_input_dim=7, coordinate_dim=3,
        query_input_dim=7, operating_input_dim=11, output_dim=3, architecture='A2-D', context_dim=8,
        num_heads=2, context_depth=2, local_hidden_dim=8, dropout=0., init_seed=42).get_config()
    config = relation_anchored_config('relation_anchored', base, input_dataset='wind', query_tile_size=32)
    model = WindRelationAnchoredAdapter(normalizer, profile, core_config=config, environment_shape=(2, 2, 2))
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        output, work = execute_observation_dependencies(binding, model, pair, union)
        predicted = torch.stack([extract_observations(block, scene, output, union.mappings[0][scene])
                                 for scene in range(2)], dim=1)
        losses = observation_response_losses(blocks, (predicted,), draws)
        total = sum(losses[name] for name in ('extra_absolute', 'point_response', 'spatial_response'))
        total.backward()
    finally:
        torch.set_num_threads(previous_threads)
    assert output.shape == (2, len(union.dependencies), 3)
    assert work['scene_preparations'] == 2 and work['neural_queries'] == len(union.dependencies)
    assert torch.isfinite(total) and any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

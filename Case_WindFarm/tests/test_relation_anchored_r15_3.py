import json
from pathlib import Path

import numpy as np
import pytest
import torch

from windfarm.training.relation_anchored_r15_3_task import GRIDS, validate_wind_r15_3_recipe
from test_relation_anchored_wind import _adapter, _case


@pytest.mark.parametrize('shape', GRIDS)
def test_dense_grid_physical_measures_coverage_and_strict_reload(shape):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = _adapter(shape)
    case = _case()
    scene = model.make_scene(case, module_ids=[7, 19])
    extent = torch.as_tensor(case.support.extent_D, dtype=scene.environment_coordinates.dtype)
    assert scene.num_environment_tokens == np.prod(shape)
    torch.testing.assert_close(scene.environment_measure.sum(), extent.prod())
    torch.testing.assert_close(scene.environment_characteristic_lengths[0, 0], extent / torch.tensor(shape))
    assert scene.module_ids.tolist() == [[7, 19]]
    lower = torch.as_tensor(case.support.lower_D, dtype=extent.dtype)
    coords = torch.stack((lower, lower + extent, lower + extent * .43))[None]
    query = model.make_query(scene, coords)
    torch.testing.assert_close(query.anchor_weights.sum(-1), torch.ones_like(query.anchor_weights[..., 0]))
    assert (query.anchor_weights >= 0).all()
    restored = type(model).from_state_payload(model.state_payload(), state_dict=model.state_dict()).double()
    assert restored.environment_shape == shape
    assert all(torch.equal(model.state_dict()[k], restored.state_dict()[k]) for k in model.state_dict())


def test_r15_3_recipe_contract_and_seed_shared_tensors():
    recipes = [json.loads(Path(f'Case_WindFarm/configs/research/wind_r15_3_w{n}_train72.json').read_text())
               for n in (128, 256, 576, 1152)]
    for recipe, shape in zip(recipes, GRIDS):
        assert tuple(validate_wind_r15_3_recipe(recipe)['environment_shape']) == shape
    wrong = dict(recipes[-1], primary_queries=4096)
    with pytest.raises(ValueError):
        validate_wind_r15_3_recipe(wrong)
    wrong = dict(recipes[-1], microbatch_size=3, gradient_accumulation=False)
    with pytest.raises(ValueError):
        validate_wind_r15_3_recipe(wrong)
    states=[]
    for shape in GRIDS:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            states.append(_adapter(shape).state_dict())
    assert all(all(torch.equal(states[0][k], state[k]) for k in states[0]) for state in states[1:])


def test_validation_batch_limit_is_separate_from_training_and_legacy():
    from windfarm.training.content_hypergraph_task import WindContentHypergraphTaskV2
    task = object.__new__(WindContentHypergraphTaskV2)
    task.max_microbatch_cases = 1
    task.validation_microbatch_cases = 4
    task.recipe = {'schema': 'wind_relation_anchored_r15_3_v1'}
    with pytest.raises(ValueError, match='cannot repeat'):
        task._make_batch([1, 1], None, validation=True, query_count=4096)
    with pytest.raises(ValueError, match='case limit'):
        task._make_batch([1, 1], None, validation=False, query_count=8192)
    task.recipe = {'schema': 'wind_relation_anchored_r15_2_v2'}
    with pytest.raises(ValueError, match='case limit'):
        task._make_batch([1, 1], None, validation=True, query_count=4096)

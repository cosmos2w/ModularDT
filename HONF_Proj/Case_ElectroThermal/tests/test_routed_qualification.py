"""Exercise the routed case contract on real TRAIN/DEV inputs without fitting."""
from dataclasses import replace
import io
from pathlib import Path

import pytest
import torch

from hetmf.e_anchor_locator import attach_e_coverage
from hetmf.routed_qualification import select_panel
from hetmf.training import HETMFTask, build_model, read_config
from honf_runtime.field_training import evaluate, field_loss

DATA = Path('/data/public/ModularDT-HETMF-Dataset/v1/standard.h5')
CONFIG = Path(__file__).parents[1] / 'configs/r13_10_routed_faithful_standard.yaml'


@pytest.mark.skipif(not DATA.exists(), reason='shared Standard archive unavailable')
@pytest.mark.parametrize('config_path', [CONFIG, CONFIG.with_name('r13_10_routed_selected_standard.yaml')])
def test_real_routed_prepare_paired_gradients_and_reload(config_path):
    torch.set_num_threads(1)
    config = read_config(config_path)
    if config_path != CONFIG:
        assert config['model']['access_measure'] == 'inventory_unit_total'
        assert config['model']['ee_incidence_aggregation'] == 'post_gate'
    task = HETMFTask(DATA, config)
    data = task.dataset('train', 0)
    batch = task.collate([data[0], data[len(data) - 1]])
    data.close()
    scene = replace(batch.scene,
        module_features=batch.scene.module_features.detach().requires_grad_(),
        environment_features=batch.scene.environment_features.detach().requires_grad_(),
        module_coordinates=batch.scene.module_coordinates.detach().requires_grad_(),
        environment_coordinates=batch.scene.environment_coordinates.detach().requires_grad_())
    query = replace(batch.query, coordinates=batch.query.coordinates.detach().requires_grad_())
    batch = replace(batch, scene=scene, query=query)
    model = build_model(config)
    prediction = task.predict(model, batch)
    assert len(prediction) == 2 and all(x.shape == batch.targets.shape for x in prediction)
    field_loss(prediction, batch).backward()
    for value in (scene.module_features, scene.environment_features, scene.module_coordinates,
                  scene.environment_coordinates, query.coordinates):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    assert scene.module_features.grad[..., 2].abs().sum() > 0  # voltage
    assert all(scene.environment_features.grad[..., i].abs().sum() > 0 for i in range(3))
    memory = io.BytesIO()
    torch.save({'config': model.get_config(), 'state': model.state_dict()}, memory)
    memory.seek(0)
    payload = torch.load(memory, weights_only=True)
    restored = type(model).from_config(payload['config'])
    restored.load_state_dict(payload['state'])
    model.eval(); restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(task.predict(model, batch), task.predict(restored, batch))
        result = evaluate(restored, task, config['training'], split='dev')
    assert result['cases'] == 16 and len(result['physical_rmse_by_channel']) == 5
    assert len(select_panel(DATA, 'train', 32)) == 32
    assert len(select_panel(DATA, 'dev', 16)) == 16


@pytest.mark.skipif(not DATA.exists(), reason='shared Standard archive unavailable')
def test_local_cover_uses_live_coordinates_and_reordered_cells():
    torch.set_num_threads(1)
    config = read_config(CONFIG)
    task = HETMFTask(DATA, config)
    data = task.dataset('train', 0)
    batch = task.collate([data[0]])
    data.close()
    query = replace(batch.query, coordinates=torch.tensor([[[.3, .4], [0., 1.]]], requires_grad=True),
                    query_mask=torch.ones((1, 2), dtype=torch.bool), query_ids=None)
    attached = attach_e_coverage(query, batch.scene, 'local_e_cover')
    torch.testing.assert_close(attached.anchor_weights.sum(-1), torch.ones((1, 2)))
    (attached.anchor_weights[0, 0] * torch.arange(4)).sum().backward()
    assert query.coordinates.grad[0, 0].abs().sum() > 0
    perm = torch.randperm(144)
    scene = replace(batch.scene, environment_coordinates=batch.scene.environment_coordinates[:, perm],
                    environment_mask=batch.scene.environment_mask[:, perm])
    reordered = attach_e_coverage(query, scene, 'local_e_cover')
    torch.testing.assert_close(perm[reordered.anchor_indices[0, 0]], attached.anchor_indices[0, 0])


@pytest.mark.skipif(not DATA.exists(), reason='shared Standard archive unavailable')
def test_checkpoint_evaluator_uses_saved_case_config_and_train_normalizer(tmp_path):
    from types import SimpleNamespace
    from hetmf.training import evaluate_command
    config = read_config(CONFIG)
    config['training']['validation_queries'] = 8
    task = HETMFTask(DATA, config)
    model = build_model(config)
    checkpoint = tmp_path / 'latest.pt'
    torch.save({'binding': {'config': config, 'dataset': task.identity()},
                'model': model.state_dict(), 'epoch': 0}, checkpoint)
    result = evaluate_command(SimpleNamespace(run=str(tmp_path), checkpoint='latest.pt',
                              dataset=str(DATA), queries=8, split='dev', device='cpu'))
    assert result['cases'] == 16 and result['checkpoint_epoch'] == 0
    assert len(result['physical_rmse_by_channel']) == 5


@pytest.mark.skipif(not DATA.exists(), reason='shared Standard archive unavailable')
def test_local_cover_real_decode_backward():
    torch.set_num_threads(1)
    config = read_config(CONFIG)
    config['model']['locator_mode'] = 'local_e_cover'
    task = HETMFTask(DATA, config)
    data = task.dataset('train', 0)
    batch = task.collate([data[0]])
    data.close()
    model = build_model(config)
    full, restricted = task.predict(model, batch)
    assert full.shape == restricted.shape == batch.targets.shape
    field_loss((full, restricted), batch).backward()
    assert model.member_router.joint_head.weight.grad.abs().sum() > 0

"""Cross-case zero-fit qualification of relative EE gates on HET-MF inputs."""
import copy
import io
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from hetmf.training import HETMFTask, build_model, read_config
from honf_runtime.field_training import field_loss

DATA = Path('/data/public/ModularDT-HETMF-Dataset/v1/standard.h5')
CONFIG = Path(__file__).parents[1] / 'configs/r13_10_routed_selected_standard.yaml'


def qualify(mode):
    torch.set_num_threads(1)
    config = read_config(CONFIG)
    config['training']['queries'] = 17
    config['model'].update(ee_gate_mode=mode, ee_initial_tau=1.4907824993133545)
    task = HETMFTask(DATA, config)
    dataset = task.dataset('train', 0)
    try:
        sample = copy.deepcopy(dataset[0])
    finally:
        dataset.close()
    records = sample['inputs']['environment_records'].copy()
    # A deterministic numerical irregular-area fixture, with conserved total area;
    # material features and geometry remain the archive's actual TRAIN inputs.
    original_total = records[:, 4].sum()
    factors = np.linspace(0.5, 1.5, len(records))
    records[:, 4] *= factors
    records[:, 4] *= original_total / records[:, 4].sum()
    sample['inputs']['environment_records'] = records
    batch = task.collate([sample])
    scene = replace(batch.scene,
        environment_features=batch.scene.environment_features.detach().requires_grad_(),
        environment_measure=batch.scene.environment_measure.detach().requires_grad_())
    batch = replace(batch, scene=scene)
    assert torch.unique(scene.environment_measure).numel() > 1
    assert all(torch.unique(scene.environment_features[..., i]).numel() > 1 for i in range(3))
    model = build_model(config).train()
    outputs = task.predict(model, batch)
    assert len(outputs) == 2 and all(x.shape == (1, 17, 5) for x in outputs)
    loss = field_loss(outputs, batch)
    loss.backward()
    env_grad = scene.environment_features.grad
    measure_grad = scene.environment_measure.grad
    assert torch.isfinite(loss) and env_grad is not None and torch.isfinite(env_grad).all()
    assert all(env_grad[..., i].abs().sum() > 0 for i in range(3))
    assert measure_grad is not None and torch.isfinite(measure_grad).all() and measure_grad.abs().sum() > 0
    gate = model.applied_incidence.routes.ee
    assert gate.raw_tau.grad is not None and torch.isfinite(gate.raw_tau.grad) and gate.raw_tau.grad.abs() > 0
    memory = io.BytesIO()
    torch.save({'config': model.get_config(), 'state': model.state_dict()}, memory)
    memory.seek(0)
    saved = torch.load(memory, weights_only=True)
    restored = type(model).from_config(saved['config'])
    restored.load_state_dict(saved['state'], strict=True)
    model.eval(); restored.eval()
    with torch.no_grad():
        a, b = task.predict(model, batch), task.predict(restored, batch)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    return {'mode': mode, 'state_id': str(sample['metadata']['state_id']),
            'fields': list(a.shape), 'area_min': float(scene.environment_measure.min().detach()),
            'area_max': float(scene.environment_measure.max().detach()),
            'area_total': float(scene.environment_measure.sum().detach()),
            'environment_feature_gradient_l1': [float(env_grad[..., i].abs().sum()) for i in range(3)],
            'measure_gradient_l1': float(measure_grad.abs().sum()),
            'tau_gradient': float(gate.raw_tau.grad), 'saved_reload_max_abs': float((a-b).abs().max()),
            'fitting_performed': False, 'area_fixture': 'synthetic irregular positive measures with conserved total; archive material features'}


@pytest.mark.skipif(not DATA.exists(), reason='shared Standard archive unavailable')
@pytest.mark.parametrize('mode', ['relative_quadratic', 'relative_sigmoid'])
def test_heterogeneous_inputs_and_irregular_areas_through_routed_core(mode):
    qualify(mode)


@pytest.mark.parametrize('mode', ['relative_quadratic', 'relative_sigmoid'])
def test_relative_gate_preserves_statistics_when_a_donor_measure_is_split(mode):
    from honf_forward_core.interaction_reused_member_routing.core import AppliedInteractionIncidence

    policy = AppliedInteractionIncidence(2).double()
    policy.set_relative_ee_gate(mode, 1.4907824993133545)
    scores = torch.tensor([[[[0.1, 0.5, -0.4]], [[0.3, -0.2, 0.6]]]], dtype=torch.double)
    measure = torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.double)
    before = policy(scores, torch.ones(1, 1, 3, dtype=torch.bool), 'ee', donor_measure=measure)
    split_scores = scores[..., [0, 1, 1, 2]]
    split_measure = torch.tensor([[0.2, 0.2, 0.3, 0.3]], dtype=torch.double)
    after = policy(split_scores, torch.ones(1, 1, 4, dtype=torch.bool), 'ee', donor_measure=split_measure)
    torch.testing.assert_close(after, before[..., [0, 1, 1, 2]], atol=1e-12, rtol=1e-12)

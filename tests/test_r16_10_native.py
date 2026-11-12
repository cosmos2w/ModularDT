import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from channelthermal.training.r16_10_native_task import (
    _module_mean, build_thermal_r16_10_native_task, stratified_validation_ids,
)


def test_stratified_sample_and_module_reduction():
    cases = [{'case_id': str(i), 'structure': {'module_present': np.ones(i % 6 + 1)}} for i in range(89)]
    panel = stratified_validation_ids(cases)
    assert len(panel) == 9 and panel == stratified_validation_ids(cases)
    assert list(map(int, panel)) == sorted(map(int, panel))
    error = torch.tensor([[[1., 3.], [2., 20.]]])
    mask = torch.tensor([[[True, True], [True, False]]])
    assert torch.allclose(_module_mean(error, mask), torch.tensor([4.5]))


def test_actual_native_validation_paths():
    config = json.loads(Path('configs/research/thermal_r16_10_b0_train150.json').read_text())
    if not all(Path(config['dataset'][key]).is_file() for key in ('data_path', 'manifest_path')):
        pytest.skip('Requires the local Thermal dataset and fixed TRAIN150 manifest')
    model, task = build_thermal_r16_10_native_task(config)
    assert len(task.validation_cases) == 9
    b = next(task.validation_batches())
    b = replace(b,
        scene_inputs=replace(b.scene_inputs, structure={k: v[:1] for k, v in b.scene_inputs.structure.items()}, case_ids=b.scene_inputs.case_ids[:1]),
        receivers=replace(b.receivers, fluid_xy=b.receivers.fluid_xy[:1], local_query_points=b.receivers.local_query_points[:1], heat=b.receivers.heat[:1]),
        targets=replace(b.targets, case_ids=b.targets.case_ids[:1], case_indices=b.targets.case_indices[:1], field_targets=b.targets.field_targets[:1], point_weights=b.targets.point_weights[:1], interface_target=b.targets.interface_target[:1], interface_valid_mask=b.targets.interface_valid_mask[:1], material_targets=b.targets.material_targets[:1], outside_temperature=b.targets.outside_temperature[:1], outside_valid=b.targets.outside_valid[:1]))
    assert b.targets.interface_target.shape[-2:] == (64, 2)
    model.eval()
    with torch.no_grad():
        for mode in ('full', 'restricted'):
            with model.core.read_execution(mode):
                pred, aux = task.predict_native(model, b.scene_inputs, b.receivers, model.architecture, 'joint')
            assert aux['work'][f'member_routing_read_mode_{mode}'] == 1
            terms = task.validation_loss_terms(pred, b.targets, {})
            assert len(terms) == 9 and terms['port/outside_temperature'].weight == .05
            assert all(torch.isfinite(term.numerator) for term in terms.values())
    assert model.core.query_tile_size == 128
    model.train()
    prediction, _ = task.predict_native(model, b.scene_inputs, b.receivers, model.architecture, 'joint')
    terms = task.loss_terms(prediction, b.targets, 'joint', {})
    objective = sum(term.numerator * term.weight / term.denominator for term in terms.values())
    objective.backward()
    assert torch.isfinite(objective)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in model.parameters())
    schedule = task.optimizer_groups(model, model.architecture, 'joint')[0].schedule
    assert schedule.value(1000) == 3e-4 and schedule.value(2000) == 3e-5

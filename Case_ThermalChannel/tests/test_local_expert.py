from types import SimpleNamespace

import pytest
import torch

from channelthermal.training.local_expert_task import OrganizedTemperatureHead, prescribed_h, ThermalLocalExpertTask
from channelthermal.local_ports import FrozenThermalLocal


def structure():
    return {"module_centers": torch.tensor([[[2., 3.], [4., 3.]]]),
            "domain_length_x": torch.tensor([[12.]]), "domain_length_y": torch.tensor([[6.]]),
            "material_params": torch.tensor([[.001, .1, .01, 10., 1., .45]]),
            "module_present": torch.tensor([[1., 0.]])}


def test_prescribed_h_and_organized_head_gradients():
    s = structure()
    h = prescribed_h(s, 256, 128)
    assert torch.allclose(h, torch.tensor([[2 * 10 / 11 / (6 / 128)]]))
    head = OrganizedTemperatureHead(2, 2.)
    fields = torch.zeros(1, 2, 64, 4, requires_grad=True)
    cm = torch.randn(1, 2, 64, 2, requires_grad=True)
    ce = torch.randn(1, 2, 64, 2, requires_grad=True)
    t, angular = head((cm, ce), fields, s, torch.ones(1, 2))
    assert t.shape == (1, 2, 64) and angular.shape == (1, 2, 64, 3)
    t.sum().backward()
    assert cm.grad.abs().sum() > 0 and ce.grad.abs().sum() > 0 and fields.grad.abs().sum() > 0


def test_frozen_local_input_gradient_without_weight_gradients():
    checkpoint = 'Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_0001_20260812_143809_thermal_disk_local/checkpoints/best_total.pt'
    from pathlib import Path
    if not Path(checkpoint).exists():
        pytest.skip('Retained local checkpoint is unavailable')
    local = FrozenThermalLocal(checkpoint)
    s = structure()
    theta = torch.arange(64) * (2 * torch.pi / 64)
    t = torch.full((1, 2, 64), 20., requires_grad=True)
    a = torch.stack((theta, theta.cos(), theta.sin()), -1)[None, None].expand(1, 2, -1, -1)
    ports = torch.cat((a, t[..., None], torch.full_like(t[..., None], 14.8)), -1)
    out = local(s, torch.ones(1, 2), ports, torch.zeros(1, 3, 2))
    (out['internal_temperature'].sum() + out['interface_pred'].sum()).backward()
    assert torch.isfinite(t.grad).all() and t.grad[0, 0].abs().sum() > 0
    assert t.grad[0, 1].abs().sum() == 0
    assert all(not p.requires_grad and p.grad is None for p in local.parameters())


def test_expert_loss_uses_sampled_angles_and_valid_outside():
    task = object.__new__(ThermalLocalExpertTask)
    task.local_weight, task.outside_weight = 1., .1
    task.port_statistics = {'std': [2.]}
    task._role_losses = lambda pred, targets: {
        'temperature/material': pred.native_main['pred_internal_temperature'].sum((1, 2, 3)),
        'temperature/surface': pred.native_main['pred_interface'][..., 0].sum((1, 2)),
        'q_proxy': pred.native_main['pred_interface'][..., 1].sum((1, 2))}
    native = ThermalContentExpertDummy()
    native.expert = {'internal_temperature': torch.ones(1, 1, 1, 1),
                     'interface_pred': torch.arange(64.).view(1, 1, 64, 1).expand(-1, -1, -1, 2)}
    native.expert_ports = torch.zeros(1, 1, 64, 5)
    native.expert_valid = torch.zeros(1, 1, 64, dtype=torch.bool)
    native.expert_valid[..., 0] = True
    native.expert_present = torch.ones(1, 1)
    targets = SimpleNamespace(case_ids=('x',), interface_target=torch.zeros(1, 1, 16, 2),
                              teacher_ports=torch.zeros(1, 1, 64, 5), outside_valid=native.expert_valid)
    targets.teacher_ports[..., 0, 3] = 4
    terms = task._expert_terms(native, targets)
    assert terms['expert/temperature/surface'].numerator == sum(range(0, 64, 4))
    assert terms['expert/outside_temperature'].numerator == 4
    assert terms['expert/outside_temperature'].denominator == 1


class ThermalContentExpertDummy:
    native_prepared = None
    native_main = {}


def test_local_native_masks_scales_and_equal_live_modules():
    from channelthermal.training.content_hypergraph_task import ThermalContentHypergraphTask, ThermalContentPredictions
    task = object.__new__(ThermalContentHypergraphTask)
    task.model = SimpleNamespace(field_std=torch.ones(5), interface_target_std=torch.tensor([2., 4.]),
                                 internal_temperature_std=torch.tensor([2.]))
    valid = lambda value: SimpleNamespace(valid=torch.tensor(value, dtype=torch.bool))
    prep = SimpleNamespace(source_present=torch.tensor([[1., 1., 0.]]), stencils={
        'fluid': valid([[True]]),
        'surface': valid([[[True, True], [True, False], [True, True]]]),
        'outside': valid([[[True, False], [True, True], [True, True]]]),
        'material': valid([[[True, False], [True, True], [True, True]]])})
    pred = {'pred_field': torch.zeros(1, 1, 5),
            'pred_interface': torch.tensor([[[[2., 4.], [200., 400.]], [[4., 8.], [200., 400.]], [[200., 400.], [200., 400.]]]]),
            'pred_internal_temperature': torch.tensor([[[[2.], [200.]], [[4.], [4.]], [[200.], [200.]]]])}
    target = SimpleNamespace(field_targets=torch.zeros(1, 1, 5), point_weights=torch.ones(1, 1),
                             interface_target=torch.zeros(1, 3, 2, 2), interface_valid_mask=torch.tensor([[[1, 0], [1, 1], [1, 1]]]),
                             material_targets=torch.zeros(1, 3, 2))
    roles = task._role_losses(ThermalContentPredictions(prep, pred), target)
    for role in ('temperature/material', 'temperature/surface', 'q_proxy'):
        assert torch.allclose(roles[role], torch.tensor([2.5]))


def test_source_transfer_rejects_missing_model_tensor(monkeypatch):
    import copy
    import json
    from pathlib import Path
    from channelthermal.training import local_expert_task as module
    recipe_path = Path('configs/research/thermal_r15_2_h_frozen_train150_v1.json')
    recipe = json.loads(recipe_path.read_text())
    if not Path(recipe['local_expert']['source_checkpoint']).exists():
        pytest.skip('Retained T0 TRAIN150 checkpoint is unavailable')
    original = module.load_trusted_checkpoint
    def incomplete(*args, **kwargs):
        source = original(*args, **kwargs)
        source = copy.deepcopy(source)
        del source['model_state_dict']['core.module_raw_gate']
        return source
    monkeypatch.setattr(module, 'load_trusted_checkpoint', incomplete)
    with pytest.raises(RuntimeError, match='Missing key'):
        module.build_thermal_local_expert_task(recipe)


def test_sliced_trace_work_and_invalid_interpolation(monkeypatch):
    from channelthermal.training import local_expert_task as module
    from channelthermal.native_observations import NativeStencil
    class Core:
        query_tile_size = 2
        context_dim = 1
        def decode(self, org, query, **kwargs):
            n = query.num_queries
            return SimpleNamespace(joint_fields=torch.zeros(1, n, 4),
                trace={'context_M': torch.ones(1, n, 1), 'context_E': torch.ones(1, n, 1)},
                work_counts={'actual_pairs': n * 3})
    class Adapter:
        core = Core()
        field_mean = torch.tensor([10., 11., 12., 0., 13.])
        field_std = torch.ones(5)
        def validate_prepared(self, prepared):
            assert prepared.organization is None
        def _query_union(self, *args):
            stencil = NativeStencil(indices=torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]]),
                                    weights=torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.]]]),
                                    valid=torch.tensor([[True, False]]))
            return SimpleNamespace(num_queries=3), SimpleNamespace(unique_counts=torch.tensor([3])), {'outside': stencil}, {'outside': (1, 2)}
    monkeypatch.setattr(module, 'slice_query_v1', lambda q, start, stop: SimpleNamespace(num_queries=stop-start))
    prepared = SimpleNamespace(scene_batch=None, fluid_xy=torch.zeros(1, 1, 2), organization=None,
                               source_present=torch.ones(1, 1))
    _, fields, valid, rows, counts = module.organized_features(Adapter(), prepared, frozen=True)
    assert rows == 3 and counts['actual_pairs'] == 9
    assert valid.tolist() == [[[True, False]]]
    assert fields[0, 0, 0].tolist() == [10., 11., 12., 13.]
    assert fields[0, 0, 1].tolist() == [0., 0., 0., 0.]


def test_completed_epoch_records_transfer_source_and_phase_scope(monkeypatch):
    from channelthermal.training.r13_8_anchored_task import ThermalR138AnchoredMemberRoutingTask
    def inherited(self, model, engine, epoch, output, row):
        row['reader_epoch_context'] = {'source_kind': 'scratch'}
        row['paired_reader_native_losses'] = {'epoch_context': {'source_kind': 'scratch'}}
    monkeypatch.setattr(ThermalR138AnchoredMemberRoutingTask, 'completed_training_epoch', inherited)
    task = object.__new__(ThermalLocalExpertTask)
    task.recipe = {'initialization': {'source_kind': 'thermal_local_expert_transfer', 'source_parent_age': 100},
                   'local_expert': {'source_checkpoint': '/source/t0_e100.pt'}}
    task.phase = 'frozen'
    row = {}
    task.completed_training_epoch(None, None, 7, None, row)
    context = row['reader_epoch_context']
    assert context['source_kind'] == 'thermal_local_expert_transfer'
    assert context['source_parent_completed_epoch'] == 100 and context['phase_completed_epoch'] == 7
    assert context['additional_epochs_after_parent'] == 7 and context['source_checkpoint'] == '/source/t0_e100.pt'
    assert row['paired_reader_native_losses']['epoch_context'] == context
    assert 'only organized condition head is trainable' in row['local_expert_training_prediction_mode']
    task.phase = 'joint'
    task.completed_training_epoch(None, None, 1, None, row)
    assert 'global core and condition head trainable' in row['local_expert_training_prediction_mode']


def test_joint_loss_metadata_covers_direct_and_local_objectives():
    task = object.__new__(ThermalLocalExpertTask)
    task.phase = 'joint'
    task.native_weights = {'flow/u': .125, 'temperature/material': 1 / 6, 'q_proxy': .05}
    task.local_weight, task.outside_weight = .1, .01
    metadata = task.loss_metadata()
    assert set(metadata) == {*task.native_weights, 'expert/temperature/material',
                             'expert/temperature/surface', 'expert/q_proxy',
                             'expert/outside_temperature'}
    assert float(metadata['q_proxy']['weight']) == .05
    assert float(metadata['expert/q_proxy']['weight']) == pytest.approx(.1 / 3)
    assert 'full and restricted' in metadata['temperature/material']['formula']
    task.phase = 'frozen'
    assert set(task.loss_metadata()) == {name for name in metadata if name.startswith('expert/')}


def test_full_train_expert_starts_fresh_without_loading_global_weights(monkeypatch):
    import json
    from pathlib import Path
    from channelthermal.training import local_expert_task as module

    recipe = json.loads(Path('configs/research/thermal_r15_2_h_joint_train600_full1000_v1.json').read_text())
    model = torch.nn.Linear(2, 2)
    configured = []
    provider = SimpleNamespace(configure_expert=lambda value: configured.append(value))
    def build(base, **kwargs):
        assert base['training']['membership'] == 'full_train600'
        assert base['initialization']['pretrained_weights'] is False
        assert base['initialization']['source_parent_age'] == 0
        return model, provider
    monkeypatch.setattr(module, '_build_thermal_member_routing_task', build)
    def no_transfer(*args, **kwargs):
        raise AssertionError('Fresh full-data construction must not load global transfer weights')
    monkeypatch.setattr(module, 'load_trusted_checkpoint', no_transfer)
    assert module.build_thermal_local_expert_task(recipe) == (model, provider)
    assert configured == [recipe]


@pytest.mark.parametrize('change', ['frozen_phase', 'transfer_source', 'pretrained'])
def test_full_train_expert_rejects_transfer_initialization(change):
    import json
    from pathlib import Path
    from channelthermal.training.local_expert_task import build_thermal_local_expert_task

    recipe = json.loads(Path('configs/research/thermal_r15_2_h_joint_train600_full1000_v1.json').read_text())
    if change == 'frozen_phase':
        recipe['local_expert']['phase'] = 'frozen'
    elif change == 'transfer_source':
        recipe['local_expert']['source_checkpoint'] = '/old_subset_weights.pt'
    else:
        recipe['initialization']['pretrained_weights'] = True
    with pytest.raises(ValueError, match='start fresh'):
        build_thermal_local_expert_task(recipe)

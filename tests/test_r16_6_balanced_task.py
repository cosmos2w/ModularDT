import math
import torch
from channelthermal.training.r16_6_balanced_task import action_slots, calibration_slots, initialize_balanced_weights
from channelthermal.training.thermal_state_response_task import response_loss_terms


def test_action_placement_preserves_epoch_weight_and_calibration_allocation():
    slots = action_slots(13)
    assert [i for i in range(13) if slots.count(i) == 2] == [0, 4, 8]
    assert calibration_slots() == tuple(i for i in range(13) if i != 11)
    for n in (4, 13, 25):
        assert math.isclose(sum(n / 16 for _ in action_slots(n)) / n, 1.)
        assert all(0 <= i < n for i in action_slots(n))


def test_response_weights_are_applied_once():
    t = torch.tensor(1., requires_grad=True)
    terms = response_loss_terms(t, t, {'J': t, 'P': t}, multiplier=13/16,
                                finite_response_coefficient=.004)
    assert [terms[k].weight for k in terms] == [.25*13/16, .8*.004*13/16,
                                               .1*.004*13/16, .1*.004*13/16]


def test_source_inventory_accepts_only_named_message_extension():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(1, 1)
            self.context_blocks = torch.nn.ModuleList([torch.nn.ModuleDict({'mm': torch.nn.Module()})])
            self.context_blocks[0]['mm'].pair_message = torch.nn.Module()
            self.context_blocks[0]['mm'].pair_message.output = torch.nn.Linear(1, 1, bias=False)
    model = Toy()
    source = {k: v.clone() for k, v in model.state_dict().items() if 'pair_message' not in k}
    inventory = initialize_balanced_weights('M', model, source)
    assert inventory['new_names'] == ['context_blocks.0.mm.pair_message.output.weight']


def test_only_m_can_extend_with_unchanged_checkpoint_cadence():
    import pytest
    from test_r16_5_warm_fork import identities
    from channelthermal.training.r16_6_balanced_task import make_balanced_recipe
    from relation_anchored_train import engine_for

    p64, h0, _ = identities()
    def recipe(arm):
        return make_balanced_recipe(p64, h0, arm=arm, coefficient=.0014978225391285025,
            name=f'test_{arm}', output_root='/tmp', membership='full_train600',
            local_horizon=100, first_segment_epochs=100)
    control, message = recipe('C'), recipe('M')
    for candidate in (control, message):
        initial = engine_for(candidate, device='cpu')
        assert initial.config.total_epochs == 100
        assert initial.config.checkpoint_epochs == (100,)
    with pytest.raises(ValueError, match='Unsupported explicit same-lineage horizon extension'):
        engine_for(control, device='cpu', execution_horizon=500)
    extended = engine_for(message, device='cpu', execution_horizon=500)
    assert extended.config.total_epochs == 500
    assert extended.config.checkpoint_epochs == (100, 200, 300, 400, 500)


def test_snapshot_partition_covers_every_parameter_and_records_message_gradient():
    from channelthermal.training.r16_6_balanced_task import balanced_snapshot_groups
    from channelthermal.training.r16_5_decision_task import _accumulate_gradients, _gradient_report

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = torch.nn.Module()
            self.core.context_blocks = torch.nn.ModuleList([torch.nn.ModuleDict({'mm': torch.nn.Module()})])
            route = self.core.context_blocks[0]['mm']
            route.value = torch.nn.Linear(1, 1, bias=False)
            route.pair_message = torch.nn.Module()
            route.pair_message.output = torch.nn.Linear(1, 1, bias=False)
            self.other = torch.nn.Linear(1, 1, bias=False)
    model = Model()
    groups = balanced_snapshot_groups(model)
    pairs = [pair for group in groups.values() for pair in group]
    names = [name for name, _ in pairs]
    assert len(names) == len(set(names))
    assert set(names) == {name for name, p in model.named_parameters() if p.requires_grad}
    buffers = {name: {} for name in ('endpoint', 'delta', 'decision', 'native', 'port')}
    objective = sum(parameter.sum() for _, parameter in pairs)
    _accumulate_gradients(buffers['delta'], objective, groups)
    report = _gradient_report(buffers, {**groups, 'all_trainable': pairs})
    message_name = 'core.context_blocks.0.mm.pair_message.output.weight'
    assert groups['message_parameters'][0][0] == message_name
    assert torch.equal(buffers['delta'][message_name], torch.ones_like(model.core.context_blocks[0]['mm'].pair_message.output.weight))
    assert report['message_parameters']['norms']['delta'] == 1.
    assert math.isclose(report['all_trainable']['norms']['delta'], math.sqrt(len(pairs)))

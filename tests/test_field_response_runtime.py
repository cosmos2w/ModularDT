import numpy as np
import pytest
import torch

from honf_runtime.state_response import (FieldResponsePair, NativePairScene, attach_field_response,
                                         field_response_losses, select_degree_two_native_pairs)


def test_ragged_three_dimensional_fields_weighted_and_differentiable():
    # Distinct list lengths and masks represent observations from 3-D fields.
    first = torch.tensor([[[[1., 2.], [3., 4.], [5., 6.]],
                           [[2., 4.], [4., 6.], [6., 8.]]]], requires_grad=True)
    second = torch.tensor([[[[1., 2.]], [[3., 4.]]]], requires_grad=True)
    refs = torch.zeros((2, 3, 2))
    refs[:, 2, 1] = float('nan')
    pairs = [FieldResponsePair(first, refs, [[1, 1], [1, 1], [1, 0]],
                               [[.2, .25], [.3, .75], [.5, 0.]]),
             FieldResponsePair(second, torch.zeros((2, 1, 2)), [[1, 0]], [[1., 0.]])]
    result = field_response_losses(pairs, [1., 2.], [2., 3.])
    expected_first_abs = 2 * (.2*2.5 + .3*12.5 + .5*30.5) + 3 * (.25*2.5 + .75*6.5)
    expected_first_diff = 2 * 1 + 3 * 1
    assert result['absolute'].item() == pytest.approx((expected_first_abs + 10.) / 2)
    assert result['difference'].item() == pytest.approx((expected_first_diff + 8.) / 2)
    assert result['eligible_pair_counts'].tolist() == [2, 1]
    assert result['effective_channel_weights'].tolist() == [2., 1.5]
    (result['absolute'] + result['difference']).backward()
    assert torch.isfinite(first.grad).all() and torch.isfinite(second.grad).all()


def test_fixed_mass_not_renormalized_and_nonfinite_valid_prediction_fails():
    pred = torch.ones((1, 2, 1, 1), requires_grad=True)
    pair = FieldResponsePair(pred, np.zeros((2, 1, 1)), [[True]], [[2.]])
    result = field_response_losses([pair], [1.], [1.])
    assert result['absolute'].item() == 2.
    assert result['difference'].item() == 0.
    with pytest.raises(ValueError, match='finite'):
        field_response_losses([FieldResponsePair(pred * float('nan'), np.zeros((2, 1, 1)), [[True]], [[1.]])], [1.], [1.])


def test_degree_two_extension_respects_group_distance_range_and_split():
    scenes = [NativePairScene(str(x), 'train', 'group', x) for x in [0, 1, 2, 3, 5, 7]]
    scenes += [NativePairScene('other', 'validation', 'group', 1.5)]
    match, extended, edges = select_degree_two_native_pairs(scenes, lambda x, y: abs(x-y), lambda x, y: True)
    assert match == (('0', '1'), ('2', '3'), ('5', '7'))
    assert ('1', '2') in extended
    assert all('other' not in edge for edge in extended)
    degrees = {scene.case_id: sum(scene.case_id in edge for edge in extended) for scene in scenes}
    assert max(degrees.values()) <= 2
    lengths = {(left, right): distance for left, right, distance in edges}
    assert all(1 <= lengths[edge] <= 2 for edge in extended)
    assert select_degree_two_native_pairs(reversed(scenes), lambda x, y: abs(x-y), lambda x, y: True) == (match, extended, edges)


def test_wind_ragged_3d_linear_reference_extraction():
    from types import SimpleNamespace
    from windfarm.field_response import saved_velocity_pair
    from windfarm.splits import make_group_split

    groups = np.repeat(np.arange(5), 2)
    train = make_group_split(groups, seed=42).train
    first = int(train[0])
    second = first + 1 if first % 2 == 0 else first - 1

    class Dataset:
        _arrays = {'layout_index': groups}

        def run(self, index):
            x = np.array([0., 1., 2., 3.]) if index % 2 else np.array([0., 1.5, 3.])
            y = np.array([0., 2., 4.])
            z = np.array([0., 1., 2.])
            zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
            velocity = np.stack((xx + 2*yy + zz + index, 2*xx - zz, yy + 3*zz), axis=-1)
            return SimpleNamespace(x_m=x, y_m=y, z_m=z, U_structured=velocity,
                                   layout_index=int(groups[index]), wind_direction_deg=float(index % 2), case=str(index))

    positions = np.array([[.5, 1., .5], [2., 3., 1.5], [4., 1., 1.]])
    out = saved_velocity_pair(Dataset(), first, second, positions)
    assert out['mask'].tolist() == [[True]*3, [True]*3, [False]*3]
    assert out['masses'][:, 0].tolist() == [.5, .5, 0.]
    for row, index in enumerate((first, second)):
        expected = np.column_stack((positions[:2, 0] + 2*positions[:2, 1] + positions[:2, 2] + index,
                                    2*positions[:2, 0] - positions[:2, 2],
                                    positions[:2, 1] + 3*positions[:2, 2]))
        np.testing.assert_allclose(out['references'][row, :2], expected)
    with pytest.raises(ValueError, match='TRAIN'):
        saved_velocity_pair(Dataset(), first, int(make_group_split(groups, seed=42).test[0]), positions)


def test_generic_attachment_uses_existing_hook_and_pair_weights():
    from types import SimpleNamespace
    provider = SimpleNamespace()
    attach_field_response(provider, pairs=[('scene_one', 'scene_two'), ('scene_three', 'scene_four')],
                          count=2, seed=12, component_names=['absolute', 'difference'],
                          components=lambda model, pair, epoch: ({'absolute': model, 'difference': model * 2}, {'pair': pair}),
                          calibration=lambda: {'absolute': .25, 'difference': .1}, ramp=lambda epoch: .5)
    emitted = [entry for index in range(4) for entry in
               provider.auxiliary_update_terms(torch.tensor(3.), epoch=1, update_index=index, update_count=4)]
    assert len(emitted) == 2
    assert provider.auxiliary_loss_names == ('field_response/absolute', 'field_response/difference')
    assert emitted[0][0]['field_response/absolute'].weight == .25
    assert emitted[0][0]['field_response/difference'].weight == .1

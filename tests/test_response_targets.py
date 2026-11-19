import copy
import numpy as np
import torch
import pytest
from types import SimpleNamespace

from honf_runtime.local_response_contrasts import CommonObservations, WeightedResponseContrast, neighborhood_field_losses
from honf_runtime.response_targets import solve_bounded_rbf_target
from channelthermal.training.thermal_local_response import thermal_reynolds_target, ThermalLocalResponseObjective
from test_thermal_local_response import config


def test_weighted_rows_and_gradients_match_direct_arithmetic():
    pred = torch.randn(2, 4, 3, 2, dtype=torch.float64, requires_grad=True)
    ref = torch.randn(4, 3, 2, dtype=torch.float64)
    rows = torch.tensor([[1., -1., 0., 0.], [0., .3, -.3, 0.]], dtype=torch.float64)
    weights = torch.tensor([.2, 1.5], dtype=torch.float64)
    obs = CommonObservations(ref, torch.ones(3, 2, dtype=torch.bool), torch.full((3, 2), 1/3, dtype=torch.float64))
    loss = neighborhood_field_losses(pred, obs, WeightedResponseContrast(rows, weights), [1, 1], [1, 1]).response
    residual = pred - ref[None]
    direct = sum(weights[i] * (torch.einsum('s,vsqc->vqc', rows[i], residual).square().sum() / 6) for i in range(2))
    torch.testing.assert_close(loss, direct)
    torch.testing.assert_close(torch.autograd.grad(loss, pred, retain_graph=True)[0], torch.autograd.grad(direct, pred)[0])
    with pytest.raises(ValueError, match='cancel constants'):
        neighborhood_field_losses(pred, obs, WeightedResponseContrast(torch.ones(1, 4), torch.ones(1)), [1, 1], [1, 1])
    with pytest.raises(ValueError, match='nonnegative'):
        neighborhood_field_losses(pred, obs, WeightedResponseContrast(rows, -weights), [1, 1], [1, 1])


def test_thermal_target_and_control_share_eligibility_and_kernel_trace():
    names = ('flow/u', 'flow/v', 'flow/p', 'flow/omega', 'temperature/fluid')
    provider = SimpleNamespace(model=SimpleNamespace(field_std=torch.ones(5)), native_weights=dict.fromkeys(names, 1.))
    row = [0.2, -0.2, 0., 0.]
    target = {'0001': {'coefficients': row}, '0002': {'coefficients': [0.] * 4}}
    from honf_runtime.local_response_contrasts import build_uniform_contrast
    uniform = build_uniform_contrast(4).operator.tolist()
    for center in target:
        item = {'center_id': center, 'uniform_operator': uniform}
        target_op = ThermalLocalResponseObjective(provider, seed=0, variant='targeted_reynolds_response', target_definitions=target).contrast(item)
        control_op = ThermalLocalResponseObjective(provider, seed=0, variant='trace_matched_uniform_response', target_definitions=target).contrast(item)
        torch.testing.assert_close((target_op.operator.square() * target_op.row_weights[:, None]).sum(),
                                   (control_op.operator.square() * control_op.row_weights[:, None]).sum())


def test_ragged_three_dimensional_nonlinear_control_and_unsupported_edit():
    # Different 3D points per object, plus a non-thermal control; charts are fixed complete inputs.
    scene = {'vertices': [[0., 0., 0.], [1., 0., .2], [.1, 1., .3]], 'control': .2}
    class RaggedAdapter:
        def complete_input(self, value): return copy.deepcopy(value)
        def target(self, value):
            changed = copy.deepcopy(value); changed['control'] += .1
            return value, changed
        def chart(self, anchor, value):
            vertices = np.asarray(value['vertices'])
            padded = np.zeros((4, 3)); padded[:len(vertices)] = vertices
            present = np.arange(4) < len(vertices)
            return np.r_[padded.ravel(), present, value['control']]
        def aligned_observations(self, scenes): return [s['control']**2 for s in scenes]
    adapter = RaggedAdapter(); before, after = adapter.target(scene)
    anchor = adapter.chart(scene, scene); changed = adapter.chart(scene, after)
    fewer = copy.deepcopy(scene); fewer['vertices'] = fewer['vertices'][:2]
    more = copy.deepcopy(scene); more['vertices'].append([.4, .2, .8])
    rows = np.stack([anchor, changed, adapter.chart(scene, fewer), adapter.chart(scene, more)])
    result = solve_bounded_rbf_target(rows, anchor, changed)
    assert result.success and result.rho <= np.sqrt(.1) + 1e-6
    assert abs(result.coefficients.sum()) < 1e-8 and np.abs(result.coefficients).sum() <= 2 + 1e-8
    assert solve_bounded_rbf_target(rows, anchor, anchor).success is False
    tiny = anchor.copy(); tiny[-1] += 1e-5
    assert solve_bounded_rbf_target(rows, anchor, tiny).success
    unresolved = solve_bounded_rbf_target(rows, anchor, tiny, representation_resolution=1e-4)
    assert not unresolved.success and unresolved.rho == 1 and not unresolved.coefficients.any()
    # Two physical maps agree on all donor controls yet disagree on an unseen edit.
    donor_controls = np.array([0., .5, 1.])
    first_map = lambda z: z
    second_map = lambda z: z + np.prod(z[..., None] - donor_controls, axis=-1)
    np.testing.assert_array_equal(first_map(donor_controls), second_map(donor_controls))
    target = np.array([.2, .3])
    assert not np.isclose(np.diff(first_map(target))[0], np.diff(second_map(target))[0])


def test_thermal_reynolds_edit_preserves_inputs_and_updates_viscosity():
    before = config(); before['flow']['nu'] = 2 * .2 / 100
    before['runtime'] = {'nu': 2 * .2 / 100}
    original = copy.deepcopy(before)
    old, new = thermal_reynolds_target(before)
    assert before == original and old == original
    assert new['flow']['re'] == 110 and new['flow']['nu'] == new['runtime']['nu'] == 2 * .2 / 110
    assert new['layout'] == before['layout']
    assert new['physical_reference_status'] == 'unavailable_for_input_description'

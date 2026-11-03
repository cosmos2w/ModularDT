from types import SimpleNamespace

import numpy as np
import pytest

from honf_inverse_core.contracts import NamedContext, PhysicalDesign
from honf_inverse_core.request_schema import GeometryConstraints
from channelthermal.inverse.joint_edit_sampling import (
    Draw, Organization, common_draws, extract_organization, finite_pool_weights,
    make_loadings, nominate, propose, request_energies,
)


def _org():
    return Organization(np.array([10, 20]), np.array([3, 7]), np.ones(2), (), (),
                        np.array([[.1, .2], [.3, .4]]),
                        np.array([[.1, 0], [.3, .4]]),
                        np.array([[0, .5], [.25, 0]]), {})


def test_loadings_have_matched_variance_and_signed_covariance():
    x = np.array([[1., 1.], [2., 1.]])
    e = np.array([[1.5, 1.], [1.5, 1.5]])
    b = make_loadings("G", x, e, [24, 8])
    rng = np.random.default_rng(17)
    eps = rng.normal(size=(120000, 2, 2))
    z = rng.normal(size=(120000, 2))
    v = np.sum(b*b, axis=1)
    samples = np.sqrt(1 - .5*v)[None] * eps + np.einsum('iad,na->nid', b, z) / np.sqrt(2)
    np.testing.assert_allclose(samples.var(axis=0), np.ones((2, 2)), atol=.015)
    assert np.cov(samples[:, 0, 0], samples[:, 1, 0])[0, 1] < -.4
    np.testing.assert_allclose(make_loadings("I", x, e, [24, 8]), 0)
    zero = make_loadings("G", np.zeros((1, 2)), np.zeros((2, 2)), [24, 8])
    np.testing.assert_array_equal(zero, 0)


def test_original_id_alignment_and_stage_evidence():
    class Rows:
        def __init__(self, weights): self.weights = weights
        def row(self, batch, slot):
            return SimpleNamespace(member_index=np.array([0, 1, 2 + slot]),
                original_id=np.array([20, 10, [7, 3][slot]]), is_module=np.array([1, 1, 0]),
                access_weight=np.array([*self.weights[slot], 1.]), applied_exchange=np.array([.2, .3, 0]),
                structural_self=np.array([0, 0, 1]), measure=np.ones(3))
    scene = SimpleNamespace(module_ids=np.array([[20, 10]]), environment_ids=np.array([[7, 3]]),
                            module_mask=np.array([[1, 1]]), environment_mask=np.array([[1, 1]]),
                            environment_measure=np.array([[2., 4.]]))
    stages = {}
    for route in ("mm", "me", "em", "ee"):
        dest = np.array([[20, 10]]) if route[0] == 'm' else np.array([[7, 3]])
        donor = np.array([[20, 10]]) if route[1] == 'm' else np.array([[7, 3]])
        snapshot = SimpleNamespace(destination_ids=dest, donor_ids=donor, destination_mask=np.ones((1, 2)),
                                   donor_mask=np.ones((1, 2)), allowed=np.ones((1,2,2)),
                                   donor_measure=np.ones((1,2)), stage_identity='cycle1/'+route)
        stages[route] = dict(snapshot=snapshot, incidence=np.array([[[1., 2.], [3., 4.]]]))
    org = SimpleNamespace(scene=scene, e_access_directory=SimpleNamespace(full=Rows([[1,2],[3,4]]),
                          restricted=Rows([[.1,.2],[.3,.4]])), final_applied_context_stages=stages)
    result = extract_organization(org, [10, 20], [3, 7])
    np.testing.assert_allclose(result.restricted_module, [[.4,.3],[.2,.1]])
    np.testing.assert_allclose(result.mm, [[4,3],[2,1]])
    np.testing.assert_array_equal(result.environment_measure, [4,2])
    assert result.stages['em']['stage_identity'] == 'cycle1/em'
    with pytest.raises(ValueError): extract_organization(org, [10, 99], [3, 7])


def test_common_draws_and_rejection_keep_identity_and_controls():
    draws = common_draws(5, [10, 20], [10,20], [3,7], 4)
    repeat = common_draws(5, [20,10], [10,20], [3,7], 4)
    for a,b in zip(draws, repeat):
        assert a.pair == b.pair and a.nomination_uniform == b.nomination_uniform
        np.testing.assert_array_equal(a.epsilon, b.epsilon)
        np.testing.assert_array_equal(a.z, b.z)
    design = PhysicalDesign(np.array([[1.,1.],[0.,0.],[2.,1.]]), [1,0,1], [2,0,3])
    context = NamedContext(('module_radius','domain_length_x','domain_length_y'), [0.1,3.,2.], 'test')
    constraints = GeometryConstraints(2, 2, .2, 0, 0, 0)
    bad = Draw((10,20), np.array([[10.,0.],[0.,0.]]), np.zeros(2), .2)
    trial, reason = propose(design, context, constraints, [10,20], np.zeros((2,2,2)), bad)
    assert reason == 'displacement'
    assert trial.module_centers[0,0] > design.module_centers[0,0] + .015
    np.testing.assert_array_equal(trial.module_present, design.module_present)
    np.testing.assert_array_equal(trial.heat_powers, design.heat_powers)
    np.testing.assert_array_equal(trial.module_centers[1:], design.module_centers[1:])
    boundary = PhysicalDesign(np.array([[.101,1.],[0.,0.],[2.,1.]]), [1,0,1], [2,0,3])
    outside = Draw((10,20), np.array([[-1.,0.],[0.,0.]]), np.zeros(2), .2)
    trial, reason = propose(boundary, context, constraints, [10,20], np.zeros((2,2,2)), outside)
    assert reason == 'geometry'
    assert trial.module_centers[0,0] < .1  # rejected, never clipped or repaired


def test_finite_pool_request_and_distinct_nomination():
    energies = request_energies('A', [9.9, 10.], [1., 1.], 10., 1., 0., 1., 1.)
    np.testing.assert_allclose(energies, [0,1], atol=1e-12)
    np.testing.assert_allclose(finite_pool_weights([10000,10001]), [1/(1+np.exp(-1)), 1/(1+np.exp(1))])
    assert nominate(['a','a','b'], [0,0,10], .6, weighted=False) == 2
    assert nominate(['a','a','b'], [0,0,10], .6, weighted=True) == 0
    assert nominate([], [], .5) is None

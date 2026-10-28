import numpy as np
from honf_runtime.anchored_coverage_analysis import coverage_support


def test_joint_prefix_ties_self_and_nesting():
    access = np.array([[0.4, 0.3, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]])
    args = (access, np.array([3, 3]), np.array([0, 0, 1, 1]), np.array([7, 2, 8, 3]))
    low, mass = coverage_support(*args, 0.5)
    high, _ = coverage_support(*args, 0.9)
    assert low.tolist() == [[True, True, False, True], [True, True, False, True]]
    np.testing.assert_allclose(mass, [0.8, 0.75])
    assert np.all(~low | high)


def test_incidence_separates_structural_self_and_unserved_rows():
    from honf_runtime.anchored_coverage_analysis import incidence_summary
    weights = np.array([[1., .1, 0.], [1., 0., .2]])
    self_mask = np.array([[False, True, False], [False, False, True]])
    result = incidence_summary(weights, self_mask, service=[2, 0])
    assert result['with_self']['active_components'] == 1
    assert result['without_self']['isolated_members'] == 2
    assert result['served_with_self']['isolated_receivers'] == 1
    assert result['unused_receivers'] == 1
    assert result['overlap']['support_jaccard'] == 1 / 3
    assert result['service_weighted_overlap']['pairs'] == 0

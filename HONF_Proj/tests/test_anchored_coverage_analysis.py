import numpy as np
from honf_runtime.anchored_coverage_analysis import coverage_support


def test_regional_mass_geometry_separates_tails_and_empty_rows():
    from honf_runtime.anchored_coverage_analysis import regional_mass_geometry
    coords = np.array([[0., 0.], [1., 0.], [3., 0.]])
    weights = np.array([[2., .99, .01], [0., 1., 0.], [.99, .01, 3.]])
    result = regional_mass_geometry(weights, coords, coords, [1., 2., 3.], [0, 1, 2])
    assert result['positive_nonself_count'] == [2, 0, 2]
    assert result['zero_nonself_rows'] == 1
    assert result['prefixes']['0.9']['count'] == [1, 0, 1]
    assert result['prefixes']['0.9']['area'] == [2., None, 1.]
    assert result['prefixes']['0.9']['max_reach'] == [1., None, 3.]
    assert result['prefixes']['0.99']['count'] == [1, 0, 1]
    assert result['nearest_weighted_jaccard'][1] == 0.


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


def test_structural_removal_preserves_learned_self_and_original_normalization():
    from honf_runtime.anchored_coverage_analysis import learned_access_weights
    # Each row has one M and one ownE. Only the first owns learned EE self.
    relations = np.array([[2., 3., 0.], [2., 0., 0.]])
    structural = np.array([[0., 1., 0.], [0., 0., 1.]])
    measure = np.array([.5, .25, .25])
    unnormalized = (relations + structural) * measure
    access = unnormalized / unnormalized.sum(axis=1, keepdims=True)
    learned = learned_access_weights(access, relations, structural)
    expected = relations * measure / unnormalized.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(learned, expected)
    assert learned[0, 1] > 0  # A structural mask must not erase learned self.
    assert learned[1, 2] == 0  # This ownE was structural-only.
    assert np.all(learned.sum(axis=1) < 1)  # No counterfactual renormalization.


def test_structural_removal_on_compiled_support_changes_graph_only_where_sole_structural():
    from honf_runtime.anchored_coverage_analysis import incidence_summary, learned_access_weights
    access = np.array([[.4, .6, 0.], [.4, 0., .6]])
    # Positive learned entries outside C must stay excluded from this accounting.
    relations = np.array([[2., 1., 7.], [2., 4., 0.]])
    structural = np.array([[False, True, False], [False, False, True]])
    learned = learned_access_weights(access, relations, structural)
    assert learned[0, 2] == learned[1, 1] == 0
    result = incidence_summary(learned, np.zeros_like(structural), service=[1, 0])
    assert result['with_self']['isolated_members'] == 1
    assert result['served_with_self']['isolated_receivers'] == 1
    assert result['cardinality'] == [2, 1]
    # The unchanged legacy statistic removes learned ownE as well.
    whole_own = incidence_summary(access, structural)
    assert whole_own['without_self']['isolated_members'] == 2

import numpy as np
import pytest

from honf_viz.organization_connectivity import (
    membership_overlap, select_connections, select_native_rows,
)
from honf_viz.organization_hyperedges import _ellipse_layout


def test_identical_rows_choose_one_and_input_is_unchanged():
    a = np.array([[1., 2.], [1., 2.], [1., 2.]])
    before = a.copy()
    selected, info = select_native_rows([a], ["a", "b", "c"])
    assert selected == [0]
    assert info["coverage_met"] and not info["budget_limited"]
    assert info["covered_rows"] == 3 and info["covered_fraction"] == 1
    assert info["selected_labels"] == ["a"]
    assert np.array_equal(a, before)


def test_diversity_and_cap_are_disclosed_and_manual_count_is_exact():
    a = np.eye(4)
    labels = list("abcd")
    selected, info = select_native_rows([a], labels, resolution=.1, max_edges=2)
    assert selected == [0, 1]
    assert not info["coverage_met"] and info["budget_limited"]
    assert info["covered_rows"] == 2 and info["covered_fraction"] == .5
    assert len(info["residual_distances"]) == 4
    selected, info = select_native_rows([a], labels, edges=4, max_edges=1)
    assert selected == [0, 1, 2, 3] and info["coverage_met"]
    assert not info["budget_limited"]


def test_compact_automatic_selection_adapts_to_one_two_or_three_rows():
    profiles = [
        (np.ones((4, 2)), 1),
        (np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]]), 2),
        (np.eye(3), 3),
    ]
    for matrix, expected in profiles:
        selected, info = select_native_rows([matrix], [str(i) for i in range(len(matrix))], resolution=.1)
        assert len(selected) == expected
        assert info["max_edges"] == 4 and info["coverage_met"]
        assert info["covered_fraction"] == 1


def test_four_ellipses_preserve_all_nonempty_intersections_and_duplicates():
    masks = np.repeat(np.arange(1, 16, dtype=int), 2)
    masks = np.concatenate(([0], masks))
    before = masks.copy()
    coordinates, evidence = _ellipse_layout(masks, 4)
    centers = np.asarray(evidence["centers"])
    axes = np.asarray(evidence["semi_axes"])
    angles = np.deg2rad(evidence["angles_degrees"])
    delta = coordinates[:, None, :] - centers
    u = delta[..., 0] * np.cos(angles) + delta[..., 1] * np.sin(angles)
    v = -delta[..., 0] * np.sin(angles) + delta[..., 1] * np.cos(angles)
    realized = (((u / axes[:, 0]) ** 2 + (v / axes[:, 1]) ** 2 < 1)
                * (2 ** np.arange(4))).sum(axis=1)
    assert np.array_equal(realized, masks)
    assert np.array_equal(masks, before)
    assert len(np.unique(coordinates, axis=0)) == len(masks)
    assert evidence["exact_membership_verified"]
    assert all(axes[:, 0] != axes[:, 1])


def test_four_ellipse_empty_and_invalid_inputs():
    coordinates, evidence = _ellipse_layout(np.array([], dtype=int), 4)
    assert coordinates.shape == (0, 2) and evidence["exact_membership_verified"]
    for masks, count in [(np.array([16]), 4), (np.array([-1]), 4), (np.array([1]), 5)]:
        with pytest.raises(ValueError):
            _ellipse_layout(masks, count)


def test_four_ellipse_dense_intersections_are_spaced_across_masks():
    # Native-row example with several M members in overlapping 14 and 6,
    # and additional E members in regions 1 and 9. All remain exact vertices.
    masks = np.array([14, 15, 14, 6, 6, 14, 14, 2, 1, 9, 1, 1, 9, 9, 9, 1, 4, 1, 8])
    xy, evidence = _ellipse_layout(masks, 4)
    separation = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    separation += np.eye(len(xy)) * 100
    assert separation.min() > 1.0
    assert evidence["geometry_choice"] == "expanded-selected-intersections"
    assert evidence["exact_membership_verified"]


def test_compact_connection_defaults_limit_one_incoming_and_eight_total():
    weights = np.full((5, 5), .5)
    result = select_connections(weights, [str(i) for i in range(5)])
    assert result["max_incoming"] == 1 and result["max_connections"] == 8
    assert len(result["connections"]) == 5
    assert len({edge["receiver"] for edge in result["connections"]}) == 5


def test_matched_models_equal_weight_and_remove_own_only_in_selection():
    a = np.array([[10., 1., 0.], [0., 10., 1.], [1., 0., 10.]])
    b = a * 5
    before = a.copy()
    rows, evidence = select_native_rows([a, b], ["e0", "e1", "e2"],
                                        own_columns=np.arange(3), resolution=.1)
    assert rows == [0, 1, 2]
    assert evidence["selection_only_own_columns_removed"]
    assert evidence["model_count"] == 2
    assert np.array_equal(a, before)
    assert evidence["assignments"] == [0, 1, 2]


@pytest.mark.parametrize("matrices,labels,kwargs", [
    ([np.array([[np.nan]])], ["a"], {}),
    ([np.array([[-1.]])], ["a"], {}),
    ([np.ones((2, 2)), np.ones((2, 3))], ["a", "b"], {}),
    ([np.eye(2)], ["a", "a"], {}),
    ([np.eye(2)], ["a", "b"], {"own_columns": np.array([0, 3])}),
])
def test_invalid_native_inputs_rejected(matrices, labels, kwargs):
    with pytest.raises(ValueError):
        select_native_rows(matrices, labels, **kwargs)


def test_actual_directed_gates_bins_and_omitted_accounting():
    # Matrix entry [receiver, donor]; no transpose or self gate is fabricated.
    gate = np.array([[1., .2, .8], [.5, 1., .2], [0., .7, 1.]])
    before = gate.copy()
    result = select_connections(gate, ["M0", "M1", "M2"], max_incoming=1, max_connections=2)
    assert [(e["donor"], e["receiver"]) for e in result["connections"]] == [("M2", "M0"), ("M1", "M2")]
    assert [e["bin"] for e in result["connections"]] == ["strong", "strong"]
    assert result["positive_count"] == 5 and result["omitted_count"] == 3
    assert result["selected_weight"] == pytest.approx(1.5)
    assert result["omitted_weight"] == pytest.approx(.9)
    assert np.array_equal(gate, before)


def test_exact_bin_boundaries_and_ties_use_native_order():
    gate = np.array([[0., 1 / 3, 2 / 3], [.2, 0., .2], [0., 0., 0.]])
    result = select_connections(gate, ["M0", "M1", "M2"], max_incoming=2)
    assert [(e["bin"], e["width"]) for e in result["connections"]] == [
        ("strong", 4.4), ("medium", 2.6), ("weak", 1.2), ("weak", 1.2)]
    assert [e["donor"] for e in result["connections"][-2:]] == ["M0", "M2"]


def test_association_is_explicit_symmetric_and_zero_rows_have_no_edges():
    overlap = membership_overlap(np.array([[1., 1.], [1., 0.], [0., 0.]]))
    assert overlap[0, 1] == pytest.approx(.5) and overlap[0, 2] == 0
    result = select_connections(overlap, ["M0", "M1", "M2"], kind="association")
    assert not result["directed"] and result["positive_count"] == 1
    assert len(result["connections"]) == 1
    assert "not an applied gate" in result["meaning"]
    empty = select_connections(np.zeros((2, 2)), ["M0", "M1"])
    assert empty["connections"] == [] and empty["selected_mass_fraction"] is None
    with pytest.raises(ValueError):
        select_connections(np.array([[0., .2], [.3, 0.]]), ["M0", "M1"], kind="association")

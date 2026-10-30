"""Exact native memberships and measured connections survive compact drawing."""
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pytest

from honf_viz import organization_connected_hyperedges as drawing
from honf_viz.organization_data import OrganizationScene
from honf_viz.organization_hyperedges import _ellipse_layout


@pytest.fixture
def capture(monkeypatch):
    captures = []
    def save(fig, output, stem, title, caption, evidence):
        fig.canvas.draw()
        arrows = [p for ax in fig.axes for p in ax.patches if isinstance(p, FancyArrowPatch)]
        captures.append({"evidence": evidence, "arrow_widths": [a.get_linewidth() for a in arrows],
                         "arrow_curvatures": [a.get_connectionstyle().rad for a in arrows]})
        plt.close(fig)
        return drawing.FigureArtifact(stem, title, caption, evidence)
    monkeypatch.setattr(drawing, "_save", save)
    return captures


def view(count=4):
    weights = np.column_stack([np.full((count, 2), .02), np.eye(count)])
    return drawing.NativeHyperedgeView(
        "Measured model", weights, tuple(f"h{i}" for i in range(count)),
        ("M17", "M4") + tuple(f"E{i}" for i in range(count)), 2,
        module_connections=np.array([[0., .2], [.8, 0.]]),
        connection_provenance="cycle1/mm", metadata={"case": "synthetic", "epoch": 15},
    )


def test_compact_four_set_drawing_keeps_native_weights_and_directed_gate_values(capture):
    native = view()
    before = native.weights.copy()
    artifacts = drawing.render_native_connectivity([native], Path("memory"))
    assert len(artifacts) == 1
    data = artifacts[0].evidence
    assert len(data["global_selected_rows"]) == 4 and data["pages"] == 1
    assert len(data["ellipse_layout"]["centers"]) == 4
    assert data["ellipse_layout"]["exact_membership_verified"]
    assert {v["member"] for v in data["vertices"]} == set(native.member_labels)
    for vertex in data["vertices"]:
        column = vertex["native_column"]
        np.testing.assert_array_equal(vertex["weights_by_page_row"], native.weights[data["page_rows"], column])
    edges = data["displayed_connections"]
    assert [(e["donor"], e["receiver"], e["weight"]) for e in edges] == [
        ("M17", "M4", .8), ("M4", "M17", .2)]
    assert capture[0]["arrow_widths"] == [4.4, 1.2]
    assert capture[0]["arrow_curvatures"] == [.65, .65]
    assert data["connections"]["stage"] == "cycle1/mm"
    np.testing.assert_array_equal(native.weights, before)


def test_auto_default_is_one_compact_page_even_for_rich_native_inventory(capture):
    native = view(7)
    artifacts = drawing.render_native_connectivity([native], Path("memory"))
    assert len(artifacts) == 1
    data = artifacts[0].evidence
    assert data["selection"]["native_row_count"] == 7
    assert len(data["global_selected_rows"]) == 4
    assert data["selection"]["covered_fraction"] == pytest.approx(4 / 7)
    assert not data["selection"]["coverage_met"]


def test_matched_selection_never_pools_model_weights_and_explicit_more_rows_are_opt_in(capture):
    a = view(6)
    b = replace(a, name="Other model", weights=a.weights * .7,
                module_connections=np.array([[0., .4], [0., 0.]]))
    artifacts = drawing.render_native_connectivity([a, b], Path("memory"), edges=6)
    assert len(artifacts) == 4  # Two pages each; only explicitly requested.
    for artifact in artifacts:
        data = artifact.evidence
        assert len(data["global_selected_rows"]) == 6
        assert len(data["page_rows"]) == 3
        owner = a if artifact.stem.startswith("07_connected_Measured") else b
        np.testing.assert_array_equal(data["all_selected_native_weights"], owner.weights[data["global_selected_rows"]])
    assert artifacts[0].evidence["global_selected_rows"] == artifacts[2].evidence["global_selected_rows"]


def test_threshold_preserves_positive_modules_without_inventing_zero_own_entries(capture):
    native = drawing.NativeHyperedgeView(
        "Zero own", np.array([[.1, 0., .001], [.1, .4, 0.]]),
        ("E0", "E1"), ("M2", "E0", "E1"), 1, own_columns=np.array([1, 2]),
    )
    result = drawing.render_native_connectivity([native], Path("memory"), edges=2, e_threshold=.9)[0]
    assert [v["member"] for v in result.evidence["vertices"]] == ["M2"]
    assert result.evidence["displayed_connections"] == []
    assert not result.evidence["connections"]["available"]
    for row in result.evidence["edge_details"]:
        assert row["shown_M"] == 1 and row["shown_E"] == 0
        assert row["shown_mass"] + sum(x["weight"] for x in row["omitted_members"]) == pytest.approx(row["total_mass"])


def test_wide_presentation_preserves_ellipse_quadratic_distances():
    points, old = _ellipse_layout(np.arange(16), 4)
    mapped, new = drawing._wide_geometry(points, old)
    def quadratic(xy, geometry):
        result = []
        for center, semi, degrees in zip(geometry["centers"], geometry["semi_axes"], geometry["angles_degrees"]):
            angle = np.deg2rad(degrees)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            normalized = ((xy - center) @ rotation) / semi
            result.append(np.sum(normalized ** 2, axis=1))
        return np.array(result)
    np.testing.assert_allclose(quadratic(points, old), quadratic(mapped, new), rtol=1e-12, atol=1e-12)
    assert "presentation_affine_map" in new
    assert not np.array_equal(points, mapped)


@pytest.mark.parametrize("count", [2, 4])
def test_layout_locator_preserves_original_geometry_and_anchor_identity(capture, count):
    native = view(count)
    modules = np.array([[2., 3.], [9., 4.]])
    environment = np.column_stack([np.arange(count) * 2. + 1., np.ones(count)])
    if count == 4:
        environment[-1] = [9.25, 4.875]  # Preferred lower label would cover M4.
    scene = OrganizationScene(
        native.name, Path("saved-scene.npz"), {"coordinate_units": "m"},
        native.member_labels[:2], native.member_labels[2:], modules, environment,
        native.weights, {}, {}, np.array([.4, .6]), None, (0., 12., 0., 6.), "C_restricted")
    native = replace(native, own_columns=2 + np.arange(count), layout=scene)
    result = drawing.render_native_connectivity([native], Path("memory"), edges=count)[0].evidence
    locator = result["layout_locator"]
    assert locator["available"] and locator["coordinate_units"] == "m"
    assert locator["module_ids"] == ["M17", "M4"]
    np.testing.assert_array_equal(locator["module_xy"], modules)
    assert locator["module_radius"] == [.4, .6] and locator["bounds"] == [0., 12., 0., 6.]
    for anchor in locator["selected_anchors"]:
        row = anchor["native_row"]
        assert anchor["original_member"] == f"E{row}"
        np.testing.assert_array_equal(anchor["physical_xy"], environment[row])
        if count == 4 and row == count - 1:
            assert anchor["label_offset_points"][1] > 0
            assert anchor["label_placement_score_px2"] == 0
    assert result["physical_geometry_encoded"]
    assert not result["abstract_panel_encodes_physical_geometry"]
    np.testing.assert_array_equal(native.weights, scene.C)

"""Scientific integrity checks for saved-array global organization figures."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("scipy")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))

from honf_viz.organization_data import (
    FigureArtifact, OrganizationScene, load_scene, plot_cells, require_matched,
)
from honf_viz import organization_hyperedges as hyperedges
from honf_viz import organization_overview as overview
from honf_viz import organization_services as services


@pytest.fixture
def scene():
    xy = np.array([[0., 0.], [1., 0.], [.2, 1.], [.9, .8]])
    C = np.array([[.005, .45, 0., .02, .20, .20],
                  [.45, .005, .20, .25, .02, .10],
                  [.35, .20, .02, .05, .30, .10],
                  [.05, .40, .30, .04, .02, .19]])
    gates = {"mm": np.array([[0., .3], [.5, 0.]]),
             "em": np.array([[.1, .3], [.2, 0.], [.5, .1], [0., .4]]),
             "ee": np.array([[0., .2, .3, 0.], [.1, 0., .4, .1],
                              [.3, .1, 0., .2], [0., .3, .2, 0.]]),
             "me": np.array([[0., .4, .2, .1], [.1, 0., .8, .2]])}
    return OrganizationScene(
        "Model A", Path("saved-input.npz"), {"case": "irregular-case", "epoch": 23},
        ("M30", "M7"), ("E91", "E4", "E70", "E2"),
        np.array([[.4, .3], [.8, .2]]), xy, C, gates,
        {route: "cycle2/" + route for route in gates}, None, None,
        (-.2, 1.2, -.2, 1.2), "C_restricted",
    )


@pytest.fixture
def memory_figures(monkeypatch):
    """Exercise actual plotting/Agg rendering without writing generated assets."""
    captured = []

    def save(fig, output, stem, title, caption, evidence):
        fig.canvas.draw()
        captured.append((stem, evidence))
        plt.close(fig)
        return FigureArtifact(stem, title, caption, evidence)

    for module in (hyperedges, services):
        monkeypatch.setattr(module, "save_figure", save)
    # Overview imports this shared function inside its render entry points.
    import honf_viz.organization_data as data
    monkeypatch.setattr(data, "save_figure", save)
    return captured


def test_saved_array_loader_packs_padding_and_preserves_native_ids_without_queries(tmp_path):
    source = tmp_path / "prepared.npz"
    original = np.arange(18, dtype=float).reshape(3, 6) / 100
    arrays = {
        "module_xy": np.array([[.2, .3], [9., 9.], [.7, .5]]),
        "module_mask": np.array([True, False, True]),
        "module_ids": np.array([30, -1, 7]),
        "environment_xy": np.array([[0., 0.], [1., 0.], [.2, 1.]]),
        "environment_ids": np.array([91, 4, 70]),
        "C_restricted": original,
        "context_cycle2_mm": np.arange(9, dtype=float).reshape(3, 3) / 10,
    }
    np.savez(source, **arrays)
    source.with_suffix(".json").write_text(json.dumps({"case": "synthetic", "epoch": 23}))
    loaded = load_scene("Example", source)
    assert loaded.m_ids == ("M30", "M7")
    assert loaded.e_ids == ("E91", "E4", "E70")
    np.testing.assert_array_equal(loaded.C, original[:, [0, 2, 3, 4, 5]])
    np.testing.assert_array_equal(loaded.own_columns, [2, 3, 4])
    np.testing.assert_array_equal(loaded.gates["mm"], arrays["context_cycle2_mm"][[0, 2]][:, [0, 2]])
    shared = loaded.shared
    assert np.all(shared[np.arange(3), loaded.own_columns] == 0)
    with np.load(source) as reopened:
        np.testing.assert_array_equal(reopened["C_restricted"], original)
        assert "query_xy" not in reopened and "beta_restricted" not in reopened
    assert loaded.provenance["mm"] == "cycle2/mm"
    # Editing a display copy cannot rewrite the prepared directory.
    shared[:] = -1
    assert (loaded.C >= 0).all()


def test_matched_comparison_rejects_changed_native_geometry(scene):
    changed = scene.environment_xy.copy()
    changed[0, 0] += .01
    with pytest.raises(ValueError, match="same original layout"):
        require_matched([scene, replace(scene, name="Other", environment_xy=changed)])
    with pytest.raises(ValueError, match="aligned original M/E IDs"):
        require_matched([scene, replace(scene, m_ids=("M7", "M30"))])


def test_shared_similarity_permutation_preserves_exact_values_support_and_role_bands():
    a = np.array([[.4, 0., .1, 0.], [0., .5, 0., .2],
                  [0., 0., 0., 0.], [.1, 0., .3, 0.], [0., .2, 0., .4]])
    b = np.array([[.3, 0., .2, 0.], [0., .2, 0., .4],
                  [0., 0., 0., 0.], [.5, 0., .1, 0.], [0., .1, 0., .6]])
    rows, columns = overview._shared_permutations([a, b], 2)
    assert set(rows[:2]) == {0, 1}
    assert set(rows[2:]) == {2, 3, 4}
    assert sorted(columns.tolist()) == list(range(4))
    for original in (a, b):
        shown = original[rows][:, columns]
        np.testing.assert_array_equal(np.sort(shown.ravel()), np.sort(original.ravel()))
        restored = shown[np.argsort(rows)][:, np.argsort(columns)]
        np.testing.assert_array_equal(restored, original)
        np.testing.assert_array_equal(restored > 0, original > 0)


def test_incidence_render_retains_all_addresses_and_records_same_order_for_models(scene, memory_figures):
    other = replace(scene, name="Model B", C=scene.C * .8)
    original = scene.C.copy()
    result = overview.render_incidence([scene, other], Path("not-written"))
    assert set(result.evidence["address_ids_sorted"]) == set(scene.e_ids)
    assert len(result.evidence["address_permutation"]) == scene.n_e
    for model in (scene.name, other.name):
        assert result.evidence["models"][model]["reordering_preserves_exact_values"]
    np.testing.assert_array_equal(scene.C, original)
    assert memory_figures


def test_abstract_ellipses_encode_every_requested_membership_exactly():
    masks = np.repeat(np.arange(1, 8), 3)
    points, geometry = hyperedges._ellipse_layout(masks, 3)
    actual = np.zeros(len(points), dtype=int)
    for bit, (center, axes, angle) in enumerate(zip(
            geometry["centers"], geometry["semi_axes"], geometry["angles_degrees"])):
        dx, dy = (points - np.asarray(center)).T
        radians = np.deg2rad(angle)
        u = dx * np.cos(radians) + dy * np.sin(radians)
        v = -dx * np.sin(radians) + dy * np.cos(radians)
        actual |= (((u / axes[0]) ** 2 + (v / axes[1]) ** 2) < 1).astype(int) << bit
    np.testing.assert_array_equal(actual, masks)
    assert len(np.unique(points, axis=0)) == len(points)


def test_hyperedge_threshold_accounts_for_omitted_mass_and_keeps_all_positive_M(scene, memory_figures):
    original = scene.C.copy()
    result = hyperedges.render_native_hyperedges([scene], Path("not-written"), e_threshold=.15)
    model = result.evidence["scenes"][scene.name]
    for edge in model["edge_details"]:
        row = edge["anchor_row"]
        weights = original[row]
        keep = weights > 0
        keep[scene.n_m:] &= weights[scene.n_m:] >= .15
        own = scene.own_columns[row]
        keep[own] = weights[own] > 0
        assert edge["shown_M"] == int(np.count_nonzero(weights[:scene.n_m]))
        assert edge["shown_directory_mass"] == pytest.approx(float(weights[keep].sum()))
        assert edge["retained_directory_mass_fraction"] == pytest.approx(weights[keep].sum() / weights.sum())
        omitted = {item["member"]: item["weight"] for item in edge["omitted_positive_members"]}
        assert omitted == {scene.member_labels[i]: weights[i]
                           for i in np.flatnonzero((weights > 0) & ~keep)}
        assert edge["shown_directory_mass"] + sum(omitted.values()) == pytest.approx(weights.sum())
    np.testing.assert_array_equal(scene.C, original)


def test_service_examples_are_pooled_globally_and_matched_across_models(scene):
    other_C = np.zeros_like(scene.C)
    other_C[:, 0] = 4.
    other_C[:, scene.n_m + 2] = 3.
    other = replace(scene, name="Model B", C=other_C)
    selection = services.select_service_members([scene, other], per_role=1)
    assert [entry["id"] for entry in selection] == ["M30", "E70"]
    for entry in selection:
        column = entry["column_index"]
        expected = scene.shared[:, column].sum() + other.shared[:, column].sum()
        assert entry["pooled_nonself_directory_mass"] == pytest.approx(expected)
        assert [record["model"] for record in entry["per_model"]] == [scene.name, other.name]


def test_actual_four_route_views_keep_tiny_positive_M_edges_and_require_real_gates(scene, memory_figures):
    gates = {route: values.copy() for route, values in scene.gates.items()}
    gates["em"][0, 0] = 1e-12
    changed = replace(scene, gates=gates)
    result = services.render_incoming_neighborhoods([changed], Path("not-written"), per_role=4)
    panel = next(p for p in result.evidence["panels"] if p["receiver"] == scene.e_ids[0])
    assert scene.m_ids[0] in panel["positive_M_ids"]
    assert panel["M_gate_values"][0] == 1e-12
    assert panel["M_route"] == "em" and panel["E_route"] == "ee"
    assert panel["M_stage"] == "cycle2/em"
    assert panel["E_gate_values"][0] == 0  # No structural self was inserted.
    with pytest.raises(ValueError, match="saved actual MM, EM, EE and ME"):
        services.render_incoming_neighborhoods(
            [replace(scene, gates={k: v for k, v in gates.items() if k != "mm"})], Path("not-written"))


def test_adapter_polygon_cell_geometry_is_used_without_grid_assumptions(scene):
    polygons = np.array([[[x - .05, y - .05], [x + .08, y - .05], [x, y + .10]]
                         for x, y in scene.environment_xy])
    changed = replace(scene, polygons=polygons)
    fig, ax = plt.subplots()
    collection = plot_cells(ax, changed, np.array([.1, .2, .3, .4]), cmap="viridis")
    assert len(collection.get_paths()) == scene.n_e
    for path, expected in zip(collection.get_paths(), polygons):
        np.testing.assert_allclose(path.vertices[:3], expected)
    np.testing.assert_array_equal(collection.get_array(), [.1, .2, .3, .4])
    plt.close(fig)


def test_irregular_point_geometry_keeps_original_coordinate_order(scene):
    fig, ax = plt.subplots()
    values = np.array([.4, .1, .3, .2])
    collection = plot_cells(ax, scene, values, cmap="viridis")
    np.testing.assert_array_equal(collection.get_offsets(), scene.environment_xy)
    np.testing.assert_array_equal(collection.get_array(), values)
    plt.close(fig)


@pytest.mark.parametrize("own", [0., .4])
def test_no_M_own_only_or_empty_directory_never_invents_members(scene, memory_figures, own):
    changed = replace(
        scene, m_ids=(), module_xy=np.empty((0, 2)), module_radius=None,
        e_ids=("Efar",), environment_xy=np.array([[2., 3.]]), C=np.array([[own]]),
        gates={"mm": np.empty((0, 0)), "me": np.empty((0, 1)),
               "em": np.empty((1, 0)), "ee": np.zeros((1, 1))},
    )
    original = changed.C.copy()
    result = hyperedges.render_native_hyperedges(
        [changed], Path("not-written"), e_threshold=.9, edges=1)
    vertices = result.evidence["scenes"][changed.name]["vertices"]
    assert [v["original_member"] for v in vertices] == (["Efar"] if own > 0 else [])
    assert len(services.render_donor_services([changed], Path("not-written"))) == 1
    incoming = services.render_incoming_neighborhoods([changed], Path("not-written"))
    assert incoming.evidence["panels"][0]["positive_M_ids"] == []
    assert incoming.evidence["panels"][0]["positive_E_ids"] == []
    incidence = overview.render_incidence([changed], Path("not-written"))
    assert incidence.evidence["models"][changed.name]["nonzero_shared_entries"] == 0
    np.testing.assert_array_equal(changed.C, original)


@pytest.mark.parametrize("count", [1, 2])
def test_overview_title_does_not_overlap_panel_headings(scene, monkeypatch, count):
    import honf_viz.organization_data as data

    def inspect(fig, output, stem, title, caption, evidence):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        main = fig._suptitle.get_window_extent(renderer)
        panel_titles = [ax._left_title.get_window_extent(renderer) for ax in fig.axes if ax.get_title(loc="left")]
        assert panel_titles and all(main.y0 > panel.y1 for panel in panel_titles)
        plt.close(fig)
        return FigureArtifact(stem, title, caption, evidence)

    monkeypatch.setattr(data, "save_figure", inspect)
    scenes = [replace(scene, name="Relation-anchored")]
    if count == 2:
        scenes.append(replace(scene, name="Active-domain relation-anchored"))
    overview.render_overview(scenes, Path("not-written"))

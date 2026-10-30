"""Adaptive native hyperedge examples with separately measured connectivity.

All inputs are saved arrays. A compact page contains up to four exact sets; the
number of selected native rows is determined before pagination.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np

from .organization_connectivity import select_connections, select_native_rows
from .organization_data import FigureArtifact, OrganizationScene, require_matched
from .organization_hyperedges import _ellipse_layout


@dataclass
class NativeHyperedgeView:
    """Native row-by-original-member weights, with optional actual MM gates.

    Columns contain active M followed by original E. Rows can be E addresses,
    original M/E seeds, or native learned slots; adapters retain that meaning.
    """

    name: str
    weights: np.ndarray
    row_labels: tuple[str, ...]
    member_labels: tuple[str, ...]
    n_m: int
    own_columns: np.ndarray | None = None
    module_connections: np.ndarray | None = None
    connection_kind: str = "actual_mm"
    connection_provenance: str | None = None
    metadata: dict = field(default_factory=dict)


def view_from_scene(scene: OrganizationScene) -> NativeHyperedgeView:
    return NativeHyperedgeView(
        scene.name, scene.C, scene.e_ids, scene.member_labels, scene.n_m,
        own_columns=scene.own_columns, module_connections=scene.gates.get("mm"),
        connection_provenance=scene.provenance.get("mm"),
        metadata={**scene.identity(), "row_semantics": "native E-address restricted directory"},
    )


def _validate_views(views: list[NativeHyperedgeView]) -> None:
    if not views or len({v.name for v in views}) != len(views):
        raise ValueError("Supply views with distinct model labels.")
    ref = views[0]
    for view in views:
        w = np.asarray(view.weights)
        if (w.ndim != 2 or w.shape != (len(view.row_labels), len(view.member_labels))
                or not np.isfinite(w).all() or (w < 0).any()
                or len(set(view.member_labels)) != len(view.member_labels)
                or not 0 <= view.n_m <= len(view.member_labels)):
            raise ValueError("Native rows and original-member inventories must align.")
        if (view.row_labels != ref.row_labels or view.member_labels != ref.member_labels
                or view.n_m != ref.n_m):
            raise ValueError("Matched views must align native rows and original members.")
        if (view.own_columns is None) != (ref.own_columns is None) or (
                view.own_columns is not None
                and not np.array_equal(view.own_columns, ref.own_columns)):
            raise ValueError("Matched views must use the same actual own-member inventory.")
        if view.connection_kind not in ("actual_mm", "association"):
            raise ValueError("Connections must be actual_mm or explicitly derived association.")
        if view.module_connections is not None and np.shape(view.module_connections) != (view.n_m, view.n_m):
            raise ValueError("Module connections must follow original active M IDs.")


def _page_plan(view, rows, threshold, capacity):
    weights = np.asarray(view.weights, dtype=float)[rows]
    keep = weights > 0
    keep[:, view.n_m:] &= weights[:, view.n_m:] >= threshold
    if view.own_columns is not None:
        for edge, row in enumerate(rows):
            own = int(view.own_columns[row])
            keep[edge, own] = weights[edge, own] > 0
    bits = (keep.astype(int) * (2 ** np.arange(len(rows)))[:, None]).sum(axis=0)
    members = np.flatnonzero(bits)
    if len(members) > capacity:
        raise ValueError(
            f"{view.name}: {len(members)} displayed original vertices exceed {capacity}; "
            "use fewer rows per page or explicitly change the E display threshold. "
            "Native row selection is unchanged.")
    xy, geometry = _ellipse_layout(bits[members], len(rows))
    details = []
    for edge, row in enumerate(rows):
        total = float(weights[edge].sum())
        shown = float(weights[edge, keep[edge]].sum())
        omitted = np.flatnonzero((weights[edge] > 0) & ~keep[edge])
        details.append({
            "native_row": int(row), "native_label": view.row_labels[row],
            "shown_M": int(keep[edge, :view.n_m].sum()),
            "shown_E": int(keep[edge, view.n_m:].sum()),
            "total_mass": total, "shown_mass": shown,
            "retained_mass": shown / total if total else None,
            "omitted_members": [{"member": view.member_labels[i],
                                 "weight": float(weights[edge, i])} for i in omitted],
        })
    return weights, bits, members, xy, geometry, details


def _save(fig, output, stem, title, caption, evidence):
    import matplotlib.pyplot as plt
    folder = Path(output) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    # Fixed page geometry makes panels suitable for presentation extraction.
    fig.savefig(folder / f"{stem}.pdf")
    fig.savefig(folder / f"{stem}.png", dpi=150)
    plt.close(fig)
    return FigureArtifact(stem, title, caption, evidence)


def _wide_geometry(positions: np.ndarray, geometry: dict) -> tuple[np.ndarray, dict]:
    """Spread an abstract set layout across a wide page, preserving every set.

    A common invertible affine map transforms vertices and ellipse matrices.
    SVD recovers genuine ellipse parameters; no membership or weight changes.
    """
    centers = np.asarray(geometry["centers"], dtype=float)
    axes = np.asarray(geometry["semi_axes"], dtype=float)
    angles = np.deg2rad(geometry["angles_degrees"])
    matrices = []
    for semi, angle in zip(axes, angles):
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        matrices.append(rotation @ np.diag(semi))
    extents = np.sqrt(np.sum(np.square(matrices), axis=2))
    low, high = (centers - extents).min(0), (centers + extents).max(0)
    span = high - low
    transform = np.diag([2.5 * span[1] / span[0], 1.])
    transformed_centers = centers @ transform.T
    new_axes, new_angles, new_extents = [], [], []
    for matrix in matrices:
        mapped = transform @ matrix
        rotation, semi, _ = np.linalg.svd(mapped)
        new_axes.append(semi.tolist())
        new_angles.append(float(np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0]))))
        new_extents.append(np.sqrt(np.sum(mapped ** 2, axis=1)))
    new_extents = np.asarray(new_extents)
    lo, hi = (transformed_centers - new_extents).min(0), (transformed_centers + new_extents).max(0)
    pad = (hi - lo) * np.array([.035, .07])
    return positions @ transform.T, {
        **geometry, "centers": transformed_centers.tolist(), "semi_axes": new_axes,
        "angles_degrees": new_angles,
        "bounds": [float(lo[0]-pad[0]), float(hi[0]+pad[0]), float(lo[1]-pad[1]), float(hi[1]+pad[1])],
        "presentation_affine_map": transform.tolist(),
    }


def _verify_intersections(positions, geometry, expected):
    actual = np.zeros(len(positions), dtype=int)
    for edge, (center, semi, degrees) in enumerate(zip(
            geometry["centers"], geometry["semi_axes"], geometry["angles_degrees"])):
        angle = np.deg2rad(degrees)
        dx, dy = (positions - center).T
        u, v = dx*np.cos(angle)+dy*np.sin(angle), -dx*np.sin(angle)+dy*np.cos(angle)
        actual |= (((u/semi[0])**2+(v/semi[1])**2) < 1).astype(int) << edge
    if not np.array_equal(actual, expected):
        raise ValueError("Presentation geometry changed an original-member intersection.")


def render_native_connectivity(
    views: list[NativeHyperedgeView], output: Path, *, edges: str | int = "auto",
    resolution: float = .5, coverage: float = .9, max_edges: int = 4,
    e_threshold: float = .02, max_vertices: int = 48, per_page: int = 4,
    max_incoming: int = 1, max_connections: int = 8,
) -> list[FigureArtifact]:
    """Render original-member sets with actual directed or derived connections.

    Matched views select common actual rows. Ellipses encode exact displayed
    membership within each page; complete selected-row incidence is returned
    so cross-page intersections remain inspectable.
    """
    _validate_views(views)
    if not isinstance(per_page, int) or isinstance(per_page, bool) or not 1 <= per_page <= 4:
        raise ValueError("Exact ellipse pages support one to four rows per page.")
    if not np.isfinite(e_threshold) or e_threshold < 0 or not isinstance(max_vertices, int) or max_vertices < 1:
        raise ValueError("Use a finite nonnegative E threshold and positive vertex capacity.")
    if edges != "auto" and (not isinstance(edges, int) or isinstance(edges, bool)):
        raise ValueError("edges must be auto or an explicit integer count.")
    rows, selection = select_native_rows(
        [v.weights for v in views], list(views[0].row_labels), own_columns=views[0].own_columns,
        resolution=resolution, coverage=coverage, max_edges=max_edges,
        edges=None if edges == "auto" else edges,
    )
    # Keep the selected count independent of graphic capacity. Balanced pages
    # avoid an unnecessary three-plus-one arrangement for four native rows.
    pages = [part.tolist() for part in np.array_split(rows, int(np.ceil(len(rows) / per_page)))]
    plans = [(v, [_page_plan(v, page, e_threshold, max_vertices) for page in pages]) for v in views]
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse, FancyArrowPatch
    import textwrap

    palette = ["#218C91", "#CC922C", "#936CB2", "#4179A4", "#B5526C", "#639344",
               "#AA6A38", "#5D8890", "#9D7A33", "#776C93", "#3E8D73", "#BA6F4D"]
    colors = {row: palette[i % len(palette)] for i, row in enumerate(rows)}
    artifacts = []
    for view, page_plans in plans:
        if view.module_connections is not None and view.n_m:
            connections = select_connections(
                view.module_connections, view.member_labels[:view.n_m], kind=view.connection_kind,
                max_incoming=max_incoming, max_connections=max_connections,
            )
            connections["stage"] = view.connection_provenance
        else:
            connections = {"kind": view.connection_kind, "available": False, "connections": [],
                           "stage": view.connection_provenance,
                           "meaning": "No corresponding saved module connection matrix."}
        for page_index, (page, plan) in enumerate(zip(pages, page_plans), 1):
            weights, bits, members, positions, geometry, details = plan
            wide = len(page) >= 3
            if wide:
                positions, geometry = _wide_geometry(positions, geometry)
            _verify_intersections(positions, geometry, bits[members])
            fig = plt.figure(figsize=(16, 9))
            fig.suptitle(textwrap.fill(view.name, 66), fontsize=24, weight="bold", y=.97)
            ax = fig.add_axes([.035, .21, .93, .64] if wide else [.025, .10, .735, .77])
            ax.axis("off")
            ax.set_aspect("equal")
            for row, center, semi, angle in zip(page, geometry["centers"], geometry["semi_axes"], geometry["angles_degrees"]):
                color = colors[row]
                ax.add_patch(Ellipse(center, 2 * semi[0], 2 * semi[1], angle=angle,
                                     facecolor=color, edgecolor="none", alpha=.075, zorder=0))
                ax.add_patch(Ellipse(center, 2 * semi[0], 2 * semi[1], angle=angle,
                                     facecolor="none", edgecolor=color, linewidth=2.4, zorder=1))
            coordinate_by_label = {view.member_labels[i]: p for i, p in zip(members, positions)}
            displayed_connections = []
            directed_pairs = {(e["donor"], e["receiver"]) for e in connections["connections"]}
            for edge in connections["connections"]:
                donor, receiver = edge["donor"], edge["receiver"]
                if donor not in coordinate_by_label or receiver not in coordinate_by_label:
                    continue
                a, b = coordinate_by_label[donor], coordinate_by_label[receiver]
                reciprocal = view.connection_kind == "actual_mm" and (receiver, donor) in directed_pairs
                ax.add_patch(FancyArrowPatch(
                    a, b, arrowstyle="-|>" if view.connection_kind == "actual_mm" else "-",
                    mutation_scale=16, linewidth=edge["width"], color="#2A465C", alpha=.78,
                    linestyle="-" if view.connection_kind == "actual_mm" else "--",
                    connectionstyle=f"arc3,rad={.65 if reciprocal else .14}",
                    shrinkA=9, shrinkB=9, zorder=2,
                ))
                displayed_connections.append(edge)
            vertices = []
            for member, p in zip(members, positions):
                is_m = member < view.n_m
                ax.scatter(*p, marker="s" if is_m else "o", s=195 if is_m else 67,
                           facecolor="#2A465C" if is_m else "white", edgecolor="#2A465C",
                           linewidth=1.5, zorder=4)
                label = view.member_labels[member]
                ax.annotate(label, p, xytext=(0, 11 if is_m else 8), textcoords="offset points",
                            ha="center", va="bottom", fontsize=17 if is_m else 13.5,
                            weight="bold" if is_m else "normal", zorder=5,
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": .35, "alpha": .90})
                vertices.append({"member": label, "native_column": int(member),
                                 "mask": int(bits[member]), "abstract_xy": p.tolist(),
                                 "weights_by_page_row": weights[:, member].tolist()})
            limits = geometry.get("bounds", [-6.7, 6.7, -5.8, 5.0])
            ax.set(xlim=limits[:2], ylim=limits[2:])
            handles = [Line2D([], [], color=colors[row], lw=3, label=f"h({view.row_labels[row]})") for row in page]
            lower = [Line2D([], [], color="#2A465C", lw=w, label=l) for w, l in
                     [(1.2, "0 < w < 1/3"), (2.6, "1/3 ≤ w < 2/3"), (4.4, "2/3 ≤ w ≤ 1")]]
            line_title = "Applied MM: donor → receiver" if view.connection_kind == "actual_mm" else "Membership association"
            glyphs = [Line2D([], [], marker="s", color="none", markerfacecolor="#2A465C", markersize=10, label="Original M"),
                      Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor="#2A465C", markersize=8, label="Original E")]
            if wide:
                for rect, items, title, columns in (
                        ([.035, .02, .31, .155], handles, "Native hyperedges", 2),
                        ([.36, .02, .43, .155], lower if view.module_connections is not None else [], line_title, 3),
                        ([.82, .02, .17, .155], glyphs, None, 1)):
                    legend = fig.add_axes(rect); legend.axis("off")
                    if items:
                        legend.legend(handles=items, loc="center", title=title, ncol=columns,
                                      fontsize=16, title_fontsize=17, frameon=False,
                                      borderpad=0, columnspacing=1., handlelength=1.4)
            else:
                legend = fig.add_axes([.775, .14, .215, .69]); legend.axis("off")
                upper = legend.legend(handles=handles, loc="upper left", title="Native hyperedges", fontsize=18,
                                      title_fontsize=19, frameon=False, borderpad=0, handlelength=1.4)
                legend.add_artist(upper)
                if view.module_connections is not None:
                    line_legend = legend.legend(handles=lower, loc="center left", title=line_title.replace(": ", ":\n"),
                                                fontsize=17, title_fontsize=18, frameon=False, borderpad=0)
                    legend.add_artist(line_legend)
                legend.legend(handles=glyphs, loc="lower left", fontsize=17, frameon=False, borderpad=0)
            case = view.metadata.get("case", view.metadata.get("case_id", ""))
            page_label = f" · page {page_index}/{len(pages)}" if len(pages) > 1 else ""
            edge_label = "hyperedge" if len(rows) == 1 else "hyperedges"
            ax.set_title(f"{len(rows)} representative native {edge_label}{page_label}", fontsize=18, pad=4)
            model_slug = re.sub(r"[^A-Za-z0-9]+", "_", view.name).strip("_")
            stem = f"07_connected_{model_slug}_page{page_index:02d}"
            row_mass = "; ".join(f"{x['native_label']} {100*x['retained_mass']:.1f}%" if x["retained_mass"] is not None
                                 else f"{x['native_label']} empty" for x in details)
            line_sum = float(sum(x["weight"] for x in displayed_connections))
            positive_total = connections.get("total_weight", 0)
            line_meaning = (f"Actual {view.connection_provenance} MM gates, donor→receiver" if view.module_connections is not None and view.connection_kind == "actual_mm"
                            else "Derived symmetric shared-membership association" if view.module_connections is not None
                            else "No saved module interaction coefficients")
            caption = (
                f"{case}, {view.metadata.get('partition', 'saved scene')}, epoch {view.metadata.get('epoch', 'unspecified')}. "
                f"{view.metadata.get('row_semantics', 'Native organization rows')}. "
                f"{'Automatic profile selection' if edges == 'auto' else 'Explicit native-row selection'} uses resolution {resolution:g}, coverage quantile {coverage:g}; "
                f"{len(rows)} actual native rows selected from {len(view.row_labels)}; "
                f"{100*selection['covered_fraction']:.1f}% of native rows are within that profile distance; "
                f"residual quantile {selection['quantile_distance']:.3g}. "
                f"Displayed native rows: {', '.join(view.row_labels[i] for i in page)}. "
                f"E display threshold {e_threshold:g}; positive M and actual positive own-member entries retained. "
                f"Displayed row mass: {row_mass}. {line_meaning}; {len(displayed_connections)} lines shown, "
                f"using the three fixed coefficient bins in the legend after strongest-incoming-{max_incoming} and global-{max_connections} display limits. "
                "Original weights, omitted members/lines and cross-page incidence remain in numerical evidence. "
                "Ellipses encode exact displayed membership; abstract area/distance and display assignments have no physical group-count meaning."
            )
            evidence = {
                "identity": view.metadata, "selection": selection, "global_selected_rows": rows,
                "global_selected_labels": [view.row_labels[i] for i in rows],
                "page": page_index, "pages": len(pages), "page_rows": page,
                "E_display_threshold": e_threshold, "vertices": vertices,
                "edge_details": details, "ellipse_layout": geometry,
                "all_selected_native_weights": np.asarray(view.weights)[rows].tolist(),
                "all_member_labels": list(view.member_labels),
                "connections": connections, "displayed_connections": displayed_connections,
                "displayed_connection_weight": line_sum,
                "displayed_fraction_of_full_connection_mass": line_sum / positive_total if positive_total else None,
                "query_arrays_used": False, "physical_geometry_encoded": False,
            }
            artifacts.append(_save(fig, output, stem, "Native hyperedges with connectivity", caption, evidence))
    return artifacts


def render_connectivity_hyperedges(scenes: list[OrganizationScene], output: Path, **kwargs) -> list[FigureArtifact]:
    """Use matched saved E directories without reinterpreting native M/E roles."""
    require_matched(scenes)
    return render_native_connectivity([view_from_scene(scene) for scene in scenes], output, **kwargs)

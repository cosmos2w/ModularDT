"""Native mixed-member hyperedges with exact, display-only set geometry."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .organization_data import (
    FigureArtifact,
    OrganizationScene,
    apply_style,
    decorate_domain,
    require_matched,
    save_figure,
)


def select_representative_rows(
    scenes: list[OrganizationScene], edges: int = 3,
) -> tuple[list[int], dict]:
    """Select common directory exemplars from all aligned scene rows.

    Each scene contributes equally after row-wise L2 normalization. Own-E
    membership is omitted only for selection, to compare shared organization.
    The resulting clusters summarize a display; they are not learned groups.
    """
    require_matched(scenes)
    if not isinstance(edges, int) or not 1 <= edges <= 3:
        raise ValueError("The exact ellipse display supports one to three representative edges.")
    profiles = []
    for scene in scenes:
        shared = scene.shared
        norms = np.linalg.norm(shared, axis=1, keepdims=True)
        profiles.append(np.divide(shared, norms, out=np.zeros_like(shared), where=norms > 0))
    features = np.concatenate(profiles, axis=1) / np.sqrt(len(scenes))
    squared_norms = np.einsum("ij,ij->i", features, features)
    distances = np.sqrt(np.maximum(
        squared_norms[:, None] + squared_norms[None, :] - 2 * features @ features.T, 0,
    ))
    np.fill_diagonal(distances, 0)
    medoids = [int(distances.sum(axis=1).argmin())]
    for _ in range(min(edges, scenes[0].n_e) - 1):
        nearest = distances[:, medoids].min(axis=1)
        nearest[medoids] = -1
        candidate = int(nearest.argmax())
        if nearest[candidate] <= 1e-7:
            break
        medoids.append(candidate)
    history = [medoids.copy()]
    for _ in range(30):
        assignments = np.argmin(distances[:, medoids], axis=1)
        updated = []
        for group, old_medoid in enumerate(medoids):
            rows = np.flatnonzero(assignments == group)
            updated.append(int(rows[np.argmin(distances[np.ix_(rows, rows)].sum(axis=1))])
                           if len(rows) else old_medoid)
        if updated == medoids:
            break
        medoids = updated
        history.append(medoids.copy())
    xy = scenes[0].environment_xy
    medoids = sorted(medoids, key=lambda row: (float(xy[row, 0]), float(xy[row, 1]), row))
    assignments = np.argmin(distances[:, medoids], axis=1)
    return medoids, {
        "method": "Equal-scene concatenated L2-normalized shared directory profiles; global medoid, farthest-first initialization, then within-cluster medoid refinement; display order by physical x/y.",
        "selected_rows": medoids,
        "selected_addresses": [scenes[0].e_ids[row] for row in medoids],
        "selection_history_rows": history,
        "display_cluster_sizes": np.bincount(assignments, minlength=len(medoids)).tolist(),
        "display_assignment_by_row": assignments.tolist(),
        "own_diagonal_removed_only_for_selection": True,
    }


def _ellipse_layout(masks: np.ndarray, count: int) -> tuple[np.ndarray, dict]:
    """Place original vertices inside exactly their enclosing edge subsets."""
    masks = np.asarray(masks)
    if (isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 4
            or masks.ndim != 1 or not np.issubdtype(masks.dtype, np.integer)
            or np.any(masks < 0) or np.any(masks >= 2 ** count)):
        raise ValueError("Ellipse layout requires one to four edges and valid integer membership masks.")
    if count == 4:
        # Four genuine ellipses with all fifteen nonempty intersection regions.
        # Uniform scaling and translation keep the curves inside the existing
        # panel bounds without altering any set relationship.
        centers = .8 * np.array([
            [-1.5951006084154455, .9755269730929066],
            [-.28632181443576066, .5747441877069148],
            [-.40475071955780617, -.5961656927393335],
            [-1.8467073553515796, 1.395486016193479],
        ]) + np.array([0., -.3])
        axes = .8 * np.array([
            [5.42402525508483, 2.827692066979509],
            [5.9111528372095234, 1.617718294435266],
            [5.038704929117802, 2.2852439906309847],
            [5.389089135193927, 1.7210901569446246],
        ])
        angles = np.array([.4467172550047604, 1.8846789817011098,
                           .1511271554653648, 1.954395987807259])
        step, min_margin = .12, .12
    else:
        centers = np.array([[-1.6, 1.0], [1.6, 1.0], [0, -1.6]])[:count]
        axes = np.array([[4.6, 3.2], [4.6, 3.2], [4.8, 3.2]])[:count]
        angles = np.deg2rad([12, -14, -28])[:count]
        step, min_margin = .20, .19

    def distances(points: np.ndarray) -> np.ndarray:
        result = []
        for angle, center, semi_axes in zip(angles, centers, axes):
            delta = points - center
            u = delta[:, 0] * np.cos(angle) + delta[:, 1] * np.sin(angle)
            v = -delta[:, 0] * np.sin(angle) + delta[:, 1] * np.cos(angle)
            result.append((u / semi_axes[0]) ** 2 + (v / semi_axes[1]) ** 2)
        return np.asarray(result).T

    grid = np.array([(x, y) for x in np.arange(-6.7, 6.7, step)
                     for y in np.arange(-5.8, 5.0, step)])
    powers = 2 ** np.arange(count)
    geometry_choice = "all-intersections"
    if count == 4 and len(masks):
        # A second genuine four-ellipse arrangement gives more room to some
        # dense intersections. Select by available grid positions per shown
        # vertex, using only exact requested membership patterns. The general
        # all-intersections geometry remains a fallback for any other pattern.
        alternative = (
            np.array([[.8377522270843414, -1.2888227430232133],
                      [-.7795766025205335, .49432232040123636],
                      [-.8716917511525661, 1.12586130091612],
                      [1.6402172998754379, .619582135760012]]),
            np.array([[4.757429204178687, 1.8732211269111732],
                      [4.408358477698659, 1.8668510884430085],
                      [3.788363826266445, 1.6084822786724147],
                      [4.068598642704986, 2.8514910253690307]]),
            np.array([.9964213569524447, 2.2169538302803593,
                      1.6004274560620109, .7574702790800931]),
        )
        base = (centers, axes, angles)
        required, multiplicities = np.unique(masks, return_counts=True)
        choices = []
        for geometry in (base, alternative):
            centers, axes, angles = geometry
            test_distance = distances(grid)
            test_masks = ((test_distance < 1) * powers).sum(axis=1)
            test_margin = np.min(abs(test_distance - 1), axis=1)
            available = np.array([np.count_nonzero((test_masks == mask) & (test_margin > min_margin))
                                  for mask in required])
            choices.append(float(np.min(available / multiplicities)))
        index = int(np.argmax(choices))
        centers, axes, angles = (base, alternative)[index]
        geometry_choice = ("all-intersections", "expanded-selected-intersections")[index]
    grid_distance = distances(grid)
    grid_masks = ((grid_distance < 1) * powers).sum(axis=1)
    margin = np.min(abs(grid_distance - 1), axis=1)
    coordinates = np.empty((len(masks), 2))
    mask_order = np.unique(masks)
    if count == 4:
        # Fill the most constrained requested regions first, then space each
        # new point from all previously placed vertices, not just its own set.
        mask_order = sorted(mask_order, key=lambda mask: (
            np.count_nonzero((grid_masks == mask) & (margin > min_margin))
            / np.count_nonzero(masks == mask), int(mask)))
    placed = []
    for mask in mask_order:
        vertex_indices = np.flatnonzero(masks == mask)
        candidate_indices = np.flatnonzero((grid_masks == mask) & (margin > min_margin))
        if len(candidate_indices) < len(vertex_indices):
            raise ValueError("Too many vertices in one ellipse intersection; raise the E display threshold or select fewer edges.")
        points = grid[candidate_indices]
        chosen = []
        nearest = (np.min(np.linalg.norm(points[:, None] - np.asarray(placed)[None, :], axis=2), axis=1)
                   if count == 4 and placed else np.full(len(points), np.inf))
        for _ in vertex_indices:
            scores = (nearest + np.minimum(margin[candidate_indices], 1) * .18
                      if np.isfinite(nearest).all() else margin[candidate_indices])
            scores[chosen] = -1
            chosen.append(int(scores.argmax()))
            nearest = np.minimum(nearest, np.linalg.norm(points - points[chosen[-1]], axis=1))
        coordinates[vertex_indices] = points[chosen]
        placed.extend(points[chosen].tolist())
    actual_masks = ((distances(coordinates) < 1) * powers).sum(axis=1)
    if not np.array_equal(actual_masks, masks):
        raise ValueError("Ellipse layout does not match the original-member incidence.")
    return coordinates, {
        "centers": centers.tolist(), "semi_axes": axes.tolist(),
        "angles_degrees": np.rad2deg(angles).tolist(),
        "bounds": [-6.7, 6.7, -5.8, 5.0],
        "geometry_choice": geometry_choice,
        "exact_membership_verified": True,
    }


def render_native_hyperedges(
    scenes: list[OrganizationScene], output: Path, *,
    e_threshold: float = .02, edges: int = 3, max_vertices: int = 48,
) -> FigureArtifact:
    """Render one or two matched scenes using original M/E vertices.

    The E threshold is a disclosed visualization setting, never an execution
    rule. Positive own-E entries and every positive M entry remain visible.
    """
    require_matched(scenes)
    if len(scenes) > 2:
        raise ValueError("Native overlapping hyperedges support at most two scenes; request separate figures.")
    if not np.isfinite(e_threshold) or e_threshold < 0:
        raise ValueError("The E display threshold must be finite and nonnegative.")
    if not isinstance(max_vertices, int) or max_vertices < 1:
        raise ValueError("The explicit vertex display capacity must be a positive integer.")
    rows, selection = select_representative_rows(scenes, edges)
    count = len(rows)
    plans = []
    for scene in scenes:
        weights = scene.C[rows]
        keep = weights > 0
        keep[:, scene.n_m:] &= weights[:, scene.n_m:] >= e_threshold
        for edge, row in enumerate(rows):
            own = scene.own_columns[row]
            keep[edge, own] = weights[edge, own] > 0
        masks = (keep.astype(int) * (2 ** np.arange(count))[:, None]).sum(axis=0)
        members = np.flatnonzero(masks)
        if len(members) > max_vertices:
            raise ValueError(f"{scene.name} has {len(members)} displayed vertices, exceeding the explicit capacity {max_vertices}; raise the E display threshold or select fewer edges.")
        coordinates, ellipses = _ellipse_layout(masks[members], count)
        details = []
        for edge, row in enumerate(rows):
            omitted = np.flatnonzero((weights[edge] > 0) & ~keep[edge])
            total = float(weights[edge].sum())
            shown = float(weights[edge, keep[edge]].sum())
            details.append({
                "anchor": scene.e_ids[row], "anchor_row": row,
                "anchor_xy": scene.environment_xy[row].tolist(),
                "shown_M": int(keep[edge, :scene.n_m].sum()),
                "shown_E": int(keep[edge, scene.n_m:].sum()),
                "total_directory_mass": total, "shown_directory_mass": shown,
                "retained_directory_mass_fraction": shown / total if total > 0 else None,
                "omitted_positive_members": [{"member": scene.member_labels[i],
                                               "weight": float(weights[edge, i])} for i in omitted],
            })
        plans.append((scene, weights, masks, members, coordinates, ellipses, details))

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse

    apply_style()
    colors = ["#2D9CA5", "#DCA23C", "#9A77BD"]
    ink, module_color = "#263640", "#435A6C"
    fig = plt.figure(figsize=(8.8 * len(scenes), 8.7))
    title = "Mixed-member organization"
    fig.suptitle(title, fontsize=25, weight="bold", y=.98)
    evidence = {"selection": selection, "E_display_threshold": e_threshold,
                "max_vertices": max_vertices, "query_arrays_used": False,
                "interpretation": "Selected native directory rows, not learned clusters. Abstract coordinates encode exact set membership; ellipse areas and distances carry no physical meaning.",
                "scenes": {}}
    for panel, (scene, weights, masks, members, coordinates, ellipses, details) in enumerate(plans):
        panel_width = 1 / len(scenes)
        ax = fig.add_axes([panel * panel_width + .025 * panel_width, .19,
                           .95 * panel_width, .68])
        ax.axis("off")
        ax.set_aspect("equal")
        for center, semi_axes, angle, color in zip(
                ellipses["centers"], ellipses["semi_axes"], ellipses["angles_degrees"], colors):
            ax.add_patch(Ellipse(center, width=2 * semi_axes[0], height=2 * semi_axes[1],
                                 angle=angle, facecolor=color, edgecolor="none", alpha=.10, zorder=0))
            ax.add_patch(Ellipse(center, width=2 * semi_axes[0], height=2 * semi_axes[1],
                                 angle=angle, facecolor="none", edgecolor=color, linewidth=2, zorder=1))
        vertices = []
        own_edge = {int(scene.own_columns[row]): edge for edge, row in enumerate(rows)}
        for member, xy in zip(members, coordinates):
            is_m = member < scene.n_m
            own = own_edge.get(int(member))
            edge_color = colors[own] if own is not None else (module_color if is_m else "#596E7B")
            ax.scatter(*xy, s=205 if is_m else 75, marker="s" if is_m else "o",
                       facecolor=module_color if is_m else "white", edgecolor=edge_color,
                       linewidth=2 if own is not None else 1.2, zorder=4)
            label = scene.member_labels[member]
            ax.annotate(label, xy, xytext=(0, 12 if is_m else 9), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10.5 if is_m else 9.6,
                        weight="bold" if is_m else "normal", color=ink, zorder=5,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": .1, "alpha": .72})
            vertices.append({"original_member": label, "mask": int(masks[member]),
                             "abstract_xy": xy.tolist(), "weights_by_edge": weights[:, member].tolist()})
        label_xy = [(-4.5, 4.1), (4.6, 4.1), (-4.1, -5.1)]
        for row, xy, color in zip(rows, label_xy, colors):
            ax.text(*xy, f"h({scene.e_ids[row]})", color=color, ha="center", va="center",
                    fontsize=15, weight="bold", zorder=7,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.3})
        ax.set(xlim=(-6.7, 6.7), ylim=(-5.8, 5.0))
        ax.set_title(scene.name, fontsize=17, weight="bold", pad=12)
        evidence["scenes"][scene.name] = {
            "identity": scene.identity(), "vertices": vertices, "edge_details": details,
            "ellipse_layout": ellipses,
        }
    locator = fig.add_axes([.43, .025, .14, .125])
    reference = scenes[0]
    locator.scatter(*reference.environment_xy.T, s=4, color="#D9DEE1", zorder=1)
    decorate_domain(locator, reference)
    for row, color in zip(rows, colors):
        xy = reference.environment_xy[row]
        locator.scatter(*xy, s=28, color=color, zorder=5)
        locator.annotate(reference.e_ids[row], xy, xytext=(4, 4), textcoords="offset points",
                         fontsize=7.5, weight="bold", color=color)
    locator.set(xticks=[], yticks=[], xlabel="", ylabel="")
    locator.set_title("Anchor locations", fontsize=10, pad=4)
    for spine in locator.spines.values():
        spine.set_color("#ADB6BC")
    handles = [
        Line2D([], [], marker="s", markersize=10, linestyle="", markerfacecolor=module_color,
               markeredgecolor=module_color, label="Original M"),
        Line2D([], [], marker="o", markersize=7, linestyle="", markerfacecolor="white",
               markeredgecolor="#596E7B", label="Original E"),
        Line2D([], [], marker="o", markersize=7, linestyle="", markerfacecolor="white",
               markeredgecolor=colors[0], markeredgewidth=2, label="Own E (edge color)"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.045, .025),
               frameon=False, fontsize=10.5)
    labels = ", ".join(reference.e_ids[row] for row in rows)
    mass_notes = []
    for scene, _, _, _, _, _, details in plans:
        fractions = [item["retained_directory_mass_fraction"] for item in details
                     if item["retained_directory_mass_fraction"] is not None]
        if fractions:
            mass_notes.append(f"{scene.name}: {100 * min(fractions):.2f}–{100 * max(fractions):.2f}%")
    caption = (f"The same {len(rows)} representative native directory rows ({labels}) were selected from all "
               f"{reference.n_e} E addresses using equally weighted, normalized shared-directory profiles and deterministic medoid refinement; these are display exemplars, not learned groups. Each square is an original M and each circle an original E; the edge-colored rim marks its own E. Every vertex lies inside exactly its displayed hyperedges. E entries are displayed at directory weight ≥ {e_threshold:g}, with positive own-E entries and all positive M entries retained. Retained mass across selected edges: {'; '.join(mass_notes) if mass_notes else 'all selected directory rows are empty'}. The threshold changes no model execution; exact memberships, omissions and provenance are retained in the evidence. Ellipse shapes, areas and abstract distances are illustrative; the locator preserves physical anchor coordinates. No query samples or query-read weights are used.")
    return save_figure(fig, output, "06_native_overlapping_hyperedges", title, caption, evidence)

"""Whole-scene organization maps and value-preserving incidence reordering."""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, lengths, out=np.zeros_like(values), where=lengths > 0)


def _similarity_order(profiles: np.ndarray) -> np.ndarray:
    """Return a leaf permutation; neither values nor memberships are modified."""
    if len(profiles) < 2:
        return np.arange(len(profiles), dtype=int)
    normalized = _normalise_rows(profiles)
    # Cosine is undefined for an empty profile. Empty profiles have no overlap
    # with populated ones and are equivalent to each other for this ordering.
    distances = pdist(normalized, metric="cosine")
    empty = np.linalg.norm(normalized, axis=1) == 0
    pairs = np.triu_indices(len(normalized), 1)
    distances = np.where(empty[pairs[0]] & empty[pairs[1]], 0.0, distances)
    distances = np.clip(np.nan_to_num(distances, nan=1.0), 0.0, 2.0)
    return leaves_list(linkage(distances, method="average", optimal_ordering=True))


def _shared_permutations(matrices: list[np.ndarray], n_modules: int) -> tuple[np.ndarray, np.ndarray]:
    """Weight models equally and retain separate M/E member bands."""
    row_profiles = np.concatenate([_normalise_rows(values) for values in matrices], axis=1)
    column_profiles = np.concatenate([_normalise_rows(values.T) for values in matrices], axis=1)
    rows = np.r_[_similarity_order(row_profiles[:n_modules]), n_modules + _similarity_order(row_profiles[n_modules:])]
    columns = _similarity_order(column_profiles)
    return rows.astype(int), columns.astype(int)


def render_overview(scenes, output):
    """Show whole-domain dominant service and M participation in those regions."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch, Wedge
    from .organization_data import apply_style, decorate_domain, member_colors, plot_cells, save_figure

    apply_style()
    reference = scenes[0]
    winners, strengths = {}, {}
    region_set = set()
    for scene in scenes:
        shared = scene.shared
        strength = shared.max(axis=1)
        winner = shared.argmax(axis=1)
        winner = np.where(strength > 0, winner, -1)
        winners[scene.name], strengths[scene.name] = winner, strength
        region_set.update(int(i) for i in winner if i >= 0)
    region_columns = sorted(region_set)
    labels = [reference.member_labels[i] for i in region_columns]
    colors = member_colors(labels)
    cmap = ListedColormap(["#EEEEEE"] + [colors[label] for label in labels])
    single = len(scenes) == 1
    fig, axes = plt.subplots(len(scenes), 2, figsize=(16, 9 if single else 5.8 * len(scenes)),
                             squeeze=False, gridspec_kw={"width_ratios": [1.7, 1.0], "wspace": .25, "hspace": .35})
    evidence = {"display_regions": labels, "models": {}, "query_arrays_used": False,
                "region_rule": "argmax of the nonself prepared directory; ties follow original member order; all-zero rows remain unassigned",
                "participation_rule": "sum each original M directory column over addresses in each display region, then normalize across regions per M"}
    table_image = None
    for row, scene in enumerate(scenes):
        winner = winners[scene.name]
        region_index = {col: index + 1 for index, col in enumerate(region_columns)}
        coded = np.array([region_index.get(int(i), 0) for i in winner], dtype=float)
        ax = axes[row, 0]
        plot_cells(ax, scene, coded, cmap=cmap, vmin=-.5, vmax=len(labels)+.5)
        decorate_domain(ax, scene, labels=False)
        mass = np.zeros((scene.n_m, len(region_columns)))
        for col, member in enumerate(region_columns):
            mass[:, col] = scene.shared[winner == member, :scene.n_m].sum(axis=0)
        fractions = _normalise_mass(mass)
        extent = max(scene.bounds[1] - scene.bounds[0], scene.bounds[3] - scene.bounds[2])
        for i, xy in enumerate(scene.module_xy):
            radius = scene.module_radius[i] if scene.module_radius is not None else extent * .025
            if fractions[i].sum() > 0:
                angle = 90.
                for col, share in enumerate(fractions[i]):
                    if share <= 0:
                        continue
                    end = angle + share * 360
                    ax.add_patch(Wedge(xy, radius * 1.08, angle, end,
                                       width=radius * .30, facecolor=colors[labels[col]],
                                       edgecolor="white", linewidth=.4, zorder=5))
                    angle = end
            ax.text(*xy, scene.m_ids[i], ha="center", va="center", fontsize=11, fontweight="bold", zorder=6)
        ax.set_title("Dominant member" if single else f"{scene.name} · service regions", loc="left", fontsize=18)
        ax.set(xlabel="x", ylabel="y")
        ax.xaxis.label.set_size(15); ax.yaxis.label.set_size(15)
        ax.tick_params(labelsize=13)
        tax = axes[row, 1]
        tax.set_title("M participation" if single else f"{scene.name} · M participation", loc="left", fontsize=18)
        if len(labels) and scene.n_m:
            table_image = tax.imshow(fractions, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            tax.set_yticks(range(scene.n_m), scene.m_ids)
            tax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
            tax.set(xlabel="Dominant original member", ylabel="Original M")
            tax.tick_params(labelsize=13)
            tax.xaxis.label.set_size(15); tax.yaxis.label.set_size(15)
            if scene.n_m <= 18 and len(labels) <= 14:
                for i, j in np.argwhere(fractions >= .08):
                    tax.text(j, i, f"{fractions[i,j]:.0%}", ha="center", va="center", fontsize=13,
                             color="white" if fractions[i,j] >= .55 else "#23343F")
        else:
            tax.text(.5, .5, "No shared M participation", ha="center", va="center", transform=tax.transAxes)
            tax.set_axis_off()
        evidence["models"][scene.name] = {
            "region_for_each_E": [reference.member_labels[i] if i >= 0 else None for i in winner],
            "unassigned_addresses": [scene.e_ids[i] for i in np.flatnonzero(winner < 0)],
            "M_by_region_mass": mass.tolist(), "M_by_region_fraction": fractions.tolist(),
            "M_ids": list(scene.m_ids), "address_ids": list(scene.e_ids)}
    handles = [Patch(facecolor=colors[label], label=label) for label in labels]
    if any(np.any(winners[s.name] < 0) for s in scenes):
        handles.append(Patch(facecolor="#EEEEEE", label="Own only / no shared member"))
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(8, len(handles)), frameon=False,
                   bbox_to_anchor=(.47, .06), fontsize=14)
    legend_height = .07 + .035 * max(0, (len(handles) - 1) // 8)
    fig.subplots_adjust(left=.05, right=.90, top=.84, bottom=legend_height + .08)
    if single:
        axes[0, 0].set_position([.055, .28, .53, .48])
        axes[0, 1].set_position([.66, .28, .235, .48])
    if table_image is not None:
        cax = fig.add_axes([.925, .28, .014, .48] if single else [.93, .25, .014, .40])
        cb = fig.colorbar(table_image, cax=cax)
        cb.set_label("M service fraction", fontsize=14)
        cb.ax.tick_params(labelsize=13)
    title = "Whole-layout access regions"
    fig.suptitle(title, fontsize=24, fontweight="bold", y=.955)
    caption = "Each physical E cell is colored by its strongest nonself member in the prepared directory. These are display regions, not newly learned or merged hyperedges. M ring sectors and the table show where each original M's directory weight is distributed across those same regions; fractions sum to one for an M with positive shared service. Neutral cells have no positive nonself directory entry. All original members and soft directory weights remain unchanged; no query weights enter this figure."
    return save_figure(fig, output, "01_global_joint_ME_regions", title, caption, evidence)


def _normalise_mass(values: np.ndarray) -> np.ndarray:
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)


def render_incidence(scenes, output):
    """Compare physical and shared similarity order without changing incidence."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from .organization_data import apply_style, save_figure

    apply_style()
    reference = scenes[0]
    physical = np.lexsort((reference.environment_xy[:, 0], reference.environment_xy[:, 1]))
    donors = np.zeros(reference.n_e, dtype=bool)
    for scene in scenes:
        donors |= np.any(scene.shared[:, scene.n_m:] > 0, axis=0)
    e_rows = np.array([i for i in physical if donors[i]], dtype=int)
    member_columns = np.r_[np.arange(reference.n_m), reference.n_m + e_rows]
    member_labels = [reference.member_labels[i] for i in member_columns]
    matrices = [scene.shared[:, member_columns].T for scene in scenes]
    if len(member_columns):
        sorted_rows, sorted_columns = _shared_permutations(matrices, reference.n_m)
    else:
        sorted_rows, sorted_columns = np.array([], dtype=int), physical.copy()
    positive = np.concatenate([matrix[matrix > 0] for matrix in matrices])
    vmax = float(positive.max()) if positive.size else 1.
    # A shared display floor changes colors only, never numerical evidence.
    vmin = min(1e-4, vmax / 100)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    norm = LogNorm(vmin=vmin, vmax=vmax)
    single = len(scenes) == 1
    fig, axes = plt.subplots(len(scenes), 2, figsize=(16, 9 if single else 5.8 * len(scenes)), squeeze=False,
                             gridspec_kw={"wspace": .26, "hspace": .32})
    evidence = {"member_ids_physical": member_labels,
                "address_ids_physical": [reference.e_ids[i] for i in physical],
                "member_ids_sorted": [member_labels[i] for i in sorted_rows],
                "address_ids_sorted": [reference.e_ids[i] for i in sorted_columns],
                "row_permutation": sorted_rows.tolist(), "address_permutation": sorted_columns.tolist(),
                "color_scale": {"normalization": "log", "vmin": vmin, "vmax": vmax,
                                "zero_color": "white"},
                "sorting": "Average-linkage cosine clustering with optimal leaf ordering, using concatenated unit-normalized model profiles and separate M/E member bands. Identical permutations are applied to all compared models; there is no prescribed group count.",
                "query_arrays_used": False, "models": {}}
    image = None
    for row, (scene, matrix) in enumerate(zip(scenes, matrices)):
        for col in range(2):
            ax = axes[row, col]
            rows = np.arange(len(member_columns)) if col == 0 else sorted_rows
            columns = physical if col == 0 else sorted_columns
            displayed = matrix[rows][:, columns]
            if len(member_columns):
                image = ax.imshow(np.ma.masked_less_equal(displayed, 0), origin="upper", aspect="auto",
                                  interpolation="nearest", cmap=cmap, norm=norm)
                if 0 < reference.n_m < len(member_columns):
                    ax.axhline(reference.n_m - .5, color="#D23B36", lw=1.3)
                # Sample IDs at a uniform readable spacing; the complete crosswalk is retained.
                tick_count = min(11, len(member_columns))
                yt = np.unique(np.linspace(0, len(member_columns)-1, tick_count).astype(int))
                ax.set_yticks(yt, [member_labels[rows[i]] for i in yt], fontsize=13)
                xt = np.unique(np.linspace(0, reference.n_e-1, min(8, reference.n_e)).astype(int))
                ax.set_xticks(xt, [reference.e_ids[columns[i]] for i in xt], fontsize=13)
            else:
                ax.text(.5, .5, "No nonself directory members", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            order_title = "Physical order" if col == 0 else "Similarity order"
            ax.set_title(order_title if single else f"{scene.name} · {order_title}", loc="left", fontsize=18)
            ax.set(xlabel="E address", ylabel="Original member (M / E)")
            ax.xaxis.label.set_size(15); ax.yaxis.label.set_size(15)
        evidence["models"][scene.name] = {
            "nonzero_shared_entries": int(np.count_nonzero(matrix)),
            "shared_weight_sum": float(matrix.sum()),
            "nonself_E_donors": int(np.any(scene.shared[:, scene.n_m:] > 0, axis=0).sum()),
            "own_diagonal_mass_hidden": float(scene.C[np.arange(scene.n_e), scene.own_columns].sum()),
            "reordering_preserves_exact_values": bool(np.array_equal(np.sort(matrix.ravel()), np.sort(matrix[sorted_rows][:, sorted_columns].ravel())))}
    fig.subplots_adjust(left=.075, right=.89, bottom=.16, top=.82)
    if image is not None:
        cax = fig.add_axes([.925, .23, .014, .52])
        cb = fig.colorbar(image, cax=cax)
        cb.set_label("Directory weight", fontsize=14)
        cb.ax.tick_params(labelsize=13)
    title = "Native incidence"
    fig.suptitle(title, fontsize=24, fontweight="bold", y=.955)
    caption = "Columns are all original E addresses; rows are all active M members followed by the union of E donors with positive nonself service in the compared models. The own-E diagonal is hidden only in this display. Left panels use physical y/x order; right panels use a single shared data-driven row/column permutation, with separate M/E row bands. Sorting preserves every matrix value and nonzero, imposes no group count, and cannot turn shared hubs into disjoint groups. All panels share an absolute logarithmic color scale; zeros are white. Original IDs and the complete permutation are retained in the numerical evidence."
    return save_figure(fig, output, "02_directory_physical_vs_sorted", title, caption, evidence)

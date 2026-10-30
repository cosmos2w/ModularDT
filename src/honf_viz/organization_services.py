"""Matched, scene-wide member services and actual contextual neighborhoods.

All inputs are already prepared arrays. Directory services and contextual gates
remain distinct quantities; neither is interpreted as query contribution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .organization_data import (
    FigureArtifact, OrganizationScene, apply_style, decorate_domain, plot_cells,
    require_matched, save_figure,
)


def select_service_members(scenes: list[OrganizationScene], *, per_role: int = 2) -> list[dict]:
    """Choose matched native IDs by pooled global nonself directory service.

    Selection is performed independently within each input role. A zero-scoring
    role still receives deterministic examples, recorded with their zero scores,
    rather than disappearing from the comparison. No grouping is learned here.
    """
    require_matched(scenes)
    if isinstance(per_role, bool) or not isinstance(per_role, int) or per_role < 1:
        raise ValueError("per_role must be a positive integer.")
    reference = scenes[0]
    selection = []
    totals = np.stack([scene.shared.sum(axis=0) for scene in scenes])
    for role, start, labels in (("M", 0, reference.m_ids),
                                 ("E", reference.n_m, reference.e_ids)):
        aggregate = totals[:, start:start + len(labels)].sum(axis=0)
        order = np.argsort(-aggregate, kind="stable")[:per_role]
        for index in order:
            index = int(index)
            column = start + index
            selection.append({
                "role": role, "id": labels[index], "role_index": index,
                "column_index": column, "pooled_nonself_directory_mass": float(aggregate[index]),
                "per_model": [{"model": s.name, "directory_mass": float(totals[i, column]),
                               "served_E_addresses": int(np.count_nonzero(s.shared[:, column]))}
                              for i, s in enumerate(scenes)],
            })
    return selection


def _plot_module_gate_receivers(ax, scene, gates, source_xy, cmap, norm):
    """Draw every positive actual M receiver; arrows run donor -> receiver."""
    from matplotlib.patches import FancyArrowPatch
    for index in np.flatnonzero(gates > 0):
        weight = float(gates[index])
        xy = scene.module_xy[index]
        color = cmap(norm(weight))
        if not np.array_equal(source_xy, xy):
            ax.add_patch(FancyArrowPatch(
                source_xy, xy, arrowstyle="-|>", mutation_scale=9,
                shrinkA=7, shrinkB=7, connectionstyle="arc3,rad=.04",
                color=color, linewidth=.6 + 2 * weight,
                linestyle=(0, (3, 2)), zorder=5,
            ))
        ax.scatter(*xy, marker="s", s=70, c=[color], edgecolors="#263743",
                   linewidths=.7, zorder=6)


def _label_modules(ax, scene, *, fontsize=8, avoid_xy=None):
    offset = (scene.bounds[3] - scene.bounds[2]) * .045
    for label, xy in zip(scene.m_ids, scene.module_xy):
        label_y = xy[1] + offset
        alignment = "bottom"
        if avoid_xy is not None:
            label_pixel = ax.transData.transform((xy[0], label_y))
            marker_pixel = ax.transData.transform(avoid_xy)
            # Account for the marker radius and label height in display space.
            margin = ax.figure.dpi / 72
            if (abs(label_pixel[0] - marker_pixel[0]) < (fontsize * len(label) * .4 + 10) * margin
                    and abs(label_pixel[1] - marker_pixel[1]) < (fontsize + 10) * margin):
                label_y = xy[1] - offset
                alignment = "top"
        ax.text(xy[0], label_y, label, ha="center", va=alignment,
                fontsize=fontsize, color="#23343F", zorder=10,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": .3})


def _color_scale(values):
    from matplotlib.colors import Normalize
    maximum = max((float(np.max(a)) for a in values if a.size), default=0.)
    return Normalize(vmin=0., vmax=maximum if maximum > 0 else 1.)


def render_donor_services(scenes: list[OrganizationScene], output: Path, *,
                          per_role: int = 2) -> list[FigureArtifact]:
    """Show C services to E addresses plus separately scaled actual M gates.

    Each figure uses a single donor role, matching the same globally selected
    original IDs in all model rows. C colors and G colors use separate scales.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    selection = select_service_members(scenes, per_role=per_role)
    apply_style()
    c_cmap = plt.get_cmap("viridis").copy()
    c_cmap.set_bad("white")
    g_cmap = plt.get_cmap("Oranges")
    g_norm = Normalize(0, 1)
    artifacts = []
    for role in ("M", "E"):
        members = [m for m in selection if m["role"] == role]
        if not members:
            continue
        c_norm = _color_scale([s.shared[:, m["column_index"]] for s in scenes for m in members])
        single = len(scenes) == 1
        fig, axes = plt.subplots(len(scenes), len(members),
                                 figsize=(16, 9) if single else (5.2 * len(members), 3.5 * len(scenes) + 1.55),
                                 squeeze=False)
        fig.subplots_adjust(left=.055, right=.98, bottom=.20 if single else .16,
                            top=.80 if single else .85, wspace=.18, hspace=.25)
        title = f"{role}-member services"
        fig.suptitle(title, fontsize=24 if single else 21, fontweight="bold", y=.975)
        panel_evidence = []
        present_routes = set()
        for row, scene in enumerate(scenes):
            for col, member in enumerate(members):
                ax = axes[row, col]
                values = scene.shared[:, member["column_index"]]
                shown = np.where(values > 0, values, np.nan)
                plot_cells(ax, scene, shown, cmap=c_cmap, norm=c_norm)
                decorate_domain(ax, scene)
                index = member["role_index"]
                source_xy = (scene.module_xy if role == "M" else scene.environment_xy)[index]
                route = "mm" if role == "M" else "me"
                actual = scene.gates.get(route)
                gates = actual[:, index] if actual is not None else None
                if gates is not None:
                    present_routes.add(route)
                    _plot_module_gate_receivers(ax, scene, gates, source_xy, g_cmap, g_norm)
                _label_modules(ax, scene, fontsize=13 if single else 8, avoid_xy=source_xy)
                ax.scatter(*source_xy, s=190, marker="o", facecolors="none",
                           edgecolors="#CE2736", linewidths=2, zorder=11)
                if role == "E":
                    ax.annotate(member["id"], source_xy, xytext=(5, 7), textcoords="offset points",
                                fontsize=13 if single else 9, color="#CE2736", zorder=12,
                                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": .3})
                panel_title = member["id"] if single else f'{scene.name} · {member["id"]}'
                ax.set_title(panel_title, fontsize=20 if single else 13, pad=9)
                ax.tick_params(labelsize=13 if single else 9)
                if single:
                    ax.xaxis.label.set_size(14)
                    ax.yaxis.label.set_size(14)
                panel_evidence.append({
                    "model": scene.name, "supplier": member["id"], "role": role,
                    "C_values_to_E_addresses": values.tolist(),
                    "C_positive_E_ids": [scene.e_ids[i] for i in np.flatnonzero(values > 0)],
                    "M_receiver_route": route if gates is not None else None,
                    "M_receiver_stage": scene.provenance.get(route),
                    "G_values_to_M_receivers": gates.tolist() if gates is not None else None,
                    "G_positive_M_ids": ([scene.m_ids[i] for i in np.flatnonzero(gates > 0)]
                                         if gates is not None else []),
                })
        handles = [Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="#CE2736",
                          markeredgewidth=2, linestyle="none", markersize=10, label="Supplier")]
        if present_routes:
            handles.extend([
                Line2D([], [], marker="s", markerfacecolor=g_cmap(.6), markeredgecolor="#263743",
                       linestyle="none", markersize=8, label="M receiver · G"),
                Line2D([], [], color=g_cmap(.8), linestyle="--", label="Supplier → M receiver"),
            ])
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .935),
                   ncol=len(handles), frameon=False, fontsize=14 if single else 10)
        cax = fig.add_axes([.12, .075, .36, .018])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=c_norm, cmap=c_cmap), cax=cax, orientation="horizontal")
        cb.set_label("Prepared directory C · E services", fontsize=14 if single else 10)
        cb.ax.tick_params(labelsize=12 if single else 10)
        if present_routes:
            gax = fig.add_axes([.61, .075, .27, .018])
            gb = fig.colorbar(plt.cm.ScalarMappable(norm=g_norm, cmap=g_cmap), cax=gax, orientation="horizontal")
            gb.set_label("Actual contextual gate G · M receivers", fontsize=14 if single else 10)
            gb.ax.tick_params(labelsize=12 if single else 10)
        caption = (
            f"Matched {role} suppliers are selected by pooled nonself directory mass across all supplied models, "
            "independently within M and E roles. Each E cell shows the prepared directory weight C for accessing "
            "the highlighted original supplier; own-E directory entries are omitted only from this service display. "
            "Orange squares and dashed supplier-to-M arrows show actual contextual gates G on a separate 0–1 scale, "
            f"using {'MM (M←M)' if role == 'M' else 'ME (M←E)'}. Every positive M receiver is shown without an added threshold. "
            "C and G belong to distinct operations and are not added together; G stages are recorded in the evidence. "
            "White cells have zero nonself C. The red ring locates the supplier, not a query. "
            "Selection is a display choice and does not create learned groups."
        )
        if not present_routes:
            caption += " No matching actual M-receiver gate was exported, so no M gate overlay is drawn."
        artifacts.append(save_figure(fig, output, f"03_directory_services_{len(artifacts) + 1:02d}", title,
                                     caption, {"weight": "C for E services; actual G for M receivers",
                                               "selection": members, "panels": panel_evidence,
                                               "C_scale": [c_norm.vmin, c_norm.vmax], "G_scale": [0, 1]}))
    return artifacts


def render_incoming_neighborhoods(scenes: list[OrganizationScene], output: Path, *,
                                  per_role: int = 2, receiver_role: str | None = None) -> FigureArtifact:
    """Show actual neighborhoods, optionally for only M or E receivers."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    selection = select_service_members(scenes, per_role=per_role)
    if receiver_role not in (None, "M", "E"):
        raise ValueError("receiver_role must be M, E, or None.")
    if receiver_role is not None:
        selection = [member for member in selection if member["role"] == receiver_role]
        if not selection:
            raise ValueError(f"No {receiver_role} receivers are available.")
    if any(set(scene.gates) < {"mm", "em", "ee", "me"} for scene in scenes):
        raise ValueError("Four-route neighborhoods require saved actual MM, EM, EE and ME gates for every model.")
    apply_style()
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    norm = Normalize(0, 1)
    single = len(scenes) == 1
    if single:
        columns = min(2, len(selection))
        rows = (len(selection) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(16, 9), squeeze=False)
        for unused in axes.flat[len(selection):]:
            unused.set_visible(False)
        fig.subplots_adjust(left=.05, right=.98, bottom=.16, top=.80, wspace=.20, hspace=.45)
    else:
        fig, axes = plt.subplots(len(scenes), len(selection),
                                 figsize=(4.7 * len(selection), 3.45 * len(scenes) + 1.4), squeeze=False)
        fig.subplots_adjust(left=.04, right=.985, bottom=.15, top=.84, wspace=.17, hspace=.25)
    title = f"Actual {receiver_role} receiver neighborhoods" if receiver_role else "Actual M / E information neighborhoods"
    fig.suptitle(title, fontsize=24 if single else 21, fontweight="bold", y=.975)
    panels = []
    for row, scene in enumerate(scenes):
        for col, member in enumerate(selection):
            ax = axes.flat[col] if single else axes[row, col]
            member_role, index = member["role"], member["role_index"]
            m_route, e_route = ("mm", "me") if member_role == "M" else ("em", "ee")
            mg, eg = scene.gates[m_route][index], scene.gates[e_route][index]
            plot_cells(ax, scene, np.where(eg > 0, eg, np.nan), cmap=cmap, norm=norm)
            decorate_domain(ax, scene)
            xy = (scene.module_xy if member_role == "M" else scene.environment_xy)[index]
            for donor in np.flatnonzero(mg > 0):
                weight = float(mg[donor])
                dxy = scene.module_xy[donor]
                if not np.array_equal(dxy, xy):
                    ax.add_patch(FancyArrowPatch(
                        dxy, xy, arrowstyle="-|>", mutation_scale=8, shrinkA=7, shrinkB=8,
                        connectionstyle="arc3,rad=.04", color=cmap(norm(weight)),
                        linewidth=.6 + 2 * weight, linestyle=(0, (3, 2)), zorder=5,
                    ))
                ax.scatter(*dxy, s=70, marker="s", c=[cmap(norm(weight))],
                           edgecolors="#263743", linewidths=.7, zorder=6)
            _label_modules(ax, scene, fontsize=12 if single else 8, avoid_xy=xy)
            ax.scatter(*xy, s=190, marker="s", facecolors="none", edgecolors="#CE2736",
                       linewidths=2, zorder=11)
            ax.scatter(*xy, s=55, marker="x", c="#CE2736", linewidths=1.7, zorder=12)
            panel_title = f'{member["id"]} · {m_route.upper()} + {e_route.upper()}'
            if not single:
                panel_title = f'{scene.name} · {panel_title}'
            ax.set_title(panel_title, fontsize=19 if single else 11.5, pad=9)
            ax.tick_params(labelsize=12 if single else 8)
            if single:
                ax.xaxis.label.set_size(13)
                ax.yaxis.label.set_size(13)
            panels.append({"model": scene.name, "receiver": member["id"],
                           "M_route": m_route, "E_route": e_route,
                           "M_stage": scene.provenance[m_route], "E_stage": scene.provenance[e_route],
                           "positive_M_ids": [scene.m_ids[i] for i in np.flatnonzero(mg > 0)],
                           "positive_E_ids": [scene.e_ids[i] for i in np.flatnonzero(eg > 0)],
                           "M_gate_values": mg.tolist(), "E_gate_values": eg.tolist()})
    handles = [Line2D([], [], marker="s", markerfacecolor="none", markeredgecolor="#CE2736",
                      markeredgewidth=2, linestyle="none", markersize=10, label="Receiver"),
               Line2D([], [], marker="s", markerfacecolor=cmap(.5), markeredgecolor="#263743",
                      linestyle="none", markersize=8, label="M donor"),
               Line2D([], [], color="#263743", linestyle="--", label="M donor → receiver")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .93),
               ncol=3, frameon=False, fontsize=14 if single else 10)
    cax = fig.add_axes([.30, .07, .40, .018])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label("Actual contextual relation gate G", fontsize=14 if single else 10)
    cb.ax.tick_params(labelsize=12 if single else 10)
    caption = (
        "The same original receivers are displayed for all models, selected by pooled nonself directory service "
        "mass independently within M and E roles. For an M receiver, E cells show its ME (M←E) gate row and M "
        "donors its MM (M←M) gate row; for an E receiver, E cells show EE (E←E) and M donors EM (E←M). "
        "All panels share the actual G scale, with exact zeros white and all positive M donor relations shown. "
        "Dashed arrows run from M donor to receiver. The contextual stages execute sequentially MM→EM→EE→ME, "
        "so combined neighborhoods do not imply simultaneous influence. No structural own-E access is added. "
        "Red squares identify original M/E receivers, not queries. These are actual interaction gates, not "
        "prepared directory C or executed query weights."
    )
    stem = "05_four_route_context_neighborhoods" + (f"_{receiver_role}" if receiver_role else "")
    return save_figure(fig, output, stem, title, caption,
                       {"weight": "actual applied contextual G", "selection": selection,
                        "panels": panels, "scale": [0, 1]})

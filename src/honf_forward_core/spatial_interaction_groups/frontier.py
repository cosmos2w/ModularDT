"""Balanced geometry candidates and deterministic, reversible group cuts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class BalancedGridTemplate:
    """Geometry-only tree template in canonical grid-cell order."""

    nx: int
    ny: int
    x_values: tuple[float, ...]
    y_values: tuple[float, ...]
    leaf_cells: tuple[tuple[int, int], ...]
    children: tuple[tuple[int, int] | None, ...]
    parents: tuple[int, ...]
    depths: tuple[int, ...]
    intervals: tuple[tuple[int, int], ...]
    box_min: tuple[tuple[float, float], ...]
    box_max: tuple[tuple[float, float], ...]
    grid_step: tuple[float, float]

    @property
    def leaf_count(self) -> int:
        return self.nx * self.ny

    @property
    def node_count(self) -> int:
        return len(self.children)

    @property
    def root(self) -> int:
        return self.node_count - 1

    @property
    def max_depth(self) -> int:
        return max(self.depths, default=0)

    def bind(self, coordinates: np.ndarray) -> "BalancedGridTree":
        """Attach this canonical template to one raw record ordering."""
        coordinates = np.asarray(coordinates, dtype=np.float64)
        lookup = {(float(x), float(y)): i for i, (x, y) in enumerate(coordinates)}
        expected = {(x, y) for y in self.y_values for x in self.x_values}
        if set(lookup) != expected:
            raise ValueError("Environment coordinates differ from the cached regular-grid geometry.")
        order = tuple(lookup[(self.x_values[ix], self.y_values[iy])] for ix, iy in self.leaf_cells)
        return BalancedGridTree(self, order)


@dataclass(frozen=True)
class BalancedGridTree:
    """A canonical tree bound to original E rows for one scene."""

    template: BalancedGridTemplate
    leaf_order: tuple[int, ...]

    @property
    def nx(self) -> int:
        return self.template.nx

    @property
    def ny(self) -> int:
        return self.template.ny

    @property
    def leaf_count(self) -> int:
        return self.template.leaf_count

    @property
    def node_count(self) -> int:
        return self.template.node_count

    @property
    def root(self) -> int:
        return self.template.root

    @property
    def max_depth(self) -> int:
        return self.template.max_depth

    @property
    def children(self) -> tuple[tuple[int, int] | None, ...]:
        return self.template.children

    @property
    def parents(self) -> tuple[int, ...]:
        return self.template.parents

    @property
    def depths(self) -> tuple[int, ...]:
        return self.template.depths

    @property
    def intervals(self) -> tuple[tuple[int, int], ...]:
        return self.template.intervals

    @property
    def box_min(self) -> tuple[tuple[float, float], ...]:
        return self.template.box_min

    @property
    def box_max(self) -> tuple[tuple[float, float], ...]:
        return self.template.box_max

    @property
    def grid_step(self) -> tuple[float, float]:
        return self.template.grid_step

    @property
    def internal_nodes(self) -> tuple[int, ...]:
        return tuple(i for i, child in enumerate(self.children) if child is not None)

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(end - start for start, end in self.intervals)

    def cover_at_depth(self, depth: int) -> tuple[int, ...]:
        """Return the legal top-down cover at a fixed tree depth."""
        if depth < 0:
            raise ValueError("Cover depth must be nonnegative.")
        active: list[int] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            children = self.children[node]
            if children is None or self.depths[node] >= depth:
                active.append(node)
            else:
                stack.extend(reversed(children))
        return tuple(active)

    def cover_nearest_k(self, target: int) -> tuple[int, ...]:
        if target < 1:
            raise ValueError("Target group count must be positive.")
        covers = [self.cover_at_depth(depth) for depth in range(self.max_depth + 1)]
        return min(covers, key=lambda cover: (abs(len(cover) - target), len(cover), cover))

    def validate_cover(self, nodes: Sequence[int]) -> tuple[int, ...]:
        nodes = tuple(int(node) for node in nodes)
        if not nodes or len(set(nodes)) != len(nodes):
            raise ValueError("A forced cover must contain unique tree nodes.")
        if any(node < 0 or node >= self.node_count for node in nodes):
            raise ValueError("A forced cover contains an unknown tree node.")
        intervals = sorted(self.intervals[node] for node in nodes)
        cursor = 0
        for start, end in intervals:
            if start != cursor or end <= start:
                raise ValueError("Forced nodes do not form a nonoverlapping complete cover.")
            cursor = end
        if cursor != self.leaf_count:
            raise ValueError("Forced nodes do not cover every environment record.")
        return tuple(sorted(nodes, key=lambda node: self.intervals[node][0]))

    def cut_from_decisions(
        self, split: Sequence[bool], override: Any = None
    ) -> tuple[int, ...]:
        """Build a legal cut, changing only the requested path and descendants."""
        if len(split) != self.node_count:
            raise ValueError("One split decision per hierarchy node is required.")
        if override is None:
            forced_cover = None
            mode = None
            node = None
        elif isinstance(override, Mapping):
            if "cover" in override:
                forced_cover = self.validate_cover(override["cover"])
                mode = None
                node = None
            elif "depth" in override:
                forced_cover = self.cover_at_depth(int(override["depth"]))
                mode = None
                node = None
            else:
                raise ValueError("Forced frontier mapping must contain 'cover' or 'depth'.")
        elif isinstance(override, Sequence) and not isinstance(override, (str, bytes)):
            if len(override) not in (2, 3):
                raise ValueError("A node override is (node, mode[, extra_depth]).")
            node = int(override[0])
            mode = str(override[1])
            extra_depth = int(override[2]) if len(override) == 3 else 1
            forced_cover = None
            if node < 0 or node >= self.node_count or self.children[node] is None:
                raise ValueError("A parent/descendant probe must target an internal node.")
            if mode not in {"parent", "children", "grandchildren"}:
                raise ValueError("Node override mode must be parent, children, or grandchildren.")
            if extra_depth < 1:
                raise ValueError("A descendant override must retain at least one level.")
            if extra_depth > 2:
                raise ValueError("A descendant override may open only one or two levels.")
            if len(override) == 3 and mode != "parent":
                mode = "children" if extra_depth == 1 else "grandchildren"
        else:
            raise TypeError("Unsupported forced-frontier value.")

        if forced_cover is not None:
            return forced_cover

        decisions = list(bool(value) for value in split)
        if node is not None:
            path: list[int] = []
            current = node
            while current != self.root:
                current = self.parents[current]
                path.append(current)
            for ancestor in path:
                decisions[ancestor] = True
            decisions[node] = mode != "parent"
            children = self.children[node]
            assert children is not None
            levels = 2 if mode == "grandchildren" else 1
            current_level = list(children)
            for level in range(1, levels + 1):
                next_level: list[int] = []
                for descendant in current_level:
                    if self.children[descendant] is None:
                        continue
                    decisions[descendant] = level < levels
                    if level < levels:
                        next_level.extend(self.children[descendant] or ())
                current_level = next_level

        active: list[int] = []
        stack = [self.root]
        while stack:
            current = stack.pop()
            children = self.children[current]
            if children is None or not decisions[current]:
                active.append(current)
            else:
                stack.extend(reversed(children))
        return tuple(active)

    def labels_for_cut(self, active_nodes: Sequence[int]) -> tuple[int, ...]:
        active = self.validate_cover(active_nodes)
        labels = [-1] * self.leaf_count
        for group, node in enumerate(active):
            start, end = self.intervals[node]
            for leaf_slot in range(start, end):
                labels[self.leaf_order[leaf_slot]] = group
        if any(label < 0 for label in labels):
            raise RuntimeError("Tree cut failed to label an original environment record.")
        return tuple(labels)


def geometry_key(coordinates: np.ndarray) -> tuple[tuple[float, float], ...]:
    """Exact, order-independent geometry key; IDs and record order are excluded."""
    points = np.asarray(coordinates, dtype=np.float64)
    return tuple(sorted((float(x), float(y)) for x, y in points))


def build_balanced_grid_template(
    coordinates: np.ndarray, nx: int, ny: int
) -> BalancedGridTemplate:
    """Build a deterministic binary hierarchy by longest physical extent."""
    points = np.asarray(coordinates, dtype=np.float64)
    if points.shape != (nx * ny, 2) or nx < 1 or ny < 1:
        raise ValueError("Coordinates must be the complete declared two-dimensional grid.")
    if not np.isfinite(points).all():
        raise ValueError("Grid coordinates must be finite.")
    x_values = tuple(float(value) for value in np.unique(points[:, 0]))
    y_values = tuple(float(value) for value in np.unique(points[:, 1]))
    if len(x_values) != nx or len(y_values) != ny:
        raise ValueError("Declared grid dimensions differ from unique physical coordinates.")
    expected = {(x, y) for y in y_values for x in x_values}
    if len(set((float(x), float(y)) for x, y in points)) != nx * ny or set(
        (float(x), float(y)) for x, y in points
    ) != expected:
        raise ValueError("Environment coordinates must be a complete Cartesian grid.")

    x_diffs = np.diff(np.asarray(x_values, dtype=np.float64))
    y_diffs = np.diff(np.asarray(y_values, dtype=np.float64))
    if (x_diffs <= 0).any() or (y_diffs <= 0).any():
        raise ValueError("Grid coordinates must be strictly ordered on each axis.")
    candidates = [float(value) for value in np.concatenate((x_diffs, y_diffs)) if value > 0]
    fallback = min(candidates) if candidates else 1.0
    grid_step = (
        float(np.min(x_diffs)) if x_diffs.size else fallback,
        float(np.min(y_diffs)) if y_diffs.size else fallback,
    )

    @dataclass
    class _Node:
        depth: int
        x0: int
        x1: int
        y0: int
        y1: int
        left: Any = None
        right: Any = None
        leaf: bool = False
        node_id: int = -1

    def build(x0: int, x1: int, y0: int, y1: int, depth: int) -> _Node:
        if x1 - x0 == 1 and y1 - y0 == 1:
            return _Node(depth, x0, x1, y0, y1, leaf=True)
        x_extent = x_values[x1 - 1] - x_values[x0] if x1 - x0 > 1 else -1.0
        y_extent = y_values[y1 - 1] - y_values[y0] if y1 - y0 > 1 else -1.0
        if x_extent >= y_extent and x1 - x0 > 1:
            middle = x0 + (x1 - x0) // 2
            return _Node(
                depth,
                x0,
                x1,
                y0,
                y1,
                build(x0, middle, y0, y1, depth + 1),
                build(middle, x1, y0, y1, depth + 1),
            )
        middle = y0 + (y1 - y0) // 2
        return _Node(
            depth,
            x0,
            x1,
            y0,
            y1,
            build(x0, x1, y0, middle, depth + 1),
            build(x0, x1, middle, y1, depth + 1),
        )

    root = build(0, nx, 0, ny, 0)
    leaf_cells: list[tuple[int, int]] = []

    def assign_leaves(node: _Node) -> None:
        if node.leaf:
            node.node_id = len(leaf_cells)
            leaf_cells.append((node.x0, node.y0))
            return
        assign_leaves(node.left)
        assign_leaves(node.right)

    assign_leaves(root)
    count = nx * ny
    children: list[tuple[int, int] | None] = [None] * count
    parents = [-1] * count
    depths = [0] * count
    intervals: list[tuple[int, int]] = [(0, 0)] * count
    box_min: list[tuple[float, float]] = [(0.0, 0.0)] * count
    box_max: list[tuple[float, float]] = [(0.0, 0.0)] * count

    def assign_internal(node: _Node) -> tuple[int, int, int]:
        if node.leaf:
            start = node.node_id
            end = start + 1
            node_id = node.node_id
        else:
            left_id, start, middle = assign_internal(node.left)
            right_id, _right_start, end = assign_internal(node.right)
            node_id = len(children)
            children.append((left_id, right_id))
            parents.append(-1)
            depths.append(0)
            intervals.append((0, 0))
            box_min.append((0.0, 0.0))
            box_max.append((0.0, 0.0))
            parents[left_id] = node_id
            parents[right_id] = node_id
            children[node_id] = (left_id, right_id)
            start = min(start, middle)
        depths[node_id] = node.depth
        intervals[node_id] = (start, end)
        box_min[node_id] = (x_values[node.x0], y_values[node.y0])
        box_max[node_id] = (x_values[node.x1 - 1], y_values[node.y1 - 1])
        return node_id, start, end

    root_id, _start, _end = assign_internal(root)
    if root_id != len(children) - 1 or len(children) != 2 * nx * ny - 1:
        raise RuntimeError("Balanced hierarchy construction produced an invalid node layout.")
    return BalancedGridTemplate(
        nx,
        ny,
        x_values,
        y_values,
        tuple(leaf_cells),
        tuple(children),
        tuple(parents),
        tuple(depths),
        tuple(intervals),
        tuple(box_min),
        tuple(box_max),
        grid_step,
    )

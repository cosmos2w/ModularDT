"""Prepared receiver-tree geometry and batched access with unchanged tree math."""

import torch

from .adaptive_interaction_cover import endpoint_smoothstep


def build_receiver_tree_geometry(trees, capacity):
    """Build fresh continuous boundaries and discrete indices for one phase.

    Each tree declares weighted or historical unweighted child centroids.
    Nothing here persists outside the caller-owned prepared state.
    """
    axes = []
    parents = []
    left = []
    depths = []
    leaves = []
    valid = []
    boundary = []
    overlap = []
    for tree in trees:
        n = len(tree.nodes)
        axis = [0] * capacity
        parent = [0] * capacity
        isleft = [False] * capacity
        depth = [-1] * capacity
        leaf = [True] * capacity
        depth[0] = 0
        bounds = []
        widths = []
        for index, node in enumerate(tree.nodes):
            if node.is_leaf:
                bounds.append(tree.universe.coordinates.new_zeros(()))
                widths.append(tree.universe.coordinates.new_ones(()))
            else:
                axis[index] = node.split_axis
                leaf[index] = False
                for child, left_child in ((node.left, True), (node.right, False)):
                    parent[child] = index
                    isleft[child] = left_child
                    depth[child] = depth[index] + 1
                bounds.append(tree.split_boundary(index))
                widths.append(tree.overlap_fraction * tree.universe.coordinate_scale[node.split_axis])
        boundary.append(torch.nn.functional.pad(torch.stack(bounds), (0, capacity - n)))
        overlap.append(torch.nn.functional.pad(torch.stack(widths), (0, capacity - n), value=1.0))
        axes.append(axis)
        parents.append(parent)
        left.append(isleft)
        depths.append(depth)
        leaves.append(leaf)
        valid.append([True] * n + [False] * (capacity - n))
    device = trees[0].universe.coordinates.device
    return {
        "axes": torch.tensor(axes, device=device),
        "parents": torch.tensor(parents, device=device),
        "left": torch.tensor(left, device=device),
        "depths": torch.tensor(depths, device=device),
        "leaves": torch.tensor(leaves, device=device),
        "valid": torch.tensor(valid, device=device),
        "boundary": torch.stack(boundary),
        "overlap": torch.stack(overlap),
        "spatial_dim": trees[0].universe.coordinates.shape[-1],
    }


def receiver_tree_access(queries, gates, data, max_depth=3, *, soft_precision=False):
    """Evaluate the original recursive access with batched depth propagation.

    Endpoint gates keep the existing surrogate derivative. Padded nodes have
    no permission or gradient, while all continuous geometry stays live.
    """
    if queries.ndim != 3 or gates.ndim != 2 or queries.shape[0] != gates.shape[0]:
        raise ValueError("Receiver queries and split gates must have shape[B,R,d] and[B,K]")
    if queries.shape[-1] != data["spatial_dim"] or gates.shape != data["axes"].shape:
        raise ValueError("Receiver/gate axes differ from prepared receiver trees")
    if queries.device != gates.device or queries.device != data["boundary"].device:
        raise ValueError("Receiver queries, gates and anchors must use one device")
    if bool((data["valid"] & (~torch.isfinite(gates) | (gates < 0) | (gates > 1))).any()):
        raise ValueError("Split gates must be finite and in[0,1]")
    if soft_precision:
        gates = gates.to(torch.float64)
    gates = torch.where(data["valid"], gates, torch.zeros_like(gates))
    batch, receivers, _ = queries.shape
    capacity = gates.shape[1]
    smooth = endpoint_smoothstep(gates)
    if gates.requires_grad:
        endpoint = (gates.detach() == 0) | (gates.detach() == 1)
        surrogate = torch.where(endpoint, gates, smooth)
        smooth = surrogate + (smooth - surrogate).detach()
    axes = data["axes"][:, None, :, None].expand(-1, receivers, -1, -1)
    if soft_precision:
        # Geometry factors and recursive products stay wide in the shadow;
        # the hard path preserves the original mixed-dtype arithmetic.
        permission_queries = queries.to(torch.float64)
        boundary = data["boundary"].to(torch.float64)
        overlap = data["overlap"].to(torch.float64)
        along = permission_queries[:, :, None].expand(-1, -1, capacity, -1).gather(-1, axes).squeeze(-1)
        weight = endpoint_smoothstep(
            (boundary[:, None] + overlap[:, None] / 2 - along) / overlap[:, None]
        )
    else:
        along = queries[:, :, None].expand(-1, -1, capacity, -1).gather(-1, axes).squeeze(-1)
        weight = endpoint_smoothstep(
            (data["boundary"][:, None] + data["overlap"][:, None] / 2 - along) / data["overlap"][:, None]
        )
    parent = data["parents"][:, None].expand(-1, receivers, -1)
    parent_gate = smooth.gather(1, data["parents"])[:, None]
    parent_left = weight.gather(2, parent)
    branch = torch.where(data["left"][:, None], parent_left, 1 - parent_left)
    incoming_source = permission_queries if soft_precision else queries
    incoming = incoming_source.new_tensor([1.0] + [0.0] * (capacity - 1))[None, None].expand(batch, receivers, -1)
    for depth in range(1, max_depth + 1):
        contribution = incoming.gather(2, parent) * parent_gate * branch
        incoming = torch.where((data["depths"] == depth)[:, None], contribution, incoming)
    result = torch.where(data["leaves"][:, None], incoming, incoming * (1 - smooth[:, None]))
    return torch.where(data["valid"][:, None], result, torch.zeros_like(result))

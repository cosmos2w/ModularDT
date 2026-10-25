"""Local-grid tree nomination and reversible keyed E frontiers.

Only edge-local distances are evaluated. Tree storage is linear: children and
additive node statistics, followed by one traversal to label active ancestry.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GroupFrontier:
    nodes: tuple
    children: tuple
    labels: torch.Tensor
    log_probability: torch.Tensor
    reached: tuple
    probabilities: torch.Tensor
    edge_count: int
    policy_rows: int

    @property
    def log_prob(self):
        return self.log_probability

    @property
    def K(self):
        return len(self.nodes)


def grid_edges(nx, ny):
    return tuple((y * nx + x, y * nx + x + 1) for y in range(ny) for x in range(nx - 1)) + tuple(
        (y * nx + x, (y + 1) * nx + x) for y in range(ny - 1) for x in range(nx)
    )


class LocalFrontier(nn.Module):
    def __init__(self, width, nx, ny):
        super().__init__()
        self.nx, self.ny = nx, ny
        self.projection = nn.Linear(width, 16, bias=False)
        self.policy = nn.Sequential(nn.Linear(16 + 4 + 8, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.policy[-1].bias)
        self.register_buffer("distance_scales", torch.ones(3))
        self.register_buffer("scales_calibrated", torch.tensor(False))
        self.register_buffer("edges", torch.tensor(grid_edges(nx, ny), dtype=torch.long))

    @staticmethod
    def schedule(epoch):
        t = min(1.0, max(0.0, (float(epoch) - 1.0) / 499.0))
        smooth = 0.5 - 0.5 * math.cos(math.pi * t)
        return 2.0 - smooth, smooth

    def forward(self, states, relation, coordinates, mass, *, adaptive, training, epoch, seed, forced=None):
        n = len(states)
        if not adaptive:
            return GroupFrontier(
                tuple(range(n)),
                (),
                torch.arange(n, device=states.device),
                states.sum() * 0.0,
                (),
                states.new_empty(0),
                0,
                0,
            )
        if n != self.nx * self.ny:
            raise ValueError("Adaptive frontier requires the complete declared regular E catalogue.")
        z = self.projection(states)
        r = torch.cat((relation.permute(1, 0, 2).flatten(1), (1 - relation.sum(-1)).T), -1)
        a, b = self.edges.T
        dx = (coordinates[a] - coordinates[b]).square().sum(-1)
        dr = (r[a] - r[b]).square().mean(-1)
        dz = (z[a] - z[b]).square().mean(-1)
        components = torch.stack((dr, dz, dx), -1)
        # Calibrated on an input-selected TRAIN panel before launch; zero
        # variance components are explicitly ignored.
        scales = self.distance_scales.to(components)
        costs = (components / scales.clamp_min(1.0e-12) * (scales > 0)).sum(-1)
        edges = self.edges.cpu().tolist()
        host_costs = costs.detach().cpu().tolist()
        order = sorted(range(len(edges)), key=lambda j: (host_costs[j], *edges[j]))
        parent = list(range(n))
        root = list(range(n))
        children = [None] * n

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for j in order:
            a, b = edges[j]
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            ca, cb = root[ra], root[rb]
            if ca > cb:
                ca, cb = cb, ca
            children.append((ca, cb))
            parent[rb] = ra
            root[ra] = len(children) - 1
        # Linear additive summaries, never node-by-original memberships.
        sums = [z[i] * mass[i] for i in range(n)]
        masses = [mass[i] for i in range(n)]
        lows = [coordinates[i] for i in range(n)]
        highs = list(lows)
        counts = [1] * n
        discrepancy = [z.new_zeros(())] * n
        for a, b in children[n:]:
            sums.append(sums[a] + sums[b])
            masses.append(masses[a] + masses[b])
            lows.append(torch.minimum(lows[a], lows[b]))
            highs.append(torch.maximum(highs[a], highs[b]))
            counts.append(counts[a] + counts[b])
            discrepancy.append(
                (sums[a] / masses[a].clamp_min(1.0e-12) - sums[b] / masses[b].clamp_min(1.0e-12)).square().mean()
            )
        means = torch.stack(sums) / torch.stack(masses).clamp_min(1.0e-12)[:, None]
        lo, hi = torch.stack(lows), torch.stack(highs)
        extras = torch.stack(
            (
                torch.stack(masses),
                z.new_tensor(counts) / n,
                torch.stack(discrepancy),
                z.new_tensor([0 if c is None else 1 for c in children]),
            ),
            -1,
        )
        relation_summary = torch.stack(
            (r.mean(-1), r.square().mean(-1), relation.sum(-1).mean(0), relation.square().sum(-1).mean(0)), -1
        )
        relation_sums = [relation_summary[i] * mass[i] for i in range(n)]
        for a, b in children[n:]:
            relation_sums.append(relation_sums[a] + relation_sums[b])
        relation_means = torch.stack(relation_sums) / torch.stack(masses).clamp_min(1.0e-12)[:, None]
        descriptors = torch.cat((means, lo, hi, extras, relation_means), -1)
        logits = self.policy(descriptors[n:]).flatten()
        temperature, _ = self.schedule(epoch)
        probabilities = torch.sigmoid(logits / temperature)
        # NumPy SeedSequence is ordinary keyed sampling, independent of tiles
        # and global CUDA RNG. No cryptographic identity machinery is needed.
        rng = np.random.default_rng(np.random.SeedSequence([int(v) & 0xFFFFFFFF for v in seed]))
        host_probabilities = probabilities.detach().cpu().tolist()
        host_logits = logits.detach().cpu().tolist()
        active = []
        reached = []
        terms = []
        stack = [len(children) - 1]
        while stack:
            node = stack.pop()
            if children[node] is None:
                active.append(node)
                continue
            j = node - n
            reached.append(node)
            stop = bool(rng.random() < host_probabilities[j]) if training else host_logits[j] >= 0
            if forced is not None and node == forced[0]:
                stop = forced[1] == "parent"
            if training:
                terms.append(-F.softplus(-logits[j] / temperature) if stop else -F.softplus(logits[j] / temperature))
            if stop:
                active.append(node)
            else:
                stack.extend(reversed(children[node]))
        labels = torch.empty(n, dtype=torch.long, device=states.device)
        for group, node in enumerate(active):
            stack = [node]
            while stack:
                v = stack.pop()
                if children[v] is None:
                    labels[v] = group
                else:
                    stack.extend(children[v])
        logp = torch.stack(terms).sum() if terms else states.sum() * 0.0
        return GroupFrontier(
            tuple(active), tuple(children), labels, logp, tuple(reached), probabilities, len(edges), len(logits)
        )

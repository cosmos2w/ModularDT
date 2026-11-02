"""Detached co-service proposals for frozen-model layout optimization.

The caller owns objective evaluation, physical feasibility, history and plotting.
One call to ``step`` consumes the already evaluated loss, with no field searches.
"""
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class Graph:
    affinity: Tensor
    laplacian: Tensor
    transform: Tensor


def _graph(affinity: Tensor) -> Graph:
    s = affinity.detach().clone()
    s.fill_diagonal_(0)
    degree = s.sum(dim=1)
    inv_sqrt = torch.zeros_like(degree)
    positive = degree > 0
    inv_sqrt[positive] = degree[positive].rsqrt()
    normalized = inv_sqrt[:, None] * s * inv_sqrt[None, :]
    laplacian = torch.diag(positive.to(s.dtype)) - normalized
    identity = torch.eye(s.shape[0], dtype=s.dtype, device=s.device)
    return Graph(s, laplacian, torch.linalg.inv(identity + laplacian))


def learned_graph(B: Tensor, mu_e: Tensor) -> Graph:
    """Co-service from executed full Pi columns for movable original members."""
    B, mu_e = B.detach(), mu_e.detach().reshape(-1)
    if B.ndim != 2 or B.shape[0] != mu_e.numel():
        raise ValueError("B must have shape (E, movable M), matching mu_e")
    if not torch.isfinite(B).all() or not torch.isfinite(mu_e).all():
        raise ValueError("graph inputs must be finite")
    if (B < 0).any() or (mu_e < 0).any() or not (mu_e > 0).any():
        raise ValueError("membership and physical measures must be nonnegative with positive measure")
    weights = mu_e / mu_e.sum()
    return _graph(B.T @ (weights[:, None] * B))


def start_bandwidth(xy: Tensor) -> Tensor:
    """Median unordered pair distance at the start, fixed for the trail."""
    d = torch.pdist(xy.detach())
    if d.numel() == 0:
        raise ValueError("at least two movable members are required")
    bandwidth = torch.quantile(d, 0.5)
    if not torch.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("start median distance must be positive and finite")
    return bandwidth


def geometry_graph(xy: Tensor, bandwidth: Tensor | float) -> Graph:
    xy = xy.detach()
    scale = torch.as_tensor(bandwidth, dtype=xy.dtype, device=xy.device)
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError("bandwidth must be positive and finite")
    distance2 = torch.cdist(xy, xy).square()
    return _graph(torch.exp(-distance2 / (2 * scale.square())))


def _edge_contrast(graph: Graph) -> float:
    n = graph.affinity.shape[0]
    edges = graph.affinity[torch.triu_indices(n, n, 1, device=graph.affinity.device).unbind()]
    mean = edges.mean()
    return float((edges.std(unbiased=False) / mean).item()) if mean > 0 else 0.0


class ProposalAdam:
    """Ordinary Adam followed by detached graph mixing and common projection.

    ``xy`` is cloned into the leaf parameter ``self.xy``; evaluate losses using
    that parameter. ``project(old, proposed)`` returns (coordinates, metadata).
    The latter may include case-specific feasibility and rejection information.
    """
    def __init__(self, xy: Tensor, lr: float, arm: str):
        if arm not in {"A0", "A1", "A2", "A3"}:
            raise ValueError("arm must be A0, A1, A2 or A3")
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must have shape (movable members, 2)")
        self.xy = torch.nn.Parameter(xy.detach().clone())
        self.optimizer = torch.optim.Adam([self.xy], lr=lr)
        self.arm = arm
        self.bandwidth = start_bandwidth(self.xy)
        self.epoch = 0
        self.graph = None
        self.graph_construction_seconds = 0.0
        self._graph_diagnostics = {}

    def step(self, loss: Tensor, project: Callable, learned_B: Tensor | None = None,
             mu_e: Tensor | None = None) -> dict:
        t0 = perf_counter()
        refresh = self.epoch == 0 or (self.arm in {"A2", "A3"} and self.epoch % 10 == 0)
        graph_seconds = 0.0
        if refresh and self.arm != "A0":
            if self.xy.is_cuda:
                torch.cuda.synchronize(self.xy.device)
            start = perf_counter()
            geometry = geometry_graph(self.xy, self.bandwidth)
            learned = None
            if learned_B is not None and mu_e is not None:
                learned = learned_graph(learned_B, mu_e)
            if self.arm in {"A1", "A2"} and learned is None:
                raise ValueError("learned arms need executed full Pi and physical E measures at refresh")
            self.graph = geometry if self.arm == "A3" else learned
            if self.xy.is_cuda:
                torch.cuda.synchronize(self.xy.device)
            graph_seconds = perf_counter() - start
            self.graph_construction_seconds += graph_seconds
            self._graph_diagnostics = {
                "graph_affinity": self.graph.affinity.cpu().tolist(),
                "graph_laplacian": self.graph.laplacian.cpu().tolist(),
                "graph_spectrum": torch.linalg.eigvalsh(self.graph.laplacian).cpu().tolist(),
                "graph_edge_contrast": _edge_contrast(self.graph),
                "learned_geometry_laplacian_difference": (
                    float(torch.linalg.matrix_norm(learned.laplacian - geometry.laplacian).item())
                    if learned is not None else None),
            }
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.xy.grad is None:
            raise ValueError("loss has no gradient with respect to design")
        gradient = self.xy.grad.detach().clone()
        old = self.xy.detach().clone()
        self.optimizer.step()
        with torch.no_grad():
            ordinary = self.xy.detach() - old
            proposal = ordinary
            if self.graph is not None:
                mixed = self.graph.transform @ ordinary
                ordinary_rms, mixed_rms = ordinary.square().mean().sqrt(), mixed.square().mean().sqrt()
                proposal = mixed * (ordinary_rms / mixed_rms) if mixed_rms > 0 else mixed
            accepted, metadata = project(old.clone(), old + proposal)
            accepted = accepted.detach()
            if accepted.shape != old.shape or not torch.isfinite(accepted).all():
                raise ValueError("projection must return finite coordinates of the same shape")
            self.xy.copy_(accepted)
            realized = accepted - old
        if self.xy.is_cuda:
            torch.cuda.synchronize(self.xy.device)
        self.epoch += 1
        return {
            "epoch": self.epoch, "objective": float(loss.detach().item()),
            "objective_evaluations": self.epoch, "gradient_evaluations": self.epoch,
            "coordinates_before": old.cpu().tolist(), "coordinates_after": accepted.cpu().tolist(),
            "ordinary_displacement": ordinary.cpu().tolist(), "proposed_displacement": proposal.cpu().tolist(),
            "realized_displacement": realized.cpu().tolist(),
            "gradient_dot_realized": float((gradient * realized).sum().item()),
            "movement_rms": float(realized.square().mean().sqrt().item()),
            "proposal_change_rms": float((proposal - ordinary).square().mean().sqrt().item()),
            "graph_refreshed": bool(refresh and self.arm != "A0"),
            "graph_construction_seconds": graph_seconds,
            "graph_construction_seconds_total": self.graph_construction_seconds,
            "step_seconds": perf_counter() - t0,
            "projection": metadata, **self._graph_diagnostics,
        }

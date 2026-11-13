"""Bounded request-guided search primitives for frozen Thermal scorers.

All geometry in this module stays in the precision of the physical input.  The
caller owns native rebuilding, model inference, validation and persistence.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Mapping

import numpy as np

SEED = 20261112
ATTEMPTS = 16


def roi_mask(points, lengths, centers, present, radius, request):
    p = np.asarray(points, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    c = np.asarray(centers, dtype=np.float64)[np.asarray(present, dtype=bool)]
    bounds = {'A': (0.10, 0.45), 'B': (0.55, 0.90)}[request]
    norm = p / lengths
    inside = ((norm[:, 0] >= .25) & (norm[:, 0] <= .90) &
              (norm[:, 1] >= bounds[0]) & (norm[:, 1] <= bounds[1]))
    if c.size:
        inside &= (np.linalg.norm(p[:, None, :] - c[None, :, :], axis=-1) > 1.15 * radius).all(axis=1)
    if not inside.any():
        raise ValueError('empty fixed safe-fluid ROI')
    return inside


def pullback_weights(indices, stencil_weights, valid, receiver_weights, nqueries):
    """Push normalized physical receiver weights into the deduplicated union."""
    idx = np.asarray(indices, dtype=np.int64)
    s = np.asarray(stencil_weights, dtype=np.float64)
    v = np.asarray(valid, dtype=bool)
    w = np.asarray(receiver_weights, dtype=np.float64)
    if idx.shape != s.shape or idx.ndim != 2 or v.shape != idx.shape[:1] or w.shape != v.shape:
        raise ValueError('inconsistent stencil axes')
    if np.any(idx < 0) or np.any(idx >= nqueries) or np.any(s < 0) or np.any(w < 0) or not np.isfinite(s).all() or not np.isfinite(w).all():
        raise ValueError('invalid stencil or receiver weights')
    if np.any(w[~v] != 0) or not np.isclose(w.sum(), 1) or not np.allclose(s[v].sum(axis=1), 1):
        raise ValueError('receiver and interpolation weights must be normalized')
    out = np.zeros(nqueries, dtype=np.float64)
    np.add.at(out, idx.ravel(), (s * w[:, None]).ravel())
    return out


def routing_scores(query_weights, traces, membership):
    """Accumulate real tile traces; indices are global positions in the union.

    Each tuple is (global indices, coalesced beta_M, applied address weights).
    Membership contains actual joint address rows and original M columns.
    """
    w = np.asarray(query_weights, dtype=np.float64)
    D = np.asarray(membership, dtype=np.float64)
    if w.ndim != 1 or not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1):
        raise ValueError('request query weights must form a normalized physical measure')
    if D.ndim != 2 or (D < 0).any() or not np.isfinite(D).all():
        raise ValueError('invalid joint membership')
    seen = np.zeros(len(w), dtype=bool)
    service = np.zeros(D.shape[1], dtype=np.float64)
    address = np.zeros(D.shape[0], dtype=np.float64)
    for ids, beta, lam in traces:
        ids = np.asarray(ids, dtype=np.int64)
        beta, lam = np.asarray(beta, dtype=np.float64), np.asarray(lam, dtype=np.float64)
        if ids.ndim != 1 or beta.shape != (len(ids), D.shape[1]) or lam.shape != (len(ids), D.shape[0]):
            raise ValueError('trace axes do not match query and joint membership')
        if (ids < 0).any() or (ids >= len(w)).any() or len(np.unique(ids)) != len(ids) or seen[ids].any():
            raise ValueError('trace indices must be unique global query indices')
        if (beta < 0).any() or (lam < 0).any() or not np.isfinite(beta).all() or not np.isfinite(lam).all():
            raise ValueError('invalid recorded routing weights')
        seen[ids] = True
        service += w[ids] @ beta
        address += w[ids] @ lam
    if not seen[w > 0].all():
        raise ValueError('ROI queries missing from trace')
    return service, D.T @ (address[:, None] * D), address


def block_probabilities(member_ids, service, affinity):
    ids = tuple(int(i) for i in member_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('editable members must be distinct and nonempty')
    s, h = np.asarray(service, dtype=np.float64), np.asarray(affinity, dtype=np.float64)
    if s.shape != (len(ids),) or h.shape != (len(ids), len(ids)) or (s < 0).any() or (h < 0).any() or not np.isfinite(s).all() or not np.isfinite(h).all():
        raise ValueError('invalid block scores')
    s = s / s.sum() if s.sum() else np.zeros_like(s)
    off = h.copy(); np.fill_diagonal(off, 0)
    h = h / off.max() if off.max() else np.zeros_like(h)
    result = {}
    for size in (1, 2):
        blocks = list(combinations(range(len(ids)), min(size, len(ids))))
        raw = np.array([s[b[0]] if len(b) == 1 else (s[b[0]] + s[b[1]]) * (1 + h[b[0], b[1]]) for b in blocks])
        guided = raw / raw.sum() if raw.sum() else np.full(len(raw), 1 / len(raw))
        result[size] = (tuple(tuple(ids[i] for i in b) for b in blocks), .8 * guided + .2 / len(blocks), raw)
    return result


def geometry_scores(centers, radius, lengths, request):
    c = np.asarray(centers, dtype=np.float64)
    L = np.asarray(lengths, dtype=np.float64)
    y = {'A': (.10, .45), 'B': (.55, .90)}[request]
    lo, hi = L * [.25, y[0]], L * [.90, y[1]]
    d = np.linalg.norm(np.maximum(np.maximum(lo - c, c - hi), 0), axis=1)
    ell = 2.15 * radius
    return np.exp(-d*d/(2*ell*ell)), np.exp(-np.sum((c[:, None] - c[None, :])**2, axis=-1)/(2*ell*ell))


def propose(structure: Mapping, editable_ids, probabilities, radius, layout_index, request_index, attempt, valid: Callable):
    """Attempt using original array slots; caller retains their original IDs.

    Pass the source-precision physical configuration, not the FP32 neural
    structure. Invalid attempts are retained and never redrawn.
    """
    if not 0 <= attempt < ATTEMPTS:
        raise ValueError('attempt outside fixed budget')
    ids = tuple(int(i) for i in editable_ids)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, layout_index, request_index, attempt]))
    size = 1 + int(rng.random() >= .5)
    u = rng.random()
    # Draw for every original member, keeping selected members aligned across policies.
    angle = rng.random(len(ids)) * (2 * np.pi)
    magnitude = np.sqrt(rng.random(len(ids))) * (.15 * radius)
    offsets = np.column_stack((np.cos(angle), np.sin(angle))) * magnitude[:, None]
    blocks, p, _ = probabilities[size]
    block = blocks[min(np.searchsorted(np.cumsum(p), u, side='right'), len(blocks)-1)]
    result = deepcopy(structure)
    centers = np.array(structure['module_centers'], copy=True)
    if centers.dtype != np.float64:
        raise ValueError('physical centers must retain FP64 source precision')
    for member in block:
        centers[member] += offsets[ids.index(member)]
    result['module_centers'] = centers
    return {'attempt': attempt, 'block': block, 'structure': result, 'valid': bool(valid(result))}


def select_predicted(baseline_f, predictions, pressure_allowance):
    """Candidate records are ordered, with ties resolved in favour of earlier entries."""
    best, best_f = None, float(baseline_f)
    for item in predictions:
        if item['valid'] and np.isfinite(item['f']) and np.isfinite(item['delta_p']) and item['delta_p'] <= pressure_allowance and item['f'] < best_f:
            best, best_f = item, item['f']
    return best


def physical_accept(baseline_f, f, delta_p, pressure_allowance, *, solved=True):
    if not solved or not np.isfinite(f) or not np.isfinite(delta_p):
        return 'unresolved'
    return 'finalist' if delta_p <= pressure_allowance and f < baseline_f else 'baseline'


def cross_swap(f_a_from_b, f_a_from_a, f_b_from_a, f_b_from_b, scale_a, scale_b):
    if scale_a <= 0 or scale_b <= 0:
        raise ValueError('ROI scales must be positive')
    return .5 * ((f_a_from_b-f_a_from_a)/scale_a + (f_b_from_a-f_b_from_b)/scale_b)


@dataclass
class SolveBudget:
    """Deduplicate identities and charge every start, including failures.

    The caller must enforce the remaining worker-time budget as a hard external
    solver timeout; this ledger can only account for elapsed time after a call.
    """
    max_starts: int = 54
    max_worker_hours: float = 6.0
    starts: int = 0
    worker_hours: float = 0.0

    def __post_init__(self):
        if not 0 <= self.max_starts <= 54 or not 0 < self.max_worker_hours <= 6:
            raise ValueError("budget exceeds the declared ceiling")
        self.results = {}
        self._finished = set()

    def start(self, key):
        if key in self.results:
            return False
        if self.starts >= self.max_starts or self.worker_hours >= self.max_worker_hours:
            raise RuntimeError('physical budget exhausted')
        self.starts += 1
        self.results[key] = None
        return True

    def finish(self, key, result, worker_hours):
        if key not in self.results or key in self._finished or not np.isfinite(worker_hours) or worker_hours < 0:
            raise ValueError('unknown or already finished solve')
        self.worker_hours += worker_hours
        self.results[key] = result
        self._finished.add(key)


def capture_routing_scores(model, provider, prepared, query_weights, *, strict_binding=False, retain_tiles=False):
    """Trace the actual P95 E-address directory in bounded union slices."""
    import torch
    from honf_forward_core.content_hypergraph.query_chunks_v1 import slice_query_v1
    w = np.asarray(query_weights, dtype=np.float64)
    query = prepared.query_batch
    if query.coordinates.shape[0] != 1 or len(w) != query.num_queries:
        raise ValueError('one baseline and its complete union are required')
    tile = int(model.core.query_tile_size)
    if tile < 1:
        raise ValueError('query tile must be positive')
    org = prepared.organization
    module_ids = org.scene.module_ids[0].detach().cpu().numpy()
    environment_ids = org.scene.environment_ids[0].detach().cpu().numpy()
    provenance = {route: {'stage_identity': str(info['stage_identity']),
                          'route': str(info['route']),
                          'receiver_role': str(info['receiver_role']),
                          'donor_role': str(info['donor_role'])}
                  for route, info in org.anchored_route_provenance.items()}
    if set(provenance) != {'em', 'ee'}:
        raise ValueError('missing actual anchored EM/EE stage provenance')
    if strict_binding:
        for route, donor in (('em', 'M'), ('ee', 'E')):
            info = provenance[route]
            if (info['route'] != route or info['receiver_role'] != 'E' or info['donor_role'] != donor
                    or not info['stage_identity'].endswith('/' + route)):
                raise ValueError('invalid applied route provenance')
        if provenance['em']['stage_identity'].rsplit('/', 1)[0] != provenance['ee']['stage_identity'].rsplit('/', 1)[0]:
            raise ValueError('applied EM and EE do not share the final contextual stage')
    directory = org.e_access_directory.restricted
    m, e = len(module_ids), len(environment_ids)
    if strict_binding and not np.isclose(org.e_access_directory.retained_mass, .95):
        raise ValueError('request search requires the prepared P95 directory')
    membership = np.zeros((e, m + e), dtype=np.float64)
    for anchor in range(e):
        row = directory.row(0, anchor)
        indices = row.member_index.detach().cpu().numpy()
        weights = row.access_weight.detach().cpu().numpy()
        if (indices < 0).any() or (indices >= m + e).any():
            raise ValueError('directory refers to a missing original member')
        if strict_binding:
            actual_ids = row.original_id.detach().cpu().numpy()
            actual_roles = row.is_module.detach().cpu().numpy()
            if (not np.array_equal(actual_ids, np.concatenate((module_ids, environment_ids))[indices])
                    or not np.array_equal(actual_roles, indices < m)):
                raise ValueError('directory member identities or roles are inconsistent')
        np.add.at(membership[anchor], indices, weights)
    slices = []
    recorded_tiles = []
    trace_pairs = 0
    with torch.no_grad(), model.core.read_execution('restricted'):
        for start in range(0, len(w), tile):
            stop = min(start + tile, len(w))
            if not np.any(w[start:stop] > 0):
                continue
            part = slice_query_v1(query, start, stop)
            trace = model.core.capture_interaction_routing_trace(org, part)
            if 'anchor_read_member_index' in trace:
                trace_pairs += int(trace['anchor_read_member_index'].numel())
            elif strict_binding:
                raise ValueError('bounded trace is missing its executed member-pair count')
            if strict_binding:
                expected = {'original_module_ids': org.scene.module_ids,
                            'original_environment_ids': org.scene.environment_ids,
                            'anchored_applied_EM': org.anchored_applied_EM,
                            'anchored_applied_EE': org.anchored_applied_EE,
                            'e_access_em_destination_ids': org.scene.environment_ids,
                            'e_access_em_donor_ids': org.scene.module_ids,
                            'e_access_ee_destination_ids': org.scene.environment_ids,
                            'e_access_ee_donor_ids': org.scene.environment_ids}
                if any(key not in trace or not torch.equal(trace[key], value) for key, value in expected.items()):
                    raise ValueError('bounded trace is not bound to the applied stage and original IDs')
                if (not torch.equal(trace['anchor_indices'], part.anchor_indices)
                        or not torch.equal(trace['anchor_weights'], part.anchor_weights)):
                    raise ValueError('bounded trace address coefficients differ from the executed query')
            beta = trace['member_weights_M'][0].detach().cpu().numpy()
            anchors = trace['anchor_indices'][0].detach().cpu().numpy()
            weights = trace['anchor_weights'][0].detach().cpu().numpy()
            address = np.zeros((stop-start, e), dtype=np.float64)
            rows = np.broadcast_to(np.arange(stop-start)[:, None], anchors.shape)
            positive = weights > 0
            np.add.at(address, (rows[positive], anchors[positive]), weights[positive])
            slices.append((np.arange(start, stop), beta, address))
            if retain_tiles:
                recorded_tiles.append({'query_indices': np.arange(start, stop),
                                       'member_coefficients_M': beta,
                                       'anchor_indices': anchors, 'anchor_weights': weights})
    service, overlap, address = routing_scores(w, slices, membership[:, :m])
    return {'service': service, 'overlap': overlap, 'address_mass': address,
            'joint_membership': membership, 'module_ids': module_ids,
            'environment_ids': environment_ids, 'query_indices': np.flatnonzero(w > 0),
            'traced_tiles': len(slices), 'traced_query_rows': sum(len(ids) for ids, _, _ in slices),
            'traced_member_pairs': trace_pairs,
            'applied_em': org.anchored_applied_EM[0].detach().cpu().numpy(),
            'applied_ee': org.anchored_applied_EE[0].detach().cpu().numpy(),
            'route_provenance': provenance, **({'trace_tiles': recorded_tiles} if retain_tiles else {})}

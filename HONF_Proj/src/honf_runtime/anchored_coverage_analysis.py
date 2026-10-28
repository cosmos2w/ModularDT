"""Offline analysis of saved receiver-first access rows, without fitting."""
from __future__ import annotations

import numpy as np


def regional_mass_geometry(weights, receiver_xy, donor_xy, donor_measure, own_indices,
                           fractions=(0.90, 0.95, 0.99)):
    """Describe nonself environmental mass, area and reach in physical coordinates.

    ``weights`` are actual nonnegative row weights (e.g. c, R or admitted C),
    not a positivity mask. The own donor is excluded before each row is
    normalized. Each prefix contains the smallest number of donors reaching
    the requested fraction, with index as a deterministic tie breaker. Area
    is the sum of the original physical donor measures; area_fraction uses all
    valid environment measure, including own. Empty nonself rows have null
    area/reach and are counted explicitly. No geometric cutoff is imposed.
    """
    w = np.asarray(weights, dtype=np.float64)
    r = np.asarray(receiver_xy, dtype=np.float64)
    d = np.asarray(donor_xy, dtype=np.float64)
    area = np.asarray(donor_measure, dtype=np.float64)
    own = np.asarray(own_indices)
    if w.ndim != 2 or r.ndim != 2 or d.ndim != 2 or r.shape[1] != d.shape[1]:
        raise ValueError('Expected weights [receiver,donor] and aligned coordinate matrices.')
    if r.shape[0] != w.shape[0] or d.shape[0] != w.shape[1] or area.shape != (w.shape[1],) or own.shape != (w.shape[0],):
        raise ValueError('Coordinates, measures and own indices must match the weight axes.')
    if any(not np.isfinite(x).all() for x in (w, r, d, area)) or (w < 0).any() or (area < 0).any():
        raise ValueError('Weights, coordinates and measures must be finite and nonnegative where applicable.')
    if not np.issubdtype(own.dtype, np.integer) or np.any((own < 0) | (own >= w.shape[1])):
        raise ValueError('Each receiver needs a valid own donor index.')
    fractions = tuple(float(f) for f in fractions)
    if any(not 0 < f <= 1 for f in fractions):
        raise ValueError('Mass fractions must be in (0,1].')
    x = w.copy()
    x[np.arange(len(own)), own] = 0
    x[:, area <= 0] = 0
    total = x.sum(axis=1)
    normalized = np.divide(x, total[:, None], out=np.zeros_like(x), where=total[:, None] > 0)
    result = {'nonself_mass': total.tolist(), 'zero_nonself_rows': int((total == 0).sum()),
              'positive_nonself_count': (x > 0).sum(axis=1).tolist(), 'prefixes': {}}
    for fraction in fractions:
        counts, areas, shares, reaches = [], [], [], []
        for i, row in enumerate(normalized):
            positive = np.flatnonzero(row > 0)
            if not len(positive):
                counts.append(0); areas.append(None); shares.append(None); reaches.append(None)
                continue
            order = positive[np.lexsort((positive, -row[positive]))]
            n = min(len(order), int(np.searchsorted(np.cumsum(row[order]), fraction, side='left')) + 1)
            selected = order[:n]
            covered = float(area[selected].sum())
            counts.append(n); areas.append(covered)
            shares.append(covered / float(area.sum()) if area.sum() > 0 else None)
            reaches.append(float(np.linalg.norm(d[selected] - r[i], axis=1).max()))
        result['prefixes'][str(fraction)] = {'count': counts, 'area': areas,
                                               'area_fraction': shares, 'max_reach': reaches}
    # Compare each receiver to its nearest distinct receiver. Weighted Jaccard
    # is on normalized nonself distributions, not raw positive support.
    nearest, overlap = [], []
    for i in range(len(r)):
        distances = np.linalg.norm(r - r[i], axis=1)
        distances[i] = np.inf
        if len(r) < 2:
            nearest.append(None); overlap.append(None); continue
        j = int(np.argmin(distances)); nearest.append(j)
        union = np.maximum(normalized[i], normalized[j]).sum()
        overlap.append(float(np.minimum(normalized[i], normalized[j]).sum() / union) if union > 0 else None)
    result['nearest_receiver_index'] = nearest
    result['nearest_weighted_jaccard'] = overlap
    return result


def coverage_support(access, own_indices, member_roles, member_ids, retained_mass):
    """Reconstruct the whole-row prefix and mandatory self from saved R.

    Rows contain the normalized joint M/E access weights; role code 0 orders
    before 1, then original ID breaks ties. Returns boolean support and attained
    mass. This is an analysis of saved rows, not an execution path.
    """
    access = np.asarray(access)
    own_indices = np.asarray(own_indices)
    roles = np.asarray(member_roles)
    ids = np.asarray(member_ids)
    if access.ndim != 2 or own_indices.shape != (access.shape[0],):
        raise ValueError('Expected access [receivers,members] and one own index per receiver.')
    if roles.shape != (access.shape[1],) or ids.shape != roles.shape:
        raise ValueError('Roles and original IDs must match the member axis.')
    if not 0 < retained_mass <= 1 or not np.isfinite(access).all() or np.any(access < 0):
        raise ValueError('Coverage and access weights must be finite and nonnegative.')
    out = np.zeros(access.shape, dtype=bool)
    for row, weights in enumerate(access):
        if not np.any(weights > 0):
            continue
        own = int(own_indices[row])
        if own < 0 or own >= weights.size or weights[own] <= 0:
            raise ValueError('Active rows require positive own-E access.')
        order = sorted(np.flatnonzero(weights > 0), key=lambda i: (-float(weights[i]), roles[i], ids[i]))
        cumulative = np.cumsum(weights[order], dtype=np.float64)
        count = min(len(order), int(np.searchsorted(cumulative, retained_mass, side='left')) + 1)
        out[row, order[:count]] = True
        out[row, own] = True
    return out, np.sum(np.where(out, access, 0), axis=1)


def learned_access_weights(access, applied_relations, structural_self):
    """Remove only S's contribution from saved access R proportional to (G+S)mu.

    The original row normalization and compiled support stay fixed. In an own-E
    slot shared by learned G and structural S, R*G/(G+S) retains the learned
    part; deleting that entire slot would also delete learned self. This is an
    incidence accounting operation, not recompilation or a prediction ablation.
    It applies to joint access weights, not nonlinear query-used beta.
    """
    access = np.asarray(access, dtype=np.float64)
    applied_relations = np.asarray(applied_relations, dtype=np.float64)
    structural_self = np.asarray(structural_self, dtype=np.float64)
    if access.ndim != 2 or not (access.shape == applied_relations.shape == structural_self.shape):
        raise ValueError('Access, applied relations and structural ledger must share [receiver, member] shape.')
    if any(not np.isfinite(x).all() or np.any(x < 0) for x in (access, applied_relations, structural_self)):
        raise ValueError('Access, applied relations and structural ledger must be finite and nonnegative.')
    total = applied_relations + structural_self
    fraction = np.divide(applied_relations, total, out=np.zeros_like(total), where=total > 0)
    return access * fraction


def incidence_summary(weights, structural_self, *, service=None):
    """Describe receiver/member incidence without equating self with learned links.

    Weights are nonnegative joint access or actual accumulated query weights.
    The mask marks own-E slots: ``without_self`` removes the whole coalesced
    incidence, including any learned self. To exclude only the structural
    contribution, first use ``learned_access_weights`` on saved joint access.
    Pairwise overlap is calculated over distinct nonempty receiver rows. The
    active bipartite graph includes every receiver and member as a vertex.
    Service-weighted graph statistics retain only actually served receiver rows.
    """
    weights = np.asarray(weights, dtype=np.float64)
    structural_self = np.asarray(structural_self, dtype=bool)
    if weights.ndim != 2 or weights.shape != structural_self.shape:
        raise ValueError('Weights and structural-self masks must have matching [receiver, member] shapes.')
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('Incidence weights must be finite and nonnegative.')
    nrows, nmembers = weights.shape
    if service is None:
        service = np.ones(nrows, dtype=np.float64)
    service = np.asarray(service, dtype=np.float64)
    if service.shape != (nrows,) or not np.isfinite(service).all() or (service < 0).any():
        raise ValueError('Service must have one finite nonnegative weight per receiver.')

    def graph(mask):
        parent = list(range(nrows + nmembers))
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for row, member in zip(*np.nonzero(mask)):
            parent[root(row)] = root(nrows + member)
        active = np.r_[mask.any(axis=1), mask.any(axis=0)]
        components = {}
        for i in np.flatnonzero(active):
            representative = root(int(i))
            components[representative] = components.get(representative, 0) + 1
        return {'active_components': len(components), 'largest_active_component_vertices': max(components.values(), default=0),
                'isolated_receivers': int((~active[:nrows]).sum()), 'isolated_members': int((~active[nrows:]).sum())}

    def overlap(array, row_weights):
        supports = array > 0
        intersection = supports.astype(np.int64) @ supports.astype(np.int64).T
        cardinality = supports.sum(axis=1)
        pairs = np.triu_indices(nrows, k=1)
        union = cardinality[:, None] + cardinality[None, :] - intersection
        valid = (union[pairs] > 0) & (row_weights[pairs[0]] > 0) & (row_weights[pairs[1]] > 0)
        ii, jj = pairs[0][valid], pairs[1][valid]
        if not len(ii):
            return {'support_jaccard': None, 'weighted_overlap': None, 'duplicate_support_pairs': 0, 'pairs': 0}
        jaccard = intersection[ii, jj] / union[ii, jj]
        minimum = np.minimum(array[ii], array[jj]).sum(axis=1)
        maximum = np.maximum(array[ii], array[jj]).sum(axis=1)
        weights_pair = row_weights[ii] * row_weights[jj]
        return {'support_jaccard': float(np.average(jaccard, weights=weights_pair)),
                'weighted_overlap': float(np.average(minimum / maximum, weights=weights_pair)),
                'duplicate_support_pairs': int(np.count_nonzero(jaccard == 1)), 'pairs': int(len(ii))}

    support = weights > 0
    no_self = support & ~structural_self
    used = service > 0
    return {'cardinality': support.sum(axis=1).tolist(), 'unused_receivers': int((~used).sum()),
            'overlap': overlap(weights, np.ones(nrows)), 'service_weighted_overlap': overlap(weights, service),
            'with_self': graph(support), 'without_self': graph(no_self),
            'served_with_self': graph(support & used[:, None]),
            'served_without_self': graph(no_self & used[:, None])}

"""Bounded source-level geometry changes at fixed binary and action budgets."""

import numpy as np
import torch


def geometry_action_budget(
    density,
    weight,
    support,
    control,
    receivers,
    sources,
    measures,
    pair_valid,
    near,
    source_ids,
    *,
    max_support_switches=32,
    max_active_swaps=32,
    max_candidate_checks=512,
):
    """Improve geometry without changing row/column support degrees.

    Joint tuples move only within receiver rows and exactly equal source
    measure strata. Invalid pairs and every positive near envelope stay fixed.
    Source-column weighted sums may change. This bounded greedy control is not
    an optimum or a shared-group organizer. Its permissions are frozen; current
    physical kernel/source values are evaluated separately by the caller.
    """
    if min(max_support_switches, max_active_swaps, max_candidate_checks) < 0:
        raise ValueError("Geometry budgets must be nonnegative")
    tensors = (density, weight, support, control)
    arrays = [value.detach().cpu().numpy().copy() for value in tensors]
    _rho, w, positive, _controls = arrays
    valid = pair_valid.detach().cpu().numpy()
    movable = valid.copy()
    if near is not None:
        movable &= near.detach().cpu().numpy() == 0
    x, y, mu, ids = [value.detach().cpu().numpy() for value in (receivers, sources, measures, source_ids)]
    stats = {
        "support_switches": 0,
        "active_tuple_swaps": 0,
        "candidate_checks": 0,
        "geometry_cost_before": 0.0,
        "geometry_cost_after": 0.0,
        "source_weighted_sums_preserved": True,
        "candidate_budget_limited": False,
        "support_budget_limited": False,
        "active_swap_budget_limited": False,
        "bounded_greedy_not_global_optimum": True,
        "candidate_check_unit": "Donor source pairs; each scans all eligible partner receiver rows, not a scalar-operation cap",
    }
    initial_support = positive.copy()
    initial_weight = w.copy()

    def swap(case, row, first, second):
        for value in arrays:
            saved = value[case, row, first].copy()
            value[case, row, first] = value[case, row, second]
            value[case, row, second] = saved

    for case in range(w.shape[0]):
        distance = ((x[case, :, None].astype(np.float64) - y[case, None].astype(np.float64)) ** 2).sum(-1)
        mass = mu[case].astype(np.float64)
        if not np.isfinite(distance[valid[case]]).all():
            raise ValueError("Valid physical receiver/source geometry must be finite")
        if not np.isfinite(mass[valid[case].any(0)]).all():
            raise ValueError("Valid physical source measures must be finite")
        # Invalid padding may contain NaN coordinates. It contributes neither
        # cost nor candidate rank, and its permission/control tuples stay fixed.
        distance = np.where(valid[case], distance, 0.0)
        before_mass = np.where(valid[case], mass[None] * initial_weight[case], 0.0)
        stats["geometry_cost_before"] += float((before_mass * distance).sum())
        switches = active_swaps = checks = 0
        for row in range(w.shape[1]):
            if switches >= max_support_switches or checks >= max_candidate_checks:
                break
            active = np.flatnonzero(positive[case, row] & movable[case, row])
            inactive = np.flatnonzero(~positive[case, row] & movable[case, row])
            active = active[np.lexsort((ids[case, active], -distance[row, active]))][:4]
            inactive = inactive[np.lexsort((ids[case, inactive], distance[row, inactive]))][:4]
            for first in active:
                for second in inactive:
                    if switches >= max_support_switches or checks >= max_candidate_checks:
                        break
                    checks += 1
                    if mass[first] != mass[second] or not positive[case, row, first] or positive[case, row, second]:
                        continue
                    partners = np.flatnonzero(
                        movable[case, :, first]
                        & movable[case, :, second]
                        & positive[case, :, second]
                        & ~positive[case, :, first]
                    )
                    if not len(partners):
                        continue
                    delta = mass[first] * (
                        float(w[case, row, first]) * (distance[row, second] - distance[row, first])
                        + w[case, partners, second].astype(np.float64)
                        * (distance[partners, first] - distance[partners, second])
                    )
                    winner = int(np.argmin(delta))
                    if delta[winner] < -1e-12:
                        partner = partners[winner]
                        swap(case, row, first, second)
                        swap(case, partner, first, second)
                        switches += 1
        # Reorder already active actions separately; their binary graph is fixed.
        for row in range(w.shape[1]):
            if active_swaps >= max_active_swaps or checks >= max_candidate_checks:
                break
            active = np.flatnonzero(positive[case, row] & movable[case, row])
            far = active[np.lexsort((ids[case, active], -distance[row, active]))][:4]
            close = active[np.lexsort((ids[case, active], distance[row, active]))][:4]
            for first in far:
                for second in close:
                    if active_swaps >= max_active_swaps or checks >= max_candidate_checks:
                        break
                    checks += 1
                    if first == second or mass[first] != mass[second]:
                        continue
                    delta = (
                        mass[first]
                        * float(w[case, row, first] - w[case, row, second])
                        * (distance[row, second] - distance[row, first])
                    )
                    if delta < -1e-12:
                        swap(case, row, first, second)
                        active_swaps += 1
        stats["support_switches"] += switches
        stats["active_tuple_swaps"] += active_swaps
        stats["candidate_checks"] += checks
        after_mass = np.where(valid[case], mass[None] * w[case], 0.0)
        stats["geometry_cost_after"] += float((after_mass * distance).sum())
        stats["candidate_budget_limited"] |= checks >= max_candidate_checks
        stats["support_budget_limited"] |= switches >= max_support_switches
        stats["active_swap_budget_limited"] |= active_swaps >= max_active_swaps
        stats["source_weighted_sums_preserved"] &= bool(
            np.array_equal(
                before_mass.sum(0),
                after_mass.sum(0),
            )
        )
    stats["changed_support_pairs"] = int(np.count_nonzero(initial_support != positive))
    stats["changed_weight_positions"] = int(np.count_nonzero(initial_weight != w))
    stats["effective_support_control"] = stats["changed_support_pairs"] > 0
    stats["support_degenerate_or_budget_limited"] = stats["support_switches"] == 0
    return tuple(
        torch.as_tensor(value, device=original.device, dtype=original.dtype)
        for value, original in zip(arrays, tensors, strict=True)
    ), stats


__all__ = ["geometry_action_budget"]

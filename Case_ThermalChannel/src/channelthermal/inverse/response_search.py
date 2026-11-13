"""R17-2 fixed-budget, request-local response search for Thermal designs.

The scorer and physical solver are injected by the case-owned caller.  Each
score must rebuild the complete edited scene; no raster geometry derivative or
physical truth is used by these proposal policies.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import time
from typing import Callable, Mapping

import numpy as np

from .request_search import SEED, block_probabilities

POLICIES = ("H-S", "G-S", "H-A", "G-A", "U")
SLOTS = 48
BATCH = 12
MAX_SCENE_SLOTS = 13968
MAX_SOLVE_STARTS = 800
MAX_WORKER_SECONDS = 6 * 3600
MAX_GPU_SECONDS = 4 * 3600


@dataclass
class SearchBudget:
    """Shared R17-2 slot and GPU-time ledger, including preparation/replay."""

    max_slots: int = MAX_SCENE_SLOTS
    max_gpu_seconds: float = MAX_GPU_SECONDS
    slots: int = 0
    gpu_seconds: float = 0.0
    scene_calls: int = 0
    query_pairs: int = 0

    def __post_init__(self):
        if not 0 <= self.max_slots <= MAX_SCENE_SLOTS or not 0 < self.max_gpu_seconds <= MAX_GPU_SECONDS:
            raise ValueError("search budget exceeds the R17-2 ceiling")
        if not 0 <= self.slots <= self.max_slots or not 0 <= self.gpu_seconds <= self.max_gpu_seconds:
            raise ValueError("invalid used search budget")

    def check(self):
        if self.gpu_seconds >= self.max_gpu_seconds:
            raise RuntimeError("R17-2 GPU budget exhausted")

    def charge_slot(self):
        self.check()
        if self.slots >= self.max_slots:
            raise RuntimeError("R17-2 scene-slot budget exhausted")
        self.slots += 1

    def charge_gpu(self, seconds, *, scene_calls=0, query_pairs=0):
        if not np.isfinite(seconds) or seconds < 0 or scene_calls < 0 or query_pairs < 0:
            raise ValueError("invalid measured search cost")
        self.gpu_seconds += float(seconds)
        self.scene_calls += int(scene_calls)
        self.query_pairs += int(query_pairs)
        if self.gpu_seconds > self.max_gpu_seconds:
            raise RuntimeError("R17-2 GPU budget exceeded; retain the partial group")


@dataclass
class PhysicalBudget:
    """R17-2 start/worker ledger; repeated keys never create extra starts."""

    max_starts: int = MAX_SOLVE_STARTS
    max_worker_seconds: float = MAX_WORKER_SECONDS
    starts: int = 0
    worker_seconds: float = 0.0

    def __post_init__(self):
        if not 0 <= self.max_starts <= MAX_SOLVE_STARTS or not 0 < self.max_worker_seconds <= MAX_WORKER_SECONDS:
            raise ValueError("physical budget exceeds the R17-2 ceiling")
        if not 0 <= self.starts <= self.max_starts or not 0 <= self.worker_seconds <= self.max_worker_seconds:
            raise ValueError("invalid used physical budget")
        self.results = {}
        self._finished = set()

    def start(self, key):
        if key in self.results:
            return False
        if self.starts >= self.max_starts or self.worker_seconds >= self.max_worker_seconds:
            raise RuntimeError("R17-2 physical budget exhausted")
        self.starts += 1
        self.results[key] = None
        return True

    def finish(self, key, result, worker_seconds):
        if key not in self.results or key in self._finished or not np.isfinite(worker_seconds) or worker_seconds < 0:
            raise ValueError("unknown or already finished physical solve")
        self.worker_seconds += float(worker_seconds)
        self.results[key] = result
        self._finished.add(key)
        if self.worker_seconds > self.max_worker_seconds:
            raise RuntimeError("R17-2 worker budget exceeded; retain the solved evidence")


def correlation_covariance(kernel):
    """Normalize positive-mass rows; zero-mass members are independent."""
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim != 2 or k.shape[0] != k.shape[1] or not np.isfinite(k).all() or not np.allclose(k, k.T):
        raise ValueError("kernel must be finite, square and symmetric")
    diagonal = np.diag(k)
    if np.any(diagonal < 0):
        raise ValueError("kernel diagonal must be nonnegative")
    positive = diagonal > 0
    corr = np.zeros_like(k)
    den = np.sqrt(np.outer(diagonal, diagonal))
    np.divide(k, den, out=corr, where=den > 0)
    corr[~positive, :] = 0
    corr[:, ~positive] = 0
    np.fill_diagonal(corr, 1.)
    # The prepared-overlap and geometric kernels are PSD.  Reject a mismatched
    # input instead of silently changing the declared covariance.
    if np.linalg.eigvalsh(corr).min(initial=0.) < -1.e-8:
        raise ValueError("kernel correlation is not positive semidefinite")
    return .9 * corr + .1 * np.eye(len(k))


def project_pressure(offsets, pressure_gradient, allowance, dmax):
    """Project one whole block, then shrink it uniformly into the disks."""
    v = np.asarray(offsets, dtype=np.float64)
    g = np.asarray(pressure_gradient, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 2 or g.shape != v.shape or not np.isfinite(v).all() or not np.isfinite(g).all():
        raise ValueError("invalid displacement or available pressure gradient")
    if not np.isfinite(allowance) or allowance < 0 or not np.isfinite(dmax) or dmax <= 0:
        raise ValueError("invalid pressure allowance or displacement bound")
    norm2 = float(np.sum(g * g))
    correction = max(0., (float(np.sum(g * v)) - allowance) / norm2) if norm2 else 0.
    result = v - correction * g
    maxnorm = np.linalg.norm(result, axis=1).max(initial=0.)
    if maxnorm > dmax:
        result *= dmax / maxnorm
    return result


def _rng(layout_index, request_index, slot):
    if min(layout_index, request_index, slot) < 0:
        raise ValueError("search keys must be nonnegative")
    return np.random.default_rng(np.random.SeedSequence([SEED, layout_index, request_index, slot]))


def _draws(layout_index, request_index, slot, n):
    r = _rng(layout_index, request_index, slot)
    return {"size": 1 + int(r.random() >= .5), "mixture": r.random(), "block": r.random(),
            "angle": r.random(n) * (2 * np.pi), "radius": np.sqrt(r.random(n)),
            "amplitude": .25 + .75 * r.random(n), "noise": (r.random(n) - .5) * np.pi / 3,
            "normal": r.standard_normal((n, 2))}


def _pick(blocks, probabilities, u):
    p = np.asarray(probabilities, dtype=np.float64)
    if not len(blocks) or p.shape != (len(blocks),) or (p < 0).any() or not np.isfinite(p).all() or p.sum() <= 0:
        raise ValueError("invalid block probability law")
    p = p / p.sum()
    return blocks[min(int(np.searchsorted(np.cumsum(p), u, side="right")), len(blocks) - 1)]


def _choose_block(laws, size, draws, *, uniform, eligible=None):
    blocks, _, raw = laws[size]
    if uniform:
        return _pick(blocks, np.ones(len(blocks)), draws["block"]), "uniform"
    allowed = [i for i, b in enumerate(blocks) if eligible is None or set(b) <= eligible]
    if not allowed:
        raise ValueError("no eligible guided block")
    raw = np.asarray(raw)[allowed]
    if raw.sum() == 0:
        raw = np.ones(len(allowed))
    return _pick([blocks[i] for i in allowed], raw, draws["block"]), "guided"


def _config_with_centers(config, centers):
    result = deepcopy(config)
    result["layout"]["centers"] = np.asarray(centers, dtype=np.float64).tolist()
    # The only mutable physical input is the original-member center array.
    locked = deepcopy(result)
    locked["layout"]["centers"] = deepcopy(config["layout"]["centers"])
    if locked != config:
        raise ValueError("proposal changed prescribed controls")
    return result


def _center_key(centers):
    return np.ascontiguousarray(centers, dtype=np.float64).tobytes()


def select_finalists(baseline_f, rows, allowance):
    """Choose both portfolio entries using predictions alone, in slot order."""
    if not np.isfinite(baseline_f) or not np.isfinite(allowance) or allowance < 0:
        raise ValueError("invalid predicted baseline or pressure allowance")
    valid = [r for r in rows if r["valid"] and not r["baseline"] and r["prediction"] is not None]
    finite = [r for r in valid if np.isfinite(r["prediction"]["f"]) and np.isfinite(r["prediction"]["p"])]
    best_objective = min(finite, key=lambda r: (r["prediction"]["f"], r["slot"]), default=None)
    feasible = [r for r in finite if r["prediction"]["f"] < baseline_f and r["delta_p"] <= allowance]
    best_feasible = min(feasible, key=lambda r: (r["prediction"]["f"], r["slot"]), default=None)
    return {"feasible": None if best_feasible is None else best_feasible["slot"],
            "objective": None if best_objective is None else best_objective["slot"],
            "portfolio": list(dict.fromkeys(r["slot"] for r in (best_feasible, best_objective) if r is not None))}


def _probe_gradients(rows, baseline, nominated, step):
    result = {member: {"f": np.full(2, np.nan), "p": np.full(2, np.nan), "method": [None, None]}
              for member in nominated}
    for member in nominated:
        for axis in range(2):
            pair = [next(r for r in rows if r["probe"] == (member, axis, sign)) for sign in (1, -1)]
            for field in ("f", "p"):
                good = [r["valid"] and r["prediction"] is not None and np.isfinite(r["prediction"][field]) for r in pair]
                if all(good):
                    value = (pair[0]["prediction"][field] - pair[1]["prediction"][field]) / (2 * step)
                    method = "centered"
                elif good[0]:
                    value = (pair[0]["prediction"][field] - baseline[field]) / step
                    method = "forward"
                elif good[1]:
                    value = (baseline[field] - pair[1]["prediction"][field]) / step
                    method = "backward"
                else:
                    value, method = np.nan, "unavailable"
                result[member][field][axis] = value
                result[member]["method"][axis] = method
    return result


def search(config: Mapping, *, policy: str, original_member_ids, service, affinity,
           baseline: Mapping[str, float], pressure_allowance: float, roi_scale: float,
           layout_index: int, request_index: int, valid: Callable, score: Callable,
           budget: SearchBudget | None = None, observer: Callable | None = None):
    """Run exactly 48 charged slots unless a hard budget or scorer error stops it.

    ``score`` receives the full source-FP64 edited configuration and returns
    ``f`` and ``p`` in physical units, plus optional ``query_pairs``/metadata.
    ``baseline`` comes from the same scorer and the fixed request ROI. Arrays in
    the returned rows preserve all attempted scenes, even invalid/duplicate ones.
    """
    if policy not in POLICIES:
        raise ValueError("unknown R17-2 policy")
    source = config["layout"]["centers"]
    if isinstance(source, np.ndarray) and source.dtype != np.float64:
        raise ValueError("physical centers must retain FP64 source precision")
    centers = np.asarray(source, dtype=np.float64)
    ids_in_slots = tuple(int(v) for v in original_member_ids)
    if centers.ndim != 2 or centers.shape != (len(ids_in_slots), 2) or not len(ids_in_slots) or len(set(ids_in_slots)) != len(ids_in_slots) or not np.isfinite(centers).all():
        raise ValueError("invalid original-member scene")
    if any(not np.isfinite(float(baseline[k])) for k in ("f", "p")) or not np.isfinite(roi_scale) or roi_scale <= 0 or not np.isfinite(pressure_allowance) or pressure_allowance <= 0:
        raise ValueError("invalid fixed baseline, ROI scale or pressure allowance")
    radius = float(config["domain"]["module_radius"])
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("invalid physical radius")
    dmax = .15 * radius
    order = np.argsort(ids_in_slots)
    ids = tuple(ids_in_slots[i] for i in order)
    slot_for = {member: ids_in_slots.index(member) for member in ids}
    index_for = {member: i for i, member in enumerate(ids)}
    service = np.asarray(service, dtype=np.float64)[order]
    kernel = np.asarray(affinity, dtype=np.float64)[np.ix_(order, order)]
    laws = block_probabilities(ids, service, kernel)
    covariance = correlation_covariance(kernel) if policy.endswith("-A") else None
    budget = budget if budget is not None else SearchBudget()
    rows, seen = [], {_center_key(centers): dict(baseline)}
    means = np.zeros((len(ids), 2), dtype=np.float64)
    means_history = []
    nominated = ()
    probe_schedule = []
    if policy.endswith("-S"):
        nomination = _draws(layout_index, request_index, 0, len(ids))
        nominated, _ = _choose_block(laws, min(2, len(ids)), nomination,
                                     uniform=nomination["mixture"] < .2)
        nominated = tuple(sorted(nominated))
        probe_schedule = [(member, axis, sign) for member in nominated for axis in (0, 1) for sign in (1, -1)]
    gradients = None
    started = time.monotonic()
    for slot in range(SLOTS):
        budget.charge_slot()
        draw = _draws(layout_index, request_index, slot, len(ids))
        edited = centers.copy()
        probe = probe_schedule[slot] if slot < len(probe_schedule) else None
        branch = "probe" if probe else None
        block = ()
        if probe is not None:
            member, axis, sign = probe
            edited[slot_for[member], axis] += sign * .05 * radius
            block = (member,)
        else:
            if policy.endswith("-S") and gradients is None:
                gradients = _probe_gradients(rows, baseline, nominated, .05 * radius)
            size = min(draw["size"], len(ids))
            use_uniform = policy == "U" or draw["mixture"] < .2
            eligible = set(nominated) if policy.endswith("-S") and not use_uniform else None
            block, branch = _choose_block(laws, size, draw, uniform=use_uniform, eligible=eligible)
            ix = [index_for[m] for m in block]
            if use_uniform:
                offsets = dmax * draw["radius"][ix, None] * np.column_stack((np.cos(draw["angle"][ix]), np.sin(draw["angle"][ix])))
            elif policy.endswith("-S"):
                offsets, pg = [], []
                for member, i in zip(block, ix):
                    gf = gradients[member]["f"]
                    available = np.isfinite(gf)
                    direction = -np.where(available, gf, 0.)
                    norm = np.linalg.norm(direction)
                    if norm == 0:
                        direction = draw["radius"][i] * np.array([np.cos(draw["angle"][i]), np.sin(draw["angle"][i])])
                        branch = "guided_exploratory_direction"
                    else:
                        direction = direction / norm * draw["amplitude"][i]
                        angle = draw["noise"][i]
                        direction = np.array([direction[0]*np.cos(angle)-direction[1]*np.sin(angle),
                                              direction[0]*np.sin(angle)+direction[1]*np.cos(angle)])
                    offsets.append(dmax * direction)
                    pg.append(np.nan_to_num(gradients[member]["p"], nan=0.))
                offsets = project_pressure(offsets, pg, pressure_allowance, dmax)
            else:
                sub = covariance[np.ix_(ix, ix)]
                offsets = means[ix] + .5 * dmax * (np.linalg.cholesky(sub) @ draw["normal"][ix])
                norms = np.linalg.norm(offsets, axis=1)
                offsets *= np.minimum(1., dmax / np.maximum(norms, np.finfo(float).tiny))[:, None]
            for member, offset in zip(block, offsets):
                edited[slot_for[member]] += offset
        candidate = _config_with_centers(config, edited)
        admissible = bool(valid(candidate))
        key = _center_key(edited)
        duplicate = key in seen
        prediction = None
        pending_error = None
        if admissible:
            if duplicate:
                prediction = seen[key]
            else:
                try:
                    budget.check()
                    before = time.monotonic()
                    try:
                        prediction = dict(score(candidate))
                    finally:
                        seconds = time.monotonic() - before
                        budget.charge_gpu(seconds, scene_calls=1,
                                          query_pairs=0 if prediction is None else int(prediction.get("query_pairs", 0)))
                    if any(not np.isfinite(float(prediction[k])) for k in ("f", "p")):
                        raise ValueError("full nonlinear scorer returned nonfinite prediction")
                    seen[key] = prediction
                except Exception as exc:
                    pending_error = exc
        delta_p = None
        if prediction is not None:
            try:
                f_value, p_value = float(prediction["f"]), float(prediction["p"])
                if not np.isfinite([f_value, p_value]).all():
                    raise ValueError("full nonlinear scorer returned nonfinite prediction")
                delta_p = p_value - baseline["p"]
            except (KeyError, TypeError, ValueError) as exc:
                if pending_error is None:
                    pending_error = ValueError(f"invalid full nonlinear scorer prediction: {exc}")
        row = {"slot": slot, "block": tuple(block), "branch": branch, "probe": probe,
               "valid": admissible, "duplicate": duplicate, "baseline": key == _center_key(centers),
               "centers": edited, "prediction": prediction, "delta_p": delta_p,
               "error": None if pending_error is None else repr(pending_error)}
        rows.append(row)
        if observer is not None:
            observer(row)
        if covariance is not None and (slot + 1) % BATCH == 0 and pending_error is None:
            unique = {}
            for item in rows:
                if item["valid"] and not item["baseline"] and item["prediction"] is not None:
                    unique.setdefault(_center_key(item["centers"]), item)
            ranked = sorted(unique.values(), key=lambda r: (
                -((baseline["f"] - r["prediction"]["f"]) / roi_scale
                  - max(0., (r["delta_p"] - pressure_allowance) / (2 * pressure_allowance))), r["slot"]))
            if ranked:
                displacement = np.stack([(r["centers"] - centers)[order] for r in ranked[:4]])
                means = .5 * means + .5 * displacement.mean(axis=0)
            means_history.append(means.copy())
        if pending_error is not None:
            raise pending_error
    finalists = select_finalists(float(baseline["f"]), rows, pressure_allowance)
    raw = next((row["slot"] for row in rows if row["probe"] is None and row["valid"]), None)
    return {"policy": policy, "slots": rows, "nominated": nominated, "gradients": gradients,
            "covariance": covariance, "mean_history": means_history, "finalists": finalists,
            "raw_slot": raw, "elapsed_seconds": time.monotonic() - started,
            "actual_scene_calls": sum(r["valid"] and not r["duplicate"] for r in rows),
            "probabilities": laws, "member_ids": ids}


def physical_key(config, numerical_policy):
    """Exact physical configuration and numerical-policy identity for reuse."""
    keys = ("domain", "flow", "thermal", "layout", "local_module")
    if not numerical_policy:
        raise ValueError("actual physical numerical policy is required")
    return json.dumps({"configuration": {k: config[k] for k in keys}, "numerical_policy": numerical_policy},
                      sort_keys=True, separators=(",", ":"), allow_nan=False)


def reduce_physical_fields(field, fluid_mask, fluid_xy, roi_mask, domain_length_x):
    """Re-reduce retained fields against this request's fixed physical mask."""
    values = np.asarray(field)
    if values.ndim == 3:
        values = values.reshape(-1, values.shape[-1])
    valid = np.asarray(fluid_mask, dtype=bool).reshape(-1)
    xy = np.asarray(fluid_xy, dtype=np.float64).reshape(-1, 2)
    roi = np.asarray(roi_mask, dtype=bool).reshape(-1)
    if values.ndim != 2 or values.shape[1] < 5 or len(values) != len(valid) or len(xy) != len(valid) or len(roi) != len(valid):
        raise ValueError("physical field, mask and receiver axes differ")
    if not roi.any() or np.any(roi & ~valid) or not np.isfinite(values[valid]).all() or not np.isfinite(xy).all() or not np.isfinite(domain_length_x) or domain_length_x <= 0:
        raise ValueError("invalid physical reference or request mask")
    left = valid & (xy[:, 0] <= .08 * domain_length_x)
    right = valid & (xy[:, 0] >= .92 * domain_length_x)
    if not left.any() or not right.any():
        raise ValueError("physical pressure bands are empty")
    return {"f": float(values[roi, 4].astype(np.float64).mean()),
            "p": float(values[left, 2].astype(np.float64).mean() - values[right, 2].astype(np.float64).mean())}


class PhysicalFieldCache:
    """Reuse complete retained fields, never a request-specific scalar result."""

    def __init__(self, loader):
        self.loader = loader
        self.references = {}

    def remember(self, config, numerical_policy, reference):
        key = physical_key(config, numerical_policy)
        if key in self.references and self.references[key] != reference:
            raise ValueError("physical identity already has a different retained reference")
        self.references[key] = reference
        return key

    def reduce(self, key, roi_mask):
        reference = self.references[key]
        values = self.loader(reference)
        return reduce_physical_fields(values["field"], values["fluid_mask"], values["fluid_xy"],
                                      roi_mask, values["domain_length_x"])


def validate_physical_convergence(rows, config):
    """Recheck the literal final saved convergence window and solver policy.

    The final window must contain only finite records, as in the packed-data
    audit. The solver stores nine significant decimal digits in its CSV; the
    small tolerance below accounts only for that serialization rounding.
    """
    thermal, runtime = config["thermal"], config["runtime"]
    window = int(thermal["convergence_window"])
    absolute, relative = float(thermal["convergence_tol"]), float(thermal["convergence_rel_tol"])
    if (not thermal["stop_on_convergence"] or window < 1 or not np.isfinite([absolute, relative]).all()
            or min(absolute, relative) < 0 or not rows):
        raise ValueError("invalid saved physical convergence policy")
    final = rows[-1]
    if int(final["converged"]) != 1 or int(final["convergence_window_ok"]) != 1:
        raise ValueError("physical reference did not converge with a complete window")
    if any(int(row["converged"]) for row in rows[:-1]):
        raise ValueError("physical frame history continued after convergence")
    if (float(final["time"]) < float(thermal["min_solve_time"])
            or (thermal["require_heat_active_for_convergence"] and not int(final["heat_active"]))):
        raise ValueError("physical convergence did not satisfy the source time/heat policy")
    if len(rows) < window:
        raise ValueError("insufficient convergence history")
    recent = np.asarray([(float(row["delta_inf"]), float(row["delta_l2_rel"]))
                         for row in rows[-window:]], dtype=np.float64)
    if not np.isfinite(recent).all():
        raise ValueError("nonfinite metric in the final convergence window")
    if (recent < 0).any() or recent[:, 0].max() > absolute * (1 + 1.e-8) + 1.e-15 or recent[:, 1].max() > relative * (1 + 1.e-8) + 1.e-15:
        raise ValueError("physical convergence window exceeds the saved policy")
    if (not bool(runtime["converged"]) or int(runtime["converged_step"]) != int(final["step"])
            or not np.isclose(float(runtime["converged_time"]), float(final["time"]), rtol=0, atol=5.e-8)
            or not np.isclose(float(runtime["final_delta_inf"]), float(final["delta_inf"]), rtol=1.e-8, atol=1.e-15)
            or not np.isclose(float(runtime["final_delta_l2_rel"]), float(final["delta_l2_rel"]), rtol=1.e-8, atol=1.e-15)):
        raise ValueError("physical convergence metadata disagrees with the final frame")
    return {"window": window, "max_delta_inf": float(recent[:, 0].max()),
            "max_delta_l2_rel": float(recent[:, 1].max()), "converged_step": int(final["step"]),
            "converged_time": float(final["time"]), "csv_rounding_rtol": 1.e-8}


def accept_portfolio(baseline, finalists, physical_results, pressure_allowance, resolution):
    """Keep failures explicit; a failed finalist can still change the winner."""
    if pressure_allowance < 0 or resolution < 0 or not all(np.isfinite(float(baseline[k])) for k in ("f", "p")):
        raise ValueError("invalid baseline or acceptance settings")
    accepted, best = None, float(baseline["f"])
    statuses = []
    for slot in sorted(finalists):
        result = physical_results.get(slot)
        if result is None or result.get("status") != "solved":
            statuses.append({"slot": slot, "status": "unresolved"})
            continue
        f, p = float(result["f"]), float(result["p"])
        if not np.isfinite(f) or not np.isfinite(p):
            statuses.append({"slot": slot, "status": "unresolved"})
            continue
        feasible = p - baseline["p"] <= pressure_allowance
        improves = f < baseline["f"]
        statuses.append({"slot": slot, "status": "solved", "pressure_violation": not feasible,
                         "worsened": f > baseline["f"], "gain": float(baseline["f"] - f)})
        if feasible and improves and f < best:
            accepted, best = slot, f
    incomplete = any(x["status"] == "unresolved" for x in statuses)
    gain = float(baseline["f"] - best)
    return {"status": "incomplete" if incomplete else "resolved",
            "accepted": None if incomplete else accepted,
            "gain": None if incomplete else gain,
            "above_resolution": None if incomplete else gain > resolution,
            "provisional_accepted": accepted, "provisional_gain": gain,
            "finalist_statuses": statuses}

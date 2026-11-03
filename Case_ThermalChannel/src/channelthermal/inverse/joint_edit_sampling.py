"""Thermal original-member joint edit proposals and finite-pool selection.

All coordinates and radius are in physical units. Prepared organization must be
from the same ordinary predictor preparation as the design being edited.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Sequence
import numpy as np

from honf_inverse_core.contracts import PhysicalDesign
from .geometry import evaluate_geometry


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class Organization:
    module_ids: np.ndarray
    environment_ids: np.ndarray
    environment_measure: np.ndarray
    full_rows: tuple[dict[str, np.ndarray], ...]
    restricted_rows: tuple[dict[str, np.ndarray], ...]
    full_module: np.ndarray  # [E,M], actual Pi access weights
    restricted_module: np.ndarray  # [E,M], actual D access weights
    mm: np.ndarray  # [M,M], receiver <- donor
    stages: dict[str, dict[str, Any]]


def extract_organization(prepared, module_ids, environment_ids, *, batch_index=0):
    """Extract actual packed full/restricted E rows and four final stages by ID."""
    org = getattr(prepared, "organization", prepared)
    scene, directory = org.scene, org.e_access_directory
    mids, eids = np.asarray(module_ids), np.asarray(environment_ids)
    smids = _numpy(scene.module_ids)[batch_index]
    seids = _numpy(scene.environment_ids)[batch_index]
    mmask = _numpy(scene.module_mask)[batch_index].astype(bool)
    emask = _numpy(scene.environment_mask)[batch_index].astype(bool)
    if len(set(mids.tolist())) != len(mids) or len(set(eids.tolist())) != len(eids):
        raise ValueError("Original IDs must be unique within each role.")
    if set(mids.tolist()) != set(smids[mmask].tolist()) or set(eids.tolist()) != set(seids[emask].tolist()):
        raise ValueError("Requested IDs must exactly match the prepared active inventory.")
    mindex = {int(v): i for i, v in enumerate(mids)}
    eindex = {int(v): i for i, v in enumerate(eids)}
    measure = np.zeros(len(eids))
    for slot in np.flatnonzero(emask):
        measure[eindex[int(seids[slot])]] = _numpy(scene.environment_measure)[batch_index, slot]
    if np.any(measure <= 0):
        raise ValueError("Active E addresses require positive physical measure.")

    def rows_and_matrix(packed):
        rows = [None] * len(eids)
        matrix = np.zeros((len(eids), len(mids)))
        for slot in np.flatnonzero(emask):
            a = eindex[int(seids[slot])]
            row = packed.row(batch_index, int(slot))
            values = {name: _numpy(getattr(row, name)).copy() for name in
                      ("member_index", "original_id", "is_module", "access_weight", "applied_exchange", "structural_self", "measure")}
            rows[a] = values
            for ident, is_m, weight in zip(values["original_id"], values["is_module"], values["access_weight"]):
                if is_m:
                    matrix[a, mindex[int(ident)]] = weight
        return tuple(rows), matrix

    full_rows, full = rows_and_matrix(directory.full)
    restricted_rows, restricted = rows_and_matrix(directory.restricted)
    stages = {}
    for route in ("mm", "me", "em", "ee"):
        item = org.final_applied_context_stages.get(route)
        if not item or item.get("snapshot") is None or item.get("incidence") is None:
            raise ValueError(f"Actual {route.upper()} stage unavailable in ordinary preparation.")
        snapshot = item["snapshot"]
        incidence = _numpy(item["incidence"])[batch_index].copy()
        dest = _numpy(snapshot.destination_ids)[batch_index].copy()
        donor = _numpy(snapshot.donor_ids)[batch_index].copy()
        dmask = _numpy(snapshot.destination_mask)[batch_index].astype(bool)
        smask = _numpy(snapshot.donor_mask)[batch_index].astype(bool)
        if incidence.shape != (len(dest), len(donor)):
            raise ValueError("Applied incidence and snapshot axes differ.")
        stages[route] = dict(incidence=incidence, destination_ids=dest, donor_ids=donor,
                             destination_mask=dmask, donor_mask=smask,
                             allowed=_numpy(snapshot.allowed)[batch_index].copy(),
                             donor_measure=_numpy(snapshot.donor_measure)[batch_index].copy(),
                             stage_identity=snapshot.stage_identity)
    stage = stages["mm"]
    mm = np.zeros((len(mids), len(mids)))
    for i in np.flatnonzero(stage["destination_mask"]):
        for j in np.flatnonzero(stage["donor_mask"]):
            mm[mindex[int(stage["destination_ids"][i])], mindex[int(stage["donor_ids"][j])]] = stage["incidence"][i, j]
    return Organization(mids.copy(), eids.copy(), measure, full_rows, restricted_rows, full, restricted, mm, stages)


def make_loadings(arm, module_xy, environment_xy, domain_lengths, organization=None, *, kappa=0.25):
    """Return normalized b[M,E,2]; grid cells are the canonical 24 by 8."""
    x = np.asarray(module_xy, dtype=float)
    e = np.asarray(environment_xy, dtype=float)
    cell = np.asarray(domain_lengths, dtype=float) / np.array([24., 8.])
    if x.ndim != 2 or x.shape[1] != 2 or e.ndim != 2 or e.shape[1] != 2 or cell.shape != (2,) or np.any(cell <= 0):
        raise ValueError("Expected physical coordinates [M,2], [E,2], and positive domain lengths.")
    u = (x[:, None, :] - e[None, :, :]) / cell
    if arm == "I":
        h = np.zeros_like(u)
    elif arm == "G":
        h = np.exp(-0.5 * np.sum(u * u, axis=2))[:, :, None] * np.tanh(u)
    elif arm == "H":
        if organization is None or organization.restricted_module.shape != (len(e), len(x)) or organization.mm.shape != (len(x), len(x)):
            raise ValueError("H requires ID-aligned restricted directory and MM stage.")
        direction = np.sum(organization.mm[:, :, None] * (x[None, :, :] - x[:, None, :]) / cell, axis=1)
        h = organization.restricted_module.T[:, :, None] * np.tanh(u + kappa * direction[:, None, :])
    else:
        raise ValueError("arm must be I, G, or H")
    norm = np.sqrt(np.sum(h * h, axis=1, keepdims=True))
    return h / np.maximum(norm, 1e-8)


@dataclass(frozen=True)
class Draw:
    pair: tuple[int, int]  # original module IDs
    epsilon: np.ndarray  # aligned with module_ids
    z: np.ndarray  # aligned with environment_ids
    nomination_uniform: float


def common_draws(seed, eligible_ids, module_ids, environment_ids, count):
    """Generate all random numbers before feasibility checks for shared arms."""
    mids, eids = np.asarray(module_ids), np.asarray(environment_ids)
    eligible = sorted(int(x) for x in eligible_ids)
    if len(set(eligible)) != len(eligible) or not set(eligible).issubset(set(mids.tolist())):
        raise ValueError("Eligible original IDs must be unique members of the inventory.")
    pairs = list(combinations(eligible, 2))
    if not pairs:
        raise ValueError("At least two editable modules are required.")
    rng = np.random.default_rng(seed)
    return [Draw(pairs[int(rng.integers(len(pairs)))], rng.standard_normal((len(mids), 2)),
                 rng.standard_normal(len(eids)), float(rng.random())) for _ in range(count)]


def propose(design, context, constraints, module_ids, loadings, draw, *, rho=1 / np.sqrt(2)):
    """Construct exactly one trial and reject excessive displacement or geometry."""
    ids = np.asarray(module_ids)
    active_slots = np.flatnonzero(design.module_present > .5)
    if len(ids) != len(active_slots) or len(set(ids.tolist())) != len(ids):
        raise ValueError("module_ids must correspond in order to active padded slots.")
    b = np.asarray(loadings, dtype=float)
    if b.shape != (len(ids), len(draw.z), 2) or draw.epsilon.shape != (len(ids), 2):
        raise ValueError("Draw and loadings must align with original inventories.")
    chosen = [int(np.flatnonzero(ids == ident)[0]) for ident in draw.pair]
    if len(set(chosen)) != 2:
        raise ValueError("A draw must select two different active IDs.")
    v = np.sum(b * b, axis=1)
    if np.any(v > 1 + 1e-10) or not 0 <= rho <= 1:
        raise ValueError("Invalid normalized loadings or rho.")
    radius = context.as_mapping()["module_radius"]
    delta = .05 * radius * (np.sqrt(np.maximum(1 - rho * rho * v, 0)) * draw.epsilon + rho * np.einsum("iad,a->id", b, draw.z))
    centers = design.module_centers.copy()
    centers[active_slots[chosen]] += delta[chosen]
    trial = PhysicalDesign(centers, design.module_present.copy(), design.heat_powers.copy(), design.module_family_id)
    if np.any(np.linalg.norm(delta[chosen], axis=1) > .15 * radius):
        return trial, "displacement"
    if not evaluate_geometry(trial, context, constraints).valid:
        return trial, "geometry"
    return trial, None


def request_energies(request, j, p, j0, p0, tin, temperature_scale, fit_pressure_rms):
    """Finite-pool request energy from baseline-anchored physical predictions."""
    st = max(j0 - tin, .01 * temperature_scale)
    sp = max(abs(p0), .01 * fit_pressure_rms)
    if st <= 0 or sp <= 0:
        raise ValueError("Request scales must be positive.")
    j, p = np.asarray(j, dtype=float), np.asarray(p, dtype=float)
    if request == "A":
        objective = (j - (j0 - .01 * st)) / (.01 * st)
        constraint = (p - (p0 + .05 * sp)) / (.05 * sp)
    elif request == "B":
        objective = (p - (p0 - .01 * sp)) / (.01 * sp)
        constraint = (j - j0) / (.01 * st)
    else:
        raise ValueError("request must be A or B")
    return np.maximum(objective, 0)**2 + np.maximum(constraint, 0)**2


def finite_pool_weights(energies):
    energies = np.asarray(energies, dtype=float)
    if energies.ndim != 1 or not np.isfinite(energies).all():
        raise ValueError("Finite pool energies must be a finite vector.")
    if not len(energies):
        return energies.copy()
    weights = np.exp(-(energies - energies.min()))
    return weights / weights.sum()


def nominate(keys: Sequence, energies, uniform, *, weighted=True):
    """Select among distinct designs, preserving first occurrence; None if empty."""
    if not 0 <= uniform < 1:
        raise ValueError("Nomination uniform must be in [0,1).")
    energies = np.asarray(energies, dtype=float)
    if len(keys) != len(energies):
        raise ValueError("Every occurrence needs its energy.")
    first = {}
    for i, key in enumerate(keys):
        first.setdefault(key, i)
    indices = np.asarray(list(first.values()), dtype=int)
    if len(indices) == 0:
        return None
    weights = finite_pool_weights(energies[indices]) if weighted else np.full(len(indices), 1 / len(indices))
    return int(indices[min(np.searchsorted(np.cumsum(weights), uniform, side="right"), len(indices)-1)])

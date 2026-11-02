"""FIT-only response probes over saved Thermal geometry edits.

All organization inputs are actual full-mode traces, indexed by original IDs.
This module does not prepare a model, modify its state, or solve new designs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Mapping, Sequence

import numpy as np

from channelthermal.interaction_evidence.response_atlas import load_response_atlas_stencil
from channelthermal.interaction_evidence.quantities import smooth_peak_temperature

FIT = ("0001", "0318", "0333", "0348")
DEV = ("0304", "0320", "0335", "0350")
EDITS = ("i_minus", "i_plus", "j_minus", "j_plus", "mm", "mp", "pm", "pp")
FEATURES = ("dx_i/r", "dy_i/r", "dx_j/r", "dy_j/r", "distance/r", "heat_difference/scale",
            "module_count", "total_heat", "inlet_speed", "reynolds", "conductivity_ratio", "radius/domain_height",
            "baseline_smooth_temperature", "baseline_pressure_drop", "temperature_directional_change", "pressure_directional_change",
            "baseline_member_temperature_i", "baseline_member_temperature_j", "query_M_i", "query_M_j",
            "shared_E_membership", "MM_i_from_j", "MM_j_from_i", "EM_moved_mass")


def atlas_edits(path: str | Path, *, beta: float) -> list[dict]:
    """Join saved baseline and trials by original ID, including physical targets."""
    stencil, metadata = load_response_atlas_stencil(path)
    with np.load(path, allow_pickle=False) as archive:
        family_metadata = json.loads(str(archive["family_metadata_json"].item()))
    ids = tuple(map(str, family_metadata.get("position_panel", {}).get("module_ids", ())))
    if len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError("Atlas must declare two distinct position-panel module IDs")
    base = {m.module_id: m for m in stencil.baseline.design.active_modules}
    if not set(ids) <= set(base):
        raise ValueError("Position IDs are not original active module IDs")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("Smooth-peak beta must be positive")
    def objective(record):
        role = record.output.roles["solid_temperature"]
        channel = role.channel_names.index("temperature")
        values = role.values[:, channel]
        valid = role.valid_mask[:, channel]
        return smooth_peak_temperature(values[valid], beta)
    baseline = stencil.baseline
    rows = []
    for label in EDITS:
        trial = stencil.variants[label]
        mods = {m.module_id: m for m in trial.design.active_modules}
        if set(mods) != set(base) or any(mods[k].heating != base[k].heating for k in base):
            raise ValueError("Geometry edits must preserve IDs and heating")
        changes = {k: np.asarray(mods[k].position_xy) - np.asarray(base[k].position_xy) for k in base}
        if any(np.any(changes[k] != 0) for k in base if k not in ids):
            raise ValueError("Unexpected changed module outside the declared panel")
        moved = sum(bool(np.any(changes[k] != 0)) for k in ids)
        if moved != (1 if label in EDITS[:4] else 2):
            raise ValueError(f"Unexpected actual geometry for {label}")
        rows.append(dict(family=stencil.anchor_id, label=label, member_ids=ids,
                         displacement=np.stack([changes[k] for k in ids]),
                         baseline_positions=np.stack([base[k].position_xy for k in ids]),
                         baseline_heating=np.array([base[k].heating for k in ids]),
                         target=np.array([objective(trial) - objective(baseline),
                                          trial.output.pressure_drop - baseline.output.pressure_drop])))
    return rows


def organization_features(*, member_ids: Sequence[str], source_ids: Sequence[str], query_m: np.ndarray,
                          objective_weights: np.ndarray, anchor_indices: np.ndarray, anchor_weights: np.ndarray,
                          e_measures: np.ndarray, directory_m: np.ndarray, mm: np.ndarray, em: np.ndarray,
                          displacement: np.ndarray, permutation: Sequence[int] | None = None) -> np.ndarray:
    """Summarize actual receiver←donor traces; permutation breaks module alignment only.

    Inputs: query_m [Q,M], anchor arrays [Q,K], directory_m and em [E,M],
    mm [M,M], objective_weights [Q], e_measures [E]. No trace is fabricated.
    """
    ids = tuple(map(str, source_ids)); chosen = tuple(map(str, member_ids)); m = len(ids)
    if len(set(ids)) != m or len(chosen) != 2 or any(k not in ids for k in chosen):
        raise ValueError("Trace and edit IDs must identify two original members")
    q = np.asarray(query_m, float); d = np.asarray(directory_m, float); mm = np.asarray(mm, float); em = np.asarray(em, float)
    w = np.asarray(objective_weights, float); ai = np.asarray(anchor_indices, int); aw = np.asarray(anchor_weights, float); measures = np.asarray(e_measures, float)
    delta = np.asarray(displacement, float)
    if q.ndim != 2 or q.shape[1] != m or d.ndim != 2 or d.shape[1] != m or em.shape != d.shape or mm.shape != (m,m) or w.shape != (len(q),) or ai.shape != aw.shape or ai.shape[0] != len(q) or measures.shape != (len(d),) or delta.shape != (2,2):
        raise ValueError("Organization trace dimensions do not align")
    if not all(np.isfinite(x).all() for x in (q,d,mm,em,w,aw,measures,delta)) or np.any(w < 0) or np.any(aw < 0) or np.any(measures < 0) or w.sum() <= 0:
        raise ValueError("Trace inputs must be finite with valid nonnegative objective weights")
    if np.any((ai < 0) & (aw != 0)) or np.any(ai >= len(d)):
        raise ValueError("Invalid E anchor indices")
    perm = np.arange(m) if permutation is None else np.asarray(permutation, int)
    if perm.shape != (m,) or set(perm.tolist()) != set(range(m)):
        raise ValueError("Permutation must be a bijection of the original module slots")
    i,j = (int(perm[ids.index(k)]) for k in chosen)
    w = w / w.sum(); ew = np.zeros(len(d))
    valid = ai >= 0
    np.add.at(ew, ai[valid], (w[:,None] * aw)[valid])
    ew *= measures
    if ew.sum() <= 0:
        raise ValueError("Objective-region E weights have zero mass")
    ew /= ew.sum()
    moved = np.any(delta != 0, axis=1); pair = bool(np.all(moved))
    return np.array([w @ q[:,i], w @ q[:,j], ew @ np.minimum(d[:,i], d[:,j]) if pair else 0.,
                     mm[i,j] if pair else 0., mm[j,i] if pair else 0.,
                     sum(float(ew @ em[:,idx]) for idx, active in zip((i,j), moved) if active)])


def feature_row(edit: Mapping, *, radius: float, heat_scale: float, context: Sequence[float],
                forward: Sequence[float], organization: Sequence[float]) -> np.ndarray:
    if radius <= 0 or heat_scale <= 0:
        raise ValueError("Radius and FIT heat scale must be positive")
    disp = np.asarray(edit["displacement"], float); pos = np.asarray(edit["baseline_positions"], float)
    heat = np.asarray(edit["baseline_heating"], float)
    row = np.r_[disp.reshape(-1)/radius, np.linalg.norm(pos[0]-pos[1])/radius,
                (heat[0]-heat[1])/heat_scale, context, forward, organization].astype(float)
    if row.shape != (24,) or not np.isfinite(row).all():
        raise ValueError("Expected six slots each for geometry, context, forward and organization")
    return row


@dataclass(frozen=True)
class RidgeProbe:
    mean: np.ndarray
    scale: np.ndarray
    target_scale: np.ndarray
    intercept: np.ndarray
    coefficients: np.ndarray

    def predict(self, x):
        x = np.asarray(x, float)
        return (((x - self.mean) / self.scale) @ self.coefficients + self.intercept) * self.target_scale


def fit_probe(x, y, inherited_scales, *, family_ids: Sequence[str]) -> RidgeProbe:
    """One analytical calibration epoch: mean squared standardized error + L2=1."""
    x = np.asarray(x, float); y = np.asarray(y, float); inherited = np.asarray(inherited_scales, float)
    if x.ndim != 2 or not 1 <= x.shape[1] <= 24 or y.shape != (len(x), 2) or len(family_ids) != len(x) or set(family_ids) != set(FIT):
        raise ValueError("Calibration requires exactly the four FIT families and <=24 features")
    if inherited.shape != (2,) or np.any(inherited <= 0) or not all(np.isfinite(z).all() for z in (x,y,inherited)):
        raise ValueError("Calibration inputs and positive inherited scales must be finite")
    mean = x.mean(axis=0); scale = x.std(axis=0); scale = np.where(scale > 0, scale, 1.)
    target_scale = np.maximum(np.sqrt(np.mean(y*y, axis=0)), 1e-3*inherited)
    z = (x-mean)/scale; t = y/target_scale; intercept = t.mean(axis=0)
    coef = np.linalg.solve(z.T @ z / len(x) + np.eye(x.shape[1]), z.T @ (t-intercept) / len(x))
    return RidgeProbe(mean, scale, target_scale, intercept, coef)


def evaluate_probe(probe: RidgeProbe, x, y, families: Sequence[str]) -> dict:
    """Per-family physical and FIT-scale normalized MAE and ordering regret."""
    prediction = probe.predict(x); y = np.asarray(y, float); families = np.asarray(families)
    if prediction.shape != y.shape or y.ndim != 2 or y.shape[1] != 2 or families.shape != (len(y),):
        raise ValueError("Predictions, targets and families must align")
    result = {}
    for family in sorted(set(families)):
        mask = families == family; p = prediction[mask]; actual = y[mask]
        error = np.abs(p-actual)
        order = np.argsort(p[:,0]); truth = np.argsort(actual[:,0]); k = min(3,len(p))
        def ranking(column):
            predicted_order = np.argsort(p[:,column], kind="stable")
            physical_order = np.argsort(actual[:,column], kind="stable")
            pairs = np.triu_indices(len(p), 1)
            physical_difference = actual[pairs[0],column] - actual[pairs[1],column]
            predicted_difference = p[pairs[0],column] - p[pairs[1],column]
            resolved = physical_difference != 0
            return {"predicted_order": predicted_order.tolist(), "physical_order": physical_order.tolist(),
                    "pairwise_order_error_fraction": (float(np.mean(np.sign(predicted_difference[resolved]) !=
                        np.sign(physical_difference[resolved]))) if resolved.any() else None),
                    "top1_regret": float(actual[predicted_order[0],column]-actual[physical_order[0],column]),
                    "top3_regret": float(actual[predicted_order[:k],column].min()-actual[physical_order[0],column])}
        temperature = ranking(0); pressure = ranking(1)
        result[str(family)] = {"mae": error.mean(axis=0).tolist(), "rmse": np.sqrt(np.mean(error**2,axis=0)).tolist(),
                               "normalized_mae": (error.mean(axis=0)/probe.target_scale).tolist(),
                               "temperature_ranking": temperature, "pressure_ranking": pressure,
                               "top1_regret_temperature": temperature["top1_regret"],
                               "top3_regret_temperature": temperature["top3_regret"],
                               "temperature_order": temperature["predicted_order"],
                               "physical_temperature_order": temperature["physical_order"]}
    return result


def fit_three_arms(aligned, permuted, targets, families, inherited_scales, *, organization_slots=tuple(range(6))):
    """Calibrate controls separately with identical 24 padded slots; costs three epochs."""
    aligned = np.asarray(aligned, float); permuted = np.asarray(permuted, float)
    if aligned.shape != permuted.shape or aligned.ndim != 2 or aligned.shape[1] != 24 or not np.array_equal(aligned[:,:18], permuted[:,:18]):
        raise ValueError("Aligned and permuted arms must share all non-organization inputs")
    slots = tuple(organization_slots)
    if len(set(slots)) != len(slots) or any(not isinstance(i, int) or i < 0 or i >= 6 for i in slots):
        raise ValueError("Organization slots must be unique indices from 0 to 5")
    columns = list(range(18)) + [18 + i for i in slots]
    aligned = aligned[:, columns]; permuted = permuted[:, columns]
    no_org = aligned.copy(); no_org[:,18:] = 0
    inputs = {"no_organization": no_org, "aligned": aligned, "permuted": permuted}
    probes = {key: fit_probe(x, targets, inherited_scales, family_ids=families) for key,x in inputs.items()}
    losses = {}
    targets = np.asarray(targets, float)
    for name, probe in probes.items():
        x = inputs[name]; y = targets / probe.target_scale
        initial = np.mean((y - y.mean(axis=0)) ** 2)
        final = np.mean(((probe.predict(x)-targets)/probe.target_scale)**2) + np.sum(probe.coefficients**2) / targets.shape[1]
        losses[name] = {"initial_constant_mse": float(initial), "final_mse_plus_l2": float(final)}
    return probes, losses


def useful_transfer(results: Mapping[str, Mapping], *, pressure_usable: bool) -> bool:
    """Apply declared equal-family temperature criterion; caller assesses pressure scale."""
    arms = ("no_organization", "aligned", "permuted")
    if set(results) != set(arms) or any(set(results[a]) != set(DEV) for a in arms):
        raise ValueError("Transfer disposition requires all four DEV families in all arms")
    scores = {a: np.array([results[a][f]["normalized_mae"][0] for f in DEV]) for a in arms}
    return bool(pressure_usable and all(scores["aligned"].mean() < scores[a].mean() and
                np.count_nonzero(scores["aligned"] < scores[a]) >= 3 for a in ("no_organization", "permuted")))


def request_distribution(*, useful: bool, fit_temperature_changes, predicted_changes, predicted_baseline_pressure: float,
                         physical_changes, physical_baseline_pressure: float, displacement_cost,
                         fit_temperature_scale: float, inherited_pressure_scale: float):
    """Nine-option request distributions and retrospective physical costs.

    Rows must be no-op first followed by the eight saved edits. FIT changes
    alone define requests. The no-op response is fixed to exactly zero.
    """
    if not useful:
        raise ValueError("Conditional distributions require measured useful transfer")
    fit = np.asarray(fit_temperature_changes, float); pred = np.asarray(predicted_changes, float).copy()
    physical = np.asarray(physical_changes, float).copy(); edit = np.asarray(displacement_cost, float)
    if pred.shape != (9,2) or physical.shape != (9,2) or edit.shape != (9,) or fit_temperature_scale <= 0 or inherited_pressure_scale <= 0 or not all(np.isfinite(x).all() for x in (fit,pred,physical,edit)):
        raise ValueError("Expected finite nine-option responses and positive FIT scales")
    if edit[0] != 0 or np.any(edit < 0):
        raise ValueError("No-op displacement cost must be zero and costs nonnegative")
    beneficial = -fit[fit < 0]
    if not len(beneficial):
        raise ValueError("FIT has no beneficial cooling edits")
    pred[0] = 0.; physical[0] = 0.
    targets = -np.quantile(beneficial, [0.25,0.75])
    pressure_scale = max(abs(predicted_baseline_pressure), inherited_pressure_scale)
    rng = np.random.default_rng(0); rows = []
    for target in targets:
        for multiplier in (1.01,1.05):
            limit = multiplier * predicted_baseline_pressure
            def costs(changes, baseline):
                mismatch = ((changes[:,0]-target)/fit_temperature_scale)**2
                violation = np.maximum(baseline+changes[:,1]-limit,0)/pressure_scale
                return mismatch + 10*violation**2 + .01*edit
            utility = costs(pred, predicted_baseline_pressure); actual = costs(physical, physical_baseline_pressure)
            weights = np.exp(-(utility - utility.min())); probabilities = weights/weights.sum()
            draws = rng.choice(9, size=3, replace=True, p=probabilities)
            rows.append(dict(target=float(target), pressure_limit=float(limit), probabilities=probabilities.tolist(),
                             expected_regret=float(probabilities @ actual - actual.min()),
                             best_of_three_regret=float(actual[draws].min()-actual.min()), draws=draws.tolist(),
                             physical_costs=actual.tolist(),
                             expected_absolute_temperature_error=float(probabilities @ np.abs(physical[:,0]-target)),
                             expected_pressure_violation=float(probabilities @ np.maximum(physical_baseline_pressure+physical[:,1]-limit,0)),
                             probability_noop=float(probabilities[0]), probability_single=float(probabilities[1:5].sum()),
                             probability_pair=float(probabilities[5:9].sum())))
    return rows


def probe_history(probe: RidgeProbe, predictions, targets, fit_mask, dev_mask) -> dict:
    """Evaluate retained constant/final state; never calibrates or changes coefficients."""
    pred = np.asarray(predictions, float); y = np.asarray(targets, float) / probe.target_scale
    fit = np.asarray(fit_mask, bool); dev = np.asarray(dev_mask, bool)
    if pred.shape != y.shape or y.ndim != 2 or fit.shape != (len(y),) or dev.shape != fit.shape or not fit.any() or not dev.any() or np.any(fit & dev):
        raise ValueError("Prediction and disjoint FIT/DEV masks must align")
    initial = np.broadcast_to(probe.intercept, y.shape)
    final = pred / probe.target_scale
    mse = lambda a, mask: float(np.mean((a[mask] - y[mask])**2))
    penalty = float(np.sum(probe.coefficients**2) / y.shape[1])
    train_mse = [mse(initial,fit),mse(final,fit)]
    return {"epochs": [0,1], "fit_mse": train_mse, "fit_ridge_l2": [0.,penalty],
            "fit_total_mse_plus_l2": [train_mse[0],train_mse[1]+penalty],
            "dev_mse": [mse(initial,dev),mse(final,dev)],
            "definition": "mean squared standardized target error over FIT examples and two targets plus mean per-target coefficient L2; intercept unpenalized"}


def plot_probe_history(histories: Mapping, path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(13,4))
    panels = (("fit_total_mse_plus_l2", "FIT total MSE + ridge L2"),
              ("fit_mse", "FIT standardized target MSE"), ("dev_mse", "DEV standardized target MSE (exposed)"))
    for arm, history in histories.items():
        for ax, (key, _) in zip(axes, panels):
            ax.plot(history["epochs"],history[key],marker="o",label=arm)
    for ax, (_, title) in zip(axes,panels):
        ax.set_title(title); ax.set_xlabel("Analytical calibration epoch"); ax.set_xticks([0,1]); ax.grid(); ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def request_distributions(*, predictions_by_arm: Mapping, predicted_baseline_pressure: float, **common):
    """Score all probe arms against one shared baseline, requests and physical catalogue."""
    if not predictions_by_arm:
        raise ValueError("At least one probe arm is required")
    return {name: request_distribution(predicted_changes=prediction,
            predicted_baseline_pressure=predicted_baseline_pressure, **common)
            for name,prediction in predictions_by_arm.items()}

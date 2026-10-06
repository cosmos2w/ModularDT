"""Declared original-TRAIN heat response refinement, without reference solves.

This objective is separate from the historical frozen-interface campaign. Its
response scales and calibration consume only the named fit families; primary
normalization and the selected primary population remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from channelthermal.data.datasets import GlobalChannelThermalDataset
from channelthermal.interaction_evidence.response_atlas import load_response_atlas_stencil
from channelthermal.interaction_evidence.response_dataset import ResponseStencil
from channelthermal.interaction_evidence.types import EvidenceSource, EvidenceSplit
from channelthermal.response_control.contracts import DesignInput, role_queries_from_stencil
from channelthermal.response_control.native import DifferentiableThermalOperator
from channelthermal.response_control.sampling import ReceiverSamplingConfig, sample_training_stencil
from channelthermal.response_control.thermal import pressure_drop_from_field
from honf_runtime.paths import resolve_path

from .campaign_null_response import NULL_CHANNELS, null_role_queries
from .campaign_work import CampaignForwardWork, measured_wrapper_calls, merge_forward_work

FIT_ANCHORS = ("0001", "0318", "0333", "0348")
DEVELOPMENT_ANCHORS = ("0304", "0320", "0335", "0350")
AUDIT_ANCHORS = ("0277", "0291", "0294", "0687")
HEAT_VARIANTS = ("heat_transfer_minus", "heat_transfer_plus")
THERMAL_CHANNELS = {
    "fluid_fields": "temperature", "interface": "T_surface", "solid_temperature": "temperature",
}
HEAT_NULL_CAPABILITY = "thermalchannel_analytic_heat_independent_flow_v1"
REFINEMENT_DEFAULTS = {
    "auxiliary_absolute_weight": .25,
    "q_proxy_weight": .25,
    "response_floor_fraction": .001,
    "heat_null_capability": HEAT_NULL_CAPABILITY,
    "null_fractions": [.05, .10],
    "null_fluid_queries": 256,
    "sampling_seed": 0,
}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _physical_signature(centers, heating, context) -> str:
    # The atlas was solved from these float32-rounded physical inputs.
    return _digest({"positions": np.asarray(centers, dtype=np.float32).tolist(),
                    "heating": np.asarray(heating, dtype=np.float32).tolist(),
                    "context": {key: float(value) for key, value in context.items()}})


def _record_signature(record) -> str:
    modules = record.design.active_modules
    return _physical_signature([item.position_xy for item in modules], [item.heating for item in modules],
                               record.context.values)


def _source_context(group) -> dict[str, float]:
    context = {key: float(value) for key, value in group["material_parameters"].attrs.items()}
    if "case_config_json" not in group:
        raise ValueError("Source physical provenance requires saved case configuration.")
    config = json.loads(group["case_config_json"][()].decode())
    context.update(domain_length_x=float(config["domain"]["lx"]), domain_length_y=float(config["domain"]["ly"]))
    return context


def response_refinement_declaration(atlas_directory: str | Path) -> dict[str, Any]:
    """Return the explicit campaign fragments needed for preparation/calibration."""
    directory = Path(atlas_directory).resolve()
    return {"response_addendum": {
        "policy_version": "train_heat_families_v1", "fit_anchor_ids": list(FIT_ANCHORS),
        "response_development_anchor_ids": list(DEVELOPMENT_ANCHORS), "fixed_audit_anchor_ids": list(AUDIT_ANCHORS),
        "variant_labels": list(HEAT_VARIANTS),
        "response_stencils": [str(directory / f"train_{anchor}_responses.npz") for anchor in FIT_ANCHORS],
    }, "response_refinement": {**REFINEMENT_DEFAULTS, "calibration": None}}


def validate_response_addendum(declaration: Mapping[str, Any], dataset_config=None) -> tuple[str, ...]:
    """Reject substitutions and withheld-family labels before any scale fitting."""
    if "response_addendum" in declaration:
        declaration = declaration["response_addendum"]
    expected = {
        "policy_version": "train_heat_families_v1", "fit_anchor_ids": list(FIT_ANCHORS),
        "response_development_anchor_ids": list(DEVELOPMENT_ANCHORS), "fixed_audit_anchor_ids": list(AUDIT_ANCHORS),
        "variant_labels": list(HEAT_VARIANTS),
    }
    for key, value in expected.items():
        if declaration.get(key) != value:
            raise ValueError(f"Response addendum requires declared {key}={value!r}.")
    if set(declaration) != {*expected, "response_stencils"}:
        raise ValueError("Unknown response addendum fields.")
    if dataset_config is not None:
        from channelthermal.data.development_split import development_case_ids

        development = dataset_config.get("development_subset")
        if development is None:
            raise ValueError("Response addendum requires explicit unchanged fixed25 primary identity.")
        train = development_case_ids(development, "train")
        validation = development_case_ids(development, "test")
        if len(train) != 150 or len(validation) != 22 or "0273" in validation:
            raise ValueError("Response addendum requires fixed25 primary150/validation22 and duplicate exclusion.")
    paths = declaration.get("response_stencils", [])
    if isinstance(paths, str) or len(paths) != len(FIT_ANCHORS) or len(set(paths)) != len(paths):
        raise ValueError("Response addendum requires four unique original-TRAIN atlas paths.")
    return tuple(str(resolve_path(path)) for path in paths)


def validate_response_refinement_declaration(settings, config=None):
    """Validate the maintained Thermal-only objective without needing fitted coefficients."""
    validate_response_addendum(settings)
    declared = settings.get("response_refinement")
    if not isinstance(declared, dict) or set(declared) != {*REFINEMENT_DEFAULTS, "calibration"}:
        raise ValueError("Response refinement requires the complete explicit objective declaration.")
    for key, value in REFINEMENT_DEFAULTS.items():
        if declared[key] != value:
            raise ValueError(f"Matched response refinement requires {key}={value!r}.")
    if config is not None and getattr(config, "channelthermal", None) is None:
        raise ValueError("Heat-null capability belongs to the ThermalChannel adapter only.")
    if declared["calibration"] is not None:
        for key in ("response_coefficient", "null_coefficient"):
            value = float(declared["calibration"].get(key, 0.))
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Response refinement calibration coefficients must be finite positive values.")
    return declared


def load_refinement_stencils(declaration: Mapping[str, Any]):
    """Load only existing heat states; validate source metadata and receiver joins."""
    result = []
    for anchor, path in zip(FIT_ANCHORS, validate_response_addendum(declaration)):
        stencil, metadata = load_response_atlas_stencil(path)
        family = "duplicate_family:0001+0273" if anchor == "0001" else f"stored_family:{anchor}"
        if (stencil.anchor_id != anchor or stencil.physical_family_id != family
                or stencil.split is not EvidenceSplit.TRAIN or stencil.source is not EvidenceSource.REFERENCE_SOLVER
                or metadata.get("anchor_id") != anchor or metadata.get("family_id") != family
                or metadata.get("split") != "train"):
            raise ValueError(f"Response fit-family/source identity mismatch for {anchor}.")
        if not set(HEAT_VARIANTS) <= set(stencil.variants):
            raise ValueError(f"Required saved heat states unavailable for {anchor}.")
        pair = ResponseStencil(stencil.baseline, {label: stencil.variants[label] for label in HEAT_VARIANTS})
        for label, record in zip(("baseline", *HEAT_VARIANTS), pair.records):
            evidence = metadata.get("records", {}).get(label, {})
            if evidence.get("status") != "converged" or evidence.get("source") != "reference_solver":
                raise ValueError(f"Stored reference validity missing for {anchor}/{label}.")
            if not evidence.get("runtime", {}).get("converged") or not Path(evidence.get("case_dir", "")).is_dir():
                raise ValueError(f"Converged raw-source provenance unavailable for {anchor}/{label}.")
            base_modules, trial_modules = pair.baseline.design.active_modules, record.design.active_modules
            if any(base.position_xy != trial.position_xy for base, trial in zip(base_modules, trial_modules)):
                raise ValueError("Heat refinement cannot use a geometry response.")
            if label != "baseline":
                increment = np.asarray([trial.heating - base.heating for base, trial in zip(base_modules, trial_modules)])
                if np.count_nonzero(increment) != 2 or not np.isclose(increment.sum(), 0., atol=1e-6):
                    raise ValueError("Heat refinement requires nonzero two-module balanced transfers.")
            for role in THERMAL_CHANNELS:
                pair.common_mask(role)  # Exact query, module identity, mask, quadrature and coordinate joins.
        minus, plus = (np.asarray([module.heating for module in pair.variants[label].design.active_modules])
                       for label in HEAT_VARIANTS)
        base_heat = np.asarray([module.heating for module in pair.baseline.design.active_modules])
        if not np.allclose(minus + plus, 2 * base_heat, atol=1e-6, rtol=0.):
            raise ValueError("Saved heat directions must be opposite transfers around their own baseline.")
        result.append((pair, metadata))
    if len({stencil.physical_family_id for stencil, _ in result}) != len(result):
        raise ValueError("Response fit physical families must be unique.")
    return tuple(result)


def response_addendum_identity(declaration):
    """Bind immutable saved inputs/targets even when their file paths do not change."""
    loaded = load_refinement_stencils(declaration)
    return [{"anchor_id": stencil.anchor_id, "physical_family_id": stencil.physical_family_id,
             "atlas_npz_sha256": _file_digest(Path(stencil.baseline.provenance["atlas_npz"])),
             "atlas_json_sha256": _file_digest(Path(stencil.baseline.provenance["atlas_json"]))}
            for stencil, _ in loaded]


def response_eligibility_inventory(declaration, dataset_path, primary_manifest) -> dict[str, Any]:
    """Inspect saved records and original input partitions, never invoking a solver."""
    loaded = load_refinement_stencils(declaration)
    primary_train = set(primary_manifest["partitions"]["train"]["case_ids"])
    primary_validation = set(primary_manifest["partitions"]["test"]["case_ids"])
    rows, signatures = [], {}
    with h5py.File(dataset_path, "r") as h5:
        source_splits = {case.decode() if isinstance(case, bytes) else str(case):
                         split.decode() if isinstance(split, bytes) else str(split)
                         for case, split in zip(h5["case_ids"], h5["splits"])}
        for anchor in (*FIT_ANCHORS, *DEVELOPMENT_ANCHORS, *AUDIT_ANCHORS, "0273"):
            group = h5["cases"][anchor]
            if source_splits[anchor] != str(group.attrs["split"]):
                raise ValueError("Original partition catalogue disagrees with saved case source metadata.")
            present = np.asarray(group["module_present"]) > .5
            centers, heat = np.asarray(group["module_centers"])[present], np.asarray(group["heat_powers"])[present]
            context = _source_context(group)
            signatures[anchor] = _physical_signature(centers, heat, context)
            rows.append({"anchor_id": anchor, "source_partition": str(group.attrs["split"]),
                         "module_count": int(present.sum()), "input_signature": signatures[anchor],
                         "primary_train": anchor in primary_train, "primary_validation": anchor in primary_validation,
                         "source_case_dir": str(group.attrs["source_case_dir"])})
        by_anchor = {row["anchor_id"]: row for row in rows}
        for stencil, _ in loaded:
            anchor = stencil.anchor_id
            if by_anchor[anchor]["source_partition"] != "train" or _record_signature(stencil.baseline) != signatures[anchor]:
                raise ValueError(f"Atlas baseline differs from original TRAIN rounded inputs for {anchor}.")
            if signatures[anchor] in {signatures[value] for value in (*DEVELOPMENT_ANCHORS, *AUDIT_ANCHORS)}:
                raise ValueError("Fit family physically overlaps a withheld response family.")
        if signatures["0001"] != signatures["0273"] or "0273" in primary_validation:
            raise ValueError("Expected 0001/0273 duplicate relationship or exclusion failed.")
    fit_rows = []
    for stencil, metadata in loaded:
        path = Path(stencil.baseline.provenance["atlas_npz"])
        fit_rows.append({"anchor_id": stencil.anchor_id, "physical_family_id": stencil.physical_family_id,
                         "module_ids": list(stencil.baseline.design.active_module_ids),
                         "extra_auxiliary_anchor": stencil.anchor_id not in primary_train,
                         "atlas_npz": str(path.resolve()), "atlas_npz_sha256": _file_digest(path),
                         "atlas_json_sha256": _file_digest(path.with_suffix(".json")),
                         "heat_labels": list(HEAT_VARIANTS), "receiver_counts": {
                             role: output.values.shape[0] for role, output in stencil.baseline.output.roles.items()},
                         "source": stencil.source.value, "source_status": "converged",
                         "baseline_source": metadata["records"]["baseline"]["case_dir"]})
    result = {"policy_version": declaration["policy_version"], "primary_manifest_fingerprint": primary_manifest["manifest_sha256"],
              "primary_training_cases": len(primary_train), "primary_validation_cases": len(primary_validation),
              "extra_auxiliary_anchor_families": sum(row["extra_auxiliary_anchor"] for row in fit_rows),
              "fit_families": fit_rows, "source_input_inventory": rows,
              "response_development_exposure": "original TRAIN; response labels withheld in refinement; historical exposure retained",
              "duplicate_relationship": "0001 original TRAIN equals 0273 original TEST physical inputs; 0273 excluded from primary validation",
              "solver_attempts": 0, "cumulative_solver_attempts": "326/326"}
    result["inventory_fingerprint"] = _digest(result)
    return result


def primary_role_scales(stats) -> dict[str, np.ndarray]:
    result = {
        "fluid_fields": np.asarray(stats["field_std_by_channel"], dtype=np.float64).reshape(-1),
        "interface": np.asarray(stats.get("interface_targets_std", stats.get("interface_target_std")), dtype=np.float64).reshape(-1),
        "solid_temperature": np.asarray(stats["internal_temperature_std"], dtype=np.float64).reshape(-1),
    }
    if any(not np.isfinite(value).all() or np.any(value <= 0) for value in result.values()):
        raise ValueError("Auxiliary scales require finite positive primary-TRAIN standard deviations.")
    return result


def _weighted_numpy(values, role, channel, mask):
    weights = np.asarray(role.quadrature_weights) * mask[:, channel]
    if weights.sum() <= 0:
        raise ValueError("Response scale role/channel has no observed positive quadrature support.")
    return float((np.where(mask[:, channel], values[:, channel], 0.) * weights).sum() / weights.sum())


def fit_response_scales(stencils: Sequence[ResponseStencil], primary_stats, *, floor_fraction=.001) -> dict[str, Any]:
    """Equal-family/equal-direction RMS using only declared original-TRAIN pairs."""
    if tuple(stencil.anchor_id for stencil in stencils) != FIT_ANCHORS:
        raise ValueError("Response scale fitting requires exactly the declared four fit anchors in order.")
    if not math.isfinite(float(floor_fraction)) or floor_fraction <= 0:
        raise ValueError("Response scale floor fraction must be finite positive.")
    primary = primary_role_scales(primary_stats)
    channels = {**THERMAL_CHANNELS, "q_normal": "q_normal"}
    result = {}
    for name, channel_name in channels.items():
        role_name = "interface" if name == "q_normal" else name
        response_squared, roundoff_squared = [], []
        for stencil in stencils:
            if stencil.split is not EvidenceSplit.TRAIN or tuple(stencil.variants) != HEAT_VARIANTS:
                raise ValueError("Response scales exclude development/audit and non-heat labels.")
            baseline = stencil.baseline.output.roles[role_name]
            channel = baseline.channel_names.index(channel_name)
            for label in HEAT_VARIANTS:
                change = stencil.finite_change(label, role_name)
                response_squared.append(_weighted_numpy(change.delta ** 2, baseline, channel, change.valid_mask))
                trial = stencil.variants[label].output.roles[role_name]
                # Conservative saved float32 spacing indicator, not physical/grid uncertainty.
                spacing_squared = (np.spacing(baseline.values.astype(np.float32)).astype(np.float64) ** 2
                                   + np.spacing(trial.values.astype(np.float32)).astype(np.float64) ** 2)
                roundoff_squared.append(_weighted_numpy(spacing_squared, baseline, channel, change.valid_mask))
        rms = math.sqrt(float(np.mean(response_squared)))
        indicator = math.sqrt(float(np.mean(roundoff_squared)))
        primary_floor = float(floor_fraction) * float(primary[role_name][channel])
        floor = max(indicator, primary_floor)
        result[name] = {"channel": channel_name, "response_rms": rms, "storage_roundoff_indicator": indicator,
                        "primary_std_floor": primary_floor, "floor": floor, "scale": max(rms, floor)}
    summary = {"roles": result, "fit_anchors": list(FIT_ANCHORS), "variant_labels": list(HEAT_VARIANTS),
            "basis": "equal family/direction; positive quadrature normalized within role/channel; saved float32 spacing floor",
            "physical_error_certificate": False}
    summary["scale_fingerprint"] = _digest(summary)
    return summary


def normalized_role_error(predicted, target, role, scales, *, channels=None, valid_mask=None):
    """Mean per-channel squared error after independently normalizing support."""
    indices = list(range(predicted.shape[1])) if channels is None else list(channels)
    target = torch.as_tensor(np.array(target, copy=True), device=predicted.device, dtype=predicted.dtype)
    mask = role.valid_mask if valid_mask is None else valid_mask
    mask = torch.as_tensor(np.array(mask, copy=True), device=predicted.device, dtype=torch.bool)
    weights = torch.as_tensor(np.array(role.quadrature_weights, copy=True), device=predicted.device, dtype=predicted.dtype)
    scales = torch.as_tensor(scales, device=predicted.device, dtype=predicted.dtype).reshape(-1)
    if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
        raise ValueError("Auxiliary role scales must be finite positive values.")
    terms = []
    for local, index in enumerate(indices):
        w = weights * mask[:, index]
        if not bool(w.sum() > 0):
            raise ValueError("Auxiliary role/channel has no positive observed support.")
        scale = scales[index] if scales.numel() == predicted.shape[1] else scales[local]
        error = torch.where(mask[:, index], predicted[:, index] - target[:, index], 0.)
        terms.append(((error / scale).square() * w).sum() / w.sum())
    return torch.stack(terms).mean()


def response_pair_losses(pair, before, after, primary_stats, response_scales, *, q_proxy_weight=.25):
    """Both-state case/role/channel means and three thermal change-role means."""
    label = next(iter(pair.variants))
    primary = primary_role_scales(primary_stats)
    absolute = []
    for record, predictions in ((pair.baseline, before), (pair.variants[label], after)):
        role_terms = [normalized_role_error(predictions[name], role.values, role, primary[name])
                      for name, role in record.output.roles.items()]
        absolute.append(torch.stack(role_terms).mean())
    thermal = []
    for name, channel_name in THERMAL_CHANNELS.items():
        role = pair.baseline.output.roles[name]
        change = pair.finite_change(label, name)
        thermal.append(normalized_role_error(after[name] - before[name], change.delta, role,
                       [response_scales["roles"][name]["scale"]], channels=[role.channel_names.index(channel_name)],
                       valid_mask=change.valid_mask))
    role = pair.baseline.output.roles["interface"]
    change = pair.finite_change(label, "interface")
    proxy = normalized_role_error(after["interface"] - before["interface"], change.delta, role,
                    [response_scales["roles"]["q_normal"]["scale"]], channels=[role.channel_names.index("q_normal")],
                    valid_mask=change.valid_mask)
    thermal_mean = torch.stack(thermal).mean()
    return {"absolute": torch.stack(absolute).mean(), "response": thermal_mean + q_proxy_weight * proxy,
            "thermal_response": thermal_mean, "q_proxy_response": proxy}


def feasible_heat_transfer(heat, present, *, seed, fraction, lower_heat, upper_heat):
    """Deterministic input-only transfer; M1 checks heat dependency without fixed sum."""
    heat = np.asarray(heat, dtype=np.float64)
    active = np.flatnonzero(np.asarray(present) > .5)
    if fraction not in (.05, .10) or not np.isfinite(heat).all() or lower_heat > upper_heat:
        raise ValueError("Refinement null requires finite heat and declared .05/.10 steps.")
    rng = np.random.default_rng(seed)
    candidates = [(int(i), int(j)) for i in active for j in active if i != j]
    if active.size == 1:
        candidates = [(int(active[0]), int(active[0]))]
    for position in rng.permutation(len(candidates)):
        donor, receiver = candidates[int(position)]
        if donor == receiver:
            distances = (upper_heat - heat[donor], heat[donor] - lower_heat)
            sign = 1. if distances[0] >= distances[1] else -1.
            bound = max(distances)
        else:
            sign = 1.
            bound = min(heat[donor] - lower_heat, upper_heat - heat[receiver])
        amplitude = float(fraction * bound)
        if amplitude <= 1e-8:
            continue
        direction = np.zeros_like(heat)
        if donor == receiver:
            direction[donor] = sign * amplitude
        else:
            direction[donor], direction[receiver] = -amplitude, amplitude
        return direction, {"donor": donor, "receiver": receiver, "amplitude": amplitude,
                           "feasible_bound": float(bound), "fraction": fraction,
                           "fixed_total": donor != receiver, "signed_heat_increment": direction.tolist()}
    return None


def gradient_component_norms(primary_loss, components, model) -> dict[str, float]:
    """Calibration reads gradients without populating .grad or changing model state."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Refinement calibration requires the declared nonempty trainable scope.")
    result = {}
    for name, loss in {"primary": primary_loss, "response": components["response"], "null": components["null"]}.items():
        grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        norm = math.sqrt(sum(float(gradient.detach().double().square().sum()) for gradient in grads if gradient is not None))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError(f"Disconnected or invalid {name} calibration gradient; cannot assign zero coefficient.")
        result[name] = norm
    return result


def pooled_gradient_coefficients(measurements) -> dict[str, Any]:
    """One common pair from pooled RMS initial gradient norms across both arms."""
    if len(measurements) != 8:
        raise ValueError("Shared calibration must span four TRAIN fit families in each of two arms.")
    if len({row.get("arm") for row in measurements}) != 2:
        raise ValueError("Shared calibration requires exactly two arms.")
    for arm in {row["arm"] for row in measurements}:
        rows = [row for row in measurements if row["arm"] == arm]
        if len(rows) != len(FIT_ANCHORS) or {row.get("anchor_id") for row in rows} != set(FIT_ANCHORS):
            raise ValueError("Coefficient calibration cannot consume withheld response families.")
    norms = {}
    for key in ("primary", "response", "null"):
        values = [float(row[key]) for row in measurements]
        if any(not math.isfinite(value) or value <= 1e-12 for value in values):
            raise ValueError("Calibration component norms must be finite and nonzero.")
        norms[key] = math.sqrt(float(np.mean(np.square(values))))
    return {"response_coefficient": .25 * norms["primary"] / norms["response"],
            "null_coefficient": .10 * norms["primary"] / norms["null"], "pooled_gradient_norms": norms,
            "response_gradient_budget": .25, "null_gradient_budget": .10,
            "basis": "pooled RMS initial norms in identical declared trainable scope; TRAIN only",
            "measurements": list(measurements)}


class NativeResponseRefinement:
    """One fit pair and one diverse primary-input null at the last native boundary."""

    def __init__(self, model, dataset, dataset_config, settings):
        self.model, self.dataset_config = model, dataset_config
        validate_response_refinement_declaration(settings, model.config)
        validate_response_addendum(settings, dataset_config)
        self.settings = {**REFINEMENT_DEFAULTS, **settings.get("response_refinement", {})}
        for key, value in REFINEMENT_DEFAULTS.items():
            if self.settings[key] != value:
                raise ValueError(f"Matched response refinement requires {key}={value!r}.")
        if getattr(model.config, "channelthermal", None) is None:
            raise ValueError("Heat-null capability belongs to the ThermalChannel adapter only.")
        self.loaded = load_refinement_stencils(settings["response_addendum"])
        self.stats = dataset.normalizer.stats
        self.scales = fit_response_scales([stencil for stencil, _ in self.loaded], self.stats)
        self.dataset = GlobalChannelThermalDataset(dataset.path, split="train", points_per_case=256,
            normalize_inputs=False, normalize_targets=False, random_point_sampling=False,
            normalizer=dataset.normalizer, case_ids=dataset.selected_case_ids)
        if len(self.dataset.selected_case_ids) != 150:
            raise ValueError("Response refinement null exposure requires the unchanged primary150.")
        from channelthermal.data.development_split import development_case_ids

        if list(self.dataset.selected_case_ids) != list(development_case_ids(dataset_config["development_subset"], "train")):
            raise ValueError("Heat-null inputs must be exactly the selected primary TRAIN cases.")
        values = []
        for case_id in self.dataset.selected_case_ids:
            group = self.dataset.h5["cases"][case_id]
            config = json.loads(group["case_config_json"][()].decode())
            if config.get("flow", {}).get("flow_model") != "analytic_wake":
                raise ValueError("Heat-null dependency is verified only for the inspected analytic Thermal generator.")
            present = np.asarray(group["module_present"]) > .5
            values.extend(np.asarray(group["heat_powers"])[present].tolist())
        self.lower_heat, self.upper_heat = max(0., float(min(values))), float(max(values))
        self.sampling = ReceiverSamplingConfig(max_fluid_queries=1024, solid_queries_per_module=128,
                                               hot_solid_points_per_module=16, random_seed=0)
        self.sampled = tuple(sample_training_stencil(stencil, config=self.sampling).stencil for stencil, _ in self.loaded)
        self.calibration = self.settings.get("calibration")

    def freeze_calibration(self, calibration):
        for key in ("response_coefficient", "null_coefficient"):
            if not math.isfinite(float(calibration[key])) or float(calibration[key]) <= 0:
                raise ValueError("Refinement coefficients must be finite positive calibrated values.")
        if self.calibration is not None and self.calibration != calibration:
            raise ValueError("Refinement coefficients are frozen, including on resume.")
        self.calibration = dict(calibration)

    def _operator(self, sample, query_batch_size):
        return DifferentiableThermalOperator(self.model, sample, dataset_config=self.dataset_config,
                    normalization_stats=self.stats, query_batch_size=query_batch_size)

    def components(self, epoch):
        """Build unweighted differentiable components; epoch is genuine refinement age."""
        if epoch < 1:
            raise ValueError("Refinement response age starts at one.")
        started = time.perf_counter()
        family_index = (epoch - 1) % len(self.sampled)
        label = HEAT_VARIANTS[((epoch - 1) // len(self.sampled)) % len(HEAT_VARIANTS)]
        family = self.sampled[family_index]
        pair = ResponseStencil(family.baseline, {label: family.variants[label]})
        parameter = next(self.model.parameters())
        device = parameter.device
        queries = role_queries_from_stencil(pair, device=device)
        base_design = DesignInput.from_state(pair.baseline.design, device=device)
        trial_design = DesignInput.from_state(pair.variants[label].design, device=device)
        operator = self._operator(self.dataset[0], 1024)
        records = {}
        with CampaignForwardWork(self.model.core) as measured:
            before = operator(base_design, dict(pair.baseline.context.values), queries).role_values
            after = operator(trial_design, dict(pair.baseline.context.values), queries).role_values
        merge_forward_work(records, measured.records)
        components = response_pair_losses(pair, before, after, self.stats, self.scales)
        response_seconds = time.perf_counter() - started
        null, null_metrics, null_records = self.null_component(epoch)
        components["null"] = null
        merge_forward_work(records, null_records)
        count = measured_wrapper_calls(records)
        metrics = {"response_family": pair.physical_family_id, "response_anchor": pair.anchor_id,
                   "response_variant": label, "response_loss": float(components["response"].detach()),
                   "response_thermal_loss": float(components["thermal_response"].detach()),
                   "response_q_proxy_loss": float(components["q_proxy_response"].detach()),
                   "response_auxiliary_absolute_loss": float(components["absolute"].detach()),
                   "auxiliary_absolute_loss": float(components["absolute"].detach()),
                   "thermal_response_loss": float(components["thermal_response"].detach()),
                   "q_proxy_response_loss": float(components["q_proxy_response"].detach()),
                   "response_examples": 2, "response_wrapper_calls": 2,
                   "response_queries": 2 * sum(query.query_features.shape[0] for query in queries.values()),
                   "refinement_total_wrapper_calls": count if count is not None else 4,
                   "response_seconds": response_seconds, "response_forward_work": measured.records,
                   "refinement_auxiliary_seconds": time.perf_counter() - started,
                   "refinement_auxiliary_forward_work": records,
                   "response_reference": "existing rounded-input TRAIN atlas; q-normal proxy separately weighted",
                   "response_scale_basis": self.scales, **null_metrics}
        return components, metrics

    def null_component(self, epoch):
        started = time.perf_counter()
        fraction = self.settings["null_fractions"][(epoch - 1) % 2]
        count = len(self.dataset.selected_case_ids)
        skipped = []
        for offset in range(count):
            index = (epoch - 1 + offset) % count
            sample = self.dataset[index]
            structure = sample["structure"]
            transfer = feasible_heat_transfer(structure["heat_powers"], structure["module_present"],
                    seed=(epoch - 1) // count * count + index, fraction=fraction,
                    lower_heat=self.lower_heat, upper_heat=self.upper_heat)
            if transfer is None:
                skipped.append(self.dataset.selected_case_ids[index])
                continue
            direction, description = transfer
            break
        else:
            raise ValueError("No non-negligible feasible TRAIN heat-null perturbation.")
        parameter = next(self.model.parameters())
        device, dtype = parameter.device, parameter.dtype
        material = np.asarray(structure["material_params"]).reshape(-1)
        context = {name: float(material[i]) for i, name in enumerate(
            ("nu", "solid_alpha", "fluid_alpha", "solid_k", "fluid_k", "module_radius"))}
        context.update({name: float(np.asarray(structure[name]).reshape(-1)[0]) for name in
            ("re", "u_in", "domain_length_x", "domain_length_y")})
        design = DesignInput(torch.as_tensor(structure["module_centers"], dtype=dtype, device=device),
            torch.as_tensor(structure["heat_powers"], dtype=dtype, device=device),
            torch.as_tensor(structure["module_present"] > .5, device=device))
        trial = DesignInput(design.module_positions, design.module_heating + torch.as_tensor(direction, dtype=dtype, device=device),
                            design.module_present)
        queries = null_role_queries(sample, self.model.config.channelthermal.field_names, device=device,
                                    fluid_count=self.settings["null_fluid_queries"])
        operator = self._operator(sample, 256)
        with CampaignForwardWork(self.model.core) as measured:
            before = operator(design, context, queries).role_values["fluid_fields"]
            after = operator(trial, context, queries).role_values["fluid_fields"]
        indices = [queries["fluid_fields"].channel_names.index(name) for name in NULL_CHANNELS]
        increment = after[:, indices] - before[:, indices]
        scales = torch.as_tensor(self.stats["field_std_by_channel"], dtype=dtype, device=device).flatten()[indices]
        # Normalize the declared dimensionless fraction: .05/.10 do not silently alter the loss definition.
        loss = (increment / scales / fraction).square().mean()
        raw_rms = increment.detach().square().mean(0).sqrt()
        description.update(case_id=self.dataset.selected_case_ids[index], negligible_step_skips=skipped,
            replacement_rule="next primary case in fixed order; same epoch fraction; fail after one full cycle",
            heat_bounds=[self.lower_heat, self.upper_heat],
            raw_increment_rms={name: float(value) for name, value in zip(NULL_CHANNELS, raw_rms)},
            increment_per_unit_heat_rms={name: float(value / description["amplitude"]) for name, value in zip(NULL_CHANNELS, raw_rms)})
        pressure = pressure_drop_from_field(after, queries["fluid_fields"], trial, context) - pressure_drop_from_field(
            before, queries["fluid_fields"], design, context)
        description["pressure_drop_increment"] = float(pressure.detach())
        return loss, {"heat_null_loss": float(loss.detach()), "heat_null_cases": [description],
                      "null_step_fraction": fraction, "null_step_heat": description["amplitude"],
                      "null_case_id": description["case_id"],
                      "heat_null_wrapper_calls": 2, "heat_null_primary_fluid_queries": 512,
                      "heat_null_role_queries": 2 * sum(query.query_features.shape[0] for query in queries.values()),
                      "heat_null_seconds": time.perf_counter() - started,
                      "heat_null_forward_work": measured.records,
                      "heat_null_scope": "one primary TRAIN input; both calls differentiable; no thermal trial label or solver",
                      "heat_null_fraction_normalization": "increment / primary_std / declared_fraction"}, measured.records

    def __call__(self, epoch, native_loss, accumulation_weight):
        if self.calibration is None:
            raise ValueError("Freeze pooled two-arm TRAIN calibration before refinement updates.")
        self.freeze_calibration(self.calibration)
        if not math.isfinite(float(accumulation_weight)) or accumulation_weight <= 0:
            raise ValueError("Auxiliary accumulation weight must be finite and positive.")
        components, metrics = self.components(epoch)
        total = (self.settings["auxiliary_absolute_weight"] * components["absolute"]
                 + self.calibration["response_coefficient"] * components["response"]
                 + self.calibration["null_coefficient"] * components["null"])
        metrics.update(response_coefficient=self.calibration["response_coefficient"],
                       heat_null_coefficient=self.calibration["null_coefficient"],
                       response_auxiliary_absolute_weight=self.settings["auxiliary_absolute_weight"])
        # run_epoch multiplies the entire last microbatch loss by this weight;
        # invert it here so the complete once-per-epoch auxiliary term survives.
        return total / accumulation_weight, metrics

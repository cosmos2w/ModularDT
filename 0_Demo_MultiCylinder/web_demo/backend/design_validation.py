from __future__ import annotations

import math
from typing import Dict, List

from model_registry import resolve_phase_bin_config
from schemas import DesignRequest, ValidationResult


def model_limits(config: Dict) -> Dict[str, float]:
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    return {
        "max_num_cylinders": int(
            model_cfg.get("max_num_cylinders", dataset_cfg.get("max_num_cylinders", 8))
        ),
        "domain_length_x": float(model_cfg.get("domain_length_x", 24.0)),
        "domain_length_y": float(model_cfg.get("domain_length_y", 12.0)),
        "re_scale": float(model_cfg.get("re_scale", 200.0)),
    }


def _periodic_distance(ax: float, ay: float, bx: float, by: float, lx: float, ly: float) -> float:
    dx = abs(ax - bx)
    dy = abs(ay - by)
    dx = min(dx, lx - dx)
    dy = min(dy, ly - dy)
    return math.sqrt(dx * dx + dy * dy)


def validate_design(request: DesignRequest, config: Dict, manifest_entry: Dict | None = None) -> ValidationResult:
    limits = model_limits(config)
    max_num_cylinders = int(limits["max_num_cylinders"])
    lx = float(limits["domain_length_x"])
    ly = float(limits["domain_length_y"])
    warnings: List[str] = []
    valid = True
    phase_cfg = resolve_phase_bin_config(config, manifest_entry)
    requested_phase_bins = int(request.phase_bins)
    max_phase_bins = int(phase_cfg["max_phase_bins"])
    phase_bin_policy = str(phase_cfg["phase_bin_policy"])
    effective_phase_bins = requested_phase_bins

    if len(request.cylinders) > max_num_cylinders:
        valid = False
        warnings.append(f"Too many cylinders: {len(request.cylinders)} > max_num_cylinders={max_num_cylinders}.")
    if not math.isfinite(request.re) or request.re <= 0:
        valid = False
        warnings.append("Re must be finite and positive.")
    if requested_phase_bins > max_phase_bins:
        if phase_bin_policy == "reject":
            valid = False
            warnings.append(f"Requested phase bins {requested_phase_bins} exceeds configured max {max_phase_bins}.")
        else:
            effective_phase_bins = max_phase_bins
            warnings.append(
                f"Requested phase bins {requested_phase_bins} exceeds configured max {max_phase_bins}; "
                f"inference will use {max_phase_bins}."
            )

    for idx, cyl in enumerate(request.cylinders):
        if not math.isfinite(cyl.x) or not math.isfinite(cyl.y):
            valid = False
            warnings.append(f"Cylinder C{idx} has non-finite coordinates.")
            continue
        if not (0.0 <= cyl.x < lx):
            valid = False
            warnings.append(f"Cylinder C{idx} x={cyl.x} is outside [0, {lx}).")
        if not (0.0 <= cyl.y < ly):
            valid = False
            warnings.append(f"Cylinder C{idx} y={cyl.y} is outside [0, {ly}).")

    manifest_entry = manifest_entry or {}
    re_min = manifest_entry.get("expected_re_min")
    re_max = manifest_entry.get("expected_re_max")
    if re_min is not None and request.re < float(re_min):
        warnings.append(f"Re={request.re} is below the expected range minimum {re_min}.")
    if re_max is not None and request.re > float(re_max):
        warnings.append(f"Re={request.re} is above the expected range maximum {re_max}.")

    radius = float(manifest_entry.get("cylinder_radius", 0.5))
    min_gap = float(manifest_entry.get("min_gap", 0.0))
    min_allowed = 2.0 * radius + min_gap
    for i, a in enumerate(request.cylinders):
        for j in range(i + 1, len(request.cylinders)):
            b = request.cylinders[j]
            dist = _periodic_distance(a.x, a.y, b.x, b.y, lx, ly)
            if dist < min_allowed:
                warnings.append(
                    f"Cylinders C{i} and C{j} may overlap under periodic geometry "
                    f"(distance={dist:.3g}, minimum recommended={min_allowed:.3g})."
                )

    return ValidationResult(
        valid=valid,
        warnings=warnings,
        max_num_cylinders=max_num_cylinders,
        domain_length_x=lx,
        domain_length_y=ly,
        requested_phase_bins=requested_phase_bins,
        effective_phase_bins=effective_phase_bins,
        max_phase_bins=max_phase_bins,
        phase_bin_policy=phase_bin_policy,  # type: ignore[arg-type]
    )

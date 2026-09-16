#!/usr/bin/env python3
"""Derivative diagnostics for the routed pairwise HONF experiment.

This module contains the small numerical subgraph checks and the bounded
real-case probes required by Goal 1.  It is intentionally evaluation-only:
the real-case helpers load a labelled checkpoint, run the ordinary
ThermalChannel P0/P1/P2 forward path, and never modify model parameters or
training state.

The two kinds of source influence are kept separate throughout this file.
``routed_fine_source_probe`` holds the prepared state, route identities,
selected priors, and contextual tensors fixed while changing one fine source
value.  ``position_derivative_probe`` and ``heat_derivative_probe`` rerun the
full physical loop, so their total influence may include preparation, common
coarse/local paths, and port feedback.  Agreement between AD and finite
differences is an implementation self-check; it is not a physical derivative
certificate.

Run from ``HONF_Proj/``.  With no checkpoint the command runs only the small
float64 route and support-transition fixtures::

    python tools/diagnostics/routing_derivative_checks.py toy \
        --output diagnostics/generated/routing_derivative_toy.json

For the real anchor rows use explicit checkpoint labels::

    python tools/diagnostics/routing_derivative_checks.py anchors \
        --checkpoint run2000=/path/to/epoch_0500_model.pt \
        --case-id 0273 --case-id 0298 --device cuda:2 \
        --output diagnostics/generated/run2000_derivatives.json

The real probes use position steps ``0.01 R`` and ``0.005 R`` and heat steps
``0.01`` and ``0.005`` in the checkpoint's normalized input heat units.  The
heat convention is explicit because the design documents prescribe the pair
of steps for position but do not prescribe an independent heat scale.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "Case_ThermalChannel" / "src",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from channelthermal.evaluation.loading import make_batch  # type: ignore[import-not-found]
from channelthermal.evaluation.prepared import select_sample  # type: ignore[import-not-found]

# The maintained Stage-3 helpers are the canonical case loader and geometric
# validity implementation.  They are imported only for diagnostics; this
# file owns the derivative orchestration and its output schema.
from run_stage3_interface_study import (  # type: ignore[import-not-found]
    CheckpointSpec,
    _active_centers,
    _direction_for_sample,
    _domain_lengths,
    _field_kpis,
    _load_dataset,
    _load_model_spec,
    _query_points,
    _valid_geometry,
    parse_checkpoint_specs,
)

from honf_forward_core.interface_fields.routing_index.pair_join import (  # type: ignore[import-not-found]
    compile_two_hop_pairs,
)
from honf_forward_core.interface_fields.routing_index.sparse_projection import (  # type: ignore[import-not-found]
    build_typed_source_incidence,
    ordinary_source_sparsemax,
    source_measure_sparsemax,
)
from honf_forward_core.interface_fields.routing_index.types import (  # type: ignore[import-not-found]
    PackedPairs,
)
from honf_runtime.compat import select_device  # type: ignore[import-not-found]

SCHEMA_VERSION = 1
POSITION_STEP_FRACTIONS = (0.01, 0.005)
HEAT_STEP_VALUES = (0.01, 0.005)
FINE_SOURCE_STEPS = (1.0e-3, 5.0e-4)
DEFAULT_ANCHOR_CASES = ("0273", "0298")


class DiagnosticUnavailable(RuntimeError):
    """Raised when a requested diagnostic hook is absent from a model."""


def _validate_steps(steps: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(step) for step in steps)
    if not values or any(step <= 0.0 or not math.isfinite(step) for step in values):
        raise ValueError("steps must contain finite positive values")
    return values


def _scalar(value: torch.Tensor | float, *, name: str = "function output") -> torch.Tensor:
    """Return a scalar tensor and reject accidental vector reductions."""

    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if value.numel() != 1:
        raise ValueError(f"{name} must contain one scalar, got shape {tuple(value.shape)}")
    return value.reshape(())


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    finite = torch.isfinite(detached) if detached.is_floating_point() else torch.ones_like(detached, dtype=torch.bool)
    result: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "count": int(detached.numel()),
        "finite": bool(finite.all()),
    }
    if detached.numel() and detached.is_floating_point():
        finite_values = detached[finite]
        if finite_values.numel():
            result.update(
                {
                    "min": float(finite_values.min().cpu()),
                    "max": float(finite_values.max().cpu()),
                    "mean": float(finite_values.mean().cpu()),
                    "l2": float(torch.linalg.vector_norm(finite_values).cpu()),
                }
            )
    return result


def _json_value(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() <= 256:
            return value.detach().cpu().tolist()
        return _tensor_summary(value)
    if isinstance(value, np.ndarray):
        if value.size <= 256:
            return value.tolist()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "count": int(value.size),
        }
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _status_from_rows(rows: Mapping[str, Any]) -> str:
    """Collapse step-level statuses with failures taking precedence.

    Successful rows omit a status in a few older artifacts, so an omitted
    value is interpreted as ``ok`` for backwards-compatible result reading.
    This helper is deliberately local to the diagnostic schema; it does not
    alter any model or routing status.
    """

    statuses = [str(row.get("status", "ok")) for row in rows.values() if isinstance(row, Mapping)]
    if "failed" in statuses:
        return "failed"
    if "ok" in statuses:
        return "ok"
    return "unavailable"


def _status_from_nested(*values: Any) -> str:
    """Collapse nested diagnostic statuses, preserving failure information."""

    statuses: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            status = value.get("status")
            if status is not None:
                statuses.append(str(status))
            for key, item in value.items():
                if key != "status":
                    collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    for value in values:
        collect(value)
    if "failed" in statuses:
        return "failed"
    if "ok" in statuses:
        return "ok"
    return "unavailable"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one compact derivative artifact at the requested path."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def central_difference(
    function: Callable[[torch.Tensor], torch.Tensor],
    value: torch.Tensor,
    direction: torch.Tensor,
    step: float,
) -> torch.Tensor:
    """Evaluate a signed central difference without retaining FD graphs."""

    step = float(step)
    _validate_steps((step,))
    if tuple(value.shape) != tuple(direction.shape):
        raise ValueError("value and direction must have identical shapes")
    plus = value.detach() + step * direction.detach()
    minus = value.detach() - step * direction.detach()
    with torch.no_grad():
        plus_value = _scalar(function(plus), name="central-difference output")
        minus_value = _scalar(function(minus), name="central-difference output")
    return (plus_value - minus_value) / (2.0 * step)


def directional_autograd(
    function: Callable[[torch.Tensor], torch.Tensor],
    value: torch.Tensor,
    direction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return a scalar directional derivative and the full input gradient."""

    if tuple(value.shape) != tuple(direction.shape):
        raise ValueError("value and direction must have identical shapes")
    probe = value.detach().clone().requires_grad_(True)
    output = _scalar(function(probe), name="autograd output")
    gradient = torch.autograd.grad(output, probe, allow_unused=True)[0]
    if gradient is None:
        return output.detach().new_zeros(()), None
    return torch.sum(gradient * direction.detach()), gradient


def scalar_derivative_probe(
    function: Callable[[torch.Tensor], torch.Tensor],
    value: torch.Tensor,
    direction: torch.Tensor,
    steps: Sequence[float],
    *,
    value_units: str,
    output_units: str,
    output_scale: float = 1.0,
    input_scale: float = 1.0,
) -> dict[str, Any]:
    """Compare one signed AD derivative with central differences at fixed steps.

    ``output_scale`` converts the function output from model units to an
    optional physical unit.  ``input_scale`` is the physical value represented
    by one model-input unit, so the reported physical derivative is
    ``output_scale / input_scale`` times the model derivative.
    """

    step_values = _validate_steps(steps)
    ad, gradient = directional_autograd(function, value, direction)
    ad_float = float(ad.detach().cpu())
    rows: dict[str, Any] = {}
    for step in step_values:
        fd = central_difference(function, value, direction, step)
        fd_float = float(fd.detach().cpu())
        factor = float(output_scale) / float(input_scale)
        rows[f"h={step:g}"] = {
            "step": step,
            "autograd": ad_float,
            "central_fd": fd_float,
            "absolute_error": abs(ad_float - fd_float),
            "relative_error": abs(ad_float - fd_float) / max(abs(ad_float), abs(fd_float), 1.0e-12),
            "autograd_physical": ad_float * factor,
            "central_fd_physical": fd_float * factor,
            "absolute_error_physical": abs(ad_float - fd_float) * abs(factor),
            "signed": True,
        }
    return {
        "value_units": value_units,
        "output_units": output_units,
        "physical_output_units": output_units if output_scale != 1.0 else None,
        "output_scale": float(output_scale),
        "input_scale": float(input_scale),
        "gradient_finite": gradient is None or bool(torch.isfinite(gradient).all()),
        "gradient_is_none": gradient is None,
        "steps": rows,
    }


def vector_jvp(
    function: Callable[[torch.Tensor], torch.Tensor],
    value: torch.Tensor,
    direction: torch.Tensor,
    steps: Sequence[float],
) -> dict[str, Any]:
    """Compare a vector-valued JVP with central finite-difference vectors."""

    step_values = _validate_steps(steps)
    if tuple(value.shape) != tuple(direction.shape):
        raise ValueError("value and direction must have identical shapes")
    zero = value.new_zeros(())

    def scalar_parameter(delta: torch.Tensor) -> torch.Tensor:
        return function(value.detach() + delta * direction.detach())

    _, tangent = torch.autograd.functional.jvp(
        scalar_parameter,
        (zero,),
        (value.new_ones(()),),
        strict=False,
    )
    rows: dict[str, Any] = {}
    for step in step_values:
        with torch.no_grad():
            plus = scalar_parameter(value.new_tensor(step))
            minus = scalar_parameter(value.new_tensor(-step))
            finite = (plus - minus) / (2.0 * step)
        difference = tangent.detach() - finite.detach()
        rows[f"h={step:g}"] = {
            "step": step,
            "status": "ok",
            "jvp": _tensor_summary(tangent),
            "central_fd": _tensor_summary(finite),
            "difference": _tensor_summary(difference),
            "relative_l2_error": float(
                torch.linalg.vector_norm(difference).cpu()
                / torch.linalg.vector_norm(finite).clamp_min(1.0e-12).cpu()
            ),
            "signed": True,
        }
    return {"status": "ok", "steps": rows, "jvp_available": True}


def _default_route_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.float64
    source_logits = torch.tensor(
        [[[1.10, 0.90, 0.70], [0.80, 1.15, 0.95], [1.00, 0.82, 1.17]]],
        dtype=dtype,
    )
    query_logits = torch.tensor(
        [[[1.08, 0.92, 0.76], [0.84, 1.12, 0.98], [1.03, 0.79, 1.16], [0.91, 1.05, 0.83]]],
        dtype=dtype,
    )
    source_weights = torch.tensor([[0.20, 0.50, 0.30]], dtype=dtype)
    return source_logits, query_logits, source_weights


def _dense_route_prior(
    source_logits: torch.Tensor,
    query_logits: torch.Tensor,
    source_weights: torch.Tensor,
    source_valid: torch.Tensor | None = None,
    query_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, Any, PackedPairs]:
    """Evaluate the fixed-route differentiable z -> A -> mu -> d -> Pi graph."""

    membership = ordinary_source_sparsemax(source_logits, source_valid)
    incidence = build_typed_source_incidence(
        source_weights,
        membership,
        source_valid=source_valid,
    )
    projection = source_measure_sparsemax(
        query_logits,
        incidence.hub_measure,
        query_valid,
    )
    # Denominator-cancelled two-hop prior.  This dense expression is used only
    # by the tiny numerical diagnostic; production execution uses the packed
    # positive-pair join and never materializes this tensor.
    prior = source_weights[:, None, :] * torch.einsum(
        "bqk,bnk->bqn", projection.density, membership
    )
    pairs = compile_two_hop_pairs(projection.density, incidence)
    return prior, membership, projection, pairs


def fixed_route_jvp(
    source_logits: torch.Tensor,
    query_logits: torch.Tensor,
    source_weights: torch.Tensor,
    *,
    direction: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    source_valid: torch.Tensor | None = None,
    query_valid: torch.Tensor | None = None,
    steps: Sequence[float] = (1.0e-6, 5.0e-7),
) -> dict[str, Any]:
    """Run a fixed-support JVP/FD check through ``z,A,mu,d,Pi``."""

    inputs = tuple(value.detach().clone().requires_grad_(True) for value in (source_logits, query_logits, source_weights))
    if direction is None:
        directions = tuple(torch.full_like(value, 0.013 + 0.007 * index) for index, value in enumerate(inputs))
    else:
        directions = tuple(value.detach().to(dtype=input_value.dtype) for value, input_value in zip(direction, inputs, strict=True))
    if any(tuple(value.shape) != tuple(input_value.shape) for value, input_value in zip(directions, inputs, strict=True)):
        raise ValueError("fixed-route directions must align with source_logits, query_logits, and source_weights")

    def function(z: torch.Tensor, q: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return _dense_route_prior(z, q, weights, source_valid, query_valid)[0]

    _, tangent = torch.autograd.functional.jvp(function, inputs, directions, strict=False)
    jvp_available = True

    base_prior, membership, projection, pairs = _dense_route_prior(*inputs, source_valid, query_valid)
    hub_measure = torch.einsum("bn,bnk->bk", inputs[2], membership)
    rows: dict[str, Any] = {}
    for step in _validate_steps(steps):
        with torch.no_grad():
            plus = function(*(value + step * delta for value, delta in zip(inputs, directions, strict=True)))
            minus = function(*(value - step * delta for value, delta in zip(inputs, directions, strict=True)))
            finite = (plus - minus) / (2.0 * step)
        row: dict[str, Any] = {
            "step": step,
            "central_fd": _tensor_summary(finite),
            "signed": True,
        }
        difference = tangent.detach() - finite.detach()
        row.update(
            {
                "status": "ok",
                "jvp": _tensor_summary(tangent),
                "difference": _tensor_summary(difference),
                "relative_l2_error": float(
                    torch.linalg.vector_norm(difference).cpu()
                    / torch.linalg.vector_norm(finite).clamp_min(1.0e-12).cpu()
                ),
            }
        )
        rows[f"h={step:g}"] = row

    packed_values = base_prior[
        pairs.batch_index,
        pairs.receiver_index,
        pairs.source_index,
    ]
    pair_error = packed_values - pairs.prior.to(dtype=packed_values.dtype)
    return {
        "status": _status_from_rows(rows),
        "inputs": {
            "source_logits": _tensor_summary(inputs[0]),
            "query_logits": _tensor_summary(inputs[1]),
            "source_weights": _tensor_summary(inputs[2]),
        },
        "membership": _tensor_summary(membership),
        "query_density": _tensor_summary(projection.density),
        "query_probability": _tensor_summary(projection.probability),
        "hub_measure": _tensor_summary(hub_measure),
        "base_prior": _tensor_summary(base_prior),
        "packed_pair_count": int(pairs.unique_pair_count),
        "raw_path_count": int(pairs.raw_path_count),
        "packed_prior_dense_max_error": float(pair_error.abs().max().detach().cpu()) if pair_error.numel() else 0.0,
        "packed_prior_requires_grad": bool(pairs.prior.requires_grad),
        "jvp_available": jvp_available,
        "steps": rows,
    }


def fixed_route_gradcheck(
    source_logits: torch.Tensor,
    query_logits: torch.Tensor,
    source_weights: torch.Tensor,
    *,
    source_valid: torch.Tensor | None = None,
    query_valid: torch.Tensor | None = None,
    eps: float = 1.0e-6,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-3,
) -> dict[str, Any]:
    """Run torch's float64 gradcheck away from sparsemax support ties."""

    inputs = tuple(value.detach().clone().double().requires_grad_(True) for value in (source_logits, query_logits, source_weights))

    def function(z: torch.Tensor, q: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return _dense_route_prior(z, q, weights, source_valid, query_valid)[0]

    try:
        passed = bool(
            torch.autograd.gradcheck(
                function,
                inputs,
                eps=float(eps),
                atol=float(atol),
                rtol=float(rtol),
                check_undefined_grad=True,
            )
        )
        return {
            "status": "ok" if passed else "failed",
            "passed": passed,
            "eps": float(eps),
            "atol": float(atol),
            "rtol": float(rtol),
        }
    except (RuntimeError, ValueError) as exc:
        return {
            "status": "failed",
            "passed": False,
            "eps": float(eps),
            "atol": float(atol),
            "rtol": float(rtol),
            "error": f"{type(exc).__name__}: {exc!s}",
        }


def support_transition_probe(step: float = 1.0e-6) -> dict[str, Any]:
    """Record labelled one-sided sparsemax support-entry and support-exit rows."""

    step = _validate_steps((step,))[0]
    result: dict[str, Any] = {}
    for label, sign in (("support_exit", 1.0), ("support_entry", -1.0)):
        values: dict[str, torch.Tensor] = {}
        supports: dict[str, list[int]] = {}
        for side, coordinate in (("left", -step), ("boundary", 0.0), ("right", step)):
            logits = torch.tensor([[1.0 + sign * coordinate, 0.0]], dtype=torch.float64)
            projected = ordinary_source_sparsemax(logits)
            values[side] = projected
            supports[side] = [int(index) for index in torch.nonzero(projected[0] > 0.0, as_tuple=False).reshape(-1)]
        left_derivative = (values["boundary"] - values["left"]) / step
        right_derivative = (values["right"] - values["boundary"]) / step
        result[label] = {
            "label": label,
            "parameterization": "logits=[1 + sign*t, 0]",
            "step": step,
            "support_left": supports["left"],
            "support_boundary": supports["boundary"],
            "support_right": supports["right"],
            "left_one_sided_derivative": left_derivative[0].tolist(),
            "right_one_sided_derivative": right_derivative[0].tolist(),
            "left_boundary_jump_linf": float((values["boundary"] - values["left"]).abs().max()),
            "right_boundary_jump_linf": float((values["right"] - values["boundary"]).abs().max()),
            "continuity_checked": True,
            "values": {name: value[0].tolist() for name, value in values.items()},
        }
    return result


def conditional_fine_source_omitted_pair_fixture(
    steps: Sequence[float] = FINE_SOURCE_STEPS,
) -> dict[str, Any]:
    """Exercise the production QM/QE readers with a fixed omitted fine pair.

    This numerical fixture supplies a pair map; it does not claim that the
    trained model selected that map. Both QE keys and values vary with the
    prepared environmental source state while routing stays fixed.
    """
    from honf_forward_core.interface_fields.routed_pairwise import RoutedPairwiseField
    from honf_forward_core.interface_fields.types import EncodedInterfaceCase

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(73)
        dtype = torch.float64
        source = torch.randn(1, 3, 8, dtype=dtype)
        coordinates = torch.tensor([[[.2, .4], [1.3, .7], [.8, 1.1]]], dtype=dtype)
        receivers = torch.tensor([[[.6, .5]]], dtype=dtype)
        features = torch.randn(1, 1, 8, dtype=dtype)
        encoded = EncodedInterfaceCase(
            module_tokens=source, env_tokens=source, global_token=torch.randn(1, 8, dtype=dtype),
            module_centers=coordinates, env_coords=coordinates,
            module_present=torch.ones(1, 3, dtype=dtype),
            module_features=torch.zeros(1, 3, 2, dtype=dtype), env_features=None,
            env_weights=torch.tensor([[.2, .3, .5]], dtype=dtype),
            coordinate_scale=torch.ones(1, 1, 2, dtype=dtype),
        )
        pairs = PackedPairs(
            batch_index=torch.tensor([0, 0]), receiver_index=torch.tensor([0, 0]),
            source_index=torch.tensor([0, 2]), prior=torch.tensor([.4, .6], dtype=dtype),
            raw_path_count=2, unique_pair_count=2,
        )
        backend = RoutedPairwiseField(8, 6, 2, 2).double().eval()
        rows = {}
        for kind in ("module", "environment"):
            def response(value: torch.Tensor, source_kind=kind) -> torch.Tensor:
                if source_kind == "module":
                    return backend.read_module_pairs({"module_tokens": value}, encoded, receivers, pairs)
                keys, values = backend.project_environment_sources(value)
                return backend.read_environment_pairs(
                    {"env_tokens": value, "env_keys": keys, "env_values": values},
                    encoded, receivers, features, pairs,
                )

            with torch.no_grad():
                response(source)  # Materialize lazy response layers before derivative probes.
            selected = vector_jvp(response, source, _source_direction(source, 0, kind=kind), steps)
            omitted = vector_jvp(response, source, _source_direction(source, 1, kind=kind), steps)
            omitted_rows = list(omitted.get("steps", {}).values())
            omitted_zero = bool(omitted_rows) and all(
                row.get("status", "ok") == "ok"
                and float(row.get("central_fd", {}).get("l2", math.inf)) == 0.
                and float(row.get("jvp", {}).get("l2", math.inf)) == 0.
                for row in omitted_rows
            )
            selected_nonzero = any(
                float(row.get("jvp", {}).get("l2", 0.)) > 0.
                for row in selected.get("steps", {}).values()
            )
            rows[kind] = {
                "status": "ok" if omitted_zero and selected_nonzero else "failed",
                "selected": selected, "omitted": omitted,
                "selected_nonzero_direct_effect": selected_nonzero,
                "omitted_zero_direct_effect": omitted_zero,
            }
    return {
        "status": "ok" if all(row["status"] == "ok" for row in rows.values()) else "failed",
        "fixture": "production_fine_readers_conditional_omitted_pair",
        "numerical_fixture": True, "formal_model_sparsity": False,
        "routing_identity_fixed": True, "selected_source_index": 0,
        "omitted_source_index": 1, "pair_count": 2,
        "source_state_units": "prepared H-dimensional source states",
        "environment_content": "both projected keys and values change with the selected environmental state",
        "omitted_zero_direct_effect": all(row["omitted_zero_direct_effect"] for row in rows.values()),
        **rows,
    }


def run_toy_route_checks() -> dict[str, Any]:
    """Run fixed-route composition, packed-prior, and support fixtures."""

    source_logits, query_logits, source_weights = _default_route_fixture()
    checks: dict[str, Any] = {
        "fixed_route_jvp": fixed_route_jvp(source_logits, query_logits, source_weights),
        "fixed_route_gradcheck": fixed_route_gradcheck(source_logits, query_logits, source_weights),
        "support_transitions": support_transition_probe(),
        "conditional_fine_source_omitted_pair_fixture": conditional_fine_source_omitted_pair_fixture(),
        "interpretation": {
            "route_values": "ordinary source sparsemax, source-measure query sparsemax, and denominator-cancelled two-hop Pi remain in the differentiable scalar graph",
            "support_boundary": "boundary rows use one-sided derivatives; a single central derivative is not interpreted as unique",
            "physical_status": "toy route checks validate implementation arithmetic only and are not physical evidence",
        },
    }
    statuses = [
        str(value.get("status", "ok"))
        for key, value in checks.items()
        if key != "interpretation" and isinstance(value, Mapping)
    ]
    status_counts = {"ok": statuses.count("ok"), "unavailable": statuses.count("unavailable"), "failed": statuses.count("failed")}
    checks["status"] = "failed" if status_counts["failed"] else ("ok" if status_counts["ok"] else "unavailable")
    checks["status_counts"] = status_counts
    return checks


def _copy_detached_mapping(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {key: _copy_detached_mapping(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_detached_mapping(item) for item in value)
    if isinstance(value, list):
        return [_copy_detached_mapping(item) for item in value]
    return value


def _forward_full_model(
    model: Any,
    sample: Mapping[str, Any],
    query: torch.Tensor,
    device: torch.device,
    *,
    centers: torch.Tensor | None = None,
    heat_powers: torch.Tensor | None = None,
    return_prepared_state: bool = False,
    return_routing_maps: bool = False,
) -> dict[str, Any]:
    """Run the full wrapper path with differentiable geometry/heat tensors."""

    query_array = query.detach().cpu().numpy().astype(np.float32)
    if query_array.ndim == 3 and query_array.shape[0] == 1:
        query_array = query_array[0]
    batch = make_batch(dict(sample), query_array, device)
    structure = batch["structure"]
    if centers is not None:
        structure["module_centers"] = centers
    if heat_powers is not None:
        structure["heat_powers"] = heat_powers
    # Rebuild local parameters from the live heat tensor.  Passing the sample's
    # cached local parameters would silently freeze the attribute derivative.
    return model(
        structure,
        batch["query_xy"],
        interface_condition=batch.get("interface_condition"),
        local_module_params=None,
        teacher_port_tokens=batch.get("teacher_port_tokens"),
        local_query_points=batch.get("module_internal_query_points"),
        local_port_condition_mode="predicted",
        return_prepared_state=bool(return_prepared_state),
        return_routing_maps=bool(return_routing_maps),
    )


def _model_scalar(
    model: Any,
    sample: Mapping[str, Any],
    query: torch.Tensor,
    device: torch.device,
    metric: str,
    *,
    centers: torch.Tensor | None = None,
    heat_powers: torch.Tensor | None = None,
) -> torch.Tensor:
    output = _forward_full_model(
        model,
        sample,
        query,
        device,
        centers=centers,
        heat_powers=heat_powers,
    )
    values = _field_kpis(output["pred_field"], query, list(model.config.channelthermal.field_names))
    if metric not in values:
        raise DiagnosticUnavailable(f"field KPI {metric!r} is unavailable")
    return values[metric]


def _field_output_scale(model: Any, metric: str) -> float:
    """Return the target-normalizer scale for a scalar field KPI."""

    if not bool(getattr(model, "global_normalize_targets", False)):
        return 1.0
    stats = getattr(model, "global_normalization_stats", {})
    raw = stats.get("field_std_by_channel") if isinstance(stats, Mapping) else None
    names = list(model.config.channelthermal.field_names)
    if raw is None:
        return 1.0
    if metric == "mean_temperature" and "temperature" in names:
        index = names.index("temperature")
    elif metric == "pressure_drop" and "p" in names:
        index = names.index("p")
    else:
        return 1.0
    values = np.asarray(raw).reshape(-1)
    return float(values[index]) if index < len(values) and abs(float(values[index])) > 1.0e-12 else 1.0


def _field_output_units(model: Any) -> str:
    """Describe the scalar output space used by the loaded checkpoint."""

    return (
        "normalized model-output units"
        if bool(getattr(model, "global_normalize_targets", False))
        else "model physical-output units"
    )


def _heat_input_scale(checkpoint: Mapping[str, Any]) -> float:
    """Return physical heat units per normalized model input unit."""

    train = checkpoint.get("train_config", {})
    dataset_cfg = train.get("dataset", {}) if isinstance(train, Mapping) else {}
    if not bool(dataset_cfg.get("normalize_inputs", False)):
        return 1.0
    stats = checkpoint.get("global_normalization_stats", {})
    if not isinstance(stats, Mapping) or "heat_power_std" not in stats:
        return 1.0
    value = float(np.asarray(stats["heat_power_std"]).reshape(-1)[0])
    return value if math.isfinite(value) and value > 1.0e-12 else 1.0


def position_derivative_probe(
    model: Any,
    sample: Mapping[str, Any],
    query: torch.Tensor,
    device: torch.device,
    *,
    module_index: int | None = None,
    direction: Sequence[float] | None = None,
    step_fractions: Sequence[float] = POSITION_STEP_FRACTIONS,
    metrics: Sequence[str] = ("mean_temperature", "pressure_drop"),
) -> dict[str, Any]:
    """Probe full physical position derivatives at ``0.01R`` and ``0.005R``."""

    centers_np, present = _active_centers(sample)
    active = np.flatnonzero(present)
    if not len(active):
        raise DiagnosticUnavailable("case has no active modules")
    index = int(active[0] if module_index is None else module_index)
    if index < 0 or index >= len(centers_np) or not present[index]:
        raise ValueError(f"module_index={index} is not active")
    radius = float(model.config.core_honf.module_radius)
    fractions = _validate_steps(step_fractions)
    steps = tuple(fraction * radius for fraction in fractions)
    if direction is None:
        try:
            direction_array = _direction_for_sample(sample, model, index, max(steps))
        except ValueError as exc:
            # No legal central-difference direction is a geometry limitation;
            # callers distinguish this explicit availability result from a
            # model/runtime failure raised while evaluating a valid layout.
            raise DiagnosticUnavailable(f"no valid geometry direction: {exc!s}") from exc
    else:
        direction_array = np.asarray(direction, dtype=np.float64).reshape(-1)
        if direction_array.shape != (2,) or not np.isfinite(direction_array).all() or np.linalg.norm(direction_array) == 0.0:
            raise ValueError("direction must be a finite nonzero 2-vector")
        direction_array = direction_array / np.linalg.norm(direction_array)
    base_centers = torch.from_numpy(centers_np.astype(np.float32)).unsqueeze(0).to(device)
    coordinate_direction = torch.zeros_like(base_centers)
    coordinate_direction[0, index] = torch.as_tensor(direction_array, device=device, dtype=base_centers.dtype)
    lx, ly = _domain_lengths(sample, model)

    def function(value: torch.Tensor, metric_name: str) -> torch.Tensor:
        return _model_scalar(model, sample, query, device, metric_name, centers=value)

    result: dict[str, Any] = {
        "status": "ok",
        "module_index": index,
        "direction": direction_array.tolist(),
        "steps": {},
        "coordinate_units": "model geometry coordinate units",
        "functional_units": f"{_field_output_units(model)} per geometry-coordinate unit",
        "full_physical_loop": True,
        "physical_conversion": "field target standard deviation from checkpoint when target normalization is enabled",
    }
    for metric in metrics:
        try:
            ad, gradient = directional_autograd(
                lambda value, metric_name=metric: function(value, metric_name),
                base_centers,
                coordinate_direction,
            )
            metric_rows: dict[str, Any] = {}
            output_scale = _field_output_scale(model, metric)
            for fraction, step in zip(fractions, steps, strict=True):
                plus = centers_np.copy()
                minus = centers_np.copy()
                plus[index] += direction_array * step
                minus[index] -= direction_array * step
                valid = _valid_geometry(plus, present, radius, lx, ly) and _valid_geometry(
                    minus, present, radius, lx, ly
                )
                key = f"h={fraction:g}r"
                if not valid:
                    metric_rows[key] = {
                        "step": step,
                        "step_fraction_of_radius": fraction,
                        "status": "unavailable",
                        "reason": "central-difference step leaves valid geometry domain",
                    }
                    continue
                with torch.no_grad():
                    plus_value = function(
                        torch.from_numpy(plus.astype(np.float32)).unsqueeze(0).to(device), metric
                    )
                    minus_value = function(
                        torch.from_numpy(minus.astype(np.float32)).unsqueeze(0).to(device), metric
                    )
                fd = (plus_value - minus_value) / (2.0 * step)
                ad_float = float(ad.detach().cpu())
                fd_float = float(fd.detach().cpu())
                metric_rows[key] = {
                    "step": step,
                    "step_fraction_of_radius": fraction,
                    "status": "ok",
                    "autograd": ad_float,
                    "central_fd": fd_float,
                    "absolute_error": abs(ad_float - fd_float),
                    "relative_error": abs(ad_float - fd_float) / max(abs(ad_float), abs(fd_float), 1.0e-12),
                    "autograd_physical": ad_float * output_scale,
                    "central_fd_physical": fd_float * output_scale,
                    "absolute_error_physical": abs(ad_float - fd_float) * abs(output_scale),
                    "signed": True,
                }
            result["steps"][metric] = metric_rows
            result.setdefault("output_scales", {})[metric] = output_scale
            result["gradient_finite"] = result.get("gradient_finite", True) and (
                gradient is None or bool(torch.isfinite(gradient).all())
            )
        except DiagnosticUnavailable as exc:
            result["steps"][metric] = {
                "status": "unavailable",
                "reason": f"{type(exc).__name__}: {exc!s}",
            }
        except (RuntimeError, ValueError) as exc:
            result["steps"][metric] = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc!s}",
            }
    result["status"] = _status_from_nested(result["steps"])
    return result


def heat_derivative_probe(
    model: Any,
    sample: Mapping[str, Any],
    query: torch.Tensor,
    device: torch.device,
    checkpoint: Mapping[str, Any],
    *,
    module_index: int | None = None,
    steps: Sequence[float] = HEAT_STEP_VALUES,
    metrics: Sequence[str] = ("mean_temperature", "pressure_drop"),
) -> dict[str, Any]:
    """Probe a full physical heat derivative in normalized and physical units."""

    heat = np.asarray(sample["structure"]["heat_powers"], dtype=np.float32)
    if heat.ndim == 1:
        heat = heat[None, :]
    if heat.ndim != 2 or heat.shape[0] != 1:
        raise ValueError(f"heat_powers must have one-case shape [1,M], got {heat.shape}")
    present = np.asarray(sample["structure"]["module_present"], dtype=np.float32).reshape(-1) > 0.5
    active = np.flatnonzero(present)
    if not len(active):
        raise DiagnosticUnavailable("case has no active modules")
    index = int(active[0] if module_index is None else module_index)
    if index < 0 or index >= heat.shape[1] or not present[index]:
        raise ValueError(f"module_index={index} is not active")
    base_heat = torch.from_numpy(heat).to(device)
    heat_direction = torch.zeros_like(base_heat)
    heat_direction[0, index] = 1.0
    step_values = _validate_steps(steps)
    input_scale = _heat_input_scale(checkpoint)
    result: dict[str, Any] = {
        "status": "ok",
        "module_index": index,
        "steps": {},
        "attribute_units": (
            "normalized checkpoint dataset heat-power input units"
            if input_scale != 1.0
            else "physical checkpoint dataset heat-power input units"
        ),
        "physical_attribute_units": "dataset physical heat-power units when normalization statistics are available",
        "functional_units": f"{_field_output_units(model)} per checkpoint heat-power input unit",
        "full_physical_loop": True,
        "input_physical_scale": input_scale,
        "physical_conversion": "field and heat standard deviations from checkpoint when input/target normalization is enabled",
        "step_policy": "0.01 and 0.005 normalized heat-power input units",
    }
    for metric in metrics:
        try:
            function = lambda value, metric_name=metric: _model_scalar(
                model,
                sample,
                query,
                device,
                metric_name,
                heat_powers=value,
            )
            ad, gradient = directional_autograd(function, base_heat, heat_direction)
            output_scale = _field_output_scale(model, metric)
            metric_rows: dict[str, Any] = {}
            for step in step_values:
                fd = central_difference(function, base_heat, heat_direction, step)
                ad_float = float(ad.detach().cpu())
                fd_float = float(fd.detach().cpu())
                factor = output_scale / input_scale
                physical_step = step * input_scale
                metric_rows[f"h={step:g}"] = {
                    "step": step,
                    "step_normalized": step,
                    "step_physical": physical_step,
                    "h_normalized": step,
                    "h_physical": physical_step,
                    "status": "ok",
                    "autograd": ad_float,
                    "central_fd": fd_float,
                    "absolute_error": abs(ad_float - fd_float),
                    "relative_error": abs(ad_float - fd_float) / max(abs(ad_float), abs(fd_float), 1.0e-12),
                    "autograd_physical": ad_float * factor,
                    "central_fd_physical": fd_float * factor,
                    "absolute_error_physical": abs(ad_float - fd_float) * abs(factor),
                    "signed": True,
                }
            result["steps"][metric] = metric_rows
            result.setdefault("output_scales", {})[metric] = output_scale
            result["gradient_finite"] = result.get("gradient_finite", True) and (
                gradient is None or bool(torch.isfinite(gradient).all())
            )
        except DiagnosticUnavailable as exc:
            result["steps"][metric] = {
                "status": "unavailable",
                "reason": f"{type(exc).__name__}: {exc!s}",
            }
        except (RuntimeError, ValueError) as exc:
            result["steps"][metric] = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc!s}",
            }
    result["status"] = _status_from_nested(result["steps"])
    result["step_policy"] = {
        "normalized": list(step_values),
        "physical": [step * input_scale for step in step_values],
        "physical_scale": input_scale,
    }
    return result


def _packed_pairs_from_routing_aux(aux: Mapping[str, Any], prefix: str) -> PackedPairs:
    names = tuple(f"routing_{prefix}_pair_{suffix}" for suffix in ("batch", "receiver", "source", "prior"))
    if any(name not in aux or not torch.is_tensor(aux[name]) for name in names):
        raise DiagnosticUnavailable(f"routing pair maps are unavailable for {prefix}")
    batch_index, receiver_index, source_index, prior = (aux[name].detach() for name in names)
    raw_name = f"routing_{prefix}_raw_path_count"
    unique_name = f"routing_{prefix}_unique_pair_count"
    raw = int(aux[raw_name].detach().cpu()) if raw_name in aux and torch.is_tensor(aux[raw_name]) else int(prior.numel())
    unique = int(aux[unique_name].detach().cpu()) if unique_name in aux and torch.is_tensor(aux[unique_name]) else int(prior.numel())
    return PackedPairs(batch_index, receiver_index, source_index, prior, raw, unique)


def _pairs_for_receiver(
    pairs: PackedPairs,
    *,
    batch_index: int,
    receiver_index: int,
) -> PackedPairs:
    """Filter a packed map to one receiver and remap its receiver to zero."""

    mask = (pairs.batch_index == int(batch_index)) & (pairs.receiver_index == int(receiver_index))
    selected = int(mask.sum().detach().cpu())
    return PackedPairs(
        pairs.batch_index[mask],
        torch.zeros_like(pairs.receiver_index[mask]),
        pairs.source_index[mask],
        pairs.prior[mask],
        raw_path_count=selected,
        unique_pair_count=selected,
    )


def _receiver_source_sets(
    pairs: PackedPairs,
    *,
    receiver_count: int,
    valid_indices: Sequence[int],
    batch_index: int = 0,
) -> list[tuple[int, list[int], list[int], PackedPairs]]:
    """Return selected/omitted source sets independently for each receiver."""

    valid = [int(index) for index in valid_indices]
    rows: list[tuple[int, list[int], list[int], PackedPairs]] = []
    for receiver_index in range(int(receiver_count)):
        local_pairs = _pairs_for_receiver(
            pairs,
            batch_index=batch_index,
            receiver_index=receiver_index,
        )
        selected = {
            int(value)
            for value in local_pairs.source_index.detach().cpu().tolist()
            if int(value) in valid
        }
        selected_indices = [index for index in valid if index in selected]
        omitted_indices = [index for index in valid if index not in selected]
        rows.append((receiver_index, selected_indices, omitted_indices, local_pairs))
    return rows


def _source_direction(source: torch.Tensor, source_index: int, *, kind: str) -> torch.Tensor:
    direction = torch.zeros_like(source)
    if kind == "module":
        if source.ndim != 3:
            raise ValueError(f"module source state must be [B,M,H], got {tuple(source.shape)}")
        direction[:, source_index, :] = 1.0
    elif kind == "environment":
        if source.ndim != 3:
            raise ValueError(f"environment source tokens must be [B,E,H], got {tuple(source.shape)}")
        direction[:, source_index, :] = 1.0
    else:
        raise ValueError(f"unknown source kind {kind!r}")
    return direction / torch.linalg.vector_norm(direction).clamp_min(1.0e-12)


def _direct_source_response(
    backend: Any,
    backend_state: Mapping[str, Any],
    encoded: Any,
    prepared: Any,
    receivers: torch.Tensor,
    receiver_features: torch.Tensor,
    pairs: PackedPairs,
    source: torch.Tensor,
    direction: torch.Tensor,
    kind: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    fixed_state = _copy_detached_mapping(dict(backend_state))
    fixed_pairs = PackedPairs(
        pairs.batch_index.detach(),
        pairs.receiver_index.detach(),
        pairs.source_index.detach(),
        pairs.prior.detach(),
        pairs.raw_path_count,
        pairs.unique_pair_count,
    )
    del prepared

    def response(delta: torch.Tensor) -> torch.Tensor:
        state = dict(fixed_state)
        candidate = source.detach() + delta * direction.detach()
        if kind == "module":
            state["module_tokens"] = candidate
            value = backend.read_module_pairs(state, encoded, receivers, fixed_pairs)
        else:
            # Environment source influence is defined on the prepared token
            # state.  Re-project both K and V for each perturbation, while the
            # routing pairs, priors, receiver/query features, and all other
            # preparation remain fixed.  Keeping cached projections here would
            # measure V-only dependence and miss the selected-source K path.
            state["env_tokens"] = candidate
            state.pop("env_keys", None)
            state.pop("env_values", None)
            value = backend.read_environment_pairs(
                state,
                encoded,
                receivers,
                receiver_features,
                fixed_pairs,
            )
        return value

    return response


def _direct_vector_probe(
    response: Callable[[torch.Tensor], torch.Tensor],
    steps: Sequence[float] = FINE_SOURCE_STEPS,
    *,
    zero: torch.Tensor | None = None,
) -> dict[str, Any]:
    if zero is None:
        zero = torch.zeros((), dtype=torch.float32)
    _, tangent = torch.autograd.functional.jvp(
        response,
        (zero,),
        (torch.ones_like(zero),),
        strict=False,
    )
    rows: dict[str, Any] = {}
    for step in _validate_steps(steps):
        with torch.no_grad():
            plus = response(zero.new_tensor(step))
            minus = response(zero.new_tensor(-step))
            finite = (plus - minus) / (2.0 * step)
        row: dict[str, Any] = {
            "step": step,
            "central_fd": _tensor_summary(finite),
            "signed": True,
        }
        difference = tangent.detach() - finite.detach()
        row.update(
            {
                "status": "ok",
                "jvp": _tensor_summary(tangent),
                "difference": _tensor_summary(difference),
                "relative_l2_error": float(
                    torch.linalg.vector_norm(difference).cpu()
                    / torch.linalg.vector_norm(finite).clamp_min(1.0e-12).cpu()
                ),
            }
        )
        rows[f"h={step:g}"] = row
    return {"status": "ok", "steps": rows, "jvp_available": True}


def _one_source_direct_probe(
    backend: Any,
    backend_state: Mapping[str, Any],
    encoded: Any,
    prepared: Any,
    receivers: torch.Tensor,
    receiver_features: torch.Tensor,
    pairs: PackedPairs,
    source_index: int,
    kind: str,
    *,
    receiver_index: int | None = None,
) -> dict[str, Any]:
    source_key = "module_tokens" if kind == "module" else "env_tokens"
    source = backend_state.get(source_key)
    if not torch.is_tensor(source):
        raise DiagnosticUnavailable(f"backend state has no {source_key}")
    direction = _source_direction(source, source_index, kind=kind)
    response = _direct_source_response(
        backend,
        backend_state,
        encoded,
        prepared,
        receivers,
        receiver_features,
        pairs,
        source,
        direction,
        kind,
    )
    return {
        "status": "ok",
        "receiver_index": None if receiver_index is None else int(receiver_index),
        "source_index": int(source_index),
        "source_kind": kind,
        "source_shape": list(source.shape),
        "source_definition": (
            "prepared module_tokens [B,M,H]"
            if kind == "module"
            else "prepared env_tokens [B,E,H]; project_environment_sources recomputed K and V per perturbation"
        ),
        "response": _direct_vector_probe(
            response,
            zero=torch.zeros((), device=source.device, dtype=source.dtype),
        ),
        "routing_identity_fixed": True,
        "pair_prior_fixed": True,
        "contextual_source_states_fixed": True,
        "fine_geometry_fixed": True,
    }


def routed_fine_source_probe(
    model: Any,
    model_outputs: Mapping[str, Any],
    query: torch.Tensor,
    *,
    steps: Sequence[float] = FINE_SOURCE_STEPS,
) -> dict[str, Any]:
    """Measure selected and omitted direct source effects with route state fixed."""

    architecture = str(model.config.core_honf.forward_architecture)
    if architecture != "routed_pairwise_honf":
        raise DiagnosticUnavailable(f"direct routed-source probe requires routed_pairwise_honf, got {architecture!r}")
    wrapper = model_outputs.get("prepared_state")
    if wrapper is None:
        raise DiagnosticUnavailable("prepared state is required for direct source probes")
    prepared = getattr(wrapper, "prepared", wrapper)
    backend = model.core.backend
    backend_state = prepared.backend_state
    encoded = prepared.encoded
    receiver_features = model.core._receiver_features(prepared, query)
    aux = model_outputs.get("routing_aux", {})
    if not isinstance(aux, Mapping):
        raise DiagnosticUnavailable("routing_aux is unavailable")
    result: dict[str, Any] = {
        "status": "ok",
        "query_count": int(query.shape[1]),
        "steps": list(_validate_steps(steps)),
        "module": {},
        "environment": {},
        "attribution": {
            "direct": "only selected fine source values vary; route identities, priors, contextual source states, receiver geometry, and other branches are fixed",
            "omitted_pairs": "an omitted source is expected to have exactly zero direct effect because the fine reader never gathers it; selection is audited per receiver, not by a query-wide source union",
            "environment_source": "prepared env_tokens [B,E,H] vary; project_environment_sources recomputes both K and V while routing and other context remain fixed",
            "full_model": "full position/heat probes are reported separately and may include preparation, coarse/local context, and physical feedback",
        },
    }
    for kind, prefix in (("module", "module"), ("environment", "environment")):
        try:
            pairs = _packed_pairs_from_routing_aux(aux, prefix)
            source_key = "module_tokens" if kind == "module" else "env_tokens"
            source = backend_state.get(source_key)
            if not torch.is_tensor(source):
                raise DiagnosticUnavailable(f"backend state has no {source_key}")
            source_count = int(source.shape[1])
            if kind == "module":
                valid = torch.nonzero(encoded.module_present[0] > 0.5, as_tuple=False).reshape(-1).tolist()
                valid_indices = [int(value) for value in valid]
            else:
                valid_indices = list(range(source_count))
            receiver_count = int(query.shape[1])
            if receiver_count <= 0:
                raise ValueError("direct source probe requires at least one query receiver")
            receiver_rows = _receiver_source_sets(
                pairs,
                receiver_count=receiver_count,
                valid_indices=valid_indices,
            )
            # A query-wide union can hide an omitted edge on an individual
            # receiver.  Choose the first receiver that has a valid omitted
            # source and keep all direct reads local to that receiver.
            chosen = next((row for row in receiver_rows if row[2]), receiver_rows[0])
            receiver_index, selected_candidates, omitted_candidates, local_pairs = chosen
            local_query = query[:, receiver_index : receiver_index + 1]
            local_receiver_features = receiver_features[:, receiver_index : receiver_index + 1]
            rows: dict[str, Any] = {
                "status": "unavailable",
                "batch_index": 0,
                "receiver_index": int(receiver_index),
                "selection_scope": "one_receiver",
                "valid_source_count": len(valid_indices),
                "selected_pair_source_count": len(selected_candidates),
                "omitted_source_count": len(omitted_candidates),
                "selected_pair_count": int(local_pairs.unique_pair_count),
                "selected": {},
                "omitted": {},
            }
            if selected_candidates:
                rows["selected"] = _one_source_direct_probe(
                    backend,
                    backend_state,
                    encoded,
                    prepared,
                    local_query,
                    local_receiver_features,
                    local_pairs,
                    selected_candidates[0],
                    kind,
                    receiver_index=receiver_index,
                )
            else:
                rows["selected"] = {"status": "unavailable", "reason": "no selected source in queried pairs"}
            if omitted_candidates:
                rows["omitted"] = _one_source_direct_probe(
                    backend,
                    backend_state,
                    encoded,
                    prepared,
                    local_query,
                    local_receiver_features,
                    local_pairs,
                    omitted_candidates[0],
                    kind,
                    receiver_index=receiver_index,
                )
            else:
                rows["omitted"] = {
                    "status": "unavailable",
                    "reason": "all valid sources were selected for this fixed query set",
                }
            rows["status"] = _status_from_nested(rows["selected"], rows["omitted"])
            result[kind] = rows
        except DiagnosticUnavailable as exc:
            result[kind] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc!s}"}
        except (RuntimeError, ValueError) as exc:
            result[kind] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc!s}"}
    result["status"] = _status_from_nested(result["module"], result["environment"])
    return result


def compare_direct_and_full_influence(
    direct_function: Callable[[torch.Tensor], torch.Tensor],
    full_function: Callable[[torch.Tensor], torch.Tensor],
    value: torch.Tensor,
    direction: torch.Tensor,
    *,
    steps: Sequence[float] = FINE_SOURCE_STEPS,
    direct_value_units: str = "fixed prepared source-state units",
    full_value_units: str = "full model design units",
) -> dict[str, Any]:
    """Return explicitly separate direct-source and full-model derivative rows."""

    return {
        "direct_source": scalar_derivative_probe(
            direct_function,
            value,
            direction,
            steps,
            value_units=direct_value_units,
            output_units="direct fine response units",
        ),
        "full_model": scalar_derivative_probe(
            full_function,
            value,
            direction,
            steps,
            value_units=full_value_units,
            output_units="full model output units",
        ),
        "separation": {
            "direct_route_fixed": True,
            "full_model_recomputes_preparation_and_physical_loop": True,
            "interpretation": "a zero omitted direct edge does not imply zero total module influence after upstream preparation and physical coupling are allowed to change",
        },
    }


def _omitted_source_index(direct: Mapping[str, Any], kind: str) -> int | None:
    """Return the receiver-local omitted source selected by the direct audit."""

    kind_rows = direct.get(kind)
    if not isinstance(kind_rows, Mapping):
        return None
    omitted = kind_rows.get("omitted")
    if not isinstance(omitted, Mapping) or omitted.get("status") != "ok":
        return None
    source_index = omitted.get("source_index")
    return int(source_index) if isinstance(source_index, (int, np.integer)) else None


def _case_status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"ok": 0, "unavailable": 0, "failed": 0}
    for row in rows:
        status = str(row.get("status", "ok"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def run_anchor_derivative_checks(
    model: Any,
    checkpoint: Mapping[str, Any],
    dataset: Any,
    device: torch.device,
    *,
    case_ids: Sequence[str] = DEFAULT_ANCHOR_CASES,
    query_count: int = 32,
) -> dict[str, Any]:
    """Run exact 0273/0298 full-loop and fixed-route derivative diagnostics."""

    rows: list[dict[str, Any]] = []
    for requested_case in case_ids:
        case_id = str(requested_case)
        row: dict[str, Any] = {"case_id": case_id, "status": "ok"}
        try:
            sample = select_sample(dataset, case_id, 0)
        except KeyError as exc:
            # A missing requested anchor is unavailable data, whereas a
            # runtime/value error from a loaded case is an actual diagnostic
            # failure and is handled below.
            row.update({"status": "unavailable", "reason": f"{type(exc).__name__}: {exc!s}"})
            rows.append(row)
            continue
        try:
            query_np = _query_points(sample, int(query_count))
            query = torch.from_numpy(query_np).unsqueeze(0).to(device)
            _centers, present = _active_centers(sample)
            active = np.flatnonzero(present)
            if not len(active):
                raise DiagnosticUnavailable("case has no active modules")
            with torch.no_grad():
                outputs = _forward_full_model(
                    model,
                    sample,
                    query,
                    device,
                    return_prepared_state=True,
                    return_routing_maps=True,
                )
            index = int(active[0])
            row["module_index"] = index
            row["query_count"] = int(query_count)
            row["position_full_model"] = position_derivative_probe(
                model,
                sample,
                query,
                device,
                module_index=index,
            )
            row["heat_full_model"] = heat_derivative_probe(
                model,
                sample,
                query,
                device,
                checkpoint,
                module_index=index,
            )
            row["direct_fine_source"] = routed_fine_source_probe(model, outputs, query)
            direct = row["direct_fine_source"]
            omitted_module_index = (
                _omitted_source_index(direct, "module") if isinstance(direct, Mapping) else None
            )
            if omitted_module_index is not None:
                receiver_index = int(direct["module"]["omitted"]["receiver_index"])
                receiver_query = query[:, receiver_index:receiver_index + 1]
                row["same_omitted_module_index"] = omitted_module_index
                row["same_omitted_module_full_model"] = {
                    "module_index": omitted_module_index,
                    "receiver_index": receiver_index,
                    "query_count": 1,
                    "interpretation": "Full-model temperature influence at the same receiver that omits this direct module pair; no other field-query receivers enter this functional.",
                    "position": position_derivative_probe(
                        model,
                        sample,
                        receiver_query,
                        device,
                        module_index=omitted_module_index,
                        metrics=("mean_temperature",),
                    ),
                    "heat": heat_derivative_probe(
                        model,
                        sample,
                        receiver_query,
                        device,
                        checkpoint,
                        module_index=omitted_module_index,
                        metrics=("mean_temperature",),
                    ),
                }
            row["status"] = _status_from_nested(
                row.get("position_full_model"),
                row.get("heat_full_model"),
                row.get("direct_fine_source"),
                row.get("same_omitted_module_full_model"),
            )
        except DiagnosticUnavailable as exc:
            row.update({"status": "unavailable", "reason": f"{type(exc).__name__}: {exc!s}"})
        except (RuntimeError, ValueError) as exc:
            row.update({"status": "failed", "reason": f"{type(exc).__name__}: {exc!s}"})
        rows.append(row)
    status_counts = _case_status_counts(rows)
    if status_counts["failed"]:
        status = "failed"
    elif status_counts["ok"]:
        status = "ok"
    else:
        status = "unavailable"
    return {
        "status": status,
        "status_counts": status_counts,
        "case_ids": [str(case_id) for case_id in case_ids],
        "query_count": int(query_count),
        "rows": rows,
        "conditional_fine_source_omitted_pair_fixture": conditional_fine_source_omitted_pair_fixture(),
        "support_transitions": support_transition_probe(),
        "interpretation": {
            "position": "full P0/P1/P2 physical wrapper path; signed mean-temperature and pressure-drop derivatives remain in normalized output units unless physical target scales are available",
            "heat": "full P0/P1/P2 physical wrapper path with live heat rebuilding local parameters; each row reports normalized h and physical h*heat_std when checkpoint input normalization statistics are available",
            "direct_vs_full": "direct fine-source rows freeze routing/context and are not interchangeable with full position/heat influence rows",
            "physical_certificate": "model AD/FD agreement is implementation self-consistency only; it is not a physical solver derivative certificate",
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="task", required=True)

    toy = subparsers.add_parser("toy", help="run float64 route and support-transition fixtures")
    toy.add_argument("--output", type=Path, required=True)
    toy.set_defaults(handler=lambda args: {"schema_version": SCHEMA_VERSION, "task": "toy", **run_toy_route_checks()})

    anchors = subparsers.add_parser("anchors", help="run exact real-anchor derivative checks")
    anchors.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    anchors.add_argument("--case-id", action="append", default=None, metavar="CASE_ID")
    anchors.add_argument("--dataset", default=None)
    anchors.add_argument("--split", default="test")
    anchors.add_argument("--device", default="auto")
    anchors.add_argument("--query-count", type=int, default=32)
    anchors.add_argument("--output", type=Path, required=True)
    anchors.set_defaults(handler=_run_anchor_command)
    return parser


def _run_anchor_command(args: argparse.Namespace) -> dict[str, Any]:
    specs = parse_checkpoint_specs(args.checkpoint)
    if len(specs) != 1:
        raise ValueError("anchors requires exactly one labelled checkpoint")
    if int(args.query_count) <= 0:
        raise ValueError("--query-count must be positive")
    device = select_device(args.device)
    spec: CheckpointSpec = specs[0]
    model, checkpoint = _load_model_spec(spec, device)
    dataset_args = argparse.Namespace(dataset=args.dataset, split=args.split)
    dataset, dataset_path = _load_dataset(checkpoint, dataset_args)
    try:
        payload = run_anchor_derivative_checks(
            model,
            checkpoint,
            dataset,
            device,
            case_ids=tuple(args.case_id or DEFAULT_ANCHOR_CASES),
            query_count=int(args.query_count),
        )
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
    payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "task": "anchors",
            "checkpoint": {
                "label": spec.label,
                "path": str(spec.path),
                "epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))),
            },
            "dataset": str(dataset_path),
            "device": str(device),
        }
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = args.handler(args)
    write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "central_difference",
    "compare_direct_and_full_influence",
    "fixed_route_gradcheck",
    "fixed_route_jvp",
    "heat_derivative_probe",
    "main",
    "position_derivative_probe",
    "routed_fine_source_probe",
    "run_anchor_derivative_checks",
    "run_toy_route_checks",
    "scalar_derivative_probe",
    "support_transition_probe",
    "vector_jvp",
    "write_json",
]

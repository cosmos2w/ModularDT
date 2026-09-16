"""CPU numerical checks for the Goal-1 module-hub routing algebra.

This helper is independent of model loading and training.  It evaluates one
small, fixed float64 fixture through the live routing operators

``source logits -> ordinary memberships -> induced hub measure -> query
source-measure density -> packed two-hop prior``

and compares the packed prior with the dense mathematical reference.  The
same fixture is used for finite-dimensional ``gradcheck``/JVP checks and for
the source and conditional-hub splitting identities.  A separate two-source
sparsemax boundary fixture records the actual support identities and the
one-sided continuity behavior.

Run from ``HONF_Proj/``::

    python tools/diagnostics/routing_numerical_checks.py \
        --study-root diagnostics/generated/interface_operator_study/dynamic_sparse_routing

The default artifact is written below ``<study-root>/numerics``.  A callable
``run_routing_numerical_checks`` is also provided for the dynamic routing
diagnostic driver; it never touches a checkpoint or allocates a GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from honf_forward_core.interface_fields.routing_index import (  # type: ignore[import-not-found]
    build_typed_source_incidence,
    compile_two_hop_pairs,
    ordinary_source_sparsemax,
    source_measure_sparsemax,
)

SCHEMA_VERSION = 1
DTYPE = torch.float64


def _fixture_inputs(*, requires_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the stable composition fixture shared with routing tests.

    The values match ``tests/test_routing_composition.py``.  All baseline
    positive supports are separated from a sparsemax support boundary, so
    numerical derivatives test the live algebra on a fixed support.
    """

    source_logits = torch.tensor(
        [[[0.2, -0.1, 0.5], [0.3, 0.1, -0.2]]],
        dtype=DTYPE,
        requires_grad=requires_grad,
    )
    query_logits = torch.tensor(
        [[[0.2, -0.1, 0.4], [1.3, -0.5, 0.2]]],
        dtype=DTYPE,
        requires_grad=requires_grad,
    )
    source_weights = torch.tensor(
        [[0.2, 0.8]],
        dtype=DTYPE,
        requires_grad=requires_grad,
    )
    return source_logits, query_logits, source_weights


def _dense_from_pairs(
    pairs: Any,
    *,
    batch: int,
    query_count: int,
    source_count: int,
    dtype: torch.dtype = DTYPE,
) -> torch.Tensor:
    """Materialize only the small fixture's packed pairs for comparison."""

    dense = torch.zeros(
        (batch, query_count, source_count),
        dtype=dtype,
        device=pairs.prior.device,
    )
    if pairs.prior.numel():
        dense[pairs.batch_index, pairs.receiver_index, pairs.source_index] = pairs.prior
    return dense


def _compose(
    source_logits: torch.Tensor,
    query_logits: torch.Tensor,
    source_weights: torch.Tensor,
    *,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Run the live sparse routing composition and return dense ``Pi``.

    The dense result is assembled from ``compile_two_hop_pairs`` only after
    positive paths have been deduplicated.  The returned components remain
    attached to the autograd graph when requested.
    """

    membership = ordinary_source_sparsemax(source_logits)
    incidence = build_typed_source_incidence(source_weights, membership)
    projection = source_measure_sparsemax(query_logits, incidence.hub_measure)
    pairs = compile_two_hop_pairs(projection.density, incidence)
    packed = _dense_from_pairs(
        pairs,
        batch=int(query_logits.shape[0]),
        query_count=int(query_logits.shape[1]),
        source_count=int(source_logits.shape[1]),
        dtype=pairs.prior.dtype,
    )
    if not return_components:
        return packed
    components: dict[str, Any] = {
        "membership": membership,
        "incidence": incidence,
        "hub_measure": incidence.hub_measure,
        "projection": projection,
        "density": projection.density,
        "probability": projection.probability,
        "pairs": pairs,
    }
    return packed, components


def _dense_reference(
    source_logits: torch.Tensor,
    query_logits: torch.Tensor,
    source_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Evaluate Equation (two-hop prior) without integer pair packing."""

    membership = ordinary_source_sparsemax(source_logits)
    hub_measure = torch.einsum(
        "bn,bnk->bk",
        source_weights,
        membership,
    )
    projection = source_measure_sparsemax(query_logits, hub_measure)
    dense = source_weights[:, None, :] * torch.einsum(
        "bqk,bnk->bqn",
        projection.density,
        membership,
    )
    return dense, {
        "membership": membership,
        "hub_measure": hub_measure,
        "projection": projection,
        "density": projection.density,
        "probability": projection.probability,
    }


def _float_summary(value: torch.Tensor) -> dict[str, Any]:
    """Return compact finite statistics for a CPU tensor."""

    detached = value.detach().to(device="cpu")
    finite = torch.isfinite(detached)
    selected = detached[finite]
    result: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "count": int(detached.numel()),
        "finite_count": int(finite.sum().item()),
        "nonzero_count": int((detached != 0).sum().item()),
    }
    if selected.numel():
        result.update(
            {
                "min": float(selected.min().item()),
                "max": float(selected.max().item()),
                "mean": float(selected.mean().item()),
                "l2": float(torch.linalg.vector_norm(selected).item()),
            }
        )
    return result


def _difference_summary(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | bool]:
    """Summarize an absolute/relative tensor difference."""

    delta = (left.detach() - right.detach()).abs()
    scale = right.detach().abs().clamp_min(torch.finfo(right.dtype).tiny)
    relative = delta / scale
    max_abs = float(delta.max().item()) if delta.numel() else 0.0
    max_relative = float(relative.max().item()) if relative.numel() else 0.0
    return {
        "max_abs": max_abs,
        "max_relative": max_relative,
        "finite": bool(torch.isfinite(delta).all().item() and torch.isfinite(relative).all().item()),
    }


def _gradient_record(name: str, value: torch.Tensor | None) -> dict[str, Any]:
    """Record whether one live composition variable receives a finite grad."""

    if value is None:
        return {"name": name, "available": False, "finite": False, "nonzero": False}
    finite = bool(torch.isfinite(value).all().item())
    nonzero = bool(torch.any(value != 0).item())
    result: dict[str, Any] = {
        "name": name,
        "available": True,
        "finite": finite,
        "nonzero": nonzero,
        "summary": _float_summary(value),
    }
    return result


def _live_gradient_check() -> dict[str, Any]:
    """Check gradients of all live route stages with one scalar objective."""

    source_logits, query_logits, source_weights = _fixture_inputs(requires_grad=True)
    packed, packed_components = _compose(
        source_logits,
        query_logits,
        source_weights,
        return_components=True,
    )
    assert isinstance(packed_components, dict)
    membership = packed_components["membership"]
    hub_measure = packed_components["hub_measure"]
    density = packed_components["density"]
    probability = packed_components["probability"]
    prior = packed_components["pairs"].prior
    # A nonconstant deterministic probe exercises every route output while
    # preserving the fixed support of the fixture.
    probe = torch.tensor([[[1.0, 1.7], [2.3, 0.4]]], dtype=DTYPE)
    objective = (
        (packed * probe).sum()
        + 0.13 * membership.square().sum()
        + 0.07 * probability.square().sum()
    )
    values = {
        "source_logits": source_logits,
        "query_logits": query_logits,
        "source_weights": source_weights,
        "membership_A": membership,
        "hub_measure_mu": hub_measure,
        "query_density_d": density,
        "query_probability_alpha": probability,
        "pair_prior_Pi": prior,
    }
    gradients = torch.autograd.grad(
        objective,
        tuple(values.values()),
        allow_unused=True,
        retain_graph=False,
    )
    records = [_gradient_record(name, gradient) for name, gradient in zip(values, gradients)]
    return {
        "objective": float(objective.detach().item()),
        "gradients": records,
        "all_available": all(record["available"] for record in records),
        "all_finite": all(record["finite"] for record in records),
        "all_expected_nonzero": all(record["nonzero"] for record in records),
    }


def _gradcheck_and_jvp() -> dict[str, Any]:
    """Run float64 reverse-mode gradcheck and nonconstant live JVPs."""

    source_logits, query_logits, source_weights = _fixture_inputs(requires_grad=True)

    def function(x: torch.Tensor, z: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        result = _compose(x, z, omega)
        assert isinstance(result, torch.Tensor)
        return result

    gradcheck_ok = bool(
        torch.autograd.gradcheck(
            function,
            (source_logits, query_logits, source_weights),
            eps=1.0e-6,
            atol=1.0e-5,
            rtol=1.0e-4,
        )
    )
    tangents = (
        torch.tensor(
            [[[0.3, -0.4, 0.2], [-0.2, 0.1, 0.5]]],
            dtype=DTYPE,
        ),
        torch.tensor(
            [[[0.3, -0.4, 0.2], [-0.2, 0.1, 0.5]]],
            dtype=DTYPE,
        ),
        torch.tensor([[0.15, -0.25]], dtype=DTYPE),
    )
    baseline = function(source_logits, query_logits, source_weights)
    _, joint_jvp = torch.autograd.functional.jvp(
        function,
        (source_logits, query_logits, source_weights),
        tangents,
        create_graph=False,
    )
    h = 1.0e-5
    plus = function(
        source_logits + h * tangents[0],
        query_logits + h * tangents[1],
        source_weights + h * tangents[2],
    )
    minus = function(
        source_logits - h * tangents[0],
        query_logits - h * tangents[1],
        source_weights - h * tangents[2],
    )
    finite_difference = (plus - minus) / (2.0 * h)
    joint_error = _difference_summary(joint_jvp, finite_difference)

    component_jvps: dict[str, dict[str, Any]] = {}
    labels = ("source_logits", "query_logits", "source_weights")
    inputs = (source_logits, query_logits, source_weights)
    for index, label in enumerate(labels):
        local_tangents = tuple(
            tangents[item] if item == index else torch.zeros_like(inputs[item])
            for item in range(3)
        )
        _, jvp = torch.autograd.functional.jvp(
            function,
            inputs,
            local_tangents,
            create_graph=False,
        )
        local_plus = function(*(value + h * tangent for value, tangent in zip(inputs, local_tangents)))
        local_minus = function(*(value - h * tangent for value, tangent in zip(inputs, local_tangents)))
        local_fd = (local_plus - local_minus) / (2.0 * h)
        component_jvps[label] = {
            "jvp": _float_summary(jvp),
            "finite_difference": _float_summary(local_fd),
            "error": _difference_summary(jvp, local_fd),
        }
    return {
        "gradcheck": {
            "ok": gradcheck_ok,
            "eps": 1.0e-6,
            "atol": 1.0e-5,
            "rtol": 1.0e-4,
        },
        "jvp": {
            "finite_difference_step": h,
            "joint": {
                "jvp": _float_summary(joint_jvp),
                "finite_difference": _float_summary(finite_difference),
                "error": joint_error,
            },
            "components": component_jvps,
        },
        "baseline": _float_summary(baseline),
    }


def _dense_reference_check() -> dict[str, Any]:
    """Compare packed/deduplicated Π with the dense two-hop expression."""

    source_logits, query_logits, source_weights = _fixture_inputs(requires_grad=True)
    packed, components = _compose(
        source_logits,
        query_logits,
        source_weights,
        return_components=True,
    )
    reference, reference_components = _dense_reference(
        source_logits,
        query_logits,
        source_weights,
    )
    pairs = components["pairs"]
    prior_difference = _difference_summary(packed, reference)
    # The query projection is a probability over the occupied hub measure;
    # the route density itself obeys the weighted source-measure constraint.
    density_mass = (reference_components["hub_measure"][:, None, :] * reference_components["density"]).sum(dim=-1)
    probability_mass = reference_components["probability"].sum(dim=-1)
    return {
        "packed_vs_dense": prior_difference,
        "packed_requires_grad": bool(pairs.prior.requires_grad),
        "membership_requires_grad": bool(components["membership"].requires_grad),
        "hub_measure_requires_grad": bool(components["hub_measure"].requires_grad),
        "density_requires_grad": bool(components["density"].requires_grad),
        "pair_counts": {
            "raw_paths": int(pairs.raw_path_count),
            "unique_pairs": int(pairs.unique_pair_count),
            "duplicate_expansion": float(pairs.duplicate_expansion),
        },
        "dense_reference": _float_summary(reference),
        "weighted_density_mass": _float_summary(density_mass),
        "weighted_density_mass_error": float((density_mass - 1.0).abs().max().item()),
        "query_probability_mass": _float_summary(probability_mass),
        "query_probability_mass_error": float((probability_mass - 1.0).abs().max().item()),
    }


def _effective_from_values(
    membership: torch.Tensor,
    source_weights: torch.Tensor,
    query_logits: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a fixed-membership effective prior for split identities."""

    incidence = build_typed_source_incidence(source_weights, membership)
    projection = source_measure_sparsemax(query_logits, incidence.hub_measure)
    pairs = compile_two_hop_pairs(projection.density, incidence)
    return _dense_from_pairs(
        pairs,
        batch=int(query_logits.shape[0]),
        query_count=int(query_logits.shape[1]),
        source_count=int(membership.shape[1]),
        dtype=pairs.prior.dtype,
    )


def _splitting_checks() -> dict[str, Any]:
    """Check source quadrature duplication and conditional hub splitting."""

    membership = torch.tensor(
        [[[0.6, 0.4, 0.0], [0.1, 0.2, 0.7], [0.0, 0.3, 0.7]]],
        dtype=DTYPE,
    )
    source_weights = torch.tensor([[0.2, 0.3, 0.5]], dtype=DTYPE)
    query_logits = torch.tensor(
        [[[1.4, -0.3, 0.2], [-0.4, 0.3, 0.1]]],
        dtype=DTYPE,
    )
    base = _effective_from_values(membership, source_weights, query_logits)

    split_membership = membership[:, [0, 0, 1, 2]]
    split_weights = torch.tensor([[0.07, 0.13, 0.3, 0.5]], dtype=DTYPE)
    source_split = _effective_from_values(split_membership, split_weights, query_logits)
    source_reconstructed = torch.stack(
        (
            source_split[..., 0] + source_split[..., 1],
            source_split[..., 2],
            source_split[..., 3],
        ),
        dim=-1,
    )
    source_error = _difference_summary(source_reconstructed, base)

    # Split the first hub's incidence conditionally.  Query logits duplicate
    # the first hub; ordinary source memberships are not rerun on a duplicated
    # raw candidate bank for this identity.
    conditional_membership = torch.cat(
        (membership[..., :1] * 0.35, membership[..., :1] * 0.65, membership[..., 1:]),
        dim=-1,
    )
    conditional_logits = torch.cat((query_logits[..., :1], query_logits), dim=-1)
    conditional_split = _effective_from_values(
        conditional_membership,
        source_weights,
        conditional_logits,
    )
    hub_error = _difference_summary(conditional_split, base)
    return {
        "source_quadrature_duplication": {
            "base_source_count": int(base.shape[-1]),
            "split_source_count": int(source_split.shape[-1]),
            "reconstructed": _float_summary(source_reconstructed),
            "error": source_error,
        },
        "conditional_hub_split": {
            "base_hub_count": int(membership.shape[-1]),
            "split_hub_count": int(conditional_membership.shape[-1]),
            "error": hub_error,
        },
    }


def _support_transition_check() -> dict[str, Any]:
    """Record exact supports and one-sided derivatives at a sparsemax entry."""

    h = 1.0e-6

    def projected(delta: torch.Tensor) -> torch.Tensor:
        logits = torch.stack((1.0 + delta, delta.new_zeros(()))).reshape(1, 2)
        return ordinary_source_sparsemax(logits)

    def projected_at(value: float, *, with_grad: bool = False) -> torch.Tensor:
        delta = torch.tensor(value, dtype=DTYPE, requires_grad=with_grad)
        return projected(delta)

    center = projected_at(0.0)
    left = projected_at(-h)
    right = projected_at(h)
    left_support = torch.nonzero(left[0] > 0.0, as_tuple=False).flatten().tolist()
    center_support = torch.nonzero(center[0] > 0.0, as_tuple=False).flatten().tolist()
    right_support = torch.nonzero(right[0] > 0.0, as_tuple=False).flatten().tolist()
    left_derivative = (center - left) / h
    right_derivative = (right - center) / h

    # These are local AD derivatives on each fixed-support side.  At the
    # exact boundary the implementation chooses its deterministic local
    # branch; the two one-sided finite differences remain the reportable
    # derivatives for the nonsmooth transition.
    left_delta = torch.tensor(-h, dtype=DTYPE, requires_grad=True)
    left_projection = projected(left_delta)
    left_ad = torch.autograd.grad(left_projection[0, 0], left_delta, retain_graph=False)[0]
    right_delta = torch.tensor(h, dtype=DTYPE, requires_grad=True)
    right_projection = projected(right_delta)
    right_ad = torch.autograd.grad(right_projection[0, 0], right_delta, retain_graph=False)[0]
    center_delta = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
    center_projection = projected(center_delta)
    center_ad = torch.autograd.grad(center_projection[0, 0], center_delta, retain_graph=False)[0]

    continuity_gap = float((left - right).abs().max().item())
    return {
        "step": h,
        "logits": {
            "left": [1.0 - h, 0.0],
            "center": [1.0, 0.0],
            "right": [1.0 + h, 0.0],
        },
        "support_identities": {
            "left": [int(value) for value in left_support],
            "center": [int(value) for value in center_support],
            "right": [int(value) for value in right_support],
        },
        "projected_values": {
            "left": left[0].tolist(),
            "center": center[0].tolist(),
            "right": right[0].tolist(),
        },
        "one_sided_finite_difference": {
            "left_to_center": left_derivative[0].tolist(),
            "center_to_right": right_derivative[0].tolist(),
        },
        "local_autograd_first_coordinate": {
            "left": float(left_ad.item()),
            "center": float(center_ad.item()),
            "right": float(right_ad.item()),
        },
        "continuity": {
            "max_left_right_gap": continuity_gap,
            "max_left_center_gap": float((left - center).abs().max().item()),
            "max_center_right_gap": float((center - right).abs().max().item()),
            "continuous_at_step": bool(continuity_gap <= h),
        },
        "interpretation": (
            "The center support is a deterministic local branch; support entry is "
            "assessed with the two one-sided derivatives."
        ),
    }


def _json_value(value: Any) -> Any:
    """Convert nested tensors/scalars to strict JSON values."""

    if torch.is_tensor(value):
        if value.numel() == 1:
            return _json_value(value.detach().cpu().item())
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_routing_numerical_checks(
    *,
    output: Path | None = None,
    study_root: Path | None = None,
) -> dict[str, Any]:
    """Run all CPU checks and optionally save one JSON artifact.

    ``output`` takes precedence.  Otherwise, when ``study_root`` is given,
    the artifact is written to ``study_root / "numerics" /
    "routing_numerical_checks.json"``.  With neither argument the checks are
    returned without filesystem output.
    """

    if output is not None and study_root is not None:
        raise ValueError("Pass output or study_root, not both.")
    with torch.inference_mode(False):
        dense_reference = _dense_reference_check()
        gradients = _live_gradient_check()
        derivatives = _gradcheck_and_jvp()
        splitting = _splitting_checks()
        support_transition = _support_transition_check()

    checks = {
        "dense_reference": dense_reference,
        "live_gradients": gradients,
        "gradcheck_and_jvp": derivatives,
        "splitting_invariance": splitting,
        "support_transition": support_transition,
    }
    pass_conditions = {
        "dense_reference": dense_reference["packed_vs_dense"]["finite"]
        and dense_reference["packed_vs_dense"]["max_abs"] <= 1.0e-12
        and dense_reference["weighted_density_mass_error"] <= 1.0e-12
        and dense_reference["query_probability_mass_error"] <= 1.0e-12,
        "live_gradients": gradients["all_available"] and gradients["all_finite"],
        "gradcheck": derivatives["gradcheck"]["ok"],
        "jvp": derivatives["jvp"]["joint"]["error"]["finite"]
        and derivatives["jvp"]["joint"]["error"]["max_abs"] <= 1.0e-7,
        "splitting": all(
            value["error"]["finite"] and value["error"]["max_abs"] <= 1.0e-12
            for value in splitting.values()
        ),
        "support_transition": support_transition["continuity"]["continuous_at_step"]
        and support_transition["support_identities"] == {
            "left": [0, 1],
            "center": [0],
            "right": [0],
        },
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": "routing_numerical_checks",
        "status": "passed" if all(pass_conditions.values()) else "failed",
        "device": "cpu",
        "dtype": str(DTYPE),
        "checks": checks,
        "pass_conditions": pass_conditions,
        "interpretation": {
            "route_values": (
                "Memberships, hub measures, query densities, probabilities, and Pi are "
                "routing quantities; no hub descriptor is used as a field-value substitute."
            ),
            "support": (
                "Sparsemax support identities are integer branch decisions. The center "
                "derivative is local to the deterministic branch; one-sided derivatives "
                "are reported at entry."
            ),
            "reference": (
                "The packed result is compared with omega[i] * sum_k A[i,k] * d[q,k], "
                "after positive receiver-source path deduplication."
            ),
        },
    }
    target: Path | None = None
    if output is not None:
        target = Path(output).expanduser().resolve()
    elif study_root is not None:
        target = Path(study_root).expanduser().resolve() / "numerics" / "routing_numerical_checks.json"
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload["output"] = str(target)
        target.write_text(
            json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return _json_value(payload)


# A short alias makes the callable convenient for an orchestrator that already
# uses ``run_checks`` names while keeping the specific public API discoverable.
run_checks = run_routing_numerical_checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root",
        type=Path,
        default=Path("diagnostics/generated/interface_operator_study/dynamic_sparse_routing"),
        help="Existing-style study root; output goes under its numerics directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit JSON path; takes precedence over --study-root.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_routing_numerical_checks(output=args.output, study_root=None if args.output else args.study_root)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

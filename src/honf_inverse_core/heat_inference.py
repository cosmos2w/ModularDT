"""Frozen-surrogate fixed-total heat inference with complete numerical trails.

The caller owns the differentiable heat tensor and predictor. Observations,
held sensors and public total are separate inputs; hidden individual heat
allocations are never required. Graph and size-matched random block updates
share their initialization and step budget. This is surrogate optimization,
not independent physical validation or a conditional generative model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter

import torch

from .models.centered_simplex_velocity import _project_simplex

HeatPrediction = Callable[[torch.Tensor], Mapping[str, object]]


class UnsupportedHeatTotal(ValueError):
    """The public total has no allocation inside the declared heat bounds."""


def heat_feasibility_tolerance(total, active_count, heat_bounds, *, dtype):
    """Four native rounding units for sum, cast and fraction restoration.

    Bounds stay fixed. A boundary public sum can differ from its exact real
    sum by this amount; retained physical allocations must expose their signed
    total residual and stay inside the original representable heat caps.
    """
    lower, upper = (float(value) for value in heat_bounds)
    scale = max(1., abs(float(total)), int(active_count) * max(abs(lower), abs(upper)))
    return 4. * torch.finfo(dtype).eps * scale


@torch.no_grad()
def project_capped_heat(values: torch.Tensor, active: torch.Tensor, total: torch.Tensor,
                        heat_bounds: tuple[float, float], *, native_dtype=None) -> torch.Tensor:
    """Project onto physical heat caps and a native-precision fixed total.

    Solve the scalar water-filling threshold in FP64, then cast the completed
    allocation. Feasible native inputs remain unchanged when their signed
    real-sum residual fits ``heat_feasibility_tolerance``; no heat cap moves.
    This helper is for optimizer proposals, not a differentiable
    physical response or a topology derivative.
    """
    lower, upper = (float(value) for value in heat_bounds)
    if not 0 <= lower <= upper or not torch.isfinite(torch.tensor([lower, upper], dtype=torch.float64)).all():
        raise ValueError("Heat bounds must be finite, nonnegative and ordered")
    if values.ndim != 1 or values.shape != active.shape or total.numel() != 1:
        raise ValueError("Physical heat and active mask must be [M], with scalar total")
    if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(total).all()):
        raise ValueError("Heat projection requires finite inputs")
    ids = torch.nonzero(active.bool(), as_tuple=False).flatten()
    count, requested = ids.numel(), float(total)
    if not count:
        raise ValueError("Heat projection requires active modules")
    tolerance = heat_feasibility_tolerance(total, count, heat_bounds, dtype=native_dtype or values.dtype)
    if requested < count * lower - tolerance or requested > count * upper + tolerance:
        raise UnsupportedHeatTotal(
            f"Public total {requested:.17g} is outside [{count * lower:.17g}, {count * upper:.17g}] "
            f"for {count} active modules and frozen heat bounds [{lower:.17g}, {upper:.17g}] "
            f"(native summation tolerance {tolerance:.17g})")
    # Preserve an already feasible native allocation, including an endpoint
    # whose FP32 public sum rounds to the neighboring representable number.
    selected = values[ids]
    if bool(((selected >= lower) & (selected <= upper)).all()) and abs(float(
            selected.double().sum() - total.double())) <= tolerance:
        output = torch.zeros_like(values)
        output[ids] = selected
        return output
    source = values[ids].to(torch.float64)
    left, right = source.min() - upper, source.max() - lower
    # Endpoint rounding affects total feasibility, never the physical caps.
    target = total.to(device=values.device, dtype=torch.float64).reshape(()).clamp(count * lower, count * upper)
    for _ in range(80):
        threshold = (left + right) * .5
        above = (source - threshold).clamp(lower, upper).sum() > target
        left = torch.where(above, threshold, left)
        right = torch.where(above, right, threshold)
    projected = (source - (left + right) * .5).clamp(lower, upper)
    output = torch.zeros_like(values)
    output[ids] = projected.to(values.dtype)
    if abs(float(output.double().sum() - total.double())) > tolerance:
        raise RuntimeError("Capped projection total residual exceeds the declared native summation tolerance")
    return output


def public_heat_starts(active, total, *, starts=2, seed=0, heat_bounds=None):
    """Uniform and seeded public allocations; no hidden heat is accepted."""
    if starts < 1:
        raise ValueError("At least one public start is required")
    ids = torch.nonzero(active.bool(), as_tuple=False).flatten()
    if not ids.numel():
        raise ValueError("Public heat starts require active modules")
    initial = active.to(total.dtype) / ids.numel()
    generator = torch.Generator().manual_seed(int(seed))
    output = []
    for index in range(starts):
        fraction = initial.clone()
        if index:
            draw = torch.rand(ids.numel(), generator=generator, dtype=total.dtype).to(active.device) + .2
            fraction[ids] = draw / draw.sum()
        heat = fraction * total
        if heat_bounds is not None:
            heat = project_capped_heat(heat, active, total, heat_bounds)
        output.append(heat)
    return tuple(output)


def fixed_total_basis(active: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    """Orthonormal Helmert columns in physical module order; M=1 has none."""
    ids = torch.nonzero(active.bool(), as_tuple=False).flatten()
    basis = torch.zeros((active.numel(), max(0, ids.numel() - 1)), device=active.device, dtype=dtype)
    for column in range(basis.shape[1]):
        denominator = float((column + 1) * (column + 2)) ** .5
        basis[ids[:column+1], column] = 1.0 / denominator
        basis[ids[column+1], column] = -(column + 1) / denominator
    return basis


def observation_identifiability(predictor: HeatPrediction, heat: torch.Tensor,
                              active: torch.Tensor) -> dict[str, torch.Tensor | int | float]:
    """Local observation Jacobian on fixed-total directions, before recovery."""
    basis = fixed_total_basis(active, dtype=heat.dtype)
    if basis.shape[1] == 0:
        return {"jacobian": heat.new_zeros((0, 0)), "singular_values": heat.new_zeros(0),
                "rank": 0, "free_dimensions": 0, "condition_number": 1.0, "rank_tolerance": 0.0}
    jacobian = torch.autograd.functional.jacobian(
        lambda values: predictor(values)["observed"].reshape(-1), heat, vectorize=False)
    restricted = jacobian @ basis
    singular = torch.linalg.svdvals(restricted)
    tolerance = (float(singular.max()) * max(restricted.shape) * torch.finfo(heat.dtype).eps) if singular.numel() else 0.0
    rank = int((singular > tolerance).sum())
    smallest = float(singular.min()) if singular.numel() else 0.0
    condition = float(singular.max()) / smallest if smallest > tolerance and rank == basis.shape[1] else float("inf")
    return {"jacobian": restricted.detach(), "singular_values": singular.detach(), "rank": rank,
            "free_dimensions": basis.shape[1], "condition_number": condition, "rank_tolerance": tolerance}


def _rmse(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if values.shape != reference.shape or not values.numel():
        raise ValueError("Sensor prediction/reference must be nonempty and shape aligned")
    return (values - reference).square().mean().sqrt()


def _valid_groups(groups: object, active: torch.Tensor) -> tuple[torch.Tensor, ...]:
    valid = active.bool()
    output = []
    for group in (groups if isinstance(groups, Sequence) else ()):
        ids = torch.as_tensor(group, device=active.device, dtype=torch.long).flatten().unique(sorted=True)
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= active.numel()):
            raise ValueError("Graph update group names an invalid physical module row")
        ids = ids[valid[ids]]
        if ids.numel() >= 2:
            output.append(ids)
    return tuple(output)


@dataclass(frozen=True)
class HeatInferenceTrail:
    mode: str
    heat: tuple[torch.Tensor, ...]
    iterations: tuple[int, ...]
    observed_rmse: tuple[float, ...]
    held_rmse: tuple[float, ...]
    observed_predictions: tuple[torch.Tensor, ...]
    held_predictions: tuple[torch.Tensor, ...]
    peaks: tuple[torch.Tensor, ...]
    pressure: tuple[torch.Tensor, ...]
    selected_modules: tuple[torch.Tensor, ...]
    elapsed_seconds: tuple[float, ...]
    optimizer_steps: int
    meaningful_graph_steps: int
    group_fallback_steps: int
    accepted_steps: int = 0
    rejected_steps: int = 0
    forward_calls: tuple[int, ...] = ()
    vjp_calls: tuple[int, ...] = ()
    trial_evaluations: tuple[int, ...] = ()
    step_receipts: tuple[dict, ...] = ()
    status: str = "completed"
    unsupported_reason: str | None = None
    heat_bounds: tuple[float, float] | None = None
    feasibility_tolerance: float | None = None
    total_residual: tuple[float, ...] = ()
    bound_excess: tuple[float, ...] = ()


def fixed_total_heat_inference(predictor: HeatPrediction, observed: torch.Tensor,
                              held: torch.Tensor, active: torch.Tensor, total: torch.Tensor,
                              initial_fraction: torch.Tensor, *, mode: str = "joint",
                              steps: int = 30, learning_rate: float = .05,
                              block_stream: tuple[float, ...] | None = None,
                              permutation_stream: tuple[torch.Tensor, ...] | None = None,
                              block_size_stream: tuple[int, ...] | None = None) -> HeatInferenceTrail:
    """Projected Adam with fixed-total block proposals and same-sized controls.

    ``graph`` selects an actual input-only source group. ``ungrouped`` chooses
    the number of active modules recorded on the graph trajectory when
    ``block_size_stream`` is supplied, from a caller-specified permutation
    stream. Otherwise sizes follow its own current groups. The group is fixed during each proposal; the next prediction
    rebuilds connectivity. If no useful group exists, use a full joint step
    and report the fallback separately.
    """
    if mode not in {"joint", "graph", "ungrouped"} or steps < 0 or learning_rate <= 0:
        raise ValueError("Valid update mode, nonnegative steps and positive learning rate are required")
    if active.ndim != 1 or initial_fraction.shape != active.shape or total.numel() != 1:
        raise ValueError("Active modules/fractions must be [M] and supplied total scalar")
    if not bool(active.bool().any()) or not torch.isfinite(total).all() or float(total) < 0:
        raise ValueError("At least one active module and finite nonnegative total are required")
    if not torch.isfinite(initial_fraction).all():
        raise ValueError("Initial fraction must be finite")
    fractions = _project_simplex(initial_fraction[None], active[None])[0].detach().requires_grad_()
    optimizer = torch.optim.Adam((fractions,), lr=learning_rate)
    ids = torch.nonzero(active.bool(), as_tuple=False).flatten()
    deterministic = ids.numel() == 1 or float(total) == 0
    if block_size_stream is not None:
        if mode != "ungrouped":
            raise ValueError("A recorded block-size stream applies only to ungrouped controls")
        if not deterministic and (len(block_size_stream) < steps or any(
                not isinstance(size, int) or not 2 <= size <= ids.numel() for size in block_size_stream[:steps])):
            raise ValueError("Recorded block sizes must cover every step and lie between two and active M")
    block_stream = block_stream or tuple(0.0 for _ in range(steps))
    permutation_stream = permutation_stream or tuple(ids for _ in range(steps))
    if len(block_stream) < steps or len(permutation_stream) < steps:
        raise ValueError("Caller update streams must cover every optimization step")
    values, obs_error, held_error, obs_predictions, held_predictions = [], [], [], [], []
    peaks, pressure, selected, elapsed = [], [], [], []
    meaningful, fallbacks, actual_steps = 0, 0, 0
    start = perf_counter()
    for iteration in range((0 if deterministic else steps) + 1):
        heat = fractions * total
        prediction = predictor(heat)
        obs, unseen = prediction["observed"], prediction["held"]
        if not torch.is_tensor(obs) or not torch.is_tensor(unseen) or not torch.isfinite(obs).all() or not torch.isfinite(unseen).all():
            raise ValueError("Surrogate sensor predictions must be finite tensors")
        values.append(heat.detach().clone())
        obs_error.append(float(_rmse(obs, observed).detach()))
        held_error.append(float(_rmse(unseen, held).detach()))
        obs_predictions.append(obs.detach().clone())
        held_predictions.append(unseen.detach().clone())
        peaks.append(torch.as_tensor(prediction.get("peaks", []), device=heat.device).detach().clone())
        pressure.append(torch.as_tensor(prediction.get("pressure", []), device=heat.device).detach().clone())
        if heat.is_cuda:
            torch.cuda.synchronize(heat.device)
        elapsed.append(perf_counter() - start)
        if iteration == steps or deterministic:
            break
        optimizer.zero_grad(set_to_none=True)
        (obs - observed).square().mean().backward()
        if fractions.grad is None or not torch.isfinite(fractions.grad).all():
            raise RuntimeError("Caller-owned heat has no finite observation gradient")
        groups = _valid_groups(prediction.get("groups", ()), active)
        chosen = ids
        if mode != "joint":
            nontrivial = tuple(group for group in groups if group.numel() < ids.numel())
            if mode == "ungrouped" and block_size_stream is not None:
                size = block_size_stream[iteration]
                meaningful += int(size < ids.numel())
                fallbacks += int(size == ids.numel())
                permutation = torch.as_tensor(permutation_stream[iteration], device=ids.device)
                if not torch.equal(permutation.sort().values, ids):
                    raise ValueError("Ungrouped stream must permute every active physical module exactly once")
                chosen = permutation[:size]
            elif nontrivial:
                index = min(len(nontrivial) - 1, int(block_stream[iteration] * len(nontrivial)))
                group = nontrivial[index]
                meaningful += 1
                if mode == "graph":
                    chosen = group
                else:
                    permutation = torch.as_tensor(permutation_stream[iteration], device=ids.device)
                    if not torch.equal(permutation.sort().values, ids):
                        raise ValueError("Ungrouped stream must permute every active physical module exactly once")
                    chosen = permutation[:group.numel()]
            else:
                fallbacks += 1
        # The proposal is a physical set. Canonical slot order also makes a
        # full-size random control use the same FP32 reductions/projection as
        # the joint fallback, rather than injecting permutation roundoff.
        chosen = chosen.sort().values
        selected.append(chosen.detach().clone())
        previous = fractions.detach().clone()
        mask = torch.zeros_like(active, dtype=torch.bool)
        mask[chosen] = True
        fractions.grad = torch.where(mask, fractions.grad, torch.zeros_like(fractions.grad))
        optimizer.step()
        with torch.no_grad():
            # Adam momentum cannot update outside this proposal group.
            fractions[~mask] = previous[~mask]
            subtotal = previous[chosen].sum()
            if float(subtotal) > 0:
                projection = _project_simplex((fractions[chosen] / subtotal)[None],
                                             torch.ones((1, chosen.numel()), device=heat.device, dtype=torch.bool))[0]
                fractions[chosen] = projection * subtotal
            else:
                fractions[chosen] = 0
            fractions[~active.bool()] = 0
        actual_steps += 1
    return HeatInferenceTrail(mode, tuple(values), tuple(range(len(values))), tuple(obs_error), tuple(held_error),
        tuple(obs_predictions), tuple(held_predictions), tuple(peaks), tuple(pressure), tuple(selected), tuple(elapsed),
        actual_steps, meaningful, fallbacks)


def bounded_trust_heat_inference(predictor: HeatPrediction, observed: torch.Tensor,
                                held: torch.Tensor, active: torch.Tensor, total: torch.Tensor,
                                initial_fraction: torch.Tensor, *, mode="joint", steps=10,
                                learning_rate=.05, proposal_predictor=None,
                                block_stream=None, permutation_stream=None,
                                block_size_stream=None, heat_bounds=None,
                                state_callback=None) -> HeatInferenceTrail:
    """Simplex-tangent trust proposals, at most two observed-only trials.

    ``learning_rate`` is a maximum change in an allocation fraction per local
    proposal. The first/half-radius trials recompute continuous physics with
    the anchor's declared discrete topology. A provisionally improving trial
    is rebuilt under the ordinary operator; a worse rebuilt observed value
    is rejected. Held observations never select a proposal or step radius.
    Every attempted forward (including invalid topology) and VJP is charged.
    Optional physical ``heat_bounds`` apply to starts and every block proposal.
    An infeasible public total returns an unsupported trail before model access.
    ``state_callback`` receives each retained state immediately, including the
    initial state, accepted steps, rejected steps and cumulative call charges.
    """
    if mode not in {"joint", "graph", "ungrouped"} or not 0 <= steps <= 10 or not 0 < learning_rate <= .25:
        raise ValueError("Trust inference requires a declared mode, at most ten steps and radius in (0,.25].")
    if active.ndim != 1 or initial_fraction.shape != active.shape or total.numel() != 1:
        raise ValueError("Active modules/fractions must be [M] and supplied total scalar")
    if not bool(active.bool().any()) or not bool(torch.isfinite(total).all()) or float(total) < 0:
        raise ValueError("At least one active module and finite nonnegative total are required")
    if not bool(torch.isfinite(initial_fraction).all()):
        raise ValueError("Initial fraction must be finite")
    ids = torch.nonzero(active.bool(), as_tuple=False).flatten()
    block_stream = tuple(0. for _ in range(steps)) if block_stream is None else block_stream
    permutation_stream = tuple(ids for _ in range(steps)) if permutation_stream is None else permutation_stream
    if len(block_stream) < steps or len(permutation_stream) < steps:
        raise ValueError("Caller update streams must cover every trust step")
    if block_size_stream is not None and (mode != "ungrouped" or len(block_size_stream) < steps or
        any(not isinstance(size, int) or not 2 <= size <= ids.numel() for size in block_size_stream[:steps])):
        raise ValueError("Recorded block sizes must cover every random-control step")
    if heat_bounds is None:
        fractions = _project_simplex(initial_fraction[None], active[None])[0]
    else:
        try:
            initial_heat = project_capped_heat(initial_fraction * total, active, total, heat_bounds)
        except UnsupportedHeatTotal as exc:
            return HeatInferenceTrail(mode, (), (), (), (), (), (), (), (), (), (),
                0, 0, 0, status="unsupported", unsupported_reason=str(exc),
                heat_bounds=tuple(float(value) for value in heat_bounds),
                feasibility_tolerance=heat_feasibility_tolerance(total, ids.numel(), heat_bounds,
                                                               dtype=initial_fraction.dtype))
        # Wide allocation ratios restore FP32 physical cap endpoints exactly
        # after division/multiplication, without changing physical kernels.
        fractions = initial_heat.double() / total.double() if float(total) > 0 else torch.zeros_like(initial_fraction).double()
    fractions = fractions.detach().requires_grad_()
    forward_count, vjp_count = 0, 0
    tolerance = (heat_feasibility_tolerance(total, ids.numel(), heat_bounds, dtype=initial_fraction.dtype)
                 if heat_bounds is not None else None)
    residuals, bound_excesses = [], []

    def physical_heat(values):
        heat = values * total
        if heat_bounds is not None:
            heat = heat.to(initial_fraction.dtype)
            lower, upper = heat_bounds
            excess = torch.maximum((lower - heat[ids]).clamp_min(0).max(),
                                   (heat[ids] - upper).clamp_min(0).max())
            if float(excess) != 0. or abs(float(heat.double().sum() - total.double())) > tolerance:
                raise RuntimeError("Fraction restoration violated frozen heat caps or native summation tolerance")
        return heat

    def call(values, reference=None, *, proposal=False):
        nonlocal forward_count
        forward_count += 1
        if proposal and proposal_predictor is not None:
            return proposal_predictor(physical_heat(values), reference)
        return predictor(physical_heat(values))

    heat_rows, obs_error, held_error, obs_rows, held_rows = [], [], [], [], []
    peaks, pressure, selected, elapsed, forwards, vjps, trials, receipts = [], [], [], [], [], [], [], []
    meaningful, fallbacks, accepted = 0, 0, 0
    started = perf_counter()

    def topology_difference(before, after):
        if before is None or after is None:
            return None
        changed = 0
        for phase, prior in before.states.items():
            current = after.states[phase]
            for tau, gates in prior.strategy_data.get("gates", {}).items():
                changed += int((gates > 0).logical_xor(current.strategy_data["gates"][tau] > 0).sum())
            for tau, members in prior.control_memberships.items():
                for kind, density in members.items():
                    changed += int((density > 0).logical_xor(current.control_memberships[tau][kind] > 0).sum())
        for key, access in before.accesses.items():
            if key not in after.accesses or access["shape"] != after.accesses[key]["shape"]:
                return {"changed": True, "receiver_stream_changed": True}
            changed += int((access["edge_access"] > 0).logical_xor(after.accesses[key]["edge_access"] > 0).sum())
        return {"changed": changed > 0, "changed_discrete_entries": changed,
                "scope": "split gates, both control-donor supports and receiver-edge supports"}

    def append(prediction):
        obs, unseen = prediction["observed"], prediction["held"]
        if not torch.is_tensor(obs) or not torch.is_tensor(unseen) or not bool(torch.isfinite(obs).all() and torch.isfinite(unseen).all()):
            raise ValueError("Surrogate sensor predictions must be finite tensors")
        retained_heat = physical_heat(fractions).detach().clone()
        heat_rows.append(retained_heat)
        residuals.append(float(retained_heat.double().sum() - total.double()))
        if heat_bounds is not None:
            lower, upper = heat_bounds
            bound_excesses.append(float(torch.maximum((lower-retained_heat[ids]).clamp_min(0).max(),
                                                       (retained_heat[ids]-upper).clamp_min(0).max())))
        else:
            bound_excesses.append(0.)
        obs_error.append(float(_rmse(obs, observed).detach()))
        held_error.append(float(_rmse(unseen, held).detach()))
        obs_rows.append(obs.detach().clone())
        held_rows.append(unseen.detach().clone())
        peaks.append(torch.as_tensor(prediction.get("peaks", []), device=fractions.device).detach().clone())
        pressure.append(torch.as_tensor(prediction.get("pressure", []), device=fractions.device).detach().clone())
        if fractions.is_cuda:
            torch.cuda.synchronize(fractions.device)
        elapsed.append(perf_counter() - started)
        forwards.append(forward_count)
        vjps.append(vjp_count)
        if state_callback is not None:
            state_callback({"iteration": len(heat_rows) - 1,
                "heat": heat_rows[-1], "observed": obs_rows[-1], "held": held_rows[-1],
                "observed_rmse": obs_error[-1], "held_rmse": held_error[-1],
                "forward_calls": forward_count, "vjp_calls": vjp_count,
                "accepted_steps": accepted, "receipt": receipts[-1] if receipts else None,
                "total_residual": residuals[-1], "bound_excess": bound_excesses[-1],
                "feasibility_tolerance": tolerance})

    prediction = call(fractions)
    append(prediction)
    attempted = 0 if ids.numel() == 1 or float(total) == 0 else steps
    for iteration in range(attempted):
        objective = (prediction["observed"] - observed).square().mean()
        gradient, = torch.autograd.grad(objective, fractions)
        vjp_count += 1
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("Caller-owned heat has no finite observation gradient")
        groups = tuple(group for group in _valid_groups(prediction.get("groups", ()), active) if group.numel() < ids.numel())
        chosen = ids
        if mode == "graph":
            if groups:
                chosen = groups[min(len(groups)-1, int(block_stream[iteration] * len(groups)))]
                meaningful += 1
            else:
                fallbacks += 1
        elif mode == "ungrouped":
            size = block_size_stream[iteration] if block_size_stream is not None else (
                int(groups[min(len(groups)-1, int(block_stream[iteration] * len(groups)))].numel()) if groups else int(ids.numel()))
            permutation = torch.as_tensor(permutation_stream[iteration], device=ids.device)
            if not torch.equal(permutation.sort().values, ids):
                raise ValueError("Random control must permute every active physical module exactly once")
            chosen = permutation[:size]
            meaningful += int(size < ids.numel())
            fallbacks += int(size == ids.numel())
        chosen = chosen.sort().values
        selected.append(chosen.detach().clone())
        previous = fractions.detach().clone()
        subtotal = previous[chosen].sum()
        tangent = gradient[chosen] - gradient[chosen].mean()
        scale = tangent.abs().max()
        direction = -tangent / torch.where(scale > 0, scale, torch.ones_like(scale))
        reference = prediction.get("topology_record")
        receipt = {"attempt": iteration + 1, "selected_modules": chosen.detach().cpu().tolist(),
            "selection": mode, "full_joint_fallback": mode != "joint" and chosen.numel() == ids.numel(),
            "anchor_observed_mse": float(objective.detach()), "trials": [], "accepted": False,
            "topology_scope": "discrete topology fixed; continuous states/memberships/controls recomputed" if reference is not None else "full-access/no discrete organizer"}
        receipt["graph_block_provenance"] = next((row for row in prediction.get("control_donor_receipts", ())
            if row["module_value_or_control_donors"] == chosen.detach().cpu().tolist()), None) if mode == "graph" else None
        accepted_prediction = None
        attempted_trials = 0
        if float(scale) > 0 and float(subtotal) > 0:
            for trial, radius in enumerate((learning_rate, learning_rate * .5), 1):
                candidate = previous.clone()
                if heat_bounds is None:
                    projected = _project_simplex(((previous[chosen] + radius * direction) / subtotal)[None],
                        torch.ones((1, chosen.numel()), device=active.device, dtype=torch.bool))[0]
                    candidate[chosen] = projected * subtotal
                else:
                    block_heat = (previous[chosen] + radius * direction) * total
                    candidate[chosen] = project_capped_heat(block_heat,
                        torch.ones_like(chosen, dtype=torch.bool), subtotal * total, heat_bounds,
                        native_dtype=initial_fraction.dtype) / total
                actual_delta = candidate[chosen] - previous[chosen]
                largest = actual_delta.abs().max()
                if float(largest) > radius:
                    candidate[chosen] = previous[chosen] + actual_delta * (radius / largest)
                candidate[~active.bool()] = 0
                attempted_trials += 1
                row = {"trial": trial, "fraction_radius": radius, "topology_valid": True}
                try:
                    with torch.no_grad():
                        proposed = call(candidate, reference, proposal=True)
                    score = float((proposed["observed"] - observed).square().mean())
                    row["observed_mse"] = score
                except ValueError as exc:
                    # Only the maintained active-topology validity exception
                    # is a trial rejection. All other coding/data errors fail.
                    from honf_forward_core.interface_fields.topology_probe import FixedTopologyInvalid
                    if not isinstance(exc, FixedTopologyInvalid):
                        raise
                    row.update(topology_valid=False, rejected_reason=str(exc))
                    receipt["trials"].append(row)
                    continue
                receipt["trials"].append(row)
                if not torch.isfinite(torch.tensor(score)) or score >= float(objective.detach()):
                    continue
                rebuilt_fraction = candidate.detach().requires_grad_()
                rebuilt = call(rebuilt_fraction)
                rebuilt_score = float((rebuilt["observed"] - observed).square().mean().detach())
                row["rebuilt_observed_mse"] = rebuilt_score
                if torch.isfinite(torch.tensor(rebuilt_score)) and rebuilt_score < float(objective.detach()):
                    fractions = rebuilt_fraction
                    accepted_prediction = rebuilt
                    receipt["accepted"] = True
                    receipt["ordinary_rebuild_topology_change"] = topology_difference(reference, rebuilt.get("topology_record"))
                    accepted += 1
                    break
                row["rejected_reason"] = "ordinary rebuild did not improve observed objective"
        trials.append(attempted_trials)
        if accepted_prediction is None:
            fractions = previous.detach().requires_grad_()
            # The previous graph was consumed by its VJP. A fresh ordinary
            # call provides the next gradient and is explicitly charged.
            prediction = call(fractions)
            receipt["rejected_reason"] = "no valid improving trial" if attempted_trials else "zero feasible tangent"
        else:
            prediction = accepted_prediction
        receipts.append(receipt)
        append(prediction)
    return HeatInferenceTrail(mode, tuple(heat_rows), tuple(range(len(heat_rows))), tuple(obs_error), tuple(held_error),
        tuple(obs_rows), tuple(held_rows), tuple(peaks), tuple(pressure), tuple(selected), tuple(elapsed),
        attempted, meaningful, fallbacks, accepted, attempted - accepted, tuple(forwards), tuple(vjps),
        tuple(trials), tuple(receipts), heat_bounds=tuple(float(value) for value in heat_bounds) if heat_bounds is not None else None,
        feasibility_tolerance=tolerance, total_residual=tuple(residuals), bound_excess=tuple(bound_excesses))


__all__ = ["HeatInferenceTrail", "UnsupportedHeatTotal", "bounded_trust_heat_inference", "fixed_total_basis", "fixed_total_heat_inference", "heat_feasibility_tolerance", "observation_identifiability", "project_capped_heat", "public_heat_starts"]

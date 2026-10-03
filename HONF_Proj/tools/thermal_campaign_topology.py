#!/usr/bin/env python3
"""Saved-checkpoint bounded physical paths and numerical invariance evidence.

Fixed topology preserves recorded hard active sets and recomputes continuous
physics. Invalid affine continuations terminate that branch; rebuilt paths
remain available. A switch is measured before any switch-continuity claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for source in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel/src", PROJECT_ROOT / "tools"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from thermal_campaign_benchmark import allowed_device, load_native, low_high_indices, native_arguments, output_directory
from thermal_campaign_heat_inference import native_heat_predictor, sensor_panel

from honf_forward_core.interface_fields.topology_probe import FixedTopologyInvalid, OrganizerTopologyProbe


def topology_arrays(record):
    output = {}
    if record is None:
        return output
    for phase, state in record.states.items():
        for tau, member in state.memberships.items():
            prefix = f"P{phase}/{tau}"
            admission = state.strategy_data.get("typed_admission", {}).get(tau, state.admission)
            output[f"{prefix}/candidate_source_support"] = (member > 0).cpu().numpy()
            output[f"{prefix}/admitted_source_support"] = ((member > 0) & (admission > 0)[..., None]).cpu().numpy()
            output[f"{prefix}/admission_support"] = (admission > 0).cpu().numpy()
            output[f"{prefix}/membership_density"] = member.cpu().numpy()
            output[f"{prefix}/controls"] = state.controls[tau].cpu().numpy()
            if "trees" in state.strategy_data:
                output[f"{prefix}/split_decisions"] = state.strategy_data["gates"][tau].cpu().numpy() > 0
                nodes = [[(list(node.anchor_indices), node.left, node.right, node.split_axis) for node in tree.nodes]
                         for tree in state.strategy_data["trees"][tau]]
                output[f"{prefix}/tree_node_ids"] = np.asarray(json.dumps(nodes))
    for (phase, tau, index), access in record.accesses.items():
        prefix = f"P{phase}/{tau}/read{index}"
        output[f"{prefix}/receiver_group_support"] = (access["edge_access"] > 0).cpu().numpy()
        output[f"{prefix}/effective_pair_support"] = access["support"].cpu().numpy()
    return output


def topology_changes(before, after):
    first, second = topology_arrays(before), topology_arrays(after)
    changes = {}
    for key in set(first) | set(second):
        if not key.endswith(("support", "decisions", "node_ids")):
            continue
        if key not in first or key not in second or first[key].shape != second[key].shape:
            changes[key] = "shape_or_stream_changed"
        elif first[key].dtype.kind in "US":
            if not np.array_equal(first[key], second[key]):
                changes[key] = "partition_changed"
        else:
            changed = int(np.count_nonzero(first[key] != second[key]))
            if changed:
                changes[key] = changed
    return changes


def _save_point(directory, label, parameter, prediction, record):
    arrays = {name: value.detach().cpu().numpy() for name, value in prediction.items() if torch.is_tensor(value)}
    arrays.update({f"topology/{key}": value for key, value in topology_arrays(record).items()})
    arrays["path_parameter"] = np.asarray(parameter)
    path = directory / f"{label}.npz"
    np.savez_compressed(path, **arrays)
    return str(path)


def _finite_difference_probe(forward, *, epsilon_values=(1e-2, 1e-3, 1e-4), device="cpu"):
    """Compare autograd and central differences per physical output channel."""
    parameter = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)
    keys = ("fields", "interface", "solid")
    def vector(value):
        prediction, _ = forward(value)
        return torch.cat([prediction[key].reshape(-1) for key in keys])
    value, derivative = torch.autograd.functional.jvp(vector, parameter, torch.ones_like(parameter))
    reference, topology = forward(parameter.detach())
    result = {"autograd_output_vector": value.detach().cpu().numpy(), "autograd_derivative_vector": derivative.detach().cpu().numpy()}
    rows = []
    for epsilon in epsilon_values:
        high, high_topology = forward(parameter.detach() + epsilon)
        low, low_topology = forward(parameter.detach() - epsilon)
        finite = torch.cat([(high[key] - low[key]).reshape(-1) / (2 * epsilon) for key in keys])
        result[f"finite_difference_{epsilon:g}"] = finite.detach().cpu().numpy()
        changes = topology_changes(low_topology, high_topology)
        scores, offset = {}, 0
        for key in keys:
            shape = reference[key].shape
            count = reference[key].numel()
            error = (finite[offset:offset+count] - derivative[offset:offset+count]).reshape(shape)
            if key in ("fields", "interface"):
                scores[key] = {"rmse_by_channel": error.square().mean(tuple(range(error.ndim - 1))).sqrt().detach().cpu().tolist(),
                    "max_abs_by_channel": error.abs().flatten(0, -2).amax(0).detach().cpu().tolist()}
            else:
                scores[key] = {"rmse": float(error.square().mean().sqrt()), "max_abs": float(error.abs().max())}
            offset += count
        rows.append({"epsilon": epsilon, "topology_changes_between_difference_endpoints": changes,
            "topology_available": topology is not None,
            "within_same_discrete_topology": not bool(changes) if topology is not None else None, "derivative_error": scores})
    return result, rows


def finite_difference_probe(forward, *, epsilon_values=(1e-2, 1e-3, 1e-4), device="cpu"):
    # JVP uses reverse-over-reverse AD. Flash SDPA backward has no second
    # derivative on CPU; the math implementation evaluates the same attention
    # equation and supports this numerical diagnostic. Both AD and differences
    # use this dispatch, separately from the timing/path execution policy.
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH):
        arrays, rows = _finite_difference_probe(forward, epsilon_values=epsilon_values, device=device)
    for row in rows:
        row["attention_implementation"] = "math SDPA for higher-order derivative diagnostic"
    return arrays, rows


def physical_path(model, checkpoint, sample, kind, *, amplitude=.2):
    device = next(model.parameters()).device
    heat = torch.as_tensor(sample["structure"]["heat_powers"], device=device, dtype=torch.float32)
    positions = torch.as_tensor(sample["structure"]["module_centers"], device=device, dtype=torch.float32)
    active = np.flatnonzero(sample["structure"]["module_present"] > .5)
    sensors, _, _, observed, held = sensor_panel(sample)
    direction = torch.zeros_like(heat)
    if kind == "heat":
        positive = [int(index) for index in active if float(heat[index]) > 0]
        if len(positive) < 2:
            raise ValueError("Fixed-total heat path requires two positive active allocations")
        scale = min(float(heat[positive[0]]), float(heat[positive[-1]])) * amplitude
        direction[positive[0]], direction[positive[-1]] = scale, -scale
    displacement = torch.zeros_like(positions)
    radius = float(np.asarray(sample["structure"]["material_params"])[5])
    displacement[int(active[0]), 0] = radius * amplitude
    organizer = getattr(model.core.backend, "organizer", None)

    def forward(parameter, fixed=None):
        value = heat + parameter * direction if kind == "heat" else heat
        coords = positions + parameter * displacement if kind == "geometry" else positions
        context = {"u_in": float(sample["structure"]["u_in"][0]) * (1 + amplitude * parameter)} if kind == "u_in" else None
        predictor, _ = native_heat_predictor(model, checkpoint, sample, sensors, observed, held,
            positions_override=coords, context_override=context)
        if organizer is None:
            return predictor(value), None
        with OrganizerTopologyProbe(organizer, fixed) as scope:
            prediction = predictor(value)
        return prediction, scope.record

    return forward, {"kind": kind, "amplitude": amplitude,
        "heat_direction": direction.cpu().tolist(), "position_direction": displacement.cpu().tolist() if kind == "geometry" else None,
        "query_panel": "14 fixed named physical sensors; separate material/angle receiver roles",
        "validity_limit": "fixed source IDs/masses and query-stream shape; negative affine access or lost tree connection stops fixed branch"}


def evaluate_path(model, checkpoint, sample, directory, kind, *, points=17, amplitude=.2, bisections=8):
    directory.mkdir(parents=True, exist_ok=True)
    forward, description = physical_path(model, checkpoint, sample, kind, amplitude=amplitude)
    device = next(model.parameters()).device
    anchor, reference = forward(torch.zeros((), device=device))
    _save_point(directory, "anchor", 0., anchor, reference)
    rows, rebuilt = [], {}
    fixed_invalid = {"negative": None, "positive": None}
    grid = np.linspace(-1., 1., points)
    # Move outwards from the anchor independently on either side.
    for number, value in enumerate(sorted(grid, key=lambda item: (abs(item), item))):
        with torch.no_grad():
            prediction, record = forward(torch.tensor(value, dtype=torch.float32, device=device))
        rebuilt[float(value)] = (prediction, record)
        row = {"parameter": float(value), "rebuilt": _save_point(directory, f"rebuilt_{number:03d}", value, prediction, record)}
        side = "negative" if value < 0 else "positive"
        if reference is not None and fixed_invalid[side] is None:
            try:
                with torch.no_grad():
                    fixed_prediction, fixed_record = forward(torch.tensor(value, dtype=torch.float32, device=device), reference)
                row["fixed"] = _save_point(directory, f"fixed_{number:03d}", value, fixed_prediction, fixed_record)
                row["fixed_valid"] = True
            except FixedTopologyInvalid as exc:
                fixed_invalid[side] = float(value)
                row.update(fixed_valid=False, fixed_invalid_reason=str(exc))
        elif reference is not None:
            row.update(fixed_valid=False, fixed_invalid_reason="branch terminated at earlier point")
        rows.append(row)
    switches = []
    for low, high in pairwise(grid):
        changes = topology_changes(rebuilt[float(low)][1], rebuilt[float(high)][1])
        if changes:
            switches.append({"left": float(low), "right": float(high), "changes": changes})
    brackets = []
    if switches:
        left, right = switches[0]["left"], switches[0]["right"]
        left_prediction, left_record = rebuilt[left]
        right_prediction, right_record = rebuilt[right]
        for index in range(bisections + 1):
            change = (right_prediction["fields"] - left_prediction["fields"]).detach()
            brackets.append({"left": left, "right": right, "width": right-left,
                "physical_field_max_change_by_channel": change.abs().amax(0).cpu().tolist(),
                "left_arrays": _save_point(directory, f"bracket_{index:02d}_left", left, left_prediction, left_record),
                "right_arrays": _save_point(directory, f"bracket_{index:02d}_right", right, right_prediction, right_record)})
            if index == bisections:
                break
            middle = (left + right) / 2
            with torch.no_grad():
                prediction, record = forward(torch.tensor(middle, device=device))
            if topology_changes(left_record, record):
                right, right_prediction, right_record = middle, prediction, record
            else:
                left, left_prediction, left_record = middle, prediction, record
    derivatives = {}
    for mode in ("rebuilt", "fixed"):
        if mode == "fixed" and reference is None:
            continue
        try:
            arrays, measured = finite_difference_probe(lambda parameter, mode=mode: forward(parameter, reference if mode == "fixed" else None), device=device)
            np.savez_compressed(directory / f"{mode}_finite_differences.npz", **arrays)
            derivatives[mode] = measured
        except FixedTopologyInvalid as exc:
            derivatives[mode] = {"invalid": str(exc)}
    payload = {**description, "topology_available": reference is not None,
        "measured_switches": switches, "switch_found": bool(switches), "switch_brackets": brackets,
        "continuity_claim": "No mathematical cross-switch continuity claim; saved shrinking brackets and physical channel changes are evidence to assess.",
        "fixed_branch_first_invalid": fixed_invalid, "points": sorted(rows, key=lambda row: row["parameter"]),
        "finite_differences": derivatives}
    (directory / "path.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def numerical_check(before, after, *, rtol=2e-5, atol=2e-6):
    delta = (after - before).detach()
    return {"finite": bool(torch.isfinite(after).all()), "max_abs": float(delta.abs().max()) if delta.numel() else 0.,
        "rmse": float(delta.square().mean().sqrt()) if delta.numel() else 0.,
        "passed": bool(torch.allclose(before, after, rtol=rtol, atol=atol)), "rtol": rtol, "atol": atol}


def fine_core_invariance(backend, encoded, states, queries, features):
    """Source-resolved fine operator; hold the receiver anchor universe fixed.

    The atom split duplicates an identical physical state, coordinate, length
    and group density, halves its measure, and retains collective controls.
    This read identity is separate from full organizer rebuild sensitivity.
    """
    results, arrays = {}, {}
    prepared = backend.prepare(encoded, states)
    original, _ = backend.read(prepared, encoded, queries, features)
    arrays["original"] = original.detach().cpu().numpy()
    count = states.shape[1]
    order = torch.arange(count - 1, -1, -1, device=states.device)
    values = {name: getattr(encoded, name)[:, order] for name in ("module_tokens", "module_centers", "module_present", "module_features")}
    for name in ("module_source_ids", "module_characteristic_lengths"):
        value = getattr(encoded, name)
        if value is not None:
            values[name] = value[:, order]
    permuted = replace(encoded, **values)
    actual, _ = backend.read(backend.prepare(permuted, states[:, order]), permuted, queries, features)
    results["module_permutation"] = numerical_check(original, actual)
    arrays["module_permutation"] = actual.detach().cpu().numpy()
    padded = {}
    for name in ("module_tokens", "module_centers", "module_present", "module_features"):
        value = getattr(encoded, name)
        padded[name] = torch.cat((value, torch.zeros_like(value[:, :1])), 1)
    for name in ("module_source_ids", "module_characteristic_lengths"):
        value = getattr(encoded, name)
        if value is not None:
            extra = value.amax(1, keepdim=True) + 1 if name.endswith("ids") else torch.ones_like(value[:, :1])
            padded[name] = torch.cat((value, extra), 1)
    record = replace(encoded, **padded)
    padded_states = torch.cat((states, torch.zeros_like(states[:, :1])), 1)
    actual, _ = backend.read(backend.prepare(record, padded_states), record, queries, features)
    results["inactive_module_padding"] = numerical_check(original, actual)
    arrays["inactive_module_padding"] = actual.detach().cpu().numpy()
    if hasattr(backend, "organizer") and encoded.env_weights.shape[1]:
        from honf_forward_core.interface_fields.typed_hypergraph_state import SOURCE_TYPE
        weights = encoded.env_weights.clone()
        weights[:, 0] *= .5
        values = {"env_weights": torch.cat((weights, weights[:, :1]), 1)}
        for name in ("env_tokens", "env_coords", "env_features", "env_region_ids", "env_source_ids", "env_characteristic_lengths"):
            value = getattr(encoded, name)
            if value is not None:
                values[name] = torch.cat((value, value[:, :1]), 1)
        refined = replace(encoded, **values)
        plan = prepared["hypergraph_plan"]
        member = {tau: torch.cat((value, value[..., :1]), -1) if SOURCE_TYPE[tau] == "E" else value
                  for tau, value in plan.memberships.items()}
        metadata = {}
        for field in ("source_coords", "source_measures", "source_valid", "source_ids", "source_lengths"):
            catalogue = getattr(plan, field)
            if catalogue is None or "E" not in catalogue:
                continue
            old = catalogue["E"]
            env = values["env_weights"] if field == "source_measures" else torch.cat((old, old[:, :1]), 1)
            metadata[field] = {**catalogue, "E": env}
        fixed_plan = replace(plan, memberships=member, **metadata)
        fixed = {**prepared, "hypergraph_plan": fixed_plan,
            "env_tokens": torch.cat((prepared["env_tokens"], prepared["env_tokens"][:, :1]), 1),
            "projected_key": torch.cat((prepared["projected_key"], prepared["projected_key"][:, :, :1]), -2),
            "projected_value": torch.cat((prepared["projected_value"], prepared["projected_value"][:, :, :1]), -2)}
        # The old preparation ledgers have the original source/receiver axes.
        # Rebind their access records to the refined physical quadrature for
        # structural accounting; contextual physical source states stay fixed.
        present = refined.module_present > .5
        fixed["hypergraph_accesses"] = {**prepared["hypergraph_accesses"],
            "ME": backend._access(fixed_plan, refined.module_centers, "ME", states, present[:, :, None]),
            "EM": backend._access(fixed_plan, refined.env_coords, "EM", refined.env_tokens, present[:, None])}
        actual, _ = backend.read(fixed, refined, queries, features)
        results["quadrature_atom_split_fixed_prepared_action"] = numerical_check(original, actual)
        arrays["quadrature_atom_split_fixed_prepared_action"] = actual.detach().cpu().numpy()
        rebuilt, _ = backend.read(backend.prepare(refined, states), refined, queries, features)
        results["quadrature_atom_split_with_organizer_rebuild"] = numerical_check(original, rebuilt)
        arrays["quadrature_atom_split_with_organizer_rebuild"] = rebuilt.detach().cpu().numpy()
    return results, arrays


def evaluate_invariance(model, sample, raw_sample, device, directory):
    sensors, _, _, _, _ = sensor_panel(raw_sample)
    with torch.no_grad():
        arguments = native_arguments(model, sample, sensors, device)
        result = model(**arguments)
        prepared = result["prepared_state"]
        queries = arguments["query_xy"]
        baseline = model.decode_prepared(prepared, queries)["pred_field"]
        reverse = torch.arange(queries.shape[1] - 1, -1, -1, device=device)
        ordered = model.decode_prepared(prepared, queries[:, reverse])["pred_field"][:, reverse]
        chunks = torch.cat([model.decode_prepared(prepared, queries[:, start:start+3])["pred_field"]
                            for start in range(0, queries.shape[1], 3)], 1)
        shared = prepared.prepared
        features = model.core._receiver_features(shared, queries)
        checks, arrays = fine_core_invariance(model.core.backend, shared.encoded, shared.module_states, queries, features)
        checks.update(native_prepared_query_order=numerical_check(baseline, ordered),
                      native_prepared_query_chunking=numerical_check(baseline, chunks))
        arrays.update(native_field=baseline.cpu().numpy(), query_reordered=ordered.cpu().numpy(), query_chunked=chunks.cpu().numpy())
    np.savez_compressed(directory / "invariance.npz", **arrays)
    payload = {"fine_scope": "P2 source-preserving fine context after native local coupling; module tests rebuild only shared fine organizer/kernel",
        "query_scope": "full native prepared field decoding; checkpoint output normalization",
        "quadrature_definition": "identical atom state/position/length/group density, half measure each, fixed anchor universe; full organizer rebuild sensitivity reported separately",
        "checks": checks}
    (directory / "invariance.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def evaluate(args):
    device = allowed_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    model, checkpoint, normalized, raw, dataset_path = load_native(args.checkpoint, args.dataset, device)
    output = output_directory(args.output_dir)
    rows = []
    for index in low_high_indices(raw, count=args.cases):
        sample, reference = normalized[index], raw[index]
        directory = output / str(reference["case_id"])
        directory.mkdir(parents=True, exist_ok=True)
        row = {"case_id": str(reference["case_id"]), "module_count": int((reference["structure"]["module_present"] > .5).sum()),
            "invariance": evaluate_invariance(model, sample, reference, device, directory), "paths": {}}
        for kind in args.paths:
            try:
                row["paths"][kind] = evaluate_path(model, checkpoint, reference, directory / kind, kind,
                    points=args.points, amplitude=args.amplitude, bisections=args.bisections)
            except ValueError as exc:
                row["paths"][kind] = {"unavailable": str(exc)}
        rows.append(row)
        payload = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": checkpoint.get("epoch"),
                "architecture": model.config.core_honf.forward_architecture, "dataset": str(dataset_path), "partition": "exposed development",
            "field_channels": list(model.config.channelthermal.field_names),
            "interface_channels": ["surface_temperature", "q_normal_proxy"],
            "device": str(device), "reference_limit": "frozen surrogate numerical/gradient checks; no new physical solve",
            "rows": rows}
        (output / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        print(f"{reference['case_id']}: saved invariance and bounded path evidence", flush=True)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--paths", nargs="+", choices=("heat", "geometry", "u_in"), default=["heat", "geometry"])
    parser.add_argument("--points", type=int, default=17)
    parser.add_argument("--amplitude", type=float, default=.2)
    parser.add_argument("--bisections", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.cases, args.cpu_threads) < 1 or args.points < 3 or args.points % 2 != 1 or not 0 < args.amplitude < 1 or args.bisections < 0:
        parser.error("positive sizes, odd path points >=3, amplitude in (0,1) and nonnegative bisections required")
    return args


if __name__ == "__main__":
    print(evaluate(parse_args()))

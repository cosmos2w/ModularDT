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
from contextlib import AbstractContextManager
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
from honf_forward_core.interface_fields.typed_hypergraph_state import SOURCE_TYPE, source_moments, structural_cost


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
    discrete = not before.is_floating_point() and not before.is_complex()
    if discrete:
        before, after = before.to(torch.float64), after.to(torch.float64)
    delta = (after - before).detach()
    finite = bool(torch.isfinite(before).all() & torch.isfinite(after).all())
    margin = delta.abs() / (atol + rtol * before.detach().abs())
    return {"finite": finite, "max_abs": (float(delta.abs().max()) if delta.numel() else 0.) if finite else None,
        "rmse": (float(delta.square().mean().sqrt()) if delta.numel() else 0.) if finite else None,
        "max_tolerance_ratio": (float(margin.max()) if margin.numel() else 0.) if finite else None,
        "passed": bool(torch.equal(before, after) if discrete else torch.allclose(before, after, rtol=rtol, atol=atol)),
        "comparison": "exact discrete metadata" if discrete else "floating allclose", "rtol": rtol, "atol": atol}


class RefinedEnvironmentBuilder:
    """Equivalent atom catalogue for an entire native wrapper rebuild.

    The original builder owns physical geometry/features/lengths. Explicit
    child masses prevent the core's historical area/E fallback from changing
    the measure. Fine atoms and every P0/P1/P2 preparation are rebuilt.
    """

    def __init__(self, original, variant, *, gradients=False):
        self.original, self.variant, self.gradients = original, variant, gradients
        self.base = None

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, **kwargs):
        environment = self.original(**kwargs)
        if environment.env_weights is None or environment.env_characteristic_lengths is None:
            raise ValueError("Whole-rebuild equivalence requires adapter-owned masses and physical lengths")
        if self.gradients:
            values = {name: getattr(environment, name).detach().clone().requires_grad_()
                      for name in ("env_coords", "env_features", "env_weights", "env_characteristic_lengths")}
            environment = replace(environment, **values)
        self.base = environment
        count, device = environment.env_weights.shape[-1], environment.env_weights.device
        split = {}
        if self.variant == "split_equal":
            split[0] = (.5, .5)
        elif self.variant == "split_unequal":
            split[0] = (.3, .7)
        elif self.variant == "split_multiple":
            split = {atom: (.3, .7) for atom in sorted({0, count // 2, count - 1})}
        elif self.variant not in ("original", "permutation"):
            raise ValueError(f"Unknown atom-equivalence variant {self.variant!r}")
        parents, fractions = [], []
        for atom in range(count):
            children = split.get(atom, (1.,))
            parents.extend([atom] * len(children))
            fractions.extend(children)
        if self.variant == "permutation":
            parents.reverse()
        self.parents = torch.tensor(parents, device=device)
        self.fractions = environment.env_weights.new_tensor(fractions)
        values = {name: (None if getattr(environment, name) is None else getattr(environment, name)[:, self.parents])
                  for name in ("env_coords", "env_features", "env_region_ids", "env_characteristic_lengths")}
        values["env_weights"] = environment.env_weights[:, self.parents] * self.fractions
        self.refined = replace(environment, **values)
        return self.refined


class _NativeRebuildCapture(AbstractContextManager):
    """Local live phase/access/decode capture; restores every monkeypatch."""

    def __init__(self, model, *, fixed=None, annotate_measure=True):
        self.model, self.fixed = model, fixed
        self.annotate_measure = annotate_measure
        self.states, self.accesses, self.decodes, self.counts = {}, {}, {}, {}
        self.input_states = {}
        self.explicit_measure_annotation = False
        self.encoded_environment_measures = None

    def __enter__(self):
        self.organizer = getattr(self.model.core.backend, "organizer", None)
        self.originals, self.owned = {}, {}
        methods = [(self.model.core, "decode_queries")]
        if hasattr(self.model.core, "encode_case"):
            methods.append((self.model.core, "encode_case"))
        if self.organizer is not None:
            methods += [(self.organizer, "prepare"), (self.organizer, "access")]
        for owner, name in methods:
            self.originals[name] = getattr(owner, name)
            self.owned[name] = name in owner.__dict__
        self.methods = methods

        def prepare(*args, **kwargs):
            phase = int(str(kwargs.get("phase", 0)).removeprefix("P"))
            kwargs["capture_topology"] = True
            if self.fixed is not None:
                kwargs["fixed_topology"] = self.fixed.states[phase]
            state = self.originals["prepare"](*args, **kwargs)
            if phase in self.states:
                raise ValueError("Whole-wrapper invariance expects one organizer preparation per physical phase")
            self.states[phase] = state
            self.input_states[phase] = (args[0], args[1])
            return state

        def access(state, receivers, mechanism, *args, **kwargs):
            tau = str(mechanism).upper()
            route = state.phase, tau
            index = self.counts.get(route, 0)
            self.counts[route] = index + 1
            key = *route, index
            kwargs["capture_topology"] = True
            if self.fixed is not None:
                recorded = self.fixed.accesses[key]
                kwargs["fixed_receiver_access"] = {"edge_access": recorded.edge_access.detach()}
            value = self.originals["access"](state, receivers, mechanism, *args, **kwargs)
            self.accesses[key] = value
            return value

        def decode(prepared, *args, **kwargs):
            value = self.originals["decode_queries"](prepared, *args, **kwargs)
            context = kwargs.get("interaction_context")
            phase = getattr(context, "phase", None) or "unspecified"
            role = getattr(context, "receiver_role", None) or "unspecified"
            key = f"{phase}/{role}/read{len(self.decodes)}"
            self.decodes[key] = value["pred_field"]
            return value

        def encode(batch, *args, **kwargs):
            builder = self.model.environment_builder
            if self.annotate_measure and batch.env_weights is None and isinstance(builder, RefinedEnvironmentBuilder):
                # Historical Thermal paths ignore the builder's optional
                # mass metadata. Bind it explicitly ONLY for this physical
                # equivalence diagnostic; otherwise area/E changes on split.
                batch = replace(batch, env_weights=builder.refined.env_weights)
                self.explicit_measure_annotation = True
            value = self.originals["encode_case"](batch, *args, **kwargs)
            self.encoded_environment_measures = value.env_weights
            return value

        self.model.core.decode_queries = decode
        if "encode_case" in self.originals:
            self.model.core.encode_case = encode
        if self.organizer is not None:
            self.organizer.prepare, self.organizer.access = prepare, access
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for owner, name in self.methods:
            if self.owned[name]:
                setattr(owner, name, self.originals[name])
            else:
                delattr(owner, name)
        return False


def _pullback_atom_axis(value, parents, fractions, axis, count):
    """Mass-fraction average children onto their original physical parent."""
    moved = value.movedim(axis, 0)
    scale = fractions.reshape((-1,) + (1,) * (moved.ndim - 1))
    reduced = value.new_zeros((count, *moved.shape[1:])).index_add(0, parents, moved * scale)
    return reduced.movedim(0, axis)


def _wrapper_physical_outputs(result, capture, checkpoint):
    """Use checkpoint-owned output transform; ports retain declared features."""
    from channelthermal.response_control.native import _physical_output
    settings = checkpoint.get("train_config", {}).get("dataset", {}) if checkpoint else {}
    stats = checkpoint.get("global_normalization_stats", {}) if checkpoint else {}
    normalized = bool(settings.get("normalize_targets", False))
    mapping = {"pred_field": ("field_mean_by_channel", "field_std_by_channel"),
               "pred_internal_temperature": ("internal_temperature_mean", "internal_temperature_std"),
               "pred_interface": (("interface_targets_mean", "interface_targets_std") if "interface_targets_mean" in stats
                                  else ("interface_target_mean", "interface_target_std"))}
    output = {}
    for name, keys in mapping.items():
        if torch.is_tensor(result.get(name)):
            output[name] = _physical_output(result[name], stats, *keys, normalize_targets=normalized)
    for name in ("pred_port_condition", "pred_port_condition_raw", "local_port_condition_used"):
        if torch.is_tensor(result.get(name)):
            output[name] = result[name]
    for key, value in capture.decodes.items():
        output[f"phase_decode/{key}"] = _physical_output(value, stats, *mapping["pred_field"], normalize_targets=normalized)
    return output


def _rebuild_representation_arrays(capture, builder):
    tensors = {}
    count = builder.base.env_weights.shape[-1]
    parents, fractions = builder.parents, builder.fractions
    for phase, state in capture.states.items():
        for kind, coordinates in state.source_coords.items():
            tensors[f"P{phase}/source_coords_{kind}"] = (
                _pullback_atom_axis(coordinates, parents, fractions, 1, count) if kind == "E" else coordinates)
        for tau, control in state.controls.items():
            tensors[f"P{phase}/{tau}/group_controls"] = control
            member = state.memberships[tau]
            tensors[f"P{phase}/{tau}/value_donors"] = (
                _pullback_atom_axis(member, parents, fractions, -1, count) if SOURCE_TYPE[tau] == "E" else member)
            for name in ("typed_centres", "typed_admission"):
                if tau in state.strategy_data.get(name, {}):
                    tensors[f"P{phase}/{tau}/{name}"] = state.strategy_data[name][tau]
            donors = getattr(state, "control_memberships", {})
            for kind, member in donors.get(tau, {}).items():
                tensors[f"P{phase}/{tau}/control_donors_{kind}"] = (
                    _pullback_atom_axis(member, parents, fractions, -1, count) if kind == "E" else member)
            trees = state.strategy_data.get("trees", {}).get(tau)
            if trees:
                for case, tree in enumerate(trees):
                    geometry = state.strategy_data["access_geometry"]["M" if tau in ("MM", "ME") else "E" if tau == "EM" else "Q"]
                    tensors[f"P{phase}/{tau}/case{case}/boundaries"] = geometry["boundary"][case]
                    for name in ("axes", "parents", "left", "depths", "leaves", "valid", "overlap"):
                        tensors[f"P{phase}/{tau}/case{case}/geometry_{name}"] = geometry[name][case]
                    points = [tree.universe.coordinates[list(node.anchor_indices)] for node in tree.nodes]
                    tensors[f"P{phase}/{tau}/case{case}/node_bbox_min"] = torch.stack([point.amin(0) for point in points])
                    tensors[f"P{phase}/{tau}/case{case}/node_bbox_max"] = torch.stack([point.amax(0) for point in points])
                    if tree.canonical_index is not None:
                        tensors[f"P{phase}/{tau}/case{case}/index_coords"] = tree.canonical_index.coordinates
                        tensors[f"P{phase}/{tau}/case{case}/index_mass"] = tree.canonical_index.weights
                        tensors[f"P{phase}/{tau}/case{case}/index_roles"] = tree.canonical_index.roles.to(control.dtype)
                        block_count = tree.canonical_index.coordinates.shape[0]
                        support = control.new_zeros((len(tree.nodes), block_count))
                        for node, candidate in enumerate(tree.nodes):
                            support[node, tree.canonical_index.atom_to_block[list(candidate.anchor_indices)]] = 1
                        tensors[f"P{phase}/{tau}/case{case}/canonical_node_support"] = support
        first_accesses = {}
        for (access_phase, tau, index), access in capture.accesses.items():
            if access_phase != phase:
                continue
            if not hasattr(access, "control"):
                # Actual native reads retain only affine projected actions.
                # Reconstruct full vectors solely for this representation
                # diagnostic; physical execution and its input VJP stay on
                # the projected path captured above.
                kind = SOURCE_TYPE[tau]
                reconstructed = source_moments(access.edge_access, state.memberships[tau], state.controls[tau],
                    state.source_measures[kind], state.source_valid[kind],
                    pair_valid=access.diagnostics["pair_valid"], near=access.near)
                access = replace(reconstructed, diagnostics=access.diagnostics)
            first_accesses.setdefault(tau, access)
            for name, value in (("density", access.density), ("control_moment", access.density[..., None] * access.control)):
                if SOURCE_TYPE[tau] == "E":
                    value = _pullback_atom_axis(value, parents, fractions, 2, count)
                if tau == "EM":
                    value = _pullback_atom_axis(value, parents, fractions, 1, count)
                tensors[f"P{phase}/{tau}/read{index}/{name}"] = value
        cost, metrics = structural_cost(first_accesses, state)
        tensors[f"P{phase}/structural_cost"] = cost
        for name, value in metrics.items():
            if "smooth" in name:
                tensors[f"P{phase}/structural/{name}"] = value
    tensors["input/total_environment_mass"] = builder.base.env_weights[:, builder.parents].mul(builder.fractions).sum(-1)
    if capture.encoded_environment_measures is not None:
        tensors["input/encoded_parent_environment_mass"] = _pullback_atom_axis(
            capture.encoded_environment_measures / builder.fractions[None], builder.parents,
            builder.fractions, 1, count)
    return tensors


def whole_wrapper_invariance(model, sample, raw_sample, *, checkpoint=None, directory=None,
                             detailed=False, gradients=False):
    """Measure complete native Thermal source equivalence on supplied cases.

    Memberships/encoder/ports/frozen Stage-A/P0/P1/P2 all rebuild each time.
    The caller selects only protocol-authorized cases; this function never
    reads or enlarges dataset membership. Detailed mode adds equal and
    multiple splits; light all22 mode uses permutation and a 30/70 split.
    """
    if model.training:
        raise ValueError("Whole-wrapper invariance is evaluation-only")
    if gradients and not detailed:
        raise ValueError("First-gradient checks are limited to the detailed representative panel")
    device = next(model.parameters()).device
    sensors, _, _, _, _ = sensor_panel(raw_sample)
    arguments = native_arguments(model, sample, sensors, device)
    original_builder = model.environment_builder
    native_wrapper_calls = 0

    def evaluate_variant(variant, *, differentiate=False, reference=None, annotate_measure=True):
        nonlocal native_wrapper_calls
        builder = RefinedEnvironmentBuilder(original_builder, variant, gradients=differentiate)
        model.environment_builder = builder
        try:
            with torch.set_grad_enabled(differentiate), _NativeRebuildCapture(model, fixed=reference,
                annotate_measure=annotate_measure) as capture:
                result = model(**arguments)
                native_wrapper_calls += 1
                physical = _wrapper_physical_outputs(result, capture, checkpoint)
                representation = _rebuild_representation_arrays(capture, builder)
                derivative = {}
                if differentiate:
                    # A deterministic linear physical-output probe. The base
                    # catalogue stays original-shaped before the refinement,
                    # so autograd performs the correct child pullback.
                    objective = sum(value.mean() for value in physical.values() if value.numel())
                    names = ("env_coords", "env_features", "env_weights", "env_characteristic_lengths")
                    inputs = tuple(getattr(builder.base, name) for name in names)
                    values = torch.autograd.grad(objective, inputs, allow_unused=True)
                    capture.gradient_unused = [name for name, value in zip(names, values) if value is None]
                    derivative = {name: (torch.zeros_like(source) if value is None else value)
                                  for name, source, value in zip(names, inputs, values)}
            return physical, representation, derivative, capture, builder
        finally:
            model.environment_builder = original_builder

    original, baseline_representation, _, baseline_capture, base_builder = evaluate_variant("original")
    arrays = {f"original/physical/{key}": value.detach().cpu().numpy() for key, value in original.items()}
    arrays.update({f"original/representation/{key}": value.detach().cpu().numpy() for key, value in baseline_representation.items()})
    annotation_check = None
    if baseline_capture.explicit_measure_annotation:
        ordinary, _, _, _, _ = evaluate_variant("original", annotate_measure=False)
        annotation_check = {key: numerical_check(value, ordinary[key], atol=2e-5) for key, value in original.items()}
        arrays.update({f"ordinary_unannotated/physical/{key}": value.detach().cpu().numpy() for key, value in ordinary.items()})
    variants = ["permutation", "split_unequal"]
    if detailed:
        variants = ["permutation", "split_equal", "split_unequal", "split_multiple"]
    baseline_grad = None
    if gradients:
        _, _, baseline_grad, _, _ = evaluate_variant("original", differentiate=True, reference=baseline_capture)
    checks = {}
    for variant in variants:
        physical, representation, _, capture, builder = evaluate_variant(variant)
        measured = {"physical": {key: numerical_check(value, physical[key], atol=2e-5) for key, value in original.items()},
                    "physical_fine_core_threshold": {key: numerical_check(value, physical[key]) for key, value in original.items()},
                    "representation": {key: numerical_check(value, representation[key])
                        if key in representation and value.shape == representation[key].shape else {"passed": False, "shape_changed": True}
                        for key, value in baseline_representation.items()},
                    "parent_map": builder.parents.cpu().tolist(), "child_mass_fractions": builder.fractions.cpu().tolist()}
        if gradients:
            try:
                _, _, actual_grad, _, _ = evaluate_variant(variant, differentiate=True, reference=capture)
                measured["fixed_topology_first_gradients"] = {name: numerical_check(value, actual_grad[name], atol=1e-6)
                    for name, value in baseline_grad.items()}
                arrays.update({f"{variant}/gradient/{name}": value.detach().cpu().numpy() for name, value in actual_grad.items()})
            except FixedTopologyInvalid as error:
                measured["fixed_topology_first_gradients"] = {"invalid": str(error), "passed": False}
        arrays.update({f"{variant}/physical/{key}": value.detach().cpu().numpy() for key, value in physical.items()})
        arrays.update({f"{variant}/representation/{key}": value.detach().cpu().numpy() for key, value in representation.items()})
        checks[variant] = measured
    if baseline_grad is not None:
        arrays.update({f"original/gradient/{name}": value.detach().cpu().numpy() for name, value in baseline_grad.items()})
    arrays["input/environment_coords"] = base_builder.base.env_coords.detach().cpu().numpy()
    arrays["input/environment_mass"] = base_builder.base.env_weights.detach().cpu().numpy()
    arrays["input/environment_lengths"] = base_builder.base.env_characteristic_lengths.detach().cpu().numpy()
    payload = {"case_id": str(raw_sample["case_id"]), "checks": checks,
        "scope": "complete native encoder, organizer, autonomous ports, frozen Stage-A and all P0/P1/P2 rebuilds",
        "historical_explicit_measure_annotation": baseline_capture.explicit_measure_annotation,
        "historical_adapter_limit": "When annotation is required, the historical ordinary adapter does not natively transport refined weights; without this diagnostic binding, its uniform area/E fallback changes quadrature on split",
        "annotation_baseline_vs_ordinary_native": annotation_check,
        "native_wrapper_calls": native_wrapper_calls,
        "phases_measured": sorted(baseline_capture.states), "queries": len(sensors),
        "query_selection": "fixed input-only named physical sensor panel; query batches do not define the tree",
        "structural_scope": "case-balanced physical measures; first actual access per typed route in each phase",
        "physical_output_units": ("checkpoint-owned native benchmark transform" if checkpoint is not None else "model output units; no checkpoint transform supplied"),
        "tolerance_policy": "Native physical outputs use maintained complete-wrapper atol2e-5/rtol2e-5; inherited fine-core atol2e-6 is also reported separately without concealing near-zero roundoff failures",
        "gradient_scope": "separately frozen native active topology for each representation; deterministic linear physical-output probe with child coordinate/feature gradients summed and mass gradients fraction-weighted",
        "reference_limit": "same-checkpoint numerical equivalence; no physical response solve or candidate design reference"}
    if directory is not None:
        directory = output_directory(directory)
        np.savez_compressed(directory / "whole_wrapper_invariance.npz", **arrays)
        (directory / "whole_wrapper_invariance.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload, arrays


def trained_control_locality(model, sample, raw_sample, *, directory=None):
    """Conditional P0 content derivatives at actual trained control weights.

    Donor weights and receiver geometry stay fixed for attribution. A
    separately rebuilt planner is tested independently. Any forced exclusion
    is explicitly a manipulated diagnostic, never learned sparse support.
    """
    if model.training:
        raise ValueError("Conditional trained-control attribution requires evaluation mode")
    organizer = getattr(model.core.backend, "organizer", None)
    if organizer is None or not getattr(organizer, "faithful_controls", False):
        return {"available": False, "reason": "Checkpoint has no membership-local collective controls"}, {}
    device = next(model.parameters()).device
    sensors, _, _, _, _ = sensor_panel(raw_sample)
    with torch.no_grad(), _NativeRebuildCapture(model) as capture:
        model(**native_arguments(model, sample, sensors, device))
    state = capture.states[0]
    encoded, modules = capture.input_states[0]
    tau = "QM"
    admission = state.strategy_data["typed_admission"][tau]
    groups = torch.nonzero(admission[0] > 0, as_tuple=False).flatten().tolist()
    selected = None
    for group in groups:
        for kind in ("M", "E"):
            member = state.control_memberships[tau][kind][0, group]
            omitted = torch.nonzero(state.source_valid[kind][0] & (member == 0), as_tuple=False).flatten()
            admitted = torch.nonzero(state.source_valid[kind][0] & (member > 0), as_tuple=False).flatten()
            if omitted.numel() and admitted.numel():
                selected = group, kind, int(omitted[0]), int(admitted[0]), False
                break
        if selected is not None:
            break
    if selected is None:
        group = groups[0]
        kind = "M" if int(state.source_valid["M"][0].sum()) >= 2 else "E"
        sources = torch.nonzero(state.source_valid[kind][0], as_tuple=False).flatten()
        selected = group, kind, int(sources[0]), int(sources[1]), True
    group, kind, excluded, admitted, manipulated = selected
    members = {route: {donor: value.detach().clone() for donor, value in donors.items()}
               for route, donors in state.control_memberships.items()}
    if manipulated:
        members[tau][kind][0, group, excluded] = 0
    fixed = replace(state, control_memberships={route: {donor: value.detach() for donor, value in donors.items()}
                                              for route, donors in state.control_memberships.items()})
    m = modules.detach().clone().requires_grad_()
    e = encoded.env_tokens.detach().clone().requires_grad_()

    def vector(module_content, environment_content):
        live = replace(encoded, env_tokens=environment_content)
        return organizer.recompute_controls(fixed, live, module_content)[tau][0, group]

    jac_m, jac_e = torch.autograd.functional.jacobian(vector, (m, e), vectorize=False)
    jacobian = jac_m if kind == "M" else jac_e
    per_source_norm = jacobian[:, 0].square().sum((0, 2)).sqrt()
    actual_inventory = {}
    for donor, jac in (("M", jac_m), ("E", jac_e)):
        member = state.control_memberships[tau][donor][0, group]
        norm = jac[:, 0].square().sum((0, 2)).sqrt()
        valid = state.source_valid[donor][0]
        def donor_rows(mask, donor=donor, member=member, norm=norm):
            return [{"source_slot": int(slot), "source_id": int(state.source_ids[donor][0, slot]),
                     "edge_id": f"P0/{tau}/g{group}/{donor}:{int(state.source_ids[donor][0, slot])}",
                     "coordinates": state.source_coords[donor][0, slot].cpu().tolist(),
                     "control_membership": float(member[slot]), "control_jacobian_norm": float(norm[slot])}
                    for slot in torch.nonzero(mask, as_tuple=False).flatten().tolist()]
        actual_inventory[donor] = {"tested_valid_source_count": int(valid.sum()),
            "tested_excluded_source_count": int((valid & (member == 0)).sum()),
            "excluded_sources": donor_rows(valid & (member == 0)),
            "positive_control_donors_with_nonzero_jacobian": donor_rows(valid & (member > 0) & (norm > 0))}
    actual_exclusion_available = any(value["tested_excluded_source_count"] for value in actual_inventory.values())
    manipulated_result = None
    if manipulated:
        fixed = replace(state, control_memberships=members)
        diagnostic_m, diagnostic_e = torch.autograd.functional.jacobian(vector, (m, e), vectorize=False)
        diagnostic_jac = diagnostic_m if kind == "M" else diagnostic_e
        diagnostic_norm = diagnostic_jac[:, 0].square().sum((0, 2)).sqrt()
        manipulated_result = {"origin": "explicitly manipulated conditional diagnostic; not trained exclusion evidence",
            "donor_type": kind, "excluded_source_slot": excluded, "admitted_source_slot": admitted,
            "excluded_control_derivative_norm": float(diagnostic_norm[excluded]),
            "excluded_control_derivative_zero": bool(diagnostic_norm[excluded] == 0),
            "admitted_control_derivative_norm": float(diagnostic_norm[admitted])}
    # Rebuild the richer planner using the same actual content. For an M
    # donor, read a DIFFERENT target source logit: the derivative can then
    # arrive through planning summaries, not that donor's direct token slot.
    refreshed = organizer.prepare(replace(encoded, env_tokens=e), m, phase=0, capture_topology=True)
    target_kind = kind
    planning_scalar = refreshed.strategy_data["all_source_logits"][tau][target_kind][0][group, admitted]
    source = m if kind == "M" else e
    planning_grad = torch.autograd.grad(planning_scalar, source, allow_unused=True)[0]
    planning_norm = 0. if planning_grad is None else float(planning_grad[0, excluded].norm())
    arrays = {"source_coords_M": state.source_coords["M"].cpu().numpy(),
              "source_coords_E": state.source_coords["E"].cpu().numpy(),
              "conditional_control_jacobian_M": jac_m.detach().cpu().numpy(),
              "conditional_control_jacobian_E": jac_e.detach().cpu().numpy(),
              "conditional_source_derivative_norm": per_source_norm.detach().cpu().numpy(),
              "actual_group_admission": admission.cpu().numpy()}
    inventory = {}
    for phase, phase_state in capture.states.items():
        inventory[f"P{phase}"] = {"dependency_provenance": phase_state.dependency_provenance, "routes": {}}
        for route, donors in phase_state.control_memberships.items():
            active = phase_state.strategy_data["typed_admission"][route] > 0
            inventory[f"P{phase}"]["routes"][route] = {}
            for donor, member in donors.items():
                valid = phase_state.source_valid[donor][:, None] & active[..., None]
                inventory[f"P{phase}"]["routes"][route][donor] = {
                    "admitted_control_donor_entries": int((valid & (member > 0)).sum()),
                    "actually_excluded_control_donor_entries": int((valid & (member == 0)).sum())}
                arrays[f"P{phase}/{route}/control_membership_{donor}"] = member.cpu().numpy()
    payload = {"available": True, "case_id": str(raw_sample["case_id"]), "phase": "P0", "mechanism": tau,
        "group": group, "donor_type": kind, "excluded_donor_slot": excluded if not manipulated else None, "admitted_donor_slot": admitted,
        "exclusion_origin": "explicitly manipulated conditional diagnostic" if manipulated else "actual learned zero control membership",
        "actual_conditional_exclusion_test_available": actual_exclusion_available,
        "actual_conditional_exclusion_test_status": "measured actual excluded donors" if actual_exclusion_available else "vacuous/unavailable: no valid source excluded from the tested M or E control donor sets",
        "actual_tested_control_donors": actual_inventory,
        "manipulated_conditional_diagnostic": manipulated_result,
        "excluded_control_derivative_norm": float(per_source_norm[excluded]) if not manipulated else None,
        "admitted_control_derivative_norm": float(per_source_norm[admitted]),
        "excluded_control_derivative_zero": bool(per_source_norm[excluded] == 0) if not manipulated else None,
        "admitted_control_derivative_nonzero": bool(per_source_norm[admitted] > 0),
        "separate_rebuilt_planner_derivative_norm": planning_norm,
        "separate_rebuilt_planner_derivative_nonzero": planning_norm > 0,
        "planner_probe": "different admitted source logit at the same group, with the tested donor's input content live; hypothetical exclusion is separate when no actual zero membership exists",
        "planner_probe_source": {"donor_type": kind, "source_slot": excluded,
            "source_id": int(state.source_ids[kind][0, excluded]),
            "actual_control_membership": float(state.control_memberships[tau][kind][0, group, excluded])},
        "ancestry_and_actual_donor_inventory": inventory,
        "charged_work": {"native_wrapper_calls": 1, "separate_organizer_rebuilds": 1,
                         "control_vector_vjps": organizer.control_dim * (2 if manipulated else 1)},
        "limits": "Conditional token-content derivatives are computational attribution; P1/P2 states carry disclosed upstream ancestry; no physical-causality claim"}
    if directory is not None:
        directory = output_directory(directory)
        np.savez_compressed(directory / "trained_control_locality.npz", **arrays)
        (directory / "trained_control_locality.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload, arrays


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
    if getattr(args, "case_ids", None) is not None:
        requested = [str(value).zfill(4) for value in args.case_ids]
        if len(set(requested)) != len(requested) or any(value not in raw.selected_case_ids for value in requested):
            raise ValueError("Explicit topology cases must be unique members of the bound evaluation partition")
        indices = [raw.selected_case_ids.index(value) for value in requested]
    else:
        indices = low_high_indices(raw, count=args.cases)
    for index in indices:
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
    parser.add_argument("--case-ids", nargs="+", help="Explicit cases within the checkpoint-bound evaluation partition")
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

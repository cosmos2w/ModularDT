"""Transparent evaluation-only recording of actual typed native access calls."""

import json
from contextlib import AbstractContextManager

import numpy as np
import torch


def _flatten(prefix, value, arrays):
    if torch.is_tensor(value):
        arrays[prefix] = value.detach().cpu().numpy().copy()
    elif isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}/{key}", item, arrays)


def control_summary(control):
    """Small deterministic probes and finite summaries, including empty axes."""
    batch, receivers, sources, channels = control.shape
    receiver_indices = torch.linspace(0, receivers - 1, min(4, receivers), device=control.device).long()
    source_indices = torch.linspace(0, sources - 1, min(16, sources), device=control.device).long()
    return {
        "control_probe_receiver_indices": receiver_indices,
        "control_probe_source_indices": source_indices,
        "control_probe": control[:, receiver_indices][:, :, source_indices],
        "control_receiver_max_abs": control.abs().amax(dim=(-1, -2))
        if sources and channels
        else control.new_zeros(batch, receivers),
        "control_receiver_nonzero_count": torch.count_nonzero(control, dim=(-1, -2)),
        "control_source_channel_mean": control.mean(dim=1)
        if receivers
        else control.new_zeros(batch, sources, channels),
    }


def projected_action_summary(projected):
    """Explicit small affine-action summaries, never full control vectors."""
    return {name.replace("control_", "projected_action_", 1): value
            for name, value in control_summary(projected).items()}


def project_diagnostic_control(backend, control, mechanism, *, mode="normal"):
    """Project an already materialized explicit diagnostic access."""
    gains = getattr(backend, "control_gain", None)
    if gains is None:
        return None
    gain = gains[mechanism](control.to(gains[mechanism].weight.dtype))
    if mechanism == "QE":
        score = backend.control_score(control.to(backend.control_score.weight.dtype))
        gain = torch.cat((score, gain), -1)
    return torch.zeros_like(gain) if mode == "control_identity" else gain


class TypedWorkEvidenceRecorder(AbstractContextManager):
    """Record post-intervention plans and source weights without extra reads.

    Every actual prepare/read access keeps its physical receiver coordinates,
    source identities and phase. No anchor-only reconstruction replaces a
    native access. Methods are restored on success or failure.
    """

    def __init__(self, backend):
        if backend.training:
            raise ValueError("Typed work evidence is evaluation-only")
        self.backend = backend
        self.arrays = {}
        self._counts = {}
        self._access_prefixes = {}

    def __enter__(self):
        names = ["prepare", "_access"]
        if callable(getattr(self.backend, "_numerical_access", None)):
            names.append("_numerical_access")
        if callable(getattr(self.backend, "_ledger", None)):
            names.append("_ledger")
        self._owned = {name: name in self.backend.__dict__ for name in names}
        self._original = {name: getattr(self.backend, name) for name in self._owned}

        def record(value, plan, receivers, mechanism, *, mode=None):
            tau = str(mechanism).upper()
            route = (int(plan.phase), tau)
            index = self._counts.get(route, 0)
            self._counts[route] = index + 1
            prefix = f"access/P{plan.phase}/{tau}/{index:05d}"
            self._access_prefixes[id(value)] = prefix
            kind = "M" if tau in {"MM", "EM", "QM"} else "E"
            for name, tensor in {
                "receivers": receivers,
                "source_ids": plan.source_ids[kind],
                "source_coords": plan.source_coords[kind],
                "source_measures": plan.source_measures[kind],
                "source_valid": plan.source_valid[kind],
                "support": value.support,
                "density": value.density,
                "weight": value.weight,
                "edge_access": value.edge_access,
                "near": value.near,
            }.items():
                if tensor is not None:
                    _flatten(f"{prefix}/{name}", tensor, self.arrays)
            _flatten(f"{prefix}/diagnostics", value.diagnostics, self.arrays)
            # Keep small deterministic probes and complete per-receiver control
            # summaries rather than the large R x S x 16 tensor.
            if hasattr(value, "control"):
                for name, tensor in control_summary(value.control).items():
                    _flatten(f"{prefix}/{name}", tensor, self.arrays)
                projected = project_diagnostic_control(self.backend, value.control, tau,
                    mode=mode or getattr(self.backend, "plan_intervention", "normal"))
            else:
                projected = value.projected
            if projected is not None:
                for name, tensor in projected_action_summary(projected).items():
                    _flatten(f"{prefix}/{name}", tensor, self.arrays)
                self.arrays[f"{prefix}/projected_action_semantics"] = np.asarray(
                    "affine gain" if tau != "QE" else "affine score channels followed by affine gain channels")
            return value

        def access(plan, receivers, mechanism, *args, **kwargs):
            value = self._original["_access"](plan, receivers, mechanism, *args, **kwargs)
            return record(value, plan, receivers, mechanism, mode=kwargs.get("mode"))

        def numerical_access(plan, receivers, mechanism, *args, **kwargs):
            route = (int(plan.phase), str(mechanism).upper())
            before = self._counts.get(route, 0)
            value = self._original["_numerical_access"](plan, receivers, mechanism, *args, **kwargs)
            prefix = f"access/P{route[0]}/{route[1]}/{before:05d}"
            # A full-control fallback calls the wrapped public _access once.
            # Deduplicate only that nested call, never a reused Python object
            # address from an earlier numerical chunk.
            if self._counts.get(route, 0) > before and self._access_prefixes.get(id(value)) == prefix:
                return value
            return record(value, plan, receivers, mechanism, mode=kwargs.get("mode"))

        def prepare(*args, **kwargs):
            state = self._original["prepare"](*args, **kwargs)
            plan = state["hypergraph_plan"]
            prefix = f"phase/P{plan.phase}"
            if f"{prefix}/group_admission" in self.arrays:
                raise ValueError("Expected one native preparation per physical phase")
            # Tensor-H's numerical control organization differs from the
            # full-access Global-C value plan carried underneath it.
            exported = (self.backend.export_typed_state(state)
                        if hasattr(self.backend, "tensor_residual") else plan.export())
            for name in (
                "source_ids",
                "source_coords",
                "source_measures",
                "source_valid",
                "source_lengths",
                "group_admission",
                "source_membership",
                "control_membership",
                "control_presence",
                "group_controls",
                "group_centres",
                "diagnostics",
            ):
                _flatten(f"{prefix}/{name}", exported[name], self.arrays)
            if "tensor_residual" in exported:
                for name in ("base_route_controls", "value_membership", "control_donor_measures",
                             "residual_gamma", "global_centering", "reference_receiver_coords",
                             "reference_receiver_weights"):
                    _flatten(f"{prefix}/{name}", exported[name], self.arrays)
                tensor_plan = exported["tensor_residual"]["plan"]
                for name in ("centres", "valid", "descriptors", "keys", "admission", "density",
                             "measures", "global_state", "coordinate_scale"):
                    _flatten(f"{prefix}/tensor_source_plan/{name}", getattr(tensor_plan, name), self.arrays)
                self.arrays[f"{prefix}/additional_age"] = np.asarray(exported["additional_age"])
                self.arrays[f"{prefix}/sparse_fraction"] = np.asarray(exported["sparse_fraction"])
                self.arrays[f"{prefix}/control_presence_scope"] = np.asarray("base Global-C full-access value plan")
            self.arrays[f"{prefix}/dependency_provenance_json"] = np.asarray(json.dumps(
                exported.get("dependency_provenance", {}), sort_keys=True))
            for name in ("typed_admission", "typed_centres"):
                _flatten(f"{prefix}/{name}", plan.strategy_data.get(name, {}), self.arrays)
            return state

        self.backend.prepare, self.backend._access = prepare, access
        if "_numerical_access" in self._original:
            self.backend._numerical_access = numerical_access
        if "_ledger" in self._original:

            def ledger(mechanism, source_access, work):
                result = self._original["_ledger"](mechanism, source_access, work)
                prefix = self._access_prefixes[id(source_access)]
                _flatten(f"{prefix}/executor", result, self.arrays)
                return result

            self.backend._ledger = ledger
        return self

    def __exit__(self, *_args):
        for name, owned in self._owned.items():
            if owned:
                setattr(self.backend, name, self._original[name])
            else:
                delattr(self.backend, name)
        return False


def compare_native_access(reference, changed):
    """Audit changes at identical actual physical source/receiver rows."""
    prefixes = sorted(
        key.removesuffix("/support") for key in reference if key.startswith("access/") and key.endswith("/support")
    )
    changed_prefixes = sorted(
        key.removesuffix("/support") for key in changed if key.startswith("access/") and key.endswith("/support")
    )
    if prefixes != changed_prefixes:
        raise ValueError("Native receiver streams differ between interventions")
    routes = {}
    for prefix in prefixes:
        route = "/".join(prefix.split("/")[1:3])
        row = routes.setdefault(
            route,
            {
                "calls": 0,
                "normal_positive_pairs": 0,
                "intervention_positive_pairs": 0,
                "changed_pairs": 0,
                "max_absolute_weight_change": 0.0,
                "source_geometry_identity_equal": True,
                "per_receiver_weight_multiset_equal": True,
                "per_receiver_support_cardinality_equal": True,
                "control_probe_equal": True,
                "control_receiver_summary_equal": True,
                "projected_action_probe_equal": True,
                "projected_action_receiver_summary_equal": True,
                "normal_eligible_pairs": 0,
                "intervention_eligible_pairs": 0,
            },
        )
        for name in ("receivers", "source_ids", "source_coords", "source_measures", "source_valid"):
            same = np.array_equal(reference[f"{prefix}/{name}"], changed[f"{prefix}/{name}"])
            row["source_geometry_identity_equal"] &= same
        if not row["source_geometry_identity_equal"]:
            raise ValueError("Intervention changed physical receiver/source geometry, identity or measures")
        before, after = reference[f"{prefix}/support"], changed[f"{prefix}/support"]
        weight, alternative = reference[f"{prefix}/weight"], changed[f"{prefix}/weight"]
        row["calls"] += 1
        for arm, arrays in (("normal", reference), ("intervention", changed)):
            eligible_key = f"{prefix}/diagnostics/eligible_pairs"
            row[f"{arm}_eligible_pairs"] += (
                int(arrays[eligible_key])
                if eligible_key in arrays
                else int(arrays[f"{prefix}/source_valid"].sum() * arrays[f"{prefix}/receivers"].shape[1])
            )
        for representation in ("control", "projected_action"):
            probe_key = f"{prefix}/{representation}_probe"
            probe_stat = f"{representation}_probe_equal"
            summary_stat = f"{representation}_receiver_summary_equal"
            if probe_key not in reference or probe_key not in changed:
                row[probe_stat] = None
                row[summary_stat] = None
                continue
            # Project-before-mixture changes FP operation order within the
            # tested tolerance; control-vector legacy comparisons stay exact.
            def equal(before, after, _representation=representation):
                return (np.allclose(before, after, atol=1e-6, rtol=1e-6)
                        if _representation == "projected_action" else np.array_equal(before, after))
            row[probe_stat] &= bool(equal(reference[probe_key], changed[probe_key]))
            row[summary_stat] &= all(equal(reference[f"{prefix}/{name}"], changed[f"{prefix}/{name}"])
                for name in (f"{representation}_receiver_max_abs", f"{representation}_receiver_nonzero_count",
                             f"{representation}_source_channel_mean"))
        row["normal_positive_pairs"] += int(np.count_nonzero(before))
        row["intervention_positive_pairs"] += int(np.count_nonzero(after))
        row["changed_pairs"] += int(np.count_nonzero(before != after))
        row["max_absolute_weight_change"] = max(
            row["max_absolute_weight_change"], float(np.max(np.abs(weight - alternative))) if weight.size else 0.0
        )
        row["per_receiver_weight_multiset_equal"] &= bool(
            np.allclose(np.sort(weight, axis=-1), np.sort(alternative, axis=-1), atol=1e-6, rtol=1e-6)
        )
        row["per_receiver_support_cardinality_equal"] &= bool(np.array_equal(before.sum(-1), after.sum(-1)))
    return routes


__all__ = ["TypedWorkEvidenceRecorder", "compare_native_access", "control_summary", "project_diagnostic_control", "projected_action_summary"]

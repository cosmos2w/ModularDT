"""Frozen evaluation isolation of controls with saved normal permissions."""

from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from honf_forward_core.interface_fields.typed_hypergraph_state import (
    ProjectedSourceAccess,
    prepare_projected_action,
    source_moments,
)

from .geometry_budget import geometry_action_budget
from .typed_work_evidence import control_summary, project_diagnostic_control, projected_action_summary


class FixedReferenceAccessReplay(AbstractContextManager):
    """Replay saved permissions or actions while keeping physical values live.

    Source/receiver identity, geometry, measure and eligibility must match at
    every phase/route/call. Current organizer plans are still computed and are
    not reference plans; returned access tensors are the authoritative replay.
    Older recorder files lack density: reconstruct it from their saved normal
    membership/access and verify their normalized weight and support equations.
    ``normal_fixed_actions`` retains both normal permissions and pair controls,
    allowing a changed prescribed context to be evaluated at fixed actions.
    """

    def __init__(self, backend, reference, *, mode="control_identity"):
        if backend.training:
            raise ValueError("Reference access replay is evaluation-only")
        self.backend = backend
        if mode not in {"control_identity", "normal_fixed_actions", "full_access_fixed_controls", "geometry_reference_actions"}:
            raise ValueError("Unknown reference action mode")
        self.mode = mode
        self._file = np.load(Path(reference), allow_pickle=False) if isinstance(reference, (str, Path)) else None
        self.reference = self._file if self._file is not None else reference
        self.expected = {
            key.removesuffix("/weight")
            for key in self.reference
            if key.startswith("access/") and key.endswith("/weight")
        }
        if not self.expected:
            if self._file is not None:
                self._file.close()
            raise ValueError("Reference has no actual native access calls")
        self.seen = set()
        self.counts = {}
        self.density_reconstruction_max_weight_error = 0.0
        self.action_statistics = {}

    def __enter__(self):
        self._owned = "_access" in self.backend.__dict__
        self._original = self.backend._access
        self._previous_mode = self.backend.plan_intervention
        self._previous_control_execution = getattr(self.backend, "control_execution", None)
        self._numerical_owned = "_numerical_access" in self.backend.__dict__
        self._numerical_original = getattr(self.backend, "_numerical_access", None)
        if self._previous_control_execution is not None:
            # Full vectors are deliberate only inside this explicit diagnostic
            # intervention context; ordinary inference keeps projected actions.
            self.backend.control_execution = "full_control"
        self.backend.set_plan_intervention("control_identity" if self.mode == "control_identity" else "normal")

        def access(plan, receivers, mechanism, *args, **kwargs):
            if kwargs.get("soft", False) or getattr(self.backend, "permission_mode", "hard") != "hard":
                raise ValueError("Reference replay requires hard frozen evaluation")
            current = self._original(plan, receivers, mechanism, *args, **kwargs)
            route = (int(plan.phase), str(mechanism).upper())
            ordinal = self.counts.get(route, 0)
            self.counts[route] = ordinal + 1
            prefix = f"access/P{route[0]}/{route[1]}/{ordinal:05d}"
            if prefix not in self.expected:
                raise ValueError(f"Extra or mismatched native reference call: {prefix}")
            kind = "M" if route[1] in {"MM", "EM", "QM"} else "E"
            for name, value in {
                "receivers": receivers,
                "source_ids": plan.source_ids[kind],
                "source_coords": plan.source_coords[kind],
                "source_measures": plan.source_measures[kind],
                "source_valid": plan.source_valid[kind],
                "diagnostics/pair_valid": current.diagnostics["pair_valid"],
            }.items():
                if not np.array_equal(value.detach().cpu().numpy(), self.reference[f"{prefix}/{name}"]):
                    raise ValueError(f"Reference {name} mismatch at {prefix}")

            def tensor(name):
                return torch.as_tensor(self.reference[f"{prefix}/{name}"], device=current.weight.device)

            edge_access, weight, support = tensor("edge_access"), tensor("weight"), tensor("support")
            if weight.shape != current.weight.shape or edge_access.shape[:2] != current.edge_access.shape[:2]:
                raise ValueError(f"Reference permission shape mismatch at {prefix}")
            near = tensor("near") if f"{prefix}/near" in self.reference else None
            if (near is None) != (current.near is None) or (near is not None and not torch.equal(near, current.near)):
                raise ValueError(f"Reference near envelope mismatch at {prefix}")
            membership = torch.as_tensor(
                self.reference[f"phase/P{route[0]}/source_membership/{route[1]}"], device=weight.device
            )
            reconstructed = source_moments(
                edge_access,
                membership,
                weight.new_zeros((*membership.shape[:2], current.control.shape[-1])),
                tensor("source_measures"),
                tensor("source_valid"),
                pair_valid=tensor("diagnostics/pair_valid"),
                near=near,
            )
            if not torch.allclose(reconstructed.weight, weight, atol=1e-6, rtol=1e-6) or not torch.equal(
                reconstructed.support, support
            ):
                raise ValueError(f"Reference normalized weight/support equation mismatch at {prefix}")
            self.density_reconstruction_max_weight_error = max(
                self.density_reconstruction_max_weight_error,
                float((reconstructed.weight - weight).abs().max()) if weight.numel() else 0.0,
            )
            density = tensor("density") if f"{prefix}/density" in self.reference else reconstructed.density
            if not torch.allclose(density, reconstructed.density, atol=1e-6, rtol=1e-6):
                raise ValueError(f"Reference density equation mismatch at {prefix}")
            diagnostics = {
                key.removeprefix(f"{prefix}/diagnostics/"): torch.as_tensor(self.reference[key], device=weight.device)
                for key in self.reference
                if key.startswith(f"{prefix}/diagnostics/")
            }
            control = torch.zeros_like(current.control)
            if self.mode != "control_identity":
                group_controls = torch.as_tensor(
                    self.reference[f"phase/P{route[0]}/group_controls/{route[1]}"], device=weight.device
                )
                normal_control = source_moments(
                    edge_access,
                    membership,
                    group_controls,
                    tensor("source_measures"),
                    tensor("source_valid"),
                    pair_valid=tensor("diagnostics/pair_valid"),
                    near=near,
                ).control
                if f"{prefix}/control_probe" in self.reference:
                    summaries = control_summary(normal_control)
                else:
                    projected = project_diagnostic_control(self.backend, normal_control, route[1])
                    if projected is None:
                        raise ValueError("Projected reference requires the native physical projection")
                    summaries = projected_action_summary(projected)
                for name, observed in summaries.items():
                    expected = tensor(name)
                    if observed.dtype.is_floating_point:
                        equal = torch.allclose(observed, expected, atol=1e-6, rtol=1e-6)
                    else:
                        equal = torch.equal(observed, expected)
                    if not equal:
                        raise ValueError(f"Reference reconstructed control {name} mismatch at {prefix}")
                control = normal_control
                valid = diagnostics["pair_valid"]
                if self.mode == "full_access_fixed_controls":
                    density = valid.to(weight.dtype)
                    weight = density.clone()
                    support = valid.clone()
                elif self.mode == "geometry_reference_actions":
                    (density, weight, support, control), statistics = geometry_action_budget(
                        density,
                        weight,
                        support,
                        control,
                        receivers,
                        tensor("source_coords"),
                        tensor("source_measures"),
                        valid,
                        near,
                        tensor("source_ids"),
                    )
                    self.action_statistics[prefix] = statistics
                if self.mode != "normal_fixed_actions":
                    diagnostics["unique_pairs"] = support.sum()
                    diagnostics["far_unique_pairs"] = (valid & (density > 0)).sum()
                    # Reassigned source actions have no new shared-group path decomposition.
                    diagnostics.pop("repeated_paths_removed", None)
            self.seen.add(prefix)
            return replace(
                current,
                density=density,
                weight=weight,
                edge_access=edge_access,
                support=support,
                near=near,
                diagnostics=diagnostics,
                control=control,
            )

        self.backend._access = access
        if callable(self._numerical_original):
            def numerical_access(plan, receivers, mechanism, actions, *args, **kwargs):
                kwargs.pop("diagnostics", None)
                return self.backend._access(plan, receivers, mechanism, *args, **kwargs)
            self.backend._numerical_access = numerical_access
        return self

    def __exit__(self, exception_type, *_args):
        if self._owned:
            self.backend._access = self._original
        else:
            delattr(self.backend, "_access")
        if callable(self._numerical_original):
            if self._numerical_owned:
                self.backend._numerical_access = self._numerical_original
            else:
                delattr(self.backend, "_numerical_access")
        if self._previous_control_execution is not None:
            self.backend.control_execution = self._previous_control_execution
        self.backend.set_plan_intervention(self._previous_mode)
        if self._file is not None:
            self._file.close()
        if exception_type is None and self.seen != self.expected:
            raise ValueError(f"Missing native reference calls: {sorted(self.expected - self.seen)}")
        return False


class ProjectedRouteUniformReplay(AbstractContextManager):
    """Hold normal permissions; replace affine actions by route-wide means.

    For each phase/case/route, the mean weights every recorded actual call by
    receiver measure * source measure * normal fine weight. Repeated physical
    reads are deliberately counted. Query receivers use equal-point measure.
    Means are formed before tanh, independently for QE score/gain channels;
    a source-independent QE score cancels in the source softmax. No physical
    values or full receiver/source/control-vector tensors are reconstructed.
    """

    def __init__(self, backend, reference):
        if backend.training:
            raise ValueError("Route-uniform replay is evaluation-only")
        self.backend = backend
        self._file = np.load(Path(reference), allow_pickle=False) if isinstance(reference, (str, Path)) else None
        self.reference = self._file if self._file is not None else reference
        self.expected = {key.removesuffix("/weight") for key in self.reference
                         if key.startswith("access/") and key.endswith("/weight")}
        if not self.expected:
            if self._file is not None:
                self._file.close()
            raise ValueError("Reference has no actual native access calls")
        self.means, self.denominators, self.action_statistics = {}, {}, {}
        self.seen, self.counts = set(), {}
        self.density_reconstruction_max_weight_error = 0.0

    def _tensor(self, prefix, name):
        device = self.backend.control_score.weight.device
        return torch.as_tensor(self.reference[f"{prefix}/{name}"], device=device)

    @staticmethod
    def _route(prefix):
        parts = prefix.split("/")
        return int(parts[1].removeprefix("P")), parts[2]

    def _receiver_measure(self, prefix, route):
        key = f"{prefix}/diagnostics/receiver_measures"
        if key in self.reference:
            return self._tensor(prefix, "diagnostics/receiver_measures"), "recorded_receiver_measure"
        kind = "M" if route[1] in ("MM", "ME") else "E" if route[1] == "EM" else None
        if kind is not None:
            return self._tensor(f"phase/P{route[0]}", f"source_measures/{kind}"), "legacy_physical_receiver_measure"
        return torch.ones(self.reference[f"{prefix}/receivers"].shape[:2],
                          device=self.backend.control_score.weight.device), "equal_query_point_measure"

    def _normal_access(self, prefix, action, membership, controls):
        near = self._tensor(prefix, "near") if f"{prefix}/near" in self.reference else None
        normal = source_moments(self._tensor(prefix, "edge_access"), membership, controls,
            self._tensor(prefix, "source_measures"), self._tensor(prefix, "source_valid"),
            pair_valid=self._tensor(prefix, "diagnostics/pair_valid"), near=near,
            prepared_action=action, include_diagnostics=False)
        for name in ("weight", "density", "support"):
            if f"{prefix}/{name}" not in self.reference:
                if name == "density":
                    continue  # Legacy density is recovered by the same equation.
                raise ValueError(f"Reference missing {name} at {prefix}")
            expected, value = self._tensor(prefix, name), getattr(normal, name)
            equal = (torch.equal(value, expected) if name == "support" else
                     torch.allclose(value, expected, atol=1e-6, rtol=1e-6))
            if not equal:
                raise ValueError(f"Reference {name} equation mismatch at {prefix}")
            if name == "weight" and value.numel():
                self.density_reconstruction_max_weight_error = max(
                    self.density_reconstruction_max_weight_error, float((value - expected).abs().max()))
        for name, observed in projected_action_summary(normal.projected).items():
            if f"{prefix}/{name}" not in self.reference:
                raise ValueError(f"Reference missing projected probe {name} at {prefix}")
            expected = self._tensor(prefix, name)
            equal = (torch.allclose(observed, expected, atol=1e-6, rtol=1e-6)
                     if observed.dtype.is_floating_point else torch.equal(observed, expected))
            if not equal:
                raise ValueError(f"Reference projection/checkpoint mismatch for {name} at {prefix}")
        return normal

    def _prepare_means(self):
        routes = sorted({self._route(prefix) for prefix in self.expected})
        with torch.no_grad():
            for route in routes:
                phase, tau = route
                prefixes = sorted(prefix for prefix in self.expected if self._route(prefix) == route)
                if [int(prefix.rsplit("/", 1)[1]) for prefix in prefixes] != list(range(len(prefixes))):
                    raise ValueError(f"Reference call ordinals are not contiguous for {route}")
                membership = self._tensor(f"phase/P{phase}", f"source_membership/{tau}")
                controls = self._tensor(f"phase/P{phase}", f"group_controls/{tau}")
                gain = self.backend.control_gain[tau]
                weight, bias = gain.weight, gain.bias
                if tau == "QE":
                    weight = torch.cat((self.backend.control_score.weight, weight), 0)
                    bias = torch.cat((self.backend.control_score.bias, bias), 0)
                action = prepare_projected_action(membership, controls, weight, bias)
                numerator = controls.new_zeros((controls.shape[0], weight.shape[0]), dtype=torch.float64)
                denominator = numerator.new_zeros(controls.shape[0])
                centred_second = torch.zeros_like(numerator)
                running_mean = torch.zeros_like(numerator)
                pairs = denominator.new_zeros(controls.shape[0], dtype=torch.long)
                allocated = 0
                measure_sources = set()
                for prefix in prefixes:
                    normal = self._normal_access(prefix, action, membership, controls)
                    receiver_mass, origin = self._receiver_measure(prefix, route)
                    measure_sources.add(origin)
                    if receiver_mass.shape != normal.weight.shape[:2]:
                        raise ValueError(f"Reference receiver measure shape mismatch at {prefix}")
                    source_mass = self._tensor(prefix, "source_measures")
                    if not all(bool(torch.isfinite(value).all()) for value in
                               (receiver_mass, source_mass, normal.weight, normal.density,
                                normal.edge_access, normal.projected)):
                        raise ValueError(f"Nonfinite normal action/measure at {prefix}")
                    if any(bool((value < 0).any()) for value in
                           (receiver_mass, source_mass, normal.weight, normal.density, normal.edge_access)):
                        raise ValueError(f"Negative normal measure/weight at {prefix}")
                    mass = (receiver_mass.double()[:, :, None] * source_mass.double()[:, None]
                            * self._tensor(prefix, "weight").double())
                    affine = normal.projected.double()
                    numerator += (mass[..., None] * affine).sum((1, 2))
                    chunk_mass = mass.sum((1, 2))
                    chunk_mean = (mass[..., None] * affine).sum((1, 2)) / torch.where(
                        chunk_mass > 0, chunk_mass, torch.ones_like(chunk_mass))[:, None]
                    chunk_second = (mass[..., None] * (affine - chunk_mean[:, None, None]).square()).sum((1, 2))
                    total = denominator + chunk_mass
                    safe_total = torch.where(total > 0, total, torch.ones_like(total))
                    delta = chunk_mean - running_mean
                    centred_second += chunk_second + delta.square() * (denominator * chunk_mass / safe_total)[:, None]
                    running_mean += delta * (chunk_mass / safe_total)[:, None]
                    denominator = total
                    pairs += (mass > 0).sum((1, 2))
                    allocated += mass.numel()
                mean = numerator / torch.where(denominator > 0, denominator, torch.ones_like(denominator))[:, None]
                variance = centred_second / torch.where(denominator > 0, denominator, torch.ones_like(denominator))[:, None]
                self.means[route], self.denominators[route] = mean.to(bias.dtype), denominator
                cast_error = mean - self.means[route].double()
                change_rms = (variance + cast_error.square()).sqrt()
                self.action_statistics[f"P{phase}/{tau}"] = {
                    "calls": len(prefixes), "allocated_pair_stream_rows": allocated,
                    "positive_measure_pair_stream_rows_by_case": pairs.cpu().tolist(),
                    "denominator_by_case": denominator.cpu().tolist(),
                    "affine_mean_by_case": self.means[route].cpu().tolist(),
                    "affine_accumulation_mean_float64_by_case": mean.cpu().tolist(),
                    "mean_cast_max_abs_error": float(cast_error.abs().max()) if cast_error.numel() else 0.0,
                    "weighted_affine_change_rms_by_case_channel": change_rms.cpu().tolist(),
                    "weighted_affine_change_exceeds_projection_atol_by_case_channel": (change_rms > 1e-6).cpu().tolist(),
                    "projection_check_atol": 1e-6, "projection_check_rtol": 1e-6,
                    "zero_measure_cases": (denominator == 0).cpu().tolist(),
                    "receiver_measure_sources": sorted(measure_sources),
                    "weighting": "mu_receiver * mu_source * saved_normal_fine_weight; all actual calls",
                    "repeated_physical_reads_counted": True,
                    "diagnostic_projected_reconstruction_calls": len(prefixes),
                    "full_pair_control_vectors_materialized": False,
                    "qe_constant_score_cancels": tau == "QE",
                }

    def __enter__(self):
        if (self.backend.training or getattr(self.backend, "permission_mode", "hard") != "hard"
                or self.backend.plan_intervention != "normal"
                or self.backend.control_execution != "projected"):
            if self._file is not None:
                self._file.close()
            raise ValueError("Route-uniform replay requires ordinary hard projected evaluation")
        try:
            self._prepare_means()
        except BaseException:
            if self._file is not None:
                self._file.close()
            raise
        self._owned = "_numerical_access" in self.backend.__dict__
        self._original = self.backend._numerical_access

        def numerical_access(plan, receivers, mechanism, actions, *args, **kwargs):
            if (self.backend.training or kwargs.get("soft", False)
                    or kwargs.get("mode", "normal") not in (None, "normal") or actions is None):
                raise ValueError("Route-uniform replay requires ordinary projected actions")
            current = self._original(plan, receivers, mechanism, actions, *args, **kwargs)
            if not isinstance(current, ProjectedSourceAccess):
                raise TypeError("Route-uniform replay must retain projected source access")
            route = int(plan.phase), str(mechanism).upper()
            ordinal = self.counts.get(route, 0)
            self.counts[route] = ordinal + 1
            prefix = f"access/P{route[0]}/{route[1]}/{ordinal:05d}"
            if prefix not in self.expected:
                raise ValueError(f"Extra or mismatched native reference call: {prefix}")
            kind = "M" if route[1] in ("MM", "EM", "QM") else "E"
            for name, value in {"receivers": receivers, "source_ids": plan.source_ids[kind],
                    "source_coords": plan.source_coords[kind], "source_measures": plan.source_measures[kind],
                    "source_valid": plan.source_valid[kind],
                    "diagnostics/pair_valid": current.diagnostics["pair_valid"]}.items():
                if not np.array_equal(value.detach().cpu().numpy(), self.reference[f"{prefix}/{name}"]):
                    raise ValueError(f"Reference {name} mismatch at {prefix}")
            receiver_mass, _ = self._receiver_measure(prefix, route)
            receiver_kind = "M" if route[1] in ("MM", "ME") else "E" if route[1] == "EM" else None
            actual_receiver_mass = current.diagnostics.get("receiver_measures",
                plan.source_measures[receiver_kind] if receiver_kind is not None else receivers.new_ones(receivers.shape[:2]))
            if not torch.equal(receiver_mass, actual_receiver_mass):
                raise ValueError(f"Reference receiver measure mismatch at {prefix}")
            near = self._tensor(prefix, "near") if f"{prefix}/near" in self.reference else None
            if (near is None) != (current.near is None) or (near is not None and not torch.equal(near, current.near)):
                raise ValueError(f"Reference near envelope mismatch at {prefix}")
            saved = {name: self._tensor(prefix, name) for name in ("weight", "support", "edge_access")}
            if saved["weight"].shape != current.weight.shape or saved["edge_access"].shape != current.edge_access.shape:
                raise ValueError(f"Reference permission shape mismatch at {prefix}")
            density_key = f"{prefix}/density"
            if density_key in self.reference:
                density = self._tensor(prefix, "density")
            else:
                membership = self._tensor(f"phase/P{route[0]}", f"source_membership/{route[1]}")
                density = torch.bmm(saved["edge_access"], membership)
                density = torch.where(current.diagnostics["pair_valid"], density, torch.zeros_like(density))
            diagnostics = {key.removeprefix(f"{prefix}/diagnostics/"): self._tensor(prefix, key.removeprefix(f"{prefix}/"))
                           for key in self.reference if key.startswith(f"{prefix}/diagnostics/")}
            affine = self.means[route][:, None, None].expand_as(current.projected)
            self.seen.add(prefix)
            return replace(current, density=density, near=near, diagnostics=diagnostics,
                           projected=affine, **saved)

        self.backend._numerical_access = numerical_access
        return self

    def __exit__(self, exception_type, *_args):
        if self._owned:
            self.backend._numerical_access = self._original
        else:
            delattr(self.backend, "_numerical_access")
        if self._file is not None:
            self._file.close()
        if exception_type is None and self.seen != self.expected:
            raise ValueError(f"Missing native reference calls: {sorted(self.expected - self.seen)}")
        return False


__all__ = ["FixedReferenceAccessReplay", "ProjectedRouteUniformReplay"]

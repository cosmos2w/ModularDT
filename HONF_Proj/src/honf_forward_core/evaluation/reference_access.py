"""Frozen evaluation isolation of controls with saved normal permissions."""

from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from honf_forward_core.interface_fields.typed_hypergraph_state import source_moments

from .geometry_budget import geometry_action_budget
from .typed_work_evidence import control_summary, project_diagnostic_control, projected_action_summary


class FixedReferenceAccessReplay(AbstractContextManager):
    """Replay permissions alone, leaving all current physical source values live.

    Source/receiver identity, geometry, measure and eligibility must match at
    every phase/route/call. Current organizer plans are still computed and are
    not reference plans; returned access tensors are the authoritative replay.
    Older recorder files lack density: reconstruct it from their saved normal
    membership/access and verify their normalized weight and support equations.
    """

    def __init__(self, backend, reference, *, mode="control_identity"):
        if backend.training:
            raise ValueError("Reference access replay is evaluation-only")
        self.backend = backend
        if mode not in {"control_identity", "full_access_fixed_controls", "geometry_reference_actions"}:
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
                else:
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
                diagnostics["unique_pairs"] = support.sum()
                diagnostics["far_unique_pairs"] = (valid & (density > 0)).sum()
                # Source-resolved actions do not assert a new shared-group path decomposition.
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


__all__ = ["FixedReferenceAccessReplay"]

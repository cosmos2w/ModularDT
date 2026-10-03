"""Frozen evaluation isolation of controls with saved normal permissions."""

from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from honf_forward_core.interface_fields.typed_hypergraph_state import source_moments


class FixedReferenceAccessReplay(AbstractContextManager):
    """Replay permissions alone, leaving all current physical source values live.

    Source/receiver identity, geometry, measure and eligibility must match at
    every phase/route/call. Current organizer plans are still computed and are
    not reference plans; returned access tensors are the authoritative replay.
    Older recorder files lack density: reconstruct it from their saved normal
    membership/access and verify their normalized weight and support equations.
    """

    def __init__(self, backend, reference):
        if backend.training:
            raise ValueError("Reference access replay is evaluation-only")
        self.backend = backend
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

    def __enter__(self):
        self._owned = "_access" in self.backend.__dict__
        self._original = self.backend._access
        self._previous_mode = self.backend.plan_intervention
        self.backend.set_plan_intervention("control_identity")

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
            self.seen.add(prefix)
            return replace(
                current,
                density=density,
                weight=weight,
                edge_access=edge_access,
                support=support,
                near=near,
                diagnostics=diagnostics,
                control=torch.zeros_like(current.control),
            )

        self.backend._access = access
        return self

    def __exit__(self, exception_type, *_args):
        if self._owned:
            self.backend._access = self._original
        else:
            delattr(self.backend, "_access")
        self.backend.set_plan_intervention(self._previous_mode)
        if self._file is not None:
            self._file.close()
        if exception_type is None and self.seen != self.expected:
            raise ValueError(f"Missing native reference calls: {sorted(self.expected - self.seen)}")
        return False


__all__ = ["FixedReferenceAccessReplay"]

"""Evaluation-only continuation of a recorded discrete organizer topology.

No soft permissions replace the hard operator. The historical default fixes
active source/group sets while their affine sparsemax continuation and
physical controls remain live. Negative continuation values terminate that
mathematical branch, rather than establish physical design infeasibility.
The explicitly named fixed-frontier mode keeps receiver organization while
recomputing ordinary source projections; source donor support can change.
"""

from contextlib import AbstractContextManager
from dataclasses import dataclass

import torch

from honf_forward_core.training.hypergraph_shadow import detach_tree


class FixedTopologyInvalid(ValueError):
    """A local continuation cannot be used as nonnegative physical access."""


def fixed_active_projection(logits, reference_logits, reference_density, measure):
    """Affine sparsemax continuation with exact recorded values at its anchor.

    Measures must be unchanged along this local probe. They are uniform for
    receiver sparsemax and normalized physical measures for source density.
    The Jacobian remains the active-set weighted centering operation.
    """
    if logits.shape != reference_logits.shape or logits.shape != reference_density.shape:
        raise FixedTopologyInvalid("Projection axes differ from recorded topology")
    support = reference_density > 0
    work_dtype = torch.float64 if logits.dtype in (torch.float16, torch.bfloat16, torch.float32) else logits.dtype
    mu = measure
    while mu.ndim < logits.ndim:
        mu = mu.unsqueeze(-2)
    mu = torch.broadcast_to(mu, logits.shape).to(work_dtype)
    delta = logits.to(work_dtype) - reference_logits.to(device=logits.device, dtype=work_dtype)
    weight = torch.where(support, mu, torch.zeros_like(mu))
    total = weight.sum(-1, keepdim=True)
    correction = (weight * torch.where(support, delta, torch.zeros_like(delta))).sum(-1, keepdim=True)
    correction = correction / torch.where(total > 0, total, torch.ones_like(total))
    continued = torch.where(support, reference_density.to(device=logits.device, dtype=work_dtype) + delta - correction,
                            torch.zeros_like(delta))
    if not bool(torch.isfinite(continued).all()) or bool((continued < 0).any()):
        minimum = float(continued.detach().amin()) if continued.numel() else 0.
        raise FixedTopologyInvalid(f"Affine active-set continuation is negative or nonfinite (minimum={minimum:.8g})")
    return continued.to(logits.dtype)


def validate_catalogue(current, reference):
    for kind in ("M", "E"):
        for field in ("source_valid", "source_ids", "source_measures"):
            now = current[field][kind]
            old = getattr(reference, field)[kind]
            if now.shape != old.shape or not torch.equal(now.detach(), old.to(now).detach()):
                raise FixedTopologyInvalid(f"Fixed topology requires unchanged physical {kind} {field}")


@dataclass
class OrganizerTopologyRecord:
    states: dict
    accesses: dict


class OrganizerTopologyProbe(AbstractContextManager):
    """Record or replay prepare/access decisions on one identical query stream.

    A record is local to a complete wrapper call. Physical receiver rows must
    retain their order/shape. Methods are restored even if a branch exits its
    valid region; this context never changes training behavior or checkpoints.
    ``fixed_frontier_live_membership`` is receiver-tree-only and requires an
    existing record. Its live source projection is distinct from the default
    fixed-active-set continuation; receiver connectivity remains recorded.
    """

    def __init__(self, organizer, reference=None, *, topology_mode="fixed_active_set"):
        if organizer.training:
            raise ValueError("Topology probes require an evaluation organizer")
        if topology_mode not in {"fixed_active_set", "fixed_frontier_live_membership"}:
            raise ValueError(f"Unknown topology continuation mode: {topology_mode!r}")
        if topology_mode != "fixed_active_set":
            from .adaptive_receiver_hypergraph import AdaptiveReceiverHypergraph
            if reference is None:
                raise ValueError("Fixed-frontier continuation requires a reference record")
            if not isinstance(organizer, AdaptiveReceiverHypergraph):
                raise ValueError("Fixed-frontier live membership requires a receiver-tree organizer")
        self.organizer = organizer
        self.reference = reference
        self.topology_mode = topology_mode
        self.record = OrganizerTopologyRecord({}, {})
        self._counts = {}

    def __enter__(self):
        self._owned = {name: name in self.organizer.__dict__ for name in ("prepare", "access")}
        self._originals = {name: getattr(self.organizer, name) for name in ("prepare", "access")}

        def prepare(*args, **kwargs):
            phase = kwargs.get("phase", 0)
            phase = int(str(phase).removeprefix("P"))
            if phase in self.record.states:
                raise ValueError("A topology probe requires one prepare per physical phase")
            kwargs["capture_topology"] = True
            if self.reference is not None:
                kwargs["fixed_topology"] = self.reference.states[phase]
                if self.topology_mode != "fixed_active_set":
                    kwargs["topology_mode"] = self.topology_mode
            state = self._originals["prepare"](*args, **kwargs)
            self.record.states[phase] = detach_tree(state)
            return state

        def access(state, receivers, mechanism, *args, **kwargs):
            tau = str(mechanism).upper()
            route = (state.phase, tau)
            index = self._counts.get(route, 0)
            self._counts[route] = index + 1
            key = (*route, index)
            kwargs["capture_topology"] = True
            if self.reference is not None:
                saved = self.reference.accesses.get(key)
                if saved is None or tuple(receivers.shape) != saved["shape"]:
                    raise FixedTopologyInvalid("Receiver stream differs from recorded topology")
                kwargs["fixed_receiver_access"] = saved
            value = self._originals["access"](state, receivers, mechanism, *args, **kwargs)
            self.record.accesses[key] = {"shape": tuple(receivers.shape),
                "edge_access": value.edge_access.detach().clone(),
                "receiver_logits": value.diagnostics.get("receiver_logits", None),
                "support": value.support.detach().clone()}
            if torch.is_tensor(self.record.accesses[key]["receiver_logits"]):
                self.record.accesses[key]["receiver_logits"] = self.record.accesses[key]["receiver_logits"].detach().clone()
            return value

        self.organizer.prepare, self.organizer.access = prepare, access
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for name, owned in self._owned.items():
            if owned:
                setattr(self.organizer, name, self._originals[name])
            else:
                delattr(self.organizer, name)
        return False


__all__ = ["FixedTopologyInvalid", "OrganizerTopologyProbe", "OrganizerTopologyRecord", "fixed_active_projection"]

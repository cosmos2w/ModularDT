"""Small typed modules and numerics for native MM/ME/EM aggregation.

The route scores are evaluated from pair features before value messages. The
P-C control only sees a destination state and the global context. This module
does not own native IDs, measures, or prepared-state lifetimes; the shared core
supplies those so dataset adapters keep their physical input contracts.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

NATIVE_ROUTE_TYPES = ("MM", "ME", "EM")
_ROUTE_ENDPOINT_TYPES = {
    "MM": ("physical_source", "physical_source"),
    "ME": ("environment_record", "physical_source"),
    "EM": ("physical_source", "environment_record"),
}


def _score_mlp(inputs: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(),
                         nn.Linear(hidden, 1, bias=False))


def _gain_mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(),
                         nn.Linear(hidden, outputs))


def zero_final_projection(module: nn.Sequential) -> None:
    """Attach the identity function while leaving a live final-layer VJP."""
    final = module[-1]
    if not isinstance(final, nn.Linear):
        raise TypeError("Native routing MLP must end in Linear.")
    with torch.no_grad():
        final.weight.zero_()
        if final.bias is not None:
            final.bias.zero_()


class NativeInteractionRouting(nn.Module):
    """P-R typed score networks or a parameter-matched P-C destination gain."""

    SCORE_HIDDEN = 32
    SCORE_BOUND = 2.0

    def __init__(self, *, policy: str, pair_width: int, destination_width: int,
                 context_width: int, message_width: int, control_hidden: int | None,
                 initialization_seed: int) -> None:
        super().__init__()
        if policy not in ("learned_density", "destination_gain"):
            raise ValueError("Native interaction policy must be learned_density or destination_gain.")
        if min(pair_width, destination_width, context_width, message_width) < 1:
            raise ValueError("Native interaction widths must be positive.")
        if policy == "learned_density" and control_hidden is not None:
            raise ValueError("P-R has a fixed score width and no P-C control width.")
        if policy == "destination_gain" and (type(control_hidden) is not int or control_hidden < 1):
            raise ValueError("P-C requires its sealed positive hidden width.")

        self.policy = policy
        self.control_hidden = control_hidden
        self.score_mlps = nn.ModuleDict()
        self.aggregate_modulators = nn.ModuleDict()
        # This private RNG stream must not advance initialization of any
        # inherited P tensor or of later shared/readout modules.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(initialization_seed))
            if policy == "learned_density":
                for route in NATIVE_ROUTE_TYPES:
                    self.score_mlps[route] = _score_mlp(pair_width, self.SCORE_HIDDEN)
                    zero_final_projection(self.score_mlps[route])
            else:
                input_width = destination_width + context_width
                for route in NATIVE_ROUTE_TYPES:
                    self.aggregate_modulators[route] = _gain_mlp(
                        input_width, control_hidden, message_width)
                    zero_final_projection(self.aggregate_modulators[route])

    def score(self, route: str, pair_features: torch.Tensor) -> torch.Tensor:
        if self.policy != "learned_density":
            raise ValueError("Only P-R defines donor-selective route scores.")
        if route not in NATIVE_ROUTE_TYPES:
            raise ValueError(f"Unknown native route type {route!r}.")
        raw = self.score_mlps[route](pair_features)
        if not bool(torch.isfinite(raw).all()):
            raise FloatingPointError(f"Native {route} score MLP produced non-finite values.")
        return self.SCORE_BOUND * torch.tanh(raw / self.SCORE_BOUND).squeeze(-1)

    def gain(self, route: str, destination: torch.Tensor,
             context: torch.Tensor) -> torch.Tensor:
        if self.policy != "destination_gain":
            raise ValueError("Only P-C defines destination-only aggregate modulation.")
        if route not in NATIVE_ROUTE_TYPES:
            raise ValueError(f"Unknown native route type {route!r}.")
        context = context[:, None].expand(-1, destination.shape[1], -1)
        raw = self.aggregate_modulators[route](torch.cat((destination, context), -1))
        if not bool(torch.isfinite(raw).all()):
            raise FloatingPointError(f"Native {route} destination-gain MLP produced non-finite values.")
        return torch.tanh(raw)

    def added_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def density_normalize(base_weights: torch.Tensor,
                      scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact-zero-start density and measure-preserving donor weights.

    ``base_weights`` already contains physical measure, presence, self-exclusion
    and the legacy normalization. The expm1 form preserves that actual row mass
    and gives bit-exact ``rho == 1``/``weights == base_weights`` at zero scores.
    """
    if base_weights.shape != scores.shape:
        raise ValueError("Native route scores and base weights must have equal shapes.")
    if base_weights.ndim < 1:
        raise ValueError("Native route weights need a donor axis.")
    mass = base_weights.sum(-1, keepdim=True)
    safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
    score_delta = torch.expm1(scores)
    mean_delta = (base_weights * score_delta).sum(-1, keepdim=True) / safe_mass
    denominator = 1 + mean_delta
    normalized_density = 1 + (score_delta - mean_delta) / denominator
    rho = torch.where(mass > 0, normalized_density, torch.ones_like(normalized_density))
    weights = base_weights * rho
    return rho, weights


def normalize_intervention(value: Mapping | None) -> tuple[str | None, tuple[str, ...]]:
    """Validate a replayable route intervention without mutating caller data."""
    if value is None:
        return None, ()
    if not isinstance(value, Mapping):
        raise TypeError("native_route_intervention must be a mapping or None.")
    if set(value) - {"kind", "route_types"}:
        raise ValueError("Native route intervention has unsupported fields.")
    kind = value.get("kind")
    if kind not in ("density_reset", "equal_measure_reassignment"):
        raise ValueError("Native route intervention kind must be density_reset or equal_measure_reassignment.")
    route_types = value.get("route_types", NATIVE_ROUTE_TYPES)
    if isinstance(route_types, str):
        route_types = (route_types,)
    if not isinstance(route_types, (tuple, list)) or not route_types:
        raise ValueError("Native route intervention must select at least one typed route.")
    route_types = tuple(route_types)
    if any(route not in NATIVE_ROUTE_TYPES for route in route_types):
        raise ValueError("Native route intervention contains an unknown route type.")
    if len(set(route_types)) != len(route_types):
        raise ValueError("Native route intervention route types must be unique.")
    return kind, route_types


def reassign_equal_measure_densities(
    density: torch.Tensor,
    base_weights: torch.Tensor,
    measures: torch.Tensor,
    present: torch.Tensor,
    donor_ids: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Move densities among equal-measure/presence donors by stable native ID.

    Values and IDs remain fixed. Buckets also require positive base mass, so
    MM self-exclusion and absent/padded records cannot enter an intervention.
    """
    if density.shape != base_weights.shape or density.ndim != 3:
        raise ValueError("Density and base weights must align as [batch,destination,donor].")
    batch, destinations, donors = density.shape
    if measures.shape != (batch, donors) or present.shape != (batch, donors):
        raise ValueError("Native route donor measure/presence must align with the donor axis.")
    if donor_ids.shape != (batch, donors) or donor_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Native route donor IDs must be integer [batch,donor] values.")

    reassigned = density.clone()
    changed = skipped = eligible_buckets = eligible_donors = 0
    for batch_index in range(batch):
        keys = torch.stack((measures[batch_index], present[batch_index]), dim=-1)
        _unique, bucket_ids = torch.unique(keys, sorted=True, return_inverse=True, dim=0)
        active = (measures[batch_index] > 0) & (present[batch_index] > 0)
        for destination in range(destinations):
            route_active = active & (base_weights[batch_index, destination] > 0)
            for bucket_id in torch.unique(bucket_ids[route_active], sorted=True):
                members = torch.nonzero(route_active & (bucket_ids == bucket_id),
                                        as_tuple=False).flatten()
                if members.numel() < 2:
                    skipped += 1
                    continue
                eligible_buckets += 1
                eligible_donors += int(members.numel())
                ordered = members[torch.argsort(donor_ids[batch_index, members], stable=True)]
                original = density[batch_index, destination, ordered]
                moved = original.roll(1)
                reassigned[batch_index, destination, ordered] = moved
                changed += int((moved != original).sum().item())
    return reassigned, {
        "changed_donors": changed,
        "skipped_singleton_buckets": skipped,
        "eligible_buckets": eligible_buckets,
        "eligible_donors": eligible_donors,
    }


def route_endpoint_types(route: str) -> tuple[str, str]:
    """Return explicit ``(donor_type, destination_type)`` labels."""
    try:
        return _ROUTE_ENDPOINT_TYPES[route]
    except KeyError as error:
        raise ValueError(f"Unknown native route type {route!r}.") from error


__all__ = [
    "NATIVE_ROUTE_TYPES",
    "NativeInteractionRouting",
    "density_normalize",
    "normalize_intervention",
    "reassign_equal_measure_densities",
    "route_endpoint_types",
]

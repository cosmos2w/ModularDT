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
    """P-R/P-R2 typed score networks or a P-C destination gain."""

    SCORE_HIDDEN = 32
    SCORE_BOUND = 2.0

    def __init__(self, *, policy: str, pair_width: int, destination_width: int,
                 context_width: int, message_width: int, control_hidden: int | None,
                 initialization_seed: int, views: int = 1) -> None:
        super().__init__()
        if policy not in ("learned_density", "destination_gain"):
            raise ValueError("Native interaction policy must be learned_density or destination_gain.")
        if min(pair_width, destination_width, context_width, message_width) < 1:
            raise ValueError("Native interaction widths must be positive.")
        if policy == "learned_density" and control_hidden is not None:
            raise ValueError("P-R/P-R2 have a fixed score width and no P-C control width.")
        if policy == "destination_gain" and (type(control_hidden) is not int or control_hidden < 1):
            raise ValueError("P-C requires its sealed positive hidden width.")
        if type(views) is not int or views not in (1, 2):
            raise ValueError("Native learned-density routing supports one or two views.")
        if policy == "destination_gain" and views != 1:
            raise ValueError("P-C destination gains use one aggregate view.")
        if views == 2 and message_width != 128:
            raise ValueError("P-R2 requires one existing 128-coordinate message with two 64-coordinate views.")

        # Keep the active architecture contract available to the shared core
        # without registering any buffers or parameters. These values are
        # bound into prepared-state freshness checks, while checkpoint tensor
        # inventories remain unchanged.
        self.pair_width = pair_width
        self.destination_width = destination_width
        self.context_width = context_width
        self.message_width = message_width
        self.initialization_seed = int(initialization_seed)
        self._routing_semantics_generation = 0
        self._policy = policy
        self.control_hidden = control_hidden
        self._views = views
        self._feature_blocks = ((message_width,) if views == 1 else (64, 64))
        self.score_mlps = nn.ModuleDict()
        self.aggregate_modulators = nn.ModuleDict()
        # This private RNG stream must not advance initialization of any
        # inherited P tensor or of later shared/readout modules.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(initialization_seed))
            if policy == "learned_density":
                for route in NATIVE_ROUTE_TYPES:
                    scorer = _score_mlp(pair_width, self.SCORE_HIDDEN)
                    if views == 1:
                        zero_final_projection(scorer)
                    else:
                        # Consume precisely the legacy scalar network's RNG
                        # draws, then replace only its zeroed output projection.
                        # Restoring this point keeps every later route trunk
                        # bit-identical to R1 for the same initialization seed.
                        rng_state = torch.random.get_rng_state()
                        projection = nn.Linear(self.SCORE_HIDDEN, views, bias=False)
                        torch.random.set_rng_state(rng_state)
                        zero_final_projection(nn.Sequential(projection))
                        scorer[-1] = projection
                    self.score_mlps[route] = scorer
            else:
                input_width = destination_width + context_width
                for route in NATIVE_ROUTE_TYPES:
                    self.aggregate_modulators[route] = _gain_mlp(
                        input_width, control_hidden, message_width)
                    zero_final_projection(self.aggregate_modulators[route])

    @property
    def routing_semantics_generation(self) -> int:
        return self._routing_semantics_generation

    def __setattr__(self, name: str, value) -> None:
        if name in ('policy', 'views', 'feature_blocks') and hasattr(
                self, '_routing_semantics_generation'):
            self._set_routing_semantic(name, value)
            return
        super().__setattr__(name, value)

    def _set_routing_semantic(self, name: str, value) -> None:
        private_name = f"_{name}"
        previous = getattr(self, private_name, None)
        if previous is not None:
            changed = type(previous) is not type(value)
            if not changed:
                try:
                    comparison = previous == value
                    changed = (not comparison if type(comparison) is bool
                               else repr(previous) != repr(value))
                except (TypeError, ValueError, RuntimeError):
                    changed = True
            if changed:
                self._routing_semantics_generation += 1
        setattr(self, private_name, value)

    @property
    def policy(self) -> str:
        return self._policy

    @policy.setter
    def policy(self, value: str) -> None:
        self._set_routing_semantic("policy", value)

    @property
    def views(self) -> int:
        return self._views

    @views.setter
    def views(self, value: int) -> None:
        self._set_routing_semantic("views", value)

    @property
    def feature_blocks(self) -> tuple[int, ...]:
        return self._feature_blocks

    @feature_blocks.setter
    def feature_blocks(self, value: tuple[int, ...]) -> None:
        self._set_routing_semantic("feature_blocks", value)

    def score(self, route: str, pair_features: torch.Tensor, *,
              finite_flags: list[tuple[str, torch.Tensor]] | None = None) -> torch.Tensor:
        if self.policy != "learned_density":
            raise ValueError("Only P-R/P-R2 define donor-selective route scores.")
        if route not in NATIVE_ROUTE_TYPES:
            raise ValueError(f"Unknown native route type {route!r}.")
        raw = self.score_mlps[route](pair_features)
        raw_finite = torch.isfinite(raw).all()
        if finite_flags is None:
            if not bool(raw_finite):
                raise FloatingPointError(f"Native {route} score MLP produced non-finite values.")
        else:
            finite_flags.append((f"Native {route} raw score MLP", raw_finite))
        bounded = self.SCORE_BOUND * torch.tanh(raw / self.SCORE_BOUND)
        return bounded.squeeze(-1) if self.views == 1 else bounded

    def gain(self, route: str, destination: torch.Tensor,
             context: torch.Tensor, *,
             finite_flags: list[tuple[str, torch.Tensor]] | None = None) -> torch.Tensor:
        if self.policy != "destination_gain":
            raise ValueError("Only P-C defines destination-only aggregate modulation.")
        if route not in NATIVE_ROUTE_TYPES:
            raise ValueError(f"Unknown native route type {route!r}.")
        context = context[:, None].expand(-1, destination.shape[1], -1)
        raw = self.aggregate_modulators[route](torch.cat((destination, context), -1))
        raw_finite = torch.isfinite(raw).all()
        if finite_flags is None:
            if not bool(raw_finite):
                raise FloatingPointError(f"Native {route} destination-gain MLP produced non-finite values.")
        else:
            finite_flags.append((f"Native {route} raw destination-gain MLP", raw_finite))
        result = torch.tanh(raw)
        if finite_flags is not None:
            finite_flags.append((f"Native {route} destination gain", torch.isfinite(result).all()))
        return result

    def added_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def density_normalize(base_weights: torch.Tensor,
                      scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact-zero-start density and measure-preserving donor weights.

    ``base_weights`` already contains physical measure, presence, self-exclusion
    and the legacy normalization. The expm1 form preserves that actual row mass
    and gives bit-exact ``rho == 1``/``weights == base_weights`` at zero scores.
    """
    if base_weights.ndim < 1:
        raise ValueError("Native route weights need a donor axis.")
    if scores.shape == base_weights.shape:
        # Keep the schema-v1 scalar path's tensor operations and reduction
        # order unchanged so historical P-R behavior remains exact.
        mass = base_weights.sum(-1, keepdim=True)
        safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
        score_delta = torch.expm1(scores)
        mean_delta = (base_weights * score_delta).sum(-1, keepdim=True) / safe_mass
        denominator = 1 + mean_delta
        normalized_density = 1 + (score_delta - mean_delta) / denominator
        rho = torch.where(mass > 0, normalized_density, torch.ones_like(normalized_density))
        weights = base_weights * rho
        return rho, weights
    if scores.ndim != base_weights.ndim + 1 or scores.shape[:-1] != base_weights.shape:
        raise ValueError("Native route scores must match base weights, with an optional final view axis.")
    if scores.shape[-1] not in (1, 2):
        raise ValueError("Native route score view axis must have size one or two.")
    mass = base_weights.sum(-1, keepdim=True)
    safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
    score_delta = torch.expm1(scores)
    base_by_view = base_weights.unsqueeze(-1)
    mean_delta = (base_by_view * score_delta).sum(-2, keepdim=True) / safe_mass.unsqueeze(-1)
    denominator = 1 + mean_delta
    normalized_density = 1 + (score_delta - mean_delta) / denominator
    rho = torch.where(mass.unsqueeze(-1) > 0, normalized_density,
                      torch.ones_like(normalized_density))
    weights = base_by_view * rho
    return rho, weights


def normalize_intervention(value: Mapping | None) -> tuple[str | None, tuple[str, ...]]:
    """Validate a replayable route intervention without mutating caller data."""
    if value is None:
        return None, ()
    if not isinstance(value, Mapping):
        raise TypeError("native_route_intervention must be a mapping or None.")
    if set(value) - {"kind", "route_types", "view"}:
        raise ValueError("Native route intervention has unsupported fields.")
    kind = value.get("kind")
    if kind not in ("density_reset", "average_collapse", "equal_measure_reassignment"):
        raise ValueError("Native route intervention kind is unsupported.")
    if "view" in value:
        view = value["view"]
        if type(view) is not int or view < 0:
            raise ValueError("Native route intervention view must be a nonnegative integer.")
        if kind == "average_collapse":
            raise ValueError("Average-collapse interventions cannot select one view.")
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


def intervention_view(value: Mapping | None) -> int | None:
    """Return the optional single-view selector after intervention validation."""
    if value is None or "view" not in value:
        return None
    return value["view"]


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
    if density.ndim == 4:
        if density.shape[:-1] != base_weights.shape:
            raise ValueError("View-resolved density and base weights must align on their donor axes.")
        per_view = [reassign_equal_measure_densities(
            density[..., view], base_weights, measures, present, donor_ids)
            for view in range(density.shape[-1])]
        receipts = [receipt for _moved, receipt in per_view]
        return torch.stack([moved for moved, _receipt in per_view], dim=-1), {
            key: sum(receipt[key] for receipt in receipts)
            for key in receipts[0]
        }
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
    "intervention_view",
    "normalize_intervention",
    "reassign_equal_measure_densities",
    "route_endpoint_types",
]

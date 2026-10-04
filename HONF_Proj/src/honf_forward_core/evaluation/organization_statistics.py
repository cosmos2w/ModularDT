"""Bounded scalar evidence from actual typed plans and numerical accesses.

These summaries describe the learned computation. Neither positive support,
receiver groups, nor donor concentration identifies physical causality.
"""

import math
import zlib
from contextlib import AbstractContextManager

import numpy as np
import torch

from .typed_work_evidence import project_diagnostic_control


class TypedOrganizationStatistics(AbstractContextManager):
    """Summarize one native wrapper without additional physical reads.

    Distinct receiver actions concatenate source densities and affine projected
    controls after eligibility masking. Coordinates/source slots are not
    action coordinates. Each component is rounded at ``signature_tolerance``;
    at most ``signature_capacity`` complete signatures per phase/route are
    retained transiently. A saturated count is explicitly a lower bound.
    Whole-stream sums/counts remain exact even after signature saturation.
    """

    def __init__(self, backend, *, signature_tolerance=1e-6, signature_capacity=4096):
        if backend.training:
            raise ValueError("Organization statistics are frozen-evaluation only")
        if not np.isfinite(signature_tolerance) or signature_tolerance <= 0 or signature_capacity < 1:
            raise ValueError("Positive finite signature tolerance and capacity required")
        self.backend = backend
        self.tolerance = float(signature_tolerance)
        self.capacity = int(signature_capacity)
        self.phases = {}
        self.routes = {}
        self._signatures = {}
        self._signature_saturated = set()
        self._signature_nonfinite = set()

    def _distinct(self, key, rows):
        if key in self._signature_saturated or not rows.shape[0]:
            return
        scaled = rows.astype(np.float64) / self.tolerance
        if not np.isfinite(scaled).all() or np.any(np.abs(scaled) > np.iinfo(np.int64).max - 1024):
            self._signature_nonfinite.add(key)
            return
        quantized = np.rint(scaled).astype(np.int64)
        signatures = self._signatures.setdefault(key, set())
        # Complete compressed rows avoid digest collisions. They are transient,
        # bounded, and never serialized in the scalar report.
        for row in quantized:
            signature = zlib.compress(row.tobytes(), level=1)
            if signature in signatures:
                continue
            if len(signatures) == self.capacity:
                self._signature_saturated.add(key)
                break
            signatures.add(signature)

    @torch.no_grad()
    def _plan(self, state):
        plan = state["hypergraph_plan"]
        phase = f"P{plan.phase}"
        if phase in self.phases:
            raise ValueError("Expected one native preparation per physical phase")
        routes = {}
        for tau, membership in plan.memberships.items():
            admission = plan.strategy_data.get("typed_admission", {}).get(tau, plan.admission)
            active = admission > 0
            gain = self.backend.control_gain[tau]
            projected = gain(plan.controls[tau].to(gain.weight.dtype))
            if tau == "QE":
                projected = torch.cat((self.backend.control_score(plan.controls[tau].to(gain.weight.dtype)), projected), -1)
            if state.get("hypergraph_intervention") == "control_identity":
                projected = torch.zeros_like(projected)
            signature_rows = torch.cat((membership, projected), -1)[active].detach().cpu().numpy()
            signature_key = f"{phase}/{tau}/groups"
            self._distinct(signature_key, signature_rows)
            donor = {}
            for kind, density in plan.control_memberships.get(tau, {}).items():
                mass = density * plan.source_measures[kind][:, None]
                total = mass.sum(-1)
                distribution = mass / torch.where(total > 0, total, torch.ones_like(total))[..., None]
                valid = active & (total > 0)
                donor[kind] = {
                    "active_source_bearing_groups": int(valid.sum()),
                    "positive_group_source_pairs": int(((density > 0) & active[..., None]).sum()),
                    "hhi_sum": float(distribution.square().sum(-1)[valid].sum()),
                    "maximum_source_fraction": float(distribution[valid].amax()) if bool(valid.any()) else None,
                    "measure": "phase-owned physical source measure times donor density",
                }
            routes[tau] = {
                "allocated_group_slots": int(active.numel()),
                "real_candidate_nodes": int(plan.strategy_data["node_valid"][tau].sum())
                    if "node_valid" in plan.strategy_data else int(active.numel()),
                "active_frontier_groups": int(active.sum()),
                "positive_value_memberships": int(((membership > 0) & active[..., None]).sum()),
                "control_donors": donor,
                "group_projected_square_sum": float(projected[active].double().square().sum()),
                "group_projected_scalar_count": int(projected[active].numel()),
                "prepared_action_input_group_rows": int(active.numel()),
                "prepared_projected_source_moment_rows": int(membership.numel()),
            }
        self.phases[phase] = {"routes": routes, "dependency_provenance": plan.dependency_provenance,
                              "scope": "current organizer preparation; diagnostic reference replay may replace numerical access/actions; frontier is not executed receiver participation"}

    @torch.no_grad()
    def _access(self, plan, receivers, tau, value):
        key = f"P{plan.phase}/{tau}"
        kind = "M" if tau in {"MM", "EM", "QM"} else "E"
        valid = value.diagnostics["pair_valid"]
        projected = (value.projected if hasattr(value, "projected") else
                     project_diagnostic_control(self.backend, value.control, tau,
                         mode=getattr(self.backend, "plan_intervention", "normal")))
        source_mass = plan.source_measures[kind][:, None].to(value.weight.dtype)
        receiver_mass = value.diagnostics.get("receiver_measures")
        if receiver_mass is None:
            receiver_kind = "M" if tau in {"MM", "ME"} else "E" if tau == "EM" else None
            receiver_mass = (plan.source_measures[receiver_kind] if receiver_kind is not None
                             else receivers.new_ones(receivers.shape[:2]))
        actual_mass = source_mass * value.weight
        total = actual_mass.sum(-1, keepdim=True)
        probability = actual_mass / torch.where(total > 0, total, torch.ones_like(total))
        valid_rows = total[..., 0] > 0
        mass_pairs = source_mass * receiver_mass[..., None] * value.support
        mass_count = mass_pairs.sum() * projected.shape[-1]
        projected_sum = (projected.double().square() * mass_pairs[..., None]).sum()
        gain = projected[..., projected.shape[-1] // 2:] if tau == "QE" else projected
        gain_sum = ((1. + torch.tanh(gain.double())).square() * mass_pairs[..., None]).sum()
        gain_count = mass_pairs.sum() * gain.shape[-1]
        # Count every actual receiver row, including separately charged repeated
        # reads. Native support percentage uses eligible pairs, not padding.
        packed = torch.stack([value.weight.new_tensor(receivers.shape[0] * receivers.shape[1]),
            valid.sum(), value.support.sum(), (value.support & (source_mass > 0) & (receiver_mass[..., None] > 0)).sum(), (valid & (source_mass > 0) & (receiver_mass[..., None] > 0)).sum(), valid_rows.sum(), receiver_mass.sum(),
            (value.edge_access > 0).sum(), (value.edge_access * receiver_mass[..., None]).sum(),
            (probability.square().sum(-1) * receiver_mass * valid_rows).sum(),
            (receiver_mass * valid_rows).sum(),
            ((value.density - 1).square() * valid * source_mass * receiver_mass[..., None]).sum(),
            ((value.weight - 1).square() * valid * source_mass * receiver_mass[..., None]).sum(),
            (valid * source_mass * receiver_mass[..., None]).sum(), projected_sum, mass_count,
            gain_sum, gain_count]).detach().cpu().tolist()
        names = ("receiver_rows", "eligible_pairs", "positive_value_pairs", "positive_measure_value_pairs", "positive_measure_eligible_pairs", "source_bearing_receiver_rows",
                 "receiver_measure_sum", "positive_receiver_group_paths", "receiver_group_participation_mass",
                 "receiver_weighted_source_hhi_sum", "source_bearing_receiver_measure_sum",
                 "density_identity_squared_deviation_sum", "normalized_weight_identity_squared_deviation_sum", "eligible_pair_measure_sum",
                 "supported_projected_square_sum", "supported_projected_scalar_measure_sum",
                 "supported_effective_gain_square_sum", "supported_gain_scalar_measure_sum")
        row = self.routes.setdefault(key, {"access_calls": 0, **dict.fromkeys(names, 0.), "executed_work": {}})
        row["access_calls"] += 1
        for name, number in zip(names, packed):
            row[name] += number
        representation = "projected" if hasattr(value, "projected") else "full_control"
        row[f"{representation}_numerical_pair_rows"] = row.get(f"{representation}_numerical_pair_rows", 0) + value.weight.numel()
        row["projected_action_channels"] = projected.shape[-1]
        if "repeated_paths_removed" in value.diagnostics:
            repeated = float(value.diagnostics["repeated_paths_removed"])
        elif (hasattr(value, "projected") and value.edge_access.shape[-1] == plan.memberships[tau].shape[1]
              and self.backend.plan_intervention not in {"full_access", "root_union"}):
            paths = torch.bmm((value.edge_access > 0).float(), (plan.memberships[tau] > 0).float())
            repeated = float((torch.where(valid, paths, torch.zeros_like(paths)) - (valid & (value.density > 0)).to(paths.dtype)).sum())
        else:
            repeated = None
        if repeated is not None:
            row["repeated_far_group_paths_removed"] = row.get("repeated_far_group_paths_removed", 0.) + repeated
        row["repeated_far_group_path_access_calls"] = row.get("repeated_far_group_path_access_calls", 0) + int(repeated is not None)
        if key not in self._signature_saturated:
            signature_valid = valid & (source_mass > 0)
            masked_projected = torch.where(signature_valid[..., None], projected, torch.zeros_like(projected))
            signature_rows = torch.cat((torch.where(signature_valid, value.density, torch.zeros_like(value.density))[..., None],
                                     masked_projected), -1).flatten(2).reshape(value.weight.shape[0] * value.weight.shape[1], value.weight.shape[-1] * (projected.shape[-1] + 1))
            self._distinct(key, signature_rows[receiver_mass.reshape(-1) > 0].detach().cpu().numpy())
        participation = (value.edge_access * receiver_mass[..., None]).sum((0, 1)).detach().cpu().tolist()
        if "per_group_receiver_participation_mass" not in row:
            row["per_group_receiver_participation_mass"] = [0.] * len(participation)
        if len(participation) != len(row["per_group_receiver_participation_mass"]):
            raise ValueError("Native phase/route changed its allocated group axis")
        row["per_group_receiver_participation_mass"] = [a + b for a, b in zip(row["per_group_receiver_participation_mass"], participation)]
        return value

    def __enter__(self):
        access_name = "_numerical_access" if callable(getattr(self.backend, "_numerical_access", None)) else "_access"
        names = ("prepare", access_name, "_ledger")
        self._owned = {name: name in self.backend.__dict__ for name in names}
        self._originals = {name: getattr(self.backend, name) for name in names}
        self._access_routes = {}

        def prepare(*args, **kwargs):
            state = self._originals["prepare"](*args, **kwargs)
            self._plan(state)
            return state

        def access(plan, receivers, tau, *args, **kwargs):
            value = self._originals[access_name](plan, receivers, tau, *args, **kwargs)
            self._access_routes[id(value)] = f"P{plan.phase}/{tau}"
            return self._access(plan, receivers, tau, value)

        def ledger(tau, value, work):
            result = self._originals["_ledger"](tau, value, work)
            route = self._access_routes.pop(id(value))
            measured = {key: float(number) for key, number in result.items()
                        if key.endswith(("executed_rows", "allocated_rows", "fine_calls", "attention_cells"))}
            row = self.routes[route]["executed_work"]
            for name, number in measured.items():
                row[name] = row.get(name, 0.) + number
            return result

        self.backend.prepare = prepare
        setattr(self.backend, access_name, access)
        self.backend._ledger = ledger
        return self

    def __exit__(self, *_args):
        for name, original in self._originals.items():
            if self._owned[name]:
                setattr(self.backend, name, original)
            else:
                delattr(self.backend, name)
        return False

    def summary(self):
        result = {"phases": self.phases, "native_routes": self.routes,
                  "signature_tolerance": self.tolerance, "signature_capacity": self.capacity,
                  "signature_semantics": "complete density and affine projected channels rounded at tolerance; eligible positive-measure source slots and positive-measure receivers",
                  "physical_claim": "computation and donor/action statistics; physical causality unmeasured"}
        for phase, values in self.phases.items():
            for tau, row in values["routes"].items():
                key = f"{phase}/{tau}/groups"
                row["distinct_quantized_group_actions"] = len(self._signatures.get(key, ()))
                row["distinct_group_count_exact"] = key not in self._signature_saturated | self._signature_nonfinite
        for key, row in self.routes.items():
            row["distinct_quantized_receiver_actions"] = len(self._signatures.get(key, ()))
            row["distinct_receiver_count_exact"] = key not in self._signature_saturated | self._signature_nonfinite
            row["distinct_receiver_count_scope"] = "all native positive-measure receiver rows" if row["distinct_receiver_count_exact"] else "lower bound; bounded signature storage or nonfinite signature"
            row["normalized_weight_identity_rms"] = (math.sqrt(row["normalized_weight_identity_squared_deviation_sum"] /
                row["eligible_pair_measure_sum"]) if row["eligible_pair_measure_sum"] else None)
            row["receiver_weighted_source_hhi"] = (row["receiver_weighted_source_hhi_sum"] /
                row["source_bearing_receiver_measure_sum"] if row["source_bearing_receiver_measure_sum"] else None)
            row["source_participation_equivalent"] = (1. / row["receiver_weighted_source_hhi"]
                if row["receiver_weighted_source_hhi"] else None)
            row["supported_projected_action_rms"] = (math.sqrt(row["supported_projected_square_sum"] /
                row["supported_projected_scalar_measure_sum"]) if row["supported_projected_scalar_measure_sum"] else None)
            row["supported_effective_gain_rms"] = (math.sqrt(row["supported_effective_gain_square_sum"] /
                row["supported_gain_scalar_measure_sum"]) if row["supported_gain_scalar_measure_sum"] else None)
            row["executed_receiver_group_count"] = sum(value > 0 for value in row["per_group_receiver_participation_mass"])
            row["receiver_group_count_scope"] = "positive receiver-measure access A; a participating rule may lack an eligible value donor"
            participation = row["per_group_receiver_participation_mass"]
            total = sum(participation)
            squares = sum(value * value for value in participation)
            row["receiver_group_participation_equivalent"] = total * total / squares if squares else None
            row["participation_semantics"] = "source: reciprocal receiver-weighted source HHI; groups: inverse HHI of aggregate receiver-measure participation"
            row["repeated_path_scope"] = "eligible far group paths before source deduplication; mandatory near paths excluded; recorded-access call coverage explicit"
            row["projected_action_semantics"] = "affine score then gain channels" if key.endswith("QE") else "affine gain"
            row["native_support_fraction"] = row["positive_value_pairs"] / row["eligible_pairs"] if row["eligible_pairs"] else None
            row["positive_measure_support_fraction"] = row["positive_measure_value_pairs"] / row["positive_measure_eligible_pairs"] if row["positive_measure_eligible_pairs"] else None
        return result


__all__ = ["TypedOrganizationStatistics"]

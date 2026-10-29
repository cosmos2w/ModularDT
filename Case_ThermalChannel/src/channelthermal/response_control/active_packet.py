"""Thermal-specific matched grouped/direct packet plans for native Dense fields.

The module only builds plans from the current encoded input and the P0 base
module state.  It never reads field targets.  The direct-pair control uses
the same global input context, budget, physical anchors, and typed source
features as the grouped organizer, while its QE policy re-scores each live
native read panel.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    MechanismPlan,
)
from honf_forward_core.interface_fields.budgeted_frontier import (
    BudgetConditionedDirectPairScorer,
    canonical_pair_catalog,
    direct_pair_weights_at_threshold,
    enumerate_frontier_cuts,
    frontier_paths,
    project_direct_pair_budget_by_fraction,
    project_unique_pair_budget,
    split_gates_for_frontier,
)
from honf_forward_core.interface_fields.input_cover_organizer import InputOnlyCoverOrganizer
from honf_forward_core.interface_fields.native_joint_shadow import _detach_inputs

from .contracts import AbsolutePrediction, DesignInput, RoleQuery
from .native import DifferentiableThermalOperator


SUPPORTED_EXTRA_ROUTES = {"MM", "ME"}
THERMAL_ERROR_ROLES = ("fluid_fields", "interface", "solid_temperature")


def _case_scale(encoded: Any, case_index: int) -> torch.Tensor:
    scale = encoded.coordinate_scale
    if scale.ndim == 1:
        return scale
    if scale.ndim == 2:
        return scale[case_index]
    if scale.ndim == 3 and int(scale.shape[1]) == 1:
        return scale[case_index, 0]
    raise ValueError("Thermal coordinate_scale must have shape [d], [B,d], or [B,1,d].")


def _source_features(encoded: Any, route: str, case_index: int) -> torch.Tensor:
    if route in {"MM", "EM", "QM"}:
        return encoded.module_tokens[case_index]
    return encoded.env_tokens[case_index]


def _context_state(encoded: Any, module_state: torch.Tensor) -> Mapping[str, torch.Tensor]:
    """Supply exactly the native P0 inputs accepted by the organizer."""

    return {
        "module_states": module_state,
        "environment_states": encoded.env_tokens,
        "global_state": encoded.global_token,
    }


class _ContextualDirectPairScorer:
    """Add the current case's input-only global token without changing bindings."""

    def __init__(self, scorer: BudgetConditionedDirectPairScorer, global_token: torch.Tensor) -> None:
        self.scorer = scorer
        self.global_token = global_token.detach().clone()

    def __call__(
        self,
        receiver_features: torch.Tensor,
        source_features: torch.Tensor,
        receiver_coordinates: torch.Tensor,
        source_coordinates: torch.Tensor,
        *,
        mechanism: str,
        budget_fraction: float | torch.Tensor,
    ) -> torch.Tensor:
        global_token = self.global_token.to(
            device=receiver_features.device, dtype=receiver_features.dtype
        ).reshape(1, -1)
        receiver_context = global_token.expand(receiver_features.shape[0], -1)
        source_context = global_token.to(
            device=source_features.device, dtype=source_features.dtype
        ).expand(source_features.shape[0], -1)
        scorer_inputs = (
            torch.cat((receiver_features.detach(), receiver_context), dim=-1),
            torch.cat((source_features.detach(), source_context), dim=-1),
            receiver_coordinates.detach(),
            source_coordinates.detach(),
        )

        def score(
            receiver_input: torch.Tensor,
            source_input: torch.Tensor,
            receiver_xy: torch.Tensor,
            source_xy: torch.Tensor,
        ) -> torch.Tensor:
            return self.scorer(
                receiver_input,
                source_input,
                receiver_xy,
                source_xy,
                mechanism=mechanism,
                budget_fraction=budget_fraction,
            )

        if torch.is_grad_enabled() and any(parameter.requires_grad for parameter in self.scorer.parameters()):
            # Recompute all pair-feature/MLP chunks in backward. Checkpointing
            # only the final MLP would still retain the large [Q,S,D] input.
            return checkpoint(score, *scorer_inputs, use_reentrant=False)
        return score(*scorer_inputs)


@dataclass(frozen=True)
class RouteWorkRecord:
    mechanism: str
    requested_fraction: float
    requested_work: float
    achieved_work: float
    full_access_work: float
    selected_unique_pairs: int
    full_unique_pairs: int
    sparse_success: bool
    executor: str

    @property
    def achieved_fraction(self) -> float:
        if self.full_access_work <= 0.0:
            return 0.0
        return self.achieved_work / self.full_access_work


class ThermalCoverPlanBuilder:
    """Build grouped G or independently scored direct-pair P plans.

    The budget map must contain QE and exactly one of MM/ME. Missing route
    keys retain the core's explicit full-access bypass semantics and are
    included in ``last_records``. By default the deepest bounded depth-three
    cut is used during Stage A; Stage C can provide a case-dependent selector.
    """

    def __init__(
        self,
        *,
        core: Any,
        budget_fractions: Mapping[str, float],
        mode: str,
        organizer: InputOnlyCoverOrganizer | None = None,
        direct_scorer: BudgetConditionedDirectPairScorer | None = None,
        extra_route: str,
        temperature: float = 1.0,
        max_depth: int = 3,
        frontier_selector: Any | None = None,
    ) -> None:
        normalized = {str(key).upper(): float(value) for key, value in budget_fractions.items()}
        if set(normalized) != {"QE", str(extra_route).upper()}:
            raise ValueError("Thermal packet budgets must contain QE plus exactly one MM or ME route.")
        if str(extra_route).upper() not in SUPPORTED_EXTRA_ROUTES:
            raise ValueError("Thermal's bounded matched extra route must be MM or ME.")
        if any(not 0.0 < value <= 1.0 for value in normalized.values()):
            raise ValueError("Thermal route budgets must lie in (0,1].")
        if mode not in {"G", "P"}:
            raise ValueError("mode must be G (grouped organizer) or P (direct-pair control).")
        if temperature <= 0.0 or max_depth < 0:
            raise ValueError("temperature and max_depth must be positive/nonnegative.")
        if mode == "G" and organizer is None:
            raise ValueError("G mode requires an InputOnlyCoverOrganizer.")
        if mode == "P" and direct_scorer is None:
            raise ValueError("P mode requires a BudgetConditionedDirectPairScorer.")
        self.core = core
        self.budget_fractions = normalized
        self.mode = mode
        self.organizer = organizer
        self.direct_scorer = direct_scorer
        self.extra_route = str(extra_route).upper()
        self.temperature = float(temperature)
        self.max_depth = int(max_depth)
        self.frontier_selector = frontier_selector
        self.last_records: tuple[dict[str, Any], ...] = ()
        self.last_plans: tuple[MechanismPlan, ...] = ()
        self.last_scores: tuple[Any, ...] = ()
        self.last_encoded: Any | None = None
        self.last_trees: tuple[Any, ...] = ()

    @property
    def exact_full_access(self) -> bool:
        """Whether every declared route is the full-access warm/mix endpoint."""

        return all(value == 1.0 for value in self.budget_fractions.values())

    def _frontier(self, tree: Any, case_index: int, encoded: Any, module_state: torch.Tensor) -> tuple[int, ...]:
        if self.frontier_selector is not None:
            cut = tuple(int(node) for node in self.frontier_selector(case_index, encoded, module_state, tree))
            if cut not in enumerate_frontier_cuts(tree, max_depth=self.max_depth):
                raise ValueError("frontier_selector returned a cut outside the bounded depth-three family.")
            return cut
        return enumerate_frontier_cuts(tree, max_depth=self.max_depth)[-1]

    def __call__(
        self,
        encoded: Any,
        base_module_state: torch.Tensor,
        trees: Sequence[Any],
    ) -> tuple[MechanismPlan, ...]:
        if len(trees) != int(encoded.module_present.shape[0]):
            raise ValueError("one Thermal receiver tree is required per encoded case.")
        detached_encoded = _detach_inputs(encoded)
        detached_state = _detach_inputs(base_module_state)
        self.last_encoded = detached_encoded
        self.last_trees = tuple(trees)
        cuts = tuple(
            self._frontier(tree, case, detached_encoded, detached_state)
            for case, tree in enumerate(trees)
        )
        if self.mode == "G":
            assert self.organizer is not None
            scores = self.organizer.score_cases(
                detached_encoded,
                _context_state(detached_encoded, detached_state),
                trees,
                budgets=self.budget_fractions,
            )
            self.last_scores = tuple(scores)
            plans = self.organizer.plans_from_scores(
                scores,
                detached_encoded,
                trees,
                hard=self.hard,
                straight_through_hard=False,
                frontier_cuts=cuts,
                budget_fractions=self.budget_fractions,
            )
            self.last_records = tuple(
                self._grouped_records(case, scores[case], plan, tree, detached_encoded, cuts[case])
                for case, (plan, tree) in enumerate(zip(plans, trees, strict=True))
            )
            self.last_plans = tuple(plans)
            return plans

        assert self.direct_scorer is not None
        self.last_scores = ()
        plans_out: list[MechanismPlan] = []
        records: list[dict[str, Any]] = []
        for case, tree in enumerate(trees):
            module_present = detached_encoded.module_present[case]
            plan = MechanismPlan(
                tree=tree,
                split_gates=split_gates_for_frontier(tree, cuts[case], max_depth=self.max_depth),
                module_present=module_present,
                environment_count=int(detached_encoded.env_coords.shape[1]),
            )
            contextual = _ContextualDirectPairScorer(
                self.direct_scorer, detached_encoded.global_token[case]
            )
            case_records: dict[str, Any] = {
                "mode": "P",
                "frontier": list(cuts[case]),
                "routes": {},
                "direct_scorer_factorized_first_layer": bool(
                    getattr(self.direct_scorer, "factorized_first_layer", False)
                ),
            }
            for mechanism, fraction in self.budget_fractions.items():
                catalog = canonical_pair_catalog(
                    detached_encoded, tree, mechanism, case_index=case
                )
                source_coordinates = (
                    detached_encoded.module_centers[case]
                    if mechanism == "MM"
                    else detached_encoded.env_coords[case]
                )
                source_features = _source_features(detached_encoded, mechanism, case)
                # The grouped tree anchor panel and every native preparation panel
                # use this checkpoint's Fourier coordinate scale and features.
                receiver_coordinates = catalog.receiver_coordinates
                receiver_features = self._receiver_features(detached_encoded, receiver_coordinates, case)
                if self.hard:
                    # Hard-plan scores only determine detached routing decisions;
                    # the soft shadow owns route gradients for the P arm.
                    with torch.no_grad():
                        route_scores = contextual(
                            receiver_features,
                            source_features,
                            receiver_coordinates,
                            source_coordinates,
                            mechanism=mechanism,
                            budget_fraction=fraction,
                        )
                else:
                    route_scores = contextual(
                        receiver_features,
                        source_features,
                        receiver_coordinates,
                        source_coordinates,
                        mechanism=mechanism,
                        budget_fraction=fraction,
                    )
                projection = project_direct_pair_budget_by_fraction(
                    route_scores.detach(),
                    budget_fraction=fraction,
                    eligible_pairs=catalog.pair_validity,
                    receiver_weights=catalog.receiver_weights,
                )
                thresholds = direct_pair_weights_at_threshold(
                    route_scores,
                    projection,
                    eligible_pairs=catalog.pair_validity,
                    temperature=self.temperature,
                )
                if mechanism == "QE":
                    plan = plan.with_direct_pair_policy(
                        "QE",
                        contextual,
                        source_coordinates=source_coordinates,
                        source_features=source_features,
                        source_validity=catalog.source_validity,
                        hard_threshold=thresholds.hard_threshold,
                        soft_threshold=thresholds.soft_threshold,
                        budget_fraction=fraction,
                        hard=self.hard,
                        temperature=self.temperature,
                    )
                    # QE work is calibrated on canonical tree anchors. The live
                    # reader re-scores exact current P0/P1/P2 query chunks.
                    executor = "dynamic_direct_pair_policy_per_native_read"
                else:
                    weights = thresholds.hard_weights if self.hard else thresholds.soft_weights
                    plan = plan.with_direct_pair_access(
                        mechanism,
                        receiver_coordinates,
                        weights,
                        receiver_validity=catalog.receiver_validity,
                    )
                    executor = "exact_static_direct_pairs"
                case_records["routes"][mechanism] = RouteWorkRecord(
                    mechanism=mechanism,
                    requested_fraction=fraction,
                    requested_work=projection.requested_work,
                    achieved_work=projection.achieved_work,
                    full_access_work=projection.full_access_work,
                    selected_unique_pairs=projection.selected_unique_pairs,
                    full_unique_pairs=projection.full_unique_pairs,
                    sparse_success=projection.sparse_success,
                    executor=executor,
                )
            case_records["full_access_bypass_routes"] = list(plan.explicit_bypass_keys)
            case_records.update(
                self._realized_action_evidence(
                    plan, detached_encoded, tree, case, cuts[case]
                )
            )
            self._add_total_route_work(case_records, detached_encoded, tree, case)
            plans_out.append(plan)
            records.append(case_records)
        self.last_records = tuple(records)
        self.last_plans = tuple(plans_out)
        return tuple(plans_out)

    @property
    def hard(self) -> bool:
        """The caller sets this for a fresh hard-value or soft-shadow pass."""

        return bool(getattr(self, "_hard", True))

    @hard.setter
    def hard(self, value: bool) -> None:
        self._hard = bool(value)

    def _receiver_features(self, encoded: Any, coordinates: torch.Tensor, case_index: int) -> torch.Tensor:
        scale = _case_scale(encoded, case_index).to(device=coordinates.device, dtype=coordinates.dtype)
        result = self.core.receiver_fourier((coordinates / scale)[None])
        return result[0].detach()

    def _grouped_records(
        self,
        case: int,
        scores: Any,
        plan: MechanismPlan,
        tree: Any,
        encoded: Any,
        frontier: tuple[int, ...],
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "mode": "G",
            "frontier": list(frontier),
            "full_access_bypass_routes": list(plan.explicit_bypass_keys),
            "routes": {},
        }
        for mechanism, fraction in self.budget_fractions.items():
            catalog = canonical_pair_catalog(encoded, tree, mechanism, case_index=case)
            projection = project_unique_pair_budget(
                scores.mechanism_logits[mechanism],
                tree,
                plan.split_gates.detach(),
                budget_fraction=fraction,
                source_validity=catalog.source_validity,
                receiver_validity=catalog.receiver_validity,
                pair_validity=catalog.pair_validity,
                receiver_weights=catalog.receiver_weights,
                receiver_coordinates=catalog.receiver_coordinates,
            )
            record["routes"][mechanism] = RouteWorkRecord(
                mechanism=mechanism,
                requested_fraction=fraction,
                requested_work=projection.requested_work,
                achieved_work=projection.achieved_work,
                full_access_work=projection.full_access_work,
                selected_unique_pairs=projection.selected_unique_pairs,
                full_unique_pairs=projection.full_unique_pairs,
                sparse_success=projection.sparse_success,
                executor="grouped_frontier_packets",
            )
        record.update(
            self._realized_action_evidence(plan, encoded, tree, case, frontier)
        )
        self._add_total_route_work(record, encoded, tree, case)
        return record

    def _realized_action_evidence(
        self,
        plan: MechanismPlan,
        encoded: Any,
        tree: Any,
        case: int,
        frontier: tuple[int, ...],
    ) -> dict[str, Any]:
        """Record the realized cut and typed source-mask evidence for this plan.

        The cut K used by the action selector is based on the union of typed
        source-permission signatures, not the sum of route counts. Packed masks
        retain enough information to measure support turnover across weights.
        """

        mechanisms = (self.extra_route, "QE")
        route_masks: dict[str, dict[str, Any]] = {}
        for mechanism in mechanisms:
            catalog = canonical_pair_catalog(encoded, tree, mechanism, case_index=case)
            direct_policy = plan.direct_pair_policy_for(mechanism)
            direct_access = plan.direct_pair_access_for(mechanism)
            direct = direct_policy is not None or direct_access is not None
            source_valid = catalog.source_validity.to(dtype=torch.bool)
            valid_cpu = source_valid.detach().to(device="cpu", dtype=torch.uint8).numpy()
            valid_bits = np.packbits(valid_cpu, bitorder="little").tobytes()
            if direct_policy is not None:
                # Runtime policies already score the actual native read panels.
                # Do not score a second, potentially much larger canonical panel
                # just to build a training diagnostic.
                route_masks[mechanism] = {
                    "permission_scope": "dynamic_direct_policy_runtime_not_rematerialized",
                    "receiver_axis": "canonical_native_receiver_queries",
                    "receiver_count": int(catalog.receiver_coordinates.shape[0]),
                    "source_index_space": "canonical_pair_catalog_source_order",
                    "source_count": int(source_valid.numel()),
                    "valid_source_count": int(source_valid.sum().detach().cpu()),
                    "source_validity_bits_little_endian_hex": valid_bits.hex(),
                    "support_materialized": False,
                    "support_note": "Runtime direct policy masks are not recomputed for logging; route work is recorded from the live execution.",
                }
                continue
            if direct_access is not None:
                receiver_coordinates = catalog.receiver_coordinates
                if direct_access.receiver_coordinates.shape != receiver_coordinates.shape or not torch.equal(
                    direct_access.receiver_coordinates.detach(), receiver_coordinates.detach()
                ):
                    raise ValueError(
                        f"Realized {mechanism} direct access does not match the canonical receiver axis."
                    )
                permission = direct_access.weights
                receiver_axis = "canonical_native_receiver_queries"
            else:
                permission = plan.permission_matrix(mechanism)[
                    list(frontier)
                ]
                receiver_axis = "realized_frontier_nodes"
            if permission.ndim != 2:
                raise ValueError(f"Realized {mechanism} permission must be a receiver/source matrix.")
            if direct and permission.shape[0] != catalog.receiver_coordinates.shape[0]:
                raise ValueError(
                    f"Realized {mechanism} direct-access rows do not align with canonical native queries."
                )
            if not direct and permission.shape[0] != len(frontier):
                raise ValueError(
                    f"Realized {mechanism} permission rows do not align with its {receiver_axis} axis."
                )
            if permission.shape[1] != source_valid.numel():
                raise ValueError(
                    f"Realized {mechanism} permission sources do not align with canonical validity."
                )
            hard_support = (permission.detach() >= 0.5) & source_valid[None]
            support_cpu = hard_support.to(device="cpu", dtype=torch.uint8).numpy()
            packed_rows = [
                np.packbits(row, bitorder="little").tobytes()
                for row in support_cpu
            ]
            valid_cpu = source_valid.detach().to(device="cpu", dtype=torch.uint8).numpy()
            valid_bits = np.packbits(valid_cpu, bitorder="little").tobytes()
            packed_matrix = np.packbits(support_cpu.reshape(-1), bitorder="little").tobytes()
            matrix_shape_bytes = np.asarray(support_cpu.shape, dtype=np.int64).tobytes()
            route_masks[mechanism] = {
                "permission_scope": "direct_pair_at_canonical_native_receiver_queries" if direct else "realized_plan_at_frontier_nodes",
                "receiver_axis": receiver_axis,
                "receiver_count": int(permission.shape[0]),
                "hard_support_threshold": 0.5,
                "support_materialized": True,
                "source_index_space": "canonical_pair_catalog_source_order",
                "source_count": int(source_valid.numel()),
                "valid_source_count": int(source_valid.sum().detach().cpu()),
                "source_validity_bits_little_endian_hex": valid_bits.hex(),
                "permission_matrix_shape": [int(value) for value in support_cpu.shape],
                "permission_matrix_sha256": hashlib.sha256(matrix_shape_bytes + packed_matrix).hexdigest(),
                "active_receiver_count_by_source": support_cpu.sum(axis=0, dtype=np.int64).tolist(),
            }
            if not direct:
                route_masks[mechanism]["permission_bits_by_receiver_little_endian_hex"] = [
                    row.hex() for row in packed_rows
                ]
                route_masks[mechanism]["permission_mask_sha256_by_receiver"] = [
                    hashlib.sha256(row).hexdigest() for row in packed_rows
                ]

        signatures: list[bytes] = []
        packet_aligned = all(
            route_masks[mechanism]["receiver_axis"] == "realized_frontier_nodes"
            and route_masks[mechanism]["receiver_count"] == len(frontier)
            for mechanism in mechanisms
        )
        signature_nonempty: list[bool] = []
        if packet_aligned:
            for packet_index in range(len(frontier)):
                combined = bytearray()
                has_source = False
                for mechanism in mechanisms:
                    support = bytes.fromhex(
                        route_masks[mechanism]["permission_bits_by_receiver_little_endian_hex"][packet_index]
                    )
                    has_source = has_source or any(support)
                    combined.extend(len(support).to_bytes(4, "little"))
                    combined.extend(support)
                signatures.append(bytes(combined))
                signature_nonempty.append(has_source)
        else:
            signatures = []
            signature_nonempty = []
        first_for_signature: dict[bytes, int] = {}
        nonempty: set[bytes] = set()
        collapsed: list[dict[str, Any]] = []
        for index, signature in enumerate(signatures):
            if signature_nonempty[index]:
                nonempty.add(signature)
            if not signature_nonempty[index]:
                collapsed.append({
                    "frontier_index": index,
                    "frontier_node_id": int(frontier[index]),
                    "joint_support_signature_hex": signature.hex(),
                    "reason": "empty_source_support",
                })
            elif signature in first_for_signature:
                collapsed.append({
                    "frontier_index": index,
                    "frontier_node_id": int(frontier[index]),
                    "joint_support_signature_hex": signature.hex(),
                    "collapsed_into_frontier_index": first_for_signature[signature],
                    "reason": "duplicate_joint_source_support",
                })
            else:
                first_for_signature[signature] = index
        paths = frontier_paths(tree, frontier, max_depth=self.max_depth)
        return {
            "execution_hardness": "hard" if self.hard else "soft",
            "frontier_paths": list(paths),
            "raw_frontier_k": len(frontier),
            "nonredundant_k": len(nonempty) if packet_aligned else None,
            "nonredundant_k_status": "measured_from_realized_typed_source_support" if packet_aligned else "not_applicable_to_direct_pair_receiver_axis",
            "joint_support_mechanisms": list(mechanisms),
            "joint_support_signature_hex_by_frontier": [
                signature.hex() for signature in signatures
            ],
            "collapsed_cut_rows": collapsed,
            "source_mask_evidence": route_masks,
        }

    @staticmethod
    def _add_total_route_work(
        record: dict[str, Any], encoded: Any, tree: Any, case: int
    ) -> None:
        """Record truthful all-mechanism canonical work, including full bypasses."""

        total_achieved = 0.0
        total_full = 0.0
        achieved_pairs = 0
        full_pairs = 0
        route_totals: dict[str, dict[str, Any]] = {}
        for mechanism in ("MM", "ME", "EM", "QM", "QE"):
            catalog = canonical_pair_catalog(encoded, tree, mechanism, case_index=case)
            weights = catalog.receiver_weights.to(
                device=catalog.pair_validity.device, dtype=torch.float64
            )[:, None]
            full_work = float(
                (catalog.pair_validity.to(torch.float64) * weights).sum().detach().cpu()
            )
            full_unique = int(catalog.pair_validity.sum().detach().cpu())
            route = record["routes"].get(mechanism)
            if route is None:
                achieved_work = full_work
                selected_unique = full_unique
                executor = "explicit_full_access_bypass"
            else:
                achieved_work = float(route.achieved_work)
                selected_unique = int(route.selected_unique_pairs)
                executor = str(route.executor)
            route_totals[mechanism] = {
                "achieved_work": achieved_work,
                "full_access_work": full_work,
                "selected_unique_pairs": selected_unique,
                "full_unique_pairs": full_unique,
                "work_fraction": achieved_work / full_work if full_work > 0.0 else 0.0,
                "executor": executor,
            }
            total_achieved += achieved_work
            total_full += full_work
            achieved_pairs += selected_unique
            full_pairs += full_unique
        record["all_mechanism_route_work"] = route_totals
        record["all_mechanisms_total"] = {
            "achieved_work": total_achieved,
            "full_access_work": total_full,
            "selected_unique_pairs": achieved_pairs,
            "full_unique_pairs": full_pairs,
            "work_fraction": total_achieved / total_full if total_full > 0.0 else 0.0,
        }


class ThermalHardValueSoftOperator:
    """Pair full Thermal role predictions with isolated G/P scorer gradients."""

    def __init__(
        self,
        operator: DifferentiableThermalOperator,
        hard_plan_builder: ThermalCoverPlanBuilder,
        soft_plan_builder: ThermalCoverPlanBuilder,
    ) -> None:
        if hard_plan_builder.mode != soft_plan_builder.mode:
            raise ValueError("hard and soft branches must use the same G or P route family.")
        if hard_plan_builder.budget_fractions != soft_plan_builder.budget_fractions:
            raise ValueError("hard and soft branches must use identical physical budget fractions.")
        if hard_plan_builder.extra_route != soft_plan_builder.extra_route:
            raise ValueError("hard and soft branches must use the same additional typed route.")
        self.operator = operator
        self.hard_plan_builder = hard_plan_builder
        self.soft_plan_builder = soft_plan_builder
        self.last_hard: AbsolutePrediction | None = None
        self.last_soft: AbsolutePrediction | None = None

    def __call__(
        self,
        design: DesignInput,
        context: Mapping[str, Any],
        role_queries: Mapping[str, RoleQuery],
        *,
        soft_shadow_scale: float = 1.0,
    ) -> AbsolutePrediction:
        self.hard_plan_builder.hard = True
        self.soft_plan_builder.hard = False
        shadow_scale = float(soft_shadow_scale)
        if not np.isfinite(shadow_scale) or shadow_scale < 0.0:
            raise ValueError("soft_shadow_scale must be finite and nonnegative.")
        if self.hard_plan_builder.exact_full_access and self.soft_plan_builder.exact_full_access:
            # The budget-one warm/mix endpoint is the untouched checkpoint
            # Dense path. Do not build trees or route through a plan that only
            # happens to grant all pairs.
            self.last_hard = self.operator(design, context, role_queries)
            self.last_soft = None
            return self.last_hard
        self.last_hard = self.operator(
            design,
            context,
            role_queries,
            cover_plan_builder=self.hard_plan_builder,
            detach_model_parameters=False,
        )
        if shadow_scale == 0.0:
            self.last_soft = None
            return self.last_hard
        self.last_soft = self.operator(
            design,
            context,
            role_queries,
            cover_plan_builder=self.soft_plan_builder,
            detach_model_parameters=True,
        )
        if set(self.last_hard.role_values) != set(THERMAL_ERROR_ROLES):
            raise RuntimeError("The hard Thermal forward did not produce all three physical error roles.")
        if set(self.last_soft.role_values) != set(THERMAL_ERROR_ROLES):
            raise RuntimeError("The soft Thermal shadow did not produce all three physical error roles.")
        values: dict[str, torch.Tensor] = {}
        for name in THERMAL_ERROR_ROLES:
            hard = self.last_hard.role_values[name]
            soft = self.last_soft.role_values[name]
            if hard.shape != soft.shape:
                raise ValueError(f"Hard and soft {name} outputs have different physical panels.")
            values[name] = hard + shadow_scale * (soft - soft.detach())
        return AbsolutePrediction(
            role_values=values,
            receiver_world_xy=self.last_hard.receiver_world_xy,
        )


def make_direct_scorer(
    core: Any,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    hidden_dim: int = 96,
    factorized_first_layer: bool = False,
) -> BudgetConditionedDirectPairScorer:
    """Size the P scorer from this checkpoint's Fourier and token widths."""

    receiver_features = core.receiver_fourier(
        torch.zeros((1, 1, int(core.config.spatial_dim)), device=device, dtype=dtype)
    )
    token_dim = int(core.config.hidden_dim)
    return BudgetConditionedDirectPairScorer(
        receiver_feature_dim=int(receiver_features.shape[-1]) + token_dim,
        source_feature_dim=2 * token_dim,
        hidden_dim=hidden_dim,
        factorized_first_layer=factorized_first_layer,
    ).to(device=device, dtype=dtype)


__all__ = [
    "RouteWorkRecord",
    "SUPPORTED_EXTRA_ROUTES",
    "THERMAL_ERROR_ROLES",
    "ThermalCoverPlanBuilder",
    "ThermalHardValueSoftOperator",
    "make_direct_scorer",
]

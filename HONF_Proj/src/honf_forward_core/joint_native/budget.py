"""Typed exact/relaxed work, balanced saved cadence and serialized duals."""

from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch import Tensor, nn

from ..native_retrieval.budget_state import BudgetDualState
from ..native_retrieval.budgeted_graph import ExpectedWork
from ..native_retrieval.contracts import _require
from .contracts import JointNativeDirectory, JointReadPlan, TypedNativeRecords


def _work(support_probability: Tensor, floor_probability: Tensor, present: Tensor,
          query_mask: Tensor) -> ExpectedWork:
    """Keep query-weighted numerators so request tiles aggregate before cases."""
    probability = support_probability * present[:, None] * query_mask[..., None]
    floor = floor_probability * present[:, None] * query_mask[..., None]
    expected_rows, floor_rows = probability.sum(-1), floor.sum(-1)
    possible = present.sum(-1).to(probability)[:, None] * query_mask
    removable = possible - floor_rows
    retained = (expected_rows - floor_rows).clamp_min(0)
    query_retention = torch.where(removable > 0, retained / removable.clamp_min(1e-20), torch.zeros_like(removable))
    case_expected, case_floor = expected_rows.sum(1), floor_rows.sum(1)
    case_possible, case_removable = possible.sum(1), removable.sum(1)
    case_retention = torch.where(case_removable > 0, (case_expected-case_floor)/case_removable.clamp_min(1e-20),
                                 torch.zeros_like(case_removable))
    case_fraction = torch.where(case_possible > 0, case_expected/case_possible.clamp_min(1), torch.zeros_like(case_possible))
    return ExpectedWork(probability, expected_rows, floor_rows, removable, query_retention, case_retention,
                        case_fraction, case_expected, case_possible, case_floor, case_removable)


def expected_typed_work(directory: JointNativeDirectory, dispatch_probabilities: Tensor,
                        protected_mask: Tensor, *, query_mask=None) -> dict[str, ExpectedWork]:
    """Exact unit-noise work; scaled/deterministic callers must label relaxation.

    The floor is the seed-weighted actual self/cross/protection union, rather
    than max(1, protection count). Membership and dispatch must be independent
    conditional on the prepared target-free scene, with dispatch before E reads.
    """
    directory.assert_current()
    batch, queries, seeds = dispatch_probabilities.shape
    if protected_mask.shape != (batch, queries, seeds) or protected_mask.dtype != torch.bool:
        raise ValueError("Expected typed work requires boolean [B,Q,M+E] protections.")
    if seeds != directory.records.ids.shape[-1]:
        raise ValueError("Dispatch probabilities must follow all native seeds.")
    if query_mask is None:
        query_mask = torch.ones((batch, queries), device=dispatch_probabilities.device, dtype=torch.bool)
    _require(torch.isfinite(dispatch_probabilities) & (dispatch_probabilities >= 0), "Invalid seed probabilities.")
    expected_sum = (directory.records.present.any(-1)[:, None] & query_mask).to(dispatch_probabilities)
    _require(torch.isclose(dispatch_probabilities.sum(-1), expected_sum, atol=1e-5, rtol=1e-5),
             "Dispatch probabilities must sum to one for each nonempty valid query.")
    _require(~protected_mask | directory.records.eligible[:, None], "Protections require eligible records.")
    membership = torch.bmm(dispatch_probabilities, directory.membership_probabilities)
    floor_membership = torch.bmm(dispatch_probabilities, directory.mandatory_membership.to(dispatch_probabilities))
    probability = torch.where(protected_mask, torch.ones_like(membership), membership)
    floor = torch.where(protected_mask, torch.ones_like(floor_membership), floor_membership)
    m = directory.records.module_count
    return {name: _work(probability[..., sl], floor[..., sl], directory.records.eligible[:, sl], query_mask)
            for name, sl in (("M", slice(None, m)), ("E", slice(m, None)))}


def direct_typed_work(plan: JointReadPlan, records: TypedNativeRecords, *, query_mask=None,
                      observed=False) -> dict[str, ExpectedWork]:
    if query_mask is None:
        query_mask = torch.ones(plan.hard_support.shape[:2], device=plan.hard_support.device, dtype=torch.bool)
    probability = plan.hard_support.to(plan.value_gates) if observed else plan.probabilities
    if probability is None:
        raise ValueError("Direct work requires independently sampled typed probabilities.")
    floor = plan.mandatory_support.to(probability)
    m = records.module_count
    return {name: _work(probability[..., sl], floor[..., sl], records.eligible[:, sl], query_mask)
            for name, sl in (("M", slice(None, m)), ("E", slice(m, None)))}


def observed_typed_work(plan: JointReadPlan, records: TypedNativeRecords, *, query_mask=None):
    if query_mask is None:
        query_mask = torch.ones(plan.hard_support.shape[:2], device=plan.hard_support.device, dtype=torch.bool)
    probability = plan.hard_support.to(plan.value_gates)
    floor = plan.mandatory_support.to(probability)
    m = records.module_count
    return {name: _work(probability[..., sl], floor[..., sl], records.eligible[:, sl], query_mask)
            for name, sl in (("M", slice(None, m)), ("E", slice(m, None)))}


def aggregate_case_constraints(entries: list[tuple[dict, dict]], target_M: float, target_E: float):
    """Sum all objective requests per case before equal-case macro reduction.

    Each entry contains (differentiable expected work, controller work). Entries
    must have the same case axis, including auxiliary and response queries.
    Returned numerators/denominators compose across microbatches without hinging
    each case or microbatch independently.
    """
    if not entries:
        return {}
    output = {}
    for name, target in (("M", target_M), ("E", target_E)):
        expected = sum((entry[0][name].case_expected_rows for entry in entries))
        floor = sum((entry[0][name].case_floor_rows for entry in entries))
        possible = sum((entry[0][name].case_possible_rows for entry in entries))
        observed = sum((entry[1][name].case_expected_rows for entry in entries))
        observed_floor = sum((entry[1][name].case_floor_rows for entry in entries))
        target_rows = floor + target * (possible-floor)
        observed_target = observed_floor + target * (possible-observed_floor)
        eligible_case = possible > 0
        residual = torch.where(eligible_case, (expected-target_rows)/possible.clamp_min(1), torch.zeros_like(possible))
        controller = torch.where(eligible_case, (observed-observed_target)/possible.clamp_min(1), torch.zeros_like(possible))
        output[name] = {"residual_numerator": residual.sum(), "denominator": eligible_case.sum(),
                        "controller_numerator": controller.detach().sum()}
    return output


class JointBudgetSchedule:
    schema = "joint_native_budget_schedule_v1"
    dual_update_every = 10

    @staticmethod
    def state_at_epoch(epoch: int, macro_index_zero_based: int | None = None, *, sparse_start_epoch=3001):
        age = int(epoch) - int(sparse_start_epoch) + 1
        if age < 0:
            raise ValueError("Sparse schedule cannot precede the shared warm-up boundary.")
        if macro_index_zero_based is not None and macro_index_zero_based < 0:
            raise ValueError("Macro index must be zero-based and nonnegative.")
        if age == 0:
            phase, target_M, target_E, temperature, noise = "all_open_boundary", 1., 1., 1., 1.
        elif age <= 200:
            progress = (age-1)/199
            phase, target_M, target_E = "stochastic_dense_router", .98-.03*progress, .98-.08*progress
            temperature, noise = 1.-.5*progress, 1.
        elif age <= 1000:
            progress = (age-201)/799
            phase, target_M, target_E = "balanced_stochastic_router", .95-.10*progress, .90-.30*progress
            temperature, noise = .5, 1.-progress
        else:
            phase, target_M, target_E, temperature, noise = "deterministic_hard", .85, .60, .5, 0.
        exact = 1 <= age <= 200
        dense = None if macro_index_zero_based is None else (age <= 200 or
                ((age-1+int(macro_index_zero_based)) % 4 == 0))
        return {"schema": JointBudgetSchedule.schema, "absolute_epoch": int(epoch), "sparse_age": age,
                "sparse_start_epoch": int(sparse_start_epoch), "phase": phase,
                "target_M": target_M, "target_E": target_E, "temperature": temperature,
                "noise_scale": noise, "deterministic": age == 0 or age > 1000 or noise == 0,
                "dense_router_step": dense, "stochastic_expectation_exact": exact,
                "budget_controller_source": "unit_noise_expectation" if exact else "observed_hard_work"}


class TypedWorkDual(BudgetDualState):
    """Round9 state schema reusing the established augmented-dual arithmetic."""

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.register_buffer("observed_macro_count",torch.tensor(0,dtype=torch.long))
        self.register_buffer("at_cap_macro_count",torch.tensor(0,dtype=torch.long))

    def get_extra_state(self):
        return {"rng_identity": self.rng_identity, "schema": JointBudgetSchedule.schema}

    def set_extra_state(self, state):
        if state.get("schema") != JointBudgetSchedule.schema:
            raise ValueError("Joint dual checkpoint schema mismatch.")
        self.rng_identity = state["rng_identity"]

    @property
    def phase(self):
        return {0:"uninitialized", 1:"all_open_boundary", 2:"stochastic_dense_router",
                3:"balanced_stochastic_router", 4:"deterministic_hard"}.get(int(self.phase_code),"unknown")

    @torch.no_grad()
    def observe_residual(self, residual, macro_step):
        if int(macro_step) != int(self.macro_step) + 1:
            raise ValueError("Joint dual observes each successful macro exactly once in sequence.")
        updated=super().observe_residual(residual, macro_step)
        self.observed_macro_count.add_(1)
        self.at_cap_macro_count.add_(((self.lambda_max>0) & (self.dual_lambda>=self.lambda_max)).long())
        return updated


class JointBudgetState(nn.Module):
    """Saved phase anchor, epoch/macro cadence, and independent consumed-type duals."""

    def __init__(self, lambda_M=0., lambda_E=0., *, sparse_start_epoch=3001, dtype=torch.float32):
        super().__init__()
        self.duals = nn.ModuleDict({name: TypedWorkDual(value, rng_identity="round9_"+name, dtype=dtype)
                                   for name, value in (("M", lambda_M), ("E", lambda_E))})
        self.register_buffer("sparse_start_epoch", torch.tensor(int(sparse_start_epoch), dtype=torch.long))
        self.register_buffer("epoch", torch.tensor(int(sparse_start_epoch)-1, dtype=torch.long))
        self.register_buffer("macro_index", torch.tensor(0, dtype=torch.long))

    def get_extra_state(self):
        return {"schema": JointBudgetSchedule.schema}

    def set_extra_state(self, state):
        if state.get("schema") != JointBudgetSchedule.schema:
            raise ValueError("Joint budget checkpoint schema mismatch.")

    @torch.no_grad()
    def set_clock(self, epoch, macro_index_zero_based=0):
        state = JointBudgetSchedule.state_at_epoch(epoch, macro_index_zero_based,
                                                   sparse_start_epoch=int(self.sparse_start_epoch))
        self.epoch.fill_(int(epoch)); self.macro_index.fill_(int(macro_index_zero_based))
        for name, dual in self.duals.items():
            dual.target_removable_retention.fill_(state["target_"+name])
            dual.temperature.fill_(state["temperature"]); dual.noise_scale.fill_(state["noise_scale"])
            dual.phase_code.fill_({"all_open_boundary":1,"stochastic_dense_router":2,
                                   "balanced_stochastic_router":3,"deterministic_hard":4}[state["phase"]])
        return state

    def schedule(self):
        return JointBudgetSchedule.state_at_epoch(int(self.epoch), int(self.macro_index),
                                                  sparse_start_epoch=int(self.sparse_start_epoch))

    def penalty(self, residual_M, residual_E):
        return self.duals["M"].penalty_residual(residual_M) + self.duals["E"].penalty_residual(residual_E)

    def observe_residuals(self, residual_M, residual_E, successful_macro_step):
        return {name: self.duals[name].observe_residual(value, successful_macro_step)
                for name, value in (("M", residual_M), ("E", residual_E))}


def calibrated_dual_weight(native_gradient_norm: float, work_gradient_norm: float, fraction=.05) -> float:
    if not all(math.isfinite(value) and value > 0 for value in (native_gradient_norm, work_gradient_norm, fraction)):
        raise ValueError("Dual calibration requires supported, finite, positive native/work gradient norms.")
    return fraction * native_gradient_norm / work_gradient_norm


def collect_work_observation(model, organization, query, plan, raw_M, raw_E, protected_M, protected_E):
    """One read's work, keyed by original scene identity across native roles."""
    from .graph import final_support
    from .routing import dispatch_joint_native_seed, direct_typed_gates
    records = organization.joint_records
    qmask = query.query_mask
    if qmask is None:
        qmask = torch.ones(plan.hard_support.shape[:2], device=raw_M.device, dtype=torch.bool)
    protected = torch.cat((protected_M, protected_E), -1)
    state = model._mode()[0]
    if model.routing_mode == "joint":
        directory = organization.joint_directory
        expected = expected_typed_work(directory, plan.dispatch.probabilities, protected, query_mask=qmask)
        with torch.no_grad():
            dispatch = dispatch_joint_native_seed(raw_M, raw_E, records, model.router.dispatch_parameters,
                                                 deterministic=True, query_mask=qmask)
            _require(~dispatch.tie_indeterminate, "Deterministic controller has indeterminate seed ties.")
            deterministic = replace(directory, membership_hard=directory.deterministic_hard)
            hard = final_support(dispatch.dispatch_seed, deterministic, protected, query_mask=qmask)
            floor = final_support(dispatch.dispatch_seed, deterministic, protected, query_mask=qmask, mandatory_only=True)
            hard_plan = JointReadPlan(hard, hard.to(raw_M), floor, protected, records.module_count, dispatch)
            observed = observed_typed_work(hard_plan, records, query_mask=qmask)
    else:
        expected = direct_typed_work(plan, records, query_mask=qmask)
        with torch.no_grad():
            deterministic = direct_typed_gates(raw_M, raw_E, records, model.router.direct_parameters,
                                                protected, query_mask=qmask, deterministic=True)
            observed = observed_typed_work(deterministic, records, query_mask=qmask)
    controller = expected if state["stochastic_expectation_exact"] else observed
    scene = organization.scene
    keys = scene.scene_key
    if not isinstance(keys, (tuple, list)) or len(keys) != scene.batch_size:
        keys = tuple((repr(keys), i) for i in range(scene.batch_size))
    else:
        keys = tuple(repr(key) for key in keys)
    return keys, expected, controller, observed


def microbatch_result(model):
    """Aggregate auxiliary/response requests by scene, then normalize per case."""
    if model.routing_mode == "dense":
        return {"constraints": {}, "telemetry": {}}
    state = model._mode()[0]
    cases = {"M": {}, "E": {}}
    for keys, expected, controller, observed in model._joint_entries:
        for kind in ("M", "E"):
            for index, key in enumerate(keys):
                e, c, o = expected[kind], controller[kind], observed[kind]
                values = (e.case_expected_rows[index], e.case_floor_rows[index], e.case_possible_rows[index],
                          c.case_expected_rows[index], c.case_floor_rows[index],
                          o.case_expected_rows[index], o.case_floor_rows[index])
                if key in cases[kind]:
                    values = tuple(old+new for old, new in zip(cases[kind][key], values))
                cases[kind][key] = values
    zero = next(model.router.parameters()).sum() * 0
    constraints, telemetry = {}, {}
    for kind in ("M", "E"):
        residuals, controllers, relaxed_fractions, hard_fractions, targets = [], [], [], [], []
        for expected, floor, possible, controller, controller_floor, observed, observed_floor in cases[kind].values():
            # Empty types contribute no constraint denominator. Keep this
            # decision input-only and outside checkpointed physical primitives.
            if not bool(possible.detach() > 0):
                continue
            denominator = possible.clamp_min(1)
            rho = state["target_"+kind]
            target = (floor + rho*(possible-floor))/denominator
            controller_target = (controller_floor + rho*(possible-controller_floor))/denominator
            residuals.append(expected/denominator-target)
            controllers.append(controller/denominator-controller_target)
            relaxed_fractions.append(expected.detach()/denominator)
            hard_fractions.append(observed/denominator)
            targets.append(target.detach())
        constraints[kind] = {"residual_numerator": sum(residuals, zero), "denominator": len(residuals),
                             "controller_numerator": sum(controllers, zero.detach()).detach()}
        telemetry.update({"joint_"+kind+"_relaxed_case_sum": sum(relaxed_fractions, zero.detach()),
                          "joint_"+kind+"_deterministic_case_sum": sum(hard_fractions, zero.detach()),
                          "joint_"+kind+"_target_case_sum": sum(targets, zero.detach()),
                          "joint_"+kind+"_case_requests": len(residuals)})
    telemetry.update(joint_dense_router_step=int(state["dense_router_step"]),
                     joint_exact_expectation=int(state["stochastic_expectation_exact"]))
    # The caller consumes numerators for backward; do not retain request graphs
    # after returning them or accumulate duplicate observations on repeated calls.
    model._joint_entries = []
    return {"constraints": constraints, "telemetry": telemetry}


def after_update(model, epoch, update_index, residuals):
    if model.routing_mode == "dense":
        return {}
    model.joint_budget.set_clock(epoch, update_index)
    values = {}
    for kind in ("M", "E"):
        residual = residuals.get(kind)
        if residual is None:
            raise ValueError("A successful joint macro must provide both typed controller residuals.")
        values[kind] = model.joint_budget.duals[kind].observe_residual(
            residual, int(model.joint_budget.duals[kind].macro_step)+1)
    model._joint_macro_clock += 1
    return values

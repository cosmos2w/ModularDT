"""Round8 binary native-list dispatch, preserving the original A2-D field laws.

The router estimator is deliberately biased. Its hard forward is the deployed
field; its dense backward provides hypothetical omitted-value task signals.
It is never the exported derivative of a discrete inference decision.
"""
from __future__ import annotations

import hashlib
import weakref
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ..content_hypergraph.model_v2 import ContentHypergraphCoreV2
from .core import NativeRetrievalCore


class BudgetedNativeCore(NativeRetrievalCore):
    schema = "native_budgeted_dispatch_v2"

    def __init__(self, base_model_config: Mapping[str, Any], *,
                 native_backend: str = "dense_hard_reference", routing_seed: int = 0,
                 membership_bias: float = 4.0, lambda0: float = 0.0,
                 environment_value_reader: bool = False) -> None:
        if native_backend not in {"dense_hard_reference", "flat_selected_reference", "grouped_native"}:
            raise ValueError("Round8 requires a declared binary native-value backend.")
        self.budgeted_backend = native_backend
        self.routing_seed = int(routing_seed)
        self.initial_membership_bias = float(membership_bias)
        self.initial_lambda0 = float(lambda0)
        self._budgeted_all_open = False
        self._budgeted_gate_override = None
        self._budgeted_entries = []
        self._budgeted_observations = []
        self._budgeted_update_index = 0
        self._budgeted_macro_clock = 0
        super().__init__(base_model_config, native_backend="selected",
                         environment_value_reader=environment_value_reader)
        from .budget_state import BudgetDualState
        from .budgeted_graph import BudgetedGraphParameters
        from .seed_dispatch import SeedDispatchParameters
        self.budgeted_router = nn.Module()
        self.budgeted_router.membership = BudgetedGraphParameters(self.num_heads)
        self.budgeted_router.dispatch = SeedDispatchParameters(self.num_heads)
        self.budgeted_router.dual = BudgetDualState(lambda0=lambda0)
        with torch.no_grad():
            self.budgeted_router.membership.membership_bias.fill_(membership_bias)
        self.register_buffer("budgeted_macro_step", torch.tensor(0, dtype=torch.long))
        self.budgeted_router.dual.apply_schedule(self.native_routing_state())

    @classmethod
    def from_a2d(cls, parent: ContentHypergraphCoreV2, **options):
        if type(parent) is not ContentHypergraphCoreV2:
            raise ValueError("Round8 parent must be the original full A2-D core.")
        model = cls(parent.get_config(), **options).to(next(parent.parameters()))
        state = model.state_dict()
        state.update(parent.state_dict())
        model.load_state_dict(state, strict=True)
        model.train(parent.training)
        return model

    def get_config(self) -> dict[str, Any]:
        base = ContentHypergraphCoreV2.get_config(self)
        base["schema"] = ContentHypergraphCoreV2.schema
        return {"schema": self.schema,
                "base_model_config": base,
                "native_backend": self.budgeted_backend,
                "routing_seed": self.routing_seed,
                "membership_bias": self.initial_membership_bias,
                "lambda0": self.initial_lambda0,
                "environment_value_reader": False}

    @classmethod
    def from_config(cls, config):
        value = dict(config)
        if value.pop("schema", None) != cls.schema:
            raise ValueError("Expected native_budgeted_dispatch_v2.")
        return cls(**value)

    def _config_key(self):
        base = ContentHypergraphCoreV2.get_config(self)
        return (tuple(sorted(base.items())), self.budgeted_backend, self.routing_seed,
                getattr(self, "_native_epoch", 500), self._budgeted_update_index,
                self._budgeted_all_open, id(self._budgeted_gate_override))

    def native_routing_state(self):
        from .budget_state import BudgetSchedule
        state = BudgetSchedule.state_at_epoch(self.native_epoch, self._budgeted_macro_clock + 1)
        state["work_probability_temperature"] = 1.0 if self.native_epoch <= 700 else state["temperature"]
        return state

    def set_native_training_epoch(self, epoch):
        super().set_native_training_epoch(epoch)
        if epoch and hasattr(self, "budgeted_router"):
            self.budgeted_router.dual.apply_schedule(self.native_routing_state())

    def _uses_native_retrieval(self):
        return not self._budgeted_all_open and self.native_epoch > 500

    def _native_capture_route_names(self):
        return ("mm",)

    def budgeted_begin_update(self, epoch: int, update_index: int):
        self.set_native_training_epoch(epoch)
        self._budgeted_update_index = int(update_index)
        self._budgeted_observations = []
        self.budgeted_begin_microbatch()
        return self.native_routing_state()

    def budgeted_begin_microbatch(self):
        self._budgeted_entries = []

    def budgeted_optimizer_parameter_names(self):
        return tuple(name for name, parameter in self.named_parameters()
                     if name.startswith("budgeted_router.") and parameter.requires_grad)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if hasattr(self, "budgeted_macro_step"):
            self._budgeted_macro_clock = int(self.budgeted_macro_step)
            self._native_epoch = int(self.native_training_epoch)
        return result

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Restore Python mirrors when PyTorch loads this core through an adapter."""
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        if hasattr(self, "native_training_epoch"):
            self._native_epoch = int(self.native_training_epoch)
        if hasattr(self, "budgeted_macro_step"):
            self._budgeted_macro_clock = int(self.budgeted_macro_step)

    def _scene_keys(self, scene):
        key = scene.scene_key
        if isinstance(key, (tuple, list)) and len(key) == scene.batch_size:
            return tuple(map(str, key))
        return tuple(repr(key) for _ in range(scene.batch_size))

    def _uniform(self, left, right, scene, namespace):
        """Counter-keyed noise: IDs label RNG draws only, never scores.

        This integer construction keeps a draw invariant to request chunking
        and native token permutation. There is no global RNG consumption.
        """
        prime = 2147483647
        seeds = []
        for key in self._scene_keys(scene):
            payload = repr(("round8_native_router_v2", self.routing_seed,
                            self.native_epoch, self._budgeted_update_index,
                            key, namespace)).encode()
            seeds.append(int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % prime)
        seed = torch.tensor(seeds, device=left.device, dtype=torch.long)[:, None, None]
        counter = (left.long()[..., None].remainder(prime) * 48271
                   + right.long()[:, None, :].remainder(prime) * 69621 + seed).remainder(prime)
        counter = torch.bitwise_xor(counter, counter >> 13)
        counter = (counter * 16807 + 12345).remainder(prime)
        counter = torch.bitwise_xor(counter, counter >> 11)
        counter = (counter * 48271 + 17).remainder(prime)
        return ((counter.to(torch.float64) + .5) / prime).clamp(1e-7, 1-1e-7)

    def _router_mode(self):
        state = self.native_routing_state()
        stochastic = self.training and not state["deterministic"]
        # The persisted device buffer is mirrored in Python and advanced only
        # after successful macro updates by the training engine.
        dense = self.native_epoch <= 700 or (self._budgeted_macro_clock + 1) % 4 == 0
        return state, stochastic, dense and self.training and torch.is_grad_enabled()

    def _finalize_native_preparation(self, organization, cycle_trace):
        if cycle_trace is None:
            return organization
        from .budgeted_graph import build_budgeted_native_graph
        from .contracts import NativeTokenRecords, NativeTokenStore
        scene = organization.scene
        def records(kind, keys=None, values=None):
            def heads(value):
                return None if value is None else value.reshape(
                    value.shape[0], value.shape[1], self.num_heads, -1).transpose(1, 2)
            return NativeTokenRecords(kind, getattr(scene, kind + "_ids"),
                getattr(scene, kind + "_coordinates"), getattr(scene, kind + "_characteristic_lengths"),
                getattr(scene, kind + "_mask"), getattr(scene, kind + "_measure"),
                getattr(scene, kind + "_features"), getattr(organization, kind + "_states"),
                heads(keys), heads(values))
        store = NativeTokenStore(records("module", organization.direct_source_keys,
                                        organization.direct_source_values), records("environment"),
            organization.model_owner, organization.model_parameter_versions,
            organization.model_buffer_versions, organization.scene_tensor_versions,
            organization.scene_identity_snapshot, organization.model_config_key,
            frame_key=scene.coordinate_frame_key, normalizer_key=scene.normalizer_key,
            live_autograd_graph=torch.is_grad_enabled())
        state, stochastic, dense_router = self._router_mode()
        noise = None
        if stochastic:
            u = self._uniform(scene.module_ids, scene.module_ids, scene, "membership")
            noise = (torch.log(u)-torch.log1p(-u)).to(organization.module_states.dtype)
            noise = noise * state["noise_scale"]
        snapshot = cycle_trace["mm_native_route_snapshot"]
        graph = build_budgeted_native_graph(snapshot.raw_scores, scene.module_mask,
            scene.module_ids, scene.module_coordinates, self.budgeted_router.membership,
            module_characteristic_lengths=scene.module_characteristic_lengths,
            temperature=state["temperature"], logistic_noise=noise,
            deterministic=not stochastic, straight_through=dense_router)
        organization.native_token_store = store
        organization.native_hypergraph = graph
        organization.budgeted_route_snapshot = snapshot
        store_ref = weakref.ref(store)

        def mark_backward_consumed(gradient):
            current_store = store_ref()
            if current_store is not None:
                current_store.mark_backward_consumed()
            return gradient

        for tensor in (organization.module_states, organization.environment_states):
            if torch.is_grad_enabled() and tensor.requires_grad:
                tensor.register_hook(mark_backward_consumed)
        organization.work_counts["native_route_capture_score_rows"] = (
            scene.batch_size * self.num_heads * scene.num_modules * scene.num_modules)
        return organization

    def _validate_prepared_state(self, organization):
        ContentHypergraphCoreV2._validate_prepared_state(self, organization)
        store = getattr(organization, "native_token_store", None)
        if self._uses_native_retrieval() and store is None:
            raise ValueError("Round8 requires a live native store and binary graph.")
        if store is not None:
            organization.native_hypergraph.assert_current()
            organization.budgeted_route_snapshot.assert_current()
            store.assert_current(model_owner=self._owner_token,
                model_parameter_versions=self._parameter_version_stamp(),
                model_buffer_versions=self._buffer_version_stamp(),
                scene_tensor_versions=organization.scene.tensor_version_stamp(),
                scene_identity=(organization.scene.scene_key, organization.scene.normalizer_key,
                                organization.scene.coordinate_frame_key), model_config_key=self._config_key())

    def _decode_native_retrieval(self, organization, query, query_states, local_pairs, *, return_messages=False):
        from .budgeted_graph import expected_final_work, final_support
        from .budgeted_reader import BudgetedNativeReader, dense_hard_router_gate
        from .reader import NativeSelectedReader
        from .seed_dispatch import dispatch_native_seed
        scores = NativeSelectedReader.score_queries(self.direct_reader, organization, query, query_states)
        graph, scene = organization.native_hypergraph, organization.scene
        protected = torch.zeros((query.batch_size, query.num_queries, scene.num_modules),
                                dtype=torch.bool, device=query_states.device)
        if local_pairs.count:
            protected[local_pairs.batch_index, local_pairs.query_index, local_pairs.module_index] = (
                local_pairs.support_weight > 0)
        state, stochastic, dense_router = self._router_mode()
        qmask = query.query_mask
        if qmask is None:
            qmask = torch.ones(query_states.shape[:2], dtype=torch.bool, device=query_states.device)
        noise = None
        if stochastic:
            if query.query_ids is None:
                raise ValueError("Stochastic dispatch requires native query IDs for chunk-stable noise.")
            u = self._uniform(query.query_ids, scene.module_ids, scene, "dispatch")
            noise = (-torch.log(-torch.log(u))).to(query_states.dtype) * state["noise_scale"]
        dispatch = dispatch_native_seed(scores.raw_logits, graph.seed_valid, graph.source_geometry,
            self.budgeted_router.dispatch, gumbel_noise=noise, deterministic=not stochastic,
            straight_through=dense_router, query_mask=qmask, temperature=state["temperature"],
            seed_ids=graph.seed_ids)
        if dispatch.tie_indeterminate.device.type == "cuda":
            torch._assert_async(~dispatch.tie_indeterminate.any(),
                                "Indistinguishable seed tie requires an explicit tied-union reference.")
        elif bool(dispatch.tie_indeterminate.any()):
            raise ValueError("Indistinguishable seed tie requires an explicit tied-union reference.")
        if torch.is_grad_enabled() and self.training:
            work_dispatch, work_membership = self._work_probabilities(graph, dispatch, qmask)
            work = expected_final_work(work_dispatch, work_membership,
                                       protected, qmask, scene.module_mask)
            deterministic_dispatch = dispatch_native_seed(scores.raw_logits, graph.seed_valid,
                graph.source_geometry, self.budgeted_router.dispatch, deterministic=True, query_mask=qmask,
                seed_ids=graph.seed_ids)
            deterministic_graph = graph.deterministic_hard.to(query_states.dtype)
            deterministic_work = expected_final_work(deterministic_dispatch.hard * qmask[..., None], deterministic_graph,
                                                     protected, qmask, scene.module_mask)
            self._budgeted_entries.append((self._scene_keys(scene), work, deterministic_work))
        gate = (dense_hard_router_gate(protected, dispatch.straight_through, graph.membership_straight_through)
                if dense_router else final_support(dispatch.dispatch_seed, graph, protected,
                                                   query_mask=qmask))
        gate = gate * qmask[..., None].to(gate.dtype)
        backend = "dense_hard_reference" if dense_router else self.budgeted_backend
        if self._budgeted_gate_override is not None:
            gate = self._budgeted_gate_override(gate, organization, query, dispatch, protected)
            # A same-weight support intervention is no longer the graph's
            # grouped execution plan. Preserve its field using the reference.
            backend = "dense_hard_reference"
        protection_trace_inputs = (
            {} if self._budgeted_gate_override is not None else
            {"graph": graph, "dispatch_plan": dispatch, "protected_mask": protected}
        )
        result = BudgetedNativeReader.evaluate(self.direct_reader, organization, query, query_states, gate,
            backend=backend, scores=scores, return_messages=return_messages,
            measure_reference=self._module_measure_reference(scene, query_states.dtype),
            **protection_trace_inputs)
        result.trace.update(native_dispatch_seed=dispatch.dispatch_seed,
                            native_dispatch_seed_ids=dispatch.dispatch_seed_ids,
                            native_membership_hard=graph.membership_hard,
                            native_membership_probabilities=graph.membership_probabilities,
                            native_protected_direct=protected)
        return result

    def _work_probabilities(self, graph, dispatch, query_mask):
        """Unit-noise marginals first; an explicitly biased relaxation later.

        The ST temperature is not a Bernoulli sampling temperature. Substituting
        it in the first phase would make the claimed expectation incorrect.
        """
        if self.native_epoch <= 700:
            return dispatch.probabilities * query_mask[..., None], graph.membership_probabilities
        temperature = self.native_routing_state()["work_probability_temperature"]
        allowed = graph.seed_valid[:, :, None] & graph.module_present[:, None, :]
        modules = graph.module_present.shape[-1]
        diagonal = torch.eye(modules, dtype=torch.bool, device=allowed.device)[None]
        membership = torch.sigmoid(graph.membership_logits / temperature) * allowed
        membership = torch.where(diagonal & allowed, torch.ones_like(membership), membership)
        valid = query_mask[..., None] & graph.seed_valid[:, None, :]
        has_seed = valid.any(dim=-1, keepdim=True)
        logits = torch.where(has_seed, dispatch.dispatch_logits / temperature,
                             torch.zeros_like(dispatch.dispatch_logits))
        probabilities = torch.softmax(logits, dim=-1) * valid.to(logits.dtype)
        return probabilities, membership

    def budgeted_microbatch_result(self):
        """Aggregate tiles/roles before equal-case normalization, preserving gradients."""
        cases = {}
        for keys, relaxed, deterministic in self._budgeted_entries:
            for i, key in enumerate(keys):
                values = (relaxed.case_expected_rows[i], deterministic.case_expected_rows[i],
                          relaxed.case_possible_rows[i], relaxed.floor_rows_per_query[i].sum())
                cases[key] = tuple(a+b for a, b in zip(cases[key], values)) if key in cases else values
        if not cases:
            zero = self.budgeted_router.membership.bias.sum() * 0
            return {"work_penalty_denominator": 0,
                    "constraint_residual_numerator": zero,
                    "relaxed_case_sum": zero.detach(), "controller_case_sum": zero.detach(),
                    "target_case_sum": zero.detach(), "telemetry": {}}
        rho = self.native_routing_state()["rho"]
        relaxed_fractions, hard_fractions, targets = [], [], []
        for relaxed, hard, possible, floor in cases.values():
            denominator = possible.clamp_min(1)
            relaxed_fractions.append(relaxed / denominator)
            hard_fractions.append(hard / denominator)
            targets.append((floor + rho * (possible-floor)) / denominator)
        relaxed = torch.stack(relaxed_fractions)
        hard = torch.stack(hard_fractions)
        target = torch.stack(targets)
        residual = relaxed - target
        controller = relaxed if self.native_epoch <= 700 else hard
        self._budgeted_observations.append(((controller-target).detach().sum(), len(cases)))
        return {"work_penalty_denominator": len(cases),
                "constraint_residual_numerator": residual.sum(),
                "relaxed_case_sum": relaxed.detach().sum(),
                "controller_case_sum": controller.detach().sum(),
                "target_case_sum": target.detach().sum(),
                "telemetry": {"budgeted_relaxed_case_sum": relaxed.detach().sum(),
                              "budgeted_deterministic_case_sum": hard.detach().sum(),
                              "budgeted_target_case_sum": target.detach().sum(),
                              "budgeted_case_requests": len(cases),
                              "budgeted_dense_router_step": int(self._router_mode()[2]),
                              "budgeted_work_probability_temperature": self.native_routing_state()["work_probability_temperature"],
                              "budgeted_exact_expectation": int(self.native_epoch <= 700)}}

    def budgeted_after_update(self, epoch, update_index):
        if self._budgeted_observations:
            count = sum(count for _, count in self._budgeted_observations)
            residual = sum(value for value, _ in self._budgeted_observations) / max(count, 1)
            self.budgeted_router.dual.observe_residual(residual, self._budgeted_macro_clock + 1)
        self.budgeted_macro_step.add_(1)
        self._budgeted_macro_clock += 1
        self._budgeted_observations = []

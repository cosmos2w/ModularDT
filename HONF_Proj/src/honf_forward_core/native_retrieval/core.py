"""Opt-in A2-D continuation with native token retrieval and unchanged laws."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ..content_hypergraph.model_v2 import ContentHypergraphCoreV2


class NativeRetrievalCore(ContentHypergraphCoreV2):
    """Keep every original parameter name and add only an execution policy.

    Environmental records are retained upstream and in the typed graph. This
    primary family has no direct environmental field-value reader.
    """

    schema = "native_retrieval_core_v1"

    def __init__(
        self,
        base_model_config: Mapping[str, Any],
        *,
        native_tau_mm: float = .20,
        native_tau_me: float = .20,
        native_tau_q: float = .20,
        native_backend: str = "selected",
        environment_value_reader: bool = False,
    ) -> None:
        base = dict(base_model_config)
        if base.pop("schema", None) != ContentHypergraphCoreV2.schema:
            raise ValueError("Native retrieval requires an exact version-2 A2-D base config.")
        if base.get("architecture") != "A2-D" or base.get("source_read_policy", "full") != "full":
            raise ValueError("Native retrieval requires uncompressed full A2-D, without another selector.")
        if environment_value_reader is not False:
            raise ValueError("Direct E values require the separately controlled pilot family.")
        if native_backend not in ("selected", "dense_masked", "compiled_selected"):
            raise ValueError("Native backend must compute the selected-permission function.")
        for name, value in (("MM", native_tau_mm), ("ME", native_tau_me), ("Q", native_tau_q)):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value < .5:
                raise ValueError(f"Native {name} threshold must be finite in [0,.5).")
        # Set before super: initialization can ask for the current config.
        self.native_tau_mm, self.native_tau_me, self.native_tau_q = map(
            float, (native_tau_mm, native_tau_me, native_tau_q))
        self.native_backend = native_backend
        self._native_force_selected = False
        self._native_permission_override = None
        super().__init__(**base)
        self.register_buffer("native_training_epoch", torch.tensor(500, dtype=torch.long))

    @classmethod
    def from_a2d(cls, parent: ContentHypergraphCoreV2, **options) -> NativeRetrievalCore:
        if type(parent) is not ContentHypergraphCoreV2:
            raise ValueError("The native retrieval parent must be the maintained A2-D core itself.")
        model = cls(parent.get_config(), **options).to(next(parent.parameters()))
        state = dict(parent.state_dict())
        state["native_training_epoch"] = model.native_training_epoch
        model.load_state_dict(state, strict=True)
        model.train(parent.training)
        return model

    def get_config(self) -> dict[str, Any]:
        base = super().get_config()
        base["schema"] = ContentHypergraphCoreV2.schema
        return {
            "schema": self.schema, "base_model_config": base,
            "native_tau_mm": self.native_tau_mm, "native_tau_me": self.native_tau_me,
            "native_tau_q": self.native_tau_q, "native_backend": self.native_backend,
            "environment_value_reader": False,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> NativeRetrievalCore:
        values = dict(config)
        if values.pop("schema", None) != cls.schema:
            raise ValueError("Expected native_retrieval_core_v1.")
        return cls(**values)

    def _config_key(self):
        # Tuples of scalar policy and the base config avoid mutable dict keys.
        base = super().get_config()
        epoch = getattr(self, "_native_epoch", 500)
        return (tuple(sorted(base.items())), self.native_tau_mm, self.native_tau_me,
                self.native_tau_q, self.native_backend, epoch,
                self._native_force_selected, id(self._native_permission_override))

    @property
    def native_epoch(self) -> int:
        return getattr(self, "_native_epoch", 500)

    def set_native_training_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or not 0 <= epoch <= 2500:
            raise ValueError("Native retrieval absolute epoch must be in [0,2500].")
        if epoch == 0:  # native inference adapters use zero to preserve saved state
            return
        if self.native_epoch != epoch:
            self.native_training_epoch.fill_(epoch)
        self._native_epoch = epoch

    def set_source_read_training_epoch(self, epoch: int) -> None:
        self.set_native_training_epoch(epoch)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._native_epoch = int(self.native_training_epoch)
        return result

    def native_routing_state(self) -> dict[str, Any]:
        epoch = self.native_epoch
        exploration = 1001 <= epoch <= 2000 and epoch % 8 == 0
        strength = max(0., min(1., (epoch - 501) / 499.))
        if exploration:
            strength = 0.
        return {"absolute_epoch": epoch, "exploration": exploration,
                "strength": strength, "tau_mm": strength * self.native_tau_mm,
                "tau_me": strength * self.native_tau_me,
                "tau_q": strength * self.native_tau_q,
                "environment_value_reader": False}

    def _uses_native_retrieval(self) -> bool:
        return (self._native_force_selected or self._native_permission_override is not None
                or self.native_routing_state()["strength"] > 0)

    def _captures_native_routes(self) -> bool:
        return self._uses_native_retrieval()

    def _finalize_native_preparation(self, organization, cycle_trace):
        if cycle_trace is None:
            return organization
        from .contracts import NativeTokenRecords, NativeTokenStore
        from .hypergraph import build_native_hypergraph

        scene = organization.scene
        def records(kind, keys=None, values=None):
            prefix = "module" if kind == "module" else "environment"
            states = getattr(organization, prefix + "_states")
            def heads(value):
                if value is None:
                    return None
                return value.reshape(value.shape[0], value.shape[1], self.num_heads, -1).transpose(1, 2)
            return NativeTokenRecords(
                kind, getattr(scene, prefix + "_ids"), getattr(scene, prefix + "_coordinates"),
                getattr(scene, prefix + "_characteristic_lengths"),
                getattr(scene, prefix + "_mask"),
                getattr(scene, prefix + "_measure"), getattr(scene, prefix + "_features"),
                states, heads(keys), heads(values))

        store = NativeTokenStore(
            records("module", organization.direct_source_keys, organization.direct_source_values),
            records("environment"), organization.model_owner, organization.model_parameter_versions,
            organization.model_buffer_versions, organization.scene_tensor_versions,
            organization.scene_identity_snapshot, organization.model_config_key,
            frame_key=scene.coordinate_frame_key, normalizer_key=scene.normalizer_key,
            live_autograd_graph=torch.is_grad_enabled())
        state = self.native_routing_state()
        graph = build_native_hypergraph(
            cycle_trace["mm_native_route_snapshot"], cycle_trace["me_native_route_snapshot"],
            tau_mm=state["tau_mm"], tau_me=state["tau_me"])
        organization.native_token_store = store
        organization.native_hypergraph = graph
        for tensor in (organization.module_states, organization.environment_states):
            if torch.is_grad_enabled() and tensor.requires_grad:
                def consumed(gradient, store=store):
                    store.mark_backward_consumed()
                    return gradient
                tensor.register_hook(consumed)
        organization.work_counts["native_route_capture_score_rows"] = (
            scene.batch_size * self.num_heads * scene.num_modules
            * (scene.num_modules + scene.num_environment_tokens))
        return organization

    def _validate_prepared_state(self, organization):
        super()._validate_prepared_state(organization)
        store = getattr(organization, "native_token_store", None)
        if self._uses_native_retrieval() and store is None:
            raise ValueError("Native retrieval requires its own live prepared token store.")
        if store is not None:
            organization.native_hypergraph.assert_current(
                owner_token=store.model_owner, parameter_versions=store.model_parameter_versions,
                buffer_versions=store.model_buffer_versions, scene_identity=store.scene_identity,
                model_config_key=store.model_config_key,
                scene_tensor_versions=store.scene_tensor_versions)
            store.assert_current(
                model_owner=self._owner_token, model_parameter_versions=self._parameter_version_stamp(),
                model_buffer_versions=self._buffer_version_stamp(),
                scene_tensor_versions=organization.scene.tensor_version_stamp(),
                scene_identity=(organization.scene.scene_key, organization.scene.normalizer_key,
                                organization.scene.coordinate_frame_key), model_config_key=self._config_key())

    def _decode_native_retrieval(self, organization, query, query_states, local_pairs, *, return_messages=False):
        from .planning import build_query_retrieval_plan
        from .reader import NativeSelectedReader

        scores = NativeSelectedReader.score_queries(self.direct_reader, organization, query, query_states)
        protected = torch.zeros(
            (query.batch_size, query.num_queries, organization.scene.num_modules),
            device=query_states.device, dtype=query_states.dtype)
        if local_pairs.count:
            protected[local_pairs.batch_index, local_pairs.query_index, local_pairs.module_index] = (
                local_pairs.support_weight)
        plan = build_query_retrieval_plan(
            scores.raw_logits, organization.native_hypergraph, tau_q=self.native_routing_state()["tau_q"],
            query_ids=query.query_ids, query_mask=query.query_mask, protected_module=protected,
            query_batch=query, token_store=organization.native_token_store,
            include_environment=False)
        if self._native_permission_override is not None:
            plan = self._native_permission_override(plan)
        return NativeSelectedReader.evaluate(
            self.direct_reader, organization, query, query_states, plan,
            backend=self.native_backend, return_messages=return_messages,
            measure_reference=self._module_measure_reference(organization.scene, query_states.dtype),
            scores=scores)

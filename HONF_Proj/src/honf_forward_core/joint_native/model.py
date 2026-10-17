"""Joint native E-conditioned reader, preserving original A2-D parameter names."""
from __future__ import annotations

import hashlib
import inspect
import math
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
from torch import nn

from ..content_hypergraph.model_v2 import ContentHypergraphCoreV2
from ..content_hypergraph.readers_v2 import DirectSourceReaderV2
from ..content_hypergraph.decoding import pack_local_pairs
from ..native_retrieval.contracts import NativeTokenRecords, NativeTokenStore
from ..native_retrieval.contracts import tensor_versions, _validate_ids
from .contracts import TypedNativeRecords
from .readers import TypedRead, raw_scores, read_typed


@dataclass(frozen=True)
class StoredNativeReadRoutes:
    """Fixed original-ID supports for explicit fresh-state counterfactuals."""
    module_ids: torch.Tensor
    environment_ids: torch.Tensor
    query_ids: torch.Tensor
    support: torch.Tensor
    module_present: torch.Tensor
    environment_present: torch.Tensor
    query_present: torch.Tensor
    _versions: tuple = field(init=False, repr=False)

    def __post_init__(self):
        for name in ("module","environment","query"):
            _validate_ids(getattr(self,name+"_ids"),getattr(self,name+"_present"),name+" stored IDs")
        if any(value.device!=self.support.device for value in (self.module_ids,self.environment_ids,self.query_ids,
                self.module_present,self.environment_present,self.query_present)):
            raise ValueError("Stored routes must share one device.")
        if self.support.dtype != torch.bool or self.support.shape != (
                self.query_ids.shape[0], self.query_ids.shape[1],
                self.module_ids.shape[1]+self.environment_ids.shape[1]):
            raise ValueError("Stored routes require boolean [B,Q,M+E] support.")
        object.__setattr__(self,"_versions",tensor_versions(self.module_ids,self.environment_ids,self.query_ids,self.support,
            self.module_present,self.environment_present,self.query_present))

    def assert_current(self):
        if tensor_versions(self.module_ids,self.environment_ids,self.query_ids,self.support,
                self.module_present,self.environment_present,self.query_present) != self._versions:
            raise ValueError("Stored native-ID routes changed after capture.")


class JointNativeCore(ContentHypergraphCoreV2):
    """One typed reader shared by dense, joint-directory and direct controls."""

    schema = "joint_native_hypergraph_v1"

    @classmethod
    def validate_config(cls, config):
        if not isinstance(config, Mapping) or config.get("schema") != cls.schema:
            raise ValueError("Expected joint_native_hypergraph_v1 configuration.")
        allowed = {"schema", "base_model_config", "routing_mode", "native_backend", "routing_seed",
                   "environment_value_reader", "checkpoint_values", "query_tile_size"}
        if set(config) - allowed:
            raise ValueError(f"Unknown joint native configuration keys: {sorted(set(config)-allowed)}")
        if set(config) != allowed:
            raise ValueError(f"Missing joint native configuration keys: {sorted(allowed-set(config))}")
        if type(config["routing_seed"]) is not int or config["routing_seed"] < 0:
            raise ValueError("routing_seed must be a nonnegative integer.")
        if config.get("environment_value_reader", True) is not True:
            raise ValueError("Joint native models require environment_value_reader=True.")
        if config.get("routing_mode", "dense") not in {"dense", "joint", "direct"}:
            raise ValueError("routing_mode must be dense, joint or direct.")
        if config.get("native_backend", "flat_typed_selected") not in {
                "flat_typed_selected", "dense_typed_reference", "dense_ME"}:
            raise ValueError("Unknown typed native backend.")
        if type(config.get("checkpoint_values", False)) is not bool:
            raise ValueError("checkpoint_values must be boolean.")
        tile = config.get("query_tile_size")
        if tile is not None and (type(tile) is not int or tile < 1):
            raise ValueError("query_tile_size must be positive.")
        base = dict(config.get("base_model_config", {}))
        if base.get("schema") != ContentHypergraphCoreV2.schema or base.get("architecture") != "A2-D":
            raise ValueError("Joint native base must be original content_hypergraph_core_v2 A2-D.")
        keys = set(inspect.signature(ContentHypergraphCoreV2.__init__).parameters) - {"self"}
        if set(base) - keys - {"schema"}:
            raise ValueError("Unknown original A2-D configuration field.")
        if base.get("source_read_policy", "full") != "full":
            raise ValueError("Joint native models retain the original full M parameterization.")
        signature = inspect.signature(ContentHypergraphCoreV2.__init__)
        resolved = {key:parameter.default for key,parameter in signature.parameters.items()
                    if key != "self" and parameter.default is not inspect.Parameter.empty}
        resolved.update({key:value for key,value in base.items() if key != "schema"})
        required = {key for key,parameter in signature.parameters.items()
                    if key != "self" and parameter.default is inspect.Parameter.empty}
        if required-set(resolved):
            raise ValueError("Joint native base is missing required original feature dimensions.")
        for key in ("module_input_dim","environment_input_dim","coordinate_dim","output_dim",
                    "context_dim","num_heads","context_depth","local_hidden_dim"):
            if type(resolved[key]) is not int or resolved[key]<1:
                raise ValueError(f"Original A2-D {key} must be a positive integer.")
        for key in ("query_input_dim","operating_input_dim","thermal_response_rank","init_seed"):
            if type(resolved[key]) is not int or resolved[key]<0:
                raise ValueError(f"Original A2-D {key} must be a nonnegative integer.")
        if resolved["context_dim"] % resolved["num_heads"] or resolved["local_hidden_dim"] % resolved["num_heads"]:
            raise ValueError("Original A2-D widths must be divisible by heads.")
        if resolved["context_dim"] != resolved["local_hidden_dim"]:
            raise ValueError("Joint native values require matching original state and reader widths.")
        if resolved["context_depth"] != 2:
            raise ValueError("Round9 retains the original two-cycle contextualizer.")
        if resolved["query_edge_geometry_mode"] is not None or resolved["reader_execution_backend"] not in {"legacy","reduced_projection"}:
            raise ValueError("Unsupported original A2-D decoder geometry/backend.")
        if resolved["source_value_conditioning"] not in {"none","source_scale"}:
            raise ValueError("Unsupported original native source conditioning.")
        if not isinstance(resolved["dropout"],(int,float)) or not math.isfinite(resolved["dropout"]) or not 0<=resolved["dropout"]<1:
            raise ValueError("Original dropout must be finite and lie in [0,1).")
        return base

    def __init__(self, base_model_config, *, routing_mode="dense", native_backend="flat_typed_selected",
                 routing_seed=0, environment_value_reader=True, checkpoint_values=False, query_tile_size=None):
        configuration = dict(schema=self.schema, base_model_config=base_model_config, routing_mode=routing_mode,
            native_backend=native_backend, routing_seed=routing_seed, environment_value_reader=environment_value_reader,
            checkpoint_values=checkpoint_values, query_tile_size=query_tile_size)
        base = self.validate_config(configuration)
        self.routing_mode, self.native_backend = routing_mode, native_backend
        self.routing_seed, self.checkpoint_values = int(routing_seed), checkpoint_values
        self.query_tile_size = query_tile_size
        self._joint_epoch, self._joint_update_index, self._joint_macro_clock = (2500 if routing_mode=="dense" else 3000), 0, 0
        self._joint_entries = []
        self._joint_route_override = None
        base.pop("schema")
        super().__init__(**base)
        self.register_buffer("joint_epoch", torch.tensor(self._joint_epoch, dtype=torch.long))
        self.register_buffer("joint_update_index", torch.tensor(0, dtype=torch.long))
        self.register_buffer("joint_macro_clock", torch.tensor(0, dtype=torch.long))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._stable_seed("environment_reader"))
            self.environment_reader = DirectSourceReaderV2(self.local_hidden_dim, self.coordinate_dim,
                self.num_heads, thermal=False, source_value_conditioning=self.source_value_conditioning)
            self.environment_conditioning = nn.Linear(self.context_dim, self.context_dim, bias=False)
        # Name-based initialization leaves all original M tensors bitwise intact.
        self._initialize_named_parameters()
        with torch.no_grad():
            self.environment_conditioning.weight.zero_()
        if routing_mode != "dense":
            from .graph import JointNativeRouter
            from .budget import JointBudgetState
            self.router = JointNativeRouter(self.num_heads, mode="hypergraph" if routing_mode == "joint" else "direct")
            self.joint_budget = JointBudgetState()

    @classmethod
    def from_a2d(cls, parent, **options):
        if type(parent) not in {ContentHypergraphCoreV2, cls}:
            raise ValueError("Joint native migration requires the original A2-D or dense ME parent.")
        base = (ContentHypergraphCoreV2.get_config(parent) if type(parent) is ContentHypergraphCoreV2
                else parent.get_config()["base_model_config"])
        base["schema"] = ContentHypergraphCoreV2.schema
        model = cls(base, **options).to(next(parent.parameters()))
        state = model.state_dict()
        parent_state = parent.state_dict()
        if set(parent_state) - set(state):
            raise ValueError("Parent contains incompatible parameters.")
        state.update(parent_state)
        model.load_state_dict(state, strict=True)
        if model.routing_mode != "dense" and type(parent) is cls and parent.routing_mode == "dense":
            model.set_training_clock(3000, 0)
        model.train(parent.training)
        return model

    def get_config(self):
        base = ContentHypergraphCoreV2.get_config(self)
        base["schema"] = ContentHypergraphCoreV2.schema
        return dict(schema=self.schema, base_model_config=base, routing_mode=self.routing_mode,
            native_backend=self.native_backend, routing_seed=self.routing_seed, environment_value_reader=True,
            checkpoint_values=self.checkpoint_values, query_tile_size=self.query_tile_size)

    @classmethod
    def from_config(cls, config):
        cls.validate_config(config)
        value = dict(config)
        value.pop("schema")
        return cls(**value)

    def _config_key(self):
        base = tuple(sorted(ContentHypergraphCoreV2.get_config(self).items()))
        return (base, self.routing_mode, self.native_backend, self.routing_seed, self.checkpoint_values,
                self.query_tile_size, self._joint_epoch, self._joint_update_index,
                self._joint_macro_clock, id(self._joint_route_override))

    def _captures_native_routes(self):
        return self.routing_mode == "joint" and self._joint_route_override is None

    def _native_capture_route_names(self):
        return ("mm", "me", "em", "ee")

    def _uses_native_retrieval(self):
        return True

    def _mode(self):
        # Absolute sparse ages and balanced macro exposure are dataset-independent.
        from .budget import JointBudgetSchedule
        state = JointBudgetSchedule.state_at_epoch(self._joint_epoch, self._joint_update_index)
        return state, self.training and not state["deterministic"], self.training and torch.is_grad_enabled() and state["dense_router_step"]

    def _uniform(self, left, right, scene, namespace):
        prime = 2147483647
        keys = scene.scene_key
        keys = keys if isinstance(keys, (tuple, list)) and len(keys) == scene.batch_size else [repr(keys)] * scene.batch_size
        seeds = [int.from_bytes(hashlib.sha256(repr(("joint_native_v1", self.routing_seed,
            self._joint_epoch, self._joint_update_index, str(key), namespace)).encode()).digest()[:8], "little") % prime for key in keys]
        seed = torch.tensor(seeds, device=left.device, dtype=torch.long)[:, None, None]
        value = (left.long()[..., None].remainder(prime)*48271 + right.long()[:, None].remainder(prime)*69621 + seed).remainder(prime)
        value = torch.bitwise_xor(value, value >> 13)
        value = (value*16807+12345).remainder(prime)
        value = torch.bitwise_xor(value, value >> 11)
        return (((value*48271+17).remainder(prime).double()+.5)/prime).clamp(1e-7, 1-1e-7)

    def _finalize_native_preparation(self, organization, cycle_trace):
        scene = organization.scene
        ek, ev = self.environment_reader.prepare_sources(organization.environment_states)
        organization.joint_environment_keys, organization.joint_environment_values = ek, ev
        organization.joint_projection_versions = tensor_versions(ek, ev)
        organization.joint_records = TypedNativeRecords.from_scene(scene)
        def records(kind, keys, values):
            def heads(value):
                return value.reshape(value.shape[0], value.shape[1], self.num_heads,
                                     value.shape[-1]//self.num_heads).transpose(1, 2)
            return NativeTokenRecords(kind, getattr(scene, kind+"_ids"), getattr(scene, kind+"_coordinates"),
                getattr(scene, kind+"_characteristic_lengths"), getattr(scene, kind+"_mask"),
                getattr(scene, kind+"_measure"), getattr(scene, kind+"_features"),
                getattr(organization, kind+"_states"), heads(keys), heads(values))
        store = NativeTokenStore(records("module", organization.direct_source_keys, organization.direct_source_values),
            records("environment", ek, ev), organization.model_owner, organization.model_parameter_versions,
            organization.model_buffer_versions, organization.scene_tensor_versions,
            organization.scene_identity_snapshot, organization.model_config_key,
            frame_key=scene.coordinate_frame_key, normalizer_key=scene.normalizer_key,
            live_autograd_graph=torch.is_grad_enabled())
        organization.native_token_store = store
        organization.work_counts["joint_environment_prepare_projection_rows"] = scene.batch_size * scene.num_environment_tokens
        organization.work_counts["joint_native_record_rows"] = scene.batch_size * (scene.num_modules + scene.num_environment_tokens)
        if self.routing_mode == "joint" and self._joint_route_override is None:
            state, stochastic, dense_router = self._mode()
            capture = {name: cycle_trace[name+"_native_route_snapshot"] for name in self._native_capture_route_names()}
            noise = {}
            if stochastic:
                for route in self._native_capture_route_names():
                    left = scene.module_ids if route[0] == "m" else scene.environment_ids
                    right = scene.module_ids if route[1] == "m" else scene.environment_ids
                    u = self._uniform(left, right, scene, "membership:"+route)
                    noise[route] = (torch.log(u)-torch.log1p(-u)).to(organization.module_states)
            organization.joint_directory = self.router.build(capture, scene, deterministic=not stochastic,
                straight_through=dense_router, temperature=state["temperature"], noise_scale=state["noise_scale"],
                logistic_noise=noise or None)
            organization.joint_capture = capture
            organization.work_counts["joint_route_capture_score_rows"] = (
                scene.batch_size * self.num_heads * (scene.num_modules+scene.num_environment_tokens)**2)
        store_ref = weakref.ref(store)
        def mark_consumed(gradient):
            live_store = store_ref()
            if live_store is not None:
                live_store.mark_backward_consumed()
            return gradient
        for tensor in (organization.module_states, organization.environment_states):
            if torch.is_grad_enabled() and tensor.requires_grad:
                tensor.register_hook(mark_consumed)
        return organization

    def _validate_prepared_state(self, organization):
        super()._validate_prepared_state(organization)
        if tensor_versions(organization.joint_environment_keys, organization.joint_environment_values) != organization.joint_projection_versions:
            raise ValueError("Prepared E projections changed after preparation.")
        organization.joint_records.assert_current()
        store = organization.native_token_store
        store.assert_current(model_owner=self._owner_token,
            model_parameter_versions=self._parameter_version_stamp(), model_buffer_versions=self._buffer_version_stamp(),
            scene_tensor_versions=organization.scene.tensor_version_stamp(),
            scene_identity=(organization.scene.scene_key, organization.scene.normalizer_key, organization.scene.coordinate_frame_key),
            model_config_key=self._config_key())
        if self.routing_mode == "joint" and self._joint_route_override is None:
            organization.joint_directory.assert_current()
            for snapshot in organization.joint_capture.values():
                snapshot.assert_current()

    def _read_type(self, kind, organization, query, qstates, gate, backend, return_messages=False):
        scene = organization.scene
        is_m = kind == "module"
        valid = getattr(scene, kind+"_mask")
        measure = getattr(scene, kind+"_measure")
        if not is_m:
            valid = valid & (measure > 0)
        references = self._measure_references(scene, qstates.dtype)
        if is_m and backend == "dense_ME" and self._joint_route_override is None and not self.checkpoint_values and self.query_tile_size is None:
            # The literal parent boundary retains its original reduction order.
            context, alpha, response, messages = self.direct_reader(scene, query, qstates,
                organization.frame_scale, references[0], organization.direct_source_keys,
                organization.direct_source_values, return_messages=return_messages,
                execution_backend=self.reader_execution_backend)
            rows = scene.batch_size * query.num_queries * scene.num_modules
            selected_rows = int((gate.detach()>0).nonzero(as_tuple=False).shape[0])
            counts = dict(value_rows=rows, selected_rows=selected_rows, candidate_rows=rows, score_rows=rows,
                message_projection_rows=rows, scalar_projection_rows=rows if self.thermal_response_rank else 0,
                fourier_entries=rows*7*self.coordinate_dim, hidden_entries=rows*self.local_hidden_dim,
                padded_rows=0, invalid_candidate_rows_available=0, dense_value_rows=rows,
                checkpoint_forward_rows=0, checkpoint_recomputation_rows_available=0)
            return TypedRead(context, alpha, response, messages, counts)
        return read_typed(self.direct_reader if is_m else self.environment_reader, query, qstates,
            coordinates=getattr(scene, kind+"_coordinates"), lengths=getattr(scene, kind+"_characteristic_lengths"),
            valid=valid, measure=measure, reference=references[0 if is_m else 1], frame_scale=organization.frame_scale,
            keys=organization.direct_source_keys if is_m else organization.joint_environment_keys,
            values=organization.direct_source_values if is_m else organization.joint_environment_values,
            gate=gate, backend=backend, return_messages=return_messages, checkpoint_values=self.checkpoint_values,
            query_tile_size=self.query_tile_size)

    def _prepare_native_decode(self, organization, query, query_states, *, trace_enabled=False):
        scene = organization.scene
        batch, queries = query_states.shape[:2]
        plan = None
        backend = "dense_ME" if self.routing_mode == "dense" else self.native_backend
        boundary = self.routing_mode != "dense" and self._joint_epoch == 3000
        if boundary:
            backend = "dense_ME"
        if self._joint_route_override is not None:
            gates = self._joint_route_override
        elif self.routing_mode == "dense" or boundary:
            gates = torch.cat((scene.module_mask, scene.environment_mask & (scene.environment_measure>0)), -1)
            gates = gates[:, None].expand(batch, queries, -1)
        else:
            from .routing import nearest_environment_protection
            records = organization.joint_records
            qmask = (query.query_mask if query.query_mask is not None else torch.ones((batch,queries), dtype=torch.bool, device=query_states.device))
            local_pairs = pack_local_pairs(scene, query, organization.frame_scale)
            pm = torch.zeros((batch,queries,scene.num_modules), dtype=torch.bool, device=query_states.device)
            if local_pairs.count:
                pm[local_pairs.batch_index, local_pairs.query_index, local_pairs.module_index] = local_pairs.support_weight > 0
            pe = nearest_environment_protection(records, query.coordinates, query_mask=qmask)
            raw_m = raw_scores(self.direct_reader, query, query_states, scene.module_coordinates,
                scene.module_characteristic_lengths, organization.frame_scale, organization.direct_source_keys)
            raw_e = raw_scores(self.environment_reader, query, query_states, scene.environment_coordinates,
                scene.environment_characteristic_lengths, organization.frame_scale, organization.joint_environment_keys)
            state, stochastic, dense_router = self._mode()
            noise, direct_noise = None, None
            if stochastic:
                if query.query_ids is None:
                    raise ValueError("Stochastic native routing requires stable query IDs.")
                # Type-specific RNG namespaces prevent M:0/E:0 collisions.
                um = self._uniform(query.query_ids, scene.module_ids, scene, "query:module")
                ue = self._uniform(query.query_ids, scene.environment_ids, scene, "query:environment")
                u = torch.cat((um, ue), -1)
                noise = (-torch.log(-torch.log(u))).to(query_states)
                logistic = (torch.log(u)-torch.log1p(-u)).to(query_states)
                direct_noise = {"module":logistic[..., :scene.num_modules], "environment":logistic[..., scene.num_modules:]}
            plan = self.router.route(getattr(organization, "joint_directory", records), raw_m, raw_e, pm, pe,
                query_mask=qmask, deterministic=not stochastic, straight_through=dense_router,
                noise_scale=state["noise_scale"], temperature=state["temperature"],
                gumbel_noise=noise, direct_logistic_noise=direct_noise,query_ids=query.query_ids)
            gates = plan.value_gates if dense_router else plan.hard_support
            if dense_router:
                backend = "dense_typed_reference"
            self._record_joint_work(organization, query, plan, raw_m, raw_e, pm, pe)
        eligible = torch.cat((scene.module_mask,scene.environment_mask & (scene.environment_measure>0)),-1)
        gates = gates * eligible[:,None]
        if query.query_mask is not None:
            gates = gates * query.query_mask[...,None]
        e = self._read_type("environment", organization, query, query_states, gates[..., scene.num_modules:], backend)
        conditioned = query_states + self.environment_conditioning(e.context)
        return conditioned, (gates, backend, e, plan, trace_enabled)

    def _decode_native_values(self, organization, query, query_states, local_pairs, *, return_messages=False, decode_context=None):
        gates, backend, environment, plan, trace_enabled = decode_context
        modules = organization.scene.num_modules
        m = self._read_type("module", organization, query, query_states, gates[..., :modules], backend, return_messages)
        counts = {"joint_M_"+key:value for key,value in m.counts.items()}
        counts.update({"joint_E_"+key:value for key,value in environment.counts.items()})
        counts.update(fine_executed_rows=m.counts["value_rows"], fine_selected_rows=m.counts["selected_rows"],
            fine_message_projection_rows=m.counts["message_projection_rows"], fine_scalar_projection_rows=m.counts["scalar_projection_rows"],
            fine_fourier_feature_entries=m.counts["fourier_entries"], fine_hidden_entries=m.counts["hidden_entries"],
            source_score_rows=m.counts["score_rows"], native_environment_consumed_ids=environment.counts["selected_rows"],
            native_environment_metadata_ids=0, joint_pre_E_M_score_rows=0 if self.routing_mode=="dense" else m.counts["candidate_rows"],
            joint_pre_E_E_score_rows=0 if self.routing_mode=="dense" else environment.counts["candidate_rows"])
        trace = {}
        if trace_enabled:
            trace = {"native_hard_support": gates[..., :modules].detach()>0,
                     "joint_hard_support":gates.detach()>0,
                     "joint_environment_support": gates[..., modules:].detach()>0,
                     "joint_environment_access": environment.alpha,
                     "joint_environment_context": environment.context,
                     "joint_conditioned_query": query_states,
                     "native_module_ids":organization.scene.module_ids,
                     "joint_environment_ids":organization.scene.environment_ids,
                     "joint_record_types":organization.joint_records.types}
            if self.routing_mode == "joint" and hasattr(organization,"joint_directory"):
                directory=organization.joint_directory
                trace.update(native_membership_hard=directory.membership_hard,
                    native_membership_probabilities=directory.membership_probabilities,
                    joint_self_membership=directory.self_membership,
                    joint_cross_membership=directory.cross_membership)
            if plan is not None:
                trace["joint_protected_mask"]=plan.protected_mask
                trace["joint_mandatory_support"]=plan.mandatory_support
            if plan is not None and plan.seed_index is not None:
                trace["native_dispatch_seed"] = plan.seed_index
        return SimpleNamespace(context=m.context, alpha=m.alpha, response=m.response, messages=m.messages,
                               counts=counts, trace=trace)

    def _record_joint_work(self, organization, query, plan, raw_m, raw_e, protected_m, protected_e):
        if not (self.training and torch.is_grad_enabled()):
            return
        from .budget import collect_work_observation
        self._joint_entries.append(collect_work_observation(self, organization, query, plan, raw_m, raw_e,
                                                           protected_m, protected_e))

    @torch.no_grad()
    def capture_read_routes(self, organization, query):
        """Capture actual original-ID read supports after ordinary state validation."""
        if self.training or query.query_ids is None:
            raise ValueError("Stored routes require evaluation mode and stable receiver IDs.")
        self.validate_prepared_state(organization,query)
        _, context = self._prepare_native_decode(organization,query,self._query_states(organization,query))
        gates = context[0]
        return StoredNativeReadRoutes(organization.scene.module_ids.detach().clone(),
            organization.scene.environment_ids.detach().clone(),query.query_ids.detach().clone(),
            gates.detach().gt(0).clone(),organization.scene.module_mask.detach().clone(),
            organization.scene.environment_mask.detach().clone(),
            (query.query_mask.detach().clone() if query.query_mask is not None else torch.ones_like(query.query_ids,dtype=torch.bool)))

    @torch.no_grad()
    def decode_counterfactual_routes(self, scene, query, routes, controls=None, **decode_options):
        """Apply stored typed IDs to freshly rebuilt states, including changed inputs.

        This explicit same-read-set intervention does not reuse prepared graph,
        source projections, geometry, or contextualized record tensors.
        """
        if self.training or not isinstance(routes,StoredNativeReadRoutes) or query.query_ids is None:
            raise ValueError("Counterfactual routes require evaluation mode and captured native IDs.")
        routes.assert_current()
        def correspondence(current,stored,current_present,stored_present,require_inventory=False):
            if current.shape[0]!=stored.shape[0] or current.device!=stored.device:
                raise ValueError("Counterfactual route batch/device identity differs.")
            equal=(current[:,:,None] == stored[:,None,:]) & current_present[:,:,None] & stored_present[:,None,:]
            if not bool(((equal.sum(-1)==1) | ~current_present).all()):
                raise ValueError("Counterfactual routes require unique corresponding native IDs.")
            if require_inventory and not bool(((equal.sum(1)==1) | ~stored_present).all()):
                raise ValueError("Counterfactual routes require the same present typed inventory.")
            return equal.long().argmax(-1) if stored.shape[1] else current.new_empty(current.shape)
        qpresent=query.query_mask if query.query_mask is not None else torch.ones_like(query.query_ids,dtype=torch.bool)
        qi=correspondence(query.query_ids,routes.query_ids,qpresent,routes.query_present)
        mi=correspondence(scene.module_ids,routes.module_ids,scene.module_mask,routes.module_present,True)
        ei=correspondence(scene.environment_ids,routes.environment_ids,scene.environment_mask,routes.environment_present,True)
        old_m=routes.module_ids.shape[1]
        selected=routes.support.gather(1,qi[...,None].expand(-1,-1,routes.support.shape[-1]))
        support=torch.cat((selected[...,:old_m].gather(2,mi[:,None].expand(-1,query.num_queries,-1)),
                           selected[...,old_m:].gather(2,ei[:,None].expand(-1,query.num_queries,-1))),-1)
        eligible=torch.cat((scene.module_mask,scene.environment_mask & (scene.environment_measure>0)),-1)
        support=support & eligible[:,None]
        previous=self._joint_route_override
        self._joint_route_override=support
        try:
            fresh=self.prepare(scene)
            return self.decode(fresh,query,controls,**decode_options)
        finally:
            self._joint_route_override=previous

    def set_training_clock(self, epoch, global_step):
        self._joint_epoch, self._joint_macro_clock = int(epoch), int(global_step)
        self.joint_epoch.fill_(int(epoch))
        self.joint_macro_clock.fill_(int(global_step))

    def joint_begin_update(self, epoch, update_index):
        self._joint_epoch, self._joint_update_index = int(epoch), int(update_index)
        self.joint_epoch.fill_(int(epoch))
        self.joint_update_index.fill_(int(update_index))
        self.joint_begin_microbatch()
        if self.routing_mode == "dense":
            return {"absolute_epoch":int(epoch), "constraints_enabled":False}
        return self.joint_budget.set_clock(epoch, update_index)

    def joint_begin_microbatch(self):
        self._joint_entries = []

    def joint_microbatch_result(self):
        from .budget import microbatch_result
        return microbatch_result(self)

    def joint_constraint_duals(self):
        return {} if self.routing_mode == "dense" else {kind:self.joint_budget.duals[kind] for kind in ("M", "E")}

    def joint_after_update(self, epoch, update_index, residuals):
        from .budget import after_update
        result = after_update(self, epoch, update_index, residuals)
        self._joint_macro_clock += int(self.routing_mode == "dense")
        self.joint_macro_clock.fill_(self._joint_macro_clock)
        return result

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        self._joint_epoch = int(self.joint_epoch)
        self._joint_update_index = int(self.joint_update_index)
        self._joint_macro_clock = int(self.joint_macro_clock)


__all__ = ["JointNativeCore", "StoredNativeReadRoutes"]

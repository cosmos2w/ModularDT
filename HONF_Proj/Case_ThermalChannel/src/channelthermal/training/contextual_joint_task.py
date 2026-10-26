"""Native Thermal observations for the joint contextual frontier experiment.

The existing sampler, targets, masks, field reductions and native evaluation
are reused. Prescribed heat belongs to the prepared context, not an affine
application. Only admitted selected-TRAIN references may supply gradients.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from honf_runtime.unified_training import OptimizerGroupSpec, ScheduleSpec, TaskBatch, _sampling_dataset_id
from ..contextual_joint import ThermalContextualJointAdapter
from ..data.datasets import fit_global_normalizer
from ..data.development_split import development_case_ids, read_case_catalog
from .content_hypergraph_task import (
    ThermalContentHypergraphTask, ThermalContentSceneInputs, ThermalContentReceivers,
    ThermalContentTargets, ThermalContentPredictions, THERMAL_CONTENT_LOSS_WEIGHTS,
    THERMAL_CONTENT_SAMPLING_VERSION, _ids_sha256,
)
from .joint_task import _load_data_binding, _read_selected_cases, validate_full_train_followup_membership
from .unified_task import DEFAULT_ATLAS_DIRECTORY, _sample_primary, _structure_only


class ThermalContextualJointTask(ThermalContentHypergraphTask):
    def __init__(self, *, model, training_cases, validation_cases, manifest, data_path,
                 recipe, device, manifest_path=None, catalog=(), source_binding=None,
                 response_families=None, native_only=False):
        self.model = self.adapter = model
        self.training_cases = tuple(training_cases)
        self.validation_cases = tuple(validation_cases)
        self.recipe = copy.deepcopy(recipe)
        self.manifest = copy.deepcopy(manifest)
        self.base_manifest_sha256 = str(manifest["manifest_sha256"])
        self.data_path, self.manifest_path = Path(data_path), manifest_path
        self.source_binding = dict(source_binding or {})
        self.device = torch.device(device)
        self.protocol = "development" if len(training_cases) == 150 else "contextual_full_train100"
        self.validation_scope = "fixed25_v1_DEV22_exposed" if self.protocol == "development" else "canonical_VAL89_exposed"
        self.training_case_ids = tuple(str(c["case_id"]) for c in training_cases)
        self.validation_case_ids = tuple(str(c["case_id"]) for c in validation_cases)
        self._train_index_by_id = {v: i for i, v in enumerate(self.training_case_ids)}
        self._validation_index_by_id = {v: i for i, v in enumerate(self.validation_case_ids)}
        self.case_split_by_id = {str(c["case_id"]): str(c["split"]) for c in catalog}
        self.catalog_case_ids = tuple(self.case_split_by_id)
        self.seed = 0
        training = recipe.get("training", {})
        self.primary_queries = int(training.get("primary_queries", 1024))
        self.microbatch_size = int(training.get("microbatch_cases", 24))
        self.effective_batch_size = int(training.get("effective_batch_size", 48))
        self.total_epochs = int(training.get("local_horizon", training.get("total_epochs", 2000)))
        self.sampling_version = THERMAL_CONTENT_SAMPLING_VERSION
        self.sampling_dataset_id = _sampling_dataset_id({"dataset": "ThermalChannel",
            "subset_manifest_sha256": manifest["manifest_sha256"]})
        self.training_samples = self.validation_samples = 0
        self._validation_batches_cache = None
        self.atlas_directory = Path(recipe.get("atlas_directory", DEFAULT_ATLAS_DIRECTORY)).resolve()
        self.native_only = bool(native_only)
        self.train_families = ()
        self.response_scales = {}
        self.state = {"primary_updates": 0, "response_microbatches": 0,
                      "response_scene_visits": 0, "functional_comparisons": 0,
                      "calibration_scene_visits": 0, "calibrations": [], "baselines": {}}
        self._pending = []
        self.native_weights = dict(recipe.get("objective", {}).get("native_weights", THERMAL_CONTENT_LOSS_WEIGHTS))
        if self.native_only:
            self.organizer = None
            self.responses = None
        else:
            self.organizer = self._make_organizer()
            from .contextual_joint_responses import JointResponseSupervision
            families = (tuple(map(str, response_families)) if response_families is not None else
                        tuple(v for v in ("0001", "0318", "0333", "0348") if v in self.training_case_ids))
            if any(family not in self.training_case_ids for family in families):
                raise ValueError("Joint response families must be admitted selected TRAIN cases.")
            self.responses = JointResponseSupervision(self.atlas_directory, families, model.normalization_stats)
        self._budget_binding = None
        self._segment_started = time.monotonic()

    def _make_organizer(self):
        from .contextual_joint_objective import ContextualJointObjectiveRuntime
        return ContextualJointObjectiveRuntime(self.model, self)

    def identity_payload(self):
        return {"schema": "thermal_contextual_frontier_provider_v1", "dataset": "ThermalChannel",
            "architecture": self.model.architecture, "protocol": self.protocol,
            "training_case_ids": list(self.training_case_ids),
            "validation_case_ids": list(self.validation_case_ids), "validation_scope": self.validation_scope,
            "subset_manifest_sha256": self.manifest["manifest_sha256"],
            "normalization_fit_scope": "selected TRAIN only", "adapter_state": self.model.state_payload(),
            "training": {"seed": self.seed, "total_epochs": self.total_epochs,
                "primary_queries": self.primary_queries, "effective_batch_size": self.effective_batch_size,
                "microbatch_size": self.microbatch_size, "optimizer_schedule": self._schedule_payload(),
                "weight_decay": 1e-5},
            "objective": {"native_weights": self.native_weights,
                "joint_fields": ["u", "v", "p", "temperature"],
                "affine_temperature_constraint": False, "heat_independent_flow_constraint": False},
            "prescribed_heat_in_context": True,
            "response_training_family_ids": ([] if self.native_only else
                (["0348"] if self.protocol == "development" else ["0001", "0318", "0333", "0348"])),
            "source_binding": self.source_binding,
            "response_supervision": None if self.native_only else self.responses.metadata,
            "recipe": self.recipe,
            "solver_attempts": 0, "test_read": False}

    def preparation_summary(self):
        return {"training_cases": len(self.training_cases), "validation_cases": len(self.validation_cases),
                "validation_scope": self.validation_scope, "model": self.model.get_config()}

    def _schedule_payload(self):
        return {"peak_lr": 3e-4, "warmup_start_lr": 3e-4, "warmup_epochs": 0,
                "hold_through_epoch": self.total_epochs-1, "total_epochs": self.total_epochs, "final_lr": 3e-4}

    def optimizer_groups(self, model, arm, stage):
        if model is not self.model or arm != model.architecture:
            raise ValueError("Wrong contextual joint optimizer owner.")
        return (OptimizerGroupSpec("joint", tuple(n for n,p in model.named_parameters() if p.requires_grad),
                    ScheduleSpec(**self._schedule_payload()), weight_decay=1e-5),)

    def make_batch(self, case_keys, key):
        if key.sampling_version != self.sampling_version or key.dataset_id != self.sampling_dataset_id:
            raise ValueError("Thermal primary sampler identity mismatch.")
        if key.epoch < 1 or (not self.native_only and key.epoch > self.total_epochs):
            raise ValueError("Epoch outside the absolute master schedule.")
        self.model.core.set_organization_clock(key.epoch, key.update_index)
        indices = tuple(self._train_index_by_id[str(v)] for v in case_keys)
        sampled, sampling = _sample_primary(self.training_cases, indices, epoch=key.epoch, key=key,
            training=True, budget={"fluid_queries": self.primary_queries,
                "material_queries_per_module": 32, "surface_stride": 4}, device=self.device)
        case_ids = tuple(str(v) for v in case_keys)
        self.training_samples += len(indices)
        return TaskBatch(
            scene_inputs=ThermalContentSceneInputs(_structure_only(sampled["structure"]), case_ids=case_ids),
            receivers=ThermalContentReceivers(sampled["query_xy"], sampled["module_internal_query_points"],
                sampled["structure"]["heat_powers"], ntheta=16),
            targets=ThermalContentTargets(case_ids, indices, key.epoch, sampled["field_targets"],
                sampled["point_weights"], sampled["interface_target"],
                sampled["interface_condition_valid_mask"], sampled["module_internal_temperature_points"]),
            auxiliary={"query_sampling_sha256": sampling}, case_keys=case_ids)

    def make_scene(self, scene_inputs):
        return scene_inputs

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch=0, temperature=None):
        if model is not self.model or execution_mode != model.architecture:
            raise ValueError("Wrong contextual model/arm.")
        prepared = model.prepare_native(scene.structure, receivers.fluid_xy, physical_heat=receivers.heat,
            local_query_points=receivers.local_query_points, ntheta=receivers.ntheta, scene_key=scene.case_ids)
        main, decode = model.apply_native_with_work_counts(prepared, receivers.heat)
        work = {"native_fluid_queries": int(receivers.fluid_xy.shape[0]*receivers.fluid_xy.shape[1]),
                "native_union_rows": int(prepared.union.unique_counts.sum()),
                "native_local_pairs": int(prepared.query_batch.local_weights.numel())}
        work.update({f"core_prepare_primary_{k}": int(v) for k,v in prepared.organization.work_counts.items()})
        work.update({f"core_decode_primary_{k}": int(v) for k,v in decode.items()})
        predictions = ThermalContentPredictions(prepared, main, work=work)
        return predictions, {"work": work}

    def _weighted_native(self, predictions, targets):
        roles = self._role_losses(predictions, targets)
        return sum(self.native_weights[name] * value for name, value in roles.items())

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        from honf_runtime.unified_training import LossTerm
        roles = self._role_losses(predictions, targets)
        if (not self.native_only and self.model.training and torch.is_grad_enabled()
                and targets.epoch > 0):
            native = sum(self.native_weights[name] * value for name, value in roles.items())
            self.organizer.collect(predictions, native, targets.epoch, case_ids=targets.case_ids)
        return {name: LossTerm(value.sum(), len(targets.case_ids), self.native_weights[name])
                for name, value in roles.items()}

    def _calibration_batches(self, provider, case_ids, epoch, runtime):
        from honf_runtime.unified_training import SamplingKey
        for offset in range(0,len(case_ids),self.microbatch_size):
            key=SamplingKey(self.seed,epoch,0,offset//self.microbatch_size,"joint",self.model.architecture,
                sampling_version=self.sampling_version,dataset_id=self.sampling_dataset_id)
            batch=self.make_batch(case_ids[offset:offset+self.microbatch_size],key)
            predictions,_=self.predict_native(self.model,batch.scene_inputs,batch.receivers,
                self.model.architecture,"joint",epoch=epoch)
            yield predictions,self._weighted_native(predictions,batch.targets),batch.targets.case_ids
        from .contextual_joint_sharing import functional_probe
        for index in range(4):
            terms, detail = functional_probe(self,index,epoch,calibrating=True)
            for name,term in terms.items():
                runtime.add_raw_gradients(name,term["bank"],term["gradients"],samples=term["samples"],
                    objective_value=term["objective_value"])
            runtime.calibration_extra.append(detail)

    def _response_gradient(self, native, *, index, calibrate=False):
        from honf_runtime.thermal_soft_organization import gradient_norm, gradient_balance
        named={n:p for n,p in self.model.named_parameters() if p.requires_grad}
        loss,detail=self.responses.loss(self.model,index)
        gradients=torch.autograd.grad(loss,tuple(named.values()),allow_unused=True)
        raw={n:g.detach() for (n,p),g in zip(named.items(),gradients) if g is not None}
        shared=tuple(n for n in named if any(v in n for v in
            ("module_encoder.","environment_encoder.","operating_encoder.","context_cycles.","fusion_trunk.","field_head.")))
        balance=gradient_balance(native,raw,shared)
        a,b=balance["native_norm"],balance["term_norm"]
        if calibrate:
            self.state["response_coefficient"]=0.1*a/b if a>0 and b>0 else 0.
            self.state.setdefault("response_calibrations",[]).append({"epoch":int(self.model.core.organization_epoch),
                "balance":balance,"coefficient":self.state["response_coefficient"],"kind":detail["kind"]})
        coefficient=self.state.get("response_coefficient",0.)
        cap=min(1.,.1*a/(coefficient*b)) if coefficient*b>0 else 1.
        return loss,detail,raw,coefficient*cap,{"native_shared_norm":a,"raw_shared_norm":b,
            "raw_full_norm":gradient_norm(raw),"coefficient":coefficient,"cap_scale":cap}

    def before_optimizer_step(self, model, optimizer, losses, work, *, epoch, update_index):
        if self.native_only:
            self.state["primary_updates"] = (
                (int(epoch) - 1) * math.ceil(len(self.training_cases) / self.effective_batch_size)
                + int(update_index) + 1
            )
            self._check_wall_clock_budget()
            return
        named=dict(model.named_parameters())
        native={n:p.grad.detach().clone() for n,p in named.items() if p.grad is not None}
        if update_index==0 and epoch in (1,21,100):
            self.organizer.calibrate(self,epoch,optimizer=optimizer,panel_runner=self._calibration_batches)
            # A no-step response calibration uses existing admitted TRAIN labels.
            _,detail,_,_,_=self._response_gradient(native,index=0 if epoch==1 else 1,calibrate=True)
            self.state["calibration_response_scene_visits"]=self.state.get("calibration_response_scene_visits",0)+2
        primary_update=(epoch-1)*math.ceil(len(self.training_cases)/self.effective_batch_size)+update_index+1
        maximum=int(self.recipe.get("objective",{}).get("functional_comparison_max_per_arm",100))
        if primary_update%4==0 and self.state["functional_comparisons"]<maximum:
            from .contextual_joint_sharing import functional_probe
            terms,detail=functional_probe(self,self.state["functional_comparisons"],epoch)
            for name,term in terms.items():
                self.organizer.add_raw_gradients(name,term["bank"],term["gradients"],
                    samples=term["samples"],objective_value=term["objective_value"])
            self.state.setdefault("functional_records",[]).append(detail)
            self.state["functional_comparisons"]+=1
            work["functional_comparisons"]=1
            work["functional_scene_visits"]=detail.get("scene_visits",0)
            work["functional_native_forward_calls"]=detail.get("native_forward_calls",0)
        self.organizer.before_optimizer_step(model,optimizer,losses,work,epoch=epoch,update_index=update_index)
        if primary_update%4==0:
            loss,detail,gradients,scale,balance=self._response_gradient(native,index=primary_update//4-1)
            for name,gradient in gradients.items():
                contribution=gradient*scale
                if named[name].grad is None:
                    named[name].grad=contribution.clone()
                else:
                    named[name].grad.add_(contribution)
            self.state["response_microbatches"]+=1
            self.state["response_scene_visits"]+=2
            exposure=detail.get("exposure",{})
            if not exposure and detail.get("pairs"):
                exposure=detail["pairs"][0].get("exposure",{})
            self.state.setdefault("response_records",[]).append({"update":primary_update,"kind":detail["kind"],
                "loss":float(loss.detach()),"exposure":exposure,"gradient":balance})
            losses["response"]=float(loss.detach())*scale
            work["response_microbatches"]=1
            work["response_scene_visits"]=2
        self.state["primary_updates"]=primary_update
        self._check_wall_clock_budget()

    def _check_wall_clock_budget(self):
        if self._budget_binding is not None:
            from honf_runtime.unified_training import _atomic_json
            output,budget,reserve=self._budget_binding
            elapsed=self.state.get("training_elapsed_seconds",0.)+time.monotonic()-self._segment_started
            if elapsed>=budget-reserve:
                path=output/"CLEAN_STOP_REQUEST.json"
                if not path.exists():
                    _atomic_json(path,{"schema":"ordinary_clean_stop_request_v1",
                        "request_id":"contextual-frontier-budget","requested_unix":time.time(),
                        "requested_by":"internal_contextual_budget","reason":"finite training wall budget reached"})

    def bind_wall_clock_budget(self, *, output_dir, budget_seconds, saver_reserve_seconds=600):
        self._budget_binding=(Path(output_dir),float(budget_seconds),float(saver_reserve_seconds))
        self._segment_started=time.monotonic()

    def training_state_dict(self):
        state=copy.deepcopy(self.state)
        if not self.native_only:
            state["organizer"]=self.organizer.state_dict()
        state["training_samples"]=self.training_samples
        state["validation_samples"]=self.validation_samples
        if self._budget_binding is not None:
            state["training_elapsed_seconds"]=state.get("training_elapsed_seconds",0.)+time.monotonic()-self._segment_started
        return state

    def load_training_state_dict(self, state):
        elapsed=self.state.get("training_elapsed_seconds",0.)
        if self._budget_binding is not None:
            elapsed+=time.monotonic()-self._segment_started
        self.state = copy.deepcopy(state)
        self.state["training_elapsed_seconds"]=max(elapsed,self.state.get("training_elapsed_seconds",0.))
        self.training_samples=int(self.state.pop("training_samples",self.training_samples))
        self.validation_samples=int(self.state.pop("validation_samples",self.validation_samples))
        if "organizer" in self.state and self.organizer is not None:
            self.organizer.load_state_dict(self.state.pop("organizer"))
        self._segment_started=time.monotonic()
        self._pending = []


def build_thermal_contextual_frontier_task(config, *, arm=None, device="cpu"):
    recipe = copy.deepcopy(dict(config))
    dataset = recipe.get("dataset", {})
    if not isinstance(dataset, dict):
        dataset = {}
    manifest, data_path, manifest_path = _load_data_binding(dataset.get("manifest_path"), dataset.get("data_path"))
    source, catalog = read_case_catalog(data_path)
    full = recipe.get("training", {}).get("membership") in ("full_train600", "scratch_train600_exposed_val89")
    if full:
        train_ids = tuple(str(r["case_id"]) for r in catalog if r["split"] == "train")
        val_ids = tuple(str(r["case_id"]) for r in catalog if r["split"] == "test" and r["case_id"] != "0273")
        validate_full_train_followup_membership(train_ids, val_ids)
    else:
        train_ids, val_ids = tuple(development_case_ids(manifest,"train")), tuple(development_case_ids(manifest,"test"))
    normalizer = fit_global_normalizer(data_path, train_ids)
    training = _read_selected_cases(data_path,train_ids,split="train",normalizer=normalizer,include_grid=False)
    validation = _read_selected_cases(data_path,val_ids,split="test",normalizer=normalizer,include_grid=False)
    heat_values = np.concatenate([np.asarray(c["structure"]["heat_powers"])[np.asarray(c["structure"]["module_present"])>0.5] for c in training])
    scale = max(float(np.sqrt(np.mean(heat_values.astype(np.float64)**2))),1e-6)
    core_config = {"module_input_dim":9,"environment_input_dim":6,"operating_input_dim":12,
        "coordinate_dim":2,"query_input_dim":0,"output_dim":4,"context_dim":128,
        "num_heads":4,"context_depth":2,"init_seed":0,**recipe.get("contextual_frontier", {}).get("model", recipe.get("model", {})),
        "architecture":arm or recipe.get("architecture","J-H"), "schema":"contextual_organization_joint_v1"}
    model = ThermalContextualJointAdapter(normalizer.stats,core_config=core_config,heat_scale=scale).to(device)
    provider = ThermalContextualJointTask(model=model,training_cases=training,validation_cases=validation,
        manifest=manifest,data_path=data_path,recipe=recipe,device=device,manifest_path=manifest_path,
        catalog=catalog,source_binding={"dataset_path":str(data_path),"existing_catalog":source})
    return model,provider

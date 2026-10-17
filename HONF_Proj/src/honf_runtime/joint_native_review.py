"""Bounded, input-selected TRAIN routing reviews with exact checkpointed history."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),default=repr).encode()).hexdigest()


def _tensor_digest(*values):
    digest=hashlib.sha256()
    for value in values:
        if value is None:
            digest.update(b"None");continue
        array=value.detach().cpu().contiguous().numpy()
        digest.update(repr((array.shape,array.dtype.str)).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def _query_input_digest(value):
    """Hash receiver/control inputs recursively without copying target objects."""
    if torch.is_tensor(value):return _tensor_digest(value)
    if isinstance(value,np.ndarray):return hashlib.sha256(value.tobytes()).hexdigest()
    if isinstance(value,dict):return _digest({str(k):_query_input_digest(v) for k,v in value.items()})
    if isinstance(value,(tuple,list)):return _digest([_query_input_digest(v) for v in value])
    if hasattr(value,"__dict__"):return _query_input_digest(vars(value))
    return repr(value)


def _pack(mask):
    array=mask.detach().cpu().numpy().astype(bool,copy=False)
    return {"shape":tuple(array.shape),"bits":torch.from_numpy(np.packbits(array.reshape(-1)).copy())}


def _unpack(value):
    count=int(np.prod(value["shape"]))
    return np.unpackbits(value["bits"].cpu().numpy(),count=count).reshape(value["shape"]).astype(bool)


def _changed(current,previous):
    if previous is None:return None
    if tuple(current["shape"])!=tuple(previous["shape"]):
        raise ValueError("Frozen TRAIN review support axes changed.")
    a,b=_unpack(current),_unpack(previous)
    return {"added":int((a & ~b).sum()),"removed":int((~a & b).sum()),"changed":int((a^b).sum())}


def _loss_changes(current,previous):
    return {name:{"value":value,"previous":None if previous is None else previous.get(name),
                  "delta":None if previous is None or name not in previous else value-previous[name]}
            for name,value in current.items()}


class JointNativeTrainReview:
    """Every-25-epoch diagnostics, never a validation selector or optimizer policy.

    Previous packed masks live in provider checkpoint state. An exact resume
    therefore compares against its own last saved review, even when an output
    directory contains a later review from an interrupted segment.
    """

    schema="joint_native_train_review_v1"

    def __init__(self, *, panel_cases=4, query_epoch=3001, threshold_probability_band=(.45,.55)):
        if not 1<=panel_cases<=4 or query_epoch!=3001:
            raise ValueError("Round9 review uses at most four fixed TRAIN scenes and e3001 queries.")
        if not 0<threshold_probability_band[0]<.5<threshold_probability_band[1]<1:
            raise ValueError("Review threshold band must straddle the hard probability threshold.")
        self.panel_cases=int(panel_cases);self.query_epoch=int(query_epoch)
        self.threshold_probability_band=tuple(map(float,threshold_probability_band))
        self.panel=None;self.previous=None

    def state_dict(self):
        return {"schema":self.schema,"panel_cases":self.panel_cases,"query_epoch":self.query_epoch,
                "threshold_probability_band":self.threshold_probability_band,
                "panel":copy.deepcopy(self.panel),"previous":copy.deepcopy(self.previous)}

    def load_state_dict(self,state):
        if (state.get("schema")!=self.schema or state.get("panel_cases")!=self.panel_cases
                or state.get("query_epoch")!=self.query_epoch
                or tuple(state.get("threshold_probability_band",()))!=self.threshold_probability_band):
            raise ValueError("TRAIN review checkpoint has a different fixed panel policy.")
        self.panel=copy.deepcopy(state["panel"]);self.previous=copy.deepcopy(state["previous"])

    def _select_panel(self,provider,engine):
        cases,namespace=engine.epoch_case_order(provider,self.query_epoch)
        counts={}
        if hasattr(provider,"training_cases"):
            for case in provider.training_cases:
                counts[str(case["case_id"])]=int(np.asarray(case["structure"]["module_present"]).sum())
        elif hasattr(provider,"train_rows") and hasattr(provider,"view"):
            counts={int(row):int(provider.view.metadata["n_turbines"][int(row)]) for row in provider.train_rows}
        selected=[]
        if counts:
            strata=sorted({counts[key] for key in cases})
            chosen=list(dict.fromkeys(strata[index] for index in (0,min(1,len(strata)-1),len(strata)//2,-1)))
            for count in chosen:
                selected.append(min((key for key in cases if counts[key]==count),key=lambda key:str(key)))
        for key in sorted(cases,key=lambda key:str(key)):
            if len(selected)>=self.panel_cases:break
            if key not in selected:selected.append(key)
        selected=selected[:self.panel_cases]
        return {"case_keys":selected,"module_counts":[counts.get(key) for key in selected],
                "sampling_dataset_id":namespace,"query_epoch":self.query_epoch,"query_update_index":0,
                "selection":"input module-count min/next/median/max strata and stable original ID; no errors or targets"}

    def _fixed_batches(self,provider,engine):
        selected=self._select_panel(provider,engine)
        if self.panel is None:self.panel=selected
        elif self.panel!=selected:raise ValueError("TRAIN review panel/source membership changed on resume.")
        return engine.prepare_update_microbatches(provider,arm="A2-D",phase="joint",epoch=self.query_epoch,
            update_index=0,update_cases=self.panel["case_keys"],sampling_dataset_id=self.panel["sampling_dataset_id"])

    def _evaluate_panel(self,model,provider,engine,batches,epoch):
        from honf_forward_core.joint_native.budget import expected_typed_work,observed_typed_work,direct_typed_work
        from honf_forward_core.joint_native.contracts import JointNativeDirectory
        core=getattr(model,"core",model);router=core.router
        previous_route=router.__dict__.get("route");had_route="route" in router.__dict__;original_route=router.route
        captures={};case_work={"M":{},"E":{}};terms={};denominators={};weights={}
        threshold={"M":[0,0],"E":[0,0]};seen_graphs=set()
        def route(_self,directory,raw_M,raw_E,protected_M,protected_E,**options):
            plan=original_route(directory,raw_M,raw_E,protected_M,protected_E,**options)
            records=directory.records if isinstance(directory,JointNativeDirectory) else directory
            qmask=options.get("query_mask")
            if qmask is None:qmask=torch.ones(plan.hard_support.shape[:2],device=raw_M.device,dtype=torch.bool)
            protected=torch.cat((protected_M,protected_E),-1)
            relaxed=(expected_typed_work(directory,plan.dispatch.probabilities,protected,query_mask=qmask)
                     if isinstance(directory,JointNativeDirectory) else direct_typed_work(plan,records,query_mask=qmask))
            hard=observed_typed_work(plan,records,query_mask=qmask)
            # Geometry/IDs form the scene identity, preserving the distinction
            # between different scenes that happen to have the same donor IDs.
            catalogue=_tensor_digest(records.ids,records.types,records.present,records.coordinates,records.lengths)
            query_key=context["batch"]+":"+str(context["route_index"])
            context["route_index"]+=1
            entry={"catalogue":catalogue,"typed_ids":records.ids.detach().cpu().clone(),
                   "module_count":records.module_count,"M_support":_pack(plan.M_mask),"E_support":_pack(plan.E_mask)}
            if options.get("query_ids") is not None:
                entry["query_ids"]=options["query_ids"].detach().cpu().clone()
            if isinstance(directory,JointNativeDirectory):
                entry.update(M_incidence=_pack(directory.membership_hard[...,:records.module_count]),
                             E_incidence=_pack(directory.membership_hard[...,records.module_count:]))
                graph_identity=context["batch"]+":"+catalogue
                if graph_identity not in seen_graphs:
                    valid=records.present[:,:,None] & records.eligible[:,None] & ~directory.mandatory_membership
                    for kind,sl in (("M",slice(None,records.module_count)),("E",slice(records.module_count,None))):
                        probability=directory.membership_probabilities[...,sl];eligible=valid[...,sl]
                        threshold[kind][0]+=int(((probability>=self.threshold_probability_band[0]) &
                            (probability<=self.threshold_probability_band[1]) & eligible).sum())
                        threshold[kind][1]+=int(eligible.sum())
                    seen_graphs.add(graph_identity)
            else:
                valid=qmask[...,None] & records.eligible[:,None] & ~plan.mandatory_support
                for kind,sl in (("M",slice(None,records.module_count)),("E",slice(records.module_count,None))):
                    probability=plan.probabilities[...,sl];eligible=valid[...,sl]
                    threshold[kind][0]+=int(((probability>=self.threshold_probability_band[0]) &
                        (probability<=self.threshold_probability_band[1]) & eligible).sum())
                    threshold[kind][1]+=int(eligible.sum())
            captures[query_key]=entry
            for kind in ("M","E"):
                for index,key in enumerate(context["case_keys"]):
                    values=(float(relaxed[kind].case_expected_rows[index]),float(hard[kind].case_expected_rows[index]),
                            float(relaxed[kind].case_possible_rows[index]),float(relaxed[kind].case_floor_rows[index]),
                            float(hard[kind].case_floor_rows[index]))
                    case_work[kind][key]=tuple(a+b for a,b in zip(case_work[kind].get(key,(0.,)*5),values))
            return plan
        context={}
        router.route=MethodType(route,router)
        try:
            for index,batch in enumerate(batches):
                context.update(batch=str(index),route_index=0,case_keys=tuple(map(repr,batch.case_keys)))
                scene=provider.make_scene(batch.scene_inputs)
                predictions,auxiliary=provider.predict_native(model,scene,batch.receivers,"A2-D","joint",
                    epoch=epoch,temperature=engine.config.temperature_for_epoch(epoch))
                for name,term in provider.loss_terms(predictions,batch.targets,"joint",auxiliary).items():
                    terms[name]=terms.get(name,0.)+float(term.numerator)
                    denominators[name]=denominators.get(name,0.)+float(term.denominator)
                    weights[name]=float(term.weight)
                del predictions,auxiliary,scene
        finally:
            if had_route:router.route=previous_route
            else:del router.route
        work={}
        for kind in ("M","E"):
            values=list(case_work[kind].values());valid=[v for v in values if v[2]>0]
            work[kind]={"hard_native_fraction":float(np.mean([v[1]/v[2] for v in valid])) if valid else None,
                "relaxed_native_fraction":float(np.mean([v[0]/v[2] for v in valid])) if valid else None,
                "hard_removable_retention":float(np.mean([(v[1]-v[4])/(v[2]-v[4]) for v in valid if v[2]>v[4]])) if any(v[2]>v[4] for v in valid) else None,
                "relaxed_removable_retention":float(np.mean([(v[0]-v[3])/(v[2]-v[3]) for v in valid if v[2]>v[3]])) if any(v[2]>v[3] for v in valid) else None,
                "hard_rows":sum(v[1] for v in valid),"possible_rows":sum(v[2] for v in valid),
                "hard_floor_rows":sum(v[4] for v in valid),"relaxed_floor_rows":sum(v[3] for v in valid),"case_count":len(valid),
                "near_threshold_nonmandatory_fraction":threshold[kind][0]/threshold[kind][1] if threshold[kind][1] else None,
                "near_threshold_pairs":threshold[kind][0],"nonmandatory_pairs":threshold[kind][1]}
        losses={name:terms[name]/denominators[name] if denominators[name] else 0. for name in terms}
        return {"work":work,"losses":losses,"loss_denominators":denominators,"loss_weights":weights,"captures":captures}

    @torch.no_grad()
    def review_epoch(self,model,provider,engine,epoch,layout,*,training_row=None):
        from .unified_training import capture_rng_state,restore_rng_state
        started=time.perf_counter();core=getattr(model,"core",model)
        if core.routing_mode not in ("joint","direct") or epoch<3001 or (epoch-3000)%25:
            raise ValueError("TRAIN routing reviews require an H/S sparse-age multiple of 25.")
        if self.previous is not None and epoch<=self.previous["epoch"]:
            raise ValueError("TRAIN review ages must advance beyond the checkpointed previous review.")
        rng=capture_rng_state();was_training=model.training;entries=core._joint_entries
        counters={name:copy.deepcopy(getattr(provider,name)) for name in (
            "training_samples","training_queries","physical_training_samples","physical_training_queries",
            "validation_samples","validation_queries") if hasattr(provider,name)}
        clocks=(core._joint_epoch,core._joint_update_index,core._joint_macro_clock)
        try:
            batches=self._fixed_batches(provider,engine)
            query_hashes=[batch.auxiliary.get("query_sampling_sha256") if hasattr(batch.auxiliary,"get") else None for batch in batches]
            receiver_hashes=[_query_input_digest(batch.receivers) for batch in batches]
            identity=_digest({"panel":self.panel,"query_sampling_sha256":query_hashes,"receiver_inputs":receiver_hashes})
            if self.previous is not None and self.previous["identity"]!=identity:
                raise ValueError("TRAIN review fixed query identity changed across milestones.")
            model.eval();current=self._evaluate_panel(model,provider,engine,batches,epoch)
            changes={};cliffs=[]
            for key,capture in current["captures"].items():
                previous=None if self.previous is None else self.previous["captures"].get(key)
                if self.previous is not None and (previous is None or previous["catalogue"]!=capture["catalogue"]):
                    raise ValueError("TRAIN review geometry/type catalogue changed.")
                changes[key]={name:_changed(value,None if previous is None else previous.get(name))
                              for name,value in capture.items() if name.endswith(("_support","_incidence"))}
                if previous is not None:
                    for name,delta in changes[key].items():
                        current_mask=_unpack(capture[name]);old_mask=_unpack(previous[name])
                        examples=[]
                        for batch_index,row,member in np.argwhere(current_mask^old_mask)[:12]:
                            absolute_member=member+(capture["module_count"] if name.startswith("E_") else 0)
                            examples.append({"batch_index":int(batch_index),
                                "receiver_or_seed_axis":int(row),"member_type":name[0],
                                "member_native_id":int(capture["typed_ids"][batch_index,absolute_member]),
                                "added":bool(current_mask[batch_index,row,member])})
                            if name.endswith("_support") and "query_ids" in capture:
                                examples[-1]["receiver_native_id"]=int(capture["query_ids"][batch_index,row])
                            if name.endswith("_incidence"):
                                examples[-1]["seed_type"]="M" if row<capture["module_count"] else "E"
                                examples[-1]["seed_native_id"]=int(capture["typed_ids"][batch_index,row])
                        delta["changed_ID_examples"]=examples
            for kind in ("M","E"):
                before=None if self.previous is None else self.previous["work"][kind]["hard_native_fraction"]
                now=current["work"][kind]["hard_native_fraction"]
                delta=None if before is None or now is None else now-before
                current["work"][kind]["change_since_previous_review"]=delta
                if delta is not None and abs(delta)>.15:cliffs.append(kind)
            replay=None
            if cliffs:
                repeated=self._evaluate_panel(model,provider,engine,batches,epoch)
                same=(current["losses"]==repeated["losses"] and all(
                    torch.equal(value["bits"],repeated["captures"][key][name]["bits"])
                    for key,capture in current["captures"].items() for name,value in capture.items()
                    if name.endswith(("_support","_incidence"))))
                replay={"performed":True,"same_batch_identity":identity,"support_and_native_terms_repeat_exactly":same,
                        "policy_change":False,"optimizer_updates":0}
                if not same:raise ValueError("Fixed TRAIN cliff replay did not reproduce deterministic support/terms.")
            duals={kind:{"lambda":float(dual.dual_lambda),"lambda_max":float(dual.lambda_max),
                "at_cap":bool(dual.lambda_max>0 and dual.dual_lambda>=dual.lambda_max),"phase":dual.phase,
                "successful_macros":int(dual.macro_step),"observed_macros":int(dual.observed_macro_count),
                "at_cap_macros":int(dual.at_cap_macro_count)} for kind,dual in core.joint_budget.duals.items()}
            for kind,row in duals.items():
                previous=None if self.previous is None else self.previous.get("duals",{}).get(kind)
                observed=row["observed_macros"]-(0 if previous is None else previous["observed_macros"])
                capped=row["at_cap_macros"]-(0 if previous is None else previous["at_cap_macros"])
                row["interval_observed_macros"]=observed;row["interval_at_cap_macros"]=capped
                row["interval_cap_occupancy"]=capped/observed if observed else None
            epoch_losses={} if training_row is None else {k:float(v) for k,v in training_row.get("train_losses",{}).items() if not k.startswith("joint_work_penalty")}
            summary={"schema":self.schema,"epoch":int(epoch),"sparse_age":epoch-3000,"panel":self.panel,
                "panel_query_identity":identity,"query_sampling_sha256":query_hashes,"receiver_input_sha256":receiver_hashes,
                "scope":"fixed exposed TRAIN input-selected panel; primary and native auxiliary roles; no DEV/TEST selector",
                "work_probability_interpretation":"unnoised unit-Bernoulli/categorical marginals; a labelled relaxation outside the unit-noise phase",
                "near_threshold_probability_band":self.threshold_probability_band,"work":current["work"],
                "changed_original_ID_support":changes,"duals":duals,
                "native_panel_loss_terms":_loss_changes(current["losses"],None if self.previous is None else self.previous["losses"]),
                "native_panel_loss_denominators":current["loss_denominators"],"native_panel_loss_weights":current["loss_weights"],
                "native_epoch_loss_terms":_loss_changes(epoch_losses,None if self.previous is None else self.previous.get("epoch_losses",{})),
                "retention_cliff_types":cliffs,"retention_cliff_threshold":.15,"fixed_batch_replay":replay,
                "policy_reset_or_cutoff":False,"optimizer_updates":0,
                "elapsed_seconds_before_receipt":time.perf_counter()-started}
            path=Path(layout.write_path(f"joint_train_review_e{epoch:04d}.json",category="logs"));path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
            self.previous={"epoch":int(epoch),"identity":identity,"captures":current["captures"],
                           "work":current["work"],"losses":current["losses"],"epoch_losses":epoch_losses,"duals":duals}
            return {"path":str(path),"epoch":int(epoch),"panel_query_identity":identity,
                    "work":current["work"],"retention_cliff_types":cliffs,"duals":duals,
                    "elapsed_seconds":time.perf_counter()-started}
        finally:
            core._joint_entries=entries;model.train(was_training);restore_rng_state(rng)
            for name,value in counters.items():setattr(provider,name,value)
            if clocks!=(core._joint_epoch,core._joint_update_index,core._joint_macro_clock):
                raise ValueError("TRAIN review changed a saved scientific training clock.")

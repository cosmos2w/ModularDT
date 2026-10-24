"""A bounded executed parent-stop versus descendant-frontier intervention."""
from dataclasses import replace
import torch


def functional_probe(provider, index, epoch, *, calibrating=False):
    """Return raw functional/risk VJPs and observations, preserving source slots.

    Both branches reuse the same contextual tensors and candidate forest. Only
    one subtree's live directory changes; native normalization is recomputed.
    This measures model dependence, not physical causality.
    """
    from honf_runtime.unified_training import SamplingKey
    from honf_runtime.thermal_soft_organization import select_calibration_panel
    model=provider.model
    core=model.core
    # The matched leaf model receives the same supervised probes and exposure;
    # its ordinary predictions still force every frontier to the leaves.
    panel=select_calibration_panel(provider.training_cases,limit=min(96,len(provider.training_cases)))
    bank="module" if index%2==0 else "environment"
    if bank=="module":
        counts={str(c["case_id"]):int((torch.as_tensor(c["structure"]["module_present"])>.5).sum())
                for c in provider.training_cases}
        panel=tuple(case_id for case_id in panel if counts[case_id]>=2)
    if not panel:
        return {},{"bank":bank,"skipped":"no input-selected legal internal candidates"}
    case_id=panel[(index//2)%len(panel)]
    key=SamplingKey(0,epoch,0,0,"joint",model.architecture,
        sampling_version=provider.sampling_version,dataset_id=provider.sampling_dataset_id)
    old_clock=(int(core.organization_epoch),int(core.organization_update))
    old_modes={module:module.training for module in model.modules()}
    try:
        model.eval()
        batch=provider.make_batch((case_id,),key)
        batch=replace(batch,
            receivers=replace(batch.receivers,fluid_xy=batch.receivers.fluid_xy[:,:64],
                local_query_points=batch.receivers.local_query_points[...,:8,:]),
            targets=replace(batch.targets,field_targets=batch.targets.field_targets[:,:64],
                point_weights=batch.targets.point_weights[:,:64],material_targets=batch.targets.material_targets[...,:8]))
        predictions,_=provider.predict_native(model,batch.scene_inputs,batch.receivers,model.architecture,"joint",epoch=epoch)
        native=predictions.native_prepared
        prepared=native.organization
        record=(prepared.module_banks if bank=="module" else prepared.environment_banks)[0]
        if record.forest is None or not record.forest.internal_indices:
            return {},{"case_id":case_id,"bank":bank,"skipped":"no legal internal node"}
        # Input-selected depth coverage, without choosing nodes from errors.
        levels={}
        for candidate in record.forest.internal_indices:
            depth=0
            parent=record.forest.nodes[candidate].parent
            while parent is not None:
                depth+=1
                parent=record.forest.nodes[parent].parent
            levels.setdefault(depth,[]).append(candidate)
        depths=sorted(levels)
        depth=depths[(index//2)%len(depths)]
        candidates=sorted(levels[depth])
        node_id=candidates[(index//(2*len(depths)))%len(candidates)]
        coarse=core.intervene_frontier(prepared,bank,0,node_id,stop=True)
        fine=core.intervene_frontier(prepared,bank,0,node_id,stop=False)
        coarse_native=replace(native,organization=coarse)
        fine_native=replace(native,organization=fine)
        coarse_output=model.apply_native(coarse_native,batch.receivers.heat)
        fine_output=model.apply_native(fine_native,batch.receivers.heat)
        coarse_prediction=replace(predictions,native_prepared=coarse_native,native_main=coarse_output)
        fine_prediction=replace(predictions,native_prepared=fine_native,native_main=fine_output)
        # Native observation masks remain the same because geometry is fixed.
        delta_target=replace(batch.targets,field_targets=fine_output["pred_field"].detach(),
            interface_target=fine_output["pred_interface"].detach(),
            material_targets=fine_output["pred_internal_temperature"][...,0].detach())
        differences=provider._role_losses(coarse_prediction,delta_target)
        difference=sum(provider.native_weights[n]*v for n,v in differences.items()).mean()
        native_error=provider._weighted_native(fine_prediction,batch.targets).detach().mean()
        frontier=record.frontier
        temperature=core.frontier_policies[bank].temperature_for_epoch(epoch)
        probability=torch.sigmoid(frontier.stop_logits[node_id]/temperature)
        risk_logit=frontier.risk_logits[node_id].reshape(())
        target=torch.log1p(difference.detach()/native_error.clamp_min(1e-12))
        share=probability.detach()*difference
        risk=(risk_logit-target).square()+probability*target.detach()
        inventory=core.organizer_parameters(bank)
        named=dict(model.named_parameters())
        reverse={id(parameter):name for name,parameter in named.items()}
        parameters=tuple(inventory.values())
        terms={}
        for name,scalar in ((f"share_{bank}",share),(f"risk_{bank}",risk)):
            gradients=torch.autograd.grad(scalar,parameters,retain_graph=True,allow_unused=True)
            terms[name]={"bank":bank,"gradients":{reverse[id(p)]:g.detach() for p,g in zip(parameters,gradients) if g is not None},
                         "samples":1,"objective_value":float(scalar.detach())}
        priority={}
        native_roles=provider._role_losses(fine_prediction,batch.targets)
        for role in ("temperature/fluid","temperature/surface","temperature/material"):
            change=float(torch.sqrt(differences[role].detach().mean()))
            rmse=float(torch.sqrt(native_roles[role].detach().mean()))
            priority[role]={"standardized_change_rms":change,"native_standardized_rmse":rmse,
                            "within_tenth_native_rmse":change<=.1*rmse}
        detail={"case_id":case_id,"bank":bank,"node_id":node_id,"depth":depth,
            "ancestry_ids":list(record.forest.nodes[node_id].ancestry_ids),
            "coarse_K":(coarse.module_banks if bank=="module" else coarse.environment_banks)[0].live_count,
            "fine_K":(fine.module_banks if bank=="module" else fine.environment_banks)[0].live_count,
            "probability":float(probability.detach()),"weighted_difference":float(difference.detach()),
            "risk_target":float(target),"priority":priority,
            "query_rows":int(native.union.unique_counts.sum()),"native_forward_calls":3,
            "scene_visits":1,"encoder_reused_for_intervention":True,"calibrating":calibrating}
        return terms,detail
    finally:
        core.set_organization_clock(*old_clock)
        for module,mode in old_modes.items():
            module.training=mode

#!/usr/bin/env python3
"""Extract frozen full-mode Thermal features and calibrate R15-6 FIT probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "Case_ThermalChannel" / "src")]

import numpy as np
import torch

from channelthermal.interaction_evidence.response_atlas import load_response_atlas_stencil
from channelthermal.inverse.relation_anchored import ThermalDesignEvaluator
from channelthermal.response_control.transfer_probe import (FIT, DEV, FEATURES, atlas_edits,
    organization_features, feature_row, fit_three_arms, evaluate_probe)
from honf_forward_core.content_hypergraph.query_chunks_v1 import select_query_rows_v1
from honf_runtime.contextual_frontier_evaluation import load_contextual_checkpoint
from honf_runtime.compat import load_trusted_checkpoint


def numpy(t):
    return t.detach().cpu().numpy()


def capture_pressure_trace(model, prepared, pressure_weights):
    """Capture actual full-mode access at signed native pressure-gradient queries."""
    selected = torch.nonzero(pressure_weights != 0).flatten()
    if not len(selected):
        raise ValueError("No pressure-objective native query contribution")
    query = select_query_rows_v1(prepared.query_batch, selected.tolist())
    with torch.no_grad():
        trace = model.core.decode(prepared.organization, query, return_trace=True,
                                  intervention={"read_mode": "full"}).trace
    if trace is None or "member_weights_M" not in trace or "member_weights_E" not in trace:
        raise ValueError("Full pressure query trace is unavailable")
    return selected, query, trace


def extract_layout(model, record, rows, *, heat_scale, rng):
    """Fresh full-mode prepare; derivatives and actual full-directory query trace."""
    evaluator = ThermalDesignEvaluator(model, record)
    original_prepare = model.prepare_native
    def traced_prepare(*args, **kwargs):
        kwargs['capture_context_trace'] = True
        return original_prepare(*args, **kwargs)
    model.prepare_native = traced_prepare
    started = time.perf_counter()
    try:
        centers = evaluator._tensor([m.position_xy for m in record.design.active_modules]).requires_grad_(True)
        result = evaluator.evaluate_design(centers, read_mode='full')
    finally:
        model.prepare_native = original_prepare
    grad_t, = torch.autograd.grad(result.objective, centers, retain_graph=True)
    grad_p, = torch.autograd.grad(result.pressure_drop, centers, retain_graph=True)
    native_pressure_gradient, = torch.autograd.grad(result.pressure_drop, result.fields["pred_field"], retain_graph=True)
    receiver_weights = native_pressure_gradient[0,:,2].detach()
    receiver_indices = torch.nonzero(receiver_weights != 0).flatten()
    if not len(receiver_indices):
        raise ValueError("No native pressure receivers contribute to the objective")
    pressure_gradient, = torch.autograd.grad(result.pressure_drop, result.core_output.joint_fields, retain_graph=True)
    pressure_weights = pressure_gradient[0,:,2].detach()
    pressure_selected, pressure_query, pressure_trace = capture_pressure_trace(model, result.prepared, pressure_weights)
    query_gradient, = torch.autograd.grad(result.objective, result.core_output.joint_fields)
    weights = query_gradient[0,:,3].detach()
    if bool((weights < -1e-7).any()) or not bool((weights > 0).any()):
        raise ValueError('Objective gradient must give nonnegative native query weights')
    selected = torch.nonzero(weights > 0).flatten()
    query = select_query_rows_v1(result.prepared.query_batch, selected.tolist())
    with torch.no_grad():
        trace = model.core.decode(result.prepared.organization, query, return_trace=True,
                                  intervention={'read_mode':'full'}).trace
    if trace is None or 'member_weights_M' not in trace:
        raise ValueError('Full coalesced query M trace is unavailable')
    org = result.prepared.organization
    mm = org.final_applied_context_stages['mm']['incidence'][0]
    em = org.final_applied_context_stages['em']['incidence'][0]
    m = len(evaluator.ids); e = org.scene.num_environment_tokens
    directories = {}
    for mode in ('full', 'restricted'):
        directory = getattr(org.e_access_directory, mode)
        dense = np.zeros((e,m+e), float)
        for k in range(e):
            row = directory.row(0,k)
            slots = numpy(row.member_index).astype(int)
            dense[k,slots] = numpy(row.access_weight)
        directories[mode] = dense
    pi = directories['full'][:,:m]
    permutation = rng.permutation(m)
    values = result.fields['pred_internal_temperature'][0,:,:,0]
    valid = result.prepared.stencils['material'].valid.reshape(values.shape)
    member_max = torch.stack([values[i][valid[i]].max() for i in range(m)])
    context = record.context.values; heat = np.asarray([a.heating for a in record.design.active_modules])
    ctx = [m, heat.sum(), context['u_in'], context['re'], context['solid_k']/context['fluid_k'],
           context['module_radius']/context['domain_length_y']]
    q = numpy(trace['member_weights_M'][0]); ai = numpy(query.anchor_indices[0]); aw = numpy(query.anchor_weights[0])
    qw = numpy(weights[selected]); measures = numpy(org.scene.environment_measure[0])
    shared = dict(source_ids=evaluator.ids, query_m=q, objective_weights=qw, anchor_indices=ai,
                  anchor_weights=aw, e_measures=measures, directory_m=pi, mm=numpy(mm), em=numpy(em))
    aligned=[]; permuted=[]
    for row in rows:
        d = row['displacement']; indices = [evaluator.ids.index(i) for i in row['member_ids']]
        full_delta = torch.zeros_like(centers)
        full_delta[indices] = evaluator._tensor(d)
        forward = [result.objective.item(), result.pressure_drop.item(), float((grad_t*full_delta).sum()),
                   float((grad_p*full_delta).sum()), *numpy(member_max[indices]).tolist()]
        kw = dict(member_ids=row['member_ids'], displacement=d, **shared)
        aligned.append(feature_row(row, radius=context['module_radius'], heat_scale=heat_scale, context=ctx,
                                   forward=forward, organization=organization_features(**kw)))
        permuted.append(feature_row(row, radius=context['module_radius'], heat_scale=heat_scale, context=ctx,
                                    forward=forward, organization=organization_features(**kw,permutation=permutation)))
    saved = dict(pressure_receiver_indices=numpy(receiver_indices),
                 pressure_receiver_coordinates=numpy(evaluator.fluid[0,receiver_indices]),
                 pressure_receiver_signed_derivative=numpy(receiver_weights[receiver_indices]),
                 pressure_receiver_prediction=numpy(result.fields["pred_field"][0,receiver_indices,2]),
                 pressure_query_indices=numpy(pressure_selected),
                 pressure_query_coordinates=numpy(pressure_query.coordinates[0]),
                 pressure_signed_query_derivative=numpy(pressure_weights[pressure_selected]),
                 pressure_query_m=numpy(pressure_trace["member_weights_M"][0]),
                 pressure_query_e=numpy(pressure_trace["member_weights_E"][0]),
                 pressure_anchor_indices=numpy(pressure_query.anchor_indices[0]),
                 pressure_anchor_weights=numpy(pressure_query.anchor_weights[0]),
                 query_indices=numpy(selected), objective_query_weights=qw, query_m=q,
                 query_e=numpy(trace['member_weights_E'][0]),
                 directory_full=directories['full'], directory_restricted=directories['restricted'], anchor_indices=ai,
                 anchor_weights=aw, e_measures=measures, directory_m=pi, mm=numpy(mm), em=numpy(em),
                 permutation=permutation, module_ids=np.array(evaluator.ids),
                 environment_ids=numpy(org.scene.environment_ids[0]), module_coordinates=numpy(org.scene.module_coordinates[0]),
                 environment_coordinates=numpy(org.scene.environment_coordinates[0]), gradient_temperature=numpy(grad_t),
                 gradient_pressure=numpy(grad_p))
    for cycle, stage in enumerate(org.context_stage_traces):
        for route in ('mm','em','ee','me'):
            saved[f'cycle{cycle}_{route}'] = numpy(stage[route+'_applied_incidence'][0])
    return np.stack(aligned),np.stack(permuted),saved,time.perf_counter()-started


def load_probe_model(path, device):
    """Dispatch from the saved recipe; preserve legacy R14 loading."""
    saved = load_trusted_checkpoint(path, map_location="cpu")
    spec = saved.get("experiment_identity", {}).get("recipe", {}).get("thermal_port_response")
    if spec is not None:
        from channelthermal.training.thermal_port_response_task import load_thermal_port_response_checkpoint
        model, _local, payload = load_thermal_port_response_checkpoint(path, device=device)
        return model, payload
    model, _provider, payload = load_contextual_checkpoint(path, device)
    return model, payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True); parser.add_argument('--atlas',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--trace-only', action='store_true', help='Save traces only; perform no probe calibration')
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    model,payload=load_probe_model(args.checkpoint,args.device)
    tau=.05*float(model.internal_temperature_std[0]); beta=1./tau
    stencils={f:load_response_atlas_stencil(args.atlas / f'train_{f}_responses.npz')[0] for f in FIT+DEV}
    rows={f:atlas_edits(args.atlas / f'train_{f}_responses.npz',beta=beta) for f in FIT+DEV}
    diffs=np.array([r['baseline_heating'][0]-r['baseline_heating'][1] for f in FIT for r in rows[f]])
    heat_scale=max(float(np.sqrt(np.mean(diffs**2))),1e-12)
    args.output.mkdir(parents=True)
    aligned=[]; permuted=[]; targets=[]; families=[]; timings={}; rng=np.random.default_rng(0)
    for family in FIT+DEV:
        a,p,evidence,seconds=extract_layout(model,stencils[family].baseline,rows[family],heat_scale=heat_scale,rng=rng)
        np.savez_compressed(args.output/f'{family}_trace.npz',**evidence)
        aligned.extend(a); permuted.extend(p); targets.extend(r['target'] for r in rows[family]); families.extend([family]*8)
        timings[family]=seconds
        print(f'{family}: {seconds:.2f}s',flush=True)
    if args.trace_only:
        (args.output/'trace_summary.json').write_text(json.dumps({'checkpoint':str(args.checkpoint), 'timing_seconds':timings, 'fit_epochs':0, 'stage_provenance':'actual staged MM/EM/EE/ME incidences; full and restricted E directories; full-mode material and pressure query coefficients; signed pressure derivatives'},indent=2)+'\n')
        return
    aligned=np.asarray(aligned); permuted=np.asarray(permuted); targets=np.asarray(targets); families=np.asarray(families)
    fit=np.isin(families,FIT); dev=np.isin(families,DEV)
    scales=[float(model.internal_temperature_std[0]), float(model.field_std[2])]
    probes,losses=fit_three_arms(aligned[fit],permuted[fit],targets[fit],families[fit],scales)
    results={}; predictions={}; histories={}
    from channelthermal.response_control.transfer_probe import probe_history, plot_probe_history
    for arm,probe in probes.items():
        x=aligned.copy() if arm!='permuted' else permuted.copy()
        if arm=='no_organization': x[:,18:]=0
        predictions[arm]=probe.predict(x)
        results[arm]=evaluate_probe(probe,x[dev],targets[dev],families[dev])
        histories[arm]=probe_history(probe, predictions[arm], targets, fit, dev)
        np.savez_compressed(args.output/f'{arm}_probe.npz',mean=probe.mean,scale=probe.scale,target_scale=probe.target_scale,
                            intercept=probe.intercept,coefficients=probe.coefficients,predictions=predictions[arm])
    plot_probe_history(histories, args.output/'probe_loss_history.pdf')
    (args.output/'probe_loss_history.json').write_text(json.dumps(histories, indent=2)+'\n')
    np.savez_compressed(args.output/'features.npz',aligned=aligned,permuted=permuted,targets=targets,families=families,
                        labels=np.array([r['label'] for f in FIT+DEV for r in rows[f]]))
    report={'checkpoint':str(args.checkpoint),'fit_families':FIT,'dev_families':DEV,'features':FEATURES,
            'fit_epochs':3,'beta':beta,'heat_scale_fit':heat_scale,'inherited_scales':scales,'losses':losses,
            'dev_family_results':results,'timing_seconds':timings,'trace_provenance':'full-mode native material and pressure objective queries; final-cycle MM and E<-M; per-layout permutation seed 0',
            'note':'DEV is exposed and development-selected. Pressure request usability not adjudicated.'}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()

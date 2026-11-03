"""Mask-aware held-out finite-response reductions for R15-8."""
from __future__ import annotations

import itertools
import math

import numpy as np


def field_metrics(prediction, target, mask):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError('Prediction, target and physical validity mask must align.')
    if not mask.any() or not np.isfinite(prediction[mask]).all() or not np.isfinite(target[mask]).all():
        raise ValueError('A physical field comparison needs finite, valid receivers.')
    diff = prediction[mask] - target[mask]
    return {'count': int(mask.sum()), 'rmse': float(np.sqrt(np.mean(diff * diff))),
            'mae': float(np.mean(np.abs(diff))), 'prediction_mean': float(prediction[mask].mean()),
            'physical_mean': float(target[mask].mean())}


def fieldwise_flow_diagnostics(baseline, repeated, heat_changed, mask):
    """No-op and heat-null differences on each channel's own native units."""
    base, repeat, heat = (np.asarray(x, dtype=np.float64) for x in (baseline, repeated, heat_changed))
    mask = np.asarray(mask, dtype=bool)
    if base.shape != repeat.shape or base.shape != heat.shape or base.ndim != 2 or base.shape[1] < 4 or mask.shape != base.shape[:1] or not mask.any():
        raise ValueError('Flow diagnostics require aligned nonempty native fluid receivers.')
    output = {}
    for i, (name, units) in enumerate((('u','dataset velocity units'), ('v','dataset velocity units'),
                                        ('p','dataset pressure units'), ('omega','dataset vorticity units'))):
        values = {}
        for label, other in (('zero_edit', repeat), ('heat_only', heat)):
            diff = (other[mask, i] - base[mask, i])
            if not np.isfinite(diff).all():
                raise ValueError('Nonfinite prediction on a valid flow diagnostic receiver.')
            values[label] = {'max_abs': float(np.max(np.abs(diff))),
                             'rmse': float(np.sqrt(np.mean(diff * diff)))}
        output[name] = {'units': units, 'valid_count': int(mask.sum()), **values}
    return output


def summarize_actions(rows, floors, *, family_ids=None):
    """Equal-family all-action and resolved metrics; retain no-resolution coverage."""
    result = {}
    observed = {row['case_id'] for row in rows}
    families = sorted(observed if family_ids is None else set(family_ids))
    if not observed.issubset(families):
        raise ValueError('Observed actions contain a family outside the declared cohort.')
    for key in ('J', 'P'):
        floor = float(floors[key])
        if not math.isfinite(floor) or floor <= 0:
            raise ValueError('TRAIN-derived resolution floors must be positive and finite.')
        family_rows, resolved_rows, near_null_rows = [], [], []
        signs = signs_total = order_wrong = order_total = 0
        all_signs = all_signs_total = all_order_wrong = all_order_total = 0
        for family in families:
            cases = [r for r in rows if r['case_id'] == family]
            if not cases:
                continue
            errors = [abs(r['delta'][key]['model'] - r['delta'][key]['physical']) for r in cases]
            family_rows.append(float(np.mean(errors)))
            resolved = [r for r in cases if abs(r['delta'][key]['physical']) > floor]
            near_null = [r for r in cases if abs(r['delta'][key]['physical']) <= floor]
            if near_null:
                near_null_rows.append(float(np.mean([abs(r['delta'][key]['model']-r['delta'][key]['physical']) for r in near_null])))
            if resolved:
                resolved_rows.append({'case_id': family,
                    'mae': float(np.mean([abs(r['delta'][key]['model']-r['delta'][key]['physical']) for r in resolved])),
                    'mean_effect': float(np.mean([abs(r['delta'][key]['physical']) for r in resolved]))})
            for row in resolved:
                signs += int(np.sign(row['delta'][key]['model']) == np.sign(row['delta'][key]['physical']))
                signs_total += 1
            for row in cases:
                all_signs += int(np.sign(row['delta'][key]['model']) == np.sign(row['delta'][key]['physical']))
                all_signs_total += 1
            for a, b in itertools.combinations(cases, 2):
                physical = a['delta'][key]['physical'] - b['delta'][key]['physical']
                model = a['delta'][key]['model'] - b['delta'][key]['model']
                all_order_wrong += int(np.sign(model) != np.sign(physical)); all_order_total += 1
                if abs(physical) > 2 * floor:
                    order_wrong += int(np.sign(model) != np.sign(physical)); order_total += 1
        resolved_mae = float(np.mean([r['mae'] for r in resolved_rows])) if resolved_rows else None
        effect = float(np.mean([r['mean_effect'] for r in resolved_rows])) if resolved_rows else None
        result[key] = {'floor': floor, 'family_count': len(families),
            'families_with_actions': len(family_rows), 'families_without_actions': len(families)-len(family_rows),
            'all_action_count': all_signs_total, 'all_sign_correct': all_signs,
            'all_order_wrong': all_order_wrong, 'all_order_count': all_order_total,
            'all_action_equal_family_mae': float(np.mean(family_rows)) if family_rows else None,
            'resolved_family_count': len(resolved_rows), 'unresolved_family_count': len(families)-len(resolved_rows),
            'resolved_equal_family_mae': resolved_mae, 'resolved_equal_family_mean_abs_effect': effect,
            'near_null_family_count': len(near_null_rows),
            'near_null_equal_family_mae': float(np.mean(near_null_rows)) if near_null_rows else None,
            'resolved_mae_over_effect': resolved_mae/effect if effect else None,
            'resolved_sign_correct': signs, 'resolved_sign_count': signs_total,
            'resolved_order_wrong': order_wrong, 'resolved_order_count': order_total}
    return result


def intermodule_fluid_mask(fluid_xy, fluid_valid, module_xy):
    """Fluid receivers inside the convex hull of at least three noncollinear centres."""
    from scipy.spatial import Delaunay, QhullError
    xy = np.asarray(fluid_xy, dtype=np.float64)
    valid = np.asarray(fluid_valid, dtype=bool)
    centers = np.asarray(module_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or valid.shape != (len(xy),) or centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError('Native fluid and module coordinates must be aligned 2D points.')
    if len(centers) < 3 or np.linalg.matrix_rank(centers - centers.mean(axis=0)) < 2:
        return np.zeros(len(xy), dtype=bool), 'fewer than three noncollinear module centers'
    try:
        mask = valid & (Delaunay(centers).find_simplex(xy) >= 0)
    except QhullError:
        return np.zeros(len(xy), dtype=bool), 'degenerate convex hull'
    return mask, None if mask.any() else 'no valid fluid receivers within module-center hull'


def evaluate_native_panel(model, provider, saved, case_ids, export_ids=()):
    """Evaluate exposed native validation once, separating direct and local paths.

    Intermodule T uses valid native fluid cells in the convex hull of original
    present module centres; unavailable cases remain visible in coverage.
    """
    import time
    import torch
    from honf_runtime import contextual_frontier_evaluation as frontier
    from honf_runtime.joint_native_evaluation import thermal_record_and_arrays
    from honf_runtime.member_routing_evaluation import _native_arrays_with_residuals, _accumulate_work

    ids = tuple(map(str, case_ids)); export_ids = set(map(str, export_ids))
    if not ids or len(set(ids)) != len(ids) or not export_ids.issubset(ids):
        raise ValueError('A unique nonempty validation panel and in-panel exports are required.')
    # P64's literal file epoch 100 is already included in its inherited age
    # 5100; new finite-response checkpoints use their new-epoch clock.
    finite = saved.get('experiment_identity', {}).get('recipe', {}).get('thermal_finite_response')
    epoch = int(saved['epoch']) if finite else 0
    start = time.perf_counter(); cpu = time.process_time(); model.eval()
    records, direct_rows, assembled_records, extra_rows, exports, work = [], [], [], [], {}, {}
    with torch.no_grad():
        for batch in provider.native_evaluation_batches(split='validation',case_keys=ids,microbatch_size=1):
            case = str(batch.case_keys[0])
            if case != ids[len(records)] or tuple(map(str,batch.targets.case_ids)) != (case,):
                raise ValueError('Native provider changed the declared case order.')
            pred, aux = provider.predict_native(model,batch.scene_inputs,batch.receivers,model.architecture,'validation',epoch=epoch)
            direct, _ = thermal_record_and_arrays(pred,batch)
            fields=direct['fields']; peaks=direct['module_peaks']
            direct_rows.append({'case_id':case,'module_count':direct['module_count'],
                'flow_rmse':{k:fields[k]['rmse'] for k in ('u','v','p','omega')},
                'roles':{**{k:{'rmse':fields[k]['rmse']} for k in ('temperature_fluid','temperature_surface','temperature_material')},
                         'temperature_material_endpoint_maxima':{'per_module':peaks}}, 'q_proxy_rmse':fields['q_proxy']['rmse']})
            assembled = provider.validation_metrics(pred,batch.targets,aux)
            assembled_records.append(assembled)
            prep=pred.native_prepared; target=batch.targets; mode='restricted'
            convert=lambda x: x.detach().cpu().numpy()
            present=convert(prep.source_present[0]).astype(bool)
            fluid=convert(prep.stencils['fluid'].valid[0]).astype(bool) & (convert(target.point_weights[0])>0)
            xy=convert(batch.receivers.fluid_xy[0]); centers=convert(prep.scene_batch.module_coordinates[0])[present]
            hull,hull_reason=intermodule_fluid_mask(xy,fluid,centers)
            field=convert(pred.native_main['pred_field'][0]); field_target=convert(target.field_targets[0])
            local=pred.local1[mode]; teacher=convert(target.teacher_ports[0,...,3])
            outside=convert(pred.outside64[mode][0]); outside_valid=(convert(prep.stencils['outside'].valid[0]).reshape(outside.shape)
                & convert(target.outside_valid[0]).astype(bool) & present[:,None] & np.isfinite(teacher))
            surface_valid=(convert(prep.stencils['surface'].valid[0]).reshape(outside.shape)
                & convert(target.interface_valid64[0]).astype(bool) & present[:,None])
            interface=convert(local['interface_pred'][0]); interface_target=convert(target.interface_target64[0])
            local_surface_valid=surface_valid & np.isfinite(interface_target[...,0])
            local_q_valid=surface_valid & outside_valid & np.isfinite(interface_target[...,1])
            material=convert(local['internal_temperature'][0,...,0]); material_target=convert(target.material_targets[0])
            material_valid=(convert(prep.stencils['material'].valid[0]).reshape(material.shape)
                & present[:,None] & np.isfinite(material_target))
            def available(prediction, reference, mask):
                return field_metrics(prediction,reference,mask) if mask.any() else None
            extra={'case_id':case,'module_count':int(present.sum()),
                'outside_T':available(outside,teacher,outside_valid),
                'local_material_T':available(material,material_target,material_valid),
                'local_surface_T':available(interface[...,0],interface_target[...,0],local_surface_valid),
                'local_q':available(interface[...,1],interface_target[...,1],local_q_valid),
                'intermodule_fluid_T':field_metrics(field[:,4],field_target[:,4],hull) if hull_reason is None else None,
                'intermodule_unavailable_reason':hull_reason}
            extra_rows.append(extra); records.append(direct)
            _accumulate_work(work,provider.work_counts(batch,pred,aux))
            if case in export_ids:
                arrays=_native_arrays_with_residuals(pred,batch,case)
                arrays.update(outside_prediction=outside,outside_target=teacher,outside_valid=outside_valid,
                    local_material_prediction=material,local_material_target=material_target,local_material_valid=material_valid,
                    local_interface_prediction=interface,local_interface_target=interface_target,
                    local_surface_valid=local_surface_valid,local_q_valid=local_q_valid,
                    intermodule_fluid_valid=hull)
                exports[case]=arrays
    if tuple(row['case_id'] for row in records) != ids: raise ValueError('Native panel did not complete its declared membership.')
    assembled=provider.reduce_native_evaluation_metrics(assembled_records,split='validation',case_keys=ids)
    counts=frontier._known_module_counts(provider)
    direct_summary=frontier._summarize_nine_observables(direct_rows,known_module_counts=counts)
    assembled_summary=frontier._summarize_nine_observables(assembled['per_case'],known_module_counts=counts)
    additional={}
    for key in ('outside_T','local_material_T','local_surface_T','local_q','intermodule_fluid_T'):
        values=[(r['case_id'],r['module_count'],r[key]['rmse']) for r in extra_rows if r[key] is not None]
        additional[key]={**frontier._summary(values,unit='dataset heat-flux proxy' if key=='local_q' else 'dataset temperature',
            value_definition='equal-case native receiver RMSE',known_module_counts=counts), 'unavailable_case_count':len(ids)-len(values)}
    return {'split':'validation','case_ids':list(ids),'case_count':len(ids),'metrics':direct_summary,
        'direct_joint_native_metrics':direct_summary,'direct_case_records':records,
        'assembled_local_metrics':assembled_summary,'maintained_native_reduction':assembled,
        'additional_metrics':additional,'additional_case_records':extra_rows,
        'intermodule_definition':'valid native fluid points inside convex hull of at least three noncollinear present module centers; M1/M2 and degenerate hulls unavailable',
        'port_definition':'64-angle stored outside T and frozen-local predictions using the same ordinary restricted native preparation',
        'executed_work_counts':{k:int(v) if float(v).is_integer() else float(v) for k,v in sorted(work.items())},
        'representative_exports':sorted(exports),'representative_arrays':exports,
        'elapsed_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu}

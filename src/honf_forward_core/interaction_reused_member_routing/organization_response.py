"""Bounded, inference-only diagnostics for original-member read changes.

The two-factor decomposition is defined only for the same physical query and
persistent original member identities. It explains the computation, not a
physical cause. Messages for the support union are recomputed from each side's
own prepared state; their cost is reported separately from ordinary decoding.
"""
from __future__ import annotations

from collections.abc import Mapping
import time

import torch

from .anchored import AnchoredMemberRoutingCore


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def contextual_coefficients_trace_request(core):
    """Request actual per-head coefficients for each executed contextual stage."""
    return {'include': [f'cycle{cycle}/{route}_attention'
                        for cycle in range(core.context_depth) for route in ('mm', 'em', 'ee', 'me')]}


def capture_applied_relationships(core, prepared):
    """Copy actual directed stage gates and prepared E-access rows, with IDs.

    Prepare with ``trace_request=contextual_coefficients_trace_request(core)``
    to retain actual per-head reduction coefficients as well as gates. Missing
    coefficients are explicitly unavailable, never reconstructed from gates.
    This never treats a transposed lookup as an executed route. Copies cannot
    change the prepared state; existing version checks apply to each decode.
    """
    core._validate_prepared_state(prepared)
    if not prepared.context_stage_traces:
        raise ValueError('Prepare with return_trace=True to capture actual contextual stages.')
    stages = []
    provenance = {row['stage']: row for row in prepared.context_stage_metadata}
    for cycle in prepared.context_stage_traces:
        for route in ('mm', 'em', 'ee', 'me'):
            snapshot = cycle.get(route + '_native_route_snapshot')
            if snapshot is None:
                continue
            snapshot.assert_current()
            incidence = cycle.get(route + '_applied_incidence')
            if incidence is None:
                incidence = snapshot.applied_incidence
            if incidence is None:
                if snapshot.allowed.numel():
                    raise ValueError(f'No actual applied gate for {snapshot.stage_identity}.')
                incidence = snapshot.raw_scores.new_zeros(snapshot.allowed.shape)
            if snapshot.route != route or not snapshot.stage_identity.endswith('/' + route):
                raise ValueError('Context route and captured stage provenance disagree.')
            trace_key = snapshot.stage_identity + '_attention'
            coefficients = (prepared.trace or {}).get(trace_key)
            if coefficients is not None and (coefficients.ndim != 4
                    or coefficients.shape[0] != incidence.shape[0]
                    or coefficients.shape[2:] != incidence.shape[1:]):
                raise ValueError(f'Actual coefficients have invalid axes for {snapshot.stage_identity}.')
            stages.append({'stage': snapshot.stage_identity, 'route': route,
                           'receiver_role': route[0].upper(), 'donor_role': route[1].upper(),
                           'source_state_version': provenance[snapshot.stage_identity]['source_state_version'],
                           'application': provenance[snapshot.stage_identity]['application'],
                           'receiver_ids': snapshot.destination_ids.detach().clone(),
                           'donor_ids': snapshot.donor_ids.detach().clone(),
                           'allowed': snapshot.allowed.detach().clone(),
                           'gate': incidence.detach().clone(),
                           'coefficients': None if coefficients is None else coefficients.detach().clone(),
                           'message_mass': (None if snapshot.applied_message_mass is None else
                                            snapshot.applied_message_mass.detach().clone())})
    if not stages:
        raise ValueError('The captured scene has no applied contextual stages.')
    access = {}
    access['own_e_member_index'] = prepared.e_access_directory.own_e_member_index.detach().clone()
    for mode in ('full', 'restricted'):
        rows = getattr(prepared.e_access_directory, mode)
        access[mode] = {name: getattr(rows, name).detach().clone() for name in
                        ('offsets', 'member_index', 'original_id', 'is_module', 'access_weight',
                         'applied_exchange', 'structural_self', 'measure')}
    access['provenance'] = {route: {'stage': item['stage_identity'], 'route': item['route'],
                                   'receiver_role': item['receiver_role'], 'donor_role': item['donor_role'],
                                   'receiver_ids': item['destination_ids'].detach().clone(),
                                   'donor_ids': item['donor_ids'].detach().clone()}
                            for route, item in prepared.anchored_route_provenance.items()}
    return {'contextual_stages': stages, 'prepared_e_access': access}


def select_context_relations(relationships, *, seed=0, count=4):
    """Select high, low and seeded controls by applied mass, never target error.

    Mass is the mean of the actual per-head reduction coefficients for this
    directed edge. One scene is supported. Nonzero eligible relations are
    sorted deterministically by mass and original IDs; seeded controls are
    sampled from the remaining set.
    """
    if not 1 <= count <= 4:
        raise ValueError('Choose between one and four contextual relations.')
    candidates = []
    for stage in relationships['contextual_stages']:
        if stage['gate'].shape[0] != 1:
            raise ValueError('Relation selection requires one scene.')
        if stage['coefficients'] is None:
            if not bool((stage['allowed'] & (stage['gate'] > 0)).any()):
                continue
            raise ValueError('Request actual contextual coefficients before selecting relation mass.')
        mass = stage['coefficients'][0].mean(0)
        positive = (stage['allowed'][0] & (stage['gate'][0] > 0) & (mass > 0)).nonzero().cpu().tolist()
        for receiver, donor in positive:
            candidates.append({'stage': stage['stage'], 'route': stage['route'],
                               'destination_id': int(stage['receiver_ids'][0, receiver]),
                               'donor_id': int(stage['donor_ids'][0, donor]),
                               'gate': float(stage['gate'][0, receiver, donor]),
                               'mean_coefficient': float(mass[receiver, donor])})
    candidates.sort(key=lambda x: (x['mean_coefficient'], x['stage'], x['destination_id'], x['donor_id']))
    if not candidates:
        return []
    chosen = [(len(candidates) - 1, 'high_mass')]
    if count > 1 and len(candidates) > 1:
        chosen.append((0, 'low_mass'))
    remaining = [i for i in range(len(candidates)) if i not in {x[0] for x in chosen}]
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    if remaining:
        order = torch.randperm(len(remaining), generator=generator).tolist()
        chosen.extend((remaining[i], 'seeded_control') for i in order[:max(0, count - len(chosen))])
    return [{**candidates[index], 'selection': label} for index, label in chosen]


def prepare_relation_removal(core, scene, relation):
    """Rebuild the complete scene after removing one named applied relation."""
    required = ('stage', 'route', 'destination_id', 'donor_id')
    removal = {name: relation[name] for name in required}
    prepared = core.prepare(scene, return_trace=True, context_message_removal=removal)
    relationships = capture_applied_relationships(core, prepared)
    stages = [stage for stage in relationships['contextual_stages'] if stage['stage'] == removal['stage']
              and stage['route'] == removal['route']]
    if len(stages) != 1:
        raise ValueError('The requested relation stage was not executed exactly once.')
    stage = stages[0]
    receiver = stage['receiver_ids'] == int(removal['destination_id'])
    donor = stage['donor_ids'] == int(removal['donor_id'])
    hit = receiver[:, :, None] & donor[:, None, :]
    if not bool(hit.any()) or bool((stage['gate'][hit] != 0).any()):
        raise ValueError('The named original relation was absent or not removed.')
    return prepared


def _same_optional(left, right):
    return (left is None and right is None) or (
        isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right))


def _matched_slots(scene_before, scene_after, role):
    name = 'module' if role == 'M' else 'environment'
    ids_before, ids_after = (getattr(scene, name + '_ids')[0] for scene in (scene_before, scene_after))
    mask_before, mask_after = (getattr(scene, name + '_mask')[0] for scene in (scene_before, scene_after))
    slots_before, slots_after = mask_before.nonzero().flatten(), mask_after.nonzero().flatten()
    values_before, values_after = ids_before[slots_before], ids_after[slots_after]
    if values_before.unique().numel() != values_before.numel() or values_after.unique().numel() != values_after.numel():
        raise ValueError('Original member IDs must be unique within each role.')
    order_before, order_after = values_before.argsort(), values_after.argsort()
    if not torch.equal(values_before[order_before], values_after[order_after]):
        raise ValueError('Insertion, removal or changed member roles have no two-factor attribution.')
    return slots_before[order_before], slots_after[order_after], values_before[order_before]


def finite_read_decomposition(core, before, query_before, after, query_after, *,
                              identity_confirmed=False, read_mode='restricted', intervention=None):
    """Decompose a fixed-query finite change and allocate it through the joint head.

    The caller must establish persistent member identity from the source edit,
    not geometric matching. The query coordinates, masks, IDs, features and
    actual encoded query state must match exactly. Nonlinear-head intermediate
    values are counterfactual latent inputs; no physical solve is implied.

    Returns detached tensors and separate nominal/diagnostic value work counts.
    One scene and at most one query tile bound the retained capture size.
    """
    if not isinstance(core, AnchoredMemberRoutingCore) or getattr(core, '_uses_hyperedge_rows', False):
        raise ValueError('This decomposition requires the standard anchored original-member joint head.')
    if core.training:
        raise ValueError('Set the model to evaluation mode for a stable diagnostic.')
    if identity_confirmed is not True:
        raise ValueError('The case adapter must confirm persistent original member identities.')
    if before.scene.batch_size != 1 or after.scene.batch_size != 1:
        raise ValueError('Capture one edited scene pair at a time.')
    if read_mode not in ('restricted', 'full'):
        raise ValueError('Choose restricted or full original-member access.')
    core.validate_prepared_state(before, query_before)
    core.validate_prepared_state(after, query_after)
    if query_before.num_queries > core.query_tile_size or query_after.num_queries > core.query_tile_size:
        raise ValueError('Capture no more than one query tile at a time.')
    for name in ('coordinates', 'query_mask', 'query_ids', 'features'):
        if not _same_optional(getattr(query_before, name), getattr(query_after, name)):
            return {'available': False, 'reason': f'Query {name} changed; moving queries need an additional factor.'}
    if query_before.coordinate_frame_key != query_after.coordinate_frame_key:
        return {'available': False, 'reason': 'Query coordinate frames differ.'}
    try:
        matched = {role: _matched_slots(before.scene, after.scene, role) for role in ('M', 'E')}
    except ValueError as error:
        return {'available': False, 'reason': str(error)}
    _synchronize(query_before.coordinates.device)
    started = time.perf_counter()
    with torch.no_grad():
        qs_before = core._query_states(before, query_before)
        qs_after = core._query_states(after, query_after)
        if not torch.equal(qs_before, qs_after):
            return {'available': False, 'reason': 'Query encoding changed; a third computational factor is required.'}
        options = {**(intervention or {}), 'read_mode': read_mode}
        out_before = core.decode(before, query_before, intervention=options, return_trace=True)
        out_after = core.decode(after, query_after, intervention=options, return_trace=True)
        role_results = {}
        delta_access, delta_message = {}, {}
        work = {}
        for role in ('M', 'E'):
            slot_before, slot_after, ids = matched[role]
            weights_before = out_before.trace[f'member_weights_{role}'][0, :, slot_before]
            weights_after = out_after.trace[f'member_weights_{role}'][0, :, slot_after]
            # Advanced tensor indexing is [queries, members].
            union = ((weights_before > 0) | (weights_after > 0)).nonzero()
            qindex, local = union.unbind(-1)
            batch = torch.zeros_like(qindex)
            message_before, count_before = core._selected_member_values(
                before, query_before, qs_before, role, batch, qindex, slot_before[local])
            message_after, count_after = core._selected_member_values(
                after, query_after, qs_after, role, batch, qindex, slot_after[local])
            beta_before, beta_after = weights_before[qindex, local], weights_after[qindex, local]
            access_rows = (beta_after - beta_before)[:, None] * (message_before + message_after) / 2
            message_rows = (beta_before + beta_after)[:, None] * (message_after - message_before)
            message_rows = message_rows / 2
            query_shape = (query_before.num_queries, core.context_dim)
            access = qs_before.new_zeros(query_shape).index_add(0, qindex, access_rows)[None]
            message = qs_before.new_zeros(query_shape).index_add(0, qindex, message_rows)[None]
            delta_access[role], delta_message[role] = access, message
            role_results[role] = {'query_index': qindex.detach(), 'original_id': ids[local].detach(),
                                  'beta_before': beta_before.detach(), 'beta_after': beta_after.detach(),
                                  'message_before': message_before.detach(), 'message_after': message_after.detach(),
                                  'access_change': access.detach(), 'message_change': message.detach()}
            work[role] = {'union_pairs': int(len(qindex)),
                          'diagnostic_message_pairs_before': int(len(qindex)),
                          'diagnostic_message_pairs_after': int(len(qindex)),
                          'missing_from_before_support': int((beta_before == 0).sum()),
                          'missing_from_after_support': int((beta_after == 0).sum()),
                          'already_selected_recomputed_before': int((beta_before > 0).sum()),
                          'already_selected_recomputed_after': int((beta_after > 0).sum()),
                          'message_work_before': {k: int(v) for k, v in count_before.items()},
                          'message_work_after': {k: int(v) for k, v in count_after.items()}}
        h_before = torch.cat((out_before.trace['context_M'], out_before.trace['context_E']), dim=-1)
        h_after = torch.cat((out_after.trace['context_M'], out_after.trace['context_E']), dim=-1)
        access = torch.cat((delta_access['M'], delta_access['E']), dim=-1)
        message = torch.cat((delta_message['M'], delta_message['E']), dim=-1)
        def head(context):
            output = core.member_router.joint_head(core.member_router.joint_trunk(torch.cat((qs_before, context), -1)))
            return output * core._query_mask(query_before)[..., None].to(output)
        base, last = head(h_before), head(h_after)
        access_middle, message_middle = head(h_before + access), head(h_before + message)
        allocated_access = ((access_middle - base) + (last - message_middle)) / 2
        allocated_message = ((message_middle - base) + (last - access_middle)) / 2
        _synchronize(query_before.coordinates.device)
        elapsed = time.perf_counter() - started
        return {'available': True, 'read_mode': read_mode, 'roles': role_results,
                'query_coordinates': query_before.coordinates.detach().clone(),
                'query_ids': None if query_before.query_ids is None else query_before.query_ids.detach().clone(),
                'context_before': h_before.detach(), 'context_after': h_after.detach(),
                'access_change': access.detach(), 'message_change': message.detach(),
                'field_before': out_before.joint_fields.detach(), 'field_after': out_after.joint_fields.detach(),
                'head_access_allocation': allocated_access.detach(), 'head_message_allocation': allocated_message.detach(),
                'context_reconstruction_residual': (h_after - h_before - access - message).detach(),
                'field_allocation_residual': (out_after.joint_fields - out_before.joint_fields
                                              - allocated_access - allocated_message).detach(),
                'work': {'nominal_decode_before': dict(out_before.work_counts),
                         'nominal_decode_after': dict(out_after.work_counts),
                         'diagnostic_message_recomputation': work,
                         'diagnostic_joint_head_evaluations': 4,
                         'elapsed_seconds_including_nominal_decodes': elapsed},
                'interpretation': 'Computational allocation; messages include upstream contextual changes. '
                                  'Intermediate contexts are not solved physical scenes.'}

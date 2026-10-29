"""Vectorized live EM/ME/EE route composition for the R13-12 reader."""
import torch
from .guided import prior_weighted_scores
from .anchored import build_e_access_measure


def _prefix(weights, coverage):
    order = torch.argsort(weights.detach(), dim=-1, descending=True, stable=True)
    sorted_weights = weights.detach().gather(-1, order)
    before = sorted_weights.cumsum(-1) - sorted_weights
    selected = (before < sorted_weights.sum(-1, keepdim=True) * coverage) & (sorted_weights > 0)
    return torch.zeros_like(selected).scatter(-1, order, selected)


def decode_tile(core, prepared, query, modes, *, return_trace=False, intervention=None):
    controls = {} if intervention is None else dict(intervention)
    if set(controls) - {'read_mode', 'remove_module_mediated_environment', 'suppress_module_values', 'exclude_environment_ids'}:
        raise ValueError('Unsupported mediated read intervention.')
    scene = prepared.scene
    m, e = scene.num_modules, scene.num_environment_tokens
    total = m + e
    reads = ((query.anchor_weights > 0) & core._query_mask(query)[..., None]).nonzero(as_tuple=False)
    b, q, k = reads.unbind(-1)
    a = query.anchor_indices[b, q, k]
    r = len(reads)
    qstates = core._query_states(prepared, query)
    weights = query.anchor_weights[b, q, k]
    if not r:
        zero = qstates.new_zeros(query.batch_size, query.num_queries, core.output_dim)
        return {mode: (zero, None, {}) for mode in modes}, {}
    tiny = torch.nextafter(qstates.new_zeros(()), qstates.new_ones(()))
    measure = build_e_access_measure(scene.module_mask, scene.environment_mask, scene.environment_measure,
                                     dtype=qstates.dtype, access_measure=core.access_measure)
    mm, em = measure[:, :m], measure[:, m:]
    em = em * scene.environment_mask.to(em)
    mm = mm * scene.module_mask.to(mm)
    direct_mass = prepared.anchored_applied_EE[b, a] * em[b]
    direct_mass = direct_mass.scatter_add(1, a[:, None], em[b, a, None])
    d = direct_mass.sum(-1, keepdim=True)
    direct = direct_mass / d.clamp_min(tiny)
    module_mass = prepared.anchored_applied_EM[b, a] * mm[b]
    source_prior = torch.cat((d, module_mass), -1)
    valid_source = source_prior > 0
    sr, ss = valid_source.nonzero(as_tuple=True)
    source_member = torch.where(ss == 0, a[sr] + m, ss - 1)
    source_key = (b[sr] * query.num_queries + q[sr]) * total + source_member
    source_union = torch.unique(source_key, sorted=True)
    normalized_query = core._routing_features(qstates)
    query_terms = {role: torch.nn.functional.linear(normalized_query, scorer[0].weight[:, :core.context_dim], scorer[0].bias)
                   for role, scorer in (("M", core.member_router.member_M), ("E", core.member_router.member_E))}
    source_raw, qnorm, counts = core._raw_union_scores(prepared, query, qstates, source_union, query_terms=query_terms)
    raw = source_prior.new_zeros(source_prior.shape).index_put((sr, ss), source_raw[torch.searchsorted(source_union, source_key)])
    ell = (4 * torch.tanh(raw / 4) * valid_source if core.score_calibration == 'bounded_tanh'
           else prior_weighted_scores(raw, source_prior, valid_source))
    logits = torch.where(valid_source, torch.where(valid_source, source_prior, torch.ones_like(source_prior)).log() + ell,
                         torch.full_like(source_prior, -torch.inf))
    alpha = torch.softmax(logits, -1)
    alpha = torch.where(valid_source, alpha.clamp_min(tiny), torch.zeros_like(alpha))
    alpha = alpha / alpha.sum(-1, keepdim=True)

    # Build the actual positive ME edge directory once per tile. Its row IDs
    # encode (batch, M); expansion below includes only selected query sources.
    me_mass = prepared.mediated_applied_ME * em[:, None, :]
    eb, ei, ej = (me_mass > 0).nonzero(as_tuple=True)
    edge_row = eb * m + ei
    edge_mass = me_mass[eb, ei, ej]
    row_counts = torch.bincount(edge_row, minlength=scene.batch_size * m)
    row_offsets = torch.cat((row_counts.new_zeros(1), row_counts.cumsum(0)))
    raw_row_mass = me_mass.sum(-1).reshape(-1)
    # Condition each actual (scene, anchor, M) row once, shared across
    # queries and both paired modes. Only positive native ME edges are packed.
    source_selections = {}
    used = torch.zeros_like(valid_source[:, 1:])
    for mode in modes:
        selected = _prefix(alpha, core.source_retained_mass) if mode in ('restricted', 'source95_all') else valid_source
        selected = torch.cat((torch.ones_like(selected[:, :1]), selected[:, 1:]), -1)
        source_selections[mode] = selected
        used = used | (selected[:, 1:] & valid_source[:, 1:])
    all_r, all_m = used.nonzero(as_tuple=True)
    unique_address = torch.unique((b[all_r] * e + a[all_r]) * m + all_m, sorted=True)
    unique_m = unique_address.remainder(m)
    unique_anchor = torch.div(unique_address, m, rounding_mode='floor').remainder(e)
    unique_batch = torch.div(unique_address, m * e, rounding_mode='floor')
    unique_row = unique_batch * m + unique_m
    unique_lengths = row_counts[unique_row]
    unique_offsets = torch.cat((unique_lengths.new_zeros(1), unique_lengths.cumsum(0)))
    unique_owner = torch.repeat_interleave(torch.arange(len(unique_row), device=b.device), unique_lengths)
    unique_local = torch.arange(len(unique_owner), device=b.device) - (unique_lengths.cumsum(0) - unique_lengths)[unique_owner]
    unique_edge = row_offsets[unique_row[unique_owner]] + unique_local
    unique_destination = ej[unique_edge]
    if core.mediated_mode == 'G' and unique_edge.numel():
        unique_logits = edge_mass[unique_edge].log() + prepared.mediated_ee_geometry[unique_batch[unique_owner], unique_anchor[unique_owner], unique_destination]
        maxima = torch.segment_reduce(unique_logits, 'max', lengths=unique_lengths, initial=-torch.inf)
        numerator = (unique_logits - maxima[unique_owner]).exp().clamp_min(tiny)
        denominator = torch.segment_reduce(numerator, 'sum', lengths=unique_lengths)
        unique_conditional = numerator / denominator[unique_owner]
    elif core.mediated_mode == 'P':
        unique_conditional = edge_mass[unique_edge] / raw_row_mass[unique_row[unique_owner]]
    else:
        unique_conditional = edge_mass[unique_edge]
    # Reorder the much smaller unique-anchor edge directory by destination
    # once. Query expansion preserves (read, destination, M) order, so both
    # modes reuse one deterministic segmented reduction without sorting the
    # much larger query-expanded edge list.
    anchor_address = torch.unique(b * e + a, sorted=True)
    edge_anchor = torch.div(unique_address[unique_owner], m, rounding_mode='floor')
    edge_order = torch.argsort(edge_anchor * e + unique_destination, stable=True)
    ordered_anchor = torch.searchsorted(anchor_address, edge_anchor[edge_order])
    ordered_m = unique_m[unique_owner[edge_order]]
    ordered_destination = unique_destination[edge_order]
    ordered_conditional = unique_conditional[edge_order]
    anchor_counts = torch.bincount(ordered_anchor, minlength=len(anchor_address))
    anchor_offsets = torch.cat((anchor_counts.new_zeros(1), anchor_counts.cumsum(0)))
    read_anchor = torch.searchsorted(anchor_address, b * e + a)
    read_lengths = anchor_counts[read_anchor]
    edge_r = torch.repeat_interleave(torch.arange(r, device=b.device), read_lengths)
    edge_local = torch.arange(len(edge_r), device=b.device) - (read_lengths.cumsum(0) - read_lengths)[edge_r]
    edge_index = anchor_offsets[read_anchor[edge_r]] + edge_local
    edge_m = ordered_m[edge_index]
    destination = ordered_destination[edge_index]
    conditional = ordered_conditional[edge_index]
    destination_key = edge_r * e + destination
    unique_key, lengths_by_key = torch.unique_consecutive(destination_key, return_counts=True)
    packed, debug = {}, {}
    all_keys = [source_union]
    for mode in modes:
        selected = source_selections[mode]
        use_m = selected[:, 1:] & valid_source[:, 1:]
        route = alpha[edge_r, edge_m + 1] * conditional * use_m[edge_r, edge_m]
        if controls.get('remove_module_mediated_environment', False):
            route = torch.zeros_like(route)
        environment = alpha[:, :1] * direct
        # Fixed-order segmented sums keep coverage selection stable during
        # checkpoint replay; all modes share the same packed destination keys.
        if route.numel():
            grouped = torch.segment_reduce(route, 'sum', lengths=lengths_by_key)
            flat = environment.reshape(-1)
            environment = flat.index_copy(0, unique_key, flat[unique_key] + grouped).reshape(r, e)
        own = environment.gather(1, a[:, None]).clamp_min(tiny)
        environment = environment.scatter(1, a[:, None], own)
        excluded = controls.get('exclude_environment_ids', ())
        if excluded:
            ids = torch.as_tensor(excluded, dtype=scene.environment_ids.dtype, device=b.device)
            forbidden = torch.isin(scene.environment_ids[b], ids)
            forbidden = forbidden & (torch.arange(e, device=b.device)[None, :] != a[:, None])
            environment = environment.masked_fill(forbidden, 0)
        e_keep = environment > 0
        if mode in ('restricted', 'all_e95'):
            e_keep = _prefix(environment, core.environment_retained_mass)
            e_keep = e_keep.scatter(1, a[:, None], True)
        er, es = e_keep.nonzero(as_tuple=True)
        selected_r, selected_m = use_m.nonzero(as_tuple=True)
        members = torch.cat((selected_m, es + m))
        rid = torch.cat((selected_r, er))
        prior = torch.cat((alpha[selected_r, selected_m + 1], environment[er, es]))
        source_ell = torch.cat((ell[selected_r, selected_m + 1], environment.new_zeros(len(er))))
        pair_key = (b[rid] * query.num_queries + q[rid]) * total + members
        if m:
            original = torch.where(members < m, scene.module_ids[b[rid], members.clamp(max=m-1)], scene.environment_ids[b[rid], (members-m).clamp(min=0)])
        else:
            original = scene.environment_ids[b[rid], members]
        packed[mode] = dict(reads=reads, anchors=a, weights=weights, read_id=rid, members=members,
                            prior=prior, source_ell=source_ell, original=original, is_module=members < m,
                            pair_key=pair_key, suppress_module_values=bool(controls.get('suppress_module_values', False)))
        all_keys.append(pair_key)
        if return_trace:
            debug[mode] = (selected, environment, edge_r, edge_m, destination, conditional, raw_row_mass.reshape(scene.batch_size, m))
    union = torch.unique(torch.cat(all_keys), sorted=True)
    new_keys = union[~torch.isin(union, source_union)]
    new_raw, _, new_counts = core._raw_union_scores(prepared, query, qstates, new_keys, query_terms=query_terms)
    raw_union = torch.cat((source_raw, new_raw))[torch.argsort(torch.cat((source_union, new_keys)))]
    for key in counts:
        counts[key] += new_counts[key] if 'raw_score_pairs' in key else 0
    outputs = {}
    for mode in modes:
        fields, trace, local_counts = core._read_packed(prepared, query, qstates, qnorm, packed[mode], union, raw_union, return_trace=return_trace)
        if trace is not None:
            selected, environment, edge_r, edge_m, destination, conditional, row_mass = debug[mode]
            # Small explicit inspection panel only; no persistent scene-wide E×M×E tensor.
            direct_rows, conditional_rows, conditional_slots = [], [], []
            addresses = b * e + a
            for address in torch.unique(addresses).detach().cpu().tolist():
                bi, ai = divmod(address, e)
                rid = (addresses == address).nonzero(as_tuple=False).flatten()
                slots = (module_mass[rid[0]] > 0).nonzero(as_tuple=False).flatten()
                first = rid[0]
                use = edge_r == first
                rowmap = torch.full((m,), -1, dtype=torch.long, device=b.device)
                rowmap[slots] = torch.arange(len(slots), device=b.device)
                matrix = environment.new_zeros((len(slots), e)).index_put((rowmap[edge_m[use]], destination[use]), conditional[use])
                direct_rows.append((rid, direct[first]))
                conditional_rows.append((rid, matrix, row_mass[bi, slots]))
                conditional_slots.append(slots)
            trace.update(mediated_source_alpha=alpha, mediated_source_prior=source_prior, mediated_source_ell=ell,
                         mediated_source_members=source_member, mediated_source_read_index=sr,
                         mediated_applied_ME=prepared.mediated_applied_ME, mediated_ee_geometry=prepared.mediated_ee_geometry,
                         mediated_me_stage=prepared.mediated_me_stage, mediated_source_selected=selected,
                         mediated_environment_prior=environment, mediated_direct_rows=direct_rows,
                         mediated_conditional_rows=conditional_rows, mediated_conditional_module_slots=conditional_slots,
                         mediated_conditioned_anchor_module_rows=unique_address)
        outputs[mode] = fields, trace, local_counts
    return outputs, counts

"""Bind existing Thermal fields and native linear observation maps.

No objective or edited-scene labels live here. Physical cell-centre fields are
compared in the fixed frame. Material and port records are individually
observed: module slots from independently generated scenes are not identities.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import h5py
import numpy as np
import torch
from scipy.ndimage import label

from honf_forward_core.content_hypergraph import QueryBatch
from honf_runtime.observation_blocks import FieldComponent, LinearExtraction, ObservationBlock, QueryDependency

from ..data.datasets import _local_disk_query_points
from ..e_anchor_locator import attach_thermal_e_anchor_query
from ..native_observations import build_native_solid_mask, native_cell_center_stencil
from .thermal_native_pair_task import build_thermal_native_pair_task, describe_native_scene


_TRAIN15 = ('0368','0369','0411','0412','0311','0102','0330','0518','0519','0344','0345','0185','0186','0360','0619')
_FRAME = 'thermal_physical_xy'
_TIME = 'converged_final_window'
_ASSESSMENT9 = ('0686','0687','0645','0646','0281','0283','0284','0294','0297')
_UNIT = 'native dataset units'


def _stencil(points, lengths, nx, ny):
    points = torch.as_tensor(points, dtype=torch.float32).reshape(1,-1,2)
    value = native_cell_center_stencil(points, torch.as_tensor(lengths, dtype=torch.float32).reshape(1,2), nx=nx, ny=ny)
    return (value.indices[0].cpu().numpy(), value.weights[0].cpu().numpy(), value.valid[0].cpu().numpy())


def _coordinates(group):
    x, y = np.asarray(group['x_grid']), np.asarray(group['y_grid'])
    if x.ndim == y.ndim == 1:
        x, y = np.meshgrid(x, y, indexing='xy')
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError('Thermal native coordinates must share one physical cell-centre array.')
    return np.stack((x,y), -1)


def _deps(case_id, xy):
    return tuple(QueryDependency(case_id, _FRAME, _TIME, int(i), tuple(point))
                 for i, point in enumerate(np.asarray(xy).reshape(-1,2)))


def _extraction(dependencies, dependency_indices, observations, components, outputs, coefficients):
    return LinearExtraction(dependencies, np.asarray(dependency_indices, dtype=np.int64),
        np.asarray(observations, dtype=np.int64), np.asarray(components, dtype=np.int64),
        np.asarray(outputs, dtype=np.int64), np.asarray(coefficients, dtype=np.float32))


def _field_map(dependencies, geometry, nx, ny, lengths):
    n = nx * ny
    ids = np.arange(n, dtype=np.int64)
    # Physical u/v/p and T are direct joint-field values at each cell.
    d = [ids, ids, ids, ids]
    r = [ids, ids, ids, ids]
    c = [np.full(n, x, dtype=np.int64) for x in (0,1,2,4)]
    o = [np.full(n, x, dtype=np.int64) for x in (0,1,2,3)]
    w = [np.ones(n, dtype=np.float32) for _ in range(4)]
    ix, iy = ids % nx, ids // nx
    left, right = np.maximum(ix-1,0), np.minimum(ix+1,nx-1)
    down, up = np.maximum(iy-1,0), np.minimum(iy+1,ny-1)
    neighbours = (iy*nx+right, iy*nx+left, up*nx+ix, down*nx+ix)
    spacing = np.asarray(lengths, dtype=np.float32) / np.asarray((nx,ny), dtype=np.float32)
    dx, dy = (right-left).astype(np.float32)*spacing[0], (up-down).astype(np.float32)*spacing[1]
    fluid = ~np.asarray(geometry, dtype=bool).reshape(-1)
    for index, output, coefficient in zip(neighbours, (1,1,0,0), (1/dx,-1/dx,-1/dy,1/dy)):
        d.append(index); r.append(ids); c.append(np.full(n,3,dtype=np.int64)); o.append(np.full(n,output,dtype=np.int64))
        w.append(coefficient * fluid[index] * fluid[ids])
    return _extraction(dependencies, np.concatenate(d), np.concatenate(r), np.concatenate(c),
                       np.concatenate(o), np.concatenate(w))


def _fluid_block(groups, configs, pair, scales, weights):
    xy = _coordinates(groups[0]); other = _coordinates(groups[1])
    if not np.array_equal(xy, other):
        raise ValueError('Thermal fixed-frame pair has different native cell-centre coordinates.')
    ny, nx = xy.shape[:2]
    lengths = tuple(float(configs[0]['domain'][k]) for k in ('lx','ly'))
    if any(tuple(float(config['domain'][k]) for k in ('lx','ly')) != lengths for config in configs):
        raise ValueError('Thermal fixed-frame pair has different physical domain lengths.')
    refs = np.stack([np.asarray(group['steady_field']).reshape(-1,5) for group in groups])
    if refs.shape != (2,nx*ny,5):
        raise ValueError('Thermal native fields must contain u, v, p, omega and temperature.')
    geometry = [np.asarray(group['module_mask'], dtype=bool) for group in groups]
    valid = np.stack([(~mask).reshape(-1,1) & np.isfinite(ref) for mask,ref in zip(geometry,refs)])
    pair_mask = valid[0] & valid[1]
    component_labels, _ = label(pair_mask.any(axis=1).reshape(ny,nx), structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    # Non-paired support does not create response edges; keep it in separate
    # groups so it cannot interrupt the fixed-frame response graph.
    labels = component_labels.reshape(-1)
    groups_key = tuple(('common', int(v)) if v else ('individual', int(i)) for i,v in enumerate(labels))
    fields = tuple(FieldComponent(name, unit, output, float(scale), float(weight))
        for name,unit,output,scale,weight in zip(('u','v','p','omega','temperature'),
            ('native velocity units','native velocity units','native pressure units','native vorticity units','native temperature units'),
            (0,1,2,1,3),scales,weights))
    deps = [_deps(case, xy) for case in pair]
    maps = tuple(_field_map(dep, mask, nx, ny, lengths) for dep,mask in zip(deps,geometry))
    mass = np.full((2,nx*ny,5), lengths[0]*lengths[1]/(nx*ny), dtype=np.float32)
    return ObservationBlock(name='fluid_cell_centres', components=fields, scene_ids=tuple(pair),
        partition='native', frame=_FRAME, time=_TIME, receiver_ids=tuple(f'cell_{i}' for i in range(nx*ny)),
        coordinates=xy.reshape(-1,2), references=refs, absolute_mask=valid, pair_mask=pair_mask,
        edge_mask=pair_mask.copy(), absolute_measures=mass, pair_measures=mass[0], extractions=maps,
        neighbor_groups=groups_key, domain_tags=('fluid',)*(nx*ny), correspondence='fixed_frame')


def _individual_block(name, pair, xy, scene_rows, components, physical_weights, stencils, *, output=3):
    """Pack two independent native observation lists without asserting identity."""
    n = sum(len(row['coordinates']) for row in scene_rows)
    width = len(components)
    coords = np.concatenate([row['coordinates'] for row in scene_rows])
    refs = np.zeros((2,n,width), dtype=np.result_type(*(row['references'].dtype for row in scene_rows)))
    valid = np.zeros((2,n,width), dtype=bool)
    masses = np.zeros((2,n,width), dtype=np.float32)
    receiver_ids, neighbor_groups, member_ids = [], [], []
    maps = []
    start = 0
    for scene_index, (scene_id, row, terms, dependencies) in enumerate(zip(pair, scene_rows, stencils,
                                                                            (_deps(pair[0],xy),_deps(pair[1],xy)))):
        size = len(row['coordinates']); end = start + size
        refs[scene_index,start:end] = row['references']
        valid[scene_index,start:end] = row['valid']
        masses[scene_index,start:end] = np.broadcast_to(np.asarray(physical_weights[scene_index])[:,None], (size,width))
        receiver_ids.extend(f'{scene_id}:{item}' for item in row['receiver_ids'])
        member_ids.extend((scene_id,int(member)) for member in row['members'])
        neighbor_groups.extend((scene_id,int(member)) for member in row['members'])
        dep, obs, comp, coef = terms
        maps.append(_extraction(dependencies, dep, np.asarray(obs)+start, comp,
                                np.full(len(dep),output,dtype=np.int64), coef))
        start = end
    return ObservationBlock(name=name, components=components, scene_ids=tuple(pair), partition='native',
        frame=_FRAME, time=_TIME, receiver_ids=tuple(receiver_ids), coordinates=coords, references=refs,
        absolute_mask=valid, pair_mask=np.zeros((n,width),dtype=bool), edge_mask=np.zeros((n,width),dtype=bool),
        absolute_measures=masses, pair_measures=np.zeros((n,width),dtype=np.float32), extractions=tuple(maps),
        neighbor_groups=tuple(neighbor_groups), member_ids=tuple(member_ids), domain_tags=(name,)*n,
        correspondence='none')


def _map_stencil(indices, weights, component, *, factor=1.):
    n = len(indices)
    return (indices.reshape(-1), np.repeat(np.arange(n,dtype=np.int64), indices.shape[1]),
            np.full(n*indices.shape[1], component, dtype=np.int64), (weights*factor).reshape(-1))


def _individual_blocks(groups, configs, pair, xy, model, weights, outside_weight):
    ny,nx = xy.shape[:2]
    lengths = np.asarray((configs[0]['domain']['lx'],configs[0]['domain']['ly']), dtype=np.float32)
    material_rows, surface_rows, outside_rows = [], [], []
    material_terms, surface_terms, outside_terms = [], [], []
    material_mass, surface_mass, outside_mass = [], [], []
    for group,config in zip(groups,configs):
        present = np.asarray(group['module_present'], dtype=bool)
        slots = np.flatnonzero(present)
        centers = np.asarray(group['module_centers'])[slots].astype(np.float32,copy=False)
        radius = np.float32(config['domain']['module_radius'])
        local_mask = np.asarray(group['module_internal_mask'],dtype=bool)
        local = _local_disk_query_points(local_mask)
        # The maintained model creates these points from saved FP32 inputs.
        material_xy = (torch.from_numpy(centers)[:,None] + torch.tensor(radius)*torch.from_numpy(local)[None]).numpy().reshape(-1,2)
        material_ref = np.asarray(group['module_internal_temperature'])[slots][:,local_mask].reshape(-1,1)
        mi,mw,mv = _stencil(material_xy,lengths,nx,ny)
        members = np.repeat(slots,len(local))
        material_rows.append({'coordinates':material_xy,'references':material_ref,'valid':mv[:,None]&np.isfinite(material_ref),
            'members':members,'receiver_ids':[f'material_{m}_{j}' for m in slots for j in range(len(local))]})
        material_terms.append(_map_stencil(mi,mw,0))
        material_mass.append(np.full(len(material_xy), np.pi*radius**2/max(len(local),1),dtype=np.float32))
        theta = torch.arange(64,dtype=torch.float32)*(2*torch.pi/64)
        normals = torch.stack((theta.cos(),theta.sin()),-1)
        delta = torch.minimum((torch.as_tensor(lengths)/torch.tensor([nx,ny])).amin().reshape(1), .15*torch.tensor(radius))[0]
        surface = (torch.from_numpy(centers)[:,None] + torch.tensor(radius)*normals[None]).numpy().reshape(-1,2)
        outside = (torch.from_numpy(centers)[:,None] + (torch.tensor(radius)+delta)*normals[None]).numpy().reshape(-1,2)
        si,sw,sv = _stencil(surface,lengths,nx,ny)
        oi,ow,ov = _stencil(outside,lengths,nx,ny)
        targets = np.asarray(group['interface_target'])[slots].reshape(-1,2)
        interface_valid = np.asarray(group['interface_condition_valid_mask'])[slots].reshape(-1) > .5
        members_surface = np.repeat(slots,64)
        valid = np.stack((sv & interface_valid & np.isfinite(targets[:,0]),
                          sv & ov & interface_valid & np.isfinite(targets[:,1])),axis=-1)
        receiver_ids = [f'port_{m}_{j}' for m in slots for j in range(64)]
        surface_rows.append({'coordinates':surface,'references':targets,'valid':valid,
                             'members':members_surface,'receiver_ids':receiver_ids})
        k_s = float(group['material_parameters'].attrs.get('solid_k', config['thermal'].get('solid_k',1.)))
        k_f = float(group['material_parameters'].attrs.get('fluid_k', config['thermal'].get('fluid_k',1.)))
        conductivity = np.float32(2*k_s*k_f/(k_s+k_f))
        multiplier = float(conductivity / np.float32(delta))
        chunks = (_map_stencil(si,sw,0), _map_stencil(si,sw,1,factor=multiplier),
                  _map_stencil(oi,ow,1,factor=-multiplier))
        surface_terms.append(tuple(np.concatenate([entry[i] for entry in chunks]) for i in range(4)))
        surface_mass.append(np.full(len(surface), 2*np.pi*radius/64,dtype=np.float32))
        outside_ref = np.asarray(group['interface_condition'])[slots,:,3].reshape(-1,1)
        outside_rows.append({'coordinates':outside,'references':outside_ref,'valid':ov[:,None]&np.isfinite(outside_ref),
                             'members':members_surface,'receiver_ids':receiver_ids})
        outside_terms.append(_map_stencil(oi,ow,0))
        outside_mass.append(np.full(len(outside), 2*np.pi*(radius+float(delta))/64,dtype=np.float32))
    material_component = (FieldComponent('material_temperature','native temperature units',3,float(model.internal_temperature_std[0]),float(weights['temperature/material'])),)
    surface_components = (FieldComponent('surface_temperature','native temperature units',3,float(model.interface_target_std[0]),float(weights['temperature/surface'])),
                          FieldComponent('normal_flux_proxy','native flux-proxy units',3,float(model.interface_target_std[1]),float(weights['q_proxy'])))
    outside_component = (FieldComponent('outside_temperature','native temperature units',3,float(model.field_std[4]),float(outside_weight)),)
    return (_individual_block('material_temperatures',pair,xy,material_rows,material_component,material_mass,material_terms),
            _individual_block('surface_observations',pair,xy,surface_rows,surface_components,surface_mass,surface_terms),
            _individual_block('outside_temperatures',pair,xy,outside_rows,outside_component,outside_mass,outside_terms))


_FINITE_FLUID = (
    ('flow/u', 'native velocity units', 0, 0, 'masked_u'),
    ('flow/v', 'native velocity units', 1, 1, 'masked_v'),
    ('flow/p', 'native pressure units', 2, 2, 'pressure'),
    ('flow/omega', 'native vorticity units', 3, 1, 'native_curl'),
    ('temperature/fluid', 'native temperature units', 4, 3, 'interpolated_temperature'),
)
_FINITE_SUPPORT = {
    'flow/u': 'fluid_sampled_points', 'flow/v': 'fluid_sampled_points',
    'flow/p': 'fluid_sampled_points', 'flow/omega': 'fluid_sampled_points',
    'temperature/fluid': 'fluid_sampled_points',
    'temperature/material': 'module_interior',
    'temperature/surface': 'module_surface', 'q_proxy': 'module_surface',
    'port/outside_temperature': 'module_outside_surface',
}


def _finite_seed(seed, case_id, population):
    """Choose one source-only stream per scene and source-row population."""
    text = '\0'.join((str(seed), str(case_id), population)).encode('utf-8')
    return int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), 'little')


def _finite_stencil_valid(points, lengths, nx, ny):
    """Match native FP32 stencil validity without retaining all its corners."""
    points = np.asarray(points)
    valid = np.isfinite(points).all(axis=-1)
    result = np.zeros(len(points), dtype=bool)
    if not valid.any():
        return result
    xy = torch.as_tensor(points[valid], dtype=torch.float32)
    spacing = torch.as_tensor(lengths, dtype=torch.float32) / xy.new_tensor([nx, ny])
    location = xy / spacing - .5
    rounded = location.round()
    near = (location - rounded).abs() <= 8 * torch.finfo(location.dtype).eps * max(nx, ny)
    location = location + torch.where(near, rounded - location, location.new_zeros(()))
    result[valid] = ((location >= 0) & (location <= location.new_tensor([nx - 1, ny - 1]))).all(-1).numpy()
    return result


def _finite_extraction(case_id, lengths, nx, ny, node_indices, output_indices, coefficients):
    """Build a compact map after selection, keeping each row's terms distinct."""
    nodes = np.asarray(node_indices, dtype=np.int64)
    outputs = np.broadcast_to(np.asarray(output_indices, dtype=np.int64), nodes.shape)
    coefficients = np.broadcast_to(np.asarray(coefficients), nodes.shape)
    if nodes.ndim != 2 or nodes.shape[0] < 1 or nodes.shape != coefficients.shape:
        raise ValueError('Selected Thermal extraction terms must be nonempty [row, term] arrays.')
    # Zero coefficients require no decoded field, except that a physically
    # zero row still retains one explicit term so its extraction is defined.
    keep = coefficients != 0
    keep[~keep.any(axis=1), 0] = True
    unique, inverse = np.unique(nodes[keep], return_inverse=True)
    if np.any(unique < 0) or np.any(unique >= nx * ny):
        raise ValueError('Thermal extraction refers to an unavailable native cell.')
    spacing = torch.as_tensor(lengths, dtype=torch.float32) / torch.tensor([nx, ny], dtype=torch.float32)
    ids = torch.as_tensor(unique, dtype=torch.long)
    coordinates = (torch.stack((ids % nx + .5, ids // nx + .5), dim=-1) * spacing).numpy()
    dependencies = tuple(QueryDependency(str(case_id), _FRAME, _TIME, int(index), tuple(point))
                         for index, point in zip(unique, coordinates))
    rows = np.broadcast_to(np.arange(nodes.shape[0], dtype=np.int64)[:, None], nodes.shape)
    return LinearExtraction(dependencies, inverse.astype(np.int64, copy=False), rows[keep],
                            np.zeros(len(inverse), dtype=np.int64), outputs[keep], coefficients[keep])


def _finite_component(name, unit, output, scale, weight):
    return FieldComponent(name, unit, output, float(scale), float(weight))


def _finite_family(case_id, component, coordinates, references, measures, row_ids, extraction):
    from honf_runtime.finite_observations import FiniteObservationFamily

    return FiniteObservationFamily(
        name=component.name, component=component, support=_FINITE_SUPPORT[component.name],
        frame=_FRAME, time=_TIME, extraction_id=component.name + ':native_linear_v1',
        native_weight=component.weight, row_ids=tuple(row_ids), coordinates=np.asarray(coordinates),
        measures=np.asarray(measures), references=np.asarray(references), extraction=extraction)


def _finite_scene_families(group, case_id, model, weights, *, count, seed):
    """Read source validity transiently and retain only selected observations."""
    from honf_runtime.finite_observations import select_source_rows

    config = json.loads(group['case_config_json'][()])
    domain = config['domain']
    nx, ny = int(model.nx), int(model.ny)
    if (int(domain['nx']), int(domain['ny'])) != (nx, ny):
        raise ValueError('Thermal observation source and predictor have different cell dimensions.')
    lengths = np.asarray((domain['lx'], domain['ly']), dtype=np.float32)
    radius = np.float32(domain['module_radius'])
    if not np.isfinite(lengths).all() or np.any(lengths <= 0) or not np.isfinite(radius) or radius <= 0:
        raise ValueError('Thermal observation geometry must be finite and positive.')
    centers = np.asarray(group['module_centers']).astype(np.float32, copy=False)
    present = np.asarray(group['module_present']) > .5
    if centers.shape != (len(present), 2) or not np.isfinite(centers[present]).all():
        raise ValueError('Thermal module centres and presence must align.')
    slots = np.flatnonzero(present)
    solid = build_native_solid_mask(torch.from_numpy(centers[None]), torch.from_numpy(present[None]),
        torch.tensor([[radius]]), torch.from_numpy(lengths[None]), nx=nx, ny=ny)[0].numpy().reshape(-1)
    families = []

    rows = np.asarray(group['sampled_points'])
    point_weights = (np.asarray(group['sampled_point_weights']) if 'sampled_point_weights' in group
                     else np.ones(len(rows), dtype=rows.dtype))
    if rows.ndim != 2 or rows.shape[1] != 7 or point_weights.shape != (len(rows),):
        raise ValueError('Thermal sampled points must be [row, x, y, u, v, p, omega, T] with row weights.')
    xy = rows[:, :2]
    valid_xy = _finite_stencil_valid(xy, lengths, nx, ny)
    positive_weights = np.isfinite(point_weights) & (point_weights > 0)
    # The saved point weights are native dimensionless training weights, not
    # quadrature areas. Retain their relative values and give each source row
    # one cell area; repeated sampled rows intentionally retain multiplicity.
    cell_area = float(domain['lx']) * float(domain['ly']) / (nx * ny)
    point_mass = np.asarray(point_weights, dtype=np.result_type(point_weights.dtype, np.float64)) * cell_area
    fluid_seed = _finite_seed(seed, case_id, 'sampled_points')
    for name, unit, source_column, output, operation in _FINITE_FLUID:
        if name not in weights:
            continue
        chosen = select_source_rows(valid_xy & positive_weights & np.isfinite(rows[:, source_column + 2]),
                                    count=count, seed=fluid_seed)
        if not len(chosen):
            continue
        coordinates = xy[chosen]
        coords = torch.as_tensor(coordinates, dtype=torch.float32)
        spacing = torch.from_numpy(lengths) / torch.tensor([nx, ny], dtype=torch.float32)
        location = coords / spacing - .5
        rounded = location.round()
        tolerance = min(8 * torch.finfo(location.dtype).eps * max(nx, ny), 1.e-3)
        if bool(((location - rounded).abs() > tolerance).any()):
            raise ValueError('Native Thermal fluid observations require cell-centred source rows.')
        ix, iy = rounded[:, 0].long().numpy(), rounded[:, 1].long().numpy()
        centre = iy * nx + ix
        if operation == 'interpolated_temperature':
            nodes, coefficients, _ = _stencil(coordinates, lengths, nx, ny)
            output_indices = np.full_like(nodes, 3)
        elif operation == 'native_curl':
            left, right = np.maximum(ix - 1, 0), np.minimum(ix + 1, nx - 1)
            down, up = np.maximum(iy - 1, 0), np.minimum(iy + 1, ny - 1)
            nodes = np.stack((iy * nx + right, iy * nx + left, up * nx + ix, down * nx + ix), axis=1)
            output_indices = np.broadcast_to(np.array([1, 1, 0, 0]), nodes.shape)
            dx = (right - left).astype(np.float32) * np.float32(spacing[0])
            dy = (up - down).astype(np.float32) * np.float32(spacing[1])
            coefficients = np.stack((1 / dx, -1 / dx, -1 / dy, 1 / dy), axis=1)
            coefficients *= (~solid[nodes] & ~solid[centre, None])
        else:
            nodes = centre[:, None]
            output_indices = np.full_like(nodes, output)
            coefficients = np.ones(nodes.shape, dtype=np.float32)
            if operation in ('masked_u', 'masked_v'):
                coefficients *= ~solid[nodes]
        component = _finite_component(name, unit, output, model.field_std[source_column], weights[name])
        extraction = _finite_extraction(case_id, lengths, nx, ny, nodes, output_indices, coefficients)
        families.append(_finite_family(case_id, component, coordinates, rows[chosen, source_column + 2],
            point_mass[chosen], (f'sampled_point:{int(row)}' for row in chosen), extraction))

    local_mask = np.asarray(group['module_internal_mask'], dtype=bool)
    if local_mask.ndim != 2:
        raise ValueError('Thermal material source mask must be two-dimensional.')
    local = _local_disk_query_points(local_mask)
    material = np.asarray(group['module_internal_temperature'])
    if material.shape != (len(present), *local_mask.shape):
        raise ValueError('Thermal material values must match the module and local-mask dimensions.')
    material_xy = (torch.from_numpy(centers[slots])[:, None] + torch.tensor(radius) *
                   torch.from_numpy(local)[None]).numpy().reshape(-1, 2)
    material_ref = material[slots][:, local_mask].reshape(-1)
    material_slots = np.repeat(slots, len(local))
    local_ids = np.tile(np.flatnonzero(local_mask), len(slots))
    name = 'temperature/material'
    if name in weights:
        chosen = select_source_rows(_finite_stencil_valid(material_xy, lengths, nx, ny) & np.isfinite(material_ref),
            count=count, seed=_finite_seed(seed, case_id, 'material'))
        if len(chosen):
            nodes, coefficients, _ = _stencil(material_xy[chosen], lengths, nx, ny)
            extraction = _finite_extraction(case_id, lengths, nx, ny, nodes, np.full_like(nodes, 3), coefficients)
            component = _finite_component(name, 'native temperature units', 3,
                                           model.internal_temperature_std[0], weights[name])
            area = np.pi * float(domain['module_radius']) ** 2 / max(len(local), 1)
            ids = (f'module:{int(material_slots[row])}:internal_cell:{int(local_ids[row])}' for row in chosen)
            families.append(_finite_family(case_id, component, material_xy[chosen], material_ref[chosen],
                                            np.full(len(chosen), area), ids, extraction))

    angles = torch.arange(64, dtype=torch.float32) * (2 * torch.pi / 64)
    normals = torch.stack((angles.cos(), angles.sin()), -1)
    delta = torch.minimum((torch.from_numpy(lengths) / torch.tensor([nx, ny])).amin(), .15 * torch.tensor(radius))
    surface_xy = (torch.from_numpy(centers[slots])[:, None] + torch.tensor(radius) * normals[None]).numpy().reshape(-1, 2)
    outside_xy = (torch.from_numpy(centers[slots])[:, None] + (torch.tensor(radius) + delta) * normals[None]).numpy().reshape(-1, 2)
    interface = np.asarray(group['interface_target'])
    if interface.shape != (len(present), 64, 2):
        raise ValueError('Thermal interface targets must contain 64 two-component rows per module.')
    observed = (np.asarray(group['interface_condition_valid_mask']) > .5
                if 'interface_condition_valid_mask' in group else np.ones((len(present), 64), dtype=bool))
    if observed.shape != (len(present), 64):
        raise ValueError('Thermal interface validity must match the module and angle axes.')
    observed = observed[slots].reshape(-1)
    condition = np.asarray(group['interface_condition'])
    feature_names = group.file.get('interface_condition_feature_names')
    if feature_names is None:
        outside_column = 3
    else:
        names = tuple(item.decode() if isinstance(item, bytes) else str(item) for item in feature_names[:])
        if 'T_outside' not in names:
            raise ValueError('Thermal interface conditions do not identify outside temperature.')
        outside_column = names.index('T_outside')
    if condition.ndim != 3 or condition.shape[:2] != (len(present), 64) or outside_column >= condition.shape[-1]:
        raise ValueError('Thermal interface conditions must align with native module and angle rows.')
    outside_ref = condition[slots, :, outside_column].reshape(-1)
    surface_ref = interface[slots].reshape(-1, 2)
    surface_valid = _finite_stencil_valid(surface_xy, lengths, nx, ny)
    outside_valid = _finite_stencil_valid(outside_xy, lengths, nx, ny)
    port_slots = np.repeat(slots, 64)
    port_angles = np.tile(np.arange(64), len(slots))
    surface_seed = _finite_seed(seed, case_id, 'surface')
    material_attrs = group['material_parameters'].attrs
    solid_k = torch.tensor(float(material_attrs.get('solid_k', config['thermal'].get('solid_k', 1.))), dtype=torch.float32)
    fluid_k = torch.tensor(float(material_attrs.get('fluid_k', config['thermal'].get('fluid_k', 1.))), dtype=torch.float32)
    conductivity = 2 * solid_k * fluid_k / (solid_k + fluid_k).clamp_min(1.e-12)
    if not bool(torch.isfinite(conductivity)) or float(conductivity) <= 0 or float(delta) <= 0:
        raise ValueError('Thermal interface conductivity and offset must be finite and positive.')
    surface_specs = (
        ('temperature/surface', surface_xy, surface_ref[:, 0], surface_valid & observed,
         model.interface_target_std[0], 'native temperature units'),
        ('q_proxy', surface_xy, surface_ref[:, 1], surface_valid & outside_valid & observed,
         model.interface_target_std[1], 'native flux-proxy units'),
        ('port/outside_temperature', outside_xy, outside_ref, outside_valid,
         model.field_std[4], 'native temperature units'),
    )
    for name, coordinates, reference, valid, scale, unit in surface_specs:
        if name not in weights:
            continue
        chosen = select_source_rows(valid & np.isfinite(reference), count=count, seed=surface_seed)
        if not len(chosen):
            continue
        nodes, coefficients, _ = _stencil(coordinates[chosen], lengths, nx, ny)
        if name == 'q_proxy':
            outside_nodes, outside_weights, _ = _stencil(outside_xy[chosen], lengths, nx, ny)
            # Preserve the native FP32 stencil values and multiplier without a
            # second FP32 rounding when their sparse products are assembled.
            multiplier = float((conductivity / delta).item())
            nodes = np.concatenate((nodes, outside_nodes), axis=1)
            coefficients = np.concatenate((coefficients.astype(np.float64) * multiplier,
                                           -outside_weights.astype(np.float64) * multiplier), axis=1)
        extraction = _finite_extraction(case_id, lengths, nx, ny, nodes, np.full_like(nodes, 3), coefficients)
        component = _finite_component(name, unit, 3, scale, weights[name])
        offset = float(delta) if name == 'port/outside_temperature' else 0.
        arc_length = 2 * np.pi * (float(domain['module_radius']) + offset) / 64
        ids = (f'module:{int(port_slots[row])}:port:{int(port_angles[row])}' for row in chosen)
        families.append(_finite_family(case_id, component, coordinates[chosen], reference[chosen],
                                        np.full(len(chosen), arc_length), ids, extraction))
    return tuple(families)


class ThermalObservationBinding:
    def __init__(self, provider, recipe):
        self.provider, self.recipe = provider, recipe

    def identity_payload(self):
        model = self.provider.model
        return {'schema':'thermal_native_observation_blocks_v1', 'frame':_FRAME, 'time':_TIME,
            'block_names':list(self.block_names()),
            'components':['u','v','p','omega','temperature','material_temperature','surface_temperature','normal_flux_proxy','outside_temperature'],
            'field_scales':[float(x) for x in model.field_std.detach().cpu()],
            'material_scale':float(model.internal_temperature_std[0]),
            'surface_scales':[float(x) for x in model.interface_target_std.detach().cpu()],
            'native_weights':{str(k):float(v) for k,v in self.provider.native_weights.items()},
            'outside_weight':float(self.provider.outside_temperature_weight),
            'source':'existing base arrays; fixed physical cell centres; per-scene material and port observations',
            'source_precision':'saved reference and geometry arrays retain their source dtype; native model maps use FP32',
            'block_aggregation':'common-pair weight 1; each individual-scene support weight 0.5; native component weights retained'}

    def block_names(self):
        return ('fluid_common_pair','fluid_scene_0_only','fluid_scene_1_only',
                'material_temperatures_scene_0','material_temperatures_scene_1',
                'surface_observations_scene_0','surface_observations_scene_1',
                'outside_temperatures_scene_0','outside_temperatures_scene_1')

    def training_pairs(self):
        result = self.recipe.get('thermal_field_response',{}).get('pairs',())
        if not result:
            from .thermal_field_response import select_thermal_field_pairs
            disjoint, extended, _ = select_thermal_field_pairs(
                self.provider.data_path, self.provider.training_case_ids,
                case_split_by_id=self.provider.case_split_by_id)
            result = disjoint if self.recipe.get('observation_response', {}).get('mode') == 'coupled_native' else extended
        return tuple(tuple(map(str,p)) for p in result)

    def validation_pairs(self):
        return tuple(tuple(map(str,p)) for p in self.recipe.get('thermal_field_response',{}).get('validation_pairs',()))

    def calibration_case_ids(self):
        return tuple(self.recipe.get('observation_response',{}).get('calibration_case_ids',_TRAIN15))

    def native_case_key(self, case_id): return str(case_id)

    def finite_observation_family_weights(self):
        """Return configured native importances once, including absent scenes."""
        source = dict(self.provider.native_weights)
        source['port/outside_temperature'] = self.provider.outside_temperature_weight
        unknown = set(source) - set(_FINITE_SUPPORT)
        if unknown:
            raise ValueError(f'Unsupported Thermal native observation families: {sorted(unknown)}.')
        result = {}
        for name, weight in source.items():
            value = float(weight)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'Invalid native observation importance for {name!r}.')
            if value > 0:
                result[name] = value
        if not result:
            raise ValueError('Thermal finite observations need at least one positive native importance.')
        return result

    def finite_observation_unsupported(self):
        """State which derived source quantities cannot be sparse linear probes."""
        return {
            'module_peak_temperature': 'A maximum over material rows is nonlinear; the existing validation metric remains separate.',
            'interface_h_proxy': 'The interface h quantities use ratios, absolute values or clipping and have no supported linear extraction.',
        }

    def finite_observation_frame(self):
        """One input-only physical frame for the complete declared TRAIN set."""
        from honf_runtime.finite_observations import AffineFrame

        training = tuple(map(str, self.provider.training_case_ids))
        if not training:
            raise ValueError('A Thermal coordinate frame requires a declared TRAIN membership.')
        maximum = np.zeros(2, dtype=np.float64)
        with h5py.File(self.provider.data_path, 'r') as data:
            for case_id in training:
                config = json.loads(data['cases'][case_id]['case_config_json'][()])
                lengths = np.asarray((config['domain']['lx'], config['domain']['ly']), dtype=np.float64)
                if not np.isfinite(lengths).all() or np.any(lengths <= 0):
                    raise ValueError(f'Invalid TRAIN physical domain for Thermal case {case_id}.')
                maximum = np.maximum(maximum, lengths)
        return AffineFrame(_FRAME, (0., 0.), tuple(maximum))

    def finite_observation_scene(self, case_id, *, count=64, seed=0):
        """Select existing native rows independently for this scene and family.

        The complete source arrays and validity masks are temporary; returned
        references, identities and linear extraction maps contain only selected
        rows and the cell-centre dependencies needed to evaluate them.
        """
        from honf_runtime.finite_observations import FiniteObservationScene

        case_id = str(case_id)
        allowed = set(map(str, self.provider.training_case_ids)) | set(map(str, self.provider.validation_case_ids))
        if case_id not in allowed:
            raise ValueError('Thermal finite observations require a declared TRAIN or validation case.')
        if type(count) is not int or not 1 <= count <= 64 or type(seed) is not int or seed < 0:
            raise ValueError('Thermal finite observations require 1–64 rows and a nonnegative seed.')
        weights = self.finite_observation_family_weights()
        with h5py.File(self.provider.data_path, 'r') as data:
            families = _finite_scene_families(data['cases'][case_id], case_id, self.provider.model,
                                               weights, count=count, seed=seed)
        return FiniteObservationScene(case_id, families)

    def complete_source_observation_scene_for_evaluation(self, case_id):
        """Describe one complete source support for immediate, decoded-field review.

        This separate diagnostic is intentionally not used by the fixed-probe
        training interface or cached across cases. The caller reduces this
        temporary one-scene object against already-decoded physical fields.
        """
        from honf_runtime.finite_observations import FiniteObservationScene

        case_id = str(case_id)
        allowed = set(map(str, self.provider.training_case_ids)) | set(map(str, self.provider.validation_case_ids))
        if case_id not in allowed:
            raise ValueError('Thermal complete-source review requires a declared TRAIN or validation case.')
        with h5py.File(self.provider.data_path, 'r') as data:
            group = data['cases'][case_id]
            present_count = int((np.asarray(group['module_present']) > .5).sum())
            local_count = int(np.asarray(group['module_internal_mask'], dtype=bool).sum())
            maximum = max(1, len(group['sampled_points']), present_count * local_count, present_count * 64)
            families = _finite_scene_families(group, case_id, self.provider.model,
                self.finite_observation_family_weights(), count=maximum, seed=0)
        return FiniteObservationScene(case_id, families)

    def execute_finite_observation_dependencies(self, model, case_ids, dependencies_by_case, *,
                                                read_mode, native_batch=None, native_predictions=None):
        """Decode selected dependencies together, reusing the live native organization.

        Unrequested scenes in a reused native microbatch receive masked padding;
        no prepared state is sliced, detached, or saved beyond this call. Without
        a reusable prediction, prepare the requested native scene batch once.
        Returned tensors are physical [view, dependency, four joint outputs].
        """
        if read_mode is None:
            read_mode = 'paired'
        if model is not self.provider.model or read_mode not in ('full', 'restricted', 'paired'):
            raise ValueError('Thermal finite execution needs its bound model and full, restricted or paired reader.')
        requested = tuple(map(str, case_ids))
        if not requested or len(set(requested)) != len(requested):
            raise ValueError('Thermal finite execution needs unique requested case IDs.')
        if set(dependencies_by_case) != set(requested):
            raise ValueError('Thermal dependency lists must cover exactly the requested cases.')
        dependencies_by_case = {case: tuple(dependencies_by_case[case]) for case in requested}
        for case, dependencies in dependencies_by_case.items():
            if any(dep.scene_id != case or dep.frame != _FRAME or dep.time != _TIME
                   or not isinstance(dep.native_id, (int, np.integer))
                   or not 0 <= int(dep.native_id) < model.nx * model.ny
                   for dep in dependencies):
                raise ValueError('A Thermal dependency has the wrong scene, physical frame, time or cell identity.')
        count = sum(len(dependencies) for dependencies in dependencies_by_case.values())
        view_count = 2 if read_mode == 'paired' else 1
        work = {'neural_queries': count, 'padded_queries': 0, 'padding_queries': 0, 'local_pairs': 0,
                'scene_preparations': 0, 'reused_scene_preparations': 0}
        if count == 0:
            parameter = next(model.parameters())
            return {case: parameter.new_empty((view_count, 0, 4)) for case in requested}, work

        if native_predictions is not None and native_batch is None:
            raise ValueError('A prepared native prediction requires its original native batch.')
        if native_batch is None:
            splits = {self.provider.case_split_by_id[case] for case in requested}
            if len(splits) != 1:
                raise ValueError('Thermal fallback batches must share one declared data partition.')
            cases = self.provider._load_native_cases(requested)
            native_batch = self.provider._native_case_batch(cases, range(len(cases)), split=next(iter(splits)))
        batch_cases = tuple(map(str, native_batch.scene_inputs.case_ids))
        target_cases = tuple(map(str, native_batch.targets.case_ids))
        if (not batch_cases or len(set(batch_cases)) != len(batch_cases) or batch_cases != target_cases
                or not set(requested).issubset(batch_cases)):
            raise ValueError('Thermal native batch case order does not match its targets or requested scenes.')

        from .joint_member_routing_task import ThermalJointPairPredictions

        native_views = ((native_predictions.full, native_predictions.restricted)
                        if isinstance(native_predictions, ThermalJointPairPredictions) else (native_predictions,))
        prepared = getattr(native_views[0], 'native_prepared', None) if native_views[0] is not None else None
        if prepared is not None:
            if (getattr(prepared, 'owner', None) != id(model) or prepared.scene_key != batch_cases
                    or any(getattr(view, 'native_prepared', None) is not prepared for view in native_views)
                    or prepared.scene_batch.scene_key != prepared.organization.scene_key):
                raise ValueError('Thermal native organization does not belong to this model and ordered batch.')
            # prepare_native shallow-copies this exact structure and heat; an
            # identity check prevents reusing an unrelated batch with same IDs.
            if (any(prepared.structure.get(key) is not value
                    for key, value in native_batch.scene_inputs.structure.items())
                    or prepared.structure.get('heat_powers') is not native_batch.receivers.heat):
                raise ValueError('Thermal native organization was prepared from different scene inputs.')
            scene, organization = prepared.scene_batch, prepared.organization
            work['reused_scene_preparations'] = 1
        else:
            structure = {**native_batch.scene_inputs.structure, 'heat_powers': native_batch.receivers.heat}
            scene = model.scene_batch(structure, scene_key=batch_cases)
            organization = model.core.prepare(scene)
            work['scene_preparations'] = 1
            work.update({f'prepare_{name}': int(value)
                         for name, value in organization.work_counts.items()})

        parameter = next(model.parameters())
        width = max(len(dependencies_by_case.get(case, ())) for case in batch_cases)
        coordinates = parameter.new_zeros((len(batch_cases), width, 2))
        indices = torch.zeros((len(batch_cases), width), dtype=torch.long, device=parameter.device)
        mask = torch.zeros((len(batch_cases), width), dtype=torch.bool, device=parameter.device)
        for batch_index, case in enumerate(batch_cases):
            dependencies = dependencies_by_case.get(case, ())
            if dependencies:
                size = len(dependencies)
                coordinates[batch_index, :size] = torch.as_tensor(
                    [dep.coordinates for dep in dependencies], dtype=parameter.dtype, device=parameter.device)
                indices[batch_index, :size] = torch.as_tensor(
                    [dep.native_id for dep in dependencies], dtype=torch.long, device=parameter.device)
                mask[batch_index, :size] = True
        packed = model._packed_local_pairs(scene, coordinates, mask)
        query = QueryBatch(coordinates=coordinates, query_mask=mask, query_ids=indices,
            local_batch_indices=packed[0], local_query_indices=packed[1], local_module_indices=packed[2],
            local_weights=packed[3], scene_key=scene.scene_key, coordinate_frame_key=scene.coordinate_frame_key,
            receiver_key=('thermal_finite_observation_batch', tuple(len(dependencies_by_case.get(c, ())) for c in batch_cases)))
        if getattr(model.core, 'locator_mode', None) is not None:
            query = attach_thermal_e_anchor_query(query, scene, model.core.locator_mode,
                environment_nx=model.environment_nx, environment_ny=model.environment_ny)
            # Padded and zero-weight coverage slots remain masked, with valid
            # gather indices even for implementations that inspect padding.
            query = replace(query, anchor_indices=query.anchor_indices.clamp_min(0))
        if read_mode == 'paired':
            full, restricted, shared = model.core.decode_paired(organization, query)
            outputs = (full, restricted)
            counts = ((f'paired_{name}', output.work_counts) for name, output in zip(('full', 'restricted'), outputs))
            work.update({f'paired_{name}': int(value) for name, value in shared.items()})
        else:
            output = model.core.decode(organization, query, intervention={'read_mode': read_mode})
            outputs = (output,)
            counts = ((f'single_{read_mode}', output.work_counts),)
        for view, values in counts:
            work.update({f'{view}_{name}': int(value) for name, value in values.items()})
        raw = torch.stack([output.joint_fields for output in outputs])
        if raw.shape != (view_count, len(batch_cases), width, 4):
            raise ValueError('Thermal finite decoder did not return one four-field output per query slot.')
        scale = torch.stack((model.field_std[0], model.field_std[1], model.field_std[2], model.field_std[4])).to(raw.device)
        offset = torch.stack((model.field_mean[0], model.field_mean[1], model.field_mean[2], model.field_mean[4])).to(raw.device)
        dtype = torch.promote_types(raw.dtype, torch.promote_types(scale.dtype, offset.dtype))
        physical = raw.to(dtype) * scale.to(dtype) + offset.to(dtype)
        work['padded_queries'] = len(batch_cases) * width
        work['padding_queries'] = work['padded_queries'] - count
        work['local_pairs'] = int(packed[3].numel())
        positions = {case: index for index, case in enumerate(batch_cases)}
        return {case: physical[:, positions[case], :len(dependencies_by_case[case])] for case in requested}, work

    def coupled_native_population(self, case_id):
        """Describe the provider-loaded row population and its native extraction.

        The five native field observations reuse the existing interpolation and
        curl output. Material and port records have no cross-scene member
        correspondence and continue to receive their ordinary native losses.
        """
        from honf_runtime.coupled_observations import NativeObservationPopulation, ObservationKey

        case_id = str(case_id)
        cases = {str(case['case_id']): case for case in
                 (*self.provider.training_cases, *self.provider.validation_cases)}
        if case_id not in cases:
            raise ValueError('Coupled observations must be loaded TRAIN or declared validation records.')
        case = cases[case_id]
        xy = np.asarray(case['query_xy'])
        reference = np.asarray(case['field_targets'])
        weights = np.asarray(case['point_weights'])
        if xy.ndim != 2 or xy.shape[1] != 2 or reference.shape != (len(xy), 5) or weights.shape != (len(xy),):
            raise ValueError('The actual native fluid row population is malformed.')
        model = self.provider.model
        lengths = np.asarray([np.asarray(case['structure'][name]).reshape(-1)[0]
                              for name in ('domain_length_x', 'domain_length_y')], dtype=np.float32)
        coords = torch.as_tensor(xy, dtype=torch.float32).reshape(1, -1, 2)
        lengths_tensor = torch.as_tensor(lengths).reshape(1, 2)
        stencil = native_cell_center_stencil(coords, lengths_tensor, nx=model.nx, ny=model.ny)
        indices = stencil.indices[0].numpy()
        interpolation_weights = stencil.weights[0].numpy()
        spacing = lengths_tensor / lengths_tensor.new_tensor([model.nx, model.ny])
        cell = (coords / spacing[:, None] - .5).round().long()[0].numpy()
        if not np.allclose(coords.numpy()[0] / spacing.numpy()[0] - .5, cell, rtol=0,
                           atol=min(8 * np.finfo(np.float32).eps * max(model.nx, model.ny), 1.e-3)):
            raise ValueError('Native curl observations require the actual cell-centred provider rows.')
        x, y = cell[:, 0], cell[:, 1]
        centre = y * model.nx + x
        neighbours = np.stack((centre, y * model.nx + np.maximum(x-1, 0),
                               y * model.nx + np.minimum(x+1, model.nx-1),
                               np.maximum(y-1, 0) * model.nx + x,
                               np.minimum(y+1, model.ny-1) * model.nx + x), axis=1)
        dependencies = tuple(((int(centre[i]),), (int(centre[i]),), (int(centre[i]),),
                              tuple(map(int, neighbours[i])),
                              tuple(map(int, indices[i][interpolation_weights[i] != 0])))
                             for i in range(len(xy)))
        names = ('flow/u', 'flow/v', 'flow/p', 'flow/omega', 'temperature/fluid')
        units = ('native velocity units', 'native velocity units', 'native pressure units',
                 'native vorticity units', 'native temperature units')
        components = tuple(FieldComponent(name, units[i], i, float(model.field_std[i]),
                                          float(self.provider.native_weights[name])) for i, name in enumerate(names))
        keys = tuple(ObservationKey(_FRAME, _TIME, tuple(map(float, point)),
                                    'fixed_physical_frame', 'native_fluid_observation') for point in xy)
        saved_mask = case.get('module_mask')
        if saved_mask is None:
            with h5py.File(self.provider.data_path, 'r') as data:
                saved_mask = np.asarray(data['cases'][case_id]['module_mask'])
        solid = np.asarray(saved_mask)
        if solid.shape != (model.ny, model.nx) or solid.dtype.kind not in 'bui' or not np.isin(solid, (0, 1)).all():
            raise ValueError('Native fluid response requires the saved scene geometry mask.')
        valid = ((weights[:, None] > 0) & np.isfinite(reference) & ~solid.reshape(-1)[centre, None].astype(bool)
                 & stencil.valid[0].numpy()[:, None])
        mass = np.broadcast_to(weights[:, None], reference.shape).copy()
        return NativeObservationPopulation(keys, components, reference, mass, valid,
                                           np.ones_like(valid), dependencies)

    def coupled_native_budget(self, case_id):
        del case_id
        return int(self.provider.primary_queries)

    def coupled_native_sampling_law(self, case_id):
        del case_id
        return 'uniform_without_replacement'

    def coupled_view_loss_names(self):
        return (*self.provider.native_weights, 'port/outside_temperature')

    def coupled_validation_batch(self, pair, sample):
        from honf_runtime.unified_training import TaskBatch
        from .content_hypergraph_task import (ThermalContentSceneInputs, ThermalContentReceivers,
                                              ThermalContentTargets, THERMAL_CONTENT_SURFACE_POINTS)
        from .unified_task import _sample_primary, _structure_only
        from .content_hypergraph_task import RESPONSE_SURFACE_STRIDE

        lookup = {str(case['case_id']): index for index, case in enumerate(self.provider.validation_cases)}
        if not set(pair) <= set(lookup):
            raise ValueError('Coupled validation pairs must use loaded validation records.')
        indices = tuple(lookup[case] for case in pair)
        sampled, _ = _sample_primary(self.provider.validation_cases, indices, epoch=0, key=None,
            training=False, budget={'fluid_queries': self.provider.primary_queries,
                'material_queries_per_module': 32, 'surface_stride': RESPONSE_SURFACE_STRIDE},
            device=self.provider.device, sampling_audit=False,
            validation_sampling_indices=getattr(self.provider, 'validation_sampling_indices', None),
            fluid_rows_by_case={pair[0]: sample.left_rows, pair[1]: sample.right_rows})
        batch = TaskBatch(scene_inputs=ThermalContentSceneInputs(_structure_only(sampled['structure']), case_ids=pair),
            receivers=ThermalContentReceivers(sampled['query_xy'], sampled['module_internal_query_points'],
                sampled['structure']['heat_powers'], ntheta=THERMAL_CONTENT_SURFACE_POINTS,
                response_ntheta=THERMAL_CONTENT_SURFACE_POINTS),
            targets=ThermalContentTargets(case_ids=pair, case_indices=indices, epoch=0,
                field_targets=sampled['field_targets'], point_weights=sampled['point_weights'],
                interface_target=sampled['interface_target'],
                interface_valid_mask=sampled['interface_condition_valid_mask'],
                material_targets=sampled['module_internal_temperature_points'], response=None),
            auxiliary={'coupled_native': ((pair, sample),)}, case_keys=pair)
        return self.provider._with_native_ports(batch)

    def coupled_native_extraction(self, predictions, batch_index, batch):
        """Expose physical native values and the IDs actually decoded for a scene."""
        from .joint_member_routing_task import ThermalJointPairPredictions

        del batch
        views = ((predictions.full, predictions.restricted)
                 if isinstance(predictions, ThermalJointPairPredictions) else (predictions,))
        fields = torch.stack([view.native_main['pred_field'][batch_index] for view in views])
        prepared = views[0].native_prepared
        if not hasattr(prepared, '_coupled_actual_native_ids'):
            # Padding repeats the first real ID, so all IDs in a padded row are
            # actual executed IDs. Transfer once for the entire microbatch.
            ids = prepared.union.grid_indices.detach().cpu().numpy()
            prepared._coupled_actual_native_ids = tuple(tuple(map(int, row)) for row in ids)
        return fields, prepared._coupled_actual_native_ids[batch_index]

    def validate_pair(self, pair):
        allowed = set(map(str,self.provider.training_case_ids)) | set(map(str,self.provider.validation_case_ids))
        if len(pair) != 2 or not set(pair) <= allowed or self.provider.case_split_by_id[pair[0]] != self.provider.case_split_by_id[pair[1]]:
            raise ValueError('Thermal observation pairs must remain within the selected native partition.')

    def for_evaluation(self, pairs):
        """Bind the explicitly declared historically exposed panel read-only."""
        requested = {case for pair in pairs for case in pair}
        permitted = set(map(str, self.recipe.get('observation_response', {}).get('evaluation_case_ids', _ASSESSMENT9)))
        existing = set(map(str, self.provider.training_case_ids)) | set(map(str, self.provider.validation_case_ids))
        if not requested <= existing | permitted:
            raise ValueError('Observation evaluation case lies outside the declared exposed panel.')
        copied = copy.copy(self.provider)
        copied.case_split_by_id = dict(self.provider.case_split_by_id)
        with h5py.File(self.provider.data_path, 'r') as h5:
            for case in requested - set(copied.case_split_by_id):
                if case not in h5['cases']:
                    raise ValueError(f'Unknown Thermal evaluation case {case}.')
                copied.case_split_by_id[case] = str(h5['cases'][case].attrs['split'])
        copied.validation_case_ids = tuple(dict.fromkeys((*map(str, self.provider.validation_case_ids), *sorted(requested-existing))))
        binding = copy.copy(self)
        binding.provider = copied
        for pair in pairs:
            binding.validate_pair(pair)
        return binding

    def observation_blocks(self, pair):
        self.validate_pair(pair)
        with h5py.File(self.provider.data_path,'r') as h5:
            groups = [h5['cases'][case_id] for case_id in pair]
            configs = [json.loads(group['case_config_json'][()]) for group in groups]
            split = self.provider.case_split_by_id[pair[0]]
            if describe_native_scene(pair[0],split,configs[0]).compatibility_key != describe_native_scene(pair[1],split,configs[1]).compatibility_key:
                raise ValueError('Thermal pair differs in prescribed native physics or fixed frame.')
            xy = _coordinates(groups[0])
            names = ('flow/u','flow/v','flow/p','flow/omega','temperature/fluid')
            fields = _fluid_block(groups,configs,pair,self.provider.model.field_std.detach().cpu().numpy(),
                                  [self.provider.native_weights[n] for n in names])
            additional = _individual_blocks(groups,configs,pair,xy,self.provider.model,
                                              self.provider.native_weights,self.provider.outside_temperature_weight)
        common = fields.pair_mask
        paired = replace(fields, name='fluid_common_pair', absolute_mask=np.stack((common, common)),
                         pair_mask=common, edge_mask=common, block_weight=1.)
        individual = []
        for side in range(2):
            own = fields.absolute_mask[side] & ~fields.absolute_mask[1-side]
            masks = np.zeros_like(fields.absolute_mask)
            masks[side] = own
            zero = np.zeros_like(fields.pair_mask)
            labels, _ = label(own.any(axis=1).reshape(xy.shape[:2]),
                              structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
            group_ids = tuple(('individual', side, int(value)) if value else ('unused', int(i))
                              for i, value in enumerate(labels.reshape(-1)))
            individual.append(replace(fields, name=f'fluid_scene_{side}_only', absolute_mask=masks,
                pair_mask=zero, edge_mask=zero, block_weight=.5, correspondence='none', neighbor_groups=group_ids))
        # Each native material/port list is separately eligible and separately
        # sampled. Original module slots remain local to their source scene.
        singles = []
        for block in additional:
            for side in range(2):
                masks = np.zeros_like(block.absolute_mask)
                masks[side] = block.absolute_mask[side]
                singles.append(replace(block, name=f'{block.name}_scene_{side}', absolute_mask=masks,
                                       block_weight=.5))
        return (paired, *individual, *singles)

    def field_request(self, model, case_id, dependencies):
        cases = self.provider._load_native_cases((case_id,))
        split = self.provider.case_split_by_id[case_id]
        parameter = next(model.parameters())
        batch = self.provider._native_case_batch(cases, [0], split=split)
        structure = {**batch.scene_inputs.structure, 'heat_powers': batch.receivers.heat}
        scene = model.scene_batch(structure, scene_key=(case_id,))
        coords = torch.as_tensor([dep.coordinates for dep in dependencies], dtype=parameter.dtype,
                                 device=parameter.device)[None]
        ids = torch.as_tensor([dep.native_id for dep in dependencies], device=parameter.device)[None]
        mask = torch.ones(coords.shape[:2], dtype=torch.bool, device=parameter.device)
        packed = model._packed_local_pairs(scene, coords, mask)
        query = QueryBatch(coordinates=coords, query_mask=mask, query_ids=ids,
            local_batch_indices=packed[0], local_query_indices=packed[1], local_module_indices=packed[2],
            local_weights=packed[3], scene_key=scene.scene_key, coordinate_frame_key=scene.coordinate_frame_key,
            receiver_key=('thermal_observation_cell_union', len(dependencies)))
        query = attach_thermal_e_anchor_query(query, scene, model.core.locator_mode,
                environment_nx=model.environment_nx, environment_ny=model.environment_ny)
        mean = torch.stack((model.field_mean[0], model.field_mean[1], model.field_mean[2], model.field_mean[4]))
        std = torch.stack((model.field_std[0], model.field_std[1], model.field_std[2], model.field_std[4]))
        return scene, query, std, mean


def build_observation_capability(recipe, *, device, resume_payload=None):
    base = copy.deepcopy(recipe)
    base['training_profile'] = 'native_pair'
    base['thermal_native_pair'] = {'variant':'field_only','prediction_batching':'paired'}
    if recipe.get('observation_response', {}).get('mode') == 'coupled_native':
        base['coupled_native_runtime'] = True
    base.pop('observation_response',None)
    model, provider = build_thermal_native_pair_task(base,device=device,resume_payload=resume_payload,
        _field_response_base=True,
        _fresh_coupled_base=recipe.get('observation_response', {}).get('mode') == 'coupled_native')
    return model,provider,ThermalObservationBinding(provider,recipe)

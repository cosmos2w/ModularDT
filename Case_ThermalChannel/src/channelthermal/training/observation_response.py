"""Bind existing Thermal fields and native linear observation maps.

No objective or edited-scene labels live here. Physical cell-centre fields are
compared in the fixed frame. Material and port records are individually
observed: module slots from independently generated scenes are not identities.
"""
from __future__ import annotations

import copy
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
from ..native_observations import native_cell_center_stencil
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
            _, result, _ = select_thermal_field_pairs(self.provider.data_path, self.provider.training_case_ids,
                                                       case_split_by_id=self.provider.case_split_by_id)
        return tuple(tuple(map(str,p)) for p in result)

    def validation_pairs(self):
        return tuple(tuple(map(str,p)) for p in self.recipe.get('thermal_field_response',{}).get('validation_pairs',()))

    def calibration_case_ids(self):
        return tuple(self.recipe.get('observation_response',{}).get('calibration_case_ids',_TRAIN15))

    def native_case_key(self, case_id): return str(case_id)

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
    base.pop('observation_response',None)
    model, provider = build_thermal_native_pair_task(base,device=device,resume_payload=resume_payload,
                                                      _field_response_base=True)
    return model,provider,ThermalObservationBinding(provider,recipe)

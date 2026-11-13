"""Versioned, self-contained Thermal finite-response HDF5 reader.

Physical observations and the effective raw-frame view remain separate. A new
read-only handle is opened for each operation, so forked loader processes never
share an HDF5 handle. Paths recorded in metadata are provenance, not inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

SCHEMA = 'thermal_finite_responses_v1'
RAW_VIEW = 'r17_1_raw'
LEGACY_VIEW = 'r16_10_legacy'


def _text(dataset):
    value = dataset[()]
    if isinstance(value, np.ndarray):
        return value.tobytes().decode('utf-8')
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def _json(dataset):
    return json.loads(_text(dataset))


def _arrays(group):
    return {name: (_arrays(value) if isinstance(value, h5py.Group) else value[()])
            for name, value in group.items()}


def _equal_arrays(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal_arrays(left[k], right[k]) for k in left)
    a, b = np.asarray(left), np.asarray(right)
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a, b))


def raw_endpoint_from_arrays(config, raw, *, raw_case_dir):
    """Construct the literal raw-frame effective view without changing precision."""
    field = np.stack([raw[k] for k in ('u', 'v', 'p', 'omega', 'temperature')], axis=-1)
    response = raw['interface_response']
    names = [s.decode() if isinstance(s, bytes) else str(s) for s in raw['interface_feature_names']]
    indices = [names.index(s) for s in ('theta', 'normal_x', 'normal_y', 'T_outside', 'T_surface', 'q_normal')]
    theta, nxv, nyv, outside, surface, q = (response[..., i] for i in indices)
    difference = surface - outside
    denominator = np.where(np.abs(difference) < 1.e-3, np.where(difference < 0, -1.e-3, 1.e-3), difference).astype(np.float32)
    h = q / denominator
    h_valid = (np.abs(difference) >= 1.e-3) & np.isfinite(q) & np.isfinite(h) & (h >= 0) & (h <= 1.e4)
    teacher = np.stack((theta, nxv, nyv, outside, np.clip(np.nan_to_num(h, nan=0., posinf=1.e4, neginf=0.), 0., 1.e4)), -1).astype(np.float32)
    d = config['domain']; centers = np.asarray(config['layout']['centers'], dtype=np.float64)
    radius = float(d['module_radius']); delta = min(d['lx']/d['nx'], d['ly']/d['ny'], .15*radius)
    angle = np.arange(response.shape[1], dtype=np.float64) * 2*np.pi / response.shape[1]
    xy = centers[:, None, :] + (radius + delta) * np.stack([np.cos(angle), np.sin(angle)], -1)[None]
    outside_valid = ((xy[..., 0] >= .5*d['lx']/d['nx']) & (xy[..., 0] <= d['lx']-.5*d['lx']/d['nx']) &
                     (xy[..., 1] >= .5*d['ly']/d['ny']) & (xy[..., 1] <= d['ly']-.5*d['ly']/d['ny']))
    return {'raw_case_dir': str(raw_case_dir), 'config': config, 'field': field,
            'fluid_mask': ~raw['module_mask'].astype(bool), 'module_id_grid': raw['module_id'],
            'material': raw['module_internal_temperature'], 'material_mask': raw['module_internal_mask'].astype(bool),
            'interface_target': np.stack((surface, q), -1), 'teacher_ports': teacher,
            'teacher_h_valid': h_valid, 'outside_valid': outside_valid}


class FiniteResponseStore:
    """Read scoped ordered families and endpoint observations from one pack."""

    def __init__(self, path):
        self.path = str(Path(path).expanduser())
        with h5py.File(self.path, 'r') as h5:
            if h5.attrs.get('schema') != SCHEMA or not bool(h5.attrs.get('complete', False)):
                raise ValueError('Unsupported or incomplete Thermal finite-response pack.')
            rows = _json(h5['index/families_json'])
        self._rows = {str(row['family_id']): row for row in rows}
        if len(self._rows) != len(rows):
            raise ValueError('Duplicate family identities in finite-response index.')
        self._ordered = tuple(self._rows)

    def _family(self, family, *, role, population):
        family = str(family)
        row = self._rows.get(family)
        if row is None or row['role'] != role:
            raise ValueError('Family is outside the requested response role.')
        if population is not None and (role != 'fit' or population not in row['populations']):
            raise ValueError('Family is outside the requested FIT population.')
        return row

    def family_ids(self, *, role, population=None, source=None):
        if population is not None and (role != 'fit' or population not in ('TRAIN150', 'TRAIN600')):
            raise ValueError('Population filtering requires TRAIN150 or TRAIN600 FIT.')
        return tuple(family for family in self._ordered
                     if self._rows[family]['role'] == role
                     and (population is None or population in self._rows[family]['populations'])
                     and (source is None or self._rows[family]['source'] == source))

    def actions(self, family, *, role, population=None):
        row = self._family(family, role=role, population=population)
        return tuple(a['action'] for a in row['actions'] if a['role'] == role
                     and (population is None or population in a['populations']))

    def _action(self, family, action, *, role, population):
        row = self._family(family, role=role, population=population)
        for item in row['actions']:
            if item['action'] == action and item['role'] == role and (population is None or population in item['populations']):
                return item
        raise ValueError('Action is outside the requested role or FIT population.')

    def family_metadata(self, family, *, role, population=None):
        row = self._family(family, role=role, population=population)
        with h5py.File(self.path, 'r') as h5:
            return _json(h5[f"families/{row['key']}/metadata_json"])

    def action_metadata(self, family, action, *, role, population=None):
        row = self._family(family, role=role, population=population)
        item = self._action(family, action, role=role, population=population)
        with h5py.File(self.path, 'r') as h5:
            return _json(h5[f"families/{row['key']}/actions/{item['key']}/metadata_json"])

    def _endpoint_key(self, family, action, *, role, population):
        row = self._family(family, role=role, population=population)
        return row['baseline_endpoint'] if action == 'baseline' else self._action(family, action, role=role, population=population)['endpoint']

    def endpoint(self, family, action='baseline', *, role, population=None, view=RAW_VIEW):
        if view != RAW_VIEW:
            raise ValueError('Use raw endpoint plus observation_record for the explicit legacy view.')
        key = self._endpoint_key(family, action, role=role, population=population)
        return self._raw_endpoint(key)

    def _raw_endpoint(self, key):
        with h5py.File(self.path, 'r') as h5:
            group = h5[f'endpoints/{key}']
            config = _json(group['configuration/actual_json'])
            raw = {name: dataset[...] for name, dataset in group['raw_frame/arrays'].items()}
            source = _json(group['provenance_json'])['raw_case_dir']
        return raw_endpoint_from_arrays(config, raw, raw_case_dir=source)

    def reference_metadata(self, native_id):
        with h5py.File(self.path, 'r') as h5:
            return _json(h5[f'reference_baselines/{str(native_id)}/metadata_json'])

    def reference_baseline(self, native_id):
        """Read an input-selected TRAIN15/VAL9 baseline, preserving raw precision."""
        return self._raw_endpoint(self.reference_metadata(native_id)['endpoint'])

    def reference_provenance(self, native_id):
        key = self.reference_metadata(native_id)['endpoint']
        with h5py.File(self.path, 'r') as h5:
            return _json(h5[f'endpoints/{key}/provenance_json'])

    def reference_frame_index(self, native_id):
        key = self.reference_metadata(native_id)['endpoint']
        with h5py.File(self.path, 'r') as h5:
            return _text(h5[f'endpoints/{key}/raw_frame/frame_index_csv'])

    def native_baseline(self, family, *, role, population=None):
        row = self._family(family, role=role, population=population)
        native_id = row['native_case_id']
        if native_id is None:
            raise ValueError('This family has no native-H5 baseline mapping.')
        with h5py.File(self.path, 'r') as h5:
            return _arrays(h5[f'catalogues/{native_id}'])

    def legacy_view_components(self, family, action, *, role, population=None):
        """Expose every literal R16-10 input and label component for this pair.

        Original families use the native-H5 baseline and edited record-backed
        field/surface/material, plus raw baseline and edited outside ports. New
        families use both raw endpoints. Missing required records fail explicitly.
        """
        row = self._family(family, role=role, population=population)
        self._action(family, action, role=role, population=population)
        if role != 'fit' or row['source'] not in ('original', 'new'):
            raise ValueError('The R16-10 training view is only defined for original/new FIT families.')
        result = {'profile': 'native_baseline_record_edit' if row['source'] == 'original' else 'raw_pair',
                  'native_baseline': self.native_baseline(family, role=role, population=population),
                  'raw_baseline': self.endpoint(family, role=role, population=population),
                  'raw_edited': self.endpoint(family, action, role=role, population=population),
                  'edited_record': None}
        if row['source'] == 'original':
            from ..interaction_evidence.storage import solve_record_from_arrays
            metadata, arrays = self.observation_record(family, action, role=role, population=population)
            result['edited_record'] = solve_record_from_arrays(metadata, arrays)
            if result['edited_record'].output is None:
                raise ValueError('Legacy edited observation record has no physical output.')
        return result

    def legacy_pair(self, provider, family, action, *, role, population=None, stage='r16_10'):
        """Build the maintained original/new legacy pair with exact saved labels.

        ``response_pair`` selects the historical original native-baseline
        stage; ``r16_10`` selects its objective's subsequent raw-baseline
        rebuild. The new-family legacy path uses raw endpoints at both stages.
        Native-state provider access is allowed, but its case must equal the
        packed catalogue before it is used for collation.
        """
        if stage not in ('response_pair', 'r16_10'):
            raise ValueError('Unknown literal legacy stage.')
        components = self.legacy_view_components(family, action, role=role, population=population)
        with h5py.File(provider.data_path, 'r') as h5:
            provider_native = _arrays(h5[f'cases/{family}'])
        if not _equal_arrays(provider_native, components['native_baseline']):
            raise ValueError('Provider native case differs from the packed legacy baseline.')
        original = provider._load_native_cases([str(family)])[0]
        from ..training.thermal_state_response_task import _native_endpoint_batch
        raw_b, raw_e = components['raw_baseline'], components['raw_edited']
        if not np.array_equal(raw_b['material_mask'], raw_e['material_mask']):
            raise ValueError('Legacy material receiver ordering differs.')
        if components['profile'] == 'native_baseline_record_edit' and stage == 'response_pair':
            from ..training.content_hypergraph_task import ThermalContentHypergraphTask
            baseline = ThermalContentHypergraphTask._native_case_batch(provider, [original], [0], split='train', ntheta=64)
        else:
            baseline = _native_endpoint_batch(provider, original, raw_b, 'baseline')
        edited = _native_endpoint_batch(provider, original, raw_e, action, components['edited_record'])
        import torch
        if not torch.equal(baseline.receivers.heat, edited.receivers.heat):
            raise ValueError('A legacy response edit changed prescribed heat.')
        return {'family_id': str(family), 'action': action, 'role': role,
                'baseline': baseline, 'edited': edited, 'raw_baseline': raw_b, 'raw_edited': raw_e}

    def observation_record(self, family, action, *, role, population=None):
        """Return exact saved record metadata and typed arrays; never substitute raw."""
        key = self._endpoint_key(family, action, role=role, population=population)
        with h5py.File(self.path, 'r') as h5:
            endpoint = h5[f'endpoints/{key}']
            if 'observation_record' not in endpoint:
                raise ValueError('This endpoint has no saved physical-observation record.')
            group = endpoint['observation_record']
            return _json(group['metadata_json']), {name: array[...] for name, array in group['arrays'].items()}

    def physical_delta(self, family, action, *, role, population=None, kind='saved'):
        if kind not in ('saved', 'derived'):
            raise ValueError('Physical delta kind must be saved or derived.')
        row = self._family(family, role=role, population=population)
        item = self._action(family, action, role=role, population=population)
        with h5py.File(self.path, 'r') as h5:
            group = h5[f"families/{row['key']}/actions/{item['key']}"]
            if f'{kind}_delta_json' not in group:
                raise ValueError(f'This action has no {kind} physical delta.')
            return _json(group[f'{kind}_delta_json'])

    def physical_delta_provenance(self, family, action, *, role, population=None, kind='saved'):
        row = self._family(family, role=role, population=population)
        item = self._action(family, action, role=role, population=population)
        with h5py.File(self.path, 'r') as h5:
            group = h5[f"families/{row['key']}"]
            if kind == 'saved':
                path = f"actions/{item['key']}/saved_delta_provenance_json"
                if path not in group:
                    raise ValueError('This action has no saved physical delta.')
                return _json(group[path])
            if kind == 'derived':
                return _json(group['functional_definition_json'])
            raise ValueError('Physical delta kind must be saved or derived.')

    def population_metadata(self, population):
        if population not in ('TRAIN150', 'TRAIN600'):
            raise ValueError('Unknown training population.')
        with h5py.File(self.path, 'r') as h5:
            return _json(h5[f'populations/{population}/metadata_json'])

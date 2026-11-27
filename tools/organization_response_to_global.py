#!/usr/bin/env python3
"""Convert a saved edit's actual organization capture for the global plotter.

This reads only the diagnostic JSON/NPZ; it does not load a model or create
new relationships. Output NPZ and adjacent JSON follow organization_data's
original-ID and receiver <- donor conventions.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def convert(source, output, side):
    source, output = Path(source), Path(output)
    if side not in ('before', 'after'):
        raise ValueError('Choose before or after.')
    if output.suffix != '.npz' or output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError('Choose a new .npz output and adjacent JSON path.')
    meta = json.loads(source.read_text())
    if meta['schema'] != 'thermal_saved_edit_organization_response_v1':
        raise ValueError('Unsupported saved edit organization capture.')
    with np.load(source.parent / meta['arrays'], allow_pickle=False) as arrays:
        def array(reference):
            return np.asarray(arrays[reference['array_key']])
        geometry = meta['capture']['geometry'][side]
        get = lambda name: array(geometry[name])[0]
        m_ids, e_ids = get('module_ids'), get('environment_ids')
        m_mask, e_mask = get('module_mask').astype(bool), get('environment_mask').astype(bool)
        if not e_mask.all():
            raise ValueError('The global view requires all original E rows; no synthetic rows are added.')
        nm, ne = len(m_ids), len(e_ids)
        all_ids = np.concatenate((m_ids, e_ids))
        result = {'module_xy': get('module_coordinates'), 'module_ids': m_ids,
                  'module_mask': m_mask, 'environment_xy': get('environment_coordinates'),
                  'environment_ids': e_ids, 'module_physical_measure': get('module_measure'),
                  'environment_physical_measure': get('environment_measure'),
                  'domain_bounds': array(geometry['domain_bounds']),
                  'radius': np.asarray(geometry['radius'])}
        relationship = meta['capture']['relationships'][side]
        access = relationship['prepared_e_access']
        if not np.array_equal(array(access['own_e_member_index'])[0], np.arange(ne)):
            raise ValueError('The E-address rows do not match the original E slots.')
        for mode in ('full', 'restricted'):
            rows = access[mode]
            offsets = array(rows['offsets'])[0]
            slots, ids = array(rows['member_index']), array(rows['original_id'])
            roles, weights = array(rows['is_module']).astype(bool), array(rows['access_weight'])
            if len(offsets) != ne + 1 or offsets[0] != 0 or offsets[-1] != len(slots):
                raise ValueError('Packed original-member access offsets differ from E rows.')
            if ((slots < 0) | (slots >= nm + ne)).any() or not np.array_equal(ids, all_ids[slots]):
                raise ValueError('Packed original-member IDs do not match the saved scene.')
            if not np.array_equal(roles, slots < nm):
                raise ValueError('Packed member roles do not match their original slots.')
            dense = np.zeros((ne, nm + ne), dtype=weights.dtype)
            for row in range(ne):
                start, stop = map(int, offsets[row:row + 2])
                if len(np.unique(slots[start:stop])) != stop - start:
                    raise ValueError('An E row contains duplicate original-member slots.')
                dense[row, slots[start:stop]] = weights[start:stop]
            result['C_' + mode] = dense
            result['support_' + mode] = dense > 0
        routes = []
        for stage in relationship['contextual_stages']:
            identity, route = stage['stage'], stage['route']
            cycle, separator, actual_route = identity.partition('/')
            if not separator or not cycle.startswith('cycle') or actual_route != route:
                raise ValueError('An actual route lacks matching stage provenance.')
            key = f'context_{cycle}_{route}'
            receiver = m_ids if route[0] == 'm' else e_ids
            donor = m_ids if route[1] == 'm' else e_ids
            if (not np.array_equal(array(stage['receiver_ids'])[0], receiver)
                    or not np.array_equal(array(stage['donor_ids'])[0], donor)):
                raise ValueError('An actual stage does not use the original member IDs.')
            result[key] = array(stage['gate'])[0]
            result[key + '_allowed'] = array(stage['allowed'])[0]
            result[key + '_receiver_ids'], result[key + '_donor_ids'] = receiver, donor
            if stage['message_mass'] is not None:
                result[key + '_message_mass'] = array(stage['message_mass'])[0]
            routes.append({'stage': identity, 'receiver_role': stage['receiver_role'],
                           'donor_role': stage['donor_role'],
                           'source_state_version': stage['source_state_version'],
                           'application': stage['application'], 'direction': 'receiver <- donor', 'gate': key})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        np.savez_compressed(stream, **result)
    exported = {'case': meta['family'], 'action': meta['action'], 'scene': side,
                'partition': meta['role'], 'epoch': meta['epoch'], 'checkpoint': meta['checkpoint'],
                'coordinate_units': 'source geometry units', 'projection': 'native 2D xy',
                'sample_selection': meta['query_selection'], 'read_mode': meta['read_mode'],
                'within_role_access_prior': meta['within_role_access_prior'],
                'retained_mass_setting': meta['retained_mass_setting'],
                'directory_provenance': 'Actual prepared E-address rows over original M then E members.',
                'context_stages': routes, 'capture_source': os.path.relpath(source, output.parent),
                'capture_scope': 'Actual applied staged relations and prepared access; no model replay.'}
    with output.with_suffix('.json').open('x', encoding='utf-8') as stream:
        json.dump(exported, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True, type=Path, help='Diagnostic JSON with adjacent NPZ.')
    parser.add_argument('--side', required=True, choices=('before', 'after'))
    parser.add_argument('--output', required=True, type=Path, help='New global-plot NPZ path.')
    args = parser.parse_args()
    convert(args.capture, args.output, args.side)
    print(json.dumps({'scene': args.side, 'output': str(args.output), 'metadata': str(args.output.with_suffix('.json'))}))


if __name__ == '__main__':
    main()

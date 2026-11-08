"""Explicit raw physical endpoint pairs for the R16-8 content experiment.

The input-selected manifest binds each family to its original TRAIN case and
converged raw baseline, and each action to a converged raw edited endpoint.
Nothing in this reader selects cases, solves physics, or fits scales.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .thermal_response_raw import load_raw_endpoint
from .thermal_state_response_task import _native_endpoint_batch


class ThermalR168Corpus:
    """Read an explicit corpus; its roles prevent withheld-action exposure."""

    def __init__(self, manifest_path):
        self.manifest_path = str(manifest_path)
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest.get('schema') != 'thermal_r16_8_corpus_v1':
            raise ValueError('Expected thermal_r16_8_corpus_v1.')
        self.cases, self.records = {}, {}
        for case in manifest['cases']:
            family = str(case['case_id'])
            if family in self.cases or case.get('role') not in ('fit', 'native_seen', 'fresh_layout'):
                raise ValueError('Corpus cases require unique IDs and an explicit role.')
            if not case.get('source_case_dir') or not case.get('source_config'):
                raise ValueError('Corpus case requires its raw baseline and source configuration.')
            self.cases[family] = case
        self._actions = {family: [] for family in self.cases}
        for entry in manifest['actions']:
            family, action = str(entry['case_id']), str(entry['action'])
            key = family, action
            if family not in self.cases or not action or key in self.records:
                raise ValueError('Actions need unique IDs within an existing case.')
            if entry.get('role') != self.cases[family]['role']:
                raise ValueError('Action and family roles differ.')
            if entry.get('status') != 'converged' or not entry.get('raw_case_dir'):
                raise ValueError('An action requires a converged raw endpoint.')
            delta = entry.get('physical_delta', {})
            if set(delta) != {'J', 'P'} or any(not math.isfinite(float(v)) for v in delta.values()):
                raise ValueError('An action requires finite native J/P response labels.')
            self.records[key] = entry
            self._actions[family].append(action)
        if any(len(actions) != 4 for actions in self._actions.values()):
            raise ValueError('Each corpus family requires four declared actions.')

    def family_ids(self, role):
        if role not in ('fit', 'native_seen', 'fresh_layout'):
            raise ValueError('Unknown physical corpus role.')
        return tuple(sorted(k for k, case in self.cases.items() if case['role'] == role))

    def actions_for_family(self, family):
        return tuple(self._actions[str(family)])

    def action_entry(self, family, action, *, role):
        family = str(family)
        if family not in self.cases or self.cases[family]['role'] != role:
            raise ValueError('Physical action is outside its requested role.')
        return self.records[(family, action)]

    def physical_delta(self, family, action, *, role='fit'):
        values = self.action_entry(family, action, role=role)['physical_delta']
        return {name: float(values[name]) for name in ('J', 'P')}

    def pair(self, provider, family, action, *, role='fit'):
        """Build both endpoints from raw physical records, never HDF5 labels."""
        family = str(family)
        case = self.cases[family]
        entry = self.action_entry(family, action, role=role)
        if role == 'fresh_layout':
            raise ValueError('Fresh layouts do not have a TRAIN provider case.')
        raw_b = load_raw_endpoint(case['source_case_dir'])
        raw_e = load_raw_endpoint(entry['raw_case_dir'])
        if raw_b['config'] != case['source_config']:
            raise ValueError('Raw baseline differs from declared source configuration.')
        for section in ('domain', 'thermal', 'local_module'):
            if any(raw_e['config'][section].get(key) != value
                   for key, value in raw_b['config'][section].items()):
                raise ValueError(f'Physical edit changed {section} controls.')
        base_layout, edit_layout = raw_b['config']['layout'], raw_e['config']['layout']
        for key in ('num_modules', 'seed', 'heat_powers'):
            if base_layout[key] != edit_layout[key]:
                raise ValueError(f'Physical edit changed layout {key}.')
        if edit_layout['centers'] != entry['centers']:
            raise ValueError('Physical edit differs from declared source-precision centers.')
        from ..interaction_evidence.reference_adapter import operating_context_from_config
        if operating_context_from_config(raw_b['config']).values != operating_context_from_config(raw_e['config']).values:
            raise ValueError('Physical edit changed operating controls.')
        if not np.array_equal(raw_b['material_mask'], raw_e['material_mask']):
            raise ValueError('Attached material receivers differ between endpoints.')
        original = provider._load_native_cases([family])[0]
        count = int(base_layout['num_modules'])
        if (not np.array_equal(np.asarray(original['structure']['module_centers'][:count]),
                               np.asarray(base_layout['centers'], dtype=np.float32))
                or not np.array_equal(np.asarray(original['structure']['heat_powers'][:count]),
                                      np.asarray(base_layout['heat_powers'], dtype=np.float32))):
            raise ValueError('Raw baseline and TRAIN model input geometry or heat differ.')
        from ..data.datasets import _local_disk_query_points
        if not np.array_equal(np.asarray(original['module_internal_query_points']), _local_disk_query_points(raw_b['material_mask'])):
            raise ValueError('TRAIN and raw attached material receiver coordinates differ.')
        baseline = _native_endpoint_batch(provider, original, raw_b, 'baseline')
        edited = _native_endpoint_batch(provider, original, raw_e, action)
        if not torch.equal(baseline.receivers.heat, edited.receivers.heat):
            raise ValueError('A geometry action changed prescribed heat.')
        return {'family_id': family, 'action': action, 'role': role,
                'baseline': baseline, 'edited': edited,
                'raw_baseline': raw_b, 'raw_edited': raw_e}

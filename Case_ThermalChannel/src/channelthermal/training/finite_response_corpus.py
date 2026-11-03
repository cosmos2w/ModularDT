"""Construct R15-8 fit pairs from converged physical raw endpoints."""
from __future__ import annotations

import copy
import json
import math
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch

from ..data.datasets import _local_disk_query_points
from .content_hypergraph_task import ThermalContentHypergraphTask
from .thermal_finite_corpus import load_raw_endpoint
from .thermal_port_response_task import ThermalPortResponseTargets
from .thermal_native_functionals import native_design_functionals


def _endpoint(provider, original, raw, label):
    case = copy.deepcopy(original)
    count = len(raw['config']['layout']['centers'])
    present = np.asarray(case['structure']['module_present']) > .5
    if count != int(present.sum()) or not np.array_equal(np.flatnonzero(present), np.arange(count)):
        raise ValueError('Raw endpoint must preserve original active module order.')
    case['case_id'] = f"{original['case_id']}:{label}"
    case['structure']['module_centers'][:count] = np.asarray(raw['config']['layout']['centers'], dtype=np.float32)
    case['structure']['heat_powers'][:count] = np.asarray(raw['config']['layout']['heat_powers'], dtype=np.float32)
    case['steady_field'] = raw['field']
    case['module_mask'] = (~raw['fluid_mask']).astype(np.uint8)
    case['module_internal_query_points'] = _local_disk_query_points(raw['material_mask'])
    material = raw['material'][:, raw['material_mask']]
    width = len(present)
    case['module_internal_temperature_points'] = np.zeros((width, material.shape[-1]), dtype=np.float32)
    case['module_internal_temperature_points'][:count] = material
    case['interface_target'] = np.zeros((width, 64, 2), dtype=np.float32)
    case['interface_target'][:count] = raw['interface_target']
    case['interface_condition_valid_mask'] = np.zeros((width, 64), dtype=np.float32)
    # Match the inherited P64 absolute target mask; physical delta masks below
    # are separately defined from the actual surface and outside observations.
    case['interface_condition_valid_mask'][:count] = raw['teacher_h_valid']
    batch = ThermalContentHypergraphTask._native_case_batch(provider, [case], [0], split='train', ntheta=64)
    width = batch.scene_inputs.structure['module_present'].shape[1]
    ports = np.zeros((width, 64, 5), dtype=np.float32)
    valid = np.zeros((width, 64), dtype=bool)
    outside = np.zeros((width, 64), dtype=bool)
    ports[:count], valid[:count], outside[:count] = raw['teacher_ports'], raw['teacher_h_valid'], raw['outside_valid']
    device = provider.device
    targets = ThermalPortResponseTargets(**vars(batch.targets), teacher_ports=torch.as_tensor(ports[None], device=device),
        teacher_h_valid=torch.as_tensor(valid[None], device=device), outside_valid=torch.as_tensor(outside[None], device=device),
        interface_target64=torch.as_tensor(case['interface_target'][None, :width], device=device),
        interface_valid64=torch.as_tensor(case['interface_condition_valid_mask'][None, :width], device=device))
    return replace(batch, targets=targets)


def make_fit_pair(provider, family_id, baseline_path, edited_path, action, scales, *, seed,
                  original=None, raw_baseline=None, raw_edited=None):
    """Create one fit pair, retaining all functional receivers and 512 field points.

    `scales` is a fixed fitting-derived mapping containing eight role names and
    J/P. Caller selects only fitting families and owns deterministic scheduling.
    """
    family_id = str(family_id)
    if family_id not in provider.training_case_ids:
        raise ValueError('Physical pair family must be a selected TRAIN case.')
    original = provider._load_native_cases([family_id])[0] if original is None else original
    raw_b = load_raw_endpoint(baseline_path) if raw_baseline is None else raw_baseline
    raw_e = load_raw_endpoint(edited_path) if raw_edited is None else raw_edited
    if not np.array_equal(raw_b['material_mask'], raw_e['material_mask']):
        raise ValueError('Original attached material receiver catalogue changed.')
    b, e = _endpoint(provider, original, raw_b, 'baseline'), _endpoint(provider, original, raw_e, action)
    common = raw_b['fluid_mask'].reshape(-1) & raw_e['fluid_mask'].reshape(-1)
    chosen = np.flatnonzero(common)
    rng = np.random.default_rng(seed)
    if len(chosen) > 512:
        chosen = rng.choice(chosen, 512, replace=False)
    sampled = np.zeros_like(common); sampled[chosen] = True
    delta = {}
    names = ('flow/u', 'flow/v', 'flow/p', 'flow/omega', 'temperature/fluid')
    for channel, name in enumerate(names):
        target = e.targets.field_targets[..., channel] - b.targets.field_targets[..., channel]
        delta[name] = {'target': target, 'mask': sampled[None], 'scale': scales[name]}
    for name, channel in (('temperature/surface', 0), ('q_proxy', 1)):
        target = e.targets.interface_target[..., channel] - b.targets.interface_target[..., channel]
        mask = torch.isfinite(b.targets.interface_target[..., channel]) & torch.isfinite(e.targets.interface_target[..., channel])
        if channel == 1:
            mask = mask & b.targets.outside_valid & e.targets.outside_valid
        delta[name] = {'target': target, 'mask': mask, 'scale': scales[name]}
    target = e.targets.material_targets - b.targets.material_targets
    present = b.scene_inputs.structure['module_present'].bool()[..., None]
    delta['temperature/material'] = {'target': target, 'mask': present.expand_as(target),
                                    'scale': scales['temperature/material']}
    functions = {}
    outputs = []
    tau = .05 * float(provider.model.internal_temperature_std[0])
    for name, raw, batch in (('baseline', raw_b, b), ('edited', raw_e, e)):
        field = batch.targets.field_targets[..., 2]
        material = batch.targets.material_targets
        matmask = batch.scene_inputs.structure['module_present'].bool()[..., None].expand_as(material)
        fluidmask = torch.as_tensor(raw['fluid_mask'].reshape(1, -1), device=field.device)
        x = batch.receivers.fluid_xy[..., 0]
        length = float(raw['config']['domain']['lx'])
        functions[name] = {'material_mask': matmask, 'fluid_mask': fluidmask, 'fluid_x': x,
                           'domain_length_x': length}
        outputs.append(native_design_functionals(material, matmask, field, fluidmask, x, length, tau))
    for i, name in enumerate(('J', 'P')):
        functions[name] = {'target_delta': float(outputs[1][i] - outputs[0][i]), 'scale': scales[name]}
    return {'role': 'fit', 'family_id': family_id, 'action': action, 'baseline': b, 'edited': e,
            'delta': delta, 'functionals': functions}


class FitPairCorpus:
    """Fixed fitting-only physical inventory with deterministic pair schedule."""
    ACTIONS = ('single_plus', 'single_minus', 'joint_plus', 'joint_minus')
    ROLES = ('flow/u', 'flow/v', 'flow/p', 'flow/omega', 'temperature/fluid',
             'temperature/surface', 'temperature/material', 'q_proxy', 'J', 'P')

    def __init__(self, provider, recipe_path, ledger_path, *, qualification=False, scales=None):
        self.provider, self.scales = provider, scales
        self._cache = OrderedDict()
        self._cache_bytes = 0
        # Leave room below the 512 MiB cap for dictionary and array overhead.
        self._cache_limit = 480 * 1024 * 1024
        recipe = json.loads(Path(recipe_path).read_text())
        self.cases = [c for c in recipe['cases'] if c['role'] == 'fit' and c['feasible']
                      and (not qualification or c['qualification'])]
        if not self.cases or any(c['case_id'] not in provider.training_case_ids for c in self.cases):
            raise ValueError('Fit corpus must contain only selected TRAIN families.')
        rows = [json.loads(line) for line in Path(ledger_path).read_text().splitlines() if line.strip()]
        started = {r['attempt']: r for r in rows if r['event'] == 'started'
                   and r['role'] == 'fit' and r.get('allocation') == 'edited'}
        finished = {r['attempt']: r for r in rows if r['event'] == 'finished'}
        self.paths = {}
        for attempt, start in started.items():
            result = finished.get(attempt)
            if result and result.get('status') == 'converged' and result.get('serialization_status') == 'stored':
                key = (start['case_id'], start['action'])
                if key in self.paths:
                    raise ValueError('Multiple converged physical attempts need explicit disposition.')
                self.paths[key] = result['raw_case_dir']
        missing = [(c['case_id'], action) for c in self.cases for action in self.ACTIONS
                   if (c['case_id'], action) not in self.paths]
        if missing:
            raise ValueError(f'Fitting corpus is incomplete: {len(missing)} physical actions missing.')

    def pair(self, case, action, scales, *, seed):
        baseline_path = case['source_case_dir']
        edited_path = self.paths[(case['case_id'], action)]
        original = self._cached(('original', case['case_id']),
                                lambda: self.provider._load_native_cases([case['case_id']])[0])
        baseline = self._cached(('raw', baseline_path), lambda: load_raw_endpoint(baseline_path))
        edited = self._cached(('raw', edited_path), lambda: load_raw_endpoint(edited_path))
        pair = make_fit_pair(self.provider, case['case_id'], case['source_case_dir'],
            edited_path, action, scales, seed=seed, original=original, raw_baseline=baseline, raw_edited=edited)
        for name, expected in (('baseline', case['centers']), ('edited', case['selection']['actions'][action])):
            structure = pair[name].scene_inputs.structure
            actual = structure['module_centers'][0].detach().cpu().numpy()
            if not np.array_equal(actual, np.asarray(expected, dtype=np.float32)):
                raise ValueError('Raw endpoint geometry does not match the input-only action recipe.')
        b = pair['baseline'].receivers.heat
        e = pair['edited'].receivers.heat
        if not torch.equal(b, e):
            raise ValueError('Physical geometry edit changed prescribed heat.')
        return pair

    def _cached(self, key, load):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key][0]
        value = load()
        def size(item):
            if isinstance(item, np.ndarray):
                return item.nbytes + 256
            if isinstance(item, dict):
                return 512 + sum(size(v) for v in item.values())
            if isinstance(item, (tuple, list)):
                return 128 + sum(size(v) for v in item)
            return 128
        amount = size(value)
        if amount <= self._cache_limit:
            while self._cache and self._cache_bytes + amount > self._cache_limit:
                _, (_, removed) = self._cache.popitem(last=False)
                self._cache_bytes -= removed
            self._cache[key] = (value, amount)
            self._cache_bytes += amount
        return value

    def __call__(self, key):
        if self.scales is None:
            raise ValueError('Training requires fixed fitting-derived response scales.')
        index = (int(key.epoch) - 1) * math.ceil(len(self.provider.training_cases) / self.provider.effective_batch_size) + int(key.update_index)
        case = self.cases[(index // len(self.ACTIONS)) % len(self.cases)]
        action = self.ACTIONS[index % len(self.ACTIONS)]
        kind = action.split('_')[0]
        return self.pair(case, action, self.scales[kind], seed=15808 + index)


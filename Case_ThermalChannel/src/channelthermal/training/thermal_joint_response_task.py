"""Fresh Thermal native-plus-finite-response training, independent of R16 corpora.

Physical family manifests describe only selected TRAIN inputs and their solved
endpoint observations. This module neither selects withheld data nor solves it.
"""
from __future__ import annotations

import copy
import json
import math
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from honf_runtime.state_response import ResponseObjectiveConfig
from honf_runtime.unified_training import LossTerm

from ..physical_provenance import validate_thermal_endpoint_config
from .r13_6_member_routing_task import _build_thermal_member_routing_task
from .r13_8_anchored_task import _VARIANTS, ARM_DETAIL, DETAIL_CONFIG_SCHEMA, DETAIL_CORE_SCHEMA, DETAIL_SOURCE_KIND
from .thermal_family_objective import ThermalFamilyObjectiveMixin
from .thermal_native_port_task import ThermalNativePortTask
from .thermal_response_raw import load_raw_endpoint
from .thermal_state_response_task import FIELD_ROLES, _native_endpoint_batch, _reference_delta_squares
from .r16_10_corpus import EligibleFamilySchedule

SCHEMA = 'thermal_joint_response_v1'
FAMILY_SCHEMA = 'thermal_response_families_v1'
FUNCTIONALS = ('J', 'P')
LEGACY_SELECTORS = ('r16_10_posttrain', 'r16_10_family', 'r16_8_content', 'r16_7_prescribed',
                    'r16_6_balanced', 'r16_5_warm_fork', 'r16_4_warm_fork',
                    'thermal_state_response', 'thermal_port_response')


def _check_legacy_selectors(recipe):
    conflicts = [name for name in LEGACY_SELECTORS if recipe.get(name) is not None]
    if conflicts:
        raise ValueError(f'The new state_response profile cannot combine legacy selectors: {conflicts}')


def build_thermal_joint_response_config(base_recipe, *, fit_manifest=None, scale_floors=None,
        response_scales=None, calibration_path=None, families_per_epoch=4, family_seed=0,
        angle_count=64, outside_temperature_weight=.05, tau=.4149511814117432,
        enabled=True, endpoint_weight=1., response_weight=1., field_response_weight=.8,
        functional_weights=None, warmup_epochs=10, ramp_epochs=40,
        coefficient_mode='gradient_norm', gradient_norm_ratios=None):
    """Return a portable recipe; paths and all physical scale choices are explicit.

    Exactly one of ``scale_floors``, ``response_scales`` and ``calibration_path``
    is required for an enabled objective. Floors fit scales from FIT labels;
    explicit values use supplied scales; a calibration file contains both
    ``response_scales``, ``coefficients``, ``native_case_ids`` and ``fit_family_ids``
    from the selected TRAIN cohort. Disabling the objective needs none.
    """
    recipe = copy.deepcopy(dict(base_recipe))
    _check_legacy_selectors(recipe)
    objective = recipe.get('objective')
    if isinstance(objective, dict):
        objective.pop('response_fit', None)
        objective.pop('response_training_families', None)
    recipe['training_profile'] = 'state_response'
    weights = {'J': .1, 'P': .1} if functional_weights is None else dict(functional_weights)
    response = {'enabled': enabled, 'endpoint_weight': endpoint_weight, 'response_weight': response_weight,
                'field_response_weight': field_response_weight, 'functional_weights': weights,
                'warmup_epochs': warmup_epochs, 'ramp_epochs': ramp_epochs,
                'coefficient_mode': coefficient_mode}
    if gradient_norm_ratios is not None:
        response['gradient_norm_ratios'] = dict(gradient_norm_ratios)
    ResponseObjectiveConfig.from_mapping(response)
    selections = [(name, value) for name, value in (('fit', scale_floors),
                  ('explicit', response_scales), ('calibration', calibration_path)) if value is not None]
    if enabled and (not fit_manifest or len(selections) != 1):
        raise ValueError('Enabled state/response needs a FIT manifest and exactly one scale source.')
    if len(selections) > 1:
        raise ValueError('Choose only one response scale source.')
    scales = None
    if selections:
        mode, value = selections[0]
        scales = {'mode': mode, ('floors' if mode == 'fit' else 'values' if mode == 'explicit' else 'path'):
                  (copy.deepcopy(dict(value)) if mode != 'calibration' else str(value))}
    recipe['state_response'] = response
    recipe['thermal_joint_response'] = {'schema': SCHEMA, 'fit_manifest': None if fit_manifest is None else str(fit_manifest),
        'scales': scales, 'families_per_epoch': families_per_epoch, 'family_seed': family_seed,
        'native_ports': {'angle_count': angle_count, 'outside_temperature_weight': outside_temperature_weight},
        'tau': tau}
    return recipe


def _response_config(recipe):
    supplied = recipe.get('state_response', {})
    if not isinstance(supplied, dict):
        raise TypeError('state_response must be a mapping.')
    values = {'field_response_weight': .8, 'functional_weights': {'J': .1, 'P': .1}, **supplied}
    config = ResponseObjectiveConfig.from_mapping(values)
    if set(config.functional_weights) != set(FUNCTIONALS):
        raise ValueError('Thermal response requires explicit J and P functional weights.')
    return config


def _scales(values):
    if not isinstance(values, dict) or set(values) != {*FIELD_ROLES, *FUNCTIONALS}:
        raise ValueError('Response scales require all native field roles plus J and P.')
    result = {}
    for name, value in values.items():
        if isinstance(value, dict):
            value = value.get('scale')
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f'Invalid response scale for {name}.') from None
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f'Response scale for {name} must be finite and positive.')
        result[name] = number
    return result


def _path(path, root):
    if not isinstance(path, str) or not path:
        raise ValueError('Every physical endpoint needs an explicit path.')
    value = Path(path).expanduser()
    return value if value.is_absolute() else root / value


class ThermalFitFamilyCorpus:
    """Portable input-selected FIT families with explicit physical J/P labels."""

    def __init__(self, manifest_path, training_ids):
        path = Path(manifest_path).expanduser()
        manifest = json.loads(path.read_text())
        if manifest.get('schema') != FAMILY_SCHEMA:
            raise ValueError(f'Expected {FAMILY_SCHEMA}.')
        self.path, self.root = str(path), path.parent
        self.cases, self.records, self._actions = {}, {}, {}
        self._family_cache = None
        training = set(map(str, training_ids))
        for source in manifest.get('cases', ()):
            case = copy.deepcopy(source)
            family = str(case.get('case_id', ''))
            if not family or family in self.cases or family not in training or case.get('role', 'fit') != 'fit':
                raise ValueError('FIT families must be unique members of the native TRAIN selection.')
            if not isinstance(case.get('source_config'), dict):
                raise TypeError('Every FIT family needs its complete prescribed source config.')
            case['source_case_dir'] = str(_path(case.get('source_case_dir'), self.root))
            self.cases[family] = case
            self._actions[family] = []
        if not self.cases:
            raise ValueError('The FIT manifest must contain at least one TRAIN family.')
        for source in manifest.get('actions', ()):
            row = copy.deepcopy(source)
            family, action = str(row.get('case_id', '')), str(row.get('action', ''))
            if family not in self.cases or not action or (family, action) in self.records or row.get('role', 'fit') != 'fit':
                raise ValueError('FIT actions must have unique IDs within a declared TRAIN family.')
            labels = row.get('physical_delta')
            if not isinstance(labels, dict) or set(labels) != set(FUNCTIONALS):
                raise ValueError('Every enabled FIT action requires physical J/P response labels.')
            try:
                labels = {name: float(labels[name]) for name in FUNCTIONALS}
            except (TypeError, ValueError):
                raise ValueError('Every FIT action needs finite physical J/P response labels.') from None
            if not all(math.isfinite(value) for value in labels.values()):
                raise ValueError('Every FIT action needs finite physical J/P response labels.')
            if not isinstance(row.get('centers'), list):
                raise TypeError('Every physical edit needs explicit ordered module centers.')
            row['physical_delta'] = labels
            row['raw_case_dir'] = str(_path(row.get('raw_case_dir'), self.root))
            self.records[(family, action)] = row
            self._actions[family].append(action)
        if any(not actions for actions in self._actions.values()):
            raise ValueError('Every FIT family needs at least one labeled physical action.')

    def family_ids(self):
        return tuple(self.cases)

    def actions(self, family):
        return tuple(self._actions[family])

    def physical_delta(self, family, action):
        return dict(self.records[(family, action)]['physical_delta'])

    @staticmethod
    def _verify_saved(record, source_config, actual_config, centers):
        validate_thermal_endpoint_config(source_config, actual_config, centers=centers,
            expected_actual_config=record.get('actual_solver_config'),
            numerical_policy=record.get('numerical_policy'))

    def verify_training_controls(self, provider):
        """Bind each supplied source config to the original HDF5 TRAIN case."""
        import h5py

        from ..data.datasets import _decode_scalar_string
        with h5py.File(provider.data_path, 'r') as h5:
            for family, case in self.cases.items():
                group = h5['cases'][family]
                if 'case_config_json' not in group:
                    raise ValueError(f'TRAIN family {family} lacks prescribed source provenance.')
                native_config = json.loads(_decode_scalar_string(group['case_config_json'][()]))
                validate_thermal_endpoint_config(native_config, case['source_config'],
                    centers=native_config['layout']['centers'])

    @contextmanager
    def family_context(self, family):
        if self._family_cache is not None or family not in self.cases:
            raise ValueError('Family input cache requires one declared, non-nested family.')
        self._family_cache = {'family': family}
        try:
            yield
        finally:
            self._family_cache = None

    def pair(self, provider, family, action):
        from ..data.datasets import _local_disk_query_points
        case, row = self.cases[family], self.records[(family, action)]
        cache = self._family_cache if self._family_cache is not None and self._family_cache['family'] == family else None
        baseline = cache.get('baseline') if cache is not None else None
        if baseline is None:
            baseline = load_raw_endpoint(case['source_case_dir'])
            if cache is not None:
                cache['baseline'] = baseline
        edited = load_raw_endpoint(row['raw_case_dir'])
        config = case['source_config']
        self._verify_saved(case, config, baseline['config'], config['layout']['centers'])
        self._verify_saved(row, config, edited['config'], row['centers'])
        if (not np.array_equal(baseline['material_mask'], edited['material_mask'])
                or baseline['field'].shape != edited['field'].shape):
            raise ValueError('Physical endpoints do not share the ordered native receiver catalogue.')
        original = cache.get('original') if cache is not None else None
        if original is None:
            original = provider._load_native_cases([family])[0]
            if cache is not None:
                cache['original'] = original
        count = int(config['layout']['num_modules'])
        if (not np.array_equal(np.asarray(original['structure']['module_centers'][:count]),
                               np.asarray(config['layout']['centers'], dtype=np.float32))
                or not np.array_equal(np.asarray(original['structure']['heat_powers'][:count]),
                                      np.asarray(config['layout']['heat_powers'], dtype=np.float32))
                or not np.array_equal(np.asarray(original['module_internal_query_points']),
                                      _local_disk_query_points(baseline['material_mask']))):
            raise ValueError('FIT baseline differs from the native TRAIN member or receiver identity.')
        base_batch = cache.get('base_batch') if cache is not None else None
        if base_batch is None:
            base_batch = _native_endpoint_batch(provider, original, baseline, 'baseline')
            if cache is not None:
                cache['base_batch'] = base_batch
        edit_batch = _native_endpoint_batch(provider, original, edited, action)
        if not torch.equal(base_batch.receivers.heat, edit_batch.receivers.heat):
            raise ValueError('A response edit changed the prescribed heat controls.')
        return {'family_id': family, 'action': action, 'role': 'fit',
                'baseline': base_batch, 'edited': edit_batch,
                'raw_baseline': baseline, 'raw_edited': edited}


class ThermalFitFamilyObjective(ThermalFamilyObjectiveMixin):
    def __init__(self, provider, corpus, scales, config, tau):
        self.provider, self.corpus, self.response_scales = provider, corpus, scales
        self.response_objective_config, self.tau = config, float(tau)
        self.sensitivity = self.research_scales = None
        if not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError('Thermal functional smooth-maximum tau must be finite and positive.')

    def actions(self, source, family):
        if source != 'fit':
            raise ValueError('Only declared FIT families may support training.')
        return self.corpus.actions(family)

    def pair(self, source, family, action):
        self.actions(source, family)
        return self.corpus.pair(self.provider, family, action)

    def label(self, source, family, action):
        self.actions(source, family)
        return self.corpus.physical_delta(family, action)


class ThermalPairedFitFamilyObjective(ThermalFitFamilyObjective):
    """Portable manifest with explicit, separately scheduled FIT sources."""

    def actions(self, source, family):
        if family not in self.schedule.families.get(source, ()):
            raise ValueError('Family is outside its declared eligible FIT source.')
        actions = self.corpus.actions(family)
        if len(actions) != 4:
            raise ValueError('Paired FIT families require exactly four actions.')
        return actions

    def pair(self, source, family, action):
        self.actions(source, family)
        return self.corpus.pair(self.provider, family, action)

    def label(self, source, family, action):
        self.actions(source, family)
        return self.corpus.physical_delta(family, action)

    def family_components(self, model, source, family, **kwargs):
        with self.corpus.family_context(family):
            return super().family_components(model, source, family, **kwargs)


def _fit_response_scales(provider, corpus, floors):
    floors = _scales(floors)
    all_family = {name: [] for name in floors}
    for family in corpus.family_ids():
        per_action = {name: [] for name in floors}
        for action in corpus.actions(family):
            pair = corpus.pair(provider, family, action)
            for name, (delta, mask) in _reference_delta_squares(pair).items():
                if name in ('temperature/surface', 'q_proxy'):
                    stride = 64 // provider.port_angle_count
                    delta, mask = delta[..., ::stride], mask[..., ::stride]
                if not bool(mask.any()) or not bool(torch.isfinite(delta[mask]).all()):
                    raise ValueError(f'No valid FIT response receivers for {family}/{action}/{name}.')
                per_action[name].append(float(delta[mask].double().square().mean()))
            labels = corpus.physical_delta(family, action)
            for name in FUNCTIONALS:
                per_action[name].append(labels[name] ** 2)
        for name in floors:
            all_family[name].append(sum(per_action[name]) / len(per_action[name]))
    return {name: max(math.sqrt(sum(values) / len(values)), floors[name])
            for name, values in all_family.items()}


def _norm(parameters, loss=None):
    if loss is None:
        gradients = (p.grad for p in parameters)
    else:
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients if g is not None))


def calibrate_paired_response(model, provider, *, epoch, native_ids, fit_families):
    """Measure grid coefficients and catalogue gradients on fixed TRAIN panels.

    The caller chooses the declared input-only panels and persists the receipt.
    No optimizer update, parameter, gradient, mode or random state is retained.
    """
    from .r16_10_diagnostics import gradient_diagnostics, preserved_state
    objective = provider.family_objective
    native, panel = tuple(map(str, native_ids)), tuple(tuple(row) for row in fit_families)
    if (epoch != 10 or objective.auxiliary_fluid_measure != 'grid' or len(native) != 15
            or len(set(native)) != 15 or not set(native).issubset(provider.training_case_ids)
            or len(panel) != 4 or len(set(panel)) != 4
            or any(f not in objective.schedule.families.get(s, ()) for s, f in panel)):
        raise ValueError('Paired calibration requires epoch-10 grid and valid TRAIN15/FIT4 panels.')
    def native_loss():
        for batch in provider.native_evaluation_batches(split='train', case_keys=native, microbatch_size=1):
            prediction, _ = provider.predict_native(model, batch.scene_inputs, batch.receivers,
                                                     model.architecture, 'joint', epoch=epoch)
            yield provider.endpoint_state_loss(prediction, batch) / len(native)
    def family_loss(component, measure):
        def pieces():
            objective.auxiliary_fluid_measure = measure
            for source, family in panel:
                values, _ = objective.family_components(model, source, family, components=('endpoint', 'response'))
                yield values[component] / len(panel)
        return pieces
    losses = {'native': native_loss, **{f'{measure}_{name}': family_loss(name, measure)
              for measure in ('grid', 'catalogue') for name in ('endpoint', 'response')}}
    try:
        with preserved_state(model):
            model.train()
            result = gradient_diagnostics(model, losses, require_nonzero=True)
    finally:
        objective.auxiliary_fluid_measure = 'grid'
    norms = result['groups']['global']['norms']
    return {'epoch': epoch, 'measure': 'grid', 'native_case_ids': list(native),
            'fit_families': [list(row) for row in panel], 'response_scales': dict(objective.response_scales),
            'gradient_diagnostics': result,
            'coefficients': {'endpoint': .25 * norms['native'] / norms['grid_endpoint'],
                             'response': .10 * norms['native'] / norms['grid_response']}}


def _attach_response(provider, objective, config, *, count, seed, restored_calibration=None,
                     external_calibration=None):
    families = objective.corpus.family_ids()
    if type(count) is not int or not 1 <= count <= len(families) or type(seed) is not int:
        raise ValueError('FIT family count must fit the TRAIN family set and seed must be an integer.')
    order = tuple(families[i] for i in np.random.default_rng(seed).permutation(len(families)))
    calibration = copy.deepcopy(restored_calibration)
    if external_calibration is not None:
        supplied = external_calibration.get('coefficients') if isinstance(external_calibration, dict) else None
        if not isinstance(supplied, dict) or set(supplied) != {'endpoint', 'response'}:
            raise ValueError('Calibration file needs endpoint and response coefficients.')
        weights = {'endpoint': config.endpoint_weight, 'response': config.response_weight}
        coefficients = {key: float(supplied[key]) * weights[key] for key in weights}
        if any(not math.isfinite(value) or value < 0 for value in coefficients.values()):
            raise ValueError('Calibration coefficients must be finite and nonnegative.')
        supplied_calibration = {'source': 'explicit calibration file', 'coefficients': coefficients,
            'native_case_ids': list(external_calibration['native_case_ids']),
            'fit_family_ids': list(external_calibration['fit_family_ids'])}
        if calibration is not None and calibration != supplied_calibration:
            raise ValueError('Saved calibration differs from the current TRAIN calibration file.')
        calibration = supplied_calibration
    if calibration is None and config.coefficient_mode == 'fixed':
        calibration = {'source': 'explicit fixed weights', 'coefficients': config.coefficients()}
    state = {'calibration': calibration}
    native_snapshot = {'norm': None, 'case_ids': ()}
    active = tuple(name for name, weight in (('endpoint', config.endpoint_weight), ('response', config.response_weight))
                   if (float(calibration['coefficients'][name]) > 0 if calibration is not None
                       else weight > 0 and config.gradient_norm_ratios[name] > 0))
    if not active:
        raise ValueError('Enabled state/response needs a positive endpoint or response weight.')
    provider.auxiliary_loss_names = tuple('family/' + name for name in active)
    provider.family_objective = objective
    def state_response_metadata(self):
        return {'calibration': copy.deepcopy(state['calibration'])}
    provider.state_response_metadata = MethodType(state_response_metadata, provider)
    def needs_native_gradient_norm(self, *, epoch, update_index):
        return state['calibration'] is None and config.ramp(epoch) > 0 and update_index == 0
    def record_native_gradient_norm(self, value, *, case_keys):
        ids = tuple(map(str, case_keys))
        if not ids or len(set(ids)) != len(ids) or not set(ids).issubset(self.training_case_ids):
            raise ValueError('Calibration native cases must be distinct selected TRAIN members.')
        native_snapshot.update(norm=float(value), case_ids=ids)
    provider.needs_native_gradient_norm = MethodType(needs_native_gradient_norm, provider)
    provider.record_native_gradient_norm = MethodType(record_native_gradient_norm, provider)
    previous_identity = provider.identity_payload
    def identity_payload(self):
        result = dict(previous_identity())
        result['native_only_training'] = False
        result['response_training_family_ids'] = list(families)
        result['response_supervision'] = 'explicit selected TRAIN FIT families'
        result['objective'].update(response_training_families=list(families), response_fit=True)
        result['thermal_joint_response'] = {'schema': SCHEMA, 'fit_manifest': objective.corpus.path,
            'families': list(families), 'family_order': list(order), 'families_per_epoch': count,
            'response_scales': dict(objective.response_scales), 'tau': objective.tau}
        return result
    provider.identity_payload = MethodType(identity_payload, provider)
    previous_metadata = provider.loss_metadata
    def loss_metadata(self):
        result = dict(previous_metadata())
        for name in active:
            result['family/' + name] = {'panel': 'physics', 'label': 'FIT family ' + name,
                'unit': 'scaled squared error', 'formula': 'equal FIT family and action mean',
                'weight': 'configured ramp and calibrated coefficient'}
        return result
    provider.loss_metadata = MethodType(loss_metadata, provider)
    def auxiliary_update_terms(self, model, *, epoch, update_index, update_count):
        ramp = config.ramp(epoch)
        if not ramp:
            return
        offset = (epoch - config.warmup_epochs - 1) * count
        selected = tuple(order[(offset + index) % len(order)] for index in range(count))
        slots = np.linspace(0, update_count - 1, count, dtype=int)
        for family, slot in zip(selected, slots):
            if int(slot) != update_index:
                continue
            components, work = objective.family_components(model, 'fit', family, components=active)
            if state['calibration'] is None:
                params = tuple(p for p in model.parameters() if p.requires_grad)
                native_norm = native_snapshot['norm']
                if native_norm is None:
                    raise ValueError('TRAIN-only native gradient snapshot is missing for response calibration.')
                component_norms = {name: (_norm(params, components[name]) if name in active else 0.)
                                   for name in ('endpoint', 'response')}
                coefficients = config.coefficients(native_norm, component_norms)
                state['calibration'] = {'source': 'first selected TRAIN native update and FIT family',
                    'epoch': epoch, 'family': family, 'native_case_ids': list(native_snapshot['case_ids']),
                    'native_gradient_norm': native_norm,
                    'component_gradient_norms': component_norms, 'coefficients': coefficients}
            coeffs = state['calibration']['coefficients']
            terms = {'family/' + name: LossTerm(components[name], 1., ramp * float(coeffs[name])
                                               * update_count / len(selected)) for name in active}
            work.update(family_presentations=1, edited_presentations=len(objective.actions('fit', family)))
            yield terms, work
    provider.auxiliary_update_terms = MethodType(auxiliary_update_terms, provider)
    old_state = getattr(provider, 'training_state_dict', dict)
    old_load = getattr(provider, 'load_training_state_dict', lambda saved: None)
    def training_state_dict(self):
        return {'native': old_state(), 'thermal_joint_response': copy.deepcopy(state)}
    def load_training_state_dict(self, saved):
        old_load(saved['native'])
        restored = saved['thermal_joint_response']['calibration']
        if restored is not None:
            coefficients = restored.get('coefficients', {})
            if set(coefficients) != {'endpoint', 'response'} or any(
                    not math.isfinite(float(value)) or float(value) < 0 for value in coefficients.values()):
                raise ValueError('Saved TRAIN response coefficients are invalid.')
        state['calibration'] = copy.deepcopy(restored)
    provider.training_state_dict = MethodType(training_state_dict, provider)
    provider.load_training_state_dict = MethodType(load_training_state_dict, provider)
    return provider


def _attach_paired_response(provider, objective, config, spec, restored_state=None):
    """Keep calibration pending until the common measured grid receipt exists."""
    sources = {name: [family for family, row in objective.corpus.cases.items()
                      if row.get('source') == name] for name in ('original', 'new')}
    if tuple(map(len, sources.values())) != (15, 25) or sum(map(len, sources.values())) != len(objective.corpus.cases):
        raise ValueError('TRAIN150 requires exactly 15 original and 25 new FIT families.')
    objective.schedule = EligibleFamilySchedule(sources['original'], sources['new'], provider.training_case_ids, seed=0)
    if any(len(objective.actions(source, family)) != 4 for source in sources for family in sources[source]):
        raise ValueError('Every family requires four edits.')
    counts = {'original': 1, 'new': 2}
    calibration_path = Path(spec['calibration_path']).expanduser()
    candidate = spec['qualified_tile']
    if type(candidate) is not int or candidate not in (128, 256, 512):
        raise ValueError('Qualified tile must be 128, 256 or 512.')
    decision_path = Path(spec['executor_decision_path']).expanduser()
    state = {'calibration': None, 'executor': {'active_tile': None, 'decision': None}}
    provider.family_objective = objective
    provider.training_history_in_logs = True
    provider.auxiliary_loss_names = ('family/endpoint', 'family/response')
    previous_metadata = provider.loss_metadata
    def loss_metadata(self):
        result = dict(previous_metadata())
        for name in ('endpoint', 'response'):
            result['family/' + name] = {'panel': 'physics', 'label': 'FIT family ' + name,
                'unit': 'scaled squared error', 'formula': 'equal FIT family and action mean',
                'weight': 'common grid calibration and epoch ramp'}
        return result
    provider.loss_metadata = MethodType(loss_metadata, provider)
    def load_calibration():
        receipt = json.loads(calibration_path.read_text())
        native = tuple(map(str, receipt.get('native_case_ids', ())))
        fit = tuple((str(row[0]), str(row[1])) for row in receipt.get('fit_families', ()))
        def panel(values):
            values = tuple(values)
            selected = set(np.random.default_rng(20261108).choice(len(values),
                           size=max(1, int(np.floor(.1 * len(values) + .5))), replace=False).tolist())
            return tuple(value for index, value in enumerate(values) if index in selected)
        population = tuple((source, family) for source in ('original', 'new')
                           for family in objective.schedule.families[source])
        if (len(native) != 15 or len(set(native)) != 15 or not set(native).issubset(provider.training_case_ids)
                or len(fit) != 4 or len(set(fit)) != 4
                or any(f not in objective.schedule.families.get(s, ()) for s, f in fit)
                or native != panel(provider.training_case_ids) or fit != panel(population)
                or receipt.get('measure') != 'grid' or receipt.get('epoch') != 10
                or _scales(receipt.get('response_scales')) != objective.response_scales):
            raise ValueError('Common calibration must identify the TRAIN15/FIT4 grid panel and matching scales.')
        coeffs = receipt.get('coefficients', {})
        if set(coeffs) != {'endpoint', 'response'} or any(
                not math.isfinite(float(v)) or float(v) <= 0 for v in coeffs.values()):
            raise ValueError('Common calibration requires positive measured endpoint and response coefficients.')
        return copy.deepcopy(receipt)
    def auxiliary_update_terms(self, model, *, epoch, update_index, update_count):
        ramp = config.ramp(epoch)
        if not ramp:
            return
        if state['calibration'] is None:
            state['calibration'] = load_calibration()
        selected = objective.schedule.epoch(epoch, counts)
        slots = np.linspace(0, update_count - 1, len(selected), dtype=int)
        for (source, family), slot in zip(selected, slots):
            if int(slot) != update_index:
                continue
            components, work = objective.family_components(model, source, family, components=('endpoint', 'response'))
            terms = {'family/' + name: LossTerm(components[name], 1., ramp * float(state['calibration']['coefficients'][name])
                     * update_count / len(selected)) for name in ('endpoint', 'response')}
            work.update(family_presentations=1, edited_presentations=4)
            yield terms, work
    provider.auxiliary_update_terms = MethodType(auxiliary_update_terms, provider)
    provider.state_response_metadata = MethodType(lambda self: {'calibration': copy.deepcopy(state['calibration'])}, provider)
    provider.response_epoch_family_ids = MethodType(
        lambda self, epoch: [f'{source}:{family}' for source, family in objective.schedule.current]
        if epoch > 10 else [], provider)
    def begin_training_epoch(self, model, epoch):
        if epoch <= 50:
            tile = candidate
        elif epoch <= 70:
            tile = 128 if epoch <= 55 or epoch >= 66 else candidate
        else:
            decision = json.loads(decision_path.read_text())
            if decision.get('qualified_tile') != candidate or decision.get('selected_tile') not in (128, candidate):
                raise ValueError('Executor decision does not match the qualified ABBA schedule.')
            if state['executor']['decision'] is not None and state['executor']['decision'] != decision:
                raise ValueError('Executor decision differs from the saved paired state.')
            state['executor']['decision'] = decision
            tile = decision['selected_tile']
        core = getattr(model, 'core', model)
        if not hasattr(core, 'query_tile_size'):
            raise ValueError('Paired executor requires the anchored query tile.')
        core.query_tile_size = tile
        state['executor']['active_tile'] = tile
        return {'active_tile': tile, 'qualified_tile': candidate,
                'decision': copy.deepcopy(state['executor']['decision'])}
    provider.begin_training_epoch = MethodType(begin_training_epoch, provider)
    old_state = getattr(provider, 'training_state_dict', lambda: {})
    old_load = getattr(provider, 'load_training_state_dict', lambda saved: None)
    def training_state_dict(self):
        return {'native': old_state(), 'thermal_joint_response': {'calibration': copy.deepcopy(state['calibration']),
                'schedule': objective.schedule.state_dict(), 'executor': copy.deepcopy(state['executor'])}}
    def load_training_state_dict(self, saved):
        old_load(saved['native'])
        paired = saved['thermal_joint_response']
        objective.schedule.load_state_dict(paired['schedule'])
        state['calibration'] = copy.deepcopy(paired['calibration'])
        state['executor'] = copy.deepcopy(paired['executor'])
        if state['calibration'] is not None and state['calibration'] != load_calibration():
            raise ValueError('Common calibration differs from the saved paired state.')
    provider.training_state_dict = MethodType(training_state_dict, provider)
    provider.load_training_state_dict = MethodType(load_training_state_dict, provider)
    previous_identity = provider.identity_payload
    def identity_payload(self):
        result = dict(previous_identity())
        result['native_only_training'] = False
        result['response_training_family_ids'] = list(objective.corpus.family_ids())
        result['response_supervision'] = 'explicit selected TRAIN FIT families'
        result['objective'].update(response_training_families=list(objective.corpus.family_ids()), response_fit=True)
        result['thermal_joint_response'] = {'schema': SCHEMA, 'fit_manifest': objective.corpus.path,
            'families': {s: list(ids) for s, ids in objective.schedule.families.items()},
            'counts': counts, 'response_scales': dict(objective.response_scales),
            'measure': objective.auxiliary_fluid_measure, 'calibration_path': str(calibration_path), 'tau': objective.tau,
            'qualified_tile': candidate, 'executor_decision_path': str(decision_path)}
        return result
    provider.identity_payload = MethodType(identity_payload, provider)
    return provider


def build_thermal_joint_response_task(recipe, *, arm=None, device='cpu', resume_payload=None):
    """Build the clean native task and attach explicitly labeled FIT response when enabled."""
    _check_legacy_selectors(recipe)
    config = _response_config(recipe)
    spec = recipe.get('thermal_joint_response', {})
    if not isinstance(spec, dict) or spec.get('schema') != SCHEMA:
        raise ValueError(f'Expected {SCHEMA} case settings.')
    if config.enabled and (not spec.get('fit_manifest') or not isinstance(spec.get('scales'), dict)):
        raise ValueError('Enabled state/response requires a FIT family manifest, physical labels and scales.')
    arm = arm or recipe['member_routing'].get('arm', ARM_DETAIL)
    identity = (resume_payload or {}).get('experiment_identity', {}).get('provider_identity', {})
    model, provider = _build_thermal_member_routing_task(recipe, arm=arm, device=device,
        task_type=ThermalNativePortTask,
        variants={DETAIL_CORE_SCHEMA: {**_VARIANTS[DETAIL_CORE_SCHEMA], 'arm': arm}},
        section_schema=DETAIL_CONFIG_SCHEMA, source_kind=DETAIL_SOURCE_KIND,
        lineage='Thermal native state/response', required_address_mode=None,
        saved_adapter_state=identity.get('adapter_state'),
        saved_normalization_source=identity.get('source_binding', {}).get('normalization_source'))
    provider.configure_native()
    previous_identity = provider.identity_payload
    def identity_with_response_config(self):
        result = {**previous_identity(), 'state_response': asdict(config)}
        if recipe.get('training', {}).get('membership') == 'catalog_train':
            split = recipe.get('evaluation', {}).get('validation_split', 'validation').lower()
            result['validation_source_split'] = split
            result['test_read'] = split == 'test'
        return result
    provider.identity_payload = MethodType(identity_with_response_config, provider)
    provider.state_response_metadata = MethodType(lambda self: {'calibration': None}, provider)
    provider.validation_scope = ('explicit_input_selected_validation' if recipe.get('evaluation', {}).get('validation_case_ids') is not None
                                 else 'input_only_sample_of_declared_validation')
    if not config.enabled:
        return model, provider
    corpus = ThermalFitFamilyCorpus(spec['fit_manifest'], provider.training_case_ids)
    corpus.verify_training_controls(provider)
    scale_spec = spec['scales']
    mode = scale_spec.get('mode')
    calibration_file = None
    if mode == 'calibration':
        calibration_file = json.loads(Path(scale_spec['path']).expanduser().read_text())
        native_ids = tuple(map(str, calibration_file.get('native_case_ids', ())))
        fit_ids = tuple(map(str, calibration_file.get('fit_family_ids', ())))
        if (not native_ids or len(set(native_ids)) != len(native_ids)
                or not set(native_ids).issubset(provider.training_case_ids)
                or not fit_ids or len(set(fit_ids)) != len(fit_ids)
                or not set(fit_ids).issubset(corpus.family_ids())):
            raise ValueError('Calibration case and family IDs must be explicit selected TRAIN/FIT members.')
        scales = _scales(calibration_file.get('response_scales'))
        external_coefficients = calibration_file
        if calibration_file.get('coefficients') is None:
            raise ValueError('Calibration file must contain actual TRAIN-only coefficients.')
    elif mode == 'explicit':
        scales, external_coefficients = _scales(scale_spec.get('values')), None
    elif mode == 'fit':
        saved_scales = identity.get('thermal_joint_response', {}).get('response_scales')
        floors = scale_spec.get('floors')
        if spec.get('paired') is not None and scale_spec.get('repeatability') is not None:
            repeatability = scale_spec['repeatability']
            functional = scale_spec.get('functional_floors', {})
            if set(repeatability) != set(FIELD_ROLES) or set(functional) != set(FUNCTIONALS):
                raise ValueError('Paired scale fitting needs every field repeatability and both functional floors.')
            state_scales = {name: float(model.field_std[i]) for i, name in enumerate(FIELD_ROLES[:5])}
            state_scales.update({'temperature/surface': float(model.interface_target_std[0]),
                                 'temperature/material': float(model.internal_temperature_std[0]),
                                 'q_proxy': float(model.interface_target_std[1])})
            floors = {name: max(.01 * state_scales[name], float(repeatability[name])) for name in FIELD_ROLES}
            floors.update(functional)
        scales = (_scales(saved_scales) if saved_scales is not None else
                  _fit_response_scales(provider, corpus, floors))
        external_coefficients = None
    else:
        raise ValueError('Response scales must use fit, explicit, or calibration mode.')
    paired = spec.get('paired')
    objective_type = ThermalPairedFitFamilyObjective if paired is not None else ThermalFitFamilyObjective
    objective = objective_type(provider, corpus, scales, config, spec.get('tau', .4149511814117432))
    saved_state = (resume_payload or {}).get('provider_training_state', {}).get('thermal_joint_response', {})
    if paired is not None:
        if config.warmup_epochs != 10 or config.ramp_epochs != 40 or paired.get('measure') not in ('grid', 'catalogue'):
            raise ValueError('Paired TRAIN150 needs the declared ramp and grid or catalogue measure.')
        objective.auxiliary_fluid_measure = paired['measure']
        _attach_paired_response(provider, objective, config, paired, restored_state=saved_state)
    else:
        _attach_response(provider, objective, config, count=spec.get('families_per_epoch', 4),
                     seed=spec.get('family_seed', 0), restored_calibration=saved_state.get('calibration'),
                     external_calibration=external_coefficients)
    if identity and identity.get('thermal_joint_response') != provider.identity_payload()['thermal_joint_response']:
        raise ValueError('Saved Thermal FIT families or response scales differ from this resume.')
    return model, provider

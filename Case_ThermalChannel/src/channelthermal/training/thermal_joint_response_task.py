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
from .thermal_family_objective import ThermalFamilyObjectiveMixin, family_reference_roi_deltas, roi_support_runs
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
        coefficient_mode='gradient_norm', gradient_norm_ratios=None, response_h5=None,
        population=None, response_view='r17_1_raw'):
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
    if fit_manifest and response_h5:
        raise ValueError('Choose a FIT manifest or a direct response H5, not both.')
    if response_h5 and (population not in ('TRAIN150', 'TRAIN600') or response_view != 'r17_1_raw'):
        raise ValueError('Direct training requires an explicit population and R17-1 raw view.')
    if enabled and (not (fit_manifest or response_h5) or len(selections) != 1):
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
    if response_h5:
        recipe['thermal_joint_response']['h5'] = {'path': str(response_h5), 'population': population,
                                                    'view': response_view}
    return recipe


def build_r17_2_response_config(base_recipe, *, variant, population, response_scales,
        roi_profile, calibration_path, fit_manifest=None, response_h5=None, view='r17_1_raw'):
    """Configure one fresh R17-2 paired recipe with a shared TRAIN calibration.

    A raw manifest remains available for zero-update parity; formal recipes use
    the direct H5 after dataset parity. The calibration file may be pending until
    the declared common epoch-10 native snapshot has been measured.
    """
    if variant not in ('R-G', 'E', 'C', 'F') or population not in ('TRAIN150', 'TRAIN600'):
        raise ValueError('Choose an R17-2 variant and a declared TRAIN population.')
    if view != 'r17_1_raw' or not calibration_path or roi_profile is None:
        raise ValueError('R17-2 requires its raw effective view, ROI profile and common calibration path.')
    recipe = build_thermal_joint_response_config(base_recipe, fit_manifest=fit_manifest,
        response_h5=response_h5, population=population, response_view=view,
        response_scales=response_scales, response_weight=0. if variant == 'E' else 1.)
    recipe['thermal_joint_response']['paired'] = {'variant': variant, 'population': population, 'view': view,
        'measure': 'catalogue' if variant == 'C' else 'grid',
        'roi_profile': str(roi_profile) if isinstance(roi_profile, Path) else copy.deepcopy(roi_profile),
        'calibration_path': str(calibration_path)}
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
            baseline = self._read_endpoint(family, 'baseline')
            if cache is not None:
                cache['baseline'] = baseline
        edited = self._read_endpoint(family, action)
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

    def _read_endpoint(self, family, action):
        path = (self.cases[family]['source_case_dir'] if action == 'baseline'
                else self.records[(family, action)]['raw_case_dir'])
        return load_raw_endpoint(path)


class ThermalH5FitFamilyCorpus(ThermalFitFamilyCorpus):
    """FIT-only direct H5 view implementing the established family contract."""

    def __init__(self, path, training_ids, *, population, view='r17_1_raw'):
        from ..data.finite_responses import FiniteResponseStore

        if population not in ('TRAIN150', 'TRAIN600') or view != 'r17_1_raw':
            raise ValueError('Direct FIT training requires a declared population and R17-1 raw view.')
        self.path = str(Path(path).expanduser())
        self.population, self.view = population, view
        self.store = FiniteResponseStore(self.path)
        self.cases, self.records, self._actions, self._family_cache = {}, {}, {}, None
        training = set(map(str, training_ids))
        families = tuple(map(str, self.store.family_ids(role='fit', population=population)))
        if not families or len(set(families)) != len(families) or not set(families) <= training:
            raise ValueError('Direct FIT families must be unique selected TRAIN members.')
        for family in families:
            case = copy.deepcopy(self.store.family_metadata(family, role='fit', population=population))
            if not isinstance(case.get('source_config'), dict) or case.get('source') not in ('original', 'new'):
                raise ValueError('Direct FIT family lacks source controls or original/new identity.')
            self.cases[family] = case
            actions = tuple(self.store.actions(family, role='fit', population=population))
            if not actions or len(set(actions)) != len(actions):
                raise ValueError('Direct FIT actions must be nonempty and unique.')
            self._actions[family] = list(actions)
            for action in actions:
                row = copy.deepcopy(self.store.action_metadata(family, action, role='fit', population=population))
                labels = self.store.physical_delta(family, action, role='fit', population=population, kind='saved')
                if (not isinstance(labels, dict) or set(labels) != set(FUNCTIONALS)
                        or not all(math.isfinite(float(labels[name])) for name in FUNCTIONALS)
                        or not isinstance(row.get('centers'), list)):
                    raise ValueError('Direct FIT actions need saved finite J/P and ordered centers.')
                row['physical_delta'] = {name: float(labels[name]) for name in FUNCTIONALS}
                self.records[(family, action)] = row

    def _read_endpoint(self, family, action):
        return self.store.endpoint(family, action=action, role='fit', population=self.population, view=self.view)


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


class ThermalPairedFormalNativeTask(ThermalNativePortTask):
    """Full-population paired recipe: hold then cosine decay through 2000 epochs."""
    def _schedule_payload(self):
        lr = float(self.recipe['optimizer']['lr'])
        return {'peak_lr': lr, 'warmup_start_lr': lr, 'warmup_epochs': 0,
                'hold_through_epoch': 1000,
                'total_epochs': int(self.recipe['training']['optimizer_schedule_horizon']),
                'final_lr': float(self.recipe['optimizer'].get('final_lr', 3e-5))}


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


def fit_training_roi_scales(provider, corpus, *, population, view, resolution, reference):
    """Fit A/B scales over eligible TRAIN families with measured resolution floors.

    ``reference`` identifies compatible saved TRAIN repeat/reference records;
    this function never creates a solve or infers a missing resolution.
    """
    if population not in ('TRAIN150', 'TRAIN600') or view != 'r17_1_raw':
        raise ValueError('ROI scale fitting requires a declared TRAIN population and observation view.')
    if not isinstance(resolution, dict) or set(resolution) != {'A', 'B'}:
        raise ValueError('Measured A/B repeat-reference resolution is required.')
    measured = {name: float(value) for name, value in resolution.items()}
    if any(not math.isfinite(value) or value <= 0 for value in measured.values()):
        raise ValueError('Measured A/B resolution must be positive and finite.')
    if (not isinstance(reference, dict) or reference.get('role') != 'train'
            or reference.get('view') != view or not reference.get('source')
            or not reference.get('native_case_ids')
            or not set(map(str, reference['native_case_ids'])) <= set(map(str, provider.training_case_ids))):
        raise ValueError('Resolution must identify compatible saved TRAIN references and their view.')
    squares = {'A': [], 'B': []}
    support_counts, support_runs = {}, {}
    families = corpus.family_ids()
    if not families or not set(families) <= set(map(str, provider.training_case_ids)):
        raise ValueError('ROI scales require selected TRAIN FIT families only.')
    for family in families:
        with corpus.family_context(family):
            pairs = [corpus.pair(provider, family, action) for action in corpus.actions(family)]
            if not pairs:
                raise ValueError('ROI scale fitting requires physical edits.')
            batches = [pairs[0]['baseline']] + [pair['edited'] for pair in pairs]
            support, deltas = family_reference_roi_deltas(batches, pairs[0]['raw_baseline'])
            support_counts[family] = {name: int(mask.sum()) for name, mask in support.items()}
            support_runs[family] = {name: roi_support_runs(mask) for name, mask in support.items()}
            for name in squares:
                squares[name].append(sum(float(delta.square()) for delta in deltas[name]) / len(deltas[name]))
    rms = {name: math.sqrt(sum(values) / len(values)) for name, values in squares.items()}
    return {'schema': 'thermal_training_roi_scales_v1', 'population': population, 'view': view,
            'fit_family_ids': list(families), 'resolution': measured, 'resolution_reference': copy.deepcopy(reference),
            'floors': {name: 4. * value for name, value in measured.items()}, 'rms': rms,
            'scales': {name: max(rms[name], 4. * measured[name]) for name in squares},
            'support_counts': support_counts, 'support_runs': support_runs}


def _validated_roi_profile(profile, *, population, view, families, training_ids):
    if isinstance(profile, (str, Path)):
        profile = json.loads(Path(profile).expanduser().read_text())
    if (not isinstance(profile, dict) or profile.get('schema') != 'thermal_training_roi_scales_v1'
            or profile.get('population') != population or profile.get('view') != view
            or tuple(map(str, profile.get('fit_family_ids', ()))) != tuple(families)):
        raise ValueError('ROI scale profile must match the ordered TRAIN FIT population and view.')
    reference = profile.get('resolution_reference', {})
    if (reference.get('role') != 'train' or reference.get('view') != view or not reference.get('source')
            or not reference.get('native_case_ids')
            or not set(map(str, reference['native_case_ids'])) <= set(map(str, training_ids))):
        raise ValueError('ROI resolution profile lacks compatible TRAIN reference provenance.')
    for key in ('resolution', 'floors', 'rms', 'scales'):
        if not isinstance(profile.get(key), dict) or set(profile[key]) != {'A', 'B'}:
            raise ValueError(f'ROI profile requires A/B {key}.')
    for name in ('A', 'B'):
        resolution, floor, rms, scale = (float(profile[key][name]) for key in ('resolution', 'floors', 'rms', 'scales'))
        if (not all(math.isfinite(value) for value in (resolution, floor, rms, scale))
                or resolution <= 0 or rms < 0 or floor != 4. * resolution or scale != max(floor, rms)):
            raise ValueError('ROI scale must equal the TRAIN RMS with its four-times-resolution floor.')
    counts = profile.get('support_counts', {})
    if set(counts) != set(families) or any(set(counts[family]) != {'A', 'B'}
            or any(type(counts[family][name]) is not int or counts[family][name] < 1 for name in ('A', 'B'))
            for family in families):
        raise ValueError('ROI profile must record nonempty common supports for every FIT family.')
    runs = profile.get('support_runs', {})
    if set(runs) != set(families):
        raise ValueError('ROI profile must record each ordered common support.')
    for family in families:
        if set(runs[family]) != {'A', 'B'}:
            raise ValueError('ROI profile requires both region supports.')
        for name in ('A', 'B'):
            pairs = runs[family][name]
            if (not isinstance(pairs, list) or not pairs
                    or any(not isinstance(pair, list) or len(pair) != 2
                           or type(pair[0]) is not int or type(pair[1]) is not int
                           or pair[0] < 0 or pair[1] <= pair[0] for pair in pairs)
                    or any(left[1] >= right[0] for left, right in zip(pairs, pairs[1:]))
                    or sum(stop - start for start, stop in pairs) != counts[family][name]):
                raise ValueError('ROI support ranges must reproduce their declared receiver counts.')
    return copy.deepcopy(profile)


def _norm(parameters, loss=None):
    if loss is None:
        gradients = (p.grad for p in parameters)
    else:
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients if g is not None))


def _paired_population(families):
    sizes = tuple(len(families[name]) for name in ('original', 'new'))
    if sizes == (15, 25):
        return {'original': 1, 'new': 2}, 15, 4
    if sizes == (48, 96):
        return {'original': 4, 'new': 8}, 60, 14
    raise ValueError('Paired fitting requires the eligible TRAIN150 or TRAIN600 FIT families.')


def calibrate_paired_response(model, provider, *, epoch, native_ids, fit_families):
    """Measure grid coefficients and catalogue gradients on fixed TRAIN panels.

    The caller chooses the declared input-only panels and persists the receipt.
    No optimizer update, parameter, gradient, mode or random state is retained.
    """
    from .r16_10_diagnostics import gradient_diagnostics, preserved_state
    objective = provider.family_objective
    native, panel = tuple(map(str, native_ids)), tuple(tuple(row) for row in fit_families)
    _, native_count, family_count = _paired_population(objective.schedule.families)
    if (epoch != 10 or objective.auxiliary_fluid_measure != 'grid' or len(native) != native_count
            or len(set(native)) != native_count or not set(native).issubset(provider.training_case_ids)
            or len(panel) != family_count or len(set(panel)) != family_count
            or any(f not in objective.schedule.families.get(s, ()) for s, f in panel)):
        raise ValueError(f'Paired calibration requires epoch-10 grid and valid TRAIN{native_count}/FIT{family_count} panels.')
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


def _r17_2_panel(values):
    values = tuple(values)
    selected = set(np.random.default_rng(20261108).choice(
        len(values), size=max(1, int(np.floor(.1 * len(values) + .5))), replace=False).tolist())
    return tuple(value for index, value in enumerate(values) if index in selected)


def calibrate_r17_2_response(model, provider, *, epoch, native_ids, fit_families):
    """Measure the four common TRAIN-only coefficients without optimizer updates."""
    from honf_runtime.unified_training import SamplingKey
    from .r16_10_diagnostics import gradient_diagnostics, preserved_state
    objective = provider.family_objective
    native = tuple(map(str, native_ids))
    panel = tuple((str(source), str(family)) for source, family in fit_families)
    _, native_count, family_count = _paired_population(objective.schedule.families)
    population = tuple((source, family) for source in ('original', 'new')
                       for family in objective.schedule.families[source])
    if (epoch != 10 or len(native) != native_count or len(panel) != family_count
            or native != _r17_2_panel(provider.training_case_ids) or panel != _r17_2_panel(population)
            or getattr(objective, 'roi_scales', None) is None):
        raise ValueError(f'R17-2 calibration requires the fixed TRAIN{native_count}/FIT{family_count} panel and ROI scales.')
    if (provider.sampling_version != SamplingKey.CASE_EPOCH_VERSION or provider.primary_queries != 1024
            or provider.port_angle_count != 64):
        raise ValueError('R17-2 calibration requires the sampled native TRAIN query and port semantics.')
    # The case/epoch sampler deliberately ignores update and microbatch packing.
    # Using singleton batches gives the equal-case panel mean while reproducing
    # each case's literal epoch-10 native training draws.
    sampling_key = SamplingKey(seed=provider.seed, epoch=epoch, update_index=0, microbatch_index=0,
        stage='joint', arm=provider.recipe.get('training', {}).get('sampling_arm', 'R13-3-Relative'),
        sampling_version=provider.sampling_version, dataset_id=provider.sampling_dataset_id)
    native_draws = []
    def native_loss():
        for case_id in native:
            batch = provider.make_batch((case_id,), sampling_key)
            if (tuple(batch.case_keys) != (case_id,) or batch.targets.field_targets.shape[-2] != 1024
                    or batch.targets.material_targets.shape[-1] != 32
                    or batch.targets.interface_target.shape[-2] != 64 or batch.receivers.ntheta != 64):
                raise ValueError('Calibration batch differs from sampled native TRAIN receivers.')
            native_draws.append(case_id)
            prediction, _ = provider.predict_native(model, batch.scene_inputs, batch.receivers,
                                                     model.architecture, 'joint', epoch=epoch)
            yield provider.endpoint_state_loss(prediction, batch) / len(native)
    def family_loss(component, measure):
        def pieces():
            objective.auxiliary_fluid_measure = measure
            for source, family in panel:
                values, _ = objective.family_components(model, source, family, components=(component,))
                yield values[component] / len(panel)
        return pieces
    losses = {'native': native_loss, 'endpoint': family_loss('endpoint', 'grid'),
              'response_grid': family_loss('response', 'grid'),
              'response_catalogue': family_loss('response', 'catalogue'),
              'functional': family_loss('functional', 'grid')}
    original_measure = objective.auxiliary_fluid_measure
    original_training_samples = provider.training_samples
    try:
        with preserved_state(model):
            model.train()
            result = gradient_diagnostics(model, losses, require_nonzero=True)
    finally:
        objective.auxiliary_fluid_measure = original_measure
        provider.training_samples = original_training_samples
    if len(native_draws) != len(native):
        raise ValueError('Calibration did not measure the complete sampled native panel exactly once.')
    norms = result['groups']['global']['norms']
    coefficients = {name: (.25 if name == 'endpoint' else .10) * norms['native'] / norms[name]
                    for name in ('endpoint', 'response_grid', 'response_catalogue', 'functional')}
    return {'schema': 'thermal_r17_2_calibration_v1', 'epoch': epoch,
            'native_case_ids': list(native), 'fit_families': [list(row) for row in panel],
            'response_scales': dict(objective.response_scales), 'roi_scales': dict(objective.roi_scales),
            'native_sampling': {'seed': sampling_key.seed, 'epoch': sampling_key.epoch,
                'sampling_version': sampling_key.sampling_version, 'dataset_id': sampling_key.dataset_id,
                'stage': sampling_key.stage, 'arm': sampling_key.arm, 'update_index': sampling_key.update_index,
                'microbatch_index': sampling_key.microbatch_index, 'cases_per_batch': 1,
                'fluid_queries_per_case': 1024, 'material_queries_per_module': 32,
                'surface_and_outside_angles': 64,
                'semantics': 'case_epoch_v1 per-case native streams; update, microbatch, stage and arm do not affect draws'},
            'gradient_diagnostics': result, 'coefficients': coefficients}


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


def _attach_r17_2_response(provider, objective, config, spec, *, population, view):
    """Attach the four R17-2 recipes with one deferred population calibration."""
    variants = {'R-G': ('grid', 'response', 'response_grid'),
                'E': ('grid', None, None),
                'C': ('catalogue', 'response', 'response_catalogue'),
                'F': ('grid', 'functional', 'functional')}
    variant = spec.get('variant')
    if variant not in variants:
        raise ValueError('R17-2 variant must be R-G, E, C or F.')
    measure, component, coefficient = variants[variant]
    if spec.get('measure', measure) != measure:
        raise ValueError('R17-2 auxiliary measure does not match the declared recipe.')
    if (config.warmup_epochs != 10 or config.ramp_epochs != 40 or config.endpoint_weight != 1.
            or config.field_response_weight != .8 or dict(config.functional_weights) != {'J': .1, 'P': .1}
            or config.response_weight not in ((0., 1.) if variant == 'E' else (1.,))):
        raise ValueError('R17-2 needs the declared fixed endpoint, response and ramp semantics.')
    sources = {name: [family for family, row in objective.corpus.cases.items()
                      if row.get('source') == name] for name in ('original', 'new')}
    counts, native_count, family_count = _paired_population(sources)
    if ((population, native_count) not in (('TRAIN150', 15), ('TRAIN600', 60))
            or len(provider.training_case_ids) != {'TRAIN150': 150, 'TRAIN600': 600}[population]):
        raise ValueError('R17-2 native membership and FIT families must match the declared TRAIN population.')
    if sum(map(len, sources.values())) != len(objective.corpus.cases):
        raise ValueError('Each R17-2 FIT family must declare its original or new source.')
    objective.schedule = EligibleFamilySchedule(sources['original'], sources['new'], provider.training_case_ids,
                                                 seed=0, preserve_order=True)
    if any(len(objective.actions(source, family)) != 4 for source in sources for family in sources[source]):
        raise ValueError('Each R17-2 FIT family requires its four eligible edits.')
    profile = _validated_roi_profile(spec.get('roi_profile'), population=population, view=view,
        families=objective.corpus.family_ids(), training_ids=provider.training_case_ids)
    objective.roi_scales = dict(profile['scales'])
    objective.roi_support_runs = copy.deepcopy(profile['support_runs'])
    objective.auxiliary_fluid_measure = measure
    calibration_path = Path(spec['calibration_path']).expanduser()
    names = ('endpoint',) if component is None else ('endpoint', component)
    state = {'calibration': None}
    provider.family_objective = objective
    provider.roi_scale_profile = profile
    provider.training_history_in_logs = True
    provider.auxiliary_loss_names = tuple('family/' + name for name in names)
    previous_metadata = provider.loss_metadata
    def loss_metadata(self):
        result = dict(previous_metadata())
        for name in names:
            result['family/' + name] = {'panel': 'physics', 'label': 'FIT family ' + name,
                'unit': 'scaled squared error',
                'formula': ('equal FIT family, five endpoints and paired full/P95 view mean' if name == 'endpoint'
                            else 'equal FIT family/action and paired full/P95 view mean'),
                'weight': 'fixed TRAIN calibration and epoch ramp'}
        return result
    provider.loss_metadata = MethodType(loss_metadata, provider)
    def load_calibration():
        receipt = json.loads(calibration_path.read_text())
        native = tuple(map(str, receipt.get('native_case_ids', ())))
        fit = tuple((str(row[0]), str(row[1])) for row in receipt.get('fit_families', ()))
        family_population = tuple((source, family) for source in ('original', 'new')
                                  for family in objective.schedule.families[source])
        coeffs = receipt.get('coefficients', {})
        if (receipt.get('schema') != 'thermal_r17_2_calibration_v1' or receipt.get('epoch') != 10
                or len(native) != native_count or native != _r17_2_panel(provider.training_case_ids)
                or len(fit) != family_count or fit != _r17_2_panel(family_population)
                or _scales(receipt.get('response_scales')) != objective.response_scales
                or receipt.get('roi_scales') != objective.roi_scales
                or set(coeffs) != {'endpoint', 'response_grid', 'response_catalogue', 'functional'}
                or any(not math.isfinite(float(value)) or float(value) <= 0 for value in coeffs.values())):
            raise ValueError('R17-2 calibration must match the fixed TRAIN/FIT panel, scales and four measured coefficients.')
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
            components, work = objective.family_components(model, source, family, components=names)
            weights = {'endpoint': state['calibration']['coefficients']['endpoint']}
            if component is not None:
                weights[component] = state['calibration']['coefficients'][coefficient]
            terms = {'family/' + name: LossTerm(components[name], 1.,
                     ramp * float(weights[name]) * update_count / len(selected)) for name in names}
            work.update(family_presentations=1, edited_presentations=4)
            yield terms, work
    provider.auxiliary_update_terms = MethodType(auxiliary_update_terms, provider)
    provider.state_response_metadata = MethodType(lambda self: {'calibration': copy.deepcopy(state['calibration'])}, provider)
    provider.response_epoch_family_ids = MethodType(
        lambda self, epoch: [f'{source}:{family}' for source, family in objective.schedule.current]
        if epoch > 10 else [], provider)
    def begin_training_epoch(self, model, epoch):
        core = getattr(model, 'core', model)
        if not hasattr(core, 'query_tile_size'):
            raise ValueError('R17-2 paired execution requires an anchored query tile.')
        core.query_tile_size = 128
        return {'active_tile': 128}
    provider.begin_training_epoch = MethodType(begin_training_epoch, provider)
    old_state = getattr(provider, 'training_state_dict', lambda: {})
    old_load = getattr(provider, 'load_training_state_dict', lambda saved: None)
    def training_state_dict(self):
        return {'native': old_state(), 'thermal_joint_response': {'calibration': copy.deepcopy(state['calibration']),
                'schedule': objective.schedule.state_dict()}}
    def load_training_state_dict(self, saved):
        old_load(saved['native'])
        paired = saved['thermal_joint_response']
        objective.schedule.load_state_dict(paired['schedule'])
        state['calibration'] = copy.deepcopy(paired['calibration'])
        if state['calibration'] is not None and state['calibration'] != load_calibration():
            raise ValueError('R17-2 calibration differs from the saved paired state.')
    provider.training_state_dict = MethodType(training_state_dict, provider)
    provider.load_training_state_dict = MethodType(load_training_state_dict, provider)
    previous_identity = provider.identity_payload
    def identity_payload(self):
        result = dict(previous_identity())
        result['native_only_training'] = False
        result['response_training_family_ids'] = list(objective.corpus.family_ids())
        result['response_supervision'] = 'explicit selected TRAIN FIT families'
        result['objective'].update(response_training_families=list(objective.corpus.family_ids()), response_fit=True)
        result['thermal_joint_response'] = {'schema': SCHEMA, 'response_source': objective.corpus.path,
            'population': population, 'view': view, 'variant': variant,
            'families': {source: list(ids) for source, ids in objective.schedule.families.items()},
            'counts': counts, 'response_scales': dict(objective.response_scales),
            'roi_profile': copy.deepcopy(profile), 'measure': measure,
            'calibration_path': str(calibration_path), 'tau': objective.tau, 'query_tile_size': 128}
        return result
    provider.identity_payload = MethodType(identity_payload, provider)
    return provider


def _attach_paired_response(provider, objective, config, spec, restored_state=None):
    """Keep calibration pending until the common measured grid receipt exists."""
    sources = {name: [family for family, row in objective.corpus.cases.items()
                      if row.get('source') == name] for name in ('original', 'new')}
    counts, native_count, family_count = _paired_population(sources)
    if sum(map(len, sources.values())) != len(objective.corpus.cases):
        raise ValueError('Every paired FIT family must declare its original or new source.')
    objective.schedule = EligibleFamilySchedule(sources['original'], sources['new'], provider.training_case_ids, seed=0)
    if any(len(objective.actions(source, family)) != 4 for source in sources for family in sources[source]):
        raise ValueError('Every family requires four edits.')
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
        if (len(native) != native_count or len(set(native)) != native_count or not set(native).issubset(provider.training_case_ids)
                or len(fit) != family_count or len(set(fit)) != family_count
                or any(f not in objective.schedule.families.get(s, ()) for s, f in fit)
                or native != panel(provider.training_case_ids) or fit != panel(population)
                or receipt.get('measure') != 'grid' or receipt.get('epoch') != 10
                or _scales(receipt.get('response_scales')) != objective.response_scales):
            raise ValueError(f'Common calibration must identify the TRAIN{native_count}/FIT{family_count} grid panel and matching scales.')
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
        if native_count == 60 or epoch <= 50:
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
    h5_spec = spec.get('h5')
    if config.enabled and not spec.get('fit_manifest') and not h5_spec:
        raise ValueError('Enabled state/response requires a FIT family manifest or direct response H5, physical labels and scales.')
    if config.enabled and (bool(spec.get('fit_manifest')) == bool(h5_spec) or not isinstance(spec.get('scales'), dict)):
        raise ValueError('Enabled state/response requires exactly one FIT source and physical scales.')
    arm = arm or recipe['member_routing'].get('arm', ARM_DETAIL)
    identity = (resume_payload or {}).get('experiment_identity', {}).get('provider_identity', {})
    formal = (spec.get('paired') is not None and recipe.get('training', {}).get('membership')
              in ('full_train600', 'scratch_train600_exposed_val89'))
    model, provider = _build_thermal_member_routing_task(recipe, arm=arm, device=device,
        task_type=ThermalPairedFormalNativeTask if formal else ThermalNativePortTask,
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
    if h5_spec:
        if not isinstance(h5_spec, dict):
            raise ValueError('Direct H5 source must declare path, population and view.')
        corpus = ThermalH5FitFamilyCorpus(h5_spec['path'], provider.training_case_ids,
                                         population=h5_spec['population'], view=h5_spec['view'])
    else:
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
        if paired.get('variant') is not None:
            population = paired.get('population')
            view = paired.get('view', 'r17_1_raw')
            if h5_spec and (h5_spec['population'] != population or h5_spec['view'] != view):
                raise ValueError('R17-2 recipe population/view differs from its direct H5 source.')
            if population not in ('TRAIN150', 'TRAIN600') or view != 'r17_1_raw':
                raise ValueError('R17-2 requires the declared training population and R17-1 raw view.')
            _attach_r17_2_response(provider, objective, config, paired, population=population, view=view)
        else:
            if config.warmup_epochs != 10 or config.ramp_epochs != 40 or paired.get('measure') not in ('grid', 'catalogue'):
                raise ValueError('Paired training needs the declared ramp and grid or catalogue measure.')
            objective.auxiliary_fluid_measure = paired['measure']
            _attach_paired_response(provider, objective, config, paired, restored_state=saved_state)
    else:
        _attach_response(provider, objective, config, count=spec.get('families_per_epoch', 4),
                     seed=spec.get('family_seed', 0), restored_calibration=saved_state.get('calibration'),
                     external_calibration=external_coefficients)
    if identity and identity.get('thermal_joint_response') != provider.identity_payload()['thermal_joint_response']:
        raise ValueError('Saved Thermal FIT families or response scales differ from this resume.')
    return model, provider

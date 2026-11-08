"""Scratch R16-10 physical family objectives and provider adapter."""
from __future__ import annotations

import math
from types import MethodType, SimpleNamespace
from dataclasses import fields, replace
import numpy as np
import torch

from honf_runtime.unified_training import LossTerm
from .r16_10_corpus import EligibleFamilySchedule
from .r16_5_decision_task import decision_losses, _endpoint_functionals
from .joint_member_routing_task import ThermalJointPairPredictions
from .thermal_state_response_task import (ACTIONS, FIELD_ROLES, _field_components,
    _equal_module_mean, _reference_delta_squares, fit_response_scales, response_pair, _native_endpoint_batch)


def auxiliary_ramp(epoch):
    return min(1., max(0., (int(epoch) - 10) / 40.))


def _views(prediction):
    return ([prediction.full, prediction.restricted]
            if isinstance(prediction, ThermalJointPairPredictions) else [prediction])


def _stack_family_batches(batches):
    """Combine the same family's one-case endpoints without changing their order."""
    def combine(values):
        first = values[0]
        if torch.is_tensor(first):
            return torch.cat(values, dim=0)
        if isinstance(first, dict):
            if any(set(value) != set(first) for value in values):
                raise ValueError('Endpoint input structures differ.')
            return {key: combine([value[key] for value in values]) for key in first}
        if isinstance(first, tuple):
            return sum(values, ())
        if any(value != first for value in values):
            raise ValueError('Endpoint batch metadata differs.')
        return first
    def combine_dataclass(rows):
        return replace(rows[0], **{field.name: combine([getattr(row, field.name) for row in rows])
                                  for field in fields(rows[0])})
    return replace(batches[0],
        scene_inputs=combine_dataclass([batch.scene_inputs for batch in batches]),
        receivers=combine_dataclass([batch.receivers for batch in batches]),
        targets=combine_dataclass([batch.targets for batch in batches]),
        case_keys=sum((batch.case_keys for batch in batches), ()))


def _split_family_prediction(prediction, count):
    """Take differentiable views of the fields and masks consumed by family losses."""
    def branch_slice(branch, index):
        prepared = branch.native_prepared
        stencil = {name: SimpleNamespace(valid=item.valid[index:index + 1])
                   for name, item in prepared.stencils.items()}
        local_prepared = SimpleNamespace(stencils=stencil, source_present=prepared.source_present[index:index + 1])
        return SimpleNamespace(native_prepared=local_prepared,
            native_main={name: value[index:index + 1] for name, value in branch.native_main.items()})
    if not isinstance(prediction, ThermalJointPairPredictions):
        raise ValueError('Batched family execution requires paired access views.')
    return [ThermalJointPairPredictions(branch_slice(prediction.full, i),
              branch_slice(prediction.restricted, i), {}) for i in range(count)]


class R1610FamilyObjective:
    """Only explicitly eligible FIT families are opened and used for fitting."""
    def __init__(self, provider, original, new, native_ids, original_labels, *, seed=0,
                 response_scales=None, sensitivity=None, research_scales=None,
                 tau=0.4149511814117432):
        self.provider, self.corpora = provider, {'original': original, 'new': new}
        self.schedule = EligibleFamilySchedule(original.family_ids('fit'), new.family_ids('fit'), native_ids, seed=seed)
        self.original_labels = original_labels
        self.response_scales, self.sensitivity, self.research_scales = response_scales, sensitivity, research_scales
        self.tau = float(tau)

    def actions(self, source, family):
        if family not in self.schedule.families[source]:
            raise ValueError('Family is outside native TRAIN/FIT intersection.')
        actions = ACTIONS if source == 'original' else self.corpora[source].actions_for_family(family)
        if len(actions) != 4:
            raise ValueError('Each family must contain four actions.')
        return tuple(actions)

    def pair(self, source, family, action):
        if action not in self.actions(source, family):
            raise ValueError('Action is outside the eligible family.')
        corpus = self.corpora[source]
        if source == 'new':
            return corpus.pair(self.provider, family, action, role='fit')
        pair = response_pair(self.provider, corpus, family, action, role='fit')
        original = self.provider._load_native_cases([family])[0]
        pair['baseline'] = _native_endpoint_batch(self.provider, original, pair['raw_baseline'], 'baseline')
        return pair

    def label(self, source, family, action):
        self.actions(source, family)
        if source == 'original':
            return self.original_labels[(family, action)]
        return self.corpora[source].physical_delta(family, action, role='fit')

    def fit_scales(self, state_scales, repeatability, floors):
        stats = {}
        for source, families in self.schedule.families.items():
            for family in families:
                row = {}
                for canonical, action in zip(ACTIONS, self.actions(source, family)):
                    pair = self.pair(source, family, action)
                    values = _reference_delta_squares(pair)
                    item = {}
                    for name, (delta, mask) in values.items():
                        if not bool(mask.any()):
                            raise ValueError(f'Empty response support {source}/{family}/{action}/{name}.')
                        item[name] = (delta[mask].detach().cpu().numpy(), np.ones(int(mask.sum()), dtype=bool))
                    item.update(self.label(source, family, action))
                    row[canonical] = item
                stats[(source, family)] = row
        self.response_scales = fit_response_scales(stats, state_scales, repeatability, floors)
        return self.response_scales

    def fit_research_scales(self, sensitivity):
        self.sensitivity = dict(sensitivity)
        result = {}
        for name in ('J', 'P'):
            radius = 4 * float(sensitivity[name])
            values = [abs(float(self.label(source, family, action)[name]))
                      for source, families in self.schedule.families.items() for family in families
                      for action in self.actions(source, family)]
            resolved = np.asarray(values)[np.asarray(values) > radius]
            if not math.isfinite(radius) or radius <= 0 or not len(resolved):
                raise ValueError(f'No resolved eligible FIT actions for {name}.')
            result[name] = float(np.quantile(resolved, .25))
        self.research_scales = result
        return result

    def family_components(self, model, source, family, *, batched=True):
        if self.response_scales is None or self.sensitivity is None or self.research_scales is None:
            raise ValueError('Eligible FIT response and decision scales must be configured.')
        actions = self.actions(source, family)
        pairs = [self.pair(source, family, action) for action in actions]
        batches = [pairs[0]['baseline']] + [p['edited'] for p in pairs]
        raws = [pairs[0]['raw_baseline']] + [p['raw_edited'] for p in pairs]
        batches = [self.provider.with_raw_port_targets(b, raw) for b, raw in zip(batches, raws)]
        predictions, work = [], {}
        prediction_batches = [_stack_family_batches(batches)] if batched else batches
        for batch in prediction_batches:
            pred, aux = self.provider.predict_native(model, batch.scene_inputs, batch.receivers,
                                                      model.architecture, 'joint', epoch=0)
            if batched:
                predictions.extend(_split_family_prediction(pred, len(batches)))
            else:
                predictions.append(pred)
            for key, value in aux.get('work', {}).items():
                if isinstance(value, (int, float)):
                    work[key] = work.get(key, 0.) + float(value)
        views = [_views(p) for p in predictions]
        if any(len(v) != 2 for v in views):
            raise ValueError('R16-10 requires both full and restricted joint views.')
        endpoint = sum(self.provider.endpoint_state_loss(p, b) for p, b in zip(predictions, batches)) / 5
        fields = [[_field_components(pred, batch) for pred in row] for row, batch in zip(views, batches)]
        functions = torch.stack([_endpoint_functionals(pred, batch, raw, self.tau)
                                 for pred, batch, raw in zip(predictions, batches, raws)], dim=1)
        labels = [self.label(source, family, action) for action in actions]
        field_losses, j_losses, p_losses = [], [], []
        for action_index, label in enumerate(labels, start=1):
            for view in range(2):
                terms = []
                for name in FIELD_ROLES:
                    bp, bt, bm = fields[0][view][name]
                    ep, et, em = fields[action_index][view][name]
                    if bp.shape != ep.shape or bt.shape != et.shape:
                        raise ValueError('Response receiver identity or shape changed.')
                    mask = bm & em
                    if not bool(mask.any()):
                        raise ValueError(f'No common support for {name}.')
                    scale = self.response_scales[name]
                    scale = float(scale['scale'] if isinstance(scale, dict) else scale)
                    terms.append(float(self.provider.native_weights[name]) * _equal_module_mean((ep-bp-et+bt)/scale, mask))
                field_losses.append(sum(terms) / sum(float(self.provider.native_weights[n]) for n in FIELD_ROLES))
                for idx, (name, destination) in enumerate((('J', j_losses), ('P', p_losses))):
                    scale = self.response_scales[name]
                    scale = float(scale['scale'] if isinstance(scale, dict) else scale)
                    destination.append(((functions[view, action_index, idx] - functions[view, 0, idx] - float(label[name])) / scale).square())
        mean = lambda x: sum(x) / len(x)
        response = .8 * mean(field_losses) + .1 * mean(j_losses) + .1 * mean(p_losses)
        truth = [[float(label[n]) for n in ('J', 'P')] for label in labels]
        decision, counts = decision_losses(functions, truth, self.sensitivity, self.research_scales,
                                           {n: self.response_scales[n] for n in ('J', 'P')})
        objective = .4 * decision['feasibility'] + .4 * decision['ranking'] + .2 * decision['near_null']
        work.update({f'decision_{k}': v for k, v in counts.items()})
        return {'endpoint': endpoint, 'response': response, 'decision': objective}, work

    def panel_components(self, model, panel):
        rows = [self.family_components(model, source, family)[0] for source, family in panel]
        if not rows:
            raise ValueError('Calibration panel is empty.')
        return {name: sum(row[name] for row in rows) / len(rows) for name in ('endpoint', 'response', 'decision')}


def attach_family_objective(provider, objective, *, arm, coefficients, families_per_epoch):
    if arm not in ('E', 'R', 'D', 'V'):
        raise ValueError('Unknown auxiliary arm.')
    names = ('endpoint',) if arm == 'E' else (('endpoint', 'response', 'decision') if arm == 'D' else ('endpoint', 'response'))
    if any(not math.isfinite(float(coefficients[n])) or float(coefficients[n]) <= 0 for n in names):
        raise ValueError('Auxiliary coefficients must be positive and finite.')
    provider.family_objective = objective
    previous_identity = provider.identity_payload
    def identity_payload(self):
        payload = dict(previous_identity())
        payload['r16_10_family'] = {'arm': arm, 'coefficients': dict(coefficients),
            'response_scales': objective.response_scales, 'sensitivity': objective.sensitivity,
            'research_scales': objective.research_scales,
            'eligible_families': {k: list(v) for k, v in objective.schedule.families.items()}}
        return payload
    provider.identity_payload = MethodType(identity_payload, provider)
    provider.auxiliary_loss_names = tuple('family/' + n for n in names)
    def auxiliary_update_terms(self, model, *, epoch, update_index, update_count):
        ramp = auxiliary_ramp(epoch)
        if not ramp:
            return
        families = objective.schedule.epoch(epoch, families_per_epoch)
        slots = np.linspace(0, update_count - 1, len(families), dtype=int)
        for (source, family), slot in zip(families, slots):
            if int(slot) != update_index:
                continue
            components, work = objective.family_components(model, source, family)
            # The engine reports an arithmetic mean of native update means, including
            # its short last batch. Match that reduction exactly at the epoch level.
            terms = {'family/' + n: LossTerm(components[n], 1., ramp * float(coefficients[n]) * update_count / len(families)) for n in names}
            work.update(family_presentations=1, edited_presentations=4)
            yield terms, work
    provider.auxiliary_update_terms = MethodType(auxiliary_update_terms, provider)
    old_state = getattr(provider, 'training_state_dict', lambda: {})
    old_load = getattr(provider, 'load_training_state_dict', lambda state: None)
    def training_state_dict(self):
        return {'native': old_state(), 'r16_10_family': objective.schedule.state_dict()}
    def load_training_state_dict(self, state):
        old_load(state['native'])
        objective.schedule.load_state_dict(state['r16_10_family'])
    provider.training_state_dict = MethodType(training_state_dict, provider)
    provider.load_training_state_dict = MethodType(load_training_state_dict, provider)
    return provider


def build_thermal_r16_10_task(recipe, *, arm, device='cpu', resume_payload=None):
    """Construct a scratch provider; auxiliary arms require population calibration."""
    import json
    from pathlib import Path
    from .r16_10_native_task import build_thermal_r16_10_native_task
    from .thermal_state_response_task import ThermalResponseCorpus
    from .r16_8_corpus import ThermalR168Corpus
    spec = recipe['r16_10_family']
    if spec.get('schema') != 'thermal_r16_10_family_v1' or spec.get('population') not in (150, 600):
        raise ValueError('Invalid R16-10 family recipe.')
    resume_payload = resume_payload or {}
    identity = resume_payload.get('experiment_identity', {}).get('provider_identity', {})
    family_arm = spec['arm']
    calibration = (json.loads(Path(spec['coefficients_path']).read_text())
                   if family_arm != 'B0' else None)
    adapter_state = identity.get('adapter_state')
    normalization_source = identity.get('source_binding', {}).get('normalization_source')
    if not identity and calibration is not None:
        adapter_state = calibration['adapter_state']
        normalization_source = calibration['normalization_source']
    model, provider = build_thermal_r16_10_native_task(recipe, arm=arm, device=device,
        saved_adapter_state=adapter_state, saved_normalization_source=normalization_source)
    if family_arm == 'V':
        from .r16_8_content_task import enable_content
        enable_content(model, 'C8')
    if family_arm == 'B0':
        return model, provider
    original = ThermalResponseCorpus(spec['original_input_recipe'], spec['original_ledger'])
    new = ThermalR168Corpus(spec['new_fit_manifest'])
    labels = {(row['family'], row['action']): row['delta'] for row in calibration['original_labels']}
    objective = R1610FamilyObjective(provider, original, new, provider.training_case_ids, labels,
        response_scales=calibration['response_scales'], sensitivity=calibration['sensitivity'],
        research_scales=calibration['research_scales'])
    expected = (15, 25) if spec['population'] == 150 else (48, 96)
    if tuple(len(objective.schedule.families[k]) for k in ('original', 'new')) != expected:
        raise ValueError('Eligible original/new family membership disagrees with population.')
    expected_labels = {(family, action) for family in objective.schedule.families['original'] for action in ACTIONS}
    if set(labels) != expected_labels:
        raise ValueError('Calibration original labels must contain exactly eligible actions.')
    counts = {'original': 1, 'new': 2} if spec['population'] == 150 else {'original': 4, 'new': 8}
    attach_family_objective(provider, objective, arm=family_arm,
                            coefficients=calibration['coefficients'], families_per_epoch=counts)
    if identity and identity.get('r16_10_family') != provider.identity_payload()['r16_10_family']:
        raise ValueError('Saved R16-10 family objective differs from current calibration.')
    return model, provider

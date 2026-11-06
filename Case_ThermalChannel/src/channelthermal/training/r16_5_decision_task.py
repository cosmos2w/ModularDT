"""R16-5 native/port fitting and pooled, full-field decision supervision."""
from __future__ import annotations

import copy
import math
import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from honf_runtime.unified_training import LossTerm
from ..native_functionals import native_design_functionals
from .joint_member_routing_task import ThermalJointPairPredictions, _PairedMemberRoutingTask
from .r13_8_anchored_task import ARM_DETAIL
from .thermal_port_response_task import ThermalPortResponseTask
from .thermal_state_response_task import (
    ACTIONS, ThermalStateResponseTask, predict_response_pair, response_loss_terms, response_pair,
)

DECISION_WEIGHTS = {'feasibility': .02, 'ranking': .02, 'near_null': .01}
FUNCTIONS = ('J', 'P')


def fit_decision_scales(labels, families, sensitivity):
    """Fit Q25(|delta|), excluding unresolved actions, using original FIT only."""
    if set(sensitivity) != set(FUNCTIONS):
        raise ValueError('Decision sensitivity requires J and P.')
    radius = {name: 4 * float(sensitivity[name]) for name in FUNCTIONS}
    scales = {}
    for name in FUNCTIONS:
        if not math.isfinite(radius[name]) or radius[name] <= 0:
            raise ValueError('Decision sensitivities must be positive and finite.')
        values = np.asarray([abs(float(labels[(family, action)][name]))
                             for family in families for action in ACTIONS], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite FIT functional response.')
        resolved = values[values > radius[name]]
        if not len(resolved):
            raise ValueError(f'No resolved FIT actions for {name} decision scale.')
        scales[name] = float(np.quantile(resolved, .25))
    return scales


def decision_research_metadata(response_labels, sensitivity, response_scales):
    """Compute the shared original-FIT research bands and label inventory once."""
    families = sorted({family for family, _ in response_labels})
    expected = {(family, action) for family in families for action in ACTIONS}
    if len(families) != 48 or set(response_labels) != expected:
        raise ValueError('Decision research metadata requires the complete original FIT48.')
    scales = fit_decision_scales(response_labels, families, sensitivity)
    radius = {name: 4 * float(sensitivity[name]) for name in FUNCTIONS}
    counts = {}
    for name in FUNCTIONS:
        values = np.asarray([float(response_labels[(family, action)][name])
                             for family in families for action in ACTIONS])
        feasible, infeasible = values + radius[name] <= 0, values - radius[name] > 0
        counts[name] = {'feasible': int(feasible.sum()), 'infeasible': int(infeasible.sum()),
                        'ambiguous': int((~feasible & ~infeasible).sum()),
                        'near_null': int((np.abs(values) <= radius[name]).sum())}
    return {'research_scales': scales, 'sensitivity': dict(sensitivity), 'radii': radius,
            'response_scales': {name: float(response_scales[name]['scale']
                               if isinstance(response_scales[name], dict) else response_scales[name])
                                for name in FUNCTIONS}, 'truth_counts': counts,
            'fit_families': len(families), 'fit_actions': len(expected)}


def decision_losses(predictions, physical_deltas, sensitivity, research_scales, response_scales):
    """Losses for one family; axes are [view, baseline+4 actions, J/P].

    Baseline deltas are identically zero. Ambiguous actions are omitted from
    binary feasibility and the resolved feasible ranking set. The returned
    losses average views and fixed requests, including zero-valued empty groups.
    """
    if predictions.ndim != 3 or tuple(predictions.shape[1:]) != (5, 2) or not predictions.shape[0]:
        raise ValueError('Decision pool requires one baseline and four actions per view.')
    if torch.is_tensor(physical_deltas):
        physical_deltas = physical_deltas.detach().cpu().numpy()
    y = np.asarray(physical_deltas, dtype=np.float64)
    if y.shape != (4, 2) or not np.isfinite(y).all():
        raise ValueError('Decision pool requires four finite physical J/P deltas.')
    if any(set(v) != set(FUNCTIONS) for v in (sensitivity, research_scales, response_scales)):
        raise ValueError('Decision metadata requires J and P.')
    radii = [4 * float(sensitivity[name]) for name in FUNCTIONS]
    m = [float(research_scales[name]) for name in FUNCTIONS]
    s = [float(response_scales[name]['scale'] if isinstance(response_scales[name], dict)
               else response_scales[name]) for name in FUNCTIONS]
    if any(not math.isfinite(v) or v <= 0 for v in (*radii, *m, *s)):
        raise ValueError('Decision scales and sensitivity must be positive and finite.')
    delta = predictions.to(torch.float64) - predictions[:, :1, :].to(torch.float64)
    observed_delta = delta.detach().cpu().numpy()
    truth = np.concatenate((np.zeros((1, 2), dtype=np.float64), y), axis=0)
    zero = delta.sum() * 0
    feasibility, ranking, nulls = [], [], []
    counts = {'requests': 0, 'feasible_actions': 0, 'infeasible_actions': 0,
              'ambiguous_actions': 0, 'ranking_pairs': 0, 'null_J': 0, 'null_P': 0,
              'active_feasible_hinges': 0, 'active_infeasible_hinges': 0,
              'active_ranking_hinges': 0, 'active_null_J_hinges': 0, 'active_null_P_hinges': 0}
    for view in range(predictions.shape[0]):
        for objective, constraint in ((0, 1), (1, 0)):
            for band in (0., 1., 2.):
                bound = band * m[constraint]
                observed = y[:, constraint]
                feasible = observed + radii[constraint] <= bound
                infeasible = observed - radii[constraint] > bound
                counts['requests'] += 1
                counts['feasible_actions'] += int(feasible.sum())
                counts['infeasible_actions'] += int(infeasible.sum())
                counts['ambiguous_actions'] += int((~feasible & ~infeasible).sum())
                margin = np.minimum(.25 * m[constraint], np.abs(observed - bound) - radii[constraint])
                observed_prediction = observed_delta[view, 1:, constraint]
                counts['active_feasible_hinges'] += int((feasible & (observed_prediction - bound + margin > 0)).sum())
                counts['active_infeasible_hinges'] += int((infeasible & (bound + margin - observed_prediction > 0)).sum())
                predicted = delta[view, 1:, constraint]
                classes = []
                for mask, sign in ((feasible, 1.), (infeasible, -1.)):
                    indices = np.flatnonzero(mask).tolist()
                    if indices:
                        residual = sign * (predicted[indices] - bound) + predicted.new_tensor(margin[indices])
                        classes.append(F.huber_loss(torch.relu(residual) / m[constraint],
                                                    torch.zeros_like(residual), reduction='mean'))
                feasibility.append(sum(classes) / len(classes) if classes else zero)
                eligible = [0] + [i + 1 for i in range(4) if bool(feasible[i])]
                pair_values, pair_weights = [], []
                for i in eligible:
                    for j in eligible:
                        ri = 0. if i == 0 else radii[objective]
                        rj = 0. if j == 0 else radii[objective]
                        gap = truth[j, objective] - truth[i, objective] - ri - rj
                        if float(gap) <= 0:
                            continue
                        d = min(.25 * m[objective], float(gap))
                        weight = min(1., float(gap) / s[objective])
                        counts['active_ranking_hinges'] += int(
                            observed_delta[view, i, objective] - observed_delta[view, j, objective] + d > 0)
                        hinge = torch.relu(delta[view, i, objective] - delta[view, j, objective] + d) / m[objective]
                        pair_values.append(F.huber_loss(hinge, torch.zeros_like(hinge), reduction='sum') * weight)
                        pair_weights.append(weight)
                counts['ranking_pairs'] += len(pair_values)
                ranking.append(sum(pair_values) / sum(pair_weights) if pair_values else zero)
        for index, name in enumerate(FUNCTIONS):
            mask = np.abs(y[:, index]) <= radii[index]
            counts[f'null_{name}'] += int(mask.sum())
            counts[f'active_null_{name}_hinges'] += int(
                (mask & (np.abs(observed_delta[view, 1:, index]) > radii[index])).sum())
            error = torch.relu(delta[view, 1:, index].abs() - radii[index]) / m[index]
            indices = np.flatnonzero(mask).tolist()
            nulls.append(F.huber_loss(error[indices], torch.zeros_like(error[indices]), reduction='mean')
                         if indices else zero)
    return ({'feasibility': sum(feasibility) / len(feasibility),
             'ranking': sum(ranking) / len(ranking),
             'near_null': sum(nulls) / len(nulls)}, counts)


def pooled_scalar_gradients(values, labels, sensitivity, research_scales, response_scales):
    """Differentiate the actual pooled loss only with respect to endpoint scalars."""
    leaves = values.detach().to(torch.float64).requires_grad_(True)
    losses, counts = decision_losses(leaves, labels, sensitivity, research_scales, response_scales)
    objective = sum(DECISION_WEIGHTS[name] * losses[name] for name in DECISION_WEIGHTS)
    gradient, = torch.autograd.grad(objective, leaves)
    return {name: value.detach() for name, value in losses.items()}, gradient.detach(), counts


def _rng_snapshot(device):
    device = torch.device(device)
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def _set_rng(snapshot, device):
    py, numpy, cpu, cuda = snapshot
    random.setstate(py)
    np.random.set_state(numpy)
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state(cuda, torch.device(device))


@contextmanager
def replay_rng(snapshot, device):
    """Replay the original draw without advancing any persistent RNG stream."""
    after_original = _rng_snapshot(device)
    try:
        _set_rng(snapshot, device)
        yield
    finally:
        _set_rng(after_original, device)


def replay_endpoint(forward, original, derivative, rng, device):
    """Replay one full-field forward and form its exact scalar-chain VJP."""
    with replay_rng(rng, device):
        values, auxiliary = forward()
    difference = (values.detach() - original).abs()
    return values, (values * derivative).sum(), difference, auxiliary


def _endpoint_functionals(prediction, batch, raw, tau):
    branches = ([prediction.full, prediction.restricted]
                if isinstance(prediction, ThermalJointPairPredictions) else [prediction])
    values = []
    for branch in branches:
        material = branch.native_main['pred_internal_temperature'][..., 0]
        pressure = branch.native_main['pred_field'][..., 2]
        prepared = branch.native_prepared
        material_mask = (prepared.stencils['material'].valid.reshape(material.shape)
                         & prepared.source_present[..., None].bool()
                         & torch.isfinite(batch.targets.material_targets))
        fluid_mask = (batch.targets.point_weights.bool()
                      & prepared.stencils['fluid'].valid.reshape(pressure.shape)
                      & torch.isfinite(batch.targets.field_targets[..., 2]))
        values.append(torch.stack(native_design_functionals(
            material, material_mask, pressure, fluid_mask, batch.receivers.fluid_xy[..., 0],
            raw['config']['domain']['lx'], tau)))
    return torch.stack(values)


class _DirectResponseProvider:
    """Use joint native fields for FIT endpoints, never baseline port targets."""
    def __init__(self, provider, local_epoch):
        self.provider, self.local_epoch = provider, local_epoch
        self.snapshots = []

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch=0, **kwargs):
        self.snapshots.append(_rng_snapshot(self.provider.device))
        return self.provider.predict_fit_native(model, scene, receivers, execution_mode, phase,
                                                local_epoch=self.local_epoch, **kwargs)


def _parameter_groups(model):
    groups = {'context_values': [], 'relation_calibration': [], 'joint_trunk': []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        route = '.context_blocks.' in name and any(f'.{r}.' in name for r in ('mm', 'em', 'ee', 'me'))
        if route and ('.value.' in name or '.output.' in name):
            groups['context_values'].append((name, parameter))
        elif ((route and not ('.value.' in name or '.output.' in name))
              or any(f'.member_router.{key}.' in name for key in ('proposal_M', 'member_M', 'member_E'))
              or 'calibration' in name):
            groups['relation_calibration'].append((name, parameter))
        elif '.member_router.joint_trunk.' in name or '.member_router.joint_head.' in name:
            groups['joint_trunk'].append((name, parameter))
    return groups


def _accumulate_gradients(store, term, groups):
    named = [pair for pairs in groups.values() for pair in pairs]
    if not named or not term.requires_grad:
        return
    grad = torch.autograd.grad(term, [p for _, p in named], retain_graph=True, allow_unused=True)
    for (name, _), value in zip(named, grad):
        if value is None:
            continue
        value = value.detach()
        store[name] = value.clone() if name not in store else store[name] + value


def _gradient_report(buffers, groups):
    report = {}
    for group, named in groups.items():
        names = [name for name, _ in named]
        norms = {}
        for term, values in buffers.items():
            norms[term] = math.sqrt(sum(float(values[n].double().square().sum())
                                        for n in names if n in values))
        cosines = {}
        for first, second in (('endpoint', 'delta'), ('endpoint', 'decision'), ('delta', 'decision')):
            den = norms[first] * norms[second]
            dot = sum(float((buffers[first][n].double() * buffers[second][n].double()).sum())
                      for n in names if n in buffers[first] and n in buffers[second])
            cosines[f'{first}_vs_{second}'] = dot / den if den else None
        report[group] = {'parameter_names': names, 'norms': norms, 'cosines': cosines}
    return report


class ThermalR165DecisionTask(ThermalPortResponseTask, ThermalStateResponseTask):
    """P64 native supervision plus original FIT, optionally decision penalties."""
    auxiliary_loss_names = ThermalStateResponseTask.auxiliary_loss_names + tuple(
        f'decision/{name}' for name in DECISION_WEIGHTS)

    def _validate_initialization(self, initialization):
        if (self.source_kind != 'thermal_r16_5_p64_weight_only_v1'
                or self.source_parent_age != 5100 or initialization.get('pretrained_weights') is not True
                or initialization.get('source_arm') != ARM_DETAIL
                or not initialization.get('source_checkpoint')):
            raise ValueError('R16-5 requires its declared P64 weight-only initialization.')

    def configure_decision(self, spec):
        if (spec.get('schema') != 'thermal_r16_5_decision_v1'
                or spec.get('arm') not in ('PDelta', 'PDeltaD')):
            raise ValueError('Expected the R16-5 PDelta or PDeltaD decision recipe.')
        if self.response_finite_response_coefficient != .1 or self.response_pilot is not None:
            raise ValueError('R16-5 requires the original common FIT response objective.')
        self.decision_spec = copy.deepcopy(spec)
        self.decision_sensitivity = {name: float(spec['sensitivity'][name]) for name in FUNCTIONS}
        self.decision_scales = {name: float(spec['research_scales'][name]) for name in FUNCTIONS}
        if any(not math.isfinite(value) or value <= 0 for value in
               (*self.decision_sensitivity.values(), *self.decision_scales.values())):
            raise ValueError('Saved FIT decision sensitivities and research scales must be positive.')
        self._decision_epoch = None
        self._decision_counts = {}
        self._decision_replay_difference = {}
        self._snapshot_epoch = None
        self._snapshot_buffers = None
        self._clock_epoch = None

    def _record_clock(self, kind, local_epoch):
        if not hasattr(self, 'decision_spec'):
            return
        local_epoch = int(local_epoch)
        if self._clock_epoch != local_epoch:
            self._clock_epoch, self._clock_observations = local_epoch, {}
        row = self._clock_observations.setdefault(kind, {'actual_epochs': set(), 'count': 0})
        row['actual_epochs'].add(int(self.model.core._joint_epoch))
        row['count'] += 1

    def make_batch(self, case_keys, key):
        # The engine's SamplingKey already contains 5100 + local epoch.
        batch = super(ThermalPortResponseTask, self).make_batch(case_keys, key)
        self.model.core.set_training_clock(int(key.epoch), key.update_index)
        self._record_clock('native_batch', int(key.epoch) - self.source_parent_age)
        return self._with_port_targets(batch)

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch=0, **kwargs):
        prediction = super().predict_native(model, scene, receivers, execution_mode, phase,
                                            epoch=epoch, **kwargs)
        if model.training:
            self._record_clock('native_forward', epoch)
        return prediction

    def fit_sampler_epoch(self, local_epoch):
        return self.source_parent_age + ThermalStateResponseTask.fit_sampler_epoch(self, local_epoch)

    def predict_fit_native(self, model, scene, receivers, execution_mode, phase, *, local_epoch,
                           replay=False, **kwargs):
        model.core.set_training_clock(self.source_parent_age + int(local_epoch),
                                      int(getattr(model.core, '_joint_update_index', 0)))
        prediction = _PairedMemberRoutingTask.predict_native(self, model, scene, receivers,
                                                             execution_mode, phase, **kwargs)
        if model.training:
            self._record_clock('fit_replay' if replay else 'fit_forward', local_epoch)
        return prediction

    def _want_snapshot(self, epoch, update_index, update_count):
        return epoch in (1, 10) and update_count is not None and update_index == 0

    def _prepare_snapshot(self, model, epoch):
        if self._snapshot_epoch != epoch:
            self._snapshot_epoch = epoch
            self._snapshot_groups = _parameter_groups(model)
            self._snapshot_buffers = {name: {} for name in ('endpoint', 'delta', 'decision', 'native', 'port')}

    def diagnose_native_training_terms(self, model, terms, planned, *, epoch, update_index, update_count):
        if not self._want_snapshot(epoch, update_index, update_count):
            return
        self._prepare_snapshot(model, epoch)
        for label, accept in (('native', lambda name: name.startswith('final/')),
                              ('port', lambda name: not name.startswith('final/'))):
            pieces = [term.numerator * (float(term.weight) / planned[name])
                      for name, term in terms.items() if accept(name)]
            if pieces:
                _accumulate_gradients(self._snapshot_buffers[label], sum(pieces), self._snapshot_groups)

    def auxiliary_update_terms(self, model, *, epoch, update_index, update_count):
        if update_count < 1:
            raise ValueError('Auxiliary FIT requires a positive native update count.')
        if self._decision_epoch != epoch:
            self._decision_epoch, self._decision_counts, self._decision_replay_difference = epoch, {}, {}
        families = self.response_corpus.fit_families_for_epoch(self.fit_sampler_epoch(epoch), self.seed)
        slots = np.linspace(0, update_count - 1, 4, dtype=int)
        for family_index, (family, slot) in enumerate(zip(families, slots)):
            if int(slot) != int(update_index):
                continue
            snapshot = self._want_snapshot(epoch, update_index, update_count) and family_index == 0
            if snapshot:
                self._prepare_snapshot(model, epoch)
                self._snapshot_family = str(family)
            endpoints, scalars, labels = [], [], []
            for action in ACTIONS:
                pair = response_pair(self, self.response_corpus, family, action, role='fit')
                direct = _DirectResponseProvider(self, epoch)
                losses = predict_response_pair(direct, model, pair, self.response_scales,
                                                self.response_labels[(family, action)],
                                                self.response_spec['tau'])
                terms = response_loss_terms(losses['endpoint'], losses['field_delta'],
                    {'J': losses['J_delta'], 'P': losses['P_delta']}, multiplier=update_count / 16,
                    finite_response_coefficient=.1)
                if snapshot:
                    _accumulate_gradients(self._snapshot_buffers['endpoint'],
                                          .25 * losses['endpoint'] / 4, self._snapshot_groups)
                    delta = (.08 * losses['field_delta'] + .01 * losses['J_delta'] + .01 * losses['P_delta']) / 4
                    _accumulate_gradients(self._snapshot_buffers['delta'], delta, self._snapshot_groups)
                if self.decision_spec['arm'] == 'PDeltaD':
                    observations = losses['observations']
                    current = [torch.stack([torch.stack(row['functionals'][i]) for row in observations]).detach()
                               for i in (0, 1)]
                    if not endpoints:
                        endpoints.append((pair['baseline'], {'config': {'domain': {
                            'lx': pair['raw_baseline']['config']['domain']['lx']}}}, direct.snapshots[0]))
                        scalars.append(current[0])
                    endpoints.append((pair['edited'], {'config': {'domain': {
                        'lx': pair['raw_edited']['config']['domain']['lx']}}}, direct.snapshots[1]))
                    scalars.append(current[1])
                    labels.append([self.response_labels[(family, action)][name] for name in FUNCTIONS])
                    del observations, current
                work = dict(losses['work'])
                work.update(response_actions=1, response_families=int(action == ACTIONS[0]))
                self._record_response_action(epoch, family, action, 'original_fit')
                if action == ACTIONS[0] and family not in self._response_epoch_families:
                    self._response_epoch_families.append(family)
                yield terms, work
                del losses, terms, work, direct, pair
            if self.decision_spec['arm'] != 'PDeltaD':
                continue
            values = torch.stack(scalars, dim=1)
            pooled, derivatives, counts = pooled_scalar_gradients(
                values, labels, self.decision_sensitivity, self.decision_scales,
                {name: self.response_scales[name] for name in FUNCTIONS})
            for name, count in counts.items():
                self._decision_counts[name] = self._decision_counts.get(name, 0) + count
            self._decision_counts['families'] = self._decision_counts.get('families', 0) + 1
            for index, (batch, raw, rng) in enumerate(endpoints):
                def forward():
                    prediction, auxiliary = self.predict_fit_native(model, batch.scene_inputs, batch.receivers,
                        model.architecture, 'joint', local_epoch=epoch, replay=True)
                    return _endpoint_functionals(prediction, batch, raw, self.response_spec['tau']), auxiliary
                replay, surrogate, difference, auxiliary = replay_endpoint(
                    forward, values[:, index], derivatives[:, index], rng, self.device)
                for view in range(difference.shape[0]):
                    for channel, name in enumerate(FUNCTIONS):
                        key = f'view{view}/{name}'
                        self._decision_replay_difference[key] = max(
                            self._decision_replay_difference.get(key, 0.), float(difference[view, channel]))
                if snapshot:
                    _accumulate_gradients(self._snapshot_buffers['decision'], surrogate, self._snapshot_groups)
                terms = {}
                for name, weight in DECISION_WEIGHTS.items():
                    actual = pooled[name] if index == 0 else pooled[name] * 0
                    if name == 'feasibility':
                        actual = actual + (surrogate - surrogate.detach()) / weight
                    terms[f'decision/{name}'] = LossTerm(actual, 1, weight * update_count / 4)
                work = {'decision_replayed_endpoints': 1}
                work.update({f'decision_replay_{name}': value for name, value in auxiliary.get('work', {}).items()})
                yield terms, work
                del replay, surrogate, terms, work, auxiliary

    def auxiliary_diagnostic_values(self, terms, *, update_count):
        result = {}
        for name, term in terms.items():
            if name.startswith('decision/'):
                result[name] = float(term.numerator.detach()) * update_count / 4
            elif not name.endswith('/endpoint'):
                result[name] = float(term.numerator.detach()) * update_count / 16
        return result

    def loss_metadata(self):
        result = ThermalPortResponseTask.loss_metadata(self)
        result.update(ThermalStateResponseTask.loss_metadata(self))
        for name, weight in DECISION_WEIGHTS.items():
            result[f'decision/{name}'] = {
                'panel': 'physics', 'label': f'pooled decision {name}', 'unit': 'Huber loss',
                'formula': 'equal FIT family, view and request means; full native field functionals',
                'weight': str(weight if self.decision_spec['arm'] == 'PDeltaD' else 0.)}
        return result

    def identity_payload(self):
        result = ThermalPortResponseTask.identity_payload(self)
        result['normalization_fit_scope'] = 'inherited source TRAIN600; no refit'
        result['loss_lineage']['history_epoch_basis'] = 'local; source model age inherited separately'
        result['r16_5_decision'] = {**copy.deepcopy(self.decision_spec),
            'research_scales': dict(self.decision_scales), 'radius': {k: 4*v for k, v in self.decision_sensitivity.items()},
            'scale_fit': 'Q25 absolute resolved original FIT48 action responses',
            'baseline': 'one canonical baseline per family and view',
            'sampler_and_core_clock': '5100 + local epoch',
            'replay': 'one full-field endpoint scalar VJP at a time; original RNG restored'}
        result['r16_5_source_tensor_inventory'] = copy.deepcopy(
            getattr(self, 'r16_5_source_tensor_inventory', {}))
        if hasattr(self, 'r16_5_execution'):
            result['r16_5_execution'] = copy.deepcopy(self.r16_5_execution)
        return result

    def completed_training_epoch(self, model, engine, epoch, output, row):
        super().completed_training_epoch(model, engine, epoch, output, row)
        row['decision_training'] = {'arm': self.decision_spec['arm'], 'counts': dict(self._decision_counts),
                                    'replay_max_abs_difference': dict(self._decision_replay_difference)}
        if self._clock_epoch == epoch:
            row['decision_training']['clocks'] = {
                'source_age': self.source_parent_age, 'local_epoch': epoch,
                'paths': {name: {'actual_epochs': sorted(values['actual_epochs']), 'count': values['count']}
                          for name, values in self._clock_observations.items()}}
        if self._snapshot_epoch == epoch and self._snapshot_buffers is not None:
            row['decision_training']['gradient_snapshot'] = {
                'family_id': self._snapshot_family,
                'selection': 'first input-selected ordinary FIT family of this epoch',
                'scope': 'family endpoint/delta/decision, separate native and port gradients for its native update',
                'groups': _gradient_report(self._snapshot_buffers, self._snapshot_groups)}


def qualify_full_field_replay(provider, model, family, *, local_epoch=1, checkpoint_retained=False):
    """Read-only retained-graph versus scalar-VJP qualification on one FIT family.

    Both full and restricted views use all native field receivers and their
    own endpoint masks. No optimizer or parameter gradients are changed. The
    optional non-reentrant checkpoint bounds the direct graph's activation
    memory while preserving its forward RNG state.
    """
    import time
    from torch.utils.checkpoint import checkpoint

    if family not in provider.response_corpus.family_ids('fit'):
        raise ValueError('Replay qualification accepts only an original FIT family.')
    if type(local_epoch) is not int or local_epoch < 1:
        raise ValueError('Replay qualification requires a positive local epoch.')
    device = torch.device(provider.device)
    prior_rng = _rng_snapshot(device)
    original_mode = model.training
    old_clock = (int(model.core._joint_epoch), int(model.core._joint_update_index))
    old_clock_epoch = getattr(provider, '_clock_epoch', None)
    old_observations = copy.deepcopy(getattr(provider, '_clock_observations', {}))
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not params:
        raise ValueError('Replay qualification needs trainable model parameters.')
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    try:
        model.train(True)
        endpoints, labels = [], []
        for action in ACTIONS:
            pair = response_pair(provider, provider.response_corpus, family, action, role='fit')
            if not endpoints:
                endpoints.append((pair['baseline'], {'config': {'domain': {
                    'lx': pair['raw_baseline']['config']['domain']['lx']}}}))
            endpoints.append((pair['edited'], {'config': {'domain': {
                'lx': pair['raw_edited']['config']['domain']['lx']}}}))
            labels.append([provider.response_labels[(family, action)][name] for name in FUNCTIONS])
        snapshots, outputs = [], []
        for batch, raw in endpoints:
            snapshots.append(_rng_snapshot(device))
            def endpoint_forward(batch=batch, raw=raw):
                prediction, _ = provider.predict_fit_native(model, batch.scene_inputs, batch.receivers,
                    model.architecture, 'joint', local_epoch=local_epoch)
                return _endpoint_functionals(prediction, batch, raw, provider.response_spec['tau'])
            output = (checkpoint(endpoint_forward, use_reentrant=False, preserve_rng_state=True)
                      if checkpoint_retained else endpoint_forward())
            outputs.append(output)
        originals = torch.stack(outputs, dim=1)
        if originals.shape[0] != 2:
            raise ValueError('R16-5 qualification requires the full and P95 views.')
        scalar_values = originals.detach()
        s = {name: provider.response_scales[name] for name in FUNCTIONS}
        direct_gradients, derivatives, actual_losses = [], [], []
        for view in range(2):
            losses, _ = decision_losses(originals[view:view + 1], labels, provider.decision_sensitivity,
                                         provider.decision_scales, s)
            actual_losses.append({name: float(value.detach()) for name, value in losses.items()})
            objective = sum(DECISION_WEIGHTS[name] * losses[name] for name in DECISION_WEIGHTS)
            gradients = torch.autograd.grad(objective, [p for _, p in params], retain_graph=view == 0,
                                            allow_unused=True)
            direct_gradients.append([None if g is None else g.detach().cpu().double() for g in gradients])
            _, deriv, _ = pooled_scalar_gradients(scalar_values[view:view + 1], labels,
                provider.decision_sensitivity, provider.decision_scales, s)
            derivatives.append(deriv[0])
        del originals, outputs
        replay_gradients = [[None] * len(params) for _ in range(2)]
        max_differences = torch.zeros((2, 2), dtype=torch.float64)
        for index, ((batch, raw), rng) in enumerate(zip(endpoints, snapshots)):
            def replay_forward(batch=batch, raw=raw):
                prediction, aux = provider.predict_fit_native(model, batch.scene_inputs, batch.receivers,
                    model.architecture, 'joint', local_epoch=local_epoch, replay=True)
                return _endpoint_functionals(prediction, batch, raw, provider.response_spec['tau']), aux
            with replay_rng(rng, device):
                values, _ = replay_forward()
            max_differences = torch.maximum(max_differences,
                (values.detach() - scalar_values[:, index]).abs().cpu().double())
            for view in range(2):
                surrogate = (values[view] * derivatives[view][index]).sum()
                gradients = torch.autograd.grad(surrogate, [p for _, p in params],
                                                retain_graph=view == 0, allow_unused=True)
                for position, gradient in enumerate(gradients):
                    if gradient is None:
                        continue
                    value = gradient.detach().cpu().double()
                    previous = replay_gradients[view][position]
                    replay_gradients[view][position] = value if previous is None else previous + value
            del values
        view_rows = []
        for view in range(2):
            direct, replay = direct_gradients[view], replay_gradients[view]
            direct_norm2 = replay_norm2 = difference_norm2 = max_abs = dot = 0.
            for a, b in zip(direct, replay):
                if a is None and b is None:
                    continue
                if a is None:
                    a = torch.zeros_like(b)
                if b is None:
                    b = torch.zeros_like(a)
                difference = a - b
                direct_norm2 += float(a.square().sum())
                replay_norm2 += float(b.square().sum())
                difference_norm2 += float(difference.square().sum())
                dot += float((a * b).sum())
                max_abs = max(max_abs, float(difference.abs().max()))
            direct_norm, replay_norm = math.sqrt(direct_norm2), math.sqrt(replay_norm2)
            view_rows.append({'view': ('full', 'P95')[view], 'losses': actual_losses[view],
                'max_scalar_abs_difference': dict(zip(FUNCTIONS, max_differences[view].tolist())),
                'direct_gradient_norm': direct_norm, 'replay_gradient_norm': replay_norm,
                'gradient_difference_norm': math.sqrt(difference_norm2),
                'gradient_relative_difference': math.sqrt(difference_norm2) / direct_norm if direct_norm else None,
                'gradient_max_abs_difference': max_abs,
                'gradient_cosine': dot / (direct_norm * replay_norm) if direct_norm and replay_norm else None})
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        return {'family_id': str(family), 'local_epoch': local_epoch,
            'source_age': provider.source_parent_age, 'actual_core_epoch': provider.source_parent_age + local_epoch,
            'checkpoint_retained_graph': bool(checkpoint_retained), 'full_native_fields': True,
            'views': view_rows, 'trainable_parameter_count': len(params),
            'wall_seconds': time.perf_counter() - wall_start,
            'process_cpu_seconds': time.process_time() - cpu_start,
            'child_cpu_seconds': 0.,
            'peak_device_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}
    finally:
        _set_rng(prior_rng, device)
        model.train(original_mode)
        model.core.set_training_clock(*old_clock)
        provider._clock_epoch, provider._clock_observations = old_clock_epoch, old_observations

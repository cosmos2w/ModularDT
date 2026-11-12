"""Thermal physical endpoint-family losses with case-neutral reductions."""
from __future__ import annotations

from dataclasses import fields, replace
from types import SimpleNamespace

import torch

from honf_runtime.state_response import (
    ResponseObjectiveConfig,
    combine_response_losses,
    equal_mean,
    normalized_delta_mse,
    normalized_response_mse,
    weighted_mean,
)

from ..native_functionals import native_design_functionals
from .joint_member_routing_task import ThermalJointPairPredictions
from .thermal_state_response_task import FIELD_ROLES, _field_components


def _endpoint_functionals(prediction, batch, raw, tau):
    """Return this endpoint's Thermal J/P using its own physical support."""
    branches = _views(prediction)
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
        raise TypeError('Batched family execution requires paired access views.')
    return [ThermalJointPairPredictions(branch_slice(prediction.full, i),
              branch_slice(prediction.restricted, i), {}) for i in range(count)]


class ThermalFamilyObjectiveMixin:
    def family_components(self, model, source, family, *, batched=True, components=None):
        """Evaluate selected losses; calibration may explicitly request all three."""
        names = frozenset(('endpoint', 'response', 'decision') if components is None else components)
        if not names or not names <= {'endpoint', 'response', 'decision'}:
            raise ValueError('Unknown or empty family objective components.')
        if names & {'response', 'decision'} and self.response_scales is None:
            raise ValueError('Family response scales must be configured.')
        if 'decision' in names and (self.sensitivity is None or self.research_scales is None):
            raise ValueError('Family decision scales must be configured.')
        actions = self.actions(source, family)
        pairs = [self.pair(source, family, action) for action in actions]
        if not pairs:
            raise ValueError('A family needs at least one physical action.')
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
            raise ValueError('Thermal family fitting requires both full and restricted joint views.')
        result = {}
        if 'endpoint' in names:
            result['endpoint'] = equal_mean(self.provider.endpoint_state_loss(p, b)
                                            for p, b in zip(predictions, batches))
        if names & {'response', 'decision'}:
            # Receiver identities/order and physical controls are checked when
            # the case-owned corpus constructs each native endpoint pair.
            functions = torch.stack([_endpoint_functionals(pred, batch, raw, self.tau)
                                     for pred, batch, raw in zip(predictions, batches, raws)], dim=1)
            labels = [self.label(source, family, action) for action in actions]
        if 'response' in names:
            fields = [[_field_components(pred, batch) for pred in row] for row, batch in zip(views, batches)]
            field_losses, functional_losses = [], {'J': [], 'P': []}
            for action_index, label in enumerate(labels, start=1):
                for view in range(2):
                    terms, weights = [], []
                    for name in FIELD_ROLES:
                        bp, bt, bm = fields[0][view][name]
                        ep, et, em = fields[action_index][view][name]
                        if bp.shape != ep.shape or bt.shape != et.shape:
                            raise ValueError('Response receiver identity or shape changed.')
                        mask = bm & em
                        if not bool(mask.any()):
                            raise ValueError(f'No common support for {name}.')
                        terms.append(normalized_response_mse(bp, ep, bt, et, mask,
                            self.response_scales[name], group_dim=-1 if bp.ndim != 2 else None))
                        weights.append(float(self.provider.native_weights[name]))
                    field_losses.append(weighted_mean(terms, weights))
                    for index, name in enumerate(('J', 'P')):
                        functional_losses[name].append(normalized_delta_mse(
                            functions[view, action_index, index] - functions[view, 0, index],
                            float(label[name]), self.response_scales[name]))
            config = getattr(self, 'response_objective_config', None) or ResponseObjectiveConfig(
                field_response_weight=.8, functional_weights={'J': .1, 'P': .1})
            result['response'] = combine_response_losses(equal_mean(field_losses),
                {name: equal_mean(values) for name, values in functional_losses.items()}, config)
        if 'decision' in names:
            truth = [[float(label[n]) for n in ('J', 'P')] for label in labels]
            from .r16_5_decision_task import decision_losses
            decision, counts = decision_losses(functions, truth, self.sensitivity, self.research_scales,
                                               {n: self.response_scales[n] for n in ('J', 'P')})
            result['decision'] = .4 * decision['feasibility'] + .4 * decision['ranking'] + .2 * decision['near_null']
            work.update({f'decision_{key}': value for key, value in counts.items()})
        return result, work

"""Small coupled analytic observations exercising the public training path.

This is a software fixture, not evidence about a physical case or learning
quality. The input records, target functions and missing values are generated
in memory; no dataset or edited labels are written.
"""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import torch
from torch import nn

from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2, QueryBatch, SceneBatch
from honf_forward_core.models import build_model, relation_anchored_config
from .observation_blocks import FieldComponent, LinearExtraction, ObservationBlock, QueryDependency
from .unified_training import LossTerm, OptimizerGroupSpec, SamplingKey, ScheduleSpec, TaskBatch, _sampling_dataset_id


_TRAIN = tuple(f'fixture_{i}' for i in range(6))
_VALIDATION = ('fixture_6', 'fixture_7')
_PAIRS = (('fixture_0', 'fixture_1'), ('fixture_2', 'fixture_3'), ('fixture_4', 'fixture_5'))


def _number(case_id):
    return int(str(case_id).removeprefix('fixture_'))


def _coordinates(case_id, count=None):
    n = _number(case_id)
    count = 4 + n % 3 if count is None else count
    i = np.arange(count, dtype=np.float64)
    return np.stack((.1 + .11 * i, .15 + .06 * (i % 3), .2 + .05 * (i % 2)), axis=-1)


def _reference(case_id, coordinates, width):
    n = _number(case_id)
    q = np.asarray(coordinates, dtype=np.float64)
    a, b, c = q.T
    control = .15 * (n + 1)
    # Every output depends jointly on the scene and all three coordinates.
    return np.stack([np.sin((k + 1) * a + control * b) + (k + 1) * .1 * b * c
                     + control * (a + c) + .03 * (k + 1) * n for k in range(width)], -1)


class AnalyticJointModel(nn.Module):
    """The maintained joint original-member HONF predictor with a tiny head."""
    architecture = 'relation_anchored'

    def __init__(self, width, seed):
        super().__init__()
        base = ContentHypergraphCoreV2(module_input_dim=2, environment_input_dim=2,
            coordinate_dim=3, query_input_dim=0, operating_input_dim=2, output_dim=width,
            architecture='A2-D', context_dim=8, num_heads=2, context_depth=2,
            local_hidden_dim=8, dropout=0., init_seed=seed).get_config()
        config = relation_anchored_config('relation_anchored', base, input_dataset='native_records', query_tile_size=32)
        self.core = build_model('relation_anchored', config)

    def scene(self, case_id):
        parameter = next(self.parameters())
        n = _number(case_id)
        count = 1 + n % 3
        m = torch.arange(count, device=parameter.device, dtype=parameter.dtype)
        mc = torch.stack((.2 + m * .16, .25 + m * .05, .3 + m * .04), -1)[None]
        mf = torch.stack((m + 1, m * .1 + .1 * n), -1)[None]
        # Original E records are retained individually; coverage chooses one
        # permitted E record at each query for this deliberately small fixture.
        ec = parameter.new_tensor([[.2,.2,.2], [.8,.2,.2], [.2,.8,.2], [.8,.8,.2],
                                   [.2,.2,.8], [.8,.2,.8], [.2,.8,.8], [.8,.8,.8]])[None]
        ef = torch.stack((ec[..., 0] + ec[..., 1], ec[..., 2] + .03 * n), -1)
        return SceneBatch(module_features=mf, environment_features=ef, module_coordinates=mc,
            environment_coordinates=ec, module_mask=torch.ones((1,count), dtype=torch.bool, device=parameter.device),
            environment_mask=torch.ones((1,8), dtype=torch.bool, device=parameter.device),
            module_measure=parameter.new_ones((1,count)), environment_measure=parameter.new_full((1,8), .125),
            module_ids=torch.arange(10, 10+count, device=parameter.device)[None],
            environment_ids=torch.arange(100,108,device=parameter.device)[None],
            local_radius=parameter.new_full((1,count), .5), operating_features=parameter.new_tensor([[n/8, .1*n]]),
            coordinate_origin=parameter.new_zeros((1,3)), coordinate_scale=parameter.new_ones((1,3)),
            module_measure_reference=parameter.new_ones(1), environment_measure_reference=parameter.new_ones(1),
            scene_key=str(case_id), normalizer_key='analytic_fixture_identity', coordinate_frame_key='analytic_fixture_xyz',
            module_characteristic_lengths=parameter.new_full((1,count,3), .2),
            environment_characteristic_lengths=parameter.new_full((1,8,3), .5))

    def field_request(self, case_id, coordinates):
        scene = self.scene(case_id)
        q = torch.as_tensor(coordinates, dtype=scene.module_features.dtype, device=scene.module_features.device)[None]
        nearest = torch.cdist(q, scene.environment_coordinates).argmin(-1)
        query = QueryBatch(coordinates=q, query_mask=torch.ones(q.shape[:2], dtype=torch.bool, device=q.device),
            query_ids=torch.arange(q.shape[1], device=q.device)[None], scene_key=scene.scene_key,
            coordinate_frame_key=scene.coordinate_frame_key, anchor_indices=nearest[...,None],
            anchor_weights=torch.ones((*q.shape[:2],1), dtype=q.dtype, device=q.device))
        return scene, query

    def predict(self, case_id, coordinates, *, read_mode=None):
        scene, query = self.field_request(case_id, coordinates)
        organization = self.core.prepare(scene)
        if read_mode is None:
            full, restricted, shared = self.core.decode_paired(organization, query)
            fields = torch.stack((full.joint_fields[0], restricted.joint_fields[0]))
        else:
            with self.core.read_execution(read_mode):
                output = self.core.decode(organization, query)
            fields, shared = output.joint_fields, output.work_counts
        work = {'neural_queries': int(query.coordinates.shape[1]), 'scene_preparations': 1}
        for key, value in shared.items():
            if isinstance(value, (int, float)):
                work[f'paired_{key}'] = int(value)
        return fields, work


class AnalyticNativeProvider:
    def __init__(self, recipe, model, device):
        self.recipe, self.model, self.device = copy.deepcopy(recipe), model, torch.device(device)
        self.width = int(recipe.get('fixture', {}).get('output_dim', 7))
        self.seed = int(recipe.get('sampling_seed', recipe['seed']))
        self.sampling_version = SamplingKey.CASE_EPOCH_VERSION
        self.sampling_dataset_id = _sampling_dataset_id(self.identity_payload())
        self.training_samples = self.validation_samples = 0

    def identity_payload(self):
        return {'dataset': 'analytic_fixture', 'training_case_ids': list(_TRAIN),
                'validation_case_ids': list(_VALIDATION), 'output_dim': self.width,
                'model': self.model.core.get_config(), 'reference': 'coupled_analytic_fixture_v1'}

    def epoch_cases(self, epoch, seed):
        del epoch
        if int(seed) != self.seed:
            raise ValueError('Fixture sampling seed differs from its recipe.')
        return _TRAIN

    def make_batch(self, case_keys, key):
        keys = tuple(map(str, case_keys))
        sampler = getattr(self, 'sample_coupled_native_rows', None)
        overrides, carrier = sampler(keys, key) if callable(sampler) else ({}, None)
        self.training_samples += len(keys)
        coordinates = tuple(_coordinates(c)[overrides[c]] if c in overrides else _coordinates(c) for c in keys)
        rows = [torch.as_tensor(_reference(c, q, self.width), device=self.device, dtype=torch.float32)
                for c, q in zip(keys, coordinates)]
        return TaskBatch(scene_inputs=keys, receivers=coordinates, targets=tuple(rows),
                         auxiliary={'coupled_native': carrier} if carrier is not None else {}, case_keys=keys)

    def make_scene(self, scene_inputs):
        return scene_inputs

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch, temperature):
        del phase, epoch, temperature
        if model is not self.model or execution_mode != model.architecture:
            raise ValueError('Fixture requires its own joint predictor.')
        selected = getattr(self, 'coupled_native_reader_mode', None) if model.training else None
        predicted = [model.predict(case, q, read_mode=selected)[0] for case, q in zip(scene, receivers)]
        return tuple(predicted), None

    def loss_denominators(self, batches, phase, arm):
        del phase, arm
        return {'native': sum(t.numel() for batch in batches for t in batch.targets)}

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        del phase, auxiliary_state
        value = sum((p - t[None]).square().mean(0).sum() for p, t in zip(predictions, targets))
        return {'native': LossTerm(value, sum(t.numel() for t in targets))}

    def validation_batches(self):
        key = SamplingKey(self.seed, 1, 0, 0, 'joint', self.model.architecture)
        for case in _VALIDATION:
            self.validation_samples += 1
            count = self.training_samples
            batch = self.make_batch((case,), key)
            self.training_samples = count
            yield batch

    def validation_loss_terms(self, predictions, targets, auxiliary_state, *, batch, arm):
        del batch, arm
        return self.loss_terms(predictions, targets, 'joint', auxiliary_state)

    def validation_metrics(self, predictions, targets, auxiliary_state):
        del auxiliary_state
        return {'squared_error': sum(float((p[-1] - t).square().sum()) for p, t in zip(predictions, targets)),
                'count': sum(t.numel() for t in targets)}

    def reduce_native_metrics(self, rows):
        return {'field_score': sum(r['squared_error'] for r in rows) / sum(r['count'] for r in rows)}

    def optimizer_groups(self, model, arm, stage):
        if model is not self.model or arm != model.architecture or stage != 'joint':
            raise ValueError('Fixture optimizer requires its joint model.')
        horizon = max(int(self.recipe['total_epochs']) + 1, 2)
        return (OptimizerGroupSpec('joint', tuple(n for n, p in model.named_parameters() if p.requires_grad),
            ScheduleSpec(peak_lr=.001, warmup_start_lr=.001, hold_through_epoch=horizon-1,
                         total_epochs=horizon, final_lr=.001), weight_decay=0.),)

    def work_counts(self, batch, predictions, auxiliary_state):
        del predictions, auxiliary_state
        return {'native_queries': sum(len(x) for x in batch.receivers)}

    def loss_metadata(self):
        return {'native': {'panel': 'physics', 'label': 'native fixture fields',
                           'unit': 'squared fixture units', 'formula': 'equal native observed elements'}}

    def training_history_columns(self):
        return ('train_losses.native', 'work_counts.native_queries', 'work_counts.gradient_norm')

    def training_state_dict(self):
        return {'training_samples': self.training_samples, 'validation_samples': self.validation_samples}

    def load_training_state_dict(self, state):
        self.training_samples = int(state['training_samples'])
        self.validation_samples = int(state['validation_samples'])


class AnalyticObservationBinding:
    def __init__(self, provider):
        self.provider = provider

    def identity_payload(self):
        return {'schema': 'analytic_coupled_observations_v1', 'output_dim': self.provider.width,
                'block_names': list(self.block_names()),
                'coordinates': 'three-dimensional ragged source coordinates', 'units': 'fixture units'}

    def block_names(self):
        return ('fixture_common_pair','fixture_common_scene_0_only','fixture_common_scene_1_only',
                'fixture_individual_scene_0','fixture_individual_scene_1')

    def training_pairs(self): return _PAIRS
    def validation_pairs(self): return (tuple(_VALIDATION),)
    def calibration_case_ids(self): return _TRAIN
    def native_case_key(self, case_id): return case_id

    def coupled_native_population(self, case_id):
        from .coupled_observations import NativeObservationPopulation, ObservationKey

        xy = _coordinates(case_id)
        width = self.provider.width
        components = tuple(FieldComponent(f'field_{i}', 'fixture units', i, 1., 1.) for i in range(width))
        keys = tuple(ObservationKey('fixture_frame', 0, tuple(p), 'fixed_frame', 'direct') for p in xy)
        dependencies = tuple(tuple((tuple(p),) for _ in components) for p in xy)
        shape = (len(xy), width)
        return NativeObservationPopulation(keys, components, _reference(case_id, xy, width),
                                           np.ones(shape), np.ones(shape, dtype=bool),
                                           np.ones(shape, dtype=bool), dependencies)

    def coupled_native_budget(self, case_id):
        return len(_coordinates(case_id))

    def coupled_native_sampling_law(self, case_id):
        del case_id
        return 'uniform_without_replacement'

    def coupled_view_loss_names(self):
        return ('native',)

    def coupled_validation_batch(self, pair, sample):
        coordinates = (_coordinates(pair[0])[sample.left_rows], _coordinates(pair[1])[sample.right_rows])
        targets = tuple(torch.as_tensor(_reference(case, q, self.provider.width),
                                        device=self.provider.device, dtype=torch.float32)
                        for case, q in zip(pair, coordinates))
        return TaskBatch(scene_inputs=pair, receivers=coordinates, targets=targets,
                         auxiliary={'coupled_native': ((pair, sample),)}, case_keys=pair)

    def coupled_native_extraction(self, predictions, batch_index, batch):
        return predictions[batch_index], tuple(map(tuple, batch.receivers[batch_index]))

    def validate_pair(self, pair):
        if not (set(pair) <= set(_TRAIN) or set(pair) <= set(_VALIDATION)):
            raise ValueError('Fixture pairs must stay within one declared partition.')

    def observation_blocks(self, pair):
        blocks = []
        for block_index, size in enumerate((5, 3)):
            coords = _coordinates(pair[0], size).copy()
            coords[:, 1] += block_index * .1
            n, width = len(coords), self.provider.width
            refs = np.stack([_reference(case, coords, width) for case in pair])
            absolute = np.ones((2,n,width), dtype=bool)
            absolute[0, 0, -1] = False
            if block_index:
                absolute[1, -1, 0] = False
            pair_mask = absolute[0] & absolute[1]
            if block_index:
                pair_mask[:] = False  # available absolute measurements without correspondence
            edge_mask = pair_mask.copy()
            measure = np.broadcast_to(np.linspace(.5, 1.5, n)[None,:,None], (2,n,width)).copy()
            components = tuple(FieldComponent(f'field_{j}', 'fixture_unit', j, 1., 1./width) for j in range(width))
            maps = []
            for case in pair:
                deps = tuple(QueryDependency(case, 'analytic_fixture_xyz', 'steady', f'point_{i}', tuple(q))
                             for i, q in enumerate(coords))
                indices = np.repeat(np.arange(n), width)
                channels = np.tile(np.arange(width), n)
                maps.append(LinearExtraction(deps, indices, indices, channels, channels, np.ones(n*width)))
            blocks.append(ObservationBlock(name=f'fixture_observations_{block_index}', components=components,
                scene_ids=tuple(pair), partition='train' if pair[0] in _TRAIN else 'validation',
                frame='analytic_fixture_xyz', time='steady', receiver_ids=tuple(f'r{i}' for i in range(n)),
                coordinates=coords, references=refs, absolute_mask=absolute, pair_mask=pair_mask,
                edge_mask=edge_mask, absolute_measures=measure, pair_measures=measure[0], extractions=tuple(maps),
                block_weight=1., neighbor_groups=tuple('same_support' for _ in range(n))))
        common = blocks[0].pair_mask
        paired = replace(blocks[0], name='fixture_common_pair', absolute_mask=np.stack((common,common)),
                         pair_mask=common, edge_mask=common, block_weight=1.)
        support = []
        for side in range(2):
            masks = np.zeros_like(blocks[0].absolute_mask)
            masks[side] = blocks[0].absolute_mask[side] & ~blocks[0].absolute_mask[1-side]
            support.append(replace(blocks[0], name=f'fixture_common_scene_{side}_only',
                absolute_mask=masks, pair_mask=np.zeros_like(common), edge_mask=np.zeros_like(common),
                block_weight=.5, correspondence='none'))
        for side in range(2):
            masks = np.zeros_like(blocks[1].absolute_mask)
            masks[side] = blocks[1].absolute_mask[side]
            support.append(replace(blocks[1], name=f'fixture_individual_scene_{side}', absolute_mask=masks,
                block_weight=.5, correspondence='none'))
        return (paired,*support)

    def field_request(self, model, case_id, dependencies):
        scene, query = model.field_request(case_id, [dep.coordinates for dep in dependencies])
        return scene, query, 1., 0.


def build_observation_capability(recipe, *, device, resume_payload=None):
    del resume_payload
    width = int(recipe.get('fixture', {}).get('output_dim', 7))
    if width not in (2, 7):
        raise ValueError('The maintained coupled fixture declares output_dim 2 or 7.')
    model = AnalyticJointModel(width, int(recipe['seed'])).to(device)
    provider = AnalyticNativeProvider(recipe, model, device)
    return model, provider, AnalyticObservationBinding(provider)

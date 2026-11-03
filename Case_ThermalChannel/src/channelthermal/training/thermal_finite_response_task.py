"""R15-8 source identity, native functionals and response balance helpers.

The literal P64 transfer epoch and inherited model age are separate clocks.
The old port-transfer loaders retain their original strict contracts.
"""
from __future__ import annotations

import math
import statistics
import copy
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch

from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.unified_training import LossTerm
from ..member_routing import ThermalDetailAnchoredMemberRoutingAdapter
from ..data.datasets import H5Normalizer
from ..data.development_split import read_case_catalog
from .joint_task import _load_data_binding, _read_selected_cases
from .member_routing_task import _training_membership
from .r13_8_anchored_task import ARM_DETAIL, DETAIL_SOURCE_KIND, DETAIL_CORE_SCHEMA, _VARIANTS
from .thermal_port_response_task import (load_thermal_port_response_checkpoint,
    ThermalPortResponseTask, ThermalPortResponseScene)
from .thermal_native_functionals import native_design_functionals


SOURCE_EPOCH = 100
SOURCE_AGE = 5100
SCHEMA = 'thermal_finite_response_v1'


def load_finite_response_source(path, *, device='cpu', pair_content=False, pair_seed=15808):
    """Strictly reconstruct the common P64 source and optionally add C's parameters."""
    # Reconstructors initialize modules before loading weights. Isolate all of
    # those draws as well as the new pair parameters from the case/query RNG.
    with torch.random.fork_rng(devices=[]):
        return _load_finite_response_source(path, device=device, pair_content=pair_content, pair_seed=pair_seed)


def _load_finite_response_source(path, *, device, pair_content, pair_seed):
    model, local, saved = load_thermal_port_response_checkpoint(path, device=device)
    source_recipe = saved['experiment_identity']['recipe']
    spec = source_recipe['thermal_port_response']
    if (int(saved['epoch']) != SOURCE_EPOCH or spec.get('arm') != 'P64'
            or int(spec['source_epoch']) != 5000
            or source_recipe['training']['membership'] not in ('full_train600', 'scratch_train600_exposed_val89')):
        raise ValueError('R15-8 source must be literal P64 epoch 100 with inherited age 5000.')
    if pair_content:
        from honf_forward_core.interaction_reused_member_routing.pair_content import PairContentDetailAnchoredCore
        legacy_core = model.core
        core = PairContentDetailAnchoredCore.from_legacy_config(legacy_core.get_config(), pair_seed=pair_seed)
        core.load_legacy_state_dict(legacy_core.state_dict())
        model = ThermalPairContentAdapter(model.normalization_stats, core=core, heat_scale=model.heat_scale,
                                          nx=model.nx, ny=model.ny, environment_nx=model.environment_nx,
                                          environment_ny=model.environment_ny).to(device)
        # Preserve every inherited adapter buffer; reject any unexplained key.
        expected_new = {k for k in model.state_dict() if '.pair_content.' in k}
        legacy = saved['model_state_dict']
        if set(legacy) != set(model.state_dict()) - expected_new:
            raise RuntimeError('P64 adapter state inventory differs from C inherited state.')
        model.load_state_dict(legacy, strict=False)
    model.core.set_training_clock(SOURCE_AGE, 0)
    return model, local, saved


class ThermalPairContentAdapter(ThermalDetailAnchoredMemberRoutingAdapter):
    """Versioned case adapter for the optional experimental contextual core."""
    SCHEMA = 'thermal_pair_content_detail_anchored_adapter_v1'
    CORE_SCHEMA = 'interaction_reused_detail_anchored_pair_content_v1'

    def __init__(self, normalization_stats, *, core_config=None, core=None, **kwargs):
        if core is None:
            from honf_forward_core.interaction_reused_member_routing.pair_content import PairContentDetailAnchoredCore
            core = PairContentDetailAnchoredCore.from_config(core_config)
        super().__init__(normalization_stats, core=core, **kwargs)


# The case-owned task reuses the existing detail training behavior. This only
# registers the experimental adapter; legacy model dispatch is unchanged.
_VARIANTS.setdefault(ThermalPairContentAdapter.CORE_SCHEMA,
    {**_VARIANTS[DETAIL_CORE_SCHEMA], 'adapter': ThermalPairContentAdapter})


class ThermalFiniteResponseTask(ThermalPortResponseTask):
    """P64 primary task with one complete physical pair per accumulated update.

    The fit-only sampler returns a mapping with baseline/edited TaskBatch
    endpoints and per-channel aligned target/mask/scale entries under delta.
    Both endpoints include all native functional receivers and outside teachers.
    """

    def configure_finite_response(self, recipe, pair_sampler):
        self.configure_port_response(recipe)
        self.source_parent_age = SOURCE_AGE
        self.finite_spec = copy.deepcopy(recipe['thermal_finite_response'])
        self.pair_sampler = pair_sampler

    def make_batch(self, case_keys, key):
        batch = super().make_batch(case_keys, key)
        if key.microbatch_index != 0:
            return batch
        pair = self.pair_sampler(key)
        if pair.get('role') != 'fit' or pair.get('family_id') not in self.training_case_ids:
            raise ValueError('Only admitted fitting families may supply a training pair.')
        scenes = tuple((name, pair[name].scene_inputs, pair[name].receivers)
                       for name in ('baseline', 'edited'))
        scene = ThermalPortResponseScene(batch.scene_inputs.structure,
            batch.scene_inputs.response_structure, batch.scene_inputs.case_ids, scenes)
        return replace(batch, scene_inputs=scene, targets=replace(batch.targets, response_pack=pair))

    def identity_payload(self):
        result = super().identity_payload()
        result['thermal_finite_response'] = copy.deepcopy(self.finite_spec)
        result['finite_response_scales'] = copy.deepcopy(getattr(self.pair_sampler, 'scales', None))
        result['normalization_fit_scope'] = 'inherited P64 TRAIN600 source; no refit'
        return result

    def _terms(self, prediction, targets):
        pair = targets.response_pack
        primary = super()._terms(prediction, replace(targets, response_pack=None))
        if pair is None:
            return primary
        designs = prediction.response_designs
        if set(designs) != {'baseline', 'edited'}:
            raise ValueError('Exactly two complete physical endpoint predictions are required.')
        endpoint = []
        for name in ('baseline', 'edited'):
            terms = super()._terms(designs[name], replace(pair[name].targets, response_pack=None))
            endpoint.append(sum(t.weight * t.numerator / t.denominator for t in terms.values()))
        primary['finite/endpoint'] = LossTerm(sum(endpoint) / 2, 1, .25)
        if self.finite_spec['arm'] in ('B', 'C'):
            primary['finite/delta'] = LossTerm(self._finite_delta(designs, pair), 1,
                                               float(self.finite_spec['lambda_delta']))
        return primary

    def _finite_delta(self, designs, pair):
        names = ('flow/u', 'flow/v', 'flow/p', 'flow/omega', 'temperature/fluid',
                 'temperature/surface', 'temperature/material', 'q_proxy')
        fields = [('pred_field', i) for i in range(5)] + [
            ('pred_interface', 0), ('pred_internal_temperature', 0), ('pred_interface', 1)]
        weight_sum = sum(float(self.native_weights[n]) for n in names)
        if weight_sum <= 0:
            raise ValueError('Native response weights must have positive sum.')
        branches = []
        for mode in ('full', 'restricted'):
            base, edited = (getattr(designs[n], mode) for n in ('baseline', 'edited'))
            field_loss = 0
            for name, (field, channel) in zip(names, fields):
                data = pair['delta'][name]
                b, e = base.native_main[field][..., channel], edited.native_main[field][..., channel]
                target = torch.as_tensor(data['target'], device=b.device, dtype=b.dtype)
                mask = torch.as_tensor(data['mask'], device=b.device, dtype=torch.bool)
                if target.shape != b.shape or mask.shape != b.shape or e.shape != b.shape:
                    raise ValueError('Aligned delta receivers must have exactly matching shapes.')
                scale = float(data['scale'])
                if not math.isfinite(scale) or scale <= 0 or not mask.any():
                    raise ValueError('Each delta channel requires a positive fitting scale and valid receivers.')
                # The supplied mask must already be the common physical mask;
                # additionally enforce validity of both native observation stencils.
                stencil = 'fluid' if field == 'pred_field' else 'material' if field == 'pred_internal_temperature' else 'surface'
                mask = mask & base.native_prepared.stencils[stencil].valid.reshape(mask.shape) & edited.native_prepared.stencils[stencil].valid.reshape(mask.shape)
                if name == 'q_proxy':
                    mask = mask & base.native_prepared.stencils['outside'].valid.reshape(mask.shape) & edited.native_prepared.stencils['outside'].valid.reshape(mask.shape)
                if not mask.any() or not torch.isfinite(target[mask]).all() or not torch.isfinite(b[mask]).all() or not torch.isfinite(e[mask]).all():
                    raise ValueError('Missing or nonfinite aligned physical delta receivers.')
                field_loss = field_loss + float(self.native_weights[name]) * (((e[mask] - b[mask] - target[mask]) / scale).square().mean()) / weight_sum
            values = []
            for name, branch in (('baseline', base), ('edited', edited)):
                data = pair['functionals'][name]
                material = branch.native_main['pred_internal_temperature'][..., 0]
                pressure = branch.native_main['pred_field'][..., 2]
                prepared = branch.native_prepared
                material_mask = (torch.as_tensor(data['material_mask'], device=material.device, dtype=torch.bool)
                    & prepared.stencils['material'].valid.reshape(material.shape)
                    & prepared.source_present[..., None].bool())
                fluid_mask = (torch.as_tensor(data['fluid_mask'], device=pressure.device, dtype=torch.bool)
                    & prepared.stencils['fluid'].valid.reshape(pressure.shape))
                tau = .05 * float(self.model.internal_temperature_std[0])
                values.append(native_design_functionals(material, material_mask, pressure,
                    fluid_mask, data['fluid_x'], data['domain_length_x'], tau))
            extras = []
            for index, name in enumerate(('J', 'P')):
                datum = pair['functionals'][name]
                scale = float(datum['scale'])
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError('Functional response scale must be positive and finite.')
                extras.append(((values[1][index] - values[0][index] - float(datum['target_delta'])) / scale).square())
            branches.append(.8 * field_loss + .1 * extras[0] + .1 * extras[1])
        return sum(branches) / 2

    def loss_denominators(self, batches, phase='joint', arm=ARM_DETAIL):
        # Parent R64 pack denominators do not apply to this every-update pair.
        count = sum(len(batch.targets.case_ids) for batch in batches)
        result = {**{f'final/{role}': count for role in self.native_weights},
                  **{f'local/{role}': count for role in ('temperature/material', 'temperature/surface', 'q_proxy')},
                  'interface/consistency': count, 'port/outside_temperature': count}
        packs = sum(batch.targets.response_pack is not None for batch in batches)
        if packs:
            if packs != 1:
                raise ValueError('Exactly one pair is permitted per accumulated update.')
            result['finite/endpoint'] = 1
            if self.finite_spec['arm'] in ('B', 'C'):
                result['finite/delta'] = 1
        return result

    def work_counts(self, batch, predictions, auxiliary_state):
        """Add teacher calls after loss evaluation has populated child work."""
        counts = dict(super().work_counts(batch, predictions, auxiliary_state))
        designs = getattr(predictions, 'response_designs', {})
        teachers = int(counts.get('teacher_local_calls', 0))
        for name in ('baseline', 'edited'):
            child = designs.get(name)
            child_work = getattr(child, 'work', None) or {}
            child_teachers = int(child_work.get('teacher_local_calls', 0))
            counts[f'response_{name}_teacher_local_calls'] = child_teachers
            teachers += child_teachers
        counts['total_teacher_local_calls'] = teachers
        # The top-level local_calls already includes conditioned calls for both
        # endpoints; only the separately executed teacher calls are added.
        counts['total_local_calls_including_teacher'] = int(counts.get('local_calls', 0)) + teachers
        return counts

    def loss_metadata(self):
        result = super().loss_metadata()
        result['finite/endpoint'] = {'panel': 'response', 'label': 'paired endpoint state',
            'unit': 'weighted scaled squared error', 'formula': 'Mean full P64 objective over both endpoints.',
            'weight': '0.25'}
        if self.finite_spec['arm'] in ('B', 'C'):
            result['finite/delta'] = {'panel': 'response', 'label': 'paired finite response',
                'unit': 'scaled squared error', 'formula': '0.8 native fields plus 0.1 each native J and P; mean full/restricted.',
                'weight': str(self.finite_spec['lambda_delta'])}
        return result


def build_thermal_finite_response_task(recipe, *, pair_sampler=None, arm=ARM_DETAIL, device='cpu'):
    """Build from the literal common source using its saved TRAIN600 transforms."""
    spec = recipe.get('thermal_finite_response', {})
    if (spec.get('schema') != SCHEMA or spec.get('enabled') is not True
            or spec.get('arm') not in ('A', 'B', 'C')
            or spec.get('source_file_epoch') != SOURCE_EPOCH or spec.get('inherited_age') != SOURCE_AGE
            or spec.get('source_checkpoint') != recipe.get('thermal_port_response', {}).get('source_checkpoint')
            or recipe.get('thermal_port_response', {}).get('arm') != 'P64'):
        raise ValueError('R15-8 must declare the literal P64 source and distinct inherited clock.')
    coefficient = float(spec.get('lambda_delta', 0))
    if not math.isfinite(coefficient) or not 0 <= coefficient <= 1:
        raise ValueError('Fixed response coefficient must be finite and in [0,1].')
    model, _, source = load_finite_response_source(spec['source_checkpoint'], device=device,
        pair_content=spec['arm'] == 'C', pair_seed=int(spec.get('pair_seed', 15808)))
    if model.architecture != arm:
        raise ValueError('Requested arm differs from the inherited model architecture.')
    source_recipe = source['experiment_identity']['recipe']
    if any(source_recipe['dataset'][k] != recipe['dataset'][k] for k in ('name', 'data_path', 'manifest_path')):
        raise ValueError('R15-8 dataset differs from the saved source dataset.')
    source_model = copy.deepcopy(source_recipe['member_routing']['model'])
    requested = copy.deepcopy(recipe['member_routing']['model'])
    source_model.pop('query_tile_size', None)
    requested.pop('query_tile_size', None)
    if source_model != requested:
        raise ValueError('R15-8 inherited model configuration differs from source.')
    manifest, data_path, manifest_path = _load_data_binding(recipe['dataset'].get('manifest_path'),
                                                            recipe['dataset'].get('data_path'))
    catalog_source, catalog = read_case_catalog(data_path)
    train_ids, val_ids = _training_membership(recipe, catalog, manifest)
    normalizer = H5Normalizer({k: np.asarray(v, dtype=np.float32) for k, v in model.normalization_stats.items()})
    training = _read_selected_cases(data_path, train_ids, split='train', normalizer=normalizer, include_grid=False)
    validation = _read_selected_cases(data_path, val_ids, split='test', normalizer=normalizer, include_grid=False)
    st = data_path.stat()
    base = copy.deepcopy(recipe)
    base['initialization'].update(source_kind=DETAIL_SOURCE_KIND, source_checkpoint=None, pretrained_weights=False,
        source_parent_age=0, source_arm=f"scratch_common_seed{base['initialization']['seed']}")
    base['member_routing']['health'] = {'enabled': False}
    provider = ThermalFiniteResponseTask(model=model, training_cases=training, validation_cases=validation,
        manifest=manifest, data_path=data_path, recipe=base, device=device, manifest_path=manifest_path,
        catalog=catalog, source_binding={'dataset_path': str(data_path), 'dataset_size_bytes': int(st.st_size),
            'dataset_mtime_ns': int(st.st_mtime_ns), 'catalog_metadata_sha256': catalog_source['metadata_sha256'],
            'manifest_path': None if manifest_path is None else str(manifest_path),
            'manifest_sha256': manifest['manifest_sha256'], 'normalization_source': 'saved P64 TRAIN600',
            'dataset_copy_created': False, 'persistent_cache_created': False})
    provider.configure_finite_response(recipe, pair_sampler)
    if pair_sampler is None:
        from .finite_response_corpus import FitPairCorpus
        scales = json.loads(Path(spec['scales_path']).read_text())['scale_by_action_class']
        provider.pair_sampler = FitPairCorpus(provider, spec['corpus_recipe'], spec['corpus_ledger'],
            qualification=recipe['training']['membership'] == 'fixed25_train150_dev22', scales=scales)
    return model, provider


def load_finite_response_checkpoint(path, *, device='cpu'):
    """Reconstruct a new study checkpoint and validate its separate age clocks."""
    saved = load_trusted_checkpoint(path, map_location='cpu')
    identity = saved.get('experiment_identity', {})
    recipe = identity.get('recipe', {})
    spec = recipe.get('thermal_finite_response', {})
    provider = identity.get('provider_identity', {})
    adapter = provider.get('adapter_state', {})
    epoch = saved.get('epoch')
    if (identity.get('workflow') != 'relation_anchored_case_training_v1'
            or spec.get('schema') != SCHEMA or spec.get('enabled') is not True
            or spec.get('arm') not in ('A', 'B', 'C')
            or spec.get('source_file_epoch') != SOURCE_EPOCH
            or spec.get('inherited_age') != SOURCE_AGE
            or not spec.get('source_checkpoint')
            or spec.get('source_checkpoint') != recipe.get('thermal_port_response', {}).get('source_checkpoint')
            or provider.get('thermal_finite_response') != spec
            or not isinstance(epoch, int) or epoch < 0
            or recipe.get('member_routing', {}).get('arm') != saved.get('arm')):
        raise ValueError('Inconsistent R15-8 checkpoint identity or clocks.')
    source = load_trusted_checkpoint(spec['source_checkpoint'], map_location='cpu')
    source_recipe = source.get('experiment_identity', {}).get('recipe', {})
    source_spec = source_recipe.get('thermal_port_response', {})
    if (source.get('epoch') != SOURCE_EPOCH or source_spec.get('arm') != 'P64'
            or source_spec.get('source_epoch') != 5000
            or source_recipe.get('training', {}).get('membership') not in ('full_train600', 'scratch_train600_exposed_val89')):
        raise ValueError('The saved source file does not have the declared P64 clock.')
    scales = json.loads(Path(spec['scales_path']).read_text())['scale_by_action_class']
    if provider.get('finite_response_scales') != scales:
        raise ValueError('Fixed finite-response scales differ from the checkpoint identity.')
    cls = ThermalPairContentAdapter if spec['arm'] == 'C' else ThermalDetailAnchoredMemberRoutingAdapter
    with torch.random.fork_rng(devices=[]):
        model = cls.from_state_payload(adapter, device=device, state_dict=saved['model_state_dict'])
    model.core.set_training_clock(SOURCE_AGE + epoch, 0)
    model.eval()
    return model, saved


def calibrate_response_balance(examples, parameters, *, blocks=None):
    """Balance 12 actual accumulated primary-plus-endpoint and delta graphs.

    Each example is a pair of scalar losses already reduced exactly as one
    engine update. Gradients include every supplied inherited parameter.
    """
    parameters = tuple(parameters)
    if not parameters:
        raise ValueError('Calibration requires inherited parameters.')
    indices = {id(p): i for i, p in enumerate(parameters)}
    block_indices = {}
    for name, members in (blocks or {}).items():
        members = tuple(members)
        if not members or any(id(p) not in indices for p in members):
            raise ValueError('Calibration blocks must contain inherited parameters.')
        block_indices[name] = [indices[id(p)] for p in members]
    rows = []
    for absolute, delta in examples:
        gradients = [torch.autograd.grad(absolute, parameters, retain_graph=True, allow_unused=True),
                     torch.autograd.grad(delta, parameters, allow_unused=True)]
        def inner(a, b):
            return sum((x.detach().double() * y.detach().double()).sum().item()
                       for x, y in zip(a, b) if x is not None and y is not None)
        norms = [math.sqrt(inner(g, g)) for g in gradients]
        dot = inner(*gradients)
        if not all(math.isfinite(v) for v in (*norms, dot)):
            raise FloatingPointError('Nonfinite calibration gradient.')
        row = {'absolute_norm': norms[0], 'delta_norm': norms[1],
                     'cosine': dot / (norms[0] * norms[1]) if all(norms) else None,
                     'absolute_zero': norms[0] == 0, 'delta_zero': norms[1] == 0}
        row['blocks'] = {}
        for name, selected in block_indices.items():
            ga, gd = ([g[i] for i in selected] for g in gradients)
            na, nd = math.sqrt(inner(ga, ga)), math.sqrt(inner(gd, gd))
            row['blocks'][name] = {'absolute_norm': na, 'delta_norm': nd,
                'cosine': inner(ga, gd) / (na * nd) if na and nd else None}
        rows.append(row)
    if len(rows) != 12:
        raise ValueError('Calibration requires exactly 12 accumulated examples.')
    ratio = statistics.median(r['absolute_norm'] / max(r['delta_norm'], 1e-12) for r in rows)
    return min(1.0, .25 * ratio), rows

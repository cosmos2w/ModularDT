"""HET-MF task and portable commands over the shared field trainer."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .honf_adapter import ElectroThermalHONFAdapter
from .processing import FIELD_ORDER
from .training_data import HETMFDataset


class HETMFTask:
    """Own physical records, target transforms and split-specific sampling."""
    def __init__(self, path, config):
        self.path, self.config = str(Path(path).resolve()), config
        name = config['model']['name']
        if name in ('a2d', 'routed_anchored'):
            self.adapter = ElectroThermalHONFAdapter(device='cpu')
        else:
            from .classic_models import HETMFClassicAdapter
            self.adapter = HETMFClassicAdapter(device='cpu')
        panel = config.get('qualification_panel')
        self._panel = None
        if panel is not None:
            from .routed_qualification import select_panel
            self._panel = {split: select_panel(self.path, split, int(panel[split])) for split in ('train', 'dev')}
        reference = self.dataset('train', 0)
        self._identity = {'schema': 'hetmf_training_h5_v1',
                          'payload_sha256': reference.dataset_payload_sha256,
                          'plan_hash': reference.dataset_plan_hash,
                          'normalization_train_membership_sha256': reference.normalization_train_membership_sha256,
                          'field_order': list(FIELD_ORDER)}
        self._normalization = reference.normalization
        if self._panel is not None:
            self._identity['qualification_state_ids'] = {
                split: [state for _, state in members] for split, members in self._panel.items()
            }
        if name == 'routed_anchored':
            fields = self._normalization['fields']
            self._identity['field_normalizer'] = {
                'mean': np.asarray(fields['mean']).tolist(),
                'std': np.asarray(fields['std']).tolist(),
                'count': int(fields['count']),
                'fit_split': 'train',
            }
        reference.close()

    def dataset(self, split, epoch):
        settings = self.config['training']
        dataset = HETMFDataset(self.path, split=split,
                              query_count=int(settings['queries'] if split == 'train' else settings['validation_queries']),
                              seed=int(settings['seed']), normalize=True)
        dataset.set_epoch(epoch if split == 'train' else 0)
        if self._panel is not None:
            if split not in ('train', 'dev'):
                dataset.close()
                raise ValueError('qualification panel permits only TRAIN and DEV')
            selected = self._panel[split]
            dataset.indices = dataset.indices[[i for i, _ in selected]]
        return dataset

    def collate(self, samples):
        return self.adapter.collate(samples)

    def predict(self, model, batch):
        if self.config['model']['name'] == 'routed_anchored':
            from .e_anchor_locator import attach_e_coverage
            query = attach_e_coverage(batch.query, batch.scene, model.locator_mode)
            prepared = model.prepare(batch.scene)
            if model.training:
                full, restricted, _ = model.decode_paired(prepared, query)
                return full.joint_fields, restricted.joint_fields
            return model.decode(prepared, query, intervention={'read_mode': 'restricted'}).joint_fields
        if self.config['model']['name'] == 'a2d':
            return model.decode(model.prepare(batch.scene), batch.query).nonlinear_field
        return model(batch.scene)

    def identity(self):
        return copy.deepcopy(self._identity)

    def physical_scale(self):
        return self._normalization['fields']['mean'], self._normalization['fields']['std']


def read_config(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict) or value.get('schema') != 'hetmf_fit_v1':
        raise ValueError('expected a hetmf_fit_v1 YAML configuration')
    if value['model']['name'] not in ('six_hyperedge', 'dense_pairwise', 'a2d', 'routed_anchored'):
        raise ValueError('choose six_hyperedge, dense_pairwise, a2d, or routed_anchored')
    settings = value['training']
    for key in ('epochs', 'batch_size', 'queries', 'validation_queries'):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f'training.{key} must be a positive integer')
    if int(settings.get('num_workers', 0)) < 0 or int(settings['seed']) < 0:
        raise ValueError('nonnegative worker count and seed required')
    for key in ('learning_rate', 'weight_decay', 'gradient_clip_norm'):
        if not np.isfinite(settings[key]) or settings[key] < 0 or (key != 'weight_decay' and settings[key] == 0):
            raise ValueError(f'invalid training.{key}')
    if any(not np.isfinite(value['storage'][k]) or value['storage'][k] <= 0
           for k in ('retained_budget_gib', 'minimum_free_gib')):
        raise ValueError('positive explicit output budget and headroom required')
    return value


def build_model(config, device='cpu'):
    from honf_runtime.field_training import seed_all
    seed_all(int(config['training']['seed']))
    options = dict(config['model'])
    name = options.pop('name')
    options.pop('display_name', None)
    if name == 'routed_anchored':
        from honf_forward_core.interaction_reused_member_routing import AnchoredMemberRoutingCore
        base = ElectroThermalHONFAdapter.model_config(context_dim=options.pop('context_dim'), num_heads=options.pop('num_heads'), init_seed=int(config['training']['seed']))
        return AnchoredMemberRoutingCore(base, input_dataset='hetmf_standard', **options).to(device)
    if name == 'a2d':
        return ElectroThermalHONFAdapter(device=device).build_model(**options)
    from .classic_models import build_classic_model
    return build_classic_model(name, **options).to(device)


def dry_run(config, dataset, *, device='cpu'):
    """Qualify two TRAIN cases and 17 queries; no optimizer or output writes."""
    task = HETMFTask(dataset, config)
    data = HETMFDataset(dataset, split='train', query_count=17, seed=config['training']['seed'], normalize=True)
    try:
        if task._panel is not None:
            data.indices = data.indices[[i for i, _ in task._panel['train']]]
        batch = task.collate([data[0], data[1]]).to(device)
        model = build_model(config, device)
        prediction = task.predict(model, batch)
        predictions = prediction if isinstance(prediction, tuple) else (prediction,)
        if any(value.shape != (2, 17, 5) or not torch.isfinite(value).all() for value in predictions):
            raise RuntimeError('field interface failed')
        from honf_runtime.field_training import field_loss
        loss = field_loss(prediction, batch)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('nonfinite or absent field gradients')
        return {'model': config['model']['display_name'], 'dataset': task.identity(),
                'train_cases': len(data), 'declared_epochs': config['training']['epochs'],
                'qualification_cases': 2, 'qualification_queries': 17,
                'field_shape': list(predictions[0].shape), 'finite_backward': True,
                'fitting_performed': False, 'requested_batch_memory_qualified': False}
    finally:
        data.close()


def train_command(args):
    config = read_config(args.config)
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.dry_run:
        return dry_run(config, args.dataset, device=args.device)
    if not args.yes:
        raise PermissionError('fitting requires explicit --yes; use --dry-run for the read-only interface check')
    from honf_runtime.field_training import fit
    task = HETMFTask(args.dataset, config)
    model = build_model(config, args.device)
    return fit(model, task, config['training'], output=args.output, device=args.device,
               resume=args.resume, config=config)


def evaluate_command(args):
    from honf_runtime.field_training import load_checkpoint, evaluate
    run = Path(args.run)
    checkpoint = run / args.checkpoint
    payload = load_checkpoint(checkpoint, args.device)
    config = copy.deepcopy(payload['binding']['config'])
    dataset = args.dataset or payload.get('dataset_location')
    if not dataset:
        raise ValueError('supply --dataset for this checkpoint')
    task = HETMFTask(dataset, config)
    if task.identity() != payload['binding']['dataset']:
        raise ValueError('evaluation dataset/normalizer differs from the checkpoint')
    if args.queries is not None:
        config['training']['validation_queries'] = args.queries
    model = build_model(config, args.device)
    model.load_state_dict(payload['model'], strict=True)
    result = evaluate(model, task, config['training'], split=args.split, device=args.device)
    result['field_order'] = list(FIELD_ORDER)
    result['checkpoint_epoch'] = payload['epoch']
    return result

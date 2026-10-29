"""Small case-neutral field trainer with atomic, epoch-boundary resume.

A task owns data, collation, prediction, normalization and physical metrics.
This module owns the common epoch loop, optimizer, artifacts and RNG state.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random
import shutil
import time
import tempfile
from typing import Protocol, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class FieldTask(Protocol):
    def dataset(self, split: str, epoch: int): ...
    def collate(self, samples): ...
    def predict(self, model, batch): ...
    def identity(self) -> dict: ...
    def physical_scale(self) -> tuple[np.ndarray, np.ndarray]: ...


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def _atomic_save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path, device='cpu') -> dict:
    with torch.serialization.safe_globals([torch.nn.parameter.UninitializedParameter]):
        return torch.load(path, map_location=device, weights_only=True)


def _loader(task, split, epoch, settings):
    data = task.dataset(split, epoch)
    indices = list(range(len(data)))
    if split == 'train':
        indices = np.random.default_rng(int(settings['seed']) + epoch).permutation(indices).tolist()
    workers = int(settings.get('num_workers', 0))
    options = {'multiprocessing_context': 'spawn'} if workers else {}
    # Recreate workers each epoch so dataset.set_epoch reaches their copies.
    return DataLoader(data, sampler=indices, batch_size=int(settings['batch_size']),
                      num_workers=workers, collate_fn=task.collate, persistent_workers=False, **options)


def field_loss(prediction, batch):
    if isinstance(prediction, tuple):
        return sum(field_loss(field, batch) for field in prediction) / len(prediction)
    if prediction.shape != batch.targets.shape or not torch.isfinite(prediction).all():
        raise ValueError('model must return finite [B,Q,C] fields aligned with targets')
    weights = batch.query_weights
    return (((prediction - batch.targets).square().mean(-1) * weights).sum(-1)
            / weights.sum(-1).clamp_min(1e-12)).mean()


@torch.no_grad()
def evaluate(model, task, settings, *, split='dev', device='cpu') -> dict:
    model.eval()
    squared = target_squared = None
    score = 0.0
    cases = queries = 0
    mean, std = task.physical_scale()
    mean, std = torch.as_tensor(mean, device=device), torch.as_tensor(std, device=device)
    for batch in _loader(task, split, 0, settings):
        batch = batch.to(device)
        prediction = task.predict(model, batch)
        count = int(prediction.shape[0])
        score += float(field_loss(prediction, batch)) * count
        error = (prediction - batch.targets).double() * std
        physical = batch.targets.double() * std + mean
        s, t = error.square().sum((0, 1)), physical.square().sum((0, 1))
        squared = s if squared is None else squared + s
        target_squared = t if target_squared is None else target_squared + t
        cases += count
        queries += count * int(prediction.shape[1])
    if not cases:
        raise ValueError(f'empty {split} membership')
    return {'split': split, 'cases': cases, 'queries': queries,
            'normalized_field_mse': score / cases,
            'physical_rmse_by_channel': (squared / queries).sqrt().cpu().tolist(),
            'physical_relative_l2_by_channel': (squared / target_squared.clamp_min(1e-24)).sqrt().cpu().tolist()}


def _rng_state():
    n = np.random.get_state()
    state = {'python': random.getstate(), 'numpy': (n[0], n[1].tolist(), n[2], n[3], n[4]),
             'torch': torch.get_rng_state()}
    if torch.cuda.is_initialized():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if 'cuda' in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def _curve(output, history):
    with tempfile.TemporaryDirectory(prefix="curve-", dir=output / "temporary") as cache:
        # This import occurs only inside an authorized fit, after output budgeting.
        os.environ['MPLCONFIGDIR'] = cache
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_pdf import FigureCanvasPdf
        figure = Figure(figsize=(7, 4))
        FigureCanvasPdf(figure)
        ax = figure.subplots()
        train = [r for r in history if 'train_normalized_field_mse' in r]
        valid = [r for r in history if 'validation' in r]
        ax.plot([r['epoch'] for r in train], [r['train_normalized_field_mse'] for r in train], label='sampled TRAIN')
        ax.plot([r['epoch'] for r in valid], [r['validation']['normalized_field_mse'] for r in valid], label='fixed DEV')
        ax.set(xlabel='Epoch', ylabel='Normalized field MSE', yscale='log')
        ax.legend()
        figure.tight_layout()
        temporary = output / 'loss_curve.tmp.pdf'
        figure.savefig(temporary)
        os.replace(temporary, output / 'loss_curve.pdf')

def fit(model, task, settings, *, output, device='cpu', resume=None, config=None):
    """Fit only the declared training membership, retaining a resumable state."""
    import fcntl
    output = Path(output).resolve()
    if Path('/data2') not in output.parents:
        raise ValueError('training outputs must be beneath mounted /data2')
    if not os.path.ismount('/data2'):
        raise RuntimeError('/data2 is not mounted')
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError('use a new output directory or explicitly resume latest.pt')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _fit_owned(model, task, settings, output, device, resume, config)


def _fit_owned(model, task, settings, output, device, resume, config):
    storage = (config or {}).get('storage', {})
    budget = int(float(storage.get('retained_budget_gib', 2)) * 2**30)
    free_floor = int(float(storage.get('minimum_free_gib', 100)) * 2**30)
    if shutil.disk_usage(output).free - budget < free_floor:
        raise RuntimeError('declared output budget would breach free-space headroom')
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings['learning_rate']),
                                 weight_decay=float(settings['weight_decay']))
    binding = {'dataset': task.identity(), 'config': copy.deepcopy(config or {})}
    if 'training' in binding['config']:
        binding['config']['training'].pop('epochs', None)
    history = []
    epoch = 0
    best = float('inf')
    if resume:
        payload = load_checkpoint(resume, device)
        previous_binding = copy.deepcopy(payload['binding'])
        if 'training' in previous_binding['config']:
            previous_binding['config']['training'].pop('epochs', None)
        if previous_binding != binding:
            raise ValueError('resume requires the same dataset, normalizer and model/training configuration')
        model.load_state_dict(payload['model'], strict=True)
        optimizer.load_state_dict(payload['optimizer'])
        epoch, history, best = payload['epoch'], payload['history'], payload['best_field_mse']
        _restore_rng(payload['rng'])
    stop = int(settings['epochs'])
    if stop <= epoch:
        raise ValueError('requested epoch stop must exceed the completed epoch')
    _atomic_json(output / 'storage_budget.json', {'retained_budget_bytes': budget,
                 'minimum_free_bytes': free_floor, 'free_before_bytes': shutil.disk_usage(output).free})
    (output / 'temporary').mkdir(exist_ok=True)
    _atomic_json(output / 'config.json', config or {})
    if not history:
        history.append({'epoch': 0, 'validation': evaluate(model, task, settings, device=device)})
        _atomic_json(output / 'history.json', history)
        _curve(output, history)
    prior_elapsed = float(history[-1].get('elapsed_fit_seconds', 0.0))
    start = time.monotonic()
    for next_epoch in range(epoch + 1, stop + 1):
        if (output / 'STOP').exists():
            break
        model.train()
        total = 0.0
        visits = 0
        loader = _loader(task, 'train', next_epoch, settings)
        progress = tqdm(loader, desc=f'Epoch {next_epoch}/{stop}', disable=not settings.get('progress', True))
        for batch in progress:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = field_loss(task.predict(model, batch), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings['gradient_clip_norm']), error_if_nonfinite=True)
            optimizer.step()
            count = len(batch.targets)
            total += float(loss.detach()) * count
            visits += count
            progress.set_postfix(field_mse=f'{total/visits:.4g}')
        if visits != len(loader.dataset):
            raise RuntimeError('training epoch did not visit the declared membership exactly once')
        row = {'epoch': next_epoch, 'train_cases': visits, 'train_normalized_field_mse': total / visits,
               'elapsed_fit_seconds': prior_elapsed + time.monotonic() - start,
               'validation': evaluate(model, task, settings, device=device)}
        history.append(row)
        epoch = next_epoch
        improved = row['validation']['normalized_field_mse'] < best
        best = min(best, row['validation']['normalized_field_mse'])
        payload = {'schema': 'honf_field_fit_v1', 'epoch': epoch, 'binding': binding,
                   'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'rng': _rng_state(), 'history': history, 'best_field_mse': best,
                   'dataset_location': getattr(task, 'path', None)}
        payload['binding'] = {'dataset': task.identity(), 'config': copy.deepcopy(config or {})}
        _atomic_save(output / 'latest.pt', payload)
        if improved:
            _atomic_save(output / 'best_field.pt', payload)
        if epoch % 100 == 0:
            _atomic_save(output / f'epoch_{epoch:04d}.pt', payload)
        _atomic_json(output / 'history.json', history)
        if epoch % 10 == 0 or epoch == stop:
            _curve(output, history)
        size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if size > budget or shutil.disk_usage(output).free < free_floor:
            raise RuntimeError('training output exceeded its declared storage budget')
    _curve(output, history)
    _atomic_json(output / 'summary.json', {'completed_epoch': epoch, 'declared_stop_epoch': stop,
                 'best_dev_normalized_field_mse': best if np.isfinite(best) else None,
                 'dataset': task.identity(), 'stopped_cleanly': epoch < stop})
    return {'completed_epoch': epoch, 'output': str(output)}

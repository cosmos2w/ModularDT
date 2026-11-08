"""Side-effect-free gradient calibration and actual-update diagnostics for R16-10.

Callers supply mean-reduced losses on the already selected TRAIN-only panels.
No data loading or panel selection occurs here.
"""
from __future__ import annotations

import copy
import random
from contextlib import contextmanager
from itertools import combinations

import numpy as np
import torch


@contextmanager
def preserved_state(model, optimizer=None):
    """Restore parameters, buffers, gradients, mode, optimizer and all RNGs even on error."""
    state = copy.deepcopy(model.state_dict())
    modes = [(m, m.training) for m in model.modules()]
    grads = [(p, None if p.grad is None else p.grad.detach().clone()) for p in model.parameters()]
    optim = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    try:
        yield
    finally:
        model.load_state_dict(state)
        for module, training in modes:
            module.training = training
        for param, grad in grads:
            param.grad = grad
        if optimizer is not None:
            optimizer.load_state_dict(optim)
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])


def _parameters(model):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not params:
        raise ValueError('No trainable parameters')
    return params


def _pieces(loss_fn):
    """Yield pre-normalized scalar pieces without keeping previous graphs alive."""
    value = loss_fn()
    pieces = (value,) if isinstance(value, torch.Tensor) else value
    seen = False
    for piece in pieces:
        seen = True
        if not isinstance(piece, torch.Tensor) or piece.numel() != 1 or not torch.isfinite(piece).all():
            raise ValueError('Loss must consist of finite scalar tensors')
        yield piece
    if not seen:
        raise ValueError('Loss has no pieces')


def _loss_value(loss_fn):
    return sum(float(piece.detach()) for piece in _pieces(loss_fn))


def _gradient(loss_fn, params):
    grads = [torch.zeros_like(p) for _, p in params]
    loss = 0.
    connected = False
    for piece in _pieces(loss_fn):
        loss += float(piece.detach())
        if not piece.requires_grad:
            raise ValueError('Loss is disconnected from trainable parameters')
        part = torch.autograd.grad(piece, [p for _, p in params], allow_unused=True)
        connected |= any(g is not None for g in part)
        for dest, grad in zip(grads, part):
            if grad is not None:
                dest.add_(grad.detach())
    if not connected:
        raise ValueError('Loss is disconnected from trainable parameters')
    if any(not torch.isfinite(g).all() for g in grads):
        raise ValueError('Nonfinite gradient')
    return loss, grads


def _norm(grads):
    return float(torch.sqrt(sum((g.double().square().sum() for g in grads), torch.tensor(0., dtype=torch.double, device=grads[0].device))).item())


def _cos(a, b):
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return None
    return float(sum((x.double() * y.double()).sum().item() for x, y in zip(a, b)) / (na * nb))


def gradient_diagnostics(model, loss_fns, parameter_groups=None, *, require_nonzero=False):
    """Return mean losses, global/group gradient norms and pairwise cosine similarities.

    parameter_groups maps names to iterables of exact model parameter names. Unused
    parameters are treated as zero. Satisfied hinge losses may have zero gradients;
    coefficient calibration explicitly requires nonzero global norms.
    """
    params = _parameters(model)
    names = [n for n, _ in params]
    groups = {'global': names}
    if parameter_groups:
        for name, members in parameter_groups.items():
            members = list(members)
            if name == 'global' or not members or not set(members).issubset(names):
                raise ValueError(f'Invalid parameter group: {name}')
            groups[name] = members
    losses, gradients = {}, {}
    with preserved_state(model):
        for key, fn in loss_fns.items():
            losses[key], gradients[key] = _gradient(fn, params)
            if require_nonzero and _norm(gradients[key]) == 0:
                raise ValueError(f'Zero gradient for {key}')
    result = {'losses': losses, 'groups': {}}
    for group, members in groups.items():
        indices = [i for i, name in enumerate(names) if name in members]
        subset = {key: [g[i] for i in indices] for key, g in gradients.items()}
        result['groups'][group] = {
            'norms': {key: _norm(g) for key, g in subset.items()},
            'cosines': {f'{a}:{b}': _cos(subset[a], subset[b]) for a, b in combinations(subset, 2)},
        }
    return result


def calibrate_components(model, loss_fns, parameter_groups=None):
    """Calculate frozen E/R/D coefficients from a common B0 checkpoint.

    loss_fns must contain exactly native, endpoint, response, decision; the
    calling code is responsible for input-only sampled eligible panel selection.
    """
    if set(loss_fns) != {'native', 'endpoint', 'response', 'decision'}:
        raise ValueError('Calibration requires native, endpoint, response and decision losses')
    result = gradient_diagnostics(model, loss_fns, parameter_groups, require_nonzero=True)
    norms = result['groups']['global']['norms']
    result['coefficients'] = {key: rho * norms['native'] / norms[key] for key, rho in
                              (('endpoint', .25), ('response', .10), ('decision', .05))}
    return result


def counterfactual_updates(model, optimizer, loss_fns, combined_weights, clip_grad_norm=1.0, parameter_groups=None):
    """Compare native and combined actual optimizer updates, restoring all state.

    Loss closures are reevaluated after each step to report immediate changes;
    optimizer must be the run's AdamW, including its current moment state.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError('Counterfactual requires the actual AdamW optimizer')
    if 'native' not in loss_fns or 'native' not in combined_weights or not set(combined_weights).issubset(loss_fns):
        raise ValueError('Missing native or combined objective components')
    if clip_grad_norm <= 0:
        raise ValueError('clip_grad_norm must be positive')
    diagnostics = gradient_diagnostics(model, loss_fns, parameter_groups)
    params = _parameters(model)
    output = {'gradient_diagnostics': diagnostics, 'updates': {}}
    with preserved_state(model, optimizer):
        initial_model = copy.deepcopy(model.state_dict())
        initial_optimizer = copy.deepcopy(optimizer.state_dict())
        initial_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                       torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
        for name, weights in (('native_only', {'native': 1.0}), ('combined', combined_weights)):
            model.load_state_dict(initial_model)
            # Optimizer loading can reuse same-device state tensors. Give each
            # counterfactual its own moments so the first update cannot advance
            # the second update's reference state.
            optimizer.load_state_dict(copy.deepcopy(initial_optimizer))
            random.setstate(initial_rng[0]); np.random.set_state(initial_rng[1]); torch.set_rng_state(initial_rng[2])
            if initial_rng[3] is not None:
                torch.cuda.set_rng_state_all(initial_rng[3])
            before = {key: _loss_value(fn) for key, fn in loss_fns.items()}
            if not all(np.isfinite(v) for v in before.values()):
                raise ValueError('Nonfinite loss before update')
            optimizer.zero_grad(set_to_none=True)
            for key, weight in weights.items():
                if not np.isfinite(weight):
                    raise ValueError('Nonfinite objective weight')
                for piece in _pieces(loss_fns[key]):
                    (float(weight) * piece).backward()
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in params], clip_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            delta = [p.detach() - initial_model[n] for n, p in params]
            random.setstate(initial_rng[0]); np.random.set_state(initial_rng[1]); torch.set_rng_state(initial_rng[2])
            if initial_rng[3] is not None:
                torch.cuda.set_rng_state_all(initial_rng[3])
            after = {key: _loss_value(fn) for key, fn in loss_fns.items()}
            if not all(np.isfinite(v) for v in after.values()):
                raise ValueError('Nonfinite loss after update')
            output['updates'][name] = {'before': before, 'after': after,
                'changes': {key: after[key] - before[key] for key in before},
                'gradient_norm_before_clip': float(norm),
                'clip_scale': min(1., clip_grad_norm / float(norm)) if float(norm) else 1.,
                'displacement_norm': _norm(delta)}
    return output

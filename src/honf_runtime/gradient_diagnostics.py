"""Reusable state-preserving gradient calibration diagnostics."""
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



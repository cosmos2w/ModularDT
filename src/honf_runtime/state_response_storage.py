"""Storage preparation for a new state-response fit or same-run resume."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from . import storage_policy

_DATA2 = Path('/data2')


def _existing_parent(path: Path) -> Path:
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def prepare_state_response_storage(config: Mapping[str, Any], output: str | Path,
                                   model: torch.nn.Module, *, completed_epoch: int,
                                   stop_after: int, checkpoint_epochs: Sequence[int]) -> dict[str, Any]:
    """Verify DATA2 bindings, location and byte budgets before a fit writes.

    ``completed_epoch`` is zero for a fresh run, otherwise the epoch read from
    the same-run checkpoint. ``checkpoint_epochs`` must be the actual engine
    policy, including any authorized extension. The returned snapshot records
    free space, retained output, the checkpoint/temporary estimates, budgets and
    headroom. The caller records it and owns any case-specific formal-run alias.
    """
    storage = config.get('storage')
    if not isinstance(storage, Mapping):
        raise TypeError('State-response fitting requires an explicit storage configuration.')
    root_value = storage.get('output_root')
    if not isinstance(root_value, str) or not root_value:
        raise ValueError('storage.output_root must name the DATA2 run root.')
    for name in ('max_output_bytes', 'max_peak_temporary_bytes', 'minimum_free_headroom_bytes'):
        value = storage.get(name)
        if type(value) is not int or value < 0 or (name != 'minimum_free_headroom_bytes' and value == 0):
            raise ValueError(f'storage.{name} must be a valid byte budget.')
    if (type(completed_epoch) is not int or type(stop_after) is not int
            or completed_epoch < 0 or stop_after <= completed_epoch):
        raise ValueError('The requested stop must advance the completed epoch.')
    if (not isinstance(checkpoint_epochs, Sequence) or isinstance(checkpoint_epochs, (str, bytes))
            or any(type(epoch) is not int or epoch < 1 for epoch in checkpoint_epochs)
            or tuple(sorted(set(checkpoint_epochs))) != tuple(checkpoint_epochs)):
        raise ValueError('Checkpoint epochs must be a sorted unique sequence of positive epochs.')

    bindings = storage_policy._ensure_workspace_storage_bindings()
    data2 = _DATA2.resolve(strict=True)
    root = Path(root_value).expanduser().resolve(strict=False)
    target = Path(output).expanduser().resolve(strict=False)
    if not root.is_relative_to(data2) or root == data2:
        raise ValueError('storage.output_root must resolve beneath /data2.')
    if not target.is_relative_to(root):
        raise ValueError(f'Run output must be inside {root}.')
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f'The configured output root is not a directory: {root}')
    data2_device = data2.stat().st_dev
    if _existing_parent(root).stat().st_dev != data2_device:
        raise RuntimeError('The configured output root is not on the mounted /data2 filesystem.')
    if _existing_parent(target).stat().st_dev != data2_device:
        raise RuntimeError('The run output is not on the mounted /data2 filesystem.')
    if completed_epoch == 0:
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise FileExistsError(f'A fresh fit requires an empty output directory: {target}')
    elif not target.is_dir():
        raise ValueError('A resumed fit requires its existing run directory.')

    # Verify placement and existing usage before accounting for new artifacts.
    storage_policy.verify_storage(config, output=target, estimated_bytes=0, peak_temporary_bytes=0)
    estimated, temporary = storage_policy._storage_estimate(
        model, completed_epoch=completed_epoch, stop_after=stop_after, config=config,
        checkpoint_epochs=tuple(checkpoint_epochs))
    snapshot = storage_policy._prepare_storage(config, target, estimated_bytes=estimated,
                                            peak_temporary_bytes=temporary, bindings=bindings)
    snapshot.update({'run_output': str(target), 'completed_epoch': completed_epoch,
                     'stop_after_epoch': stop_after, 'checkpoint_epochs': list(checkpoint_epochs)})
    return snapshot

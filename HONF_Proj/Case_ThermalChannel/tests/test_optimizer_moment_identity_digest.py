from __future__ import annotations

import copy
import hashlib
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from channelthermal.training.unified_task import _tensor_digest


def _legacy_flat_digest(named):
    digest = hashlib.sha256()
    for name, value in sorted(named.items()):
        digest.update(name.encode("utf-8") + b"\0")
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        else:
            array = np.ascontiguousarray(value)
            digest.update(str((array.shape, array.dtype.str)).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def test_flat_normalization_digest_keeps_legacy_bytes():
    stats = {
        "field_mean_by_channel": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "field_std_by_channel": np.asarray([4.0, 5.0, 6.0], dtype=np.float32),
    }
    assert _tensor_digest(stats) == _legacy_flat_digest(stats)


def test_nested_adamw_moment_digest_is_recursive_and_stable():
    state = OrderedDict([
        ("layer.weight", {"step": torch.tensor(7.0),
                          "exp_avg": torch.arange(6, dtype=torch.float32).reshape(2, 3),
                          "exp_avg_sq": torch.arange(6, dtype=torch.float32).reshape(2, 3).square()}),
        ("layer.bias", {"step": torch.tensor(7.0),
                        "exp_avg": torch.tensor([0.25, -0.5]),
                        "exp_avg_sq": torch.tensor([0.0625, 0.25])}),
    ])
    cloned = copy.deepcopy(state)
    reordered = OrderedDict(
        (name, OrderedDict(reversed(list(moment.items()))))
        for name, moment in reversed(list(cloned.items()))
    )

    original_digest = _tensor_digest(state)
    assert original_digest == _tensor_digest(cloned)
    assert original_digest == _tensor_digest(reordered)

    changed = copy.deepcopy(state)
    changed["layer.weight"]["exp_avg"][0, 0] += 1.0
    assert _tensor_digest(changed) != original_digest

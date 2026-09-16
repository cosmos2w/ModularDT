"""CPU regressions for detached route-map export arrays."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "tools" / "diagnostics"), str(_ROOT / "src")]

import run_dynamic_sparse_routing_study as study


def test_map_tensor_conversion_detaches_and_returns_finite_cpu_array() -> None:
    value = torch.tensor([[1.0, -2.0], [3.5, 4.0]], requires_grad=True)

    converted = study._map_value_to_cpu_array(value)

    assert isinstance(converted, np.ndarray)
    assert converted.dtype == np.float32
    assert np.array_equal(converted, value.detach().numpy())
    assert np.isfinite(converted).all()


def test_selected_map_export_keeps_cpu_geometry_arrays_finite() -> None:
    class _PortHead:
        @staticmethod
        def fixed_theta_tokens(count: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
            return torch.tensor([[0.0, 1.0, 0.0]], device=device, dtype=dtype).expand(count, -1)

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(
                core_honf=SimpleNamespace(
                    module_radius=0.45,
                    interface_model=SimpleNamespace(
                        routing=SimpleNamespace(strategy="mean_shift")
                    ),
                )
            )
            self.local_coupling = SimpleNamespace(port_head=_PortHead())

    model = _Model()
    encoded = SimpleNamespace(
        module_centers=torch.tensor(
            [[[0.0, 0.0], [1.0, 1.0]]], dtype=torch.float32, requires_grad=True
        ),
        module_present=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        env_coords=torch.tensor(
            [[[0.0, 1.0], [1.0, 1.0]]], dtype=torch.float32, requires_grad=True
        ),
        env_weights=torch.tensor([[1.0, 2.0]], dtype=torch.float32, requires_grad=True),
    )
    outputs = {
        "prepared_state": SimpleNamespace(prepared=SimpleNamespace(encoded=encoded)),
        "interaction_aux": {},
        "pred_port_condition": torch.zeros((1, 2, 1, 1), dtype=torch.float32),
    }
    sample = {
        "structure": {
            "module_present": np.array([1.0, 1.0], dtype=np.float32),
            "module_centers": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        }
    }
    preparation_records = {
        "p0_port": [
            {
                "_routing_maps": {
                    "source_membership": torch.tensor(
                        [[[0.25, 0.75], [0.5, 0.5]]], dtype=torch.float32, requires_grad=True
                    )
                }
            }
        ]
    }

    maps, selection = study._selected_routing_map_arrays(
        model,
        outputs,
        sample,
        np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32),
        preparation_records,
        max_map_values=128,
    )

    assert selection["query_count"] == 2
    assert maps["module_centers"].shape == (2, 2)
    assert maps["env_coords"].shape == (2, 2)
    assert maps["p0_port__source_membership"].shape == (2, 2)
    for value in maps.values():
        assert isinstance(value, np.ndarray)
        if np.issubdtype(value.dtype, np.number):
            assert np.isfinite(value).all()

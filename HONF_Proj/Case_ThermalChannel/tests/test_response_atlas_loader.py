from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from channelthermal.interaction_evidence import load_response_atlas_stencil


def _write_atlas(tmp_path: Path, *, flow_nu: float | None = None, runtime_nu: float = 0.018, fluid_alpha: float = 0.02) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    module_id = "anchor:module:0"
    config: dict[str, Any] = {
        "domain": {"nx": 128, "ny": 64, "lx": 12.0, "ly": 6.0, "module_radius": 0.45},
        "flow": {"re": 50.0, "u_in": 1.0, "nu": flow_nu, "viscosity_scale": 1.0},
        "thermal": {
            "solid_alpha": 0.01,
            "fluid_alpha": fluid_alpha,
            "solid_k": 1.0,
            "fluid_k": 1.5,
        },
    }
    family_metadata = {
        "anchor_id": "anchor",
        "physical_family_id": "family",
        "case_config": config,
        "runtime": {"nu": runtime_nu},
    }
    arrays: dict[str, Any] = {
        "all_labels": np.asarray(["baseline", "i_plus"], dtype="U"),
        "active_module_ids": np.asarray([module_id], dtype="U"),
        "module_centers_xy": np.asarray([[[1.0, 2.0]], [[1.1, 2.0]]], dtype=np.float32),
        "heating": np.asarray([[1.0], [1.0]], dtype=np.float32),
        "source": np.asarray(["reference_solver", "reference_solver"], dtype="U"),
        "split": np.asarray(["train", "train"], dtype="U"),
        "elapsed_seconds": np.asarray([1.0, 1.0], dtype=np.float64),
        "elapsed_seconds_measured": np.asarray([True, True]),
        "pressure_drop": np.asarray([0.1, 0.1], dtype=np.float64),
        "module_peak_temperature": np.asarray([[10.0], [10.1]], dtype=np.float32),
        "family_metadata_json": np.asarray(json.dumps(family_metadata)),
    }
    role_data = {
        "fluid_fields": (
            np.zeros((2, 1, 5), dtype=np.float32),
            np.asarray([[0.0, 0.0]], dtype=np.float64),
            ("u", "v", "p", "omega", "temperature"),
            ("dataset velocity", "dataset velocity", "dataset pressure", "dataset vorticity", "dataset temperature"),
            ("grid:0",),
            (),
        ),
        "interface": (
            np.zeros((2, 1, 2), dtype=np.float32),
            np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64),
            ("T_surface", "q_normal"),
            ("dataset temperature", "dataset heat-flux proxy"),
            (f"{module_id}:port:0",),
            (module_id,),
        ),
        "solid_temperature": (
            np.zeros((2, 1, 1), dtype=np.float32),
            np.asarray([[0.0, 0.0]], dtype=np.float64),
            ("temperature",),
            ("dataset temperature",),
            (f"{module_id}:solid:0",),
            (module_id,),
        ),
    }
    for role, (values, query, channels, units, query_ids, receiver_ids) in role_data.items():
        arrays[f"{role}_values"] = values
        arrays[f"{role}_valid_mask"] = np.ones_like(values, dtype=bool)
        arrays[f"{role}_query_features"] = query
        arrays[f"{role}_channel_names"] = np.asarray(channels, dtype="U")
        arrays[f"{role}_channel_units"] = np.asarray(units, dtype="U")
        arrays[f"{role}_query_ids"] = np.asarray(query_ids, dtype="U")
        arrays[f"{role}_receiver_module_ids"] = np.asarray(receiver_ids, dtype="U")

    npz_path = tmp_path / "family.npz"
    np.savez_compressed(npz_path, **arrays)
    npz_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "anchor_id": "anchor",
                "family_id": "family",
                "split": "train",
                "records": {"baseline": {"case_dir": "/raw/baseline"}},
            }
        ),
        encoding="utf-8",
    )
    return npz_path


def test_loader_exposes_full_checkpoint_native_thermal_context(tmp_path: Path) -> None:
    path = _write_atlas(tmp_path)

    stencil, _ = load_response_atlas_stencil(path)
    context = stencil.baseline.context

    assert set(context.values) == {
        "re",
        "u_in",
        "nu",
        "solid_alpha",
        "fluid_alpha",
        "solid_k",
        "fluid_k",
        "module_radius",
        "domain_length_x",
        "domain_length_y",
    }
    assert context.values["nu"] == pytest.approx(0.018)
    assert context.values["solid_alpha"] == pytest.approx(0.01)
    assert context.values["fluid_alpha"] == pytest.approx(0.02)
    assert context.values["solid_k"] == pytest.approx(1.0)
    assert context.values["fluid_k"] == pytest.approx(1.5)
    assert stencil.variants["i_plus"].context.values == context.values
    assert stencil.baseline.output is not None
    assert stencil.baseline.output.case_dir == "/raw/baseline"


def test_loader_rejects_inconsistent_or_nonfinite_native_context(tmp_path: Path) -> None:
    inconsistent = _write_atlas(tmp_path / "inconsistent", flow_nu=0.02, runtime_nu=0.018)
    with pytest.raises(ValueError, match="disagrees with recorded runtime nu"):
        load_response_atlas_stencil(inconsistent)

    nonfinite = _write_atlas(tmp_path / "nonfinite", fluid_alpha=float("nan"))
    with pytest.raises(ValueError, match="non-finite fields"):
        load_response_atlas_stencil(nonfinite)

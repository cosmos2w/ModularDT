"""Public observations and separate supervision from retained response solves.

Only the final-review baseline/heat-transfer-plus pair is accepted. Loading
does not run a model/solver or fit normalization. Full solved fields are never
returned as part of the public sample.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for source in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel/src", PROJECT_ROOT / "tools"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from channelthermal.interaction_evidence.response_atlas import _context
from thermal_campaign_heat_inference import sensor_panel

MATERIAL_NAMES = ("nu", "solid_alpha", "fluid_alpha", "solid_k", "fluid_k", "module_radius")
PAIR_LABELS = ("baseline", "heat_transfer_plus")


def _readonly(value, dtype=None):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class AtlasPublicObservation:
    """Known inputs, six observed temperatures and fourteen public query rows."""

    case_id: str
    public_sample: Mapping
    sensor_coordinates: np.ndarray
    sensor_names: tuple[str, ...]
    sensor_query_ids: tuple[str, ...]
    sensor_grid_rows: np.ndarray
    observed_rows: np.ndarray
    held_rows: np.ndarray
    observed_temperatures: np.ndarray
    temperature_unit: str
    public_total_heat: float
    source_module_ids: tuple[str, ...]
    source_id_to_slot: Mapping[str, int]
    metadata: Mapping


@dataclass(frozen=True)
class AtlasHeatSupervision:
    """Evaluation-only clean heat and held temperatures; never public inputs."""

    case_id: str
    heat: np.ndarray
    held_temperatures: np.ndarray
    held_sensor_names: tuple[str, ...]
    held_query_ids: tuple[str, ...]


def _json(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected saved metadata object: {path}")  # noqa: TRY004 - invalid saved file
    return value


def _group_geometry(arrays, prefix, module_ids, width):
    features = np.asarray(arrays[f"{prefix}_query_features"])
    receivers = np.asarray(arrays[f"{prefix}_receiver_module_ids"])
    if (features.ndim != 2 or features.shape[1] != width or len(receivers) != len(features)
            or not np.isfinite(features).all() or set(receivers.tolist()) != set(module_ids)):
        raise ValueError(f"Invalid saved {prefix} query geometry/module IDs")
    groups = [features[receivers == module_id] for module_id in module_ids]
    if not groups[0].size or any(not np.array_equal(groups[0], group) for group in groups[1:]):
        raise ValueError(f"Native shared local geometry is incompatible with saved {prefix} queries")
    return np.stack(groups)


def load_final_review_pair(npz_path: str | Path, *, module_capacity: int = 12):
    """Return ``((baseline_public, hidden), (plus_public, hidden))``.

    Source module order becomes the first active native slots. Public samples
    have no ``heat_powers``, ``steady_field`` or solved interface values. The
    caller must supply train-only normalization and separately consume hidden
    supervision. Raw per-solve configs are required to verify context, rather
    than assuming the family's baseline context also applied to its variant.
    """
    path = Path(npz_path).expanduser().resolve()
    metadata = _json(path.with_suffix(".json"))
    with np.load(path, allow_pickle=False) as arrays:
        labels = arrays["all_labels"].tolist()
        if any(labels.count(label) != 1 for label in PAIR_LABELS):
            raise ValueError("Saved atlas needs exactly one baseline and heat_transfer_plus")
        rows = [labels.index(label) for label in PAIR_LABELS]
        module_ids = tuple(str(value) for value in arrays["active_module_ids"])
        count = len(module_ids)
        if (count not in (3, 5, 7, 10) or len(set(module_ids)) != count
                or any(not value for value in module_ids)):
            raise ValueError("Expected distinct final-review source module IDs for M3/5/7/10")
        if isinstance(module_capacity, bool) or not isinstance(module_capacity, int) or module_capacity < count:
            raise ValueError("Native module capacity must cover all saved source IDs")
        if any(str(arrays["split"][row]) != "final_review" for row in rows):
            raise ValueError("Alternative observation pairs must use FINAL_REVIEW records")
        if metadata.get("split") != "final_review":
            raise ValueError("Adjacent atlas metadata must identify FINAL_REVIEW")
        family = _json_text(arrays["family_metadata_json"])
        family_id = metadata.get("family_id")
        if (not isinstance(family_id, str) or not family_id or family_id in (".", "..")
                or "/" in family_id or "\\" in family_id):
            raise ValueError("Saved family ID must be one nonempty path component")
        if (family.get("physical_family_id") != family_id
                or family.get("split") != "final_review"):
            raise ValueError("Saved family identity/partition disagrees between archive and JSON")
        context = dict(_context(family).values)
        if any(context[name] <= 0 for name in (*MATERIAL_NAMES, "domain_length_x", "domain_length_y")):
            raise ValueError("Archived material parameters and domain lengths must be positive")
        centers = np.asarray(arrays["module_centers_xy"])[rows]
        heat = np.asarray(arrays["heating"])[rows]
        if (centers.shape != (2, count, 2) or heat.shape != (2, count)
                or not np.isfinite(centers).all() or not np.isfinite(heat).all() or (heat < 0).any()):
            raise ValueError("Saved pair geometry/heat must be finite and heat nonnegative")
        if not np.array_equal(centers[0], centers[1]):
            raise ValueError("Alternative task must preserve exact saved geometry")
        if np.array_equal(heat[0], heat[1]):
            raise ValueError("Alternative task must change the saved heat allocation")
        # Exact saved-atom equality plus the actual frontend FP32 padded sum.
        if heat[0].astype(np.float64).sum() != heat[1].astype(np.float64).sum():
            raise ValueError("Alternative task must preserve exact saved total heat")
        padded_heat = np.pad(heat.astype(np.float32), ((0, 0), (0, module_capacity-count)))
        total = padded_heat.sum(-1)
        if total[0] != total[1] or not np.isfinite(total).all() or total[0] <= 0:
            raise ValueError("Alternative task must preserve positive native public total heat")

        config_paths = []
        configs = []
        for label, row in zip(PAIR_LABELS, rows):
            record = metadata.get("records", {}).get(label, {})
            if (str(arrays["source"][row]) != "reference_solver" or record.get("source") != "reference_solver"
                    or record.get("status") != "converged" or record.get("runtime", {}).get("converged") is not True):
                raise ValueError("Observation task needs a converged independent saved reference solve")
            config_path = Path(record.get("case_dir", "")) / "case_config.json"
            if not config_path.is_file():
                raise ValueError(f"Missing per-solve public-context provenance: {config_path}")
            config = _json(config_path)
            current = dict(_context({"case_config": config, "runtime": config.get("runtime", {})}).values)
            if (any(current[name] != context[name] for name in context)
                    or float(record["runtime"].get("nu", float("nan"))) != context["nu"]):
                raise ValueError("Saved pair must preserve material and operating context")
            layout = config.get("layout", {})
            if (not np.array_equal(np.asarray(layout.get("centers"), dtype=np.float32), centers[0])
                    or not np.array_equal(np.asarray(layout.get("heat_powers"), dtype=np.float32), heat[len(configs)])):
                raise ValueError("Per-solve source geometry/heat disagrees with archive source-ID order")
            config_paths.append(config_path)
            configs.append(config)

        grid = np.asarray(arrays["fluid_fields_query_features"])
        ny, nx = int(configs[0]["domain"]["ny"]), int(configs[0]["domain"]["nx"])
        if (ny <= 0 or nx <= 0 or grid.shape != (ny*nx, 2) or not np.isfinite(grid).all()
                or (ny, nx) != (int(configs[1]["domain"]["ny"]), int(configs[1]["domain"]["nx"]))):
            raise ValueError("Saved fluid query grid/config mismatch")
        x, y = grid[:, 0].reshape(ny, nx), grid[:, 1].reshape(ny, nx)
        if not (np.array_equal(x, np.broadcast_to(x[:1], x.shape))
                and np.array_equal(y, np.broadcast_to(y[:, :1], y.shape))
                and np.all(np.diff(x[0]) > 0) and np.all(np.diff(y[:, 0]) > 0)):
            raise ValueError("Saved fluid queries are not the native ordered rectangular grid")
        # Native archives contain cell-centre queries, not domain-boundary nodes.
        expected_x = ((np.arange(nx)+.5)*context["domain_length_x"]/nx).astype(np.float32)
        expected_y = ((np.arange(ny)+.5)*context["domain_length_y"]/ny).astype(np.float32)
        if not np.array_equal(x[0], expected_x) or not np.array_equal(y[:,0], expected_y):
            raise ValueError("Saved native cell-centre grid disagrees with archived public domain bounds")
        ports = _group_geometry(arrays, "interface", module_ids, 3)
        local = _group_geometry(arrays, "solid_temperature", module_ids, 2)[0]
        padded_centers = np.pad(centers[0].astype(np.float32), ((0, module_capacity-count), (0, 0)))
        present = np.arange(module_capacity) < count
        structure = MappingProxyType({
            "module_centers": _readonly(padded_centers), "module_present": _readonly(present, np.float32),
            "material_params": _readonly([context[name] for name in MATERIAL_NAMES], np.float32),
            **{name: _readonly([context[name]], np.float32) for name in
               ("re", "u_in", "domain_length_x", "domain_length_y")},
            "public_total_heat": float(total[0]),
        })
        public_sample = MappingProxyType({"structure": structure,
            "x_grid": _readonly(x), "y_grid": _readonly(y),
            "interface_condition": _readonly(np.pad(ports, ((0,module_capacity-count),(0,0),(0,0)))),
            "module_internal_query_points": _readonly(local)})
        sensor_coords, indices, names, observed, held = sensor_panel(public_sample)
        names = tuple(names)
        query_ids = tuple(str(value) for value in arrays["fluid_fields_query_ids"])
        if len(query_ids) != len(grid) or len(set(query_ids)) != len(grid):
            raise ValueError("Saved fluid query IDs must uniquely identify each stored grid row")
        sensor_ids = tuple(query_ids[index] for index in indices)
        if len(set(sensor_ids)) != 14 or not np.array_equal(sensor_coords, grid[indices].astype(np.float32)):
            raise ValueError("Named sensors must exactly select saved physical query coordinates")
        channels = arrays["fluid_fields_channel_names"].tolist()
        if channels.count("temperature") != 1 or "p" not in channels:
            raise ValueError("Saved fluid schema needs temperature and pressure channels")
        temperature = channels.index("temperature")
        fields = np.asarray(arrays["fluid_fields_values"])[rows]
        masks = np.asarray(arrays["fluid_fields_valid_mask"])[rows]
        if (fields.shape != (2, len(grid), len(channels)) or masks.shape != fields.shape
                or not masks[:, indices, temperature].all()
                or not np.isfinite(fields[:, indices, temperature]).all()):
            raise ValueError("Every named observed/held sensor must have a valid finite saved reference")
        pressure = channels.index("p")
        if not masks[:, indices[-2:], pressure].all() or not np.isfinite(fields[:, indices[-2:], pressure]).all():
            raise ValueError("Named inlet/outlet must select valid saved fluid pressure queries")
        temperatures = fields[:, indices, temperature]
        if np.array_equal(temperatures[0, observed], temperatures[1, observed]):
            raise ValueError("Alternative reference must change at least one observed temperature")
        unit = str(arrays["fluid_fields_channel_units"][temperature])
        mapping = MappingProxyType(dict(zip(module_ids, range(count))))
        result = []
        for pair_row, label in enumerate(PAIR_LABELS):
            case_id = f"{metadata['family_id']}:{label}"
            provenance = MappingProxyType({"source": "reference_solver", "partition": "final_review",
                "previously_exposed": True, "exposure_scope": "existing finite-response review cohorts",
                "family_id": metadata["family_id"], "variant": label,
                "atlas_npz": str(path),
                "atlas_json": str(path.with_suffix('.json')), "reference_case_config": str(config_paths[pair_row]),
                "reference_limit": "retained analytic/shared-grid benchmark; not new CFD validation"})
            public = AtlasPublicObservation(case_id, public_sample, _readonly(sensor_coords), names, sensor_ids,
                _readonly(indices), _readonly(observed), _readonly(held),
                _readonly(temperatures[pair_row, observed]), unit, float(total[0]), module_ids, mapping, provenance)
            hidden = AtlasHeatSupervision(case_id, _readonly(padded_heat[pair_row]),
                _readonly(temperatures[pair_row, held]), tuple(names[row] for row in held),
                tuple(sensor_ids[row] for row in held))
            result.append((public, hidden))
        return tuple(result)


def _json_text(array):
    value = json.loads(str(np.asarray(array).item()))
    if not isinstance(value, dict):
        raise ValueError("Atlas family metadata must be an object")  # noqa: TRY004 - invalid saved file
    return value

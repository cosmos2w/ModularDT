"""
Visualize and post-process a saved multi-cylinder case directory.

The script reads the scene-compatible `.npz` arrays produced by the simulation
script, generates selected frame plots, optionally writes a GIF, and extracts a
small set of physically meaningful quantities of interest (QoIs).

If you pass train/0001 or test/0001, it searches only inside that split.
If you pass just 0001, it searches in this order: Data_Saved/, then Data_Saved/train/, then Data_Saved/test/.

The same command can also visualize preprocessed cases under
Data_Saved/Processed_Inert_Dataset by adding `--dataset processed`. Processed
cases are plotted from `canonical_cycle.npz`, so the frame ids are phase-bin
indices rather than raw saved-frame ids.

python src/visualize_multicylinder_case.py --case_dir train/0160 --save-gif
python src/visualize_multicylinder_case.py --dataset processed --case_dir train/0160 --save-gif

"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import warnings

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from tqdm.auto import tqdm

from multicyl_common import (
    SimulationConfig,
    build_uniform_grid,
    config_from_dict,
    default_data_dir,
    periodic_offsets,
    resolve_data_path,
)


@dataclass
class CaseSource:
    """Resolved location and storage format for one visualizable case."""

    case_dir: Path
    dataset: str  # "raw" or "processed"


@dataclass
class LoadedCase:
    """In-memory metadata needed by plotting and QoI extraction."""

    source: CaseSource
    cfg: SimulationConfig
    frame_ids: List[int]
    frame_values: Dict[int, float]
    frame_value_name: str
    canonical_cycle: Optional[np.ndarray] = None
    channel_order: Optional[List[str]] = None
    output_dir: Optional[Path] = None

    @property
    def case_dir(self) -> Path:
        return self.source.case_dir

    @property
    def is_processed(self) -> bool:
        return self.source.dataset == "processed"

    @property
    def scene_dir(self) -> Path:
        return self.case_dir / "scene"


def default_processed_data_dir() -> Path:
    """Return the canonical processed inert dataset directory."""
    return default_data_dir() / "Processed_Inert_Dataset"


def _is_raw_case_dir(path: Path) -> bool:
    return (path / "case_config.json").exists()


def _is_processed_case_dir(path: Path) -> bool:
    return (path / "structure.json").exists() and (path / "canonical_cycle.npz").exists()


def _dedupe_paths(paths: Sequence[Path]) -> List[Path]:
    seen = set()
    deduped: List[Path] = []
    for path in paths:
        try:
            key = path.resolve()
        except FileNotFoundError:
            key = path.absolute()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _path_parts(case_arg: str) -> List[str]:
    return [part for part in Path(case_arg).expanduser().parts if part not in {"", "."}]


def _requested_split(case_arg: str, dataset: str) -> Optional[str]:
    parts = _path_parts(case_arg)
    if parts and parts[0] in {"train", "test"}:
        return parts[0]

    if dataset == "processed" and default_processed_data_dir().name in parts:
        idx = parts.index(default_processed_data_dir().name)
        if idx + 1 < len(parts) and parts[idx + 1] in {"train", "test"}:
            return parts[idx + 1]

    return None


def _case_token(case_arg: str) -> str:
    return Path(case_arg).expanduser().name.strip()


def _direct_candidates(case_arg: str, processed_root: Path) -> List[Path]:
    raw = Path(case_arg).expanduser()
    candidates: List[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(resolve_data_path(case_arg))
        candidates.append(processed_root / raw)
    return _dedupe_paths(candidates)


def _search_case_roots(dataset: str, requested_split: Optional[str], processed_root: Path) -> List[Path]:
    if dataset == "processed":
        root = processed_root
    else:
        root = default_data_dir().resolve()

    if requested_split:
        return [root / requested_split]
    return [root, root / "train", root / "test"]


def _matching_case_dirs(root: Path, token: str, dataset: str) -> List[Path]:
    if not root.exists():
        return []

    if token.isdigit():
        candidates = sorted(root.glob(f"case_{token}_*"))
    else:
        candidates = [root / token]

    predicate = _is_processed_case_dir if dataset == "processed" else _is_raw_case_dir
    return [path for path in candidates if path.is_dir() and predicate(path)]


def _resolve_from_dataset(case_arg: str, dataset: str, processed_root: Path) -> Optional[CaseSource]:
    token = _case_token(case_arg)
    requested_split = _requested_split(case_arg, dataset)

    for candidate in _direct_candidates(case_arg, processed_root):
        if dataset == "processed" and _is_processed_case_dir(candidate):
            return CaseSource(candidate.resolve(), "processed")
        if dataset == "raw" and _is_raw_case_dir(candidate):
            return CaseSource(candidate.resolve(), "raw")

    matches: List[Path] = []
    for root in _search_case_roots(dataset, requested_split, processed_root):
        matches.extend(_matching_case_dirs(root, token, dataset))

    if len(matches) == 1:
        return CaseSource(matches[0].resolve(), dataset)
    if len(matches) > 1:
        raise ValueError(
            f"Multiple {dataset} case directories matched '{case_arg}': "
            + ", ".join(path.name for path in matches)
        )
    return None


def resolve_case_source(case_arg: str, dataset: str, processed_root: Path) -> CaseSource:
    """Resolve a case argument to an actual raw or processed case directory.

    Accepted forms:
    - bare case id, e.g. `0001`
    - split-qualified case id, e.g. `train/0001` or `test/0001`
    - case directory name, e.g. `case_0001_20260417_142729_multicyl`
    - path relative to `Data_Saved/`
    - absolute path
    """
    processed_root = processed_root.expanduser().resolve()
    if dataset not in {"auto", "raw", "processed"}:
        raise ValueError("--dataset must be one of auto, raw, or processed.")

    parts = _path_parts(case_arg)
    mentions_processed_root = default_processed_data_dir().name in parts
    search_order = [dataset] if dataset != "auto" else (["processed", "raw"] if mentions_processed_root else ["raw", "processed"])

    for dataset_name in search_order:
        source = _resolve_from_dataset(case_arg, dataset_name, processed_root)
        if source is not None:
            return source

    warnings.warn(
        f"Could not find case '{case_arg}' in raw data under {default_data_dir()} "
        f"or processed data under {processed_root}.",
        stacklevel=2,
    )
    raise FileNotFoundError(
        f"Could not resolve case '{case_arg}'. "
        "Pass a case id like '0001', a split-qualified id like 'train/0001', "
        "a full case directory name, or an absolute path."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a saved multi-cylinder case.")
    parser.add_argument(
        "--case_dir",
        type=str,
        required=True,
        help=f"Case directory or case name under {default_data_dir()}.",
    )
    parser.add_argument(
        "--dataset",
        choices=["auto", "raw", "processed"],
        default="auto",
        help=(
            "Dataset layout to visualize. Use 'processed' for "
            f"{default_processed_data_dir()} canonical-cycle cases."
        ),
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=default_processed_data_dir(),
        help="Root directory containing processed train/ and test/ case folders.",
    )
    parser.add_argument(
        "--frames",
        type=str,
        default=None,
        help="Comma-separated frame/phase-bin indices to plot. Default: first, 25%%, 50%%, last.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots and QoI CSVs. Default: <case_dir>/plots, with a local fallback if read-only.",
    )
    parser.add_argument("--save-gif", action="store_true", help="Render a GIF over all frames/phase bins.")
    parser.add_argument(
        "--gif-field",
        choices=["vorticity", "speed", "temperature", "pressure"],
        default=None,
        help="Field to animate. Default: vorticity for inert, temperature for active.",
    )
    parser.add_argument("--gif-dpi", type=int, default=90, help="DPI used when rendering GIF frames.")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second for GIF output.")
    return parser.parse_args()


def load_raw_case_config(case_dir: Path) -> SimulationConfig:
    with (case_dir / "case_config.json").open("r", encoding="utf-8") as f:
        raw = json.load(f)
    raw_cfg = {k: v for k, v in raw.items() if k in {"mode", "domain", "flow", "thermal", "layout", "save"}}
    return config_from_dict(raw_cfg)


def load_processed_case_config(case_dir: Path) -> SimulationConfig:
    with (case_dir / "structure.json").open("r", encoding="utf-8") as f:
        structure = json.load(f)

    domain = dict(structure.get("domain", {}))
    if "cylinder_radius" in structure:
        domain["cylinder_radius"] = structure["cylinder_radius"]

    raw_cfg = {
        "mode": structure.get("mode", "inert"),
        "domain": domain,
        "flow": {"re": structure.get("re", 100.0)},
        "thermal": structure.get("thermal", {}),
        "layout": {
            "num_cylinders": structure.get("num_cylinders", len(structure.get("cylinder_centers", []) or [])),
            "centers": structure.get("cylinder_centers"),
            "heat_powers": structure.get("heat_powers"),
        },
        "save": {
            "case_id": structure.get("case_id", case_dir.name),
            "root_dir": str(case_dir.parent),
        },
    }
    return config_from_dict(raw_cfg)


def _read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_frame_times(case_dir: Path) -> Dict[int, float]:
    rows = _read_csv_rows(case_dir / "frame_index.csv")
    return {int(row["saved_frame"]): float(row["time"]) for row in rows}


def load_processed_phase_values(case_dir: Path, num_phase_bins: int) -> Dict[int, float]:
    with np.load(case_dir / "canonical_cycle.npz") as data:
        if "phase_bin_centers" in data:
            phase_centers = np.asarray(data["phase_bin_centers"], dtype=np.float64).reshape(-1)
        else:
            phase_centers = np.linspace(0.0, 1.0, num_phase_bins, endpoint=False, dtype=np.float64)

    if phase_centers.size != num_phase_bins:
        phase_centers = np.linspace(0.0, 1.0, num_phase_bins, endpoint=False, dtype=np.float64)
    return {idx: float(phase_centers[idx]) for idx in range(num_phase_bins)}


def decode_channel_order(values: np.ndarray) -> List[str]:
    decoded: List[str] = []
    for item in np.asarray(values).reshape(-1):
        if isinstance(item, bytes):
            decoded.append(item.decode("utf-8"))
        elif isinstance(item, np.generic):
            decoded.append(str(item.item()))
        else:
            decoded.append(str(item))
    return decoded


def infer_channel_order(field_dim: int) -> List[str]:
    if field_dim == 4:
        return ["u", "v", "p", "omega"]
    if field_dim == 5:
        return ["u", "v", "p", "omega", "temperature"]
    return [f"channel_{idx}" for idx in range(field_dim)]


def load_case(source: CaseSource) -> LoadedCase:
    if source.dataset == "raw":
        cfg = load_raw_case_config(source.case_dir)
        frame_ids = list_saved_frames(source.case_dir / "scene", property_name="velocity")
        return LoadedCase(
            source=source,
            cfg=cfg,
            frame_ids=frame_ids,
            frame_values=load_frame_times(source.case_dir),
            frame_value_name="time",
        )

    cfg = load_processed_case_config(source.case_dir)
    with np.load(source.case_dir / "canonical_cycle.npz") as data:
        if "canonical_cycle" not in data:
            raise KeyError(f"Missing canonical_cycle in {source.case_dir / 'canonical_cycle.npz'}")
        canonical_cycle = np.asarray(data["canonical_cycle"], dtype=np.float32)
        if canonical_cycle.ndim != 4:
            raise ValueError(f"canonical_cycle must have shape [phase, ny, nx, channels], got {canonical_cycle.shape}.")
        channel_order = (
            decode_channel_order(data["channel_order"])
            if "channel_order" in data
            else infer_channel_order(int(canonical_cycle.shape[-1]))
        )

    frame_ids = list(range(int(canonical_cycle.shape[0])))
    return LoadedCase(
        source=source,
        cfg=cfg,
        frame_ids=frame_ids,
        frame_values=load_processed_phase_values(source.case_dir, num_phase_bins=len(frame_ids)),
        frame_value_name="tau",
        canonical_cycle=canonical_cycle,
        channel_order=channel_order,
    )


def _load_npz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        last_key = list(data.keys())[-1]
        return np.array(data[last_key])


def _candidate_field_names(name: str) -> List[str]:
    return [name, name.lower(), name.capitalize()]


def _find_existing_field_path(scene_dir: Path, name: str, frame: int) -> Optional[Path]:
    for candidate in _candidate_field_names(name):
        path = scene_dir / f"{candidate}_{frame:06d}.npz"
        if path.exists():
            return path
    return None


def list_saved_frames(scene_dir: Path, property_name: str = "velocity") -> List[int]:
    files: List[Path] = []
    for candidate in _candidate_field_names(property_name):
        files = sorted(scene_dir.glob(f"{candidate}_*.npz"))
        if files:
            break
    return [int(f.stem.split("_")[-1]) for f in files]


def _normalize_scalar_field(arr: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    """Convert saved scalar fields to plotting/QoI shape (ny, nx)."""
    nx, ny = cfg.domain.nx, cfg.domain.ny

    if arr.shape == (ny, nx):
        return arr
    if arr.shape == (nx, ny):
        return arr.T
    if arr.shape == (ny + 1, nx + 1):
        return arr[:-1, :-1]
    if arr.shape == (nx + 1, ny + 1):
        return arr[:-1, :-1].T

    raise ValueError(
        f"Unsupported scalar field shape {arr.shape}. "
        f"Expected one of {(ny, nx)}, {(nx, ny)}, {(ny + 1, nx + 1)}, {(nx + 1, ny + 1)}."
    )


def _normalize_vector_field(arr: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    """Convert saved vector fields to plotting/QoI shape (ny, nx, 2)."""
    nx, ny = cfg.domain.nx, cfg.domain.ny

    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(f"Unsupported vector field shape {arr.shape}. Expected a trailing vector dimension of size 2.")
    if arr.shape[:2] == (ny, nx):
        return arr
    if arr.shape[:2] == (nx, ny):
        return np.transpose(arr, (1, 0, 2))

    raise ValueError(f"Unsupported vector field shape {arr.shape}. Expected {(ny, nx, 2)} or {(nx, ny, 2)}.")


def load_raw_fields_for_frame(scene_dir: Path, frame: int, cfg: SimulationConfig) -> Dict[str, Optional[np.ndarray]]:
    def maybe(name: str) -> Optional[np.ndarray]:
        path = _find_existing_field_path(scene_dir, name, frame)
        return _load_npz(path) if path is not None else None

    velocity = maybe("velocity")
    pressure = maybe("pressure")
    vorticity = maybe("vorticity")
    temperature = maybe("temperature")

    return {
        "velocity": _normalize_vector_field(velocity, cfg) if velocity is not None else None,
        "pressure": _normalize_scalar_field(pressure, cfg) if pressure is not None else None,
        "vorticity": _normalize_scalar_field(vorticity, cfg) if vorticity is not None else None,
        "temperature": _normalize_scalar_field(temperature, cfg) if temperature is not None else None,
    }


def _channel_index(channel_order: Sequence[str], aliases: Sequence[str]) -> Optional[int]:
    normalized = {name.lower().strip(): idx for idx, name in enumerate(channel_order)}
    for alias in aliases:
        idx = normalized.get(alias.lower().strip())
        if idx is not None:
            return idx
    return None


def _processed_scalar_channel(
    frame_tensor: np.ndarray,
    channel_order: Sequence[str],
    aliases: Sequence[str],
    cfg: SimulationConfig,
) -> Optional[np.ndarray]:
    idx = _channel_index(channel_order, aliases)
    if idx is None:
        return None
    return _normalize_scalar_field(frame_tensor[..., idx], cfg)


def load_processed_fields_for_frame(case: LoadedCase, frame: int) -> Dict[str, Optional[np.ndarray]]:
    if case.canonical_cycle is None:
        raise ValueError("Processed case is missing canonical_cycle data.")
    if case.channel_order is None:
        channel_order = infer_channel_order(int(case.canonical_cycle.shape[-1]))
    else:
        channel_order = case.channel_order

    frame_tensor = np.asarray(case.canonical_cycle[frame], dtype=np.float32)
    ux = _processed_scalar_channel(frame_tensor, channel_order, ["u", "ux", "velocity_x"], case.cfg)
    uy = _processed_scalar_channel(frame_tensor, channel_order, ["v", "uy", "velocity_y"], case.cfg)
    if ux is None or uy is None:
        raise ValueError(f"Processed channel_order must contain u/v velocity channels, got {channel_order}.")

    velocity = np.stack([ux, uy], axis=-1)
    pressure = _processed_scalar_channel(frame_tensor, channel_order, ["p", "pressure"], case.cfg)
    vorticity = _processed_scalar_channel(frame_tensor, channel_order, ["omega", "vorticity"], case.cfg)
    temperature = _processed_scalar_channel(frame_tensor, channel_order, ["temperature", "temp"], case.cfg)

    return {
        "velocity": velocity,
        "pressure": pressure,
        "vorticity": vorticity,
        "temperature": temperature,
    }


def load_fields_for_frame(case: LoadedCase, frame: int) -> Dict[str, Optional[np.ndarray]]:
    if case.is_processed:
        return load_processed_fields_for_frame(case, frame)
    return load_raw_fields_for_frame(case.scene_dir, frame, case.cfg)


def choose_default_frames(frame_ids: Sequence[int]) -> List[int]:
    if not frame_ids:
        raise ValueError("No frames found.")
    idxs = [0, int(round(0.25 * (len(frame_ids) - 1))), int(round(0.5 * (len(frame_ids) - 1))), len(frame_ids) - 1]
    return sorted({frame_ids[i] for i in idxs})


def parse_frame_request(frame_ids: Sequence[int], frame_arg: Optional[str]) -> List[int]:
    if frame_arg is None:
        return choose_default_frames(frame_ids)
    requested = sorted({int(v.strip()) for v in frame_arg.split(",") if v.strip()})
    missing = [i for i in requested if i not in frame_ids]
    if missing:
        raise ValueError(f"Requested frames do not exist: {missing}")
    return requested


def overlay_cylinders(ax, cfg: SimulationConfig) -> None:
    for cx, cy in cfg.layout.centers or []:
        ax.add_patch(Circle((cx, cy), cfg.domain.cylinder_radius, fill=False, color="k", lw=1.5))


def compute_qois(cfg: SimulationConfig, fields: Dict[str, Optional[np.ndarray]]) -> Dict[str, float]:
    velocity = fields["velocity"]
    if velocity is None:
        raise ValueError("Velocity field is required for QoI extraction.")

    ux = velocity[..., 0]
    uy = velocity[..., 1]
    speed = np.sqrt(ux * ux + uy * uy)
    qoi: Dict[str, float] = {
        "mean_speed": float(np.mean(speed)),
        "max_speed": float(np.max(speed)),
        "reverse_flow_fraction": float(np.mean(ux < 0.0)),
    }

    vort = fields["vorticity"]
    if vort is not None:
        qoi["mean_abs_vorticity"] = float(np.mean(np.abs(vort)))
        qoi["enstrophy"] = float(np.mean(vort * vort))

    temp = fields["temperature"]
    if temp is not None:
        qoi["mean_temperature"] = float(np.mean(temp))
        qoi["max_temperature"] = float(np.max(temp))

    xx, yy = build_uniform_grid(cfg)
    probe_dx = 3.0 * cfg.domain.cylinder_radius
    probe_dy = 2.0 * cfg.domain.cylinder_radius
    shell_width = max(cfg.domain.lx / cfg.domain.nx, cfg.domain.ly / cfg.domain.ny)

    for idx, (cx, cy) in enumerate(cfg.layout.centers or []):
        dx = periodic_offsets(xx - (cx + probe_dx), cfg.domain.lx)
        dy = periodic_offsets(yy - cy, cfg.domain.ly)
        wake_mask = (np.abs(dx) <= probe_dx) & (np.abs(dy) <= probe_dy)
        qoi[f"wake_deficit_cyl_{idx:02d}"] = float(cfg.flow.u_bulk - np.mean(ux[wake_mask]))

        if temp is not None:
            dx0 = periodic_offsets(xx - cx, cfg.domain.lx)
            dy0 = periodic_offsets(yy - cy, cfg.domain.ly)
            rr = np.sqrt(dx0 * dx0 + dy0 * dy0)
            annulus = (rr >= cfg.domain.cylinder_radius) & (rr <= cfg.domain.cylinder_radius + shell_width)
            qoi[f"surface_temperature_proxy_cyl_{idx:02d}"] = float(np.mean(temp[annulus]))

    return qoi


def frame_title_label(case: LoadedCase, frame: int) -> str:
    value = case.frame_values.get(frame)
    if case.is_processed:
        if value is None:
            return f"phase bin {frame:04d}"
        return f"phase bin {frame:04d} | tau={value:.3f}"
    return f"frame {frame:04d}"


def frame_output_stem(case: LoadedCase, frame: int) -> str:
    prefix = "phase" if case.is_processed else "frame"
    return f"{prefix}_{frame:04d}"


def make_qoi_row(case: LoadedCase, frame: int, qois: Dict[str, float]) -> Dict[str, Any]:
    if case.is_processed:
        row: Dict[str, Any] = {"phase_bin": int(frame)}
        row["tau"] = float(case.frame_values.get(frame, frame))
    else:
        row = {"saved_frame": int(frame)}
        row["time"] = float(case.frame_values.get(frame, frame))
    row.update(qois)
    return row


def plots_dir_for_case(case: LoadedCase) -> Path:
    if case.output_dir is not None:
        case.output_dir.mkdir(parents=True, exist_ok=True)
        return case.output_dir

    preferred = case.case_dir / "plots"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    except OSError as exc:
        fallback = Path.cwd() / "plots" / case.source.dataset / case.case_dir.name
        fallback.mkdir(parents=True, exist_ok=True)
        case.output_dir = fallback
        warnings.warn(
            f"Could not write plots to {preferred} ({exc}); using {fallback}.",
            stacklevel=2,
        )
        return fallback


def render_frame(case: LoadedCase, frame: int, fields: Dict[str, Optional[np.ndarray]], *, dpi: int = 150) -> Path:
    cfg = case.cfg
    plots_dir = plots_dir_for_case(case)

    velocity = fields["velocity"]
    if velocity is None:
        raise ValueError("Velocity field is required for plotting.")
    ux = velocity[..., 0]
    uy = velocity[..., 1]
    speed = np.sqrt(ux * ux + uy * uy)

    arrays: List[np.ndarray] = [speed]
    titles: List[str] = ["Speed"]
    cmaps: List[str] = ["viridis"]

    if fields["vorticity"] is not None:
        arrays.append(fields["vorticity"])
        titles.append("Vorticity")
        cmaps.append("RdBu_r")
    if cfg.thermal.enabled and fields["temperature"] is not None:
        arrays.append(fields["temperature"])
        titles.append("Temperature")
        cmaps.append("inferno")

    fig, axes = plt.subplots(1, len(arrays), figsize=(5.0 * len(arrays), 4.0), constrained_layout=True, dpi=dpi)
    if len(arrays) == 1:
        axes = [axes]

    extent = (0.0, cfg.domain.lx, 0.0, cfg.domain.ly)
    for ax, arr, title, cmap in zip(axes, arrays, titles, cmaps):
        im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, aspect="equal")
        overlay_cylinders(ax, cfg)
        ax.set_title(f"{title} | {frame_title_label(case, frame)}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_path = plots_dir / f"{frame_output_stem(case, frame)}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def render_gif(case: LoadedCase, frame_ids: Sequence[int], gif_field: str, fps: int, dpi: int) -> Path:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError("GIF rendering requires imageio. Install it or run without --save-gif.") from exc

    cfg = case.cfg
    plots_dir = plots_dir_for_case(case)
    gif_path = plots_dir / f"animation_{gif_field}.gif"
    frames_rgba: List[np.ndarray] = []
    extent = (0.0, cfg.domain.lx, 0.0, cfg.domain.ly)

    with tqdm(total=len(frame_ids), desc=f"Rendering GIF ({gif_field})", unit="frame", dynamic_ncols=True) as gif_bar:
        for frame in frame_ids:
            fields = load_fields_for_frame(case, frame)
            velocity = fields["velocity"]
            if velocity is None:
                raise ValueError("Velocity field is required for GIF rendering.")
            ux = velocity[..., 0]
            uy = velocity[..., 1]
            speed = np.sqrt(ux * ux + uy * uy)

            if gif_field == "speed":
                arr, cmap, title = speed, "viridis", "Speed"
            elif gif_field == "temperature":
                if fields["temperature"] is None:
                    raise ValueError("Temperature field not found; cannot render temperature GIF.")
                arr, cmap, title = fields["temperature"], "inferno", "Temperature"
            elif gif_field == "pressure":
                if fields["pressure"] is None:
                    raise ValueError("Pressure field not found; cannot render pressure GIF.")
                arr, cmap, title = fields["pressure"], "magma", "Pressure"
            else:
                if fields["vorticity"] is None:
                    raise ValueError("Vorticity field not found; cannot render vorticity GIF.")
                arr, cmap, title = fields["vorticity"], "RdBu_r", "Vorticity"

            fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=dpi, constrained_layout=True)
            im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, aspect="equal")
            overlay_cylinders(ax, cfg)
            ax.set_title(f"{title} | {frame_title_label(case, frame)}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.canvas.draw()
            frame_rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
            frames_rgba.append(frame_rgba)
            plt.close(fig)
            gif_bar.update(1)

    imageio.mimsave(gif_path, frames_rgba, duration=1.0 / max(fps, 1))
    tqdm.write(f"Saved GIF to: {gif_path}")
    return gif_path


def _row_fieldnames(rows: Sequence[Dict[str, Any]]) -> List[str]:
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def save_qoi_outputs(case: LoadedCase, qoi_rows: List[Dict[str, Any]], csv_name: str, plot_timeseries: bool) -> None:
    plots_dir = plots_dir_for_case(case)

    if not qoi_rows:
        return

    fieldnames = _row_fieldnames(qoi_rows)
    with (plots_dir / csv_name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(qoi_rows)

    if not plot_timeseries or len(qoi_rows) <= 1:
        return

    x_key = "time" if "time" in fieldnames else "tau" if "tau" in fieldnames else fieldnames[0]
    numeric_cols = [c for c in fieldnames if c not in {"saved_frame", "phase_bin", "time", "tau"}]
    if not numeric_cols:
        return

    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(8.0, max(2.0, 2.2 * len(numeric_cols))), constrained_layout=True)
    if len(numeric_cols) == 1:
        axes = [axes]
    x_values = np.asarray([float(row[x_key]) for row in qoi_rows], dtype=np.float64)
    for ax, col in zip(axes, numeric_cols):
        y_values = np.asarray([float(row[col]) if col in row else np.nan for row in qoi_rows], dtype=np.float64)
        ax.plot(x_values, y_values, lw=1.8)
        ax.set_xlabel(x_key)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "qoi_timeseries.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    source = resolve_case_source(args.case_dir, dataset=args.dataset, processed_root=args.processed_root)
    case = load_case(source)
    if args.output_dir is not None:
        case.output_dir = args.output_dir.expanduser().resolve()
    selected_frames = parse_frame_request(case.frame_ids, args.frames)

    selected_rows: List[Dict[str, Any]] = []
    for frame in selected_frames:
        fields = load_fields_for_frame(case, frame)
        render_frame(case, frame, fields)
        qois = compute_qois(case.cfg, fields)
        selected_rows.append(make_qoi_row(case, frame, qois))

    save_qoi_outputs(case, selected_rows, csv_name="qoi_selected_frames.csv", plot_timeseries=False)

    if args.save_gif:
        default_gif_field = "temperature" if case.cfg.thermal.enabled else "vorticity"
        gif_field = args.gif_field or default_gif_field
        render_gif(case, case.frame_ids, gif_field=gif_field, fps=args.fps, dpi=args.gif_dpi)

        full_rows: List[Dict[str, Any]] = []
        for frame in case.frame_ids:
            fields = load_fields_for_frame(case, frame)
            qois = compute_qois(case.cfg, fields)
            full_rows.append(make_qoi_row(case, frame, qois))
        save_qoi_outputs(case, full_rows, csv_name="qoi_all_frames.csv", plot_timeseries=True)


if __name__ == "__main__":
    main()

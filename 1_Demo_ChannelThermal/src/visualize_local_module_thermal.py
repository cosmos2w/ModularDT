"""Visualize raw and processed local module thermal cases.

The script mirrors ``visualize_channelthermal_case.py`` but targets the Stage-A
local conduction data contract:

* raw ``local_solution.npz`` or ``scene/frame_000000.npz`` cases
* processed ``Processed_LocalModule_Dataset/packed_dataset.h5`` cases

It produces plots for the local temperature field, disk mask, condition inputs
(``T_env`` and ``h``), and solved interface targets (``T_surface`` and
``q_normal``). This makes the input/target separation visible when checking
future Stage-A surrogate datasets.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - only needed for processed views
    h5py = None

import _bootstrap_imports  # noqa: F401
from channelthermal_common import read_json, resolve_data_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize raw or processed local module thermal cases.")
    parser.add_argument("--case-dir", type=Path, default=None, help="Raw local module case directory to visualize.")
    parser.add_argument("--processed-h5", type=Path, default=None, help="Processed local packed_dataset.h5 to visualize.")
    parser.add_argument("--case-id", type=str, default=None, help="Processed case id/key. Defaults to first case.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for PNG outputs.")
    return parser.parse_args()


def raw_case_candidates() -> List[Path]:
    """Return all raw local module case directories in deterministic order."""
    root = resolve_data_path("./Data_Saved/LocalModule_Raw")
    return sorted(
        path
        for path in root.glob("case_*")
        if path.is_dir() and (path / "case_config.json").exists() and ((path / "local_solution.npz").exists() or (path / "scene").exists())
    )


def latest_raw_case() -> Path:
    """Return the newest raw local module case."""
    root = resolve_data_path("./Data_Saved/LocalModule_Raw")
    candidates = raw_case_candidates()
    if not candidates:
        raise FileNotFoundError(f"No raw local module cases found under {root}.")
    return candidates[-1]


def normalize_case_id(case_id: str) -> str:
    """Normalize short numeric IDs to the local batch case-id convention."""
    cleaned = str(case_id).strip()
    if cleaned.isdigit():
        return f"local_{int(cleaned):04d}"
    if cleaned.startswith("case_local_"):
        parts = cleaned.split("_")
        if len(parts) >= 3 and parts[2].isdigit():
            return f"local_{int(parts[2]):04d}"
    return cleaned


def resolve_raw_case_id(case_id: str) -> Path:
    """Resolve a raw local case by config save.case_id or directory name.

    Examples that resolve to the same case:
        ``0010``, ``local_0010``, ``case_local_0010_...``.
    """
    candidates = raw_case_candidates()
    if not candidates:
        raise FileNotFoundError(f"No raw local module cases found under {resolve_data_path('./Data_Saved/LocalModule_Raw')}.")

    normalized = normalize_case_id(case_id)
    matches: List[Path] = []
    for path in candidates:
        cfg_payload = read_json(path / "case_config.json")
        saved_case_id = str(cfg_payload.get("save", {}).get("case_id", ""))
        if saved_case_id == normalized or path.name == case_id or path.name.startswith(f"case_{normalized}_"):
            matches.append(path)

    if not matches:
        available = [str(read_json(path / "case_config.json").get("save", {}).get("case_id", path.name)) for path in candidates[-20:]]
        raise FileNotFoundError(
            f"No raw local module case matched --case-id {case_id!r} (normalized to {normalized!r}). "
            f"Recent available IDs: {available}"
        )
    return matches[-1]


def resolve_case_dir(path: Path | None, case_id: str | None = None) -> Path:
    if path is None:
        if case_id is not None:
            return resolve_raw_case_id(case_id)
        return latest_raw_case()
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    direct = (Path.cwd() / expanded).resolve()
    if direct.exists():
        return direct
    return resolve_data_path(expanded)


def load_raw_payload(case_dir: Path) -> Dict[str, np.ndarray]:
    solution_path = case_dir / "local_solution.npz"
    if not solution_path.exists():
        solution_path = case_dir / "scene" / "frame_000000.npz"
    with np.load(solution_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def decode_h5_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def decode_names(values: np.ndarray) -> List[str]:
    return [decode_h5_string(value) for value in values]


def mask_temperature(temperature: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.asarray(temperature, dtype=np.float32).copy()
    masked[~mask.astype(bool)] = np.nan
    return masked


def finite_range(values: np.ndarray) -> Tuple[float, float]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.min(finite)), float(np.max(finite))


def plot_temperature_map(payload: Dict[str, np.ndarray], title: str, output_path: Path) -> None:
    """Plot the solved local temperature field on the disk."""
    local_x = payload["local_x"]
    local_y = payload["local_y"]
    temperature = mask_temperature(payload["temperature"], payload["disk_mask"])
    t_min, t_max = finite_range(temperature)
    fig, ax = plt.subplots(figsize=(5.5, 4.8), constrained_layout=True)
    image = ax.imshow(
        temperature,
        origin="lower",
        extent=[float(np.min(local_x)), float(np.max(local_x)), float(np.min(local_y)), float(np.max(local_y))],
        cmap="inferno",
    )
    circle = Circle((0.0, 0.0), 1.0, fill=False, color="white", linewidth=1.2)
    ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.set_xlabel("xi")
    ax.set_ylabel("eta")
    ax.set_title(f"{title}\nT range [{t_min:.3g}, {t_max:.3g}]")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="T")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=160)
    plt.close(fig)


def plot_disk_mask(payload: Dict[str, np.ndarray], title: str, output_path: Path) -> None:
    """Plot the binary disk mask used for internal query/target extraction."""
    local_x = payload["local_x"]
    local_y = payload["local_y"]
    fig, ax = plt.subplots(figsize=(4.8, 4.4), constrained_layout=True)
    image = ax.imshow(
        payload["disk_mask"],
        origin="lower",
        extent=[float(np.min(local_x)), float(np.max(local_x)), float(np.min(local_y)), float(np.max(local_y))],
        cmap="gray_r",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("xi")
    ax.set_ylabel("eta")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="inside disk")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=160)
    plt.close(fig)


def plot_boundary_conditions(payload: Dict[str, np.ndarray], title: str, output_path: Path) -> None:
    """Plot condition inputs separately from solved interface targets."""
    theta = payload["theta"]
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.0), sharex=True, constrained_layout=True)
    # Draw the target first and the known input second. In the polar solver the
    # two temperature curves can be close, and drawing T_surface last can hide
    # the T_env curve while still showing it in the legend.
    axes[0].plot(theta, payload["T_surface"], label="target: T_surface", color="tab:red", linewidth=1.8, alpha=0.78, zorder=2)
    axes[0].plot(
        theta,
        payload["T_env"],
        label="input: T_env",
        color="tab:blue",
        linestyle="--",
        linewidth=2.2,
        marker=".",
        markersize=3.0,
        markevery=max(1, len(theta) // 24),
        zorder=3,
    )
    axes[0].set_ylabel("temperature")
    t_values = np.concatenate([np.asarray(payload["T_env"]).reshape(-1), np.asarray(payload["T_surface"]).reshape(-1)])
    finite = t_values[np.isfinite(t_values)]
    if finite.size:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        pad = max(0.05 * (hi - lo), 1.0e-6)
        axes[0].set_ylim(lo - pad, hi + pad)
    axes[0].legend(loc="best")

    axes[1].plot(theta, payload["h"], color="tab:green")
    axes[1].set_ylabel("input: h")

    axes[2].plot(theta, payload["q_normal"], color="tab:purple")
    axes[2].axhline(0.0, color="0.25", linewidth=0.8)
    axes[2].set_xlabel("theta")
    axes[2].set_ylabel("target: q_normal")
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=160)
    plt.close(fig)


def plot_port_phase(payload: Dict[str, np.ndarray], title: str, output_path: Path) -> None:
    """Plot target heat flux around the circular port/interface."""
    theta = payload["theta"]
    fig, ax = plt.subplots(figsize=(5.5, 5.2), constrained_layout=True)
    scatter = ax.scatter(
        np.cos(theta),
        np.sin(theta),
        c=payload["q_normal"],
        s=45,
        cmap="coolwarm",
        edgecolors="black",
        linewidths=0.3,
    )
    circle = Circle((0.0, 0.0), 1.0, fill=False, color="0.2", linewidth=1.0)
    ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.set_xlabel("cos(theta)")
    ax.set_ylabel("sin(theta)")
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="q_normal")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=160)
    plt.close(fig)


def plot_condition_coefficients(payload: Dict[str, np.ndarray], title: str, output_path: Path) -> None:
    """Plot Fourier coefficients used to generate raw boundary conditions."""
    coeffs = payload.get("condition_coefficients")
    names = payload.get("condition_coefficient_names")
    if coeffs is None or names is None:
        return
    decoded = decode_names(names)
    if coeffs.size == 0:
        return
    modes = coeffs[:, decoded.index("mode")]
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.4), sharex=True, constrained_layout=True)
    axes[0].bar(modes - 0.16, coeffs[:, decoded.index("T_cos")], width=0.32, label="T_cos")
    axes[0].bar(modes + 0.16, coeffs[:, decoded.index("T_sin")], width=0.32, label="T_sin")
    axes[0].set_ylabel("T coeff")
    axes[0].legend(loc="best")
    axes[1].bar(modes - 0.16, coeffs[:, decoded.index("h_cos")], width=0.32, label="h_cos")
    axes[1].bar(modes + 0.16, coeffs[:, decoded.index("h_sin")], width=0.32, label="h_sin")
    axes[1].set_xlabel("mode")
    axes[1].set_ylabel("h coeff")
    axes[1].legend(loc="best")
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=160)
    plt.close(fig)


def plot_all(payload: Dict[str, np.ndarray], title_prefix: str, output_dir: Path, stem: str) -> None:
    plot_temperature_map(payload, f"{title_prefix}: internal temperature", output_dir / f"{stem}_temperature.png")
    plot_disk_mask(payload, f"{title_prefix}: disk mask", output_dir / f"{stem}_disk_mask.png")
    plot_boundary_conditions(payload, f"{title_prefix}: boundary and interface", output_dir / f"{stem}_boundary_interface.png")
    plot_port_phase(payload, f"{title_prefix}: interface flux around disk", output_dir / f"{stem}_port_flux.png")
    plot_condition_coefficients(payload, f"{title_prefix}: boundary modes", output_dir / f"{stem}_condition_coefficients.png")


def visualize_raw(case_dir_arg: Path | None, output_dir: Path | None, case_id_arg: str | None = None) -> None:
    """Visualize one raw local case directory."""
    case_dir = resolve_case_dir(case_dir_arg, case_id_arg)
    payload = load_raw_payload(case_dir)
    cfg_payload = read_json(case_dir / "case_config.json")
    case_id = str(cfg_payload.get("save", {}).get("case_id", case_dir.name))
    out_dir = output_dir.expanduser().resolve() if output_dir is not None else case_dir / "plots"
    plot_all(payload, f"Raw local case {case_id}", out_dir, f"raw_{case_id}")
    t_min, t_max = finite_range(mask_temperature(payload["temperature"], payload["disk_mask"]))
    print(f"Raw local temperature range: [{t_min:.6g}, {t_max:.6g}]")
    print(f"Saved raw local module visualizations to: {out_dir}")


def reconstruct_processed_payload(group, h5) -> Dict[str, np.ndarray]:
    """Reconstruct plotting arrays from a processed HDF5 case group."""
    local_grid = group["local_grid"][()]
    local_mask = group["local_mask"][()].astype(bool)
    temperature = np.zeros(local_mask.shape, dtype=np.float32)
    temperature[local_mask] = group["internal_temperature_targets"][()]

    port_tokens = group["port_tokens"][()]
    port_name_key = "port_input_feature_names" if "port_input_feature_names" in h5 else "port_feature_names"
    port_names = decode_names(h5[port_name_key][()])
    interface_target_names = decode_names(h5["interface_target_names"][()])
    interface_targets = group["interface_targets"][()]
    theta = port_tokens[:, port_names.index("theta")]
    t_env = port_tokens[:, port_names.index("T_env")]
    h_theta = port_tokens[:, port_names.index("h")]
    t_surface = interface_targets[:, interface_target_names.index("T_surface")]
    q_normal = interface_targets[:, interface_target_names.index("q_normal")]

    payload = {
        "local_x": local_grid[..., 0],
        "local_y": local_grid[..., 1],
        "temperature": temperature,
        "disk_mask": local_mask.astype(np.uint8),
        "theta": theta,
        "T_env": t_env,
        "h": h_theta,
        "T_surface": t_surface,
        "q_normal": q_normal,
    }
    return payload


def visualize_processed(processed_h5_arg: Path, case_id: str | None, output_dir: Path | None) -> None:
    """Visualize one processed local HDF5 case."""
    if h5py is None:
        raise ImportError("h5py is required for processed HDF5 visualization.")
    h5_path = processed_h5_arg.expanduser()
    if not h5_path.is_absolute():
        h5_path = resolve_data_path(h5_path)
    h5_path = h5_path.resolve()
    with h5py.File(h5_path, "r") as h5:
        cases_group = h5["cases"]
        key = normalize_case_id(case_id) if case_id is not None else sorted(cases_group.keys())[0]
        if key not in cases_group:
            raise KeyError(f"Case '{key}' not found in {h5_path}. Available: {sorted(cases_group.keys())}")
        payload = reconstruct_processed_payload(cases_group[key], h5)
    out_dir = output_dir.expanduser().resolve() if output_dir is not None else h5_path.parent / "plots"
    plot_all(payload, f"Processed local case {key}", out_dir, f"processed_{key}")
    t_min, t_max = finite_range(mask_temperature(payload["temperature"], payload["disk_mask"]))
    print(f"Processed local temperature range: [{t_min:.6g}, {t_max:.6g}]")
    print(f"Saved processed local module visualizations to: {out_dir}")


def main() -> int:
    args = parse_args()
    if args.case_dir is None and args.processed_h5 is None:
        visualize_raw(None, args.output_dir, args.case_id)
        return 0
    if args.case_dir is not None:
        visualize_raw(args.case_dir, args.output_dir, args.case_id)
    if args.processed_h5 is not None:
        visualize_processed(args.processed_h5, args.case_id, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

"""
Sample, verify, and rank inverse-design candidates.

python src/evaluate_inverse.py \
  --inverse-run Saved_Model_Inverse/CaseInv_inert_case0010_demo003_20260502_212153 \
  --checkpoint latest_model.pt \
  --target-json inverse_targets/balanced_low_enstrophy_valid_wake_demo.json \
  --n-samples 64 \
  --verify-top-k 16 \
  --save-verified-top-k 4 \
  --simulation-verify \
  --simulation-verify-top-k 1 \
  --device cuda:0


"""

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm.auto import tqdm

from inverse_kpi import (
    DEFAULT_KPI_NAMES,
    build_target_spec_vector,
    compute_cycle_kpis,
    score_candidate_kpis,
)
from model_inverse import HypergraphInverseDesignFlow, InverseModelConfig, periodic_min_distance
from train_inverse import (
    DEMO_ROOT,
    build_structure_from_centers,
    extract_forward_latent_targets,
    load_forward_model,
    make_eval_grid,
    predict_cycle_for_centers,
    read_json,
    resolve_demo_path,
    safe_torch_load,
    select_device,
    write_json,
)
from multicyl_common import SimulationConfig, config_from_dict, dataclass_to_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the inverse-design generator.")
    parser.add_argument("--inverse-run", type=str, required=True, help="Inverse run directory containing best_model.pt/latest_model.pt.")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt", help="Inverse checkpoint filename.")
    parser.add_argument("--target-json", type=str, default=None, help="Target KPI JSON file.")
    parser.add_argument("--n-samples", type=int, default=64, help="Number of inverse candidates to sample.")
    parser.add_argument("--verify-top-k", type=int, default=16, help="Number of sampled candidates to forward-verify.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device.")
    parser.add_argument("--seed", type=int, default=123, help="Sampling seed.")
    parser.add_argument("--n-steps", type=int, default=32, help="Inverse ODE integration steps.")
    parser.add_argument("--phase-bins", type=int, default=None, help="Forward verification phase bins.")
    parser.add_argument("--nx", type=int, default=None, help="Forward verification grid x cells.")
    parser.add_argument("--ny", type=int, default=None, help="Forward verification grid y cells.")
    parser.add_argument("--re", type=float, default=None, help="Simple target Re if --target-json is omitted.")
    parser.add_argument("--num-cylinders-min", type=int, default=None, help="Simple count lower bound.")
    parser.add_argument("--num-cylinders-max", type=int, default=None, help="Simple count upper bound.")
    parser.add_argument("--min-center-distance", type=float, default=None, help="Simple geometry preference.")
    parser.add_argument(
        "--kpi",
        action="append",
        default=[],
        help="Simple KPI target, e.g. enstrophy:range:0.08:0.16 or pressure_range:max:0.08.",
    )
    parser.add_argument("--refine-top-k", type=int, default=0, help="Reserved for optional post-sampling refinement; default disabled.")
    parser.add_argument("--refine-steps", type=int, default=0, help="Reserved for optional post-sampling refinement; default disabled.")
    parser.add_argument("--forward-backend", choices=["deterministic", "generative"], default=None, help="Forward verifier backend.")
    parser.add_argument("--generative-run", type=str, default=None, help="Stage-2 generative forward verifier run directory.")
    parser.add_argument("--generative-checkpoint", type=str, default=None, help="Stage-2 generative forward verifier checkpoint filename.")
    parser.add_argument("--generative-num-samples", type=int, default=None, help="Number of generative verifier samples per candidate.")
    parser.add_argument("--generative-n-steps", type=int, default=None, help="Generative verifier rectified-flow ODE steps.")
    parser.add_argument("--generative-ode-solver", choices=["euler", "heun"], default=None, help="Generative verifier ODE solver.")
    parser.add_argument("--uncertainty-penalty-weight", type=float, default=None, help="Weight for KPI uncertainty penalty in generative verification.")
    parser.add_argument("--prefilter-diversity", action="store_true", help="Prefer candidates with broader layout spread before forward verification.")
    parser.add_argument("--prefilter-min-x-span", type=float, default=None, help="Minimum preferred physical x-span before verification.")
    parser.add_argument("--prefilter-min-y-span", type=float, default=None, help="Minimum preferred physical y-span before verification.")
    parser.add_argument("--prefilter-cluster-penalty-weight", type=float, default=0.25, help="Weight for cluster penalty in pre-verification ranking.")
    parser.add_argument(
        "--save-verified-top-k",
        type=int,
        default=4,
        help="Number of ranked forward-model verified candidates to save with cycle data and quicklook visualizations.",
    )
    parser.add_argument(
        "--save-all-sampled-designs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save one lightweight JSON design file for every raw sampled candidate. By default only selected verified candidates get subdirectories.",
    )
    parser.add_argument(
        "--simulation-verify",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the real PhiFlow forward simulator for selected ranked candidates and compare simulated KPIs with generated KPIs.",
    )
    parser.add_argument("--simulation-verify-top-k", type=int, default=1, help="Number of ranked, model-verified candidates to simulate when --simulation-verify is enabled.")
    parser.add_argument("--simulation-config-json", type=str, default=None, help="Optional base simulator config JSON. Candidate centers/Re/output paths are still overridden.")
    parser.add_argument("--simulation-mode", choices=["inert", "active"], default="inert", help="Simulator mode for real forward verification.")
    parser.add_argument("--simulation-device", choices=["cpu", "gpu"], default=None, help="Simulator runtime device. Defaults to gpu when --device is CUDA, otherwise cpu.")
    parser.add_argument("--simulation-gpu-id", type=int, default=None, help="GPU id for real simulator verification.")
    parser.add_argument("--simulation-preprocess-device", type=str, default=None, help="Torch device used by preprocessing after simulation. Defaults to --device.")
    parser.add_argument("--simulation-nx", type=int, default=None, help="Override simulator grid nx for real verification.")
    parser.add_argument("--simulation-ny", type=int, default=None, help="Override simulator grid ny for real verification.")
    parser.add_argument("--simulation-phase-bins", type=int, default=None, help="Preprocessing phase bins for real simulation verification. Defaults to --phase-bins/effective verifier bins.")
    parser.add_argument("--simulation-warmup-cycles", type=float, default=None, help="Override simulator warmup cycles for real verification.")
    parser.add_argument("--simulation-save-cycles", type=float, default=None, help="Override simulator saved cycles for real verification.")
    parser.add_argument("--simulation-frames-per-cycle", type=int, default=None, help="Override simulator saved frames per estimated shedding cycle.")
    parser.add_argument("--simulation-dt", type=float, default=None, help="Override simulator time step for real verification.")
    return parser.parse_args()


def current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def progress_enabled() -> bool:
    return sys.stdout.isatty()


def _cuda_device_index(device_arg: str) -> int:
    text = str(device_arg or "").strip().lower()
    if text.startswith("cuda:"):
        try:
            return max(0, int(text.split(":", 1)[1]))
        except ValueError:
            return 0
    return 0


def _candidate_output_dir(candidate_dirs: Mapping[int, Path], candidate: Mapping[str, Any], fallback: Path) -> Path:
    sample_idx = int(candidate.get("sample_index", -1))
    return candidate_dirs.get(sample_idx, fallback)


def write_candidate_snapshot(candidate: Mapping[str, Any], out_dir: Path, name: str = "candidate_result.json") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / name, json_safe(candidate))


def initialize_candidate_dirs(candidates: Sequence[Mapping[str, Any]], out_dir: Path) -> Dict[int, Path]:
    root = out_dir / "candidates"
    root.mkdir(parents=True, exist_ok=True)
    dirs: Dict[int, Path] = {}
    for candidate in candidates:
        sample_idx = int(candidate.get("sample_index", len(dirs)))
        rank = candidate.get("rank", "")
        prefix = f"rank_{int(rank):03d}_" if str(rank) != "" else ""
        cand_dir = root / f"{prefix}sample_{sample_idx:03d}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        dirs[sample_idx] = cand_dir
        write_candidate_snapshot(candidate, cand_dir, name="candidate_design.json")
    return dirs


def resolve_simulation_config_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in (Path.cwd(), DEMO_ROOT, DEMO_ROOT / "Configs"):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (DEMO_ROOT / "Configs" / path).resolve()


def _run_logged_subprocess(
    cmd: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    label: str,
    echo_markers: Sequence[str],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_file.write(line)
            text = line.strip()
            if text and any(marker in text for marker in echo_markers):
                print(f"[{label}] {text}")
        return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"{label} failed with exit code {return_code}. See log: {log_path}")


def _load_simulation_base_config(args: argparse.Namespace) -> SimulationConfig:
    if args.simulation_config_json:
        config_path = resolve_simulation_config_path(args.simulation_config_json)
        with config_path.open("r", encoding="utf-8") as f:
            return config_from_dict(json.load(f))
    return SimulationConfig().finalize()


def write_simulation_config_for_candidate(
    candidate: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    raw_root: Path,
    re_value: float,
    lx: float,
    ly: float,
) -> Path:
    cfg = _load_simulation_base_config(args)
    centers = np.asarray(candidate["centers"], dtype=np.float32).reshape(-1, 2)
    cfg.mode = str(args.simulation_mode)
    cfg.domain.lx = float(lx)
    cfg.domain.ly = float(ly)
    if args.simulation_nx is not None:
        cfg.domain.nx = int(args.simulation_nx)
    if args.simulation_ny is not None:
        cfg.domain.ny = int(args.simulation_ny)
    cfg.flow.re = float(re_value)
    if args.simulation_warmup_cycles is not None:
        cfg.flow.warmup_cycles = float(args.simulation_warmup_cycles)
    if args.simulation_save_cycles is not None:
        cfg.flow.save_cycles = float(args.simulation_save_cycles)
    if args.simulation_frames_per_cycle is not None:
        cfg.flow.frames_per_cycle = int(args.simulation_frames_per_cycle)
    if args.simulation_dt is not None:
        cfg.flow.dt = float(args.simulation_dt)
    cfg.layout.centers = centers.astype(float).tolist()
    cfg.layout.num_cylinders = int(centers.shape[0])
    cfg.layout.heat_powers = None
    cfg.save.root_dir = str(raw_root)
    cfg.save.case_id = f"inv_s{int(candidate.get('sample_index', 0)):03d}"
    cfg.save.tag = "simulation_verify"
    sim_device = args.simulation_device or ("gpu" if str(args.device).lower().startswith("cuda") else "cpu")
    cfg.execution.device = str(sim_device)
    cfg.execution.gpu_id = int(args.simulation_gpu_id if args.simulation_gpu_id is not None else _cuda_device_index(str(args.device)))
    cfg = cfg.finalize()
    payload = dataclass_to_dict(cfg)
    config_path = raw_root.parent / "simulation_config.json"
    write_json(config_path, json_safe(payload))
    return config_path


def _find_single_case_dir(raw_root: Path) -> Path:
    candidates = sorted([p for p in raw_root.iterdir() if p.is_dir() and (p / "case_config.json").exists()])
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one simulated case under {raw_root}, found {len(candidates)}.")
    return candidates[0]


def _load_processed_cycle(processed_root: Path, case_dir: Path) -> Tuple[np.ndarray, List[str]]:
    cycle_path = processed_root / case_dir.name / "canonical_cycle.npz"
    if not cycle_path.exists():
        raise FileNotFoundError(f"Preprocessed canonical cycle not found: {cycle_path}")
    with np.load(cycle_path, allow_pickle=True) as data:
        cycle = np.asarray(data["canonical_cycle"], dtype=np.float32)
        order_arr = np.asarray(data["channel_order"]) if "channel_order" in data.files else np.asarray(["u", "v", "p", "omega"])
        channel_order = [str(v) for v in order_arr.reshape(-1)]
    return cycle, channel_order


def _kpi_comparison(generated_kpis: Mapping[str, Any], simulation_kpis: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    comparison: Dict[str, Dict[str, float]] = {}
    for name in sorted(set(generated_kpis.keys()) | set(simulation_kpis.keys())):
        try:
            generated = float(generated_kpis[name])
            simulated = float(simulation_kpis[name])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(generated) and math.isfinite(simulated)):
            continue
        comparison[name] = {
            "generated": generated,
            "simulation": simulated,
            "abs_delta": abs(generated - simulated),
            "rel_delta": abs(generated - simulated) / max(abs(simulated), 1.0e-12),
        }
    return comparison


def write_kpi_comparison_csv(comparison: Mapping[str, Mapping[str, float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["kpi", "ml_prediction", "simulation", "abs_delta", "rel_delta"])
        writer.writeheader()
        for name, row in comparison.items():
            writer.writerow(
                {
                    "kpi": name,
                    "ml_prediction": row.get("generated", ""),
                    "simulation": row.get("simulation", ""),
                    "abs_delta": row.get("abs_delta", ""),
                    "rel_delta": row.get("rel_delta", ""),
                }
            )


def plot_kpi_ml_vs_simulation(
    comparison: Mapping[str, Mapping[str, float]],
    out_path: Path,
    *,
    target_payload: Mapping[str, Any],
) -> None:
    target_names = [str(name) for name in (target_payload.get("kpis") or {}).keys()]
    names = [name for name in target_names if name in comparison]
    if not names:
        names = [name for name in DEFAULT_KPI_NAMES if name in comparison]
    if not names:
        names = list(comparison.keys())[:10]
    if not names:
        return

    ml_vals = [float(comparison[name]["generated"]) for name in names]
    sim_vals = [float(comparison[name]["simulation"]) for name in names]
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8.0, 1.25 * len(names)), 4.8), dpi=150)
    ax.bar(x - 0.5 * width, ml_vals, width=width, label="ML prediction", color="#4c78a8")
    ax.bar(x + 0.5 * width, sim_vals, width=width, label="Simulation", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("KPI value")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_ml_simulation_field_comparison(
    ml_cycle: np.ndarray,
    sim_cycle: np.ndarray,
    centers: np.ndarray,
    out_path: Path,
    *,
    ml_channel_order: Sequence[str],
    sim_channel_order: Sequence[str],
    lx: float,
    ly: float,
) -> None:
    ml_names = [str(name).lower() for name in ml_channel_order]
    sim_names = [str(name).lower() for name in sim_channel_order]
    channel_name = "omega" if "omega" in ml_names and "omega" in sim_names else (ml_names[0] if ml_names else "ch0")
    ml_idx = ml_names.index(channel_name) if channel_name in ml_names else 0
    sim_idx = sim_names.index(channel_name) if channel_name in sim_names else 0
    ml_field = np.asarray(ml_cycle[0, ..., ml_idx], dtype=np.float32)
    sim_field = np.asarray(sim_cycle[0, ..., sim_idx], dtype=np.float32)
    diff_field = sim_field - ml_field if sim_field.shape == ml_field.shape else None
    finite_stack = [ml_field[np.isfinite(ml_field)], sim_field[np.isfinite(sim_field)]]
    combined = np.concatenate([arr.reshape(-1) for arr in finite_stack if arr.size > 0])
    if combined.size:
        vmin, vmax = _field_color_limits(combined, channel_name)
    else:
        vmin, vmax = None, None

    cols = 3 if diff_field is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(5.0 * cols, 4.0), dpi=150, constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    extent = (0.0, float(lx), 0.0, float(ly))
    for ax, field, title in ((axes_arr[0], ml_field, "ML prediction"), (axes_arr[1], sim_field, "Simulation")):
        im = ax.imshow(field, origin="lower", extent=extent, cmap=channel_cmap(channel_name), vmin=vmin, vmax=vmax, aspect="equal")
        overlay_cylinders(ax, centers, linewidth=1.0)
        ax.set_title(f"{title} {channel_name}")
        ax.set_xlim(0, lx)
        ax.set_ylim(0, ly)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    if diff_field is not None:
        dv = float(np.percentile(np.abs(diff_field[np.isfinite(diff_field)]), 99.0)) if np.any(np.isfinite(diff_field)) else 1.0
        dv = max(dv, 1.0e-12)
        im = axes_arr[2].imshow(diff_field, origin="lower", extent=extent, cmap="RdBu_r", vmin=-dv, vmax=dv, aspect="equal")
        overlay_cylinders(axes_arr[2], centers, linewidth=1.0)
        axes_arr[2].set_title(f"Simulation - ML {channel_name}")
        axes_arr[2].set_xlim(0, lx)
        axes_arr[2].set_ylim(0, ly)
        fig.colorbar(im, ax=axes_arr[2], fraction=0.046, pad=0.03)
    fig.savefig(out_path)
    plt.close(fig)


def run_simulation_verification(
    candidates: Sequence[Dict[str, Any]],
    *,
    args: argparse.Namespace,
    out_dir: Path,
    candidate_dirs: Mapping[int, Path],
    target_spec: Mapping[str, Any],
    target_payload: Mapping[str, Any],
    re_value: float,
    lx: float,
    ly: float,
    phase_bins: int,
) -> None:
    if not args.simulation_verify:
        return
    sim_k = min(max(int(args.simulation_verify_top_k), 0), len(candidates))
    if sim_k <= 0:
        print("[simulation] requested simulation verification, but no verified candidates are available.")
        return

    print(f"[simulation] running real forward verification for top {sim_k} candidate(s).")
    bar = tqdm(candidates[:sim_k], desc="Simulation verification", unit="case", disable=not progress_enabled())
    for sim_rank, candidate in enumerate(bar):
        cand_dir = _candidate_output_dir(candidate_dirs, candidate, out_dir / f"simulation_candidate_{sim_rank:03d}")
        cand_dir.mkdir(parents=True, exist_ok=True)
        if progress_enabled():
            bar.set_postfix_str(f"sample={int(candidate.get('sample_index', sim_rank)):03d}")
        print(f"[simulation] candidate rank={candidate.get('rank', sim_rank)} sample={candidate.get('sample_index', sim_rank)}: preparing real simulator run.")

        raw_root = cand_dir / "simulation_raw"
        processed_root = cand_dir / "simulation_processed"
        raw_root.mkdir(parents=True, exist_ok=True)
        processed_root.mkdir(parents=True, exist_ok=True)
        config_path = write_simulation_config_for_candidate(
            candidate,
            args=args,
            raw_root=raw_root,
            re_value=re_value,
            lx=lx,
            ly=ly,
        )

        sim_log = cand_dir / "simulation.log"
        sim_runner = (
            "import json, sys; "
            "from pathlib import Path; "
            "sys.path.insert(0, 'src'); "
            "from multicyl_common import config_from_dict; "
            "from simulate_multicylinder_phiflow import run_case; "
            "cfg = config_from_dict(json.load(open(sys.argv[1], 'r', encoding='utf-8'))); "
            "print(f'Prepared configuration: case_id={cfg.save.case_id}, mode={cfg.mode}, device={cfg.execution.device}, gpu_id={cfg.execution.gpu_id}, cylinders={cfg.layout.num_cylinders}, Re={cfg.flow.re}'); "
            "case_dir = run_case(cfg); "
            "print(f'Simulation complete. Saved case to: {case_dir}')"
        )
        _run_logged_subprocess(
            [sys.executable, "-c", sim_runner, str(config_path)],
            cwd=DEMO_ROOT,
            log_path=sim_log,
            label="simulation",
            echo_markers=(
                "Prepared configuration",
                "Starting simulation",
                "Created case directory",
                "Runtime summary",
                "Simulation complete",
            ),
        )
        case_dir = _find_single_case_dir(raw_root)
        print(f"[simulation] candidate sample={candidate.get('sample_index', sim_rank)}: preprocessing simulated frames.")
        preprocess_phase_bins = int(args.simulation_phase_bins or phase_bins)
        preprocess_device = str(args.simulation_preprocess_device or args.device)
        preprocess_log = cand_dir / "simulation_preprocess.log"
        _run_logged_subprocess(
            [
                sys.executable,
                "src/preprocess_multicyl_dataset.py",
                "--input-root",
                str(raw_root),
                "--output-root",
                str(processed_root),
                "--device",
                preprocess_device,
                "--phase-bins",
                str(preprocess_phase_bins),
                "--save-cycles",
                "1",
                "--points-per-phase-bin",
                "0",
                "--sampling-mode",
                "uniform",
                "--save-full-canonical-cycles",
            ],
            cwd=DEMO_ROOT,
            log_path=preprocess_log,
            label="preprocess",
            echo_markers=(
                "[INFO] Using torch device",
                "[INFO] Discovered",
                "[INFO] Loaded",
                "Canonical cycle method",
                "Saving processed outputs",
                "Finished.",
                "Preprocessing finished",
                "Wrote packed dataset",
            ),
        )

        sim_cycle, sim_channel_order = _load_processed_cycle(processed_root, case_dir)
        sim_kpis = compute_cycle_kpis(sim_cycle, x_grid=None, y_grid=None, channel_order=sim_channel_order, domain={"lx": lx, "ly": ly})
        sim_kpis["num_cylinders"] = int(candidate.get("num_cylinders", candidate.get("count", 0)))
        sim_kpis["min_center_distance"] = float(candidate.get("min_pair_distance", 0.0))
        sim_kpis["x_span"] = float(candidate.get("x_span", 0.0))
        sim_kpis["y_span"] = float(candidate.get("y_span", 0.0))
        sim_kpis["valid"] = bool(candidate.get("validity", {}).get("valid", True))
        generated_kpis = candidate.get("kpis", {}) if isinstance(candidate.get("kpis", {}), Mapping) else {}
        comparison = _kpi_comparison(generated_kpis, sim_kpis)
        sim_score = score_candidate_kpis(sim_kpis, target_spec)
        generated_score = float(candidate.get("score", float("nan")))
        candidate["simulation_verified"] = True
        candidate["simulation_verification"] = {
            "case_dir": str(case_dir),
            "processed_root": str(processed_root),
            "simulation_log": str(sim_log),
            "preprocess_log": str(preprocess_log),
            "phase_bins": preprocess_phase_bins,
            "channel_order": sim_channel_order,
            "cycle_shape": list(sim_cycle.shape),
            "generated_kpis": generated_kpis,
            "ground_truth_kpis": sim_kpis,
            "kpi_comparison": comparison,
            "generated_score": generated_score,
            "ground_truth_score": float(sim_score["total_score"]),
            "ground_truth_per_kpi_errors": sim_score.get("per_kpi_errors", {}),
            "score_delta": float(sim_score["total_score"] - generated_score) if math.isfinite(generated_score) else None,
        }
        np.savez_compressed(
            cand_dir / "simulation_canonical_cycle.npz",
            canonical_cycle=sim_cycle.astype(np.float32),
            channel_order=np.asarray(sim_channel_order),
        )
        centers = np.asarray(candidate["centers"], dtype=np.float32).reshape(-1, 2)
        plot_candidate_flow(sim_cycle, centers, cand_dir / "simulation_flow.png", channel_order=sim_channel_order, lx=lx, ly=ly)
        try_save_cycle_gif(sim_cycle, cand_dir / "simulation_cycle.gif", sim_channel_order, centers, lx=lx, ly=ly)
        plot_kpi_ml_vs_simulation(comparison, cand_dir / "simulation_kpi_comparison.png", target_payload=target_payload)
        write_kpi_comparison_csv(comparison, cand_dir / "simulation_kpi_comparison.csv")
        ml_cycle_path = cand_dir / "generated_verifier_cycle.npz"
        if ml_cycle_path.exists():
            with np.load(ml_cycle_path, allow_pickle=True) as data:
                ml_cycle = np.asarray(data["cycle_mean"], dtype=np.float32)
                ml_order = [str(v) for v in np.asarray(data["channel_order"]).reshape(-1)] if "channel_order" in data.files else ["u", "v", "p", "omega"]
            plot_ml_simulation_field_comparison(
                ml_cycle,
                sim_cycle,
                centers,
                cand_dir / "ml_vs_simulation_field.png",
                ml_channel_order=ml_order,
                sim_channel_order=sim_channel_order,
                lx=lx,
                ly=ly,
            )
        write_json(cand_dir / "simulation_verification.json", json_safe(candidate["simulation_verification"]))
        write_candidate_snapshot(candidate, cand_dir)
        print(
            "[simulation] candidate "
            f"sample={candidate.get('sample_index', sim_rank)}: "
            f"generated_score={generated_score:.4e}, ground_truth_score={float(sim_score['total_score']):.4e}."
        )


def parse_simple_kpi(entries: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for entry in entries:
        pieces = entry.split(":")
        if len(pieces) < 2:
            raise ValueError(f"Invalid --kpi entry {entry!r}; expected name:mode:values.")
        name, mode = pieces[0], pieces[1].lower()
        if mode == "range":
            if len(pieces) < 4:
                raise ValueError(f"Range KPI {entry!r} needs low and high.")
            out[name] = {"mode": "range", "low": float(pieces[2]), "high": float(pieces[3]), "weight": 1.0}
        elif mode in {"max", "upper", "at_most"}:
            if len(pieces) < 3:
                raise ValueError(f"Max KPI {entry!r} needs high.")
            out[name] = {"mode": "max", "high": float(pieces[2]), "weight": 1.0}
        elif mode in {"min", "lower", "at_least"}:
            if len(pieces) < 3:
                raise ValueError(f"Min KPI {entry!r} needs low.")
            out[name] = {"mode": "min", "low": float(pieces[2]), "weight": 1.0}
        elif mode in {"minimize", "maximize"}:
            out[name] = {"mode": mode, "weight": float(pieces[2]) if len(pieces) >= 3 else 1.0}
        else:
            if len(pieces) < 3:
                raise ValueError(f"Exact KPI {entry!r} needs value.")
            out[name] = {"mode": "exact", "value": float(pieces[2]), "weight": 1.0}
    return out


def load_target_payload(args: argparse.Namespace) -> Dict[str, Any]:
    if args.target_json:
        path = Path(args.target_json).expanduser()
        if not path.is_absolute():
            local = Path.cwd() / path
            path = local if local.exists() else DEMO_ROOT / path
        return read_json(path.resolve())
    return {
        "Re": 100.0 if args.re is None else float(args.re),
        "num_cylinders_min": args.num_cylinders_min,
        "num_cylinders_max": args.num_cylinders_max,
        "kpis": parse_simple_kpi(args.kpi),
        "preferences": {"min_center_distance": args.min_center_distance},
    }


GEN_FORWARD_UNAVAILABLE = (
    "Generative stage-2 forward verifier is unavailable. "
    "Use backend=deterministic or provide a valid stage-2 checkpoint."
)


def channel_cmap(name: str) -> str:
    return {
        "u": "coolwarm",
        "v": "coolwarm",
        "p": "magma",
        "omega": "RdBu_r",
        "temperature": "inferno",
    }.get(str(name).lower(), "coolwarm")


def _field_color_limits(values: np.ndarray, name: str) -> Tuple[Optional[float], Optional[float]]:
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, None
    name = str(name).lower()
    if name in {"u", "v", "omega"}:
        vmax = float(np.percentile(np.abs(finite), 99.0))
        if not math.isfinite(vmax) or vmax <= 1.0e-12:
            vmax = float(np.max(np.abs(finite)))
        return -vmax, vmax
    vmin = float(np.percentile(finite, 1.0))
    vmax = float(np.percentile(finite, 99.0))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or abs(vmax - vmin) <= 1.0e-12:
        return float(np.min(finite)), float(np.max(finite))
    return vmin, vmax


def overlay_cylinders(ax: plt.Axes, centers: np.ndarray, *, radius: float = 0.5, linewidth: float = 1.0) -> None:
    centers_arr = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    for idx, (cx, cy) in enumerate(centers_arr):
        ax.add_patch(plt.Circle((float(cx), float(cy)), radius, fill=False, color="k", lw=linewidth, zorder=8))
        ax.text(float(cx), float(cy), f"C{idx}", fontsize=7.5, ha="center", va="center", color="k", zorder=9)


def apply_forward_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    verifier_cfg = cfg.setdefault("forward_verifier", {})
    if args.forward_backend is not None:
        verifier_cfg["backend"] = str(args.forward_backend)
    if args.generative_run is not None:
        verifier_cfg["generative_run_dir"] = str(args.generative_run)
        verifier_cfg["generative_enabled"] = True
    if args.generative_checkpoint is not None:
        verifier_cfg["generative_checkpoint_name"] = str(args.generative_checkpoint)
        verifier_cfg["generative_enabled"] = True
    if args.generative_num_samples is not None:
        verifier_cfg["generative_num_samples"] = int(args.generative_num_samples)
    if args.generative_n_steps is not None:
        verifier_cfg["generative_n_steps"] = int(args.generative_n_steps)
    if args.generative_ode_solver is not None:
        verifier_cfg["generative_ode_solver"] = str(args.generative_ode_solver)
    if args.uncertainty_penalty_weight is not None:
        verifier_cfg["uncertainty_penalty_weight"] = float(args.uncertainty_penalty_weight)


def _read_optional_json(path: Path) -> Dict[str, Any]:
    return read_json(path) if path.exists() else {}


def _normalize_forward_verifier_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    forward_cfg = dict(config.get("forward_model", {}) if isinstance(config.get("forward_model", {}), Mapping) else {})
    verifier_cfg = dict(config.get("forward_verifier", {}) if isinstance(config.get("forward_verifier", {}), Mapping) else {})
    merged: Dict[str, Any] = {
        "backend": str(verifier_cfg.get("backend", forward_cfg.get("backend", "deterministic"))).lower(),
        "deterministic_run_dir": verifier_cfg.get("deterministic_run_dir", forward_cfg.get("run_dir", "./Saved_Model/Case0010_20260428_084416")),
        "deterministic_checkpoint_name": verifier_cfg.get("deterministic_checkpoint_name", forward_cfg.get("checkpoint_name", "best_model.pt")),
        "deterministic_config_name": verifier_cfg.get("deterministic_config_name", forward_cfg.get("config_name", "resolved_train_config.json")),
        "generative_run_dir": verifier_cfg.get("generative_run_dir", forward_cfg.get("generative_run_dir", "")),
        "generative_checkpoint_name": verifier_cfg.get("generative_checkpoint_name", forward_cfg.get("generative_checkpoint_name", "best_gen.pt")),
        "generative_config_name": verifier_cfg.get("generative_config_name", forward_cfg.get("generative_config_name", "resolved_train_gen_config.json")),
        "generative_stage1_checkpoint": verifier_cfg.get("generative_stage1_checkpoint", forward_cfg.get("generative_stage1_checkpoint", "")),
        "generative_enabled": bool(verifier_cfg.get("generative_enabled", forward_cfg.get("generative_enabled", False))),
        "generative_num_samples": int(verifier_cfg.get("generative_num_samples", forward_cfg.get("generative_num_samples", 8))),
        "generative_n_steps": int(verifier_cfg.get("generative_n_steps", forward_cfg.get("generative_n_steps", 16))),
        "generative_ode_solver": str(verifier_cfg.get("generative_ode_solver", forward_cfg.get("generative_ode_solver", "heun"))),
        "generative_kpi_stat": str(verifier_cfg.get("generative_kpi_stat", forward_cfg.get("generative_kpi_stat", "mean"))),
        "uncertainty_penalty_weight": float(verifier_cfg.get("uncertainty_penalty_weight", forward_cfg.get("uncertainty_penalty_weight", 0.05))),
        "query_batch_size": int(forward_cfg.get("query_batch_size", 32768)),
    }
    if merged["backend"] not in {"deterministic", "generative"}:
        raise ValueError("forward_verifier.backend must be one of: deterministic, generative.")
    return merged


def _extract_channel_order(*payloads: Optional[Mapping[str, Any]], field_dim: int = 4) -> List[str]:
    def normalize(value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text or text == "auto":
                return None
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except json.JSONDecodeError:
                pass
            return [piece.strip() for piece in text.split(",") if piece.strip()]
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        return None

    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        direct = normalize(payload.get("channel_order"))
        if direct:
            return direct[:field_dim]
        for key in ("dataset", "data", "model", "generation"):
            section = payload.get(key)
            if isinstance(section, Mapping):
                nested = normalize(section.get("channel_order"))
                if nested:
                    return nested[:field_dim]
    return ["u", "v", "p", "omega"][:field_dim]


class ForwardVerifier:
    backend: str
    checkpoint_path: Path
    channel_order: List[str]

    def predict_cycle_for_centers(
        self,
        centers: np.ndarray,
        re_value: float,
        phase_bins: int,
        nx: int,
        ny: int,
        query_batch_size: int,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class DeterministicForwardVerifier(ForwardVerifier):
    def __init__(self, config: Mapping[str, Any], device: torch.device) -> None:
        verifier_cfg = _normalize_forward_verifier_config(config)
        run_dir = resolve_demo_path(str(verifier_cfg["deterministic_run_dir"]))
        det_cfg = {
            "run_dir": str(run_dir),
            "checkpoint_name": str(verifier_cfg["deterministic_checkpoint_name"]),
            "config_name": str(verifier_cfg["deterministic_config_name"]),
        }
        self.model, self.model_cfg, self.checkpoint_path = load_forward_model(det_cfg, device)
        resolved_cfg = _read_optional_json(run_dir / str(verifier_cfg["deterministic_config_name"]))
        inv_cfg = config.get("inverse_model", {}) if isinstance(config.get("inverse_model", {}), Mapping) else {}
        self.max_num_cylinders = int(inv_cfg.get("max_num_cylinders", config.get("dataset", {}).get("max_num_cylinders", 8)))
        self.lx = float(inv_cfg.get("domain_length_x", self.model_cfg.get("domain_length_x", 24.0)))
        self.ly = float(inv_cfg.get("domain_length_y", self.model_cfg.get("domain_length_y", 12.0)))
        self.device = device
        self.backend = "deterministic"
        field_dim = int(self.model_cfg.get("field_dim", 4))
        self.channel_order = _extract_channel_order(self.model_cfg, resolved_cfg, field_dim=field_dim)

    def predict_cycle_for_centers(
        self,
        centers: np.ndarray,
        re_value: float,
        phase_bins: int,
        nx: int,
        ny: int,
        query_batch_size: int,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        del seed
        cycle, aux = predict_cycle_for_centers(
            self.model,
            self.model_cfg,
            centers,
            re_value=re_value,
            max_num_cylinders=self.max_num_cylinders,
            phase_bins=phase_bins,
            nx=nx,
            ny=ny,
            lx=self.lx,
            ly=self.ly,
            query_batch_size=query_batch_size,
            device=self.device,
        )
        return {
            "cycle_mean": cycle,
            "cycle_samples": None,
            "cycle_std": None,
            "aux": aux,
            "backend": "deterministic",
            "channel_order": list(self.channel_order),
        }


class GenerativeForwardVerifier(ForwardVerifier):
    def __init__(self, config: Mapping[str, Any], device: torch.device) -> None:
        verifier_cfg = _normalize_forward_verifier_config(config)
        run_dir_value = str(verifier_cfg.get("generative_run_dir", "")).strip()
        if not run_dir_value:
            raise ValueError(GEN_FORWARD_UNAVAILABLE)
        run_dir = resolve_demo_path(run_dir_value)
        ckpt_path = run_dir / str(verifier_cfg.get("generative_checkpoint_name", "best_gen.pt"))
        if not ckpt_path.exists():
            raise ValueError(GEN_FORWARD_UNAVAILABLE)
        try:
            from evaluate_gen import _build_checkpoint_global_condition_vector, load_generator
            from model_gen import build_dense_condition_grid, denormalize_grid
            from train_gen import deterministic_grid_forward, load_deterministic_model
        except Exception as exc:  # pragma: no cover - import environment issue.
            raise RuntimeError(GEN_FORWARD_UNAVAILABLE) from exc

        try:
            self.flow, self.ema, self.stats, self.ckpt = load_generator(ckpt_path, device)
        except Exception as exc:
            raise ValueError(GEN_FORWARD_UNAVAILABLE) from exc

        deterministic_checkpoint = self.ckpt.get("deterministic_checkpoint_path") or self.ckpt.get("config", {}).get("deterministic_model", {}).get("checkpoint_path")
        if not deterministic_checkpoint:
            det_run = resolve_demo_path(str(verifier_cfg["deterministic_run_dir"]))
            deterministic_checkpoint = str(det_run / str(verifier_cfg["deterministic_checkpoint_name"]))
        self.det_model, self.det_model_cfg, self.det_checkpoint_path = load_deterministic_model({"checkpoint_path": str(deterministic_checkpoint)}, device)
        self._build_global_condition = _build_checkpoint_global_condition_vector
        self._build_dense_condition_grid = build_dense_condition_grid
        self._denormalize_grid = denormalize_grid
        self._deterministic_grid_forward = deterministic_grid_forward

        self.device = device
        self.backend = "generative"
        self.checkpoint_path = ckpt_path
        self.num_samples = max(1, int(verifier_cfg["generative_num_samples"]))
        self.n_steps = max(1, int(verifier_cfg["generative_n_steps"]))
        self.ode_solver = str(verifier_cfg["generative_ode_solver"])
        self.include_field = bool(self.ckpt.get("config", {}).get("stage2", {}).get("conditioning", {}).get("include_pred_field", True))
        self.max_num_cylinders = int(config.get("inverse_model", {}).get("max_num_cylinders", config.get("dataset", {}).get("max_num_cylinders", 8)))
        inv_cfg = config.get("inverse_model", {}) if isinstance(config.get("inverse_model", {}), Mapping) else {}
        self.lx = float(inv_cfg.get("domain_length_x", self.det_model_cfg.get("domain_length_x", 24.0)))
        self.ly = float(inv_cfg.get("domain_length_y", self.det_model_cfg.get("domain_length_y", 12.0)))
        field_dim = int(self.ckpt.get("n_fields", self.det_model_cfg.get("field_dim", 4)))
        self.channel_order = _extract_channel_order(self.ckpt, self.ckpt.get("config", {}), self.det_model_cfg, field_dim=field_dim)

    def predict_cycle_for_centers(
        self,
        centers: np.ndarray,
        re_value: float,
        phase_bins: int,
        nx: int,
        ny: int,
        query_batch_size: int,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        if int(nx) != int(self.flow.ae.num_x) or int(ny) != int(self.flow.ae.num_y):
            raise ValueError(
                "Generative verifier grid must match the stage-2 AE grid "
                f"({self.flow.ae.num_x}x{self.flow.ae.num_y}); got {nx}x{ny}."
            )
        structure_one = build_structure_from_centers(
            centers,
            re_value=re_value,
            max_num_cylinders=self.max_num_cylinders,
            device=self.device,
            future_module_feature_dim=int(self.det_model_cfg.get("future_module_feature_dim", 0)),
        )
        t_count = int(phase_bins)
        structure = {
            key: value.expand((t_count,) + tuple(value.shape[1:])).contiguous() if value.shape[0] == 1 else value
            for key, value in structure_one.items()
        }
        x_grid, y_grid = make_eval_grid(nx, ny, self.lx, self.ly, self.device)
        x_batch = x_grid.unsqueeze(0).expand(t_count, -1, -1).contiguous()
        y_batch = y_grid.unsqueeze(0).expand(t_count, -1, -1).contiguous()
        tau_values = torch.linspace(0.0, 1.0, t_count + 1, dtype=torch.float32, device=self.device)[:-1].view(t_count, 1)

        det_out = self._deterministic_grid_forward(
            self.det_model,
            structure,
            x_batch,
            y_batch,
            tau_values,
            query_time=tau_values,
            query_batch_size=int(query_batch_size),
        )
        global_cond = self._build_global_condition(det_out, structure, expected_dim=int(self.ckpt["global_cond_dim"]))
        cond_grid = self._build_dense_condition_grid(
            det_mean=det_out["pred_mean"],
            det_residual=det_out["pred_residual"],
            det_field=det_out["pred_field"],
            x_grid=x_batch,
            y_grid=y_batch,
            tau=tau_values,
            thermal_time=tau_values,
            re_values=structure["re_values"],
            stats=self.stats.to(self.device, dtype=det_out["pred_mean"].dtype),
            domain_length_x=float(self.det_model_cfg.get("domain_length_x", self.lx)),
            domain_length_y=float(self.det_model_cfg.get("domain_length_y", self.ly)),
            re_scale=float(self.det_model_cfg.get("re_scale", 200.0)),
            include_field=self.include_field,
        )

        samples: List[np.ndarray] = []
        base_seed = 1234 if seed is None else int(seed)
        context = self.ema.average_parameters(self.flow.velocity_net) if self.ema is not None else torch.no_grad()
        with torch.no_grad(), context:
            for sample_idx in range(self.num_samples):
                gen_res_norm = self.flow.sample(
                    cond_grid,
                    global_cond,
                    n_steps=self.n_steps,
                    ode_solver=self.ode_solver,
                    seed=base_seed + sample_idx,
                )
                gen_res = self._denormalize_grid(gen_res_norm, self.stats.to(self.device, dtype=gen_res_norm.dtype))
                gen_field = det_out["pred_mean"] + gen_res
                samples.append(gen_field.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32))

        sample_arr = np.stack(samples, axis=0)
        aux = {k: v for k, v in det_out.items() if k not in {"pred_field", "pred_mean", "pred_residual"}}
        return {
            "cycle_mean": sample_arr.mean(axis=0).astype(np.float32),
            "cycle_samples": sample_arr,
            "cycle_std": sample_arr.std(axis=0).astype(np.float32),
            "aux": aux,
            "backend": "generative",
            "channel_order": list(self.channel_order),
        }


def load_forward_verifier(config: Mapping[str, Any], device: torch.device) -> ForwardVerifier:
    verifier_cfg = _normalize_forward_verifier_config(config)
    if verifier_cfg["backend"] == "generative":
        return GenerativeForwardVerifier(config, device)
    return DeterministicForwardVerifier(config, device)


def load_inverse_checkpoint(inverse_run: Path, checkpoint_name: str, device: torch.device) -> Tuple[HypergraphInverseDesignFlow, Dict[str, Any], Dict[str, Any], Path]:
    ckpt_path = inverse_run / checkpoint_name
    if not ckpt_path.exists() and checkpoint_name == "best_model.pt":
        ckpt_path = inverse_run / "latest_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Inverse checkpoint not found: {ckpt_path}")
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    model_cfg = ckpt.get("inverse_model_config")
    if model_cfg is None:
        model_cfg = ckpt.get("config", {}).get("inverse_model")
    if model_cfg is None:
        raise KeyError(f"Checkpoint {ckpt_path} does not contain inverse_model_config.")
    model = HypergraphInverseDesignFlow(InverseModelConfig.from_dict(model_cfg))
    state = ckpt.get("model_state_dict")
    if state is None:
        raise KeyError(f"Checkpoint {ckpt_path} does not contain model_state_dict.")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, ckpt, dict(model_cfg), ckpt_path


def target_spec_from_payload(
    payload: Mapping[str, Any],
    *,
    kpi_names: Sequence[str],
    kpi_stats: Optional[Mapping[str, Any]],
    normalize: bool,
    max_num_cylinders: int,
    re_scale: float,
    domain_length_scale: float,
) -> Dict[str, Any]:
    preferences = payload.get("preferences", {}) if isinstance(payload.get("preferences", {}), Mapping) else {}
    min_center_distance = payload.get("min_center_distance", preferences.get("min_center_distance"))
    spec = build_target_spec_vector(
        kpi_names=kpi_names,
        kpi_targets=payload.get("kpis", {}),
        stats=kpi_stats,
        normalize=normalize,
        re_value=payload.get("Re", payload.get("re")),
        num_cylinders_min=payload.get("num_cylinders_min"),
        num_cylinders_max=payload.get("num_cylinders_max"),
        min_center_distance=min_center_distance,
        max_num_cylinders=max_num_cylinders,
        re_scale=re_scale,
        domain_length_scale=domain_length_scale,
        return_spec=True,
    )
    spec["preferences"] = dict(preferences)
    if "min_x_span" in preferences:
        spec["constraints"]["min_x_span"] = float(preferences["min_x_span"])
    if "min_y_span" in preferences:
        spec["constraints"]["min_y_span"] = float(preferences["min_y_span"])
    return spec


def layout_diagnostics(
    centers: np.ndarray,
    *,
    lx: float,
    ly: float,
    min_center_distance: float,
) -> Dict[str, float]:
    arr = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    if arr.shape[0] == 0:
        return {
            "min_pair_distance": 0.0,
            "x_span": 0.0,
            "y_span": 0.0,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "cluster_penalty": 1.0,
            "spread_score": 0.0,
        }
    min_pair = periodic_min_distance(arr, lx, ly)
    if not math.isfinite(min_pair):
        min_pair = max(float(lx), float(ly))
    x_span = float(np.max(arr[:, 0]) - np.min(arr[:, 0])) if arr.shape[0] > 1 else 0.0
    y_span = float(np.max(arr[:, 1]) - np.min(arr[:, 1])) if arr.shape[0] > 1 else 0.0
    cluster_penalty = max(0.0, float(min_center_distance) - float(min_pair)) / max(float(min_center_distance), 1.0e-8)
    spread_score = 0.5 * (x_span / max(float(lx), 1.0e-8) + y_span / max(float(ly), 1.0e-8))
    return {
        "min_pair_distance": float(min_pair),
        "x_span": x_span,
        "y_span": y_span,
        "centroid_x": float(np.mean(arr[:, 0])),
        "centroid_y": float(np.mean(arr[:, 1])),
        "cluster_penalty": float(cluster_penalty),
        "spread_score": float(spread_score),
    }


def count_probabilities_summary(values: Any, top_k: int = 3) -> str:
    probs = np.asarray(values, dtype=np.float64).reshape(-1)
    if probs.size == 0:
        return ""
    order = np.argsort(probs)[::-1][: max(int(top_k), 1)]
    return ";".join(f"{int(idx)}:{float(probs[idx]):.3f}" for idx in order)


def candidate_prefilter_key(candidate: Mapping[str, Any]) -> Tuple[float, float, float, float, float, float]:
    validity = candidate.get("validity", {})
    valid = bool(validity.get("valid", False)) if isinstance(validity, Mapping) else False
    min_dist = float(candidate.get("min_pair_distance", validity.get("min_pair_distance", 0.0) if isinstance(validity, Mapping) else 0.0))
    x_deficit = max(0.0, float(candidate.get("prefilter_min_x_span", 0.0)) - float(candidate.get("x_span", 0.0)))
    y_deficit = max(0.0, float(candidate.get("prefilter_min_y_span", 0.0)) - float(candidate.get("y_span", 0.0)))
    cluster_weight = float(candidate.get("prefilter_cluster_penalty_weight", 0.25))
    cluster_penalty = cluster_weight * float(candidate.get("cluster_penalty", 0.0))
    span_penalty = x_deficit / max(float(candidate.get("prefilter_min_x_span", 0.0)), 1.0) + y_deficit / max(float(candidate.get("prefilter_min_y_span", 0.0)), 1.0)
    count_error = float(candidate.get("target_count_error", abs(int(candidate.get("count", 0)))))
    spread_bonus = float(candidate.get("spread_score", 0.0)) if bool(candidate.get("prefilter_diversity", False)) else 0.0
    return (0.0 if valid else 1.0, span_penalty + cluster_penalty, -min_dist, count_error, -spread_bonus, float(candidate.get("sample_index", 0)))


def plot_candidate_flow(
    cycle: np.ndarray,
    centers: np.ndarray,
    out_path: Path,
    *,
    channel_order: Sequence[str],
    lx: float,
    ly: float,
) -> None:
    cycle_arr = np.asarray(cycle, dtype=np.float32)
    frame = cycle_arr[0]
    names = list(channel_order)[: frame.shape[-1]]
    n_channels = min(len(names), frame.shape[-1])
    if n_channels == 0:
        return

    cols = min(3, n_channels)
    rows = int(math.ceil(n_channels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.7 * rows), dpi=150, constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    extent = (0.0, float(lx), 0.0, float(ly))
    for idx, ax in enumerate(axes_arr):
        if idx >= n_channels:
            ax.axis("off")
            continue
        name = names[idx] if idx < len(names) else f"ch{idx}"
        vmin, vmax = _field_color_limits(cycle_arr[..., idx], name)
        im = ax.imshow(
            frame[..., idx],
            origin="lower",
            extent=extent,
            cmap=channel_cmap(name),
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        overlay_cylinders(ax, centers, linewidth=1.0)
        ax.set_title(f"Pred {name}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0, lx)
        ax.set_ylim(0, ly)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Inverse candidate verified flow | phase=0")
    fig.savefig(out_path)
    plt.close(fig)


def plot_organization(aux: Mapping[str, torch.Tensor], centers: np.ndarray, out_path: Path, *, lx: float, ly: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.scatter(centers[:, 0], centers[:, 1], s=40, c="#1f77b4", edgecolors="white", linewidths=0.8, label="cylinders")
    source = aux.get("hyper_source_coords")
    wake = aux.get("hyper_wake_coords")
    strength = aux.get("hyper_strength")
    if source is not None and wake is not None:
        src = source.detach().cpu().numpy().reshape(-1, 2)
        wk = wake.detach().cpu().numpy().reshape(-1, 2)
        src_phys = np.column_stack([src[:, 0] * lx, src[:, 1] * ly])
        wk_phys = np.column_stack([wk[:, 0] * lx, wk[:, 1] * ly])
        if strength is not None:
            weights = strength.detach().cpu().numpy().reshape(-1)
            weights = weights / max(float(np.max(np.abs(weights))), 1.0e-8)
        else:
            weights = np.ones(src_phys.shape[0], dtype=np.float32)
        for i in range(src_phys.shape[0]):
            ax.arrow(
                src_phys[i, 0],
                src_phys[i, 1],
                wk_phys[i, 0] - src_phys[i, 0],
                wk_phys[i, 1] - src_phys[i, 1],
                width=0.015 * max(lx, ly),
                head_width=0.18,
                alpha=0.25 + 0.6 * float(weights[i]),
                color="#d62728",
                length_includes_head=True,
            )
        ax.scatter(src_phys[:, 0], src_phys[:, 1], s=18, c="#d62728", label="hyper source")
        ax.scatter(wk_phys[:, 0], wk_phys[:, 1], s=18, c="#2ca02c", label="wake center")
    ax.set_xlim(0, lx)
    ax.set_ylim(0, ly)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def try_plot_rich_organization(
    aux: Mapping[str, torch.Tensor],
    centers: np.ndarray,
    out_dir: Path,
    *,
    rank_idx: int,
    lx: float,
    ly: float,
) -> bool:
    try:
        from organizer_viz import render_soft_organization
    except Exception:
        return False
    try:
        x = (np.arange(96, dtype=np.float32) + 0.5) * float(lx) / 96.0
        y = (np.arange(48, dtype=np.float32) + 0.5) * float(ly) / 48.0
        xx, yy = np.meshgrid(x, y)
        case = {
            "case_id": f"inverse_candidate_{rank_idx:03d}",
            "centers": np.asarray(centers, dtype=np.float32).reshape(-1, 2),
            "x_grid": xx.astype(np.float32),
            "y_grid": yy.astype(np.float32),
            "cylinder_radius": 0.5,
        }
        paths = render_soft_organization(
            out_dir,
            dict(aux),
            case,
            tau_value=0.0,
            phase_idx=rank_idx,
            organization_view="all",
            assignment_view="raw",
            show_table=True,
        )
        desired = {
            "physical": out_dir / f"candidate_{rank_idx:03d}_organization_physical.png",
            "matrices": out_dir / f"candidate_{rank_idx:03d}_organization_matrices.png",
            "sankey": out_dir / f"candidate_{rank_idx:03d}_organization_sankey.png",
            "schematic": out_dir / f"candidate_{rank_idx:03d}_organization_schematic.png",
        }
        wrote_any = False
        for key, target in desired.items():
            source = paths.get(key)
            if source and Path(source).exists():
                shutil.copyfile(source, target)
                wrote_any = True
        return wrote_any
    except Exception:
        return False


def plot_sampled_layouts(candidates: Sequence[Mapping[str, Any]], out_path: Path, *, lx: float, ly: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    scores = [float(c.get("score", float("nan"))) for c in candidates]
    finite_scores = [s for s in scores if math.isfinite(s)]
    fallback = max(finite_scores) if finite_scores else 1.0
    for idx, candidate in enumerate(candidates):
        centers = np.asarray(candidate["centers"], dtype=np.float32).reshape(-1, 2)
        score = scores[idx] if math.isfinite(scores[idx]) else fallback
        color = plt.cm.viridis(1.0 - min(score / max(fallback, 1.0e-8), 1.0))
        ax.scatter(centers[:, 0], centers[:, 1], s=14, color=color, alpha=0.65)
    ax.set_xlim(0, lx)
    ax.set_ylim(0, ly)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title("Sampled layouts colored by verified score")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_kpi_target_vs_achieved(
    verified: Sequence[Mapping[str, Any]],
    target_payload: Mapping[str, Any],
    out_path: Path,
) -> None:
    target_kpis = list((target_payload.get("kpis") or {}).keys())
    if not target_kpis or not verified:
        return
    top = verified[: min(5, len(verified))]
    x = np.arange(len(target_kpis))
    width = 0.8 / max(len(top), 1)
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(target_kpis)), 4.5), dpi=150)
    for i, candidate in enumerate(top):
        vals = [float(candidate.get("kpis", {}).get(name, 0.0)) for name in target_kpis]
        ax.bar(x + i * width, vals, width=width, label=f"rank {i}")
    for idx, name in enumerate(target_kpis):
        spec = target_payload["kpis"][name]
        if not isinstance(spec, Mapping):
            ax.axhline(float(spec), color="black", lw=0.7, alpha=0.25)
            continue
        mode = str(spec.get("mode", "exact"))
        if mode == "range":
            ax.vlines(idx + 0.4, float(spec.get("low", 0.0)), float(spec.get("high", 0.0)), color="black", lw=2.0)
        elif "value" in spec:
            ax.scatter([idx + 0.4], [float(spec["value"])], color="black", s=18, zorder=5)
        elif "high" in spec:
            ax.scatter([idx + 0.4], [float(spec["high"])], color="black", marker="v", s=18, zorder=5)
        elif "low" in spec:
            ax.scatter([idx + 0.4], [float(spec["low"])], color="black", marker="^", s=18, zorder=5)
    ax.set_xticks(x + 0.4)
    ax.set_xticklabels(target_kpis, rotation=30, ha="right")
    ax.set_ylabel("KPI value")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def padded_layout_vector(centers: np.ndarray, max_num_cylinders: int) -> np.ndarray:
    arr = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    order = np.lexsort((arr[:, 1], arr[:, 0])) if arr.shape[0] else []
    arr = arr[order] if arr.shape[0] else arr
    padded = np.zeros((max_num_cylinders, 2), dtype=np.float32)
    padded[: min(max_num_cylinders, arr.shape[0])] = arr[:max_num_cylinders]
    return padded.reshape(-1)


def _selected_kpi_names(target_spec: Mapping[str, Any], kpis_std: Mapping[str, float]) -> List[str]:
    target_entries = target_spec.get("kpi_targets", {})
    selected = [str(name) for name in target_entries.keys() if str(name) in kpis_std]
    return selected or [str(name) for name in kpis_std.keys()]


def score_verifier_result(
    result: Mapping[str, Any],
    target_spec: Mapping[str, Any],
    *,
    channel_order: Sequence[str],
    domain: Mapping[str, float],
    kpi_stat: str,
    uncertainty_penalty_weight: float,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any], float]:
    cycle_mean = np.asarray(result["cycle_mean"], dtype=np.float32)
    samples = result.get("cycle_samples")
    if samples is None:
        kpis = compute_cycle_kpis(cycle_mean, x_grid=None, y_grid=None, channel_order=channel_order, domain=domain)
        score = score_candidate_kpis(kpis, target_spec)
        return kpis, {}, score, 0.0

    sample_arr = np.asarray(samples, dtype=np.float32)
    sample_kpis = [
        compute_cycle_kpis(sample_arr[idx], x_grid=None, y_grid=None, channel_order=channel_order, domain=domain)
        for idx in range(sample_arr.shape[0])
    ]
    mean_field_kpis = compute_cycle_kpis(cycle_mean, x_grid=None, y_grid=None, channel_order=channel_order, domain=domain)
    names = sorted({name for row in sample_kpis for name in row.keys()})
    kpis_mean_from_samples = {
        name: float(np.mean([row.get(name, float("nan")) for row in sample_kpis]))
        for name in names
    }
    kpis_std = {
        name: float(np.nanstd([row.get(name, float("nan")) for row in sample_kpis]))
        for name in names
    }
    stat = str(kpi_stat).lower().strip()
    kpis = mean_field_kpis if stat in {"field_mean", "ensemble_mean", "mean_field"} else kpis_mean_from_samples
    base_score = score_candidate_kpis(kpis, target_spec)
    selected = _selected_kpi_names(target_spec, kpis_std)
    uncertainty = float(np.nanmean([kpis_std[name] for name in selected])) if selected else 0.0
    uncertainty_penalty = float(uncertainty_penalty_weight) * uncertainty
    score = dict(base_score)
    score["base_score"] = float(base_score["total_score"])
    score["total_score"] = float(base_score["total_score"] + uncertainty_penalty)
    score["uncertainty_penalty"] = uncertainty_penalty
    score["kpi_uncertainty"] = uncertainty
    return kpis, kpis_std, score, uncertainty_penalty


def plot_diversity(candidates: Sequence[Mapping[str, Any]], out_path: Path, *, max_num_cylinders: int) -> None:
    if len(candidates) < 2:
        return
    vecs = np.stack([padded_layout_vector(np.asarray(c["centers"]), max_num_cylinders) for c in candidates], axis=0)
    dists: List[float] = []
    for i in range(vecs.shape[0]):
        for j in range(i + 1, vecs.shape[0]):
            dists.append(float(np.linalg.norm(vecs[i] - vecs[j])))
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.hist(dists, bins=min(24, max(6, len(dists) // 2)), color="#4c78a8", alpha=0.85)
    ax.set_xlabel("Padded layout distance")
    ax.set_ylabel("Pair count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def try_save_cycle_gif(
    cycle: np.ndarray,
    out_path: Path,
    channel_order: Sequence[str],
    centers: Optional[np.ndarray] = None,
    *,
    lx: float = 26.0,
    ly: float = 8.0,
) -> None:
    names = [str(name).lower() for name in channel_order]
    channel_idx = names.index("omega") if "omega" in names else min(3, cycle.shape[-1] - 1)
    channel_name = names[channel_idx] if channel_idx < len(names) else f"ch{channel_idx}"
    field = np.asarray(cycle[..., channel_idx], dtype=np.float32)
    vmin, vmax = _field_color_limits(field, channel_name)
    extent = (0.0, float(lx), 0.0, float(ly))

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120, constrained_layout=True)
    im = ax.imshow(field[0], origin="lower", extent=extent, cmap=channel_cmap(channel_name), vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(f"Pred {channel_name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, lx)
    ax.set_ylim(0, ly)
    if centers is not None:
        overlay_cylinders(ax, centers, linewidth=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    phase_values = np.linspace(0.0, 1.0, field.shape[0], endpoint=False)

    def update(frame_idx: int):
        im.set_data(field[frame_idx])
        fig.suptitle(f"phase_tau={phase_values[frame_idx]:.3f}")
        return [im]

    anim = FuncAnimation(fig, update, frames=field.shape[0], blit=False)
    anim.save(out_path, writer=PillowWriter(fps=10))
    plt.close(fig)


def save_forward_candidate_artifacts(
    candidate: Mapping[str, Any],
    artifact: Mapping[str, Any],
    out_dir: Path,
    *,
    lx: float,
    ly: float,
) -> None:
    centers = np.asarray(candidate["centers"], dtype=np.float32).reshape(-1, 2)
    cycle = np.asarray(artifact["cycle_mean"], dtype=np.float32)
    channel_order = list(artifact.get("channel_order") or ["u", "v", "p", "omega"])
    cycle_std = artifact.get("cycle_std")
    np.savez_compressed(
        out_dir / "generated_verifier_cycle.npz",
        cycle_mean=cycle.astype(np.float32),
        cycle_std=np.asarray(cycle_std, dtype=np.float32) if cycle_std is not None else np.asarray([], dtype=np.float32),
        centers=centers.astype(np.float32),
        channel_order=np.asarray(channel_order),
        backend=np.asarray([str(artifact.get("backend", candidate.get("verifier_backend", "")))]),
    )
    plot_candidate_flow(cycle, centers, out_dir / "ml_flow.png", channel_order=channel_order, lx=lx, ly=ly)
    aux = artifact.get("aux", {})
    if isinstance(aux, Mapping):
        if not try_plot_rich_organization(aux, centers, out_dir, rank_idx=int(candidate.get("rank", 0) or 0), lx=lx, ly=ly):
            plot_organization(aux, centers, out_dir / "ml_organization.png", lx=lx, ly=ly)
    try_save_cycle_gif(cycle, out_dir / "ml_cycle.gif", channel_order, centers, lx=lx, ly=ly)
    write_candidate_snapshot(candidate, out_dir, name="candidate_result.json")


def write_candidates_csv(candidates: Sequence[Mapping[str, Any]], path: Path) -> None:
    keys = [
        "rank",
        "verified",
        "score",
        "uncertainty_penalty",
        "verifier_backend",
        "simulation_verified",
        "simulation_score",
        "simulation_score_delta",
        "simulation_case_dir",
        "simulation_kpi_comparison_json",
        "constraint_penalty",
        "latent_consistency",
        "Re",
        "num_cylinders",
        "centers_json",
        "valid",
        "min_pair_distance",
        "x_span",
        "y_span",
        "centroid_x",
        "centroid_y",
        "cluster_penalty",
        "spread_score",
        "repaired",
        "raw_count",
        "count_probabilities_summary",
        "per_kpi_errors_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for candidate in candidates:
            validity = candidate.get("validity", {})
            sim = candidate.get("simulation_verification", {}) if isinstance(candidate.get("simulation_verification", {}), Mapping) else {}
            writer.writerow(
                {
                    "rank": candidate.get("rank", ""),
                    "verified": bool(candidate.get("verified", False)),
                    "score": candidate.get("score", ""),
                    "uncertainty_penalty": candidate.get("uncertainty_penalty", ""),
                    "verifier_backend": candidate.get("verifier_backend", ""),
                    "simulation_verified": bool(candidate.get("simulation_verified", False)),
                    "simulation_score": sim.get("ground_truth_score", ""),
                    "simulation_score_delta": sim.get("score_delta", ""),
                    "simulation_case_dir": sim.get("case_dir", ""),
                    "simulation_kpi_comparison_json": json.dumps(json_safe(sim.get("kpi_comparison", {}))),
                    "constraint_penalty": candidate.get("constraint_penalty", ""),
                    "latent_consistency": candidate.get("latent_consistency", ""),
                    "Re": candidate.get("Re", ""),
                    "num_cylinders": candidate.get("num_cylinders", candidate.get("count", "")),
                    "centers_json": json.dumps(json_safe(candidate.get("centers", []))),
                    "valid": validity.get("valid", ""),
                    "min_pair_distance": candidate.get("min_pair_distance", validity.get("min_pair_distance", "")),
                    "x_span": candidate.get("x_span", ""),
                    "y_span": candidate.get("y_span", ""),
                    "centroid_x": candidate.get("centroid_x", ""),
                    "centroid_y": candidate.get("centroid_y", ""),
                    "cluster_penalty": candidate.get("cluster_penalty", ""),
                    "spread_score": candidate.get("spread_score", ""),
                    "repaired": candidate.get("repaired", validity.get("repaired", "")),
                    "raw_count": candidate.get("raw_count", ""),
                    "count_probabilities_summary": candidate.get("count_probabilities_summary", ""),
                    "per_kpi_errors_json": json.dumps(json_safe(candidate.get("per_kpi_errors", {}))),
                }
            )


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    args.device = str(device)
    inverse_run = resolve_demo_path(args.inverse_run)
    model, ckpt, inv_model_cfg, ckpt_path = load_inverse_checkpoint(inverse_run, args.checkpoint, device)
    cfg = ckpt.get("config", {})
    apply_forward_cli_overrides(cfg, args)
    kpi_names = ckpt.get("kpi_names", cfg.get("target_kpis", {}).get("names", DEFAULT_KPI_NAMES))
    kpi_stats = ckpt.get("kpi_stats", cfg.get("target_kpis", {}).get("stats"))
    normalize = bool(cfg.get("target_kpis", {}).get("normalize", True))
    target_payload = load_target_payload(args)
    re_value = float(target_payload.get("Re", target_payload.get("re", 100.0)))
    preferences = target_payload.get("preferences", {}) if isinstance(target_payload.get("preferences", {}), Mapping) else {}
    min_center_distance = target_payload.get("min_center_distance", preferences.get("min_center_distance", 1.1))
    min_center_distance = 1.1 if min_center_distance is None else float(min_center_distance)

    target_spec = target_spec_from_payload(
        target_payload,
        kpi_names=kpi_names,
        kpi_stats=kpi_stats,
        normalize=normalize,
        max_num_cylinders=int(inv_model_cfg.get("max_num_cylinders", 8)),
        re_scale=float(inv_model_cfg.get("re_scale", 200.0)),
        domain_length_scale=max(float(inv_model_cfg.get("domain_length_x", 24.0)), float(inv_model_cfg.get("domain_length_y", 12.0))),
    )
    target_vec = torch.from_numpy(np.asarray(target_spec["vector"], dtype=np.float32)).to(device=device)
    samples = model.sample_designs(
        target_vec,
        n_samples=int(args.n_samples),
        n_steps=int(args.n_steps),
        seed=int(args.seed),
        min_center_distance=min_center_distance,
        device=device,
    )

    verifier = load_forward_verifier(cfg, device)
    verifier_cfg = _normalize_forward_verifier_config(cfg)
    validation_cfg = cfg.get("validation", {})
    phase_bins = int(args.phase_bins or validation_cfg.get("forward_verify_phase_bins", 12))
    nx = int(args.nx or validation_cfg.get("forward_verify_nx", 96))
    ny = int(args.ny or validation_cfg.get("forward_verify_ny", 48))
    query_batch_size = int(verifier_cfg.get("query_batch_size", cfg.get("forward_model", {}).get("query_batch_size", 32768)))
    lx = float(inv_model_cfg.get("domain_length_x", 24.0))
    ly = float(inv_model_cfg.get("domain_length_y", 12.0))
    prefilter_min_x_span = float(args.prefilter_min_x_span if args.prefilter_min_x_span is not None else preferences.get("min_x_span", 0.0))
    prefilter_min_y_span = float(args.prefilter_min_y_span if args.prefilter_min_y_span is not None else preferences.get("min_y_span", 0.0))
    target_count_mid = 0.5 * (
        float(target_payload.get("num_cylinders_min", target_payload.get("num_cylinders_max", 0)) or 0)
        + float(target_payload.get("num_cylinders_max", target_payload.get("num_cylinders_min", int(inv_model_cfg.get("max_num_cylinders", 8)))) or int(inv_model_cfg.get("max_num_cylinders", 8)))
    )

    out_dir = inverse_run / "evaluation" / f"inverse_eval_{current_timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "target_spec.json", json_safe({"payload": target_payload, "target_spec": target_spec}))

    candidates: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        candidate = dict(sample)
        candidate.update(
            layout_diagnostics(
                np.asarray(sample["centers"], dtype=np.float32),
                lx=lx,
                ly=ly,
                min_center_distance=min_center_distance,
            )
        )
        candidate["sample_index"] = idx
        candidate["Re"] = re_value
        candidate["num_cylinders"] = int(sample["count"])
        candidate["verified"] = False
        candidate["score"] = float("inf")
        candidate["target_count_error"] = abs(float(candidate["num_cylinders"]) - target_count_mid)
        candidate["prefilter_diversity"] = bool(args.prefilter_diversity)
        candidate["prefilter_min_x_span"] = prefilter_min_x_span
        candidate["prefilter_min_y_span"] = prefilter_min_y_span
        candidate["prefilter_cluster_penalty_weight"] = float(args.prefilter_cluster_penalty_weight)
        candidate["repaired"] = bool(candidate.get("validity", {}).get("repaired", False))
        candidate["count_probabilities_summary"] = count_probabilities_summary(candidate.get("count_probabilities", []))
        candidates.append(candidate)

    verify_k = min(max(int(args.verify_top_k), 0), len(candidates))
    verify_indices = [c["sample_index"] for c in sorted(candidates, key=candidate_prefilter_key)[:verify_k]]
    verified_candidates: List[Dict[str, Any]] = []
    forward_artifacts: Dict[int, Dict[str, Any]] = {}
    verify_iter = tqdm(verify_indices, desc="Model verification", unit="candidate", disable=not progress_enabled())
    for rank_idx, sample_idx in enumerate(verify_iter):
        candidate = candidates[sample_idx]
        if progress_enabled():
            verify_iter.set_postfix_str(f"sample={sample_idx:03d}")
        centers = np.asarray(candidate["centers"], dtype=np.float32).reshape(-1, 2)
        verifier_result = verifier.predict_cycle_for_centers(
            centers,
            re_value,
            phase_bins,
            nx,
            ny,
            query_batch_size,
            seed=int(args.seed) + rank_idx,
        )
        cycle = np.asarray(verifier_result["cycle_mean"], dtype=np.float32)
        aux = verifier_result.get("aux", {})
        channel_order = list(verifier_result.get("channel_order") or getattr(verifier, "channel_order", None) or ["u", "v", "p", "omega"])
        kpis, kpis_std, score, uncertainty_penalty = score_verifier_result(
            verifier_result,
            target_spec,
            channel_order=channel_order,
            domain={"lx": lx, "ly": ly},
            kpi_stat=str(verifier_cfg.get("generative_kpi_stat", "mean")),
            uncertainty_penalty_weight=float(verifier_cfg.get("uncertainty_penalty_weight", 0.05)),
        )
        kpis["num_cylinders"] = int(candidate["num_cylinders"])
        kpis["min_center_distance"] = float(periodic_min_distance(centers, lx, ly))
        kpis["x_span"] = float(candidate.get("x_span", 0.0))
        kpis["y_span"] = float(candidate.get("y_span", 0.0))
        kpis["valid"] = bool(candidate.get("validity", {}).get("valid", True))
        if kpis_std:
            base_score = score_candidate_kpis(kpis, target_spec)
            selected = _selected_kpi_names(target_spec, kpis_std)
            kpi_uncertainty = float(np.nanmean([kpis_std[name] for name in selected])) if selected else 0.0
            uncertainty_penalty = float(verifier_cfg.get("uncertainty_penalty_weight", 0.05)) * kpi_uncertainty
            score = dict(base_score)
            score["base_score"] = float(base_score["total_score"])
            score["kpi_uncertainty"] = kpi_uncertainty
            score["uncertainty_penalty"] = uncertainty_penalty
            score["total_score"] = float(base_score["total_score"] + uncertainty_penalty)
        else:
            score = score_candidate_kpis(kpis, target_spec)
            uncertainty_penalty = 0.0
        behavior_forward, org_forward = extract_forward_latent_targets(aux)
        behavior_hat = torch.from_numpy(np.asarray(candidate["behavior_latent_hat"], dtype=np.float32))
        org_hat = torch.from_numpy(np.asarray(candidate["organization_latent_hat"], dtype=np.float32))
        behavior_mse = torch.mean((behavior_forward[: behavior_hat.numel()] - behavior_hat[: behavior_forward.numel()]) ** 2).item()
        org_dim = min(org_forward.numel(), org_hat.numel())
        org_mse = torch.mean((org_forward[:org_dim] - org_hat[:org_dim]) ** 2).item() if org_dim > 0 else float("nan")
        latent_consistency = float(behavior_mse + (0.0 if not math.isfinite(org_mse) else org_mse))

        candidate.update(
            {
                "verified": True,
                "kpis": kpis,
                "kpis_std": kpis_std,
                "score": float(score["total_score"]),
                "uncertainty_penalty": float(uncertainty_penalty),
                "kpi_uncertainty": float(score.get("kpi_uncertainty", 0.0)),
                "per_kpi_errors": score["per_kpi_errors"],
                "constraint_penalty": float(score["constraint_penalty"]),
                "latent_consistency": latent_consistency,
                "behavior_consistency_mse": float(behavior_mse),
                "organization_consistency_mse": float(org_mse),
                "cycle_shape": list(cycle.shape),
                "cycle_std_shape": list(np.asarray(verifier_result["cycle_std"]).shape) if verifier_result.get("cycle_std") is not None else None,
                "verifier_backend": str(verifier_result.get("backend", verifier.backend)),
            }
        )
        verified_candidates.append(candidate)
        forward_artifacts[int(candidate["sample_index"])] = {
            "cycle_mean": cycle,
            "cycle_std": verifier_result.get("cycle_std"),
            "aux": aux,
            "channel_order": channel_order,
            "backend": str(verifier_result.get("backend", verifier.backend)),
        }

    ranked = sorted(
        candidates,
        key=lambda c: (
            0 if bool(c.get("verified", False)) else 1,
            0 if bool(c.get("validity", {}).get("valid", False)) else 1,
            float(c.get("score", float("inf"))),
            float(c.get("latent_consistency", float("inf"))),
        ),
    )
    for rank, candidate in enumerate(ranked):
        candidate["rank"] = rank if candidate.get("verified", False) else ""

    verified_ranked = [c for c in ranked if c.get("verified", False)]
    save_top_k = min(max(int(args.save_verified_top_k), 0), len(verified_ranked))
    sim_top_k = min(max(int(args.simulation_verify_top_k), 0), len(verified_ranked)) if args.simulation_verify else 0
    selected_for_dirs: List[Dict[str, Any]] = []
    seen_samples: set[int] = set()
    for candidate in list(verified_ranked[:save_top_k]) + list(verified_ranked[:sim_top_k]):
        sample_idx = int(candidate.get("sample_index", -1))
        if sample_idx not in seen_samples:
            selected_for_dirs.append(candidate)
            seen_samples.add(sample_idx)
    candidate_dirs = initialize_candidate_dirs(selected_for_dirs, out_dir) if selected_for_dirs else {}
    if args.save_all_sampled_designs:
        sampled_design_dir = out_dir / "sampled_designs"
        sampled_design_dir.mkdir(parents=True, exist_ok=True)
        for candidate in candidates:
            write_candidate_snapshot(candidate, sampled_design_dir, name=f"sample_{int(candidate['sample_index']):03d}.json")
    if candidate_dirs:
        print(f"[output] saving {len(candidate_dirs)} selected candidate folder(s) under {out_dir / 'candidates'}")
    for candidate in selected_for_dirs:
        sample_idx = int(candidate.get("sample_index", -1))
        cand_dir = _candidate_output_dir(candidate_dirs, candidate, out_dir)
        artifact = forward_artifacts.get(sample_idx)
        if artifact is not None:
            save_forward_candidate_artifacts(candidate, artifact, cand_dir, lx=lx, ly=ly)
    run_simulation_verification(
        verified_ranked,
        args=args,
        out_dir=out_dir,
        candidate_dirs=candidate_dirs,
        target_spec=target_spec,
        target_payload=target_payload,
        re_value=re_value,
        lx=lx,
        ly=ly,
        phase_bins=phase_bins,
    )
    write_candidates_csv(ranked, out_dir / "inverse_candidates.csv")
    write_json(
        out_dir / "inverse_candidates.json",
        json_safe(
            {
                "inverse_run": str(inverse_run),
                "checkpoint": str(ckpt_path),
                "forward_checkpoint": str(verifier.checkpoint_path),
                "forward_verifier_backend": verifier.backend,
                "target": target_payload,
                "candidates": ranked,
            }
        ),
    )
    plot_sampled_layouts(ranked, out_dir / "sampled_layouts_by_score.png", lx=lx, ly=ly)
    plot_kpi_target_vs_achieved(verified_ranked, target_payload, out_dir / "kpi_target_vs_achieved.png")
    plot_diversity(ranked, out_dir / "layout_diversity.png", max_num_cylinders=int(inv_model_cfg.get("max_num_cylinders", 8)))

    if args.refine_top_k or args.refine_steps:
        print("[refine] optional refinement is reserved in this demo build; sampled candidates were evaluated directly.")

    best = verified_ranked[0] if verified_ranked else None
    print(f"[done] wrote inverse evaluation to {out_dir}")
    if best is not None:
        print(f"[best] score={best['score']:.4e} count={best['num_cylinders']} centers={json.dumps(json_safe(best['centers']))}")


INERT_CHANNEL_ORDER = ("u", "v", "p", "omega")


if __name__ == "__main__":
    main()

from __future__ import annotations

"""
Sample, verify, and rank inverse-design candidates.

python src/evaluate_inverse.py \
  --inverse-run Saved_Model_Inverse/CaseInv_inert_case0010_demo002_20260502_202359 \
  --checkpoint latest_model.pt \
  --target-json inverse_targets/balanced_low_enstrophy_valid_wake_demo.json \
  --n-samples 64 \
  --verify-top-k 16 \
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np
import torch

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
    return parser.parse_args()


def current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def json_safe(value: Any) -> Any:
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
    return build_target_spec_vector(
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


def candidate_prefilter_key(candidate: Mapping[str, Any]) -> Tuple[int, float, int]:
    validity = candidate.get("validity", {})
    valid = bool(validity.get("valid", False)) if isinstance(validity, Mapping) else False
    min_dist = float(validity.get("min_pair_distance", 0.0)) if isinstance(validity, Mapping) else 0.0
    return (0 if valid else 1, -min_dist, int(candidate.get("count", 0)))


def plot_candidate_flow(
    cycle: np.ndarray,
    centers: np.ndarray,
    out_path: Path,
    *,
    channel_order: Sequence[str],
    lx: float,
    ly: float,
) -> None:
    frame = np.asarray(cycle[0], dtype=np.float32)
    names = list(channel_order)[: frame.shape[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(9, 5), dpi=150, constrained_layout=True)
    cmaps = {"u": "coolwarm", "v": "coolwarm", "p": "magma", "omega": "RdBu_r"}
    for ax, idx in zip(axes.reshape(-1), range(min(4, frame.shape[-1]))):
        name = names[idx] if idx < len(names) else f"ch{idx}"
        im = ax.imshow(frame[..., idx], origin="lower", extent=[0, lx, 0, ly], cmap=cmaps.get(name, "viridis"), aspect="auto")
        ax.scatter(centers[:, 0], centers[:, 1], s=24, c="white", edgecolors="black", linewidths=0.7)
        ax.set_title(name)
        ax.set_xlim(0, lx)
        ax.set_ylim(0, ly)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
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


def try_save_cycle_gif(cycle: np.ndarray, out_path: Path, channel_order: Sequence[str]) -> None:
    try:
        import imageio.v2 as imageio
    except Exception:
        return
    names = [str(name).lower() for name in channel_order]
    omega_idx = names.index("omega") if "omega" in names else min(3, cycle.shape[-1] - 1)
    omega = cycle[..., omega_idx]
    vmax = max(float(np.max(np.abs(omega))), 1.0e-8)
    frames = []
    for frame in omega:
        normalized = np.clip(0.5 + 0.5 * frame / vmax, 0.0, 1.0)
        rgba = plt.cm.RdBu_r(normalized)
        frames.append((rgba[..., :3] * 255).astype(np.uint8))
    imageio.mimsave(out_path, frames, duration=0.12)


def write_candidates_csv(candidates: Sequence[Mapping[str, Any]], path: Path) -> None:
    keys = [
        "rank",
        "verified",
        "score",
        "uncertainty_penalty",
        "verifier_backend",
        "constraint_penalty",
        "latent_consistency",
        "Re",
        "num_cylinders",
        "centers_json",
        "valid",
        "min_pair_distance",
        "per_kpi_errors_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for candidate in candidates:
            validity = candidate.get("validity", {})
            writer.writerow(
                {
                    "rank": candidate.get("rank", ""),
                    "verified": bool(candidate.get("verified", False)),
                    "score": candidate.get("score", ""),
                    "uncertainty_penalty": candidate.get("uncertainty_penalty", ""),
                    "verifier_backend": candidate.get("verifier_backend", ""),
                    "constraint_penalty": candidate.get("constraint_penalty", ""),
                    "latent_consistency": candidate.get("latent_consistency", ""),
                    "Re": candidate.get("Re", ""),
                    "num_cylinders": candidate.get("num_cylinders", candidate.get("count", "")),
                    "centers_json": json.dumps(json_safe(candidate.get("centers", []))),
                    "valid": validity.get("valid", ""),
                    "min_pair_distance": validity.get("min_pair_distance", ""),
                    "per_kpi_errors_json": json.dumps(json_safe(candidate.get("per_kpi_errors", {}))),
                }
            )


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
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

    out_dir = inverse_run / "evaluation" / f"inverse_eval_{current_timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "target_spec.json", json_safe({"payload": target_payload, "target_spec": target_spec}))

    candidates: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        candidate = dict(sample)
        candidate["sample_index"] = idx
        candidate["Re"] = re_value
        candidate["num_cylinders"] = int(sample["count"])
        candidate["verified"] = False
        candidate["score"] = float("inf")
        candidates.append(candidate)

    verify_k = min(max(int(args.verify_top_k), 0), len(candidates))
    verify_indices = [c["sample_index"] for c in sorted(candidates, key=candidate_prefilter_key)[:verify_k]]
    verified_candidates: List[Dict[str, Any]] = []
    for rank_idx, sample_idx in enumerate(verify_indices):
        candidate = candidates[sample_idx]
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
        if rank_idx < 5:
            plot_candidate_flow(cycle, centers, out_dir / f"candidate_{rank_idx:03d}_flow.png", channel_order=channel_order, lx=lx, ly=ly)
            if not try_plot_rich_organization(aux, centers, out_dir, rank_idx=rank_idx, lx=lx, ly=ly):
                plot_organization(aux, centers, out_dir / f"candidate_{rank_idx:03d}_organization.png", lx=lx, ly=ly)
            try_save_cycle_gif(cycle, out_dir / f"candidate_{rank_idx:03d}_cycle.gif", channel_order)

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

"""Ablation runner for unified forward-model experiments."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Dict, List

import torch

from case_adapters import ChannelThermalAdapter, MultiCylinderAdapter, make_synthetic_batch
from diagnostics import (
    compute_basic_field_metrics,
    compute_hypergraph_diagnostics,
    compute_organizer_regularization,
    plot_ablation_summary,
    save_diagnostics_json,
)
from train_unified import SANDBOX_ROOT, load_json, resolve_device, train_one_run
from unified_model_core import UnifiedHypergraphNeuralField
from unified_types import AblationConfig, UnifiedForwardConfig


def main() -> None:
    args = parse_args()
    base_payload = load_json(args.config)
    if args.device is not None:
        base_payload.setdefault("training", {})["device"] = args.device
    plan_payload = load_json(args.plan)
    if args.case == "channelthermal":
        run_channelthermal_ablations(base_payload, plan_payload)
    else:
        run_synthetic_ablations(args.case, base_payload, plan_payload)


def run_channelthermal_ablations(base_payload: Dict[str, Any], plan_payload: Dict[str, Any]) -> None:
    ablation_root = SANDBOX_ROOT / "results" / "ablations" / "channelthermal"
    root = make_sweep_dir(ablation_root, label="ablation")
    rows: List[Dict[str, Any]] = []
    for item in plan_payload.get("ablations", []):
        ablation = AblationConfig.from_dict(item)
        payload = resolved_payload_for_ablation(base_payload, ablation)
        run_dir = root / ablation.name
        summary = train_one_run(
            payload,
            case="channelthermal",
            output_dir=run_dir,
            use_output_dir_as_run_dir=True,
        )
        actual_run_dir = Path(summary.get("run_dir", run_dir))
        row = ablation_summary_row(ablation, summary, actual_run_dir)
        rows.append(row)

    write_summary_csv(root / "ablation_summary.csv", rows)
    save_diagnostics_json({"ablations": rows}, root / "ablation_summary.json")
    plot_ablation_summary(rows, root / "ablation_summary.png", metric_key="best_val_field_mse_physical")
    plot_ablation_summary(rows, root / "ablation_summary_field_mse.png", metric_key="best_val_field_mse_physical")
    plot_ablation_summary(rows, root / "ablation_summary_temperature_mse.png", metric_key="best_val_temperature_mse")
    plot_ablation_summary(rows, root / "ablation_summary_active_edges.png", metric_key="active_edge_count")
    plot_multi_metric_summary(
        rows,
        root / "ablation_summary_mass_entropy.png",
        ["env_mass_entropy_norm", "module_mass_entropy_norm"],
        "mass entropy norm",
    )
    print(json.dumps({"wrote": str(root), "ablations": rows}, indent=2, sort_keys=True))


def ablation_summary_row(ablation: AblationConfig, summary: Dict[str, Any], actual_run_dir: Path) -> Dict[str, Any]:
    best_loss = summary.get("best_by_loss_metrics", summary.get("best_metrics", {}))
    best_field = summary.get("best_by_field_mse_metrics", summary.get("best_metrics", {}))
    best_temp = summary.get("best_by_temperature_mse_metrics", summary.get("best_metrics", {}))
    best = best_field or best_loss or summary.get("latest_metrics", {})
    return {
            "name": ablation.name,
            "best_val_loss": summary.get("best_val_loss", float("nan")),
            "best_val_field_mse_physical": summary.get(
                "best_val_field_mse_physical",
                best_field.get("val_field_mse_physical", best_field.get("val_field_mse", float("nan"))),
            ),
            "best_val_temperature_mse": summary.get(
                "best_val_temperature_mse",
                best_temp.get("val_temperature_mse", float("nan")),
            ),
            "val_temperature_mse_near_module": best.get("val_temperature_mse_near_module", float("nan")),
            "val_temperature_mse_far_field": best.get("val_temperature_mse_far_field", float("nan")),
            "val_temperature_mse_downstream_mid_box": best.get("val_temperature_mse_downstream_mid_box", float("nan")),
            "active_edge_count": best.get("val_active_edge_count", float("nan")),
            "soft_active_edge_count": best.get("val_soft_active_edge_count", float("nan")),
            "env_mass_entropy_norm": best.get("val_env_mass_entropy_norm", float("nan")),
            "module_mass_entropy_norm": best.get("val_module_mass_entropy_norm", float("nan")),
            "env_mass_max": best.get("val_env_mass_max", best.get("val_env_mass_raw_max", float("nan"))),
            "module_mass_max": best.get("val_module_mass_max", best.get("val_module_mass_raw_max", float("nan"))),
            "A_mh_entropy": best.get("val_A_mh_entropy", float("nan")),
            "A_eh_entropy": best.get("val_A_eh_entropy", float("nan")),
            "direct_residual_gate": best.get("val_direct_residual_gate", float("nan")),
            "uses_hyper_context": best.get("val_uses_hyper_context", float("nan")),
            "uses_global_context": best.get("val_uses_global_context", float("nan")),
            "uses_direct_context": best.get("val_uses_direct_context", float("nan")),
            "uses_near_module_context": best.get("val_uses_near_module_context", float("nan")),
            "org_reg_loss": best.get("val_org_reg_loss", float("nan")),
            "selected_checkpoint_for_summary": str(actual_run_dir / "best_by_field_mse_model.pt"),
            "run_dir": str(actual_run_dir),
            "notes": ablation.notes,
        }


def make_sweep_dir(parent: Path, label: str = "ablation") -> Path:
    """Create one timestamped ablation sweep directory with collision retries."""
    parent.mkdir(parents=True, exist_ok=True)
    suffix = sanitize_label(label)
    pattern = re.compile(r"^Run_(\d{4})_\d{8}_\d{6}")
    for _ in range(1000):
        max_id = 0
        for child in parent.iterdir():
            if child.is_dir():
                match = pattern.match(child.name)
                if match:
                    max_id = max(max_id, int(match.group(1)))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"Run_{max_id + 1:04d}_{timestamp}" + (f"_{suffix}" if suffix else "")
        path = parent / name
        try:
            path.mkdir(parents=False, exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a unique ablation sweep directory under {parent}")


def sanitize_label(label: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label).strip())
    return text.strip("_")[:80]


def run_synthetic_ablations(case: str, base_payload: Dict[str, Any], plan_payload: Dict[str, Any]) -> None:
    training_cfg = base_payload.get("training", {})
    torch.manual_seed(int(training_cfg.get("seed", 0)))
    device = resolve_device(training_cfg.get("device", "auto"))
    batch = load_batch(case, base_payload, batch_size=int(training_cfg.get("batch_size", 2))).to(device)

    rows: List[Dict[str, Any]] = []
    for item in plan_payload.get("ablations", []):
        ablation = AblationConfig.from_dict(item)
        payload = resolved_payload_for_ablation(base_payload, ablation)
        config = UnifiedForwardConfig.from_dict(payload.get("model", {}))
        model = UnifiedHypergraphNeuralField(config).to(device)
        model.eval()
        with torch.no_grad():
            output = model(batch)
        metrics = compute_basic_field_metrics(output["pred_field"], batch.target_field)
        metrics.update(compute_hypergraph_diagnostics(output))
        metrics.update(tensor_metrics_to_float(compute_organizer_regularization(output, config, payload.get("training", {}))))
        row = {
            "name": ablation.name,
            "notes": ablation.notes,
            "case_name": batch.case_name,
            "synthetic": bool(batch.metadata.get("synthetic", False)),
            **metrics,
            "config": config.to_dict(),
        }
        rows.append(row)

    out_path = SANDBOX_ROOT / "results" / "ablation_manifest_resolved.json"
    save_diagnostics_json({"ablations": rows}, out_path)
    plot_ablation_summary(rows, SANDBOX_ROOT / "results" / "ablation_summary.png", metric_key="field_mse")
    print(json.dumps({"wrote": str(out_path), "ablations": rows}, indent=2, sort_keys=True))


def resolved_payload_for_ablation(base_payload: Dict[str, Any], ablation: AblationConfig) -> Dict[str, Any]:
    payload = json.loads(json.dumps(base_payload))
    payload.setdefault("model", {}).update(
        {
            "decoder_mode": ablation.decoder_mode,
            "use_A_me_auxiliary": ablation.use_A_me_auxiliary,
            "use_direct_module_env_decoder": ablation.use_direct_module_env_decoder,
            "use_near_module_context": ablation.use_near_module_context,
            "use_global_context": ablation.use_global_context,
            "num_hyperedges": ablation.num_hyperedges,
        }
    )
    optional_model_fields = (
        "direct_residual_gate_init",
        "use_hyper_geometry_bias",
        "hyper_geometry_bias_scale",
            "num_env_tokens_x",
            "num_env_tokens_y",
            "hidden_dim",
            "query_fourier_frequencies",
            "boundary_feature_mode",
            "position_fourier_frequencies",
            "use_position_fourier_for_modules",
            "use_position_fourier_for_env",
            "use_hypergraph_gated_pairwise_kernel",
            "pairwise_kernel_gate_init",
            "pairwise_kernel_fourier_frequencies",
            "pairwise_kernel_num_layers",
            "pairwise_kernel_include_module_token",
            "pairwise_kernel_include_module_features",
        )
    for field in optional_model_fields:
        value = getattr(ablation, field)
        if value is not None:
            payload["model"][field] = value
    if ablation.module_heat_feature_mode is not None:
        payload.setdefault("training", {})["module_heat_feature_mode"] = ablation.module_heat_feature_mode
    if ablation.use_hyper_context is not None:
        payload["model"]["use_hyper_context"] = ablation.use_hyper_context
    if ablation.model_overrides:
        recursive_update(payload.setdefault("model", {}), ablation.model_overrides)
    if ablation.training_overrides:
        recursive_update(payload.setdefault("training", {}), ablation.training_overrides)
    return payload


def recursive_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            recursive_update(base[key], value)
        else:
            base[key] = value
    return base


def tensor_metrics_to_float(metrics: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            out[key] = float(value.detach().cpu())
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def plot_multi_metric_summary(rows: List[Dict[str, Any]], path: Path, metric_keys: List[str], ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ablation_runner] matplotlib is not installed; skipping multi-metric plot.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [str(row.get("name", idx)) for idx, row in enumerate(rows)]
    xs = torch.arange(len(names), dtype=torch.float32)
    width = 0.8 / max(len(metric_keys), 1)
    fig, ax = plt.subplots(figsize=(9, 4), dpi=160)
    for idx, key in enumerate(metric_keys):
        values = [float(row.get(key, float("nan"))) for row in rows]
        offset = (idx - (len(metric_keys) - 1) / 2.0) * width
        ax.bar((xs + offset).tolist(), values, width=width, label=key)
    ax.set_xticks(xs.tolist(), names, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def load_batch(case: str, payload: Dict[str, Any], batch_size: int = 2):
    data_cfg = payload.get("data", {})
    if case == "synthetic":
        return make_synthetic_batch("channel", batch_size=batch_size, points_per_case=192)
    if case == "channelthermal":
        return ChannelThermalAdapter(data_cfg.get("channelthermal_dataset_path", ChannelThermalAdapter().dataset_path)).load_one_batch(
            batch_size=batch_size,
            points_per_case=192,
        )
    return MultiCylinderAdapter(data_cfg.get("multicylinder_dataset_path", MultiCylinderAdapter().dataset_path)).load_one_batch(
        batch_size=batch_size,
        points_per_case=192,
    )


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/unified_forward_config_template.json")
    parser.add_argument("--plan", default="configs/ablation_plan_template.json")
    parser.add_argument("--case", choices=["synthetic", "channelthermal", "multicylinder"], default="synthetic")
    parser.add_argument("--device", default=None, help="Override training.device, for example cpu, cuda, cuda:0, or cuda:1.")
    return parser.parse_args()


if __name__ == "__main__":
    main()

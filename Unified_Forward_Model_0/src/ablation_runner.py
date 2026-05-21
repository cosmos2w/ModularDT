"""Placeholder ablation runner for one-batch unified forward evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from case_adapters import ChannelThermalAdapter, MultiCylinderAdapter, make_synthetic_batch
from diagnostics import (
    compute_basic_field_metrics,
    compute_hypergraph_diagnostics,
    plot_ablation_summary,
    save_diagnostics_json,
)
from unified_model_core import UnifiedHypergraphNeuralField
from unified_types import AblationConfig, UnifiedForwardConfig


SANDBOX_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = parse_args()
    base_payload = load_json(args.config)
    plan_payload = load_json(args.plan)
    training_cfg = base_payload.get("training", {})
    torch.manual_seed(int(training_cfg.get("seed", 0)))
    device = torch.device("cuda" if training_cfg.get("device", "auto") == "auto" and torch.cuda.is_available() else "cpu")
    batch = load_batch(args.case, base_payload, batch_size=int(training_cfg.get("batch_size", 2))).to(device)

    rows: List[Dict[str, Any]] = []
    for item in plan_payload.get("ablations", []):
        ablation = AblationConfig.from_dict(item)
        model_payload = dict(base_payload.get("model", {}))
        model_payload.update(
            {
                "decoder_mode": ablation.decoder_mode,
                "use_A_me_auxiliary": ablation.use_A_me_auxiliary,
                "use_direct_module_env_decoder": ablation.use_direct_module_env_decoder,
                "use_near_module_context": ablation.use_near_module_context,
                "use_global_context": ablation.use_global_context,
                "num_hyperedges": ablation.num_hyperedges,
            }
        )
        config = UnifiedForwardConfig.from_dict(model_payload)
        model = UnifiedHypergraphNeuralField(config).to(device)
        model.eval()
        with torch.no_grad():
            output = model(batch)
        metrics = compute_basic_field_metrics(output["pred_field"], batch.target_field)
        metrics.update(compute_hypergraph_diagnostics(output))
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
    plot_ablation_summary(rows, SANDBOX_ROOT / "results" / "ablation_summary.png")
    print(json.dumps({"wrote": str(out_path), "ablations": rows}, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/unified_forward_config_template.json")
    parser.add_argument("--plan", default="configs/ablation_plan_template.json")
    parser.add_argument("--case", choices=["synthetic", "channelthermal", "multicylinder"], default="synthetic")
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = SANDBOX_ROOT / config_path
    return json.loads(config_path.read_text(encoding="utf-8"))


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


if __name__ == "__main__":
    main()

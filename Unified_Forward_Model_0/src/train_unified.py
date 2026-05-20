"""Dry-run training entry point for the unified forward-model sandbox."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F

from case_adapters import ChannelThermalAdapter, MultiCylinderAdapter, make_synthetic_batch
from diagnostics import (
    compute_basic_field_metrics,
    compute_hypergraph_diagnostics,
    plot_organization_overview,
    save_diagnostics_json,
)
from unified_model_core import UnifiedHypergraphNeuralField
from unified_types import BatchData, UnifiedForwardConfig


SANDBOX_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = parse_args()
    payload = load_json(args.config)
    training_cfg = payload.get("training", {})
    torch.manual_seed(int(training_cfg.get("seed", 0)))
    device = resolve_device(training_cfg.get("device", "auto"))
    config = UnifiedForwardConfig.from_dict(payload.get("model", {}))

    batch = load_batch(args.case, payload, batch_size=int(training_cfg.get("batch_size", 1)))
    model = UnifiedHypergraphNeuralField(config).to(device)
    batch = batch.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_cfg.get("learning_rate", 3e-4)))
    model.train()
    steps = max(int(args.max_steps), 1)
    last_output: Dict[str, Any] = {}
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        pred = output["pred_field"]
        if batch.target_field is not None:
            loss = F.mse_loss(pred, batch.target_field.float())
            loss.backward()
            if not args.dry_run:
                optimizer.step()
        last_output = output

    metrics = compute_basic_field_metrics(last_output["pred_field"], batch.target_field)
    metrics.update(compute_hypergraph_diagnostics(last_output))
    metrics.update({"case": batch.case_name, "dry_run": bool(args.dry_run), "max_steps": steps})
    out_dir = SANDBOX_ROOT / "results" / "dry_run"
    save_diagnostics_json({"metrics": metrics, "batch": batch.to_dict(), "config": config.to_dict()}, out_dir / "train_metrics.json")
    plot_organization_overview(last_output, out_dir / "train_organization.png")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/unified_forward_config_template.json")
    parser.add_argument("--case", choices=["channelthermal", "multicylinder", "synthetic"], default="synthetic")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1)
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = SANDBOX_ROOT / config_path
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_device(value: Any) -> torch.device:
    if value in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(value))


def load_batch(case: str, payload: Dict[str, Any], batch_size: int = 1) -> BatchData:
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

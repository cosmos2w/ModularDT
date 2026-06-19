"""CHANNELTHERMAL-SPECIFIC global forward-mode smoke tests.

Inputs are the Prompt-2 global NewHONF config and existing packed global HDF5
dataset. Outputs are finite-output assertions for global fallback, local
teacher ports, local predicted ports, and local mixed ports. This script is
ChannelThermal-specific because it exercises the legacy global forward
signature and local surrogate coupling contract.
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import torch

SRC_NEW_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_NEW_ROOT))

import _bootstrap_imports  # noqa: F401
from _data.channelthermal_datasets import GlobalChannelThermalDataset
from _helpers.model_utils import read_json, recursive_to_device, resolve_demo_path, select_device
from _models_channelthermal.channelthermal_full_model import ChannelThermalHONFModel
from train import build_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test global NewHONF forward modes.")
    parser.add_argument("--config", type=str, default="./Configs_new/train_global_honf_template.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--points", type=int, default=32)
    return parser.parse_args()


def assert_finite(name: str, value: torch.Tensor) -> None:
    if value.numel() > 0 and not torch.isfinite(value).all():
        raise AssertionError(f"{name} has non-finite values.")


def tensorize(value):
    if isinstance(value, dict):
        return {key: tensorize(item) for key, item in value.items()}
    if hasattr(value, "shape"):
        return torch.from_numpy(value).unsqueeze(0)
    return value


def run_mode(base_cfg: dict, dataset: GlobalChannelThermalDataset, batch: dict, device: torch.device, *, name: str, use_local: bool, mode: str) -> None:
    cfg = deepcopy(base_cfg)
    cfg.setdefault("model", {}).setdefault("channelthermal", {})
    cfg["model"]["channelthermal"]["use_local_surrogate"] = bool(use_local)
    cfg["model"]["channelthermal"]["internal_prediction_mode"] = "auto" if use_local else "global_head"
    if not use_local:
        cfg["model"]["channelthermal"]["local_surrogate_checkpoint_path"] = None
    model_config = build_model_config(cfg, dataset)
    model = ChannelThermalHONFModel(model_config).to(device)
    model.set_global_target_normalization(dataset.normalizer.stats, normalize_targets=bool(cfg.get("dataset", {}).get("normalize_targets", False)))
    model.eval()
    with torch.no_grad():
        out = model(
            batch["structure"],
            batch["query_xy"],
            interface_condition=batch.get("interface_condition"),
            local_module_params=batch.get("local_module_params"),
            teacher_port_tokens=batch.get("teacher_port_tokens"),
            local_query_points=batch.get("module_internal_query_points"),
            local_port_condition_mode=mode,
            mixed_teacher_ratio=0.5,
        )
    for key in ("pred_field", "pred_internal_temperature", "pred_interface", "pred_port_condition", "module_response_latent"):
        assert_finite(f"{name}.{key}", out[key])
    expected_source = "local_surrogate" if use_local else "global_head"
    if out["interface_source"] != expected_source:
        raise AssertionError(f"{name}: expected interface_source={expected_source}, got {out['interface_source']!r}")
    print(
        f"[ok] {name}: source={out['interface_source']} field={tuple(out['pred_field'].shape)} "
        f"internal={tuple(out['pred_internal_temperature'].shape)} interface={tuple(out['pred_interface'].shape)}"
    )


def main() -> int:
    args = parse_args()
    cfg = read_json(resolve_demo_path(args.config))
    dataset_cfg = cfg.get("dataset", {})
    dataset = GlobalChannelThermalDataset(
        dataset_cfg.get("packed_h5_path", "./Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5"),
        split=dataset_cfg.get("train_split", "train"),
        points_per_case=int(args.points),
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=False,
    )
    if len(dataset) == 0:
        raise RuntimeError("No global cases available for smoke test.")
    device = select_device(args.device)
    sample = dataset[0]
    batch = recursive_to_device({key: tensorize(value) for key, value in sample.items() if key != "case_id"}, device)
    run_mode(cfg, dataset, batch, device, name="fallback_global_head", use_local=False, mode="predicted")
    run_mode(cfg, dataset, batch, device, name="local_teacher", use_local=True, mode="teacher")
    run_mode(cfg, dataset, batch, device, name="local_predicted", use_local=True, mode="predicted")
    run_mode(cfg, dataset, batch, device, name="local_mixed", use_local=True, mode="mixed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

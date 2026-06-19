"""CHANNELTHERMAL-SPECIFIC NewHONF hardening regressions.

Inputs are the Prompt-4 global config, the existing packed ChannelThermal HDF5
dataset, and the copied Stage-A local checkpoint. Outputs are assertions for
decoder-call efficiency, frozen local-surrogate mode behavior, full-organizer
mechanism diagnostics on auxiliary port queries, and self-contained global
checkpoint loading. This script is ChannelThermal-specific test infrastructure.
"""

from __future__ import annotations

import argparse
import copy
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch

SRC_NEW_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_NEW_ROOT))

import _bootstrap_imports  # noqa: F401
from _data.channelthermal_datasets import GlobalChannelThermalDataset
from _helpers.model_utils import read_json, recursive_to_device, resolve_demo_path, select_device
from _models_channelthermal.channelthermal_full_model import ChannelThermalHONFModel
from evaluate import load_model
from train import build_model_config, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NewHONF hardening regression checks.")
    parser.add_argument("--config", type=str, default="./Configs_new/train_global_honf_template.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--points", type=int, default=32)
    return parser.parse_args()


def tensorize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: tensorize(item) for key, item in value.items()}
    if hasattr(value, "shape"):
        return torch.from_numpy(value).unsqueeze(0)
    return value


def build_batch(cfg: Dict[str, Any], points: int, device: torch.device) -> tuple[GlobalChannelThermalDataset, Dict[str, Any]]:
    dataset_cfg = cfg.get("dataset", {})
    dataset = GlobalChannelThermalDataset(
        dataset_cfg.get("packed_h5_path", "./Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5"),
        split=dataset_cfg.get("train_split", "train"),
        points_per_case=int(points),
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=False,
    )
    batch = recursive_to_device({key: tensorize(value) for key, value in dataset[0].items() if key != "case_id"}, device)
    return dataset, batch


def forward_kwargs(batch: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "structure": batch["structure"],
        "query_xy": batch["query_xy"],
        "interface_condition": batch.get("interface_condition"),
        "local_module_params": batch.get("local_module_params"),
        "teacher_port_tokens": batch.get("teacher_port_tokens"),
        "local_query_points": batch.get("module_internal_query_points"),
        "local_port_condition_mode": "predicted",
        "mixed_teacher_ratio": 0.5,
    }


def assert_close_outputs(reference: Dict[str, torch.Tensor], candidate: Dict[str, torch.Tensor]) -> None:
    for key in ("pred_field", "pred_internal_temperature", "pred_interface", "pred_port_condition"):
        if not torch.allclose(reference[key], candidate[key], atol=1.0e-6, rtol=1.0e-6):
            raise AssertionError(f"checkpoint round-trip output mismatch for {key}")


def main() -> int:
    args = parse_args()
    device = select_device(args.device)
    cfg = read_json(resolve_demo_path(args.config))
    dataset, batch = build_batch(cfg, args.points, device)

    count_cfg = copy.deepcopy(cfg)
    count_cfg.setdefault("model", {}).setdefault("physical_correction", {})["interaction_refinement_steps"] = 0
    model_config = build_model_config(count_cfg, dataset)
    model = ChannelThermalHONFModel(model_config).to(device)
    model.set_global_target_normalization(dataset.normalizer.stats, normalize_targets=bool(cfg.get("dataset", {}).get("normalize_targets", False)))
    model.eval()
    decoder_calls = 0

    def hook(*_args: Any, **_kwargs: Any) -> None:
        nonlocal decoder_calls
        decoder_calls += 1

    handle = model.core.decoder.register_forward_hook(hook)
    with torch.no_grad():
        model(**forward_kwargs(batch))
    handle.remove()
    if decoder_calls != 1:
        raise AssertionError(f"expected one final field decoder call, got {decoder_calls}")

    model.train()
    if model.local_coupling.local_surrogate is None or model.local_coupling.local_surrogate.training:
        raise AssertionError("frozen local surrogate did not remain eval() after model.train().")
    model.eval()
    local_params = batch["local_module_params"][:, :2]
    local_ports = batch["teacher_port_tokens"][:, :2]
    present = batch["structure"]["module_present"][:, :2]
    query = batch["module_internal_query_points"]
    with torch.no_grad():
        first = model.local_coupling.call_local_surrogate(local_params, local_ports, query, present)
        second = model.local_coupling.call_local_surrogate(local_params, local_ports, query, present)
    for key in ("internal_temperature", "interface_pred", "module_response_latent"):
        if not torch.allclose(first[key], second[key], atol=1.0e-7, rtol=1.0e-7):
            raise AssertionError(f"frozen local surrogate output is not deterministic for {key}")

    mechanism_cfg = copy.deepcopy(cfg)
    mechanism_cfg.setdefault("model", {}).setdefault("core_honf", {})["use_hyper_mechanism_encoder"] = True
    mechanism_cfg.setdefault("model", {}).setdefault("physical_correction", {})["interaction_refinement_steps"] = 1
    mechanism_model = ChannelThermalHONFModel(build_model_config(mechanism_cfg, dataset)).to(device)
    mechanism_model.set_global_target_normalization(dataset.normalizer.stats, normalize_targets=bool(cfg.get("dataset", {}).get("normalize_targets", False)))
    mechanism_model.eval()
    with torch.no_grad():
        mech_out = mechanism_model(**forward_kwargs(batch), return_port_global_consistency=True)
    field_flag = mech_out["routing_aux"].get("use_hyper_mechanism_encoder")
    port_flag = mech_out["routing_aux"].get("port_global_use_hyper_mechanism_encoder")
    refinement_flag = mech_out.get("refinement_use_hyper_mechanism_encoder")
    for name, value in (("field", field_flag), ("port_global", port_flag), ("refinement", refinement_flag)):
        if not torch.is_tensor(value) or float(value.detach().cpu()) != 1.0:
            raise AssertionError(f"{name} query path did not use the mechanism encoder: {value}")

    with tempfile.TemporaryDirectory(prefix="newhonf_roundtrip_") as tmp:
        ckpt_path = Path(tmp) / "roundtrip.pt"
        model.eval()
        with torch.no_grad():
            reference = model(**forward_kwargs(batch))
        save_checkpoint(
            ckpt_path,
            model=model,
            model_config=model_config,
            train_config=cfg,
            dataset=dataset,
            epoch=1,
            best_metric=0.0,
            optimizer=None,
            best_metrics={"roundtrip": 0.0},
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        checkpoint["model_config"]["channelthermal"]["local_surrogate_checkpoint_path"] = "./missing_external_local_checkpoint.pt"
        checkpoint["local_surrogate_checkpoint_path"] = "./missing_external_local_checkpoint.pt"
        torch.save(checkpoint, ckpt_path)
        loaded_model, _ = load_model(ckpt_path, device)
        with torch.no_grad():
            candidate = loaded_model(**forward_kwargs(batch))
        assert_close_outputs(reference, candidate)

    print("[ok] NewHONF hardening regressions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


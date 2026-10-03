#!/usr/bin/env python3
"""Materialize the five native Thermal profiles on CPU and inventory parameters."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from channelthermal.data.collation import ChannelThermalBatchCollator
from channelthermal.data.datasets import GlobalChannelThermalDataset
from channelthermal.model import ChannelThermalHONFModel
from channelthermal.plugin import create_plugin
from channelthermal.training.epoch import make_model_inputs
from channelthermal.workflows.train_forward import build_model_config
from torch.utils.data import DataLoader

from honf_forward_core.training.parameter_inventory import materialized_parameter_inventory
from honf_runtime.case_protocol import WorkflowRequest
from honf_runtime.config_loader import load_config_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-dir", type=Path, default=Path("src/config_core/forward/thermal_campaign"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {}
    dataset = None
    for arm in ("b-native", "b-fine", "h-tree", "h-overlap", "h-local"):
        profile = args.profiles_dir / f"{arm}_e100.json"
        cfg = create_plugin()._forward_config(load_config_bundle(profile), WorkflowRequest("forward"), args.output.parent)
        if dataset is None:
            settings = cfg["dataset"]
            dataset = GlobalChannelThermalDataset(settings["packed_h5_path"], split="train", points_per_case=17,
                normalize_inputs=True, normalize_targets=True, random_point_sampling=False,
                include_grid=False, include_structure_targets=False)
            batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=ChannelThermalBatchCollator())))
        torch.manual_seed(0)
        model = ChannelThermalHONFModel(build_model_config(cfg, dataset)).to("cpu")
        model.set_global_target_normalization(dataset.normalizer.stats, normalize_targets=True)
        model.eval()
        with torch.no_grad():
            model(**make_model_inputs(batch, local_port_condition_mode="predicted", mixed_teacher_ratio=0.,
                                      return_predicted_port_outputs=True, return_port_global_consistency=True))
        report[arm] = {"profile": str(profile), "architecture": model.config.core_honf.forward_architecture,
                       "native_materialization_case": batch["case_id"], "query_count": 17,
                       **materialized_parameter_inventory(model)}
        print(json.dumps({"arm": arm, "totals": report[arm]["totals"]}), flush=True)
        del model
        gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

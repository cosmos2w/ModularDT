#!/usr/bin/env python3
"""Replay accepted forward checkpoints and verify compact golden digests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Case_ThermalChannel" / "src"))

from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer  # noqa: E402
from channelthermal.workflows.evaluate_forward import (  # noqa: E402
    load_model,
    predict_case,
    select_sample,
)
from channelthermal.workflows.train_forward import (  # noqa: E402
    _validate_optimizer_resume_compatibility,
    build_forward_optimizer,
)
from honf_runtime.compat import strip_module_prefix  # noqa: E402


DEFAULT_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "forward_cleanup" / "golden_replay.json"
ORGANIZER_ARRAYS = (
    "A_mh",
    "A_eh",
    "hyper_state",
    "hyper_source_coords",
    "hyper_source_scale",
    "hyper_region_coords",
    "hyper_region_scale",
    "hyper_module_mass",
    "hyper_env_mass",
    "hyper_module_purity",
    "hyper_env_purity",
    "edge_active_mask",
    "effective_edge_mask",
)
OUTPUT_ARRAYS = (
    "pred_field_grid",
    "pred_internal_temperature",
    "pred_interface",
    "pred_port_condition",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Local checkpoint/hash manifest used for this replay.",
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--update", action="store_true")
    parser.add_argument(
        "--check-fused-parity",
        action="store_true",
        help="Also compare accepted edge-explicit predictions with fused dense execution.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def array_record(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": digest.hexdigest(),
    }


def state_inventory(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    entries = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in sorted(state.items())
    ]
    return {
        "key_count": len(entries),
        "sha256": canonical_json_digest(entries),
        "ordered_names_sha256": hashlib.sha256(
            "\n".join(state).encode("utf-8")
        ).hexdigest(),
        "entries": entries,
    }


def normalization_inventory(checkpoint: dict[str, Any]) -> dict[str, Any]:
    stats = checkpoint.get("global_normalization_stats", {})
    arrays = {key: array_record(value) for key, value in sorted(stats.items())}
    return {"sha256": canonical_json_digest(arrays), "arrays": arrays}


def replay_reference(
    label: str,
    reference: dict[str, Any],
    device: torch.device,
    *,
    check_fused_parity: bool = False,
) -> dict[str, Any]:
    checkpoint_path = PROJECT_ROOT / reference["checkpoint"]
    actual_checkpoint_sha = file_sha256(checkpoint_path)
    if actual_checkpoint_sha != reference["checkpoint_sha256"]:
        raise RuntimeError(
            f"{label}: checkpoint SHA-256 mismatch: "
            f"{actual_checkpoint_sha} != {reference['checkpoint_sha256']}"
        )

    model, checkpoint = load_model(checkpoint_path, device)
    state = strip_module_prefix(checkpoint["model_state_dict"])
    model.load_state_dict(state, strict=True)
    if int(checkpoint.get("epoch", -1)) != int(reference["epoch"]):
        raise RuntimeError(f"{label}: checkpoint epoch changed")

    optimizer, optimizer_inventory = build_forward_optimizer(
        model,
        checkpoint["train_config"]["training"],
    )
    _validate_optimizer_resume_compatibility(checkpoint, optimizer_inventory)
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if not optimizer_state:
        raise RuntimeError(f"{label}: accepted golden checkpoint lacks optimizer state")
    optimizer.load_state_dict(optimizer_state)
    optimizer_record = {
        "group_count": len(optimizer.param_groups),
        "state_entry_count": len(optimizer.state),
        "group_structure_sha256": optimizer_inventory["group_structure_sha256"],
        "parameter_name_digests": [
            group["ordered_names_sha256"] for group in optimizer_inventory["groups"]
        ],
        "model_parameter_order_sha256": hashlib.sha256(
            "\n".join(name for name, _ in model.named_parameters()).encode("utf-8")
        ).hexdigest(),
    }

    dataset_config = checkpoint["train_config"]["dataset"]
    stats = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in checkpoint.get("global_normalization_stats", {}).items()
    }
    dataset = GlobalChannelThermalDataset(
        dataset_config["packed_h5_path"],
        split=reference["split"],
        points_per_case=1,
        normalize_inputs=bool(dataset_config.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_config.get("normalize_targets", False)),
        random_point_sampling=False,
        include_grid=True,
        include_structure_targets=True,
        normalizer=H5Normalizer(stats),
    )
    fused_parity: dict[str, float] | None = None
    try:
        sample = select_sample(dataset, reference["case_id"], 0)
        result = predict_case(
            model,
            sample,
            device,
            query_batch_size=int(reference["query_batch_size"]),
            local_port_condition_mode="predicted",
            mixed_teacher_ratio=0.5,
            return_routing_maps=True,
            return_topology_signature=True,
        )
        if check_fused_parity:
            configs = []
            seen: set[int] = set()
            for candidate in (
                model.config.core_honf,
                model.core.config,
                model.core.decoder.config,
                model.core.decoder.pairwise_kernel.config,
            ):
                if id(candidate) not in seen:
                    seen.add(id(candidate))
                    configs.append(candidate)
            originals = [candidate.pairwise_aggregation_mode for candidate in configs]
            state_keys_before = tuple(model.state_dict())
            try:
                for candidate in configs:
                    candidate.pairwise_aggregation_mode = "fused_query_module"
                fused = predict_case(
                    model,
                    sample,
                    device,
                    query_batch_size=int(reference["query_batch_size"]),
                    local_port_condition_mode="predicted",
                    mixed_teacher_ratio=0.5,
                    return_routing_maps=True,
                    return_topology_signature=True,
                )
            finally:
                for candidate, value in zip(configs, originals):
                    candidate.pairwise_aggregation_mode = value
            if tuple(model.state_dict()) != state_keys_before:
                raise RuntimeError(f"{label}: fused parity check changed state_dict structure")
            fused_parity = {}
            failed_outputs = []
            for key in OUTPUT_ARRAYS:
                historical_values = np.asarray(result[key], dtype=np.float64)
                fused_values = np.asarray(fused[key], dtype=np.float64)
                fused_parity[key] = float(
                    np.max(np.abs(historical_values - fused_values))
                )
                if not np.allclose(
                    historical_values,
                    fused_values,
                    rtol=2.0e-6,
                    atol=2.0e-6,
                ):
                    failed_outputs.append(key)
            if failed_outputs:
                raise RuntimeError(
                    f"{label}: fused dense parity failed rtol=atol=2e-6 for "
                    f"{failed_outputs}: {fused_parity}"
                )
    finally:
        dataset.close()

    query_xy = np.stack(
        [np.asarray(sample["x_grid"]).reshape(-1), np.asarray(sample["y_grid"]).reshape(-1)],
        axis=-1,
    ).astype(np.float32)
    outputs = {key: array_record(result[key]) for key in OUTPUT_ARRAYS}
    organizer = {
        key: array_record(result["organizer_aux"][key]) for key in ORGANIZER_ARRAYS
    }
    routing = {
        key: array_record(value) for key, value in sorted(result["routing_maps"].items())
    }
    record = {
        "case_id": str(sample["case_id"]),
        "checkpoint_sha256": actual_checkpoint_sha,
        "epoch": int(checkpoint["epoch"]),
        "resolved_config_sha256": canonical_json_digest(checkpoint["train_config"]),
        "dataset_fingerprint": checkpoint.get(
            "dataset_fingerprint", dataset_config.get("dataset_fingerprint")
        ),
        "query": array_record(query_xy),
        "normalization": normalization_inventory(checkpoint),
        "state_dict": state_inventory(state),
        "optimizer_resume": optimizer_record,
        "outputs": outputs,
        "organizer": organizer,
        "routing": routing,
        "routing_summary": {
            key: float(value) for key, value in sorted(result["routing_aux"].items())
        },
    }
    if fused_parity is not None:
        print(f"{label}: fused dense parity {fused_parity}")
    return record


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    actual = {
        "schema_version": 1,
        "case_contract": "ThermalChannel test case 0653, predicted ports, checkpoint normalization",
        "references": {
            label: replay_reference(
                label,
                reference,
                device,
                check_fused_parity=(
                    bool(args.check_fused_parity)
                    and label == "run1401_best_field_e4585"
                ),
            )
            for label, reference in manifest["golden_references"].items()
        },
    }
    if args.update:
        args.fixture.parent.mkdir(parents=True, exist_ok=True)
        args.fixture.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.fixture}")
        return 0
    expected = json.loads(args.fixture.read_text(encoding="utf-8"))
    if actual != expected:
        for label in sorted(actual["references"]):
            if actual["references"][label] != expected.get("references", {}).get(label):
                print(f"mismatch: {label}", file=sys.stderr)
        return 1
    for label, record in actual["references"].items():
        print(
            f"{label}: exact replay; keys={record['state_dict']['key_count']} "
            f"optimizer_groups={record['optimizer_resume']['group_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

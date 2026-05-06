"""Pack local module conduction cases for Stage-A surrogate training."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from tqdm.auto import tqdm

try:
    import h5py
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError("preprocess_local_module_dataset.py requires h5py.") from exc

from channelthermal_common import config_from_dict, find_case_dirs, read_json, resolve_data_path


MODULE_PARAM_NAMES = (
    "q_internal",
    "solid_k",
    "solid_alpha",
    "h_mean",
    "h_std",
    "T_env_mean",
    "T_env_std",
    "T_mean",
    "T_max",
)


@dataclass
class LocalRawCase:
    split_hint: str
    case_dir: Path
    cfg_payload: Dict[str, Any]
    payload: Dict[str, np.ndarray]


@dataclass
class LocalProcessedCase:
    case_key: str
    split: str
    case_dir: Path
    cfg_payload: Dict[str, Any]
    module_params: np.ndarray
    port_tokens: np.ndarray
    internal_query_points: np.ndarray
    internal_temperature_targets: np.ndarray
    interface_targets: np.ndarray
    local_grid: np.ndarray
    local_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack local module thermal cases into HDF5.")
    parser.add_argument("--input-root", type=Path, default=Path("./Data_Saved/LocalModule_Raw"), help="Raw local case root.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./Data_Saved/Processed_LocalModule_Dataset"),
        help="Processed local dataset root.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Train split fraction for unsplit raw folders.")
    parser.add_argument("--seed", type=int, default=321, help="Split RNG seed.")
    return parser.parse_args()


def discover_local_cases(input_root: Path) -> List[LocalRawCase]:
    cases: List[LocalRawCase] = []
    for split_hint, case_dir in find_case_dirs(input_root):
        solution_path = case_dir / "local_solution.npz"
        if not solution_path.exists():
            frame_path = case_dir / "scene" / "frame_000000.npz"
            if not frame_path.exists():
                continue
            solution_path = frame_path
        with np.load(solution_path, allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files}
        cfg_payload = read_json(case_dir / "case_config.json")
        cases.append(LocalRawCase(split_hint=split_hint, case_dir=case_dir, cfg_payload=cfg_payload, payload=payload))
    return cases


def assign_splits(raw_cases: Sequence[LocalRawCase], train_fraction: float, seed: int) -> Dict[Path, str]:
    unsplit = [raw.case_dir for raw in raw_cases if raw.split_hint not in {"train", "test"}]
    assignments: Dict[Path, str] = {}
    for raw in raw_cases:
        if raw.split_hint in {"train", "test"}:
            assignments[raw.case_dir] = raw.split_hint
    if unsplit:
        rng = np.random.default_rng(seed)
        order = list(unsplit)
        rng.shuffle(order)
        if len(order) == 1:
            train_count = 1
        else:
            train_count = int(round(np.clip(train_fraction, 0.0, 1.0) * len(order)))
            train_count = min(max(train_count, 1), len(order) - 1)
        for idx, path in enumerate(order):
            assignments[path] = "train" if idx < train_count else "test"
    return assignments


def decode_names(values: np.ndarray) -> List[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def unique_case_key(base_key: str, existing: set[str]) -> str:
    key = base_key
    suffix = 1
    while key in existing:
        suffix += 1
        key = f"{base_key}_{suffix}"
    existing.add(key)
    return key


def process_local_case(raw: LocalRawCase, case_key: str, split: str) -> LocalProcessedCase:
    cfg = config_from_dict(raw.cfg_payload)
    payload = raw.payload
    local_x = payload["local_x"].astype(np.float32)
    local_y = payload["local_y"].astype(np.float32)
    local_grid = np.stack([local_x, local_y], axis=-1)
    local_mask = payload["disk_mask"].astype(np.uint8)
    temperature = payload["temperature"].astype(np.float32)
    mask_bool = local_mask.astype(bool)
    internal_query_points = local_grid[mask_bool].astype(np.float32)
    internal_temperature_targets = temperature[mask_bool].astype(np.float32)
    h = payload["h"].astype(np.float32)
    t_env = payload["T_env"].astype(np.float32)
    q_internal = float(payload["q_internal"][0])
    module_params = np.asarray(
        [
            q_internal,
            float(cfg.thermal.solid_k),
            float(cfg.thermal.solid_alpha),
            float(np.mean(h)),
            float(np.std(h)),
            float(np.mean(t_env)),
            float(np.std(t_env)),
            float(np.mean(internal_temperature_targets)),
            float(np.max(internal_temperature_targets)),
        ],
        dtype=np.float32,
    )
    return LocalProcessedCase(
        case_key=case_key,
        split=split,
        case_dir=raw.case_dir,
        cfg_payload=raw.cfg_payload,
        module_params=module_params,
        port_tokens=payload["port_tokens"].astype(np.float32),
        internal_query_points=internal_query_points,
        internal_temperature_targets=internal_temperature_targets,
        interface_targets=payload["interface_targets"].astype(np.float32),
        local_grid=local_grid.astype(np.float32),
        local_mask=local_mask,
    )


def validate_uniform_shapes(processed: Sequence[LocalProcessedCase]) -> None:
    first = processed[0]
    expected = {
        "module_params": first.module_params.shape,
        "port_tokens": first.port_tokens.shape,
        "internal_query_points": first.internal_query_points.shape,
        "internal_temperature_targets": first.internal_temperature_targets.shape,
        "interface_targets": first.interface_targets.shape,
        "local_grid": first.local_grid.shape,
        "local_mask": first.local_mask.shape,
    }
    for item in processed[1:]:
        for name, shape in expected.items():
            if getattr(item, name).shape != shape:
                raise ValueError(
                    f"Local cases must share shapes for packed root arrays. "
                    f"{item.case_key} has {name}{getattr(item, name).shape}, expected {shape}."
                )


def write_index_csv(output_root: Path, processed: Sequence[LocalProcessedCase]) -> None:
    with (output_root / "local_case_index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_key", "split", "case_dir", "q_internal"])
        writer.writeheader()
        for item in processed:
            writer.writerow(
                {
                    "case_key": item.case_key,
                    "split": item.split,
                    "case_dir": str(item.case_dir),
                    "q_internal": float(item.module_params[0]),
                }
            )


def write_h5(output_root: Path, processed: Sequence[LocalProcessedCase], raw_cases: Sequence[LocalRawCase]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    validate_uniform_shapes(processed)
    h5_path = output_root / "packed_dataset.h5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    port_feature_names = decode_names(raw_cases[0].payload["port_feature_names"])
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["dataset_type"] = "local_module_steady_conduction"
        h5.attrs["num_cases"] = len(processed)
        h5.create_dataset("case_ids", data=np.asarray([item.case_key for item in processed], dtype=string_dtype))
        h5.create_dataset("splits", data=np.asarray([item.split for item in processed], dtype=string_dtype))
        h5.create_dataset("module_param_names", data=np.asarray(MODULE_PARAM_NAMES, dtype=string_dtype))
        h5.create_dataset("port_feature_names", data=np.asarray(port_feature_names, dtype=string_dtype))
        h5.create_dataset("interface_target_names", data=np.asarray(("T_surface", "q_normal"), dtype=string_dtype))
        h5.create_dataset("module_params", data=np.stack([item.module_params for item in processed]), compression="gzip")
        h5.create_dataset("port_tokens", data=np.stack([item.port_tokens for item in processed]), compression="gzip")
        h5.create_dataset(
            "internal_query_points",
            data=np.stack([item.internal_query_points for item in processed]),
            compression="gzip",
        )
        h5.create_dataset(
            "internal_temperature_targets",
            data=np.stack([item.internal_temperature_targets for item in processed]),
            compression="gzip",
        )
        h5.create_dataset("interface_targets", data=np.stack([item.interface_targets for item in processed]), compression="gzip")
        h5.create_dataset("local_grid", data=np.stack([item.local_grid for item in processed]), compression="gzip")
        h5.create_dataset("local_mask", data=np.stack([item.local_mask for item in processed]), compression="gzip")

        cases_group = h5.create_group("cases")
        for item in processed:
            group = cases_group.create_group(item.case_key)
            group.attrs["split"] = item.split
            group.attrs["source_case_dir"] = str(item.case_dir)
            group.create_dataset("module_params", data=item.module_params)
            group.create_dataset("port_tokens", data=item.port_tokens, compression="gzip")
            group.create_dataset("internal_query_points", data=item.internal_query_points, compression="gzip")
            group.create_dataset("internal_temperature_targets", data=item.internal_temperature_targets, compression="gzip")
            group.create_dataset("interface_targets", data=item.interface_targets, compression="gzip")
            group.create_dataset("local_grid", data=item.local_grid, compression="gzip")
            group.create_dataset("local_mask", data=item.local_mask, compression="gzip")
            group.create_dataset("case_config_json", data=json.dumps(item.cfg_payload, indent=2), dtype=string_dtype)
    write_index_csv(output_root, processed)
    return h5_path


def main() -> int:
    args = parse_args()
    input_root = resolve_data_path(args.input_root)
    output_root = resolve_data_path(args.output_root)
    raw_cases = discover_local_cases(input_root)
    if not raw_cases:
        tqdm.write(f"No local module cases found under: {input_root}")
        return 1

    assignments = assign_splits(raw_cases, args.train_fraction, args.seed)
    processed: List[LocalProcessedCase] = []
    existing: set[str] = set()
    for raw in tqdm(raw_cases, desc="Preprocessing local cases", unit="case", dynamic_ncols=True):
        base_key = str(config_from_dict(raw.cfg_payload).save.case_id) or raw.case_dir.name
        case_key = unique_case_key(base_key, existing)
        processed.append(process_local_case(raw, case_key, assignments.get(raw.case_dir, "train")))

    h5_path = write_h5(output_root, processed, raw_cases)
    tqdm.write(f"Packed {len(processed)} local module cases into: {h5_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

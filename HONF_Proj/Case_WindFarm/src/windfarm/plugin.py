"""WindFarm case plugin for the ordinary HONF train/evaluate entry points."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from honf_runtime.case_protocol import WorkflowRequest
from honf_runtime.checkpoints import checkpoint_filename
from honf_runtime.config_loader import ConfigBundle
from honf_runtime.paths import resolve_path
from honf_runtime.run_store import RunStore

from .io import WindFarmDataset, discover_volume_root

DATASET_KEYS = {
    "manifest",
    "locations",
    "compact_dataset_id",
    "volume_dataset_id",
    "compact_path",
    "volume_path",
    "derived_view",
    "split_indices",
    "normalization",
    "sampling_metadata",
    "train_split",
    "validation_split",
    "test_split",
    "group_key",
    "batch_size",
    "val_batch_size",
    "num_workers",
    "dynamic_module_padding",
    "max_modules_per_batch",
    "q_train",
    "q_volume",
    "q_band",
    "env_token_shape",
    "validation_volume_queries",
    "validation_band_queries",
    "target_channels",
    "sample_seed",
    "memory_map_volume",
    "require_complete_volume_runs",
    "allow_npz_metadata_fallback",
}
LOSS_KEYS = {
    "field_mse_weight",
    "channel_weights",
    "volume_loss_weight",
    "band_loss_weight",
    "volume_sampling_fraction",
    "band_sampling_fraction",
}
EVALUATION_KEYS = {
    "split",
    "checkpoint",
    "query_batch_size",
    "validation_volume_queries",
    "validation_band_queries",
    "test_volume_queries",
    "test_band_queries",
    "save_case_metrics",
}


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(key for key in payload if key not in allowed and not str(key).startswith("_"))
    if unknown:
        raise ValueError(f"Unknown {label} settings: {unknown}")


def _read_locations(config: Mapping[str, Any]) -> dict[str, Path]:
    locations_ref = config.get("locations")
    if not locations_ref:
        raise ValueError("WindFarm dataset.locations is required.")
    locations_path = resolve_path(str(locations_ref))
    if not locations_path.is_file():
        raise FileNotFoundError(f"WindFarm dataset locations map not found: {locations_path}")
    payload = json.loads(locations_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"WindFarm locations map must be an object: {locations_path}")
    result: dict[str, Path] = {}
    for key in ("wind_farm_tensor_v1", "wind_farm_volume_v1"):
        if key not in payload:
            raise KeyError(f"WindFarm locations map is missing {key!r}.")
        result[key] = resolve_path(str(payload[key]))
    return result


def _material_paths(dataset_config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    """Resolve compact, volume, and derived-study paths without copying data."""

    locations = _read_locations(dataset_config)
    compact = locations["wind_farm_tensor_v1"]
    volume = discover_volume_root(locations["wind_farm_volume_v1"])
    derived_value = dataset_config.get("derived_view", "project://Case_WindFarm/Dataset/derived/forward_velocity_v1")
    derived = resolve_path(str(derived_value))
    return compact, volume, derived


def _single_run_dir(saved_root: Path, run_id: str) -> Path:
    normalized = f"{int(str(run_id)):04d}"
    candidates = sorted(path for path in saved_root.glob(f"Run_{normalized}_*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No WindFarm run with Run_ID={normalized} under {saved_root}.")
    if len(candidates) > 1:
        formatted = "\n  ".join(str(path) for path in candidates)
        raise RuntimeError(f"Run_ID={normalized} is ambiguous; pass an explicit checkpoint path:\n  {formatted}")
    return candidates[0]


class WindFarmPlugin:
    """Connect WindFarm resources and field-only workflows to HONF runtime."""

    case_id = "WindFarm"
    display_name = "WindFarm native 3-D velocity"
    version = "0.1.0"

    def validate_config(self, bundle: ConfigBundle) -> None:
        case = bundle.case
        _reject_unknown(dict(case.get("dataset") or {}), DATASET_KEYS, "dataset")
        _reject_unknown(dict(case.get("model") or {}), set(), "model")
        _reject_unknown(dict(case.get("loss") or {}), LOSS_KEYS, "loss")
        _reject_unknown(dict(case.get("evaluation") or {}), EVALUATION_KEYS, "evaluation")
        if case.get("local_modules"):
            raise ValueError("WindFarm is a field-only case and does not accept local_modules.")
        dataset = dict(case.get("dataset") or {})
        if dataset.get("compact_dataset_id") != "wind_farm_tensor_v1":
            raise ValueError("WindFarm forward profiles must select compact_dataset_id=wind_farm_tensor_v1.")
        if dataset.get("volume_dataset_id") != "wind_farm_volume_v1":
            raise ValueError("WindFarm forward profiles must select volume_dataset_id=wind_farm_volume_v1.")
        if dataset.get("group_key", "layout_index") != "layout_index":
            raise ValueError("WindFarm splits must be grouped by layout_index.")
        if list(dataset.get("target_channels", ["Ux", "Uy", "Uz"])) != ["Ux", "Uy", "Uz"]:
            raise ValueError("The first WindFarm study trains exactly [Ux, Uy, Uz].")
        token_shape = tuple(int(value) for value in dataset.get("env_token_shape", [16, 8, 4]))
        if len(token_shape) != 3 or any(value <= 0 for value in token_shape):
            raise ValueError("dataset.env_token_shape must contain three positive dimensions.")
        if int(dataset.get("batch_size", 8)) <= 0:
            raise ValueError("dataset.batch_size must be positive.")
        for name in ("q_train", "q_volume", "q_band"):
            if int(dataset.get(name, 0)) <= 0:
                raise ValueError(f"dataset.{name} must be positive.")
        if int(dataset["q_train"]) != int(dataset["q_volume"]) + int(dataset["q_band"]):
            raise ValueError("dataset.q_train must equal q_volume + q_band for the shared sampled objective.")
        model_payload = dict(bundle.effective.get("model") or {})
        core_payload = dict(model_payload.get("core_honf") or {})
        if int(core_payload.get("field_dim", 0)) != 3 or int(core_payload.get("spatial_dim", 3)) != 3:
            raise ValueError("WindFarm forward profiles require field_dim=3 and spatial_dim=3.")
        if list(core_payload.get("coordinate_scale", [])) != [50.0, 38.0, 6.25]:
            raise ValueError("WindFarm forward profiles require coordinate_scale=[50,38,6.25].")
        if str(core_payload.get("geometry_mode", "nonperiodic")) != "nonperiodic":
            raise ValueError("WindFarm forward profiles require nonperiodic geometry.")
        architecture = str(core_payload.get("forward_architecture", "legacy_honf"))
        if architecture not in {"legacy_honf", "dense_pairwise_field"}:
            raise ValueError(f"Unsupported WindFarm architecture {architecture!r}.")

    def inspect_launch(self, bundle: ConfigBundle, request: WorkflowRequest) -> Mapping[str, Any]:
        del request
        dataset_config = dict(bundle.case.get("dataset") or {})
        compact, volume, derived = _material_paths(dataset_config)
        if not compact.is_file():
            raise FileNotFoundError(f"WindFarm compact metadata resource not found: {compact}")
        # Opening the mmap and checking the metadata is a real resource check;
        # no full field scan or checksum is introduced here.
        dataset = WindFarmDataset(volume)
        completed = dataset.array("completed")
        if bool(dataset_config.get("require_complete_volume_runs", True)) and not bool(np.all(completed == 1)):
            raise ValueError("WindFarm volume contains a run with completed != 1.")
        return {
            "dataset ID": dataset_config.get("volume_dataset_id"),
            "compact dataset": str(compact),
            "volume dataset": str(volume),
            "derived view": str(derived),
            "dataset runs": dataset.n_runs,
            "volume mmap": True,
            "workflow": "field-only velocity",
        }

    def _forward_config(self, bundle: ConfigBundle, run_dir: Path) -> dict[str, Any]:
        cfg = copy.deepcopy(bundle.effective)
        dataset_config = cfg.setdefault("dataset", {})
        compact, volume, derived = _material_paths(dataset_config)
        dataset_config["compact_path"] = str(compact)
        dataset_config["volume_path"] = str(volume)
        dataset_config["derived_view"] = str(derived)
        dataset_config["split_indices"] = str(derived / "split_indices.npz")
        dataset_config["normalization"] = str(derived / "normalization.json")
        dataset_config["sampling_metadata"] = str(derived / "sampling_metadata.json")
        dataset_config["dataset_id"] = dataset_config.get("volume_dataset_id", "wind_farm_volume_v1")
        manifest_path = resolve_path(str(dataset_config["manifest"]))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset_config["dataset_schema"] = manifest["datasets"][dataset_config["volume_dataset_id"]]["schema"]
        # The volume resource intentionally has no whole-directory digest.
        dataset_config["dataset_fingerprint"] = None
        cfg["paths"] = {"saved_model_dir": str(run_dir.parent)}
        cfg.setdefault("training", {})["run_name"] = str(cfg.get("run", {}).get("name", "windfarm"))
        return cfg

    def train(self, bundle: ConfigBundle, request: WorkflowRequest, *, run_dir: Path) -> int:
        if request.workflow != "forward":
            raise ValueError(f"Unsupported WindFarm training workflow: {request.workflow!r}")
        from .workflows.train_forward import run_from_config

        config = self._forward_config(bundle, run_dir)
        return int(run_from_config(config, request, run_dir_override=run_dir))

    def evaluate(self, bundle: ConfigBundle, request: WorkflowRequest) -> int:
        if request.workflow not in {"forward", "compare"}:
            raise ValueError(f"Unsupported WindFarm evaluation workflow: {request.workflow!r}")
        from .workflows.evaluate_forward import evaluate_cli

        store = RunStore(bundle.effective.get("run", {}).get("output_root", "project://Trained_Results"))
        saved_root = Path(request.saved_root) if request.saved_root else store.family_root(
            case_id=self.case_id,
            workflow="forward",
            model_family="honf_forward",
        )
        if not saved_root.is_absolute():
            saved_root = resolve_path(saved_root)
        if request.workflow == "compare":
            run_ids: list[str] = []
            extra_args = list(request.extra_args)
            for index, value in enumerate(extra_args[:-1]):
                if str(value) == "--Run_ID":
                    run_ids.append(str(extra_args[index + 1]))
            if len(run_ids) < 2:
                raise ValueError("WindFarm compare evaluation requires at least two --Run_ID values.")
            selector = request.checkpoint or bundle.effective.get("evaluation", {}).get("checkpoint", "best_field")
            dataset_config = dict(bundle.case.get("dataset") or {})
            compact, volume, derived = _material_paths(dataset_config)
            comparison_root = (
                Path(request.output_dir).expanduser().resolve()
                if request.output_dir is not None
                else saved_root / "comparisons"
            )
            comparison_root.mkdir(parents=True, exist_ok=True)
            records: dict[str, Any] = {}
            forwarded_args: list[str] = []
            skip_next = False
            for value in extra_args:
                if skip_next:
                    skip_next = False
                elif str(value) == "--Run_ID":
                    skip_next = True
                else:
                    forwarded_args.append(str(value))
            for run_id in run_ids:
                run_dir = _single_run_dir(saved_root, run_id)
                checkpoint_path = run_dir / checkpoint_filename(str(selector))
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(f"WindFarm comparison checkpoint not found: {checkpoint_path}")
                output = comparison_root / f"Run_{int(run_id):04d}"
                evaluate_cli(
                    config=bundle.effective,
                    checkpoint=str(checkpoint_path),
                    volume_path=volume,
                    compact_path=compact,
                    derived_view=derived,
                    device=request.device,
                    output_dir=str(output),
                    argv=tuple(forwarded_args),
                    workflow="forward",
                )
                records[f"{int(run_id):04d}"] = {
                    "checkpoint": str(checkpoint_path),
                    "metrics": str(output / "metrics.json"),
                }
            (comparison_root / "comparison.json").write_text(
                json.dumps({"selector": str(selector), "runs": records}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return 0
        checkpoint = request.checkpoint
        run_dir = None
        if request.run_id and checkpoint and Path(str(checkpoint)).suffix != ".pt":
            run_dir = _single_run_dir(saved_root, request.run_id)
            checkpoint = str(run_dir / checkpoint_filename(str(checkpoint)))
        elif request.run_id and not checkpoint:
            run_dir = _single_run_dir(saved_root, request.run_id)
            checkpoint = str(run_dir / checkpoint_filename("best_field"))
        elif checkpoint and Path(str(checkpoint)).suffix == ".pt":
            run_dir = Path(checkpoint).expanduser().resolve().parent
        elif checkpoint:
            raise ValueError("A named WindFarm checkpoint requires --run-id.")
        elif request.workflow == "forward":
            raise ValueError("WindFarm evaluation requires --run-id or an explicit checkpoint path.")
        dataset_config = dict(bundle.case.get("dataset") or {})
        compact, volume, derived = _material_paths(dataset_config)
        output_dir = request.output_dir
        if output_dir is None and run_dir is not None:
            output_dir = str(run_dir / "evaluations" / f"windfarm_{request.workflow}")
        argv = list(request.extra_args)
        return int(
            evaluate_cli(
                config=bundle.effective,
                checkpoint=checkpoint,
                volume_path=volume,
                compact_path=compact,
                derived_view=derived,
                device=request.device,
                output_dir=output_dir,
                argv=argv,
                workflow=request.workflow,
            )
        )
def create_plugin() -> WindFarmPlugin:
    return WindFarmPlugin()


__all__ = ["WindFarmPlugin", "create_plugin"]

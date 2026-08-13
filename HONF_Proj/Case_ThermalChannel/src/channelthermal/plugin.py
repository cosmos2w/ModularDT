"""ThermalChannel implementation of the generic HONF case protocol."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from honf_runtime.case_protocol import WorkflowRequest
from honf_runtime.config_loader import ConfigBundle
from honf_runtime.paths import resolve_path
from honf_runtime.run_store import RunStore, atomic_write_json

from .config import ChannelThermalSpecificConfig
from .local_surrogate.model import LocalModuleConfig
from .local_surrogate.spec import THERMAL_DISK_SPEC
from .resources import DatasetRegistry, DatasetResource


DATASET_KEYS = {
    "manifest", "locations", "train_split", "val_split", "points_per_case",
    "val_points_per_case", "batch_size", "val_batch_size", "num_workers",
    "normalize_inputs", "normalize_targets", "random_point_sampling",
    "require_converged", "allow_train_as_validation",
}
LOCAL_COUPLING_KEYS = {
    "use_local_surrogate", "freeze_local_surrogate", "local_surrogate_checkpoint",
    "local_surrogate_latent_dim", "local_module_params_from_used_ports",
    "default_num_interface_points",
}
PHYSICAL_CORRECTION_KEYS = {
    "local_surrogate_flux_mode", "local_surrogate_flux_blend_alpha",
    "interaction_refinement_steps", "port_global_consistency_radius_offset",
    "port_global_consistency_num_points",
}
LOSS_KEYS = {
    "field_mse_weight", "temperature_weight", "field_channel_weights",
    "internal_temperature_weight", "interface_weight", "interface_loss_type",
    "interface_target_weights", "port_condition_weight", "port_supervised_weight",
    "port_temperature_weight", "port_h_weight", "port_h_loss_type",
    "port_temperature_scale", "port_smoothness_weight",
    "port_global_consistency_weight", "port_global_consistency_teacher_weight",
    "predicted_consistency_weight", "predicted_consistency_warmup_epochs",
    "organizer_regularization",
}
ORGANIZER_LOSS_KEYS = {
    "enabled", "active_edge_weight", "target_active_edges", "edge_strength_threshold",
    "edge_strength_temperature", "env_mass_entropy_floor_weight",
    "module_mass_entropy_floor_weight", "min_mass_entropy_fraction", "max_mass_weight",
    "max_mass_fraction", "duplicate_weight", "duplicate_similarity_threshold",
}
EVALUATION_KEYS = {
    "split", "checkpoint", "query_batch_size", "local_port_condition_mode",
    "mixed_teacher_ratio",
    "temperature_display_mode", "organization_view", "organization_style",
    "organization_link_threshold", "return_routing_maps", "routing_view",
    "export_hypergraph_plan",
}
LOCAL_MODULE_KEYS = {
    "dataset_id", "global_alignment_dataset_id", "source", "global_alignment_split",
    "global_alignment_val_split", "local_synthetic_weight", "global_alignment_weight",
    "train_split", "val_split", "batch_size", "val_batch_size", "num_workers",
    "normalize_inputs", "normalize_targets", "allow_train_as_validation", "model", "loss",
}
LOCAL_LOSS_KEYS = {
    "internal_mse_weight", "interface_mse_weight", "interface_target_weights",
    "interface_smoothness_weight", "plot_smoothness_loss",
}


class ThermalChannelPlugin:
    """Connect named ThermalChannel resources and workflows to HONF runtime."""

    case_id = "ThermalChannel"
    display_name = "Steady Thermal Channel"
    version = "0.1.0"
    local_module_specs = {THERMAL_DISK_SPEC.module_id: THERMAL_DISK_SPEC}

    @staticmethod
    def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
        unknown = sorted(key for key in payload if key not in allowed and not str(key).startswith("_"))
        if unknown:
            raise ValueError(f"Unknown {label} settings: {unknown}")

    def validate_config(self, bundle: ConfigBundle) -> None:
        """Strictly validate every maintained ThermalChannel namespace."""

        case = bundle.case
        dataset = dict(case.get("dataset") or {})
        model = dict(case.get("model") or {})
        loss = dict(case.get("loss") or {})
        evaluation = dict(case.get("evaluation") or {})
        local_modules = dict(case.get("local_modules") or {})
        self._reject_unknown(dataset, DATASET_KEYS, "dataset")
        self._reject_unknown(model, {"channelthermal", "local_coupling", "physical_correction"}, "model")
        self._reject_unknown(
            dict(model.get("channelthermal") or {}),
            set(ChannelThermalSpecificConfig.__dataclass_fields__),  # type: ignore[attr-defined]
            "model.channelthermal",
        )
        local_coupling = dict(model.get("local_coupling") or {})
        self._reject_unknown(local_coupling, LOCAL_COUPLING_KEYS, "model.local_coupling")
        checkpoint = local_coupling.get("local_surrogate_checkpoint")
        if checkpoint is not None:
            if not isinstance(checkpoint, Mapping):
                raise TypeError("model.local_coupling.local_surrogate_checkpoint must be an object.")
            self._reject_unknown(checkpoint, {"source", "path"}, "local_surrogate_checkpoint")
            if not checkpoint.get("path"):
                raise ValueError("local_surrogate_checkpoint.path must be non-empty.")
        self._reject_unknown(
            dict(model.get("physical_correction") or {}),
            PHYSICAL_CORRECTION_KEYS,
            "model.physical_correction",
        )
        self._reject_unknown(loss, LOSS_KEYS, "loss")
        organizer = loss.get("organizer_regularization")
        if organizer is not None:
            if not isinstance(organizer, Mapping):
                raise TypeError("loss.organizer_regularization must be an object.")
            self._reject_unknown(organizer, ORGANIZER_LOSS_KEYS, "loss.organizer_regularization")
        self._reject_unknown(evaluation, EVALUATION_KEYS, "evaluation")
        if not local_modules:
            raise ValueError("At least one local_modules entry is required for ThermalChannel.")
        for module_id, raw_spec in local_modules.items():
            if module_id not in self.local_module_specs:
                raise ValueError(f"No installed LocalModuleSpec exists for {module_id!r}.")
            if not isinstance(raw_spec, Mapping):
                raise TypeError(f"local_modules.{module_id} must be an object.")
            spec = dict(raw_spec)
            self._reject_unknown(spec, LOCAL_MODULE_KEYS, f"local_modules.{module_id}")
            self._reject_unknown(
                dict(spec.get("model") or {}),
                set(LocalModuleConfig.__dataclass_fields__),  # type: ignore[attr-defined]
                f"local_modules.{module_id}.model",
            )
            self._reject_unknown(
                dict(spec.get("loss") or {}),
                LOCAL_LOSS_KEYS,
                f"local_modules.{module_id}.loss",
            )
            configured_latent = dict(spec.get("model") or {}).get("latent_dim")
            if configured_latent not in (None, "auto") and int(configured_latent) != int(
                self.local_module_specs[module_id].latent_dim
            ):
                raise ValueError(
                    f"local_modules.{module_id}.model.latent_dim must match its LocalModuleSpec "
                    f"({self.local_module_specs[module_id].latent_dim})."
                )

    def _registry(self, bundle: ConfigBundle) -> DatasetRegistry:
        return DatasetRegistry.from_case_config(bundle.case["dataset"])

    def _dataset_id(self, bundle: ConfigBundle, request: WorkflowRequest) -> str:
        selection = bundle.effective["case"].get("selection", {})
        configured = selection.get("dataset_id")
        if configured:
            return str(configured)
        if request.workflow == "local_module":
            local_id = str(selection.get("local_module_id", "thermal_disk"))
            return str(bundle.case["local_modules"][local_id]["dataset_id"])
        raise ValueError("The core profile must select case.dataset_id.")

    def _resource(self, bundle: ConfigBundle, request: WorkflowRequest) -> DatasetResource:
        registry = self._registry(bundle)
        resource = registry.resolve(self._dataset_id(bundle, request))
        registry.validate(resource)
        return resource

    def inspect_launch(self, bundle: ConfigBundle, request: WorkflowRequest) -> Mapping[str, Any]:
        resource = self._resource(bundle, request)
        facts: dict[str, Any] = {
            "dataset ID": resource.dataset_id,
            "dataset path": str(resource.path),
            "dataset schema": resource.record.get("schema"),
            "dataset cases": resource.record.get("num_cases"),
            "dataset splits": resource.record.get("splits"),
            "dataset sha256": resource.fingerprint,
        }
        if request.workflow == "forward":
            uses_local = bool(bundle.case.get("model", {}).get("local_coupling", {}).get("use_local_surrogate", True))
            if uses_local:
                checkpoint = self._local_checkpoint_path(bundle, request)
                facts["local module"] = "thermal_disk (frozen)"
                facts["local checkpoint"] = str(checkpoint)
            else:
                facts["local module"] = "disabled (global fallback heads)"
        elif request.workflow == "local_module":
            local_id = bundle.effective["case"].get("selection", {}).get(
                "local_module_id", "thermal_disk"
            )
            facts["local module"] = local_id
            facts["local module schema"] = self.local_module_specs[str(local_id)].schema_version
        return facts

    def _local_checkpoint_path(self, bundle: ConfigBundle, request: WorkflowRequest) -> Path:
        if request.local_checkpoint:
            path = resolve_path(request.local_checkpoint)
        else:
            local_cfg = bundle.case.get("model", {}).get("local_coupling", {})
            reference = local_cfg.get("local_surrogate_checkpoint")
            if not isinstance(reference, dict) or not reference.get("path"):
                raise ValueError(
                    "Forward ThermalChannel training requires a local surrogate checkpoint. "
                    "Set model.local_coupling.local_surrogate_checkpoint or pass --local-checkpoint."
                )
            path = resolve_path(reference["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Local surrogate checkpoint not found: {path}")
        return path

    def _forward_config(self, bundle: ConfigBundle, request: WorkflowRequest, run_dir: Path) -> dict[str, Any]:
        cfg = copy.deepcopy(bundle.effective)
        resource = self._resource(bundle, request)
        cfg["dataset"]["packed_h5_path"] = str(resource.path)
        cfg["dataset"]["dataset_id"] = resource.dataset_id
        cfg["dataset"]["dataset_schema"] = resource.record.get("schema")
        cfg["dataset"]["dataset_fingerprint"] = resource.fingerprint
        cfg["dataset"].pop("manifest", None)
        cfg["dataset"].pop("locations", None)
        local_cfg = cfg["model"].setdefault("local_coupling", {})
        local_cfg.pop("local_surrogate_checkpoint", None)
        if bool(local_cfg.get("use_local_surrogate", True)):
            local_cfg["local_surrogate_checkpoint_path"] = str(self._local_checkpoint_path(bundle, request))
        else:
            local_cfg["local_surrogate_checkpoint_path"] = None
        cfg["paths"] = {"saved_model_dir": str(run_dir.parent)}
        cfg.setdefault("training", {})["run_name"] = str(cfg.get("run", {}).get("name", "forward"))
        return cfg

    def _local_config(self, bundle: ConfigBundle, request: WorkflowRequest, run_dir: Path) -> dict[str, Any]:
        selection = bundle.effective["case"].get("selection", {})
        local_id = str(selection.get("local_module_id", "thermal_disk"))
        local_spec = copy.deepcopy(bundle.case["local_modules"][local_id])
        registry = self._registry(bundle)
        local_resource = registry.resolve(str(local_spec.pop("dataset_id")))
        global_resource = registry.resolve(str(local_spec.pop("global_alignment_dataset_id")))
        registry.validate(local_resource)
        registry.validate(global_resource)
        model_cfg = local_spec.pop("model")
        loss_cfg = local_spec.pop("loss")
        dataset_cfg = local_spec
        dataset_cfg["packed_h5_path"] = str(local_resource.path)
        dataset_cfg["global_alignment_packed_h5_path"] = str(global_resource.path)
        dataset_cfg["dataset_id"] = local_resource.dataset_id
        dataset_cfg["dataset_schema"] = local_resource.record.get("schema")
        dataset_cfg["dataset_fingerprint"] = local_resource.fingerprint
        dataset_cfg["global_alignment_dataset_id"] = global_resource.dataset_id
        dataset_cfg["global_alignment_dataset_schema"] = global_resource.record.get("schema")
        dataset_cfg["global_alignment_dataset_fingerprint"] = global_resource.fingerprint
        cfg = {
            "schema_version": 1,
            "Run_ID": bundle.effective["Run_ID"],
            "dataset": dataset_cfg,
            "model": model_cfg,
            "loss": loss_cfg,
            "training": copy.deepcopy(bundle.effective["training"]),
            "paths": {"saved_model_dir": str(run_dir.parent)},
            "case": copy.deepcopy(bundle.effective["case"]),
            "run": copy.deepcopy(bundle.effective["run"]),
        }
        cfg["training"]["run_name"] = str(cfg["run"].get("name", local_id))
        return cfg

    @staticmethod
    def _workflow_args(request: WorkflowRequest, *, init_checkpoint: str | None = None) -> argparse.Namespace:
        """Adapt the common request to the proven workflow's CLI namespace."""

        return argparse.Namespace(
            config=None,
            device=request.device,
            epochs=request.epochs,
            max_train_batches=request.max_train_batches,
            max_val_batches=request.max_val_batches,
            run_name=request.run_name,
            run_id=request.run_id,
            resume_checkpoint=request.resume_checkpoint,
            init_checkpoint=init_checkpoint,
        )

    def train(
        self,
        bundle: ConfigBundle,
        request: WorkflowRequest,
        *,
        run_dir: Path,
    ) -> int:
        if request.workflow == "forward":
            from .workflows.train_forward import run_from_config

            config = self._forward_config(bundle, request, run_dir)
            return run_from_config(config, self._workflow_args(request), run_dir_override=run_dir)
        if request.workflow == "local_module":
            from .workflows.train_local import run_from_config

            config = self._local_config(bundle, request, run_dir)
            init_checkpoint = request.local_checkpoint or config["training"].get("init_checkpoint_path")
            return run_from_config(
                config,
                self._workflow_args(request, init_checkpoint=init_checkpoint),
                run_dir_override=run_dir,
            )
        raise ValueError(f"Unsupported ThermalChannel training workflow: {request.workflow!r}")

    def evaluate(self, bundle: ConfigBundle, request: WorkflowRequest) -> int:
        """Dispatch evaluation while preserving the complete case CLI surface."""

        output_root = bundle.effective.get("run", {}).get("output_root", "project://Trained_Results")
        store = RunStore(output_root)
        argv = list(request.extra_args)
        if request.device:
            argv.extend(["--device", request.device])
        if request.output_dir:
            argv.extend(["--output-dir", request.output_dir])
        if request.workflow == "forward":
            from .workflows.evaluate_forward import main

            saved_root = request.saved_root or str(
                store.family_root(case_id=self.case_id, workflow="forward", model_family="honf_forward")
            )
            effective = self._evaluation_effective_config(
                bundle,
                Path(saved_root),
                request.run_id,
                workflow="forward",
                checkpoint=request.checkpoint,
            )
            selection = effective["case"].get("selection", {})
            registry = DatasetRegistry.from_case_config(effective["dataset"])
            self._append_evaluation_defaults(argv, effective.get("evaluation", {}), workflow="forward")
            dataset_id = str(selection.get("dataset_id", "thermal_channel_global_v1"))
            resource = registry.resolve(dataset_id, explicit_path=request.dataset)
            registry.validate(resource)
            argv.extend(["--dataset", str(resource.path)])
            checkpoint = request.checkpoint or str(bundle.case.get("evaluation", {}).get("checkpoint", "best_predicted"))
            if checkpoint == "best_autonomous":
                checkpoint = "best_predicted"
            argv.extend(["--checkpoint", checkpoint])
            direct_checkpoint = Path(str(checkpoint)).expanduser().suffix == ".pt"
            run_id = request.run_id or (None if direct_checkpoint else bundle.effective.get("Run_ID"))
            if run_id:
                argv.extend(["--Run_ID", str(run_id)])
            argv.extend(["--saved-root", saved_root])
            status = int(main(argv))
            self._record_latest_evaluation(Path(saved_root), run_id, checkpoint, "eval_global")
            return status

        if request.workflow == "local_module":
            from .workflows.evaluate_local import main

            configured_selection = bundle.effective["case"].get("selection", {})
            configured_local_id = str(configured_selection.get("local_module_id", "thermal_disk"))
            saved_root = request.saved_root or str(
                store.family_root(
                    case_id=self.case_id,
                    workflow="local_module",
                    model_family="honf_forward",
                    local_module_id=configured_local_id,
                )
            )
            effective = self._evaluation_effective_config(
                bundle,
                Path(saved_root),
                request.run_id,
                workflow="local_module",
                checkpoint=request.checkpoint,
            )
            selection = effective["case"].get("selection", {})
            registry = DatasetRegistry.from_case_config(effective["dataset"])
            self._append_evaluation_defaults(argv, effective.get("evaluation", {}), workflow="local_module")
            local_id = str(selection.get("local_module_id", "thermal_disk"))
            local_spec = effective["local_modules"][local_id]
            resource = registry.resolve(str(local_spec["dataset_id"]), explicit_path=request.dataset)
            registry.validate(resource)
            argv.extend(["--dataset", str(resource.path)])
            argv.extend(["--checkpoint", request.checkpoint or "best"])
            direct_checkpoint = Path(str(request.checkpoint or "best")).expanduser().suffix == ".pt"
            run_id = request.run_id or (None if direct_checkpoint else bundle.effective.get("Run_ID"))
            if run_id:
                argv.extend(["--Run_ID", str(run_id)])
            argv.extend(["--saved-root", saved_root])
            status = int(main(argv))
            self._record_latest_evaluation(Path(saved_root), run_id, request.checkpoint or "best", "eval_local")
            return status

        if request.workflow == "compare":
            from .workflows.compare_models import main

            selection = bundle.effective["case"].get("selection", {})
            registry = self._registry(bundle)
            self._append_evaluation_defaults(
                argv,
                bundle.effective.get("evaluation", {}),
                workflow="compare",
            )
            dataset_id = str(selection.get("dataset_id", "thermal_channel_global_v1"))
            resource = registry.resolve(dataset_id, explicit_path=request.dataset)
            registry.validate(resource)
            argv.extend(["--dataset", str(resource.path)])
            saved_root = request.saved_root or str(
                store.family_root(case_id=self.case_id, workflow="forward", model_family="honf_forward")
            )
            argv.extend(["--saved-root", saved_root])
            status = int(main(argv))
            self._record_comparison(Path(saved_root), request.output_dir, argv)
            return status

        raise ValueError(f"Unsupported ThermalChannel evaluation workflow: {request.workflow!r}")

    def _evaluation_effective_config(
        self,
        bundle: ConfigBundle,
        saved_root: Path,
        run_id: str | None,
        *,
        workflow: str,
        checkpoint: str | None,
    ) -> Mapping[str, Any]:
        """Load immutable source-run settings for one managed evaluation."""

        run_dir: Path | None = None
        if run_id:
            normalized = f"{int(str(run_id)):04d}"
            candidates = sorted(path for path in saved_root.glob(f"Run_{normalized}_*") if path.is_dir())
            if not candidates:
                raise FileNotFoundError(f"No run with Run_ID={normalized} under {saved_root}.")
            if len(candidates) > 1:
                formatted = "\n  ".join(str(path) for path in candidates)
                raise RuntimeError(f"Run_ID={normalized} is ambiguous; pass an explicit checkpoint path:\n  {formatted}")
            run_dir = candidates[0]
        elif checkpoint:
            candidate = resolve_path(checkpoint)
            if candidate.suffix == ".pt" and (candidate.parent / "run_manifest.json").is_file():
                run_dir = candidate.parent
        if run_dir is None:
            return bundle.effective

        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        expected = {
            "case_id": self.case_id,
            "model_family": "honf_forward",
            "workflow": workflow,
        }
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Evaluation source-run identity mismatch: {mismatches}")
        resolved_path = run_dir / "configs" / "resolved_config.json"
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Managed run lacks immutable resolved config: {resolved_path}")
        effective = json.loads(resolved_path.read_text(encoding="utf-8"))
        if effective.get("case", {}).get("id") != self.case_id:
            raise ValueError("Resolved evaluation config case identity does not match the plugin.")
        return effective

    @staticmethod
    def _append_evaluation_defaults(
        argv: list[str],
        evaluation: Mapping[str, Any],
        *,
        workflow: str,
    ) -> None:
        """Apply case-profile defaults only when the CLI did not override them."""

        value_options = {
            "--split": "split",
            "--query-batch-size": "query_batch_size",
            "--local-port-condition-mode": "local_port_condition_mode",
            "--mixed-teacher-ratio": "mixed_teacher_ratio",
            "--temperature-display-mode": "temperature_display_mode",
            "--organization-view": "organization_view",
            "--organization-style": "organization_style",
            "--organization-link-threshold": "organization_link_threshold",
            "--routing-view": "routing_view",
        }
        supported_values = {
            "forward": set(value_options),
            "local_module": {"--split"},
            "compare": {
                "--split",
                "--query-batch-size",
                "--local-port-condition-mode",
                "--mixed-teacher-ratio",
            },
        }.get(workflow, set())
        for option, key in value_options.items():
            if option not in supported_values:
                continue
            if key not in evaluation or evaluation[key] is None:
                continue
            if any(item == option or item.startswith(f"{option}=") for item in argv):
                continue
            argv.extend([option, str(evaluation[key])])
        boolean_options = {
            "--return-routing-maps": "return_routing_maps",
            "--export-hypergraph-plan": "export_hypergraph_plan",
        }
        supported_booleans = {
            "forward": set(boolean_options),
            "compare": {"--return-routing-maps"},
        }.get(workflow, set())
        for option, key in boolean_options.items():
            if option not in supported_booleans:
                continue
            if bool(evaluation.get(key, False)) and option not in argv:
                argv.append(option)

    @staticmethod
    def _record_latest_evaluation(
        saved_root: Path,
        run_id: Any,
        checkpoint: Any,
        directory_name: str,
    ) -> None:
        """Attach the newest generated evaluation directory to its run manifest."""

        if run_id:
            normalized = f"{int(str(run_id)):04d}"
            candidates = [path for path in saved_root.glob(f"Run_{normalized}_*") if path.is_dir()]
            if not candidates:
                return
            run_dir = max(candidates, key=lambda path: path.stat().st_mtime)
        else:
            checkpoint_path = Path(str(checkpoint)).expanduser().resolve()
            run_dir = checkpoint_path.parent
        evaluation_root = run_dir / directory_name
        children = [path for path in evaluation_root.iterdir() if path.is_dir()] if evaluation_root.exists() else []
        if children:
            evaluation_dir = max(children, key=lambda path: path.stat().st_mtime)
            resolved_checkpoint = None
            for summary_name in ("summary.json", "evaluation_summary.json"):
                summary_path = evaluation_dir / summary_name
                if summary_path.is_file():
                    resolved_checkpoint = json.loads(summary_path.read_text(encoding="utf-8")).get("checkpoint")
                    break
            ThermalChannelPlugin._write_artifact_manifest(
                evaluation_dir,
                filename="evaluation_manifest.json",
                kind=directory_name,
                source_run=run_dir,
                checkpoint=str(checkpoint),
                resolved_checkpoint=resolved_checkpoint,
            )
            RunStore.record_evaluation(run_dir, evaluation_dir)

    @staticmethod
    def _write_artifact_manifest(
        artifact_dir: Path,
        *,
        filename: str,
        kind: str,
        source_run: Path | None,
        checkpoint: str | None = None,
        resolved_checkpoint: str | None = None,
        arguments: list[str] | None = None,
    ) -> None:
        """Write a compact inventory for one completed post-processing job."""

        artifacts = [
            {"path": str(path.relative_to(artifact_dir)), "size_bytes": path.stat().st_size}
            for path in sorted(artifact_dir.rglob("*"))
            if path.is_file() and path.name != filename
        ]
        atomic_write_json(
            artifact_dir / filename,
            {
                "schema_version": 1,
                "status": "completed",
                "kind": kind,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_run": None if source_run is None else str(source_run.resolve()),
                "checkpoint": checkpoint,
                "resolved_checkpoint": resolved_checkpoint,
                "checkpoint_fallback_from": (
                    checkpoint
                    if str(checkpoint).lower() in {"best_predicted", "predicted", "autonomous"}
                    and resolved_checkpoint is not None
                    and Path(resolved_checkpoint).name == "best_model.pt"
                    else None
                ),
                "arguments": list(arguments or []),
                "artifacts": artifacts,
            },
        )

    @staticmethod
    def _record_comparison(saved_root: Path, output_dir: str | None, argv: list[str]) -> None:
        """Inventory the explicit or newest comparison output directory."""

        if output_dir:
            comparison_dir = resolve_path(output_dir)
        else:
            comparison_root = saved_root / "CompareModels"
            children = [path for path in comparison_root.iterdir() if path.is_dir()] if comparison_root.exists() else []
            if not children:
                return
            comparison_dir = max(children, key=lambda path: path.stat().st_mtime)
        if comparison_dir.is_dir():
            ThermalChannelPlugin._write_artifact_manifest(
                comparison_dir,
                filename="comparison_manifest.json",
                kind="compare",
                source_run=None,
                arguments=argv,
            )


def create_plugin() -> ThermalChannelPlugin:
    """Entry-point factory referenced by the case configuration."""

    return ThermalChannelPlugin()

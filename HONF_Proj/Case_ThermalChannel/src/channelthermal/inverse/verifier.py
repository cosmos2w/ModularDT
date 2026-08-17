"""Frozen autonomous HONF verifier for ThermalChannel inverse designs.

Physical design ``D`` and context ``c`` are replayed through the maintained
standalone forward model. Request ``R`` selects exact output functionals.
Planned compact mechanism ``G`` is external to this verifier; realized compact
plan ``G_hat`` is extracted from the final organizer after local-response
fusion using the current canonical full-plan exporter.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from honf_inverse_core.contracts import NamedContext, PhysicalDesign, VerificationResult
from honf_inverse_core.request_schema import GeometryConstraints
from honf_runtime.compat import load_trusted_checkpoint, strip_module_prefix

from channelthermal.data.datasets import CHANNEL_ORDER, GlobalChannelThermalDataset, H5Normalizer
from channelthermal.workflows.evaluate_forward import denormalize_predictions, load_model, predict_case

from .compact_plan import extract_compact_plan
from .context import context_from_forward_structure, forward_material_params
from .functionals import evaluate_supported_functionals
from .geometry import design_from_forward_structure, evaluate_geometry


RETURN_OUTPUT_NAMES = frozenset({"environment", "internal", "interface", "predicted_ports"})


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute a streaming SHA-256 for one local artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    array = np.asarray(value)
    return array.dtype.kind not in {"f", "c"} or bool(np.isfinite(array).all())


class FrozenThermalChannelVerifier:
    """Load one self-contained forward checkpoint and verify mapped dataset cases."""

    def __init__(
        self,
        checkpoint_path: str | Path = "best_predicted_model.pt",
        *,
        device: str | torch.device = "cpu",
        dataset_path: str | Path | None = None,
        query_batch_size: int = 32768,
        require_self_contained: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Frozen forward checkpoint not found: {self.checkpoint_path}")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {self.device}")
        self.query_batch_size = int(query_batch_size)
        if self.query_batch_size <= 0:
            raise ValueError("query_batch_size must be positive.")
        header = load_trusted_checkpoint(self.checkpoint_path, map_location="cpu")
        if require_self_contained:
            self._validate_self_contained(header)
        self.checkpoint_sha256 = file_sha256(self.checkpoint_path)
        self.model, self.checkpoint = load_model(self.checkpoint_path, self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._assert_frozen_eval()

        train_dataset_cfg = dict(self.checkpoint.get("train_config", {}).get("dataset", {}))
        selected_dataset = dataset_path or train_dataset_cfg.get("packed_h5_path")
        if selected_dataset is None:
            raise ValueError("Frozen verifier requires dataset_path or checkpoint train_config.dataset.packed_h5_path.")
        self.dataset_path = Path(selected_dataset).expanduser().resolve()
        if not self.dataset_path.is_file():
            raise FileNotFoundError(f"Frozen verifier dataset not found: {self.dataset_path}")
        checkpoint_stats = {
            name: np.asarray(value, dtype=np.float32)
            for name, value in self.checkpoint.get("global_normalization_stats", {}).items()
        }
        checkpoint_normalizer = H5Normalizer(checkpoint_stats) if checkpoint_stats else None
        self.normalize_inputs = bool(train_dataset_cfg.get("normalize_inputs", False))
        self.normalize_targets = bool(train_dataset_cfg.get("normalize_targets", False))
        common = dict(
            split="all",
            points_per_case=1,
            random_point_sampling=False,
            include_grid=True,
        )
        self.inference_dataset = GlobalChannelThermalDataset(
            self.dataset_path,
            normalize_inputs=self.normalize_inputs,
            normalize_targets=self.normalize_targets,
            normalizer=checkpoint_normalizer,
            **common,
        )
        self.raw_dataset = GlobalChannelThermalDataset(
            self.dataset_path,
            normalize_inputs=False,
            normalize_targets=False,
            **common,
        )
        self._case_to_index = {str(case_id): index for index, case_id in enumerate(self.raw_dataset.selected_case_ids)}
        if list(map(str, self.inference_dataset.selected_case_ids)) != list(map(str, self.raw_dataset.selected_case_ids)):
            raise ValueError("Inference/raw verifier datasets are not aligned by case ID.")
        template = self.raw_dataset[0]
        self.global_grid_shape = tuple(int(value) for value in np.asarray(template["x_grid"]).shape)
        self.local_grid_size = int(np.asarray(template["module_internal_mask"]).shape[0])

    @staticmethod
    def _validate_self_contained(checkpoint: Mapping[str, Any]) -> None:
        model_cfg = dict(checkpoint.get("model_config", {}))
        case_cfg = dict(model_cfg.get("channelthermal", {}))
        if not bool(case_cfg.get("use_local_surrogate", True)):
            return
        if not isinstance(checkpoint.get("local_model_config"), Mapping):
            raise ValueError("Self-contained forward checkpoint is missing embedded local_model_config.")
        state = strip_module_prefix(checkpoint.get("model_state_dict", {}))
        if not any(str(key).startswith("local_coupling.local_surrogate.") for key in state):
            raise ValueError("Self-contained forward checkpoint is missing embedded local-surrogate state.")

    def _assert_frozen_eval(self) -> None:
        if self.model.training:
            raise RuntimeError("Frozen verifier model left eval mode.")
        trainable = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        if trainable:
            raise RuntimeError(f"Frozen verifier has trainable parameters: {trainable[:5]}")

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(map(str, self.raw_dataset.selected_case_ids))

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "forward_checkpoint_identifier": self.checkpoint_path.name,
            "forward_checkpoint_path": str(self.checkpoint_path),
            "forward_checkpoint_sha256": self.checkpoint_sha256,
            "forward_checkpoint_size": int(self.checkpoint_path.stat().st_size),
            "forward_checkpoint_schema_version": self.checkpoint.get("checkpoint_schema_version"),
            "forward_case_id": self.checkpoint.get("case_id"),
            "forward_model_family": self.checkpoint.get("model_family"),
            "forward_workflow": self.checkpoint.get("workflow"),
            "forward_model_config": self.checkpoint.get("model_config", {}),
            "forward_train_config": self.checkpoint.get("train_config", {}),
            "forward_normalization_config": self.checkpoint.get("global_normalization_config", {}),
            "dataset_path": str(self.dataset_path),
            "dataset_id": self.checkpoint.get("dataset_id"),
            "dataset_fingerprint": self.checkpoint.get("dataset_fingerprint"),
        }

    def _samples(self, case_id: str | None, case_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.case_ids:
            raise RuntimeError("Frozen verifier dataset has no cases.")
        if case_id is None:
            index = min(max(int(case_index), 0), len(self.case_ids) - 1)
        else:
            try:
                index = self._case_to_index[str(case_id)]
            except KeyError as exc:
                raise KeyError(f"case_id={case_id!r} is not present in the verifier dataset.") from exc
        inference = self.inference_dataset[index]
        raw = self.raw_dataset[index]
        if str(inference["case_id"]) != str(raw["case_id"]):
            raise RuntimeError("Frozen verifier inference/raw sample IDs diverged.")
        return inference, raw

    def verify_case(
        self,
        *,
        case_id: str | None = None,
        case_index: int = 0,
        regional_requests: Sequence[tuple[str, Sequence[float]]] = (),
        geometry_constraints: GeometryConstraints | None = None,
        return_outputs: Iterable[str] = (),
    ) -> VerificationResult:
        """Verify one mapped dataset case through autonomous predicted ports."""

        inference, raw = self._samples(case_id, case_index)
        return self._verify_samples(
            inference,
            raw,
            regional_requests=regional_requests,
            geometry_constraints=geometry_constraints,
            return_outputs=return_outputs,
        )

    def _design_samples(
        self,
        design: PhysicalDesign,
        context: NamedContext,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Construct strict raw/inference samples from `D,c`, using only grid shapes."""

        if design.max_modules != self.raw_dataset.max_num_modules:
            raise ValueError(
                f"Generated design uses M={design.max_modules}; checkpoint dataset requires "
                f"M={self.raw_dataset.max_num_modules}."
            )
        values = context.as_mapping()
        expected = (
            float(self.model.config.core_honf.domain_length_x),
            float(self.model.config.core_honf.domain_length_y),
        )
        actual = (values["domain_length_x"], values["domain_length_y"])
        if not np.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6):
            raise ValueError(f"Context domain {actual} does not match frozen checkpoint domain {expected}.")
        ny, nx = self.global_grid_shape
        x_line = np.linspace(0.0, actual[0], nx, dtype=np.float32)
        y_line = np.linspace(0.0, actual[1], ny, dtype=np.float32)
        y_grid, x_grid = np.meshgrid(y_line, x_line, indexing="ij")
        axis = np.linspace(-1.0, 1.0, self.local_grid_size, dtype=np.float32)
        local_y, local_x = np.meshgrid(axis, axis, indexing="ij")
        local_mask = local_x**2 + local_y**2 <= 1.0
        local_query = np.stack([local_x[local_mask], local_y[local_mask]], axis=-1).astype(np.float32)
        raw_structure = {
            "re": np.asarray([values["re"]], dtype=np.float32),
            "u_in": np.asarray([values["u_in"]], dtype=np.float32),
            "module_centers": design.module_centers.copy(),
            "heat_powers": design.heat_powers.copy(),
            "module_present": design.module_present.copy(),
            "material_params": forward_material_params(context),
            "domain_length_x": np.asarray([actual[0]], dtype=np.float32),
            "domain_length_y": np.asarray([actual[1]], dtype=np.float32),
        }
        inference_structure = dict(raw_structure)
        if self.normalize_inputs:
            inference_structure["heat_powers"] = self.inference_dataset.normalizer.normalize_heat_power(
                design.heat_powers
            )
        local_module_params = np.zeros((design.max_modules, 7), dtype=np.float32)
        local_module_params[:, 0] = design.heat_powers
        local_module_params[:, 1] = values["solid_k"]
        local_module_params[:, 2] = values["solid_alpha"]
        local_module_params *= design.module_present[:, None]
        common = {
            "module_internal_query_points": local_query,
            # The predicted-port path refreshes h/T summaries, while these
            # immutable physical columns reproduce the maintained dataset
            # adapter without borrowing source-case boundary values.
            "local_module_params": local_module_params,
            "x_grid": x_grid,
            "y_grid": y_grid,
            "case_id": "generated_inverse_design",
        }
        return ({"structure": inference_structure, **common}, {"structure": raw_structure, **common})

    def verify_design(
        self,
        design: PhysicalDesign,
        context: NamedContext,
        *,
        regional_requests: Sequence[tuple[str, Sequence[float]]] = (),
        geometry_constraints: GeometryConstraints | None = None,
        return_outputs: Iterable[str] = (),
    ) -> VerificationResult:
        """Verify an arbitrary generated `D,c` without borrowing a source case."""

        inference, raw = self._design_samples(design, context)
        return self._verify_samples(
            inference,
            raw,
            regional_requests=regional_requests,
            geometry_constraints=geometry_constraints,
            return_outputs=return_outputs,
        )

    def _verify_samples(
        self,
        inference: Mapping[str, Any],
        raw: Mapping[str, Any],
        *,
        regional_requests: Sequence[tuple[str, Sequence[float]]],
        geometry_constraints: GeometryConstraints | None,
        return_outputs: Iterable[str],
    ) -> VerificationResult:
        requested_outputs = frozenset(str(value) for value in return_outputs)
        unknown = sorted(requested_outputs - RETURN_OUTPUT_NAMES)
        if unknown:
            raise ValueError(f"Unknown verifier return_outputs: {unknown}")
        with torch.inference_mode():
            predictions = predict_case(
                self.model,
                dict(inference),
                self.device,
                query_batch_size=self.query_batch_size,
                local_port_condition_mode="predicted",
                mixed_teacher_ratio=0.0,
                return_routing_maps=False,
            )
        predictions = denormalize_predictions(predictions, self.inference_dataset, self.normalize_targets)
        if not _all_finite(
            {
                "field": predictions["pred_field_grid"],
                "internal": predictions["pred_internal_temperature"],
                "interface": predictions["pred_interface"],
                "ports": predictions["pred_port_condition"],
            }
        ):
            raise ValueError("Frozen verifier returned non-finite predicted-port or physical outputs.")
        design = design_from_forward_structure(raw["structure"])
        context = context_from_forward_structure(raw["structure"])
        full_plan = self.model.extract_hypergraph_plan(
            predictions["organizer_aux"],
            raw["structure"]["module_present"],
            detach=True,
        )
        compact = extract_compact_plan(full_plan, design, context)
        functionals = evaluate_supported_functionals(
            pred_field_grid=predictions["pred_field_grid"],
            x_grid=raw["x_grid"],
            y_grid=raw["y_grid"],
            channel_order=self.inference_dataset.channel_order or list(CHANNEL_ORDER),
            design=design,
            context=context,
            pred_internal_temperature=predictions["pred_internal_temperature"],
            regional_requests=regional_requests,
        )
        geometry = (
            {}
            if geometry_constraints is None
            else evaluate_geometry(design, context, geometry_constraints).to_dict()
        )
        outputs: dict[str, Any] = {}
        if "environment" in requested_outputs:
            outputs["environment"] = {
                "x_grid": np.asarray(raw["x_grid"], dtype=np.float32),
                "y_grid": np.asarray(raw["y_grid"], dtype=np.float32),
                "pred_field_grid": np.asarray(predictions["pred_field_grid"], dtype=np.float32),
                "channel_order": list(self.inference_dataset.channel_order or CHANNEL_ORDER),
            }
        if "internal" in requested_outputs:
            outputs["internal"] = np.asarray(predictions["pred_internal_temperature"], dtype=np.float32)
        if "interface" in requested_outputs:
            outputs["interface"] = np.asarray(predictions["pred_interface"], dtype=np.float32)
        if "predicted_ports" in requested_outputs:
            outputs["predicted_ports"] = np.asarray(predictions["pred_port_condition"], dtype=np.float32)
        self._assert_frozen_eval()
        return VerificationResult(
            design=design,
            context=context,
            compact_plan=compact,
            full_plan=full_plan,
            functionals=functionals,
            geometry=geometry,
            checkpoint_provenance={
                **self.provenance,
                "request_schema_version": 1,
                "compact_plan_schema_version": 1,
                "canonical_full_plan_schema_version": 2,
                "local_port_condition_mode": "predicted",
                "organizer_source": "final_after_local_response_fusion",
            },
            outputs=outputs,
        )


__all__ = ["FrozenThermalChannelVerifier", "RETURN_OUTPUT_NAMES", "file_sha256"]

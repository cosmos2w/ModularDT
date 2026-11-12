"""Maintained native Thermal provider for R13-6 faithful and typed readers."""
from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np
import torch

from ..data.datasets import fit_global_normalizer
from ..data.development_split import read_case_catalog
from .content_hypergraph_task import ThermalContentPredictions
from .joint_member_routing_task import (
    _PairedMemberRoutingTask,
    ThermalJointPairPredictions,
)
from .member_routing_task import ThermalInteractionReusedMemberRoutingTask, _training_membership
from .joint_task import _load_data_binding, _read_selected_cases


CONFIG_SECTION_SCHEMA = "thermal_r13_6_member_routing_v1"
SOURCE_KIND = "scratch_r13_6_member_routing"
JOINT_BLOCKS = ("H_MM_applied", "H_EM_applied", "H_EE_applied", "H_ME_applied")

class _AnchoredPairedMemberRoutingTask(_PairedMemberRoutingTask):
    """Reuse R13-5's objective and typed health with the actual paired core call."""

    def _paired_predictions(self, prepared, physical_heat):
        self.model.validate_prepared(prepared)
        if physical_heat is not None:
            given = torch.as_tensor(physical_heat, device=prepared.scene_batch.module_features.device)
            expected = prepared.structure["heat_powers"]
            if given.shape != expected.shape or not torch.equal(given.to(expected), expected):
                raise ValueError("Heat changed: rebuild the complete joint prepared context.")
        paired = self.model.core.decode_paired(prepared.organization, prepared.query_batch)
        if not isinstance(paired, tuple) or len(paired) != 3:
            raise TypeError(f"{self.LINEAGE} paired decode must return full, restricted and shared work outputs.")
        full_output, restricted_output, shared_counts = paired

        def native(output):
            fields = getattr(output, "joint_fields", None)
            if not torch.is_tensor(fields) or fields.ndim != 3 or fields.shape[-1] != 4:
                raise ValueError(f"{self.LINEAGE} paired decoder must return [batch, queries, 4] joint_fields.")
            return self.model._extract_native_roles(prepared, fields[..., 3:4], fields[..., :3])

        counts_by_mode = {
            "full": {str(name): int(value) for name, value in full_output.work_counts.items()},
            "restricted": ({} if restricted_output is full_output else
                           {str(name): int(value) for name, value in restricted_output.work_counts.items()}),
        }
        work = {
            "native_fluid_queries": int(prepared.fluid_xy.shape[0] * prepared.fluid_xy.shape[1]),
            "native_union_rows": int(prepared.union.unique_counts.sum()),
            "native_local_pairs": int(prepared.query_batch.local_weights.numel()),
        }
        work.update({f"core_prepare_primary_{name}": int(value)
                     for name, value in prepared.organization.work_counts.items()})
        for mode, counts in counts_by_mode.items():
            work.update({f"core_decode_joint_{mode}_{name}": value for name, value in counts.items()})
        if not isinstance(shared_counts, Mapping):
            raise TypeError(f"{self.LINEAGE} paired decode shared work must be a mapping.")
        work.update({f"core_decode_joint_shared_{name}": int(value)
                     for name, value in shared_counts.items()})
        full_native = native(full_output)
        restricted_native = full_native if restricted_output is full_output else native(restricted_output)
        return ThermalJointPairPredictions(
            ThermalContentPredictions(prepared, full_native, work=counts_by_mode["full"]),
            ThermalContentPredictions(prepared, restricted_native, work=counts_by_mode["restricted"]), work)


def _fresh_member_routing_adapter(adapter_type, stats, model_config, *, heat_scale, seed, device):
    """Give a fresh raster-area model the same common initialization as nine inputs."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        if model_config["base_model_config"]["module_input_dim"] == 10:
            from .state_response_fork import PROJECTION, insert_area_column

            common_config = copy.deepcopy(model_config)
            common_config["base_model_config"]["module_input_dim"] = 9
            common = adapter_type(stats, core_config=common_config, heat_scale=heat_scale)
            model = adapter_type(stats, core_config=model_config, heat_scale=heat_scale)
            state = common.state_dict()
            state[PROJECTION] = insert_area_column(state[PROJECTION])
            model.load_state_dict(state, strict=True)
        else:
            model = adapter_type(stats, core_config=model_config, heat_scale=heat_scale)
        return model.to(device)


def _response_validation_ids(recipe, candidates):
    """Select only the explicit or first approximately ten percent of input IDs."""
    candidates = tuple(map(str, candidates))
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Validation population must contain unique input IDs.")
    evaluation = recipe.get("evaluation", {})
    explicit = evaluation.get("validation_case_ids")
    if explicit is None:
        fraction = float(evaluation.get("sample_fraction", .1))
        if not np.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("Validation sample_fraction must lie in (0, 1].")
        count = max(1, int(len(candidates) * fraction + .5))
        seed = evaluation.get("sample_seed")
        if seed is None:
            return candidates[:count]
        if type(seed) is not int:
            raise ValueError("Validation sample_seed must be an integer.")
        selected = set(np.random.default_rng(seed).choice(len(candidates), count, replace=False))
        return tuple(value for index, value in enumerate(candidates) if index in selected)
    selected = tuple(map(str, explicit))
    if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(candidates):
        raise ValueError("Explicit validation IDs must be unique members of the declared validation population.")
    return selected


def _response_catalog_binding(recipe):
    """Use native TRAIN and a declared validation split without a fixed25 manifest."""
    from honf_runtime.compat import resolve_demo_path

    dataset = recipe.get("dataset", {})
    if not dataset.get("data_path"):
        raise ValueError("catalog_train requires an explicit native Thermal dataset path.")
    if dataset.get("manifest_path") is not None:
        raise ValueError("catalog_train uses the dataset's own catalogue; remove the fixed25 manifest path.")
    data_path = resolve_demo_path(dataset["data_path"])
    if not data_path.is_file():
        raise FileNotFoundError(f"The native Thermal packed dataset is unavailable: {data_path}")
    source, catalog = read_case_catalog(data_path)
    available = tuple(str(row["case_id"]) for row in catalog if row["split"] == "train")
    selected = recipe.get("training", {}).get("train_case_ids")
    if selected is None:
        train_ids = available
    else:
        if isinstance(selected, (str, bytes)):
            raise ValueError("train_case_ids must be a sequence of original TRAIN IDs.")
        train_ids = tuple(map(str, selected))
    if not train_ids or len(set(train_ids)) != len(train_ids) or not set(train_ids).issubset(available):
        raise ValueError("catalog_train requires nonempty unique IDs from the native TRAIN split.")
    validation_split = recipe.get("evaluation", {}).get("validation_split", "validation")
    if not isinstance(validation_split, str) or not validation_split or validation_split.lower() in ("all", "train"):
        raise ValueError("catalog_train needs a declared non-TRAIN validation split.")
    validation_split = validation_split.lower()
    validation = tuple(str(row["case_id"]) for row in catalog if row["split"] == validation_split)
    if not validation:
        raise ValueError(f"No native cases exist in validation split {validation_split!r}.")
    # Reuse the existing catalogue's metadata identity; no new population freeze.
    manifest = {"manifest_sha256": source["metadata_sha256"]}
    return manifest, data_path, source, catalog, train_ids, validation, validation_split


def _build_thermal_member_routing_task(config, *, arm=None, device="cpu", task_type,
                                       variants, section_schema,
                                       source_kind, lineage,
                                       required_address_mode="ME", saved_adapter_state=None,
                                       saved_normalization_source=None):
    """Build a versioned adapter and the shared native TRAIN/VAL provider."""
    recipe = copy.deepcopy(dict(config))
    section = recipe.get("member_routing", {})
    model_config = section.get("model")
    selected_arm = str(arm or section.get("arm", ""))
    if section.get("schema") != section_schema:
        raise ValueError(f"{lineage} requires {section_schema}.")
    core_schema = model_config.get("schema") if isinstance(model_config, Mapping) else None
    variant = variants.get(core_schema)
    if (variant is None or selected_arm != variant["arm"]
            or not isinstance(model_config, Mapping)
            or model_config.get("architecture") != selected_arm
            or (required_address_mode is not None
                and model_config.get("address_mode") != required_address_mode)):
        raise ValueError(f"{lineage} model config must match its strict ME candidate schema and arm.")
    if recipe.get("initialization", {}).get("source_kind") != source_kind:
        raise ValueError(f"{lineage} initialization must declare {source_kind}.")

    catalog_train = (recipe.get("training_profile") == "state_response"
                     and recipe.get("training", {}).get("membership") == "catalog_train")
    if catalog_train:
        (manifest, data_path, catalog_source, catalog, train_ids,
         validation_ids, validation_split) = _response_catalog_binding(recipe)
        manifest_path = None
    else:
        manifest, data_path, manifest_path = _load_data_binding(
            recipe["dataset"].get("manifest_path"), recipe["dataset"].get("data_path"),
        )
        catalog_source, catalog = read_case_catalog(data_path)
        train_ids, validation_ids = _training_membership(recipe, catalog, manifest)
        validation_split = "test"
    if recipe.get("training_profile") == "state_response":
        validation_ids = _response_validation_ids(recipe, validation_ids)
    if saved_adapter_state is None:
        normalizer = fit_global_normalizer(data_path, train_ids)
    else:
        from ..data.datasets import H5Normalizer

        stats = saved_adapter_state.get("normalization_stats")
        if not isinstance(stats, Mapping) or not stats:
            raise ValueError("Inherited adapter state must contain source normalization statistics.")
        normalizer = H5Normalizer({key: np.asarray(value, dtype=np.float32) for key, value in stats.items()})
    training_cases = _read_selected_cases(
        data_path, train_ids, split="train", normalizer=normalizer, include_grid=False,
        include_raster_geometry=model_config["base_model_config"]["module_input_dim"] == 10,
        include_prescribed_patch=recipe.get("r16_7_prescribed") is not None,
    )
    validation_cases = _read_selected_cases(
        data_path, validation_ids, split=validation_split, normalizer=normalizer, include_grid=False,
        include_raster_geometry=model_config["base_model_config"]["module_input_dim"] == 10,
        include_prescribed_patch=recipe.get("r16_7_prescribed") is not None,
    )
    if saved_adapter_state is None:
        heat_values = np.concatenate([
            np.asarray(case["structure"]["heat_powers"])[
                np.asarray(case["structure"]["module_present"]) > 0.5
            ] for case in training_cases
        ])
        heat_scale = max(float(np.sqrt(np.mean(heat_values.astype(np.float64) ** 2))), 1e-6)
    else:
        heat_scale = float(saved_adapter_state["heat_scale"])
        if not np.isfinite(heat_scale) or heat_scale <= 0:
            raise ValueError("Inherited adapter heat scale must be finite and positive.")
    model = _fresh_member_routing_adapter(
        variant["adapter"], normalizer.stats, model_config, heat_scale=heat_scale,
        seed=recipe["initialization"].get("seed", 0), device=device)
    source_stat = data_path.stat()
    provider = task_type(
        model=model, training_cases=training_cases, validation_cases=validation_cases,
        manifest=manifest, data_path=data_path, recipe=recipe, device=device,
        manifest_path=manifest_path, catalog=catalog,
        source_binding={
            "dataset_path": str(data_path), "dataset_size_bytes": int(source_stat.st_size),
            "dataset_mtime_ns": int(source_stat.st_mtime_ns),
            "catalog_metadata_sha256": catalog_source["metadata_sha256"],
            "manifest_path": None if manifest_path is None else str(manifest_path),
            "manifest_sha256": manifest["manifest_sha256"],
            "normalization_source": ("TRAIN-only fit_global_normalizer" if saved_adapter_state is None
                                     else saved_normalization_source or "inherited source TRAIN600 adapter state"),
            "dataset_copy_created": False, "persistent_cache_created": False,
        },
    )
    if catalog_train:
        provider.protocol = "catalog_train"
    return model, provider

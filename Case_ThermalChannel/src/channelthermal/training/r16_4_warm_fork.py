"""Weight-only R16-4 forks of the literal mature Thermal H0 checkpoint."""
from __future__ import annotations

import copy

import torch

from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.unified_training import SamplingKey, TrainingEngine, _sampling_dataset_id
from ..member_routing import ThermalHyperedgeNonlinearReadAdapter
from .r13_6_member_routing_task import _build_thermal_member_routing_task
from .r13_8_anchored_task import (
    ARM_DETAIL, DETAIL_CONFIG_SCHEMA, DETAIL_CORE_SCHEMA, DETAIL_SOURCE_KIND, _VARIANTS,
)
from .thermal_state_response_task import ThermalStateResponseTask, _restore_response_scales


SOURCE_CHECKPOINT = "/data2/wanglz/ModularDT/r16_1_thermal_20261104/train600/thermal_state_response_h0_train600_s0/checkpoints/epoch_1000_model.pt"
SOURCE_AGE = 1000
SOURCE_KIND = "weight_only_r16_4_mature_h0"
FORK_SCHEMA = "thermal_r16_4_weight_only_fork_v1"
HYPEREDGE_SECTION_SCHEMA = "thermal_r16_4_nonlinear_hyperedge_member_routing_v1"
HYPEREDGE_ARM = "R16-4-Hyperedge-LocalECover"


def _new_core_config(source_recipe, mode):
    from honf_forward_core.interaction_reused_member_routing import HyperedgeNonlinearReadCore

    return HyperedgeNonlinearReadCore.from_detail_config(source_recipe["member_routing"]["model"], mode)


def make_warm_fork_recipe(source_recipe, *, arm, name, output_root, membership, local_horizon,
                          first_segment_epochs):
    """Derive a local-epoch recipe without changing the source scientific settings."""
    if arm not in ("B", "C", "G"):
        raise ValueError("The R16-4 arm must be B, C or G.")
    if (membership not in ("fixed25_train150_dev22", "full_train600")
            or type(local_horizon) is not int or type(first_segment_epochs) is not int
            or not 1 <= first_segment_epochs <= local_horizon):
        raise ValueError("Declare the fixed qualification or formal membership and local epoch horizon.")
    recipe = copy.deepcopy(source_recipe)
    recipe["name"] = str(name)
    recipe["storage"]["output_root"] = str(output_root)
    recipe["storage"]["formal_forward_run_alias"] = membership == "full_train600"
    recipe["initialization"] = {
        "source_kind": SOURCE_KIND, "source_checkpoint": SOURCE_CHECKPOINT,
        "pretrained_weights": True, "source_arm": ARM_DETAIL,
        "source_parent_age": SOURCE_AGE, "normalization_mode": "inherited_train600_stats", "seed": 0,
    }
    recipe["r16_4_warm_fork"] = {"schema": FORK_SCHEMA, "arm": arm,
                                "residual_initialization": "core_dedicated_r16_4_hyperedge_read"}
    recipe["training"].update(membership=membership, local_horizon=local_horizon,
                              first_segment_epochs=first_segment_epochs,
                              checkpoint_epochs=[local_horizon])
    if arm != "B":
        recipe["member_routing"]["schema"] = HYPEREDGE_SECTION_SCHEMA
        recipe["member_routing"]["arm"] = HYPEREDGE_ARM
        recipe["member_routing"]["model"] = _new_core_config(source_recipe, arm)
    return recipe


def validate_source_and_recipe(source, recipe):
    """Validate the literal source and the settings needed for matched adaptation."""
    fork = recipe.get("r16_4_warm_fork", {})
    initialization = recipe.get("initialization", {})
    if (fork != {"schema": FORK_SCHEMA, "arm": fork.get("arm"),
                 "residual_initialization": "core_dedicated_r16_4_hyperedge_read"}
            or fork.get("arm") not in ("B", "C", "G")
            or initialization != {"source_kind": SOURCE_KIND, "source_checkpoint": SOURCE_CHECKPOINT,
                "pretrained_weights": True, "source_arm": ARM_DETAIL, "source_parent_age": SOURCE_AGE,
                "normalization_mode": "inherited_train600_stats", "seed": 0}):
        raise ValueError("R16-4 requires its explicit literal H0 weight-only source declaration.")
    if (source.get("epoch") != SOURCE_AGE or source.get("current_epoch") != SOURCE_AGE
            or source.get("arm") != ARM_DETAIL
            or [row.get("epoch") for row in source.get("history", [])] != list(range(1, SOURCE_AGE + 1))):
        raise ValueError("The source must be the complete literal H0 epoch1000 checkpoint.")
    identity = source["experiment_identity"]
    parent = identity["recipe"]
    provider = identity["provider_identity"]
    engine = identity["engine_config"]
    if (parent["initialization"]["source_kind"] != DETAIL_SOURCE_KIND
            or parent["training"]["membership"] != "full_train600"
            or len(provider["training_case_ids"]) != 600
            or parent["member_routing"]["model"].get("reader_capability", "anchored") != "anchored"
            or parent["thermal_state_response"].get("pilot") is not None
            or parent["thermal_state_response"].get("finite_response_coefficient", .1) != .1
            or engine["sampling_version"] != SamplingKey.CASE_EPOCH_VERSION
            or engine["deterministic_case_order"] is not True
            or engine["seed"] != 0 or engine["training_mode"] != "joint"):
        raise ValueError("The declared source is not the original full-TRAIN600 H0 response lineage.")
    for key in ("dataset", "optimizer", "objective", "thermal_state_response"):
        if recipe.get(key) != parent.get(key):
            raise ValueError(f"Warm fork changed source scientific setting {key}.")
    for key in ("seed", "primary_queries", "effective_batch_size", "microbatch_cases",
                "optimizer_schedule_horizon", "sampling_arm"):
        if recipe["training"].get(key) != parent["training"].get(key):
            raise ValueError(f"Warm fork changed source training setting {key}.")
    if recipe["training"].get("membership") not in ("fixed25_train150_dev22", "full_train600"):
        raise ValueError("Warm fork requires the fixed TRAIN150 or source TRAIN600 membership.")
    source_section = parent["member_routing"]
    target_section = recipe["member_routing"]
    expected = copy.deepcopy(source_section)
    if fork["arm"] != "B":
        expected.update(schema=HYPEREDGE_SECTION_SCHEMA, arm=HYPEREDGE_ARM,
                        model=_new_core_config(parent, fork["arm"]))
    if target_section != expected:
        raise ValueError("Warm fork changed settings outside the declared B/C/G reader.")
    if (_sampling_dataset_id(provider) != identity["sampling_dataset_id"]
            or len(provider["response_training_family_ids"]) != 48):
        raise ValueError("Source native sampling namespace or original FIT48 membership changed.")
    _restore_response_scales(provider["response_scales"])
    return provider


def initialize_shared_weights(model, source_state, arm):
    """Strictly load every shared tensor and retain exactly the declared C/G tensors."""
    from honf_forward_core.interaction_reused_member_routing import HyperedgeNonlinearReadCore

    current = model.state_dict()
    extra = set() if arm == "B" else {"core." + key for key in HyperedgeNonlinearReadCore.NEW_STATE_KEYS}
    if arm not in ("B", "C", "G") or set(current) - set(source_state) != extra or set(source_state) - set(current):
        raise ValueError("Warm fork shared/new tensor inventories differ from the declared architecture.")
    for key, value in source_state.items():
        if not torch.is_tensor(value) or value.shape != current[key].shape or value.dtype != current[key].dtype:
            raise ValueError(f"Warm fork source tensor shape or dtype differs for {key}.")
    model.load_state_dict({**current, **source_state}, strict=True)
    return {"shared_tensor_names": sorted(source_state), "new_tensor_names": sorted(extra),
            "optimizer": "fresh AdamW; no inherited moments", "source_history": SOURCE_CHECKPOINT}


class ThermalR164WarmForkTask(ThermalStateResponseTask):
    """Native paired response task with local history and source-age FIT sampling."""

    def _anchored_variant(self):
        if self.model.CORE_SCHEMA == DETAIL_CORE_SCHEMA:
            return _VARIANTS[DETAIL_CORE_SCHEMA]
        if self.model.CORE_SCHEMA == ThermalHyperedgeNonlinearReadAdapter.CORE_SCHEMA:
            return {**_VARIANTS[DETAIL_CORE_SCHEMA], "adapter": ThermalHyperedgeNonlinearReadAdapter,
                    "candidate": "R16-4"}
        return None

    def _validate_initialization(self, initialization):
        if (self.source_kind != SOURCE_KIND or self.source_parent_age != SOURCE_AGE
                or initialization.get("source_checkpoint") != SOURCE_CHECKPOINT
                or initialization.get("pretrained_weights") is not True
                or initialization.get("source_arm") != ARM_DETAIL):
            raise ValueError("R16-4 requires the literal weight-only mature H0 source.")

    def fit_sampler_epoch(self, local_epoch):
        return SOURCE_AGE + super().fit_sampler_epoch(local_epoch)

    def identity_payload(self):
        result = super().identity_payload()
        result["normalization_fit_scope"] = "inherited source TRAIN600; no refit"
        result["loss_lineage"]["new_reader_epochs_start_at"] = 1
        result["loss_lineage"]["history_epoch_basis"] = "local; source backbone age is inherited separately"
        result["warm_fork"] = {"schema": FORK_SCHEMA, "arm": self.recipe["r16_4_warm_fork"]["arm"],
            "source_checkpoint": SOURCE_CHECKPOINT, "source_parent_age": SOURCE_AGE,
            "optimizer": "fresh AdamW", "fit_sampler_epoch": "source_parent_age + local_epoch",
            "native_sampler_epoch": "source_parent_age + local_epoch",
            "residual_initialization": "dedicated deterministic core initializer"}
        return result


class ThermalR164WarmForkEngine(TrainingEngine):
    def sampler_epoch(self, fit_epoch):
        return SOURCE_AGE + super().sampler_epoch(fit_epoch)

    def sampler_epoch_contract(self):
        return {"schema": "thermal_r16_4_source_age_sampler_v1", "source_parent_age": SOURCE_AGE,
                "fit_epoch_to_sampler_epoch": "source_parent_age + local_epoch"}


def build_thermal_r16_4_warm_fork_task(recipe, *, arm=None, device="cpu"):
    """Build an independent arm from the named source; never restore its optimizer/history."""
    source = load_trusted_checkpoint(SOURCE_CHECKPOINT, map_location="cpu")
    source_provider = validate_source_and_recipe(source, recipe)
    fork_arm = recipe["r16_4_warm_fork"]["arm"]
    model_arm = recipe["member_routing"]["arm"]
    if arm is not None and arm != model_arm:
        raise ValueError("Warm fork model arm differs from its recipe.")
    variant = (_VARIANTS[DETAIL_CORE_SCHEMA] if fork_arm == "B" else
               {**_VARIANTS[DETAIL_CORE_SCHEMA], "adapter": ThermalHyperedgeNonlinearReadAdapter})
    model, provider = _build_thermal_member_routing_task(
        recipe, arm=model_arm, device=device, task_type=ThermalR164WarmForkTask,
        variants={recipe["member_routing"]["model"]["schema"]: {**variant, "arm": model_arm}},
        section_schema=DETAIL_CONFIG_SCHEMA if fork_arm == "B" else HYPEREDGE_SECTION_SCHEMA,
        source_kind=SOURCE_KIND, lineage="R16-4 weight-only fork", required_address_mode=None,
        saved_adapter_state=source_provider["adapter_state"])
    source_adapter = source_provider["adapter_state"]
    target_adapter = model.state_payload()
    for key in ("normalization_stats", "heat_scale", "native_grid_nx_ny", "environment_grid_nx_ny"):
        if target_adapter[key] != source_adapter[key]:
            raise ValueError(f"Warm fork failed to preserve source adapter {key}.")
    for key in ("dataset_path", "dataset_size_bytes", "dataset_mtime_ns", "catalog_metadata_sha256", "manifest_sha256"):
        if provider.source_binding[key] != source_provider["source_binding"][key]:
            raise ValueError(f"Warm fork dataset binding differs from source {key}.")
    source_ids = source_provider["training_case_ids"]
    if (recipe["training"]["membership"] == "full_train600" and list(provider.training_case_ids) != source_ids
            or recipe["training"]["membership"] == "fixed25_train150_dev22"
            and (len(provider.training_case_ids) != 150 or not set(provider.training_case_ids) <= set(source_ids))):
        raise ValueError("Warm fork changed its declared TRAIN membership.")
    from ..data.datasets import H5Normalizer
    from .joint_task import _read_selected_cases
    import numpy as np

    val_ids = tuple(case_id for case_id in provider.catalog_case_ids
                    if provider.case_split_by_id[case_id] == "test" and case_id != "0273")
    if list(val_ids) != source_provider["validation_case_ids"]:
        raise ValueError("Warm fork validation membership differs from the source VAL89.")
    normalizer = H5Normalizer({key: np.asarray(value, dtype=np.float32)
                               for key, value in source_adapter["normalization_stats"].items()})
    provider.validation_cases = _read_selected_cases(provider.data_path, val_ids, split="test", normalizer=normalizer,
                                                      include_grid=False, include_raster_geometry=False)
    provider.validation_case_ids = val_ids
    provider._validation_index_by_id = {value: index for index, value in enumerate(val_ids)}
    provider.validation_scope = "canonical89_exposed_original_test_excluding_0273"
    provider._validation_batches_cache = None
    provider.configure_state_response(recipe["thermal_state_response"], response_scales=source_provider["response_scales"])
    if (list(provider.response_corpus.family_ids("fit")) != source_provider["response_training_family_ids"]
            or _sampling_dataset_id(provider.identity_payload()) != source["experiment_identity"]["sampling_dataset_id"]):
        raise ValueError("Warm fork changed the original FIT48 or native sampling namespace.")
    initialize_shared_weights(model, source["model_state_dict"], fork_arm)
    return model, provider

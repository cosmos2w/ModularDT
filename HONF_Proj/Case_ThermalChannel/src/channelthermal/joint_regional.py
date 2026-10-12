"""Fresh joint Thermal adapter for the regional hypergraph field core.

This adapter reuses the native Thermal input whitelist, receiver stencils and
shared-grid extraction, while its only learned module is one freshly built
``JointRegionalFieldCore``. Heating remains an application-time control and is
never part of the prepared configuration representation.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from honf_forward_core.interface_fields.interaction_preserving_joint import InteractionPreservingJointCore
from honf_forward_core.interface_fields.joint_regional import JointRegionalFieldCore

from .flow_curl import (
    NATIVE_CURL_READOUT_LAW,
    NativeCurlReadStencil,
    build_native_curl_read_stencil,
    native_curl_readout_contract,
    physical_uvp_native_curl,
)
from .source_response import (
    CONTEXT_KEYS,
    CONTEXT_WIDTH,
    ENVIRONMENT_WIDTH,
    SOURCE_WIDTH,
    THERMAL_INTERACTION_CONTROL_UNITS,
    PreparedNativeResponse,
    ThermalSourceResponse,
    _parameter_signature,
    _thermal_interaction_dependency,
)

JOINT_THERMAL_ID = "thermal_joint_regional_v1"
JOINT_THERMAL_CHECKPOINT_SCHEMA = "thermal_joint_regional_checkpoint_v1"
JOINT_THERMAL_CHANNEL_ORDER = ("u", "v", "p", "omega", "temperature")
JOINT_THERMAL_MODES = ("J-direct", "J-geometry", "J-H")
INTERACTION_PRESERVING_THERMAL_MODES = ("P", "P-G", "P-H")
NATIVE_ROUTING_THERMAL_MODES = ("P-R", "P-C", "P-R2")
NATIVE_ROUTING_CHILD_THERMAL_MODES = ("P-R", "P-C")
ALL_INTERACTION_PRESERVING_THERMAL_MODES = (
    INTERACTION_PRESERVING_THERMAL_MODES + NATIVE_ROUTING_THERMAL_MODES
)
NATIVE_CURL_THERMAL_FAMILY = "thermal_interaction_preserving_native_curl_v1"
NATIVE_CURL_THERMAL_CHECKPOINT_SCHEMA = "thermal_interaction_preserving_native_curl_checkpoint_v1"
NATIVE_ROUTING_THERMAL_FAMILY = "thermal_native_interaction_routing_v1"
NATIVE_ROUTING_THERMAL_CHECKPOINT_SCHEMA = "thermal_native_interaction_routing_checkpoint_v1"
NATIVE_ROUTING_R2_THERMAL_CHECKPOINT_SCHEMA = "thermal_native_interaction_routing_checkpoint_v2"
NATIVE_CURL_ROUTING_THERMAL_FAMILY = "thermal_native_curl_interaction_routing_v1"
NATIVE_CURL_ROUTING_THERMAL_CHECKPOINT_SCHEMA = "thermal_native_curl_interaction_routing_checkpoint_v1"
NATIVE_CURL_ROUTING_R2_THERMAL_CHECKPOINT_SCHEMA = "thermal_native_curl_interaction_routing_checkpoint_v2"
NATIVE_ROUTING_CONFIG_SCHEMA = "native_interaction_routing_v1"
NATIVE_ROUTING_R2_CONFIG_SCHEMA = "native_interaction_routing_v2"
NATIVE_ROUTING_CHILD_SCHEMA = "native_interaction_routing_child_v1"


def _thermal_interaction_family(mode: str, native_curl: bool) -> str:
    if mode in NATIVE_ROUTING_THERMAL_MODES:
        return (NATIVE_CURL_ROUTING_THERMAL_FAMILY if native_curl
                else NATIVE_ROUTING_THERMAL_FAMILY)
    return NATIVE_CURL_THERMAL_FAMILY if native_curl else InteractionPreservingThermalAdapter.FAMILY


def _thermal_native_routing_config(mode: str, seed: int) -> dict[str, Any]:
    if mode == "P-R2":
        return {
            "schema": NATIVE_ROUTING_R2_CONFIG_SCHEMA,
            "mode": mode,
            "policy": "learned_density",
            "initialization_seed": seed,
            "score_hidden": 32,
            "score_bound": 2.0,
            "views": 2,
            "feature_blocks": [64, 64],
        }
    config = {
        "schema": NATIVE_ROUTING_CONFIG_SCHEMA,
        "mode": mode,
        "policy": "learned_density" if mode == "P-R" else "destination_gain",
        "initialization_seed": seed,
    }
    if mode == "P-R":
        config.update(score_hidden=32, score_bound=2.0)
    elif mode == "P-C":
        config["control_hidden"] = 32
    else:
        raise ValueError("Native routing config requires mode P-R or P-C.")
    return config


def _is_sha256_hex(value: Any) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


FRESH_SEGMENTED_MULTIVIEW_WORKFLOW = "fresh_segmented_multiview_2500_v1"
FRESH_MATCHED_INITIALIZATION_SCHEMA = "matched_fresh_multiview_initialization_v1"
FRESH_MATCHED_ROUTING_PREFIXES = ["core.native_routing."]


def _validate_thermal_fresh_receipt_shape(
    marker: Mapping[str, Any], provider_identity: Mapping[str, Any], *,
    mode: str, seed: int, sampling_version: str, deterministic_case_order: bool,
) -> None:
    """Reject malformed or detached fresh-start metadata before model loading."""
    from honf_runtime.unified_training import _sampling_dataset_id

    expected_keys = {
        "schema", "active_mode", "native_validation_policy", "source_checkpoint",
        "inherited_parameter_names", "optimizer_state_start",
        "common_parameter_exclusion_prefixes", "common_initialization_sha256",
        "initial_model_state_sha256", "active_model_config_sha256",
        "provider_identity_sha256", "provider_scale_binding_sha256", "sampler_namespace",
    }
    if set(marker) != expected_keys:
        raise ValueError("Fresh Thermal initialization receipt has an incomplete or extended schema.")
    if marker.get("schema") != FRESH_MATCHED_INITIALIZATION_SCHEMA:
        raise ValueError("Fresh matched initialization schema is unsupported.")
    digest_names = (
        "common_initialization_sha256", "initial_model_state_sha256",
        "active_model_config_sha256", "provider_identity_sha256",
        "provider_scale_binding_sha256",
    )
    if any(not _is_sha256_hex(marker.get(name)) for name in digest_names):
        raise ValueError("Fresh Thermal initialization receipt contains a malformed SHA-256 digest.")
    if marker.get("common_parameter_exclusion_prefixes") != FRESH_MATCHED_ROUTING_PREFIXES:
        raise ValueError("Fresh Thermal common digest must exclude only private routing tensors.")
    if (marker.get("active_mode") != mode
            or marker.get("native_validation_policy") != "consolidated_route_flags_v1"
            or marker.get("source_checkpoint") is not None
            or marker.get("inherited_parameter_names") != []
            or marker.get("optimizer_state_start") != "fresh_zero_moments"):
        raise ValueError("Fresh Thermal receipt does not describe a no-parent, zero-moment start.")
    if (type(seed) is not int or type(sampling_version) is not str
            or type(deterministic_case_order) is not bool or deterministic_case_order is not True):
        raise ValueError("Fresh Thermal receipt sampler controls have noncanonical types.")
    expected_sampler = {
        "sampling_version": sampling_version,
        "seed": seed,
        "dataset_id": _sampling_dataset_id(provider_identity),
        "case_order": "seed_epoch_v1_deterministic_permutation",
        "case_order_arm_independent": True,
        "query_stream_keying": "dataset_seed_epoch_named_case_role_stream",
        "query_stream_arm_independent": True,
        "query_stream_update_independent": True,
        "query_stream_microbatch_independent": True,
    }
    sampler = marker.get("sampler_namespace")
    if (not isinstance(sampler, Mapping) or set(sampler) != set(expected_sampler)
            or dict(sampler) != expected_sampler
            or type(sampler.get("seed")) is not int
            or any(type(sampler.get(name)) is not bool for name in (
                "case_order_arm_independent", "query_stream_arm_independent",
                "query_stream_update_independent", "query_stream_microbatch_independent",
            ))):
        raise ValueError("Fresh Thermal receipt sampler namespace is not bound to this TRAIN provider.")


def _thermal_fresh_engine_mode(
    payload: Mapping[str, Any], engine_identity: Mapping[str, Any],
    provider_identity: Mapping[str, Any], recipe: Mapping[str, Any],
) -> str:
    """Validate the separate no-parent fresh P/R1/R2 engine-checkpoint contract."""
    from honf_runtime.unified_training import fresh_matched_recipe_seed

    declared_seed = fresh_matched_recipe_seed(recipe)
    if (payload.get("workflow") != FRESH_SEGMENTED_MULTIVIEW_WORKFLOW
            or engine_identity.get("workflow") != FRESH_SEGMENTED_MULTIVIEW_WORKFLOW):
        raise ValueError("Fresh Thermal engine checkpoint workflow marker is inconsistent.")
    if not JointThermalRegionalAdapter._native_input_equal(
            engine_identity.get("provider_identity"), provider_identity):
        raise ValueError("Fresh Thermal engine identity and provider binding disagree.")
    mode = recipe.get("mode")
    if (type(recipe.get("schema_version")) is not int or recipe.get("schema_version") != 1
            or recipe.get("comparison_protocol") != FRESH_SEGMENTED_MULTIVIEW_WORKFLOW
            or recipe.get("task") != "thermal"
            or recipe.get("fresh_matched_comparison") is not True
            or recipe.get("dataset_protocol") != "fixed25_v1"
            or type(recipe.get("seed")) is not int or recipe.get("seed") != declared_seed
            or type(recipe.get("total_epochs")) is not int or recipe.get("total_epochs") != 2500
            or mode not in ("P", "P-R", "P-R2")
            or recipe.get("flow_readout_law") != NATIVE_CURL_READOUT_LAW
            or recipe.get("native_validation_policy") != "consolidated_route_flags_v1"):
        raise ValueError("Fresh Thermal engine recipe is not the sealed fixed25 native-curl P/R1/R2 protocol.")
    engine_config = engine_identity.get("engine_config")
    schedule = recipe.get("engine_schedule")
    if (not isinstance(engine_config, Mapping) or not isinstance(schedule, Mapping)
            or engine_config.get("seed") != recipe.get("seed")
            or engine_config.get("total_epochs") != recipe.get("total_epochs")
            or engine_config.get("sampling_version") != schedule.get("sampling_version")
            or engine_config.get("sampling_version") != "case_epoch_v1"
            or engine_config.get("deterministic_case_order") is not True
            or engine_config.get("training_mode") != "joint"):
        raise ValueError("Fresh Thermal engine sampler/schedule differs from its resolved recipe.")
    model_config = provider_identity.get("model_config")
    if (not isinstance(model_config, Mapping)
            or provider_identity.get("task") != "ThermalChannel"
            or provider_identity.get("fresh_matched_comparison") is not True
            or provider_identity.get("dataset_split") != "fixed25_v1"
            or provider_identity.get("training_membership_scope") != "fixed25_v1_selected_train_150"
            or provider_identity.get("arm") != mode or model_config.get("mode") != mode
            or payload.get("arm") != mode):
        raise ValueError("Fresh Thermal engine arm, provider and recipe do not match.")
    contract = recipe.get("model_contract")
    route_contract = contract.get("native_routing") if isinstance(contract, Mapping) else None
    route_config = model_config.get("native_routing")
    expected_route_contract = {
        "P": {"enabled": False, "mode": "P"},
        "P-R": {
            "schema": NATIVE_ROUTING_CONFIG_SCHEMA, "mode": "P-R",
            "policy": "learned_density", "views": 1, "score_hidden": 32, "score_bound": 2.0,
        },
        "P-R2": {
            "schema": NATIVE_ROUTING_R2_CONFIG_SCHEMA, "mode": "P-R2",
            "policy": "learned_density", "views": 2, "feature_blocks": [64, 64],
            "score_hidden": 32, "score_bound": 2.0,
        },
    }[mode]
    if mode in NATIVE_ROUTING_THERMAL_MODES:
        if (type(route_contract) is not dict
                or type(route_contract.get("score_hidden")) is not int
                or type(route_contract.get("score_bound")) is not float
                or type(route_contract.get("views")) is not int):
            raise TypeError("Fresh Thermal recipe route widths and bounds must use their canonical numeric types.")
        if mode == "P-R2" and (
                type(route_contract.get("feature_blocks")) is not list
                or any(type(value) is not int for value in route_contract["feature_blocks"])):
            raise TypeError("Fresh Thermal P-R2 recipe view metadata must use exact integer types.")
    elif (type(route_contract) is not dict
          or type(route_contract.get("enabled")) is not bool
          or type(route_contract.get("mode")) is not str):
        raise TypeError("Fresh Thermal direct-P route contract must use its canonical boolean and string types.")
    if route_contract != expected_route_contract:
        raise ValueError("Fresh Thermal recipe route contract is not canonical for its arm.")
    if mode == "P":
        if route_config is not None:
            raise ValueError("Fresh Thermal direct P cannot declare provider native-routing metadata.")
    else:
        if (not isinstance(route_config, Mapping)
                or type(route_config.get("initialization_seed")) is not int
                or type(route_config.get("score_hidden")) is not int
                or type(route_config.get("score_bound")) is not float):
            raise TypeError("Fresh Thermal provider route metadata has noncanonical numeric types.")
        if mode == "P-R2" and (
                type(route_config.get("views")) is not int
                or type(route_config.get("feature_blocks")) is not list
                or any(type(value) is not int for value in route_config["feature_blocks"])):
            raise TypeError("Fresh Thermal provider P-R2 metadata must use exact integer types.")
        if (route_config.get("schema") != expected_route_contract["schema"]
                or route_config.get("mode") != mode
                or route_config.get("policy") != "learned_density"
                or route_config.get("initialization_seed") != recipe.get("seed")
                or route_config.get("score_hidden") != 32
                or route_config.get("score_bound") != 2.0
                or (mode == "P-R2" and (route_config.get("views") != 2
                                        or route_config.get("feature_blocks") != [64, 64]))):
            raise ValueError("Fresh Thermal recipe and provider route metadata differ.")
    if (engine_identity.get("native_validation_policy") != "consolidated_route_flags_v1"
            or engine_identity.get("initialization") != "fresh_common_trunk_private_router_rng"
            or engine_identity.get("inherited_parameter_names") != []
            or engine_identity.get("optimizer_state_start") != "fresh_zero_moments"
            or engine_identity.get("external_learned_field_files") != []
            or payload.get("branch_parent") is not None):
        raise ValueError("Fresh Thermal engine checkpoint contains inherited state or a mismatched policy.")
    marker = engine_identity.get("fresh_initialization")
    if not isinstance(marker, Mapping):
        raise TypeError("Fresh Thermal engine checkpoint has no matched-initialization receipt.")
    _validate_thermal_fresh_receipt_shape(
        marker, provider_identity, mode=mode, seed=recipe["seed"],
        sampling_version=engine_config["sampling_version"],
        deterministic_case_order=engine_config["deterministic_case_order"],
    )
    if (marker.get("active_mode") != mode
            or marker.get("native_validation_policy") != "consolidated_route_flags_v1"
            or marker.get("source_checkpoint") is not None
            or marker.get("inherited_parameter_names") != []
            or marker.get("optimizer_state_start") != "fresh_zero_moments"
            or engine_identity.get("initial_model_state_sha256")
            != marker.get("initial_model_state_sha256")):
        raise ValueError("Fresh Thermal engine receipt does not match its no-parent arm identity.")
    return str(mode)


def _thermal_engine_child_mode(
    payload: Mapping[str, Any], engine_identity: Mapping[str, Any],
    provider_identity: Mapping[str, Any], recipe: Mapping[str, Any],
) -> str:
    """Resolve an engine checkpoint's active mode without rewriting its parent recipe.

    Native-routing engine checkpoints are architecture children of literal P.
    Their top-level recipe therefore remains the exact parent recipe while the
    provider config and checkpoint arm identify P-R/P-C. The separate child
    receipt must bind those identities back to the exact parent checksum.
    """
    if not JointThermalRegionalAdapter._native_input_equal(
            engine_identity.get("provider_identity"), provider_identity):
        raise ValueError("Thermal engine identity and engine provider binding disagree.")
    model_config = provider_identity.get("model_config")
    if not isinstance(model_config, Mapping):
        raise TypeError("Thermal engine provider identity must contain its model config.")
    recipe_mode = recipe.get("mode")
    arm = payload.get("arm")
    provider_mode = model_config.get("mode")
    provider_arm = provider_identity.get("arm")
    branch_parent = payload.get("branch_parent")
    receipt = (branch_parent.get("architecture_child_initialization")
               if isinstance(branch_parent, Mapping) else None)
    routed_modes = set(NATIVE_ROUTING_THERMAL_MODES)

    if receipt is None:
        active_routed_modes = routed_modes.intersection((recipe_mode, arm, provider_mode, provider_arm))
        if active_routed_modes:
            raise ValueError("Thermal native-routing engine checkpoints require a verified architecture-child receipt.")
        if (arm != recipe_mode or provider_mode != recipe_mode
                or (provider_arm is not None and provider_arm != recipe_mode)):
            raise ValueError("Thermal engine checkpoint arm, recipe and provider model mode disagree.")
        return str(recipe_mode)
    if not isinstance(branch_parent, Mapping) or not isinstance(receipt, Mapping):
        raise TypeError("Thermal architecture-child checkpoint lineage must be mappings.")

    target_mode = arm
    if target_mode not in NATIVE_ROUTING_CHILD_THERMAL_MODES:
        raise ValueError("Thermal architecture-child arm must be P-R or P-C.")
    if not (recipe_mode == "P" and provider_mode == target_mode
            and provider_arm == target_mode and receipt.get("target_mode") == target_mode):
        raise ValueError("Thermal architecture-child target differs from its arm, retained P recipe or provider config.")
    source_sha256 = branch_parent.get("sha256")
    source_epoch = branch_parent.get("epoch")
    parent_identity = branch_parent.get("identity")
    if (receipt.get("schema") != NATIVE_ROUTING_CHILD_SCHEMA
            or receipt.get("dataset") != "ThermalChannel"
            or receipt.get("source_arm") != "P"
            or type(receipt.get("source_epoch")) is not int
            or receipt.get("source_epoch") != 1000
            or type(source_epoch) is not int or source_epoch != 1000
            or not _is_sha256_hex(source_sha256)
            or receipt.get("source_checkpoint_sha256") != source_sha256
            or not isinstance(parent_identity, Mapping)):
        raise ValueError("Thermal architecture-child receipt does not bind the exact e1000 P parent checksum and source.")

    parent_recipe = parent_identity.get("recipe")
    parent_provider = parent_identity.get("provider_identity")
    if (not isinstance(parent_recipe, Mapping) or not isinstance(parent_provider, Mapping)
            or dict(parent_recipe) != dict(recipe)
            or not JointThermalRegionalAdapter._native_input_equal(
                receipt.get("parent_provider_identity"), parent_provider)):
        raise ValueError("Thermal architecture-child receipt does not preserve the exact parent recipe/provider identity.")
    flow_law = recipe.get("flow_readout_law")
    if flow_law not in (None, NATIVE_CURL_READOUT_LAW):
        raise ValueError("Thermal architecture-child parent declares an unknown flow readout law.")
    native_curl = flow_law == NATIVE_CURL_READOUT_LAW
    parent_config = parent_provider.get("model_config")
    if (parent_provider.get("task") != "ThermalChannel"
            or parent_provider.get("arm") != "P"
            or parent_provider.get("model_family") != _thermal_interaction_family("P", native_curl)
            or not isinstance(parent_config, Mapping)
            or parent_config.get("mode") != "P"
            or parent_config.get("family") != _thermal_interaction_family("P", native_curl)
            or "native_routing" in parent_config
            or parent_config.get("regional_anchors") != 0):
        raise ValueError("Thermal architecture-child receipt parent is not the canonical direct-P family.")

    expected_child_config = copy.deepcopy(dict(parent_config))
    expected_family = _thermal_interaction_family(target_mode, native_curl)
    expected_child_config.update({
        "family": expected_family,
        "mode": target_mode,
        "native_routing": _thermal_native_routing_config(target_mode, int(parent_config["seed"])),
    })
    expected_child_provider = copy.deepcopy(dict(parent_provider))
    expected_child_provider.update({
        "arm": target_mode,
        "model_family": expected_family,
        "model_config": expected_child_config,
    })
    if not JointThermalRegionalAdapter._native_input_equal(
            dict(provider_identity), expected_child_provider):
        raise ValueError("Thermal architecture child changed its family/config or another parent provider binding.")
    routing_config = expected_child_config["native_routing"]
    if receipt.get("routing_configuration") != routing_config:
        raise ValueError("Thermal architecture-child receipt routing configuration differs from the child provider.")

    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not all(torch.is_tensor(value) for value in state.values()):
        raise TypeError("Thermal architecture-child state must be a tensor mapping.")
    routed_names = sorted(name for name in state if name.startswith("core.native_routing."))
    inherited_names = sorted(set(state) - set(routed_names))
    if (not routed_names or receipt.get("added_parameter_names") != routed_names
            or receipt.get("exact_parent_state_copied") is not True
            or receipt.get("all_parameters_trainable") is not True):
        raise ValueError("Thermal architecture-child receipt does not enumerate only its native-routing parameters.")
    for field, names in (("inherited_parameter_inventory", inherited_names),
                         ("new_parameter_inventory", routed_names)):
        inventory = receipt.get(field)
        if not isinstance(inventory, Mapping) or sorted(inventory) != names:
            raise ValueError(f"Thermal architecture-child receipt {field} differs from the checkpoint tensor inventory.")
        for name in names:
            record = inventory[name]
            value = state[name]
            if (not isinstance(record, Mapping)
                    or record.get("shape") != list(value.shape)
                    or record.get("dtype") != str(value.dtype)
                    or record.get("elements") != value.numel()):
                raise ValueError(f"Thermal architecture-child receipt tensor metadata differs for {name!r}.")
    for field in ("inherited_state_sha256", "new_parameter_initial_state_sha256",
                  "child_initial_state_sha256"):
        if not _is_sha256_hex(receipt.get(field)):
            raise ValueError(f"Thermal architecture-child receipt has an invalid {field}.")
    return str(target_mode)


def _validated_normalization_stats(stats: Mapping[str, Any]) -> dict[str, np.ndarray]:
    result = {str(name): np.asarray(value, dtype=np.float32).copy() for name, value in stats.items()}
    required = {
        "field_mean_by_channel": (5,),
        "field_std_by_channel": (5,),
        "interface_targets_std": (2,),
        "internal_temperature_std": (1,),
    }
    if "interface_targets_std" not in result and "interface_target_std" in result:
        result["interface_targets_std"] = result["interface_target_std"].copy()
    if "interface_targets_mean" not in result and "interface_target_mean" in result:
        result["interface_targets_mean"] = result["interface_target_mean"].copy()
    for name, shape in required.items():
        if name not in result or result[name].shape != shape:
            raise ValueError(f"Joint Thermal normalization requires {name} with shape {shape}.")
        if not np.isfinite(result[name]).all():
            raise ValueError(f"Joint Thermal normalization {name} must be finite.")
    if np.any(result["field_std_by_channel"] <= 0) or np.any(result["interface_targets_std"] <= 0):
        raise ValueError("Joint Thermal field and interface scales must be positive.")
    if float(result["internal_temperature_std"].reshape(-1)[0]) <= 0:
        raise ValueError("Joint Thermal material-temperature scale must be positive.")
    return result


class JointThermalRegionalAdapter(ThermalSourceResponse):
    """Thermal native adapter backed only by a new joint regional core.

    ``ThermalSourceResponse`` methods are reused without invoking its legacy
    constructor. In particular, this class owns no dormant source-response
    operator parameters: ``core`` is constructed directly as the new joint
    core, and every active parameter belongs to that core.
    """

    def __init__(
        self,
        *,
        mode: str = "J-H",
        normalization_stats: Mapping[str, Any],
        hidden: int = 128,
        message: int = 128,
        regional_anchors: int = 16,
        depth: int = 2,
        receiver_tile: int = 512,
        nx: int = 128,
        ny: int = 64,
        environment_nx: int = 24,
        environment_ny: int = 8,
        forcing_scale: float = 1.0,
        locality_prior_strength: float = 0.0,
        seed: int = 0,
    ) -> None:
        if mode not in JOINT_THERMAL_MODES:
            raise ValueError(f"Joint Thermal mode must be one of {JOINT_THERMAL_MODES}.")
        if receiver_tile < 1:
            raise ValueError("Joint Thermal receiver tile must be positive.")
        if nx < 2 or ny < 2 or environment_nx < 1 or environment_ny < 1:
            raise ValueError("Joint Thermal native and environment grids must be positive and nontrivial.")
        if not np.isfinite(forcing_scale) or forcing_scale <= 0:
            raise ValueError("Joint Thermal forcing scale must be positive and finite.")
        if isinstance(locality_prior_strength, (bool, np.bool_)):
            raise TypeError("Joint Thermal locality prior strength must be numeric, not boolean.")
        try:
            locality_prior_strength = float(locality_prior_strength)
        except (TypeError, ValueError) as error:
            raise ValueError("Joint Thermal locality prior strength must be finite and nonnegative.") from error
        if not np.isfinite(locality_prior_strength) or locality_prior_strength < 0:
            raise ValueError("Joint Thermal locality prior strength must be finite and nonnegative.")
        if locality_prior_strength and mode != "J-H":
            raise ValueError("Joint Thermal locality prior is only supported by J-H.")

        # Deliberately bypass ThermalSourceResponse.__init__: it creates the
        # older response operator, whose parameters are not part of this model.
        nn.Module.__init__(self)
        self.mode = str(mode)
        self.seed = int(seed)
        self.normalization_stats = _validated_normalization_stats(normalization_stats)
        self.receiver_tile = int(receiver_tile)
        self.nx, self.ny = int(nx), int(ny)
        self.environment_nx, self.environment_ny = int(environment_nx), int(environment_ny)
        self.forcing_scale = float(forcing_scale)
        self.locality_prior_strength = locality_prior_strength
        self.h_effective_eps, self.h_effective_max = 1.0e-3, 1.0e4
        self.environment_flow_context = False
        self.environment_flow_projection = None
        self.core = JointRegionalFieldCore(
            SOURCE_WIDTH,
            CONTEXT_WIDTH,
            ENVIRONMENT_WIDTH,
            spatial_dim=2,
            hidden=int(hidden),
            message=int(message),
            mode=self.mode,
            regional_anchors=int(regional_anchors),
            depth=int(depth),
            field_outputs=4,
            affine_outputs=1,
            query_width=0,
            forcing_scale=self.forcing_scale,
            locality_prior_strength=self.locality_prior_strength,
        )
        self.core_config = copy.deepcopy(dict(self.core.config))
        self._joint_model_config = {
            "mode": self.mode,
            "hidden": int(hidden),
            "message": int(message),
            "regional_anchors": int(regional_anchors),
            "depth": int(depth),
            "receiver_tile": self.receiver_tile,
            "nx": self.nx,
            "ny": self.ny,
            "environment_nx": self.environment_nx,
            "environment_ny": self.environment_ny,
            "forcing_scale": self.forcing_scale,
            "seed": self.seed,
        }
        if self.locality_prior_strength:
            self._joint_model_config["locality_prior_strength"] = self.locality_prior_strength

    @property
    def interaction_dependency(self):
        return _thermal_interaction_dependency(environment_flow_context=False)

    @property
    def interaction_control_units(self):
        return dict(THERMAL_INTERACTION_CONTROL_UNITS)

    @property
    def interaction_output_laws(self) -> dict[str, str]:
        """Laws for the two distinct output blocks in this joint model."""
        return {"flow": "nonlinear", "temperature_response": "affine"}

    @staticmethod
    def _source_identity(structure, centers, present):
        """Return the core's integer IDs and an ordered external identity catalogue."""
        batch, modules = centers.shape[:2]
        if modules < 1:
            raise ValueError("Thermal source identity catalogues require at least one source slot.")
        raw = structure.get("module_source_ids")
        catalogue_rows = None
        if raw is None:
            source_ids = torch.arange(modules, device=centers.device, dtype=torch.long)[None].expand(batch, -1)
            catalogue_rows = source_ids.detach().cpu().tolist()
        elif torch.is_tensor(raw):
            values = raw.to(device=centers.device)
            if values.ndim == 1 and values.shape[0] == modules:
                values = values[None].expand(batch, -1)
            if tuple(values.shape) != (batch, modules):
                raise ValueError("Thermal source identities must match the [B,M] source-slot catalogue.")
            if values.dtype == torch.bool or values.dtype.is_complex:
                raise TypeError("Thermal source identities must be integer-valued physical IDs or typed strings.")
            if values.is_floating_point():
                if (
                    not bool(torch.isfinite(values).all())
                    or not bool((values == values.round()).all())
                    or bool((values < -(2**63)).any())
                    or bool((values >= 2**63).any())
                ):
                    raise ValueError("Numeric Thermal source identities must be finite integral int64 values.")
            elif values.dtype == torch.uint64 and bool((values > torch.iinfo(torch.int64).max).any()):
                raise ValueError("Numeric Thermal source identities must fit in int64.")
            source_ids = values.to(dtype=torch.long)
            catalogue_rows = source_ids.detach().cpu().tolist()
        else:
            try:
                values = np.asarray(raw)
            except (TypeError, ValueError) as error:
                raise TypeError("Thermal source identities must be numeric IDs or a rectangular typed catalogue.") from error
            if values.ndim == 1 and values.shape[0] == modules:
                values = np.broadcast_to(values[None], (batch, modules))
            if tuple(values.shape) != (batch, modules):
                raise ValueError("Thermal source identities must match the [B,M] source-slot catalogue.")

            numeric = values.dtype.kind in "iu" or (
                values.dtype.kind == "f" and np.isfinite(values).all() and np.equal(values, np.rint(values)).all()
            )
            if values.dtype.kind == "f" and not numeric:
                raise ValueError("Numeric Thermal source identities must be finite integral int64 values.")
            if numeric:
                if values.dtype.kind == "u" and values.size and int(values.max()) > np.iinfo(np.int64).max:
                    raise ValueError("Numeric Thermal source identities must fit in int64.")
                if values.dtype.kind == "f" and values.size and (
                    float(values.min()) < -(2**63) or float(values.max()) >= 2**63
                ):
                    raise ValueError("Numeric Thermal source identities must fit in int64.")
                source_ids = torch.as_tensor(np.array(values, dtype=np.int64, copy=True), device=centers.device)
                catalogue_rows = source_ids.detach().cpu().tolist()
            else:
                flat = values.reshape(-1).tolist()
                if not all(isinstance(value, (str, bytes, np.str_, np.bytes_)) for value in flat):
                    raise TypeError("Typed Thermal source identities must contain only strings or bytes.")
                typed_rows = [
                    [value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
                     for value in row]
                    for row in values.tolist()
                ]
                if any(not value for row in typed_rows for value in row):
                    raise ValueError("Typed Thermal source identities must not be empty.")
                source_ids = torch.arange(modules, device=centers.device, dtype=torch.long)[None].expand(batch, -1)
                catalogue_rows = typed_rows

        active = present.detach() > 0.5
        for row_index in range(batch):
            active_ids = source_ids[row_index, active[row_index]]
            if bool((active_ids == -1).any()):
                raise ValueError("Active Thermal physical source identity -1 is reserved for inactive edge slots.")
            if active_ids.numel() != torch.unique(active_ids).numel():
                raise ValueError("Active Thermal physical source identities must be unique within each case.")
            typed_row = catalogue_rows[row_index]
            if typed_row and isinstance(typed_row[0], str):
                active_typed = [typed_row[index] for index in active[row_index].nonzero().flatten().cpu().tolist()]
                if len(active_typed) != len(set(active_typed)):
                    raise ValueError("Active Thermal typed source identities must be unique within each case.")

        if batch == 1:
            catalogue = tuple(catalogue_rows[0])
        else:
            catalogue = tuple(tuple(row) for row in catalogue_rows)
        return source_ids, catalogue

    def context_tensors(self, structure):
        """Preserve numeric physical IDs; map typed IDs through an explicit slot catalogue."""
        tensors = super().context_tensors(structure)
        source_ids, _ = self._source_identity(structure, tensors["centers"], tensors["present"])
        tensors["source_ids"] = source_ids
        return tensors

    @staticmethod
    def _bind_source_id_catalogue(prepared, catalogue):
        snapshot = copy.deepcopy(catalogue)
        prepared.source_id_catalogue = catalogue
        prepared.source_id_catalogue_snapshot = snapshot
        prepared.context.source_id_catalogue = catalogue
        prepared.context.source_id_catalogue_snapshot = copy.deepcopy(snapshot)
        prepared.response.source_id_catalogue = catalogue
        prepared.response.source_id_catalogue_snapshot = copy.deepcopy(snapshot)

    def adapter_config(self) -> dict[str, Any]:
        return {
            "nx": self.nx,
            "ny": self.ny,
            "environment_nx": self.environment_nx,
            "environment_ny": self.environment_ny,
            "forcing_scale": self.forcing_scale,
            "h_effective_eps": self.h_effective_eps,
            "h_effective_max": self.h_effective_max,
            "joint_output_laws": self.interaction_output_laws,
        }

    def _prepare_context_tensors(self, tensors, environment_flow_features, *,
                                 routing_trace=False, native_route_intervention=None):
        """Prepare joint state while declaring only the affine T block as a control response.

        The shared ``ThermalSourceResponse`` helper compares model-level
        ``output_law`` with a single-output dependency. A joint core has both a
        nonlinear field block and an affine temperature block, so validate the
        latter explicitly and attach the Thermal temperature dependency to the
        prepared affine response context.
        """
        dependency = _thermal_interaction_dependency(environment_flow_context=False)
        if getattr(self.core, "response_output_law", None) != dependency.output_law:
            raise ValueError("Joint Thermal affine response block disagrees with its heat dependency.")
        if environment_flow_features is not None:
            raise ValueError("Joint Thermal configuration does not accept predicted-flow environment features.")
        if isinstance(self.core, InteractionPreservingJointCore):
            prepared = self.core.prepare_context(
                **tensors,
                routing_trace=routing_trace,
                native_route_intervention=native_route_intervention,
            )
        else:
            if routing_trace or native_route_intervention is not None:
                raise ValueError("Native routing traces and interventions require the P-family Thermal core.")
            prepared = self.core.prepare_context(**tensors)
        prepared.dependency = dependency
        prepared.model_output_laws = self.interaction_output_laws
        return prepared

    def prepare_context(self, structure, *, environment_flow_features=None):
        tensors = self.context_tensors(structure)
        prepared = self._prepare_context_tensors(tensors, environment_flow_features)
        _, catalogue = self._source_identity(structure, tensors["centers"], tensors["present"])
        prepared.source_id_catalogue = catalogue
        prepared.source_id_catalogue_snapshot = copy.deepcopy(catalogue)
        return prepared

    def prepare_native(self, structure, fluid_xy, *, chunk_size=None,
                       retained_access_mass=None, receiver_edge_executor='dense',
                       routing_trace=False, native_route_intervention=None, **kwargs):
        tile = self.receiver_tile if chunk_size is None else int(chunk_size)
        if tile < 1:
            raise ValueError("Joint Thermal receiver chunk size must be positive.")
        if 'receiver_read_options' in kwargs:
            raise ValueError('Use the joint retained_access_mass and receiver_edge_executor arguments.')
        read_options = {
            'retained_access_mass': retained_access_mass,
            'receiver_edge_executor': receiver_edge_executor,
        }
        prepared = super().prepare_native(
            structure, fluid_xy, chunk_size=tile, receiver_read_options=read_options,
            routing_trace=routing_trace,
            native_route_intervention=native_route_intervention,
            **kwargs,
        )
        _, catalogue = self._source_identity(structure, prepared.context.centers, prepared.context.present)
        self._bind_source_id_catalogue(prepared, catalogue)
        prepared.normalization_stats_snapshot = {
            name: value.copy() for name, value in self.normalization_stats.items()
        }
        prepared.flow_receivers = fluid_xy
        prepared.flow_receiver_snapshot = fluid_xy.detach().clone()
        prepared.joint_read_options = read_options
        prepared.joint_read_options_snapshot = dict(read_options)
        return prepared

    def validate_prepared(self, prepared, structure=None):
        if not isinstance(prepared, PreparedNativeResponse) or prepared.owner != id(self):
            raise ValueError("Native response preparation belongs to another adapter/request.")
        if prepared.joint_read_options != prepared.joint_read_options_snapshot:
            raise ValueError('Prepared joint receiver execution changed; rebuild the native state.')
        if _parameter_signature(self) != prepared.parameter_signature:
            raise ValueError("Prepared Thermal adapter weights changed; rebuild the response operator.")
        if any(not torch.equal(value, snapshot) for value, snapshot in prepared.receiver_snapshots):
            raise ValueError("Native receiver catalogue changed; rebuild the response operator.")
        structure = prepared.structure if structure is None else structure
        current = {key: value for key, value in structure.items() if key in CONTEXT_KEYS}
        if set(current) != set(prepared.input_snapshot) or any(
            not self._native_input_equal(current[key], value)
            for key, value in prepared.input_snapshot.items()
        ):
            raise ValueError("Geometry/context changed; rebuild the response operator.")
        stats_snapshot = getattr(prepared, "normalization_stats_snapshot", None)
        if (
            stats_snapshot is None
            or set(stats_snapshot) != set(self.normalization_stats)
            or any(
                not np.array_equal(self.normalization_stats[name], stats_snapshot[name])
                for name in stats_snapshot
            )
        ):
            raise ValueError("Joint Thermal normalization changed; rebuild the prepared native state.")
        receivers = getattr(prepared, "flow_receivers", None)
        snapshot = getattr(prepared, "flow_receiver_snapshot", None)
        if not torch.is_tensor(receivers) or not torch.is_tensor(snapshot) or not torch.equal(receivers, snapshot):
            raise ValueError("Joint Thermal flow receiver catalogue changed; rebuild the prepared state.")
        catalogue = getattr(prepared, "source_id_catalogue", None)
        catalogue_snapshot = getattr(prepared, "source_id_catalogue_snapshot", None)
        response_catalogue = getattr(prepared.response, "source_id_catalogue", None)
        context_catalogue = getattr(prepared.context, "source_id_catalogue", None)
        response_catalogue_snapshot = getattr(prepared.response, "source_id_catalogue_snapshot", None)
        context_catalogue_snapshot = getattr(prepared.context, "source_id_catalogue_snapshot", None)
        if (
            catalogue is None
            or catalogue != catalogue_snapshot
            or catalogue != response_catalogue
            or catalogue != response_catalogue_snapshot
            or catalogue != context_catalogue
            or catalogue != context_catalogue_snapshot
        ):
            raise ValueError("Joint Thermal physical source-ID catalogue changed; rebuild the prepared state.")

    @staticmethod
    def _native_input_equal(left, right):
        if torch.is_tensor(left) or torch.is_tensor(right):
            return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            try:
                return np.array_equal(np.asarray(left), np.asarray(right))
            except (TypeError, ValueError):
                return False
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            return (
                isinstance(left, Mapping)
                and isinstance(right, Mapping)
                and set(left) == set(right)
                and all(JointThermalRegionalAdapter._native_input_equal(left[key], right[key]) for key in left)
            )
        if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
            return (
                isinstance(left, (tuple, list))
                and isinstance(right, (tuple, list))
                and len(left) == len(right)
                and all(JointThermalRegionalAdapter._native_input_equal(a, b) for a, b in zip(left, right, strict=True))
            )
        try:
            value = left == right
            return bool(value) if not torch.is_tensor(value) else value.numel() == 1 and bool(value)
        except (TypeError, ValueError, RuntimeError):
            return False

    def _apply_native_fp32(self, prepared, physical_heat, increment, compression):
        """Apply the all-source joint affine read without legacy compression kwargs."""
        if increment and compression is not None:
            raise ValueError("Joint Thermal increments retain every original physical source column.")
        fn = self.core.apply_increment if increment else self.core.apply_forcing
        grid_values = fn(prepared.response, physical_heat.to(prepared.source_present))
        role = {
            name: self._interpolate(grid_values, stencil).reshape(
                grid_values.shape[0], *prepared.role_shapes[name], -1
            )
            for name, stencil in prepared.stencils.items()
        }
        surface = role["surface"] * prepared.source_present[:, :, None, None]
        outside = role["outside"] * prepared.source_present[:, :, None, None]
        q = -prepared.interface_conductivity[:, None, :, None] / prepared.delta[:, None, :, None] * (
            outside - surface
        )
        return role, surface, outside, q

    def apply_native(
        self,
        prepared,
        physical_heat,
        *,
        increment: bool = False,
        compression=None,
        accumulation_dtype=None,
    ) -> dict[str, Any]:
        output = super().apply_native(
            prepared,
            physical_heat,
            increment=increment,
            compression=compression,
            accumulation_dtype=accumulation_dtype,
        )
        if increment:
            # The applied control is heat; it has no edge into configuration flow.
            flow = output["fluid_temperature"].new_zeros((*output["fluid_temperature"].shape[:-1], 4))
        else:
            flow_normalized = self.core.predict_fields(
                prepared.context,
                prepared.flow_receivers,
                chunk_size=self.receiver_tile,
                **prepared.joint_read_options,
            )
            if flow_normalized.ndim != 3 or flow_normalized.shape[-1] != 4:
                raise ValueError("Joint Thermal field head must return standardized [B,Q,4] u/v/p/omega.")
            means = flow_normalized.new_tensor(self.normalization_stats["field_mean_by_channel"][:4])
            scales = flow_normalized.new_tensor(self.normalization_stats["field_std_by_channel"][:4])
            flow = flow_normalized * scales + means
        output["pred_field"] = torch.cat((flow, output["fluid_temperature"]), dim=-1)
        return output

    def predict_native_sample(
        self,
        sample: Mapping[str, Any],
        device=None,
        *,
        ntheta: int = 64,
        routing_trace: bool = False,
        native_route_intervention: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if isinstance(ntheta, (bool, np.bool_)) or not isinstance(ntheta, (int, np.integer)):
            raise TypeError("Native interface receiver count ntheta must be a positive integer.")
        if ntheta < 1:
            raise ValueError("Native interface receiver count ntheta must be positive.")
        ntheta = int(ntheta)
        device = torch.device(device) if device is not None else next(self.parameters()).device
        structure = {}
        for key, value in sample["structure"].items():
            if key not in CONTEXT_KEYS:
                continue
            if key == "module_source_ids":
                structure[key] = value.to(device=device) if torch.is_tensor(value) else copy.deepcopy(value)
            else:
                structure[key] = torch.as_tensor(value, device=device, dtype=torch.float32)[None]
        fluid_xy = np.stack((np.asarray(sample["x_grid"]).reshape(-1), np.asarray(sample["y_grid"]).reshape(-1)), -1)
        fluid = torch.as_tensor(fluid_xy, device=device, dtype=torch.float32)[None]
        local = torch.as_tensor(sample["module_internal_query_points"], device=device, dtype=torch.float32)
        if local.ndim == 2:
            local = local[None]
        heat = torch.as_tensor(sample["structure"]["heat_powers"], device=device, dtype=torch.float32)[None]
        with torch.no_grad():
            prepared = self.prepare_native(
                structure, fluid, local_query_points=local, ntheta=ntheta,
                routing_trace=routing_trace,
                native_route_intervention=native_route_intervention,
                **({"native_solid_mask": torch.as_tensor(
                    sample["module_mask"], device=device, dtype=torch.bool
                )[None]} if getattr(self, "flow_readout_law", None) == NATIVE_CURL_READOUT_LAW else {}),
            )
            output = self.apply_native(prepared, heat)
        ny, nx = np.asarray(sample["x_grid"]).shape
        result = {
            "pred_field_grid": output["pred_field"][0].cpu().numpy().reshape(ny, nx, 5),
            "pred_interface": output["pred_interface"][0].cpu().numpy(),
            "pred_internal_temperature": output["pred_internal_temperature"][0, ..., 0].cpu().numpy(),
            "pred_port_condition": output["pred_port_condition"][0].cpu().numpy(),
            "initial_port_status": output["initial_port_status"],
        }
        if routing_trace:
            result["native_routing_trace"] = prepared.context.native_routing_trace
            result["native_routing_trace_metadata"] = prepared.context.native_routing_trace_metadata
            result["fluid_stencil_valid"] = (
                prepared.stencils["fluid"].valid[0].cpu().numpy().reshape(ny, nx)
            )
        return result

    def prepare_record(self, record, device=None, *, chunk_size=None, native_solid_mask=None):
        """Prepare native physical receivers from source IDs and input metadata."""
        device = torch.device(device) if device is not None else next(self.parameters()).device
        modules, context = record.design.modules, record.context.values
        tensor = lambda value: torch.as_tensor(value, device=device, dtype=torch.float32)
        centers = tensor([module.position_xy for module in modules])[None]
        present = tensor([float(module.active) for module in modules])[None]
        material = tensor(
            [context[key] for key in ("nu", "solid_alpha", "fluid_alpha", "solid_k", "fluid_k", "module_radius")]
        )[None]
        structure = {
            "module_centers": centers,
            "module_present": present,
            "material_params": material,
            "module_source_ids": tuple(module.module_id for module in modules),
        }
        for key in ("re", "u_in", "domain_length_x", "domain_length_y"):
            structure[key] = tensor([context[key]])[None]
        roles = record.output.roles
        fluid = tensor(roles["fluid_fields"].query_features[:, :2])[None]
        ids = tuple(module.module_id for module in modules)
        interface_rows = [
            np.flatnonzero(np.asarray(roles["interface"].receiver_module_ids) == module_id) for module_id in ids
        ]
        material_rows = [
            np.flatnonzero(np.asarray(roles["solid_temperature"].receiver_module_ids) == module_id)
            for module_id in ids
        ]
        if len({len(rows) for rows in interface_rows}) != 1 or len({len(rows) for rows in material_rows}) != 1:
            raise ValueError("Native Thermal role receivers require equal per-module counts.")
        ntheta = len(interface_rows[0])
        expected_theta = np.arange(ntheta) * (2 * np.pi / ntheta)
        for rows in interface_rows:
            query = np.asarray(roles["interface"].query_features)[rows]
            if not np.allclose(query[:, 0], expected_theta, rtol=0, atol=2.0e-6):
                raise ValueError("Native interface receivers must retain original angular ordering.")
        local = tensor(
            np.stack([np.asarray(roles["solid_temperature"].query_features)[rows, :2] for rows in material_rows])
        )[None]
        prepared = self.prepare_native(
            structure,
            fluid,
            local_query_points=local,
            ntheta=ntheta,
            chunk_size=chunk_size,
            **({"native_solid_mask": native_solid_mask} if native_solid_mask is not None else {}),
        )
        prepared.record_source_ids = ids
        prepared.interface_rows = interface_rows
        prepared.material_rows = material_rows
        prepared.interface_row_count = len(roles["interface"].query_features)
        prepared.material_row_count = len(roles["solid_temperature"].query_features)
        prepared.module_physical_heat = tensor([module.heating for module in modules])[None]
        return prepared

    @staticmethod
    def _record_role_outputs(prepared, output):
        interface = output["pred_interface"].new_empty(prepared.interface_row_count, 2)
        material = output["pred_internal_temperature"].new_empty(prepared.material_row_count, 1)
        for index, rows in enumerate(prepared.interface_rows):
            interface[torch.as_tensor(rows, device=interface.device)] = output["pred_interface"][0, index]
        for index, rows in enumerate(prepared.material_rows):
            material[torch.as_tensor(rows, device=material.device)] = output["pred_internal_temperature"][0, index]
        return interface, material

    def apply_record(self, prepared, physical_heat=None, *, accumulation_dtype=None):
        heat = prepared.module_physical_heat if physical_heat is None else physical_heat
        heat = torch.as_tensor(
            heat,
            device=prepared.source_present.device,
            dtype=accumulation_dtype or prepared.source_present.dtype,
        )
        if heat.ndim == 1:
            heat = heat[None]
        output = self.apply_native(prepared, heat, accumulation_dtype=accumulation_dtype)
        interface, material = self._record_role_outputs(prepared, output)
        return {"fluid_fields": output["pred_field"][0], "interface": interface, "solid_temperature": material}

    def apply_record_increment(
        self,
        prepared,
        delta_heat,
        *,
        compression=None,
        accumulation_dtype=None,
    ):
        if compression is not None:
            raise ValueError("Joint Thermal increments retain every original physical source column.")
        delta_heat = torch.as_tensor(
            delta_heat,
            device=prepared.source_present.device,
            dtype=accumulation_dtype or prepared.source_present.dtype,
        )
        if delta_heat.ndim == 1:
            delta_heat = delta_heat[None]
        output = self.apply_native(
            prepared,
            delta_heat,
            increment=True,
            accumulation_dtype=accumulation_dtype,
        )
        interface, material = self._record_role_outputs(prepared, output)
        return {"fluid_fields": output["pred_field"][0], "interface": interface, "solid_temperature": material}

    def predict_record(self, record, device=None):
        with torch.no_grad():
            output = self.apply_record(self.prepare_record(record, device))
        return {key: value.cpu().numpy() for key, value in output.items()}

    def model_config(self) -> dict[str, Any]:
        return copy.deepcopy(self._joint_model_config)

    def checkpoint_payload(self, *, provider_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "checkpoint_schema": JOINT_THERMAL_CHECKPOINT_SCHEMA,
            "model_family": JOINT_THERMAL_ID,
            "channel_order": list(JOINT_THERMAL_CHANNEL_ORDER),
            "model_config": self.model_config(),
            "adapter_config": self.adapter_config(),
            "model_state_dict": self.state_dict(),
            "global_normalization_stats": copy.deepcopy(self.normalization_stats),
        }
        if provider_identity is not None:
            payload["experiment_identity"] = {"provider_identity": copy.deepcopy(dict(provider_identity))}
        return payload


class InteractionPreservingThermalAdapter(JointThermalRegionalAdapter):
    """Opt-in Thermal wrapper around the source-conditioned P-family core.

    The native Thermal role extraction and affine physical-heat application
    remain shared with ``JointThermalRegionalAdapter``. The only differences
    are the freshly constructed core and its explicit mode/configuration.
    """

    FAMILY = "thermal_interaction_preserving_joint_v1"

    def __init__(
        self,
        *,
        mode: str = "P",
        normalization_stats: Mapping[str, Any],
        hidden: int = 128,
        message: int = 128,
        regional_anchors: int = 0,
        depth: int = 2,
        receiver_tile: int = 512,
        nx: int = 128,
        ny: int = 64,
        environment_nx: int = 24,
        environment_ny: int = 8,
        forcing_scale: float = 1.0,
        collective_width: int = 64,
        locality_prior_strength: float | None = None,
        seed: int = 0,
        max_sources: int = 12,
        flow_readout_law: str | None = None,
        factorized_affine_execution: bool = False,
    ) -> None:
        if flow_readout_law not in (None, NATIVE_CURL_READOUT_LAW):
            raise ValueError("Unknown interaction-preserving Thermal flow readout law.")
        if flow_readout_law and (nx != 128 or ny != 64):
            raise ValueError("The native-curl Thermal model is bound to the generator's 128x64 grid.")
        if mode not in ALL_INTERACTION_PRESERVING_THERMAL_MODES:
            raise ValueError(f"Interaction-preserving Thermal mode must be one of {INTERACTION_PRESERVING_THERMAL_MODES}.")
        if depth != 2:
            raise ValueError("The interaction-preserving Thermal core has exactly two pair rounds.")
        direct_mode = mode in ("P",) + NATIVE_ROUTING_THERMAL_MODES
        if direct_mode:
            if regional_anchors != 0 or locality_prior_strength not in (None, 0, 0.0):
                raise ValueError("Thermal P/P-R/P-C have no collective anchors or locality prior.")
        else:
            if regional_anchors < 1 or locality_prior_strength is None:
                raise ValueError("Thermal P-G/P-H require explicit collective anchors and locality strength.")
        if receiver_tile < 1 or nx < 2 or ny < 2 or environment_nx < 1 or environment_ny < 1:
            raise ValueError("Interaction-preserving Thermal grids and receiver tile must be positive.")
        if not np.isfinite(forcing_scale) or forcing_scale <= 0:
            raise ValueError("Interaction-preserving Thermal forcing scale must be positive and finite.")
        if max_sources < 1:
            raise ValueError("Thermal source capacity must be positive.")
        if type(factorized_affine_execution) is not bool:
            raise TypeError("factorized_affine_execution must be a bool.")

        # Initialize only the base adapter state; do not construct or serialize
        # an intermediate J core. Pair and collective tensors are initialized
        # under the P core's common seed contract.
        nn.Module.__init__(self)
        self.mode = str(mode)
        self.seed = int(seed)
        self.flow_readout_law = flow_readout_law
        if mode in NATIVE_ROUTING_THERMAL_MODES:
            self.family_id = (NATIVE_CURL_ROUTING_THERMAL_FAMILY if flow_readout_law
                              else NATIVE_ROUTING_THERMAL_FAMILY)
        else:
            self.family_id = NATIVE_CURL_THERMAL_FAMILY if flow_readout_law else self.FAMILY
        self.normalization_stats = _validated_normalization_stats(normalization_stats)
        self.receiver_tile = int(receiver_tile)
        self.nx, self.ny = int(nx), int(ny)
        self.environment_nx, self.environment_ny = int(environment_nx), int(environment_ny)
        self.forcing_scale = float(forcing_scale)
        self.locality_prior_strength = 0.0 if locality_prior_strength is None else float(locality_prior_strength)
        if not np.isfinite(self.locality_prior_strength) or self.locality_prior_strength < 0:
            raise ValueError("Interaction-preserving Thermal locality strength must be finite and nonnegative.")
        if mode in ("P",) + NATIVE_ROUTING_THERMAL_MODES and self.locality_prior_strength != 0:
            raise ValueError("Thermal P/P-R/P-C have no collective locality prior.")
        if mode == "P-G" and self.locality_prior_strength != 1.0:
            raise ValueError("The recovery P-G geometric locality prior is sealed to 1.0.")
        if mode == "P-H" and self.locality_prior_strength != 1.0:
            raise ValueError("The recovery P-H locality prior is sealed to 1.0.")
        self.h_effective_eps, self.h_effective_max = 1.0e-3, 1.0e4
        self.environment_flow_context = False
        self.environment_flow_projection = None
        self.core = InteractionPreservingJointCore(
            SOURCE_WIDTH,
            CONTEXT_WIDTH,
            ENVIRONMENT_WIDTH,
            spatial_dim=2,
            hidden=int(hidden),
            message=int(message),
            mode=self.mode,
            collective_width=int(collective_width),
            regional_anchors=int(regional_anchors),
            field_outputs=3 if flow_readout_law else 4,
            **({"initialization_reference_field_outputs": 4} if flow_readout_law else {}),
            affine_outputs=1,
            query_width=0,
            max_sources=int(max_sources),
            forcing_scale=self.forcing_scale,
            zero_offset=True,
            initialization_seed=int(seed),
            locality_prior_strength=(self.locality_prior_strength if mode in ("P-G", "P-H") else None),
            factorized_affine_execution=factorized_affine_execution,
            native_control_hidden=(32 if mode == "P-C" else None),
        )
        self.core_config = copy.deepcopy(dict(self.core.config))
        self._joint_model_config = {
            "family": self.family_id,
            "mode": self.mode,
            "source_width": SOURCE_WIDTH,
            "context_width": CONTEXT_WIDTH,
            "environment_width": ENVIRONMENT_WIDTH,
            "spatial_dim": 2,
            "hidden": int(hidden),
            "message": int(message),
            "collective_width": int(collective_width),
            "regional_anchors": int(regional_anchors),
            "depth": int(depth),
            "max_sources": int(max_sources),
            "field_outputs": 3 if flow_readout_law else 4,
            "affine_outputs": 1,
            "query_width": 0,
            "receiver_tile": self.receiver_tile,
            "nx": self.nx,
            "ny": self.ny,
            "environment_nx": self.environment_nx,
            "environment_ny": self.environment_ny,
            "forcing_scale": self.forcing_scale,
            "zero_offset": True,
            "seed": self.seed,
        }
        if flow_readout_law:
            self._joint_model_config["flow_readout_law"] = flow_readout_law
            self._joint_model_config["initialization_reference_field_outputs"] = 4
        if mode in ("P-G", "P-H"):
            self._joint_model_config["locality_prior_strength"] = self.locality_prior_strength
        if mode in NATIVE_ROUTING_THERMAL_MODES:
            self._joint_model_config["native_routing"] = copy.deepcopy(
                self.core.config["native_routing"])
        if factorized_affine_execution:
            self._joint_model_config["factorized_affine_execution"] = True

    def prepare_native(self, structure, fluid_xy, *, chunk_size=None, retained_access_mass=None,
                       receiver_edge_executor="dense", native_solid_mask=None,
                       routing_trace=False, native_route_intervention=None, **kwargs):
        """Use the inherited physical receiver builder without J-only read knobs."""
        if retained_access_mass is not None or receiver_edge_executor != "dense":
            raise ValueError("The interaction-preserving Thermal core uses its declared dense direct read.")
        tile = self.receiver_tile if chunk_size is None else int(chunk_size)
        if tile < 1:
            raise ValueError("Thermal receiver chunk size must be positive.")
        if "receiver_read_options" in kwargs:
            raise ValueError("P-family receiver preparation does not accept legacy retained-access options.")
        prepared = ThermalSourceResponse.prepare_native(
            self, structure, fluid_xy, chunk_size=tile,
            routing_trace=routing_trace,
            native_route_intervention=native_route_intervention,
            **kwargs,
        )
        _, catalogue = self._source_identity(
            structure, prepared.context.centers, prepared.context.present
        )
        self._bind_source_id_catalogue(prepared, catalogue)
        prepared.normalization_stats_snapshot = {
            name: value.copy() for name, value in self.normalization_stats.items()
        }
        prepared.flow_receivers = fluid_xy
        prepared.flow_receiver_snapshot = fluid_xy.detach().clone()
        prepared.joint_read_options = {}
        prepared.joint_read_options_snapshot = {}
        if self.flow_readout_law:
            if native_solid_mask is None:
                raise ValueError("Native-curl preparation requires the saved geometry-only native_solid_mask.")
            stencil = build_native_curl_read_stencil(
                fluid_xy, prepared.context.lengths, native_solid_mask, nx=self.nx, ny=self.ny,
            )
            prepared.native_curl_stencil = stencil
            prepared.native_curl_stencil_snapshot = {
                name: value.detach().clone() for name, value in vars(stencil).items()
            }
            prepared.native_solid_mask = native_solid_mask
            prepared.native_solid_mask_snapshot = native_solid_mask.detach().clone()
            prepared.flow_query_receivers = stencil.query_xy
        elif native_solid_mask is not None:
            raise ValueError("Independent-head preparation does not accept native-curl mask metadata.")
        return prepared

    def validate_prepared(self, prepared, structure=None):
        super().validate_prepared(prepared, structure)
        if self.flow_readout_law:
            stencil = getattr(prepared, "native_curl_stencil", None)
            snapshot = getattr(prepared, "native_curl_stencil_snapshot", None)
            if (not isinstance(stencil, NativeCurlReadStencil) or not isinstance(snapshot, dict)
                    or set(vars(stencil)) != set(snapshot)
                    or any(not torch.equal(getattr(stencil, name), saved) for name, saved in snapshot.items())):
                raise ValueError("Prepared native-curl stencil changed; rebuild the receiver state.")
            mask = getattr(prepared, "native_solid_mask", None)
            saved_mask = getattr(prepared, "native_solid_mask_snapshot", None)
            if (not torch.is_tensor(mask) or not torch.is_tensor(saved_mask)
                    or not torch.equal(mask, saved_mask)
                    or not torch.equal(prepared.flow_query_receivers, snapshot["query_xy"])):
                raise ValueError("Prepared native-curl geometry mask or node receivers changed; rebuild the state.")

    def apply_native(self, prepared, physical_heat, *, increment=False, compression=None, accumulation_dtype=None):
        if not self.flow_readout_law:
            return super().apply_native(
                prepared, physical_heat, increment=increment, compression=compression,
                accumulation_dtype=accumulation_dtype,
            )
        # Apply the same affine temperature extraction, then the actual three
        # nonlinear configuration projections and differentiable native curl.
        output = ThermalSourceResponse.apply_native(
            self, prepared, physical_heat, increment=increment, compression=compression,
            accumulation_dtype=accumulation_dtype,
        )
        if increment:
            flow = output["fluid_temperature"].new_zeros((*output["fluid_temperature"].shape[:-1], 4))
        else:
            normalized_uvp = self.core.predict_fields(
                prepared.context, prepared.flow_query_receivers, chunk_size=self.receiver_tile,
            )
            flow = physical_uvp_native_curl(
                normalized_uvp, prepared.native_curl_stencil,
                self.normalization_stats["field_mean_by_channel"],
                self.normalization_stats["field_std_by_channel"],
            )
        output["pred_field"] = torch.cat((flow, output["fluid_temperature"]), dim=-1)
        return output

    def adapter_config(self):
        config = super().adapter_config()
        if self.flow_readout_law:
            config["flow_readout_law"] = self.flow_readout_law
            config["native_curl_contract"] = native_curl_readout_contract()
        return config

    def model_config(self) -> dict[str, Any]:
        config = copy.deepcopy(self._joint_model_config)
        if self.core.factorized_affine_execution:
            config["factorized_affine_execution"] = True
        else:
            config.pop("factorized_affine_execution", None)
        return config

    def checkpoint_payload(self, *, provider_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self.mode in NATIVE_ROUTING_THERMAL_MODES:
            if self.mode == "P-R2":
                checkpoint_schema = (NATIVE_CURL_ROUTING_R2_THERMAL_CHECKPOINT_SCHEMA
                                     if self.flow_readout_law else
                                     NATIVE_ROUTING_R2_THERMAL_CHECKPOINT_SCHEMA)
            else:
                checkpoint_schema = (NATIVE_CURL_ROUTING_THERMAL_CHECKPOINT_SCHEMA
                                     if self.flow_readout_law else
                                     NATIVE_ROUTING_THERMAL_CHECKPOINT_SCHEMA)
        else:
            checkpoint_schema = (NATIVE_CURL_THERMAL_CHECKPOINT_SCHEMA
                                 if self.flow_readout_law else
                                 "thermal_interaction_preserving_joint_checkpoint_v1")
        payload = super().checkpoint_payload(provider_identity=provider_identity)
        payload["model_family"] = self.family_id
        payload["checkpoint_schema"] = checkpoint_schema
        return payload


def load_joint_thermal_checkpoint(source: Mapping[str, Any] | str | Path, device="cpu") -> JointThermalRegionalAdapter:
    """Load one standalone candidate without consulting any legacy model path."""
    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        payload = torch.load(Path(source), map_location="cpu", weights_only=False)
    if payload.get("checkpoint_schema") != JOINT_THERMAL_CHECKPOINT_SCHEMA:
        raise ValueError("Checkpoint is not a self-contained joint Thermal regional model.")
    if payload.get("model_family") != JOINT_THERMAL_ID:
        raise ValueError("Checkpoint model family differs from the joint Thermal regional adapter.")
    if payload.get("channel_order") != list(JOINT_THERMAL_CHANNEL_ORDER):
        raise ValueError("Checkpoint Thermal channel order is not the native u/v/p/omega/temperature order.")
    if any(key in payload for key in ("parent_checkpoint", "flow_checkpoint", "thermal_parent", "flow_parent")):
        raise ValueError("Joint Thermal checkpoints cannot bind an external learned field parent.")
    config = payload.get("model_config")
    stats = payload.get("global_normalization_stats")
    state = payload.get("model_state_dict")
    if not isinstance(config, Mapping) or not isinstance(stats, Mapping) or not isinstance(state, Mapping):
        raise TypeError("Joint Thermal checkpoint config, normalization and tensors must be mappings.")
    model = JointThermalRegionalAdapter(normalization_stats=stats, **dict(config))
    if any(not name.startswith("core.") for name in state):
        raise ValueError("Joint Thermal checkpoint contains tensors outside its single active core.")
    model.load_state_dict(state, strict=True)
    if any(not name.startswith("core.") for name, _ in model.named_parameters()):
        raise RuntimeError("Joint Thermal adapter registered parameters outside the new joint core.")
    return model.to(device)


def load_interaction_preserving_thermal_checkpoint(
    source: Mapping[str, Any] | str | Path, device="cpu"
) -> InteractionPreservingThermalAdapter:
    """Load a P-family Thermal model or an engine checkpoint without legacy parents.

    The maintained trainer saves ``model_state_dict`` inside its generic engine
    checkpoint, while standalone exports use the explicit adapter schema. Both
    paths are validated against the same native dimensions, coordinate frame,
    output laws, initialization seed, and forcing contract before strict state
    loading.
    """
    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        payload = torch.load(Path(source), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("Interaction-preserving Thermal checkpoint must be a mapping.")
    if any(key in payload for key in (
        "parent_checkpoint", "flow_checkpoint", "thermal_parent", "flow_parent", "parent_model"
    )):
        raise ValueError("Interaction-preserving Thermal checkpoints cannot bind an external learned parent.")

    engine_identity = None
    fresh_engine_checkpoint = False
    standalone_schemas = {
        "thermal_interaction_preserving_joint_checkpoint_v1": (False, False),
        NATIVE_CURL_THERMAL_CHECKPOINT_SCHEMA: (True, False),
        NATIVE_ROUTING_THERMAL_CHECKPOINT_SCHEMA: (False, True),
        NATIVE_ROUTING_R2_THERMAL_CHECKPOINT_SCHEMA: (False, True),
        NATIVE_CURL_ROUTING_THERMAL_CHECKPOINT_SCHEMA: (True, True),
        NATIVE_CURL_ROUTING_R2_THERMAL_CHECKPOINT_SCHEMA: (True, True),
    }
    if payload.get("checkpoint_schema") in standalone_schemas:
        native_curl, native_routing = standalone_schemas[payload["checkpoint_schema"]]
        if native_routing:
            expected_family = (NATIVE_CURL_ROUTING_THERMAL_FAMILY if native_curl
                               else NATIVE_ROUTING_THERMAL_FAMILY)
        else:
            expected_family = NATIVE_CURL_THERMAL_FAMILY if native_curl else InteractionPreservingThermalAdapter.FAMILY
        if payload.get("model_family") != expected_family:
            raise ValueError("Thermal P-family standalone checkpoint family differs from its schema.")
        if payload.get("channel_order") != list(JOINT_THERMAL_CHANNEL_ORDER):
            raise ValueError("Thermal P-family channel order must be native u/v/p/omega/temperature.")
        config = payload.get("model_config")
        adapter = payload.get("adapter_config")
        stats = payload.get("global_normalization_stats")
        state = payload.get("model_state_dict")
    elif payload.get("checkpoint_schema_version") == 1:
        fresh_engine_checkpoint = payload.get("workflow") == FRESH_SEGMENTED_MULTIVIEW_WORKFLOW
        if not fresh_engine_checkpoint and payload.get("workflow") != "joint_regional_fields_v1":
            raise ValueError("Engine checkpoint workflow is not the maintained joint regional trainer.")
        engine_identity = payload.get("experiment_identity")
        if not isinstance(engine_identity, Mapping):
            raise TypeError("Engine checkpoint has no sealed experiment identity.")
        provider_identity = engine_identity.get("provider_identity")
        recipe = engine_identity.get("recipe")
        if not isinstance(provider_identity, Mapping) or not isinstance(recipe, Mapping):
            raise TypeError("Engine checkpoint must bind both provider identity and the resolved recipe.")
        if fresh_engine_checkpoint:
            child_mode = _thermal_fresh_engine_mode(payload, engine_identity, provider_identity, recipe)
        else:
            if (engine_identity.get("workflow") != payload.get("workflow")
                    or recipe.get("fresh_matched_comparison") is True):
                raise ValueError("Fresh Thermal engine checkpoints require their separate workflow and receipt.")
            child_mode = None
        if (recipe.get("task") != "thermal" or recipe.get("mode") not in ALL_INTERACTION_PRESERVING_THERMAL_MODES
                or engine_identity.get("external_learned_field_files") != []):
            raise ValueError("Engine checkpoint is not a fresh, self-contained Thermal P-family run.")
        if provider_identity.get("task") != "ThermalChannel":
            raise ValueError("Engine checkpoint provider is not the maintained Thermal task.")
        if child_mode is None:
            child_mode = _thermal_engine_child_mode(
                payload, engine_identity, provider_identity, recipe,
            )
        native_curl = recipe.get("flow_readout_law") == NATIVE_CURL_READOUT_LAW
        native_routing = child_mode in NATIVE_ROUTING_THERMAL_MODES
        if recipe.get("flow_readout_law") not in (None, NATIVE_CURL_READOUT_LAW):
            raise ValueError("Unknown sealed Thermal flow readout law.")
        expected_family = _thermal_interaction_family(child_mode, native_curl)
        if provider_identity.get("model_family") != expected_family:
            raise ValueError("Engine checkpoint provider model family is not the P-family Thermal adapter.")
        contract = recipe.get("model_contract")
        if not isinstance(contract, Mapping):
            raise TypeError("Engine checkpoint recipe has no explicit Thermal frame/architecture contract.")
        source_contract = contract.get("source")
        environment_contract = contract.get("environment")
        receiver_contract = contract.get("receivers")
        collective_contract = contract.get("collective")
        if not all(isinstance(value, Mapping) for value in (
            source_contract, environment_contract, receiver_contract, collective_contract
        )):
            raise TypeError("Engine checkpoint Thermal source, environment, receiver and collective contracts are incomplete.")
        graph_mode = child_mode in ("P-G", "P-H")
        expected_collective = (
            {"enabled": False, "placement": "none"} if not graph_mode else {
                "enabled": True,
                "placement": "one typed block between the two pair-message rounds",
                "physical_source_edges": "one edge centered on each active source; Thermal scale is physical source radius; Wind scale is rotor diameter in D",
                "regional_anchors": "K deterministic Halton anchors from radical-inverse bases (2,3) in 2D or (2,3,5) in 3D, indices 1..K mapped over domain_origin + unit * lengths",
                "regional_anchor_scale": "per-axis domain lengths / K^(1/spatial_dimension)",
                "membership": "source and geometry-token measure-aware softmax; physical measure enters donor mass exactly once",
                "mode_incidence_and_access": "P-G uses fixed geometric incidence/access; P-H adds learned score residuals over the same lambda=1 geometric prior, with zero-initialized residual scores",
                "return_rule": "normalized transpose of typed donor incidence for context sharing",
                "initialization": "zero node/environment/receiver output projections recover P exactly at initialization",
            }
        )
        if (contract.get("input_frame") != "packed Thermal native x-y coordinates; source centres/radii normalized by physical domain lengths"
                or contract.get("native_shared_grid_shape") != [128, 64]
                or source_contract.get("feature_width") != 8
                or source_contract.get("context_width") != 14
                or source_contract.get("max_sources") != 12
                or environment_contract.get("feature_width") != 8
                or environment_contract.get("shape") != [24, 8]
                or environment_contract.get("count") != 192
                or receiver_contract.get("frame") != "packed native x-y physical coordinates"
                or receiver_contract.get("feature_width") != 0
                or receiver_contract.get("tile") != 512
                or dict(collective_contract) != expected_collective):
            raise ValueError("Engine checkpoint Thermal frame, grid, or collective placement is not canonical.")
        readouts = contract.get("readouts", {})
        expected_flow_head = {
            "law": "nonlinear source-conditioned field read", "outputs": 4,
            "order": ["u", "v", "p", "omega"],
        }
        if native_curl:
            expected_flow_head = {
                "law": "nonlinear source-conditioned field read", "outputs": 3,
                "order": ["u", "v", "p"], "initialization_reference_field_outputs": 4,
            }
            if (readouts.get("derived_omega") != native_curl_readout_contract()
                    or receiver_contract.get("native_solid_mask") != "saved boolean [B,ny,nx] geometry-only receiver metadata; excluded from learned context"):
                raise ValueError("Engine checkpoint native-curl operator/mask contract is not canonical.")
        elif "derived_omega" in readouts or "native_solid_mask" in receiver_contract:
            raise ValueError("Independent-head checkpoint cannot declare a derived-omega readout.")
        if (readouts.get("flow_head") != expected_flow_head
                or readouts.get("temperature_head") != {
                    "law": "source-resolved affine heat response", "outputs": 1, "zero_offset": True}
                or readouts.get("source_read_networks") != {
                    "flow": "field_source_read", "temperature": "affine_source_read", "distinct": True}
                or readouts.get("forcing_scale") != 1.0):
            raise ValueError("Engine checkpoint Thermal output/read-network/forcing contract is not canonical.")
        config = provider_identity.get("model_config")
        adapter = provider_identity.get("adapter_config")
        stats = provider_identity.get("normalization_stats")
        state = payload.get("model_state_dict")
    else:
        raise ValueError("Checkpoint is neither a standalone Thermal P-family export nor an engine checkpoint.")

    if not all(isinstance(value, Mapping) for value in (config, adapter, stats, state)):
        raise TypeError("Thermal P-family config, adapter, normalization and state must be mappings.")
    config = dict(config)
    adapter = dict(adapter)
    mode = config.get("mode")
    if mode not in ALL_INTERACTION_PRESERVING_THERMAL_MODES:
        raise ValueError("Thermal checkpoint config has an unsupported direct or collective mode.")
    native_routing = mode in NATIVE_ROUTING_THERMAL_MODES
    expected_family = _thermal_interaction_family(mode, native_curl)
    if config.get("family") != expected_family:
        raise ValueError("Thermal P-family checkpoint config has a different model family.")
    checkpoint_has_native_routing = (
        (mode in NATIVE_ROUTING_THERMAL_MODES)
        if engine_identity is not None else
        payload.get("checkpoint_schema") in (
            NATIVE_ROUTING_THERMAL_CHECKPOINT_SCHEMA,
            NATIVE_ROUTING_R2_THERMAL_CHECKPOINT_SCHEMA,
            NATIVE_CURL_ROUTING_THERMAL_CHECKPOINT_SCHEMA,
            NATIVE_CURL_ROUTING_R2_THERMAL_CHECKPOINT_SCHEMA,
        )
    )
    if native_routing != checkpoint_has_native_routing:
        raise ValueError("Thermal route checkpoint schema does not match its configured routed mode.")
    anchors = 0 if mode in ("P",) + NATIVE_ROUTING_THERMAL_MODES else 16
    expected_config = {
        "family": expected_family,
        "mode": mode,
        "source_width": 8,
        "context_width": 14,
        "environment_width": 8,
        "spatial_dim": 2,
        "hidden": 128,
        "message": 128,
        "collective_width": 64,
        "regional_anchors": anchors,
        "depth": 2,
        "max_sources": 12,
        "field_outputs": 3 if native_curl else 4,
        "affine_outputs": 1,
        "query_width": 0,
        "receiver_tile": 512,
        "nx": 128,
        "ny": 64,
        "environment_nx": 24,
        "environment_ny": 8,
        "forcing_scale": 1.0,
        "zero_offset": True,
        "seed": 0,
    }
    if native_curl:
        expected_config["flow_readout_law"] = NATIVE_CURL_READOUT_LAW
        expected_config["initialization_reference_field_outputs"] = 4
    if mode in NATIVE_ROUTING_THERMAL_MODES:
        expected_config["native_routing"] = _thermal_native_routing_config(mode, expected_config["seed"])
    if mode in ("P-G", "P-H"):
        expected_config["locality_prior_strength"] = 1.0
    factorized_affine_execution = config.get("factorized_affine_execution", False)
    if type(factorized_affine_execution) is not bool:
        raise TypeError("Thermal factorized execution mode must be a bool.")
    if factorized_affine_execution:
        expected_config["factorized_affine_execution"] = True
    if native_routing:
        route_config = config.get("native_routing")
        if not isinstance(route_config, Mapping):
            raise TypeError("Thermal native-routing config must be a mapping.")
        if type(route_config.get("initialization_seed")) is not int:
            raise TypeError("Thermal native-routing seed must be an integer.")
        if mode == "P-R" and type(route_config.get("score_bound")) is not float:
            raise TypeError("Thermal P-R score bound must be stored as a float.")
        if mode == "P-R2":
            if type(route_config.get("score_hidden")) is not int:
                raise TypeError("Thermal P-R2 score hidden width must be an integer.")
            if type(route_config.get("score_bound")) is not float:
                raise TypeError("Thermal P-R2 score bound must be stored as a float.")
            if type(route_config.get("views")) is not int:
                raise TypeError("Thermal P-R2 view count must be an integer.")
            if (type(route_config.get("feature_blocks")) is not list
                    or any(type(value) is not int for value in route_config["feature_blocks"])):
                raise TypeError("Thermal P-R2 feature blocks must be a list of integers.")
    if config != expected_config:
        raise ValueError("Thermal P-family model config differs from the sealed 8/14/8, H128, E192 native contract.")
    expected_adapter = {
        "nx": 128,
        "ny": 64,
        "environment_nx": 24,
        "environment_ny": 8,
        "forcing_scale": 1.0,
        "h_effective_eps": 1.0e-3,
        "h_effective_max": 1.0e4,
        "joint_output_laws": {"flow": "nonlinear", "temperature_response": "affine"},
    }
    if native_curl:
        expected_adapter["flow_readout_law"] = NATIVE_CURL_READOUT_LAW
        expected_adapter["native_curl_contract"] = native_curl_readout_contract()
    if adapter != expected_adapter:
        raise ValueError("Thermal P-family adapter frame/grid/output/forcing contract differs from the native contract.")
    if engine_identity is not None:
        recipe = engine_identity["recipe"]
        if recipe.get("seed") != expected_config["seed"]:
            raise ValueError("Thermal engine checkpoint seed differs from the sealed fresh initialization seed.")
        if recipe.get("regional_anchors") != anchors:
            raise ValueError("Thermal engine checkpoint anchor count differs from its P-family mode.")
    constructor_config = {
        name: value for name, value in config.items()
        if name not in {"family", "source_width", "context_width", "environment_width", "spatial_dim",
                        "field_outputs", "affine_outputs", "query_width", "zero_offset",
                        "initialization_reference_field_outputs", "native_routing"}
    }
    model = InteractionPreservingThermalAdapter(normalization_stats=stats, **constructor_config)
    if model.model_config() != config:
        raise ValueError("Thermal P-family model config does not round-trip canonically.")
    if any(not isinstance(name, str) or not name.startswith("core.") for name in state):
        raise ValueError("Thermal P-family checkpoint tensors must belong to its one active core.")
    if fresh_engine_checkpoint:
        from honf_runtime.unified_training import _validate_fresh_matched_initialization

        assert engine_identity is not None
        fresh_recipe = engine_identity["recipe"]
        engine_config = engine_identity["engine_config"]
        marker = engine_identity["fresh_initialization"]
        _validate_fresh_matched_initialization(
            marker, model, provider_identity,
            arm=child_mode,
            seed=fresh_recipe["seed"],
            sampling_version=engine_config["sampling_version"],
            deterministic_case_order=engine_config["deterministic_case_order"],
        )
    model.load_state_dict(state, strict=True)
    if fresh_engine_checkpoint:
        setter = getattr(model.core, "set_consolidated_native_validation", None)
        if not callable(setter):
            raise TypeError("Fresh Thermal P-family adapter does not expose consolidated validation controls.")
        setter(True)
        if getattr(model.core, "consolidated_native_validation", None) is not True:
            raise RuntimeError("Fresh Thermal consolidated native-validation policy did not reactivate after load.")
    if any(not name.startswith("core.") for name, _ in model.named_parameters()):
        raise RuntimeError("Thermal P-family adapter registered parameters outside the new joint core.")
    return model.to(device)

"""Opt-in Wind adapter for source-conditioned pair-core recovery arms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from honf_forward_core.interface_fields.interaction_preserving_joint import InteractionPreservingJointCore
from torch import nn

from .geometry import POSITIONAL_SCALE_D
from .joint_regional import (
    PreparedWindJointRegional,
    WindFarmJointRegionalModel,
    wind_receiver_features,
)
from .normalization import VelocityNormalizer, VerticalProfileBaseline

INTERACTION_PRESERVING_WIND_MODES = ("P", "P-G", "P-H")
NATIVE_ROUTING_WIND_MODES = ("P-R", "P-C")
ALL_INTERACTION_PRESERVING_WIND_MODES = (
    INTERACTION_PRESERVING_WIND_MODES + NATIVE_ROUTING_WIND_MODES
)
NATIVE_ROUTING_WIND_FAMILY = "wind_native_interaction_routing_v1"
NATIVE_ROUTING_CONFIG_SCHEMA = "native_interaction_routing_v1"


class InteractionPreservingWindModel(WindFarmJointRegionalModel):
    """Keep the deployed source-resolved nonlinear Wind readout as the P path."""

    FAMILY = "wind_interaction_preserving_joint_v1"
    NATIVE_ROUTING_FAMILY = NATIVE_ROUTING_WIND_FAMILY

    def __init__(
        self,
        *,
        velocity_transform: VelocityNormalizer,
        background_profile: VerticalProfileBaseline,
        mode: str = "P",
        hidden: int = 128,
        message: int = 128,
        regional_anchors: int = 0,
        depth: int = 2,
        collective_width: int = 64,
        receiver_tile: int = 512,
        seed: int = 42,
        max_sources: int = 30,
        locality_prior_strength: float | None = None,
        factorized_affine_execution: bool = False,
        native_control_hidden: int | None = None,
    ) -> None:
        if mode not in ALL_INTERACTION_PRESERVING_WIND_MODES:
            raise ValueError(f"Interaction-preserving Wind mode must be one of {ALL_INTERACTION_PRESERVING_WIND_MODES}.")
        if background_profile is None:
            raise ValueError("Wind P-family prediction requires its selected TRAIN-fitted height profile.")
        if receiver_tile < 1 or depth != 2 or max_sources < 1:
            raise ValueError("Wind P-family uses positive receiver/source capacities and exactly two pair rounds.")
        direct_mode = mode in ("P",) + NATIVE_ROUTING_WIND_MODES
        if direct_mode:
            if regional_anchors != 0 or locality_prior_strength not in (None, 0, 0.0):
                raise ValueError("Wind P/P-R/P-C have no collective anchors or locality prior.")
        elif regional_anchors < 1 or locality_prior_strength is None:
            raise ValueError("Wind P-G/P-H require explicit collective anchors and locality strength.")
        if mode == "P-C":
            if native_control_hidden is None:
                native_control_hidden = 34
            if native_control_hidden != 34:
                raise ValueError("Wind P-C uses the parameter-matched native control width 34.")
        elif native_control_hidden is not None:
            raise ValueError("native_control_hidden is defined only for P-C.")
        locality = 0.0 if locality_prior_strength is None else float(locality_prior_strength)
        if not np.isfinite(locality) or locality < 0:
            raise ValueError("Wind locality prior must be finite and nonnegative.")
        if direct_mode and locality != 0:
            raise ValueError("Wind P/P-R/P-C have no collective locality prior.")
        if mode in ("P-G", "P-H") and locality != 1.0:
            raise ValueError("The recovery Wind P-G/P-H locality prior is sealed to 1.0.")
        if type(factorized_affine_execution) is not bool:
            raise TypeError("factorized_affine_execution must be a bool.")

        nn.Module.__init__(self)
        self.velocity_transform = velocity_transform
        self.background_profile = background_profile
        self.mode = str(mode)
        self.family_id = (self.NATIVE_ROUTING_FAMILY if mode in NATIVE_ROUTING_WIND_MODES
                          else self.FAMILY)
        self.seed = int(seed)
        self.receiver_tile = int(receiver_tile)
        self.max_sources = int(max_sources)
        self.locality_prior_strength = locality
        self.core_config = {
            "source_width": 2,
            "context_width": 11,
            "environment_width": 7,
            "spatial_dim": 3,
            "hidden": int(hidden),
            "message": int(message),
            "mode": self.mode,
            "collective_width": int(collective_width),
            "regional_anchors": int(regional_anchors),
            "depth": int(depth),
            "field_outputs": 3,
            "affine_outputs": 0,
            "query_width": 7,
            "max_sources": self.max_sources,
            "initialization_seed": self.seed,
        }
        if mode in ("P-G", "P-H"):
            self.core_config["locality_prior_strength"] = locality
        if factorized_affine_execution:
            self.core_config["factorized_affine_execution"] = True
        self.core = InteractionPreservingJointCore(
            2,
            11,
            7,
            spatial_dim=3,
            hidden=int(hidden),
            message=int(message),
            mode=self.mode,
            collective_width=int(collective_width),
            regional_anchors=int(regional_anchors),
            field_outputs=3,
            affine_outputs=0,
            query_width=7,
            max_sources=self.max_sources,
            initialization_seed=self.seed,
            locality_prior_strength=(locality if mode != "P" else None),
            factorized_affine_execution=factorized_affine_execution,
            native_control_hidden=native_control_hidden,
        )
        if mode in NATIVE_ROUTING_WIND_MODES:
            self.core_config["native_routing"] = dict(self.core.config["native_routing"])

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> InteractionPreservingWindModel:
        payload = dict(config)
        if payload.get("family") not in (cls.FAMILY, cls.NATIVE_ROUTING_FAMILY) or payload.get("dataset") != "WindFarm":
            raise ValueError("Saved config is not an interaction-preserving Wind model.")
        expected_top = {
            "family", "dataset", "seed", "receiver_tile", "core", "velocity_transform",
            "background_profile", "native_coordinate_frame", "native_position_scale_D",
            "receiver_features", "environment_token_shape", "environment_measure", "source_ids",
            "output_units", "output_law", "background_interpretation", "exact_affine_increments",
        }
        if set(payload) != expected_top:
            raise ValueError("Saved Wind P-family config has missing or unsupported top-level fields.")
        core = payload["core"]
        if not isinstance(core, Mapping):
            raise TypeError("Saved Wind P-family core config must be a mapping.")
        core = dict(core)
        required_core = {
            "source_width", "context_width", "environment_width", "spatial_dim", "hidden", "message",
            "mode", "collective_width", "regional_anchors", "depth", "field_outputs", "affine_outputs",
            "query_width", "max_sources", "initialization_seed",
        }
        legacy_core_options = ({"locality_prior_strength"}, {"factorized_affine_execution"},
                               {"locality_prior_strength", "factorized_affine_execution"})
        allowed_core = {frozenset(required_core)} | {
            frozenset(required_core | option) for option in legacy_core_options
        }
        native_core_options = ({"native_routing"}, {"native_routing", "factorized_affine_execution"})
        allowed_core |= {frozenset(required_core | option) for option in native_core_options}
        if frozenset(core) not in allowed_core:
            raise ValueError("Saved Wind P-family core dimensions are incomplete or unsupported.")
        if ("factorized_affine_execution" in core and
                type(core["factorized_affine_execution"]) is not bool):
            raise TypeError("Saved Wind factorized execution mode must be a bool.")
        if type(payload["seed"]) is not int or type(core["initialization_seed"]) is not int:
            raise TypeError("Saved Wind P-family initialization seeds must be integers.")
        if core["initialization_seed"] != payload["seed"]:
            raise ValueError("Wind model seed and core initialization seed differ.")
        if type(payload["receiver_tile"]) is not int or payload["receiver_tile"] < 1:
            raise ValueError("Saved Wind P-family receiver tile must be a positive integer.")
        exact_core = {
            "source_width": 2, "context_width": 11, "environment_width": 7, "spatial_dim": 3,
            "field_outputs": 3, "affine_outputs": 0, "query_width": 7, "depth": 2,
            "collective_width": 64, "max_sources": 30,
        }
        if any(core.get(name) != value or type(core.get(name)) is not int for name, value in exact_core.items()):
            raise ValueError("Saved Wind P-family dimensions differ from its sealed source/receiver contract.")
        if type(core.get("hidden")) is not int or core["hidden"] != 128:
            raise ValueError("Saved Wind P-family hidden width differs from the recovery contract (128).")
        if type(core.get("message")) is not int or core["message"] != 128:
            raise ValueError("Saved Wind P-family message width differs from the recovery contract (128).")
        mode = core.get("mode")
        if mode not in ALL_INTERACTION_PRESERVING_WIND_MODES:
            raise ValueError("Saved Wind P-family mode is unsupported.")
        native_routing_mode = mode in NATIVE_ROUTING_WIND_MODES
        expected_family = cls.NATIVE_ROUTING_FAMILY if native_routing_mode else cls.FAMILY
        if payload.get("family") != expected_family:
            raise ValueError("Saved Wind family ID does not match its direct or native-routing mode.")
        if native_routing_mode:
            route = core.get("native_routing")
            if not isinstance(route, Mapping):
                raise TypeError("Saved Wind native routing config must be a mapping.")
            route = dict(route)
            if mode == "P-R":
                expected_route = {
                    "schema": NATIVE_ROUTING_CONFIG_SCHEMA,
                    "mode": "P-R",
                    "policy": "learned_density",
                    "initialization_seed": payload["seed"],
                    "score_hidden": 32,
                    "score_bound": 2.0,
                }
            else:
                expected_route = {
                    "schema": NATIVE_ROUTING_CONFIG_SCHEMA,
                    "mode": "P-C",
                    "policy": "destination_gain",
                    "initialization_seed": payload["seed"],
                    "control_hidden": 34,
                }
            if route != expected_route:
                raise ValueError("Saved Wind native-routing policy, schema, seed or sealed width is not canonical.")
            if type(route.get("initialization_seed")) is not int:
                raise TypeError("Saved Wind native-routing seed must be an integer.")
            if mode == "P-R" and type(route.get("score_bound")) is not float:
                raise TypeError("Saved Wind P-R score bound must be stored as a float.")
        elif "native_routing" in core:
            raise ValueError("Legacy Wind P/P-G/P-H configs cannot declare a native-routing module.")
        anchors = core.get("regional_anchors")
        direct_mode = mode in ("P",) + NATIVE_ROUTING_WIND_MODES
        if type(anchors) is not int or (direct_mode and anchors != 0) or (not direct_mode and anchors < 1):
            raise ValueError("Saved Wind collective anchor inventory does not match its mode.")
        collective_mode = mode in ("P-G", "P-H")
        if collective_mode != ("locality_prior_strength" in core):
            raise ValueError("Saved Wind locality setting does not match its P/P-G/P-H mode.")
        if mode in ("P-G", "P-H") and core["locality_prior_strength"] != 1.0:
            raise ValueError("Saved Wind P-G/P-H locality prior differs from the sealed value 1.0.")
        expected_features = {
            "width": 7,
            "quantities": [
                "receiver_minus_domain_origin divided by [50,38,6.25] D",
                "domain_upper_minus_receiver divided by [50,38,6.25] D",
                "absolute_height divided by 6.25 D",
            ],
        }
        fixed_contract = {
            "native_coordinate_frame": "rotor_diameters",
            "native_position_scale_D": [float(value) for value in POSITIONAL_SCALE_D],
            "receiver_features": expected_features,
            "environment_token_shape": [2, 2, 2],
            "environment_measure": "geometry-only native domain support volume in rotor_diameters^3 per token",
            "source_ids": "original turbine slot IDs; explicit and unique per active scene",
            "output_units": "m/s",
            "output_law": "TRAIN height profile m/s + standardized source-conditioned nonlinear residual * u_ref_mps * safe_std",
            "background_interpretation": "TRAIN-fitted empirical height profile, including mean wake structure",
            "exact_affine_increments": False,
        }
        if any(payload.get(name) != expected for name, expected in fixed_contract.items()):
            raise ValueError("Saved Wind P-family frame, feature inventory, measure or output law differs from the native contract.")
        normalizer_payload = payload["velocity_transform"]
        profile_payload = payload["background_profile"]
        if not isinstance(normalizer_payload, Mapping) or not isinstance(profile_payload, Mapping):
            raise TypeError("Saved Wind transforms must be mappings.")
        normalizer = VelocityNormalizer.from_dict(dict(normalizer_payload))
        profile = VerticalProfileBaseline.from_dict(dict(profile_payload))
        if normalizer.to_dict() != dict(normalizer_payload) or profile.to_dict() != dict(profile_payload):
            raise ValueError("Saved Wind normalization/profile fields do not round-trip under the maintained TRAIN schema.")
        model = cls(
            velocity_transform=normalizer,
            background_profile=profile,
            mode=str(mode),
            hidden=core["hidden"],
            message=core["message"],
            regional_anchors=anchors,
            depth=core["depth"],
            collective_width=core["collective_width"],
            receiver_tile=payload["receiver_tile"],
            seed=payload["seed"],
            max_sources=core["max_sources"],
            locality_prior_strength=(float(core["locality_prior_strength"])
                                     if collective_mode else None),
            factorized_affine_execution=core.get("factorized_affine_execution", False),
            native_control_hidden=(34 if mode == "P-C" else None),
        )
        if model.export_config() != payload:
            raise ValueError("Saved Wind P-family model config is not a canonical exact round trip.")
        return model

    def export_config(self) -> dict[str, Any]:
        core_config = {key: (dict(value) if isinstance(value, Mapping) else value)
                       for key, value in self.core_config.items()}
        if self.core.factorized_affine_execution:
            core_config["factorized_affine_execution"] = True
        else:
            core_config.pop("factorized_affine_execution", None)
        return {
            "family": self.family_id,
            "dataset": "WindFarm",
            "seed": self.seed,
            "receiver_tile": self.receiver_tile,
            "core": core_config,
            "velocity_transform": self.velocity_transform.to_dict(),
            "background_profile": self.background_profile.to_dict(),
            "native_coordinate_frame": "rotor_diameters",
            "native_position_scale_D": list(POSITIONAL_SCALE_D),
            "receiver_features": {
                "width": 7,
                "quantities": [
                    "receiver_minus_domain_origin divided by [50,38,6.25] D",
                    "domain_upper_minus_receiver divided by [50,38,6.25] D",
                    "absolute_height divided by 6.25 D",
                ],
            },
            "environment_token_shape": [2, 2, 2],
            "environment_measure": "geometry-only native domain support volume in rotor_diameters^3 per token",
            "source_ids": "original turbine slot IDs; explicit and unique per active scene",
            "output_units": "m/s",
            "output_law": "TRAIN height profile m/s + standardized source-conditioned nonlinear residual * u_ref_mps * safe_std",
            "background_interpretation": "TRAIN-fitted empirical height profile, including mean wake structure",
            "exact_affine_increments": False,
        }

    def predict_standardized(
        self,
        prepared: PreparedWindJointRegional,
        receivers_D: torch.Tensor,
        *,
        receiver_features: torch.Tensor | None = None,
        chunk_size: int | None = None,
        retained_access_mass: float | None = None,
        receiver_edge_executor: str = "dense",
    ) -> torch.Tensor:
        self._assert_prepared(prepared)
        if retained_access_mass is not None or receiver_edge_executor != "dense":
            raise ValueError("P-family Wind uses its declared full source-conditioned receiver read.")
        features = (
            wind_receiver_features(prepared.scene, receivers_D)
            if receiver_features is None
            else receiver_features
        )
        if tuple(features.shape) != (*receivers_D.shape[:2], 7):
            raise ValueError("Wind receiver features must align with [B,Q] and have width seven.")
        result = self.core.predict_fields(
            prepared.core_context,
            receivers_D,
            receiver_features=features,
            chunk_size=self.receiver_tile if chunk_size is None else int(chunk_size),
        )
        if tuple(result.shape) != (*receivers_D.shape[:2], 3):
            raise RuntimeError("P-family Wind core returned an invalid three-component physical residual.")
        return result


__all__ = [
    "ALL_INTERACTION_PRESERVING_WIND_MODES",
    "INTERACTION_PRESERVING_WIND_MODES",
    "NATIVE_ROUTING_WIND_FAMILY",
    "NATIVE_ROUTING_WIND_MODES",
    "InteractionPreservingWindModel",
]

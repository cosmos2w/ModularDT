"""Versioned Round6 receiver-readout extension for scene-prepared H cores."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor

from .hyperedge_readout import (
    GeometryMode,
    HyperedgeReceiverReadout,
    PreparedHyperedgeReadoutValues,
    ReceiverAccess,
    ReceiverReduction,
)
from .model import PreparedInheritedOrganization
from .moment_model import MomentHyperedgeCoreV1


class HyperedgeReadoutCoreV1(MomentHyperedgeCoreV1):
    """H core with an opt-in, matched local or full-bank receiver correction.

    The serialized model binds the receiver coordinate convention and access
    boundary. Round6 recipes must set all three explicitly and keep
    ``moment_variant='H'``.
    """

    schema = "hyperedge_readout_core_v1"

    def __init__(
        self,
        *,
        receiver_reduction: ReceiverReduction,
        receiver_access: ReceiverAccess,
        geometry_mode: GeometryMode,
        receiver_readout_init_seed: int = 0,
        **kwargs,
    ) -> None:
        moment_variant = kwargs.pop("moment_variant", "H")
        if moment_variant != "H":
            raise ValueError("Round6 receiver readout requires the completed H scene state.")
        if type(receiver_readout_init_seed) is not int:
            raise TypeError("receiver_readout_init_seed must be an integer.")
        super().__init__(moment_variant="H", **kwargs)
        self.receiver_reduction = receiver_reduction
        self.receiver_access = receiver_access
        self.geometry_mode = geometry_mode
        self.receiver_readout_init_seed = receiver_readout_init_seed
        self.receiver_readout = HyperedgeReceiverReadout(
            coordinate_dim=self.coordinate_dim,
            reduction=receiver_reduction,
            access=receiver_access,
            geometry_mode=geometry_mode,
            anchor_state_dim=self.context_dim,
            query_state_dim=self.context_dim,
            context_dim=self.context_dim,
            init_seed=receiver_readout_init_seed,
        )

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            schema=self.schema,
            receiver_reduction=self.receiver_reduction,
            receiver_access=self.receiver_access,
            geometry_mode=self.geometry_mode,
            receiver_readout_init_seed=self.receiver_readout_init_seed,
        )
        return config

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> HyperedgeReadoutCoreV1:
        values = dict(config)
        if values.pop("schema", None) != cls.schema:
            raise ValueError(f"Expected {cls.schema} configuration.")
        return cls(**values)

    def _attach_receiver_projection(self, organization: PreparedInheritedOrganization) -> None:
        prepared = getattr(organization, "moment_hyperedges", None)
        if self.moment_variant != "H" or prepared is None or prepared.processed_state is None:
            raise ValueError("Round6 readout projections require the completed H processed_state.")
        scene = organization.scene
        organization.receiver_readout_values = self.receiver_readout.prepare_scene(
            prepared.processed_state,
            scene.environment_ids,
            frame_key=scene.coordinate_frame_key,
            scene_version=organization.scene_versions,
            owner_token=organization.owner_token,
        )
        values = organization.receiver_readout_values.values
        if values.requires_grad:
            values.register_hook(organization.graph_lifetime.mark_consumed)
        batch, anchors = values.shape[:2]
        organization.work_counts.update(
            receiver_readout_preparation_projection_rows=batch * anchors,
            receiver_readout_preparation_multiply_adds=batch * anchors * self.context_dim * 64,
            receiver_readout_prepared_projection_bytes=values.numel() * values.element_size(),
        )

    def prepare(self, scene, *, return_trace: bool = False) -> PreparedInheritedOrganization:
        organization = super().prepare(scene, return_trace=return_trace)
        self._attach_receiver_projection(organization)
        organization.factor_versions = self._factors(organization)
        return organization

    def validate_prepared_state(self, organization: PreparedInheritedOrganization, query=None) -> None:
        super().validate_prepared_state(organization, query)
        payload = organization.receiver_readout_values
        prepared = getattr(organization, "moment_hyperedges", None)
        if not isinstance(payload, PreparedHyperedgeReadoutValues) or prepared is None:
            raise ValueError("Round6 H organizations require their owner-bound A_Z Z projection.")
        payload.validate_binding(
            frame_key=organization.scene.coordinate_frame_key,
            scene_version=organization.scene_versions,
            owner_token=organization.owner_token,
        )
        if payload.source_state is not prepared.processed_state:
            raise ValueError("Prepared A_Z Z is not derived from the completed H processed state.")
        if payload.anchor_ids is not organization.scene.environment_ids:
            raise ValueError("Prepared A_Z Z changed environment-anchor ID ownership.")

    def _apply_receiver_context(
        self,
        organization: PreparedInheritedOrganization,
        query,
        query_states: Tensor,
        context: Tensor,
        *,
        return_trace: bool,
    ):
        payload = organization.receiver_readout_values
        if not isinstance(payload, PreparedHyperedgeReadoutValues):
            raise TypeError("Round6 receiver readout was not prepared with the H scene.")
        exact_counts = self.collect_exact_work_counts
        if self.receiver_access == "local_cover":
            correction, trace, work = self.receiver_readout.local_correction(
                prepared=payload,
                query_states=query_states,
                query_coordinates=query.coordinates,
                anchor_coordinates=organization.scene.environment_coordinates,
                anchor_mask=organization.scene.environment_mask,
                frame_scale=organization.frame_scale,
                cover=query.receiver_cover,
                query_mask=query.query_mask,
                collect_exact_work_counts=exact_counts,
                return_trace=return_trace,
                expected_frame_key=organization.scene.coordinate_frame_key,
                expected_scene_version=organization.scene_versions,
                expected_owner_token=organization.owner_token,
            )
        else:
            correction, trace, work = self.receiver_readout.full_bank_correction(
                prepared=payload,
                query_states=query_states,
                query_coordinates=query.coordinates,
                anchor_coordinates=organization.scene.environment_coordinates,
                anchor_mask=organization.scene.environment_mask,
                anchor_measure=organization.scene.environment_measure,
                frame_scale=organization.frame_scale,
                query_mask=query.query_mask,
                collect_exact_work_counts=exact_counts,
                return_trace=return_trace,
                expected_frame_key=organization.scene.coordinate_frame_key,
                expected_scene_version=organization.scene_versions,
                expected_owner_token=organization.owner_token,
            )
        return context + correction, trace, work

    def intervene_associations(self, organization, module_probabilities, environment_probabilities, *, label: str):
        result = super().intervene_associations(
            organization, module_probabilities, environment_probabilities, label=label
        )
        self._attach_receiver_projection(result)
        result.factor_versions = self._factors(result)
        return result

    def zero_new_outputs_for_diagnostic(self, organization: PreparedInheritedOrganization):
        result = super().zero_new_outputs_for_diagnostic(organization)
        self._attach_receiver_projection(result)
        result.factor_versions = self._factors(result)
        return result


__all__ = ["HyperedgeReadoutCoreV1"]

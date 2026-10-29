"""IAG-C extension with scene-prepared typed moment hyperedges."""

from __future__ import annotations

import hashlib
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor

from .contracts import PreparedProjectedAnchorValues
from .hyperedge_processor import HyperedgeProcessorOutput, MomentHyperedgeProcessor
from .model import (
    InheritedAttentionCoreV1,
    PreparedInheritedAffineQuery,
    PreparedInheritedOrganization,
    _GraphLifetime,
    _versions,
)
from .moments import (
    HeadwiseMomentLift,
    PreparedMomentHyperedges,
    PreparedMomentHyperedgeView,
    _tensor_tree,
    build_moment_descriptor,
    compute_typed_moments,
    export_prepared_moment_hyperedges,
)
from .readers import gather_cover_indexed


class MomentHyperedgeCoreV1(InheritedAttentionCoreV1):
    """Scene-prepared moment hyperedge model with a stable C0/M/H schema.

    C0 is the existing IAG-C indexed/projected path and adds no parameters.
    M and H retain the identical typed moments and trainable parameter
    inventory; only identity versus SiLU between the processor's two affine
    layers changes.  Thermal source transport is an exact linear extension of
    the inherited heat law and is built from raw factors at the requested
    arithmetic dtype.
    """

    schema = "scene_prepared_moment_hyperedge_core_v1"
    moment_schema = "scene_prepared_moment_hyperedges_v1"

    def __init__(
        self,
        *,
        module_input_dim: int,
        environment_input_dim: int,
        coordinate_dim: int,
        query_input_dim: int = 0,
        operating_input_dim: int = 0,
        output_dim: int = 3,
        architecture: str = "IAG-C",
        context_dim: int = 128,
        num_heads: int = 4,
        context_depth: int = 2,
        local_hidden_dim: int = 128,
        dropout: float = 0.0,
        init_seed: int = 0,
        signed_carrier_width: int = 8,
        thermal_response: bool = False,
        normalization: str = "soft",
        scene_executor: str = "dense",
        query_executor: str = "indexed_projected",
        collect_exact_work_counts: bool | None = None,
        moment_variant: str = "C0",
        moment_width: int = 8,
        processor_hidden_dim: int = 128,
        moment_init_seed: int = 0,
    ) -> None:
        if architecture != "IAG-C":
            raise ValueError("Scene-prepared moment hyperedges support IAG-C only.")
        if (context_dim, num_heads, signed_carrier_width) != (128, 4, 8):
            raise ValueError("Round5 fixes context width 128, four heads and eight signed carriers.")
        if query_executor != "indexed_projected":
            raise ValueError("Round5 moment hyperedges require the qualified indexed_projected query executor.")
        if moment_variant not in ("C0", "M", "H"):
            raise ValueError("moment_variant must be C0, M or H.")
        if moment_width != 8 or processor_hidden_dim != 128:
            raise ValueError("Round5 fixes eight moment coordinates and hidden width 128.")
        if type(moment_init_seed) is not int or type(moment_width) is not int or type(processor_hidden_dim) is not int:
            raise ValueError("Moment dimensions and initialization seed must be integers.")

        super().__init__(
            module_input_dim=module_input_dim,
            environment_input_dim=environment_input_dim,
            coordinate_dim=coordinate_dim,
            query_input_dim=query_input_dim,
            operating_input_dim=operating_input_dim,
            output_dim=output_dim,
            architecture=architecture,
            context_dim=context_dim,
            num_heads=num_heads,
            context_depth=context_depth,
            local_hidden_dim=local_hidden_dim,
            dropout=dropout,
            init_seed=init_seed,
            signed_carrier_width=signed_carrier_width,
            thermal_response=thermal_response,
            normalization=normalization,
            scene_executor=scene_executor,
            query_executor=query_executor,
            collect_exact_work_counts=collect_exact_work_counts,
        )
        self.moment_variant = moment_variant
        self.moment_width = int(moment_width)
        self.processor_hidden_dim = int(processor_hidden_dim)
        self.moment_init_seed = int(moment_init_seed)

        self.module_moment_lift: HeadwiseMomentLift | None = None
        self.environment_moment_lift: HeadwiseMomentLift | None = None
        self.hyperedge_processor: MomentHyperedgeProcessor | None = None
        if moment_variant in ("M", "H"):
            self.module_moment_lift = HeadwiseMomentLift(
                self.num_heads, self.moment_width, self.context_dim // self.num_heads,
                init_seed=self._named_seed("module_moment_lift.weight"),
            )
            self.environment_moment_lift = HeadwiseMomentLift(
                self.num_heads, self.moment_width, self.context_dim // self.num_heads,
                init_seed=self._named_seed("environment_moment_lift.weight"),
            )
            self.hyperedge_processor = MomentHyperedgeProcessor(
                708 if self.coordinate_dim == 2 else 814,
                self.coordinate_dim,
                processor=moment_variant,
                num_heads=self.num_heads,
                moment_width=self.moment_width,
                context_dim=self.context_dim,
                hidden_dim=self.processor_hidden_dim,
                thermal_transport=self.thermal_response,
                init_seed=self._named_seed("hyperedge_processor.input_projection.weight"),
            )

    def _named_seed(self, name: str) -> int:
        digest = hashlib.blake2b(f"{self.moment_init_seed}:{name}".encode(), digest_size=8).digest()
        return int.from_bytes(digest, "little") % (2**31 - 1)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            schema=self.schema,
            moment_variant=self.moment_variant,
            moment_width=self.moment_width,
            processor_hidden_dim=self.processor_hidden_dim,
            moment_init_seed=self.moment_init_seed,
        )
        return config

    @staticmethod
    def _factors(organization: PreparedInheritedOrganization) -> tuple:
        base = InheritedAttentionCoreV1._factors(organization)
        return (*base, *_versions(getattr(organization, "moment_hyperedges", None)))

    def _prepare_scene_readout(
        self, organization: PreparedInheritedOrganization
    ) -> PreparedProjectedAnchorValues | None:
        prepared = getattr(organization, "moment_hyperedges", None)
        if prepared is None or self.moment_variant == "C0":
            return super()._prepare_scene_readout(organization)
        return PreparedProjectedAnchorValues(
            values=prepared.processed_state,
            anchor_ids=organization.scene.environment_ids,
            frame_key=organization.scene.coordinate_frame_key,
            scene_version=organization.scene_versions,
            owner_token=organization.owner_token,
        )

    def prepare(self, scene, *, return_trace: bool = False) -> PreparedInheritedOrganization:
        organization = super().prepare(scene, return_trace=return_trace)
        organization.moment_hyperedges = None
        if self.moment_variant != "C0":
            self._attach_moment_hyperedges(organization)
        organization.factor_versions = self._factors(organization)
        return organization

    def _attach_moment_hyperedges(
        self,
        organization: PreparedInheritedOrganization,
        *,
        module_probabilities: Tensor | None = None,
        environment_probabilities: Tensor | None = None,
    ) -> PreparedMomentHyperedges:
        if self.hyperedge_processor is None or self.module_moment_lift is None or self.environment_moment_lift is None:
            raise RuntimeError("M/H moment modules were not constructed.")
        scene = organization.scene
        if organization.prepared_readout_values is None:
            raise RuntimeError("Moment hyperedges require the indexed_projected C0 anchor readout.")

        module_p = (organization.module_atlas.probabilities if module_probabilities is None
                    else module_probabilities)
        environment_p = (organization.environment_atlas.probabilities if environment_probabilities is None
                         else environment_probabilities)
        frame_scale = organization.frame_scale
        xi_module = (scene.module_coordinates.to(organization.module_states.dtype)
                     - organization.frame_origin[:, None]) / frame_scale[:, None]
        xi_environment = (scene.environment_coordinates.to(organization.environment_states.dtype)
                          - organization.frame_origin[:, None]) / frame_scale[:, None]
        module_lengths = scene.module_characteristic_lengths.to(organization.module_states.dtype) / frame_scale[:, None]
        environment_lengths = (
            scene.environment_characteristic_lengths.to(organization.environment_states.dtype) / frame_scale[:, None]
        )

        module_stats = compute_typed_moments(
            kind="module",
            values=organization.module_values.values,
            full_value_mean=organization.module_anchor_values.values,
            probabilities=module_p,
            lift_weight=self.module_moment_lift.weight,
            normalized_donor_coordinates=xi_module,
            normalized_anchor_coordinates=xi_environment,
            normalized_donor_characteristic_lengths=module_lengths,
            donor_valid=organization.module_atlas.donor_valid,
            anchor_valid=organization.module_atlas.destination_valid,
            donor_ids=organization.module_atlas.donor_ids,
            anchor_ids=organization.module_atlas.anchor_ids,
        )
        environment_stats = compute_typed_moments(
            kind="environment",
            values=organization.environment_values.values,
            full_value_mean=organization.environment_anchor_values.values,
            probabilities=environment_p,
            lift_weight=self.environment_moment_lift.weight,
            normalized_donor_coordinates=xi_environment,
            normalized_anchor_coordinates=xi_environment,
            normalized_donor_characteristic_lengths=environment_lengths,
            donor_valid=organization.environment_atlas.donor_valid,
            anchor_valid=organization.environment_atlas.destination_valid,
            donor_ids=organization.environment_atlas.donor_ids,
            anchor_ids=organization.environment_atlas.anchor_ids,
        )
        for label, stats in (("module", module_stats), ("environment", environment_stats)):
            bad = stats.audit.latent_large_negative.any() | stats.audit.spatial_large_negative.any()
            if bad.device.type == "cuda":
                torch._assert_async(~bad, f"{label} moment variance exceeded its preregistered roundoff allowance.")
            elif bool(bad):
                raise ValueError(f"{label} moment variance exceeded its preregistered roundoff allowance.")
        descriptor = build_moment_descriptor(
            module_stats,
            environment_stats,
            normalized_anchor_coordinates=xi_environment,
            normalized_anchor_characteristic_lengths=environment_lengths,
            operating_state=organization.operating_state,
        )
        descriptor_finite = torch.isfinite(descriptor).all()
        if descriptor_finite.device.type == "cuda":
            torch._assert_async(descriptor_finite, "Moment hyperedge descriptor contains nonfinite values.")
        elif not bool(descriptor_finite):
            raise ValueError("Moment hyperedge descriptor contains nonfinite values.")
        base_state = organization.prepared_readout_values.values
        prepared = PreparedMomentHyperedges(
            module=module_stats,
            environment=environment_stats,
            descriptor=descriptor,
            base_projected_state=base_state,
            module_measure=organization.module_atlas.normalized_measure,
            source_carriers=organization.source_carriers,
            module_coordinates=xi_module,
            environment_coordinates=xi_environment,
            module_ids=scene.module_ids,
            environment_ids=scene.environment_ids,
            module_mask=scene.module_mask,
            environment_mask=scene.environment_mask,
            source_module_coordinates=scene.module_coordinates,
            source_environment_coordinates=scene.environment_coordinates,
            frame_key=scene.coordinate_frame_key,
            scene_version=organization.scene_versions,
            owner_token=organization.owner_token,
            normalizer_key=scene.normalizer_key,
            scene_key=scene.scene_key,
            graph_lifetime=organization.graph_lifetime,
        )
        outputs = self.process_moments(descriptor, anchor_mask=scene.environment_mask)
        processed = (base_state + outputs.context_residual) * scene.environment_mask[..., None].to(base_state.dtype)
        prepared = prepared.with_processor_outputs(
            preactivation=outputs.preactivation,
            hidden=outputs.hidden,
            context_residual=outputs.context_residual,
            processed_state=processed,
            heat_mix_constant=outputs.heat_mix_constant,
            heat_mix_spatial=outputs.heat_mix_spatial,
            graph_lifetime=organization.graph_lifetime,
        )
        organization.moment_hyperedges = prepared
        organization.prepared_readout_values = self._prepare_scene_readout(organization)
        self._record_moment_work_counts(organization, prepared)
        return prepared

    @staticmethod
    def _storage_map(tensors: tuple[Tensor, ...]) -> dict[tuple[str, int, int], int]:
        """Return unique backing storages and byte sizes, counting aliases once."""
        observed: dict[tuple[str, int, int], int] = {}
        for tensor in tensors:
            storage = tensor.untyped_storage()
            size = int(storage.nbytes())
            key = (str(tensor.device), int(storage.data_ptr()), size)
            observed[key] = size
        return observed

    def _record_moment_work_counts(
        self,
        organization: PreparedInheritedOrganization,
        prepared: PreparedMomentHyperedges,
    ) -> None:
        """Attach shape-derived M/H costs and cache storage to the preparation receipt."""
        module_p, environment_p = prepared.module.probabilities, prepared.environment.probabilities
        batch, heads, anchors, module_donors = module_p.shape
        environment_donors = environment_p.shape[-1]
        coordinate_dim = self.coordinate_dim
        descriptor_width = int(prepared.descriptor.shape[-1])
        hidden_width = self.processor_hidden_dim
        moment_width = self.moment_width
        # The donor-only lift includes shifted latent moments and the complete
        # small spatial second-moment matrix used by the stable row-mass formula.
        lift_channels = 32 + 10 * coordinate_dim + coordinate_dim * coordinate_dim
        donor_rows = batch * heads * (module_donors + environment_donors)
        head_lift_macs = donor_rows * 32 * moment_width
        association_lift_macs = batch * heads * anchors * (module_donors + environment_donors) * lift_channels
        descriptor_rows = batch * anchors
        input_projection_macs = descriptor_rows * descriptor_width * hidden_width
        context_projection_macs = descriptor_rows * hidden_width * self.context_dim
        heat_projection_macs = (
            descriptor_rows * hidden_width * heads * self.signed_carrier_width * (1 + coordinate_dim)
            if self.thermal_response else 0
        )
        processor_macs = input_projection_macs + context_projection_macs + heat_projection_macs

        exact_counters = self.collect_exact_work_counts
        if exact_counters:
            # Count unique allocations by backing storage across cache aliases.
            # The incremental figure excludes tensors retained by the base org.
            package_tensors = prepared.tensors()
            package_storages = self._storage_map(package_tensors)
            base_values = tuple(
                getattr(organization, item.name)
                for item in dataclass_fields(organization)
                if item.name != "moment_hyperedges"
            )
            base_storages = self._storage_map(_tensor_tree(base_values))
            incremental_storages = {
                key: size for key, size in package_storages.items() if key not in base_storages
            }

            module_audit = prepared.module.audit
            environment_audit = prepared.environment.audit
            audit_counts = torch.stack((
                module_audit.latent_clamped.sum() + environment_audit.latent_clamped.sum(),
                module_audit.spatial_clamped.sum() + environment_audit.spatial_clamped.sum(),
                module_audit.latent_large_negative.sum() + environment_audit.latent_large_negative.sum(),
                module_audit.spatial_large_negative.sum() + environment_audit.spatial_large_negative.sum(),
            )).detach().to(torch.int64).cpu().tolist()

        work = organization.work_counts
        work.update({
            "moment_typed_lifted_donor_rows": donor_rows,
            "moment_donor_lift_channels_per_row": lift_channels,
            "moment_typed_lifted_donor_scalar_entries": donor_rows * lift_channels,
            "moment_headwise_lift_multiply_adds": head_lift_macs,
            "moment_association_lift_contraction_multiply_adds": association_lift_macs,
            "moment_association_lift_contraction_flops": 2 * association_lift_macs,
            "moment_processed_anchor_rows": descriptor_rows,
            "moment_descriptor_width": descriptor_width,
            "moment_descriptor_scalar_entries": descriptor_rows * descriptor_width,
            "moment_processor_input_projection_multiply_adds": input_projection_macs,
            "moment_processor_context_projection_multiply_adds": context_projection_macs,
            "moment_processor_heat_projection_multiply_adds": heat_projection_macs,
            "moment_processor_linear_multiply_adds": processor_macs,
            "moment_processor_linear_flops": 2 * processor_macs,
            "moment_processor_silu_elements": descriptor_rows * hidden_width if self.moment_variant == "H" else 0,
            "moment_processed_context_add_elements": descriptor_rows * self.context_dim,
            "projected_context_gather_value_reads_per_slot": self.context_dim,
            "projected_context_gather_bytes_per_slot": self.context_dim * prepared.processed_state.element_size(),
            "moment_optional_counters_available": int(exact_counters),
        })
        if exact_counters:
            work.update({
                "moment_cache_logical_tensor_count": len(package_tensors),
                "moment_cache_unique_storage_count": len(package_storages),
                "moment_cache_unique_storage_bytes": sum(package_storages.values()),
                "moment_cache_incremental_storage_count": len(incremental_storages),
                "moment_cache_incremental_storage_bytes": sum(incremental_storages.values()),
                "moment_latent_variance_clamped_entries": int(audit_counts[0]),
                "moment_spatial_variance_clamped_entries": int(audit_counts[1]),
                "moment_latent_variance_large_negative_entries": int(audit_counts[2]),
                "moment_spatial_variance_large_negative_entries": int(audit_counts[3]),
            })
        if self.thermal_response:
            factor_entries = batch * heads * anchors * self.signed_carrier_width * module_donors
            work.update({
                "thermal_source_response_factor_entries_per_application": factor_entries,
                "thermal_source_response_factor_materialized_bytes_fp32_per_application": factor_entries * 4,
                "thermal_source_response_factor_materialized_bytes_fp64_per_application": factor_entries * 8,
                "thermal_source_response_factor_retained_bytes": 0,
            })

    def _query_work(self, organization, query, pack, *, controls: bool, field: bool, features: bool = True,
                    collect_exact_work_counts: bool | None = None) -> dict[str, int]:
        work = super()._query_work(
            organization,
            query,
            pack,
            controls=controls,
            field=field,
            features=features,
            collect_exact_work_counts=collect_exact_work_counts,
        )
        if organization.prepared_readout_values is not None:
            reads = work["projected_context_gather_slots"] * self.context_dim
            work["projected_context_gather_value_reads"] = reads
            work["projected_context_gather_bytes"] = reads * organization.prepared_readout_values.values.element_size()
        else:
            work["projected_context_gather_value_reads"] = 0
            work["projected_context_gather_bytes"] = 0
        return work

    def process_moments(
        self,
        descriptor: Tensor,
        *,
        anchor_mask: Tensor | None = None,
    ) -> HyperedgeProcessorOutput:
        """Evaluate the pure M/H map at externally supplied descriptor values."""
        if self.hyperedge_processor is None:
            raise ValueError("The C0 control has no moment processor.")
        return self.hyperedge_processor(descriptor, anchor_mask=anchor_mask)

    def export_prepared_moment_hyperedges(
        self, organization: PreparedInheritedOrganization
    ) -> PreparedMomentHyperedgeView:
        self.validate_prepared_state(organization)
        prepared = getattr(organization, "moment_hyperedges", None)
        if prepared is None:
            raise ValueError("C0 has no moment hyperedge cache to export.")
        return export_prepared_moment_hyperedges(prepared)

    def validate_prepared_state(self, organization: PreparedInheritedOrganization, query=None) -> None:
        super().validate_prepared_state(organization, query)
        prepared = getattr(organization, "moment_hyperedges", None)
        if self.moment_variant == "C0":
            if prepared is not None:
                raise ValueError("C0 organizations must not carry new moment processor factors.")
            return
        if not isinstance(prepared, PreparedMomentHyperedges):
            raise TypeError("M/H prepared organizations require their scene-owned moment cache.")
        scene = organization.scene
        prepared.validate_binding(
            frame_key=scene.coordinate_frame_key,
            scene_version=organization.scene_versions,
            owner_token=organization.owner_token,
            normalizer_key=scene.normalizer_key,
            scene_key=scene.scene_key,
            module_ids=scene.module_ids,
            environment_ids=scene.environment_ids,
            module_coordinates=scene.module_coordinates,
            environment_coordinates=scene.environment_coordinates,
        )
        if prepared.module.probabilities is not organization.module_atlas.probabilities:
            raise ValueError("Prepared module moments use a stale association atlas.")
        if prepared.environment.probabilities is not organization.environment_atlas.probabilities:
            raise ValueError("Prepared environment moments use a stale association atlas.")
        for label, stats, atlas, values, lift in (
            ("module", prepared.module, organization.module_atlas,
             organization.module_values, self.module_moment_lift),
            ("environment", prepared.environment, organization.environment_atlas,
             organization.environment_values, self.environment_moment_lift),
        ):
            if stats.values is not values.values or stats.lift_weight is not lift.weight:
                raise ValueError(f"Prepared {label} moment values or lift weights are stale.")
            if (stats.donor_valid is not atlas.donor_valid or stats.anchor_valid is not atlas.destination_valid
                    or stats.donor_ids is not atlas.donor_ids or stats.anchor_ids is not atlas.anchor_ids):
                raise ValueError(f"Prepared {label} moment source/destination identities are stale.")
        if prepared.module_measure is not organization.module_atlas.normalized_measure:
            raise ValueError("Prepared module source measure is stale.")
        if prepared.source_carriers is not organization.source_carriers:
            raise ValueError("Prepared source carrier factors are stale.")
        if prepared.module_mask is not scene.module_mask or prepared.environment_mask is not scene.environment_mask:
            raise ValueError("Prepared scene source/anchor masks are stale.")
        if organization.prepared_readout_values is None or organization.prepared_readout_values.values is not prepared.processed_state:
            raise ValueError("Prepared projected context must be the exact processed moment state.")

    def intervene_associations(self, organization, module_probabilities, environment_probabilities, *, label: str):
        result = super().intervene_associations(
            organization, module_probabilities, environment_probabilities, label=label
        )
        result.moment_hyperedges = None
        if self.moment_variant != "C0":
            self._attach_moment_hyperedges(
                result,
                module_probabilities=result.module_atlas.probabilities,
                environment_probabilities=result.environment_atlas.probabilities,
            )
        result.factor_versions = self._factors(result)
        return result

    def _source_transport_coefficients(self, organization: PreparedInheritedOrganization,
                                       dtype: torch.dtype) -> Tensor:
        """Build exact requested-dtype ``[B,H,E,R,M]`` factors from raw inputs."""
        prepared = getattr(organization, "moment_hyperedges", None)
        if self.moment_variant not in ("M", "H") or prepared is None:
            raise ValueError("Source moment transport coefficients exist only for prepared M/H scenes.")
        if not self.thermal_response or organization.source_carriers is None:
            raise ValueError("Source moment transport is available only to Thermal M/H.")
        if prepared.heat_mix_constant is None or prepared.heat_mix_spatial is None:
            raise ValueError("Thermal moment transport requires both prepared mix-output tensors.")
        if prepared.module_measure is not organization.module_atlas.normalized_measure:
            raise ValueError("Thermal moment transport measure is stale.")
        if prepared.source_carriers is not organization.source_carriers:
            raise ValueError("Thermal moment transport carriers are stale.")
        atlas = organization.module_atlas
        measure = prepared.module_measure.to(dtype)
        safe_measure = torch.where(atlas.donor_valid, measure, torch.ones_like(measure))
        density = atlas.probabilities.to(dtype) / safe_measure[:, None, None, :]
        density = density * atlas.donor_valid[:, None, None, :].to(dtype)
        carriers = organization.source_carriers.to(dtype)
        coordinates = prepared.module_coordinates.to(dtype)
        anchors = prepared.environment_coordinates.to(dtype)
        relative = coordinates[:, None, :, :] - anchors[:, :, None, :]
        mix0 = prepared.heat_mix_constant.to(dtype)
        mix_spatial = prepared.heat_mix_spatial.to(dtype)
        spatial_mix = torch.einsum("bherd,bemd->bherm", mix_spatial, relative)
        receiver_mix = 1.0 + mix0[..., None] + spatial_mix
        coefficients = (density[..., None] * carriers[:, :, None, :, :]).permute(0, 1, 2, 4, 3)
        coefficients = coefficients * receiver_mix
        if not torch.isfinite(coefficients).all():
            raise ValueError("Moment source transport is nonfinite in the requested arithmetic.")
        return coefficients

    def _anchor_response(self, organization: PreparedInheritedOrganization, controls: Tensor,
                         dtype: torch.dtype) -> Tensor:
        if self.moment_variant == "C0":
            return super()._anchor_response(organization, controls, dtype)
        self._validate_controls(organization, controls)
        coefficients = self._source_transport_coefficients(organization, dtype)
        return torch.einsum("bherm,bm->bher", coefficients, controls.to(dtype))

    def heat_vjp_prepared_affine(
        self,
        prepared_query: PreparedInheritedAffineQuery,
        cotangent: Tensor,
        *,
        accumulation_dtype: torch.dtype | None = None,
    ) -> Tensor:
        if self.moment_variant == "C0":
            return super().heat_vjp_prepared_affine(
                prepared_query, cotangent, accumulation_dtype=accumulation_dtype
            )
        self._validate_affine(prepared_query)
        organization, query = prepared_query.organization, prepared_query.query
        if cotangent.shape != query.coordinates.shape[:2] or cotangent.device != query.coordinates.device:
            raise ValueError("Cotangent must match native [B,Q] receivers and device.")
        if not torch.isfinite(cotangent).all():
            raise ValueError("Cotangent must be finite.")
        dtype = self._dtype(accumulation_dtype, cotangent.dtype)
        cover = query.receiver_cover
        assert cover is not None
        batch, heads, anchors, _ = organization.module_atlas.probabilities.shape
        weights = cotangent.to(dtype)
        if query.query_mask is not None:
            weights = weights * query.query_mask.to(dtype)
        adjoint = weights[:, :, None, None] * prepared_query.phi.to(dtype)

        # Apply Lambda^T before source transport, matching the same indexed
        # anchor map used by ordinary/JVP evaluation.
        indices = cover.anchor_indices[:, None].expand(batch, heads, query.num_queries, cover.slots)
        b = torch.arange(batch, device=indices.device)[:, None, None, None]
        h = torch.arange(heads, device=indices.device)[None, :, None, None]
        flat_indices = ((b * heads + h) * anchors + indices).reshape(-1)
        values = adjoint.permute(0, 2, 1, 3)[:, :, :, None, :] * (
            cover.weights.to(dtype) * cover.validity.to(dtype))[:, None, :, :, None]
        anchor_adjoint = weights.new_zeros(batch * heads * anchors, self.signed_carrier_width)
        anchor_adjoint = anchor_adjoint.index_add(
            0, flat_indices, values.reshape(-1, self.signed_carrier_width)
        ).reshape(batch, heads, anchors, self.signed_carrier_width)
        coefficients = self._source_transport_coefficients(organization, dtype)
        gradient = torch.einsum("bher,bherm->bm", anchor_adjoint, coefficients)

        pack = prepared_query.local_pairs
        if pack.count:
            local = (weights[pack.batch_index, pack.query_index]
                     * pack.support_weight.to(dtype)
                     * prepared_query.local_coefficients.to(dtype))
            gradient = gradient.reshape(-1).index_add(
                0,
                pack.batch_index * organization.scene.num_modules + pack.module_index,
                local,
            ).reshape(batch, organization.scene.num_modules)
        return gradient

    def export_source_kernels(
        self,
        organization: PreparedInheritedOrganization,
        query,
        *,
        accumulation_dtype: torch.dtype | None = None,
        local_gates: Tensor | None = None,
        max_entries: int = 1_048_576,
    ) -> Tensor:
        if self.moment_variant == "C0":
            return super().export_source_kernels(
                organization, query, accumulation_dtype=accumulation_dtype,
                local_gates=local_gates, max_entries=max_entries,
            )
        if query.batch_size * query.num_queries * organization.scene.num_modules > max_entries:
            raise ValueError("Explicit source-kernel export exceeds its bounded diagnostic budget.")
        if not self.thermal_response:
            raise ValueError("This IAG core has no affine thermal response.")
        _field, phi, pack, coefficients_local, _mc, _ec, _work, _trace, _receiver_readout_trace = self._decode_features(
            organization, query, local_gates, field_head=False
        )
        assert phi is not None and query.receiver_cover is not None
        dtype = self._dtype(accumulation_dtype, organization.module_states.dtype)
        transport = self._source_transport_coefficients(organization, dtype)
        kernels = organization.module_states.new_zeros(
            query.batch_size, query.num_queries, organization.scene.num_modules, dtype=dtype
        )

        # Gather one [B,E,M] coefficient plane at a time.  This bounds live
        # query-space storage to [B,Q,M], never [B,Q,slots,H,R,M].
        for head in range(self.num_heads):
            for carrier in range(self.signed_carrier_width):
                anchor_plane = transport[:, head, :, carrier, :][:, None, :, :]
                query_plane = gather_cover_indexed(
                    anchor_plane,
                    query.receiver_cover,
                    anchor_ids=organization.scene.environment_ids,
                    validate_identity=False,
                ).squeeze(2)
                kernels = kernels + query_plane * phi[:, :, head, carrier, None].to(dtype)

        if pack is not None and pack.count:
            indices = ((pack.batch_index * query.num_queries + pack.query_index)
                       * organization.scene.num_modules + pack.module_index)
            kernels = kernels.reshape(-1).index_add(
                0, indices, pack.support_weight.to(dtype) * coefficients_local.to(dtype)
            ).reshape_as(kernels)
        if query.query_mask is not None:
            kernels = kernels * query.query_mask[:, :, None].to(dtype)
        return kernels

    def zero_new_outputs_for_diagnostic(
        self, organization: PreparedInheritedOrganization
    ) -> PreparedInheritedOrganization:
        """Return a new valid organization with only M/H outputs zeroed.

        The original projected C0 state, memberships, local response and typed
        associations remain attached.  This is an eval-only counterfactual;
        it never mutates a live prepared cache.
        """
        if self.training or torch.is_grad_enabled():
            raise ValueError("Zero-output diagnostics require eval mode under no_grad.")
        if self.moment_variant == "C0":
            raise ValueError("C0 has no new moment outputs to intervene on.")
        self.validate_prepared_state(organization)
        prepared = organization.moment_hyperedges
        lifetime = _GraphLifetime()
        zero_context = torch.zeros_like(prepared.context_residual)
        zero_constant = None if prepared.heat_mix_constant is None else torch.zeros_like(prepared.heat_mix_constant)
        zero_spatial = None if prepared.heat_mix_spatial is None else torch.zeros_like(prepared.heat_mix_spatial)
        zeroed = PreparedMomentHyperedges(
            module=prepared.module,
            environment=prepared.environment,
            descriptor=prepared.descriptor,
            base_projected_state=prepared.base_projected_state,
            module_measure=prepared.module_measure,
            source_carriers=prepared.source_carriers,
            module_coordinates=prepared.module_coordinates,
            environment_coordinates=prepared.environment_coordinates,
            module_ids=prepared.module_ids,
            environment_ids=prepared.environment_ids,
            module_mask=prepared.module_mask,
            environment_mask=prepared.environment_mask,
            source_module_coordinates=prepared.source_module_coordinates,
            source_environment_coordinates=prepared.source_environment_coordinates,
            frame_key=prepared.frame_key,
            scene_version=prepared.scene_version,
            owner_token=prepared.owner_token,
            normalizer_key=prepared.normalizer_key,
            scene_key=prepared.scene_key,
            processor_preactivation=prepared.processor_preactivation,
            processor_hidden=prepared.processor_hidden,
            context_residual=zero_context,
            processed_state=prepared.base_projected_state,
            heat_mix_constant=zero_constant,
            heat_mix_spatial=zero_spatial,
            graph_lifetime=lifetime,
        )
        result = replace(
            organization,
            graph_lifetime=lifetime,
            prepared_readout_values=None,
            factor_versions=(),
            trace={**(organization.trace or {}), "moment_hyperedge_outputs_zeroed": True},
        )
        result.moment_hyperedges = zeroed
        zeroed.register_lifetime_hooks(lifetime)
        result.prepared_readout_values = self._prepare_scene_readout(result)
        self._record_moment_work_counts(result, zeroed)
        result.factor_versions = self._factors(result)
        return result


__all__ = ["MomentHyperedgeCoreV1"]

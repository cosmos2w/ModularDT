"""Thermal frozen-packet interface and matched heat-diffusion experiments.

This module keeps the hidden heat target out of the candidate-graph callback.
The provider receives only the current centered logits plus an immutable view
of known geometry, operating context, supplied total heat, and fixed sensors.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
import random
import time

import numpy as np
import torch
from torch import nn

from channelthermal.environment import ChannelThermalEnvironmentBuilder
from channelthermal.model import ChannelThermalHONFModel
from honf_forward_core.config import BatchData
from honf_forward_core.interface_fields.budgeted_frontier import (
    FrontierSelection,
    enumerate_frontier_cuts,
    select_frontier_by_predictions,
)
from honf_forward_core.interface_fields.input_cover_organizer import InputOnlyCoverOrganizer
from honf_forward_core.interface_fields.adaptive_interaction_cover import MechanismPlan
from honf_forward_core.interface_fields.interaction_interface import (
    InteractionInterface,
    interaction_interface_from_plan,
)
from honf_inverse_core.models.frozen_packet_diffusion import (
    ConditionalPacketDenoiser,
    DiffusionCondition,
    FrozenPacketDiffusion,
    LinksForCandidate,
    PacketLinks,
    heat_from_logits,
)

from .heat_allocation import ThermalHeatBatch, ThermalHeatFeatureScaler, ThermalHeatTask, collate_thermal_heat_tasks


@dataclass(frozen=True)
class ThermalProviderKnownInputs:
    """Known per-case inputs exposed to candidate graph construction."""

    module_centers: torch.Tensor
    sensor_xy: torch.Tensor
    physical_context: torch.Tensor
    total_heat: torch.Tensor

    def __post_init__(self) -> None:
        batch, modules, spatial_dim = self.module_centers.shape
        if spatial_dim != 2 or self.sensor_xy.ndim != 3 or self.sensor_xy.shape[0] != batch:
            raise ValueError("Thermal known coordinates must be batched 2-D tensors.")
        if self.physical_context.shape != (batch, 10):
            raise ValueError("Known operating context must use the ten-field Thermal schema.")
        if not torch.isfinite(self.module_centers).all() or not torch.isfinite(self.sensor_xy).all():
            raise ValueError("Known physical coordinates must be finite.")
        if not torch.isfinite(self.physical_context).all() or bool((self.physical_context <= 0).any()):
            raise ValueError("Known operating context must be positive and finite.")
        if self.total_heat.shape != (batch, 1) or not torch.isfinite(self.total_heat).all() or torch.any(self.total_heat <= 0):
            raise ValueError("Supplied total heat must be a positive finite [B,1] tensor.")
        if modules < 1:
            raise ValueError("At least one padded module slot is required.")


def provider_known_inputs(batch: ThermalHeatBatch) -> ThermalProviderKnownInputs:
    """Strip all clean target fields before constructing the inverse provider."""

    return ThermalProviderKnownInputs(
        module_centers=batch.module_centers.detach(),
        sensor_xy=batch.sensor_xy.detach(),
        physical_context=batch.physical_context.detach(),
        total_heat=batch.total_heat.detach(),
    )


def packet_links_from_interfaces(
    interfaces: Sequence[InteractionInterface],
    module_centers: torch.Tensor,
    sensor_xy: torch.Tensor,
    *,
    expected_module_valid: torch.Tensor,
    phase_by_mechanism: Mapping[str, str | None] | None = None,
) -> PacketLinks:
    """Build inverse links from the actual typed forward organizer routes.

    ``MM`` supplies module-to-module aggregation, ``QM`` supplies the sparse
    observed-sensor-to-module route, and the ME/EM/QE matrices preserve the
    three distinct module/environment/query paths. No route is inferred from
    QE or from another typed mechanism.
    """

    batch = len(interfaces)
    if batch < 1 or module_centers.ndim != 3 or sensor_xy.ndim != 3:
        raise ValueError("One interaction interface and batched coordinates are required per case.")
    if module_centers.shape[0] != batch or sensor_xy.shape[0] != batch:
        raise ValueError("Interface count and known coordinate batches must match.")
    if expected_module_valid.shape != module_centers.shape[:2]:
        raise ValueError("Expected module validity must align with the padded geometry.")
    if module_centers.shape[-1] != 2 or sensor_xy.shape[-1] != 2:
        raise ValueError("Thermal module and sensor coordinates must be 2-D.")
    phases = {str(key).upper(): value for key, value in (phase_by_mechanism or {}).items()}
    allowed = {"MM", "ME", "EM", "QM", "QE"}
    if set(phases) - allowed:
        raise ValueError(f"Unknown interaction phase keys: {sorted(set(phases) - allowed)}")

    device, dtype = module_centers.device, module_centers.dtype
    max_modules = int(module_centers.shape[1])
    max_sensors = int(sensor_xy.shape[1])
    embedding_dim: int | None = None
    module_matrices: list[torch.Tensor] = []
    sensor_module_matrices: list[torch.Tensor] = []
    module_embeddings: list[torch.Tensor] = []
    module_environment: list[torch.Tensor] = []
    environment_module: list[torch.Tensor] = []
    sensor_environment: list[torch.Tensor] = []
    environment_embeddings: list[torch.Tensor] = []
    environment_valid: list[torch.Tensor] = []
    environment_count: int | None = None

    def phase(mechanism: str) -> str | None:
        return phases.get(mechanism)

    for case, interface in enumerate(interfaces):
        native_modules = interface.module_descriptors["coordinates"].shape[0]
        if int(native_modules) != max_modules:
            raise ValueError("Candidate interface module axis must match the padded inverse condition.")
        interface_valid = interface.module_validity.to(device=device) > 0.5
        expected_valid = expected_module_valid[case].to(device=device) > 0.5
        if not torch.equal(interface_valid, expected_valid):
            raise ValueError("Candidate interface validity changed from the known design mask.")
        mm = interface.module_pair_weights(phase=phase("MM")).to(device=device, dtype=dtype)
        qm = interface.receiver_source_weights(
            "QM", sensor_xy[case], phase=phase("QM"),
        ).to(device=device, dtype=dtype)
        if mm.shape != (max_modules, max_modules) or qm.shape != (max_sensors, max_modules):
            raise ValueError("Candidate MM/QM route shapes do not match module and sensor panels.")

        module_embedding = interface.module_embeddings
        env_embedding = interface.environment_embeddings
        if module_embedding is None or env_embedding is None:
            raise ValueError("Frozen forward/organizer embeddings are required for Thermal packet reuse.")
        module_embedding = module_embedding.to(device=device, dtype=dtype)
        env_embedding = env_embedding.to(device=device, dtype=dtype)
        if module_embedding.shape[0] != max_modules or env_embedding.ndim != 2:
            raise ValueError("Frozen module/environment embeddings have incompatible source axes.")
        if embedding_dim is None:
            embedding_dim = int(module_embedding.shape[-1])
            environment_count = int(env_embedding.shape[0])
        if int(module_embedding.shape[-1]) != embedding_dim:
            raise ValueError("Module embedding widths must agree across cases.")
        if int(env_embedding.shape[0]) != environment_count or int(env_embedding.shape[-1]) != embedding_dim:
            raise ValueError("Environment embedding dimensions must agree across cases.")

        env_coords = interface.environment_descriptors["coordinates"].to(device=device, dtype=dtype)
        me = interface.receiver_source_weights(
            "ME", module_centers[case], phase=phase("ME"),
        ).to(device=device, dtype=dtype)
        em = interface.receiver_source_weights(
            "EM", env_coords, phase=phase("EM"),
        ).to(device=device, dtype=dtype)
        qe = interface.receiver_source_weights(
            "QE", sensor_xy[case], phase=phase("QE"),
        ).to(device=device, dtype=dtype)
        if me.shape != (max_modules, int(environment_count)):
            raise ValueError("ME typed route has an incompatible module/environment shape.")
        if em.shape != (int(environment_count), max_modules):
            raise ValueError("EM typed route has an incompatible environment/module shape.")
        if qe.shape != (max_sensors, int(environment_count)):
            raise ValueError("QE typed route has an incompatible sensor/environment shape.")

        module_matrices.append(mm)
        sensor_module_matrices.append(qm)
        module_embeddings.append(module_embedding)
        module_environment.append(me)
        environment_module.append(em)
        sensor_environment.append(qe)
        environment_embeddings.append(env_embedding)
        environment_valid.append(interface.environment_validity.to(device=device))

    assert embedding_dim is not None and environment_count is not None
    return PacketLinks(
        module_source=torch.stack(module_matrices),
        sensor_source=torch.stack(sensor_module_matrices),
        module_embeddings=torch.stack(module_embeddings),
        module_environment=torch.stack(module_environment),
        environment_module=torch.stack(environment_module),
        sensor_environment=torch.stack(sensor_environment),
        environment_embeddings=torch.stack(environment_embeddings),
        environment_valid=torch.stack(environment_valid),
    )


InterfaceBuilder = Callable[
    [torch.Tensor, ThermalProviderKnownInputs, DiffusionCondition],
    Sequence[InteractionInterface],
]


@dataclass(frozen=True)
class ThermalFrontierEvidence:
    """Per-case learned cut selection and explicit unsupported fallback."""

    selected_frontier: tuple[int, ...] | None
    least_risk_frontier: tuple[int, ...]
    unsupported_at_budget: bool
    selected_index: int | None
    least_risk_index: int
    predicted_frontier: tuple[int, ...]
    prediction_scope: str
    predicted_work: float
    predicted_role_distortion: tuple[float, ...]


def _frontier_evidence_for_selection(
    selection: FrontierSelection,
    prediction: Any,
) -> ThermalFrontierEvidence:
    fallback = selection.unsupported_at_budget
    chosen = selection.least_risk_frontier if fallback else selection.selected_frontier
    prediction_index = selection.least_risk_index if fallback else selection.selected_index
    if chosen is None or prediction_index is None:
        raise RuntimeError("Frontier selector returned neither a selected nor least-risk prediction.")
    return ThermalFrontierEvidence(
        selected_frontier=selection.selected_frontier,
        least_risk_frontier=selection.least_risk_frontier,
        unsupported_at_budget=fallback,
        selected_index=selection.selected_index,
        least_risk_index=selection.least_risk_index,
        predicted_frontier=chosen,
        prediction_scope="least_risk_research_only" if fallback else "selected_deployable",
        predicted_work=float(prediction.predicted_work[prediction_index].detach().cpu()),
        predicted_role_distortion=tuple(
            float(value)
            for value in prediction.role_distortion[prediction_index].detach().cpu().tolist()
        ),
    )


class ThermalCandidateInterfaceBuilder:
    """Encode current candidate heat and export its frozen selected interface."""

    def __init__(
        self,
        model: ChannelThermalHONFModel,
        organizer: InputOnlyCoverOrganizer,
        *,
        budget_fractions: Mapping[str, float],
        role_tolerance: Sequence[float],
        numerical_state_version: str,
        evidence_scope: str = "frozen-forward-organizer-candidate-rebuild",
        max_frontier_depth: int = 3,
        dataset_config: Mapping[str, Any] | None = None,
        normalization_stats: Mapping[str, Any] | None = None,
        plan_intervention: Callable[[MechanismPlan, int], tuple[MechanismPlan, Mapping[str, Any]]] | None = None,
    ) -> None:
        self.model = model
        self.organizer = organizer
        self.model.eval().requires_grad_(False)
        self.organizer.eval().requires_grad_(False)
        self.budget_fractions = {str(key).upper(): float(value) for key, value in budget_fractions.items()}
        self.role_tolerance = torch.as_tensor(tuple(role_tolerance), dtype=torch.float32)
        self.numerical_state_version = str(numerical_state_version)
        self.evidence_scope = str(evidence_scope)
        self.max_frontier_depth = int(max_frontier_depth)
        self.dataset_config = dict(dataset_config or {})
        self.normalization_stats = dict(normalization_stats or model.global_normalization_stats)
        self.plan_intervention = plan_intervention
        if self.max_frontier_depth < 0:
            raise ValueError("max_frontier_depth must be nonnegative.")
        if organizer.frontier_utility_head is None:
            raise ValueError("Thermal inverse reuse requires a trained frontier utility head.")
        if organizer.frontier_utility_head.role_count != int(self.role_tolerance.numel()):
            raise ValueError("Thermal frontier tolerances must cover every learned receiver role.")
        if not self.numerical_state_version.strip() or not self.evidence_scope.strip():
            raise ValueError("Candidate graph identity and evidence scope cannot be empty.")
        self.last_frontier_evidence: tuple[ThermalFrontierEvidence, ...] = ()
        self.last_intervention_records: tuple[Mapping[str, Any], ...] = ()

    def _encode_candidate(
        self,
        candidate_heat: torch.Tensor,
        known: ThermalProviderKnownInputs,
        condition: DiffusionCondition,
    ):
        model = self.model
        centers = known.module_centers.to(device=candidate_heat.device, dtype=candidate_heat.dtype)
        present = condition.module_valid.to(device=candidate_heat.device, dtype=candidate_heat.dtype)
        physical = known.physical_context.to(device=candidate_heat.device, dtype=candidate_heat.dtype)
        re = physical[:, 0:1]
        u_in = physical[:, 1:2]
        lx = physical[:, 2:3]
        ly = physical[:, 3:4]
        material = torch.stack(
            (physical[:, 4], physical[:, 5], physical[:, 6], physical[:, 7], physical[:, 8], physical[:, 9]),
            dim=-1,
        )
        for physical_length, checkpoint_length, name in (
            (lx, float(model.config.core_honf.domain_length_x), "domain_length_x"),
            (ly, float(model.config.core_honf.domain_length_y), "domain_length_y"),
            (physical[:, 9:10], float(model.config.core_honf.module_radius), "module_radius"),
        ):
            expected = physical_length.new_full(physical_length.shape, checkpoint_length)
            if not torch.allclose(physical_length, expected, atol=1.0e-6, rtol=1.0e-6):
                raise ValueError(f"Known Thermal {name} differs from the frozen checkpoint geometry.")
        heat_for_adapter = candidate_heat.squeeze(-1)
        if bool(self.dataset_config.get("normalize_inputs", False)):
            if not {"heat_power_mean", "heat_power_std"}.issubset(self.normalization_stats):
                raise KeyError("Frozen Thermal candidate encoding lacks input heat normalization statistics.")
            heat_mean = torch.as_tensor(
                self.normalization_stats["heat_power_mean"], device=centers.device, dtype=centers.dtype
            ).reshape(-1)[0]
            heat_std = torch.as_tensor(
                self.normalization_stats["heat_power_std"], device=centers.device, dtype=centers.dtype
            ).reshape(-1)[0].clamp_min(1.0e-8)
            heat_for_adapter = (heat_for_adapter - heat_mean) / heat_std
        adapter = model.input_adapter(
            re=re,
            u_in=u_in,
            module_centers=centers,
            heat_powers=heat_for_adapter * float(model.config.channelthermal.heat_scale),
            module_present=present,
            material_params=material,
            domain_length_x=lx,
            domain_length_y=ly,
        )
        env = ChannelThermalEnvironmentBuilder()(
            batch_size=int(centers.shape[0]),
            num_env_tokens_x=int(model.config.core_honf.num_env_tokens_x),
            num_env_tokens_y=int(model.config.core_honf.num_env_tokens_y),
            domain_length_x=float(model.config.core_honf.domain_length_x),
            domain_length_y=float(model.config.core_honf.domain_length_y),
            device=centers.device,
            dtype=centers.dtype,
        )

        # Run1804 is the Dense native architecture. Its receiver tree uses
        # the backend default universe (environment quadrature plus active
        # module centers); P0/P1 port and solid roles remain actual supervised
        # query panels in the Thermal wrapper rather than tree anchors.
        return model.core.encode_case(
            BatchData(
                module_centers=adapter.module_centers,
                module_present=adapter.module_present,
                module_features=adapter.module_features,
                global_context=adapter.global_context,
                query_xy=known.sensor_xy.to(device=centers.device, dtype=centers.dtype),
                query_time=None,
                target_field=None,
                case_name="thermal-inverse-candidate",
                metadata={},
                env_coords=env.env_coords,
                env_features=env.env_features,
            )
        )

    @torch.no_grad()
    def __call__(
        self,
        candidate_heat: torch.Tensor,
        known: ThermalProviderKnownInputs,
        condition: DiffusionCondition,
    ) -> Sequence[InteractionInterface]:
        self.model.eval()
        self.organizer.eval()
        encoded = self._encode_candidate(candidate_heat, known, condition)
        if not bool(getattr(self.model.core.backend, "optional_native_policy", False)):
            raise TypeError("Frozen Thermal interaction reuse requires the checkpoint-native Dense cover backend.")
        trees = self.model.core.backend.build_case_trees(encoded)
        score_inputs = {
            "module_states": encoded.module_tokens,
            "environment_states": encoded.env_tokens,
            "global_state": encoded.global_token,
        }
        scores = self.organizer.score_cases(
            encoded,
            score_inputs,
            trees,
            budgets=self.budget_fractions,
        )
        selected_cuts: list[tuple[int, ...]] = []
        evidence: list[ThermalFrontierEvidence] = []
        for case, tree in enumerate(trees):
            cuts = enumerate_frontier_cuts(tree, max_depth=self.max_frontier_depth)
            prediction = self.organizer.score_frontiers(
                scores[case],
                tree,
                budget_vector=scores[case].budget_vector,
                cuts=cuts,
                max_depth=self.max_frontier_depth,
            )
            selection: FrontierSelection = select_frontier_by_predictions(
                prediction,
                role_tolerance=self.role_tolerance.to(
                    device=prediction.role_distortion.device,
                    dtype=prediction.role_distortion.dtype,
                ),
            )
            fallback = selection.unsupported_at_budget
            chosen = selection.least_risk_frontier if fallback else selection.selected_frontier
            if chosen is None:
                raise RuntimeError("Frontier selector returned neither a deployment cut nor research fallback.")
            selected_cuts.append(chosen)
            evidence.append(_frontier_evidence_for_selection(selection, prediction))
        plans = self.organizer.plans_from_scores(
            scores,
            encoded,
            trees,
            hard=True,
            frontier_cuts=selected_cuts,
            budget_fractions=self.budget_fractions,
        )
        if self.plan_intervention is not None:
            intervened = tuple(self.plan_intervention(plan, case) for case, plan in enumerate(plans))
            plans = tuple(item[0] for item in intervened)
            self.last_intervention_records = tuple(item[1] for item in intervened)
        else:
            self.last_intervention_records = ()
        self.last_frontier_evidence = tuple(evidence)
        return tuple(
            interaction_interface_from_plan(
                plans[case],
                encoded,
                scores[case],
                numerical_state_version=self.numerical_state_version,
                evidence_scope=self.evidence_scope,
                case_index=case,
            )
            for case in range(len(plans))
        )


class ThermalCandidatePacketProvider:
    """Rebuild typed candidate interfaces from current heat logits."""

    def __init__(
        self,
        known: ThermalProviderKnownInputs,
        interface_builder: InterfaceBuilder,
        *,
        phase_by_mechanism: Mapping[str, str | None] | None = None,
    ) -> None:
        self.known = known
        self.interface_builder = interface_builder
        self.phase_by_mechanism = dict(phase_by_mechanism or {})
        self.last_candidate_heat: torch.Tensor | None = None
        self.last_interfaces: tuple[InteractionInterface, ...] = ()
        self.last_frontier_evidence: tuple[ThermalFrontierEvidence, ...] = ()

    def __call__(self, state: torch.Tensor, condition: DiffusionCondition) -> PacketLinks:
        candidate_heat = heat_from_logits(state, self.known.total_heat, condition.design_mask)
        interfaces = tuple(self.interface_builder(candidate_heat, self.known, condition))
        if len(interfaces) != int(condition.known_state.shape[0]):
            raise ValueError("Candidate interface builder returned the wrong number of cases.")
        links = packet_links_from_interfaces(
            interfaces,
            self.known.module_centers,
            self.known.sensor_xy,
            expected_module_valid=condition.module_valid,
            phase_by_mechanism=self.phase_by_mechanism,
        )
        self.last_candidate_heat = candidate_heat.detach().clone()
        self.last_interfaces = interfaces
        self.last_frontier_evidence = tuple(
            getattr(self.interface_builder, "last_frontier_evidence", ())
        )
        return links


@dataclass(frozen=True)
class MatchedDiffusionTrainingResult:
    graph_model: FrozenPacketDiffusion
    dense_model: FrozenPacketDiffusion
    graph_losses: tuple[float, ...]
    dense_losses: tuple[float, ...]
    initial_denoiser_hash: str
    frozen_state_hashes_before: Mapping[str, str]
    frozen_state_hashes_after: Mapping[str, str]
    updates_per_arm: int


def _module_state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _assert_state_unchanged(before: Mapping[str, torch.Tensor], module: nn.Module, name: str) -> None:
    after = module.state_dict()
    if before.keys() != after.keys():
        raise RuntimeError(f"Frozen module {name!r} changed its state_dict keys during inverse training.")
    changed = [key for key in before if not torch.equal(before[key], after[key].detach().cpu())]
    if changed:
        raise RuntimeError(f"Frozen module {name!r} mutated during inverse training: {changed[:6]}.")


def _cpu_state(value: Any) -> Any:
    """Copy optimizer/model checkpoint data to CPU without retaining live tensors."""

    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    return deepcopy(value)


def train_matched_heat_diffusion(
    train_tasks: Sequence[ThermalHeatTask],
    *,
    scaler: ThermalHeatFeatureScaler,
    denoiser_template: ConditionalPacketDenoiser,
    provider_factory: Callable[[ThermalProviderKnownInputs], LinksForCandidate],
    frozen_modules: Mapping[str, nn.Module],
    updates: int,
    batch_size: int = 4,
    steps: int = 20,
    learning_rate: float = 2.0e-4,
    seed: int = 74029,
    device: torch.device | str = "cpu",
    resume_payload: Mapping[str, Any] | None = None,
    checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
    attempt_callback: Callable[[Mapping[str, Any]], None] | None = None,
    update_callback: Callable[[Mapping[str, Any]], None] | None = None,
    checkpoint_every_updates: int = 10,
) -> MatchedDiffusionTrainingResult:
    """Fit matched inverse denoisers with atomic resume and attempt hooks.

    ``updates`` is the final completed-update ceiling, including when
    resuming. The callback before each ``optimizer.step`` should durably log
    the attempted arm/update; a checkpoint callback sees both arms only after
    a matched update has completed, together with both optimizer states and
    both update/noise RNG streams.
    """

    if not train_tasks or any(task.split != "train" for task in train_tasks):
        raise ValueError("Matched inverse fitting requires training-split tasks only.")
    if any(task.module_valid.sum() < 2 for task in train_tasks):
        raise ValueError("M=1 tasks have no nontrivial heat-allocation degree of freedom.")
    if not 1 <= int(updates) <= 800 or not 1 <= int(batch_size) <= len(train_tasks):
        raise ValueError("Use 1..800 updates and a batch size within the task cohort.")
    if not 2 <= int(steps) <= 20 or learning_rate <= 0.0:
        raise ValueError("Diffusion steps must be 2..20 and learning_rate positive.")
    if checkpoint_every_updates < 1:
        raise ValueError("checkpoint_every_updates must be positive.")
    target_device = torch.device(device)

    frozen_before: dict[str, dict[str, torch.Tensor]] = {}
    frozen_hashes_before: dict[str, str] = {}
    for name, module in frozen_modules.items():
        module.eval()
        module.requires_grad_(False)
        frozen_before[name] = {
            key: value.detach().to(device="cpu").clone()
            for key, value in module.state_dict().items()
        }
        frozen_hashes_before[name] = _module_state_hash(module)

    graph_model = FrozenPacketDiffusion(deepcopy(denoiser_template), task="heat", steps=steps).to(target_device)
    dense_model = FrozenPacketDiffusion(deepcopy(denoiser_template), task="heat", steps=steps).to(target_device)
    graph_model.train()
    dense_model.train()
    initial_hash = _module_state_hash(graph_model.denoiser)
    if _module_state_hash(dense_model.denoiser) != initial_hash:
        raise RuntimeError("Matched inverse arms did not start with identical denoiser weights.")
    graph_optimizer = torch.optim.AdamW(graph_model.denoiser.parameters(), lr=learning_rate)
    dense_optimizer = torch.optim.AdamW(dense_model.denoiser.parameters(), lr=learning_rate)

    index_rng = torch.Generator(device="cpu").manual_seed(int(seed))
    noise_rng = torch.Generator(device="cpu").manual_seed(int(seed) ^ 0x5A17)
    graph_losses: list[float] = []
    dense_losses: list[float] = []
    start_update = 0
    graph_attempted = 0
    dense_attempted = 0
    task_ids = tuple(str(task.case_id) for task in train_tasks)
    recipe = {
        "task": "thermal_heat_allocation",
        "train_case_ids": list(task_ids),
        "batch_size": int(batch_size),
        "diffusion_steps": int(steps),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "checkpoint_every_updates": int(checkpoint_every_updates),
    }
    initial_hash = _module_state_hash(graph_model.denoiser)

    if resume_payload is not None:
        if int(resume_payload.get("schema_version", -1)) != 1:
            raise ValueError("Inverse resume checkpoint schema is unsupported.")
        saved_recipe = dict(resume_payload.get("recipe", {}))
        if saved_recipe != recipe:
            raise ValueError("Inverse resume checkpoint task cohort or matched recipe differs.")
        start_update = int(resume_payload.get("completed_updates", -1))
        if not 0 <= start_update < int(updates):
            raise ValueError("Inverse resume checkpoint is outside the requested final update cap.")
        initial_hash = str(resume_payload.get("initial_denoiser_hash", ""))
        if not initial_hash:
            raise ValueError("Inverse resume checkpoint lacks its matched initial-denoiser hash.")
        graph_model.load_state_dict(resume_payload["graph_model_state"], strict=True)
        dense_model.load_state_dict(resume_payload["dense_model_state"], strict=True)
        graph_optimizer.load_state_dict(resume_payload["graph_optimizer_state"])
        dense_optimizer.load_state_dict(resume_payload["dense_optimizer_state"])
        index_rng.set_state(resume_payload["index_rng_state"])
        noise_rng.set_state(resume_payload["noise_rng_state"])
        graph_losses = [float(value) for value in resume_payload.get("graph_losses", ())]
        dense_losses = [float(value) for value in resume_payload.get("dense_losses", ())]
        graph_attempted = int(resume_payload.get("graph_attempted_calls", start_update))
        dense_attempted = int(resume_payload.get("dense_attempted_calls", start_update))
        torch.set_rng_state(resume_payload["torch_rng_state"])
        if torch.cuda.is_available() and resume_payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(resume_payload["cuda_rng_state_all"])
        random.setstate(resume_payload["python_rng_state"])
        np.random.set_state(resume_payload["numpy_rng_state"])
    elif _module_state_hash(dense_model.denoiser) != initial_hash:
        raise RuntimeError("Matched inverse arms did not start with identical denoiser weights.")

    def checkpoint_state(completed: int) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "completed_updates": int(completed),
            "recipe": recipe,
            "initial_denoiser_hash": initial_hash,
            "graph_model_state": _cpu_state(graph_model.state_dict()),
            "dense_model_state": _cpu_state(dense_model.state_dict()),
            "graph_optimizer_state": _cpu_state(graph_optimizer.state_dict()),
            "dense_optimizer_state": _cpu_state(dense_optimizer.state_dict()),
            "index_rng_state": index_rng.get_state().clone(),
            "noise_rng_state": noise_rng.get_state().clone(),
            "torch_rng_state": torch.get_rng_state().clone(),
            "cuda_rng_state_all": (
                [value.cpu().clone() for value in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else None
            ),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "graph_losses": list(graph_losses),
            "dense_losses": list(dense_losses),
            "graph_attempted_calls": int(graph_attempted),
            "dense_attempted_calls": int(dense_attempted),
        }

    if checkpoint_callback is not None and resume_payload is None:
        checkpoint_callback(checkpoint_state(0))

    for _update in range(start_update, int(updates)):
        indices = torch.randint(len(train_tasks), (batch_size,), generator=index_rng).tolist()
        selected = [train_tasks[index] for index in indices]
        batch = collate_thermal_heat_tasks(selected, scaler=scaler, device=target_device)
        timesteps = torch.randint(steps, (batch_size,), generator=noise_rng).to(target_device)
        noise = torch.randn(
            batch.clean_state.shape,
            generator=noise_rng,
            dtype=batch.clean_state.dtype,
            device="cpu",
        ).to(target_device)
        provider = provider_factory(provider_known_inputs(batch))
        for arm, model, optimizer, dense, losses in (
            ("I-G", graph_model, graph_optimizer, False, graph_losses),
            ("I-dense", dense_model, dense_optimizer, True, dense_losses),
        ):
            attempted = graph_attempted if arm == "I-G" else dense_attempted
            if attempted >= 800:
                raise RuntimeError(f"{arm} exceeded the 800 attempted inverse-update ceiling.")
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(
                batch.clean_state,
                batch.condition,
                provider,
                dense=dense,
                timesteps=timesteps,
                noise=noise,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Matched diffusion training produced a nonfinite loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), max_norm=1.0)
            if attempt_callback is not None:
                attempt_callback({
                    "arm": arm,
                    "completed_updates_before_attempt": int(_update),
                    "attempted_calls_before_attempt": int(attempted),
                    "next_attempted_call": int(attempted) + 1,
                    "recorded_before_optimizer_step": True,
                    "timestamp_unix": time.time(),
                })
            if arm == "I-G":
                graph_attempted += 1
            else:
                dense_attempted += 1
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            if update_callback is not None:
                update_callback({
                    "arm": arm,
                    "completed_update": int(_update) + 1,
                    "attempted_calls": int(graph_attempted if arm == "I-G" else dense_attempted),
                    "loss": loss_value,
                })
        completed = int(_update) + 1
        if checkpoint_callback is not None and (
            completed % int(checkpoint_every_updates) == 0 or completed == int(updates)
        ):
            checkpoint_callback(checkpoint_state(completed))

    graph_model.eval()
    dense_model.eval()
    frozen_hashes_after: dict[str, str] = {}
    for name, module in frozen_modules.items():
        _assert_state_unchanged(frozen_before[name], module, name)
        frozen_hashes_after[name] = _module_state_hash(module)
        if frozen_hashes_after[name] != frozen_hashes_before[name]:
            raise RuntimeError(f"Frozen module {name!r} changed its state hash during inverse training.")
    return MatchedDiffusionTrainingResult(
        graph_model=graph_model,
        dense_model=dense_model,
        graph_losses=tuple(graph_losses),
        dense_losses=tuple(dense_losses),
        initial_denoiser_hash=initial_hash,
        frozen_state_hashes_before=frozen_hashes_before,
        frozen_state_hashes_after=frozen_hashes_after,
        updates_per_arm=int(updates),
    )


@dataclass(frozen=True)
class ThermalInverseSamples:
    """All generated logits, heat allocations, and intermediate sample trails."""

    case_ids: tuple[str, ...]
    total_heat: torch.Tensor
    reference_heat: torch.Tensor
    graph_heat: torch.Tensor
    dense_heat: torch.Tensor
    graph_heat_trails: torch.Tensor
    dense_heat_trails: torch.Tensor
    graph_logit_trails: torch.Tensor
    dense_logit_trails: torch.Tensor
    trail_timesteps: tuple[int, ...]


def _sample_arm(
    model: FrozenPacketDiffusion,
    condition: DiffusionCondition,
    provider: LinksForCandidate,
    *,
    total_heat: torch.Tensor,
    samples: int,
    seed: int,
    dense: bool,
    save_every: int,
 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    final_heat: list[torch.Tensor] = []
    trails: list[torch.Tensor] = []
    heat_trails: list[torch.Tensor] = []
    saved_times: tuple[int, ...] | None = None
    for sample_index in range(samples):
        sample_seed = int(seed + 104729 * sample_index)
        initial_rng = torch.Generator(device="cpu").manual_seed(sample_seed)
        initial_noise = torch.randn(condition.known_state.shape, generator=initial_rng).to(
            device=condition.known_state.device,
            dtype=condition.known_state.dtype,
        )
        reverse_rng = torch.Generator(device=condition.known_state.device).manual_seed(sample_seed ^ 0x33B1)
        trail = model.sample(
            condition,
            provider,
            dense=dense,
            initial_noise=initial_noise,
            generator=reverse_rng,
            save_every=save_every,
        )
        saved_times = trail.timesteps if saved_times is None else saved_times
        if trail.timesteps != saved_times:
            raise RuntimeError("Diffusion sample trails do not share a common timestep grid.")
        final_heat.append(heat_from_logits(trail.final_state, total_heat, condition.design_mask).squeeze(-1))
        trails.append(torch.stack(trail.states, dim=0))
        heat_trails.append(
            torch.stack(
                [
                    heat_from_logits(value, total_heat, condition.design_mask).squeeze(-1)
                    for value in trail.states
                ],
                dim=0,
            )
        )
    assert saved_times is not None
    return torch.stack(final_heat), torch.stack(trails), torch.stack(heat_trails), saved_times


@torch.no_grad()
def sample_matched_heat_arms(
    graph_model: FrozenPacketDiffusion,
    dense_model: FrozenPacketDiffusion,
    tasks: Sequence[ThermalHeatTask],
    *,
    scaler: ThermalHeatFeatureScaler,
    provider_factory: Callable[[ThermalProviderKnownInputs], LinksForCandidate],
    samples_per_case: int = 8,
    seed: int = 17021,
    save_every: int = 4,
    device: torch.device | str = "cpu",
) -> ThermalInverseSamples:
    """Generate matched held-out allocations and keep every reverse trail."""

    if not 8 <= len(tasks) <= 12:
        raise ValueError("Nontrivial inverse evaluation requires 8..12 fixed cases.")
    if any(task.split != "test" or task.module_valid.sum() < 2 for task in tasks):
        raise ValueError("Inverse samples require held-out M>=2 test cases.")
    if samples_per_case != 8:
        raise ValueError("The bounded Thermal review uses exactly eight samples per condition.")
    target_device = torch.device(device)
    batch = collate_thermal_heat_tasks(tasks, scaler=scaler, device=target_device)
    provider = provider_factory(provider_known_inputs(batch))
    graph_samples, graph_trails, graph_heat_trails, graph_times = _sample_arm(
        graph_model,
        batch.condition,
        provider,
        total_heat=batch.total_heat,
        samples=samples_per_case,
        seed=seed,
        dense=False,
        save_every=save_every,
    )
    dense_samples, dense_trails, dense_heat_trails, dense_times = _sample_arm(
        dense_model,
        batch.condition,
        provider,
        total_heat=batch.total_heat,
        samples=samples_per_case,
        seed=seed,
        dense=True,
        save_every=save_every,
    )
    if graph_times != dense_times:
        raise RuntimeError("Matched inverse arms recorded different denoising timesteps.")
    return ThermalInverseSamples(
        case_ids=batch.case_ids,
        total_heat=batch.total_heat.detach().cpu(),
        reference_heat=batch.clean_heat.detach().cpu(),
        graph_heat=graph_samples.detach().cpu(),
        dense_heat=dense_samples.detach().cpu(),
        graph_heat_trails=graph_heat_trails.detach().cpu(),
        dense_heat_trails=dense_heat_trails.detach().cpu(),
        graph_logit_trails=graph_trails.detach().cpu(),
        dense_logit_trails=dense_trails.detach().cpu(),
        trail_timesteps=graph_times,
    )


__all__ = [
    "MatchedDiffusionTrainingResult",
    "ThermalCandidatePacketProvider",
    "ThermalFrontierEvidence",
    "ThermalInverseSamples",
    "ThermalProviderKnownInputs",
    "packet_links_from_interfaces",
    "provider_known_inputs",
    "sample_matched_heat_arms",
    "train_matched_heat_diffusion",
]

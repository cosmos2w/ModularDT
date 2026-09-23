"""Focused algebra and integration contracts for Run 1500."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.interface_fields import (
    InterfaceFieldCore,
    MassCompetitiveGroupControlPairwiseField,
    MassCompetitiveGroupPlan,
    MassCompetitiveGroupRouter,
    mass_competition,
)
from honf_forward_core.interface_fields.types import EncodedInterfaceCase
from honf_runtime.config_loader import load_config_bundle
from tools.diagnostics.mass_competitive_evidence import (
    canonicalize_case,
    summarize_population,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = (
    PROJECT_ROOT
    / "src/config_core/forward/mass_competitive_group_control_honf_context.json"
)


def _small_payload(architecture: str = "mass_competitive_group_control_honf") -> dict[str, object]:
    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))["model"]["core_honf"]
    payload = copy.deepcopy(payload)
    payload.update(
        {
            "forward_architecture": architecture,
            "field_dim": 5,
            "hidden_dim": 16,
            "domain_length_x": 12.0,
            "domain_length_y": 6.0,
            "module_radius": 0.45,
            "query_fourier_frequencies": 2,
            "position_fourier_frequencies": 2,
        }
    )
    payload["interface_model"].update(
        {
            "message_hidden_dim": 8,
            "attention_heads": 2,
            "relative_fourier_frequencies": 2,
            "receiver_chunk_size": 3,
            "activation_checkpointing": False,
        }
    )
    if architecture == "group_control_pairwise_honf":
        payload["interface_model"]["group_count"] = 6
    return payload


def _encoded(seed: int = 1500) -> tuple[EncodedInterfaceCase, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    batch, modules, environments, hidden = 2, 5, 13, 16
    extent = torch.tensor([12.0, 6.0])
    module_present = torch.tensor(
        [[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]]
    )
    encoded = EncodedInterfaceCase(
        module_tokens=torch.randn(batch, modules, hidden, generator=generator),
        env_tokens=torch.randn(batch, environments, hidden, generator=generator),
        global_token=torch.randn(batch, hidden, generator=generator),
        module_centers=torch.rand(batch, modules, 2, generator=generator) * extent,
        env_coords=torch.rand(batch, environments, 2, generator=generator) * extent,
        module_present=module_present,
        module_features=None,
        env_features=None,
        env_weights=torch.rand(batch, environments, generator=generator) + 0.2,
        coordinate_scale=extent,
    )
    return (
        encoded,
        torch.randn(batch, modules, hidden, generator=generator),
        torch.randn(batch, environments, hidden, generator=generator),
    )


def _batch(seed: int = 1500, queries: int = 7) -> BatchData:
    generator = torch.Generator().manual_seed(seed)
    extent = torch.tensor([12.0, 6.0])
    return BatchData(
        module_centers=torch.rand(2, 5, 2, generator=generator) * extent,
        module_present=torch.tensor(
            [[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]]
        ),
        module_features=torch.randn(2, 5, 3, generator=generator),
        global_context=torch.randn(2, 4, generator=generator),
        query_xy=torch.rand(2, queries, 2, generator=generator) * extent,
        query_time=None,
        target_field=torch.randn(2, queries, 5, generator=generator),
        case_name="run1500-mass-competitive-test",
        metadata={},
        env_coords=torch.rand(2, 13, 2, generator=generator) * extent,
        env_features=torch.randn(2, 13, 2, generator=generator),
        env_weights=torch.rand(2, 13, generator=generator) + 0.2,
    )


def test_fixed_mass_competition_is_simplex_sparse_and_permutation_equivariant() -> None:
    pi = torch.tensor(
        [
            [0.52, 0.18, 0.12, 0.07, 0.04, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.001],
            [0.30, 0.25, 0.20, 0.10, 0.06, 0.04, 0.02, 0.01, 0.008, 0.006, 0.004, 0.002],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    gamma = mass_competition(pi)
    torch.testing.assert_close(gamma.sum(dim=-1), torch.ones(2, dtype=torch.float64))
    assert torch.all(gamma >= 0.0)
    assert torch.all((gamma > 0.0).sum(dim=-1).ge(1))
    assert torch.all((gamma > 0.0).sum(dim=-1).lt(12))
    permutation = torch.tensor([7, 0, 11, 2, 5, 9, 1, 10, 4, 8, 3, 6])
    permuted = mass_competition(pi[:, permutation])
    torch.testing.assert_close(permuted, gamma[:, permutation], rtol=0.0, atol=1.0e-12)
    gamma.square().sum().backward()
    assert pi.grad is not None and torch.isfinite(pi.grad).all()
    assert torch.count_nonzero(pi.grad) > 0


def test_router_final_assignments_mask_zero_gamma_and_reuse_p0_plan() -> None:
    encoded, module_states, environment_states = _encoded()
    router = MassCompetitiveGroupRouter(
        16,
        group_count=12,
        control_dim=16,
        fourier_frequencies=2,
    )
    p0 = router.prepare(encoded, module_states, environment_states)
    assert isinstance(p0.plan, MassCompetitiveGroupPlan)
    proposal_occupied = p0.plan.proposal_occupied
    precompetition_mass = (
        p0.plan.precompetition_module_mass_p0
        + p0.plan.precompetition_environment_mass_p0
    )
    assert torch.all(precompetition_mass.masked_select(~proposal_occupied) == 0.0)
    torch.testing.assert_close(p0.gamma.sum(dim=-1), torch.ones(2), atol=2.0e-5, rtol=0.0)
    torch.testing.assert_close(p0.pi_precompetition.sum(dim=-1), torch.ones(2), atol=2.0e-5, rtol=0.0)
    torch.testing.assert_close(
        p0.kappa,
        p0.gamma.square().sum(dim=-1).reciprocal(),
        atol=2.0e-5,
        rtol=0.0,
    )
    zero = p0.gamma == 0.0
    assert not bool((p0.module_membership * zero[:, None, :]).ne(0.0).any())
    assert not bool((p0.environment_membership * zero[:, None, :]).ne(0.0).any())
    module_rows = p0.module_membership.sum(dim=-1)
    torch.testing.assert_close(
        module_rows[encoded.module_present > 0.5],
        torch.ones_like(module_rows[encoded.module_present > 0.5]),
        atol=2.0e-5,
        rtol=0.0,
    )
    assert torch.all(module_rows[encoded.module_present <= 0.5] == 0.0)
    torch.testing.assert_close(
        p0.environment_membership.sum(dim=-1),
        torch.ones_like(p0.environment_membership[..., 0]),
        atol=2.0e-5,
        rtol=0.0,
    )

    p1 = router.prepare(
        encoded,
        module_states + 0.1,
        environment_states - 0.1,
        plan=p0.plan,
    )
    assert p1.plan is p0.plan
    torch.testing.assert_close(p1.gamma, p0.plan.gamma, rtol=0.0, atol=0.0)
    route = router.route_queries(encoded, p1, encoded.module_centers[:, :3])
    torch.testing.assert_close(
        route.assignment.sum(dim=-1),
        torch.ones_like(route.assignment[..., 0]),
        atol=2.0e-5,
        rtol=0.0,
    )
    assert not bool((route.assignment * zero[:, None, :]).ne(0.0).any())


def test_profile_core_forward_backward_and_full_packed_parity() -> None:
    torch.manual_seed(1500)
    config = UnifiedForwardConfig.from_dict(_small_payload())
    core = InterfaceFieldCore(config).train()
    assert isinstance(core.backend, MassCompetitiveGroupControlPairwiseField)
    batch = _batch()
    encoded = core.encode_case(batch)
    p0 = core.prepare(encoded, encoded.module_tokens, return_routing_maps=True)
    p1 = core.prepare(
        encoded,
        encoded.module_tokens + 0.1,
        phase_shared_state=p0.phase_shared_state,
        return_routing_maps=True,
    )
    assert p1.phase_shared_state is p0.phase_shared_state
    assert isinstance(p0.phase_shared_state, MassCompetitiveGroupPlan)
    assert "mass_competitive_gamma" in p0.interaction_aux
    full = core.read(p1, batch.query_xy, return_routing_maps=True)
    assert torch.isfinite(full.context).all()
    assert "mass_competitive_module_mask_support" in full.interaction_aux
    full.context.square().mean().backward()
    gradient = core.backend.router.group_codes.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0

    core.eval()
    full_state = core.prepare(encoded, encoded.module_tokens)
    full_context = core.read(full_state, batch.query_xy).context
    core.backend.set_execution_mode("packed")
    packed_state = core.prepare(encoded, encoded.module_tokens)
    packed_context = core.read(packed_state, batch.query_xy).context
    torch.testing.assert_close(full_context, packed_context, rtol=3.0e-5, atol=3.0e-6)


def test_mass_evidence_reads_live_pi_gamma_and_support_ledgers() -> None:
    core = InterfaceFieldCore(UnifiedForwardConfig.from_dict(_small_payload())).eval()
    batch = _batch(seed=1502, queries=9)
    encoded = core.encode_case(batch)
    prepared = core.prepare(
        encoded,
        encoded.module_tokens,
        return_routing_maps=True,
    )
    read = core.read(prepared, batch.query_xy, return_routing_maps=True)
    payload = {
        "module_coords": batch.module_centers,
        "environment_coords": batch.env_coords,
        "query_coords": batch.query_xy,
        "module_present": batch.module_present,
        **prepared.interaction_aux,
        **read.interaction_aux,
    }
    record = canonicalize_case(payload, case_index=0, query_count=9, kmax=12)
    row = record["case"]
    assert row["kcase"] == sum(value > 0.0 for value in row["gamma"])
    assert row["kcase"] == row["kplan"]
    assert row["kappa"] == pytest.approx(
        1.0 / sum(value * value for value in row["gamma"]), rel=2.0e-5
    )
    assert record["maps"]["pi"].shape == (12,)
    assert record["maps"]["gamma"].shape == (12,)

    unbatched = dict(payload)
    for key in (
        "mass_competitive_plan_active_mask",
        "mass_competitive_plan_prototype_ids",
        "mass_competitive_plan_packed_valid",
    ):
        value = unbatched[key]
        unbatched[key] = value[0] if value.ndim > 1 else value
    unbatched_record = canonicalize_case(
        unbatched,
        case_index=0,
        query_count=9,
        kmax=12,
    )
    assert unbatched_record["case"]["active_mask"] == row["active_mask"]

    collapsed = summarize_population([row, row], expected_cases=2)
    gate = collapsed["continuation_gate"]
    assert not gate["healthy_nontrivial_hypergraph"]
    assert not gate["case_dependent_kcase"]
    assert gate["decision"] == "stop_no_case_dependent_discrete_k"


@pytest.mark.parametrize(
    "architecture",
    ["group_control_pairwise_honf", "occupancy_adaptive_group_control_honf"],
)
def test_historical_group_control_modes_strict_load_unchanged(architecture: str) -> None:
    config = UnifiedForwardConfig.from_dict(_small_payload(architecture))
    source = InterfaceFieldCore(config).eval()
    batch = _batch(seed=1501, queries=3)
    encoded = source.encode_case(batch)
    source.prepare(encoded, encoded.module_tokens)
    state = copy.deepcopy(source.state_dict())
    restored = InterfaceFieldCore(config).eval()
    restored.load_state_dict(state, strict=True)


def test_run1500_profile_identity_and_no_budget_or_schedule() -> None:
    bundle = load_config_bundle(
        "project://src/config_core/forward/mass_competitive_group_control_honf_context.json"
    )
    profile = bundle.effective
    assert profile["model"]["core_honf"]["forward_architecture"] == (
        "mass_competitive_group_control_honf"
    )
    assert profile["model"]["core_honf"]["interface_model"]["group_count"] == 12
    assert profile["model"]["core_honf"]["interface_model"]["group_control_dim"] == 16
    assert profile["training"]["seed"] == 0
    assert profile["training"]["learning_rate"] == 3.0e-4
    assert profile["training"]["port_curriculum"]["mode"] == "predicted"
    assert profile["training"]["port_curriculum"]["schedule"] == "none"
    assert profile["loss"]["case_group_budget_weight"] == 0.0
    assert profile["loss"]["organizer_regularization"]["enabled"] is False
    assert profile["run"] == {
        "id": "1500",
        "name": "mass_competitive_adaptive_hypergraph",
        "output_root": "project://Trained_Results",
    }

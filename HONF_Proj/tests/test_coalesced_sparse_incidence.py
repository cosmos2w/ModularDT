"""Physical integration checks for the Run 1503-v2 coalesced reader."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.interface_fields import InterfaceFieldCore
from honf_runtime.config_loader import load_config_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARENT_PROFILE = (
    PROJECT_ROOT
    / "src/config_core/forward/sparse_incidence_group_control_honf_context.json"
)


def _payload(architecture: str = "sparse_incidence_group_control_honf") -> dict[str, object]:
    payload = json.loads(PARENT_PROFILE.read_text(encoding="utf-8"))["model"][
        "core_honf"
    ]
    payload = copy.deepcopy(payload)
    payload.update(
        {
            "field_dim": 5,
            "hidden_dim": 16,
            "domain_length_x": 12.0,
            "domain_length_y": 6.0,
            "module_radius": 0.45,
            "query_fourier_frequencies": 2,
            "position_fourier_frequencies": 2,
            "forward_architecture": architecture,
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
    if architecture in {
        "sparse_incidence_group_control_honf",
        "coalesced_sparse_incidence_honf",
    }:
        payload["interface_model"]["environment_refinement_normalizer"] = "sparsemax"
    if architecture == "coalesced_sparse_incidence_honf":
        payload["interface_model"].update(
            {
                "fusion_max_iterations": 64,
                "fusion_eta_final": 0.5,
                "fusion_ramp_epochs": 150,
            }
        )
    return payload


def _batch(seed: int = 1503001, queries: int = 7) -> BatchData:
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
        case_name="run1503-v2-coalesced-test",
        metadata={},
        env_coords=torch.rand(2, 13, 2, generator=generator) * extent,
        env_features=torch.randn(2, 13, 2, generator=generator),
        env_weights=torch.rand(2, 13, generator=generator) + 0.2,
    )


def _paired_cores(seed: int = 1503002) -> tuple[InterfaceFieldCore, InterfaceFieldCore]:
    torch.manual_seed(seed)
    parent = InterfaceFieldCore(
        UnifiedForwardConfig.from_dict(_payload())
    )
    candidate = InterfaceFieldCore(
        UnifiedForwardConfig.from_dict(
            _payload("coalesced_sparse_incidence_honf")
        )
    )
    # Materialize every LazyLinear before loading so parent and candidate get
    # the same concrete tensors instead of two independently initialized lazy
    # parameters that were both uninitialized at load time.
    parent.eval()
    candidate.eval()
    candidate.set_training_progress(epoch=0, total_epochs=500)
    warmup = _batch(seed=seed + 1, queries=2)
    with torch.no_grad():
        for core in (parent, candidate):
            encoded = core.encode_case(warmup)
            prepared = core.prepare(encoded, encoded.module_tokens)
            core.decode_queries(prepared, warmup.query_xy)
    candidate.load_state_dict(parent.state_dict(), strict=True)
    return parent, candidate


def test_profile_round_trip_and_epoch_strength_restore() -> None:
    bundle = load_config_bundle(
        "project://src/config_core/forward/coalesced_sparse_incidence_honf_context.json"
    )
    profile = bundle.effective
    assert profile["run"]["id"] == "1504"
    assert profile["run"]["name"] == "1503_v2_reversible_edge_coalescence"
    assert profile["model"]["core_honf"]["forward_architecture"] == (
        "coalesced_sparse_incidence_honf"
    )
    config = UnifiedForwardConfig.from_dict(profile["model"]["core_honf"])
    assert UnifiedForwardConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()

    _, candidate = _paired_cores()
    candidate.set_training_progress(epoch=50, total_epochs=500)
    assert candidate.selection_state() == {"epoch": 50, "total_epochs": 500}
    assert candidate.backend._fusion_strength() == 1.0 / 6.0
    candidate.set_training_progress(epoch=150, total_epochs=500)
    assert candidate.backend._fusion_strength() == 0.5


def test_epoch_zero_full_field_forward_is_bitwise_parent_identity() -> None:
    parent, candidate = _paired_cores(seed=1503003)
    parent.eval()
    candidate.eval()
    candidate.set_training_progress(epoch=0, total_epochs=500)
    batch = _batch(seed=1503004)
    parent_encoded = parent.encode_case(batch)
    candidate_encoded = candidate.encode_case(batch)
    with torch.no_grad():
        parent_prepared = parent.prepare(
            parent_encoded, parent_encoded.module_tokens
        )
        candidate_prepared = candidate.prepare(
            candidate_encoded, candidate_encoded.module_tokens
        )
        parent_output = parent.decode_queries(parent_prepared, batch.query_xy)
        candidate_output = candidate.decode_queries(
            candidate_prepared, batch.query_xy
        )
    torch.testing.assert_close(
        candidate_output["pred_field"],
        parent_output["pred_field"],
        atol=0.0,
        rtol=0.0,
    )


def test_nonzero_fusion_prepares_conserved_moments_and_optimizer_step() -> None:
    torch.manual_seed(1503005)
    core = InterfaceFieldCore(
        UnifiedForwardConfig.from_dict(
            _payload("coalesced_sparse_incidence_honf")
        )
    ).train()
    core.set_training_progress(epoch=50, total_epochs=500)
    batch = _batch(seed=1503006, queries=5)
    encoded = core.encode_case(batch)
    prepared = core.prepare(
        encoded, encoded.module_tokens, return_routing_maps=True
    )
    state = prepared.backend_state
    controls = state["group_control_state"]
    plan = state["coalescence_plan"]
    assert plan.fusion_strength[0].item() == pytest.approx(1.0 / 6.0)

    # Verify source-resolved B = sum_k A^0[:,k] h_k using an independent
    # constituent loop, including rows with unequal incidence.
    provisional_a = state["coalescence_provisional_module_incidence"]
    provisional_h = state["coalescence_provisional_group_control"]
    expected = provisional_a.new_zeros(
        provisional_a.shape[0],
        provisional_a.shape[1],
        provisional_a.shape[2],
        provisional_h.shape[-1],
    )
    for batch_index in range(provisional_a.shape[0]):
        for group in range(provisional_a.shape[-1]):
            packed = int(plan.class_index[batch_index, group].item())
            if packed >= 0:
                expected[batch_index, :, packed] += (
                    provisional_a[batch_index, :, group, None]
                    * provisional_h[batch_index, group]
                )
    torch.testing.assert_close(controls.module_source_moment, expected)
    assert state["module_control_bank"].shape[1] == int(plan.multiplicity.shape[-1])
    assert state["environment_head_source_control"].shape[1] == int(
        plan.multiplicity.shape[-1]
    )

    before = next(core.parameters()).detach().clone()
    optimizer = torch.optim.Adam(core.parameters(), lr=1.0e-3)
    optimizer.zero_grad(set_to_none=True)
    encoded = core.encode_case(batch)
    prepared = core.prepare(encoded, encoded.module_tokens)
    output = core.decode_queries(prepared, batch.query_xy)
    loss = torch.nn.functional.mse_loss(output["pred_field"], batch.target_field)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in core.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    optimizer.step()
    after = next(core.parameters()).detach()
    assert not torch.equal(before, after)


def test_coalescence_query_maps_merge_across_unequal_receiver_chunks() -> None:
    torch.manual_seed(1503007)
    core = InterfaceFieldCore(
        UnifiedForwardConfig.from_dict(
            _payload("coalesced_sparse_incidence_honf")
        )
    ).eval()
    core.set_training_progress(epoch=50, total_epochs=500)
    batch = _batch(seed=1503008, queries=7)
    encoded = core.encode_case(batch)
    prepared = core.prepare(encoded, encoded.module_tokens, return_routing_maps=True)
    chunked = core.read(
        prepared,
        batch.query_xy,
        receiver_chunk_size=3,
        return_routing_maps=True,
    )
    unchunked = core.read(
        prepared,
        batch.query_xy,
        receiver_chunk_size=7,
        return_routing_maps=True,
    )
    torch.testing.assert_close(chunked.context, unchunked.context, atol=5e-5, rtol=5e-5)
    for key in (
        "coalescence_query_mass",
        "coalescence_query_density",
        "coalescence_query_logits",
        "coalescence_virtual_constituent_support",
    ):
        assert chunked.interaction_aux[key].shape[1] == batch.query_xy.shape[1]
        torch.testing.assert_close(
            chunked.interaction_aux[key],
            unchunked.interaction_aux[key],
            atol=5e-5,
            rtol=5e-5,
        )
    assert torch.equal(
        prepared.interaction_aux["coalescence_provisional_phase_occupied"],
        prepared.interaction_aux["sparse_incidence_phase_occupied"],
    )

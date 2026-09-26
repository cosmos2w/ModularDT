"""Full-access refit boundary and native-dimensional fine-reader checks."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from honf_forward_core.config import BatchData, InterfaceFieldConfig, UnifiedForwardConfig
from honf_forward_core.interface_fields.adaptive_cover_field import AdaptiveCoverPairwiseField
from honf_forward_core.interface_fields.adaptive_interaction_cover import AdaptiveCoverPlan
from honf_forward_core.interface_fields.checkpoint_warm_start import (
    warm_start_three_term_full_access,
)
from honf_forward_core.interface_fields.core import InterfaceFieldCore
from honf_forward_core.interface_fields.dense_pairwise import DensePairwiseField
from honf_forward_core.interface_fields.three_term_context import ThreeTermInterfaceContext


def _config(architecture: str, spatial_dim: int) -> UnifiedForwardConfig:
    return UnifiedForwardConfig(
        forward_architecture=architecture,
        field_dim=3,
        spatial_dim=spatial_dim,
        coordinate_scale=[12.0, 6.0] if spatial_dim == 2 else [50.0, 38.0, 6.25],
        hidden_dim=16,
        dropout=0.0,
        geometry_mode="nonperiodic",
        boundary_feature_mode="none",
        interface_model=InterfaceFieldConfig(
            message_hidden_dim=8,
            attention_heads=2,
            relative_fourier_frequencies=2,
            receiver_chunk_size=3,
            group_count=12,
            group_control_dim=16,
            source_normalizer="entmax15",
            environment_refinement_normalizer=(
                "sparsemax" if architecture == "source_conditioned_pairwise_honf" else "entmax15"
            ),
        ),
    )


def _batch(spatial_dim: int, device: torch.device) -> BatchData:
    generator = torch.Generator().manual_seed(243)

    def random(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=generator).to(device)

    batch = BatchData(
        module_centers=random(1, 12, spatial_dim),
        module_present=torch.ones(1, 12, device=device),
        module_features=random(1, 12, 3),
        global_context=random(1, 4),
        query_xy=random(1, 5, spatial_dim),
        query_time=None,
        target_field=random(1, 5, 3),
        case_name="full-access-refit-test",
        metadata={},
        env_coords=random(1, 9, spatial_dim),
        env_features=random(1, 9, 2),
        env_weights=random(1, 9) + 0.1,
    )
    return batch


def _materialize(core: InterfaceFieldCore, batch: BatchData) -> None:
    encoded = core.encode_case(batch)
    prepared = core.prepare(encoded, encoded.module_tokens)
    core.decode_queries(prepared, batch.query_xy)


@pytest.mark.parametrize("spatial_dim", [2, 3])
def test_full_access_refit_reads_all_sources_and_has_design_gradients(spatial_dim: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = InterfaceFieldCore(_config("three_term_full_access_honf", spatial_dim)).to(device)
    batch = _batch(spatial_dim, device)
    assert isinstance(core.backend, DensePairwiseField)
    assert isinstance(core.common, ThreeTermInterfaceContext)
    for chunk_size in (2, 5):
        design = batch.module_centers.clone().requires_grad_(True)
        queries = batch.query_xy.clone().requires_grad_(True)
        current = replace(batch, module_centers=design, query_xy=queries)
        encoded = core.encode_case(current)
        prepared = core.prepare(encoded, encoded.module_tokens)
        prediction = core.decode_queries(
            prepared,
            queries,
            receiver_chunk_size=chunk_size,
        )["pred_field"]
        assert prediction.shape == (1, 5, 3)
        assert torch.isfinite(prediction).all()
        design_grad, query_grad = torch.autograd.grad(prediction.square().mean(), (design, queries))
        assert torch.isfinite(design_grad).all()
        assert torch.isfinite(query_grad).all()
        assert float(design_grad.abs().sum()) > 0.0
        assert float(query_grad.abs().sum()) > 0.0


def test_source_conditioned_warm_start_is_strict_refit_and_discards_organizer() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = _batch(2, device)
    source = InterfaceFieldCore(_config("source_conditioned_pairwise_honf", 2)).to(device)
    target = InterfaceFieldCore(_config("three_term_full_access_honf", 2)).to(device)
    _materialize(source, batch)
    _materialize(target, batch)
    converted, inventory = warm_start_three_term_full_access(
        source.state_dict(), target.state_dict(),
        source_architecture="source_conditioned_pairwise_honf",
    )
    target.load_state_dict(converted, strict=True)
    assert inventory["initialized"] == ()
    assert inventory["prediction_identity_claim"] is False
    assert any(name.startswith("backend.router.") for name in inventory["discarded_source"])
    assert set(inventory["copied"]) == set(target.state_dict())
    missing_name = inventory["copied"][0]
    with pytest.raises(ValueError, match="missing or unmaterialized"):
        warm_start_three_term_full_access(
            {key: value for key, value in source.state_dict().items() if key != missing_name},
            target.state_dict(),
            source_architecture="source_conditioned_pairwise_honf",
        )


@pytest.mark.parametrize("spatial_dim", [2, 3])
def test_adaptive_cover_full_access_and_packed_parent_identity(spatial_dim: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = _batch(spatial_dim, device)
    # Use two cases to exercise per-case scale and packed batch indexing.
    batch = replace(
        batch,
        module_centers=torch.cat((batch.module_centers, batch.module_centers + 0.05)),
        module_present=batch.module_present.expand(2, -1).clone(),
        module_features=batch.module_features.expand(2, -1, -1).clone(),
        global_context=batch.global_context.expand(2, -1).clone(),
        query_xy=batch.query_xy.expand(2, -1, -1).clone(),
        target_field=batch.target_field.expand(2, -1, -1).clone(),
        env_coords=batch.env_coords.expand(2, -1, -1).clone(),
        env_features=batch.env_features.expand(2, -1, -1).clone(),
        env_weights=batch.env_weights.expand(2, -1).clone(),
    )
    incumbent = InterfaceFieldCore(_config("three_term_full_access_honf", spatial_dim)).to(device)
    adaptive = InterfaceFieldCore(_config("adaptive_interaction_cover_honf", spatial_dim)).to(device)
    assert isinstance(adaptive.backend, AdaptiveCoverPairwiseField)
    _materialize(incumbent, batch)
    _materialize(adaptive, batch)
    adaptive.load_state_dict(incumbent.state_dict(), strict=True)

    def predict(core: InterfaceFieldCore, *, packed: bool) -> tuple[torch.Tensor, torch.Tensor]:
        centers = batch.module_centers.clone().requires_grad_(True)
        current = replace(batch, module_centers=centers)
        encoded = core.encode_case(current)
        prepared = core.prepare(encoded, encoded.module_tokens)
        if packed:
            core.backend.set_cover_mode("external")
        result = core.decode_queries(prepared, batch.query_xy, receiver_chunk_size=3)["pred_field"]
        grad = torch.autograd.grad(result.square().sum(), centers)[0]
        return result, grad

    reference, reference_gradient = predict(incumbent, packed=False)
    direct, direct_gradient = predict(adaptive, packed=False)
    torch.testing.assert_close(direct, reference, rtol=0.0, atol=0.0)
    torch.testing.assert_close(direct_gradient, reference_gradient, rtol=0.0, atol=0.0)
    packed, packed_gradient = predict(adaptive, packed=True)
    torch.testing.assert_close(packed, reference, rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(packed_gradient, reference_gradient, rtol=2.0e-4, atol=2.0e-5)

    encoded = adaptive.encode_case(batch)
    prepared = adaptive.prepare(encoded, encoded.module_tokens)
    preparation = prepared.interaction_aux
    assert int(preparation["cover_prepare_mm_executed_rows"]) == 2 * 12 * 12
    assert int(preparation["cover_prepare_me_executed_rows"]) == 2 * 12 * 9
    assert int(preparation["cover_prepare_em_executed_rows"]) == 2 * 9 * 12
    assert int(preparation["cover_prepare_padded_or_self_rows"]) == 2 * 12
    read = adaptive.read(prepared, batch.query_xy, receiver_chunk_size=2)
    ledger = read.interaction_aux
    assert int(ledger["cover_qm_unique_rows"]) == 2 * 5 * 12
    assert int(ledger["cover_qe_unique_rows"]) == 2 * 5 * 9
    assert int(ledger["cover_qm_rectangular_rows"]) == 2 * 5 * 12
    assert int(ledger["cover_qe_rectangular_rows"]) == 2 * 5 * 9
    assert int(ledger["cover_query_count"]) == 2 * 5
    assert int(ledger["cover_query_degree_sum"]) == 2 * 5
    assert int(ledger["cover_query_degree_max"]) == 1
    torch.testing.assert_close(ledger["cover_k_case"], torch.ones(2, device=device, dtype=torch.long))


def test_sparse_cover_matches_global_rectangular_weighted_reference_and_gradients() -> None:
    """A selective source union must preserve one global environmental softmax."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = InterfaceFieldCore(_config("adaptive_interaction_cover_honf", 2)).to(device)
    batch = _batch(2, device)
    centers = batch.module_centers.clone().requires_grad_(True)
    receivers = torch.tensor([[[0.01, 0.01], [0.99, 0.99]]], device=device, requires_grad=True)
    encoded = core.encode_case(replace(batch, module_centers=centers))
    prepared = core.prepare(encoded, encoded.module_tokens)
    state = prepared.backend_state
    backend = core.backend
    assert isinstance(backend, AdaptiveCoverPairwiseField)
    plan: AdaptiveCoverPlan = state["cover_plans"][0]
    plan = plan.with_split(0, 1.0)
    module_membership = plan.module_membership.clone()
    environment_membership = plan.environment_membership.clone()
    module_membership[1:] = 0.0
    environment_membership[1:] = 0.0
    module_membership[1, 0] = module_membership[2, 1] = 1.0
    environment_membership[1, :4] = 1.0
    environment_membership[2, 4:] = 1.0
    plan = replace(
        plan, module_membership=module_membership,
        environment_membership=environment_membership,
    )
    state = backend.with_external_plans(state, (plan,))
    receiver_features = core._receiver_features(prepared, receivers)
    backend.set_cover_mode("external")
    selected, aux = backend.read(state, encoded, receivers, receiver_features)
    assert int(aux["cover_qm_unique_rows"]) < 24
    assert int(aux["cover_qe_unique_rows"]) < 18

    access = plan.access(receivers[0])
    scale = encoded.coordinate_scale
    module_relative = (receivers[:, :, None] - encoded.module_centers[:, None]) / scale
    module_input = torch.cat((
        state["module_tokens"][:, None].expand(-1, 2, -1, -1),
        backend.relative_fourier(module_relative),
        encoded.global_token[:, None, None].expand(-1, 2, 12, -1),
    ), dim=-1)
    module_messages = backend.query_module_message(module_input)
    module_weight = access.module_source[None, :, :, None] * encoded.module_present[:, None, :, None]
    module_context = backend.query_module_output(
        (module_messages * module_weight).sum(dim=2)
        / (1.0 + encoded.module_present.sum(dim=1)[:, None, None])
    )

    key, value = backend.env_attention.project_source(state["env_tokens"])
    query = backend.env_attention.project_query(backend.env_query(receiver_features))
    env_relative = (receivers[:, :, None] - encoded.env_coords[:, None]) / scale
    bias = backend.env_geometry_bias(backend.relative_fourier(env_relative)).permute(0, 3, 1, 2)
    logits = (
        (query[:, :, :, None] * key[:, :, None]).sum(dim=-1)
        / math.sqrt(float(backend.env_attention.head_dim))
        + bias
    )
    prior = access.environment_source[None] * encoded.env_weights[:, None]
    logits = logits + prior[:, None].clamp_min(torch.finfo(prior.dtype).tiny).log()
    logits = logits.masked_fill(prior[:, None] == 0.0, -torch.inf)
    attention = torch.softmax(logits, dim=-1)
    env_rows = (attention[..., None] * value[:, :, None]).sum(dim=-2)
    env_context = backend.env_attention.output(
        env_rows.transpose(1, 2).reshape(1, 2, backend.hidden_dim)
    )
    rectangular = module_context + env_context
    torch.testing.assert_close(selected, rectangular, rtol=2.0e-5, atol=2.0e-6)
    selected_grad = torch.autograd.grad(selected.square().sum(), (centers, receivers), retain_graph=True)
    reference_grad = torch.autograd.grad(rectangular.square().sum(), (centers, receivers))
    for observed, reference in zip(selected_grad, reference_grad, strict=True):
        torch.testing.assert_close(observed, reference, rtol=3.0e-4, atol=3.0e-5)


def test_cover_row_ledger_distinguishes_padded_dense_work_from_packed_work() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = InterfaceFieldCore(_config("adaptive_interaction_cover_honf", 2)).to(device)
    batch = _batch(2, device)
    present = batch.module_present.clone()
    present[0, -1] = 0.0
    batch = replace(batch, module_present=present)
    encoded = core.encode_case(batch)
    prepared = core.prepare(encoded, encoded.module_tokens)
    rectangular = core.read(prepared, batch.query_xy, receiver_chunk_size=2)
    dense_ledger = rectangular.interaction_aux
    assert int(dense_ledger["cover_qm_unique_rows"]) == 5 * 11
    assert int(dense_ledger["cover_qm_executed_rows"]) == 5 * 12
    assert int(dense_ledger["cover_qm_padded_rows"]) == 5
    assert int(dense_ledger["cover_qm_rectangular_rows"]) == 5 * 12

    assert isinstance(core.backend, AdaptiveCoverPairwiseField)
    core.backend.set_cover_mode("external")
    packed = core.read(prepared, batch.query_xy, receiver_chunk_size=2)
    packed_ledger = packed.interaction_aux
    assert int(packed_ledger["cover_qm_unique_rows"]) == 5 * 11
    assert int(packed_ledger["cover_qm_executed_rows"]) == 5 * 11
    assert int(packed_ledger["cover_qm_padded_rows"]) == 0
    assert int(packed_ledger["cover_query_count"]) == 5
    assert int(packed_ledger["cover_query_degree_sum"]) == 5
    assert int(packed_ledger["cover_query_degree_max"]) == 1
    torch.testing.assert_close(packed.context, rectangular.context, rtol=2.0e-5, atol=2.0e-6)

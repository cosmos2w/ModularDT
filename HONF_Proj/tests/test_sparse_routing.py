from __future__ import annotations

import pytest
import torch

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.model import HONFNeuralField
from honf_forward_core.routing import entmax15, locality_bias, normalize_assignment


def test_entmax15_has_exact_zeros_nonnegative_unit_mass() -> None:
    logits = torch.tensor([[4.0, 1.0, -3.0], [2.0, 2.0, -6.0]])
    probabilities = entmax15(logits)

    assert torch.count_nonzero(probabilities == 0) >= 2
    assert torch.all(probabilities >= 0)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2), rtol=0.0, atol=1.0e-6)
    assert probabilities[1, 0] > 0 and probabilities[1, 1] > 0


def test_entmax15_masking_and_empty_rows_are_exact() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0], [3.0, 2.0, 1.0]])
    mask = torch.tensor([[True, False, True], [False, False, False]])
    probabilities = entmax15(logits, mask=mask)

    assert probabilities[0, 1] == 0
    assert probabilities[0].sum() == 1
    assert torch.count_nonzero(probabilities[1]) == 0


def test_bounded_gaussian_locality_is_finite_smooth_inside_cap_and_bounded() -> None:
    radius_square = torch.tensor([0.0, 0.25, 1.0, 4.0, 9.0, 16.0], requires_grad=True)
    bias = locality_bias(
        radius_square,
        mode="bounded_gaussian",
        strength=1.0,
        radius_cap=3.0,
    )

    torch.testing.assert_close(
        bias,
        torch.tensor([0.0, -0.125, -0.5, -2.0, -4.5, -4.5]),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.isfinite(bias).all()
    assert torch.all(torch.exp(bias) > 0)
    bias[:4].sum().backward()
    torch.testing.assert_close(radius_square.grad[:4], torch.full((4,), -0.5))


def test_compact_locality_formula_remains_available() -> None:
    radius_square = torch.tensor([0.0, 0.25, 1.0, 4.0])
    expected = torch.log(torch.relu(1.0 - radius_square).square() + 1.0e-6)

    actual = locality_bias(
        radius_square,
        mode="compact_kernel",
        strength=1.0,
        radius_cap=3.0,
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_bounded_gaussian_does_not_preempt_entmax_at_unit_radius() -> None:
    radius_square = torch.tensor([[0.0, 1.21]])
    gaussian_routes = entmax15(
        locality_bias(radius_square, mode="bounded_gaussian", strength=1.0, radius_cap=3.0)
    )
    compact_routes = entmax15(
        locality_bias(radius_square, mode="compact_kernel", strength=1.0, radius_cap=3.0)
    )

    assert gaussian_routes[0, 1] > 0
    assert compact_routes[0, 1] == 0


@pytest.mark.parametrize("mode", ["softmax", "entmax15"])
def test_assignment_normalizers_have_finite_gradients(mode: str) -> None:
    logits = torch.tensor([[1.7, 0.9, -0.2, -1.5]], requires_grad=True)
    probabilities = normalize_assignment(logits, mode=mode)
    loss = (probabilities * torch.tensor([[0.2, -0.3, 0.7, 0.1]])).sum()
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_entmax15_cpu_and_cuda_match() -> None:
    logits = torch.tensor([[2.1, 1.8, 0.2, -3.0], [0.3, -0.4, 1.2, 0.8]])
    cpu = entmax15(logits)
    cuda = entmax15(logits.cuda()).cpu()

    torch.testing.assert_close(cpu, cuda, rtol=1.0e-6, atol=1.0e-6)


def _config(*, periodic: bool = False, locality_mode: str = "bounded_gaussian") -> UnifiedForwardConfig:
    return UnifiedForwardConfig(
        field_dim=2,
        domain_length_x=6.0,
        domain_length_y=3.0,
        coordinate_scale=[6.0, 3.0],
        periodic_axes=[0] if periodic else [],
        num_env_tokens_x=6,
        num_env_tokens_y=3,
        num_hyperedges=3,
        organizer_mode="exchangeable_slots",
        edge_capacity=6,
        initial_active_edges=6,
        minimum_active_edges=1,
        edge_selection_mode="all",
        module_assignment_normalizer="entmax15",
        environment_assignment_normalizer="entmax15",
        query_assignment_normalizer="entmax15",
        entmax_alpha=1.5,
        environment_locality_mode=locality_mode,
        environment_locality_strength=1.0,
        locality_radius_cap=3.0,
        minimum_region_scale=0.08,
        hidden_dim=20,
        dropout=0.0,
        decoder_mode="enhanced_honf_pairwise",
        pairwise_kernel_hidden_dim=20,
        pairwise_kernel_num_layers=2,
        mechanism_state_mode="descriptor_first",
        field_assembly_mode="edge_additive",
        routing_execution="dense",
    )


def _batch() -> BatchData:
    generator = torch.Generator().manual_seed(131)
    return BatchData(
        module_centers=torch.rand(2, 5, 2, generator=generator) * torch.tensor([6.0, 3.0]),
        module_present=torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0]]),
        module_features=torch.randn(2, 5, 4, generator=generator),
        global_context=torch.randn(2, 3, generator=generator),
        query_xy=torch.rand(2, 17, 2, generator=generator) * torch.tensor([6.0, 3.0]),
        query_time=None,
        target_field=None,
        case_name="sparse-routing-test",
        metadata={},
    )


def test_entmax_model_routes_are_sparse_normalized_and_finite() -> None:
    torch.manual_seed(137)
    model = HONFNeuralField(_config()).train()
    with torch.no_grad():
        model.organizer.exchangeable.module_query.weight.mul_(5.0)
        model.organizer.exchangeable.module_key.weight.mul_(5.0)
    batch = _batch()
    output = model(batch)
    output["pred_field"].square().mean().backward()

    active_modules = batch.module_present > 0
    torch.testing.assert_close(
        output["candidate_A_mh"].sum(dim=-1)[active_modules],
        torch.ones_like(output["candidate_A_mh"].sum(dim=-1)[active_modules]),
    )
    torch.testing.assert_close(
        output["candidate_A_eh"].sum(dim=-1),
        torch.ones_like(output["candidate_A_eh"].sum(dim=-1)),
    )
    assert torch.count_nonzero(output["candidate_A_mh"] == 0) > 0
    assert torch.count_nonzero(output["candidate_A_eh"] == 0) > 0
    assert output["candidate_module_nonzero_fraction"].max() < 1.0
    assert output["candidate_environment_nonzero_fraction"].max() < 1.0
    assert output["query_assignment_nonzero_fraction"] < 1.0
    assert output["routing_execution"] == "dense"
    assert torch.isfinite(output["pred_field"]).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_locality_makes_a_distant_irrelevant_edge_exactly_zero() -> None:
    model = HONFNeuralField(_config()).eval()
    query = torch.tensor([[[0.1, 1.5]]])
    organizer = {
        "hyper_region_coords": torch.tensor([[[0.2, 1.5], [5.0, 1.5]]]),
        "hyper_region_scale": torch.tensor([[[0.8, 0.8], [0.8, 0.8]]]),
    }
    bias = model.decoder._query_locality_bias(query, organizer)
    routes = entmax15(bias)

    assert routes[0, 0, 0] > 0
    assert routes[0, 0, 1] == 0


def test_nearby_edges_can_overlap_under_entmax() -> None:
    routes = entmax15(torch.tensor([[1.5, 1.5, -5.0]]))

    assert routes[0, 0] > 0
    assert routes[0, 1] > 0
    assert routes[0, 2] == 0


def test_query_locality_respects_periodic_axis() -> None:
    model = HONFNeuralField(_config(periodic=True)).eval()
    query = torch.tensor([[[5.95, 1.5]]])
    organizer = {
        "hyper_region_coords": torch.tensor([[[0.05, 1.5], [3.0, 1.5]]]),
        "hyper_region_scale": torch.tensor([[[0.5, 0.8], [0.5, 0.8]]]),
    }
    routes = entmax15(model.decoder._query_locality_bias(query, organizer))

    assert routes[0, 0, 0] > 0
    assert routes[0, 0, 1] == 0

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.model import HONFNeuralField
from honf_forward_core.organizer import deterministic_slot_codes


def _config(*, capacity: int = 6, initial: int = 4, minimum: int = 1) -> UnifiedForwardConfig:
    return UnifiedForwardConfig(
        field_dim=3,
        domain_length_x=6.0,
        domain_length_y=3.0,
        coordinate_scale=[6.0, 3.0],
        periodic_axes=[],
        num_env_tokens_x=4,
        num_env_tokens_y=2,
        num_hyperedges=3,
        organizer_mode="exchangeable_slots",
        edge_capacity=capacity,
        initial_active_edges=initial,
        minimum_active_edges=minimum,
        slot_refinement_steps=2,
        edge_selection_mode="quality_coverage",
        selection_warmup_epochs=3,
        selection_coverage_rate=0.8,
        selection_token_threshold=0.4,
        selection_maximum_redundancy=0.95,
        module_assignment_normalizer="softmax",
        environment_assignment_normalizer="softmax",
        query_assignment_normalizer="softmax",
        hidden_dim=24,
        dropout=0.0,
        decoder_mode="enhanced_honf_pairwise",
        pairwise_kernel_hidden_dim=24,
        pairwise_kernel_num_layers=2,
        mechanism_state_mode="descriptor_first",
        field_assembly_mode="edge_additive",
        routing_execution="dense",
    )


def _batch(module_width: int = 4) -> BatchData:
    generator = torch.Generator().manual_seed(101)
    active_width = min(module_width, 4)
    centers = torch.rand(2, active_width, 2, generator=generator) * torch.tensor([6.0, 3.0])
    features = torch.randn(2, active_width, 5, generator=generator)
    present = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])[:, :active_width]
    if module_width > active_width:
        padding = module_width - active_width
        centers = torch.cat([centers, torch.zeros(2, padding, 2)], dim=1)
        features = torch.cat([features, torch.zeros(2, padding, 5)], dim=1)
        present = torch.cat([present, torch.zeros(2, padding)], dim=1)
    return BatchData(
        module_centers=centers,
        module_present=present,
        module_features=features,
        global_context=torch.randn(2, 4, generator=generator),
        query_xy=torch.rand(2, 11, 2, generator=generator) * torch.tensor([6.0, 3.0]),
        query_time=None,
        target_field=None,
        case_name="exchangeable-test",
        metadata={},
    )


def _model(config: UnifiedForwardConfig | None = None) -> HONFNeuralField:
    torch.manual_seed(103)
    return HONFNeuralField(config or _config())


def test_slot_codes_are_deterministic_zero_mean_and_not_parameters() -> None:
    first = deterministic_slot_codes(6, 24, mode="sinusoidal", device=torch.device("cpu"), dtype=torch.float32)
    second = deterministic_slot_codes(6, 24, mode="sinusoidal", device=torch.device("cpu"), dtype=torch.float32)
    model = _model()

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.mean(dim=0), torch.zeros(24), rtol=0.0, atol=2.0e-7)
    assert not any(isinstance(module, nn.Embedding) for module in model.modules())
    assert not any("slot_code" in name for name, _ in model.named_parameters())


def test_candidate_code_permutation_is_equivariant_and_field_invariant() -> None:
    model = _model().eval()
    batch = _batch()
    permutation = torch.tensor([4, 1, 5, 0, 3, 2])
    with torch.no_grad():
        reference = model.encode_and_organize(batch)
        candidate = model.organizer(
            module_tokens=reference["module_tokens"],
            env_tokens=reference["env_tokens"],
            module_centers=batch.module_centers,
            env_coords=reference["env_coords"],
            module_present=batch.module_present,
            candidate_codes=reference["candidate_slot_codes"][:, permutation],
        )
        candidate["module_features_raw"] = batch.module_features
        reference_field = model.decode_queries(
            batch.query_xy,
            None,
            reference,
            reference["global_token"],
            return_edge_fields=True,
        )
        candidate_field = model.decode_queries(
            batch.query_xy,
            None,
            candidate,
            reference["global_token"],
            return_edge_fields=True,
        )

    for key in ("candidate_hyper_state", "edge_quality", "edge_active_mask", "hyper_state"):
        torch.testing.assert_close(reference[key][:, permutation], candidate[key], rtol=1.0e-6, atol=1.0e-6)
    torch.testing.assert_close(
        reference["candidate_A_mh"][:, :, permutation],
        candidate["candidate_A_mh"],
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        reference["candidate_A_eh"][:, :, permutation],
        candidate["candidate_A_eh"],
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    torch.testing.assert_close(reference_field["pred_field"], candidate_field["pred_field"], rtol=1.0e-6, atol=1.0e-6)
    torch.testing.assert_close(
        reference_field["pred_field_by_edge"][:, :, permutation],
        candidate_field["pred_field_by_edge"],
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_module_permutation_and_padding_width_preserve_field() -> None:
    model = _model().eval()
    batch = _batch()
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = copy.copy(batch)
    permuted.module_centers = batch.module_centers[:, permutation]
    permuted.module_present = batch.module_present[:, permutation]
    permuted.module_features = batch.module_features[:, permutation]
    padded = _batch(module_width=10)
    padded.global_context = batch.global_context
    padded.query_xy = batch.query_xy
    with torch.no_grad():
        reference = model(batch)["pred_field"]
        permuted_field = model(permuted)["pred_field"]
        padded_field = model(padded)["pred_field"]

    torch.testing.assert_close(reference, permuted_field, rtol=1.0e-6, atol=1.0e-6)
    torch.testing.assert_close(reference, padded_field, rtol=1.0e-6, atol=1.0e-6)


def test_capacity_changes_runtime_shapes_not_parameter_shapes() -> None:
    model = _model().eval()
    batch = _batch()
    with torch.no_grad():
        model.set_edge_capacity(6)
        output_6 = model(batch)
        parameter_shapes_6 = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        model.set_edge_capacity(10)
        output_10 = model(batch)
        parameter_shapes_10 = {name: tuple(value.shape) for name, value in model.state_dict().items()}

    assert output_6["candidate_A_mh"].shape[-1] == 6
    assert output_10["candidate_A_mh"].shape[-1] == 10
    assert parameter_shapes_6 == parameter_shapes_10

    other = _model(_config(capacity=10)).eval()
    with torch.no_grad():
        other(batch)
    assert {name: tuple(value.shape) for name, value in other.state_dict().items()} == parameter_shapes_10


def test_selection_bounds_mass_and_inactive_edges() -> None:
    model = _model().eval()
    batch = _batch(module_width=10)
    with torch.no_grad():
        output = model(batch)

    assert torch.all(output["active_edge_count"] >= 1)
    assert torch.all(output["active_edge_count"] <= 6)
    active_modules = batch.module_present > 0
    torch.testing.assert_close(
        output["A_mh"].sum(dim=-1)[active_modules],
        torch.ones_like(output["A_mh"].sum(dim=-1)[active_modules]),
    )
    assert torch.count_nonzero(output["A_mh"].sum(dim=-1)[~active_modules]) == 0
    torch.testing.assert_close(output["A_eh"].sum(dim=-1), torch.ones_like(output["A_eh"].sum(dim=-1)))
    inactive = output["edge_active_mask"] <= 0
    assert torch.count_nonzero(output["A_mh"].transpose(1, 2)[inactive]) == 0
    assert torch.count_nonzero(output["A_eh"].transpose(1, 2)[inactive]) == 0
    assert torch.isfinite(output["edge_quality"]).all()
    assert torch.isfinite(output["pred_field"]).all()
    assert torch.any(output["candidate_A_eh"].std(dim=-1) > 0)


def test_training_warmup_uses_initial_active_count() -> None:
    model = _model().train()
    model.set_training_progress(epoch=0, total_epochs=10)
    output = model(_batch())

    torch.testing.assert_close(output["active_edge_count"], torch.full((2,), 4.0))


def test_single_candidate_and_inactive_padding_have_finite_backward() -> None:
    model = _model(_config(capacity=1, initial=1, minimum=1)).train()
    batch = _batch(module_width=12)
    output = model(batch)
    output["pred_field"].square().mean().backward()

    assert torch.equal(output["active_edge_count"], torch.ones(2))
    assert torch.isfinite(output["pred_field"]).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())

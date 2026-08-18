from __future__ import annotations

import torch

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.model import HONFNeuralField


def _config(*, assembly: str = "edge_additive", mechanism: str = "descriptor_first") -> UnifiedForwardConfig:
    return UnifiedForwardConfig(
        field_dim=3,
        domain_length_x=6.0,
        domain_length_y=3.0,
        coordinate_scale=[6.0, 3.0],
        periodic_axes=[],
        num_env_tokens_x=4,
        num_env_tokens_y=2,
        num_hyperedges=3,
        hidden_dim=24,
        dropout=0.0,
        decoder_mode="enhanced_honf_pairwise",
        pairwise_kernel_hidden_dim=24,
        pairwise_kernel_num_layers=2,
        mechanism_state_mode=mechanism,
        field_assembly_mode=assembly,
        routing_execution="dense",
    )


def _batch(device: torch.device | str = "cpu") -> BatchData:
    generator = torch.Generator().manual_seed(71)
    return BatchData(
        module_centers=(torch.rand(2, 4, 2, generator=generator) * torch.tensor([6.0, 3.0])).to(device),
        module_present=torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]], device=device),
        module_features=torch.randn(2, 4, 5, generator=generator).to(device),
        global_context=torch.randn(2, 4, generator=generator).to(device),
        query_xy=(torch.rand(2, 13, 2, generator=generator) * torch.tensor([6.0, 3.0])).to(device),
        query_time=None,
        target_field=None,
        case_name="additive-test",
        metadata={},
    )


def _model(*, assembly: str = "edge_additive", mechanism: str = "descriptor_first") -> HONFNeuralField:
    torch.manual_seed(73)
    return HONFNeuralField(_config(assembly=assembly, mechanism=mechanism))


def test_descriptor_first_organizer_features_include_scales_and_purities() -> None:
    model = _model().eval()
    with torch.no_grad():
        output = model(_batch())

    assert output["hyper_source_scale"].shape == (2, 3, 2)
    assert output["hyper_region_scale"].shape == (2, 3, 2)
    assert output["hyper_module_purity"].shape == (2, 3)
    assert output["hyper_env_purity"].shape == (2, 3)
    assert output["mechanism_descriptor_features"].shape == (2, 3, 16)
    assert torch.isfinite(output["mechanism_descriptor_features"]).all()


def test_additive_field_has_exact_exported_closure() -> None:
    model = _model().eval()
    with torch.no_grad():
        output = model(_batch(), return_edge_fields=True)

    reconstructed = output["pred_field_background"] + output["pred_field_by_edge"].sum(dim=2)
    torch.testing.assert_close(output["pred_field"], reconstructed, rtol=0.0, atol=0.0)
    assert output["edge_contribution_abs_mean"].shape == (2, 3, 3)
    assert output["edge_contribution_rms"].shape == (2, 3, 3)
    assert output["edge_contribution_energy_fraction"].shape == (2, 3, 3)


def test_inactive_edge_contributes_exactly_zero() -> None:
    model = _model().eval()
    batch = _batch()
    with torch.no_grad():
        encoded = model.encode_and_organize(batch)
        encoded["edge_active_mask"] = encoded["edge_active_mask"].clone()
        encoded["edge_active_mask"][:, -1] = 0.0
        output = model.decode_queries(
            batch.query_xy,
            None,
            encoded,
            encoded["global_token"],
            return_routing_maps=True,
            return_edge_fields=True,
        )

    assert torch.count_nonzero(output["pred_field_by_edge"][:, :, -1]) == 0
    assert torch.count_nonzero(output["query_hyper_attention"][:, :, -1]) == 0
    torch.testing.assert_close(
        output["pred_field"],
        output["pred_field_background"] + output["pred_field_by_edge"].sum(dim=2),
        rtol=0.0,
        atol=0.0,
    )


def _permute_edges(organized: dict[str, torch.Tensor], permutation: torch.Tensor) -> dict[str, torch.Tensor]:
    result = dict(organized)
    for key in ("A_mh", "A_eh"):
        result[key] = organized[key][..., permutation]
    for key in (
        "hyper_state",
        "hyper_source_coords",
        "hyper_region_coords",
        "hyper_source_variance",
        "hyper_source_scale",
        "hyper_region_variance",
        "hyper_region_scale",
        "hyper_module_mass_raw",
        "hyper_env_mass_raw",
        "hyper_module_mass",
        "hyper_env_mass",
        "hyper_module_purity",
        "hyper_env_purity",
        "hyper_strength",
        "edge_quality",
        "edge_active_mask",
        "mechanism_geometry_features",
        "mechanism_mass_features",
        "mechanism_raw_features",
        "mechanism_descriptor_features",
        "hyper_source_region_distance",
        "hyper_source_region_downstream",
        "hyper_source_region_lateral",
    ):
        result[key] = organized[key][:, permutation]
    return result


def test_consistent_fixed_edge_permutation_preserves_total_field() -> None:
    model = _model().eval()
    batch = _batch()
    permutation = torch.tensor([2, 0, 1])
    with torch.no_grad():
        organized = model.encode_and_organize(batch)
        reference = model.decode_queries(
            batch.query_xy,
            None,
            organized,
            organized["global_token"],
            return_edge_fields=True,
        )
        candidate = model.decode_queries(
            batch.query_xy,
            None,
            _permute_edges(organized, permutation),
            organized["global_token"],
            return_edge_fields=True,
        )

    torch.testing.assert_close(reference["pred_field"], candidate["pred_field"], rtol=1.0e-6, atol=1.0e-6)
    torch.testing.assert_close(
        reference["pred_field_by_edge"][:, :, permutation],
        candidate["pred_field_by_edge"],
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_additive_gradients_reach_descriptor_and_edge_encoders() -> None:
    model = _model().train()
    output = model(_batch())
    output["pred_field"].square().mean().backward()

    descriptor_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.decoder.mechanism_encoder.descriptor_encoder.parameters()
        if parameter.grad is not None
    )
    edge_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.decoder.edge_head.parameters()
        if parameter.grad is not None
    )
    assert descriptor_grad > 0.0
    assert edge_grad > 0.0


def test_additive_prepared_chunks_match_one_shot() -> None:
    model = _model().eval()
    batch = _batch()
    with torch.no_grad():
        organized = model.encode_and_organize(batch)
        reference = model.decode_queries(
            batch.query_xy,
            None,
            organized,
            organized["global_token"],
            return_edge_fields=True,
        )
        chunks = [
            model.decode_queries(
                query_chunk,
                None,
                organized,
                organized["global_token"],
                return_edge_fields=True,
            )
            for query_chunk in torch.tensor_split(batch.query_xy, 3, dim=1)
            if query_chunk.shape[1]
        ]

    for key, dim in (("pred_field", 1), ("pred_field_background", 1), ("pred_field_by_edge", 1)):
        torch.testing.assert_close(
            reference[key],
            torch.cat([chunk[key] for chunk in chunks], dim=dim),
            rtol=1.0e-6,
            atol=1.0e-6,
        )


def test_field_modes_instantiate_only_their_output_modules() -> None:
    context_model = _model(assembly="context_fusion", mechanism="residual_concat")
    additive_model = _model()
    context_names = set(dict(context_model.decoder.named_parameters()))
    additive_names = set(dict(additive_model.decoder.named_parameters()))

    assert any(name.startswith("pred_head.") for name in context_names)
    assert not any(name.startswith("edge_head.") for name in context_names)
    assert any(name.startswith("edge_head.") for name in additive_names)
    assert not any(name.startswith("pred_head.") for name in additive_names)

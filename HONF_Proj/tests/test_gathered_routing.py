from __future__ import annotations

import copy
from unittest.mock import patch

import torch

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.decoder import (
    HypergraphGatedPairwiseKernel,
    _routed_module_retention_statistics,
)
from honf_forward_core.model import HONFNeuralField


def _config(
    execution: str,
    *,
    module_limit: int,
    edge_limit: int,
) -> UnifiedForwardConfig:
    return UnifiedForwardConfig(
        field_dim=3,
        domain_length_x=6.0,
        domain_length_y=3.0,
        coordinate_scale=[6.0, 3.0],
        periodic_axes=[],
        num_env_tokens_x=5,
        num_env_tokens_y=3,
        num_hyperedges=4,
        organizer_mode="exchangeable_slots",
        edge_capacity=6,
        initial_active_edges=6,
        minimum_active_edges=1,
        edge_selection_mode="all",
        module_assignment_normalizer="entmax15",
        environment_assignment_normalizer="entmax15",
        query_assignment_normalizer="entmax15",
        environment_locality_mode="compact_kernel",
        environment_locality_strength=1.0,
        minimum_region_scale=0.08,
        hidden_dim=24,
        dropout=0.0,
        decoder_mode="enhanced_honf_pairwise",
        pairwise_kernel_hidden_dim=24,
        pairwise_kernel_num_layers=2,
        mechanism_state_mode="descriptor_first",
        field_assembly_mode="edge_additive",
        routing_execution=execution,
        query_module_limit=module_limit,
        query_edge_limit=edge_limit,
    )


def _batch() -> BatchData:
    generator = torch.Generator().manual_seed(151)
    return BatchData(
        module_centers=torch.rand(2, 8, 2, generator=generator) * torch.tensor([6.0, 3.0]),
        module_present=torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        module_features=torch.randn(2, 8, 5, generator=generator),
        global_context=torch.randn(2, 4, generator=generator),
        query_xy=torch.rand(2, 19, 2, generator=generator) * torch.tensor([6.0, 3.0]),
        query_time=None,
        target_field=None,
        case_name="gathered-routing-test",
        metadata={},
    )


def _initialized_model(config: UnifiedForwardConfig, *, seed: int) -> HONFNeuralField:
    torch.manual_seed(seed)
    model = HONFNeuralField(config).eval()
    with torch.no_grad():
        model(_batch())
    return model


def test_full_limit_gathered_matches_dense_with_same_weights() -> None:
    dense = _initialized_model(_config("dense", module_limit=16, edge_limit=8), seed=157)
    gathered = _initialized_model(_config("gathered", module_limit=16, edge_limit=8), seed=163)
    gathered.load_state_dict(copy.deepcopy(dense.state_dict()), strict=True)
    batch = _batch()
    with torch.no_grad():
        dense_output = dense(batch, return_edge_fields=True)
        gathered_output = gathered(batch, return_edge_fields=True)

    for key in (
        "pred_field",
        "pred_field_background",
        "pred_field_by_edge",
        "edge_contribution_abs_mean",
        "edge_contribution_rms",
        "edge_contribution_energy_fraction",
    ):
        torch.testing.assert_close(dense_output[key], gathered_output[key], rtol=2.0e-6, atol=2.0e-6)


def test_routed_module_retention_uses_only_routed_query_edge_pairs() -> None:
    retained = torch.tensor([[[1.0, 0.0, 1.0], [0.8, 0.0, 0.4]]])
    routed = torch.tensor([[[True, False, True], [True, False, True]]])

    diagnostics = _routed_module_retention_statistics(retained, routed)

    assert diagnostics["routed_query_edge_pair_count"] == 4
    torch.testing.assert_close(diagnostics["routed_module_retained_mass_mean"], torch.tensor(0.8))
    torch.testing.assert_close(diagnostics["routed_module_retained_mass_min"], torch.tensor(0.4))
    torch.testing.assert_close(
        diagnostics["routed_module_retained_mass_p05"],
        torch.quantile(torch.tensor([1.0, 1.0, 0.8, 0.4]), 0.05),
    )

    full = _routed_module_retention_statistics(
        torch.tensor([[[1.0, 0.0, 1.0, 0.0]]]),
        torch.tensor([[[True, False, True, False]]]),
    )
    assert full["routed_module_retained_mass_mean"] == 1.0
    assert full["routed_module_retained_mass_p05"] == 1.0
    assert full["routed_module_retained_mass_min"] == 1.0


def test_retention_diagnostics_do_not_change_predictions_or_checkpoint_structure() -> None:
    model = _initialized_model(_config("gathered", module_limit=3, edge_limit=2), seed=159)
    batch = _batch()
    checkpoint = copy.deepcopy(model.state_dict())

    with torch.no_grad():
        revised = model(batch)["pred_field"]
    with patch(
        "honf_forward_core.decoder._routed_module_retention_statistics",
        return_value={
            "routed_module_retained_mass_mean": revised.new_zeros(()),
            "routed_module_retained_mass_p05": revised.new_zeros(()),
            "routed_module_retained_mass_min": revised.new_zeros(()),
            "routed_query_edge_pair_count": revised.new_zeros(()),
        },
    ):
        with torch.no_grad():
            legacy_diagnostic_path = model(batch)["pred_field"]

    torch.testing.assert_close(revised, legacy_diagnostic_path, rtol=0.0, atol=0.0)
    assert tuple(model.state_dict()) == tuple(checkpoint)
    restored = _initialized_model(_config("gathered", module_limit=3, edge_limit=2), seed=161)
    restored.load_state_dict(checkpoint, strict=True)
    assert tuple(restored.state_dict()) == tuple(checkpoint)


def test_gathering_happens_before_pair_and_edge_mlps() -> None:
    model = _initialized_model(_config("gathered", module_limit=2, edge_limit=2), seed=167)
    pair_shapes: list[tuple[int, ...]] = []
    edge_shapes: list[tuple[int, ...]] = []

    def pair_hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        pair_shapes.append(tuple(inputs[0].shape))

    def edge_hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        edge_shapes.append(tuple(inputs[0].shape))

    pair_handle = model.decoder.pairwise_kernel.pair_mlp.register_forward_pre_hook(pair_hook)
    edge_handle = model.decoder.edge_head.register_forward_pre_hook(edge_hook)
    with torch.no_grad():
        output = model(_batch())
    pair_handle.remove()
    edge_handle.remove()

    assert pair_shapes
    assert all(len(shape) == 3 and shape[-2] <= 2 for shape in pair_shapes)
    assert edge_shapes
    assert all(len(shape) == 2 for shape in edge_shapes)
    assert output["pairwise_selected_modules"] == 2
    assert output["pairwise_selection_ratio"] < 1
    assert output["edge_head_selection_ratio"] < 1
    assert "pred_field_by_edge" not in output


def test_gathered_counters_exclude_inactive_modules() -> None:
    model = _initialized_model(_config("gathered", module_limit=8, edge_limit=6), seed=173)
    batch = _batch()
    with torch.no_grad():
        output = model(batch)

    expected_pairs = batch.query_xy.shape[1] * int(batch.module_present.sum())
    assert int(output["pairwise_evaluated_pair_count"]) == expected_pairs
    assert output["pairwise_selected_modules"] == 3.5
    assert output["pairwise_selected_modules"] < batch.module_present.shape[1]


def test_limited_gathered_backward_is_finite() -> None:
    torch.manual_seed(179)
    model = HONFNeuralField(_config("gathered", module_limit=2, edge_limit=2)).train()
    output = model(_batch())
    output["pred_field"].square().mean().backward()

    assert torch.isfinite(output["pred_field"]).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_gathered_query_chunks_match_one_shot() -> None:
    model = _initialized_model(_config("gathered", module_limit=2, edge_limit=2), seed=181)
    batch = _batch()
    with torch.no_grad():
        organized = model.encode_and_organize(batch)
        reference = model.decode_queries(
            batch.query_xy,
            None,
            organized,
            organized["global_token"],
        )["pred_field"]
        chunks = [
            model.decode_queries(
                query_chunk,
                None,
                organized,
                organized["global_token"],
            )["pred_field"]
            for query_chunk in torch.tensor_split(batch.query_xy, 4, dim=1)
            if query_chunk.shape[1]
        ]

    torch.testing.assert_close(reference, torch.cat(chunks, dim=1), rtol=2.0e-6, atol=2.0e-6)


def test_gathered_module_truncation_renormalizes_each_query_edge() -> None:
    config = _config("gathered", module_limit=1, edge_limit=2)
    kernel = HypergraphGatedPairwiseKernel(config)

    class Ones(torch.nn.Module):
        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return torch.ones(*values.shape[:-1], config.hidden_dim, device=values.device, dtype=values.dtype)

    kernel.pair_mlp = Ones()
    query_xy = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    module_centers = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    module_tokens = torch.zeros(1, 3, config.hidden_dim)
    module_present = torch.ones(1, 3)
    edge_module_weight = torch.tensor([[[0.9, 0.0], [0.1, 0.5], [0.0, 0.5]]])
    hyper_attention = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    contexts, _, _, retained = kernel._gathered_edge_pair_context(
        query_xy,
        module_centers,
        module_tokens,
        module_present,
        edge_module_weight,
        hyper_attention,
        None,
    )

    assert retained.shape == (1, 2, 2)
    torch.testing.assert_close(retained[0, 0, 0], torch.tensor(0.9))
    torch.testing.assert_close(retained[0, 1, 1], torch.tensor(0.5))
    torch.testing.assert_close(contexts[0, 0, 0], torch.ones(config.hidden_dim))
    torch.testing.assert_close(contexts[0, 1, 1], torch.ones(config.hidden_dim))


def test_query_edge_limit_is_shared_mass_conserving_routing() -> None:
    config = _config("gathered", module_limit=2, edge_limit=2)
    config.module_assignment_normalizer = "softmax"
    config.environment_assignment_normalizer = "softmax"
    config.query_assignment_normalizer = "softmax"
    config.environment_locality_mode = "none"
    model = _initialized_model(config, seed=191)
    batch = _batch()
    seen_pair_routes: list[torch.Tensor] = []

    def pair_hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        seen_pair_routes.append(inputs[2].detach().clone())

    handle = model.decoder.pairwise_kernel.register_forward_pre_hook(pair_hook)
    with torch.no_grad():
        organized = model.encode_and_organize(batch)
        output = model.decode_queries(
            batch.query_xy,
            None,
            organized,
            organized["global_token"],
            return_routing_maps=True,
            return_edge_fields=True,
        )
    handle.remove()

    routes = output["query_hyper_attention"]
    assert len(seen_pair_routes) == 1
    torch.testing.assert_close(seen_pair_routes[0], routes)
    torch.testing.assert_close(routes.sum(dim=-1), torch.ones_like(routes[..., 0]))
    assert torch.all((routes > 0).sum(dim=-1) <= 2)
    retained = output["query_edge_retained_probability_mass"]
    assert torch.isfinite(retained).all()
    assert torch.all((retained > 0) & (retained <= 1.0 + 1.0e-6))
    assert torch.any(retained < 1.0 - 1.0e-6)
    assert output["retained_module_incidence_mass"].shape == routes.shape
    assert torch.isfinite(output["retained_module_incidence_mass"]).all()
    routed_mask = output["routed_query_edge_pair_mask"]
    assert torch.equal(routed_mask, routes > 0)
    retained_routed = output["retained_module_incidence_mass"][routed_mask]
    torch.testing.assert_close(
        output["routed_module_retained_mass_mean"], retained_routed.mean()
    )
    assert output["routed_query_edge_pair_count"] == routed_mask.sum()
    edge_nonzero = output["pred_field_by_edge"].norm(dim=-1) > 0
    assert torch.all(edge_nonzero.sum(dim=-1) <= 2)

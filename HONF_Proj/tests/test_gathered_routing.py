from __future__ import annotations

import copy

import torch

from honf_forward_core.config import BatchData, UnifiedForwardConfig
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

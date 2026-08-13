from __future__ import annotations

import torch
import pytest

from honf_forward_core.config import BatchData, DECODER_MODES, UnifiedForwardConfig
from honf_forward_core.model import HONFNeuralField


def _batch() -> BatchData:
    torch.manual_seed(7)
    batch, modules, queries = 2, 4, 11
    return BatchData(
        module_centers=torch.rand(batch, modules, 2) * torch.tensor([12.0, 6.0]),
        module_present=torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        module_features=torch.randn(batch, modules, 5),
        global_context=torch.randn(batch, 3),
        query_xy=torch.rand(batch, queries, 2) * torch.tensor([12.0, 6.0]),
        query_time=None,
        target_field=None,
        case_name="synthetic",
        metadata={},
    )


def _model() -> HONFNeuralField:
    torch.manual_seed(11)
    config = UnifiedForwardConfig(
        field_dim=3,
        max_num_modules=4,
        domain_length_x=12.0,
        domain_length_y=6.0,
        num_env_tokens_x=4,
        num_env_tokens_y=2,
        num_hyperedges=3,
        hidden_dim=24,
        dropout=0.0,
        decoder_mode="enhanced_honf_pairwise",
        pairwise_kernel_hidden_dim=24,
    )
    return HONFNeuralField(config).eval()


def test_core_returns_finite_contract_shapes() -> None:
    model = _model()
    batch = _batch()
    with torch.no_grad():
        output = model(batch)
    assert output["pred_field"].shape == (2, 11, 3)
    assert output["A_mh"].shape == (2, 4, 3)
    assert output["A_eh"].shape == (2, 8, 3)
    assert torch.isfinite(output["pred_field"]).all()
    assert torch.count_nonzero(output["A_mh"][:, 3] * (1.0 - batch.module_present[:, 3:4])) == 0


def test_core_is_invariant_to_consistent_module_permutation() -> None:
    model = _model()
    batch = _batch()
    # Initialize all lazy layers before comparing the same trained state.
    with torch.no_grad():
        reference = model(batch)["pred_field"]
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = BatchData(
        module_centers=batch.module_centers[:, permutation],
        module_present=batch.module_present[:, permutation],
        module_features=batch.module_features[:, permutation],
        global_context=batch.global_context,
        query_xy=batch.query_xy,
        query_time=None,
        target_field=None,
        case_name=batch.case_name,
        metadata=batch.metadata,
    )
    with torch.no_grad():
        candidate = model(permuted)["pred_field"]
    assert torch.allclose(reference, candidate, atol=1.0e-6, rtol=1.0e-6)


def test_prepared_query_chunks_match_one_shot_decode() -> None:
    model = _model()
    batch = _batch()
    with torch.no_grad():
        encoded = model.encode_and_organize(batch)
        reference = model.decode_queries(batch.query_xy, None, encoded, encoded["global_token"])["pred_field"]
        chunks = [
            model.decode_queries(chunk, None, encoded, encoded["global_token"])["pred_field"]
            for chunk in torch.tensor_split(batch.query_xy, 3, dim=1)
            if chunk.shape[1]
        ]
    assert torch.allclose(reference, torch.cat(chunks, dim=1), atol=1.0e-6, rtol=1.0e-6)


@pytest.mark.parametrize("decoder_mode", sorted(DECODER_MODES))
def test_every_preserved_decoder_mode_is_finite(decoder_mode: str) -> None:
    torch.manual_seed(19)
    config = UnifiedForwardConfig(
        field_dim=3,
        max_num_modules=4,
        domain_length_x=12.0,
        domain_length_y=6.0,
        num_env_tokens_x=4,
        num_env_tokens_y=2,
        num_hyperedges=3,
        hidden_dim=24,
        dropout=0.0,
        decoder_mode=decoder_mode,
        pairwise_kernel_hidden_dim=24,
    )
    with torch.no_grad():
        output = HONFNeuralField(config).eval()(_batch())["pred_field"]
    assert output.shape == (2, 11, 3)
    assert torch.isfinite(output).all()


def test_case_supplied_query_features_are_supported_and_shape_checked() -> None:
    batch = _batch()
    batch.query_features = torch.randn(2, 11, 4)
    model = _model()
    with torch.no_grad():
        assert model(batch)["pred_field"].shape == (2, 11, 3)
    batch.query_features = torch.randn(2, 10, 4)
    with pytest.raises(ValueError, match="query_features"):
        model(batch)


def test_vector_scale_and_axis_periodicity_are_case_neutral() -> None:
    config = UnifiedForwardConfig(
        field_dim=2,
        max_num_modules=4,
        coordinate_scale=[12.0, 6.0],
        periodic_axes=[0],
        num_env_tokens_x=4,
        num_env_tokens_y=2,
        num_hyperedges=3,
        hidden_dim=24,
        dropout=0.0,
        decoder_mode="hyper_plus_global_near",
        boundary_feature_mode="none",
        local_context_scale=0.45,
    )
    assert config.spatial_scale() == (12.0, 6.0)
    assert config.periodic_dimensions() == (0,)
    with torch.no_grad():
        output = HONFNeuralField(config).eval()(_batch())["pred_field"]
    assert output.shape == (2, 11, 2)
    assert torch.isfinite(output).all()

"""Focused CORE tests for the Phase-2 tensor residual organizer."""

from __future__ import annotations

import copy

import pytest
import torch

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.model import HONFNeuralField


def _config(**updates: object) -> UnifiedForwardConfig:
    payload: dict[str, object] = {
        "field_dim": 2,
        "domain_length_x": 4.0,
        "domain_length_y": 2.0,
        "num_env_tokens_x": 3,
        "num_env_tokens_y": 2,
        "num_hyperedges": 0,
        "organizer_mode": "case_adaptive_tensor_residual",
        "edge_capacity": 0,
        "minimum_active_edges": 1,
        "residual_interaction_dim": 8,
        "residual_mechanism_cap_multiplier": 1.5,
        "residual_stop_fraction": 0.01,
        "residual_soft_stop_temperature": 0.002,
        "residual_factor_refinement_steps": 1,
        "hidden_dim": 16,
        "dropout": 0.0,
        "decoder_mode": "enhanced_honf_pairwise",
        "pairwise_aggregation_mode": "fused_query_module",
        "pairwise_kernel_hidden_dim": 16,
    }
    payload.update(updates)
    return UnifiedForwardConfig.from_dict(payload)


def _batch(module_count: int = 4, *, seed: int = 17) -> BatchData:
    generator = torch.Generator().manual_seed(seed)
    return BatchData(
        module_centers=torch.rand(2, module_count, 2, generator=generator),
        module_present=torch.tensor(
            [([1.0] * module_count), ([1.0] * max(module_count - 2, 1) + [0.0] * min(2, module_count))]
        ),
        module_features=torch.randn(2, module_count, 3, generator=generator),
        global_context=torch.randn(2, 4, generator=generator),
        query_xy=torch.rand(2, 7, 2, generator=generator),
        query_time=None,
        target_field=None,
        case_name="tensor-adaptive-core",
        metadata={},
    )


def _pad_module_axis(batch: BatchData, width: int) -> BatchData:
    padding = int(width) - int(batch.module_present.shape[1])
    if padding < 0:
        raise ValueError("Requested width is smaller than source batch.")
    return BatchData(
        module_centers=torch.cat(
            [batch.module_centers, batch.module_centers.new_zeros(batch.module_centers.shape[0], padding, 2)],
            dim=1,
        ),
        module_present=torch.cat(
            [batch.module_present, batch.module_present.new_zeros(batch.module_present.shape[0], padding)],
            dim=1,
        ),
        module_features=torch.cat(
            [
                batch.module_features,
                batch.module_features.new_zeros(batch.module_features.shape[0], padding, batch.module_features.shape[-1]),
            ],
            dim=1,
        ),
        global_context=batch.global_context,
        query_xy=batch.query_xy,
        query_time=batch.query_time,
        target_field=batch.target_field,
        case_name=batch.case_name,
        metadata=batch.metadata,
        env_coords=batch.env_coords,
        env_features=batch.env_features,
        query_features=batch.query_features,
    )


def _select_cases(batch: BatchData, indices: list[int]) -> BatchData:
    return BatchData(
        module_centers=batch.module_centers[indices],
        module_present=batch.module_present[indices],
        module_features=batch.module_features[indices],
        global_context=batch.global_context[indices],
        query_xy=batch.query_xy[indices],
        query_time=None if batch.query_time is None else batch.query_time[indices],
        target_field=None if batch.target_field is None else batch.target_field[indices],
        case_name=batch.case_name,
        metadata=batch.metadata,
        env_coords=(
            None
            if batch.env_coords is None
            else batch.env_coords if batch.env_coords.ndim == 2 else batch.env_coords[indices]
        ),
        env_features=None if batch.env_features is None else batch.env_features[indices],
        query_features=None if batch.query_features is None else batch.query_features[indices],
    )


def test_tensor_config_validation_and_isolated_parameter_path() -> None:
    config = _config()
    assert config.organizer_mode == "case_adaptive_tensor_residual"
    assert config.residual_interaction_dim == 8
    assert config.residual_mechanism_cap_multiplier == 1.5
    for field_name, value in (
        ("residual_interaction_dim", 0),
        ("residual_mechanism_cap_multiplier", 0.99),
        ("residual_stop_fraction", 0.0),
        ("residual_soft_stop_temperature", 0.0),
        ("residual_factor_refinement_steps", 0),
    ):
        with pytest.raises(ValueError):
            _config(**{field_name: value})
    with pytest.raises(ValueError, match="num_hyperedges"):
        _config(num_hyperedges=1)
    with pytest.raises(ValueError, match="edge_capacity"):
        _config(edge_capacity=1)

    model = HONFNeuralField(config)
    batch = _batch(seed=3)
    model.encode_and_organize(batch)
    names = set(model.state_dict())
    assert any(name.startswith("organizer.case_adaptive_tensor_residual.") for name in names)
    assert all("module_score" not in name and "env_score" not in name for name in names)
    assert tuple(model.state_dict()) == tuple(HONFNeuralField(config).state_dict())


def test_tensor_interaction_shape_nonnegative_zero_baseline_and_normalized_factors() -> None:
    torch.manual_seed(5)
    model = HONFNeuralField(_config()).eval()
    output = model.encode_and_organize(_batch(seed=11), return_residual_interaction_tensor=True)
    tensor = output["residual_interaction_tensor"]
    assert tensor.shape == (2, 4, 6, 8)
    assert torch.isfinite(tensor).all()
    assert torch.all(tensor >= 0)
    assert torch.equal(tensor[1, 2:], torch.zeros_like(tensor[1, 2:]))
    assert torch.all(tensor.sum(dim=(1, 2, 3)) > 0)
    full_output = model(_batch(seed=11), return_residual_interaction_tensor=True)
    torch.testing.assert_close(full_output["residual_interaction_tensor"], tensor)
    normalized_coupling = tensor.sum(dim=-1) / tensor.sum(dim=(1, 2, 3))[:, None, None]
    torch.testing.assert_close(
        output["A_me"] * output["residual_coupling_row_mass"].unsqueeze(-1),
        normalized_coupling,
        rtol=1e-6,
        atol=1e-6,
    )
    assert "residual_interaction_tensor" not in model.encode_and_organize(_batch(seed=11))

    slots = torch.arange(output["residual_module_factor"].shape[-1])
    viable = slots[None, :] < output["case_adaptive_edge_cap"][:, None]
    for factor in (
        output["residual_module_factor"],
        output["residual_environment_factor"],
        output["residual_content_factor"],
    ):
        torch.testing.assert_close(
            factor.sum(dim=1),
            viable.to(dtype=factor.dtype),
            rtol=1e-6,
            atol=1e-6,
        )
        assert torch.all(factor >= 0)
    trace = output["residual_fraction_trace"]
    assert torch.all(trace[:, 1:] <= trace[:, :-1] + 1e-6)
    assert torch.all(output["residual_mechanism_strength"] >= 0)

    zero_model = HONFNeuralField(_config()).eval()
    with torch.no_grad():
        for parameter in zero_model.organizer.case_adaptive_tensor_residual.parameters():
            parameter.zero_()
    zero = zero_model.encode_and_organize(_batch(seed=13), return_residual_interaction_tensor=True)
    assert torch.equal(zero["residual_interaction_tensor"], torch.zeros_like(zero["residual_interaction_tensor"]))
    assert torch.isfinite(zero["hyper_state"]).all()


def test_tensor_cap_exceeds_active_modules_and_strength_does_not_change_incidence() -> None:
    torch.manual_seed(19)
    model = HONFNeuralField(_config(residual_stop_fraction=1.0e-12)).eval()
    output = model.encode_and_organize(_batch(seed=23))
    # The two cases have four and two active modules; their safety caps are
    # six and three, respectively, and the tiny threshold keeps all cap slots.
    assert output["A_mh"].shape[-1] == 6
    torch.testing.assert_close(output["case_adaptive_edge_cap"], torch.tensor([6.0, 3.0]))
    torch.testing.assert_close(output["case_adaptive_edge_count"], output["case_adaptive_edge_cap"])
    assert torch.all(output["case_adaptive_edge_count"] > output["module_present"].sum(dim=-1))
    hard = output["hard_case_edge_mask"]
    expected_mh = output["residual_module_factor"] * hard[:, None, :]
    expected_mh = expected_mh / expected_mh.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    expected_mh = expected_mh * output["module_present"][:, :, None]
    torch.testing.assert_close(output["A_mh"], expected_mh, rtol=1e-6, atol=1e-6)

    # Amplitude is a separate output; changing it cannot alter topology
    # membership (the incidence reconstruction above never uses strength).
    assert torch.equal(output["hard_case_edge_mask"], hard)


def test_tensor_module_permutation_padding_and_batch_composition_invariance() -> None:
    torch.manual_seed(29)
    model = HONFNeuralField(_config()).eval()
    batch = _batch(seed=31)
    padded = _pad_module_axis(batch, 6)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = copy.copy(batch)
    permuted.module_centers = batch.module_centers[:, permutation]
    permuted.module_present = batch.module_present[:, permutation]
    permuted.module_features = batch.module_features[:, permutation]
    with torch.no_grad():
        reference = model(batch)
        candidate = model(padded)
        reordered = model(permuted)
    for key in (
        "pred_field",
        "A_eh",
        "hyper_state",
        "residual_fraction_trace",
        "residual_mechanism_strength",
        "hard_case_edge_mask",
        "edge_survival_soft",
    ):
        torch.testing.assert_close(reference[key], candidate[key], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(reference[key], reordered[key], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        reference["A_mh"],
        candidate["A_mh"][:, : batch.module_present.shape[1]],
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        reference["A_mh"],
        reordered["A_mh"].index_select(1, torch.argsort(permutation)),
        rtol=1e-6,
        atol=1e-6,
    )
    assert torch.equal(candidate["A_mh"][:, batch.module_present.shape[1] :], torch.zeros_like(candidate["A_mh"][:, batch.module_present.shape[1] :]))

    singleton = _select_cases(batch, [0])
    with torch.no_grad():
        isolated = model(singleton)
    for key in (
        "pred_field",
        "A_me",
        "A_mh",
        "A_eh",
        "hyper_state",
        "residual_fraction_trace",
        "residual_mechanism_strength",
        "hard_case_edge_mask",
        "edge_survival_soft",
        "case_adaptive_edge_count",
        "case_adaptive_edge_cap",
    ):
        if isolated[key].shape[-1] != reference[key].shape[-1] and key in {"A_mh", "A_eh", "hard_case_edge_mask", "edge_survival_soft"}:
            torch.testing.assert_close(reference[key][0:1, ..., : isolated[key].shape[-1]], isolated[key], rtol=1e-6, atol=1e-6)
        else:
            torch.testing.assert_close(reference[key][0:1], isolated[key], rtol=1e-6, atol=1e-6)


def test_tensor_train_eval_hard_forward_decoder_support_and_soft_count() -> None:
    torch.manual_seed(37)
    model = HONFNeuralField(_config()).train()
    batch = _batch(seed=41)
    train_output = model(batch)
    model.eval()
    eval_output = model(batch)
    for key in ("pred_field", "A_mh", "A_eh", "hyper_state", "hard_case_edge_mask"):
        torch.testing.assert_close(train_output[key], eval_output[key], rtol=1e-6, atol=1e-6)
    assert torch.equal(
        train_output["A_mh"].masked_select(train_output["hard_case_edge_mask"][:, None, :] <= 0),
        torch.zeros_like(train_output["A_mh"].masked_select(train_output["hard_case_edge_mask"][:, None, :] <= 0)),
    )
    assert torch.equal(
        train_output["A_eh"].masked_select(train_output["hard_case_edge_mask"][:, None, :] <= 0),
        torch.zeros_like(train_output["A_eh"].masked_select(train_output["hard_case_edge_mask"][:, None, :] <= 0)),
    )

    encoded = model.encode_and_organize(batch)
    decoded = model.decode_queries(
        batch.query_xy,
        None,
        encoded,
        encoded["global_token"],
        return_routing_maps=True,
    )
    attention = decoded["query_hyper_attention"]
    hard = encoded["hard_case_edge_mask"]
    assert torch.equal(
        attention.masked_select(hard[:, None, :] <= 0),
        torch.zeros_like(attention.masked_select(hard[:, None, :] <= 0)),
    )
    torch.testing.assert_close(attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1)), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        decoded["case_adaptive_soft_edge_count"],
        encoded["case_adaptive_soft_edge_count"],
        rtol=1e-6,
        atol=1e-6,
    )
    assert "edge_survival_weight" not in encoded


def test_tensor_prepared_chunk_and_dense_gathered_full_support_parity() -> None:
    torch.manual_seed(41)
    model = HONFNeuralField(_config()).eval()
    batch = _batch(seed=43)
    with torch.no_grad():
        encoded = model.encode_and_organize(batch)
        full = model.decode_queries(batch.query_xy, None, encoded, encoded["global_token"], return_routing_maps=True)
        chunks = [
            model.decode_queries(chunk, None, encoded, encoded["global_token"], return_routing_maps=True)
            for chunk in torch.tensor_split(batch.query_xy, 3, dim=1)
            if chunk.shape[1]
        ]
    torch.testing.assert_close(torch.cat([item["pred_field"] for item in chunks], dim=1), full["pred_field"], rtol=1e-6, atol=1e-6)

    dense_cfg = _config(residual_stop_fraction=1.0e-12, routing_execution="dense")
    gathered_cfg = _config(residual_stop_fraction=1.0e-12, routing_execution="gathered")
    torch.manual_seed(47)
    dense = HONFNeuralField(dense_cfg).eval()
    dense(batch)
    torch.manual_seed(48)
    gathered = HONFNeuralField(gathered_cfg).eval()
    gathered(batch)
    gathered.load_state_dict(copy.deepcopy(dense.state_dict()), strict=True)
    with torch.no_grad():
        dense_encoded = dense.encode_and_organize(batch)
        gathered_encoded = gathered.encode_and_organize(batch)
        dense_decoded = dense.decode_queries(
            batch.query_xy,
            None,
            dense_encoded,
            dense_encoded["global_token"],
            return_routing_maps=True,
        )
        gathered_decoded = gathered.decode_queries(
            batch.query_xy,
            None,
            gathered_encoded,
            gathered_encoded["global_token"],
            return_routing_maps=True,
        )
    for key in ("pred_field", "query_hyper_attention", "query_module_routing_beta", "c_pair_norm"):
        torch.testing.assert_close(dense_decoded[key], gathered_decoded[key], rtol=1e-6, atol=1e-6)


def test_tensor_gradients_reach_interaction_factor_state_routing_and_head() -> None:
    torch.manual_seed(43)
    model = HONFNeuralField(_config()).train()
    output = model(_batch(seed=47))
    output["edge_survival_soft"].retain_grad()
    output["pred_field"].square().mean().backward()
    named = dict(model.named_parameters())

    def assert_group(prefixes: tuple[str, ...]) -> None:
        group = [parameter for name, parameter in named.items() if name.startswith(prefixes)]
        assert group
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in group)
        assert any(parameter.grad.abs().sum() > 0 for parameter in group if parameter.grad is not None)

    assert_group(("organizer.case_adaptive_tensor_residual.module_content_proj.",))
    assert_group(("organizer.case_adaptive_tensor_residual.environment_content_proj.",))
    assert_group(("organizer.case_adaptive_tensor_residual.geometry_encoder.",))
    assert_group(("organizer.case_adaptive_tensor_residual.mechanism_mixer.",))
    assert_group(("decoder.query_to_hyper.", "decoder.hyper_key."))
    assert_group(("decoder.pred_head.",))
    assert output["edge_survival_soft"].grad is not None
    assert torch.isfinite(output["edge_survival_soft"].grad).all()
    assert output["edge_survival_soft"].grad.abs().sum() > 0


def test_tensor_state_shapes_and_gamma_zero_initialization_ignore_module_count() -> None:
    config = _config()
    small = HONFNeuralField(config).eval()
    large = HONFNeuralField(config).eval()
    with torch.no_grad():
        small(_batch(seed=53))
        large(_pad_module_axis(_batch(seed=53), 7))
    assert tuple(small.state_dict()) == tuple(large.state_dict())
    small_shapes = {name: tuple(parameter.shape) for name, parameter in small.named_parameters()}
    large_shapes = {name: tuple(parameter.shape) for name, parameter in large.named_parameters()}
    assert small_shapes == large_shapes
    organizer = small.organizer.case_adaptive_tensor_residual
    assert torch.equal(organizer.global_module_film[-1].weight, torch.zeros_like(organizer.global_module_film[-1].weight))
    assert torch.equal(organizer.global_module_film[-1].bias, torch.zeros_like(organizer.global_module_film[-1].bias))
    assert torch.equal(organizer.global_environment_film[-1].weight, torch.zeros_like(organizer.global_environment_film[-1].weight))
    assert torch.equal(organizer.global_environment_film[-1].bias, torch.zeros_like(organizer.global_environment_film[-1].bias))

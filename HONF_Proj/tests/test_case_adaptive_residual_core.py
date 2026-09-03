from __future__ import annotations

import copy

import pytest
import torch

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.model import HONFNeuralField
from honf_forward_core.organizer import HypergraphOrganizerCore
from honf_runtime.config_loader import load_config_bundle


def _config(**updates: object) -> UnifiedForwardConfig:
    payload: dict[str, object] = {
        "field_dim": 2,
        "domain_length_x": 4.0,
        "domain_length_y": 2.0,
        "num_env_tokens_x": 3,
        "num_env_tokens_y": 2,
        "num_hyperedges": 0,
        "organizer_mode": "case_adaptive_residual",
        "edge_capacity": 0,
        "minimum_active_edges": 1,
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
        case_name="adaptive-core",
        metadata={},
    )


def _select_cases(batch: BatchData, indices: list[int]) -> BatchData:
    """Select cases without changing their packed module width."""

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
            else (
                batch.env_coords
                if batch.env_coords.ndim == 2
                else batch.env_coords[indices]
            )
        ),
        env_features=(
            None
            if batch.env_features is None
            else batch.env_features[indices]
        ),
        query_features=(
            None
            if batch.query_features is None
            else batch.query_features[indices]
        ),
    )


def _pad_module_axis(batch: BatchData, width: int) -> BatchData:
    """Append inactive module slots while retaining every active input."""

    padding = int(width) - int(batch.module_present.shape[1])
    if padding < 0:
        raise ValueError("Requested width is smaller than the source batch.")
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
                batch.module_features.new_zeros(
                    batch.module_features.shape[0], padding, batch.module_features.shape[-1]
                ),
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


def test_config_accepts_residual_mode_and_validates_residual_fields() -> None:
    config = _config()
    assert config.num_hyperedges == 0
    assert config.edge_capacity == 0
    assert config.residual_stop_fraction == 0.02

    for name, value in (
        ("residual_stop_fraction", 0.0),
        ("residual_stop_fraction", 1.0),
        ("residual_soft_stop_temperature", 0.0),
        ("residual_factor_refinement_steps", 0),
        ("residual_coupling_fourier_frequencies", -1),
    ):
        with pytest.raises(ValueError):
            _config(**{name: value})

    with pytest.raises(ValueError, match="num_hyperedges"):
        _config(num_hyperedges=2)
    with pytest.raises(ValueError, match="edge_capacity"):
        _config(edge_capacity=2)


def test_residual_profile_resolves_through_strict_loader() -> None:
    bundle = load_config_bundle(
        "project://src/config_core/forward/case_adaptive_residual_context.json"
    )
    payload = bundle.core["model"]["core_honf"]
    assert payload["organizer_mode"] == "case_adaptive_residual"
    assert payload["num_hyperedges"] == 0
    assert payload["edge_capacity"] == 0
    assert payload["pairwise_aggregation_mode"] == "fused_query_module"


def test_residual_core_contract_math_and_packed_masking() -> None:
    torch.manual_seed(3)
    config = _config()
    model = HONFNeuralField(config).eval()
    output = model.encode_and_organize(_batch())

    assert output["A_mh"].shape == (2, 4, 4)
    assert output["A_eh"].shape == (2, 6, 4)
    assert torch.isfinite(output["residual_fraction_trace"]).all()
    assert torch.all(output["residual_fraction_trace"][:, 1:] <= output["residual_fraction_trace"][:, :-1] + 1e-6)
    assert torch.all(output["residual_mechanism_strength"] >= 0)
    assert torch.isfinite(output["residual_mechanism_strength"]).all()

    # A_me and its row masses reconstruct the normalized nonnegative coupling
    # without returning a dense residual history from the organizer.
    normalized_coupling = output["A_me"] * output["residual_coupling_row_mass"].unsqueeze(-1)
    assert torch.all(normalized_coupling >= 0)
    coupling_norm = torch.linalg.vector_norm(normalized_coupling, dim=(1, 2))
    torch.testing.assert_close(coupling_norm, torch.ones_like(coupling_norm), rtol=1e-6, atol=1e-6)

    packed_indices = torch.arange(output["residual_module_factor"].shape[-1])
    valid_steps = packed_indices[None, :] < output["case_adaptive_edge_cap"][:, None]
    torch.testing.assert_close(
        output["residual_module_factor"].sum(dim=1),
        valid_steps.to(output["residual_module_factor"].dtype),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        output["residual_environment_factor"].sum(dim=1),
        valid_steps.to(output["residual_environment_factor"].dtype),
        rtol=1e-6,
        atol=1e-6,
    )
    assert torch.all(output["residual_module_factor"] >= 0)
    assert torch.all(output["residual_environment_factor"] >= 0)
    assert torch.allclose(
        output["A_mh"].sum(dim=-1),
        output["module_present"],
        atol=2e-5,
    )
    assert torch.allclose(output["A_eh"].sum(dim=-1), torch.ones_like(output["A_eh"].sum(dim=-1)), atol=2e-5)
    assert torch.all(output["case_adaptive_edge_count"] >= 1)
    assert torch.all(output["case_adaptive_edge_count"] <= output["case_adaptive_edge_cap"])
    hard_mask = output["hard_case_edge_mask"] > 0
    for incidence in (output["A_mh"], output["A_eh"]):
        inactive = ~hard_mask[:, None, :].expand_as(incidence)
        assert torch.equal(incidence.masked_select(inactive), torch.zeros_like(incidence.masked_select(inactive)))
    tiny_base = torch.full((1, 2, 2), 0.5)
    tiny_support = torch.tensor([[1.0e-9, 2.0e-9]])
    tiny_normalized, _ = model.organizer.case_adaptive_residual._supported_factor_normalize(
        tiny_base,
        tiny_support,
        torch.ones(1, 2, dtype=torch.bool),
    )
    torch.testing.assert_close(
        tiny_normalized,
        torch.tensor([[[1.0 / 3.0, 2.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0]]]),
        rtol=1e-6,
        atol=1e-6,
    )
    # Case 1 has two padded mechanism slots after its two active modules.
    assert torch.equal(output["A_mh"][1, 2:], torch.zeros_like(output["A_mh"][1, 2:]))
    assert torch.equal(output["hyper_state"][1, 2:], torch.zeros_like(output["hyper_state"][1, 2:]))


def test_residual_core_is_module_order_invariant_and_has_gradients() -> None:
    torch.manual_seed(5)
    config = _config()
    model = HONFNeuralField(config).eval()
    batch = _batch(seed=23)
    baseline = model(batch)["pred_field"]
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = copy.copy(batch)
    permuted.module_centers = batch.module_centers[:, permutation]
    permuted.module_present = batch.module_present[:, permutation]
    permuted.module_features = batch.module_features[:, permutation]
    permuted_output = model(permuted)["pred_field"]
    torch.testing.assert_close(baseline, permuted_output, rtol=1e-6, atol=1e-6)

    model.train()
    output = model(batch)
    output["edge_survival_weight"].retain_grad()
    output["pred_field"].square().mean().backward()
    adaptive_parameters = list(model.organizer.case_adaptive_residual.parameters())
    assert adaptive_parameters
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in adaptive_parameters)
    assert any(parameter.grad.abs().sum() > 0 for parameter in adaptive_parameters if parameter.grad is not None)

    named_gradients = dict(model.named_parameters())

    def assert_group_has_gradient(prefixes: tuple[str, ...]) -> None:
        group = [
            parameter
            for name, parameter in named_gradients.items()
            if name.startswith(prefixes)
        ]
        assert group
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in group)
        assert any(parameter.grad.abs().sum() > 0 for parameter in group if parameter.grad is not None)

    assert_group_has_gradient(("organizer.case_adaptive_residual.coupling_",))
    assert_group_has_gradient(("organizer.case_adaptive_residual.coupling_geometry.",))
    assert_group_has_gradient(
        (
            "organizer.case_adaptive_residual.environment_query.",
            "organizer.case_adaptive_residual.environment_key.",
            "organizer.case_adaptive_residual.module_query.",
            "organizer.case_adaptive_residual.module_key.",
        )
    )
    assert_group_has_gradient(("organizer.case_adaptive_residual.mechanism_mixer.",))
    assert_group_has_gradient(("decoder.query_to_hyper.", "decoder.hyper_key."))
    assert_group_has_gradient(("decoder.pred_head.",))
    assert output["edge_survival_weight"].grad is not None
    assert torch.isfinite(output["edge_survival_weight"].grad).all()
    assert output["edge_survival_weight"].grad.abs().sum() > 0


def test_decoder_uses_soft_and_hard_residual_support() -> None:
    config = _config()
    model = HONFNeuralField(config)
    batch = _batch(seed=31)
    model.eval()
    encoded = model.encode_and_organize(batch)
    decoded = model.decode_queries(batch.query_xy, None, encoded, encoded["global_token"], return_routing_maps=True)
    attention = decoded["query_hyper_attention"]
    hard_mask = encoded["hard_case_edge_mask"]
    assert torch.all(attention.masked_select(hard_mask[:, None, :] <= 0) == 0)
    assert torch.allclose(attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1)), atol=2e-5)
    torch.testing.assert_close(
        decoded["case_adaptive_soft_edge_count"],
        encoded["case_adaptive_soft_edge_count"],
        rtol=1e-6,
        atol=1e-6,
    )

    model.train()
    encoded_train = model.encode_and_organize(batch)
    assert torch.any((encoded_train["edge_survival_weight"] > 0) & (encoded_train["edge_survival_weight"] < 1))
    torch.testing.assert_close(
        encoded_train["case_adaptive_soft_edge_count"],
        encoded_train["edge_survival_weight"].sum(dim=-1),
        rtol=1e-6,
        atol=1e-6,
    )
    decoded_train = model.decode_queries(batch.query_xy, None, encoded_train, model.global_encoder(batch.global_context), return_routing_maps=True)
    assert torch.isfinite(decoded_train["pred_field"]).all()


def test_residual_stopping_has_controlled_early_and_cap_paths() -> None:
    # Zeroing the shared organizer projections produces an exactly rank-one
    # uniform coupling, so the first extracted component exhausts the
    # residual up to floating-point error.
    torch.manual_seed(13)
    early_model = HONFNeuralField(_config()).eval()
    with torch.no_grad():
        for parameter in early_model.organizer.case_adaptive_residual.parameters():
            parameter.zero_()
        early = early_model.encode_and_organize(_batch(seed=41))
    assert torch.equal(early["case_adaptive_edge_count"], torch.ones(2))
    assert torch.equal(early["case_adaptive_stop_reached"], torch.ones(2))
    assert torch.equal(early["case_adaptive_cap_hit"], torch.zeros(2))
    assert torch.all(early["residual_fraction_trace"][:, 1] <= early["residual_stop_fraction"])

    # A fixed seed and a tolerance below the deterministic finite-precision
    # residual force the specified M_b cap fallback.
    torch.manual_seed(3)
    cap_model = HONFNeuralField(_config(residual_stop_fraction=1.0e-12)).eval()
    with torch.no_grad():
        capped = cap_model.encode_and_organize(_batch(seed=17))
    torch.testing.assert_close(capped["case_adaptive_edge_count"], capped["case_adaptive_edge_cap"])
    assert torch.equal(capped["case_adaptive_cap_hit"], torch.ones(2))
    assert torch.equal(capped["case_adaptive_stop_reached"], torch.zeros(2))
    assert torch.all(capped["case_adaptive_stop_margin"] < 0)


def test_residual_organizer_is_invariant_to_inactive_padding() -> None:
    torch.manual_seed(29)
    model = HONFNeuralField(_config()).eval()
    batch = _batch(seed=53)
    padded = _pad_module_axis(batch, 6)
    with torch.no_grad():
        reference = model(batch)
        candidate = model(padded)

    assert candidate["A_mh"].shape[-1] == reference["A_mh"].shape[-1]
    assert candidate["A_eh"].shape[-1] == reference["A_eh"].shape[-1]
    assert candidate["hyper_state"].shape[1] == reference["hyper_state"].shape[1]
    for key in (
        "pred_field",
        "case_adaptive_edge_count",
        "case_adaptive_edge_cap",
        "case_adaptive_soft_edge_count",
        "case_adaptive_cap_hit",
    ):
        torch.testing.assert_close(reference[key], candidate[key], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        reference["A_me"],
        candidate["A_me"][:, : reference["A_me"].shape[1]],
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        reference["module_env_context"],
        candidate["module_env_context"][:, : reference["module_env_context"].shape[1]],
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        reference["residual_coupling_row_mass"],
        candidate["residual_coupling_row_mass"][:, : reference["residual_coupling_row_mass"].shape[1]],
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        reference["A_mh"],
        candidate["A_mh"][:, : reference["A_mh"].shape[1], : reference["A_mh"].shape[2]],
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        reference["A_eh"],
        candidate["A_eh"][:, :, : reference["A_eh"].shape[2]],
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        reference["hyper_state"],
        candidate["hyper_state"][:, : reference["hyper_state"].shape[1]],
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        reference["candidate_hyper_state"],
        candidate["candidate_hyper_state"][:, : reference["candidate_hyper_state"].shape[1]],
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        reference["residual_module_factor"],
        candidate["residual_module_factor"][:, : reference["residual_module_factor"].shape[1], : reference["residual_module_factor"].shape[2]],
        rtol=1e-6,
        atol=1e-6,
    )
    for key in ("residual_environment_factor", "residual_mechanism_strength", "hard_case_edge_mask", "edge_survival_weight"):
        torch.testing.assert_close(
            reference[key],
            candidate[key][..., : reference[key].shape[-1]],
            rtol=1e-6,
            atol=1e-6,
        )
    torch.testing.assert_close(
        reference["residual_fraction_trace"],
        candidate["residual_fraction_trace"][..., : reference["residual_fraction_trace"].shape[-1]],
        rtol=1e-6,
        atol=1e-6,
    )
    assert torch.equal(candidate["A_mh"][:, 4:], torch.zeros_like(candidate["A_mh"][:, 4:]))
    assert torch.equal(candidate["A_me"][:, 4:], torch.zeros_like(candidate["A_me"][:, 4:]))


def test_residual_organizer_is_invariant_to_batch_composition() -> None:
    torch.manual_seed(37)
    model = HONFNeuralField(_config()).eval()
    batch = _batch(seed=67)
    singleton = _select_cases(batch, [0])
    with torch.no_grad():
        batched = model(batch)
        isolated = model(singleton)
    for key in (
        "pred_field",
        "A_me",
        "A_mh",
        "A_eh",
        "hyper_state",
        "residual_fraction_trace",
        "residual_mechanism_strength",
        "edge_survival_weight",
        "hard_case_edge_mask",
        "case_adaptive_edge_count",
        "case_adaptive_edge_cap",
        "case_adaptive_soft_edge_count",
    ):
        torch.testing.assert_close(batched[key][0:1], isolated[key], rtol=1e-6, atol=1e-6)


def test_residual_evaluation_is_repeatable_and_prepared_chunks_match() -> None:
    torch.manual_seed(43)
    model = HONFNeuralField(_config()).eval()
    batch = _batch(seed=73)
    with torch.no_grad():
        first = model(batch)
        second = model(batch)
        encoded = model.encode_and_organize(batch)
        reference = model.decode_queries(
            batch.query_xy,
            None,
            encoded,
            encoded["global_token"],
        )["pred_field"]
        chunks = [
            model.decode_queries(chunk, None, encoded, encoded["global_token"])["pred_field"]
            for chunk in torch.tensor_split(batch.query_xy, 3, dim=1)
            if chunk.shape[1]
        ]
    for key in ("pred_field", "A_me", "A_mh", "A_eh", "hyper_state", "residual_fraction_trace"):
        assert torch.equal(first[key], second[key])
    torch.testing.assert_close(torch.cat(chunks, dim=1), reference, rtol=1e-6, atol=1e-6)


def test_residual_full_support_dense_and_gathered_decoding_match() -> None:
    batch = _batch(seed=79)
    dense_config = _config(
        residual_stop_fraction=1.0e-12,
        routing_execution="dense",
        query_module_limit=0,
        query_module_retained_mass_floor=1.0,
    )
    gathered_config = _config(
        residual_stop_fraction=1.0e-12,
        routing_execution="gathered",
        query_module_limit=0,
        query_module_retained_mass_floor=1.0,
    )
    torch.manual_seed(47)
    dense = HONFNeuralField(dense_config).eval()
    with torch.no_grad():
        dense_init = dense.encode_and_organize(batch)
        dense.decode_queries(
            batch.query_xy,
            None,
            dense_init,
            dense_init["global_token"],
        )
    torch.manual_seed(48)
    gathered = HONFNeuralField(gathered_config).eval()
    with torch.no_grad():
        gathered_init = gathered.encode_and_organize(batch)
        gathered.decode_queries(
            batch.query_xy,
            None,
            gathered_init,
            gathered_init["global_token"],
        )
    gathered.load_state_dict(copy.deepcopy(dense.state_dict()), strict=True)

    with torch.no_grad():
        dense_state = dense.encode_and_organize(batch)
        gathered_state = gathered.encode_and_organize(batch)
        dense_output = dense.decode_queries(
            batch.query_xy,
            None,
            dense_state,
            dense_state["global_token"],
            return_routing_maps=True,
        )
        gathered_output = gathered.decode_queries(
            batch.query_xy,
            None,
            gathered_state,
            gathered_state["global_token"],
            return_routing_maps=True,
        )
    packed_indices = torch.arange(dense_state["hard_case_edge_mask"].shape[-1])
    expected_support = packed_indices[None, :] < dense_state["case_adaptive_edge_cap"][:, None]
    assert torch.equal(dense_state["hard_case_edge_mask"] > 0, expected_support)
    assert torch.equal(dense_state["hard_case_edge_mask"], gathered_state["hard_case_edge_mask"])
    for key in ("pred_field", "query_hyper_attention", "query_module_routing_beta", "c_pair_norm"):
        torch.testing.assert_close(dense_output[key], gathered_output[key], rtol=1e-6, atol=1e-6)


def test_facade_does_not_register_fixed_projection_parameters() -> None:
    organizer = HypergraphOrganizerCore(_config())
    names = set(organizer.state_dict())
    assert names
    assert all("module_score" not in name and "env_score" not in name for name in names)
    assert all("case_adaptive_residual" in name for name in names)


def test_residual_parameter_and_state_keys_ignore_module_count() -> None:
    config = _config()
    small = HONFNeuralField(config).eval()
    large = HONFNeuralField(config).eval()
    with torch.no_grad():
        small(_batch(seed=83))
        large(_pad_module_axis(_batch(seed=83), 6))

    assert tuple(small.state_dict()) == tuple(large.state_dict())
    small_shapes = {name: tuple(parameter.shape) for name, parameter in small.named_parameters()}
    large_shapes = {name: tuple(parameter.shape) for name, parameter in large.named_parameters()}
    assert small_shapes == large_shapes
    assert sum(parameter.numel() for parameter in small.parameters()) == sum(
        parameter.numel() for parameter in large.parameters()
    )

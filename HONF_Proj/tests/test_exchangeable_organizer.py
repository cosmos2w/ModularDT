from __future__ import annotations

import copy

import torch
import torch.nn as nn

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.model import HONFNeuralField
from honf_forward_core.organizer import deterministic_slot_codes
from honf_forward_core.training.diagnostics import (
    compute_code_permutation_equivariance_diagnostics,
    compute_honf_diagnostics,
)


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
    equivariance = compute_code_permutation_equivariance_diagnostics(
        {**reference, **reference_field},
        {**candidate, **candidate_field},
        permutation,
    )
    assert equivariance["code_permutation_equivariance_max_error"] < 1.0e-5
    assert "code_permutation_candidate_source_scale_max_error" in equivariance
    assert "code_permutation_pred_field_max_error" in equivariance


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
        model.set_edge_capacity(8)
        output_8 = model(batch)
        parameter_shapes_8 = {name: tuple(value.shape) for name, value in model.state_dict().items()}

    assert output_6["candidate_A_mh"].shape[-1] == 6
    assert output_8["candidate_A_mh"].shape[-1] == 8
    assert parameter_shapes_6 == parameter_shapes_8

    other = _model(_config(capacity=8)).eval()
    with torch.no_grad():
        other(batch)
    assert {name: tuple(value.shape) for name, value in other.state_dict().items()} == parameter_shapes_8


def test_stage2_soft_bridge_selects_all_six_candidates_and_exports_candidate_diagnostics() -> None:
    payload = _config(capacity=6, initial=6).to_dict()
    payload.update(
        edge_selection_mode="all",
        module_assignment_normalizer="softmax",
        environment_assignment_normalizer="softmax",
        query_assignment_normalizer="softmax",
        environment_locality_mode="none",
        routing_execution="dense",
        query_edge_limit=0,
        query_module_limit=0,
    )
    model = _model(UnifiedForwardConfig.from_dict(payload)).eval()
    with torch.no_grad():
        output = model(_batch(), return_edge_fields=True)

    assert torch.equal(output["selected_edge_count"], torch.full((2,), 6.0))
    assert torch.equal(output["viable_selected_edge_count"], torch.full((2,), 6.0))
    assert torch.equal(output["edge_active_mask"], torch.ones_like(output["edge_active_mask"]))
    assert torch.equal(output["edge_viable_mask"], torch.ones_like(output["edge_viable_mask"]))
    assert output["candidate_module_mass_fraction"].shape == (2, 6)
    assert output["candidate_environment_mass_fraction"].shape == (2, 6)
    assert output["candidate_module_purity"].shape == (2, 6)
    assert output["candidate_environment_purity"].shape == (2, 6)
    assert output["candidate_source_scale"].shape == (2, 6, 2)
    assert output["candidate_region_scale"].shape == (2, 6, 2)

    diagnostics = compute_honf_diagnostics(
        {"pred_field": output["pred_field"], "organizer_aux": output, "routing_aux": output}
    )
    required = {
        "candidate_module_mass_fraction_min",
        "candidate_environment_mass_fraction_min",
        "candidate_module_purity_mean",
        "candidate_environment_purity_mean",
        "candidate_source_scale_mean",
        "candidate_region_scale_mean",
        "edge_contribution_fraction_min",
        "edge_contribution_fraction_max",
    }
    assert required <= diagnostics.keys()
    assert all(torch.isfinite(torch.tensor(diagnostics[key])) for key in required)


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


def test_training_warmup_uses_all_viable_candidates() -> None:
    model = _model().train()
    model.set_training_progress(epoch=0, total_epochs=10)
    output = model(_batch())

    torch.testing.assert_close(output["selected_edge_count"], output["edge_viable_mask"].sum(dim=-1))
    torch.testing.assert_close(output["selected_edge_count"], torch.full((2,), 6.0))


def test_validation_uses_the_same_epoch_warmup_selection_as_training() -> None:
    model = _model()
    model.set_training_progress(epoch=0, total_epochs=10)
    batch = _batch()

    model.train()
    with torch.no_grad():
        training_output = model(batch)
    model.eval()
    with torch.no_grad():
        validation_output = model(batch)

    torch.testing.assert_close(training_output["edge_active_mask"], validation_output["edge_active_mask"])
    torch.testing.assert_close(validation_output["selected_edge_count"], torch.full((2,), 6.0))


def test_selection_progress_survives_strict_state_dict_round_trip() -> None:
    source = _model().eval()
    source.set_training_progress(epoch=0, total_epochs=10)
    batch = _batch()
    with torch.no_grad():
        expected = source(batch)["edge_active_mask"]
    state = copy.deepcopy(source.state_dict())

    restored = _model().eval()
    with torch.no_grad():
        restored(batch)
    restored.load_state_dict(state, strict=True)
    with torch.no_grad():
        actual = restored(batch)["edge_active_mask"]

    assert restored.selection_state() == {"epoch": 0, "total_epochs": 10}
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_nonviable_candidates_are_zero_everywhere_and_cannot_generate_fields(monkeypatch) -> None:
    config = _config()
    config.edge_selection_mode = "all"
    model = _model(config).eval()
    batch = _batch()
    organizer = model.organizer.exchangeable

    def fixed_assignments(
        module_tokens: torch.Tensor,
        env_tokens: torch.Tensor,
        slots: torch.Tensor,
        module_centers: torch.Tensor,
        env_coords: torch.Tensor,
        module_present: torch.Tensor,
        cfg: UnifiedForwardConfig,
        **_kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        module_row = module_tokens.new_tensor([0.70, 0.29998, 0.0, 0.0, 1.0e-5, 1.0e-5])
        env_row = env_tokens.new_tensor([0.30, 0.69998, 0.0, 0.0, 1.0e-5, 1.0e-5])
        A_mh = module_row.view(1, 1, -1).expand(module_tokens.shape[0], module_tokens.shape[1], -1)
        A_mh = A_mh * module_present.unsqueeze(-1)
        A_eh = env_row.view(1, 1, -1).expand(env_tokens.shape[0], env_tokens.shape[1], -1)
        coords = env_coords.new_zeros(env_coords.shape[0], slots.shape[1], 2)
        scale = env_coords.new_ones(env_coords.shape[0], slots.shape[1], 2)
        return A_mh, A_eh, coords, scale

    monkeypatch.setattr(organizer, "_candidate_assignments", fixed_assignments)
    with torch.no_grad():
        encoded = model.encode_and_organize(batch)
        decoded = model.decode_queries(
            batch.query_xy,
            None,
            encoded,
            encoded["global_token"],
            return_routing_maps=True,
            return_edge_fields=True,
        )

    nonviable = encoded["edge_viable_mask"] == 0
    assert torch.count_nonzero(encoded["edge_active_mask"][nonviable]) == 0
    assert torch.count_nonzero(encoded["A_mh"].transpose(1, 2)[nonviable]) == 0
    assert torch.count_nonzero(encoded["A_eh"].transpose(1, 2)[nonviable]) == 0
    assert torch.count_nonzero(encoded["hyper_state"][nonviable]) == 0
    assert torch.count_nonzero(decoded["query_hyper_attention"].transpose(1, 2)[nonviable]) == 0
    assert torch.count_nonzero(decoded["pred_field_by_edge"].transpose(1, 2)[nonviable]) == 0


def test_novelty_and_quality_never_promote_a_nonviable_candidate() -> None:
    model = _model().eval()
    organizer = model.organizer.exchangeable
    organizer.set_training_progress(epoch=10, total_epochs=10)
    module_assignment = torch.tensor([[[0.6, 0.4, 0.0], [0.5, 0.5, 0.0]]])
    env_assignment = torch.tensor([[[0.5, 0.5, 0.0], [0.6, 0.4, 0.0]]])
    quality = torch.tensor([[0.8, 0.7, 100.0]])
    codes = torch.zeros(1, 3, 24)
    viable = torch.tensor([[True, True, False]])

    selected = organizer._select_active_edges(
        module_assignment,
        env_assignment,
        quality,
        codes,
        torch.ones(1, 2),
        viable,
        model.config,
    )

    assert selected[0, 2] == 0


def test_zero_selected_entmax_support_uses_detached_unit_mass_fallback() -> None:
    assignment = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]
    )
    selected_mask = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    module_selected, module_mass = _model().organizer.exchangeable._mask_and_renormalize(
        assignment,
        selected_mask,
        torch.tensor([[1.0, 1.0, 0.0]]),
    )
    env_selected, env_mass = _model().organizer.exchangeable._mask_and_renormalize(
        assignment,
        selected_mask,
        None,
    )

    torch.testing.assert_close(module_selected.sum(dim=-1)[0, :2], torch.ones(2))
    assert module_selected.sum(dim=-1)[0, 2] == 0
    torch.testing.assert_close(env_selected.sum(dim=-1), torch.ones(1, 3))
    assert torch.equal(module_mass, torch.zeros_like(module_mass))
    assert torch.equal(env_mass, torch.zeros_like(env_mass))


def test_selected_mass_diagnostics_are_finite() -> None:
    model = _model().eval()
    with torch.no_grad():
        output = model(_batch())
    diagnostics = compute_honf_diagnostics(
        {
            "pred_field": output["pred_field"],
            "organizer_aux": output,
            "routing_aux": output,
        }
    )

    mass_keys = [key for key in diagnostics if "mass_" in key]
    assert mass_keys
    assert all(torch.isfinite(torch.tensor(diagnostics[key])) for key in mass_keys)


def test_edge_count_diagnostics_have_distinct_unambiguous_semantics() -> None:
    pred = torch.zeros(1, 2, 3)
    strength = torch.tensor([[0.20, 0.01, 0.01, 0.01]])
    shared = {
        "pred_field": pred,
        "organizer_aux": {
            "hyper_strength": strength,
            "candidate_edge_count": torch.tensor([4.0]),
            "edge_active_mask": torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
            "effective_edge_mask": torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            "empty_selected_edge_count": torch.tensor([1.0]),
        },
        "routing_aux": {"effective_query_edge_count": torch.tensor(1.5)},
    }

    diagnostics = compute_honf_diagnostics(shared)

    assert diagnostics["candidate_edge_count"] == 4.0
    assert diagnostics["selected_edge_count"] == 3.0
    assert diagnostics["viable_selected_edge_count"] == 2.0
    assert diagnostics["functional_edge_count"] == 1.0
    assert diagnostics["empty_selected_edge_count"] == 1.0
    assert diagnostics["effective_query_edge_count"] == 1.5


def test_single_candidate_and_inactive_padding_have_finite_backward() -> None:
    model = _model(_config(capacity=1, initial=1, minimum=1)).train()
    batch = _batch(module_width=12)
    output = model(batch)
    output["pred_field"].square().mean().backward()

    assert torch.equal(output["active_edge_count"], torch.ones(2))
    assert torch.isfinite(output["pred_field"]).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())

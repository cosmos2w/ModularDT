from __future__ import annotations

import copy

import pytest
import torch

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.model import HONFNeuralField
from honf_forward_core.selection.predictive_rank import (
    build_deterministic_case_probes,
    enumerate_nonempty_edge_masks,
    select_probe_fidelity_support,
)


def _config(**overrides: object) -> UnifiedForwardConfig:
    values: dict[str, object] = {
        "field_dim": 3,
        "domain_length_x": 12.0,
        "domain_length_y": 6.0,
        "coordinate_scale": [12.0, 6.0],
        "num_env_tokens_x": 4,
        "num_env_tokens_y": 2,
        "num_hyperedges": 3,
        "hidden_dim": 24,
        "dropout": 0.0,
        "decoder_mode": "enhanced_honf_pairwise",
        "pairwise_kernel_hidden_dim": 24,
    }
    values.update(overrides)
    return UnifiedForwardConfig(**values)


def _batch(width: int = 4) -> BatchData:
    centers = torch.tensor([[[1.0, 1.0], [3.0, 2.0], [7.0, 4.0], [0.0, 0.0]]])
    present = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(1, 4, 5, generator=generator) * present[..., None]
    padding = int(width) - 4
    return BatchData(
        module_centers=torch.cat([centers, torch.zeros(1, padding, 2)], dim=1),
        module_present=torch.cat([present, torch.zeros(1, padding)], dim=1),
        module_features=torch.cat([features, torch.zeros(1, padding, 5)], dim=1),
        global_context=torch.tensor([[1.0, 2.0, 3.0]]),
        query_xy=torch.tensor(
            [[[0.2, 0.3], [2.0, 1.0], [5.0, 3.0], [8.0, 2.0], [11.5, 5.5]]]
        ),
        query_time=None,
        target_field=None,
        case_name="predictive-rank",
        metadata={},
    )


def _prepared_selection(
    model: HONFNeuralField,
    batch: BatchData,
    *,
    tolerance: float,
    channel_tolerance: float,
) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
    encoded = model.encode_and_organize(batch)
    env_coords = model._environment_coords(batch.query_xy.device, batch.query_xy.dtype)
    probes, valid = build_deterministic_case_probes(
        env_coords,
        batch.module_centers,
        batch.module_present,
        module_radius=float(model.config.module_radius),
        limit=256,
        domain_length_x=float(model.config.domain_length_x),
        domain_length_y=float(model.config.domain_length_y),
    )

    def decode(
        query_xy: torch.Tensor,
        organizer: dict[str, object],
        global_token: torch.Tensor,
    ) -> torch.Tensor:
        return model.decode_queries(query_xy, None, organizer, global_token)["pred_field"]

    selected = select_probe_fidelity_support(
        encoded,
        encoded["global_token"],
        probes,
        valid,
        decode,
        relative_rms_tolerance=tolerance,
        channel_tolerance=channel_tolerance,
    )
    return selected, probes, valid


def test_nonempty_masks_are_rank_then_code_ordered() -> None:
    masks = enumerate_nonempty_edge_masks(3)
    assert masks.shape == (7, 3)
    assert masks.sum(dim=-1).tolist() == [1, 1, 1, 2, 2, 2, 3]
    assert masks[-1].all()


def test_probe_fidelity_config_is_separate_and_fixed_bank_only() -> None:
    assert _config().case_edge_selection_mode == "none"
    assert _config(case_edge_selection_mode="probe_fidelity").organizer_mode == "fixed_projection"
    with pytest.raises(ValueError, match="requires fixed_projection"):
        _config(
            organizer_mode="exchangeable_slots",
            edge_capacity=3,
            initial_active_edges=3,
            case_edge_selection_mode="probe_fidelity",
        )


def test_probe_set_is_module_permutation_and_trailing_padding_invariant() -> None:
    batch = _batch()
    env = torch.tensor([[[0.5, 0.5], [6.0, 3.0], [11.5, 5.5]]])
    reference, reference_valid = build_deterministic_case_probes(
        env, batch.module_centers, batch.module_present, module_radius=0.45
    )
    permutation = torch.tensor([2, 0, 3, 1])
    permuted, permuted_valid = build_deterministic_case_probes(
        env,
        batch.module_centers[:, permutation],
        batch.module_present[:, permutation],
        module_radius=0.45,
    )
    padded = _batch(12)
    padded_probes, padded_valid = build_deterministic_case_probes(
        env, padded.module_centers, padded.module_present, module_radius=0.45
    )
    torch.testing.assert_close(reference, permuted, rtol=0.0, atol=0.0)
    torch.testing.assert_close(reference, padded_probes, rtol=0.0, atol=0.0)
    assert torch.equal(reference_valid, permuted_valid)
    assert torch.equal(reference_valid, padded_valid)


def test_zero_tolerance_falls_back_to_all_edges_with_bit_exact_decode() -> None:
    torch.manual_seed(23)
    model = HONFNeuralField(_config()).eval()
    batch = _batch()
    with torch.no_grad():
        encoded = model.encode_and_organize(batch)
        historical = model.decode_queries(
            batch.query_xy, None, encoded, encoded["global_token"]
        )["pred_field"]
        selected, _, _ = _prepared_selection(
            model, batch, tolerance=0.0, channel_tolerance=0.0
        )
        candidate = model.decode_queries(
            batch.query_xy, None, selected, selected["global_token"]
        )["pred_field"]
    assert int(selected["predictive_edge_count"].item()) == 3
    assert int(selected["predictive_full_edge_count"].item()) == 3
    assert selected["predictive_selected_mask"].all()
    assert "predictive_edge_mask" not in selected
    torch.testing.assert_close(candidate, historical, rtol=0.0, atol=0.0)


def test_loose_tolerance_selects_one_query_routing_edge_and_caches_prepared_mask() -> None:
    torch.manual_seed(29)
    model = HONFNeuralField(_config()).eval()
    batch = _batch()
    with torch.no_grad():
        selected, _, _ = _prepared_selection(
            model, batch, tolerance=1.0e6, channel_tolerance=1.0e6
        )
        output = model.decode_queries(
            batch.query_xy,
            None,
            selected,
            selected["global_token"],
            return_routing_maps=True,
        )
        chunks = [
            model.decode_queries(chunk, None, selected, selected["global_token"])[
                "pred_field"
            ]
            for chunk in torch.tensor_split(batch.query_xy, 3, dim=1)
            if chunk.shape[1]
        ]
    mask = selected["predictive_selected_mask"] > 0
    assert int(mask.sum()) == 1
    attention = output["query_hyper_attention"]
    assert torch.count_nonzero(attention[..., ~mask[0]]) == 0
    torch.testing.assert_close(
        torch.cat(chunks, dim=1), output["pred_field"], rtol=1.0e-6, atol=1.0e-6
    )
    assert selected["A_mh"].shape[-1] == 3
    assert selected["hyper_state"].shape[-2] == 3


def test_selection_is_repeatable_and_adds_no_checkpoint_parameters() -> None:
    torch.manual_seed(31)
    model = HONFNeuralField(_config()).eval()
    batch = _batch()
    with torch.no_grad():
        first, _, _ = _prepared_selection(
            model, batch, tolerance=0.5, channel_tolerance=0.75
        )
        second, _, _ = _prepared_selection(
            model, batch, tolerance=0.5, channel_tolerance=0.75
        )
    assert torch.equal(first["predictive_selected_mask"], second["predictive_selected_mask"])
    torch.testing.assert_close(
        first["predictive_probe_relative_rms"],
        second["predictive_probe_relative_rms"],
        rtol=0.0,
        atol=0.0,
    )

    state = copy.deepcopy(model.state_dict())
    restored = HONFNeuralField(_config(case_edge_selection_mode="probe_fidelity")).eval()
    with torch.no_grad():
        restored(_batch())
    incompatible = restored.load_state_dict(state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert tuple(restored.state_dict()) == tuple(model.state_dict())


def test_single_edge_bank_selects_full_support_without_candidate_failure() -> None:
    torch.manual_seed(37)
    model = HONFNeuralField(_config(num_hyperedges=1)).eval()
    with torch.no_grad():
        selected, _, _ = _prepared_selection(
            model, _batch(), tolerance=0.01, channel_tolerance=0.02
        )
    assert int(selected["predictive_edge_count"].item()) == 1
    assert int(selected["predictive_candidate_count"].item()) == 1
    assert selected["predictive_selected_mask"].all()
    assert "predictive_edge_mask" not in selected

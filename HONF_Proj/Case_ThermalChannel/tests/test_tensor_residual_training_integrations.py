from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from channelthermal.config import ChannelThermalHONFConfig, ChannelThermalSpecificConfig
from channelthermal.model import ChannelThermalHONFModel
from channelthermal.model_support import ChannelThermalModelSupportMixin

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.model import HONFNeuralField
from honf_forward_core.training.diagnostics import compute_honf_diagnostics


def _tensor_residual_organizer_fixture() -> dict[str, torch.Tensor]:
    """Build the compact Phase-2 organizer contract used by integration tests."""

    return {
        "hyper_strength": torch.tensor(
            [[0.40, 0.30, 0.20, 0.10], [0.40, 0.30, 0.20, 0.10]],
            dtype=torch.float32,
        ),
        "edge_active_mask": torch.ones(2, 4),
        "effective_edge_mask": torch.tensor(
            [[1.0, 0.75, 0.0, 0.0], [1.0, 0.90, 0.50, 0.0]],
            dtype=torch.float32,
        ),
        "hard_case_edge_mask": torch.tensor(
            [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        "edge_survival_soft": torch.tensor(
            [[1.0, 0.4, 0.1, 0.0], [1.0, 0.8, 0.3, 0.1]],
            dtype=torch.float32,
        ),
        "case_adaptive_edge_count": torch.tensor([2.0, 3.0]),
        "case_adaptive_edge_cap": torch.tensor([4.0, 4.0]),
        "case_adaptive_stop_reached": torch.tensor([1.0, 0.0]),
        "case_adaptive_cap_hit": torch.tensor([0.0, 1.0]),
        "residual_content_factor": torch.ones(2, 5, 4),
        "residual_module_factor": torch.ones(2, 3, 4),
        "residual_environment_factor": torch.ones(2, 2, 4),
        "residual_mechanism_strength": torch.ones(2, 4),
        "residual_fraction_trace": torch.tensor(
            [[1.0, 0.30, 0.05, 0.02, 0.01], [1.0, 0.40, 0.20, 0.10, 0.02]],
            dtype=torch.float32,
        ),
        "residual_marginal_explained_fraction": torch.tensor(
            [[0.60, 0.20, 0.10, 0.05], [0.50, 0.30, 0.10, 0.05]],
            dtype=torch.float32,
        ),
        "empty_support_count": torch.tensor([0.0, 1.0]),
        # Pre-fallback rows are the support requests that needed repair;
        # post-fallback rows describe the final repaired state and remain 0.
        "pre_fallback_zero_support_module_rows": torch.tensor([0.0, 1.0]),
        "pre_fallback_zero_support_environment_rows": torch.tensor([0.0, 1.0]),
        "post_fallback_zero_support_module_rows": torch.zeros(2),
        "post_fallback_zero_support_environment_rows": torch.zeros(2),
        "residual_monotonic_violation_max": torch.tensor([0.0, 0.002]),
    }


def test_model_support_forwards_phase2_exports_and_hard_mask() -> None:
    mixin = ChannelThermalModelSupportMixin()
    mixin.config = SimpleNamespace(
        core_honf=SimpleNamespace(organizer_mode="case_adaptive_tensor_residual")
    )
    adapter = SimpleNamespace(
        module_centers=torch.zeros(2, 3, 2),
        module_present=torch.ones(2, 3),
        heat_powers=torch.ones(2, 3),
    )
    core_output = _tensor_residual_organizer_fixture()
    core_output.update(
        {
            "module_env_context": torch.zeros(2, 3, 6),
            "residual_interaction_tensor": torch.ones(2, 3, 2, 5),
        }
    )

    aux = mixin._legacy_organizer_aux(core_output, adapter, torch.zeros(2, 2, 2))

    assert torch.equal(aux["active_hyperedge_mask"], core_output["hard_case_edge_mask"])
    for key in (
        "residual_interaction_tensor",
        "residual_content_factor",
        "residual_module_factor",
        "residual_environment_factor",
        "residual_mechanism_strength",
        "residual_fraction_trace",
        "residual_marginal_explained_fraction",
        "edge_survival_soft",
        "hard_case_edge_mask",
        "case_adaptive_edge_count",
        "case_adaptive_edge_cap",
        "case_adaptive_stop_reached",
        "case_adaptive_cap_hit",
    ):
        assert key in aux

    # The full tensor is opt-in at the organizer boundary; the compatibility
    # view does not synthesize it when an ordinary output omits it.
    compact = dict(core_output)
    compact.pop("residual_interaction_tensor")
    compact_aux = mixin._legacy_organizer_aux(compact, adapter, torch.zeros(2, 2, 2))
    assert "residual_interaction_tensor" not in compact_aux


def test_phase2_training_diagnostics_are_cheap_and_explicit() -> None:
    organizer = _tensor_residual_organizer_fixture()
    organizer["residual_interaction_tensor"] = torch.full((2, 3, 2, 5), 1.0e9)
    output = {
        "pred_field": torch.zeros(2, 1, 2),
        "organizer_aux": organizer,
        "routing_aux": {},
    }

    diagnostics = compute_honf_diagnostics(output)

    assert diagnostics["case_adaptive_edge_count_mean"] == pytest.approx(2.5)
    assert diagnostics["case_adaptive_edge_count_min"] == pytest.approx(2.0)
    assert diagnostics["case_adaptive_edge_count_max"] == pytest.approx(3.0)
    assert diagnostics["case_adaptive_edge_cap_mean"] == pytest.approx(4.0)
    assert diagnostics["case_adaptive_stop_reached_fraction"] == pytest.approx(0.5)
    assert diagnostics["case_adaptive_cap_hit_fraction"] == pytest.approx(0.5)
    assert diagnostics["residual_fraction_final_mean"] == pytest.approx(0.075)
    assert diagnostics["residual_fraction_final_p95"] == pytest.approx(0.0975)
    assert diagnostics["residual_first_marginal_mean"] == pytest.approx(0.55)
    assert diagnostics["residual_second_marginal_mean"] == pytest.approx(0.25)
    assert diagnostics["residual_third_marginal_mean"] == pytest.approx(0.10)
    assert diagnostics["case_adaptive_support_probability_gap_mean"] == pytest.approx(0.2125)
    assert diagnostics["case_adaptive_support_gap_mean"] == pytest.approx(0.2125)
    assert diagnostics["case_adaptive_empty_support_count_mean"] == pytest.approx(0.5)
    assert diagnostics["case_adaptive_fallback_support_count_mean"] == pytest.approx(1.0)
    assert all(torch.isfinite(torch.tensor(value)) for value in diagnostics.values())


def test_channelthermal_facade_keeps_interaction_tensor_opt_in() -> None:
    core_config = UnifiedForwardConfig.from_dict(
        {
            "field_dim": 2,
            "domain_length_x": 4.0,
            "domain_length_y": 2.0,
            "num_env_tokens_x": 2,
            "num_env_tokens_y": 2,
            "num_hyperedges": 0,
            "organizer_mode": "case_adaptive_tensor_residual",
            "edge_capacity": 0,
            "hidden_dim": 12,
            "dropout": 0.0,
            "decoder_mode": "enhanced_honf_pairwise",
            "pairwise_aggregation_mode": "fused_query_module",
            "pairwise_kernel_hidden_dim": 12,
            "boundary_feature_mode": "none",
        }
    )
    config = ChannelThermalHONFConfig(
        core_honf=core_config,
        channelthermal=ChannelThermalSpecificConfig(field_names=["u", "temperature"]),
    )
    model = ChannelThermalHONFModel(config, attach_local_from_checkpoint=False).eval()
    inputs = {
        "module_centers": torch.tensor([[[0.5, 0.5], [2.0, 1.0]]]),
        "heat_powers": torch.tensor([[1.0, 0.4]]),
        "module_present": torch.ones(1, 2),
        "query_xy": torch.tensor([[[0.3, 0.2], [1.6, 0.7], [3.5, 1.5]]]),
    }

    with torch.no_grad():
        compact = model(**inputs, return_routing_maps=False)
        requested = model(**inputs, return_routing_maps=True)
        explicitly_requested = model(**inputs, return_organizer_diagnostics=True)

    assert "residual_interaction_tensor" not in compact["organizer_aux"]
    assert requested["organizer_aux"]["residual_interaction_tensor"].shape == (1, 2, 4, 32)
    assert explicitly_requested["organizer_aux"]["residual_interaction_tensor"].shape == (1, 2, 4, 32)


def _fixed_config() -> UnifiedForwardConfig:
    return UnifiedForwardConfig.from_dict(
        {
            "field_dim": 2,
            "domain_length_x": 4.0,
            "domain_length_y": 2.0,
            "num_env_tokens_x": 2,
            "num_env_tokens_y": 2,
            "num_hyperedges": 2,
            "organizer_mode": "fixed_projection",
            "hidden_dim": 12,
            "dropout": 0.0,
            "decoder_mode": "enhanced_honf_pairwise",
            "pairwise_aggregation_mode": "fused_query_module",
            "pairwise_kernel_hidden_dim": 12,
        }
    )


def _fixed_batch() -> BatchData:
    return BatchData(
        module_centers=torch.tensor([[[0.5, 0.5], [1.5, 0.5], [2.5, 0.5]]]),
        module_present=torch.ones(1, 3),
        module_features=torch.randn(1, 3, 4, generator=torch.Generator().manual_seed(91)),
        global_context=torch.randn(1, 5, generator=torch.Generator().manual_seed(92)),
        query_xy=torch.tensor(
            [[[0.2, 0.2], [1.2, 0.7], [2.4, 1.3], [3.1, 0.5], [3.8, 1.8]]]
        ),
        query_time=None,
        target_field=None,
        case_name="fixed-golden",
        metadata={},
    )


def test_established_fixed_projection_golden_survives_strict_state_reload() -> None:
    torch.manual_seed(93)
    config = _fixed_config()
    batch = _fixed_batch()
    source = HONFNeuralField(config).eval()
    restored = HONFNeuralField(config).eval()
    with torch.no_grad():
        expected = source(batch)["pred_field"]
        # Materialize lazy input projections before strict loading.
        restored(batch)
    restored.load_state_dict(copy.deepcopy(source.state_dict()), strict=True)
    with torch.no_grad():
        actual = restored(batch)["pred_field"]
    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)

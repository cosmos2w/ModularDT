"""Focused recovery, normalization, and intervention tests for soft-only compact mode."""
from types import SimpleNamespace
import math

import pytest
import torch

from honf_forward_core.continuous_joint.compact import CompactGroupRetrieval
from honf_forward_core.continuous_joint.unified import CompactUnifiedJointCore, UnifiedJointCore
from tests.test_continuous_compact_groups import _organization, _retrieval
from tests.test_thermal_content_hypergraph_v2 import _core_config


def _parent():
    return UnifiedJointCore(
        _core_config("A2-D"), input_dataset="thermal", routing_mode="hypergraph",
        native_backend="packed", checkpoint_pipeline=False,
    )


def test_soft_only_cosine_schedule_config_and_resume_roundtrip():
    parent = _parent()
    model = CompactUnifiedJointCore.from_unified_parent(parent, compact_mode="soft_only")
    config = model.get_config()
    assert config["compact_mode"] == "soft_only"

    assert model.set_compact_phase(1, arm="soft_only") == pytest.approx(0.1)
    middle_epoch = 100
    middle_progress = (middle_epoch - 1) / 199.0
    expected_middle = 0.1 + 0.45 * (1.0 - math.cos(math.pi * middle_progress))
    assert model.set_compact_phase(middle_epoch, arm="soft_only") == pytest.approx(expected_middle)
    assert model.set_compact_phase(200, arm="soft_only") == pytest.approx(1.0)
    assert model.set_compact_phase(201, arm="soft_only") == pytest.approx(1.0)
    assert not bool(model.compact_cut_active)
    assert not bool(model.compact_policy_frozen)
    assert all(parameter.requires_grad for parameter in model.compact_retrieval.compatibility.parameters())
    with pytest.raises(ValueError, match="never activates a hard cut"):
        model.activate_hard_cut()

    state = model.export_compact_state()
    restored = CompactUnifiedJointCore.from_config(config)
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.load_compact_state(state)
    assert restored.compact_mode == "soft_only"
    assert restored.compact_arm == "soft_only"
    assert restored.compact_alpha == pytest.approx(1.0)
    assert restored.export_compact_state() == state


def test_legacy_config_and_state_dict_without_v2_buffers_still_load_strictly():
    model = CompactUnifiedJointCore.from_unified_parent(_parent())
    config = model.get_config()
    assert "compact_mode" not in config
    old_state = model.state_dict()
    for name in (
        "compact_mode_code", "compact_tie_loss_coefficient_M", "compact_tie_loss_coefficient_E",
        "compact_capacity_loss_coefficient_M", "compact_capacity_loss_coefficient_E",
        "compact_structure_loss_scale",
    ):
        del old_state[name]
    old_compact_state = model.export_compact_state()
    for name in (
        "compact_mode", "tie_loss_coefficient_M", "tie_loss_coefficient_E",
        "capacity_loss_coefficient_M", "capacity_loss_coefficient_E", "structure_loss_scale",
    ):
        del old_compact_state[name]

    restored = CompactUnifiedJointCore.from_config(config)
    restored.load_state_dict(old_state, strict=True)
    restored.load_compact_state(old_compact_state)
    assert restored.compact_mode == "legacy"
    assert restored.compact_arm == "common"
    assert float(restored.compact_tie_loss_coefficient_M) == float(restored.compact_tie_loss_coefficient)
    assert float(restored.compact_capacity_loss_coefficient_E) == float(restored.compact_capacity_loss_coefficient)

    partial_v2 = CompactUnifiedJointCore.from_config(config).state_dict()
    del partial_v2["compact_tie_loss_coefficient_M"]
    with pytest.raises(RuntimeError, match="compact_tie_loss_coefficient_M"):
        CompactUnifiedJointCore.from_config(config).load_state_dict(partial_v2, strict=True)


def test_bank_structural_terms_are_candidate_means_with_detached_tie_probability():
    membership = torch.tensor(
        [
            [
                [0.8, 0.2, 0.4], [0.2, 0.6, 0.2],
                [0.1, 0.9, 0.2], [0.7, 0.3, 0.6],
            ],
            [
                [0.2, 0.8, 0.1], [0.7, 0.1, 0.5],
                [0.4, 0.3, 0.9], [0.3, 0.8, 0.2],
            ],
        ], dtype=torch.float64, requires_grad=True,
    )
    pair_logits = torch.tensor([[0.0, 1.0], [math.log(1 / 3), 0.0]],
                               dtype=torch.float64, requires_grad=True)
    compact = SimpleNamespace(
        membership=membership,
        pair_mask=torch.tensor([[True, True], [True, False]]),
        pair_types=torch.tensor([[0, 1], [0, -1]]),
        pair_slots=torch.tensor([[[0, 1], [2, 3]], [[0, 1], [0, 0]]]),
        pair_logits=pair_logits,
    )
    records = SimpleNamespace(
        eligible=torch.tensor([[True, True, False], [True, True, True]]),
        module_count=1,
    )

    terms = CompactGroupRetrieval.structural_bank_losses(compact, records)
    assert terms["candidate_count_M"].tolist() == [1, 1]
    assert terms["candidate_count_E"].tolist() == [1, 0]
    assert terms["valid_M"].tolist() == [True, True]
    assert terms["valid_E"].tolist() == [True, False]
    torch.testing.assert_close(terms["C_M"], torch.tensor([[0.5, 0.0], [0.525, 0.0]], dtype=torch.float64))
    torch.testing.assert_close(terms["C_E"], torch.tensor([[0.0, 0.6], [0.0, 0.0]], dtype=torch.float64))
    probability = pair_logits.sigmoid()
    torch.testing.assert_close(terms["tie_M"], torch.tensor([0.25, 0.13125], dtype=torch.float64))
    torch.testing.assert_close(terms["capacity_M"], 1.0 - torch.tensor([0.5, 0.25], dtype=torch.float64))
    torch.testing.assert_close(terms["tie_E"][0], probability[0, 1] * 0.6)
    torch.testing.assert_close(terms["capacity_E"], torch.tensor([1.0 - probability[0, 1], 0.0], dtype=torch.float64))

    tie_logit_grad = torch.autograd.grad(terms["tie_M"].sum(), pair_logits, retain_graph=True,
                                         allow_unused=True)[0]
    assert tie_logit_grad is None
    tie_membership_grad = torch.autograd.grad(terms["tie_M"].sum(), membership, retain_graph=True)[0]
    capacity_logit_grad = torch.autograd.grad(terms["capacity_M"].sum(), pair_logits)[0]
    assert torch.isfinite(tie_membership_grad).all() and tie_membership_grad.count_nonzero() > 0
    assert torch.isfinite(capacity_logit_grad).all() and capacity_logit_grad[:, 0].abs().sum() > 0
    assert torch.count_nonzero(capacity_logit_grad[:, 1]) == 0


def test_full_shared_intervention_updates_only_selected_pair_and_keeps_both_slots():
    model = _retrieval()
    organization = _organization()
    baseline = model.build(organization, alpha=0.0, cut_active=False)
    selected = tuple(int(value) for value in baseline.pair_slots[0, 0].tolist())
    shared = model.build(
        organization, alpha=0.0, cut_active=False,
        soft_intervention="full_shared", soft_intervention_pair_slots=[selected],
    )
    left, right = selected

    assert shared.allocated_count == baseline.allocated_count == 9
    assert shared.live_counts.tolist() == baseline.live_counts.tolist()
    torch.testing.assert_close(shared.group_mask, baseline.group_mask)
    torch.testing.assert_close(shared.group_types, baseline.group_types)
    torch.testing.assert_close(shared.pair_mask, baseline.pair_mask)
    torch.testing.assert_close(shared.pair_types, baseline.pair_types)
    torch.testing.assert_close(shared.pair_slots, baseline.pair_slots)
    torch.testing.assert_close(shared.pair_logits, baseline.pair_logits)
    torch.testing.assert_close(shared.controls[:, left], shared.controls[:, right], rtol=0, atol=0)
    torch.testing.assert_close(shared.centroids[:, left], shared.centroids[:, right], rtol=0, atol=0)
    torch.testing.assert_close(shared.lengths[:, left], shared.lengths[:, right], rtol=0, atol=0)
    torch.testing.assert_close(shared.membership[:, left], shared.membership[:, right], rtol=0, atol=0)
    untouched = [index for index in range(baseline.allocated_count) if index not in selected]
    torch.testing.assert_close(shared.controls[:, untouched], baseline.controls[:, untouched], rtol=0, atol=0)
    assert shared.soft_intervention == "full_shared"
    assert shared.soft_intervention_pair_mask[0].tolist() == [True, False, False]


def test_soft_only_additive_denominators_count_only_valid_scenes_per_bank():
    core = CompactUnifiedJointCore.from_unified_parent(_parent(), compact_mode="soft_only")
    core.set_compact_phase(1, arm="soft_only")
    core.set_compact_bank_loss_coefficients(
        tie_M=2.0, tie_E=3.0, capacity_M=4.0, capacity_E=5.0, structure_scale=0.5,
    )
    components = {
        "tie_M": torch.tensor([0.2, 0.4], requires_grad=True),
        "tie_E": torch.tensor([0.1, 0.3], requires_grad=True),
        "capacity_M": torch.tensor([0.3, 0.5], requires_grad=True),
        "capacity_E": torch.tensor([0.6, 0.7], requires_grad=True),
        "valid_M": torch.tensor([True, True]),
        "valid_E": torch.tensor([False, True]),
        "candidate_count_M": torch.tensor([2, 1]),
        "candidate_count_E": torch.tensor([0, 1]),
    }
    core.joint_begin_microbatch()
    core._compact_loss_entries = [(('scene-a', 'scene-b'), components)]
    result = core.joint_microbatch_result()
    losses = result["additive_losses"]

    assert set(losses) == {
        "compact_tie_M", "compact_tie_E", "compact_capacity_M", "compact_capacity_E",
    }
    assert losses["compact_tie_M"]["denominator"] == 2
    assert losses["compact_tie_E"]["denominator"] == 1
    assert losses["compact_capacity_M"]["denominator"] == 2
    assert losses["compact_capacity_E"]["denominator"] == 1
    assert losses["compact_tie_M"]["numerator"].item() == pytest.approx(0.06)
    assert losses["compact_tie_E"]["numerator"].item() == pytest.approx(0.045)
    assert result["telemetry"]["compact_eligible_pairs_M"] == 3
    assert result["telemetry"]["compact_eligible_pairs_E"] == 1
    assert result["telemetry"]["compact_valid_scenes_M"] == 2
    assert result["telemetry"]["compact_valid_scenes_E"] == 1

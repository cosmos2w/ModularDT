"""Native hard-value/soft-organizer gradient contract on GPU-capable tensors."""

from __future__ import annotations

import pytest
import torch

from honf_forward_core.config import BatchData, InterfaceFieldConfig, UnifiedForwardConfig
from honf_forward_core.interface_fields.adaptive_interaction_cover import INTERACTION_MECHANISMS
from honf_forward_core.interface_fields.core import InterfaceFieldCore
from honf_forward_core.interface_fields.input_cover_organizer import (
    InputOnlyCoverOrganizer,
    OrganizerScores,
)
from honf_forward_core.interface_fields.native_joint_shadow import hard_value_soft_organizer_forward


class _ControlledOrganizer(InputOnlyCoverOrganizer):
    """Expose one deliberately omitted QE logit while retaining native plans."""

    def __init__(self) -> None:
        super().__init__(
            state_dim=8, module_feature_dim=2, environment_feature_dim=2,
            hidden_dim=12,
        )
        self.qe_logit = torch.nn.Parameter(torch.tensor(-2.0))

    def score_cases(self, encoded, prepared_state, trees):  # type: ignore[override]
        del prepared_state
        scores = []
        for case, tree in enumerate(trees):
            device = encoded.module_tokens.device
            nodes = len(tree.nodes)
            modules = encoded.module_present.shape[1]
            environments = encoded.env_coords.shape[1]
            module_logits = torch.full((nodes, modules), 4.0, device=device)
            environment_logits = torch.full((nodes, environments), 4.0, device=device)
            qe_logits = environment_logits.clone()
            qe_logits = torch.cat((
                self.qe_logit.expand(nodes, 1), qe_logits[:, 1:],
            ), dim=1)
            typed = {
                mechanism: (module_logits if mechanism in {"MM", "EM", "QM"} else environment_logits)
                for mechanism in INTERACTION_MECHANISMS
            }
            typed["QE"] = qe_logits
            scores.append(OrganizerScores(
                torch.full((nodes,), -4.0, device=device),
                module_logits, qe_logits, typed,
            ))
        return tuple(scores)


def _case(
    device: torch.device, *, checkpointing: bool = False,
) -> tuple[InterfaceFieldCore, BatchData, _ControlledOrganizer]:
    torch.manual_seed(3107)
    module_centers = torch.tensor([[[0.25, 0.5, 0.5], [1.5, 0.75, 0.5]]], device=device)
    env_coords = torch.tensor([[
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
    ]], device=device)
    queries = torch.tensor([[
        [0.0, 0.0, 0.0], [0.5, 0.25, 0.0], [1.0, 0.5, 0.0], [1.5, 0.75, 0.0],
    ]], device=device, requires_grad=True)
    batch = BatchData(
        module_centers=module_centers,
        module_present=torch.ones((1, 2), device=device),
        module_features=torch.randn((1, 2, 2), device=device),
        global_context=torch.randn((1, 5), device=device),
        query_xy=queries,
        query_time=None,
        target_field=torch.zeros((1, 4, 3), device=device),
        case_name="native-shadow-gradient-check",
        env_coords=env_coords,
        env_features=torch.randn((1, 4, 2), device=device),
        env_weights=torch.tensor([[1.0, 2.0, 1.5, 0.5]], device=device),
        query_features=torch.randn((1, 4, 2), device=device),
        metadata=[{}],
    )
    config = UnifiedForwardConfig(
        forward_architecture="dense_pairwise_field",
        field_dim=3,
        spatial_dim=3,
        coordinate_scale=[2.0, 2.0, 1.0],
        hidden_dim=8,
        dropout=0.0,
        geometry_mode="nonperiodic",
        boundary_feature_mode="none",
        interface_model=InterfaceFieldConfig(
            message_hidden_dim=12,
            attention_heads=2,
            coarse_latent_count=2,
            coarse_blocks=1,
            relative_fourier_frequencies=1,
            receiver_chunk_size=2,
            activation_checkpointing=checkpointing,
        ),
    )
    core = InterfaceFieldCore(config).to(device).eval()
    core.backend.set_cover_mode("external")
    core.backend.set_cover_executor("dense_masked")
    organizer = _ControlledOrganizer().to(device).eval()
    return core, batch, organizer


def test_native_shadow_exact_hard_values_and_physical_gradients() -> None:
    assert torch.cuda.is_available(), "native shadow validation requires the selected GPU"
    device = torch.device("cuda:0")
    core, batch, organizer = _case(device)
    core.set_native_interaction_policy(organizer)
    encoded = core.encode_case(batch)
    result = hard_value_soft_organizer_forward(
        core, encoded, encoded.module_tokens, organizer,
        batch.query_xy, batch.query_features, receiver_chunk_size=2,
    )
    assert torch.equal(result.prediction, result.hard_prediction)
    assert not result.hard_plans[0].permission_matrix("QE").requires_grad
    assert result.soft_plans[0].permission_matrix("QE").requires_grad
    target = result.hard_prediction.detach() + 0.1
    joint_loss = (result.prediction - target).square().mean()
    hard_loss = (result.hard_prediction - target).square().mean()
    parameters = tuple(
        parameter for name, parameter in core.named_parameters()
        if parameter.requires_grad and not name.startswith("native_interaction_policy.")
    )
    joint_gradients = torch.autograd.grad(
        joint_loss, (*parameters, batch.query_xy, organizer.qe_logit),
        retain_graph=True, allow_unused=True,
    )
    hard_gradients = torch.autograd.grad(
        hard_loss, (*parameters, batch.query_xy),
        retain_graph=True, allow_unused=True,
    )
    for actual, expected in zip(joint_gradients[:-1], hard_gradients, strict=True):
        if expected is None:
            assert actual is None
        else:
            assert actual is not None
            torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-7)
    assert joint_gradients[-1] is not None
    assert abs(float(joint_gradients[-1])) > 1.0e-10


def test_current_omitted_qe_logit_has_zero_local_gradient_despite_restore() -> None:
    assert torch.cuda.is_available(), "native attention validation requires the selected GPU"
    device = torch.device("cuda:0")
    core, batch, organizer = _case(device)
    encoded = core.encode_case(batch)
    trees = core.backend.build_case_trees(encoded)
    scores = organizer.score_cases(encoded, {}, trees)
    hard = organizer.plans_from_scores(scores, encoded, trees, hard=True)[0]
    restored_qe = hard.permission_matrix("QE").clone()
    restored_qe[0, 0] = 1.0
    restored = hard.with_permission("QE", restored_qe)
    with torch.no_grad():
        restored_prepared = core.prepare(
            encoded, encoded.module_tokens, fixed_cover_plans=(restored,),
        )
        target = core.decode_queries(
            restored_prepared, batch.query_xy, batch.query_features,
        )["pred_field"].detach()
    straight_through = organizer.plans_from_scores(
        scores, encoded, trees, hard=True, straight_through_hard=True,
    )[0]
    omitted_prepared = core.prepare(
        encoded, encoded.module_tokens, fixed_cover_plans=(straight_through,),
    )
    omitted_prediction = core.decode_queries(
        omitted_prepared, batch.query_xy, batch.query_features,
    )["pred_field"]
    loss = (omitted_prediction - target).square().mean()
    current_gradient = torch.autograd.grad(loss, organizer.qe_logit)[0]
    assert float(loss.detach()) > 1.0e-12
    assert float(current_gradient.detach()) == 0.0

    shadow = hard_value_soft_organizer_forward(
        core, encoded, encoded.module_tokens, organizer,
        batch.query_xy, batch.query_features,
    )
    repaired_loss = (shadow.prediction - target).square().mean()
    repaired_gradient = torch.autograd.grad(repaired_loss, organizer.qe_logit)[0]
    assert abs(float(repaired_gradient.detach())) > 1.0e-10


def test_shadow_restores_checkpointing_and_hard_training_gradient() -> None:
    assert torch.cuda.is_available(), "checkpointed native validation requires the selected GPU"
    device = torch.device("cuda:0")
    core, batch, organizer = _case(device, checkpointing=True)
    core.train()
    encoded = core.encode_case(batch)
    result = hard_value_soft_organizer_forward(
        core, encoded, encoded.module_tokens, organizer,
        batch.query_xy, batch.query_features,
    )
    assert core.backend.activation_checkpointing
    target = result.hard_prediction.detach() + 0.1
    joint_loss = (result.prediction - target).square().mean()
    hard_loss = (result.hard_prediction - target).square().mean()
    physical = tuple(parameter for parameter in core.parameters() if parameter.requires_grad)
    actual = torch.autograd.grad(joint_loss, physical, retain_graph=True, allow_unused=True)
    expected = torch.autograd.grad(hard_loss, physical, retain_graph=True, allow_unused=True)
    for left, right in zip(actual, expected, strict=True):
        if right is None:
            assert left is None
        else:
            assert left is not None
            torch.testing.assert_close(left, right, rtol=1.0e-5, atol=1.0e-7)


def test_native_shadow_rejects_full_access_bypass_and_packed_executor() -> None:
    assert torch.cuda.is_available(), "native shadow guard validation requires the selected GPU"
    core, batch, organizer = _case(torch.device("cuda:0"))
    encoded = core.encode_case(batch)
    core.backend.set_cover_mode("full_access")
    with pytest.raises(ValueError, match="external cover mode"):
        hard_value_soft_organizer_forward(
            core, encoded, encoded.module_tokens, organizer,
            batch.query_xy, batch.query_features,
        )
    core.backend.set_cover_mode("external")
    core.backend.set_cover_executor("packed")
    with pytest.raises(ValueError, match="dense-masked scientific executor"):
        hard_value_soft_organizer_forward(
            core, encoded, encoded.module_tokens, organizer,
            batch.query_xy, batch.query_features,
        )

from __future__ import annotations

import copy
import json
from tempfile import TemporaryDirectory

import torch
from torch import nn
from pathlib import Path
import sys

from channelthermal.response_control.active_packet import (
    _ContextualDirectPairScorer,
    ThermalCoverPlanBuilder,
    make_direct_scorer,
)
from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    CaseLocalReceiverTree,
    MechanismPlan,
    ReceiverAnchorUniverse,
)
from honf_forward_core.interface_fields.input_cover_organizer import InputOnlyCoverOrganizer
from honf_forward_core.interface_fields.types import EncodedInterfaceCase

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from run_active_packet_forward import _FrontierSchedule, _budgeted_route_status
import run_active_packet_stage_c as stage_c


class _Core(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"spatial_dim": 2, "hidden_dim": 8})()
        self.receiver_fourier = nn.Identity()


def _fixture() -> tuple[_Core, EncodedInterfaceCase, CaseLocalReceiverTree]:
    generator = torch.Generator().manual_seed(7801)
    encoded = EncodedInterfaceCase(
        module_tokens=torch.randn((1, 3, 8), generator=generator),
        env_tokens=torch.randn((1, 5, 8), generator=generator),
        global_token=torch.randn((1, 8), generator=generator),
        module_centers=torch.tensor([[[0.6, 0.4], [2.1, 1.4], [0.0, 0.0]]]),
        env_coords=torch.tensor([[[0.0, 0.0], [0.7, 0.9], [1.3, 0.5], [2.0, 1.0], [2.8, 1.8]]]),
        module_present=torch.tensor([[1.0, 1.0, 0.0]]),
        module_features=torch.randn((1, 3, 2), generator=generator),
        env_features=torch.randn((1, 5, 3), generator=generator),
        env_weights=torch.ones((1, 5)),
        coordinate_scale=torch.tensor([[[3.0, 2.0]]]),
    )
    coords = torch.cat((encoded.env_coords[0], encoded.module_centers[0, :2]), dim=0)
    weights = torch.cat((encoded.env_weights[0], torch.ones(2)))
    roles = torch.tensor([0, 0, 0, 0, 0, 1, 1])
    universe = ReceiverAnchorUniverse(coords, weights, roles, encoded.coordinate_scale[0, 0])
    tree = CaseLocalReceiverTree.build(universe, max_nodes=15, min_leaf_anchors=1)
    return _Core(), encoded, tree


def test_thermal_grouped_and_direct_builders_preserve_budget_and_gradient_contracts() -> None:
    core, encoded, tree = _fixture()
    budgets = {"QE": 0.35, "ME": 0.35}
    organizer = InputOnlyCoverOrganizer(
        state_dim=8,
        module_feature_dim=2,
        environment_feature_dim=3,
        hidden_dim=16,
    )
    grouped_soft = ThermalCoverPlanBuilder(
        core=core,
        budget_fractions=budgets,
        mode="G",
        organizer=organizer,
        extra_route="ME",
    )
    grouped_soft.hard = False
    grouped_plan = grouped_soft(
        encoded,
        encoded.module_tokens,
        (tree,),
    )[0]
    assert {key.canonical_name for key in grouped_plan.permissions} == {"ME", "QE"}
    assert grouped_plan.explicit_bypass_keys == ("MM", "EM", "QM")
    grouped_plan.permission_matrix("QE").sum().backward()
    assert any(parameter.grad is not None for parameter in organizer.parameters())
    assert isinstance(grouped_soft.last_records[0]["routes"]["ME"].sparse_success, bool)
    grouped_record = grouped_soft.last_records[0]
    grouped_totals = grouped_record["all_mechanism_route_work"]
    assert set(grouped_totals) == {"MM", "ME", "EM", "QM", "QE"}
    assert grouped_record["all_mechanisms_total"]["achieved_work"] == sum(
        row["achieved_work"] for row in grouped_totals.values()
    )
    assert grouped_record["all_mechanisms_total"]["full_access_work"] == sum(
        row["full_access_work"] for row in grouped_totals.values()
    )
    assert all(
        grouped_totals[key]["executor"] == "explicit_full_access_bypass"
        for key in ("MM", "EM", "QM")
    )

    p_scorer = make_direct_scorer(core, device="cpu", hidden_dim=16)
    direct_soft = ThermalCoverPlanBuilder(
        core=core,
        budget_fractions=budgets,
        mode="P",
        direct_scorer=p_scorer,
        extra_route="ME",
    )
    direct_soft.hard = False
    direct_plan = direct_soft(encoded, encoded.module_tokens, (tree,))[0]
    assert direct_plan.direct_pair_access_for("ME") is not None
    assert direct_plan.direct_pair_policy_for("QE") is not None
    assert direct_plan.explicit_bypass_keys == ("MM", "EM", "QM")
    direct_record = direct_soft.last_records[0]
    direct_totals = direct_record["all_mechanism_route_work"]
    assert direct_record["all_mechanisms_total"]["achieved_work"] == sum(
        row["achieved_work"] for row in direct_totals.values()
    )
    assert direct_record["all_mechanisms_total"]["full_access_work"] == sum(
        row["full_access_work"] for row in direct_totals.values()
    )
    qe_receivers = encoded.env_coords[0, :3]
    qe_features = core.receiver_fourier(qe_receivers[None])[0]
    qe_access = direct_plan.access_for(
        "QE", qe_receivers, receiver_features=qe_features, module_present=encoded.module_present[0]
    )
    me_access = direct_plan.access_for(
        "ME",
        encoded.module_centers[0],
        receiver_features=None,
        module_present=encoded.module_present[0],
    )
    (qe_access.sum() + me_access.sum()).backward()
    assert any(parameter.grad is not None for parameter in p_scorer.parameters())
    for mechanism in budgets:
        record = direct_soft.last_records[0]["routes"][mechanism]
        assert record.full_access_work > 0
        assert record.achieved_work <= record.full_access_work
        assert isinstance(record.sparse_success, bool)


def test_stage_c_rewiring_preserves_packet_sizes_and_source_degrees() -> None:
    _, encoded, tree = _fixture()
    plan = MechanismPlan.full_access(
        tree,
        encoded.module_present[0],
        int(encoded.env_coords.shape[1]),
    )
    for node_index, node in enumerate(tree.nodes):
        if not node.is_leaf:
            plan = plan.with_split(node_index, 1.0)
    active_packet_rows = torch.nonzero(
        (tree.access(tree.universe.coordinates, plan.split_gates) > 0).any(dim=0),
        as_tuple=False,
    ).flatten()
    assert active_packet_rows.numel() >= 2

    # Seed two active packet rows with a valid 2x2 switch for QE and Thermal's
    # selected extra ME route. The test verifies both the intervention and its
    # exact row-size/source-degree invariants.
    first, second = (int(value) for value in active_packet_rows[:2].tolist())
    for mechanism in ("QE", "ME"):
        membership = torch.zeros_like(plan.permission_matrix(mechanism))
        membership[first, 0] = 1.0
        membership[second, 1] = 1.0
        plan = plan.with_permission(mechanism, membership)

    for mechanism in ("QE", "ME"):
        before = plan.permission_matrix(mechanism)
        rewired, info = stage_c._rewire_membership(plan, mechanism, seed=731)
        after = rewired.permission_matrix(mechanism)
        assert info["accepted_switches"] > 0
        assert torch.equal(before.sum(dim=1), after.sum(dim=1))
        assert torch.equal(before.sum(dim=0), after.sum(dim=0))

def test_extended_thermal_budget_curriculum_matches_declared_mix_without_forcing_k() -> None:
    schedule = _FrontierSchedule(
        warm_updates=50,
        seed=2317,
        extra_route="MM",
        extended_curriculum=True,
    )
    observed = []
    for update in range(300):
        schedule.current_update = update
        observed.append(schedule.budgets()["QE"])
    assert observed[:50] == [1.0] * 50
    assert observed[50:200] == [0.9] * 150
    assert {value: observed[200:300].count(value) for value in (1.0, 0.9, 0.75)} == {
        1.0: 20,
        0.9: 60,
        0.75: 20,
    }
    replay = _FrontierSchedule(
        warm_updates=50,
        seed=2317,
        extra_route="MM",
        extended_curriculum=True,
    )
    replayed = []
    for update in range(300):
        replay.current_update = update
        replayed.append(replay.budgets()["QE"])
    assert replayed == observed


def test_budgeted_route_status_keeps_absence_or_full_access_out_of_sparse_success() -> None:
    from types import SimpleNamespace

    assert _budgeted_route_status(SimpleNamespace(
        requested_fraction=0.9,
        full_unique_pairs=0,
        selected_unique_pairs=0,
        sparse_success=False,
    )) == "no_eligible_pairs"
    assert _budgeted_route_status(SimpleNamespace(
        requested_fraction=0.9,
        full_unique_pairs=10,
        selected_unique_pairs=0,
        sparse_success=False,
    )) == "empty_projection_no_sparse_success"
    assert _budgeted_route_status(SimpleNamespace(
        requested_fraction=0.9,
        full_unique_pairs=10,
        selected_unique_pairs=10,
        sparse_success=False,
    )) == "no_savings_at_budget"
    assert _budgeted_route_status(SimpleNamespace(
        requested_fraction=0.9,
        full_unique_pairs=10,
        selected_unique_pairs=7,
        sparse_success=True,
    )) == "sparse_success"


def test_u100_to_u300_resume_preserves_update_indexed_budget_and_frontier_stream() -> None:
    _core, _encoded, tree = _fixture()
    short = _FrontierSchedule(warm_updates=50, seed=2317, extra_route="MM")
    extended_g = _FrontierSchedule(
        warm_updates=50, seed=2317, extra_route="MM", extended_curriculum=True
    )
    extended_p = _FrontierSchedule(
        warm_updates=50, seed=2317, extra_route="MM", extended_curriculum=True
    )
    # Existing updates 1-100 and the primary segment through update 200 are
    # indexed from completed updates, not rescaled by the requested cap.
    for completed in range(200):
        short.current_update = completed
        extended_g.current_update = completed
        extended_p.current_update = completed
        assert short.budgets() == extended_g.budgets() == extended_p.budgets()
        assert short.cut(tree) == extended_g.cut(tree) == extended_p.cut(tree)
    # The post-200 stress curriculum is replayed identically for both arms.
    observed = []
    for completed in range(200, 300):
        extended_g.current_update = completed
        extended_p.current_update = completed
        budget_g = extended_g.budgets()
        budget_p = extended_p.budgets()
        cut_g = extended_g.cut(tree)
        cut_p = extended_p.cut(tree)
        assert budget_g == budget_p
        assert cut_g == cut_p
        observed.append(budget_g["QE"])
    assert {value: observed.count(value) for value in (1.0, 0.9, 0.75)} == {
        1.0: 20,
        0.9: 60,
        0.75: 20,
    }


def test_contextual_direct_scorer_checkpoint_preserves_values_and_parameter_gradients() -> None:
    core, encoded, _tree = _fixture()
    torch.manual_seed(831)
    checkpointed_scorer = make_direct_scorer(core, device="cpu", hidden_dim=16)
    reference_scorer = copy.deepcopy(checkpointed_scorer)
    contextual = _ContextualDirectPairScorer(checkpointed_scorer, encoded.global_token[0])
    receiver_xy = torch.randn((9, 2), generator=torch.Generator().manual_seed(91))
    source_xy = encoded.env_coords[0]
    receiver_features = core.receiver_fourier(receiver_xy[None])[0]
    source_features = encoded.env_tokens[0]

    actual = contextual(
        receiver_features,
        source_features,
        receiver_xy,
        source_xy,
        mechanism="QE",
        budget_fraction=0.9,
    )
    actual_loss = actual.square().mean() + 0.13 * actual.sum()
    actual_loss.backward()

    global_token = encoded.global_token[0]
    reference = reference_scorer(
        torch.cat((receiver_features, global_token.expand(receiver_features.shape[0], -1)), dim=-1),
        torch.cat((source_features, global_token.expand(source_features.shape[0], -1)), dim=-1),
        receiver_xy,
        source_xy,
        mechanism="QE",
        budget_fraction=0.9,
    )
    reference_loss = reference.square().mean() + 0.13 * reference.sum()
    reference_loss.backward()

    torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)
    for (actual_name, actual_parameter), (reference_name, reference_parameter) in zip(
        checkpointed_scorer.named_parameters(), reference_scorer.named_parameters(), strict=True
    ):
        assert actual_name == reference_name
        if actual_parameter.grad is None or reference_parameter.grad is None:
            assert actual_parameter.grad is None and reference_parameter.grad is None
            continue
        torch.testing.assert_close(actual_parameter.grad, reference_parameter.grad, rtol=1e-6, atol=1e-7)


def test_contextual_direct_scorer_preserves_query_chunk_outputs_and_gradients() -> None:
    core, encoded, _tree = _fixture()
    torch.manual_seed(991)
    scorer = make_direct_scorer(core, device="cpu", hidden_dim=16)
    contextual = _ContextualDirectPairScorer(scorer, encoded.global_token[0])
    generator = torch.Generator().manual_seed(992)
    receiver_xy = torch.randn((700, 2), generator=generator)
    receiver_features = core.receiver_fourier(receiver_xy[None])[0]
    source_xy = encoded.env_coords[0]
    source_features = encoded.env_tokens[0]
    outputs: dict[int, torch.Tensor] = {}
    gradients: dict[int, torch.Tensor] = {}

    for query_chunk in (512, 2048):
        scorer.zero_grad(set_to_none=True)
        output = torch.cat(
            [
                contextual(
                    receiver_features[start : start + query_chunk],
                    source_features,
                    receiver_xy[start : start + query_chunk],
                    source_xy,
                    mechanism="QE",
                    budget_fraction=0.9,
                )
                for start in range(0, receiver_xy.shape[0], query_chunk)
            ],
            dim=0,
        )
        output.square().mean().backward()
        outputs[query_chunk] = output.detach().clone()
        gradients[query_chunk] = torch.cat(
            [parameter.grad.detach().flatten() for parameter in scorer.parameters() if parameter.grad is not None]
        )

    torch.testing.assert_close(outputs[512], outputs[2048], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(gradients[512], gradients[2048], rtol=1e-6, atol=1e-7)


def test_stage_c_forward_attempt_ledger_counts_failed_and_interrupted_calls() -> None:
    previous = (
        stage_c._FORWARD_ATTEMPT_LEDGER,
        stage_c._FORWARD_ATTEMPT_PHASE,
        stage_c._FORWARD_ATTEMPT_INVOCATION,
        stage_c._FORWARD_ATTEMPT_NEXT_ID,
    )
    try:
        with TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "native_state_forward_attempts.jsonl"
            stage_c._configure_forward_attempt_ledger(ledger, "fit")
            completed = stage_c._begin_native_state_forward(
                "train-family-a", "baseline", torch.device("cpu")
            )
            stage_c._end_native_state_forward(completed, "completed")
            failed = stage_c._begin_native_state_forward(
                "train-family-a", "i_plus", torch.device("cpu")
            )
            stage_c._end_native_state_forward(failed, "failed", RuntimeError("bounded failure"))

            stage_c._configure_forward_attempt_ledger(ledger, "held-eval")
            stage_c._begin_native_state_forward("held-family", "baseline", torch.device("cpu"))
            summary = stage_c._summarize_forward_attempt_ledger(ledger)
            assert summary["attempted_native_state_forwards"] == 3
            assert summary["completed_native_state_forwards"] == 1
            assert summary["failed_native_state_forwards"] == 1
            assert summary["pending_or_interrupted_native_state_forwards"] == 1
            assert summary["by_phase"]["fit"]["attempted"] == 2
            assert summary["by_phase"]["held-eval"]["pending_or_interrupted"] == 1

            output = Path(temporary) / "ledger_summary.json"
            stage_c._write_json(output, summary, immutable=True)
            assert json.loads(output.read_text(encoding="utf-8"))["attempted_native_state_forwards"] == 3
    finally:
        (
            stage_c._FORWARD_ATTEMPT_LEDGER,
            stage_c._FORWARD_ATTEMPT_PHASE,
            stage_c._FORWARD_ATTEMPT_INVOCATION,
            stage_c._FORWARD_ATTEMPT_NEXT_ID,
        ) = previous

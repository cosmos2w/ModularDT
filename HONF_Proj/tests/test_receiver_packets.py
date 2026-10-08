from dataclasses import replace

import pytest
import torch

from honf_forward_core.interface_fields import receiver_packets as packet_module
from honf_forward_core.interface_fields.receiver_packets import (
    ActionDomain,
    InteractionBinding,
    PacketUnion,
    ReceiverPacket,
    ReceiverPairScorer,
    apply_affine_packet_increment,
    apply_differentiable_fixed_route,
    build_receiver_pair_features,
    compile_receiver_request,
    deterministic_cover,
    exact_teacher_sensitivities,
    execution_source_mask,
    fit_receiver_pair_scorer,
    geometry_equal_k_controls,
    prepared_context_fingerprints,
    propose_receiver_packets,
    response_rms_budget,
    scorer_parameter_sha256,
    union_receiver_packets,
)
from honf_forward_core.interface_fields.source_response_operator import SourceResponseOperator


def _binding(
    *,
    prepared_context=None,
    receiver_ids=("r0", "r1"),
    receiver_coordinates=None,
    capability="affine_scalar_increment",
):
    if prepared_context is None:
        _operator, prepared_context, _receivers = _operator_and_state()
    geometry_fingerprint, context_fingerprint = prepared_context_fingerprints(prepared_context)
    coordinate_by_id = {
        "r0": (0.05, 0.05),
        "r1": (0.4, 0.8),
        "r2": (0.9, 0.9),
        "r3": (0.2, 0.8),
    }
    return InteractionBinding(
        dataset_id="test-dataset",
        scene_id="layout-17",
        geometry_fingerprint=geometry_fingerprint,
        context_fingerprint=context_fingerprint,
        source_ids=("s0", "s1", "s2"),
        source_slots=(0, 1, 2),
        prepared_source_ids=("0", "1", "2"),
        receiver_ids=receiver_ids,
        receiver_coordinates=(
            tuple(coordinate_by_id[value] for value in receiver_ids)
            if receiver_coordinates is None
            else receiver_coordinates
        ),
        environment_context_ids=("env-global-0", "env-global-1"),
        output_roles=("temperature",),
        control_roles=("heat",),
        units=(("control", "W"), ("output.temperature", "K"), ("position", "m")),
        capability=capability,
        ancestry=("source-source-context", "source-environment-context", "global-context"),
    )


def _action(binding):
    return ActionDomain("heat", "W", binding.source_ids, (0.2, 0.1, 0.0), require_balanced=False)


def _packet(binding, action, receiver_ids, source_ids, *, packet_id="p"):
    return ReceiverPacket(
        packet_id,
        binding.fingerprint,
        action.fingerprint,
        tuple(receiver_ids),
        tuple(source_ids),
        binding.environment_context_ids,
        binding.ancestry,
        binding.output_roles,
        binding.control_roles,
        binding.units,
        residual_margin=0.1,
    )


def test_packet_union_deduplicates_overlap_and_full_falls_back_deterministically():
    binding = _binding()
    action = _action(binding)
    packets = (
        _packet(binding, action, ("r0",), ("s1", "s0"), packet_id="left"),
        _packet(binding, action, ("r0",), ("s1", "s2"), packet_id="right"),
        _packet(binding, action, ("r1",), ("s2",), packet_id="other"),
    )
    union = union_receiver_packets(binding, action, packets, ("r0", "r1"))
    assert union.selected_source_ids == (("s0", "s1", "s2"), ("s2",))
    assert not any(union.fallback)
    assert union.source_keep.tolist() == [[True, True, True], [False, False, True]]

    bad_packet = _packet(binding, action, ("r0",), ("s0",), packet_id="bad")
    bad_action = ActionDomain("heat", "W", binding.source_ids, (0.1, 0.1, 0.0))
    invalid = union_receiver_packets(binding, bad_action, (bad_packet,), ("r0", "r1"))
    assert invalid.fallback == (True, True)
    assert invalid.source_keep.tolist() == [[True, True, True], [True, True, True]]
    assert invalid.reasons == ("action_domain_mismatch", "no_packet_covers_receiver")


def test_packet_binding_checks_complete_context_ancestry_and_physical_action_units():
    binding = _binding()
    action = _action(binding)
    packet = _packet(binding, action, ("r0",), ("s0",))
    altered_ancestry = ReceiverPacket(
        packet.packet_id,
        packet.binding_fingerprint,
        packet.action_domain_fingerprint,
        packet.receiver_ids,
        packet.source_ids,
        packet.environment_ancestry_ids,
        ("source-source-context", "missing-environment-edge"),
        packet.output_roles,
        packet.control_roles,
        packet.units,
        residual_margin=packet.residual_margin,
    )
    mismatch = union_receiver_packets(binding, action, (altered_ancestry,), ("r0",))
    assert mismatch.fallback == (True,)
    assert mismatch.reasons == ("context_ancestry_mismatch",)

    wrong_unit = ActionDomain("heat", "kW", binding.source_ids, (0.2, 0.1, 0.0))
    wrong_unit_packet = _packet(binding, wrong_unit, ("r0",), ("s0",), packet_id="bad-unit")
    mismatch = union_receiver_packets(binding, wrong_unit, (wrong_unit_packet,), ("r0",))
    assert mismatch.fallback == (True,)
    assert mismatch.reasons == ("action_control_or_unit_binding_mismatch",)


def test_union_validates_each_packet_once_and_rechecks_on_the_next_call(monkeypatch):
    binding = _binding()
    action = _action(binding)
    packet = _packet(binding, action, binding.receiver_ids, ("s0",))
    original = packet_module._packet_valid
    calls = []

    def counted(packet, binding, action):
        calls.append(packet.packet_id)
        return original(packet, binding, action)

    monkeypatch.setattr(packet_module, "_packet_valid", counted)
    valid = union_receiver_packets(binding, action, (packet,), binding.receiver_ids)
    assert not any(valid.fallback)
    assert calls == [packet.packet_id]
    changed_action = ActionDomain("heat", "W", binding.source_ids, (0.1, 0.1, 0.0))
    stale = union_receiver_packets(binding, changed_action, (packet,), binding.receiver_ids)
    assert all(stale.fallback)
    assert stale.reasons == ("action_domain_mismatch",) * len(binding.receiver_ids)
    assert calls == [packet.packet_id, packet.packet_id]


def test_exact_teacher_separates_raw_sensitivity_from_action_mass_and_budget():
    kernel = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
    weights = torch.tensor([0.25, 0.75])
    radii = torch.tensor([0.1, 0.5])
    raw, action_mass = exact_teacher_sensitivities(kernel, weights, radii)
    torch.testing.assert_close(raw, torch.tensor([0.25 + 6.75, 1.0]).sqrt())
    torch.testing.assert_close(action_mass, raw * radii)
    assert response_rms_budget(0.01, 0.4) == pytest.approx(0.004)
    assert response_rms_budget(0.02, 0.4) == pytest.approx(0.008)
    with pytest.raises(ValueError, match="exactly 1% or 2%"):
        response_rms_budget(0.03, 0.4)

    selected, omitted = deterministic_cover(action_mass, ("a", "b"), 0.1)
    assert selected == ("a", "b")
    assert omitted == 0


def test_equal_k_geometry_controls_keep_protected_near_source_and_physical_ties():
    coordinates = torch.tensor([[0.1, 0.0], [-0.1, 0.0], [0.3, 0.0], [-0.3, 0.0]])
    controls = geometry_equal_k_controls(
        coordinates,
        torch.zeros(2),
        ("d", "b", "c", "a"),
        torch.tensor([1.0, 0.0]),
        3,
        protected_source_ids=("c",),
    )
    assert len(controls["nearest"]) == len(controls["upstream"]) == 3
    assert "c" in controls["nearest"] and "c" in controls["upstream"]
    assert controls["nearest"] != controls["upstream"]


def test_context_geometry_pair_features_support_three_dimensions_and_typed_roles():
    features = build_receiver_pair_features(
        source_states=torch.ones(1, 3, 4),
        global_state=torch.zeros(1, 4),
        source_coordinates=torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        receiver_coordinates=torch.tensor([[[0.2, 0.3, 0.4], [0.8, 0.2, 0.1]]]),
        domain_lengths=torch.ones(1, 3),
        source_lengths=torch.full((1, 3), 0.2),
        receiver_features=torch.ones(1, 2, 2),
        output_type_features=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        control_type_features=torch.ones(1, 2, 1),
    )
    assert features.shape[:3] == (1, 2, 3)
    assert torch.isfinite(features).all()


def test_pair_proposer_features_include_prepared_environment_state_pool():
    kwargs = {
        "source_states": torch.ones(1, 2, 4),
        "global_state": torch.zeros(1, 4),
        "source_coordinates": torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]),
        "receiver_coordinates": torch.tensor([[[0.5, 0.5]]]),
        "domain_lengths": torch.ones(1, 2),
        "source_lengths": torch.full((1, 2), 0.1),
        "receiver_features": torch.ones(1, 1, 1),
        "output_type_features": torch.ones(1, 1, 1),
        "control_type_features": torch.ones(1, 1, 1),
    }
    no_environment = build_receiver_pair_features(**kwargs)
    environment = build_receiver_pair_features(
        **kwargs,
        environment_states=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]]]),
        environment_measure=torch.tensor([[0.25, 0.75]]),
    )
    torch.testing.assert_close(
        environment[..., 8:12], torch.tensor([[[[0.25, 2.25, 0.0, 0.0], [0.25, 2.25, 0.0, 0.0]]]])
    )
    torch.testing.assert_close(no_environment[..., 8:12], torch.zeros_like(no_environment[..., 8:12]))
    assert environment.shape[-1] == no_environment.shape[-1]


def test_fit_uses_whole_layout_calibration_and_proposal_needs_no_full_kernel():
    torch.manual_seed(13)
    scorer = ReceiverPairScorer(input_width=5, hidden_width=8)
    assert scorer.parameter_count <= 50_000
    features = torch.randn(48, 5)
    target = torch.linspace(0.01, 0.8, 48)
    layout_ids = tuple(f"layout-{index // 4:02d}" for index in range(48))
    receipt = fit_receiver_pair_scorer(scorer, features, target, layout_ids, epochs=2, seed=7, batch_size=12)
    assert receipt.parameter_count == scorer.parameter_count
    assert receipt.epochs == 2 and receipt.optimizer_updates > 0
    assert len(receipt.parameter_sha256) == 64
    assert receipt.parameter_sha256 == scorer_parameter_sha256(scorer)
    assert set(receipt.training_layout_ids).isdisjoint(receipt.calibration_layout_ids)
    assert receipt.train_only_log_floor > 0
    assert receipt.train_only_residual_margin >= 0

    binding = _binding()
    action = _action(binding)
    proposal_features = torch.randn(2, 3, 5)
    packets = propose_receiver_packets(
        scorer,
        receipt,
        proposal_features,
        binding,
        action,
        (("r0",), ("r1",)),
        budget_fraction=0.01,
        train_response_rms=0.44,
        protected_near_source_ids=(("s1",), ()),
    )
    assert len(packets) == 2
    assert "s1" in packets[0].source_ids
    union = union_receiver_packets(binding, action, packets, ("r0", "r1"))
    assert not any(union.fallback)

    with torch.no_grad():
        scorer.network[0].weight[0, 0].add_(0.01)
    with pytest.raises(ValueError, match="weights differ"):
        propose_receiver_packets(
            scorer,
            receipt,
            proposal_features,
            binding,
            action,
            (("r0",), ("r1",)),
            budget_fraction=0.01,
            train_response_rms=0.44,
        )

    uncalibrated = propose_receiver_packets(
        scorer,
        None,
        torch.randn(2, 3, 5),
        binding,
        action,
        (("r0",), ("r1",)),
        budget_fraction=0.02,
        train_response_rms=0.44,
    )
    fallback = union_receiver_packets(binding, action, uncalibrated, ("r0", "r1"))
    assert all(fallback.fallback)
    assert fallback.source_keep.all()


def test_mode_contract_full_default_advisory_and_experimental_masks():
    binding = _binding()
    action = _action(binding)
    packet = _packet(binding, action, ("r0", "r1"), ("s1",))
    union = union_receiver_packets(binding, action, (packet,), ("r0", "r1"))
    full, full_receipt = execution_source_mask("full", proposed=union)
    advisory, advisory_receipt = execution_source_mask("packet_advisory", proposed=union)
    experimental, experimental_receipt = execution_source_mask("packet_experimental", proposed=union)
    assert full.all() and advisory.all()
    assert torch.equal(experimental, union.source_keep)
    assert not full_receipt["packet_mask_executed"]
    assert not advisory_receipt["packet_mask_executed"]
    assert experimental_receipt["packet_mask_executed"]


def _operator_and_state():
    torch.manual_seed(9)
    operator = SourceResponseOperator(
        source_width=2,
        context_width=1,
        environment_width=3,
        mode="direct",
        spatial_dim=2,
        hidden=8,
        message=8,
        output_width=1,
        far_hidden=8,
    )
    centers = torch.tensor([[[0.1, 0.1], [0.9, 0.9], [0.5, 0.4]]])
    context = operator.prepare_context(
        sources=torch.randn(1, 3, 2),
        context=torch.ones(1, 1),
        centers=centers,
        present=torch.ones(1, 3),
        lengths=torch.ones(1, 2),
        source_lengths=torch.full((1, 3), 0.05),
        environment_tokens=torch.empty(1, 0, 3),
        environment_coords=torch.empty(1, 0, 2),
    )
    receivers = torch.tensor([[[0.05, 0.05], [0.4, 0.8], [0.9, 0.9], [0.2, 0.8]]])
    return operator, context, receivers


def test_subset_core_gathers_before_fine_mlp_and_exact_all_source_parity():
    operator, context, receivers = _operator_and_state()
    all_sources = torch.ones(1, receivers.shape[1], 3, dtype=torch.bool)
    full = operator.prepare_receivers(context, receivers)
    subset_all = operator.prepare_receivers_subset(context, receivers, all_sources)
    torch.testing.assert_close(
        operator.apply_increment(full, torch.tensor([[0.1, -0.1, 0.05]])),
        operator.apply_increment(subset_all, torch.tensor([[0.1, -0.1, 0.05]])),
    )
    keep = torch.zeros_like(all_sources)
    keep[..., 0] = True
    partial = operator.prepare_receivers_subset(context, receivers, keep)
    receipt = partial.execution_receipt
    assert receipt["fine_rows"] < receipt["full_far_rows"]
    assert receipt["near_rows"] > 0
    assert receipt["context_ancestry"].startswith("full")


def test_inverse_consumer_keeps_exact_baseline_and_invalidates_stale_packet_to_full():
    operator, context, receivers = _operator_and_state()
    binding = _binding(prepared_context=context, receiver_ids=("r0", "r1", "r2", "r3"))
    action = ActionDomain("heat", "W", binding.source_ids, (0.2, 0.2, 0.2))
    packet = _packet(binding, action, binding.receiver_ids, ("s0",))
    union = union_receiver_packets(binding, action, (packet,), binding.receiver_ids)
    delta = torch.tensor([[0.1, -0.05, 0.0]])
    exact_baseline = torch.full((1, receivers.shape[1], 1), 7.0)

    expected = (
        operator.apply_increment(
            operator.prepare_receivers(context, receivers),
            delta,
        )
        + exact_baseline
    )
    full_result, full_receipt = apply_affine_packet_increment(
        operator,
        context,
        receivers,
        exact_baseline,
        delta,
        binding,
        binding,
        action,
        mode="full",
        requested_receiver_ids=binding.receiver_ids,
    )
    advisory_result, advisory_receipt = apply_affine_packet_increment(
        operator,
        context,
        receivers,
        exact_baseline,
        delta,
        binding,
        binding,
        action,
        union,
        mode="packet_advisory",
    )
    experimental_result, experimental_receipt = apply_affine_packet_increment(
        operator,
        context,
        receivers,
        exact_baseline,
        delta,
        binding,
        binding,
        action,
        union,
        mode="packet_experimental",
    )
    torch.testing.assert_close(full_result, expected)
    torch.testing.assert_close(advisory_result, expected)
    assert torch.equal(full_result - exact_baseline, expected - exact_baseline)
    assert experimental_receipt["executed_subset_receipt"]["fine_rows"] < 4 * 3
    assert not full_receipt["packet_proposal_used"]
    assert not full_receipt["used_pre_mlp_subset_gather"]
    assert not advisory_receipt["used_pre_mlp_subset_gather"]
    assert advisory_receipt["executed_subset_receipt"]["fine_rows"] == receivers.shape[1] * 3
    assert experimental_receipt["packet_mask_executed"]
    assert not torch.equal(experimental_result, expected)

    uncalibrated = ReceiverPacket(
        "uncalibrated",
        binding.fingerprint,
        action.fingerprint,
        binding.receiver_ids,
        ("s0",),
        binding.environment_context_ids,
        binding.ancestry,
        binding.output_roles,
        binding.control_roles,
        binding.units,
    )
    uncalibrated_union = union_receiver_packets(binding, action, (uncalibrated,), binding.receiver_ids)
    assert all(uncalibrated_union.fallback)
    fallback_result, fallback_receipt = apply_affine_packet_increment(
        operator,
        context,
        receivers,
        exact_baseline,
        delta,
        binding,
        binding,
        action,
        uncalibrated_union,
        mode="packet_experimental",
    )
    torch.testing.assert_close(fallback_result, expected)
    assert not fallback_receipt["used_pre_mlp_subset_gather"]
    assert fallback_receipt["executed_subset_receipt"]["fine_rows"] == 4 * 3

    new_centers = context.centers.detach().clone()
    new_centers[0, 1, 0] -= 0.03
    new_context = operator.prepare_context(
        sources=torch.randn(1, 3, 2),
        context=torch.ones(1, 1),
        centers=new_centers,
        present=torch.ones(1, 3),
        lengths=torch.ones(1, 2),
        source_lengths=torch.full((1, 3), 0.05),
        environment_tokens=torch.empty(1, 0, 3),
        environment_coords=torch.empty(1, 0, 2),
    )
    new_binding = _binding(prepared_context=new_context, receiver_ids=binding.receiver_ids)
    with pytest.raises(ValueError, match="stale scene"):
        apply_affine_packet_increment(
            operator,
            new_context,
            receivers,
            exact_baseline,
            delta,
            binding,
            new_binding,
            action,
            union,
            mode="packet_experimental",
        )
    binding = _binding(prepared_context=context, receiver_ids=("r0", "r1", "r2", "r3"))
    wrong_unit = ActionDomain("heat", "kW", binding.source_ids, (0.2, 0.2, 0.2))
    with pytest.raises(ValueError, match="role, physical unit"):
        apply_affine_packet_increment(
            operator,
            context,
            receivers,
            torch.zeros(1, 4, 1),
            torch.zeros(1, 3),
            binding,
            binding,
            wrong_unit,
            None,
            mode="full",
            requested_receiver_ids=binding.receiver_ids,
        )

    context.centers[0, 0, 0] += 0.001
    action = ActionDomain("heat", "W", binding.source_ids, (0.2, 0.2, 0.2))
    with pytest.raises(ValueError, match="changed; rebuild"):
        apply_affine_packet_increment(
            operator,
            context,
            receivers,
            torch.zeros(1, 4, 1),
            torch.zeros(1, 3),
            binding,
            binding,
            action,
            None,
            mode="full",
            requested_receiver_ids=binding.receiver_ids,
        )
    fresh_context_response = operator.prepare_receivers(new_context, receivers)
    fresh_baseline = operator.apply_forcing(fresh_context_response, torch.tensor([[0.4, 0.3, 0.2]]))
    stale_result, stale_receipt = apply_affine_packet_increment(
        operator,
        new_context,
        receivers,
        fresh_baseline,
        delta,
        binding,
        new_binding,
        action,
        union,
        mode="packet_experimental",
        baseline_binding=new_binding,
    )
    exact_new = operator.apply_increment(fresh_context_response, delta) + fresh_baseline
    torch.testing.assert_close(stale_result, exact_new)
    assert stale_receipt["stale_binding_fallback"]
    assert stale_receipt["fallback_receivers"] == receivers.shape[1]


def test_receiver_binding_rejects_same_ids_with_changed_coordinates_then_falls_back_fresh():
    operator, context, receivers = _operator_and_state()
    receiver_ids = ("r0", "r1", "r2", "r3")
    old_binding = _binding(prepared_context=context, receiver_ids=receiver_ids)
    action = ActionDomain("heat", "W", old_binding.source_ids, (0.2, 0.2, 0.2))
    packet = _packet(old_binding, action, receiver_ids, ("s0",))
    packet_union = union_receiver_packets(old_binding, action, (packet,), receiver_ids)
    delta = torch.tensor([[0.05, -0.025, -0.025]])
    heat = torch.tensor([[0.4, 0.3, 0.2]])
    old_response = operator.prepare_receivers(context, receivers)
    old_baseline = operator.apply_forcing(old_response, heat)
    moved_receivers = receivers.clone()
    moved_receivers[:, 1, 0] += 0.001

    with pytest.raises(ValueError, match="coordinates differ"):
        apply_affine_packet_increment(
            operator,
            context,
            moved_receivers,
            old_baseline,
            delta,
            old_binding,
            old_binding,
            action,
            packet_union,
            mode="packet_experimental",
            requested_receiver_ids=receiver_ids,
        )

    new_binding = _binding(
        prepared_context=context,
        receiver_ids=receiver_ids,
        receiver_coordinates=tuple(tuple(value for value in row) for row in moved_receivers[0].tolist()),
    )
    with pytest.raises(ValueError, match="stale scene"):
        apply_affine_packet_increment(
            operator,
            context,
            moved_receivers,
            old_baseline,
            delta,
            old_binding,
            new_binding,
            action,
            packet_union,
            mode="packet_experimental",
            requested_receiver_ids=receiver_ids,
            baseline_binding=old_binding,
        )
    new_response = operator.prepare_receivers(context, moved_receivers)
    new_baseline = operator.apply_forcing(new_response, heat)
    result, receipt = apply_affine_packet_increment(
        operator,
        context,
        moved_receivers,
        new_baseline,
        delta,
        old_binding,
        new_binding,
        action,
        packet_union,
        mode="packet_experimental",
        requested_receiver_ids=receiver_ids,
        baseline_binding=new_binding,
    )
    expected = operator.apply_increment(new_response, delta) + new_baseline
    torch.testing.assert_close(result, expected)
    assert receipt["stale_binding_fallback"]
    assert receipt["executed_subset_receipt"]["mode"] == "packet_experimental_full_fallback"
    assert receipt["receiver_binding_validation_seconds"] >= 0


def test_packet_union_rejects_foreign_receiver_order_source_order_and_action_binding():
    operator, context, receivers = _operator_and_state()
    receiver_ids = ("r0", "r1", "r2", "r3")
    binding = _binding(receiver_ids=receiver_ids)
    action = ActionDomain("heat", "W", binding.source_ids, (0.2, 0.2, 0.2))
    packet = _packet(binding, action, receiver_ids, ("s0", "s1"))
    union = union_receiver_packets(binding, action, (packet,), receiver_ids)
    delta = torch.tensor([[0.05, -0.025, -0.025]])
    baseline = torch.zeros(1, 4, 1)

    with pytest.raises(ValueError, match="receiver IDs or order"):
        apply_affine_packet_increment(
            operator,
            context,
            receivers.flip(1),
            baseline,
            delta,
            binding,
            binding,
            action,
            union,
            mode="packet_experimental",
            requested_receiver_ids=tuple(reversed(receiver_ids)),
        )

    reversed_source_union = PacketUnion(
        receiver_ids=union.receiver_ids,
        source_ids=tuple(reversed(union.source_ids)),
        binding_fingerprint=union.binding_fingerprint,
        action_domain_fingerprint=union.action_domain_fingerprint,
        source_keep=union.source_keep.flip(-1),
        fallback=union.fallback,
        reasons=union.reasons,
        selected_source_ids=tuple(tuple(reversed(row)) for row in union.selected_source_ids),
    )
    with pytest.raises(ValueError, match="physical source order"):
        apply_affine_packet_increment(
            operator,
            context,
            receivers,
            baseline,
            delta,
            binding,
            binding,
            action,
            reversed_source_union,
            mode="packet_experimental",
            requested_receiver_ids=receiver_ids,
        )

    changed_action = ActionDomain("heat", "W", binding.source_ids, (0.1, 0.1, 0.1))
    action_mismatch_union = union_receiver_packets(binding, changed_action, (packet,), receiver_ids)
    with pytest.raises(ValueError, match="different action domain"):
        apply_affine_packet_increment(
            operator,
            context,
            receivers,
            baseline,
            delta,
            binding,
            binding,
            action,
            action_mismatch_union,
            mode="packet_experimental",
            requested_receiver_ids=receiver_ids,
        )

    mutated_fallback = union_receiver_packets(binding, action, (), receiver_ids)
    mutated_fallback.source_keep.zero_()
    with pytest.raises(ValueError, match="modified after immutable"):
        apply_affine_packet_increment(
            operator,
            context,
            receivers,
            baseline,
            delta,
            binding,
            binding,
            action,
            mutated_fallback,
            mode="packet_experimental",
            requested_receiver_ids=receiver_ids,
        )

    bypassed_version_union = union_receiver_packets(binding, action, (), receiver_ids)
    bypassed_version_union.source_keep.data.zero_()
    with pytest.raises(ValueError, match="no longer matches its immutable"):
        apply_affine_packet_increment(
            operator,
            context,
            receivers,
            baseline,
            delta,
            binding,
            binding,
            action,
            bypassed_version_union,
            mode="packet_experimental",
            requested_receiver_ids=receiver_ids,
        )

    one_source_packet = _packet(binding, action, receiver_ids, ("s0",), packet_id="one-source")
    mutated_partial = union_receiver_packets(binding, action, (one_source_packet,), receiver_ids)
    mutated_partial.source_keep[:, 0] = False
    mutated_partial.source_keep[:, 2] = True
    with pytest.raises(ValueError, match="modified after immutable"):
        apply_affine_packet_increment(
            operator,
            context,
            receivers,
            baseline,
            delta,
            binding,
            binding,
            action,
            mutated_partial,
            mode="packet_experimental",
            requested_receiver_ids=receiver_ids,
        )


def test_full_route_row_receipt_counts_padded_fine_mlp_rows_separately_from_active_proposal_rows():
    operator, context, receivers = _operator_and_state()
    context = operator.prepare_context(
        sources=context.source_states.new_ones(1, 3, 2),
        context=context.global_state.new_ones(1, 1),
        centers=context.centers,
        present=torch.tensor([[1.0, 0.0, 1.0]]),
        lengths=torch.ones(1, 2),
        source_lengths=torch.tensor([[0.05, 0.0, 0.05]]),
        environment_tokens=torch.empty(1, 0, 3),
        environment_coords=torch.empty(1, 0, 2),
    )
    base_binding = _binding(prepared_context=context, receiver_ids=("r0", "r1", "r2", "r3"))
    binding = InteractionBinding(
        dataset_id=base_binding.dataset_id,
        scene_id=base_binding.scene_id,
        geometry_fingerprint=base_binding.geometry_fingerprint,
        context_fingerprint=base_binding.context_fingerprint,
        source_ids=("s0", "s2"),
        source_slots=(0, 2),
        prepared_source_ids=("0", "2"),
        receiver_ids=base_binding.receiver_ids,
        receiver_coordinates=base_binding.receiver_coordinates,
        environment_context_ids=base_binding.environment_context_ids,
        output_roles=base_binding.output_roles,
        control_roles=base_binding.control_roles,
        units=base_binding.units,
        capability=base_binding.capability,
        ancestry=base_binding.ancestry,
    )
    action = ActionDomain("heat", "W", binding.source_ids, (0.2, 0.2))
    baseline = torch.zeros(1, receivers.shape[1], 1)
    _prediction, receipt = apply_affine_packet_increment(
        operator,
        context,
        receivers,
        baseline,
        torch.tensor([[0.05, -0.025]]),
        binding,
        binding,
        action,
        mode="full",
        requested_receiver_ids=binding.receiver_ids,
    )
    execution = receipt["executed_subset_receipt"]
    assert receipt["proposed_active_source_pair_rows"] == receivers.shape[1] * 2
    assert receipt["executed_source_rows"] == execution["full_far_rows"]
    assert receipt["executed_source_rows"] == receivers.shape[1] * context.centers.shape[1]

    packet = _packet(binding, action, binding.receiver_ids, ("s0",))
    union = union_receiver_packets(binding, action, (packet,), binding.receiver_ids)
    _prediction, advisory_receipt = apply_affine_packet_increment(
        operator,
        context,
        receivers,
        baseline,
        torch.tensor([[0.05, -0.025]]),
        binding,
        binding,
        action,
        union,
        mode="packet_advisory",
        requested_receiver_ids=binding.receiver_ids,
    )
    assert advisory_receipt["proposed_active_source_pair_rows"] == receivers.shape[1]
    assert advisory_receipt["selected_active_source_pair_rows"] == receivers.shape[1] * 2
    assert advisory_receipt["executed_source_rows"] == receivers.shape[1] * context.centers.shape[1]


def test_packet_increment_rejects_nonlinear_capability_and_stale_prepared_tensors():
    operator, context, receivers = _operator_and_state()
    nonlinear = _binding(
        prepared_context=context,
        receiver_ids=("r0", "r1", "r2", "r3"),
        capability="nonlinear_field",
    )
    action = ActionDomain("heat", "W", nonlinear.source_ids, (0.2, 0.2, 0.2))
    union = union_receiver_packets(nonlinear, action, (), nonlinear.receiver_ids)
    with pytest.raises(ValueError, match="affine scalar-control"):
        apply_affine_packet_increment(
            operator,
            context,
            receivers,
            torch.zeros(1, 4, 1),
            torch.zeros(1, 3),
            nonlinear,
            nonlinear,
            action,
            union,
            mode="packet_experimental",
        )


def test_prepared_context_fingerprint_rejects_same_operator_cross_scene_and_changed_states():
    operator, context_a, receivers = _operator_and_state()
    receiver_ids = ("r0", "r1", "r2", "r3")
    binding = _binding(prepared_context=context_a, receiver_ids=receiver_ids)
    action = ActionDomain("heat", "W", binding.source_ids, (0.2, 0.1, 0.0), require_balanced=False)
    baseline = torch.zeros(1, receivers.shape[1], 1)
    delta = torch.tensor([[0.05, -0.025, 0.0]])

    changed_centers = context_a.centers.detach().clone()
    changed_centers[0, 0, 0] += 0.02
    context_b = operator.prepare_context(
        sources=torch.randn(1, 3, 2),
        context=torch.ones(1, 1),
        centers=changed_centers,
        present=context_a.present.clone(),
        lengths=context_a.lengths.clone(),
        source_lengths=context_a.source_lengths.clone(),
        environment_tokens=torch.empty(1, 0, 3),
        environment_coords=torch.empty(1, 0, 2),
    )
    with pytest.raises(ValueError, match="Prepared context fingerprints differ"):
        apply_affine_packet_increment(
            operator,
            context_b,
            receivers,
            baseline,
            delta,
            binding,
            binding,
            action,
            mode="full",
            requested_receiver_ids=receiver_ids,
        )
    original_fingerprints = prepared_context_fingerprints(context_a)
    context_a.global_state[0, 0] += 0.01
    assert prepared_context_fingerprints(context_a) != original_fingerprints
    with pytest.raises(ValueError, match="Prepared context fingerprints differ"):
        apply_affine_packet_increment(
            operator,
            context_a,
            receivers,
            baseline,
            delta,
            binding,
            binding,
            action,
            mode="full",
            requested_receiver_ids=receiver_ids,
        )


def test_affine_packet_consumer_rejects_unbound_query_features():
    operator = SourceResponseOperator(
        source_width=2,
        context_width=1,
        environment_width=3,
        mode="direct",
        spatial_dim=2,
        hidden=8,
        message=8,
        output_width=1,
        far_hidden=8,
        query_width=1,
    )
    centers = torch.tensor([[[0.1, 0.1], [0.9, 0.9], [0.5, 0.4]]])
    context = operator.prepare_context(
        torch.randn(1, 3, 2),
        torch.ones(1, 1),
        centers,
        torch.ones(1, 3),
        torch.ones(1, 2),
        torch.full((1, 3), 0.05),
        environment_tokens=torch.empty(1, 0, 3),
        environment_coords=torch.empty(1, 0, 2),
    )
    receivers = torch.tensor([[[0.05, 0.05], [0.4, 0.8]]])
    binding = _binding(
        prepared_context=context,
        receiver_ids=("r0", "r1"),
        receiver_coordinates=((0.05, 0.05), (0.4, 0.8)),
    )
    action = ActionDomain("heat", "W", binding.source_ids, (0.2, 0.2, 0.2))
    baseline_features = torch.zeros(1, 2, 1)
    baseline = operator.apply_forcing(
        operator.prepare_receivers(context, receivers, receiver_features=baseline_features),
        torch.ones(1, 3),
    )
    with pytest.raises(ValueError, match="receiver-feature-aware exact-baseline binding"):
        apply_affine_packet_increment(
            operator,
            context,
            receivers,
            baseline,
            torch.tensor([[0.05, -0.025, -0.025]]),
            binding,
            binding,
            action,
            mode="full",
            requested_receiver_ids=binding.receiver_ids,
            receiver_features=torch.ones(1, 2, 1),
        )


def test_affine_packet_consumer_rejects_grouped_readout_with_mutable_derived_cache():
    operator = SourceResponseOperator(
        source_width=2,
        context_width=1,
        environment_width=3,
        mode="group",
        spatial_dim=2,
        hidden=8,
        message=8,
        output_width=1,
        far_hidden=8,
    )
    centers = torch.tensor([[[0.1, 0.1], [0.9, 0.9], [0.5, 0.4]]])
    context = operator.prepare_context(
        torch.randn(1, 3, 2),
        torch.ones(1, 1),
        centers,
        torch.ones(1, 3),
        torch.ones(1, 2),
        torch.full((1, 3), 0.05),
        environment_tokens=torch.empty(1, 0, 3),
        environment_coords=torch.empty(1, 0, 2),
    )
    receivers = torch.tensor([[[0.05, 0.05], [0.4, 0.8]]])
    binding = _binding(
        prepared_context=context,
        receiver_ids=("r0", "r1"),
        receiver_coordinates=((0.05, 0.05), (0.4, 0.8)),
    )
    action = ActionDomain("heat", "W", binding.source_ids, (0.2, 0.2, 0.2))
    context.group_states[0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="grouped readouts require derived-cache binding"):
        apply_affine_packet_increment(
            operator,
            context,
            receivers,
            torch.zeros(1, 2, 1),
            torch.tensor([[0.05, -0.025, -0.025]]),
            binding,
            binding,
            action,
            mode="full",
            requested_receiver_ids=binding.receiver_ids,
        )

def test_active_physical_packets_map_to_padded_prepared_source_slots():
    operator, _context, receivers = _operator_and_state()
    centers = torch.tensor([[[0.1, 0.1], [0.4, 0.4], [0.9, 0.9]]])
    context = operator.prepare_context(
        sources=torch.randn(1, 3, 2),
        context=torch.ones(1, 1),
        centers=centers,
        present=torch.tensor([[1.0, 0.0, 1.0]]),
        lengths=torch.ones(1, 2),
        source_lengths=torch.tensor([[0.05, 0.0, 0.05]]),
        environment_tokens=torch.empty(1, 0, 3),
        environment_coords=torch.empty(1, 0, 2),
    )
    binding = InteractionBinding(
        dataset_id="test-dataset",
        scene_id="padded-layout",
        geometry_fingerprint=prepared_context_fingerprints(context)[0],
        context_fingerprint=prepared_context_fingerprints(context)[1],
        source_ids=("physical-a", "physical-c"),
        source_slots=(0, 2),
        prepared_source_ids=("0", "2"),
        receiver_ids=("r0", "r1", "r2", "r3"),
        receiver_coordinates=((0.05, 0.05), (0.4, 0.8), (0.9, 0.9), (0.2, 0.8)),
        environment_context_ids=(),
        output_roles=("temperature",),
        control_roles=("heat",),
        units=(("control", "native heating-rate units"), ("output.temperature", "native temperature units")),
        capability="affine_scalar_increment",
        ancestry=("full-context",),
    )
    action = ActionDomain("heat", "native heating-rate units", binding.source_ids, (0.2, 0.2))
    packet = ReceiverPacket(
        "padded-packet",
        binding.fingerprint,
        action.fingerprint,
        binding.receiver_ids,
        ("physical-c",),
        (),
        binding.ancestry,
        binding.output_roles,
        binding.control_roles,
        binding.units,
        residual_margin=0.1,
    )
    union = union_receiver_packets(binding, action, (packet,), binding.receiver_ids)
    delta = torch.tensor([[0.1, -0.05]])
    baseline = torch.full((1, receivers.shape[1], 1), 2.0)
    experimental, receipt = apply_affine_packet_increment(
        operator,
        context,
        receivers,
        baseline,
        delta,
        binding,
        binding,
        action,
        union,
        mode="packet_experimental",
        requested_receiver_ids=binding.receiver_ids,
    )
    full_response = operator.prepare_receivers(context, receivers)
    expected = operator.apply_increment(full_response, torch.tensor([[0.1, 0.0, -0.05]])) + baseline
    assert receipt["source_slots"] == [0, 2]
    assert receipt["prepared_source_ids"] == ["0", "2"]
    assert receipt["executed_subset_receipt"]["fine_rows"] < receipt["executed_subset_receipt"]["full_far_rows"]
    assert not torch.allclose(experimental, expected)

    full_retention_packet = _packet(binding, action, binding.receiver_ids, binding.source_ids, packet_id="all-active")
    full_retention_union = union_receiver_packets(
        binding, action, (full_retention_packet,), binding.receiver_ids
    )
    full_retention_prediction, full_retention_receipt = apply_affine_packet_increment(
        operator,
        context,
        receivers,
        baseline,
        delta,
        binding,
        binding,
        action,
        full_retention_union,
        mode="packet_experimental",
        requested_receiver_ids=binding.receiver_ids,
    )
    torch.testing.assert_close(full_retention_prediction, expected)
    assert not full_retention_receipt["used_pre_mlp_subset_gather"]
    assert full_retention_receipt["fallback_receivers"] == 0
    assert full_retention_receipt["executed_source_rows"] == receivers.shape[1] * 3

    full_fallback_union = union_receiver_packets(binding, action, (), binding.receiver_ids)
    fallback_prediction, fallback_receipt = apply_affine_packet_increment(
        operator,
        context,
        receivers,
        baseline,
        delta,
        binding,
        binding,
        action,
        full_fallback_union,
        mode="packet_experimental",
        requested_receiver_ids=binding.receiver_ids,
    )
    torch.testing.assert_close(fallback_prediction, expected)
    assert not fallback_receipt["used_pre_mlp_subset_gather"]
    assert fallback_receipt["fallback_receivers"] == receivers.shape[1]
    assert fallback_receipt["executed_source_rows"] == receivers.shape[1] * 3
    assert fallback_receipt["full_far_rows"] == receivers.shape[1] * 3


def test_full_mode_rejects_inactive_or_misidentified_prepared_source_slots():
    operator, _context, receivers = _operator_and_state()
    centers = torch.tensor([[[0.1, 0.1], [0.4, 0.4], [0.9, 0.9]]])
    context = operator.prepare_context(
        sources=torch.randn(1, 3, 2),
        context=torch.ones(1, 1),
        centers=centers,
        present=torch.tensor([[1.0, 0.0, 1.0]]),
        lengths=torch.ones(1, 2),
        source_lengths=torch.tensor([[0.05, 0.0, 0.05]]),
        environment_tokens=torch.empty(1, 0, 3),
        environment_coords=torch.empty(1, 0, 2),
    )
    valid = InteractionBinding(
        dataset_id="test-dataset",
        scene_id="padded-layout",
        geometry_fingerprint=prepared_context_fingerprints(context)[0],
        context_fingerprint=prepared_context_fingerprints(context)[1],
        source_ids=("physical-a", "physical-c"),
        source_slots=(0, 2),
        prepared_source_ids=("0", "2"),
        receiver_ids=("r0", "r1", "r2", "r3"),
        receiver_coordinates=((0.05, 0.05), (0.4, 0.8), (0.9, 0.9), (0.2, 0.8)),
        environment_context_ids=(),
        output_roles=("temperature",),
        control_roles=("heat",),
        units=(("control", "native heating-rate units"), ("output.temperature", "native temperature units")),
        capability="affine_scalar_increment",
        ancestry=("full-context",),
    )
    invalid_bindings = (
        replace(valid, source_slots=(0, 1), prepared_source_ids=("0", "1")),
        replace(valid, prepared_source_ids=("2", "0")),
        replace(valid, source_ids=("physical-a",), source_slots=(0,), prepared_source_ids=("0",)),
    )
    baseline = torch.zeros(1, receivers.shape[1], 1)
    for binding in invalid_bindings:
        action = ActionDomain(
            "heat",
            "native heating-rate units",
            binding.source_ids,
            tuple(0.2 for _ in binding.source_ids),
        )
        with pytest.raises(ValueError, match="prepared active source catalog|native source IDs"):
            apply_affine_packet_increment(
                operator,
                context,
                receivers,
                baseline,
                torch.zeros(1, len(binding.source_ids)),
                binding,
                binding,
                action,
                mode="full",
                requested_receiver_ids=binding.receiver_ids,
            )


def _compiled_receiver_request(operator, context, receivers, *, packets=None, roles=None, fallback_roles=()):
    receiver_ids = ("r0", "r1", "r2", "r3")
    binding = _binding(prepared_context=context, receiver_ids=receiver_ids)
    action = ActionDomain("heat", "W", binding.source_ids, (0.2, 0.1, 0.0))
    if packets is None:
        packets = (_packet(binding, action, receiver_ids, ("s0",)),)
    request = compile_receiver_request(
        binding,
        action,
        packets,
        list(receiver_ids),
        operator=operator,
        prepared_context=context,
        receiver_roles=roles,
        full_fallback_roles=fallback_roles,
    )
    return binding, action, request


def _balanced_packet_case():
    operator, context, receivers = _operator_and_state()
    receiver_ids = ("r0", "r1", "r2", "r3")
    binding = _binding(prepared_context=context, receiver_ids=receiver_ids)
    action = ActionDomain("heat", "W", binding.source_ids, (0.2, 0.2, 0.2), require_balanced=True)
    packet = _packet(binding, action, receiver_ids, ("s0",))
    union = union_receiver_packets(binding, action, (packet,), receiver_ids)
    request = compile_receiver_request(
        binding,
        action,
        (packet,),
        receiver_ids,
        operator=operator,
        prepared_context=context,
    )
    baseline = torch.zeros(receivers.shape[0], receivers.shape[1], 1)
    return operator, context, receivers, binding, action, union, request, baseline


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_balanced_action_accepts_valid_float32_and_float64_controls(dtype):
    (
        _operator,
        _context,
        _receivers,
        binding,
        action,
        _union,
        request,
        _baseline,
    ) = _balanced_packet_case()
    balanced = torch.tensor([[0.1, -0.1, 0.0]], dtype=dtype)
    assert action.accepts(balanced, binding.source_ids)
    assert packet_module._compiled_action_valid_rows(balanced, request).tolist() == [True]


def test_balance_guard_is_scale_relative_and_accepts_float32_sum_others_rounding():
    (
        operator,
        context,
        receivers,
        binding,
        action,
        union,
        request,
        baseline,
    ) = _balanced_packet_case()
    formed_from_sum = torch.tensor([[0.05, 0.06, 0.0]], dtype=torch.float32)
    formed_from_sum[:, 2] = -formed_from_sum[:, :2].sum(-1)
    assert action.accepts(formed_from_sum, binding.source_ids)
    assert packet_module._compiled_action_valid_rows(formed_from_sum, request).tolist() == [True]

    tiny_unbalanced = torch.tensor([[1.0e-6, -0.95e-6, 0.0]], dtype=torch.float32)
    for increment in (tiny_unbalanced, tiny_unbalanced * 1000.0):
        assert not action.accepts(increment, binding.source_ids)
        assert packet_module._compiled_action_valid_rows(increment, request).tolist() == [False]

    _value_result, value_receipt = apply_affine_packet_increment(
        operator,
        context,
        receivers,
        baseline,
        tiny_unbalanced,
        binding,
        binding,
        action,
        union,
        mode="packet_experimental",
    )
    assert not value_receipt["action_domain_valid"]
    assert not value_receipt["used_pre_mlp_subset_gather"]
    _compiled_result, compiled_receipt = apply_differentiable_fixed_route(
        operator,
        context,
        receivers,
        baseline,
        tiny_unbalanced,
        request,
        current_binding=binding,
        baseline_binding=binding,
        action_domain=action,
        mode="packet_experimental",
    )
    assert compiled_receipt["invalid_action_batch_rows_full_fallback"] == 1
    assert not compiled_receipt["used_pre_mlp_subset_gather"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_unbalanced_controls_force_full_access_in_both_consumers(dtype):
    (
        operator,
        context,
        receivers,
        binding,
        action,
        union,
        request,
        baseline,
    ) = _balanced_packet_case()
    unbalanced = torch.tensor([[0.1, -0.095, 0.0]], dtype=dtype)
    assert not action.accepts(unbalanced, binding.source_ids)
    assert packet_module._compiled_action_valid_rows(unbalanced, request).tolist() == [False]

    value_result, value_receipt = apply_affine_packet_increment(
        operator,
        context,
        receivers,
        baseline,
        unbalanced,
        binding,
        binding,
        action,
        union,
        mode="packet_experimental",
    )
    assert value_result.shape == baseline.shape
    assert not value_receipt["action_domain_valid"]
    assert not value_receipt["used_pre_mlp_subset_gather"]
    assert value_receipt["executed_subset_receipt"]["fine_rows"] == receivers.shape[0] * receivers.shape[1] * 3

    compiled_result, compiled_receipt = apply_differentiable_fixed_route(
        operator,
        context,
        receivers,
        baseline,
        unbalanced,
        request,
        current_binding=binding,
        baseline_binding=binding,
        action_domain=action,
        mode="packet_experimental",
    )
    assert compiled_result.shape == baseline.shape
    assert compiled_receipt["invalid_action_batch_rows_full_fallback"] == 1
    assert not compiled_receipt["used_pre_mlp_subset_gather"]
    assert compiled_receipt["executed_subset_receipt"]["fine_rows"] == receivers.shape[0] * receivers.shape[1] * 3


def test_compiled_request_deduplicates_overlapping_packets_and_owns_immutable_catalogs():
    operator, context, _receivers = _operator_and_state()
    binding = _binding(
        prepared_context=context,
        receiver_ids=("r0", "r1", "r2", "r3"),
    )
    binding = replace(binding, output_roles=("fluid", "solid"))
    action = ActionDomain("heat", "W", binding.source_ids, (0.2, 0.1, 0.0))
    first = _packet(binding, action, ("r0", "r1"), ("s0",), packet_id="first")
    second = _packet(binding, action, ("r0", "r2"), ("s0", "s1"), packet_id="second")
    receiver_ids = ["r0", "r1", "r2", "r3"]
    request = compile_receiver_request(
        binding,
        action,
        (first, second),
        receiver_ids,
        operator=operator,
        prepared_context=context,
        receiver_roles=("fluid", "fluid", "solid", "fluid"),
        full_fallback_roles=("solid",),
    )
    receiver_ids.reverse()

    assert request.receiver_ids == ("r0", "r1", "r2", "r3")
    assert request.packet_ids == ("first", "second")
    assert request.selected_source_ids == (
        ("s0", "s1"),
        ("s0",),
        ("s0", "s1", "s2"),
        ("s0", "s1", "s2"),
    )
    assert request.source_keep.tolist() == [
        [True, True, False],
        [True, False, False],
        [True, True, True],
        [True, True, True],
    ]
    assert request.fallback == (False, False, True, True)
    assert request.fallback_reasons[2] == "unsupported_receiver_role:solid"
    assert request.fallback_reasons[3] == "no_packet_covers_receiver"
    assert request.role_partitions == (("fluid", (0, 1, 3)), ("solid", (2,)))
    assert request.fallback_role_counts == (("fluid", 1), ("solid", 1))

    diagnostic_mask = request.source_keep
    diagnostic_mask.zero_()
    assert request.source_keep.tolist()[0] == [True, True, False]


def test_compiled_request_validates_each_packet_once_at_compile_boundary(monkeypatch):
    operator, context, _receivers = _operator_and_state()
    binding = _binding(prepared_context=context, receiver_ids=("r0", "r1", "r2", "r3"))
    action = ActionDomain("heat", "W", binding.source_ids, (0.2, 0.1, 0.0))
    packet = _packet(binding, action, binding.receiver_ids, ("s0",), packet_id="counted")
    original = packet_module._packet_valid
    calls = []

    def counted(packet, binding, action):
        calls.append(packet.packet_id)
        return original(packet, binding, action)

    monkeypatch.setattr(packet_module, "_packet_valid", counted)
    compile_receiver_request(
        binding,
        action,
        (packet,),
        binding.receiver_ids,
        operator=operator,
        prepared_context=context,
    )
    assert calls == ["counted"]


def test_fixed_route_differentiable_consumer_preserves_gradients_and_row_local_fallback():
    operator, _unused_context, receiver_values = _operator_and_state()
    sources = torch.randn(1, 3, 2, requires_grad=True)
    centers = torch.tensor(
        [[[0.1, 0.1], [0.9, 0.9], [0.5, 0.4]]],
        requires_grad=True,
    )
    context = operator.prepare_context(
        sources=sources,
        context=torch.ones(1, 1),
        centers=centers,
        present=torch.ones(1, 3),
        lengths=torch.ones(1, 2),
        source_lengths=torch.full((1, 3), 0.05),
        environment_tokens=torch.empty(1, 0, 3),
        environment_coords=torch.empty(1, 0, 2),
    )
    receivers = receiver_values.clone().requires_grad_()
    binding = _binding(prepared_context=context, receiver_ids=("r0", "r1", "r2", "r3"))
    action = ActionDomain("heat", "W", binding.source_ids, (0.2, 0.1, 0.0))
    packets = (
        _packet(binding, action, ("r0", "r1"), ("s0",), packet_id="fluid-a"),
        _packet(binding, action, ("r2",), ("s0", "s1"), packet_id="solid"),
    )
    request = compile_receiver_request(
        binding,
        action,
        packets,
        binding.receiver_ids,
        operator=operator,
        prepared_context=context,
        receiver_roles=("fluid", "fluid", "solid", "fluid"),
        full_fallback_roles=("solid",),
    )
    heat = torch.tensor([[0.3, 0.2, 0.0]])
    baseline_response = operator.prepare_receivers(context, receivers)
    exact_baseline = operator.apply_forcing(
        baseline_response,
        heat,
        accumulation_dtype=torch.float64,
    )
    delta = torch.tensor([[0.1, -0.05, 0.0]], requires_grad=True)

    result, receipt = apply_differentiable_fixed_route(
        operator,
        context,
        receivers,
        exact_baseline,
        delta,
        request,
        current_binding=binding,
        baseline_binding=binding,
        action_domain=action,
        mode="packet_experimental",
    )
    full_response = operator.prepare_receivers(context, receivers)
    full_expected = exact_baseline + operator.apply_increment(
        full_response,
        delta,
        accumulation_dtype=torch.float64,
    )
    torch.testing.assert_close(result[0, 2], full_expected[0, 2])
    assert receipt["fallback_receivers"] == 2
    assert receipt["used_pre_mlp_subset_gather"]
    assert receipt["autograd_enabled"]
    assert receipt["fixed_route_derivative"]
    assert receipt["proposal_full_kernel_calls"] == 0
    assert result.dtype == torch.float64

    gradients = torch.autograd.grad(
        result.square().sum(),
        (receivers, delta, sources, centers),
        allow_unused=True,
    )
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert all(float(gradient.abs().sum()) > 0 for gradient in gradients)


def test_fixed_route_guard_rejects_receiver_binding_mask_context_action_and_optimizer_mutation():
    operator, context, receivers = _operator_and_state()
    binding, action, request = _compiled_receiver_request(operator, context, receivers)
    exact_baseline = operator.apply_forcing(
        operator.prepare_receivers(context, receivers),
        torch.tensor([[0.3, 0.2, 0.0]]),
    )
    delta = torch.tensor([[0.1, -0.05, 0.0]])
    kwargs = {
        "operator": operator,
        "prepared_context": context,
        "receivers": receivers,
        "exact_baseline": exact_baseline,
        "delta_control": delta,
        "request": request,
        "current_binding": binding,
        "baseline_binding": binding,
        "action_domain": action,
    }

    with pytest.raises(ValueError, match="coordinates or order"):
        apply_differentiable_fixed_route(**(kwargs | {"receivers": receivers.flip(1)}))
    foreign_baseline_binding = replace(binding, scene_id="same-values-new-owner")
    with pytest.raises(ValueError, match="baseline belongs to a replaced or stale scene"):
        apply_differentiable_fixed_route(
            **(kwargs | {"baseline_binding": foreign_baseline_binding})
        )

    request._source_keep_active.zero_()
    with pytest.raises(ValueError, match="active source mask was modified"):
        apply_differentiable_fixed_route(**kwargs)

    operator, context, receivers = _operator_and_state()
    binding, action, request = _compiled_receiver_request(operator, context, receivers)
    baseline = operator.apply_forcing(operator.prepare_receivers(context, receivers), torch.ones(1, 3))
    kwargs = {
        "operator": operator,
        "prepared_context": context,
        "receivers": receivers,
        "exact_baseline": baseline,
        "delta_control": delta,
        "request": request,
        "current_binding": binding,
        "baseline_binding": binding,
        "action_domain": action,
    }
    context.global_state.add_(0.01)
    with pytest.raises(ValueError, match="prepared tensor ownership changed"):
        apply_differentiable_fixed_route(**kwargs)

    operator, context, receivers = _operator_and_state()
    binding, action, request = _compiled_receiver_request(operator, context, receivers)
    baseline = operator.apply_forcing(operator.prepare_receivers(context, receivers), torch.ones(1, 3))
    kwargs = {
        "operator": operator,
        "prepared_context": context,
        "receivers": receivers,
        "exact_baseline": baseline,
        "delta_control": delta,
        "request": request,
        "current_binding": binding,
        "baseline_binding": binding,
        "action_domain": action,
    }
    optimizer = torch.optim.SGD(operator.parameters(), lr=1e-4)
    for parameter in operator.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    with pytest.raises(ValueError, match="model weights changed"):
        apply_differentiable_fixed_route(**kwargs)


def test_compiled_request_rejects_source_unit_action_config_parameter_and_receiver_mutation():
    def assert_stale(mutate, expected_message):
        operator, context, receivers = _operator_and_state()
        binding, action, request = _compiled_receiver_request(operator, context, receivers)
        baseline = operator.apply_forcing(
            operator.prepare_receivers(context, receivers),
            torch.ones(1, 3),
        )
        delta = torch.tensor([[0.05, -0.025, 0.0]])
        mutate(binding, action, operator)
        with pytest.raises(ValueError, match=expected_message):
            apply_differentiable_fixed_route(
                operator,
                context,
                receivers,
                baseline,
                delta,
                request,
                current_binding=binding,
                baseline_binding=binding,
                action_domain=action,
            )

    assert_stale(
        lambda binding, _action, _operator: object.__setattr__(
            binding, "source_ids", ("s1", "s0", "s2")
        ),
        "stale scene binding",
    )
    assert_stale(
        lambda binding, _action, _operator: object.__setattr__(
            binding,
            "units",
            (("control", "kW"), ("output.temperature", "K"), ("position", "m")),
        ),
        "stale scene binding",
    )
    assert_stale(
        lambda _binding, action, _operator: object.__setattr__(action, "unit", "kW"),
        "action role, unit, or source metadata was replaced",
    )
    assert_stale(
        lambda _binding, _action, operator: operator.config.__setitem__("forcing_scale", 2.0),
        "configuration or normalization changed",
    )
    assert_stale(
        lambda _binding, _action, operator: setattr(
            operator.near_head[0],
            "weight",
            torch.nn.Parameter(operator.near_head[0].weight.detach().clone()),
        ),
        "model weights changed",
    )

    operator, context, receivers = _operator_and_state()
    _binding, _action, request = _compiled_receiver_request(operator, context, receivers)
    coordinates = receivers.clone()
    request.validate_receivers(coordinates)
    coordinates[0, 0, 0] += 1e-3
    with pytest.raises(ValueError, match="coordinates or order"):
        request.validate_receivers(coordinates)


def test_fixed_route_invalid_action_falls_back_per_batch_row_without_disabling_others():
    operator, context_one, receiver_values = _operator_and_state()
    context = operator.prepare_context(
        sources=torch.randn(2, 3, 2),
        context=torch.ones(2, 1),
        centers=context_one.centers.expand(2, -1, -1).clone(),
        present=torch.ones(2, 3),
        lengths=torch.ones(2, 2),
        source_lengths=torch.full((2, 3), 0.05),
        environment_tokens=torch.empty(2, 0, 3),
        environment_coords=torch.empty(2, 0, 2),
    )
    receivers = receiver_values.expand(2, -1, -1).clone()
    binding = _binding(prepared_context=context, receiver_ids=("r0", "r1", "r2", "r3"))
    action = ActionDomain("heat", "W", binding.source_ids, (0.2, 0.1, 0.0))
    packet = _packet(binding, action, binding.receiver_ids, ("s0",))
    request = compile_receiver_request(
        binding,
        action,
        (packet,),
        binding.receiver_ids,
        operator=operator,
        prepared_context=context,
    )
    baseline = operator.apply_forcing(
        operator.prepare_receivers(context, receivers),
        torch.ones(2, 3),
    )
    delta = torch.tensor([[0.1, -0.05, 0.0], [0.3, -0.05, 0.0]])
    result, receipt = apply_differentiable_fixed_route(
        operator,
        context,
        receivers,
        baseline,
        delta,
        request,
        current_binding=binding,
        baseline_binding=binding,
        action_domain=action,
    )
    full = operator.apply_increment(
        operator.prepare_receivers(context, receivers),
        delta,
        accumulation_dtype=torch.float64,
    )
    torch.testing.assert_close(result[1], baseline[1] + full[1])
    assert receipt["invalid_action_batch_rows_full_fallback"] == 1
    assert receipt["used_pre_mlp_subset_gather"]
    assert receipt["executed_subset_receipt"]["fine_rows"] < receipt["executed_subset_receipt"]["full_far_rows"]

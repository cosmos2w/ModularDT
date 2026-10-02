from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from windfarm.normalization import VelocityNormalizer
from windfarm.workflows.directed_packet_pair import (
    ROLE_NAMES,
    _check_training_deadline,
    _load_role_scales,
    _parse_utc_deadline,
    _role_loss,
    _trim_history_to_checkpoint,
)
from windfarm.workflows.evaluate_directed_packet_pair import (
    _direct_geometry_same_work,
    _disjoint_panel,
    _effective_same_work_rewire,
    _max_receiver_tree_depth,
    _pair_mask_changes,
    _restore_native_role_sample,
    _source_union_collapse,
    _training_exposure_audit,
)
from windfarm.workflows.joint_forward import NativeRoleCatalogue

from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    CaseLocalReceiverTree,
    MechanismPlan,
    ReceiverAnchorUniverse,
)
from honf_forward_core.interface_fields.directional_packets import (
    DirectionalPacketOrganizer,
    _bounded_depth_two_tree,
)
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def _encoded(
    centers: torch.Tensor, present: torch.Tensor | None = None
) -> tuple[EncodedInterfaceCase, CaseLocalReceiverTree]:
    device, dtype = centers.device, centers.dtype
    batch, modules, dimension = 1, int(centers.shape[0]), int(centers.shape[1])
    env_coords = torch.stack((torch.linspace(-1.0, 1.0, 8), torch.zeros(8), torch.ones(8)), dim=-1)[:, :dimension].to(
        device=device, dtype=dtype
    )[None]
    environment = ReceiverAnchorUniverse(
        coordinates=env_coords[0],
        weights=torch.ones(8, device=device, dtype=dtype),
        roles=torch.zeros(8, device=device, dtype=torch.long),
        coordinate_scale=torch.ones(dimension, device=device, dtype=dtype),
    )
    base_tree = CaseLocalReceiverTree.build(environment, max_nodes=7, min_leaf_anchors=2, max_depth=2)
    if present is None:
        present = torch.ones((modules,), device=device, dtype=dtype)
    present = present.to(device=device, dtype=dtype).reshape(1, modules)
    encoded = EncodedInterfaceCase(
        module_tokens=torch.randn(batch, modules, 4, device=device, dtype=dtype),
        env_tokens=torch.randn(batch, 8, 4, device=device, dtype=dtype),
        global_token=torch.randn(batch, 4, device=device, dtype=dtype),
        module_centers=centers[None],
        env_coords=env_coords,
        module_present=present,
        module_features=torch.randn(batch, modules, 2, device=device, dtype=dtype),
        env_features=torch.randn(batch, 8, 3, device=device, dtype=dtype),
        env_weights=torch.ones(batch, 8, device=device, dtype=dtype),
        coordinate_scale=torch.ones(batch, dimension, device=device, dtype=dtype),
    )
    return encoded, base_tree


def _organizer(mode: str, frame: torch.Tensor | None = None) -> DirectionalPacketOrganizer:
    return DirectionalPacketOrganizer(
        state_dim=4,
        module_feature_dim=2,
        environment_feature_dim=3,
        hidden_dim=16,
        feature_mode=mode,
        physical_frame=frame,
    )


def test_optional_training_deadline_allows_none_and_future_but_rejects_expired() -> None:
    parsed = _parse_utc_deadline("2026-10-03T00:14:00Z")
    assert parsed.tzinfo == timezone.utc
    assert parsed.isoformat() == "2026-10-03T00:14:00+00:00"
    _check_training_deadline("test", 1, None)
    _check_training_deadline("test", 1, datetime.now(timezone.utc) + timedelta(minutes=1))
    expired = datetime.now(timezone.utc) - timedelta(seconds=1)
    with pytest.raises(RuntimeError, match="authorized training/sampling stop"):
        _check_training_deadline("test", 1, expired)


def test_depth_two_tree_is_bounded_and_defaults_remain_unbounded() -> None:
    coords = torch.arange(48, dtype=torch.float32).reshape(16, 3)
    universe = ReceiverAnchorUniverse(
        coords,
        torch.ones(16),
        torch.zeros(16, dtype=torch.long),
        torch.ones(3),
    )
    depth_two = CaseLocalReceiverTree.build(universe, max_nodes=31, min_leaf_anchors=1, max_depth=2)
    legacy_default = CaseLocalReceiverTree.build(universe, max_nodes=31, min_leaf_anchors=1)
    assert len(depth_two.nodes) <= 7
    assert len(legacy_default.nodes) > len(depth_two.nodes)


def test_receiver_tree_depth_uses_indexed_child_links() -> None:
    nodes = (
        SimpleNamespace(left=1, right=2),
        SimpleNamespace(left=3, right=4),
        SimpleNamespace(left=None, right=None),
        SimpleNamespace(left=None, right=None),
        SimpleNamespace(left=None, right=None),
    )
    assert _max_receiver_tree_depth(nodes) == 2
    assert _max_receiver_tree_depth((SimpleNamespace(left=None, right=None),)) == 0


def test_persisted_native_panel_restores_exact_indices_coordinates_and_targets() -> None:
    coordinates = np.arange(15, dtype=np.float32).reshape(5, 3)
    target = (coordinates + 0.25).astype(np.float32)
    case = SimpleNamespace(run=SimpleNamespace(U=target))
    catalogue = SimpleNamespace(coordinates_D=coordinates, geometry_sha256="fixture")
    cache = SimpleNamespace(get=lambda _case: catalogue)
    roles = tuple(ROLE_NAMES)
    payload = {
        "flat_indices": np.arange(5, dtype=np.int64),
        "coordinates_D": coordinates.copy(),
        "reference_velocity_mps": target.copy(),
        "role_slices": {role: [index, index + 1] for index, role in enumerate(roles)},
        "role_query_counts": {role: 1 for role in roles},
        "role_support_volume_m3": {role: 2.0 for role in roles},
    }

    sample = _restore_native_role_sample(case, payload, cache)

    assert np.array_equal(sample.flat_indices, payload["flat_indices"])
    assert np.array_equal(sample.coordinates_D, coordinates)
    assert np.array_equal(sample.target_mps, target)
    assert sample.geometry_sha256 == "fixture"


def test_zero_residual_is_exact_geometry_prior_and_unbudgeted_routes_are_full() -> None:
    centers = torch.tensor(
        [
            [0.0, 0.0, 0.875],
            [2.0, 0.0, 0.875],
            [4.0, 0.0, 0.875],
            [6.0, 1.0, 0.875],
            [8.0, 1.0, 0.875],
            [10.0, 1.0, 0.875],
            [12.0, 2.0, 0.875],
            [14.0, 2.0, 0.875],
        ]
    )
    encoded, base_tree = _encoded(centers)
    organizer = _organizer("directional")
    scores = organizer.score_cases(encoded, {}, (base_tree,))
    item = scores[0]
    assert torch.count_nonzero(organizer.directional_pair_scorer[-1].weight) == 0
    assert torch.equal(item.mechanism_logits["MM"], item.geometry_prior)
    plans = organizer.plans_from_scores(scores, encoded, (base_tree,), hard=True)
    plan = plans[0]
    assert plan.direct_pair_access_for("MM") is not None
    assert plan.direct_pair_access_for("MM").weights.shape == (8, 8)
    for route in ("ME", "EM", "QM", "QE"):
        assert torch.equal(
            plan.permission_matrix(route),
            torch.ones_like(plan.permission_matrix(route)),
        )


def test_directional_and_summary_controls_share_width_but_differ_in_features() -> None:
    centers = torch.tensor(
        [
            [0.0, 0.0, 0.875],
            [1.0, 2.0, 0.875],
            [3.0, 1.0, 0.875],
            [5.0, 4.0, 0.875],
            [7.0, 3.0, 0.875],
            [9.0, 0.0, 0.875],
            [11.0, 2.0, 0.875],
            [13.0, 5.0, 0.875],
        ]
    )
    encoded, base_tree = _encoded(centers)
    directional = _organizer("directional")
    summary = _organizer("summary")
    dir_score = directional.score_cases(encoded, {}, (base_tree,))[0]
    sum_score = summary.score_cases(encoded, {}, (base_tree,))[0]
    assert dir_score.raw_pair_features.shape[-1] == sum_score.raw_pair_features.shape[-1] == 13
    assert torch.equal(dir_score.geometry_prior, sum_score.geometry_prior)
    assert not torch.equal(dir_score.raw_pair_features, sum_score.raw_pair_features)
    assert torch.equal(dir_score.mechanism_logits["MM"], sum_score.mechanism_logits["MM"])


def test_common_rotation_with_transformed_frame_preserves_directional_features() -> None:
    centers = torch.tensor(
        [
            [0.0, 0.0, 0.875],
            [1.0, 2.0, 0.875],
            [3.0, 1.0, 0.875],
            [5.0, 4.0, 0.875],
            [7.0, 3.0, 0.875],
            [9.0, 0.0, 0.875],
            [11.0, 2.0, 0.875],
            [13.0, 5.0, 0.875],
        ]
    )
    angle = torch.tensor(0.37)
    rotation = torch.tensor(
        [[torch.cos(angle), -torch.sin(angle), 0.0], [torch.sin(angle), torch.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    encoded, base_tree = _encoded(centers)
    rotated_encoded, rotated_base_tree = _encoded(centers @ rotation.T)
    original = _organizer("directional", torch.eye(3))
    rotated = _organizer("directional", rotation)
    original_tree, _ = _bounded_depth_two_tree(encoded, 0, physical_frame=torch.eye(3))
    rotated_tree, _ = _bounded_depth_two_tree(rotated_encoded, 0, physical_frame=rotation)
    original_features = original.packet_features(original_tree, torch.arange(8), centers)
    rotated_features = rotated.packet_features(rotated_tree, torch.arange(8), rotated_encoded.module_centers[0])
    assert torch.allclose(original_features, rotated_features, atol=1.0e-5, rtol=1.0e-5)
    original_scores = original.score_cases(encoded, {}, (base_tree,))
    rotated_scores = rotated.score_cases(rotated_encoded, {}, (rotated_base_tree,))
    original_plan = original.plans_from_scores(
        original_scores, encoded, (base_tree,), hard=True, frontier_cuts=((1, 2),)
    )[0]
    rotated_plan = rotated.plans_from_scores(
        rotated_scores, rotated_encoded, (rotated_base_tree,), hard=True, frontier_cuts=((1, 2),)
    )[0]
    original_pairs = original_plan.direct_pair_access_for("MM").weights
    rotated_pairs = rotated_plan.direct_pair_access_for("MM").weights
    assert torch.equal(original_pairs, rotated_pairs)
    del base_tree, rotated_base_tree


def test_two_dimensional_packets_mark_padded_vertical_axis_absent() -> None:
    centers = torch.tensor([[float(i), float(i % 3)] for i in range(8)])
    encoded, _base_tree = _encoded(centers)
    organizer = _organizer("directional")
    tree, active = _bounded_depth_two_tree(encoded, 0)
    features = organizer.packet_features(tree, active, encoded.module_centers[0])
    assert features.shape[-1] == 13
    assert torch.all(features[..., 12] == 0.0)
    assert torch.all(features[..., 10] == 1.0)
    assert torch.all(features[..., 11] == 1.0)


def test_padded_mm_permissions_keep_native_ids_measures_and_permute_covariantly() -> None:
    centers = torch.tensor(
        [
            [0.0, 0.0, 0.875],
            [1.0, 2.0, 0.875],
            [4.0, 1.0, 0.875],
            [7.0, 5.0, 0.875],
            [9.0, 2.0, 0.875],
            [13.0, 6.0, 0.875],
            [40.0, 20.0, 0.875],
            [50.0, -20.0, 0.875],
        ]
    )
    present = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    encoded, base_tree = _encoded(centers, present)
    organizer = _organizer("directional")
    organizer._progress.fill_(1.0)
    with torch.no_grad():
        final = organizer.directional_pair_scorer[-1]
        final.weight.copy_(torch.linspace(-0.03, 0.03, final.weight.numel()).reshape_as(final.weight))
        final.bias.fill_(0.017)
    original_scores = organizer.score_cases(encoded, {}, (base_tree,))
    original_plan = organizer.plans_from_scores(
        original_scores, encoded, (base_tree,), hard=True, frontier_cuts=((0,),)
    )[0]
    original_access = original_plan.direct_pair_access_for("MM")
    assert original_access is not None
    original_weights = original_access.weights
    active_ids = torch.nonzero(present > 0.5, as_tuple=False).flatten()
    assert torch.equal(original_access.receiver_coordinates, centers)
    assert torch.equal(original_scores[0].active_module_ids, active_ids)
    assert torch.equal(
        original_scores[0].receiver_tree.universe.coordinates,
        centers.index_select(0, active_ids),
    )
    assert torch.equal(
        original_scores[0].receiver_tree.universe.weights,
        torch.ones(active_ids.numel()),
    )
    assert int(original_weights[active_ids, :][:, active_ids].sum()) > 0
    # Invalid modules are padded sources and are masked by native execution;
    # the direct access matrix is indexed by the full source array. Its
    # receiver columns must still exclude padded module anchors.
    assert torch.count_nonzero(original_weights[:, 6:]) == 0
    full_access = MechanismPlan.full_access(base_tree, present, 8)
    for route in ("ME", "EM", "QM", "QE"):
        assert torch.equal(
            original_plan.permission_matrix(route),
            full_access.permission_matrix(route),
        )

    permutation = torch.tensor([3, 1, 5, 0, 4, 2, 7, 6])
    permuted_centers = centers.index_select(0, permutation)
    permuted_present = present.index_select(0, permutation)
    permuted_encoded, permuted_base_tree = _encoded(permuted_centers, permuted_present)
    permuted_encoded = replace(
        permuted_encoded,
        module_tokens=encoded.module_tokens.index_select(1, permutation),
        module_features=encoded.module_features.index_select(1, permutation),
    )
    permuted_scores = organizer.score_cases(permuted_encoded, {}, (permuted_base_tree,))
    permuted_plan = organizer.plans_from_scores(
        permuted_scores,
        permuted_encoded,
        (permuted_base_tree,),
        hard=True,
        frontier_cuts=((0,),),
    )[0]
    permuted_access = permuted_plan.direct_pair_access_for("MM")
    assert permuted_access is not None
    expected_permuted = original_weights.index_select(0, permutation).index_select(1, permutation)
    assert torch.allclose(permuted_access.weights, expected_permuted, atol=1.0e-6, rtol=1.0e-6)


def test_native_mm_executor_excludes_self_pairs_for_compiled_local_permissions() -> None:
    from tests.test_native_joint_shadow import _case as make_native_case

    core, batch, _legacy_organizer = make_native_case(torch.device("cpu"))
    core.eval()
    encoded = core.encode_case(batch)
    trees = core.backend.build_case_trees(encoded)
    organizer = DirectionalPacketOrganizer(
        state_dim=int(encoded.module_tokens.shape[-1]),
        module_feature_dim=int(batch.module_features.shape[-1]),
        environment_feature_dim=int(batch.env_features.shape[-1]),
        hidden_dim=16,
    )
    scores = organizer.score_cases(encoded, {}, trees)
    plan = organizer.plans_from_scores(scores, encoded, trees, hard=True, frontier_cuts=((0,),))[0]
    direct = plan.direct_pair_access_for("MM")
    assert direct is not None
    assert torch.equal(direct.receiver_coordinates, encoded.module_centers[0])
    valid = encoded.module_present[0] > 0.5
    effective = (direct.weights > 0) & valid[:, None] & valid[None, :]
    effective.fill_diagonal_(False)
    expected_unique_nonself = int(effective.sum())

    prepared = core.prepare(encoded, encoded.module_tokens, fixed_cover_plans=(plan,))
    ledger = prepared.backend_state["cover_preparation_ledger"]
    assert int(ledger["cover_prepare_mm_unique_pairs"]) == expected_unique_nonself
    assert int(ledger["cover_prepare_mm_unique_pairs"]) <= int(valid.sum()) * (int(valid.sum()) - 1)


def test_native_full_access_plan_matches_default_prediction() -> None:
    from tests.test_native_joint_shadow import _case as make_native_case

    core, batch, _organizer = make_native_case(torch.device("cpu"))
    core.eval()
    batch = replace(
        batch,
        module_centers=torch.cat((batch.module_centers, batch.module_centers[:, :1]), dim=1),
        module_present=torch.cat((batch.module_present, batch.module_present[:, :1] * 0.0), dim=1),
        module_features=torch.cat((batch.module_features, batch.module_features[:, :1] * 0.0), dim=1),
    )
    encoded = core.encode_case(batch)
    assert torch.any(encoded.module_present == 0)
    trees = core.backend.build_case_trees(encoded)
    default_prepared = core.prepare(encoded, encoded.module_tokens)
    explicit_plan = MechanismPlan.full_access(trees[0], encoded.module_present[0], int(encoded.env_coords.shape[1]))
    explicit_prepared = core.prepare(encoded, encoded.module_tokens, fixed_cover_plans=(explicit_plan,))
    packet_organizer = DirectionalPacketOrganizer(
        state_dim=int(encoded.module_tokens.shape[-1]),
        module_feature_dim=int(batch.module_features.shape[-1]),
        environment_feature_dim=int(batch.env_features.shape[-1]),
        hidden_dim=16,
    )
    packet_organizer._progress.fill_(1.0)
    with torch.no_grad():
        final = packet_organizer.directional_pair_scorer[-1]
        final.weight.fill_(0.01)
        final.bias.fill_(-0.02)
    score_inputs = {
        "module_states": encoded.module_tokens,
        "environment_states": encoded.env_tokens,
        "global_state": encoded.global_token,
    }
    scores = packet_organizer.score_cases(encoded, score_inputs, trees)
    direct_full_plan = packet_organizer.plans_from_scores(
        scores,
        encoded,
        trees,
        hard=True,
        frontier_cuts=((0,),),
        budget_fractions={"MM": 1.0},
    )[0]
    assert direct_full_plan.direct_pair_access_for("MM") is not None
    assert torch.any(encoded.module_present == 0)
    for route in ("ME", "EM", "QM", "QE"):
        assert torch.equal(direct_full_plan.permission_matrix(route), explicit_plan.permission_matrix(route))
    direct_full_prepared = core.prepare(encoded, encoded.module_tokens, fixed_cover_plans=(direct_full_plan,))
    default_prediction = core.decode_queries(default_prepared, batch.query_xy, query_features=batch.query_features)[
        "pred_field"
    ]
    explicit_prediction = core.decode_queries(explicit_prepared, batch.query_xy, query_features=batch.query_features)[
        "pred_field"
    ]
    direct_full_prediction = core.decode_queries(
        direct_full_prepared, batch.query_xy, query_features=batch.query_features
    )["pred_field"]
    assert torch.equal(default_prediction, explicit_prediction)
    assert torch.allclose(default_prediction, direct_full_prediction, atol=1.0e-6, rtol=1.0e-6)


def test_role_objective_matches_maintained_equal_five_role_mean_with_unequal_counts() -> None:
    scripts = Path(__file__).resolve().parents[1] / "Case_WindFarm" / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        from run_active_packet_reuse import _role_objective as maintained_role_objective
    finally:
        sys.path.remove(str(scripts))

    role_counts = (9, 2, 3, 5, 1)
    role_names = ("volume", "hub_slab", "downstream_envelope", "near_turbine", "background")
    role_slices: dict[str, slice] = {}
    offset = 0
    pieces = []
    for count, role_value, role in zip(role_counts, (1.0, 2.0, 3.0, 4.0, 5.0), role_names):
        role_slices[role] = slice(offset, offset + count)
        offset += count
        pieces.append(torch.full((1, count, 3), role_value))
    prediction = torch.cat(pieces, dim=1)
    target = torch.zeros_like(prediction)
    normalizer = VelocityNormalizer(
        mean=torch.zeros(3).numpy(),
        std=torch.ones(3).numpy(),
        safe_std=torch.ones(3).numpy(),
        u_ref_mps=1.0,
    )
    scales = _load_role_scales()

    actual, actual_mse = _role_loss(prediction, target, role_slices, normalizer, scales)
    expected, expected_mse, _expected_rmse = maintained_role_objective(
        prediction, target, role_slices, normalizer, scales
    )
    hand_calculated = torch.stack(
        [torch.tensor((index + 1.0) ** 2 / scales[role] ** 2) for index, role in enumerate(role_names)]
    ).mean()

    assert len(set(role_counts)) > 1
    assert len(set(scales.values())) == 5
    assert torch.allclose(actual, expected, atol=1.0e-7, rtol=1.0e-7)
    assert torch.allclose(actual, hand_calculated, atol=1.0e-7, rtol=1.0e-7)
    assert actual_mse == expected_mse


def test_resume_history_archives_discarded_rows_before_checkpoint_trim(tmp_path: Path) -> None:
    history = tmp_path / "updates.jsonl"
    original = (
        '{"phase":"scorer_only_restricted","scorer_update":1}\n'
        '{"phase":"stage1_readonly_full_access_check","scorer_update":3}\n'
        '{"phase":"coadapt_restricted","scorer_update":301,"coadapt_update":1}\n'
        '{"phase":"coadapt_full_access_replay","scorer_update":301,"coadapt_update":2}\n'
    )
    history.write_text(original, encoding="utf-8")

    _trim_history_to_checkpoint(history, scorer_updates=300, coadapt_updates=0)

    records = [line for line in history.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert '"scorer_update":1' in records[0]
    assert '"scorer_update":3' in records[1]
    archive = tmp_path / "discarded_resume_updates.jsonl"
    assert archive.read_text(encoding="utf-8") == "".join(original.splitlines(keepends=True)[2:])
    attempts = [
        json.loads(line) for line in (tmp_path / "resume_attempts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert attempts[0]["discarded_row_count"] == 2
    assert attempts[0]["discarded_rows_path"] == archive.name


def test_disjoint_q2048_sampler_rejects_panel_one_union_and_records_measure() -> None:
    roles = ("volume", "hub_slab", "downstream_envelope", "near_turbine", "background")
    role_indices = {
        "hub_slab": np.asarray([0, 1, 2, 3], dtype=np.uint32),
        "downstream_envelope": np.asarray([2, 3, 4, 5], dtype=np.uint32),
        "near_turbine": np.asarray([0, 2, 6, 8], dtype=np.uint32),
        "background": np.asarray([1, 4, 7, 9], dtype=np.uint32),
    }
    cdf = {
        "volume": np.linspace(0.1, 1.0, 10, dtype=np.float64),
        "hub_slab": np.asarray([0.1, 0.3, 0.6, 1.0]),
        "downstream_envelope": np.asarray([0.2, 0.4, 0.7, 1.0]),
        "near_turbine": np.asarray([0.15, 0.35, 0.75, 1.0]),
        "background": np.asarray([0.1, 0.25, 0.6, 1.0]),
    }
    catalogue = NativeRoleCatalogue(
        layout_index=1,
        geometry_sha256="synthetic-test-only",
        coordinates_D=np.arange(30, dtype=np.float32).reshape(10, 3),
        role_indices=role_indices,
        role_cdf=cdf,
        role_support_volume_m3={role: float(len(role_indices.get(role, np.arange(10)))) for role in roles},
        cached_nbytes=0,
    )

    class FakeCache:
        def get(self, _case):
            return catalogue

    case = SimpleNamespace(run=SimpleNamespace(U=np.arange(30, dtype=np.float32).reshape(10, 3)))
    counts = {role: 6 for role in roles}
    panel_two, metadata = _disjoint_panel(
        case,
        np.random.default_rng(20261002),
        counts,
        catalogue_cache=FakeCache(),
        excluded_flat_indices=np.asarray([0, 1, 2, 2, 2]),
    )
    assert panel_two.flat_indices.shape == (30,)
    assert metadata["cross_panel_unique_flat_index_intersection_count"] == 0
    assert metadata["cross_panel_intersection_count_by_panel_two_role"] == {role: 0 for role in roles}
    assert metadata["role_support"]["volume"]["excluded_unique_native_cells"] == 3
    assert metadata["role_support"]["volume"]["remaining_support_fraction"] == 0.7
    assert "conditional" in metadata["finite_exclusion_note"]


def test_sparse_action_controls_preserve_their_declared_work_and_valid_pairs() -> None:
    centers = torch.tensor(
        [
            [0.0, 0.0, 0.875],
            [1.0, 0.0, 0.875],
            [2.0, 0.0, 0.875],
            [3.0, 1.0, 0.875],
            [4.0, 1.0, 0.875],
            [80.0, 80.0, 0.875],
        ]
    )
    present = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    encoded, _tree = _encoded(centers, present)
    organizer = _organizer("directional")
    parent = torch.zeros((6, 6), dtype=torch.float32)
    parent[0, 1] = 0.25
    parent[0, 2] = 0.75
    parent[1, 2] = 0.50
    parent[2, 0] = 0.25
    parent[3, 0] = 1.00
    active = encoded.module_present[0] > 0.5

    union = _source_union_collapse(parent, active)
    geometry = _direct_geometry_same_work(organizer, encoded, parent)
    rewired, rewire_info = _effective_same_work_rewire(parent, active)

    assert rewired is not None and rewire_info["effective"]
    for candidate in (union, geometry, rewired):
        assert torch.count_nonzero(torch.diagonal(candidate)) == 0
        assert torch.count_nonzero(candidate[5]) == 0
        assert torch.count_nonzero(candidate[:, 5]) == 0
    assert int((geometry > 0).sum()) == int((parent > 0).sum())
    assert int((rewired > 0).sum()) == int((parent > 0).sum())
    assert torch.equal((geometry > 0).sum(dim=1), (parent > 0).sum(dim=1))
    assert torch.allclose(
        geometry.to(torch.float64).sum(dim=1),
        parent.to(torch.float64).sum(dim=1),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert torch.allclose(
        rewired.to(torch.float64).sum(),
        parent.to(torch.float64).sum(),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert torch.equal(torch.sort(geometry[geometry > 0]).values, torch.sort(parent[parent > 0]).values)
    assert torch.equal(torch.sort(rewired[rewired > 0]).values, torch.sort(parent[parent > 0]).values)
    selected_sources = (parent > 0).any(dim=0) & active
    for receiver in torch.nonzero(active, as_tuple=False).flatten().tolist():
        expected = selected_sources.clone()
        expected[receiver] = False
        assert torch.equal(union[receiver], expected)
    changes = _pair_mask_changes(parent, rewired)
    assert changes["added_pair_count"] == changes["removed_pair_count"] > 0
    assert changes["changed_permission_count"] > 0
    assert not torch.equal(rewired, parent)


def test_m30_degree_matched_geometry_preserves_fractional_weight_multisets() -> None:
    module_count = 30
    axis = torch.arange(module_count, dtype=torch.float32)
    centers = torch.stack((axis, torch.sin(axis), torch.cos(axis)), dim=-1)
    encoded, _tree = _encoded(centers)
    parent = torch.zeros((module_count, module_count), dtype=torch.float32)
    for receiver in range(module_count):
        degree = receiver % 7 + 3
        for offset in range(degree):
            source = (receiver * 11 + offset * 7 + 3) % module_count
            if source == receiver:
                source = (source + 1) % module_count
            value = ((receiver * 17 + offset * 29 + 11) % 89 + 5) / 100.0
            parent[receiver, source] = value

    geometry = _direct_geometry_same_work(_organizer("directional"), encoded, parent)
    rewired, rewire_metadata = _effective_same_work_rewire(parent, encoded.module_present[0] > 0.5)

    assert rewired is not None
    assert rewire_metadata["preserves_exact_pair_work"]
    assert rewire_metadata["preserves_permission_mass"]
    assert rewire_metadata["pair_work"] == int((parent > 0).sum())
    assert torch.equal(torch.sort(rewired[rewired > 0]).values, torch.sort(parent[parent > 0]).values)
    assert torch.equal((geometry > 0).sum(dim=1), (parent > 0).sum(dim=1))
    assert torch.allclose(
        geometry.to(torch.float64).sum(dim=1),
        parent.to(torch.float64).sum(dim=1),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    for receiver in range(module_count):
        assert torch.equal(
            torch.sort(geometry[receiver][geometry[receiver] > 0]).values,
            torch.sort(parent[receiver][parent[receiver] > 0]).values,
        )
    assert bool(((geometry.sum(dim=1) - parent.sum(dim=1)).abs() > 0).any())


def test_training_exposure_audit_counts_optimizer_rows_once_and_by_layout(tmp_path: Path) -> None:
    for arm in ("w_dir", "w_summary"):
        arm_dir = tmp_path / arm
        arm_dir.mkdir()
        records = [
            {
                "phase": "scorer_only_restricted",
                "layout_index": 4,
                "scorer_optimizer_step": True,
                "selected_action": {
                    "selected_cut_nodes": [0],
                    "realized_nonredundant_MM_packet_count": 1,
                    "distinct_exact_action_count": 4,
                },
            },
            {"phase": "stage1_readonly_full_access_check", "scorer_optimizer_step": False},
            {
                "phase": "coadapt_restricted",
                "layout_index": 4,
                "physical_optimizer_step": True,
                "scorer_optimizer_step": True,
                "selected_action": {
                    "selected_cut_nodes": [1, 2],
                    "realized_nonredundant_MM_packet_count": 2,
                    "distinct_exact_action_count": 3,
                },
            },
            {"phase": "coadapt_full_access_replay", "physical_optimizer_step": True},
        ]
        (arm_dir / "updates.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )

    audit = _training_exposure_audit(tmp_path)
    for arm in ("w_dir", "w_summary"):
        values = audit[arm]
        assert values["logged_optimizer_step_count"] == 3
        assert values["read_only_full_check_count"] == 1
        assert values["restricted_action_exposure_count"] == 2
        assert values["per_layout_cut_node_tuple_exposure"]["4"] == {"0": 1, "1,2": 1}
        assert values["per_layout_realized_nonredundant_k_exposure"]["4"] == {"1": 1, "2": 1}

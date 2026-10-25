from types import SimpleNamespace

import numpy as np
import torch

from honf_runtime.spatial_interaction_group_evaluation import (
    _bound_native_validation_membership,
    _canonical_val89_ids,
    _fixed_dev22_ids,
    _fixed_train150_ids,
    _fixed_train_mean,
    _gain_pair,
    _fixed_spatial_covers,
    _frontier_cut_summary,
    _json_safe,
    _latent_residual_summary,
    _latent_group_residuals,
    _spatial_control_probe,
    _native_coordinate_arrays,
    _spatial_prepared_tensor_bytes,
    _spatial_support_arrays,
    _surface_residual_detail,
    _stratified_organizer_candidate,
    _stratified_organizer_nodes,
    _utility_probe_aggregate,
    _tree_shape,
)
from honf_runtime.contextual_frontier_evaluation import _contextual_checkpoint_parts
from honf_forward_core.spatial_interaction_groups.frontier import build_balanced_grid_template


class _Frontier:
    def __init__(self):
        self.children = (None, None, None, (0, 1), (3, 2))
        self.nodes = (4,)
        self.reached = (4,)
        self.probabilities = torch.tensor([0.8, 0.7])


def test_tree_shape_reports_depth_imbalance_and_reached_stop_margin():
    result = _tree_shape(_Frontier(), leaf_count=3)

    assert result["candidate_tree_node_count"] == 5
    assert result["maximum_tree_depth"] == 2
    assert result["active_K"] == 1
    assert result["active_subtree_leaf_counts"] == [3]
    assert result["reached_nodes"][0]["node"] == 4
    assert result["reached_nodes"][0]["depth"] == 0
    assert abs(result["reached_nodes"][0]["signed_stop_margin_probability"] - 0.2) < 1.0e-6
    assert result["reached_nodes"][0]["temperature_scaled_stop_logit_margin"] > 0
    assert result["maximum_absolute_child_size_difference"] == 1


def test_first_order_fit_reduces_weighted_affine_latent_residual_and_ignores_zero_mass():
    coordinates = torch.tensor(
        [[-1.0, 0.0], [1.0, 0.0], [-1.0, 1.0], [1.0, 1.0], [0.0, 0.5]],
        dtype=torch.float64,
    )
    slopes = torch.tensor([[2.0, -0.5], [1.25, 3.0]], dtype=torch.float64)
    states = torch.cat(
        (coordinates[:4] @ slopes, torch.tensor([[1000.0, -900.0]], dtype=torch.float64))
    )
    mass = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0], dtype=torch.float64)
    labels = torch.zeros(5, dtype=torch.long)

    result = _latent_group_residuals(states, coordinates, mass, labels)

    assert result["active_group_count"] == 1
    assert result["groups"][0]["member_count"] == 4
    assert result["first_order_latent_rmse_mass_pooled"] < 1.0e-5
    assert result["first_order_latent_rmse_mass_pooled"] < result["mean_only_latent_rmse_mass_pooled"]


def test_first_order_fit_handles_rank_deficient_geometry():
    coordinates = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    states = torch.tensor([[0.0, 1.0], [2.0, -1.0], [0.0, 1.0]], dtype=torch.float64)

    result = _latent_group_residuals(
        states,
        coordinates,
        torch.ones(3, dtype=torch.float64),
        torch.zeros(3, dtype=torch.long),
    )

    assert result["active_group_count"] == 1
    assert torch.isfinite(torch.tensor(result["first_order_latent_rmse_mass_pooled"]))
    assert result["first_order_latent_rmse_mass_pooled"] <= result["mean_only_latent_rmse_mass_pooled"]


def test_surface_detail_separates_module_bias_from_angular_residual():
    residual = torch.tensor([[[1.0, 3.0, 1.0, 3.0], [80.0, 80.0, 80.0, 80.0]]])
    pred = torch.stack((residual, torch.zeros_like(residual)), dim=-1)
    target = torch.zeros_like(pred)
    valid = torch.tensor([[[True, True, True, True], [False, False, False, False]]])
    prepared = SimpleNamespace(
        source_present=torch.tensor([[True, False]]),
        stencils={"surface": SimpleNamespace(valid=valid.reshape(-1))},
    )
    prediction = SimpleNamespace(
        native_main={"pred_interface": pred},
        native_prepared=prepared,
    )
    batch = SimpleNamespace(
        targets=SimpleNamespace(
            interface_target=target,
            interface_valid_mask=valid,
        ),
        scene_inputs=SimpleNamespace(
            structure={"module_ids": np.asarray([["module-a", "module-b"]])},
        ),
        case_keys=("surface-case",),
    )

    summary, arrays = _surface_residual_detail(prediction, batch)

    stats = summary["channels"]["temperature_surface"]
    assert stats["module_count_with_valid_angles"] == 1
    assert stats["module_weighted_bias_rms"] == 2.0
    assert stats["valid_angle_pooled_demeaned_rmse"] == 1.0
    assert np.isnan(arrays["temperature_surface_residual"][1]).all()


def test_compact_latent_residual_summary_compares_group_mean_and_first_order_fits():
    group = SimpleNamespace(
        latent_constant_residual=torch.tensor([2.0, 8.0]),
        latent_residual=torch.tensor([0.5, 1.0]),
        mass=torch.tensor([1.0, 3.0]),
        group_means=torch.zeros((2, 2)),
        group_nodes=(10, 11),
    )
    prepared = SimpleNamespace(layers=((group,), (group,)))

    result = _latent_residual_summary(prepared)

    first_stage = result["stages"][0]
    assert first_stage["active_group_count"] == 2
    assert np.isclose(first_stage["mean_only_latent_rmse_mass_pooled"], np.sqrt(3.25))
    assert np.isclose(first_stage["first_order_latent_rmse_mass_pooled"], np.sqrt(0.4375))


def test_spatial_support_export_retains_exact_cut_and_headwise_relation_arrays():
    coordinates = np.asarray([(float(x), float(y)) for y in range(8) for x in range(24)])
    tree = build_balanced_grid_template(coordinates, 24, 8).bind(coordinates)
    root = tree.root
    frontier = SimpleNamespace(
        tree=tree,
        nodes=(root,),
        candidate_nodes=tree.internal_nodes,
        utilities=torch.zeros(len(tree.internal_nodes)),
        labels=torch.zeros(192, dtype=torch.long),
        selection_mode="learned",
        exploration_depth=None,
        policy_rows=len(tree.internal_nodes),
    )
    group = SimpleNamespace(
        frontier=frontier,
        hierarchy=tree,
        group_nodes=(root,),
        live_group_count=1,
        keys=torch.arange(8, dtype=torch.float64).reshape(1, 2, 4),
        value_coefficients=torch.arange(12, dtype=torch.float64).reshape(1, 2, 2, 3),
        group_means=torch.zeros((1, 3)),
        group_slopes=torch.zeros((1, 2, 3)),
        latent_constant_residual=torch.zeros(1),
        latent_residual=torch.zeros(1),
        mass=torch.ones(1),
        centers=torch.zeros((1, 2)),
        scales=torch.ones((1, 2)),
        relation_mean=torch.ones((1, 2, 2)),
        all_node_means=torch.zeros((tree.node_count, 3)),
        all_node_slopes=torch.zeros((tree.node_count, 2, 3)),
        all_node_constant_residual=torch.zeros(tree.node_count),
        all_node_residual=torch.zeros(tree.node_count),
        all_node_mass=torch.ones(tree.node_count),
        all_node_centers=torch.zeros((tree.node_count, 2)),
        all_node_scales=torch.ones((tree.node_count, 2)),
        all_node_relation_mean=torch.ones((tree.node_count, 2, 2)),
        all_node_relation_magnitude=torch.ones(tree.node_count),
    )
    scene = SimpleNamespace(
        module_coordinates=torch.tensor([[[1.0, 2.0], [4.0, 5.0]]]),
        module_characteristic_lengths=torch.tensor([[[0.5, 0.5], [1.0, 1.0]]]),
        module_mask=torch.tensor([[True, True]]),
        module_ids=torch.tensor([[10, 20]]),
        coordinate_origin=torch.tensor([[0.0, 0.0]]),
        coordinate_scale=torch.tensor([[12.0, 6.0]]),
        coordinate_frame_key="thermal_physical_rectangular_xy_v2",
        environment_coordinates=torch.tensor(coordinates[None], dtype=torch.float32),
        environment_mask=torch.ones((1, 192), dtype=torch.bool),
        environment_measure=torch.ones((1, 192)),
        environment_measure_reference=torch.ones((1,)),
    )
    prepared = SimpleNamespace(
        scene=scene,
        frame_origin=torch.zeros((1, 2)),
        frame_scale=torch.ones((1, 2)),
        layers=((group,), (group,)),
    )

    cut = _frontier_cut_summary(group)
    arrays = _spatial_support_arrays(prepared, "0631", physical_heat=torch.tensor([[2.0, 3.0]]))

    assert cut["K"] == 1 and cut["P"] == 2
    assert cut["K_times_P"] == 2
    assert cut["candidate_policy_rows"] == len(tree.internal_nodes) == 191
    assert cut["candidate_internal_node_count"] == 191
    assert cut["allocated_group_count"] == cut["positive_mass_group_count"] == 1
    assert cut["active_nodes_with_ancestry_sizes"][0]["subtree_leaf_count"] == 192
    assert arrays["case_0631__spatial__stage0__environment_group_labels"].shape == (192,)
    assert arrays["case_0631__spatial__stage0__group_relation_mean_headwise"].shape == (1, 2, 2)
    assert arrays["case_0631__spatial__stage0__group_keys"].shape == (1, 2, 4)
    assert arrays["case_0631__spatial__stage0__group_keys"].dtype == np.float32
    assert np.array_equal(
        arrays["case_0631__spatial__stage0__group_keys"],
        np.arange(8, dtype=np.float32).reshape(1, 2, 4),
    )
    assert arrays["case_0631__spatial__stage0__group_value_coefficients"].shape == (1, 2, 2, 3)
    assert arrays["case_0631__spatial__stage0__group_value_coefficients"].dtype == np.float32
    assert np.array_equal(
        arrays["case_0631__spatial__stage0__group_value_coefficients"],
        np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3),
    )
    assert arrays["case_0631__spatial__stage0__all_node_relation_mean_headwise"].shape == (tree.node_count, 2, 2)
    assert arrays["case_0631__spatial__module_coordinates"].shape == (2, 2)
    assert arrays["case_0631__spatial__module_ids"].tolist() == [10, 20]
    assert arrays["case_0631__spatial__module_mask"].tolist() == [True, True]
    assert arrays["case_0631__spatial__module_physical_heat"].tolist() == [2.0, 3.0]


def test_frontier_cut_P_is_retained_value_modes_not_candidate_policy_rows():
    coordinates = np.asarray([(float(x), float(y)) for y in range(8) for x in range(24)])
    tree = build_balanced_grid_template(coordinates, 24, 8).bind(coordinates)
    root = tree.root
    frontier = SimpleNamespace(
        nodes=(root,),
        candidate_nodes=tree.internal_nodes,
        utilities=torch.zeros(len(tree.internal_nodes)),
        policy_rows=len(tree.internal_nodes),
        selection_mode="learned",
        exploration_depth=None,
    )

    for retained_modes in (1, 3):
        group = SimpleNamespace(
            frontier=frontier,
            hierarchy=tree,
            value_coefficients=torch.empty((0, retained_modes, 4, 8)),
            live_group_count=1,
        )
        summary = _frontier_cut_summary(group)
        assert summary["K"] == 1
        assert summary["P"] == retained_modes
        assert summary["K_times_P"] == retained_modes
        assert summary["candidate_policy_rows"] == 191
        assert summary["candidate_internal_node_count"] == 191


def test_native_coordinate_export_saves_fluid_and_reconstructs_material_physical_xy():
    scene = SimpleNamespace(
        module_coordinates=torch.tensor([[[2.0, 3.0], [5.0, 6.0]]]),
        module_characteristic_lengths=torch.tensor([[[0.5, 0.5], [1.0, 1.0]]]),
        module_ids=torch.tensor([[10, 20]]),
        module_mask=torch.tensor([[True, True]]),
        environment_coordinates=torch.zeros((1, 192, 2)),
        environment_mask=torch.ones((1, 192), dtype=torch.bool),
        coordinate_origin=torch.tensor([[0.0, 0.0]]),
        coordinate_scale=torch.tensor([[12.0, 6.0]]),
        coordinate_frame_key="thermal_physical_rectangular_xy_v2",
    )
    prepared = SimpleNamespace(
        scene_batch=scene,
        source_present=torch.tensor([[True, True]]),
    )
    prediction = SimpleNamespace(native_prepared=prepared)
    fluid_xy = torch.tensor([[[0.25, 0.5], [1.25, 1.5]]])
    local_xy = torch.tensor([[[[0.0, 0.0], [1.0, -1.0]], [[0.5, 0.5], [-0.5, -0.5]]]])
    batch = SimpleNamespace(
        receivers=SimpleNamespace(fluid_xy=fluid_xy, local_query_points=local_xy),
    )

    arrays = _native_coordinate_arrays(prediction, batch)

    assert np.array_equal(arrays["fluid_query_coordinates"], fluid_xy.numpy()[0])
    assert arrays["material_receiver_coordinates"].shape == (2, 2, 2)
    assert np.allclose(
        arrays["material_receiver_coordinates"],
        np.asarray([[[2.0, 3.0], [2.5, 2.5]], [[5.5, 6.5], [4.5, 5.5]]]),
    )
    assert arrays["material_receiver_module_ids"].tolist() == [[10, 10], [20, 20]]
    assert arrays["coordinate_frame_key"].item() == "thermal_physical_rectangular_xy_v2"


def test_spatial_prepared_tensor_bytes_walks_nested_state_and_deduplicates_views():
    shared = torch.arange(8, dtype=torch.float64)
    prepared = SimpleNamespace(
        scene=SimpleNamespace(shared_view=shared[1:]),
        layers=((SimpleNamespace(nested={"left": shared[:4], "right": [shared[4:]]}),),),
        auxiliary=(torch.ones(2, dtype=torch.float32),),
    )

    assert _spatial_prepared_tensor_bytes(prepared) == 8 * 8 + 2 * 4


def test_bounded_control_probe_returns_jvp_and_fd_from_tensor_only_readout(monkeypatch):
    from honf_runtime import thermal_geometry_supervision

    monkeypatch.setattr(
        thermal_geometry_supervision,
        "_native_inputs",
        lambda _record, _module_ids, device, dtype: (
            {
                "module_ids": ("m0", "m1"),
                "t_in": torch.zeros((1, 1), device=device, dtype=dtype),
                "t_wall": torch.zeros((1, 1), device=device, dtype=dtype),
            },
            torch.tensor([[1.0, 2.0]], device=device, dtype=dtype),
        ),
    )

    class _Model(torch.nn.Module):
        heat_scale = 1.0

        def prepare_native(self, _structure, fluid, *, physical_heat, local_query_points, **_kwargs):
            modules = physical_heat.shape[1]
            surface_count = modules * 4
            material_count = modules * local_query_points.shape[1]
            valid = lambda count: torch.ones((1, count), dtype=torch.bool, device=physical_heat.device)
            return SimpleNamespace(
                stencils={
                    "fluid": SimpleNamespace(valid=valid(fluid.shape[1])),
                    "surface": SimpleNamespace(valid=valid(surface_count)),
                    "outside": SimpleNamespace(valid=valid(surface_count)),
                    "material": SimpleNamespace(valid=valid(material_count)),
                },
                source_present=torch.ones((1, modules), dtype=torch.bool, device=physical_heat.device),
                organization=SimpleNamespace(
                    layers=((SimpleNamespace(frontier=SimpleNamespace(nodes=(1,))),),)
                ),
            )

        def apply_native(self, _prepared, controls):
            signal = controls.sum(dim=1, keepdim=True)
            fluid = signal.expand(-1, 3)
            field = torch.stack((fluid, fluid, fluid, fluid, fluid), dim=-1)
            surface = controls[:, :, None].expand(-1, -1, 4)
            interface = torch.stack((surface, surface * 0.1), dim=-1)
            material = controls[:, :, None, None].expand(-1, -1, 2, 1)
            return {
                "pred_field": field,
                "pred_interface": interface,
                "pred_internal_temperature": material,
            }

    record = SimpleNamespace(design=SimpleNamespace(active_module_ids=("m0", "m1")))
    stencil = SimpleNamespace(anchor_id="0348", baseline=record)
    panel = {
        "fluid_query_features": np.zeros((3, 2), dtype=np.float32),
        "local_query_features": np.zeros((2, 2), dtype=np.float32),
    }

    summary, arrays = _spatial_control_probe(
        _Model(), stencil, panel, device=torch.device("cpu"), dtype=torch.float32,
    )

    assert len(summary["source_column_probe"]["fixed_cut_jvp"]) == 8
    assert len(summary["source_column_probe"]["reprepared_finite_difference"]) == 8
    assert summary["homogeneous_boundary_zero_heat"]["heat_input_vector"] == [0.0, 0.0]
    assert summary["homogeneous_boundary_zero_heat"]["explicit_boundary_inputs"] == {
        "t_in": [[0.0]], "t_wall": [[0.0]],
    }
    assert np.array_equal(arrays["probe_homogeneous_boundary_zero_heat_input"], np.zeros((1, 2)))
    assert np.array_equal(arrays["probe_homogeneous_boundary_t_in"], np.zeros((1, 1)))
    assert np.array_equal(arrays["probe_homogeneous_boundary_t_wall"], np.zeros((1, 1)))
    assert np.allclose(
        summary["source_column_probe"]["fixed_cut_jvp"],
        summary["source_column_probe"]["reprepared_finite_difference"],
        atol=3.0e-4,
    )


def test_round13_checkpoint_schema_is_reconstructed_as_spatial_adapter():
    payload = {
        "arm": "F13-1",
        "experiment_identity": {
            "workflow": "spatial_interaction_groups_joint_training_v1",
            "provider_identity": {
                "schema": "thermal_spatial_interaction_groups_provider_v1",
                "architecture": "F13-1",
                "spatial_interaction_group_schema": "spatial_interaction_group_fields_v1",
                "adapter_state": {
                    "schema": "thermal_spatial_interaction_groups_adapter_v1",
                    "core": {"schema": "spatial_interaction_group_fields_v1"},
                },
                "recipe": {},
            },
        },
    }

    identity, provider_identity, saved = _contextual_checkpoint_parts(payload)

    assert identity["workflow"] == "spatial_interaction_groups_joint_training_v1"
    assert provider_identity["schema"] == "thermal_spatial_interaction_groups_provider_v1"
    assert saved["arm"] == "F13-1"
    assert saved["adapter_state"]["core"]["schema"] == "spatial_interaction_group_fields_v1"


def test_json_safe_converts_nonfinite_values_for_evidence_json():
    result = _json_safe({"finite": np.float64(2.0), "nan": float("nan"), "inf": float("inf")})

    assert result == {"finite": 2.0, "nan": None, "inf": None}


def test_declared_five_covers_are_distinct_legal_24_by_8_cuts():
    coordinates = np.asarray([(float(x), float(y)) for y in range(8) for x in range(24)])
    tree = build_balanced_grid_template(coordinates, 24, 8).bind(coordinates)
    frontier = SimpleNamespace(tree=tree, cover_nearest_k=tree.cover_nearest_k)
    prepared = SimpleNamespace(layers=((SimpleNamespace(frontier=frontier),),))

    covers, details = _fixed_spatial_covers(prepared)

    assert tuple(covers) == ("root_K1", "nearest_K8", "nearest_K32", "nearest_K128", "leaf_K192")
    assert len(set(covers.values())) == 5
    assert [details[name]["K"] for name in covers] == [len(covers[name]) for name in covers]
    assert all(tree.validate_cover(nodes) == nodes for nodes in covers.values())


def test_organizer24_candidate_selection_stratifies_full_depth_size_range():
    coordinates = np.asarray([(float(x), float(y)) for y in range(8) for x in range(24)])
    tree = build_balanced_grid_template(coordinates, 24, 8).bind(coordinates)
    metadata = [
        {"node_id": node, "depth": tree.depths[node], "size": tree.sizes[node]}
        for node in tree.internal_nodes
    ]
    ordered = sorted(metadata, key=lambda item: (item["depth"], item["size"], item["node_id"]))

    choices = [
        _stratified_organizer_candidate(metadata, panel_index=index, panel_count=24)
        for index in range(24)
    ]
    ranks = [rank for rank, _item in choices]
    selected = [item for _rank, item in choices]

    assert ranks[0] == 0 and ranks[-1] == len(ordered) - 1
    assert ranks == sorted(set(ranks))
    assert [item["node_id"] for item in selected] == [ordered[rank]["node_id"] for rank in ranks]
    assert selected[0]["depth"] == min(item["depth"] for item in metadata)
    assert selected[-1]["depth"] == max(item["depth"] for item in metadata)
    assert len({item["depth"] for item in selected}) >= 6


def test_organizer_utility_nodes_cover_three_input_only_depth_bands():
    metadata = [
        {"node_id": node, "depth": depth, "size": max(1, 192 // (2 ** depth))}
        for depth, nodes in ((0, (382,)), (1, (380, 381)), (2, (375, 376, 377, 378)),
                             (3, (360, 361, 362, 363, 364, 365, 366, 367)))
        for node in nodes
    ]

    selected = _stratified_organizer_nodes(metadata, panel_index=3, panel_count=24)

    assert [band for band, _ in selected] == ["coarse", "intermediate", "deep"]
    assert len({int(row["depth"]) for _, row in selected}) == 3
    assert all("node_id" in row and "size" in row for _, row in selected)


def test_utility_probe_aggregate_reports_regret_mae_and_conditional_role_signs():
    rows = [
        {
            "case_id": "train-a",
            "module_count": 3,
            "predicted_normalized_weighted_gain_proxy": 0.2,
            "raw_weighted_native_gain": 2.0,
            "relative_weighted_native_gain": 0.3,
            "selected_action_regret_raw_weighted_native_units": 0.0,
            "relative_rolewise_native_gains": {"a": 0.2},
            "node_depth": 1,
            "depth_band": "coarse",
            "module_count": 3,
            "proxy_action": "children",
            "observed_native_better_action": "children",
        },
        {
            "case_id": "train-b",
            "module_count": 5,
            "predicted_normalized_weighted_gain_proxy": -0.1,
            "raw_weighted_native_gain": 1.0,
            "relative_weighted_native_gain": 0.1,
            "selected_action_regret_raw_weighted_native_units": 0.1,
            "relative_rolewise_native_gains": {"a": 0.1},
            "node_depth": 4,
            "depth_band": "deep",
            "module_count": 5,
            "proxy_action": "parent",
            "observed_native_better_action": "children",
        },
    ]

    summary = _utility_probe_aggregate(rows)

    assert summary["probe_count"] == 2
    assert np.isclose(summary["relative_gain_regression_mae"], 0.15)
    assert np.isclose(summary["mean_raw_weighted_native_gain"], 1.5)
    assert np.isclose(summary["total_selected_action_regret_raw_weighted_native_units"], 0.1)
    assert summary["coverage_by_depth"] == {"1": 1, "4": 1}
    assert summary["coverage_by_depth_band"] == {"coarse": 1, "deep": 1}
    assert summary["coverage_by_module_count"] == {"3": 1, "5": 1}
    assert summary["coverage_by_case_id"] == {"train-a": 1, "train-b": 1}
    assert summary["rolewise_sign_agreement_conditional_on_magnitude"]["a"]["sign_agreement"] == 0.5


def test_relative_gain_matches_training_target_and_is_scale_invariant():
    gain = _gain_pair(2.0, 1.0)
    scaled = _gain_pair(2000.0, 1000.0)

    assert gain["raw_gain"] == 1.0
    assert gain["relative_gain_denominator"] == 1.50000001
    assert np.isclose(gain["relative_gain"], 2.0 / 3.0)
    assert np.isclose(scaled["relative_gain"], gain["relative_gain"], rtol=0, atol=1e-8)

    # The role denominators are independent: physical conversion applies only
    # to raw standardized-MSE differences, not to their relative targets.
    flow = _gain_pair(2.0, 1.0)
    temperature = _gain_pair(20.0, 10.0)
    assert np.isclose(flow["relative_gain"], temperature["relative_gain"])
    role_scales = {"flow/u": 3.0, "temperature": 0.5}
    physical_squared = {
        "flow/u": flow["raw_gain"] * role_scales["flow/u"] ** 2,
        "temperature": temperature["raw_gain"] * role_scales["temperature"] ** 2,
    }
    assert physical_squared == {"flow/u": 9.0, "temperature": 2.5}


def test_canonical_val89_binding_is_temporary_and_matches_native_test_catalog():
    case_ids = _canonical_val89_ids()
    provider = SimpleNamespace(
        validation_case_ids=("dev-a", "dev-b"),
        _validation_index_by_id={"dev-a": 0, "dev-b": 1},
        _validation_batches_cache=("old",),
        catalog_case_ids=case_ids,
        case_split_by_id={case_id: "test" for case_id in case_ids},
    )
    original_index = provider._validation_index_by_id
    original_cache = provider._validation_batches_cache

    with _bound_native_validation_membership(provider, case_ids):
        assert provider.validation_case_ids == case_ids
        assert len(provider._validation_index_by_id) == 89
        assert provider._validation_batches_cache is None

    assert provider.validation_case_ids == ("dev-a", "dev-b")
    assert provider._validation_index_by_id is original_index
    assert provider._validation_batches_cache is original_cache


def test_fixed_train_mean_uses_only_the_declared_retained_training_membership():
    cases = [
        {
            "case_id": "a",
            "interface_target": np.asarray([[[2.0, 10.0], [4.0, 20.0]]]),
            "structure": {"module_present": np.asarray([1.0])},
            "interface_condition_valid_mask": np.asarray([[1.0, 1.0]]),
            "module_internal_temperature_points": np.asarray([[3.0, 5.0]]),
        },
        {
            "case_id": "b",
            "interface_target": np.asarray([[[6.0, 30.0], [8.0, 40.0]]]),
            "structure": {"module_present": np.asarray([1.0])},
            "interface_condition_valid_mask": np.asarray([[1.0, 0.0]]),
            "module_internal_temperature_points": np.asarray([[7.0, 9.0]]),
        },
        {
            "case_id": "outside",
            "interface_target": np.asarray([[[100.0, 100.0]]]),
            "structure": {"module_present": np.asarray([1.0])},
            "interface_condition_valid_mask": np.asarray([[1.0]]),
            "module_internal_temperature_points": np.asarray([[100.0]]),
        },
    ]
    provider = SimpleNamespace(
        training_cases=cases,
        model=SimpleNamespace(normalization_stats={"field_mean_by_channel": [1.0, 2.0]}),
    )

    means = _fixed_train_mean(provider, ("a", "b"))

    assert means == {"field": [1.0, 2.0], "surface": 4.0, "q": 20.0, "material": 6.0}


def test_full600_provider_resolves_train150_from_initialization_lineage_and_manifest():
    train150 = tuple(f"tr{index:03d}" for index in range(150))
    train600 = tuple(f"tr{index:03d}" for index in range(600))
    validation89 = tuple(f"va{index:03d}" for index in range(89))
    source_provider = {"training_case_ids": list(train150)}
    provider = SimpleNamespace(
        manifest={"partitions": {"train": {"case_ids": list(train150)}}},
        training_case_ids=train600,
        validation_case_ids=validation89,
        case_split_by_id={
            **{case_id: "train" for case_id in train600},
            **{case_id: "test" for case_id in validation89},
        },
        initialization_provider=source_provider,
        initialization_identity={"training_case_ids": list(train150)},
    )
    provider_identity = {
        "training_case_ids": list(train600),
        "validation_case_ids": list(validation89),
        "initialization": {"parent_training_case_ids": list(train150)},
    }

    assert not hasattr(provider, "_research_source_payload")
    assert _fixed_train150_ids(provider, provider_identity) == train150


def test_full600_train150_lineage_mismatch_fails_closed():
    train150 = tuple(f"tr{index:03d}" for index in range(150))
    wrong_parent = tuple(f"tr{index:03d}" for index in range(1, 151))
    train600 = tuple(f"tr{index:03d}" for index in range(600))
    provider = SimpleNamespace(
        manifest={"partitions": {"train": {"case_ids": list(train150)}}},
        training_case_ids=train600,
        case_split_by_id={case_id: "train" for case_id in train600},
        initialization_provider={"training_case_ids": list(train150)},
        initialization_identity={"training_case_ids": list(train150)},
    )
    provider_identity = {"initialization": {"parent_training_case_ids": list(wrong_parent)}}

    try:
        _fixed_train150_ids(provider, provider_identity)
    except ValueError as error:
        assert "identities are absent or disagree" in str(error)
    else:
        raise AssertionError("Mismatched full-data baseline source lineage must be rejected.")


def _dev22_lineage_fixture(monkeypatch, *, full_train: bool, corrupt_parent: bool = False):
    dev_train = tuple(f"tr{index:03d}" for index in range(150))
    dev22 = tuple(f"dv{index:03d}" for index in range(22))
    val89 = (*dev22, *(f"va{index:03d}" for index in range(67)))
    manifest_sha = "a" * 64
    manifest = {
        "manifest_sha256": manifest_sha,
        "partitions": {
            "train": {"case_ids": list(dev_train)},
            "test": {"case_ids": list(dev22)},
        },
    }
    parent_validation = tuple(f"bad{index:03d}" for index in range(22)) if corrupt_parent else dev22
    parent_provider = {
        "training_case_ids": list(dev_train),
        "validation_case_ids": list(parent_validation),
        "subset_manifest_sha256": manifest_sha,
    }
    parent_identity = {
        "training_membership": "fixed25_train150_dev22",
        "provider_identity": dict(parent_provider),
    }
    monkeypatch.setattr(
        "honf_runtime.spatial_interaction_group_evaluation._canonical_val89_ids",
        lambda: tuple(val89),
    )
    if full_train:
        active_train = tuple(f"tr{index:03d}" for index in range(600))
        active_validation = tuple(val89)
        membership = "full_train600"
    else:
        active_train = dev_train
        active_validation = dev22
        membership = "fixed25_train150_dev22"
    provider = SimpleNamespace(
        manifest=manifest,
        initialization_provider=parent_provider,
        initialization_identity=parent_identity,
        training_case_ids=active_train,
        validation_case_ids=active_validation,
    )
    experiment_identity = {
        "training_membership": membership,
        "provider_identity": {
            "training_case_ids": list(active_train),
            "validation_case_ids": list(active_validation),
        },
    }
    return provider, experiment_identity, dev22


def test_fixed_dev22_resolver_accepts_dev150_and_full600_val89(monkeypatch):
    dev_provider, dev_identity, expected_dev22 = _dev22_lineage_fixture(
        monkeypatch, full_train=False,
    )
    assert _fixed_dev22_ids(dev_provider, dev_identity) == expected_dev22

    full_provider, full_identity, expected_dev22 = _dev22_lineage_fixture(
        monkeypatch, full_train=True,
    )
    assert len(full_provider.training_case_ids) == 600
    assert len(full_provider.validation_case_ids) == 89
    assert _fixed_dev22_ids(full_provider, full_identity) == expected_dev22


def test_fixed_dev22_resolver_rejects_wrong_full_validation_and_parent_lineage(monkeypatch):
    provider, identity, _expected = _dev22_lineage_fixture(monkeypatch, full_train=True)
    wrong_validation = provider.validation_case_ids[:-1]
    provider.validation_case_ids = wrong_validation
    identity["provider_identity"]["validation_case_ids"] = list(wrong_validation)
    try:
        _fixed_dev22_ids(provider, identity)
    except ValueError as error:
        assert "canonical VAL89" in str(error)
    else:
        raise AssertionError("Full-TRAIN provider with wrong VAL89 membership must be rejected.")

    provider, identity, _expected = _dev22_lineage_fixture(
        monkeypatch, full_train=True, corrupt_parent=True,
    )
    try:
        _fixed_dev22_ids(provider, identity)
    except ValueError as error:
        assert "fixed25 TRAIN150/DEV22" in str(error) or "fixed25_train150_dev22" in str(error)
    else:
        raise AssertionError("Wrong inherited DEV22 source lineage must be rejected.")

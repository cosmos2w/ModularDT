"""Saved observation adapter contracts; synthetic fixtures never solve physics."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from thermal_campaign_atlas_observations import load_final_review_pair


def saved_pair(tmp_path):
    module_ids = np.array(["saved:zeta", "saved:alpha", "saved:middle"])
    centers = np.array([[2., 3.], [5., 2.], [8., 4.]], dtype=np.float32)
    heat = np.array([[1., 2., 3.], [1.125, 1.875, 3.]], dtype=np.float32)
    x, y = np.meshgrid(((np.arange(48)+.5)*12/48).astype(np.float32),
                       ((np.arange(32)+.5)*6/32).astype(np.float32))
    grid = np.stack([x.ravel(), y.ravel()], -1)
    theta = np.arange(8, dtype=np.float32)*np.float32(2*np.pi/8)
    ports = np.stack([theta, np.cos(theta), np.sin(theta)], -1)
    local = np.array([[0., 0.], [.25, .1], [-.1, .2]], dtype=np.float32)
    fields = np.zeros((2, len(grid), 5), dtype=np.float32)
    fields[:, :, 2] = (12-grid[:, 0])[None]
    fields[0, :, 4] = grid[:, 0]+grid[:, 1]
    fields[1, :, 4] = fields[0, :, 4]+.1*(grid[:, 0]+1)
    config = {"domain": {"lx":12., "ly":6., "module_radius":.45, "nx":48, "ny":32},
              "flow": {"re":50., "u_in":1., "nu":.018},
              "thermal": {"solid_alpha":.01, "fluid_alpha":.02, "solid_k":1., "fluid_k":1.},
              "layout": {"centers":centers.tolist(), "heat_powers":heat[0].tolist()},
              "runtime": {"converged":True, "nu":.018}}
    metadata = {"family_id":"generated_family:M3:seed123", "split":"final_review", "records":{}}
    for i, label in enumerate(("baseline", "heat_transfer_plus")):
        directory = tmp_path/label
        directory.mkdir()
        raw = copy.deepcopy(config)
        raw["layout"]["heat_powers"] = heat[i].tolist()
        (directory/"case_config.json").write_text(json.dumps(raw))
        metadata["records"][label] = {"case_dir":str(directory), "source":"reference_solver",
                                       "status":"converged", "runtime":config["runtime"]}
    arrays = {"all_labels":np.array(["baseline", "heat_transfer_plus"]),
              "active_module_ids":module_ids, "module_centers_xy":np.stack([centers, centers]),
              "heating":heat, "split":np.array(["final_review"]*2), "source":np.array(["reference_solver"]*2),
              "family_metadata_json":np.array(json.dumps({"case_config":config,
                  "physical_family_id":metadata["family_id"], "split":"final_review"})),
              "fluid_fields_query_features":grid,
              "fluid_fields_query_ids":np.array([f"grid:{i}" for i in range(len(grid))]),
              "fluid_fields_channel_names":np.array(["u", "v", "p", "omega", "temperature"]),
              "fluid_fields_channel_units":np.array(["velocity", "velocity", "pressure", "vorticity", "temperature"]),
              "fluid_fields_values":fields, "fluid_fields_valid_mask":np.ones_like(fields, dtype=bool),
              "interface_query_features":np.tile(ports,(3,1)),
              "interface_receiver_module_ids":np.repeat(module_ids,len(ports)),
              "solid_temperature_query_features":np.tile(local,(3,1)),
              "solid_temperature_receiver_module_ids":np.repeat(module_ids,len(local))}
    path = tmp_path/"pair.npz"
    np.savez(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(metadata))
    return path, arrays, metadata


def test_public_sample_has_no_hidden_targets_and_preserves_source_order(tmp_path):
    path, arrays, _ = saved_pair(tmp_path)
    (baseline, hidden), (plus, plus_hidden) = load_final_review_pair(path)
    public = baseline.public_sample
    assert set(public) == {"structure", "x_grid", "y_grid", "interface_condition", "module_internal_query_points"}
    assert set(public["structure"]) == {"module_centers", "module_present", "material_params", "re", "u_in",
                                      "domain_length_x", "domain_length_y", "public_total_heat"}
    assert not hasattr(baseline, "heat") and not hasattr(baseline, "held_temperatures")
    assert public is plus.public_sample and baseline.source_id_to_slot == {"saved:zeta":0,"saved:alpha":1,"saved:middle":2}
    np.testing.assert_array_equal(public["structure"]["module_centers"][:3], arrays["module_centers_xy"][0])
    np.testing.assert_array_equal(public["structure"]["material_params"], np.array([.018,.01,.02,1.,1.,.45],np.float32))
    assert public["structure"]["re"].item() == 50 and baseline.public_total_heat == plus.public_total_heat == 6
    assert public["interface_condition"].shape == (12,8,3)
    assert public["module_internal_query_points"].shape == (3,2)
    assert hidden.heat.shape == (12,) and np.count_nonzero(hidden.heat) == 3
    assert hidden.heat.sum() == plus_hidden.heat.sum() == baseline.public_total_heat
    assert hidden.case_id == baseline.case_id and plus_hidden.case_id == plus.case_id
    np.testing.assert_array_equal(hidden.heat[:3], arrays["heating"][0])
    assert not np.array_equal(hidden.heat, plus_hidden.heat)
    assert baseline.metadata["partition"] == "final_review" and baseline.metadata["previously_exposed"]
    assert baseline.metadata["source"] == "reference_solver"
    with pytest.raises(ValueError, match="read-only"):
        hidden.heat[0] = 99
    with pytest.raises(TypeError):
        public["structure"]["heat_powers"] = hidden.heat


def test_named_observed_and_held_values_select_exact_stored_rows(tmp_path):
    path, arrays, _ = saved_pair(tmp_path)
    records = load_final_review_pair(path)
    for i, (public, hidden) in enumerate(records):
        assert public.sensor_names[-2:] == ("inlet_midline", "outlet_midline")
        assert public.sensor_coordinates.shape == (14,2)
        assert len(set(public.sensor_grid_rows[:12])) == 12
        np.testing.assert_array_equal(public.observed_rows, [0,2,4,6,8,10])
        np.testing.assert_array_equal(public.held_rows, [1,3,5,7,9,11])
        assert not set(public.observed_rows)&set(public.held_rows)
        np.testing.assert_array_equal(public.sensor_coordinates, arrays["fluid_fields_query_features"][public.sensor_grid_rows])
        np.testing.assert_array_equal(public.observed_temperatures,
            arrays["fluid_fields_values"][i,public.sensor_grid_rows[public.observed_rows],4])
        np.testing.assert_array_equal(hidden.held_temperatures,
            arrays["fluid_fields_values"][i,public.sensor_grid_rows[public.held_rows],4])
        assert hidden.held_query_ids == tuple(public.sensor_query_ids[row] for row in public.held_rows)
        assert hidden.held_sensor_names == tuple(public.sensor_names[row] for row in public.held_rows)
    assert not np.array_equal(records[0][0].observed_temperatures,records[1][0].observed_temperatures)


@pytest.mark.parametrize("mismatch", ["geometry", "total", "negative", "partition", "source", "ids", "grid", "bounds", "local", "family"])
def test_invalid_saved_pairs_are_rejected(tmp_path, mismatch):
    path, arrays, _ = saved_pair(tmp_path)
    if mismatch == "geometry": arrays["module_centers_xy"][1,0,0] += .1
    elif mismatch == "total": arrays["heating"][1,0] += .25
    elif mismatch == "negative": arrays["heating"][1,0] = -.5
    elif mismatch == "partition": arrays["split"][1] = "calibration"
    elif mismatch == "source": arrays["source"][1] = "surrogate"
    elif mismatch == "ids": arrays["active_module_ids"][1] = arrays["active_module_ids"][0]
    elif mismatch == "grid": arrays["fluid_fields_query_features"][[1,2]] = arrays["fluid_fields_query_features"][[2,1]]
    elif mismatch == "bounds": arrays["fluid_fields_query_features"][:,0] += .5
    elif mismatch == "local": arrays["solid_temperature_query_features"][3,0] += .1
    elif mismatch == "family":
        metadata = json.loads(path.with_suffix(".json").read_text())
        metadata["family_id"] = "other_family"
        path.with_suffix(".json").write_text(json.dumps(metadata))
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        load_final_review_pair(path)


@pytest.mark.parametrize("role", ["observed", "held", "pressure"])
def test_invalid_named_reference_masks_rejected(tmp_path, role):
    path, arrays, _ = saved_pair(tmp_path)
    public = load_final_review_pair(path)[0][0]
    row = public.observed_rows[0] if role == "observed" else public.held_rows[0] if role == "held" else 12
    channel = 2 if role == "pressure" else 4
    arrays["fluid_fields_valid_mask"][1, public.sensor_grid_rows[row], channel] = False
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="valid"):
        load_final_review_pair(path)


@pytest.mark.parametrize("mismatch", ["context", "material", "source_order", "missing_config", "status"])
def test_per_solve_provenance_is_checked_independently(tmp_path, mismatch):
    path, _, metadata = saved_pair(tmp_path)
    config_path = tmp_path/"heat_transfer_plus/case_config.json"
    config = json.loads(config_path.read_text())
    if mismatch == "context": config["flow"]["re"] = 100.
    elif mismatch == "material": config["thermal"]["solid_k"] = 2.
    elif mismatch == "source_order": config["layout"]["centers"].reverse()
    elif mismatch == "status":
        metadata["records"]["heat_transfer_plus"]["status"] = "failed"
        path.with_suffix(".json").write_text(json.dumps(metadata))
    if mismatch == "missing_config": config_path.unlink()
    else: config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_final_review_pair(path)


def test_changed_hidden_held_target_does_not_enter_public_record(tmp_path):
    path, arrays, _ = saved_pair(tmp_path)
    original, hidden = load_final_review_pair(path)[0]
    arrays["fluid_fields_values"][0,original.sensor_grid_rows[original.held_rows],4] += 100
    np.savez(path, **arrays)
    revised, revised_hidden = load_final_review_pair(path)[0]
    np.testing.assert_array_equal(original.observed_temperatures, revised.observed_temperatures)
    np.testing.assert_array_equal(original.public_sample["structure"]["module_centers"],revised.public_sample["structure"]["module_centers"])
    assert not np.array_equal(hidden.held_temperatures,revised_hidden.held_temperatures)


@pytest.mark.parametrize("family_id", ["", ".", "..", "parent/child", "parent\\child"])
def test_family_id_is_a_single_output_path_component(tmp_path, family_id):
    path, arrays, metadata = saved_pair(tmp_path)
    metadata["family_id"] = family_id
    family = json.loads(str(arrays["family_metadata_json"].item()))
    family["physical_family_id"] = family_id
    arrays["family_metadata_json"] = np.array(json.dumps(family))
    np.savez(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="path component"):
        load_final_review_pair(path)

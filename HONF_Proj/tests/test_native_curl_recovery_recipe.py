from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

def _runner():
    path = REPO_ROOT / "tools" / "joint_regional_train.py"
    name = "native_curl_candidate_runner_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def _recipe(mode: str, run_id: str):
    cli = _runner()
    recipe_name = {
        "P": "thermal_native_curl_interaction_preserving_p.json",
        "P-G": "thermal_native_curl_interaction_preserving_p_g.json",
        "P-H": "thermal_native_curl_interaction_preserving_p_h.json",
    }[mode]
    recipe_path = REPO_ROOT / "src/config_core/forward/joint_regional" / recipe_name
    recipe = copy.deepcopy(cli.read_recipe(recipe_path))
    recipe.update({
        "mode": mode, "run_id": run_id,
        "flow_readout_law": "native_curl_cell_centred_v1",
        "model_contract": cli._recovery_model_contract(
            "thermal", mode, "native_curl_cell_centred_v1"),
        "calibration_receipt": {
            "receipt_path": "diagnostics/generated/interaction_recovery_20261010/thermal_native_curl_fresh_P_calibration.json",
            "required_payload_sha256": "sealed_after_fresh_P_native_curl_calibration",
        },
        "auxiliary_calibration": None,
    })
    if mode == "P":
        recipe["regional_anchors"] = 0
        recipe["locality_prior_strength"] = 0.0
    else:
        recipe["regional_anchors"] = 16
        recipe["locality_prior_strength"] = 1.0
        recipe["model_contract"]["collective"]["enabled"] = True
    return cli, recipe

@pytest.mark.parametrize(("mode", "run_id"), [("P", "T4111"), ("P-G", "T4112"), ("P-H", "T4113")])
def test_native_curl_recipe_contract_is_opt_in_fresh_and_three_output(mode, run_id):
    cli, recipe = _recipe(mode, run_id)
    checked = cli.validate_recipe(recipe)
    assert checked["flow_readout_law"] == "native_curl_cell_centred_v1"
    flow = checked["model_contract"]["readouts"]["flow_head"]
    assert flow == {"law": "nonlinear source-conditioned field read", "outputs": 3,
                    "order": ["u", "v", "p"], "initialization_reference_field_outputs": 4}
    assert checked["model_contract"]["readouts"]["derived_omega"]["omega_projection_rows"] == 0
    assert checked["auxiliary_calibration"] is None

def test_native_curl_recipe_rejects_wind_j_family_wrong_law_and_reused_calibration():
    cli, recipe = _recipe("P", "T4111")
    with pytest.raises(ValueError):
        cli.validate_recipe({**recipe, "flow_readout_law": "other"})
    with pytest.raises(ValueError):
        cli.validate_recipe({**recipe, "task": "wind", "run_id": "W2401"})
    with pytest.raises(ValueError):
        cli.validate_recipe({**recipe, "mode": "J-H", "run_id": "J-H"})
    with pytest.raises(ValueError, match="Unresolved Thermal calibration"):
        cli.validate_recipe({**recipe, "auxiliary_calibration": {"payload_sha256": "old-four-output"}})

def test_native_curl_only_extends_source_identity_for_new_law():
    cli = _runner()
    native = cli._recovery_source_files("native_curl_cell_centred_v1")
    assert native[:-1] == cli.RECOVERY_SOURCE_FILES
    assert native[-1] == "HONF_Proj/Case_ThermalChannel/src/channelthermal/flow_curl.py"
    assert cli._recovery_source_files() == cli.RECOVERY_SOURCE_FILES


def test_native_curl_fresh_p_calibration_receipt_binds_law_specific_sentinel(tmp_path, monkeypatch):
    cli, recipe = _recipe("P", "T4111")
    cli.ROOT = tmp_path
    law = "native_curl_cell_centred_v1"
    active_identity = {"source_set_sha256": "a" * 64, "source_git_commit": "fixture-commit"}

    def mock_source_identity(*, require_clean, flow_readout_law=None):
        assert require_clean is True
        assert flow_readout_law == law
        return active_identity

    monkeypatch.setattr(cli, "training_source_identity", mock_source_identity)
    payload = {
        "training_source_identity": active_identity,
        "flow_readout_law": law,
        "response_coefficient": 0.75,
        "operator_coefficient": 0.125,
    }
    payload["payload_sha256"] = cli._canonical_payload_sha256(payload)
    receipt_path = tmp_path / recipe["calibration_receipt"]["receipt_path"]
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(payload, sort_keys=True))

    resolved = cli._bind_calibration_receipt(recipe)

    assert resolved["calibration_receipt"]["required_payload_sha256"] == payload["payload_sha256"]
    assert resolved["calibration_receipt"]["file_sha256"] == cli.hashlib.sha256(
        receipt_path.read_bytes()).hexdigest()
    assert resolved["auxiliary_calibration"] == payload
    assert resolved["response_coefficient"] == 0.75
    assert resolved["operator_coefficient"] == 0.125


def test_native_curl_fresh_p_calibration_receipt_rejects_mismatched_law(tmp_path, monkeypatch):
    cli, recipe = _recipe("P", "T4111")
    cli.ROOT = tmp_path
    active_identity = {"source_set_sha256": "b" * 64, "source_git_commit": "fixture-commit"}
    monkeypatch.setattr(cli, "training_source_identity", lambda **kwargs: active_identity)
    payload = {
        "training_source_identity": active_identity,
        "flow_readout_law": "different_law",
        "response_coefficient": 0.75,
        "operator_coefficient": 0.125,
    }
    payload["payload_sha256"] = cli._canonical_payload_sha256(payload)
    receipt_path = tmp_path / recipe["calibration_receipt"]["receipt_path"]
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(payload, sort_keys=True))

    with pytest.raises(ValueError, match="different flow readout law"):
        cli._bind_calibration_receipt(recipe)


def test_legacy_unresolved_calibration_still_requires_auxiliary_key_absent():
    cli = _runner()
    path = REPO_ROOT / "src/config_core/forward/joint_regional/thermal_interaction_preserving_p.json"
    recipe = cli.read_recipe(path)
    assert "auxiliary_calibration" not in recipe
    assert cli.validate_recipe(recipe)["calibration_receipt"]["required_payload_sha256"] == (
        "sealed_after_fresh_P_calibration"
    )

    with pytest.raises(ValueError, match="Unresolved Thermal calibration"):
        cli.validate_recipe({**recipe, "auxiliary_calibration": None})

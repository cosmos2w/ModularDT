from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
for _directory in (PROJECT / "src", PROJECT / "Case_ThermalChannel/src"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from channelthermal.contextual_joint import ThermalContextualJointAdapter
from honf_runtime import contextual_frontier_training
from honf_runtime.contextual_frontier_evaluation import evaluate_native, load_contextual_checkpoint


def _normalization_stats():
    return {
        "field_mean_by_channel": [0.0, 0.0, 0.0, 0.0, 10.0],
        "field_std_by_channel": [1.0, 1.0, 1.0, 1.0, 2.0],
        "interface_targets_std": [2.0, 3.0],
        "internal_temperature_std": [4.0],
    }


def _model(architecture: str = "J-D"):
    return ThermalContextualJointAdapter(
        _normalization_stats(),
        heat_scale=2.0,
        nx=8,
        ny=6,
        core_config={
            "schema": "contextual_organization_joint_v1",
            "architecture": architecture,
            "module_input_dim": 9,
            "environment_input_dim": 6,
            "operating_input_dim": 12,
            "coordinate_dim": 2,
            "output_dim": 4,
            "context_dim": 16,
            "local_hidden_dim": 16,
            "num_heads": 4,
            "context_depth": 1,
            "init_seed": 0,
        },
    )


def test_checkpoint_loader_uses_saved_adapter_and_preserves_saved_dtype(tmp_path: Path, monkeypatch) -> None:
    saved_model = _model().double()
    recipe = {"dataset": {"name": "thermal"}, "evaluation": {"max_output_bytes": 1024}}
    provider_identity = {
        "schema": "thermal_contextual_frontier_provider_v1",
        "architecture": "J-D",
        "adapter_state": saved_model.state_payload(),
        "recipe": recipe,
    }
    payload = {
        "workflow": "contextual_frontier_joint_training_v1",
        "arm": "J-D",
        "epoch": 20,
        "model_state_dict": saved_model.state_dict(),
        "experiment_identity": {
            "workflow": "contextual_frontier_joint_training_v1",
            "provider_identity": provider_identity,
        },
    }
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint)

    fresh_model = _model()
    provider = SimpleNamespace(
        model=fresh_model,
        adapter=fresh_model,
        organizer=SimpleNamespace(model=fresh_model, core=fresh_model.core),
    )
    factory_calls = []

    def fake_build_task(config, arm, device):
        factory_calls.append((config, arm, device))
        return fresh_model, provider

    monkeypatch.setattr(contextual_frontier_training, "_build_task", fake_build_task)
    loaded, loaded_provider, loaded_payload = load_contextual_checkpoint(checkpoint, "cpu")

    assert loaded_payload["checkpoint_path"] == str(checkpoint.resolve())
    assert factory_calls == [(recipe, "J-D", torch.device("cpu"))]
    assert loaded_provider is provider
    assert loaded_provider.model is loaded
    assert loaded_provider.adapter is loaded
    assert loaded_provider.organizer.model is loaded
    assert loaded_provider.organizer.core is loaded.core
    assert next(loaded.parameters()).dtype == torch.float64
    for key, value in saved_model.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(loaded.state_dict()[key], value)


def _production_anchored_checkpoint(mode, locator):
    from channelthermal.member_routing import ThermalAnchoredMemberRoutingAdapter
    from honf_forward_core.interaction_reused_member_routing.anchored import AnchoredMemberRoutingCore
    from tests.test_r13_8_anchored_core_helpers import _base_config

    core = AnchoredMemberRoutingCore(_base_config(), mediated_mode=mode, locator_mode=locator,
                                    source_retained_mass=.91, environment_retained_mass=.87)
    model = ThermalAnchoredMemberRoutingAdapter(_normalization_stats(), core=core).double()
    arm = model.architecture
    adapter_state = model.state_payload()
    recipe = {"member_routing": {"arm": arm, "model": dict(adapter_state["core"])}}
    provider_identity = {
        "schema": "thermal_r13_8_e_anchored_member_routing_provider_v1",
        "architecture": arm,
        "member_routing_schema": "interaction_reused_e_anchored_member_routing_v1",
        "adapter_state": adapter_state,
        "recipe": recipe,
    }
    return model, {"arm": arm, "epoch": 2, "model_state_dict": model.state_dict(),
                   "experiment_identity": {
                       "workflow": "interaction_reused_e_anchored_member_routing_training_v1",
                       "provider_identity": provider_identity}}


@pytest.mark.parametrize("mode", ("none", "P", "G"))
@pytest.mark.parametrize("locator", ("e_anchor", "local_e_cover"))
def test_production_anchored_checkpoint_loader_preserves_historical_and_mediated_states(
    tmp_path: Path, monkeypatch, mode, locator,
) -> None:
    # Real adapter metadata and real parameter state exercise the public loader,
    # including Adapter.from_state_payload, rather than just the arm allowlist.
    model, payload = _production_anchored_checkpoint(mode, locator)
    path = tmp_path / "anchored.pt"
    torch.save(payload, path)
    provider = SimpleNamespace(model=None, adapter=None)
    monkeypatch.setattr(contextual_frontier_training, "_build_task", lambda *_args: (model, provider))
    loaded, actual_provider, actual_payload = load_contextual_checkpoint(path, "cpu")
    assert actual_provider.model is loaded and actual_provider.adapter is loaded
    assert loaded.architecture == model.architecture
    assert loaded.core.locator_mode == locator and loaded.core.mediated_mode == mode
    assert actual_payload["epoch"] == 2
    assert next(loaded.parameters()).dtype == torch.float64
    assert loaded.get_config() == model.get_config()
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], tensor, rtol=0, atol=0)


@pytest.mark.parametrize("field,value", (("mediated_mode", "G"), ("source_retained_mass", .9),
                                         ("environment_retained_mass", .9)))
def test_mediated_checkpoint_loader_rejects_recipe_equation_mismatch(tmp_path: Path, field, value) -> None:
    _, payload = _production_anchored_checkpoint("P", "e_anchor")
    payload["experiment_identity"]["provider_identity"]["recipe"]["member_routing"]["model"][field] = value
    path = tmp_path / "mismatched.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="adapter, core locator, access law"):
        load_contextual_checkpoint(path, "cpu")


class _FakeCore:
    def __init__(self):
        self.clock = None

    def set_organization_clock(self, epoch, update):
        self.clock = (epoch, update)


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.architecture = "J-H"
        self.core = _FakeCore()


def _native_batch(case_id: str, module_count: int):
    present = torch.zeros((1, module_count), dtype=torch.float32)
    present[0] = 1
    field_target = torch.zeros((1, 3, 5), dtype=torch.float32)
    field_target[0, :, 4] = 10
    interface_target = torch.zeros((1, module_count, 2, 2), dtype=torch.float32)
    interface_target[..., 0] = 12
    material_target = torch.full((1, module_count, 2), 14.0)
    targets = SimpleNamespace(
        case_ids=(case_id,),
        field_targets=field_target,
        point_weights=torch.ones((1, 3)),
        interface_target=interface_target,
        interface_valid_mask=torch.ones((1, module_count, 2)),
        material_targets=material_target,
    )
    receivers = SimpleNamespace(heat=torch.ones((1, module_count)))
    return SimpleNamespace(
        case_keys=(case_id,),
        scene_inputs=SimpleNamespace(case_ids=(case_id,)),
        targets=targets,
        receivers=receivers,
    )


class _FakeProvider:
    def __init__(self, model):
        self.model = model
        self.adapter = model
        self.device = torch.device("cpu")
        self.validation_case_ids = ("dev-a", "dev-b", "dev-c")
        self.validation_cases = tuple(
            {"case_id": case_id, "structure": {"module_present": np.ones(count, dtype=np.float32)}}
            for case_id, count in (("dev-a", 3), ("dev-b", 5), ("dev-c", 12))
        )
        self.training_cases = (
            {"case_id": "train-m7", "structure": {"module_present": np.ones(7, dtype=np.float32)}},
        )
        self.batches = (_native_batch("dev-a", 3), _native_batch("dev-b", 5), _native_batch("dev-c", 12))

    def native_evaluation_batches(self, *, split, case_keys, microbatch_size):
        assert split == "validation"
        assert tuple(case_keys) == self.validation_case_ids
        assert microbatch_size == 1
        yield from self.batches

    def predict_native(self, model, scene, receivers, arm, phase, *, epoch):
        assert model is self.model
        assert arm == "J-H" and phase == "validation" and epoch == 20
        case_id = scene.case_ids[0]
        batch = next(item for item in self.batches if item.case_keys[0] == case_id)
        module_count = batch.receivers.heat.shape[1]
        pred_field = batch.targets.field_targets.clone()
        pred_field[..., 0] += 1.0
        pred_field[..., 1] += 2.0
        pred_field[..., 2] += 3.0
        pred_field[..., 3] += 4.0
        pred_field[..., 4] += 5.0
        pred_interface = batch.targets.interface_target + 1.0
        pred_material = batch.targets.material_targets[..., None] + 2.0
        prepared = SimpleNamespace(
            stencils={
                "fluid": SimpleNamespace(valid=torch.ones((1, 3), dtype=torch.bool)),
                "surface": SimpleNamespace(valid=torch.ones((1, module_count, 2), dtype=torch.bool)),
                "outside": SimpleNamespace(valid=torch.ones((1, module_count, 2), dtype=torch.bool)),
                "material": SimpleNamespace(valid=torch.ones((1, module_count, 2), dtype=torch.bool)),
            },
            source_present=torch.ones((1, module_count)),
        )
        prediction = SimpleNamespace(
            native_main={
                "pred_field": pred_field,
                "pred_interface": pred_interface,
                "pred_internal_temperature": pred_material,
            },
            native_prepared=prepared,
        )
        return prediction, {"work": {"native_fluid_queries": 3}}

    def validation_metrics(self, prediction, targets, auxiliary):
        del auxiliary
        case_id = targets.case_ids[0]
        module_count = int(targets.interface_valid_mask.shape[1])
        return {
            "case_rows": [{
                "case_id": case_id,
                "module_count": module_count,
                "flow_rmse": {"u": 1.0, "v": 2.0, "p": 3.0, "omega": 4.0},
                "roles": {
                    "temperature_fluid": {"rmse": 5.0},
                    "temperature_surface": {"rmse": 6.0},
                    "temperature_material": {"rmse": 7.0},
                    "temperature_material_endpoint_maxima": {
                        "per_module": [{"error": -8.0}, {"error": 9.0}],
                    },
                },
                "q_proxy_rmse": 8.0,
                "flow_unit": "native_flow",
                "temperature_unit": "native_temperature",
            }],
        }

    def reduce_native_evaluation_metrics(self, records, *, split, case_keys):
        rows = [row for record in records for row in record["case_rows"]]
        assert split == "validation"
        assert tuple(row["case_id"] for row in rows) == tuple(case_keys)
        return {"case_ids": list(case_keys), "per_case": rows, "scope": "fixed DEV22 test panel"}

    def work_counts(self, batch, prediction, auxiliary):
        del batch, prediction
        return auxiliary["work"]


def test_native_evaluation_streams_cases_and_writes_bounded_numerical_evidence(tmp_path: Path, monkeypatch) -> None:
    model = _FakeModel()
    provider = _FakeProvider(model)
    recipe = {
        "evaluation": {"max_output_bytes": 1024 * 1024, "representative_case_limit": 4},
        "storage": {"minimum_free_headroom_bytes": 0},
    }
    payload = {
        "epoch": 20,
        "arm": "J-H",
        "experiment_identity": {
            "workflow": "contextual_frontier_joint_training_v1",
            "provider_identity": {
                "schema": "thermal_contextual_frontier_provider_v1",
                "architecture": "J-H",
                "adapter_state": {},
                "validation_scope": "fixed25_v1_DEV22_exposed",
                "subset_manifest_sha256": "manifest",
                "recipe": recipe,
            },
        },
    }
    monkeypatch.setattr(
        "honf_runtime.contextual_frontier_evaluation.load_contextual_checkpoint",
        lambda checkpoint, device: (model, provider, payload),
    )
    monkeypatch.setattr(
        "honf_runtime.contextual_frontier_evaluation._filesystem_free_bytes",
        lambda output_parent: 2 * 1024**3,
    )
    checkpoint = tmp_path / "latest_model.pt"
    checkpoint.write_bytes(b"trusted fixture path")
    output = tmp_path / "evaluation"

    result = evaluate_native(checkpoint, "cpu", output)

    saved = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["case_count"] == 3
    assert saved["checkpoint_selection"]["kind"] == "latest"
    assert saved["checkpoint_epoch"] == 20
    assert saved["metrics"]["u"]["mean"] == 1.0
    assert saved["metrics"]["u"]["p90"] == 1.0
    peak = saved["metrics"]["module_peak_endpoint_rmse"]
    assert peak["mean"] == pytest.approx(np.sqrt((8.0**2 + 9.0**2) / 2.0))
    assert peak["worst_case_id"] == "dev-a"
    per_case_peaks = saved["maintained_native_reduction"]["per_case"][0]["roles"]
    assert len(per_case_peaks["temperature_material_endpoint_maxima"]["per_module"]) == 2
    assert saved["metrics"]["temperature_material"]["by_module_count"]["M5"]["mean"] == 7.0
    assert saved["metrics"]["temperature_material"]["by_module_count"]["M12"]["case_count"] == 1
    assert saved["metrics"]["u"]["by_module_count"]["M7"] == {
        "case_count": 0,
        "mean": None,
        "p90": None,
        "worst": None,
        "worst_case_id": None,
    }
    assert saved["executed_work_counts"]["native_fluid_queries"] == 9
    assert len(saved["representative_exports"]) == 3
    with np.load(output / "case_dev-a.npz", allow_pickle=False) as arrays:
        assert arrays["pred_field"].shape == (3, 5)
        assert arrays["target_field"].shape == (3, 5)
        assert arrays["pred_surface_temperature"].shape == (3, 2)
        assert arrays["pred_material_temperature"].shape == (3, 2)
        assert arrays["case_id"].item() == "dev-a"
    with pytest.raises(FileExistsError):
        evaluate_native(checkpoint, "cpu", output)

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import channelthermal.training.joint_task as joint_task_module
import numpy as np
import pytest
import torch
from channelthermal.flow_curl import NATIVE_CURL_READOUT_LAW
from channelthermal.training.joint_task import JointThermalTask
from channelthermal.training.unified_task import TRAIN_RESPONSE_IDS


class _CalibrationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = torch.nn.Linear(1, 1)
        self.mode = "P"
        self.flow_readout_law = NATIVE_CURL_READOUT_LAW
        self.seed = 714
        self.receiver_tile = 8

    def prepare_native(self, *args, **kwargs):
        return SimpleNamespace()

    def apply_native(self, *args, **kwargs):
        return torch.ones((), dtype=torch.float32)


def _response_sample(family_id: str):
    return {
        "family_id": family_id,
        "structure": {"module_present": torch.ones(1, 1)},
        "fluid_xy": torch.zeros(1, 1, 2),
        "local": torch.zeros(1, 1, 2),
        "heat": torch.ones(1, 1),
        "native_solid_mask": torch.zeros(1, 1, 1, dtype=torch.bool),
        "fluid_target": torch.zeros(1, 1),
        "surface_target": torch.zeros(1, 1),
        "surface_mask": torch.ones(1, 1),
        "material_target": torch.zeros(1, 1),
        "material_mask": torch.ones(1, 1),
    }


def test_native_curl_provider_calibration_emits_and_validates_its_law_receipt(tmp_path, monkeypatch):
    task = object.__new__(JointThermalTask)
    task.joint_mode = "P"
    task.flow_readout_law = NATIVE_CURL_READOUT_LAW
    task._validation_ids = []
    task._train_ids = [f"case_{index:03d}" for index in range(150)]
    counts = [1, 3, 10, 12] + [2] * 146
    task.training_cases = [
        {"structure": {"module_present": np.ones(count, dtype=np.float32)}} for count in counts
    ]
    task.model = _CalibrationModel()
    task.device = torch.device("cpu")
    task.receiver_tile = task.model.receiver_tile
    task.auxiliary_calibration = None
    task.response_weight = 0.1
    task.operator_weight = 0.1
    task.response_coefficient = 0.1
    task.operator_coefficient = 0.1
    task.q_proxy_weight = float(joint_task_module.Q_PROXY_COEFFICIENT)
    task.operator_rows_per_case = joint_task_module.OPERATOR_ROWS_PER_CASE
    task.budget = {
        "fluid_queries": 32,
        "surface_stride": 8,
        "material_queries_per_module": 16,
    }
    task.manifest = {"manifest_sha256": "m" * 64}
    task.source_binding = {"flow_readout_law": NATIVE_CURL_READOUT_LAW}
    task.stats = {"flow": torch.tensor([0.0, 1.0])}
    task.atlas_directory = Path(tmp_path)
    task.train_families = [{"family_id": family_id} for family_id in TRAIN_RESPONSE_IDS]
    for family_id in TRAIN_RESPONSE_IDS:
        (Path(tmp_path) / f"train_{family_id}_responses.npz").write_bytes(family_id.encode())

    task._batch_from_indices = lambda *args, **kwargs: SimpleNamespace(
        scene_inputs={}, receivers=None, targets={},
    )
    task.make_scene = lambda inputs: inputs
    task.predict_native = lambda *args, **kwargs: (
        SimpleNamespace(native_main=None, native_prepared=None), None,
    )
    task._native_role_losses = lambda *args, **kwargs: {
        "flow_group": torch.tensor([1.0]), "thermal_group": torch.tensor([1.0]),
    }
    task._operator_loss = lambda *args, **kwargs: (torch.tensor([1.0]), {})
    task._prepare_response_sample = lambda family, **kwargs: _response_sample(str(family["family_id"]))
    task._response_loss = lambda *args, **kwargs: torch.tensor([1.0])
    task._build_loss_metadata = dict
    monkeypatch.setattr(joint_task_module, "_gradient_norm", lambda *args, **kwargs: 1.0)

    receipt = JointThermalTask._calibrate_interaction_preserving_auxiliary_coefficients(task)

    assert receipt["flow_readout_law"] == NATIVE_CURL_READOUT_LAW
    assert receipt["response_family_ids"] == list(TRAIN_RESPONSE_IDS)
    assert receipt["validation_values_read"] is False
    assert receipt["optimizer_steps"] == 0
    assert task.auxiliary_calibration == receipt
    with pytest.raises(ValueError, match="different flow readout law"):
        JointThermalTask._validate_auxiliary_calibration(
            task, {**receipt, "flow_readout_law": "other_law"},
        )

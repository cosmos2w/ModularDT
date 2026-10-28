"""The maintained checkpoint loader reconstructs both registered score laws."""
import copy
from types import SimpleNamespace

import pytest
import torch

from channelthermal.member_routing import ThermalAnchoredMemberRoutingAdapter
from honf_forward_core.interaction_reused_member_routing import AnchoredMemberRoutingCore
from honf_runtime.contextual_frontier_evaluation import (
    _contextual_checkpoint_parts, load_contextual_checkpoint,
)
from tests.test_interaction_reused_member_routing import _base_config
from tests.test_native_budgeted_postfit import _thermal_stats


@pytest.mark.parametrize("calibration", ("inherited", "bounded_tanh"))
def test_anchored_loader_roundtrips_score_law_and_state(tmp_path, monkeypatch, calibration):
    core = AnchoredMemberRoutingCore(_base_config(), locator_mode="e_anchor",
                                    input_dataset="thermal", score_calibration=calibration)
    model = ThermalAnchoredMemberRoutingAdapter(_thermal_stats(), core=core)
    arm = model.architecture
    identity = {"workflow": "interaction_reused_e_anchored_member_routing_training_v1",
                "provider_identity": {
                    "schema": "thermal_r13_8_e_anchored_member_routing_provider_v1",
                    "architecture": arm,
                    "adapter_state": model.state_payload(),
                    "recipe": {"member_routing": {"arm": arm, "model": core.get_config()}},
                    "member_routing_schema": core.schema,
                }}
    payload = {"experiment_identity": identity, "arm": arm, "model_state_dict": model.state_dict()}
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    provider = SimpleNamespace()
    monkeypatch.setattr("honf_runtime.contextual_frontier_training._build_task",
                        lambda recipe, arm, device: (model, provider))
    restored, restored_provider, _ = load_contextual_checkpoint(path, "cpu")
    assert restored.core.score_calibration == calibration
    assert restored_provider.model is restored
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    mismatched = copy.deepcopy(payload)
    mismatched["experiment_identity"]["provider_identity"]["recipe"]["member_routing"]["model"]["score_calibration"] = (
        "bounded_tanh" if calibration == "inherited" else "inherited"
    )
    with pytest.raises(ValueError, match="adapter, core locator"):
        _contextual_checkpoint_parts(mismatched)

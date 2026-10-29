"""Retained Thermal six-hyperedge and dense checkpoints resume the next update exactly."""
from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from channelthermal.config import ChannelThermalHONFConfig
from channelthermal.evaluation.loading import load_model
from channelthermal.model import ChannelThermalHONFModel
from channelthermal.training.checkpoints import _restore_rng_state, _validate_resume_checkpoint, save_checkpoint
from honf_runtime.compat import load_trusted_checkpoint


def _config(architecture):
    core = {
        "forward_architecture": architecture, "hidden_dim": 16, "field_dim": 5,
        "domain_length_x": 12.0, "domain_length_y": 4.0, "coordinate_scale": [12.0, 4.0],
        "module_radius": 0.45, "num_env_tokens_x": 4, "num_env_tokens_y": 3,
        "dropout": 0.1, "boundary_feature_mode": "none", "position_fourier_frequencies": 2,
        "query_fourier_frequencies": 2,
    }
    if architecture == "legacy_honf":
        core.update(num_hyperedges=6, decoder_mode="enhanced_honf_pairwise_only",
                    use_hyper_value_context=False, pairwise_aggregation_mode="fused_query_module",
                    pairwise_module_token_source="organizer_contextualized", routing_execution="dense",
                    query_module_retained_mass_floor=1.0, pairwise_kernel_hidden_dim=16)
    else:
        core["interface_model"] = {"message_hidden_dim": 16, "attention_heads": 4,
            "coarse_latent_count": 4, "coarse_blocks": 1, "relative_fourier_frequencies": 2,
            "receiver_chunk_size": 3, "activation_checkpointing": False}
    return ChannelThermalHONFConfig.from_dict({"core_honf": core, "channelthermal": {
        "use_local_surrogate": False, "internal_prediction_mode": "global_head",
        "default_num_interface_points": 8, "fallback_hidden_dim": 16,
    }})


def _inputs():
    return {"query_xy": torch.tensor([[[0.5, 0.4], [5.0, 2.0], [10.0, 3.0]]]),
            "re": torch.tensor([[100.0]]), "u_in": torch.tensor([[1.0]]),
            "module_centers": torch.tensor([[[3.0, 1.0], [8.0, 2.0]]]),
            "heat_powers": torch.tensor([[1.0, 2.0]]), "module_present": torch.ones(1, 2),
            "material_params": torch.tensor([[0.01, 0.02, 0.03, 1.0, 0.5, 0.45]])}


def _step(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    # Use all three saved RNG streams, plus dropout in the actual model.
    offset = random.random() + float(np.random.uniform())
    target = torch.randn(1, 3, 5) + offset
    loss = (model(**_inputs())["pred_field"] - target).square().mean()
    loss.backward()
    optimizer.step()


def _assert_equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("architecture", ["legacy_honf", "dense_pairwise_field"])
def test_legacy_checkpoint_loader_and_next_optimizer_update_match(architecture, tmp_path):
    random.seed(57)
    np.random.seed(57)
    torch.manual_seed(57)
    cfg = _config(architecture)
    original = ChannelThermalHONFModel(cfg, attach_local_from_checkpoint=False)
    original.train()
    with torch.no_grad():
        original(**_inputs())  # Materialize lazy layers before creating AdamW.
    original.set_training_progress(epoch=1, total_epochs=100)
    optimizer = torch.optim.AdamW(original.parameters(), lr=3e-4, weight_decay=1e-5)
    _step(original, optimizer)
    dataset = SimpleNamespace(channel_order=["u", "v", "p", "omega", "T"], field_dim=5,
        interface_condition_feature_names=[], interface_target_names=[], max_num_modules=2,
        selected_module_counts=[2], normalizer=SimpleNamespace(stats={"field_std": np.ones(5, np.float32)}))
    dataset_config = {"dataset_id": "fixture", "dataset_schema": "test", "dataset_fingerprint": "fixture"}
    train_config = {"dataset": dataset_config, "training": {"epochs": 100}}
    path = tmp_path / "latest_model.pt"
    save_checkpoint(path, model=original, model_config=cfg, train_config=train_config, dataset=dataset,
                    epoch=1, best_metric=1.0, optimizer=optimizer)
    saved = load_trusted_checkpoint(path, map_location="cpu")
    _step(original, optimizer)
    expected_model = copy.deepcopy(original.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())

    resumed, loaded = load_model(path, torch.device("cpu"))
    _validate_resume_checkpoint(loaded, model=resumed, model_config=cfg, dataset=dataset,
                                dataset_config=dataset_config)
    assert list(original.state_dict()) == list(resumed.state_dict()) == list(saved["model_state_dict"])
    assert [name for name, _ in original.named_parameters()] == [name for name, _ in resumed.named_parameters()]
    assert resumed.selection_state() == saved["selection_state"]
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=3e-4, weight_decay=1e-5)
    resumed_optimizer.load_state_dict(saved["optimizer_state_dict"])
    assert resumed_optimizer.state_dict()["state"]
    resumed.train()
    _restore_rng_state(saved)
    _step(resumed, resumed_optimizer)
    _assert_equal(resumed.state_dict(), expected_model)
    _assert_equal(resumed_optimizer.state_dict(), expected_optimizer)


def test_workflow_checkpoint_injection_is_scoped_even_on_failure(monkeypatch):
    from channelthermal.evaluation import loading
    from channelthermal.workflows import evaluate_forward

    previous = loading.load_trusted_checkpoint
    injected = lambda *args, **kwargs: None
    monkeypatch.setattr(evaluate_forward, "load_trusted_checkpoint", injected)

    def fail(path, device):
        assert loading.load_trusted_checkpoint is injected
        raise RuntimeError("loader failed")

    monkeypatch.setattr(loading, "load_model", fail)
    with pytest.raises(RuntimeError, match="loader failed"):
        evaluate_forward.load_model("unused", torch.device("cpu"))
    assert loading.load_trusted_checkpoint is previous

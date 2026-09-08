"""Focused Stage-2 configuration and profile checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from honf_forward_core.config import UnifiedForwardConfig
from honf_runtime.config_loader import load_config_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_URI = "project://src/config_core/forward/sparse_interface_honf_context.json"
GEOMETRY_PROFILE_URI = "project://src/config_core/forward/sparse_interface_geometry_read_context.json"


def _interface_settings(*, support_spacing_factor: float | None = None) -> dict[str, object]:
    settings: dict[str, object] = {
        "message_hidden_dim": 128,
        "attention_heads": 4,
        "coarse_latent_count": 8,
        "coarse_blocks": 1,
        "local_radius_factor": 2.5,
        "relative_fourier_frequencies": 4,
        "receiver_chunk_size": 128,
        "activation_checkpointing": True,
    }
    if support_spacing_factor is not None:
        settings["support_spacing_factor"] = support_spacing_factor
    return settings


def test_sparse_interface_profile_is_complete_matched_and_registered() -> None:
    profile_path = PROJECT_ROOT / "src/config_core/forward/sparse_interface_honf_context.json"
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    bundle = load_config_bundle(PROFILE_URI)
    core = bundle.effective["model"]["core_honf"]

    assert payload["profile_name"] == "sparse_interface_honf_context"
    assert core["forward_architecture"] == "sparse_interface_honf"
    assert core["interface_model"] == {
        **_interface_settings(support_spacing_factor=4.0),
    }
    assert "main_latent_count" not in core["interface_model"]
    assert "main_latent_blocks" not in core["interface_model"]
    assert "organizer_mode" not in core
    assert "decoder_mode" not in core
    assert core["hidden_dim"] == 256
    assert (core["num_env_tokens_x"], core["num_env_tokens_y"]) == (24, 8)
    assert payload["training"]["epochs"] == 500
    assert payload["training"]["device"] == "cuda:0"
    assert payload["training"]["learning_rate"] == pytest.approx(3.0e-4)
    assert payload["training"]["weight_decay"] == pytest.approx(1.0e-5)
    assert payload["training"]["amp"] is False
    assert payload["training"]["gradient_clip_norm"] == pytest.approx(1.0)
    assert payload["run"]["id"] == "1802"

    registry = json.loads(
        (PROJECT_ROOT / "src/config_core/forward/profile_registry.json").read_text(encoding="utf-8")
    )
    entry = next(item for item in registry["profiles"] if item["name"] == payload["profile_name"])
    assert entry["status"] == "candidate"
    assert entry["base"] is None
    assert registry["recommended_forward_profile"] == "stage7_structured_context"


def test_sparse_spacing_is_architecture_specific_and_serializes_cleanly() -> None:
    sparse = UnifiedForwardConfig.from_dict(
        {
            "forward_architecture": "sparse_interface_honf",
            "hidden_dim": 256,
            "interface_model": _interface_settings(support_spacing_factor=4.0),
        }
    )
    serialized = sparse.to_dict()
    assert serialized["interface_model"]["support_spacing_factor"] == pytest.approx(4.0)
    assert "main_latent_count" not in serialized["interface_model"]
    assert "main_latent_blocks" not in serialized["interface_model"]

    with pytest.raises(ValueError, match="requires interface_model.support_spacing_factor"):
        UnifiedForwardConfig.from_dict(
            {
                "forward_architecture": "sparse_interface_honf",
                "hidden_dim": 256,
                "interface_model": _interface_settings(),
            }
        )
    with pytest.raises(ValueError, match="must be positive"):
        UnifiedForwardConfig.from_dict(
            {
                "forward_architecture": "sparse_interface_honf",
                "hidden_dim": 256,
                "interface_model": _interface_settings(support_spacing_factor=-1.0),
            }
        )

    dense = UnifiedForwardConfig.from_dict(
        {
            "forward_architecture": "dense_pairwise_field",
            "hidden_dim": 256,
            "interface_model": _interface_settings(),
        }
    )
    assert "support_spacing_factor" not in dense.to_dict()["interface_model"]
    with pytest.raises(ValueError, match="only valid for sparse_interface_honf"):
        UnifiedForwardConfig.from_dict(
            {
                "forward_architecture": "dense_pairwise_field",
                "hidden_dim": 256,
                "interface_model": _interface_settings(support_spacing_factor=4.0),
            }
        )


def test_group_reader_mode_defaults_historically_and_candidate_profile_is_registered() -> None:
    historical = UnifiedForwardConfig.from_dict(
        {
            "forward_architecture": "sparse_interface_honf",
            "hidden_dim": 256,
            "interface_model": _interface_settings(support_spacing_factor=4.0),
        }
    )
    assert historical.interface_model is not None
    assert historical.interface_model.group_read_mode == "null_softmax"

    candidate = UnifiedForwardConfig.from_dict(
        {
            "forward_architecture": "sparse_interface_honf",
            "hidden_dim": 256,
            "interface_model": _interface_settings(
                support_spacing_factor=4.0,
            ) | {"group_read_mode": "geometry_envelope_attention"},
        }
    )
    assert candidate.interface_model is not None
    assert candidate.interface_model.group_read_mode == "geometry_envelope_attention"
    with pytest.raises(ValueError, match="group_read_mode"):
        UnifiedForwardConfig.from_dict(
            {
                "forward_architecture": "sparse_interface_honf",
                "hidden_dim": 256,
                "interface_model": _interface_settings(
                    support_spacing_factor=4.0,
                ) | {"group_read_mode": "unsupported"},
            }
        )

    payload = json.loads(
        (PROJECT_ROOT / "src/config_core/forward/sparse_interface_geometry_read_context.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["profile_name"] == "sparse_interface_geometry_read_context"
    assert payload["model"]["core_honf"]["interface_model"]["group_read_mode"] == "geometry_envelope_attention"
    assert payload["run"] == {
        "id": "1805",
        "name": "sparse_interface_geometry_read",
        "output_root": "project://Trained_Results",
    }
    bundle = load_config_bundle(GEOMETRY_PROFILE_URI)
    core = bundle.effective["model"]["core_honf"]
    assert core["interface_model"]["group_read_mode"] == "geometry_envelope_attention"
    registry = json.loads(
        (PROJECT_ROOT / "src/config_core/forward/profile_registry.json").read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in registry["profiles"]
        if item["name"] == "sparse_interface_geometry_read_context"
    )
    assert entry["status"] == "candidate"
    assert entry["base"] is None

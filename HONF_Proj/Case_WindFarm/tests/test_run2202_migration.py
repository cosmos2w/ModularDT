from __future__ import annotations

import builtins
import os
from pathlib import Path

import pytest
import torch

from windfarm import migration
from windfarm.migration import map_run2202_direct_state
from windfarm.normalization import VelocityNormalizer, VerticalProfileBaseline


def _matching_states() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    target = {
        f"core.direct.layer_{index:02d}": torch.zeros((index % 3 + 1,), dtype=torch.float32)
        for index in range(52)
    }
    source = {key: value.clone() for key, value in target.items()}
    source.update({f"core.refinement.layer_{index:02d}": torch.ones(1) for index in range(14)})
    source["unrelated.training_buffer"] = torch.ones(1)
    return source, target


def test_run2202_direct_mapping_selects_exactly_the_52_compatible_tensors() -> None:
    source, target = _matching_states()

    mapped = map_run2202_direct_state(source, target)

    assert set(mapped) == set(target)
    assert len(mapped) == 52
    assert all(torch.equal(mapped[key], target[key]) for key in target)
    assert not any(key.startswith("core.refinement.") for key in mapped)


def test_run2202_direct_mapping_rejects_missing_target_tensor() -> None:
    source, target = _matching_states()
    target.pop("core.direct.layer_00")

    with pytest.raises(ValueError, match="exactly the existing 52-key P bridge"):
        map_run2202_direct_state(source, target)


def test_run2202_direct_mapping_rejects_extra_target_tensor() -> None:
    source, target = _matching_states()
    target["core.unmapped"] = torch.zeros(2)

    with pytest.raises(ValueError, match="exactly the existing 52-key P bridge"):
        map_run2202_direct_state(source, target)


def test_run2202_direct_mapping_rejects_shape_mismatch() -> None:
    source, target = _matching_states()
    source["core.direct.layer_00"] = torch.zeros(2)

    with pytest.raises(ValueError, match="has shape"):
        map_run2202_direct_state(source, target)


def test_run2202_direct_mapping_rejects_dtype_mismatch() -> None:
    source, target = _matching_states()
    source["core.direct.layer_00"] = source["core.direct.layer_00"].to(torch.float64)

    with pytest.raises(ValueError, match="dtype"):
        map_run2202_direct_state(source, target)


def test_run2202_direct_mapping_rejects_non_tensor_state_values() -> None:
    source, target = _matching_states()
    source["core.direct.layer_00"] = [0.0]

    with pytest.raises(TypeError, match="must be a tensor"):
        map_run2202_direct_state(source, target)


def test_embedded_normalizer_must_match_p_config_after_local_digest_recompute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = {"schema_version": 1, "bin_centers_D": [0.2, 0.8], "values_mps": [7.0, 8.0]}
    normalization = {
        "schema_version": 1,
        "mean_dimensionless": [0.1, 0.2, 0.3],
        "vertical_profile_baseline": profile,
    }
    config = {
        "velocity_transform": {
            key: value for key, value in normalization.items() if key != "vertical_profile_baseline"
        },
        "background_profile": profile,
    }
    initial_digest = migration.stable_json_sha256(normalization)
    monkeypatch.setattr(migration, "RUN2202_NORMALIZATION_DOCUMENT_SHA256", initial_digest)
    migration._validate_embedded_normalizer_binding(
        normalization, config, profile, normalization_document_sha256=initial_digest
    )

    tampered = dict(normalization)
    tampered["mean_dimensionless"] = [0.9, 0.2, 0.3]
    recomputed_digest = migration.stable_json_sha256(tampered)
    monkeypatch.setattr(migration, "RUN2202_NORMALIZATION_DOCUMENT_SHA256", recomputed_digest)
    with pytest.raises(ValueError, match="P config, embedded TRAIN normalizer and saved profile"):
        migration._validate_embedded_normalizer_binding(
            tampered, config, profile, normalization_document_sha256=recomputed_digest
        )


def test_run2202_p_config_rejects_self_rehashed_tokenization_tamper() -> None:
    normalizer = VelocityNormalizer(
        mean=torch.zeros(3).numpy(),
        std=torch.ones(3).numpy(),
        safe_std=torch.ones(3).numpy(),
        sample_count_per_row=1,
        source_rows=420,
    )
    profile = VerticalProfileBaseline(
        bin_centers_D=torch.tensor([0.5, 1.5]).numpy(),
        values_mps=torch.zeros((2, 3)).numpy(),
        counts=torch.ones(2, dtype=torch.int64).numpy(),
    )
    profile_document = profile.to_dict()
    normalization_document = {
        **normalizer.to_dict(),
        "vertical_profile_baseline": profile_document,
    }
    config = migration._expected_run2202_p_config(normalization_document, profile_document, seed=42)
    migration._validate_run2202_p_config(config, normalization_document, profile_document, seed=42)

    tampered = {**config, "environment_token_shape": [2, 2, 4]}
    recomputed_local_digest = migration.stable_json_sha256(tampered)
    assert recomputed_local_digest != migration.stable_json_sha256(config)
    with pytest.raises(ValueError, match="literal source-bound constructor, E8 tokenization"):
        migration._validate_run2202_p_config(
            tampered, normalization_document, profile_document, seed=42
        )


def test_self_contained_reload_reads_only_migration_artifact_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_value = os.environ.get("RUN2202_MIGRATION_ARTIFACT")
    if not artifact_value:
        pytest.skip("Set RUN2202_MIGRATION_ARTIFACT to run the archived-artifact isolation check.")
    artifact = Path(artifact_value).expanduser().resolve()
    if not artifact.is_file():
        pytest.skip(f"Configured migration artifact is unavailable: {artifact}")

    run_dir = Path(__file__).resolve().parents[2] / (
        "Trained_Results/WindFarm/HONF_Forward_Runs/"
        "Run_2202_20261008_194400_w3_h128_full_train420_q4096_e5000"
    )
    forbidden_normalizer_names = {"train_only_normalization.json", "normalization.json"}
    source_reads: list[str] = []
    trusted_loads: list[str] = []

    def path_string(candidate: object) -> str | None:
        if isinstance(candidate, int):
            return None
        try:
            return str(Path(os.fspath(candidate)).expanduser().resolve())
        except (TypeError, ValueError):
            return None

    def reject_source_path(candidate: object) -> None:
        resolved = path_string(candidate)
        if resolved is None:
            return
        source_reads.append(resolved)
        source_path = Path(resolved)
        if source_path == artifact:
            return
        if source_path.is_relative_to(run_dir) or source_path.name in forbidden_normalizer_names:
            raise AssertionError(f"Self-contained reload attempted external source read: {resolved}")

    original_builtin_open = builtins.open
    original_path_open = Path.open
    original_os_open = os.open
    original_checkpoint_loader = migration.load_trusted_checkpoint

    def guarded_builtin_open(file: object, *args: object, **kwargs: object):
        reject_source_path(file)
        return original_builtin_open(file, *args, **kwargs)

    def guarded_path_open(self: Path, *args: object, **kwargs: object):
        reject_source_path(self)
        return original_path_open(self, *args, **kwargs)

    def guarded_os_open(path: object, flags: int, *args: object, **kwargs: object):
        reject_source_path(path)
        return original_os_open(path, flags, *args, **kwargs)

    def guarded_checkpoint_load(candidate: str | Path, *args: object, **kwargs: object):
        resolved = str(Path(candidate).expanduser().resolve())
        trusted_loads.append(resolved)
        if resolved != str(artifact):
            raise AssertionError(f"Migration loader attempted another checkpoint load: {resolved}")
        return original_checkpoint_loader(candidate, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_builtin_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    monkeypatch.setattr(migration, "load_trusted_checkpoint", guarded_checkpoint_load)

    model, provenance = migration.load_run2202_migration(artifact, map_location="cpu")

    assert trusted_loads == [str(artifact)]
    assert str(artifact) in source_reads
    assert model.training is False
    assert provenance["source_checkpoint_sha256"] == migration.RUN2202_EPOCH_CHECKPOINT_SHA256[5000]
    assert provenance["legacy_feature_and_training_identity"] == provenance["source_identity"]["provider_identity"]
    assert len(provenance["source_checkpoint_metadata"]["history"]) == 5000
    assert provenance["source_checkpoint_metadata"]["optimizer_state_by_name"]

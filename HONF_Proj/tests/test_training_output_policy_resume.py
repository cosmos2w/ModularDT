"""Output-only continuation keeps every scientific and lineage binding sealed."""

import copy

import pytest

from honf_runtime.training_output_policy import (
    OUTPUT_POLICY_SCHEMA, canonical_sha256, source_changes, validate_output_policy_amendment,
)


def identities():
    recipe = {"schema": "content_hypergraph_recipe_v2", "dataset": "thermal", "architecture": "A2-D",
              "protocol": "formal_d_full_train_v1", "total_epochs": 5000, "approved_stop_after": 5000,
              "seed": 0, "execution": {"monitor_every": 10, "curve_every": 100}, "model": {"width": 128}}
    old_files = {"src/honf_runtime/unified_training.py": "a" * 64}
    saved = {"workflow": "content_hypergraph_joint_v2", "recipe": recipe,
             "recipe_sha256": canonical_sha256(recipe),
             "loaded_source": {"files": old_files, "sha256": canonical_sha256(old_files)},
             "provider_identity": {"recipe": copy.deepcopy(recipe), "TRAIN": "full600", "normalizer": "TRAIN_only", "native_law": "v2"},
             "engine_config": {"seed": 0, "microbatch_cases": 24, "effective_cases": 48,
                               "curve_every": 100, "monitor_every": 10, "total_epochs": 5000},
             "selection_policy": {"field_metric": "field_score"},
             "optimizer_schedule_contract": {"lr": 3e-4, "weight_decay": 1e-5}}
    current = copy.deepcopy(saved)
    current["recipe"]["execution"]["curve_every"] = 10
    current["recipe_sha256"] = canonical_sha256(current["recipe"])
    current["provider_identity"]["recipe"] = copy.deepcopy(current["recipe"])
    current["engine_config"].pop("curve_every")  # New default is omitted from the compatibility payload.
    new_files = {"src/honf_runtime/unified_training.py": "b" * 64,
                 "src/honf_runtime/training_output_policy.py": "c" * 64}
    current["loaded_source"] = {"files": new_files, "sha256": canonical_sha256(new_files)}
    declaration = {"schema": OUTPUT_POLICY_SCHEMA, "source_epoch": 216,
                   "source_checkpoint_sha256": "d" * 64, "run_directory": "/round/thermal",
                   "reason": "user requested output and progress changes only",
                   "saved_loaded_source_sha256": saved["loaded_source"]["sha256"],
                   "current_loaded_source_sha256": current["loaded_source"]["sha256"],
                   "source_changes": source_changes(saved["loaded_source"], current["loaded_source"])}
    return saved, current, declaration


def validate(saved, current, declaration, **overrides):
    arguments = {"source_sha256": "d" * 64, "source_epoch": 216, "stop_after": 5000,
                 "run_directory": "/round/thermal"}
    arguments.update(overrides)
    return validate_output_policy_amendment(saved, current, declaration, **arguments)


def test_explicit_output_only_continuation_keeps_the_scientific_contract():
    saved, current, declaration = identities()
    original = copy.deepcopy((saved, current, declaration))
    result = validate(saved, current, declaration)
    assert result["kind"] == "explicit_training_output_policy_continuation"
    assert result["output_policy"] == {"curve_every": 10, "validation_sidecars": "declared_milestones_and_final", "progress_enabled": True}
    assert (saved, current, declaration) == original


@pytest.mark.parametrize("change", ["seed", "batch", "queries", "normalization", "native_law", "model", "selection", "schedule", "validation", "horizon", "H6"])
def test_output_continuation_rejects_scientific_changes(change):
    saved, current, declaration = identities()
    if change == "seed": current["engine_config"]["seed"] = 42
    elif change == "batch": current["engine_config"]["effective_cases"] = 24
    elif change == "queries": current["provider_identity"]["queries"] = 4096
    elif change == "normalization": current["provider_identity"]["normalizer"] = "quarter_TRAIN"
    elif change == "native_law": current["provider_identity"]["native_law"] = "changed"
    elif change == "model": current["recipe"]["model"]["width"] = 256
    elif change == "selection": current["selection_policy"]["field_metric"] = "another"
    elif change == "schedule": current["optimizer_schedule_contract"]["lr"] = 1e-4
    elif change == "validation": current["engine_config"]["monitor_every"] = 1
    elif change == "horizon": current["engine_config"]["total_epochs"] = 6000
    elif change == "H6": current["recipe"]["architecture"] = "A2-H6"
    with pytest.raises(ValueError, match="scientific and lineage"):
        validate(saved, current, declaration)


@pytest.mark.parametrize("overrides", [{"source_sha256": "wrong"}, {"source_epoch": 215},
                                      {"stop_after": 5001}, {"stop_after": 1000},
                                      {"run_directory": "/another/run"}])
def test_output_continuation_rejects_wrong_checkpoint_run_or_stop(overrides):
    with pytest.raises(ValueError):
        validate(*identities(), **overrides)


def test_output_continuation_rejects_unreviewed_source_or_tampered_manifest():
    saved, current, declaration = identities()
    current["loaded_source"]["files"]["src/honf_forward_core/model.py"] = "f" * 64
    current["loaded_source"]["sha256"] = canonical_sha256(current["loaded_source"]["files"])
    declaration["current_loaded_source_sha256"] = current["loaded_source"]["sha256"]
    declaration["source_changes"] = source_changes(saved["loaded_source"], current["loaded_source"])
    with pytest.raises(ValueError): validate(saved, current, declaration)
    saved, current, declaration = identities()
    declaration["source_changes"].pop()
    with pytest.raises(ValueError): validate(saved, current, declaration)
    saved, current, declaration = identities()
    current["loaded_source"]["sha256"] = "wrong"
    with pytest.raises(ValueError): validate(saved, current, declaration)

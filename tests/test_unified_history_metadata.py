import copy
import json

from honf_runtime.unified_training import _compact_training_history, _expand_training_history, _write_training_history, read_training_history


def test_compact_roundtrip_preserves_all_reviews_and_variable_work(tmp_path):
    history = []
    for epoch in (1, 2):
        context = {"schema": "v1", "source": "scratch", "epoch": epoch}
        objective = {"terms": {"u": epoch / 10}, "definition": "native", "training_reduction": "sampled"}
        history.append({"epoch": epoch, "arm": "S", "phase": "joint", "reader_epoch_context": context,
                        "paired_reader_native_losses": {"epoch_context": copy.deepcopy(context), "full": epoch},
                        "work_counts": {"always_zero": 0, "varying": epoch - 1},
                        "validation": {"per_case": [{"id": "a", "value": epoch}], "loss_objective": objective},
                        "validation_objective": copy.deepcopy(objective)})
    original = copy.deepcopy(history)
    rows, metadata = _compact_training_history(history)
    assert history == original
    assert _expand_training_history(rows, metadata) == original
    assert metadata["zero_work_counts"] == {"always_zero": 0}
    assert metadata["shared"] == {"arm": "S", "phase": "joint"}
    assert len(metadata["objectives"]) == 1
    path = tmp_path / "history.json"
    _write_training_history(path, history, compact=True)
    assert read_training_history(path) == original
    _write_training_history(path, history, compact=False)
    assert json.loads(path.read_text()) == original
    assert read_training_history(path) == original  # stale sidecar is ignored


def test_metadata_update_keeps_previous_history_readable_across_interruption(tmp_path):
    from honf_runtime.unified_training import _atomic_json

    path = tmp_path / "history.json"
    old = [{"epoch": 1, "arm": "S", "phase": "joint", "work_counts": {"idle": 0},
            "reader_epoch_context": {"source": "one", "epoch": 1},
            "validation": {"loss_objective": {"definition": "first", "terms": {"u": 0.1}}}}]
    _write_training_history(path, old, compact=True)
    new = copy.deepcopy(old)
    new.append({"epoch": 2, "arm": "S", "phase": "hard", "work_counts": {"idle": 4},
                "reader_epoch_context": {"source": "two", "epoch": 2},
                "validation": {"loss_objective": {"definition": "second", "terms": {"u": 0.2}}}})
    new.append({"epoch": 3, "work_counts": {}, "reader_epoch_context": {}})
    metadata_path = tmp_path / "history_metadata.json"
    old_metadata = json.loads(metadata_path.read_text())
    rows, extended_metadata = _compact_training_history(new, old_metadata)
    _atomic_json(metadata_path, extended_metadata)
    assert read_training_history(path) == old
    assert _expand_training_history(rows, extended_metadata) == new
    _write_training_history(path, new, compact=True)
    assert read_training_history(path) == new

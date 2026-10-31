"""Exercise the case entry point's training and sampling configuration."""

import importlib.util
from pathlib import Path


_spec = importlib.util.spec_from_file_location(
    "relation_anchored_train", Path(__file__).resolve().parents[1] / "tools/relation_anchored_train.py")
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


class _Cases:
    def identity_payload(self):
        return {"dataset": "WindFarm", "subset_manifest_sha256": "same-training-membership"}

    def epoch_cases(self, epoch, seed):
        return tuple(range(72))


def test_joint_horizon_output_and_case_stream_are_independent_of_microbatch():
    recipe = {"seed": 42, "microbatch_size": 1, "effective_batch_size": 24, "total_epochs": 200}
    one = cli.engine_for(recipe, device="cpu")
    four = cli.engine_for({**recipe, "microbatch_size": 4}, device="cpu")
    assert one.epoch_case_order(_Cases(), 1) == four.epoch_case_order(_Cases(), 1)
    assert one.config.sampling_version == "case_epoch_v1"
    assert one.config.training_mode == "joint"
    assert one.config.total_epochs == 200
    assert one.config.checkpoint_epochs == (100, 200)
    assert one.config.latest_every == 1
    assert one.config.curve_every == 10
    assert one.config.write_initial_artifacts


def test_formal_horizon_is_fixed_independently_of_requested_stop():
    engine = cli.engine_for({"dataset": {"name": "thermal"}, "training": {
        "seed": 0, "microbatch_cases": 6, "effective_batch_size": 48, "local_horizon": 500,
    }}, device="cpu")
    assert engine.config.total_epochs == 500
    assert engine.config.checkpoint_epochs == (100, 200, 300, 400, 500)


def test_thermal_recipes_declare_horizon_beyond_first_stop():
    import json

    root = Path(__file__).resolve().parents[1] / "configs/research"
    for name in ("joint", "localports"):
        for population, horizon in ((150, 200), (600, 500)):
            recipe = json.loads((root / f"thermal_r15_2_{name}_train{population}_v1.json").read_text())
            engine = cli.engine_for(recipe, device="cpu")
            assert engine.config.total_epochs == recipe["training"]["optimizer_schedule_horizon"] == horizon
            assert recipe["training"]["first_segment_epochs"] == 100
            assert engine.config.checkpoint_epochs == tuple(range(100, horizon + 1, 100))


def test_fit_passes_resume_and_stop_to_the_existing_engine(monkeypatch, capsys):
    import json

    recipe = {"dataset": "wind", "architecture": "relation_anchored"}
    monkeypatch.setattr(Path, "read_text", lambda self: json.dumps(recipe))
    model, provider = object(), object()
    monkeypatch.setattr(cli, "build", lambda value, device: (model, provider))
    calls = {}

    class Engine:
        def fit(self, actual_model, actual_provider, output, **kwargs):
            calls.update(model=actual_model, provider=actual_provider, output=output, **kwargs)
            return {"completed_epoch": 500}

    monkeypatch.setattr(cli, "engine_for", lambda value, device: Engine())
    assert cli.main(["fit", "--recipe", "run.json", "--output", "run", "--resume", "latest",
                     "--stop-after", "500"]) == 0
    assert calls["model"] is model and calls["provider"] is provider
    assert calls["resume_checkpoint"] == "latest" and calls["stop_after"] == 500
    assert calls["identity"]["recipe"] == recipe
    assert json.loads(capsys.readouterr().out)["completed_epoch"] == 500


def test_total_curves_include_weighted_port_terms(monkeypatch):
    import pytest
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    import honf_runtime.unified_training as runtime

    plotted = {}
    original_plot = Axes.plot

    def capture(axis, x, y, *args, **kwargs):
        if kwargs.get("label") in ("TRAIN: recorded weighted sum", "VALIDATION: recorded total"):
            plotted[kwargs["label"]] = list(y)
        return original_plot(axis, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", capture)
    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "_atomic_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime.os, "replace", lambda *args, **kwargs: None)
    history = [{"epoch": 1, "phase": "joint", "arm": "R14-Detail-LocalECover",
                "train_losses": {"flow/u": 0.4, "port/outside_temperature": 0.2, "port/h": 0.1},
                "validation_objective": {"terms": {"flow/u": 0.3,
                    "port/outside_temperature": 0.15, "port/h": 0.05}}}]
    runtime._render_loss_curves(history, Path("/tmp"), "field_score")
    assert plotted["TRAIN: recorded weighted sum"] == pytest.approx([0.7])
    assert plotted["VALIDATION: recorded total"] == pytest.approx([0.5])

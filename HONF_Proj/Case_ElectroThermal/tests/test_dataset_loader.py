from hetmf.dataset import PilotDataset
from hetmf.environment import generate_environment


def test_model_inputs_whitelist_excludes_generation_and_target_metadata():
    dataset = PilotDataset.__new__(PilotDataset)
    dataset.scenes = {"x": {"environment": generate_environment(1, "correlated_mosaic"),
                            "contacts": [{"id": "pad-0", "x": 0.2, "y": 0.3, "r": 0.07,
                                          "gamma": 2.0, "u": 0.5, "secret_provenance": "excluded"}],
                            "split": "test", "terminal_current": 1.0}}
    inputs = dataset.inputs("x")
    assert set(inputs) == {"environment_records", "contacts", "renderer", "units"}
    assert len(inputs["environment_records"]) == 144
    assert "family" not in inputs and "seed" not in inputs
    assert "secret_provenance" not in inputs["contacts"][0]

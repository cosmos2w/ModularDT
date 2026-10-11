from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "full_train_followup_contract", ROOT / "tools/full_train_followup_contract.py",
)
contract = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(contract)


def recipe(task: str = "wind", mode: str = "P") -> dict:
    return {
        "task": task,
        "mode": mode,
        "run_id": contract.RUN_IDS[task][mode],
        "execution_protocol": contract.PROTOCOL,
        "full_train_followup1000": True,
        "formal_full": False,
        "dataset_protocol": contract.DATA[task]["dataset_protocol"],
        "flow_readout_law": contract.THERMAL_FLOW_READOUT_LAW if task == "thermal" else None,
        "initialization": "fresh_all_trainable",
        "parent_checkpoint": None,
        "total_epochs": contract.SCHEDULE_HORIZON,
        "approved_stop_after": contract.APPROVED_STOP,
        "optimizer_schedule": dict(contract.FULL_SCHEDULE),
        "checkpoint_epochs": list(range(100, 1001, 100)),
        "launch_policy": "root_protocol_go",
        "manual_full_followup": True,
        "root_protocol_go": True,
        "locality_prior_strength": 0.0 if mode == "P" else 1.0,
        "model_contract": {"same": "mechanism"},
        "objective_contract": {"same": "objective"},
        "optimizer_name": "AdamW",
        "optimizer_betas": [0.9, 0.999],
        "optimizer_eps": 1.0e-8,
        "gradient_clip_norm": 1.0,
        "weight_decay": 1.0e-5,
        "auxiliary_calibration_fallback": False,
        "calibration_receipt": ({
            "dataset_protocol": contract.DATA["thermal"]["dataset_protocol"],
            "required_payload_sha256": "a" * 64,
        } if task == "thermal" else None),
        "population_binding": {"training_count": contract.DATA[task]["train_count"]},
    }


def checkpoint(candidate: dict, epoch: int) -> dict:
    return {"epoch": epoch, "experiment_identity": {"recipe": copy.deepcopy(candidate)}}


def test_full_followup_execution_is_exactly_start100_resume500_resume1000() -> None:
    candidate = recipe()
    contract.validate_recipe(candidate, command="start", stop_after=100)
    contract.validate_recipe(candidate, command="resume", stop_after=500,
                             resume_checkpoint_payload=checkpoint(candidate, 100))
    contract.validate_recipe(candidate, command="resume", stop_after=1000,
                             resume_checkpoint_payload=checkpoint(candidate, 500))


@pytest.mark.parametrize("stop_after", (500, 1000, 1001, 5000, None))
def test_fresh_full_followup_start_stops_at_first_review(stop_after: int | None) -> None:
    with pytest.raises(ValueError, match="epoch 100"):
        contract.validate_recipe(recipe(), command="start", stop_after=stop_after)


@pytest.mark.parametrize(("source_epoch", "stop_after"), ((100, 1000), (500, 1001), (500, 5000)))
def test_resume_rejects_skipped_reviews_and_extension(source_epoch: int, stop_after: int) -> None:
    with pytest.raises(ValueError, match="100-to-500 or 500-to-1000"):
        contract.validate_recipe(recipe(), command="resume", stop_after=stop_after,
                                 resume_checkpoint_payload=checkpoint(recipe(), source_epoch))


def test_resume_rejects_quarter_checkpoint_and_branched_lineage() -> None:
    candidate = recipe()
    quarter = checkpoint(candidate, 100)
    source_recipe = quarter["experiment_identity"]["recipe"]
    source_recipe["dataset_protocol"] = "wind_shared_fixed24_v1"
    with pytest.raises(ValueError, match="another run, population"):
        contract.validate_recipe(candidate, command="resume", stop_after=500,
                                 resume_checkpoint_payload=quarter)
    quarter = checkpoint(candidate, 100)
    quarter["experiment_identity"]["recipe"]["parent_checkpoint"] = "quarter.pt"
    with pytest.raises(ValueError, match="fresh, root-approved"):
        contract.validate_recipe(candidate, command="resume", stop_after=500,
                                 resume_checkpoint_payload=quarter)
    with pytest.raises(ValueError, match="branches are forbidden"):
        contract.validate_recipe(candidate, command="branch")


def test_resume_rejects_epoch1000_as_a_continuation_source() -> None:
    with pytest.raises(ValueError, match="latest epoch 100 or 500"):
        contract.validate_recipe(recipe(), command="resume", stop_after=1000,
                                 resume_checkpoint_payload=checkpoint(recipe(), 1000))

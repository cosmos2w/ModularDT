"""Identity guards for the read-only classic-to-response comparison bridge."""

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from channelthermal.data.datasets import H5Normalizer
from thermal_source_response_evaluate import (
    CLASSIC_IDENTITIES,
    classic_input_sample_for_heat,
    validate_classic_selection,
)


@pytest.mark.parametrize("classic_id", ("Run1804", "Run1502", "Run1804_e5000_latest", "Run1502_e5000_latest"))
def test_classic_bridge_accepts_only_the_retained_checkpoint_identity(classic_id):
    identity = CLASSIC_IDENTITIES[classic_id]
    checkpoint = {"epoch": identity["epoch"]}

    actual = validate_classic_selection(classic_id, checkpoint, identity["sha256"])

    assert actual == {
        "classic_id": classic_id,
        "epoch": identity["epoch"],
        "sha256": identity["sha256"],
    }


@pytest.mark.parametrize(
    "classic_id,checkpoint,digest,reason",
    (
        ("Run1804", {"epoch": 5000}, CLASSIC_IDENTITIES["Run1804"]["sha256"], "must be e4738"),
        ("Run1502", {"epoch": 4794}, "0" * 64, "SHA-256"),
        ("Run1804", {}, CLASSIC_IDENTITIES["Run1804"]["sha256"], "got None"),
    ),
)
def test_classic_bridge_rejects_wrong_age_or_weights(classic_id, checkpoint, digest, reason):
    with pytest.raises(ValueError, match=reason):
        validate_classic_selection(classic_id, checkpoint, digest)


def test_classic_bridge_rejects_unlisted_identity():
    with pytest.raises(ValueError, match="Unknown historical comparison identity"):
        validate_classic_selection("Run9999", {"epoch": 1}, "0" * 64)


def test_classic_counted_input_uses_checkpoint_transform_and_physical_local_heat():
    raw = {"structure": {"module_present": np.asarray([1, 0, 1], dtype=np.float32),
                          "heat_powers": np.asarray([0.2, 0.0, 0.4], dtype=np.float32)}}
    normalized = {"structure": {"heat_powers": np.asarray([0.0, 0.0, 0.0], dtype=np.float32)},
                  "local_module_params": np.ones((3, 7), dtype=np.float32)}
    normalizer = H5Normalizer({"heat_power_mean": np.asarray([1.0], dtype=np.float32),
                               "heat_power_std": np.asarray([2.0], dtype=np.float32)})

    result = classic_input_sample_for_heat(normalized, raw, [3.0, 5.0], normalizer, normalize_inputs=True)

    np.testing.assert_allclose(result["structure"]["heat_powers"], [1.0, -0.5, 2.0])
    np.testing.assert_array_equal(result["local_module_params"][:, 0], [3.0, 0.0, 5.0])
    np.testing.assert_array_equal(result["local_module_params"][1], 0)
    np.testing.assert_array_equal(normalized["local_module_params"], 1)


def test_classic_counted_input_rejects_wrong_source_count():
    sample = {"structure": {"heat_powers": np.zeros(2, dtype=np.float32)},
              "local_module_params": np.zeros((2, 7), dtype=np.float32)}
    with pytest.raises(ValueError, match="one finite value per active physical source"):
        classic_input_sample_for_heat(sample, {"structure": {"module_present": np.ones(2),
            "heat_powers": np.ones(2)}}, [1.0], H5Normalizer(), normalize_inputs=False)

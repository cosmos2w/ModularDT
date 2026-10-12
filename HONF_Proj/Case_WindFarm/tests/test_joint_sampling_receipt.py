"""The joint provider records actual queries without changing their selection."""

from types import SimpleNamespace

import numpy as np
from honf_runtime.unified_training import SamplingKey

from windfarm.training.joint_task import WindJointRegionalTask
from windfarm.training.unified_task import _query_sampling_sha256
from windfarm.workflows.joint_forward import ROLE_NAMES


def test_joint_query_receipt_binds_actual_cells_without_consuming_rng():
    task = object.__new__(WindJointRegionalTask)
    task.max_microbatch_cases = 1
    task._train_row_set = {6}
    task.primary_queries = 5
    task.training_samples = 0
    task.validation_samples = 0
    scene = SimpleNamespace(
        case_name="case", layout_index=2, wind_direction_deg=270.0,
        module_present=np.ones(2, dtype=np.float32),
    )
    sample = SimpleNamespace(
        coordinates_D=np.arange(15, dtype=np.float32).reshape(5, 3),
        target_mps=np.ones((5, 3), dtype=np.float32),
        flat_indices=np.arange(5, dtype=np.int64),
        role_sample_counts={role: 1 for role in ROLE_NAMES},
    )
    calls = []

    def sampled(row, key, *, validation):
        calls.append((row, key, validation))
        return scene, sample

    task._sample_row = sampled
    key = SamplingKey(42, 1101, 0, 0, "joint", "P-R",
                      sampling_version=SamplingKey.CASE_EPOCH_VERSION,
                      dataset_id="fixed")
    batch = task._make_batch((6,), key, training=True)
    assert calls == [(6, key, False)]
    assert task.training_samples == 1
    assert batch.auxiliary["query_sampling_sha256"] == _query_sampling_sha256((6,), (sample,), key)
    np.testing.assert_array_equal(batch.receivers.coordinates_D[0], sample.coordinates_D)
    np.testing.assert_array_equal(batch.targets.velocity_mps[0], sample.target_mps)
    digest = batch.auxiliary["query_sampling_sha256"]
    control_key = SamplingKey(42, 1101, 0, 0, "joint", "P-C",
                              sampling_version=SamplingKey.CASE_EPOCH_VERSION,
                              dataset_id="fixed")
    assert task._make_batch((6,), control_key, training=True).auxiliary["query_sampling_sha256"] == digest
    sample.target_mps[:] = 7.0
    assert task._make_batch((6,), key, training=True).auxiliary["query_sampling_sha256"] == digest
    sample.flat_indices[0] = 99
    assert task._make_batch((6,), key, training=True).auxiliary["query_sampling_sha256"] != digest

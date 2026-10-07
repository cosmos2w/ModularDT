"""Focused numerical checks for the post-fit receiver-local explanatory view."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from thermal_source_response_local_view import (
    bounded_radii,
    geometry_orders,
    learned_cover,
    make_patch_catalog,
    make_response_row,
    response_scores,
    retained_kernel_precision_audit,
    verify_counted_identity,
)


def test_fixed_patch_catalog_covers_grid_and_normalizes_each_fluid_patch():
    x = (np.arange(8) + 0.5) / 8
    y = (np.arange(8) + 0.5) / 8
    yy, xx = np.meshgrid(y, x, indexing="ij")
    xy = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    fluid = np.ones(xy.shape[0], dtype=bool)
    fluid[0] = False

    catalog = make_patch_catalog(xy, fluid, domain_x=1, domain_y=1, nx=4, ny=4)

    assert catalog["labels"].shape == (64,)
    assert np.all(catalog["labels"] >= 0) and np.all(catalog["labels"] < 16)
    for patch in range(16):
        weights = catalog["weights"][catalog["labels"] == patch]
        if weights.sum() > 0:
            assert weights.sum() == pytest.approx(1.0)
    assert catalog["weights"][0] == 0


def test_learned_cover_is_minimal_and_keeps_physical_source_ids():
    scores = np.asarray([0.8, 0.4, 0.2])
    cover = learned_cover(scores, 0.2, ("source-c", "source-a", "source-b"))

    assert cover["selected_source_ids"] == ["source-c", "source-a"]
    assert cover["retained_count"] == 2
    assert cover["omitted_triangle_bound"] == pytest.approx(0.2)
    assert cover["minimal_for_budget"]


def test_input_geometry_controls_order_upstream_to_flow_direction():
    centers = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    ids = ("left", "middle", "right")

    positive = geometry_orders(centers, [1.0, 0.0], ids, 1.0)
    negative = geometry_orders(centers, [1.0, 0.0], ids, -1.0)

    assert positive["nearest"] == [1, 0, 2]
    assert positive["upstream"][:2] == [1, 0]
    assert negative["upstream"][:2] == [1, 2]


def test_saved_delta_omission_is_bounded_and_physical_error_stays_separate():
    kernel = np.asarray([[1.0, 0.5], [0.25, 1.5], [0.0, 0.5]])
    radii = np.asarray([0.2, 0.2])
    weights = np.asarray([0.25, 0.5, 0.25])
    scores = response_scores(kernel, weights, radii)
    delta_heat = np.asarray([0.1, -0.1])
    full = kernel @ delta_heat
    reference = full + np.asarray([0.01, -0.02, 0.01])

    row = make_response_row(kernel, delta_heat, reference, [0], np.arange(3), weights, scores)

    assert row["bound_holds_for_saved_delta"]
    assert row["measured_model_omission_rms"] <= row["omitted_triangle_bound"] + 1e-12
    assert row["full_model_physical_response_rmse"] > 0
    assert row["covered_model_physical_response_rmse"] != row["measured_model_omission_rms"]


def test_forcing_radii_are_training_box_bounded_and_ignore_padding():
    heat = np.asarray([0.8, 1.2, 0.0])
    present = np.asarray([True, True, False])

    radii = bounded_radii(heat, present, train_minimum=0.5, train_maximum=2.0)

    assert radii[0] == pytest.approx(0.2)
    assert radii[1] == pytest.approx(0.2)
    assert radii[2] == 0


def test_fp64_kernel_audit_reports_roundoff_against_retained_fp32_operator():
    precise = np.asarray([[1.0, 1e-5], [0.25, -0.5]], dtype=np.float64)
    retained = precise.astype(np.float32).astype(np.float64)

    audit = retained_kernel_precision_audit(precise, retained)

    assert audit["max_abs"] < 1e-6
    assert audit["rms"] < 1e-6
    with pytest.raises(ValueError, match="differs materially"):
        retained_kernel_precision_audit(precise, precise + 2e-5)


def test_counted_identity_is_frozen_and_checkpoint_exact(tmp_path):
    checkpoint = tmp_path / "epoch_2500.pt"
    checkpoint.write_bytes(b"selected-checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload = {
        "checkpoint": str(checkpoint), "checkpoint_epoch": 2500, "checkpoint_sha256": digest,
        "frozen_state_unchanged": True, "solver_attempts": 0, "optimizer_updates": 0,
        "scope": "retained physical response", "cohort": "counted",
        "families": [{"case_id": case_id} for case_id in ("0277", "0291", "0294", "0687")],
    }
    (tmp_path / "summary.json").write_text(json.dumps(payload))

    identity = verify_counted_identity(tmp_path, checkpoint, digest, 2500)

    assert identity["family_ids"] == ["0277", "0291", "0294", "0687"]
    assert identity["solver_attempts"] == 0 and identity["optimizer_updates"] == 0
    with pytest.raises(ValueError, match="checkpoint SHA"):
        verify_counted_identity(tmp_path, checkpoint, "bad", 2500)

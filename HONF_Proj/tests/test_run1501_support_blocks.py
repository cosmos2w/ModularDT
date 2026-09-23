"""Exact observed-support union executor contracts for Run 1501."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import torch

from honf_forward_core.config import UnifiedForwardConfig
from honf_forward_core.interface_fields import InterfaceFieldCore
from honf_forward_core.interface_fields.support_block_reader import (
    observed_support_blocks,
    pack_positive_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = PROJECT_ROOT / "src/config_core/forward/sparse_incidence_group_control_honf_context.json"


def _small_payload() -> dict[str, object]:
    import json

    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))["model"]["core_honf"]
    payload = dict(payload)
    payload.update(
        {
            "field_dim": 5,
            "hidden_dim": 16,
            "domain_length_x": 12.0,
            "domain_length_y": 6.0,
            "module_radius": 0.45,
            "query_fourier_frequencies": 2,
            "position_fourier_frequencies": 2,
        }
    )
    payload["interface_model"] = dict(payload["interface_model"])
    payload["interface_model"].update(
        {
            "message_hidden_dim": 8,
            "attention_heads": 2,
            "relative_fourier_frequencies": 2,
            "receiver_chunk_size": 3,
            "activation_checkpointing": False,
        }
    )
    return payload


def _batch(seed: int = 1501, queries: int = 7):
    generator = torch.Generator().manual_seed(seed)
    extent = torch.tensor([12.0, 6.0])
    from honf_forward_core.config import BatchData

    return BatchData(
        module_centers=torch.rand(2, 5, 2, generator=generator) * extent,
        module_present=torch.tensor(
            [[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]]
        ),
        module_features=torch.randn(2, 5, 3, generator=generator),
        global_context=torch.randn(2, 4, generator=generator),
        query_xy=torch.rand(2, queries, 2, generator=generator) * extent,
        query_time=None,
        target_field=torch.randn(2, queries, 5, generator=generator),
        case_name="run1501-support-block-test",
        metadata={},
        env_coords=torch.rand(2, 13, 2, generator=generator) * extent,
        env_features=torch.randn(2, 13, 2, generator=generator),
        env_weights=torch.rand(2, 13, generator=generator) + 0.2,
    )


def test_observed_support_blocks_form_exact_unique_unions_without_group_paths() -> None:
    query = torch.zeros(1, 5, 12, dtype=torch.float64)
    query[0, 0, [0, 2]] = torch.tensor([0.4, 0.6], dtype=torch.float64)
    query[0, 1, [0, 2]] = torch.tensor([0.2, 0.8], dtype=torch.float64)
    query[0, 2, 1] = 1.0
    # Query 3 intentionally has no support and must remain an empty block.
    query[0, 4, [0, 2]] = torch.tensor([0.7, 0.3], dtype=torch.float64)

    source = torch.zeros(1, 5, 12, dtype=torch.float64)
    source[0, 0, 0] = 1.0
    source[0, 1, 2] = 1.0
    source[0, 2, [1, 3]] = torch.tensor([0.5, 0.5], dtype=torch.float64)
    source[0, 3, [0, 1, 2]] = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    measure = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0]], dtype=torch.float64)

    blocks = list(observed_support_blocks(query, source, source_measure=measure))
    assert [block.query_signature for block in blocks] == [0, 2, 5]
    assert [block.query_index.tolist() for block in blocks] == [[3], [2], [0, 1, 4]]
    assert [block.source_index.tolist() for block in blocks] == [[], [2, 3], [0, 1, 3]]
    assert sum(block.executed_rows for block in blocks) == 11

    direct = (
        query.gt(0)[:, :, None, :]
        & source.gt(0)[:, None, :, :]
    ).any(dim=-1) & measure[:, None, :].gt(0)
    assert int(direct.sum()) == 11
    assert int(pack_positive_support(query).max()) == 5
    assert all(block.source_index.unique().numel() == block.source_index.numel() for block in blocks)


def _prepared_pair(*, seed: int = 1501012, queries: int = 9):
    torch.manual_seed(seed)
    config = UnifiedForwardConfig.from_dict(_small_payload())
    batch = _batch(seed=seed, queries=queries)
    reference = InterfaceFieldCore(config).eval()
    encoded = reference.encode_case(batch)
    # Exercise mixed-B scale broadcasting in the block geometry path.
    encoded = replace(
        encoded,
        coordinate_scale=torch.tensor([[12.0, 6.0], [10.0, 5.0]]),
    )
    prepared = reference.prepare(encoded, encoded.module_tokens)
    # Materialize the lazy receiver projection before cloning the reference.
    reference.read(prepared, batch.query_xy[:, :1], receiver_chunk_size=1)
    blocked = copy.deepcopy(reference)
    blocked_encoded = blocked.encode_case(batch)
    blocked_encoded = replace(
        blocked_encoded,
        coordinate_scale=torch.tensor([[12.0, 6.0], [10.0, 5.0]]),
    )
    blocked_prepared = blocked.prepare(blocked_encoded, blocked_encoded.module_tokens)
    blocked.backend.executor_policy = "support_blocks"
    return batch, reference, prepared, blocked, blocked_prepared


def test_support_blocks_match_rectangular_forward_on_mixed_batch() -> None:
    batch, reference, prepared, blocked, blocked_prepared = _prepared_pair()
    with torch.no_grad():
        expected = reference.read(
            prepared,
            batch.query_xy,
            receiver_chunk_size=4,
            return_routing_maps=True,
        )
        actual = blocked.read(
            blocked_prepared,
            batch.query_xy,
            receiver_chunk_size=4,
            return_routing_maps=True,
        )
    torch.testing.assert_close(actual.context, expected.context, rtol=5.0e-5, atol=5.0e-6)
    for prefix in ("module", "environment"):
        assert actual.interaction_aux[f"group_control_{prefix}_unique_pairs"].item() == expected.interaction_aux[
            f"group_control_{prefix}_unique_pairs"
        ].item()
        assert actual.interaction_aux[f"group_control_{prefix}_fine_rows"].item() == actual.interaction_aux[
            f"group_control_{prefix}_unique_pairs"
        ].item()
        assert actual.interaction_aux[f"group_control_{prefix}_executor_selected"].eq(1)
    assert actual.interaction_aux["group_control_module_fine_rows"].item() < expected.interaction_aux[
        "group_control_module_fine_rows"
    ].item()


def test_support_blocks_preserve_connected_first_gradients() -> None:
    batch, reference, prepared, blocked, blocked_prepared = _prepared_pair(
        seed=1501013, queries=7
    )
    query_reference = batch.query_xy.clone().requires_grad_(True)
    query_blocked = batch.query_xy.clone().requires_grad_(True)
    output_reference = reference.read(prepared, query_reference, receiver_chunk_size=7).context
    output_blocked = blocked.read(blocked_prepared, query_blocked, receiver_chunk_size=7).context
    loss_reference = output_reference.square().mean()
    loss_blocked = output_blocked.square().mean()
    parameters_reference = (
        query_reference,
        reference.backend.router.group_codes,
        reference.backend.module_control_gain.weight,
        reference.backend.environment_value_control.weight,
        reference.backend.environment_score_control.weight,
    )
    parameters_blocked = (
        query_blocked,
        blocked.backend.router.group_codes,
        blocked.backend.module_control_gain.weight,
        blocked.backend.environment_value_control.weight,
        blocked.backend.environment_score_control.weight,
    )
    gradients_reference = torch.autograd.grad(loss_reference, parameters_reference)
    gradients_blocked = torch.autograd.grad(loss_blocked, parameters_blocked)
    torch.testing.assert_close(output_blocked, output_reference, rtol=5.0e-5, atol=5.0e-6)
    for expected, actual in zip(gradients_reference, gradients_blocked, strict=True):
        torch.testing.assert_close(actual, expected, rtol=8.0e-5, atol=8.0e-6)
        assert torch.isfinite(actual).all()


def test_support_blocks_keep_empty_environment_type_exactly_zero() -> None:
    batch, reference, prepared, blocked, blocked_prepared = _prepared_pair(
        seed=1501014, queries=6
    )
    controls = blocked_prepared.backend_state["group_control_state"]
    empty_controls = replace(
        controls,
        environment_membership=torch.zeros_like(controls.environment_membership),
        environment_mass=torch.zeros_like(controls.environment_mass),
    )
    empty_state = dict(blocked_prepared.backend_state)
    empty_state["group_control_state"] = empty_controls
    empty_prepared = replace(blocked_prepared, backend_state=empty_state)
    receiver_features = blocked._receiver_features(empty_prepared, batch.query_xy)
    route = blocked.backend._route(
        empty_state,
        empty_prepared.encoded,
        batch.query_xy,
        receiver_features,
    )
    environment, environment_aux = blocked.backend._read_environment(
        empty_state,
        empty_prepared.encoded,
        batch.query_xy,
        receiver_features,
        route,
        include_diagnostics=True,
    )
    assert torch.equal(environment, torch.zeros_like(environment))
    assert environment_aux["group_control_environment_fine_rows"].item() == 0.0
    assert environment_aux["group_control_environment_executor_selected"].item() == 1.0

    # The historical rectangular reference has the same empty-type boundary.
    reference_state = dict(prepared.backend_state)
    reference_state["group_control_state"] = replace(
        reference_state["group_control_state"],
        environment_membership=torch.zeros_like(controls.environment_membership),
        environment_mass=torch.zeros_like(controls.environment_mass),
    )
    reference_features = reference._receiver_features(
        replace(prepared, backend_state=reference_state), batch.query_xy
    )
    reference_route = reference.backend._route(
        reference_state,
        prepared.encoded,
        batch.query_xy,
        reference_features,
    )
    expected, _ = reference.backend._read_environment(
        reference_state,
        prepared.encoded,
        batch.query_xy,
        reference_features,
        reference_route,
        include_diagnostics=True,
    )
    torch.testing.assert_close(environment, expected, rtol=0.0, atol=0.0)

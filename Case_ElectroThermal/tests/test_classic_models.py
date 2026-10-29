"""Focused CPU contracts for the standard HET-MF model facades."""

import copy
import pickle

import numpy as np
import pytest
import torch
from torch.nn.parameter import UninitializedParameter

from hetmf.classic_models import (
    DENSE_PAIRWISE,
    SIX_HYPEREDGE,
    HETMFClassicAdapter,
    build_classic_model,
)
from hetmf.processing import FIELD_ORDER


def _sample(contact_count: int, *, query_count: int = 7, slots: int = 6, state_id: str | None = None):
    t = (np.arange(12, dtype=np.float64) + 0.5) / 12
    x, y = np.meshgrid(t, t)
    environment = np.column_stack((
        x.ravel(), y.ravel(), np.full(144, 1 / 12), np.full(144, 1 / 12),
        np.full(144, 1 / 144), np.linspace(1.0, 2.0, 144), np.linspace(0.5, 1.5, 144),
        np.linspace(0.2, 1.2, 144),
    ))
    contacts = np.full((slots, 5), np.nan, dtype=np.float64)
    contacts[:contact_count] = np.column_stack((
        np.linspace(0.15, 0.85, contact_count), np.full(contact_count, 0.5),
        np.linspace(0.02, 0.04, contact_count), np.linspace(0.4, 1.8, contact_count),
        np.linspace(-0.6, 0.7, contact_count),
    ))
    contact_mask = np.arange(slots) < contact_count
    query_xy = np.column_stack((
        np.linspace(0.03, 0.97, query_count), np.full(query_count, 0.43),
    ))
    fields = np.arange(query_count * len(FIELD_ORDER), dtype=np.float32).reshape(query_count, -1) / 19
    return {
        "inputs": {
            "environment_records": environment,
            "contacts": contacts,
            "contact_mask": contact_mask,
        },
        "query_xy": query_xy,
        "query_ids": np.arange(query_count, dtype=np.int64),
        "fields": fields,
        "query_weight": np.full(query_count, 1 / query_count, dtype=np.float64),
        "metadata": {
            "state_id": state_id or f"fixture-m{contact_count}",
            "split": "train",
            "seed": 17,
            "native_cell_ids": np.arange(query_count, dtype=np.int64),
            "contact_ids": tuple([f"contact-{i}" for i in range(contact_count)] + [""] * (slots - contact_count)),
        },
    }


def _state_snapshot(model):
    return {
        key: ("uninitialized", type(value))
        if isinstance(value, UninitializedParameter)
        else value.detach().clone()
        for key, value in model.state_dict().items()
    }


def _assert_snapshot_matches(model, snapshot):
    current = model.state_dict()
    assert current.keys() == snapshot.keys()
    for key, before in snapshot.items():
        after = current[key]
        if isinstance(before, tuple) and before[0] == "uninitialized":
            assert isinstance(after, before[1])
        else:
            assert not isinstance(after, UninitializedParameter)
            torch.testing.assert_close(after, before, rtol=0, atol=0)


def _small_model(name: str):
    shared = {
        "hidden_dim": 16,
        "seed": 0,
        "position_fourier_frequencies": 2,
        "query_fourier_frequencies": 2,
    }
    if name == SIX_HYPEREDGE:
        return build_classic_model(
            name,
            **shared,
            num_hyperedges=6,
            pairwise_kernel_hidden_dim=16,
            pairwise_kernel_num_layers=2,
            pairwise_kernel_fourier_frequencies=2,
        )
    return build_classic_model(
        name,
        **shared,
        message_hidden_dim=16,
        attention_heads=2,
        receiver_chunk_size=4,
        relative_fourier_frequencies=2,
        coarse_latent_count=2,
        coarse_blocks=1,
        main_latent_count=3,
        main_latent_blocks=1,
        activation_checkpointing=False,
    )


@pytest.fixture
def cpu_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("name", [SIX_HYPEREDGE, DENSE_PAIRWISE])
def test_generic_core_forward_backward_and_input_contract(name, cpu_torch_threads):
    from honf_forward_core import HONFNeuralField
    from honf_forward_core.interface_fields import InterfaceFieldCore

    adapter = HETMFClassicAdapter()
    ready = adapter.collate([_sample(2), _sample(5)])
    model = _small_model(name)
    prediction = model(ready)
    initial = _state_snapshot(model)
    assert prediction.shape == (2, 7, 5)
    assert torch.isfinite(prediction).all()
    assert ready.scene.target_field is None
    assert ready.scene.module_present.sum(dim=1).tolist() == [2.0, 5.0]
    assert ready.scene.module_features.shape == (2, 5, 3)
    assert ready.scene.env_features.shape == (2, 144, 3)
    assert ready.scene.env_weights.shape == (2, 144)
    assert ready.query_ids.shape == (2, 7)
    assert ready.contact_ids[0] == ("contact-0", "contact-1", None, None, None)
    assert ready.environment_ids[0] == tuple(range(144))
    torch.testing.assert_close(ready.module_measures[0, :2], torch.tensor(np.pi * np.array([0.02, 0.04]) ** 2, dtype=torch.float32))
    torch.testing.assert_close(ready.environment_measures, ready.scene.env_weights)
    torch.testing.assert_close(
        ready.scene.module_features[0, :2, 1],
        torch.log(torch.tensor([0.4, 1.8], dtype=torch.float32)),
    )

    if name == SIX_HYPEREDGE:
        assert isinstance(model.core, HONFNeuralField)
        assert model.core.config.num_hyperedges == 6
        assert model.core.config.decoder_mode == "enhanced_honf_pairwise_no_global_near"
        assert model.core.config.decoder_uses("hyper")
        assert model.core.config.decoder_uses("pairwise")
        assert not model.core.config.decoder_uses("global")
        assert not model.core.config.decoder_uses("near")
        assert not model.core.config.decoder_uses("direct")
        assert model.core.config.use_hyper_value_context is False
        assert model.core.config.pairwise_module_token_source == "organizer_contextualized"
    else:
        assert isinstance(model.core, InterfaceFieldCore)
        assert model.core.config.forward_architecture == "dense_pairwise_field"

    loss = ((prediction - ready.targets).square().mean(dim=-1) * ready.query_weights).sum(dim=1).mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)
    _assert_snapshot_matches(model, initial)
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


@pytest.mark.parametrize("name", [SIX_HYPEREDGE, DENSE_PAIRWISE])
def test_contact_permutation_padding_and_target_separation(name, cpu_torch_threads):
    adapter = HETMFClassicAdapter()
    model = _small_model(name).eval()
    sample = _sample(2, state_id="same-input")
    permuted = copy.deepcopy(sample)
    order = np.array([4, 1, 5, 0, 3, 2])
    for field in ("contacts", "contact_mask"):
        permuted["inputs"][field] = permuted["inputs"][field][order]
    permuted["metadata"]["contact_ids"] = tuple(permuted["metadata"]["contact_ids"][i] for i in order)

    changed_targets = copy.deepcopy(sample)
    changed_targets["fields"] = changed_targets["fields"] + 100
    changed_targets["metadata"].update(split="test", seed=-900, solver_residual=123.0)
    changed_targets["inputs"]["secret_target"] = np.full((7, 5), 1e6)

    with torch.no_grad():
        alone = model(adapter.collate([sample]))
        padded = model(adapter.collate([sample, _sample(5, state_id="larger")]))[:1]
        reordered = model(adapter.collate([permuted]))
        label_changed = adapter.collate([changed_targets])
        changed_prediction = model(label_changed)

    torch.testing.assert_close(alone, padded, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(alone, reordered, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(alone, changed_prediction, atol=0, rtol=0)
    assert not torch.equal(adapter.collate([sample]).targets, label_changed.targets)
    assert label_changed.scene.target_field is None
    assert label_changed.contact_ids == adapter.collate([sample]).contact_ids


def test_factory_uses_canonical_names_and_seeded_cpu_initialization(cpu_torch_threads):
    first = _small_model(SIX_HYPEREDGE)
    second = _small_model(SIX_HYPEREDGE)
    assert first.display_name == "Six-hyperedge organization controlling contextualized member reads"
    second_state = second.state_dict()
    for key, value in first.state_dict().items():
        other = second_state[key]
        if isinstance(value, UninitializedParameter):
            assert isinstance(other, UninitializedParameter)
        else:
            torch.testing.assert_close(value, other, rtol=0, atol=0)
    with pytest.raises(ValueError, match="six hyperedge slots"):
        build_classic_model(SIX_HYPEREDGE, num_hyperedges=4)
    with pytest.raises(ValueError, match="canonical HET-MF model name"):
        build_classic_model(SIX_HYPEREDGE, display_name="classic HONF")


@pytest.mark.parametrize("name", [SIX_HYPEREDGE, DENSE_PAIRWISE])
def test_weights_only_state_dict_roundtrip_with_uninitialized_lazy_parameters(name, tmp_path, cpu_torch_threads):
    adapter = HETMFClassicAdapter()
    batch = adapter.collate([_sample(3)]).scene
    model = _small_model(name)
    state = model.state_dict()
    assert any(isinstance(value, UninitializedParameter) for value in state.values())

    checkpoint = tmp_path / f"{name}_initial_state.pt"
    torch.save(state, checkpoint)
    try:
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        assert "torch.nn.parameter.UninitializedParameter" in str(exc)
    with torch.serialization.safe_globals([UninitializedParameter]):
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert any(isinstance(value, UninitializedParameter) for value in loaded.values())
    restored = _small_model(name)
    restored.load_state_dict(loaded, strict=True)

    rng_state = torch.random.get_rng_state()
    with torch.no_grad():
        expected = model.predict(batch)
        torch.random.set_rng_state(rng_state)
        actual = restored.predict(batch)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert all(parameter.device.type == "cpu" for parameter in restored.parameters())

import numpy as np
import pytest
import torch

from honf_runtime.local_response_contrasts import (
    CommonObservations, build_local_contrast, build_uniform_contrast,
    direction_uncovered_fraction, neighborhood_field_losses,
    projected_factor_rank, seeded_deck_selection,
)


def kernel(contrast):
    return contrast.operator.T @ contrast.operator


def test_uniform_pair_truth_offsets_and_gradients():
    refs = torch.tensor([[[2.]], [[4.]]], dtype=torch.float64)
    obs = CommonObservations(refs, torch.ones((1, 1), dtype=torch.bool), torch.ones((1, 1)))
    pred = torch.tensor([[[[3.]], [[7.]]]], requires_grad=True)
    losses = neighborhood_field_losses(pred, obs, build_uniform_contrast(2), [1.], [1.])
    assert losses.absolute.item() == pytest.approx(5.)
    assert losses.response.item() == pytest.approx(4.)
    losses.response.backward()
    assert pred.grad.flatten().tolist() == pytest.approx([-4., 4.])
    perfect = neighborhood_field_losses(refs.unsqueeze(0), obs, build_uniform_contrast(2), [1.], [1.])
    assert perfect.absolute == perfect.response == 0
    shifted = neighborhood_field_losses(refs.unsqueeze(0) + 3, obs, build_uniform_contrast(2), [1.], [1.])
    assert shifted.absolute.item() == pytest.approx(9.)
    assert shifted.response.item() == pytest.approx(0., abs=1e-28)


def test_kernel_invariants_and_permutation():
    x = torch.tensor([[0., 0.], [2., 0.], [0., 1.], [3., 1.], [2., 2.]], dtype=torch.float64, requires_grad=True)
    contrast = build_local_contrast(x)
    k = kernel(contrast)
    eigen = torch.linalg.eigvalsh(k)
    assert not contrast.operator.requires_grad
    assert torch.allclose(k @ torch.ones(5, dtype=torch.float64), torch.zeros(5, dtype=torch.float64), atol=1e-14)
    assert torch.trace(k).item() == pytest.approx(2.)
    assert eigen[0] > -1e-14
    assert eigen[1] >= 1 / 4 - 1e-14
    assert eigen[-1] <= 2 + 1e-14
    perm = torch.tensor([3, 1, 4, 0, 2])
    assert torch.allclose(kernel(build_local_contrast(x[perm])), k[perm][:, perm], atol=1e-14)
    assert torch.allclose(kernel(build_local_contrast(x * 1e-12)), k, atol=1e-14)


def test_equal_spectrum_rank_one_and_two_scene_are_uniform():
    for x in (torch.tensor([[0.], [1.], [3.]], dtype=torch.float64),
              torch.eye(4, dtype=torch.float64), torch.tensor([[0., 0.], [3., 2.]])):
        assert torch.allclose(kernel(build_local_contrast(x)), kernel(build_uniform_contrast(len(x))), atol=1e-14)


def test_tied_cutoff_and_precision_floor():
    h = build_uniform_contrast(4).operator / (2 / 3) ** .5
    singular = torch.tensor([30., 1., 1.], dtype=torch.float64)
    chart = h.T @ torch.diag(singular * (3 ** .5))
    # A tie straddling the cutoff is discarded as a block.
    assert build_local_contrast(chart).rank == 1
    precise = np.array([[1., 0.], [1. + 1e-10, 0.], [1., 2e-10]], dtype=np.float64)
    assert build_local_contrast(precise).rank == 2
    assert build_local_contrast(precise.tolist()).rank == 2
    assert build_local_contrast(precise, 1e-9).rank == 0
    contrast = build_local_contrast(precise, absolute_floor=1e-9)
    assert contrast.numerical_floor == 1e-9
    assert contrast.rank == 0
    assert projected_factor_rank(precise, [], [0, 1], absolute_floor=1e-9) == 0
    assert direction_uncovered_fraction(precise, [1., 0.], absolute_floor=1e-9) == pytest.approx(1.)
    with pytest.raises(ValueError, match='absolute_floor'):
        build_local_contrast(precise, absolute_floor=float('nan'))


def test_crossed_interaction_is_not_lost_and_confounding_projection():
    x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], dtype=torch.float64)
    interaction = x[:, 0] * x[:, 1]
    assert (kernel(build_local_contrast(x)) @ interaction).norm() > 0
    assert (interaction @ kernel(build_local_contrast(x)) @ interaction).item() == pytest.approx(8 / 3)
    confounded = torch.tensor([[-2., -2.], [-1., -1.], [1., 1.], [2., 2.]], dtype=torch.float64)
    assert projected_factor_rank(confounded, [1], [0]) == 0
    assert direction_uncovered_fraction(confounded, [1., -1.]) == pytest.approx(1.)
    assert direction_uncovered_fraction(confounded, [1., 1.]) == pytest.approx(0., abs=1e-14)


def test_fixed_mass_duplicate_missing_channel_and_nonthermal_metadata():
    # Two identical receiver coordinates retain their separate sampled mass.
    refs = torch.tensor([[[1., float('nan')], [1., float('nan')]],
                         [[3., float('nan')], [3., float('nan')]]], dtype=torch.float64)
    obs = CommonObservations(refs, torch.tensor([[True, False], [True, False]]),
                             torch.tensor([[.25, 0.], [.5, 0.]], dtype=torch.float64),
                             {'coordinates': np.array([[1.0000000001, 2., 3.], [1.0000000001, 2., 3.]], dtype=np.float64),
                              'time': 2.5, 'frame': 'earth-fixed'})
    pred = refs.clone().unsqueeze(0)
    pred[..., 0] += torch.tensor([1., 3.], dtype=torch.float64).view(1, 2, 1)
    losses = neighborhood_field_losses(pred, obs, build_uniform_contrast(2), [1., 2.], [2., 7.])
    assert losses.channel_available.tolist() == [True, False]
    assert losses.absolute.item() == pytest.approx(7.5)
    assert losses.response.item() == pytest.approx(6.)
    assert obs.metadata['coordinates'][0, 0] != np.float64(np.float32(obs.metadata['coordinates'][0, 0]))


def test_seeded_decks_reproducible_and_keep_tail():
    ids = ('case-one', 'case-two', 'case-three', 'case-four')
    all_items = seeded_deck_selection(ids, 43, 0, 12)
    assert all_items == seeded_deck_selection(ids, 43, 0, 5) + seeded_deck_selection(ids, 43, 5, 7)
    assert all(set(all_items[i:i + 4]) == set(ids) for i in range(0, 12, 4))
    with pytest.raises(ValueError):
        seeded_deck_selection(('duplicate', 'duplicate'), 1, 0)

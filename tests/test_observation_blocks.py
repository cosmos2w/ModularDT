"""Physical observation extraction, fixed-population estimates and gradients."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from honf_runtime.observation_blocks import (
    FieldComponent,
    LinearExtraction,
    ObservationBlock,
    ObservationDraw,
    QueryDependency,
    SamplingBatch,
    extract_observations,
    observation_response_losses,
    prepare_observation_sampling,
    sample_observation_blocks,
    union_observation_dependencies,
)


def make_block(name='field', *, n=2, dimensions=2, rows=4, groups=None, mask=None,
               pair_mask=None, edge_mask=None, references=None, measures=None, weight=1.0):
    points = np.zeros((rows, dimensions), dtype=np.float64)
    points[:, 0] = np.arange(rows)
    points[:, 1] = np.arange(rows) % 2
    components = tuple(FieldComponent(f'component_{i}', f'unit_{i}', i, i + 1., 1. / n) for i in range(n))
    ids = tuple(f'receiver_{i}' for i in range(rows))
    absolute = np.ones((2, rows, n), dtype=bool) if mask is None else mask.copy()
    pair = absolute[0] & absolute[1] if pair_mask is None else pair_mask.copy()
    edge = pair.copy() if edge_mask is None else edge_mask.copy()
    reference = np.zeros((2, rows, n)) if references is None else references.copy()
    mass = np.ones((2, rows, n)) if measures is None else measures.copy()
    pair_mass = np.mean(mass, axis=0)
    extractions = []
    for scene in ('scene_0', 'scene_1'):
        dependencies = tuple(QueryDependency(scene, 'fixed', 0, receiver, tuple(point))
                             for receiver, point in zip(ids, points))
        obs = np.repeat(np.arange(rows), n)
        channel = np.tile(np.arange(n), rows)
        extractions.append(LinearExtraction(dependencies, obs, obs, channel, channel, np.ones(rows * n)))
    return ObservationBlock(name, components, ('scene_0', 'scene_1'), 'train', 'fixed', 0, ids,
                            points, reference, absolute, pair, edge, mass, pair_mass, tuple(extractions),
                            block_weight=weight, neighbor_groups=groups)


def evaluate(blocks, predictions, plan, records):
    return observation_response_losses(blocks, predictions, SamplingBatch(plan, tuple(records)))


def test_enumerated_sampling_expectation_matches_fixed_population_objective():
    mask = np.ones((2, 4, 2), dtype=bool)
    mask[0, 1, 1] = False
    mask[1, 3, 0] = False
    mass = np.arange(1, 17, dtype=float).reshape(2, 4, 2)
    first = make_block('varying', mask=mask, groups=('near', 'near', 'near', 'isolated'),
                       measures=mass, weight=2.)
    second = make_block('other', rows=2, groups=('solo_0', 'solo_1'), weight=3.)
    blocks = (first, second)
    predictions = (torch.arange(16, dtype=torch.float64).reshape(1, 2, 4, 2) / 3,
                   torch.arange(8, dtype=torch.float64).reshape(1, 2, 2, 2) / 2)
    plan = prepare_observation_sampling(blocks, max_neighbors=2)
    enumerated = np.zeros(3)
    for block_index, population in enumerate(plan.populations):
        for seed in population.eligible:
            choices = population.neighbors[int(seed)] or (None,)
            probability = plan.probabilities[block_index] / len(population.eligible) / len(choices)
            for neighbor in choices:
                result = evaluate(blocks, predictions, plan, [ObservationDraw(block_index, int(seed), neighbor)])
                enumerated += probability * np.asarray([float(result[key]) for key in
                    ('extra_absolute', 'point_response', 'spatial_response')])

    # Independently sum the full declared observation/conditional-neighbor
    # population, including each term's own component-specific valid mass.
    expected = np.zeros(3)
    for block, prediction, population in zip(blocks, predictions, plan.populations):
        values = prediction.numpy()[0] / np.asarray([c.scale for c in block.components])
        abs_mask, pair, edge = block.absolute_mask, block.pair_mask, block.edge_mask
        abs_num = np.zeros((2, 2)); abs_den = np.zeros((2, 2))
        point_num = np.zeros(2); point_den = np.zeros(2)
        spatial_num = np.zeros(2); spatial_den = np.zeros(2)
        for seed in population.eligible:
            choices = population.neighbors[int(seed)] or (None,)
            for neighbor in choices:
                endpoint_rows = (int(seed),) if neighbor is None else (int(seed), neighbor)
                for endpoint in endpoint_rows:
                    portion = 1. / len(choices) / len(endpoint_rows)
                    for scene in range(2):
                        factor = portion * block.absolute_measures[scene, seed] * abs_mask[scene, endpoint]
                        abs_num[scene] += factor * values[scene, endpoint] ** 2
                        abs_den[scene] += factor
                    factor = portion * block.pair_measures[seed] * pair[endpoint]
                    point_num += factor * (values[1, endpoint] - values[0, endpoint]) ** 2
                    point_den += factor
                if neighbor is not None:
                    factor = block.pair_measures[seed] * edge[seed] * edge[neighbor] / len(choices)
                    contrast = ((values[1, seed] - values[0, seed]) -
                                (values[1, neighbor] - values[0, neighbor]))
                    spatial_num += factor * contrast ** 2
                    spatial_den += factor
        component_weight = np.asarray([c.weight for c in block.components])
        for component in range(2):
            supported = abs_den[:, component] > 0
            if supported.any():
                expected[0] += (block.block_weight * component_weight[component] *
                                np.mean(abs_num[supported, component] / abs_den[supported, component]))
            if point_den[component] > 0:
                expected[1] += block.block_weight * component_weight[component] * point_num[component] / point_den[component]
            if spatial_den[component] > 0:
                expected[2] += block.block_weight * component_weight[component] * spatial_num[component] / spatial_den[component]
    np.testing.assert_allclose(enumerated, expected, rtol=1.e-12, atol=1.e-12)


@pytest.mark.parametrize(('n', 'dimensions', 'rows'), [(2, 2, 5), (7, 3, 8)])
def test_coupled_ragged_components_masks_offsets_sign_and_channel_order(n, dimensions, rows):
    mask = np.ones((2, rows, n), dtype=bool)
    mask[0, 0, 0] = False
    mask[1, -1, -1] = False
    refs = np.arange(2 * rows * n, dtype=float).reshape(2, rows, n) / 10
    refs[~mask] = np.nan
    block = make_block(n=n, dimensions=dimensions, rows=rows, mask=mask, references=refs)
    plan = prepare_observation_sampling((block,))
    draws = sample_observation_blocks(plan, 20, np.random.default_rng(14))
    reference = torch.as_tensor(np.nan_to_num(refs), dtype=torch.float64)[None]
    exact = observation_response_losses((block,), (reference.clone(),), draws)
    for key in ('extra_absolute', 'point_response', 'spatial_response'):
        assert float(exact[key]) == 0.
    common = reference + torch.arange(1, n + 1, dtype=torch.float64)[None, None, None, :] * 2
    offset = observation_response_losses((block,), (common,), draws)
    assert float(offset['extra_absolute']) > 0
    assert float(offset['point_response']) == pytest.approx(0., abs=1.e-28)
    assert float(offset['spatial_response']) == pytest.approx(0., abs=1.e-28)
    altered = common.clone()
    altered[:, 1, :, :] += torch.as_tensor(block.coordinates[:, :1] * np.arange(1, n + 1), dtype=torch.float64)
    actual = observation_response_losses((block,), (altered,), draws)
    assert float(actual['point_response']) > 0
    assert float(actual['spatial_response']) > 0
    reverse = 2 * reference - altered
    reversed_result = observation_response_losses((block,), (reverse,), draws)
    for key in ('extra_absolute', 'point_response', 'spatial_response'):
        torch.testing.assert_close(actual[key], reversed_result[key])
    # Reordering components and all corresponding declarations cannot change a loss.
    order = list(reversed(range(n)))
    reordered_extractions = []
    inverse = np.argsort(order)
    for extraction in block.extractions:
        reordered_extractions.append(replace(extraction, component_indices=inverse[extraction.component_indices]))
    reordered = replace(block, components=tuple(block.components[i] for i in order),
        references=block.references[:, :, order], absolute_mask=block.absolute_mask[:, :, order],
        pair_mask=block.pair_mask[:, order], edge_mask=block.edge_mask[:, order],
        absolute_measures=block.absolute_measures[:, :, order], pair_measures=block.pair_measures[:, order],
        extractions=tuple(reordered_extractions))
    reordered_plan = prepare_observation_sampling((reordered,))
    result = observation_response_losses((reordered,), (altered[..., order],), SamplingBatch(reordered_plan, draws.draws))
    for key in ('extra_absolute', 'point_response', 'spatial_response'):
        torch.testing.assert_close(actual[key], result[key])


def test_neighbors_are_stable_disconnected_and_direct_four_value_response():
    block = make_block(n=2, rows=4, groups=('one', 'one', 'two', 'two'))
    plan = prepare_observation_sampling((block,), max_neighbors=4)
    assert plan.populations[0].neighbors == ((1,), (0,), (3,), (2,))
    assert all((seed < 2) == (neighbor < 2)
               for seed, choices in enumerate(plan.populations[0].neighbors) for neighbor in choices)
    pred = torch.tensor([[[[1., 2.], [3., 4.], [5., 6.], [7., 8.]],
                          [[9., 8.], [8., 5.], [1., 3.], [4., 7.]]]], dtype=torch.float64)
    draw = ObservationDraw(0, 0, 1)
    result = evaluate((block,), (pred,), plan, [draw])
    # Four-field-value expression, including the sampled estimator's population factor.
    direct = ((pred[0, 1, 0] - pred[0, 0, 0]) - (pred[0, 1, 1] - pred[0, 0, 1]))
    direct = direct / torch.tensor([1., 2.], dtype=torch.float64)
    torch.testing.assert_close(result['spatial_response'], (direct.square() * .5).sum())
    # A separate support component has a separate constant spatial null mode.
    piecewise = torch.zeros_like(pred)
    piecewise[:, 1, :2] = 5.
    piecewise[:, 1, 2:] = -3.
    assert float(evaluate((block,), (piecewise,), plan, [draw])['spatial_response']) == 0.


def test_union_restores_duplicate_losses_gradients_and_linear_extraction():
    first = make_block('first', n=2, rows=3)
    second = make_block('second', n=2, rows=3)
    # Exercise a signed two-query extraction for one observed field.
    extractions = []
    for extraction in first.extractions:
        extractions.append(replace(extraction,
            dependency_indices=np.append(extraction.dependency_indices, 1),
            observation_indices=np.append(extraction.observation_indices, 0),
            component_indices=np.append(extraction.component_indices, 0),
            output_indices=np.append(extraction.output_indices, 0),
            coefficients=np.append(extraction.coefficients, -.25)))
    first = replace(first, extractions=tuple(extractions))
    blocks = (first, second)
    plan = prepare_observation_sampling(blocks)
    batch = SamplingBatch(plan, (ObservationDraw(0, 0, 1), ObservationDraw(0, 0, 1),
                                 ObservationDraw(1, 0, 1)))
    union = union_observation_dependencies(blocks, batch)
    assert len(union.dependencies) == 4  # two scenes and two positions
    assert union.requested_dependencies == 8
    model = torch.nn.Linear(2, 2, bias=True, dtype=torch.float64)

    def query(dependencies):
        positions = torch.tensor([item.coordinates for item in dependencies], dtype=torch.float64)
        offsets = torch.tensor([0. if item.scene_id == 'scene_0' else 1. for item in dependencies], dtype=torch.float64)
        return (model(positions) + offsets[:, None])[None]

    def evaluate_mode(merged):
        if merged:
            predictions = query(union.dependencies)
            gathered = [torch.stack([extract_observations(block, scene, predictions, union.mappings[index][scene])
                                      for scene in range(2)], dim=1)
                        for index, block in enumerate(blocks)]
        else:
            gathered = []
            for block in blocks:
                scenes = []
                for scene, extraction in enumerate(block.extractions):
                    predicted = query(extraction.dependencies)
                    scenes.append(extract_observations(block, scene, predicted, np.arange(len(extraction.dependencies))))
                gathered.append(torch.stack(scenes, dim=1))
        losses = observation_response_losses(blocks, gathered, batch)
        total = sum(losses[key] for key in ('extra_absolute', 'point_response', 'spatial_response'))
        gradients = torch.autograd.grad(total, tuple(model.parameters()))
        return gathered, losses, gradients

    merged = evaluate_mode(True)
    unmerged = evaluate_mode(False)
    repeat = evaluate_mode(False)
    for group in (unmerged, repeat):
        for actual, expected in zip(merged[0], group[0]):
            # Merged execution computes the selected positions only.
            torch.testing.assert_close(actual[:, :, :2], expected[:, :, :2], rtol=1.e-12, atol=1.e-12)
        for key in ('extra_absolute', 'point_response', 'spatial_response'):
            torch.testing.assert_close(merged[1][key], group[1][key], rtol=1.e-12, atol=1.e-12)
        for actual, expected in zip(merged[2], group[2]):
            torch.testing.assert_close(actual, expected, rtol=1.e-12, atol=1.e-12)


def test_dependency_identity_includes_exact_coordinates_and_missing_support_is_not_zero():
    block = make_block(rows=2)
    dep = block.extractions[0].dependencies[0]
    changed_dep = replace(dep, coordinates=(dep.coordinates[0] + .1, dep.coordinates[1]))
    extraction = replace(block.extractions[0], dependencies=(changed_dep,) + block.extractions[0].dependencies[1:])
    moved = replace(block, name='moved', extractions=(extraction, block.extractions[1]))
    union = union_observation_dependencies((block, moved))
    assert len(union.dependencies) == 5
    invalid = block.absolute_mask.copy()
    invalid[0, 0, 0] = False
    with pytest.raises(ValueError, match='Pair support'):
        replace(block, absolute_mask=invalid)


def test_reference_and_extraction_source_precision_survives_float32_predictions():
    block = make_block(rows=2)
    epsilon = 2. ** -40
    reference = np.full((2, 2, 2), 1. + epsilon, dtype=np.float64)
    extraction = block.extractions[0]
    coeff = np.ones_like(extraction.coefficients, dtype=np.float64)
    coeff[0] += epsilon
    first = replace(extraction, coefficients=coeff)
    block = replace(block, references=reference, extractions=(first, block.extractions[1]))
    source = torch.ones((1, 2, 2), dtype=torch.float32, requires_grad=True)
    extracted = extract_observations(block, 0, source, np.arange(2))
    assert extracted.dtype == torch.float64
    assert extracted[0, 0, 0].item() == 1. + epsilon
    predictions = torch.ones((1, 2, 2, 2), dtype=torch.float32, requires_grad=True)
    plan = prepare_observation_sampling((block,))
    result = observation_response_losses((block,), (predictions,),
        SamplingBatch(plan, (ObservationDraw(0, 0, 1),)))
    assert result['extra_absolute'].dtype == torch.float64
    assert float(result['extra_absolute']) > 0.
    result['extra_absolute'].backward()
    assert predictions.grad.dtype == torch.float32
    assert torch.count_nonzero(predictions.grad) > 0


def test_component_estimates_reconstruct_totals_and_expose_fixed_coverage():
    first = make_block('first', n=2, rows=3, weight=2.)
    second = make_block('second', n=2, rows=2, weight=4.)
    blocks = (first, second)
    plan = prepare_observation_sampling(blocks)
    batch = SamplingBatch(plan, (ObservationDraw(0, 0, 1), ObservationDraw(0, 0, 1),
                                 ObservationDraw(0, 2, 1)))
    predictions = (torch.arange(12, dtype=torch.float64).reshape(1, 2, 3, 2),
                   torch.arange(8, dtype=torch.float64).reshape(1, 2, 2, 2))
    result = observation_response_losses(blocks, predictions, batch, include_component_estimates=True)
    for prefix, loss_name in (('absolute', 'extra_absolute'), ('point', 'point_response'),
                              ('spatial', 'spatial_response')):
        rows = [entry for by_component in result['component_estimates'].values() for entry in by_component.values()]
        reconstructed = sum(row['block_weight'] * row['component_weight'] * row[f'{prefix}_mse_normalized']
                            for row in rows)
        assert reconstructed == pytest.approx(float(result[loss_name]), abs=1.e-12)
        for row in rows:
            assert row[f'{prefix}_mse_physical'] == pytest.approx(row[f'{prefix}_mse_normalized'] * row['scale'] ** 2)
    assert result['component_estimates']['first']['component_0']['absolute_draws'] == 3
    assert result['component_estimates']['second']['component_0']['absolute_draws'] == 0
    assert result['component_estimates']['second']['component_0']['point_eligible_mass'] > 0.


def test_union_omits_terms_for_missing_individual_scene_support():
    block = make_block(rows=3)
    mask = block.absolute_mask.copy()
    mask[1] = False
    unmatched = replace(block, absolute_mask=mask, pair_mask=np.zeros_like(block.pair_mask),
                        edge_mask=np.zeros_like(block.edge_mask), correspondence='none')
    plan = prepare_observation_sampling((unmatched,))
    draws = SamplingBatch(plan, (ObservationDraw(0, 0, 1),))
    union = union_observation_dependencies((unmatched,), draws)
    assert len(union.dependencies) == 2
    assert {dependency.scene_id for dependency in union.dependencies} == {'scene_0'}
    assert np.all(union.mappings[0][1] == -1)


def test_nearest_ties_use_ids_and_domain_tags_do_not_bridge():
    block = make_block(rows=5)
    points = np.array([[0., 0.], [1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    block = replace(block, coordinates=points, receiver_ids=('center', 'z', 'a', 'm', 'b'))
    plan = prepare_observation_sampling((block,), max_neighbors=2)
    assert tuple(block.receiver_ids[index] for index in plan.populations[0].neighbors[0]) == ('a', 'b')
    block = replace(block, domain_tags=('shared', 'other', 'shared', 'shared', 'other'))
    plan = prepare_observation_sampling((block,), max_neighbors=2)
    assert tuple(block.receiver_ids[index] for index in plan.populations[0].neighbors[0]) == ('a', 'm')

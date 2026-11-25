"""Native sample-law and fixed-measure response checks on small populations."""

from collections import defaultdict
from itertools import product
from math import comb, perm

import numpy as np
import pytest
import torch

from honf_runtime.coupled_observations import (
    NativeCoupledSample,
    NativeObservationPopulation,
    ObservationKey,
    coupled_response_loss,
    prepare_native_coupling,
    sample_native_coupling,
)
from honf_runtime.observation_blocks import FieldComponent


def key(position, *, time=0, frame='fixed', correspondence='shared_frame', extraction='native'):
    return ObservationKey(frame, time, (float(position),), correspondence, extraction)


def population(scene, keys, *, component_count=2, references=None, measures=None,
               valid=None, complete=None, extraction_ids=None):
    rows = len(keys)
    fields = tuple(FieldComponent(f'field_{index}', f'unit_{index}', index, index + 1., index + 1.)
                   for index in range(component_count))
    shape = (rows, component_count)
    references = np.zeros(shape) if references is None else np.asarray(references, dtype=np.float64)
    measures = np.ones(shape) if measures is None else np.asarray(measures, dtype=np.float64)
    valid = np.ones(shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    complete = np.ones(shape, dtype=bool) if complete is None else np.asarray(complete, dtype=bool)
    dependencies = tuple(tuple(((scene, row, component, 'query'),) for component in range(component_count))
                         for row in range(rows))
    return NativeObservationPopulation(keys, fields, references, measures, valid, complete,
                                       dependencies, extraction_ids)


def generators(seed):
    streams = np.random.SeedSequence(seed).spawn(5)
    return dict(zip(('anchor_rng', 'left_completion_rng', 'right_completion_rng',
                     'left_order_rng', 'right_order_rng'), (np.random.default_rng(stream) for stream in streams)))


def native_dependencies(population, panel):
    return {dependency for row in panel for component in population.dependencies[int(row)] for dependency in component}


class NeedChoice(Exception):
    def __init__(self, probabilities):
        self.probabilities = probabilities


class Branches:
    def __init__(self, script):
        self.script, self.cursor = script, 0

    def choose(self, probabilities):
        probabilities = np.asarray(probabilities, dtype=float)
        probabilities = probabilities / probabilities.sum()
        if self.cursor == len(self.script):
            raise NeedChoice(probabilities)
        value = self.script[self.cursor]
        self.cursor += 1
        return value


class EnumeratedGenerator(np.random.Generator):
    """Make the real sampler expose every discrete random choice for enumeration."""

    def __init__(self, branches, common_probability):
        super().__init__(np.random.PCG64(0))
        self.branches = branches
        self.common_probability = common_probability

    def random(self, size=None):
        count = int(size)
        c = self.common_probability
        return np.asarray([c / 2 if self.branches.choose([c, 1 - c]) == 0 else (1 + c) / 2
                           for _ in range(count)])

    def choice(self, values, size=None, replace=True, p=None):
        options = list(range(values)) if isinstance(values, (int, np.integer)) else list(values)
        probabilities = [1.] * len(options) if p is None else list(p)
        result = []
        for _ in range(int(size)):
            index = self.branches.choose(probabilities)
            result.append(options[index])
            if not replace:
                del options[index]
                del probabilities[index]
        return np.asarray(result, dtype=np.int64)

    def integers(self, high, *args, **kwargs):
        return self.branches.choose([1.] * int(high))

    def permutation(self, length):
        return self.choice(int(length), size=int(length), replace=False)


def enumerate_samples(plan, left_budget, right_budget, *, anchor_count):
    """Enumerate the implementation's actual random calls with their probabilities."""
    pending = [([], 1.)]
    results = []
    while pending:
        script, probability = pending.pop()
        branches = Branches(script)
        rngs = [EnumeratedGenerator(branches, plan.common_probability) for _ in range(5)]
        try:
            sample = sample_native_coupling(plan, left_budget, right_budget, anchor_count=anchor_count,
                **dict(zip(('anchor_rng', 'left_completion_rng', 'right_completion_rng',
                            'left_order_rng', 'right_order_rng'), rngs)))
        except NeedChoice as fork:
            pending.extend((script + [index], probability * float(weight))
                           for index, weight in enumerate(fork.probabilities) if weight > 0)
        else:
            results.append((sample, probability))
    return results


@pytest.mark.parametrize(('left_keys', 'right_keys', 'left_budget', 'right_budget'), [
    ([key(0), key(0), key(1)], [key(0), key(2), key(2)], 2, 2),
    ([key(0), key(1), key(2)], [key(0), key(1), key(2), key(3)], 2, 2),
    ([key(0), key(1), key(2)], [key(0), key(1), key(2)], 2, 2),
    ([key(0), key(1), key(2)], [key(3), key(4), key(5)], 2, 2),
])
def test_exact_ordered_and_subset_laws_with_repeated_anchors(left_keys, right_keys, left_budget, right_budget):
    plan = prepare_native_coupling(population('left', left_keys), population('right', right_keys))
    outcomes = enumerate_samples(plan, left_budget, right_budget, anchor_count=2)
    assert sum(probability for _, probability in outcomes) == pytest.approx(1.)
    for side, nrows, budget in (('left', len(left_keys), left_budget), ('right', len(right_keys), right_budget)):
        subsets, ordered = defaultdict(float), defaultdict(float)
        for sample, probability in outcomes:
            rows = getattr(sample, f'{side}_rows')
            anchors = getattr(sample, f'{side}_anchor_rows')
            positions = getattr(sample, f'{side}_anchor_positions')
            assert len(set(rows)) == budget
            np.testing.assert_array_equal(rows[positions], anchors)
            subsets[tuple(sorted(rows))] += probability
            ordered[tuple(rows)] += probability
        assert len(subsets) == comb(nrows, budget)
        assert len(ordered) == perm(nrows, budget)
        np.testing.assert_allclose(list(subsets.values()), 1. / comb(nrows, budget), atol=1.e-12)
        np.testing.assert_allclose(list(ordered.values()), 1. / perm(nrows, budget), atol=1.e-12)


def exact_response_target(left, right):
    """Independent literal finite-population definition, including repeated rows."""
    result = []
    for component in range(len(left.components)):
        bucket_masses = []
        terms = []
        groups = set((left.keys[row], left.extraction_ids[row][component])
                     for row in range(len(left.keys)) if left.keys[row] is not None)
        for observation, extraction in groups:
            if extraction is None:
                continue
            valid_rows = []
            for pop in (left, right):
                valid_rows.append([row for row in range(len(pop.keys))
                    if pop.keys[row] == observation and pop.extraction_ids[row][component] == extraction
                    and pop.valid[row, component] and pop.dependency_complete[row, component]
                    and pop.dependencies[row][component] and pop.measures[row, component] > 0])
            lrows, rrows = valid_rows
            lm = sum(left.measures[row, component] for row in lrows)
            rm = sum(right.measures[row, component] for row in rrows)
            if not lm or not rm:
                continue
            bucket_mass = min(lm, rm)
            bucket_masses.append(bucket_mass)
            for lrow, rrow in product(lrows, rrows):
                mu = bucket_mass * left.measures[lrow, component] / lm * right.measures[rrow, component] / rm
                # Test predictions are an explicit function of the row and scene.
                le = (prediction('left', lrow, component) - left.references[lrow, component])
                re = (prediction('right', rrow, component) - right.references[rrow, component])
                terms.append(mu * ((re - le) / left.components[component].scale) ** 2)
        result.append(None if not bucket_masses else sum(terms) / sum(bucket_masses))
    return result


def prediction(scene, row, component):
    return (row + 1) * (component + 2) * (1. if scene == 'left' else 1.7)


def panel_predictions(scene, population, rows, *, requires_grad=False):
    values = [[prediction(scene, int(row), component) for component in range(len(population.components))]
              for row in rows]
    return torch.tensor(values, dtype=torch.float64, requires_grad=requires_grad)


def test_fixed_measure_expectation_with_unequal_measures_missing_fields_and_multiple_frames():
    left_keys = [key(0), key(0), key(1), key(0, time=1)]
    right_keys = [key(0), key(0), key(0), key(1), key(0, time=1)]
    left = population('left', left_keys, component_count=3,
        references=np.arange(12).reshape(4, 3) / 13,
        measures=np.arange(1, 13).reshape(4, 3),
        complete=[[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1]],
        extraction_ids=[['direct', 'direct', 'direct'], ['direct', 'direct', 'direct'],
                        ['direct', 'different', 'direct'], ['direct', 'direct', 'direct']])
    right = population('right', right_keys, component_count=3,
        references=np.arange(15).reshape(5, 3) / 17,
        measures=np.arange(15, 0, -1).reshape(5, 3),
        valid=[[1, 1, 0]] * 5,
        extraction_ids=[['direct', 'direct', 'direct'], ['direct', 'direct', 'direct'],
                        ['direct', 'different', 'direct'], ['direct', 'direct', 'direct'],
                        ['direct', 'direct', 'direct']])
    plan = prepare_native_coupling(left, right)
    outcomes = enumerate_samples(plan, 1, 1, anchor_count=1)
    expected = np.zeros(3)
    for sample, probability in outcomes:
        result = coupled_response_loss(plan, sample,
            panel_predictions('left', left, sample.left_rows), panel_predictions('right', right, sample.right_rows),
            left_dependencies=native_dependencies(left, sample.left_rows),
            right_dependencies=native_dependencies(right, sample.right_rows))
        for component in range(2):
            expected[component] += probability * float(result.per_component[f'field_{component}'])
        assert result.per_component['field_2'] is None
    target = exact_response_target(left, right)
    np.testing.assert_allclose(expected[:2], target[:2], rtol=1.e-12, atol=1.e-12)
    assert target[2] is None
    assert plan.dependency_ineligible_rows[0, 1] == 1


def test_duplicate_anchor_contributions_gradients_and_panel_order():
    left = population('left', [key(0), key(0), key(1)], component_count=5,
                      measures=[[1, 2, 3, 4, 5], [2, 1, 4, 3, 1], [3, 2, 1, 2, 4]])
    right = population('right', [key(0), key(0), key(1)], component_count=5,
                       measures=[[3, 1, 2, 4, 1], [1, 3, 4, 2, 4], [2, 4, 1, 1, 2]])
    plan = prepare_native_coupling(left, right)
    # One occurrence is repeated, while the other is different. Repetitions
    # must remain separate terms despite their single native prediction.
    sample = NativeCoupledSample(plan, np.array([0, 1, 2]), np.array([1, 0, 2]),
        np.array([0, 0, 1]), np.array([1, 1, 0]), np.array([0, 0, 1]), np.array([0, 0, 1]),
        np.ones(3, dtype=bool), np.full(3, (2 / 3) / 4))
    left_pred = panel_predictions('left', left, sample.left_rows, requires_grad=True)
    right_pred = panel_predictions('right', right, sample.right_rows, requires_grad=True)
    dependencies = dict(left_dependencies=native_dependencies(left, sample.left_rows),
                        right_dependencies=native_dependencies(right, sample.right_rows))
    result = coupled_response_loss(plan, sample, left_pred, right_pred, **dependencies)
    assert result.realized_common_draws == 3
    assert all(value == 3 for value in result.eligible_draws.values())
    result.loss.backward()
    assert torch.isfinite(left_pred.grad).all() and torch.isfinite(right_pred.grad).all()
    # Independently evaluate each repeated contribution against the fixed mass.
    weights = result.inverse_probability_weights['field_0']
    residual = ((right_pred.detach()[sample.right_anchor_positions, 0] -
                 torch.as_tensor(right.references[sample.right_anchor_rows, 0])) -
                (left_pred.detach()[sample.left_anchor_positions, 0] -
                 torch.as_tensor(left.references[sample.left_anchor_rows, 0])))
    expected = (residual.square() * torch.as_tensor(weights)).sum() / (3 * result.eligible_mass['field_0'])
    torch.testing.assert_close(result.per_component['field_0'], expected)
    assert weights[0] == weights[1]
    flipped = NativeCoupledSample(plan, sample.left_rows[::-1].copy(), sample.right_rows[::-1].copy(),
        sample.left_anchor_rows, sample.right_anchor_rows, 2 - sample.left_anchor_positions,
        2 - sample.right_anchor_positions, sample.common, sample.common_row_probability)
    reordered = coupled_response_loss(plan, flipped, left_pred.flip(0), right_pred.flip(0), **dependencies)
    torch.testing.assert_close(result.loss, reordered.loss)


def test_exact_identity_dependency_eligibility_and_no_hidden_queries():
    first = key(0)
    nearby = ObservationKey('fixed', 0, (np.nextafter(0., 1.),), 'shared_frame', 'native')
    left = population('left', [first, key(1), key(2), key(3)], complete=[[1, 0], [1, 1], [1, 1], [1, 1]])
    right = population('right', [first, key(1, time=1), nearby,
                                  key(3, correspondence='other_member')])
    plan = prepare_native_coupling(left, right)
    assert plan.common_probability == pytest.approx(.25)
    assert plan.eligible_mass.tolist() == [1., 0.]
    # All rows are ordinary native rows, including unsupported extractions.
    sample = sample_native_coupling(plan, 4, 4, anchor_count=4, **generators(6))
    assert set(sample.left_rows) == set(range(4)) and set(sample.right_rows) == set(range(4))
    # Use a guaranteed common observation to prove no decoder or stencil
    # completion is invoked by the reducer.
    common = NativeCoupledSample(plan, sample.left_rows, sample.right_rows, np.array([0]), np.array([0]),
        np.flatnonzero(sample.left_rows == 0), np.flatnonzero(sample.right_rows == 0),
        np.array([True]), np.array([.25]))
    lpred = panel_predictions('left', left, common.left_rows, requires_grad=True)
    rpred = panel_predictions('right', right, common.right_rows, requires_grad=True)
    with pytest.raises(ValueError, match='omitted declared extraction dependencies'):
        coupled_response_loss(plan, common, lpred, rpred, left_dependencies=set(),
                              right_dependencies=native_dependencies(right, common.right_rows))
    result = coupled_response_loss(plan, common, lpred, rpred,
        left_dependencies=native_dependencies(left, common.left_rows),
        right_dependencies=native_dependencies(right, common.right_rows))
    assert result.per_component['field_0'] is not None and result.per_component['field_1'] is None
    result.loss.backward()
    assert torch.isfinite(lpred.grad).all() and torch.isfinite(rpred.grad).all()
    unsupported = prepare_native_coupling(left, right, sampling_law='weighted_without_replacement')
    assert not unsupported.available and 'Unsupported' in unsupported.reason


def test_zero_match_is_zero_not_unavailable_and_handles_two_views():
    left = population('left', [key(0), key(1)], component_count=2)
    right = population('right', [key(0), key(2)], component_count=2)
    plan = prepare_native_coupling(left, right)
    sample = NativeCoupledSample(plan, np.array([1]), np.array([1]), np.array([1]), np.array([1]),
        np.array([0]), np.array([0]), np.array([False]), np.array([0.]))
    lpred = torch.tensor([[[float('nan'), 2.]], [[3., 4.]]], requires_grad=True)
    rpred = torch.tensor([[[5., 6.]], [[7., 8.]]], requires_grad=True)
    result = coupled_response_loss(plan, sample, lpred, rpred)
    assert result.loss == 0 and result.per_component['field_0'] is not None
    result.loss.backward()
    assert torch.isfinite(lpred.grad).all() and torch.isfinite(rpred.grad).all()


@pytest.mark.parametrize('component_count', [1, 3, 7, 11])
def test_analytic_arbitrary_fields_and_companion_query_independence(component_count):
    keys = [key(index) for index in range(4)]
    left = population('left', keys, component_count=component_count)
    right = population('right', keys, component_count=component_count)
    plan = prepare_native_coupling(left, right)
    parameters = torch.linspace(.25, 2., component_count, dtype=torch.float64, requires_grad=True)

    def evaluate(scene, rows, extra_scene=False):
        # A pointwise analytic joint-field model with an optional unrelated
        # companion scene in the same call. Its result at the shared position
        # must be independent of the remaining query positions and scenes.
        positions = torch.as_tensor(np.asarray(rows, dtype=float), dtype=torch.float64)
        controls = torch.full_like(positions, 1. if scene == 'left' else 2.)
        if extra_scene:
            positions = torch.cat((positions, torch.tensor([70., 80.], dtype=torch.float64)))
            controls = torch.cat((controls, torch.tensor([9., 9.], dtype=torch.float64)))
        return ((positions[:, None] + controls[:, None]) * parameters[None])[:len(rows)]

    def calculate(companion, extra_scene):
        sample = NativeCoupledSample(plan, np.array([0, companion]), np.array([0, companion]),
            np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([True]), np.array([.25]))
        result = coupled_response_loss(plan, sample, evaluate('left', sample.left_rows, extra_scene),
            evaluate('right', sample.right_rows, extra_scene),
            left_dependencies=native_dependencies(left, sample.left_rows),
            right_dependencies=native_dependencies(right, sample.right_rows))
        gradient, = torch.autograd.grad(result.loss, parameters)
        return result.loss, gradient

    first, first_gradient = calculate(1, False)
    other, other_gradient = calculate(3, True)
    torch.testing.assert_close(first, other)
    torch.testing.assert_close(first_gradient, other_gradient)
    assert torch.isfinite(first_gradient).all()


def test_multi_query_extraction_uses_only_complete_native_stencil():
    first = population('left', [key(0)], component_count=3, complete=[[1, 0, 1]])
    second = population('right', [key(0)], component_count=3)
    # Field 0 is an already-extracted linear difference of two native queries.
    # Field 1 is not guaranteed complete, while field 2 remains usable.
    def with_stencil(pop, scene):
        dependencies = (((scene, 0, 0, 'query'), (scene, 0, 0, 'neighbor')),
                        ((scene, 0, 1, 'query'),), ((scene, 0, 2, 'query'),))
        return NativeObservationPopulation(pop.keys, pop.components, pop.references, pop.measures,
            pop.valid, pop.dependency_complete, (dependencies,))

    left, right = with_stencil(first, 'left'), with_stencil(second, 'right')
    plan = prepare_native_coupling(left, right)
    sample = sample_native_coupling(plan, 1, 1, anchor_count=1, **generators(4))
    left_query = torch.tensor([3., 2., 5., 7.], requires_grad=True)
    right_query = torch.tensor([8., 4., 6., 9.], requires_grad=True)
    extracted_left = torch.stack((left_query[0] - .5 * left_query[1], left_query[2], left_query[3]))[None]
    extracted_right = torch.stack((right_query[0] - .5 * right_query[1], right_query[2], right_query[3]))[None]
    left_native = native_dependencies(left, sample.left_rows)
    right_native = native_dependencies(right, sample.right_rows)
    with pytest.raises(ValueError, match='omitted declared extraction dependencies'):
        coupled_response_loss(plan, sample, extracted_left, extracted_right,
            left_dependencies=left_native - {('left', 0, 0, 'neighbor')}, right_dependencies=right_native)
    result = coupled_response_loss(plan, sample, extracted_left, extracted_right,
        left_dependencies=left_native, right_dependencies=right_native)
    assert result.per_component['field_1'] is None
    assert result.per_component['field_2'] is not None
    torch.testing.assert_close(result.per_component['field_0'], torch.tensor(16., dtype=torch.float64))
    result.loss.backward()
    assert torch.isfinite(left_query.grad).all() and torch.isfinite(right_query.grad).all()


def test_repeated_immutable_population_metadata_is_shared():
    fields = tuple(FieldComponent(f'field_{index}', 'unit', index, 1.) for index in range(3))
    first, second = key(0), key(0)
    direct = ((4,), (4,), (4,))
    pop = NativeObservationPopulation((first, second), fields, np.zeros((2, 3)), np.ones((2, 3)),
        np.ones((2, 3), dtype=bool), np.ones((2, 3), dtype=bool), (direct, direct))
    assert pop.keys[0] is pop.keys[1]
    assert pop.extraction_ids[0] is pop.extraction_ids[1]
    assert pop.dependencies[0] is pop.dependencies[1]
    assert pop.dependencies[0][0] is pop.dependencies[0][1]

"""Analytic checks for selected-support observation losses and extraction."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from honf_runtime.finite_observations import (
    AffineFrame,
    FiniteObservationFamily,
    FiniteObservationScene,
    compatible_scene_families,
    evaluate_observation_scene,
    extract_family_predictions,
    finite_scene_contrast,
    select_source_rows,
)
from honf_runtime.observation_blocks import FieldComponent, LinearExtraction, QueryDependency


def direct_family(scene_id, *, name='solid', component=None, coordinates=None, references=None,
                  measures=None, row_ids=None, native_weight=2., frame='world', support='solids'):
    coordinates = np.asarray(coordinates if coordinates is not None else [[0., 0.], [1., 1.]])
    count = len(coordinates)
    references = np.asarray(references if references is not None else np.zeros(count))
    measures = np.asarray(measures if measures is not None else np.ones(count))
    row_ids = tuple(row_ids if row_ids is not None else range(count))
    component = component or FieldComponent('temperature', 'K', 3, 2., 1.)
    dependencies = tuple(QueryDependency(scene_id, frame or 'no_coordinates', 'steady',
                                         (scene_id, index), tuple(position) if len(position) else (0.,))
                         for index, position in enumerate(coordinates))
    extraction = LinearExtraction(dependencies, np.arange(count), np.arange(count), np.zeros(count, dtype=int),
                                  np.full(count, component.output_index, dtype=int), np.ones(count))
    return FiniteObservationFamily(name=name, component=component, support=support, frame=frame, time='steady',
        extraction_id='direct_value', native_weight=native_weight, row_ids=row_ids, coordinates=coordinates,
        measures=measures, references=references, extraction=extraction)


def evaluate(scene_id, family, prediction, *, weights=None, frame=None, basis='coordinate'):
    return evaluate_observation_scene(FiniteObservationScene(scene_id, (family,)),
        {family.name: torch.as_tensor(prediction, dtype=torch.float64)},
        configured_weights=weights or {family.name: family.native_weight},
        frames={} if frame is None else {frame.identity: frame}, basis=basis)


def test_source_selection_preserves_distinct_rows_and_original_physical_measures():
    valid = np.array([True, False, True, True, True, True, False, True])
    first = select_source_rows(valid, count=4, seed=4321)
    assert np.array_equal(first, select_source_rows(valid, count=4, seed=4321))
    assert len(first) == len(set(first)) == 4
    assert all(valid[first]) and np.all(np.diff(first) > 0)
    assert np.array_equal(select_source_rows(valid, count=64, seed=4321), [0, 2, 3, 4, 5, 7])
    coordinates = np.array([[0., 0.], [0., 0.], [1., 0.]])
    measures = np.array([1., 3., 8.])
    rows = select_source_rows(np.ones(3, dtype=bool), count=64, seed=2)
    family = direct_family('scene', coordinates=coordinates[rows], measures=measures[rows],
                           row_ids=np.array(['member_one', 'member_two', 'member_three'])[rows])
    assert family.row_ids == ('member_one', 'member_two', 'member_three')
    np.testing.assert_array_equal(family.coordinates[0], family.coordinates[1])
    np.testing.assert_array_equal(family.measures, [1., 3., 8.])
    with pytest.raises(ValueError, match='boolean'):
        select_source_rows([1, 0, 1], seed=0)


def test_independent_ragged_scene_sums_fixed_missing_family_weights_and_zero_values():
    frame = AffineFrame('world', (0., 0.), (2., 2.))
    first = direct_family('first', coordinates=[[0., 0.], [2., 0.], [0., 2.]],
        references=[10., 20., 30.], measures=[1., 2., 1.], row_ids=['one', 'two', 'three'])
    second = direct_family('second', coordinates=[[2., 2.], [1., 1.]],
        references=[5., 9.], measures=[3., 1.], row_ids=['unrelated', 'also_unrelated'])
    weights = {'solid': 2., 'surface': 6.}
    first_result = evaluate('first', first, [12., 24., 24.], weights=weights, frame=frame)
    second_result = evaluate('second', second, [13., 13.], weights=weights, frame=frame)
    # Residuals are [1, 2, -3] and [4, 2]. Each physical support is
    # normalized independently, then native importance remains 2 / (2 + 6).
    torch.testing.assert_close(first_result.families['solid'].moments, torch.tensor([.5, 1.5, -2.], dtype=torch.float64))
    torch.testing.assert_close(second_result.families['solid'].moments, torch.tensor([3.5, 3., 3.], dtype=torch.float64))
    assert first_result.anchor.item() == pytest.approx(4.5 * .25)
    assert second_result.anchor.item() == pytest.approx(13. * .25)
    contrast = finite_scene_contrast(first_result, second_result, configured_weights=weights)
    assert contrast.loss.item() == pytest.approx((9. + 2.25 + 25.) / 3. * .25)
    assert contrast.eligible_families == ('solid',)
    assert contrast.missing_families == ('surface',)
    zeros = evaluate('first', first, first.references, weights=weights, frame=frame)
    assert zeros.anchor.item() == 0.
    assert zeros.families['solid'].moments.eq(0).all()
    assert zeros.missing_families == ('surface',)


def test_distinct_support_weights_and_duplicate_representation_splitting():
    material = direct_family('scene', name='material', native_weight=2., support='material')
    surface = direct_family('scene', name='surface', native_weight=6., support='surface')
    assert material.component.output_index == surface.component.output_index
    scene = FiniteObservationScene('scene', (material, surface))
    value = evaluate_observation_scene(scene, {'material': torch.tensor([2., 2.]), 'surface': torch.tensor([4., 4.])},
                                       configured_weights={'material': 2., 'surface': 6.})
    assert value.anchor.item() == pytest.approx((2. * 1. + 6. * 4.) / 8.)
    representation_one = replace(material, name='material_first_representation', native_weight=1.)
    representation_two = replace(material, name='material_second_representation', native_weight=1.)
    split_scene = FiniteObservationScene('scene', (representation_one, representation_two, surface))
    split = evaluate_observation_scene(split_scene, {'material_first_representation': torch.tensor([2., 2.]),
        'material_second_representation': torch.tensor([2., 2.]), 'surface': torch.tensor([4., 4.])},
        configured_weights={'material_first_representation': 1., 'material_second_representation': 1., 'surface': 6.})
    assert split.anchor.item() == pytest.approx(value.anchor.item())


def test_cancelled_scene_contrast_does_not_remove_absolute_errors():
    first = direct_family('first', coordinates=[[0., 0.], [1., 1.]], references=[4., 8.], measures=[1., 1.])
    second = direct_family('second', coordinates=[[0., 0.], [1., 1.]], references=[2., 6.], measures=[1., 1.])
    frame = AffineFrame('world', (0., 0.), (1., 1.))
    left = evaluate('first', first, [10., 14.], frame=frame)
    right = evaluate('second', second, [8., 12.], frame=frame)
    assert left.anchor.item() == right.anchor.item() == 9.
    assert finite_scene_contrast(left, right, configured_weights={'solid': 2.}).loss.item() == 0.


def test_view_dimensions_are_preserved_for_caller_weighting():
    first = direct_family('first', references=[0., 0.])
    second = direct_family('second', references=[0., 0.])
    predictions = torch.tensor([[2., 2.], [4., 4.]], dtype=torch.float64, requires_grad=True)
    first_result = evaluate_observation_scene(FiniteObservationScene('first', (first,)), {'solid': predictions},
                                              configured_weights={'solid': 2.})
    second_result = evaluate_observation_scene(FiniteObservationScene('second', (second,)),
        {'solid': torch.tensor([[4., 4.], [8., 8.]], dtype=torch.float64)}, configured_weights={'solid': 2.})
    torch.testing.assert_close(first_result.anchor, torch.tensor([1., 4.], dtype=torch.float64))
    response = finite_scene_contrast(first_result, second_result, configured_weights={'solid': 2.})
    torch.testing.assert_close(response.loss, torch.tensor([1., 4.], dtype=torch.float64))
    (response.loss * torch.tensor([.75, .25])).sum().backward()
    assert predictions.grad is not None and bool((predictions.grad != 0).all())


def test_matching_constant_and_coordinate_observations_can_hide_local_error():
    frame = AffineFrame('world', (0.,), (3.,))
    first = direct_family('first', coordinates=[[0.], [1.], [2.], [3.]],
                          references=[0., 0., 0., 0.])
    second = direct_family('second', coordinates=[[0.], [1.], [2.], [3.]],
                           references=[0., 0., 0., 0.])
    # The normalized residual vector [1, -3, 3, -1] is orthogonal to both
    # the constant and affine coordinate vectors; its point MSE is still 5.
    left = evaluate('first', first, [2., -6., 6., -2.], frame=frame)
    right = evaluate('second', second, [0., 0., 0., 0.], frame=frame)
    torch.testing.assert_close(left.families['solid'].moments, torch.zeros(2, dtype=torch.float64), atol=1e-15, rtol=0)
    assert left.anchor.item() == 5.
    assert finite_scene_contrast(left, right, configured_weights={'solid': 2.}).loss.item() == pytest.approx(0., abs=1e-30)


def test_common_frame_no_clipping_constant_fallback_and_three_node_hats():
    frame = AffineFrame('world', (0., 0.), (2., 2.))
    family = direct_family('scene', coordinates=[[0., 0.], [1., 1.], [2., 2.], [3., 1.]],
                           references=[0., 0., 0., 0.], measures=[1., 1., 1., 1.])
    coordinate = evaluate('scene', family, [2., 2., 2., 2.], frame=frame)
    # xi_x = [-1, 0, 1, 2], xi_y = [-1, 0, 1, 0], so x is not clipped.
    torch.testing.assert_close(coordinate.families['solid'].moments, torch.tensor([1., .5, 0.], dtype=torch.float64))
    assert coordinate.families['solid'].out_of_frame_count == 1
    fallback = evaluate('scene', family, [2., 2., 2., 2.])
    assert fallback.families['solid'].basis == 'constant'
    assert fallback.families['solid'].frame_definition is None
    assert fallback.families['solid'].moments.tolist() == [1.]
    hats = evaluate('scene', family, [2., 2., 2., 2.], frame=frame, basis='tensor_hats')
    # Three in-frame corners/centre each contribute 1/4; the out-of-frame
    # coordinate is evaluated without clipping and contributes zero here.
    expected = torch.tensor([.25, 0., 0., 0., .25, 0., 0., 0., .25], dtype=torch.float64)
    torch.testing.assert_close(hats.families['solid'].moments, expected)
    assert hats.families['solid'].out_of_frame_count == 1


def test_incompatible_support_units_frames_and_affine_maps_fail_clearly():
    first = direct_family('first')
    second = direct_family('second')
    frame = AffineFrame('world', (0., 0.), (2., 2.))
    first_result = evaluate('first', first, [0., 0.], frame=frame)
    incompatible = [replace(second, component=replace(second.component, unit='degC')),
                    replace(second, support='fluid'), replace(second, frame='other_world'),
                    replace(second, time='other_time'), replace(second, extraction_id='other_extraction')]
    for changed in incompatible:
        with pytest.raises(ValueError, match='Incompatible'):
            compatible_scene_families(FiniteObservationScene('first', (first,)),
                FiniteObservationScene('second', (changed,)), configured_weights={'solid': 2.})
        changed_frame = AffineFrame(changed.frame, (0., 0.), (2., 2.))
        right = evaluate('second', changed, [0., 0.], frame=changed_frame)
        with pytest.raises(ValueError, match='Incompatible'):
            finite_scene_contrast(first_result, right, configured_weights={'solid': 2.})
    recentered = evaluate('second', second, [0., 0.], frame=AffineFrame('world', (0., 0.), (1., 1.)))
    with pytest.raises(ValueError, match='common affine frame'):
        finite_scene_contrast(first_result, recentered, configured_weights={'solid': 2.})
    absent = evaluate_observation_scene(FiniteObservationScene('empty', ()), {}, configured_weights={'solid': 2.})
    assert compatible_scene_families(first_result.scene, absent.scene,
                                      configured_weights={'solid': 2.}).eligible_families == ()
    missing = finite_scene_contrast(first_result, absent, configured_weights={'solid': 2.})
    assert absent.anchor is None and missing.loss is None
    assert missing.eligible_families == () and missing.missing_families == ('solid',)


def test_sparse_stencil_union_affine_conversion_and_gradients():
    family = direct_family('scene', coordinates=[[0., 0.], [1., 1.]])
    dependencies = tuple(QueryDependency('scene', 'world', 'steady', index, (float(index), 0.)) for index in range(3))
    extraction = LinearExtraction(dependencies, np.array([1, 0, 2, 0]), np.array([0, 0, 1, 1]),
        np.array([0, 0, 0, 0]), np.array([0, 1, 1, 0]), np.array([2., -1., .5, 1.]))
    family = replace(family, extraction=extraction)
    # The execution union order is C, A, B; family dependencies are A, B, C.
    prediction = torch.tensor([[5., 8.], [1., 4.], [3., 6.]], dtype=torch.float64, requires_grad=True)
    value = extract_family_predictions(family, prediction, dependency_indices=[1, 2, 0],
                                       output_scale=[2., 3.], output_offset=[10., -1.])
    torch.testing.assert_close(value, torch.tensor([21., 23.5], dtype=torch.float64))
    value.sum().backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[0., 1.5], [2., -3.], [4., 0.]], dtype=torch.float64))
    with pytest.raises(ValueError, match='every family dependency'):
        extract_family_predictions(family, prediction, dependency_indices=[1, 0])
    incomplete = LinearExtraction(dependencies, np.array([0]), np.array([0]), np.array([0]),
                                  np.array([0]), np.array([1.]))
    with pytest.raises(ValueError, match='Every selected row'):
        replace(family, extraction=incomplete)


def test_seven_output_nonthermal_scene_fixture_and_joint_gradients():
    # Seven observable components in an arbitrary porous-flow/reaction case;
    # each uses a different joint output and independently selected supports.
    fields = [FieldComponent(f'field_{index}', f'unit_{index}', index, float(index + 1)) for index in range(7)]
    weights = {f'support_{index}': float(index + 1) for index in range(7)}
    relation = torch.nn.Parameter(torch.tensor(.7, dtype=torch.float64))
    content = torch.nn.Parameter(torch.linspace(.2, 1.4, 7, dtype=torch.float64))
    frame = AffineFrame('world', (0., 0.), (2., 2.))
    scene_results = []
    for scene_id, coordinates in [('upstream', [[0., 0.], [1., 0.], [1., 1.]]),
                                   ('downstream', [[0., 2.], [2., 1.]])]:
        position = torch.tensor(coordinates, dtype=torch.float64)
        x, y = position[:, 0:1], position[:, 1:2]
        joint_output = relation * (1. + x + y) + content * (1. + x - .25 * y)
        families, predictions = [], {}
        for index, component in enumerate(fields):
            family = direct_family(scene_id, name=f'support_{index}', component=component,
                coordinates=coordinates, references=np.full(len(coordinates), .3 * index),
                measures=np.arange(1, len(coordinates) + 1), native_weight=weights[f'support_{index}'],
                support=f'porous_support_{index}', row_ids=[(scene_id, index, row) for row in range(len(coordinates))])
            families.append(family)
            predictions[family.name] = extract_family_predictions(family, joint_output)
        scene_results.append(evaluate_observation_scene(FiniteObservationScene(scene_id, families), predictions,
            configured_weights=weights, frames={'world': frame}))
    contrast = finite_scene_contrast(*scene_results, configured_weights=weights)
    assert len(contrast.eligible_families) == 7
    loss = scene_results[0].anchor + scene_results[1].anchor + contrast.loss
    loss.backward()
    assert torch.isfinite(loss) and relation.grad.abs().item() > 0
    assert torch.isfinite(content.grad).all() and bool((content.grad.abs() > 0).all())

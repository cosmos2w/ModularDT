"""The figure metrics reflect the valid points in the plotted physical fields."""
import re
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _values(ax):
    labels = [ax.get_title()] + [text.get_text() for text in ax.texts]
    return [float(value) for label in labels for value in re.findall(r'RMSE\s*=\s*([\d.eE+-]+)', label)]


def test_thermal_internal_rmse_excludes_cells_outside_module_mask(tmp_path, monkeypatch):
    from channelthermal.evaluation_tools.plots import plot_internal
    original_close = plt.close
    monkeypatch.setattr(plt, 'close', lambda *args: None)
    sample = {'structure': {'module_present': np.array([1])},
              'module_internal_mask': np.array([[True, True], [True, False]]),
              'module_internal_temperature_points': np.array([[0., 0., 0.]])}
    try:
        assert plot_internal(tmp_path/'field.png', sample, np.array([[[3.], [4.], [0.]]]))
        fig=plt.gcf()
        assert _values(fig.axes[2]) == pytest.approx([np.sqrt(25/3)], rel=1e-4)
    finally:
        original_close('all')


@pytest.mark.parametrize('kind', ['native', 'comparison'])
def test_wind_error_panels_report_each_velocity_component(tmp_path, monkeypatch, kind):
    from windfarm.study_spatial import render_native_plane
    from windfarm.comparative_visualization import render_comparison_plane
    original_close=plt.close
    monkeypatch.setattr(plt, 'close', lambda *args: None)
    target=np.zeros((2, 3, 3), dtype=np.float32)
    prediction=np.broadcast_to(np.array([1., 2., 3.]), target.shape).copy()
    plane={'target': target, 'horizontal_axis':'x', 'vertical_axis':'y', 'fixed_axis':'z',
           'horizontal_D':np.arange(3.), 'vertical_D':np.arange(2.), 'actual_m':0.}
    try:
        if kind == 'native':
            render_native_plane(plane, prediction, np.empty((0,3)), 'case', tmp_path/'native.png')
            panel_axes=plt.gcf().axes[:9]
            values=[_values(panel_axes[i]) for i in (2,5,8)]
            assert values == [[1.], [2.], [3.]]
        else:
            render_comparison_plane(plane, {'classic':prediction, 'dense':2*prediction},
                                    np.empty((0,3)), 'case', tmp_path/'comparison.png')
            panel_axes=plt.gcf().axes[:15]
            values=[_values(panel_axes[i]) for i in (3,4,8,9,13,14)]
            assert values == [[1.], [2.], [2.], [4.], [3.], [6.]]
    finally:
        original_close('all')

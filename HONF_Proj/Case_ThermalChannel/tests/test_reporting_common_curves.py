import json

import matplotlib.figure

from channelthermal.training.reporting import _read_metric_history, _save_common_csv_loss_plot


def test_legacy_csv_common_panels_use_only_saved_measurements(tmp_path, monkeypatch):
    path = tmp_path / 'metrics.csv'
    path.write_text('epoch,loss_total,val_loss_total,loss_field,val_field_mse\n1,4,,3,2\n2,2,1,1,0.5\n')
    history = _read_metric_history(path)
    captured = []
    def savefig(figure, target, **kwargs):
        captured.append([(ax.get_title(), [(line.get_label(), list(line.get_xdata()), list(line.get_ydata())) for line in ax.lines]) for ax in figure.axes])
        target.write_bytes(b'plot')
    monkeypatch.setattr(matplotlib.figure.Figure, 'savefig', savefig)
    _save_common_csv_loss_plot(history, tmp_path, run_name='legacy')
    panels = dict(captured[0])
    assert set(panels) == {'Weighted objective (reductions may differ)', 'Validation field error', 'Field prediction loss'}
    assert panels['Weighted objective (reductions may differ)'][1] == ('Validation', [2.0], [1.0])
    assert panels['Validation field error'][0][2] == [2.0, 0.5]
    metadata = json.loads((tmp_path / 'loss_curve_metadata.json').read_text())
    assert metadata['panels']['Weighted objective (reductions may differ)'] == ['loss_total', 'val_loss_total']
    assert 'cannot be reconstructed' in metadata['missing_validation_losses']
    assert path.read_text().startswith('epoch,loss_total')


def test_expanded_1804_rows_omit_ambiguous_validation_values(tmp_path):
    import csv
    import math
    from channelthermal.training.reporting import _read_common_csv_history

    path = tmp_path / 'metrics.csv'
    header = ['epoch', 'loss_total', 'val_loss_total', 'val_field_mse', 'val_temperature_mse',
              'val_predicted_loss_total', 'val_predicted_field_mse', 'val_predicted_temperature_mse']
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerow([1, 2, 3, 4, 5, 6, 7, 8])
        writer.writerow([2, 1] + [9] * (len(header) + 38 - 5) + [.3, .4, .5])
    history = _read_common_csv_history(path, expanded_predicted_1804=True)
    assert history['loss_total'] == [2., 1.]
    assert history['val_loss_total'][0] == 3. and math.isnan(history['val_loss_total'][1])
    assert history['val_field_mse'][0] == 4. and math.isnan(history['val_field_mse'][1])
    assert history['val_temperature_mse'][0] == 5. and math.isnan(history['val_temperature_mse'][1])
    assert math.isnan(history['val_predicted_loss_total'][1])

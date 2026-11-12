#!/usr/bin/env python3
"""Summarize saved Thermal action evaluations by baseline exposure and edit."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / 'src', ROOT / 'Case_ThermalChannel/src', ROOT / 'Case_WindFarm/src', ROOT / 'tools'):
    sys.path.insert(0, str(directory))

import numpy as np
from channelthermal.physical_response import summarize_actions
from thermal_state_response_evaluate import summarize_fields


def stratify(evaluation, primary_case_ids):
    """Retain all actions and read paths; group families equally within a stratum."""
    primary = set(map(str, primary_case_ids))
    floors = evaluation['functional_floors']
    result = {}
    for view, data in evaluation['views'].items():
        rows = data['actions']
        if not rows:
            raise ValueError('Response evaluation must contain actions.')
        groupings = defaultdict(lambda: defaultdict(list))
        for row in rows:
            source_delta = float(row['strata']['represented_source_delta'])
            change = 'unchanged' if abs(source_delta) <= 1e-12 else ('increase' if source_delta > 0 else 'decrease')
            labels = {'baseline_exposure': 'primary_seen' if str(row['case_id']) in primary else 'primary_unseen',
                      'action_type': row['action'].split('_')[0], 'sign': row['action'].split('_')[1],
                      'amplitude': str(row['strata']['amplitude']), 'module_count': str(row['strata']['module_count']),
                      'represented_source_change': change}
            for category, label in labels.items():
                groupings[category][label].append(row)

        def summarize(selected):
            families = sorted({str(row['case_id']) for row in selected})
            functions = summarize_actions(selected, floors, family_ids=families)
            for name, metrics in functions.items():
                effect = float(np.mean([np.mean([abs(row['delta'][name]['physical']) for row in selected
                                                  if str(row['case_id']) == family]) for family in families]))
                metrics['all_action_equal_family_mean_abs_effect'] = effect
                metrics['all_action_mae_over_mean_abs_effect'] = (
                    metrics['all_action_equal_family_mae'] / effect if effect > floors[name] else None)
                metrics['near_null_action_count'] = metrics['all_action_count'] - metrics['resolved_sign_count']
            return {'family_ids': families, 'family_count': len(families), 'action_count': len(selected),
                    'functionals': functions, 'fields': summarize_fields(selected)}

        result[view] = {'all_actions': summarize(rows),
                        'strata': {category: {label: summarize(selected) for label, selected in groups.items()}
                                   for category, groups in groupings.items()}}
    return {'role': evaluation['role'], 'checkpoint': evaluation['checkpoint'], 'epoch': evaluation['epoch'],
            'primary_membership_count': len(primary),
            'functional_floors': floors, 'views': result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity', type=Path, required=True,
                        help='Saved experiment_identity.json with primary TRAIN case IDs')
    parser.add_argument('--input', type=Path, action='append', required=True,
                        help='Saved action evaluation JSON; repeat for several runs or partitions')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    identity = json.loads(args.identity.read_text())
    primary = identity['provider_identity']['training_case_ids']
    results = []
    for path in args.input:
        data = json.loads(path.read_text())
        results.append({'source': str(path), **stratify(data, primary)})
    payload = {'schema': 'thermal_response_exposure_strata_v1',
               'primary_membership_source': str(args.identity),
               'primary_membership_count': len(set(map(str, primary))),
               'primary_membership_label': identity['recipe']['training']['membership'],
               'interpretation': 'Seen and unseen refer only to the supplied primary training membership; FIT edits are supervised and holdout edits are withheld.',
               'aggregation': 'Within each stratum average actions within family, then average families equally.',
               'source_change_threshold': 1e-12,
               'field_relative_error_note': 'Physical reference RMS and RMSE use the same valid receivers. Ratio is unavailable for an exact-zero reference; no numerical floor indicates physical resolution. Old action files may lack reference RMS.',
               'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, args.output)
    print(json.dumps({'output': str(args.output), 'evaluations': len(results)}))


if __name__ == '__main__':
    main()

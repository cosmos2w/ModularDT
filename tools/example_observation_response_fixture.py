#!/usr/bin/env python3
"""Run and verify a tiny CPU observation-response example on DATA2.

This is a software fixture with generated analytic values, not a scientific
training experiment. It exercises the public fit, saved-config resume and
observation-evaluation commands in separate processes.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))

import numpy as np
import torch

from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.storage_policy import _ensure_workspace_storage_bindings, verify_storage


def recipe_for(output_root):
    return {'schema':'analytic_observation_response_example_v1', 'name':'Coupled_Seven_Field_Observation_Example',
        'dataset':'analytic_fixture', 'architecture':'relation_anchored', 'seed':7, 'sampling_seed':7,
        'total_epochs':2, 'microbatch_size':3, 'effective_batch_size':3, 'fixture':{'output_dim':7},
        'training_profile':'observation_response', 'observation_response':{
            'adapter':'analytic_fixture', 'variant':'spatial_contrasts', 'pair_seed':3,
            'pairs_per_epoch':3, 'seeds_per_pair':8, 'fixture_no_ramp':True,
            'calibration':{'coefficients':{'extra_absolute':.2,'point_response':.1,'spatial_response':.05}}},
        'execution':{'monitor_every':1,'monitor_epochs':[1,2],'checkpoint_epochs':[1,2],
                     'latest_every':1,'curve_every':10,'progress_enabled':False},
        'storage':{'output_root':str(output_root), 'max_output_bytes':1073741824,
                   'max_peak_temporary_bytes':268435456,'minimum_free_headroom_bytes':322122547200}}


def _same(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left,right)
    if isinstance(left,np.ndarray):
        return isinstance(right,np.ndarray) and np.array_equal(left,right)
    if isinstance(left,dict):
        return isinstance(right,dict) and left.keys()==right.keys() and all(_same(left[k],right[k]) for k in left)
    if isinstance(left,(tuple,list)):
        return isinstance(right,type(left)) and len(left)==len(right) and all(_same(a,b) for a,b in zip(left,right))
    return left==right


def _rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def _check_runs(full, resumed):
    first=load_trusted_checkpoint(full/'checkpoints/latest_model.pt',map_location='cpu')
    second=load_trusted_checkpoint(resumed/'checkpoints/latest_model.pt',map_location='cpu')
    keys=('epoch','model_state_dict','optimizer_state_by_name','rng_state','sampler_state',
          'provider_training_state','best_field_score','optimizer_group_names','optimizer_schedule_contract')
    for key in keys:
        if not _same(first[key],second[key]):
            raise AssertionError(f'Fresh-process continuation changed {key}.')
    full_defs=json.loads((full/'configs/metric_definitions.json').read_text())
    resumed_defs=json.loads((resumed/'configs/metric_definitions.json').read_text())
    if full_defs != resumed_defs:
        raise AssertionError('Ordered recording columns or loss definitions differ.')
    full_rows,resumed_rows=_rows(full/'metrics/history.csv'),_rows(resumed/'metrics/history.csv')
    if len(full_rows)!=2 or len(resumed_rows)!=2:
        raise AssertionError('Both lineages must retain both completed epochs.')
    volatile={'train_seconds','work_counts.observation_response_planning_seconds',
              'work_counts.observation_response_prediction_seconds'}
    for left,right in zip(full_rows,resumed_rows):
        for key in full_defs['training_columns']:
            if key not in volatile and left[key]!=right[key]:
                raise AssertionError(f'Recorded column {key} differs at epoch {left["epoch"]}.')
    if _rows(full/'metrics/validation.csv') != _rows(resumed/'metrics/validation.csv'):
        raise AssertionError('Validation histories differ after resume.')
    return {'completed_epochs':2, 'checkpoint_states_equal':list(keys),
            'ordered_columns_equal':True, 'recorded_scientific_values_equal':True,
            'validation_history_equal':True, 'elapsed_time_columns_excluded':sorted(volatile)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True,help='New empty directory below mounted /data2')
    args=parser.parse_args(argv)
    root=args.output_root.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Choose a new empty fixture output root; existing results are protected.')
    recipe=recipe_for(root)
    _ensure_workspace_storage_bindings()
    verify_storage(recipe,output=root,estimated_bytes=256*1024**2,peak_temporary_bytes=128*1024**2)
    root.mkdir(parents=True,exist_ok=True)
    recipe_path=root/'fixture_recipe.json'; recipe_path.write_text(json.dumps(recipe,indent=2)+'\n')
    panel_path=root/'fixture_pair_panel.json'; panel_path.write_text(json.dumps(
        {'name':'Coupled_Analytic_Validation_Pair','pairs':[['fixture_6','fixture_7']]},indent=2)+'\n')
    env=os.environ.copy(); env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    cli=PROJECT/'tools/relation_anchored_train.py'
    uninterrupted=root/'Uninterrupted_Seven_Field_Fixture'
    segmented=root/'Resumed_Seven_Field_Fixture'
    commands=[('uninterrupted', [str(cli),'fit','--recipe',str(recipe_path),'--output',str(uninterrupted),'--stop-after','2']),
              ('initial', [str(cli),'fit','--recipe',str(recipe_path),'--output',str(segmented),'--stop-after','1']),
              ('resume', [str(cli),'fit','--recipe',str(segmented/'configs/resolved_config.json'),
                          '--output',str(segmented),'--resume','latest','--stop-after','2'])]
    for name,args in commands:
        completed=subprocess.run([sys.executable,*args],cwd=PROJECT,env=env,text=True,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,check=False)
        (root/f'{name}.log').write_text(completed.stdout)
        if completed.returncode:
            raise RuntimeError(f'Fixture {name} exited {completed.returncode}; inspect {root / (name + ".log")}')
    receipt=_check_runs(uninterrupted,segmented)
    evaluation=root/'fixture_evaluation.json'
    result=subprocess.run([sys.executable,str(PROJECT/'tools/observation_response_evaluate.py'),
        '--recipe',str(segmented/'configs/resolved_config.json'),'--run',str(segmented),
        '--pair-panel',str(panel_path),'--output',str(evaluation),'--device','cpu'],cwd=PROJECT,env=env,
        text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False)
    (root/'evaluate.log').write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'Fixture evaluation exited {result.returncode}; inspect {root / "evaluate.log"}')
    data=json.loads(evaluation.read_text())
    if set(data['summary']) != {'full','restricted'} or data['pair_count'] != 1:
        raise AssertionError('Fixture evaluation did not produce both declared physical views.')
    receipt.update(evaluation=str(evaluation), output_root=str(root), sample='fixture_6, fixture_7')
    (root/'fixture_recovery_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()

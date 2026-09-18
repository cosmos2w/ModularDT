"""One fixed-coefficient calibration on two real batches; no optimizer updates."""
from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_routing_optimization as training
import run_dynamic_sparse_routing_study as study
from benchmark_sparse_execution import set_mode
from channelthermal.training.epoch import (
    assemble_channelthermal_loss_terms,
    effective_port_global_weight,
    make_model_inputs,
)

from honf_forward_core.config import RoutingSparsificationConfig
from honf_forward_core.interface_fields.core import InterfaceFieldCore
from honf_runtime.compat import recursive_to_device


def science_model(parent, device, backend):
    """Keep loaded case/local normalization; rebuild only the opt-in core."""
    model = copy.deepcopy(parent)
    routing = model.config.core_honf.interface_model.routing
    routing.execution = 'compiled_exact'
    routing.qe_backend = backend
    routing.sparsification = RoutingSparsificationConfig(enabled=True, learn_typed_temperatures=True)
    model.core = InterfaceFieldCore(model.config.core_honf).to(device)
    source = parent.core.state_dict()
    temperatures = {f'routing_log_temperatures.{k}': v for k, v in model.core.routing_log_temperatures.items()}
    # Ordinary strict loading with exactly four explicitly initialized entries.
    model.core.load_state_dict({**source, **temperatures}, strict=True)
    return model


def gradient_norm(loss, parameters, *, retain_graph):
    if not loss.requires_grad:
        return 0., {name: 0. for name, _ in parameters}
    grads = torch.autograd.grad(loss, [p for _, p in parameters], retain_graph=retain_graph, allow_unused=True)
    norms = {name: float(g.double().norm()) if g is not None else 0.
             for (name, _), g in zip(parameters, grads)}
    return math.sqrt(sum(v*v for v in norms.values())), norms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--backend', choices=['torch', 'triton'], default='torch')
    args = p.parse_args()
    torch.manual_seed(0)
    device = torch.device('cuda:0')
    parent, checkpoint = study._load_model_spec(study.parse_checkpoint_specs([args.checkpoint])[0], device)
    set_mode(parent, 'compiled_exact', args.backend)
    model = science_model(parent, device, args.backend)
    dataset, _ = study._load_dataset(checkpoint, SimpleNamespace(dataset=None, split='train'),
                                   points_per_case_override=1024, random_point_sampling_override=False)
    dataset.include_grid = False
    schedule = training._training_schedule(checkpoint)
    _, loss_cfg, mode, ratio, internal_weight, interface_weight, predicted_weight = schedule
    parameters = [(n, v) for n, v in model.named_parameters() if v.requires_grad and
                  n.startswith(('core.backend.router.', 'core.routing_log_temperatures.'))]
    payload = {'checkpoint': args.checkpoint, 'parent_epoch': checkpoint['epoch'], 'backend': args.backend,
               'costs': {'module': 1., 'environment': 1.}, 'epsilon': .05,
               'objective': 'equal-weight induced-pair-count surrogate', 'rows': [],
               'parameter_names': [n for n, _ in parameters], 'optimizer_updates': 0}
    for bucket in training._make_buckets([dataset], batch_size=48, requested='both'):
        loader = training._build_loader(dataset, checkpoint, bucket, batch_size=48)
        batch = recursive_to_device(next(iter(loader)), device)
        inputs = make_model_inputs(batch, local_port_condition_mode=mode, mixed_teacher_ratio=ratio,
                    return_predicted_port_outputs=predicted_weight > 0.,
                    return_port_global_consistency=effective_port_global_weight(loss_cfg, mode, ratio) != 0.)
        model.eval()
        with torch.no_grad():
            reference = parent(**inputs)['pred_field']
            initial = model(**inputs)['pred_field']
            init_error = float((initial-reference).norm()/reference.norm())
        del reference, initial
        model.train()
        output = model(**inputs)
        terms = assemble_channelthermal_loss_terms(output, batch, model, loss_cfg,
                    local_port_condition_mode=mode, mixed_teacher_ratio=ratio,
                    effective_internal_temperature_weight=internal_weight,
                    effective_interface_weight=interface_weight, predicted_consistency_weight=predicted_weight)
        task_norm, task_parameters = gradient_norm(terms['loss'], parameters, retain_graph=True)
        cost_norm, cost_parameters = gradient_norm(terms['loss_paircost'], parameters, retain_graph=False)
        payload['rows'].append({'bucket': bucket.label, 'batch_size': 48, 'queries': 1024,
            'case_ids': bucket.case_ids, 'initial_field_relative_error': init_error,
            'physical_loss': float(terms['loss']), 'paircost': float(terms['loss_paircost']),
            'task_gradient_norm': task_norm, 'cost_gradient_norm': cost_norm,
            'task_parameter_norms': task_parameters, 'cost_parameter_norms': cost_parameters})
        del output, terms, batch, inputs
        study.write_json(args.output, payload)
    task_norm = math.sqrt(sum(r['task_gradient_norm']**2 for r in payload['rows']))
    cost_norm = math.sqrt(sum(r['cost_gradient_norm']**2 for r in payload['rows']))
    if not math.isfinite(task_norm) or not math.isfinite(cost_norm) or cost_norm <= 0:
        payload['status'] = 'zero_or_nonfinite_cost_gradient_no_coefficient'
        study.write_json(args.output, payload)
        raise RuntimeError('Cannot calibrate a fixed coefficient from the measured gradients.')
    payload.update(status='calibrated', fixed_lambda=.02*task_norm/cost_norm,
                   aggregate_task_norm=task_norm, aggregate_cost_norm=cost_norm,
                   aggregate='sqrt(sum of squared batch norms)', target_initial_gradient_ratio=.02)
    study.write_json(args.output, payload)
    dataset.close()
    print(args.output, payload['fixed_lambda'])


if __name__ == '__main__':
    main()

"""Manual full-data recipes using the real shared interaction training engine.

This command does not promote a development checkpoint. Preparation has a
separate full-TRAIN identity and freshly fitted transforms; metadata-only
preparation and dry-run validation never create an optimizer or a run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_WORKFLOW = 'unified_interaction_refinement_formal_v1'


def _paths():
    for path in (PROJECT_ROOT / 'src', PROJECT_ROOT / 'tools',
                 PROJECT_ROOT / 'Case_ThermalChannel/src', PROJECT_ROOT / 'Case_WindFarm/src'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _factory(task):
    _paths()
    if task == 'thermal':
        from channelthermal.training import unified_formal
    elif task == 'wind':
        from windfarm.training import unified_formal
    else:
        raise ValueError('A formal recipe must name Thermal or Wind explicitly.')
    return unified_formal


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _write(path, value):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def _read(path):
    value = json.loads(Path(path).expanduser().resolve().read_text())
    if not isinstance(value, dict):
        raise TypeError('Recipe must be a JSON object.')
    return value


def _profile(task, *, startup_benchmark=False, monitoring=None, microbatch_cases=None):
    from honf_runtime.unified_training import EngineConfig, SelectionPolicy

    micro, effective, seed = (8, 48, 0) if task == 'thermal' else (4, 24, 42)
    if microbatch_cases is not None:
        if type(microbatch_cases) is not int or not 1 <= microbatch_cases <= effective:
            raise ValueError('Explicit microbatch size must be an integer inside the unchanged effective batch.')
        micro = microbatch_cases
    config = EngineConfig(seed=seed, microbatch_cases=micro, effective_cases=effective,
        total_epochs=5000, warmup_epochs=500, open_through_epoch=600,
        soft_through_epoch=800, monitor_every=100, gradient_clip=1.0)
    if monitoring is not None:
        required = {'validation_every', 'validation_epochs', 'checkpoint_epochs', 'latest_every', 'curve_every'}
        if set(monitoring) != required:
            raise ValueError('Formal monitoring contract keys differ.')
        if type(monitoring['validation_every']) is not int or monitoring['validation_every'] < 1:
            raise ValueError('Formal validation interval must be a positive integer.')
        config = EngineConfig(seed=seed, microbatch_cases=micro, effective_cases=effective,
            total_epochs=5000, warmup_epochs=500, open_through_epoch=600,
            soft_through_epoch=800, monitor_every=monitoring['validation_every'], gradient_clip=1.0,
            monitor_epochs=tuple(monitoring['validation_epochs']),
            checkpoint_epochs=tuple(monitoring['checkpoint_epochs']),
            latest_every=monitoring['latest_every'], curve_every=monitoring['curve_every'])
        if not config.checkpoint_epochs or config.checkpoint_epochs[-1] != 5000:
            raise ValueError('The formal retained milestones must include terminal epoch5000.')
    # Fresh full-data weights have no inherited development response guard.
    # Native response metrics remain dataset-owned, separately reported.
    selection = SelectionPolicy(field_metric='field_score', response_guard_metric=None)
    return config, selection


def _load_recipe(path):
    recipe = _read(path)
    saved_sha = recipe.pop('recipe_sha256', None)
    if saved_sha != _canonical_sha(recipe):
        raise ValueError('Formal recipe hash does not verify.')
    if recipe.get('workflow') != FORMAL_WORKFLOW or recipe.get('task') not in ('thermal', 'wind'):
        raise ValueError('This is not a separately bound unified formal recipe.')
    if recipe.get('execution_arm') not in ('full_detail', 'adaptive_detail'):
        raise ValueError('A formal recipe must choose one declared execution arm.')
    native = recipe['dataset_recipe']
    if (recipe.get('ready_for_training') is not bool(native.get('ready_for_training', False))
            or recipe.get('horizon_epochs') != 5000
            or recipe.get('fine_hold_through_epoch') != 2000
            or recipe.get('development_weights_loaded') is not False
            or recipe.get('optimizer_started') is not False
            or recipe.get('new_solver_attempts') != 0):
        raise ValueError('Formal workflow readiness, fresh-state or horizon contract differs.')
    _factory(recipe['task']).validate_recipe(native)
    _profile(recipe['task'], monitoring=recipe.get('monitoring'))
    recipe['recipe_sha256'] = saved_sha
    return recipe


def prepare(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    output = Path(args.recipe).expanduser().resolve()
    if output.exists():
        raise ValueError('Preparation requires a new recipe path; preserved recipes are not overwritten.')
    monitoring = {
        'validation_every': args.validation_every, 'validation_epochs': args.validation_epochs,
        'checkpoint_epochs': args.checkpoint_epochs,
        'latest_every': args.latest_every, 'curve_every': args.curve_every,
    }
    _paths()
    _profile(args.task, monitoring=monitoring)
    config = {'seed': 0 if args.task == 'thermal' else 42,
              'output_dir': str(output.parent / 'prepared_data')}
    for name in ('flow_checkpoint', 'data_root', 'dataset_path'):
        value = getattr(args, name, None)
        if value:
            config[name] = str(Path(value).expanduser().resolve())
    native = _factory(args.task).prepare_recipe(config, metadata_only=args.metadata_only)
    recipe = {'workflow': FORMAL_WORKFLOW, 'task': args.task,
        'execution_arm': args.arm, 'dataset_recipe': native,
        'created_unix': time.time(), 'ready_for_training': bool(native.get('ready_for_training', False)),
        'development_weights_loaded': False, 'optimizer_started': False,
        'normalization_rule': 'freshly fitted on all bound original TRAIN cases only',
        'horizon_epochs': 5000, 'fine_hold_through_epoch': 2000,
        'new_solver_attempts': 0}
    recipe['monitoring'] = monitoring
    recipe['recipe_sha256'] = _canonical_sha(recipe)
    _write(output, recipe)
    return {'recipe': str(output), 'recipe_sha256': recipe['recipe_sha256'],
            'ready_for_training': recipe['ready_for_training'], 'optimizer_started': False}


def dry_run(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    recipe = _load_recipe(args.recipe)
    config, selection = _profile(recipe['task'], monitoring=recipe.get('monitoring'))
    return {'status': 'formal_metadata_and_parser_checked',
        'recipe_sha256': recipe['recipe_sha256'],
        'ready_for_training': recipe['ready_for_training'],
        'dataset_checks': _factory(recipe['task']).validate_recipe(recipe['dataset_recipe']),
        'engine_config': config.__dict__, 'selection_policy': selection.__dict__,
        'optimizer_created': False, 'training_started': False, 'checkpoint_written': False}


def train(args, *, resume=False):
    _paths()
    from unified_interaction_train import _model_state_sha256

    from honf_runtime.reproducibility import seed_all
    from honf_runtime.unified_training import TrainingEngine

    recipe = _load_recipe(args.recipe)
    if not recipe['ready_for_training']:
        raise ValueError('Metadata-only recipes cannot train; prepare fresh full-TRAIN transforms first.')
    output = Path(args.run_dir).expanduser().resolve()
    if not resume and output.exists() and next(output.iterdir(), None) is not None:
        raise ValueError('A formal start requires a new empty run directory; use exact resume for saved runs.')
    config, selection = _profile(recipe['task'], startup_benchmark=args.startup_benchmark,
                                 monitoring=recipe.get('monitoring'),
                                 microbatch_cases=getattr(args, 'microbatch_cases', None))
    if args.startup_benchmark and not 1 <= args.stop_after <= 3:
        raise ValueError('The separately labelled manual startup benchmark is limited to three complete epochs.')
    if not 1 <= args.stop_after <= 5000:
        raise ValueError('Formal stop must be inside its separately declared 5000-epoch horizon.')
    seed_all(config.seed)
    native = {**recipe['dataset_recipe'], 'device': args.device,
              'startup_benchmark': bool(args.startup_benchmark)}
    model, provider, optimizer_seed = _factory(recipe['task']).create_task(native)
    if optimizer_seed is not None:
        raise ValueError('The formal recipe requires fresh optimizer moments.')
    identity = {'workflow': FORMAL_WORKFLOW, 'task': recipe['task'],
        'recipe_sha256': recipe['recipe_sha256'], 'run_identity': output.name,
        'startup_benchmark': bool(args.startup_benchmark),
        'initial_model_state_sha256': _model_state_sha256(model)}
    engine = TrainingEngine(config, device=args.device, selection=selection)
    return engine.fit(model, provider, output, identity=identity, arm=recipe['execution_arm'],
        stop_after=args.stop_after, resume_checkpoint=args.checkpoint if resume else None,
        allow_microbatch_change=resume and getattr(args, 'microbatch_cases', None) is not None)


def status(args):
    _paths()
    from unified_interaction_train import _command_status
    return _command_status(args)


def clean_stop(args):
    _paths()
    from unified_interaction_train import _command_clean_stop
    return _command_clean_stop(args)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='bind a new full-data recipe, without an optimizer')
    prep.add_argument('--task', choices=('thermal', 'wind'), required=True)
    prep.add_argument('--recipe', required=True)
    prep.add_argument('--arm', choices=('full_detail', 'adaptive_detail'), required=True)
    prep.add_argument('--metadata-only', action='store_true', help='do not fit transforms or construct a model')
    prep.add_argument('--flow-checkpoint')
    prep.add_argument('--data-root')
    prep.add_argument('--dataset-path')
    prep.add_argument('--validation-every', type=int, default=500)
    prep.add_argument('--validation-epochs', type=int, nargs='*', default=[100])
    prep.add_argument('--checkpoint-epochs', type=int, nargs='+', default=[100, 500, 1000, 2000, 2500, 5000])
    prep.add_argument('--latest-every', type=int, default=100)
    prep.add_argument('--curve-every', type=int, default=100)
    prep.set_defaults(handler=prepare)
    dry = commands.add_parser('dry-run', help='validate bindings/parser only; no full-data optimizer or step')
    dry.add_argument('--recipe', required=True)
    dry.set_defaults(handler=dry_run)
    for name in ('start', 'resume'):
        command = commands.add_parser(name, help='manually fit or exactly resume the chosen full-data candidate')
        command.add_argument('--recipe', required=True)
        command.add_argument('--run-dir', required=True)
        command.add_argument('--device', default='cuda:0')
        command.add_argument('--stop-after', type=int, default=5000)
        command.add_argument('--startup-benchmark', action='store_true')
        if name == 'resume':
            command.add_argument('--checkpoint', required=True)
            command.add_argument('--microbatch-cases', type=int,
                help='explicit microbatch-only continuation amendment; effective batch and schedule stay sealed')
        command.set_defaults(handler=(lambda args: train(args, resume=True)) if name == 'resume' else train)
    stat = commands.add_parser('status', help='read existing shared-engine receipts')
    stat.add_argument('--run-dir', required=True)
    stat.set_defaults(handler=status)
    stop = commands.add_parser('clean-stop', help='request an acknowledged stop after a complete epoch')
    stop.add_argument('--run-dir', required=True)
    stop.add_argument('--note', default='manual operator requested clean stop')
    stop.set_defaults(handler=clean_stop)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result if isinstance(result, int) else 0


if __name__ == '__main__':
    raise SystemExit(main())

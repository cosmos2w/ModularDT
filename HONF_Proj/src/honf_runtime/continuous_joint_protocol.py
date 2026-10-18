"""Separately sealed Round10 continuous trajectories from literal dense e3000."""
from __future__ import annotations
import copy
from pathlib import Path
from .joint_native_protocol import canonical_sha256,file_sha256,_source,validate_source

CONTINUOUS_FAMILY='continuous_joint_native_v1'
CONTINUOUS_WORKFLOW='continuous_joint_train_v1'
CONTINUOUS_PROTOCOL='round10_continuous_joint_v1'
UNIFIED_FAMILY='unified_continuous_joint_native_v1'
UNIFIED_PROTOCOL='round10_unified_joint_v1'
PARENT_SHA={'thermal':'4afbefa80d8a9aaa1ed3deba5cc7032b914e85f9fc20484e81b39a622e2d904c',
            'wind':'f367756a2292f04342991d2b9db788ae3bbbb048a9bacf67840f21910e914fbf'}


def make_continuous_joint_recipe(source_checkpoint,*,dataset,arm,resource_reservation,model_options=None,calibration_receipt=None,admission_receipt=None,context_initialization_receipt=None):
    path=Path(source_checkpoint).resolve(strict=True);payload=_source(path)
    base=validate_source(payload,arm='H9-ME',dataset=dataset)
    identity=payload['experiment_identity']
    value=dict(schema='round10_continuous_joint_recipe_v1',family=CONTINUOUS_FAMILY,workflow=CONTINUOUS_WORKFLOW,
        protocol=CONTINUOUS_PROTOCOL,dataset=dataset,arm=arm,source_checkpoint=str(path),source_checkpoint_sha256=file_sha256(path),
        source_epoch=3000,source_recipe_sha256=identity['recipe_sha256'],
        source_provider_identity_sha256=canonical_sha256(identity['provider_identity']),base_recipe=base,
        microbatch_cases=8,effective_cases=48 if dataset=='thermal' else 16,final_epoch=5000,
        sampling_arm='A2-D',sampling_version='case_epoch_v1',model_options=dict(model_options or {}),
        calibration=None if calibration_receipt is None else {'path':str(Path(calibration_receipt).resolve(strict=True)),
        'sha256':file_sha256(calibration_receipt)},resource_reservation=str(Path(resource_reservation).resolve()),
        schedule_extension='retain_exact_source_terminal_lr_through_e5000',
        execution=dict(checkpoint_every=100,curve_every=10,progress_enabled=True))
    if arm in ('U10-D','U10-H','U10-S'):
        from honf_forward_core.continuous_joint.unified import local_input_adapter
        contract,_=local_input_adapter(dataset)
        value.update(schema='round10_unified_joint_recipe_v1',family=UNIFIED_FAMILY,protocol=UNIFIED_PROTOCOL,
            input_capability='local_records_context_v1',input_feature_contract=contract(),
            unified_admission=None if admission_receipt is None else dict(
                path=str(Path(admission_receipt).resolve(strict=True)),sha256=file_sha256(admission_receipt)))
        if context_initialization_receipt is not None:
            value['context_initialization']=dict(path=str(Path(context_initialization_receipt).resolve(strict=True)),
                sha256=file_sha256(context_initialization_receipt))
    elif admission_receipt is not None or context_initialization_receipt is not None:
        raise ValueError('Unified admission applies only to the matched U triplet.')
    return validate_continuous_joint_recipe(value,verify_source=True)


def validate_continuous_joint_recipe(value,*,verify_source=False):
    recipe=copy.deepcopy(dict(value))
    from .joint_native_protocol import validate_joint_native_recipe
    unified=recipe.get('arm') in ('U10-D','U10-H','U10-S')
    legacy=copy.deepcopy(recipe)
    if unified:
        for key in ('input_capability','input_feature_contract','unified_admission','context_initialization'):
            legacy.pop(key,None)
    legacy.update(schema='round9_joint_native_recipe_v1',family='joint_native_hypergraph_v1',workflow='joint_native_train_v1',
        protocol='round9_joint_native_v1',arm='H9-ME',model_options={})
    validate_joint_native_recipe(legacy,verify_source=False)
    if (recipe['schema']!=('round10_unified_joint_recipe_v1' if unified else 'round10_continuous_joint_recipe_v1') or recipe['family']!=(UNIFIED_FAMILY if unified else CONTINUOUS_FAMILY)
        or recipe['workflow']!=CONTINUOUS_WORKFLOW or recipe['protocol']!=(UNIFIED_PROTOCOL if unified else CONTINUOUS_PROTOCOL)
        or recipe['arm'] not in ('C10-H','C10-S','U10-D','U10-H','U10-S') or recipe['source_epoch']!=3000
        or recipe['final_epoch']!=5000 or recipe['microbatch_cases']!=8
        or recipe['source_checkpoint_sha256']!=PARENT_SHA[recipe['dataset']]):
        raise ValueError('Continuous trajectory identity, common parent or full horizon differs.')
    if set(recipe['model_options'])-{'native_backend','checkpoint_pipeline','query_tile_size','entmax_temperature'}:
        raise ValueError('Continuous execution options include an undeclared semantic change.')
    if unified:
        from honf_forward_core.continuous_joint.unified import local_input_adapter
        contract,_=local_input_adapter(recipe['dataset'])
        if recipe.get('input_capability')!='local_records_context_v1' or recipe.get('input_feature_contract')!=contract():
            raise ValueError('Unified named local-input feature capability differs.')
        if 'unified_admission' not in recipe:
            raise ValueError('Unified recipe must declare its pending or sealed admission.')
        context=recipe.get('context_initialization')
        if context is not None and (recipe['arm']!='U10-S' or set(context)!={'path','sha256'}):
            raise ValueError('Shared contextual initialization must bind U-S to its matched input-only U-H calibration.')
    if verify_source:
        if file_sha256(recipe['source_checkpoint'])!=recipe['source_checkpoint_sha256']:raise ValueError('Continuous parent bytes changed.')
        payload=_source(recipe['source_checkpoint']);identity=payload['experiment_identity']
        if (validate_source(payload,arm='H9-ME',dataset=recipe['dataset'])!=recipe['base_recipe']
            or identity['recipe_sha256']!=recipe['source_recipe_sha256']
            or canonical_sha256(identity['provider_identity'])!=recipe['source_provider_identity_sha256']):
            raise ValueError('Continuous parent sampling/provider binding changed.')
        if recipe['calibration'] is not None and file_sha256(recipe['calibration']['path'])!=recipe['calibration']['sha256']:
            raise ValueError('Continuous calibration bytes changed.')
        if unified and recipe.get('context_initialization') is not None:
            binding=recipe['context_initialization']
            if file_sha256(binding['path'])!=binding['sha256']:
                raise ValueError('Shared contextual initialization receipt changed.')
        if unified and recipe['unified_admission'] is not None:
            admission=recipe['unified_admission']
            if file_sha256(admission['path'])!=admission['sha256']:
                raise ValueError('Unified admission receipt bytes changed.')
    return recipe


def continuous_joint_child_declaration(recipe):
    value=validate_continuous_joint_recipe(recipe)
    return dict(schema='round10_continuous_joint_child_v1',recipe_sha256=canonical_sha256(value),
        source_checkpoint_sha256=value['source_checkpoint_sha256'],source_epoch=3000,arm=value['arm'],final_epoch=5000)


def validate_continuous_joint_stop(recipe,stop_after):
    if type(stop_after) is not int or not 3000<stop_after<=5000 or stop_after%100:
        raise ValueError('Continuous stop must preserve a declared hundred-epoch boundary through e5000.')


def validate_unified_admission(recipe):
    """Scientific fit admission; preparation remains CPU-only and ungated."""
    import json,math
    value=validate_continuous_joint_recipe(recipe,verify_source=True)
    if value['arm'] not in ('U10-D','U10-H','U10-S'):return None
    binding=value['unified_admission']
    if binding is None:raise ValueError('Unified triplet awaits the declared endpoint/safety/resource admission.')
    receipt=json.loads(Path(binding['path']).read_text())
    expected=dict(schema='round10_unified_triplet_admission_v1',dataset=value['dataset'],status='admitted',
        track_b_epoch=5000,source_checkpoint_sha256=value['source_checkpoint_sha256'],
        source_target_isolation_passed=True,continuous_value_gradient_passed=True,
        native_law_passed=True,fresh_state_passed=True,track_b_catastrophic_collapse=False,
        complete_triplet_reserved=True,resource_reservation_sha256=file_sha256(value['resource_reservation']))
    if any(receipt.get(key)!=item for key,item in expected.items()):
        raise ValueError('Unified endpoint/safety/complete-triplet resource gate was not satisfied.')
    if not isinstance(receipt.get('track_b_endpoint_checkpoint_sha256'),str) or len(receipt['track_b_endpoint_checkpoint_sha256'])!=64:
        raise ValueError('Unified admission must identify the literal Track-B endpoint.')
    endpoint=Path(receipt.get('track_b_endpoint_checkpoint',''))
    if not endpoint.is_file() or file_sha256(endpoint)!=receipt['track_b_endpoint_checkpoint_sha256']:
        raise ValueError('Unified admission literal Track-B endpoint bytes are unavailable or differ.')
    saved=_source(endpoint);endpoint_identity=saved.get('experiment_identity',{})
    endpoint_recipe=endpoint_identity.get('recipe',{})
    if (saved.get('epoch')!=5000 or endpoint_identity.get('family')!=CONTINUOUS_FAMILY
        or endpoint_recipe.get('arm')!='C10-H' or endpoint_recipe.get('dataset')!=value['dataset']
        or endpoint_recipe.get('source_checkpoint_sha256')!=value['source_checkpoint_sha256']):
        raise ValueError('Unified admission endpoint is not the matched completed C-H trajectory.')
    times=receipt.get('native_epoch_seconds',{})
    if set(times)!={'U10-D','U10-H','U10-S'} or any(type(t) not in (int,float) or not math.isfinite(t) or t<=0 for t in times.values()):
        raise ValueError('Unified complete native epoch timings are missing.')
    forecast=receipt.get('complete_triplet_seconds');remaining=receipt.get('reserved_remaining_process_seconds')
    if (type(forecast) not in (int,float) or type(remaining) not in (int,float)
        or not math.isfinite(forecast) or not math.isfinite(remaining)
        or forecast<2000*sum(times.values()) or remaining<forecast):
        raise ValueError('Unified full-horizon triplet forecast exceeds its declared reservation.')
    return receipt

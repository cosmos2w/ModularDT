"""Inference from a saved joint-field model using only its native data binding."""
from __future__ import annotations

import copy
import torch

from .r13_6_member_routing_task import _build_thermal_member_routing_task
from .r13_8_anchored_task import (
    _VARIANTS, DETAIL_CONFIG_SCHEMA, DETAIL_CORE_SCHEMA, DETAIL_SOURCE_KIND,
)
from .thermal_joint_response_task import ThermalPairedFormalNativeTask


def build_native_checkpoint_inference(saved, *, validation_case_ids, device='cpu'):
    """Strictly restore native predictions without constructing a response corpus.

    The evaluation panel is a separate inference request. The checkpoint's
    architecture, source recipe and saved native normalization are preserved;
    this function neither restores an edited-scene provider nor trains a model.
    """
    identity = saved['experiment_identity']
    source_recipe = identity['recipe']
    native = identity['provider_identity']
    if (source_recipe.get('training_profile') != 'state_response'
            or source_recipe.get('dataset', {}).get('name') != 'thermal'
            or source_recipe.get('member_routing', {}).get('model', {}).get('schema') != DETAIL_CORE_SCHEMA):
        raise ValueError('Expected a saved native joint-field relation-anchored Thermal model.')
    panel = tuple(map(str, validation_case_ids))
    if not panel or len(set(panel)) != len(panel):
        raise ValueError('The inference panel must contain unique original case IDs.')
    if 'adapter_state' not in native or 'normalization_source' not in native.get('source_binding', {}):
        raise ValueError('The checkpoint lacks its saved native normalization.')
    recipe = copy.deepcopy(source_recipe)
    recipe['evaluation']['validation_case_ids'] = list(panel)
    recipe['evaluation']['validation_population'] = 'canonical_VAL89'
    # Retain the physical native-port extraction settings, with no corpus path,
    # ROI profile, response scales or calibration binding in the working recipe.
    ports = copy.deepcopy(recipe.get('thermal_joint_response', {}).get('native_ports', {}))
    recipe['thermal_joint_response'] = {'native_ports': ports}
    recipe['state_response'] = {'enabled': False}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    arm = recipe['member_routing']['arm']
    model, provider = _build_thermal_member_routing_task(
        recipe, arm=arm, device=device, task_type=ThermalPairedFormalNativeTask,
        variants={DETAIL_CORE_SCHEMA: {**_VARIANTS[DETAIL_CORE_SCHEMA], 'arm': arm}},
        section_schema=DETAIL_CONFIG_SCHEMA, source_kind=DETAIL_SOURCE_KIND,
        lineage='Saved native-field inference', required_address_mode=None,
        saved_adapter_state=native['adapter_state'],
        saved_normalization_source=native['source_binding']['normalization_source'],
    )
    provider.configure_native()
    if set(provider.validation_case_ids) != set(panel):
        raise ValueError('The inference panel is outside the native validation population.')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.to(device).eval()
    return model, provider

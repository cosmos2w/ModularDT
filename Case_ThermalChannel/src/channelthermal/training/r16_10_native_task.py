"""Historical R16-10 native provider and literal scratch recipe."""
from __future__ import annotations

import numpy as np

from ..data.datasets import H5Normalizer
from .joint_task import _read_selected_cases
from .r13_6_member_routing_task import _build_thermal_member_routing_task
from .r13_8_anchored_task import (
    ARM_DETAIL, DETAIL_CONFIG_SCHEMA, DETAIL_CORE_SCHEMA, DETAIL_SOURCE_KIND, _VARIANTS,
)
from .thermal_native_port_task import NativePortTargets, ThermalNativePortTask, _module_mean


def stratified_validation_ids(cases, seed=20261108):
    """Select nine source-ordered examples across native module-count strata."""
    if len(cases) != 89:
        raise ValueError('Expected canonical VAL89 before selecting the validation panel.')
    rng = np.random.default_rng(seed)
    groups = {}
    for i, case in enumerate(cases):
        count = int((np.asarray(case['structure']['module_present']) > .5).sum())
        groups.setdefault(count, []).append(i)
    queues = {k: list(rng.permutation(v)) for k, v in sorted(groups.items())}
    chosen = []
    while len(chosen) < 9:
        for key in queues:
            if queues[key] and len(chosen) < 9:
                chosen.append(queues[key].pop())
    return tuple(str(cases[i]['case_id']) for i in sorted(chosen))


class ThermalR1610NativeTask(ThermalNativePortTask):
    """R16-10 keeps its saved optimizer schedule and 64-angle native objective."""

    def _schedule_payload(self):
        return dict(peak_lr=3e-4, warmup_start_lr=3e-4, warmup_epochs=0,
                    hold_through_epoch=1000, total_epochs=2000, final_lr=3e-5)


def build_thermal_r16_10_native_task(recipe, *, arm=ARM_DETAIL, device='cpu', task_type=ThermalR1610NativeTask,
                                     saved_adapter_state=None, saved_normalization_source=None,
                                     source_kind=DETAIL_SOURCE_KIND, lineage='R16-10 scratch'):
    training = recipe['training']
    if (int(training['effective_batch_size']) != 48 or not 1 <= int(training['microbatch_cases']) <= 24
            or int(recipe['member_routing']['model']['query_tile_size']) != 128):
        raise ValueError('R16-10 requires effective batch 48, microbatch at most 24 and training tile 128.')
    model, provider = _build_thermal_member_routing_task(recipe, arm=arm, device=device, task_type=task_type,
        variants={DETAIL_CORE_SCHEMA: {**_VARIANTS[DETAIL_CORE_SCHEMA], 'arm': arm}},
        section_schema=DETAIL_CONFIG_SCHEMA, source_kind=source_kind, lineage=lineage, required_address_mode=None,
        saved_adapter_state=saved_adapter_state, saved_normalization_source=saved_normalization_source)
    val_ids = [case_id for case_id in provider.catalog_case_ids
               if provider.case_split_by_id[case_id] == 'test' and case_id != '0273']
    normalizer = H5Normalizer({k: np.asarray(v, dtype=np.float32) for k, v in model.normalization_stats.items()})
    cases = _read_selected_cases(provider.data_path, val_ids, split='test', normalizer=normalizer, include_grid=False)
    selected = set(stratified_validation_ids(cases))
    provider.validation_cases = tuple(c for c in cases if str(c['case_id']) in selected)
    provider.validation_case_ids = tuple(str(c['case_id']) for c in provider.validation_cases)
    if tuple(recipe['evaluation']['validation_case_ids']) != provider.validation_case_ids:
        raise ValueError('R16-10 configured validation IDs disagree with the sampled canonical panel.')
    provider._validation_index_by_id = {v: i for i, v in enumerate(provider.validation_case_ids)}
    provider.validation_scope = 'input_stratified_seed20261108_VAL9_of_89'
    provider.configure_native()
    return model, provider

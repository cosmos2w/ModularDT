"""Fresh, frozen native Thermal inference for the joint-edit study."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .joint_edit_sampling import extract_organization
from .relation_anchored import ThermalDesignEvaluator
from ..thermal_coupling import local_response_from_outside, outside_from_native_output
from ..training.thermal_port_response_task import load_thermal_port_response_checkpoint


def _array(value):
    return value.detach().cpu().numpy().copy() if torch.is_tensor(value) else np.asarray(value).copy()


class FrozenJointEditPredictor:
    """One fixed P64 checkpoint and its declared frozen local asset.

    ``evaluate`` makes one ordinary preparation and one native decode for each
    design. Local diagnostics reuse the outside64 values from that decode.
    Timing includes device synchronization; no reference labels are read.
    """

    def __init__(self, checkpoint, device):
        started = time.perf_counter()
        self.checkpoint = str(checkpoint)
        self.device = torch.device(device)
        self.model, self.local, self.payload = load_thermal_port_response_checkpoint(checkpoint, device=device)
        spec = self.payload['experiment_identity']['recipe']['thermal_port_response']
        if spec['arm'] != 'P64' or int(self.payload['epoch']) != 100 or int(spec['source_epoch']) != 5000:
            raise ValueError('Joint-edit predictor requires P64 source epoch 5000, new epoch 100.')
        self.model.eval()
        self.local.eval()
        for module in (self.model, self.local):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.temperature_scale = float(self.model.internal_temperature_std[0].detach().cpu())
        self._initial_state = tuple(
            {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
            for module in (self.model, self.local)
        )
        self._sync()
        self.load_seconds = time.perf_counter() - started

    def unchanged(self):
        """Check both frozen components against their CPU state at loading."""
        for module, original in zip((self.model, self.local), self._initial_state):
            current = module.state_dict()
            if current.keys() != original.keys() or any(
                not torch.equal(value.detach().cpu(), original[name]) for name, value in current.items()
            ):
                return False
        return True

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def evaluate(self, record, centers, save_path=None, full=False):
        """Return physical scalars, actual organization, arrays, work and timing.

        Centers follow original active module order. ``full`` changes only the
        native read mode and is intended for the small diagnostic replay.
        ``save_path`` is optional and should be used for retained nominees only.
        """
        self._sync()
        started = time.perf_counter()
        evaluator = ThermalDesignEvaluator(self.model, record)
        tensor = evaluator._tensor(np.asarray(centers, dtype=np.float32))
        times = {}
        original_prepare = self.model.prepare_native
        original_apply = self.model.apply_native_with_work_counts

        def measured(name, fn):
            def call(*args, **kwargs):
                self._sync()
                start = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    self._sync()
                    times[name] = time.perf_counter() - start
            return call

        # Evaluate through the maintained case functional while measuring its
        # exact preparation and native decode/extraction operations.
        self.model.prepare_native = measured('preparation_seconds', original_prepare)
        self.model.apply_native_with_work_counts = measured('native_seconds', original_apply)
        try:
            with torch.inference_mode():
                evaluation = evaluator.evaluate_design(tensor, read_mode='full' if full else 'restricted')
                prepared = evaluation.prepared
                local_start = time.perf_counter()
                self._sync()
                outside = outside_from_native_output(self.model, prepared, evaluation.core_output)
                heat = prepared.structure['heat_powers']
                local = local_response_from_outside(self.model, self.local, prepared, heat, evaluator.local, outside)
                self._sync()
                times['local_seconds'] = time.perf_counter() - local_start
        finally:
            self.model.prepare_native = original_prepare
            self.model.apply_native_with_work_counts = original_apply

        scene = prepared.scene_batch
        module_mask = _array(scene.module_mask[0]).astype(bool)
        environment_mask = _array(scene.environment_mask[0]).astype(bool)
        mids = _array(scene.module_ids[0])[module_mask]
        eids = _array(scene.environment_ids[0])[environment_mask]
        # Native typed string IDs are metadata; the adapter uses slot-local
        # integer correspondence keys for its tensors and packed directory.
        if len(mids) != len(evaluator.ids) or not np.array_equal(mids, np.arange(len(mids))):
            raise ValueError('Prepared module slots do not match the original record order.')
        organization = extract_organization(prepared, mids, eids)
        arrays = {'module_ids': mids, 'original_module_ids': np.asarray(evaluator.ids, dtype=str), 'environment_ids': eids,
                  'module_coordinates': _array(scene.module_coordinates[0])[module_mask],
                  'environment_coordinates': _array(scene.environment_coordinates[0])[environment_mask],
                  'environment_measure': _array(scene.environment_measure[0])[environment_mask],
                  'outside64': _array(outside), 'fluid_coordinates': _array(evaluator.fluid),
                  'local_coordinates': _array(evaluator.local)}
        arrays.update({f'native_{key}': _array(value) for key, value in evaluation.fields.items() if torch.is_tensor(value)})
        arrays.update({f'local_{key}': _array(value) for key, value in local.items() if torch.is_tensor(value)})
        for role, stencil in prepared.stencils.items():
            for field in ('indices', 'weights', 'valid'):
                arrays[f'stencil_{role}_{field}'] = _array(getattr(stencil, field))
        work = {'preparations': 1, 'local_calls': 1,
                **{f'prepare_{k}': int(v) for k, v in prepared.organization.work_counts.items()},
                **{f'native_{k}': int(v) for k, v in evaluation.work.items()}}
        result = {'objective': float(evaluation.objective), 'pressure': float(evaluation.pressure_drop),
                  'sampled_max': float(evaluation.sampled_max), 'valid_material_count': int(evaluation.valid_material_count),
                  'organization': organization, 'module_ids': mids, 'environment_ids': eids,
                  'original_module_ids': arrays['original_module_ids'],
                  'environment_coordinates': arrays['environment_coordinates'],
                  'environment_measure': arrays['environment_measure'], 'arrays': arrays,
                  'work': work, 'timing': times, 'read_mode': 'full' if full else 'restricted'}
        if save_path is not None:
            np.savez_compressed(Path(save_path), **arrays)
        self._sync()
        times['total_seconds'] = time.perf_counter() - started
        return result

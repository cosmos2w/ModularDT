"""Explicit bounded fixed-coefficient interventions with owner/version checks."""
from __future__ import annotations
from dataclasses import dataclass,field
import torch
from ..native_retrieval.contracts import tensor_versions

@dataclass(frozen=True)
class FrozenContinuousReadPlan:
    owner: str
    parameter_versions: tuple
    buffer_versions: tuple
    frame: object
    normalizer: object
    module_ids: torch.Tensor
    environment_ids: torch.Tensor
    query_ids: torch.Tensor
    module_present: torch.Tensor
    environment_present: torch.Tensor
    query_present: torch.Tensor
    gates: torch.Tensor
    environment_coefficients: torch.Tensor
    module_coefficients: torch.Tensor
    _versions: tuple=field(init=False,repr=False)
    def __post_init__(self):
        object.__setattr__(self,'_versions',tensor_versions(*self.tensors()))
    def tensors(self):
        return (self.module_ids,self.environment_ids,self.query_ids,self.module_present,self.environment_present,
                self.query_present,self.gates,self.environment_coefficients,self.module_coefficients)
    def assert_current(self,core,organization,query):
        if tensor_versions(*self.tensors())!=self._versions:raise ValueError('Frozen continuous coefficients changed after capture.')
        if (core._owner_token!=self.owner or core._parameter_version_stamp()!=self.parameter_versions
            or core._buffer_version_stamp()!=self.buffer_versions):raise ValueError('Frozen continuous plan owner/weights changed.')
        scene=organization.scene
        if scene.coordinate_frame_key!=self.frame or scene.normalizer_key!=self.normalizer:
            raise ValueError('Frozen continuous plan frame/normalizer differs.')
        qmask=query.query_mask if query.query_mask is not None else torch.ones_like(query.query_ids,dtype=torch.bool)
        actual=(scene.module_ids,scene.environment_ids,query.query_ids,scene.module_mask,scene.environment_mask,qmask)
        if any(a is None or not torch.equal(a,b) for a,b in zip(actual,self.tensors()[:6])):
            raise ValueError('Frozen continuous plan native IDs/validity differ; use exact bounded correspondence.')

"""Continuous seed composition and direct typed access without winner floors."""
from __future__ import annotations
import torch
from torch import nn
from ..joint_native.routing import JointDispatchParameters, DirectGateParameters, typed_query_logits
from .incidence import ContinuousIncidencePolicy
from .primitives import positive_quadratic_gate, entmax15


class ContinuousRouter(nn.Module):
    def __init__(self,heads,*,mode='hypergraph',temperature=1.):
        super().__init__()
        if mode not in ('hypergraph','direct','dense') or temperature<=0:
            raise ValueError('Unknown continuous route or invalid fixed temperature.')
        self.mode=mode
        self.incidence=ContinuousIncidencePolicy(heads)
        self.dispatch_parameters=JointDispatchParameters(heads)
        self.direct_parameters=DirectGateParameters(heads)
        self.register_buffer('entmax_temperature',torch.tensor(float(temperature)))
        with torch.no_grad():
            self.direct_parameters.module.bias.fill_(7.)
            self.direct_parameters.environment.bias.fill_(7.)

    def forward(self, raw_m, raw_e, records, local_envelope, query_mask, incidence=None):
        eligible = records.eligible[:,None] & query_mask[...,None]
        if self.mode=='dense':
            access=eligible.to(raw_m);seeds=None
        elif self.mode=='direct':
            am=eligible[...,:records.module_count];ae=eligible[...,records.module_count:]
            access=torch.cat((positive_quadratic_gate(self.direct_parameters.module(raw_m,am))*am,
                              positive_quadratic_gate(self.direct_parameters.environment(raw_e,ae))*ae),-1)
            seeds=None
        else:
            if incidence is None:
                raise ValueError('Hypergraph retrieval requires stage-bound native incidence.')
            allowed=records.present[:,None] & query_mask[...,None]
            logits=typed_query_logits(raw_m,raw_e,records,self.dispatch_parameters,query_mask)
            seeds=entmax15(logits/self.entmax_temperature,allowed)
            access=torch.bmm(seeds,incidence)*eligible
        m=records.module_count
        gm=local_envelope+(1-local_envelope)*access[...,:m]
        return torch.cat((gm,access[...,m:]),-1)*eligible,seeds

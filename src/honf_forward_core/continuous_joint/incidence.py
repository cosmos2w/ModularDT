"""Typed native incidence; retrieval identity is separate from contextual edges."""
from __future__ import annotations
import torch
from torch import nn
from ..joint_native.routing import ScoreCalibration
from ..joint_native.graph import _validated_raw
from .primitives import positive_quadratic_gate


class ContinuousIncidencePolicy(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.routes = nn.ModuleDict({name:ScoreCalibration(heads) for name in ('mm','me','em','ee')})
        with torch.no_grad():
            for calibration in self.routes.values():
                calibration.bias.fill_(7.)

    def forward(self, raw_logits, allowed, route_name, stage_identity=None,
                destination_ids=None, donor_ids=None, donor_measure=None):
        return positive_quadratic_gate(self.routes[route_name](raw_logits,allowed)) * allowed

    def retrieval(self, capture, records):
        records.assert_current()
        m,n = records.module_count,records.ids.shape[1]
        slices = {'m':slice(0,m),'e':slice(m,n)}
        blocks = {}
        for name in ('mm','me','em','ee'):
            destination,donor = slices[name[0]],slices[name[1]]
            raw = _validated_raw(capture,records,name,destination,donor)
            allowed = records.present[:,destination,None] & records.eligible[:,None,donor]
            diagonal = torch.zeros_like(allowed)
            if name[0] == name[1]:
                diagonal = torch.eye(allowed.shape[-1],device=allowed.device,dtype=torch.bool)[None] & allowed
            # MM capture intentionally excludes self. Never calibrate its -inf.
            gate = self(raw,allowed & ~diagonal if name=='mm' else allowed,name)
            blocks[name] = torch.where(diagonal,torch.ones_like(gate),gate)
        return torch.cat((torch.cat((blocks['mm'],blocks['me']),-1),
                          torch.cat((blocks['em'],blocks['ee']),-1)),1), blocks

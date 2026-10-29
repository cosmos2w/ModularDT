"""Selected-TRAIN heat/geometry finite changes for a general joint predictor."""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import torch

from honf_runtime.thermal_geometry_supervision import (
    GeometrySupervision, GeometryResponsePair, _component_names, _pair_panel_loss,
)
from ..interaction_evidence.response_atlas import load_response_atlas_stencil


class JointResponseSupervision:
    """One bounded pair, alternating heat and single geometry, no optimizer."""
    def __init__(self, atlas_directory, family_ids, normalization_stats):
        self.family_ids = tuple(family_ids)
        self.geometry = GeometrySupervision(atlas_directory, family_ids=self.family_ids,
            fluid_queries=256, local_queries_per_module=32, ntheta=16)
        self.heat_pairs=[]
        root=Path(atlas_directory)
        if (root/"families").is_dir():
            root=root/"families"
        for family in self.family_ids:
            stencil, _ = load_response_atlas_stencil(root/f"train_{family}_responses.npz")
            for label in ("heat_transfer_plus","heat_transfer_minus"):
                roles={role:stencil.finite_change(label,role) for role in ("fluid_fields","interface","solid_temperature")}
                self.heat_pairs.append(GeometryResponsePair(family,stencil.physical_family_id,label,
                    tuple(stencil.baseline.design.active_module_ids),stencil.baseline,
                    stencil.variants[label],roles,{"family_id":family,"label":label}))
        native = list(np.asarray(normalization_stats["field_std_by_channel"]).reshape(-1))
        native += list(np.asarray(normalization_stats["interface_targets_std"]).reshape(-1))
        native += [float(np.asarray(normalization_stats["internal_temperature_std"]).reshape(-1)[0])]
        names=_component_names(self.heat_pairs[0].roles)
        self.floors={name:max(1e-8,1e-3*float(scale)) for name,scale in zip(names,native,strict=True)}
        self.geometry.scales={n:max(v,self.floors[n]) for n,v in self.geometry.scales.items()}
        self.heat_scales={}
        for name, floor in self.floors.items():
            role, channel_name=name.split("/",1)
            squares=[]
            for pair in self.heat_pairs:
                block=pair.roles[role]
                channel=block.channel_names.index(channel_name)
                mask=block.valid_mask[:,channel]&np.isfinite(block.delta[:,channel])
                squares.append(float(np.mean(block.delta[mask,channel]**2)))
            self.heat_scales[name]=max(math.sqrt(float(np.mean(squares))),floor)
        self.metadata={"family_ids":list(self.family_ids),"geometry_scales":dict(self.geometry.scales),
            "heat_scales":dict(self.heat_scales),"native_scale_floor_fraction":1e-3,
            "near_null_policy":"absolute errors; TRAIN native field-scale floor", "fluid_queries":256,
            "local_per_module":32,"ntheta":16,"cadence":"one finite pair per four primary updates, alternating heat/single geometry"}

    def loss(self, model, index):
        if index%2:
            loss,detail=self.geometry.loss(model,index//2)
            return loss,{"kind":"single_geometry",**detail}
        pair=self.heat_pairs[(index//2)%len(self.heat_pairs)]
        parameter=next(model.parameters())
        modes={module:module.training for module in model.modules()}
        model.eval()
        try:
            loss,_,_,detail=_pair_panel_loss(model,pair,self.heat_scales,schedule_index=index,
                device=parameter.device,dtype=parameter.dtype,fluid_queries=256,
                local_queries_per_module=32,ntheta=16,fluid_chunk_size=256,seed=0)
        finally:
            for module,mode in modes.items():
                module.training=mode
        return loss,{"kind":"heat",**detail}

    def sampled_group_loss(self, model, index):
        """Same finite pair and receivers, with sampled adaptive frontiers.

        Return joint endpoint log probability so the caller can add the
        score-function term; continuous response derivatives remain ordinary.
        """
        if index % 2:
            pair = self.geometry._selected_pairs(index // 2, False)[0]
            scales = self.geometry.scales
            schedule_index = index // 2
            kind = "single_geometry"
        else:
            pair = self.heat_pairs[(index // 2) % len(self.heat_pairs)]
            scales = self.heat_scales
            schedule_index = index
            kind = "heat"
        parameter = next(model.parameters())
        logps = []
        loss, _, _, detail = _pair_panel_loss(
            model, pair, scales, schedule_index=schedule_index,
            device=parameter.device, dtype=parameter.dtype, fluid_queries=256,
            local_queries_per_module=32, ntheta=16, fluid_chunk_size=256, seed=0,
            frontier_log_probabilities=logps,
        )
        return loss, {"kind": kind, **detail}, torch.stack(logps).sum()

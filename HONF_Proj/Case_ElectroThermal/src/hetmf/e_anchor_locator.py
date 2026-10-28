"""Case-owned coverage of the Standard HET-MF unit-square material cells."""
from dataclasses import replace
import torch


def attach_e_coverage(query, scene, mode):
    if mode not in ('e_anchor', 'local_e_cover'):
        raise ValueError('unknown HET-MF locator mode')
    if scene.environment_coordinates.shape[1:] != (144, 2):
        raise ValueError('HET-MF coverage requires 144 material cells')
    all_indices, all_weights = [], []
    for b in range(scene.batch_size):
        xy = scene.environment_coordinates[b]
        cell = torch.floor(xy * 12).long().clamp(0, 11)
        linear = cell[:, 1] * 12 + cell[:, 0]
        expected = (cell.to(xy.dtype) + .5) / 12
        if not bool(scene.environment_mask[b].all()) or not bool(torch.allclose(xy, expected, atol=2e-6, rtol=0)) or torch.unique(linear).numel() != 144:
            raise ValueError('HET-MF E coordinates must form the complete unit-square 12x12 lattice')
        slots = torch.empty(144, device=xy.device, dtype=torch.long)
        slots.scatter_(0, linear, torch.arange(144, device=xy.device))
        slots = slots.reshape(12, 12)
        q = query.coordinates[b]
        valid = query.query_mask[b] if query.query_mask is not None else torch.ones(q.shape[0], device=q.device, dtype=torch.bool)
        if bool((valid[:, None] & ((q < 0) | (q > 1))).any()):
            raise ValueError('HET-MF queries must lie in the unit square')
        q = q.clamp(0, 1)
        if mode == 'e_anchor':
            ij = torch.floor(q * 12).long().clamp(0, 11)
            idx = slots[ij[:, 1], ij[:, 0], None]
            w = q.new_ones((len(q), 1))
        else:
            pos = (q * 12 - .5).clamp(0, 11)
            lo = torch.floor(pos.detach()).long().clamp(0, 10)
            t = (pos - lo).clamp(0, 1)
            s = 3*t.square() - 2*t.pow(3)
            x,y = lo[:, 0],lo[:, 1]
            idx = torch.stack((slots[y,x],slots[y,x+1],slots[y+1,x],slots[y+1,x+1]), -1)
            sx,sy = s[:,0],s[:,1]
            w = torch.stack(((1-sx)*(1-sy),sx*(1-sy),(1-sx)*sy,sx*sy), -1)
            idx = torch.where(w > 0, idx, -1)
        all_indices.append(torch.where(valid[:, None], idx, -1))
        all_weights.append(torch.where(valid[:, None], w, 0))
    return replace(query, anchor_indices=torch.stack(all_indices), anchor_weights=torch.stack(all_weights))

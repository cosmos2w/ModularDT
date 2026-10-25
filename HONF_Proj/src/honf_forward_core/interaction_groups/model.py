"""Interaction-derived E aggregates used in encoding and reused by queries."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from ..content_hypergraph.spatial_features_v2 import coordinate_gamma
from ..contextual_organization.model import ContextualJointOutput, _mlp, _scene_tensors, _stamp_tensor
from .frontier import LocalFrontier


def nullable(logits, allowed=None):
    if allowed is not None:
        logits = logits.masked_fill(~allowed, -torch.inf)
    return (
        torch.softmax(torch.cat((logits, torch.zeros_like(logits[..., :1])), -1), -1)[..., : logits.shape[-1]]
        if logits.shape[-1]
        else logits
    )


class Exchange(nn.Module):
    """Pre-normalized nullable spatial attention without a learned null value."""

    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.d = d
        self.norm = nn.LayerNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.geo = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, h, bias=False))
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))

    def project(self, x):
        x = self.norm(x)
        return self.k(x).reshape(*x.shape[:-1], self.h, self.d // self.h), self.v(x).reshape(
            *x.shape[:-1], self.h, self.d // self.h
        )

    def read(self, x, keys, values, xy, yxy, mass, prior=None, allowed=None):
        q = self.q(self.norm(x)).reshape(*x.shape[:-1], self.h, self.d // self.h)
        delta = xy[..., :, None, :] - yxy[..., None, :, :]
        geo = torch.cat((delta, delta.square().sum(-1, keepdim=True)), -1)
        logits = torch.einsum("...ihd,...jhd->...hij", q, keys) / math.sqrt(self.d / self.h)
        logits = logits + self.geo(geo).movedim(-1, -3) + mass.clamp_min(1.0e-30).log()[..., None, None, :]
        if prior is not None:
            logits = logits + prior
        ok = mass[..., None, None, :] > 0
        if allowed is not None:
            ok = ok & allowed[..., None, :, :]
        weights = nullable(logits, ok)
        msg = torch.einsum("...hij,...jhd->...ihd", weights, values).flatten(-2)
        msg = self.out(msg)
        return msg, weights

    def forward(self, x, y, xy, yxy, mass, allowed=None):
        k, v = self.project(y)
        msg, r = self.read(x, k, v, xy, yxy, mass, allowed=allowed)
        x = x + msg
        return x + self.ff(x), r


@dataclass
class PreparedGroups:
    frontier: object
    keys: torch.Tensor
    values: torch.Tensor
    states: torch.Tensor
    mass: torch.Tensor
    coordinates: torch.Tensor
    B: torch.Tensor
    R: torch.Tensor
    module_states: torch.Tensor
    environment_states: torch.Tensor
    postings: torch.Tensor
    labels: torch.Tensor
    grid_origin: torch.Tensor
    grid_step: torch.Tensor
    work_counts: dict


@dataclass
class PreparedInteractionGroups:
    scene: object
    layers: tuple
    module_states: torch.Tensor
    proposal_keys: torch.Tensor
    frame_origin: torch.Tensor
    frame_scale: torch.Tensor
    model_owner: int
    parameter_versions: tuple
    buffer_versions: tuple
    scene_versions: tuple
    work_counts: dict

    @property
    def final_groups(self):
        return self.layers[-1]

    @property
    def scene_key(self):
        return self.scene.scene_key


class InteractionGroupLayer(nn.Module):
    def __init__(self, d, h, nx, ny, ee=False):
        super().__init__()
        self.mm = Exchange(d, h)
        self.me = Exchange(d, h)
        self.gm = Exchange(d, h)
        self.group_transform = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.frontier = LocalFrontier(d, nx, ny)
        # Allocate extra control weights after common weights to preserve common
        # initialization; core also loads named common tensors across arms.
        self.ee = Exchange(d, h) if ee else None
        self.d, self.h, self.nx, self.ny = d, h, nx, ny

    def forward(
        self, m, e, scene, xy, exy, *, adaptive, epoch, update, seed, layer, training, clamp=False, forced=None
    ):
        present = scene.module_mask
        allowed = present[:, :, None] & present[:, None, :]
        allowed = allowed & ~torch.eye(m.shape[1], device=m.device, dtype=torch.bool)[None]
        mm, _rmm = self.mm(m, m, xy, xy, scene.module_measure / scene.module_measure_reference[:, None], allowed)
        # M1 has a truly empty M->M update, including its FFN.
        m = torch.where((present.sum(-1) > 1)[:, None, None], mm, m) * present[..., None]
        e, R = self.me(
            e,
            m,
            exy,
            xy,
            scene.module_measure / scene.module_measure_reference[:, None],
            scene.environment_mask[:, :, None] & present[:, None, :],
        )
        if self.ee is not None:
            e, _ = self.ee(
                e,
                e,
                exy,
                exy,
                scene.environment_measure / scene.environment_measure_reference[:, None],
                scene.environment_mask[:, :, None] & scene.environment_mask[:, None, :],
            )
        groups = []
        updated = []
        for b in range(len(m)):
            mass = scene.environment_measure[b] / scene.environment_measure_reference[b]
            mass = mass * scene.environment_mask[b]
            # Catalogue includes present zero-mass records: they keep identity
            # and labels but cannot contribute an information message.
            case = scene.scene_key[b] if isinstance(scene.scene_key, (tuple, list)) else scene.scene_key
            case_int = sum((j + 1) * ord(c) for j, c in enumerate(str(case)))
            frontier = self.frontier(
                e[b],
                R[b],
                exy[b],
                mass,
                adaptive=adaptive,
                training=training,
                epoch=epoch,
                seed=(seed, epoch, update, layer, case_int),
                forced=forced,
            )
            labels = frontier.labels
            k = frontier.K
            gm = mass.new_zeros(k).index_add(0, labels, mass)
            denom = gm.clamp_min(1.0e-30)
            pooled = e.new_zeros(k, self.d).index_add(0, labels, e[b] * mass[:, None]) / denom[:, None]
            pos = exy.new_zeros(k, 2).index_add(0, labels, exy[b] * mass[:, None]) / denom[:, None]
            rbar = (
                R.new_zeros(self.h, k, m.shape[1]).index_add(1, labels, R[b] * mass[None, :, None])
                / denom[None, :, None]
            )
            states = self.group_transform(pooled) * (gm > 0)[:, None]
            keys, values = self.gm.project(states)
            msg, B = self.gm.read(
                m[b],
                keys,
                values,
                xy[b],
                pos,
                gm,
                prior=rbar.transpose(-1, -2),
                allowed=present[b, :, None].expand(-1, k),
            )
            if clamp:
                msg = msg * 0
            mb = m[b] + msg
            mb = (mb + self.gm.ff(mb)) * present[b, :, None]
            updated.append(mb)
            assoc = B.mean(0)
            posts = torch.argsort(assoc, dim=-1, descending=True, stable=True)[..., : min(4, k)]
            xs = torch.sort(torch.unique(exy[b, :, 0])).values
            ys = torch.sort(torch.unique(exy[b, :, 1])).values
            if len(xs) != self.nx or len(ys) != self.ny:
                raise ValueError("Declared E catalogue grid dimensions differ from coordinates.")
            # Raw adapter ordering is checked once here; query lookup itself
            # does not search all E rows.
            ix = torch.searchsorted(xs, exy[b, :, 0].contiguous())
            iy = torch.searchsorted(ys, exy[b, :, 1].contiguous())
            cells = torch.empty_like(labels)
            cells[iy * self.nx + ix] = labels
            origin = torch.stack((xs[0], ys[0]))
            step = torch.stack((xs[1] - xs[0], ys[1] - ys[0]))
            counts = {
                "mm_pairs": m.shape[1] ** 2 * self.h,
                "mm_live_pairs": int(allowed[b].sum()) * self.h,
                "me_pairs": m.shape[1] * len(e[b]) * self.h,
                "me_live_pairs": int(present[b].sum()) * len(e[b]) * self.h,
                "ee_pairs": len(e[b]) ** 2 * self.h if self.ee is not None else 0,
                "local_candidate_edges": frontier.edge_count,
                "metadata_nodes": 2 * len(e[b]) - 1 if adaptive else 0,
                "policy_rows": frontier.policy_rows,
                "original_e_rows": len(e[b]),
                "group_transform_rows": k,
                "group_key_value_rows": k,
                "mg_pairs": m.shape[1] * k * self.h,
                "mg_live_pairs": int(present[b].sum()) * k * self.h,
                "pool_source_rows": len(e[b]),
                "relation_pool_edges": len(e[b]) * m.shape[1] * self.h,
                "relation_pool_live_edges": len(e[b]) * int(present[b].sum()) * self.h,
                "posting_sort_entries": m.shape[1] * k,
                "organizer_projection_rows": len(e[b]) if adaptive else 0,
                "allocated_group_vectors": k,
                "spatial_index_entries": len(e[b]),
            }
            groups.append(
                PreparedGroups(
                    frontier, keys, values, states, gm, pos, B, R[b], mb, e[b], posts, cells, origin, step, counts
                )
            )
        return torch.stack(updated), e, tuple(groups)


class InteractionGroupCore(nn.Module):
    schema = "interaction_groups_v1"

    def __init__(
        self,
        module_input_dim=9,
        environment_input_dim=6,
        coordinate_dim=2,
        query_input_dim=0,
        operating_input_dim=12,
        output_dim=4,
        architecture="G-F",
        context_dim=128,
        num_heads=4,
        context_depth=2,
        local_hidden_dim=128,
        dropout=0.0,
        init_seed=0,
        query_tile_size=512,
        environment_nx=24,
        environment_ny=8,
        reader_prior=None,
        frontier_distance_scales=None,
    ):
        super().__init__()
        self._legacy_recipe_config = reader_prior is None
        reader_prior = "legacy_additive" if reader_prior is None else reader_prior
        if architecture not in ("G-F", "G-EE", "G-H", "G-EE-H"):
            raise ValueError("Unknown interaction-group arm")
        if reader_prior not in ("legacy_additive", "balanced_postings_v1"):
            raise ValueError("Unknown group reader prior")
        if coordinate_dim != 2 or context_dim % num_heads or dropout != 0:
            raise ValueError("Unsupported group dimensions/dropout")
        self.config = {
            "module_input_dim": module_input_dim,
            "environment_input_dim": environment_input_dim,
            "coordinate_dim": coordinate_dim,
            "query_input_dim": query_input_dim,
            "operating_input_dim": operating_input_dim,
            "output_dim": output_dim,
            "architecture": architecture,
            "context_dim": context_dim,
            "num_heads": num_heads,
            "context_depth": context_depth,
            "local_hidden_dim": local_hidden_dim,
            "dropout": dropout,
            "init_seed": init_seed,
            "query_tile_size": query_tile_size,
            "environment_nx": environment_nx,
            "environment_ny": environment_ny,
            "reader_prior": reader_prior,
            "frontier_distance_scales": frontier_distance_scales,
        }
        self.__dict__.update(self.config)
        # Common named tensors use identical seeds regardless of extra E->E.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            self.module_encoder = _mlp(module_input_dim + 16, context_dim, context_dim)
            self.environment_encoder = _mlp(environment_input_dim + 16, context_dim, context_dim)
            self.operating_encoder = (
                _mlp(operating_input_dim, context_dim, context_dim) if operating_input_dim else None
            )
            self.layers = nn.ModuleList(
                [
                    InteractionGroupLayer(context_dim, num_heads, environment_nx, environment_ny)
                    for _ in range(context_depth)
                ]
            )
            self.query_encoder = _mlp(query_input_dim + 14, context_dim, context_dim)
            self.proposal_key = nn.Linear(context_dim, 16, bias=False)
            self.proposal_query = nn.Linear(context_dim, 16, bias=False)
            self.group_query = nn.Linear(context_dim, context_dim, bias=False)
            self.fusion_trunk = _mlp(2 * context_dim, context_dim, context_dim)
            self.field_head = nn.Linear(context_dim, output_dim)
            if architecture in ("G-EE", "G-EE-H"):
                for layer in self.layers:
                    layer.ee = Exchange(context_dim, num_heads)
        if frontier_distance_scales is not None:
            if len(frontier_distance_scales) != context_depth:
                raise ValueError("One calibrated distance scale per layer required")
            for layer, scales in zip(self.layers, frontier_distance_scales):
                layer.frontier.distance_scales.copy_(torch.as_tensor(scales))
                layer.frontier.scales_calibrated.fill_(True)
        self.register_buffer("organization_epoch", torch.tensor(1, dtype=torch.long))
        self.register_buffer("organization_update", torch.tensor(0, dtype=torch.long))

    @property
    def adaptive_groups(self):
        return self.architecture in ("G-H", "G-EE-H")

    def get_config(self):
        config = {"schema": self.schema, **self.config}
        # Preserve legacy serialized adapter identities on exact resume.
        if self._legacy_recipe_config:
            config.pop("reader_prior")
            if config["frontier_distance_scales"] is None:
                config.pop("frontier_distance_scales")
        return config

    @classmethod
    def from_config(cls, config):
        cfg = dict(config)
        if cfg.pop("schema", None) != cls.schema:
            raise ValueError("Wrong interaction-group schema")
        return cls(**cfg)

    def set_organization_clock(self, epoch, update=0):
        self.organization_epoch.fill_(epoch)
        self.organization_update.fill_(update)

    def _parameter_version_stamp(self):
        return tuple((id(p), p._version) for p in self.parameters())

    def _buffer_version_stamp(self):
        return tuple(_stamp_tensor(p) for p in self.buffers())

    def prepare(self, scene, *, clamp_layer=None, forced_frontier=None):
        scene.validate(
            coordinate_dim=2,
            module_input_dim=self.module_input_dim,
            environment_input_dim=self.environment_input_dim,
            operating_input_dim=self.operating_input_dim,
        )
        origin, scale = scene.coordinate_origin, scene.coordinate_scale
        xy = (scene.module_coordinates - origin[:, None]) / scale[:, None]
        exy = (scene.environment_coordinates - origin[:, None]) / scale[:, None]
        m = self.module_encoder(
            torch.cat(
                (
                    scene.module_features,
                    coordinate_gamma(scene.module_coordinates, origin, scale),
                    scene.module_characteristic_lengths / scale[:, None],
                ),
                -1,
            )
        )
        e = self.environment_encoder(
            torch.cat(
                (
                    scene.environment_features,
                    coordinate_gamma(scene.environment_coordinates, origin, scale),
                    scene.environment_characteristic_lengths / scale[:, None],
                ),
                -1,
            )
        )
        if self.operating_encoder is not None:
            op = self.operating_encoder(scene.operating_features)
            m = m + op[:, None]
            e = e + op[:, None]
        m = m * scene.module_mask[..., None]
        e = e * scene.environment_mask[..., None]
        layers = []
        work = {
            "module_embedding_rows": scene.module_mask.numel(),
            "module_embedding_live_rows": int(scene.module_mask.sum()),
            "environment_embedding_rows": scene.environment_mask.numel(),
            "environment_embedding_live_rows": int(scene.environment_mask.sum()),
        }
        for j, layer in enumerate(self.layers):
            m, e, g = layer(
                m,
                e,
                scene,
                xy,
                exy,
                adaptive=self.adaptive_groups,
                epoch=int(self.organization_epoch),
                update=int(self.organization_update),
                seed=self.init_seed,
                layer=j,
                training=self.training,
                clamp=clamp_layer == j,
                forced=forced_frontier.get(j) if forced_frontier else None,
            )
            layers.append(g)
            for group in g:
                for name, value in group.work_counts.items():
                    work[name] = work.get(name, 0) + value
        return PreparedInteractionGroups(
            scene,
            tuple(layers),
            m,
            self.proposal_key(m),
            origin,
            scale,
            id(self),
            self._parameter_version_stamp(),
            self._buffer_version_stamp(),
            tuple(_stamp_tensor(v) for v in _scene_tensors(scene)),
            work,
        )

    def _assert_prepared(self, p, q):
        if p.model_owner != id(self) or p.parameter_versions != self._parameter_version_stamp():
            raise ValueError("Prepared interaction groups have a different model owner/state")
        if p.buffer_versions != self._buffer_version_stamp() or p.scene_versions != tuple(
            _stamp_tensor(v) for v in _scene_tensors(p.scene)
        ):
            raise ValueError("Prepared scene or organization clock changed")
        if q.scene_key is not None and q.scene_key != p.scene_key:
            raise ValueError("Query scene differs")
        if q.coordinate_frame_key is not None and q.coordinate_frame_key != p.scene.coordinate_frame_key:
            raise ValueError("Query coordinate frame differs")
        q.validate(batch_size=p.scene.batch_size, coordinate_dim=2, query_input_dim=self.query_input_dim)

    def _read_tile(
        self, p, b, q, coordinates, ids, *, exhaustive=False, fixed_candidates=None, association_scope="postings"
    ):
        g = p.final_groups[b]
        n = len(q)
        h = self.num_heads
        d = self.context_dim // h
        xy = (coordinates - p.frame_origin[b]) / p.frame_scale[b]
        present = p.scene.module_mask[b]
        nm = int(present.sum())
        mxy = (p.scene.module_coordinates[b] - p.frame_origin[b]) / p.frame_scale[b]
        scores = self.proposal_query(q) @ p.proposal_keys[b].T / 4
        scores = scores - (xy[:, None] - mxy[None]).square().sum(-1)
        scores = scores.masked_fill(~present[None], -torch.inf)
        selected = torch.argsort(scores, descending=True, stable=True, dim=-1)[:, : min(2, nm)]
        exploration = 0
        if self.training and nm > 1 and selected.shape[1]:
            expose = ids.remainder(16) == 0
            present_rows = torch.nonzero(present, as_tuple=False).flatten()
            cycle = (ids // 16 + int(self.organization_epoch)).remainder(nm)
            alt = present_rows[cycle]
            # Reserve last slot while preserving distinct present proposals.
            alt = torch.where(alt == selected[:, 0], present_rows[(cycle + 1).remainder(nm)], alt)
            selected = selected.clone()
            selected[:, -1] = torch.where(expose, alt, selected[:, -1])
            exploration = int(expose.sum())
        alpha = nullable(scores.gather(1, selected))
        # At most nine direct cell visits; no QxE geometric lookup.
        cell = torch.floor((xy - g.grid_origin) / g.grid_step + 0.5).long()
        offsets = torch.tensor([(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)], device=q.device)
        neighbors = cell[:, None] + offsets[None]
        valid = (
            (neighbors[..., 0] >= 0)
            & (neighbors[..., 0] < self.environment_nx)
            & (neighbors[..., 1] >= 0)
            & (neighbors[..., 1] < self.environment_ny)
        )
        safe = neighbors.clone()
        safe[..., 0].clamp_(0, self.environment_nx - 1)
        safe[..., 1].clamp_(0, self.environment_ny - 1)
        geo = g.labels[safe[..., 1] * self.environment_nx + safe[..., 0]]
        dist = ((xy[:, None] - (g.grid_origin + neighbors * g.grid_step)) / g.grid_step).square().sum(-1)
        # IDs secondary to distance. Stable ordering makes duplicate removal
        # select each group's nearest visited cell.
        byid = torch.argsort(geo, stable=True, dim=-1)
        order = byid.gather(
            1, torch.argsort(dist.gather(1, byid).masked_fill(~valid.gather(1, byid), torch.inf), stable=True, dim=-1)
        )
        geo = geo.gather(1, order)
        geo_valid = valid.gather(1, order)
        dist = dist.gather(1, order)
        earlier = torch.tril(torch.ones(9, 9, dtype=torch.bool, device=q.device), diagonal=-1)
        unique = geo_valid & ~((geo[:, :, None] == geo[:, None, :]) & earlier[None] & geo_valid[:, None, :]).any(-1)
        ranks = unique.long().cumsum(-1)
        keep = unique & (ranks <= 4)
        geo_ids = torch.full((n, 4), -1, dtype=torch.long, device=q.device)
        geo_weights = q.new_zeros(n, 4)
        row, col = torch.where(keep)
        slot = ranks[row, col] - 1
        geo_ids[row, slot] = geo[row, col]
        geo_weights[row, slot] = torch.exp(-0.5 * dist[row, col])
        if selected.shape[1]:
            posts = g.postings[selected].flatten(1)
            assoc = g.B.mean(0)[selected[:, :, None], posts[:, None, :]]
            membership = (posts[:, None, :, None] == g.postings[selected][:, :, None, :]).any(-1)
            pp = (alpha[:, :, None] * assoc * membership).sum(1)
            pp = pp.masked_fill((posts[:, :, None] == geo_ids[:, None, :]).any(-1), -torch.inf)
            ordp = torch.argsort(posts, stable=True, dim=-1)
            ordp = ordp.gather(1, torch.argsort(pp.gather(1, ordp), descending=True, stable=True, dim=-1))
            posts = posts.gather(1, ordp)
            pp = pp.gather(1, ordp)
            cnt = posts.shape[1]
            earlierp = torch.tril(torch.ones(cnt, cnt, dtype=torch.bool, device=q.device), diagonal=-1)
            up = torch.isfinite(pp) & ~((posts[:, :, None] == posts[:, None, :]) & earlierp[None]).any(-1)
            all_ids = torch.cat((geo_ids, posts.masked_fill(~up, -1)), -1)
        else:
            posts = torch.empty(n, 0, dtype=torch.long, device=q.device)
            all_ids = geo_ids
        candidate_count = (all_ids >= 0).sum(-1)
        if fixed_candidates is not None:
            candidates = fixed_candidates
        elif exhaustive:
            candidates = torch.arange(len(g.mass), device=q.device)[None].expand(n, -1)
        else:
            # Compact valid entries in geometric-first order, then cap before
            # neural group scoring. 4 reserved geometric slots, total <=8.
            compact = torch.argsort((all_ids < 0).long(), stable=True, dim=-1)
            candidates = all_ids.gather(1, compact)[:, : min(8, len(g.mass))]
        good = candidates >= 0
        safe_ids = candidates.clamp_min(0)
        # Compute address prior only on bounded retained groups. No QxK
        # learned logits, values or proposal prior allocation in production.
        geo_match = safe_ids[:, :, None] == geo_ids[:, None, :]
        geo_prior = (geo_match * geo_weights[:, None, :]).sum(-1)
        if self.reader_prior == "balanced_postings_v1":
            geo_prior = geo_prior / geo_weights.sum(-1, keepdim=True).clamp_min(1.0e-30)
        association_prior = torch.zeros_like(geo_prior)
        if selected.shape[1]:
            B = g.B.mean(0)
            bp = B[selected[:, :, None], safe_ids[:, None, :]]
            ownpost = (safe_ids[:, None, :, None] == g.postings[selected][:, :, None, :]).any(-1)
            if association_scope == "all" or exhaustive:
                ownpost = torch.ones_like(ownpost)
            if self.reader_prior == "balanced_postings_v1":
                normalizer = B.sum(-1) if association_scope == "all" or exhaustive else B.gather(1, g.postings).sum(-1)
                bp = bp / normalizer[selected][:, :, None].clamp_min(1.0e-30)
            association_prior = (alpha[:, :, None] * bp * ownpost).sum(1)
        geo_prior = geo_prior * good
        association_prior = association_prior * good
        rho = geo_prior + association_prior
        keys = g.keys[safe_ids]
        values = g.values[safe_ids]
        query = self.group_query(q).reshape(n, h, d)
        logits = torch.einsum("qhd,qchd->qhc", query, keys) / math.sqrt(d)
        logits = logits - (xy[:, None] - g.coordinates[safe_ids]).square().sum(-1)[:, None]
        logits = logits + (g.mass[safe_ids] * rho).clamp_min(1.0e-30).log()[:, None]
        weights = nullable(logits, good[:, None] & (rho > 0)[:, None] & (g.mass[safe_ids] > 0)[:, None])
        context = torch.einsum("qhc,qchd->qhd", weights, values).flatten(1)
        fields = self.field_head(self.fusion_trunk(torch.cat((q, context), -1)))
        work = {
            "query_module_scores": n * present.numel(),
            "query_module_live_scores": n * nm,
            "spatial_cell_visits": n * 9,
            "posting_visits": posts.numel(),
            "candidate_entries": int(candidate_count.sum()),
            "candidate_truncations": int((candidate_count > 8).sum()),
            "group_scores": candidates.numel() * h,
            "group_value_reads": candidates.numel(),
            "group_live_scores": int(good.sum()) * h,
            "group_live_value_reads": int(good.sum()),
            "original_e_query_reads": 0,
            "module_value_query_reads": 0,
            "empty_queries": int((~good.any(-1)).sum()),
            "exploration_queries": exploration,
            "native_query_rows": n,
        }
        return (
            fields,
            work,
            {
                "candidates": candidates,
                "proposal_ids": selected,
                "alpha": alpha,
                "weights": weights,
                "geometry_prior": geo_prior,
                "association_prior": association_prior,
            },
        )

    def decode(
        self, p, query, *, exhaustive=False, return_trace=False, candidate_traces=None, association_scope="postings"
    ):
        if association_scope not in ("postings", "all"):
            raise ValueError("Unknown diagnostic association scope")
        self._assert_prepared(p, query)
        gamma = coordinate_gamma(query.coordinates, p.frame_origin, p.frame_scale)
        q = self.query_encoder(torch.cat((query.features, gamma), -1) if self.query_input_dim else gamma)
        outputs = []
        work = {}
        traces = []
        for b in range(len(q)):
            tiles = []
            trace = []
            for start in range(0, q.shape[1], self.query_tile_size):
                end = min(q.shape[1], start + self.query_tile_size)
                ids = (
                    query.query_ids[b, start:end]
                    if query.query_ids is not None
                    else torch.arange(start, end, device=q.device)
                )
                f, w, t = self._read_tile(
                    p,
                    b,
                    q[b, start:end],
                    query.coordinates[b, start:end],
                    ids,
                    exhaustive=exhaustive,
                    association_scope=association_scope,
                    fixed_candidates=candidate_traces[b][len(tiles)]["candidates"]
                    if candidate_traces is not None
                    else None,
                )
                tiles.append(f)
                trace.append(t)
                for name, value in w.items():
                    work[name] = work.get(name, 0) + value
            outputs.append(torch.cat(tiles, 0))
            traces.append(trace)
        fields = torch.stack(outputs)
        if query.query_mask is not None:
            fields = fields * query.query_mask[..., None]
        output = ContextualJointOutput(
            nonlinear_field=fields[..., :3], thermal_temperature=fields[..., 3:4], work_counts=work, joint_fields=fields
        )
        return (output, traces) if return_trace else output

    def forward(self, scene, query):
        return self.decode(self.prepare(scene), query)

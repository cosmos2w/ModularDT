"""Measure-aware typed attention used by the new forward family."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def explicit_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    allowed: Tensor,
    additive_bias: Tensor | None = None,
) -> Tensor:
    """Reference softmax attention for ``[B,heads,N,head_width]`` tensors.

    ``allowed`` uses the same polarity as SDPA boolean masks: True means that a
    donor is eligible. Empty rows return zero and never form an all-masked
    softmax. The routine is intentionally simple and serves as a correctness
    reference, not the production path.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [B,heads,N,width].")
    if allowed.ndim != 3:
        raise ValueError("allowed must have shape [B,N_query,N_key].")
    if query.shape[:2] != key.shape[:2] or key.shape != value.shape:
        raise ValueError("Attention batch/head axes and key/value shapes must match.")
    if allowed.shape != (query.shape[0], query.shape[2], key.shape[2]):
        raise ValueError("allowed mask shape does not match attention axes.")
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    if additive_bias is not None:
        scores = scores + additive_bias
    row_has_donor = allowed.any(dim=-1, keepdim=True)
    row_mask = allowed[:, None, :, :]
    scores = scores.masked_fill(~row_mask, -torch.inf)
    scores = torch.where(row_has_donor[:, None, :, :], scores, torch.zeros_like(scores))
    weights = torch.softmax(scores, dim=-1)
    weights = weights * row_mask.to(weights.dtype)
    weights = torch.where(row_has_donor[:, None, :, :], weights, torch.zeros_like(weights))
    return torch.matmul(weights, value)


class _RouteAttention(nn.Module):
    def __init__(self, width: int, heads: int, coordinate_dim: int, dropout: float) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("context_dim must be divisible by num_heads.")
        self.width = int(width)
        self.heads = int(heads)
        self.head_width = self.width // self.heads
        self.dropout = float(dropout)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.geometry_bias = nn.Linear(coordinate_dim + 1, heads, bias=False)

    def forward(
        self,
        destination: Tensor,
        donor: Tensor,
        destination_coordinates: Tensor,
        donor_coordinates: Tensor,
        donor_measure: Tensor,
        donor_mask: Tensor,
        destination_mask: Tensor,
        coordinate_scale: Tensor,
        destination_norm: nn.Module,
        donor_norm: nn.Module,
        *,
        remove_self: bool = False,
    ) -> Tensor:
        batch, n_query, _ = destination.shape
        n_key = donor.shape[1]
        if n_key == 0:
            return destination.new_zeros(batch, n_query, self.width)

        q = self.query(destination_norm(destination)).view(batch, n_query, self.heads, self.head_width).transpose(1, 2)
        k = self.key(donor_norm(donor)).view(batch, n_key, self.heads, self.head_width).transpose(1, 2)
        v = self.value(donor_norm(donor)).view(batch, n_key, self.heads, self.head_width).transpose(1, 2)

        scale = coordinate_scale[:, None, None, :].clamp_min(torch.finfo(destination_coordinates.dtype).tiny)
        relative = (destination_coordinates[:, :, None, :] - donor_coordinates[:, None, :, :]) / scale
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        geometry = torch.cat((relative, distance), dim=-1)
        geometry_bias = self.geometry_bias(geometry).permute(0, 3, 1, 2)

        positive_measure = donor_measure > 0
        safe_measure = torch.where(positive_measure, donor_measure, torch.ones_like(donor_measure))
        # A positive constant reference cancels in each row's softmax. This is
        # the dimensionless log-measure term in the measure-aware attention law.
        log_measure = torch.log(safe_measure)
        allowed = donor_mask[:, None, :].expand(batch, n_query, n_key)
        allowed = allowed & positive_measure[:, None, :]
        allowed = allowed & destination_mask[:, :, None]
        if remove_self:
            if n_query != n_key:
                raise ValueError("Self exclusion is only valid for equal token axes.")
            diagonal = torch.eye(n_query, dtype=torch.bool, device=destination.device)
            allowed = allowed & ~diagonal[None, :, :]

        row_has_donor = allowed.any(dim=-1, keepdim=True)
        additive = geometry_bias + log_measure[:, None, None, :]
        additive = additive.masked_fill(~allowed[:, None, :, :], -torch.inf)
        # SDPA's softmax is undefined for an all-masked row. Give such a row a
        # finite temporary distribution and erase its value immediately after.
        additive = torch.where(row_has_donor[:, None, :, :], additive, torch.zeros_like(additive))
        update = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=additive,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        update = update.transpose(1, 2).reshape(batch, n_query, self.width)
        update = self.output(update)
        valid_rows = row_has_donor & destination_mask[:, :, None]
        return update * valid_rows.to(update.dtype)


class TypedAttentionBlock(nn.Module):
    """One simultaneous pre-normalized MM/ME/EM context update."""

    def __init__(
        self,
        width: int,
        heads: int,
        coordinate_dim: int,
        dropout: float = 0.0,
        update_environment: bool = True,
    ) -> None:
        super().__init__()
        self.module_norm = nn.LayerNorm(width)
        self.environment_norm = nn.LayerNorm(width)
        self.mm = _RouteAttention(width, heads, coordinate_dim, dropout)
        self.me = _RouteAttention(width, heads, coordinate_dim, dropout)
        self.em = _RouteAttention(width, heads, coordinate_dim, dropout) if update_environment else None
        self.module_ff_norm = nn.LayerNorm(width)
        self.environment_ff_norm = nn.LayerNorm(width) if update_environment else None
        self.module_ff = nn.Sequential(
            nn.Linear(width, 4 * width), nn.SiLU(), nn.Dropout(dropout), nn.Linear(4 * width, width)
        )
        self.environment_ff = (
            nn.Sequential(nn.Linear(width, 4 * width), nn.SiLU(), nn.Dropout(dropout), nn.Linear(4 * width, width))
            if update_environment
            else None
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        module_states: Tensor,
        environment_states: Tensor,
        module_coordinates: Tensor,
        environment_coordinates: Tensor,
        module_measure: Tensor,
        environment_measure: Tensor,
        module_mask: Tensor,
        environment_mask: Tensor,
        coordinate_scale: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # All three routes read the same pre-update states.
        mm_update = self.mm(
            module_states,
            module_states,
            module_coordinates,
            module_coordinates,
            module_measure,
            module_mask,
            module_mask,
            coordinate_scale,
            self.module_norm,
            self.module_norm,
            remove_self=True,
        )
        me_update = self.me(
            module_states,
            environment_states,
            module_coordinates,
            environment_coordinates,
            environment_measure,
            environment_mask,
            module_mask,
            coordinate_scale,
            self.module_norm,
            self.environment_norm,
        )
        module_next = module_states + self.dropout(mm_update) + self.dropout(me_update)
        module_next = module_next + self.dropout(self.module_ff(self.module_ff_norm(module_next)))
        if self.em is not None:
            em_update = self.em(
                environment_states,
                module_states,
                environment_coordinates,
                module_coordinates,
                module_measure,
                module_mask,
                environment_mask,
                coordinate_scale,
                self.environment_norm,
                self.module_norm,
            )
            environment_next = environment_states + self.dropout(em_update)
            environment_next = environment_next + self.dropout(
                self.environment_ff(self.environment_ff_norm(environment_next))
            )
        else:
            # A-D has no downstream consumer for a final environment update.
            # Its final ME route still reads the prior E states; avoid dormant
            # EM/FF parameters while preserving the two module update layers.
            environment_next = environment_states
        module_next = module_next * module_mask[:, :, None].to(module_next.dtype)
        environment_next = environment_next * environment_mask[:, :, None].to(environment_next.dtype)
        return module_next, environment_next

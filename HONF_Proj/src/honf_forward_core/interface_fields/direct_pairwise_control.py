"""Source-resolved, full-access modulation without collective hyperedge states.

The physical modules retain B-fine's names for matched fresh initialization.
Each typed modulation consumes only its receiver, individual source, relative
and absolute geometry, and adapter-prescribed context. Ordinary message
aggregation and earlier-phase ancestry remain many-source dependencies.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from honf_forward_core.nn import LazyMLP

from .dense_pairwise import DensePairwiseField


class DirectPairwiseControlField(DensePairwiseField):
    """Competent independently trained Pair-F capacity control, with dense work."""

    phase_diagnostic_prefix = "pair_"

    def __init__(self, hidden_dim, message_hidden_dim, num_heads, fourier_frequencies,
                 *, control_hidden_dim=80, activation_checkpointing=False):
        super().__init__(hidden_dim, message_hidden_dim, num_heads, fourier_frequencies,
                         activation_checkpointing=activation_checkpointing)
        if isinstance(control_hidden_dim, bool) or not isinstance(control_hidden_dim, int) or control_hidden_dim < 1:
            raise ValueError("Pair control width must be a positive integer")
        self.control_hidden_dim = int(control_hidden_dim)
        self.pair_controls = nn.ModuleDict({
            tau: LazyMLP(self.control_hidden_dim, out_dim=2 * num_heads if tau == "QE" else 1,
                         num_layers=2)
            for tau in ("MM", "ME", "EM", "QM", "QE")
        })
        # Identity modulation permits a first task gradient into the output
        # projection, followed by gradients into its receiver/source features.
        for head in self.pair_controls.values():
            nn.init.zeros_(head.net[-1].weight)
            nn.init.zeros_(head.net[-1].bias)

    def _control(self, tau, receiver_state, source_state, receiver_coords, source_coords, encoded):
        receivers, sources = receiver_coords.shape[1], source_coords.shape[1]
        relative = (receiver_coords[:, :, None] - source_coords[:, None]) / encoded.coordinate_scale
        content = torch.cat((
            receiver_state[:, :, None].expand(-1, -1, sources, -1),
            source_state[:, None].expand(-1, receivers, -1, -1),
            self.relative_fourier(relative),
            (receiver_coords / encoded.coordinate_scale)[:, :, None].expand(-1, -1, sources, -1),
            (source_coords / encoded.coordinate_scale)[:, None].expand(-1, receivers, -1, -1),
            encoded.global_token[:, None, None].expand(-1, receivers, sources, -1),
        ), -1)
        return self._mlp(self.pair_controls[tau], content)

    @staticmethod
    def _ledger(tau, validity, *, attention_cells=0):
        rows = validity.numel()
        return {f"pair_{tau}_executed_rows": validity.new_tensor(rows, dtype=torch.long),
                f"pair_{tau}_allocated_rows": validity.new_tensor(rows, dtype=torch.long),
                f"pair_{tau}_unique_pairs": validity.sum(),
                f"pair_{tau}_eligible_pairs": validity.sum(),
                f"pair_{tau}_fine_calls": validity.new_tensor(1, dtype=torch.long),
                # The independently learned direct-control MLP evaluates a
                # dense rectangle in addition to the physical fine kernel.
                f"pair_{tau}_control_executed_rows": validity.new_tensor(rows, dtype=torch.long),
                f"pair_{tau}_control_calls": validity.new_tensor(1, dtype=torch.long),
                f"pair_{tau}_attention_cells": validity.new_tensor(attention_cells, dtype=torch.long)}

    def prepare(self, encoded, module_states, *, return_routing_maps=False, interaction_context=None):
        centres, env, present = encoded.module_centers, encoded.env_coords, encoded.module_present
        modules, environments = centres.shape[1], env.shape[1]
        count = present.sum(1, keepdim=True)
        relative = (centres[:, :, None] - centres[:, None]) / encoded.coordinate_scale
        mm = self._mlp(self.mm_message, torch.cat((
            module_states[:, :, None].expand(-1, -1, modules, -1),
            module_states[:, None].expand(-1, modules, -1, -1), self.relative_fourier(relative)), -1))
        mm_valid = ((present[:, :, None] > .5) & (present[:, None] > .5)
                    & ~torch.eye(modules, device=present.device, dtype=torch.bool)[None])
        mm_gain = self._control("MM", module_states, module_states, centres, centres, encoded)
        a_mm = (mm * (1 + torch.tanh(mm_gain)) * mm_valid[..., None]).sum(2) / (1 + count[..., None])
        relative = (centres[:, :, None] - env[:, None]) / encoded.coordinate_scale
        me = self._mlp(self.me_message, torch.cat((
            module_states[:, :, None].expand(-1, -1, environments, -1),
            encoded.env_tokens[:, None].expand(-1, modules, -1, -1), self.relative_fourier(relative)), -1))
        me_gain = self._control("ME", module_states, encoded.env_tokens, centres, env, encoded)
        me_valid = (present[:, :, None] > .5).expand(-1, -1, environments)
        a_me = (me * (1 + torch.tanh(me_gain)) * encoded.env_weights[:, None, :, None]).sum(2)
        a_me = a_me / encoded.env_weights.sum(1)[:, None, None].clamp_min(1e-12)
        a_me = a_me * present[..., None]
        em = self._mlp(self.em_message, torch.cat((
            encoded.env_tokens[:, :, None].expand(-1, -1, modules, -1),
            module_states[:, None].expand(-1, environments, -1, -1),
            self.relative_fourier(-relative.transpose(1, 2))), -1))
        em_gain = self._control("EM", encoded.env_tokens, module_states, env, centres, encoded)
        em_valid = (present[:, None] > .5).expand(-1, environments, -1)
        a_em = (em * (1 + torch.tanh(em_gain)) * em_valid[..., None]).sum(2) / (1 + count[..., None])
        module_tokens = (module_states + self.module_update(torch.cat((
            module_states, a_mm, a_me, encoded.global_token[:, None].expand(-1, modules, -1)), -1))) * present[..., None]
        env_tokens = encoded.env_tokens + self.env_update(torch.cat((
            encoded.env_tokens, a_em, encoded.global_token[:, None].expand(-1, environments, -1)), -1))
        key, value = self.project_environment_sources(env_tokens)
        ledger = {**self._ledger("MM", mm_valid), **self._ledger("ME", me_valid), **self._ledger("EM", em_valid)}
        phase = {"P0": 0, "P1": 1, "P2": 2}.get(getattr(interaction_context, "phase", None), 0)
        return {"module_tokens": module_tokens, "env_tokens": env_tokens,
                "projected_key": key, "projected_value": value, "pair_ledger": ledger,
                "pair_phase": phase}

    def preparation_aux(self, state, *, include_diagnostics=False):
        return {**state["pair_ledger"], "pair_phase": state["module_tokens"].new_tensor(state["pair_phase"])}

    def read(self, state, encoded, receivers, receiver_features, *, return_routing_maps=False):
        batch, queries, _ = receivers.shape
        modules, environments = state["module_tokens"].shape[1], state["env_tokens"].shape[1]
        relative = (receivers[:, :, None] - encoded.module_centers[:, None]) / encoded.coordinate_scale
        messages = self._mlp(self.query_module_message, torch.cat((
            state["module_tokens"][:, None].expand(-1, queries, -1, -1),
            self.relative_fourier(relative),
            encoded.global_token[:, None, None].expand(-1, queries, modules, -1)), -1))
        qm_gain = self._control("QM", receiver_features, state["module_tokens"], receivers,
                                encoded.module_centers, encoded)
        qm_valid = (encoded.module_present[:, None] > .5).expand(-1, queries, -1)
        module_context = self.query_module_output((messages * (1 + torch.tanh(qm_gain))
            * qm_valid[..., None]).sum(2) / (1 + encoded.module_present.sum(1)[:, None, None]))
        qe_valid = (encoded.env_weights[:, None] > 0).expand(-1, queries, -1)
        if environments:
            relative = (receivers[:, :, None] - encoded.env_coords[:, None]) / encoded.coordinate_scale
            bias = self._mlp(self.env_geometry_bias, self.relative_fourier(relative)).permute(0, 3, 1, 2)
            query = self.env_attention.project_query(self.env_query(receiver_features))
            modulation = self._control("QE", receiver_features, state["env_tokens"], receivers,
                                       encoded.env_coords, encoded)
            gain, score_offset = modulation.chunk(2, -1)
            score = torch.matmul(query, state["projected_key"].transpose(-1, -2))
            score = score / math.sqrt(self.env_attention.head_dim) + bias + score_offset.permute(0, 3, 1, 2)
            safe_mass = torch.where(encoded.env_weights > 0, encoded.env_weights, torch.ones_like(encoded.env_weights))
            score = (score + safe_mass.log()[:, None, None]).masked_fill(~qe_valid[:, None], -torch.inf)
            attention = torch.softmax(score, -1)
            attention = torch.where(qe_valid[:, None].any(-1, keepdim=True), attention, torch.zeros_like(attention))
            context = torch.matmul(attention * (1 + torch.tanh(gain.permute(0, 3, 1, 2))), state["projected_value"])
            environment_context = self.env_attention.output(context.transpose(1, 2).reshape(batch, queries, self.hidden_dim))
            environment_context = environment_context * qe_valid.any(-1)[..., None]
        else:
            attention = None
            environment_context = torch.zeros_like(module_context)
        aux = {**self._ledger("QM", qm_valid), **self._ledger("QE", qe_valid,
               attention_cells=batch * queries * environments * self.env_attention.num_heads)}
        if return_routing_maps:
            aux["pair_QM_gain"] = qm_gain.detach()
            if attention is not None:
                aux["pair_QE_attention"] = attention.detach()
        return module_context + environment_context, aux

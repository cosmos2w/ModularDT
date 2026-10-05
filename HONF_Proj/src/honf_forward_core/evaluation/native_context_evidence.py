"""Opt-in evidence for the actual native coarse and local context paths.

Instrumentation observes the ordinary operator; it does not repeat physical
reads. Local support is reconstructed from that call's input geometry and the
unchanged native compact kernel. Hidden context norms are not physical energy.
"""

from contextlib import AbstractContextManager

import torch

from honf_forward_core.interface_fields.common import SharedInterfaceContext
from honf_forward_core.interface_fields.typed_hypergraph_state import build_source_catalogue


class NativeContextEvidence(AbstractContextManager):
    def __init__(self, core, *, save_arrays=False):
        if core.training or not isinstance(core.common, SharedInterfaceContext):
            raise ValueError("Native context evidence requires a frozen SharedInterfaceContext")
        if core.common.coarse_module_source != "module_states":
            raise ValueError("Native source evidence currently requires module_states coarse sources")
        self.core, self.common = core, core.common
        self.save_arrays = bool(save_arrays)
        self.arrays, self.phases = {}, {}
        self._phase = None

    def _save(self, name, value):
        if self.save_arrays:
            self.arrays[name] = value.detach().cpu().numpy()

    @staticmethod
    def _norm_sums(value):
        norms = torch.linalg.vector_norm(value.detach().double(), dim=-1)
        return float(norms.sum()), int(norms.numel())

    def __enter__(self):
        self._methods = [(self.core, "prepare"), (self.common, "prepare_coarse"),
                         (self.common, "read_coarse"), (self.common, "read_local")]
        self._originals = [(owner, name, name in owner.__dict__, getattr(owner, name))
                           for owner, name in self._methods]
        originals = {name: method for _, name, _, method in self._originals}

        def prepare_coarse(states, environment, present, measure, *args, **kwargs):
            result = originals["prepare_coarse"](states, environment, present, measure, *args, **kwargs)
            self._phase = f"P{len(self.phases)}"
            groups = int(result.shape[1])
            heads = self.common.coarse_env_attention.num_heads
            batch = int(states.shape[0])
            self.phases[self._phase] = {
                "coarse_module_source": "phase-current module_states",
                "module_source_mask_count": int((present > .5).sum()),
                "environment_positive_measure_count": int((measure > 0).sum()),
                "module_source_measure_sum": float(present.sum()),
                "environment_source_measure_sum": float(measure.sum()),
                "latent_count": groups, "read_calls": 0, "local_calls": 0,
                "coarse_source_attention_cells": batch * heads * groups * (states.shape[1] + environment.shape[1]),
                "coarse_self_attention_cells": batch * heads * groups * groups * len(self.common.coarse_blocks),
                "coarse_receiver_attention_cells": 0,
                "local_executed_message_rows": 0, "local_allocated_geometry_pairs": 0,
                "coarse_context_norm_sum": 0., "coarse_receiver_rows": 0,
                "local_context_norm_sum": 0., "local_receiver_rows": 0,
                "coarse_reach": "every prepared latent reads all present module states and all positive-measure environmental states; every receiver reads all latents",
                "module_coarse_measure": "native presence-mask attention; no added module quadrature bias",
                "environment_coarse_measure": "native log-quadrature attention prior",
            }
            for name, value in (("module_states", states), ("environment_states", environment),
                                ("module_present", present), ("environment_measure", measure),
                                ("coarse_state", result)):
                self._save(f"{self._phase}/coarse/{name}", value)
            return result

        def prepare(*args, **kwargs):
            result = originals["prepare"](*args, **kwargs)
            encoded = result.encoded
            catalogue = build_source_catalogue(encoded)
            for name in ("source_coords", "source_measures", "source_valid", "source_ids"):
                for kind, value in catalogue[name].items():
                    self._save(f"{self._phase}/coarse/{kind}_{name}", value)
            return result

        def read_coarse(*args, **kwargs):
            result = originals["read_coarse"](*args, **kwargs)
            row = self.phases[self._phase]
            call = row["read_calls"]
            row["read_calls"] += 1
            row["coarse_receiver_attention_cells"] += result.shape[0] * result.shape[1] * row["latent_count"] * self.common.coarse_read.num_heads
            norm, count = self._norm_sums(result)
            row["coarse_context_norm_sum"] += norm
            row["coarse_receiver_rows"] += count
            self._save(f"{self._phase}/read{call:04d}/coarse_contribution", result)
            return result

        def read_local(receivers, states, centres, features, present, scale, radius):
            result, counts = originals["read_local"](receivers, states, centres, features, present, scale, radius)
            row = self.phases[self._phase]
            call = row["local_calls"]
            row["local_calls"] += 1
            # Reconstruct only geometric support, using this exact call's
            # source/receiver coordinates. The physical local MLP runs once.
            distance = torch.linalg.vector_norm(receivers.detach()[:, :, None] - centres.detach()[:, None], dim=-1)
            support_radius = max(float(radius) * self.common.local_radius_factor, 1e-8)
            support = (distance < support_radius) & (present[:, None] > .5)
            kernel = ((1 - distance / support_radius).clamp_min(0).square()
                      * (1 + 2 * distance / support_radius)) * support
            normalized = kernel / (1 + kernel.sum(-1, keepdim=True))
            if not torch.equal(support.sum(-1).to(counts), counts):
                raise ValueError("Recorded native local support disagrees with executed neighbour counts")
            row["local_executed_message_rows"] += int(counts.sum())
            row["local_allocated_geometry_pairs"] += support.numel()
            norm, count = self._norm_sums(result)
            row["local_context_norm_sum"] += norm
            row["local_receiver_rows"] += count
            for name, value in (("receiver_coords", receivers), ("local_support", support),
                                ("local_kernel_weight", normalized), ("local_contribution", result)):
                self._save(f"{self._phase}/read{call:04d}/{name}", value)
            return result, counts

        self.common.prepare_coarse = prepare_coarse
        self.core.prepare = prepare
        self.common.read_coarse = read_coarse
        self.common.read_local = read_local
        return self

    def __exit__(self, *_args):
        for owner, name, owned, original in self._originals:
            if owned:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        return False

    def summary(self):
        return {"phases": self.phases,
                "measurement": "ordinary native preparation/read calls; coarse attention cells and gathered local MLP rows are distinct work units",
                "local_support": "strict native radius and presence mask, reconstructed from actual call inputs and checked against executed counts",
                "contribution_units": "hidden feature norms; additive contributions precede the shared nonlinear field head",
                "physical_claim": "computational information paths; no causal or energy interpretation"}

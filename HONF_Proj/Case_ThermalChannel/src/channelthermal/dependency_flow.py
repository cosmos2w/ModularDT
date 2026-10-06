"""Opt-in ThermalChannel composition with explicitly restricted flow inputs."""
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from honf_forward_core.interface_fields.geometry_flow_field import GeometryFlowField

DEPENDENCY_ID = "thermal_dependency_flow_v1"
FLOW_ORDER = ("u", "v", "p", "omega")
CASE_CAPABILITY = "channelthermal_analytic_wake_prescribed_flow_v1"


class ThermalFlowReader(nn.Module):
    """Audited analytic-wake benchmark policy; not a general multiphysics rule.

    Context reads Re, inlet speed, domain lengths, nu and radius. Sources read
    positions/radius and an explicit heat slot. D-sep constructs that slot
    without accessing heat; D-open reads dataset-normalized own heat.
    """
    def __init__(self, policy, reader_config=None):
        super().__init__()
        if policy not in ("D-sep", "D-open"):
            raise ValueError("Only the matched D-sep/D-open policies are supported.")
        self.policy = policy
        self.reader = GeometryFlowField(**(reader_config or {}))

    def prepare_flow(self, structure):
        centers = structure["module_centers"].float()
        present = structure["module_present"].float()
        batch = centers.shape[0]
        def column(name, fallback):
            value = structure.get(name)
            return centers.new_full((batch, 1), fallback) if value is None else value.reshape(batch, 1).float()
        lengths = torch.cat((column("domain_length_x", 12), column("domain_length_y", 4)), -1)
        material = structure["material_params"].float()
        if material.ndim == 1:
            material = material[None].expand(batch, -1)
        nu, radius = material[:, 0:1], material[:, 5:6]
        count = present.sum(1, keepdim=True)
        context = torch.cat((column("re", 0) / 100, column("u_in", 0), lengths[:, :1] / 12,
                             lengths[:, 1:] / 4, nu * 100, radius, count / 12,
                             count / lengths.prod(-1, keepdim=True)), -1)
        heat = centers.new_zeros((*centers.shape[:2], 1))
        if self.policy == "D-open":
            heat = structure["heat_powers"].float()[..., None]
        sources = torch.cat((centers / lengths[:, None],
                             (radius / lengths)[:, None].expand(-1, centers.shape[1], -1), heat), -1)
        return self.reader.prepare(sources, context, centers, present, lengths)

    def read_flow(self, prepared, query_xy, chunk_size=1024):
        return self.reader.read(prepared, query_xy.float(), chunk_size)

    def forward(self, structure, query_xy):
        return self.read_flow(self.prepare_flow(structure), query_xy)


@dataclass
class PreparedDependencyCase:
    thermal: object
    flow: object

    @property
    def prepared(self): return self.thermal.prepared

    @property
    def architecture(self): return self.thermal.architecture


class DependencySeparatedThermalModel(nn.Module):
    """One frozen thermal wrapper call plus one small learned flow reader."""
    def __init__(self, thermal, flow):
        super().__init__()
        if tuple(thermal.config.channelthermal.field_names) != (*FLOW_ORDER, "temperature"):
            raise ValueError("Dependency composition requires unchanged ThermalChannel role ordering.")
        self.thermal, self.flow = thermal, flow
        self.thermal.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.thermal.eval()
        return self

    @property
    def config(self): return self.thermal.config
    @property
    def core(self): return self.thermal.core
    @property
    def local_coupling(self): return self.thermal.local_coupling
    @property
    def input_adapter(self): return self.thermal.input_adapter
    @property
    def local_surrogate_attached(self): return self.thermal.local_surrogate_attached

    def selection_state(self): return self.thermal.selection_state()
    def set_training_progress(self, **kwargs): return self.thermal.set_training_progress(**kwargs)
    def export_typed_hypergraph(self, prepared):
        return self.thermal.export_typed_hypergraph(prepared.thermal)
    def prepare_flow(self, structure): return self.flow.prepare_flow(structure)
    def read_flow(self, prepared, query_xy): return self.flow.read_flow(prepared, query_xy)
    def forward_flow(self, structure, query_xy): return self.flow(structure, query_xy)
    def _should_use_local_outputs(self, mode): return self.thermal._should_use_local_outputs(mode)

    def evaluate_heat_batch(self, structures, query_xy, **kwargs):
        """Reuse only within one explicit fixed-layout/context heat request.

        This is not a case-ID cache: equality is checked on actual allowed
        tensors; changed geometry/context requires a fresh request.
        """
        if not structures:
            return []
        if kwargs.get("local_module_params") is not None:
            raise ValueError(
                "Heat batches rebuild local module parameters from each heat input; "
                "a shared local_module_params tensor can contain stale own heat."
            )
        first = structures[0]
        geometry_keys = ("module_centers", "module_present", "re", "u_in", "material_params",
                         "domain_length_x", "domain_length_y")
        for structure in structures[1:]:
            if any((key in first) != (key in structure) or
                   (key in first and not torch.equal(first[key], structure[key])) for key in geometry_keys):
                raise ValueError("Heat batch requires identical physical layout and prescribed context.")
        prepared = self.flow.prepare_flow(first) if self.flow.policy == "D-sep" else None
        retained_flow = self.flow.read_flow(prepared, query_xy) if prepared is not None else None
        results = []
        for structure in structures:
            flow_prepared = prepared if prepared is not None else self.flow.prepare_flow(structure)
            flow = retained_flow if retained_flow is not None else self.flow.read_flow(flow_prepared, query_xy)
            output = self.thermal(structure=structure, query_xy=query_xy, **kwargs)
            output["pred_field"] = torch.cat((flow, output["pred_field"][..., 4:5]), -1)
            if "prepared_state" in output:
                output["prepared_state"] = PreparedDependencyCase(output["prepared_state"], flow_prepared)
            results.append(output)
        return results

    def forward(self, structure=None, query_xy=None, **kwargs):
        # Prepared evaluator probes this explicit facade alias. The retained
        # wrapper names the same diagnostic request organizer diagnostics.
        if kwargs.pop("return_interaction_tensor", False):
            kwargs["return_organizer_diagnostics"] = True
        if structure is None:
            names = ("re", "u_in", "module_centers", "heat_powers", "module_present", "material_params")
            structure = {name: kwargs[name] for name in names}
        prepared_flow = self.flow.prepare_flow(structure)
        output = self.thermal(structure=structure, query_xy=query_xy, **kwargs)
        flow = self.flow.read_flow(prepared_flow, query_xy)
        output["pred_field"] = torch.cat((flow, output["pred_field"][..., 4:5]), -1)
        if "prepared_state" in output:
            output["prepared_state"] = PreparedDependencyCase(output["prepared_state"], prepared_flow)
        return output

    def decode_prepared(self, prepared, query_xy, **kwargs):
        if not isinstance(prepared, PreparedDependencyCase):
            raise TypeError("Candidate prepared decode requires both thermal and flow states.")
        output = self.thermal.decode_prepared(prepared.thermal, query_xy, **kwargs)
        flow = self.flow.read_flow(prepared.flow, query_xy, kwargs.get("receiver_chunk_size") or 1024)
        output["pred_field"] = torch.cat((flow, output["pred_field"][..., 4:5]), -1)
        return output


def load_dependency_model(path, device="cpu", checkpoint=None):
    """Strict opt-in child identity and existing trusted parent reconstruction."""
    from channelthermal.evaluation.loading import load_model
    from channelthermal.training.checkpoints import _file_sha256
    from honf_runtime.checkpoints import validate_checkpoint_identity
    from honf_runtime.compat import load_trusted_checkpoint
    child = checkpoint if checkpoint is not None else load_trusted_checkpoint(path, map_location="cpu")
    validate_checkpoint_identity(child, case_id="ThermalChannel", model_family="honf_forward", workflow="forward")
    if child.get("dependency_identity") != DEPENDENCY_ID:
        raise ValueError("Not a dependency-separated ThermalChannel checkpoint.")
    if child.get("case_capability") != CASE_CAPABILITY:
        raise ValueError("Candidate requires its explicit audited benchmark capability.")
    parent_path = Path(child["thermal_parent_checkpoint"])
    if _file_sha256(parent_path) != child["thermal_parent_sha256"]:
        raise ValueError("Frozen thermal parent checkpoint identity changed.")
    thermal, parent = load_model(parent_path, torch.device(device))
    if parent.get("channel_order") != [*FLOW_ORDER, "temperature"]:
        raise ValueError("Candidate requires the unchanged five-channel ThermalChannel order.")
    flow = ThermalFlowReader(child["dependency_policy"], child["flow_reader_config"])
    flow.load_state_dict(child["flow_state_dict"], strict=True)
    model = DependencySeparatedThermalModel(thermal, flow.to(device)).eval()
    # Existing evaluators need the inherited dataset/model metadata, while the
    # child identity and selection age remain explicit.
    metadata = dict(parent)
    metadata.update(child)
    return model, metadata

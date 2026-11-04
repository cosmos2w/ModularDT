"""Frozen local operator used by the Thermal port composition."""
from torch import nn
from .local_coupling import (LocalSurrogateCoupling, build_local_module_params_from_global,
                             update_local_module_params_from_ports)


class FrozenThermalLocal(nn.Module):
    """Strictly loaded local asset; frozen weights retain derivatives of inputs."""

    def __init__(self, checkpoint_path):
        super().__init__()
        self.coupling = LocalSurrogateCoupling(hidden_dim=8, local_surrogate_latent_dim=8,
                                               dropout=0.0, use_layer_norm=False)
        checkpoint = self.coupling.attach_from_checkpoint(checkpoint_path, freeze=True)
        if int(checkpoint.get("epoch", -1)) != 4975:
            raise ValueError("Expected the specified epoch-4975 local checkpoint.")
        for parameter in self.coupling.parameters():
            parameter.requires_grad_(False)
        self.coupling.eval()

    def forward(self, structure, heat, ports, local_query_points):
        present = structure["module_present"]
        params = build_local_module_params_from_global(
            heat, None, structure["material_params"], present)
        params = update_local_module_params_from_ports(params, ports, present)
        return self.coupling.call_local_surrogate(params, ports, local_query_points, present)

    def train(self, mode=True):
        super().train(mode)
        self.coupling.eval()
        return self

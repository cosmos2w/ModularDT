"""Focused contracts for Run-1503-v4 phase-local probe catalogues."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from channelthermal.interface_field_coupling import (
    _outside_coordinates,
    _outside_coordinates_from_normals,
    _phase_functional_probe_kwargs,
    _port_coordinates,
)


class _FixedPortHead:
    def __init__(self) -> None:
        self.calls = 0

    def fixed_theta_tokens(
        self, ntheta: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        self.calls += 1
        theta = torch.arange(ntheta, device=device, dtype=dtype) * (2.0 * torch.pi / ntheta)
        return torch.stack([theta, torch.cos(theta), torch.sin(theta)], dim=-1)


def _probe_model(
    architecture: str = "continuous_functional_coalescence_honf", epoch: int = 100
) -> SimpleNamespace:
    port_head = _FixedPortHead()
    return SimpleNamespace(
        config=SimpleNamespace(
            core_honf=SimpleNamespace(
                forward_architecture=architecture,
                module_radius=0.5,
                domain_length_x=12.0,
                domain_length_y=4.0,
            ),
            channelthermal=SimpleNamespace(port_global_consistency_radius_offset=0.1),
        ),
        core=SimpleNamespace(
            backend=SimpleNamespace(
                selection_state=lambda: {"epoch": epoch, "total_epochs": 500},
            ),
        ),
        local_coupling=SimpleNamespace(port_head=port_head),
    )


def _tokens_from_angles(
    angles: torch.Tensor, temperature: float, h_effective: float
) -> torch.Tensor:
    tokens = torch.stack([angles, torch.cos(angles), torch.sin(angles)], dim=-1)
    tail = tokens.new_empty(*tokens.shape[:-1], 2)
    tail[..., 0] = temperature
    tail[..., 1] = h_effective
    return torch.cat([tokens, tail], dim=-1)


def _families(kwargs: dict[str, object]):
    catalogue = kwargs["functional_probes"]
    return catalogue, catalogue.physical_ports, catalogue.outside_temperature


def test_phase_catalogue_uses_only_current_geometry_and_masks_padding() -> None:
    model = _probe_model()
    centers = torch.tensor(
        [
            [[3.0, 1.0], [8.0, 2.5], [0.0, 0.0]],
            [[4.0, 1.2], [7.0, 2.2], [9.0, 3.0]],
        ]
    )
    module_present = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    batch, modules, ntheta = 2, 3, 4
    geometry_ports = _port_coordinates(model, centers, ntheta)
    fixed = model.local_coupling.port_head.fixed_theta_tokens(ntheta, centers.device, centers.dtype)
    base_angles = fixed[:, 0].view(1, 1, ntheta).expand(batch, modules, -1)
    initial_tokens = _tokens_from_angles(base_angles + 0.17, temperature=21.0, h_effective=4.0)
    refined_tokens = _tokens_from_angles(base_angles + 0.31, temperature=87.0, h_effective=9.0)

    p0 = _phase_functional_probe_kwargs(
        model,
        "P0",
        physical_port_xy=geometry_ports,
        current_port_tokens=None,
        module_centers=centers,
        module_present=module_present,
        ntheta=ntheta,
    )
    # Even if a future token tensor is accidentally available to the helper,
    # P0 remains a pure geometry catalogue built before port prediction.
    p0_with_future = _phase_functional_probe_kwargs(
        model,
        "P0",
        physical_port_xy=geometry_ports,
        current_port_tokens=refined_tokens,
        module_centers=centers,
        module_present=module_present,
        ntheta=ntheta,
    )
    p1 = _phase_functional_probe_kwargs(
        model,
        "P1",
        physical_port_xy=geometry_ports,
        current_port_tokens=initial_tokens,
        module_centers=centers,
        module_present=module_present,
        ntheta=ntheta,
    )
    p1_changed_values = _phase_functional_probe_kwargs(
        model,
        "P1",
        physical_port_xy=geometry_ports,
        current_port_tokens=_tokens_from_angles(
            base_angles + 0.17, temperature=-900.0, h_effective=1200.0
        ),
        module_centers=centers,
        module_present=module_present,
        ntheta=ntheta,
    )
    p2 = _phase_functional_probe_kwargs(
        model,
        "P2",
        physical_port_xy=geometry_ports,
        current_port_tokens=refined_tokens,
        module_centers=centers,
        module_present=module_present,
        ntheta=ntheta,
    )

    c0, ports0, outside0 = _families(p0)
    _, ports0_future, outside0_future = _families(p0_with_future)
    c1, ports1, outside1 = _families(p1)
    _, ports1_changed, outside1_changed = _families(p1_changed_values)
    c2, ports2, outside2 = _families(p2)

    assert [c0.phase, c1.phase, c2.phase] == ["P0", "P1", "P2"]
    assert c0.environment is c1.environment is c2.environment is None
    assert ports0.coordinates.shape == outside0.coordinates.shape == (batch, modules * ntheta, 2)
    assert ports0.valid_mask.dtype == torch.bool
    assert ports0.valid_mask.shape == ports0.weights.shape == (batch, modules * ntheta)
    assert torch.equal(ports0.coordinates, ports0_future.coordinates)
    assert torch.equal(outside0.coordinates, outside0_future.coordinates)
    assert torch.equal(ports0.valid_mask, ports0_future.valid_mask)

    # P0 locations use the fixed physical angle and fixed normal, with no
    # predicted temperature/h values. P1/P2 use the geometry available at
    # their own phase and likewise exclude thermal values from the catalogue.
    torch.testing.assert_close(
        ports0.coordinates.reshape(batch, modules, ntheta, 2), geometry_ports
    )
    expected_p0_outside = _outside_coordinates_from_normals(
        model, fixed[:, 1:3], centers
    )
    torch.testing.assert_close(
        outside0.coordinates.reshape(batch, modules, ntheta, 2), expected_p0_outside
    )
    torch.testing.assert_close(
        ports1.coordinates.reshape(batch, modules, ntheta, 2), geometry_ports
    )
    torch.testing.assert_close(
        outside1.coordinates.reshape(batch, modules, ntheta, 2),
        _outside_coordinates(model, initial_tokens, centers),
    )
    torch.testing.assert_close(
        ports2.coordinates.reshape(batch, modules, ntheta, 2), geometry_ports
    )
    torch.testing.assert_close(
        outside2.coordinates.reshape(batch, modules, ntheta, 2),
        _outside_coordinates(model, refined_tokens, centers),
    )
    assert torch.equal(ports1.coordinates, ports1_changed.coordinates)
    assert torch.equal(outside1.coordinates, outside1_changed.coordinates)
    assert torch.equal(ports1.coordinates, ports2.coordinates)
    assert not torch.equal(outside1.coordinates, outside2.coordinates)

    for family in (ports0, outside0, ports1, outside1, ports2, outside2):
        assert family.coordinates.shape[-1] == 2
        assert torch.isfinite(family.coordinates).all()
        assert torch.isfinite(family.weights).all()
        assert (family.weights[~family.valid_mask] == 0).all()
        torch.testing.assert_close(
            family.weights.sum(dim=-1),
            torch.ones(batch, dtype=family.weights.dtype),
        )
    # Sample 0 has only one active module; its padded module probes are masked.
    assert not ports0.valid_mask[0, ntheta:].any()
    assert not outside0.valid_mask[0, ntheta:].any()


def test_historical_architectures_do_not_receive_functional_probe_kwargs() -> None:
    model = _probe_model("converged_identity_preserving_coalescence_honf")
    empty = _phase_functional_probe_kwargs(
        model,
        "P0",
        physical_port_xy=torch.zeros(1, 1, 4, 2),
        current_port_tokens=None,
        module_centers=torch.zeros(1, 1, 2),
        module_present=torch.ones(1, 1),
        ntheta=4,
    )
    assert empty == {}


def test_parent_schedule_skips_probe_construction_through_epoch_fifty() -> None:
    model = _probe_model(epoch=50)
    empty = _phase_functional_probe_kwargs(
        model,
        "P0",
        physical_port_xy=torch.zeros(1, 1, 4, 2),
        current_port_tokens=None,
        module_centers=torch.zeros(1, 1, 2),
        module_present=torch.ones(1, 1),
        ntheta=4,
    )
    assert empty == {}
    assert model.local_coupling.port_head.calls == 0

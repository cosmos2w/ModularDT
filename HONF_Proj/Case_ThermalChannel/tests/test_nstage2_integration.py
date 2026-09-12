"""Adapter/factory integration for the two independent NStage2 operators."""

import pytest
import torch

from channelthermal.config import ChannelThermalHONFConfig
from channelthermal.environment import ChannelThermalEnvironmentBuilder
from channelthermal.model import ChannelThermalHONFModel


def test_thermal_hierarchy_keeps_fine_environment_and_partial_cell_bounds():
    env = ChannelThermalEnvironmentBuilder()(
        batch_size=2, num_env_tokens_x=24, num_env_tokens_y=8,
        domain_length_x=12.0, domain_length_y=4.0,
        device=torch.device("cpu"), dtype=torch.float32,
        response_region_block_shape=(2, 2), response_tree_block_shape=(2, 2),
    )
    assert env.env_coords.shape == (2, 192, 2)
    assert env.env_features.shape == (2, 192, 7)
    tree = env.env_hierarchy
    assert tree.level_sizes == (192, 48, 12, 3, 2, 1)
    assert tree.num_nodes == 258
    assert env.env_region_ids.unique().numel() == 48
    # The last level-4 node covers the actual final four columns: no padded cells.
    level4 = tree.level_slice(4)
    torch.testing.assert_close(tree.bounds_min[level4][-1], torch.tensor([8.0, 0.0]))
    torch.testing.assert_close(tree.bounds_max[level4][-1], torch.tensor([12.0, 4.0]))


@pytest.mark.parametrize("architecture", ["hierarchical_regional_honf", "sparse_interface_honf"])
def test_candidate_wrapper_reads_ports_and_fields_from_shared_preparation(architecture):
    group_source = architecture == "sparse_interface_honf"
    options = {
        "message_hidden_dim": 24, "attention_heads": 4,
        "coarse_latent_count": 8, "coarse_blocks": 1,
        "relative_fourier_frequencies": 2, "receiver_chunk_size": 3,
        "activation_checkpointing": False,
    }
    if group_source:
        options.update(support_spacing_factor=4.0, group_read_mode="geometry_envelope_attention",
                       coarse_module_source="group_states")
    config = ChannelThermalHONFConfig.from_dict({
        "core_honf": {
            "forward_architecture": architecture, "interface_model": options,
            "hidden_dim": 32, "field_dim": 5, "domain_length_x": 4.0,
            "domain_length_y": 2.0, "coordinate_scale": [4.0, 2.0],
            "module_radius": 0.2, "num_env_tokens_x": 4, "num_env_tokens_y": 4,
            "dropout": 0.0, "boundary_feature_mode": "none",
            "position_fourier_frequencies": 2, "query_fourier_frequencies": 2,
        },
        "channelthermal": {"use_local_surrogate": False, "internal_prediction_mode": "global_head",
                           "default_num_interface_points": 8},
    })
    model = ChannelThermalHONFModel(config, attach_local_from_checkpoint=False).eval()
    coarse_environment_widths = []
    hook = model.core.common.coarse_env_attention.register_forward_pre_hook(
        lambda module, args: coarse_environment_widths.append(args[1].shape[1])
    )
    try:
        output = model(
            query_xy=torch.tensor([[[0.25, 0.25], [1.5, 0.75], [3.5, 1.75]]]),
            re=torch.tensor([[100.0]]), u_in=torch.tensor([[1.0]]),
            module_centers=torch.tensor([[[1.0, 0.75], [3.0, 1.25]]]),
            heat_powers=torch.tensor([[1.0, 2.0]]), module_present=torch.ones(1, 2),
            material_params=torch.tensor([[0.01, 0.02, 0.03, 1.0, 0.5, 0.2]]),
            return_prepared_state=True, return_routing_maps=True,
        )
    finally:
        hook.remove()
    assert torch.isfinite(output["pred_field"]).all()
    assert torch.isfinite(output["pred_port_condition"]).all()
    prepared = output["prepared_state"].prepared
    assert prepared.encoded.env_tokens.shape[1] == 16
    assert prepared.coarse_state.shape == (1, 8, 32)
    assert coarse_environment_widths and set(coarse_environment_widths) == {16}
    if group_source:
        assert hasattr(model.core.common, "coarse_group_attention")
        assert not hasattr(model.core.common, "coarse_module_attention")
        assert prepared.encoded.env_hierarchy is None
    else:
        assert hasattr(model.core.common, "coarse_module_attention")
        assert prepared.encoded.env_hierarchy.num_nodes == 21
        query = torch.tensor([[[0.1, 0.2], [0.7, 0.6], [1.2, 1.1], [2.5, 0.4], [3.6, 1.8]]])
        full = model.core.read(prepared, query, receiver_chunk_size=5, return_routing_maps=True)
        chunked = model.core.read(prepared, query, receiver_chunk_size=2, return_routing_maps=True)
        torch.testing.assert_close(full.context, chunked.context, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            full.interaction_aux["hierarchical_selected_count"],
            chunked.interaction_aux["hierarchical_selected_count"],
        )
        # Ragged maps must retain every incidence and use global query IDs
        # across chunks, even when incidence order differs by traversal level.
        def ordered_rows(aux):
            key = aux["hierarchical_incidence_query"] * 21 + aux["hierarchical_incidence_node"]
            order = key.argsort()
            return key[order], aux["hierarchical_incidence_eta"][order], aux["hierarchical_incidence_attention"][order]

        for expected, observed in zip(ordered_rows(full.interaction_aux), ordered_rows(chunked.interaction_aux)):
            torch.testing.assert_close(expected, observed, rtol=1e-5, atol=1e-6)
        assert chunked.interaction_aux["hierarchical_incidence_rows"] == chunked.interaction_aux["hierarchical_selected_count"].sum()
    assert getattr(model.core, "_interface_read_role", None) is None

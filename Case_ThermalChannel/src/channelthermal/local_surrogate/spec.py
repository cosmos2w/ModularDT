"""Stable lifecycle contract for the ThermalChannel disk surrogate."""

from honf_runtime.case_protocol import LocalModuleSpec


THERMAL_DISK_SPEC = LocalModuleSpec(
    module_id="thermal_disk",
    schema_version=1,
    module_parameter_names=(
        "q_internal",
        "solid_k",
        "solid_alpha",
        "h_mean",
        "h_std",
        "T_env_mean",
        "T_env_std",
    ),
    port_feature_names=("theta", "cos_theta", "sin_theta", "T_env", "h"),
    query_coordinate_names=("x_local", "y_local"),
    target_names=("internal_temperature", "T_surface", "q_normal"),
    latent_dim=128,
    dataset_ids=("thermal_disk_local_v1", "thermal_channel_global_v1"),
    model_factory="channelthermal.local_surrogate.model:LocalModuleSurrogate",
    checkpoint_loader="honf_runtime.compat:load_trusted_checkpoint",
    train_workflow="channelthermal.workflows.train_local:run_from_config",
    evaluate_workflow="channelthermal.workflows.evaluate_local:main",
    coupling_adapter="channelthermal.local_coupling:LocalSurrogateCoupling",
    frozen_in_parent=True,
    embedded_in_parent_checkpoint=True,
)

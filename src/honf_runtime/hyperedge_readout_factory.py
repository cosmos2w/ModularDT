"""Native Round6 task construction and standalone, provider-free inference loading."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .hyperedge_readout_child import copy_round6_common_state
from .hyperedge_readout_protocol import (
    READOUT_CORE_SCHEMA,
    READOUT_FAMILY,
    READOUT_MAX_FIT_AGE,
    READOUT_SAMPLER_OFFSET,
    READOUT_WORKFLOW,
    canonical_sha256,
    file_sha256,
    readout_attachment_declaration,
    validate_hyperedge_readout_recipe,
    verify_joint_gate_binding,
)
from .moment_hyperedge_protocol import (
    MOMENT_CORE_SCHEMA,
)
from .registry import require_model_family
from .unified_training import TrainingEngine


def configure_runtime_root(runtime_root: str | Path) -> Path:
    from .moment_hyperedge_factory import configure_runtime_root as configure_native

    return configure_native(runtime_root)


def _core_and_model(model: nn.Module, dataset: str) -> tuple[nn.Module, str]:
    if dataset == "thermal":
        core = getattr(model, "core", None)
        if not isinstance(core, nn.Module):
            raise TypeError("Thermal Round6 model must own its IAG-C core under `core`.")
        return core, "core."
    if dataset == "wind":
        return model, ""
    raise ValueError("Round6 supports only Thermal or Wind.")


def _round5_config_from_readout(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema") != READOUT_CORE_SCHEMA:
        raise ValueError("Round6 provider core does not carry the registered readout schema.")
    normalized = copy.deepcopy(dict(config))
    for name in ("receiver_reduction", "receiver_access", "geometry_mode", "receiver_readout_init_seed"):
        normalized.pop(name, None)
    normalized["schema"] = MOMENT_CORE_SCHEMA
    return normalized


def validate_readout_provider_pair(
    parent_identity: Mapping[str, Any], child_identity: Mapping[str, Any], *,
    dataset: str, core_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Allow only the enumerated readout schema/config addition to native identity."""
    if not isinstance(parent_identity, Mapping) or not isinstance(child_identity, Mapping):
        raise TypeError("Round6 native provider identities must be mappings.")
    expected_core = dict(core_config)
    expected_round5 = _round5_config_from_readout(expected_core)
    parent = copy.deepcopy(dict(parent_identity))
    child = copy.deepcopy(dict(child_identity))
    if dataset == "thermal":
        paths = (("model_config", "core"), ("adapter_state", "core"))
    elif dataset == "wind":
        paths = (("model_config",),)
    else:
        raise ValueError("Round6 native provider binding is restricted to Thermal or Wind.")
    for path in paths:
        old_cursor: Any = parent
        new_cursor: Any = child
        for key in path[:-1]:
            old_cursor = old_cursor.get(key) if isinstance(old_cursor, Mapping) else None
            new_cursor = new_cursor.get(key) if isinstance(new_cursor, Mapping) else None
        leaf = path[-1]
        old_config = old_cursor.get(leaf) if isinstance(old_cursor, Mapping) else None
        new_config = new_cursor.get(leaf) if isinstance(new_cursor, Mapping) else None
        if not isinstance(old_config, Mapping) or not isinstance(new_config, Mapping):
            raise TypeError(f"Round6 provider core identity at {path!r} must be a mapping.")
        if dict(new_config) != expected_core or dict(old_config) != expected_round5:
            raise ValueError(f"Round6 provider core at {path!r} differs from the exact H/R6 core configs.")
        new_cursor[leaf] = expected_round5
    if canonical_sha256(parent) != canonical_sha256(child):
        raise ValueError("Round6 native provider changed data, normalization, sampling, losses or transforms.")
    return {
        "schema": "hyperedge_readout_native_provider_binding_v1",
        "dataset": dataset,
        "source_provider_identity_sha256": canonical_sha256(parent),
        "readout_provider_identity_sha256": canonical_sha256(dict(child_identity)),
        "normalization": "only the Round6 receiver-readout core schema and declared reducer/access/geometry fields",
    }


def _attach_core(
    model: nn.Module, provider: Any, *, dataset: str,
    core_config: Mapping[str, Any], device: torch.device | str,
) -> dict[str, Any]:
    core, _prefix = _core_and_model(model, dataset)
    if core.get_config().get("schema") != MOMENT_CORE_SCHEMA:
        raise ValueError("Round6 can attach only to the existing scene-prepared H core.")
    if core.get_config().get("moment_variant") != "H":
        raise ValueError("Round6 can attach only to the H hyperedge state.")
    if dict(core_config) != {
        **{key: value for key, value in dict(core.get_config()).items() if key != "schema"},
        "schema": READOUT_CORE_SCHEMA,
        "receiver_reduction": core_config.get("receiver_reduction"),
        "receiver_access": core_config.get("receiver_access"),
        "geometry_mode": core_config.get("geometry_mode"),
        "receiver_readout_init_seed": core_config.get("receiver_readout_init_seed", 0),
    }:
        raise ValueError("Round6 core config changed the source H core outside its reader attachment.")
    from honf_forward_core.inherited_attention.readout_model import HyperedgeReadoutCoreV1

    new_core = HyperedgeReadoutCoreV1.from_config(dict(core_config)).to(device)
    source_core_state = core.state_dict()
    attached_core_state = copy_round6_common_state(
        new_core, source_core_state, dataset="wind",
    )
    if dataset == "thermal":
        model.core = new_core
        provider.model = model
    else:
        provider.model = new_core
        model = new_core
    if provider.model is not model:
        raise ValueError("Round6 native provider did not retain the exact attached model object.")
    return attached_core_state


def _trainable_allowlist(model: nn.Module, dataset: str) -> tuple[str, ...]:
    root = "core." if dataset == "thermal" else ""
    allowed = []
    for name, _parameter in model.named_parameters():
        if (name.startswith((root + "receiver_readout.", root + "field_head."))
                or (dataset == "thermal" and name.startswith(root + "response_phi."))):
            allowed.append(name)
    return tuple(allowed)


def _capture_frozen_state(model: nn.Module, allowed: set[str]) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name not in allowed
    }


def verify_frozen_state(
    model: nn.Module, frozen_state: Mapping[str, torch.Tensor], allowed_names: tuple[str, ...],
) -> dict[str, Any]:
    """Prove that all non-allowlisted parameters and every buffer stayed exact."""
    parameters = dict(model.named_parameters())
    state = model.state_dict()
    allowed = set(allowed_names)
    if {name for name, value in parameters.items() if value.requires_grad} != allowed:
        raise ValueError("Round6 frozen-fit requires_grad state differs from the exact allowlist.")
    if set(state) - allowed != set(frozen_state):
        raise ValueError("Round6 frozen-state snapshot inventory changed during fitting.")
    current_frozen_state = {
        name: state[name].detach().cpu() for name in frozen_state
    }
    changed = [name for name, before in frozen_state.items()
               if _tensor_state_digest(current_frozen_state[name]) != _tensor_state_digest(before)]
    frozen_digest = _state_digest(frozen_state)
    current_digest = _state_digest(current_frozen_state)
    if changed or current_digest != frozen_digest:
        raise ValueError(
            "Round6 frozen tensors changed during fit: "
            f"{changed[:12]}; captured_digest={frozen_digest}; current_digest={current_digest}."
        )
    return {
        "schema": "round6_frozen_state_verification_v1",
        "frozen_tensor_count": len(frozen_state),
        "frozen_state_sha256": frozen_digest,
        "current_frozen_state_sha256": current_digest,
        "trainable_parameter_names": list(allowed_names),
        "all_frozen_parameters_and_buffers_bitwise_fixed": True,
    }


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(_tensor_state_payload(state[name]))
    return digest.hexdigest()


def _tensor_state_payload(value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    return (
        str(tensor.dtype).encode()
        + str(tuple(tensor.shape)).encode()
        + tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    )


def _tensor_state_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_state_payload(value)).hexdigest()


class _FrozenReadoutProvider:
    """Delegate native data/loss behavior while narrowing trainable groups."""

    def __init__(
        self, base: Any, identity: Mapping[str, Any], specs: tuple[Any, ...],
        allowed: tuple[str, ...], frozen_state: Mapping[str, torch.Tensor],
    ):
        self._base = base
        self._identity = copy.deepcopy(dict(identity))
        self._specs = specs
        self._allowed = tuple(allowed)
        self._frozen_state = {name: value.detach().cpu().clone() for name, value in frozen_state.items()}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def identity_payload(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._identity)

    def optimizer_groups(self, model: nn.Module, arm: str, stage: str):
        if arm != "IAG-C" or stage != "joint":
            raise ValueError("Frozen Round6 fit uses its native joint IAG-C objective and arm.")
        if len(self._specs) != 1:
            raise ValueError("Round6 expects the one native IAG-C AdamW group.")
        from .unified_training import OptimizerGroupSpec

        source = self._specs[0]
        if set(self._allowed) - set(source.parameter_names):
            raise ValueError("Round6 trainable allowlist contains parameters outside the native optimizer group.")
        return (OptimizerGroupSpec(
            "hyperedge_readout_frozen_adamw",
            self._allowed,
            source.schedule,
            weight_decay=source.weight_decay,
            betas=source.betas,
            eps=source.eps,
        ),)

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch=0, temperature=None):
        # The frozen diagnostic must keep H's epoch-normalization buffer fixed.
        return self._base.predict_native(
            model, scene, receivers, execution_mode, phase,
            epoch=0, temperature=temperature,
        )

    def verify_post_epoch_state(self, model: nn.Module, fit_epoch: int) -> dict[str, Any]:
        result = verify_frozen_state(model, self._frozen_state, self._allowed)
        result["fit_epoch"] = int(fit_epoch)
        return result


class HyperedgeReadoutTrainingEngine(TrainingEngine):
    """Use fit-age history while preserving the sealed source sampler clock."""

    def __init__(self, config: Any, *, device: torch.device | str, selection: Any, recipe: Mapping[str, Any]):
        super().__init__(config, device=device, selection=selection)
        self.recipe = validate_hyperedge_readout_recipe(recipe)

    def sampler_epoch(self, fit_epoch: int) -> int:
        value = self.recipe
        if type(fit_epoch) is not int or fit_epoch < 1:
            raise ValueError("Round6 fit age must be a positive integer before sampler mapping.")
        if value["phase"] == "frozen_probe":
            if fit_epoch > value["fit_age_ceiling"]:
                raise ValueError("Round6 frozen fit age exceeds its sealed 1000-epoch ceiling.")
            return READOUT_SAMPLER_OFFSET + fit_epoch
        if not value["source_epoch"] < fit_epoch <= value["fit_age_ceiling"]:
            raise ValueError("Round6 joint phase must retain the absolute H-age sampler clock.")
        return fit_epoch

    def sampler_epoch_contract(self) -> Mapping[str, Any]:
        return copy.deepcopy(self.recipe["sampler_epoch_mapping"])


def make_hyperedge_readout_engine(recipe: Mapping[str, Any], *, device: torch.device | str):
    """Build the original native EngineConfig with only the Round6 phase horizon."""
    value = validate_hyperedge_readout_recipe(recipe)
    if value["phase"] == "joint_confirmation":
        verify_joint_gate_binding(value["joint_gate_binding"], dataset=value["dataset"])
    from inherited_attention_train import engine_for

    base = engine_for(value["native_recipe"], str(device))
    config = base.config
    if value["phase"] == "frozen_probe":
        epochs = int(value["fit_age_ceiling"])
        config = replace(
            base.config,
            total_epochs=epochs,
            monitor_epochs=tuple(epoch for epoch in base.config.monitor_epochs if epoch <= epochs),
            checkpoint_epochs=(
                None if base.config.checkpoint_epochs is None else
                tuple(epoch for epoch in base.config.checkpoint_epochs if epoch <= epochs)
            ),
        )
    return HyperedgeReadoutTrainingEngine(
        config, device=device, selection=base.selection, recipe=value,
    )


def _parent_binding(source_path: Path, source_payload: Mapping[str, Any], recipe: Mapping[str, Any],
                    provider_binding: Mapping[str, Any], attachment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "hyperedge_readout_parent_binding_v1",
        "phase": recipe["phase"],
        "path": str(source_path.resolve()),
        "sha256": file_sha256(source_path),
        "epoch": int(source_payload["epoch"]),
        "identity": copy.deepcopy(dict(source_payload["experiment_identity"])),
        "recipe_sha256": recipe["source_recipe_sha256"],
        "source_provider_identity_sha256": recipe["source_provider_identity_sha256"],
        "native_provider_binding": copy.deepcopy(dict(provider_binding)),
        "common_state_attachment": copy.deepcopy(dict(attachment)),
    }


def build_hyperedge_readout_task(
    recipe: Mapping[str, Any], *, device: torch.device | str = "cpu",
    runtime_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the exact native provider, attach H state and prepare the fit policy."""
    value = validate_hyperedge_readout_recipe(recipe)
    path = Path(value["source_checkpoint"]).resolve(strict=True)
    if file_sha256(path) != value["source_checkpoint_sha256"]:
        raise ValueError("Round6 H e-parent checkpoint changed after recipe creation.")
    if runtime_root is None:
        runtime_root = Path(__file__).resolve().parents[2]
    project = configure_runtime_root(runtime_root)
    from .compat import load_trusted_checkpoint
    from .hyperedge_readout_child import _check_source

    source_payload = load_trusted_checkpoint(path, map_location="cpu")
    _source_identity, source_provider_identity, _history = _check_source(source_payload, value, path)
    from inherited_attention_train import build

    model, native_provider = build(value["native_recipe"], device=str(device))
    observed_source_provider = dict(native_provider.identity_payload())
    if canonical_sha256(observed_source_provider) != canonical_sha256(source_provider_identity):
        raise ValueError("Native task/data provider identity differs from the exact H checkpoint provider.")
    source_state = source_payload.get("model_state_dict")
    if not isinstance(source_state, Mapping):
        raise TypeError("Round6 H source has no state dictionary.")
    model.load_state_dict(source_state, strict=True)
    attachment = _attach_core(
        model, native_provider,
        dataset=value["dataset"],
        core_config=value["core_config"],
        device=device,
    )
    # Wind's native provider owns the active core as `provider.model`; Thermal
    # owns it beneath its adapter. Use the provider's active object in both
    # cases after swapping the strict R6 core.
    model = native_provider.model
    child_provider_identity = dict(native_provider.identity_payload())
    provider_binding = validate_readout_provider_pair(
        source_provider_identity,
        child_provider_identity,
        dataset=value["dataset"],
        core_config=value["core_config"],
    )
    parent_binding = _parent_binding(path, source_payload, value, provider_binding, attachment)
    reader_decl = readout_attachment_declaration(value)
    reader_decl.update({
        "common_state_sha256": attachment["common_state_sha256"],
        "born_parameter_names": attachment["born_parameter_names"],
        "born_parameter_sha256": attachment["born_parameter_sha256"],
        "new_optimizer_state_count_at_birth": 0,
    })
    identity = {
        "family": READOUT_FAMILY,
        "model_family": READOUT_FAMILY,
        "workflow": READOUT_WORKFLOW,
        "protocol": value["protocol"],
        "phase": value["phase"],
        "receiver_reduction": value["receiver_reduction"],
        "receiver_access": value["receiver_access"],
        "recipe": value,
        "recipe_sha256": canonical_sha256(value),
        "branch_binding": parent_binding,
        "attachment": reader_decl,
        "sampler_epoch_mapping": copy.deepcopy(value["sampler_epoch_mapping"]),
        "native_provider_binding": provider_binding,
        "loaded_source": _loaded_source_binding(project),
    }
    if value["phase"] == "frozen_probe":
        allowed_names = _trainable_allowlist(model, value["dataset"])
        if not allowed_names:
            raise ValueError("Round6 frozen fit found no field/readout/response parameters in its exact allowlist.")
        # Native optimizer adapters require their original all-parameter model
        # to be wholly trainable when declaring the source group contract.
        full_specs = tuple(native_provider.optimizer_groups(model, "IAG-C", "joint"))
        parameters = dict(model.named_parameters())
        for name, parameter in parameters.items():
            parameter.requires_grad_(name in set(allowed_names))
        if {name for name, parameter in parameters.items() if parameter.requires_grad} != set(allowed_names):
            raise ValueError("Round6 frozen fit could not install its exact trainable-parameter allowlist.")
        frozen_state = _capture_frozen_state(model, set(allowed_names))
        training_provider = _FrozenReadoutProvider(
            native_provider, child_provider_identity, full_specs, allowed_names, frozen_state,
        )
        if value["dataset"] == "thermal":
            # Existing Thermal response sampling is scheduled through epoch 2500;
            # Round6 extends only the accepted sampler key range, not its recipe.
            native_provider._round6_sampling_epoch_limit = READOUT_SAMPLER_OFFSET + READOUT_MAX_FIT_AGE
        optimizer_seed = None
    else:
        allowed_names = tuple(dict(model.named_parameters()))
        frozen_state = {}
        training_provider = native_provider
        optimizer_seed = source_payload.get("optimizer_state_by_name")
    identity["fit_scope"] = {
        "schema": "hyperedge_readout_fit_scope_v1",
        "phase": value["phase"],
        "trainable_parameter_names": list(allowed_names),
        "native_provider_all_parameters_train_from_start_is_source_metadata": True,
        "actual_training_engine_parameter_scope_is_authoritative": True,
        "optimizer_state": "fresh_adamw_state" if value["phase"] == "frozen_probe"
                           else "inherit_common_h_named_state_only",
        "buffers_bitwise_fixed": value["phase"] == "frozen_probe",
    }
    return {
        "model": model,
        "native_provider": native_provider,
        "training_provider": training_provider,
        "provider_identity": child_provider_identity,
        "native_provider_binding": provider_binding,
        "branch_binding": parent_binding,
        "attachment": reader_decl,
        "identity": identity,
        "fit_scope": identity["fit_scope"],
        "trainable_parameter_names": allowed_names,
        "frozen_state": frozen_state,
        "optimizer_seed": optimizer_seed,
        "source_payload": source_payload,
        "source_path": path,
    }


def _loaded_source_binding(project: Path) -> dict[str, Any]:
    """Hash the fixed R6 runtime source inventory, independent of import order."""
    files: dict[str, str] = {}
    roots = ("src", "Case_ThermalChannel/src", "Case_WindFarm/src")
    for relative_root in roots:
        root = project / relative_root
        if not root.is_dir():
            raise FileNotFoundError(f"Round6 runtime source root is missing: {root}")
        for path in sorted(root.rglob("*.py")):
            if path.is_file():
                relative = path.relative_to(project).as_posix()
                files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    for relative in ("tools/inherited_attention_train.py", "tools/hyperedge_readout_train.py"):
        path = project / relative
        if not path.is_file():
            raise FileNotFoundError(f"Round6 required runtime tool is missing: {path}")
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    ordered = dict(sorted(files.items()))
    return {
        "schema": "hyperedge_readout_loaded_source_v1",
        "inventory": "project src, Thermal src, Wind src, inherited_attention_train.py, hyperedge_readout_train.py",
        "files": ordered,
        "sha256": canonical_sha256(ordered),
    }


def _validate_readout_checkpoint(payload: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    identity = payload.get("experiment_identity")
    if (not isinstance(identity, Mapping)
            or payload.get("workflow") != READOUT_WORKFLOW
            or identity.get("family") != READOUT_FAMILY
            or identity.get("model_family") != READOUT_FAMILY
            or identity.get("workflow") != READOUT_WORKFLOW):
        raise ValueError("Checkpoint is not a sealed Round6 hyperedge-readout run.")
    require_model_family(READOUT_FAMILY, "hyperedge_readout_evaluate")
    recipe = validate_hyperedge_readout_recipe(identity.get("recipe", {}))
    if (identity.get("recipe_sha256") != canonical_sha256(recipe)
            or identity.get("phase") != recipe["phase"]
            or identity.get("receiver_reduction") != recipe["receiver_reduction"]
            or identity.get("receiver_access") != recipe["receiver_access"]):
        raise ValueError("Round6 checkpoint identity does not match its strict recipe.")
    epoch = payload.get("epoch")
    minimum_epoch = 1 if recipe["phase"] == "frozen_probe" else recipe["source_epoch"] + 1
    if type(epoch) is not int or not minimum_epoch <= epoch <= recipe["fit_age_ceiling"]:
        raise ValueError("Round6 checkpoint age is outside the sealed phase fit ceiling.")
    if payload.get("arm") != "IAG-C":
        raise ValueError("Round6 checkpoint must retain the native IAG-C arm.")
    branch = identity.get("branch_binding")
    if (not isinstance(branch, Mapping)
            or branch.get("schema") != "hyperedge_readout_parent_binding_v1"
            or branch.get("phase") != recipe["phase"]
            or branch.get("path") != recipe["source_checkpoint"]
            or branch.get("sha256") != recipe["source_checkpoint_sha256"]
            or branch.get("epoch") != recipe["source_epoch"]
            or canonical_sha256(branch.get("identity", {})) == canonical_sha256({})):
        raise ValueError("Round6 checkpoint lost its exact source H parent binding.")
    provider_identity = identity.get("provider_identity")
    parent_provider = branch.get("identity", {}).get("provider_identity")
    if not isinstance(provider_identity, Mapping) or not isinstance(parent_provider, Mapping):
        raise TypeError("Round6 checkpoint native provider/adapter identities must be mappings.")
    validate_readout_provider_pair(
        parent_provider, provider_identity,
        dataset=recipe["dataset"], core_config=recipe["core_config"],
    )
    if identity.get("sampler_epoch_mapping") != recipe["sampler_epoch_mapping"]:
        raise ValueError("Round6 checkpoint lost its sealed absolute sampler-age mapping.")
    return recipe, provider_identity, branch


def load_hyperedge_readout_components(
    checkpoint: str | Path, *, runtime_root: str | Path,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load a Round6 model and native transforms without a provider or dataset."""
    configure_runtime_root(runtime_root)
    from .compat import load_trusted_checkpoint

    path = Path(checkpoint).expanduser().resolve(strict=True)
    payload = load_trusted_checkpoint(path, map_location="cpu")
    recipe, provider_identity, branch = _validate_readout_checkpoint(payload)
    from .inherited_attention_protocol import configure_inherited_attention_numerics

    configure_inherited_attention_numerics(recipe["native_recipe"]["numerical_policy"])
    adapter_state = provider_identity.get("adapter_state")
    if not isinstance(adapter_state, Mapping):
        raise TypeError("Round6 provider identity lacks its native transform payload.")
    if recipe["dataset"] == "thermal":
        from channelthermal.content_hypergraph import ThermalInheritedAttentionAdapterV1

        if (provider_identity.get("model_config", {}).get("core") != recipe["core_config"]
                or adapter_state.get("core") != recipe["core_config"]):
            raise ValueError("Thermal Round6 model/adapter configs differ from the sealed recipe.")
        model = ThermalInheritedAttentionAdapterV1.from_state_payload(adapter_state, device=device)
        adapter = model
    elif recipe["dataset"] == "wind":
        from windfarm.content_hypergraph import WindInheritedAttentionAdapterV1

        from honf_forward_core.inherited_attention.readout_model import HyperedgeReadoutCoreV1

        if dict(provider_identity.get("model_config", {})) != recipe["core_config"]:
            raise ValueError("Wind Round6 core config differs from the sealed recipe.")
        model = HyperedgeReadoutCoreV1.from_config(recipe["core_config"]).to(device)
        adapter = WindInheritedAttentionAdapterV1.from_state_payload(adapter_state, device=device)
    else:
        raise ValueError("Round6 checkpoint dataset is unsupported.")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise TypeError("Round6 checkpoint has no tensor state dictionary.")
    model.load_state_dict(state, strict=True)
    model.eval()
    return {
        "model": model,
        "native_adapter": adapter,
        "recipe": recipe,
        "branch_binding": branch,
        "provider_identity": provider_identity,
        "checkpoint": {
            "path": str(path),
            "sha256": file_sha256(path),
            "epoch": int(payload["epoch"]),
            "phase": recipe["phase"],
        },
    }


__all__ = [
    "HyperedgeReadoutTrainingEngine",
    "build_hyperedge_readout_task",
    "configure_runtime_root",
    "load_hyperedge_readout_components",
    "make_hyperedge_readout_engine",
    "validate_readout_provider_pair",
    "verify_frozen_state",
]

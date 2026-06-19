"""CHANNELTHERMAL-SPECIFIC local checkpoint compatibility test.

Inputs are an existing Stage-A local surrogate checkpoint path. Outputs are a
strict state-dict load check plus deterministic finite-output assertions for
internal temperature, interface prediction, and module response latent. This
script is ChannelThermal-specific because it validates the copied Stage-A local
surrogate checkpoint schema.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch

SRC_NEW_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_NEW_ROOT))

import _bootstrap_imports  # noqa: F401
from _helpers.model_utils import load_trusted_checkpoint, resolve_demo_path, strip_module_prefix
from _models_local.model_local import LocalModuleConfig, LocalModuleSurrogate


DEFAULT_CHECKPOINT = "./Saved_Model_LocalModule/Run_0003_20260507_224352/latest_model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict-load an old Stage-A local checkpoint with the copied model.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise AssertionError(f"{name} contains non-finite values.")


def load_copied_model(checkpoint: Dict[str, Any], device: torch.device) -> LocalModuleSurrogate:
    config = LocalModuleConfig.from_dict(checkpoint.get("model_config", {}))
    model = LocalModuleSurrogate(config).to(device)
    model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]), strict=True)
    model.eval()
    return model


def main() -> int:
    args = parse_args()
    path = resolve_demo_path(args.checkpoint)
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    device = torch.device(args.device)
    checkpoint = load_trusted_checkpoint(path, map_location=device)
    model = load_copied_model(checkpoint, device)
    cfg = model.config
    torch.manual_seed(1234)
    module_params = torch.randn(2, int(cfg.module_param_dim), device=device)
    theta = torch.linspace(0.0, 2.0 * torch.pi, 16 + 1, device=device)[:-1]
    base_ports = torch.stack([theta, torch.cos(theta), torch.sin(theta)], dim=-1)
    physical = torch.randn(2, theta.numel(), max(int(cfg.port_token_dim) - 3, 0), device=device)
    port_tokens = torch.cat([base_ports.unsqueeze(0).expand(2, -1, -1), physical], dim=-1)
    port_tokens = port_tokens[..., : int(cfg.port_token_dim)]
    internal_query = torch.randn(2, 11, 2, device=device).clamp(-1.0, 1.0)
    with torch.no_grad():
        out = model(module_params, port_tokens, internal_query)
    finite_tensor("internal_temperature", out["internal_temperature"])
    finite_tensor("interface_pred", out["interface_pred"])
    finite_tensor("module_response_latent", out["module_response_latent"])
    print(
        "[ok] strict local checkpoint compatibility: "
        f"checkpoint={path}, internal={tuple(out['internal_temperature'].shape)}, "
        f"interface={tuple(out['interface_pred'].shape)}, latent={tuple(out['module_response_latent'].shape)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

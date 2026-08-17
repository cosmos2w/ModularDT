"""Minimal non-thermal case used to enforce the public extension boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.model import HONFNeuralField
from honf_runtime.case_protocol import WorkflowRequest


class DummyCasePlugin:
    """Small installed-case analogue with no dependency on ThermalChannel."""

    case_id = "SyntheticSecondCase"
    display_name = "Synthetic extension fixture"
    version = "1.0.0"

    def __init__(self) -> None:
        config = UnifiedForwardConfig(
            field_dim=1,
            domain_length_x=1.0,
            domain_length_y=1.0,
            num_env_tokens_x=2,
            num_env_tokens_y=2,
            num_hyperedges=2,
            hidden_dim=16,
            dropout=0.0,
            decoder_mode="hyper_only",
            boundary_feature_mode="none",
        )
        self.model = HONFNeuralField(config)

    @staticmethod
    def adapt() -> BatchData:
        """Adapt a made-up physical sample to the generic core contract."""

        return BatchData(
            module_centers=torch.tensor([[[0.25, 0.5], [0.75, 0.5]]]),
            module_present=torch.ones(1, 2),
            module_features=torch.tensor([[[1.0, 0.1], [2.0, 0.2]]]),
            global_context=torch.tensor([[0.5]]),
            query_xy=torch.tensor([[[0.1, 0.2], [0.8, 0.7]]]),
            query_time=None,
            target_field=torch.zeros(1, 2, 1),
            case_name="synthetic_second_case",
            metadata={"schema_version": 1},
        )

    def inspect_launch(self, bundle: Any, request: WorkflowRequest) -> Mapping[str, Any]:
        return {"dataset ID": "synthetic_memory_v1", "dataset cases": 1}

    def validate_config(self, bundle: Any) -> None:
        return None

    def train(self, bundle: Any, request: WorkflowRequest, *, run_dir: Path) -> int:
        batch = self.adapt()
        prediction = self.model(batch)["pred_field"]
        loss = torch.mean((prediction - batch.target_field) ** 2)
        (run_dir / "dummy_metrics.json").write_text(
            json.dumps({"loss": float(loss.detach())}) + "\n", encoding="utf-8"
        )
        return 0

    def evaluate(self, bundle: Any, request: WorkflowRequest) -> int:
        prediction = self.model(self.adapt())["pred_field"]
        return 0 if torch.isfinite(prediction).all() else 1


def create_plugin() -> DummyCasePlugin:
    return DummyCasePlugin()

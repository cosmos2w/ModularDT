from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

import channelthermal.evaluation.prepared as prepared_module


class _ChunkAccountingModel:
    config = SimpleNamespace(field_dim=1)

    @staticmethod
    def _ledger(query_count: int) -> dict[str, torch.Tensor]:
        return {
            "group_control_module_fine_rows_forward": torch.tensor(float(query_count)),
            "group_control_module_fine_rows_padded": torch.tensor(float(query_count)),
            "group_control_environment_geometry_rows_forward": torch.tensor(float(2 * query_count)),
            "group_control_environment_content_dot_rows_forward": torch.tensor(float(4 * query_count)),
        }

    def __call__(self, _structure, query_xy, **_kwargs):
        query_count = int(query_xy.shape[1])
        return {
            "pred_field": query_xy.new_zeros((1, query_count, 1)),
            "pred_internal_temperature": query_xy.new_zeros((1, 1, 1)),
            "pred_interface": query_xy.new_zeros((1, 1, 1)),
            "pred_port_condition": query_xy.new_zeros((1, 1, 1)),
            "organizer_aux": {},
            "base_organizer_aux": {},
            "interaction_aux": self._ledger(query_count),
            "prepared_state": object(),
        }

    def decode_prepared(self, _prepared, query_xy, **_kwargs):
        query_count = int(query_xy.shape[1])
        return {
            "pred_field": query_xy.new_zeros((1, query_count, 1)),
            **self._ledger(query_count),
        }


def test_predict_case_aggregates_group_control_ledgers_over_outer_query_chunks(monkeypatch) -> None:
    monkeypatch.setattr(
        prepared_module,
        "make_batch",
        lambda sample, chunk, device: {
            "structure": sample.get("structure", {}),
            "query_xy": torch.from_numpy(chunk).unsqueeze(0).to(device),
        },
    )
    sample = {
        "x_grid": np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        "y_grid": np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        "structure": {},
    }
    result = prepared_module.predict_case(
        _ChunkAccountingModel(),
        sample,
        torch.device("cpu"),
        query_batch_size=3,
        local_port_condition_mode="predicted",
        mixed_teacher_ratio=0.0,
        return_routing_maps=True,
    )
    aux = result["interaction_aux"]
    assert float(np.asarray(aux["group_control_module_fine_rows_forward"])) == 4.0
    assert float(np.asarray(aux["group_control_module_fine_rows_padded"])) == 4.0
    assert float(np.asarray(aux["group_control_environment_geometry_rows_forward"])) == 8.0
    assert float(np.asarray(aux["group_control_environment_content_dot_rows_forward"])) == 16.0
    assert result["pred_field_grid"].shape == (2, 2, 1)


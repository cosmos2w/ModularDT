"""Same-input grid refinement metrics on persisted native fields."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .io import write_json_atomic


def _cells(fine: np.ndarray, ratio: int) -> np.ndarray:
    n = fine.shape[0] // ratio
    return fine.reshape(n, ratio, n, ratio).mean(axis=(1, 3))


def _face_x(fine: np.ndarray, ratio: int) -> np.ndarray:
    n = fine.shape[0] // ratio
    return fine[:, ::ratio].reshape(n, ratio, n + 1).mean(axis=1)


def _face_y(fine: np.ndarray, ratio: int) -> np.ndarray:
    n = fine.shape[1] // ratio
    return fine[::ratio, :].reshape(n + 1, n, ratio).mean(axis=2)


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / denominator) if denominator > 1e-12 else float("nan")


def compare_resolution(root: Path, state_id: str, coarse: int, fine: int) -> dict:
    if fine % coarse != 0:
        raise ValueError("fine grid must be integer multiple of coarse grid")
    ratio = fine // coarse
    with h5py.File(Path(root) / "raw" / f"{state_id}_n{coarse}.h5") as lo, \
         h5py.File(Path(root) / "raw" / f"{state_id}_n{fine}.h5") as hi:
        if lo.attrs["input_sha256"] != hi.attrs["input_sha256"]:
            raise RuntimeError("refinement inputs differ")
        metrics = {}
        for key in ("potential", "temperature_rise", "q_sheet", "q_contact"):
            metrics[f"{key}_relative_l2"] = _rel_l2(lo[key][:], _cells(hi[key][:], ratio))
        for key in ("jx_face", "heat_flux_x_face"):
            metrics[f"{key}_relative_l2"] = _rel_l2(lo[key][:], _face_x(hi[key][:], ratio))
        for key in ("jy_face", "heat_flux_y_face"):
            metrics[f"{key}_relative_l2"] = _rel_l2(lo[key][:], _face_y(hi[key][:], ratio))
        qi = lo["q_sheet"][:] + lo["q_contact"][:]
        qf = _cells(hi["q_sheet"][:] + hi["q_contact"][:], ratio)
        metrics["q_total_relative_l2"] = _rel_l2(qi, qf)
        ci, cf = lo["terminal_current"][:], hi["terminal_current"][:]
        metrics["terminal_current_relative_l2"] = _rel_l2(ci, cf)
        ti, tf = lo["temperature_rise"][:], hi["temperature_rise"][:]
        metrics["temperature_peak_relative"] = abs(float(ti.max()) - float(tf.max())) / max(abs(float(tf.max())), 1e-12)
        top_i = float(np.mean(ti[ti >= np.quantile(ti, 0.95)]))
        top_f = float(np.mean(tf[tf >= np.quantile(tf, 0.95)]))
        metrics["temperature_top5_mean_relative"] = abs(top_i - top_f) / max(abs(top_f), 1e-12)
        return {"state_id": state_id, "coarse": coarse, "fine": fine,
                "physical_input_sha256": lo.attrs["input_sha256"], **metrics}


def refinement_board(root: Path, ids: list[str], grids: tuple[int, ...] = (48, 96, 192)) -> dict:
    comparisons = [compare_resolution(root, sid, a, b) for sid in ids
                   for a, b in zip(grids[:-1], grids[1:])]
    summary = {"comparisons": comparisons,
               "max_by_pair": {f"{a}_vs_{b}": {key: max(row[key] for row in comparisons
                                                         if row["coarse"] == a and row["fine"] == b)
                                               for key in comparisons[0] if key.endswith("_relative_l2") or key.endswith("_relative")}
                               for a, b in zip(grids[:-1], grids[1:])}}
    write_json_atomic(Path(root) / "diagnostics" / "refinement.json", summary)
    return summary
